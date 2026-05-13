from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from .config import ModelConfig, WeightMapping, LLAMA_WEIGHT_MAP


def quantize_weight_to_ternary(
    weight: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    flat = weight.reshape(weight.shape[0], -1).astype(np.float32)
    abs_sum = np.abs(flat).sum(axis=1, keepdims=True)
    count = flat.shape[1]
    mean_abs = abs_sum / count
    threshold = alpha * mean_abs
    ternary = np.where(flat > threshold, 1, np.where(flat < -threshold, -1, 0)).astype(np.int8)
    nonzero_mask = ternary != 0
    scale = np.ones(weight.shape[0], dtype=np.float32)
    for i in range(weight.shape[0]):
        nz = flat[i][nonzero_mask[i]]
        if len(nz) > 0:
            scale[i] = np.mean(np.abs(nz))
    return ternary, scale


def pack_ternary_rows(ternary: np.ndarray, n: int) -> np.ndarray:
    m, n_check = ternary.shape
    assert n_check == n, f"Shape mismatch: {n_check} != {n}"
    assert n % 16 == 0, f"N must be multiple of 16, got {n}"
    packed_cols = n // 16
    packed = np.zeros(m * packed_cols, dtype=np.uint32)
    for row in range(m):
        for pc in range(packed_cols):
            word = np.uint32(0)
            for bit in range(16):
                val = ternary[row, pc * 16 + bit]
                if val == 0:
                    bits = np.uint32(0b00)
                elif val == 1:
                    bits = np.uint32(0b01)
                elif val == -1:
                    bits = np.uint32(0b10)
                else:
                    bits = np.uint32(0b00)
                word |= bits << np.uint32(bit * 2)
            packed[row * packed_cols + pc] = word
    return packed


def try_native_pack(ternary: np.ndarray, n: int) -> Optional[np.ndarray]:
    try:
        from ternary_zero import _core
        flat = ternary.flatten().astype(np.int8)
        result = _core.pack_ternary_to_u32_py(flat, n)
        return result.astype(np.uint32)
    except (ImportError, AttributeError):
        return None


@np.errstate(all='ignore')
def quantize_and_pack(
    weight: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    ternary, per_row_scale = quantize_weight_to_ternary(weight, alpha)
    n = ternary.shape[1]
    packed = try_native_pack(ternary, n)
    if packed is None:
        packed = pack_ternary_rows(ternary, n)
    global_scale = float(np.mean(per_row_scale))
    return packed, per_row_scale, global_scale


class QuantizedLayer:
    __slots__ = ("packed_weights", "per_row_scale", "global_scale", "m", "n", "bias")

    def __init__(
        self,
        packed_weights: np.ndarray,
        per_row_scale: np.ndarray,
        global_scale: float,
        m: int,
        n: int,
        bias: Optional[np.ndarray] = None,
    ):
        self.packed_weights = packed_weights
        self.per_row_scale = per_row_scale
        self.global_scale = global_scale
        self.m = m
        self.n = n
        self.bias = bias

    @classmethod
    def from_weight(
        cls,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        alpha: float = 0.5,
    ) -> "QuantizedLayer":
        m, n = weight.shape
        packed, per_row_scale, global_scale = quantize_and_pack(weight, alpha)
        return cls(packed, per_row_scale, global_scale, m, n, bias)


class QuantizedModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.embed_tokens: Optional[np.ndarray] = None
        self.lm_head: Optional[QuantizedLayer] = None
        self.norm_weight: Optional[np.ndarray] = None
        self.layers: List[Dict[str, object]] = []

    def quantize_from_state_dict(
        self,
        state_dict: Dict[str, np.ndarray],
        alpha: float = 0.5,
        weight_map: Optional[WeightMapping] = None,
        embed_fp16: bool = True,
        lm_head_quantize: bool = True,
        verbose: bool = True,
    ) -> None:
        wm = weight_map or LLAMA_WEIGHT_MAP
        cfg = self.config

        embed_w = state_dict.get(wm.embed_tokens)
        if embed_w is not None:
            self.embed_tokens = embed_w.astype(np.float32) if embed_fp16 else embed_w
            if verbose:
                mb = self.embed_tokens.nbytes / (1024 * 1024)
                print(f"  Embedding: {embed_w.shape} -> FP32 ({mb:.1f} MB)")

        lm_w = state_dict.get(wm.lm_head)
        if lm_w is not None:
            if lm_head_quantize:
                self.lm_head = QuantizedLayer.from_weight(lm_w.astype(np.float32), alpha=alpha)
                if verbose:
                    packed_mb = self.lm_head.packed_weights.nbytes / (1024 * 1024)
                    print(f"  LM Head:   {lm_w.shape} -> Ternary ({packed_mb:.1f} MB)")
            else:
                self.lm_head = None
                self._lm_head_fp32 = lm_w.astype(np.float32)

        norm_w = state_dict.get(wm.final_norm)
        if norm_w is not None:
            self.norm_weight = norm_w.astype(np.float32)
            if verbose:
                print(f"  Final Norm: {norm_w.shape}")

        for layer_idx in range(cfg.num_layers):
            prefix = f"{wm.layer_prefix}.{layer_idx}"
            layer_data = {}

            input_norm_w = state_dict.get(f"{prefix}.{wm.input_norm}")
            if input_norm_w is not None:
                layer_data["input_norm"] = input_norm_w.astype(np.float32)

            post_norm_w = state_dict.get(f"{prefix}.{wm.post_attn_norm}")
            if post_norm_w is not None:
                layer_data["post_attn_norm"] = post_norm_w.astype(np.float32)

            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                w_key = f"{prefix}.{getattr(wm, proj_name)}"
                b_key = w_key.replace(".weight", ".bias")
                w = state_dict.get(w_key)
                if w is not None:
                    w32 = w.astype(np.float32)
                    bias = state_dict.get(b_key)
                    if bias is not None:
                        bias = bias.astype(np.float32)
                    ql = QuantizedLayer.from_weight(w32, bias=bias, alpha=alpha)
                    layer_data[proj_name] = ql
                    if verbose:
                        packed_mb = ql.packed_weights.nbytes / (1024 * 1024)
                        print(f"  Layer {layer_idx:>2} {proj_name:>10}: {w.shape} -> Ternary ({packed_mb:.1f} MB)")

            self.layers.append(layer_data)

    def get_lm_head_weight(self) -> Optional[np.ndarray]:
        if hasattr(self, '_lm_head_fp32'):
            return self._lm_head_fp32
        return None


def load_safetensors(path: str) -> Dict[str, np.ndarray]:
    try:
        from safetensors import safe_open
        tensors = {}
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors
    except ImportError:
        pass

    try:
        import torch
        from pathlib import Path
        if Path(path).suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(path)
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        return {k: v.cpu().numpy() for k, v in state.items()}
    except ImportError:
        raise ImportError(
            "Install 'safetensors' or 'torch' to load model weights: "
            "pip install safetensors torch"
        )


def load_model_weights(
    model_path: str,
    config: ModelConfig,
    alpha: float = 0.5,
    weight_map: Optional[WeightMapping] = None,
    embed_fp16: bool = True,
    lm_head_quantize: bool = True,
    verbose: bool = True,
) -> QuantizedModel:
    from pathlib import Path
    p = Path(model_path)

    if verbose:
        print(f"Loading weights from: {p}")

    weight_files = sorted(p.glob("*.safetensors"))
    if not weight_files:
        weight_files = sorted(p.glob("*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No weight files (.safetensors or .bin) found at {p}")

    if verbose:
        print(f"  Found {len(weight_files)} weight file(s)")

    state_dict = {}
    for wf in weight_files:
        if verbose:
            print(f"  Loading {wf.name}...")
        state_dict.update(load_safetensors(str(wf)))

    if verbose:
        total_bytes = sum(v.nbytes for v in state_dict.values())
        print(f"  Total FP weights: {total_bytes / (1024**3):.2f} GB")
        print()
        print("Quantizing to ternary...")

    qm = QuantizedModel(config)
    qm.quantize_from_state_dict(
        state_dict, alpha=alpha, weight_map=weight_map,
        embed_fp16=embed_fp16, lm_head_quantize=lm_head_quantize,
        verbose=verbose,
    )

    del state_dict

    if verbose:
        total_packed = sum(
            ql.packed_weights.nbytes
            for layer in qm.layers
            for name, ql in layer.items()
            if isinstance(ql, QuantizedLayer)
        )
        if qm.lm_head is not None:
            total_packed += qm.lm_head.packed_weights.nbytes
        embed_bytes = qm.embed_tokens.nbytes if qm.embed_tokens is not None else 0
        total = total_packed + embed_bytes
        print()
        print(f"  Quantized weight memory: {total / (1024**2):.1f} MB")
        print(f"  Compression vs FP32:     {config.weight_bytes_fp32() / total:.1f}x")

    return qm
