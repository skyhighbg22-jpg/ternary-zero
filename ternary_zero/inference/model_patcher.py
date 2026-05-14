from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .config import ModelConfig, WeightMapping, LLAMA_WEIGHT_MAP, PRESET_CONFIGS, detect_config


# =====================================================================
# Ternary Quantization Primitives (Chunked / Streaming)
# =====================================================================

def _ternary_quantize_chunk(
    weight_row: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, float]:
    flat = weight_row.astype(np.float32).ravel()
    abs_sum = np.abs(flat).sum()
    count = flat.shape[0]
    mean_abs = abs_sum / count
    threshold = alpha * mean_abs
    ternary = np.where(flat > threshold, 1, np.where(flat < -threshold, -1, 0)).astype(np.int8)
    nonzero_mask = ternary != 0
    nz = flat[nonzero_mask]
    scale = float(np.mean(np.abs(nz))) if len(nz) > 0 else 1.0
    return ternary, scale


def _pack_ternary_rows_chunked(
    ternary_2d: np.ndarray,
    n: int,
    chunk_rows: int = 256,
) -> np.ndarray:
    m = ternary_2d.shape[0]
    assert n % 16 == 0, f"N must be multiple of 16, got {n}"
    packed_cols = n // 16
    packed = np.empty(m * packed_cols, dtype=np.uint32)

    try:
        from ternary_zero import _core
        use_native = True
    except ImportError:
        use_native = False

    for start in range(0, m, chunk_rows):
        end = min(start + chunk_rows, m)
        chunk = ternary_2d[start:end]
        if use_native:
            flat = chunk.flatten().astype(np.int8)
            result = _core.pack_ternary_to_u32_py(flat, n)
            packed[start * packed_cols : end * packed_cols] = result
        else:
            rows = end - start
            for row in range(rows):
                for pc in range(packed_cols):
                    word = np.uint32(0)
                    for bit in range(16):
                        val = chunk[row, pc * 16 + bit]
                        if val == 0:
                            bits = np.uint32(0b00)
                        elif val == 1:
                            bits = np.uint32(0b01)
                        elif val == -1:
                            bits = np.uint32(0b10)
                        else:
                            bits = np.uint32(0b00)
                        word |= bits << np.uint32(bit * 2)
                    packed[(start + row) * packed_cols + pc] = word

    return packed


# =====================================================================
# Safetensors Streaming Reader
# =====================================================================

class SafetensorsReader:
    def __init__(self, path: str):
        self.path = path
        self._header: Optional[Dict[str, Any]] = None
        self._file = None

    def _read_header(self) -> Dict[str, Any]:
        if self._header is not None:
            return self._header
        with open(self.path, "rb") as f:
            header_size_bytes = f.read(8)
            header_size = int.from_bytes(header_size_bytes, "little")
            header_json = f.read(header_size)
            self._header = json.loads(header_json)
        return self._header

    def keys(self) -> List[str]:
        header = self._read_header()
        return [k for k in header.keys() if k != "__metadata__"]

    def tensor_names(self) -> List[str]:
        return self.keys()

    def tensor_info(self, name: str) -> Tuple[List[int], str]:
        header = self._read_header()
        info = header[name]
        dtype_map = {
            "F32": "float32", "F16": "float16", "BF16": "bfloat16",
            "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
            "U8": "uint8", "BOOL": "bool",
        }
        dtype_str = dtype_map.get(info["dtype"], "float32")
        return info["shape"], dtype_str

    def read_tensor(self, name: str) -> np.ndarray:
        header = self._read_header()
        info = header[name]
        dtype_map = {
            "F32": np.float32, "F16": np.float16,
            "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
            "U8": np.uint8, "BOOL": np.bool_,
        }
        np_dtype = dtype_map.get(info["dtype"], np.float32)
        begin, end = info["data_offsets"]
        with open(self.path, "rb") as f:
            header_size_bytes = f.read(8)
            header_size = int.from_bytes(header_size_bytes, "little")
            f.seek(8 + header_size + begin)
            raw = f.read(end - begin)
        return np.frombuffer(raw, dtype=np_dtype).reshape(info["shape"]).copy()

    def stream_tensors(self) -> Iterator[Tuple[str, np.ndarray]]:
        for name in self.keys():
            yield name, self.read_tensor(name)
            gc.collect()

    def close(self):
        self._header = None


# =====================================================================
# Streaming Tensor Iterator (Multi-File)
# =====================================================================

def stream_safetensors_files(directory: str) -> Iterator[Tuple[str, np.ndarray]]:
    p = Path(directory)
    weight_files = sorted(p.glob("*.safetensors"))
    if not weight_files:
        weight_files = sorted(p.glob("*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No weight files found at {p}")

    for wf in weight_files:
        if wf.suffix == ".safetensors":
            reader = SafetensorsReader(str(wf))
            for name, tensor in reader.stream_tensors():
                yield name, tensor
            reader.close()
        else:
            import torch
            state = torch.load(str(wf), map_location="cpu", weights_only=True)
            for name, tensor in state.items():
                yield name, tensor.cpu().numpy()
            del state
        gc.collect()


# =====================================================================
# Layer Statistics
# =====================================================================

@dataclass
class LayerQuantStats:
    name: str
    original_shape: List[int]
    original_dtype: str
    original_bytes: int
    packed_bytes: int
    compression_ratio: float
    global_scale: float
    sparsity: float
    quantize_time_ms: float


@dataclass
class PatchManifest:
    model_name: str
    model_config: Dict[str, Any]
    output_dir: str
    total_original_bytes: int = 0
    total_packed_bytes: int = 0
    total_quantize_time_ms: float = 0.0
    layers: List[LayerQuantStats] = field(default_factory=list)
    embed_bytes: int = 0
    norm_bytes: int = 0

    @property
    def total_compression(self) -> float:
        if self.total_packed_bytes == 0:
            return 0.0
        return self.total_original_bytes / self.total_packed_bytes

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_config": self.model_config,
            "output_dir": self.output_dir,
            "total_original_bytes": self.total_original_bytes,
            "total_packed_bytes": self.total_packed_bytes,
            "total_compression_ratio": self.total_compression,
            "total_quantize_time_ms": self.total_quantize_time_ms,
            "embed_bytes": self.embed_bytes,
            "norm_bytes": self.norm_bytes,
            "layers": [
                {
                    "name": l.name,
                    "original_shape": l.original_shape,
                    "original_dtype": l.original_dtype,
                    "original_bytes": l.original_bytes,
                    "packed_bytes": l.packed_bytes,
                    "compression_ratio": l.compression_ratio,
                    "global_scale": l.global_scale,
                    "sparsity": l.sparsity,
                    "quantize_time_ms": l.quantize_time_ms,
                }
                for l in self.layers
            ],
        }


# =====================================================================
# Quantized Tensor Storage
# =====================================================================

def _save_packed_tensor(
    output_dir: str,
    name: str,
    packed: np.ndarray,
    per_row_scale: np.ndarray,
    global_scale: float,
    m: int,
    n: int,
    bias: Optional[np.ndarray] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    safe_name = name.replace(".", "_")
    out_path = os.path.join(output_dir, f"{safe_name}.npz")

    save_dict = {
        "packed_weights": packed,
        "per_row_scale": per_row_scale,
        "global_scale": np.float32(global_scale),
        "m": np.int32(m),
        "n": np.int32(n),
    }
    if bias is not None:
        save_dict["bias"] = bias

    np.savez_compressed(out_path, **save_dict)


# =====================================================================
# Model Patcher: Streaming Quantization Engine
# =====================================================================

class ModelPatcher:
    def __init__(
        self,
        alpha: float = 0.5,
        chunk_rows: int = 256,
        embed_fp16: bool = True,
        lm_head_quantize: bool = True,
        verbose: bool = True,
    ):
        self.alpha = alpha
        self.chunk_rows = chunk_rows
        self.embed_fp16 = embed_fp16
        self.lm_head_quantize = lm_head_quantize
        self.verbose = verbose

    def _is_quantizable(self, tensor_name: str, weight_map: WeightMapping) -> bool:
        wm = weight_map
        quantizable_suffixes = [
            wm.q_proj, wm.k_proj, wm.v_proj, wm.o_proj,
            wm.gate_proj, wm.up_proj, wm.down_proj,
        ]
        if tensor_name == wm.lm_head:
            return True
        for suffix in quantizable_suffixes:
            if tensor_name.endswith(suffix):
                return True
        return False

    def _is_norm(self, tensor_name: str, weight_map: WeightMapping) -> bool:
        wm = weight_map
        return (
            tensor_name.endswith(wm.input_norm)
            or tensor_name.endswith(wm.post_attn_norm)
            or tensor_name == wm.final_norm
        )

    def _is_embedding(self, tensor_name: str, weight_map: WeightMapping) -> bool:
        return tensor_name == weight_map.embed_tokens

    def _quantize_tensor(
        self,
        name: str,
        weight: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if weight.ndim == 1:
            weight = weight.reshape(1, -1)
        m, n = weight.shape
        if n % 16 != 0:
            pad = 16 - (n % 16)
            weight = np.pad(weight, ((0, 0), (0, pad)), mode="constant")
            n = weight.shape[1]

        ternary_all = np.empty((m, n), dtype=np.int8)
        scales = np.empty(m, dtype=np.float32)

        for start in range(0, m, self.chunk_rows):
            end = min(start + self.chunk_rows, m)
            for row_idx in range(start, end):
                ternary_row, scale = _ternary_quantize_chunk(
                    weight[row_idx], self.alpha
                )
                ternary_all[row_idx] = ternary_row
                scales[row_idx] = scale

        packed = _pack_ternary_rows_chunked(ternary_all, n, self.chunk_rows)
        global_scale = float(np.mean(scales))

        return packed, scales, global_scale

    def patch_model(
        self,
        model_path: str,
        output_dir: str,
        config: Optional[ModelConfig] = None,
        weight_map: Optional[WeightMapping] = None,
    ) -> PatchManifest:
        if config is None:
            if model_path.lower() in PRESET_CONFIGS:
                config = PRESET_CONFIGS[model_path.lower()]
            else:
                config = detect_config(model_path)

        wm = weight_map or LLAMA_WEIGHT_MAP
        os.makedirs(output_dir, exist_ok=True)

        manifest = PatchManifest(
            model_name=config.name,
            model_config={
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "num_layers": config.num_layers,
                "vocab_size": config.vocab_size,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
            },
            output_dir=output_dir,
        )

        if self.verbose:
            print(f"Model Patcher: Streaming {config.name}")
            print(f"  Source: {model_path}")
            print(f"  Output: {output_dir}")
            print(f"  Alpha:  {self.alpha}")
            print(f"  Chunk:  {self.chunk_rows} rows")
            print()

        t_total_start = time.perf_counter()
        quantizable_scales: Dict[str, Tuple[np.ndarray, float, int, int]] = {}

        for name, tensor in stream_safetensors_files(model_path):
            original_bytes = tensor.nbytes
            manifest.total_original_bytes += original_bytes

            if self._is_embedding(name, wm):
                if self.embed_fp16:
                    embed_fp32 = tensor.astype(np.float32)
                    out_path = os.path.join(output_dir, "embed_tokens.npz")
                    np.savez_compressed(out_path, weight=embed_fp32)
                    manifest.embed_bytes = embed_fp32.nbytes
                    if self.verbose:
                        print(f"  {name}: {tensor.shape} -> FP32 ({embed_fp32.nbytes / 1e6:.1f} MB)")
                del tensor
                gc.collect()
                continue

            if self._is_norm(name, wm):
                norm_fp32 = tensor.astype(np.float32)
                safe_name = name.replace(".", "_")
                out_path = os.path.join(output_dir, f"{safe_name}.npz")
                np.savez_compressed(out_path, weight=norm_fp32)
                manifest.norm_bytes += norm_fp32.nbytes
                if self.verbose:
                    print(f"  {name}: {tensor.shape} -> FP32 norm ({norm_fp32.nbytes / 1024:.1f} KB)")
                del tensor, norm_fp32
                gc.collect()
                continue

            if self._is_quantizable(name, wm):
                t0 = time.perf_counter()
                weight_f32 = tensor.astype(np.float32)
                del tensor

                packed, per_row_scale, global_scale = self._quantize_tensor(
                    name, weight_f32
                )
                m, n = weight_f32.shape
                del weight_f32

                bias = None
                _save_packed_tensor(
                    output_dir, name, packed, per_row_scale,
                    global_scale, m, n, bias,
                )

                packed_bytes = packed.nbytes + per_row_scale.nbytes
                sparsity = float(np.mean(per_row_scale == 0.0)) if per_row_scale.size > 0 else 0.0
                t1 = time.perf_counter()

                stats = LayerQuantStats(
                    name=name,
                    original_shape=[m, n],
                    original_dtype=str(tensor.dtype) if hasattr(tensor, 'dtype') else "float32",
                    original_bytes=original_bytes,
                    packed_bytes=packed_bytes,
                    compression_ratio=original_bytes / packed_bytes if packed_bytes > 0 else 0.0,
                    global_scale=global_scale,
                    sparsity=sparsity,
                    quantize_time_ms=(t1 - t0) * 1000,
                )
                manifest.layers.append(stats)
                manifest.total_packed_bytes += packed_bytes
                manifest.total_quantize_time_ms += stats.quantize_time_ms

                quantizable_scales[name] = (per_row_scale, global_scale, m, n)

                if self.verbose:
                    print(
                        f"  {name}: [{m}, {n}] -> packed "
                        f"({packed_bytes / 1e6:.2f} MB, "
                        f"{stats.compression_ratio:.1f}x, "
                        f"{stats.quantize_time_ms:.0f}ms)"
                    )

                del packed, per_row_scale
                gc.collect()
                continue

            safe_name = name.replace(".", "_")
            out_path = os.path.join(output_dir, f"{safe_name}.npz")
            np.savez_compressed(out_path, weight=tensor.astype(np.float32))
            if self.verbose:
                print(f"  {name}: {tensor.shape} -> FP32 passthrough ({original_bytes / 1024:.1f} KB)")
            del tensor
            gc.collect()

        t_total = time.perf_counter() - t_total_start

        manifest_path = os.path.join(output_dir, "patch_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        if self.verbose:
            print()
            print(f"  Patching complete in {t_total:.1f}s")
            print(f"  Original weights: {manifest.total_original_bytes / 1e9:.2f} GB")
            print(f"  Packed weights:   {manifest.total_packed_bytes / 1e6:.1f} MB")
            print(f"  Compression:      {manifest.total_compression:.1f}x")
            print(f"  Manifest:         {manifest_path}")

        return manifest


# =====================================================================
# CLI Entrypoint
# =====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ternary-Zero Model Patcher: Stream-quantize HuggingFace models to 2-bit ternary"
    )
    parser.add_argument("model_path", help="Path to HuggingFace model directory")
    parser.add_argument("output_dir", help="Output directory for quantized weights")
    parser.add_argument("--alpha", type=float, default=0.5, help="Quantization threshold alpha")
    parser.add_argument("--chunk-rows", type=int, default=256, help="Rows per quantization chunk")
    parser.add_argument("--embed-fp16", action="store_true", default=True, help="Keep embeddings in FP16")
    parser.add_argument("--lm-head-quantize", action="store_true", default=True, help="Quantize LM head")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    patcher = ModelPatcher(
        alpha=args.alpha,
        chunk_rows=args.chunk_rows,
        embed_fp16=args.embed_fp16,
        lm_head_quantize=args.lm_head_quantize,
        verbose=not args.quiet,
    )

    manifest = patcher.patch_model(args.model_path, args.output_dir)
    return manifest


if __name__ == "__main__":
    main()
