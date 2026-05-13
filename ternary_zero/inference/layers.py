from __future__ import annotations

import numpy as np
from typing import Optional

from .config import ModelConfig


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    variance = np.mean(x * x, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(variance + eps)
    return x_normed * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x_shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 500000.0,
) -> tuple:
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs_outer = np.outer(t, freqs)
    cos = np.cos(freqs_outer)
    sin = np.sin(freqs_outer)
    return cos, sin


def apply_rotary_emb(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return np.concatenate([out1, out2], axis=-1)


class TernaryLinear:
    __slots__ = ("packed_weights", "scale", "bias", "m", "n", "_use_gpu")

    def __init__(
        self,
        packed_weights: np.ndarray,
        scale: float,
        m: int,
        n: int,
        bias: Optional[np.ndarray] = None,
    ):
        self.packed_weights = packed_weights.astype(np.uint32)
        self.scale = scale
        self.m = m
        self.n = n
        self.bias = bias
        self._use_gpu: Optional[bool] = None

    @classmethod
    def from_quantized(cls, ql, force_cpu: bool = False) -> "TernaryLinear":
        layer = cls(
            packed_weights=ql.packed_weights,
            scale=ql.global_scale,
            m=ql.m,
            n=ql.n,
            bias=ql.bias,
        )
        if force_cpu:
            layer._use_gpu = False
        return layer

    def _detect_backend(self) -> str:
        if self._use_gpu is not None:
            return "cuda" if self._use_gpu else "cpu"
        try:
            from ternary_zero import _core
            if _core.has_cuda():
                self._use_gpu = True
                return "cuda"
            self._use_gpu = False
            return "cpu-rust"
        except ImportError:
            return "numpy"

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_flat = x.ravel().astype(np.float32)
        backend = self._detect_backend()
        scale = self.scale

        if backend == "cuda":
            from ternary_zero import _core
            raw = _core.ternary_gemv_gpu(
                self.packed_weights, x_flat, self.m, self.n
            )
        elif backend == "cpu-rust":
            from ternary_zero import _core
            raw = _core.ternary_gemv_cpu_packed(
                self.packed_weights, x_flat, self.m, self.n
            )
        else:
            from ..quantize import ternary_gemv_numpy
            raw = ternary_gemv_numpy(
                self.packed_weights, x_flat, self.m, self.n
            )

        output = (raw * scale).astype(np.float32)

        if self.bias is not None:
            output = output + self.bias

        return output


class Attention:
    def __init__(
        self,
        config: ModelConfig,
        q_proj: TernaryLinear,
        k_proj: TernaryLinear,
        v_proj: TernaryLinear,
        o_proj: TernaryLinear,
    ):
        self.config = config
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_queries_per_kv = config.num_queries_per_kv
        self.scale = 1.0 / np.sqrt(self.head_dim)

    def forward(
        self,
        x: np.ndarray,
        kv_cache,
        layer_idx: int,
        position: int,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
    ) -> np.ndarray:
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        q = q.reshape(self.num_heads, self.head_dim)
        k = k.reshape(self.num_kv_heads, self.head_dim)
        v = v.reshape(self.num_kv_heads, self.head_dim)

        for h in range(self.num_heads):
            q[h] = apply_rotary_emb(q[h], rope_cos[position], rope_sin[position])
        for h in range(self.num_kv_heads):
            k[h] = apply_rotary_emb(k[h], rope_cos[position], rope_sin[position])

        kv_cache.update(layer_idx, position, k, v)

        attn_output = np.zeros(self.num_heads * self.head_dim, dtype=np.float32)

        for h in range(self.num_heads):
            kv_h = h // self.num_queries_per_kv
            q_h = q[h]

            k_cache = kv_cache.get_k(layer_idx, kv_h, position + 1)
            v_cache = kv_cache.get_v(layer_idx, kv_h, position + 1)

            attn_weights = np.dot(k_cache, q_h) * self.scale
            attn_probs = softmax(attn_weights)
            attn_out = np.dot(attn_probs, v_cache)

            attn_output[h * self.head_dim : (h + 1) * self.head_dim] = attn_out

        return self.o_proj.forward(attn_output)


class FeedForward:
    def __init__(
        self,
        gate_proj: TernaryLinear,
        up_proj: TernaryLinear,
        down_proj: TernaryLinear,
    ):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = silu(self.gate_proj.forward(x))
        up = self.up_proj.forward(x)
        return self.down_proj.forward(gate * up)


class TransformerBlock:
    def __init__(
        self,
        attention: Attention,
        feed_forward: FeedForward,
        input_norm_weight: np.ndarray,
        post_attn_norm_weight: np.ndarray,
        config: ModelConfig,
    ):
        self.attention = attention
        self.feed_forward = feed_forward
        self.input_norm_weight = input_norm_weight
        self.post_attn_norm_weight = post_attn_norm_weight
        self.eps = config.rms_norm_eps

    def forward(
        self,
        x: np.ndarray,
        kv_cache,
        layer_idx: int,
        position: int,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
    ) -> np.ndarray:
        h = rms_norm(x, self.input_norm_weight, self.eps)
        attn_out = self.attention.forward(
            h, kv_cache, layer_idx, position, rope_cos, rope_sin
        )
        x = x + attn_out

        h2 = rms_norm(x, self.post_attn_norm_weight, self.eps)
        ffn_out = self.feed_forward.forward(h2)
        x = x + ffn_out

        return x


class Transformer:
    def __init__(
        self,
        config: ModelConfig,
        blocks: list,
        embed_tokens: np.ndarray,
        norm_weight: np.ndarray,
        lm_head: Optional[TernaryLinear] = None,
        lm_head_weight: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.blocks = blocks
        self.embed_tokens = embed_tokens
        self.norm_weight = norm_weight
        self.lm_head = lm_head
        self.lm_head_weight = lm_head_weight
        self.rope_cos, self.rope_sin = build_rope_cache(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

    def forward(
        self,
        token_id: int,
        kv_cache,
        position: int,
    ) -> np.ndarray:
        x = self.embed_tokens[token_id].astype(np.float32)

        for layer_idx, block in enumerate(self.blocks):
            x = block.forward(
                x, kv_cache, layer_idx, position,
                self.rope_cos, self.rope_sin,
            )

        x = rms_norm(x, self.norm_weight, self.config.rms_norm_eps)

        if self.lm_head is not None:
            logits = self.lm_head.forward(x)
        elif self.lm_head_weight is not None:
            logits = np.dot(self.lm_head_weight, x).astype(np.float32)
        else:
            logits = np.dot(self.embed_tokens.T, x).astype(np.float32)

        return logits
