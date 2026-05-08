from typing import Tuple, Union
import numpy as np

from .tensor import Tensor
from .autograd.functions import TernaryQuantizeSTE
from ._backend import has_torch, to_numpy

if has_torch():
    import torch


def ternary_quantize(
    weights: Tensor,
    alpha: float = 0.5,
) -> Tuple[Tensor, float]:
    result = TernaryQuantizeSTE.apply(weights, alpha)
    ternary = result[0] if isinstance(result, tuple) else result
    scale = result[1] if isinstance(result, tuple) else 1.0
    if isinstance(scale, Tensor):
        scale_data = scale.data
        if has_torch() and isinstance(scale_data, torch.Tensor):
            scale = float(scale_data.item())
        else:
            scale = float(scale_data)
    elif not isinstance(scale, (int, float)):
        if has_torch() and isinstance(scale, torch.Tensor):
            scale = float(scale.item())
        else:
            scale = float(scale)
    return ternary, scale


def ternary_quantize_fixed(
    weights: Tensor,
    threshold: float = 0.0,
) -> Tensor:
    flat = to_numpy(weights.data).flatten()
    result = np.where(flat > threshold, 1, np.where(flat < -threshold, -1, 0))
    return Tensor(result.reshape(weights.shape).astype(np.int8))


def dequantize_ternary(
    ternary_weights: Tensor,
    scale: float,
) -> Tensor:
    tw = ternary_weights.data
    if has_torch() and isinstance(tw, torch.Tensor):
        return Tensor(tw.float() * scale)
    return Tensor((tw.astype(np.float32) * scale))


def pack_ternary_to_u32(
    weights: Tensor,
    n: int,
) -> Tensor:
    w = to_numpy(weights.data).flatten().astype(np.int8)
    total = len(w)
    assert total % n == 0, f"weight length {total} must be multiple of N={n}"
    assert n % 16 == 0, f"N must be multiple of 16, got {n}"
    m = total // n
    packed_cols = n // 16
    packed = np.zeros(m * packed_cols, dtype=np.uint32)

    for row in range(m):
        for pc in range(packed_cols):
            word = np.uint32(0)
            for bit in range(16):
                idx = row * n + pc * 16 + bit
                val = w[idx]
                if val == 0:
                    bits = np.uint32(0b00)
                elif val == 1:
                    bits = np.uint32(0b01)
                elif val == -1:
                    bits = np.uint32(0b10)
                else:
                    raise ValueError(f"Invalid ternary value: {val}")
                word |= bits << np.uint32(bit * 2)
            packed[row * packed_cols + pc] = word

    return Tensor(packed)


def unpack_u32_to_ternary(
    packed: Tensor,
    n: int,
) -> Tensor:
    p = to_numpy(packed.data).flatten().astype(np.uint32)
    packed_cols = n // 16
    assert len(p) % packed_cols == 0
    m = len(p) // packed_cols
    weights = np.zeros(m * n, dtype=np.int8)

    for row in range(m):
        for pc in range(packed_cols):
            word = p[row * packed_cols + pc]
            for bit in range(16):
                bits = int((word >> np.uint32(bit * 2)) & np.uint32(0b11))
                if bits == 0b00:
                    val = 0
                elif bits == 0b01:
                    val = 1
                elif bits == 0b10:
                    val = -1
                else:
                    raise ValueError(f"Invalid 2-bit pattern: {bits:02b}")
                weights[row * n + pc * 16 + bit] = val

    return Tensor(weights)


def ternary_weight_analysis(weights: Tensor) -> dict:
    w = to_numpy(weights.data).flatten()
    total = len(w)
    zeros = int(np.sum(w == 0))
    positives = int(np.sum(w > 0))
    negatives = int(np.sum(w < 0))
    sparsity = zeros / total if total > 0 else 0.0
    return {
        "total": total,
        "zeros": zeros,
        "positives": positives,
        "negatives": negatives,
        "sparsity": sparsity,
        "compression_ratio_vs_fp32": 16.0,
        "compression_ratio_vs_fp16": 8.0,
    }
