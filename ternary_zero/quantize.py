from typing import Tuple, Union
import numpy as np

from .tensor import Tensor
from .autograd.functions import TernaryQuantizeSTE
from ._backend import has_torch, to_numpy

if has_torch():
    import torch


def _as_numpy_array(value, dtype=None):
    if isinstance(value, Tensor):
        array = to_numpy(value.data)
    elif has_torch() and isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        return array.astype(dtype, copy=False)
    return array


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
    nonzeros = total - zeros
    positives = int(np.sum(w > 0))
    negatives = int(np.sum(w < 0))
    sparsity = zeros / total if total > 0 else 0.0
    nonzero_fraction = nonzeros / total if total > 0 else 0.0
    return {
        "total": total,
        "zeros": zeros,
        "nonzeros": nonzeros,
        "positives": positives,
        "negatives": negatives,
        "sparsity": sparsity,
        "nonzero_fraction": nonzero_fraction,
        "effective_arithmetic_density": nonzero_fraction,
        "ideal_arithmetic_speedup_vs_dense": (
            float("inf") if nonzero_fraction == 0.0 else 1.0 / nonzero_fraction
        ),
        "compression_ratio_vs_fp32": 16.0,
        "compression_ratio_vs_fp16": 8.0,
    }


def fp16_accumulation_error_bound(
    ternary_weights: Tensor,
    activations: Union[Tensor, np.ndarray],
    scale: float = 1.0,
    unit_roundoff: float = None,
) -> dict:
    tw = _as_numpy_array(ternary_weights, dtype=np.float32)
    act = _as_numpy_array(activations, dtype=np.float32).reshape(-1)
    if tw.ndim == 1:
        tw = tw.reshape(1, -1)
    if tw.shape[1] != act.size:
        raise ValueError(
            f"activations size {act.size} must match weight width {tw.shape[1]}"
        )

    u = float(unit_roundoff) if unit_roundoff is not None else float(np.finfo(np.float16).eps / 2.0)
    row_terms = np.abs(tw * float(scale)) * np.abs(act)[None, :]
    nonzero_per_row = np.count_nonzero(tw, axis=1).astype(np.float64)
    gamma = np.empty_like(nonzero_per_row, dtype=np.float64)
    for i, count in enumerate(nonzero_per_row):
        nu = count * u
        gamma[i] = np.inf if nu >= 1.0 else nu / (1.0 - nu)
    per_row_bound = gamma * row_terms.sum(axis=1)

    return {
        "unit_roundoff": u,
        "nonzero_terms_per_row": nonzero_per_row.tolist(),
        "gamma_per_row": gamma.tolist(),
        "per_row_bound": per_row_bound.tolist(),
        "mean_abs_bound": float(np.mean(per_row_bound)) if per_row_bound.size else 0.0,
        "max_abs_bound": float(np.max(per_row_bound)) if per_row_bound.size else 0.0,
    }


def quantization_noise_analysis(
    weights: Tensor,
    alpha: float = 0.5,
    activations: Union[Tensor, np.ndarray, None] = None,
) -> dict:
    ternary, scale = ternary_quantize(weights, alpha=alpha)
    dequantized = dequantize_ternary(ternary, scale)

    original = _as_numpy_array(weights, dtype=np.float32)
    reconstructed = _as_numpy_array(dequantized, dtype=np.float32)
    error = original - reconstructed
    error_power = float(np.mean(error ** 2)) if error.size else 0.0
    signal_power = float(np.mean(original ** 2)) if original.size else 0.0
    snr_db = float("inf") if error_power == 0.0 else 10.0 * np.log10(signal_power / error_power)

    report = {
        "scale": float(scale),
        "mse": error_power,
        "rmse": float(np.sqrt(error_power)),
        "mae": float(np.mean(np.abs(error))) if error.size else 0.0,
        "max_abs_error": float(np.max(np.abs(error))) if error.size else 0.0,
        "error_variance": float(np.var(error)) if error.size else 0.0,
        "signal_power": signal_power,
        "snr_db": float(snr_db),
        "sparsity": ternary_weight_analysis(ternary)["sparsity"],
    }

    if activations is not None:
        act = _as_numpy_array(activations, dtype=np.float32)
        if original.ndim == 1:
            error_matrix = error.reshape(1, -1)
        else:
            error_matrix = error.reshape(original.shape[0], -1)

        if act.ndim == 1:
            if error_matrix.shape[1] != act.size:
                raise ValueError(
                    f"activations size {act.size} must match weight width {error_matrix.shape[1]}"
                )
            output_error = error_matrix @ act
        elif act.ndim == 2:
            if error_matrix.shape[1] != act.shape[0]:
                raise ValueError(
                    f"activation leading dimension {act.shape[0]} must match weight width {error_matrix.shape[1]}"
                )
            output_error = error_matrix @ act
        else:
            raise ValueError("activations must be a vector or matrix")

        report.update({
            "output_noise_energy": float(np.sum(output_error ** 2)),
            "output_noise_variance": float(np.mean(output_error ** 2)),
            "output_noise_l2": float(np.linalg.norm(output_error)),
        })

    return report


# =====================================================================
# NumPy-Vectorized CPU Ternary GEMV (High-Performance Fallback)
# =====================================================================
#
# Operates directly on packed u32 weights using vectorized NumPy ops.
# This is the pure-Python/NumPy CPU fallback for environments without
# the Rust native extension or CUDA (e.g., GitHub Actions CI runners).
#
# Encoding: 00=0, 01=+1, 10=-1, packed LSB-first into u32.

_TERNARY_LUT = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)


def ternary_gemv_numpy(
    packed_weights: np.ndarray,
    activations: np.ndarray,
    m: int,
    n: int,
) -> np.ndarray:
    """High-performance NumPy-vectorized ternary GEMV on packed u32 weights.

    Computes: output[m] = sum_n(decode_ternary(packed[m, n/16]) * act[n])

    Args:
        packed_weights: Flat u32 array of shape [M * (N/16)].
        activations:    FP32 array of shape [N].
        m:              Number of output rows.
        n:              Number of input features (must be multiple of 16).

    Returns:
        FP32 array of shape [M].
    """
    if n % 16 != 0:
        raise ValueError(f"N must be multiple of 16, got {n}")

    pw = np.asarray(packed_weights, dtype=np.uint32).ravel()
    act = np.asarray(activations, dtype=np.float32).ravel()
    packed_cols = n // 16

    if pw.size != m * packed_cols:
        raise ValueError(
            f"packed_weights size {pw.size} != M*(N/16) = {m * packed_cols}"
        )
    if act.size != n:
        raise ValueError(f"activations size {act.size} != N = {n}")

    pw_2d = pw.reshape(m, packed_cols)

    decoded = np.empty((m, n), dtype=np.float32)
    for bit in range(16):
        bits = (pw_2d >> np.uint32(bit * 2)) & np.uint32(0b11)
        decoded[:, bit::16] = _TERNARY_LUT[bits]

    return decoded @ act


def ternary_gemm_numpy(
    packed_weights: np.ndarray,
    activations: np.ndarray,
    m: int,
    k: int,
    n: int,
) -> np.ndarray:
    """High-performance NumPy-vectorized ternary GEMM on packed u32 weights.

    Computes: output[m, n] = sum_k(decode_ternary(packed[m, k/16]) * act[k, n])

    Args:
        packed_weights: Flat u32 array of shape [M * (K/16)].
        activations:    FP32 array of shape [K, N].
        m:              Number of output rows.
        k:              Number of input features (must be multiple of 16).
        n:              Number of output columns.

    Returns:
        FP32 array of shape [M, N].
    """
    if k % 16 != 0:
        raise ValueError(f"K must be multiple of 16, got {k}")

    pw = np.asarray(packed_weights, dtype=np.uint32).ravel()
    act = np.asarray(activations, dtype=np.float32)
    packed_cols = k // 16

    if pw.size != m * packed_cols:
        raise ValueError(
            f"packed_weights size {pw.size} != M*(K/16) = {m * packed_cols}"
        )
    if act.shape != (k, n):
        raise ValueError(f"activations shape {act.shape} != ({k}, {n})")

    pw_2d = pw.reshape(m, packed_cols)

    decoded = np.empty((m, k), dtype=np.float32)
    for bit in range(16):
        bits = (pw_2d >> np.uint32(bit * 2)) & np.uint32(0b11)
        decoded[:, bit::16] = _TERNARY_LUT[bits]

    return decoded @ act
