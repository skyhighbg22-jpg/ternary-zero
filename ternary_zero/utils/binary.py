from __future__ import annotations

import numpy as np
from ..tensor import Tensor


def pack_binary(weights: Tensor) -> Tensor:
    w = weights.data.flatten()
    n = len(w)
    packed_size = (n + 31) // 32
    packed = np.zeros(packed_size, dtype=np.uint32)

    for i in range(n):
        if w[i] > 0:
            packed[i // 32] |= np.uint32(1 << (i % 32))

    return Tensor(packed)


def unpack_binary(packed: Tensor, n: int) -> Tensor:
    p = packed.data.astype(np.uint32)
    result = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if (p[i // 32] >> np.uint32(i % 32)) & np.uint32(1):
            result[i] = 1.0
        else:
            result[i] = -1.0

    return Tensor(result)


def binary_matmul(
    a_packed: Tensor,
    b: Tensor,
    m: int,
    k: int,
    n: int,
) -> Tensor:
    a_unpacked = unpack_binary(a_packed, m * k)
    a_matrix = a_unpacked.data.reshape(m, k)
    b_matrix = b.data.reshape(k, n)
    result = a_matrix @ b_matrix
    return Tensor(result.astype(np.float32))


def bitcount(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32)
    x = x - ((x >> np.uint32(1)) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> np.uint32(2)) & np.uint32(0x33333333))
    x = (x + (x >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    x = x * np.uint32(0x01010101)
    return (x >> np.uint32(24)).astype(np.int32)


def popcount_xor(a: Tensor, b: Tensor) -> Tensor:
    a_p = a.data.astype(np.uint32)
    b_p = b.data.astype(np.uint32)
    xored = a_p ^ b_p
    counts = bitcount(xored)
    return Tensor(counts)


def hamming_distance(a: Tensor, b: Tensor) -> int:
    xored = a.data.astype(np.uint32) ^ b.data.astype(np.uint32)
    return int(np.sum(bitcount(xored)))


def binary_weight_stats(weights: Tensor) -> dict:
    w = weights.data.flatten()
    total = len(w)
    pos = int(np.sum(w > 0))
    neg = int(np.sum(w < 0))
    zero = int(np.sum(w == 0))
    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "zero": zero,
        "density": (pos + neg) / total if total > 0 else 0.0,
        "packed_size_bytes": ((total + 31) // 32) * 4,
    }
