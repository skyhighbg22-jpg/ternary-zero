from __future__ import annotations

import numpy as np
from typing import Optional

from .config import ModelConfig


class KVCache:
    def __init__(self, config: ModelConfig, max_seq_len: int):
        self.config = config
        self.max_seq_len = max_seq_len
        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.k_cache = np.zeros(
            (config.num_layers, config.num_key_value_heads, max_seq_len, config.head_dim),
            dtype=np.float32,
        )
        self.v_cache = np.zeros(
            (config.num_layers, config.num_key_value_heads, max_seq_len, config.head_dim),
            dtype=np.float32,
        )

    def update(
        self,
        layer_idx: int,
        position: int,
        k: np.ndarray,
        v: np.ndarray,
    ) -> None:
        self.k_cache[layer_idx, :, position, :] = k
        self.v_cache[layer_idx, :, position, :] = v

    def get_k(
        self,
        layer_idx: int,
        kv_head: int,
        seq_len: int,
    ) -> np.ndarray:
        return self.k_cache[layer_idx, kv_head, :seq_len]

    def get_v(
        self,
        layer_idx: int,
        kv_head: int,
        seq_len: int,
    ) -> np.ndarray:
        return self.v_cache[layer_idx, kv_head, :seq_len]

    def reset(self) -> None:
        self.k_cache[:] = 0.0
        self.v_cache[:] = 0.0

    def memory_mb(self) -> float:
        return (self.k_cache.nbytes + self.v_cache.nbytes) / (1024 * 1024)
