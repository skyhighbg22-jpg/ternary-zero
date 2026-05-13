from __future__ import annotations

import numpy as np


def sample_greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def sample_temperature(logits: np.ndarray, temperature: float) -> int:
    if temperature <= 0.0:
        return sample_greedy(logits)
    scaled = logits / temperature
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= len(logits):
        return logits
    threshold = np.sort(logits)[-k]
    logits = np.where(logits >= threshold, logits, -np.inf)
    return logits


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    if p <= 0.0 or p >= 1.0:
        return logits
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    sorted_probs = np.exp(sorted_logits - sorted_logits.max())
    sorted_probs /= sorted_probs.sum()
    cum_probs = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cum_probs, p) + 1
    mask = np.zeros_like(logits, dtype=bool)
    mask[sorted_indices[:cutoff_idx]] = True
    logits = np.where(mask, logits, -np.inf)
    return logits


def sample(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    logits = logits.copy().astype(np.float32)
    if top_k > 0:
        logits = top_k_filter(logits, top_k)
    if 0.0 < top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    if temperature <= 0.0:
        return sample_greedy(logits)
    return sample_temperature(logits, temperature)
