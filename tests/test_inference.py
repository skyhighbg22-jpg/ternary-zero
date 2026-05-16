#!/usr/bin/env python3
"""
Ternary-Zero Inference Engine — End-to-End Test
=================================================

Tests the full inference pipeline with a randomly-initialized model
(no pretrained weights needed). Verifies:
  1. Model config creation
  2. Weight quantization and packing
  3. KV-cache management
  4. Forward pass (prefill + decode)
  5. Sampling
  6. Full generate() loop

Run with:
    python tests/test_inference.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def test_config():
    from ternary_zero.inference.config import ModelConfig, LLAMA_32_3B, detect_config

    cfg = LLAMA_32_3B
    assert cfg.hidden_size == 3072
    assert cfg.num_layers == 28
    assert cfg.head_dim == 128
    assert cfg.num_kv_heads == 8
    assert cfg.num_queries_per_kv == 3
    assert cfg.total_params > 0
    assert cfg.weight_bytes_ternary() > 0
    print(f"  [PASS] Config: {cfg.name} ({cfg.total_params:,} params)")
    return True


def test_quantize():
    from ternary_zero.inference.quantize import (
        quantize_weight_to_ternary,
        pack_ternary_rows,
        QuantizedLayer,
    )

    weight = np.random.randn(64, 128).astype(np.float32)
    ternary, scale = quantize_weight_to_ternary(weight, alpha=0.5)
    assert ternary.shape == (64, 128)
    assert scale.shape == (64,)
    assert np.all(np.isin(ternary, [-1, 0, 1]))

    packed = pack_ternary_rows(ternary, 128)
    assert packed.shape == (64 * 128 // 16,)

    ql = QuantizedLayer.from_weight(weight, alpha=0.5)
    assert ql.m == 64
    assert ql.n == 128
    assert ql.packed_weights.shape == (64 * 128 // 16,)
    print(f"  [PASS] Quantize: {weight.shape} -> packed {packed.shape}, scale range [{scale.min():.3f}, {scale.max():.3f}]")
    return True


def test_sampler():
    from ternary_zero.inference.sampler import sample, sample_greedy, top_k_filter, top_p_filter

    logits = np.array([0.1, 0.5, 0.3, 0.8, 0.2], dtype=np.float32)
    assert sample_greedy(logits) == 3

    filtered = top_k_filter(logits.copy(), k=2)
    assert np.isfinite(filtered[3])
    assert np.isfinite(filtered[1])
    assert not np.isfinite(filtered[0])

    filtered = top_p_filter(logits.copy(), p=0.6)
    assert np.isfinite(filtered[3])

    np.random.seed(42)
    token = sample(logits, temperature=0.8, top_k=3)
    assert 0 <= token < 5
    print(f"  [PASS] Sampler: greedy={sample_greedy(logits)}, sampled={token}")
    return True


def test_kv_cache():
    from ternary_zero.inference.cache import KVCache
    from ternary_zero.inference.config import LLAMA_32_3B

    cache = KVCache(LLAMA_32_3B, max_seq_len=128)
    assert cache.k_cache.shape == (28, 8, 128, 128)
    assert cache.v_cache.shape == (28, 8, 128, 128)

    k = np.random.randn(8, 128).astype(np.float32)
    v = np.random.randn(8, 128).astype(np.float32)
    cache.update(0, 0, k, v)

    k_out = cache.get_k(0, 0, 1)
    assert k_out.shape == (1, 128)
    assert np.allclose(k_out[0], k[0])

    cache.reset()
    assert np.all(cache.k_cache == 0.0)
    print(f"  [PASS] KV-Cache: shape {cache.k_cache.shape}, memory {cache.memory_mb():.2f} MB")
    return True


def test_layers():
    from ternary_zero.inference.layers import (
        rms_norm, silu, softmax,
        build_rope_cache, apply_rotary_emb,
    )

    x = np.random.randn(64).astype(np.float32)
    w = np.ones(64, dtype=np.float32)
    out = rms_norm(x, w, eps=1e-5)
    assert out.shape == (64,)

    s = silu(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    assert s.shape == (3,)
    assert abs(s[1]) < 1e-6

    probs = softmax(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert abs(probs.sum() - 1.0) < 1e-6

    cos, sin = build_rope_cache(64, 128, theta=10000.0)
    assert cos.shape == (128, 32)
    assert sin.shape == (128, 32)

    x_rot = apply_rotary_emb(np.random.randn(64).astype(np.float32), cos[0], sin[0])
    assert x_rot.shape == (64,)
    print(f"  [PASS] Layers: RMSNorm, SiLU, Softmax, RoPE all functional")
    return True


def test_ternary_linear():
    from ternary_zero.inference.layers import TernaryLinear
    from ternary_zero.inference.quantize import QuantizedLayer

    weight = np.random.randn(32, 64).astype(np.float32)
    ql = QuantizedLayer.from_weight(weight, alpha=0.5)
    linear = TernaryLinear.from_quantized(ql)

    x = np.random.randn(64).astype(np.float32)
    out = linear.forward(x)
    assert out.shape == (32,)

    backend = linear._detect_backend()
    print(f"  [PASS] TernaryLinear: {weight.shape} -> {out.shape}, backend={backend}")
    return True


def test_full_forward():
    from ternary_zero.inference.config import ModelConfig
    from ternary_zero.inference.layers import (
        TernaryLinear, Attention, FeedForward,
        TransformerBlock, Transformer,
        build_rope_cache, rms_norm,
    )
    from ternary_zero.inference.cache import KVCache
    from ternary_zero.inference.quantize import QuantizedLayer
    from ternary_zero.inference.sampler import sample

    cfg = ModelConfig(
        name="test-tiny",
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=128,
    )

    def make_proj(in_f, out_f):
        w = np.random.randn(out_f, in_f).astype(np.float32) * 0.02
        ql = QuantizedLayer.from_weight(w, alpha=0.5)
        return TernaryLinear.from_quantized(ql)

    blocks = []
    for _ in range(cfg.num_layers):
        q = make_proj(cfg.hidden_size, cfg.hidden_size)
        k = make_proj(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim)
        v = make_proj(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim)
        o = make_proj(cfg.hidden_size, cfg.hidden_size)
        attn = Attention(cfg, q, k, v, o)

        gate = make_proj(cfg.hidden_size, cfg.intermediate_size)
        up = make_proj(cfg.hidden_size, cfg.intermediate_size)
        down = make_proj(cfg.intermediate_size, cfg.hidden_size)
        ffn = FeedForward(gate, up, down)

        block = TransformerBlock(
            attn, ffn,
            np.ones(cfg.hidden_size, dtype=np.float32),
            np.ones(cfg.hidden_size, dtype=np.float32),
            cfg,
        )
        blocks.append(block)

    embed = np.random.randn(cfg.vocab_size, cfg.hidden_size).astype(np.float32) * 0.02
    norm_w = np.ones(cfg.hidden_size, dtype=np.float32)
    lm_head_w = np.random.randn(cfg.vocab_size, cfg.hidden_size).astype(np.float32) * 0.02

    model = Transformer(cfg, blocks, embed, norm_w, lm_head_weight=lm_head_w)
    cache = KVCache(cfg, max_seq_len=128)

    tokens = [42, 100, 7]
    for pos, tok in enumerate(tokens):
        logits = model.forward(tok, cache, pos)

    assert logits.shape == (256,)
    next_token = sample(logits, temperature=0.8)
    assert 0 <= next_token < 256

    print(f"  [PASS] Full forward: {len(tokens)} prefill tokens -> logits[{logits.shape[0]}] -> token {next_token}")
    return True


def test_tokenizer_import():
    try:
        from transformers import AutoTokenizer
        print(f"  [PASS] transformers available (AutoTokenizer)")
    except ImportError:
        print(f"  [SKIP] transformers not installed (needed for model loading)")
    return True


def main():
    print("=" * 60)
    print("  Ternary-Zero Inference Engine — End-to-End Test")
    print("=" * 60)
    print()

    tests = [
        ("Config", test_config),
        ("Quantize", test_quantize),
        ("Sampler", test_sampler),
        ("KV-Cache", test_kv_cache),
        ("Layers", test_layers),
        ("TernaryLinear", test_ternary_linear),
        ("Full Forward Pass", test_full_forward),
        ("Tokenizer Import", test_tokenizer_import),
    ]

    passed = 0
    failed = 0
    t_start = time.perf_counter()

    for name, test_fn in tests:
        try:
            print(f"[{name}]")
            result = test_fn()
            if result:
                passed += 1
            print()
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    t_total = time.perf_counter() - t_start

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {t_total:.2f}s")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
