#!/usr/bin/env python3
"""
FP16 Baseline Comparison Harness
==================================
Measures FP16 inference baseline using HuggingFace transformers for direct
comparison against Ternary-Zero quantized models.

Produces:
  - FP16 tokens/sec (decode throughput)
  - FP16 TTFT (time to first token)
  - FP16 VRAM footprint
  - FP16 WikiText-2 perplexity
  - Speedup/compression comparison table

Usage:
  python benchmarks/fp16_baseline.py --model meta-llama/Llama-3.2-3B
  python benchmarks/fp16_baseline.py --model meta-llama/Llama-2-7b --max-tokens 64
  python benchmarks/fp16_baseline.py --compare benchmarks/output/transformer_bench_llama_3_2_3b.json

Requires:
  pip install transformers torch accelerate
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FP16BaselineResult:
    model_name: str
    total_params: int
    weight_bytes_fp16: int
    vram_mb: float
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    ttft_s: float
    per_token_latency_ms: float
    perplexity: Optional[float] = None
    eval_tokens: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    total_time_s: float = 0.0


@dataclass
class ComparisonTable:
    model_name: str
    fp16: Optional[Dict[str, Any]] = None
    ternary: Optional[Dict[str, Any]] = None
    speedup_decode: float = 0.0
    speedup_prefill: float = 0.0
    compression_ratio: float = 0.0
    vram_reduction: float = 0.0
    ppl_degradation: float = 0.0


def get_vram_mb() -> float:
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return float(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0.0


def measure_fp16_inference(
    model_name: str,
    max_tokens: int = 128,
    prompt: str = "The meaning of life is",
    verbose: bool = True,
) -> FP16BaselineResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if verbose:
        print("=" * 80)
        print("  FP16 BASELINE: INFERENCE BENCHMARK")
        print(f"  Model: {model_name}")
        print("=" * 80)
        print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    vram_before = get_vram_mb()

    t_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    t_load = time.perf_counter() - t_load_start

    if device == "cpu":
        model = model.to(device)

    vram_after = get_vram_mb()

    total_params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    if verbose:
        print(f"  Loaded in {t_load:.1f}s")
        print(f"  Parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
        print(f"  Weight memory: {weight_bytes / (1024**2):.0f} MB")
        print(f"  VRAM: {vram_before:.0f} -> {vram_after:.0f} MB")
        print()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    model.eval()
    torch.cuda.synchronize() if device == "cuda" else None

    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    torch.cuda.synchronize() if device == "cuda" else None
    t_prefill = time.perf_counter() - t_prefill_start

    generated_ids = inputs["input_ids"]
    past_key_values = outputs.past_key_values
    first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = torch.cat([generated_ids, first_token], dim=-1)

    torch.cuda.synchronize() if device == "cuda" else None
    t_decode_start = time.perf_counter()

    for i in range(1, max_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=first_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, first_token], dim=-1)

        if first_token.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize() if device == "cuda" else None
    t_decode = time.perf_counter() - t_decode_start
    t_total = time.perf_counter() - t_prefill_start

    num_generated = generated_ids.shape[1] - prompt_len
    decode_tps = num_generated / t_decode if t_decode > 0 else 0
    prefill_tps = prompt_len / t_prefill if t_prefill > 0 else 0

    result = FP16BaselineResult(
        model_name=model_name,
        total_params=total_params,
        weight_bytes_fp16=weight_bytes,
        vram_mb=max(vram_before, vram_after),
        prefill_tokens_per_sec=prefill_tps,
        decode_tokens_per_sec=decode_tps,
        ttft_s=t_prefill,
        per_token_latency_ms=(t_decode / num_generated * 1000) if num_generated > 0 else 0,
        prompt_tokens=prompt_len,
        generated_tokens=num_generated,
        prefill_time_s=t_prefill,
        decode_time_s=t_decode,
        total_time_s=t_total,
    )

    if verbose:
        print(f"  Prompt: {prompt_len} tokens")
        print(f"  Generated: {num_generated} tokens")
        print(f"  Prefill: {t_prefill:.3f}s ({prefill_tps:.1f} tok/s)")
        print(f"  Decode:  {t_decode:.3f}s ({decode_tps:.1f} tok/s)")
        print(f"  TTFT:    {t_prefill:.3f}s")
        print(f"  Per-token: {result.per_token_latency_ms:.1f}ms")
        print(f"  VRAM:    {result.vram_mb:.0f} MB")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def measure_fp16_perplexity(
    model_name: str,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    max_chunks: int = 50,
    stride: int = 512,
    verbose: bool = True,
) -> Optional[float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if verbose:
        print()
        print("=" * 80)
        print("  FP16 BASELINE: PERPLEXITY EVALUATION")
        print("=" * 80)
        print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, dataset_config, split="validation")
        text = "\n".join(ds["text"])
    except Exception:
        if verbose:
            print("  [!] Cannot load dataset for FP16 baseline PPL")
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    max_length = model.config.max_position_embeddings
    if max_length > 2048:
        max_length = 2048

    nlls = []
    num_tokens = 0
    t0 = time.perf_counter()

    for begin_loc in range(0, min(input_ids.shape[1], max_chunks * stride), stride):
        end_loc = min(begin_loc + max_length, input_ids.shape[1])
        if end_loc - begin_loc < 2:
            break

        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()
        target_chunk[:, :-1] = input_chunk[:, 1:]
        target_chunk[:, -1] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)

        nlls.append(outputs.loss.float() * (end_loc - begin_loc - 1))
        num_tokens += end_loc - begin_loc - 1

        if verbose and len(nlls) % 10 == 0:
            running_ppl = torch.exp(torch.stack(nlls).sum() / num_tokens).item()
            print(f"    Chunk {len(nlls)}: running PPL = {running_ppl:.2f}")

        if len(nlls) >= max_chunks:
            break

    eval_time = time.perf_counter() - t0
    ppl = torch.exp(torch.stack(nlls).sum() / num_tokens).item()

    if verbose:
        print(f"  FP16 PPL: {ppl:.2f} ({num_tokens:,} tokens, {eval_time:.1f}s)")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return ppl


def compare_with_ternary(
    fp16_result: FP16BaselineResult,
    ternary_json_path: str,
    verbose: bool = True,
) -> ComparisonTable:
    with open(ternary_json_path) as f:
        ternary_data = json.load(f)

    table = ComparisonTable(
        model_name=fp16_result.model_name,
        fp16=asdict(fp16_result),
        ternary=ternary_data,
    )

    t_inf = ternary_data.get("inference", {})
    if t_inf and fp16_result.decode_tokens_per_sec > 0:
        t_decode = t_inf.get("decode_tokens_per_sec", 0)
        t_prefill = t_inf.get("prefill_tokens_per_sec", 0)
        table.speedup_decode = t_decode / fp16_result.decode_tokens_per_sec if fp16_result.decode_tokens_per_sec > 0 else 0
        table.speedup_prefill = t_prefill / fp16_result.prefill_tokens_per_sec if fp16_result.prefill_tokens_per_sec > 0 else 0

    t_quant = ternary_data.get("quantization", {})
    if t_quant:
        fp16_bytes = t_quant.get("original_bytes_fp16", 0)
        ternary_bytes = t_quant.get("packed_bytes_ternary", 0)
        table.compression_ratio = fp16_bytes / ternary_bytes if ternary_bytes > 0 else 0

    if verbose:
        print()
        print("=" * 80)
        print("  FP16 vs TERNARY-ZERO COMPARISON")
        print("=" * 80)
        print()
        print(f"  Model: {table.model_name}")
        print(f"  {'Metric':<35} {'FP16':>12} {'Ternary':>12} {'Ratio':>10}")
        print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")

        if t_inf:
            print(f"  {'Decode (tok/s)':<35} {fp16_result.decode_tokens_per_sec:>12.1f} {t_inf.get('decode_tokens_per_sec', 0):>12.1f} {table.speedup_decode:>9.2f}x")
            print(f"  {'Prefill (tok/s)':<35} {fp16_result.prefill_tokens_per_sec:>12.1f} {t_inf.get('prefill_tokens_per_sec', 0):>12.1f} {table.speedup_prefill:>9.2f}x")
            print(f"  {'TTFT (s)':<35} {fp16_result.ttft_s:>12.3f} {t_inf.get('time_to_first_token_s', 0):>12.3f}")

        if t_quant:
            fp16_mb = t_quant.get("original_bytes_fp16", 0) / (1024**2)
            ternary_mb = t_quant.get("packed_bytes_ternary", 0) / (1024**2)
            print(f"  {'Weight memory (MB)':<35} {fp16_mb:>12.1f} {ternary_mb:>12.1f} {table.compression_ratio:>9.1f}x")

        t_ppl = ternary_data.get("perplexity", {})
        if t_ppl and fp16_result.perplexity:
            t_ppl_val = t_ppl.get("ternary_ppl", 0)
            table.ppl_degradation = t_ppl_val - fp16_result.perplexity
            print(f"  {'Perplexity':<35} {fp16_result.perplexity:>12.2f} {t_ppl_val:>12.2f} {table.ppl_degradation:>+9.2f}")

        t_ctx = ternary_data.get("context_scaling", {})
        if t_ctx:
            print(f"  {'Max context':<35} {'N/A':>12} {t_ctx.get('max_context_ternary', 0):>12,}")

    return table


def main():
    parser = argparse.ArgumentParser(description="FP16 Baseline Comparison")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--perplexity", action="store_true", help="Run perplexity evaluation")
    parser.add_argument("--compare", type=str, default=None, help="Ternary JSON to compare against")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    result = measure_fp16_inference(
        args.model, max_tokens=args.max_tokens, prompt=args.prompt, verbose=verbose,
    )

    if args.perplexity:
        result.perplexity = measure_fp16_perplexity(args.model, verbose=verbose)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2)
        if verbose:
            print(f"\n  Results saved to: {args.output}")

    if args.compare:
        compare_with_ternary(result, args.compare, verbose=verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
