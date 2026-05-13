"""
Comprehensive microGPT Benchmark Suite
=======================================
Measures: latency curves, throughput scaling, occupancy, real speedups,
          and transformer inference across Pure Python, NumPy, PyTorch, CuPy.

Usage:
    py benchmarks/run_benchmarks.py
"""

import argparse
import json
import os
import sys
import platform
import time
import statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from ternary_zero.perf import (
    GemvShape,
    HardwareSpec,
    MemoryTierFractions,
    compare_ternary_fp16,
    occupancy_ratio,
)


# ═══════════════════════════════════════════════════════════════════════════════
# System Info
# ═══════════════════════════════════════════════════════════════════════════════

def get_system_info():
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        info["cpu_count"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["torch_gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "not installed"
    try:
        import cupy
        info["cupy_version"] = cupy.__version__
        info["cupy_gpu"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        info["cupy_version"] = "not available"
    return info


def log_system_info(info):
    """Print system information for benchmark reproducibility."""
    print("=" * 70)
    print("System Information (for reproducibility)")
    print("=" * 70)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 70)
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 1: Per-Step Latency Curves (training)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_latency_curves(impl_name, run_fn, steps=20, num_runs=3):
    """Measure training latency with warmup and repeated runs for real statistics.

    Runs a warmup invocation (discarded), then num_runs measured invocations.
    Statistics (mean, std, median, p95) are computed from actual per-run mean
    step latencies -- no synthetic per-step data is generated.
    """
    warmup_steps = max(steps // 4, 3)
    print(f"    [warmup] {warmup_steps} steps...", end="", flush=True)
    try:
        run_fn(num_train_steps=warmup_steps, num_inference_samples=1)
    except Exception:
        pass
    print(" done.", flush=True)

    run_latencies = []
    final_r = None
    for run_i in range(num_runs):
        print(f"    [run {run_i + 1}/{num_runs}] {steps} steps...", end="", flush=True)
        r = run_fn(num_train_steps=steps, num_inference_samples=3)
        run_latencies.append(r["train_avg_step_ms"])
        final_r = r
        print(f" {r['train_avg_step_ms']:.1f} ms/step", flush=True)

    mean_ms = statistics.mean(run_latencies)
    return {
        "implementation": impl_name,
        "run_latencies_ms": run_latencies,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(run_latencies),
        "p95_ms": sorted(run_latencies)[min(int(0.95 * len(run_latencies)), len(run_latencies) - 1)],
        "std_ms": statistics.stdev(run_latencies) if len(run_latencies) > 1 else 0.0,
        "measurement_type": "per_run_aggregate",
        "num_runs": num_runs,
        "raw_results": final_r,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 2: Throughput Scaling (vary sequence length)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_throughput_scaling(impl_name, run_fn, steps=10):
    """Measure throughput at different effective sequence lengths."""
    r = run_fn(num_train_steps=steps, num_inference_samples=5)
    # The model processes variable-length names (avg ~5-6 chars)
    # Measure tokens/sec at different inference counts
    return {
        "implementation": impl_name,
        "inference_throughput_tokens_s": r["inference_throughput_tokens_s"],
        "train_throughput_steps_s": r["train_throughput_steps_s"],
        "tokens_per_sample": r.get("inference_tokens_per_sample", []),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 3: GPU Occupancy (CuPy-specific)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_gpu_occupancy():
    """Measure CuPy GPU memory utilization and SM occupancy."""
    try:
        import cupy as cp
        dev = cp.cuda.Device()
        mem_info = dev.mem_info
        props = cp.cuda.runtime.getDeviceProperties(0)
        memory_clock_khz = float(props.get("memoryClockRate", 0))
        memory_bus_width_bits = int(props.get("memoryBusWidth", 0))
        theoretical_bandwidth_gbps = (
            2.0 * memory_clock_khz * 1e3 * (memory_bus_width_bits / 8.0) / 1e9
            if memory_clock_khz > 0 and memory_bus_width_bits > 0
            else 0.0
        )
        return {
            "implementation": "cupy",
            "gpu_name": props["name"].decode(),
            "gpu_total_mem_mb": mem_info[1] / (1024**2),
            "gpu_free_mem_mb": mem_info[0] / (1024**2),
            "gpu_used_mem_mb": (mem_info[1] - mem_info[0]) / (1024**2),
            "sm_count": props["multiProcessorCount"],
            "max_threads_per_sm": props["maxThreadsPerMultiProcessor"],
            "max_threads_per_block": props["maxThreadsPerBlock"],
            "warp_size": props["warpSize"],
            "compute_capability": f"{props['major']}.{props['minor']}",
            "shared_mem_per_sm_bytes": props.get("sharedMemPerMultiprocessor", 0),
            "registers_per_sm": props.get("regsPerMultiprocessor", 0),
            "memory_clock_khz": memory_clock_khz,
            "memory_bus_width_bits": memory_bus_width_bits,
            "max_warps_per_sm": props["maxThreadsPerMultiProcessor"] / props["warpSize"],
            "theoretical_bandwidth_gbps": theoretical_bandwidth_gbps,
        }
    except Exception as e:
        return {"implementation": "cupy", "error": str(e)}


def build_gemv_projection_suite(gpu_info):
    """Generate roofline-style GEMV projections for representative decode shapes."""
    if not gpu_info or "error" in gpu_info:
        return []

    dram_gbps = gpu_info.get("theoretical_bandwidth_gbps", 0.0) or 272.0
    max_threads_per_sm = gpu_info.get("max_threads_per_sm", 1536)
    kernel_threads = min(256, gpu_info.get("max_threads_per_block", 256))
    # The custom kernel is tuned for 4 resident blocks/SM on Ada.
    occ = occupancy_ratio(min(4, max(1, int(max_threads_per_sm // kernel_threads))) * kernel_threads, max_threads_per_sm)
    hardware = HardwareSpec(
        peak_compute_gops=8_500.0,
        dram_bandwidth_gbps=dram_gbps,
        l2_bandwidth_gbps=max(dram_gbps * 1.8, dram_gbps),
        shared_bandwidth_gbps=max(dram_gbps * 8.0, dram_gbps),
        average_power_w=75.0,
    )
    shapes = [
        ("gpt2_small_layer", GemvShape(m=768, n=768, sparsity=0.5)),
        ("gpt2_medium_layer", GemvShape(m=1024, n=1024, sparsity=0.5)),
        ("llama_7b_layer", GemvShape(m=4096, n=4096, sparsity=0.5)),
        ("llama_13b_layer", GemvShape(m=5120, n=5120, sparsity=0.5)),
    ]
    projections = []
    for name, shape in shapes:
        comparison = compare_ternary_fp16(
            shape=shape,
            hardware=hardware,
            occupancy=occ,
            warp_efficiency_value=1.0,
            ternary_memory_tiers=MemoryTierFractions(dram=0.9, l2=0.1, shared=0.0),
            fp16_memory_tiers=MemoryTierFractions(dram=0.95, l2=0.05, shared=0.0),
            decode_time_us=2.5,
            reduce_time_us=1.0,
            sync_time_us=0.5,
        )
        entry = comparison.to_dict()
        entry["name"] = name
        projections.append(entry)
    return projections


# ═══════════════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="microGPT Benchmark Suite")
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--inference-samples", type=int, default=5)
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of measured runs per implementation (after warmup)")
    args = parser.parse_args()

    sys_info = get_system_info()
    log_system_info(sys_info)
    print(f"  train_steps: {args.train_steps}")
    print(f"  inference_samples: {args.inference_samples}")
    print(f"  num_runs: {args.num_runs}")
    print()
    sys.stdout.flush()

    all_latency = []
    all_throughput = []
    all_raw = {}

    # ── 1. Pure Python ──
    print("\n[1/6] Pure Python (baseline)...")
    sys.stdout.flush()
    try:
        from impl_pure_python import run_benchmark as run_pure
        lc = bench_latency_curves("pure_python", run_pure, args.train_steps, args.num_runs)
        ts = bench_throughput_scaling("pure_python", run_pure, min(args.train_steps, 10))
        all_latency.append(lc)
        all_throughput.append(ts)
        all_raw["pure_python"] = lc["raw_results"]
        print(f"  train={lc['mean_ms']:.1f}ms/step  inf={lc['raw_results']['inference_avg_latency_ms']:.1f}ms")
    except Exception as e:
        print(f"  FAILED: {e}")
    sys.stdout.flush()

    # ── 2. NumPy ──
    print("\n[2/6] NumPy (vectorized CPU)...")
    sys.stdout.flush()
    try:
        from impl_numpy import run_benchmark as run_numpy
        lc = bench_latency_curves("numpy", run_numpy, args.train_steps, args.num_runs)
        ts = bench_throughput_scaling("numpy", run_numpy, min(args.train_steps, 10))
        all_latency.append(lc)
        all_throughput.append(ts)
        all_raw["numpy"] = lc["raw_results"]
        print(f"  train={lc['mean_ms']:.1f}ms/step  inf={lc['raw_results']['inference_avg_latency_ms']:.1f}ms")
    except Exception as e:
        print(f"  FAILED: {e}")
    sys.stdout.flush()

    # ── 3. PyTorch ──
    print("\n[3/6] PyTorch (autograd)...")
    sys.stdout.flush()
    try:
        from impl_pytorch import run_benchmark as run_torch, DEVICE
        lc = bench_latency_curves("pytorch", run_torch, args.train_steps, args.num_runs)
        ts = bench_throughput_scaling("pytorch", run_torch, min(args.train_steps, 10))
        all_latency.append(lc)
        all_throughput.append(ts)
        all_raw["pytorch"] = lc["raw_results"]
        print(f"  train={lc['mean_ms']:.1f}ms/step  inf={lc['raw_results']['inference_avg_latency_ms']:.1f}ms  device={DEVICE}")
    except Exception as e:
        print(f"  FAILED: {e}")
    sys.stdout.flush()

    # ── 4. Ternary-Zero (FP32 Linear) ──
    print("\n[4/6] Ternary-Zero (FP32 Linear, primary library)...")
    sys.stdout.flush()
    try:
        from impl_ternary_zero import run_benchmark as run_tz
        lc = bench_latency_curves("ternary_zero", lambda **kw: run_tz(use_bitlinear=False, **kw), args.train_steps, args.num_runs)
        ts = bench_throughput_scaling("ternary_zero", lambda **kw: run_tz(use_bitlinear=False, **kw), min(args.train_steps, 10))
        all_latency.append(lc)
        all_throughput.append(ts)
        all_raw["ternary_zero"] = lc["raw_results"]
        print(f"  train={lc['mean_ms']:.1f}ms/step  inf={lc['raw_results']['inference_avg_latency_ms']:.1f}ms")
    except Exception as e:
        print(f"  FAILED: {e}")
    sys.stdout.flush()

    # ── 5. Ternary-Zero BitLinear ──
    print("\n[5/6] Ternary-Zero BitLinear (2-bit ternary quantized)...")
    sys.stdout.flush()
    try:
        lc = bench_latency_curves("tz_bitlinear", lambda **kw: run_tz(use_bitlinear=True, **kw), args.train_steps, args.num_runs)
        ts = bench_throughput_scaling("tz_bitlinear", lambda **kw: run_tz(use_bitlinear=True, **kw), min(args.train_steps, 10))
        all_latency.append(lc)
        all_throughput.append(ts)
        all_raw["tz_bitlinear"] = lc["raw_results"]
        print(f"  train={lc['mean_ms']:.1f}ms/step  inf={lc['raw_results']['inference_avg_latency_ms']:.1f}ms")
    except Exception as e:
        print(f"  FAILED: {e}")
    sys.stdout.flush()

    # ── 6. CuPy ──
    print("\n[6/6] CuPy (GPU)...")
    sys.stdout.flush()
    try:
        from impl_cupy import run_benchmark as run_cupy
        lc = bench_latency_curves("cupy", run_cupy, args.train_steps, args.num_runs)
        ts = bench_throughput_scaling("cupy", run_cupy, min(args.train_steps, 10))
        gpu_occ = bench_gpu_occupancy()
        all_latency.append(lc)
        all_throughput.append(ts)
        all_raw["cupy"] = lc["raw_results"]
        all_raw["cupy_gpu_occupancy"] = gpu_occ
        print(f"  train={lc['mean_ms']:.1f}ms/step  inf={lc['raw_results']['inference_avg_latency_ms']:.1f}ms  gpu={gpu_occ.get('gpu_name','?')}")
    except Exception as e:
        print(f"  FAILED: {e}")
    sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════════════
    # Generate Comparative Results
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPARATIVE RESULTS")
    print("=" * 70)

    # Speedup table
    if len(all_latency) >= 2:
        base = all_latency[0]
        print("\n--- Real Speedup vs Pure Python Baseline ---")
        print(f"{'Implementation':<20} {'Train (ms/step)':<18} {'Speedup':<10} {'Inf (ms)':<12} {'Speedup':<10} {'Mem (MB)':<10}")
        print("-" * 80)
        for lc in all_latency:
            impl = lc["implementation"]
            raw = lc["raw_results"]
            train_ms = lc["mean_ms"]
            inf_ms = raw["inference_avg_latency_ms"]
            mem = raw["train_peak_memory_mb"]
            train_spd = base["mean_ms"] / train_ms if train_ms > 0 else 0
            inf_spd = base["raw_results"]["inference_avg_latency_ms"] / inf_ms if inf_ms > 0 else 0
            print(f"{impl:<20} {train_ms:<18.1f} {train_spd:<10.1f}x {inf_ms:<12.1f} {inf_spd:<10.1f}x {mem:<10.1f}")
        sys.stdout.flush()

    # Throughput table
    print("\n--- Throughput Scaling ---")
    print(f"{'Implementation':<20} {'Train (steps/s)':<18} {'Inf (tokens/s)':<18}")
    print("-" * 56)
    for ts in all_throughput:
        print(f"{ts['implementation']:<20} {ts['train_throughput_steps_s']:<18.2f} {ts['inference_throughput_tokens_s']:<18.1f}")

    # Latency data
    print("\n--- Latency (per-run mean ms/step) ---")
    for lc in all_latency:
        impl = lc["implementation"]
        runs = lc["run_latencies_ms"]
        runs_str = ", ".join(f"{x:.1f}" for x in runs)
        print(f"  {impl}: mean={lc['mean_ms']:.1f}  std={lc['std_ms']:.1f}  "
              f"median={lc['median_ms']:.1f}  p95={lc['p95_ms']:.1f}  "
              f"runs=[{runs_str}]")

    # GPU occupancy
    if "cupy_gpu_occupancy" in all_raw:
        occ = all_raw["cupy_gpu_occupancy"]
        if "error" not in occ:
            print("\n--- GPU Occupancy (CuPy) ---")
            print(f"  GPU:              {occ['gpu_name']}")
            print(f"  Compute Cap:      {occ['compute_capability']}")
            print(f"  SM Count:         {occ['sm_count']}")
            print(f"  Max Threads/SM:   {occ['max_threads_per_sm']}")
            print(f"  Max Threads/Blk:  {occ['max_threads_per_block']}")
            print(f"  Warp Size:        {occ['warp_size']}")
            print(f"  Max Warps/SM:     {occ['max_warps_per_sm']:.0f}")
            print(f"  Theo BW:          {occ['theoretical_bandwidth_gbps']:.1f} GB/s")
            print(f"  Total VRAM:       {occ['gpu_total_mem_mb']:.0f} MB")
            print(f"  Used VRAM:        {occ['gpu_used_mem_mb']:.1f} MB")

    gemv_projection_suite = build_gemv_projection_suite(all_raw.get("cupy_gpu_occupancy"))
    if gemv_projection_suite:
        print("\n--- GEMV Roofline Projections ---")
        print(f"{'Shape':<20} {'FP16 us':<12} {'Ternary us':<12} {'Speedup':<10} {'OI':<10} {'Energy (mJ)':<12}")
        print("-" * 78)
        for projection in gemv_projection_suite:
            fp16 = projection["fp16"]
            ternary = projection["ternary"]
            print(
                f"{projection['name']:<20} "
                f"{fp16['projected_latency_us']:<12.2f} "
                f"{ternary['projected_latency_us']:<12.2f} "
                f"{projection['speedup_vs_fp16']:<10.2f} "
                f"{ternary['operational_intensity']:<10.3f} "
                f"{ternary['energy_mj']:<12.3f}"
            )

    # Transformer inference
    print("\n--- Transformer Inference Metrics ---")
    print(f"{'Implementation':<20} {'Tokens/sample':<16} {'Latency (ms)':<16} {'Tokens/s':<12}")
    print("-" * 64)
    for lc in all_latency:
        raw = lc["raw_results"]
        impl = lc["implementation"]
        tps = raw.get("inference_tokens_per_sample", [])
        avg_t = statistics.mean(tps) if tps else 0
        lat = raw["inference_avg_latency_ms"]
        tok_s = raw["inference_throughput_tokens_s"]
        print(f"{impl:<20} {avg_t:<16.1f} {lat:<16.1f} {tok_s:<12.1f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════════════
    output = {
        "system": sys_info,
        "config": {"train_steps": args.train_steps, "inference_samples": args.inference_samples, "num_runs": args.num_runs},
        "latency_curves": [{k: v for k, v in lc.items() if k != "raw_results"} for lc in all_latency],
        "throughput_scaling": all_throughput,
        "gpu_occupancy": all_raw.get("cupy_gpu_occupancy"),
        "gemv_projection_suite": gemv_projection_suite,
        "raw_results": {},
    }
    for name, raw in all_raw.items():
        if name == "cupy_gpu_occupancy":
            continue
        output["raw_results"][name] = {
            k: v for k, v in raw.items()
            if k not in ("train_losses", "inference_tokens_per_sample", "inference_latencies_ms")
        }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path = os.path.join(PROJECT_ROOT, "benchmarks", "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nJSON results: {out_path}")

    # Generate markdown table for BENCHMARKS.md
    md = generate_markdown(
        all_latency,
        all_throughput,
        all_raw,
        sys_info,
        args.num_runs,
        gemv_projection_suite,
    )
    md_path = os.path.join(PROJECT_ROOT, "benchmarks", "comparison_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown table: {md_path}")

    print("\nDone.")


def generate_markdown(
    all_latency,
    all_throughput,
    all_raw,
    sys_info,
    num_runs=3,
    gemv_projection_suite=None,
):
    """Generate markdown comparison table."""
    lines = [
        "# microGPT Implementation Benchmark: Actual Measured Results",
        "",
        f"**Date:** {sys_info['timestamp']}",
        f"**Platform:** {sys_info['platform']}",
        f"**CPU:** {sys_info['processor']} ({sys_info.get('cpu_count', '?')} cores)",
        f"**RAM:** {sys_info.get('ram_total_gb', '?')} GB",
        f"**PyTorch:** {sys_info.get('torch_version', '?')} (CUDA: {sys_info.get('torch_cuda', '?')})",
        f"**CuPy:** {sys_info.get('cupy_version', '?')}",
        "",
        "## Configuration",
        "",
        "- **Model:** microGPT (1 layer, 16 embd, 4 heads, 16 block_size, ~4192 params)",
        "- **Dataset:** Karpathy names.txt (32,033 names)",
        "- **Task:** Character-level language model training + autoregressive inference",
        f"- **Measurement:** Warmup run discarded; {num_runs} measured runs per implementation",
        "- **Statistics:** Mean, std, median, p95 computed from per-run mean step latencies",
        "",
        "## Latency (ms/step, training)",
        "",
        "| Implementation | Mean | Std Dev | Median | P95 | Runs |",
        "|---------------|------|---------|--------|-----|------|",
    ]
    for lc in all_latency:
        runs_str = ", ".join(f"{x:.1f}" for x in lc["run_latencies_ms"])
        lines.append(
            f"| {lc['implementation']} | {lc['mean_ms']:.1f} | {lc['std_ms']:.1f} | "
            f"{lc['median_ms']:.1f} | {lc['p95_ms']:.1f} | {runs_str} |"
        )

    lines += [
        "",
        "## Throughput",
        "",
        "| Implementation | Train (steps/s) | Inference (tokens/s) |",
        "|---------------|----------------|---------------------|",
    ]
    for ts in all_throughput:
        lines.append(
            f"| {ts['implementation']} | {ts['train_throughput_steps_s']:.2f} | "
            f"{ts['inference_throughput_tokens_s']:.1f} |"
        )

    lines += [
        "",
        "## Real Speedup vs Pure Python",
        "",
        "| Implementation | Train Speedup | Inference Speedup | Peak Memory (MB) |",
        "|---------------|--------------|-------------------|-----------------|",
    ]
    if all_latency:
        base = all_latency[0]
        for lc in all_latency:
            raw = lc["raw_results"]
            ts = base["mean_ms"] / lc["mean_ms"] if lc["mean_ms"] > 0 else 0
            ispd = base["raw_results"]["inference_avg_latency_ms"] / raw["inference_avg_latency_ms"] if raw["inference_avg_latency_ms"] > 0 else 0
            lines.append(
                f"| {lc['implementation']} | {ts:.1f}x | {ispd:.1f}x | "
                f"{raw['train_peak_memory_mb']:.1f} |"
            )

    # GPU occupancy
    if "cupy_gpu_occupancy" in all_raw:
        occ = all_raw["cupy_gpu_occupancy"]
        if "error" not in occ:
            lines += [
                "",
                "## GPU Occupancy (CuPy/NVIDIA)",
                "",
                "| Property | Value |",
                "|----------|-------|",
                f"| GPU | {occ['gpu_name']} |",
                f"| Compute Capability | {occ['compute_capability']} |",
                f"| SM Count | {occ['sm_count']} |",
                f"| Max Threads/SM | {occ['max_threads_per_sm']} |",
                f"| Warp Size | {occ['warp_size']} |",
                f"| Max Warps/SM | {occ['max_warps_per_sm']:.0f} |",
                f"| Theoretical BW | {occ['theoretical_bandwidth_gbps']:.1f} GB/s |",
                f"| VRAM Total | {occ['gpu_total_mem_mb']:.0f} MB |",
                f"| VRAM Used (idle) | {occ['gpu_used_mem_mb']:.1f} MB |",
            ]

    if gemv_projection_suite:
        lines += [
            "",
            "## GEMV Roofline Projections",
            "",
            "| Shape | FP16 Latency (us) | Ternary Latency (us) | Speedup | Ternary OI | Ternary Energy (mJ) |",
            "|-------|-------------------|----------------------|---------|------------|---------------------|",
        ]
        for projection in gemv_projection_suite:
            fp16 = projection["fp16"]
            ternary = projection["ternary"]
            lines.append(
                f"| {projection['name']} | {fp16['projected_latency_us']:.2f} | "
                f"{ternary['projected_latency_us']:.2f} | {projection['speedup_vs_fp16']:.2f} | "
                f"{ternary['operational_intensity']:.3f} | {ternary['energy_mj']:.3f} |"
            )

    lines += [
        "",
        "## Transformer Inference Measurements",
        "",
        "| Implementation | Avg Tokens/Sample | Latency (ms) | Tokens/s |",
        "|---------------|-------------------|-------------|----------|",
    ]
    import statistics as st
    for lc in all_latency:
        raw = lc["raw_results"]
        tps = raw.get("inference_tokens_per_sample", [])
        avg_t = st.mean(tps) if tps else 0
        lines.append(
            f"| {lc['implementation']} | {avg_t:.1f} | "
            f"{raw['inference_avg_latency_ms']:.1f} | "
            f"{raw['inference_throughput_tokens_s']:.1f} |"
        )

    lines += [
        "",
        "## Analysis",
        "",
        "### Performance Delta Explanation",
        "",
        "1. **Pure Python -> NumPy:** NumPy replaces Python scalar loops with vectorized "
        "array operations. The inner loops (linear, softmax, rmsnorm) become single C-level "
        "BLAS/LAPACK calls, eliminating Python interpreter overhead per element.",
        "",
        "2. **NumPy -> PyTorch:** PyTorch adds autograd (automatic differentiation) on top "
        "of similar vectorized operations. On CPU, PyTorch uses the same BLAS backend as NumPy "
        "(MKL/OpenBLAS), so raw compute throughput is comparable. The overhead comes from "
        "graph construction and gradient bookkeeping.",
        "",
        "3. **NumPy/PyTorch -> CuPy:** CuPy offloads all array operations to the GPU via CUDA. "
        "For this tiny model (~4K params), GPU kernel launch overhead dominates, so the speedup "
        "is modest. The advantage grows dramatically with larger models where memory bandwidth "
        "and compute parallelism dominate.",
        "",
        "### Computational Advantages",
        "",
        "- **Vectorization (NumPy):** Eliminates Python loop overhead; BLAS-optimized math",
        "- **Autograd (PyTorch):** Automatic gradient computation; GPU-ready with CUDA tensors",
        "- **GPU (CuPy):** Massive parallelism for large models; CUDA kernel fusion potential",
        "- **Pure Python:** Zero dependencies; educational clarity; complete transparency",
        "",
        "### Caveats",
        "",
        "- The microGPT model is extremely small (~4192 params). At this scale, function call "
        "overhead and Python interpreter overhead dominate, not actual compute.",
        "- Real transformer models (GPT-2: 117M params) show dramatically larger speedups with "
        "NumPy/PyTorch/CuPy because matrix multiplications dominate over interpreter overhead.",
        "- PyTorch is CPU-only in this benchmark (no CUDA build). GPU PyTorch would be competitive with CuPy.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
