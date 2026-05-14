#!/usr/bin/env python3
"""
Ternary-Zero Shape Matrix Benchmark Suite
==========================================
Executes an 80-point configuration matrix sweep across varying M and N
dimensions, measuring kernel latency and throughput for each configuration.
Outputs a comprehensive manifest.json for empirical validation.

Usage:
    python benchmarks/shape_matrix_benchmark.py
    python benchmarks/shape_matrix_benchmark.py --warmup 50 --iterations 1000
    python benchmarks/shape_matrix_benchmark.py --output benchmarks/output/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# Configuration Matrix: 80 Points (M x N)
# =====================================================================

M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
N_VALUES = [256, 512, 1024, 2048, 4096, 8192, 11008, 14336, 16384, 19456]

assert len(M_VALUES) * len(N_VALUES) == 80, (
    f"Expected 80-point matrix, got {len(M_VALUES)}x{len(N_VALUES)}={len(M_VALUES)*len(N_VALUES)}"
)


@dataclass(frozen=True)
class BenchConfig:
    m: int
    n: int
    label: str = ""

    def __post_init__(self):
        if self.n % 16 != 0:
            raise ValueError(f"N must be multiple of 16, got {self.n}")


def generate_config_matrix() -> List[BenchConfig]:
    configs = []
    for m in M_VALUES:
        for n in N_VALUES:
            label = f"M{m}_N{n}"
            configs.append(BenchConfig(m=m, n=n, label=label))
    return configs


# =====================================================================
# Benchmark Result
# =====================================================================

@dataclass
class BenchResult:
    m: int
    n: int
    label: str
    weight_bytes: int
    packed_weight_bytes: int
    activation_bytes: int
    output_bytes: int
    min_us: float = 0.0
    max_us: float = 0.0
    mean_us: float = 0.0
    median_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    std_us: float = 0.0
    gflops: float = 0.0
    bandwidth_gbps: float = 0.0
    num_iterations: int = 0
    warmup: int = 0
    backend: str = "unknown"
    success: bool = False
    error: Optional[str] = None


# =====================================================================
# GPU Kernel Benchmarker
# =====================================================================

class GpuKernelBenchmarker:
    def __init__(self, warmup: int = 50, iterations: int = 1000):
        self.warmup = warmup
        self.iterations = iterations
        self._core = None

    def _get_core(self):
        if self._core is None:
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from ternary_zero import _core
                if _core.has_cuda():
                    self._core = _core
                else:
                    self._core = None
            except ImportError:
                self._core = None
        return self._core

    def benchmark(self, config: BenchConfig) -> BenchResult:
        m, n = config.m, config.n
        weight_bytes = m * n * 2  # FP16 equivalent
        packed_weight_bytes = m * (n // 16) * 4  # packed u32
        activation_bytes = n * 2  # FP16
        output_bytes = m * 2  # FP16

        result = BenchResult(
            m=m, n=n, label=config.label,
            weight_bytes=weight_bytes,
            packed_weight_bytes=packed_weight_bytes,
            activation_bytes=activation_bytes,
            output_bytes=output_bytes,
            warmup=self.warmup,
            num_iterations=self.iterations,
        )

        core = self._get_core()
        if core is None:
            result.backend = "cpu-numpy"
            return self._benchmark_cpu(config, result)

        result.backend = "cuda"
        return self._benchmark_gpu(core, config, result)

    def _benchmark_gpu(self, core, config: BenchConfig, result: BenchResult) -> BenchResult:
        m, n = config.m, config.n
        rng = np.random.default_rng(42)
        weights = rng.choice([-1, 0, 1], size=(m * n,), p=[0.33, 0.33, 0.34]).astype(np.int8)
        activations = rng.standard_normal(n).astype(np.float32)

        try:
            packed = core.pack_ternary_to_u32_py(weights, n)
            raw = core.benchmark_kernel_gpu(
                packed, activations, m, n,
                warmup=self.warmup, iterations=self.iterations,
                use_fp32_acc=True,
            )

            result.min_us = float(raw["min_us"])
            result.max_us = float(raw["max_us"])
            result.mean_us = float(raw["mean_us"])
            result.median_us = float(raw["median_us"])
            result.p95_us = float(raw["p95_us"])
            result.gflops = float(raw["gflops"])

            timings_us = []
            if "p99_us" in raw:
                result.p99_us = float(raw["p99_us"])

            total_bytes = result.packed_weight_bytes + result.activation_bytes + result.output_bytes
            if result.mean_us > 0:
                result.bandwidth_gbps = (total_bytes / (result.mean_us * 1e-6)) / 1e9

            result.success = True
        except Exception as e:
            result.error = str(e)
            result.success = False

        return result

    def _benchmark_cpu(self, config: BenchConfig, result: BenchResult) -> BenchResult:
        m, n = config.m, config.n
        rng = np.random.default_rng(42)
        weights = rng.choice([-1, 0, 1], size=(m * n,), p=[0.33, 0.33, 0.34]).astype(np.int8)
        activations = rng.standard_normal(n).astype(np.float32)

        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from ternary_zero.quantize import ternary_gemv_numpy

            packed_cols = n // 16
            packed = np.zeros(m * packed_cols, dtype=np.uint32)
            for row in range(m):
                for pc in range(packed_cols):
                    word = np.uint32(0)
                    for bit in range(16):
                        val = weights[row * n + pc * 16 + bit]
                        if val == 0:
                            bits = np.uint32(0b00)
                        elif val == 1:
                            bits = np.uint32(0b01)
                        elif val == -1:
                            bits = np.uint32(0b10)
                        else:
                            bits = np.uint32(0b00)
                        word |= bits << np.uint32(bit * 2)
                    packed[row * packed_cols + pc] = word

            for _ in range(self.warmup):
                _ = ternary_gemv_numpy(packed, activations, m, n)

            timings = []
            for _ in range(self.iterations):
                t0 = time.perf_counter()
                _ = ternary_gemv_numpy(packed, activations, m, n)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1e6)

            timings.sort()
            result.min_us = timings[0]
            result.max_us = timings[-1]
            result.mean_us = sum(timings) / len(timings)
            result.median_us = timings[len(timings) // 2]
            result.p95_us = timings[int(len(timings) * 0.95)]
            result.p99_us = timings[int(len(timings) * 0.99)]
            result.std_us = float(np.std(timings))

            nnz = m * n * 0.5
            flops = 2.0 * nnz
            if result.mean_us > 0:
                result.gflops = flops / (result.mean_us * 1e-6) / 1e9
                total_bytes = result.packed_weight_bytes + result.activation_bytes + result.output_bytes
                result.bandwidth_gbps = (total_bytes / (result.mean_us * 1e-6)) / 1e9

            result.success = True
        except Exception as e:
            result.error = str(e)
            result.success = False

        return result


# =====================================================================
# Benchmark Suite Runner
# =====================================================================

@dataclass
class SuiteManifest:
    suite_name: str = "ternary-zero-shape-matrix"
    suite_version: str = "1.0.0"
    timestamp: str = ""
    platform: str = ""
    gpu_name: str = ""
    cuda_version: str = ""
    total_configs: int = 80
    successful_configs: int = 0
    failed_configs: int = 0
    total_time_s: float = 0.0
    warmup: int = 50
    iterations: int = 1000
    m_values: List[int] = field(default_factory=list)
    n_values: List[int] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "suite_version": self.suite_version,
            "timestamp": self.timestamp,
            "platform": self.platform,
            "gpu_name": self.gpu_name,
            "cuda_version": self.cuda_version,
            "total_configs": self.total_configs,
            "successful_configs": self.successful_configs,
            "failed_configs": self.failed_configs,
            "total_time_s": self.total_time_s,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "m_values": self.m_values,
            "n_values": self.n_values,
            "results": self.results,
            "summary": self.summary,
        }


def _detect_platform() -> str:
    import platform
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def _detect_gpu() -> Tuple[str, str]:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip().split("\n")[0]
        else:
            gpu_name = "Unknown"
    except Exception:
        gpu_name = "No GPU detected"

    cuda_version = "N/A"
    try:
        import subprocess
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line:
                    cuda_version = line.strip().split("release")[-1].strip().rstrip(",")
                    break
    except Exception:
        pass

    return gpu_name, cuda_version


def run_shape_matrix_benchmark(
    warmup: int = 50,
    iterations: int = 1000,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> SuiteManifest:
    configs = generate_config_matrix()

    manifest = SuiteManifest(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        platform=_detect_platform(),
        warmup=warmup,
        iterations=iterations,
        m_values=M_VALUES,
        n_values=N_VALUES,
        total_configs=len(configs),
    )

    manifest.gpu_name, manifest.cuda_version = _detect_gpu()

    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "manifest.json")

    if verbose:
        print("=" * 80)
        print("  TERNARY-ZERO SHAPE MATRIX BENCHMARK SUITE")
        print(f"  {len(configs)} configurations: {len(M_VALUES)} M x {len(N_VALUES)} N")
        print(f"  GPU: {manifest.gpu_name}")
        print(f"  CUDA: {manifest.cuda_version}")
        print(f"  Warmup: {warmup}, Iterations: {iterations}")
        print("=" * 80)
        print()

    benchmarker = GpuKernelBenchmarker(warmup=warmup, iterations=iterations)
    t_start = time.perf_counter()

    results_matrix: Dict[Tuple[int, int], BenchResult] = {}

    for idx, config in enumerate(configs):
        if verbose:
            print(
                f"  [{idx+1:>3}/{len(configs)}] M={config.m:>4}, N={config.n:>5} ... ",
                end="", flush=True,
            )

        result = benchmarker.benchmark(config)
        results_matrix[(config.m, config.n)] = result

        if result.success:
            manifest.successful_configs += 1
            if verbose:
                print(
                    f"median={result.median_us:>8.2f}us  "
                    f"p95={result.p95_us:>8.2f}us  "
                    f"GFLOPS={result.gflops:>7.1f}  "
                    f"BW={result.bandwidth_gbps:>6.1f} GB/s"
                )
        else:
            manifest.failed_configs += 1
            if verbose:
                print(f"FAILED: {result.error}")

        manifest.results.append(asdict(result))

    t_total = time.perf_counter() - t_start
    manifest.total_time_s = t_total

    successful = [r for r in manifest.results if r["success"]]
    if successful:
        latencies = [r["median_us"] for r in successful]
        gflops_list = [r["gflops"] for r in successful if r["gflops"] > 0]
        bw_list = [r["bandwidth_gbps"] for r in successful if r["bandwidth_gbps"] > 0]

        manifest.summary = {
            "latency_min_us": min(latencies),
            "latency_max_us": max(latencies),
            "latency_mean_us": sum(latencies) / len(latencies),
            "latency_median_us": sorted(latencies)[len(latencies) // 2],
            "gflops_max": max(gflops_list) if gflops_list else 0.0,
            "gflops_mean": sum(gflops_list) / len(gflops_list) if gflops_list else 0.0,
            "bandwidth_max_gbps": max(bw_list) if bw_list else 0.0,
            "bandwidth_mean_gbps": sum(bw_list) / len(bw_list) if bw_list else 0.0,
            "success_rate": manifest.successful_configs / manifest.total_configs,
        }

        latency_by_m: Dict[int, List[float]] = {}
        latency_by_n: Dict[int, List[float]] = {}
        for r in successful:
            m_val = r["m"]
            n_val = r["n"]
            latency_by_m.setdefault(m_val, []).append(r["median_us"])
            latency_by_n.setdefault(n_val, []).append(r["median_us"])

        manifest.summary["latency_by_m"] = {
            str(m): {
                "mean_us": sum(vals) / len(vals),
                "min_us": min(vals),
                "max_us": max(vals),
            }
            for m, vals in sorted(latency_by_m.items())
        }
        manifest.summary["latency_by_n"] = {
            str(n): {
                "mean_us": sum(vals) / len(vals),
                "min_us": min(vals),
                "max_us": max(vals),
            }
            for n, vals in sorted(latency_by_n.items())
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    if verbose:
        print()
        print("=" * 80)
        print("  BENCHMARK COMPLETE")
        print("=" * 80)
        print(f"  Total time:       {t_total:.1f}s")
        print(f"  Successful:       {manifest.successful_configs}/{manifest.total_configs}")
        print(f"  Failed:           {manifest.failed_configs}/{manifest.total_configs}")
        if manifest.summary:
            s = manifest.summary
            print(f"  Latency range:    {s['latency_min_us']:.2f} - {s['latency_max_us']:.2f} us")
            print(f"  Peak GFLOPS:      {s['gflops_max']:.1f}")
            print(f"  Peak bandwidth:   {s['bandwidth_max_gbps']:.1f} GB/s")
        print(f"  Manifest:         {output_path}")
        print("=" * 80)

    return manifest


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ternary-Zero 80-point shape matrix benchmark"
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=1000, help="Measurement iterations")
    parser.add_argument("--output", type=str, default=None, help="Output manifest.json path")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    manifest = run_shape_matrix_benchmark(
        warmup=args.warmup,
        iterations=args.iterations,
        output_path=args.output,
        verbose=not args.quiet,
    )

    return 0 if manifest.failed_configs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
