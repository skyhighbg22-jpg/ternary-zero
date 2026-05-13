#!/usr/bin/env python3
"""
NCU profile analyzer for Ternary-Zero GEMV benchmarking.

Parses Nsight Compute CSV exports, compares the custom kernel against cuBLAS,
and projects occupancy-aware roofline summaries on top of the measured data.

Usage:
  python ncu_analysis.py --custom custom.csv --cublas cublas.csv --M 1 --N 4096
"""

import argparse
import csv
import json
from dataclasses import dataclass
import os
import sys
from typing import Dict, Mapping, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ternary_zero.perf import (
    GemvShape,
    HardwareSpec,
    MemoryTierFractions,
    compare_ternary_fp16,
    issue_time_us,
)


@dataclass
class KernelMetrics:
    kernel_name: str = ""
    gpu_time_us: float = 0.0
    compute_throughput_pct: float = 0.0
    achieved_occupancy_pct: float = 0.0
    theoretical_occupancy_pct: float = 0.0
    warps_per_sm: float = 0.0
    active_warps_per_scheduler: float = 0.0
    active_threads_per_instruction: float = 0.0
    warp_execution_efficiency_pct: float = 0.0
    registers_per_thread: int = 0
    register_pressure_pct: float = 0.0
    global_load_bytes: float = 0.0
    global_store_bytes: float = 0.0
    achieved_bandwidth_gbps: float = 0.0
    theoretical_bandwidth_gbps: float = 0.0
    bandwidth_utilization_pct: float = 0.0
    l1_hit_rate_pct: float = 0.0
    l2_hit_rate_pct: float = 0.0
    shared_memory_bytes: float = 0.0
    instructions_executed: float = 0.0
    integer_instructions: float = 0.0
    bfe_instructions: float = 0.0
    prmt_instructions: float = 0.0
    lop3_instructions: float = 0.0
    shuffle_instructions: float = 0.0
    fp16_instructions: float = 0.0
    stall_memory_pct: float = 0.0
    stall_barrier_pct: float = 0.0
    stall_not_selected_pct: float = 0.0


METRIC_MAP = {
    "gpu__time_duration.sum": "gpu_time_us",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_throughput_pct",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed": "compute_throughput_pct",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed": "compute_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "sm__maximum_warps_per_active_cycle_pct": "theoretical_occupancy_pct",
    "sm__warps_active.avg.per_cycle_active": "active_warps_per_scheduler",
    "sm__warps_active.avg.per_cycle_elapsed": "warps_per_sm",
    "smsp__thread_inst_executed_per_inst_executed.ratio": "active_threads_per_instruction",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block_dynamic": "shared_memory_bytes",
    "dram__bytes.sum": "global_load_bytes",
    "dram__bytes_read.sum": "global_load_bytes",
    "dram__bytes_write.sum": "global_store_bytes",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "bandwidth_utilization_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "bandwidth_utilization_pct",
    "l1tex__t_sector_hit_rate.pct": "l1_hit_rate_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate_pct",
    "sm__sass_thread_inst_executed.sum": "instructions_executed",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum": "fp16_instructions",
    "sm__sass_thread_inst_executed_op_integer_pred_on.sum": "integer_instructions",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum": "integer_instructions",
    "sm__sass_thread_inst_executed_op_bfe_pred_on.sum": "bfe_instructions",
    "smsp__sass_thread_inst_executed_op_bfe_pred_on.sum": "bfe_instructions",
    "sm__sass_thread_inst_executed_op_prmt_pred_on.sum": "prmt_instructions",
    "smsp__sass_thread_inst_executed_op_prmt_pred_on.sum": "prmt_instructions",
    "sm__sass_thread_inst_executed_op_lop3_pred_on.sum": "lop3_instructions",
    "smsp__sass_thread_inst_executed_op_lop3_pred_on.sum": "lop3_instructions",
    "sm__sass_thread_inst_executed_op_logic_pred_on.sum": "lop3_instructions",
    "smsp__sass_thread_inst_executed_op_logic_pred_on.sum": "lop3_instructions",
    "sm__sass_thread_inst_executed_op_shuffle_pred_on.sum": "shuffle_instructions",
    "smsp__sass_thread_inst_executed_op_shuffle_pred_on.sum": "shuffle_instructions",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active": "stall_memory_pct",
    "smsp__warps_issue_stalled_barrier_per_warp_active.pct": "stall_barrier_pct",
    "smsp__warps_issue_stalled_not_selected_per_warp_active.pct": "stall_not_selected_pct",
}

METRIC_SUBSTRINGS = {
    "bfe_instructions": (
        "sass_thread_inst_executed_op_bfe",
        "sass_inst_executed_op_bfe",
        "op_bfe_pred_on",
    ),
    "prmt_instructions": (
        "sass_thread_inst_executed_op_prmt",
        "sass_inst_executed_op_prmt",
        "op_prmt_pred_on",
        "op_permute_pred_on",
    ),
    "lop3_instructions": (
        "sass_thread_inst_executed_op_lop3",
        "sass_inst_executed_op_lop3",
        "op_lop3_pred_on",
        "op_logic_pred_on",
        "op_lop_pred_on",
    ),
    "shuffle_instructions": (
        "sass_thread_inst_executed_op_shuffle",
        "sass_inst_executed_op_shuffle",
        "op_shuffle_pred_on",
        "op_shfl_pred_on",
    ),
}

DEFAULT_DECODE_THROUGHPUTS = {
    "bfe": 1.0,
    "prmt": 1.0,
    "lop3": 1.0,
    "shuffle": 1.0,
    "integer": 1.0,
}


def metric_target_for_name(name: str) -> Optional[str]:
    target = METRIC_MAP.get(name)
    if target is not None:
        return target

    lowered = name.lower()
    for inferred_target, patterns in METRIC_SUBSTRINGS.items():
        if any(pattern in lowered for pattern in patterns):
            return inferred_target
    return None


def parse_ncu_csv(filepath: str) -> KernelMetrics:
    metrics = KernelMetrics()
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Metric Name", "").strip()
            value_str = row.get("Metric Value", "0").strip()
            kernel_name = row.get("Kernel Name", "")
            if kernel_name:
                metrics.kernel_name = kernel_name

            try:
                value = float(value_str.replace(",", "").replace("%", ""))
            except (ValueError, AttributeError):
                continue

            target = metric_target_for_name(name)
            if target is None:
                continue
            current = getattr(metrics, target)
            if isinstance(current, int):
                setattr(metrics, target, max(current, int(value)))
            else:
                setattr(metrics, target, max(current, value))

    if metrics.registers_per_thread:
        metrics.register_pressure_pct = metrics.registers_per_thread / 255.0 * 100.0
    return metrics


def infer_warp_efficiency_pct(metrics: KernelMetrics) -> float:
    if metrics.warp_execution_efficiency_pct > 0:
        return metrics.warp_execution_efficiency_pct
    if metrics.active_threads_per_instruction > 0:
        return min(100.0, metrics.active_threads_per_instruction / 32.0 * 100.0)
    if metrics.theoretical_occupancy_pct > 0:
        return min(
            100.0,
            metrics.achieved_occupancy_pct / metrics.theoretical_occupancy_pct * 100.0,
        )
    return 100.0


def infer_memory_tiers(
    metrics: KernelMetrics,
    shared_fraction: float = 0.0,
) -> MemoryTierFractions:
    shared_fraction = max(0.0, min(0.5, shared_fraction))
    l2_fraction = max(0.0, min(1.0 - shared_fraction, metrics.l2_hit_rate_pct / 100.0))
    dram_fraction = max(0.0, 1.0 - l2_fraction - shared_fraction)
    if dram_fraction == 0.0 and l2_fraction == 0.0 and shared_fraction == 0.0:
        dram_fraction = 1.0
    return MemoryTierFractions(dram=dram_fraction, l2=l2_fraction, shared=shared_fraction)


def build_decode_summary(
    metrics: KernelMetrics,
    clock_ghz: float,
    throughputs_per_cycle: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    if clock_ghz <= 0:
        return {
            "mode": "disabled",
            "exact_instruction_classes": False,
            "instruction_counts": {},
            "instruction_mix_pct": {},
            "per_class_issue_us": {},
            "bottleneck_class": None,
            "decode_issue_bound_us": 0.0,
            "decode_mix_pct": 0.0,
        }

    decode_throughputs = dict(DEFAULT_DECODE_THROUGHPUTS)
    if throughputs_per_cycle:
        decode_throughputs.update(throughputs_per_cycle)

    exact_counts = {
        "bfe": metrics.bfe_instructions,
        "prmt": metrics.prmt_instructions,
        "lop3": metrics.lop3_instructions,
        "shuffle": metrics.shuffle_instructions,
    }
    exact_counts = {k: float(v) for k, v in exact_counts.items() if v > 0}

    if exact_counts:
        instruction_counts = exact_counts
        mode = "exact_ptx_classes"
        exact_instruction_classes = True
    else:
        instruction_counts = {
            k: float(v)
            for k, v in {
                "integer": metrics.integer_instructions,
                "shuffle": metrics.shuffle_instructions,
            }.items()
            if v > 0
        }
        mode = "aggregate_integer_fallback"
        exact_instruction_classes = False

    if not instruction_counts:
        return {
            "mode": mode,
            "exact_instruction_classes": exact_instruction_classes,
            "instruction_counts": {},
            "instruction_mix_pct": {},
            "per_class_issue_us": {},
            "bottleneck_class": None,
            "decode_issue_bound_us": 0.0,
            "decode_mix_pct": 0.0,
        }

    total_decode_insts = sum(instruction_counts.values())
    per_class_issue_us = {
        cls: issue_time_us(
            instruction_counts={cls: count},
            throughputs_per_cycle=decode_throughputs,
            clock_ghz=clock_ghz,
        )
        for cls, count in instruction_counts.items()
    }
    bottleneck_class = max(per_class_issue_us, key=per_class_issue_us.get)
    instruction_mix_pct = {
        cls: count / total_decode_insts * 100.0
        for cls, count in instruction_counts.items()
    }

    return {
        "mode": mode,
        "exact_instruction_classes": exact_instruction_classes,
        "instruction_counts": instruction_counts,
        "instruction_mix_pct": instruction_mix_pct,
        "per_class_issue_us": per_class_issue_us,
        "bottleneck_class": bottleneck_class,
        "decode_issue_bound_us": per_class_issue_us[bottleneck_class],
        "decode_mix_pct": total_decode_insts / max(total_decode_insts + metrics.fp16_instructions, 1.0) * 100.0,
    }


def derive_projection_summary(
    metrics: KernelMetrics,
    shape: GemvShape,
    hardware: HardwareSpec,
    clock_ghz: float,
    shared_fraction: float,
    decode_throughputs: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    warp_efficiency_pct = infer_warp_efficiency_pct(metrics)
    tiers = infer_memory_tiers(metrics, shared_fraction=shared_fraction)
    decode_summary = build_decode_summary(
        metrics,
        clock_ghz=clock_ghz,
        throughputs_per_cycle=decode_throughputs,
    )
    decode_issue_bound_us = float(decode_summary["decode_issue_bound_us"])

    comparison = compare_ternary_fp16(
        shape=shape,
        hardware=hardware,
        occupancy=max(metrics.achieved_occupancy_pct / 100.0, 1e-6),
        warp_efficiency_value=max(warp_efficiency_pct / 100.0, 1e-6),
        ternary_memory_tiers=tiers,
        fp16_memory_tiers=MemoryTierFractions(),
        decode_time_us=decode_issue_bound_us,
        reduce_time_us=1.0,
        sync_time_us=0.5,
    )

    total_measured_bytes = metrics.global_load_bytes + metrics.global_store_bytes
    measured_bandwidth_gbps = 0.0
    if metrics.gpu_time_us > 0 and total_measured_bytes > 0:
        measured_bandwidth_gbps = total_measured_bytes / metrics.gpu_time_us / 1e3

    return {
        "kernel_name": metrics.kernel_name,
        "warp_execution_efficiency_pct": warp_efficiency_pct,
        "measured_bandwidth_gbps": measured_bandwidth_gbps,
        "decode_issue_bound_us": decode_issue_bound_us,
        "decode_mix_pct": decode_summary["decode_mix_pct"],
        "decode_summary": decode_summary,
        "cache_tiers": tiers.to_dict(),
        "projection": comparison.to_dict(),
    }


def print_metric_row(label: str, custom_value, cublas_value, fmt: str = ".2f") -> None:
    print(f"{label:<38} {custom_value:>18{fmt}} {cublas_value:>18{fmt}}")


def print_comparison(
    custom: KernelMetrics,
    cublas: KernelMetrics,
    custom_summary: Optional[Dict[str, object]] = None,
    cublas_summary: Optional[Dict[str, object]] = None,
) -> None:
    width = 84
    print("=" * width)
    print("  GEMV Kernel Profiling Comparison: Custom vs cuBLAS FP16")
    print("=" * width)

    print(f"\n{'Metric':<38} {'Custom':>18} {'cuBLAS':>18}")
    print("-" * width)
    print("\n--- TIMING ---")
    print_metric_row("GPU Time (us)", custom.gpu_time_us, cublas.gpu_time_us)

    print("\n--- OCCUPANCY / WARP ---")
    print_metric_row("Achieved Occupancy (%)", custom.achieved_occupancy_pct, cublas.achieved_occupancy_pct)
    print_metric_row("Theoretical Occupancy (%)", custom.theoretical_occupancy_pct, cublas.theoretical_occupancy_pct)
    print_metric_row("Active Warps / SM", custom.warps_per_sm, cublas.warps_per_sm)
    print_metric_row("Warp Efficiency (%)", infer_warp_efficiency_pct(custom), infer_warp_efficiency_pct(cublas))

    print("\n--- MEMORY ---")
    print_metric_row("Bandwidth Utilization (%)", custom.bandwidth_utilization_pct, cublas.bandwidth_utilization_pct)
    print_metric_row("L1 Hit Rate (%)", custom.l1_hit_rate_pct, cublas.l1_hit_rate_pct)
    print_metric_row("L2 Hit Rate (%)", custom.l2_hit_rate_pct, cublas.l2_hit_rate_pct)
    print_metric_row("Global Load Bytes", custom.global_load_bytes, cublas.global_load_bytes, ".0f")
    print_metric_row("Global Store Bytes", custom.global_store_bytes, cublas.global_store_bytes, ".0f")

    print("\n--- INSTRUCTION MIX ---")
    print_metric_row("FP16 Instructions", custom.fp16_instructions, cublas.fp16_instructions, ".0f")
    print_metric_row("Integer Instructions", custom.integer_instructions, cublas.integer_instructions, ".0f")
    print_metric_row("BFE Instructions", custom.bfe_instructions, cublas.bfe_instructions, ".0f")
    print_metric_row("PRMT Instructions", custom.prmt_instructions, cublas.prmt_instructions, ".0f")
    print_metric_row("LOP3 Instructions", custom.lop3_instructions, cublas.lop3_instructions, ".0f")
    print_metric_row("Shuffle Instructions", custom.shuffle_instructions, cublas.shuffle_instructions, ".0f")
    print_metric_row("Memory Stall (%)", custom.stall_memory_pct, cublas.stall_memory_pct)

    speedup = cublas.gpu_time_us / max(custom.gpu_time_us, 1e-6)
    print("\n" + "=" * width)
    print("  TECHNICAL ANALYSIS")
    print("=" * width)
    print(f"\n  Measured speedup: {speedup:.2f}x (cuBLAS time / custom time)")

    if custom_summary is not None:
        projection = custom_summary["projection"]
        ternary = projection["ternary"]
        decode_summary = custom_summary["decode_summary"]
        decode_mix = ", ".join(
            f"{cls}={pct:.1f}%"
            for cls, pct in decode_summary["instruction_mix_pct"].items()
        ) or "n/a"
        print("\n  Custom kernel model:")
        print(f"    Warp efficiency:   {custom_summary['warp_execution_efficiency_pct']:.1f}%")
        print(f"    Decode mode:       {decode_summary['mode']}")
        print(f"    Decode mix:        {custom_summary['decode_mix_pct']:.1f}% of decode-vs-FP16 issue mix")
        print(f"    Decode classes:    {decode_mix}")
        print(f"    Decode bottleneck: {decode_summary['bottleneck_class'] or 'n/a'}")
        print(f"    Decode issue cap:  {custom_summary['decode_issue_bound_us']:.3f} us")
        print(f"    Measured BW:       {custom_summary['measured_bandwidth_gbps']:.2f} GB/s")
        print(f"    Roofline latency:  {ternary['projected_latency_us']:.2f} us")
        print(f"    Ternary OI:        {ternary['operational_intensity']:.3f} ops/byte")
        print(f"    Cache tiers:       {custom_summary['cache_tiers']}")

    if cublas_summary is not None:
        projection = cublas_summary["projection"]
        fp16 = projection["fp16"]
        print("\n  cuBLAS baseline model:")
        print(f"    Warp efficiency:   {cublas_summary['warp_execution_efficiency_pct']:.1f}%")
        print(f"    Measured BW:       {cublas_summary['measured_bandwidth_gbps']:.2f} GB/s")
        print(f"    Roofline latency:  {fp16['projected_latency_us']:.2f} us")
        print(f"    FP16 OI:           {fp16['operational_intensity']:.3f} ops/byte")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze NCU profiling reports for GEMV kernel comparison"
    )
    parser.add_argument("--custom", required=True, help="Custom kernel NCU CSV export")
    parser.add_argument("--cublas", required=True, help="cuBLAS kernel NCU CSV export")
    parser.add_argument("--M", type=int, required=True, help="Output rows for the profiled GEMV")
    parser.add_argument("--N", type=int, required=True, help="Input features for the profiled GEMV")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Zero fraction for ternary weights")
    parser.add_argument("--dram-gbps", type=float, default=272.0, help="Theoretical DRAM bandwidth")
    parser.add_argument("--l2-gbps", type=float, default=500.0, help="Approximate L2 bandwidth")
    parser.add_argument("--shared-gbps", type=float, default=2500.0, help="Approximate shared-memory bandwidth")
    parser.add_argument("--peak-compute-gops", type=float, default=8500.0, help="Peak CUDA-core throughput")
    parser.add_argument("--avg-power-w", type=float, default=75.0, help="Average board power for energy estimates")
    parser.add_argument("--clock-ghz", type=float, default=2.0, help="Approximate SM clock for issue-bound estimates")
    parser.add_argument("--shared-fraction", type=float, default=0.0, help="Optional shared-memory byte fraction")
    parser.add_argument("--bfe-throughput", type=float, default=1.0, help="Sustained BFE issue throughput per cycle")
    parser.add_argument("--prmt-throughput", type=float, default=1.0, help="Sustained PRMT issue throughput per cycle")
    parser.add_argument("--lop3-throughput", type=float, default=1.0, help="Sustained LOP3 issue throughput per cycle")
    parser.add_argument("--shuffle-throughput", type=float, default=1.0, help="Sustained shuffle issue throughput per cycle")
    parser.add_argument("--json-out", help="Write the combined analysis as JSON")
    args = parser.parse_args()

    custom = parse_ncu_csv(args.custom)
    cublas = parse_ncu_csv(args.cublas)
    hardware = HardwareSpec(
        peak_compute_gops=args.peak_compute_gops,
        dram_bandwidth_gbps=args.dram_gbps,
        l2_bandwidth_gbps=args.l2_gbps,
        shared_bandwidth_gbps=args.shared_gbps,
        average_power_w=args.avg_power_w,
    )
    shape = GemvShape(m=args.M, n=args.N, sparsity=args.sparsity)
    decode_throughputs = {
        "bfe": args.bfe_throughput,
        "prmt": args.prmt_throughput,
        "lop3": args.lop3_throughput,
        "shuffle": args.shuffle_throughput,
    }

    custom_summary = derive_projection_summary(
        custom, shape, hardware, args.clock_ghz, args.shared_fraction, decode_throughputs
    )
    cublas_summary = derive_projection_summary(
        cublas,
        GemvShape(m=args.M, n=args.N, sparsity=0.0),
        hardware,
        args.clock_ghz,
        0.0,
        decode_throughputs,
    )

    print_comparison(custom, cublas, custom_summary, cublas_summary)

    if args.json_out:
        payload = {
            "shape": {"M": args.M, "N": args.N, "sparsity": args.sparsity},
            "hardware": {
                "dram_gbps": args.dram_gbps,
                "l2_gbps": args.l2_gbps,
                "shared_gbps": args.shared_gbps,
                "peak_compute_gops": args.peak_compute_gops,
                "avg_power_w": args.avg_power_w,
                "clock_ghz": args.clock_ghz,
                "bfe_throughput": args.bfe_throughput,
                "prmt_throughput": args.prmt_throughput,
                "lop3_throughput": args.lop3_throughput,
                "shuffle_throughput": args.shuffle_throughput,
            },
            "custom": custom_summary,
            "cublas": cublas_summary,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nJSON summary: {args.json_out}")


if __name__ == "__main__":
    main()
