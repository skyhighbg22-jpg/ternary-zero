#!/usr/bin/env python3
"""
NCU Profile Analyzer for Ternary-Zero GEMV Benchmark.

Parses Nsight Compute CSV exports and generates a side-by-side
comparison table with the key profiling metrics.

Usage:
  python ncu_analysis.py --custom custom_kernel.csv --cublas cublas_kernel.csv
  python ncu_analysis.py --custom custom_kernel.ncu-rep --cublas cublas_kernel.ncu-rep

To export CSV from Nsight Compute CLI:
  ncu --import custom_kernel.ncu-rep --csv --page raw > custom_kernel.csv
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class KernelMetrics:
    kernel_name: str = ""
    
    # Timing
    gpu_time_us: float = 0.0
    
    # Compute
    sm_active_cycles: float = 0.0
    compute_throughput_pct: float = 0.0
    
    # Occupancy
    achieved_occupancy_pct: float = 0.0
    theoretical_occupancy_pct: float = 0.0
    warps_per_sm: float = 0.0
    active_warps_per_scheduler: float = 0.0
    
    # Registers
    registers_per_thread: int = 0
    register_pressure_pct: float = 0.0
    
    # Memory
    global_load_bytes: float = 0.0
    global_store_bytes: float = 0.0
    achieved_bandwidth_GBps: float = 0.0
    theoretical_bandwidth_GBps: float = 0.0
    bandwidth_utilization_pct: float = 0.0
    
    # Cache
    l1_hit_rate_pct: float = 0.0
    l2_hit_rate_pct: float = 0.0
    shared_memory_bytes: float = 0.0
    
    # Instructions
    instructions_executed: float = 0.0
    inst_per_cycle: float = 0.0
    stall_memory_pct: float = 0.0
    stall_barrier_pct: float = 0.0
    stall_not_selected_pct: float = 0.0
    
    # Transactions
    l2_read_transactions: float = 0.0
    l2_write_transactions: float = 0.0
    global_load_transactions: float = 0.0
    global_store_transactions: float = 0.0
    gld_efficiency_pct: float = 0.0
    gst_efficiency_pct: float = 0.0


METRIC_MAP = {
    "gpu__time_duration.sum": "gpu_time_us",
    "sm__cycles_active.avg": "sm_active_cycles",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "sm__warps_active.avg.per_cycle_active": "active_warps_per_scheduler",
    "launch__registers_per_thread": "registers_per_thread",
    "dram__bytes.sum": "global_load_bytes",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "bandwidth_utilization_pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "global_load_transactions",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": "global_store_transactions",
    "l1tex__t_sector_hit_rate.pct": "l1_hit_rate_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate_pct",
    "lts__t_sectors_op_read.sum": "l2_read_transactions",
    "lts__t_sectors_op_write.sum": "l2_write_transactions",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum": "instructions_executed",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active": "stall_memory_pct",
    "sm__warps_active.avg.per_cycle_elapsed": "warps_per_sm",
    "launch__shared_mem_per_block_dynamic": "shared_memory_bytes",
    "dram__bytes_read.sum": "global_load_bytes",
    "dram__bytes_write.sum": "global_store_bytes",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "bandwidth_utilization_pct",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed": "compute_throughput_pct",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed": "compute_throughput_pct",
}


def parse_ncu_csv(filepath: str) -> KernelMetrics:
    metrics = KernelMetrics()
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Metric Name", "").strip()
            value_str = row.get("Metric Value", "0").strip()
            kname = row.get("Kernel Name", "")
            
            if kname:
                metrics.kernel_name = kname
            
            try:
                value = float(value_str.replace(",", "").replace("%", ""))
            except (ValueError, AttributeError):
                continue
            
            if name in METRIC_MAP:
                setattr(metrics, METRIC_MAP[name], value)
    
    return metrics


def format_number(value: float, unit: str = "") -> str:
    if value == 0:
        return "N/A"
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f}G{unit}"
    if abs(value) >= 1e6:
        return f"{value/1e6:.2f}M{unit}"
    if abs(value) >= 1e3:
        return f"{value/1e3:.2f}K{unit}"
    return f"{value:.2f}{unit}"


def print_comparison(custom: KernelMetrics, cublas: KernelMetrics):
    W = 80
    print("=" * W)
    print("  GEMV Kernel Profiling Comparison: Custom vs cuBLAS FP16")
    print("=" * W)
    
    print(f"\n{'Metric':<40} {'Custom':>18} {'cuBLAS':>18}")
    print("-" * W)
    
    def row(label, c_val, b_val, fmt=".4f"):
        print(f"{label:<40} {c_val:>18{fmt}} {b_val:>18{fmt}}")
    
    # Timing
    print("\n--- TIMING ---")
    row("GPU Time (us)", custom.gpu_time_us, cublas.gpu_time_us)
    
    # Compute
    print("\n--- COMPUTE THROUGHPUT ---")
    row("Compute Utilization (%)", custom.compute_throughput_pct, cublas.compute_throughput_pct, ".1f")
    row("IPC", custom.inst_per_cycle, cublas.inst_per_cycle)
    
    # Occupancy
    print("\n--- OCCUPANCY ---")
    row("Achieved Occupancy (%)", custom.achieved_occupancy_pct, cublas.achieved_occupancy_pct, ".1f")
    row("Theoretical Occupancy (%)", custom.theoretical_occupancy_pct, cublas.theoretical_occupancy_pct, ".1f")
    row("Active Warps/SM", custom.warps_per_sm, cublas.warps_per_sm, ".1f")
    row("Active Warps/Scheduler", custom.active_warps_per_scheduler, cublas.active_warps_per_scheduler, ".2f")
    
    # Registers
    print("\n--- REGISTER PRESSURE ---")
    row("Registers/Thread", custom.registers_per_thread, cublas.registers_per_thread, "d")
    row("Register Pressure (%)", custom.register_pressure_pct, cublas.register_pressure_pct, ".1f")
    
    # Memory
    print("\n--- MEMORY HIERARCHY ---")
    row("Bandwidth Utilization (%)", custom.bandwidth_utilization_pct, cublas.bandwidth_utilization_pct, ".1f")
    row("Global Load Bytes", custom.global_load_bytes, cublas.global_load_bytes, ".0f")
    row("Global Store Bytes", custom.global_store_bytes, cublas.global_store_bytes, ".0f")
    row("L1 Cache Hit Rate (%)", custom.l1_hit_rate_pct, cublas.l1_hit_rate_pct, ".1f")
    row("L2 Cache Hit Rate (%)", custom.l2_hit_rate_pct, cublas.l2_hit_rate_pct, ".1f")
    row("L2 Read Transactions", custom.l2_read_transactions, cublas.l2_read_transactions, ".0f")
    row("L2 Write Transactions", custom.l2_write_transactions, cublas.l2_write_transactions, ".0f")
    
    # Stall analysis
    print("\n--- PIPELINE STALLS ---")
    row("Memory Stall (%)", custom.stall_memory_pct, cublas.stall_memory_pct, ".1f")
    row("Barrier Stall (%)", custom.stall_barrier_pct, cublas.stall_barrier_pct, ".1f")
    row("Not Selected (%)", custom.stall_not_selected_pct, cublas.stall_not_selected_pct, ".1f")
    
    # Analysis
    print("\n" + "=" * W)
    print("  TECHNICAL ANALYSIS")
    print("=" * W)
    
    speedup = cublas.gpu_time_us / max(custom.gpu_time_us, 0.001)
    
    print(f"\n  Speedup: {speedup:.2f}x (custom / cuBLAS)")
    
    if custom.bandwidth_utilization_pct < cublas.bandwidth_utilization_pct * 0.7:
        print(f"  [BOTTLENECK] Custom kernel memory bandwidth utilization")
        print(f"    ({custom.bandwidth_utilization_pct:.1f}%) is significantly lower than cuBLAS")
        print(f"    ({cublas.bandwidth_utilization_pct:.1f}%). The custom kernel is memory-bound")
        print(f"    with suboptimal memory access patterns.")
    
    if custom.achieved_occupancy_pct < cublas.achieved_occupancy_pct * 0.8:
        print(f"  [BOTTLENECK] Custom kernel occupancy ({custom.achieved_occupancy_pct:.1f}%)")
        print(f"    is lower than cuBLAS ({cublas.achieved_occupancy_pct:.1f}%).")
        print(f"    Registers/thread: {custom.registers_per_thread} vs {cublas.registers_per_thread}.")
    
    if custom.stall_memory_pct > 30:
        print(f"  [BOTTLENECK] High memory stall rate ({custom.stall_memory_pct:.1f}%)")
        print(f"    indicates memory latency hiding is insufficient.")
    
    if custom.l1_hit_rate_pct < 50:
        print(f"  [BOTTLENECK] Low L1 hit rate ({custom.l1_hit_rate_pct:.1f}%)")
        print(f"    suggests uncoalesced or strided global memory access.")
    
    # Memory savings
    print(f"\n  Memory Efficiency:")
    print(f"    Custom kernel weight size: M*N/8 bytes (2-bit packed)")
    print(f"    cuBLAS weight size:        M*N*2 bytes (FP16)")
    print(f"    Compression ratio:         16x")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NCU profiling reports for GEMV kernel comparison"
    )
    parser.add_argument("--custom", required=True, help="Custom kernel NCU CSV/report")
    parser.add_argument("--cublas", required=True, help="cuBLAS kernel NCU CSV/report")
    args = parser.parse_args()
    
    print(f"Loading custom kernel profile: {args.custom}")
    custom = parse_ncu_csv(args.custom)
    
    print(f"Loading cuBLAS kernel profile: {args.cublas}")
    cublas = parse_ncu_csv(args.cublas)
    
    print_comparison(custom, cublas)


if __name__ == "__main__":
    main()