from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class MemoryTierFractions:
    dram: float = 1.0
    l2: float = 0.0
    shared: float = 0.0

    def normalized(self) -> "MemoryTierFractions":
        total = self.dram + self.l2 + self.shared
        if total <= 0:
            raise ValueError("memory tier fractions must sum to a positive value")
        return MemoryTierFractions(
            dram=self.dram / total,
            l2=self.l2 / total,
            shared=self.shared / total,
        )

    def to_dict(self) -> Dict[str, float]:
        norm = self.normalized()
        return {
            "dram": norm.dram,
            "l2": norm.l2,
            "shared": norm.shared,
        }


@dataclass(frozen=True)
class HardwareSpec:
    peak_compute_gops: float
    dram_bandwidth_gbps: float
    l2_bandwidth_gbps: float = 0.0
    shared_bandwidth_gbps: float = 0.0
    average_power_w: float = 0.0

    def __post_init__(self) -> None:
        if self.peak_compute_gops <= 0:
            raise ValueError("peak_compute_gops must be positive")
        if self.dram_bandwidth_gbps <= 0:
            raise ValueError("dram_bandwidth_gbps must be positive")


@dataclass(frozen=True)
class GemvShape:
    m: int
    n: int
    sparsity: float = 0.0
    activation_bytes: int = 2
    output_bytes: int = 2

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0:
            raise ValueError("GEMV dimensions must be positive")
        if not 0.0 <= self.sparsity <= 1.0:
            raise ValueError("sparsity must be in [0, 1]")
        if self.activation_bytes <= 0 or self.output_bytes <= 0:
            raise ValueError("activation/output bytes must be positive")


@dataclass(frozen=True)
class GemvProjection:
    kernel: str
    useful_ops: float
    dense_ops: float
    nonzero_fraction: float
    weight_bytes: float
    activation_bytes: float
    output_bytes: float
    total_bytes: float
    operational_intensity: float
    occupancy: float
    warp_efficiency: float
    effective_bandwidth_gbps: float
    compute_ceiling_gops: float
    bandwidth_ceiling_gops: float
    roofline_gops: float
    memory_time_us: float
    compute_time_us: float
    roofline_time_us: float
    decode_time_us: float
    reduce_time_us: float
    sync_time_us: float
    projected_latency_us: float
    energy_mj: float
    cache_tiers: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        return {
            "kernel": self.kernel,
            "useful_ops": self.useful_ops,
            "dense_ops": self.dense_ops,
            "nonzero_fraction": self.nonzero_fraction,
            "weight_bytes": self.weight_bytes,
            "activation_bytes": self.activation_bytes,
            "output_bytes": self.output_bytes,
            "total_bytes": self.total_bytes,
            "operational_intensity": self.operational_intensity,
            "occupancy": self.occupancy,
            "warp_efficiency": self.warp_efficiency,
            "effective_bandwidth_gbps": self.effective_bandwidth_gbps,
            "compute_ceiling_gops": self.compute_ceiling_gops,
            "bandwidth_ceiling_gops": self.bandwidth_ceiling_gops,
            "roofline_gops": self.roofline_gops,
            "memory_time_us": self.memory_time_us,
            "compute_time_us": self.compute_time_us,
            "roofline_time_us": self.roofline_time_us,
            "decode_time_us": self.decode_time_us,
            "reduce_time_us": self.reduce_time_us,
            "sync_time_us": self.sync_time_us,
            "projected_latency_us": self.projected_latency_us,
            "energy_mj": self.energy_mj,
            "cache_tiers": dict(self.cache_tiers),
        }


@dataclass(frozen=True)
class GemvComparison:
    ternary: GemvProjection
    fp16: GemvProjection
    speedup_vs_fp16: float
    compression_ratio_vs_fp16: float
    ideal_arithmetic_speedup: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "ternary": self.ternary.to_dict(),
            "fp16": self.fp16.to_dict(),
            "speedup_vs_fp16": self.speedup_vs_fp16,
            "compression_ratio_vs_fp16": self.compression_ratio_vs_fp16,
            "ideal_arithmetic_speedup": self.ideal_arithmetic_speedup,
        }


def effective_bandwidth_gbps(
    hardware: HardwareSpec,
    memory_tiers: MemoryTierFractions,
) -> float:
    tiers = memory_tiers.normalized()
    weighted_reciprocals = []
    for fraction, bandwidth in (
        (tiers.dram, hardware.dram_bandwidth_gbps),
        (tiers.l2, hardware.l2_bandwidth_gbps),
        (tiers.shared, hardware.shared_bandwidth_gbps),
    ):
        if fraction <= 0:
            continue
        if bandwidth <= 0:
            raise ValueError("non-zero memory tier fraction requires positive bandwidth")
        weighted_reciprocals.append(fraction / bandwidth)

    if not weighted_reciprocals:
        raise ValueError("at least one memory tier must have positive bandwidth")
    return 1.0 / sum(weighted_reciprocals)


def occupancy_ratio(active_threads_per_sm: float, max_threads_per_sm: float) -> float:
    if max_threads_per_sm <= 0:
        raise ValueError("max_threads_per_sm must be positive")
    return _clamp_unit(active_threads_per_sm / max_threads_per_sm)


def warp_efficiency(active_lanes: float, warp_size: int = 32) -> float:
    if warp_size <= 0:
        raise ValueError("warp_size must be positive")
    return _clamp_unit(active_lanes / float(warp_size))


def issue_time_us(
    instruction_counts: Mapping[str, float],
    throughputs_per_cycle: Mapping[str, float],
    clock_ghz: float,
) -> float:
    if clock_ghz <= 0:
        raise ValueError("clock_ghz must be positive")

    bottleneck_cycles = 0.0
    for key, count in instruction_counts.items():
        if count <= 0:
            continue
        throughput = throughputs_per_cycle.get(key, 0.0)
        if throughput <= 0:
            raise ValueError(f"missing positive throughput for instruction class '{key}'")
        bottleneck_cycles = max(bottleneck_cycles, count / throughput)

    return bottleneck_cycles / (clock_ghz * 1_000.0)


def _estimate_projection(
    kernel: str,
    useful_ops: float,
    dense_ops: float,
    weight_bytes: float,
    activation_bytes: float,
    output_bytes: float,
    hardware: HardwareSpec,
    occupancy: float,
    warp_efficiency_value: float,
    memory_tiers: MemoryTierFractions,
    decode_time_us: float = 0.0,
    reduce_time_us: float = 0.0,
    sync_time_us: float = 0.0,
) -> GemvProjection:
    occ = _clamp_unit(occupancy)
    eta_warp = _clamp_unit(warp_efficiency_value)
    total_bytes = weight_bytes + activation_bytes + output_bytes
    effective_bw = effective_bandwidth_gbps(hardware, memory_tiers)
    operational_intensity = useful_ops / total_bytes if total_bytes > 0 else 0.0
    compute_ceiling = hardware.peak_compute_gops * occ * eta_warp
    bandwidth_ceiling = operational_intensity * effective_bw
    roofline_gops = min(compute_ceiling, bandwidth_ceiling) if useful_ops > 0 else 0.0
    memory_time_us = (total_bytes / (effective_bw * 1e9)) * 1e6 if total_bytes > 0 else 0.0
    compute_time_us = (
        (useful_ops / (compute_ceiling * 1e9)) * 1e6
        if useful_ops > 0 and compute_ceiling > 0
        else 0.0
    )
    roofline_time_us = max(memory_time_us, compute_time_us)
    projected_latency_us = roofline_time_us + decode_time_us + reduce_time_us + sync_time_us
    energy_mj = hardware.average_power_w * projected_latency_us / 1_000.0
    nonzero_fraction = useful_ops / dense_ops if dense_ops > 0 else 0.0

    return GemvProjection(
        kernel=kernel,
        useful_ops=useful_ops,
        dense_ops=dense_ops,
        nonzero_fraction=nonzero_fraction,
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
        output_bytes=output_bytes,
        total_bytes=total_bytes,
        operational_intensity=operational_intensity,
        occupancy=occ,
        warp_efficiency=eta_warp,
        effective_bandwidth_gbps=effective_bw,
        compute_ceiling_gops=compute_ceiling,
        bandwidth_ceiling_gops=bandwidth_ceiling,
        roofline_gops=roofline_gops,
        memory_time_us=memory_time_us,
        compute_time_us=compute_time_us,
        roofline_time_us=roofline_time_us,
        decode_time_us=decode_time_us,
        reduce_time_us=reduce_time_us,
        sync_time_us=sync_time_us,
        projected_latency_us=projected_latency_us,
        energy_mj=energy_mj,
        cache_tiers=memory_tiers.to_dict(),
    )


def estimate_ternary_gemv(
    shape: GemvShape,
    hardware: HardwareSpec,
    occupancy: float = 1.0,
    warp_efficiency_value: float = 1.0,
    memory_tiers: Optional[MemoryTierFractions] = None,
    decode_time_us: float = 0.0,
    reduce_time_us: float = 0.0,
    sync_time_us: float = 0.0,
) -> GemvProjection:
    nonzero_fraction = 1.0 - shape.sparsity
    dense_ops = float(shape.m * shape.n)
    useful_ops = dense_ops * nonzero_fraction
    weight_bytes = float(shape.m * shape.n) / 4.0
    tiers = memory_tiers or MemoryTierFractions()

    return _estimate_projection(
        kernel="ternary",
        useful_ops=useful_ops,
        dense_ops=dense_ops,
        weight_bytes=weight_bytes,
        activation_bytes=float(shape.n * shape.activation_bytes),
        output_bytes=float(shape.m * shape.output_bytes),
        hardware=hardware,
        occupancy=occupancy,
        warp_efficiency_value=warp_efficiency_value,
        memory_tiers=tiers,
        decode_time_us=decode_time_us,
        reduce_time_us=reduce_time_us,
        sync_time_us=sync_time_us,
    )


def estimate_fp16_gemv(
    shape: GemvShape,
    hardware: HardwareSpec,
    occupancy: float = 1.0,
    warp_efficiency_value: float = 1.0,
    memory_tiers: Optional[MemoryTierFractions] = None,
    sync_time_us: float = 0.0,
) -> GemvProjection:
    dense_ops = float(2 * shape.m * shape.n)
    tiers = memory_tiers or MemoryTierFractions()

    return _estimate_projection(
        kernel="fp16",
        useful_ops=dense_ops,
        dense_ops=dense_ops,
        weight_bytes=float(2 * shape.m * shape.n),
        activation_bytes=float(shape.n * shape.activation_bytes),
        output_bytes=float(shape.m * shape.output_bytes),
        hardware=hardware,
        occupancy=occupancy,
        warp_efficiency_value=warp_efficiency_value,
        memory_tiers=tiers,
        sync_time_us=sync_time_us,
    )


def compare_ternary_fp16(
    shape: GemvShape,
    hardware: HardwareSpec,
    occupancy: float = 1.0,
    warp_efficiency_value: float = 1.0,
    ternary_memory_tiers: Optional[MemoryTierFractions] = None,
    fp16_memory_tiers: Optional[MemoryTierFractions] = None,
    decode_time_us: float = 0.0,
    reduce_time_us: float = 0.0,
    sync_time_us: float = 0.0,
) -> GemvComparison:
    ternary = estimate_ternary_gemv(
        shape=shape,
        hardware=hardware,
        occupancy=occupancy,
        warp_efficiency_value=warp_efficiency_value,
        memory_tiers=ternary_memory_tiers,
        decode_time_us=decode_time_us,
        reduce_time_us=reduce_time_us,
        sync_time_us=sync_time_us,
    )
    fp16 = estimate_fp16_gemv(
        shape=shape,
        hardware=hardware,
        occupancy=occupancy,
        warp_efficiency_value=warp_efficiency_value,
        memory_tiers=fp16_memory_tiers,
        sync_time_us=sync_time_us,
    )
    speedup = (
        fp16.projected_latency_us / ternary.projected_latency_us
        if ternary.projected_latency_us > 0
        else math.inf
    )
    nonzero_fraction = max(1.0 - shape.sparsity, 0.0)
    ideal_arithmetic_speedup = math.inf if nonzero_fraction == 0 else 1.0 / nonzero_fraction

    return GemvComparison(
        ternary=ternary,
        fp16=fp16,
        speedup_vs_fp16=speedup,
        compression_ratio_vs_fp16=8.0,
        ideal_arithmetic_speedup=ideal_arithmetic_speedup,
    )
