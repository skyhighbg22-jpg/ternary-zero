import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ternary_zero as tz
from ternary_zero.perf import (
    GemvShape,
    HardwareSpec,
    MemoryTierFractions,
    compare_ternary_fp16,
    effective_bandwidth_gbps,
    issue_time_us,
    occupancy_ratio,
    warp_efficiency,
)


class TestPerformanceModels:
    def test_effective_bandwidth_increases_with_cache_reuse(self):
        hardware = HardwareSpec(
            peak_compute_gops=8500.0,
            dram_bandwidth_gbps=272.0,
            l2_bandwidth_gbps=500.0,
            shared_bandwidth_gbps=2500.0,
        )
        dram_only = effective_bandwidth_gbps(hardware, MemoryTierFractions(dram=1.0))
        cache_mix = effective_bandwidth_gbps(
            hardware,
            MemoryTierFractions(dram=0.7, l2=0.2, shared=0.1),
        )
        assert cache_mix > dram_only

    def test_compare_ternary_fp16_projects_speedup(self):
        hardware = HardwareSpec(
            peak_compute_gops=8500.0,
            dram_bandwidth_gbps=272.0,
            l2_bandwidth_gbps=500.0,
            shared_bandwidth_gbps=2500.0,
            average_power_w=75.0,
        )
        shape = GemvShape(m=4096, n=4096, sparsity=0.5)
        comparison = compare_ternary_fp16(
            shape=shape,
            hardware=hardware,
            occupancy=occupancy_ratio(1024, 1536),
            warp_efficiency_value=warp_efficiency(32),
            ternary_memory_tiers=MemoryTierFractions(dram=0.9, l2=0.1),
            fp16_memory_tiers=MemoryTierFractions(dram=1.0),
            decode_time_us=2.5,
            reduce_time_us=1.0,
            sync_time_us=0.5,
        )
        assert comparison.speedup_vs_fp16 > 1.0
        assert comparison.ternary.operational_intensity > 0.0
        assert comparison.ternary.energy_mj > 0.0
        assert comparison.ideal_arithmetic_speedup == 2.0

    def test_issue_time_uses_the_slowest_instruction_pipe(self):
        time_us = issue_time_us(
            instruction_counts={"bfe": 1024.0, "lop3": 256.0},
            throughputs_per_cycle={"bfe": 1.0, "lop3": 2.0},
            clock_ghz=2.0,
        )
        assert np.isclose(time_us, 0.512)


class TestQuantizationAnalysis:
    def test_weight_analysis_reports_sparsity_speedup_fields(self):
        weights = tz.tensor([1, 0, -1, 0, 1, 0, -1, 1], dtype=np.int8)
        stats = tz.quantize.ternary_weight_analysis(weights)
        assert stats["nonzeros"] == 5
        assert np.isclose(stats["nonzero_fraction"], 5 / 8)
        assert np.isclose(stats["ideal_arithmetic_speedup_vs_dense"], 8 / 5)

    def test_fp16_accumulation_error_bound_is_positive(self):
        weights = tz.tensor([[1, 0, -1, 1]], dtype=np.int8)
        activations = np.array([0.5, -0.25, 1.5, 2.0], dtype=np.float32)
        bound = tz.quantize.fp16_accumulation_error_bound(weights, activations, scale=0.5)
        assert len(bound["per_row_bound"]) == 1
        assert bound["max_abs_bound"] > 0.0
        assert bound["gamma_per_row"][0] > 0.0

    def test_quantization_noise_analysis_tracks_output_noise(self):
        weights = tz.tensor([[0.8, -0.7, 0.1, -0.05], [0.6, -0.9, 0.2, 0.0]])
        activations = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
        report = tz.quantize.quantization_noise_analysis(weights, alpha=0.5, activations=activations)
        assert report["mse"] >= 0.0
        assert report["output_noise_energy"] >= 0.0
        assert "snr_db" in report
