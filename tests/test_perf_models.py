import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import ternary_zero as tz
from ncu_analysis import KernelMetrics, build_decode_summary, metric_target_for_name
from power_monitor import PowerSampler, energy_per_token_mj, parse_power_draw_w
from ternary_zero.perf import (
    GemvShape,
    HardwareSpec,
    MemoryTierFractions,
    ScalingObservation,
    compare_ternary_fp16,
    effective_bandwidth_gbps,
    fit_latency_scaling_law,
    fit_sparsity_scaling_law,
    issue_time_us,
    occupancy_ratio,
    predict_latency_from_scaling,
    projection_to_observation,
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

    def test_projection_to_observation_keeps_shape_and_projection_terms(self):
        hardware = HardwareSpec(
            peak_compute_gops=8500.0,
            dram_bandwidth_gbps=272.0,
            l2_bandwidth_gbps=500.0,
            shared_bandwidth_gbps=2500.0,
        )
        shape = GemvShape(m=1024, n=2048, sparsity=0.25)
        comparison = compare_ternary_fp16(
            shape=shape,
            hardware=hardware,
            occupancy=occupancy_ratio(1024, 1536),
            warp_efficiency_value=warp_efficiency(32),
            ternary_memory_tiers=MemoryTierFractions(dram=0.8, l2=0.2),
            decode_time_us=2.0,
        )
        obs = projection_to_observation(shape, comparison.ternary)
        assert obs.kernel == "ternary"
        assert obs.m == shape.m
        assert obs.n == shape.n
        assert obs.sparsity == shape.sparsity
        assert obs.latency_us == comparison.ternary.projected_latency_us
        assert obs.weight_bytes == comparison.ternary.weight_bytes
        assert obs.useful_ops == comparison.ternary.useful_ops


class TestScalingFits:
    def test_latency_scaling_fit_recovers_linear_coefficients(self):
        observations = []
        intercept = 3.0
        weight_term = 0.25
        op_term = 0.5
        for weight_bytes, useful_ops in ((4.0, 8.0), (8.0, 10.0), (16.0, 20.0), (20.0, 24.0)):
            latency = intercept + weight_term * weight_bytes + op_term * useful_ops
            observations.append(
                ScalingObservation(
                    kernel="ternary",
                    m=1,
                    n=1,
                    sparsity=0.5,
                    latency_us=latency,
                    weight_bytes=weight_bytes,
                    useful_ops=useful_ops,
                    total_bytes=weight_bytes,
                    nonzero_fraction=0.5,
                )
            )

        fit = fit_latency_scaling_law(observations, kernel="ternary")
        predicted = predict_latency_from_scaling(fit, 12.0, 14.0)
        expected = intercept + weight_term * 12.0 + op_term * 14.0

        assert np.isclose(fit.intercept_us, intercept)
        assert np.isclose(fit.us_per_weight_byte, weight_term)
        assert np.isclose(fit.us_per_useful_op, op_term)
        assert np.isclose(fit.r2, 1.0)
        assert np.isclose(predicted, expected)

    def test_sparsity_scaling_fit_recovers_nonzero_fraction_slope(self):
        observations = []
        intercept = 4.0
        slope = 12.0
        for nonzero_fraction in (0.25, 0.5, 0.75, 1.0):
            observations.append(
                ScalingObservation(
                    kernel="ternary",
                    m=1,
                    n=1,
                    sparsity=1.0 - nonzero_fraction,
                    latency_us=intercept + slope * nonzero_fraction,
                    weight_bytes=16.0,
                    useful_ops=64.0 * nonzero_fraction,
                    total_bytes=20.0,
                    nonzero_fraction=nonzero_fraction,
                )
            )

        fit = fit_sparsity_scaling_law(observations, kernel="ternary")
        assert np.isclose(fit.intercept_us, intercept)
        assert np.isclose(fit.us_per_nonzero_fraction, slope)
        assert np.isclose(fit.r2, 1.0)


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


class TestPowerModeling:
    def test_power_parsing_and_energy_per_token(self):
        assert parse_power_draw_w("75.5") == 75.5
        assert parse_power_draw_w("75.5 W") == 75.5
        assert parse_power_draw_w("N/A") is None
        assert energy_per_token_mj(50.0, 0.2, 10) == 1000.0

    def test_power_sampler_summary_without_gpu_samples(self):
        sampler = PowerSampler(enabled=False, interval_s=0.01)
        with sampler:
            pass
        summary = sampler.summary(duration_s=0.5)
        assert summary["enabled"] is False
        assert summary["sample_count"] == 0
        assert summary["avg_power_w"] == 0.0
        assert summary["energy_j"] == 0.0


class TestNcuDecodeAnalysis:
    def test_metric_target_finds_exact_decode_classes(self):
        assert metric_target_for_name("smsp__sass_thread_inst_executed_op_bfe_pred_on.sum") == "bfe_instructions"
        assert metric_target_for_name("sm__sass_thread_inst_executed_op_prmt_pred_on.sum") == "prmt_instructions"
        assert metric_target_for_name("smsp__sass_thread_inst_executed_op_logic_pred_on.sum") == "lop3_instructions"

    def test_decode_summary_prefers_exact_instruction_classes(self):
        metrics = KernelMetrics(
            bfe_instructions=512.0,
            prmt_instructions=384.0,
            lop3_instructions=768.0,
            shuffle_instructions=128.0,
            integer_instructions=2048.0,
            fp16_instructions=1024.0,
        )
        summary = build_decode_summary(
            metrics,
            clock_ghz=2.0,
            throughputs_per_cycle={
                "bfe": 2.0,
                "prmt": 1.0,
                "lop3": 4.0,
                "shuffle": 1.0,
            },
        )
        assert summary["exact_instruction_classes"] is True
        assert summary["mode"] == "exact_ptx_classes"
        assert summary["bottleneck_class"] == "prmt"
        assert np.isclose(summary["decode_issue_bound_us"], 0.192)
        assert np.isclose(summary["instruction_mix_pct"]["lop3"], 42.8571428571)
