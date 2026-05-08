# Benchmark Results and Methodology

This document contains detailed benchmark results and methodology for the Ternary-Zero W2A16 GEMV kernel.

## Benchmark Suite Overview

The Ternary-Zero benchmark suite measures performance across various matrix configurations to characterize the kernel's behavior under different workloads.

### Measurement Protocol

All benchmarks follow the rigorous measurement protocol defined in [METHODOLOGY.md](./METHODOLOGY.md):

1. **Warmup Phase**: 100 iterations to populate L2 cache and stabilize GPU clocks
2. **Measurement Phase**: 1000 iterations using `cudaEvent`-based timing for microsecond precision
3. **Statistical Reporting**: Mean, median, p99, standard deviation, min, and max latency
4. **Throughput Calculation**: Effective GB/s = (weight_bytes + activation_bytes + output_bytes) / median_latency

### Shape Matrix

Benchmarks are executed across the full shape matrix:
- M (output rows): {1, 4, 16, 64, 256}
- N (input features): {1024, 2048, 4096, 8192}
- Sparsity ρ₀: {0.0, 0.25, 0.50, 0.75}
- **Total data points**: 80

### Baseline Comparisons

Performance is compared against:
- cuBLAS `cublasHgemv` (FP16 GEMV)
- cuBLAS `cublasGemmEx` (INT8 GEMV)

## Results

### Latency Matrix (μs, median ± std)

*Results will be populated after benchmark execution*

| M \ N | 1024 | 2048 | 4096 | 8192 |
|-------|------|------|------|------|
| 1 (ρ₀=0.0) | X ± Y | X ± Y | X ± Y | X ± Y |
| 1 (ρ₀=0.5) | X ± Y | X ± Y | X ± Y | X ± Y |
| 4 (ρ₀=0.0) | X ± Y | X ± Y | X ± Y | X ± Y |
| ... | ... | ... | ... | ... |

### Speedup Matrix (× vs cuBLAS FP16)

*Results will be populated after benchmark execution*

| M \ N | 1024 | 2048 | 4096 | 8192 |
|-------|------|------|------|------|
| 1 | X.X× | X.X× | X.X× | X.X× |
| 4 | X.X× | X.X× | X.X× | X.X× |
| ... | ... | ... | ... | ... |

### Effective Bandwidth (GB/s)

*Results will be populated after benchmark execution*

| M \ N | 1024 | 2048 | 4096 | 8192 |
|-------|------|------|------|------|
| 1 | XX.X | XX.X | XX.X | XX.X |
| ... | ... | ... | ... | ... |

## Nsight Profiling

For representative configurations, Nsight Compute and Nsight Systems traces are collected to validate:
- Achieved occupancy and theoretical occupancy
- Global memory throughput and L2 cache hit rate
- Warp execution efficiency and instruction mix
- Register pressure and shared memory bank conflicts
- Stall reasons and kernel launch latency

## Reproducibility

All benchmark results are stored in the `experiments/` directory with full provenance tracking as defined in [ARCHITECTURE_GOVERNANCE.md](./ARCHITECTURE_GOVERNANCE.md#adr-001-data-provenance--observability-strategy):

- `manifest.json`: Complete run metadata including git commit, hardware, and software versions
- `env_snapshot.json`: Captured runtime environment
- Raw latency and throughput CSV files
- Summary statistics JSON
- Nsight Compute and Systems traces
- Accuracy validation files (CPU reference vs GPU output)
- Thermal logs

## Running Benchmarks

To execute the full benchmark suite:

```bash
# CPU benchmarks
cargo bench --bench cpu_kernels

# GPU benchmarks (requires CUDA-enabled build)
cargo bench --bench gpu_kernels

# End-to-end transformer validation
python -m benchmarks.transformer --model gpt2-small --device cuda
```

See [EXECUTION_PLAN.md](./EXECUTION_PLAN.md) for detailed benchmarking procedures and acceptance criteria.