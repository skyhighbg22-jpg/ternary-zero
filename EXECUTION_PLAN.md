# EXECUTION PLAN: Ternary-Zero Performance Research Roadmap

**Document ID:** EP-001
**Version:** 1.0.0
**Status:** ACTIVE — BINDING
**Classification:** Technical Specification / Engineering Roadmap
**Project:** Ternary-Zero — W2A16 Sub-Byte GEMV Inference Runtime
**Target Hardware:** NVIDIA RTX 4060 (Ada Lovelace, sm_89, 272 GB/s GDDR6)
**Dependencies:** ARCHITECTURE_GOVERNANCE.md (ADR-001 through ADR-004), METHODOLOGY.md

---

## 0. Executive Summary

This document defines the formal execution plan for elevating the Ternary-Zero project from a research prototype with analytical foundations to an empirically validated, architecturally characterized, and CI-hardened system. The current state — documented in `ARCHITECTURE_GOVERNANCE.md` and validated by codebase audit — reveals a project with strong theoretical grounding (mathematical formalism, measurement protocols, risk analysis) but zero empirical data, zero CI automation, zero profiling infrastructure, and no architectural portability beyond the target sm_89.

The five workstreams defined herein are **sequenced by dependency**: CI/CD (P1) gates all downstream verification; Benchmarking (P2) produces the data that Roofline (P3) and Transformer Validation (P4) consume; Portability (P5) is the final hardening step that removes the sm_89 lock-in. Each workstream specifies deliverables, acceptance criteria, resource requirements, and inter-workstream dependencies.

### Current State Assessment

| Dimension | Status | Evidence |
|---|---|---|
| CI/CD | **Absent** | No `.github/workflows/`, no `Makefile`, no linting config |
| GPU Benchmarks | **Absent** | `benches/gemv_bench.rs` contains CPU-only Criterion stubs; zero results generated |
| Roofline Analysis | **Absent** | Theoretical derivation in METHODOLOGY.md §3.1 only; no Nsight integration |
| Transformer Validation | **Absent** | §4.2 of METHODOLOGY.md defines KPIs but no implementation exists |
| Architectural Portability | **sm_89-only** | `build.rs` hardcodes `--gpu-architecture=sm_89`; zero `#ifdef __CUDA_ARCH__` guards in `ptx_utils.h` |

---

## 1. Workstream P1: CI/CD Infrastructure

### 1.1 Problem Statement

The project has **zero automated verification**. The 36 Python tests and 6 Rust unit tests exist but are never executed in a pipeline. There is no regression detection for correctness, no build verification across toolchain versions, and no gating mechanism for the data provenance requirements defined in ADR-001. Every subsequent workstream (P2–P5) depends on the ability to verify that code changes do not regress existing functionality.

### 1.2 Technical Objectives

| ID | Objective | Scope |
|---|---|---|
| P1-O1 | **Multi-stage build pipeline** | `maturin develop --release` on Windows + Linux; CUDA compilation via `build.rs` with `nvcc` discovery |
| P1-O2 | **Correctness regression gate** | Full `pytest tests/` suite (36 tests) + Rust `cargo test` (6 unit tests) on every PR |
| P1-O3 | **Lint and format enforcement** | `cargo fmt --check`, `cargo clippy -- -D warnings`, `ruff check` + `ruff format --check` for Python |
| P1-O4 | **CUDA compilation verification** | Smoke-test that `kernel/ternary_zero.cu` compiles with CUDA 12.x targeting sm_89 on the CI runner |
| P1-O5 | **Artifact caching** | Cache `target/` (Cargo), `~/.cargo/registry/`, and CUDA Toolkit installation to reduce CI wall time below 10 minutes |
| P1-O6 | **ADR-001 data provenance gate** | Validate that `experiments/schema/experiment_metadata.schema.json` exists and is valid JSON Schema before merge |

### 1.3 Deliverables

| Deliverable | Format | Location |
|---|---|---|
| GitHub Actions workflow | YAML | `.github/workflows/ci.yml` |
| Ruff configuration | TOML | `ruff.toml` or `pyproject.toml [tool.ruff]` |
| Clippy configuration | TOML | `clippy.toml` or `Cargo.toml [lints]` |
| Rustfmt configuration | TOML | `rustfmt.toml` |
| CI documentation | Markdown | `docs/dev/CI_GUIDE.md` |

### 1.4 Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        PR / Push to main                         │
│                            │                                     │
│              ┌─────────────┴─────────────┐                       │
│              ▼                           ▼                       │
│     ┌────────────────┐         ┌────────────────┐               │
│     │  Lint & Format  │         │  Build (CUDA)  │               │
│     │  ────────────── │         │  ────────────── │               │
│     │  rustfmt --check│         │  maturin develop│               │
│     │  clippy -D warn │         │  --release      │               │
│     │  ruff check     │         │  (sm_89 via     │               │
│     │  ruff format    │         │   build.rs)     │               │
│     └───────┬────────┘         └───────┬────────┘               │
│             │                           │                        │
│             ▼                           ▼                        │
│     ┌────────────────┐         ┌────────────────┐               │
│     │  Rust Tests     │         │  Python Tests   │               │
│     │  ────────────── │         │  ────────────── │               │
│     │  cargo test     │         │  pytest tests/  │               │
│     │  (6 unit tests) │         │  (36 test cases)│               │
│     └───────┬────────┘         └───────┬────────┘               │
│             │                           │                        │
│             └─────────────┬─────────────┘                        │
│                           ▼                                      │
│                  ┌─────────────────┐                             │
│                  │  Schema Gate     │                             │
│                  │  ─────────────── │                             │
│                  │  Validate ADR-001│                             │
│                  │  schema exists   │                             │
│                  └────────┬────────┘                             │
│                           ▼                                      │
│                      ✅ MERGE                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 1.5 Success Metrics

| Metric | Target | Measurement |
|---|---|---|
| CI wall time (full pipeline) | < 10 min | GitHub Actions duration |
| Test pass rate | 100% (42/42 tests) | `pytest` + `cargo test` exit codes |
| Lint violations at baseline | 0 | `clippy` + `ruff` exit code |
| CI coverage | All PRs and pushes to `main` | Branch protection rules |
| Build reproducibility | Identical `.pyd`/`.so` from same commit | Hash comparison across two runs |

### 1.6 Strategic Justification

CI/CD is **infrastructure debt that compounds**. Without it:
- P2 benchmark results cannot be trusted (unverified code changes may corrupt measurement)
- P3 Roofline data has no code provenance (which commit produced these numbers?)
- P4 transformer validation cannot be regression-tested
- P5 portability changes introduce cross-platform breakage with no detection

Per ADR-001 §1.4, the enforcement rules require "CI gate: no merge without valid manifest" — this is currently aspirational. P1 makes it real.

### 1.7 Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| CUDA Toolkit unavailable on GitHub-hosted runners | HIGH | HIGH | Use self-hosted runner with GPU, or separate CUDA build into a matrix job with `ubuntu-latest` + `windows-latest` for non-CUDA tests |
| Maturin build failure on CI due to missing PyO3/NumPy wheel | MEDIUM | MEDIUM | Pin `maturin>=1.5,<2.0` in `pyproject.toml`; use `maturin develop` in CI |
| Long build times due to Rust compilation | MEDIUM | LOW | Cargo caching via `actions/cache`; incremental compilation |

---

## 2. Workstream P2: Empirical Benchmarking Suite

### 2.1 Problem Statement

The project contains **zero empirical performance data**. METHODOLOGY.md §2.1–2.3 defines a rigorous measurement protocol (warmup iterations, statistical reporting, shape matrix), and `ARCHITECTURE_GOVERNANCE.md` ADR-001 specifies a full data provenance schema — but none of this is implemented. The existing `benches/gemv_bench.rs` contains CPU-only Criterion stubs for pack/quantize/GEMV that have never produced results.

To transition from "optimization" to "systems research," the project must produce **high-fidelity, quantitative data** that enables:
1. Latency characterization across the full M×N×sparsity shape matrix
2. Throughput-vs-scale analysis showing memory bandwidth utilization
3. Sparsity-to-performance curves demonstrating the zero-gate benefit
4. Hardware profiling evidence (Nsight Compute) supporting bottleneck identification

### 2.2 Technical Objectives

| ID | Objective | Scope |
|---|---|---|
| P2-O1 | **GPU kernel latency benchmarking harness** | Criterion-based GPU benchmarks using `cudaEvent` timing; must measure the CUDA kernel, not CPU overhead |
| P2-O2 | **Full shape matrix execution** | All combinations of M ∈ {1, 4, 16, 64, 256} × N ∈ {1024, 2048, 4096, 8192} × ρ₀ ∈ {0.0, 0.25, 0.50, 0.75} — 80 data points minimum |
| P2-O3 | **Baseline comparison measurements** | cuBLAS `cublasHgemv` (FP16) and cuBLAS `cublasGemmEx` INT8 on identical hardware, identical session |
| P2-O4 | **Statistical rigor enforcement** | 1000 iterations per data point; report mean, median, p50, p99, std, min, max |
| P2-O5 | **Throughput computation** | Effective GB/s = (weight_bytes + activation_bytes + output_bytes) / median_latency |
| P2-O6 | **Sparsity sweep curves** | Latency-vs-ρ₀ plots for each M×N combination; fit empirical model to characterize zero-gate benefit |
| P2-O7 | **Automated result serialization** | All results written to `experiments/runs/` per ADR-001 schema; `manifest.json` validated against schema |

### 2.3 Required Deliverables

| Deliverable | Format | Location |
|---|---|---|
| GPU benchmark harness | Rust (Criterion) or Python | `benches/gemv_gpu_bench.rs` or `scripts/benchmark_gpu.py` |
| Baseline comparison harness | Python (cuBLAS via ctypes/cupy) | `scripts/benchmark_baselines.py` |
| Environment capture script | Python | `scripts/capture_env.py` (per ADR-001 §1.3) |
| Experiment config files | TOML | `experiments/configs/*.toml` |
| Latency tables | CSV + JSON | `experiments/runs/*/raw_latency.csv`, `summary_statistics.json` |
| Throughput-vs-scale plots | PNG + source | `experiments/aggregate/throughput_vs_scale.png` |
| Sparsity-to-performance curves | PNG + source | `experiments/aggregate/sparsity_sweep.png` |
| Aggregate summary | CSV | `experiments/aggregate/shape_matrix_results.csv` |

### 2.4 Benchmark Harness Specification

#### 2.4.1 Measurement Protocol (from METHODOLOGY.md §2.2)

```python
# MANDATORY measurement loop structure
def measure_kernel(kernel_fn, inputs, warmup=100, iterations=1000):
    # Phase 1: Warmup — populate L2, stabilize GPU clocks
    for _ in range(warmup):
        kernel_fn(*inputs)
    cudaDeviceSynchronize()

    # Phase 2: Measurement — cudaEvent-based timing
    latencies_us = []
    for _ in range(iterations):
        start_event = cudaEventCreate()
        end_event = cudaEventCreate()
        cudaEventRecord(start_event)
        kernel_fn(*inputs)
        cudaEventRecord(end_event)
        cudaEventSynchronize(end_event)
        elapsed_ms = cudaEventElapsedTime(start_event, end_event)
        latencies_us.append(elapsed_ms * 1000.0)  # ms → μs
        cudaEventDestroy(start_event)
        cudaEventDestroy(end_event)

    return compute_statistics(latencies_us)
```

#### 2.4.2 Shape Matrix (from METHODOLOGY.md §2.3)

| Parameter | Values | Count |
|---|---|---|
| M (output rows) | {1, 4, 16, 64, 256} | 5 |
| N (input features) | {1024, 2048, 4096, 8192} | 4 |
| Sparsity ρ₀ | {0.0, 0.25, 0.50, 0.75} | 4 |
| **Total data points** | | **80** |

#### 2.4.3 Required Output Tables

**Table 1: Latency Matrix (μs, median ± std)**

| M \ N | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| 1 (ρ₀=0.0) | X ± Y | X ± Y | X ± Y | X ± Y |
| 1 (ρ₀=0.5) | X ± Y | X ± Y | X ± Y | X ± Y |
| 4 (ρ₀=0.0) | X ± Y | X ± Y | X ± Y | X ± Y |
| ... | ... | ... | ... | ... |

**Table 2: Speedup Matrix (× vs cuBLAS FP16)**

| M \ N | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| 1 | X.X× | X.X× | X.X× | X.X× |
| 4 | X.X× | X.X× | X.X× | X.X× |
| ... | ... | ... | ... | ... |

**Table 3: Effective Bandwidth (GB/s)**

| M \ N | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| 1 | XX.X | XX.X | XX.X | XX.X |
| ... | ... | ... | ... | ... |

### 2.5 Nsight Profiling Integration

#### 2.5.1 Nsight Compute (ncu) Requirements

For at least one representative configuration per M×N shape (M=1, N=4096, ρ₀=0.5 recommended as canonical), capture:

| Metric | Nsight Compute Section | Purpose |
|---|---|---|
| **Achieved occupancy** | Occupancy | % of maximum warps per SM |
| **Theoretical occupancy** | Occupancy | Register/shared-memory-limited ceiling |
| **Global memory throughput** | Memory Workload Analysis | Achieved vs theoretical bandwidth |
| **L2 cache hit rate** | Cache | Effectiveness of `cudaAccessPropertyPersisting` |
| **DRAM throughput** | Memory Workload Analysis | Actual DRAM bytes/cycle |
| **Warp execution efficiency** | Warp State | % of active threads per warp |
| **Instruction mix** | Compute (SM) | Ratio of FP16 ALU vs integer decode vs memory |
| **Register pressure** | Source Counters | Registers per thread vs `maxrregcount=64` |
| **Shared memory bank conflicts** | Memory Workload Analysis | Shared memory access patterns |
| **Stall reasons** | Warp State | Dominant stall (memory, execution, synchronization) |

```bash
# MANDATORY Nsight Compute invocation
ncu --set full \
    --kernel-name "ternary_gemv_kernel" \
    --launch-skip 100 --launch-count 10 \
    --export experiments/runs/<run_id>/nsight/profile \
    ./target/release/benchmark_gpu --shape M=1,N=4096,sparsity=0.5
```

#### 2.5.2 Nsight Systems (nsys) Requirements

For end-to-end timeline characterization:

```bash
nsys profile \
    --trace cuda,nvtx \
    --output experiments/runs/<run_id>/nsight_systems/timeline \
    ./target/release/benchmark_gpu --shape M=1,N=4096,sparsity=0.5
```

Capture: kernel launch latency, H2D/D2H transfer overhead, GPU idle gaps, stream utilization.

### 2.6 Success Metrics

| Metric | Target | Measurement |
|---|---|---|
| Shape matrix coverage | 80/80 data points | Count of unique (M, N, ρ₀) results |
| Statistical sample size | ≥ 1000 iterations per point | Manifest `measurement_iterations` field |
| Baseline comparison | cuBLAS FP16 + INT8 measured on same hardware | Non-null speedup columns in summary |
| Nsight Compute traces | ≥ 12 traces (one per M×N, M=1) | `.ncu-rep` files in `experiments/runs/*/nsight/` |
| Nsight Systems traces | ≥ 4 traces (one per M value) | `.nsys-rep` files in `experiments/runs/*/nsight_systems/` |
| Throughput utilization | > 60% of 272 GB/s theoretical | Computed from latency and byte count |
| Repeatability | CV (coefficient of variation) < 5% for median latency | std / mean across 1000 iterations |
| Result serialization | 100% ADR-001 schema compliance | JSON Schema validation of `manifest.json` |

### 2.7 Strategic Justification

The benchmarking suite is the **empirical foundation** for all subsequent claims. Without quantitative data:
- P3 (Roofline) has no measured data points to plot against the theoretical ceiling
- P4 (Transformer validation) cannot demonstrate speedup in tokens/sec or TTFT
- The publication (§6 of METHODOLOGY.md) has no results section
- Reviewers at MLSys/IEEE Micro will reject the submission for lack of evidence

The benchmarking suite also serves as the **continuous validation** mechanism for P1 (CI) — once automated, it detects performance regressions in addition to correctness failures.

---

## 3. Workstream P3: Roofline Model Integration

### 3.1 Problem Statement

The METHODOLOGY.md §3.2 provides a theoretical comparison of the ternary kernel vs Tensor Core pathways, and §3.1 derives the efficiency inequality. However, the project lacks a **formal Roofline analysis** that plots the kernel's achieved operational intensity (FLOP/byte) against the hardware's theoretical compute and memory bandwidth ceilings. Without this, the project cannot:
1. Demonstrate that the kernel is correctly memory-bound (as predicted)
2. Quantify the gap between achieved and theoretical bandwidth
3. Identify whether decode overhead shifts the kernel from memory-bound toward compute-bound
4. Contextualize performance relative to architectural limits (not just relative to cuBLAS)

### 3.2 Technical Objectives

| ID | Objective | Scope |
|---|---|---|
| P3-O1 | **Operational intensity computation** | Calculate bytes loaded/stored and FP16 operations performed per GEMV invocation; derive FLOP/byte ratio |
| P3-O2 | **Roofline plot generation** | Plot achieved GFLOPS vs operational intensity on log-log axes with compute ceiling (8.5 TFLOPS CUDA cores) and memory ceiling (272 GB/s) |
| P3-O3 | **Multi-configuration Roofline** | Generate Roofline for all M×N shapes at ρ₀=0.0 and ρ₀=0.5 to show sparsity effect on operational intensity |
| P3-O4 | **Bandwidth utilization analysis** | Compute % of theoretical bandwidth achieved; identify bandwidth ceiling gap |
| P3-O5 | **Compute utilization analysis** | Determine if decode overhead (BFE, PRMT, LOP3) contributes non-negligible compute cycles |
| P3-O6 | **Nsight-validated Roofline** | Overlay Nsight Compute's "Roofline" section output with independently computed data |

### 3.3 Roofline Model Formulation

#### 3.3.1 Operational Intensity for Ternary GEMV

For a single GEMV of shape M×N:

| Quantity | Formula | Value (M=1, N=4096) |
|---|---|---|
| **Bytes loaded (weights)** | MN/4 (2-bit packed) | 1,024 B |
| **Bytes loaded (activations)** | 2N (FP16) | 8,192 B |
| **Bytes stored (output)** | 2M (FP16) | 2 B |
| **Total bytes** | MN/4 + 2N + 2M | 9,218 B |
| **FP16 additions** | N (one per weight) | 4,096 |
| **Decode ops** | ~3N (BFE + PRMT + LOP3) | 12,288 |
| **Total FLOPS** | N (counted as FP16 ops) | 4,096 |
| **Operational intensity** | FLOPS / bytes | 0.444 FLOP/byte |

#### 3.3.2 RTX 4060 Roofline Ceilings

```
Achieved GFLOPS (log)
    │
    │  ┌─────────────────────────────────────── ← Peak FP16 CUDA: 8,500 GFLOPS
    │  │
    │  │         ╱ ← Peak INT8: 17,000 GFLOPS (Tensor Cores)
    │  │       ╱
    │  │     ╱
    │  │   ╱
    │  │ ╱ ← Memory BW ceiling: 272 GB/s × OI = GFLOPS
    │  │╱
    │  ╱
    │╱
    └────────────────────────────────────────── Operational Intensity (FLOP/byte)
       0.01   0.1    1      10     100
```

**Expected kernel placement:** At OI ≈ 0.44 FLOP/byte, the memory ceiling predicts:
- GFLOPS_ceiling = 272 × 0.44 = **120 GFLOPS** (vs peak 8,500 = 1.4% of compute ceiling)
- The kernel is **deeply memory-bound** — compute utilization is irrelevant; bandwidth utilization is everything.

#### 3.3.3 Sparsity Effect on Operational Intensity

As ρ₀ increases, the effective bytes loaded remains constant (packed format is fixed-width) but the effective useful FLOPS decreases. However, the zero-gate means the FP16 add still executes with a zero addend — so the hardware FLOPS count does not change. The **benefit** of sparsity is:
- Reduced effective data movement (if sparse tiles can be skipped — not currently implemented)
- Reduced energy per FP16 add (zero addend ≈ 0 energy)
- **No latency benefit** with current branchless implementation

This must be empirically validated: if latency improves with ρ₀, it indicates either (a) L2 cache effects from reduced working set, or (b) compiler/hardware-level optimization of zero-add operations.

### 3.4 Deliverables

| Deliverable | Format | Location |
|---|---|---|
| Roofline computation script | Python | `scripts/roofline_analysis.py` |
| Roofline plots | PNG (300 DPI) | `experiments/aggregate/roofline_m1.png`, `roofline_m4.png`, etc. |
| Nsight Roofline overlay | PNG | `experiments/runs/*/nsight/roofline_overlay.png` |
| Bandwidth utilization table | CSV | `experiments/aggregate/bandwidth_utilization.csv` |
| Roofline interpretation document | Markdown | `docs/bench/ROOFLINE_ANALYSIS.md` |

### 3.5 Success Metrics

| Metric | Target | Measurement |
|---|---|---|
| Roofline plots generated | ≥ 4 (one per M value at ρ₀=0.0) | File existence in `experiments/aggregate/` |
| Bandwidth utilization | > 60% of 272 GB/s for N ≥ 4096 | Computed from measured latency |
| Operational intensity match | < 10% deviation from theoretical OI | Compare measured vs computed FLOP/byte |
| Memory-bound confirmation | All data points fall on memory ceiling slope | Visual inspection of Roofline plot |
| Nsight occupancy | > 50% achieved occupancy | Nsight Compute `Occupancy` section |
| Nsight L2 hit rate | > 80% for persisting weight accesses | Nsight Compute `Cache` section |

### 3.6 Strategic Justification

The Roofline model elevates the research from "we made it faster" to "we characterized where it sits relative to the hardware's theoretical limits." This is the difference between:
- **Optimization paper:** "Our kernel is 4× faster than cuBLAS FP16" (reviewer: "but is it well-optimized?")
- **Architectural characterization paper:** "Our kernel achieves 85% of theoretical memory bandwidth at OI=0.44, confirming memory-boundedness with 5% gap attributable to decode overhead" (reviewer: "this is rigorous")

MLS and IEEE Micro reviewers specifically look for Roofline analysis in systems papers. NeurIPS systems track papers without it are typically rejected for insufficient systems contribution.

---

## 4. Workstream P4: Transformer Workload Validation

### 4.1 Problem Statement

The kernel operates on synthetic tensor shapes (M×N with controlled sparsity). While this validates the kernel in isolation, it does not demonstrate **real-world applicability**. METHODOLOGY.md §4.2 defines transformer validation KPIs (tokens/sec, TTFT, VRAM footprint, perplexity degradation) but no implementation exists. The project needs a scalable proof-of-concept that:
1. Validates the end-to-end pipeline (quantize → pack → GEMV → accumulate → output)
2. Demonstrates real workload behavior (attention patterns, layer-wise sparsity distributions)
3. Produces deployment-relevant metrics (tokens/sec, context window scaling)
4. Provides a perplexity degradation baseline for the research contribution

### 4.2 Technical Objectives

| ID | Objective | Scope |
|---|---|---|
| P4-O1 | **Lightweight Transformer definition** | Implement a minimal GPT-2-style model using existing `ternary_zero.nn` primitives (Linear, BitLinear, LayerNorm, Softmax, GELU) |
| P4-O2 | **Weight quantization pipeline** | Apply STE-aware ternary quantization to all Linear layers; measure and report per-layer sparsity distribution |
| P4-O3 | **Inference loop implementation** | Autoregressive token generation with KV-cache; measure tokens/sec and TTFT |
| P4-O4 | **Perplexity evaluation** | Compute perplexity on WikiText-2 (or synthetic equivalent) for FP16 baseline and ternary-quantized model |
| P4-O5 | **VRAM footprint measurement** | Report peak GPU memory for FP16 vs ternary at identical context length |
| P4-O6 | **Context window scaling** | Demonstrate that ternary model supports 3×+ longer context at fixed VRAM |
| P4-O7 | **Multi-model scaling** | Validate on GPT-2 Small (768d) and GPT-2 Medium (1024d) minimum |

### 4.3 Target Model Specifications

| Model | Hidden Dim (N) | Layers | Attention Heads | Ternary Weight Memory | FP16 Weight Memory | Compression |
|---|---|---|---|---|---|---|
| GPT-2 Small | 768 | 12 | 12 | ~2.3 MB | ~36 MB | 15.7× |
| GPT-2 Medium | 1024 | 24 | 16 | ~6.3 MB | ~100 MB | 15.9× |
| LLaMA-7B (simulated) | 4096 | 32 | 32 | ~112 MB | ~1.8 GB | 16.1× |

**Note:** GPT-2 Small has N=768, which is below the minimum N=1024 in the shape matrix. This tests the kernel at its lower performance boundary and validates the efficiency inequality prediction (METHODOLOGY.md §3.1) that MN_min ≈ 1.56M.

### 4.4 Architecture Specification

```python
# Target: Minimal GPT-2 implementation using ternary_zero primitives
class TernaryGPT2(tz.nn.Module):
    def __init__(self, config):
        self.wte = tz.nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = tz.nn.Embedding(config.block_size, config.n_embd)
        self.h = tz.nn.Sequential(*[
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        self.ln_f = tz.nn.LayerNorm(config.n_embd)
        self.lm_head = tz.nn.Linear(config.n_embd, config.vocab_size)

class TransformerBlock(tz.nn.Module):
    def __init__(self, config):
        self.ln_1 = tz.nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)  # Q/K/V via BitLinear
        self.ln_2 = tz.nn.LayerNorm(config.n_embd)
        self.mlp = tz.nn.Sequential(
            tz.nn.BitLinear(config.n_embd, 4 * config.n_embd),  # ternary
            tz.nn.GELU(),
            tz.nn.BitLinear(4 * config.n_embd, config.n_embd),  # ternary
        )
```

**Design decision:** Attention Q/K/V projections use `BitLinear` (ternary-quantized) while the output projection and embedding layers remain full-precision `Linear`. This mirrors the BitNet architecture where attention projections are more robust to quantization than output projections.

### 4.5 Validation Protocol

#### 4.5.1 Perplexity Degradation

```python
def evaluate_perplexity_degradation():
    model_fp16 = load_gpt2_fp16("gpt2-small")
    model_ternary = quantize_to_ternary(model_fp16, alpha=0.5)

    ppl_fp16 = compute_perplexity(model_fp16, "wikitext-2-raw-v1")
    ppl_ternary = compute_perplexity(model_ternary, "wikitext-2-raw-v1")

    delta_ppl = ppl_ternary - ppl_fp16
    assert delta_ppl < 0.5, f"Perplexity degradation {delta_ppl} > 0.5"
```

#### 4.5.2 Tokens/sec Measurement

```python
def measure_generation_throughput(model, prompt_tokens, max_new_tokens=128):
    synchronize_gpu()
    t_start = high_resolution_timer()

    generated = []
    next_token = prompt_tokens
    for _ in range(max_new_tokens):
        logits = model(next_token)          # GEMV per layer
        next_token = sample(logits[:, -1])
        generated.append(next_token)

    synchronize_gpu()
    t_end = high_resolution_timer()

    tokens_per_sec = max_new_tokens / (t_end - t_start)
    ttft = time_to_first_generated_token  # measured separately
    return {"tokens_per_sec": tokens_per_sec, "ttft_s": ttft}
```

### 4.6 Deliverables

| Deliverable | Format | Location |
|---|---|---|
| GPT-2 model definition | Python | `examples/ternary_gpt2.py` |
| Weight quantization script | Python | `scripts/quantize_gpt2.py` |
| Inference benchmark script | Python | `scripts/benchmark_transformer.py` |
| Perplexity evaluation script | Python | `scripts/evaluate_perplexity.py` |
| Results tables | CSV + JSON | `experiments/aggregate/transformer_validation.csv` |
| Perplexity comparison | CSV | `experiments/aggregate/perplexity_comparison.csv` |

### 4.7 Success Metrics

| Metric | Target | Measurement |
|---|---|---|
| Tokens/sec vs FP16 baseline | > 2× speedup | Ratio of generation rates |
| Time-to-first-token | < 80% of FP16 baseline | Measured TTFT ratio |
| VRAM footprint | < 25% of FP16 baseline | `nvidia-smi` peak memory |
| Perplexity degradation | ΔPPL < 0.5 on WikiText-2 | Perplexity difference |
| Context window scaling | > 3× FP16 at fixed VRAM | Max context length comparison |
| End-to-end correctness | Generated text is coherent (qualitative) | Manual inspection |
| GPT-2 Small functional | Inference produces valid logits | Non-NaN, non-Inf output |
| GPT-2 Medium functional | Inference produces valid logits | Non-NaN, non-Inf output |

### 4.8 Strategic Justification

Transformer validation transforms the project from "a fast kernel" to "a deployable inference system." It addresses the critical question: **does the 8× memory reduction and theoretical bandwidth advantage translate to real-world inference speedup?**

Specific research contributions enabled:
1. **Quantitative evidence** that ternary quantization is viable for transformer inference (not just GEMV microbenchmarks)
2. **Perplexity-throughput Pareto frontier** — the tradeoff between model quality and inference speed
3. **Context window scaling analysis** — demonstrating that memory savings enable longer sequences, which is a unique selling point vs INT8/FP8 approaches
4. **End-to-end integration proof** — validating the full Rust→CUDA→Python pipeline under realistic workload conditions

---

## 5. Workstream P5: Architectural Portability & Abstraction

### 5.1 Problem Statement

The project is locked to NVIDIA Ada Lovelace (sm_89) via three independent mechanisms:
1. `build.rs` hardcodes `--gpu-architecture=sm_89` (line: `nvcc_args.push("--gpu-architecture=sm_89")`)
2. `kernel/ptx_utils.h` contains zero `#ifdef __CUDA_ARCH__` guards — all PTX inline assembly is unconditional
3. No runtime SM version detection exists anywhere in the Rust or CUDA codebase

This means the kernel **will not compile** on any GPU other than RTX 40-series, and will **crash at runtime** if executed on hardware where the PTX instructions behave differently. This is classified as Risk R-004 in ARCHITECTURE_GOVERNANCE.md: "HIGH probability, HIGH impact."

### 5.2 Technical Objectives

| ID | Objective | Scope |
|---|---|---|
| P5-O1 | **Architecture-conditional compilation** | Add `#ifdef __CUDA_ARCH__` guards in `ptx_utils.h` with C++ bitwise fallback paths for all PTX macros |
| P5-O2 | **Multi-architecture fat binary** | Modify `build.rs` to emit multiple `-gencode` targets: sm_70 (Volta), sm_80 (A100), sm_86 (RTX 30xx), sm_89 (RTX 40xx), sm_90 (H100) |
| P5-O3 | **Runtime SM detection** | Add `cudaDeviceGetAttribute` calls in the Rust FFI layer to query SM version; select optimal kernel variant at runtime |
| P5-O4 | **Graceful CPU fallback** | If no CUDA device is present, the Python framework must function using CPU-only reference GEMV (already partially implemented via `ndarray`) |
| P5-O5 | **FP32 accumulation mode** | Add compile-time or runtime switch for `float` accumulation in the block-level reduction (required for N > 8192 per METHODOLOGY.md §1.5) |
| P5-O6 | **Multi-arch CI validation** | Extend P1 CI to build and test against at least two architecture targets (sm_70 for minimum supported, sm_89 for primary) |
| P5-O7 | **Portable performance validation** | Verify that the C++ bitwise fallback produces identical results to the PTX path for at least one non-sm_89 architecture |

### 5.3 PTX Fallback Specification

#### 5.3.1 Current PTX Macros (from `ptx_utils.h`)

| Macro | PTX Instruction | C++ Bitwise Fallback |
|---|---|---|
| `PTX_PRMT(a, b, c)` | `prmt.b32` | `_byteswap_ulong` or manual byte permutation via shift/mask |
| `PTX_LOP3_LUT(a, b, c, lut)` | `lop3.b32` | Manual 3-input logic: `((a & b) ^ c)` or truth-table-indexed |
| `PTX_BFE(src, pos, len)` | `bfe.u32` | `(src >> pos) & ((1u << len) - 1u)` |
| `PTX_BFI(src, dst, pos, len)` | `bfi.b32` | Manual bit field insert via mask and OR |
| `PTX_SHL(src, count)` | `shl.b32` | `src << count` (native C++) |
| `PTX_SHR(src, count)` | `shr.b32` | `src >> count` (native C++) |

#### 5.3.2 Proposed `portable_utils.h` Structure

```cpp
// kernel/portable_utils.h
#pragma once

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    // SM 7.0+ (Volta and newer): all PTX intrinsics available
    #define TERNARY_USE_PTX 1
#else
    // Fallback: C++ bitwise operations
    #define TERNARY_USE_PTX 0
#endif

#if TERNARY_USE_PTX
    #define TERNARY_BFE(src, pos, len) PTX_BFE(src, pos, len)
    #define TERNARY_PRMT(a, b, c)     PTX_PRMT(a, b, c)
    #define TERNARY_LOP3(a, b, c, lut) PTX_LOP3_LUT(a, b, c, lut)
#else
    __device__ __forceinline__ uint32_t ternary_bfe(uint32_t src, uint32_t pos, uint32_t len) {
        return (src >> pos) & ((1u << len) - 1u);
    }
    __device__ __forceinline__ uint32_t ternary_prmt(uint32_t a, uint32_t b, uint32_t c) {
        // Simplified: extract byte selected by c from a|b
        uint32_t idx = c & 0x7;
        uint32_t combined = (b << 16) | (a & 0xFFFF);
        return (combined >> (idx * 4)) & 0xF;
    }
    __device__ __forceinline__ uint32_t ternary_lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t lut) {
        uint32_t result = 0;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            uint32_t idx = ((a >> i) & 1) | (((b >> i) & 1) << 1) | (((c >> i) & 1) << 2);
            result |= ((lut >> idx) & 1) << i;
        }
        return result;
    }
    #define TERNARY_BFE(src, pos, len)  ternary_bfe(src, pos, len)
    #define TERNARY_PRMT(a, b, c)      ternary_prmt(a, b, c)
    #define TERNARY_LOP3(a, b, c, lut) ternary_lop3(a, b, c, lut)
#endif
```

**Performance note:** The C++ `ternary_lop3` fallback is O(32) per call vs O(1) for the hardware instruction. On sm_70+ hardware, the PTX path is used. On older hardware, the fallback ensures **correctness at the cost of performance** — this is explicitly acceptable per the portability requirement ("cross-platform compatibility without sacrificing peak performance on target hardware").

### 5.4 Multi-Architecture Build Specification

#### 5.4.1 `build.rs` Modifications

```rust
// Current (hardcoded):
nvcc_args.push("--gpu-architecture=sm_89");

// Proposed (configurable):
let architectures = vec![
    "sm_70",  // Volta (V100, GTX 16xx)
    "sm_75",  // Turing (RTX 20xx, T4)
    "sm_80",  // Ampere (A100, A30)
    "sm_86",  // Ampere (RTX 30xx)
    "sm_89",  // Ada Lovelace (RTX 40xx) — PRIMARY TARGET
    "sm_90",  // Hopper (H100, H200)
];

for arch in &architectures {
    nvcc_args.push("-gencode");
    nvcc_args.push(&format!("arch=compute_{0},code=sm_{0}", arch.trim_start_matches("sm_")));
}
```

#### 5.4.2 Build Time Impact

| Configuration | Compile Time (est.) | Binary Size (est.) |
|---|---|---|
| sm_89 only (current) | ~15s | ~200 KB |
| 6 architectures | ~45-60s | ~800 KB |
| 6 architectures + LTO | ~90s | ~500 KB |

The ~3× build time increase is acceptable for a release build. CI can use single-architecture for fast iteration.

### 5.5 Deliverables

| Deliverable | Format | Location |
|---|---|---|
| Portable PTX utilities header | C++ | `kernel/portable_utils.h` |
| Updated CUDA kernel | C++ | `kernel/ternary_zero.cu` (modified to use `TERNARY_*` macros) |
| Multi-arch build script | Rust | `build.rs` (modified) |
| Runtime SM detection | Rust | `src/ffi.rs` (new function: `get_sm_version()`) |
| FP32 accumulation kernel variant | C++ | `kernel/ternary_zero.cu` (new `#ifdef` branch) |
| Portability test suite | Python + Rust | `tests/test_portability.py`, `tests/test_fallback.rs` |
| Architecture support matrix | Markdown | `docs/dev/ARCHITECTURE_SUPPORT.md` |

### 5.6 Success Metrics

| Metric | Target | Measurement |
|---|---|---|
| Compilation success | sm_70, sm_75, sm_80, sm_86, sm_89, sm_90 | All `-gencode` targets compile without error |
| Fallback correctness | Bitwise-identical output for PTX vs C++ paths | Element-wise comparison on 1000 random inputs |
| Runtime detection | Correct SM version reported | `cudaDeviceGetAttribute` returns expected value |
| FP32 accumulation accuracy | Max error < 10⁻⁴ for N=16384 | CPU reference comparison |
| Build time impact | < 3× current build time | CI wall time comparison |
| Binary size impact | < 5× current binary size | File size comparison |
| CI multi-arch validation | At least sm_70 + sm_89 tested in CI | CI matrix job configuration |

### 5.7 Strategic Justification

Portability is the **final barrier to publication credibility**. A submission that only works on one GPU architecture will be flagged by reviewers as a "hardware-specific hack" rather than a "general-purpose contribution." Specifically:

1. **sm_70 (Volta) support** enables validation on V100 — the most common cloud GPU for research
2. **sm_80 (Ampere) support** enables validation on A100 — the standard datacenter GPU
3. **Runtime fallback** demonstrates engineering maturity and enables the "not architecture-portable" limitation (METHODOLOGY.md §5.2) to be removed from the scope declaration
4. **FP32 accumulation** removes the N>8192 precision limitation, expanding the valid operating range

The portability work also de-risks P4 (Transformer validation) by enabling testing on any available GPU, not just the specific RTX 4060 development machine.

---

## 6. Workstream P6: 75B Model Execution on 8GB VRAM (NEW)

### 6.1 Problem Statement

The RTX 4060 has 8 GB VRAM. A 75B parameter model in ternary (W2A16) format requires ~18.75 GB for weights alone — 2.3× the VRAM budget. Standard in-memory inference is physically impossible. A new execution strategy is required: **per-layer CPU-GPU weight streaming** with double-buffered async transfers, overlapped compute, and aggressive KV-cache management.

### 6.2 Technical Objectives

| ID | Objective | Scope |
|---|---|---|
| P6-O1 | **Per-layer weight streaming API** | Rust `StreamingWeights` struct with double-buffered GpuBuffer and async H2D transfers |
| P6-O2 | **CPU-side weight management** | Quantized weights stored in pinned host memory; per-layer pack-once, stream-many |
| P6-O3 | **Overlapped pipeline** | Prefetch layer K+1's weights while executing layer K's GEMV on stream 0 |
| P6-O4 | **KV-cache memory reservation** | Pre-allocate full KV-cache in VRAM (640 MB at S=2048); never evict |
| P6-O5 | **Embedding/LM-head residency** | Keep embedding (250 MB) and LM-head (250 MB) in VRAM permanently |
| P6-O6 | **PCIe bandwidth measurement** | Measure actual H2D transfer rate; report utilization vs theoretical peak |
| P6-O7 | **End-to-end 75B inference** | Generate text from a 75B ternary model on RTX 4060; measure tokens/sec |

### 6.3 Memory Budget

| Component | Size | VRAM Residency |
|-----------|------|----------------|
| Active layer weights (buffer A) | 206 MB | Streaming (swap per layer) |
| Active layer weights (buffer B) | 206 MB | Streaming (double-buffer) |
| KV-cache (80 layers, S=2048) | 640 MB | Permanent |
| Embedding | 250 MB | Permanent |
| LM Head (ternary) | 250 MB | Permanent |
| Activation buffers | 44 MB | Permanent |
| CUDA runtime overhead | ~300 MB | Permanent |
| **TOTAL** | **~1,896 MB** | **Fits in 8 GB (6.1 GB headroom)** |

### 6.4 Execution Pipeline (Per Token)

```
For each generated token:
  1. Embed token_id → hidden state [8192] (VRAM lookup, no transfer)
  2. For layer_idx = 0..79:
     a. If layer_idx == 0: H2D transfer layer 0 weights (blocking first token)
     b. If layer_idx + 1 < 80: async H2D transfer layer K+1 weights into inactive buffer
     c. Execute layer K GEMV operations (Q/K/V/O projections + FFN)
     d. Swap active/inactive weight buffers
  3. Final RMSNorm (VRAM, no transfer)
  4. LM Head GEMV (ternary, VRAM-resident)
  5. Sample next token
```

### 6.5 Throughput Estimates

| PCIe Version | Bandwidth | Transfer/Layer | Total Transfer/Token | GEMV Time/Token | Total/Token | Tokens/sec |
|-------------|-----------|----------------|---------------------|-----------------|-------------|------------|
| PCIe 3.0 x16 | 16 GB/s | 12.9 ms | 1,031 ms | ~160 ms | ~1,191 ms | **0.84** |
| PCIe 4.0 x16 | 32 GB/s | 6.4 ms | 515 ms | ~160 ms | ~675 ms | **1.48** |
| PCIe 5.0 x16 | 64 GB/s | 3.2 ms | 258 ms | ~160 ms | ~418 ms | **2.39** |
| With double-buffer (PCIe 4.0) | 32 GB/s | ~3.8 ms effective | ~300 ms | ~160 ms | ~460 ms | **2.17** |

### 6.6 Deliverables

| Deliverable | Format | Location |
|---|---|---|
| Streaming weight manager | Rust | `src/streaming.rs` |
| CPU weight loader | Rust | `src/host_weights.rs` |
| Python streaming integration | Python | `ternary_zero/inference/streaming_engine.py` |
| 75B inference test | Python | `tests/test_75b_streaming.py` |
| PCIe bandwidth benchmark | Python | `benchmarks/pcie_bandwidth.py` |
| Memory budget calculator | Python | `scripts/memory_calculator.py` |

### 6.7 Success Metrics

| Metric | Target | Measurement |
|---|---|---|
| 75B model loads without OOM | Zero CUDA OOM errors | `cudaMalloc` returns success for all allocations |
| Text generation produces coherent output | Non-NaN logits, valid tokens | Manual inspection |
| PCIe utilization | > 70% of theoretical bandwidth | Measured H2D throughput |
| Decode throughput (PCIe 4.0) | > 1.0 tokens/sec | Timer measurement |
| KV-cache fits in VRAM | 640 MB at S=2048 | `nvidia-smi` peak memory |
| Double-buffer speedup | > 1.3× vs sequential | A/B comparison |

---

## 7. Dependency Graph & Execution Sequencing

### 7.1 Updated Dependency Graph (with P6)

```
                    ┌─────────────────────────────────────────────────┐
                    │              WORKSTREAM DEPENDENCY GRAPH         │
                    └─────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │  P1: CI/CD        │
                         │  Infrastructure   │
                         │  ─────────────    │
                         │  Blocker for all  │
                         │  downstream       │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼                           ▼
         ┌──────────────────┐         ┌──────────────────┐
         │  P2: Empirical   │         │  P5: Portability  │
         │  Benchmarking    │         │  & Abstraction    │
         │  Suite           │         │  ─────────────    │
         │  ─────────────    │         │  Can proceed in   │
         │  Produces data   │         │  parallel with P2 │
         │  for P3 and P4   │         │  (independent)    │
         └────────┬─────────┘         └────────┬─────────┘
                  │                              │
         ┌────────┴────────┐                    │
         ▼                 ▼                    │
┌──────────────────┐ ┌──────────────────┐       │
│  P3: Roofline    │ │  P4: Transformer │       │
│  Model           │ │  Validation      │       │
│  Integration     │ │  ─────────────    │       │
│  ─────────────    │ │  Consumes P2     │       │
│  Consumes P2     │ │  data + P5       │◄──────┘
│  data            │ │  portability     │  (P4 benefits
│                  │ │                  │   from P5 but
└──────────────────┘ └──────────────────┘   is not blocked)
```

### 6.1 Recommended Execution Order

| Phase | Workstream(s) | Duration (est.) | Gate |
|---|---|---|---|
| **Phase 1** | P1 (CI/CD) | 1–2 weeks | All tests green in CI |
| **Phase 2** | P2 (Benchmarking) + P5 (Portability) | 3–4 weeks | Shape matrix data exists; multi-arch compiles |
| **Phase 3** | P3 (Roofline) + P4 (Transformer) | 2–3 weeks | Roofline plots generated; GPT-2 inference functional |
| **Phase 4** | Integration & Publication Prep | 1–2 weeks | All metrics meet targets; paper draft complete |

**Total estimated duration: 7–11 weeks** (assuming 1 full-time engineer with GPU access)

### 6.2 Parallel Execution Opportunities

- **P2 and P5** are independent and can proceed simultaneously
- **P3 and P4** both consume P2 data but are independent of each other
- **P5 portability** can be validated independently of P2 benchmarking (correctness tests don't require performance data)

---

## 7. Cross-Cutting Concerns

### 7.1 ADR Compliance Matrix

| ADR Requirement | Workstream(s) | Compliance Status |
|---|---|---|
| ADR-001: `experiments/` directory structure | P2, P3, P4 | **Not implemented** — must be created in P1 |
| ADR-001: `manifest.json` schema | P2 | **Schema defined** — must be enforced by P1 CI |
| ADR-001: Environment capture script | P2 | **Pseudocode defined** — must be implemented in P2 |
| ADR-001: Git LFS for binary traces | P2, P3 | **Not configured** — must be set up in P1 |
| ADR-002: Documentation migration | All | **Not started** — lower priority than data generation |
| ADR-003: Toolchain justification | P5 | **Completed** — documented in ARCHITECTURE_GOVERNANCE.md |
| ADR-004: Scope boundary | All | **ACTIVE** — P4 (Transformer) must not expand into training framework |

### 7.2 Known Technical Debt Addressed by This Plan

| Debt Item (from METHODOLOGY.md §5.3) | Workstream | Resolution |
|---|---|---|
| PTX intrinsics: no `#ifdef __CUDA_ARCH__` fallback | P5 | `portable_utils.h` with C++ bitwise path |
| Autograd engine: recursive `_build_topo` | — | **Not addressed** — out of scope (R-001, mitigated by limiting graph depth) |
| Tensor `_version`: never checked | — | **Not addressed** — out of scope (R-002, mitigated by ADR-004 §4.7 specification) |
| FP16 accumulation: precision loss for N > 8192 | P5 | FP32 accumulation mode |
| Activation upload: allocated per-call | — | **Not addressed** — low priority (R-006 category) |
| Optimizer steps: pure NumPy CPU | — | **Not addressed** — out of scope per ADR-004 |

### 7.3 Publication Readiness Criteria

A manuscript is ready for submission when ALL of the following are true:

| Criterion | Source Workstream | Evidence |
|---|---|---|
| Formal problem statement with mathematical grounding | Pre-existing (METHODOLOGY.md §1) | LaTeX equations, encoding table |
| Empirical latency/throughput data across full shape matrix | P2 | 80+ data points with statistical rigor |
| Speedup vs cuBLAS FP16 and INT8 baselines | P2 | Measured on same hardware, same session |
| Roofline plot demonstrating memory-boundedness | P3 | Plot with data points on memory ceiling |
| Nsight Compute profiling evidence | P2 | Occupancy, bandwidth utilization, stall analysis |
| Transformer inference validation | P4 | Tokens/sec, perplexity, VRAM metrics |
| Architectural portability | P5 | At least 2 architectures validated |
| Reproducibility package | P1 + P2 | `experiments/` directory, `manifest.json`, environment snapshot |
| Ablation study (sparsity, threshold α, accumulator precision) | P2 + P3 | Sweep data across ρ₀ and α values |

---

## Appendix A: Experiment Configuration Templates

### A.1 Ternary Dense Configuration

```toml
# experiments/configs/ternary_dense.toml
[experiment]
name = "ternary_dense"
description = "Ternary GEMV kernel, zero sparsity (baseline)"
version = "1.0.0"

[kernel]
type = "ternary_gemv"
sparsity = 0.0
accumulation = "half2"
l2_persist_policy = true

[shape_matrix]
M = [1, 4, 16, 64, 256]
N = [1024, 2048, 4096, 8192]

[measurement]
warmup_iterations = 100
measurement_iterations = 1000
sync_mode = "cudaEventRecord"
clock_lock = false
power_management = "prefer_maximum_performance"

[baselines]
cublas_fp16 = true
cublas_int8 = true
```

### A.2 Sparsity Sweep Configuration

```toml
# experiments/configs/ternary_sparse_sweep.toml
[experiment]
name = "ternary_sparse_sweep"
description = "Sparsity-to-performance sweep"
version = "1.0.0"

[kernel]
type = "ternary_gemv"
sparsity = [0.0, 0.25, 0.50, 0.75]
accumulation = "half2"
l2_persist_policy = true

[shape_matrix]
M = [1, 4, 16]
N = [2048, 4096, 8192]

[measurement]
warmup_iterations = 100
measurement_iterations = 1000
sync_mode = "cudaEventRecord"

[analysis]
generate_sparsity_curve = true
fit_model = "linear"  # latency = a + b * (1 - sparsity)
```

---

## Appendix B: Acceptance Test Checklist

### B.1 P1 (CI/CD) Acceptance

- [ ] GitHub Actions workflow runs on every PR
- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `ruff check` + `ruff format --check` passes
- [ ] `cargo test` passes (6 Rust unit tests)
- [ ] `pytest tests/` passes (36 Python tests)
- [ ] CUDA compilation smoke test passes (self-hosted runner)
- [ ] ADR-001 schema file exists and is valid JSON Schema
- [ ] Git LFS configured for binary trace files

### B.2 P2 (Benchmarking) Acceptance

- [ ] GPU benchmark harness produces latency measurements
- [ ] 80/80 shape matrix data points collected
- [ ] 1000 iterations per data point
- [ ] cuBLAS FP16 baseline measured
- [ ] cuBLAS INT8 baseline measured
- [ ] All results in `experiments/runs/` with valid `manifest.json`
- [ ] Latency table generated (CSV)
- [ ] Throughput table generated (CSV)
- [ ] Speedup table generated (CSV)
- [ ] At least 12 Nsight Compute traces captured
- [ ] Statistical reporting includes mean, median, p50, p99, std

### B.3 P3 (Roofline) Acceptance

- [ ] Operational intensity computed for all M×N shapes
- [ ] Roofline plot generated with compute and memory ceilings
- [ ] All data points plotted on Roofline
- [ ] Bandwidth utilization table generated
- [ ] Nsight occupancy data collected and reported
- [ ] Memory-boundedness confirmed (all points on memory ceiling slope)
- [ ] Sparsity effect on Roofline analyzed

### B.4 P4 (Transformer) Acceptance

- [ ] GPT-2 Small model defined using `ternary_zero.nn` primitives
- [ ] GPT-2 Medium model defined using `ternary_zero.nn` primitives
- [ ] Weight quantization produces valid ternary weights
- [ ] Inference produces non-NaN, non-Inf logits
- [ ] Tokens/sec measured and reported
- [ ] TTFT measured and reported
- [ ] VRAM footprint measured for FP16 and ternary
- [ ] Perplexity computed for FP16 and ternary on WikiText-2
- [ ] ΔPPL < 0.5
- [ ] Context window scaling demonstrated (> 3× at fixed VRAM)

### B.5 P5 (Portability) Acceptance

- [ ] `portable_utils.h` created with C++ bitwise fallbacks
- [ ] All PTX macros replaced with `TERNARY_*` portable macros
- [ ] `build.rs` emits `-gencode` for sm_70, sm_75, sm_80, sm_86, sm_89, sm_90
- [ ] Runtime SM detection implemented
- [ ] FP32 accumulation mode functional
- [ ] Bitwise-identical results between PTX and C++ fallback paths
- [ ] CI validates at least sm_70 + sm_89 builds
- [ ] Architecture support matrix documented

---

## 8. Workstream P7: Version 1.0 Milestones (Model Patcher + Shape Matrix + PCIe Streaming)

### 8.1 Problem Statement

The project requires three critical engineering components to transition from a research prototype to an empirically validated inference system: (1) a HuggingFace model patcher for streaming ternary quantization, (2) an automated 80-point shape matrix benchmark suite, and (3) a double-buffered PCIe streaming engine for ultra-large-model inference. See [ROADMAP.md](ROADMAP.md) for the complete specification.

### 8.2 Technical Objectives

| ID | Objective | Scope |
|---|---|---|
| P7-O1 | **HuggingFace Model Patcher** | Streaming safetensors reader, chunked quantization, OOM-safe conversion, `patch_manifest.json` output |
| P7-O2 | **Shape Matrix Benchmark Suite** | 80-point $M \times N$ sweep, `cudaEvent` timing, `manifest.json` with aggregate statistics |
| P7-O3 | **Double-Buffered PCIe Streaming** | Async layer loader thread, double-buffer slot management, `StreamingProfile` per-layer metrics |
| P7-O4 | **Deep Optimization Specifications** | `cp.async` staging, SIMD-style bit decode, L2 persistence tuning, KV-cache quantization design |

### 8.3 Deliverables

| Deliverable | Format | Location | Status |
|---|---|---|---|
| Model patcher | Python | `ternary_zero/inference/model_patcher.py` | **Implemented** |
| Shape matrix benchmark | Python | `benchmarks/shape_matrix_benchmark.py` | **Implemented** |
| Streaming engine | Python | `ternary_zero/inference/streaming_engine.py` | **Implemented** |
| Deep optimization spec | Markdown | `ARCHITECTURE.md` §9 | **Documented** |
| Roadmap document | Markdown | `ROADMAP.md` | **Documented** |

### 8.4 Success Metrics

| Metric | Target | Measurement |
|---|---|---|
| Model patcher: Llama-3.2-1B conversion | Complete without OOM | `patch_manifest.json` generated |
| Model patcher: Peak host RAM | < 4 GB during 1B model conversion | `tracemalloc` |
| Shape matrix: Configuration coverage | 80/80 data points | `manifest.successful_configs == 80` |
| Shape matrix: Execution time | < 10 minutes for full sweep | `manifest.total_time_s` |
| Streaming: Layer descriptor correctness | All Llama projections generated | `build_llama_streaming_engine()` test |
| Streaming: Async loader thread safety | Zero deadlocks | 1000-iteration stress test |
| Deep optimization: `cp.async` spec | Complete implementation specification | `ARCHITECTURE.md` §9.1 |
| Deep optimization: KV-cache quantization | INT8 design with error bounds | `ARCHITECTURE.md` §9.4 |

### 8.5 Strategic Justification

P7 is the **empirical validation layer** that converts theoretical claims into measured data:

1. **M1 (Model Patcher)** enables end-to-end model conversion from HuggingFace → ternary format, which M2 and M3 consume.
2. **M2 (Shape Matrix)** produces the 80-point latency/throughput dataset required for peer-reviewed publication.
3. **M3 (PCIe Streaming)** enables 70B+ model execution on consumer hardware, demonstrating the system's unique value proposition.
4. **Deep Optimization** specifies the kernel-level improvements that close the gap between achieved and theoretical bandwidth.

---

*This document is binding. All implementation work MUST reference the workstream IDs (P1–P7), objective IDs (Pn-ON), and acceptance criteria defined herein. Deviations require explicit override via a superseding ADR or amendment to this plan.*
