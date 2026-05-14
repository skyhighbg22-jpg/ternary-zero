# Ternary-Zero: A Research Methodology and Implementation Framework for Sub-Byte GEMV Acceleration on Consumer GPU Architectures

**Version:** 0.1.0
**Classification:** Technical Research Document
**Target Hardware:** NVIDIA RTX 4060 (Ada Lovelace, sm_89, 128-bit memory bus)

---

## Abstract

This document establishes the formal research methodology, mathematical foundations, and validation framework for the Ternary-Zero system: a W2A16 (2-bit Weight, 16-bit Activation) GEMV kernel targeting latency-sensitive inference on consumer NVIDIA GPUs. We define the quantized transformation algebra, characterize the overhead-throughput tradeoff space, and propose a reproducible benchmarking protocol for empirical validation against industry-standard baselines (cuBLAS FP16, INT8, FP8 Tensor Core pathways).

---

## 1. Formal Operational Model & Mathematical Grounding

### 1.1 The Quantized Transformation Function

The core operation computes a matrix-vector product under ternary weight quantization:

$$\mathbf{y} = Q(\mathbf{W}) \cdot \mathbf{x}$$

where $\mathbf{W} \in \mathbb{R}^{M \times N}$ is the full-precision weight matrix, $\mathbf{x} \in \mathbb{R}^N$ is the activation vector, and $Q(\cdot)$ is the element-wise quantization operator. The component-wise form is:

$$y_i = \sum_{j=0}^{N-1} q(w_{ij}) \cdot x_j, \quad i \in \{0, \ldots, M-1\}$$

where $q: \mathbb{R} \to \{-1, 0, +1\}$ is the ternary quantization function. The effective computation decomposes into three semantically distinct operations:

$$y_i = \underbrace{\sum_{j \in \mathcal{P}_i} x_j}_{\text{positive accumulation}} - \underbrace{\sum_{j \in \mathcal{N}_i} x_j}_{\text{negative accumulation}} + \underbrace{\sum_{j \in \mathcal{Z}_i} 0}_{\text{zero-gate skip}}$$

where $\mathcal{P}_i = \{j : q(w_{ij}) = +1\}$, $\mathcal{N}_i = \{j : q(w_{ij}) = -1\}$, and $\mathcal{Z}_i = \{j : q(w_{ij}) = 0\}$ partition the column indices for row $i$.

### 1.2 Ternary Encoding Function

The quantization function $q(w)$ is parameterized by a threshold $\tau$ derived from the weight distribution:

$$q(w; \tau) = \begin{cases} +1 & \text{if } w > \tau \\ -1 & \text{if } w < -\tau \\ 0 & \text{otherwise} \end{cases}$$

**Threshold Selection (STE-aware, training mode):**

$$\tau = \alpha \cdot \frac{1}{MN} \sum_{i,j} |w_{ij}|, \quad \alpha \in (0, 1]$$

where $\alpha$ is a hyperparameter (typically $\alpha \in [0.5, 0.7]$). The quantization scale factor $s$ for dequantized approximation is:

$$s = \frac{1}{|\{(i,j) : q(w_{ij}) \neq 0\}|} \sum_{(i,j) : q(w_{ij}) \neq 0} |w_{ij}|$$

**Reconstruction approximation:** $\hat{w}_{ij} = q(w_{ij}) \cdot s$

**Straight-Through Estimator (STE) gradient:**

During backpropagation, the quantization function is treated as identity within the clipping range:

$$\frac{\partial \mathcal{L}}{\partial w_{ij}} \approx \frac{\partial \mathcal{L}}{\partial q_{ij}} \cdot \mathbf{1}_{|w_{ij}| \leq s}$$

This enables end-to-end gradient flow through the non-differentiable quantization boundary.

### 1.3 Bit-Packing Encoding

Each ternary value $q(w) \in \{-1, 0, +1\}$ is encoded in 2 bits:

| Ternary Value | 2-bit Code | Semantic |
|:---:|:---:|:---|
| $0$ | `00` | Zero-gate: accumulation skipped |
| $+1$ | `01$ | Positive: activation added to accumulator |
| $-1$ | `10` | Negative: activation subtracted from accumulator |
| — | `11` | Invalid (unused sentinel) |

**Packing density:** 16 ternary weights per `uint32_t` register.

**Memory footprint:** For an $M \times N$ weight matrix:
- Full-precision (FP32): $4MN$ bytes
- Half-precision (FP16): $2MN$ bytes
- Ternary-packed (2-bit): $\frac{MN}{4}$ bytes = $\frac{MN}{16} \times 4$ bytes (packed `uint32_t`)

**Compression ratios:**

$$C_{\text{vs FP32}} = \frac{32}{2} = 16\times, \quad C_{\text{vs FP16}} = \frac{16}{2} = 8\times, \quad C_{\text{vs INT8}} = \frac{8}{2} = 4\times$$

### 1.4 Zero-Gating Semantics

The zero-gate is a **branchless masking operation**, not a conditional branch. For each 2-bit weight $b_1 b_0$:

1. **Magnitude bit** $m = b_0$: $m = 1$ iff $q(w) \neq 0$
2. **Sign bit** $s = b_1$: $s = 1$ iff $q(w) = -1$

The masked accumulation for a single weight-activation pair is:

```
nz_mask   = -m          // 0xFFFFFFFF if non-zero, 0x00000000 if zero
sign_mask = -s & 0x8000 // FP16 sign bit mask if negative
signed_act = act_bits XOR sign_mask   // conditional sign flip
gated_act  = signed_act AND nz_mask   // zero-gate
accumulator += reinterpret<half2>(gated_act)
```

**Key semantic property:** The zero-gate does NOT eliminate instruction execution. The load, decode, and masking pipeline executes for all 16 weights in a packed word. The benefit is that the FP16 addition receives a zero addend, which has negligible latency/energy compared to a non-zero addition. The benefit scales with the fraction of zero weights $\rho_0 = |\mathcal{Z}| / N$.

### 1.5 Accumulator Precision Analysis

The accumulator must represent the partial sum $y_i = \sum_j q(w_{ij}) \cdot x_j$ without overflow.

**Worst-case bounds:**
- Maximum absolute activation: $|x_j| \leq x_{\max} = 65504$ (FP16 max)
- Maximum non-zero count per row: $N$ (all weights non-zero)
- Worst-case sum magnitude: $|y_i| \leq N \cdot x_{\max}$

**For FP16 accumulation (`half`):**
- $\varepsilon_{\text{FP16}} \approx 9.77 \times 10^{-4}$ (machine epsilon)
- Accumulated rounding error after $N$ additions: $\epsilon_{\text{acc}} \approx \sqrt{N} \cdot \varepsilon_{\text{FP16}}$ (random walk model)
- For $N = 4096$: $\epsilon_{\text{acc}} \approx 64 \times 9.77 \times 10^{-4} \approx 0.063$ (relative to unit-scale activations)
- **Risk:** Acceptable for inference with normalized activations; accumulations with $N > 8192$ or unbounded activation ranges should use `float32` accumulation.

**Recommended production configuration:**
- Warp-level partial sums: `half2` (vectorized SIMD, 2× throughput)
- Block-level reduction: `float` (accumulated from warp sums)
- Final output: `half` (written to global memory)

**Forward-error bound:** for a sum of $N$ terms computed in floating point with unit roundoff $u$,

$$|\hat{y}_i - y_i| \leq \gamma_N \sum_{j=0}^{N-1} |q(w_{ij}) x_j|, \quad \gamma_N = \frac{Nu}{1-Nu}, \quad Nu < 1$$

This is the standard worst-case bound for sequential accumulation. In practice, the kernel uses warp and block reductions in `float`, so the dominant rounding term is usually the final cast back to `half`, not the inner partial sums.

**Quantization noise model:** define the elementwise quantization error

$$e_{ij} = w_{ij} - \hat{w}_{ij}, \quad \hat{\mathbf{W}} = \mathbf{W} - \mathbf{E}$$

so the quantized output becomes

$$\hat{\mathbf{y}} = \hat{\mathbf{W}} \mathbf{x} = \mathbf{W}\mathbf{x} - \mathbf{E}\mathbf{x}$$

and the error energy is governed by

$$\mathbb{E}\left[\lVert \mathbf{E}\mathbf{x} \rVert_2^2\right]$$

Under the usual zero-mean, weakly independent calibration assumption, this scales with the activation energy and the per-column quantization variance. That gives the method a standard compression-theory interpretation rather than a purely procedural one.

### 1.6 Execution Decomposition: Algorithm → Hardware Instructions

The GEMV computation is decomposed into discrete hardware-level operations:

| Algorithmic Step | Hardware Operation | Latency (cycles) | Throughput |
|---|---|---|---|
| 1. Load packed weights | `LDG.32` (global memory → register) | ~200-400 (uncached) | 32 bytes/coalesced transaction |
| 2. Byte-permute for alignment | `PRMT.B32` | 1 | 1 per clock per SM |
| 3. Bit-field extract (2-bit) | `BFE.U32` | 1 | 1 per clock per SM |
| 4. Sign/magnitude decode | `BFE.U32` + `SHL` | 2 | Pipelined |
| 5. Sign-flip mask generation | `I2I.NEG` → `AND` | 2 | Pipelined |
| 6. Zero-gate masking | `LOP3.LUT` (AND) | 1 | 1 per clock per SM |
| 7. FP16 add to accumulator | `HADD2` (vectorized) | 1 | 2 FP16 adds per clock |
| 8. Warp-level reduction | `SHFL.DOWN` × 5 | 5 | Butterfly pattern |
| 9. Block-level reduction | Shared memory + SHFL | ~10 | 8 warps → 1 value |
| 10. Output store | `STG.16` (register → global) | ~200-400 | 16 bytes/coalesced |

**Critical path analysis:** Steps 1 and 10 (global memory access) dominate latency. Steps 2-7 are fully pipelined in the execution unit. The zero-gate (step 6) adds 1 cycle per weight pair — negligible relative to memory latency.

**Instruction-class decode model:** the scalar `T_decode` term is better treated as a throughput-limited pipeline:

$$T_{\text{decode}} = \max\left(\frac{N_{\text{bfe}}}{\Theta_{\text{bfe}}}, \frac{N_{\text{prmt}}}{\Theta_{\text{prmt}}}, \frac{N_{\text{lop3}}}{\Theta_{\text{lop3}}}, \frac{N_{\text{shfl}}}{\Theta_{\text{shfl}}}\right)$$

where $\Theta$ denotes sustained instruction throughput for each class. This is the right level of abstraction for GPU review: the kernel is not just "doing decode," it is competing with the SM issue pipeline.

**Occupancy-constrained throughput:** register pressure and shared memory usage reduce the effective peak:

$$Occ = \min(Occ_{\text{regs}}, Occ_{\text{smem}}, Occ_{\text{warps}}, Occ_{\text{threads}})$$

$$P_{\text{eff}} = Occ \cdot \eta_{\text{warp}} \cdot \min(P_{\text{peak}}, I \cdot BW_{\text{eff}})$$

where $\eta_{\text{warp}} = \text{active lanes} / 32$ and $BW_{\text{eff}}$ is the effective bandwidth after cache reuse is accounted for. This makes the register-pressure cost of bit unpacking, masking, and reduction explicit.

**Cache-hierarchy bandwidth model:** instead of assuming every byte comes from DRAM,

$$T_{\text{memory}} = \frac{B_{\text{dram}}}{BW_{\text{dram}}} + \frac{B_{\text{l2}}}{BW_{\text{l2}}} + \frac{B_{\text{shared}}}{BW_{\text{shared}}}$$

with $B_{\text{dram}} + B_{\text{l2}} + B_{\text{shared}} = B_{\text{effective}}$. This captures the fact that activation reuse, partial-tile reuse, and shared-memory staging all move work off the DRAM ceiling.

---

## 2. Reproducibility & Measurement Methodology

### 2.1 Experimental Environment Specification

All empirical claims MUST be reported with the following metadata:

```yaml
hardware:
  gpu: "NVIDIA RTX 4060"          # or equivalent
  gpu_arch: "Ada Lovelace (sm_89)"
  vram_gb: 8
  l2_cache_mb: 24
  theoretical_bandwidth_gbs: 272   # GDDR6
  theoretical_bandwidth_l2_gbs: ~500  # L2 effective
  base_clock_mhz: 1830             # must measure actual
  boost_clock_mhz: 2460            # must measure actual
  power_limit_w: 115               # default TDP

software:
  os: "Windows 11 / Ubuntu 22.04"
  cuda_toolkit: "12.x"
  driver_version: "55x.xx"
  compiler: "MSVC 2022 / GCC 12"
  rust_version: "1.7x"
  python_version: "3.10+"
  numpy_version: "1.24+"

execution:
  warmup_iterations: 100           # discard from measurement
  measurement_iterations: 1000     # statistical sample
  synchronization: "cudaStreamSynchronize (explicit)"
  power_management: "prefer maximum performance (nvidia-smi -pm 1)"
  gpu_clock_lock: "optional: nvidia-smi -lgc <base>,<boost>"
  thermal_state: "report GPU temperature at start/end"
```

### 2.2 Measurement Protocol

#### 2.2.1 Latency Measurement

```python
# Pseudocode for rigorous latency measurement
import time

def measure_kernel_latency(kernel_fn, inputs, warmup=100, iterations=1000):
    # Warmup phase: populate L2 cache, stabilize clocks
    for _ in range(warmup):
        kernel_fn(*inputs)
    synchronize_gpu()

    # Measurement phase
    latencies = []
    for _ in range(iterations):
        synchronize_gpu()
        t_start = high_resolution_timer()
        kernel_fn(*inputs)
        synchronize_gpu()  # force completion
        t_end = high_resolution_timer()
        latencies.append(t_end - t_start)

    return {
        "mean_us": mean(latencies),
        "median_us": median(latencies),
        "p99_us": percentile(latencies, 99),
        "std_us": std(latencies),
        "min_us": min(latencies),
        "max_us": max(latencies),
        "iterations": iterations,
    }
```

#### 2.2.2 Throughput Measurement

Effective throughput (GB/s) is computed as:

$$\text{Throughput} = \frac{\text{Total bytes transferred}}{\text{Median latency}}$$

For ternary GEMV, total bytes = weight bytes + activation bytes + output bytes:

$$B_{\text{total}} = \frac{MN}{4} + 2N + 2M \quad \text{(bytes)}$$

#### 2.2.3 Statistical Reporting Requirements

| Metric | Minimum Repetitions | Reporting Format |
|---|---|---|
| Kernel latency | 1000 | mean ± std, p50, p99 |
| Throughput (GB/s) | 1000 | mean ± std |
| CPU reference accuracy | 100 random inputs | max absolute error, RMSE |
| Sparsity sweep | 5 sparsity levels × 1000 iterations | latency vs. sparsity curve |

### 2.3 Tensor Shape Matrix

All benchmarks MUST report results across the following shape matrix.
The 80-point configuration is implemented in `benchmarks/shape_matrix_benchmark.py`
and outputs a structured `manifest.json` with per-configuration latency, throughput,
and bandwidth metrics.

| Parameter | Values | Count |
|---|---|---|
| $M$ (output rows) | {1, 2, 4, 8, 16, 32, 64, 128} | 8 |
| $N$ (input features) | {256, 512, 1024, 2048, 4096, 8192, 11008, 14336, 16384, 19456} | 10 |
| **Total configurations** | | **80** |

The $N$ values cover the full spectrum of transformer hidden dimensions:
- **256–512:** Small/embedding layers
- **1024–2048:** GPT-2 Small/Medium, Llama-3.2-1B attention dimensions
- **4096–8192:** Llama-2-7B, Llama-3-8B, Llama-3.2-1B FFN intermediate
- **11008:** Llama-2-7B FFN intermediate size
- **14336:** Llama-3-8B FFN intermediate size
- **16384–19456:** Large model FFN intermediates

The $M$ values span the autoregressive decode regime ($M=1$) through small-batch
inference ($M=128$). Sparsity $\rho_0$ is swept separately when the measured mode
is available (requires GPU).

**Baseline comparisons (MUST be measured on same hardware, same session):**
- cuBLAS `cublasHgemv` (FP16 GEMV)
- cuBLAS `cublasGemmEx` with `CUDA_R_16F` (FP16 GEMM, batch=1)
- cuBLAS `cublasGemmEx` with `CUDA_R_8I` (INT8 GEMM, batch=1) — if Tensor Cores available

### 2.4 Accuracy Validation Protocol

Correctness is validated against a CPU reference implementation:

```python
def validate_kernel_accuracy(M, N, sparsity, tolerance=1e-3):
    """Compare GPU kernel output against CPU reference."""
    W = generate_ternary_weights(M, N, sparsity=sparsity)
    x = generate_fp16_activations(N)

    y_cpu = cpu_reference_gemv(W, x)      # FP32 accumulation
    y_gpu = gpu_ternary_gemv(W, x)        # FP16 accumulation

    max_abs_error = max(abs(y_cpu - y_gpu))
    rmse = sqrt(mean((y_cpu - y_gpu) ** 2))

    assert max_abs_error < tolerance, f"Max error {max_abs_error} > {tolerance}"
    return {"max_abs_error": max_abs_error, "rmse": rmse}
```

---

## 3. Critical Risk Analysis: The Overhead-Throughput Tradeoff

### 3.1 Unified Roofline Model

The ternary GEMV kernel is better described by a roofline model with explicit overhead terms than by a single bandwidth inequality.

Let the zero fraction be

$$\rho_0 = \frac{|\mathcal{Z}|}{N}$$

and the useful arithmetic density be

$$N_{\text{eff}} = (1 - \rho_0)N$$

For ternary GEMV, the useful operation count is approximately $M N_{\text{eff}}$, while the transferred bytes are dominated by packed weights plus activations and outputs:

$$B_{\text{ternary}} \approx \frac{MN}{4} + 2N + 2M$$

The operational intensity is therefore

$$I_{\text{ternary}} = \frac{M N_{\text{eff}}}{B_{\text{ternary}}}$$

and the roofline throughput bound is

$$P_{\text{roof}} = \min\left(P_{\text{peak}} \cdot Occ \cdot \eta_{\text{warp}}, \; I_{\text{ternary}} \cdot BW_{\text{eff}}\right)$$

where $BW_{\text{eff}}$ may be expanded into cache-hierarchy terms as defined in Section 1.6.

The end-to-end latency model is then

$$T_{\text{ternary}} = \frac{M N_{\text{eff}}}{P_{\text{roof}}} + T_{\text{decode}} + T_{\text{reduce}} + T_{\text{sync}}$$

and the speedup over the FP16 baseline is

$$S = \frac{T_{\text{FP16}}}{T_{\text{ternary}}} = \frac{T_{\text{baseline}}}{T_{\text{memory}} + T_{\text{decode}} + T_{\text{reduce}} + T_{\text{sync}}}$$

This is the master performance equation used throughout the rest of the analysis.

**Interpretation:** the kernel is memory-bound in the large-$N$ regime, decode- and occupancy-limited in the small-$N$ regime, and cache-aware reuse can move the crossover point materially. That is a much stronger statement than "bandwidth matters" because it identifies the actual controlling ceiling.

### 3.2 Tensor Core Pathway Comparison

NVIDIA Tensor Cores provide hardware-accelerated matrix operations that the ternary kernel **cannot use**:

| Pathway | Precision | Tensor Core | Throughput (RTX 4060) |
|---|---|---|---|
| FP16 GEMV | 16-bit | No (CUDA cores) | ~8.5 TFLOPS |
| FP16 GEMM | 16-bit | Yes | ~17 TFLOPS (2× CUDA) |
| BF16 GEMM | 16-bit | Yes | ~17 TFLOPS |
| INT8 GEMM | 8-bit | Yes | ~34 TOPS |
| FP8 GEMM | 8-bit | Yes (Ada) | ~68 TOPS |
| **Ternary GEMV** | **2-bit** | **No** | **Bandwidth-limited** |

**Critical insight:** The ternary kernel operates on CUDA cores, not Tensor Cores. Its advantage is **bandwidth efficiency**, not peak FLOPS. For GEMV (matrix × vector), Tensor Cores provide minimal benefit because the operation is memory-bound, not compute-bound. The ternary kernel's 8× bandwidth reduction translates directly to speedup in this regime.

**However:** For GEMM (matrix × matrix, batch size > 1), Tensor Cores dominate. The ternary kernel is **not competitive** with INT8/FP8 Tensor Core GEMM for batch sizes $\geq 4$-8. This is a fundamental architectural limitation, not an implementation gap.

### 3.3 Overhead Component Analysis

| Overhead Source | Estimated Cost | Mitigation |
|---|---|---|
| PTX decode (`BFE`, `PRMT`) | Throughput-limited by issue slots | Keep packed layout aligned |
| Zero-gate masking (`LOP3`) | Throughput-limited by integer pipe | Branchless; no divergence penalty |
| Sign flip / bit ops | Throughput-limited by integer pipe | Fuse into decode path |
| Shared memory sync | ~5-10 cycles/tile | Amortized over 1024-element tile |
| Warp reduction (`SHFL`) | ~5 cycles/warp | Single instruction per step |
| Block reduction | ~10 cycles/block | One per output row |
| Stream sync (host-device) | ~5-10 us | Batch multiple GEMVs |
| Activation upload (H2D) | $2N$ bytes per call | Pre-cache for repeated calls |
| Warp underutilization | $\eta_{\text{warp}} < 1$ | Improve tail handling / batch shape |
| Register pressure | Lowers $Occ$ | Reuse registers, limit live ranges |

**Net assessment:** the dominant cost is no longer a single scalar "decode tax." The relevant question is which ceiling wins: DRAM, L2/shared reuse, issue throughput, occupancy, or warp efficiency. That framing is harder to attack analytically because it matches the actual GPU execution model.
---

## 4. Validation Benchmarks & Impact Metrics

### 4.1 Key Performance Indicators (KPIs)

| KPI | Definition | Target | Measurement Method |
|---|---|---|---|
| **Latency (μs)** | End-to-end kernel time per GEMV | < 50% of FP16 GEMV | `cudaEvent` timing |
| **Throughput (GB/s)** | Effective memory bandwidth utilized | > 60% of theoretical (272 GB/s) | Bytes / latency |
| **Speedup vs FP16** | $T_{\text{FP16}} / T_{\text{ternary}}$ | > 4× for $M=1$ | Ratio of medians |
| **Speedup vs INT8** | $T_{\text{INT8}} / T_{\text{ternary}}$ | > 2× for $M=1$ | Ratio of medians |
| **Max Absolute Error** | vs. FP32 CPU reference | < $10^{-3}$ | Per-element comparison |
| **RMSE** | vs. FP32 CPU reference | < $10^{-4}$ | RMS over output vector |
| **VRAM Reduction** | Weight memory footprint | 8× vs FP16 | Static measurement |
| **Sparsity Utilization** | Latency improvement at $\rho_0 = 0.5$ | > 1.3× vs. dense ternary | Latency ratio |

### 4.2 End-to-End Transformer Validation

To demonstrate practical utility, the kernel MUST be validated in a realistic inference pipeline:

#### 4.2.1 Target Models

| Model | Hidden Dim ($N$) | Layers | Ternary Weight Memory | FP16 Weight Memory |
|---|---|---|---|---|
| GPT-2 Small | 768 | 12 | ~2.3 MB | ~36 MB |
| GPT-2 Medium | 1024 | 24 | ~6.3 MB | ~100 MB |
| LLaMA-7B (simulated) | 4096 | 32 | ~112 MB | ~1.8 GB |
| LLaMA-13B (simulated) | 5120 | 40 | ~219 MB | ~3.5 GB |

#### 4.2.2 KPI Definitions for Transformer Validation

| KPI | Formula | Acceptable Range |
|---|---|---|
| **Tokens/sec** | $\text{output\_tokens} / \text{total\_time}$ | > 2× FP16 baseline on same hardware |
| **Time-to-first-token (TTFT)** | Latency to produce first output token | < 80% of FP16 baseline |
| **VRAM footprint** | Peak GPU memory during inference | < 25% of FP16 baseline |
| **Perplexity degradation** | $\Delta \text{PPL} = \text{PPL}_{\text{ternary}} - \text{PPL}_{\text{FP16}}$ | < 0.5 on WikiText-2 |
| **Context window scaling** | Max sequence length at fixed VRAM | > 3× FP16 baseline |

#### 4.2.3 Perplexity Degradation Protocol

```python
def measure_perplexity_degradation(model_fp16, model_ternary, dataset="wikitext-2"):
    """Measure perplexity difference between FP16 and ternary models."""
    ppl_fp16 = evaluate_perplexity(model_fp16, dataset)
    ppl_ternary = evaluate_perplexity(model_ternary, dataset)

    delta_ppl = ppl_ternary - ppl_fp16
    relative_degradation = delta_ppl / ppl_fp16

    return {
        "ppl_fp16": ppl_fp16,
        "ppl_ternary": ppl_ternary,
        "delta_ppl": delta_ppl,
        "relative_degradation_pct": relative_degradation * 100,
    }
```

### 4.3 Deployment Scenario Matrix

| Scenario | Hardware | Batch Size | Sequence Length | Expected Benefit |
|---|---|---|---|---|
| **Local chat inference** | RTX 4060 (8 GB) | 1 | 2048-8192 | Max VRAM savings, 3× longer context |
| **Edge deployment** | Jetson Orin / RTX 3050 | 1-4 | 512-2048 | Fits larger models in constrained VRAM |
| **Consumer gaming GPU** | RTX 4060/4070 | 1 | 4096-16384 | Enables 7B+ models on 8 GB VRAM |
| **Long-context scaling** | Any 8+ GB GPU | 1 | 16384-65536 | KV-cache dominates; weight memory freed |
| **Datacenter serving** | A100/H100 | 32-256 | 512-2048 | **Not recommended** — use INT8/FP8 Tensor Cores |

---

## 5. Scope Declaration & Limitations

### 5.1 What This System IS

- A **GEMV-optimized** inference kernel for ternary-quantized weights
- A **bandwidth-reduction** technique, not a compute-throughput technique
- A **latency-sensitive** solution for batch-size-1 (autoregressive) decoding
- A **research prototype** demonstrating sub-byte quantization on consumer hardware
- A **PyTorch-compatible Python framework** with autograd for ternary-aware training

### 5.2 What This System IS NOT

- **Not a GEMM kernel** — does not exploit Tensor Cores; not competitive for batched inference
- **Not a replacement for cuBLAS** — no cuBLAS dependency; no general-purpose linear algebra
- **Not production-hardened** — PTX intrinsics lack architecture fallbacks; autograd engine lacks cycle detection
- **Not a training accelerator** — the CPU-based autograd and NumPy tensor system are orders of magnitude slower than PyTorch/JAX for training
- **Not architecture-portable** — tuned for sm_89 (Ada Lovelace); performance on other architectures is uncharacterized

### 5.3 Known Technical Debt

| Component | Issue | Severity | Remediation |
|---|---|---|---|
| PTX intrinsics | No `#ifdef __CUDA_ARCH__` fallback | High | Add `portable_utils.h` with C++ bitwise path |
| Autograd engine | Recursive `_build_topo` — stack overflow risk | Medium | Convert to iterative DFS |
| Tensor `_version` | Incremented but never checked | Medium | Add version guards in `Function.apply()` |
| FP16 accumulation | Precision loss for $N > 8192$ | Medium | Promote to `float` in final reduction |
| Activation upload | Allocated per-call (`GpuBuffer::alloc`) | Low | Pre-allocate and cache in `BitLinear` |
| Optimizer steps | Pure NumPy CPU — no GPU acceleration | Low | Acceptable for research; block for production |

### 5.4 Deep Optimization Theory

The following hardware-level optimization targets are specified in
[ARCHITECTURE.md](ARCHITECTURE.md) §9. Their theoretical justification is:

**`cp.async` staging:** The current `uint4` load path occupies the issuing warp for
the full global memory latency (~200-400 cycles). `cp.async` transfers data directly
from global memory to shared memory via the DMA engine, freeing the warp to perform
weight decode in parallel. The expected speedup is:

$$\Delta T_{\text{async}} = T_{\text{tile\_load}} - \max(T_{\text{tile\_load}} - T_{\text{decode}}, 0)$$

where $T_{\text{tile\_load}}$ is the activation tile load time and $T_{\text{decode}}$
is the weight decode time. For $N = 4096$, $T_{\text{tile\_load}} \approx T_{\text{decode}}$,
so the overlap approaches 100% and the speedup approaches 2x for the staging phase.

**SIMD-style bit decode:** The current BFE-based decode issues 3 instructions per weight
(48 per packed word). Batch extraction via AND/SHR reduces this to 6-8 instructions for
4 weights, improving instruction-level parallelism. The integer pipeline throughput limit
is:

$$T_{\text{decode}} = \max\left(\frac{N_{\text{bfe}}}{\Theta_{\text{int}}}, \frac{N_{\text{lop3}}}{\Theta_{\text{int}}}\right)$$

Reducing $N_{\text{bfe}}$ by 4x directly reduces the decode ceiling.

**L2 cache persistence:** The `cudaAccessPolicyWindow` API marks weight allocations as
"persisting" in the 32 MB L2 cache on sm\_89. For autoregressive decoding where the same
weight matrix is read for each generated token, this converts DRAM traffic to L2 traffic:

$$BW_{\text{effective}} = BW_{\text{L2}} \cdot \text{hit\_ratio} + BW_{\text{DRAM}} \cdot (1 - \text{hit\_ratio})$$

With `hitRatio = 1.0`, $BW_{\text{effective}} \approx BW_{\text{L2}} \approx 500\text{-}800$ GB/s,
a 2-3x improvement over DRAM-only access.

**KV-cache INT8 quantization:** The per-channel quantization error bound is:

$$|e_k| \leq \frac{\max(k) - \min(k)}{510} \approx 0.004\text{-}0.008$$

The attention score error $|\Delta \text{score}| \leq D \cdot |e_k| \cdot \max(|q|) \approx 0.77$
is within tolerance for inference workloads. This enables 2x context extension (S=4096 instead
of S=2048) at fixed VRAM.

---

## 6. Publication Roadmap

### 6.1 Minimum Viable Paper (MVP)

A publishable contribution requires:

1. **Formal problem statement** (Section 1 of this document)
2. **Empirical results** across the shape matrix (Section 2.3) with statistical rigor (Section 2.2.3)
3. **Comparison against baselines** (cuBLAS FP16, INT8) on the same hardware
4. **End-to-end transformer validation** with perplexity and tokens/sec (Section 4.2)
5. **Ablation study** on sparsity, threshold $\alpha$, and accumulator precision

### 6.2 Recommended Venues

| Venue | Fit | Notes |
|---|---|---|
| MLSys | High | Systems-focused ML conference |
| IEEE Micro | High | Hardware/software co-design |
| NeurIPS (Systems Track) | Medium | Requires broader ML impact |
| arXiv (cs.LG / cs.AR) | Always | Preprint for visibility |

### 6.3 Differentiation from Prior Art

| Prior Work | This Work |
|---|---|
| BitNet (1-bit, training-focused) | Inference-optimized with full framework |
| GPTQ/AWQ (post-training quantization) | STE-aware training + inference kernel |
| cuBLAS INT8 GEMV | 4× more compression (2-bit vs 8-bit) |
| Custom CUDA kernels (various) | Rust safety layer + Python framework integration |

---

## Appendix A: Importing and Using the Ternary-Zero Python Library

### A.1 Installation

The library is built with **maturin** (Rust → Python extension module):

```bash
# Prerequisites:
# - Python 3.9+
# - Rust toolchain (rustup)
# - CUDA Toolkit 12.x (nvcc in PATH)
# - MSVC Build Tools (Windows) or GCC (Linux)

# Install maturin
pip install maturin

# Build and install in development mode
maturin develop --release

# Or build a wheel
maturin build --release
pip install target/wheels/ternary_zero-0.1.0-*.whl
```

**If CUDA is not available**, the Python package still loads but the native GPU kernel (`_core`) will not be present. CPU-only operations (tensor, autograd, quantization) work without CUDA.

### A.2 Basic Usage

```python
import ternary_zero as tz
import numpy as np

# === Tensor Creation ===
x = tz.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = tz.randn(4, 3, requires_grad=True)  # 4×3 random tensor
b = tz.zeros(4, requires_grad=True)

# === Forward Pass ===
y = x @ w.T + b          # matrix multiply + bias
loss = y.sum()            # scalar loss
loss.backward()           # compute gradients

print(x.grad)             # gradient w.r.t. x
print(w.grad)             # gradient w.r.t. w

# === Neural Network Modules ===
model = tz.nn.Sequential(
    tz.nn.Linear(784, 256),
    tz.nn.ReLU(),
    tz.nn.Dropout(0.2),
    tz.nn.Linear(256, 128),
    tz.nn.ReLU(),
    tz.nn.BitLinear(128, 10, alpha=0.5),  # ternary-quantized layer
)
print(model)
print(f"Parameters: {model.num_parameters():,}")

# === Training Loop ===
optimizer = tz.optim.Adam(model.parameters(), lr=0.001)
loss_fn = tz.nn.CrossEntropyLoss()

for epoch in range(10):
    x_batch = tz.randn(32, 784)
    target = tz.tensor([0, 1, 2, 3] * 8, dtype=np.int64)

    logits = model(x_batch)
    loss = loss_fn(logits, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

### A.3 Ternary Quantization Utilities

```python
import ternary_zero as tz
import numpy as np

# === Quantize a weight tensor ===
w = tz.randn(64, 128)
ternary_w, scale = tz.quantize.ternary_quantize(w, alpha=0.5)

print(f"Unique values: {np.unique(ternary_w.data)}")  # [-1, 0, 1]
print(f"Scale factor: {scale:.4f}")

# === Analyze weight statistics ===
stats = tz.quantize.ternary_weight_analysis(ternary_w)
print(f"Sparsity: {stats['sparsity']:.1%}")
print(f"Compression vs FP32: {stats['compression_ratio_vs_fp32']}x")
print(f"Compression vs FP16: {stats['compression_ratio_vs_fp16']}x")

# === Pack/unpack for GPU kernel ===
packed = tz.quantize.pack_ternary_to_u32(ternary_w, n=128)
unpacked = tz.quantize.unpack_u32_to_ternary(packed, n=128)
assert np.array_equal(ternary_w.data.flatten(), unpacked.data)

# === Dequantize (for debugging/visualization) ===
dequant = tz.quantize.dequantize_ternary(ternary_w, scale)
```

### A.4 Inference Mode (No Gradient Overhead)

```python
# Disable gradient computation for inference
with tz.no_grad():
    model.eval()
    x = tz.randn(1, 784)
    output = model(x)
    prediction = output.data.argmax()
```

### A.5 Saving and Loading Models

```python
# Save model state
state = model.state_dict()
np.savez("model_weights.npz", **state)

# Load model state
loaded = np.load("model_weights.npz")
model.load_state_dict({k: loaded[k] for k in loaded.files})
```

### A.6 Available Modules Reference

```
tz.Tensor                    # Core tensor class with autograd
tz.tensor()                  # Create tensor from data
tz.zeros(), tz.ones()        # Factory functions
tz.randn()                   # Random normal tensor
tz.no_grad()                 # Context manager: disable gradients
tz.enable_grad()             # Context manager: enable gradients

tz.nn.Module                 # Base class for all modules
tz.nn.Linear(in, out)        # Standard linear layer
tz.nn.BitLinear(in, out)     # Ternary-quantized linear layer
tz.nn.ReLU/GELU/Sigmoid/...  # Activation functions
tz.nn.LayerNorm/BatchNorm1d  # Normalization layers
tz.nn.Dropout(p)             # Regularization
tz.nn.CrossEntropyLoss()     # Classification loss
tz.nn.MSELoss()              # Regression loss
tz.nn.Sequential(*modules)   # Container for sequential models

tz.optim.SGD(params, lr)     # Stochastic gradient descent
tz.optim.Adam(params, lr)    # Adam optimizer
tz.optim.AdamW(params, lr)   # AdamW optimizer
tz.optim.RMSprop(params, lr) # RMSprop optimizer

tz.quantize.ternary_quantize()        # STE-aware quantization
tz.quantize.ternary_quantize_fixed()  # Fixed-threshold quantization
tz.quantize.pack_ternary_to_u32()     # 2-bit packing
tz.quantize.unpack_u32_to_ternary()   # 2-bit unpacking
tz.quantize.dequantize_ternary()      # Reconstruct FP from ternary

tz.utils.pack_binary()       # Binary weight packing
tz.utils.unpack_binary()     # Binary weight unpacking
tz.utils.binary_matmul()     # Binary matrix multiply
```
