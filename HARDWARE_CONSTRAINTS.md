# Hardware Constraints & Feasibility: Running 75B Models on 8GB VRAM

**Document ID:** HCF-001
**Version:** 1.0.0
**Status:** ACTIVE
**Classification:** Technical Analysis / Engineering Specification
**Project:** Ternary-Zero — W2A16 Ternary Inference Runtime
**Target Hardware:** NVIDIA RTX 4060 (Ada Lovelace, sm_89, 8GB GDDR6, 272 GB/s)

---

## 1. Executive Summary

This document provides a rigorous mathematical feasibility analysis for executing a 75-billion parameter Large Language Model on an NVIDIA RTX 4060 with 8GB VRAM using Ternary-Zero's W2A16 (2-bit weight, 16-bit activation) quantization. The central finding is:

**A 75B ternary model's weights alone require ~19.1 GB — more than 2× the RTX 4060's 8GB VRAM.** Full in-VRAM execution is physically impossible without layer streaming. However, a viable execution strategy exists: **CPU-GPU weight offloading** that streams ternary-packed transformer layers from system RAM to GPU VRAM on a per-layer basis, keeping only the active layer's weights, KV-cache, and activations in VRAM at any given time.

This document provides the mathematical proof, memory budget breakdown, offloading architecture, and engineering roadmap for making this work.

---

## 2. Mathematical Feasibility Analysis

### 2.1 Model Architecture Assumptions

For a 75B parameter model, we use the Llama-3.1-70B architecture as the reference (the closest publicly documented architecture to 75B):

| Parameter | Value |
|-----------|-------|
| Hidden size (d) | 8,192 |
| Intermediate size (d_ff) | 28,672 |
| Num layers (L) | 80 |
| Num attention heads | 64 |
| Num KV heads | 8 (GQA) |
| Head dim | 128 |
| Vocab size (V) | 128,256 |
| Total parameters | ~70.6B |

For a true 75B model, we scale L to 85 layers (all other dimensions identical), giving ~75.1B parameters. The analysis below uses the 70.6B reference architecture; the 75B case is a linear extrapolation.

### 2.2 Weight Memory: Ternary (W2) Format

Ternary-Zero packs 16 ternary weights {-1, 0, +1} into a single `uint32_t` using 2 bits each:

```
Bytes per weight = 2 bits / 8 = 0.25 bytes
```

#### 2.2.1 Per-Layer Weight Breakdown

| Component | Shape | Parameters | Ternary Bytes |
|-----------|-------|-----------|---------------|
| Q projection | [8192, 8192] | 67,108,864 | 16,777,216 |
| K projection | [1024, 8192] | 8,388,608 | 2,097,152 |
| V projection | [1024, 8192] | 8,388,608 | 2,097,152 |
| O projection | [8192, 8192] | 67,108,864 | 16,777,216 |
| Gate projection | [28672, 8192] | 234,881,024 | 58,720,256 |
| Up projection | [28672, 8192] | 234,881,024 | 58,720,256 |
| Down projection | [8192, 28672] | 234,881,024 | 58,720,256 |
| Input LayerNorm | [8192] | 8,192 | 8,192 |
| Post-Attn LayerNorm | [8192] | 8,192 | 8,192 |
| **Per-layer total** | | **864,645,376** | **216,161,344** |

```
Per-layer ternary weight memory = 216,161,344 bytes ≈ 206.2 MB
```

#### 2.2.2 Scale Factors

Each linear layer requires a per-row scale factor (FP32) for dequantization:

```
Scales per layer = (8192 + 1024 + 1024 + 8192 + 28672 + 28672 + 8192) = 83,968
Bytes per layer  = 83,968 × 4 bytes = 335,872 bytes ≈ 0.32 MB
```

Scale factor overhead is negligible (< 0.2% of weight memory).

#### 2.2.3 Total Model Weight Memory

| Component | Formula | Size |
|-----------|---------|------|
| Ternary weights (80 layers) | 80 × 216,161,344 | **17,292,907,520 bytes (16.11 GB)** |
| Scale factors (80 layers) | 80 × 335,872 | **26,869,760 bytes (25.6 MB)** |
| Embedding | 128,256 × 8,192 × 0.25 | **262,656,000 bytes (250.5 MB)** |
| LM Head | 128,256 × 8,192 × 0.25 | **262,656,000 bytes (250.5 MB)** |
| Final LayerNorm | 8,192 × 4 | **32,768 bytes** |
| **TOTAL** | | **17,847,122,048 bytes (16.63 GB)** |

For a true 75B model (85 layers):

```
75B ternary weight memory = 16.63 GB × (85/80) = 17.67 GB
```

### 2.3 KV-Cache Memory

For autoregressive generation with sequence length S:

```
Per-layer KV-cache = 2 × num_kv_heads × S × head_dim × sizeof(FP16)
                   = 2 × 8 × S × 128 × 2
                   = 4,096 × S bytes

Total KV-cache (80 layers) = 80 × 4,096 × S = 327,680 × S bytes
```

| Sequence Length | KV-Cache Size |
|----------------|---------------|
| 512 | 160 MB |
| 1,024 | 320 MB |
| 2,048 | 640 MB |
| 4,096 | 1,280 MB |
| 8,192 | 2,560 MB |
| 16,384 | 5,120 MB |

### 2.4 Activation Memory

For a single token decode (batch_size=1):

```
Hidden activations: 8,192 × 4 bytes (FP32) = 32 KB per layer
FFN activations: 28,672 × 4 bytes (FP32) = 112 KB per layer
Attention scores: 64 × S × 4 bytes (FP32) = 256 × S bytes

Total per-layer (S=2048): ~0.55 MB
Total 80 layers: ~44 MB (sequential, only 1-2 layers active at once)
```

### 2.5 VRAM Budget Summary (RTX 4060, 8 GB)

| Strategy | Weights | KV-Cache (S=2048) | Activations | CUDA Overhead | Total | Fits? |
|----------|---------|-------------------|-------------|---------------|-------|-------|
| **Full in-VRAM** | 16.63 GB | 0.64 GB | 0.04 GB | 0.3 GB | **17.61 GB** | **NO** |
| **Per-layer streaming** | 0.21 GB (1 layer) | 0.64 GB | 0.04 GB | 0.3 GB | **1.19 GB** | **YES** |
| **Partial offload (2 layers)** | 0.42 GB | 0.64 GB | 0.04 GB | 0.3 GB | **1.40 GB** | **YES** |

### 2.6 Conclusion: Feasibility Verdict

| Question | Answer |
|----------|--------|
| Can 75B ternary weights fit in 8GB VRAM? | **No.** 16.63 GB > 8 GB (2.08× over budget) |
| Can 75B ternary weights fit in 8GB VRAM with KV-cache? | **No.** Even weights alone exceed VRAM |
| Can a 75B model execute on RTX 4060 with offloading? | **Yes.** Per-layer streaming uses ~1.2 GB VRAM |
| What is the maximum model size for full in-VRAM? | **~30B ternary** (7.5 GB weights + overhead) |
| What is the maximum model size for comfortable in-VRAM? | **~13B ternary** (3.25 GB weights, room for 4K context) |

**The only viable strategy for 75B on 8GB is CPU-GPU layer streaming**, detailed in Section 3.

---

## 3. CPU-GPU Weight Offloading Architecture

### 3.1 Design: Per-Layer Weight Streaming

The core insight: during autoregressive token generation, each transformer layer is processed **sequentially**. At any given moment, only **one layer's weights** need to be in GPU VRAM.

```
System RAM (DDR5/DDR4)                     GPU VRAM (8 GB GDDR6)
+----------------------------------+      +---------------------------+
| Layer 0 weights  [206 MB packed] |      | Active layer weights      |
| Layer 1 weights  [206 MB packed] |      |   [206 MB packed]         |
| Layer 2 weights  [206 MB packed] |      |                           |
| ...                              |      | KV-cache [all layers]     |
| Layer 79 weights [206 MB packed] |      |   [640 MB @ seq=2048]     |
| Embedding        [250 MB FP16]  |      |                           |
| LM Head          [250 MB tern]  |      | Activation buffers        |
| Norm weights     [32 KB]        |      |   [~44 MB]                |
+----------------------------------+      |                           |
                                          | CUDA runtime overhead     |
     PCIe 4.0 x16: 32 GB/s peak          |   [~300 MB]               |
     PCIe 3.0 x16: 16 GB/s peak          +---------------------------+
```

### 3.2 Memory Budget (Per-Layer Streaming, S=2048)

| Component | Size | Notes |
|-----------|------|-------|
| Active layer packed weights | 206.2 MB | Ternary u32, loaded from CPU |
| Layer scale factors | 0.32 MB | FP32 per-row scales |
| KV-cache (80 layers) | 640 MB | FP16, stays in VRAM |
| Hidden state (current layer) | 32 KB | FP32 |
| FFN intermediate buffer | 112 KB | FP32 |
| Attention scratch | 0.5 MB | FP32 scores |
| Embedding lookup | 250 MB | FP16/FP32, stays in VRAM |
| LM Head | 250 MB | Ternary packed, stays in VRAM |
| CUDA runtime | ~300 MB | Driver, context, streams |
| **TOTAL** | **~1,647 MB** | **Fits in 8 GB with 6.35 GB headroom** |

### 3.3 Performance Impact of Weight Streaming

The dominant cost of per-layer streaming is the PCIe transfer time for each layer's weights:

```
Layer weight transfer time = 206.2 MB / PCIe_bandwidth

PCIe 4.0 x16 (32 GB/s):  206.2 / 32 = 6.44 ms per layer
PCIe 3.0 x16 (16 GB/s):  206.2 / 16 = 12.89 ms per layer
DDR5-5600 (~40 GB/s):     206.2 / 40 = 5.16 ms per layer (CPU-GPU via BAR)
```

For 80 layers per token:

```
PCIe 4.0: 80 × 6.44 ms = 515 ms → ~1.9 tokens/sec
PCIe 3.0: 80 × 12.89 ms = 1,031 ms → ~0.97 tokens/sec
```

These are **transfer-limited** lower bounds. The actual GEMV kernel execution adds ~0.5-2 ms per layer (depending on N), giving:

```
Estimated throughput:
  PCIe 4.0: ~1.5-1.8 tokens/sec
  PCIe 3.0: ~0.8-0.9 tokens/sec
```

### 3.4 Optimization: Double-Buffered Pipeline

To overlap PCIe transfers with GPU computation, use two weight buffers and CUDA streams:

```
Time →
Stream 0: [H2D Layer K]  [GEMV Layer K]  [H2D Layer K+2]  [GEMV Layer K+2]
Stream 1:        [H2D Layer K+1]  [GEMV Layer K+1]  [H2D Layer K+3]
          ↑ overlap ↑

VRAM cost: 2 × 206 MB = 412 MB (still fits comfortably)
Speedup: ~1.5-1.7× over sequential
```

### 3.5 Optimization: L2 Cache Pinning for Frequently-Used Layers

The embedding and LM head layers (250 MB each, 500 MB total) remain in VRAM permanently. They benefit from L2 cache pinning:

```
RTX 4060 L2 cache: 32 MB
Embedding: 250 MB → cannot fit in L2 (use streaming access)
LM Head: 250 MB → cannot fit in L2 (use streaming access)

For smaller models (7B ternary = 1.75 GB):
  Full weights fit in VRAM → L2 pinning effective for hot layers
```

### 3.6 Unified Memory Approach (Alternative)

CUDA Unified Memory (managed memory) allows the driver to page data between CPU and GPU automatically:

```cuda
cudaMallocManaged(&weights, weight_bytes);
// Driver pages in/out as needed
```

**Advantages:**
- Simpler code (no explicit H2D/D2H transfers)
- Driver-level prefetching can learn access patterns

**Disadvantages:**
- Page fault latency: ~10-50 µs per fault (4KB page)
- For 206 MB layer: ~50,000 page faults per layer load = ~500 ms-2.5 seconds
- Unpredictable performance (driver-dependent)
- Not suitable for latency-sensitive inference

**Recommendation:** Use explicit `cudaMemcpyAsync` with double-buffered streams instead of unified memory for deterministic, high-throughput transfers.

---

## 4. Engineering Roadmap: Ada Lovelace Optimized Kernels

### 4.1 Current Kernel Status

The existing `ternary_zero_gemv_kernel` (`kernel/ternary_zero.cu`) is:
- Compiled for sm_89 only
- Optimized for M=1 GEMV (single-token decode)
- Uses PTX BFE for 2-bit extraction
- Bank-conflict-free shared memory (stride-17)
- FP32 warp-level accumulation
- L2 cache persist policy

### 4.2 Required Kernel Enhancements for 75B Execution

| Enhancement | Priority | Impact | Complexity |
|-------------|----------|--------|------------|
| **Per-layer weight streaming API** | P0-CRITICAL | Enables 75B execution | Medium |
| **Double-buffered H2D pipeline** | P0-CRITICAL | 1.5-1.7× speedup on streaming | Medium |
| **Multi-architecture fat binary** | P1-HIGH | Portability to other GPUs | Low |
| **Fused bias+RMSNorm kernel** | P2-MEDIUM | Eliminates 2 kernel launches per layer | High |
| **Fused SiLU gate kernel** | P2-MEDIUM | Eliminates 1 kernel launch per FFN | Medium |
| **GEMM for prefill** | P2-MEDIUM | Fast prompt processing | High |
| **INT8 KV-cache** | P3-LOW | 2× longer context at same VRAM | Medium |

### 4.3 P0: Per-Layer Weight Streaming Implementation

#### 4.3.1 Rust API Design

```rust
/// Streaming weight buffer: holds one layer's packed ternary weights on GPU.
pub struct StreamingWeights {
    /// Double-buffer: two GPU allocations for ping-pong transfers
    buffers: [GpuBuffer<u32>; 2],
    /// Current active buffer index (0 or 1)
    active: usize,
    /// CUDA streams for async H2D transfers
    streams: [CudaStream; 2],
    /// Size of each buffer in elements
    buffer_len: usize,
}

impl StreamingWeights {
    pub fn new(buffer_len: usize) -> Result<Self, TernaryError>;

    /// Initiate async H2D transfer of next layer's weights into inactive buffer.
    pub fn prefetch_layer(&mut self, packed_weights_host: &[u32]) -> Result<(), TernaryError>;

    /// Swap buffers: the prefetched buffer becomes active.
    pub fn swap(&mut self);

    /// Get pointer to currently active GPU weight buffer.
    pub fn active_ptr(&self) -> *const u32;
}
```

#### 4.3.2 Python Inference Loop

```python
def generate_token_streaming(model, kv_cache, token_id, position):
    x = embed(token_id)

    for layer_idx in range(model.num_layers):
        # Prefetch next layer's weights while current layer computes
        if layer_idx + 1 < model.num_layers:
            model.streaming_weights.prefetch_layer(
                model.cpu_weights[layer_idx + 1]
            )

        # Execute current layer (weights already in VRAM from previous prefetch)
        x = model.blocks[layer_idx].forward(x, kv_cache, layer_idx, position)

        # Swap buffers for next iteration
        model.streaming_weights.swap()

    x = rms_norm(x, model.norm_weight)
    logits = lm_head(x)
    return logits
```

### 4.4 P1: Multi-Architecture Support

Modify `build.rs` to emit fat binaries:

```rust
let targets = vec!["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"];
for target in &targets {
    nvcc_args.push("-gencode");
    nvcc_args.push(&format!("arch=compute_{},code=sm_{}", 
        target.trim_start_matches("sm_"), target.trim_start_matches("sm_")));
}
```

### 4.5 P2: Fused Kernels

#### 4.5.1 Fused Ternary-GEMV + Bias + RMSNorm

Currently: 3 separate kernel launches per linear layer
- GEMV kernel → global memory output
- Bias add kernel → global memory read+write
- RMSNorm kernel → global memory read+write

Fused: 1 kernel launch, intermediate results stay in registers/shared memory

```cuda
__global__ void ternary_gemv_bias_rmsnorm_kernel(
    const uint32_t* weights,
    const __half* activations,
    const float* bias,
    const float* norm_weight,
    __half* output,
    int M, int N, float eps
) {
    // Phase 1: Ternary GEMV (same as current kernel)
    float acc = ternary_gemv_accumulate(weights, activations, M, N);

    // Phase 2: Add bias (register-resident)
    acc += bias[row];

    // Phase 3: RMSNorm (requires block-level reduction for variance)
    // ... warp-level reduction for mean(x^2) ...
    // ... normalize and scale in registers ...

    // Phase 4: Write final output
    output[row] = __float2half(result);
}
```

---

## 5. VRAM Capacity Planning Table

| Model Size | Params | Ternary Weights | Max Full-VRAM Context | Streaming Feasible? |
|-----------|--------|----------------|----------------------|---------------------|
| 1B | 1.0B | 250 MB | 131,072 tokens | Trivially yes |
| 3B | 3.0B | 750 MB | 65,536 tokens | Trivially yes |
| 7B | 6.7B | 1.68 GB | 16,384 tokens | Yes, full VRAM |
| 13B | 13.0B | 3.25 GB | 8,192 tokens | Yes, full VRAM |
| 30B | 30.0B | 7.50 GB | 512 tokens | Tight, full VRAM |
| 70B | 70.6B | 17.65 GB | N/A (streaming only) | **Yes, with streaming** |
| 75B | 75.0B | 18.75 GB | N/A (streaming only) | **Yes, with streaming** |

---

## 6. Recommendations

1. **For models ≤ 30B:** Full in-VRAM execution is possible. Focus on kernel optimization (L2 pinning, fused ops).

2. **For 75B models:** Implement per-layer weight streaming as the primary execution strategy. Target ~1.5 tok/s on PCIe 4.0.

3. **For maximum quality at 75B:** Consider a hybrid approach: keep the first and last N layers in VRAM (they're accessed every token), stream middle layers. This reduces transfer overhead by 2N layers per token.

4. **For production deployment:** Invest in NVLink or PCIe 5.0 hardware. A system with 64 GB+ system RAM and PCIe 5.0 (64 GB/s) would achieve ~3 tok/s for 75B ternary models.

5. **KV-cache optimization:** At 75B scale, KV-cache dominates VRAM. Implement INT8 KV-cache quantization to double the maximum context length, or implement sliding-window attention to bound cache size.

---

*This analysis uses published architecture specifications and the Ternary-Zero packing format. All calculations are reproducible. No model weights were downloaded or executed for this analysis.*
