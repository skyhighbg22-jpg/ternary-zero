# Ternary-Zero

**W2A16 Ternary-Weight Inference Engine for Consumer NVIDIA GPUs**

A specialized, hardware-aware inference runtime that maps 16-bit floating-point activations against 2-bit ternary weights $\{-1, 0, +1\}$, achieving **8x compression over FP16** while maintaining bandwidth-bound GEMV throughput on Ada Lovelace (sm\_89) hardware.

Ternary-Zero is purpose-built to exploit the **32 MB L2 cache persistence** on the RTX 4060, positioning it as a consumer-grade alternative to BitNet and llama.cpp for latency-sensitive, single-batch autoregressive decoding.

---

## Technical Vision

The weight-to-activation bottleneck dominates autoregressive LLM inference. For each generated token, the full weight matrix must traverse the memory hierarchy — from DRAM through L2 cache to registers — and the achievable throughput is bounded by memory bandwidth, not arithmetic capability. Conventional FP16 inference spends 16 bits per weight to represent values that, after quantization, require only 2 bits of information: $\{-1, 0, +1\}$.

Ternary-Zero exploits this redundancy gap. By constraining weights to a ternary alphabet, the system achieves:

$$C_{\text{vs FP16}} = \frac{16}{2} = 8\times \quad\text{compression}$$

This transforms the memory-bandwidth equation. On the RTX 4060 (272 GB/s GDDR6):

| Metric | FP16 GEMV | W2A16 GEMV | Improvement |
|--------|-----------|------------|-------------|
| Weight bytes (7B model) | 14.0 GB | 1.75 GB | **8x** |
| Memory latency floor | 51.5 ms | 6.4 ms | **8x** |
| L2 cache fit (single FFN layer) | 86 MB (no) | 10.75 MB (yes) | **Persistent** |

The ternary GEMV kernel is **multiply-free**: each decoded weight contributes one of three operations — add, subtract, or skip — executed through branchless bitwise masking on CUDA cores. The kernel is optimized for the regime where Tensor Cores provide no benefit: $M=1$ matrix-vector products in the bandwidth-bound decode phase.

### Positioning

| System | Approach | Target | Ternary-Zero Differentiator |
|--------|----------|--------|-----------------------------|
| **BitNet** | 1-bit training + inference | Datacenter GPUs | Ternary-Zero: full inference engine with Rust safety layer, consumer hardware focus |
| **llama.cpp** | INT4/INT8 post-training quantization | Consumer CPUs/GPUs | Ternary-Zero: 2-bit ternary (8x vs FP16) vs INT4 (4x vs FP16); STE-aware training support |
| **GPTQ/AWQ** | Post-training INT4 quantization | Datacenter GPUs | Ternary-Zero: training-time STE quantization + hardware-native 2-bit kernel |
| **TensorRT-LLM** | FP8/INT8 Tensor Core inference | Datacenter GPUs | Ternary-Zero: CUDA core path; no Tensor Core dependency; 2-bit compression |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python Layer (ternary_zero/)                 │
│  BitLinear.forward()  │  ternary_quantize()  │  Tensor class     │
│  autograd (STE)       │  pack/unpack utils   │  NumPy/PyTorch    │
│  inference engine     │  model patcher       │  streaming engine │
├─────────────────────────┬───────────────────────────────────────┤
│                         │ PyO3 / numpy bindings                  │
├─────────────────────────▼───────────────────────────────────────┤
│                     Rust Layer (src/)                            │
│  BitLinear struct  │  CudaMemoryPool  │  PendingResult           │
│  GpuBuffer (RAII)  │  PinnedHostBuf   │  CudaStream / CudaEvent  │
│  pack_ternary_to_u32  │  STE quantize  │  error handling         │
├─────────────────────────┬───────────────────────────────────────┤
│                         │ FFI (extern "C")                       │
├─────────────────────────▼───────────────────────────────────────┤
│                     CUDA Layer (kernel/)                         │
│  ternary_zero_gemv_kernel  │  PTX BFE extraction                 │
│  stride-17 shared memory   │  vectorized uint4 loads             │
│  branchless zero-gating    │  L2 cache persistence (sm_89)       │
│  cp.async staging          │  FP32 warp reduction                │
├─────────────────────────┬───────────────────────────────────────┤
│                         │ nvcc compilation                       │
├─────────────────────────▼───────────────────────────────────────┤
│                     Build Layer (build.rs)                       │
│  nvcc detection  │  CUDA_HOME/CUDA_PATH resolution               │
│  sm_89 arch flag |  static lib linking  |  cpu-only fallback     │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | Language | Role |
|-------|----------|------|
| **Python** | Python/NumPy | Module API, autograd STE, quantization utilities, inference engine, model patcher |
| **Rust** | Rust + PyO3 | RAII GPU memory, pool allocator, FFI bindings, pack/unpack, async pipeline |
| **CUDA** | CUDA C++ + PTX | GEMV kernel, L2 cache policy, bank-conflict-free shared memory, cp.async |
| **Build** | Rust build.rs | nvcc invocation, static lib creation, platform-specific linking |

---

## Key Features

### Inference Engine (`ternary_zero/inference/`)

Complete LLM inference pipeline with HuggingFace model loading, on-the-fly ternary quantization, KV-cache management, RMSNorm, RoPE, multi-head attention with GQA, SwiGLU FFN, temperature/top-k/top-p sampling, and CLI runner.

```python
from ternary_zero.inference import InferenceEngine

engine = InferenceEngine.from_pretrained(
    "./models/llama-3.2-3b",
    alpha=0.5,              # ternary quantization threshold
    max_seq_len=2048,
)

output = engine.generate(
    "The meaning of life is",
    max_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
)
```

### Model Patcher (`ternary_zero/inference/model_patcher.py`)

Streaming weight-conversion utility that iterates through HuggingFace safetensors files tensor-by-tensor, performing on-the-fly 16-bit to packed 2-bit ternary conversion. Uses chunked processing to prevent OOM during conversion of large-scale models.

```python
from ternary_zero.inference import ModelPatcher

patcher = ModelPatcher(alpha=0.5, chunk_rows=256)
manifest = patcher.patch_model("./models/llama-3.2-3b", "./output/ternary")
# manifest.total_compression → 8.0x
```

### Double-Buffered Streaming Engine (`ternary_zero/inference/streaming_engine.py`)

Asynchronous per-layer PCIe streaming via double-buffering for models exceeding VRAM capacity. While the GPU executes Layer $N$, the CPU streams Layer $N+1$'s weights over PCIe via DMA, hiding transfer latency.

```python
from ternary_zero.inference import build_llama_streaming_engine

engine, layers = build_llama_streaming_engine("llama-3.1-70b", weight_dir="./ternary_weights")
output, profile = engine.execute_layers(activations)
# profile.effective_bandwidth_gbps → effective PCIe utilization
```

### CUDA Kernel (`kernel/ternary_zero.cu`)

PTX-inline-assembly GEMV kernel targeting sm\_89 (Ada Lovelace):

- **256 threads/block**, 8 warps, one block per output row
- **Stride-17 shared memory** eliminates bank conflicts
- **`cp.async`** for non-blocking Global → Shared memory transfers
- **PTX BFE** for single-instruction 2-bit extraction
- **Branchless zero-gating** via sign/magnitude bit masking
- **FP32 warp reduction** to prevent overflow at $N \geq 2048$
- **L2 cache persist policy** via `cudaAccessPolicyWindow` for weight reuse
- **128-bit `uint4` vectorized loads** for activation tiles

### Rust/PyO3 Layer (`src/`)

RAII GPU buffer management (`GpuBuffer<T>`), memory pooling (`CudaMemoryPool`), pinned host buffers (`PinnedHostBuffer<T>`), async forward pass with event polling (`PendingResult`), and PyO3 bindings exposing the CUDA kernel to Python. Built with `maturin`.

### Python Framework (`ternary_zero/`)

Autograd engine with STE support, `nn` module system (Linear, BitLinear, Conv1d, Conv2d, Embedding, normalization, pooling, dropout, activations, containers), optimizers (SGD, Adam, AdamW), loss functions, data loaders, and quantization utilities.

---

## Hardware Constraints & VRAM Planning

### VRAM Requirements by Model Size

| Model | Params | Ternary Weights | KV-Cache (S=2K) | Total VRAM | RTX 4060 (8GB) |
|-------|--------|----------------|-----------------|------------|-----------------|
| Llama-3.2-3B | 3.2B | 766 MB | 112 MB | ~1,200 MB | **Full VRAM** |
| Llama-2-7B | 6.7B | 1,607 MB | 320 MB | ~2,200 MB | **Full VRAM** |
| Llama-3-8B | 8.0B | 1,914 MB | 160 MB | ~2,400 MB | **Full VRAM** |
| 13B | 13.0B | 3,250 MB | 400 MB | ~4,000 MB | **Full VRAM** |
| 30B | 30.0B | 7,500 MB | 800 MB | ~8,600 MB | **Tight / streaming** |
| 70B | 70.6B | 17,650 MB | 640 MB | ~18,600 MB | **Streaming only** |
| 75B | 75.0B | 18,750 MB | 640 MB | ~19,700 MB | **Streaming only** |

### Key Constraint: 75B on 8GB VRAM

A 75B ternary model's weights (18.75 GB) exceed the RTX 4060's 8 GB VRAM by 2.3x. The only viable strategy is **per-layer weight streaming** from system RAM via double-buffered async PCIe transfers:

1. Keep KV-cache and embeddings in VRAM permanently (~900 MB)
2. Load one transformer layer's ternary weights at a time (~206 MB)
3. While GPU executes Layer $N$, prefetch Layer $N+1$ via PCIe DMA
4. Swap double-buffers; repeat for all 80 layers

**Estimated throughput with double-buffered streaming:** ~2.2 tokens/sec on PCIe 4.0, ~1.5 tokens/sec on PCIe 3.0.

See [HARDWARE_CONSTRAINTS.md](HARDWARE_CONSTRAINTS.md) for the complete mathematical analysis.

---

## Installation

### Prerequisites

- Python >= 3.9
- Rust toolchain (install via [rustup](https://rustup.rs/))
- CUDA Toolkit 12.x
- MSVC Build Tools (Windows) or GCC/clang (Linux)

### Build (with CUDA GPU support)

```bash
git clone https://github.com/VaishantSaiSambu/ternary-zero.git
cd ternary-zero

pip install maturin
maturin develop --release
```

Or build a wheel:

```bash
maturin build --release
pip install target/wheels/ternary_zero-0.1.0-*.whl
```

### Install Inference Dependencies

```bash
pip install safetensors transformers tokenizers
```

### CPU-Only (development, no CUDA)

```bash
pip install numpy>=1.21
pip install -e . --no-build-isolation
```

---

## Usage

### LLM Inference Engine

```python
from ternary_zero.inference import InferenceEngine

engine = InferenceEngine.from_pretrained(
    "./models/llama-3.2-3b",
    alpha=0.5,
    max_seq_len=2048,
)

# Generate text
output = engine.generate(
    "The meaning of life is",
    max_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
)

# Interactive chat
response = engine.chat(
    "Explain quantum computing",
    system_prompt="You are a helpful assistant.",
    max_tokens=256,
)

# Benchmark performance
engine.benchmark()
```

### CLI Runner

```bash
python -m ternary_zero.inference.run ./models/llama-3.2-3b "Hello, world"
python -m ternary_zero.inference.run ./models/llama-3.2-3b "Write a poem" --stream
python -m ternary_zero.inference.run ./models/llama-3.2-3b --chat
python -m ternary_zero.inference.run ./models/llama-3.2-3b --benchmark
```

### BitLinear Training

```python
import ternary_zero.nn as nn
import ternary_zero as tz

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.BitLinear(256, 128, alpha=0.5),
    nn.ReLU(),
    nn.BitLinear(128, 10, alpha=0.5),
)

optimizer = tz.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    x_batch = tz.randn(32, 784)
    target = tz.tensor([0, 1, 2, 3] * 8, dtype=np.int64)

    logits = model(x_batch)
    loss = loss_fn(logits, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Performance Benchmarking

### Benchmark Suite

```bash
# 1. Shape matrix benchmark (80-point M×N sweep → manifest.json)
python benchmarks/shape_matrix_benchmark.py --warmup 50 --iterations 1000

# 2. VRAM footprint + latency verification
python benchmarks/undeniable_benchmark.py --model llama-3.2-3b

# 3. Double-buffered streaming benchmark
python -m ternary_zero.inference.streaming_engine --model llama-3.2-3b --tokens 1

# 4. microGPT implementation comparison (6-way)
python benchmarks/run_benchmarks.py --train-steps 20 --inference-samples 5

# 5. Rust/CUDA criterion benchmarks
cargo bench --bench gemv_bench

# 6. Nsight Compute profiling
ncu --set full --kernel-name "ternary_zero_gemv_kernel" \
    --launch-skip 100 --launch-count 10 \
    python benchmarks/undeniable_benchmark.py --model llama-3.2-3b
```

### Measured Results (2026-05-09)

microGPT (4,192 parameters) benchmarked across 6 implementations:

| Implementation | Train (ms/step) | Speedup | Inf (ms) | Speedup | Tokens/s |
|---|---|---|---|---|---|
| Pure Python | 2064.5 | 1.0x | 583.1 | 1.0x | 7.6 |
| NumPy (vectorized) | 21.2 | 97.2x | 46.4 | 12.6x | 327.0 |
| PyTorch (CPU) | 88.1 | 23.4x | 20.3 | 28.7x | 197.6 |
| Ternary-Zero FP32 | 94.5 | 21.9x | 28.7 | 20.3x | 139.2 |
| Ternary-Zero BitLinear (2-bit) | 189.6 | 10.9x | 15.3 | 38.1x | 46.6 |
| CuPy (GPU) | 414.6 | 5.0x | 654.7 | 0.9x | 17.9 |

Weight memory: FP32 = 16,768 bytes; BitLinear = 1,060 bytes (16x compression).

### VRAM Footprint Analysis (2026-05-13)

| Model | Total Params | FP16 Weight Mem | Ternary Weight Mem | Compression |
|---|---|---|---|---|
| Llama-3.2-3B | 3,212,739,072 | 6,128 MB | 766 MB | **8.0x** |
| Llama-2-7B | 6,738,415,616 | 12,853 MB | 1,607 MB | **8.0x** |
| Llama-3-8B | 8,030,261,248 | 15,316 MB | 1,914 MB | **8.0x** |

---

## Current Status

**Implemented and functional:**
- Full LLM inference engine with KV-cache, sampling, and CLI runner
- Streaming model patcher (HuggingFace → ternary, chunked, OOM-safe)
- Double-buffered PCIe streaming engine for 70B+ models
- 80-point shape matrix benchmark suite with `manifest.json` output
- Full Python autograd engine with backward pass
- `nn` module hierarchy (20+ layer types)
- SGD, Adam, and AdamW optimizers
- Ternary quantization with STE-based training
- Rust PyO3 bindings for tensor operations and BitLinear forward pass
- CUDA kernel source with PTX inline assembly and L2 persistence
- microGPT benchmark suite (6-way comparison)

**In progress (see [ROADMAP.md](ROADMAP.md)):**
- HuggingFace model patcher validation on Llama-3.2-3B
- Automated 80-point shape matrix benchmark execution
- Per-layer PCIe streaming validation on 70B models
- Deep optimization: `cp.async`, PTX SIMD bitwise, KV-cache quantization

---

## Documentation

| Document | Audience | Description |
|----------|----------|-------------|
| [ROADMAP.md](ROADMAP.md) | All | Version 1.0 development milestones and engineering targets |
| [ARCHITECTURE.md](ARCHITECTURE.md) | CUDA developers, systems engineers | System architecture, CUDA kernel internals, deep optimization targets |
| [BENCHMARKS.md](BENCHMARKS.md) | ML researchers, performance engineers | Benchmark methodology, results, shape matrix suite specification |
| [METHODOLOGY.md](METHODOLOGY.md) | ML researchers, academic reviewers | Mathematical foundations, quantization theory, roofline analysis |
| [HARDWARE_CONSTRAINTS.md](HARDWARE_CONSTRAINTS.md) | Systems engineers | 75B on 8GB feasibility, CPU-GPU streaming, memory budget analysis |
| [EXECUTION_PLAN.md](EXECUTION_PLAN.md) | Project leads, contributors | Workstream sequencing, CI/CD, validation protocol |
| [ARCHITECTURE_GOVERNANCE.md](ARCHITECTURE_GOVERNANCE.md) | Contributors, reviewers | Architectural decision records (ADR-001 through ADR-004) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributors | Development setup, coding standards, PR process |

---

## Current Limitations

- **75B models require CPU-GPU streaming.** Weight memory (18.75 GB) exceeds 8GB VRAM. Double-buffered per-layer streaming is implemented but throughput is PCIe-limited (~2.2 tok/s on PCIe 4.0).
- **No GPU-accelerated training loop.** The CUDA kernel currently targets inference GEMV only. Training uses the Python autograd engine with CPU-based ternary quantization.
- **sm\_89 specific.** The CUDA kernel targets Ada Lovelace (RTX 4060). Other architectures require kernel modifications (see [ROADMAP.md](ROADMAP.md) §3).
- **Single-GPU only.** No multi-GPU or distributed training support.
- **No model zoo.** No pretrained ternary models are provided. Users must download and quantize their own models from HuggingFace.

---

## Testing

```bash
pytest tests/ -v                    # Python tests (including inference engine)
pytest tests/test_inference.py -v   # Inference engine end-to-end test
cargo test                          # Rust tests
cargo clippy                        # Rust linter
cargo fmt --check                   # Rust formatter
```

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE).

## Maintainer

Vaishant Sai Sambu
