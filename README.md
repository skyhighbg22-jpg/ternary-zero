# Ternary-Zero

W2A16 ternary weight quantization framework for GEMV inference on consumer NVIDIA GPUs.

## What This Is

Ternary-Zero is a research codebase and **full LLM inference engine** implementing 2-bit weight, 16-bit activation (W2A16) inference kernels. Weights are quantized to ternary values {-1, 0, +1}, giving 16x compression relative to FP32 and 8x relative to FP16. The framework is a Python/Rust/CUDA hybrid:

- **Inference Engine** (`ternary_zero/inference/`) -- Complete LLM inference pipeline: model loading from HuggingFace, ternary quantization, KV-cache management, RMSNorm, RoPE, multi-head attention with GQA, SwiGLU FFN, temperature/top-k/top-p sampling, and CLI runner.
- **CUDA kernel** (`kernel/ternary_zero.cu`) -- PTX inline assembly GEMV kernel with branchless zero-gating, shared memory staging with bank-conflict-free padding, and hybrid FP16-to-FP32 accumulation. Targets `sm_89` (Ada Lovelace / RTX 4060).
- **Rust/PyO3 layer** (`src/`) -- RAII GPU buffer management, memory pooling, STE (Straight-Through Estimator) forward/backward pass, and PyO3 bindings exposing the CUDA kernel to Python. Built with `maturin`.
- **Python framework** (`ternary_zero/`) -- Autograd engine, `nn` module system (Linear, BitLinear, Conv1d, Conv2d, Embedding, normalization, pooling, dropout, activations, containers), optimizers (SGD, Adam), loss functions, data loaders, and quantization utilities.
- **Build system** -- Maturin compiles the Rust+CUDA code into a native Python extension module (`ternary_zero._core`).

## Current Status

**Implemented and functional:**
- Full LLM inference engine with KV-cache, sampling, and CLI runner (`ternary_zero/inference/`)
- Full Python autograd engine with backward pass
- `nn` module hierarchy (Linear, BitLinear, Conv1d, Conv2d, Embedding, LayerNorm, BatchNorm, MaxPool, AvgPool, Dropout, ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, Sequential, ModuleList)
- SGD and Adam optimizers
- Ternary quantization with STE-based training (`quantize.py`)
- Rust PyO3 bindings for tensor operations and BitLinear forward pass
- CUDA kernel source with PTX inline assembly
- microGPT benchmark suite (6-way comparison)

**Theoretical / not yet validated on hardware:**
- The GEMV kernel latency estimates for GPT-2/LLaMA-scale models are bandwidth-bound theoretical projections, not measured results
- The CUDA kernel has not been benchmarked against cuBLAS on actual hardware
- Multi-GPU support is not implemented
- INT4 weight support is not implemented

## Architecture

```
ternary_zero/           Python: autograd, nn modules, optimizers, quantization
  inference/            Python: Full LLM inference engine (config, quantize, layers, model, engine, sampler, tokenizer, cache)
src/                    Rust: lib.rs, bitlinear.rs, ste.rs, ffi.rs, error.rs
kernel/                 CUDA: ternary_zero.cu (PTX GEMV kernel)
                        CUDA: l2_persist.cu (L2 cache persistence manager)
                        CUDA: nvt/ternary_zero_nvtx.h (NVTX profiling markers)
benchmarks/             Python benchmark scripts + CUDA comparison harnesses
  baseline_comparison.cu    CUDA: Ternary-Zero vs cuBLAS FP16 vs INT4 dequant
  undeniable_benchmark.py   Python: VRAM footprint + latency verification
benches/                Rust criterion benchmarks (gemv_bench.rs)
tests/                  Python pytest tests (including test_inference.py)
```

## Hardware Constraints & Optimization

### VRAM Requirements by Model Size

| Model | Params | Ternary Weights | KV-Cache (S=2K) | Total VRAM | RTX 4060 (8GB) |
|-------|--------|----------------|-----------------|------------|-----------------|
| Llama-3.2-1B | 1.5B | 357 MB | 40 MB | ~700 MB | **Full VRAM** |
| Llama-2-7B | 6.7B | 1,607 MB | 320 MB | ~2,200 MB | **Full VRAM** |
| Llama-3-8B | 8.0B | 1,914 MB | 160 MB | ~2,400 MB | **Full VRAM** |
| 13B | 13.0B | 3,250 MB | 400 MB | ~4,000 MB | **Full VRAM** |
| 30B | 30.0B | 7,500 MB | 800 MB | ~8,600 MB | **Tight / streaming** |
| 70B | 70.6B | 17,650 MB | 640 MB | ~18,600 MB | **Streaming only** |
| 75B | 75.0B | 18,750 MB | 640 MB | ~19,700 MB | **Streaming only** |

### Key Constraint: 75B on 8GB VRAM

**A 75B ternary model's weights (18.75 GB) exceed the RTX 4060's 8 GB VRAM by 2.3×.** Full in-VRAM execution is impossible. The only viable strategy is **per-layer weight streaming** from system RAM:

1. Keep KV-cache and embeddings in VRAM permanently (~900 MB)
2. Load one transformer layer's ternary weights at a time (~206 MB)
3. Execute the layer's GEMV operations
4. Overwrite with the next layer's weights

**Estimated throughput with streaming:** ~1.5 tokens/sec on PCIe 4.0, ~0.9 tokens/sec on PCIe 3.0.

For models ≤ 13B, full in-VRAM execution is comfortable. For models 13-30B, it depends on context length. See `HARDWARE_CONSTRAINTS.md` for the complete mathematical analysis.

### CUDA Kernel Optimization (Ada Lovelace / sm_89)

The kernel targets the RTX 4060's Ada Lovelace architecture:
- **256 threads/block**, 8 warps, one block per output row
- **Stride-17 shared memory** eliminates bank conflicts
- **PTX BFE** for single-instruction 2-bit extraction
- **Branchless zero-gating** via sign/magnitude bit masking
- **FP32 warp reduction** to prevent overflow at N≥2048
- **L2 cache persist policy** for weight reuse across tokens
- **128-bit uint4 vectorized loads** for activation tiles

### Performance Benchmarking Guide

```bash
# 1. Inference engine benchmark (measures prefill + decode throughput)
python -m ternary_zero.inference.run ./models/llama-3.2-1b --benchmark

# 2. Kernel-level microbenchmark (measures raw GEMV latency)
python benchmarks/undeniable_benchmark.py --model llama-3.2-1b

# 3. Rust/CUDA criterion benchmarks
cargo bench --bench gemv_bench

# 4. Nsight Compute profiling (detailed kernel analysis)
ncu --set full --kernel-name "ternary_zero_gemv_kernel" \
    --launch-skip 100 --launch-count 10 \
    python benchmarks/undeniable_benchmark.py --model llama-3.2-1b

# 5. Nsight Systems timeline (end-to-end analysis)
nsys profile --trace cuda,nvtx \
    python -m ternary_zero.inference.run ./models/llama-3.2-1b --benchmark
```

**Metrics to report:**
- **Prefill throughput:** tokens/sec for prompt processing
- **Decode throughput:** tokens/sec for autoregressive generation
- **Time-to-first-token (TTFT):** latency from prompt to first generated token
- **GEMV kernel latency:** per-layer μs (median, p95, p99)
- **VRAM peak:** `nvidia-smi` maximum memory during inference
- **Bandwidth utilization:** achieved GB/s vs theoretical 272 GB/s

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

## Usage

### LLM Inference Engine

```python
from ternary_zero.inference import InferenceEngine

# Load and quantize a Llama model
engine = InferenceEngine.from_pretrained(
    "./models/llama-3.2-1b",  # path to HuggingFace model
    alpha=0.5,                 # ternary quantization threshold
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
# Generate text
python -m ternary_zero.inference.run ./models/llama-3.2-1b "Hello, world"

# Stream output
python -m ternary_zero.inference.run ./models/llama-3.2-1b "Write a poem" --stream

# Interactive chat
python -m ternary_zero.inference.run ./models/llama-3.2-1b --chat

# Run benchmark
python -m ternary_zero.inference.run ./models/llama-3.2-1b --benchmark
```

### Basic autograd

```python
import ternary_zero as tz
import numpy as np

x = tz.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = tz.randn(4, 3, requires_grad=True)
b = tz.zeros(4, requires_grad=True)

y = x @ w.T + b
loss = y.sum()
loss.backward()

print(x.grad)
print(w.grad)
```

### BitLinear layer

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

### Inference

```python
with tz.no_grad():
    model.eval()
    x = tz.randn(1, 784)
    output = model(x)
    prediction = output.data.argmax()
```

## Benchmark Results

### microGPT measured comparison (2026-05-09) — MEASURED

Karpathy's microGPT (4,192 parameters, vocab=27) benchmarked across 6 implementations.

System: Intel 12th Gen (12 cores), 15.8 GB RAM, RTX 4060 Laptop GPU, Python 3.13.2, PyTorch 2.6.0+cpu, CuPy 14.0.1.

| Implementation | Train (ms/step) | Speedup | Inf (ms) | Speedup | Tokens/s |
|---|---|---|---|---|---|
| Pure Python | 2064.5 | 1.0x | 583.1 | 1.0x | 7.6 |
| NumPy (vectorized) | 21.2 | 97.2x | 46.4 | 12.6x | 327.0 |
| PyTorch (CPU) | 88.1 | 23.4x | 20.3 | 28.7x | 197.6 |
| Ternary-Zero FP32 | 94.5 | 21.9x | 28.7 | 20.3x | 139.2 |
| Ternary-Zero BitLinear (2-bit) | 189.6 | 10.9x | 15.3 | 38.1x | 46.6 |
| CuPy (GPU) | 414.6 | 5.0x | 654.7 | 0.9x | 17.9 |

Weight memory: FP32 = 16,768 bytes; BitLinear = 1,060 bytes (16x compression).

### VRAM footprint analysis (2026-05-13) — STRUCTURED-ANALYTICAL

Computed from published Llama architecture specifications. No model weights
were downloaded. No GPU execution was required. See `BENCHMARKS.md` for
full methodology.

| Model | Total Params | FP16 Weight Mem | Ternary Weight Mem | Compression |
|---|---|---|---|---|
| Llama-3.2-1B | 1,498,482,688 | 2,858 MB | 357 MB | **8.0x** |
| Llama-2-7B | 6,738,415,616 | 12,853 MB | 1,607 MB | **8.0x** |
| Llama-3-8B | 8,030,261,248 | 15,316 MB | 1,914 MB | **8.0x** |

### W2A16 GEMV kernel — THEORETICAL estimates (not measured)

The following are roofline-style projections for the CUDA GEMV kernel on RTX 4060 (272 GB/s memory bandwidth). They account for packed weights, activation traffic, decode overhead, and occupancy assumptions, and they have **not** been validated on hardware.

| Configuration | Latency (est.) | Speedup vs FP16 (est.) | VRAM Savings |
|---|---|---|---|
| GPT-2 Small (M=1, N=768) | ~12.3 us | ~5.2x | 8x |
| GPT-2 Medium (M=1, N=1024) | ~18.7 us | ~4.8x | 8x |
| LLaMA-7B Sim (M=1, N=4096) | ~67.2 us | ~4.1x | 8x |

### Running benchmarks

```bash
# microGPT implementation comparison (measured)
python benchmarks/run_benchmarks.py --train-steps 20 --inference-samples 5

# VRAM footprint + latency verification (structured-analytical + optional GPU)
python benchmarks/undeniable_benchmark.py --model llama-3.2-1b
python benchmarks/undeniable_benchmark.py --model llama-2-7b

# Rust/CUDA kernel benchmarks (requires maturin develop --release)
cargo bench --bench gemv_bench

# CUDA baseline comparison (requires nvcc)
nvcc -O3 --use_fast_math -std=c++17 --gpu-architecture=sm_89 \
     -Ikernel -o benchmarks/baseline_comparison.exe \
     benchmarks/baseline_comparison.cu -lcublas -lcudart_static
```

Results are saved to `benchmarks/results.json` and `benchmarks/undeniable_results.json`.

## Current Limitations

- **75B models require CPU-GPU streaming.** Weight memory (18.75 GB) exceeds 8GB VRAM. Per-layer streaming is implemented but throughput is PCIe-limited (~1.5 tok/s on PCIe 4.0).
- **No GPU-accelerated training loop.** The CUDA kernel currently targets inference GEMV only. Training uses the Python autograd engine with CPU-based ternary quantization.
- **Untested at scale.** All measured benchmarks use a 4,192-parameter microGPT. Behavior at GPT-2 or LLaMA scale is theoretical.
- **CUDA kernel not profiled against cuBLAS.** The theoretical GEMV speedup estimates are derived from roofline and cache-aware performance modeling, not wall-clock measurements.
- **Baseline comparison harness written but not executed.** `benchmarks/baseline_comparison.cu` and `kernel/l2_persist.cu` are new source files that have not been compiled or run on GPU hardware. The NVTX profiling infrastructure (`kernel/nvt/ternary_zero_nvtx.h`) is also untested.
- **VRAM footprint claims are arithmetic, not empirical.** The 8.0x compression ratio is a mathematical property of 2-bit vs 16-bit encoding. No actual Llama model was loaded or profiled.
- **Single-GPU only.** No multi-GPU or distributed training support.
- **sm_89 specific.** The CUDA kernel targets Ada Lovelace (RTX 4060). Other architectures may require kernel modifications.
- **No model zoo.** No pretrained ternary models are provided. Users must download and quantize their own models from HuggingFace.
- **BitLinear training is slower than FP32.** At the microGPT scale, BitLinear training (189.6 ms/step) is ~2x slower than FP32 (94.5 ms/step) due to quantization overhead in the forward pass.

## Testing

```bash
# Python tests (including inference engine)
pytest tests/ -v

# Inference engine end-to-end test
pytest tests/test_inference.py -v

# Rust tests
cargo test

# Rust linter
cargo clippy

# Rust formatter
cargo fmt --check
```

## Documentation

- [HARDWARE_CONSTRAINTS.md](HARDWARE_CONSTRAINTS.md) -- 75B on 8GB feasibility analysis, CPU-GPU streaming architecture, kernel optimization roadmap
- [METHODOLOGY.md](METHODOLOGY.md) -- Mathematical foundations and quantization theory
- [ARCHITECTURE.md](ARCHITECTURE.md) -- System architecture, CUDA kernel internals, Rust backend
- [ARCHITECTURE_GOVERNANCE.md](ARCHITECTURE_GOVERNANCE.md) -- Architecture decision records
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md) -- Validation protocol and execution plan
- [BENCHMARKS.md](BENCHMARKS.md) -- Detailed benchmark methodology, results, and provenance labels

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and pull request process.

## License

GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE).

## Maintainer

Vaishant Sai Sambu
