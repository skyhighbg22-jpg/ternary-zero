# Ternary-Zero

W2A16 ternary weight quantization framework for GEMV inference on consumer NVIDIA GPUs.

## What This Is

Ternary-Zero is a research codebase implementing 2-bit weight, 16-bit activation (W2A16) inference kernels. Weights are quantized to ternary values {-1, 0, +1}, giving 16x compression relative to FP32 and 8x relative to FP16. The framework is a Python/Rust/CUDA hybrid:

- `CUDA kernel` (`kernel/ternary_zero.cu`) -- PTX inline assembly GEMV kernel with branchless zero-gating, shared memory staging with bank-conflict-free padding, and hybrid FP16-to-FP32 accumulation. Targets `sm_89` (Ada Lovelace / RTX 4060).
- `Rust/PyO3 layer` (`src/`) -- RAII GPU buffer management, memory pooling, STE (Straight-Through Estimator) forward/backward pass, and PyO3 bindings exposing the CUDA kernel to Python. Built with `maturin`.
- `Python framework` (`ternary_zero/`) -- Autograd engine, `nn` module system (Linear, BitLinear, Conv1d, Conv2d, Embedding, normalization, pooling, dropout, activations, containers), optimizers (SGD, Adam), loss functions, data loaders, and quantization utilities. Compatible with NumPy arrays.
- `Build system` -- Maturin compiles the Rust+CUDA code into a native Python extension module (`ternary_zero._core`).

## Current Status

**Implemented and functional:**
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
src/                    Rust: lib.rs, bitlinear.rs, ste.rs, ffi.rs, error.rs
kernel/                 CUDA: ternary_zero.cu (PTX GEMV kernel)
build.rs                Rust build script (compiles CUDA via cc)
benchmarks/             Python benchmark scripts + results.json
benches/                Rust criterion benchmarks (gemv_bench.rs)
tests/                  Python pytest tests
```

## Installation

### Prerequisites

- Python >= 3.9
- Rust toolchain (install via [rustup](https://rustup.rs/))
- CUDA Toolkit 12.x
- MSVC Build Tools (Windows) or GCC/clang (Linux)

### Build

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

### CPU-Only (development, no CUDA)

```bash
pip install numpy>=1.21
pip install -e . --no-build-isolation
```

## Usage

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

### microGPT measured comparison (2026-05-09)

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

Key observations:
- BitLinear achieves the lowest inference latency (15.3ms) at this model scale through branchless zero-gating and multiply elimination
- Ternary-Zero FP32 is competitive with PyTorch CPU, validating the Python autograd engine
- CuPy overhead dominates at 4K-parameter scale; kernel launch cost exceeds compute
- These results are on a toy model; scaling behavior to real LLM sizes is unknown

### W2A16 GEMV kernel -- theoretical estimates (not measured)

The following are bandwidth-bound projections for the CUDA GEMV kernel on RTX 4060 (272 GB/s memory bandwidth). These have **not** been validated on hardware.

| Configuration | Latency (est.) | Speedup vs FP16 (est.) | VRAM Savings |
|---|---|---|---|
| GPT-2 Small (M=1, N=768) | ~12.3 us | ~5.2x | 8x |
| GPT-2 Medium (M=1, N=1024) | ~18.7 us | ~4.8x | 8x |
| LLaMA-7B Sim (M=1, N=4096) | ~67.2 us | ~4.1x | 8x |

### Running benchmarks

```bash
python benchmarks/run_benchmarks.py --train-steps 20 --inference-samples 5
cargo bench --bench gemv_bench
```

Results are saved to `benchmarks/results.json`.

## Current Limitations

- **No GPU-accelerated training loop.** The CUDA kernel currently targets inference GEMV only. Training uses the Python autograd engine with CPU-based ternary quantization.
- **Untested at scale.** All measured benchmarks use a 4,192-parameter microGPT. Behavior at GPT-2 or LLaMA scale is theoretical.
- **CUDA kernel not profiled against cuBLAS.** The theoretical GEMV speedup estimates are derived from memory bandwidth calculations, not wall-clock measurements.
- **Single-GPU only.** No multi-GPU or distributed training support.
- **sm_89 specific.** The CUDA kernel targets Ada Lovelace (RTX 4060). Other architectures may require kernel modifications.
- **No model zoo.** No pretrained ternary models are provided.
- **BitLinear training is slower than FP32.** At the microGPT scale, BitLinear training (189.6 ms/step) is ~2x slower than FP32 (94.5 ms/step) due to quantization overhead in the forward pass.

## Testing

```bash
# Python tests
pytest tests/

# Rust tests
cargo test

# Rust linter
cargo clippy

# Rust formatter
cargo fmt --check
```

## Documentation

- [METHODOLOGY.md](METHODOLOGY.md) -- Mathematical foundations and quantization theory
- [ARCHITECTURE_GOVERNANCE.md](ARCHITECTURE_GOVERNANCE.md) -- Architecture and implementation details
- [EXECUTION_PLAN.md](EXECUTION_PLAN.md) -- Validation protocol and execution plan
- [BENCHMARKS.md](BENCHMARKS.md) -- Detailed benchmark methodology and results

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and pull request process.

## License

MIT License. See [LICENSE](LICENSE).

## Maintainer

Vaishant Sai Sambu
