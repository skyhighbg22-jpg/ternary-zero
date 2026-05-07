# ARCHITECTURE GOVERNANCE & PROJECT ROADMAP

**ADR-001 through ADR-004**
**Version:** 1.0.0
**Status:** ACTIVE
**Classification:** Architectural Decision Record (Binding)
**Project:** Ternary-Zero — W2A16 Sub-Byte GEMV Inference Runtime

---

## Preamble

This document constitutes the binding architectural governance framework for the Ternary-Zero project. It formalizes four critical dimensions — data provenance, documentation lifecycle, toolchain rationale, and strategic scope — as enforceable engineering requirements. Each section is structured as an Architectural Decision Record (ADR) per the ISO/IEC/IEEE 42010 framework: context, decision, consequences, and enforcement.

All contributors, automated pipelines, and review processes MUST treat the requirements herein as mandatory gates. Deviations require explicit override via a superseding ADR.

---

## ADR-001: Data Provenance & Observability Strategy

### Context

The current project contains zero empirical data. The `benches/gemv_bench.rs` Criterion harness defines benchmark stubs but has produced no results. The `METHODOLOGY.md` defines rigorous measurement protocols (§2.1–2.3) but prescribes no storage format, directory structure, or metadata schema. Without enforced data provenance, the project is vulnerable to:

1. **Provenance Decay** — benchmark results divorced from the exact code commit, hardware state, and configuration that produced them.
2. **Cherry-Picking** — selective reporting of favorable runs without visibility into variance or outliers.
3. **Reproduction Failure** — inability to re-run an experiment under identical conditions because environmental metadata was not captured.
4. **Reviewer Rejection** — any peer-reviewed venue (MLSys, IEEE Micro, NeurIPS) will reject submissions lacking structured experiment traces.

### Decision

Establish a formal `experiments/` directory hierarchy with a versioned metadata schema, automated capture hooks, and machine-readable trace storage.

#### 1.1 Directory Structure

```
experiments/
├── README.md                          # Experiment catalog and quick-start
├── schema/
│   ├── experiment_metadata.schema.json # JSON Schema for metadata validation
│   └── run_manifest.schema.json       # Per-run manifest schema
├── configs/
│   ├── baseline_fp16.toml             # cuBLAS FP16 baseline configuration
│   ├── baseline_int8.toml             # cuBLAS INT8 baseline configuration
│   ├── ternary_dense.toml             # Ternary kernel, ρ₀ = 0.0
│   ├── ternary_sparse_25.toml         # Ternary kernel, ρ₀ = 0.25
│   ├── ternary_sparse_50.toml         # Ternary kernel, ρ₀ = 0.50
│   └── ternary_sparse_75.toml         # Ternary kernel, ρ₀ = 0.75
├── runs/
│   ├── 2026-05-07_rtx4060_sm89_001/
│   │   ├── manifest.json              # Machine-generated run manifest
│   │   ├── env_snapshot.json          # Hardware + software environment capture
│   │   ├── raw_latency.csv            # Per-iteration latency (μs)
│   │   ├── raw_throughput.csv         # Per-iteration throughput (GB/s)
│   │   ├── summary_statistics.json    # Mean, median, p50, p95, p99, std, min, max
│   │   ├── nsight/                    # Nsight Compute profiling traces
│   │   │   ├── profile.ncu-rep       # Raw Nsight Compute report
│   │   │   └── summary.json          # Parsed roofline + occupancy data
│   │   ├── nsight_systems/           # Nsight Systems timeline traces
│   │   │   ├── timeline.nsys-rep     # Raw Nsight Systems report
│   │   │   └── gpu_kern_trace.json   # Parsed kernel execution timeline
│   │   ├── accuracy/
│   │   │   ├── cpu_reference.npy     # CPU reference output (FP32)
│   │   │   ├── gpu_output.npy        # GPU kernel output (FP16)
│   │   │   └── error_metrics.json    # Max absolute error, RMSE, per-element residuals
│   │   └── thermal_log.csv           # GPU temperature (°C) at 1 Hz during measurement
│   └── 2026-05-07_rtx4060_sm89_002/  # Subsequent run (auto-incremented)
│       └── ...
└── aggregate/
    ├── sparsity_sweep_summary.csv     # Latency vs. ρ₀ across all M×N shapes
    ├── shape_matrix_results.csv       # Full M×N grid results
    └── baseline_comparison.csv        # Ternary vs. cuBLAS FP16/INT8 ratios
```

#### 1.2 Experiment Metadata Schema

Every experiment run MUST produce a `manifest.json` conforming to the following schema:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Ternary-Zero Experiment Run Manifest",
  "type": "object",
  "required": [
    "schema_version",
    "run_id",
    "timestamp_utc",
    "git_commit",
    "git_branch",
    "git_dirty",
    "hardware",
    "software",
    "kernel_config",
    "measurement_protocol",
    "shape_matrix",
    "summary_statistics"
  ],
  "properties": {
    "schema_version": {
      "type": "string",
      "const": "1.0.0"
    },
    "run_id": {
      "type": "string",
      "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}_[a-z0-9]+_[a-z0-9]+_[0-9]{3}$",
      "description": "Format: YYYY-MM-DD_gpu_model_arch_run_number"
    },
    "timestamp_utc": {
      "type": "string",
      "format": "date-time"
    },
    "git_commit": {
      "type": "string",
      "pattern": "^[0-9a-f]{40}$",
      "description": "Full SHA-1 commit hash (no short hashes)"
    },
    "git_branch": { "type": "string" },
    "git_dirty": {
      "type": "boolean",
      "description": "true if working tree had uncommitted changes at run time"
    },
    "hardware": {
      "type": "object",
      "required": ["gpu_model", "gpu_arch", "vram_gb", "driver_version", "boost_clock_mhz", "power_limit_w"],
      "properties": {
        "gpu_model": { "type": "string" },
        "gpu_arch": { "type": "string" },
        "sm_version": { "type": "string" },
        "vram_gb": { "type": "number" },
        "l2_cache_mb": { "type": "number" },
        "theoretical_bandwidth_gbs": { "type": "number" },
        "driver_version": { "type": "string" },
        "boost_clock_mhz": { "type": "number" },
        "power_limit_w": { "type": "number" },
        "gpu_temperature_start_c": { "type": "number" },
        "gpu_temperature_end_c": { "type": "number" }
      }
    },
    "software": {
      "type": "object",
      "required": ["os", "cuda_toolkit", "rust_version", "python_version"],
      "properties": {
        "os": { "type": "string" },
        "cuda_toolkit": { "type": "string" },
        "rust_version": { "type": "string" },
        "python_version": { "type": "string" },
        "numpy_version": { "type": "string" },
        "compiler": { "type": "string" },
        "ternary_zero_version": { "type": "string" }
      }
    },
    "kernel_config": {
      "type": "object",
      "required": ["block_size", "act_tile_size", "launch_bounds"],
      "properties": {
        "block_size": { "type": "integer" },
        "act_tile_size": { "type": "integer" },
        "launch_bounds": { "type": "string" },
        "max_registers": { "type": "integer" },
        "l2_persist_policy": { "type": "boolean" },
        "ptx_fallback_used": { "type": "boolean" }
      }
    },
    "measurement_protocol": {
      "type": "object",
      "required": ["warmup_iterations", "measurement_iterations", "sync_mode"],
      "properties": {
        "warmup_iterations": { "type": "integer", "minimum": 50 },
        "measurement_iterations": { "type": "integer", "minimum": 500 },
        "sync_mode": { "type": "string", "enum": ["cudaStreamSynchronize", "cudaEventRecord"] },
        "clock_lock_applied": { "type": "boolean" },
        "power_management_mode": { "type": "string" }
      }
    },
    "shape_matrix": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["M", "N", "sparsity"],
        "properties": {
          "M": { "type": "integer" },
          "N": { "type": "integer" },
          "sparsity": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
        }
      }
    },
    "summary_statistics": {
      "type": "object",
      "description": "Keyed by 'M_N_sparsity' shape identifier",
      "additionalProperties": {
        "type": "object",
        "required": ["mean_us", "median_us", "p99_us", "std_us", "throughput_gbs"],
        "properties": {
          "mean_us": { "type": "number" },
          "median_us": { "type": "number" },
          "p99_us": { "type": "number" },
          "std_us": { "type": "number" },
          "min_us": { "type": "number" },
          "max_us": { "type": "number" },
          "throughput_gbs": { "type": "number" },
          "speedup_vs_fp16": { "type": "number" },
          "speedup_vs_int8": { "type": "number" },
          "max_abs_error": { "type": "number" },
          "rmse": { "type": "number" }
        }
      }
    }
  }
}
```

#### 1.3 Environment Capture Script

A mandatory pre-benchmark script (`scripts/capture_env.py`) MUST execute before any measurement run and produce `env_snapshot.json`:

```python
"""Environment capture for Ternary-Zero benchmark runs."""
import subprocess
import json
import platform
import datetime

def capture_environment() -> dict:
    env = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "rust_version": _run("rustc --version"),
        "cuda_toolkit": _run("nvcc --version"),
        "driver_version": _run("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "gpu_model": _run("nvidia-smi --query-gpu=name --format=csv,noheader"),
        "gpu_arch": _run("nvidia-smi --query-gpu=compute_cap --format=csv,noheader"),
        "vram_gb": _run("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"),
        "boost_clock_mhz": _run("nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits"),
        "power_limit_w": _run("nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits"),
        "gpu_temp_c": _run("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader"),
        "git_commit": _run("git rev-parse HEAD"),
        "git_branch": _run("git rev-parse --abbrev-ref HEAD"),
        "git_dirty": len(_run("git status --porcelain")) > 0,
    }
    return env

def _run(cmd: str) -> str:
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN"
```

#### 1.4 Enforcement Rules

| Rule | Enforcement | Failure Mode |
|---|---|---|
| Every benchmark run MUST produce `manifest.json` | CI gate: no merge without valid manifest | PR blocked |
| `git_commit` MUST be a full SHA-1 | Schema validation rejects short hashes | Schema validation failure |
| `git_dirty = true` runs MUST be flagged in reports | Aggregate scripts annotate dirty runs with `[DIRTY]` | Warning in report |
| Raw CSVs MUST accompany every summary statistic | Summary JSON rejected if CSVs missing | Validation failure |
| Nsight traces MUST be captured for at least one representative run per hardware | Minimum one `profile.ncu-rep` per experiment directory | Audit gate |
| Thermal log MUST be captured to detect throttling | Run rejected if GPU temp delta > 15°C | Data quality gate |

#### 1.5 Anti-Provenance-Decay Measures

1. **Immutable Run Directories** — Once created, a run directory is append-only. Files may be added (e.g., post-hoc Nsight analysis) but NEVER modified or deleted.
2. **Git LFS for Binary Traces** — `.ncu-rep`, `.nsys-rep`, and `.npy` files MUST be tracked via Git LFS with the `.gitattributes` rule:
   ```
   experiments/runs/**/nsight/*.ncu-rep filter=lfs diff=lfs merge=lfs -text
   experiments/runs/**/nsight_systems/*.nsys-rep filter=lfs diff=lfs merge=lfs -text
   experiments/runs/**/*.npy filter=lfs diff=lfs merge=lfs -text
   ```
3. **Config Hashing** — Each `.toml` config file's SHA-256 hash is recorded in the manifest. Any config change produces a new hash, making it impossible to misattribute results to a modified configuration.

---

## ADR-002: Documentation Lifecycle Management

### Context

The project currently relies on a single monolithic document, `METHODOLOGY.md` (626 lines), which conflates four distinct documentation concerns: mathematical foundations, reproducibility protocol, risk analysis, and user-facing API reference (Appendix A). This creates the following problems:

1. **Audience Mismatch** — A GPU kernel developer needs the PTX decode pipeline (§1.6); a Python user needs the import guide (Appendix A). Neither should parse the other's content.
2. **Review Bottleneck** — Any change to the usage guide requires re-reviewing the mathematical formalism, and vice versa.
3. **Discoverability Failure** — External contributors cannot locate relevant documentation without reading the entire file.
4. **Version Coupling** — The methodology document evolves on a different cadence than the API reference. Coupling them forces artificial synchronization.

### Decision

Migrate to a four-tier documentation hierarchy with clear ownership, audience, and lifecycle boundaries.

#### 2.1 Tier Definitions

| Tier | Directory | Audience | Lifecycle | Review Gate |
|---|---|---|---|---|
| **Developer** | `docs/dev/` | Core contributors, kernel engineers | Evolves with code | Code review required |
| **Research** | `docs/research/` | Academic reviewers, collaborators | Evolves with experiments | Technical lead approval |
| **User** | `docs/user/` | End users, Python developers | Evolves with API | API review required |
| **Benchmark** | `docs/bench/` | Performance engineers, reviewers | Evolves with experiment data | Automated + manual audit |

#### 2.2 Proposed Directory Structure

```
docs/
├── README.md                          # Documentation index with tier navigation
├── dev/
│   ├── ARCHITECTURE.md                # System architecture (Rust/CUDA/Python layers)
│   ├── KERNEL_GUIDE.md                # CUDA kernel internals, PTX decode pipeline
│   ├── RUST_FFI.md                    # FFI bindings, RAII patterns, error handling
│   ├── BUILD_SYSTEM.md                # build.rs, nvcc invocation, MSVC toolchain
│   ├── CONTRIBUTING.md                # Contribution guidelines, code style
│   └── adr/
│       ├── ADR-001-data-provenance.md # This document, Section 1
│       ├── ADR-002-documentation.md   # This document, Section 2
│       ├── ADR-003-toolchain.md       # This document, Section 3
│       └── ADR-004-scope.md           # This document, Section 4
├── research/
│   ├── METHODOLOGY.md                 # Formal research methodology (current file, migrated)
│   ├── MATHEMATICAL_FOUNDATIONS.md    # §1.1–1.6 of current METHODOLOGY.md
│   ├── REPRODUCIBILITY_PROTOCOL.md    # §2.1–2.4 of current METHODOLOGY.md
│   ├── RISK_ANALYSIS.md               # §3.1–3.3 of current METHODOLOGY.md
│   ├── PUBLICATION_ROADMAP.md         # §6 of current METHODOLOGY.md
│   └── DIFFERENTIATION.md             # §6.3 prior art comparison
├── user/
│   ├── QUICKSTART.md                  # 5-minute install + first inference
│   ├── API_REFERENCE.md               # Full Python API reference
│   ├── TERNARY_QUANTIZATION.md        # Quantization utilities guide
│   ├── TRAINING_GUIDE.md              # Training loop with STE-aware quantization
│   ├── DEPLOYMENT.md                  # Production deployment patterns
│   └── TROUBLESHOOTING.md             # Common issues and solutions
└── bench/
    ├── BENCHMARK_PROTOCOL.md          # How to run benchmarks correctly
    ├── BASELINE_COMPARISONS.md        # cuBLAS FP16/INT8 comparison methodology
    ├── ROOFLINE_ANALYSIS.md           # Roofline model interpretation
    └── results/
        └── index.md                   # Links to experiment runs in experiments/runs/
```

#### 2.3 Migration Path from Current Monolith

| Current Section | Target File(s) | Transformation |
|---|---|---|
| `METHODOLOGY.md` §1.1–1.6 | `docs/research/MATHEMATICAL_FOUNDATIONS.md` | Standalone; add LaTeX equation numbering |
| `METHODOLOGY.md` §2.1–2.4 | `docs/research/REPRODUCIBILITY_PROTOCOL.md` | Standalone; cross-ref `experiments/schema/` |
| `METHODOLOGY.md` §3.1–3.3 | `docs/research/RISK_ANALYSIS.md` | Standalone; add empirical validation refs |
| `METHODOLOGY.md` §4.1–4.3 | `docs/research/METHODOLOGY.md` (summary) + `docs/bench/BENCHMARK_PROTOCOL.md` | Split: KPI definitions → research, measurement procedure → bench |
| `METHODOLOGY.md` §5.1–5.3 | `docs/research/METHODOLOGY.md` (scope) + `docs/dev/ARCHITECTURE.md` | Scope declarations stay in research; tech debt → dev |
| `METHODOLOGY.md` §6.1–6.3 | `docs/research/PUBLICATION_ROADMAP.md` + `docs/research/DIFFERENTIATION.md` | Split venues from prior art |
| `METHODOLOGY.md` Appendix A | `docs/user/QUICKSTART.md` + `docs/user/API_REFERENCE.md` | Split: install → quickstart, API → reference |

#### 2.4 Cross-Reference Protocol

All documentation files MUST use relative Markdown links for cross-tier references:

```markdown
<!-- In docs/user/TRAINING_GUIDE.md -->
For the mathematical justification of Straight-Through Estimation,
see [Mathematical Foundations](../research/MATHEMATICAL_FOUNDATIONS.md#12-ternary-encoding-function).
```

Broken links are detected by a CI link-checker (`lychee` or `markdown-link-check`) and treated as build failures.

#### 2.5 Lifecycle Rules

| Rule | Enforcement |
|---|---|
| New public Python APIs MUST have corresponding `docs/user/API_REFERENCE.md` entries | PR checklist |
| Kernel changes MUST update `docs/dev/KERNEL_GUIDE.md` | Code review |
| Experiment results MUST update `docs/bench/results/index.md` | Automated by experiment runner |
| Research claims MUST cite specific experiment run IDs | Peer review gate |
| `METHODOLOGY.md` MUST be decomposed within 2 sprints of this ADR | Sprint planning |

---

## ADR-003: Architectural Rationale & Toolchain Validation

### Context

The Ternary-Zero stack comprises three layers with distinct technology choices:

| Layer | Technology | Version |
|---|---|---|
| **GPU Kernel** | CUDA C++17 + PTX inline assembly | CUDA Toolkit 12.x |
| **Orchestration** | Rust 2021 Edition + PyO3 0.22 | Rust 1.7x |
| **Framework** | Python 3.9+ + NumPy 1.21+ | Pure Python |
| **Build** | Maturin 1.5+ (PEP 517 backend) | Cargo + setuptools bridge |

The original plan incorrectly listed `cudarc` and `cublas` as dependencies — neither is present. The actual dependency graph is:

```
Cargo.toml:
  pyo3 = "0.22"          # Python bindings (extension-module feature)
  numpy = "0.22"         # PyO3-NumPy interop (PyArray types)
  half = "2"             # FP16 software emulation + hardware interop
  ndarray = "0.16"       # Rust n-dimensional array (GEMM CPU fallback)
  cc = "1"               # Build dependency only (MSVC toolchain discovery)

pyproject.toml:
  maturin >= 1.5         # PEP 517 build backend
  numpy >= 1.21          # Runtime dependency
```

This stack was selected over four legacy alternatives. This ADR documents the technical justification.

### Decision

#### 3.1 PyO3 vs. ctypes

| Criterion | ctypes | PyO3 (Selected) |
|---|---|---|
| **Type Safety** | Runtime type errors; manual `ctypes.POINTER`, `ctypes.c_int` declarations | Compile-time type checking; Rust's type system enforces ABI correctness |
| **Memory Ownership** | Caller manages `malloc`/`free` across FFI boundary; dangling pointer risk | RAII via `GpuBuffer<T>` — `Drop` impl calls `cudaFree` automatically |
| **Lifetime Management** | Manual reference counting or external GC coordination | Rust borrow checker prevents use-after-free at compile time; PyO3's `Bound<'py, T>` ties Python object lifetime to GIL scope |
| **Packaging** | Requires pre-built `.dll`/`.so`; no build system integration | Maturin produces `.whl` with embedded `.so`/`.pyd`; single `pip install` |
| **Error Propagation** | Manual `errno` checking or `GetLastError()` translation | `impl From<CudaError> for PyErr` enables `?` operator across FFI boundary |
| **Performance** | ~2-5 μs per call (Python → C → Python) | ~0.1-0.5 μs per call (direct function pointer dispatch via `#[pyfunction]` macro) |

**Verdict:** ctypes is unsuitable for a project where GPU buffer lifetimes must be precisely managed. A missed `cudaFree` in ctypes causes VRAM leaks that are invisible to Python's GC. PyO3's `Drop` integration eliminates this entire class of bugs.

#### 3.2 PyO3 vs. cffi

| Criterion | cffi | PyO3 (Selected) |
|---|---|---|
| **ABI Mode** | ABI mode: no compiler needed but manual struct layout | Full Rust type system; `#[repr(C)]` only where needed for FFI to CUDA |
| **API Mode** | Requires C header parsing; `ffi.cdef()` string | Rust macros generate bindings; no header parsing |
| **Null Safety** | No null safety; raw pointers throughout | `Option<T>` for nullable values; `NonNull` for GPU pointers |
| **Thread Safety** | Manual synchronization | `Send`/`Sync` auto-trait enforcement prevents data races |
| **Packaging** | Requires C compiler at build time (API mode) or pre-built lib (ABI mode) | Same, but Maturin handles Cargo + setuptools orchestration seamlessly |

**Verdict:** cffi's API mode offers no advantage over PyO3 and loses type safety. cffi's ABI mode is faster to set up but provides zero compile-time guarantees — unacceptable for GPU memory management.

#### 3.3 PyO3 vs. pybind11

| Criterion | pybind11 | PyO3 (Selected) |
|---|---|---|
| **Host Language** | C++ (manual memory management, RAII optional) | Rust (mandatory ownership, no undefined behavior) |
| **Memory Safety** | `std::unique_ptr` optional; raw `new`/`delete` common | All heap allocations tracked by ownership system; no `unsafe` in Python-facing code |
| **Build System** | CMake + setuptools integration (complex) | Maturin: single `pyproject.toml`, no CMake |
| **Exception Handling** | C++ exceptions → Python exceptions (unwinding across FFI) | `Result<T, E>` → `PyResult<T>` (no unwinding; explicit error propagation) |
| **Header Complexity** | Requires `<pybind11/pybind16.h>` + all transitive C++ headers | Rust macros; no header files |
| **Compile Time** | Heavy C++ template instantiation; 30-60s for moderate modules | Rust proc-macros; comparable or faster incremental builds |
| **GPU Integration** | Must wrap CUDA `.cu` files with `extern "C"` + link manually | Same `extern "C"` requirement, but Rust's `build.rs` provides structured build script integration |

**Verdict:** pybind11 is the strongest alternative, but it requires C++ — a language without mandatory memory safety. For a project that manages GPU VRAM via `cudaMalloc`/`cudaFree`, the cost of a use-after-free or double-free is a hard GPU hang, not a segfault. Rust's ownership model makes these bugs structurally impossible in safe code.

#### 3.4 Maturin vs. setuptools-rust vs. manual Cargo

| Criterion | Manual Cargo + setuptools | setuptools-rust | Maturin (Selected) |
|---|---|---|---|
| **Configuration Files** | `Cargo.toml` + `setup.py` + `MANIFEST.in` | `Cargo.toml` + `setup.py` + `pyproject.toml` | `Cargo.toml` + `pyproject.toml` only |
| **PEP 517 Compliance** | No (requires `setup.py`) | Partial | Full (`build-backend = "maturin"`) |
| **Wheel Building** | Manual `bdist_wheel` + Cargo post-build hook | Built-in but fragile | Built-in, production-grade |
| **Cross-Platform** | Manual per-platform handling | Partial | Full (Windows/Linux/macOS CI) |
| **PyPI Publishing** | Manual `twine upload` | Manual | `maturin publish` (single command) |
| **Development Mode** | `pip install -e .` requires custom build_ext | Supported | `maturin develop` (builds + installs in one step) |

**Verdict:** Maturin eliminates the `setup.py` entirely, reduces configuration to two files, and provides `maturin develop` for iterative development — a critical workflow for a Rust/Python hybrid project.

#### 3.5 Build Pipeline Architecture

The actual build pipeline (contrary to the original plan's claim of `cc::Build`) uses direct `nvcc` invocation in `build.rs`:

```
┌─────────────────────────────────────────────────────┐
│                   maturin develop                    │
│                       │                              │
│                       ▼                              │
│               cargo build --lib                      │
│                       │                              │
│                       ▼                              │
│                   build.rs                           │
│                       │                              │
│            ┌──────────┴──────────┐                   │
│            ▼                     ▼                   │
│    cc::Build (MSVC          nvcc -O3                │
│    toolchain                --gpu-architecture=sm_89 │
│    discovery only)          -std=c++17              │
│            │                     │                   │
│            │                     ▼                   │
│            │            ternary_zero.cu              │
│            │            ptx_utils.h                  │
│            │            ternary_zero.h               │
│            │                     │                   │
│            │                     ▼                   │
│            │            ternary_zero.obj             │
│            │                     │                   │
│            │                     ▼                   │
│            │            lib.exe / ar rcs             │
│            │                     │                   │
│            │                     ▼                   │
│            │            ternary_zero.lib (.a)        │
│            │                     │                   │
│            └──────────┬──────────┘                   │
│                       │                              │
│                       ▼                              │
│            rustc links lib + cudart_static           │
│                       │                              │
│                       ▼                              │
│            ternary_zero._core.pyd (.so)              │
└─────────────────────────────────────────────────────┘
```

The `cc` crate is used **exclusively** for MSVC toolchain discovery (locating `cl.exe` and `lib.exe` paths). It does NOT compile any C/C++ source files. All CUDA compilation is performed by direct `nvcc` invocation via `std::process::Command`.

#### 3.6 Dependency Justification

| Dependency | Justification | Alternative Rejected |
|---|---|---|
| `pyo3 = "0.22"` | Python FFI with RAII binding | ctypes (unsafe), cffi (no type safety), pybind11 (C++ required) |
| `numpy = "0.22"` | Zero-copy NumPy ↔ Rust array interop via `PyArray` | Manual buffer protocol (error-prone) |
| `half = "2"` | FP16 type with hardware interop (`__half_as_ushort`, `__ushort_as_half`) | Roll-your-own bit manipulation (fragile) |
| `ndarray = "0.16"` | N-dimensional array for CPU GEMM fallback and STE gradient computation | Raw `Vec<f32>` with manual indexing (error-prone) |
| `cc = "1"` | MSVC toolchain discovery only | Manual registry lookup (platform-specific) |
| `criterion = "0.5"` | Statistical benchmarking with HTML reports | `std::time::Instant` (no statistical rigor) |

---

## ADR-004: Strategic Scope & Risk Mitigation

### Context

The Ternary-Zero project currently contains approximately 4,100 lines of code across three language layers. The Python framework layer (`ternary_zero/`) includes a custom autograd engine, 20+ autograd functions, a full `nn.Module` hierarchy, four optimizers, and quantization utilities. This creates a significant risk: the project's identity as a **compressed inference runtime with research tooling** is threatened by feature creep toward a **universal deep learning framework**.

This section defines the boundary, identifies specific technical risks, and establishes enforcement mechanisms.

### Decision

#### 4.1 Project Identity Statement

**Ternary-Zero is a compressed inference runtime with research tooling.**

It is:
- A W2A16 GEMV kernel optimized for single-batch (autoregressive) inference on consumer GPUs
- A Rust orchestration layer providing RAII-managed GPU buffer lifecycle
- A Python framework providing STE-aware ternary quantization for research experimentation
- A benchmarking and validation harness for reproducible performance measurement

It is NOT:
- A general-purpose tensor computation library
- A training framework competitive with PyTorch/JAX
- A GEMM accelerator for batched inference
- A universal deep learning deployment runtime

#### 4.2 The PyTorch Reimplementation Trap

The most significant technical risk to this project is **unbounded reimplementation of PyTorch semantics**. The current `ternary_zero` Python layer already exhibits early symptoms:

| Feature | Current State | PyTorch Equivalent | Risk Level |
|---|---|---|---|
| Autograd engine | Recursive `_build_topo` (stack overflow at ~1000 nodes) | Iterative DFS with cycle detection | **HIGH** |
| Broadcasting | `_broadcast_tensors` via NumPy `broadcast_shapes` | Full NumPy broadcasting + stride tricks | MEDIUM |
| `Tensor.__matmul__` | Delegates to `autograd.functions.MatMul` | cuBLAS/cuBLASLt dispatch | MEDIUM |
| `nn.Module` | 176 lines, functional | 2000+ lines, hooks, DDP, FSDP | LOW (current) |
| Optimizers | Pure NumPy, CPU-only | CUDA fused kernels, gradient scaling | LOW (current) |
| `_version` tracking | Incremented but never checked | Used for autograd graph staleness detection | **HIGH** |

**The trap mechanism:**
1. User requests feature X (e.g., "add `torch.nn.functional.interpolate`")
2. Implementer adds a minimal version
3. Edge cases surface (e.g., "interpolate doesn't handle 5D tensors")
4. Implementer expands the feature to handle edge cases
5. Feature grows to 500+ lines, still incomplete
6. Repeat for 20 features → the project becomes a buggy PyTorch clone
7. The inference kernel — the actual value proposition — is neglected

#### 4.3 Scope Boundary Definition

**IN SCOPE — Required for inference runtime + research tooling:**

| Capability | Justification | Acceptance Criteria |
|---|---|---|
| Ternary GEMV kernel | Core value proposition | Benchmarked, validated |
| STE-aware quantization | Required for training ternary weights | Functional, tested |
| Forward pass (inference) | Required for model evaluation | Correctness verified |
| Backward pass (STE gradients) | Required for ternary-aware training | Gradient checks pass |
| Basic autograd (MatMul, Add, ReLU, Softmax, etc.) | Minimum for single-layer training | Functional for target models |
| `nn.Module` (Parameter, state_dict) | Required for model serialization | Compatible with target architectures |
| `BitLinear` layer | Core ternary primitive | Benchmarked |
| Adam/AdamW optimizer | Required for ternary-aware fine-tuning | Convergence verified |
| CPU reference GEMV/GEMM | Required for accuracy validation | Correctness verified |

**OUT OF SCOPE — Explicitly excluded:**

| Capability | Reason for Exclusion | Risk of Inclusion |
|---|---|---|
| Full broadcasting semantics | NumPy broadcasting is sufficient for inference; edge cases in training are PyTorch-compatible only by accident | Unbounded complexity; ~2000 lines for full stride-trick implementation |
| GPU-accelerated optimizers | Training is not the primary use case; CPU optimizers are functional | CUDA kernel development for marginal training speedup |
| Distributed training (DDP, FSDP) | Irrelevant for single-GPU inference | Months of engineering for a feature that contradicts project scope |
| Custom autograd functions for all PyTorch ops | Each function requires forward + backward + gradient check | Linear growth in maintenance burden; exponential growth in edge cases |
| TorchScript / ONNX export | Deployment is via the native Rust/CUDA layer | Implementation of a serialization format that adds no value |
| Dynamic shape support in autograd | Static shapes are sufficient for inference; dynamic shapes add graph reconstruction overhead | Fundamental redesign of autograd engine |
| Mixed-precision training (AMP) | FP16 activations are already the default; AMP adds loss scaling complexity | Loss scaler implementation + inf/nan detection |
| Data loading / preprocessing | Out of scope for a kernel library | Entire separate subsystem |

#### 4.4 Technical Risk Register

| Risk ID | Risk | Probability | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| **R-001** | Recursive `_build_topo` causes Python `RecursionError` at N > 1000 | HIGH (any transformer > 12 layers) | HIGH (training failure) | Convert to iterative DFS with explicit stack; add cycle detection | Kernel team |
| **R-002** | `_version` never checked → stale gradient computation | MEDIUM (silent correctness bug) | HIGH (incorrect training) | Add version guard in `Function.apply()`; raise `RuntimeError` on version mismatch | Framework team |
| **R-003** | FP16 accumulation overflow for N > 8192 | MEDIUM (depends on activation scale) | MEDIUM (numerical error) | Document limit; add `float32` accumulation mode for large N | Kernel team |
| **R-004** | PTX intrinsics fail on non-sm_89 hardware | HIGH (any non-Ada GPU) | HIGH (runtime crash) | Add `#ifdef __CUDA_ARCH__` fallback in `portable_utils.h`; detect arch at runtime | Kernel team |
| **R-005** | Broadcasting edge cases produce silent wrong gradients | MEDIUM (complex model architectures) | HIGH (silent training corruption) | Add gradient check tests for all broadcast patterns; limit broadcasting to documented subset | Framework team |
| **R-006** | Optimizer state not on GPU → CPU-GPU transfer bottleneck for large models | LOW (research only) | LOW (performance, not correctness) | Document as known limitation; do NOT implement GPU optimizers in this project | N/A |
| **R-007** | Feature creep delays kernel optimization | HIGH (user requests) | HIGH (missed research milestones) | Enforce scope boundary via this ADR; reject out-of-scope PRs | Tech lead |

#### 4.5 Broadcasting Semantic Boundary

The project MUST implement broadcasting as follows:

**Required (in-scope) broadcasting rules:**
1. Scalar × Tensor (element-wise)
2. Tensor × Tensor with matching trailing dimensions (e.g., `(3,4)` + `(4,)`)
3. Tensor × Tensor with size-1 expansion (e.g., `(3,1)` + `(1,4)` → `(3,4)`)

**Explicitly excluded broadcasting rules:**
1. Stride-based broadcasting without data copy (use `np.broadcast_to` → `.copy()`)
2. Broadcasting with negative strides
3. Broadcasting for tensors with `ndim > 5`
4. Custom stride manipulation (as_strided, unfold, etc.)

**Enforcement:** Any PR that adds stride tricks, `as_strided`, or custom broadcasting logic MUST be rejected unless accompanied by a superseding ADR.

#### 4.6 Autograd Engine Scope Boundary

The autograd engine MUST support exactly the following `Function` subclasses:

**Required (current, in `functions.py`):**
- `MatMul`, `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs`
- `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`
- `Sum`, `Mean`, `Max`, `Min`
- `Reshape`, `Transpose`, `Permute`, `Unsqueeze`, `Squeeze`
- `Log`, `Exp`, `Pow`
- `Embedding`, `CrossEntropyLoss`, `MSELoss`
- `LayerNorm`, `BatchNorm1d`
- `Dropout`

**NOT required (out of scope):**
- `Conv1d`, `Conv2d`, `Conv3d` (not needed for transformer inference)
- `MaxPool`, `AvgPool`, `AdaptiveAvgPool`
- `RNN`, `LSTM`, `GRU` (custom kernels would be needed)
- `MultiheadAttention` (decomposable into existing primitives)
- `F.interpolate`, `F.grid_sample`
- `torch.nn.utils.clip_grad_norm_` (implementable in 5 lines externally)
- Any `torchvision` or `torchaudio` operation

#### 4.7 Version Tracking Enforcement

```python
# MANDATORY addition to Function.apply():

@staticmethod
def apply(ctx, *inputs, **kwargs):
    # ... existing forward logic ...

    # Version guard: detect in-place mutation between forward and backward
    for inp in inputs:
        if isinstance(inp, Tensor) and inp.requires_grad:
            ctx._input_versions.append(inp._version)

    # In backward():
    for inp, saved_version in zip(inputs, ctx._input_versions):
        if isinstance(inp, Tensor) and inp._version != saved_version:
            raise RuntimeError(
                f"One of the differentiated Tensors appears to have been "
                f"modified in-place since forward(). This is not supported. "
                f"Version at forward: {saved_version}, current: {inp._version}"
            )
```

#### 4.8 Scope Violation Response Protocol

| Violation Type | Response | Authority |
|---|---|---|
| PR adds out-of-scope `Function` subclass | Reject with link to ADR-004 §4.6 | Any reviewer |
| PR adds GPU-accelerated optimizer | Reject with link to ADR-004 §4.3 OUT OF SCOPE | Any reviewer |
| PR adds stride tricks or `as_strided` | Reject with link to ADR-004 §4.5 | Any reviewer |
| PR adds distributed training hooks | Reject with link to ADR-004 §4.3 OUT OF SCOPE | Any reviewer |
| User requests feature that requires >500 new lines of Python | Escalate to tech lead for ADR review | Tech lead |
| Autograd engine modification that changes complexity class | Requires new ADR section | Tech lead |

---

## Appendix A: Quick Reference Card

### File Locations

| Concern | Primary File | Enforcement |
|---|---|---|
| Data provenance schema | `experiments/schema/experiment_metadata.schema.json` | JSON Schema validation |
| Documentation index | `docs/README.md` | CI link-checker |
| Toolchain justification | This document, ADR-003 | Architecture review |
| Scope boundary | This document, ADR-004 | PR review checklist |
| Mathematical foundations | `docs/research/MATHEMATICAL_FOUNDATIONS.md` | Peer review |
| Kernel internals | `docs/dev/KERNEL_GUIDE.md` | Code review |
| User API reference | `docs/user/API_REFERENCE.md` | API review |

### Decision Log

| ADR | Decision | Status | Supersedes |
|---|---|---|---|
| ADR-001 | Data provenance via `experiments/` hierarchy with JSON Schema | ACTIVE | — |
| ADR-002 | Four-tier documentation hierarchy | ACTIVE | Monolithic `METHODOLOGY.md` |
| ADR-003 | PyO3 + Maturin toolchain (not ctypes/cffi/pybind11) | ACTIVE | Original plan (incorrect deps) |
| ADR-004 | Inference runtime scope boundary; PyTorch reimplementation trap avoidance | ACTIVE | — |

---

*This document is binding. All PRs, code reviews, and sprint planning sessions MUST reference the relevant ADR sections. Deviations require a superseding ADR approved by the technical lead.*
