from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# Double-Buffer State Machine
# =====================================================================

class BufferState(Enum):
    EMPTY = auto()
    LOADING = auto()
    READY = auto()
    EXECUTING = auto()


@dataclass
class LayerDescriptor:
    layer_idx: int
    name: str
    m: int
    n: int
    packed_weight_bytes: int
    scale: float


@dataclass
class DoubleBufferSlot:
    state: BufferState = BufferState.EMPTY
    layer_idx: int = -1
    packed_weights: Optional[np.ndarray] = None
    per_row_scale: Optional[np.ndarray] = None
    global_scale: float = 1.0
    m: int = 0
    n: int = 0
    load_time_ms: float = 0.0


# =====================================================================
# Pinned Host Buffer Pool (Simulated DMA Source)
# =====================================================================

class PinnedBufferPool:
    def __init__(self, max_buffers: int = 4):
        self._pool: deque = deque()
        self._max = max_buffers
        self._lock = threading.Lock()
        self._allocated = 0

    def acquire(self, size_bytes: int) -> np.ndarray:
        with self._lock:
            for i, (buf, buf_size) in enumerate(self._pool):
                if buf_size >= size_bytes:
                    self._pool.remove((buf, buf_size))
                    return buf[:size_bytes]

            self._allocated += 1
            return np.empty(size_bytes, dtype=np.uint8)

    def release(self, buf: np.ndarray):
        with self._lock:
            if len(self._pool) < self._max:
                self._pool.append((buf, len(buf)))


# =====================================================================
# Async Layer Loader (Background Thread)
# =====================================================================

class AsyncLayerLoader:
    def __init__(
        self,
        layer_descriptors: List[LayerDescriptor],
        weight_dir: str,
    ):
        self.descriptors = {d.layer_idx: d for d in layer_descriptors}
        self.weight_dir = weight_dir
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._load_queue: deque = deque()
        self._result_slots: Dict[int, DoubleBufferSlot] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def start(self):
        self._thread = threading.Thread(target=self._loader_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def request_load(self, layer_idx: int):
        with self._cv:
            self._load_queue.append(layer_idx)
            self._cv.notify()

    def get_loaded(self, layer_idx: int) -> Optional[DoubleBufferSlot]:
        with self._lock:
            return self._result_slots.pop(layer_idx, None)

    def _loader_loop(self):
        import os

        while not self._stop_event.is_set():
            with self._cv:
                while not self._load_queue and not self._stop_event.is_set():
                    self._cv.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break
                if not self._load_queue:
                    continue
                layer_idx = self._load_queue.popleft()

            desc = self.descriptors.get(layer_idx)
            if desc is None:
                continue

            slot = DoubleBufferSlot(
                state=BufferState.LOADING,
                layer_idx=layer_idx,
                m=desc.m,
                n=desc.n,
            )

            t0 = time.perf_counter()

            safe_name = desc.name.replace(".", "_")
            npz_path = os.path.join(self.weight_dir, f"{safe_name}.npz")
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                slot.packed_weights = data["packed_weights"]
                slot.per_row_scale = data["per_row_scale"]
                slot.global_scale = float(data["global_scale"])
            else:
                rng = np.random.default_rng(layer_idx)
                packed_cols = desc.n // 16
                slot.packed_weights = rng.integers(
                    0, 2**32, size=(desc.m * packed_cols,), dtype=np.uint32
                )
                slot.per_row_scale = np.ones(desc.m, dtype=np.float32)
                slot.global_scale = 1.0

            slot.load_time_ms = (time.perf_counter() - t0) * 1000
            slot.state = BufferState.READY

            with self._lock:
                self._result_slots[layer_idx] = slot


# =====================================================================
# GEMV Executor (GPU or CPU)
# =====================================================================

class GemvExecutor:
    def __init__(self, force_cpu: bool = False):
        self._core = None
        self._force_cpu = force_cpu
        self._detect_backend()

    def _detect_backend(self):
        if self._force_cpu:
            self._backend = "numpy"
            return
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from ternary_zero import _core
            if _core.has_cuda():
                self._core = _core
                self._backend = "cuda"
            else:
                self._core = _core
                self._backend = "cpu-rust"
        except ImportError:
            self._backend = "numpy"

    def execute(
        self,
        packed_weights: np.ndarray,
        activations: np.ndarray,
        m: int,
        n: int,
        scale: float,
    ) -> np.ndarray:
        x_flat = activations.ravel().astype(np.float32)

        if self._backend == "cuda" and self._core is not None:
            raw = self._core.ternary_gemv_gpu(packed_weights, x_flat, m, n)
        elif self._backend == "cpu-rust" and self._core is not None:
            raw = self._core.ternary_gemv_cpu_packed(packed_weights, x_flat, m, n)
        else:
            from ternary_zero.quantize import ternary_gemv_numpy
            raw = ternary_gemv_numpy(packed_weights, x_flat, m, n)

        return (raw * scale).astype(np.float32)


# =====================================================================
# Streaming Inference Engine (Double-Buffered)
# =====================================================================

@dataclass
class LayerProfile:
    layer_idx: int
    name: str
    load_time_ms: float
    compute_time_ms: float
    total_time_ms: float
    weight_bytes: int
    bandwidth_gbps: float


@dataclass
class StreamingProfile:
    total_layers: int
    total_compute_ms: float
    total_load_ms: float
    total_time_ms: float
    effective_bandwidth_gbps: float
    gpu_utilization_pct: float
    layers: List[LayerProfile] = field(default_factory=list)
    tokens_per_second: float = 0.0


class DoubleBufferedStreamingEngine:
    def __init__(
        self,
        layer_descriptors: List[LayerDescriptor],
        weight_dir: str,
        executor: Optional[GemvExecutor] = None,
        prefetch_depth: int = 2,
        force_cpu: bool = False,
    ):
        self.descriptors = layer_descriptors
        self.weight_dir = weight_dir
        self.executor = executor or GemvExecutor(force_cpu=force_cpu)
        self.prefetch_depth = prefetch_depth

        self._slots: List[Optional[DoubleBufferSlot]] = [None, None]
        self._slot_locks = [threading.Lock(), threading.Lock()]
        self._active_slot = 0

        self._loader = AsyncLayerLoader(layer_descriptors, weight_dir)
        self._pinned_pool = PinnedBufferPool(max_buffers=4)

    def _swap_slots(self):
        self._active_slot = 1 - self._active_slot

    def _get_active_slot(self) -> Optional[DoubleBufferSlot]:
        with self._slot_locks[self._active_slot]:
            return self._slots[self._active_slot]

    def _set_slot(self, slot_idx: int, slot: Optional[DoubleBufferSlot]):
        with self._slot_locks[slot_idx]:
            self._slots[slot_idx] = slot

    def _try_fill_inactive_slot(self, layer_idx: int):
        inactive = 1 - self._active_slot
        loaded = self._loader.get_loaded(layer_idx)
        if loaded is not None:
            self._set_slot(inactive, loaded)

    def execute_layers(
        self,
        activations: np.ndarray,
        layer_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, StreamingProfile]:
        self._loader.start()

        profile = StreamingProfile(
            total_layers=len(self.descriptors),
            total_compute_ms=0.0,
            total_load_ms=0.0,
            total_time_ms=0.0,
            effective_bandwidth_gbps=0.0,
            gpu_utilization_pct=0.0,
        )

        x = activations.copy()
        total_weight_bytes = 0

        t_global_start = time.perf_counter()

        for prefetch_idx in range(min(self.prefetch_depth, len(self.descriptors))):
            self._loader.request_load(prefetch_idx)

        for layer_idx, desc in enumerate(self.descriptors):
            t_wait_start = time.perf_counter()

            while True:
                self._try_fill_inactive_slot(layer_idx)
                slot = self._slots[1 - self._active_slot]
                if slot is not None and slot.state == BufferState.READY:
                    break
                time.sleep(0.0001)

            t_wait = time.perf_counter() - t_wait_start
            profile.total_load_ms += t_wait * 1000

            self._swap_slots()
            active = self._get_active_slot()
            if active is None:
                raise RuntimeError(f"Layer {layer_idx}: slot is None after swap")

            active.state = BufferState.EXECUTING

            if layer_idx + self.prefetch_depth < len(self.descriptors):
                self._loader.request_load(layer_idx + self.prefetch_depth)

            t_compute_start = time.perf_counter()
            if layer_fn is not None:
                x = layer_fn(x, layer_idx)
            else:
                x = self.executor.execute(
                    active.packed_weights,
                    x,
                    active.m,
                    active.n,
                    active.global_scale,
                )
            t_compute = time.perf_counter() - t_compute_start

            weight_bytes = desc.packed_weight_bytes
            total_weight_bytes += weight_bytes

            layer_profile = LayerProfile(
                layer_idx=layer_idx,
                name=desc.name,
                load_time_ms=active.load_time_ms,
                compute_time_ms=t_compute * 1000,
                total_time_ms=active.load_time_ms + t_compute * 1000,
                weight_bytes=weight_bytes,
                bandwidth_gbps=(
                    (weight_bytes / (t_compute * 1e-6)) / 1e9 if t_compute > 0 else 0.0
                ),
            )
            profile.layers.append(layer_profile)
            profile.total_compute_ms += t_compute * 1000

            active.state = BufferState.EMPTY
            self._set_slot(self._active_slot, None)

        t_global = time.perf_counter() - t_global_start
        profile.total_time_ms = t_global * 1000

        if profile.total_time_ms > 0:
            profile.effective_bandwidth_gbps = (
                (total_weight_bytes / (profile.total_time_ms * 1e-3)) / 1e9
            )
            compute_only = profile.total_compute_ms
            profile.gpu_utilization_pct = (
                (compute_only / profile.total_time_ms) * 100
                if profile.total_time_ms > 0 else 0.0
            )
            profile.tokens_per_second = (
                1000.0 / profile.total_time_ms if profile.total_time_ms > 0 else 0.0
            )

        self._loader.stop()
        return x, profile

    def __del__(self):
        try:
            self._loader.stop()
        except Exception:
            pass


# =====================================================================
# Pre-Configured Engine Builders
# =====================================================================

def build_llama_streaming_engine(
    model_name: str = "llama-3.2-1b",
    weight_dir: str = "",
    force_cpu: bool = False,
    prefetch_depth: int = 2,
) -> Tuple[DoubleBufferedStreamingEngine, List[LayerDescriptor]]:
    from .config import PRESET_CONFIGS

    config = PRESET_CONFIGS.get(model_name.lower())
    if config is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(PRESET_CONFIGS.keys())}")

    h = config.hidden_size
    inter = config.intermediate_size
    n_heads = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head_dim = h // n_heads

    layers: List[LayerDescriptor] = []
    proj_names = [
        ("q_proj", h, h),
        ("k_proj", h, n_kv * head_dim),
        ("v_proj", h, n_kv * head_dim),
        ("o_proj", h, h),
        ("gate_proj", h, inter),
        ("up_proj", h, inter),
        ("down_proj", inter, h),
    ]

    for layer_idx in range(config.num_layers):
        for proj_name, m, n in proj_names:
            full_name = f"model.layers.{layer_idx}.{proj_name}.weight"
            packed_bytes = m * (n // 16) * 4
            layers.append(LayerDescriptor(
                layer_idx=len(layers),
                name=full_name,
                m=m,
                n=n,
                packed_weight_bytes=packed_bytes,
                scale=1.0,
            ))

    engine = DoubleBufferedStreamingEngine(
        layer_descriptors=layers,
        weight_dir=weight_dir,
        prefetch_depth=prefetch_depth,
        force_cpu=force_cpu,
    )

    return engine, layers


# =====================================================================
# Standalone Benchmark
# =====================================================================

def benchmark_streaming(
    model_name: str = "llama-3.2-1b",
    weight_dir: str = "",
    num_tokens: int = 1,
    force_cpu: bool = True,
    verbose: bool = True,
) -> StreamingProfile:
    from .config import PRESET_CONFIGS

    config = PRESET_CONFIGS.get(model_name.lower())
    if config is None:
        raise ValueError(f"Unknown model: {model_name}")

    engine, layers = build_llama_streaming_engine(
        model_name=model_name,
        weight_dir=weight_dir,
        force_cpu=force_cpu,
        prefetch_depth=2,
    )

    rng = np.random.default_rng(42)
    activations = rng.standard_normal(config.hidden_size).astype(np.float32)

    if verbose:
        print("=" * 80)
        print("  DOUBLE-BUFFERED PCIe STREAMING BENCHMARK")
        print(f"  Model: {config.name}")
        print(f"  Layers: {len(layers)}")
        print(f"  Tokens: {num_tokens}")
        print("=" * 80)
        print()

    all_profiles: List[StreamingProfile] = []

    for token_idx in range(num_tokens):
        x, profile = engine.execute_layers(activations)
        all_profiles.append(profile)
        activations = x

        if verbose:
            print(
                f"  Token {token_idx+1}: "
                f"{profile.total_time_ms:.1f}ms total, "
                f"{profile.total_compute_ms:.1f}ms compute, "
                f"{profile.effective_bandwidth_gbps:.1f} GB/s eff BW, "
                f"{profile.gpu_utilization_pct:.1f}% GPU util"
            )

    if verbose and all_profiles:
        avg_compute = sum(p.total_compute_ms for p in all_profiles) / len(all_profiles)
        avg_total = sum(p.total_time_ms for p in all_profiles) / len(all_profiles)
        avg_bw = sum(p.effective_bandwidth_gbps for p in all_profiles) / len(all_profiles)
        avg_util = sum(p.gpu_utilization_pct for p in all_profiles) / len(all_profiles)

        print()
        print(f"  Average compute time:   {avg_compute:.1f}ms")
        print(f"  Average total time:     {avg_total:.1f}ms")
        print(f"  Average eff bandwidth:  {avg_bw:.1f} GB/s")
        print(f"  Average GPU utilization:{avg_util:.1f}%")
        print("=" * 80)

    return all_profiles[-1] if all_profiles else StreamingProfile(
        total_layers=0, total_compute_ms=0, total_load_ms=0,
        total_time_ms=0, effective_bandwidth_gbps=0, gpu_utilization_pct=0,
    )


# =====================================================================
# CLI
# =====================================================================

def main():
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        description="Ternary-Zero double-buffered PCIe streaming benchmark"
    )
    parser.add_argument("--model", default="llama-3.2-1b", help="Model name")
    parser.add_argument("--weight-dir", default="", help="Path to quantized weights")
    parser.add_argument("--tokens", type=int, default=1, help="Number of tokens to generate")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    profile = benchmark_streaming(
        model_name=args.model,
        weight_dir=args.weight_dir,
        num_tokens=args.tokens,
        force_cpu=args.cpu,
        verbose=not args.quiet,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output = {
            "total_layers": profile.total_layers,
            "total_compute_ms": profile.total_compute_ms,
            "total_load_ms": profile.total_load_ms,
            "total_time_ms": profile.total_time_ms,
            "effective_bandwidth_gbps": profile.effective_bandwidth_gbps,
            "gpu_utilization_pct": profile.gpu_utilization_pct,
            "tokens_per_second": profile.tokens_per_second,
            "layers": [
                {
                    "layer_idx": l.layer_idx,
                    "name": l.name,
                    "load_time_ms": l.load_time_ms,
                    "compute_time_ms": l.compute_time_ms,
                    "total_time_ms": l.total_time_ms,
                    "weight_bytes": l.weight_bytes,
                    "bandwidth_gbps": l.bandwidth_gbps,
                }
                for l in profile.layers
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    return 0


if __name__ == "__main__":
    main()
