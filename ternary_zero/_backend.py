from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np

_HAS_TORCH = False
_HAS_CUDA = False
torch = None

try:
    import torch as _torch
    torch = _torch
    _HAS_TORCH = True
    _HAS_CUDA = _torch.cuda.is_available()
except ImportError:
    pass

_default_device: Optional[str] = None


def has_torch() -> bool:
    return _HAS_TORCH


def is_cuda_available() -> bool:
    return _HAS_CUDA


def get_default_device() -> str:
    global _default_device
    if _default_device is not None:
        return _default_device
    if _HAS_CUDA:
        return "cuda"
    if _HAS_TORCH:
        return "cpu"
    return "cpu"


def set_default_device(device: str) -> None:
    global _default_device
    _default_device = device


def get_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    return get_default_device()


def num_gpus() -> int:
    if _HAS_CUDA:
        return torch.cuda.device_count()
    return 0


def current_gpu() -> int:
    if _HAS_CUDA:
        return torch.cuda.current_device()
    return -1


def gpu_memory_info(device: Optional[str] = None) -> dict:
    if not _HAS_CUDA:
        return {"allocated": 0, "cached": 0, "total": 0}
    d = device or get_default_device()
    return {
        "allocated": torch.cuda.memory_allocated(d),
        "cached": torch.cuda.memory_reserved(d),
        "total": torch.cuda.get_device_properties(d).total_mem,
    }


class CudaStream:
    def __init__(self, device: Optional[str] = None, priority: int = 0):
        self._stream = None
        if _HAS_CUDA:
            d = device or get_default_device()
            if "cuda" in d:
                self._stream = torch.cuda.Stream(device=d, priority=priority)

    def __enter__(self):
        if self._stream is not None:
            self._stream.__enter__()
        return self

    def __exit__(self, *args):
        if self._stream is not None:
            self._stream.__exit__(*args)

    def synchronize(self):
        if self._stream is not None:
            self._stream.synchronize()


class autocast:
    def __init__(self, enabled: bool = True, dtype: Optional[str] = "float16", device: Optional[str] = None):
        self._enabled = enabled
        self._dtype_str = dtype
        self._device = device
        self._ctx = None

    def __enter__(self):
        if _HAS_TORCH and self._enabled:
            d = self._device or get_default_device()
            if "cuda" in d:
                if self._dtype_str == "bfloat16":
                    amp_dtype = torch.bfloat16
                else:
                    amp_dtype = torch.float16
                self._ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self._ctx is not None:
            self._ctx.__exit__(*args)


def to_numpy(data) -> np.ndarray:
    if _HAS_TORCH and isinstance(data, torch.Tensor):
        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32)
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    return np.array(data, dtype=np.float32)


def to_torch(data, device: Optional[str] = None, dtype=None) -> "torch.Tensor":
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not available")
    if isinstance(data, torch.Tensor):
        d = device or get_default_device()
        return data.to(device=d, dtype=dtype if dtype is not None else data.dtype)
    if isinstance(data, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(data))
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device or get_default_device())
    t = torch.tensor(data, dtype=dtype or torch.float32)
    return t.to(device or get_default_device())


def create_tensor(data, device: Optional[str] = None, dtype=None):
    if _HAS_TORCH:
        return to_torch(data, device=device, dtype=dtype)
    if isinstance(data, np.ndarray):
        return data.astype(dtype or np.float32)
    return np.array(data, dtype=dtype or np.float32)


def create_zeros(shape, device: Optional[str] = None, dtype=None):
    if _HAS_TORCH:
        torch_dtype = _np_to_torch_dtype(dtype) if dtype is not None else torch.float32
        return torch.zeros(shape, dtype=torch_dtype, device=device or get_default_device())
    return np.zeros(shape, dtype=dtype or np.float32)


def create_ones(shape, device: Optional[str] = None, dtype=None):
    if _HAS_TORCH:
        torch_dtype = _np_to_torch_dtype(dtype) if dtype is not None else torch.float32
        return torch.ones(shape, dtype=torch_dtype, device=device or get_default_device())
    return np.ones(shape, dtype=dtype or np.float32)


def create_randn(shape, device: Optional[str] = None, dtype=None):
    if _HAS_TORCH:
        torch_dtype = _np_to_torch_dtype(dtype) if dtype is not None else torch.float32
        return torch.randn(shape, dtype=torch_dtype, device=device or get_default_device())
    return np.random.randn(*shape).astype(dtype or np.float32)


def create_full(shape, fill_value, device: Optional[str] = None, dtype=None):
    if _HAS_TORCH:
        torch_dtype = _np_to_torch_dtype(dtype) if dtype is not None else torch.float32
        return torch.full(shape, fill_value, dtype=torch_dtype, device=device or get_default_device())
    return np.full(shape, fill_value, dtype=dtype or np.float32)


def create_arange(start, end=None, step=1, device: Optional[str] = None, dtype=None):
    if _HAS_TORCH:
        torch_dtype = _np_to_torch_dtype(dtype) if dtype is not None else torch.float32
        return torch.arange(start, end, step, dtype=torch_dtype, device=device or get_default_device())
    return np.arange(start, end, step, dtype=dtype or np.float32)


def create_eye(n, m=None, device: Optional[str] = None, dtype=None):
    if _HAS_TORCH:
        torch_dtype = _np_to_torch_dtype(dtype) if dtype is not None else torch.float32
        return torch.eye(n, m, dtype=torch_dtype, device=device or get_default_device())
    return np.eye(n, m, dtype=dtype or np.float32)


def _np_to_torch_dtype(np_dtype):
    if np_dtype is None:
        return torch.float32
    if not _HAS_TORCH:
        return np_dtype
    _map = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.float16: torch.float16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.int16: torch.int16,
        np.int8: torch.int8,
        np.uint8: torch.uint8,
        np.bool_: torch.bool,
    }
    return _map.get(np_dtype, torch.float32)


def _torch_to_np_dtype(torch_dtype):
    if not _HAS_TORCH:
        return np.float32
    _map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
        torch.bfloat16: np.float32,
    }
    return _map.get(torch_dtype, np.float32)


def is_tensor(data) -> bool:
    if _HAS_TORCH and isinstance(data, torch.Tensor):
        return True
    return isinstance(data, np.ndarray)


def tensor_device(data) -> str:
    if _HAS_TORCH and isinstance(data, torch.Tensor):
        return str(data.device)
    return "cpu"


def move_to_device(data, device: str):
    if _HAS_TORCH and isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def enable_tf32(enabled: bool = True):
    if _HAS_TORCH:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled


def set_benchmark(enabled: bool = True):
    if _HAS_TORCH:
        torch.backends.cudnn.benchmark = enabled


def empty_cache():
    if _HAS_CUDA:
        torch.cuda.empty_cache()


def synchronize(device: Optional[str] = None):
    if _HAS_CUDA:
        torch.cuda.synchronize(device)
