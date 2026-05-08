from __future__ import annotations

_grad_enabled = True

from ._backend import (
    get_default_device,
    set_default_device,
    is_cuda_available,
    has_torch,
    num_gpus,
    autocast as _autocast,
    CudaStream as _CudaStream,
    enable_tf32 as _enable_tf32,
    set_benchmark as _set_benchmark,
    empty_cache as _empty_cache,
    synchronize as _synchronize,
    gpu_memory_info as _gpu_memory_info,
)


def is_grad_enabled():
    return _grad_enabled


def _set_grad_enabled(enabled: bool):
    global _grad_enabled
    _grad_enabled = enabled


class enable_grad:
    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = True
        if has_torch():
            import torch
            self._torch_ctx = torch.enable_grad()
            self._torch_ctx.__enter__()
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self.prev
        if has_torch():
            self._torch_ctx.__exit__(*args)


class no_grad:
    def __init__(self):
        pass

    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False
        if has_torch():
            import torch
            self._torch_ctx = torch.no_grad()
            self._torch_ctx.__enter__()
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self.prev
        if has_torch():
            self._torch_ctx.__exit__(*args)


class autocast(_autocast):
    pass


class cuda_stream(_CudaStream):
    pass


def enable_tf32(enabled: bool = True):
    _enable_tf32(enabled)


def set_cudnn_benchmark(enabled: bool = True):
    _set_benchmark(enabled)


def empty_cuda_cache():
    _empty_cache()


def cuda_synchronize(device=None):
    _synchronize(device)


def cuda_memory_info(device=None):
    return _gpu_memory_info(device)
