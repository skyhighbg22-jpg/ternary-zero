from __future__ import annotations

from typing import Optional, Tuple, Any
import numpy as np

from ..tensor import Tensor
from .._config import is_grad_enabled
from .._backend import has_torch, to_numpy, get_default_device

if has_torch():
    import torch


class Function:
    _version: int = 1
    _saved_tensors: tuple = ()

    def __init__(self):
        self.saved_tensors: tuple = ()
        self.needs_input_grad: Tuple[bool, ...] = ()
        self.ctx: dict = {}

    @staticmethod
    def forward(ctx, *inputs, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs, **kwargs):
        if not is_grad_enabled():
            raw_inputs = []
            for inp in inputs:
                if isinstance(inp, Tensor):
                    raw_inputs.append(inp.data)
                else:
                    raw_inputs.append(inp)
            result = cls._raw_forward(*raw_inputs, **kwargs)
            if isinstance(result, tuple):
                return tuple(Tensor(r) for r in result)
            if has_torch() and isinstance(result, torch.Tensor):
                return Tensor(result.detach())
            if isinstance(result, np.ndarray):
                return Tensor(result)
            return Tensor(result)

        ctx = cls()
        ctx._version = cls._version
        ctx.needs_input_grad = tuple(
            isinstance(t, Tensor) and t.requires_grad for t in inputs
        )
        ctx._inputs = inputs
        ctx._kwargs = kwargs

        raw_inputs = [t.data if isinstance(t, Tensor) else t for t in inputs]

        result_data = cls._forward(ctx, *raw_inputs, **kwargs)

        if has_torch():
            if isinstance(result_data, tuple):
                result = tuple(Tensor(d, requires_grad=any(ctx.needs_input_grad)) for d in result_data)
            elif isinstance(result_data, torch.Tensor):
                result = Tensor(result_data, requires_grad=any(ctx.needs_input_grad))
            else:
                result = Tensor(result_data, requires_grad=any(ctx.needs_input_grad))
        else:
            if isinstance(result_data, np.ndarray):
                result = Tensor(result_data, requires_grad=any(ctx.needs_input_grad), _grad_fn=ctx)
            elif isinstance(result_data, tuple):
                result = tuple(
                    Tensor(d, requires_grad=any(ctx.needs_input_grad), _grad_fn=ctx)
                    for d in result_data
                )
            else:
                result = Tensor(result_data, requires_grad=any(ctx.needs_input_grad), _grad_fn=ctx)

        ctx._output = result
        return result

    @staticmethod
    def _raw_forward(*inputs, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _forward(ctx, *inputs, **kwargs):
        raise NotImplementedError

    @classmethod
    def _validate_version(cls, ctx):
        stored = getattr(ctx, '_version', None)
        if stored is not None and stored != cls._version:
            raise RuntimeError(
                f"Version mismatch for {cls.__name__}: "
                f"forward used v{stored}, but current is v{cls._version}. "
                f"Re-run the forward pass before calling backward."
            )
