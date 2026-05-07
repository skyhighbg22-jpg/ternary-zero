from __future__ import annotations

from typing import Optional
from collections import OrderedDict
import numpy as np

from ..tensor import Tensor


def backward(tensor: Tensor, gradient: Optional[Tensor] = None):
    if not tensor.requires_grad:
        raise RuntimeError("backward() called on a tensor that does not require grad")

    if gradient is None:
        if tensor.data.ndim == 0 or tensor.data.size == 1:
            gradient = Tensor(np.ones_like(tensor.data))
        else:
            raise RuntimeError(
                "grad can be implicitly created only for scalar outputs, "
                "consider providing a gradient argument"
            )

    tensor._ensure_grad()
    tensor.grad.data += gradient.data

    visited = set()
    topo_order = []
    _build_topo(tensor, visited, topo_order)

    for node in reversed(topo_order):
        if node._grad_fn is None:
            continue

        fn = node._grad_fn
        grad_output = node.grad

        if grad_output is None:
            continue

        grads = fn.__class__.backward(fn, grad_output.data if isinstance(grad_output, Tensor) else grad_output)

        if grads is None:
            continue
        if not isinstance(grads, tuple):
            grads = (grads,)

        inputs = fn._inputs
        for i, (inp, grad) in enumerate(zip(inputs, grads)):
            if not isinstance(inp, Tensor):
                continue
            if not inp.requires_grad:
                continue
            if grad is None:
                continue

            if isinstance(grad, (int, float, np.integer, np.floating)):
                grad = np.array(grad, dtype=np.float32)
            elif isinstance(grad, np.ndarray):
                pass
            else:
                grad = np.array(grad)

            grad_tensor = Tensor(grad)

            if grad_tensor.shape != inp.shape:
                if inp.shape == ():
                    grad_tensor = Tensor(np.array(grad_tensor.data.sum(), dtype=np.float32))
                else:
                    grad_tensor = Tensor(_unbroadcast(grad_tensor.data, inp.shape))

            inp._ensure_grad()
            inp.grad.data += grad_tensor.data


def _build_topo(node, visited, topo_order):
    node_id = id(node)
    if node_id in visited:
        return
    visited.add(node_id)

    if node._grad_fn is not None:
        for inp in node._grad_fn._inputs:
            if isinstance(inp, Tensor) and inp.requires_grad:
                _build_topo(inp, visited, topo_order)

    topo_order.append(node)


def _unbroadcast(grad, target_shape):
    if grad.shape == target_shape:
        return grad

    target_ndim = len(target_shape)
    grad_ndim = len(grad.shape)

    if grad_ndim > target_ndim:
        extra = grad_ndim - target_ndim
        grad = grad.sum(axis=tuple(range(extra)))

    reduce_axes = []
    for i, (gs, ts) in enumerate(zip(grad.shape, target_shape)):
        if ts == 1 and gs != 1:
            reduce_axes.append(i)

    if reduce_axes:
        grad = grad.sum(axis=tuple(reduce_axes), keepdims=True)

    return grad.reshape(target_shape)
