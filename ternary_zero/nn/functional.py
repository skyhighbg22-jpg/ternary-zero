from __future__ import annotations

import numpy as np
from typing import Optional

from ..tensor import Tensor
from .._backend import has_torch

if has_torch():
    import torch
    import torch.nn.functional as torchF

from ..autograd.functions import (
    ReLU as ReLUFn,
    Softmax as SoftmaxFn,
    Log as LogFn,
    MatMul,
    CrossEntropyLoss as CE,
    MSELoss as MSEFn,
    Linear as LinearFn,
)


def relu(input: Tensor) -> Tensor:
    if has_torch() and isinstance(input.data, torch.Tensor):
        return Tensor(torch.relu(input.data))
    return ReLUFn.apply(input)


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    if has_torch() and isinstance(input.data, torch.Tensor):
        return Tensor(torchF.softmax(input.data.float(), dim=dim))
    return SoftmaxFn.apply(input, dim=dim)


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    if has_torch() and isinstance(input.data, torch.Tensor):
        return Tensor(torchF.log_softmax(input.data.float(), dim=dim))
    return LogFn.apply(SoftmaxFn.apply(input, dim=dim))


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    if has_torch() and isinstance(input.data, torch.Tensor):
        b = bias.data if bias is not None else None
        return Tensor(torchF.linear(input.data, weight.data, b))
    return LinearFn.apply(input, weight, bias)


def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    if has_torch() and isinstance(input.data, torch.Tensor):
        target_data = target.data.long() if isinstance(target.data, torch.Tensor) else torch.tensor(target.data, dtype=torch.long, device=input.data.device)
        loss = torchF.cross_entropy(input.data, target_data, weight=weight, reduction=reduction)
        return Tensor(loss)
    return CE.apply(input, target)


def mse_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    if has_torch() and isinstance(input.data, torch.Tensor):
        target_data = target.data if isinstance(target.data, torch.Tensor) else torch.tensor(target.data, device=input.data.device)
        loss = torchF.mse_loss(input.data, target_data, reduction=reduction)
        return Tensor(loss)
    return MSEFn.apply(input, target)


def nll_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    if has_torch() and isinstance(input.data, torch.Tensor):
        target_data = target.data.long() if isinstance(target.data, torch.Tensor) else torch.tensor(target.data, dtype=torch.long, device=input.data.device)
        loss = torchF.nll_loss(input.data, target_data, reduction=reduction)
        return Tensor(loss)
    if target.ndim == 1:
        batch_size = input.shape[0]
        loss = -input.data[np.arange(batch_size), target.data.astype(int)]
    else:
        loss = -(input.data * target.data).sum(axis=-1)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return Tensor(np.float32(loss))


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0.0:
        return input
    if has_torch() and isinstance(input.data, torch.Tensor):
        return Tensor(torch.nn.functional.dropout(input.data, p=p, training=True))
    mask = (np.random.random(input.shape) > p).astype(input.data.dtype)
    return Tensor(input.data * mask / (1.0 - p))


def gelu(input: Tensor) -> Tensor:
    if has_torch() and isinstance(input.data, torch.Tensor):
        return Tensor(torchF.gelu(input.data))
    x = input.data
    c = np.sqrt(2.0 / np.pi)
    result = 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))
    return Tensor(result.astype(np.float32))


def sigmoid(input: Tensor) -> Tensor:
    if has_torch() and isinstance(input.data, torch.Tensor):
        return Tensor(torch.sigmoid(input.data))
    return Tensor(1.0 / (1.0 + np.exp(-input.data)))


def tanh(input: Tensor) -> Tensor:
    if has_torch() and isinstance(input.data, torch.Tensor):
        return Tensor(torch.tanh(input.data))
    return Tensor(np.tanh(input.data))


def embedding(input: Tensor, weight: Tensor) -> Tensor:
    if has_torch() and isinstance(input.data, torch.Tensor):
        return Tensor(torch.nn.functional.embedding(input.data.long(), weight.data))
    indices = input.data.astype(int)
    return Tensor(weight.data[indices])


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    if has_torch() and isinstance(input.data, torch.Tensor):
        b = bias.data if bias is not None else None
        output = torchF.conv2d(input.data, weight.data, b, stride=stride, padding=padding)
        return Tensor(output)

    x = input.data
    w = weight.data
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")

    batch, in_c, h, h_w = x.shape
    out_c, _, kh, kw = w.shape
    out_h = (h - kh) // stride + 1
    out_w = (h_w - kw) // stride + 1

    output = np.zeros((batch, out_c, out_h, out_w), dtype=np.float32)
    for b in range(batch):
        for oc in range(out_c):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride
                    w_start = ow * stride
                    patch = x[b, :, h_start:h_start + kh, w_start:w_start + kw]
                    output[b, oc, oh, ow] = np.sum(patch * w[oc])
                    if bias is not None:
                        output[b, oc, oh, ow] += bias.data[oc]

    return Tensor(output)
