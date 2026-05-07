from __future__ import annotations

import numpy as np
from typing import Optional

from ..tensor import Tensor
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
    return ReLUFn.apply(input)


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    return SoftmaxFn.apply(input, dim=dim)


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    return LogFn.apply(SoftmaxFn.apply(input, dim=dim))


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    return LinearFn.apply(input, weight, bias)


def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    return CE.apply(input, target)


def mse_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    return MSEFn.apply(input, target)


def nll_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    import numpy as np
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
    mask = (np.random.random(input.shape) > p).astype(input.data.dtype)
    return Tensor(input.data * mask / (1.0 - p))


def gelu(input: Tensor) -> Tensor:
    import numpy as np
    x = input.data
    c = np.sqrt(2.0 / np.pi)
    result = 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))
    return Tensor(result.astype(np.float32))


def sigmoid(input: Tensor) -> Tensor:
    import numpy as np
    return Tensor(1.0 / (1.0 + np.exp(-input.data)))


def tanh(input: Tensor) -> Tensor:
    import numpy as np
    return Tensor(np.tanh(input.data))


def embedding(input: Tensor, weight: Tensor) -> Tensor:
    import numpy as np
    indices = input.data.astype(int)
    return Tensor(weight.data[indices])


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    import numpy as np
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
