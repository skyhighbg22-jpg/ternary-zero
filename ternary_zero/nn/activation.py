from __future__ import annotations

import numpy as np

from .module import Module
from ..tensor import Tensor
from .._backend import has_torch

if has_torch():
    import torch
    import torch.nn.functional as torchF


class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            return Tensor(torch.relu(input.data))
        from ..autograd.functions import ReLU as ReLUFn
        return ReLUFn.apply(input)

    def extra_repr(self) -> str:
        return "inplace=False"


class GELU(Module):
    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            return Tensor(torchF.gelu(input.data))
        x = input.data
        c = np.sqrt(2.0 / np.pi)
        result = 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))
        return Tensor(result.astype(np.float32))


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            return Tensor(torch.sigmoid(input.data))
        return Tensor((1.0 / (1.0 + np.exp(-input.data))).astype(np.float32))


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            return Tensor(torch.tanh(input.data))
        return Tensor(np.tanh(input.data).astype(np.float32))


class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            return Tensor(torchF.softmax(input.data.float(), dim=self.dim))
        from ..autograd.functions import Softmax as SoftmaxFn
        return SoftmaxFn.apply(input, dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class LogSoftmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            return Tensor(torchF.log_softmax(input.data.float(), dim=self.dim))
        from ..autograd.functions import Softmax as SoftmaxFn, Log as LogFn
        sm = SoftmaxFn.apply(input, dim=self.dim)
        return LogFn.apply(sm)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
