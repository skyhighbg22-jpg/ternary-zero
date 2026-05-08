from __future__ import annotations

import numpy as np

from .module import Module
from ..tensor import Tensor
from .._backend import has_torch

if has_torch():
    import torch


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return input
        if has_torch() and isinstance(input.data, torch.Tensor):
            output = torch.nn.functional.dropout(input.data, p=self.p, training=True)
            return Tensor(output)
        mask = (np.random.random(input.shape) > self.p).astype(input.data.dtype)
        return Tensor((input.data * mask / (1.0 - self.p)).astype(np.float32))

    def extra_repr(self) -> str:
        return f"p={self.p}"
