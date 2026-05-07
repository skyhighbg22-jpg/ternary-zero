from __future__ import annotations

import numpy as np

from .module import Module
from ..tensor import Tensor


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return input
        mask = (np.random.random(input.shape) > self.p).astype(input.data.dtype)
        return Tensor((input.data * mask / (1.0 - self.p)).astype(np.float32))

    def extra_repr(self) -> str:
        return f"p={self.p}"
