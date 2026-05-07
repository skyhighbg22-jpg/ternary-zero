from __future__ import annotations

import numpy as np

from .module import Module
from ..tensor import Tensor
from ..autograd.functions import CrossEntropyLoss as CELoss, MSELoss as MSEFn


class CrossEntropyLoss(Module):
    def __init__(self, weight=None, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return CELoss.apply(input, target)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"


class MSELoss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return MSEFn.apply(input, target)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"


class L1Loss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input.data - target.data
        if self.reduction == "mean":
            loss = np.mean(np.abs(diff))
        elif self.reduction == "sum":
            loss = np.sum(np.abs(diff))
        else:
            loss = np.abs(diff)
        return Tensor(np.float32(loss))
