from __future__ import annotations

import numpy as np

from .module import Module
from ..tensor import Tensor
from ..autograd.functions import CrossEntropyLoss as CELoss, MSELoss as MSEFn
from .._backend import has_torch

if has_torch():
    import torch
    import torch.nn.functional as torchF


class CrossEntropyLoss(Module):
    def __init__(self, weight=None, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            target_data = target.data.long() if isinstance(target.data, torch.Tensor) else torch.tensor(target.data, dtype=torch.long, device=input.data.device)
            loss = torchF.cross_entropy(input.data, target_data, weight=self.weight)
            return Tensor(loss)
        return CELoss.apply(input, target)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"


class MSELoss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            loss = torchF.mse_loss(input.data, target.data if isinstance(target.data, torch.Tensor) else torch.tensor(target.data, device=input.data.device))
            return Tensor(loss)
        return MSEFn.apply(input, target)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"


class L1Loss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            loss = torchF.l1_loss(input.data, target.data if isinstance(target.data, torch.Tensor) else torch.tensor(target.data, device=input.data.device))
            return Tensor(loss)
        diff = input.data - target.data
        if self.reduction == "mean":
            loss = np.mean(np.abs(diff))
        elif self.reduction == "sum":
            loss = np.sum(np.abs(diff))
        else:
            loss = np.abs(diff)
        return Tensor(np.float32(loss))
