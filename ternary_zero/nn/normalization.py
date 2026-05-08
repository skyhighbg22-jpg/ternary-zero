from __future__ import annotations

import numpy as np

from .module import Module, Parameter
from ..tensor import Tensor
from .._backend import has_torch

if has_torch():
    import torch


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape, dtype=np.float32))
            self.bias = Parameter(np.zeros(self.normalized_shape, dtype=np.float32))
        else:
            self.weight = None
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            w = self.weight.data if self.elementwise_affine else None
            b = self.bias.data if self.elementwise_affine else None
            output = torch.nn.functional.layer_norm(
                input.data, self.normalized_shape, w, b, self.eps
            )
            return Tensor(output)

        x = input.data
        axis = tuple(range(-len(self.normalized_shape), 0))
        mean = np.mean(x, axis=axis, keepdims=True)
        var = np.var(x, axis=axis, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = self.weight.data * x_norm + self.bias.data
        return Tensor(x_norm.astype(np.float32))

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(np.ones(num_features, dtype=np.float32))
        self.bias = Parameter(np.zeros(num_features, dtype=np.float32))
        self.register_buffer("running_mean", np.zeros(num_features, dtype=np.float32))
        self.register_buffer("running_var", np.ones(num_features, dtype=np.float32))

    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            device = input.data.device
            w = self.weight.data
            b = self.bias.data
            rm = self.running_mean
            rv = self.running_var
            if isinstance(rm, np.ndarray):
                rm = torch.from_numpy(rm).to(device)
                rv = torch.from_numpy(rv).to(device)
            if self.training:
                output = torch.nn.functional.batch_norm(
                    input.data, rm, rv, w, b, self.training, self.momentum, self.eps
                )
                with torch.no_grad():
                    mean = input.data.mean(dim=0)
                    var = input.data.var(dim=0, unbiased=False)
                    self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * mean.detach().cpu().numpy())
                    self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * var.detach().cpu().numpy())
            else:
                output = torch.nn.functional.batch_norm(
                    input.data, rm, rv, w, b, False, self.momentum, self.eps
                )
            return Tensor(output)

        x = input.data
        if self.training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.weight.data * x_norm + self.bias.data
        return Tensor(out.astype(np.float32))

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}"
