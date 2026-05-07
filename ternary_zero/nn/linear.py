from __future__ import annotations

import numpy as np
from typing import Optional

from .module import Module, Parameter
from ..tensor import Tensor
from ..autograd.functions import Linear as LinearFn


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            np.random.randn(out_features, in_features).astype(np.float32)
            * np.sqrt(2.0 / in_features)
        )
        if bias:
            self.bias = Parameter(np.zeros(out_features, dtype=np.float32))
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        return LinearFn.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        bias_str = f", bias={self.bias is not None}"
        return f"in_features={self.in_features}, out_features={self.out_features}{bias_str}"


class BitLinear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.weight = Parameter(
            np.random.randn(out_features, in_features).astype(np.float32)
            * np.sqrt(2.0 / in_features)
        )
        if bias:
            self.bias = Parameter(np.zeros(out_features, dtype=np.float32))
        else:
            self.bias = None

        self.register_buffer("_ternary_weight", np.zeros((out_features, in_features), dtype=np.int8))
        self.register_buffer("_scale", np.float32(1.0))

    def quantize_weights(self):
        from ..quantize import ternary_quantize
        flat = self.weight.data.flatten()
        ternary, scale = ternary_quantize(
            Tensor(flat), alpha=self.alpha
        )
        self._ternary_weight = ternary.data.reshape(self.weight.shape).astype(np.int8)
        self._scale = scale.data.astype(np.float32) if isinstance(scale, Tensor) else np.float32(scale)

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            from ..autograd.functions import TernaryQuantizeSTE
            ternary, scale = TernaryQuantizeSTE.apply(self.weight, self.alpha)
            ternary_data = ternary if isinstance(ternary, np.ndarray) else ternary.data
            if isinstance(scale, Tensor):
                scale_val = float(scale.data)
            elif isinstance(scale, np.ndarray):
                scale_val = float(scale)
            else:
                scale_val = float(scale)
            w = Tensor((ternary_data.astype(np.float32) * scale_val))
        else:
            w = Tensor(self._ternary_weight.astype(np.float32) * float(self._scale))

        output = LinearFn.apply(input, w, self.bias)
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"alpha={self.alpha}, bias={self.bias is not None}"
        )
