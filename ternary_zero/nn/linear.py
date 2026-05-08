from __future__ import annotations

import numpy as np
from typing import Optional

from .module import Module, Parameter
from ..tensor import Tensor
from ..autograd.functions import Linear as LinearFn
from .._backend import has_torch, get_default_device, create_randn, create_zeros

if has_torch():
    import torch
    import torch.nn.functional as torchF


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w_data = (np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)).astype(np.float32)
        self.weight = Parameter(w_data)
        if bias:
            self.bias = Parameter(np.zeros(out_features, dtype=np.float32))
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            w = self.weight.data
            b = self.bias.data if self.bias is not None else None
            output = torchF.linear(input.data, w, b)
            return Tensor(output)
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

        w_data = (np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)).astype(np.float32)
        self.weight = Parameter(w_data)
        if bias:
            self.bias = Parameter(np.zeros(out_features, dtype=np.float32))
        else:
            self.bias = None

        self.register_buffer("_ternary_weight", np.zeros((out_features, in_features), dtype=np.int8))
        self.register_buffer("_scale", np.float32(1.0))

    def quantize_weights(self):
        from ..quantize import ternary_quantize
        ternary, scale = ternary_quantize(
            Tensor(self.weight.data.flatten()), alpha=self.alpha
        )
        tw = ternary.data
        if has_torch() and isinstance(tw, torch.Tensor):
            tw = tw.detach().cpu().numpy()
        self._ternary_weight = tw.reshape(self.weight.shape).astype(np.int8)
        if isinstance(scale, Tensor):
            scale_val = scale.data
            if has_torch() and isinstance(scale_val, torch.Tensor):
                scale_val = scale_val.item()
            self._scale = np.float32(scale_val)
        elif isinstance(scale, (int, float)):
            self._scale = np.float32(scale)
        else:
            self._scale = np.float32(float(scale))

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            from ..autograd.functions import TernaryQuantizeSTE
            ternary, scale = TernaryQuantizeSTE.apply(self.weight, self.alpha)
            ternary_data = ternary if isinstance(ternary, (np.ndarray,)) else ternary.data
            if isinstance(scale, Tensor):
                scale_val = scale.data
                if has_torch() and isinstance(scale_val, torch.Tensor):
                    scale_val = scale_val.item()
            elif isinstance(scale, (int, float)):
                scale_val = float(scale)
            else:
                if has_torch() and isinstance(scale, torch.Tensor):
                    scale_val = scale.item()
                else:
                    scale_val = float(scale)

            if has_torch() and isinstance(ternary_data, torch.Tensor):
                w = (ternary_data.float() * scale_val)
            else:
                w = Tensor((ternary_data.astype(np.float32) * scale_val))
        else:
            tw = self._ternary_weight
            sc = self._scale
            if has_torch() and isinstance(self.weight.data, torch.Tensor):
                device = self.weight.data.device
                if isinstance(tw, np.ndarray):
                    tw = torch.from_numpy(tw).to(device)
                if isinstance(sc, np.ndarray):
                    sc = torch.tensor(sc, device=device)
                w = tw.float() * sc.float()
            else:
                w = Tensor(self._ternary_weight.astype(np.float32) * float(self._scale))

        if has_torch() and isinstance(input.data, torch.Tensor):
            weight_data = w if isinstance(w, torch.Tensor) else w.data
            if isinstance(weight_data, Tensor):
                weight_data = weight_data.data
            bias_data = self.bias.data if self.bias is not None else None
            output = torchF.linear(input.data, weight_data, bias_data)
            return Tensor(output)

        w_tensor = w if isinstance(w, Tensor) else Tensor(w)
        output = LinearFn.apply(input, w_tensor, self.bias)
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"alpha={self.alpha}, bias={self.bias is not None}"
        )
