from __future__ import annotations

import numpy as np
from typing import Optional

from .module import Module, Parameter
from ..tensor import Tensor
from ..autograd.functions import Linear as LinearFn
from .._backend import has_torch, get_default_device, create_randn, create_zeros, to_numpy

if has_torch():
    import torch
    import torch.nn.functional as torchF


def _has_native():
    try:
        from ternary_zero import _core
        return True
    except ImportError:
        return False


def _native_has_cuda():
    try:
        from ternary_zero import _core
        return _core.has_cuda()
    except (ImportError, AttributeError):
        return False


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
    """Production-ready ternary quantized linear layer.

    Drop-in replacement for ``torch.nn.Linear`` that internally uses
    2-bit packed ternary weights {-1, 0, 1} with a scaling factor.

    Inference dispatch priority:
      1. CUDA GPU kernel (packed ternary GEMV via _core.ternary_gemv_gpu)
      2. CPU packed ternary GEMV via Rust (``_core.ternary_gemv_cpu_packed``)
      3. NumPy-vectorized fallback (``quantize.ternary_gemv_numpy``)

    Training uses Straight-Through Estimator (STE) quantization to
    maintain differentiability through the hard ternary rounding.
    """

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
        self.register_buffer("_packed_weights", np.zeros(0, dtype=np.uint32))
        self._packed_ready = False

    @classmethod
    def from_linear(cls, linear: "Linear", alpha: float = 0.5) -> "BitLinear":
        """Create a BitLinear from an existing Linear layer's weights."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            alpha=alpha,
        )
        w_np = to_numpy(linear.weight.data).copy()
        if has_torch() and isinstance(linear.weight.data, torch.Tensor):
            w_np = linear.weight.data.detach().cpu().numpy()
        layer.weight = Parameter(w_np)
        if linear.bias is not None:
            b_np = to_numpy(linear.bias.data).copy()
            if has_torch() and isinstance(linear.bias.data, torch.Tensor):
                b_np = linear.bias.data.detach().cpu().numpy()
            layer.bias = Parameter(b_np)
        return layer

    def quantize_weights(self):
        """Pre-quantize and cache ternary weights for inference."""
        from ..quantize import ternary_quantize, pack_ternary_to_u32

        ternary, scale = ternary_quantize(
            Tensor(self.weight.data.flatten()), alpha=self.alpha
        )
        tw = ternary.data
        if has_torch() and isinstance(tw, torch.Tensor):
            tw = tw.detach().cpu().numpy()
        tw = tw.reshape(self.weight.shape).astype(np.int8)
        self._ternary_weight = tw

        if isinstance(scale, Tensor):
            scale_val = scale.data
            if has_torch() and isinstance(scale_val, torch.Tensor):
                scale_val = scale_val.item()
            self._scale = np.float32(scale_val)
        elif isinstance(scale, (int, float)):
            self._scale = np.float32(scale)
        else:
            self._scale = np.float32(float(scale))

        n = self.in_features
        if n % 16 == 0:
            packed = pack_ternary_to_u32(Tensor(tw.flatten()), n)
            pw_data = packed.data
            if has_torch() and isinstance(pw_data, torch.Tensor):
                pw_data = pw_data.detach().cpu().numpy()
            self._packed_weights = pw_data.astype(np.uint32).ravel()
            self._packed_ready = True
        else:
            self._packed_ready = False

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

    def forward_native(self, input: Tensor) -> Tensor:
        """Forward pass using native packed ternary kernels.

        Uses the packed u32 ternary weight format directly, avoiding
        the dequantize-to-float path. Dispatches to the fastest
        available backend (GPU > CPU Rust > NumPy).

        Requires ``quantize_weights()`` to have been called first.
        """
        if not self._packed_ready:
            self.quantize_weights()

        input_np = to_numpy(input.data)
        if has_torch() and isinstance(input.data, torch.Tensor):
            input_np = input.data.detach().cpu().numpy()
        input_flat = input_np.ravel().astype(np.float32)

        m = self.out_features
        n = self.in_features
        scale = float(self._scale)

        try:
            from ternary_zero import _core
            if _core.has_cuda():
                result = _core.ternary_gemv_gpu(
                    self._packed_weights, input_flat, m, n
                )
                output = (result * scale).reshape(*input_np.shape[:-1], m)
                out = Tensor(output.astype(np.float32))
                if self.bias is not None:
                    bias_np = to_numpy(self.bias.data)
                    if has_torch() and isinstance(self.bias.data, torch.Tensor):
                        bias_np = self.bias.data.detach().cpu().numpy()
                    out = out + Tensor(bias_np)
                return out
            else:
                result = _core.ternary_gemv_cpu_packed(
                    self._packed_weights, input_flat, m, n
                )
                output = (result * scale).reshape(*input_np.shape[:-1], m)
                out = Tensor(output.astype(np.float32))
                if self.bias is not None:
                    bias_np = to_numpy(self.bias.data)
                    if has_torch() and isinstance(self.bias.data, torch.Tensor):
                        bias_np = self.bias.data.detach().cpu().numpy()
                    out = out + Tensor(bias_np)
                return out
        except ImportError:
            from ..quantize import ternary_gemv_numpy
            result = ternary_gemv_numpy(
                self._packed_weights, input_flat, m, n
            )
            output = (result * scale).reshape(*input_np.shape[:-1], m)
            out = Tensor(output.astype(np.float32))
            if self.bias is not None:
                bias_np = to_numpy(self.bias.data)
                if has_torch() and isinstance(self.bias.data, torch.Tensor):
                    bias_np = self.bias.data.detach().cpu().numpy()
                out = out + Tensor(bias_np)
            return out

    def extra_repr(self) -> str:
        backend = "unknown"
        if _native_has_cuda():
            backend = "cuda"
        elif _has_native():
            backend = "cpu-rust"
        else:
            backend = "numpy"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"alpha={self.alpha}, bias={self.bias is not None}, backend={backend}"
        )
