from __future__ import annotations

import numpy as np
from typing import Union, Tuple

from .module import Module
from ..tensor import Tensor
from .._backend import has_torch

if has_torch():
    import torch
    import torch.nn.functional as torchF


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return x


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)

    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            output = torchF.max_pool2d(
                input.data,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            return Tensor(output)

        x = input.data
        kh, kw = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        if pad_h > 0 or pad_w > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
                constant_values=-np.inf,
            )

        batch, channels, h, w = x.shape
        out_h = (h - kh) // stride_h + 1
        out_w = (w - kw) // stride_w + 1

        output = np.zeros((batch, channels, out_h, out_w), dtype=np.float32)

        for oh in range(out_h):
            for ow in range(out_w):
                h_start = oh * stride_h
                w_start = ow * stride_w
                patch = x[:, :, h_start:h_start + kh, w_start:w_start + kw]
                output[:, :, oh, ow] = np.max(
                    patch.reshape(batch, channels, -1), axis=2
                )

        return Tensor(output)

    def extra_repr(self) -> str:
        s = f"kernel_size={self.kernel_size}, stride={self.stride}"
        if self.padding != (0, 0):
            s += f", padding={self.padding}"
        return s


class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)

    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            output = torchF.avg_pool2d(
                input.data,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            return Tensor(output)

        x = input.data
        kh, kw = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        if pad_h > 0 or pad_w > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
            )

        batch, channels, h, w = x.shape
        out_h = (h - kh) // stride_h + 1
        out_w = (w - kw) // stride_w + 1

        output = np.zeros((batch, channels, out_h, out_w), dtype=np.float32)

        for oh in range(out_h):
            for ow in range(out_w):
                h_start = oh * stride_h
                w_start = ow * stride_w
                patch = x[:, :, h_start:h_start + kh, w_start:w_start + kw]
                output[:, :, oh, ow] = np.mean(
                    patch.reshape(batch, channels, -1), axis=2
                )

        return Tensor(output)

    def extra_repr(self) -> str:
        s = f"kernel_size={self.kernel_size}, stride={self.stride}"
        if self.padding != (0, 0):
            s += f", padding={self.padding}"
        return s


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.output_size = _pair(output_size)

    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            output = torchF.adaptive_avg_pool2d(input.data, self.output_size)
            return Tensor(output)

        x = input.data
        batch, channels, h, w = x.shape
        out_h, out_w = self.output_size

        stride_h = h // out_h
        stride_w = w // out_w
        kernel_h = h - (out_h - 1) * stride_h
        kernel_w = w - (out_w - 1) * stride_w

        output = np.zeros((batch, channels, out_h, out_w), dtype=np.float32)

        for oh in range(out_h):
            for ow in range(out_w):
                h_start = oh * stride_h
                w_start = ow * stride_w
                patch = x[:, :, h_start:h_start + kernel_h, w_start:w_start + kernel_w]
                output[:, :, oh, ow] = np.mean(
                    patch.reshape(batch, channels, -1), axis=2
                )

        return Tensor(output)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class GlobalAvgPool2d(Module):
    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            return Tensor(torch.mean(input.data, dim=(2, 3)))
        x = input.data
        return Tensor(np.mean(x, axis=(2, 3), keepdims=False).astype(np.float32))

    def extra_repr(self) -> str:
        return ""
