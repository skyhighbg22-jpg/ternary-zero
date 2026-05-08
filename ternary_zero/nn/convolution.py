from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union

from .module import Module, Parameter
from ..tensor import Tensor
from .._backend import has_torch

if has_torch():
    import torch
    import torch.nn.functional as torchF


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return x


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        kh, kw = self.kernel_size
        fan_in = in_channels * kh * kw // groups
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels // groups, kh, kw).astype(np.float32)
            * np.sqrt(2.0 / fan_in)
        )
        if bias:
            self.bias = Parameter(np.zeros(out_channels, dtype=np.float32))
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            w = self.weight.data
            b = self.bias.data if self.bias is not None else None
            output = torchF.conv2d(
                input.data, w, b,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            return Tensor(output)

        x = input.data
        w = self.weight.data
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        if pad_h > 0 or pad_w > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
            )

        batch, in_c, h, w_in = x.shape
        out_c, _, kh, kw = w.shape
        out_h = (h - kh) // stride_h + 1
        out_w = (w_in - kw) // stride_w + 1

        output = np.zeros((batch, out_c, out_h, out_w), dtype=np.float32)

        for b in range(batch):
            for oc in range(out_c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * stride_h
                        w_start = ow * stride_w
                        patch = x[b, :, h_start:h_start + kh, w_start:w_start + kw]
                        output[b, oc, oh, ow] = np.sum(patch * w[oc])
                        if self.bias is not None:
                            output[b, oc, oh, ow] += self.bias.data[oc]

        return Tensor(output)

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding != (0, 0):
            s += f", padding={self.padding}"
        if self.bias is None:
            s += ", bias=False"
        return s


class Conv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, kernel_size).astype(np.float32)
            * np.sqrt(2.0 / fan_in)
        )
        if bias:
            self.bias = Parameter(np.zeros(out_channels, dtype=np.float32))
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        if has_torch() and isinstance(input.data, torch.Tensor):
            w = self.weight.data
            b = self.bias.data if self.bias is not None else None
            output = torchF.conv1d(
                input.data, w, b,
                stride=self.stride,
                padding=self.padding,
            )
            return Tensor(output)

        x = input.data
        w = self.weight.data

        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)), mode="constant")

        batch, in_c, length = x.shape
        out_c, _, kw = w.shape
        out_len = (length - kw) // self.stride + 1

        output = np.zeros((batch, out_c, out_len), dtype=np.float32)

        for b in range(batch):
            for oc in range(out_c):
                for ol in range(out_len):
                    start = ol * self.stride
                    patch = x[b, :, start:start + kw]
                    output[b, oc, ol] = np.sum(patch * w[oc])
                    if self.bias is not None:
                        output[b, oc, ol] += self.bias.data[oc]

        return Tensor(output)

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding != 0:
            s += f", padding={self.padding}"
        if self.bias is None:
            s += ", bias=False"
        return s
