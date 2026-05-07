from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union, List

from ._config import is_grad_enabled as _is_grad_enabled


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_grad_fn", "_backward_hooks", "_version")

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _grad_fn: Optional["Function"] = None,
    ):
        if isinstance(data, Tensor):
            data = data.data
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data: np.ndarray = data.astype(data.dtype) if data.dtype != np.float32 else data
        self.requires_grad: bool = requires_grad and _is_grad_enabled()
        self.grad: Optional["Tensor"] = None
        self._grad_fn: Optional["Function"] = _grad_fn
        self._backward_hooks: dict = {}
        self._version: int = 0

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def T(self) -> "Tensor":
        return self.transpose(-2, -1)

    @property
    def grad_fn(self):
        return self._grad_fn

    @property
    def is_leaf(self) -> bool:
        return self._grad_fn is None

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    def dim(self) -> int:
        return self.ndim

    def numel(self) -> int:
        return self.data.size

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def clone(self) -> "Tensor":
        t = Tensor(self.data.copy(), requires_grad=self.requires_grad)
        if self.grad is not None:
            t.grad = self.grad.clone()
        return t

    def contiguous(self) -> "Tensor":
        if not self.data.flags["C_CONTIGUOUS"]:
            return Tensor(np.ascontiguousarray(self.data), requires_grad=self.requires_grad, _grad_fn=self._grad_fn)
        return self

    def to(self, device: str) -> "Tensor":
        if device == self.device:
            return self
        return Tensor(self.data.copy(), requires_grad=self.requires_grad, _grad_fn=self._grad_fn)

    def float(self) -> "Tensor":
        return Tensor(self.data.astype(np.float32), requires_grad=self.requires_grad, _grad_fn=self._grad_fn)

    def half(self) -> "Tensor":
        return Tensor(self.data.astype(np.float16), requires_grad=self.requires_grad, _grad_fn=self._grad_fn)

    def long(self) -> "Tensor":
        return Tensor(self.data.astype(np.int64), requires_grad=False)

    def int(self) -> "Tensor":
        return Tensor(self.data.astype(np.int32), requires_grad=False)

    def bool(self) -> "Tensor":
        return Tensor(self.data.astype(bool), requires_grad=False)

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self):
        return self.data.item()

    def fill_(self, value):
        self.data.fill(value)
        self._version += 1
        return self

    def zero_(self):
        self.data.fill(0)
        self._version += 1
        return self

    def uniform_(self, low=0.0, high=1.0):
        self.data[:] = np.random.uniform(low, high, self.shape)
        self._version += 1
        return self

    def normal_(self, mean=0.0, std=1.0):
        self.data[:] = np.random.normal(mean, std, self.shape)
        self._version += 1
        return self

    def _ensure_grad(self):
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, gradient: Optional["Tensor"] = None):
        from .autograd import engine
        engine.backward(self, gradient)

    def sum(self, dim=None, keepdim=False) -> "Tensor":
        from .autograd.functions import Sum
        return Sum.apply(self, dim=dim, keepdim=keepdim)

    def mean(self, dim=None, keepdim=False) -> "Tensor":
        from .autograd.functions import Mean
        return Mean.apply(self, dim=dim, keepdim=keepdim)

    def view(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        from .autograd.functions import Reshape
        return Reshape.apply(self, shape=shape)

    def reshape(self, *shape) -> "Tensor":
        return self.view(*shape)

    def transpose(self, dim0, dim1) -> "Tensor":
        from .autograd.functions import Transpose
        return Transpose.apply(self, dim0=dim0, dim1=dim1)

    def permute(self, *dims) -> "Tensor":
        from .autograd.functions import Permute
        return Permute.apply(self, dims=dims)

    def unsqueeze(self, dim) -> "Tensor":
        from .autograd.functions import Unsqueeze
        return Unsqueeze.apply(self, dim=dim)

    def squeeze(self, dim=None) -> "Tensor":
        from .autograd.functions import Squeeze
        return Squeeze.apply(self, dim=dim)

    def flatten(self, start_dim=0, end_dim=-1) -> "Tensor":
        shape = list(self.shape)
        if end_dim < 0:
            end_dim = len(shape) + end_dim
        new_shape = shape[:start_dim] + [-1] + shape[end_dim + 1:]
        return self.view(new_shape)

    def matmul(self, other: "Tensor") -> "Tensor":
        from .autograd.functions import MatMul
        return MatMul.apply(self, other)

    def mm(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def add(self, other, alpha=1.0) -> "Tensor":
        from .autograd.functions import Add
        if isinstance(other, (int, float)):
            other = Tensor(np.full(self.shape, other * alpha, dtype=self.data.dtype))
        elif alpha != 1.0:
            other = Tensor(other.data * alpha)
        return Add.apply(self, other)

    def sub(self, other, alpha=1.0) -> "Tensor":
        from .autograd.functions import Sub
        if isinstance(other, (int, float)):
            other = Tensor(np.full(self.shape, other * alpha, dtype=self.data.dtype))
        elif alpha != 1.0:
            other = Tensor(other.data * alpha)
        return Sub.apply(self, other)

    def mul(self, other) -> "Tensor":
        from .autograd.functions import Mul
        if isinstance(other, (int, float)):
            other = Tensor(np.full(self.shape, other, dtype=self.data.dtype))
        return Mul.apply(self, other)

    def div(self, other) -> "Tensor":
        from .autograd.functions import Div
        if isinstance(other, (int, float)):
            other = Tensor(np.full(self.shape, other, dtype=self.data.dtype))
        return Div.apply(self, other)

    def pow(self, exponent) -> "Tensor":
        from .autograd.functions import Pow
        if isinstance(exponent, (int, float)):
            exponent = Tensor(np.full(self.shape, exponent, dtype=self.data.dtype))
        return Pow.apply(self, exponent)

    def neg(self) -> "Tensor":
        from .autograd.functions import Neg
        return Neg.apply(self)

    def abs(self) -> "Tensor":
        from .autograd.functions import Abs
        return Abs.apply(self)

    def relu(self) -> "Tensor":
        from .autograd.functions import ReLU
        return ReLU.apply(self)

    def softmax(self, dim=-1) -> "Tensor":
        from .autograd.functions import Softmax
        return Softmax.apply(self, dim=dim)

    def log(self) -> "Tensor":
        from .autograd.functions import Log
        return Log.apply(self)

    def exp(self) -> "Tensor":
        from .autograd.functions import Exp
        return Exp.apply(self)

    def max(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(np.max(self.data))
        from .autograd.functions import Max
        return Max.apply(self, dim=dim, keepdim=keepdim)

    def min(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(np.min(self.data))
        from .autograd.functions import Min
        return Min.apply(self, dim=dim, keepdim=keepdim)

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full(self.shape, other, dtype=self.data.dtype))
        return other.sub(self)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.full(self.shape, other, dtype=self.data.dtype))
        return other.div(self)

    def __pow__(self, exponent):
        return self.pow(exponent)

    def __neg__(self):
        return self.neg()

    def __matmul__(self, other):
        return self.matmul(other)

    def __getitem__(self, idx):
        return Tensor(self.data[idx], requires_grad=False)

    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value
        self._version += 1

    def __repr__(self):
        grad_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        grad_fn_str = f", grad_fn={type(self._grad_fn).__name__}" if self._grad_fn else ""
        data_repr = repr(self.data).replace("array(", "").rstrip(")")
        return f"tensor({data_repr}{grad_str}{grad_fn_str})"

    def __len__(self):
        if self.data.ndim == 0:
            raise TypeError("len() of unsized tensor")
        return len(self.data)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data == other.data)
        return Tensor(self.data == other)

    def __ne__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data != other.data)
        return Tensor(self.data != other)

    def __lt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data < other.data)
        return Tensor(self.data < other)

    def __gt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data > other.data)
        return Tensor(self.data > other)

    def __le__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data <= other.data)
        return Tensor(self.data <= other)

    def __ge__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data >= other.data)
        return Tensor(self.data >= other)

    def __hash__(self):
        return id(self)


def _broadcast_tensors(a: "Tensor", b: "Tensor") -> Tuple["Tensor", "Tensor"]:
    try:
        result_shape = np.broadcast_shapes(a.shape, b.shape)
    except ValueError:
        raise ValueError(f"Cannot broadcast shapes {a.shape} and {b.shape}")
    a_data = np.broadcast_to(a.data, result_shape).copy()
    b_data = np.broadcast_to(b.data, result_shape).copy()
    return Tensor(a_data), Tensor(b_data)
