from .tensor import Tensor
from . import nn
from . import optim
from . import autograd
from . import quantize
from . import utils
from ._config import is_grad_enabled, enable_grad, no_grad

try:
    from . import _core
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "nn",
    "optim",
    "autograd",
    "quantize",
    "utils",
    "tensor",
    "zeros",
    "ones",
    "randn",
    "arange",
    "eye",
    "full",
    "zeros_like",
    "ones_like",
    "randn_like",
    "cat",
    "stack",
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
]


def tensor(data, dtype=None, requires_grad=False):
    import numpy as np
    if dtype is None:
        if isinstance(data, (int, float)):
            dtype = np.float32
        elif isinstance(data, (list, tuple)):
            dtype = np.float32
        arr = np.array(data, dtype=dtype)
    else:
        arr = np.array(data, dtype=dtype)
    return Tensor(arr, requires_grad=requires_grad)


def zeros(*shape, dtype=None, requires_grad=False):
    import numpy as np
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    if dtype is None:
        dtype = np.float32
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def ones(*shape, dtype=None, requires_grad=False):
    import numpy as np
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    if dtype is None:
        dtype = np.float32
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)


def randn(*shape, dtype=None, requires_grad=False):
    import numpy as np
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    if dtype is None:
        dtype = np.float32
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)


def arange(start, end=None, step=1, dtype=None, requires_grad=False):
    import numpy as np
    if dtype is None:
        dtype = np.float32
    return Tensor(np.arange(start, end, step, dtype=dtype), requires_grad=requires_grad)


def eye(n, m=None, dtype=None, requires_grad=False):
    import numpy as np
    if dtype is None:
        dtype = np.float32
    return Tensor(np.eye(n, m, dtype=dtype), requires_grad=requires_grad)


def full(*args, fill_value=0.0, dtype=None, requires_grad=False):
    import numpy as np
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        shape = tuple(args[0])
    else:
        shape = args
    if dtype is None:
        dtype = np.float32
    return Tensor(np.full(shape, fill_value, dtype=dtype), requires_grad=requires_grad)


def zeros_like(t, **kwargs):
    return zeros(*t.shape, dtype=t.data.dtype, **kwargs)


def ones_like(t, **kwargs):
    return ones(*t.shape, dtype=t.data.dtype, **kwargs)


def randn_like(t, **kwargs):
    return randn(*t.shape, dtype=t.data.dtype, **kwargs)


def cat(tensors, dim=0):
    import numpy as np
    if not tensors:
        raise ValueError("expected a non-empty list of tensors")
    arrays = [t.data for t in tensors]
    return Tensor(np.concatenate(arrays, axis=dim))


def stack(tensors, dim=0):
    import numpy as np
    if not tensors:
        raise ValueError("expected a non-empty list of tensors")
    arrays = [t.data for t in tensors]
    return Tensor(np.stack(arrays, axis=dim))
