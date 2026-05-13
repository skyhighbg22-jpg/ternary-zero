from .tensor import Tensor
from . import nn
from . import optim
from . import autograd
from . import quantize
from . import perf
from . import utils
from . import data
from . import distributed
from . import inference
from ._config import (
    is_grad_enabled, enable_grad, no_grad,
    get_default_device, set_default_device, is_cuda_available, has_torch,
    num_gpus, autocast, cuda_stream, enable_tf32, set_cudnn_benchmark,
    empty_cuda_cache, cuda_synchronize, cuda_memory_info,
)

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
    "perf",
    "utils",
    "data",
    "distributed",
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
    "get_default_device",
    "set_default_device",
    "is_cuda_available",
    "has_torch",
    "num_gpus",
    "autocast",
    "cuda_stream",
    "enable_tf32",
    "set_cudnn_benchmark",
    "empty_cuda_cache",
    "cuda_synchronize",
    "cuda_memory_info",
]


def tensor(data, dtype=None, requires_grad=False):
    from ._backend import has_torch, create_tensor, get_default_device
    if has_torch():
        import numpy as np
        if dtype is not None:
            from ._backend import _np_to_torch_dtype
            torch_dtype = _np_to_torch_dtype(dtype)
        else:
            import torch
            torch_dtype = torch.float32
        t = create_tensor(data, dtype=torch_dtype)
        return Tensor(t, requires_grad=requires_grad)
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
    from ._backend import has_torch, create_zeros
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    if has_torch():
        import numpy as np
        np_dtype = dtype or np.float32
        data = create_zeros(shape, dtype=np_dtype)
    else:
        import numpy as np
        if dtype is None:
            dtype = np.float32
        data = np.zeros(shape, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)


def ones(*shape, dtype=None, requires_grad=False):
    from ._backend import has_torch, create_ones
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    if has_torch():
        import numpy as np
        np_dtype = dtype or np.float32
        data = create_ones(shape, dtype=np_dtype)
    else:
        import numpy as np
        if dtype is None:
            dtype = np.float32
        data = np.ones(shape, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)


def randn(*shape, dtype=None, requires_grad=False):
    from ._backend import has_torch, create_randn
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    if has_torch():
        import numpy as np
        np_dtype = dtype or np.float32
        data = create_randn(shape, dtype=np_dtype)
    else:
        import numpy as np
        if dtype is None:
            dtype = np.float32
        data = np.random.randn(*shape).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)


def arange(start, end=None, step=1, dtype=None, requires_grad=False):
    from ._backend import has_torch, create_arange
    if has_torch():
        import numpy as np
        np_dtype = dtype or np.float32
        data = create_arange(start, end, step, dtype=np_dtype)
    else:
        import numpy as np
        if dtype is None:
            dtype = np.float32
        data = np.arange(start, end, step, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)


def eye(n, m=None, dtype=None, requires_grad=False):
    from ._backend import has_torch, create_eye
    if has_torch():
        import numpy as np
        np_dtype = dtype or np.float32
        data = create_eye(n, m, dtype=np_dtype)
    else:
        import numpy as np
        if dtype is None:
            dtype = np.float32
        data = np.eye(n, m, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)


def full(*args, fill_value=0.0, dtype=None, requires_grad=False):
    from ._backend import has_torch, create_full
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        shape = tuple(args[0])
    else:
        shape = args
    if has_torch():
        import numpy as np
        np_dtype = dtype or np.float32
        data = create_full(shape, fill_value, dtype=np_dtype)
    else:
        import numpy as np
        if dtype is None:
            dtype = np.float32
        data = np.full(shape, fill_value, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)


def zeros_like(t, **kwargs):
    return zeros(*t.shape, dtype=t.dtype, **kwargs)


def ones_like(t, **kwargs):
    return ones(*t.shape, dtype=t.dtype, **kwargs)


def randn_like(t, **kwargs):
    return randn(*t.shape, dtype=t.dtype, **kwargs)


def cat(tensors, dim=0):
    if not tensors:
        raise ValueError("expected a non-empty list of tensors")
    from ._backend import has_torch
    if has_torch():
        import torch
        arrays = [t.data for t in tensors]
        return Tensor(torch.cat(arrays, dim=dim))
    import numpy as np
    arrays = [t.data for t in tensors]
    return Tensor(np.concatenate(arrays, axis=dim))


def stack(tensors, dim=0):
    if not tensors:
        raise ValueError("expected a non-empty list of tensors")
    from ._backend import has_torch
    if has_torch():
        import torch
        arrays = [t.data for t in tensors]
        return Tensor(torch.stack(arrays, dim=dim))
    import numpy as np
    arrays = [t.data for t in tensors]
    return Tensor(np.stack(arrays, axis=dim))
