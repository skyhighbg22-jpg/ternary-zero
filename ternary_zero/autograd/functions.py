from __future__ import annotations

import numpy as np
from ..autograd.function import Function
from ..tensor import Tensor


class Add(Function):
    @staticmethod
    def _forward(ctx, a, b):
        ctx._a_shape = a.shape
        ctx._b_shape = b.shape
        return a + b

    @staticmethod
    def _raw_forward(a, b):
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        from ..autograd.engine import _unbroadcast
        grad_a = _unbroadcast(grad_output, ctx._a_shape) if ctx.needs_input_grad[0] else None
        grad_b = _unbroadcast(grad_output, ctx._b_shape) if ctx.needs_input_grad[1] else None
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def _forward(ctx, a, b):
        ctx._a_shape = a.shape
        ctx._b_shape = b.shape
        return a - b

    @staticmethod
    def _raw_forward(a, b):
        return a - b

    @staticmethod
    def backward(ctx, grad_output):
        from ..autograd.engine import _unbroadcast
        grad_a = _unbroadcast(grad_output, ctx._a_shape) if ctx.needs_input_grad[0] else None
        grad_b = _unbroadcast(-grad_output, ctx._b_shape) if ctx.needs_input_grad[1] else None
        return grad_a, grad_b


class Mul(Function):
    @staticmethod
    def _forward(ctx, a, b):
        ctx._a = a
        ctx._b = b
        ctx._a_shape = a.shape
        ctx._b_shape = b.shape
        return a * b

    @staticmethod
    def _raw_forward(a, b):
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        from ..autograd.engine import _unbroadcast
        grad_a = None
        grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = _unbroadcast(grad_output * ctx._b, ctx._a_shape)
        if ctx.needs_input_grad[1]:
            grad_b = _unbroadcast(grad_output * ctx._a, ctx._b_shape)
        return grad_a, grad_b


class Div(Function):
    @staticmethod
    def _forward(ctx, a, b):
        ctx._a = a
        ctx._b = b
        ctx._a_shape = a.shape
        ctx._b_shape = b.shape
        return a / b

    @staticmethod
    def _raw_forward(a, b):
        return a / b

    @staticmethod
    def backward(ctx, grad_output):
        from ..autograd.engine import _unbroadcast
        grad_a = None
        grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = _unbroadcast(grad_output / ctx._b, ctx._a_shape)
        if ctx.needs_input_grad[1]:
            grad_b = _unbroadcast(-grad_output * ctx._a / (ctx._b ** 2), ctx._b_shape)
        return grad_a, grad_b


class Pow(Function):
    @staticmethod
    def _forward(ctx, a, b):
        ctx._a = a
        ctx._b = b
        ctx._result = a ** b
        return ctx._result

    @staticmethod
    def _raw_forward(a, b):
        return a ** b

    @staticmethod
    def backward(ctx, grad_output):
        grad_a = None
        grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = grad_output * ctx._b * (ctx._a ** (ctx._b - 1))
        if ctx.needs_input_grad[1]:
            grad_b = grad_output * ctx._result * np.log(ctx._a)
        return grad_a, grad_b


class Neg(Function):
    @staticmethod
    def _forward(ctx, a):
        return -a

    @staticmethod
    def _raw_forward(a):
        return -a

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output if ctx.needs_input_grad[0] else None


class Abs(Function):
    @staticmethod
    def _forward(ctx, a):
        ctx._sign = np.sign(a)
        return np.abs(a)

    @staticmethod
    def _raw_forward(a):
        return np.abs(a)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx._sign if ctx.needs_input_grad[0] else None


class MatMul(Function):
    @staticmethod
    def _forward(ctx, a, b):
        ctx._a = a
        ctx._b = b
        return a @ b

    @staticmethod
    def _raw_forward(a, b):
        return a @ b

    @staticmethod
    def backward(ctx, grad_output):
        grad_a = None
        grad_b = None
        if ctx.needs_input_grad[0]:
            if grad_output.ndim == 1:
                grad_a = np.outer(grad_output, ctx._b)
            elif ctx._b.ndim == 1:
                grad_a = np.outer(grad_output, ctx._b)
            else:
                grad_a = grad_output @ ctx._b.T
        if ctx.needs_input_grad[1]:
            if ctx._a.ndim == 1:
                grad_b = np.outer(ctx._a, grad_output)
            elif grad_output.ndim == 1:
                grad_b = np.outer(ctx._a, grad_output)
            else:
                grad_b = ctx._a.T @ grad_output
        return grad_a, grad_b


class Sum(Function):
    @staticmethod
    def _forward(ctx, a, dim=None, keepdim=False):
        ctx._input_shape = a.shape
        ctx._dim = dim
        ctx._keepdim = keepdim
        return np.sum(a, axis=dim, keepdims=keepdim)

    @staticmethod
    def _raw_forward(a, dim=None, keepdim=False):
        return np.sum(a, axis=dim, keepdims=keepdim)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None,)
        grad = grad_output
        if ctx._dim is not None and not ctx._keepdim:
            grad = np.expand_dims(grad_output, axis=ctx._dim)
        return (np.broadcast_to(grad, ctx._input_shape).copy(),)


class Mean(Function):
    @staticmethod
    def _forward(ctx, a, dim=None, keepdim=False):
        ctx._input_shape = a.shape
        ctx._dim = dim
        ctx._keepdim = keepdim
        if dim is None:
            ctx._count = a.size
        else:
            ctx._count = a.shape[dim]
        return np.mean(a, axis=dim, keepdims=keepdim)

    @staticmethod
    def _raw_forward(a, dim=None, keepdim=False):
        return np.mean(a, axis=dim, keepdims=keepdim)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None,)
        grad = grad_output / ctx._count
        if ctx._dim is not None and not ctx._keepdim:
            grad = np.expand_dims(grad, axis=ctx._dim)
        return (np.broadcast_to(grad, ctx._input_shape).copy(),)


class Reshape(Function):
    @staticmethod
    def _forward(ctx, a, shape):
        ctx._input_shape = a.shape
        return np.reshape(a, shape)

    @staticmethod
    def _raw_forward(a, shape):
        return np.reshape(a, shape)

    @staticmethod
    def backward(ctx, grad_output):
        return np.reshape(grad_output, ctx._input_shape) if ctx.needs_input_grad[0] else None


class Transpose(Function):
    @staticmethod
    def _forward(ctx, a, dim0, dim1):
        ctx._dim0 = dim0
        ctx._dim1 = dim1
        return np.swapaxes(a, dim0, dim1)

    @staticmethod
    def _raw_forward(a, dim0, dim1):
        return np.swapaxes(a, dim0, dim1)

    @staticmethod
    def backward(ctx, grad_output):
        return np.swapaxes(grad_output, ctx._dim0, ctx._dim1) if ctx.needs_input_grad[0] else None


class Permute(Function):
    @staticmethod
    def _forward(ctx, a, dims):
        ctx._dims = dims
        ctx._inv_dims = tuple(np.argsort(dims))
        return np.transpose(a, dims)

    @staticmethod
    def _raw_forward(a, dims):
        return np.transpose(a, dims)

    @staticmethod
    def backward(ctx, grad_output):
        return np.transpose(grad_output, ctx._inv_dims) if ctx.needs_input_grad[0] else None


class Unsqueeze(Function):
    @staticmethod
    def _forward(ctx, a, dim):
        ctx._dim = dim
        return np.expand_dims(a, axis=dim)

    @staticmethod
    def _raw_forward(a, dim):
        return np.expand_dims(a, axis=dim)

    @staticmethod
    def backward(ctx, grad_output):
        return np.squeeze(grad_output, axis=ctx._dim) if ctx.needs_input_grad[0] else None


class Squeeze(Function):
    @staticmethod
    def _forward(ctx, a, dim):
        ctx._dim = dim
        ctx._input_shape = a.shape
        return np.squeeze(a, axis=dim)

    @staticmethod
    def _raw_forward(a, dim):
        return np.squeeze(a, axis=dim)

    @staticmethod
    def backward(ctx, grad_output):
        return np.reshape(grad_output, ctx._input_shape) if ctx.needs_input_grad[0] else None


class ReLU(Function):
    @staticmethod
    def _forward(ctx, a):
        ctx._mask = a > 0
        return a * ctx._mask

    @staticmethod
    def _raw_forward(a):
        return np.maximum(a, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx._mask if ctx.needs_input_grad[0] else None


class Softmax(Function):
    @staticmethod
    def _forward(ctx, a, dim=-1):
        shifted = a - np.max(a, axis=dim, keepdims=True)
        exp_a = np.exp(shifted)
        result = exp_a / np.sum(exp_a, axis=dim, keepdims=True)
        ctx._result = result
        ctx._dim = dim
        return result

    @staticmethod
    def _raw_forward(a, dim=-1):
        shifted = a - np.max(a, axis=dim, keepdims=True)
        exp_a = np.exp(shifted)
        return exp_a / np.sum(exp_a, axis=dim, keepdims=True)

    @staticmethod
    def backward(ctx, grad_output):
        s = ctx._result
        dot = np.sum(grad_output * s, axis=ctx._dim, keepdims=True)
        return s * (grad_output - dot) if ctx.needs_input_grad[0] else None


class Log(Function):
    @staticmethod
    def _forward(ctx, a):
        ctx._a = a
        return np.log(a)

    @staticmethod
    def _raw_forward(a):
        return np.log(a)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output / ctx._a if ctx.needs_input_grad[0] else None


class Exp(Function):
    @staticmethod
    def _forward(ctx, a):
        result = np.exp(a)
        ctx._result = result
        return result

    @staticmethod
    def _raw_forward(a):
        return np.exp(a)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx._result if ctx.needs_input_grad[0] else None


class Max(Function):
    @staticmethod
    def _forward(ctx, a, dim, keepdim):
        ctx._dim = dim
        result = np.max(a, axis=dim, keepdims=True)
        ctx._mask = (a == result)
        ctx._count = ctx._mask.sum(axis=dim, keepdims=True)
        if not keepdim:
            return result.squeeze(axis=dim)
        return result

    @staticmethod
    def _raw_forward(a, dim, keepdim):
        return np.max(a, axis=dim, keepdims=keepdim)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None
        grad = np.expand_dims(grad_output, axis=ctx._dim) if grad_output.ndim < ctx._mask.ndim else grad_output
        return (ctx._mask / ctx._count) * grad


class Min(Function):
    @staticmethod
    def _forward(ctx, a, dim, keepdim):
        ctx._dim = dim
        result = np.min(a, axis=dim, keepdims=True)
        ctx._mask = (a == result)
        ctx._count = ctx._mask.sum(axis=dim, keepdims=True)
        if not keepdim:
            return result.squeeze(axis=dim)
        return result

    @staticmethod
    def _raw_forward(a, dim, keepdim):
        return np.min(a, axis=dim, keepdims=keepdim)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None
        grad = np.expand_dims(grad_output, axis=ctx._dim) if grad_output.ndim < ctx._mask.ndim else grad_output
        return (ctx._mask / ctx._count) * grad


class Linear(Function):
    @staticmethod
    def _forward(ctx, input, weight, bias=None):
        ctx._input = input
        ctx._weight = weight
        ctx._bias = bias
        output = input @ weight.T
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def _raw_forward(input, weight, bias=None):
        output = input @ weight.T
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ ctx._weight
        if ctx.needs_input_grad[1]:
            if grad_output.ndim == 1:
                grad_weight = np.outer(grad_output, ctx._input)
            else:
                grad_weight = grad_output.T @ (ctx._input.reshape(-1, ctx._input.shape[-1]) if ctx._input.ndim > 2 else ctx._input)
        if ctx.needs_input_grad[2] and ctx._bias is not None:
            if grad_output.ndim == 1:
                grad_bias = grad_output.copy()
            else:
                grad_bias = grad_output.sum(axis=0)

        return grad_input, grad_weight, grad_bias


class TernaryQuantizeSTE(Function):
    @staticmethod
    def _forward(ctx, weights, alpha):
        ctx._input_shape = weights.shape
        flat = weights.flatten()
        abs_sum = np.sum(np.abs(flat))
        mean_abs = abs_sum / flat.size if flat.size > 0 else 0.0
        threshold = alpha * mean_abs

        mask_pos = flat > threshold
        mask_neg = flat < -threshold

        ternary = np.where(mask_pos, 1, np.where(mask_neg, -1, 0)).astype(np.int8)

        non_zero = np.abs(flat[mask_pos | mask_neg])
        scale = np.mean(non_zero) if non_zero.size > 0 else 1.0

        ctx._scale = scale
        ctx._mask = (np.abs(flat) <= threshold).astype(np.float32)

        return ternary.reshape(weights.shape), np.float32(scale)

    @staticmethod
    def _raw_forward(weights, alpha):
        flat = weights.flatten()
        abs_sum = np.sum(np.abs(flat))
        mean_abs = abs_sum / flat.size if flat.size > 0 else 0.0
        threshold = alpha * mean_abs
        mask_pos = flat > threshold
        mask_neg = flat < -threshold
        ternary = np.where(mask_pos, 1, np.where(mask_neg, -1, 0)).astype(np.int8)
        non_zero = np.abs(flat[mask_pos | mask_neg])
        scale = np.mean(non_zero) if non_zero.size > 0 else 1.0
        return ternary.reshape(weights.shape), np.float32(scale)

    @staticmethod
    def backward(ctx, grad_ternary, grad_scale):
        if not ctx.needs_input_grad[0]:
            return None
        scale = ctx._scale
        flat_grad = grad_ternary.flatten().astype(np.float32) if isinstance(grad_ternary, np.ndarray) else np.array(grad_ternary, dtype=np.float32).flatten()
        mask_flat = ctx._mask
        if flat_grad.size != mask_flat.size:
            grad = flat_grad * scale
        else:
            grad = flat_grad * scale * (1.0 - mask_flat)
        return grad.reshape(ctx._input_shape)


class CrossEntropyLoss(Function):
    @staticmethod
    def _forward(ctx, logits, targets):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
        log_probs = shifted - log_sum_exp[..., np.newaxis]

        if targets.ndim == 1:
            batch_size = logits.shape[0]
            loss = -log_probs[np.arange(batch_size), targets].mean()
        else:
            loss = -(targets * log_probs).sum(axis=-1).mean()

        ctx._log_probs = log_probs
        ctx._targets = targets
        ctx._batch_size = logits.shape[0] if logits.ndim > 0 else 1
        return np.float32(loss)

    @staticmethod
    def _raw_forward(logits, targets):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
        log_probs = shifted - log_sum_exp[..., np.newaxis]
        if targets.ndim == 1:
            batch_size = logits.shape[0]
            return np.float32(-log_probs[np.arange(batch_size), targets].mean())
        return np.float32(-(targets * log_probs).sum(axis=-1).mean())

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None

        probs = np.exp(ctx._log_probs)
        grad = probs.copy()
        targets = ctx._targets

        if targets.ndim == 1:
            batch_size = ctx._batch_size
            grad[np.arange(batch_size), targets] -= 1.0
        else:
            grad -= targets

        return grad / ctx._batch_size, None


class MSELoss(Function):
    @staticmethod
    def _forward(ctx, prediction, target):
        ctx._prediction = prediction
        ctx._target = target
        ctx._batch_size = prediction.size
        diff = prediction - target
        return np.float32(np.mean(diff ** 2))

    @staticmethod
    def _raw_forward(prediction, target):
        return np.float32(np.mean((prediction - target) ** 2))

    @staticmethod
    def backward(ctx, grad_output):
        diff = ctx._prediction - ctx._target
        grad_pred = 2.0 * diff / ctx._batch_size if ctx.needs_input_grad[0] else None
        return grad_pred, None
