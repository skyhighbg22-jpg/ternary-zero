from __future__ import annotations

from typing import Iterable, Dict, Any, Optional
import numpy as np

from .tensor import Tensor
from .nn.module import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter], defaults: dict):
        self.defaults = defaults
        self.param_groups = []
        self.state: Dict[int, dict] = {}

        if isinstance(params, (list, tuple)):
            self.add_param_group({"params": list(params)})
        else:
            self.add_param_group({"params": list(params)})

    def add_param_group(self, param_group: dict):
        for key, value in self.defaults.items():
            param_group.setdefault(key, value)
        self.param_groups.append(param_group)

    def zero_grad(self, set_to_none: bool = False):
        for group in self.param_groups:
            for param in group["params"]:
                if set_to_none:
                    param.grad = None
                elif param.grad is not None:
                    param.grad.data.fill(0)

    def step(self):
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {"state": self.state, "param_groups": self.param_groups}

    def load_state_dict(self, state_dict: dict):
        self.state = state_dict["state"]
        self.param_groups = state_dict["param_groups"]


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data.copy()

                if weight_decay != 0.0:
                    grad = grad + weight_decay * param.data

                if momentum != 0.0:
                    param_id = id(param)
                    if param_id not in self.state:
                        self.state[param_id] = {"momentum_buffer": np.zeros_like(param.data)}

                    buf = self.state[param_id]["momentum_buffer"]
                    buf[:] = momentum * buf + grad

                    if nesterov:
                        grad = grad + momentum * buf
                    else:
                        grad = buf

                param.data -= lr * grad


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data.copy()

                if weight_decay != 0.0:
                    grad = grad + weight_decay * param.data

                param_id = id(param)
                if param_id not in self.state:
                    self.state[param_id] = {
                        "step": 0,
                        "exp_avg": np.zeros_like(param.data),
                        "exp_avg_sq": np.zeros_like(param.data),
                    }

                state = self.state[param_id]
                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                step_size = lr / bias_correction1
                denom = np.sqrt(corrected_exp_avg_sq) + eps

                param.data -= step_size * corrected_exp_avg / denom


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                param.data -= lr * weight_decay * param.data

                grad = param.grad.data.copy()

                param_id = id(param)
                if param_id not in self.state:
                    self.state[param_id] = {
                        "step": 0,
                        "exp_avg": np.zeros_like(param.data),
                        "exp_avg_sq": np.zeros_like(param.data),
                    }

                state = self.state[param_id]
                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                denom = np.sqrt(corrected_exp_avg_sq) + eps
                param.data -= lr * corrected_exp_avg / denom


class RMSprop(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data.copy()

                if weight_decay != 0.0:
                    grad = grad + weight_decay * param.data

                param_id = id(param)
                if param_id not in self.state:
                    self.state[param_id] = {
                        "square_avg": np.zeros_like(param.data),
                        "momentum_buffer": np.zeros_like(param.data) if momentum > 0 else None,
                    }

                state = self.state[param_id]
                square_avg = state["square_avg"]
                square_avg[:] = alpha * square_avg + (1 - alpha) * (grad ** 2)

                avg = np.sqrt(square_avg) + eps
                update = grad / avg

                if momentum > 0:
                    buf = state["momentum_buffer"]
                    buf[:] = momentum * buf + update
                    update = buf

                param.data -= lr * update
