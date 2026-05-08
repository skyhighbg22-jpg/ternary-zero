from __future__ import annotations

from typing import Iterable, Dict, Any, Optional
import numpy as np

from .tensor import Tensor
from .nn.module import Parameter
from ._backend import has_torch, to_numpy

if has_torch():
    import torch


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
                if has_torch() and isinstance(param.data, torch.Tensor):
                    if set_to_none:
                        param.data.grad = None
                    elif param.data.grad is not None:
                        param.data.grad.zero_()
                else:
                    if set_to_none:
                        param.grad = None
                    elif param.grad is not None:
                        param.grad.data.fill(0)

    def step(self):
        raise NotImplementedError

    def state_dict(self) -> dict:
        state_np = {}
        for k, v in self.state.items():
            state_np[k] = {}
            for sk, sv in v.items():
                if has_torch() and isinstance(sv, torch.Tensor):
                    state_np[k][sk] = sv.detach().cpu().numpy()
                elif isinstance(sv, np.ndarray):
                    state_np[k][sk] = sv
                else:
                    state_np[k][sk] = sv
        return {"state": state_np, "param_groups": self.param_groups}

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
                if has_torch() and isinstance(param.data, torch.Tensor):
                    with torch.no_grad():
                        grad = param.data.grad
                        if grad is None:
                            continue
                        if weight_decay != 0.0:
                            grad = grad + weight_decay * param.data
                        if momentum != 0.0:
                            param_id = id(param)
                            if param_id not in self.state:
                                self.state[param_id] = {"momentum_buffer": torch.zeros_like(param.data)}
                            buf = self.state[param_id]["momentum_buffer"]
                            buf.mul_(momentum).add_(grad)
                            if nesterov:
                                grad = grad + momentum * buf
                            else:
                                grad = buf
                        param.data -= lr * grad
                else:
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
                if has_torch() and isinstance(param.data, torch.Tensor):
                    with torch.no_grad():
                        grad = param.data.grad
                        if grad is None:
                            continue
                        if weight_decay != 0.0:
                            grad = grad + weight_decay * param.data
                        param_id = id(param)
                        if param_id not in self.state:
                            self.state[param_id] = {
                                "step": 0,
                                "exp_avg": torch.zeros_like(param.data),
                                "exp_avg_sq": torch.zeros_like(param.data),
                            }
                        state = self.state[param_id]
                        state["step"] += 1
                        exp_avg = state["exp_avg"]
                        exp_avg_sq = state["exp_avg_sq"]
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        bias_correction1 = 1 - beta1 ** state["step"]
                        bias_correction2 = 1 - beta2 ** state["step"]
                        corrected_exp_avg = exp_avg / bias_correction1
                        corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                        step_size = lr / bias_correction1
                        denom = corrected_exp_avg_sq.sqrt().add_(eps)
                        param.data -= step_size * corrected_exp_avg / denom
                else:
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
                if has_torch() and isinstance(param.data, torch.Tensor):
                    with torch.no_grad():
                        grad = param.data.grad
                        if grad is None:
                            continue
                        param.data -= lr * weight_decay * param.data
                        param_id = id(param)
                        if param_id not in self.state:
                            self.state[param_id] = {
                                "step": 0,
                                "exp_avg": torch.zeros_like(param.data),
                                "exp_avg_sq": torch.zeros_like(param.data),
                            }
                        state = self.state[param_id]
                        state["step"] += 1
                        exp_avg = state["exp_avg"]
                        exp_avg_sq = state["exp_avg_sq"]
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        bias_correction1 = 1 - beta1 ** state["step"]
                        bias_correction2 = 1 - beta2 ** state["step"]
                        corrected_exp_avg = exp_avg / bias_correction1
                        corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                        step_size = lr / bias_correction1
                        denom = corrected_exp_avg_sq.sqrt().add_(eps)
                        param.data -= step_size * corrected_exp_avg / denom
                else:
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
                if has_torch() and isinstance(param.data, torch.Tensor):
                    with torch.no_grad():
                        grad = param.data.grad
                        if grad is None:
                            continue
                        if weight_decay != 0.0:
                            grad = grad + weight_decay * param.data
                        param_id = id(param)
                        if param_id not in self.state:
                            self.state[param_id] = {
                                "square_avg": torch.zeros_like(param.data),
                                "momentum_buffer": torch.zeros_like(param.data) if momentum > 0 else None,
                            }
                        state = self.state[param_id]
                        square_avg = state["square_avg"]
                        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                        avg = square_avg.sqrt().add_(eps)
                        update = grad / avg
                        if momentum > 0:
                            buf = state["momentum_buffer"]
                            buf.mul_(momentum).add_(update)
                            update = buf
                        param.data -= lr * update
                else:
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


class Adagrad(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.01,
        lr_decay: float = 0.0,
        eps: float = 1e-10,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            lr_decay = group["lr_decay"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if has_torch() and isinstance(param.data, torch.Tensor):
                    with torch.no_grad():
                        grad = param.data.grad
                        if grad is None:
                            continue
                        if weight_decay != 0.0:
                            grad = grad + weight_decay * param.data
                        param_id = id(param)
                        if param_id not in self.state:
                            self.state[param_id] = {"step": 0, "sum_sq": torch.zeros_like(param.data)}
                        state = self.state[param_id]
                        state["step"] += 1
                        state["sum_sq"] += grad ** 2
                        clr = lr / (1 + (state["step"] - 1) * lr_decay)
                        std = state["sum_sq"].sqrt().add_(eps)
                        param.data -= clr * grad / std
                else:
                    if param.grad is None:
                        continue
                    grad = param.grad.data.copy()
                    if weight_decay != 0.0:
                        grad = grad + weight_decay * param.data
                    param_id = id(param)
                    if param_id not in self.state:
                        self.state[param_id] = {"step": 0, "sum_sq": np.zeros_like(param.data)}
                    state = self.state[param_id]
                    state["step"] += 1
                    state["sum_sq"] += grad ** 2
                    clr = lr / (1 + (state["step"] - 1) * lr_decay)
                    std = np.sqrt(state["sum_sq"]) + eps
                    param.data -= clr * grad / std


class _LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        new_lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, new_lrs):
            group["lr"] = lr


class StepLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base_lr * factor for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float = 0.9,
        last_epoch: int = -1,
    ):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** self.last_epoch for base_lr in self.base_lrs]


class CosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return list(self.base_lrs)
        return [
            self.eta_min + (base_lr - self.eta_min)
            * (1 + np.cos(np.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]


class LinearLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]
        if self.last_epoch > self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
        return [
            base_lr * (
                self.start_factor
                + (self.end_factor - self.start_factor) * self.last_epoch / self.total_iters
            )
            for base_lr in self.base_lrs
        ]


class ReduceLROnPlateau:
    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.best = None
        self.num_bad_epochs = 0
        self.last_epoch = 0

        if mode == "min":
            self.is_better = lambda a, best: a < best - threshold
        else:
            self.is_better = lambda a, best: a > best + threshold

    def step(self, metrics: float):
        self.last_epoch += 1
        if self.best is None:
            self.best = metrics
            return

        if self.is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

    def _reduce_lr(self):
        for group in self.optimizer.param_groups:
            new_lr = max(group["lr"] * self.factor, self.min_lr)
            group["lr"] = new_lr
