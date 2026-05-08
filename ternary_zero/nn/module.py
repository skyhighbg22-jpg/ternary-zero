from __future__ import annotations

from typing import Iterator, Optional, Set, Tuple, Dict, List
from collections import OrderedDict
import numpy as np

from ..tensor import Tensor
from .._backend import has_torch, to_numpy, get_default_device

if has_torch():
    import torch


class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        if isinstance(data, Tensor):
            data = data.data
        if has_torch() and isinstance(data, torch.Tensor):
            super().__init__(data, requires_grad=requires_grad)
        elif has_torch() and isinstance(data, np.ndarray):
            super().__init__(data, requires_grad=requires_grad)
        elif has_torch():
            super().__init__(np.array(data, dtype=np.float32), requires_grad=requires_grad)
        else:
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.float32)
            super().__init__(data, requires_grad=requires_grad)

    def __repr__(self):
        return f"Parameter({repr(to_numpy(self.data))})"


class Module:
    training: bool = True

    def __init__(self):
        self._parameters: Dict[str, Parameter] = OrderedDict()
        self._modules: Dict[str, "Module"] = OrderedDict()
        self._buffers: Dict[str, "np.ndarray"] = OrderedDict()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        seen: Set[int] = set()
        for name, param in self._parameters.items():
            if id(param) not in seen:
                seen.add(id(param))
                yield param
        if recurse:
            for name, module in self._modules.items():
                for param in module.parameters(recurse=True):
                    if id(param) not in seen:
                        seen.add(id(param))
                        yield param

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        if recurse:
            for name, module in self._modules.items():
                sub_prefix = f"{prefix}.{name}" if prefix else name
                yield from module.named_parameters(prefix=sub_prefix, recurse=True)

    def modules(self) -> Iterator["Module"]:
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        yield prefix, self
        for name, module in self._modules.items():
            sub_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(prefix=sub_prefix)

    def buffers(self, recurse: bool = True):
        for buf in self._buffers.values():
            yield buf
        if recurse:
            for module in self._modules.values():
                yield from module.buffers(recurse=True)

    def register_parameter(self, name: str, param: Optional[Parameter]):
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(f"expected Parameter or None, got {type(param)}")
        self._parameters[name] = param

    def register_buffer(self, name: str, buf):
        self._buffers[name] = buf

    def add_module(self, name: str, module: Optional["Module"]):
        if module is not None and not isinstance(module, Module):
            raise TypeError(f"expected Module or None, got {type(module)}")
        self._modules[name] = module

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_modules" in self.__dict__:
            _modules = self.__dict__["_modules"]
            if name in _modules:
                return _modules[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self, set_to_none: bool = False):
        for param in self.parameters():
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                if has_torch() and isinstance(param.data, torch.Tensor):
                    if param.data.grad is not None:
                        param.data.grad.zero_()
                else:
                    param.grad.data.fill(0)

    def state_dict(self) -> Dict[str, np.ndarray]:
        state = OrderedDict()
        for name, param in self.named_parameters():
            state[name] = to_numpy(param.data).copy()
        for name, buf in self._buffers.items():
            if has_torch() and isinstance(buf, torch.Tensor):
                state[f"_buffers.{name}"] = buf.detach().cpu().numpy().copy()
            elif isinstance(buf, np.ndarray):
                state[f"_buffers.{name}"] = buf.copy()
            else:
                state[f"_buffers.{name}"] = buf
        return state

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]):
        for name, param in self.named_parameters():
            if name in state_dict:
                if has_torch() and isinstance(param.data, torch.Tensor):
                    param.data.copy_(torch.tensor(state_dict[name], device=param.data.device, dtype=param.data.dtype))
                else:
                    param.data[:] = state_dict[name]
            else:
                raise KeyError(f"missing key in state_dict: '{name}'")

    def num_parameters(self, only_trainable: bool = False) -> int:
        total = 0
        for param in self.parameters():
            if only_trainable and not param.requires_grad:
                continue
            total += param.numel()
        return total

    def extra_repr(self) -> str:
        return ""

    def __repr__(self):
        lines = []
        extra = self.extra_repr()
        if extra:
            lines.append(extra)
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _add_indent(mod_str, 2)
            lines.append(f"({name}): {mod_str}")
        main_str = f"{type(self).__name__}("
        if lines:
            main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def to(self, device: str) -> "Module":
        for param in self.parameters(recurse=False):
            if has_torch() and isinstance(param.data, torch.Tensor):
                param.data = param.data.to(device)
        for name, buf in self._buffers.items():
            if has_torch() and isinstance(buf, torch.Tensor):
                self._buffers[name] = buf.to(device)
        for module in self._modules.values():
            module.to(device)
        return self

    def cuda(self, device: Optional[str] = None) -> "Module":
        target = device or get_default_device()
        return self.to(target)

    def cpu(self) -> "Module":
        return self.to("cpu")


def _add_indent(text: str, indent: int) -> str:
    lines = text.split("\n")
    if len(lines) == 1:
        return text
    first = lines[0]
    rest = "\n".join(" " * indent + line for line in lines[1:])
    return first + "\n" + rest
