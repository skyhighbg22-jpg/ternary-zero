from __future__ import annotations

from typing import Iterator, List, Dict, Optional, Union
from collections import OrderedDict

from .module import Module
from ..tensor import Tensor


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input: Tensor) -> Tensor:
        for module in self._modules.values():
            input = module(input)
        return input

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(*list(self._modules.values())[idx])
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())


class ModuleList(Module):
    def __init__(self, modules: Optional[List[Module]] = None):
        super().__init__()
        if modules is not None:
            for idx, module in enumerate(modules):
                self.add_module(str(idx), module)

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def __setitem__(self, idx: int, module: Module):
        keys = list(self._modules.keys())
        self._modules[keys[idx]] = module

    def __delitem__(self, idx: int):
        keys = list(self._modules.keys())
        del self._modules[keys[idx]]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def append(self, module: Module):
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: Module):
        keys = list(self._modules.keys())
        if index >= len(keys):
            self.add_module(str(len(self)), module)
        else:
            new_modules = OrderedDict()
            for i, (k, v) in enumerate(self._modules.items()):
                if i == index:
                    new_modules[str(len(new_modules))] = module
                new_modules[str(len(new_modules))] = v
            if index >= len(keys):
                new_modules[str(len(new_modules))] = module
            self._modules = new_modules
        return self

    def extend(self, modules: List[Module]):
        for module in modules:
            self.append(module)
        return self

    def forward(self):
        raise NotImplementedError("ModuleList has no forward method")


class ModuleDict(Module):
    def __init__(self, modules: Optional[Dict[str, Module]] = None):
        super().__init__()
        if modules is not None:
            for name, module in modules.items():
                self.add_module(name, module)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module):
        self.add_module(key, module)

    def __delitem__(self, key: str):
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def update(self, modules: Dict[str, Module]):
        for name, module in modules.items():
            self.add_module(name, module)

    def forward(self):
        raise NotImplementedError("ModuleDict has no forward method")


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        shape = list(input.shape)
        end = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
        new_shape = shape[: self.start_dim] + [-1] + shape[end + 1 :]
        return input.view(new_shape)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"
