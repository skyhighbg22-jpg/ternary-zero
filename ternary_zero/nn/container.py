from __future__ import annotations

from typing import Iterator

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
