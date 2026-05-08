from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np

from ..tensor import Tensor
from .._backend import has_torch, to_numpy

if has_torch():
    import torch
    from torch.utils.data import Dataset as _TorchDataset, IterableDataset as _TorchIterableDataset

    class Dataset(_TorchDataset):
        def __getitem__(self, index):
            raise NotImplementedError

        def __len__(self):
            raise NotImplementedError

    class IterableDataset(_TorchIterableDataset):
        def __iter__(self):
            raise NotImplementedError

else:
    class Dataset:
        def __getitem__(self, index):
            raise NotImplementedError

        def __len__(self):
            raise NotImplementedError

    class IterableDataset:
        def __iter__(self):
            raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors: Tensor):
        if not tensors:
            raise ValueError("At least one tensor is required")
        first_len = len(tensors[0])
        for t in tensors:
            if len(t) != first_len:
                raise ValueError("All tensors must have the same length")
        self.tensors = tensors

    def __getitem__(self, index):
        if has_torch():
            return tuple(
                t.data[index] if isinstance(t.data, torch.Tensor) else torch.tensor(t.data[index])
                for t in self.tensors
            )
        return tuple(Tensor(t.data[index]) for t in self.tensors)

    def __len__(self):
        return len(self.tensors[0])
