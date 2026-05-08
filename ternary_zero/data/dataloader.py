from __future__ import annotations

from typing import Optional, Callable, Union
import numpy as np

from .._backend import has_torch

if has_torch():
    import torch
    from torch.utils.data import DataLoader as _TorchDataLoader, Dataset as _TDataset

    class DataLoader:
        def __init__(
            self,
            dataset,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
            pin_memory: bool = True,
            drop_last: bool = False,
            collate_fn: Optional[Callable] = None,
            prefetch_factor: int = 2,
            persistent_workers: bool = False,
        ):
            self._loader = _TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                drop_last=drop_last,
                collate_fn=collate_fn,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers if num_workers > 0 else False,
            )

        def __iter__(self):
            return iter(self._loader)

        def __len__(self):
            return len(self._loader)

else:
    class DataLoader:
        def __init__(
            self,
            dataset,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
            pin_memory: bool = True,
            drop_last: bool = False,
            collate_fn: Optional[Callable] = None,
            prefetch_factor: int = 2,
            persistent_workers: bool = False,
        ):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.collate_fn = collate_fn

        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                np.random.shuffle(indices)
            batch = []
            for idx in indices:
                batch.append(self.dataset[idx])
                if len(batch) == self.batch_size:
                    yield self._collate(batch)
                    batch = []
            if batch and not self.drop_last:
                yield self._collate(batch)

        def __len__(self):
            n = len(self.dataset)
            if self.drop_last:
                return n // self.batch_size
            return (n + self.batch_size - 1) // self.batch_size

        def _collate(self, batch):
            if self.collate_fn is not None:
                return self.collate_fn(batch)
            if isinstance(batch[0], tuple):
                return tuple(
                    np.stack([item[i] for item in batch])
                    for i in range(len(batch[0]))
                )
            return np.stack(batch)
