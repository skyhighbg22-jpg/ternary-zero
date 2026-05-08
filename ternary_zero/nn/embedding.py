from __future__ import annotations

import numpy as np

from .module import Module, Parameter
from ..tensor import Tensor


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = Parameter(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
            * 0.01
        )
        if padding_idx is not None:
            self.weight.data[padding_idx] = 0.0

    def forward(self, input: Tensor) -> Tensor:
        indices = input.data.astype(int)
        return Tensor(self.weight.data[indices])

    def extra_repr(self) -> str:
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        return s
