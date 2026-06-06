"""Lightweight array metadata used when derived data is evaluated lazily."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DerivedArrayInfo:
    shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    def __len__(self) -> int:
        return self.shape[0]


def derived_info_for(document, dtype=None) -> DerivedArrayInfo:
    if dtype is None:
        dtype = getattr(document.base_data, "dtype", np.dtype(float))
    return DerivedArrayInfo(tuple(int(size) for size in document.current_shape), np.dtype(dtype))
