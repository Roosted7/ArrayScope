"""Logical array sizes must stay exact without allocating their payloads."""

from math import prod
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.array_metadata import DerivedArrayInfo
from arrayscope.core.array_source import (
    CompositeArraySource,
    LazySourceArray,
    NdArraySource,
    ScaledArraySource,
    SourceReadRefused,
)


@pytest.mark.parametrize(
    "shape",
    [(2**32, 2**32), (2**32, 2**31), (np.int64(2**32), np.int64(2**32)), (), (0, 2**40)],
)
def test_metadata_size_is_exact_beyond_signed_machine_integer(shape):
    metadata = DerivedArrayInfo(shape, np.dtype("complex128"))

    assert metadata.size == prod(map(int, shape))
    assert metadata.nbytes == prod(map(int, shape)) * 16


def test_lazy_source_wrappers_preserve_huge_sizes_and_refuse_before_reading():
    backing = SimpleNamespace(shape=(2**32, 2**32), dtype=np.dtype("int16"))
    source = NdArraySource(backing)
    scaled = ScaledArraySource(backing, out_dtype=np.float32)
    composite = CompositeArraySource(source, source)

    assert source.nbytes == 2**65
    assert scaled.nbytes == 2**66
    assert composite.nbytes == 2**65
    for wrapped in (source, scaled, composite):
        lazy = LazySourceArray(wrapped)
        assert lazy.size == 2**64
        assert lazy.nbytes == wrapped.nbytes
        with pytest.raises(SourceReadRefused) as caught:
            lazy.materialize()
        assert caught.value.requested_nbytes == wrapped.nbytes
