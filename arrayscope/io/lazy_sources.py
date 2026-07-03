"""Memory-mapped lazy sources for supported on-disk formats.

First adapters for the out-of-core source protocol (ADR 0049): NumPy ``.npy``
via ``np.load(mmap_mode="r")`` and BART ``.cfl`` via a Fortran-order
``np.memmap``. Chunked stores (Zarr/HDF5-like) are a later adapter behind the
same protocol.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from arrayscope.core.array_source import LazySourceArray, NdArraySource
from arrayscope.core.memory_policy import MiB, sample_system_memory


DEFAULT_LAZY_FRACTION_OF_AVAILABLE = 0.25
MIN_LAZY_THRESHOLD_BYTES = 64 * MiB

MEMMAP_SOURCE_SUFFIXES = (".npy", ".cfl")


def supports_memmap_source(suffix: str) -> bool:
    return str(suffix).lower() in MEMMAP_SOURCE_SUFFIXES


def lazy_load_threshold_bytes(*, system_available_bytes: int | None = None) -> int:
    """Files at or above this size are opened lazily by ``load_path(lazy="auto")``."""

    if system_available_bytes is None:
        system_available_bytes = sample_system_memory().available_bytes
    return max(MIN_LAZY_THRESHOLD_BYTES, int(int(system_available_bytes) * DEFAULT_LAZY_FRACTION_OF_AVAILABLE))


def should_load_lazily(filepath, *, lazy="auto", threshold_bytes: int | None = None) -> bool:
    """Decide whether ``load_path`` should open ``filepath`` as a lazy source."""

    filepath = Path(filepath)
    if lazy is False or not supports_memmap_source(filepath.suffix):
        return False
    if lazy is True:
        return True
    if lazy != "auto":
        raise ValueError(f"lazy must be True, False, or 'auto', got {lazy!r}")
    threshold = lazy_load_threshold_bytes() if threshold_bytes is None else int(threshold_bytes)
    return int(os.path.getsize(filepath)) >= threshold


def open_memmap_source(filepath) -> LazySourceArray:
    """Open ``filepath`` (.npy or .cfl) as a lazy, memory-mapped array source."""

    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix == ".npy":
        mapped = np.load(filepath, mmap_mode="r")
        if not isinstance(mapped, np.memmap):
            raise ValueError(f"{filepath} cannot be memory-mapped (object or pickled array)")
        return LazySourceArray(NdArraySource(mapped, label=f"npy-memmap:{filepath.name}"))
    if suffix == ".cfl":
        from arrayscope.io.file_interpreters import BartLoader, remove_trailing_singletons

        loader = BartLoader(filepath)
        mapped = np.memmap(filepath, dtype=np.complex64, mode="r", shape=loader.dims, order="F")
        mapped = remove_trailing_singletons(mapped)
        return LazySourceArray(NdArraySource(mapped, label=f"cfl-memmap:{filepath.name}"))
    raise ValueError(f"unsupported lazy source format: {suffix!r} (supported: {', '.join(MEMMAP_SOURCE_SUFFIXES)})")


__all__ = [
    "DEFAULT_LAZY_FRACTION_OF_AVAILABLE",
    "MEMMAP_SOURCE_SUFFIXES",
    "MIN_LAZY_THRESHOLD_BYTES",
    "lazy_load_threshold_bytes",
    "open_memmap_source",
    "should_load_lazily",
    "supports_memmap_source",
]
