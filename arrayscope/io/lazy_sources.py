"""Memory-mapped lazy sources for supported on-disk formats.

First adapters for the out-of-core source protocol (ADR 0049): NumPy ``.npy``
via ``np.load(mmap_mode="r")`` and BART ``.cfl`` via a Fortran-order
``np.memmap``. Uncompressed NIfTI ``.nii`` is mapped at its on-disk integer
dtype and rescaled per read (:class:`ScaledArraySource`), so a 2-byte int16
scan never inflates to a resident float32/float64 copy. Chunked stores
(Zarr/HDF5-like) are a later adapter behind the same protocol.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from arrayscope.core.array_source import LazySourceArray, NdArraySource, ScaledArraySource
from arrayscope.core.memory_policy import MiB, sample_system_memory

DEFAULT_LAZY_FRACTION_OF_AVAILABLE = 0.25
MIN_LAZY_THRESHOLD_BYTES = 64 * MiB

MEMMAP_SOURCE_SUFFIXES = (".npy", ".cfl")
# Uncompressed NIfTI stores its voxels contiguously, so nibabel can hand back a
# memmap of the raw integer samples; the affine scl_slope/scl_inter are applied
# per read. ``.nii.gz`` is compressed and cannot be mapped, so it is excluded.
SCALED_SOURCE_SUFFIXES = (".nii",)


def supports_memmap_source(suffix: str) -> bool:
    return str(suffix).lower() in MEMMAP_SOURCE_SUFFIXES


def supports_lazy_source(suffix: str) -> bool:
    """True for any suffix ArrayScope can open through the out-of-core seam."""

    suffix = str(suffix).lower()
    return suffix in MEMMAP_SOURCE_SUFFIXES or suffix in SCALED_SOURCE_SUFFIXES


def lazy_load_threshold_bytes(*, system_available_bytes: int | None = None) -> int:
    """Files at or above this size are opened lazily by ``load_path(lazy="auto")``."""

    if system_available_bytes is None:
        system_available_bytes = sample_system_memory().available_bytes
    return max(
        MIN_LAZY_THRESHOLD_BYTES,
        int(int(system_available_bytes) * DEFAULT_LAZY_FRACTION_OF_AVAILABLE),
    )


def should_load_lazily(filepath, *, lazy="auto", threshold_bytes: int | None = None) -> bool:
    """Decide whether ``load_path`` should open ``filepath`` as a lazy source."""

    filepath = Path(filepath)
    if lazy is False or not supports_lazy_source(filepath.suffix):
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
    raise ValueError(
        f"unsupported lazy source format: {suffix!r} (supported: {', '.join(MEMMAP_SOURCE_SUFFIXES)})"
    )


def open_scaled_nifti_source(filepath):
    """Open an uncompressed ``.nii`` as a rescaled, memory-mapped source.

    Returns ``(LazySourceArray, axes)``. The raw integer voxels stay mapped;
    :class:`ScaledArraySource` applies ``scl_slope``/``scl_inter`` and narrows to
    float32 per region read. Raises ``ValueError`` when the file cannot be
    mapped (e.g. a scaled dtype nibabel refuses to memmap), so ``lazy="auto"``
    falls back to eager loading.
    """

    filepath = Path(filepath)
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ValueError("nibabel is required to open NIfTI sources lazily") from exc

    from arrayscope.io.file_interpreters import _nifti_axes

    image = nib.load(filepath, mmap=True)
    source = scaled_nifti_source(image, label=f"nii-memmap:{filepath.name}")
    if not isinstance(source.backing, np.memmap):
        # get_unscaled() copied into memory (compressed / non-mappable dtype);
        # a lazy source over a resident array would give no memory benefit.
        raise ValueError(f"{filepath} cannot be memory-mapped for a lazy NIfTI source")
    axes = _nifti_axes(image.header, source.shape)
    return LazySourceArray(source), axes


def scaled_nifti_source(image, *, label: str) -> ScaledArraySource:
    """Wrap a loaded nibabel image's raw voxels in a :class:`ScaledArraySource`.

    Shared by the lazy seam and the eager :class:`NiftiLoader` so both produce
    identical values: raw on-disk samples (memmap for uncompressed ``.nii``,
    in-memory for ``.nii.gz``) expanded per read into float32 — or complex64
    when the voxels are complex on disk, preserving the imaginary part that a
    float cast would silently discard — with ``scl_slope``/``scl_inter``
    applied. Never routes through nibabel's float64 scaling path.
    """

    from arrayscope.io.file_interpreters import remove_trailing_singletons

    proxy = image.dataobj
    raw = proxy.get_unscaled() if hasattr(proxy, "get_unscaled") else np.asanyarray(proxy)
    raw = remove_trailing_singletons(raw)
    kind = np.dtype(raw.dtype).kind
    if kind not in "biufc":
        raise ValueError(f"unsupported NIfTI on-disk dtype {raw.dtype!r} (RGB/structured voxels)")
    slope = float(getattr(proxy, "slope", 1.0) or 1.0)
    inter = float(getattr(proxy, "inter", 0.0) or 0.0)
    return ScaledArraySource(
        raw,
        slope=slope,
        inter=inter,
        out_dtype=np.complex64 if kind == "c" else np.float32,
        label=label,
    )


def open_lazy_source(filepath):
    """Open any lazy-capable ``filepath``; returns ``(LazySourceArray, axes)``.

    ``axes`` is ``None`` for formats that carry no axis metadata (.npy/.cfl).
    """

    suffix = Path(filepath).suffix.lower()
    if suffix in SCALED_SOURCE_SUFFIXES:
        return open_scaled_nifti_source(filepath)
    return open_memmap_source(filepath), None


__all__ = [
    "DEFAULT_LAZY_FRACTION_OF_AVAILABLE",
    "MEMMAP_SOURCE_SUFFIXES",
    "MIN_LAZY_THRESHOLD_BYTES",
    "SCALED_SOURCE_SUFFIXES",
    "lazy_load_threshold_bytes",
    "open_lazy_source",
    "open_memmap_source",
    "open_scaled_nifti_source",
    "scaled_nifti_source",
    "should_load_lazily",
    "supports_lazy_source",
    "supports_memmap_source",
]
