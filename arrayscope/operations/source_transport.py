"""Picklable identity and transport for a document's base data (ADR 0064).

Two things a compute process needs that ``id(document.base_data)`` cannot give
it: a **stable token** that names the same data in every process, and a
**descriptor** it can rebuild the data from without anyone pickling an array.

The descriptor is a small tree. Its leaves are the two ways bytes already exist
outside one interpreter's heap — a memory-mapped file (the OS page cache does
the sharing for free) and a ``SharedMemory`` segment the GUI process publishes
for an eager array. Its nodes are the source adapters that wrap them
(``ScaledArraySource``, ``CompositeArraySource``, ``LazySourceArray``). A child
reconstructs the tree and calls ``read_base_region`` itself.

Segment ownership is one-sided by ADR: the GUI process creates and unlinks;
children only attach and close. Nothing here imports Qt, and importing it must
stay cheap enough for a spawned worker.
"""

from __future__ import annotations

import atexit
import mmap
import os
import threading
import weakref
from contextlib import suppress
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

#: One eager array larger than this is not worth a private copy in shared
#: memory; the family that would have shipped it keeps running in-process.
MAX_PUBLISHED_BYTES = 1024 * 1024 * 1024
#: Total published-segment budget for this process. Publication is permanent
#: for the array's lifetime, so the ceiling is what bounds the doubling.
MAX_PUBLISHED_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


class SourceTransportUnavailable(RuntimeError):
    """A descriptor could not be rebuilt — the bytes it named are gone or changed."""


# ------------------------------------------------------------- descriptors


@dataclass(frozen=True)
class MemmapLeaf:
    """A contiguous window of a file, as ``np.memmap`` reopens it.

    ``offset`` is the byte offset of *this* array's first element, not the
    root map's: numpy leaves ``memmap.offset`` untouched when it slices or
    reshapes, so the only honest offset is the one measured from the data
    pointers. ``mtime_ns``/``size_bytes`` are checked on reconstruction, so a
    file that changed under us fails loudly instead of serving other bytes.
    """

    path: str
    offset: int
    shape: tuple[int, ...]
    dtype: str
    order: str
    mtime_ns: int
    size_bytes: int

    def reconstruct(self):
        try:
            stat = os.stat(self.path)
        except OSError as exc:
            raise SourceTransportUnavailable(f"{self.path} is unreadable: {exc}") from exc
        if int(stat.st_mtime_ns) != int(self.mtime_ns) or int(stat.st_size) != int(self.size_bytes):
            raise SourceTransportUnavailable(f"{self.path} changed since it was described")
        return np.memmap(
            self.path,
            dtype=np.dtype(self.dtype),
            mode="r",
            shape=tuple(self.shape),
            order=str(self.order),
            offset=int(self.offset),
        )


@dataclass(frozen=True)
class SharedMemoryLeaf:
    """An eager array the GUI process copied into one shared segment once."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    order: str
    nbytes: int

    def reconstruct(self):
        return _attach_shared_memory(self)


@dataclass(frozen=True)
class ScaleNode:
    """``ScaledArraySource``: an affine expansion applied per region read."""

    child: object
    slope: float
    inter: float
    out_dtype: str
    label: str
    chunk_shape: tuple[int, ...] | None

    def reconstruct(self):
        from arrayscope.core.array_source import ScaledArraySource

        return ScaledArraySource(
            self.child.reconstruct(),
            slope=float(self.slope),
            inter=float(self.inter),
            out_dtype=np.dtype(self.out_dtype),
            label=str(self.label),
            chunk_shape=self.chunk_shape,
        )


@dataclass(frozen=True)
class CompositeNode:
    """``CompositeArraySource``: two sources combined elementwise per read."""

    a: object
    b: object
    op: str
    label: str
    chunk_shape: tuple[int, ...] | None

    def reconstruct(self):
        from arrayscope.core.array_source import CompositeArraySource

        return CompositeArraySource(
            self.a.reconstruct(),
            self.b.reconstruct(),
            op=str(self.op),
            label=str(self.label),
            chunk_shape=self.chunk_shape,
            # A child owns nothing: the segments and maps outlive its copy.
            own_inputs=False,
        )


@dataclass(frozen=True)
class NdSourceNode:
    """``NdArraySource``: the adapter over a plain array-like leaf."""

    child: object
    label: str
    chunk_shape: tuple[int, ...] | None

    def reconstruct(self):
        from arrayscope.core.array_source import NdArraySource

        return NdArraySource(
            self.child.reconstruct(),
            label=str(self.label),
            chunk_shape=self.chunk_shape,
        )


@dataclass(frozen=True)
class LazyArrayNode:
    """``LazySourceArray``: the document-facing proxy over a source."""

    child: object
    materialize_budget_bytes: int | None

    def reconstruct(self):
        from arrayscope.core.array_source import LazySourceArray

        return LazySourceArray(
            self.child.reconstruct(),
            materialize_budget_bytes=self.materialize_budget_bytes,
        )


# --------------------------------------------------------------- describing


def describe_base_data(base_data, *, publish: bool = False):
    """Describe ``base_data`` as a picklable tree, or ``None`` if it cannot be.

    ``publish=True`` allows an eager array to be copied into a shared segment,
    which is a one-way commitment for the array's lifetime — so only the code
    that is about to ship a request asks for it. Everything else (notably
    :func:`base_data_token`) describes without publishing and simply gets
    ``None`` for an unpublished eager array.
    """

    if base_data is None:
        return None
    entry = _entry_for(base_data)
    if entry.descriptor is not None:
        return entry.descriptor
    descriptor = _describe(base_data, publish=bool(publish))
    if descriptor is not None:
        # Only a stable description is memoized. A publication decision is
        # permanent, so once made it is the description for good.
        entry.descriptor = descriptor
    return descriptor


def _describe(base_data, *, publish: bool):
    from arrayscope.core.array_source import (
        CompositeArraySource,
        LazySourceArray,
        NdArraySource,
        ScaledArraySource,
    )

    if isinstance(base_data, LazySourceArray):
        child = _describe(base_data.source, publish=publish)
        if child is None:
            return None
        return LazyArrayNode(
            child=child,
            materialize_budget_bytes=getattr(base_data, "_materialize_budget_bytes", None),
        )
    if isinstance(base_data, ScaledArraySource):
        child = _describe(base_data.backing, publish=publish)
        if child is None:
            return None
        return ScaleNode(
            child=child,
            slope=float(base_data._slope),
            inter=float(base_data._inter),
            out_dtype=str(np.dtype(base_data.dtype)),
            label=str(base_data.label),
            chunk_shape=base_data.chunk_shape,
        )
    if isinstance(base_data, CompositeArraySource):
        a = _describe(base_data.a, publish=publish)
        b = _describe(base_data.b, publish=publish)
        if a is None or b is None:
            return None
        return CompositeNode(
            a=a,
            b=b,
            op=str(base_data.op),
            label=str(base_data.label),
            chunk_shape=base_data.chunk_shape,
        )
    if isinstance(base_data, NdArraySource):
        child = _describe(base_data._array, publish=publish)
        if child is None:
            return None
        return NdSourceNode(
            child=child,
            label=str(base_data.label),
            chunk_shape=base_data.chunk_shape,
        )
    if isinstance(base_data, np.ndarray):
        leaf = _memmap_leaf(base_data)
        if leaf is not None:
            return leaf
        return _publish_array(base_data) if publish else None
    return None


def _memmap_leaf(array: np.ndarray) -> MemmapLeaf | None:
    """Describe ``array`` as a window of a mapped file, if it truly is one."""

    root = array
    while isinstance(root, np.ndarray) and isinstance(root.base, np.ndarray):
        root = root.base
    if not isinstance(root, np.memmap) or not isinstance(root.base, mmap.mmap):
        return None
    filename = getattr(root, "filename", None)
    if not filename:
        return None
    contiguous = bool(array.flags.c_contiguous)
    if not contiguous and not bool(array.flags.f_contiguous):
        # A strided view is not a window; reopening it would need the strides
        # too, and no source in this codebase produces one.
        return None
    delta = int(array.__array_interface__["data"][0]) - int(root.__array_interface__["data"][0])
    if delta < 0:
        return None
    offset = int(getattr(root, "offset", 0)) + delta
    try:
        stat = os.stat(filename)
    except OSError:
        return None
    if offset + int(array.nbytes) > int(stat.st_size):
        return None
    return MemmapLeaf(
        path=os.fspath(filename),
        offset=offset,
        shape=tuple(int(size) for size in array.shape),
        dtype=str(np.dtype(array.dtype)),
        order="C" if contiguous else "F",
        mtime_ns=int(stat.st_mtime_ns),
        size_bytes=int(stat.st_size),
    )


# ------------------------------------------------------- shared publication

_published: dict[str, object] = {}
_published_bytes = 0
_attached: dict[str, tuple] = {}
_shared_lock = threading.Lock()
_publication_enabled = False


def enable_shared_publication() -> None:
    """Allow eager arrays to be copied into shared memory in this process.

    Publishing doubles an eager array for the viewer's life, and a process
    that runs every task on its own threads would never read the copy. So the
    default is off and the one caller is the GUI process deciding to run a
    compute pool at all; a memory-mapped source is unaffected either way,
    because it is shared by the page cache and never copied.
    """

    global _publication_enabled
    _publication_enabled = True


def shared_publication_enabled() -> bool:
    """Is this process publishing arrays into shared memory at all?

    One switch, two publishers: base data here, cached operation stages in
    ``operations/stage_segments.py``. A viewer that runs every task on its own
    threads reads neither, so neither pays for a copy.
    """

    return bool(_publication_enabled)


def _publish_array(array: np.ndarray) -> SharedMemoryLeaf | None:
    """Copy an eager array into one shared segment this process owns forever."""

    global _published_bytes
    if not _publication_enabled:
        return None
    if array.dtype.hasobject or array.nbytes <= 0:
        return None
    if int(array.nbytes) > MAX_PUBLISHED_BYTES:
        return None
    # Only an F-contiguous array is worth copying out in Fortran order; a
    # strided view has no layout to preserve and lands C-ordered.
    order = "F" if array.flags.f_contiguous and not array.flags.c_contiguous else "C"
    try:
        from multiprocessing import shared_memory
    except ImportError:  # pragma: no cover - platform without shared memory
        return None
    with _shared_lock:
        if _published_bytes + int(array.nbytes) > MAX_PUBLISHED_TOTAL_BYTES:
            return None
        try:
            segment = shared_memory.SharedMemory(create=True, size=int(array.nbytes))
        except (OSError, ValueError):
            return None
        _published[segment.name] = segment
        _published_bytes += int(array.nbytes)
    view = np.ndarray(array.shape, dtype=array.dtype, buffer=segment.buf, order=order)
    view[...] = array
    return SharedMemoryLeaf(
        name=str(segment.name),
        shape=tuple(int(size) for size in array.shape),
        dtype=str(np.dtype(array.dtype)),
        order=order,
        nbytes=int(array.nbytes),
    )


def _attach_shared_memory(leaf: SharedMemoryLeaf):
    """Attach (once per process) and hand back a view; never unlink."""

    with _shared_lock:
        cached = _attached.get(leaf.name)
        if cached is not None:
            return cached[1]
    from multiprocessing import shared_memory

    try:
        segment = shared_memory.SharedMemory(name=leaf.name)
    except FileNotFoundError as exc:
        raise SourceTransportUnavailable(f"shared segment {leaf.name} is gone") from exc
    view = np.ndarray(
        tuple(leaf.shape),
        dtype=np.dtype(leaf.dtype),
        buffer=segment.buf,
        order=str(leaf.order),
    )
    with _shared_lock:
        cached = _attached.get(leaf.name)
        if cached is not None:
            segment.close()
            return cached[1]
        # The segment must outlive the view: a closed segment leaves the array
        # pointing at unmapped memory, which is a segfault, not an exception.
        _attached[leaf.name] = (segment, view)
    return view


def _release_published_segments() -> None:
    with _shared_lock:
        segments = tuple(_published.values())
        _published.clear()
        attached = tuple(_attached.values())
        _attached.clear()
    for segment, _view in attached:
        with suppress(BufferError, OSError):
            segment.close()
    for segment in segments:
        try:
            segment.close()
            segment.unlink()
        except (BufferError, OSError):
            pass


atexit.register(_release_published_segments)


# ------------------------------------------------------------------ tokens


class _Entry:
    __slots__ = ("descriptor", "token")

    def __init__(self, token, descriptor) -> None:
        self.token = token
        self.descriptor = descriptor


_entries: dict[int, _Entry] = {}
_entry_lock = threading.Lock()


def base_data_token(base_data):
    """A picklable token that names this base data in any process.

    File-backed data is named by its descriptor, so two documents over the same
    file agree — which is the point: a process-local ``id()`` could never be
    the first element of a cache key a worker also computes. Everything else
    gets a uuid minted once per object and dropped when the object dies; that
    is stable for the object's lifetime, which is all a cache key needs.
    """

    if base_data is None:
        return None
    return _entry_for(base_data).token


def _entry_for(base_data) -> _Entry:
    key = id(base_data)
    # The hit is the hot path — every cache key in the montage asks for a token
    # — so it reads the dict directly. A single ``get`` needs no lock, and the
    # finalizer that removes an entry only ever runs for an object the caller
    # is not holding, so a hit cannot be dropped underneath this read.
    entry = _entries.get(key)
    if entry is not None:
        return entry
    # Describing stats the filesystem, so it happens outside the lock; a race
    # only costs a duplicate description, never a duplicate token.
    descriptor = _describe(base_data, publish=False)
    token = descriptor if descriptor is not None else f"array:{uuid4().hex}"
    entry = _Entry(token, descriptor)
    try:
        reference = weakref.ref(base_data, lambda _ref, key=key: _drop_entry(key))
    except TypeError:
        # Nothing to hang a finalizer on: fall back to today's exact semantics
        # rather than caching a token an id reuse could hand to another object.
        return _Entry(("id", key), descriptor)
    with _entry_lock:
        existing = _entries.get(key)
        if existing is not None:
            return existing
        entry.token = token
        _entries[key] = entry
        _references[key] = reference
    return entry


_references: dict[int, weakref.ref] = {}


def _drop_entry(key: int) -> None:
    with _entry_lock:
        _entries.pop(key, None)
        _references.pop(key, None)


__all__ = [
    "MAX_PUBLISHED_BYTES",
    "MAX_PUBLISHED_TOTAL_BYTES",
    "CompositeNode",
    "LazyArrayNode",
    "MemmapLeaf",
    "NdSourceNode",
    "ScaleNode",
    "SharedMemoryLeaf",
    "SourceTransportUnavailable",
    "base_data_token",
    "describe_base_data",
    "enable_shared_publication",
    "shared_publication_enabled",
]
