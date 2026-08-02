"""Per-stage shared-memory segments: a cached stage a child process can read.

ADR 0064 family 6 stopped at the stage cache: an evaluation whose region plan
declares cache candidates reads a ~234 MB array the GUI process owns, and no
descriptor could name it, so the whole FFT-prefix workload stayed on the
in-process threads that share the GUI's GIL. This is the missing descriptor.

A stage value stored into :class:`~arrayscope.operations.stage_cache.StageCache`
is copied once into its own ``SharedMemory`` segment. The copy costs one memcpy
on whichever thread stored the value — for a montage that is the
``STAGE_MATERIALIZATION`` kernel worker, never the GUI thread — and buys every
later tile a mapping instead of a transfer.

Ownership follows ``source_transport``: the GUI process creates and unlinks, a
child only attaches and closes. Three things release a segment, and they are
deliberately redundant because a 234 MB leak is not a rounding error: the cache's
eviction hook (deterministic), the stage value's own death (a dropped window's
cache never evicts anything), and ``atexit``.

Nothing here imports Qt, and the numpy view a child builds keeps its segment
mapped, so ``close()`` on a segment somebody is still reading raises
``BufferError`` instead of handing out a dangling pointer.
"""

from __future__ import annotations

import atexit
import threading
import weakref
from contextlib import suppress
from dataclasses import dataclass

import numpy as np

from arrayscope.operations.regions import RegionSpec, region_contains
from arrayscope.operations.source_transport import shared_publication_enabled

#: The canonical volume is 336x336x272 — 117 MB as float32, 234 MB once an FFT
#: prefix has made it complex64. One segment that size is the point of the
#: exercise; a stage larger than this is not worth doubling for, and the
#: evaluations that depend on it keep running in-process.
MAX_PUBLISHED_STAGE_BYTES = 1024 * 1024 * 1024

#: How many published segments one child keeps mapped. A mapping is cheap but
#: not free — an unlinked segment's pages live until the last mapping closes —
#: and a montage session only ever reads its current stage plus the one it is
#: handing over from.
CHILD_ATTACHMENT_LIMIT = 3


@dataclass(frozen=True)
class StageSegment:
    """One published stage value: the bytes, and what the cache called them.

    ``shape``/``dtype`` describe the stored array; ``stage_shape``/``stage_dtype``
    are the *key's* — the stage's declared output shape and dtype, which a
    region-cut value does not match. Both are needed because
    :meth:`AttachedStageCache.get_containing` answers exactly the question
    ``StageCache.get_containing`` answers, and that one compares the key's.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    order: str
    nbytes: int
    stage_index: int
    stage_shape: tuple[int, ...]
    stage_dtype: str
    region: RegionSpec
    priority: str
    recompute_cost: float
    visible_reuse: bool
    prefetch_only: bool

    def attach(self):
        """This segment as a read-only ``StageValue`` in the current process."""

        from arrayscope.operations.stage_cache import StageValue

        return StageValue(
            data=attach_stage_array(self),
            region=self.region,
            stage_index=int(self.stage_index),
            nbytes=int(self.nbytes),
            priority=str(self.priority),
            recompute_cost=float(self.recompute_cost),
            visible_reuse=bool(self.visible_reuse),
            prefetch_only=bool(self.prefetch_only),
        )


class SegmentUnavailable(RuntimeError):
    """The bytes a segment named are gone — the publisher released them."""


# ------------------------------------------------------------- publication


class _Publication:
    """One owned segment: the handle to unlink, and who it was published for.

    ``value_ref`` is weak on purpose — a registry that held the stage value
    would keep 234 MB alive after its cache dropped it, which is the leak this
    module exists to avoid — and its callback is the release of last resort.
    """

    __slots__ = ("descriptor", "segment", "value_id", "value_ref")

    def __init__(self, segment, descriptor, value_ref, value_id) -> None:
        self.segment = segment
        self.descriptor = descriptor
        self.value_ref = value_ref
        self.value_id = int(value_id)


_lock = threading.RLock()
_published: dict[object, _Publication] = {}
_keys_by_value: dict[int, object] = {}
_published_bytes = 0
publications = 0
releases = 0


def publish_stage_value(key, value, *, budget_bytes: int = 0) -> StageSegment | None:
    """Copy a stored stage value into a segment children can map.

    Returns the descriptor, or ``None`` when this value is not worth (or not
    able to be) published — publication off, no array, over the per-value or
    total budget. A ``None`` here is never a correctness problem: the
    evaluations that would have used the segment keep running in-process.

    ``budget_bytes`` is the owning cache's envelope. Published bytes cannot
    exceed cached bytes anyway (every entry releases when it evicts), so this
    is the belt to that braces.
    """

    global _published_bytes, publications

    if not shared_publication_enabled():
        return None
    array = getattr(value, "data", None)
    region = getattr(value, "region", None)
    if not isinstance(array, np.ndarray) or not isinstance(region, RegionSpec):
        return None
    if array.dtype.hasobject or int(array.nbytes) <= 0:
        return None
    if int(array.nbytes) > MAX_PUBLISHED_STAGE_BYTES:
        return None
    try:
        from multiprocessing import shared_memory
    except ImportError:  # pragma: no cover - platform without shared memory
        return None
    budget = int(budget_bytes) if int(budget_bytes) > 0 else MAX_PUBLISHED_STAGE_BYTES
    # An F-contiguous array is the only one worth writing out in Fortran order;
    # a strided view has no layout to preserve and lands C-ordered.
    order = "F" if array.flags.f_contiguous and not array.flags.c_contiguous else "C"
    with _lock:
        existing = _published.get(key)
        if existing is not None and existing.value_ref() is value:
            return existing.descriptor
        if _published_bytes + int(array.nbytes) > budget:
            return None
        try:
            segment = shared_memory.SharedMemory(create=True, size=int(array.nbytes))
        except (OSError, ValueError):
            return None
        _published_bytes += int(array.nbytes)
    # The memcpy runs OUTSIDE the module lock: a release (which runs under the
    # owning cache's lock, and therefore in front of a GUI-thread probe) must
    # never wait behind ~40 ms of copying.
    published = False
    try:
        view = np.ndarray(array.shape, dtype=array.dtype, buffer=segment.buf, order=order)
        view[...] = array
        # The view exports the segment's buffer; a live one makes close() raise.
        del view
        descriptor = StageSegment(
            name=str(segment.name),
            shape=tuple(int(size) for size in array.shape),
            dtype=str(np.dtype(array.dtype)),
            order=order,
            nbytes=int(array.nbytes),
            stage_index=int(getattr(value, "stage_index", 0)),
            stage_shape=tuple(int(size) for size in getattr(key, "shape", ()) or ()),
            stage_dtype=str(getattr(key, "dtype", "")),
            region=region,
            priority=str(getattr(value, "priority", "low")),
            recompute_cost=float(getattr(value, "recompute_cost", 0.0) or 0.0),
            visible_reuse=bool(getattr(value, "visible_reuse", True)),
            prefetch_only=bool(getattr(value, "prefetch_only", False)),
        )
        # Registering only now is what keeps a half-copied segment invisible:
        # a concurrent tile can never be handed a descriptor for bytes that are
        # still being written.
        with _lock:
            superseded = _published.pop(key, None)
            reference = weakref.ref(value, lambda ref, key=key: _release_dead(key, ref))
            _published[key] = _Publication(segment, descriptor, reference, id(value))
            _keys_by_value[id(value)] = key
            publications += 1
        published = True
        if superseded is not None:
            _discard(superseded)
        return descriptor
    finally:
        if not published:
            with _lock:
                _published_bytes -= int(array.nbytes)
            _close_and_unlink(segment)


def segment_for_value(value) -> StageSegment | None:
    """The segment published for this exact stage value, if it still exists."""

    if value is None:
        return None
    with _lock:
        key = _keys_by_value.get(id(value))
        if key is None:
            return None
        entry = _published.get(key)
        if entry is None or entry.value_ref() is not value:
            return None
        return entry.descriptor


def release_stage_value(key, value=None) -> bool:
    """Unlink the segment published for ``key`` (when ``value`` still owns it)."""

    global _published_bytes, releases

    with _lock:
        entry = _published.get(key)
        if entry is None:
            return False
        if value is not None and entry.value_ref() is not value:
            return False
        _published.pop(key, None)
        _keys_by_value.pop(entry.value_id, None)
        _published_bytes -= int(entry.descriptor.nbytes)
        releases += 1
    _discard(entry)
    return True


def published_bytes() -> int:
    """Bytes this PROCESS has in published stage segments, across every cache.

    Process-wide because the registry is: a viewer with two windows over the
    same document publishes each window's stage separately, and the number a
    reader wants is what the doubling actually costs.
    """

    with _lock:
        return int(_published_bytes)


def published_count() -> int:
    """Segments this process currently owns (see :func:`published_bytes`)."""

    with _lock:
        return len(_published)


def release_all_stage_segments() -> None:
    """Unlink everything this process published — shutdown, or a cache reset."""

    global _published_bytes

    with _lock:
        entries = tuple(_published.values())
        _published.clear()
        _keys_by_value.clear()
        _published_bytes = 0
    for entry in entries:
        _discard(entry)


def _release_dead(key, reference) -> None:
    """A published value died without its cache evicting it (a closed window)."""

    global _published_bytes, releases

    with _lock:
        entry = _published.get(key)
        if entry is None or entry.value_ref is not reference:
            return
        _published.pop(key, None)
        _keys_by_value.pop(entry.value_id, None)
        _published_bytes -= int(entry.descriptor.nbytes)
        releases += 1
    _discard(entry)


def _discard(entry: _Publication) -> None:
    _close_and_unlink(entry.segment)


def _close_and_unlink(segment) -> None:
    with suppress(BufferError, OSError):
        segment.close()
    with suppress(OSError):
        segment.unlink()


# ---------------------------------------------------------------- attaching

_attached: dict[str, tuple] = {}
_attachment_order: list[str] = []
_retired: list[tuple] = []
attachments = 0


def attach_stage_array(descriptor: StageSegment):
    """Map a published segment read-only in this process (once per name)."""

    global attachments

    with _lock:
        cached = _attached.get(descriptor.name)
        if cached is not None:
            return cached[1]
    from multiprocessing import shared_memory

    try:
        # ``track=False`` is the attacher's half of ADR 0064's ownership rule:
        # without it this process' resource tracker adopts the name and unlinks
        # it at exit, which is the publisher's job and nobody else's. (Python
        # 3.13+; on an older one the attacher falls back to the tracked form,
        # which is what ``source_transport`` still does for base arrays.)
        segment = shared_memory.SharedMemory(name=descriptor.name, track=False)
    except FileNotFoundError as exc:
        raise SegmentUnavailable(f"stage segment {descriptor.name} is gone") from exc
    except TypeError:  # pragma: no cover - Python without an untracked attach
        try:
            segment = shared_memory.SharedMemory(name=descriptor.name)
        except FileNotFoundError as exc:
            raise SegmentUnavailable(f"stage segment {descriptor.name} is gone") from exc
    view = np.ndarray(
        tuple(descriptor.shape),
        dtype=np.dtype(descriptor.dtype),
        buffer=segment.buf,
        order=str(descriptor.order),
    )
    # A child never writes a stage: the parent's copy is the only writer, and
    # an in-place operation reaching here would be corrupting every other tile.
    view.flags.writeable = False
    with _lock:
        cached = _attached.get(descriptor.name)
        if cached is not None:
            del view
            with suppress(BufferError, OSError):
                segment.close()
            return cached[1]
        _attached[descriptor.name] = (segment, view)
        _attachment_order.append(descriptor.name)
        attachments += 1
        while len(_attachment_order) > CHILD_ATTACHMENT_LIMIT:
            retired = _attachment_order.pop(0)
            _retired.append(_attached.pop(retired))
    return view


def close_retired_attachments() -> None:
    """Unmap segments this process stopped needing, between tasks.

    Never mid-task: the array a retired segment backs may still be the slab an
    in-flight evaluation is reading. ``close()`` on a segment with a live view
    raises ``BufferError`` rather than unmapping it, so a still-referenced
    attachment simply waits for the next call.
    """

    with _lock:
        retired = tuple(_retired)
        _retired.clear()
    for entry in retired:
        segment, view = entry
        del view
        try:
            segment.close()
        except BufferError:
            with _lock:
                _retired.append(entry)
        except OSError:
            pass


def close_all_attachments() -> None:
    with _lock:
        entries = tuple(_attached.values()) + tuple(_retired)
        _attached.clear()
        _attachment_order.clear()
        _retired.clear()
    for segment, view in entries:
        del view
        with suppress(BufferError, OSError):
            segment.close()


def _release_everything() -> None:
    close_all_attachments()
    release_all_stage_segments()


atexit.register(_release_everything)


# --------------------------------------------------------- the child's cache


class AttachedStageCache:
    """The stage cache a child sees: exactly the segments its request named.

    It answers ``get``/``get_containing`` with the same predicate
    ``StageCache`` does, minus the two terms a child cannot compare: the
    document key and the operation prefix themselves. Both are already pinned
    by the request — the child rebuilds *this* document from the source token,
    recipe and revision — so what is left to match is the stage's position
    (``len(operation_prefix)`` is the stage index), its declared shape and
    dtype, and region containment.

    It stores nothing. ``max_bytes`` is the publisher's, because
    ``_store_stage_indices`` reads it to decide which stage a transition writes
    its output region for, and a different answer there would be a different
    composition, not just a missed store.
    """

    def __init__(self, segments, *, max_bytes: int = 0) -> None:
        self._segments = tuple(segments or ())
        self._values: dict[str, object] = {}
        self.max_bytes = int(max_bytes)
        self.misses = 0

    def note_candidate(self, summary: str = "") -> None:
        return None

    def note_refused(self, summary: str = "") -> None:
        return None

    def put(self, key, value) -> bool:
        # A child never admits into a cache it does not own (ADR 0064: "the
        # parent admits results"). The value it would have stored is the one
        # the parent already holds.
        return False

    def get(self, key):
        return self.get_containing(key)

    def get_containing(self, key):
        for descriptor in self._segments:
            if len(tuple(key.operation_prefix)) != int(descriptor.stage_index):
                continue
            if str(key.dtype) != str(descriptor.stage_dtype):
                continue
            if tuple(int(size) for size in key.shape) != tuple(descriptor.stage_shape):
                continue
            if not region_contains(descriptor.region, key.region, key.shape):
                continue
            value = self._values.get(descriptor.name)
            if value is None:
                try:
                    value = descriptor.attach()
                except SegmentUnavailable:
                    # The parent released it mid-flight. A miss makes this
                    # evaluation recompute the stage from its source: slower,
                    # never wrong.
                    self.misses += 1
                    continue
                self._values[descriptor.name] = value
            return value
        self.misses += 1
        return None


__all__ = [
    "CHILD_ATTACHMENT_LIMIT",
    "MAX_PUBLISHED_STAGE_BYTES",
    "AttachedStageCache",
    "SegmentUnavailable",
    "StageSegment",
    "attach_stage_array",
    "close_all_attachments",
    "close_retired_attachments",
    "publish_stage_value",
    "published_bytes",
    "published_count",
    "release_all_stage_segments",
    "release_stage_value",
    "segment_for_value",
]
