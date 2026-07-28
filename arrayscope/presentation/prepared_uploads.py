"""Worker-prepared backend upload buffers, held in a keep-latest mailbox.

At a tiled commit the GUI thread's irreducible job is to submit already-built
commands and release their buffers. Packing a payload into the array a backend
uploads is not part of that job: it is pure work over an immutable payload and
belongs on a worker. This module is the seam between the two.

The ownership model is deliberately one-way and single-slot:

- A worker builds a buffer and **publishes** it under the exact identity it was
  built from. The buffer is immutable from that moment; the worker keeps no
  reference it will write through.
- The mailbox holds **at most one entry per slot**. A newer preparation for the
  same slot replaces the older one — under load the GUI thread should submit the
  newest ready buffer, not queue every intermediate one.
- The GUI thread **takes** a buffer by naming the identity it is about to
  submit. A take always removes the entry; it returns the buffer only when the
  stored key matches exactly. A key that does not match is a buffer prepared for
  something else, and it is dropped rather than presented.

That last rule is the correctness rule. A prepared buffer is a promise about
*which pixels these are*; presenting one under a different identity is a torn
frame, which costs more than preparing inline. The mailbox therefore never
guesses: a miss is a normal outcome, and the caller's fallback is to prepare on
the GUI thread exactly as it did before.

Buffers are large, so the mailbox is byte-bounded. Overflow evicts the
least-recently-published slots, which are the ones a governed fill is least
likely to reach next.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

# One 512x512 RGBA32F page is 4 MiB, so this holds a few dozen prepared tiles.
# The mailbox is a hand-off buffer, not a residency cache: entries live from
# admission to the commit that consumes them, and dropping one only costs the
# inline preparation that used to happen anyway.
DEFAULT_PREPARED_UPLOAD_BUDGET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class PreparedUpload:
    """One immutable buffer and the identity it was prepared from."""

    key: object
    buffer: object
    nbytes: int


@dataclass(frozen=True)
class PreparedUploadCounters:
    """Why the mailbox helped or did not, without a debugger.

    ``stale`` is the interesting one: it counts buffers that were prepared and
    then correctly refused because the slot moved on. A rising ``stale`` with a
    flat ``hits`` means preparation is chasing a target it never catches, which
    is a scheduling problem, not a correctness one.
    """

    published: int = 0
    replaced: int = 0
    hits: int = 0
    stale: int = 0
    misses: int = 0
    evicted: int = 0
    resident_entries: int = 0
    resident_bytes: int = 0


def _buffer_nbytes(buffer) -> int:
    """Best-effort size of a prepared buffer, for the mailbox byte bound."""

    nbytes = getattr(buffer, "nbytes", None)
    if nbytes is not None:
        return max(0, int(nbytes))
    return 0


class PreparedUploadMailbox:
    """Keep-latest, drop-stale hand-off of prepared buffers between threads.

    Every method is safe to call from any thread. ``publish`` is the worker
    side; ``take``/``discard``/``clear`` are the GUI-thread side, but nothing
    here depends on that split — the lock is the whole synchronisation story.
    """

    def __init__(self, *, budget_bytes: int = DEFAULT_PREPARED_UPLOAD_BUDGET_BYTES) -> None:
        self._budget_bytes = max(0, int(budget_bytes))
        self._lock = threading.Lock()
        self._entries: OrderedDict[object, PreparedUpload] = OrderedDict()
        self._resident_bytes = 0
        self._published = 0
        self._replaced = 0
        self._hits = 0
        self._stale = 0
        self._misses = 0
        self._evicted = 0

    def publish(self, slot, key, buffer, *, nbytes: int | None = None) -> bool:
        """Store ``buffer`` for ``slot``, replacing whatever that slot held.

        ``nbytes`` is required when the buffer is not itself an array — the
        mailbox cannot bound what it cannot size, and silently treating a
        wrapper as free would let residency grow without limit.

        Returns False when the buffer cannot be held at all — a buffer larger
        than the whole budget is not worth evicting everything else for, and the
        consumer's inline path covers it.
        """

        nbytes = _buffer_nbytes(buffer) if nbytes is None else max(0, int(nbytes))
        if self._budget_bytes and nbytes > self._budget_bytes:
            return False
        entry = PreparedUpload(key=key, buffer=buffer, nbytes=nbytes)
        with self._lock:
            previous = self._entries.pop(slot, None)
            if previous is not None:
                self._resident_bytes -= previous.nbytes
                self._replaced += 1
            self._entries[slot] = entry
            self._resident_bytes += nbytes
            self._published += 1
            self._evict_over_budget_locked(protect=slot)
        return True

    def take(self, slot, key):
        """Remove ``slot``'s entry and return its buffer only if ``key`` matches.

        A take always removes: a slot being committed under a different key will
        not want the old buffer on a later commit either, so holding it would
        only occupy budget that a current preparation needs.
        """

        with self._lock:
            entry = self._entries.pop(slot, None)
            if entry is None:
                self._misses += 1
                return None
            self._resident_bytes -= entry.nbytes
            if entry.key != key:
                self._stale += 1
                return None
            self._hits += 1
            return entry.buffer

    def holds(self, slot, key) -> bool:
        """Whether this exact preparation is already waiting.

        Callers submit preparation whenever a payload is admitted, and the same
        payload is legitimately re-offered many times while a governed fill
        works through it. Without this check each re-offer would repeat work a
        worker has already done and published.
        """

        with self._lock:
            entry = self._entries.get(slot)
            return entry is not None and entry.key == key

    def discard(self, slot) -> None:
        """Drop one slot's entry, if any."""

        with self._lock:
            entry = self._entries.pop(slot, None)
            if entry is not None:
                self._resident_bytes -= entry.nbytes

    def clear(self) -> None:
        """Drop every entry. Counters survive; they describe the run."""

        with self._lock:
            self._entries.clear()
            self._resident_bytes = 0

    def counters(self) -> PreparedUploadCounters:
        with self._lock:
            return PreparedUploadCounters(
                published=self._published,
                replaced=self._replaced,
                hits=self._hits,
                stale=self._stale,
                misses=self._misses,
                evicted=self._evicted,
                resident_entries=len(self._entries),
                resident_bytes=int(self._resident_bytes),
            )

    def _evict_over_budget_locked(self, *, protect=None) -> None:
        if not self._budget_bytes:
            return
        while self._resident_bytes > self._budget_bytes and len(self._entries) > 1:
            slot, entry = next(iter(self._entries.items()))
            if slot == protect:
                # The just-published entry is the one the next commit is most
                # likely to want; evict past it rather than undoing this publish.
                self._entries.move_to_end(slot)
                slot, entry = next(iter(self._entries.items()))
            self._entries.pop(slot, None)
            self._resident_bytes -= entry.nbytes
            self._evicted += 1


def prepared_upload_key(payload, variant) -> tuple:
    """Identity a prepared buffer is valid under.

    The acknowledgement identity already means "these exact pixels" everywhere
    else in the presentation protocol; reusing it here keeps the mailbox from
    inventing a second, weaker notion of sameness. ``variant`` carries whatever
    else the preparation depended on — the round levels a PyQtGraph assembly
    baked against, or the texture representation a WGPU pack targeted.
    """

    identity = getattr(payload, "tile_identity", None)
    if identity is None:
        identity = getattr(payload, "source_id", None)
    return (identity, variant)


__all__ = [
    "DEFAULT_PREPARED_UPLOAD_BUDGET_BYTES",
    "PreparedUpload",
    "PreparedUploadCounters",
    "PreparedUploadMailbox",
    "prepared_upload_key",
]
