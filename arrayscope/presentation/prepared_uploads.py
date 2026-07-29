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
from dataclasses import replace as dataclass_replace

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
    """Where every preparation went, without a debugger.

    The point of counting this finely is that "the hand-off works" and "the
    hand-off is worth its cost" are different claims, and only the second one
    matters. A high ``hits`` proves the seam is wired up; it says nothing about
    the preparations that were planned and never used. These counters close
    over the whole population, so wasted work has nowhere to hide:

        planned = submitted + deduped + skipped_resident + skipped_no_work
        submitted = executed + superseded_before_execution
        executed ≈ published + rejected
        published = hits + stale + replaced + evicted + resident_entries
        inline fallbacks at the commit = misses + stale

    ``stale`` is the one to watch. It counts buffers that were prepared and
    then correctly refused because the slot moved on; a rising ``stale`` with a
    flat ``hits`` means preparation is chasing a target it never catches, which
    is a scheduling problem, not a correctness one.
    """

    # Planning: what the GUI thread decided to do about each payload.
    submitted: int = 0
    deduped: int = 0
    skipped_resident: int = 0
    skipped_no_work: int = 0
    # Execution: what the workers did with what they were given.
    executed: int = 0
    published: int = 0
    rejected: int = 0
    # Displacement: buffers that were published and then lost their slot.
    replaced: int = 0
    evicted: int = 0
    # Consumption: what the commit found when it came to submit.
    hits: int = 0
    stale: int = 0
    misses: int = 0
    # Live state.
    resident_entries: int = 0
    resident_bytes: int = 0
    in_flight: int = 0
    peak_in_flight: int = 0

    @property
    def superseded_before_execution(self) -> int:
        """Submitted preparations the scheduler dropped before they ran.

        Not a counter of its own: per-slot supersession drops the queued record
        without running the closure, so the only honest way to see it is the
        gap between what was submitted and what a worker actually entered.
        """

        return max(0, int(self.submitted) - int(self.executed))

    @property
    def inline_fallbacks(self) -> int:
        """Commits that packed on the GUI thread after all."""

        return int(self.misses) + int(self.stale)


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
        self._submitted = 0
        self._deduped = 0
        self._skipped_resident = 0
        self._skipped_no_work = 0
        self._executed = 0
        self._published = 0
        self._rejected = 0
        self._replaced = 0
        self._evicted = 0
        self._hits = 0
        self._stale = 0
        self._misses = 0
        self._in_flight = 0
        self._peak_in_flight = 0

    def note_submitted(self, count: int = 1) -> None:
        """One planned preparation reached the scheduler."""

        with self._lock:
            self._submitted += max(0, int(count))

    def note_deduped(self, count: int = 1) -> None:
        """A planned preparation was already waiting, so it was not resubmitted."""

        with self._lock:
            self._deduped += max(0, int(count))

    def note_resident(self, count: int = 1) -> None:
        """Planning skipped a payload the backend already has physically."""

        with self._lock:
            self._skipped_resident += max(0, int(count))

    def note_no_work(self, count: int = 1) -> None:
        """Planning skipped a payload whose preparation would allocate nothing."""

        with self._lock:
            self._skipped_no_work += max(0, int(count))

    def note_executed(self, count: int = 1) -> None:
        """A worker entered a preparation closure.

        Called by the closure itself rather than inferred from a completion, so
        that ``submitted - executed`` is exactly the population the scheduler
        dropped before it ran. Also raises the in-flight gauge: between here and
        ``publish`` the worker holds an output allocation the mailbox's byte
        budget cannot see.
        """

        with self._lock:
            self._executed += max(0, int(count))
            self._in_flight += max(0, int(count))
            self._peak_in_flight = max(self._peak_in_flight, self._in_flight)

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
        entry = PreparedUpload(key=key, buffer=buffer, nbytes=nbytes)
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1
            if self._budget_bytes and nbytes > self._budget_bytes:
                self._rejected += 1
                return False
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
                submitted=self._submitted,
                deduped=self._deduped,
                skipped_resident=self._skipped_resident,
                skipped_no_work=self._skipped_no_work,
                executed=self._executed,
                published=self._published,
                rejected=self._rejected,
                replaced=self._replaced,
                evicted=self._evicted,
                hits=self._hits,
                stale=self._stale,
                misses=self._misses,
                resident_entries=len(self._entries),
                resident_bytes=int(self._resident_bytes),
                in_flight=int(self._in_flight),
                peak_in_flight=int(self._peak_in_flight),
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

    The acknowledgement identity names semantic/LOD truth, but deliberately
    ignores physical plane provenance. That is too weak for a prepared buffer:
    two fallback payloads can satisfy the same target while resolving different
    actual page sets. ``source_id`` carries that physical payload distinction.
    ``variant`` carries whatever else the preparation depended on — the CPU
    mapping a PyQtGraph assembly baked against, or the texture representation a
    WGPU pack targeted.
    """

    identity = getattr(payload, "tile_identity", None)
    source_id = getattr(payload, "source_id", None)
    return (identity, source_id, variant)


def cpu_mapping_preparation_variant(payload, levels) -> tuple:
    """Exact CPU display mapping a prepared page assembly was baked through.

    PyQtGraph maps complex pages while assembling them, so levels alone are
    insufficient: scale, symlog constant and LUT can all change output pixels
    without changing ``TileIdentity``. ``ShaderMapping.identity_key`` is the
    canonical hashable description of those inputs. The accepted transaction
    levels override any older levels still carried by the payload wrapper.
    """

    bounds = (float(levels[0]), float(levels[1]))
    mapping = getattr(payload, "shader_mapping", None)
    if mapping is None:
        mapping_key = None
    else:
        mapping_key = dataclass_replace(mapping, levels=bounds).identity_key
    return ("cpu-page-assembly", bounds, mapping_key)


__all__ = [
    "DEFAULT_PREPARED_UPLOAD_BUDGET_BYTES",
    "PreparedUpload",
    "PreparedUploadCounters",
    "PreparedUploadMailbox",
    "cpu_mapping_preparation_variant",
    "prepared_upload_key",
]
