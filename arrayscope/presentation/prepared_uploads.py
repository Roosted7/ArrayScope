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
    generation: int = 0


@dataclass(frozen=True)
class PreparedUploadCounters:
    """Where every preparation went, without a debugger.

    The point of counting this finely is that "the hand-off works" and "the
    hand-off is worth its cost" are different claims, and only the second one
    matters. A high ``hits`` proves the seam is wired up; it says nothing about
    the preparations that were planned and never used.

    Two populations are tracked, and they are different things. A **task** is a
    unit of planned work; a **buffer** is what a task may leave behind. Each has
    exactly one terminal outcome, and both close while work is still in flight,
    not only at a settled final snapshot:

        planned  = submitted + deduped + skipped_resident + skipped_no_work
                   + skipped_stale_round + submit_rejected
        submitted = pending + in_flight + published + publication_refused
                    + failed_after_start + dropped_before_start
        published = hits + stale + replaced + evicted + reset_discarded
                    + resident_entries

    ``in_flight`` carries its weight twice over: it is the bound on concurrent
    output allocation the byte budget cannot see, *and* it is the "entered a
    worker but has not settled yet" state without which the task equation does
    not close mid-flight.

    An earlier version derived "superseded before execution" as
    ``submitted - executed``, which silently counted every task still sitting in
    the scheduler as though it had been dropped. ``pending`` is now a real state
    and ``dropped_before_start`` a real outcome, so the equation holds at any
    instant rather than only once the queue has drained.

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
    skipped_stale_round: int = 0
    submit_rejected: int = 0
    # Task states and terminal outcomes. Every submitted task is in exactly one.
    pending: int = 0
    started: int = 0
    published: int = 0
    publication_refused: int = 0
    failed_after_start: int = 0
    dropped_before_start: int = 0
    # Why a publication was refused, as a breakdown of publication_refused.
    superseded_publish: int = 0
    oversized: int = 0
    # Buffer outcomes: a published buffer leaves the mailbox exactly one way.
    hits: int = 0
    stale: int = 0
    replaced: int = 0
    evicted: int = 0
    reset_discarded: int = 0
    # Commit-side view.
    misses: int = 0
    # Live state.
    resident_entries: int = 0
    resident_bytes: int = 0
    in_flight: int = 0
    peak_in_flight: int = 0

    @property
    def inline_fallbacks(self) -> int:
        """Commits that packed on the GUI thread after all."""

        return int(self.misses) + int(self.stale)

    def task_accounting_error(self) -> int:
        """Submitted tasks minus every state one can be in. Must be zero.

        Holds at any instant, including with tasks queued and tasks inside a
        worker closure — which is the whole point of checking it.
        """

        return int(self.submitted) - (
            int(self.pending)
            + int(self.in_flight)
            + int(self.published)
            + int(self.publication_refused)
            + int(self.failed_after_start)
            + int(self.dropped_before_start)
        )

    def buffer_accounting_error(self) -> int:
        """Published buffers minus everything that could have become of them."""

        return int(self.published) - (
            int(self.hits)
            + int(self.stale)
            + int(self.replaced)
            + int(self.evicted)
            + int(self.reset_discarded)
            + int(self.resident_entries)
        )


class PreparedUploadTask:
    """One planned preparation, from planning to exactly one terminal outcome.

    The worker side is a context manager, so the in-flight window is closed in
    ``finally`` and cannot be leaked by an exception. Before this existed the
    closure raised straight past the decrement and the gauge stayed high for the
    life of the process, which is precisely the case an in-flight bound is for.

    Nothing here touches Qt. ``dropped`` is called from the kernel's completion
    drain, and it only takes the mailbox's own lock.
    """

    __slots__ = ("_key", "_lock", "_mailbox", "_settled", "_slot", "_started", "generation")

    def __init__(self, mailbox: PreparedUploadMailbox, slot: int, key, generation: int) -> None:
        self._mailbox = mailbox
        self._slot = int(slot)
        self._key = key
        self.generation = int(generation)
        self._started = False
        self._settled = False
        # `dropped` arrives on the GUI thread while `__enter__` may be running
        # on a worker. The kernel's own ordering makes the race unlikely rather
        # than impossible, and "exactly one terminal outcome" is the property
        # this whole class exists to provide.
        self._lock = threading.Lock()

    def _claim(self) -> bool:
        """Take the single terminal-outcome slot, or report it already taken."""

        with self._lock:
            if self._settled:
                return False
            self._settled = True
            return True

    @property
    def slot(self) -> int:
        return self._slot

    @property
    def key(self):
        return self._key

    def submitted(self) -> None:
        """The scheduler accepted it; it is now pending."""

        self._mailbox._task_submitted()

    def submit_rejected(self) -> None:
        """Roll back admission when the scheduler refused it at the door."""

        if self._claim():
            self._mailbox._task_submit_rejected()

    def dropped(self) -> None:
        """Superseded or cancelled. Only terminal if it never entered a worker.

        Idempotent, and deliberately a no-op once the closure has run: the
        kernel reports staleness for tasks that completed and were then
        superseded, and those already have a terminal outcome of their own.
        """

        with self._lock:
            if self._settled or self._started:
                return
            self._settled = True
        self._mailbox._task_dropped_before_start()

    def __enter__(self) -> PreparedUploadTask:
        with self._lock:
            self._started = True
        self._mailbox._task_started()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._claim():
            self._mailbox._task_finished("failed" if exc_type is not None else "refused")
        return False

    def publish(self, buffer, *, nbytes: int | None = None) -> bool:
        """Publish this task's buffer and settle its outcome.

        Settling here rather than in ``__exit__`` is what distinguishes a
        refused publication from a closure that simply never called publish;
        both leave the mailbox empty, and only one of them is a bug.
        """

        stored = self._mailbox.publish(
            self._slot,
            self._key,
            buffer,
            nbytes=nbytes,
            generation=self.generation,
        )
        if self._claim():
            self._mailbox._task_finished("published" if stored else "refused")
        return stored


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
        self._skipped_stale_round = 0
        self._submit_rejected = 0
        self._pending = 0
        self._started = 0
        self._published_tasks = 0
        self._publication_refused = 0
        self._failed_after_start = 0
        self._dropped_before_start = 0
        self._superseded_publish = 0
        self._oversized = 0
        self._published = 0
        self._replaced = 0
        self._evicted = 0
        self._reset_discarded = 0
        self._hits = 0
        self._stale = 0
        self._misses = 0
        self._in_flight = 0
        self._peak_in_flight = 0
        self._generation = 0

    def next_generation(self) -> int:
        """Claim the ordering token for one planned preparation.

        Handed out on the GUI thread at planning time, which is the only place
        that sees preparations for a slot in the order the round wants them.
        Workers finish out of order; this is what lets ``publish`` tell a late
        straggler from a genuinely newer buffer.
        """

        with self._lock:
            self._generation += 1
            return self._generation

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

    def note_stale_round(self, count: int = 1) -> None:
        """Planning skipped payloads whose round levels have not caught up.

        Distinct from every other skip: these are payloads the hand-off *would*
        have prepared, refused because the buffer would be guaranteed stale
        rather than merely unlikely to be used.
        """

        with self._lock:
            self._skipped_stale_round += max(0, int(count))

    def plan(self, slot, key) -> PreparedUploadTask:
        """Claim a task for ``slot``, in the ``PLANNED`` state.

        Returned before the scheduler has seen anything, because submission can
        itself fail: ``Kernel.submit`` returns None for a task refused at the
        door, and counting that as submitted-then-vanished is one of the ways
        the old equations did not close.
        """

        return PreparedUploadTask(self, int(slot), key, self.next_generation())

    def publish(self, slot, key, buffer, *, nbytes: int | None = None, generation: int = 0) -> bool:
        """Store ``buffer`` for ``slot``, unless that slot already holds newer.

        ``nbytes`` is required when the buffer is not itself an array — the
        mailbox cannot bound what it cannot size, and silently treating a
        wrapper as free would let residency grow without limit.

        ``generation`` is the ordering token from ``next_generation``. A running
        preparation cannot be recalled once it has started, so a superseded one
        still finishes and still arrives here — after the replacement it lost
        to, if that replacement was quicker. Overwriting on arrival order would
        let the straggler evict the buffer the commit actually wants, turning a
        hit into a stale take and an inline pack. Refusing the older generation
        keeps the newest prepared buffer, which is what a keep-latest mailbox
        promises.

        Returns False when the buffer is not stored — refused as older, or too
        large for the whole budget to be worth evicting everything else for. The
        consumer's inline path covers both.

        Task accounting belongs to :class:`PreparedUploadTask`; publishing
        without one is supported for tests and for callers that never entered a
        worker closure.
        """

        nbytes = _buffer_nbytes(buffer) if nbytes is None else max(0, int(nbytes))
        generation = int(generation)
        entry = PreparedUpload(key=key, buffer=buffer, nbytes=nbytes, generation=generation)
        with self._lock:
            if self._budget_bytes and nbytes > self._budget_bytes:
                self._oversized += 1
                return False
            previous = self._entries.get(slot)
            if previous is not None and generation and previous.generation > generation:
                self._superseded_publish += 1
                return False
            if previous is not None:
                self._entries.pop(slot, None)
                self._resident_bytes -= previous.nbytes
                self._replaced += 1
            self._entries[slot] = entry
            self._resident_bytes += nbytes
            self._published += 1
            self._evict_over_budget_locked(protect=slot)
        return True

    # ------------------------------------------------- task state transitions

    def _task_submitted(self) -> None:
        with self._lock:
            self._submitted += 1
            self._pending += 1

    def _task_submit_rejected(self) -> None:
        with self._lock:
            # Admission is recorded before Kernel.submit(), because submit may
            # wake an inline backend (or a fast worker) before it returns. A
            # refusal is the one path that rolls that provisional admission
            # back out of the submitted population.
            self._submitted = max(0, self._submitted - 1)
            self._pending = max(0, self._pending - 1)
            self._submit_rejected += 1

    def _task_started(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)
            self._started += 1
            self._in_flight += 1
            self._peak_in_flight = max(self._peak_in_flight, self._in_flight)

    def _task_finished(self, outcome: str) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            if outcome == "published":
                self._published_tasks += 1
            elif outcome == "failed":
                self._failed_after_start += 1
            else:
                self._publication_refused += 1

    def _task_dropped_before_start(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)
            self._dropped_before_start += 1

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
        """Drop one slot's entry, if any, as a reset rather than a consumption."""

        with self._lock:
            entry = self._entries.pop(slot, None)
            if entry is not None:
                self._resident_bytes -= entry.nbytes
                self._reset_discarded += 1

    def clear(self) -> None:
        """Drop every entry at a residency reset. Counters survive the run.

        Each dropped buffer takes the ``reset_discarded`` outcome. Clearing used
        to remove entries silently, which left `published` permanently ahead of
        everything that could be said to have become of those buffers — the
        buffer equation only closed on a run that never reset.
        """

        with self._lock:
            self._reset_discarded += len(self._entries)
            self._entries.clear()
            self._resident_bytes = 0

    def counters(self) -> PreparedUploadCounters:
        with self._lock:
            return PreparedUploadCounters(
                submitted=self._submitted,
                deduped=self._deduped,
                skipped_resident=self._skipped_resident,
                skipped_no_work=self._skipped_no_work,
                skipped_stale_round=self._skipped_stale_round,
                submit_rejected=self._submit_rejected,
                pending=self._pending,
                started=self._started,
                published=self._published_tasks,
                publication_refused=self._publication_refused,
                failed_after_start=self._failed_after_start,
                dropped_before_start=self._dropped_before_start,
                superseded_publish=self._superseded_publish,
                oversized=self._oversized,
                replaced=self._replaced,
                evicted=self._evicted,
                reset_discarded=self._reset_discarded,
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
    "PreparedUploadTask",
    "cpu_mapping_preparation_variant",
    "prepared_upload_key",
]
