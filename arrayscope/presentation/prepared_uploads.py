"""Worker-prepared backend upload buffers, held in a content-keyed mailbox.

At a tiled commit the GUI thread's irreducible job is to submit already-built
commands and release their buffers. Packing a payload into the array a backend
uploads is not part of that job: it is pure work over an immutable payload and
belongs on a worker. This module is the seam between the two.

The ownership model is deliberately one-way and content-addressed:

- A worker builds a buffer and **publishes** it under the exact identity it was
  built from. The buffer is immutable from that moment; the worker keeps no
  reference it will write through.
- The mailbox is keyed by that identity, so a buffer is found by *what it is*,
  never by where it was going. This is the correctness rule stated positively:
  a prepared buffer is a promise about which pixels these are, and presenting
  one under a different identity is a torn frame that costs more than preparing
  inline. A key that is not present is a miss — a normal outcome whose fallback
  is to prepare on the GUI thread exactly as before.
- The GUI thread **takes** a buffer by naming the identity it is about to
  submit. A take removes the entry.

Keying was originally by *slot*, with a matching key required on take. That is
strictly weaker, and a montage index scroll is where it shows: after one step
slot *i* wants exactly the buffer slot *i+1* was holding, so every prepared
buffer was refused as stale and re-prepared inline — 179 stale takes against 3
hits on one measured scrub (2026-07-31). The identity being asked for was in the
mailbox the whole time, one slot over.

Buffers are large, so the mailbox is byte-bounded. Overflow evicts the
least-recently-published entries, which are the ones a governed fill is least
likely to reach next.

The mailbox is a hand-off and only a hand-off: it answers "did a worker finish
this before the commit asked?". :class:`AssemblyRetention` answers the other,
independent question — "have we *ever* built these exact pixels?" — because a
montage index scroll re-asks for content it already owns, one slot over.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

# One 512x512 RGBA32F page is 4 MiB, so this holds a few dozen prepared tiles.
# The mailbox is a hand-off buffer, not a residency cache: entries live from
# admission to the commit that consumes them, and dropping one only costs the
# inline preparation that used to happen anyway. What a commit should be able
# to find *after* consuming it belongs in `AssemblyRetention`, below.
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
        published = hits + replaced + evicted + reset_discarded
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

    ``misses`` is the one to watch. It counts commits that asked for an
    identity nothing had prepared and packed it themselves; a rising ``misses``
    with a flat ``hits`` means preparation is chasing a target it never
    catches, which is a scheduling problem, not a correctness one. It counts
    only commits that really packed: a caller that asks ``holds`` and then
    declines the work has not fallen back to anything.

    There is deliberately no "prepared for something else" outcome any more.
    Under slot keying that was a real terminal state and the dominant one on a
    scroll; under content keying such a buffer is simply still in the mailbox
    under its own key, and it is either taken later or evicted.
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

        return int(self.misses)

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
    """Content-keyed hand-off of prepared buffers between threads.

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
        """Claim a task for ``key``, in the ``PLANNED`` state.

        ``slot`` is carried for the scheduler's benefit only — it names the
        tile a task belongs to for supersession and diagnostics. The buffer it
        eventually publishes is filed under ``key``.

        Returned before the scheduler has seen anything, because submission can
        itself fail: ``Kernel.submit`` returns None for a task refused at the
        door, and counting that as submitted-then-vanished is one of the ways
        the old equations did not close.
        """

        return PreparedUploadTask(self, int(slot), key, self.next_generation())

    def publish(self, key, buffer, *, nbytes: int | None = None, generation: int = 0) -> bool:
        """Store ``buffer`` under ``key``, unless a newer one is already there.

        ``nbytes`` is required when the buffer is not itself an array — the
        mailbox cannot bound what it cannot size, and silently treating a
        wrapper as free would let residency grow without limit.

        ``generation`` is the ordering token from ``next_generation``. Two
        buffers under one key are the same pixels by the key's own contract, so
        the ordering no longer decides which is *correct* — it decides which
        allocation is kept, and keeping the newer one keeps the arrival a
        straggler cannot displace.

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
            previous = self._entries.get(key)
            if previous is not None and generation and previous.generation > generation:
                self._superseded_publish += 1
                return False
            if previous is not None:
                self._entries.pop(key, None)
                self._resident_bytes -= previous.nbytes
                self._replaced += 1
            self._entries[key] = entry
            self._resident_bytes += nbytes
            self._published += 1
            self._evict_over_budget_locked(protect=key)
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

    def take(self, key):
        """Remove ``key``'s entry and return its buffer, or None if absent.

        There is no longer a "wrong buffer" outcome to report: a key that is
        not here simply was not prepared, and a buffer prepared for other
        pixels is sitting under its own key waiting for whoever wants it.
        """

        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                self._misses += 1
                return None
            self._resident_bytes -= entry.nbytes
            self._hits += 1
            return entry.buffer

    def holds(self, key) -> bool:
        """Whether this exact preparation is already waiting.

        Callers submit preparation whenever a payload is admitted, and the same
        payload is legitimately re-offered many times while a governed fill
        works through it. Without this check each re-offer would repeat work a
        worker has already done and published.

        Also lets a consumer ask "is this ready?" without consuming it, which
        is what a caller about to *decline* the work needs: counting a miss for
        a payload the caller then defers unassembled inflated ``misses`` with
        commits that never packed anything.
        """

        with self._lock:
            return key in self._entries

    def discard(self, key) -> None:
        """Drop one entry, if any, as a reset rather than a consumption."""

        with self._lock:
            entry = self._entries.pop(key, None)
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
            key, entry = next(iter(self._entries.items()))
            if key == protect:
                # The just-published entry is the one the next commit is most
                # likely to want; evict past it rather than undoing this publish.
                self._entries.move_to_end(key)
                key, entry = next(iter(self._entries.items()))
            self._entries.pop(key, None)
            self._resident_bytes -= entry.nbytes
            self._evicted += 1


# A retained assembly's pixels are, in the steady state, the very pixels an
# ImageItem is already holding: the item keeps the array alive whether or not
# this index exists. What the index buys is the ability to *find* them again,
# so the marginal residency it adds is only the entries no item still draws.
# Sized like the mailbox for the same reason — a scroll grid of large complex
# tiles has to fit, a whole 272-tile field does not have to.
DEFAULT_ASSEMBLY_RETENTION_BUDGET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class AssemblyRetentionCounters:
    """Whether re-derivation is being avoided, and what it costs to avoid it."""

    hits: int = 0
    misses: int = 0
    stored: int = 0
    replaced: int = 0
    evicted: int = 0
    reset_discarded: int = 0
    resident_entries: int = 0
    resident_bytes: int = 0


@dataclass(frozen=True)
class RetainedAssembly:
    """Everything one assembly derived, minus the payload it derived it from.

    Deliberately not the assembled payload object: that object names the slot
    it was built for, and the whole point of retaining it is to serve a
    *different* slot on the next scroll step. What is reusable is the derived
    part — the pixels and the fields the assembler overwrote — which is exactly
    what ``fields`` holds, ready to be applied to whichever payload asks next.
    """

    fields: dict
    extras: tuple
    nbytes: int


def derived_payload_fields(source, derived) -> dict:
    """Fields ``derived`` replaced, by object identity rather than value.

    ``dataclasses.replace`` copies untouched fields by reference, so an
    identity comparison names precisely what the assembler produced and never
    has to compare two arrays elementwise. A field the assembler left alone is
    the same object and is not carried, which is what keeps slot-owned fields
    (``tile_number`` above all) out of the retained delta.
    """

    from dataclasses import fields as dataclass_fields

    return {
        field.name: getattr(derived, field.name)
        for field in dataclass_fields(derived)
        if getattr(derived, field.name) is not getattr(source, field.name, None)
    }


class AssemblyRetention:
    """Content-keyed retention of derived tile pixels, byte-bounded, LRU.

    A montage index scroll moves content between slots: after one step, slot
    *i* shows what slot *i+1* showed. The pixels are identical and are still
    on screen — but the commit re-derived them, because the hand-off that
    could have carried them is keyed by slot and consumed on read. Re-deriving
    also mints a new array, and PyQtGraph item residency is keyed on the
    identity of that array (``_direct_payload_source_id``), so the fresh
    assembly missed the resident item too and paid a second, avoidable upload.

    Keying on content instead of slot fixes both at once: the same array object
    comes back, so the resident item matches and neither the assembly nor the
    upload runs. The key is the one the mailbox already trusts to hand a
    worker-built buffer to a commit, so this store makes no new promise about
    when two payloads may share pixels.

    Single-threaded by contract — the GUI commit path owns it. The mailbox is
    the thing with a lock, because the mailbox is the thing workers touch.
    """

    def __init__(self, *, budget_bytes: int = DEFAULT_ASSEMBLY_RETENTION_BUDGET_BYTES) -> None:
        self._budget_bytes = max(0, int(budget_bytes))
        self._entries: OrderedDict[object, RetainedAssembly] = OrderedDict()
        self._resident_bytes = 0
        self._hits = 0
        self._misses = 0
        self._stored = 0
        self._replaced = 0
        self._evicted = 0
        self._reset_discarded = 0

    def get(self, key) -> RetainedAssembly | None:
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return entry

    def put(self, key, *, source, derived, extras: tuple = (), nbytes: int = 0) -> None:
        fields = derived_payload_fields(source, derived)
        if not fields:
            # The assembler returned its input unchanged, so there is nothing
            # to retain and a hit would save nothing.
            return
        self.store(key, fields=fields, extras=extras, nbytes=nbytes)

    def store(self, key, *, fields: dict, extras: tuple = (), nbytes: int = 0) -> None:
        nbytes = max(0, int(nbytes))
        if self._budget_bytes and nbytes > self._budget_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._resident_bytes -= previous.nbytes
            self._replaced += 1
        self._entries[key] = RetainedAssembly(
            fields=dict(fields), extras=tuple(extras), nbytes=nbytes
        )
        self._resident_bytes += nbytes
        self._stored += 1
        self._evict_over_budget(protect=key)

    def clear(self) -> None:
        """Drop everything at a residency reset boundary."""

        self._reset_discarded += len(self._entries)
        self._entries.clear()
        self._resident_bytes = 0

    def counters(self) -> AssemblyRetentionCounters:
        return AssemblyRetentionCounters(
            hits=self._hits,
            misses=self._misses,
            stored=self._stored,
            replaced=self._replaced,
            evicted=self._evicted,
            reset_discarded=self._reset_discarded,
            resident_entries=len(self._entries),
            resident_bytes=int(self._resident_bytes),
        )

    def _evict_over_budget(self, *, protect=None) -> None:
        if not self._budget_bytes:
            return
        while self._resident_bytes > self._budget_bytes and len(self._entries) > 1:
            key = next(iter(self._entries))
            if key == protect:
                # Evicting what was just stored would make the store a no-op
                # with all of its cost; drop the next-least-recent instead.
                self._entries.move_to_end(key)
                key = next(iter(self._entries))
            entry = self._entries.pop(key)
            self._resident_bytes -= entry.nbytes
            self._evicted += 1


def prepared_upload_key(payload, variant) -> tuple:
    """Identity a prepared buffer is valid under.

    The acknowledgement identity names semantic/LOD truth, but deliberately
    ignores physical plane provenance. That is too weak for a prepared buffer:
    two fallback payloads can satisfy the same target while resolving different
    actual page sets. ``variant`` carries whatever else the preparation depended
    on — the CPU mapping a PyQtGraph assembly baked against, or the texture
    representation a WGPU pack targeted.

    ``source_id`` was originally trusted to carry the physical distinction, and
    for the *semantically* distinct case it does. Progressive refinement is the
    case it does not cover: the same tile keeps one ``source_id`` and one
    identity while its coarse pages are replaced by fine ones, and the derived
    pixels change completely (``test_pyqtgraph_replaces_coarse_page_pixels_
    when_fine_pages_arrive``). The resolved page set therefore belongs in the
    key. The mailbox survived this because ``take`` consumes and a publish is
    rarely outstanding across a refinement; a store that *retains* would serve
    the coarse pixels forever.
    """

    identity = getattr(payload, "tile_identity", None)
    source_id = getattr(payload, "source_id", None)
    return (identity, source_id, page_binding_token(payload), variant)


def page_binding_token(payload) -> tuple | None:
    """Which actual pages a page-backed payload currently resolves to.

    ``None`` for a payload with no page backing: there is nothing to refine, so
    identity and source id already say everything about its pixels.
    """

    backing = getattr(payload, "page_backing", None)
    if backing is None:
        return None
    return tuple(
        (target, None if resolution is None else resolution.actual_key)
        for target, resolution in zip(
            backing.requested_keys,
            backing.candidate_resolutions,
            strict=True,
        )
    )


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
    "DEFAULT_ASSEMBLY_RETENTION_BUDGET_BYTES",
    "DEFAULT_PREPARED_UPLOAD_BUDGET_BYTES",
    "AssemblyRetention",
    "AssemblyRetentionCounters",
    "PreparedUpload",
    "PreparedUploadCounters",
    "PreparedUploadMailbox",
    "PreparedUploadTask",
    "RetainedAssembly",
    "cpu_mapping_preparation_variant",
    "derived_payload_fields",
    "page_binding_token",
    "prepared_upload_key",
]
