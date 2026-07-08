"""Single-owner tile lifecycle state machine (ADR 0051).

One :class:`TileLifecycle` instance per montage render session owns every
tile's lifecycle along three orthogonal axes:

- **semantic** — is the exact value computed? (``unplanned → planned →
  evaluating → evaluated | declined | skipped``)
- **residency** — which pyramid levels exist for this tile's source, and who
  claimed them (``claimed(owner) → materializing → resident`` per level key)
- **presentation** — what does the backend show, confirmed only by commit
  acknowledgement (``unpresented → emitted → presented | parked``)

The machine is a functional core: every input is an explicit event method,
every output is a return value (effect tuples / re-arm lists); it performs no
I/O, no Qt, and no scheduling.  Callers execute effects and MUST report
outcomes back as events — silence is the defect class this module exists to
kill (ADR 0050's leaked claims, wedged stage keys, immortal "loading" tiles,
and idle upsert loops were all optimistic bookkeeping around ignored returns).

Structural rules (ADR 0051):

1. Nothing enters ``PRESENTED`` except ``commit_acknowledged`` with backend
   acceptance.
2. Every claim reaches ``RESIDENT`` or is released; ``session_replaced`` and
   every declined path emit ``ReleaseClaim`` effects mechanically, and
   :meth:`dangling_claims` makes a leak a queryable fact.
3. Emit-once, park, re-arm: a declined out-of-scope upsert parks at
   acknowledgement; only a scope change re-arms it.
4. The machine never assumes an effect ran.
5. (Backend, P4) derived GPU state is invalidated at upload granularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator


class Semantic(str, Enum):
    UNPLANNED = "unplanned"
    PLANNED = "planned"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    DECLINED = "declined"
    SKIPPED = "skipped"


class Presentation(str, Enum):
    UNPRESENTED = "unpresented"
    EMITTED = "emitted"
    PRESENTED = "presented"
    PARKED = "parked"


class LevelPhase(str, Enum):
    CLAIMED = "claimed"
    MATERIALIZING = "materializing"
    RESIDENT = "resident"
    RELEASED = "released"


class ClaimOwner(str, Enum):
    EVALUATION = "evaluation"
    CHAIN = "chain"
    WALK = "walk"
    INGEST = "ingest"
    PREVIEW = "preview"


@dataclass(frozen=True)
class ReleaseClaim:
    """Effect: the caller must release this singleflight claim on the pyramid."""

    tile_number: int
    level_key: object
    owner: ClaimOwner


@dataclass
class _LevelEntry:
    phase: LevelPhase
    owner: ClaimOwner
    request: object = None
    order: int = 0


@dataclass
class TileRecord:
    tile_number: int
    semantic: Semantic = Semantic.UNPLANNED
    presentation: Presentation = Presentation.UNPRESENTED
    #: payload identity last emitted toward the backend (emit-once key).
    emitted_source_id: object = None
    #: payload identity the backend acknowledged as shown.
    presented_source_id: object = None
    #: payload identity the backend most recently reported for this slot.
    backend_source_id: object = None
    parked_reason: str = ""
    levels: dict = field(default_factory=dict)  # level_key -> _LevelEntry
    #: (wanted_identity, backend_identity) pairs the machine gave up on after
    #: bounded identity rejections: the backend would not converge, so
    #: convergence passes must not reopen the loop for exactly these pairs.
    #: Cleared by a fresh semantic result.
    resigned: set = field(default_factory=set)
    #: P2 sets-as-views: this tile still owes exact content to the screen.
    #: Set at plan/demote/dequeue time; cleared mechanically when the
    #: backend confirms an EVALUATED tile's payload (rule 1), when the tile
    #: parks out of scope (rule 3), when it is skipped, or when the caller
    #: explicitly descopes it.  The legacy ``loading_tiles`` set is a view
    #: over this flag — the 2026-07-05 auto-levels wedge was exactly a
    #: confirmed-presented tile whose parallel ``loading_tiles`` entry no
    #: code path owned.
    load_intent: bool = False
    #: P2 sets-as-views: an evaluation request for this tile is in flight
    #: (the legacy ``active_tile_requests`` set is a view over this flag).
    request_active: bool = False
    #: P2 stage fan-in: the reusable-stage key this tile waits on, or None.
    #: Recorded by ``stage_attached`` so "loading without an evaluation
    #: request" is a queryable machine fact instead of set correlation.
    stage_key: object = None


class TileLifecycle:
    """Single owner of per-tile lifecycle state for one render session."""

    def __init__(self) -> None:
        self._records: dict[int, TileRecord] = {}
        self._parked: set[int] = set()
        self._evaluating: set[int] = set()
        self._presented: set[int] = set()
        self._loading: set[int] = set()
        self._active_requests: set[int] = set()
        self._skipped: set[int] = set()
        self._stage_blocked: dict[object, set[int]] = {}
        self._identity_rejections = 0
        #: (tile, emitted_id, backend_id) -> consecutive rejection count; a
        #: pair rejected IDENTITY_RESIGN_AFTER times is resigned (see below).
        self._identity_rejection_counts: dict[tuple[int, object, object], int] = {}
        self._level_order = 0

    #: After this many identical rejections the machine records the backend's
    #: identity as the presented truth and stops the re-emit loop: a backend
    #: that keeps a slot on the same wrong identity is not converging, and an
    #: unbounded upload loop is worse than a diagnosed stale tile
    #: (backend_stale_identities stays nonzero — the wedge stays visible).
    IDENTITY_RESIGN_AFTER = 3

    @property
    def identity_rejections(self) -> int:
        """Acks refused because the backend slot held a different identity."""

        return int(self._identity_rejections)

    @property
    def backend_presented_identities(self) -> dict[int, object]:
        """Latest backend slot identity snapshot, owned by the lifecycle."""

        return {
            int(rec.tile_number): rec.backend_source_id
            for rec in self._records.values()
            if rec.backend_source_id is not None
        }

    # -- record access -----------------------------------------------------

    def record(self, tile_number: int) -> TileRecord:
        index = int(tile_number)
        rec = self._records.get(index)
        if rec is None:
            rec = TileRecord(index)
            self._records[index] = rec
        return rec

    def peek(self, tile_number: int) -> TileRecord | None:
        return self._records.get(int(tile_number))

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[TileRecord]:
        return iter(self._records.values())

    def feedback_signature(self, tile_numbers: Iterable[int] = ()) -> tuple[tuple[object, ...], ...]:
        """Compact lifecycle-owned work signature for feedback reuse.

        The governor should reset learned pacing when the presentation work
        changes class.  Texture shape and bytes belong to the display layer,
        but preview/exact ownership, level phase, and presented/emitted state
        are lifecycle facts; expose them here so callers do not rediscover the
        machine state through parallel counters.
        """

        scope = {int(tile) for tile in tile_numbers}
        records = (
            (self._records.get(tile) for tile in sorted(scope))
            if scope
            else tuple(self._records.get(tile) for tile in sorted(self._records))
        )
        groups: set[tuple[object, ...]] = set()
        for rec in records:
            if rec is None:
                continue
            level_groups = tuple(
                sorted(
                    (
                        str(entry.owner.value),
                        str(entry.phase.value),
                        getattr(key, "component", None),
                        tuple(getattr(key, "level_xy", ()) or ()),
                    )
                    for key, entry in rec.levels.items()
                    if entry.phase is not LevelPhase.RELEASED
                )
            )
            group = (
                str(rec.semantic.value),
                str(rec.presentation.value),
                level_groups,
            )
            groups.add(group)
        return tuple(sorted(groups, key=repr))

    # -- semantic axis -----------------------------------------------------

    def plan_applied(self, tile_numbers: Iterable[int]) -> None:
        for tile_number in tile_numbers:
            rec = self.record(tile_number)
            if rec.semantic in (Semantic.UNPLANNED, Semantic.DECLINED):
                rec.semantic = Semantic.PLANNED

    def evaluation_started(self, tile_number: int) -> None:
        rec = self.record(tile_number)
        rec.semantic = Semantic.EVALUATING
        self._evaluating.add(rec.tile_number)
        self._skipped.discard(rec.tile_number)

    def evaluation_completed(self, tile_number: int) -> None:
        """A fresh semantic result exists; any prior presentation identity is stale."""

        rec = self.record(tile_number)
        rec.semantic = Semantic.EVALUATED
        self._evaluating.discard(rec.tile_number)
        self._skipped.discard(rec.tile_number)
        self._request_cleared(rec)
        self._stage_unbound(rec)
        # Fresh identity: the previous emit/park no longer refers to what the
        # next commit will carry, and the backend has not seen the new one.
        # Resignations are per stale pair — a new result gets fresh chances.
        self._unpark(rec)
        self._presented.discard(rec.tile_number)
        rec.presentation = Presentation.UNPRESENTED
        rec.presented_source_id = None
        rec.emitted_source_id = None
        rec.resigned.clear()

    def evaluation_declined(self, tile_number: int) -> tuple[ReleaseClaim, ...]:
        """Admission declined or work dropped: back to planned, claims released."""

        rec = self.record(tile_number)
        if rec.semantic is Semantic.EVALUATING:
            rec.semantic = Semantic.PLANNED
        self._evaluating.discard(rec.tile_number)
        self._request_cleared(rec)
        return self._release_owned(rec, ClaimOwner.EVALUATION)

    def evaluation_dropped(self, tile_number: int) -> None:
        """A computed result was discarded (eviction/rebuild): back to planned.

        P2: ``rendered_tiles`` routes every write through the machine, so a
        result leaving the session must demote the semantic axis — park
        eligibility (rule 3) reads ``EVALUATED`` as "a re-presentable result
        exists", which is no longer true.
        """

        rec = self._records.get(int(tile_number))
        if rec is not None and rec.semantic is Semantic.EVALUATED:
            rec.semantic = Semantic.PLANNED

    def tile_skipped(self, tile_number: int) -> None:
        rec = self.record(tile_number)
        rec.semantic = Semantic.SKIPPED
        self._evaluating.discard(rec.tile_number)
        self._skipped.add(rec.tile_number)
        self._load_cleared(rec)
        self._request_cleared(rec)
        self._stage_unbound(rec)

    def tile_unskipped(self, tile_number: int) -> None:
        """A skipped tile re-enters the plan (index-window demote path)."""

        rec = self._records.get(int(tile_number))
        if rec is not None and rec.semantic is Semantic.SKIPPED:
            rec.semantic = Semantic.PLANNED
            self._skipped.discard(rec.tile_number)

    # -- load intent / evaluation requests (P2 sets-as-views) ---------------

    def load_marked(self, tile_number: int) -> None:
        """This tile owes exact content to the screen (legacy ``loading_tiles``)."""

        rec = self.record(tile_number)
        rec.load_intent = True
        self._loading.add(rec.tile_number)

    def load_cleared(self, tile_number: int) -> None:
        rec = self._records.get(int(tile_number))
        if rec is not None:
            self._load_cleared(rec)

    def evaluation_requested(self, tile_number: int) -> None:
        """An evaluation request is in flight (legacy ``active_tile_requests``)."""

        rec = self.record(tile_number)
        rec.request_active = True
        self._active_requests.add(rec.tile_number)

    def evaluation_request_cleared(self, tile_number: int) -> None:
        rec = self._records.get(int(tile_number))
        if rec is not None:
            self._request_cleared(rec)

    def evaluation_requests_cleared(self) -> None:
        """Session retarget: every in-flight request is superseded wholesale."""

        for index in tuple(self._active_requests):
            self._request_cleared(self._records[index])

    # -- stage fan-in (P2: stages report through events) ---------------------

    def stage_attached(self, tile_number: int, stage_key: object) -> None:
        """This tile's evaluation waits on a reusable stage materialization.

        A stage-blocked tile is loading WITHOUT an evaluation request BY
        DESIGN; recording the binding makes that a per-record fact the
        dispatch derivation can read instead of correlating parallel sets.
        """

        rec = self.record(tile_number)
        self._stage_unbound(rec)
        rec.stage_key = stage_key
        self._stage_blocked.setdefault(stage_key, set()).add(rec.tile_number)

    def stage_detached(self, tile_number: int) -> None:
        rec = self._records.get(int(tile_number))
        if rec is not None:
            self._stage_unbound(rec)

    def stage_resolved(self, stage_key: object) -> tuple[int, ...]:
        """A stage completed/failed/released: unbind every stage-blocked tile.

        Returns the tiles that were waiting so the caller can route them to
        evaluation (value arrived) or requeue/decline (stage lost).
        """

        waiting = tuple(sorted(self._stage_blocked.pop(stage_key, ())))
        for index in waiting:
            rec = self._records.get(index)
            if rec is not None and rec.stage_key == stage_key:
                rec.stage_key = None
        return waiting

    def stage_bindings_replaced(self, bindings) -> None:
        """Reconcile every tile↔stage binding against the fan-in queues.

        The fan-in state remains the queue implementation; this event keeps
        the machine's per-record binding equal to it after every fan-in
        mutation (merge/activate/release/fail), so "blocked by stage X" is a
        record fact and never set correlation.  Idempotent by construction.
        """

        desired: dict[object, set[int]] = {
            key: {int(tile) for tile in tuple(tiles or ())}
            for key, tiles in dict(bindings or {}).items()
        }
        for key in tuple(self._stage_blocked):
            want = desired.get(key, set())
            waiting = self._stage_blocked.get(key, set())
            for index in tuple(waiting):
                if index not in want:
                    rec = self._records.get(index)
                    if rec is not None and rec.stage_key == key:
                        rec.stage_key = None
                    waiting.discard(index)
            if not waiting:
                self._stage_blocked.pop(key, None)
        for key, want in desired.items():
            for index in want:
                rec = self.record(index)
                if rec.stage_key != key:
                    self._stage_unbound(rec)
                    rec.stage_key = key
                    self._stage_blocked.setdefault(key, set()).add(rec.tile_number)

    @property
    def stage_blocked_tiles(self) -> frozenset[int]:
        return frozenset(
            index for waiting in self._stage_blocked.values() for index in waiting
        )

    # -- residency axis ----------------------------------------------------

    def level_claimed(
        self,
        tile_number: int,
        level_key,
        owner: ClaimOwner,
        *,
        request: object = None,
    ) -> None:
        rec = self.record(tile_number)
        self._level_order += 1
        rec.levels[level_key] = _LevelEntry(
            LevelPhase.CLAIMED,
            ClaimOwner(owner),
            request=request,
            order=int(self._level_order),
        )

    def level_materializing(self, tile_number: int, level_key) -> None:
        rec = self.record(tile_number)
        entry = rec.levels.get(level_key)
        if entry is not None and entry.phase is LevelPhase.CLAIMED:
            entry.phase = LevelPhase.MATERIALIZING

    def level_resident(self, tile_number: int, level_key) -> None:
        rec = self.record(tile_number)
        entry = rec.levels.get(level_key)
        if entry is None:
            rec.levels[level_key] = _LevelEntry(LevelPhase.RESIDENT, ClaimOwner.INGEST)
        else:
            entry.phase = LevelPhase.RESIDENT

    def level_declined(self, tile_number: int, level_key) -> tuple[ReleaseClaim, ...]:
        rec = self.record(tile_number)
        entry = rec.levels.get(level_key)
        if entry is None or entry.phase in (LevelPhase.RESIDENT, LevelPhase.RELEASED):
            return ()
        entry.phase = LevelPhase.RELEASED
        return (ReleaseClaim(rec.tile_number, level_key, entry.owner),)

    def materialization_planned(
        self,
        tile_number: int,
        request: object,
        *,
        owner: ClaimOwner = ClaimOwner.CHAIN,
    ) -> None:
        """Record all singleflight claims held by a materialization request."""

        for level_key in _request_level_keys(request):
            self.level_claimed(tile_number, level_key, owner, request=request)

    def materialization_started(self, request: object) -> None:
        for rec, level_key, _entry in self._entries_for_request(request):
            self.level_materializing(rec.tile_number, level_key)

    def materialization_resident(self, request: object) -> None:
        for rec, level_key, _entry in self._entries_for_request(request, include_released=True):
            self.level_resident(rec.tile_number, level_key)

    def materialization_released(self, request: object) -> tuple[ReleaseClaim, ...]:
        effects: list[ReleaseClaim] = []
        for rec, level_key, entry in self._entries_for_request(request):
            if entry.phase in (LevelPhase.CLAIMED, LevelPhase.MATERIALIZING):
                entry.phase = LevelPhase.RELEASED
                effects.append(ReleaseClaim(rec.tile_number, level_key, entry.owner))
        return tuple(effects)

    def pending_materializations(self) -> tuple[object, ...]:
        """Dispatchable LOD materialization requests derived from claimed records."""

        seen: set[int] = set()
        pending: list[tuple[int, object]] = []
        for rec in self._records.values():
            for entry in rec.levels.values():
                request = entry.request
                if entry.phase is not LevelPhase.CLAIMED or request is None:
                    continue
                identity = id(request)
                if identity in seen:
                    continue
                seen.add(identity)
                pending.append((int(entry.order), request))
        return tuple(request for _order, request in sorted(pending, key=lambda item: item[0]))

    def session_replaced(self) -> tuple[ReleaseClaim, ...]:
        """Release every non-resident claim; the session's records are done."""

        effects: list[ReleaseClaim] = []
        for rec in self._records.values():
            for level_key, entry in tuple(rec.levels.items()):
                if entry.phase in (LevelPhase.CLAIMED, LevelPhase.MATERIALIZING):
                    effects.append(
                        ReleaseClaim(rec.tile_number, level_key, entry.owner)
                    )
                    entry.phase = LevelPhase.RELEASED
        return tuple(effects)

    def dangling_claims(self) -> tuple[ReleaseClaim, ...]:
        """Queryable rule-2 audit: claims that never reached resident/released."""

        return tuple(
            ReleaseClaim(rec.tile_number, level_key, entry.owner)
            for rec in self._records.values()
            for level_key, entry in rec.levels.items()
            if entry.phase in (LevelPhase.CLAIMED, LevelPhase.MATERIALIZING)
        )

    # -- presentation axis ---------------------------------------------------

    def upsert_emitted(self, tile_number: int, source_id: object = None) -> None:
        rec = self.record(tile_number)
        self._unpark(rec)
        rec.presentation = Presentation.EMITTED
        rec.emitted_source_id = source_id

    def backend_presented_snapshot(self, identities) -> None:
        """Record the backend's latest drawn-slot identities.

        This does not by itself make a tile semantically presented.  It only
        stores physical truth in the lifecycle so convergence can compare that
        truth with the current desired payload without a parallel session map.
        """

        if identities is None:
            return
        normalized = {int(tile): identity for tile, identity in dict(identities).items()}
        for rec in self._records.values():
            if int(rec.tile_number) not in normalized:
                rec.backend_source_id = None
        for tile, identity in normalized.items():
            self.record(tile).backend_source_id = identity

    def commit_acknowledged(
        self,
        *,
        emitted_tiles: Iterable[int],
        accepted_tiles: Iterable[int],
        active_scope: Iterable[int],
        removed_tiles: Iterable[int] = (),
        stale: bool = False,
        presented_identities=None,
    ) -> frozenset[int]:
        """The single entry into ``PRESENTED`` (rule 1) and ``PARKED`` (rule 3).

        ``emitted_tiles`` are the delta's upserts; ``accepted_tiles`` the
        backend-accepted subset; a declined upsert parks unless the tile is in
        the active scope (in scope the commit loop legitimately retries).
        A stale report confirms nothing.

        **Identity-aware acknowledgement (P2, the machine invariant that
        subsumes the 2026-07-05 false-ack family):** when the backend supplies
        ``presented_identities`` (tile → source_id its slot actually holds),
        an emitted tile is confirmed only if the slot holds the identity the
        machine emitted.  Tile-number intersection alone opened three doors —
        the uniforms-only path, parked-but-drawn, and stale report reuse —
        each patched per-site before this invariant existed.

        Returns the identity-confirmed accepted set; callers must use it (not
        their own intersection) to clear dirty/pending bookkeeping, so the
        decision lives in exactly one place.  Park eligibility is the
        machine's own semantic axis (``EVALUATED`` = a re-presentable result
        exists); the P1 ``parkable_tiles`` crutch is gone.
        """

        if stale:
            return frozenset()
        accepted = {int(tile) for tile in accepted_tiles}
        active = {int(tile) for tile in active_scope}
        identities = (
            None
            if presented_identities is None
            else {int(tile): identity for tile, identity in dict(presented_identities).items()}
        )
        if identities is not None:
            self.backend_presented_snapshot(identities)
        confirmed: set[int] = set()
        for tile_number in emitted_tiles:
            index = int(tile_number)
            rec = self.record(index)
            accept = index in accepted
            presented_identity = rec.emitted_source_id
            if (
                accept
                and identities is not None
                and rec.emitted_source_id is not None
                and index in identities
                and identities[index] != rec.emitted_source_id
            ):
                # Rule 1: a slot that does not hold what we emitted did not
                # present it, whatever the report's tile numbers say.
                pair = (index, rec.emitted_source_id, identities[index])
                count = self._identity_rejection_counts.get(pair, 0) + 1
                self._identity_rejections += 1
                if count >= self.IDENTITY_RESIGN_AFTER:
                    # Resigned acceptance: record the backend's identity as
                    # the presented truth (never pretend our emit landed) and
                    # let the caller clear dirty — bounded retries, wedge
                    # stays visible in backend_stale_identities.  The pair is
                    # remembered so convergence passes skip exactly it (and
                    # nothing else).
                    self._identity_rejection_counts.pop(pair, None)
                    rec.resigned.add((rec.emitted_source_id, identities[index]))
                    presented_identity = identities[index]
                else:
                    self._identity_rejection_counts[pair] = count
                    accept = False
            if accept:
                self._unpark(rec)
                rec.presentation = Presentation.PRESENTED
                rec.presented_source_id = presented_identity
                rec.backend_source_id = presented_identity
                self._presented.add(index)
                confirmed.add(index)
                if rec.semantic is Semantic.EVALUATED:
                    # Sets-as-views: a confirmed EVALUATED tile owes the
                    # screen nothing — it is not "loading", whatever the
                    # backend report's presented set says (the 2026-07-05
                    # auto-levels wedge was this flag surviving here).
                    self._load_cleared(rec)
            elif index not in active:
                # Rule 3: never blind-retry an upsert a viewport-scoped
                # backend will keep declining. Park only if a semantic result
                # exists to re-present; otherwise there is nothing to arm.
                if rec.semantic is Semantic.EVALUATED:
                    rec.presentation = Presentation.PARKED
                    rec.parked_reason = "declined-out-of-scope"
                    self._parked.add(index)
                    # Parked = descoped, not loading; the scope re-arm
                    # re-dirties it through ordinary bookkeeping.
                    self._load_cleared(rec)
                elif rec.presentation is not Presentation.PRESENTED:
                    rec.presentation = Presentation.UNPRESENTED
        for tile_number in removed_tiles:
            index = int(tile_number)
            rec = self._records.get(index)
            if rec is None:
                continue
            self._unpark(rec)
            self._presented.discard(index)
            rec.presentation = Presentation.UNPRESENTED
            rec.presented_source_id = None
            rec.emitted_source_id = None
            rec.backend_source_id = None
        return frozenset(confirmed)

    def presentation_confirmed(self, tile_numbers: Iterable[int]) -> None:
        """Backend confirmed these tiles as presented (rule 1, second source).

        Resident-retarget commits re-show acknowledged payloads without new
        upserts, so ``commit_acknowledged`` sees no accepted upserts for
        them; the commit report's ``presented_tiles`` is still backend
        acknowledgement and is the only caller of this event.
        """

        for tile_number in tile_numbers:
            rec = self.record(int(tile_number))
            self._unpark(rec)
            rec.presentation = Presentation.PRESENTED
            if rec.emitted_source_id is not None:
                rec.presented_source_id = rec.emitted_source_id
            elif rec.backend_source_id is not None:
                rec.presented_source_id = rec.backend_source_id
            self._presented.add(rec.tile_number)
            if rec.semantic is Semantic.EVALUATED:
                self._load_cleared(rec)

    def presentation_discarded(self, tile_number: int) -> None:
        rec = self._records.get(int(tile_number))
        if rec is None:
            return
        self._presented.discard(rec.tile_number)
        if rec.presentation is Presentation.PRESENTED:
            rec.presentation = Presentation.UNPRESENTED
        rec.presented_source_id = None

    def rearm_for_scope(self, active_scope: Iterable[int]) -> tuple[int, ...]:
        """Rule 3 re-arm: parked tiles entering the active scope want an upsert.

        Returns the tiles to re-dirty; their records leave ``PARKED``.
        """

        rearmed: list[int] = []
        for index in tuple(self._parked):
            if index in active_scope:
                rec = self._records[index]
                self._unpark(rec)
                rec.presentation = Presentation.UNPRESENTED
                rearmed.append(index)
        return tuple(sorted(rearmed))

    # -- views ---------------------------------------------------------------

    @property
    def parked_tiles(self) -> frozenset[int]:
        return frozenset(self._parked)

    @property
    def evaluating_tiles(self) -> frozenset[int]:
        return frozenset(self._evaluating)

    @property
    def presented_tiles(self) -> frozenset[int]:
        return frozenset(self._presented)

    @property
    def loading_tiles(self) -> frozenset[int]:
        """Tiles that still owe exact content (view behind ``loading_tiles``)."""

        return frozenset(self._loading)

    @property
    def active_request_tiles(self) -> frozenset[int]:
        """Tiles with an evaluation request in flight (view behind
        ``active_tile_requests``)."""

        return frozenset(self._active_requests)

    @property
    def skipped_tiles(self) -> frozenset[int]:
        """Tiles the plan skipped (view behind ``skipped_tiles``)."""

        return frozenset(self._skipped)

    def counters(self) -> dict[str, int]:
        """Cheap diagnostics summary (rendered into diagnostics snapshots)."""

        return {
            "records": len(self._records),
            "evaluating": len(self._evaluating),
            "parked": len(self._parked),
            "presented": len(self._presented),
            "loading": len(self._loading),
            "active_requests": len(self._active_requests),
            "skipped": len(self._skipped),
            "stage_blocked": sum(len(w) for w in self._stage_blocked.values()),
            "dangling_claims": len(self.dangling_claims()),
            "identity_rejections": int(self._identity_rejections),
        }

    # -- internal ------------------------------------------------------------

    def _unpark(self, rec: TileRecord) -> None:
        self._parked.discard(rec.tile_number)
        if rec.presentation is Presentation.PARKED:
            rec.presentation = Presentation.UNPRESENTED
        rec.parked_reason = ""

    def _load_cleared(self, rec: TileRecord) -> None:
        rec.load_intent = False
        self._loading.discard(rec.tile_number)

    def _request_cleared(self, rec: TileRecord) -> None:
        rec.request_active = False
        self._active_requests.discard(rec.tile_number)

    def _stage_unbound(self, rec: TileRecord) -> None:
        key = rec.stage_key
        rec.stage_key = None
        if key is not None:
            waiting = self._stage_blocked.get(key)
            if waiting is not None:
                waiting.discard(rec.tile_number)
                if not waiting:
                    self._stage_blocked.pop(key, None)

    def _release_owned(
        self, rec: TileRecord, owner: ClaimOwner
    ) -> tuple[ReleaseClaim, ...]:
        effects: list[ReleaseClaim] = []
        for level_key, entry in tuple(rec.levels.items()):
            if entry.owner is owner and entry.phase in (
                LevelPhase.CLAIMED,
                LevelPhase.MATERIALIZING,
            ):
                effects.append(ReleaseClaim(rec.tile_number, level_key, entry.owner))
                entry.phase = LevelPhase.RELEASED
        return tuple(effects)

    def _entries_for_request(
        self,
        request: object,
        *,
        include_released: bool = False,
    ) -> tuple[tuple[TileRecord, object, _LevelEntry], ...]:
        entries: list[tuple[TileRecord, object, _LevelEntry]] = []
        for rec in self._records.values():
            for level_key, entry in rec.levels.items():
                if entry.request is not request:
                    continue
                if (
                    not include_released
                    and entry.phase in (LevelPhase.RESIDENT, LevelPhase.RELEASED)
                ):
                    continue
                entries.append((rec, level_key, entry))
        return tuple(sorted(entries, key=lambda item: int(item[2].order)))


def _request_level_keys(request: object) -> tuple[object, ...]:
    chain = tuple(getattr(request, "chain", ()) or ())
    keys: list[object] = []
    if chain:
        for level_key, _rel in chain:
            if level_key is not None:
                keys.append(level_key)
    else:
        key = getattr(request, "key", None)
        if key is not None:
            keys.append(key)
    return tuple(keys)
