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


@dataclass
class TileRecord:
    tile_number: int
    semantic: Semantic = Semantic.UNPLANNED
    presentation: Presentation = Presentation.UNPRESENTED
    #: payload identity last emitted toward the backend (emit-once key).
    emitted_source_id: object = None
    #: payload identity the backend acknowledged as shown.
    presented_source_id: object = None
    parked_reason: str = ""
    levels: dict = field(default_factory=dict)  # level_key -> _LevelEntry


class TileLifecycle:
    """Single owner of per-tile lifecycle state for one render session."""

    def __init__(self) -> None:
        self._records: dict[int, TileRecord] = {}
        self._parked: set[int] = set()
        self._evaluating: set[int] = set()
        self._presented: set[int] = set()
        self._identity_rejections = 0
        #: (tile, emitted_id, backend_id) -> consecutive rejection count; a
        #: pair rejected IDENTITY_RESIGN_AFTER times is resigned (see below).
        self._identity_rejection_counts: dict[tuple[int, object, object], int] = {}

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

    def evaluation_completed(self, tile_number: int) -> None:
        """A fresh semantic result exists; any prior presentation identity is stale."""

        rec = self.record(tile_number)
        rec.semantic = Semantic.EVALUATED
        self._evaluating.discard(rec.tile_number)
        # Fresh identity: the previous emit/park no longer refers to what the
        # next commit will carry, and the backend has not seen the new one.
        self._unpark(rec)
        if rec.presentation is not Presentation.PRESENTED:
            rec.presentation = Presentation.UNPRESENTED
        rec.emitted_source_id = None

    def evaluation_declined(self, tile_number: int) -> tuple[ReleaseClaim, ...]:
        """Admission declined or work dropped: back to planned, claims released."""

        rec = self.record(tile_number)
        if rec.semantic is Semantic.EVALUATING:
            rec.semantic = Semantic.PLANNED
        self._evaluating.discard(rec.tile_number)
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

    # -- residency axis ----------------------------------------------------

    def level_claimed(self, tile_number: int, level_key, owner: ClaimOwner) -> None:
        rec = self.record(tile_number)
        rec.levels[level_key] = _LevelEntry(LevelPhase.CLAIMED, ClaimOwner(owner))

    def level_materializing(self, tile_number: int, level_key) -> None:
        rec = self.record(tile_number)
        entry = rec.levels.get(level_key)
        if entry is not None:
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
        entry = rec.levels.pop(level_key, None)
        if entry is None or entry.phase is LevelPhase.RESIDENT:
            if entry is not None:
                rec.levels[level_key] = entry
            return ()
        return (ReleaseClaim(rec.tile_number, level_key, entry.owner),)

    def session_replaced(self) -> tuple[ReleaseClaim, ...]:
        """Release every non-resident claim; the session's records are done."""

        effects: list[ReleaseClaim] = []
        for rec in self._records.values():
            for level_key, entry in tuple(rec.levels.items()):
                if entry.phase is not LevelPhase.RESIDENT:
                    effects.append(
                        ReleaseClaim(rec.tile_number, level_key, entry.owner)
                    )
                    rec.levels.pop(level_key, None)
        return tuple(effects)

    def dangling_claims(self) -> tuple[ReleaseClaim, ...]:
        """Queryable rule-2 audit: claims that never reached resident/released."""

        return tuple(
            ReleaseClaim(rec.tile_number, level_key, entry.owner)
            for rec in self._records.values()
            for level_key, entry in rec.levels.items()
            if entry.phase is not LevelPhase.RESIDENT
        )

    # -- presentation axis ---------------------------------------------------

    def upsert_emitted(self, tile_number: int, source_id: object = None) -> None:
        rec = self.record(tile_number)
        self._unpark(rec)
        rec.presentation = Presentation.EMITTED
        rec.emitted_source_id = source_id

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
                    # stays visible in backend_stale_identities.
                    self._identity_rejection_counts.pop(pair, None)
                    presented_identity = identities[index]
                else:
                    self._identity_rejection_counts[pair] = count
                    accept = False
            if accept:
                self._unpark(rec)
                rec.presentation = Presentation.PRESENTED
                rec.presented_source_id = presented_identity
                self._presented.add(index)
                confirmed.add(index)
            elif index not in active:
                # Rule 3: never blind-retry an upsert a viewport-scoped
                # backend will keep declining. Park only if a semantic result
                # exists to re-present; otherwise there is nothing to arm.
                if rec.semantic is Semantic.EVALUATED:
                    rec.presentation = Presentation.PARKED
                    rec.parked_reason = "declined-out-of-scope"
                    self._parked.add(index)
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
            self._presented.add(rec.tile_number)

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

    def counters(self) -> dict[str, int]:
        """Cheap diagnostics summary (rendered into diagnostics snapshots)."""

        return {
            "records": len(self._records),
            "evaluating": len(self._evaluating),
            "parked": len(self._parked),
            "presented": len(self._presented),
            "dangling_claims": len(self.dangling_claims()),
            "identity_rejections": int(self._identity_rejections),
        }

    # -- internal ------------------------------------------------------------

    def _unpark(self, rec: TileRecord) -> None:
        self._parked.discard(rec.tile_number)
        if rec.presentation is Presentation.PARKED:
            rec.presentation = Presentation.UNPRESENTED
        rec.parked_reason = ""

    def _release_owned(
        self, rec: TileRecord, owner: ClaimOwner
    ) -> tuple[ReleaseClaim, ...]:
        effects: list[ReleaseClaim] = []
        for level_key, entry in tuple(rec.levels.items()):
            if entry.owner is owner and entry.phase is not LevelPhase.RESIDENT:
                effects.append(ReleaseClaim(rec.tile_number, level_key, entry.owner))
                rec.levels.pop(level_key, None)
        return tuple(effects)
