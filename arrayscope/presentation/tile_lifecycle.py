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
        parkable_tiles: Iterable[int] | None = None,
    ) -> None:
        """The single entry into ``PRESENTED`` (rule 1) and ``PARKED`` (rule 3).

        ``emitted_tiles`` are the delta's upserts; ``accepted_tiles`` the
        backend-accepted subset; a declined upsert parks unless the tile is in
        the active scope (in scope the commit loop legitimately retries).
        A stale report confirms nothing.

        ``parkable_tiles`` is a migration parameter (ADR 0051 P1→P2): while
        legacy collections still bypass ``evaluation_completed`` (sessions
        seeded by direct ``rendered_tiles`` writes), the caller supplies which
        tiles hold a re-presentable result.  Once the semantic axis is
        authoritative (P2), omit it and the machine's own ``EVALUATED`` state
        decides.
        """

        if stale:
            return
        accepted = {int(tile) for tile in accepted_tiles}
        active = {int(tile) for tile in active_scope}
        parkable = (
            None if parkable_tiles is None else {int(tile) for tile in parkable_tiles}
        )
        for tile_number in emitted_tiles:
            index = int(tile_number)
            rec = self.record(index)
            if index in accepted:
                self._unpark(rec)
                rec.presentation = Presentation.PRESENTED
                rec.presented_source_id = rec.emitted_source_id
                self._presented.add(index)
            elif index not in active:
                # Rule 3: never blind-retry an upsert a viewport-scoped
                # backend will keep declining. Park only if a semantic result
                # exists to re-present; otherwise there is nothing to arm.
                can_park = (
                    rec.semantic is Semantic.EVALUATED
                    if parkable is None
                    else index in parkable
                )
                if can_park:
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
