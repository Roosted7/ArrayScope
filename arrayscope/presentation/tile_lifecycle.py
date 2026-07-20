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

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum

from arrayscope.core.trace import emit_trace, trace_enabled
from arrayscope.display.model.tile_identity import TileIdentity


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


class TilePhase(str, Enum):
    """Observable phase of one region in the current frame target."""

    OUT_OF_SCOPE = "out_of_scope"
    NEEDS_FIRST_PIXEL = "needs_first_pixel"
    FALLBACK_SHOWN = "fallback_shown"
    TARGET_SCHEDULABLE = "target_schedulable"
    TARGET_WAITING_STAGE = "target_waiting_stage"
    TARGET_RUNNING = "target_running"
    TARGET_READY = "target_ready"
    TARGET_EMITTED = "target_emitted"
    TARGET_PRESENTED = "target_presented"
    SKIPPED = "skipped"
    FAILED = "failed"


def _quality_lod_satisfies_target(
    quality: str,
    payload_level: int,
    target_level: int,
) -> bool:
    """Whether presented pixels already meet one display LOD demand.

    Exact pixels satisfy their level or any coarser demand.  Pixels retained
    as fallback against an earlier demand satisfy a *strictly* coarser later
    demand when they are already finer; an equal-level fallback still owes
    exact target work.  This prevents quality demotion without letting a
    genuinely coarse preview claim exact settlement.
    """

    payload_level = max(0, int(payload_level))
    target_level = max(0, int(target_level))
    quality = "fallback" if str(quality or "exact") == "preview" else str(quality or "exact")
    return bool(
        payload_level <= target_level and (quality == "exact" or payload_level < target_level)
    )


@dataclass(frozen=True)
class TilePayloadRef:
    """Backend-independent identity metadata for a presentable payload."""

    source_id: object
    quality: str
    lod_level: int
    source_index: int
    texture_kind: object = None
    shader_mapping_key: object = None
    identity: TileIdentity | None = None
    payload: object = None

    def __post_init__(self) -> None:
        quality = str(self.quality or "exact")
        if quality == "preview":
            quality = "fallback"
        if quality not in {"fallback", "exact"}:
            raise ValueError(f"tile payload quality must be fallback/exact, got {quality!r}")
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "lod_level", max(0, int(self.lod_level)))
        object.__setattr__(self, "source_index", int(self.source_index))

    @property
    def is_target_quality(self) -> bool:
        return self.quality == "exact"

    @property
    def acknowledged_identity(self) -> object:
        return self.identity if self.identity is not None else self.source_id

    def satisfies_target(self, target: TileTarget) -> bool:
        if self.identity is not None or target.identity is not None:
            return bool(
                self.identity is not None
                and target.identity is not None
                and self.identity.semantic_key == target.identity.semantic_key
                and _quality_lod_satisfies_target(
                    self.quality,
                    self.lod_level,
                    target.lod_level,
                )
            )
        return bool(
            int(self.source_index) == int(target.source_index)
            and _quality_lod_satisfies_target(
                self.quality,
                self.lod_level,
                target.lod_level,
            )
        )


@dataclass(frozen=True)
class TileTarget:
    tile_number: int
    source_index: int
    semantic_source_id: object
    lod_level: int = 0
    quality: str = "exact"
    visible: bool = True
    identity: TileIdentity | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile_number", int(self.tile_number))
        object.__setattr__(self, "source_index", int(self.source_index))
        object.__setattr__(self, "lod_level", max(0, int(self.lod_level)))
        if str(self.quality or "exact") != "exact":
            raise ValueError("tile targets are exact; preview is only a fallback")
        object.__setattr__(self, "quality", "exact")
        object.__setattr__(self, "visible", bool(self.visible))


@dataclass(frozen=True)
class TileTaskClaim:
    task_key: object
    stage_key: object = None

    def __post_init__(self) -> None:
        if self.task_key is None and self.stage_key is None:
            raise ValueError("a tile claim needs an admitted task key or a stage key")


@dataclass(frozen=True)
class TilePresentationCommand:
    tile_number: int
    payload: object
    payload_ref: TilePayloadRef


@dataclass(frozen=True)
class TileLifecycleSnapshot:
    counts: Mapping[str, int]
    visible_tiles: int
    first_pixels_presented: bool
    visible_target_settled: bool
    orphan_running: int
    parked_without_producer: int


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
    #: Last backend-acknowledged payload quality/LOD level.  These are
    #: presentation facts, not LOD-planning inputs; they let presentation
    #: choose a safe fallback without re-deriving ownership from renderer maps.
    presented_quality: str = ""
    presented_level: int | None = None
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
    #: explicitly descopes it.  Session loading views read this flag.
    load_intent: bool = False
    #: P2 sets-as-views: an evaluation request for this tile is in flight
    #: Session active-request views read this flag.
    request_active: bool = False
    #: P2 stage fan-in: the reusable-stage key this tile waits on, or None.
    #: Recorded by ``stage_attached`` so "loading without an evaluation
    #: request" is a queryable machine fact instead of set correlation.
    stage_key: object = None
    #: Pipeline-owned reduced preview/floor claims keyed by rung integer.
    #: The value carries both LOD and semantic-window identity: montage slots
    #: are reused across index scrolls, so ``(slot, rung, level)`` alone can
    #: refer to a completely different source population.
    preview_claims: dict[int, tuple[int, object]] = field(default_factory=dict)
    #: Opaque metadata for the current kernel evaluation claim. The lifecycle
    #: owns the presence of the claim; effects own the metadata shape.
    evaluation_claim: object = None
    #: Payload objects known to be safe first-pixel fallbacks for this slot,
    #: keyed by their source identity.  The lifecycle owns the ordering
    #: decision; the display/session layer still owns the payload contents.
    presentable_payloads: dict[object, object] = field(default_factory=dict)
    #: Canonical current target and task/dependency obligation.  These fields
    #: absorb the former parallel TileLedger; all per-tile truth now lives on
    #: this record.
    target: TileTarget | None = None
    task_claim: TileTaskClaim | None = None
    stage_producer_key: object = None
    failed_reason: str = ""
    #: Trace bookkeeping: the current target requirement's closure is already
    #: on the bus (a fresh satisfying ``backend_ack`` or one
    #: ``target_satisfied_retained`` edge).  Reset when a new requirement is
    #: adopted, so every requirement gets exactly one closure statement.
    satisfaction_traced: bool = False

    @property
    def active(self) -> bool:
        return bool(
            self.target is not None
            and self.target.visible
            and self.semantic is not Semantic.SKIPPED
            and not self.failed_reason
        )

    @property
    def acknowledged_payload(self) -> TilePayloadRef | None:
        payload = self._payload_for_identity(self.presented_source_id)
        if payload is None:
            return None
        quality = str(self.presented_quality or payload.quality)
        if quality == "preview":
            quality = "fallback"
        level = payload.lod_level if self.presented_level is None else int(self.presented_level)
        return replace(payload, quality=quality, lod_level=level)

    @property
    def backend_payload(self) -> TilePayloadRef | None:
        return self._payload_for_identity(self.backend_source_id)

    @property
    def target_payload(self) -> TilePayloadRef | None:
        if self.target is None:
            return None
        candidates = (
            ref
            for ref in (
                payload_ref_from_display_payload(value)
                for value in self.presentable_payloads.values()
            )
            if ref.satisfies_target(self.target)
        )
        return min(candidates, key=lambda ref: (ref.lod_level, repr(ref.source_id)), default=None)

    @property
    def fallback_payload(self) -> TilePayloadRef | None:
        if self.target is None:
            return None
        candidates = (
            ref
            for ref in (
                payload_ref_from_display_payload(value)
                for value in self.presentable_payloads.values()
            )
            if ref.source_index == self.target.source_index
            and not ref.satisfies_target(self.target)
        )
        return min(
            candidates,
            key=lambda ref: (0 if ref.quality == "exact" else 1, ref.lod_level),
            default=None,
        )

    @property
    def target_payload_satisfies(self) -> bool:
        return self.target_payload is not None

    @property
    def target_settled(self) -> bool:
        fields = self._presented_payload_fields()
        return bool(
            self.target is not None
            and fields is not None
            and fields[2] == int(self.target.source_index)
            and _quality_lod_satisfies_target(
                fields[0],
                fields[1],
                self.target.lod_level,
            )
            and self.backend_source_id == self.presented_source_id
        )

    @property
    def first_pixel_presented(self) -> bool:
        fields = self._presented_payload_fields()
        return bool(
            self.active
            and self.target is not None
            and fields is not None
            and fields[2] == int(self.target.source_index)
            and self.backend_source_id == self.presented_source_id
        )

    def _presented_payload_fields(self) -> tuple[str, int, int] | None:
        """Read acknowledged payload facts without normalizing a new ref."""

        if self.presented_source_id is None:
            return None
        payload = self.presentable_payloads.get(self.presented_source_id)
        if payload is None:
            if self.target is None:
                return None
            return (
                str(self.presented_quality or "exact"),
                0 if self.presented_level is None else int(self.presented_level),
                int(self.target.source_index),
            )
        quality = str(self.presented_quality or getattr(payload, "quality", "exact") or "exact")
        if quality == "preview":
            quality = "fallback"
        lod = getattr(payload, "lod", None)
        level = (
            int(getattr(lod, "level", 0) or 0)
            if self.presented_level is None
            else int(self.presented_level)
        )
        return quality, level, int(getattr(payload, "source_index", -1))

    @property
    def phase(self) -> TilePhase:
        if self.target is None or not self.target.visible:
            return TilePhase.OUT_OF_SCOPE
        if self.semantic is Semantic.SKIPPED:
            return TilePhase.SKIPPED
        if self.failed_reason:
            return TilePhase.FAILED
        if self.target_settled:
            return TilePhase.TARGET_PRESENTED
        if self.presentation is Presentation.EMITTED:
            return TilePhase.TARGET_EMITTED
        if self.target_payload_satisfies:
            return TilePhase.TARGET_READY
        if self.task_claim is not None and self.task_claim.task_key is not None:
            return TilePhase.TARGET_RUNNING
        if self.stage_key is not None:
            return TilePhase.TARGET_WAITING_STAGE
        if self.first_pixel_presented:
            return TilePhase.FALLBACK_SHOWN
        if self.acknowledged_payload is None:
            return TilePhase.NEEDS_FIRST_PIXEL
        return TilePhase.TARGET_SCHEDULABLE

    def _payload_for_identity(self, source_id) -> TilePayloadRef | None:
        if source_id is None:
            return None
        payload = self.presentable_payloads.get(source_id)
        if payload is not None:
            return payload_ref_from_display_payload(payload)
        for candidate in self.presentable_payloads.values():
            ref = payload_ref_from_display_payload(candidate)
            if ref.acknowledged_identity == source_id:
                return ref
        if self.target is None:
            return None
        return TilePayloadRef(
            source_id=source_id,
            quality=self.presented_quality or "exact",
            lod_level=0 if self.presented_level is None else self.presented_level,
            source_index=self.target.source_index,
        )


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
        self._level_revision = 0

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
    def level_revision(self) -> int:
        """Monotonic epoch for presentation-floor residency truth."""

        return int(self._level_revision)

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

    # ``row`` is the vocabulary used by diagnostics and model tests.  It is
    # deliberately the same object as ``record``: there is no parallel tile
    # ledger to reconcile.
    row = record

    def retarget(self, targets: Mapping[int, TileTarget]) -> None:
        """Atomically adopt the current visible target set."""

        wanted = {int(tile): target for tile, target in dict(targets).items()}
        for tile_number, target in wanted.items():
            rec = self.record(tile_number)
            if rec.target == target:
                continue
            previous_source = None if rec.target is None else rec.target.source_index
            rec.target = target
            rec.failed_reason = ""
            rec.task_claim = None
            rec.stage_producer_key = None
            _trace_lifecycle(rec, "target_required")
            if previous_source is not None and previous_source != target.source_index:
                # A slot may be retargeted hundreds of times during a fast
                # montage scrub.  Presentable payloads are first-pixel
                # candidates for the *current* source, not a per-slot history.
                # Keep current-source values that may already have been
                # remapped into this record and the one payload whose identity
                # the backend still reports physically resident.  Retaining
                # every superseded value made target_payload/backlog queries
                # grow without bound (and turned a scrub into millions of
                # payload normalizations).
                rec.presentable_payloads = {
                    source_id: payload
                    for source_id, payload in rec.presentable_payloads.items()
                    if source_id == rec.backend_source_id
                    or int(getattr(payload, "source_index", -1)) == int(target.source_index)
                }
                # Physical backend truth survives retarget for diagnostics,
                # but it is no longer a semantically presented current tile.
                rec.presented_source_id = None
                rec.presented_quality = ""
                rec.presented_level = None
                rec.presentation = Presentation.UNPRESENTED
                self._presented.discard(tile_number)
            # Idempotent backends never re-upload or re-acknowledge a payload
            # that already satisfies the new requirement, so this is the last
            # moment production can say the requirement was closed by
            # retention rather than left unsatisfied.
            rec.satisfaction_traced = False
            self._note_retained_satisfaction(rec)
        for tile_number, rec in tuple(self._records.items()):
            if tile_number in wanted or rec.target is None:
                continue
            rec.target = TileTarget(
                tile_number=rec.target.tile_number,
                source_index=rec.target.source_index,
                semantic_source_id=rec.target.semantic_source_id,
                lod_level=rec.target.lod_level,
                visible=False,
            )
            rec.task_claim = None
            rec.stage_producer_key = None
            _trace_lifecycle(rec, "target_released")

    def fallback_ready(self, tile_number: int, payload: TilePayloadRef | object) -> None:
        ref = _coerce_payload_ref(payload)
        rec = self.record(tile_number)
        if rec.target is None or ref.source_index != rec.target.source_index:
            return
        rec.presentable_payloads[ref.source_id] = _stored_payload(ref)
        _trace_lifecycle(rec, "fallback_ready", payload=ref)

    def target_ready(self, tile_number: int, payload: TilePayloadRef | object) -> None:
        ref = _coerce_payload_ref(payload)
        rec = self.record(tile_number)
        if rec.target is None or not ref.satisfies_target(rec.target):
            return
        rec.presentable_payloads[ref.source_id] = _stored_payload(ref)
        rec.task_claim = None
        rec.stage_producer_key = None
        self._stage_unbound(rec)
        _trace_lifecycle(rec, "target_ready", payload=ref)

    def target_invalidated(self, tile_number: int) -> None:
        rec = self.record(tile_number)
        rec.task_claim = None
        rec.stage_producer_key = None
        rec.evaluation_claim = None
        if rec.presentation is Presentation.PRESENTED:
            rec.presented_quality = "preview"
        if rec.presentation is not Presentation.PRESENTED:
            rec.presentation = Presentation.UNPRESENTED

    def task_requested(self, tile_number: int, *, stage_key=None, stage_producer_key=None) -> None:
        rec = self.record(tile_number)
        rec.stage_key = stage_key
        rec.stage_producer_key = stage_producer_key
        _trace_lifecycle(rec, "task_requested", stage_key=stage_key)

    def task_admitted(
        self, tile_number: int, task_key, *, stage_key=None, stage_producer_key=None
    ) -> None:
        if task_key is None and stage_key is None:
            return
        rec = self.record(tile_number)
        rec.task_claim = TileTaskClaim(task_key, stage_key)
        rec.stage_key = stage_key
        rec.stage_producer_key = stage_producer_key
        _trace_lifecycle(rec, "task_admitted", task_key=task_key, stage_key=stage_key)

    def task_released(self, tile_number: int, *, reason: str = "") -> None:
        rec = self.record(tile_number)
        rec.task_claim = None
        rec.stage_producer_key = None
        rec.failed_reason = (
            str(reason or "")
            if reason and reason not in {"stale", "dropped", "cancelled", "completed"}
            else ""
        )
        _trace_lifecycle(rec, "task_released", reason=str(reason or ""))

    def stage_waiting(self, tile_number: int, stage_key, producer_key) -> None:
        if stage_key is None or producer_key is None:
            return
        self.stage_attached(tile_number, stage_key)
        self.record(tile_number).stage_producer_key = producer_key

    def stage_ready(self, stage_key) -> tuple[int, ...]:
        waiting = self.stage_resolved(stage_key)
        for tile_number in waiting:
            self.record(tile_number).stage_producer_key = None
        return waiting

    def commit_emitted(self, upserts: Mapping[int, object]) -> None:
        for tile_number, payload in dict(upserts).items():
            ref = _coerce_payload_ref(payload)
            rec = self.record(tile_number)
            rec.presentable_payloads[ref.source_id] = _stored_payload(ref)
            rec.presented_quality = "preview" if ref.quality == "fallback" else ref.quality
            rec.presented_level = ref.lod_level
            self.upsert_emitted(tile_number, ref.acknowledged_identity)
            _trace_lifecycle(rec, "commit_emitted", payload=ref)

    def backend_ack(
        self,
        accepted_payloads: Mapping[int, object],
        *,
        backend_payloads: Mapping[int, object] | None = None,
    ) -> tuple[int, ...]:
        accepted_refs = {
            int(tile): _coerce_payload_ref(payload)
            for tile, payload in dict(accepted_payloads).items()
        }
        backend_refs = {
            int(tile): _coerce_payload_ref(payload)
            for tile, payload in dict(backend_payloads or accepted_payloads).items()
        }
        metadata_accepted = {
            tile
            for tile, ref in accepted_refs.items()
            if _payload_refs_match(
                self.record(tile)._payload_for_identity(self.record(tile).emitted_source_id), ref
            )
        }
        merged_backend = dict(self.backend_presented_identities)
        merged_backend.update(
            {tile: ref.acknowledged_identity for tile, ref in backend_refs.items()}
        )
        confirmed = self.commit_acknowledged(
            emitted_tiles=tuple(accepted_refs),
            accepted_tiles=tuple(metadata_accepted),
            active_scope=tuple(tile for tile, rec in self._records.items() if rec.active),
            presented_identities=merged_backend,
        )
        for tile, ref in backend_refs.items():
            emit_trace(
                "backend_ack",
                tile=int(tile),
                source_index=int(ref.source_index),
                identity=ref.acknowledged_identity,
                quality=str(ref.quality),
                level=int(ref.lod_level),
                accepted=bool(tile in confirmed),
            )
            _trace_lifecycle(
                self.record(tile),
                "backend_ack",
                payload=ref,
                accepted=bool(tile in confirmed),
            )
        return tuple(sorted(confirmed))

    def presentation_changes(
        self, *, max_items: int | None = None
    ) -> tuple[TilePresentationCommand, ...]:
        commands: list[TilePresentationCommand] = []
        for tile_number, rec in sorted(self._records.items()):
            if not rec.active or rec.presentation is Presentation.EMITTED:
                continue
            payload = rec.target_payload or rec.fallback_payload
            if payload is None:
                continue
            if _payload_refs_match(rec.acknowledged_payload, payload):
                continue
            commands.append(TilePresentationCommand(tile_number, payload.payload, payload))
            if max_items is not None and len(commands) >= max(0, int(max_items)):
                break
        return tuple(commands)

    def visible_first_pixels_presented(self) -> bool:
        rows = tuple(rec for rec in self._records.values() if rec.active)
        return bool(not rows or all(rec.first_pixel_presented for rec in rows))

    def first_pixels_presented(self, tile_numbers) -> bool:
        """Return whether every tile in one required scope has a first pixel.

        Active lifecycle rows may include retained or near-viewport residency.
        Callers that own a narrower semantic obligation must name that set;
        completion is coverage of those unique identities, never an event
        count or an assertion about the wider active cache population.
        """

        required = tuple(dict.fromkeys(int(tile) for tile in tuple(tile_numbers or ())))
        if not required:
            return True
        return all(
            (record := self._records.get(tile_number)) is not None and record.first_pixel_presented
            for tile_number in required
        )

    def visible_target_settled(self) -> bool:
        rows = tuple(rec for rec in self._records.values() if rec.active)
        return bool(not rows or all(rec.target_settled for rec in rows))

    def target_unsettled_tiles(self, tile_numbers) -> tuple[int, ...]:
        """Return scoped target obligations without redefining visibility."""

        unsettled = []
        for tile_number in tuple(tile_numbers or ()):
            rec = self._records.get(int(tile_number))
            if rec is None or not rec.target_settled:
                unsettled.append(int(tile_number))
        return tuple(dict.fromkeys(unsettled))

    def target_settled(self, tile_numbers) -> bool:
        return not self.target_unsettled_tiles(tile_numbers)

    def snapshot(self) -> TileLifecycleSnapshot:
        rows = tuple(rec for rec in self._records.values() if rec.active)
        counts: dict[str, int] = {}
        for rec in rows:
            counts[rec.phase.value] = counts.get(rec.phase.value, 0) + 1
        return TileLifecycleSnapshot(
            counts=dict(sorted(counts.items())),
            visible_tiles=len(rows),
            first_pixels_presented=self.visible_first_pixels_presented(),
            visible_target_settled=self.visible_target_settled(),
            orphan_running=sum(1 for rec in rows if rec.request_active and rec.task_claim is None),
            parked_without_producer=sum(
                1 for rec in rows if rec.stage_key is not None and rec.stage_producer_key is None
            ),
        )

    def peek(self, tile_number: int) -> TileRecord | None:
        return self._records.get(int(tile_number))

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[TileRecord]:
        return iter(self._records.values())

    def feedback_signature(
        self, tile_numbers: Iterable[int] = ()
    ) -> tuple[tuple[object, ...], ...]:
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
        """A fresh semantic result exists; any prior presentation stays visible.

        Replacement is a payload/pending-upsert fact.  Presentation changes
        only when the backend acknowledges a new identity or physically evicts
        a slot; otherwise a ready exact result can briefly turn an already
        visible tile black before the replacement commit lands.
        """

        rec = self.record(tile_number)
        rec.semantic = Semantic.EVALUATED
        self._evaluating.discard(rec.tile_number)
        self._skipped.discard(rec.tile_number)
        self._request_cleared(rec)
        self._stage_unbound(rec)
        # Fresh identity: the previous emit no longer refers to what the next
        # commit will carry, but the old presented slot remains valid first
        # pixels until a backend acknowledgement replaces or removes it.
        self._unpark(rec)
        if rec.presentation is not Presentation.PRESENTED:
            rec.presentation = Presentation.UNPRESENTED
            rec.presented_source_id = None
            rec.presented_quality = ""
            rec.presented_level = None
        rec.emitted_source_id = None
        rec.resigned.clear()

    def remember_presentable(self, tile_number: int, payload: object) -> None:
        """Remember a payload that may be reused as an atomic fallback."""

        source_id = getattr(payload, "source_id", None)
        if source_id is None:
            return
        self.record(tile_number).presentable_payloads[source_id] = payload

    def payload_is_current(self, tile_number: int, payload: object) -> bool:
        """Whether ``payload`` is known-safe for this slot's current target.

        Materialization mechanics are deliberately irrelevant here. A reduced
        shared-target payload may never appear in a renderer-local native
        ``rendered_tiles`` map, yet it is still current target-quality content.
        """

        rec = self._records.get(int(tile_number))
        if rec is None or rec.target is None or payload is None:
            return False
        ref = payload_ref_from_display_payload(payload)
        return bool(
            int(ref.source_index) == int(rec.target.source_index)
            and ref.source_id in rec.presentable_payloads
        )

    def current_presentable_payload(self, tile_number: int):
        """Return the best target/fallback payload for the current slot."""

        rec = self._records.get(int(tile_number))
        if rec is None:
            return None
        ref = rec.target_payload or rec.fallback_payload
        return None if ref is None else ref.payload

    def best_presentable(
        self, tile_number: int, semantic_source=None, target_level: int | None = None
    ):
        """Return the best safe fallback payload for this tile, or ``None``.

        Ordering is exact before preview, then finer/equal LOD before coarser
        for the same semantic source.  The method is intentionally payload-
        shape agnostic so lifecycle owns the policy without importing display
        model types.
        """

        rec = self._records.get(int(tile_number))
        if rec is None or not rec.presentable_payloads:
            return None
        target_level = 0 if target_level is None else int(target_level)
        candidates = []
        for payload in rec.presentable_payloads.values():
            source_id = getattr(payload, "source_id", None)
            if (
                semantic_source is not None
                and _payload_base_source_id(source_id) != semantic_source
            ):
                continue
            quality = str(getattr(payload, "quality", "exact") or "exact")
            lod = getattr(payload, "lod", None)
            level = int(getattr(lod, "level", 0) or 0)
            # exact/finer current source, target/equal, preview current source,
            # coarser fallback.  Negative level distance means finer than
            # requested and is preferred over coarser.
            exact_rank = 0 if quality == "exact" else 1
            level_rank = 0 if level <= target_level else 1
            candidates.append((exact_rank, level_rank, abs(level - target_level), level, payload))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[:4])
        return candidates[0][4]

    def acknowledge_presented(
        self,
        tile_number: int,
        payload_identity: object,
        quality: str = "exact",
        level: int | None = None,
    ) -> bool:
        """Record backend-acknowledged presentation metadata for one tile."""

        rec = self.record(tile_number)
        normalized_quality = str(quality or "exact")
        normalized_level = None if level is None else int(level)
        if (
            rec.presentation is Presentation.PRESENTED
            and rec.presented_source_id == payload_identity
            and rec.backend_source_id == payload_identity
            and rec.presented_quality == normalized_quality
            and rec.presented_level == normalized_level
        ):
            # A commit report may repeat the backend's complete active set even
            # when it accepted no upserts.  That is confirmation of existing
            # physical truth, not a new acknowledgement edge.  Keeping this
            # transition idempotent prevents diagnostics from turning no-op
            # presentation polls into thousands of fictitious backend acks.
            return False
        self._unpark(rec)
        rec.presentation = Presentation.PRESENTED
        rec.presented_source_id = payload_identity
        rec.backend_source_id = payload_identity
        rec.presented_quality = normalized_quality
        rec.presented_level = normalized_level
        self._presented.add(rec.tile_number)
        if rec.semantic is Semantic.EVALUATED:
            self._load_cleared(rec)
        emit_trace(
            "backend_ack",
            tile=int(rec.tile_number),
            source_index=None if rec.target is None else int(rec.target.source_index),
            identity=payload_identity,
            quality=str(rec.presented_quality),
            level=rec.presented_level,
            accepted=True,
        )
        _trace_lifecycle(rec, "presented", identity=payload_identity)
        if rec.target_settled:
            if (
                rec.target is not None
                and normalized_quality == "exact"
                and normalized_level is not None
                and normalized_level <= int(rec.target.lod_level)
            ):
                # The ``backend_ack`` just emitted is itself replay-visible
                # exact settlement; no retained-satisfaction edge is owed.
                rec.satisfaction_traced = True
            else:
                # Settled by retained already-finer fallback pixels: the ack
                # on the bus carries fallback quality, which replay must not
                # count as exact settlement.
                self._note_retained_satisfaction(rec)
        return True

    def may_remove_visible(self, tile_number: int, *, memory_pressure: bool = False) -> bool:
        """Whether a visible tile may be physically removed right now."""

        if memory_pressure:
            return True
        rec = self._records.get(int(tile_number))
        if rec is None:
            return True
        if not rec.active:
            return True
        backend = rec.backend_payload
        # A slot mapped to a different semantic source must clear rather than
        # lie about values. For the same source (quality/LOD/layout changes),
        # an unpresented successor is the strongest reason to retain pixels.
        return bool(
            backend is not None
            and rec.target is not None
            and int(backend.source_index) != int(rec.target.source_index)
        )

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
        """This tile owes exact content to the screen."""

        rec = self.record(tile_number)
        rec.load_intent = True
        self._loading.add(rec.tile_number)

    def load_cleared(self, tile_number: int) -> None:
        rec = self._records.get(int(tile_number))
        if rec is not None:
            self._load_cleared(rec)

    def evaluation_requested(self, tile_number: int) -> None:
        """An evaluation request is in flight."""

        rec = self.record(tile_number)
        rec.request_active = True
        self._active_requests.add(rec.tile_number)

    def evaluation_claimed(self, tile_number: int, claim: object) -> None:
        rec = self.record(tile_number)
        rec.evaluation_claim = claim
        rec.request_active = True
        self._active_requests.add(rec.tile_number)

    def evaluation_claim_for(self, tile_number: int) -> object:
        rec = self._records.get(int(tile_number))
        return None if rec is None else rec.evaluation_claim

    def evaluation_request_cleared(self, tile_number: int) -> None:
        rec = self._records.get(int(tile_number))
        if rec is not None:
            self._request_cleared(rec)

    def evaluation_requests_cleared(self) -> None:
        """Session retarget: every in-flight request is superseded wholesale."""

        for index in tuple(self._active_requests):
            self._request_cleared(self._records[index])

    def preview_claimed(
        self, tile_number: int, rung: int, level: int, semantic_key: object
    ) -> bool:
        rec = self.record(tile_number)
        rung = int(rung)
        level = int(level)
        claim = (level, semantic_key)
        if rec.preview_claims.get(rung) == claim:
            return False
        rec.preview_claims[rung] = claim
        return True

    def preview_claim_matches(
        self, tile_number: int, rung: int, level: int, semantic_key: object
    ) -> bool:
        rec = self._records.get(int(tile_number))
        return bool(
            rec is not None and rec.preview_claims.get(int(rung)) == (int(level), semantic_key)
        )

    def preview_released(
        self,
        tile_number: int,
        rung: int,
        level: int,
        semantic_key: object,
    ) -> bool:
        rec = self._records.get(int(tile_number))
        if rec is None:
            return False
        rung = int(rung)
        if rung not in rec.preview_claims:
            return False
        if rec.preview_claims.get(rung) != (int(level), semantic_key):
            return False
        rec.preview_claims.pop(rung, None)
        return True

    def preview_claim_active(self, tile_number: int, semantic_key: object) -> bool:
        rec = self._records.get(int(tile_number))
        return bool(
            rec is not None
            and any(key == semantic_key for _level, key in rec.preview_claims.values())
        )

    def materialization_request_for(self, tile_number: int, level_key=None):
        rec = self._records.get(int(tile_number))
        if rec is None:
            return None
        for key, entry in rec.levels.items():
            if level_key is not None and key != level_key:
                continue
            if entry.phase in (LevelPhase.CLAIMED, LevelPhase.MATERIALIZING):
                return entry.request
        return None

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
        return frozenset(index for waiting in self._stage_blocked.values() for index in waiting)

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
        self._level_revision += 1
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
            self._level_revision += 1
            _trace_lifecycle(rec, "level_materializing", level_key=level_key)

    def level_resident(self, tile_number: int, level_key) -> None:
        rec = self.record(tile_number)
        entry = rec.levels.get(level_key)
        if entry is None:
            rec.levels[level_key] = _LevelEntry(LevelPhase.RESIDENT, ClaimOwner.INGEST)
        else:
            entry.phase = LevelPhase.RESIDENT
        # Repeated resident publication can carry updated preview metadata
        # even when the phase itself is unchanged.
        self._level_revision += 1
        _trace_lifecycle(rec, "level_resident", level_key=level_key)

    def level_declined(self, tile_number: int, level_key) -> tuple[ReleaseClaim, ...]:
        rec = self.record(tile_number)
        entry = rec.levels.get(level_key)
        if entry is None or entry.phase in (LevelPhase.RESIDENT, LevelPhase.RELEASED):
            return ()
        entry.phase = LevelPhase.RELEASED
        self._level_revision += 1
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
                self._level_revision += 1
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

    def active_materializations(self) -> tuple[object, ...]:
        """Claimed or materializing requests currently owned by lifecycle."""

        seen: set[int] = set()
        requests: list[tuple[int, object]] = []
        for rec in self._records.values():
            for entry in rec.levels.values():
                request = entry.request
                if (
                    entry.phase not in (LevelPhase.CLAIMED, LevelPhase.MATERIALIZING)
                    or request is None
                ):
                    continue
                identity = id(request)
                if identity in seen:
                    continue
                seen.add(identity)
                requests.append((int(entry.order), request))
        return tuple(request for _order, request in sorted(requests, key=lambda item: item[0]))

    def session_replaced(self) -> tuple[ReleaseClaim, ...]:
        """Release every non-resident claim; the session's records are done."""

        effects: list[ReleaseClaim] = []
        for rec in self._records.values():
            for level_key, entry in tuple(rec.levels.items()):
                if entry.phase in (LevelPhase.CLAIMED, LevelPhase.MATERIALIZING):
                    effects.append(ReleaseClaim(rec.tile_number, level_key, entry.owner))
                    entry.phase = LevelPhase.RELEASED
                    self._level_revision += 1
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
                self.acknowledge_presented(
                    index,
                    presented_identity,
                    rec.presented_quality or "exact",
                    rec.presented_level,
                )
                confirmed.add(index)
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
            rec.presented_quality = ""
            rec.presented_level = None
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
            # This is backend acknowledgement without an upsert, so no
            # ``backend_ack`` event exists for replay to close the target on.
            self._note_retained_satisfaction(rec)

    def note_retained_satisfaction(self, tile_numbers: Iterable[int]) -> None:
        """Commit-path re-affirmation of targets closed by retained payloads.

        A commit cycle that ends with the required scope settled and nothing
        to upsert owns the fact that no backend acknowledgement will follow.
        Safety net for presentation state restored outside the ordinary
        retarget/acknowledge events; idempotent per requirement.
        """

        for tile_number in tuple(tile_numbers or ()):
            rec = self._records.get(int(tile_number))
            if rec is not None:
                self._note_retained_satisfaction(rec)

    def _note_retained_satisfaction(self, rec: TileRecord) -> None:
        """Emit ``target_satisfied_retained`` once per target requirement.

        The rule from docs/testing/stress-and-trace-strategy.md: a subsystem
        that satisfies an obligation without doing work must say so in the
        trace, or replay cannot tell "satisfied cheaply" from "never
        satisfied".  ``quality``/``level`` describe the retained payload so
        ``trace_verify`` can re-judge the closure against later compatible
        retargets with the same settlement rule production uses.
        """

        if rec.satisfaction_traced or not trace_enabled() or not rec.target_settled:
            return
        fields = rec._presented_payload_fields()
        if fields is None:
            return
        rec.satisfaction_traced = True
        _trace_lifecycle(
            rec,
            "target_satisfied_retained",
            quality=str(fields[0]),
            level=int(fields[1]),
        )

    def presentation_discarded(self, tile_number: int) -> None:
        rec = self._records.get(int(tile_number))
        if rec is None:
            return
        self._presented.discard(rec.tile_number)
        if rec.presentation is Presentation.PRESENTED:
            rec.presentation = Presentation.UNPRESENTED
        rec.presented_source_id = None
        rec.presented_quality = ""
        rec.presented_level = None

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
        rec.evaluation_claim = None
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

    def _release_owned(self, rec: TileRecord, owner: ClaimOwner) -> tuple[ReleaseClaim, ...]:
        effects: list[ReleaseClaim] = []
        for level_key, entry in tuple(rec.levels.items()):
            if entry.owner is owner and entry.phase in (
                LevelPhase.CLAIMED,
                LevelPhase.MATERIALIZING,
            ):
                effects.append(ReleaseClaim(rec.tile_number, level_key, entry.owner))
                entry.phase = LevelPhase.RELEASED
                self._level_revision += 1
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
                if not include_released and entry.phase in (
                    LevelPhase.RESIDENT,
                    LevelPhase.RELEASED,
                ):
                    continue
                entries.append((rec, level_key, entry))
        return tuple(sorted(entries, key=lambda item: int(item[2].order)))


def _payload_base_source_id(source_id) -> object:
    if isinstance(source_id, tuple) and len(source_id) >= 3 and source_id[1] == "texture_kind":
        return source_id[0]
    if isinstance(source_id, tuple) and "floor" in source_id:
        marker = source_id.index("floor")
        prefix = source_id[:marker]
        if not prefix:
            return None
        return prefix[0] if len(prefix) == 1 else prefix
    if isinstance(source_id, tuple) and "texture_kind" in source_id:
        marker = source_id.index("texture_kind")
        prefix = source_id[:marker]
        if not prefix:
            return None
        return prefix[0] if len(prefix) == 1 else prefix
    return source_id


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


def payload_ref_from_display_payload(payload) -> TilePayloadRef:
    if isinstance(payload, TilePayloadRef):
        return payload
    lod = getattr(payload, "lod", None)
    texture_kind = getattr(payload, "texture_kind", None)
    shader_mapping = getattr(payload, "shader_mapping", None)
    quality = str(getattr(payload, "quality", "exact") or "exact")
    policy_lod_level = int(
        getattr(
            payload,
            "conservative_actual_lod_level",
            int(getattr(lod, "level", 0) or 0),
        )
    )
    return TilePayloadRef(
        source_id=getattr(payload, "source_id", None),
        quality="fallback" if quality == "preview" else quality,
        lod_level=policy_lod_level,
        source_index=int(getattr(payload, "source_index", -1)),
        texture_kind=None if texture_kind is None else getattr(texture_kind, "value", texture_kind),
        shader_mapping_key=None
        if shader_mapping is None
        else getattr(shader_mapping, "identity_key", shader_mapping),
        identity=getattr(payload, "tile_identity", None),
        payload=payload,
    )


def _coerce_payload_ref(payload) -> TilePayloadRef:
    return payload_ref_from_display_payload(payload)


def _stored_payload(ref: TilePayloadRef):
    payload = ref.payload
    if payload is not None and getattr(payload, "source_id", None) == ref.source_id:
        return payload
    return ref


def _payload_refs_match(left: TilePayloadRef | None, right: TilePayloadRef | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(
        left.source_id == right.source_id
        and left.quality == right.quality
        and left.lod_level == right.lod_level
        and left.source_index == right.source_index
        and left.texture_kind == right.texture_kind
        and left.shader_mapping_key == right.shader_mapping_key
        and left.identity == right.identity
    )


def _trace_lifecycle(rec: TileRecord, edge: str, *, payload=None, **fields) -> None:
    if not trace_enabled():
        return
    target = rec.target
    emit_trace(
        "lifecycle",
        edge=str(edge),
        tile=int(rec.tile_number),
        source_index=None if target is None else int(target.source_index),
        target_level=None if target is None else int(target.lod_level),
        semantic=str(rec.semantic.value),
        presentation=str(rec.presentation.value),
        presented_quality=str(rec.presented_quality or ""),
        presented_level=rec.presented_level,
        payload_source=None if payload is None else getattr(payload, "source_id", None),
        payload_quality=None if payload is None else getattr(payload, "quality", None),
        payload_level=None if payload is None else getattr(payload, "lod_level", None),
        **fields,
    )
