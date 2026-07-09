"""Qt-free event ledger for montage tile presentation.

The ledger is the semantic contract for a visible montage tile.  It does not
schedule work, upload textures, or repair other collections.  Callers feed it
events; callers then execute the commands/projections it derives.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping


class TileLedgerPhase(str, Enum):
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


@dataclass(frozen=True)
class TilePayloadRef:
    """Identity metadata a backend must preserve for one tile payload."""

    source_id: object
    quality: str
    lod_level: int
    source_index: int
    texture_kind: object = None
    shader_mapping_key: object = None
    payload: object = None

    def __post_init__(self) -> None:
        quality = str(self.quality or "exact")
        if quality not in {"fallback", "preview", "exact"}:
            raise ValueError(f"tile payload quality must be fallback/preview/exact, got {quality!r}")
        if quality == "preview":
            quality = "fallback"
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "lod_level", max(0, int(self.lod_level)))
        object.__setattr__(self, "source_index", int(self.source_index))

    @property
    def is_target_quality(self) -> bool:
        return self.quality == "exact"

    def satisfies_target(self, target: "TileTarget") -> bool:
        return bool(
            self.is_target_quality
            and int(self.source_index) == int(target.source_index)
            and int(self.lod_level) <= int(target.lod_level)
        )


@dataclass(frozen=True)
class TileTarget:
    tile_number: int
    source_index: int
    semantic_source_id: object
    lod_level: int = 0
    quality: str = "exact"
    visible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile_number", int(self.tile_number))
        object.__setattr__(self, "source_index", int(self.source_index))
        object.__setattr__(self, "lod_level", max(0, int(self.lod_level)))
        quality = str(self.quality or "exact")
        if quality != "exact":
            raise ValueError("montage tile targets are exact; preview is only a fallback")
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "visible", bool(self.visible))


@dataclass(frozen=True)
class TileTaskClaim:
    task_key: object
    stage_key: object = None

    def __post_init__(self) -> None:
        if self.task_key is None and self.stage_key is None:
            raise ValueError("a tile claim needs an admitted task key or a stage key")


@dataclass(frozen=True)
class TileLedgerRow:
    tile_number: int
    target: TileTarget | None = None
    fallback_payload: TilePayloadRef | None = None
    target_payload: TilePayloadRef | None = None
    task_claim: TileTaskClaim | None = None
    emitted_payload: TilePayloadRef | None = None
    acknowledged_payload: TilePayloadRef | None = None
    backend_payload: TilePayloadRef | None = None
    stage_key: object = None
    stage_producer_key: object = None
    skipped: bool = False
    failed_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile_number", int(self.tile_number))

    @property
    def active(self) -> bool:
        return bool(self.target is not None and self.target.visible and not self.skipped and not self.failed_reason)

    @property
    def target_payload_satisfies(self) -> bool:
        return bool(self.target is not None and self.target_payload is not None and self.target_payload.satisfies_target(self.target))

    @property
    def acknowledged_target_satisfies(self) -> bool:
        return bool(
            self.target is not None
            and self.acknowledged_payload is not None
            and self.acknowledged_payload.satisfies_target(self.target)
            and self.backend_payload_matches(self.acknowledged_payload)
        )

    @property
    def first_pixel_presented(self) -> bool:
        return bool(
            self.active
            and self.acknowledged_payload is not None
            and self.target is not None
            and int(self.acknowledged_payload.source_index) == int(self.target.source_index)
            and self.backend_payload_matches(self.acknowledged_payload)
        )

    @property
    def target_settled(self) -> bool:
        return self.acknowledged_target_satisfies

    @property
    def fallback_shown(self) -> bool:
        return bool(self.first_pixel_presented and not self.target_settled)

    @property
    def runnable_without_claim(self) -> bool:
        return bool(self.active and not self.target_payload_satisfies and self.task_claim is None and self.stage_key is None)

    @property
    def phase(self) -> TileLedgerPhase:
        if self.target is None or not bool(getattr(self.target, "visible", False)):
            return TileLedgerPhase.OUT_OF_SCOPE
        if self.skipped:
            return TileLedgerPhase.SKIPPED
        if self.failed_reason:
            return TileLedgerPhase.FAILED
        if self.target_settled:
            return TileLedgerPhase.TARGET_PRESENTED
        if self.emitted_payload is not None:
            return TileLedgerPhase.TARGET_EMITTED
        if self.target_payload_satisfies:
            return TileLedgerPhase.TARGET_READY
        if self.task_claim is not None and self.task_claim.task_key is not None:
            return TileLedgerPhase.TARGET_RUNNING
        if self.stage_key is not None:
            return TileLedgerPhase.TARGET_WAITING_STAGE
        if self.fallback_shown:
            return TileLedgerPhase.FALLBACK_SHOWN
        if self.acknowledged_payload is None:
            return TileLedgerPhase.NEEDS_FIRST_PIXEL
        return TileLedgerPhase.TARGET_SCHEDULABLE

    def backend_payload_matches(self, payload: TilePayloadRef | None) -> bool:
        if payload is None or self.backend_payload is None:
            return payload is not None
        return _payload_matches(self.backend_payload, payload)


@dataclass(frozen=True)
class TilePresentationCommand:
    tile_number: int
    payload: object
    payload_ref: TilePayloadRef


@dataclass(frozen=True)
class TileLedgerSnapshot:
    counts: Mapping[str, int]
    visible_tiles: int
    first_pixels_presented: bool
    visible_target_settled: bool
    orphan_running: int
    parked_without_producer: int


@dataclass
class TileLedger:
    rows: dict[int, TileLedgerRow] = field(default_factory=dict)

    def row(self, tile_number: int) -> TileLedgerRow:
        index = int(tile_number)
        row = self.rows.get(index)
        if row is None:
            row = TileLedgerRow(index)
            self.rows[index] = row
        return row

    def retarget(self, targets: Mapping[int, TileTarget]) -> None:
        wanted = {int(tile): target for tile, target in dict(targets).items()}
        for tile_number, target in wanted.items():
            current = self.row(tile_number)
            if current.target != target:
                self.rows[tile_number] = TileLedgerRow(
                    tile_number,
                    target=target,
                    fallback_payload=_compatible_payload(current.fallback_payload, target, allow_fallback=True),
                    target_payload=_compatible_payload(current.target_payload, target, allow_fallback=False),
                    acknowledged_payload=_compatible_payload(current.acknowledged_payload, target, allow_fallback=True),
                    backend_payload=_compatible_payload(current.backend_payload, target, allow_fallback=True),
                )
        for tile_number, current in tuple(self.rows.items()):
            if tile_number not in wanted and current.target is not None:
                self.rows[tile_number] = replace(current, target=replace(current.target, visible=False))

    def fallback_ready(self, tile_number: int, payload: TilePayloadRef) -> None:
        row = self.row(tile_number)
        if row.target is None or int(payload.source_index) != int(row.target.source_index):
            return
        self.rows[int(tile_number)] = replace(row, fallback_payload=replace(payload, quality="fallback"))

    def target_ready(self, tile_number: int, payload: TilePayloadRef) -> None:
        row = self.row(tile_number)
        if row.target is None or not payload.satisfies_target(row.target):
            return
        self.rows[int(tile_number)] = replace(row, target_payload=payload, task_claim=None, stage_key=None, stage_producer_key=None)

    def target_invalidated(self, tile_number: int) -> None:
        row = self.row(tile_number)
        acknowledged = row.acknowledged_payload
        backend = row.backend_payload
        fallback = row.fallback_payload
        if acknowledged is not None:
            fallback = replace(acknowledged, quality="fallback")
            acknowledged = fallback
        if backend is not None:
            backend = replace(backend, quality="fallback")
        self.rows[int(tile_number)] = replace(
            row,
            fallback_payload=fallback,
            target_payload=None,
            task_claim=None,
            emitted_payload=None,
            acknowledged_payload=acknowledged,
            backend_payload=backend,
        )

    def task_requested(self, tile_number: int, *, stage_key=None, stage_producer_key=None) -> None:
        row = self.row(tile_number)
        self.rows[int(tile_number)] = replace(row, stage_key=stage_key, stage_producer_key=stage_producer_key)

    def task_admitted(self, tile_number: int, task_key, *, stage_key=None, stage_producer_key=None) -> None:
        if task_key is None and stage_key is None:
            return
        row = self.row(tile_number)
        self.rows[int(tile_number)] = replace(
            row,
            task_claim=TileTaskClaim(task_key, stage_key),
            stage_key=stage_key,
            stage_producer_key=stage_producer_key,
        )

    def task_released(self, tile_number: int, *, reason: str = "") -> None:
        row = self.row(tile_number)
        failed = str(reason or "") if reason and reason not in {"stale", "dropped", "cancelled"} else ""
        self.rows[int(tile_number)] = replace(row, task_claim=None, stage_key=None, stage_producer_key=None, failed_reason=failed)

    def stage_waiting(self, tile_number: int, stage_key, producer_key) -> None:
        row = self.row(tile_number)
        if stage_key is None or producer_key is None:
            return
        self.rows[int(tile_number)] = replace(row, stage_key=stage_key, stage_producer_key=producer_key)

    def stage_ready(self, stage_key) -> tuple[int, ...]:
        woke: list[int] = []
        for tile_number, row in tuple(self.rows.items()):
            if row.stage_key == stage_key:
                self.rows[tile_number] = replace(row, stage_key=None, stage_producer_key=None)
                woke.append(int(tile_number))
        return tuple(sorted(woke))

    def stage_failed(self, stage_key, *, reason: str = "stage failed") -> tuple[int, ...]:
        released: list[int] = []
        for tile_number, row in tuple(self.rows.items()):
            if row.stage_key == stage_key:
                self.rows[tile_number] = replace(row, stage_key=None, stage_producer_key=None, task_claim=None, failed_reason=str(reason))
                released.append(int(tile_number))
        return tuple(sorted(released))

    def skipped(self, tile_number: int, reason: str = "skipped") -> None:
        row = self.row(tile_number)
        self.rows[int(tile_number)] = replace(row, skipped=True, failed_reason="", task_claim=None, emitted_payload=None)

    def commit_emitted(self, upserts: Mapping[int, object]) -> None:
        for tile_number, payload in dict(upserts).items():
            ref = _coerce_payload_ref(payload)
            row = self.row(tile_number)
            self.rows[int(tile_number)] = replace(row, emitted_payload=ref)

    def backend_ack(self, accepted_payloads: Mapping[int, object], *, backend_payloads: Mapping[int, object] | None = None) -> tuple[int, ...]:
        backend_refs = {
            int(tile): _coerce_payload_ref(payload)
            for tile, payload in dict(backend_payloads or accepted_payloads).items()
        }
        accepted: list[int] = []
        for tile_number, payload in dict(accepted_payloads).items():
            index = int(tile_number)
            ref = _coerce_payload_ref(payload)
            row = self.row(index)
            backend = backend_refs.get(index)
            if row.emitted_payload is None or not _payload_matches(ref, row.emitted_payload):
                if backend is not None:
                    self.rows[index] = replace(row, backend_payload=backend)
                continue
            if backend is not None and not _payload_matches(backend, ref):
                self.rows[index] = replace(row, backend_payload=backend)
                continue
            self.rows[index] = replace(
                row,
                acknowledged_payload=ref,
                backend_payload=backend or ref,
                emitted_payload=None,
            )
            accepted.append(index)
        return tuple(sorted(accepted))

    def backend_snapshot(self, payloads: Mapping[int, object]) -> None:
        for tile_number, payload in dict(payloads or {}).items():
            row = self.row(tile_number)
            self.rows[int(tile_number)] = replace(row, backend_payload=_coerce_payload_ref(payload))

    def presentation_changes(self, *, max_items: int | None = None) -> tuple[TilePresentationCommand, ...]:
        commands: list[TilePresentationCommand] = []
        for tile_number, row in sorted(self.rows.items()):
            if not row.active or row.emitted_payload is not None:
                continue
            payload = row.target_payload if row.target_payload_satisfies else row.fallback_payload
            if payload is None:
                continue
            if row.acknowledged_payload is not None and _payload_matches(row.acknowledged_payload, payload):
                continue
            commands.append(TilePresentationCommand(tile_number, payload.payload, payload))
            if max_items is not None and len(commands) >= max(0, int(max_items)):
                break
        return tuple(commands)

    def visible_first_pixels_presented(self) -> bool:
        rows = tuple(row for row in self.rows.values() if row.active)
        return bool(not rows or all(row.first_pixel_presented for row in rows))

    def visible_target_settled(self) -> bool:
        rows = tuple(row for row in self.rows.values() if row.active)
        return bool(not rows or all(row.target_settled for row in rows))

    def snapshot(self) -> TileLedgerSnapshot:
        counts: dict[str, int] = {}
        active_rows = [row for row in self.rows.values() if row.active]
        for row in active_rows:
            counts[row.phase.value] = counts.get(row.phase.value, 0) + 1
        orphan = sum(1 for row in active_rows if row.phase is TileLedgerPhase.TARGET_RUNNING and row.task_claim is None)
        parked = sum(
            1
            for row in active_rows
            if row.stage_key is not None and row.stage_producer_key is None
        )
        return TileLedgerSnapshot(
            counts=dict(sorted(counts.items())),
            visible_tiles=len(active_rows),
            first_pixels_presented=self.visible_first_pixels_presented(),
            visible_target_settled=self.visible_target_settled(),
            orphan_running=int(orphan),
            parked_without_producer=int(parked),
        )


def payload_ref_from_display_payload(payload) -> TilePayloadRef:
    lod = getattr(payload, "lod", None)
    texture_kind = getattr(payload, "texture_kind", None)
    shader_mapping = getattr(payload, "shader_mapping", None)
    quality = str(getattr(payload, "quality", "exact") or "exact")
    return TilePayloadRef(
        source_id=getattr(payload, "source_id", None),
        quality="fallback" if quality == "preview" else quality,
        lod_level=int(getattr(lod, "level", 0) or 0),
        source_index=int(getattr(payload, "source_index", -1)),
        texture_kind=None if texture_kind is None else getattr(texture_kind, "value", texture_kind),
        shader_mapping_key=None if shader_mapping is None else getattr(shader_mapping, "identity_key", shader_mapping),
        payload=payload,
    )


def _coerce_payload_ref(payload) -> TilePayloadRef:
    if isinstance(payload, TilePayloadRef):
        return payload
    return payload_ref_from_display_payload(payload)


def _compatible_payload(payload: TilePayloadRef | None, target: TileTarget, *, allow_fallback: bool) -> TilePayloadRef | None:
    if payload is None:
        return None
    if int(payload.source_index) != int(target.source_index):
        return None
    if payload.quality != "exact" and not allow_fallback:
        return None
    if payload.quality == "exact" and not payload.satisfies_target(target):
        return None
    return payload


def _payload_matches(left: TilePayloadRef, right: TilePayloadRef) -> bool:
    return bool(
        left.source_id == right.source_id
        and left.quality == right.quality
        and int(left.lod_level) == int(right.lod_level)
        and int(left.source_index) == int(right.source_index)
        and left.texture_kind == right.texture_kind
        and left.shader_mapping_key == right.shader_mapping_key
    )


__all__ = [
    "TileLedger",
    "TileLedgerPhase",
    "TileLedgerRow",
    "TileLedgerSnapshot",
    "TilePayloadRef",
    "TilePresentationCommand",
    "TileTarget",
    "TileTaskClaim",
    "payload_ref_from_display_payload",
]
