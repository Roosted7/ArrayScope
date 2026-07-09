"""Qt-free tile obligation projection.

The lifecycle records are the durable facts, but presentation needs a single
read model that answers "what does this visible tile still owe?" without
correlating dirty/upsert/loading/stage maps at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass


def _payload_quality(payload) -> str:
    return str(getattr(payload, "quality", "exact") or "exact")


def _payload_level(payload) -> int:
    lod = getattr(payload, "lod", None)
    return int(getattr(lod, "level", 0) or 0)


@dataclass(frozen=True)
class TileObligation:
    tile_number: int
    source_index: int
    visible: bool
    skipped: bool
    pending: bool
    rendered: bool
    payload_present: bool
    payload_source_matches: bool
    payload_identity: object = None
    payload_quality: str = ""
    payload_level: int = 0
    presented: bool = False
    presented_identity: object = None
    presented_quality: str = ""
    presented_level: int | None = None
    backend_identity: object = None
    target_level: int = 0
    evaluation_active: bool = False
    loading: bool = False
    dirty: bool = False
    pending_upsert: bool = False
    stage_key: object = None
    stage_ready: bool = False
    stage_live: bool = False

    @property
    def first_pixel_presented(self) -> bool:
        if not self.visible or self.skipped or not self.presented:
            return False
        if not self.payload_present or not self.payload_source_matches:
            return False
        if self.backend_identity is not None and self.backend_identity != self.payload_identity:
            return False
        return True

    @property
    def target_payload_present(self) -> bool:
        if not self.payload_present or not self.payload_source_matches:
            return False
        if self.payload_quality != "exact":
            return False
        # A finer resident level is always a valid display target.
        return int(self.payload_level) <= int(self.target_level)

    @property
    def target_presented(self) -> bool:
        if not self.first_pixel_presented or not self.target_payload_present:
            return False
        if self.presented_identity is not None and self.presented_identity != self.payload_identity:
            return False
        if self.presented_quality and self.presented_quality != "exact":
            return False
        return True

    @property
    def preview_fallback_visible(self) -> bool:
        return bool(self.first_pixel_presented and not self.target_presented)

    @property
    def target_unsettled(self) -> bool:
        return bool(self.visible and not self.skipped and not self.target_presented)

    @property
    def target_payload_owed(self) -> bool:
        return bool(self.target_unsettled and self.rendered and not self.target_payload_present)

    @property
    def active_without_path(self) -> bool:
        return bool(
            self.evaluation_active
            and not self.rendered
            and not self.pending
            and self.stage_key is None
        )

    @property
    def stage_lost(self) -> bool:
        return bool(
            self.stage_key is not None
            and not self.stage_ready
            and not self.stage_live
            and not self.rendered
        )


@dataclass(frozen=True)
class TileObligationPlan:
    obligations: tuple[TileObligation, ...]

    def by_tile(self) -> dict[int, TileObligation]:
        return {int(item.tile_number): item for item in self.obligations}

    @property
    def visible_target_unsettled(self) -> tuple[int, ...]:
        return tuple(int(item.tile_number) for item in self.obligations if item.target_unsettled)

    @property
    def preview_fallbacks(self) -> tuple[int, ...]:
        return tuple(int(item.tile_number) for item in self.obligations if item.preview_fallback_visible)

    @property
    def target_payload_owed(self) -> tuple[int, ...]:
        return tuple(int(item.tile_number) for item in self.obligations if item.target_payload_owed)

    @property
    def active_without_path(self) -> tuple[int, ...]:
        return tuple(int(item.tile_number) for item in self.obligations if item.active_without_path)

    @property
    def stage_ready(self) -> tuple[int, ...]:
        return tuple(
            int(item.tile_number)
            for item in self.obligations
            if item.stage_key is not None and item.stage_ready and not item.rendered
        )

    @property
    def stage_lost(self) -> tuple[int, ...]:
        return tuple(int(item.tile_number) for item in self.obligations if item.stage_lost)

    @property
    def stage_blocked(self) -> tuple[int, ...]:
        return tuple(
            int(item.tile_number)
            for item in self.obligations
            if item.stage_key is not None and not item.stage_ready and item.stage_live
        )

    def counters(self) -> dict[str, int]:
        return {
            "visible_target_unsettled": len(self.visible_target_unsettled),
            "preview_fallbacks": len(self.preview_fallbacks),
            "target_payload_owed": len(self.target_payload_owed),
            "active_without_path": len(self.active_without_path),
            "stage_ready": len(self.stage_ready),
            "stage_lost": len(self.stage_lost),
            "stage_blocked": len(self.stage_blocked),
        }


def build_tile_obligation_plan(session) -> TileObligationPlan:
    visible = {int(tile) for tile in tuple(getattr(session, "visible_tile_numbers", ()) or ())}
    skipped = {int(tile) for tile in tuple(getattr(session, "skipped_tiles", ()) or ())}
    rendered = {int(tile) for tile in getattr(session, "rendered_tiles", {})}
    pending = {int(tile) for tile in tuple(getattr(session, "pending_tile_numbers", lambda: ())() or ())}
    loading = {int(tile) for tile in tuple(getattr(session, "loading_tiles", ()) or ())}
    active = {int(tile) for tile in tuple(getattr(session, "active_tile_requests", ()) or ())}
    dirty = {int(tile) for tile in getattr(session, "dirty_payloads", {})}
    upserts = {int(tile) for tile in getattr(session, "pending_payload_upserts", {})}
    lifecycle = getattr(session, "lifecycle", None)
    backend = dict(getattr(lifecycle, "backend_presented_identities", {}) or {})
    presented = {int(tile) for tile in tuple(getattr(lifecycle, "presented_tiles", ()) or ())}
    display_payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    state_payloads = dict(getattr(getattr(session, "tile_presentation_state", None), "payloads", {}) or {})
    stage_fan_in = getattr(session, "stage_fan_in", None)
    stage_keys = dict(getattr(stage_fan_in, "tile_stage_keys", {}) or {})
    stage_values = dict(getattr(stage_fan_in, "values", {}) or {})
    live_stage_keys = (
        set(getattr(stage_fan_in, "active_requests", set()) or set())
        | set(getattr(stage_fan_in, "attached_requests", set()) or set())
        | set(stage_values)
    )
    demand = getattr(getattr(session, "lod_policy_decision", None), "demand", None)
    target_level = int(getattr(demand, "desired_level", 0) or 0)
    plan_tiles = {
        int(getattr(tile, "montage_index")): tile
        for tile in tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
    }
    scope = set(visible) | set(presented) | set(rendered) | set(active) | set(pending) | set(dirty) | set(upserts)
    obligations: list[TileObligation] = []
    for tile_number in sorted(scope):
        tile = plan_tiles.get(int(tile_number))
        source_index = -1 if tile is None else int(getattr(tile, "source_index", -1))
        payload = display_payloads.get(int(tile_number)) or state_payloads.get(int(tile_number))
        payload_identity = None if payload is None else getattr(payload, "source_id", None)
        payload_source_matches = bool(
            payload is not None
            and source_index >= 0
            and int(getattr(payload, "source_index", -2)) == int(source_index)
        )
        rec = None if lifecycle is None else lifecycle.peek(int(tile_number))
        stage_key = stage_keys.get(int(tile_number))
        obligations.append(
            TileObligation(
                tile_number=int(tile_number),
                source_index=source_index,
                visible=int(tile_number) in visible,
                skipped=int(tile_number) in skipped,
                pending=int(tile_number) in pending,
                rendered=int(tile_number) in rendered,
                payload_present=payload is not None,
                payload_source_matches=payload_source_matches,
                payload_identity=payload_identity,
                payload_quality="" if payload is None else _payload_quality(payload),
                payload_level=0 if payload is None else _payload_level(payload),
                presented=int(tile_number) in presented,
                presented_identity=None if rec is None else rec.presented_source_id,
                presented_quality="" if rec is None else str(rec.presented_quality or ""),
                presented_level=None if rec is None else rec.presented_level,
                backend_identity=backend.get(int(tile_number)),
                target_level=target_level,
                evaluation_active=int(tile_number) in active,
                loading=int(tile_number) in loading,
                dirty=int(tile_number) in dirty,
                pending_upsert=int(tile_number) in upserts,
                stage_key=stage_key,
                stage_ready=bool(stage_key is not None and stage_key in stage_values),
                stage_live=bool(stage_key is not None and stage_key in live_stage_keys),
            )
        )
    return TileObligationPlan(tuple(obligations))


__all__ = ["TileObligation", "TileObligationPlan", "build_tile_obligation_plan"]
