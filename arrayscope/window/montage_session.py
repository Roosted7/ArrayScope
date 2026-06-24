"""Qt-free state for progressive montage rendering."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from time import monotonic

import numpy as np

from arrayscope.display.lod import LodInfo, select_lod_factor
from arrayscope.display.montage import (
    MontagePlan,
    MontageTile,
    MontageTileState,
    MontageViewportCanvas,
    RenderedTile,
    montage_rect_for_viewport,
    patch_rendered_tile_into_canvas,
)
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.display.model.frame import DisplayTilePayload, TileCommitReport, TilePresentationDelta, TilePresentationState
from arrayscope.window.montage_priority import MontageTilePriorityQueue, TilePriorityContext, tile_numbers


def _shader_mapping_key(mapping):
    return None if mapping is None else getattr(mapping, "identity_key", mapping)


def _viewport_tiles(
    plan: MontagePlan,
    *,
    view_range,
    viewport_shape: tuple[int, int],
    margin_tiles: int,
) -> tuple[MontageTile, ...]:
    rect = montage_rect_for_viewport(
        plan,
        view_range=view_range,
        viewport_shape=viewport_shape,
    )
    return plan.tiles_intersecting(
        ((rect[0], rect[2]), (rect[1], rect[3])),
        margin_tiles=max(0, int(margin_tiles)),
    )


def _cap_tile_upserts(
    upserts: dict[int, DisplayTilePayload],
    *,
    active_tiles: tuple[int, ...],
    max_upserts: int | None,
    max_upsert_bytes: int | None,
) -> dict[int, DisplayTilePayload]:
    if not upserts:
        return {}
    item_cap = None if max_upserts is None else max(0, int(max_upserts))
    byte_cap = None if max_upsert_bytes is None else max(0, int(max_upsert_bytes))
    if item_cap is None and byte_cap is None:
        return upserts
    if item_cap == 0 or byte_cap == 0:
        return {}

    del active_tiles
    ordered = tuple((int(tile), payload) for tile, payload in upserts.items())
    capped: dict[int, DisplayTilePayload] = {}
    used_bytes = 0
    for tile, payload in ordered:
        if item_cap is not None and len(capped) >= item_cap:
            break
        payload_bytes = int(getattr(payload, "nbytes", 0) or 0)
        if byte_cap is not None and capped and used_bytes + payload_bytes > byte_cap:
            break
        capped[int(tile)] = payload
        used_bytes += max(0, payload_bytes)
    return capped


@dataclass
class MontageRenderSession:
    session_id: int
    key: object
    render_generation: int
    level_key: object
    level_expected_indices: tuple[int, ...]
    plan: MontagePlan
    view_state: object
    document: object
    montage_axis: int
    colormap_lut: object | None
    viewport_shape: tuple[int, int]
    view_range: object
    output_dtype: np.dtype
    rgb: bool
    window_mode: object
    force_auto: bool
    visible_tiles: tuple[MontageTile, ...]
    rendered_tiles: dict[int, RenderedTile]
    loading_tiles: set[int]
    skipped_tiles: set[int]
    pending_tiles: MontageTilePriorityQueue | deque[MontageTile] | list[MontageTile]
    active_tile_requests: set[int] = field(default_factory=set)
    presented_tiles: set[int] = field(default_factory=set)
    tile_stage_keys: dict[int, object] = field(default_factory=dict)
    stage_waiting_tiles: dict[object, list[MontageTile]] = field(default_factory=dict)
    active_stage_requests: set[object] = field(default_factory=set)
    attached_stage_requests: set[object] = field(default_factory=set)
    stage_values: dict[object, object] = field(default_factory=dict)
    lead_stage_warmups: dict[int, object] = field(default_factory=dict)
    canvas: MontageViewportCanvas | None = None
    canvas_data: np.ndarray | None = None
    canvas_histogram_data: np.ndarray | None = None
    canvas_rect: tuple[int, int, int, int] | None = None
    tile_states: list[MontageTileState] = field(default_factory=list)
    dirty_rects: list[tuple[int, int, int, int]] = field(default_factory=list)
    dirty_tiles: list[int] = field(default_factory=list)
    flush_pending: bool = False
    last_commit_monotonic: float = 0.0
    final_commit_pending: bool = False
    show_loading_overlays: bool = False
    defer_side_panels: bool = False
    display_committed: bool = False
    applied_level_source: object | None = None
    user_levels_override: tuple[float, float] | None = None
    pending_level_tiles: deque[RenderedTile] = field(default_factory=deque)
    pending_level_sources: set[int] = field(default_factory=set)
    pending_refined_level_tiles: deque[RenderedTile] = field(default_factory=deque)
    pending_refined_level_sources: set[int] = field(default_factory=set)
    pending_completed_tiles: deque[tuple[MontageTile, object]] = field(default_factory=deque)
    tile_compute_cache_hits: int = 0
    tile_compute_stage_backed: int = 0
    tile_compute_direct: int = 0
    tile_compute_waiting_for_stage: int = 0
    lead_direct_tiles: int = 0
    stage_backed_tiles_pending: int = 0
    retained_stage_index: int | None = None
    retained_stage_decision: str = ""
    repeated_expensive_stage_per_tile: bool = False
    tile_source_ids: dict[int, object] = field(default_factory=dict)
    display_tile_payloads: dict[int, DisplayTilePayload] = field(default_factory=dict)
    dirty_payloads: OrderedDict[int, None] = field(default_factory=OrderedDict)
    pending_payload_upserts: OrderedDict[int, None] = field(default_factory=OrderedDict)
    pending_removals: set[int] = field(default_factory=set)
    tile_presentation_state: TilePresentationState = field(default_factory=TilePresentationState)
    structure_revision: int = 0
    payload_revision: int = 0
    visibility_revision: int = 0
    level_revision: int = 0
    histogram_revision: int = 0
    viewport_revision: int = 0
    tile_state_revision: int = 0
    priority_focus: tuple[float, float] | None = None
    priority_retargeted_tiles: int = 0
    priority_fairness_pops: int = 0
    tile_lod_factor: int = 1
    desired_tile_lod_factor: int = 1
    _last_active_tiles: tuple[int, ...] = ()
    _last_planned_tiles: tuple[int, ...] = ()
    _last_near_tiles: tuple[int, ...] = ()
    _near_tile_numbers_cache_key: tuple[object, ...] | None = None
    _near_tile_numbers_cache: tuple[int, ...] = ()
    _tile_states_cached_revision: int = -1
    _tile_states_cached_tuple: tuple[MontageTileState, ...] = ()

    def __post_init__(self) -> None:
        # These queues are drained throughout progressive rendering.  The
        # visible tile queue is indexed so viewport/hover retargeting updates
        # metadata instead of sorting inside high-frequency callbacks.
        pending = tuple(self.pending_tiles or ())
        context = self._tile_priority_context()
        if isinstance(self.pending_tiles, MontageTilePriorityQueue):
            self.pending_tiles.set_context(context, max_items=len(self.pending_tiles))
        else:
            self.pending_tiles = MontageTilePriorityQueue(pending, context=context)
        self.stage_waiting_tiles = {
            key: (
                value
                if isinstance(value, MontageTilePriorityQueue)
                else MontageTilePriorityQueue(tuple(value or ()), context=context)
            )
            for key, value in dict(self.stage_waiting_tiles or {}).items()
        }
        self.pending_level_tiles = deque(self.pending_level_tiles)
        self.pending_level_sources = {
            int(source) for source in (self.pending_level_sources or ())
        } or {int(item.tile.source_index) for item in self.pending_level_tiles}
        self.pending_refined_level_tiles = deque(self.pending_refined_level_tiles)
        self.pending_refined_level_sources = {
            int(source) for source in (self.pending_refined_level_sources or ())
        } or {int(item.tile.source_index) for item in self.pending_refined_level_tiles}
        self.pending_completed_tiles = deque(self.pending_completed_tiles)
        for index in sorted(int(tile) for tile in self.rendered_tiles):
            self.dirty_payloads.setdefault(int(index), None)

    def is_tile_loaded(self, tile) -> bool:
        return int(tile.montage_index) in self.rendered_tiles

    def retarget_viewport(
        self,
        *,
        view_range,
        viewport_shape: tuple[int, int],
        coverage_margin_tiles: int = 1,
        near_margin_tiles: int = 2,
        priority_focus: tuple[float, float] | None = None,
        priority_retarget_limit: int = 64,
    ) -> tuple[tuple[MontageTile, ...], bool]:
        """Retarget draw and compute coverage without replacing the session.

        The current draw set is kept separate from loaded payload ownership.
        Tiles outside the viewport can therefore remain GPU-resident and be
        reused when the user pans back.
        """

        self.view_range = view_range
        self.viewport_shape = (max(1, int(viewport_shape[0])), max(1, int(viewport_shape[1])))
        self.priority_focus = priority_focus
        self._selected_lod_factor()
        active = _viewport_tiles(
            self.plan,
            view_range=view_range,
            viewport_shape=self.viewport_shape,
            margin_tiles=0,
        )
        coverage = _viewport_tiles(
            self.plan,
            view_range=view_range,
            viewport_shape=self.viewport_shape,
            margin_tiles=max(0, int(coverage_margin_tiles)),
        )
        near = _viewport_tiles(
            self.plan,
            view_range=view_range,
            viewport_shape=self.viewport_shape,
            margin_tiles=max(0, int(near_margin_tiles)),
        )
        known = set(int(index) for index in self.rendered_tiles)
        known.update(int(index) for index in self.loading_tiles)
        known.update(int(index) for index in self.skipped_tiles)
        known.update(int(index) for index in self.active_tile_requests)
        known.update(int(tile.montage_index) for tile in self.pending_tiles)
        additions = tuple(tile for tile in coverage if int(tile.montage_index) not in known)
        active_numbers = tuple(int(tile.montage_index) for tile in active)
        near_numbers = tuple(int(tile.montage_index) for tile in near)
        presentation_changed = (
            active_numbers != tuple(int(tile.montage_index) for tile in self.visible_tiles)
            or near_numbers != self._last_near_tiles
        )
        self.visible_tiles = active
        self.priority_retargeted_tiles = self.retarget_tile_priority(
            focus=priority_focus,
            max_items=max(1, int(priority_retarget_limit)),
            active_tiles=active_numbers,
            near_tiles=near_numbers,
        )
        return additions, bool(presentation_changed)

    def expand_viewport_coverage(
        self,
        *,
        view_range,
        viewport_shape: tuple[int, int],
        margin_tiles: int = 1,
    ) -> tuple[MontageTile, ...]:
        additions, _changed = self.retarget_viewport(
            view_range=view_range,
            viewport_shape=viewport_shape,
            coverage_margin_tiles=margin_tiles,
        )
        return additions

    def mark_loaded(self, rendered: RenderedTile) -> None:
        """Compatibility alias for the materialization stage.

        A tile is not user-visible merely because CPU/GPU source data has
        arrived.  It becomes loaded only once the presentation backend has
        accepted it through ``mark_presented``.
        """

        self.mark_materialized(rendered)

    def mark_materialized(self, rendered: RenderedTile) -> None:
        index = int(rendered.tile.montage_index)
        self.rendered_tiles[index] = rendered
        # A replacement tile must be assigned a fresh semantic source identity
        # and presentation payload on the next commit.  Other loaded tiles keep
        # their cached wrappers across progressive batches.
        self.tile_source_ids.pop(index, None)
        self.display_tile_payloads.pop(index, None)
        self.dirty_payloads[index] = None
        self.active_tile_requests.discard(index)
        self.skipped_tiles.discard(index)
        self.discard_pending_tile(index)
        if index not in self.presented_tiles:
            self.loading_tiles.add(index)
            self.mark_tile_state(rendered.tile, MontageTileState.LOADING)

    def mark_presented(self, tile_numbers) -> None:
        for tile_number in tuple(tile_numbers or ()):
            index = int(tile_number)
            if index not in self.rendered_tiles:
                continue
            self.presented_tiles.add(index)
            self.loading_tiles.discard(index)
            self.skipped_tiles.discard(index)
            self.dirty_payloads.pop(index, None)
            self.pending_removals.discard(index)
            if 0 <= index < len(self.plan.tiles):
                self.mark_tile_state(self.plan.tiles[index], MontageTileState.LOADED)

    def snapshot_display_tile_payloads(self, source_ids: dict[int, object]) -> dict[int, DisplayTilePayload]:
        """Return immutable-by-convention payload wrappers for loaded tiles.

        Arrays are not copied.  Stable wrappers are retained across progressive
        commits, avoiding repeated dataclass construction and dtype/shape
        normalization for every tile already on screen.
        """

        loaded = {int(index) for index in self.rendered_tiles}
        for stale in tuple(self.display_tile_payloads):
            if int(stale) not in loaded:
                self.display_tile_payloads.pop(int(stale), None)
        lod_factor = self._selected_lod_factor()
        for tile_number, rendered in self.rendered_tiles.items():
            self._ensure_display_tile_payload(int(tile_number), rendered, source_ids, lod_factor=lod_factor)
        return dict(self.display_tile_payloads)

    def _ensure_display_tile_payload(
        self,
        tile_number: int,
        rendered: RenderedTile,
        source_ids: dict[int, object],
        *,
        lod_factor: int,
    ) -> DisplayTilePayload:
        tile_number = int(tile_number)
        base_source_id = source_ids.get(tile_number, ("rendered_tile", tile_number, id(rendered.image)))
        mapping = getattr(rendered, "shader_mapping", None)
        texture_kind = getattr(rendered, "texture_kind", None)
        exact_image = np.asarray(rendered.image)
        exact_histogram = None if rendered.histogram_data is None else np.asarray(rendered.histogram_data)
        semantic = getattr(rendered, "semantic_data", None)
        semantic = exact_image if semantic is None else np.asarray(semantic)
        lod = self._planned_lod_info(rendered, factor=lod_factor)
        texture_data, texture_histogram, lod = self._texture_for_rendered_tile(rendered, factor=lod_factor)
        del texture_histogram
        source_id = self._payload_source_id(
            base_source_id,
            texture_kind=texture_kind,
            mapping=mapping,
            lod=lod,
            texture_data=texture_data,
        )
        previous = self.display_tile_payloads.get(tile_number)
        if (
            previous is not None
            and _base_source_id(previous.source_id) == base_source_id
            and previous.source_id == source_id
            and previous.image is exact_image
            and previous.histogram_data is exact_histogram
            and _shader_mapping_key(previous.shader_mapping) == _shader_mapping_key(mapping)
        ):
            return previous
        payload = DisplayTilePayload(
            tile_number=tile_number,
            source_index=int(rendered.tile.source_index),
            image=exact_image,
            histogram_data=exact_histogram,
            source_id=source_id,
            texture_data=texture_data,
            texture_kind=texture_kind,
            semantic_data=semantic,
            semantic_histogram_data=exact_histogram,
            source_shape=tuple(int(value) for value in exact_image.shape[:2]),
            lod=lod,
            shader_mapping=mapping,
        )
        self.display_tile_payloads[tile_number] = payload
        return payload

    def seed_display_tile_payloads(self, previous_payloads: dict[int, DisplayTilePayload], source_ids: dict[int, object]) -> None:
        """Reuse compatible wrappers *and committed presentation ownership*.

        Retargeting a montage is a placement change, not evidence that resident
        tiles disappeared.  Seeding only the wrapper cache left the committed
        tiled state empty in fresh sessions, so destructive removals could be
        emitted before replacement upserts.  Exact source identities are still
        required, so semantic changes do not reuse stale pixels.
        """

        if not previous_payloads or not self.rendered_tiles:
            return
        lod_factor = self._selected_lod_factor()
        by_source = {payload.source_id: payload for payload in dict(previous_payloads).values()}
        seeded_state = dict(getattr(self.tile_presentation_state, "payloads", {}) or {})
        changed_state = False
        for tile_number, rendered in self.rendered_tiles.items():
            tile_number = int(tile_number)
            previous = self.display_tile_payloads.get(tile_number)
            owns_committed_presentation = tile_number in self.presented_tiles
            if previous is None:
                base_source_id = source_ids.get(tile_number, ("rendered_tile", tile_number, id(rendered.image)))
                texture_data, _texture_histogram, lod = self._texture_for_rendered_tile(rendered, factor=lod_factor)
                source_id = self._payload_source_id(
                    base_source_id,
                    texture_kind=getattr(rendered, "texture_kind", None),
                    mapping=getattr(rendered, "shader_mapping", None),
                    lod=lod,
                    texture_data=texture_data,
                )
                previous = by_source.get(source_id)
                owns_committed_presentation = previous is not None
            if previous is None:
                continue
            if int(previous.tile_number) == tile_number and int(previous.source_index) == int(rendered.tile.source_index):
                payload = previous
            else:
                payload = replace(
                    previous,
                    tile_number=tile_number,
                    source_index=int(rendered.tile.source_index),
                )
            self.display_tile_payloads[tile_number] = payload
            if owns_committed_presentation and seeded_state.get(tile_number) is not payload:
                seeded_state[tile_number] = payload
                self.pending_payload_upserts[tile_number] = None
                changed_state = True
        if changed_state:
            self.tile_presentation_state = TilePresentationState(
                seeded_state,
                revision=int(getattr(self.tile_presentation_state, "revision", 0)),
            )
            self.presented_tiles.update(int(tile) for tile in seeded_state)
            self.invalidate_tile_states()

    def _payload_source_id(self, base_source_id, *, texture_kind, mapping, lod: LodInfo, texture_data) -> tuple[object, ...]:
        del mapping
        prefix = tuple(base_source_id) if isinstance(base_source_id, tuple) else (base_source_id,)
        return (
            *prefix,
            "texture_kind",
            None if texture_kind is None else getattr(texture_kind, "value", texture_kind),
            # Shader uniforms do not change texture content.  Keeping this
            # compatibility marker avoids invalidating existing source-key
            # parsing while preventing level/LUT changes from re-uploading.
            "shader",
            None,
            "lod",
            int(lod.factor),
            int(lod.level),
            int(lod.gutter),
            "content",
            _array_content_token(texture_data),
        )

    def _selected_lod_factor(self) -> int:
        desired = select_lod_factor(
            self.view_range,
            self.viewport_shape,
            self.plan.tile_shape,
            previous_factor=self.desired_tile_lod_factor,
        )
        self.desired_tile_lod_factor = int(desired)
        # CPU pyramid construction used to happen synchronously from
        # snapshot_display_tile_payloads(), which is a UI commit path.  Until a
        # worker/GPU LOD cache can retain adjacent levels, keep the exact texture
        # resident and let hardware filtering handle zoomed-out sampling.
        self.tile_lod_factor = 1
        return 1

    def _planned_lod_info(self, rendered: RenderedTile, *, factor: int) -> LodInfo:
        del factor
        texture_kind = getattr(rendered, "texture_kind", None)
        if texture_kind is not None and not isinstance(texture_kind, TexturePlaneKind):
            texture_kind = TexturePlaneKind(getattr(texture_kind, "value", texture_kind))
        if texture_kind == TexturePlaneKind.COMPLEX_RG32F and getattr(rendered, "semantic_data", None) is not None:
            source = np.asarray(rendered.semantic_data)
        else:
            source = np.asarray(rendered.image)
        source_shape = tuple(int(value) for value in source.shape[:2])
        return LodInfo(level=0, factor=1, source_shape=source_shape, texture_shape=source_shape, gutter=0)

    def _texture_for_rendered_tile(self, rendered: RenderedTile, *, factor: int | None = None) -> tuple[np.ndarray, np.ndarray | None, LodInfo]:
        del factor
        texture_kind = getattr(rendered, "texture_kind", None)
        if texture_kind is not None and not isinstance(texture_kind, TexturePlaneKind):
            texture_kind = TexturePlaneKind(getattr(texture_kind, "value", texture_kind))
        if texture_kind == TexturePlaneKind.COMPLEX_RG32F and getattr(rendered, "semantic_data", None) is not None:
            source = np.asarray(rendered.semantic_data)
        else:
            source = np.asarray(rendered.image)
        histogram = None if rendered.histogram_data is None else np.asarray(rendered.histogram_data)
        source_shape = tuple(int(value) for value in source.shape[:2])
        lod = LodInfo(level=0, factor=1, source_shape=source_shape, texture_shape=source_shape, gutter=0)
        return source, histogram, lod

    def build_tile_presentation(
        self,
        source_ids: dict[int, object] | None,
        *,
        cold_deadline_ms: float | None = None,
        max_upserts: int | None = None,
        max_upsert_bytes: int | None = None,
    ) -> tuple[TilePresentationState, TilePresentationDelta]:
        source_ids = dict(source_ids or {})
        previous_state = self.tile_presentation_state
        previous_payloads = dict(previous_state.payloads)
        loaded = {int(index) for index in self.rendered_tiles}
        for stale in tuple(self.display_tile_payloads):
            if int(stale) not in loaded:
                self.display_tile_payloads.pop(int(stale), None)
                self.pending_removals.add(int(stale))
                self.dirty_payloads.pop(int(stale), None)
                self.pending_payload_upserts.pop(int(stale), None)
        lod_factor = self._selected_lod_factor()
        dirty_payload_tiles = tuple(
            dict.fromkeys(
                (
                    *(int(tile) for tile in self.dirty_payloads),
                    *(int(tile) for tile in self.pending_payload_upserts),
                )
            )
        )
        for tile_number in dirty_payload_tiles:
            rendered = self.rendered_tiles.get(int(tile_number))
            if rendered is not None:
                self._ensure_display_tile_payload(int(tile_number), rendered, source_ids, lod_factor=lod_factor)
        current_payloads = self.display_tile_payloads
        current_loaded = set(self.rendered_tiles)
        planned = tuple(
            int(tile.montage_index)
            for tile in tuple(self.visible_tiles)
            if int(tile.montage_index) not in self.skipped_tiles
        )
        active = tuple(int(tile) for tile in planned if int(tile) in current_loaded)
        valid_tile_count = len(tuple(getattr(self.plan, "tiles", ()) or ()))
        removals = tuple(
            sorted(
                {
                    int(tile)
                    for tile in previous_payloads
                    if int(tile) < 0 or int(tile) >= valid_tile_count or int(tile) in self.skipped_tiles
                }.union(int(tile) for tile in self.pending_removals)
            )
        )
        upserts: dict[int, DisplayTilePayload] = {}
        for tile_number in dirty_payload_tiles:
            payload = current_payloads.get(int(tile_number))
            if payload is None:
                continue
            previous = previous_payloads.get(int(tile_number))
            force_upsert = int(tile_number) in self.pending_payload_upserts
            if previous is payload and not force_upsert:
                continue
            if (
                not force_upsert
                and previous is not None
                and previous.source_id == payload.source_id
                and previous.image is payload.image
                and previous.histogram_data is payload.histogram_data
                and _shader_mapping_key(previous.shader_mapping) == _shader_mapping_key(payload.shader_mapping)
            ):
                continue
            upserts[int(tile_number)] = payload

        upserts = _cap_tile_upserts(
            upserts,
            active_tiles=active,
            max_upserts=max_upserts,
            max_upsert_bytes=max_upsert_bytes,
        )
        if max_upserts is not None or max_upsert_bytes is not None:
            admitted = set(int(tile) for tile in upserts)
            admitted.update(int(tile) for tile in self.presented_tiles)
            active = tuple(int(tile) for tile in active if int(tile) in admitted)
        near = tuple(tile for tile in self._near_tile_numbers(margin_tiles=2) if int(tile) not in self.skipped_tiles)
        # Residency is keyed by the complete texture-content identity carried
        # by DisplayTilePayload.source_id, not the evaluator's base tile key.
        # Supplying the base key here made inactive near-viewport tiles look
        # unrelated to their resident atlas slots, so the LRU could evict the
        # very tiles that warm residency was meant to protect.
        near_source_ids = {
            int(tile): (
                current_payloads[int(tile)].source_id
                if int(tile) in current_payloads
                else source_ids[int(tile)]
            )
            for tile in near
            if int(tile) in current_payloads or int(tile) in source_ids
        }

        force_refresh = False
        clear_reason = ""

        base_revision = int(getattr(previous_state, "revision", 0))
        target_revision = base_revision + (1 if upserts or removals else 0)
        if upserts or removals:
            self.payload_revision += 1
        if active != self._last_active_tiles or planned != self._last_planned_tiles:
            self.visibility_revision += 1
        if near != self._last_near_tiles:
            self.viewport_revision += 1
        if planned != self._last_planned_tiles:
            self.structure_revision += 1
        self._last_active_tiles = active
        self._last_planned_tiles = planned
        self._last_near_tiles = near

        delta = TilePresentationDelta(
            structure_revision=self.structure_revision,
            payload_revision=self.payload_revision,
            visibility_revision=self.visibility_revision,
            level_revision=self.level_revision,
            histogram_revision=self.histogram_revision,
            viewport_revision=self.viewport_revision,
            base_revision=base_revision,
            target_revision=target_revision,
            cold_deadline_ms=cold_deadline_ms,
            upserts=upserts,
            removals=removals,
            active_tiles=active,
            planned_tiles=planned,
            near_tiles=near,
            near_tile_source_ids=near_source_ids,
            force_refresh=force_refresh,
            clear_reason=clear_reason,
        )
        state = previous_state.apply_delta(delta)
        return state, delta

    def acknowledge_tile_presentation(self, delta: TilePresentationDelta, report: TileCommitReport | None) -> TilePresentationState:
        if report is None:
            report = TileCommitReport(
                presented_tiles=self.tile_presentation_state.apply_delta(delta).active_payloads(delta),
                removed_tiles=delta.removals,
            )
        report = report if isinstance(report, TileCommitReport) else TileCommitReport()
        acknowledged = self.tile_presentation_state.acknowledge_delta(delta, report)
        self.tile_presentation_state = acknowledged
        for tile in report.presented_tiles:
            self.dirty_payloads.pop(int(tile), None)
            self.pending_payload_upserts.pop(int(tile), None)
        for tile in report.removed_tiles:
            self.pending_removals.discard(int(tile))
            self.dirty_payloads.pop(int(tile), None)
            self.pending_payload_upserts.pop(int(tile), None)
            self.display_tile_payloads.pop(int(tile), None)
        return acknowledged

    def mark_loading(self, tile: MontageTile) -> None:
        index = int(tile.montage_index)
        if index not in self.rendered_tiles and index not in self.skipped_tiles:
            self.loading_tiles.add(index)
            self.mark_tile_state(tile, MontageTileState.LOADING)

    def mark_skipped(self, tile: MontageTile) -> None:
        index = int(tile.montage_index)
        if index not in self.rendered_tiles:
            self.active_tile_requests.discard(index)
            self.loading_tiles.discard(index)
            self.skipped_tiles.add(index)
            self.pending_removals.add(index)
            self.dirty_payloads.pop(index, None)
            self.display_tile_payloads.pop(index, None)
            self.tile_source_ids.pop(index, None)
            self.discard_pending_tile(index)
            self.mark_tile_state(tile, MontageTileState.SKIPPED)

    def next_tile(self) -> MontageTile | None:
        self._ensure_pending_priority_queue()
        while self.pending_tiles:
            tile = self.pending_tiles.pop()
            index = int(tile.montage_index)
            if index not in self.rendered_tiles and index not in self.skipped_tiles:
                self.mark_loading(tile)
                self.active_tile_requests.add(index)
                self.priority_fairness_pops = int(getattr(self.pending_tiles, "fairness_pops", 0) or 0)
                return tile
        return None

    def rendered_tuple(self) -> tuple[RenderedTile, ...]:
        return tuple(sorted(self.rendered_tiles.values(), key=lambda rendered: rendered.tile.montage_index))

    def loading_tile_tuple(self) -> tuple[MontageTile, ...]:
        return tuple(self.plan.tiles[index] for index in sorted(self.loading_tiles) if 0 <= index < len(self.plan.tiles))

    def skipped_tile_tuple(self) -> tuple[MontageTile, ...]:
        return tuple(self.plan.tiles[index] for index in sorted(self.skipped_tiles) if 0 <= index < len(self.plan.tiles))

    def ensure_tile_states(self) -> tuple[MontageTileState, ...]:
        if (
            int(self._tile_states_cached_revision) == int(self.tile_state_revision)
            and len(self._tile_states_cached_tuple) == len(tuple(self.plan.tiles))
        ):
            return self._tile_states_cached_tuple
        states = [MontageTileState.UNLOADED for _tile in self.plan.tiles]
        for index in tuple(self.skipped_tiles):
            index = int(index)
            if 0 <= index < len(states):
                states[index] = MontageTileState.SKIPPED
        for index in set(self.loading_tiles) | (set(self.rendered_tiles) - set(self.presented_tiles)):
            index = int(index)
            if 0 <= index < len(states) and states[index] != MontageTileState.SKIPPED:
                states[index] = MontageTileState.LOADING
        for index in set(self.presented_tiles):
            index = int(index)
            if 0 <= index < len(states):
                states[index] = MontageTileState.LOADED
        self.tile_states = states
        self._tile_states_cached_revision = int(self.tile_state_revision)
        self._tile_states_cached_tuple = tuple(self.tile_states)
        if self.canvas is not None:
            object.__setattr__(self.canvas, "tile_states", self._tile_states_cached_tuple)
        return self._tile_states_cached_tuple

    def is_complete(self) -> bool:
        return not (
            self.pending_tiles
            or self.loading_tiles
            or self.pending_completed_tiles
            or self.active_tile_requests
            or self.active_stage_requests
            or self.attached_stage_requests
            or self.stage_waiting_tiles
            or self.final_commit_pending
            or self.flush_pending
            or self.dirty_payloads
            or self.pending_payload_upserts
            or self.pending_removals
        )

    def initialize_canvas(self, canvas: MontageViewportCanvas) -> None:
        self.canvas = canvas
        self.canvas_data = canvas.data
        self.canvas_histogram_data = canvas.histogram_data
        self.canvas_rect = tuple(int(value) for value in canvas.canvas_rect)
        self.tile_states = list(canvas.tile_states)
        self.invalidate_tile_states()
        self.dirty_rects.clear()
        self.dirty_tiles.clear()

    def has_canvas(self) -> bool:
        return self.canvas is not None

    def current_canvas(self) -> MontageViewportCanvas:
        if self.canvas is None:
            raise RuntimeError("montage session has no canvas")
        if self.tile_states:
            object.__setattr__(self.canvas, "tile_states", tuple(self.tile_states))
        return self.canvas

    def mark_tile_state(self, tile: MontageTile, state: MontageTileState) -> None:
        index = int(tile.montage_index)
        if not self.tile_states:
            self.tile_states = [MontageTileState.UNLOADED for _tile in self.plan.tiles]
        if 0 <= index < len(self.tile_states):
            state = MontageTileState(state)
            if self.tile_states[index] == state:
                return
            self.tile_states[index] = state
            self.invalidate_tile_states()
            if self.canvas is not None:
                object.__setattr__(self.canvas, "tile_states", tuple(self.tile_states))

    def invalidate_tile_states(self) -> None:
        self.tile_state_revision += 1
        self._tile_states_cached_revision = -1
        self._tile_states_cached_tuple = ()

    def patch_rendered_tile(self, rendered: RenderedTile) -> bool:
        if self.canvas is None:
            return False
        dirty = patch_rendered_tile_into_canvas(rendered, self.canvas)
        index = int(rendered.tile.montage_index)
        self.rendered_tiles[index] = rendered
        self.mark_presented((index,))
        if dirty is None:
            return False
        self.dirty_rects.append(tuple(int(value) for value in dirty))
        self.dirty_tiles.append(int(rendered.tile.montage_index))
        return True

    def consume_dirty_rects(self) -> tuple[tuple[int, int, int, int], ...]:
        rects = tuple(self.dirty_rects)
        self.dirty_rects.clear()
        return rects

    def consume_dirty_tiles(self) -> tuple[int, ...]:
        tiles = tuple(dict.fromkeys(int(tile) for tile in self.dirty_tiles))
        self.dirty_tiles.clear()
        return tiles

    def note_committed(self) -> None:
        self.last_commit_monotonic = monotonic()
        self.final_commit_pending = False
        self.flush_pending = False

    def enqueue_pending_tile(self, tile: MontageTile) -> bool:
        self._ensure_pending_priority_queue()
        before = len(self.pending_tiles)
        self.pending_tiles.append(tile)
        return len(self.pending_tiles) > before

    def enqueue_pending_tiles(self, tiles) -> int:
        added = 0
        for tile in tuple(tiles or ()):
            if self.enqueue_pending_tile(tile):
                added += 1
        return int(added)

    def discard_pending_tile(self, tile_or_index) -> bool:
        self._ensure_pending_priority_queue()
        return bool(self.pending_tiles.discard(tile_or_index))

    def prune_pending_tiles(self, keep: set[int] | frozenset[int]) -> int:
        self._ensure_pending_priority_queue()
        return int(self.pending_tiles.prune(keep))

    def pending_tile_numbers(self) -> tuple[int, ...]:
        self._ensure_pending_priority_queue()
        return tile_numbers(self.pending_tiles)

    def append_stage_waiting_tiles(self, key, tiles) -> int:
        waiting = self.stage_waiting_tiles.get(key)
        if not isinstance(waiting, MontageTilePriorityQueue):
            waiting = MontageTilePriorityQueue(tuple(waiting or ()), context=self._tile_priority_context())
            self.stage_waiting_tiles[key] = waiting
        before = len(waiting)
        waiting.extend(tuple(tiles or ()))
        return max(0, len(waiting) - before)

    def retarget_tile_priority(
        self,
        *,
        focus=None,
        max_items: int = 64,
        active_tiles=None,
        near_tiles=None,
    ) -> int:
        if active_tiles is None:
            active_tiles = tuple(int(tile.montage_index) for tile in self.visible_tiles)
        if near_tiles is None:
            near_tiles = tuple(
                int(tile.montage_index)
                for tile in _viewport_tiles(
                    self.plan,
                    view_range=self.view_range,
                    viewport_shape=self.viewport_shape,
                    margin_tiles=2,
                )
            )
        if focus is not None:
            self.priority_focus = focus
        context = self._tile_priority_context(
            active_tiles=active_tiles,
            near_tiles=near_tiles,
            priority_tiles=self._priority_focus_tile_numbers(),
        )
        self._ensure_pending_priority_queue(context=context)
        self.priority_retargeted_tiles = self.pending_tiles.set_context(
            context,
            max_items=max(1, int(max_items)),
        )
        remaining = max(0, int(max_items) - int(self.priority_retargeted_tiles))
        for waiting in tuple(self.stage_waiting_tiles.values()):
            if remaining <= 0:
                break
            if hasattr(waiting, "set_context"):
                remaining -= int(waiting.set_context(context, max_items=remaining))
        return int(self.priority_retargeted_tiles)

    def _tile_priority_context(self, *, active_tiles=None, near_tiles=None, priority_tiles=None) -> TilePriorityContext:
        if active_tiles is None:
            active_tiles = tuple(int(tile.montage_index) for tile in self.visible_tiles)
        if near_tiles is None:
            try:
                near_tiles = self._near_tile_numbers(margin_tiles=2)
            except Exception:
                near_tiles = ()
        return TilePriorityContext.from_tiles(
            view_range=self.view_range,
            focus=self.priority_focus,
            visible_tiles=active_tiles,
            near_tiles=near_tiles,
            priority_tiles=priority_tiles or (),
        )

    def _near_tile_numbers(self, *, margin_tiles: int) -> tuple[int, ...]:
        key = (
            id(self.plan),
            _view_range_cache_key(self.view_range),
            tuple(int(value) for value in tuple(self.viewport_shape or ())),
            int(margin_tiles),
        )
        if key == self._near_tile_numbers_cache_key:
            return self._near_tile_numbers_cache
        tiles = tuple(
            int(tile.montage_index)
            for tile in _viewport_tiles(
                self.plan,
                view_range=self.view_range,
                viewport_shape=self.viewport_shape,
                margin_tiles=int(margin_tiles),
            )
        )
        self._near_tile_numbers_cache_key = key
        self._near_tile_numbers_cache = tiles
        return tiles

    def _priority_focus_tile_numbers(self) -> tuple[int, ...]:
        focus = self.priority_focus
        if focus is None:
            return ()
        try:
            x = float(focus[0])
            y = float(focus[1])
            geometry = self.plan.geometry
            tile_width = int(geometry.tile_width)
            tile_height = int(geometry.tile_height)
            gap = max(0, int(geometry.gap))
            stride_x = max(1, tile_width + gap)
            stride_y = max(1, tile_height + gap)
            col = int(x // stride_x)
            row = int(y // stride_y)
            local_x = x - col * stride_x
            local_y = y - row * stride_y
            if local_x < 0 or local_y < 0 or local_x >= tile_width or local_y >= tile_height:
                return ()
            if col < 0 or row < 0 or col >= int(geometry.columns):
                return ()
            index = row * int(geometry.columns) + col
            if 0 <= index < len(tuple(self.plan.tiles)):
                return (int(index),)
        except Exception:
            return ()
        return ()

    def _ensure_pending_priority_queue(self, *, context: TilePriorityContext | None = None) -> None:
        if isinstance(self.pending_tiles, MontageTilePriorityQueue):
            return
        self.pending_tiles = MontageTilePriorityQueue(
            tuple(self.pending_tiles or ()),
            context=context or self._tile_priority_context(),
        )


def _base_source_id(source_id) -> object:
    if isinstance(source_id, tuple) and len(source_id) >= 3 and source_id[1] == "texture_kind":
        return source_id[0]
    if isinstance(source_id, tuple) and "texture_kind" in source_id:
        marker = source_id.index("texture_kind")
        prefix = source_id[:marker]
        if not prefix:
            return None
        return prefix[0] if len(prefix) == 1 else prefix
    return source_id


def _array_content_token(array) -> tuple[object, ...]:
    values = np.asarray(array)
    shape = tuple(int(value) for value in values.shape)
    dtype = values.dtype.str
    return shape, dtype, id(values)


def _view_range_cache_key(view_range) -> tuple[tuple[float, ...], ...] | tuple[object, ...]:
    try:
        return tuple(tuple(float(value) for value in tuple(axis)) for axis in tuple(view_range or ()))
    except Exception:
        return (repr(view_range),)
