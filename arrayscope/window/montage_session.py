"""Qt-free state for progressive montage rendering."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from time import monotonic

import numpy as np

from arrayscope.display.lod import (
    LOD_POLICY_NATIVE_ONLY,
    LodInfo,
    LodPolicyDecision,
    native_lod_policy,
)
from arrayscope.display.pyramid import PyramidLevelKey
from arrayscope.display.montage import (
    MontagePlan,
    MontageTile,
    MontageTileState,
    RenderedTile,
    montage_rect_for_viewport,
)
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.display.model.frame import DisplayTilePayload, TileCommitReport, TilePresentationDelta, TilePresentationState
from arrayscope.display.model.level_convergence import ProgressiveTileLevelConvergence, UniformLevelConvergence
from arrayscope.display.model.presentation_generation import (
    PresentationGenerationSnapshot as LevelPresentationSnapshot,
    PresentationGenerationTracker,
    levels_match,
)
from arrayscope.display.model.tile_admission import TileAdmissionQueue
from arrayscope.operations.stage_fanin import StageFanInState
from arrayscope.presentation import TileLifecycle
from arrayscope.display.model.tile_priority import MontageTilePriorityQueue, TilePriorityContext, tile_numbers
from arrayscope.window import montage_lod
from arrayscope.window.montage_lod import (  # noqa: F401  (re-exports; canonical home is montage_lod)
    LodMaterializationRequest,
    admit_ingest_reduction,
    admit_preview_reduction,
    pyramid_key_for_rendered,
    texture_source_for_rendered,
    viewport_identity as _viewport_identity,
)


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


def _payload_residency_key(payload: DisplayTilePayload) -> tuple[object, object, object, object]:
    # This is the scheduler-side version of the backend resident key: enough
    # identity to tell a cheap tile retarget from new pixel work, without
    # exposing PyQtGraph ImageItems or VisPy atlas slots to the session.
    return (
        payload.source_id,
        id(payload.image),
        None if payload.histogram_data is None else id(payload.histogram_data),
        _shader_mapping_key(payload.shader_mapping),
    )


def _resident_retarget_upsert_tiles(
    upserts: dict[int, DisplayTilePayload],
    previous_payloads: dict[int, DisplayTilePayload],
) -> set[int]:
    if not upserts or not previous_payloads:
        return set()
    previous_tiles_by_key: dict[tuple[object, object, object, object], set[int]] = {}
    for tile, payload in previous_payloads.items():
        previous_tiles_by_key.setdefault(_payload_residency_key(payload), set()).add(int(tile))
    return {
        int(tile)
        for tile, payload in upserts.items()
        if any(previous_tile != int(tile) for previous_tile in previous_tiles_by_key.get(_payload_residency_key(payload), set()))
    }


class LifecycleRenderedTiles(dict):
    """``rendered_tiles`` with the machine's semantic axis kept authoritative.

    ADR 0051 P2: every write is an event.  Production code routes through
    ``mark_materialized`` (which fires ``evaluation_completed`` itself), but
    fixtures and repair paths that assign ``session.rendered_tiles[i]``
    directly stay correct because the collection *is* the event source —
    a result arriving marks the tile ``EVALUATED``, a result leaving demotes
    it.  This is what lets park eligibility read the semantic axis instead
    of the ``parkable_tiles`` crutch.
    """

    def __init__(self, lifecycle, initial=None):
        super().__init__()
        self._lifecycle = lifecycle
        if initial:
            self.update(initial)

    def __setitem__(self, key, value):
        index = int(key)
        super().__setitem__(index, value)
        self._lifecycle.evaluation_completed(index)

    def __delitem__(self, key):
        index = int(key)
        super().__delitem__(index)
        self._lifecycle.evaluation_dropped(index)

    def pop(self, key, *default):
        index = int(key)
        present = index in self
        result = super().pop(index, *default)
        if present:
            self._lifecycle.evaluation_dropped(index)
        return result

    def clear(self):
        indices = tuple(self.keys())
        super().clear()
        for index in indices:
            self._lifecycle.evaluation_dropped(index)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        index = int(key)
        if index not in self:
            self[index] = default
        return self[index]


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
    montage_axis: int | None
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
    stage_fan_in: StageFanInState = field(default_factory=StageFanInState)
    tile_states: list[MontageTileState] = field(default_factory=list)
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
    level_scan_cursor: int = 0
    level_scan_remaining_tiles: int = 0
    pending_refined_level_tiles: deque[RenderedTile] = field(default_factory=deque)
    pending_refined_level_sources: set[int] = field(default_factory=set)
    pending_completed_tiles: deque[tuple[MontageTile, object]] = field(default_factory=deque)
    tile_compute_cache_hits: int = 0
    tile_compute_stage_backed: int = 0
    tile_compute_direct: int = 0
    tile_compute_waiting_for_stage: int = 0
    tile_compute_stage_backed_ms: float = 0.0
    tile_compute_direct_ms: float = 0.0
    tile_compute_stage_backed_max_ms: float = 0.0
    tile_compute_direct_max_ms: float = 0.0
    lead_direct_tiles: int = 0
    stage_backed_tiles_pending: int = 0
    retained_stage_index: int | None = None
    retained_stage_decision: str = ""
    repeated_expensive_stage_per_tile: bool = False
    frame_plan: object | None = None
    tile_source_ids: dict[int, object] = field(default_factory=dict)
    display_tile_payloads: dict[int, DisplayTilePayload] = field(default_factory=dict)
    dirty_payloads: OrderedDict[int, None] = field(default_factory=OrderedDict)
    pending_payload_upserts: OrderedDict[int, None] = field(default_factory=OrderedDict)
    pending_removals: set[int] = field(default_factory=set)
    visible_tile_numbers: frozenset[int] = field(default_factory=frozenset)
    level_generation: PresentationGenerationTracker = field(default_factory=PresentationGenerationTracker)
    _level_update_pending: bool = False
    tile_presentation_state: TilePresentationState = field(default_factory=TilePresentationState)
    structure_revision: int = 0
    payload_revision: int = 0
    visibility_revision: int = 0
    histogram_revision: int = 0
    viewport_revision: int = 0
    tile_state_revision: int = 0
    priority_focus: tuple[float, float] | None = None
    priority_retargeted_tiles: int = 0
    # First presented tile numbers in presentation order (capped): makes
    # priority-order violations observable in diagnostics logs.
    presented_order: list[int] = field(default_factory=list)
    presentation_geometry_changed: bool = False
    _layout_geometry_changed_pending: bool = False
    lod_policy_decision: LodPolicyDecision = field(
        default_factory=lambda: native_lod_policy(None, (1, 1), (1, 1))
    )
    # ADR 0050: "native-only" keeps production behavior; "resident" presents
    # the closest pyramid level that is actually materialized and resident.
    lod_policy_mode: str = LOD_POLICY_NATIVE_ONLY
    lod_pyramid: object | None = None
    # (tile_number, PyramidLevelKey, source array) triples for the renderer to
    # schedule as background materializations.  Filled only under the
    # "resident" policy after a singleflight claim on the pyramid cache.
    pending_lod_requests: list = field(default_factory=list)
    lod_materializations_completed: int = 0
    acknowledged_source_ids: set = field(default_factory=set)
    lod_floor_presentations: int = 0
    lod_target_revision: int = 0
    # ADR 0051: single owner of per-tile lifecycle state.  The presentation
    # axis (emit-once / park / re-arm and acknowledged presentation) is
    # authoritative here; the semantic axis is mirrored from the legacy
    # collections during the migration (parity in diagnostics).
    lifecycle: TileLifecycle = field(default_factory=TileLifecycle)
    # Ground truth from the last backend report: tile -> the payload identity
    # its drawn slot ACTUALLY holds (ADR 0051 rule 1).  Presentation
    # convergence compares against this, never against session bookkeeping.
    last_presented_identities: dict = field(default_factory=dict)
    # Window-agnostic texel identity base for pyramid floors/previews
    # (montage_tile_semantic_key); falls back to session key when unset.
    semantic_key: object = None
    # Bounded retry state for identity reconciliation:
    # tile -> ((shown_identity, wanted_identity), attempts).
    _reconcile_attempts: dict = field(default_factory=dict)
    lod_preview_pyramid: object | None = None
    lod_preview_level: int = 0
    # ADR 0050 WP1: a display-LOD level swap rebuilds the payload wrapper but
    # must carry the tile's finest already-computed semantic stats forward
    # unchanged.  `cross_level_reuses` counts swaps that reused the retained
    # native histogram/level objects; `recomputes` counts swaps that minted
    # new stat objects for unchanged native content (must stay 0).
    lod_stats_cross_level_reuses: int = 0
    lod_stats_recomputes: int = 0
    # ADR 0050 WP2: demanded pyramid levels are derived from the finest
    # already-resident coarser level when factors divide evenly, instead of
    # re-reducing from the native plane.
    lod_cross_level_reductions: int = 0
    _last_active_tiles: tuple[int, ...] = ()
    _last_planned_tiles: tuple[int, ...] = ()
    _last_near_tiles: tuple[int, ...] = ()
    _last_viewport_identity: tuple[object, ...] | None = None
    _lod_refresh_viewport_identity: tuple[object, ...] | None = None
    _near_tile_numbers_cache_key: tuple[object, ...] | None = None
    _priority_context: TilePriorityContext | None = None
    _near_tile_numbers_cache: tuple[int, ...] = ()
    _tile_states_cached_revision: int = -1
    _tile_states_cached_tuple: tuple[MontageTileState, ...] = ()

    @property
    def parked_dirty_payloads(self) -> frozenset[int]:
        """Read-only view; the lifecycle machine owns parking (ADR 0051)."""

        return self.lifecycle.parked_tiles

    @property
    def level_revision(self) -> int:
        return int(self.level_generation.revision)

    @level_revision.setter
    def level_revision(self, value: int) -> None:
        self.level_generation.revision = int(value)

    def __post_init__(self) -> None:
        # These queues are drained throughout progressive rendering.  The
        # visible tile queue is indexed so viewport/hover retargeting updates
        # metadata instead of sorting inside high-frequency callbacks.
        pending = tuple(self.pending_tiles or ())
        self.pending_tiles = MontageTilePriorityQueue(
            pending,
            context_provider=self._tile_priority_context,
        )
        self.pending_level_tiles = deque(self.pending_level_tiles)
        self.pending_level_sources = {
            int(source) for source in (self.pending_level_sources or ())
        } or {int(item.tile.source_index) for item in self.pending_level_tiles}
        self.pending_refined_level_tiles = deque(self.pending_refined_level_tiles)
        self.pending_refined_level_sources = {
            int(source) for source in (self.pending_refined_level_sources or ())
        } or {int(item.tile.source_index) for item in self.pending_refined_level_tiles}
        self.pending_completed_tiles = deque(self.pending_completed_tiles)
        self.ensure_stage_waiting_priority_queues()
        self.visible_tile_numbers = frozenset(int(tile.montage_index) for tile in tuple(self.visible_tiles or ()))
        self._selected_lod_factor()
        self.update_level_presentation_scope()
        # ADR 0051: seed the lifecycle machine so its semantic axis matches a
        # session built from cached results (rendered == evaluated).  P2:
        # rendered_tiles becomes the event-routing collection, so every later
        # write (including direct fixture assignment) keeps the semantic axis
        # authoritative.
        self.lifecycle.plan_applied(int(tile.montage_index) for tile in tuple(self.visible_tiles or ()))
        for index in sorted(int(tile) for tile in self.skipped_tiles):
            self.lifecycle.tile_skipped(int(index))
        self.rendered_tiles = LifecycleRenderedTiles(self.lifecycle, self.rendered_tiles)
        for index in sorted(int(tile) for tile in self.rendered_tiles):
            self.dirty_payloads.setdefault(int(index), None)

    def is_tile_loaded(self, tile) -> bool:
        return int(tile.montage_index) in self.rendered_tiles

    def retarget_viewport(
        self,
        *,
        view_range,
        viewport_shape: tuple[int, int],
        plan=None,
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

        self.viewport_shape = (max(1, int(viewport_shape[0])), max(1, int(viewport_shape[1])))
        viewport_identity = _viewport_identity(view_range, self.viewport_shape)
        viewport_changed = viewport_identity != self._last_viewport_identity
        self.view_range = view_range
        layout_changed = False
        if plan is not None:
            layout_changed = getattr(plan, "geometry", None) != getattr(self.plan, "geometry", None)
            self.plan = plan
            if layout_changed:
                self._layout_geometry_changed_pending = True
                self._remap_queued_tiles_to_plan()
        self.priority_focus = priority_focus
        self._selected_lod_factor()
        plan_tiles = tuple(getattr(self.plan, "tiles", ()) or ())
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
        planned_numbers = tuple(int(tile.montage_index) for tile in plan_tiles)
        previous_visible_numbers = tuple(int(tile.montage_index) for tile in self.visible_tiles)
        presentation_changed = (
            layout_changed
            or viewport_changed
            or active_numbers != previous_visible_numbers
            or near_numbers != self._last_near_tiles
            or planned_numbers != self._last_planned_tiles
        )
        self.visible_tiles = active
        self.visible_tile_numbers = frozenset(active_numbers)
        if presentation_changed:
            self.update_level_presentation_scope()
        self.priority_retargeted_tiles = self.retarget_tile_priority(
            focus=priority_focus,
            max_items=max(1, int(priority_retarget_limit)),
            active_tiles=active_numbers,
            near_tiles=near_numbers,
        )
        return additions, bool(presentation_changed)

    def retarget_index_window(
        self,
        *,
        session_id: int,
        key,
        semantic_key,
        level_key,
        render_generation: int,
        view_state,
        plan: MontagePlan,
        frame_plan,
        all_indices,
        new_source_ids: dict[int, object],
        cached_tiles: dict[int, RenderedTile],
        visible_tiles: tuple[MontageTile, ...],
    ) -> dict:
        """Re-key this session to a new index window without a rebirth.

        ADR 0051 P2 (session-rebirth cost): when only the montage index
        window changes — layout geometry, viewport, document, and every
        presentation input identical — the session object survives.  The
        lifecycle machine, the backend acknowledgement state
        (``tile_presentation_state``, ``last_presented_identities``), and the
        currently drawn payloads all stay; only the semantic mapping
        tile -> source index moves.  Every tile whose source changed goes
        through the ordinary per-tile seams (``mark_materialized`` for cache
        hits, demotion for misses), so the standing budgeted commit/flush
        machinery converges the screen exactly as it does for evaluation
        results — floors first, exact content as it lands.

        Identity semantics match a rebirth: ``session_id`` and ``key`` are
        new, so in-flight completions for the old window are parked into the
        reusable cache by the existing current-session gates, and stale
        commit reports acknowledge nothing (delta_key binding).

        The caller has already verified eligibility and released undrained
        pyramid claims; stage planning state is reset by the caller
        (deferred-planning path).  Returns remap statistics plus the drained
        ``stale_completions`` for the caller to store.
        """

        old_source_ids = dict(self.tile_source_ids)
        self.session_id = int(session_id)
        self.key = key
        self.semantic_key = semantic_key
        self.level_key = level_key
        self.render_generation = int(render_generation)
        self.view_state = view_state
        self.plan = plan
        self.frame_plan = frame_plan
        self.level_expected_indices = tuple(int(index) for index in all_indices)
        # In-flight work for the old window is superseded wholesale (same
        # semantics as a rebirth): completions route to the reusable cache
        # via the session_id gate, and the deferred-planning pass reschedules
        # everything still missing.
        self.active_tile_requests.clear()
        stale_completions = tuple(self.pending_completed_tiles)
        self.pending_completed_tiles.clear()
        self.pending_level_tiles.clear()
        self.pending_level_sources.clear()
        self.pending_refined_level_tiles.clear()
        self.pending_refined_level_sources.clear()
        self.pending_tiles.prune(frozenset())
        hits = misses = unchanged = 0
        for tile in tuple(getattr(plan, "tiles", ()) or ()):
            index = int(tile.montage_index)
            new_source = new_source_ids.get(index)
            if (
                new_source is not None
                and new_source == old_source_ids.get(index)
                and index in self.rendered_tiles
            ):
                unchanged += 1
                continue
            rendered = cached_tiles.get(index)
            if rendered is not None:
                self.mark_materialized(rendered)
                hits += 1
            else:
                # Demote: the drawn slot keeps its (now stale) content until
                # the floor or a fresh evaluation replaces it — never blank.
                self.rendered_tiles.pop(index, None)
                self.tile_source_ids.pop(index, None)
                self.display_tile_payloads.pop(index, None)
                self.level_generation.forget_tile(index)
                self.presented_tiles.discard(index)
                self.skipped_tiles.discard(index)
                self.dirty_payloads[index] = None
                self.loading_tiles.add(index)
                if 0 <= index < len(plan.tiles):
                    self.mark_tile_state(plan.tiles[index], MontageTileState.LOADING)
                misses += 1
        self.visible_tiles = tuple(visible_tiles)
        self.visible_tile_numbers = frozenset(
            int(tile.montage_index) for tile in self.visible_tiles
        )
        self._selected_lod_factor()
        self.update_level_presentation_scope()
        # Re-decide LOD demands for the new sources; resident floors from the
        # window-agnostic pyramid identity re-present immediately.
        self.refresh_lod_for_viewport()
        return {
            "hits": int(hits),
            "misses": int(misses),
            "unchanged": int(unchanged),
            "stale_completions": stale_completions,
        }

    def update_level_presentation_scope(self) -> None:
        if not self.display_tile_payloads and not self.presented_tiles:
            return
        active = frozenset(
            int(tile.montage_index)
            for tile in tuple(self.visible_tiles)
            if int(tile.montage_index) in self.display_tile_payloads
            and int(tile.montage_index) in self.presented_tiles
            # Preview (floor) payloads carry no semantic level evidence and
            # can never converge; letting them into the scope kept the level
            # target permanently stale, which forced every commit through
            # the full rebuild path in a timer loop at idle.
            and str(getattr(self.display_tile_payloads[int(tile.montage_index)], "quality", "exact")) == "exact"
        )
        self.level_generation.set_active_tiles(active)
        if self.level_generation.target_levels is not None:
            snapshot = self.level_generation.snapshot(
                pending_upserts=tuple(self.pending_payload_upserts),
                active_tile_count=len(self.visible_tile_numbers),
            )
            self._level_update_pending = not bool(snapshot.settled)

    def begin_level_presentation_update(self, levels) -> bool:
        """Start or continue a progressive level generation.

        Histogram drags emit an immediate preview followed by a finish signal
        carrying the same numeric levels.  Treat that finish signal as a
        request to drain the existing generation, not as a new generation.
        Reissuing an already-settled target is a no-op.

        Returns ``True`` only while at least one currently presented tile still
        needs the requested levels.  A new target is still retained when no
        tile is active so subsequently materialized tiles inherit it.
        """

        needs_work = ProgressiveTileLevelConvergence().begin(
            self.level_generation,
            levels,
            source=self.applied_level_source,
            active_tiles=self.level_generation.active_tiles,
        )
        self._level_update_pending = bool(needs_work)
        return bool(needs_work)

    def level_presentation_snapshot(self) -> LevelPresentationSnapshot:
        """Return the current semantic convergence state for the level target."""

        self.update_level_presentation_scope()
        return self.level_generation.snapshot(
            pending_upserts=tuple(self.pending_payload_upserts),
            active_tile_count=len(self.visible_tile_numbers),
        )

    def has_pending_level_update(self) -> bool:
        return bool(self._level_update_pending and not self.level_presentation_snapshot().settled)

    def set_level_update_pending(self, pending: bool) -> None:
        self._level_update_pending = bool(pending)

    def acknowledge_uniform_level_presentation(self, levels) -> None:
        """Accept one shader-level update for every active tiled surface.

        A shader backend changes one shared presentation uniform rather than
        redrawing individual payloads.  Record that as a single semantic
        acknowledgement while keeping per-tile values available for the same
        convergence diagnostics used by CPU-windowed backends.
        """

        UniformLevelConvergence().begin(
            self.level_generation,
            levels,
            source=self.applied_level_source,
            active_tiles=self.level_generation.active_tiles,
        )
        self._level_update_pending = False

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
        self.level_generation.forget_tile(index)
        self.dirty_payloads[index] = None
        self.active_tile_requests.discard(index)
        self.skipped_tiles.discard(index)
        self.discard_pending_tile(index)
        self.presented_tiles.discard(index)
        self.loading_tiles.add(index)
        self.lifecycle.evaluation_completed(index)
        self.mark_tile_state(rendered.tile, MontageTileState.LOADING)

    def mark_presented(self, tile_numbers) -> None:
        # Collect level-scope additions and apply them once at the end:
        # extending the frozenset per presented tile makes a full-montage
        # commit O(n^2) in the tile count.
        level_scope_additions: list[int] = []
        confirmed: list[int] = []
        for tile_number in tuple(tile_numbers or ()):
            index = int(tile_number)
            if index not in self.rendered_tiles:
                continue
            confirmed.append(index)
            if index not in self.presented_tiles and len(self.presented_order) < 64:
                self.presented_order.append(index)
            self.presented_tiles.add(index)
            if index in self.visible_tile_numbers and index in self.display_tile_payloads:
                level_scope_additions.append(index)
            self.loading_tiles.discard(index)
            self.skipped_tiles.discard(index)
            self.dirty_payloads.pop(index, None)
            self.pending_removals.discard(index)
            if 0 <= index < len(self.plan.tiles):
                self.mark_tile_state(self.plan.tiles[index], MontageTileState.LOADED)
        if confirmed:
            # ADR 0051 rule 1: the report's presented set is backend
            # acknowledgement, so resident-retarget commits (no upserts)
            # still reach the machine's PRESENTED state.
            self.lifecycle.presentation_confirmed(confirmed)
        if level_scope_additions:
            self.level_generation.set_active_tiles(
                (*self.level_generation.active_tiles, *level_scope_additions)
            )

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
        exact_level_data = None if getattr(rendered, "level_data", None) is None else np.asarray(rendered.level_data)
        level_stats = getattr(rendered, "level_stats", None)
        semantic = getattr(rendered, "semantic_data", None)
        semantic = exact_image if semantic is None else np.asarray(semantic)
        texture_data, texture_histogram, lod = self._texture_for_rendered_tile(rendered)
        source_id = self._payload_source_id(
            base_source_id,
            texture_kind=texture_kind,
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
            and previous.level_data is exact_level_data
            and previous.level_stats is level_stats
            and _shader_mapping_key(previous.shader_mapping) == _shader_mapping_key(mapping)
        ):
            return previous
        if (
            previous is not None
            and _base_source_id(previous.source_id) == base_source_id
            and previous.image is exact_image
            and int(getattr(getattr(previous, "lod", None), "level", 0) or 0) != int(lod.level)
        ):
            # Same native content presented at a different display-LOD level:
            # the level swap must be invisible to the histogram/level system.
            if (
                previous.histogram_data is exact_histogram
                and previous.level_data is exact_level_data
                and previous.level_stats is level_stats
            ):
                self.lod_stats_cross_level_reuses += 1
            else:
                self.lod_stats_recomputes += 1
        payload = DisplayTilePayload(
            tile_number=tile_number,
            source_index=int(rendered.tile.source_index),
            image=exact_image,
            histogram_data=exact_histogram,
            source_id=source_id,
            texture_data=texture_data,
            texture_kind=texture_kind,
            semantic_data=semantic,
            semantic_histogram_data=exact_histogram if texture_histogram is None else texture_histogram,
            source_shape=tuple(int(value) for value in exact_image.shape[:2]),
            lod=lod,
            shader_mapping=mapping,
            level_data=exact_level_data,
            level_stats=level_stats,
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
        # ADR 0050: a payload whose base (semantic) identity matches but whose
        # LOD level differs is still this tile's content, resident on the
        # backend.  Seeding it keeps the tile presented at the old level; the
        # LOD refresh then converges it through an ordinary identity swap.
        # Without this, a level change across sessions read as a black or
        # placeholder tile until fresh payload work committed.
        by_base: dict[object, DisplayTilePayload] = {}
        if self._resident_lod_active():
            for payload in dict(previous_payloads).values():
                base = _base_source_id(payload.source_id)
                if base is not None:
                    by_base.setdefault(base, payload)
        seeded_state = dict(getattr(self.tile_presentation_state, "payloads", {}) or {})
        changed_state = False
        for tile_number, rendered in self.rendered_tiles.items():
            tile_number = int(tile_number)
            previous = self.display_tile_payloads.get(tile_number)
            owns_committed_presentation = tile_number in self.presented_tiles
            if previous is None:
                base_source_id = source_ids.get(tile_number, ("rendered_tile", tile_number, id(rendered.image)))
                texture_data, _texture_histogram, lod = self._texture_for_rendered_tile(rendered)
                source_id = self._payload_source_id(
                    base_source_id,
                    texture_kind=getattr(rendered, "texture_kind", None),
                    lod=lod,
                    texture_data=texture_data,
                )
                previous = by_source.get(source_id)
                if previous is None:
                    previous = by_base.get(base_source_id)
                owns_committed_presentation = previous is not None
            if previous is None:
                continue
            retargeted = int(previous.tile_number) != tile_number
            if not retargeted and int(previous.source_index) == int(rendered.tile.source_index):
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
                self.pending_payload_upserts.pop(tile_number, None)
                if retargeted:
                    self.pending_payload_upserts[tile_number] = None
                changed_state = True
        if changed_state:
            self.tile_presentation_state = TilePresentationState(
                seeded_state,
                revision=int(getattr(self.tile_presentation_state, "revision", 0)),
            )
            self.presented_tiles.update(int(tile) for tile in seeded_state)
            self.invalidate_tile_states()

    def _payload_source_id(self, base_source_id, *, texture_kind, lod: LodInfo, texture_data) -> tuple[object, ...]:
        prefix = tuple(base_source_id) if isinstance(base_source_id, tuple) else (base_source_id,)
        return (
            *prefix,
            "texture_kind",
            None if texture_kind is None else getattr(texture_kind, "value", texture_kind),
            "lod",
            int(lod.factor),
            int(lod.level),
            int(lod.gutter),
            "content",
            _array_content_token(texture_data),
        )

    def _selected_lod_factor(self) -> int:
        return montage_lod.selected_lod_factor(self)

    def _resident_lod_active(self) -> bool:
        return montage_lod.resident_lod_active(self)

    def presented_lod_summary(self) -> tuple[int, int, tuple[int, int]]:
        """Plurality-presented (level, factor, factor_xy); see :mod:`montage_lod`."""

        return montage_lod.presented_lod_summary(self)

    def ingest_lod_demand(self) -> object | None:
        """Demand snapshot for worker-side reduce-at-ingest; see :mod:`montage_lod`."""

        return montage_lod.ingest_lod_demand(self)

    def refresh_lod_for_viewport(self) -> bool:
        """Re-evaluate LOD demand after a camera-only retarget; see :mod:`montage_lod`."""

        return montage_lod.refresh_lod_for_viewport(self)

    def _session_resident_levels(self, previous_factor: int) -> tuple[int, ...]:
        return montage_lod.session_resident_levels(self, previous_factor)

    def tile_semantic_source_id(self, source_index) -> tuple[object, ...]:
        """Semantic content identity of one montage tile (ADR 0050).

        Keyed by ``semantic_key`` (montage_tile_semantic_key), which is
        window-agnostic: equal keys mean equal source texels for a given
        source index across rendered-tile rebuilds, session recreations, AND
        index-window changes — floors/previews computed under one window
        present instantly under another (field defect 2026-07-05).  Falls
        back to the session key for sessions built without one (tests).
        """

        base = self.semantic_key if self.semantic_key is not None else self.key
        return ("montage-tile", base, int(source_index))

    def _pyramid_key_for(self, rendered: RenderedTile, *, demand, level: int) -> PyramidLevelKey:
        return montage_lod.pyramid_key_for(self, rendered, demand=demand, level=level)

    def _lod_materialization_request(
        self,
        rendered: RenderedTile,
        *,
        demand,
        level: int,
        key: PyramidLevelKey,
        native_source: np.ndarray | None = None,
    ) -> LodMaterializationRequest:
        return montage_lod.plan_materialization(
            self, rendered, demand=demand, level=level, key=key, native_source=native_source
        )

    def _floor_component_tags(self) -> tuple[str, ...]:
        return montage_lod.floor_component_tags(self)

    def _best_floor_key(self, source_index: int):
        return montage_lod.best_floor_key(self, source_index)

    def _floor_can_progress(self, tile_number: int, tile=None) -> bool:
        return montage_lod.floor_can_progress(self, tile_number, tile=tile)

    def _ensure_floor_payloads(self, tile_numbers) -> None:
        return montage_lod.ensure_floor_payloads(self, tile_numbers)

    def _texture_source_for(self, rendered: RenderedTile) -> tuple[np.ndarray, np.ndarray | None, TexturePlaneKind | None]:
        return texture_source_for_rendered(rendered)

    def _resident_texture_for_rendered_tile(
        self,
        rendered: RenderedTile,
        *,
        source: np.ndarray,
        histogram: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, LodInfo]:
        return montage_lod.resident_texture_for_rendered_tile(self, rendered, source=source, histogram=histogram)

    def _texture_for_rendered_tile(self, rendered: RenderedTile) -> tuple[np.ndarray, np.ndarray | None, LodInfo]:
        source, histogram, _texture_kind = self._texture_source_for(rendered)
        if self._resident_lod_active():
            return self._resident_texture_for_rendered_tile(rendered, source=source, histogram=histogram)
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
        upsert_cost_fn=None,
    ) -> tuple[TilePresentationState, TilePresentationDelta]:
        source_ids = dict(source_ids or {})
        previous_state = self.tile_presentation_state
        previous_payloads = dict(previous_state.payloads)
        loaded = {int(index) for index in self.rendered_tiles}
        planned_numbers = {
            int(tile.montage_index)
            for tile in tuple(self.visible_tiles)
            if int(tile.montage_index) not in self.skipped_tiles
        }
        for stale in tuple(self.display_tile_payloads):
            if int(stale) not in loaded:
                payload = self.display_tile_payloads.get(int(stale))
                if (
                    payload is not None
                    and str(getattr(payload, "quality", "exact")) == "preview"
                    and int(stale) in planned_numbers
                ):
                    # Presentation floor (ADR 0050): a planned tile presenting
                    # a resident coarser level keeps it until its exact
                    # replacement is acknowledged; only leaving the plan
                    # removes it.
                    continue
                self.display_tile_payloads.pop(int(stale), None)
                self.pending_removals.add(int(stale))
                self.dirty_payloads.pop(int(stale), None)
                self.pending_payload_upserts.pop(int(stale), None)
                self.level_generation.forget_tile(int(stale))
        lod_factor = self._selected_lod_factor()
        current_loaded = set(self.rendered_tiles)
        planned = tuple(
            int(tile.montage_index)
            for tile in tuple(self.visible_tiles)
            if int(tile.montage_index) not in self.skipped_tiles
        )
        active = tuple(int(tile) for tile in planned if int(tile) in current_loaded)
        stale_level_tiles = ()
        if self.has_pending_level_update():
            # Filter before prioritizing: ordering every active tile per commit
            # makes the drain of a large stale backlog O(n^2) in commits.
            stale_candidates = tuple(
                int(tile)
                for tile in active
                if int(tile) in self.display_tile_payloads
                and int(tile) in previous_payloads
                and str(getattr(self.display_tile_payloads[int(tile)], "quality", "exact")) == "exact"
                and not self._tile_matches_current_level_target(int(tile), self.level_generation.target_levels)
            )
            stale_level_tiles = self._prioritized_tile_numbers(stale_candidates)
        # Parked dirty entries re-arm when their tile enters the active
        # scope (see acknowledge_tile_presentation: a non-active upsert the
        # backend declines parks instead of re-arming, or finalization would
        # retry an unacceptable upsert forever).
        # Drawn-identity reconciliation (ADR 0051 rule 1; field defects
        # 2026-07-05): a drawn tile whose ON-SCREEN identity differs from the
        # payload the session now holds is visibly stale no matter what any
        # bookkeeping says.  The backend's reported slot identities are the
        # ground truth (session acknowledgement records were the liar in
        # every wedge so far); the acknowledged state is the fallback for
        # backends that do not report identities.  Stale tiles re-present
        # and join the active scope so the viewport-scoped backend accepts
        # them; retries are bounded per (shown, wanted) identity pair so a
        # backend that cannot converge (e.g. atlas capacity) never turns
        # this into an idle commit loop.
        stale_drawn: list[int] = []
        backend_identities = dict(getattr(self, "last_presented_identities", None) or {})
        if backend_identities:
            for tile_number, shown_identity in backend_identities.items():
                current = self.display_tile_payloads.get(int(tile_number))
                if current is None or current.source_id == shown_identity:
                    self._reconcile_attempts.pop(int(tile_number), None)
                    continue
                rec = self.lifecycle.peek(int(tile_number))
                if rec is not None and (current.source_id, shown_identity) in rec.resigned:
                    # P2: the machine resigned exactly this (wanted, shown)
                    # pair after bounded identity rejections — the backend
                    # would not converge; re-presenting would reopen the loop
                    # it bounded.  Any OTHER mismatch (e.g. payload moved on
                    # without an emit, or a rebuilt session inheriting stale
                    # slots) is a legitimate repair and proceeds below.
                    self._reconcile_attempts.pop(int(tile_number), None)
                    continue
                pair = (shown_identity, current.source_id)
                prior_pair, attempts = self._reconcile_attempts.get(int(tile_number), (None, 0))
                if prior_pair != pair:
                    attempts = 0
                if attempts >= 3:
                    continue
                self._reconcile_attempts[int(tile_number)] = (pair, attempts + 1)
                stale_drawn.append(int(tile_number))
        else:
            for tile_number, acknowledged in previous_payloads.items():
                current = self.display_tile_payloads.get(int(tile_number))
                if current is None or current is acknowledged:
                    continue
                if current.source_id != acknowledged.source_id:
                    stale_drawn.append(int(tile_number))
        if stale_drawn:
            for tile_number in sorted(stale_drawn):
                self.dirty_payloads[int(tile_number)] = None
                # Force the upsert: session bookkeeping may consider this
                # identity already presented (that bookkeeping is exactly
                # what the backend truth contradicts).
                self.pending_payload_upserts[int(tile_number)] = None
            active = tuple(dict.fromkeys((*active, *sorted(stale_drawn))))
        # ADR 0051: the lifecycle machine owns park/re-arm; this is its
        # rule-3 scope event.
        active_scope = set(active)
        for tile_number in self.lifecycle.rearm_for_scope(active_scope):
            self.dirty_payloads[int(tile_number)] = None
        dirty_payload_tiles = tuple(
            dict.fromkeys(
                (
                    *(int(tile) for tile in self.dirty_payloads),
                    *(int(tile) for tile in self.pending_payload_upserts),
                    *stale_level_tiles,
                )
            )
        )
        if max_upserts is not None or max_upsert_bytes is not None or stale_level_tiles:
            dirty_payload_tiles = self._prioritized_tile_numbers(dirty_payload_tiles)
        # Payload construction is bounded by the same admission budget that
        # caps uploads: a retarget/scrub step marks every tile dirty, and
        # building all N wrappers synchronously before admitting 4 of them
        # made the budgeted commit O(N) anyway (session-rebirth cost, ADR
        # 0051 P2).  Unbuilt tiles keep their dirty entry — the next commit
        # continues in priority order.  Slightly over-build so byte-capped
        # admission still has choices.
        build_limit = None
        if max_upserts is not None:
            build_limit = max(int(max_upserts) * 2, 8)
        built = 0
        for tile_number in dirty_payload_tiles:
            if build_limit is not None and built >= build_limit:
                break
            rendered = self.rendered_tiles.get(int(tile_number))
            if rendered is not None:
                self._ensure_display_tile_payload(int(tile_number), rendered, source_ids, lod_factor=lod_factor)
                built += 1
        self._ensure_floor_payloads(planned_numbers - set(current_loaded))
        # Progress guarantee: a dirty entry is a promise that a build can
        # produce an upsert for the tile.  Without a rendered result and
        # without a pending upsert (floor included), no build can keep that
        # promise — dropping the entry lets the commit loop settle instead
        # of rescheduling forever (100% single-core spin).  A later rendered
        # result re-dirties the tile through mark_loaded/mark_materialized.
        for tile_number in tuple(self.dirty_payloads):
            if (
                int(tile_number) not in self.rendered_tiles
                and int(tile_number) not in self.pending_payload_upserts
            ):
                self.dirty_payloads.pop(int(tile_number), None)
        current_payloads = self.display_tile_payloads
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
            force_upsert = int(tile_number) in self.pending_payload_upserts or int(tile_number) in stale_level_tiles
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

        resident_retarget_tiles = _resident_retarget_upsert_tiles(upserts, previous_payloads)
        # Level swaps re-presenting a backend-acknowledged identity are
        # residency remaps, not uploads: they bypass cold admission like
        # other resident retargets, so a burst of swaps converges in one
        # commit while governed upload limits stay untouched (ADR 0050).
        resident_retarget_tiles |= {
            int(tile)
            for tile, payload in upserts.items()
            if payload.source_id in self.acknowledged_source_ids
        }
        resident_retarget_tiles.difference_update(int(tile) for tile in stale_level_tiles)
        resident_retarget_tiles.update(
            int(tile)
            for tile, payload in upserts.items()
            if int(tile) in self.pending_payload_upserts
            and int(tile) not in stale_level_tiles
            and previous_payloads.get(int(tile)) is payload
        )
        resident_retarget_upserts = {
            int(tile): payload
            for tile, payload in upserts.items()
            if int(tile) in resident_retarget_tiles
        }
        cold_upserts = {
            int(tile): payload
            for tile, payload in upserts.items()
            if int(tile) not in resident_retarget_tiles
        }
        admission = TileAdmissionQueue(self._tile_priority_context()).admit(
            tuple(cold_upserts),
            retained=(),
            cost_fn=(
                (lambda tile: int(upsert_cost_fn(cold_upserts[int(tile)])))
                if upsert_cost_fn is not None
                else (lambda tile: int(getattr(cold_upserts[int(tile)], "nbytes", 0) or 0))
            ),
            max_items=max_upserts,
            max_bytes=max_upsert_bytes,
            deadline_ms=cold_deadline_ms,
        )
        capped_cold_upserts = {
            int(tile): cold_upserts[int(tile)]
            for tile in admission.admitted
            if int(tile) in cold_upserts
        }
        upserts = {
            int(tile): payload
            for tile, payload in upserts.items()
            if int(tile) in resident_retarget_upserts or int(tile) in capped_cold_upserts
        }
        if max_upserts is not None or max_upsert_bytes is not None:
            admitted = set(int(tile) for tile in upserts)
            admitted.update(int(tile) for tile in self.presented_tiles)
            admitted.update(int(tile) for tile in previous_payloads)
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

        # ADR 0051: record the emit (emit-once identity) with the machine.
        for tile_number, payload in upserts.items():
            self.lifecycle.upsert_emitted(int(tile_number), payload.source_id)

        base_revision = int(getattr(previous_state, "revision", 0))
        target_revision = base_revision + (1 if upserts or removals else 0)
        if upserts or removals:
            self.payload_revision += 1
        viewport_identity = _viewport_identity(self.view_range, self.viewport_shape)
        viewport_changed = viewport_identity != self._last_viewport_identity
        presentation_geometry_changed = bool(
            getattr(self, "_layout_geometry_changed_pending", False)
            or viewport_changed
            or active != self._last_active_tiles
            or planned != self._last_planned_tiles
            or near != self._last_near_tiles
        )
        self.presentation_geometry_changed = presentation_geometry_changed
        self._layout_geometry_changed_pending = False
        if active != self._last_active_tiles or planned != self._last_planned_tiles:
            self.visibility_revision += 1
        if viewport_changed or near != self._last_near_tiles:
            self.viewport_revision += 1
        if planned != self._last_planned_tiles:
            self.structure_revision += 1
        self._last_active_tiles = active
        self._last_planned_tiles = planned
        self._last_near_tiles = near
        self._last_viewport_identity = viewport_identity

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

    def acknowledge_tile_presentation(
        self,
        delta: TilePresentationDelta,
        report: TileCommitReport,
        *,
        levels: tuple[float, float] | None = None,
    ) -> TilePresentationState:
        if not isinstance(report, TileCommitReport):
            raise TypeError("tile presentation acknowledgement requires a TileCommitReport")
        if getattr(report, "presented_identities", None) is not None:
            # The pool's slot identities are the newest known truth about the
            # screen regardless of which delta the report was built for.
            self.last_presented_identities = dict(report.presented_identities)
        if not report.acknowledges(delta):
            # ADR 0051 rule 1 (field defect 2026-07-05, JSONL 112841): the
            # committer's last report belongs to an OLDER delta — this delta
            # never reached the backend (skipped/superseded commit).
            # Acknowledge nothing and park nothing: every dirty entry stays
            # armed, so the next flush re-presents through a real commit.
            return self.tile_presentation_state
        level_delta_stale = bool(
            self.has_pending_level_update()
            and dict(delta.upserts)
            and int(delta.level_revision) != int(self.level_revision)
        )
        if level_delta_stale:
            report = replace(report, stale=True, committed_upserts=())
        # ADR 0051 P2: the machine makes the acceptance decision in ONE place
        # — identity-aware (backend slot identities vs emitted identities),
        # rule-1/rule-3 park logic from its own semantic axis.  Legacy state
        # below consumes the machine's verdict via the filtered report; it
        # never re-derives acceptance from tile numbers.
        active_scope = {int(tile) for tile in tuple(getattr(delta, "active_tiles", ()) or ())}
        report_accepted = report.accepted_upserts(delta)
        machine_accepted = self.lifecycle.commit_acknowledged(
            emitted_tiles=(int(tile) for tile in tuple(delta.upserts)),
            accepted_tiles=report_accepted,
            active_scope=active_scope,
            removed_tiles=(int(tile) for tile in report.removed_tiles),
            stale=bool(getattr(report, "stale", False)),
            presented_identities=getattr(report, "presented_identities", None),
        )
        if set(machine_accepted) != set(report_accepted):
            report = replace(report, committed_upserts=frozenset(machine_accepted))
        acknowledged = self.tile_presentation_state.acknowledge_delta(delta, report)
        self.tile_presentation_state = acknowledged
        accepted_upserts = report.accepted_upserts(delta)
        # Remember acknowledged payload identities: re-presenting one is a
        # residency remap for the backend, so commit batching may treat it as
        # nearly free instead of charging full texture bytes (ADR 0050 —
        # prompt level-swap convergence). Bounded: identities are small
        # tuples and the set resets with the session.
        for tile_number in accepted_upserts:
            payload = acknowledged.payloads.get(int(tile_number))
            if payload is not None:
                self.acknowledged_source_ids.add(payload.source_id)
        committed_levels = None if levels is None else (float(levels[0]), float(levels[1]))
        if committed_levels is not None:
            ProgressiveTileLevelConvergence().acknowledge(
                self.level_generation,
                target_revision=int(delta.level_revision),
                accepted_tiles=accepted_upserts,
                levels=committed_levels,
            )
        for tile in accepted_upserts:
            self.dirty_payloads.pop(int(tile), None)
            self.pending_payload_upserts.pop(int(tile), None)
        # Viewport-scoped backends accept only the active set (ADR 0044);
        # non-active upserts they decline are parked, not retried — every
        # payload stays cached and re-arms when the tile becomes active.
        # Without this, each commit re-emitted the same unacceptable upserts
        # and finalization never settled (idle commit/draw loop).
        # P2: the machine already made the presented/parked decision above;
        # these pops consume its verdict (accepted = machine-confirmed set).
        accepted = {int(tile) for tile in accepted_upserts}
        for tile in tuple(delta.upserts):
            index = int(tile)
            if index in accepted or index in active_scope:
                continue
            self.dirty_payloads.pop(index, None)
            self.pending_payload_upserts.pop(index, None)
        for tile in report.removed_tiles:
            self.pending_removals.discard(int(tile))
            self.dirty_payloads.pop(int(tile), None)
            self.pending_payload_upserts.pop(int(tile), None)
            self.level_generation.forget_tile(int(tile))
            self.display_tile_payloads.pop(int(tile), None)
        return acknowledged

    def backend_identity_mismatch_tiles(self) -> tuple[int, ...]:
        """Drawn slots whose backend identity differs from the current payload
        and that convergence has not exhausted (field defect 2026-07-05 #3).

        The drawn-identity reconciliation runs inside ``build_tile_presentation``
        — i.e. only during a commit — but the report that REVEALS staleness
        arrives at the END of a commit.  A settled session (dirty and upserts
        empty) therefore froze with ``backend_stale_identities`` nonzero until
        an unrelated event happened to schedule a commit.  The renderer calls
        this after every acknowledgement and schedules a follow-up commit when
        it is non-empty; bounded by the same resigned-pair and attempt limits
        as the reconciliation itself, so a non-converging backend cannot loop.
        """

        shown_map = getattr(self, "last_presented_identities", None) or {}
        if not shown_map:
            return ()
        mismatched: list[int] = []
        for tile_number, shown_identity in shown_map.items():
            current = self.display_tile_payloads.get(int(tile_number))
            if current is None or current.source_id == shown_identity:
                continue
            rec = self.lifecycle.peek(int(tile_number))
            if rec is not None and (current.source_id, shown_identity) in rec.resigned:
                continue
            pair = (shown_identity, current.source_id)
            prior_pair, attempts = self._reconcile_attempts.get(int(tile_number), (None, 0))
            if prior_pair == pair and attempts >= 3:
                continue
            mismatched.append(int(tile_number))
        return tuple(sorted(mismatched))

    def requeue_orphaned_loading_tiles(self) -> int:
        """Re-enqueue planned tiles stuck in loading with no work attached.

        A tile can be dequeued, marked loading, and then lose its evaluation
        (declined admission, crashed worker path).  Loading with no pending,
        active, or completed-pending work is unservable state: the visible
        plan can never complete and finalization retries forever.  Repair by
        returning such tiles to the pending queue; scheduling stays bounded
        by the ordinary drain.  Idempotent and cheap — set arithmetic plus
        one enqueue per orphan.
        """

        if bool(getattr(self, "stage_planning_deferred", False)):
            # Interaction-burst session: missing tiles intentionally have no
            # work attached until the deferred stage planning runs.  Requeuing
            # them here would start direct per-tile evaluations without the
            # stage fan-in (each tile computing the expensive stage alone).
            return 0
        orphaned = (
            set(int(t) for t in self.loading_tiles)
            - set(int(t) for t in self.active_tile_requests)
            - set(int(t) for t in self.rendered_tiles)
            - {
                int(getattr(entry[0] if isinstance(entry, tuple) else entry, "montage_index", -1))
                for entry in tuple(self.pending_completed_tiles)
            }
            - {int(t.montage_index) for t in self.pending_tiles}
        )
        if not orphaned:
            return 0
        by_number = {int(t.montage_index): t for t in tuple(self.plan.tiles)}
        requeued = 0
        for index in sorted(orphaned):
            tile = by_number.get(int(index))
            # P1: no residency claims are routed through the machine yet, so
            # the decline can have no ReleaseClaim effects to execute (rule 4:
            # never silently drop effects -- fail loudly if P3 wiring lands
            # without updating this repair path).
            if self.lifecycle.evaluation_declined(int(index)):
                raise AssertionError("unexecuted ReleaseClaim effects in requeue repair")
            if tile is None or int(index) in self.skipped_tiles:
                self.loading_tiles.discard(int(index))
                continue
            self.loading_tiles.discard(int(index))
            if self.enqueue_pending_tile(tile):
                requeued += 1
        return requeued

    def mark_loading(self, tile: MontageTile) -> None:
        index = int(tile.montage_index)
        if index not in self.rendered_tiles and index not in self.skipped_tiles:
            self.loading_tiles.add(index)
            self.lifecycle.evaluation_started(index)
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
            self.level_generation.forget_tile(index)
            self.tile_source_ids.pop(index, None)
            self.discard_pending_tile(index)
            self.lifecycle.tile_skipped(index)
            self.mark_tile_state(tile, MontageTileState.SKIPPED)

    def next_tile(self) -> MontageTile | None:
        self._ensure_pending_priority_queue()
        while self.pending_tiles:
            tile = self.pending_tiles.pop()
            index = int(tile.montage_index)
            if index not in self.rendered_tiles and index not in self.skipped_tiles:
                self.mark_loading(tile)
                self.active_tile_requests.add(index)
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
        return self._tile_states_cached_tuple

    def is_complete(self) -> bool:
        return not (
            self.pending_tiles
            or self.loading_tiles
            or self.pending_completed_tiles
            or self.active_tile_requests
            or self.stage_fan_in.active_requests
            or self.stage_fan_in.attached_requests
            or self.stage_fan_in.waiting_tiles
            or self.final_commit_pending
            or self.flush_pending
            or self.dirty_payloads
            or self.pending_payload_upserts
            or self.pending_removals
            or self.has_pending_level_update()
        )

    def visible_plan_complete(self) -> bool:
        required = set(int(tile) for tile in self.visible_tile_numbers) - set(int(tile) for tile in self.skipped_tiles)
        if not required:
            return True
        if self.has_stale_level_presentations():
            return False
        return required.issubset(set(int(tile) for tile in self.presented_tiles))

    def has_stale_level_presentations(self) -> bool:
        snapshot = self.level_presentation_snapshot()
        return bool(self.has_pending_level_update() and int(snapshot.stale_count) > 0)

    def _tile_matches_current_level_target(self, tile: int, target: tuple[float, float] | None) -> bool:
        if target is None:
            return True
        return (
            levels_match(self.level_generation.tile_values.get(int(tile)), target)
            and self.level_generation.tile_revisions.get(int(tile)) == int(self.level_revision)
        )

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

    def invalidate_tile_states(self) -> None:
        self.tile_state_revision += 1
        self._tile_states_cached_revision = -1
        self._tile_states_cached_tuple = ()

    def consume_dirty_rects(self) -> tuple[tuple[int, int, int, int], ...]:
        return ()

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
        waiting = self.stage_fan_in.waiting_tiles.get(key)
        if not isinstance(waiting, MontageTilePriorityQueue):
            waiting = MontageTilePriorityQueue(tuple(waiting or ()), context_provider=self._tile_priority_context)
            self.stage_fan_in.waiting_tiles[key] = waiting
        before = len(waiting)
        waiting.extend(tuple(tiles or ()))
        return max(0, len(waiting) - before)

    def _remap_queued_tiles_to_plan(self) -> None:
        """Rebind queued tile objects to the current plan's geometry.

        Session invariant: every queued tile belongs to ``self.plan``. A
        layout reflow (column-count change during window-shape settling)
        moves each montage index to a new position; tile objects captured
        under the previous geometry would be *scheduled* by their stale
        coordinates but *drawn* at the new ones, so the fill visibly ignores
        the priority order no matter how correct the scheduling context is.
        """
        tiles = tuple(getattr(self.plan, "tiles", ()) or ())

        def remap(tile):
            index = int(getattr(tile, "montage_index", -1))
            return tiles[index] if 0 <= index < len(tiles) else tile

        if isinstance(self.pending_tiles, MontageTilePriorityQueue):
            pending = tuple(self.pending_tiles.insertion_tiles())
        else:
            pending = tuple(self.pending_tiles or ())
        self.pending_tiles = MontageTilePriorityQueue(
            tuple(remap(tile) for tile in pending),
            context_provider=self._tile_priority_context,
        )
        for key, waiting in tuple(self.stage_fan_in.waiting_tiles.items()):
            self.stage_fan_in.waiting_tiles[key] = MontageTilePriorityQueue(
                tuple(remap(tile) for tile in tuple(waiting)),
                context_provider=self._tile_priority_context,
            )

    def ensure_stage_waiting_priority_queues(self) -> None:
        """Order stage fan-in waiting tiles by viewport/focus priority.

        Stage plans collect waiting tiles in plan (row-major) order; released
        as-is, budget-capped activation batches would fill the montage from a
        corner no matter what the pending queue's priority says.
        """
        if not self.stage_fan_in.waiting_tiles:
            return
        for key, waiting in tuple(self.stage_fan_in.waiting_tiles.items()):
            if not isinstance(waiting, MontageTilePriorityQueue):
                self.stage_fan_in.waiting_tiles[key] = MontageTilePriorityQueue(
                    tuple(waiting or ()), context_provider=self._tile_priority_context
                )

    def retarget_tile_priority(
        self,
        *,
        focus=None,
        max_items: int = 64,
        active_tiles=None,
        near_tiles=None,
        view_range=None,
    ) -> int:
        # ``view_range`` overrides the range used for priority scoring only;
        # ``self.view_range`` is viewport bookkeeping shared with level and
        # commit scoping and must not be retargeted from priority refreshes.
        range_for_priority = self.view_range if view_range is None else view_range
        if active_tiles is None:
            active_tiles = tuple(int(tile.montage_index) for tile in self.visible_tiles)
        if near_tiles is None:
            near_tiles = tuple(
                int(tile.montage_index)
                for tile in _viewport_tiles(
                    self.plan,
                    view_range=range_for_priority,
                    viewport_shape=self.viewport_shape,
                    margin_tiles=2,
                )
            )
        if focus is not None:
            self.priority_focus = focus
        context = self._build_tile_priority_context(
            active_tiles=active_tiles,
            near_tiles=near_tiles,
            priority_tiles=self._priority_focus_tile_numbers(),
            view_range=range_for_priority,
        )
        # Every ordering consumer (pending queue, stage fan-in waiting queues,
        # per-commit upsert admission, prefetch candidates) reads this one
        # context through _tile_priority_context; retargets are the only
        # writer, and queues resolve it live via their context provider.
        # Rebuilding the context ad hoc per consumer let different stages of
        # the pipeline order the same fill around different anchors.
        del max_items
        self._priority_context = context
        self._ensure_pending_priority_queue()
        self.priority_retargeted_tiles = len(self.pending_tiles) + sum(
            len(waiting) for waiting in self.stage_fan_in.waiting_tiles.values()
        )
        return int(self.priority_retargeted_tiles)

    def _tile_priority_context(self) -> TilePriorityContext:
        """The session's single effective ordering context.

        Updated only by :meth:`retarget_tile_priority`; built lazily before
        the first retarget.
        """
        context = getattr(self, "_priority_context", None)
        if context is None:
            context = self._build_tile_priority_context()
            self._priority_context = context
        return context

    def _build_tile_priority_context(self, *, active_tiles=None, near_tiles=None, priority_tiles=None, view_range=None) -> TilePriorityContext:
        if active_tiles is None:
            active_tiles = tuple(int(tile.montage_index) for tile in self.visible_tiles)
        if near_tiles is None:
            try:
                near_tiles = self._near_tile_numbers(margin_tiles=2)
            except Exception:
                near_tiles = ()
        return TilePriorityContext.from_tiles(
            view_range=self.view_range if view_range is None else view_range,
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

    def _prioritized_tile_numbers(self, tiles) -> tuple[int, ...]:
        requested = tuple(dict.fromkeys(int(tile) for tile in tuple(tiles or ())))
        if len(requested) <= 1:
            return requested
        valid_tiles = []
        fallback = []
        plan_tiles = tuple(getattr(self.plan, "tiles", ()) or ())
        for tile_number in requested:
            if 0 <= int(tile_number) < len(plan_tiles):
                valid_tiles.append(plan_tiles[int(tile_number)])
            else:
                fallback.append(int(tile_number))
        if not valid_tiles:
            return requested
        queue = MontageTilePriorityQueue(valid_tiles, context=self._tile_priority_context())
        ordered = list(tile_numbers(queue.ordered_tiles()))
        ordered.extend(tile for tile in fallback if tile not in ordered)
        return tuple(int(tile) for tile in ordered if int(tile) in set(requested))

    def _ensure_pending_priority_queue(self, *, context: TilePriorityContext | None = None) -> None:
        del context
        if isinstance(self.pending_tiles, MontageTilePriorityQueue):
            return
        self.pending_tiles = MontageTilePriorityQueue(
            tuple(self.pending_tiles or ()),
            context_provider=self._tile_priority_context,
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
