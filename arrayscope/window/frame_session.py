"""Qt-free context and derived views for progressive image-frame rendering."""

from __future__ import annotations

import os
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from time import monotonic

import numpy as np

from arrayscope.core.trace import emit_trace
from arrayscope.display.lod import (
    LOD_POLICY_NATIVE_ONLY,
    LodInfo,
    LodPolicyDecision,
    native_lod_policy,
)
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    TileCommitReport,
    TilePresentationDelta,
    TilePresentationState,
)
from arrayscope.display.model.level_convergence import (
    ProgressiveTileLevelConvergence,
    UniformLevelConvergence,
)
from arrayscope.display.model.presentation_generation import (
    PresentationGenerationSnapshot as LevelPresentationSnapshot,
)
from arrayscope.display.model.presentation_generation import (
    PresentationGenerationTracker,
    levels_match,
)
from arrayscope.display.model.tile_admission import TileAdmissionQueue
from arrayscope.display.model.tile_identity import (
    TileIdentity,
    TileLodIdentity,
    TilePresentationIdentity,
    array_plane_identities,
    complex_mapping_identity,
    tile_ack_identity,
    tile_truth_record,
)
from arrayscope.display.model.tile_priority import (
    TilePriorityContext,
    prioritize_tile_numbers,
)
from arrayscope.display.montage import (
    MontagePlan,
    MontageTile,
    MontageTileState,
    RenderedTile,
    montage_rect_for_viewport,
)
from arrayscope.display.pyramid import MaterializedLodPage
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.operations.stage_fanin import StageFanInState
from arrayscope.presentation import (
    ClaimOwner,
    LevelPhase,
    ReleaseClaim,
    TileLifecycle,
    TileTarget,
    payload_ref_from_display_payload,
)
from arrayscope.render import lod as render_lod
from arrayscope.render.lod import (  # noqa: F401  (re-exports; canonical home is render_lod)
    LodPageMaterializationRequest,
    page_set_key_for_rendered,
    texture_source_for_rendered,
)
from arrayscope.render.lod import (
    viewport_identity as _viewport_identity,
)
from arrayscope.render.progressive_scheduling import ProgressiveSchedulingPolicy


@dataclass(frozen=True)
class PreviewFloorMetadata:
    shader_mapping: object | None = None
    texture_kind: TexturePlaneKind | None = None
    level_data: np.ndarray | None = None
    level_stats: object | None = None
    quality: str = "preview"


@dataclass(frozen=True)
class SemanticLevelEvidenceTarget:
    """Immutable full-population statistics obligation for one frame."""

    generation: object
    level_key: object
    expected_sources: tuple[int, ...]
    pixel_limit: int
    aggregate_sample_limit: int
    blocking_batch_limit: int
    background_batch_limit: int

    @property
    def target_population(self) -> int:
        return len(self.expected_sources)


@dataclass
class SemanticLevelEvidenceProgress:
    """Bounded scheduler progress; LevelStatsService is its only mutator."""

    target: SemanticLevelEvidenceTarget
    cursor: int = 0
    covered_sources: set[int] = field(default_factory=set)
    covered_sources_sample: list[int] = field(default_factory=list)
    current_batch_limit: int = 1
    inflight_generation: object | None = None
    blocking_reason: str = "waiting-semantic-sources"

    @property
    def pending_batches(self) -> int:
        remaining = max(0, int(self.target.target_population) - len(self.covered_sources))
        limit = max(1, int(self.current_batch_limit))
        return (remaining + limit - 1) // limit

    def record_covered(self, source_index: int) -> bool:
        source_index = int(source_index)
        if source_index in self.covered_sources:
            return False
        self.covered_sources.add(source_index)
        if len(self.covered_sources_sample) < 64:
            self.covered_sources_sample.append(source_index)
        return True


def _shader_mapping_key(mapping):
    return None if mapping is None else getattr(mapping, "identity_key", mapping)


def _debug_lod_pass_texture(texture, *, quality: str):
    mode = str(os.environ.get("ARRAYSCOPE_LOD_DEBUG_PASS_MARKER", "") or "").strip().lower()
    if not mode:
        return texture
    if str(quality) != "exact":
        return texture
    values = np.asarray(texture)
    if mode in {"final-mirror-x", "final-mirror-negate"}:
        values = np.flip(values, axis=1)
    if mode in {"final-negate", "final-mirror-negate"}:
        if np.issubdtype(values.dtype, np.integer):
            info = np.iinfo(values.dtype)
            values = np.asarray(info.max - values)
        else:
            values = -values
    return np.ascontiguousarray(values)


def _payload_has_level_presentation_evidence(payload, *, shader_display: bool) -> bool:
    quality = str(getattr(payload, "quality", "exact") or "exact")
    if quality == "exact":
        return True
    page_backing = getattr(payload, "page_backing", None)
    if page_backing is not None and tuple(getattr(page_backing, "materialized_pages", ()) or ()):
        # A CPU/page floor preview is intentionally non-semantic (shader
        # previews below may instead carry explicit re-window evidence);
        # canonical reduced values are sufficient to re-window a non-preview
        # physical fallback without promoting it to exact semantic quality.
        return quality != "preview"
    if not bool(shader_display):
        return False
    if getattr(payload, "level_stats", None) is not None:
        return True
    if getattr(payload, "level_data", None) is not None:
        return True
    return getattr(payload, "histogram_data", None) is not None


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
        if any(
            previous_tile != int(tile)
            for previous_tile in previous_tiles_by_key.get(_payload_residency_key(payload), set())
        )
    }


def _free_retarget_tiles(
    payloads: dict[int, DisplayTilePayload],
    *,
    logical_resident_tiles,
    physical_resident_fn,
    pace_resident_retargets: bool,
) -> frozenset[int]:
    """Return retargets proven to bypass every backend admission cap.

    A backend physical-residency seam is authoritative when present.  The
    scheduler's acknowledged source identities are intentionally not used as
    a substitute: an acknowledged atlas page may since have been evicted.
    Backends without physical residency truth retain the existing policy in
    which only explicitly unpaced logical remaps are free.
    """

    if callable(physical_resident_fn):
        return frozenset(
            int(tile) for tile, payload in payloads.items() if bool(physical_resident_fn(payload))
        )
    if pace_resident_retargets:
        return frozenset()
    return frozenset(int(tile) for tile in logical_resident_tiles)


def _montage_plan_topology(plan: MontagePlan) -> tuple:
    """Return layout identity without folding semantic source indices into it."""

    return (
        tuple(int(value) for value in plan.tile_shape),
        tuple(int(value) for value in plan.grid_shape),
        int(plan.columns),
        int(plan.rows),
        int(plan.gap),
        tuple(
            (
                int(tile.montage_index),
                int(tile.row),
                int(tile.col),
                int(tile.x0),
                int(tile.y0),
                int(tile.width),
                int(tile.height),
            )
            for tile in plan.tiles
        ),
    )


def _physical_rebind_transaction(
    payloads: dict[int, DisplayTilePayload],
    *,
    free_retarget_tiles,
    physical_resident_fn,
) -> dict[int, DisplayTilePayload]:
    """Keep a mixed persistent-GPU delta from delaying resident rebinds.

    The persistent backend defers physically cold replacements during an
    active gesture.  One cold member in a mixed delta would therefore delay
    every already-resident page mapping in that delta.  Emit the complete
    physically free cohort first; cold members remain in the session's
    pending-upsert store for the next bounded transaction.  Semantic successor
    handoffs bypass ordinary admission limits and never enter this split.
    """

    free = frozenset(int(tile) for tile in tuple(free_retarget_tiles or ()))
    if (
        not callable(physical_resident_fn)
        or not free
        or all(int(tile) in free for tile in payloads)
    ):
        return payloads
    return {int(tile): payload for tile, payload in payloads.items() if int(tile) in free}


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


class _LifecycleTileSetView:
    """Set-like event view over one lifecycle index."""

    __slots__ = ("_lifecycle",)

    #: name of the lifecycle property this view reads.
    _view = ""

    def __init__(self, lifecycle, seed=()):
        self._lifecycle = lifecycle
        for index in tuple(seed or ()):
            self.add(int(index))

    def _snapshot(self) -> frozenset[int]:
        return getattr(self._lifecycle, self._view)

    def _event_add(self, index: int) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _event_discard(self, index: int) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def add(self, index) -> None:
        self._event_add(int(index))

    def discard(self, index) -> None:
        self._event_discard(int(index))

    def remove(self, index) -> None:
        if int(index) not in self._snapshot():
            raise KeyError(index)
        self._event_discard(int(index))

    def clear(self) -> None:
        for index in tuple(self._snapshot()):
            self._event_discard(int(index))

    def update(self, indices) -> None:
        for index in tuple(indices or ()):
            self._event_add(int(index))

    def __contains__(self, index) -> bool:
        try:
            return int(index) in self._snapshot()
        except (TypeError, ValueError):
            return False

    def __iter__(self):
        return iter(sorted(self._snapshot()))

    def __len__(self) -> int:
        return len(self._snapshot())

    def __bool__(self) -> bool:
        return bool(self._snapshot())

    # Mutable view: equality is by current contents, so instances are
    # deliberately unhashable.
    __hash__ = None

    def __eq__(self, other) -> bool:
        if isinstance(other, _LifecycleTileSetView):
            return self._snapshot() == other._snapshot()
        if isinstance(other, (set, frozenset)):
            return set(self._snapshot()) == other
        return NotImplemented

    def __repr__(self) -> str:  # diagnostics/debugging only
        return f"{type(self).__name__}({sorted(self._snapshot())})"


class LifecycleLoadingTiles(_LifecycleTileSetView):
    _view = "loading_tiles"

    def _event_add(self, index: int) -> None:
        self._lifecycle.load_marked(index)

    def _event_discard(self, index: int) -> None:
        self._lifecycle.load_cleared(index)


class LifecycleActiveRequests(_LifecycleTileSetView):
    _view = "active_request_tiles"

    def _event_add(self, index: int) -> None:
        self._lifecycle.evaluation_requested(index)

    def _event_discard(self, index: int) -> None:
        self._lifecycle.evaluation_request_cleared(index)


class LifecycleSkippedTiles(_LifecycleTileSetView):
    _view = "skipped_tiles"

    def _event_add(self, index: int) -> None:
        self._lifecycle.tile_skipped(index)

    def _event_discard(self, index: int) -> None:
        self._lifecycle.tile_unskipped(index)


class LifecycleRungMaterializations:
    """List-like view over lifecycle-owned rung residency claims.

    The request object carries worker input (source plane and reduction chain),
    but its existence is not a second truth source: the lifecycle record owns
    the per-level claim and its phase.  Iteration returns records in claim order;
    draining marks claims materializing, and terminal paths release or mark them
    resident through machine events.
    """

    __slots__ = ("_session",)

    def __init__(self, session, seed=()):
        self._session = session
        for request in tuple(seed or ()):
            self.append(request)

    @property
    def _lifecycle(self):
        return self._session.lifecycle

    def _snapshot(self) -> tuple:
        return self._lifecycle.pending_materializations()

    def append(self, request) -> None:
        tile_number = int(request.tile_number)
        self._lifecycle.materialization_planned(
            tile_number,
            request,
            owner=ClaimOwner.CHAIN,
        )

    def drain(self) -> tuple:
        requests = self._snapshot()
        for request in requests:
            self._lifecycle.materialization_started(request)
        return requests

    def mark_started(self, request) -> None:
        self._lifecycle.materialization_started(request)

    def mark_resident(self, request) -> bool:
        pyramid = getattr(self._session, "lod_page_cache", None)
        if not render_lod._page_set_exact(pyramid, request.key):
            self.release(request)
            return False
        self._lifecycle.materialization_resident(request)
        return True

    def release(self, request) -> tuple[ReleaseClaim, ...]:
        effects = self._lifecycle.materialization_released(request)
        pyramid = getattr(self._session, "lod_page_cache", None)
        release = getattr(pyramid, "release_owner_claims", None)
        if callable(release):
            release(request.owner)
        return effects

    def clear(self) -> None:
        for request in self._snapshot():
            self._apply_release_effects(self.release(request))

    def pop(self, index: int = -1):
        requests = list(self._snapshot())
        request = requests.pop(index)
        self._lifecycle.materialization_started(request)
        return request

    def __getitem__(self, index):
        return self._snapshot()[index]

    def __iter__(self):
        return iter(self._snapshot())

    def __len__(self) -> int:
        return len(self._snapshot())

    def __bool__(self) -> bool:
        return bool(self._snapshot())

    # Mutable view: equality is by current contents, so instances are
    # deliberately unhashable.
    __hash__ = None

    def __eq__(self, other) -> bool:
        if isinstance(other, LifecycleRungMaterializations):
            return self._snapshot() == other._snapshot()
        if isinstance(other, (list, tuple)):
            return list(self._snapshot()) == list(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._snapshot())!r})"

    def _apply_release_effects(self, effects) -> None:
        # Page claims release by request owner in ``release``. Lifecycle
        # effects carry the tile/rung key and must never be reinterpreted as
        # cache keys.
        tuple(effects or ())


def _stage_tile_index(tile_or_index) -> int:
    try:
        return int(tile_or_index.montage_index)
    except AttributeError:
        return int(tile_or_index)


def _stage_bindings_by_key(tile_stage_keys) -> dict[object, tuple[int, ...]]:
    bindings: dict[object, list[int]] = {}
    for tile_number, key in dict(tile_stage_keys or {}).items():
        bindings.setdefault(key, []).append(int(tile_number))
    return {key: tuple(sorted(values)) for key, values in bindings.items()}


class LifecycleStageFanIn(StageFanInState):
    """Stage fan-in whose tile↔stage bindings report through machine events.

    Every mutation updates lifecycle records, so stage ownership is a direct
    tile fact rather than a correlation across maps.
    """

    def __init__(self, lifecycle, state: StageFanInState | None = None):
        state = state if state is not None else StageFanInState()
        super().__init__()
        self.active_requests = state.active_requests
        self.attached_requests = state.attached_requests
        self.values = state.values
        self.tile_stage_keys = state.tile_stage_keys
        self.tile_stage_plans = state.tile_stage_plans
        self.tile_stage_candidates = state.tile_stage_candidates
        self.lead_warmups = state.lead_warmups
        self._lifecycle = lifecycle
        self._report_bindings()

    def _report_bindings(self) -> None:
        self._lifecycle.stage_bindings_replaced(_stage_bindings_by_key(self.tile_stage_keys))

    def merge_plan(self, plan: dict) -> None:
        super().merge_plan(plan)
        self._report_bindings()

    def activate_value(self, key, value, *, max_items: int | None = None):
        batch = super().activate_value(key, value, max_items=max_items)
        self._report_bindings()
        return batch

    def release_missing(self, key, *, max_items: int | None = None):
        batch = super().release_missing(key, max_items=max_items)
        self._report_bindings()
        return batch

    def detach_unbound_requests(self) -> None:
        super().detach_unbound_requests()
        self._report_bindings()

    def fail(self, key):
        waiting = super().fail(key)
        self._report_bindings()
        return waiting


@dataclass
class FrameSession:
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
    shader_display: bool = False
    # ADR 0055 G3: window-invariant anchoring for non-montage sessions on
    # atlas backends (display.source_anchoring.SourceAnchoring); stamps
    # exact payloads with a PayloadSourceAnchor so backend residency can
    # survive display-window shifts. None for montage sessions and
    # non-windowable chains.
    source_anchoring: object | None = None
    active_tile_requests: set[int] = field(default_factory=set)
    stage_fan_in: StageFanInState = field(default_factory=StageFanInState)
    tile_states: list[MontageTileState] = field(default_factory=list)
    dirty_tiles: list[int] = field(default_factory=list)
    flush_pending: bool = False
    last_commit_monotonic: float = 0.0
    # Monotonic count of executed commit batches (including acknowledgement-
    # only and no-op batches).  The stall watchdog folds this into its
    # signature so a slow-but-live drain (e.g. one upsert per batch at 22 Hz)
    # never reads as a frozen session while commits are still landing.
    commit_batches: int = 0
    final_commit_pending: bool = False
    show_loading_overlays: bool = False
    defer_side_panels: bool = False
    display_committed: bool = False
    # One explicit backend pass owed even when semantic payloads are unchanged.
    backend_refresh_pending: bool = False
    applied_level_source: object | None = None
    user_levels_override: tuple[float, float] | None = None
    pending_level_tiles: deque[RenderedTile] = field(default_factory=deque)
    pending_level_sources: set[int] = field(default_factory=set)
    level_scan_cursor: int = 0
    level_scan_remaining_tiles: int = 0
    level_evidence_inflight: bool = False
    level_evidence_generation: object | None = None
    histogram_aggregate_inflight: bool = False
    histogram_aggregate_generation: object | None = None
    semantic_level_evidence_target: SemanticLevelEvidenceTarget | None = None
    semantic_level_evidence_progress: SemanticLevelEvidenceProgress | None = None
    first_pass_quality: str | None = None
    first_pass_histogram_published: bool = False
    pending_refined_level_tiles: deque[RenderedTile] = field(default_factory=deque)
    pending_refined_level_sources: set[int] = field(default_factory=set)
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
    stage_planning_deferred: bool = False
    stage_planning_async: bool = False
    deferred_missing_tiles: tuple[MontageTile, ...] = ()
    frame_plan: object | None = None
    pipeline: object | None = None
    tile_source_ids: dict[int, object] = field(default_factory=dict)
    display_tile_payloads: dict[int, DisplayTilePayload] = field(default_factory=dict)
    dirty_payloads: OrderedDict[int, None] = field(default_factory=OrderedDict)
    pending_payload_upserts: OrderedDict[int, None] = field(default_factory=OrderedDict)
    pending_removals: set[int] = field(default_factory=set)
    visible_tile_numbers: frozenset[int] = field(default_factory=frozenset)
    level_generation: PresentationGenerationTracker = field(
        default_factory=PresentationGenerationTracker
    )
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
    # A compatible successor needs one atomic physical handoff.
    # Quality convergence after that handoff must use ordinary per-tile deltas;
    # rebuilding the all-slot transaction re-acknowledges settled tiles and can
    # starve the entering tile's target rung indefinitely.
    # This is the sole owner of whether that handoff is still owed. It is
    # armed by the transition decision and cleared only by a complete backend
    # acknowledgement; lifecycle settlement is not a second completion owner.
    atomic_successor_pending: bool = False
    lod_policy_decision: LodPolicyDecision = field(
        default_factory=lambda: native_lod_policy(None, (1, 1), (1, 1))
    )
    # ADR 0050: "native-only" keeps production behavior; "resident" presents
    # the closest pyramid level that is actually materialized and resident.
    lod_policy_mode: str = LOD_POLICY_NATIVE_ONLY
    # Why native-only applies when the desired factor exceeds 1 (user policy
    # choice vs. resident LOD not yet adopted on the active backend).
    lod_native_reason: str | None = None
    lod_page_cache: object | None = None
    # Immutable canonical routes are reused across viewport retargets. Physical
    # residency has its own cache revision; changing it must not rebuild page
    # geometry for every floor query.
    _lod_page_set_key_cache: dict[object, object] = field(default_factory=dict)
    # List-like view over lifecycle-owned rung materialization claims.  Filled
    # only under the "resident" policy after a singleflight claim on the
    # pyramid cache; the lifecycle record, not this attribute, owns truth.
    pending_rung_materializations: list = field(default_factory=list)
    lod_materializations_completed: int = 0
    acknowledged_source_ids: set = field(default_factory=set)
    lod_floor_presentations: int = 0
    lod_target_revision: int = 0
    lifecycle: TileLifecycle = field(default_factory=TileLifecycle)
    scheduling_policy: ProgressiveSchedulingPolicy = field(
        default_factory=ProgressiveSchedulingPolicy
    )
    # Window-agnostic texel identity base for pyramid floors/previews
    # (montage_tile_semantic_key); falls back to session key when unset.
    semantic_key: object = None
    # Bounded retry state for backend identity mismatches:
    # tile -> ((shown_identity, wanted_identity), attempts).
    _identity_retry_attempts: dict = field(default_factory=dict)
    lod_preview_level: int = 0
    lod_preview_min_level: int = 0
    tile_residency_budget_bytes: int = 0
    lod_preview_metadata: dict[object, PreviewFloorMetadata] = field(default_factory=dict)
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

    def rearm_visible_parked_payloads(self) -> tuple[int, ...]:
        """Re-arm parked tiles that are visible and have something to present."""

        planned = {
            int(tile.montage_index)
            for tile in tuple(getattr(self, "visible_tiles", ()) or ())
            if int(tile.montage_index) not in self.skipped_tiles
        }
        presentable = {
            int(tile)
            for tile in planned
            if int(tile) in self.display_tile_payloads or int(tile) in self.rendered_tiles
        }
        rearmed = self.lifecycle.rearm_for_scope(presentable)
        for tile_number in rearmed:
            if (
                int(tile_number) in self.display_tile_payloads
                or int(tile_number) in self.rendered_tiles
            ):
                self.dirty_payloads[int(tile_number)] = None
        return tuple(rearmed)

    def required_tile_numbers(self) -> tuple[int, ...]:
        """The one set that admission, evidence, and completion must render."""

        frame_plan = getattr(self, "frame_plan", None)
        required = (
            tuple(getattr(frame_plan, "active_region_ids", ()) or ())
            if frame_plan is not None
            else tuple(self.visible_tile_numbers)
        )
        return tuple(
            sorted({int(tile) for tile in required} - {int(tile) for tile in self.skipped_tiles})
        )

    def required_target_unsettled_tiles(self) -> tuple[int, ...]:
        return self.lifecycle.target_unsettled_tiles(self.required_tile_numbers())

    def required_target_settled(self) -> bool:
        return not self.required_target_unsettled_tiles()

    def required_first_pixels_presented(self) -> bool:
        """Whether the unique current required scope has physical coverage."""

        return self.lifecycle.first_pixels_presented(self.required_tile_numbers())

    def atomic_successor_required_scope(self) -> tuple[int, ...]:
        """Required on-screen slots eligible for one atomic handoff.

        ``visible_tile_numbers`` is legacy terminology for the broader
        presentation-coverage set. FramePlan owns the actual on-screen
        requirement. Offscreen coverage may retain predecessor payloads, but
        every required tile must belong to coverage and cross atomically.
        """

        required = tuple(int(tile) for tile in self.required_tile_numbers())
        coverage = {int(tile) for tile in self.visible_tile_numbers}.difference(
            int(tile) for tile in self.skipped_tiles
        )
        return required if required and set(required).issubset(coverage) else ()

    def note_first_pass_quality(self, quality: str) -> bool:
        """Latch the one display quality allowed to contribute rough evidence."""

        quality = str(quality or "exact")
        if self.first_pass_quality is None:
            self.first_pass_quality = quality
        return self.first_pass_quality == quality

    def observe_physically_presented_first_pass_quality(self, payloads) -> bool:
        """Latch first-pass quality from one complete backend identity snapshot.

        Index-window retargeting deliberately resets the first-pass evidence
        generation while retaining compatible backend pixels.  Those payloads
        do not pass through admission again, so admission-time observation
        alone can leave ``first_pass_quality`` unset forever.  The backend
        snapshot is the canonical crossing: every required tile must report
        the identity of the current payload before its quality can seed the
        coverage evidence pass.

        This is coverage truth only.  A preview/fallback quality remains a
        non-exact first pass and does not acknowledge or settle the exact tile
        target.
        """

        current_payloads = dict(payloads or {})
        backend_identities = dict(self.lifecycle.backend_presented_identities)
        qualities: set[str] = set()
        for tile_number in self.required_tile_numbers():
            index = int(tile_number)
            payload = current_payloads.get(index)
            if payload is None or backend_identities.get(index) != tile_ack_identity(payload):
                return False
            quality = str(getattr(payload, "quality", "exact") or "exact")
            if quality == "fallback":
                quality = "preview"
            if quality not in {"preview", "exact"}:
                return False
            qualities.add(quality)
        if not qualities:
            return False
        observed = "preview" if "preview" in qualities else "exact"
        if self.first_pass_quality == "exact" and observed == "preview":
            # The acknowledged backend snapshot is the quality owner.  A
            # retained exact floor can complete first, then be replaced by a
            # preview/fallback for the new target while wgpu histogram
            # evidence is in flight.  Keeping the earlier exact latch makes
            # the mixed physical frame incapable of closing first-pass
            # coverage even though every required tile is on screen.
            self.first_pass_quality = "preview"
            emit_trace(
                "first_pass_quality",
                event="widened_to_preview",
                session_id=int(self.session_id),
                required_tiles=len(self.required_tile_numbers()),
            )
        if self.first_pass_quality is None:
            return self.note_first_pass_quality(observed)
        return bool(
            self.first_pass_quality == observed
            or (self.first_pass_quality == "preview" and observed == "exact")
        )

    def first_pass_accepts_quality(self, quality: str) -> bool:
        """Whether a current payload proves its slot for the latched pass.

        A preview pass may reuse an already committed exact source without
        downgrading it.  That exact payload is stronger physical evidence than
        the preview obligation and must not hold the pass open forever.
        """

        first_pass = self.first_pass_quality
        payload_quality = str(quality or "exact")
        if payload_quality == "fallback":
            payload_quality = "preview"
        return bool(
            first_pass is not None
            and (
                payload_quality == str(first_pass)
                or (str(first_pass) == "preview" and payload_quality == "exact")
            )
        )

    def first_pass_pixels_presented(self) -> bool:
        """Whether every required target is acknowledged at the latched quality."""

        quality = self.first_pass_quality
        plan_tiles = {
            int(getattr(tile, "montage_index", offset)): tile
            for offset, tile in enumerate(tuple(getattr(self.plan, "tiles", ()) or ()))
        }
        tile_numbers = tuple(self.required_tile_numbers())
        if quality is None or not tile_numbers:
            return False
        for tile_number in tile_numbers:
            tile = plan_tiles.get(int(tile_number))
            if tile is None:
                return False
            payload = self.display_tile_payloads.get(tile_number)
            if (
                payload is None
                or int(getattr(payload, "source_index", -1)) != int(tile.source_index)
                or not self.first_pass_accepts_quality(
                    str(getattr(payload, "quality", "exact") or "exact")
                )
                or tile_number not in self.lifecycle.presented_tiles
            ):
                return False
        return True

    @property
    def level_revision(self) -> int:
        return int(self.level_generation.revision)

    @level_revision.setter
    def level_revision(self, value: int) -> None:
        self.level_generation.revision = int(value)

    def __post_init__(self) -> None:
        self.pending_level_tiles = deque(self.pending_level_tiles)
        self.pending_level_sources = {
            int(source) for source in (self.pending_level_sources or ())
        } or {int(item.tile.source_index) for item in self.pending_level_tiles}
        self.pending_refined_level_tiles = deque(self.pending_refined_level_tiles)
        self.pending_refined_level_sources = {
            int(source) for source in (self.pending_refined_level_sources or ())
        } or {int(item.tile.source_index) for item in self.pending_refined_level_tiles}
        self.visible_tile_numbers = frozenset(
            int(tile.montage_index) for tile in tuple(self.visible_tiles or ())
        )
        self._selected_lod_factor()
        self.update_level_presentation_scope()
        # ADR 0051: seed the lifecycle machine so its semantic axis matches a
        # session built from cached results (rendered == evaluated).  P2:
        # rendered_tiles becomes the event-routing collection, so every later
        # write (including direct fixture assignment) keeps the semantic axis
        # authoritative.
        self.lifecycle.plan_applied(
            int(tile.montage_index) for tile in tuple(self.visible_tiles or ())
        )
        # P2 sets-as-views: the constructor arguments seed the machine, then
        # the attributes BECOME views over it — the machine is the only owner
        # and every later mutation is an event.
        self.lifecycle.plan_applied(int(tile) for tile in tuple(self.loading_tiles or ()))
        self.loading_tiles = LifecycleLoadingTiles(
            self.lifecycle, sorted(int(tile) for tile in tuple(self.loading_tiles or ()))
        )
        self.skipped_tiles = LifecycleSkippedTiles(
            self.lifecycle, sorted(int(tile) for tile in tuple(self.skipped_tiles or ()))
        )
        self.active_tile_requests = LifecycleActiveRequests(
            self.lifecycle, sorted(int(tile) for tile in tuple(self.active_tile_requests or ()))
        )
        self.rendered_tiles = LifecycleRenderedTiles(self.lifecycle, self.rendered_tiles)
        self.attach_stage_fan_in(self.stage_fan_in)
        self.pending_rung_materializations = LifecycleRungMaterializations(
            self, self.pending_rung_materializations
        )
        for index in sorted(int(tile) for tile in self.rendered_tiles):
            self.dirty_payloads.setdefault(int(index), None)
        self.sync_lifecycle_scope()

    def attach_stage_fan_in(self, state: StageFanInState) -> None:
        """Install (or replace) the stage fan-in, machine bindings included.

        Every replacement site must come through here: assigning a bare
        ``StageFanInState`` would silently stop reporting stage events
        (ADR 0051 P2 — stages report through events).
        """

        if isinstance(state, LifecycleStageFanIn) and state._lifecycle is self.lifecycle:
            self.stage_fan_in = state
            return
        self.stage_fan_in = LifecycleStageFanIn(self.lifecycle, state)

    def _sync_lifecycle_targets(self) -> dict[int, TileTarget]:
        """Publish the current semantic targets without opening scheduling."""

        demand = getattr(getattr(self, "lod_policy_decision", None), "demand", None)
        target_level = int(getattr(demand, "desired_level", 0) or 0)
        targets: dict[int, TileTarget] = {}
        for tile in tuple(getattr(self, "visible_tiles", ()) or ()):
            index = int(tile.montage_index)
            if index in self.skipped_tiles:
                continue
            source_index = int(tile.source_index)
            targets[index] = TileTarget(
                tile_number=index,
                source_index=source_index,
                semantic_source_id=self.tile_semantic_source_id(source_index),
                lod_level=target_level,
                identity=self.tile_target_identity(tile, lod_level=target_level),
            )
        target_signature = tuple(
            (tile, target.source_index, target.semantic_source_id, target.lod_level)
            for tile, target in sorted(targets.items())
        )
        if getattr(self, "_lifecycle_target_signature", None) != target_signature:
            self.lifecycle.retarget(targets)
            self._lifecycle_target_signature = target_signature
        return targets

    def sync_lifecycle_scope(self) -> None:
        targets = self._sync_lifecycle_targets()
        required = {int(tile) for tile in self.required_tile_numbers()}
        scheduling_scope_signature = tuple(
            (tile, target.source_index, target.semantic_source_id)
            for tile, target in sorted(targets.items())
            if int(tile) in required
        )
        self.scheduling_policy.retarget(
            scheduling_scope_signature,
            tuple(sorted(required.intersection(targets))),
            progressive=bool(self.shader_display or self._resident_lod_active()),
        )

        # Payload mutation sites report lifecycle events directly.  This scan
        # remains as a safety net for restored/cached presentation state, but
        # unchanged payload objects must not be re-normalized on every
        # settlement query: that made each query O(visible tiles * payload
        # identity construction), and a 272-tile fill performed millions of
        # redundant conversions.
        seen = getattr(self, "_lifecycle_payload_objects", None)
        if seen is None:
            seen = {}
            self._lifecycle_payload_objects = seen
        payload_sets = (
            getattr(self, "display_tile_payloads", {}),
            getattr(getattr(self, "tile_presentation_state", None), "payloads", {}),
        )
        for payloads in payload_sets:
            for tile_number, payload in tuple((payloads or {}).items()):
                marker = id(payload)
                key = (id(payloads), int(tile_number))
                if seen.get(key) == marker:
                    continue
                seen[key] = marker
                self.record_tile_payload(payload)
        self._rearm_required_first_pixel_payloads()

    def _rearm_required_first_pixel_payloads(self) -> tuple[int, ...]:
        """Arm retained payloads whose required slots are not physically drawn."""

        rearmed: list[int] = []
        for tile_number in self.required_tile_numbers():
            index = int(tile_number)
            record = self.lifecycle.peek(index)
            if record is not None and record.first_pixel_presented:
                continue
            payload = self.display_tile_payloads.get(index)
            if payload is None or not self.lifecycle.payload_is_current(index, payload):
                continue
            self.dirty_payloads[index] = None
            self.pending_payload_upserts[index] = None
            rearmed.append(index)
        return tuple(rearmed)

    def record_tile_payload(self, payload) -> None:
        ref = payload_ref_from_display_payload(payload)
        if ref.quality == "exact":
            self.lifecycle.target_ready(int(ref.payload.tile_number), ref)
        else:
            self.lifecycle.fallback_ready(int(ref.payload.tile_number), ref)

    def lifecycle_snapshot(self):
        return self.lifecycle.snapshot()

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
    ) -> tuple[tuple[MontageTile, ...], bool]:
        """Retarget draw and compute coverage without replacing the session.

        The current draw set is kept separate from loaded payload ownership.
        Tiles outside the viewport can therefore remain GPU-resident and be
        reused when the user pans back.
        """

        self.viewport_shape = (max(1, int(viewport_shape[0])), max(1, int(viewport_shape[1])))
        self.view_range = view_range
        layout_changed = False
        if plan is not None:
            previous_geometry = getattr(self.plan, "geometry", None)
            next_geometry = getattr(plan, "geometry", None)
            layout_changed = next_geometry != previous_geometry
            self.plan = plan
            if layout_changed:
                self._layout_geometry_changed_pending = True
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
        known = {int(index) for index in self.rendered_tiles}
        known.update(int(index) for index in self.loading_tiles)
        known.update(int(index) for index in self.skipped_tiles)
        known.update(int(index) for index in self.active_tile_requests)
        additions = tuple(tile for tile in coverage if int(tile.montage_index) not in known)
        active_numbers = tuple(int(tile.montage_index) for tile in active)
        near_numbers = tuple(int(tile.montage_index) for tile in near)
        previous_visible_numbers = tuple(int(tile.montage_index) for tile in self.visible_tiles)
        presentation_changed = bool(layout_changed or active_numbers != previous_visible_numbers)
        self.visible_tiles = active
        self.visible_tile_numbers = frozenset(active_numbers)
        self.sync_lifecycle_scope()
        if presentation_changed:
            self.update_level_presentation_scope()
        self.priority_retargeted_tiles = self.retarget_tile_priority(
            focus=priority_focus,
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
        (``tile_presentation_state`` and lifecycle backend identities), and
        the currently drawn payloads all stay; only the semantic mapping
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
        (deferred-planning path). Returns remap statistics.
        """

        old_source_ids = dict(self.tile_source_ids)
        old_plan_topology = _montage_plan_topology(self.plan)
        old_rendered_tiles = dict(self.rendered_tiles)
        old_display_payloads = dict(self.display_tile_payloads)
        old_state_payloads = dict(getattr(self.tile_presentation_state, "payloads", {}) or {})
        had_complete_predecessor = bool(
            self.atomic_successor_pending or self.required_first_pixels_presented()
        )
        old_indices_by_source = {
            source_id: int(index)
            for index, source_id in old_source_ids.items()
            if source_id is not None
        }
        # Shared-transform/floor payloads can be the only resident semantic
        # evidence for a tile.  Derive the source map from those payloads too;
        # relying on the renderer-local source-id cache made a partially
        # populated cache turn an overlapping scroll into fresh preview work.
        for index, payload in {**old_state_payloads, **old_display_payloads}.items():
            source_id = _base_source_id(getattr(payload, "source_id", None))
            if source_id is None:
                continue
            try:
                old_indices_by_source.setdefault(source_id, int(index))
            except TypeError:
                # Source identities are expected to be hashable, but a custom
                # operation must not make retargeting itself fail if it isn't.
                continue
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
        self.pending_level_tiles.clear()
        self.pending_level_sources.clear()
        self.pending_refined_level_tiles.clear()
        self.pending_refined_level_sources.clear()
        # The in-flight flag belongs to the old level/session generation. Its
        # callback will reject itself by generation; carrying this bare bool
        # into the new window prevents that window from admitting any evidence
        # work and strands all level presentations stale.
        self.level_evidence_inflight = False
        self.level_evidence_generation = None
        self.histogram_aggregate_inflight = False
        self.histogram_aggregate_generation = None
        self.invalidate_semantic_level_evidence()
        self.first_pass_quality = None
        self.first_pass_histogram_published = False
        # The evidence scan indexed the OLD window's level key; restart the
        # counters so the new key's evidence is armed from scratch (the
        # renderer re-marks the scan whenever a commit parks on evidence).
        self.level_scan_cursor = 0
        self.level_scan_remaining_tiles = 0
        # Publish the successor target before cache hits enter the ordinary
        # materialization seam. A source retarget must first retire the old
        # source's evaluation/loading claims; mark_materialized then creates
        # the new source's loading obligation. Doing this in the opposite
        # order made lifecycle.retarget erase the successor claim it had just
        # created (the old implementation only passed by carrying the stale
        # predecessor claim across the source change).
        self.visible_tiles = tuple(visible_tiles)
        self.visible_tile_numbers = frozenset(
            int(tile.montage_index) for tile in self.visible_tiles
        )
        self._selected_lod_factor()
        self._sync_lifecycle_targets()
        hits = misses = unchanged = remapped = 0
        changed_slots: set[int] = set()
        plan_tiles_by_number = {
            int(tile.montage_index): tile for tile in tuple(getattr(plan, "tiles", ()) or ())
        }
        planned_numbers = set(plan_tiles_by_number)
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
            old_index = old_indices_by_source.get(new_source)
            if rendered is None and old_index is not None and old_index in old_rendered_tiles:
                rendered = replace(old_rendered_tiles[int(old_index)], tile=tile)
                remapped += 1
            resident_payload = None
            if rendered is None and old_index is not None:
                resident_payload = old_display_payloads.get(int(old_index))
                if resident_payload is None:
                    resident_payload = old_state_payloads.get(int(old_index))
            if rendered is not None:
                self.mark_materialized(rendered)
                if new_source is not None:
                    self.tile_source_ids[index] = new_source
                if new_source != old_source_ids.get(index):
                    changed_slots.add(index)
                    self.lifecycle.presentation_discarded(index)
                hits += 1
            elif resident_payload is not None:
                # Shared reduced transforms can own a complete current
                # payload without creating a renderer-local RenderedTile.
                # Index-window retargeting used to ignore that lifecycle
                # payload and key reuse only through ``rendered_tiles``. A
                # one-index scroll therefore rebuilt all 60 previews instead
                # of remapping the 59 overlapping semantic sources.
                # This branch deliberately has no current native RenderedTile.
                # Remove the predecessor slot entry before installing the
                # resident remap; otherwise a later presentation build sees
                # ``rendered_tiles[index]`` and overwrites the correct remap
                # with the old slot source (mixed +N source window at idle).
                self.rendered_tiles.pop(index, None)
                payload = replace(
                    resident_payload,
                    tile_number=index,
                    source_index=int(tile.source_index),
                )
                self.lifecycle.presentation_discarded(index)
                self.display_tile_payloads[index] = payload
                if new_source is not None:
                    self.tile_source_ids[index] = new_source
                self.lifecycle.remember_presentable(index, payload)
                self.record_tile_payload(payload)
                self.pending_payload_upserts[index] = None
                self.pending_removals.discard(index)
                self.skipped_tiles.discard(index)
                changed_slots.add(index)
                self.mark_tile_state(tile, MontageTileState.LOADED)
                remapped += 1
                hits += 1
            else:
                # Demote: the semantic slot is active for a new source, while
                # the backend may still hold the previous source's pixels.
                # Queue a cheap removal immediately; a same-cycle correct
                # resident/floor/exact upsert can still win in the delta.
                self.rendered_tiles.pop(index, None)
                self.tile_source_ids.pop(index, None)
                self.display_tile_payloads.pop(index, None)
                self.level_generation.forget_tile(index)
                changed_slots.add(index)
                self.lifecycle.presentation_discarded(index)
                self.skipped_tiles.discard(index)
                self.dirty_payloads[index] = None
                self.pending_removals.add(index)
                if 0 <= index < len(plan.tiles):
                    self.mark_tile_state(plan.tiles[index], MontageTileState.UNLOADED)
                misses += 1
        obsolete_slots = (
            {int(index) for index in old_rendered_tiles}
            | {int(index) for index in old_display_payloads}
            | {int(index) for index in old_state_payloads}
            | {int(index) for index in old_source_ids}
        ) - planned_numbers
        for index in sorted(obsolete_slots):
            self.rendered_tiles.pop(int(index), None)
            self.tile_source_ids.pop(int(index), None)
            self.display_tile_payloads.pop(int(index), None)
            self.level_generation.forget_tile(int(index))
            self.lifecycle.presentation_discarded(int(index))
            self.skipped_tiles.discard(int(index))
            self.dirty_payloads.pop(int(index), None)
            self.pending_payload_upserts.pop(int(index), None)
            self.pending_removals.discard(int(index))
        if obsolete_slots or changed_slots:
            # A CPU-windowed backend cannot expose a partially replaced source
            # window.  Frame effects consumes this backend-specific guard; it
            # is harmless on shader backends.  Crucially, ordinary montage
            # index scrolling changes semantic sources without changing layout
            # geometry, so the former geometry-only arming missed the exact
            # full -> subset -> full transition seen on screen.
            # Atomic retention is meaningful only while the predecessor and
            # successor share one physical slot topology.  When a montage
            # expands, shrinks, or changes layout, the old frame cannot own
            # the new slots.  Arming an atomic handoff in that case parked the
            # new geometry behind hidden residency for every successor tile
            # (field failure: 60 visible predecessors, 272 successor slots),
            # so neither backend could publish the honest progressive frame.
            # Same-topology source-window swaps hand off the frame-plan's
            # required on-screen scope atomically. Broader coverage slots are
            # not part of that completeness barrier.
            self.atomic_successor_pending = bool(
                had_complete_predecessor
                and old_plan_topology == _montage_plan_topology(plan)
                and bool(self.atomic_successor_required_scope())
            )
            retained_state = {
                int(tile): payload
                for tile, payload in old_state_payloads.items()
                if int(tile) in planned_numbers
                and int(tile) not in changed_slots
                and int(getattr(payload, "source_index", -1))
                == int(getattr(plan_tiles_by_number[int(tile)], "source_index", -2))
            }
            if retained_state != old_state_payloads:
                self.tile_presentation_state = TilePresentationState(
                    retained_state,
                    revision=int(getattr(self.tile_presentation_state, "revision", 0)),
                )
        self.sync_lifecycle_scope()
        self.update_level_presentation_scope()
        self.mark_ladder_swaps_for_viewport()
        # Retargeting is itself a presentation-producing mutation. A final
        # index/viewport change can arrive after the previous bounded commit
        # cleared these flags; leaving newly-created dirty/upsert/removal
        # obligations unarmed strands a semantically live session with no
        # backend pixels and no remaining kernel completion to wake it.
        if self.dirty_payloads or self.pending_payload_upserts or self.pending_removals:
            self.flush_pending = True
            self.final_commit_pending = True
        return {
            "hits": int(hits),
            "misses": int(misses),
            "unchanged": int(unchanged),
            "remapped": int(remapped),
        }

    def update_level_presentation_scope(self) -> None:
        presented_tiles = self.lifecycle.presented_tiles
        if not self.display_tile_payloads and not presented_tiles:
            return
        active = frozenset(
            int(tile.montage_index)
            for tile in tuple(self.visible_tiles)
            if int(tile.montage_index) in self.display_tile_payloads
            and int(tile.montage_index) in presented_tiles
            and _payload_has_level_presentation_evidence(
                self.display_tile_payloads[int(tile.montage_index)],
                shader_display=bool(getattr(self, "shader_display", False)),
            )
        )
        self.level_generation.set_active_tiles(active)
        if self.level_generation.target_levels is not None:
            snapshot = self.level_generation.snapshot(
                pending_upserts=tuple(self.pending_payload_upserts),
                active_tile_count=len(self.visible_tile_numbers),
            )
            self._level_update_pending = not bool(snapshot.settled)

    def begin_level_presentation_update(self, levels, *, source=None) -> bool:
        """Start or continue a progressive level generation.

        Histogram drags emit an immediate preview followed by a finish signal
        carrying the same numeric levels.  Treat that finish signal as a
        request to drain the existing generation, not as a new generation.
        Reissuing an already-settled target is a no-op.

        Returns ``True`` only while at least one currently presented tile still
        needs the requested levels.  A new target is still retained when no
        tile is active so subsequently materialized tiles inherit it.
        """

        previous_levels = self.level_generation.target_levels
        previous_revision = int(self.level_generation.revision)
        source = self.applied_level_source if source is None else source
        needs_work = ProgressiveTileLevelConvergence().begin(
            self.level_generation,
            levels,
            source=source,
            active_tiles=self.level_generation.active_tiles,
        )
        emit_trace(
            "level_target",
            session_id=int(self.session_id),
            previous_levels=previous_levels,
            target_levels=self.level_generation.target_levels,
            previous_revision=previous_revision,
            revision=int(self.level_generation.revision),
            source_rank=int(getattr(source, "rank", 0) or 0),
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

    def invalidate_semantic_level_evidence(self) -> None:
        """Drop scheduler progress when the semantic frame generation changes."""

        self.semantic_level_evidence_target = None
        self.semantic_level_evidence_progress = None

    def semantic_level_evidence_diagnostics(self) -> dict[str, object]:
        target = self.semantic_level_evidence_target
        progress = self.semantic_level_evidence_progress
        if target is None or progress is None:
            return {
                "target_population": 0,
                "covered_sources": (),
                "covered_source_count": 0,
                "pending_batches": 0,
                "inflight_generation": None,
                "blocking_reason": "inactive",
                "source_batch_limit": 0,
                "pixel_limit": 0,
            }
        return {
            "target_population": int(target.target_population),
            "covered_sources": tuple(progress.covered_sources_sample),
            "covered_source_count": len(progress.covered_sources),
            "pending_batches": int(progress.pending_batches),
            "inflight_generation": progress.inflight_generation,
            "blocking_reason": str(progress.blocking_reason),
            "source_batch_limit": int(progress.current_batch_limit),
            "pixel_limit": int(target.pixel_limit),
        }

    def set_level_update_pending(self, pending: bool) -> None:
        self._level_update_pending = bool(pending)

    def acknowledge_uniform_level_presentation(self, levels) -> None:
        """Accept one shader-level update for every active tiled surface.

        A shader backend changes one shared presentation uniform rather than
        redrawing individual payloads.  Record that as a single semantic
        acknowledgement while keeping per-tile values available for the same
        convergence diagnostics used by CPU-windowed backends.
        """

        UniformLevelConvergence().acknowledge(
            self.level_generation,
            target_revision=int(self.level_generation.revision),
            active_tiles=self.level_generation.active_tiles,
            levels=levels,
        )
        self._level_update_pending = not bool(
            self.level_generation.snapshot(
                pending_upserts=tuple(self.pending_payload_upserts),
                active_tile_count=len(self.visible_tile_numbers),
            ).settled
        )

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

    def mark_materialized(self, rendered: RenderedTile) -> None:
        index = int(rendered.tile.montage_index)
        self.lifecycle.target_invalidated(index)
        self.rendered_tiles[index] = rendered
        # A replacement tile must be assigned a fresh semantic source identity
        # and presentation payload on the next commit.  Other loaded tiles keep
        # their cached wrappers across progressive batches.
        self.tile_source_ids.pop(index, None)
        self.display_tile_payloads.pop(index, None)
        self.level_generation.forget_tile(index)
        self.dirty_payloads[index] = None
        self.flush_pending = True
        self.final_commit_pending = True
        self.stage_fan_in.tile_stage_keys.pop(index, None)
        self.stage_fan_in.detach_unbound_requests()
        self.active_tile_requests.discard(index)
        self.skipped_tiles.discard(index)
        self.loading_tiles.add(index)
        self.lifecycle.evaluation_completed(index)
        self.mark_tile_state(rendered.tile, MontageTileState.LOADING)

    def mark_presented(self, tile_numbers) -> None:
        # Collect level-scope additions and apply them once at the end:
        # extending the frozenset per presented tile makes a full-montage
        # commit O(n^2) in the tile count.
        level_scope_additions: list[int] = []
        confirmed: list[int] = []
        shown_identities = dict(self.lifecycle.backend_presented_identities)
        for tile_number in tuple(tile_numbers or ()):
            index = int(tile_number)
            payload = self.display_tile_payloads.get(index)
            preview_presented = (
                payload is not None and str(getattr(payload, "quality", "exact")) == "preview"
            )
            target_lod_presented = _payload_is_reduced_target(payload)
            if (
                index not in self.rendered_tiles
                and not preview_presented
                and not target_lod_presented
            ):
                continue
            if (
                shown_identities
                and payload is not None
                and shown_identities.get(index) != tile_ack_identity(payload)
            ):
                # Backend-active is not the same as current-presented.  PyQtGraph
                # can keep an item visible while it still holds the previous
                # payload identity; treating that as presented made scrolls settle
                # with stale/missing mixed-LOD slots.
                self.lifecycle.presentation_discarded(index)
                self.dirty_payloads[index] = None
                continue
            confirmed.append(index)
            if index not in self.lifecycle.presented_tiles and len(self.presented_order) < 64:
                self.presented_order.append(index)
            if index in self.visible_tile_numbers and index in self.display_tile_payloads:
                level_scope_additions.append(index)
                self.lifecycle.remember_presentable(index, self.display_tile_payloads[index])
                self.lifecycle.acknowledge_presented(
                    index,
                    tile_ack_identity(self.display_tile_payloads[index]),
                    str(getattr(self.display_tile_payloads[index], "quality", "exact") or "exact"),
                    int(
                        getattr(getattr(self.display_tile_payloads[index], "lod", None), "level", 0)
                        or 0
                    ),
                )
            if not (preview_presented and index in self.lifecycle.evaluating_tiles):
                self.loading_tiles.discard(index)
            self.skipped_tiles.discard(index)
            self.dirty_payloads.pop(index, None)
            if preview_presented and index in self.rendered_tiles:
                self.dirty_payloads[index] = None
            self.pending_removals.discard(index)
            if 0 <= index < len(self.plan.tiles):
                state = (
                    MontageTileState.LOADING
                    if preview_presented and index in self.lifecycle.evaluating_tiles
                    else MontageTileState.LOADED
                )
                self.mark_tile_state(self.plan.tiles[index], state)
        if confirmed:
            # ADR 0051 rule 1: the report's presented set is backend
            # acknowledgement, so resident-retarget commits (no upserts)
            # still reach the machine's PRESENTED state.
            self.lifecycle.presentation_confirmed(confirmed)
        if level_scope_additions:
            self.level_generation.set_active_tiles(
                (*self.level_generation.active_tiles, *level_scope_additions)
            )

    def snapshot_display_tile_payloads(
        self, source_ids: dict[int, object]
    ) -> dict[int, DisplayTilePayload]:
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
            self._ensure_display_tile_payload(
                int(tile_number), rendered, source_ids, lod_factor=lod_factor
            )
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
        base_source_id = source_ids.get(
            tile_number, ("rendered_tile", tile_number, id(rendered.image))
        )
        mapping = getattr(rendered, "shader_mapping", None)
        exact_image = np.asarray(rendered.image)
        exact_histogram = (
            None if rendered.histogram_data is None else np.asarray(rendered.histogram_data)
        )
        exact_level_data = (
            None
            if getattr(rendered, "level_data", None) is None
            else np.asarray(rendered.level_data)
        )
        level_stats = getattr(rendered, "level_stats", None)
        semantic = getattr(rendered, "semantic_data", None)
        semantic = exact_image if semantic is None else np.asarray(semantic)
        semantic_histogram = getattr(rendered, "semantic_histogram_data", None)
        semantic_histogram = (
            exact_histogram if semantic_histogram is None else np.asarray(semantic_histogram)
        )
        previous = self.display_tile_payloads.get(tile_number)
        acknowledged_previous = dict(
            getattr(self.tile_presentation_state, "payloads", {}) or {}
        ).get(tile_number)
        backend_identity = dict(self.lifecycle.backend_presented_identities).get(tile_number)
        if acknowledged_previous is not None and backend_identity != tile_ack_identity(
            acknowledged_previous
        ):
            acknowledged_previous = None
        # Quality preservation protects physical presentation truth, not a
        # wrapper merely built for a prior candidate.  An unacknowledged
        # native wrapper must not block a legitimate first reduced commit.
        preserve_candidate = acknowledged_previous
        if (
            preserve_candidate is not None
            and _base_source_id(preserve_candidate.source_id) == base_source_id
            and render_lod.preserve_finer_presented_payload(self, preserve_candidate)
        ):
            texture_data = np.asarray(
                preserve_candidate.texture_data
                if preserve_candidate.texture_data is not None
                else preserve_candidate.image
            )
            texture_histogram = preserve_candidate.histogram_data
            lod = preserve_candidate.lod
            texture_kind = preserve_candidate.texture_kind
            page_backing = preserve_candidate.page_backing
        else:
            texture_data, texture_histogram, lod, texture_kind, page_backing = (
                self._texture_for_rendered_tile(rendered)
            )
        texture_data = _debug_lod_pass_texture(texture_data, quality="exact")
        display_image = np.asarray(texture_data)
        display_histogram = None if texture_histogram is None else np.asarray(texture_histogram)
        source_id = self._payload_source_id(
            base_source_id,
            texture_kind=texture_kind,
            lod=lod,
        )
        if (
            previous is not None
            and _base_source_id(previous.source_id) == base_source_id
            and previous.source_id == source_id
            and previous.image is display_image
            and previous.histogram_data is display_histogram
            and previous.semantic_data is semantic
            and previous.semantic_histogram_data is semantic_histogram
            and previous.level_data is exact_level_data
            and previous.level_stats is level_stats
            and _shader_mapping_key(previous.shader_mapping) == _shader_mapping_key(mapping)
        ):
            current_identity = self.tile_payload_identity(
                rendered.tile,
                texture_data=texture_data,
                texture_kind=(
                    TexturePlaneKind.RGB8
                    if not bool(self.shader_display)
                    and texture_kind == TexturePlaneKind.COMPLEX_RG32F
                    else texture_kind
                ),
                shader_mapping=mapping,
                lod=lod,
                quality="exact",
            )
            if previous.tile_identity != current_identity:
                # Axis/flip changes reuse the same materialized plane, but the
                # backend acknowledgement must describe the new semantic view.
                previous = replace(previous, tile_identity=current_identity)
                self.display_tile_payloads[tile_number] = previous
            self.lifecycle.remember_presentable(tile_number, previous)
            self.record_tile_payload(previous)
            return previous
        if (
            previous is not None
            and _base_source_id(previous.source_id) == base_source_id
            and previous.semantic_data is semantic
            and int(getattr(getattr(previous, "lod", None), "level", 0) or 0) != int(lod.level)
        ):
            # Same native content presented at a different display-LOD level:
            # the level swap must be invisible to the histogram/level system.
            if (
                previous.semantic_histogram_data is semantic_histogram
                and previous.level_data is exact_level_data
                and previous.level_stats is level_stats
            ):
                self.lod_stats_cross_level_reuses += 1
            else:
                self.lod_stats_recomputes += 1
        payload = DisplayTilePayload(
            tile_number=tile_number,
            source_index=int(rendered.tile.source_index),
            image=display_image,
            histogram_data=display_histogram,
            source_id=source_id,
            texture_data=texture_data,
            texture_kind=texture_kind,
            semantic_data=semantic,
            semantic_histogram_data=semantic_histogram,
            source_shape=tuple(int(value) for value in exact_image.shape[:2]),
            lod=lod,
            shader_mapping=mapping,
            # The anchor rect is the NATIVE source extent the plane covers.
            # ``exact_image`` is already reduced on the ingest-reduction path,
            # so a reduced texture anchors by its LOD's native source shape
            # (ADR 0056 G5: reduced planes take chunked residency too).
            source_anchor=self._payload_source_anchor(
                lod.source_shape if lod is not None else exact_image.shape[:2]
            ),
            page_backing=page_backing,
            level_data=exact_level_data,
            level_stats=level_stats,
            tile_identity=self.tile_payload_identity(
                rendered.tile,
                texture_data=texture_data,
                texture_kind=(
                    TexturePlaneKind.RGB8
                    if not bool(self.shader_display)
                    and texture_kind == TexturePlaneKind.COMPLEX_RG32F
                    else texture_kind
                ),
                shader_mapping=mapping,
                lod=lod,
                quality="exact",
            ),
            presentation_identity=self.tile_presentation_identity(mapping),
        )
        self.display_tile_payloads[tile_number] = payload
        self.lifecycle.remember_presentable(tile_number, payload)
        self.record_tile_payload(payload)
        return payload

    def seed_display_tile_payloads(
        self,
        previous_payloads: dict[int, DisplayTilePayload],
        source_ids: dict[int, object],
        *,
        tile_numbers=None,
    ) -> None:
        """Reuse compatible wrappers without inventing presentation ownership.

        Retargeting a montage is a placement change, not evidence that resident
        tiles disappeared.  The backend's presented identity map is the only
        proof that a slot already holds the current payload; otherwise seeding
        only prepares a desired payload and forces an identity-checked upsert.
        Exact source identities are still required, so semantic changes do not
        reuse stale pixels.
        """

        if not previous_payloads or not self.rendered_tiles:
            return
        by_source = {payload.source_id: payload for payload in dict(previous_payloads).values()}
        # ADR 0050: a payload whose base (semantic) identity matches but whose
        # LOD level differs is still this tile's content, resident on the
        # backend.  Seeding keeps that payload available at the old level; the
        # LOD refresh then converges it through an ordinary identity-checked
        # swap.
        # Without this, a level change across sessions read as a black or
        # placeholder tile until fresh payload work committed.
        by_base: dict[object, DisplayTilePayload] = {}
        if self._resident_lod_active():
            for payload in dict(previous_payloads).values():
                base = _base_source_id(payload.source_id)
                if base is not None:
                    by_base.setdefault(base, payload)
        seeded_state = dict(getattr(self.tile_presentation_state, "payloads", {}) or {})
        backend_identities = dict(self.lifecycle.backend_presented_identities)
        changed_state = False
        confirmed_tiles: list[int] = []
        if tile_numbers is None:
            candidate_numbers = tuple(int(tile) for tile in self.rendered_tiles)
        else:
            candidate_numbers = tuple(
                int(tile) for tile in tuple(tile_numbers or ()) if int(tile) in self.rendered_tiles
            )
        for tile_number in candidate_numbers:
            rendered = self.rendered_tiles.get(int(tile_number))
            if rendered is None:
                continue
            tile_number = int(tile_number)
            base_source_id = source_ids.get(
                tile_number,
                ("rendered_tile", tile_number, id(rendered.image)),
            )
            previous = self.display_tile_payloads.get(tile_number)
            if previous is not None and _base_source_id(previous.source_id) != base_source_id:
                previous = None
            if previous is None:
                previous = by_base.get(base_source_id)
                if previous is None:
                    _texture_data, _texture_histogram, lod, texture_kind, _page_backing = (
                        self._texture_for_rendered_tile(rendered)
                    )
                    source_id = self._payload_source_id(
                        base_source_id,
                        texture_kind=texture_kind,
                        lod=lod,
                    )
                    previous = by_source.get(source_id)
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
                    tile_identity=self.tile_payload_identity(
                        rendered.tile,
                        texture_data=previous.texture_data,
                        texture_kind=previous.texture_kind,
                        shader_mapping=previous.shader_mapping,
                        lod=previous.lod,
                        quality=previous.quality,
                    ),
                )
            self.display_tile_payloads[tile_number] = payload
            self.record_tile_payload(payload)
            self.acknowledged_source_ids.add(payload.source_id)
            backend_confirms_payload = tile_number in backend_identities and backend_identities.get(
                tile_number
            ) == tile_ack_identity(payload)
            if backend_confirms_payload:
                if seeded_state.get(tile_number) is not payload:
                    seeded_state[tile_number] = payload
                    changed_state = True
                self.pending_payload_upserts.pop(tile_number, None)
                confirmed_tiles.append(tile_number)
            else:
                # Wrapper reuse is only desired payload state.  The lifecycle
                # remains unpresented until the backend reports this identity
                # in the slot, and the forced upsert keeps the remap visible
                # to the ordinary identity-checked commit path.
                self.pending_payload_upserts[tile_number] = None
            self.lifecycle.remember_presentable(tile_number, payload)
        if changed_state:
            self.tile_presentation_state = TilePresentationState(
                seeded_state,
                revision=int(getattr(self.tile_presentation_state, "revision", 0)),
            )
        if confirmed_tiles:
            self.lifecycle.presentation_confirmed(confirmed_tiles)
        if changed_state or confirmed_tiles:
            self.invalidate_tile_states()

    def _payload_source_id(
        self, base_source_id, *, texture_kind, lod: LodInfo
    ) -> tuple[object, ...]:
        prefix = tuple(base_source_id) if isinstance(base_source_id, tuple) else (base_source_id,)
        return (
            *prefix,
            "texture_kind",
            None if texture_kind is None else getattr(texture_kind, "value", texture_kind),
            "lod",
            int(lod.factor),
            int(lod.level),
            int(lod.gutter),
        )

    def tile_target_identity(self, tile: MontageTile, *, lod_level: int) -> TileIdentity:
        """Typed requested identity, independent of viewport and levels/LUT."""

        view_state = tile.view_state
        channel = getattr(
            getattr(view_state, "channel", None), "value", getattr(view_state, "channel", "real")
        )
        source_is_complex = np.issubdtype(np.dtype(self.output_dtype), np.complexfloating)
        shader_display = bool(getattr(self, "shader_display", False))
        if shader_display and source_is_complex:
            texture_kind = TexturePlaneKind.COMPLEX_RG32F
        elif str(channel) == "complex":
            texture_kind = TexturePlaneKind.RGB8
        else:
            texture_kind = TexturePlaneKind.SCALAR_R32F
        mapping = (
            ("phase_color", "abs", "mapped")
            if str(channel) == "complex"
            else ("scalar", str(channel), "mapped")
        )
        return self._tile_identity(
            tile,
            texture_kind=texture_kind,
            complex_mapping=mapping,
            lod=TileLodIdentity(level=lod_level, factor=1 << max(0, int(lod_level))),
            quality="exact",
        )

    def tile_target_identities(self, tile_numbers) -> dict[int, TileIdentity]:
        """Return lifecycle-owned typed targets for a presentation scope."""

        targets: dict[int, TileIdentity] = {}
        for tile_number in tuple(tile_numbers or ()):
            index = int(tile_number)
            record = self.lifecycle.peek(index)
            identity = None if record is None or record.target is None else record.target.identity
            if identity is not None:
                targets[index] = identity
        return targets

    def tile_payload_identity(
        self,
        tile: MontageTile,
        *,
        texture_data,
        texture_kind,
        shader_mapping,
        lod,
        quality: str,
    ) -> TileIdentity:
        texture_values = np.asarray(texture_data)
        if texture_kind is None:
            if np.iscomplexobj(texture_values) or (
                texture_values.ndim >= 3 and texture_values.shape[-1] == 2
            ):
                texture_kind = TexturePlaneKind.COMPLEX_RG32F
            elif texture_values.ndim >= 3 and texture_values.shape[-1] in (3, 4):
                texture_kind = TexturePlaneKind.RGB8
            else:
                texture_kind = TexturePlaneKind.SCALAR_R32F
        lod_identity = TileLodIdentity(
            level=int(getattr(lod, "level", 0) or 0),
            factor=int(getattr(lod, "factor", 1) or 1),
            gutter=int(getattr(lod, "gutter", 0) or 0),
        )
        real_plane, imag_plane = array_plane_identities(texture_values)
        mapping_identity = complex_mapping_identity(shader_mapping)
        if mapping_identity is None:
            mapping_identity = self.tile_target_identity(
                tile, lod_level=lod_identity.level
            ).complex_mapping
        return self._tile_identity(
            tile,
            texture_kind=texture_kind,
            complex_mapping=mapping_identity,
            lod=lod_identity,
            quality=quality,
            real_plane=real_plane,
            imag_plane=imag_plane,
        )

    def tile_presentation_identity(self, shader_mapping) -> TilePresentationIdentity:
        levels = getattr(getattr(self, "level_generation", None), "target_levels", None)
        return TilePresentationIdentity(
            levels_generation=int(getattr(self, "level_revision", 0) or 0),
            levels=levels,
            scale=getattr(shader_mapping, "scale", None),
            lut_identity=getattr(shader_mapping, "lut_identity", None),
        )

    def bind_payloads_to_level_generation(self) -> int:
        """Bind current wrappers to the uniform generation they will expose."""

        rebound = 0
        for tile_number in tuple(self.display_tile_payloads):
            payload = self.display_tile_payloads.get(tile_number)
            if payload is None:
                continue
            identity = self.tile_presentation_identity(payload.shader_mapping)
            if payload.presentation_identity == identity:
                continue
            payload = replace(payload, presentation_identity=identity)
            self.display_tile_payloads[tile_number] = payload
            self.lifecycle.remember_presentable(tile_number, payload)
            rebound += 1
        return int(rebound)

    def _tile_identity(
        self,
        tile: MontageTile,
        *,
        texture_kind,
        complex_mapping,
        lod: TileLodIdentity,
        quality: str,
        real_plane=None,
        imag_plane=None,
    ) -> TileIdentity:
        document = self.document
        base_data = getattr(document, "base_data", None)
        view_state = tile.view_state
        semantic_generation = (
            tuple(int(value) for value in getattr(view_state, "shape", ())),
            tuple(int(value) for value in getattr(view_state, "slice_indices", ())),
            tuple(getattr(view_state, "axis_range_indices", ()) or ()),
            tuple(bool(value) for value in getattr(view_state, "axis_fftshifted", ())),
        )
        return TileIdentity(
            document_generation=(id(base_data), int(getattr(document, "revision", 0) or 0)),
            operation_key=tuple(getattr(document, "steps", ()) or ()),
            source_index=int(tile.source_index),
            image_axes=tuple(int(axis) for axis in getattr(view_state, "image_axes", ()) or ()),
            axis_flips=tuple(
                bool(value) for value in getattr(view_state, "axis_flipped", ()) or ()
            ),
            channel=getattr(view_state, "channel", "real"),
            complex_mapping=complex_mapping,
            texture_kind=texture_kind,
            semantic_generation=semantic_generation,
            lod=lod,
            quality=quality,
            real_plane=real_plane,
            imag_plane=imag_plane,
        )

    def _selected_lod_factor(self) -> int:
        return render_lod.selected_lod_factor(self)

    def _resident_lod_active(self) -> bool:
        return render_lod.resident_lod_active(self)

    def presented_lod_summary(self) -> tuple[int, int, tuple[int, int]]:
        """Plurality-presented (level, factor, factor_xy); see :mod:`render_lod`."""

        return render_lod.presented_lod_summary(self)

    def ingest_lod_demand(self) -> object | None:
        """Demand snapshot for worker-side reduce-at-ingest; see :mod:`render_lod`."""

        return render_lod.ingest_lod_demand(self)

    def mark_ladder_swaps_for_viewport(self, *, refresh_demand: bool = True) -> bool:
        """Re-evaluate LOD demand after a camera-only retarget; see :mod:`render_lod`."""

        return render_lod.mark_ladder_swaps_for_viewport(
            self,
            refresh_demand=bool(refresh_demand),
        )

    def _session_resident_levels(self, previous_factor: int) -> tuple[int, ...]:
        return render_lod.session_resident_levels(self, previous_factor)

    def _payload_source_anchor(self, native_shape) -> object | None:
        """Window-invariant anchor for a non-montage exact payload (ADR 0055 G3).

        The exact plane covers the whole display window; its native source
        rect is the window start (anchored axes) plus the native extent. A
        non-anchored axis keeps start 0 — consistent, because that axis's
        window stays folded into the anchoring content key.
        """

        anchoring = self.source_anchoring
        if anchoring is None or self.montage_axis is not None:
            return None
        starts = getattr(anchoring, "anchored_starts", None)
        content_key = getattr(anchoring, "content_key", None)
        if starts is None or content_key is None:
            return None
        from arrayscope.display.model.frame import PayloadSourceAnchor

        y_start = int(starts[0] or 0)
        x_start = int(starts[1] or 0)
        height, width = (int(native_shape[0]), int(native_shape[1]))
        return PayloadSourceAnchor(
            content_key=content_key,
            source_rect=(y_start, y_start + height, x_start, x_start + width),
        )

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

    def _lod_page_set_key_for(self, rendered: RenderedTile, *, demand, level: int):
        return render_lod.page_set_key_for(self, rendered, demand=demand, level=level)

    def _lod_materialization_request(
        self,
        rendered: RenderedTile,
        *,
        demand,
        level: int,
        key,
        native_source: np.ndarray | None = None,
    ) -> LodPageMaterializationRequest:
        return render_lod.plan_materialization(
            self, rendered, demand=demand, level=level, key=key, native_source=native_source
        )

    def _best_floor_key(self, source_index: int, *, tile_number: int | None = None):
        return render_lod.best_floor_key(self, source_index, tile_number=tile_number)

    def _floor_can_progress(self, tile_number: int, tile=None) -> bool:
        return render_lod.floor_can_progress(self, tile_number, tile=tile)

    def _ensure_floor_payloads(self, tile_numbers, *, max_count: int | None = None) -> None:
        return render_lod.ensure_floor_payloads(self, tile_numbers, max_count=max_count)

    def preview_floor_metadata(self, key) -> PreviewFloorMetadata | None:
        return self.lod_preview_metadata.get(key)

    def admit_preview_plane(
        self,
        tile_number: int,
        key,
        plane,
        histogram=None,
        *,
        shader_mapping=None,
        texture_kind=None,
        level_data=None,
        level_stats=None,
        quality: str = "preview",
    ) -> bool:
        cache = self.lod_page_cache
        if cache is None:
            return False
        pages = tuple(plane) if isinstance(plane, (tuple, list)) else ()
        if not pages or any(not isinstance(page, MaterializedLodPage) for page in pages):
            raise TypeError("preview admission requires checked canonical materialized pages")
        if tuple(page.key for page in pages) != tuple(plan.key for plan in key.plans):
            raise ValueError("preview page values disagree with the requested canonical plan")
        owner = ("preview-page-admission", id(self), int(tile_number), key)
        claimed = cache.claim_plans(key.plans, owner)
        if claimed:
            if not cache.begin_owner_work(owner):
                cache.release_owner_claims(owner)
                raise RuntimeError("preview page claims disappeared before admission")
            claimed_keys = {plan.key for plan in claimed}
            try:
                for page in pages:
                    if page.key in claimed_keys:
                        cache.admit_as(page.key, page, owner=owner)
            finally:
                cache.finish_owner_work(owner)
        if not render_lod._page_set_exact(cache, key):
            return False
        metadata = PreviewFloorMetadata(
            shader_mapping=shader_mapping,
            texture_kind=texture_kind,
            level_data=None if level_data is None else np.asarray(level_data),
            level_stats=level_stats,
            quality=str(quality or "preview"),
        )
        if (
            any(
                value is not None
                for value in (
                    metadata.shader_mapping,
                    metadata.texture_kind,
                    metadata.level_data,
                    metadata.level_stats,
                )
            )
            or metadata.quality != "preview"
        ):
            self.lod_preview_metadata[key] = metadata
        rec = self.lifecycle.peek(int(tile_number))
        entry = None if rec is None else rec.levels.get(key)
        if entry is None or entry.owner is not ClaimOwner.PREVIEW:
            self.lifecycle.level_claimed(
                int(tile_number), key, ClaimOwner.PREVIEW, request=("preview-floor", key)
            )
        self.lifecycle.level_resident(int(tile_number), key)
        return True

    def release_preview_claim(self, tile_number: int, key) -> None:
        self.lifecycle.level_declined(int(tile_number), key)
        self.lod_preview_metadata.pop(key, None)

    def _lod_preview_floor_first_fill_active(self, planned_numbers) -> bool:
        """Whether a preview-floor fill is IN PROGRESS (scheduling preference).

        This governs which payloads the builder prefers to construct while
        preview claims are in flight — a wave-local scheduling fact. It is
        deliberately NOT the correctness contract: the required-generation
        phase belongs to ``scheduling_policy``. Conflating the two made the barrier wave-local
        (field report 2026-07-17: exact refinement marched across a black
        field, 192 preview acknowledgements after the first exact ack).
        """

        if not self._resident_lod_active():
            return False
        if not self.scheduling_policy.verdict.coverage_open:
            return False
        planned = tuple(int(tile) for tile in tuple(planned_numbers or ()))
        if not planned:
            return False
        return any(self._tile_preview_floor_pending(tile) for tile in planned)

    def _tile_preview_floor_pending(self, tile_number: int) -> bool:
        payloads = dict(getattr(self.tile_presentation_state, "payloads", {}) or {})
        payload = payloads.get(int(tile_number))
        if payload is not None and str(getattr(payload, "quality", "exact")) in {
            "preview",
            "exact",
        }:
            return False
        rec = self.lifecycle.peek(int(tile_number))
        if rec is None:
            return False
        for entry in rec.levels.values():
            if entry.owner is not ClaimOwner.PREVIEW:
                continue
            if entry.phase in (LevelPhase.CLAIMED, LevelPhase.MATERIALIZING, LevelPhase.RESIDENT):
                return True
        return False

    def _texture_source_for(
        self, rendered: RenderedTile
    ) -> tuple[np.ndarray, np.ndarray | None, TexturePlaneKind | None]:
        return texture_source_for_rendered(
            rendered,
            shader_display=bool(getattr(self, "shader_display", True)),
        )

    def _resident_texture_for_rendered_tile(
        self,
        rendered: RenderedTile,
        *,
        source: np.ndarray,
        histogram: np.ndarray | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        LodInfo,
        object | None,
        TexturePlaneKind | None,
    ]:
        return render_lod.resident_texture_for_rendered_tile(
            self, rendered, source=source, histogram=histogram
        )

    def _texture_for_rendered_tile(
        self,
        rendered: RenderedTile,
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        LodInfo,
        TexturePlaneKind | None,
        object | None,
    ]:
        source, histogram, texture_kind = self._texture_source_for(rendered)
        if self._resident_lod_active():
            texture, texture_histogram, lod, page_backing, actual_kind = (
                self._resident_texture_for_rendered_tile(
                    rendered,
                    source=source,
                    histogram=histogram,
                )
            )
            return texture, texture_histogram, lod, actual_kind, page_backing
        source_shape = tuple(int(value) for value in source.shape[:2])
        lod = LodInfo(
            level=0, factor=1, source_shape=source_shape, texture_shape=source_shape, gutter=0
        )
        return source, histogram, lod, texture_kind, None

    def _paced_pending_presentation_followup(
        self,
        *,
        cold_deadline_ms: float | None,
        max_upserts: int | None,
        max_upsert_bytes: int | None,
        upsert_cost_fn,
        physical_resident_fn,
        pace_resident_retargets: bool,
    ) -> tuple[TilePresentationState, TilePresentationDelta] | None:
        """Emit the next already-built backend transaction slice.

        A backend deadline may acknowledge only part of a proposed delta. The
        unacknowledged payloads remain in ``pending_payload_upserts``. Once
        structure and visibility are stable, rebuilding the complete semantic
        transaction before every follow-up is both redundant and O(grid size).
        This path consumes only payloads the lifecycle already owns; any
        structural/removal/level ambiguity falls back to full reconciliation.
        """

        if max_upserts is None:
            return None
        if not self.pending_payload_upserts:
            return None
        if (
            self.pending_removals
            or self.has_pending_level_update()
            or self.has_stale_level_presentations()
        ):
            return None
        if bool(getattr(self, "_layout_geometry_changed_pending", False)):
            return None
        planned = tuple(
            dict.fromkeys(
                int(tile.montage_index)
                for tile in tuple(self.visible_tiles)
                if int(tile.montage_index) not in self.skipped_tiles
            )
        )
        if not planned or planned != tuple(self._last_planned_tiles):
            return None
        active = tuple(self._last_active_tiles)
        if not active:
            return None
        active_set = set(active)
        plan_tiles_by_number = {
            int(tile.montage_index): tile for tile in tuple(getattr(self.plan, "tiles", ()) or ())
        }
        candidate_numbers = tuple(dict.fromkeys(int(tile) for tile in self.pending_payload_upserts))
        candidate_numbers = tuple(
            tile
            for tile in candidate_numbers
            if tile in active_set
            and _payload_matches_current_tile(
                self,
                int(tile),
                self.display_tile_payloads.get(int(tile)),
                plan_tiles_by_number,
            )
        )
        unpresented = active_set.difference(self.lifecycle.presented_tiles)
        if unpresented:
            coverage_candidates = tuple(
                tile for tile in candidate_numbers if int(tile) in unpresented
            )
            if not coverage_candidates:
                # The cached follow-up can only drain wrappers built by the
                # previous transaction.  When visible holes still exist but
                # none of those wrappers covers a hole, fall through to the
                # full builder so it spends its bounded build budget creating
                # missing first pixels.  Draining preview upgrades here
                # starved TARGET_READY edge tiles indefinitely.
                return None
            candidate_numbers = coverage_candidates
        if not candidate_numbers:
            # Slot numbers remain 0..N across an index-window scroll. A cached
            # follow-up containing the predecessor source mapping is therefore
            # not reusable merely because ``planned == _last_planned_tiles``;
            # fall through to the identity-aware full builder, which repairs
            # stale wrappers and mappings.
            return None
        ordered = self._prioritized_tile_numbers(candidate_numbers)
        payloads = {int(tile): self.display_tile_payloads[int(tile)] for tile in ordered}
        if not payloads:
            return None
        resident_retargets = {
            int(tile)
            for tile, payload in payloads.items()
            if payload.source_id in self.acknowledged_source_ids
        }
        free_retarget_tiles = _free_retarget_tiles(
            payloads,
            logical_resident_tiles=resident_retargets,
            physical_resident_fn=physical_resident_fn,
            pace_resident_retargets=pace_resident_retargets,
        )
        payloads = _physical_rebind_transaction(
            payloads,
            free_retarget_tiles=free_retarget_tiles,
            physical_resident_fn=physical_resident_fn,
        )
        plan_tiles_by_number = {
            int(tile.montage_index): tile for tile in tuple(getattr(self.plan, "tiles", ()) or ())
        }
        admission_candidates = tuple(
            plan_tiles_by_number.get(int(tile), int(tile)) for tile in payloads
        )
        admission = TileAdmissionQueue(self.tile_priority_context()).admit(
            admission_candidates,
            retained=(),
            free_fn=(lambda tile: int(tile) in free_retarget_tiles)
            if free_retarget_tiles
            else None,
            cost_fn=(
                lambda tile: (
                    0
                    if int(tile) in free_retarget_tiles
                    else int(
                        upsert_cost_fn(payloads[int(tile)])
                        if upsert_cost_fn is not None
                        else getattr(payloads[int(tile)], "nbytes", 0) or 0
                    )
                )
            ),
            max_items=max_upserts,
            max_bytes=max_upsert_bytes,
            deadline_ms=cold_deadline_ms,
        )
        upserts = {
            int(tile): payloads[int(tile)] for tile in admission.admitted if int(tile) in payloads
        }
        if not upserts:
            return None
        previous_state = self.tile_presentation_state
        self.payload_revision += 1
        near = tuple(self._last_near_tiles)
        near_source_ids = {
            int(tile): self.display_tile_payloads[int(tile)].source_id
            for tile in near
            if int(tile) in self.display_tile_payloads
        }
        delta = TilePresentationDelta(
            structure_revision=self.structure_revision,
            payload_revision=self.payload_revision,
            visibility_revision=self.visibility_revision,
            level_revision=self.level_revision,
            histogram_revision=self.histogram_revision,
            viewport_revision=self.viewport_revision,
            base_revision=int(previous_state.revision),
            target_revision=int(previous_state.revision) + 1,
            transaction_generation=int(self.session_id),
            cold_deadline_ms=cold_deadline_ms,
            upserts=upserts,
            active_tiles=active,
            planned_tiles=planned,
            near_tiles=near,
            near_tile_source_ids=near_source_ids,
            target_identities=self.tile_target_identities(active),
        )
        return previous_state.apply_delta(delta), delta

    def build_tile_presentation(
        self,
        source_ids: dict[int, object] | None,
        *,
        cold_deadline_ms: float | None = None,
        max_upserts: int | None = None,
        max_upsert_bytes: int | None = None,
        upsert_cost_fn=None,
        physical_resident_fn=None,
        pace_resident_retargets: bool = False,
    ) -> tuple[TilePresentationState, TilePresentationDelta]:
        source_ids = dict(source_ids or {})
        pending_followup = self._paced_pending_presentation_followup(
            cold_deadline_ms=cold_deadline_ms,
            max_upserts=max_upserts,
            max_upsert_bytes=max_upsert_bytes,
            upsert_cost_fn=upsert_cost_fn,
            physical_resident_fn=physical_resident_fn,
            pace_resident_retargets=pace_resident_retargets,
        )
        if pending_followup is not None:
            return pending_followup
        previous_state = self.tile_presentation_state
        previous_payloads = dict(previous_state.payloads)
        materialized = {int(index) for index in self.rendered_tiles}
        planned_numbers = set(self.visible_tile_numbers).difference(self.skipped_tiles)
        plan_tiles_cache = getattr(self, "_plan_tiles_by_number_cache", None)
        if plan_tiles_cache is None or plan_tiles_cache[0] is not self.plan:
            plan_tiles_by_number = {
                int(tile.montage_index): tile
                for tile in tuple(getattr(self.plan, "tiles", ()) or ())
            }
            self._plan_tiles_by_number_cache = (self.plan, plan_tiles_by_number)
        else:
            plan_tiles_by_number = plan_tiles_cache[1]
        dirty_numbers = {int(tile) for tile in self.dirty_payloads}
        dirty_numbers.update(int(tile) for tile in self.pending_payload_upserts)
        reconcile_numbers = dirty_numbers.intersection(planned_numbers)
        reconcile_numbers.update(planned_numbers.difference(previous_payloads))
        # Lifecycle owns target/fallback readiness. Shared reduced transforms
        # produce valid target payloads without creating native per-tile
        # ``rendered_tiles`` entries; treating that renderer-local map as the
        # readiness authority removed every such payload from the commit.
        for tile_number in reconcile_numbers:
            current = self.display_tile_payloads.get(int(tile_number))
            if self.lifecycle.payload_is_current(int(tile_number), current):
                materialized.add(int(tile_number))
                continue
            payload = self.lifecycle.current_presentable_payload(int(tile_number))
            if payload is None or getattr(payload, "source_id", None) is None:
                continue
            self.display_tile_payloads[int(tile_number)] = payload
            materialized.add(int(tile_number))
        for stale in tuple(self.display_tile_payloads):
            if int(stale) not in materialized:
                payload = self.display_tile_payloads.get(int(stale))
                if (
                    payload is not None
                    and int(stale) in planned_numbers
                    and self.lifecycle.payload_is_current(int(stale), payload)
                ):
                    # Atomic presentation: a planned tile with compatible
                    # preview/exact pixels keeps them until a replacement is
                    # acknowledged; dirty means "prepare replacement", not
                    # "remove the visible slot".
                    self.lifecycle.remember_presentable(int(stale), payload)
                    continue
                self.display_tile_payloads.pop(int(stale), None)
                self.pending_removals.add(int(stale))
                self.dirty_payloads.pop(int(stale), None)
                self.pending_payload_upserts.pop(int(stale), None)
                self.level_generation.forget_tile(int(stale))
        for tile_number, payload in tuple(self.display_tile_payloads.items()):
            index = int(tile_number)
            if index not in materialized:
                continue
            tile = plan_tiles_by_number.get(index)
            if tile is not None and int(getattr(payload, "source_index", -1)) == int(
                tile.source_index
            ):
                continue
            self.display_tile_payloads.pop(index, None)
            self.tile_source_ids.pop(index, None)
            self.level_generation.forget_tile(index)
            self.dirty_payloads[index] = None
        lod_factor = self._selected_lod_factor()
        current_loaded = set(self.rendered_tiles)
        planned = tuple(
            dict.fromkeys(
                int(tile.montage_index)
                for tile in tuple(self.visible_tiles)
                if int(tile.montage_index) in planned_numbers
            )
        )
        # ``active_tiles`` is the visible presentation scope, not the subset
        # whose successor payload happens to be ready in this drain.  A
        # persistent backend uses active scope to decide which existing slot
        # mappings remain drawn; shrinking it to ``current_loaded`` made a
        # complete predecessor montage become 12/24/... tiles while a new
        # operation or LOD materialized.  Payload absence means "retain the
        # compatible committed slot", never "hide this visible tile".
        active = tuple(int(tile) for tile in planned)
        stale_level_tiles = ()
        if self.has_pending_level_update():
            # Filter before prioritizing: ordering every active tile per commit
            # makes the drain of a large stale backlog O(n^2) in commits.
            stale_candidates = tuple(
                int(tile)
                for tile in active
                if int(tile) in self.display_tile_payloads
                and int(tile) in previous_payloads
                and str(getattr(self.display_tile_payloads[int(tile)], "quality", "exact"))
                == "exact"
                and not self._tile_matches_current_level_target(
                    int(tile), self.level_generation.target_levels
                )
            )
            stale_level_tiles = self._prioritized_tile_numbers(stale_candidates)
        # Parked dirty entries re-arm when their tile enters the active
        # scope (see acknowledge_tile_presentation: a non-active upsert the
        # backend declines parks instead of re-arming, or finalization would
        # retry an unacceptable upsert forever).
        # Backend slot identity is physical screen evidence.  A drawn tile
        # whose slot identity differs from its current payload is re-emitted
        # through the bounded commit path; repeated identical mismatches stop
        # retrying and remain visible in diagnostics.
        stale_drawn: list[int] = []
        stale_identity_removals: set[int] = set()
        backend_identities = dict(self.lifecycle.backend_presented_identities)
        if backend_identities:
            # Backend truth can be ahead of the session's acknowledged
            # TilePresentationState when a geometry/visibility commit reports
            # already-current resident slots without committed upserts.  If we
            # leave that split in place, the next VisPy commit receives an
            # active tile with no active payload and clears a perfectly correct
            # atlas mapping.  Rehydrate the acknowledged state from the one
            # allowed source of truth: backend identity == current payload.
            retained_payloads = None
            retained_tiles: list[int] = []
            for tile_number in planned_numbers:
                current = self.display_tile_payloads.get(int(tile_number))
                if current is None:
                    continue
                if backend_identities.get(int(tile_number)) != tile_ack_identity(current):
                    continue
                if previous_payloads.get(int(tile_number)) is not current:
                    if retained_payloads is None:
                        retained_payloads = dict(previous_payloads)
                    retained_payloads[int(tile_number)] = current
                    previous_payloads[int(tile_number)] = current
                retained_tiles.append(int(tile_number))
            # The same canonical pass also consumes stale pending-upsert
            # markers. A capped follow-up commit must not trickle an already
            # presented coarse/startup floor back through the cold upload cap
            # (field trace 2026-07-09: 272 presented dropped to 28/47/88).
            for tile_number in retained_tiles:
                if _preview_upgrade_owed(
                    self, int(tile_number), self.display_tile_payloads.get(int(tile_number))
                ):
                    self.dirty_payloads[int(tile_number)] = None
                    continue
                self.pending_payload_upserts.pop(int(tile_number), None)
            if retained_payloads is not None:
                previous_state = TilePresentationState(
                    retained_payloads,
                    revision=int(getattr(previous_state, "revision", 0)),
                )
                self.tile_presentation_state = previous_state
            if retained_tiles:
                self.lifecycle.presentation_confirmed(retained_tiles)
        if backend_identities:
            for tile_number, shown_identity in backend_identities.items():
                current = self.display_tile_payloads.get(int(tile_number))
                acknowledged = previous_payloads.get(int(tile_number))
                plan_tile = None
                if 0 <= int(tile_number) < len(getattr(self.plan, "tiles", ()) or ()):
                    plan_tile = self.plan.tiles[int(tile_number)]
                acknowledged_stale_for_plan = (
                    acknowledged is not None
                    and plan_tile is not None
                    and int(getattr(acknowledged, "source_index", -1))
                    != int(plan_tile.source_index)
                )
                if current is None and acknowledged_stale_for_plan:
                    self._identity_retry_attempts.pop(int(tile_number), None)
                    self.lifecycle.presentation_discarded(int(tile_number))
                    stale_identity_removals.add(int(tile_number))
                    if int(tile_number) in self.rendered_tiles:
                        self.display_tile_payloads.pop(int(tile_number), None)
                        self.dirty_payloads[int(tile_number)] = None
                        self.pending_payload_upserts[int(tile_number)] = None
                        stale_drawn.append(int(tile_number))
                    continue
                current_identity = None if current is None else tile_ack_identity(current)
                if current is None or current_identity == shown_identity:
                    self._identity_retry_attempts.pop(int(tile_number), None)
                    continue
                rec = self.lifecycle.peek(int(tile_number))
                if rec is not None and (current_identity, shown_identity) in rec.resigned:
                    # P2: the machine resigned exactly this (wanted, shown)
                    # pair after bounded identity rejections — the backend
                    # would not converge; re-presenting would reopen the loop
                    # it bounded.  Any OTHER mismatch (e.g. payload moved on
                    # without an emit, or a rebuilt session inheriting stale
                    # slots) is a legitimate repair and proceeds below.
                    self._identity_retry_attempts.pop(int(tile_number), None)
                    continue
                pair = (shown_identity, current_identity)
                prior_pair, attempts = self._identity_retry_attempts.get(
                    int(tile_number), (None, 0)
                )
                if prior_pair != pair:
                    attempts = 0
                if attempts >= 3:
                    continue
                self._identity_retry_attempts[int(tile_number)] = (pair, attempts + 1)
                if int(tile_number) not in planned_numbers or self.lifecycle.may_remove_visible(
                    int(tile_number)
                ):
                    self.lifecycle.presentation_discarded(int(tile_number))
                    stale_identity_removals.add(int(tile_number))
                stale_drawn.append(int(tile_number))
        else:
            for tile_number, acknowledged in previous_payloads.items():
                current = self.display_tile_payloads.get(int(tile_number))
                if current is None or current is acknowledged:
                    continue
                if tile_ack_identity(current) != tile_ack_identity(acknowledged):
                    if int(tile_number) not in planned_numbers or self.lifecycle.may_remove_visible(
                        int(tile_number)
                    ):
                        self.lifecycle.presentation_discarded(int(tile_number))
                        stale_identity_removals.add(int(tile_number))
                    stale_drawn.append(int(tile_number))
        if stale_drawn:
            for tile_number in sorted(stale_drawn):
                self.dirty_payloads[int(tile_number)] = None
                # Force the upsert: session bookkeeping may consider this
                # identity already presented (that bookkeeping is exactly
                # what the backend truth contradicts).
                self.pending_payload_upserts[int(tile_number)] = None
            active = tuple(dict.fromkeys((*active, *sorted(stale_drawn))))
        # A viewport can move after a payload becomes ready but before its
        # queued upsert is emitted.  The result remains reusable in the tile
        # and page caches; its old presentation obligation does not.  Keeping
        # that obligation outside the new active scope leaves commit debt with
        # no admissible backend transaction and wedges finalization at idle.
        active_scope = {int(tile) for tile in active}
        for tile_number in set(self.pending_payload_upserts) - active_scope:
            self.dirty_payloads.pop(int(tile_number), None)
            self.pending_payload_upserts.pop(int(tile_number), None)
        # ADR 0051: the lifecycle machine owns park/re-arm; this is its
        # rule-3 scope event.
        for tile_number in self.lifecycle.rearm_for_scope(active_scope):
            self.dirty_payloads[int(tile_number)] = None
        preview_upgrade_tiles = tuple(
            int(tile)
            for tile in active
            if _preview_upgrade_owed(self, int(tile), self.display_tile_payloads.get(int(tile)))
        )
        for tile_number in preview_upgrade_tiles:
            self.dirty_payloads[int(tile_number)] = None
        missing_payload_tiles = tuple(
            int(tile) for tile in active if int(tile) not in self.display_tile_payloads
        )
        for tile_number in missing_payload_tiles:
            self.dirty_payloads[int(tile_number)] = None
        unpresented_tiles = tuple(
            int(tile)
            for tile in active
            if (
                (record := self.lifecycle.peek(int(tile))) is None
                or not record.first_pixel_presented
            )
        )
        self._rearm_required_first_pixel_payloads()
        priority_context = self.tile_priority_context()

        def prioritize(tiles) -> tuple[int, ...]:
            return prioritize_tile_numbers(
                tiles,
                plan_tiles=tuple(getattr(self.plan, "tiles", ()) or ()),
                context=priority_context,
            )

        lifecycle_change_tiles = tuple(
            int(command.tile_number) for command in self.lifecycle.presentation_changes()
        )
        dirty_payload_tiles = tuple(
            dict.fromkeys(
                (
                    *(int(tile) for tile in self.dirty_payloads),
                    *(int(tile) for tile in self.pending_payload_upserts),
                    *lifecycle_change_tiles,
                    *missing_payload_tiles,
                    *preview_upgrade_tiles,
                    *stale_level_tiles,
                )
            )
        )
        if max_upserts is not None or max_upsert_bytes is not None or stale_level_tiles:
            dirty_payload_tiles = prioritize(dirty_payload_tiles)
        # Payload construction is bounded by the same admission budget that
        # caps uploads: a retarget/scrub step marks every tile dirty, and
        # building all N wrappers synchronously before admitting 4 of them
        # made the budgeted commit O(N) anyway (session-rebirth cost, ADR
        # 0051 P2).  Unbuilt tiles keep their dirty entry — the next commit
        # continues in priority order.  Slightly over-build so byte-capped
        # admission still has choices.
        build_limit = None
        if max_upserts is not None:
            # Build only what this transaction can admit. Over-building 2x
            # made a nominal 12-item VisPy slice spend 40 ms reducing/wrapping
            # 24 tiles before a 20 ms backend apply, breaching the 50 ms GUI
            # hard gate without improving committed throughput.
            build_limit = max(int(max_upserts), 8)
        floor_first_fill_active = self._lod_preview_floor_first_fill_active(planned_numbers)
        if floor_first_fill_active:
            floor_payload_tiles = tuple(planned_numbers)
            if max_upserts is not None:
                floor_payload_tiles = prioritize(floor_payload_tiles)
            self._ensure_floor_payloads(
                floor_payload_tiles,
                max_count=build_limit if pace_resident_retargets else None,
            )
        built = 0
        for tile_number in dirty_payload_tiles:
            if build_limit is not None and built >= build_limit:
                break
            if floor_first_fill_active and int(tile_number) in planned_numbers:
                tile = plan_tiles_by_number.get(int(tile_number))
                floor_payload = self.display_tile_payloads.get(int(tile_number))
                floor_owned = bool(
                    _payload_matches_current_tile(
                        self,
                        int(tile_number),
                        floor_payload,
                        plan_tiles_by_number,
                    )
                    and str(getattr(floor_payload, "quality", "") or "") in {"preview", "fallback"}
                ) or self._floor_can_progress(int(tile_number), tile=tile)
                if floor_owned or not bool(self.atomic_successor_pending):
                    continue
                # Ground rule 11: an atomic successor may wait only when the
                # missing complement has a live owner.  Floor-first is a
                # frame-wide visual preference, but floor residency is
                # per-tile.  A rendered tile with no resolvable floor has no
                # preview producer for the ladder to schedule; withholding
                # its native wrapper here leaves the atomic builder at
                # N-k/N forever.  Use the already-owned native result for
                # that tile while the rest of the successor keeps its coarse
                # floors.  Later refinement still follows the ordinary rung
                # path.
            rendered = self.rendered_tiles.get(int(tile_number))
            if rendered is not None:
                self._ensure_display_tile_payload(
                    int(tile_number), rendered, source_ids, lod_factor=lod_factor
                )
                built += 1
        if not floor_first_fill_active:
            floor_payload_tiles = tuple(planned_numbers - set(current_loaded))
            floor_build_limit = None
            if max_upserts is not None:
                floor_payload_tiles = prioritize(floor_payload_tiles)
                floor_build_limit = build_limit if pace_resident_retargets else None
            self._ensure_floor_payloads(floor_payload_tiles, max_count=floor_build_limit)
        unpresented_active_tiles = tuple(
            int(tile)
            for tile in active
            if int(tile) not in self.lifecycle.presented_tiles
            and _force_unpresented_upsert(
                self,
                int(tile),
                previous_payloads=previous_payloads,
                backend_identities=backend_identities,
            )
        )
        for tile_number in unpresented_active_tiles:
            self.dirty_payloads[int(tile_number)] = None
            self.pending_payload_upserts[int(tile_number)] = None
        dirty_payload_tiles = tuple(
            dict.fromkeys(
                (
                    *dirty_payload_tiles,
                    *(int(tile) for tile in self.pending_payload_upserts),
                )
            )
        )
        if max_upserts is not None or max_upsert_bytes is not None or stale_level_tiles:
            dirty_payload_tiles = prioritize(dirty_payload_tiles)
        presented_preview_tiles = tuple(
            int(tile)
            for tile in planned_numbers
            if int(tile) in self.lifecycle.presented_tiles
            and str(getattr(self.display_tile_payloads.get(int(tile)), "quality", "exact"))
            == "preview"
        )
        floor_active_tiles = tuple(
            int(tile)
            for tile in self.pending_payload_upserts
            if int(tile) in planned_numbers
            and str(getattr(self.display_tile_payloads.get(int(tile)), "quality", "exact"))
            == "preview"
        )
        if floor_active_tiles or presented_preview_tiles:
            active = tuple(dict.fromkeys((*active, *presented_preview_tiles, *floor_active_tiles)))
        # Progress guarantee: a pending upsert is payload-backed.  A retained
        # stale backend slot may be visibly present while the replacement exact
        # tile is still evaluating, but without a current desired payload there
        # is nothing a commit can emit.  Keep the semantic loading claim, not a
        # fake upsert backlog.
        for tile_number in tuple(self.pending_payload_upserts):
            if int(tile_number) not in self.display_tile_payloads:
                self.pending_payload_upserts.pop(int(tile_number), None)
        # A dirty entry is a promise that a build can produce an upsert for the
        # tile.  Without a rendered result and without a display payload (floor
        # included), no build can keep that promise — dropping the entry lets
        # the commit loop settle while the evaluation claim remains visible.
        # A later rendered result re-dirties the tile through mark_materialized.
        for tile_number in tuple(self.dirty_payloads):
            if (
                int(tile_number) not in self.rendered_tiles
                and int(tile_number) not in self.display_tile_payloads
            ):
                if self._floor_can_progress(int(tile_number)):
                    continue
                self.dirty_payloads.pop(int(tile_number), None)
        current_payloads = self.display_tile_payloads
        # Payload creation can occur later in this same build (for example a
        # shared reduced target admitted by the LOD ladder). Reconcile queued
        # removals at the transaction boundary, after every payload-building
        # step, so a newly current lifecycle payload wins atomically even when
        # its upsert is deferred by the per-commit cap.
        for tile_number in planned_numbers:
            if self.lifecycle.payload_is_current(
                int(tile_number), current_payloads.get(int(tile_number))
            ):
                self.pending_removals.discard(int(tile_number))
        valid_tile_count = len(tuple(getattr(self.plan, "tiles", ()) or ()))
        physical_pending_removals = {
            int(tile)
            for tile in self.pending_removals
            if int(tile) not in planned_numbers or self.lifecycle.may_remove_visible(int(tile))
        }
        self.pending_removals.intersection_update(physical_pending_removals)
        physical_stale_removals = {
            int(tile)
            for tile in stale_identity_removals
            if int(tile) not in planned_numbers or self.lifecycle.may_remove_visible(int(tile))
        }
        removals = tuple(
            sorted(
                {
                    int(tile)
                    for tile in previous_payloads
                    if int(tile) < 0
                    or int(tile) >= valid_tile_count
                    or int(tile) in self.skipped_tiles
                }.union(physical_pending_removals).union(physical_stale_removals)
            )
        )
        upserts: dict[int, DisplayTilePayload] = {}
        for tile_number in dirty_payload_tiles:
            payload = current_payloads.get(int(tile_number))
            if payload is None:
                continue
            shown_identity = backend_identities.get(int(tile_number))
            rec = self.lifecycle.peek(int(tile_number))
            if (
                shown_identity is not None
                and rec is not None
                and (tile_ack_identity(payload), shown_identity) in rec.resigned
            ):
                # The lifecycle already bounded retries for exactly this
                # wanted/shown pair. Coverage admission must not reopen that
                # loop merely because the tile is semantically unpresented.
                continue
            previous = previous_payloads.get(int(tile_number))
            force_upsert = (
                int(tile_number) in self.pending_payload_upserts
                or int(tile_number) in stale_level_tiles
            )
            if previous is payload and not force_upsert:
                if _preview_upgrade_owed(self, int(tile_number), payload):
                    self.dirty_payloads[int(tile_number)] = None
                    continue
                # The dirty record's promise is already kept: the presented
                # payload IS this payload.  Popping the record here is what
                # lets the commit loop settle — leaving it made every commit
                # a no-op that never cleared it (2026-07-05 probe stall:
                # 19 dirty tiles at idle re-armed each build, watchdog
                # firing every second; ADR 0051 rule 3).
                self.dirty_payloads.pop(int(tile_number), None)
                continue
            if (
                not force_upsert
                and previous is not None
                and previous.source_id == payload.source_id
                and _shader_mapping_key(previous.shader_mapping)
                == _shader_mapping_key(payload.shader_mapping)
            ):
                # ``source_id`` is the canonical materialization identity.
                # Rebuilt Python wrappers (or separately sampled histogram
                # arrays) do not mean new backend pixels; requiring object
                # identity here re-emitted the same acknowledged preview on
                # every upgrade replan and sustained an infinite commit/draw
                # loop. New pixel content must carry a new source identity.
                if _preview_upgrade_owed(self, int(tile_number), payload):
                    self.dirty_payloads[int(tile_number)] = None
                    continue
                self.dirty_payloads.pop(int(tile_number), None)
                continue
            upserts[int(tile_number)] = payload

        resident_retarget_tiles = _resident_retarget_upsert_tiles(upserts, previous_payloads)
        # Level swaps re-presenting a backend-acknowledged identity are
        # residency remaps, not uploads: they never charge the byte budget
        # (ADR 0050).  Whether they also bypass the ITEM cap is backend
        # truth, not a fixed rule (decided 2026-07-06): on a persistent
        # GPU-residency backend a remap is instant, so a burst of swaps
        # converges in one commit; on backends that rebuild items per
        # re-level (PyQtGraph) the caller passes
        # `pace_resident_retargets=True` and remaps stream through the item
        # cap in priority order like everything else.
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
        cold_upserts = {
            int(tile): payload
            for tile, payload in upserts.items()
            if int(tile) not in resident_retarget_tiles
        }
        active_set = {int(tile) for tile in active}
        all_candidate_upserts = {
            int(tile): payload for tile, payload in upserts.items() if int(tile) in active_set
        }
        # Presentation never withholds better ready data (progressive
        # presentation contract, 2026-07-18): the preview/refinement split is
        # enforced where COMPUTE is submitted (the ladder's phase barrier),
        # never here at commit.
        required = set(self.required_tile_numbers())
        required_unsettled = set(self.lifecycle.target_unsettled_tiles(required))
        if self.display_committed and required_unsettled:
            # Coverage-ring payloads remain drawn/retained, but they may not
            # consume a backend transaction while an actual on-screen target
            # is unfinished. Otherwise a small zoom target shared an 8-item
            # upload slice with five near tiles and could sit stale for seconds
            # despite all required pixels already being materialized.
            all_candidate_upserts = {
                int(tile): payload
                for tile, payload in all_candidate_upserts.items()
                if int(tile) in required
            }
            cold_upserts = {
                int(tile): payload
                for tile, payload in cold_upserts.items()
                if int(tile) in required
            }
            resident_retarget_tiles.intersection_update(required)
        coverage_upserts = {
            int(tile): payload
            for tile, payload in all_candidate_upserts.items()
            if int(tile) in set(unpresented_tiles)
        }
        if coverage_upserts:
            # First-pixel coverage is a transaction class, not merely a
            # priority hint. Mixing it with upgrades allowed center-focused
            # replacement work to consume every bounded batch while edge
            # slots remained absent for hundreds of commits.
            all_candidate_upserts = coverage_upserts
            cold_upserts = {
                int(tile): payload
                for tile, payload in cold_upserts.items()
                if int(tile) in coverage_upserts
            }
            resident_retarget_tiles.intersection_update(coverage_upserts)
        free_retarget_tiles = _free_retarget_tiles(
            all_candidate_upserts,
            logical_resident_tiles=resident_retarget_tiles,
            physical_resident_fn=physical_resident_fn,
            pace_resident_retargets=pace_resident_retargets,
        )
        all_candidate_upserts = _physical_rebind_transaction(
            all_candidate_upserts,
            free_retarget_tiles=free_retarget_tiles,
            physical_resident_fn=physical_resident_fn,
        )
        zero_byte_retarget_tiles = (
            free_retarget_tiles
            if physical_resident_fn is not None
            else frozenset(resident_retarget_tiles)
        )
        cold_upserts = {
            int(tile): payload
            for tile, payload in all_candidate_upserts.items()
            if int(tile) not in zero_byte_retarget_tiles
        }
        plan_tiles_by_number = {
            int(tile.montage_index): tile for tile in tuple(getattr(self.plan, "tiles", ()) or ())
        }
        admission_candidates = tuple(
            plan_tiles_by_number.get(int(tile), int(tile)) for tile in all_candidate_upserts
        )
        admission = TileAdmissionQueue(priority_context).admit(
            admission_candidates,
            retained=(),
            free_fn=(lambda tile: int(tile) in free_retarget_tiles)
            if free_retarget_tiles
            else None,
            cost_fn=(
                (
                    lambda tile: (
                        0
                        if int(tile) in zero_byte_retarget_tiles
                        else int(upsert_cost_fn(cold_upserts[int(tile)]))
                    )
                )
                if upsert_cost_fn is not None
                else (
                    lambda tile: (
                        0
                        if int(tile) in zero_byte_retarget_tiles
                        else int(getattr(cold_upserts[int(tile)], "nbytes", 0) or 0)
                    )
                )
            ),
            max_items=max_upserts,
            max_bytes=max_upsert_bytes,
            deadline_ms=cold_deadline_ms,
        )
        capped_upserts = {
            int(tile): all_candidate_upserts[int(tile)]
            for tile in admission.admitted
            if int(tile) in all_candidate_upserts
        }
        # Admission owns both membership and order. Re-filtering the original
        # candidate mapping preserved membership but silently restored its
        # insertion order. That was mostly hidden by small capped uploads, but
        # VisPy item-free batches may admit the whole remaining cohort: after
        # the first eight center tiles, the backend then acknowledged the rest
        # row-by-row. Carry the canonical admission order to the backend.
        upserts = capped_upserts
        near = tuple(
            tile
            for tile in self._near_tile_numbers(margin_tiles=2)
            if int(tile) not in self.skipped_tiles
        )
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

        # A tile that has a fresh payload to upsert is, by definition,
        # present — it must NOT also be removed in the same delta (the delta
        # forbids it). This resolves the coalescing race where a tile was
        # queued for removal (skip/out-of-plan) and then resurrected by a
        # late LOD-level completion before the removal committed: the upsert
        # wins here, and the tile drops out of the pending-removal queue so a
        # future build does not re-remove the now-present tile.
        if removals and upserts:
            conflicting = {int(tile) for tile in removals if int(tile) in upserts}
            if conflicting:
                removals = tuple(tile for tile in removals if int(tile) not in conflicting)
                self.pending_removals.difference_update(conflicting)

        force_refresh = bool(self.backend_refresh_pending)
        clear_reason = ""

        base_revision = int(getattr(previous_state, "revision", 0))
        target_revision = base_revision + (1 if upserts or removals else 0)
        if upserts or removals:
            self.payload_revision += 1
        viewport_identity = _viewport_identity(self.view_range, self.viewport_shape)
        viewport_changed = viewport_identity != self._last_viewport_identity
        presentation_geometry_changed = bool(
            getattr(self, "_layout_geometry_changed_pending", False)
            or planned != self._last_planned_tiles
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
            transaction_generation=int(self.session_id),
            cold_deadline_ms=cold_deadline_ms,
            upserts=upserts,
            removals=removals,
            active_tiles=active,
            planned_tiles=planned,
            near_tiles=near,
            near_tile_source_ids=near_source_ids,
            target_identities=self.tile_target_identities(active),
            force_refresh=force_refresh,
            clear_reason=clear_reason,
        )
        state = previous_state.apply_delta(delta)
        return state, delta

    def build_atomic_successor_presentation(
        self,
    ) -> tuple[TilePresentationState, TilePresentationDelta] | None:
        """Build a complete compatible successor without full reconciliation.

        The general builder repairs arbitrary lifecycle, visibility, removal,
        and level states. A VisPy scroll successor has a narrower contract:
        unchanged slots and one current payload for every required on-screen
        tile. After validating that contract, construct that immutable delta
        directly. Ambiguous cases return ``None`` to the general builder.
        """

        self._atomic_fast_reject_reason = ""
        if bool(getattr(self, "_layout_geometry_changed_pending", False)):
            self._atomic_fast_reject_reason = "layout"
            return None
        required = tuple(self.atomic_successor_required_scope())
        if not required or not set(required).issubset(self._last_planned_tiles):
            self._atomic_fast_reject_reason = "planned"
            return None
        plan_tiles_by_number = {
            int(tile.montage_index): tile for tile in tuple(getattr(self.plan, "tiles", ()) or ())
        }
        previous_state = self.tile_presentation_state
        # Cache-hit scroll windows commonly have 58-59 payload mirrors ready
        # and one entering edge represented only by a resident LOD floor.
        # Materialize only those missing wrappers; scanning/rebuilding all 60
        # erased the fast path's benefit.
        missing_floor = tuple(
            int(tile_number)
            for tile_number in required
            if not _payload_matches_current_tile(
                self,
                int(tile_number),
                self.display_tile_payloads.get(int(tile_number)),
                plan_tiles_by_number,
            )
            and not _payload_matches_current_tile(
                self,
                int(tile_number),
                self.lifecycle.current_presentable_payload(int(tile_number)),
                plan_tiles_by_number,
            )
        )
        if missing_floor:
            self._ensure_floor_payloads(missing_floor)
            # A cache-hit successor can own an exact native result without a
            # resident reduced floor or payload wrapper. Materialize only the
            # still-missing required wrappers; scanning the broader coverage
            # set here would restore the old O(all visible slots) prepass.
            lod_factor = int(self._selected_lod_factor())
            for tile_number in missing_floor:
                current = self.display_tile_payloads.get(int(tile_number))
                if _payload_matches_current_tile(
                    self,
                    int(tile_number),
                    current,
                    plan_tiles_by_number,
                ):
                    continue
                rendered = self.rendered_tiles.get(int(tile_number))
                if rendered is not None:
                    self._ensure_display_tile_payload(
                        int(tile_number),
                        rendered,
                        self.tile_source_ids,
                        lod_factor=lod_factor,
                    )
        payloads = {}
        for tile_number in required:
            payload = self.display_tile_payloads.get(int(tile_number))
            if not _payload_matches_current_tile(
                self,
                int(tile_number),
                payload,
                plan_tiles_by_number,
            ):
                payload = self.lifecycle.current_presentable_payload(int(tile_number))
            if not _payload_matches_current_tile(
                self,
                int(tile_number),
                payload,
                plan_tiles_by_number,
            ):
                self._atomic_fast_reject_reason = f"payload:{int(tile_number)}"
                return None
            self.display_tile_payloads[int(tile_number)] = payload
            payloads[int(tile_number)] = payload
        self.payload_revision += 1
        near = tuple(
            tile
            for tile in self._near_tile_numbers(margin_tiles=2)
            if int(tile) not in self.skipped_tiles
        )
        near_source_ids = {
            int(tile): self.display_tile_payloads[int(tile)].source_id
            for tile in near
            if int(tile) in self.display_tile_payloads
        }
        delta = TilePresentationDelta(
            structure_revision=self.structure_revision,
            payload_revision=self.payload_revision,
            visibility_revision=self.visibility_revision,
            level_revision=self.level_revision,
            histogram_revision=self.histogram_revision,
            viewport_revision=self.viewport_revision,
            base_revision=int(previous_state.revision),
            target_revision=int(previous_state.revision) + 1,
            transaction_generation=int(self.session_id),
            upserts=payloads,
            active_tiles=required,
            planned_tiles=required,
            near_tiles=near,
            near_tile_source_ids=near_source_ids,
            target_identities=self.tile_target_identities(required),
            atomic_handoff=True,
        )
        return previous_state.apply_delta(delta), delta

    def acknowledge_atomic_successor(
        self,
        delta: TilePresentationDelta,
        report: TileCommitReport | None,
        acknowledged: TilePresentationState,
    ) -> bool:
        """Close the pending handoff after one complete backend transaction."""

        if (
            report is None
            or bool(getattr(report, "stale", False))
            or not report.acknowledges(delta)
        ):
            return False
        required = tuple(int(tile) for tile in delta.active_tiles)
        if not required or {int(tile) for tile in delta.upserts} != set(required):
            return False
        if set(report.accepted_upserts_in_order(delta)) != set(required):
            return False
        acknowledged_payloads = dict(acknowledged.payloads)
        if any(
            tile not in acknowledged_payloads
            or tile_ack_identity(acknowledged_payloads[tile])
            != tile_ack_identity(delta.upserts[tile])
            for tile in required
        ):
            return False
        self.atomic_successor_pending = False
        return True

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
            # The pool's slot identities are the newest known physical truth
            # about the screen.  Store that truth in the lifecycle, not in a
            # parallel session map.
            self.lifecycle.backend_presented_snapshot(report.presented_identities)
        if not report.acknowledges(delta):
            # ADR 0051 rule 1 (field defect 2026-07-05, JSONL 112841): the
            # committer's last report belongs to an OLDER delta — this delta
            # never reached the backend (skipped/superseded commit).
            # Acknowledge nothing and park nothing: every dirty entry stays
            # armed, so the next flush re-presents through a real commit.
            return self.tile_presentation_state
        # Production marks the handoff immediately before invoking the
        # backend. Keeping acknowledgement self-contained is both defensive
        # (an accepted backend report proves that handoff occurred) and lets
        # transaction-level tests model the boundary without reaching into
        # renderer effects. Repeating the same emitted identities is
        # idempotent.
        self.lifecycle.commit_emitted(delta.upserts)
        level_delta_stale = bool(
            self.has_pending_level_update()
            and dict(delta.upserts)
            and int(delta.level_revision) != int(self.level_revision)
        )
        if level_delta_stale:
            report = replace(report, stale=True, committed_upserts=())
        # The lifecycle makes the acceptance decision in one place: backend slot
        # identity, emitted identity, and semantic quality all converge there.
        # Session mirrors consume the filtered report and never infer acceptance
        # from tile numbers alone.
        active_scope = {int(tile) for tile in tuple(getattr(delta, "active_tiles", ()) or ())}
        report_accepted = report.accepted_upserts_in_order(delta)
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
        accepted_upserts = report.accepted_upserts_in_order(delta)
        # Remember acknowledged payload identities: re-presenting one is a
        # residency remap for the backend, so commit batching may treat it as
        # nearly free instead of charging full texture bytes (ADR 0050 —
        # prompt level-swap convergence). Bounded: identities are small
        # tuples and the set resets with the session.
        for tile_number in accepted_upserts:
            payload = acknowledged.payloads.get(int(tile_number))
            if payload is not None:
                self.acknowledged_source_ids.add(payload.source_id)
                self.lifecycle.remember_presentable(int(tile_number), payload)
                self.lifecycle.acknowledge_presented(
                    int(tile_number),
                    tile_ack_identity(payload),
                    str(getattr(payload, "quality", "exact") or "exact"),
                    int(getattr(getattr(payload, "lod", None), "level", 0) or 0),
                )
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
        for tile in accepted_upserts:
            index = int(tile)
            payload = acknowledged.payloads.get(index)
            if (
                index in self.rendered_tiles
                and payload is not None
                and str(getattr(payload, "quality", "exact")) == "preview"
            ):
                self.dirty_payloads[index] = None
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
        """Drawn slots whose backend identity differs from the current payload."""

        shown_map = self.lifecycle.backend_presented_identities
        if not shown_map:
            return ()
        mismatched: list[int] = []
        plan_tiles = {
            int(tile.montage_index): tile
            for tile in tuple(getattr(getattr(self, "plan", None), "tiles", ()) or ())
        }
        state_payloads = dict(getattr(self.tile_presentation_state, "payloads", {}) or {})
        for tile_number, shown_identity in shown_map.items():
            current = self.display_tile_payloads.get(int(tile_number))
            tile = plan_tiles.get(int(tile_number))
            if (
                current is not None
                and tile is not None
                and int(getattr(current, "source_index", -1)) != int(tile.source_index)
                and shown_identity == tile_ack_identity(current)
            ):
                mismatched.append(int(tile_number))
                continue
            if current is None:
                acknowledged = state_payloads.get(int(tile_number))
                if (
                    acknowledged is not None
                    and tile is not None
                    and int(getattr(acknowledged, "source_index", -1)) != int(tile.source_index)
                    and shown_identity == tile_ack_identity(acknowledged)
                ):
                    mismatched.append(int(tile_number))
                continue
            current_identity = None if current is None else tile_ack_identity(current)
            if current is None or current_identity == shown_identity:
                continue
            rec = self.lifecycle.peek(int(tile_number))
            if rec is not None and (current_identity, shown_identity) in rec.resigned:
                continue
            pair = (shown_identity, current_identity)
            prior_pair, attempts = self._identity_retry_attempts.get(int(tile_number), (None, 0))
            if prior_pair == pair and attempts >= 3:
                continue
            mismatched.append(int(tile_number))
        return tuple(sorted(mismatched))

    def diagnostic_tile_identity_rows(
        self,
        *,
        limit: int = 20,
        include_all_visible: bool = False,
    ) -> tuple[dict[str, object], ...]:
        """Focused per-slot truth table for montage identity stalls."""

        plan_tiles = {
            int(tile.montage_index): tile
            for tile in tuple(getattr(getattr(self, "plan", None), "tiles", ()) or ())
        }
        presented = {int(tile) for tile in self.lifecycle.presented_tiles}
        visible = {int(tile) for tile in self.visible_tile_numbers}
        target_unsettled = {int(tile) for tile in self.required_target_unsettled_tiles()}
        loading = {int(tile) for tile in self.loading_tiles}
        active = {int(tile) for tile in self.active_tile_requests}
        dirty = {int(tile) for tile in self.dirty_payloads}
        upserts = {int(tile) for tile in self.pending_payload_upserts}
        backend = dict(self.lifecycle.backend_presented_identities)
        desired_payloads = dict(self.display_tile_payloads)
        state_payloads = dict(getattr(self.tile_presentation_state, "payloads", {}) or {})
        suspect = set(visible - presented)
        suspect.update(target_unsettled | loading | active | dirty | upserts)
        for tile_number, payload in desired_payloads.items():
            tile = plan_tiles.get(int(tile_number))
            if tile is not None and int(getattr(payload, "source_index", -1)) != int(
                tile.source_index
            ):
                suspect.add(int(tile_number))
        for tile_number, payload in state_payloads.items():
            tile = plan_tiles.get(int(tile_number))
            if tile is not None and int(getattr(payload, "source_index", -1)) != int(
                tile.source_index
            ):
                suspect.add(int(tile_number))
        for tile_number, identity in backend.items():
            payload = desired_payloads.get(int(tile_number))
            state_payload = state_payloads.get(int(tile_number))
            if (payload is not None and identity != tile_ack_identity(payload)) or (
                payload is None
                and state_payload is not None
                and identity == tile_ack_identity(state_payload)
            ):
                suspect.add(int(tile_number))
        ordered = (
            self._prioritized_tile_numbers(tuple(visible))
            if include_all_visible
            else (self._prioritized_tile_numbers(tuple(suspect)) if suspect else ())
        )
        if not ordered:
            ordered = tuple(sorted(visible))[: max(0, int(limit))]
        rows = []
        for tile_number in tuple(ordered)[: max(0, int(limit))]:
            tile_number = int(tile_number)
            tile = plan_tiles.get(tile_number)
            desired = desired_payloads.get(tile_number)
            state_payload = state_payloads.get(tile_number)
            backend_identity = backend.get(tile_number)
            rec = self.lifecycle.peek(tile_number)
            source_index = None if tile is None else int(tile.source_index)
            semantic_source = None if tile is None else self.tile_semantic_source_id(source_index)
            desired_source_index = (
                None if desired is None else int(getattr(desired, "source_index", -1))
            )
            state_source_index = (
                None if state_payload is None else int(getattr(state_payload, "source_index", -1))
            )
            desired_lod = getattr(desired, "lod", None)
            state_lod = getattr(state_payload, "lod", None)
            evaluation_claim = None if rec is None else getattr(rec, "evaluation_claim", None)
            evaluation_claim_source_index = (
                None
                if evaluation_claim is None
                else int(getattr(evaluation_claim, "source_index", -1))
            )
            visible_first_pixel_complete = self._tile_presentation_matches_current_plan(tile_number)
            resident_levels: list[int] = []
            if rec is not None and tile is not None:
                for key, entry in dict(getattr(rec, "levels", {}) or {}).items():
                    if getattr(entry, "phase", None) is not LevelPhase.RESIDENT:
                        continue
                    if getattr(key, "source_id", None) != semantic_source:
                        continue
                    if int(getattr(key, "tile_id", -1)) != int(source_index):
                        continue
                    level_xy = tuple(getattr(key, "level_xy", ()) or ())
                    if level_xy:
                        resident_levels.append(max(int(value) for value in level_xy))
            rows.append(
                {
                    **tile_truth_record(
                        tile_number=tile_number,
                        target=None if rec is None or rec.target is None else rec.target.identity,
                        acknowledged=backend_identity,
                        payload=desired,
                    ),
                    "tile": tile_number,
                    "source_index": source_index,
                    "semantic_source": _diag_identity(semantic_source),
                    "base_source": _diag_identity(self.tile_source_ids.get(tile_number)),
                    "desired_payload_source": _diag_identity(getattr(desired, "source_id", None)),
                    "desired_payload_source_index": desired_source_index,
                    "desired_payload_quality": ""
                    if desired is None
                    else str(getattr(desired, "quality", "")),
                    "desired_payload_lod": None
                    if desired_lod is None
                    else int(getattr(desired_lod, "level", 0) or 0),
                    "state_payload_source": _diag_identity(
                        getattr(state_payload, "source_id", None)
                    ),
                    "state_payload_source_index": state_source_index,
                    "state_payload_quality": ""
                    if state_payload is None
                    else str(getattr(state_payload, "quality", "")),
                    "state_payload_lod": None
                    if state_lod is None
                    else int(getattr(state_lod, "level", 0) or 0),
                    "backend_source": _diag_identity(backend_identity),
                    "desired_matches_current_source": bool(
                        desired is not None
                        and source_index is not None
                        and desired_source_index == source_index
                    ),
                    "state_matches_current_source": bool(
                        state_payload is not None
                        and source_index is not None
                        and state_source_index == source_index
                    ),
                    "backend_matches_desired": bool(
                        desired is not None and backend_identity == tile_ack_identity(desired)
                    ),
                    "backend_matches_state": bool(
                        state_payload is not None
                        and backend_identity == tile_ack_identity(state_payload)
                    ),
                    "evaluation_claim_source_index": evaluation_claim_source_index,
                    "evaluation_claim_rung": None
                    if evaluation_claim is None
                    else int(getattr(evaluation_claim, "rung", -1)),
                    "evaluation_claim_level": None
                    if evaluation_claim is None
                    else int(getattr(evaluation_claim, "level", -1)),
                    "evaluation_claim_matches_current_source": bool(
                        evaluation_claim is not None
                        and source_index is not None
                        and evaluation_claim_source_index == source_index
                    ),
                    "preview_claims": ()
                    if rec is None
                    else tuple(
                        (
                            int(rung),
                            int(claim[0]),
                            bool(claim[1] == (self.key, semantic_source)),
                        )
                        for rung, claim in sorted(rec.preview_claims.items())
                    ),
                    "semantic_state": "" if rec is None else str(rec.semantic.value),
                    "presentation_state": "" if rec is None else str(rec.presentation.value),
                    "presented_quality": ""
                    if rec is None
                    else str(getattr(rec, "presented_quality", "")),
                    "presented_lod": None if rec is None else getattr(rec, "presented_level", None),
                    "visible_first_pixel_complete": bool(visible_first_pixel_complete),
                    "rendered": tile_number in self.rendered_tiles,
                    "presented": tile_number in presented,
                    "target_unsettled": tile_number in target_unsettled,
                    "loading": tile_number in loading,
                    "active": tile_number in active,
                    "dirty": tile_number in dirty,
                    "pending_upsert": tile_number in upserts,
                    "resident_levels_current_source": tuple(sorted(set(resident_levels))),
                }
            )
        return tuple(rows)

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
            self.lifecycle.tile_skipped(index)
            self.mark_tile_state(tile, MontageTileState.SKIPPED)

    def rendered_tuple(self) -> tuple[RenderedTile, ...]:
        return tuple(
            sorted(self.rendered_tiles.values(), key=lambda rendered: rendered.tile.montage_index)
        )

    def loading_tile_tuple(self) -> tuple[MontageTile, ...]:
        return tuple(
            self.plan.tiles[index]
            for index in sorted(self.loading_tiles)
            if 0 <= index < len(self.plan.tiles)
        )

    def skipped_tile_tuple(self) -> tuple[MontageTile, ...]:
        return tuple(
            self.plan.tiles[index]
            for index in sorted(self.skipped_tiles)
            if 0 <= index < len(self.plan.tiles)
        )

    def ensure_tile_states(self) -> tuple[MontageTileState, ...]:
        if int(self._tile_states_cached_revision) == int(self.tile_state_revision) and len(
            self._tile_states_cached_tuple
        ) == len(tuple(self.plan.tiles)):
            return self._tile_states_cached_tuple
        states = [MontageTileState.UNLOADED for _tile in self.plan.tiles]
        for index in tuple(self.skipped_tiles):
            index = int(index)
            if 0 <= index < len(states):
                states[index] = MontageTileState.SKIPPED
        presented_tiles = set(self.lifecycle.presented_tiles)
        loading_numbers = set(self.loading_tiles) | (set(self.rendered_tiles) - presented_tiles)
        for index in loading_numbers:
            index = int(index)
            if 0 <= index < len(states) and states[index] != MontageTileState.SKIPPED:
                states[index] = MontageTileState.LOADING
        for index in presented_tiles - loading_numbers:
            index = int(index)
            if 0 <= index < len(states):
                states[index] = MontageTileState.LOADED
        self.tile_states = states
        self._tile_states_cached_revision = int(self.tile_state_revision)
        self._tile_states_cached_tuple = tuple(self.tile_states)
        return self._tile_states_cached_tuple

    def is_complete(self) -> bool:
        return not (
            not self.required_target_settled()
            or self.stage_planning_deferred
            or self.deferred_missing_tiles
            or self.loading_tiles
            or self.active_tile_requests
            or self.stage_fan_in.active_requests
            or self.stage_fan_in.attached_requests
            or self.stage_fan_in.has_waiting()
            or self.final_commit_pending
            or self.flush_pending
            or self.dirty_payloads
            or self.pending_payload_upserts
            or self.pending_removals
            or self.has_pending_level_update()
            or self.has_unrefined_preview_payloads()
        )

    def has_unrefined_preview_payloads(self) -> bool:
        return bool(self.unrefined_preview_tiles(include_already_dirty=True))

    def unrefined_preview_tiles(self, *, include_already_dirty: bool = False) -> tuple[int, ...]:
        planned = {int(tile) for tile in self.required_tile_numbers()} - {
            int(tile) for tile in self.skipped_tiles
        }
        if not planned:
            return ()
        payloads = dict(getattr(self.tile_presentation_state, "payloads", {}) or {})
        if not payloads:
            payloads = dict(self.display_tile_payloads)
        tiles: list[int] = []
        for tile in planned:
            if not include_already_dirty and (
                int(tile) in self.dirty_payloads or int(tile) in self.pending_payload_upserts
            ):
                continue
            payload = payloads.get(int(tile))
            if payload is not None and str(getattr(payload, "quality", "exact")) == "preview":
                tiles.append(int(tile))
        return tuple(sorted(tiles))

    def mark_preview_refinements_dirty(self, tile_numbers) -> None:
        changed = False
        for tile_number in tuple(tile_numbers or ()):
            index = int(tile_number)
            if index in self.rendered_tiles:
                self.dirty_payloads[index] = None
                changed = True
        if changed:
            self.flush_pending = True
            self.final_commit_pending = True

    def visible_plan_complete(self) -> bool:
        if self.has_stale_level_presentations():
            return False
        return self.required_target_settled()

    def visible_first_pixels_presented(self) -> bool:
        return self.required_first_pixels_presented()

    def _tile_presentation_matches_current_plan(self, tile_number: int) -> bool:
        index = int(tile_number)
        if index not in self.lifecycle.presented_tiles:
            return False
        if not (0 <= index < len(getattr(self.plan, "tiles", ()) or ())):
            return False
        tile = self.plan.tiles[index]
        payload = self.display_tile_payloads.get(index)
        if payload is None:
            payload = getattr(self.tile_presentation_state, "payloads", {}).get(index)
        backend = dict(self.lifecycle.backend_presented_identities)
        if payload is None:
            return not backend
        if int(getattr(payload, "source_index", -1)) != int(tile.source_index):
            return False
        return not (backend and backend.get(index) != tile_ack_identity(payload))

    def has_stale_level_presentations(self) -> bool:
        snapshot = self.level_presentation_snapshot()
        return bool(self.has_pending_level_update() and int(snapshot.stale_count) > 0)

    def _tile_matches_current_level_target(
        self, tile: int, target: tuple[float, float] | None
    ) -> bool:
        if target is None:
            return True
        return levels_match(
            self.level_generation.tile_values.get(int(tile)), target
        ) and self.level_generation.tile_revisions.get(int(tile)) == int(self.level_revision)

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
        self.commit_batches += 1
        # Upserts under identity-rejected backoff are not commit debt: the
        # backend just proved it rejects exactly these payloads against
        # exactly these targets, so re-arming the flush from them replays a
        # guaranteed rejection at full flush rate (session-148 follow-up).
        # The commit path clears the backoff as soon as either side of that
        # comparison changes.
        backoff = frozenset(getattr(self, "_identity_rejected_backoff_tiles", ()) or ())
        dirty = self.dirty_payloads
        pending = self.pending_payload_upserts
        if backoff:
            dirty = {tile for tile in dirty if int(tile) not in backoff}
            pending = {tile for tile in pending if int(tile) not in backoff}
        commit_owed = bool(dirty or pending or self.pending_removals)
        self.final_commit_pending = commit_owed
        self.flush_pending = commit_owed

    def retarget_tile_priority(
        self,
        *,
        focus=None,
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
        # Every ordering consumer (per-commit upsert admission and prefetch
        # candidates) reads this one context through tile_priority_context;
        # retargets are the only writer.
        # Rebuilding the context ad hoc per consumer let different stages of
        # the pipeline order the same fill around different anchors.
        self._priority_context = context
        self.priority_retargeted_tiles = len(
            {int(tile) for tile in tuple(active_tiles or ())}
            | {int(tile) for tile in tuple(near_tiles or ())}
        )
        return int(self.priority_retargeted_tiles)

    def tile_priority_context(self) -> TilePriorityContext:
        """The session's single effective ordering context.

        Updated only by :meth:`retarget_tile_priority`; built lazily before
        the first retarget.
        """
        context = getattr(self, "_priority_context", None)
        if context is None:
            context = self._build_tile_priority_context()
            self._priority_context = context
        return context

    def _build_tile_priority_context(
        self, *, active_tiles=None, near_tiles=None, priority_tiles=None, view_range=None
    ) -> TilePriorityContext:
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
        return prioritize_tile_numbers(
            tiles,
            plan_tiles=tuple(getattr(self.plan, "tiles", ()) or ()),
            context=self.tile_priority_context(),
        )


@dataclass(frozen=True)
class PresentationTransitionDecision:
    retain_pixels: bool
    atomic_successor: bool
    reason: str
    detail: str = ""


def plan_presentation_transition(
    previous_session,
    session,
    *,
    predecessor_visible: bool,
) -> PresentationTransitionDecision:
    """Plan truthful predecessor retention across a compatible rebirth.

    This is the deciding owner for retaining drawn (stale-but-honest) pixels
    across the transition instead of blanking the surface. It covers both a
    sliced-image change and a montage semantic/layout successor.
    The latter normally reuses ``retarget_index_window``, but rapid churn may
    replace an unfinished session; excluding that rebirth allowed a bounded
    four-tile preview commit to collapse a complete 100-tile VisPy surface.

    The predicate is deliberately
    conservative (ADR 0051 correctness history): the exact document/operation
    source,
    colormap, window/levels mode, shader/CPU backend semantics, and LOD policy
    must match exactly.
    Operation steps may not differ: they change the semantic source. The old
    residency remains reusable, but its mappings are hidden until the new
    source presents. This is distinct from compatible LOD fallback, which may
    retain a coarser rendering only under the same semantic source identity.
    The camera deliberately does not participate (see the note inline).
    Comparing semantic intent against the DYING session is chain-safe only
    when the physical surface still draws a predecessor.  The surface-owned
    ``predecessor_visible`` fact makes that condition explicit: a session
    born after an incompatible blank cannot resurrect an atomic handoff and
    wait forever for coverage that no longer exists.

    Retention is presentation-only.  It can never resurrect the ADR 0051
    stale-acknowledgement class because nothing here touches the session:
    the new lifecycle starts cold, payload seeding still requires exact
    source identities, and hover/probe reads stay gated by the render
    generation (`_is_committed_display_frame_current`), which every rebirth
    advances.
    """

    from arrayscope.operations.evaluator import _document_key

    def reject(reason: str, detail: str = "") -> PresentationTransitionDecision:
        return PresentationTransitionDecision(False, False, str(reason), str(detail))

    if previous_session is None or session is None:
        return reject("missing-session")
    if not bool(predecessor_visible):
        return reject("predecessor-hidden")
    previous_axis = getattr(previous_session, "montage_axis", None)
    axis = getattr(session, "montage_axis", None)
    montage_axis_changed = previous_axis != axis
    if axis is None and bool(getattr(session, "force_auto", False)):
        return reject("force-auto")
    if getattr(session, "skipped_tiles", None) or getattr(previous_session, "skipped_tiles", None):
        return reject("skipped-tiles")
    # Exact document identity includes operation steps without hashing the
    # array. Residency may outlive this boundary for fast reverts, but visible
    # fallback may not: old operation pixels are the wrong semantic source.
    if _document_key(previous_session.document) != _document_key(session.document):
        return reject("document")
    if previous_session.window_mode != session.window_mode:
        return reject("window-mode")
    if previous_session.user_levels_override != session.user_levels_override:
        return reject("user-levels")
    if previous_session.colormap_lut is not session.colormap_lut:
        return reject("colormap")
    if bool(getattr(previous_session, "shader_display", False)) != bool(
        getattr(session, "shader_display", False)
    ):
        return reject("shader-display")
    if getattr(previous_session, "lod_policy_mode", None) != getattr(
        session, "lod_policy_mode", None
    ):
        return reject("lod-policy")
    # The surface contract permits retained pixels only for a slice/source
    # index successor.  Normalize exactly that source-selection field, then
    # compare the remaining semantic view state here, in the one transition
    # owner.  In particular, image axes and flips change where source samples
    # are drawn; keeping their old mappings visible is not a stale preview but
    # the wrong presentation.  Auto-derived layout geometry remains outside
    # ViewState, so harmless scrollbar/column settling does not participate.
    previous_state = previous_session.view_state
    state = session.view_state
    if type(previous_state) is not type(state):
        return reject("view-state-type")
    if montage_axis_changed:
        # Montage entry/exit is a topology change of the same semantic
        # source, not a new source: with every identity gate above already
        # satisfied, the settled predecessor is an honest visual bridge
        # ("video player between frames") until the successor's first
        # bounded delta replaces it. Blanking here put a multi-second black
        # window between a plane and its own montage (R8 continuity gate,
        # 2026-07-18 blackout dossier). Only the montage selection fields may
        # differ; the predecessor cannot complement successor slots across
        # topologies, so the bridge never arms the all-slot atomic handoff.
        try:
            aligned = replace(
                state,
                montage_axis=previous_state.montage_axis,
                montage_columns=previous_state.montage_columns,
                montage_indices=previous_state.montage_indices,
                montage_text=previous_state.montage_text,
            )
        except (AttributeError, TypeError, ValueError):
            return reject("montage-axis")
        if aligned != previous_state:
            return reject("montage-axis", "view-state")
        first_pixels = getattr(previous_session, "required_first_pixels_presented", None)
        if not bool(callable(first_pixels) and first_pixels()) and not bool(
            getattr(previous_session, "presentation_bridge_pending", False)
        ):
            return reject("montage-axis", "predecessor-incomplete")
        return PresentationTransitionDecision(True, False, "montage-axis-bridge")
    if axis is None:
        try:
            aligned = replace(state, slice_indices=previous_state.slice_indices)
        except (AttributeError, TypeError, ValueError):
            return reject("slice-state")
        if aligned != previous_state:
            return reject("view-state")
        return PresentationTransitionDecision(True, False, "slice-compatible")
    try:
        aligned = replace(
            state,
            montage_indices=previous_state.montage_indices,
            montage_text=previous_state.montage_text,
        )
    except (AttributeError, TypeError, ValueError):
        return reject("montage-state")
    if aligned != previous_state:
        return reject("view-state")
    same_topology = _montage_plan_topology(previous_session.plan) == _montage_plan_topology(
        session.plan
    )
    first_pixels = getattr(previous_session, "required_first_pixels_presented", None)
    complete_predecessor = bool(
        bool(callable(first_pixels) and first_pixels())
        or getattr(previous_session, "atomic_successor_pending", False)
    )
    if not complete_predecessor:
        # A rebirth can replace a bridge successor before its first commit.
        # The physical surface then still draws the ORIGINAL bridge
        # predecessor (predecessor_visible above is surface-owned truth), so
        # the honest bridge continues; arming the all-slot atomic handoff
        # against a pixel-less predecessor is what the completeness check
        # exists to prevent.
        if bool(getattr(previous_session, "presentation_bridge_pending", False)):
            return PresentationTransitionDecision(True, False, "montage-axis-bridge")
        return reject("predecessor-incomplete")
    if not same_topology:
        # Retaining the predecessor is still an honest visual bridge, but it
        # cannot own slots that do not exist in its layout.  The successor
        # must therefore publish normal bounded deltas instead of waiting for
        # an all-slot hidden warm that the predecessor cannot complement.
        return PresentationTransitionDecision(
            True,
            False,
            "montage-topology-change",
        )
    if not session.atomic_successor_required_scope():
        # A required tile outside presentation coverage has no physical slot
        # in this transaction. Retain the predecessor, but do not arm an
        # impossible handoff.
        return PresentationTransitionDecision(True, False, "montage-partial-viewport")
    # A montage rebirth has a cold lifecycle even though the physical surface
    # still owns a compatible predecessor. Hand off every required on-screen
    # slot together so no partial successor replaces that complete frame.
    return PresentationTransitionDecision(True, True, "montage-compatible")


def _base_source_id(source_id) -> object:
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


def _payload_matches_current_tile(session, tile_number: int, payload, plan_tiles_by_number) -> bool:
    tile = dict(plan_tiles_by_number or {}).get(int(tile_number))
    if tile is None or payload is None:
        return False
    if int(getattr(payload, "source_index", -1)) != int(getattr(tile, "source_index", -2)):
        return False
    payload_identity = getattr(payload, "tile_identity", None)
    record = session.lifecycle.peek(int(tile_number))
    target_identity = None if record is None or record.target is None else record.target.identity
    if payload_identity is not None and target_identity is not None:
        # Typed semantic truth is authoritative once both sides provide it.
        # Falling through to the materialization source id would let a cached
        # plane minted for old axes/flips masquerade as the current mapping.
        return bool(payload_identity.satisfies_target(target_identity))
    base_source_id = _base_source_id(getattr(payload, "source_id", None))
    if base_source_id == session.tile_semantic_source_id(int(tile.source_index)):
        return True
    return (
        isinstance(base_source_id, tuple)
        and len(base_source_id) >= 2
        and base_source_id[0] == "rendered_tile"
        and int(base_source_id[1]) == int(tile_number)
        and getattr(payload, "source_id", None)
        in getattr(session, "acknowledged_source_ids", set())
    )


def _preview_upgrade_owed(session, tile_number: int, payload=None) -> bool:
    """Return whether a visible preview must still converge to rendered pixels."""

    index = int(tile_number)
    payload = session.display_tile_payloads.get(index) if payload is None else payload
    if payload is None or str(getattr(payload, "quality", "exact") or "exact") != "preview":
        return False
    rendered = getattr(session, "rendered_tiles", {}).get(index)
    if rendered is None:
        return False
    plan_tiles = tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
    if 0 <= index < len(plan_tiles):
        plan_tile = plan_tiles[index]
        if int(getattr(payload, "source_index", -1)) != int(getattr(plan_tile, "source_index", -2)):
            return False
        if int(getattr(getattr(rendered, "tile", None), "source_index", -1)) != int(
            getattr(plan_tile, "source_index", -2)
        ):
            return False
    if index in getattr(session, "skipped_tiles", set()):
        return False
    return index in getattr(session, "visible_tile_numbers", ())


def _payload_is_reduced_target(payload) -> bool:
    if payload is None or str(getattr(payload, "quality", "exact") or "exact") == "preview":
        return False
    lod = getattr(payload, "lod", None)
    return bool(lod is not None and int(getattr(lod, "level", 0) or 0) > 0)


def _force_unpresented_upsert(
    session, tile_number: int, *, previous_payloads, backend_identities
) -> bool:
    """Whether an unpresented active slot needs an emitted payload now."""

    tile_number = int(tile_number)
    payload = session.display_tile_payloads.get(tile_number)
    if payload is None:
        return False
    previous = dict(previous_payloads or {}).get(tile_number)
    shown_map = dict(backend_identities or {})
    if shown_map:
        shown = shown_map.get(tile_number)
        payload_identity = tile_ack_identity(payload)
        if shown == payload_identity:
            return False
        if shown is not None:
            rec = session.lifecycle.peek(tile_number)
            if rec is not None and (payload_identity, shown) in rec.resigned:
                return False
        return True
    return previous is None


def _diag_identity(value, *, limit: int = 180) -> str:
    if value is None:
        return ""
    text = repr(value)
    if len(text) <= int(limit):
        return text
    return text[: max(0, int(limit) - 3)] + "..."


def _view_range_cache_key(view_range) -> tuple[tuple[float, ...], ...] | tuple[object, ...]:
    try:
        return tuple(
            tuple(float(value) for value in tuple(axis)) for axis in tuple(view_range or ())
        )
    except Exception:
        return (repr(view_range),)
