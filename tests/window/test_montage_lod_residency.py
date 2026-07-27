"""Qt-free contract tests for resident-LOD montage sessions (ADR 0050)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.frame_targets import FrameTarget
from arrayscope.core.view_state import ChannelMode, ViewState
from arrayscope.display.backend_contract import PYQTGRAPH_CAPABILITIES
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.lod import (
    LOD_POLICY_NATIVE_ONLY,
    LOD_POLICY_RESIDENT,
    LodDemand,
    LodInfo,
    select_lod_demand,
)
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    PayloadSourceAnchor,
    TileCommitReport,
    TilePresentationState,
)
from arrayscope.display.model.tile_identity import tile_ack_identity
from arrayscope.display.model.tile_priority import TilePriorityContext, prioritize_tile_numbers
from arrayscope.display.montage import MontagePlan, MontageTile, RenderedTile, make_montage_plan
from arrayscope.display.pyramid import (
    LodPageCache,
    MaterializedLodPage,
    materialize_lod_page,
    reduce_box_mean,
)
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.display.slice_engine import make_shader_image_from_slab
from arrayscope.display.source_anchoring import SourceAnchoring
from arrayscope.kernel import Lane, Priority
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT
from arrayscope.presentation import ClaimOwner, Presentation
from arrayscope.render import effects as render_effects
from arrayscope.render import lod as render_lod
from arrayscope.render.ladder import LadderPolicy, LodLadder, Rung, RungStep
from arrayscope.render.lod import LodPageSetKey, admit_retained_preview_level
from arrayscope.render.stages import CommitBatch, LodAdmissionScope, RenderIntent
from arrayscope.window import frame_effects as montage_commit
from arrayscope.window.frame_effects import FramePipelineEffects, _priority_ordered_tile_delta
from arrayscope.window.frame_session import (
    FrameSession,
    _base_source_id,
    _payload_matches_current_tile,
    page_set_key_for_rendered,
    plan_presentation_transition,
    texture_source_for_rendered,
)

TILE = 64
# Two 64x64 tiles seen through a viewport that shows four source texels per
# screen pixel: demand is factor 4 / level 2 on both axes.
ZOOMED_OUT_RANGE = ((0.0, 4.0 * 2 * TILE), (0.0, 4.0 * TILE))
VIEWPORT = (TILE, 2 * TILE)


def _tiles(count=2):
    return tuple(
        MontageTile(
            montage_index=index,
            source_index=index,
            row=0,
            col=index,
            x0=index * TILE,
            y0=0,
            width=TILE,
            height=TILE,
            view_state=None,
        )
        for index in range(count)
    )


def _session(*, mode=LOD_POLICY_RESIDENT, pyramid=None, view_range=ZOOMED_OUT_RANGE, count=2):
    tiles = _tiles(count)
    session = FrameSession(
        session_id=1,
        key=("test",),
        render_generation=1,
        level_key=None,
        level_expected_indices=tuple(range(count)),
        plan=MontagePlan(
            axis=0,
            tile_shape=(TILE, TILE),
            grid_shape=(1, count),
            columns=count,
            rows=1,
            gap=0,
            tiles=tiles,
        ),
        view_state=None,
        document=None,
        montage_axis=0,
        colormap_lut=None,
        viewport_shape=VIEWPORT,
        view_range=view_range,
        output_dtype=np.dtype("float32"),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        lod_policy_mode=mode,
        lod_page_cache=pyramid,
    )
    for index, tile in enumerate(tiles):
        image = np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE) + index
        session.rendered_tiles[index] = RenderedTile(
            tile=tile,
            image=image,
            histogram_data=image,
            eval_ms=0.0,
            slab_shape=image.shape,
            slab_nbytes=image.nbytes,
        )
        session.dirty_payloads[index] = None
    return session


def _phase_shader_rendered(session, values: np.ndarray) -> RenderedTile:
    """Install one live complex/angle tile without bypassing display routing."""

    values = np.ascontiguousarray(values, dtype=np.complex64)
    state = ViewState.from_shape(values.shape).with_channel(ChannelMode.ANGLE)
    display = make_shader_image_from_slab(
        values,
        SimpleNamespace(view_state=state, ranged_axes=()),
    )
    tile = session.plan.tiles[0]
    rendered = RenderedTile(
        tile=tile,
        image=display.data,
        histogram_data=display.histogram_data,
        eval_ms=0.0,
        slab_shape=values.shape,
        slab_nbytes=values.nbytes,
        shader_mapping=display.shader_mapping,
        texture_kind=display.texture_kind,
        semantic_data=display.semantic_data,
        lod_source_data=display.lod_source_data,
        level_data=display.level_data,
    )
    session.output_dtype = np.dtype(np.complex64)
    session.view_state = state
    session.shader_display = True
    session.rendered_tiles[0] = rendered
    session.display_tile_payloads.clear()
    session.dirty_payloads[0] = None
    return rendered


def _materialize(session, request):
    """Run one request through the canonical checked page route."""

    if hasattr(session.pending_rung_materializations, "mark_started"):
        session.pending_rung_materializations.mark_started(request)
    pages = []
    try:
        for plan in request.claimed_plans:
            page = materialize_lod_page(
                request.source,
                source_origin_yx=request.source_origin_yx,
                plan=plan,
            )
            pages.append(session.lod_page_cache.admit_as(plan.key, page, owner=request.owner))
    finally:
        session.lod_page_cache.release_owner_claims(request.owner)
    if hasattr(session.pending_rung_materializations, "mark_resident"):
        session.pending_rung_materializations.mark_resident(request)
    return request.key, tuple(pages)


def _release(session, request):
    """Drop one request the way every non-run scheduling path must: all claims."""

    if hasattr(session.pending_rung_materializations, "release"):
        session.pending_rung_materializations.release(request)


def _claim_preview_resident(session, tile_number: int, key) -> None:
    session.lifecycle.level_claimed(
        int(tile_number), key, ClaimOwner.PREVIEW, request=("test-preview", key)
    )
    session.lifecycle.level_resident(int(tile_number), key)


def _admit_demand_level_for_test(pyramid, demand, rendered, *, semantic_source_id):
    level = int(demand.desired_level)
    if pyramid is None or level <= 0:
        return None
    key = page_set_key_for_rendered(
        rendered, demand=demand, level=level, semantic_source_id=semantic_source_id
    )
    source, _histogram, _kind = texture_source_for_rendered(rendered)
    return _admit_page_set(pyramid, key, source)


def _admit_page_set(cache, key, source):
    owner = ("test-page-set", id(key), id(source))
    claimed = cache.claim_plans(key.plans, owner)
    if not claimed:
        return None
    pages = []
    try:
        for plan in claimed:
            page = materialize_lod_page(source, source_origin_yx=(0, 0), plan=plan)
            pages.append(cache.admit_as(plan.key, page, owner=owner))
    finally:
        cache.release_owner_claims(owner)
    resident = cache.resolved_pages(key.plans)
    assert resident is not None
    return resident[0].values if len(resident) == 1 else resident


def _materialized_page_set(key, source):
    return tuple(
        materialize_lod_page(source, source_origin_yx=(0, 0), plan=plan) for plan in key.plans
    )


class _RungPrepareRenderer:
    def __init__(self, *, kernel=None) -> None:
        self.kernel = kernel

    def _frame_session_is_current(self, session) -> bool:
        return True

    def request_montage_replan(self, session) -> None:
        session._test_replan_requested = True


class _StageProducerKernel:
    def __init__(self, *keys, completed=()) -> None:
        self.live = set(keys)
        self.completed = set(completed)

    def has_live_task(self, key) -> bool:
        return key in self.live

    def has_completed_task(self, key) -> bool:
        return key in self.completed


def _pipeline_intent_for(session, *, semantic_key=None, viewport_key="vp"):
    return RenderIntent(
        semantic_key=getattr(session, "key", None) if semantic_key is None else semantic_key,
        viewport_key=viewport_key,
        presentation_key="presentation",
        view_range=getattr(session, "view_range", None),
        viewport_shape=getattr(session, "viewport_shape", None),
        tile_source_ids=tuple(
            (
                int(tile.montage_index),
                session.tile_semantic_source_id(int(tile.source_index)),
            )
            for tile in tuple(session.plan.tiles)
        ),
        tile_source_indices=tuple(
            (int(tile.montage_index), int(tile.source_index)) for tile in tuple(session.plan.tiles)
        ),
    )


def _pipeline_scope_for(session):
    return LodAdmissionScope(
        visible_tile_numbers=tuple(
            int(tile.montage_index) for tile in tuple(session.visible_tiles)
        ),
        near_tile_numbers=tuple(int(tile.montage_index) for tile in tuple(session.visible_tiles)),
        visible_missing_count=len(tuple(session.visible_tiles)),
    )


def _exact_step(tile_number=0):
    return RungStep(
        tile_number=int(tile_number),
        rung=Rung.EXACT,
        level=0,
        reduce_from_native=False,
        lane=Lane.VISIBLE_MATERIALIZATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="test",
    )


def _desired_materialization_step(tile_number=0, *, level=2):
    return RungStep(
        tile_number=int(tile_number),
        rung=Rung.DESIRED,
        level=int(level),
        reduce_from_native=False,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )


def _settle_first_pixels(session) -> None:
    """Close the phase-1 coverage pass for phase-2 machinery tests.

    Progressive presentation contract (2026-07-18): materialization,
    pressure, and refinement planning are phase-2 work and only run after
    phase-1 coverage completes. These fixtures exercise that machinery, so
    they start from a covered scope.
    """

    from arrayscope.presentation.tile_lifecycle import TileTarget

    tiles = tuple(session.plan.tiles)
    session.lifecycle.retarget(
        {
            int(tile.montage_index): TileTarget(
                tile_number=int(tile.montage_index),
                source_index=int(tile.source_index),
                semantic_source_id=session.tile_semantic_source_id(int(tile.source_index)),
            )
            for tile in tiles
        }
    )
    session.lifecycle.backend_presented_snapshot(
        {
            int(tile.montage_index): session.tile_semantic_source_id(int(tile.source_index))
            for tile in tiles
        }
    )
    session.lifecycle.presentation_confirmed(tuple(int(tile.montage_index) for tile in tiles))


def _plan_rung_materializations(session) -> tuple:
    demand = session.lod_policy_decision.demand
    policy = LadderPolicy(
        mode=session.lod_policy_mode,
        floor_level=max(1, int(getattr(session, "lod_preview_level", 0) or 4)),
        reduced_input_available=True,
    )
    effects = FramePipelineEffects(_RungPrepareRenderer(), session)
    for step in LodLadder(policy).plan(render_effects.tile_lod_states(session, demand), demand):
        if step.rung == Rung.DESIRED:
            effects.prepare_rung(None, step)
    return tuple(session.lifecycle.active_materializations())


def test_native_only_mode_is_unchanged_by_default():
    session = _session(mode=LOD_POLICY_NATIVE_ONLY, pyramid=None)
    _state, delta = session.build_tile_presentation({})

    assert session.lod_policy_mode == LOD_POLICY_NATIVE_ONLY
    assert session.lod_policy_decision.policy == "native-only"
    assert session.lod_policy_decision.applied_factor == 1
    assert session.pending_rung_materializations == []
    for payload in delta.upserts.values():
        assert payload.lod.level == 0
        assert payload.texture_data.shape[:2] == (TILE, TILE)


def test_uncapped_backend_batch_preserves_canonical_admission_order():
    session = _session(
        mode=LOD_POLICY_NATIVE_ONLY,
        count=12,
        view_range=((0.0, 12.0 * TILE), (0.0, TILE)),
    )
    expected = prioritize_tile_numbers(
        range(12),
        plan_tiles=session.plan.tiles,
        context=session.tile_priority_context(),
    )

    _state, delta = session.build_tile_presentation({})

    assert expected != tuple(range(12))
    assert tuple(delta.upserts) == expected


def test_first_pixel_coverage_preserves_canonical_focus_band_order():
    session = _session(
        mode=LOD_POLICY_NATIVE_ONLY,
        count=12,
        view_range=((0.0, 12.0 * TILE), (0.0, TILE)),
    )
    session._priority_context = TilePriorityContext.from_tiles(
        view_range=session.view_range,
        visible_tiles=range(12),
        priority_tiles=(0,),
    )
    expected = prioritize_tile_numbers(
        range(12),
        plan_tiles=session.plan.tiles,
        context=session.tile_priority_context(),
    )

    _state, delta = session.build_tile_presentation({})

    assert expected[0] == 0
    assert tuple(delta.upserts) == expected


def test_backend_boundary_reorders_stale_delta_with_current_camera_context():
    session = _session(
        mode=LOD_POLICY_NATIVE_ONLY,
        count=12,
        view_range=((0.0, 12.0 * TILE), (0.0, TILE)),
    )
    _state, delta = session.build_tile_presentation({})
    stale = replace(delta, upserts=dict(reversed(tuple(delta.upserts.items()))))
    session._priority_context = TilePriorityContext.from_tiles(
        view_range=session.view_range,
        visible_tiles=range(12),
        priority_tiles=(0,),
    )
    canonical = prioritize_tile_numbers(
        range(12),
        plan_tiles=session.plan.tiles,
        context=session.tile_priority_context(),
    )
    expected = tuple(tile for tile in canonical if tile in stale.upserts)

    ordered = _priority_ordered_tile_delta(session, stale)

    assert tuple(ordered.upserts) == expected
    assert tuple(ordered.priority_ranks) == expected
    assert list(ordered.priority_ranks.values()) == sorted(ordered.priority_ranks.values())


def test_visible_replacement_retains_presented_payload_until_acknowledged():
    session = _session(count=1)
    source_ids = {0: session.tile_semantic_source_id(0)}
    _state, delta = session.build_tile_presentation(source_ids)
    old_payload = delta.upserts[0]
    report = TileCommitReport(
        presented_tiles=frozenset({0}),
        committed_upserts=frozenset({0}),
        delta_key=(delta.base_revision, delta.target_revision),
        presented_identities={0: tile_ack_identity(old_payload)},
    )
    session.acknowledge_tile_presentation(delta, report)
    assert session.lifecycle.record(0).presentation is Presentation.PRESENTED

    tile = session.plan.tiles[0]
    replacement = RenderedTile(
        tile=tile,
        image=np.ones((TILE, TILE), dtype=np.float32) * 7,
        histogram_data=np.ones((TILE, TILE), dtype=np.float32) * 7,
        eval_ms=0.0,
        slab_shape=(TILE, TILE),
        slab_nbytes=TILE * TILE * 4,
    )
    session.mark_materialized(replacement)
    assert session.lifecycle.record(0).presentation is Presentation.PRESENTED

    state2, delta2 = session.build_tile_presentation(source_ids, max_upserts=0)

    assert 0 not in delta2.removals
    assert 0 in delta2.active_tiles
    assert state2.payloads[0].source_id == old_payload.source_id
    assert session.lifecycle.record(0).presentation is Presentation.PRESENTED


def test_preview_level_tracks_coarser_viewport_demand():
    far_zoom = ((0.0, 32.0 * 2 * TILE), (0.0, 32.0 * TILE))
    session = _session(
        pyramid=LodPageCache(max_bytes=1 << 24),
        view_range=far_zoom,
        count=272,
    )
    session.lod_preview_min_level = 4
    session.lod_preview_level = 4

    session._selected_lod_factor()

    assert session.lod_policy_decision.demand.desired_level >= 5
    assert session.lod_preview_level == session.lod_policy_decision.demand.desired_level + 5
    assert (
        render_effects.preview_evaluation_level(session, session.lod_policy_decision.demand)
        == session.lod_preview_level
    )


def test_resident_mode_falls_back_to_native_and_records_missing_levels():
    session = _session(pyramid=LodPageCache(max_bytes=1 << 20))
    _settle_first_pixels(session)
    _state, delta = session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))

    decision = session.lod_policy_decision
    assert decision.policy == "resident"
    assert decision.demand.desired_level == 2
    # Nothing is materialized yet: applied stays native, never blocking.
    assert decision.applied_level == 0
    for payload in delta.upserts.values():
        assert payload.lod.level == 0
        assert payload.texture_data.shape[:2] == (TILE, TILE)
    # Every tile recorded its demanded-but-missing level exactly once.
    assert len(requests) == 2
    tiles = sorted(request.tile_number for request in requests)
    assert tiles == [0, 1]
    for request in requests:
        assert isinstance(request.key, LodPageSetKey)
        assert request.key.factor_xy == (4, 4)
        assert request.reduce_factor_xy == (4, 4)
        assert request.source.shape == (TILE, TILE)
    assert len(session.lifecycle.dangling_claims()) == 2


def test_duplicate_materialization_requests_coalesce():
    session = _session(pyramid=LodPageCache(max_bytes=1 << 20))
    _settle_first_pixels(session)
    session.build_tile_presentation({})
    first = list(_plan_rung_materializations(session))

    # A second commit while requests are pending must not re-claim them.
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})

    assert list(_plan_rung_materializations(session)) == first
    assert len(first) == 2


def test_materialization_request_owns_one_rung_and_only_its_missing_pages():

    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    _settle_first_pixels(session)
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    assert len(requests) == 2
    for request in requests:
        assert tuple(request.chain) == ((request.key, (1, 1)),)
        assert tuple(plan.key for plan in request.claimed_plans) == request.key.page_keys
        assert request.source.shape == (TILE, TILE)
        _materialize(session, request)

    assert len(pyramid) == 2
    assert pyramid.pending_count == 0


def test_coarser_materialization_is_direct_source_despite_resident_finer_pages():

    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, view_range=((0.0, 3.0 * 2 * TILE), (0.0, 3.0 * TILE)))
    _settle_first_pixels(session)
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    for request in requests:
        assert tuple(step_key.level_xy for step_key, _rel in request.chain) == ((1, 1),)
        _materialize(session, request)
    finer_values = {page.key: page.values.copy() for page in pyramid.resident_pages()}

    session.view_range = ((0.0, 5.0 * 2 * TILE), (0.0, 5.0 * TILE))
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    assert len(requests) == 2
    for request in requests:
        assert request.source.shape[:2] == (TILE, TILE)
        _materialize(session, request)
    assert all(
        np.array_equal(pyramid.peek(key).values, values) for key, values in finer_values.items()
    )
    assert len(pyramid) == 4
    assert pyramid.pending_count == 0


def test_page_singleflight_attaches_without_stealing_foreign_claim():

    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=1)
    rendered = session.rendered_tiles[0]
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE), previous_factor=1)
    target = page_set_key_for_rendered(
        rendered, demand=demand, level=2, semantic_source_id=session.tile_semantic_source_id(0)
    )
    foreign_owner = ("foreign",)
    assert pyramid.claim_plans(target.plans, foreign_owner) == target.plans
    request = render_lod.plan_materialization(session, rendered, demand=demand, level=2, key=target)
    assert request.claimed_plans == ()
    assert pyramid.pending_count == len(target.plans)
    pyramid.release_owner_claims(foreign_owner)
    assert pyramid.pending_count == 0


def _exercise_visible_request_attached_to_prefetch_claims(*, resident_prefix: int) -> None:
    """Complete one retained prefetch while visible planning is attached."""

    from arrayscope.display.slice_engine import DisplayImage
    from arrayscope.operations.evaluator import EvaluationResult
    from arrayscope.window import montage_prefetch

    size = 1028  # L2 yields 257x257 stored samples: four canonical pages.
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    old_tile = session.plan.tiles[0]
    tile = replace(old_tile, width=size, height=size)
    session.plan = replace(
        session.plan,
        tile_shape=(size, size),
        tiles=(tile,),
    )
    session.visible_tiles = (tile,)
    session.visible_tile_numbers = frozenset({0})
    source = np.arange(size * size, dtype=np.float32).reshape(size, size)
    rendered = replace(
        session.rendered_tiles[0],
        tile=tile,
        image=source,
        histogram_data=source,
        slab_shape=source.shape,
        slab_nbytes=source.nbytes,
    )
    session.rendered_tiles = {0: rendered}
    session.lod_preview_level = 2
    demand = session.lod_policy_decision.demand
    key = render_lod.page_set_key_for_tile(
        session,
        tile,
        demand=demand,
        level=2,
    )
    assert len(key.plans) == 4
    assert (
        page_set_key_for_rendered(
            rendered,
            demand=demand,
            level=2,
            semantic_source_id=session.tile_semantic_source_id(0),
        )
        == key
    )

    if resident_prefix:
        setup_owner = ("prefetch-race-resident-prefix", int(resident_prefix))
        prefix_plans = key.plans[:resident_prefix]
        assert pyramid.claim_plans(prefix_plans, setup_owner) == prefix_plans
        for plan in prefix_plans:
            pyramid.admit_as(
                plan.key,
                materialize_lod_page(source, source_origin_yx=(0, 0), plan=plan),
                owner=setup_owner,
            )
        pyramid.release_owner_claims(setup_owner)

    claim = montage_prefetch._claim_walk_preview(session, tile)
    assert claim is not None
    assert claim.key == key
    assert len(claim.claimed_plans) == len(key.plans) - resident_prefix

    # Visible planning attaches to the same complete target. All of its
    # non-resident pages are foreign prefetch claims, so it must wait for the
    # prefetch terminal wake instead of duplicating work or raising.
    visible_request = render_lod.plan_materialization(
        session,
        rendered,
        demand=demand,
        level=2,
        key=key,
    )
    assert visible_request.claimed_plans == ()
    assert pyramid.pending_count == len(claim.claimed_plans)

    result = EvaluationResult(
        value=DisplayImage(data=source, histogram_data=source),
        eval_ms=0.0,
        slab_shape=source.shape,
        slab_nbytes=source.nbytes,
    )
    residents_before = tuple(page.key for page in pyramid.resident_pages())
    pages = montage_prefetch._materialize_walk_preview(
        session,
        tile,
        result,
        claim,
        shader_display=False,
    )
    assert all(isinstance(page, MaterializedLodPage) for page in pages)
    assert tuple(page.key for page in pyramid.resident_pages()) == residents_before
    assert pyramid.pending_count == len(claim.claimed_plans)

    assert montage_prefetch._admit_walk_preview_result(
        session,
        tile,
        claim,
        pages,
    )
    montage_prefetch._release_walk_preview_claim(session, claim)
    assert pyramid.exact_pages(key.plans) is not None
    assert pyramid.pending_count == 0
    record = session.lifecycle.peek(0)
    assert record is not None
    assert record.levels[key].phase.value == "resident"

    presentation_wakes = []
    session.pipeline = SimpleNamespace(
        effects=SimpleNamespace(
            request_presentation=lambda: presentation_wakes.append("presentation")
        )
    )
    renderer = _RungPrepareRenderer()
    montage_prefetch._wake_walk_preview_admission(renderer, session)
    assert session._test_replan_requested is True
    assert presentation_wakes == ["presentation"]


def test_visible_request_converges_after_all_pages_are_foreign_prefetch_claims():
    _exercise_visible_request_attached_to_prefetch_claims(resident_prefix=0)


def test_visible_request_converges_after_partial_foreign_prefetch_claims():
    _exercise_visible_request_attached_to_prefetch_claims(resident_prefix=1)


def test_visible_worker_declines_until_partially_foreign_prefetch_pages_arrive():
    """An overlapping target may own boundaries while prefetch owns interior."""

    size = 1028
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    _settle_first_pixels(session)
    tile = replace(session.plan.tiles[0], width=size, height=size)
    session.plan = replace(session.plan, tile_shape=(size, size), tiles=(tile,))
    session.visible_tiles = (tile,)
    session.visible_tile_numbers = frozenset({0})
    source = np.arange(size * size, dtype=np.float32).reshape(size, size)
    rendered = replace(
        session.rendered_tiles[0],
        tile=tile,
        image=source,
        histogram_data=source,
        slab_shape=source.shape,
        slab_nbytes=source.nbytes,
    )
    session.rendered_tiles = {0: rendered}
    demand = session.lod_policy_decision.demand
    key = render_lod.page_set_key_for_tile(
        session,
        tile,
        demand=demand,
        level=2,
    )
    assert len(key.plans) == 4

    prefetch_owner = ("overlapping-prefetch", key)
    prefetch_plans = key.plans[:2]
    assert pyramid.claim_plans(prefetch_plans, prefetch_owner) == prefetch_plans

    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _desired_materialization_step(level=2)
    assert effects.prepare_rung(intent, step)
    request = session.lifecycle.materialization_request_for(0, key)
    assert request is not None
    assert request.claimed_plans == key.plans[2:]

    # The visible worker completes only what it owns. Whole-set exactness is a
    # GUI admission decision, so the foreign interior is a declined/replan
    # state rather than a worker exception.
    worker_result = effects.evaluate_rung(intent, step)()
    assert worker_result == ("materialized", request)
    assert pyramid.exact_pages(key.plans) is None
    assert pyramid.pending_count == len(prefetch_plans)
    effects._admit_ready_payloads(((step, worker_result),))
    assert session.lifecycle.materialization_request_for(0, key) is None
    assert session.lifecycle.dangling_claims() == ()
    assert session._test_replan_requested is True

    for plan in prefetch_plans:
        pyramid.admit_as(
            plan.key,
            materialize_lod_page(source, source_origin_yx=(0, 0), plan=plan),
            owner=prefetch_owner,
        )
    pyramid.release_owner_claims(prefetch_owner)
    exact_pages = pyramid.exact_pages(key.plans)
    assert exact_pages is not None
    assert session.admit_preview_plane(0, key, exact_pages)
    assert pyramid.pending_count == 0


def test_coarse_ancestor_does_not_suppress_finer_materialization_claim():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=1)
    _settle_first_pixels(session)
    rendered = session.rendered_tiles[0]
    demand = session.lod_policy_decision.demand
    semantic_id = session.tile_semantic_source_id(0)
    coarse = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=4,
        semantic_source_id=semantic_id,
    )
    target = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=semantic_id,
    )
    _admit_page_set(pyramid, coarse, np.asarray(rendered.image))
    assert render_lod._page_set_complete(pyramid, target)
    assert not render_lod._page_set_exact(pyramid, target)
    effects = FramePipelineEffects(_RungPrepareRenderer(), session)
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=False,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="finer target",
    )

    assert effects.prepare_rung(_pipeline_intent_for(session), step)
    request = session.lifecycle.materialization_request_for(0, target)
    assert request is not None
    assert request.key == target
    assert tuple(plan.key for plan in request.claimed_plans) == target.page_keys
    session.pending_rung_materializations.release(request)
    assert pyramid.pending_count == 0


def test_coarse_ancestor_does_not_skip_retained_finer_preview():
    preview = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=preview, count=1)
    rendered = session.rendered_tiles[0]
    demand = session.lod_policy_decision.demand
    semantic_id = session.tile_semantic_source_id(0)
    coarse = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=4,
        semantic_source_id=semantic_id,
    )
    target = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=semantic_id,
    )
    _admit_page_set(preview, coarse, np.asarray(rendered.image))
    assert render_lod._page_set_complete(preview, target)
    assert not render_lod._page_set_exact(preview, target)

    admitted = admit_retained_preview_level(
        preview,
        rendered,
        semantic_source_id=semantic_id,
        preview_level=2,
    )

    assert admitted == target
    assert render_lod._page_set_exact(preview, target)


def test_oversized_finer_page_releases_claim_instead_of_admitting_coarse_fallback():
    # L4 is 4x4 float32 (64 bytes), while the requested L2 page is 16x16
    # (1024 bytes).  The target is drawable through L4 but cannot be admitted
    # exactly under this cache budget.
    pyramid = LodPageCache(max_bytes=64)
    session = _session(pyramid=pyramid, count=1)
    rendered = session.rendered_tiles[0]
    demand = session.lod_policy_decision.demand
    semantic_id = session.tile_semantic_source_id(0)
    coarse = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=4,
        semantic_source_id=semantic_id,
    )
    target = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=semantic_id,
    )
    _admit_page_set(pyramid, coarse, np.asarray(rendered.image))
    request = render_lod.plan_materialization(
        session,
        rendered,
        demand=demand,
        level=2,
        key=target,
    )
    session.pending_rung_materializations.append(request)
    session.pending_rung_materializations.mark_started(request)
    for page_plan in request.claimed_plans:
        page = materialize_lod_page(
            request.source,
            source_origin_yx=request.source_origin_yx,
            plan=page_plan,
        )
        pyramid.admit_as(page_plan.key, page, owner=request.owner)

    assert render_lod._page_set_complete(pyramid, target)
    assert not render_lod._page_set_exact(pyramid, target)
    assert session.pending_rung_materializations.mark_resident(request) is False
    assert pyramid.pending_count == 0
    assert session.lifecycle.dangling_claims() == ()


def test_exact_page_eviction_before_gui_admission_declines_and_replans():
    """A completed worker may lose bounded-cache residency before GUI admission."""

    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=1)
    _settle_first_pixels(session)
    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _desired_materialization_step()

    assert effects.prepare_rung(intent, step)
    request = session.lifecycle.materialization_request_for(0)
    assert request is not None
    for page_plan in request.claimed_plans:
        page = materialize_lod_page(
            request.source,
            source_origin_yx=request.source_origin_yx,
            plan=page_plan,
        )
        pyramid.admit_as(page_plan.key, page, owner=request.owner)
    assert render_lod._page_set_exact(pyramid, request.key)

    pyramid.resize(max_entries=0)
    assert not render_lod._page_set_exact(pyramid, request.key)

    effects._admit_ready_payloads(((step, ("materialized", request)),))

    assert session.lifecycle.materialization_request_for(0) is None
    assert session.lifecycle.dangling_claims() == ()
    assert pyramid.pending_count == 0
    assert session.lod_materializations_completed == 0
    assert session._test_replan_requested is True


def test_materialized_rung_for_removed_tile_releases_lifecycle_claim():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=1)
    _settle_first_pixels(session)
    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _desired_materialization_step()

    assert effects.prepare_rung(intent, step)
    request = session.lifecycle.materialization_request_for(0)
    assert request is not None
    for page_plan in request.claimed_plans:
        page = materialize_lod_page(
            request.source,
            source_origin_yx=request.source_origin_yx,
            plan=page_plan,
        )
        pyramid.admit_as(page_plan.key, page, owner=request.owner)
    assert render_lod._page_set_exact(pyramid, request.key)

    old_tile = session.plan.tiles[0]
    session.plan = replace(
        session.plan,
        tiles=(replace(old_tile, montage_index=9, source_index=9),),
    )
    effects._admit_ready_payloads(((step, ("materialized", request)),))

    assert session.lifecycle.materialization_request_for(0) is None
    assert session.lifecycle.dangling_claims() == ()
    assert pyramid.pending_count == 0
    assert session.lod_materializations_completed == 0
    assert session._test_replan_requested is True


def test_dropped_rung_for_removed_tile_releases_page_and_lifecycle_claims():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=1)
    _settle_first_pixels(session)
    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _desired_materialization_step()

    assert effects.prepare_rung(intent, step)
    request = session.lifecycle.materialization_request_for(0)
    assert request is not None
    assert pyramid.pending_count == len(request.claimed_plans)

    old_tile = session.plan.tiles[0]
    session.plan = replace(
        session.plan,
        tiles=(replace(old_tile, montage_index=9, source_index=9),),
    )
    effects.rung_dropped(intent, step)

    assert session.lifecycle.materialization_request_for(0) is None
    assert session.lifecycle.dangling_claims() == ()
    assert pyramid.pending_count == 0
    assert session._test_replan_requested is True


def test_memory_pressure_admits_reduced_level_with_distinct_identity_and_shape():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    _settle_first_pixels(session)
    _state, delta = session.build_tile_presentation({})
    native_ids = {tile: payload.source_id for tile, payload in delta.upserts.items()}
    session.tile_residency_budget_bytes = (
        sum(int(np.asarray(payload.texture_data).nbytes) for payload in delta.upserts.values()) - 1
    )

    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    for request in requests:
        _materialize(session, request)

    session.dirty_payloads.update({0: None, 1: None})
    _state, delta = session.build_tile_presentation({})

    assert session.lod_policy_decision.applied_level == 2
    assert set(delta.upserts) == {0, 1}
    for tile, payload in delta.upserts.items():
        assert payload.lod.level == 2
        assert payload.lod.factor == 4
        assert payload.page_backing is not None
        assert payload.page_backing.requested_lod.texture_shape == (TILE // 4, TILE // 4)
        assert payload.texture_data.shape[:2] == (TILE // 4, TILE // 4)
        # Presentation identity separates levels: a reduced payload can never
        # share a residency key with the native payload it replaces.
        assert payload.source_id != native_ids[tile]
        assert payload.image.shape[:2] == (TILE // 4, TILE // 4)
        assert payload.semantic_data.shape[:2] == (TILE, TILE)
        # Exact semantic sources are untouched by display LOD.
        assert payload.semantic_histogram_data.shape[:2] == (TILE, TILE)
    assert session.pending_rung_materializations == []


def test_pressure_uses_available_coarse_per_tile_and_reports_common_level():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    _settle_first_pixels(session)
    _state, native_delta = session.build_tile_presentation({})
    session.tile_residency_budget_bytes = (
        sum(
            int(np.asarray(payload.texture_data).nbytes)
            for payload in native_delta.upserts.values()
        )
        - 1
    )
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    # Materialize the demanded level for tile 0 only.
    for request in requests:
        if request.tile_number == 0:
            _materialize(session, request)
        else:
            _release(session, request)

    session.dirty_payloads.update({0: None, 1: None})
    _state, delta = session.build_tile_presentation({})

    # Tile 0 presents the reduced level; tile 1 retains native until its
    # replacement is resident.
    assert delta.upserts[0].lod.level == 2
    assert delta.upserts[1].lod.level == 0
    assert delta.upserts[0].texture_data.shape[:2] == (TILE // 4, TILE // 4)
    assert delta.upserts[1].texture_data.shape[:2] == (TILE, TILE)
    assert delta.upserts[0].source_id != delta.upserts[1].source_id
    # The session-wide decision only claims what every tile can present.
    assert session.lod_policy_decision.applied_level == 0


def test_threshold_recrossing_hits_the_pyramid_cache_without_new_requests():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    _settle_first_pixels(session)
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    for request in requests:
        _materialize(session, request)
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})
    assert session.lod_policy_decision.applied_level == 2

    # Zoom in to native...
    session.view_range = ((0.0, float(2 * TILE)), (0.0, float(TILE)))
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})
    assert session.lod_policy_decision.applied_level == 0

    # ...and back out: the level comes from the cache, no new materialization.
    revision_before = pyramid.revision
    session.view_range = ZOOMED_OUT_RANGE
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})

    assert session.lod_policy_decision.applied_level == 2
    assert session.pending_rung_materializations == []
    assert pyramid.revision == revision_before
    assert pyramid.pending_count == 0


def test_no_reduction_work_happens_inside_presentation_builds(monkeypatch):
    import arrayscope.display.pyramid as pyramid_module

    def _fail(*_args, **_kwargs):
        raise AssertionError("reduce_box_mean must never run in a presentation build")

    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    monkeypatch.setattr(pyramid_module, "reduce_box_mean", _fail)

    session.build_tile_presentation({})
    session.snapshot_display_tile_payloads({})


def _cold_session(*, pyramid, count=2):
    """A resident-mode session whose tiles have not been computed yet."""

    session = _session(pyramid=pyramid, count=count)
    session.rendered_tiles.clear()
    session.display_tile_payloads.clear()
    session.dirty_payloads.clear()
    return session


def _rendered(tile, offset: float = 0.0) -> RenderedTile:
    image = np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE) + offset
    return RenderedTile(
        tile=tile,
        image=image,
        histogram_data=image,
        eval_ms=0.0,
        slab_shape=image.shape,
        slab_nbytes=image.nbytes,
    )


def test_worker_ingest_reduction_presents_demanded_level_first():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _cold_session(pyramid=pyramid)
    demand = session.ingest_lod_demand()
    assert demand is not None
    assert demand.desired_level == 2

    # Worker side: the native tile is computed, then reduced and admitted as
    # part of the same materialization, before the result reaches the GUI.
    rendered = _rendered(session.plan.tiles[0])
    assert (
        _admit_demand_level_for_test(
            pyramid,
            demand,
            rendered,
            semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
        )
        is not None
    )
    assert len(pyramid) == 1
    assert pyramid.pending_count == 0
    # Singleflight: the level is resident, a second admission is a no-op.
    assert (
        _admit_demand_level_for_test(
            pyramid,
            demand,
            rendered,
            semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
        )
        is None
    )

    # GUI side: the first presentation build selects the reduced level.  No
    # native payload is ever emitted for the tile and nothing is re-requested.
    session.mark_materialized(rendered)
    _state, delta = session.build_tile_presentation({})
    payload = delta.upserts[0]
    assert payload.lod.level == 2
    assert payload.lod.factor == 4
    assert payload.texture_data.shape[:2] == (TILE // 4, TILE // 4)
    assert payload.image.shape[:2] == (TILE // 4, TILE // 4)
    assert payload.histogram_data is None
    assert payload.semantic_histogram_data.shape[:2] == (TILE, TILE)
    # Exact semantic sources stay native.
    assert payload.semantic_data.shape[:2] == (TILE, TILE)
    assert payload.semantic_histogram_data.shape[:2] == (TILE, TILE)
    assert session.pending_rung_materializations == []


def test_native_only_and_native_scale_sessions_have_no_ingest_demand():
    assert _session(mode=LOD_POLICY_NATIVE_ONLY, pyramid=None).ingest_lod_demand() is None
    zoomed_in = _session(
        pyramid=LodPageCache(max_bytes=1 << 20),
        view_range=((0.0, float(2 * TILE)), (0.0, float(TILE))),
    )
    assert zoomed_in.ingest_lod_demand() is None


def test_demand_flip_materialization_cannot_demote_finer_native_without_pressure():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _cold_session(pyramid=pyramid)
    _settle_first_pixels(session)
    demand = session.ingest_lod_demand()
    assert demand.desired_level == 2

    # The viewport changes while the tile is in flight: level 1 is now wanted
    # (three source texels per screen pixel).
    session.view_range = ((0.0, 3.0 * 2 * TILE), (0.0, 3.0 * TILE))

    # The worker still completes against its scheduling-time snapshot.
    rendered = _rendered(session.plan.tiles[0])
    assert (
        _admit_demand_level_for_test(
            pyramid, demand, rendered, semantic_source_id=("test-tile", rendered.tile.source_index)
        )
        is not None
    )

    # Presentation never over-reduces with the stale level; the ordinary
    # streaming path may still populate level 1 for future pressure/reuse.
    session.mark_materialized(rendered)
    _state, delta = session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    assert session.lod_policy_decision.demand.desired_level == 1
    assert delta.upserts[0].lod.level == 0
    assert len(requests) == 1
    request = requests[0]
    assert request.tile_number == 0
    assert request.key.factor_xy == (2, 2)

    # No-demotion protects physical truth.  Make the native candidate a real
    # backend acknowledgement before the later materialization completes.
    _acknowledge(session, delta)

    _materialize(session, request)
    session.dirty_payloads[0] = None
    _state, delta = session.build_tile_presentation({})
    assert not delta.upserts, "resident coarse arrival must not churn acknowledged native pixels"
    presented = session.tile_presentation_state.payloads[0]
    assert presented.lod.level == 0
    assert presented.texture_data.shape[:2] == (TILE, TILE)


def _acknowledge(session, delta):
    session.acknowledge_tile_presentation(
        delta,
        TileCommitReport(
            presented_tiles=frozenset(int(tile) for tile in delta.upserts),
            committed_upserts=frozenset(int(tile) for tile in delta.upserts),
        ),
    )


def test_presented_lod_summary_reports_plurality_of_presented_payloads():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=3)
    _settle_first_pixels(session)

    # Nothing committed yet: fall back to the session-wide decision (native).
    assert session.presented_lod_summary() == (0, 1, (1, 1))

    _state, native_delta = session.build_tile_presentation({})
    session.tile_residency_budget_bytes = (
        sum(
            int(np.asarray(payload.texture_data).nbytes)
            for payload in native_delta.upserts.values()
        )
        - 1
    )
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    for request in requests:
        if request.tile_number in (0, 1):
            _materialize(session, request)
        else:
            _release(session, request)
    session.dirty_payloads.update({0: None, 1: None, 2: None})
    _state, delta = session.build_tile_presentation({})
    _acknowledge(session, delta)

    # Two of three tiles present the finest resident display level; the session-wide decision still
    # reads native because tile 2 is not resident yet.  Diagnostics must
    # describe the screen, not the consensus.
    assert session.lod_policy_decision.applied_level == 0
    assert session.presented_lod_summary() == (2, 4, (4, 4))


def test_presented_lod_summary_tie_prefers_the_finer_level():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=2)
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    for request in requests:
        if request.tile_number == 0:
            _materialize(session, request)
        else:
            _release(session, request)
    session.dirty_payloads.update({0: None, 1: None})
    _state, delta = session.build_tile_presentation({})
    _acknowledge(session, delta)

    assert session.presented_lod_summary() == (0, 1, (1, 1))


ZOOMED_IN_RANGE = ((0.0, float(2 * TILE)), (0.0, float(TILE)))


def _present_native(session):
    """Build, acknowledge, and mark presented so tiles own native payloads."""

    _state, delta = session.build_tile_presentation({})
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    session.pending_rung_materializations.clear()
    return delta


def _admit_zoomed_out_levels(session, level=2):
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in session.rendered_tiles.values():
        key = page_set_key_for_rendered(
            rendered,
            demand=demand,
            level=level,
            semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
        )
        _admit_page_set(session.lod_page_cache, key, np.asarray(rendered.image))


def test_camera_only_retarget_keeps_finer_native_without_residency_pressure():
    """Resident coarse data cannot demote exact finer pixels for bookkeeping."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    assert all(
        payload.lod.level == 0 for payload in session.tile_presentation_state.payloads.values()
    )
    _admit_zoomed_out_levels(session)

    # Camera-only zoom out: the demanded coarse level is already resident,
    # but atlas-class consolidation is not authority to lower visible quality.
    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    revision_before = int(session.viewport_revision)
    swap_ready = session.mark_ladder_swaps_for_viewport()

    assert swap_ready is False
    assert session.pending_rung_materializations == [], "cached levels must not be re-requested"
    assert not session.dirty_payloads

    # A worker may have built the coarse desired wrapper before the backend
    # acknowledges it. Desired bookkeeping is not physical presentation
    # truth: the acknowledged native payload must still win.
    acknowledged_native = session.tile_presentation_state.payloads[0]
    rendered = session.rendered_tiles[0]
    source, histogram, _kind = session._texture_source_for(rendered)
    coarse, coarse_histogram, coarse_lod, coarse_pages, coarse_kind = (
        session._resident_texture_for_rendered_tile(
            rendered,
            source=source,
            histogram=histogram,
        )
    )
    assert coarse_lod.level == 2
    coarse_source_id = session._payload_source_id(
        _base_source_id(acknowledged_native.source_id),
        texture_kind=coarse_kind,
        lod=coarse_lod,
    )
    coarse_identity = session.tile_payload_identity(
        rendered.tile,
        texture_data=coarse,
        texture_kind=coarse_kind,
        shader_mapping=rendered.shader_mapping,
        lod=coarse_lod,
        quality="exact",
    )
    session.display_tile_payloads[0] = replace(
        acknowledged_native,
        image=coarse,
        texture_data=coarse,
        histogram_data=coarse_histogram,
        source_id=coarse_source_id,
        texture_kind=coarse_kind,
        lod=coarse_lod,
        page_backing=coarse_pages,
        tile_identity=coarse_identity,
    )
    assert session.mark_ladder_swaps_for_viewport() is False
    assert render_lod.preserve_finer_presented_payload(session, acknowledged_native)

    cache_revision = pyramid.revision
    # Unrelated level/lifecycle work may rebuild a wrapper, but it still may
    # not select the coarser resident texture in the absence of pressure.
    ensure_payload = session._ensure_display_tile_payload
    ensured_levels = []

    def record_ensure(*args, **kwargs):
        payload = ensure_payload(*args, **kwargs)
        ensured_levels.append(int(payload.lod.level))
        return payload

    session._ensure_display_tile_payload = record_ensure
    session.dirty_payloads[0] = None
    _state, delta = session.build_tile_presentation({})
    assert ensured_levels == [0]
    assert not delta.upserts
    assert delta.removals == ()
    assert pyramid.revision == cache_revision
    assert {payload.lod.level for payload in session.display_tile_payloads.values()} == {0}

    # A second refresh with the same viewport is a no-op (no revision creep,
    # no commit request, no dirty tiles).
    assert session.mark_ladder_swaps_for_viewport() is False
    assert int(session.viewport_revision) >= revision_before


def test_logical_payload_budget_never_authorizes_visible_demotion():
    """Only physical backend pressure may replace finer visible pixels."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    _admit_zoomed_out_levels(session)
    native_bytes = sum(
        int(np.asarray(payload.texture_data).nbytes)
        for payload in session.tile_presentation_state.payloads.values()
    )
    session.tile_residency_budget_bytes = native_bytes - 1

    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    assert session.mark_ladder_swaps_for_viewport() is False
    assert not session.dirty_payloads

    state, delta = session.build_tile_presentation({})
    assert delta.upserts == {}
    assert delta.removals == ()
    assert {payload.lod.level for payload in state.payloads.values()} == {0}


def test_camera_only_retarget_requests_missing_levels_with_new_lod_revision():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    revision_before = int(session.lod_target_revision)

    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    swap_ready = session.mark_ladder_swaps_for_viewport()
    requests = list(_plan_rung_materializations(session))

    # Native payloads are already current and finer than the demand, so zoom
    # does not schedule coarser materialization or swap commits.
    assert swap_ready is False
    assert requests == []
    assert int(session.lod_target_revision) > revision_before
    # Native payloads stay presented untouched.
    assert not session.dirty_payloads
    assert all(
        payload.lod.level == 0 for payload in session.tile_presentation_state.payloads.values()
    )

    # Refreshes across the same zoom gesture remain no-ops.
    assert session.mark_ladder_swaps_for_viewport() is False
    assert list(_plan_rung_materializations(session)) == []


def test_refresh_is_native_only_noop():
    session = _session(mode=LOD_POLICY_NATIVE_ONLY, pyramid=None, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    assert session.mark_ladder_swaps_for_viewport() is False
    assert session.pending_rung_materializations == []
    assert not session.dirty_payloads


def test_logical_budget_cannot_turn_dirty_rebuild_into_demotion():
    """Unrelated dirty work preserves acknowledged finer payloads."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    native_payloads = dict(session.tile_presentation_state.payloads)
    assert set(native_payloads) == {0, 1}
    session.tile_residency_budget_bytes = (
        sum(int(np.asarray(payload.texture_data).nbytes) for payload in native_payloads.values())
        - 1
    )
    _admit_zoomed_out_levels(session)
    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    session.dirty_payloads.update({0: None, 1: None})

    # A session-side byte estimate cannot decide that native pixels should be
    # replaced. The backend owns physical reclaim and keeps this complete
    # predecessor if a candidate cannot fit.
    state, delta = session.build_tile_presentation({}, max_upserts=1)
    assert delta.removals == ()
    assert delta.upserts == {}
    assert dict(state.payloads) == native_payloads
    states = session.ensure_tile_states()
    assert str(states[0].value) == "loaded"
    assert str(states[1].value) == "loaded"


def test_previous_payloads_keep_visible_tiles_active_when_replacement_not_ready():
    """Retained visible pixels are active presentation, not a blank gap."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    retained_payloads = dict(session.tile_presentation_state.payloads)

    # Model a retarget/reload moment where lifecycle/presentation state still
    # owns compatible visible pixels, but the new target payload map is not
    # ready yet. The visible layer must keep presenting those pixels instead
    # of shrinking active scope to black.
    session.display_tile_payloads.clear()
    session.rendered_tiles.clear()
    session.dirty_payloads.update({0: None, 1: None})

    state, delta = session.build_tile_presentation({}, max_upserts=0)

    assert delta.removals == ()
    assert delta.upserts == {}
    assert delta.active_tiles == (0, 1)
    assert dict(state.payloads) == retained_payloads
    assert str(session.ensure_tile_states()[0].value) == "loaded"
    assert str(session.ensure_tile_states()[1].value) == "loaded"


def test_seeding_new_session_keeps_stale_level_payload_ready_for_identity_ack():
    """Session replacement must reuse a resident payload at the *old* level.

    When the pyramid no longer holds the seeded level, the tile keeps
    the stale-level payload ready for an identity-checked remap instead of
    rebuilding it from native.
    """

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_OUT_RANGE)
    _admit_zoomed_out_levels(session)
    _state, delta = session.build_tile_presentation({})
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    previous_payloads = dict(session.tile_presentation_state.payloads)
    assert {payload.lod.level for payload in previous_payloads.values()} == {2}

    # Fresh session for the same content and viewport; the pyramid cache was
    # dropped in between (worst case for seeding).
    replacement = _session(pyramid=LodPageCache(max_bytes=1 << 24), view_range=ZOOMED_OUT_RANGE)
    for index, rendered in tuple(session.rendered_tiles.items()):
        replacement.rendered_tiles[int(index)] = rendered
        replacement.dirty_payloads[int(index)] = None
    replacement.seed_display_tile_payloads(previous_payloads, {})

    # Both tiles are retained as desired payloads, still at level 2.  They do
    # not enter committed presentation state until backend identity truth says
    # those slots actually hold the retained payloads.
    assert set(replacement.display_tile_payloads) == {0, 1}
    assert {payload.lod.level for payload in replacement.display_tile_payloads.values()} == {2}
    assert set(replacement.pending_payload_upserts) == {0, 1}
    assert replacement.lifecycle.presented_tiles == frozenset()

    # The refresh accepts the retained level as resident evidence: no
    # down-swap churn to native while the demanded level rematerializes.
    assert replacement.mark_ladder_swaps_for_viewport() is False
    assert not replacement.dirty_payloads or set(replacement.dirty_payloads) == {0, 1}


def test_seed_display_payloads_resolves_retained_sources_by_semantic_key():
    """Retained payload caches are source-keyed, never iteration-slot keyed."""

    retained = _session(count=4)
    source_ids = {index: ("src", index) for index in range(4)}
    _state, delta = retained.build_tile_presentation(source_ids)
    _acknowledge(retained, delta)
    retained.mark_presented(tuple(delta.upserts))
    retained_payloads = {
        ("src", 2): retained.tile_presentation_state.payloads[2],
        ("src", 3): retained.tile_presentation_state.payloads[3],
    }

    partial = _session(count=2)
    partial.plan = _shifted_plan(count=2, offset=2)
    partial.visible_tiles = tuple(partial.plan.tiles)
    partial.rendered_tiles = {
        int(tile.montage_index): replace(retained.rendered_tiles[int(tile.source_index)], tile=tile)
        for tile in partial.plan.tiles
    }
    partial.dirty_payloads.update({0: None, 1: None})

    partial.seed_display_tile_payloads(
        retained_payloads,
        {0: ("src", 2), 1: ("src", 3)},
    )

    assert set(partial.display_tile_payloads) == {0, 1}
    assert partial.display_tile_payloads[0].source_index == 2
    assert partial.display_tile_payloads[1].source_index == 3
    assert (
        partial.display_tile_payloads[0].source_id
        == retained.tile_presentation_state.payloads[2].source_id
    )
    assert (
        partial.display_tile_payloads[1].source_id
        == retained.tile_presentation_state.payloads[3].source_id
    )
    assert set(partial.pending_payload_upserts) == {0, 1}


# --- Zero redundant histogram/level work across LOD levels (ADR 0050) ---

LEVEL1_RANGE = ((0.0, 3.0 * 2 * TILE), (0.0, 3.0 * TILE))
# From a presented factor-2 demand, promotion hysteresis needs > 4.6 source
# texels per screen pixel before level 2 becomes desired.
FAR_OUT_RANGE = ((0.0, 6.0 * 2 * TILE), (0.0, 6.0 * TILE))


def _attach_native_stats(session):
    from dataclasses import replace as dc_replace

    from arrayscope.display.model.montage_levels import sample_tile_level_stats

    for index, rendered in dict(session.rendered_tiles).items():
        stats = sample_tile_level_stats(
            rendered.image, int(rendered.tile.source_index), refined=True
        )
        session.rendered_tiles[index] = dc_replace(
            rendered, level_data=rendered.image, level_stats=stats
        )


def test_level_swap_carries_native_stats_and_recomputes_nothing():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _attach_native_stats(session)
    _state, native_delta = session.build_tile_presentation({})
    assert native_delta.upserts
    assert session.lod_stats_recomputes == 0

    # Exercise a legitimate initial reduced presentation: the native wrappers
    # above were never acknowledged, so resident L2 may become the first
    # physical commit.  Logical byte estimates are deliberately irrelevant.
    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    _admit_zoomed_out_levels(session)
    session.dirty_payloads.update({0: None, 1: None})
    _state, delta = session.build_tile_presentation({})

    for tile_number, payload in delta.upserts.items():
        rendered = session.rendered_tiles[int(tile_number)]
        assert payload.lod.level == 2
        # The finest already-computed semantic stats ride along unchanged:
        # a display-LOD swap is invisible to the histogram/level system.
        assert payload.semantic_histogram_data is np.asarray(rendered.histogram_data)
        assert payload.level_data is np.asarray(rendered.level_data)
        assert payload.level_stats is rendered.level_stats
        assert payload.semantic_data is np.asarray(rendered.image)
    assert session.lod_stats_cross_level_reuses == len(delta.upserts) > 0
    assert session.lod_stats_recomputes == 0
    _acknowledge(session, delta)

    # Moving finer (level 2 -> native) reuses the same stats objects too.
    session.retarget_viewport(view_range=ZOOMED_IN_RANGE, viewport_shape=VIEWPORT)
    session.mark_ladder_swaps_for_viewport()
    _state, delta = session.build_tile_presentation({})
    for tile_number, payload in delta.upserts.items():
        rendered = session.rendered_tiles[int(tile_number)]
        assert payload.lod.level == 0
        assert payload.level_stats is rendered.level_stats
        assert payload.histogram_data is np.asarray(rendered.histogram_data)
    assert session.lod_stats_recomputes == 0


def test_level_swap_keeps_semantic_histogram_identity():
    from arrayscope.display.model.tiled_histogram_identity import (
        tiled_histogram_key as _tiled_histogram_key,
    )
    from arrayscope.display.model.tiled_histogram_identity import (
        tiled_semantic_histogram_identity as _tiled_semantic_histogram_identity,
    )

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _state, native_delta = session.build_tile_presentation({})
    native_payloads = dict(native_delta.upserts)
    assert native_payloads

    # The unacknowledged native candidates may be replaced by the first
    # reduced commit; histogram identity must still remain semantic/native.
    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    _admit_zoomed_out_levels(session)
    session.dirty_payloads.update({0: None, 1: None})
    _state, delta = session.build_tile_presentation({})
    swapped_payloads = {**native_payloads, **dict(delta.upserts)}
    assert any(payload.lod.level == 2 for payload in swapped_payloads.values())

    # Texture identity changed for every swapped tile...
    assert {payload.source_id for payload in swapped_payloads.values()} != {
        payload.source_id for payload in native_payloads.values()
    }
    # ...but the semantic histogram identity, and therefore the histogram
    # stream key, is unchanged: a level swap produces ZERO histogram work.
    assert _tiled_semantic_histogram_identity(
        swapped_payloads
    ) == _tiled_semantic_histogram_identity(native_payloads)
    key_before = _tiled_histogram_key(
        (0.0, 1.0),
        histogram_plot_data=None,
        tile_delta=delta,
        semantic_identity=_tiled_semantic_histogram_identity(native_payloads),
    )
    key_after = _tiled_histogram_key(
        (0.0, 1.0),
        histogram_plot_data=None,
        tile_delta=delta,
        semantic_identity=_tiled_semantic_histogram_identity(swapped_payloads),
    )
    assert key_before == key_after


def test_coarser_level_stays_direct_source_with_finer_pages_resident():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=LEVEL1_RANGE)
    _settle_first_pixels(session)
    session.build_tile_presentation({})
    level1_requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    assert {request.key.level for request in level1_requests} == {1}
    for request in level1_requests:
        _materialize(session, request)

    session.retarget_viewport(view_range=FAR_OUT_RANGE, viewport_shape=VIEWPORT)
    session.mark_ladder_swaps_for_viewport()
    level2_requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    assert {request.key.level for request in level2_requests} == {2}
    for request in level2_requests:
        assert request.source.shape[:2] == (TILE, TILE)
        assert request.key.factor_xy == (4, 4)
        key, _pages = _materialize(session, request)
        page = pyramid.resolved_pages(key.plans)[0]
        rendered = session.rendered_tiles[int(request.tile_number)]
        native = reduce_box_mean(np.asarray(rendered.image), (4, 4))
        assert np.allclose(page.values, native, rtol=1e-6, atol=1e-6)


def test_uneven_tiles_fall_back_to_native_reduction_source():
    # 63 is not divisible by 4: partial trailing boxes must always reduce
    # from the single canonical native plane so level content never depends
    # on which levels happened to be resident.
    tile = 63
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=LEVEL1_RANGE)
    _settle_first_pixels(session)
    for index, rendered in dict(session.rendered_tiles).items():
        from dataclasses import replace as dc_replace

        image = np.asarray(rendered.image)[:tile, :tile].copy()
        session.rendered_tiles[index] = dc_replace(rendered, image=image, histogram_data=image)
    session.build_tile_presentation({})
    for request in list(_plan_rung_materializations(session)):
        _materialize(session, request)
    session.pending_rung_materializations.clear()

    session.retarget_viewport(view_range=FAR_OUT_RANGE, viewport_shape=VIEWPORT)
    session.mark_ladder_swaps_for_viewport()
    requests = list(_plan_rung_materializations(session))
    assert requests, "the coarser level must still be requested"
    for request in requests:
        assert request.source.shape[:2] == (tile, tile)
        assert request.reduce_factor_xy == request.key.factor_xy


def test_floor_presents_resident_level_for_unrendered_tile_instead_of_placeholder():
    """ADR 0050 floor invariant: any resident level beats a placeholder."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))

    # Tile 1 was materialized at level 2 in an earlier pass (semantic key),
    # then its rendered object was dropped — e.g. a pan re-entered the tile.
    rendered = session.rendered_tiles[1]
    key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    _admit_page_set(pyramid, key, np.asarray(rendered.image))
    _claim_preview_resident(session, 1, key)
    del session.rendered_tiles[1]
    session.dirty_payloads.clear()

    _state, delta = session.build_tile_presentation({})

    payload = delta.upserts.get(1) or session.display_tile_payloads.get(1)
    assert payload is not None, "unrendered tile with a resident level must present it"
    assert payload.quality == "preview"
    assert payload.lod.level == 2
    assert payload.texture_data.shape[:2] == (TILE // 4, TILE // 4)

    # Preview payloads never provide semantic values.
    from arrayscope.display.model.frame import TiledValueSource

    source = TiledValueSource(payloads={1: payload})

    class _Mapping:
        tile_number = 1
        local_y = 0
        local_x = 0

    assert source.value_at(_Mapping()) is None

    # Survival/no-flicker is a physical claim; acknowledge the first floor
    # before checking later presentation builds.
    _acknowledge(session, delta)

    # The floor payload survives subsequent builds while the tile stays
    # planned-but-unrendered (no flicker back to placeholder).
    _state, delta2 = session.build_tile_presentation({})
    assert 1 in session.display_tile_payloads
    assert 1 not in delta2.removals if hasattr(delta2, "removals") else True

    # When the exact result arrives, it replaces the preview.
    image = np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE) + 1
    session.rendered_tiles[1] = RenderedTile(
        tile=session.plan.tiles[1],
        image=image,
        histogram_data=image,
        eval_ms=0.0,
        slab_shape=image.shape,
        slab_nbytes=image.nbytes,
    )
    session.dirty_payloads[1] = None
    _state, _delta3 = session.build_tile_presentation({})
    replaced = session.display_tile_payloads[1]
    assert replaced.quality == "exact"


def test_floors_survive_index_window_changes_via_semantic_key():
    """Field defect 2026-07-05 (missing corner tiles 'there in other views'):
    the pyramid identity was keyed by the session key, which includes the
    sibling-index selection — every index-window change renamed identical
    texels and refilled previously computed tiles cold from black.  Sessions
    sharing a window-agnostic ``semantic_key`` must share floors."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    semantic = ("texels", "doc", 2)

    session_a = _session(pyramid=pyramid, count=2)
    session_a.semantic_key = semantic
    session_a.key = ("window", "4:104")
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session_a.rendered_tiles[1]
    key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=session_a.tile_semantic_source_id(rendered.tile.source_index),
    )
    _admit_page_set(pyramid, key, np.asarray(rendered.image))

    session_b = _session(pyramid=pyramid, count=2)
    session_b.semantic_key = semantic
    session_b.key = ("window", "100:200")
    del session_b.rendered_tiles[1]
    session_b.dirty_payloads.pop(1, None)

    from arrayscope.render import lod as render_lod

    best = render_lod.best_floor_key(session_b, 1)
    assert best is not None, "floor computed under window A must be resident under window B"
    assert best[1] == 2

    _state, delta = session_b.build_tile_presentation({})
    payload = delta.upserts.get(1) or session_b.display_tile_payloads.get(1)
    assert payload is not None
    assert payload.quality == "preview"


def test_backend_reported_identities_drive_convergence():
    """ADR 0051 rule 1, ground-truth edition (field defects 2026-07-05): the
    session's own acknowledgement records lied in every stale-LOD wedge, so
    convergence now compares against the identities the backend reports its
    drawn slots ACTUALLY hold.  A mismatch re-presents the tile inside the
    active scope; agreement settles; retries are bounded."""

    session = _session(pyramid=LodPageCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    current_identity = {
        int(tile): tile_ack_identity(payload) for tile, payload in dict(delta.upserts).items()
    }
    # Backend says tile 1's slot still shows some OLD identity.
    report = TileCommitReport(
        presented_tiles=(0, 1),
        committed_upserts=(0, 1),
        delta_key=(int(delta.base_revision), int(delta.target_revision)),
        presented_identities={0: current_identity[0], 1: ("stale", 6)},
    )
    session.acknowledge_tile_presentation(delta, report)
    session.mark_presented((0, 1))

    _state2, delta2 = session.build_tile_presentation({})
    assert 1 in delta2.upserts, "backend-stale tile must re-present"
    assert 1 in tuple(delta2.active_tiles)
    assert 0 not in delta2.upserts, "agreeing tile stays settled"

    # Backend now confirms the wanted identity: converged, no more work.
    report2 = TileCommitReport(
        presented_tiles=(0, 1),
        committed_upserts=(1,),
        delta_key=(int(delta2.base_revision), int(delta2.target_revision)),
        presented_identities=dict(current_identity),
    )
    session.acknowledge_tile_presentation(delta2, report2)
    _state3, delta3 = session.build_tile_presentation({})
    assert 1 not in delta3.upserts
    assert not session._identity_retry_attempts


def test_backend_identity_retries_are_bounded():
    """A backend that cannot converge must not turn identity retry into a loop."""

    session = _session(pyramid=LodPageCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    report = TileCommitReport(
        presented_tiles=(0, 1),
        committed_upserts=(0, 1),
        delta_key=(int(delta.base_revision), int(delta.target_revision)),
        presented_identities={1: ("stuck", 6)},
    )
    session.acknowledge_tile_presentation(delta, report)
    emitted = 0
    last_delta = delta
    for _round in range(6):
        _state, last_delta = session.build_tile_presentation({})
        if 1 in last_delta.upserts:
            emitted += 1
            # Backend keeps reporting the same stuck identity.
            stuck = TileCommitReport(
                presented_tiles=(0, 1),
                delta_key=(int(last_delta.base_revision), int(last_delta.target_revision)),
                presented_identities={1: ("stuck", 6)},
            )
            session.acknowledge_tile_presentation(last_delta, stuck)
    assert emitted <= 3, f"unbounded identity retries: {emitted}"


def test_settled_mismatch_is_queryable_for_followup_commit():
    """Backend slot mismatches remain queryable after acknowledgement.

    The renderer asks ``backend_identity_mismatch_tiles()`` after every
    acknowledgement; it must report actionable mismatches, exclude resigned
    pairs and exhausted
    attempts, and empty out on convergence."""

    session = _session(pyramid=LodPageCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    current_identity = {
        int(tile): tile_ack_identity(payload) for tile, payload in dict(delta.upserts).items()
    }
    report = TileCommitReport(
        presented_tiles=(0, 1),
        committed_upserts=(0, 1),
        delta_key=(int(delta.base_revision), int(delta.target_revision)),
        presented_identities={0: current_identity[0], 1: ("stale", 6)},
    )
    session.acknowledge_tile_presentation(delta, report)
    assert session.backend_identity_mismatch_tiles() == (1,), (
        "the ack itself must expose the settled wedge for a follow-up commit"
    )
    # Convergence empties the query.
    _state2, delta2 = session.build_tile_presentation({})
    session.acknowledge_tile_presentation(
        delta2,
        TileCommitReport(
            presented_tiles=(0, 1),
            committed_upserts=tuple(delta2.upserts),
            delta_key=(int(delta2.base_revision), int(delta2.target_revision)),
            presented_identities=dict(current_identity),
        ),
    )
    assert session.backend_identity_mismatch_tiles() == ()


def test_inherited_identities_heal_on_a_rebuilt_session():
    """A rebuilt session (scrub step) inherits backend slots — and now also
    the identity ground truth (renderer seeds lifecycle backend identities from
    the dying session).  Its first build must repair inherited stale slots
    instead of settling blind on top of them (sid-68 wedge, JSONL 131233)."""

    session = _session(pyramid=LodPageCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    # Fresh session, fresh machine: only the inherited map knows slot 1 is
    # stale.  (The renderer copies this dict across replacement.)
    backend_truth = {
        int(tile): tile_ack_identity(payload) for tile, payload in dict(delta.upserts).items()
    }
    backend_truth[1] = ("previous-session-level", 5)
    session.lifecycle.backend_presented_snapshot(backend_truth)
    session.dirty_payloads.clear()
    session.pending_payload_upserts.clear()
    assert session.backend_identity_mismatch_tiles() == (1,)
    _state2, delta2 = session.build_tile_presentation({})
    assert 1 in delta2.upserts, "inherited stale slot must re-present"
    assert 1 in tuple(delta2.active_tiles)


def test_resigned_pair_stops_convergence_but_new_result_rearms():
    """The machine's resignation (bounded identity rejections) must silence
    exactly the resigned (wanted, shown) pair — the mismatch query and the
    build identity path skip it — while a fresh evaluation clears the
    resignation and gets new chances."""

    session = _session(pyramid=LodPageCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    wanted = tile_ack_identity(dict(delta.upserts)[1])
    rec = session.lifecycle.record(1)
    rec.resigned.add((wanted, ("stuck", 6)))
    session.lifecycle.backend_presented_snapshot({1: ("stuck", 6)})
    session.dirty_payloads.clear()
    session.pending_payload_upserts.clear()
    assert session.backend_identity_mismatch_tiles() == ()
    _state2, delta2 = session.build_tile_presentation({})
    assert 1 not in delta2.upserts, "resigned pair must not re-present"
    # A fresh semantic result clears the resignation.
    session.lifecycle.evaluation_completed(1)
    assert rec.resigned == set()


def test_report_bound_to_an_older_delta_acknowledges_nothing():
    """Field defect 2026-07-05 (JSONL 112841): a skipped/superseded commit
    leaves the committer's last report pointing at an OLDER delta; every ack
    site fetches that attribute, so the new delta's upserts were falsely
    accepted by tile-number intersection — 100 level-swap payloads
    acknowledged with zero layer uploads, stale LOD until the next real
    commit.  A report causally bound to a different delta must change
    nothing: dirty entries stay armed for the next flush."""

    session = _session(pyramid=LodPageCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    assert 0 in session.dirty_payloads
    assert 1 in session.dirty_payloads

    stale_report = TileCommitReport(
        presented_tiles=(0, 1),
        delta_key=(int(delta.base_revision) + 5, int(delta.target_revision) + 5),
    )
    acknowledged = session.acknowledge_tile_presentation(delta, stale_report)
    assert dict(acknowledged.payloads) == {}, "mismatched report must acknowledge nothing"
    assert 0 in session.dirty_payloads, "dirty stays armed"
    assert 1 in session.dirty_payloads, "dirty stays armed"
    assert session.parked_dirty_payloads == frozenset(), "and nothing parks"

    bound = TileCommitReport(
        presented_tiles=(0, 1),
        committed_upserts=(0, 1),
        delta_key=(int(delta.base_revision), int(delta.target_revision)),
    )
    acknowledged2 = session.acknowledge_tile_presentation(delta, bound)
    assert set(acknowledged2.payloads) == {0, 1}
    assert not session.dirty_payloads


def test_report_revision_collision_from_older_session_acknowledges_nothing():
    """Revision pairs repeat after retarget; session generation is causal."""

    session = _session(pyramid=LodPageCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    assert delta.transaction_generation == session.session_id

    stale = TileCommitReport(
        presented_tiles=(0, 1),
        committed_upserts=(0, 1),
        delta_key=(delta.base_revision, delta.target_revision),
        transaction_generation=int(session.session_id) - 1,
    )

    acknowledged = session.acknowledge_tile_presentation(delta, stale)

    assert dict(acknowledged.payloads) == {}
    assert set(session.dirty_payloads) == {0, 1}


def test_mark_presented_rejects_backend_active_tile_with_stale_identity():
    """A backend-visible slot is not current presentation unless its identity
    matches the session payload.  PyQtGraph can keep old items visible during
    a bounded retarget; those tiles must stay dirty instead of making the
    session appear settled."""

    from dataclasses import replace as dc_replace

    session = _session(pyramid=LodPageCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    report = TileCommitReport(presented_tiles=(0, 1), committed_upserts=(0, 1))
    session.acknowledge_tile_presentation(delta, report)
    session.mark_presented((0, 1))
    assert session.lifecycle.presented_tiles == frozenset({0, 1})

    old_identity = tile_ack_identity(session.display_tile_payloads[1])
    fresh_identity = dc_replace(old_identity, semantic_generation=("fresh", 1))
    session.display_tile_payloads[1] = dc_replace(
        session.display_tile_payloads[1],
        source_id=("fresh", 1),
        tile_identity=fresh_identity,
    )
    session.dirty_payloads.clear()
    session.lifecycle.backend_presented_snapshot(
        {
            0: tile_ack_identity(session.display_tile_payloads[0]),
            1: old_identity,
        }
    )

    session.mark_presented((0, 1))

    assert 0 in session.lifecycle.presented_tiles
    assert 1 not in session.lifecycle.presented_tiles
    assert 1 in session.dirty_payloads


def test_drawn_tile_with_outdated_acknowledged_identity_represents_and_rejoins_active():
    """Field defect 2026-07-05 (JSONL 110937, sid=81): level-2 swap upserts
    for near-scope tiles were declined and parked while the backend kept
    DRAWING their acknowledged level-6 slots — visibly stale rows no
    viewport-derived scope would ever repair.  A drawn tile whose
    acknowledged identity differs from the session's current payload must
    re-present and join the delta's active scope so the backend accepts it."""

    session = _session(pyramid=LodPageCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    report = TileCommitReport(presented_tiles=(0, 1), committed_upserts=(0, 1))
    session.acknowledge_tile_presentation(delta, report)
    session.mark_presented((0, 1))

    # Tile 1 leaves the viewport-derived scope but its slot stays drawn.
    session.visible_tiles = (session.plan.tiles[0],)
    # The backend still shows an OLDER identity than the session now holds.
    session.lifecycle.backend_presented_snapshot(
        {0: session.display_tile_payloads[0].source_id, 1: ("stale", 6)}
    )

    _state2, delta2 = session.build_tile_presentation({})
    assert 1 in delta2.upserts, "drawn tile with outdated acknowledged identity must re-present"
    assert 1 in tuple(delta2.active_tiles), "and must be acceptable to the viewport-scoped backend"

    report2 = TileCommitReport(presented_tiles=(0, 1), committed_upserts=(1,))
    session.acknowledge_tile_presentation(delta2, report2)
    _state3, delta3 = session.build_tile_presentation({})
    assert 1 not in delta3.upserts, "identity retry converges and the session settles"


def test_refresh_replans_missing_desired_level_at_unchanged_viewport():
    """Settle-repair contract (field defect 2026-07-05, JSONL 110937,
    sid=80): supersession can kill planned materializations after the last
    camera-driven refresh; a later refresh with an UNCHANGED viewport
    identity must still re-request the missing demanded level, or tiles
    wedge on a coarser resident level until the next pan."""

    from arrayscope.render import lod as render_lod

    session = _session(pyramid=LodPageCache(max_bytes=1 << 24), count=2)
    assert session.mark_ladder_swaps_for_viewport() is not None
    first = list(_plan_rung_materializations(session))
    assert first, "zoomed-out demand plans materializations for the missing level"

    # Simulate supersession/session churn dropping the planned work.
    released = render_lod.release_session_claims(session)
    assert released == sum(
        1 for request in first for step_key, _rel in request.chain if step_key is not None
    )
    assert not session.pending_rung_materializations

    session.mark_ladder_swaps_for_viewport()
    replanned = list(_plan_rung_materializations(session))
    assert replanned, "idle refresh must re-plan the demanded level after its claims were released"


def test_floor_presents_blank_tile_even_while_exact_evaluation_is_in_flight():
    """Field report 2026-07-05: slow stage-backed fills left tiles BLACK for
    seconds while their floor planes sat resident — because the floor pass
    skipped every tile with an active exact request.  A blank tile floors
    regardless of in-flight work; only an existing preview defers to the
    imminent exact replacement (anti-churn)."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))

    rendered = session.rendered_tiles[1]
    key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    _admit_page_set(pyramid, key, np.asarray(rendered.image))
    del session.rendered_tiles[1]
    session.dirty_payloads.clear()
    # The exact evaluation is in flight (slow stage compute).
    session.active_tile_requests.add(1)

    assert session._floor_can_progress(1)
    _state, delta = session.build_tile_presentation({})
    payload = delta.upserts.get(1) or session.display_tile_payloads.get(1)
    assert payload is not None, (
        "blank tile with resident floor must present it despite in-flight eval"
    )
    assert payload.quality == "preview"
    assert payload.lod.level == 2

    # Anti-churn: once the preview is on screen, the in-flight exact request
    # suppresses further floor improvements for this tile.
    assert not session._floor_can_progress(1)


def test_floor_payload_construction_honors_batch_cap():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=12)
    _admit_zoomed_out_levels(session)
    session.rendered_tiles.clear()
    session.display_tile_payloads.clear()
    session.pending_payload_upserts.clear()

    session._ensure_floor_payloads(tuple(range(12)), max_count=3)

    assert len(session.display_tile_payloads) == 3
    assert len(session.pending_payload_upserts) == 3
    assert session.lod_floor_presentations == 3


def test_floor_tile_with_native_demand_settles_instead_of_spinning():
    """Regression: a preview-floored tile under native demand must not keep
    the dirty set non-empty forever (100% single-core commit-loop spin)."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2, view_range=((0.0, 2 * TILE), (0.0, TILE)))
    zoomed_out = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[1]
    key = page_set_key_for_rendered(
        rendered,
        demand=zoomed_out,
        level=2,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    _admit_page_set(pyramid, key, np.asarray(rendered.image))
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)

    # Demand is native (zoomed in); the tile floors at level 2.
    _state, delta = session.build_tile_presentation({})
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    floored = session.display_tile_payloads.get(1)
    assert floored is not None
    assert floored.quality == "preview"

    # A camera-only refresh may request commits but must never mark the
    # unrendered tile dirty...
    session.mark_ladder_swaps_for_viewport()
    assert 1 not in session.dirty_payloads

    # ...and repeated builds settle: nothing dirty, nothing pending, the
    # floor payload keeps presenting.
    for _ in range(3):
        _state, delta = session.build_tile_presentation({})
        _acknowledge(session, delta)
        session.mark_presented(tuple(delta.upserts))
    assert 1 not in session.dirty_payloads
    assert not session.pending_payload_upserts
    assert session.display_tile_payloads[1].quality == "preview"


def test_floor_payload_upgrades_when_closer_level_becomes_resident():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[1]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    coarse = page_set_key_for_rendered(
        rendered, demand=demand, level=4, semantic_source_id=semantic_id
    )
    _admit_page_set(pyramid, coarse, np.asarray(rendered.image))
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)

    _state, delta = session.build_tile_presentation({})
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    initial = session.display_tile_payloads[1]
    assert initial.lod.level == 2
    assert initial.page_backing.requested_lod.level == 2
    assert initial.page_backing.resolved_page_set.coarsest_actual_level == 4
    assert initial.actual_lod_factor == 16
    assert initial.page_backing.materialized_pages[0].key.lod.reduction == (4, 4)
    record = session.lifecycle.peek(1)
    assert record is not None
    assert not record.target_settled

    # The demanded level 2 materializes later; the floor upgrades to it.
    better = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=semantic_id,
    )
    image = np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE) + 1
    _admit_page_set(pyramid, better, image)
    assert session._floor_can_progress(1)
    _state, delta = session.build_tile_presentation({})
    upgraded = session.display_tile_payloads[1]
    assert upgraded.quality == "preview"
    assert upgraded.lod.level == 2
    assert upgraded.page_backing.requested_lod.level == 2
    assert upgraded.page_backing.resolved_page_set.coarsest_actual_level == 2
    assert upgraded.actual_lod_factor == 4
    assert upgraded.page_backing.materialized_pages[0].key.lod.reduction == (2, 2)


def test_reduced_target_payload_is_not_preview_when_target_lod_is_reduced():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    assert int(demand.desired_level) > 0

    rendered = session.rendered_tiles[1]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = page_set_key_for_rendered(
        rendered, demand=demand, level=demand.desired_level, semantic_source_id=semantic_id
    )
    source, _histogram, texture_kind = texture_source_for_rendered(rendered)
    pages = _materialized_page_set(key, source)

    del session.rendered_tiles[1]
    session.dirty_payloads.clear()
    assert session.admit_preview_plane(
        1,
        key,
        pages,
        None,
        texture_kind=texture_kind,
        level_data=np.asarray([0.0, 3000.0], dtype=np.float32),
        quality="exact",
    )

    session._ensure_floor_payloads((1,))
    payload = session.display_tile_payloads[1]
    assert payload.quality == "exact"
    assert payload.lod.level == int(demand.desired_level)
    assert 1 in session.pending_payload_upserts

    # Shared transform targets have a materialization identity that is not the
    # native per-tile evaluator identity. Lifecycle target ownership, not the
    # renderer-local ``rendered_tiles`` map, keeps this payload current.
    payload = replace(payload, source_id=("shared-transform-target", 1, payload.source_id))
    session.display_tile_payloads[1] = payload
    session.record_tile_payload(payload)
    session.dirty_payloads[1] = None
    session.pending_payload_upserts[1] = None

    for _ in range(3):
        _state, delta = session.build_tile_presentation({})
        assert 1 not in delta.removals
        _acknowledge(session, delta)
        session.mark_presented(tuple(delta.upserts))
        if 1 in session.lifecycle.presented_tiles:
            break
    assert 1 in session.lifecycle.presented_tiles
    assert 1 not in session.unrefined_preview_tiles(include_already_dirty=True)


def test_reduced_target_promotes_existing_same_level_preview_wrapper():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles.pop(1)
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=demand.desired_level,
        semantic_source_id=semantic_id,
    )
    pages = _materialized_page_set(key, np.asarray(rendered.image))
    session.dirty_payloads.clear()

    assert session.admit_preview_plane(1, key, pages, quality="preview")
    session._ensure_floor_payloads((1,))
    assert session.display_tile_payloads[1].quality == "preview"
    session.pending_payload_upserts.clear()

    assert session.admit_preview_plane(1, key, pages, quality="exact")
    assert session._floor_can_progress(1)
    session._ensure_floor_payloads((1,))

    assert session.display_tile_payloads[1].quality == "exact"
    assert session.display_tile_payloads[1].lod.level == int(demand.desired_level)
    assert 1 in session.pending_payload_upserts


def test_exact_reduced_target_upgrades_to_finer_resident_level():
    """Resident target truth must also drive the presented payload level."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    desired = int(demand.desired_level)
    coarse_level = desired + 2
    rendered = session.rendered_tiles[1]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)

    def admit(level):
        key = page_set_key_for_rendered(
            rendered,
            demand=demand,
            level=int(level),
            semantic_source_id=semantic_id,
        )
        pages = _materialized_page_set(key, np.asarray(rendered.image))
        assert session.admit_preview_plane(1, key, pages, quality="exact")

    del session.rendered_tiles[1]
    session.dirty_payloads.clear()
    admit(coarse_level)
    session._ensure_floor_payloads((1,))
    assert session.display_tile_payloads[1].quality == "exact"
    coarse_payload = session.display_tile_payloads[1]
    assert coarse_payload.lod.level == coarse_level
    assert coarse_payload.page_backing.materialized_pages[0].key.lod.reduction == (
        coarse_level,
        coarse_level,
    )
    session.pending_payload_upserts.clear()

    admit(desired)
    assert session._floor_can_progress(1)
    session._ensure_floor_payloads((1,))
    assert session.display_tile_payloads[1].quality == "exact"
    assert session.display_tile_payloads[1].lod.level == desired
    assert 1 in session.pending_payload_upserts


def test_active_lifecycle_target_never_emits_removal_when_upsert_is_deferred():
    session = _session(mode=LOD_POLICY_NATIVE_ONLY, count=1)
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    image = np.ones((TILE, TILE), dtype=np.float32)
    payload = DisplayTilePayload(
        tile_number=0,
        source_index=0,
        image=image,
        histogram_data=image,
        source_id=("shared-transform-target", 0),
        lod=LodInfo(level=0, factor=1, source_shape=image.shape, texture_shape=image.shape),
        quality="exact",
    )
    session.display_tile_payloads[0] = payload
    session.record_tile_payload(payload)
    session.dirty_payloads[0] = None
    session.pending_payload_upserts[0] = None

    _state, current_delta = session.build_tile_presentation({}, max_upserts=0)

    assert current_delta.removals == ()


def test_unacknowledged_first_frame_payload_remains_pending_when_seeded_as_fallback():
    session = _session(mode=LOD_POLICY_NATIVE_ONLY, count=1)
    rendered = session.rendered_tiles[0]
    session._ensure_display_tile_payload(0, rendered, {}, lod_factor=1)
    payload = session.display_tile_payloads[0]
    session.record_tile_payload(payload)
    session.tile_presentation_state = TilePresentationState({0: payload})
    session.dirty_payloads[0] = None
    session.pending_payload_upserts[0] = None

    _state, delta = session.build_tile_presentation({}, max_upserts=1)

    assert 0 in delta.upserts
    assert 0 in session.pending_payload_upserts
    assert 0 not in session.lifecycle.presented_tiles


def test_same_plane_flip_rebinds_payload_semantic_identity():
    """A resident plane may be reused, but its wrapper must describe this view."""

    session = _session(mode=LOD_POLICY_NATIVE_ONLY, count=1)
    state = ViewState.from_shape((TILE, TILE, 8)).with_image_axes(0, 1)
    tile = replace(session.plan.tiles[0], view_state=state)
    rendered = replace(session.rendered_tiles[0], tile=tile)
    source_ids = {0: session.tile_semantic_source_id(0)}

    first = session._ensure_display_tile_payload(0, rendered, source_ids, lod_factor=1)
    flipped_state = state.with_axis_flipped(1, True)
    flipped_tile = replace(tile, view_state=flipped_state)
    expected = session.tile_payload_identity(
        flipped_tile,
        texture_data=first.texture_data,
        texture_kind=first.texture_kind,
        shader_mapping=first.shader_mapping,
        lod=first.lod,
        quality=first.quality,
    )
    from arrayscope.presentation.tile_lifecycle import TileTarget

    session.lifecycle.retarget(
        {
            0: TileTarget(
                tile_number=0,
                source_index=0,
                semantic_source_id=session.tile_semantic_source_id(0),
                lod_level=0,
                identity=expected,
            )
        }
    )
    assert not _payload_matches_current_tile(session, 0, first, {0: flipped_tile})
    flipped = session._ensure_display_tile_payload(
        0,
        replace(rendered, tile=flipped_tile),
        source_ids,
        lod_factor=1,
    )

    assert flipped.source_id == first.source_id
    assert flipped.texture_data is first.texture_data
    assert flipped is not first
    assert flipped.tile_identity.axis_flips == flipped_state.axis_flipped
    assert flipped.tile_identity != first.tile_identity


def test_lod_refresh_owns_its_supersession_counter_not_viewport_revision():
    """Regression: refresh bumped viewport_revision without replanning,
    churning priority-retarget work identities at idle."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid)
    before_viewport = int(session.viewport_revision)
    before_lod = int(getattr(session, "lod_target_revision", 0))
    session.mark_ladder_swaps_for_viewport()
    assert int(session.viewport_revision) == before_viewport
    assert int(session.lod_target_revision) == before_lod + 1
    # Unchanged viewport: no further bumps.
    session.mark_ladder_swaps_for_viewport()
    assert int(session.lod_target_revision) == before_lod + 1


def test_floor_payloads_never_stall_level_convergence():
    """Regression: preview payloads in the level scope kept the target
    permanently stale, spinning full commits at idle (60-75% CPU)."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[1]
    key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    _admit_page_set(pyramid, key, np.asarray(rendered.image))
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)

    _state, delta = session.build_tile_presentation({})
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    assert session.display_tile_payloads[1].quality == "preview"

    # A level target arrives; the floored tile must not count as stale
    # (it carries no semantic level evidence), so convergence can settle
    # on the exact tiles alone.
    session.begin_level_presentation_update((0.0, 1.0))
    session.update_level_presentation_scope()
    active = session.level_generation.active_tiles
    assert 1 not in active, "preview payloads must stay outside the level scope"


def test_orphaned_loading_state_does_not_suppress_pipeline_target():
    """A stale loading marker cannot become a second scheduling owner."""

    session = _session(pyramid=LodPageCache(max_bytes=1 << 20))
    # Simulate a lost evaluation: lifecycle still says loading, but no task
    # claim exists. The ladder must derive the missing producer from target
    # state instead of relying on a repair queue.
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)
    session.loading_tiles.add(1)
    demand = session.lod_policy_decision.demand
    states = render_effects.tile_lod_states(session, demand)
    policy = LadderPolicy(
        mode=session.lod_policy_mode,
        floor_level=4,
        reduced_input_available=True,
    )
    steps = LodLadder(policy).plan(states, demand)

    assert 1 in session.required_target_unsettled_tiles()
    assert any(int(step.tile_number) == 1 for step in steps)


def test_parked_dirty_tiles_rearm_when_the_viewport_makes_them_active():
    """Offscreen dirty work stays armed and presents when it becomes active."""

    session = _session(pyramid=LodPageCache(max_bytes=1 << 20), count=2)
    # Tile 1 rendered and dirty, but currently outside the active scope.
    session.visible_tiles = (session.plan.tiles[0],)
    session.visible_tile_numbers = frozenset({0})
    session.sync_lifecycle_scope()
    _state, delta = session.build_tile_presentation({})
    # First-pixel admission spends its bounded transaction only on visible
    # coverage. Offscreen prepared work is retained, not emitted and parked.
    assert 1 not in delta.upserts
    report = TileCommitReport(presented_tiles=(0,))
    session.acknowledge_tile_presentation(delta, report)
    assert 1 not in session.parked_dirty_payloads
    assert 1 in session.dirty_payloads

    # Idle does not emit the offscreen tile.
    _state, delta2 = session.build_tile_presentation({})
    assert 1 not in delta2.upserts
    assert 1 in session.dirty_payloads

    # The viewport brings tile 1 back: the parked entry re-arms and the
    # payload presents through the ordinary delta.
    session.visible_tiles = tuple(session.plan.tiles)
    session.visible_tile_numbers = frozenset({0, 1})
    session.sync_lifecycle_scope()
    _state, delta3 = session.build_tile_presentation({})
    assert 1 not in session.parked_dirty_payloads
    assert 1 in delta3.upserts


def test_retained_preview_uses_the_same_direct_canonical_route():

    preview = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=LodPageCache(max_bytes=1 << 20))
    rendered = session.rendered_tiles[0]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)

    key = admit_retained_preview_level(
        preview, rendered, semantic_source_id=semantic_id, preview_level=3
    )
    assert key is not None
    assert key.level_xy == (3, 3)
    page = preview.resolved_pages(key.plans)[0]
    assert page.values.shape == (TILE // 8, TILE // 8)
    expected = np.asarray(rendered.image).reshape(TILE // 8, 8, TILE // 8, 8).mean(axis=(1, 3))
    assert np.allclose(page.values, expected, atol=1e-4)

    # A second cache takes the same direct route; cache history cannot select
    # different numeric values for the canonical identity.
    preview2 = LodPageCache(max_bytes=1 << 20)
    key2 = admit_retained_preview_level(
        preview2,
        rendered,
        semantic_source_id=semantic_id,
        preview_level=3,
    )
    assert key2 is not None
    assert np.allclose(preview2.resolved_pages(key2.plans)[0].values, expected, atol=1e-4)

    # Singleflight: second admission is a no-op.
    assert not admit_retained_preview_level(
        preview, rendered, semantic_source_id=semantic_id, preview_level=3
    )


def test_retained_preview_protects_its_claim_until_checked_admission(monkeypatch):
    preview = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=preview, count=1)
    rendered = session.rendered_tiles[0]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    original_materialize = render_lod.materialize_lod_page
    cancellation_results = []

    def materialize_while_cancelled(*args, plan, **kwargs):
        owner = preview.claimed_by(plan.key)
        assert owner is not None
        cancellation_results.append(preview.release_owner_claims(owner))
        return original_materialize(*args, plan=plan, **kwargs)

    monkeypatch.setattr(render_lod, "materialize_lod_page", materialize_while_cancelled)

    key = admit_retained_preview_level(
        preview,
        rendered,
        semantic_source_id=semantic_id,
        preview_level=3,
    )

    assert key is not None
    assert cancellation_results
    assert all(result == () for result in cancellation_results)
    assert preview.exact_pages(key.plans) is not None
    assert preview.pending_count == 0


def test_retained_preview_wrong_page_is_loud_and_releases_its_claim(monkeypatch):
    preview = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=preview, count=1)
    rendered = session.rendered_tiles[0]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    wrong_key = page_set_key_for_rendered(
        rendered,
        demand=session.lod_policy_decision.demand,
        level=3,
        semantic_source_id=semantic_id,
    )
    wrong_page = materialize_lod_page(
        np.asarray(rendered.image),
        source_origin_yx=(0, 0),
        plan=wrong_key.plans[0],
    )
    monkeypatch.setattr(render_lod, "materialize_lod_page", lambda *args, **kwargs: wrong_page)

    with pytest.raises(ValueError, match="wrong key"):
        admit_retained_preview_level(
            preview,
            rendered,
            semantic_source_id=semantic_id,
            preview_level=2,
        )

    assert len(preview) == 0
    assert preview.pending_count == 0


def test_ineligible_retained_preview_never_leaves_an_owner_claim():
    preview = LodPageCache(max_bytes=8)
    session = _session(pyramid=preview, count=1)
    rendered = session.rendered_tiles[0]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = page_set_key_for_rendered(
        rendered,
        demand=session.lod_policy_decision.demand,
        level=2,
        semantic_source_id=semantic_id,
    )

    assert (
        admit_retained_preview_level(
            preview,
            rendered,
            semantic_source_id=semantic_id,
            preview_level=2,
        )
        is None
    )
    assert preview.plan_set_ineligible(key.plans)
    assert preview.pending_count == 0
    assert len(preview) == 0


def test_floor_presents_from_pinned_preview_when_main_pyramid_lost_the_level():
    """Scroll-back contract: main-cache churn can never blank a tile that was
    ever computed — the pinned preview level floors it."""

    main = LodPageCache(max_bytes=1 << 20)
    preview = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=main, count=2)
    session.lod_page_cache = preview
    session.lod_preview_level = 3

    rendered = session.rendered_tiles[1]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = admit_retained_preview_level(
        preview, rendered, semantic_source_id=semantic_id, preview_level=3
    )
    assert key is not None
    _claim_preview_resident(session, 1, key)
    # Tile 1 loses its rendered result and has nothing in the MAIN pyramid.
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)

    _state, delta = session.build_tile_presentation({})
    payload = delta.upserts.get(1) or session.display_tile_payloads.get(1)
    assert payload is not None, "preview level must floor the tile"
    assert payload.quality == "preview"
    assert payload.lod.level == 2
    assert payload.page_backing.resolved_page_set.coarsest_actual_level == 3
    assert payload.texture_data.shape[:2] == (TILE // 8, TILE // 8)


def test_rgb_retained_preview_is_rejected_instead_of_becoming_false_canonical_data():

    main = LodPageCache(max_bytes=1 << 20)
    preview = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=main, count=1)
    session.lod_page_cache = preview
    session.lod_preview_level = 2
    session.rgb = True

    tile = session.plan.tiles[0]
    rgb = np.zeros((TILE, TILE, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(TILE, dtype=np.uint8)[:, None]
    histogram = np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE)
    rendered = RenderedTile(
        tile=tile,
        image=rgb,
        histogram_data=histogram,
        eval_ms=0.0,
        slab_shape=rgb.shape,
        slab_nbytes=rgb.nbytes,
    )
    semantic_id = session.tile_semantic_source_id(tile.source_index)
    with pytest.raises(ValueError, match="scalar or complex"):
        admit_retained_preview_level(
            preview,
            rendered,
            semantic_source_id=semantic_id,
            preview_level=2,
            shader_display=False,
        )
    assert len(preview) == 0


def test_rgb_preview_without_display_histogram_is_not_admitted():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=1)
    session.rgb = True

    tile = session.plan.tiles[0]
    rgb = np.zeros((TILE, TILE, 3), dtype=np.uint8)
    rendered = RenderedTile(
        tile=tile,
        image=rgb,
        histogram_data=None,
        eval_ms=0.0,
        slab_shape=rgb.shape,
        slab_nbytes=rgb.nbytes,
    )
    semantic_id = session.tile_semantic_source_id(tile.source_index)

    with pytest.raises(ValueError, match="scalar or complex"):
        admit_retained_preview_level(
            pyramid,
            rendered,
            semantic_source_id=semantic_id,
            preview_level=2,
            shader_display=False,
        )
    assert len(pyramid) == 0


def test_preview_payload_at_acceptable_level_still_refines_to_exact():
    """Regression (screenshot: blocky tiles among exact neighbors): a
    preview payload whose level falls inside acceptable_levels looked
    converged to refresh and never refined, though the rendered result
    was available for a cheap exact rebuild."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[1]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = page_set_key_for_rendered(
        rendered, demand=demand, level=2, semantic_source_id=semantic_id
    )
    _admit_page_set(pyramid, key, np.asarray(rendered.image))

    # Floor the tile while unrendered, then the rendered result returns
    # (e.g. session reseed) without any dirty mark.
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)
    _state, delta = session.build_tile_presentation({})
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    assert session.display_tile_payloads[1].quality == "preview"
    image = np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE)
    session.rendered_tiles[1] = RenderedTile(
        tile=session.plan.tiles[1],
        image=image,
        histogram_data=image,
        eval_ms=0.0,
        slab_shape=image.shape,
        slab_nbytes=image.nbytes,
    )

    # Camera refresh must dirty the preview tile even though its level (2)
    # is inside acceptable_levels, and the next build must go exact.
    session.mark_ladder_swaps_for_viewport()
    assert 1 in session.dirty_payloads
    _state, _delta = session.build_tile_presentation({})
    assert session.display_tile_payloads[1].quality == "exact"


def test_shared_preview_floor_is_presented_before_exact_refinement():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = page_set_key_for_rendered(
            rendered, demand=demand, level=2, semantic_source_id=semantic_id
        )
        _admit_page_set(pyramid, key, np.asarray(rendered.image))
        _claim_preview_resident(session, rendered.tile.montage_index, key)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=2)

    assert set(preview_delta.upserts) == {0, 1}
    assert {payload.quality for payload in preview_delta.upserts.values()} == {"preview"}
    assert {payload.lod.level for payload in preview_delta.upserts.values()} == {2}

    _acknowledge(session, preview_delta)
    session.mark_presented(tuple(preview_delta.upserts))

    preview_refinements = session.unrefined_preview_tiles()
    assert set(preview_refinements) == set()
    assert set(session.dirty_payloads) == {0, 1}
    session.mark_preview_refinements_dirty(preview_refinements)
    assert set(session.dirty_payloads) == {0, 1}

    _state, exact_delta = session.build_tile_presentation({}, max_upserts=2)

    assert set(exact_delta.upserts) == {0, 1}
    assert {payload.quality for payload in exact_delta.upserts.values()} == {"exact"}
    assert {payload.lod.level for payload in exact_delta.upserts.values()} == {2}


def test_shared_preview_worker_rows_admit_as_checked_canonical_pages():
    """The real worker result must cross the GUI seam as checked pages."""

    data = np.arange(TILE * TILE * 2, dtype=np.float32).reshape(TILE, TILE, 2)
    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_montage_axis(2, columns=2, indices=(0, 1), text=":")
    )
    plan = make_montage_plan(
        state,
        axis=2,
        indices=(0, 1),
        tile_shape=(TILE, TILE),
        columns=2,
        viewport_shape=VIEWPORT,
    )
    cache = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=cache, count=2)
    session.document = ArrayDocument(data)
    session.view_state = state
    session.plan = plan
    session.montage_axis = 2
    session.visible_tiles = plan.tiles
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    session.lod_policy_mode = LOD_POLICY_NATIVE_ONLY
    session.lod_preview_level = 1
    demand = LodDemand(
        desired_level=0,
        desired_factor=1,
        desired_factor_xy=(1, 1),
        acceptable_levels=(0, 1, 2),
        source_texels_per_pixel_xy=(1.0, 1.0),
        reason="shared-preview-admission-test",
    )

    rows = render_effects.evaluate_shared_preview(
        session,
        plan.tiles[0],
        plan.tiles,
        demand=demand,
        level=1,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )
    renderer = _RungPrepareRenderer()
    renderer._rendered_tile_for_current_payload = lambda *_args, **_kwargs: None
    renderer._admit_first_pass_level_evidence = lambda *_args, **_kwargs: None
    frame_effects = FramePipelineEffects(renderer, session)
    ensure_calls = []
    original_ensure_floor_payloads = session._ensure_floor_payloads

    def ensure_floor_payloads(tile_numbers, *, max_count=None):
        ensure_calls.append(tuple(int(tile) for tile in tile_numbers))
        return original_ensure_floor_payloads(tile_numbers, max_count=max_count)

    session._ensure_floor_payloads = ensure_floor_payloads

    assert rows
    assert all(all(isinstance(page, MaterializedLodPage) for page in row[2]) for row in rows)
    assert frame_effects._admit_reduced_display_payload(
        None,
        int(rows[0][0]),
        rows,
        quality="preview",
    )
    assert ensure_calls == [(0, 1)]

    for tile_number, key, pages, *_rest in rows:
        assert cache.exact_pages(key.plans) is not None
        assert tuple(page.key for page in pages) == tuple(plan.key for plan in key.plans)
        record = session.lifecycle.peek(int(tile_number))
        assert record is not None
        assert record.levels[key].phase.name == "RESIDENT"

    # A later direct route at the same physical level has a different value
    # identity and outranks the non-semantic preview instead of being
    # suppressed by its residency.
    tile = plan.tiles[0]
    source = np.ascontiguousarray(data[..., int(tile.source_index)])
    rendered = RenderedTile(
        tile=tile,
        image=source,
        histogram_data=source,
        eval_ms=0.0,
        slab_shape=source.shape,
        slab_nbytes=source.nbytes,
    )
    exact_key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=1,
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
    )
    exact_pages = _materialized_page_set(exact_key, source)
    assert exact_key.page_keys != rows[0][1].page_keys
    assert session.admit_preview_plane(
        int(tile.montage_index),
        exact_key,
        exact_pages,
        quality="exact",
    )
    session.lod_policy_mode = LOD_POLICY_RESIDENT
    best = session._best_floor_key(
        int(tile.source_index),
        tile_number=int(tile.montage_index),
    )
    assert best is not None
    assert best[0] == exact_key


def test_presented_preview_keeps_active_exact_work_loading():
    """A first-pixel preview ack must not clear the exact work owed state."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[0]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = page_set_key_for_rendered(
        rendered, demand=demand, level=2, semantic_source_id=semantic_id
    )
    _admit_page_set(pyramid, key, np.asarray(rendered.image))
    _claim_preview_resident(session, 0, key)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=1)
    assert preview_delta.upserts[0].quality == "preview"

    session.rendered_tiles.clear()
    session.lifecycle.evaluation_started(0)
    session.lifecycle.evaluation_claimed(0, object())
    session.loading_tiles.add(0)
    _acknowledge(session, preview_delta)
    session.mark_presented((0,))

    assert 0 in session.lifecycle.presented_tiles
    assert 0 in session.lifecycle.evaluating_tiles
    assert 0 in session.active_tile_requests
    assert 0 in session.loading_tiles
    assert not session.is_complete()
    assert session.ensure_tile_states()[0].name == "LOADING"


def test_preview_floor_does_not_complete_full_refinement():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = page_set_key_for_rendered(
            rendered, demand=demand, level=2, semantic_source_id=semantic_id
        )
        _admit_page_set(pyramid, key, np.asarray(rendered.image))
        _claim_preview_resident(session, rendered.tile.montage_index, key)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=2)
    _acknowledge(session, preview_delta)
    session.mark_presented(tuple(preview_delta.upserts))

    assert session.visible_first_pixels_presented()
    assert not session.visible_plan_complete()
    assert not session.is_complete()

    session.mark_preview_refinements_dirty(session.unrefined_preview_tiles())
    _state, exact_delta = session.build_tile_presentation({}, max_upserts=2)
    _acknowledge(session, exact_delta)
    session.mark_presented(tuple(exact_delta.upserts))

    assert session.is_complete()


def test_stage_backed_rung_dep_uses_kernel_stage_task_key():
    session = _session(count=1)
    stage_key = ("stage", "in-flight")
    session.stage_fan_in.tile_stage_keys[0] = stage_key
    session.stage_fan_in.active_requests.add(stage_key)
    effects = FramePipelineEffects(
        _RungPrepareRenderer(kernel=_StageProducerKernel(stage_key)), session
    )
    step = _exact_step(0)

    assert effects.rung_deps(_pipeline_intent_for(session), step) == (stage_key,)


def test_stage_backed_rung_has_no_dependency_without_live_stage_producer():
    session = _session(count=1)
    stage_key = ("stage", "orphan")
    session.stage_fan_in.tile_stage_keys[0] = stage_key
    session.stage_fan_in.active_requests.add(stage_key)
    effects = FramePipelineEffects(_RungPrepareRenderer(), session)
    step = _exact_step(0)

    assert effects.rung_deps(_pipeline_intent_for(session), step) == ()


def test_stage_backed_rung_admission_records_live_stage_producer():
    session = _session(count=1)
    session.rendered_tiles.clear()
    stage_key = ("stage", "in-flight")
    session.stage_fan_in.tile_stage_keys[0] = stage_key
    renderer = _RungPrepareRenderer(kernel=_StageProducerKernel(stage_key))
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _exact_step(0)

    effects.rung_admitted(intent, step, ("task", 0))

    row = session.lifecycle.row(0)
    assert row.task_claim.task_key == ("task", 0)
    assert row.stage_key == stage_key
    assert row.stage_producer_key == stage_key


def test_exact_rung_enters_running_only_after_kernel_admission():
    session = _session(count=1)
    session.rendered_tiles.clear()
    session.dirty_payloads.pop(0, None)
    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _exact_step(0)

    assert effects.prepare_rung(intent, step)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles

    effects.rung_admitted(intent, step, ("task", 0))

    assert 0 in session.active_tile_requests
    assert 0 in session.loading_tiles
    assert session.lifecycle.row(0).task_claim.task_key == ("task", 0)


def test_prepare_rung_releases_active_claim_when_task_is_not_live():
    session = _session(count=1)
    session.rendered_tiles.clear()
    session.dirty_payloads.pop(0, None)
    renderer = _RungPrepareRenderer(kernel=_StageProducerKernel())
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _exact_step(0)

    effects.rung_admitted(intent, step, ("task", "missing"))

    assert 0 in session.active_tile_requests
    assert session.lifecycle.row(0).task_claim.task_key == ("task", "missing")

    assert effects.prepare_rung(intent, step)

    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.lifecycle.row(0).task_claim is None


def test_preview_drop_does_not_clear_target_task_claim():
    session = _session(count=1)
    session.rendered_tiles.clear()
    session.dirty_payloads.pop(0, None)
    renderer = _RungPrepareRenderer(kernel=_StageProducerKernel(("task", "target")))
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    target_step = _exact_step(0)
    preview_step = RungStep(
        tile_number=0,
        rung=Rung.FLOOR,
        level=4,
        reduce_from_native=False,
        lane=Lane.DISPLAY_PREVIEW,
        priority=Priority.VISIBLE_IMAGE,
        reason="preview",
    )

    effects.rung_admitted(intent, target_step, ("task", "target"))
    claim_identity = effects._preview_claim_identity(intent, session.plan.tiles[0])
    session.lifecycle.preview_claimed(0, int(Rung.FLOOR), 4, claim_identity)

    effects.rung_dropped(intent, preview_step)

    assert not session.lifecycle.preview_claim_matches(0, int(Rung.FLOOR), 4, claim_identity)
    assert session.lifecycle.row(0).task_claim.task_key == ("task", "target")
    assert 0 in session.active_tile_requests
    assert session._test_replan_requested is True


def test_tile_state_snapshot_releases_active_claim_without_live_task():
    session = _session(count=1)
    session.rendered_tiles.clear()
    session.dirty_payloads.pop(0, None)
    renderer = _RungPrepareRenderer(kernel=_StageProducerKernel())
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _exact_step(0)

    effects.rung_admitted(intent, step, ("task", "gone"))
    session.lifecycle.task_released(0, reason="dropped")

    states = effects.tile_states(
        intent, session.lod_policy_decision.demand, _pipeline_scope_for(session)
    )

    assert tuple(state.tile_number for state in states) == (0,)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.lifecycle.evaluating_tiles == frozenset()


def test_tile_state_snapshot_keeps_live_active_claim_out_of_ladder():
    session = _session(count=1)
    session.rendered_tiles.clear()
    session.dirty_payloads.pop(0, None)
    renderer = _RungPrepareRenderer(kernel=_StageProducerKernel(("task", "live")))
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _exact_step(0)

    effects.rung_admitted(intent, step, ("task", "live"))

    states = effects.tile_states(
        intent, session.lod_policy_decision.demand, _pipeline_scope_for(session)
    )

    assert states == ()
    assert 0 in session.active_tile_requests
    assert 0 in session.loading_tiles


def test_tile_state_does_not_treat_orphan_reduced_residency_as_committable():
    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    tile = session.plan.tiles[0]
    semantic_id = session.tile_semantic_source_id(tile.source_index)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    key = page_set_key_for_rendered(
        session.rendered_tiles[0],
        demand=demand,
        level=2,
        semantic_source_id=semantic_id,
    )
    session.lifecycle.level_claimed(0, key, ClaimOwner.PREVIEW, request=("resident", key))
    session.lifecycle.level_resident(0, key)
    session.rendered_tiles.clear()
    session.display_tile_payloads.clear()
    session.tile_presentation_state = TilePresentationState()
    session.dirty_payloads.clear()
    effects = FramePipelineEffects(_RungPrepareRenderer(), session)

    states = effects.tile_states(
        _pipeline_intent_for(session),
        session.lod_policy_decision.demand,
        _pipeline_scope_for(session),
    )

    assert len(states) == 1
    assert states[0].resident_levels == ()


def test_preview_floor_commit_activates_every_planned_preview_tile_before_exact():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=4)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = page_set_key_for_rendered(
            rendered, demand=demand, level=2, semantic_source_id=semantic_id
        )
        _admit_page_set(pyramid, key, np.asarray(rendered.image))
        _claim_preview_resident(session, rendered.tile.montage_index, key)

    # Exact results already exist for the priority-center tiles.  The preview
    # floor still owns the first fill: the commit must activate the whole
    # planned preview set, not flash exact islands one tile at a time.
    for tile_number in (0, 3):
        del session.rendered_tiles[tile_number]
        session.dirty_payloads.pop(tile_number, None)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=4)

    assert set(preview_delta.upserts) == {0, 1, 2, 3}
    assert set(preview_delta.active_tiles) == {0, 1, 2, 3}
    assert {payload.quality for payload in preview_delta.upserts.values()} == {"preview"}


def test_cold_preview_floor_uploads_obey_item_cap():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=4)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = page_set_key_for_rendered(
            rendered, demand=demand, level=2, semantic_source_id=semantic_id
        )
        _admit_page_set(pyramid, key, np.asarray(rendered.image))
        _claim_preview_resident(session, rendered.tile.montage_index, key)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=1)

    admitted = set(preview_delta.upserts)
    assert len(admitted) == 1
    assert set(preview_delta.active_tiles) == {0, 1, 2, 3}
    admitted_tile = next(iter(admitted))
    assert preview_delta.upserts[admitted_tile].quality == "preview"
    assert set(session.pending_payload_upserts) == {0, 1, 2, 3}


def test_coverage_pass_blocks_targets_until_required_preview_coverage_closes():
    """ADR 0059: no target work executes before required-set preview coverage.

    The retired shared scheduler's acknowledge-all task does not return.
    Lifecycle first-pixel truth closes the ordinary scheduling phase, and the
    ladder keeps every target rung on the gated preparation lane until then.
    """

    from arrayscope.render.ladder import LadderPolicy, LodLadder, Rung, TileLodState
    from arrayscope.render.progressive_scheduling import (
        SchedulingPhase,
        SchedulingVerdict,
    )

    ladder = LodLadder(
        policy=LadderPolicy(
            mode="resident",
            floor_level=4,
            reduced_input_available=True,
        )
    )
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))

    coverage = SchedulingVerdict(1, SchedulingPhase.COVERAGE, (0, 1))
    refine = SchedulingVerdict(1, SchedulingPhase.REFINE, (0, 1))
    covered = TileLodState(
        tile_number=0,
        presented_level=4,
        presented_quality="preview",
    )
    covered_steps = ladder.plan_tile(covered, demand, coverage)
    assert not any(step.rung is Rung.DESIRED for step in covered_steps), (
        "one tile's preview acknowledgement must not admit its target while "
        "another required tile is blank"
    )

    blank_with_preview_path = TileLodState(
        tile_number=1,
        allow_preview=True,
    )
    steps = ladder.plan_tile(blank_with_preview_path, demand, coverage)
    assert any(step.rung is Rung.FLOOR for step in steps)
    assert not any(step.rung is Rung.DESIRED for step in steps), (
        "DESIRED must wait for this tile's own coarse acknowledgement"
    )

    closed = TileLodState(
        tile_number=2,
        presented_level=4,
        presented_quality="preview",
    )
    assert any(step.rung is Rung.DESIRED for step in ladder.plan_tile(closed, demand, refine)), (
        "refinement must plan immediately once the pass closes"
    )


def test_atomic_successor_uses_native_for_tiles_without_a_resolvable_floor():
    """Ring 0 gate for field stall 57: no atomic wait without a floor owner."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    native_range = ((0.0, float(2 * TILE)), (0.0, float(TILE)))
    session = _session(pyramid=pyramid, count=4, view_range=native_range)
    session.lod_preview_level = 4
    session.atomic_successor_pending = True

    # Only one successor tile still has resolvable page coverage.  The other
    # three already own current native RenderedTiles, so the ladder schedules
    # no producer for them (the exact result already exists).
    rendered = session.rendered_tiles[0]
    demand = session.lod_policy_decision.demand
    assert demand.desired_level == 0
    key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=4,
        semantic_source_id=session.tile_semantic_source_id(0),
    )
    _admit_page_set(pyramid, key, np.asarray(rendered.image))
    _claim_preview_resident(session, 0, key)

    # The fast atomic builder is lookup-only and correctly declines the
    # incomplete cohort.  General presentation owns the honest fallback: keep
    # tile 0's coarse floor and wrap current native pixels for tiles 1..3.
    assert session.build_atomic_successor_presentation() is None
    _state, delta = session.build_tile_presentation({})

    assert set(delta.active_tiles) == {0, 1, 2, 3}
    assert set(delta.upserts) == {0, 1, 2, 3}
    assert delta.upserts[0].quality == "preview"
    assert delta.upserts[0].lod.level == 4
    assert {
        tile: (payload.quality, payload.lod.level)
        for tile, payload in delta.upserts.items()
        if tile != 0
    } == {1: ("exact", 0), 2: ("exact", 0), 3: ("exact", 0)}


def test_atomic_successor_accepts_ordinary_typed_payload_identities():
    """A production display source id must not hide its canonical tile identity."""

    session = _session(count=2)
    source_ids = {index: ("display-tile", index) for index in range(2)}
    state, delta = session.build_tile_presentation(source_ids)
    session.acknowledge_tile_presentation(
        delta,
        TileCommitReport(
            presented_tiles=state.active_payloads(delta),
            committed_upserts=frozenset(delta.upserts),
        ),
    )
    session.mark_presented(state.active_payloads(delta))
    session.atomic_successor_pending = True

    atomic = session.build_atomic_successor_presentation()

    assert atomic is not None, session._atomic_fast_reject_reason
    _state, successor = atomic
    assert tuple(successor.upserts) == (0, 1)


def test_acknowledged_preview_with_exact_result_rearms_exact_refinement():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = page_set_key_for_rendered(
            rendered, demand=demand, level=2, semantic_source_id=semantic_id
        )
        _admit_page_set(pyramid, key, np.asarray(rendered.image))
        _claim_preview_resident(session, rendered.tile.montage_index, key)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=2)
    _acknowledge(session, preview_delta)
    session.mark_presented(tuple(preview_delta.upserts))

    assert {payload.quality for payload in preview_delta.upserts.values()} == {"preview"}
    assert set(session.dirty_payloads) == {0, 1}
    _state, exact_delta = session.build_tile_presentation({}, max_upserts=2)
    assert {payload.quality for payload in exact_delta.upserts.values()} == {"exact"}


def test_offscreen_floor_claim_cannot_block_required_exact_atomic_successor():
    """Reproduce session 39: required L4 pixels must advance in REFINE.

    The margin-expanded session can still own an offscreen preview producer
    after the physical frame-plan scope has completed coverage.  That shell
    work must not keep the atomic presentation builder in floor-first mode and
    suppress the already-rendered exact wrappers for the required center.
    """

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=4)
    session.frame_plan = SimpleNamespace(active_region_ids=(1, 2))
    session.sync_lifecycle_scope()
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))

    for tile_number in (1, 2):
        rendered = session.rendered_tiles[tile_number]
        key = page_set_key_for_rendered(
            rendered,
            demand=demand,
            level=2,
            semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
        )
        _admit_page_set(pyramid, key, np.asarray(rendered.image))
        _claim_preview_resident(session, tile_number, key)

    # Tile 0 models a margin-shell floor producer that is still live but is
    # not part of FramePlan.active_region_ids.  It has no ready payload yet.
    shell = session.rendered_tiles.pop(0)
    session.rendered_tiles.pop(3)
    session.dirty_payloads.pop(0, None)
    session.dirty_payloads.pop(3, None)
    shell_key = page_set_key_for_rendered(
        shell,
        demand=demand,
        level=2,
        semantic_source_id=session.tile_semantic_source_id(shell.tile.source_index),
    )
    session.lifecycle.level_claimed(
        0,
        shell_key,
        ClaimOwner.PREVIEW,
        request=("offscreen-preview", shell_key),
    )
    session.lifecycle.level_materializing(0, shell_key)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=2)
    assert set(preview_delta.upserts) == {1, 2}
    _acknowledge(session, preview_delta)
    session.mark_presented(tuple(preview_delta.upserts))
    assert session.scheduling_policy.observe(session.lifecycle)
    assert not session.scheduling_policy.verdict.coverage_open

    session.atomic_successor_pending = True
    session.mark_preview_refinements_dirty((1, 2))
    _state, exact_delta = session.build_tile_presentation({}, max_upserts=2)

    assert set(exact_delta.upserts) == {1, 2}
    assert {payload.quality for payload in exact_delta.upserts.values()} == {"exact"}


def test_backend_confirmed_preview_does_not_settle_when_exact_exists():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = page_set_key_for_rendered(
            rendered, demand=demand, level=2, semantic_source_id=semantic_id
        )
        _admit_page_set(pyramid, key, np.asarray(rendered.image))
        _claim_preview_resident(session, rendered.tile.montage_index, key)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=2)
    _acknowledge(session, preview_delta)
    session.mark_presented(tuple(preview_delta.upserts))
    preview_identities = {
        int(tile): payload.source_id for tile, payload in preview_delta.upserts.items()
    }
    session.lifecycle.backend_presented_snapshot(preview_identities)

    assert {payload.quality for payload in preview_delta.upserts.values()} == {"preview"}

    _state, exact_delta = session.build_tile_presentation({}, max_upserts=2)

    assert set(exact_delta.upserts) == {0, 1}
    assert {payload.quality for payload in exact_delta.upserts.values()} == {"exact"}
    assert all(
        exact_delta.upserts[int(tile)].source_id != preview_identities[int(tile)]
        for tile in exact_delta.upserts
    )


def test_wgpu_shader_floor_uses_canonical_complex_pages():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    session.shader_display = True
    tile = session.plan.tiles[0]
    complex_values = np.ones((TILE, TILE), dtype=np.complex64)
    rendered = replace(
        session.rendered_tiles[0],
        image=complex_values,
        semantic_data=complex_values,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
    )
    session.output_dtype = np.dtype(np.complex64)
    semantic_id = session.tile_semantic_source_id(tile.source_index)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    complex_key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=semantic_id,
    )
    _admit_page_set(pyramid, complex_key, complex_values)

    best = render_lod.best_floor_key(session, int(tile.source_index), tile_number=0)

    assert best is not None
    assert best[0] == complex_key


def test_floor_prefers_resident_l2_over_l4_for_coarser_l6_demand():
    """Unrendered floor selection obeys the same no-demotion rank."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    tile = session.plan.tiles[0]
    rendered = session.rendered_tiles[0]
    demand = select_lod_demand(
        ((0.0, 8192.0), (0.0, 4096.0)),
        VIEWPORT,
        (TILE, TILE),
    )
    assert demand.desired_level == 6
    session.lod_policy_decision = replace(session.lod_policy_decision, demand=demand)
    semantic_id = session.tile_semantic_source_id(tile.source_index)
    keys = []
    for level in (2, 4):
        key = page_set_key_for_rendered(
            rendered,
            demand=demand,
            level=level,
            semantic_source_id=semantic_id,
        )
        _admit_page_set(pyramid, key, np.asarray(rendered.image))
        _claim_preview_resident(session, tile.montage_index, key)
        keys.append(key)

    best = render_lod.best_floor_key(
        session,
        int(tile.source_index),
        tile_number=int(tile.montage_index),
    )

    assert best is not None
    assert best[0] == keys[0]
    assert best[1] == 2


def test_best_floor_key_memo_bounds_scans_per_residency_revision(monkeypatch):
    """Repeated retarget consumers scan each tile once per residency epoch."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=8)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    session.lod_policy_decision = replace(session.lod_policy_decision, demand=demand)
    for tile_number, rendered in tuple(session.rendered_tiles.items()):
        key = page_set_key_for_rendered(
            rendered,
            demand=demand,
            level=2,
            semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
        )
        _admit_page_set(pyramid, key, np.asarray(rendered.image))
        _claim_preview_resident(session, tile_number, key)

    resolutions = 0
    original = render_lod._page_set_resolution

    def counted_resolution(*args, **kwargs):
        nonlocal resolutions
        resolutions += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(render_lod, "_page_set_resolution", counted_resolution)
    first = tuple(
        render_lod.best_floor_key(
            session,
            int(tile.source_index),
            tile_number=int(tile.montage_index),
        )
        for tile in session.plan.tiles
    )
    first_pass_resolutions = resolutions
    second = tuple(
        render_lod.best_floor_key(
            session,
            int(tile.source_index),
            tile_number=int(tile.montage_index),
        )
        for tile in session.plan.tiles
    )

    assert first == second
    assert first_pass_resolutions > 0
    assert resolutions == first_pass_resolutions


def test_best_floor_key_memo_rejects_stale_store_after_epoch_race(monkeypatch):
    """An older compute cannot overwrite a memo refreshed at a newer epoch."""

    session = _session(pyramid=LodPageCache(max_bytes=1 << 20), count=1)
    stale = object()
    fresh = object()
    computes = 0

    def racing_compute(*args, **kwargs):
        nonlocal computes
        computes += 1
        if computes == 1:
            session.lifecycle.level_claimed(
                0,
                ("interleaved-floor",),
                ClaimOwner.CHAIN,
            )
            assert render_lod.best_floor_key(session, 0, tile_number=0) is fresh
            return stale
        return fresh

    monkeypatch.setattr(render_lod, "_compute_best_floor_key", racing_compute)

    assert render_lod.best_floor_key(session, 0, tile_number=0) is stale
    assert render_lod.best_floor_key(session, 0, tile_number=0) is fresh
    assert computes == 2


def test_floor_ranks_better_physical_fallback_ahead_of_coarser_exact_target():
    """Exact requested identity cannot outrank the pixels actually sampled."""

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    tile = session.plan.tiles[0]
    rendered = session.rendered_tiles[0]
    demand = LodDemand(
        desired_level=6,
        desired_factor=64,
        desired_factor_xy=(64, 64),
        # Keep L7 out of the exact target scan: its resident pages are the
        # physical ancestor for desired L6, not a requested rung identity.
        acceptable_levels=(8,),
        source_texels_per_pixel_xy=(64.0, 64.0),
        reason="physical fallback rank regression",
    )
    session.lod_policy_decision = replace(session.lod_policy_decision, demand=demand)
    semantic_id = session.tile_semantic_source_id(tile.source_index)
    keys = {}
    for level in (7, 8):
        key = page_set_key_for_rendered(
            rendered,
            demand=demand,
            level=level,
            semantic_source_id=semantic_id,
        )
        _admit_page_set(pyramid, key, np.asarray(rendered.image))
        keys[level] = key
    _claim_preview_resident(session, tile.montage_index, keys[8])

    best = render_lod.best_floor_key(
        session,
        int(tile.source_index),
        tile_number=int(tile.montage_index),
    )

    assert best is not None
    assert best[0].level == 6
    assert best[1] == 7


def test_preview_admission_rejects_unchecked_whole_plane_values():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    session.shader_display = True
    tile = session.plan.tiles[0]
    rendered = session.rendered_tiles[0]
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
    )

    with pytest.raises(TypeError, match="checked canonical"):
        session.admit_preview_plane(0, key, np.zeros((TILE // 4, TILE // 4)))
    assert not render_lod._page_set_complete(pyramid, key)


def test_preview_page_admission_protects_claims_until_the_set_is_complete(monkeypatch):
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    rendered = session.rendered_tiles[0]
    key = page_set_key_for_rendered(
        rendered,
        demand=select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE)),
        level=2,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    pages = _materialized_page_set(key, np.asarray(rendered.image))
    original_admit = pyramid.admit_as
    cancellation_results = []

    def admit_while_cancelled(page_key, page, *, owner):
        cancellation_results.append(pyramid.release_owner_claims(owner))
        return original_admit(page_key, page, owner=owner)

    monkeypatch.setattr(pyramid, "admit_as", admit_while_cancelled)

    assert session.admit_preview_plane(0, key, pages)
    assert cancellation_results
    assert all(result == () for result in cancellation_results)
    assert pyramid.exact_pages(key.plans) is not None
    assert pyramid.pending_count == 0


def test_preview_page_admission_failure_is_loud_and_releases_all_claims(monkeypatch):
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    rendered = session.rendered_tiles[0]
    key = page_set_key_for_rendered(
        rendered,
        demand=select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE)),
        level=2,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    pages = _materialized_page_set(key, np.asarray(rendered.image))

    def reject_checked_page(*args, **kwargs):
        raise ValueError("forced checked page admission failure")

    monkeypatch.setattr(pyramid, "admit_as", reject_checked_page)

    with pytest.raises(ValueError, match="forced checked page admission failure"):
        session.admit_preview_plane(0, key, pages)

    assert pyramid.pending_count == 0
    retry_owner = ("preview-admission-retry",)
    assert pyramid.claim_plans(key.plans, retry_owner) == key.plans
    assert pyramid.release_owner_claims(retry_owner) == key.page_keys


def test_ineligible_preview_page_set_is_declined_without_claim_or_lifecycle_state():
    pyramid = LodPageCache(max_bytes=8)
    session = _session(pyramid=pyramid, count=1)
    rendered = session.rendered_tiles[0]
    key = page_set_key_for_rendered(
        rendered,
        demand=select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE)),
        level=2,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    pages = _materialized_page_set(key, np.asarray(rendered.image))

    assert not session.admit_preview_plane(0, key, pages)
    assert pyramid.plan_set_ineligible(key.plans)
    assert pyramid.pending_count == 0
    assert session.lifecycle.dangling_claims() == ()


def test_acknowledged_preview_floor_stays_active_until_exact_refinement():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=4)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = page_set_key_for_rendered(
            rendered, demand=demand, level=2, semantic_source_id=semantic_id
        )
        _admit_page_set(pyramid, key, np.asarray(rendered.image))
        _claim_preview_resident(session, rendered.tile.montage_index, key)
    for tile_number in (0, 3):
        del session.rendered_tiles[tile_number]
        session.dirty_payloads.pop(tile_number, None)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=4)
    _acknowledge(session, preview_delta)
    session.mark_presented(tuple(preview_delta.upserts))
    session.mark_preview_refinements_dirty((1, 2))

    _state, exact_delta = session.build_tile_presentation({}, max_upserts=2)

    assert set(exact_delta.upserts) == {1, 2}
    assert {payload.quality for payload in exact_delta.upserts.values()} == {"exact"}
    assert set(exact_delta.active_tiles) == {0, 1, 2, 3}


def test_preview_floor_claim_release_unblocks_exact_payload():
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    rendered = session.rendered_tiles[0]
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = page_set_key_for_rendered(
        rendered, demand=demand, level=2, semantic_source_id=semantic_id
    )
    session.lifecycle.level_claimed(0, key, ClaimOwner.PREVIEW, request=("test-preview", key))
    session.lifecycle.level_materializing(0, key)

    _state, blocked_delta = session.build_tile_presentation({}, max_upserts=1)

    assert blocked_delta.upserts == {}
    session.release_preview_claim(0, key)
    assert session.lifecycle.dangling_claims() == ()
    assert pyramid.pending_count == 0

    _state, exact_delta = session.build_tile_presentation({}, max_upserts=1)

    assert exact_delta.upserts[0].quality == "exact"


def test_replaced_session_releases_owner_scoped_page_claims():
    from arrayscope.render.lod import release_session_claims

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    session.lod_preview_level = 4
    rendered = session.rendered_tiles[0]
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = page_set_key_for_rendered(
        rendered, demand=demand, level=4, semantic_source_id=semantic_id
    )
    request = render_lod.plan_materialization(session, rendered, demand=demand, level=4, key=key)
    session.pending_rung_materializations.append(request)
    assert pyramid.pending_count == len(key.plans)

    assert release_session_claims(session) == len(key.plans)

    assert session.lifecycle.dangling_claims() == ()
    assert pyramid.pending_count == 0


def test_preview_floor_target_prefers_preview_cache_over_requested_level():
    pyramid = LodPageCache(max_bytes=1 << 24)
    preview = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    session.lod_page_cache = preview
    session.lod_preview_level = 4
    rendered = session.rendered_tiles[0]
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    requested_key = page_set_key_for_rendered(
        rendered, demand=demand, level=2, semantic_source_id=semantic_id
    )
    preview_key = page_set_key_for_rendered(
        rendered, demand=demand, level=4, semantic_source_id=semantic_id
    )
    _admit_page_set(pyramid, requested_key, np.asarray(rendered.image))
    _admit_page_set(preview, preview_key, np.asarray(rendered.image))
    _claim_preview_resident(session, 0, preview_key)

    del session.rendered_tiles[0]
    session.dirty_payloads.clear()
    _state, delta = session.build_tile_presentation({}, max_upserts=1)

    assert delta.upserts[0].quality == "preview"
    assert delta.upserts[0].lod.level == 2
    assert delta.upserts[0].page_backing.resolved_page_set.coarsest_actual_level == 4
    assert delta.upserts[0].texture_data.shape[:2] == (TILE // 16, TILE // 16)


def test_lod_debug_pass_marker_mirrors_final_payload_only(monkeypatch):
    monkeypatch.setenv("ARRAYSCOPE_LOD_DEBUG_PASS_MARKER", "final-mirror-x")
    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    image = np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE)
    session.rendered_tiles[0] = RenderedTile(
        tile=session.plan.tiles[0],
        image=image,
        histogram_data=image,
        eval_ms=0.0,
        slab_shape=image.shape,
        slab_nbytes=image.nbytes,
    )
    session.dirty_payloads[0] = None
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    semantic_id = session.tile_semantic_source_id(0)
    key = page_set_key_for_rendered(
        session.rendered_tiles[0], demand=demand, level=2, semantic_source_id=semantic_id
    )
    _admit_page_set(pyramid, key, image)
    preview = pyramid.resolved_pages(key.plans)[0].values
    _claim_preview_resident(session, 0, key)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=1)
    _acknowledge(session, preview_delta)
    session.mark_presented(tuple(preview_delta.upserts))
    session.mark_preview_refinements_dirty(session.unrefined_preview_tiles())
    _state, exact_delta = session.build_tile_presentation({}, max_upserts=1)

    assert np.array_equal(preview_delta.upserts[0].texture_data, preview)
    assert np.array_equal(exact_delta.upserts[0].texture_data, np.fliplr(preview))
    assert exact_delta.upserts[0].semantic_data is not None
    assert np.array_equal(exact_delta.upserts[0].semantic_data, image)


def test_replaced_session_releases_undrained_request_claims():
    """A dying session's planned-but-undrained requests must free their claims.

    The pyramid is renderer-shared and its keys are semantic: a claim leaked
    on session replacement blocks the SAME levels when the user scrubs back
    to that slice, wedging the tile at the wrong LOD forever (the stale-LOD
    regression of 2026-07-04).
    """

    from arrayscope.render.lod import release_session_claims

    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    _settle_first_pixels(session)
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    assert len(requests) == 2
    assert pyramid.pending_count > 0

    released = release_session_claims(session)

    assert released == 2
    assert session.pending_rung_materializations == []
    assert pyramid.pending_count == 0
    # The same slice revisited (equal session key) can claim its levels again.
    replacement = _session(pyramid=pyramid)
    _settle_first_pixels(replacement)
    replacement.build_tile_presentation({})
    replacement_requests = list(_plan_rung_materializations(replacement))
    assert len(replacement_requests) == 2


def test_pending_lod_request_view_clear_releases_pyramid_claims():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    _settle_first_pixels(session)
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    assert len(requests) == 2
    assert pyramid.pending_count == 2

    from arrayscope.render.lod import release_session_claims

    assert release_session_claims(session) == 2
    assert pyramid.pending_count == 0
    assert session.lifecycle.dangling_claims() == ()


def test_diagnostics_lod_reason_follows_the_presented_level():
    """The reason text must describe the screen, not the last policy run."""

    from arrayscope.display.lod import (
        LOD_REASON_RESIDENT_MATCH,
        LOD_REASON_RESIDENT_NATIVE_FALLBACK,
    )
    from arrayscope.window.diagnostics_snapshot import _presented_lod_reason

    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    session.build_tile_presentation({})
    decision = session.lod_policy_decision
    assert decision.demand.desired_level == 2

    # Nothing presented at the demanded level yet: native is an explicit fallback.
    assert _presented_lod_reason(decision, (0, 1, (1, 1))) == LOD_REASON_RESIDENT_NATIVE_FALLBACK
    # The screen converged (ingest-presented level 2) without a policy rerun:
    # the stale decision must not keep reporting "materializes".
    assert _presented_lod_reason(decision, (2, 4, (4, 4))) == LOD_REASON_RESIDENT_MATCH


def _shifted_plan(count=2, offset=1):
    tiles = tuple(
        MontageTile(
            montage_index=index,
            source_index=index + offset,
            row=0,
            col=index,
            x0=index * TILE,
            y0=0,
            width=TILE,
            height=TILE,
            view_state=None,
        )
        for index in range(count)
    )
    return MontagePlan(
        axis=0,
        tile_shape=(TILE, TILE),
        grid_shape=(1, count),
        columns=count,
        rows=1,
        gap=0,
        tiles=tiles,
    )


def _retarget(
    session,
    plan,
    new_source_ids,
    cached_tiles=None,
    *,
    semantic_key=("semantic", "retargeted"),
    source_anchoring=None,
):
    return session.retarget_index_window(
        session_id=session.session_id + 1,
        key=("test", "retargeted"),
        semantic_key=semantic_key,
        level_key=("level", "retargeted"),
        render_generation=session.render_generation + 1,
        view_state=None,
        source_anchoring=(
            session.source_anchoring if source_anchoring is None else source_anchoring
        ),
        plan=plan,
        frame_plan=session.frame_plan,
        all_indices=tuple(int(t.source_index) for t in plan.tiles),
        new_source_ids=new_source_ids,
        cached_tiles=dict(cached_tiles or {}),
        visible_tiles=tuple(plan.tiles),
    )


def test_rendered_anchor_uses_worker_view_snapshot_after_live_session_retarget():
    """Rapid crop churn must not pair an old slab with the newest crop origin."""

    old_state = (
        ViewState.from_shape((336, 336, 80))
        .with_image_axes(0, 1)
        .with_axis_range(0, indices=tuple(range(80, 230)), text="80:230")
        .with_axis_range(1, indices=tuple(range(55, 155)), text="55:155")
        .with_montage_axis(2, columns=6, indices=tuple(range(15, 65)), text="15:65")
    )
    new_state = old_state.with_axis_range(
        0, indices=tuple(range(92, 242)), text="92:242"
    ).with_axis_range(1, indices=tuple(range(67, 167)), text="67:167")
    tile = MontageTile(
        montage_index=0,
        source_index=42,
        row=0,
        col=0,
        x0=0,
        y0=0,
        width=100,
        height=150,
        view_state=old_state.tile_state_for_slice(2, 42),
    )
    source = np.zeros((150, 100), dtype=np.float32)
    rendered = RenderedTile(
        tile=tile,
        image=source,
        histogram_data=source,
        eval_ms=0.0,
        slab_shape=source.shape,
        slab_nbytes=source.nbytes,
    )
    session = _session(count=1)
    session.view_state = new_state
    session.montage_axis = 2
    session.canonical_orientation = True
    session.source_anchoring = SourceAnchoring(
        anchored_starts=(92, 67),
        source_starts_yx=(92, 67),
        content_key=("window-free",),
    )

    anchor = session.payload_source_anchor_for_rendered(rendered, source.shape)
    origin = render_lod.source_origin_yx_for_rendered(session, rendered, source)

    assert anchor.source_rect == (80, 230, 55, 155)
    assert origin == (80, 55)
    assert tuple(session.source_anchoring.source_starts_yx) == (92, 67)


def test_retarget_index_window_remaps_hits_misses_and_unchanged():
    """ADR 0051 P2: an index-window scrub reuses the session object.

    Tile 0's source is unchanged (stays presented, no dirty mark), tile 1's
    source changed with a cache hit (mark_materialized seam), and the plan is
    re-keyed without touching backend acknowledgement state.
    """

    session = _session(count=2)
    plan = _shifted_plan(count=2, offset=0)
    # Tile 0 keeps source 0; tile 1 moves to source 9.
    plan_tiles = list(plan.tiles)
    plan_tiles[1] = MontageTile(
        montage_index=1,
        source_index=9,
        row=0,
        col=1,
        x0=TILE,
        y0=0,
        width=TILE,
        height=TILE,
        view_state=None,
    )
    plan = MontagePlan(
        axis=0,
        tile_shape=(TILE, TILE),
        grid_shape=(1, 2),
        columns=2,
        rows=1,
        gap=0,
        tiles=tuple(plan_tiles),
    )
    session.tile_source_ids = {0: ("src", 0), 1: ("src", 1)}
    session.lifecycle.presentation_confirmed((0, 1))
    session.loading_tiles.clear()
    session.dirty_payloads.clear()
    backend_truth = {0: ("shown", 0), 1: ("shown", 1)}
    session.lifecycle.backend_presented_snapshot(backend_truth)

    hit = RenderedTile(
        tile=plan.tiles[1],
        image=np.ones((TILE, TILE), dtype=np.float32),
        histogram_data=None,
        eval_ms=0.0,
        slab_shape=(TILE, TILE),
        slab_nbytes=TILE * TILE * 4,
    )
    stats = _retarget(
        session,
        plan,
        new_source_ids={0: ("src", 0), 1: ("src", 9)},
        cached_tiles={1: hit},
    )

    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["unchanged"] == 1
    assert session.key == ("test", "retargeted")
    assert session.session_id == 2
    # Unchanged tile: still presented, not re-marked dirty.
    assert 0 in session.lifecycle.presented_tiles
    assert 0 not in session.dirty_payloads
    assert 0 in session.rendered_tiles
    # Changed tile went through the ordinary materialization seam.
    assert session.rendered_tiles[1] is hit
    assert 1 in session.dirty_payloads
    assert 1 not in session.lifecycle.presented_tiles
    assert 1 in session.loading_tiles
    # Backend acknowledgement ground truth survives the retarget.
    assert session.lifecycle.backend_presented_identities == backend_truth


def test_paced_followup_rejects_same_slots_with_new_source_mapping():
    """A scroll keeps slot numbers but changes the slices they represent."""

    session = _session(count=2)
    session.build_tile_presentation({0: ("src", 0), 1: ("src", 1)})
    assert set(session.pending_payload_upserts) == {0, 1}
    session._last_planned_tiles = (0, 1)

    session.plan = _shifted_plan(count=2, offset=2)
    session.visible_tiles = tuple(session.plan.tiles)
    session.visible_tile_numbers = frozenset({0, 1})
    session.sync_lifecycle_scope()

    followup = session._paced_pending_presentation_followup(
        cold_deadline_ms=None,
        max_upserts=1,
        max_upsert_bytes=None,
        upsert_cost_fn=None,
        physical_resident_fn=None,
        pace_resident_retargets=False,
    )

    assert followup is None


def test_partial_index_window_reuses_sources_that_move_to_new_slots():
    """A partial montage filters/remaps sources; it must not reload them by slot."""

    session = _session(count=4)
    old_source_ids = {index: ("src", index) for index in range(4)}
    _state, delta = session.build_tile_presentation(old_source_ids)
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    session.tile_source_ids = dict(old_source_ids)
    session.dirty_payloads.clear()
    assert set(session.lifecycle.presented_tiles) == {0, 1, 2, 3}

    partial = _shifted_plan(count=2, offset=2)
    stats = _retarget(
        session,
        partial,
        new_source_ids={0: ("src", 2), 1: ("src", 3)},
        cached_tiles={},
    )

    assert stats["misses"] == 0
    assert stats["remapped"] == 2
    assert set(session.rendered_tiles) == {0, 1}
    assert set(session.display_tile_payloads) == set()
    assert set(session.lifecycle.presented_tiles).isdisjoint({2, 3})
    assert session.pending_removals == set()

    state, delta = session.build_tile_presentation({0: ("src", 2), 1: ("src", 3)})

    assert delta.removals == ()
    assert set(delta.active_tiles) == {0, 1}
    assert set(delta.upserts) == {0, 1}
    assert state.payloads[0].source_index == 2
    assert state.payloads[1].source_index == 3
    assert session.tile_source_ids == {0: ("src", 2), 1: ("src", 3)}


def test_partial_index_window_remaps_lifecycle_payloads_without_rendered_tiles():
    """Shared-transform payloads reuse by source even without RenderedTile rows."""

    session = _session(count=4)
    session.view_state = (
        ViewState.from_shape((4, TILE, TILE))
        .with_image_axes(1, 2)
        .with_montage_axis(0, columns=4, indices=(0, 1, 2, 3), text="0:4")
    )
    session.source_anchoring = SimpleNamespace(
        source_starts_yx=(0, 0),
        content_key=("current-windowless-view",),
    )
    old_source_ids = {index: ("src", index) for index in range(4)}
    _state, delta = session.build_tile_presentation(old_source_ids)
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    session.tile_source_ids = dict(old_source_ids)
    # Shared reduced targets deliberately do not populate rendered_tiles; the
    # lifecycle/display payload is still valid resident presentation input.
    session.rendered_tiles.clear()
    assert set(session.display_tile_payloads) == {0, 1, 2, 3}
    stale = replace(
        session.display_tile_payloads[2],
        source_anchor=PayloadSourceAnchor(
            ("single-image-session",),
            (0, TILE, 0, TILE),
            plane_shape=(TILE, TILE),
        ),
    )
    session.display_tile_payloads[2] = stale
    session.lifecycle.remember_presentable(2, stale)

    partial = _shifted_plan(count=2, offset=2)
    shifted_anchoring = SimpleNamespace(
        source_starts_yx=(5, 7),
        content_key=("current-windowless-view",),
    )
    stats = _retarget(
        session,
        partial,
        new_source_ids={0: ("src", 2), 1: ("src", 3)},
        cached_tiles={},
        source_anchoring=shifted_anchoring,
    )

    assert stats["misses"] == 0
    assert stats["remapped"] == 2
    assert set(session.display_tile_payloads) == {0, 1}
    assert session.display_tile_payloads[0].source_index == 2
    assert session.display_tile_payloads[1].source_index == 3
    remapped_anchor = session.display_tile_payloads[0].source_anchor
    assert remapped_anchor.content_key == (
        ("current-windowless-view",),
        "montage-source",
        2,
    )
    assert session.source_anchoring is shifted_anchoring
    assert remapped_anchor.source_rect == (5, 5 + TILE, 7, 7 + TILE)
    assert set(session.pending_payload_upserts) == {0, 1}
    # The successor shrinks the physical slot topology. The retained values
    # remain useful inputs, but they cannot own an atomic wait for a different
    # layout; bounded ordinary deltas must publish that geometry honestly.
    assert session.atomic_successor_pending is False


def test_expanded_index_window_does_not_arm_ownerless_atomic_successor():
    """A complete small predecessor cannot own the new slots of a larger plan."""

    session = _session(count=4)
    old_source_ids = {index: ("src", index) for index in range(4)}
    _state, delta = session.build_tile_presentation(old_source_ids)
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    session.tile_source_ids = dict(old_source_ids)
    session.rendered_tiles.clear()
    assert session.required_first_pixels_presented()

    expanded = _shifted_plan(count=8, offset=0)
    _retarget(
        session,
        expanded,
        new_source_ids={index: ("src", index) for index in range(8)},
        cached_tiles={},
    )

    assert not session.atomic_successor_pending
    assert session.build_atomic_successor_presentation() is None
    assert session.flush_pending
    assert session.final_commit_pending


def test_same_layout_montage_rebirth_retains_pixels_and_arms_atomic_successor():
    """Rapid churn may rebirth before the ordinary in-place retarget is eligible."""

    previous = _session(count=4)
    successor = _session(count=4)
    document = ArrayDocument(np.zeros((7, TILE, TILE), dtype=np.float32))
    previous.document = document
    successor.document = document
    previous.view_state = (
        ViewState.from_shape((7, TILE, TILE))
        .with_image_axes(1, 2)
        .with_montage_axis(0, columns=4, indices=(0, 1, 2, 3), text="0:4")
    )
    previous._sync_lifecycle_targets()
    _state, previous_delta = previous.build_tile_presentation({})
    _acknowledge(previous, previous_delta)
    previous.mark_presented(tuple(previous_delta.upserts))
    assert previous.required_first_pixels_presented()
    successor.view_state = previous.view_state.with_montage_axis(
        0,
        columns=4,
        indices=(3, 4, 5, 6),
        text="3:7",
    )
    # Auto-level intent is successor presentation metadata. The old pixels
    # and old uniform remain an honest fallback until the atomic handoff.
    successor.force_auto = True
    successor.session_id = 2
    successor_tiles = tuple(
        replace(tile, source_index=index + 3) for index, tile in enumerate(_tiles(4))
    )
    successor.plan = MontagePlan(
        axis=0,
        tile_shape=(TILE, TILE),
        grid_shape=(1, 4),
        columns=4,
        rows=1,
        gap=0,
        tiles=successor_tiles,
    )
    successor.visible_tiles = successor_tiles

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )
    assert decision.retain_pixels
    assert decision.atomic_successor


def test_montage_axis_entry_retains_settled_predecessor_as_bridge():
    """Plane→montage keeps the settled plane visible until the first delta.

    Blanking at the axis change put a multi-second black window between a
    plane and its own montage (R8 continuity gate, 2026-07-18 blackout
    dossier). The bridge is honest retention only: no all-slot atomic
    handoff, and an unsettled predecessor still blanks.
    """

    previous = _session(count=1)
    successor = _session(count=4)
    document = ArrayDocument(np.zeros((4, TILE, TILE), dtype=np.float32))
    previous.document = document
    successor.document = document
    plane_state = ViewState.from_shape((4, TILE, TILE)).with_image_axes(1, 2)
    previous.montage_axis = None
    previous.view_state = plane_state
    previous._sync_lifecycle_targets()
    successor.view_state = plane_state.with_montage_axis(
        0, columns=4, indices=(0, 1, 2, 3), text="0:4"
    )
    _state, previous_delta = previous.build_tile_presentation({})
    _acknowledge(previous, previous_delta)
    previous.mark_presented(tuple(previous_delta.upserts))
    assert previous.required_first_pixels_presented()

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )

    assert decision.retain_pixels
    assert not decision.atomic_successor
    assert decision.reason == "montage-axis-bridge"


def test_montage_axis_entry_blanks_unsettled_predecessor():
    previous = _session(count=1)
    successor = _session(count=4)
    document = ArrayDocument(np.zeros((4, TILE, TILE), dtype=np.float32))
    previous.document = document
    successor.document = document
    plane_state = ViewState.from_shape((4, TILE, TILE)).with_image_axes(1, 2)
    previous.montage_axis = None
    previous.view_state = plane_state
    successor.view_state = plane_state.with_montage_axis(
        0, columns=4, indices=(0, 1, 2, 3), text="0:4"
    )
    assert not previous.required_first_pixels_presented()

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )

    assert not decision.retain_pixels
    assert decision.reason == "montage-axis"
    assert decision.detail == "predecessor-incomplete"


def test_montage_rebirth_continues_pending_bridge_from_pixel_less_predecessor():
    """A rebirth before the bridge successor's first commit keeps the bridge.

    The surface still draws the ORIGINAL bridge predecessor; blanking against
    the pixel-less dying session re-opened the entry black window (fixed3
    trace: session 3 reject predecessor-incomplete at 0.25 s).
    """

    previous = _session(count=4)
    successor = _session(count=4)
    document = ArrayDocument(np.zeros((4, TILE, TILE), dtype=np.float32))
    previous.document = document
    successor.document = document
    montage_state = (
        ViewState.from_shape((4, TILE, TILE))
        .with_image_axes(1, 2)
        .with_montage_axis(0, columns=4, indices=(0, 1, 2, 3), text="0:4")
    )
    previous.view_state = montage_state
    successor.view_state = montage_state
    previous.presentation_bridge_pending = True
    assert not previous.required_first_pixels_presented()

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )

    assert decision.retain_pixels
    assert not decision.atomic_successor
    assert decision.reason == "montage-axis-bridge"


def test_montage_axis_change_with_other_view_state_drift_still_blanks():
    """The bridge covers exactly the montage selection; any other drift hides."""

    previous = _session(count=1)
    successor = _session(count=4)
    document = ArrayDocument(np.zeros((4, TILE, TILE), dtype=np.float32))
    previous.document = document
    successor.document = document
    plane_state = ViewState.from_shape((4, TILE, TILE)).with_image_axes(1, 2)
    previous.montage_axis = None
    previous.view_state = plane_state
    successor.view_state = plane_state.transposed_image_axes().with_montage_axis(
        0, columns=4, indices=(0, 1, 2, 3), text="0:4"
    )
    _state, previous_delta = previous.build_tile_presentation({})
    _acknowledge(previous, previous_delta)
    previous.mark_presented(tuple(previous_delta.upserts))

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )

    assert not decision.retain_pixels
    assert decision.reason == "montage-axis"


def test_expanded_montage_rebirth_retains_pixels_without_atomic_wait():
    """A small predecessor cannot own the additional slots of a larger layout."""

    previous = _session(count=4)
    successor = _session(count=8)
    document = ArrayDocument(np.zeros((8, TILE, TILE), dtype=np.float32))
    previous.document = document
    successor.document = document
    previous.view_state = (
        ViewState.from_shape((8, TILE, TILE))
        .with_image_axes(1, 2)
        .with_montage_axis(0, columns=4, indices=(0, 1, 2, 3), text="0:4")
    )
    successor.view_state = previous.view_state.with_montage_axis(
        0,
        # Layout columns are auto-derived outside ViewState in the live path;
        # keep semantic intent stable while the physical plan expands.
        columns=4,
        indices=tuple(range(8)),
        text="0:8",
    )
    previous._sync_lifecycle_targets()
    _state, previous_delta = previous.build_tile_presentation({})
    _acknowledge(previous, previous_delta)
    previous.mark_presented(tuple(previous_delta.upserts))
    assert previous.required_first_pixels_presented()

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )

    assert decision.retain_pixels
    assert not decision.atomic_successor
    assert decision.reason == "montage-topology-change"


def test_transposed_axes_cannot_retain_predecessor_mappings():
    """Resident pixels survive, but their old source-to-world mapping is hidden."""

    previous = _session(count=4)
    successor = _session(count=4)
    document = ArrayDocument(np.zeros((7, TILE, TILE), dtype=np.float32))
    previous.document = document
    successor.document = document
    previous.view_state = (
        ViewState.from_shape((7, TILE, TILE))
        .with_image_axes(1, 2)
        .with_montage_axis(0, columns=4, indices=(0, 1, 2, 3), text="0:4")
    )
    successor.view_state = previous.view_state.transposed_image_axes()
    previous._sync_lifecycle_targets()
    _state, previous_delta = previous.build_tile_presentation({})
    _acknowledge(previous, previous_delta)
    previous.mark_presented(tuple(previous_delta.upserts))
    assert previous.required_first_pixels_presented()

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )

    assert not decision.retain_pixels
    assert not decision.atomic_successor
    assert decision.reason == "view-state"


def test_same_stage_operation_successor_hides_incompatible_predecessor():
    """Same base storage does not make different operation pixels compatible."""

    previous = _session(count=4)
    successor = _session(count=4)
    base = np.zeros((7, TILE, TILE), dtype=np.float32)
    previous.document = ArrayDocument(base)
    successor.document = ArrayDocument(
        base,
        operations=(CenteredFFT(axis=2),),
    )
    # The operation pipeline changes both values and inferred representation.
    # Its pages may remain resident for a later revert, but they cannot remain
    # visible as fallback for the successor target.
    successor.output_dtype = np.dtype("complex64")
    successor.rgb = True
    view_state = (
        ViewState.from_shape((7, TILE, TILE))
        .with_image_axes(1, 2)
        .with_montage_axis(0, columns=4, indices=(0, 1, 2, 3), text="0:4")
    )
    previous.view_state = view_state
    successor.view_state = view_state
    successor.view_state = replace(successor.view_state, channel=ChannelMode.COMPLEX)
    successor.session_id = 2
    successor.plan = replace(
        successor.plan,
        grid_shape=(1, 13),
        columns=13,
    )
    previous.plan = replace(
        previous.plan,
        grid_shape=(1, 14),
        columns=14,
    )

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )
    assert not decision.retain_pixels
    assert not decision.atomic_successor
    assert decision.reason == "document"


def test_different_base_document_cannot_retain_predecessor():
    previous = _session(count=4)
    successor = _session(count=4)
    previous.document = ArrayDocument(np.zeros((7, TILE, TILE), dtype=np.float32))
    successor.document = ArrayDocument(np.zeros((7, TILE, TILE), dtype=np.float32))
    view_state = (
        ViewState.from_shape((7, TILE, TILE))
        .with_image_axes(1, 2)
        .with_montage_axis(0, columns=4, indices=(0, 1, 2, 3), text="0:4")
    )
    previous.view_state = view_state
    successor.view_state = view_state

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )
    assert not decision.retain_pixels
    assert not decision.atomic_successor
    assert decision.reason == "document"


def test_different_surface_axis_cannot_retain_predecessor():
    previous = _session(count=4)
    successor = _session(count=3)
    document = ArrayDocument(np.zeros((4, TILE, TILE), dtype=np.float32))
    previous.document = document
    successor.document = document
    previous.montage_axis = 1

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )
    assert not decision.retain_pixels
    assert not decision.atomic_successor


def test_hidden_operation_successor_cannot_resurrect_atomic_handoff():
    """A compatible rebirth after an incompatible blank has no predecessor."""

    previous = _session(count=4)
    successor = _session(count=4)
    document = ArrayDocument(np.zeros((7, TILE, TILE), dtype=np.float32))
    previous.document = document
    successor.document = document
    previous.session_id = 2
    successor.session_id = 3

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=False,
    )

    assert not decision.retain_pixels
    assert not decision.atomic_successor
    assert decision.reason == "predecessor-hidden"


def test_visible_partial_montage_cannot_arm_atomic_handoff():
    """One streamed tile is visibility, not complete predecessor coverage."""

    previous = _session(count=4)
    successor = _session(count=4)
    document = ArrayDocument(np.zeros((7, TILE, TILE), dtype=np.float32))
    previous.document = document
    successor.document = document
    view_state = (
        ViewState.from_shape((7, TILE, TILE))
        .with_image_axes(1, 2)
        .with_montage_axis(0, columns=4, indices=(0, 1, 2, 3), text="0:4")
    )
    previous.view_state = view_state
    successor.view_state = view_state
    _state, delta = previous.build_tile_presentation({})
    partial = TileCommitReport(
        presented_tiles=frozenset({0}),
        committed_upserts=frozenset({0}),
    )
    previous.acknowledge_tile_presentation(delta, partial)
    previous.mark_presented((0,))
    assert not previous.required_first_pixels_presented()

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )

    assert not decision.retain_pixels
    assert not decision.atomic_successor
    assert decision.reason == "predecessor-incomplete"


def test_atomic_predecessor_chain_remains_complete_across_rapid_rebirth():
    """The pending atomic obligation is the transitive coverage proof."""

    previous = _session(count=4)
    successor = _session(count=4)
    document = ArrayDocument(np.zeros((7, TILE, TILE), dtype=np.float32))
    previous.document = document
    successor.document = document
    view_state = (
        ViewState.from_shape((7, TILE, TILE))
        .with_image_axes(1, 2)
        .with_montage_axis(0, columns=4, indices=(0, 1, 2, 3), text="0:4")
    )
    previous.view_state = view_state
    successor.view_state = view_state
    previous.atomic_successor_pending = True

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )

    assert decision.retain_pixels
    assert decision.atomic_successor
    assert decision.reason == "montage-compatible"


def test_displayed_axis_crop_rebirth_arms_and_chains_atomic_successor(monkeypatch):
    """Displayed-axis ranges are source selections, not incompatible views."""

    previous = _session(count=4)
    successor = _session(count=4)
    document = ArrayDocument(np.zeros((TILE, TILE, 7), dtype=np.float32))
    previous.document = document
    successor.document = document
    view_state = (
        ViewState.from_shape((TILE, TILE, 7))
        .with_image_axes(1, 0)
        .with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text="0:4")
    )
    previous.view_state = view_state.with_axis_range(
        0,
        indices=tuple(range(8, 24)),
        text="8:24",
    )
    successor.view_state = view_state.with_axis_range(
        0,
        indices=tuple(range(7, 23)),
        text="7:23",
    )
    monkeypatch.setattr(previous, "required_first_pixels_presented", lambda: True)

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )

    assert decision.retain_pixels
    assert decision.atomic_successor
    assert decision.reason == "montage-compatible"

    # A third scrollbar input can replace the pixel-less successor before its
    # transaction lands. The inherited pending obligation proves that the
    # original complete predecessor is still physically visible.
    successor.atomic_successor_pending = True
    chained = _session(count=4)
    chained.document = document
    chained.view_state = view_state.with_axis_range(
        0,
        indices=tuple(range(6, 22)),
        text="6:22",
    )
    decision = plan_presentation_transition(
        successor,
        chained,
        predecessor_visible=True,
    )

    assert decision.retain_pixels
    assert decision.atomic_successor
    assert decision.reason == "montage-compatible"


def test_partial_viewport_rebirth_atomically_hands_off_required_scope():
    previous = _session(count=4)
    successor = _session(count=4)
    document = ArrayDocument(np.zeros((7, TILE, TILE), dtype=np.float32))
    previous.document = document
    successor.document = document
    view_state = (
        ViewState.from_shape((7, TILE, TILE))
        .with_image_axes(1, 2)
        .with_montage_axis(0, columns=4, indices=(0, 1, 2, 3), text="0:4")
    )
    previous.view_state = view_state
    successor.view_state = view_state
    previous.frame_plan = SimpleNamespace(active_region_ids=(1,))
    successor.frame_plan = SimpleNamespace(active_region_ids=(1,))
    previous._sync_lifecycle_targets()
    _state, delta = previous.build_tile_presentation({})
    _acknowledge(previous, delta)
    previous.mark_presented(tuple(delta.upserts))

    decision = plan_presentation_transition(
        previous,
        successor,
        predecessor_visible=True,
    )

    assert decision.retain_pixels
    assert decision.atomic_successor
    assert decision.reason == "montage-compatible"


def _real_frame_plan(session, view_range):
    state = (
        ViewState.from_shape((TILE, TILE, 4))
        .with_image_axes(0, 1)
        .with_montage_axis(2, columns=4, indices=tuple(range(4)))
    )
    session.view_state = state
    session.frame_plan = FramePlanner().plan(
        target=FrameTarget(("semantic",), view_range, ("levels",), "exact-visible"),
        view_state=state,
        display_shape=session.plan.display_shape,
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        viewport_shape=VIEWPORT,
        view_range=view_range,
        montage_plan=session.plan,
    )
    return session.frame_plan


def test_atomic_handoff_has_one_owner_after_transition_arms_it():
    """Semantic-frame lag cannot override a physical handoff obligation."""

    from arrayscope.window.frame_effects import _atomic_successor_handoff_pending

    session = _session(count=4)
    session.atomic_successor_pending = True
    # The persistent tile layer can still own the complete predecessor while
    # the committed semantic frame is absent or non-tiled.  The transition
    # owner already proved physical coverage when it armed this flag.
    session.display_tile_payloads.clear()

    assert _atomic_successor_handoff_pending(session)
    session.atomic_successor_pending = False
    assert not _atomic_successor_handoff_pending(session)


def test_atomic_successor_commit_is_not_gated_by_refinement_phase():
    """Every backend retains the old frame until a full successor is ready."""

    from arrayscope.display.backend_contract import WGPU_CAPABILITIES
    from arrayscope.window.frame_effects import _atomic_successor_commit_modes

    assert _atomic_successor_commit_modes(PYQTGRAPH_CAPABILITIES, pending=True) == (True, False)
    assert _atomic_successor_commit_modes(WGPU_CAPABILITIES, pending=True) == (False, True)
    assert _atomic_successor_commit_modes(WGPU_CAPABILITIES, pending=False) == (False, False)


def test_atomic_handoff_revalidates_required_subset_of_coverage():
    """A camera change cannot carry a handoff outside presentation coverage."""

    from arrayscope.window.frame_effects import _atomic_successor_handoff_pending

    session = _session(count=4)
    _real_frame_plan(session, ((0.0, 8.0), (0.0, 8.0)))
    assert session.frame_plan.active_region_ids == (0,)
    session.visible_tiles = session.plan.tiles[1:2]
    session.visible_tile_numbers = frozenset({1})
    session.atomic_successor_pending = True

    assert not _atomic_successor_handoff_pending(session)
    assert not session.atomic_successor_pending


def test_atomic_handoff_never_arms_for_empty_real_frame_plan():
    """An off-montage camera has no successor transaction to hand off."""

    from arrayscope.window.frame_effects import _atomic_successor_handoff_pending

    session = _session(count=4)
    outside = ((10_000.0, 10_008.0), (10_000.0, 10_008.0))
    _real_frame_plan(session, outside)
    assert session.frame_plan.active_region_ids == ()
    session.atomic_successor_pending = True

    assert not _atomic_successor_handoff_pending(session)
    assert not session.atomic_successor_pending


def test_index_window_retarget_arms_atomic_successor_pending():
    """An in-place session retarget cannot inherit the prior atomic handoff."""

    session = _session(count=4)
    session.semantic_key = ("semantic", "stable")
    old_source_ids = {index: session.tile_semantic_source_id(index) for index in range(4)}
    _state, delta = session.build_tile_presentation(old_source_ids)
    report = TileCommitReport(
        presented_tiles=frozenset(delta.upserts),
        committed_upserts=frozenset(delta.upserts),
    )
    acknowledged = session.acknowledge_tile_presentation(delta, report)
    session.atomic_successor_pending = True
    assert session.acknowledge_atomic_successor(
        delta,
        report,
        acknowledged,
    )
    assert not session.atomic_successor_pending
    session.mark_presented(tuple(delta.upserts))
    session.tile_source_ids = dict(old_source_ids)
    session.rendered_tiles.clear()

    successor_tiles = tuple(
        replace(tile, source_index=(index + 1) % 4) for index, tile in enumerate(_tiles(4))
    )
    successor = MontagePlan(
        axis=0,
        tile_shape=(TILE, TILE),
        grid_shape=(1, 4),
        columns=4,
        rows=1,
        gap=0,
        tiles=successor_tiles,
    )
    _retarget(
        session,
        successor,
        new_source_ids={
            index: session.tile_semantic_source_id((index + 1) % 4) for index in range(4)
        },
        semantic_key=session.semantic_key,
    )

    assert session.atomic_successor_pending
    atomic = session.build_atomic_successor_presentation()
    assert atomic is not None, session._atomic_fast_reject_reason
    _state, successor_delta = atomic
    assert successor_delta.atomic_handoff is True
    assert tuple(successor_delta.upserts) == (0, 1, 2, 3)
    assert tuple(
        successor_delta.upserts[index].source_index for index in successor_delta.upserts
    ) == (1, 2, 3, 0)


def test_index_window_retarget_retains_atomic_floor_while_lod_target_reopens():
    """A new LOD target must not erase the complete physical predecessor.

    Rapid source scrolling can supersede a session while its exact target is
    still refining.  The already acknowledged fallback pixels remain a full
    frame and must keep the next same-topology successor atomic.
    """

    session = _session(count=4)
    session.semantic_key = ("semantic", "stable")
    source_ids = {index: session.tile_semantic_source_id(index) for index in range(4)}
    _state, predecessor = session.build_tile_presentation(source_ids)
    _acknowledge(session, predecessor)
    session.mark_presented(tuple(predecessor.upserts))
    session.tile_source_ids = dict(source_ids)

    for tile in range(4):
        # Target refinement reopens the lifecycle row before replacement
        # pixels arrive; the acknowledged tile-presentation state remains the
        # physical predecessor drawn by the backend.
        session.lifecycle.presentation_discarded(tile)

    assert not session.required_first_pixels_presented()
    assert session.required_presentation_coverage_complete()

    successor = _shifted_plan(count=4, offset=40)
    _retarget(
        session,
        successor,
        new_source_ids={
            index: session.tile_semantic_source_id(tile.source_index)
            for index, tile in enumerate(successor.tiles)
        },
        semantic_key=session.semantic_key,
    )

    assert session.atomic_successor_pending


def test_index_window_retarget_atomically_hands_off_only_required_center():
    """Reproduce stall 16: a deep-zoom far scroll must publish its center.

    The frame plan requires one on-screen slot while the margin-expanded
    session owns four.  Scheduling intentionally produces only the required
    successor, so an all-slot atomic handoff would have no owner for the three
    shell replacements and could never commit the ready exact center.
    """

    session = _session(count=4)
    session.semantic_key = ("semantic", "stable")
    session.frame_plan = SimpleNamespace(active_region_ids=(1,))
    session.sync_lifecycle_scope()
    old_source_ids = {index: session.tile_semantic_source_id(index) for index in range(4)}
    _state, predecessor = session.build_tile_presentation(old_source_ids)
    _acknowledge(session, predecessor)
    session.mark_presented(tuple(predecessor.upserts))
    session.tile_source_ids = dict(old_source_ids)

    successor = _shifted_plan(count=4, offset=100)
    center = replace(session.rendered_tiles[1], tile=successor.tiles[1])
    _retarget(
        session,
        successor,
        new_source_ids={
            index: session.tile_semantic_source_id(tile.source_index)
            for index, tile in enumerate(successor.tiles)
        },
        cached_tiles={1: center},
        semantic_key=session.semantic_key,
    )

    assert session.atomic_successor_pending
    atomic = session.build_atomic_successor_presentation()
    assert atomic is not None, session._atomic_fast_reject_reason
    _state, delta = atomic
    assert tuple(delta.active_tiles) == (1,)
    assert tuple(delta.upserts) == (1,)
    assert delta.upserts[1].quality == "exact"


def test_atomic_successor_requires_complete_backend_acknowledgement():
    session = _session(count=2)
    session.atomic_successor_pending = True
    _state, delta = session.build_tile_presentation({0: ("src", 0), 1: ("src", 1)})
    assert delta.atomic_handoff is False
    partial = TileCommitReport(
        presented_tiles=frozenset({0}),
        committed_upserts=frozenset({0}),
    )
    acknowledged = session.acknowledge_tile_presentation(delta, partial)

    assert not session.acknowledge_atomic_successor(
        delta,
        partial,
        acknowledged,
    )
    assert session.atomic_successor_pending


def test_resident_only_remap_discards_stale_rendered_slot_owner():
    """A remapped lifecycle payload must not be overwritten by old slot data."""

    session = _session(count=4)
    old_source_ids = {index: ("src", index) for index in range(4)}
    _state, delta = session.build_tile_presentation(old_source_ids)
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    session.tile_source_ids = dict(old_source_ids)
    # Sources 2/3 are resident only through lifecycle payloads, while the new
    # destination slots 0/1 still contain unrelated native renderer entries.
    session.rendered_tiles.pop(2)
    session.rendered_tiles.pop(3)

    partial = _shifted_plan(count=2, offset=2)
    stats = _retarget(
        session,
        partial,
        new_source_ids={0: ("src", 2), 1: ("src", 3)},
        cached_tiles={},
    )

    assert stats["remapped"] == 2
    assert session.rendered_tiles == {}
    assert [session.display_tile_payloads[index].source_index for index in (0, 1)] == [2, 3]

    presentation, _delta = session.build_tile_presentation({0: ("src", 2), 1: ("src", 3)})
    assert [presentation.payloads[index].source_index for index in (0, 1)] == [2, 3]


def test_partial_index_window_derives_reuse_map_from_payload_identity():
    """A lazy/partial source-id cache must not defeat semantic payload reuse."""

    session = _session(count=4)
    old_source_ids = {index: ("src", index) for index in range(4)}
    _state, delta = session.build_tile_presentation(old_source_ids)
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    session.rendered_tiles.clear()
    # Model a lazy source cache containing only the first slot.  The display
    # payloads still carry canonical semantic identities for every source.
    session.tile_source_ids = {0: ("src", 0)}

    partial = _shifted_plan(count=2, offset=2)
    stats = _retarget(
        session,
        partial,
        new_source_ids={0: ("src", 2), 1: ("src", 3)},
        cached_tiles={},
    )

    assert stats["misses"] == 0
    assert stats["remapped"] == 2
    assert [session.display_tile_payloads[index].source_index for index in (0, 1)] == [2, 3]


def _resident_remap_session(count=4):
    """A session whose sources are resident only as lifecycle payloads."""

    session = _session(count=count)
    old_source_ids = {index: ("src", index) for index in range(count)}
    _state, delta = session.build_tile_presentation(old_source_ids)
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    session.tile_source_ids = dict(old_source_ids)
    session.rendered_tiles.clear()
    return session


def _counting(session, name):
    """Replace one session method with a call-counting wrapper."""

    original = getattr(session, name)
    calls: list[object] = []

    def wrapper(*args, **kwargs):
        calls.append(args[0] if args else None)
        return original(*args, **kwargs)

    setattr(session, name, wrapper)
    return calls


def test_index_window_retarget_reports_each_remapped_payload_once():
    """The remap and its scope sync must not both report the same payload.

    ``retarget_index_window`` installs a remapped payload and reports it to the
    lifecycle; ``sync_lifecycle_scope``'s safety-net scan then re-reported the
    very same object because its memo still held the predecessor.  Every
    remapped tile therefore paid two payload normalizations and emitted two
    identical lifecycle trace edges — measured at ~5 ms of a ~29 ms retarget on
    a 100-tile montage, and a doubled statement on the trace bus.
    """

    session = _resident_remap_session()
    reported = _counting(session, "record_tile_payload")

    stats = _retarget(
        session,
        _shifted_plan(count=2, offset=2),
        new_source_ids={0: ("src", 2), 1: ("src", 3)},
        cached_tiles={},
    )

    assert stats["remapped"] == 2
    assert sorted(int(payload.tile_number) for payload in reported) == [0, 1]
    # Reporting once is only correct if the payload is genuinely current.
    for index in (0, 1):
        payload = session.display_tile_payloads[index]
        assert session.lifecycle.payload_is_current(index, payload)
        assert payload.source_id in session.lifecycle.peek(index).presentable_payloads


def test_index_window_retarget_reports_again_when_targets_move_under_it():
    """A later target adoption invalidates the "already reported" memo.

    Adopting a new semantic source prunes the record's presentable payloads, so
    a memo entry claiming the slot's payload was already reported would suppress
    the report that puts it back — the tile would lose its first-pixel fallback.
    """

    session = _resident_remap_session()
    _retarget(
        session,
        _shifted_plan(count=2, offset=2),
        new_source_ids={0: ("src", 2), 1: ("src", 3)},
        cached_tiles={},
    )

    # No target movement: the scan trusts the remap's report and stays quiet.
    quiet = _counting(session, "record_tile_payload")
    session.sync_lifecycle_scope()
    assert quiet == []

    # The targets move to a different semantic source window underneath the
    # primed memo; the scan must report every live payload again.
    session.visible_tiles = tuple(
        replace(tile, source_index=int(tile.source_index) + 10) for tile in session.visible_tiles
    )
    reported = _counting(session, "record_tile_payload")
    session.sync_lifecycle_scope()
    assert sorted(int(payload.tile_number) for payload in reported) == [0, 1]


def test_index_window_retarget_publishes_its_lifecycle_targets_once():
    """The remap moves payloads between slots the targets already name.

    Publishing the targets before the per-tile remap and rebuilding the whole
    set afterwards cost a typed target identity per tile twice over.  The second
    build is skipped only while the target inputs are provably unmoved, so the
    guard must decline the moment one of them does move.
    """

    session = _resident_remap_session()
    syncs = _counting(session, "_sync_lifecycle_targets")

    _retarget(
        session,
        _shifted_plan(count=2, offset=2),
        new_source_ids={0: ("src", 2), 1: ("src", 3)},
        cached_tiles={},
    )

    assert len(syncs) == 1

    snapshot = session._lifecycle_target_inputs()
    assert session._lifecycle_targets_still_current(snapshot)
    # A rebuilt visible set declines even when it compares equal: the guard is
    # deliberately identity-based, so "unsure" always falls back to a recompute.
    session.visible_tiles = tuple(tile for tile in session.visible_tiles)
    assert not session._lifecycle_targets_still_current(snapshot)

    snapshot = session._lifecycle_target_inputs()
    session.skipped_tiles.add(0)
    assert not session._lifecycle_targets_still_current(snapshot)


def test_retarget_index_window_demotes_misses_with_immediate_invalidation():
    """A miss exposes no predecessor payload after surface invalidation."""

    session = _session(count=2)
    old_sources = {0: ("src", 0), 1: ("src", 1)}
    for tile in session.plan.tiles:
        image = np.full((TILE, TILE), float(tile.source_index), dtype=np.float32)
        session.mark_materialized(
            RenderedTile(
                tile=tile,
                image=image,
                histogram_data=None,
                eval_ms=0.0,
                slab_shape=(TILE, TILE),
                slab_nbytes=TILE * TILE * 4,
            )
        )
    state, delta = session.build_tile_presentation(old_sources)
    session.acknowledge_tile_presentation(
        delta, TileCommitReport(presented_tiles=state.active_payloads(delta))
    )
    session.mark_presented(state.active_payloads(delta))

    plan = _shifted_plan(count=2, offset=5)

    stats = _retarget(
        session,
        plan,
        new_source_ids={0: ("src", 5), 1: ("src", 6)},
        cached_tiles={},
    )

    assert stats["misses"] == 2
    assert stats["hits"] == 0
    for index in (0, 1):
        assert index not in session.rendered_tiles
        assert index not in session.loading_tiles
        assert index in session.dirty_payloads
        assert index not in session.tile_source_ids
    assert session.flush_pending is True
    assert session.final_commit_pending is True
    # Lifecycle semantic axis demoted (no longer evaluated).
    assert not session.lifecycle.evaluating_tiles

    replacement_state, replacement_delta = session.build_tile_presentation(
        {0: ("src", 5), 1: ("src", 6)}
    )
    # The frame controller invalidates physical mappings atomically before it
    # calls retarget_index_window.  The lifecycle must therefore publish only
    # current targets and placeholders here; synthesizing a second backend
    # removal obligation would duplicate surface ownership.
    assert replacement_delta.removals == ()
    assert replacement_state.active_payloads(replacement_delta) == {}
    assert {
        tile: identity.source_index
        for tile, identity in replacement_delta.target_identities.items()
    } == {0: 5, 1: 6}
    assert not session.visible_plan_complete()


def test_note_committed_cannot_clear_owned_presentation_backlog():
    session = _session(count=2)
    session.dirty_payloads[0] = None
    session.pending_payload_upserts[1] = None
    session.final_commit_pending = True
    session.flush_pending = True

    session.note_committed()

    assert session.final_commit_pending is True
    assert session.flush_pending is True

    session.dirty_payloads.clear()
    session.pending_payload_upserts.clear()
    session.note_committed()

    assert session.final_commit_pending is False
    assert session.flush_pending is False


def test_materialized_tile_arms_its_presentation_obligation():
    session = _session(count=2)
    session.final_commit_pending = False
    session.flush_pending = False
    tile = session.plan.tiles[0]

    session.mark_materialized(
        RenderedTile(
            tile=tile,
            image=np.ones((TILE, TILE), dtype=np.float32),
            histogram_data=None,
            eval_ms=0.0,
            slab_shape=(TILE, TILE),
            slab_nbytes=TILE * TILE * 4,
        )
    )

    assert 0 in session.dirty_payloads
    assert session.flush_pending is True
    assert session.final_commit_pending is True


def test_presentation_backlog_signature_distinguishes_same_size_retarget():
    from arrayscope.window.frame_effects import FramePipelineEffects

    session = _session(count=2)
    effects = FramePipelineEffects(object(), session)
    session.dirty_payloads.clear()
    session.dirty_payloads[0] = None
    first = effects._backlog_signature()

    session.dirty_payloads.clear()
    session.dirty_payloads[1] = None
    second = effects._backlog_signature()

    assert first != second


def test_live_phase_vector_ladder_builds_cancellation_preserving_page_payload():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=1)
    cancellation_cell = np.asarray(
        [[1.0 + 0.0j, -1.0 + 0.0j], [0.0j, 0.0j]],
        dtype=np.complex64,
    )
    values = np.tile(cancellation_cell, (TILE // 2, TILE // 2))
    rendered = _phase_shader_rendered(session, values)

    assert render_lod.resident_lod_active(session) is True
    assert render_lod.selected_lod_factor(session) == 1
    demand = session.lod_policy_decision.demand
    assert demand.desired_level == 2
    key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=demand.desired_level,
        semantic_source_id=session.tile_semantic_source_id(0),
        shader_display=True,
    )
    assert key.reducer == "phase_vector"
    _admit_page_set(pyramid, key, values)

    assert render_lod.selected_lod_factor(session) == 4
    payload = session._ensure_display_tile_payload(0, rendered, {}, lod_factor=4)

    assert payload.lod.level == 2
    assert payload.texture_kind == TexturePlaneKind.COMPLEX_RG32F
    assert payload.page_backing is not None
    assert {plan.reducer for plan in payload.page_backing.requested_plans} == {"phase_vector"}
    assert payload.histogram_data is None
    assert payload.shader_mapping == rendered.shader_mapping
    np.testing.assert_array_equal(payload.texture_data, np.zeros((16, 16), np.complex64))
    np.testing.assert_array_equal(
        payload.semantic_histogram_data,
        rendered.histogram_data,
    )


def test_live_phase_ladder_does_not_accept_resident_mean_family_as_floor():
    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=1)
    _settle_first_pixels(session)
    yy, xx = np.mgrid[:TILE, :TILE]
    values = ((1.0 + yy + xx) * np.exp(1j * (xx - yy) / 7.0)).astype(np.complex64)
    phase_rendered = _phase_shader_rendered(session, values)
    render_lod.selected_lod_factor(session)
    demand = session.lod_policy_decision.demand
    semantic_id = session.tile_semantic_source_id(0)
    phase_key = page_set_key_for_rendered(
        phase_rendered,
        demand=demand,
        level=demand.desired_level,
        semantic_source_id=semantic_id,
        shader_display=True,
    )

    complex_state = ViewState.from_shape(values.shape).with_channel(ChannelMode.COMPLEX)
    complex_display = make_shader_image_from_slab(
        values,
        SimpleNamespace(view_state=complex_state, ranged_axes=()),
    )
    mean_rendered = replace(
        phase_rendered,
        image=complex_display.data,
        histogram_data=complex_display.histogram_data,
        shader_mapping=complex_display.shader_mapping,
        texture_kind=complex_display.texture_kind,
        semantic_data=complex_display.semantic_data,
        lod_source_data=complex_display.lod_source_data,
    )
    mean_key = page_set_key_for_rendered(
        mean_rendered,
        demand=demand,
        level=demand.desired_level,
        semantic_source_id=semantic_id,
        shader_display=True,
    )
    assert phase_key.reducer == "phase_vector"
    assert mean_key.reducer == "mean"
    _admit_page_set(pyramid, mean_key, values)

    assert not render_lod._page_set_complete(pyramid, phase_key)
    assert render_lod.best_floor_key(session, 0, tile_number=0) is None
    effects = FramePipelineEffects(_RungPrepareRenderer(), session)
    assert effects.prepare_rung(
        _pipeline_intent_for(session),
        _desired_materialization_step(level=demand.desired_level),
    )
    request = session.lifecycle.materialization_request_for(0, phase_key)
    assert request is not None
    assert request.key.reducer == "phase_vector"
    assert tuple(plan.key for plan in request.claimed_plans) == phase_key.page_keys
    session.pending_rung_materializations.release(request)
    assert pyramid.pending_count == 0


def test_complex_display_keeps_resident_mean_complex_lod_active():
    session = _session(pyramid=LodPageCache(max_bytes=1 << 20))
    session.view_state = type("View", (), {"channel": "complex"})()

    assert render_lod.resident_lod_active(session) is True


def test_stale_presentation_gate_cannot_clear_successor_wakeup():
    from types import SimpleNamespace

    predecessor = SimpleNamespace(session_id=7)
    successor = SimpleNamespace(session_id=8)
    successor_owner = (8, id(successor))
    renderer = SimpleNamespace(
        _montage_presentation_gate_armed=True,
        _montage_presentation_gate_owner=successor_owner,
    )
    stale = FramePipelineEffects(renderer, predecessor)
    current = FramePipelineEffects(renderer, successor)
    commits = []
    stale.commit_pending_session = lambda: commits.append("stale")
    current.commit_pending_session = lambda: commits.append("current")
    current._session_is_current = lambda: True

    stale._on_presentation_gate()
    assert renderer._montage_presentation_gate_armed is True
    assert renderer._montage_presentation_gate_owner == successor_owner
    assert commits == []

    current._on_presentation_gate()
    assert renderer._montage_presentation_gate_armed is False
    assert renderer._montage_presentation_gate_owner is None
    assert commits == ["current"]


def test_retargeted_session_gate_event_keeps_its_immutable_owner_generation():
    session = SimpleNamespace(session_id=7)
    predecessor_owner = (7, id(session))
    renderer = SimpleNamespace(
        _montage_presentation_gate_armed=True,
        _montage_presentation_gate_owner=predecessor_owner,
    )
    effects = FramePipelineEffects(renderer, session)
    commits = []
    effects.commit_pending_session = lambda: commits.append("commit")
    effects._session_is_current = lambda: True

    session.session_id = 8
    successor_owner = (8, id(session))
    renderer._montage_presentation_gate_owner = successor_owner
    effects._on_presentation_gate(predecessor_owner)

    assert renderer._montage_presentation_gate_armed is True
    assert renderer._montage_presentation_gate_owner == successor_owner
    assert commits == []

    effects._on_presentation_gate(successor_owner)
    assert renderer._montage_presentation_gate_armed is False
    assert renderer._montage_presentation_gate_owner is None
    assert commits == ["commit"]


def test_rebuilt_payload_wrapper_with_same_source_identity_is_not_reemitted():
    session = _session(count=2)
    source_ids = {0: ("src", 0), 1: ("src", 1)}
    _state, delta = session.build_tile_presentation(source_ids)
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    previous = session.display_tile_payloads[0]
    session.display_tile_payloads[0] = replace(previous)
    session.dirty_payloads[0] = None

    _next_state, next_delta = session.build_tile_presentation(source_ids)

    assert 0 not in next_delta.upserts
    assert 0 not in session.dirty_payloads


def test_stale_backend_identity_is_not_acknowledged_when_correct_upsert_is_budgeted_out():
    session = _session(count=2)
    old_sources = {0: ("src", 0), 1: ("src", 1)}
    _old_state, old_delta = session.build_tile_presentation(old_sources)
    session.acknowledge_tile_presentation(
        old_delta,
        TileCommitReport(
            presented_tiles=frozenset(old_delta.upserts),
            committed_upserts=frozenset(old_delta.upserts),
            presented_identities={
                int(tile): payload.source_id for tile, payload in old_delta.upserts.items()
            },
        ),
    )
    session.mark_presented(old_delta.upserts)

    plan = _shifted_plan(count=2, offset=5)
    cached = {}
    for tile in plan.tiles:
        image = np.full((TILE, TILE), float(tile.source_index), dtype=np.float32)
        cached[int(tile.montage_index)] = RenderedTile(
            tile=tile,
            image=image,
            histogram_data=None,
            eval_ms=0.0,
            slab_shape=(TILE, TILE),
            slab_nbytes=TILE * TILE * 4,
        )
    _retarget(
        session,
        plan,
        new_source_ids={0: ("src", 5), 1: ("src", 6)},
        cached_tiles=cached,
    )

    _state, delta = session.build_tile_presentation(
        {0: ("src", 5), 1: ("src", 6)},
        max_upserts=1,
    )

    assert len(delta.upserts) == 1
    assert set(delta.upserts).isdisjoint(delta.removals)
    deferred = ({0, 1} - set(delta.upserts)).pop()
    assert delta.removals == ()
    assert deferred in delta.active_tiles
    assert deferred in session.dirty_payloads
    assert deferred not in _state.payloads


def test_retarget_index_window_clears_active_work_without_completion_queue():
    """In-flight completions for the old window are rejected by session id."""

    session = _session(count=2)
    session.tile_source_ids = {0: ("src", 0), 1: ("src", 1)}
    session.active_tile_requests.update({0, 1})
    session.level_evidence_inflight = True
    plan = _shifted_plan(count=2, offset=3)

    stats = _retarget(
        session,
        plan,
        new_source_ids={0: ("src", 3), 1: ("src", 4)},
        cached_tiles={},
    )

    assert stats["misses"] == 2
    assert not session.active_tile_requests
    assert session.level_evidence_inflight is False


def test_stale_rung_drop_releases_its_admitted_active_claim_after_retarget():
    """Cleanup is owned by the admitted rung, not by the live semantic key."""

    session = _session(count=1)
    session.rendered_tiles.clear()
    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    old_intent = _pipeline_intent_for(session, viewport_key="old")
    step = _exact_step(0)

    assert effects.prepare_rung(old_intent, step)
    effects.rung_admitted(old_intent, step, ("task", "old"))
    assert 0 in session.active_tile_requests
    assert 0 in session.loading_tiles
    assert session.lifecycle.evaluating_tiles == frozenset({0})

    session.key = ("retargeted",)
    effects.rung_dropped(old_intent, step)

    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.lifecycle.evaluating_tiles == frozenset()
    assert session._test_replan_requested is True


def test_presentation_gate_does_not_treat_revision_metadata_as_backlog():
    """A settled session must not re-commit merely because revisions are nonzero."""

    from types import SimpleNamespace

    session = SimpleNamespace(
        flush_pending=False,
        final_commit_pending=False,
        dirty_payloads={},
        pending_payload_upserts={},
        pending_removals=set(),
        has_pending_level_update=lambda: False,
        backend_identity_mismatch_tiles=lambda: (),
    )
    renderer = SimpleNamespace(_montage_gate_last_backlog=("previous",))
    effects = FramePipelineEffects(renderer, session)
    requests = []
    effects.request_presentation = lambda: requests.append(True)

    effects._rearm_if_backlog()

    assert requests == []
    assert renderer._montage_gate_last_backlog is None


def test_incomplete_atomic_handoff_rearms_drained_transaction_payloads():
    """An empty physical report cannot leave an atomic transaction ownerless."""

    from dataclasses import replace

    session = _session(count=2)
    session.atomic_successor_pending = True
    _state, ordinary = session.build_tile_presentation({0: ("src", 0), 1: ("src", 1)})
    delta = replace(ordinary, atomic_handoff=True)
    session.dirty_payloads.clear()
    session.pending_payload_upserts.clear()
    session.pending_removals.clear()
    session.final_commit_pending = False
    session.flush_pending = False
    empty = TileCommitReport(
        presented_tiles=frozenset(),
        committed_upserts=frozenset(),
    )
    acknowledged = session.acknowledge_tile_presentation(delta, empty)

    assert not session.acknowledge_atomic_successor(delta, empty, acknowledged)
    assert set(session.pending_payload_upserts) == {0, 1}
    assert session.final_commit_pending is True
    assert session.flush_pending is True


def test_presentation_gate_rearms_completed_first_pass_publication():
    """Complete rough evidence is an explicit metadata commit obligation."""

    from types import SimpleNamespace

    session = SimpleNamespace(
        first_pass_quality="preview",
        first_pass_histogram_published=False,
        flush_pending=False,
        final_commit_pending=False,
        dirty_payloads={},
        pending_payload_upserts={},
        pending_removals=set(),
        has_pending_level_update=lambda: False,
        has_stale_level_presentations=lambda: False,
        backend_identity_mismatch_tiles=lambda: (),
        level_revision=1,
        level_presentation_snapshot=lambda: SimpleNamespace(stale_count=0),
        display_tile_payloads={},
        rendered_tiles={},
    )
    renderer = SimpleNamespace(
        _montage_gate_last_backlog=None,
        _first_pass_level_evidence_complete=lambda _session: True,
    )
    effects = FramePipelineEffects(renderer, session)
    requests = []
    effects.request_presentation = lambda: requests.append(True)

    effects._rearm_if_backlog()

    assert requests == [True]


def test_stale_commit_batch_releases_admitted_active_claim_after_key_retarget():
    """A completion between session key mutation and pipeline replan must drop."""

    session = _session(count=1)
    session.rendered_tiles.clear()
    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    old_intent = _pipeline_intent_for(session, viewport_key="old")
    step = _exact_step(0)

    assert effects.prepare_rung(old_intent, step)
    effects.rung_admitted(old_intent, step, ("task", "old"))
    session.key = ("retargeted",)

    effects.apply_commit(
        CommitBatch(
            semantic_key=old_intent.semantic_key,
            presentation_key=old_intent.presentation_key,
            upserts=((step, object()),),
        )
    )

    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.lifecycle.evaluating_tiles == frozenset()
    assert session._test_replan_requested is True


def test_commuting_desired_reduced_input_uses_preview_claim_not_native():
    """Commuting DESIRED display targets run as reduced-display work."""

    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    _settle_first_pixels(session)
    session.lod_preview_level = 6
    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session, viewport_key="old")
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=False,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )
    level_key = replace(
        session._lod_page_set_key_for(
            session.rendered_tiles[0], demand=session.ingest_lod_demand(), level=2
        ),
        source_id=("old-source",),
    )
    request = session._lod_materialization_request(
        session.rendered_tiles[0],
        demand=session.ingest_lod_demand(),
        level=2,
        key=level_key,
    )
    session.pending_rung_materializations.append(request)
    del session.rendered_tiles[0]
    session.dirty_payloads.pop(0, None)

    assert effects.prepare_rung(intent, step)
    effects.rung_admitted(intent, step, ("display-payload", 0))
    claim_identity = effects._preview_claim_identity(intent, session.plan.tiles[0])
    assert session.lifecycle.preview_claim_matches(0, int(Rung.DESIRED), 2, claim_identity)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.pending_rung_materializations

    effects.rung_dropped(intent, step)

    assert not session.pending_rung_materializations
    assert not session.lifecycle.preview_claim_matches(0, int(Rung.DESIRED), 2, claim_identity)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.lifecycle.evaluating_tiles == frozenset()


def test_rejected_current_reduced_completion_replans_after_releasing_claim(monkeypatch):
    """An obsolete LOD result must leave a producer wakeup for the new demand."""

    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    _settle_first_pixels(session)
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=False,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )
    assert effects.prepare_rung(intent, step)
    monkeypatch.setattr(effects, "_admit_reduced_display_payload", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(effects, "request_presentation", lambda: None)

    effects.apply_commit(
        CommitBatch(
            semantic_key=session.key,
            presentation_key=intent.presentation_key,
            upserts=((step, object()),),
        )
    )

    assert not session.lifecycle.preview_claim_matches(
        0,
        int(Rung.DESIRED),
        2,
        effects._preview_claim_identity(intent, session.plan.tiles[0]),
    )
    assert session._test_replan_requested is True


def test_admitted_preview_completion_replans_unblocked_target_rung(monkeypatch):
    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = RungStep(
        tile_number=0,
        rung=Rung.FLOOR,
        level=2,
        reduce_from_native=False,
        lane=Lane.DISPLAY_PREVIEW,
        priority=Priority.VISIBLE_IMAGE,
        reason="preview rung",
    )
    assert effects.prepare_rung(intent, step)
    monkeypatch.setattr(effects, "_admit_reduced_display_payload", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(effects, "request_presentation", lambda: None)

    effects.apply_commit(
        CommitBatch(
            semantic_key=session.key,
            presentation_key=intent.presentation_key,
            upserts=((step, object()),),
        )
    )

    assert session._test_replan_requested is True


def test_ready_unacknowledged_preview_suppresses_duplicate_floor_evaluation():
    cache = LodPageCache(max_bytes=1 << 20)
    session = _session(count=1, pyramid=cache)
    rendered = session.rendered_tiles.pop(0)
    session.dirty_payloads.clear()
    demand = session.lod_policy_decision.demand
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=int(demand.desired_level) + 2,
        semantic_source_id=semantic_id,
    )
    pages = _materialized_page_set(key, np.asarray(rendered.image))
    assert session.admit_preview_plane(0, key, pages, quality="preview")
    session._ensure_floor_payloads((0,))
    assert session.display_tile_payloads[0].quality == "preview"
    assert 0 in session.pending_payload_upserts
    assert 0 not in session.lifecycle.presented_tiles

    effects = FramePipelineEffects(_RungPrepareRenderer(), session)
    step = RungStep(
        tile_number=0,
        rung=Rung.FLOOR,
        level=int(demand.desired_level) + 2,
        reduce_from_native=False,
        lane=Lane.DISPLAY_PREVIEW,
        priority=Priority.VISIBLE_IMAGE,
        reason="preview rung",
    )

    assert not effects.prepare_rung(_pipeline_intent_for(session), step)
    assert session.lifecycle.row(0).preview_claims == {}


def test_reduced_claim_identity_follows_source_when_scroll_reuses_same_slot_and_frame_key():
    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    _settle_first_pixels(session)
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    effects = FramePipelineEffects(_RungPrepareRenderer(), session)
    intent = _pipeline_intent_for(session)
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=False,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )

    assert effects.prepare_rung(intent, step)
    old_identity = effects._preview_claim_identity(intent, session.plan.tiles[0])

    session.plan = _shifted_plan(count=1, offset=5)
    session.visible_tiles = tuple(session.plan.tiles)
    session.visible_tile_numbers = frozenset({0})
    session.sync_lifecycle_scope()
    _settle_first_pixels(session)
    new_intent = _pipeline_intent_for(session)
    new_identity = effects._preview_claim_identity(new_intent, session.plan.tiles[0])

    assert new_identity != old_identity
    assert effects.prepare_rung(new_intent, step)
    assert session.lifecycle.preview_claim_matches(0, int(Rung.DESIRED), 2, new_identity)


def test_non_commuting_desired_with_retained_source_uses_page_claim():
    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    _settle_first_pixels(session)
    stage_key = ("stage", "retained")
    session.stage_fan_in.tile_stage_keys[0] = stage_key
    session.stage_fan_in.values[stage_key] = object()
    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session, viewport_key="old")
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=True,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )
    del session.rendered_tiles[0]
    session.dirty_payloads.pop(0, None)

    assert effects.prepare_rung(intent, step)
    effects.rung_admitted(intent, step, ("task", "desired"))

    claim_identity = effects._preview_claim_identity(intent, session.plan.tiles[0])
    assert session.lifecycle.preview_claim_matches(0, int(Rung.DESIRED), 2, claim_identity)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.lifecycle.evaluating_tiles == frozenset()


def test_cold_desired_direct_source_produces_page_payload():
    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    effects = FramePipelineEffects(_RungPrepareRenderer(), session)
    tile = session.plan.tiles[0]
    del session.rendered_tiles[0]
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=True,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )

    assert effects._step_produces_page_payload(step, tile)


def test_completed_evaluation_keeps_claim_until_gui_delivery():
    """A worker-complete task is still owned until its callback is drained."""

    task_key = ("task", "desired")
    kernel = _StageProducerKernel(completed=(task_key,))
    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    effects = FramePipelineEffects(_RungPrepareRenderer(kernel=kernel), session)
    del session.rendered_tiles[0]
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=0,
        reduce_from_native=False,
        lane=Lane.DISPLAY_PREVIEW,
        priority=Priority.VISIBLE_IMAGE,
        reason="native desired display level",
    )
    intent = _pipeline_intent_for(session)

    assert effects.prepare_rung(intent, step)
    effects.rung_admitted(intent, step, task_key)
    assert 0 in session.active_tile_requests

    assert effects._release_inactive_evaluation_claims((0,)) == 0
    assert 0 in session.active_tile_requests

    kernel.completed.clear()
    assert effects._release_inactive_evaluation_claims((0,)) == 1
    assert 0 not in session.active_tile_requests


def test_cold_direct_source_page_completion_atomically_promotes_preview():
    """A direct-source reduced target is still a display-page result.

    Regression for the real-Wayland zoom/pan stall: the worker had finished
    and its exact pages were resident, but ``reduce_from_native=True`` caused
    the task to hold a native evaluation claim forever.  The active claim in
    turn suppressed the exact wrapper while the acknowledged preview stayed
    on screen.  Completion must release the page claim, prepare the exact
    successor, and leave the acknowledged fallback intact until that
    successor is physically acknowledged.
    """

    cache = LodPageCache(max_bytes=1 << 20)
    session = _session(count=1, pyramid=cache)
    rendered = session.rendered_tiles.pop(0)
    session.dirty_payloads.clear()
    demand = session.lod_policy_decision.demand
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=int(demand.desired_level),
        semantic_source_id=semantic_id,
    )
    pages = _materialized_page_set(key, np.asarray(rendered.image))

    assert session.admit_preview_plane(0, key, pages, quality="preview")
    session._ensure_floor_payloads((0,))
    _state, preview_delta = session.build_tile_presentation({})
    _acknowledge(session, preview_delta)
    acknowledged_preview = session.tile_presentation_state.payloads[0]
    assert acknowledged_preview.quality == "preview"
    session.pending_payload_upserts.clear()

    renderer = _RungPrepareRenderer()
    renderer._rendered_tile_for_current_payload = lambda *_args, **_kwargs: None
    renderer._admit_first_pass_level_evidence = lambda *_args, **_kwargs: None
    effects = FramePipelineEffects(renderer, session)
    effects.request_presentation = lambda: None
    intent = _pipeline_intent_for(session)
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=int(demand.desired_level),
        reduce_from_native=True,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="direct-source desired display level",
    )

    assert effects.prepare_rung(intent, step)
    effects.rung_admitted(intent, step, ("task", "desired"))
    effects.apply_commit(
        CommitBatch(
            semantic_key=session.key,
            presentation_key=intent.presentation_key,
            upserts=((step, (key, pages, None)),),
        )
    )

    claim_identity = effects._preview_claim_identity(intent, session.plan.tiles[0])
    assert not session.lifecycle.preview_claim_matches(
        0,
        int(Rung.DESIRED),
        int(demand.desired_level),
        claim_identity,
    )
    assert session.lifecycle.evaluating_tiles == frozenset()
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.display_tile_payloads[0].quality == "exact"
    assert 0 in session.pending_payload_upserts
    assert session.tile_presentation_state.payloads[0] is acknowledged_preview


def test_presented_equal_level_fallback_keeps_pixels_without_blocking_exact_target():
    """Drawable fallback and target settlement are separate contracts.

    An equal-LOD fallback remains on screen while exact work runs, but it must
    not be counted as exact coverage and suppress the target's only producer.
    """

    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    tile = session.plan.tiles[0]
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    # A live typed identity: coverage now requires target satisfiability, not
    # just source currency (session-148 follow-up), and every production
    # payload in display_tile_payloads carries its mint identity.
    target_identity = session.lifecycle.peek(0).target.identity
    session.display_tile_payloads[0] = DisplayTilePayload(
        0,
        int(tile.source_index),
        np.ones((4, 4), dtype=np.float32),
        None,
        session.tile_semantic_source_id(tile.source_index),
        lod=LodInfo(level=2, factor=4, source_shape=(TILE, TILE), texture_shape=(4, 4)),
        quality="preview",
        tile_identity=replace(target_identity, quality="fallback"),
    )
    session.lifecycle.acknowledge_presented(
        0,
        session.tile_semantic_source_id(tile.source_index),
        quality="preview",
        level=2,
    )
    effects = FramePipelineEffects(_RungPrepareRenderer(), session)
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=True,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )

    assert effects.prepare_rung(_pipeline_intent_for(session), step)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles


def test_dead_identity_display_payload_does_not_cover_direct_tile_target():
    """Per-tile analog of the session-148 shared-coverage gate (2026-07-16).

    ``_display_payload_covers_display_target`` used to trust payload currency
    (source id + presented) plus an LOD-level compare, without asking whether
    the payload's typed identity can satisfy the tile's current lifecycle
    target.  A presented-but-retargeted payload whose identity is dead under
    the new target is rejected by every backend commit, so counting it as
    coverage denies the tile its only producer: the non-shared pipeline
    starves exactly the way the shared first-pass path did.
    """

    from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind
    from arrayscope.presentation.tile_lifecycle import TileTarget

    def identity(semantic_generation, *, quality="exact", level=2):
        return TileIdentity(
            document_generation=("doc", 0),
            operation_key=("fft",),
            source_index=0,
            image_axes=(1, 0),
            axis_flips=(False, False),
            channel="real",
            complex_mapping=("scalar", "real", "mapped"),
            texture_kind=TexturePlaneKind.SCALAR_R32F,
            semantic_generation=semantic_generation,
            lod=TileLodIdentity(level=level, factor=1 << level),
            quality=quality,
        )

    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    tile = session.plan.tiles[0]
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    source_id = session.tile_semantic_source_id(tile.source_index)
    session.display_tile_payloads[0] = DisplayTilePayload(
        0,
        int(tile.source_index),
        np.ones((4, 4), dtype=np.float32),
        None,
        source_id,
        lod=LodInfo(level=2, factor=4, source_shape=(TILE, TILE), texture_shape=(4, 4)),
        quality="preview",
        tile_identity=identity(("range", (0, 1, 2, 3)), quality="fallback"),
    )
    session.lifecycle.acknowledge_presented(0, source_id, quality="preview", level=2)
    session.lifecycle.retarget(
        {
            0: TileTarget(
                tile_number=0,
                source_index=0,
                semantic_source_id=source_id,
                lod_level=2,
                identity=identity(("range", None)),
            )
        }
    )
    effects = FramePipelineEffects(_RungPrepareRenderer(), session)
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=True,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )

    assert not effects._display_payload_covers_display_target(0, tile, step)
    assert effects.prepare_rung(_pipeline_intent_for(session), step)

    # Control: an exact payload minted under the CURRENT semantics really
    # satisfies the requested target and must keep denying the duplicate
    # per-tile producer.
    session.display_tile_payloads[0] = replace(
        session.display_tile_payloads[0],
        quality="exact",
        tile_identity=identity(("range", None), quality="exact"),
    )

    assert effects._display_payload_covers_display_target(0, tile, step)
    assert not effects.prepare_rung(_pipeline_intent_for(session), step)


def test_shader_preview_evidence_does_not_falsely_settle_exact_target(monkeypatch):
    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    session.shader_display = True
    tile = session.plan.tiles[0]
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    source_id = (
        *session.tile_semantic_source_id(tile.source_index),
        "floor",
        "complex_rg32f",
        (4, 4),
    )
    session.display_tile_payloads[0] = DisplayTilePayload(
        0,
        int(tile.source_index),
        np.ones((4, 4), dtype=np.complex64),
        None,
        source_id,
        level_data=np.asarray([1.0, 4.0], dtype=np.float32),
        quality="preview",
        lod=LodInfo(level=4, factor=16, source_shape=(TILE, TILE), texture_shape=(4, 4)),
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
    )
    session.lifecycle.acknowledge_presented(
        0,
        source_id,
        quality="preview",
        level=4,
    )
    monkeypatch.setattr(
        render_effects, "preview_pipeline_commutes_for_display_lod", lambda _session, _tile: True
    )
    effects = FramePipelineEffects(_RungPrepareRenderer(), session)

    states = effects.tile_states(
        _pipeline_intent_for(session),
        session.lod_policy_decision.demand,
        _pipeline_scope_for(session),
    )

    assert states
    assert session.required_target_unsettled_tiles() == (0,)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles


def test_ready_stage_dependent_rearms_pending_tile_until_materialized():
    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    stage_key = ("stage", "retained")
    del session.rendered_tiles[0]
    session.dirty_payloads.pop(0, None)
    session.stage_fan_in.merge_plan(
        {
            "tile_stage_keys": {0: stage_key},
            "stage_values": {stage_key: object()},
        }
    )
    session.tile_compute_waiting_for_stage = 1
    session.stage_backed_tiles_pending = 1

    assert not session.is_complete()

    assert montage_commit.rearm_ready_stage_dependents(session) == 1

    assert session.stage_fan_in.tile_stage_keys == {}
    assert session.tile_compute_waiting_for_stage == 0
    assert session.stage_backed_tiles_pending == 0
    assert session.required_target_unsettled_tiles() == (0,)
    assert not session.is_complete()


def test_stage_activation_batch_rearms_tiles_after_wait_binding_clears():
    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    stage_key = ("stage", "retained")
    del session.rendered_tiles[0]
    session.dirty_payloads.pop(0, None)
    session.stage_fan_in.merge_plan(
        {
            "tile_stage_keys": {0: stage_key},
            "attached_stage_keys": {stage_key},
        }
    )

    batch = session.stage_fan_in.activate_value(stage_key, object())

    assert batch.tiles == (0,)
    assert session.stage_fan_in.tile_stage_keys == {}
    assert session.required_target_unsettled_tiles() == (0,)
    assert not session.is_complete()


def test_orphaned_stage_dependent_releases_to_direct_tile_work():
    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    stage_key = ("stage", "orphaned")
    del session.rendered_tiles[0]
    session.dirty_payloads.pop(0, None)
    session.stage_fan_in.merge_plan({"tile_stage_keys": {0: stage_key}})

    assert not session.is_complete()

    assert montage_commit.rearm_ready_stage_dependents(session) == 1

    assert session.stage_fan_in.tile_stage_keys == {}
    assert session.required_target_unsettled_tiles() == (0,)
    assert not session.is_complete()


def test_preview_commit_ack_is_actionable_for_target_followup_replan():
    from types import SimpleNamespace

    from arrayscope.display.model.frame import TilePresentationDelta
    from arrayscope.window import frame_effects as montage_commit

    preview = DisplayTilePayload(
        0,
        0,
        np.ones((4, 4), dtype=np.float32),
        None,
        ("preview", 0),
        lod=LodInfo(level=6, factor=64, source_shape=(TILE, TILE), texture_shape=(4, 4)),
        quality="preview",
    )
    exact = DisplayTilePayload(
        1,
        1,
        np.ones((4, 4), dtype=np.float32),
        None,
        ("exact", 1),
        lod=LodInfo(level=2, factor=4, source_shape=(TILE, TILE), texture_shape=(4, 4)),
        quality="exact",
    )
    tile_state = SimpleNamespace(payloads={0: preview, 1: exact})
    delta = TilePresentationDelta(
        0, 0, 0, 0, 0, 0, upserts={0: preview, 1: exact}, active_tiles=(0, 1)
    )
    session = _session(count=2)

    assert montage_commit._commit_report_accepts_new_preview(
        session,
        TileCommitReport(presented_tiles=(0, 1), committed_upserts=(0,)),
        delta,
        tile_state,
    )
    assert not montage_commit._commit_report_accepts_new_preview(
        session,
        TileCommitReport(presented_tiles=(0, 1), committed_upserts=(1,)),
        delta,
        tile_state,
    )
    session.lifecycle.remember_presentable(0, preview)
    session.lifecycle.commit_emitted({0: preview})
    session.lifecycle.acknowledge_presented(0, preview.source_id, "preview", preview.lod.level)
    session.lifecycle.backend_presented_snapshot({0: tile_ack_identity(preview)})
    assert not montage_commit._commit_report_accepts_new_preview(
        session,
        TileCommitReport(presented_tiles=(0, 1), committed_upserts=(0,)),
        delta,
        tile_state,
    )


def test_phase_owner_uses_required_scope_not_retained_active_rows():
    session = _session(count=3, pyramid=LodPageCache(max_bytes=1 << 20))
    session.shader_display = False
    session.visible_tiles = tuple(session.plan.tiles[:2])
    session.visible_tile_numbers = frozenset((0, 1))
    session.sync_lifecycle_scope()
    for tile in session.plan.tiles[:2]:
        source_id = session.tile_semantic_source_id(tile.source_index)
        payload = DisplayTilePayload(
            int(tile.montage_index),
            int(tile.source_index),
            np.ones((8, 8), dtype=np.float32),
            None,
            source_id,
            lod=LodInfo(level=2, factor=4, source_shape=(TILE, TILE), texture_shape=(8, 8)),
            quality="preview",
        )
        session.display_tile_payloads[int(tile.montage_index)] = payload
        session.tile_presentation_state.payloads[int(tile.montage_index)] = payload
    session.lifecycle.backend_presented_snapshot(
        {
            int(tile.montage_index): session.tile_semantic_source_id(tile.source_index)
            for tile in session.plan.tiles[:2]
        }
    )
    session.lifecycle.presentation_confirmed((0, 1))

    effects = FramePipelineEffects(_RungPrepareRenderer(), session)

    assert session.required_first_pixels_presented()
    assert effects.scheduling_verdict().admits_lane(Lane.DISPLAY_PREPARATION)


def test_deferred_stage_completion_does_not_enqueue_native_tiles_for_reduced_lod(monkeypatch):
    from types import SimpleNamespace

    session = _session(count=4, pyramid=LodPageCache(max_bytes=1 << 20))
    assert session.lod_policy_mode == LOD_POLICY_RESIDENT
    assert session.lod_policy_decision.demand.desired_level > 0
    session.stage_planning_deferred = True
    session.stage_planning_async = False
    session.deferred_missing_tiles = tuple(session.plan.tiles)
    session.rendered_tiles.clear()

    monkeypatch.setattr(
        montage_commit,
        "submit_deferred_stage_fan_in_plan",
        lambda _renderer, _session, _missing: False,
    )
    monkeypatch.setattr(
        montage_commit,
        "build_stage_fan_in_plan",
        lambda _renderer, _document, _missing: montage_commit.deferred_stage_fan_in_plan(),
    )

    renderer = SimpleNamespace(
        win=SimpleNamespace(_viewport_interaction_active=False),
        _frame_session_is_current=lambda current: current is session,
        _last_montage_stage_plan_ms=0.0,
        retarget_frame_pipeline=lambda current: setattr(current, "_test_retargeted", True),
    )

    assert montage_commit.complete_deferred_stage_fan_in(renderer, session)
    assert session.deferred_missing_tiles == ()
    assert session.required_target_unsettled_tiles() == (0, 1, 2, 3)
    assert getattr(session, "_test_retargeted", False) is True


def test_deferred_stage_plan_applies_after_unrelated_render_generation_advance(monkeypatch):
    """2026-07-16 churn livelock — member 4 of the deferred-stage lost-wakeup
    family (docs/redesign/stale-empty-tiles-2026-07-16.md). The deferred
    stage-plan completion validated the SESSION's render-generation stamp
    against the renderer's global counter, which advances on every render
    request; after any unrelated repaint the stamp could never match again.
    Every completed plan was discarded and resubmitted (5,200 plan
    computations in one 4-minute churn run) while the pending tiles it owed
    producers to starved forever. Session currency has exactly one owner —
    ``(session_id, key)`` — so a completed plan for the current session must
    APPLY regardless of the global render generation."""

    from types import SimpleNamespace

    session = _session(count=4, pyramid=LodPageCache(max_bytes=1 << 20))
    session.stage_planning_deferred = True
    session.stage_planning_async = False
    session.deferred_missing_tiles = tuple(session.plan.tiles)

    captured = {}

    class _Kernel:
        def submit(self, spec, *, on_done=None, on_stale=None, on_error=None):
            captured["on_done"] = on_done
            return object()

    renderer = SimpleNamespace(
        win=SimpleNamespace(kernel=_Kernel(), _viewport_interaction_active=False),
        _frame_session=session,
        _frame_session_is_current=lambda current: current is session,
        _is_current_frame_session=lambda sid, key: sid == session.session_id and key == session.key,
        # An unrelated repaint advanced the global render generation after
        # this plan was submitted — the exact churn-stall condition.
        _is_current_render_generation=lambda generation: False,
        retarget_frame_pipeline=lambda current: setattr(current, "_test_retargeted", True),
    )

    assert montage_commit.submit_deferred_stage_fan_in_plan(
        renderer, session, tuple(session.plan.tiles)
    )
    assert session.stage_planning_async is True

    monkeypatch.setattr(
        montage_commit,
        "build_stage_fan_in_plan",
        lambda _renderer, _document, _missing, candidate_plan=None: (
            montage_commit.deferred_stage_fan_in_plan()
        ),
    )
    monkeypatch.setattr(montage_commit, "submit_stage_tasks", lambda *_args, **_kwargs: None)

    captured["on_done"]({"candidates": ()})

    assert session.stage_planning_async is False
    assert session.stage_planning_deferred is False, (
        "completed deferred stage plan was discarded on the stale session "
        "render-generation stamp instead of applying (bailed-generation livelock)"
    )
    assert session.deferred_missing_tiles == ()


@pytest.mark.parametrize(("physical_resident", "expected_upserts"), [(True, 16), (False, 1)])
def test_only_physically_resident_payloads_bypass_all_cold_caps_in_one_delta(
    physical_resident,
    expected_upserts,
):
    session = _session(
        mode=LOD_POLICY_NATIVE_ONLY, count=16, pyramid=LodPageCache(max_bytes=1 << 20)
    )
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    session.pending_payload_upserts.clear()
    for tile in session.plan.tiles:
        tile_number = int(tile.montage_index)
        image = np.full((4, 4), float(tile_number), dtype=np.float32)
        payload = DisplayTilePayload(
            tile_number=tile_number,
            source_index=int(tile.source_index),
            image=image,
            histogram_data=None,
            source_id=("preview", tile_number),
            lod=LodInfo(level=4, factor=16, source_shape=(TILE, TILE), texture_shape=image.shape),
            quality="preview",
        )
        session.display_tile_payloads[tile_number] = payload
        session.record_tile_payload(payload)
        session.pending_payload_upserts[tile_number] = None

    _state, delta = session.build_tile_presentation(
        {},
        max_upserts=1,
        max_upsert_bytes=1,
        upsert_cost_fn=lambda payload: payload.nbytes,
        physical_resident_fn=lambda _payload: physical_resident,
    )

    assert len(delta.upserts) == expected_upserts
    if physical_resident:
        assert set(delta.active_tiles) == set(range(16))


def test_physically_resident_rebinds_do_not_share_a_delta_with_cold_uploads():
    """A mixed pan/zoom cohort must expose its resident floor immediately.

    WGPU defers physically cold replacement work while a gesture is active.
    If one cold upload shares the resident rebind delta, that backend guard
    necessarily defers the resident members too and already-loaded tiles pop
    in only after the interaction-stop edge.  The admission owner therefore
    emits the complete free cohort first and leaves every cold member queued
    for the following transaction.
    """

    session = _session(
        mode=LOD_POLICY_RESIDENT,
        count=16,
        pyramid=LodPageCache(max_bytes=1 << 20),
    )
    _admit_zoomed_out_levels(session)
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    session.pending_payload_upserts.clear()
    session.display_tile_payloads.clear()
    session._ensure_floor_payloads(tuple(range(16)))
    assert len(session.display_tile_payloads) == 16
    assert all(
        payload.page_backing is not None for payload in session.display_tile_payloads.values()
    )

    resident_tiles = frozenset(range(12))
    _state, delta = session.build_tile_presentation(
        {},
        max_upserts=1,
        max_upsert_bytes=1 << 20,
        upsert_cost_fn=lambda payload: payload.nbytes,
        physical_resident_fn=lambda payload: int(payload.tile_number) in resident_tiles,
        pace_resident_retargets=False,
    )

    assert set(delta.upserts) == resident_tiles
    assert set(session.pending_payload_upserts).issuperset(set(range(12, 16)))
    _state, paced_delta = session.build_tile_presentation(
        {},
        max_upserts=1,
        max_upsert_bytes=1 << 20,
        upsert_cost_fn=lambda payload: payload.nbytes,
        physical_resident_fn=lambda payload: int(payload.tile_number) in resident_tiles,
        pace_resident_retargets=False,
    )
    assert set(paced_delta.upserts) == resident_tiles
    delta = paced_delta
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))

    _state, cold_delta = session.build_tile_presentation(
        {},
        max_upserts=1,
        max_upsert_bytes=1 << 20,
        upsert_cost_fn=lambda payload: payload.nbytes,
        physical_resident_fn=lambda payload: int(payload.tile_number) in resident_tiles,
        pace_resident_retargets=False,
    )
    assert len(cold_delta.upserts) == 1
    assert set(cold_delta.upserts).isdisjoint(resident_tiles)


def test_visible_parked_payload_rearms_instead_of_settling_missing():
    session = _session(count=4, pyramid=LodPageCache(max_bytes=1 << 20))
    session.build_tile_presentation({})
    session.dirty_payloads.clear()
    session.pending_payload_upserts.clear()
    session.lifecycle.evaluation_completed(2)
    session.lifecycle.upsert_emitted(2, session.display_tile_payloads[2].source_id)
    session.lifecycle.commit_acknowledged(
        emitted_tiles=(2,),
        accepted_tiles=(),
        active_scope=(),
        presented_identities={},
    )

    rearmed = session.rearm_visible_parked_payloads()

    assert rearmed == (2,)
    assert 2 in session.dirty_payloads
    assert 2 not in session.lifecycle.parked_tiles


def test_physically_hidden_presented_payload_rearms_on_scope_expansion():
    session = _session(count=4, pyramid=LodPageCache(max_bytes=1 << 20))
    _state, initial = session.build_tile_presentation({})
    identities = {
        int(tile): tile_ack_identity(payload) for tile, payload in initial.upserts.items()
    }
    report = TileCommitReport(
        presented_tiles=frozenset(initial.upserts),
        committed_upserts=frozenset(initial.upserts),
        presented_identities=identities,
    )
    session.acknowledge_tile_presentation(initial, report)
    session.mark_presented(report.presented_tiles)
    assert session.lifecycle.first_pixels_presented(range(4))

    # A narrow backend visibility commit keeps only slot 0 physically drawn.
    # The semantic payloads remain reusable when the required scope expands.
    session.lifecycle.backend_presented_snapshot({0: identities[0]})
    session.dirty_payloads.clear()
    session.pending_payload_upserts.clear()
    session.sync_lifecycle_scope()
    assert set(session.pending_payload_upserts) == {1, 2, 3}

    _state, repair = session.build_tile_presentation(
        {},
        max_upserts=2,
        pace_resident_retargets=True,
    )

    assert set(repair.upserts).issubset({1, 2, 3})
    assert len(repair.upserts) == 2
    assert set(session.pending_payload_upserts).issuperset({1, 2, 3}.difference(repair.upserts))


def test_source_changed_active_claim_does_not_block_retargeted_prepare():
    """Active request dedupe is source-aware, not montage-slot-only."""

    session = _session(count=1)
    session.rendered_tiles.clear()
    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    old_intent = _pipeline_intent_for(session, viewport_key="old")
    step = _exact_step(0)

    assert effects.prepare_rung(old_intent, step)
    effects.rung_admitted(old_intent, step, ("task", "old"))
    assert 0 in session.active_tile_requests

    new_tile = MontageTile(
        montage_index=0,
        source_index=9,
        row=0,
        col=0,
        x0=0,
        y0=0,
        width=TILE,
        height=TILE,
        view_state=None,
    )
    session.plan = MontagePlan(
        axis=0,
        tile_shape=(TILE, TILE),
        grid_shape=(1, 1),
        columns=1,
        rows=1,
        gap=0,
        tiles=(new_tile,),
    )
    session.key = ("retargeted",)
    new_intent = _pipeline_intent_for(session, viewport_key="new")

    assert effects.prepare_rung(new_intent, step)
    effects.rung_admitted(new_intent, step, ("task", "new"))
    assert 0 in session.active_tile_requests
    assert 0 in session.loading_tiles
    assert session.lifecycle.evaluation_claim_for(0).source_index == 9

    effects.rung_dropped(old_intent, step)
    assert 0 in session.active_tile_requests
    assert session.lifecycle.evaluation_claim_for(0).source_index == 9

    effects.rung_dropped(new_intent, step)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles


def test_stale_materialized_desired_admission_does_not_invent_native_claim():
    """A resident-level result may complete without any semantic eval claim."""

    session = _session(count=1, pyramid=LodPageCache(max_bytes=1 << 20))
    _settle_first_pixels(session)
    renderer = _RungPrepareRenderer()
    effects = FramePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=False,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )
    level_key = replace(
        session._lod_page_set_key_for(
            session.rendered_tiles[0], demand=session.ingest_lod_demand(), level=2
        ),
        source_id=("old-source",),
    )
    request = session._lod_materialization_request(
        session.rendered_tiles[0],
        demand=session.ingest_lod_demand(),
        level=2,
        key=level_key,
    )
    session.pending_rung_materializations.append(request)
    for page_plan in request.claimed_plans:
        page = materialize_lod_page(
            request.source,
            source_origin_yx=request.source_origin_yx,
            plan=page_plan,
        )
        session.lod_page_cache.admit_as(
            page_plan.key,
            page,
            owner=request.owner,
        )
    del session.rendered_tiles[0]
    session.dirty_payloads.pop(0, None)

    assert effects.prepare_rung(intent, step)
    assert 0 not in session.active_tile_requests

    effects._admit_ready_payloads(((step, ("materialized", request)),))

    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    claim_identity = effects._preview_claim_identity(intent, session.plan.tiles[0])
    assert not session.lifecycle.preview_claim_matches(0, int(Rung.DESIRED), 2, claim_identity)
    assert session.lifecycle.evaluating_tiles == frozenset()
    assert session._test_replan_requested is True


def test_stale_rung_drop_does_not_clear_newer_active_claim_for_same_slot():
    """A stale drop may clean up only the intent marker it created."""

    session = _session(count=1)
    session.rendered_tiles.clear()
    effects = FramePipelineEffects(_RungPrepareRenderer(), session)
    old_intent = _pipeline_intent_for(session, viewport_key="old")
    new_intent = _pipeline_intent_for(session, semantic_key=("retargeted",), viewport_key="new")
    step = _exact_step(0)

    assert effects.prepare_rung(old_intent, step)
    effects.rung_admitted(old_intent, step, ("task", "old"))
    session.active_tile_requests.clear()
    session.loading_tiles.clear()
    session.lifecycle.evaluation_declined(0)

    session.key = ("retargeted",)
    assert effects.prepare_rung(new_intent, step)
    effects.rung_admitted(new_intent, step, ("task", "new"))
    effects.rung_dropped(old_intent, step)

    assert 0 in session.active_tile_requests
    assert 0 in session.loading_tiles
    assert session.lifecycle.evaluating_tiles == frozenset({0})

    effects.rung_dropped(new_intent, step)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.lifecycle.evaluating_tiles == frozenset()


def test_unchecked_rgb_reduction_has_no_canonical_page_identity():

    from arrayscope.render import lod as render_lod

    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, view_range=((0.0, 3.0 * 2 * TILE), (0.0, 3.0 * TILE)))
    tile = session.plan.tiles[0]

    # An RGB (phase-hue) base with a separate native magnitude histogram.
    rgb_base = np.zeros((TILE, TILE, 3), dtype=np.uint8)
    rgb_base[..., 0] = 200
    magnitude = np.linspace(0.0, 1000.0, TILE * TILE, dtype=np.float32).reshape(TILE, TILE)
    rendered = RenderedTile(
        tile=tile,
        image=rgb_base,
        histogram_data=magnitude,
        eval_ms=0.0,
        slab_shape=rgb_base.shape,
        slab_nbytes=rgb_base.nbytes,
    )

    demand = session.lod_policy_decision.demand
    applied = int(demand.desired_level)
    assert applied >= 1
    with pytest.raises(ValueError, match="scalar or complex"):
        render_lod.page_set_key_for(session, rendered, demand=demand, level=applied)

    texture, texture_histogram, lod, backing, _kind = render_lod.resident_texture_for_rendered_tile(
        session, rendered, source=rgb_base, histogram=magnitude
    )

    assert lod.level == 0
    assert texture is rgb_base
    assert texture_histogram is magnitude
    assert backing is None


def test_reduced_rgb_resident_without_magnitude_histogram_falls_back_to_native():
    from arrayscope.render import lod as render_lod

    pyramid = LodPageCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, view_range=((0.0, 3.0 * 2 * TILE), (0.0, 3.0 * TILE)))
    tile = session.plan.tiles[0]
    rgb_base = np.zeros((TILE, TILE, 3), dtype=np.uint8)
    rendered = RenderedTile(
        tile=tile,
        image=rgb_base,
        histogram_data=None,
        eval_ms=0.0,
        slab_shape=rgb_base.shape,
        slab_nbytes=rgb_base.nbytes,
    )

    demand = session.lod_policy_decision.demand
    applied = int(demand.desired_level)
    assert applied >= 1
    texture, texture_histogram, lod, backing, _kind = render_lod.resident_texture_for_rendered_tile(
        session, rendered, source=rgb_base, histogram=None
    )

    assert lod.level == 0
    assert texture is rgb_base
    assert texture_histogram is None
    assert backing is None


def test_identical_identity_rejected_delta_recommit_is_bounded(monkeypatch):
    """Session-148 follow-up: bound identical-delta re-commits.

    When a commit's upserts are ALL identity-rejected, the pending upserts
    stay queued and the backlog check re-arms the flush, so the presenter
    rebuilt and re-committed the byte-identical delta at full flush rate
    (~25 ms of wasted geometry sync per cycle in the field trace).  One
    retry is allowed (a retarget can race the commit); an identical repeat
    must emit a loud trace and stop re-arming the flush until either the
    payload or its target identity actually changes.
    """

    from types import SimpleNamespace

    from arrayscope.display.model.frame import TilePresentationDelta
    from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity

    def identity(semantic_generation, *, quality="exact", level=2):
        return TileIdentity(
            document_generation=("doc", 0),
            operation_key=("fft",),
            source_index=0,
            image_axes=(1, 0),
            axis_flips=(False, False),
            channel="real",
            complex_mapping=("scalar", "real", "mapped"),
            texture_kind=TexturePlaneKind.SCALAR_R32F,
            semantic_generation=semantic_generation,
            lod=TileLodIdentity(level=level, factor=1 << level),
            quality=quality,
        )

    traces = []
    monkeypatch.setattr(
        montage_commit, "emit_trace", lambda kind, **fields: traces.append((kind, fields))
    )
    monkeypatch.setattr(
        "arrayscope.window.montage_prefetch.schedule_near_viewport_montage_prefetch",
        lambda _renderer, _session: None,
    )

    class _CommitRenderer(_RungPrepareRenderer):
        def __init__(self):
            super().__init__()
            self.win = SimpleNamespace()

        def _notify_file_session_montage_committed(self):
            pass

        def _settle_montage_visible_plan_if_complete(self, session):
            pass

        def _finish_frame_session_if_complete(self, session):
            pass

        def _retry_live_profile_after_montage_tile(self):
            pass

    session = _session(count=1)
    session.frame_plan = SimpleNamespace(target=None)
    tile = session.plan.tiles[0]
    source_id = session.tile_semantic_source_id(tile.source_index)
    target_identity = identity(("range", None))
    dead_payload = DisplayTilePayload(
        0,
        0,
        np.ones((16, 16), dtype=np.float32),
        None,
        source_id,
        lod=LodInfo(level=2, factor=4, source_shape=(TILE, TILE), texture_shape=(16, 16)),
        quality="preview",
        tile_identity=identity(("range", (0, 1, 2, 3)), quality="fallback"),
    )

    def delta_for(payload):
        return TilePresentationDelta(
            structure_revision=1,
            payload_revision=1,
            visibility_revision=1,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=1,
            upserts={0: payload},
            active_tiles=(0,),
            target_identities={0: target_identity},
        )

    rejected_report = TileCommitReport(
        presented_tiles=frozenset(),
        committed_upserts=frozenset(),
        identity_rejected_tiles=frozenset({0}),
    )
    tile_state = SimpleNamespace(revision=1)
    effects = FramePipelineEffects(_CommitRenderer(), session)

    def run_commit(payload, report):
        # The queued upsert the backlog check inspects: exactly the rejected
        # tile, re-queued because the backend acknowledged nothing.
        session.dirty_payloads = {0: None}
        session.pending_payload_upserts = {0: None}
        session.flush_pending = False
        session.final_commit_pending = False
        traces.clear()
        effects._finish_commit(
            report,
            tile_state,
            delta_for(payload),
            commit_start=0.0,
            preview_transition=False,
        )

    # First all-rejected commit: one retry is allowed, the flush re-arms.
    run_commit(dead_payload, rejected_report)
    assert session.flush_pending
    assert not any(kind == "identity_rejected_recommit" for kind, _fields in traces)

    # Identical repeat: loud trace, and the rejected tiles alone must not
    # re-arm the full-rate flush again.
    run_commit(dead_payload, rejected_report)
    assert not session.flush_pending
    recommits = [fields for kind, fields in traces if kind == "identity_rejected_recommit"]
    assert recommits
    assert recommits[0]["tiles"] == (0,)

    # A changed payload identity is a new delta and commits normally again.
    replaced = replace(dead_payload, tile_identity=identity(("range", None), quality="fallback"))
    run_commit(replaced, rejected_report)
    assert session.flush_pending

    # A commit that accepts its upserts clears the backoff state entirely.
    accepted_report = TileCommitReport(
        presented_tiles=frozenset({0}),
        committed_upserts=frozenset({0}),
    )
    run_commit(replaced, accepted_report)
    assert session.flush_pending
    run_commit(dead_payload, rejected_report)
    assert session.flush_pending


def test_admitted_wrapper_keeps_ladder_residency_during_atomic_wait():
    """2026-07-24 completion-drain freeze pin (field diagnostics
    ``arrayscope-diagnostics-20260724-145640.jsonl`` seq 172-181).

    During an atomic successor handoff no tile can be backend-acknowledged
    until the whole transaction swaps.  The 9e83dbe3 residency guard keyed
    only on ack-gated predicates, so it cleared ladder-visible residency for
    every admitted-but-unacknowledged wrapper and the ladder re-planned FLOOR
    for every tile on every replan gate - a self-sustaining evaluation loop
    that froze a 100-tile cropped scrub for 5-25 s.  An admitted
    current-source wrapper the shader backend can commit must keep physical
    residency visible so no duplicate FLOOR producer is planned.
    """

    pyramid = LodPageCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    # The field shape is the wgpu shader backend, where preview planes are
    # first-class presentation currency for the coverage pass.
    session.shader_display = True
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[1]
    key = page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    _admit_page_set(pyramid, key, np.asarray(rendered.image))
    _claim_preview_resident(session, 1, key)
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)

    _state, _delta = session.build_tile_presentation({})
    # NO acknowledgement: the atomic successor holds the whole transaction,
    # exactly the field state (pending upserts held, nothing presented).
    session.atomic_successor_pending = True
    assert session.display_tile_payloads[1].quality == "preview"

    states = render_effects.tile_lod_states(session, demand)
    state = next(item for item in states if int(item.tile_number) == 1)
    assert state.resident_levels, (
        "an admitted current-source wrapper awaiting the atomic swap must keep "
        "physical residency visible to the ladder"
    )

    policy = LadderPolicy(
        mode=session.lod_policy_mode,
        floor_level=4,
        reduced_input_available=True,
    )
    steps = LodLadder(policy).plan(states, demand)
    assert not any(int(step.tile_number) == 1 and step.rung == Rung.FLOOR for step in steps), (
        "no duplicate FLOOR producer may be planned while the wrapper awaits the swap"
    )


def test_shared_transform_scheduler_is_retired():
    session = _session(count=2, mode=LOD_POLICY_RESIDENT, pyramid=LodPageCache(max_bytes=1 << 20))
    effects = FramePipelineEffects(_RungPrepareRenderer(), session)

    assert not hasattr(effects, "submit_shared_transform_floor")


def test_montage_axis_fft_with_known_display_axes_reaches_the_coarse_ladder():
    """A cacheable shared reduced stage makes the ordinary ladder admissible."""

    from arrayscope.core.view_state import ViewState
    from arrayscope.render import effects as render_effects
    from arrayscope.window.frame_runtime import _reduced_input_coarse_rung_available

    data = np.ones((TILE, TILE, 8), dtype=np.float32)
    view_state = ViewState.from_shape(data.shape).with_image_axes(0, 1)
    session = _session(count=2, mode=LOD_POLICY_RESIDENT, pyramid=LodPageCache(max_bytes=1 << 20))
    session.plan = replace(
        session.plan,
        axis=2,
        tiles=tuple(replace(tile, view_state=view_state) for tile in session.plan.tiles),
    )
    session.montage_axis = 2
    session.document = ArrayDocument(data, operations=(CenteredFFT(axis=2),))
    seed = session.plan.tiles[0]

    assert render_effects.preview_pipeline_commutes_for_display_lod(session, seed) is True
    assert render_effects.preview_pipeline_is_tile_local(session, seed) is True

    session.rgb = True
    session.shader_display = True
    assert _reduced_input_coarse_rung_available(session, seed) is True

    session.shader_display = False
    assert render_effects.can_evaluate_reduced_preview(session, seed) is False
    assert _reduced_input_coarse_rung_available(session, seed) is False

    session.rgb = False
    assert _reduced_input_coarse_rung_available(session, seed) is True
