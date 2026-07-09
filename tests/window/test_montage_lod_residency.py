"""Qt-free contract tests for resident-LOD montage sessions (ADR 0050)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from arrayscope.display.lod import LOD_POLICY_NATIVE_ONLY, LOD_POLICY_RESIDENT, LodInfo, select_lod_demand
from arrayscope.display.model.frame import DisplayTilePayload, TileCommitReport
from arrayscope.display.montage import MontagePlan, MontageTile, RenderedTile
from arrayscope.display.pyramid import PyramidCache, PyramidLevelKey, reduce_box_mean
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.kernel import Lane, Priority
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT
from arrayscope.presentation import ClaimOwner, LevelPhase, Presentation
from arrayscope.render import effects as render_effects
from arrayscope.render import lod as render_lod
from arrayscope.render.ladder import LadderPolicy, LodLadder, Rung, RungStep
from arrayscope.window import montage_commit
from arrayscope.window.montage_commit import MontagePipelineEffects
from arrayscope.render.lod import admit_retained_preview_level, histogram_key_for_level_key
from arrayscope.render.stages import CommitBatch, LodAdmissionScope, RenderIntent
from arrayscope.window.montage_session import (
    MontageRenderSession,
    pyramid_key_for_rendered,
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
    session = MontageRenderSession(
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
        pending_tiles=[],
        lod_policy_mode=mode,
        pyramid_cache=pyramid,
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


def _materialize(session, request):
    """Run one request the way the worker does: walk the chain, admit each step."""

    if hasattr(session.pending_rung_materializations, "mark_started"):
        session.pending_rung_materializations.mark_started(request)
    plane = request.source
    admitted = None
    steps = tuple(getattr(request, "chain", ()) or ()) or ((request.key, request.reduce_factor_xy),)
    for step_key, rel in steps:
        plane = reduce_box_mean(plane, rel)
        if step_key is not None:
            plane = session.pyramid_cache.admit(step_key, plane)
            if step_key == request.key:
                admitted = plane
    if hasattr(session.pending_rung_materializations, "mark_resident"):
        session.pending_rung_materializations.mark_resident(request)
    return request.key, admitted


def _release(session, request):
    """Drop one request the way every non-run scheduling path must: all claims."""

    from arrayscope.render.lod import _apply_release_effects

    if hasattr(session.pending_rung_materializations, "release"):
        _apply_release_effects(session.pyramid_cache, session.pending_rung_materializations.release(request))


def _claim_preview_resident(session, tile_number: int, key) -> None:
    session.lifecycle.level_claimed(int(tile_number), key, ClaimOwner.PREVIEW, request=("test-preview", key))
    session.lifecycle.level_resident(int(tile_number), key)


def _admit_demand_level_for_test(pyramid, demand, rendered, *, semantic_source_id):
    level = int(demand.desired_level)
    if pyramid is None or level <= 0:
        return None
    key = pyramid_key_for_rendered(rendered, demand=demand, level=level, semantic_source_id=semantic_source_id)
    if not pyramid.begin_pending(key):
        return None
    try:
        source, histogram, _kind = texture_source_for_rendered(rendered)
        reduced = pyramid.admit(key, reduce_box_mean(source, key.factor_xy))
        if histogram is not None:
            hist_key = histogram_key_for_level_key(key)
            if pyramid.begin_pending(hist_key):
                try:
                    pyramid.admit(hist_key, reduce_box_mean(histogram, hist_key.factor_xy))
                except Exception:
                    pyramid.end_pending(hist_key)
        return reduced
    except Exception:
        pyramid.end_pending(key)
        raise


class _RungPrepareRenderer:
    def __init__(self, *, kernel=None) -> None:
        self.kernel = kernel

    def _montage_session_is_current(self, session) -> bool:
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
    )


def _pipeline_scope_for(session):
    return LodAdmissionScope(
        visible_tile_numbers=tuple(int(tile.montage_index) for tile in tuple(session.visible_tiles)),
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


def _plan_rung_materializations(session) -> tuple:
    demand = session.lod_policy_decision.demand
    policy = LadderPolicy(
        mode=session.lod_policy_mode,
        floor_level=max(1, int(getattr(session, "lod_preview_level", 0) or 4)),
        preview_level=max(1, int(getattr(session, "lod_preview_level", 0) or 2)),
        reduced_input_available=True,
    )
    effects = MontagePipelineEffects(_RungPrepareRenderer(), session)
    for step in LodLadder(policy).plan(render_effects.tile_lod_states(session, demand), demand):
        if step.rung == Rung.DESIRED:
            effects.prepare_rung(None, step)
    return tuple(session.lifecycle.active_materializations())


def test_native_only_mode_is_unchanged_by_default():
    session = _session(mode=LOD_POLICY_NATIVE_ONLY, pyramid=None)
    state, delta = session.build_tile_presentation({})

    assert session.lod_policy_mode == LOD_POLICY_NATIVE_ONLY
    assert session.lod_policy_decision.policy == "native-only"
    assert session.lod_policy_decision.applied_factor == 1
    assert session.pending_rung_materializations == []
    for payload in delta.upserts.values():
        assert payload.lod.level == 0
        assert payload.texture_data.shape[:2] == (TILE, TILE)


def test_visible_replacement_retains_presented_payload_until_acknowledged():
    session = _session(count=1)
    source_ids = {0: session.tile_semantic_source_id(0)}
    state, delta = session.build_tile_presentation(source_ids)
    old_payload = delta.upserts[0]
    report = TileCommitReport(
        presented_tiles=frozenset({0}),
        committed_upserts=frozenset({0}),
        delta_key=(delta.base_revision, delta.target_revision),
        presented_identities={0: old_payload.source_id},
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
    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), view_range=far_zoom)
    session.lod_preview_min_level = 4
    session.lod_preview_level = 4

    session._selected_lod_factor()

    assert session.lod_policy_decision.demand.desired_level >= 5
    assert session.lod_preview_level == session.lod_policy_decision.demand.desired_level


def test_resident_mode_falls_back_to_native_and_records_missing_levels():
    session = _session(pyramid=PyramidCache(max_bytes=1 << 20))
    state, delta = session.build_tile_presentation({})
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
    tiles = sorted(request[0] for request in requests)
    assert tiles == [0, 1]
    for request in requests:
        assert isinstance(request.key, PyramidLevelKey)
        assert request.key.factor_xy == (4, 4)
        assert request.reduce_factor_xy == (4, 4)
        assert request.source.shape == (TILE, TILE)
    assert len(session.lifecycle.dangling_claims()) == 4


def test_duplicate_materialization_requests_coalesce():
    session = _session(pyramid=PyramidCache(max_bytes=1 << 20))
    session.build_tile_presentation({})
    first = list(_plan_rung_materializations(session))

    # A second commit while requests are pending must not re-claim them.
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})

    assert list(_plan_rung_materializations(session)) == first
    assert len(first) == 2


def test_materialization_chains_through_the_missing_finer_level():
    """ADR 0050 level-chaining: one request materializes the whole ladder.

    Desired level 2 with nothing resident plans a chain that admits the
    acceptable finer level 1 on the way — each step reduces the previous
    plane (¼ of its source texels per doubling) — so the hysteresis
    fallback is already resident when the zoom crosses it later.
    """

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    assert len(requests) == 2
    for request in requests:
        chain = tuple(request.chain)
        assert [step_key.level_xy for step_key, _rel in chain] == [(1, 1), (2, 2)]
        assert [rel for _key, rel in chain] == [(2, 2), (2, 2)]
        assert chain[-1][0] == request.key
        assert request.cross_level is False
        _materialize(session, request)

    # Both levels are resident per tile and every claim is balanced.
    assert len(pyramid) == 4
    assert pyramid.pending_count == 0
    assert session.lod_chain_planned == 2

    # Zoom in one hysteresis step: level 1 is a cache hit, no new work.
    session.view_range = ((0.0, 3.0 * 2 * TILE), (0.0, 3.0 * TILE))
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})
    assert session.lod_policy_decision.applied_level == 1
    assert session.pending_rung_materializations == []


def test_chain_derives_from_the_resident_finer_level_source():
    """A resident finer level becomes the chain source, not the native plane."""

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, view_range=((0.0, 3.0 * 2 * TILE), (0.0, 3.0 * TILE)))
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    for request in requests:
        assert tuple(step_key.level_xy for step_key, _rel in request.chain) == ((1, 1),)
        _materialize(session, request)
    assert len(pyramid) == 2

    # Zoom out to level-2 demand (past the hysteresis band of the applied
    # level-1 factor): the request derives from resident level 1.
    session.view_range = ((0.0, 5.0 * 2 * TILE), (0.0, 5.0 * TILE))
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    assert len(requests) == 2
    for request in requests:
        assert request.cross_level is True
        assert request.source.shape[:2] == (TILE // 2, TILE // 2)
        assert request.reduce_factor_xy == (2, 2)
        assert tuple(step_key.level_xy for step_key, _rel in request.chain) == ((2, 2),)
        _materialize(session, request)
    assert session.lod_cross_level_reductions == 2
    assert len(pyramid) == 4
    assert pyramid.pending_count == 0


def test_chain_passes_through_an_intermediate_claimed_elsewhere():
    """A level claimed by another producer is reduced through, never admitted."""

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=1)
    rendered = session.rendered_tiles[0]
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE), previous_factor=1)
    foreign = pyramid_key_for_rendered(
        rendered, demand=demand, level=1, semantic_source_id=session.tile_semantic_source_id(0)
    )
    assert pyramid.begin_pending(foreign)

    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    assert len(requests) == 1
    chain = tuple(requests[0].chain)
    # Pass-through step: reduce through level 1 without admitting it.
    assert [(None if step_key is None else step_key.level_xy) for step_key, _rel in chain] == [None, (2, 2)]
    _materialize(session, requests[0])

    assert len(pyramid) == 1
    # The foreign claim is untouched: it belongs to its producer.
    assert pyramid.pending_count == 1
    pyramid.end_pending(foreign)


def test_admitted_level_streams_in_with_distinct_identity_and_shape():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    state, delta = session.build_tile_presentation({})
    native_ids = {tile: payload.source_id for tile, payload in delta.upserts.items()}

    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    for request in requests:
        _materialize(session, request)

    session.dirty_payloads.update({0: None, 1: None})
    state, delta = session.build_tile_presentation({})

    assert session.lod_policy_decision.applied_level == 1
    assert set(delta.upserts) == {0, 1}
    for tile, payload in delta.upserts.items():
        assert payload.lod.level == 1
        assert payload.lod.factor == 2
        assert payload.texture_data.shape[:2] == (TILE // 2, TILE // 2)
        # Presentation identity separates levels: a reduced payload can never
        # share a residency key with the native payload it replaces.
        assert payload.source_id != native_ids[tile]
        assert payload.image.shape[:2] == (TILE // 2, TILE // 2)
        assert payload.semantic_data.shape[:2] == (TILE, TILE)
        # Exact semantic sources are untouched by display LOD.
        assert payload.semantic_histogram_data.shape[:2] == (TILE, TILE)
    assert session.pending_rung_materializations == []


def test_mixed_residency_applies_per_tile_and_reports_common_level():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    # Materialize the demanded level for tile 0 only.
    for request in requests:
        if request[0] == 0:
            _materialize(session, request)
        else:
            _release(session, request)

    session.dirty_payloads.update({0: None, 1: None})
    state, delta = session.build_tile_presentation({})

    # Tile 0 presents the reduced level; tile 1 retains native until its
    # replacement is resident.
    assert delta.upserts[0].lod.level == 1
    assert delta.upserts[1].lod.level == 0
    assert delta.upserts[0].texture_data.shape[:2] == (TILE // 2, TILE // 2)
    assert delta.upserts[1].texture_data.shape[:2] == (TILE, TILE)
    assert delta.upserts[0].source_id != delta.upserts[1].source_id
    # The session-wide decision only claims what every tile can present.
    assert session.lod_policy_decision.applied_level == 0


def test_threshold_recrossing_hits_the_pyramid_cache_without_new_requests():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    for request in requests:
        _materialize(session, request)
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})
    assert session.lod_policy_decision.applied_level == 1

    # Zoom in to native...
    session.view_range = ((0.0, float(2 * TILE)), (0.0, float(TILE)))
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})
    assert session.lod_policy_decision.applied_level == 0

    # ...and back out: the level comes from the cache, no new materialization.
    hits_before = pyramid.hits
    session.view_range = ZOOMED_OUT_RANGE
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})

    assert session.lod_policy_decision.applied_level == 1
    assert session.pending_rung_materializations == []
    assert pyramid.hits > hits_before
    assert pyramid.pending_count == 0


def test_no_reduction_work_happens_inside_presentation_builds(monkeypatch):
    import arrayscope.display.pyramid as pyramid_module

    def _fail(*_args, **_kwargs):
        raise AssertionError("reduce_box_mean must never run in a presentation build")

    pyramid = PyramidCache(max_bytes=1 << 20)
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
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _cold_session(pyramid=pyramid)
    demand = session.ingest_lod_demand()
    assert demand is not None
    assert demand.desired_level == 2

    # Worker side: the native tile is computed, then reduced and admitted as
    # part of the same materialization, before the result reaches the GUI.
    rendered = _rendered(session.plan.tiles[0])
    assert _admit_demand_level_for_test(pyramid, demand, rendered, semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index)) is not None
    assert len(pyramid) == 2
    assert pyramid.pending_count == 0
    # Singleflight: the level is resident, a second admission is a no-op.
    assert _admit_demand_level_for_test(pyramid, demand, rendered, semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index)) is None

    # GUI side: the first presentation build selects the reduced level.  No
    # native payload is ever emitted for the tile and nothing is re-requested.
    session.mark_materialized(rendered)
    _state, delta = session.build_tile_presentation({})
    payload = delta.upserts[0]
    assert payload.lod.level == 2
    assert payload.lod.factor == 4
    assert payload.texture_data.shape[:2] == (TILE // 4, TILE // 4)
    assert payload.image.shape[:2] == (TILE // 4, TILE // 4)
    assert payload.histogram_data.shape[:2] == (TILE // 4, TILE // 4)
    # Exact semantic sources stay native.
    assert payload.semantic_data.shape[:2] == (TILE, TILE)
    assert payload.semantic_histogram_data.shape[:2] == (TILE, TILE)
    assert session.pending_rung_materializations == []


def test_native_only_and_native_scale_sessions_have_no_ingest_demand():
    assert _session(mode=LOD_POLICY_NATIVE_ONLY, pyramid=None).ingest_lod_demand() is None
    zoomed_in = _session(
        pyramid=PyramidCache(max_bytes=1 << 20),
        view_range=((0.0, float(2 * TILE)), (0.0, float(TILE))),
    )
    assert zoomed_in.ingest_lod_demand() is None


def test_demand_flip_during_inflight_ingest_falls_back_to_streaming():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _cold_session(pyramid=pyramid)
    demand = session.ingest_lod_demand()
    assert demand.desired_level == 2

    # The viewport changes while the tile is in flight: level 1 is now wanted
    # (three source texels per screen pixel).
    session.view_range = ((0.0, 3.0 * 2 * TILE), (0.0, 3.0 * TILE))

    # The worker still completes against its scheduling-time snapshot.
    rendered = _rendered(session.plan.tiles[0])
    assert _admit_demand_level_for_test(pyramid, demand, rendered, semantic_source_id=("test-tile", rendered.tile.source_index)) is not None

    # No special cases: presentation never over-reduces with the stale level;
    # it falls back and the ordinary streaming path materializes level 1.
    session.mark_materialized(rendered)
    _state, delta = session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    assert session.lod_policy_decision.demand.desired_level == 1
    assert delta.upserts[0].lod.level == 0
    assert len(requests) == 1
    request = requests[0]
    assert request.tile_number == 0
    assert request.key.factor_xy == (2, 2)

    _materialize(session, request)
    session.dirty_payloads[0] = None
    _state, delta = session.build_tile_presentation({})
    assert delta.upserts[0].lod.level == 1
    assert delta.upserts[0].texture_data.shape[:2] == (TILE // 2, TILE // 2)


def _acknowledge(session, delta):
    session.acknowledge_tile_presentation(
        delta,
        TileCommitReport(
            presented_tiles=frozenset(int(tile) for tile in delta.upserts),
            committed_upserts=frozenset(int(tile) for tile in delta.upserts),
        ),
    )


def test_presented_lod_summary_reports_plurality_of_presented_payloads():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=3)

    # Nothing committed yet: fall back to the session-wide decision (native).
    assert session.presented_lod_summary() == (0, 1, (1, 1))

    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    for request in requests:
        if request[0] in (0, 1):
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
    assert session.presented_lod_summary() == (1, 2, (2, 2))


def test_presented_lod_summary_tie_prefers_the_finer_level():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=2)
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    for request in requests:
        if request[0] == 0:
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
        key = pyramid_key_for_rendered(rendered, demand=demand, level=level, semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index))
        session.pyramid_cache.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))


def test_camera_only_retarget_keeps_presented_finer_level_without_swap_churn():
    """Zoom over already-correct finer payloads must not churn presentation."""

    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    assert all(payload.lod.level == 0 for payload in session.tile_presentation_state.payloads.values())
    _admit_zoomed_out_levels(session)

    # Camera-only zoom out: retarget alone, no pan, no dimension scroll, no
    # tile results.  Native pixels are already current and finer than the
    # demand; cached coarser levels must not force wrapper/atlas swaps.
    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    revision_before = int(session.viewport_revision)
    swap_ready = session.mark_ladder_swaps_for_viewport()

    assert swap_ready is False
    assert session.pending_rung_materializations == [], "cached levels must not be re-requested"
    assert not session.dirty_payloads

    hits_before = pyramid.hits
    _state, delta = session.build_tile_presentation({})
    assert delta.upserts == {}
    assert delta.removals == ()
    assert pyramid.hits == hits_before

    # A second refresh with the same viewport is a no-op (no revision creep,
    # no commit request, no dirty tiles).
    assert session.mark_ladder_swaps_for_viewport() is False
    assert int(session.viewport_revision) >= revision_before


def test_camera_only_retarget_demotes_finer_level_under_residency_pressure():
    """Visible quality demotion is a memory-pressure decision, not zoom churn."""

    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    _admit_zoomed_out_levels(session)
    native_bytes = sum(
        int(np.asarray(payload.texture_data).nbytes)
        for payload in session.tile_presentation_state.payloads.values()
    )
    session.tile_residency_budget_bytes = native_bytes - 1

    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    assert session.mark_ladder_swaps_for_viewport() is True
    assert sorted(session.dirty_payloads) == [0, 1]

    _state, delta = session.build_tile_presentation({})
    assert set(delta.upserts) == {0, 1}
    assert delta.removals == ()
    assert {payload.lod.level for payload in delta.upserts.values()} == {2}
    assert all(
        payload.texture_data.shape[:2] == (TILE // 4, TILE // 4)
        for payload in delta.upserts.values()
    )


def test_camera_only_retarget_requests_missing_levels_with_new_lod_revision():
    pyramid = PyramidCache(max_bytes=1 << 24)
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
    assert all(payload.lod.level == 0 for payload in session.tile_presentation_state.payloads.values())

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


def test_level_only_change_replaces_payload_atomically_and_deferral_keeps_old():
    """ADR 0050 contract: a LOD swap never blanks or placeholders a tile.

    A commit that changes only the presented level of an acknowledged tile
    must emit an upsert (never a removal), and when the budget defers the
    upsert the previously presented payload stays committed unchanged.
    """

    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    native_payloads = dict(session.tile_presentation_state.payloads)
    assert set(native_payloads) == {0, 1}
    _admit_zoomed_out_levels(session)
    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    session.dirty_payloads.update({0: None, 1: None})

    # Budget admits only one swap this commit.
    state, delta = session.build_tile_presentation({}, max_upserts=1)
    assert delta.removals == ()
    assert len(delta.upserts) == 1
    swapped = next(iter(delta.upserts))
    deferred = ({0, 1} - {swapped}).pop()
    # At the commit boundary both tiles are mapped: one at the new level, the
    # deferred one still at its old (acknowledged) payload.
    assert state.payloads[swapped].lod.level == 2
    assert state.payloads[deferred] is native_payloads[deferred]
    # Neither tile regressed to a loading/placeholder presentation state.
    states = session.ensure_tile_states()
    assert str(states[0].value) == "loaded"
    assert str(states[1].value) == "loaded"

    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    # The deferred tile stays dirty and swaps on the next commit, again with
    # no removal and no gap.
    state, delta = session.build_tile_presentation({})
    assert delta.removals == ()
    assert set(delta.upserts) == {deferred}
    assert state.payloads[deferred].lod.level == 2
    assert state.payloads[swapped].lod.level == 2


def test_previous_payloads_keep_visible_tiles_active_when_replacement_not_ready():
    """Retained visible pixels are active presentation, not a blank gap."""

    pyramid = PyramidCache(max_bytes=1 << 24)
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

    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_OUT_RANGE)
    _admit_zoomed_out_levels(session)
    _state, delta = session.build_tile_presentation({})
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    previous_payloads = dict(session.tile_presentation_state.payloads)
    assert {payload.lod.level for payload in previous_payloads.values()} == {2}

    # Fresh session for the same content and viewport; the pyramid cache was
    # dropped in between (worst case for seeding).
    replacement = _session(pyramid=PyramidCache(max_bytes=1 << 24), view_range=ZOOMED_OUT_RANGE)
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
    assert partial.display_tile_payloads[0].source_id == retained.tile_presentation_state.payloads[2].source_id
    assert partial.display_tile_payloads[1].source_id == retained.tile_presentation_state.payloads[3].source_id
    assert set(partial.pending_payload_upserts) == {0, 1}

# --- Zero redundant histogram/level work across LOD levels (ADR 0050) ---

LEVEL1_RANGE = ((0.0, 3.0 * 2 * TILE), (0.0, 3.0 * TILE))
# From a presented factor-2 demand, promotion hysteresis needs > 4.6 source
# texels per screen pixel before level 2 becomes desired.
FAR_OUT_RANGE = ((0.0, 6.0 * 2 * TILE), (0.0, 6.0 * TILE))


def _attach_native_stats(session):
    from arrayscope.display.model.montage_levels import sample_tile_level_stats
    from dataclasses import replace as dc_replace

    for index, rendered in dict(session.rendered_tiles).items():
        stats = sample_tile_level_stats(rendered.image, int(rendered.tile.source_index), refined=True)
        session.rendered_tiles[index] = dc_replace(rendered, level_data=rendered.image, level_stats=stats)


def test_level_swap_carries_native_stats_and_recomputes_nothing():
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _attach_native_stats(session)
    _present_native(session)
    assert session.lod_stats_recomputes == 0

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
        tiled_semantic_histogram_identity as _tiled_semantic_histogram_identity,
    )

    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    native_payloads = dict(session.tile_presentation_state.payloads)

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
    assert _tiled_semantic_histogram_identity(swapped_payloads) == _tiled_semantic_histogram_identity(native_payloads)
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


def test_coarser_level_derives_from_finest_resident_level():
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=LEVEL1_RANGE)
    session.build_tile_presentation({})
    level1_requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    assert {request.key.level for request in level1_requests} == {1}
    for request in level1_requests:
        assert request.cross_level is False
        _materialize(session, request)

    session.retarget_viewport(view_range=FAR_OUT_RANGE, viewport_shape=VIEWPORT)
    session.mark_ladder_swaps_for_viewport()
    level2_requests = list(_plan_rung_materializations(session))
    session.pending_rung_materializations.clear()
    assert {request.key.level for request in level2_requests} == {2}
    for request in level2_requests:
        # Derived level-from-level: the reduction source is the resident
        # level-1 array and only its texels are touched, not the native plane.
        assert request.cross_level is True
        assert request.source.shape[:2] == (TILE // 2, TILE // 2)
        assert request.reduce_factor_xy == (2, 2)
        assert request.key.factor_xy == (4, 4)
        derived = reduce_box_mean(request.source, request.reduce_factor_xy)
        rendered = session.rendered_tiles[int(request.tile_number)]
        native = reduce_box_mean(np.asarray(rendered.image), (4, 4))
        assert np.allclose(derived, native, rtol=1e-6, atol=1e-6)
    assert session.lod_cross_level_reductions == len(level2_requests) > 0


def test_uneven_tiles_fall_back_to_native_reduction_source():
    # 63 is not divisible by 4: partial trailing boxes must always reduce
    # from the single canonical native plane so level content never depends
    # on which levels happened to be resident.
    tile = 63
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=LEVEL1_RANGE)
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
        assert request.cross_level is False
        assert request.source.shape[:2] == (tile, tile)
        assert request.reduce_factor_xy == request.key.factor_xy


def test_floor_presents_resident_level_for_unrendered_tile_instead_of_placeholder():
    """ADR 0050 floor invariant: any resident level beats a placeholder."""

    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))

    # Tile 1 was materialized at level 2 in an earlier pass (semantic key),
    # then its rendered object was dropped — e.g. a pan re-entered the tile.
    rendered = session.rendered_tiles[1]
    key = pyramid_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)

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
    _state, delta3 = session.build_tile_presentation({})
    replaced = session.display_tile_payloads[1]
    assert replaced.quality == "exact"


def test_floors_survive_index_window_changes_via_semantic_key():
    """Field defect 2026-07-05 (missing corner tiles 'there in other views'):
    the pyramid identity was keyed by the session key, which includes the
    sibling-index selection — every index-window change renamed identical
    texels and refilled previously computed tiles cold from black.  Sessions
    sharing a window-agnostic ``semantic_key`` must share floors."""

    pyramid = PyramidCache(max_bytes=1 << 24)
    semantic = ("texels", "doc", 2)

    session_a = _session(pyramid=pyramid, count=2)
    session_a.semantic_key = semantic
    session_a.key = ("window", "4:104")
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session_a.rendered_tiles[1]
    key = pyramid_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=session_a.tile_semantic_source_id(rendered.tile.source_index),
    )
    pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))

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
    assert payload is not None and payload.quality == "preview"


def test_backend_reported_identities_drive_convergence():
    """ADR 0051 rule 1, ground-truth edition (field defects 2026-07-05): the
    session's own acknowledgement records lied in every stale-LOD wedge, so
    convergence now compares against the identities the backend reports its
    drawn slots ACTUALLY hold.  A mismatch re-presents the tile inside the
    active scope; agreement settles; retries are bounded."""

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    current_identity = {
        int(tile): payload.source_id for tile, payload in dict(delta.upserts).items()
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

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
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

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    current_identity = {
        int(tile): payload.source_id for tile, payload in dict(delta.upserts).items()
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

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    # Fresh session, fresh machine: only the inherited map knows slot 1 is
    # stale.  (The renderer copies this dict across replacement.)
    backend_truth = {
        int(tile): payload.source_id for tile, payload in dict(delta.upserts).items()
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

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    wanted = dict(delta.upserts)[1].source_id
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

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    assert 0 in session.dirty_payloads and 1 in session.dirty_payloads

    stale_report = TileCommitReport(
        presented_tiles=(0, 1),
        delta_key=(int(delta.base_revision) + 5, int(delta.target_revision) + 5),
    )
    acknowledged = session.acknowledge_tile_presentation(delta, stale_report)
    assert dict(acknowledged.payloads) == {}, "mismatched report must acknowledge nothing"
    assert 0 in session.dirty_payloads and 1 in session.dirty_payloads, "dirty stays armed"
    assert session.parked_dirty_payloads == frozenset(), "and nothing parks"

    bound = TileCommitReport(
        presented_tiles=(0, 1),
        committed_upserts=(0, 1),
        delta_key=(int(delta.base_revision), int(delta.target_revision)),
    )
    acknowledged2 = session.acknowledge_tile_presentation(delta, bound)
    assert set(acknowledged2.payloads) == {0, 1}
    assert not session.dirty_payloads


def test_mark_presented_rejects_backend_active_tile_with_stale_identity():
    """A backend-visible slot is not current presentation unless its identity
    matches the session payload.  PyQtGraph can keep old items visible during
    a bounded retarget; those tiles must stay dirty instead of making the
    session appear settled."""

    from dataclasses import replace as dc_replace

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    report = TileCommitReport(presented_tiles=(0, 1), committed_upserts=(0, 1))
    session.acknowledge_tile_presentation(delta, report)
    session.mark_presented((0, 1))
    assert session.lifecycle.presented_tiles == frozenset({0, 1})

    old_identity = session.display_tile_payloads[1].source_id
    session.display_tile_payloads[1] = dc_replace(session.display_tile_payloads[1], source_id=("fresh", 1))
    session.dirty_payloads.clear()
    session.lifecycle.backend_presented_snapshot({
        0: session.display_tile_payloads[0].source_id,
        1: old_identity,
    })

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

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
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

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
    assert session.mark_ladder_swaps_for_viewport() is not None
    first = list(_plan_rung_materializations(session))
    assert first, "zoomed-out demand plans materializations for the missing level"

    # Simulate supersession/session churn dropping the planned work.
    released = render_lod.release_session_claims(session)
    assert released == sum(1 for request in first for step_key, _rel in request.chain if step_key is not None)
    assert not session.pending_rung_materializations

    session.mark_ladder_swaps_for_viewport()
    replanned = list(_plan_rung_materializations(session))
    assert replanned, (
        "idle refresh must re-plan the demanded level after its claims were released"
    )


def test_floor_presents_blank_tile_even_while_exact_evaluation_is_in_flight():
    """Field report 2026-07-05: slow stage-backed fills left tiles BLACK for
    seconds while their floor planes sat resident — because the floor pass
    skipped every tile with an active exact request.  A blank tile floors
    regardless of in-flight work; only an existing preview defers to the
    imminent exact replacement (anti-churn)."""

    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))

    rendered = session.rendered_tiles[1]
    key = pyramid_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)
    # The exact evaluation is in flight (slow stage compute).
    session.active_tile_requests.add(1)

    assert session._floor_can_progress(1)
    _state, delta = session.build_tile_presentation({})
    payload = delta.upserts.get(1) or session.display_tile_payloads.get(1)
    assert payload is not None, "blank tile with resident floor must present it despite in-flight eval"
    assert payload.quality == "preview"
    assert payload.lod.level == 2

    # Anti-churn: once the preview is on screen, the in-flight exact request
    # suppresses further floor improvements for this tile.
    assert not session._floor_can_progress(1)


def test_floor_payload_construction_honors_batch_cap():
    pyramid = PyramidCache(max_bytes=1 << 24)
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

    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2, view_range=((0.0, 2 * TILE), (0.0, TILE)))
    zoomed_out = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[1]
    key = pyramid_key_for_rendered(
        rendered,
        demand=zoomed_out,
        level=2,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)

    # Demand is native (zoomed in); the tile floors at level 2.
    _state, delta = session.build_tile_presentation({})
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    floored = session.display_tile_payloads.get(1)
    assert floored is not None and floored.quality == "preview"

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
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[1]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    coarse = pyramid_key_for_rendered(rendered, demand=demand, level=4, semantic_source_id=semantic_id)
    pyramid.admit(coarse, reduce_box_mean(np.asarray(rendered.image), coarse.factor_xy))
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)

    _state, delta = session.build_tile_presentation({})
    _acknowledge(session, delta)
    session.mark_presented(tuple(delta.upserts))
    assert session.display_tile_payloads[1].lod.level == 4

    # The demanded level 2 materializes later; the floor upgrades to it.
    better = PyramidLevelKey(
        source_id=semantic_id,
        tile_id=coarse.tile_id,
        component=coarse.component,
        level_xy=(2, 2),
    )
    image = np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE) + 1
    pyramid.admit(better, reduce_box_mean(image, better.factor_xy))
    assert session._floor_can_progress(1)
    _state, delta = session.build_tile_presentation({})
    upgraded = session.display_tile_payloads[1]
    assert upgraded.quality == "preview"
    assert upgraded.lod.level == 2


def test_lod_refresh_owns_its_supersession_counter_not_viewport_revision():
    """Regression: refresh bumped viewport_revision without replanning,
    churning priority-retarget work identities at idle."""

    pyramid = PyramidCache(max_bytes=1 << 24)
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

    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[1]
    key = pyramid_key_for_rendered(
        rendered,
        demand=demand,
        level=2,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
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


def test_orphaned_loading_tiles_are_requeued_for_evaluation():
    """Regression: declined admission left tiles loading forever with no
    pending work, so the visible plan never completed and finalization
    retried commits in a timer loop at idle."""

    session = _session(pyramid=PyramidCache(max_bytes=1 << 20))
    # Simulate a lost evaluation: dequeued, marked loading, work vanished.
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)
    session.loading_tiles.add(1)
    assert not session.pending_tiles

    requeued = session.requeue_orphaned_loading_tiles()
    assert requeued == 1
    assert 1 not in session.loading_tiles
    assert [int(t.montage_index) for t in session.pending_tiles] == [1]

    # Idempotent: a second repair finds nothing.
    assert session.requeue_orphaned_loading_tiles() == 0


def test_parked_dirty_tiles_rearm_when_the_viewport_makes_them_active():
    """Parking non-active dirty tiles must be non-destructive: a build racing
    a retarget may hold a stale active set, and the tile must still present
    when it becomes active (user-visible as pan-revealed tiles that rendered
    but never appeared)."""

    session = _session(pyramid=PyramidCache(max_bytes=1 << 20), count=2)
    # Tile 1 rendered and dirty, but currently outside the active scope.
    session.visible_tiles = (session.plan.tiles[0],)
    _state, delta = session.build_tile_presentation({})
    # Contract: the upsert is still emitted once (persistent backends
    # accept it), but if a viewport-scoped backend declines it, the tile
    # parks at acknowledgement instead of retrying forever.
    assert 1 in delta.upserts
    report = TileCommitReport(presented_tiles=(0,))
    session.acknowledge_tile_presentation(delta, report)
    assert 1 in session.parked_dirty_payloads
    assert 1 not in session.dirty_payloads

    # Idle stays settled: further builds re-emit nothing for the parked tile.
    _state, delta2 = session.build_tile_presentation({})
    assert 1 not in delta2.upserts
    assert 1 not in session.dirty_payloads

    # The viewport brings tile 1 back: the parked entry re-arms and the
    # payload presents through the ordinary delta.
    session.visible_tiles = tuple(session.plan.tiles)
    _state, delta3 = session.build_tile_presentation({})
    assert 1 not in session.parked_dirty_payloads
    assert 1 in delta3.upserts


def test_preview_reduction_fills_pinned_cache_from_native_and_from_reduced():
    """ADR 0050 retained preview level: every evaluated tile leaves a coarse
    copy in the pinned cache; when an ingest reduction exists with evenly
    dividing shape, the preview derives from it instead of native."""

    preview = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=PyramidCache(max_bytes=1 << 20))
    rendered = session.rendered_tiles[0]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)

    assert admit_retained_preview_level(
        preview, rendered, semantic_source_id=semantic_id, preview_level=3
    )
    key = next(iter(preview.resident_keys_for(semantic_id, rendered.tile.source_index, "scalar")))
    assert key.level_xy == (3, 3)
    assert preview.peek(key).shape == (TILE // 8, TILE // 8)
    expected = np.asarray(rendered.image).reshape(TILE // 8, 8, TILE // 8, 8).mean(axis=(1, 3))
    assert np.allclose(preview.peek(key), expected, atol=1e-4)

    # Derive-from-reduced: level 1 plane divides evenly into level 3.
    preview2 = PyramidCache(max_bytes=1 << 20)
    reduced = reduce_box_mean(np.asarray(rendered.image), (2, 2))
    assert admit_retained_preview_level(
        preview2, rendered, semantic_source_id=semantic_id, preview_level=3,
        reduced=reduced, reduced_level=1,
    )
    key2 = next(iter(preview2.resident_keys_for(semantic_id, rendered.tile.source_index, "scalar")))
    assert np.allclose(preview2.peek(key2), expected, atol=1e-4)

    # Singleflight: second admission is a no-op.
    assert not admit_retained_preview_level(
        preview, rendered, semantic_source_id=semantic_id, preview_level=3
    )


def test_floor_presents_from_pinned_preview_when_main_pyramid_lost_the_level():
    """Scroll-back contract: main-cache churn can never blank a tile that was
    ever computed — the pinned preview level floors it."""

    main = PyramidCache(max_bytes=1 << 20)
    preview = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=main, count=2)
    session.pyramid_cache = preview
    session.lod_preview_level = 3

    rendered = session.rendered_tiles[1]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    assert admit_retained_preview_level(
        preview, rendered, semantic_source_id=semantic_id, preview_level=3
    )
    # Tile 1 loses its rendered result and has nothing in the MAIN pyramid.
    del session.rendered_tiles[1]
    session.dirty_payloads.pop(1, None)

    _state, delta = session.build_tile_presentation({})
    payload = delta.upserts.get(1) or session.display_tile_payloads.get(1)
    assert payload is not None, "preview level must floor the tile"
    assert payload.quality == "preview"
    assert payload.lod.level == 3
    assert payload.texture_data.shape[:2] == (TILE // 8, TILE // 8)


def test_rgb_floor_from_pinned_preview_carries_display_histogram():
    """RGB preview floors need the reduced display histogram so PyQtGraph can
    re-window existing tiles on level changes before exact content arrives."""

    main = PyramidCache(max_bytes=1 << 20)
    preview = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=main, count=1)
    session.pyramid_cache = preview
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
    assert admit_retained_preview_level(
        preview,
        rendered,
        semantic_source_id=semantic_id,
        preview_level=2,
        shader_display=False,
    )
    del session.rendered_tiles[0]
    session.display_tile_payloads.clear()
    session.dirty_payloads.clear()

    _state, delta = session.build_tile_presentation({})
    payload = delta.upserts.get(0) or session.display_tile_payloads.get(0)
    assert payload is not None
    assert payload.quality == "preview"
    assert payload.texture_kind == "rgb8"
    assert payload.texture_data.shape == (TILE // 4, TILE // 4, 3)
    assert payload.histogram_data is not None
    assert payload.histogram_data.shape == (TILE // 4, TILE // 4)
    assert payload.semantic_data is None
    assert payload.semantic_histogram_data is None


def test_rgb_preview_without_display_histogram_is_not_floor_presented():
    pyramid = PyramidCache(max_bytes=1 << 20)
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

    assert not admit_retained_preview_level(
        pyramid,
        rendered,
        semantic_source_id=semantic_id,
        preview_level=2,
        shader_display=False,
    )
    del session.rendered_tiles[0]
    session.display_tile_payloads.clear()
    session.dirty_payloads.clear()

    _state, delta = session.build_tile_presentation({})

    assert not delta.upserts
    assert not session.display_tile_payloads


def test_preview_payload_at_acceptable_level_still_refines_to_exact():
    """Regression (screenshot: blocky tiles among exact neighbors): a
    preview payload whose level falls inside acceptable_levels looked
    converged to refresh and never refined, though the rendered result
    was available for a cheap exact rebuild."""

    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[1]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = pyramid_key_for_rendered(rendered, demand=demand, level=2, semantic_source_id=semantic_id)
    pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))

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
        tile=session.plan.tiles[1], image=image, histogram_data=image,
        eval_ms=0.0, slab_shape=image.shape, slab_nbytes=image.nbytes,
    )

    # Camera refresh must dirty the preview tile even though its level (2)
    # is inside acceptable_levels, and the next build must go exact.
    session.mark_ladder_swaps_for_viewport()
    assert 1 in session.dirty_payloads
    _state, _delta = session.build_tile_presentation({})
    assert session.display_tile_payloads[1].quality == "exact"


def test_shared_preview_floor_is_presented_before_exact_refinement():
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = pyramid_key_for_rendered(rendered, demand=demand, level=2, semantic_source_id=semantic_id)
        pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
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


def test_presented_preview_keeps_active_exact_work_loading():
    """A first-pixel preview ack must not clear the exact work owed state."""

    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[0]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = pyramid_key_for_rendered(rendered, demand=demand, level=2, semantic_source_id=semantic_id)
    pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
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
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = pyramid_key_for_rendered(rendered, demand=demand, level=2, semantic_source_id=semantic_id)
        pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
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
    effects = MontagePipelineEffects(_RungPrepareRenderer(kernel=_StageProducerKernel(stage_key)), session)
    step = _exact_step(0)

    assert effects.rung_deps(_pipeline_intent_for(session), step) == (stage_key,)


def test_stage_backed_rung_has_no_dependency_without_live_stage_producer():
    session = _session(count=1)
    stage_key = ("stage", "orphan")
    session.stage_fan_in.tile_stage_keys[0] = stage_key
    session.stage_fan_in.active_requests.add(stage_key)
    effects = MontagePipelineEffects(_RungPrepareRenderer(), session)
    step = _exact_step(0)

    assert effects.rung_deps(_pipeline_intent_for(session), step) == ()


def test_stage_backed_rung_admission_records_live_stage_producer():
    session = _session(count=1)
    session.rendered_tiles.clear()
    stage_key = ("stage", "in-flight")
    session.stage_fan_in.tile_stage_keys[0] = stage_key
    renderer = _RungPrepareRenderer(kernel=_StageProducerKernel(stage_key))
    effects = MontagePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _exact_step(0)

    effects.rung_admitted(intent, step, ("task", 0))

    row = session.tile_ledger.row(0)
    assert row.task_claim.task_key == ("task", 0)
    assert row.stage_key == stage_key
    assert row.stage_producer_key == stage_key


def test_exact_rung_enters_running_only_after_kernel_admission():
    session = _session(count=1)
    session.rendered_tiles.clear()
    session.dirty_payloads.pop(0, None)
    renderer = _RungPrepareRenderer()
    effects = MontagePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _exact_step(0)

    assert effects.prepare_rung(intent, step)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles

    effects.rung_admitted(intent, step, ("task", 0))

    assert 0 in session.active_tile_requests
    assert 0 in session.loading_tiles
    assert session.tile_ledger.row(0).task_claim.task_key == ("task", 0)


def test_prepare_rung_releases_active_claim_when_task_is_not_live():
    session = _session(count=1)
    session.rendered_tiles.clear()
    session.dirty_payloads.pop(0, None)
    renderer = _RungPrepareRenderer(kernel=_StageProducerKernel())
    effects = MontagePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _exact_step(0)

    effects.rung_admitted(intent, step, ("task", "missing"))

    assert 0 in session.active_tile_requests
    assert session.tile_ledger.row(0).task_claim.task_key == ("task", "missing")

    assert effects.prepare_rung(intent, step)

    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.tile_ledger.row(0).task_claim is None


def test_preview_drop_does_not_clear_target_task_claim():
    session = _session(count=1)
    session.rendered_tiles.clear()
    session.dirty_payloads.pop(0, None)
    renderer = _RungPrepareRenderer(kernel=_StageProducerKernel(("task", "target")))
    effects = MontagePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    target_step = _exact_step(0)
    preview_step = RungStep(
        tile_number=0,
        rung=Rung.PREVIEW,
        level=4,
        reduce_from_native=False,
        lane=Lane.DISPLAY_PREVIEW,
        priority=Priority.VISIBLE_IMAGE,
        reason="preview",
    )

    effects.rung_admitted(intent, target_step, ("task", "target"))
    session.lifecycle.preview_claimed(0, int(Rung.PREVIEW), 4)

    effects.rung_dropped(intent, preview_step)

    assert not session.lifecycle.preview_claim_matches(0, int(Rung.PREVIEW), 4)
    assert session.tile_ledger.row(0).task_claim.task_key == ("task", "target")
    assert 0 in session.active_tile_requests


def test_tile_state_snapshot_releases_active_claim_without_live_task():
    session = _session(count=1)
    session.rendered_tiles.clear()
    session.dirty_payloads.pop(0, None)
    renderer = _RungPrepareRenderer(kernel=_StageProducerKernel())
    effects = MontagePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _exact_step(0)

    effects.rung_admitted(intent, step, ("task", "gone"))
    session.tile_ledger.task_released(0, reason="dropped")

    states = effects.tile_states(intent, session.lod_policy_decision.demand, _pipeline_scope_for(session))

    assert tuple(state.tile_number for state in states) == (0,)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.lifecycle.evaluating_tiles == frozenset()


def test_tile_state_snapshot_keeps_live_active_claim_out_of_ladder():
    session = _session(count=1)
    session.rendered_tiles.clear()
    session.dirty_payloads.pop(0, None)
    renderer = _RungPrepareRenderer(kernel=_StageProducerKernel(("task", "live")))
    effects = MontagePipelineEffects(renderer, session)
    intent = _pipeline_intent_for(session)
    step = _exact_step(0)

    effects.rung_admitted(intent, step, ("task", "live"))

    states = effects.tile_states(intent, session.lod_policy_decision.demand, _pipeline_scope_for(session))

    assert states == ()
    assert 0 in session.active_tile_requests
    assert 0 in session.loading_tiles


def test_preview_floor_commit_activates_every_planned_preview_tile_before_exact():
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=4)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = pyramid_key_for_rendered(rendered, demand=demand, level=2, semantic_source_id=semantic_id)
        pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
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
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=4)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = pyramid_key_for_rendered(rendered, demand=demand, level=2, semantic_source_id=semantic_id)
        pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
        _claim_preview_resident(session, rendered.tile.montage_index, key)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=1)

    admitted = set(preview_delta.upserts)
    assert len(admitted) == 1
    assert set(preview_delta.active_tiles) == {0, 1, 2, 3}
    admitted_tile = next(iter(admitted))
    assert preview_delta.upserts[admitted_tile].quality == "preview"
    assert set(session.pending_payload_upserts) == {0, 1, 2, 3}


def test_preview_floor_scope_defers_exact_until_scoped_tiles_are_covered():
    pyramid = PyramidCache(max_bytes=1 << 24)
    preview = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=4)
    session.pyramid_cache = preview
    session.lod_preview_level = 4
    session.mark_preview_floor_scope(range(4))
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[0]
    key = pyramid_key_for_rendered(
        rendered,
        demand=demand,
        level=4,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
    )
    preview.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
    _claim_preview_resident(session, 0, key)

    _state, delta = session.build_tile_presentation({}, max_upserts=4)

    assert set(delta.upserts) == {0}
    assert delta.upserts[0].quality == "preview"
    assert all(payload.quality != "exact" for payload in delta.upserts.values())
    assert set(session.dirty_payloads) == {0, 1, 2, 3}


def test_acknowledged_preview_with_exact_result_rearms_exact_refinement():
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = pyramid_key_for_rendered(rendered, demand=demand, level=2, semantic_source_id=semantic_id)
        pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
        _claim_preview_resident(session, rendered.tile.montage_index, key)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=2)
    _acknowledge(session, preview_delta)
    session.mark_presented(tuple(preview_delta.upserts))

    assert {payload.quality for payload in preview_delta.upserts.values()} == {"preview"}
    assert set(session.dirty_payloads) == {0, 1}
    _state, exact_delta = session.build_tile_presentation({}, max_upserts=2)
    assert {payload.quality for payload in exact_delta.upserts.values()} == {"exact"}


def test_backend_confirmed_preview_does_not_settle_when_exact_exists():
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=2)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = pyramid_key_for_rendered(rendered, demand=demand, level=2, semantic_source_id=semantic_id)
        pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
        _claim_preview_resident(session, rendered.tile.montage_index, key)

    _state, preview_delta = session.build_tile_presentation({}, max_upserts=2)
    _acknowledge(session, preview_delta)
    session.mark_presented(tuple(preview_delta.upserts))
    preview_identities = {
        int(tile): payload.source_id
        for tile, payload in preview_delta.upserts.items()
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


def test_vispy_shader_floor_ignores_retained_rgb_complex_preview():
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    session.shader_display = True
    tile = session.plan.tiles[0]
    semantic_id = session.tile_semantic_source_id(tile.source_index)
    rgb_key = PyramidLevelKey(
        source_id=semantic_id,
        tile_id=int(tile.source_index),
        component=str(TexturePlaneKind.RGB8.value),
        level_xy=(2, 2),
    )
    complex_key = PyramidLevelKey(
        source_id=semantic_id,
        tile_id=int(tile.source_index),
        component=str(TexturePlaneKind.COMPLEX_RG32F.value),
        level_xy=(2, 2),
    )
    pyramid.admit(rgb_key, np.zeros((TILE // 4, TILE // 4, 3), dtype=np.uint8))
    complex_plane = np.ones((TILE // 4, TILE // 4), dtype=np.complex64)
    pyramid.admit(complex_key, complex_plane)
    pyramid.admit(histogram_key_for_level_key(complex_key), np.abs(complex_plane).astype(np.float32))

    best = render_lod.best_floor_key(session, int(tile.source_index), tile_number=0)

    assert best is not None
    assert best[0] == complex_key


def test_vispy_shader_preview_admission_rejects_rgb_floor_payload():
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    session.shader_display = True
    tile = session.plan.tiles[0]
    semantic_id = session.tile_semantic_source_id(tile.source_index)
    rgb_key = PyramidLevelKey(
        source_id=semantic_id,
        tile_id=int(tile.source_index),
        component=str(TexturePlaneKind.RGB8.value),
        level_xy=(2, 2),
    )

    admitted = session.admit_preview_plane(
        0,
        rgb_key,
        np.zeros((TILE // 4, TILE // 4, 3), dtype=np.uint8),
        texture_kind=TexturePlaneKind.RGB8,
    )

    assert admitted is False
    assert pyramid.peek(rgb_key) is None


def test_acknowledged_preview_floor_stays_active_until_exact_refinement():
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=4)
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in tuple(session.rendered_tiles.values()):
        semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
        key = pyramid_key_for_rendered(rendered, demand=demand, level=2, semantic_source_id=semantic_id)
        pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))
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
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    rendered = session.rendered_tiles[0]
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = pyramid_key_for_rendered(rendered, demand=demand, level=2, semantic_source_id=semantic_id)
    assert pyramid.begin_pending(key)
    session.lifecycle.level_claimed(0, key, ClaimOwner.PREVIEW, request=("test-preview", key))
    session.lifecycle.level_materializing(0, key)

    _state, blocked_delta = session.build_tile_presentation({}, max_upserts=1)

    assert blocked_delta.upserts == {}
    session.release_preview_claim(0, key)
    assert session.lifecycle.dangling_claims() == ()

    _state, exact_delta = session.build_tile_presentation({}, max_upserts=1)

    assert exact_delta.upserts[0].quality == "exact"


def test_replaced_session_releases_preview_floor_claims_from_preview_cache():
    from arrayscope.render.lod import release_session_claims

    pyramid = PyramidCache(max_bytes=1 << 24)
    preview = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    session.pyramid_cache = preview
    session.lod_preview_level = 4
    rendered = session.rendered_tiles[0]
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    key = pyramid_key_for_rendered(rendered, demand=demand, level=4, semantic_source_id=semantic_id)
    assert preview.begin_pending(key)
    session.lifecycle.level_claimed(0, key, ClaimOwner.PREVIEW, request=("test-preview", key))
    session.lifecycle.level_materializing(0, key)
    session.lod_preview_metadata[key] = object()

    assert release_session_claims(session) == 1

    assert session.lifecycle.dangling_claims() == ()
    assert key not in session.lod_preview_metadata
    assert preview.begin_pending(key)
    preview.end_pending(key)


def test_preview_floor_target_prefers_preview_cache_over_requested_level():
    pyramid = PyramidCache(max_bytes=1 << 24)
    preview = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, count=1)
    session.pyramid_cache = preview
    session.lod_preview_level = 4
    rendered = session.rendered_tiles[0]
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    requested_key = pyramid_key_for_rendered(rendered, demand=demand, level=2, semantic_source_id=semantic_id)
    preview_key = pyramid_key_for_rendered(rendered, demand=demand, level=4, semantic_source_id=semantic_id)
    pyramid.admit(requested_key, reduce_box_mean(np.asarray(rendered.image), requested_key.factor_xy))
    preview.admit(preview_key, reduce_box_mean(np.asarray(rendered.image), preview_key.factor_xy))
    _claim_preview_resident(session, 0, preview_key)

    del session.rendered_tiles[0]
    session.dirty_payloads.clear()
    _state, delta = session.build_tile_presentation({}, max_upserts=1)

    assert delta.upserts[0].quality == "preview"
    assert delta.upserts[0].lod.level == 4
    assert delta.upserts[0].texture_data.shape[:2] == (TILE // 16, TILE // 16)


def test_lod_debug_pass_marker_mirrors_final_payload_only(monkeypatch):
    monkeypatch.setenv("ARRAYSCOPE_LOD_DEBUG_PASS_MARKER", "final-mirror-x")
    pyramid = PyramidCache(max_bytes=1 << 24)
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
    key = pyramid_key_for_rendered(session.rendered_tiles[0], demand=demand, level=2, semantic_source_id=semantic_id)
    preview = reduce_box_mean(image, key.factor_xy)
    pyramid.admit(key, preview)
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

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    assert len(requests) == 2
    assert pyramid.pending_count > 0

    released = release_session_claims(session)

    assert released == 4
    assert session.pending_rung_materializations == []
    assert pyramid.pending_count == 0
    # The same slice revisited (equal session key) can claim its levels again.
    replacement = _session(pyramid=pyramid)
    replacement.build_tile_presentation({})
    replacement_requests = list(_plan_rung_materializations(replacement))
    assert len(replacement_requests) == 2


def test_pending_lod_request_view_clear_releases_pyramid_claims():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    session.build_tile_presentation({})
    requests = list(_plan_rung_materializations(session))
    assert len(requests) == 2
    assert pyramid.pending_count == 4

    from arrayscope.render.lod import release_session_claims

    assert release_session_claims(session) == 4
    assert pyramid.pending_count == 0
    assert session.lifecycle.dangling_claims() == ()


def test_diagnostics_lod_reason_follows_the_presented_level():
    """The reason text must describe the screen, not the last policy run."""

    from arrayscope.display.lod import (
        LOD_REASON_RESIDENT_MATCH,
        LOD_REASON_RESIDENT_NATIVE_FALLBACK,
    )
    from arrayscope.window.diagnostics_snapshot import _presented_lod_reason

    pyramid = PyramidCache(max_bytes=1 << 20)
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


def _retarget(session, plan, new_source_ids, cached_tiles=None):
    return session.retarget_index_window(
        session_id=session.session_id + 1,
        key=("test", "retargeted"),
        semantic_key=("semantic", "retargeted"),
        level_key=("level", "retargeted"),
        render_generation=session.render_generation + 1,
        view_state=None,
        plan=plan,
        frame_plan=session.frame_plan,
        all_indices=tuple(int(t.source_index) for t in plan.tiles),
        new_source_ids=new_source_ids,
        cached_tiles=dict(cached_tiles or {}),
        visible_tiles=tuple(plan.tiles),
    )


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
        montage_index=1, source_index=9, row=0, col=1,
        x0=TILE, y0=0, width=TILE, height=TILE, view_state=None,
    )
    plan = MontagePlan(
        axis=0, tile_shape=(TILE, TILE), grid_shape=(1, 2),
        columns=2, rows=1, gap=0, tiles=tuple(plan_tiles),
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

    assert stats["hits"] == 1 and stats["misses"] == 0 and stats["unchanged"] == 1
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


def test_retarget_index_window_demotes_misses_with_immediate_invalidation():
    """A miss must remove stale pixels; only current-source payloads may show."""

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
    session.acknowledge_tile_presentation(delta, TileCommitReport(presented_tiles=state.active_payloads(delta)))
    session.mark_presented(state.active_payloads(delta))

    plan = _shifted_plan(count=2, offset=5)

    stats = _retarget(
        session,
        plan,
        new_source_ids={0: ("src", 5), 1: ("src", 6)},
        cached_tiles={},
    )

    assert stats["misses"] == 2 and stats["hits"] == 0
    for index in (0, 1):
        assert index not in session.rendered_tiles
        assert index not in session.loading_tiles
        assert index in session.dirty_payloads
        assert index not in session.tile_source_ids
    # Lifecycle semantic axis demoted (no longer evaluated).
    assert not session.lifecycle.evaluating_tiles

    _state, replacement_delta = session.build_tile_presentation({0: ("src", 5), 1: ("src", 6)})
    assert replacement_delta.removals == (0, 1)
    assert not session.visible_plan_complete()


def test_stale_backend_identity_is_not_acknowledged_when_correct_upsert_is_budgeted_out():
    session = _session(count=2)
    old_sources = {0: ("src", 0), 1: ("src", 1)}
    old_state, old_delta = session.build_tile_presentation(old_sources)
    session.acknowledge_tile_presentation(
        old_delta,
        TileCommitReport(
            presented_tiles=frozenset(old_delta.upserts),
            committed_upserts=frozenset(old_delta.upserts),
            presented_identities={
                int(tile): payload.source_id
                for tile, payload in old_delta.upserts.items()
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
    plan = _shifted_plan(count=2, offset=3)

    stats = _retarget(
        session,
        plan,
        new_source_ids={0: ("src", 3), 1: ("src", 4)},
        cached_tiles={},
    )

    assert stats["misses"] == 2
    assert not session.active_tile_requests


def test_stale_rung_drop_releases_its_admitted_active_claim_after_retarget():
    """Cleanup is owned by the admitted rung, not by the live semantic key."""

    session = _session(count=1)
    session.rendered_tiles.clear()
    renderer = _RungPrepareRenderer()
    effects = MontagePipelineEffects(renderer, session)
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


def test_stale_commit_batch_releases_admitted_active_claim_after_key_retarget():
    """A completion between session key mutation and pipeline replan must drop."""

    session = _session(count=1)
    session.rendered_tiles.clear()
    renderer = _RungPrepareRenderer()
    effects = MontagePipelineEffects(renderer, session)
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

    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    session.lod_preview_level = 6
    renderer = _RungPrepareRenderer()
    effects = MontagePipelineEffects(renderer, session)
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
    level_key = PyramidLevelKey(("old-source",), ("tile", 0), "texture", (2, 2))
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
    assert session.lifecycle.preview_claim_matches(0, int(Rung.DESIRED), 2)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.pending_rung_materializations

    effects.rung_dropped(intent, step)

    assert not session.pending_rung_materializations
    assert not session.lifecycle.preview_claim_matches(0, int(Rung.DESIRED), 2)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert session.lifecycle.evaluating_tiles == frozenset()


def test_non_commuting_desired_with_retained_source_uses_evaluation_claim():
    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    stage_key = ("stage", "retained")
    session.stage_fan_in.tile_stage_keys[0] = stage_key
    session.stage_fan_in.values[stage_key] = object()
    renderer = _RungPrepareRenderer()
    effects = MontagePipelineEffects(renderer, session)
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

    assert not session.lifecycle.preview_claim_matches(0, int(Rung.DESIRED), 2)
    assert 0 in session.active_tile_requests
    assert 0 in session.loading_tiles


def test_non_commuting_cold_desired_is_not_reduced_display_payload():
    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    effects = MontagePipelineEffects(_RungPrepareRenderer(), session)
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

    assert not effects._step_evaluates_reduced_display_payload(step, tile)


def test_shared_target_in_flight_blocks_per_tile_desired_after_coarse_preview():
    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    tile = session.plan.tiles[0]
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    session.display_tile_payloads[0] = DisplayTilePayload(
        0,
        0,
        np.ones((4, 4), dtype=np.float32),
        None,
        session.tile_semantic_source_id(tile.source_index),
        lod=LodInfo(level=6, factor=64, source_shape=(TILE, TILE), texture_shape=(4, 4)),
        quality="preview",
    )
    stage_key = ("stage", "retained")
    session.stage_fan_in.tile_stage_keys[0] = stage_key
    session.stage_fan_in.values[stage_key] = object()
    session.lifecycle.preview_claimed(0, int(Rung.PREVIEW), 2)
    effects = MontagePipelineEffects(_RungPrepareRenderer(), session)
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=False,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )

    assert not effects.prepare_rung(_pipeline_intent_for(session), step)

    assert session.lifecycle.preview_claim_matches(0, int(Rung.PREVIEW), 2)
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles


def test_shared_desired_claim_blocks_direct_tile_target():
    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    tile = session.plan.tiles[0]
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    session.lifecycle.preview_claimed(0, int(Rung.DESIRED), 2)
    session.enqueue_pending_tile(tile)
    effects = MontagePipelineEffects(_RungPrepareRenderer(), session)
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=True,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )

    assert not effects.prepare_rung(_pipeline_intent_for(session), step)
    assert not session.pending_tiles
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles


def test_presented_shared_desired_payload_blocks_direct_tile_target():
    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    tile = session.plan.tiles[0]
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    session.enqueue_pending_tile(tile)
    session.display_tile_payloads[0] = DisplayTilePayload(
        0,
        int(tile.source_index),
        np.ones((4, 4), dtype=np.float32),
        None,
        session.tile_semantic_source_id(tile.source_index),
        lod=LodInfo(level=2, factor=4, source_shape=(TILE, TILE), texture_shape=(4, 4)),
        quality="preview",
    )
    session.lifecycle.acknowledge_presented(
        0,
        session.tile_semantic_source_id(tile.source_index),
        quality="preview",
        level=2,
    )
    effects = MontagePipelineEffects(_RungPrepareRenderer(), session)
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=True,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )

    assert not effects.prepare_rung(_pipeline_intent_for(session), step)
    assert not session.pending_tiles
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles


def test_shared_transform_pipeline_blocks_direct_display_target(monkeypatch):
    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    session.document = ArrayDocument(
        np.ones((TILE, TILE, 8), dtype=np.float32),
        operations=(CenteredFFT(axis=2),),
    )
    tile = session.plan.tiles[0]
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    session.enqueue_pending_tile(tile)
    monkeypatch.setattr(render_effects, "preview_pipeline_commutes_for_display_lod", lambda _session, _tile: False)
    monkeypatch.setattr(
        render_effects,
        "shared_preview_is_useful",
        lambda _session, _tile, _demand, **_kwargs: True,
    )
    effects = MontagePipelineEffects(_RungPrepareRenderer(), session)
    step = RungStep(
        tile_number=0,
        rung=Rung.DESIRED,
        level=2,
        reduce_from_native=True,
        lane=Lane.DISPLAY_PREPARATION,
        priority=Priority.VISIBLE_IMAGE,
        reason="desired display level",
    )

    assert not effects.prepare_rung(_pipeline_intent_for(session), step)
    assert not session.pending_tiles
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles


def test_shared_transform_fallback_releases_per_tile_pending(monkeypatch):
    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    session.document = ArrayDocument(
        np.ones((TILE, TILE, 8), dtype=np.float32),
        operations=(CenteredFFT(axis=2),),
    )
    tile = session.plan.tiles[0]
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    session.enqueue_pending_tile(tile)
    session.display_tile_payloads[0] = DisplayTilePayload(
        0,
        int(tile.source_index),
        np.ones((4, 4), dtype=np.float32),
        None,
        session.tile_semantic_source_id(tile.source_index),
        lod=LodInfo(level=4, factor=16, source_shape=(TILE, TILE), texture_shape=(4, 4)),
        quality="preview",
    )
    session.lifecycle.acknowledge_presented(
        0,
        session.tile_semantic_source_id(tile.source_index),
        quality="preview",
        level=4,
    )
    monkeypatch.setattr(render_effects, "preview_pipeline_commutes_for_display_lod", lambda _session, _tile: False)
    monkeypatch.setattr(
        render_effects,
        "shared_preview_is_useful",
        lambda _session, _tile, _demand, **_kwargs: True,
    )
    effects = MontagePipelineEffects(_RungPrepareRenderer(), session)

    effects.release_display_owned_pending()

    assert not session.pending_tiles
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles


def test_shader_preview_evidence_releases_per_tile_pending(monkeypatch):
    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    session.shader_display = True
    tile = session.plan.tiles[0]
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    session.enqueue_pending_tile(tile)
    source_id = (*session.tile_semantic_source_id(tile.source_index), "floor", "complex_rg32f", (4, 4))
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
    monkeypatch.setattr(render_effects, "preview_pipeline_commutes_for_display_lod", lambda _session, _tile: True)
    effects = MontagePipelineEffects(_RungPrepareRenderer(), session)

    effects.release_display_owned_pending()

    assert not session.pending_tiles
    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles


def test_ready_stage_dependent_rearms_pending_tile_until_materialized():
    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
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

    assert session.pending_tile_numbers() == (0,)
    assert session.stage_fan_in.tile_stage_keys == {}
    assert session.tile_compute_waiting_for_stage == 0
    assert session.stage_backed_tiles_pending == 0
    assert not session.is_complete()


def test_stage_activation_batch_rearms_tiles_after_wait_binding_clears():
    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
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
    assert montage_commit.enqueue_stage_dependent_tiles(session, batch.tiles) == 1
    assert session.pending_tile_numbers() == (0,)
    assert not session.is_complete()


def test_orphaned_stage_dependent_releases_to_direct_tile_work():
    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    stage_key = ("stage", "orphaned")
    del session.rendered_tiles[0]
    session.dirty_payloads.pop(0, None)
    session.stage_fan_in.merge_plan({"tile_stage_keys": {0: stage_key}})

    assert not session.is_complete()

    assert montage_commit.rearm_ready_stage_dependents(session) == 1

    assert session.stage_fan_in.tile_stage_keys == {}
    assert session.pending_tile_numbers() == (0,)
    assert not session.is_complete()


def test_shared_target_marker_is_source_identity_aware():
    from arrayscope.window import montage_commit

    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    demand = session.lod_policy_decision.demand
    old_tile = session.plan.tiles[0]
    retargeted_tile = MontageTile(
        montage_index=old_tile.montage_index,
        source_index=old_tile.source_index + 9,
        row=old_tile.row,
        col=old_tile.col,
        x0=old_tile.x0,
        y0=old_tile.y0,
        width=old_tile.width,
        height=old_tile.height,
        view_state=old_tile.view_state,
    )

    old_marker = montage_commit._shared_transform_marker(
        session,
        demand=demand,
        level=6,
        tiles=(old_tile,),
        shader_display=False,
    )
    retargeted_marker = montage_commit._shared_transform_marker(
        session,
        demand=demand,
        level=6,
        tiles=(retargeted_tile,),
        shader_display=False,
    )

    assert old_marker != retargeted_marker


def test_preview_commit_ack_is_actionable_for_target_followup_replan():
    from types import SimpleNamespace

    from arrayscope.window import montage_commit

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

    assert montage_commit._commit_report_accepted_preview_payload(
        TileCommitReport(presented_tiles=(0, 1), committed_upserts=(0,)),
        tile_state,
    )
    assert not montage_commit._commit_report_accepted_preview_payload(
        TileCommitReport(presented_tiles=(0, 1), committed_upserts=(1,)),
        tile_state,
    )


def test_vispy_level_stats_queue_until_semantic_key_has_evidence():
    from types import SimpleNamespace

    from arrayscope.display.backend_contract import ImageViewBackendCapabilities
    from arrayscope.display.model.montage_levels import MontageLevelTracker, TileLevelStats
    from arrayscope.window import montage_commit

    class Renderer(SimpleNamespace):
        def __init__(self, backend_name):
            super().__init__(
                win=SimpleNamespace(
                    img_view=SimpleNamespace(
                        rendering_capabilities=ImageViewBackendCapabilities(name=backend_name)
                    )
                )
            )
            self.tracker = MontageLevelTracker()

        def _montage_level_tracker(self):
            return self.tracker

    vispy = Renderer("vispy")
    pyqtgraph = SimpleNamespace(
        win=SimpleNamespace(
            img_view=SimpleNamespace(rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"))
        )
    )
    session = SimpleNamespace(level_key=("levels", "retarget"), level_expected_indices=(0,))

    assert montage_commit._commit_should_queue_level_stats(vispy, session, first_display_commit=True)
    assert montage_commit._commit_should_queue_level_stats(vispy, session, first_display_commit=False)

    vispy.tracker.ensure_expected(session.level_key, session.level_expected_indices)
    vispy.tracker.update_from_stats(
        session.level_key,
        TileLevelStats(0, (1.0, 2.0), np.asarray([1.0, 2.0], dtype=np.float32)),
        aggregate=False,
    )

    assert not montage_commit._commit_should_queue_level_stats(vispy, session, first_display_commit=False)
    assert montage_commit._commit_should_queue_level_stats(pyqtgraph, session, first_display_commit=True)
    assert montage_commit._commit_should_queue_level_stats(pyqtgraph, session, first_display_commit=False)


def test_shared_target_waits_for_presented_preview_before_higher_quality():
    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    session.lod_preview_level = 6
    tile = session.plan.tiles[0]
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    source_id = session.tile_semantic_source_id(tile.source_index)
    preview = DisplayTilePayload(
        0,
        0,
        np.ones((4, 4), dtype=np.float32),
        None,
        source_id,
        lod=LodInfo(level=6, factor=64, source_shape=(TILE, TILE), texture_shape=(4, 4)),
        quality="preview",
    )
    session.display_tile_payloads[0] = preview
    demand = session.lod_policy_decision.demand
    desired = int(demand.desired_level)
    assert desired < 6

    assert render_effects.shared_transform_target_level(session, demand) > desired

    session.tile_presentation_state.payloads[0] = preview
    session.lifecycle.backend_presented_snapshot({0: source_id})
    session.lifecycle.presentation_confirmed((0,))

    assert render_effects.shared_transform_target_level(session, demand) == desired


def test_shared_target_upgrade_does_not_wait_for_unrelated_blank_tiles():
    session = _session(count=3, pyramid=PyramidCache(max_bytes=1 << 20))
    session.lod_preview_level = 4
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    demand = session.lod_policy_decision.demand
    desired = int(demand.desired_level)
    assert desired == 2

    tile = session.plan.tiles[0]
    source_id = session.tile_semantic_source_id(tile.source_index)
    preview = DisplayTilePayload(
        0,
        0,
        np.ones((4, 4), dtype=np.float32),
        None,
        source_id,
        lod=LodInfo(level=4, factor=16, source_shape=(TILE, TILE), texture_shape=(4, 4)),
        quality="preview",
    )
    session.display_tile_payloads[0] = preview
    session.tile_presentation_state.payloads[0] = preview
    session.lifecycle.backend_presented_snapshot({0: source_id})
    session.lifecycle.presentation_confirmed((0,))

    preview_tiles = tuple(
        render_effects.shared_transform_candidate_tiles(
            session,
            level=session.lod_preview_level,
            include_missing=True,
            require_presented_preview=False,
        )
    )
    target_tiles = tuple(
        render_effects.shared_transform_candidate_tiles(
            session,
            level=desired,
            include_missing=False,
            require_presented_preview=True,
        )
    )

    assert [int(tile.montage_index) for tile in preview_tiles] == [1, 2]
    assert [int(tile.montage_index) for tile in target_tiles] == [0]


def test_shared_transform_kernel_key_uses_full_semantic_marker():
    from types import SimpleNamespace

    from arrayscope.window import montage_commit

    class CaptureKernel:
        def __init__(self):
            self.specs = []

        def submit(self, spec, **_callbacks):
            self.specs.append(spec)
            return object()

    session = _session(count=2, pyramid=PyramidCache(max_bytes=1 << 20))
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    demand = session.lod_policy_decision.demand
    kernel = CaptureKernel()
    renderer = SimpleNamespace(
        win=SimpleNamespace(kernel=kernel),
        _montage_session_is_current=lambda _session: True,
    )
    effects = MontagePipelineEffects(renderer, session)

    assert effects._submit_shared_transform_target(
        demand=demand,
        level=2,
        tiles=(session.plan.tiles[0],),
        priority=Priority.VISIBLE_IMAGE,
        lane=Lane.DISPLAY_PREPARATION,
    )
    first_key = kernel.specs[-1].key
    assert effects._submit_shared_transform_target(
        demand=demand,
        level=2,
        tiles=(session.plan.tiles[1],),
        priority=Priority.VISIBLE_IMAGE,
        lane=Lane.DISPLAY_PREPARATION,
    )
    second_key = kernel.specs[-1].key

    assert first_key != second_key
    assert first_key[0] == "shared-target"
    assert first_key[1] == montage_commit._shared_transform_marker(
        session,
        demand=demand,
        level=2,
        tiles=(session.plan.tiles[0],),
        shader_display=False,
    )
    assert kernel.specs[-1].supersession.value == second_key[1]


def test_deferred_stage_completion_does_not_enqueue_native_tiles_for_reduced_lod(monkeypatch):
    from types import SimpleNamespace

    session = _session(count=4, pyramid=PyramidCache(max_bytes=1 << 20))
    assert session.lod_policy_mode == LOD_POLICY_RESIDENT
    assert session.lod_policy_decision.demand.desired_level > 0
    session.pending_tiles.clear()
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
        _montage_session_is_current=lambda current: current is session,
        _last_montage_stage_plan_ms=0.0,
        retarget_montage_pipeline=lambda current: setattr(current, "_test_retargeted", True),
    )

    assert montage_commit.complete_deferred_stage_fan_in(renderer, session)
    assert session.pending_tile_numbers() == ()
    assert getattr(session, "_test_retargeted", False) is True


def test_preview_payloads_bypass_item_cap_but_keep_byte_budget():
    session = _session(mode=LOD_POLICY_NATIVE_ONLY, count=4, pyramid=PyramidCache(max_bytes=1 << 20))
    session.rendered_tiles.clear()
    session.dirty_payloads.clear()
    session.pending_payload_upserts.clear()
    for tile in session.plan.tiles:
        tile_number = int(tile.montage_index)
        image = np.full((4, 4), float(tile_number), dtype=np.float32)
        session.display_tile_payloads[tile_number] = DisplayTilePayload(
            tile_number=tile_number,
            source_index=int(tile.source_index),
            image=image,
            histogram_data=None,
            source_id=("preview", tile_number),
            lod=LodInfo(level=4, factor=16, source_shape=(TILE, TILE), texture_shape=image.shape),
            quality="preview",
        )
        session.pending_payload_upserts[tile_number] = None

    _state, delta = session.build_tile_presentation(
        {},
        max_upserts=1,
        max_upsert_bytes=sum(payload.nbytes for payload in session.display_tile_payloads.values()),
        upsert_cost_fn=lambda payload: payload.nbytes,
        item_free_upsert_fn=lambda payload: str(getattr(payload, "quality", "")) == "preview",
        max_item_free_upserts=2,
    )

    assert len(delta.upserts) == 2
    assert set(delta.active_tiles) == {0, 1, 2, 3}


def test_visible_parked_payload_rearms_instead_of_settling_missing():
    session = _session(count=4, pyramid=PyramidCache(max_bytes=1 << 20))
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


def test_source_changed_active_claim_does_not_block_retargeted_prepare():
    """Active request dedupe is source-aware, not montage-slot-only."""

    session = _session(count=1)
    session.rendered_tiles.clear()
    renderer = _RungPrepareRenderer()
    effects = MontagePipelineEffects(renderer, session)
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

    session = _session(count=1, pyramid=PyramidCache(max_bytes=1 << 20))
    renderer = _RungPrepareRenderer()
    effects = MontagePipelineEffects(renderer, session)
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
    level_key = PyramidLevelKey(("old-source",), ("tile", 0), "texture", (2, 2))
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
    assert 0 not in session.active_tile_requests

    effects._admit_ready_payloads(((step, ("materialized", request)),))

    assert 0 not in session.active_tile_requests
    assert 0 not in session.loading_tiles
    assert not session.lifecycle.preview_claim_matches(0, int(Rung.DESIRED), 2)
    assert session.lifecycle.evaluating_tiles == frozenset()
    assert session._test_replan_requested is True


def test_stale_rung_drop_does_not_clear_newer_active_claim_for_same_slot():
    """A stale drop may clean up only the intent marker it created."""

    session = _session(count=1)
    session.rendered_tiles.clear()
    effects = MontagePipelineEffects(_RungPrepareRenderer(), session)
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


def test_reduced_complex_base_keeps_a_level_matched_magnitude_histogram():
    """A reduced RGB/complex base must never lose its magnitude histogram.

    Regression (field defect 2026-07): for PyQtGraph complex tiles the RGB
    plane is phase-hue and the magnitude lives in the histogram (it modulates
    brightness at display time). The on-demand level worker reduces only the
    texture plane, so a reduced level had no matching histogram and the read
    path returned ``None`` — resident-LOD complex tiles rendered phase-only
    with no magnitude, and levels had no effect. The read path must reduce the
    native magnitude to the texture's shape and cache it.
    """

    from arrayscope.render import lod as render_lod

    pyramid = PyramidCache(max_bytes=1 << 20)
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
    # Make only the reduced *texture* resident (as the level worker does),
    # deliberately WITHOUT its histogram.
    level_key = render_lod.pyramid_key_for(session, rendered, demand=demand, level=applied)
    factor_x, factor_y = level_key.factor_xy
    reduced_rgb = reduce_box_mean(rgb_base.astype(np.float32), (factor_x, factor_y)).astype(np.uint8)
    pyramid.admit(level_key, reduced_rgb)

    texture, texture_histogram, lod = render_lod.resident_texture_for_rendered_tile(
        session, rendered, source=rgb_base, histogram=magnitude
    )

    assert lod.level == applied
    assert texture.shape[:2] == reduced_rgb.shape[:2]
    # Magnitude survives, reduced to match the texture, with real variation.
    assert texture_histogram is not None
    assert texture_histogram.shape[:2] == texture.shape[:2]
    assert float(texture_histogram.max()) > float(texture_histogram.min())
    # And it is cached so later reads/level system stay level-consistent.
    assert pyramid.lookup(render_lod.histogram_key_for(session, rendered, demand=demand, level=applied)) is not None


def test_reduced_rgb_resident_without_magnitude_histogram_falls_back_to_native():
    from arrayscope.render import lod as render_lod

    pyramid = PyramidCache(max_bytes=1 << 20)
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
    level_key = render_lod.pyramid_key_for(session, rendered, demand=demand, level=applied)
    pyramid.admit(level_key, reduce_box_mean(rgb_base.astype(np.float32), level_key.factor_xy).astype(np.uint8))

    texture, texture_histogram, lod = render_lod.resident_texture_for_rendered_tile(
        session, rendered, source=rgb_base, histogram=None
    )

    assert lod.level == 0
    assert texture is rgb_base
    assert texture_histogram is None
