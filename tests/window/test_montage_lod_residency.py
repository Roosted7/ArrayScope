"""Qt-free contract tests for resident-LOD montage sessions (ADR 0050)."""

from __future__ import annotations

import numpy as np

from arrayscope.display.lod import LOD_POLICY_NATIVE_ONLY, LOD_POLICY_RESIDENT, select_lod_demand
from arrayscope.display.model.frame import TileCommitReport
from arrayscope.display.montage import MontagePlan, MontageTile, RenderedTile
from arrayscope.display.pyramid import PyramidCache, PyramidLevelKey, reduce_box_mean
from arrayscope.window.montage_session import (
    admit_preview_reduction,
    MontageRenderSession,
    admit_ingest_reduction,
    pyramid_key_for_rendered,
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
        lod_pyramid=pyramid,
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

    plane = request.source
    admitted = None
    steps = tuple(getattr(request, "chain", ()) or ()) or ((request.key, request.reduce_factor_xy),)
    for step_key, rel in steps:
        plane = reduce_box_mean(plane, rel)
        if step_key is not None:
            plane = session.lod_pyramid.admit(step_key, plane)
            if step_key == request.key:
                admitted = plane
    return request.key, admitted


def _release(session, request):
    """Drop one request the way every non-run scheduling path must: all claims."""

    from arrayscope.window.montage_lod import _release_chain_claims, _request_chain

    _release_chain_claims(session.lod_pyramid, _request_chain(request))


def test_native_only_mode_is_unchanged_by_default():
    session = _session(mode=LOD_POLICY_NATIVE_ONLY, pyramid=None)
    state, delta = session.build_tile_presentation({})

    assert session.lod_policy_mode == LOD_POLICY_NATIVE_ONLY
    assert session.lod_policy_decision.policy == "native-only"
    assert session.lod_policy_decision.applied_factor == 1
    assert session.pending_lod_requests == []
    for payload in delta.upserts.values():
        assert payload.lod.level == 0
        assert payload.texture_data.shape[:2] == (TILE, TILE)


def test_resident_mode_falls_back_to_native_and_records_missing_levels():
    session = _session(pyramid=PyramidCache(max_bytes=1 << 20))
    state, delta = session.build_tile_presentation({})

    decision = session.lod_policy_decision
    assert decision.policy == "resident"
    assert decision.demand.desired_level == 2
    # Nothing is materialized yet: applied stays native, never blocking.
    assert decision.applied_level == 0
    for payload in delta.upserts.values():
        assert payload.lod.level == 0
        assert payload.texture_data.shape[:2] == (TILE, TILE)
    # Every tile recorded its demanded-but-missing level exactly once.
    assert len(session.pending_lod_requests) == 2
    tiles = sorted(request[0] for request in session.pending_lod_requests)
    assert tiles == [0, 1]
    for request in session.pending_lod_requests:
        assert isinstance(request.key, PyramidLevelKey)
        assert request.key.factor_xy == (4, 4)
        assert request.reduce_factor_xy == (4, 4)
        assert request.source.shape == (TILE, TILE)


def test_duplicate_materialization_requests_coalesce():
    session = _session(pyramid=PyramidCache(max_bytes=1 << 20))
    session.build_tile_presentation({})
    first = list(session.pending_lod_requests)

    # A second commit while requests are pending must not re-claim them.
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})

    assert list(session.pending_lod_requests) == first


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
    requests = list(session.pending_lod_requests)
    session.pending_lod_requests.clear()
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
    assert session.pending_lod_requests == []


def test_chain_derives_from_the_resident_finer_level_source():
    """A resident finer level becomes the chain source, not the native plane."""

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, view_range=((0.0, 3.0 * 2 * TILE), (0.0, 3.0 * TILE)))
    session.build_tile_presentation({})
    requests = list(session.pending_lod_requests)
    session.pending_lod_requests.clear()
    for request in requests:
        assert tuple(step_key.level_xy for step_key, _rel in request.chain) == ((1, 1),)
        _materialize(session, request)
    assert len(pyramid) == 2

    # Zoom out to level-2 demand (past the hysteresis band of the applied
    # level-1 factor): the request derives from resident level 1.
    session.view_range = ((0.0, 5.0 * 2 * TILE), (0.0, 5.0 * TILE))
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})
    requests = list(session.pending_lod_requests)
    session.pending_lod_requests.clear()
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
    requests = list(session.pending_lod_requests)
    session.pending_lod_requests.clear()
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

    requests = list(session.pending_lod_requests)
    session.pending_lod_requests.clear()
    for request in requests:
        _materialize(session, request)

    session.dirty_payloads.update({0: None, 1: None})
    state, delta = session.build_tile_presentation({})

    assert session.lod_policy_decision.applied_level == 2
    assert set(delta.upserts) == {0, 1}
    for tile, payload in delta.upserts.items():
        assert payload.lod.level == 2
        assert payload.lod.factor == 4
        assert payload.texture_data.shape[:2] == (TILE // 4, TILE // 4)
        # Presentation identity separates levels: a reduced payload can never
        # share a residency key with the native payload it replaces.
        assert payload.source_id != native_ids[tile]
        # Exact semantic sources are untouched by display LOD.
        assert payload.image.shape[:2] == (TILE, TILE)
        assert payload.semantic_data.shape[:2] == (TILE, TILE)
        assert payload.histogram_data.shape[:2] == (TILE, TILE)
    assert session.pending_lod_requests == []


def test_mixed_residency_applies_per_tile_and_reports_common_level():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    session.build_tile_presentation({})
    requests = list(session.pending_lod_requests)
    session.pending_lod_requests.clear()
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
    assert delta.upserts[0].lod.level == 2
    assert delta.upserts[1].lod.level == 0
    assert delta.upserts[0].texture_data.shape[:2] == (TILE // 4, TILE // 4)
    assert delta.upserts[1].texture_data.shape[:2] == (TILE, TILE)
    assert delta.upserts[0].source_id != delta.upserts[1].source_id
    # The session-wide decision only claims what every tile can present.
    assert session.lod_policy_decision.applied_level == 0


def test_threshold_recrossing_hits_the_pyramid_cache_without_new_requests():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    session.build_tile_presentation({})
    requests = list(session.pending_lod_requests)
    session.pending_lod_requests.clear()
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
    hits_before = pyramid.hits
    session.view_range = ZOOMED_OUT_RANGE
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})

    assert session.lod_policy_decision.applied_level == 2
    assert session.pending_lod_requests == []
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
    assert admit_ingest_reduction(pyramid, demand, rendered, semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index)) is not None
    assert len(pyramid) == 1
    assert pyramid.pending_count == 0
    # Singleflight: the level is resident, a second admission is a no-op.
    assert admit_ingest_reduction(pyramid, demand, rendered, semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index)) is None

    # GUI side: the first presentation build selects the reduced level.  No
    # native payload is ever emitted for the tile and nothing is re-requested.
    session.mark_loaded(rendered)
    _state, delta = session.build_tile_presentation({})
    payload = delta.upserts[0]
    assert payload.lod.level == 2
    assert payload.lod.factor == 4
    assert payload.texture_data.shape[:2] == (TILE // 4, TILE // 4)
    # Exact/semantic/histogram sources stay native.
    assert payload.image.shape[:2] == (TILE, TILE)
    assert payload.histogram_data.shape[:2] == (TILE, TILE)
    assert session.pending_lod_requests == []


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
    assert admit_ingest_reduction(pyramid, demand, rendered, semantic_source_id=("test-tile", rendered.tile.source_index)) is not None

    # No special cases: presentation never over-reduces with the stale level;
    # it falls back and the ordinary streaming path materializes level 1.
    session.mark_loaded(rendered)
    _state, delta = session.build_tile_presentation({})
    assert session.lod_policy_decision.demand.desired_level == 1
    assert delta.upserts[0].lod.level == 0
    assert len(session.pending_lod_requests) == 1
    request = session.pending_lod_requests[0]
    assert request.tile_number == 0
    assert request.key.factor_xy == (2, 2)

    request = session.pending_lod_requests.pop()
    _materialize(session, request)
    session.dirty_payloads[0] = None
    _state, delta = session.build_tile_presentation({})
    assert delta.upserts[0].lod.level == 1
    assert delta.upserts[0].texture_data.shape[:2] == (TILE // 2, TILE // 2)


def _acknowledge(session, delta):
    session.acknowledge_tile_presentation(
        delta,
        TileCommitReport(presented_tiles=frozenset(int(tile) for tile in delta.upserts)),
    )


def test_presented_lod_summary_reports_plurality_of_presented_payloads():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=3)

    # Nothing committed yet: fall back to the session-wide decision (native).
    assert session.presented_lod_summary() == (0, 1, (1, 1))

    session.build_tile_presentation({})
    requests = list(session.pending_lod_requests)
    session.pending_lod_requests.clear()
    for request in requests:
        if request[0] in (0, 1):
            _materialize(session, request)
        else:
            _release(session, request)
    session.dirty_payloads.update({0: None, 1: None, 2: None})
    _state, delta = session.build_tile_presentation({})
    _acknowledge(session, delta)

    # Two of three tiles present level 2; the session-wide decision still
    # reads native because tile 2 is not resident yet.  Diagnostics must
    # describe the screen, not the consensus.
    assert session.lod_policy_decision.applied_level == 0
    assert session.presented_lod_summary() == (2, 4, (4, 4))


def test_presented_lod_summary_tie_prefers_the_finer_level():
    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid, count=2)
    session.build_tile_presentation({})
    requests = list(session.pending_lod_requests)
    session.pending_lod_requests.clear()
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
    session.pending_lod_requests.clear()
    return delta


def _admit_zoomed_out_levels(session, level=2):
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    for rendered in session.rendered_tiles.values():
        key = pyramid_key_for_rendered(rendered, demand=demand, level=level, semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index))
        session.lod_pyramid.admit(key, reduce_box_mean(np.asarray(rendered.image), key.factor_xy))


def test_camera_only_retarget_swaps_to_cached_level_without_pan_or_slice_change():
    """ADR 0050 defect: zoom must retarget LOD without any payload dirtying."""

    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    assert all(payload.lod.level == 0 for payload in session.tile_presentation_state.payloads.values())
    _admit_zoomed_out_levels(session)

    # Camera-only zoom out: retarget alone, no pan, no dimension scroll, no
    # tile results.  The demanded level is already resident, so the refresh
    # must request a presentation commit that swaps payload identities.
    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    revision_before = int(session.viewport_revision)
    swap_ready = session.refresh_lod_for_viewport()

    assert swap_ready is True
    assert session.pending_lod_requests == [], "cached levels must not be re-requested"
    assert sorted(session.dirty_payloads) == [0, 1]

    hits_before = pyramid.hits
    _state, delta = session.build_tile_presentation({})
    assert set(delta.upserts) == {0, 1}
    for payload in delta.upserts.values():
        assert payload.lod.level == 2
        assert payload.texture_data.shape[:2] == (TILE // 4, TILE // 4)
    assert pyramid.hits > hits_before
    # No removals: the swap replaces mappings, it never un-presents a tile.
    assert delta.removals == ()

    # A second refresh with the same viewport is a no-op (no revision creep,
    # no commit request, no dirty tiles).
    assert session.refresh_lod_for_viewport() is False
    assert int(session.viewport_revision) >= revision_before


def test_camera_only_retarget_requests_missing_levels_with_new_lod_revision():
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    revision_before = int(session.lod_target_revision)

    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    swap_ready = session.refresh_lod_for_viewport()

    # Nothing resident yet: no swap commit, but materializations are queued
    # under a fresh LOD target revision so stale zoom targets supersede.
    assert swap_ready is False
    assert sorted(request[0] for request in session.pending_lod_requests) == [0, 1]
    for request in session.pending_lod_requests:
        assert request.key.factor_xy == (4, 4)
        assert request.source.shape == (TILE, TILE)
    assert int(session.lod_target_revision) > revision_before
    # Native payloads stay presented untouched while levels materialize.
    assert not session.dirty_payloads
    assert all(payload.lod.level == 0 for payload in session.tile_presentation_state.payloads.values())

    # Requests are singleflighted across refreshes during a zoom gesture.
    assert session.refresh_lod_for_viewport() is False
    assert sorted(request[0] for request in session.pending_lod_requests) == [0, 1]


def test_refresh_is_native_only_noop():
    session = _session(mode=LOD_POLICY_NATIVE_ONLY, pyramid=None, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    assert session.refresh_lod_for_viewport() is False
    assert session.pending_lod_requests == []
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
    assert session.refresh_lod_for_viewport() is True

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


def test_seeding_new_session_keeps_stale_level_payload_presented():
    """Session replacement must reuse a resident payload at the *old* level.

    When the pyramid no longer holds the seeded level, the tile keeps
    presenting the stale-level payload (its texture is materialized by
    construction) instead of flashing through unpresented/native.
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

    # Both tiles are presented immediately, still at level 2, with no
    # placeholder window (they own committed presentation state).
    assert set(replacement.tile_presentation_state.payloads) == {0, 1}
    assert {
        payload.lod.level for payload in replacement.tile_presentation_state.payloads.values()
    } == {2}
    assert replacement.presented_tiles == {0, 1}

    # The refresh accepts the presented level as resident evidence: no
    # down-swap churn to native while the demanded level rematerializes.
    assert replacement.refresh_lod_for_viewport() is False
    assert not replacement.dirty_payloads or set(replacement.dirty_payloads) == {0, 1}

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
    session.refresh_lod_for_viewport()
    for request in list(session.pending_lod_requests):
        _materialize(session, request)
    session.pending_lod_requests.clear()
    assert session.refresh_lod_for_viewport() is True
    _state, delta = session.build_tile_presentation({})

    for tile_number, payload in delta.upserts.items():
        rendered = session.rendered_tiles[int(tile_number)]
        assert payload.lod.level == 2
        # The finest already-computed semantic stats ride along unchanged:
        # a display-LOD swap is invisible to the histogram/level system.
        assert payload.histogram_data is np.asarray(rendered.histogram_data)
        assert payload.level_data is np.asarray(rendered.level_data)
        assert payload.level_stats is rendered.level_stats
        assert payload.image is np.asarray(rendered.image)
    assert session.lod_stats_cross_level_reuses == len(delta.upserts) > 0
    assert session.lod_stats_recomputes == 0

    # Moving finer (level 2 -> native) reuses the same stats objects too.
    session.retarget_viewport(view_range=ZOOMED_IN_RANGE, viewport_shape=VIEWPORT)
    session.refresh_lod_for_viewport()
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
    session.refresh_lod_for_viewport()
    for request in list(session.pending_lod_requests):
        _materialize(session, request)
    session.pending_lod_requests.clear()
    session.refresh_lod_for_viewport()
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
    level1_requests = list(session.pending_lod_requests)
    session.pending_lod_requests.clear()
    assert {request.key.level for request in level1_requests} == {1}
    for request in level1_requests:
        assert request.cross_level is False
        _materialize(session, request)

    session.retarget_viewport(view_range=FAR_OUT_RANGE, viewport_shape=VIEWPORT)
    session.refresh_lod_for_viewport()
    level2_requests = list(session.pending_lod_requests)
    session.pending_lod_requests.clear()
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
    for request in list(session.pending_lod_requests):
        _materialize(session, request)
    session.pending_lod_requests.clear()

    session.retarget_viewport(view_range=FAR_OUT_RANGE, viewport_shape=VIEWPORT)
    session.refresh_lod_for_viewport()
    requests = list(session.pending_lod_requests)
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

    from arrayscope.window import montage_lod

    best = montage_lod.best_floor_key(session_b, 1)
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
    assert not session._reconcile_attempts


def test_backend_identity_reconciliation_retries_are_bounded():
    """A backend that CANNOT converge (e.g. atlas capacity) must not turn
    reconciliation into an idle commit loop: 3 attempts per identity pair."""

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
    assert emitted <= 3, f"unbounded reconciliation retries: {emitted}"


def test_settled_mismatch_is_queryable_for_followup_commit():
    """Field defect 2026-07-05 #3: the report that reveals drawn-slot
    staleness arrives at the END of a commit, and the reconciliation that
    consumes it runs inside the NEXT commit — a settled session (dirty and
    upserts drained) froze with backend_stale_identities nonzero until a pan
    happened to schedule a commit.  The renderer now asks
    backend_identity_mismatch_tiles() after every acknowledgement; it must
    report actionable mismatches, exclude resigned pairs and exhausted
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
    the identity ground truth (renderer seeds last_presented_identities from
    the dying session).  Its first build must repair inherited stale slots
    instead of settling blind on top of them (sid-68 wedge, JSONL 131233)."""

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    # Fresh session, fresh machine: only the inherited map knows slot 1 is
    # stale.  (The renderer copies this dict across replacement.)
    session.last_presented_identities = {
        int(tile): payload.source_id for tile, payload in dict(delta.upserts).items()
    }
    session.last_presented_identities[1] = ("previous-session-level", 5)
    session.dirty_payloads.clear()
    session.pending_payload_upserts.clear()
    assert session.backend_identity_mismatch_tiles() == (1,)
    _state2, delta2 = session.build_tile_presentation({})
    assert 1 in delta2.upserts, "inherited stale slot must re-present"
    assert 1 in tuple(delta2.active_tiles)


def test_resigned_pair_stops_convergence_but_new_result_rearms():
    """The machine's resignation (bounded identity rejections) must silence
    exactly the resigned (wanted, shown) pair — the mismatch query and the
    build reconciliation skip it — while a fresh evaluation clears the
    resignation and gets new chances."""

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    wanted = dict(delta.upserts)[1].source_id
    rec = session.lifecycle.record(1)
    rec.resigned.add((wanted, ("stuck", 6)))
    session.last_presented_identities = {1: ("stuck", 6)}
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


def test_drawn_tile_with_outdated_acknowledged_identity_represents_and_rejoins_active():
    """Field defect 2026-07-05 (JSONL 110937, sid=81): level-2 swap upserts
    for near-scope tiles were declined and parked while the backend kept
    DRAWING their acknowledged level-6 slots — visibly stale rows no
    viewport-derived scope would ever repair.  A drawn tile whose
    acknowledged identity differs from the session's current payload must
    re-present and join the delta's active scope so the backend accepts it."""

    from dataclasses import replace as dc_replace

    from arrayscope.display.model.frame import TilePresentationState

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
    _state, delta = session.build_tile_presentation({})
    report = TileCommitReport(presented_tiles=(0, 1), committed_upserts=(0, 1))
    session.acknowledge_tile_presentation(delta, report)
    session.mark_presented((0, 1))

    # Tile 1 leaves the viewport-derived scope but its slot stays drawn.
    session.visible_tiles = (session.plan.tiles[0],)
    # The backend still shows an OLDER identity than the session now holds.
    stale = dc_replace(session.tile_presentation_state.payloads[1], source_id=("stale", 6))
    session.tile_presentation_state = TilePresentationState(
        {**dict(session.tile_presentation_state.payloads), 1: stale},
        revision=session.tile_presentation_state.revision,
    )

    _state2, delta2 = session.build_tile_presentation({})
    assert 1 in delta2.upserts, "drawn tile with outdated acknowledged identity must re-present"
    assert 1 in tuple(delta2.active_tiles), "and must be acceptable to the viewport-scoped backend"

    report2 = TileCommitReport(presented_tiles=(0, 1), committed_upserts=(1,))
    session.acknowledge_tile_presentation(delta2, report2)
    _state3, delta3 = session.build_tile_presentation({})
    assert 1 not in delta3.upserts, "reconciliation converges and the session settles"


def test_refresh_replans_missing_desired_level_at_unchanged_viewport():
    """Settle-repair contract (field defect 2026-07-05, JSONL 110937,
    sid=80): supersession can kill planned materializations after the last
    camera-driven refresh; a later refresh with an UNCHANGED viewport
    identity must still re-request the missing demanded level, or tiles
    wedge on a coarser resident level until the next pan."""

    from arrayscope.window import montage_lod

    session = _session(pyramid=PyramidCache(max_bytes=1 << 24), count=2)
    assert session.refresh_lod_for_viewport() is not None
    first = list(session.pending_lod_requests)
    assert first, "zoomed-out demand plans materializations for the missing level"

    # Simulate supersession/session churn dropping the planned work.
    released = montage_lod.release_session_claims(session)
    assert released == len(first)
    assert not session.pending_lod_requests

    session.refresh_lod_for_viewport()
    assert session.pending_lod_requests, (
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
    session.refresh_lod_for_viewport()
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
    session.refresh_lod_for_viewport()
    assert int(session.viewport_revision) == before_viewport
    assert int(session.lod_target_revision) == before_lod + 1
    # Unchanged viewport: no further bumps.
    session.refresh_lod_for_viewport()
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
    lost = session.plan.tiles[1]
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

    assert admit_preview_reduction(
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
    assert admit_preview_reduction(
        preview2, rendered, semantic_source_id=semantic_id, preview_level=3,
        reduced=reduced, reduced_level=1,
    )
    key2 = next(iter(preview2.resident_keys_for(semantic_id, rendered.tile.source_index, "scalar")))
    assert np.allclose(preview2.peek(key2), expected, atol=1e-4)

    # Singleflight: second admission is a no-op.
    assert not admit_preview_reduction(
        preview, rendered, semantic_source_id=semantic_id, preview_level=3
    )


def test_floor_presents_from_pinned_preview_when_main_pyramid_lost_the_level():
    """Scroll-back contract: main-cache churn can never blank a tile that was
    ever computed — the pinned preview level floors it."""

    main = PyramidCache(max_bytes=1 << 20)
    preview = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=main, count=2)
    session.lod_preview_pyramid = preview
    session.lod_preview_level = 3

    rendered = session.rendered_tiles[1]
    semantic_id = session.tile_semantic_source_id(rendered.tile.source_index)
    assert admit_preview_reduction(
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
    session.refresh_lod_for_viewport()
    assert 1 in session.dirty_payloads
    _state, _delta = session.build_tile_presentation({})
    assert session.display_tile_payloads[1].quality == "exact"

def test_replaced_session_releases_undrained_request_claims():
    """A dying session's planned-but-undrained requests must free their claims.

    The pyramid is renderer-shared and its keys are semantic: a claim leaked
    on session replacement blocks the SAME levels when the user scrubs back
    to that slice, wedging the tile at the wrong LOD forever (the stale-LOD
    regression of 2026-07-04).
    """

    from arrayscope.window.montage_lod import release_session_claims

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    session.build_tile_presentation({})
    assert len(session.pending_lod_requests) == 2
    assert pyramid.pending_count > 0

    released = release_session_claims(session)

    assert released == 2
    assert session.pending_lod_requests == []
    assert pyramid.pending_count == 0
    # The same slice revisited (equal session key) can claim its levels again.
    replacement = _session(pyramid=pyramid)
    replacement.build_tile_presentation({})
    assert len(replacement.pending_lod_requests) == 2


def test_diagnostics_lod_reason_follows_the_presented_level():
    """The reason text must describe the screen, not the last policy run."""

    from arrayscope.display.lod import (
        LOD_REASON_RESIDENT_FINER,
        LOD_REASON_RESIDENT_MATCH,
    )
    from arrayscope.window.diagnostics_snapshot import _presented_lod_reason

    pyramid = PyramidCache(max_bytes=1 << 20)
    session = _session(pyramid=pyramid)
    session.build_tile_presentation({})
    decision = session.lod_policy_decision
    assert decision.demand.desired_level == 2

    # Nothing presented at the demanded level yet: finer-while-materializing.
    assert _presented_lod_reason(decision, (0, 1, (1, 1))) == LOD_REASON_RESIDENT_FINER
    # The screen converged (ingest-presented level 2) without a policy rerun:
    # the stale decision must not keep reporting "materializes".
    assert _presented_lod_reason(decision, (2, 4, (4, 4))) == LOD_REASON_RESIDENT_MATCH
