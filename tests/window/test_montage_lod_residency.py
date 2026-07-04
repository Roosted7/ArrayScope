"""Qt-free contract tests for resident-LOD montage sessions (ADR 0050)."""

from __future__ import annotations

import numpy as np

from arrayscope.display.lod import LOD_POLICY_NATIVE_ONLY, LOD_POLICY_RESIDENT, select_lod_demand
from arrayscope.display.model.frame import TileCommitReport
from arrayscope.display.montage import MontagePlan, MontageTile, RenderedTile
from arrayscope.display.pyramid import PyramidCache, PyramidLevelKey, reduce_box_mean
from arrayscope.window.montage_session import (
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
    _tile_number, key, source = request
    return key, session.lod_pyramid.admit(key, reduce_box_mean(source, key.factor_xy))


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
    for _tile, key, source in session.pending_lod_requests:
        assert isinstance(key, PyramidLevelKey)
        assert key.factor_xy == (4, 4)
        assert source.shape == (TILE, TILE)


def test_duplicate_materialization_requests_coalesce():
    session = _session(pyramid=PyramidCache(max_bytes=1 << 20))
    session.build_tile_presentation({})
    first = list(session.pending_lod_requests)

    # A second commit while requests are pending must not re-claim them.
    session.dirty_payloads.update({0: None, 1: None})
    session.build_tile_presentation({})

    assert list(session.pending_lod_requests) == first


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
            session.lod_pyramid.end_pending(request[1])

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
    assert admit_ingest_reduction(pyramid, demand, rendered)
    assert len(pyramid) == 1
    assert pyramid.pending_count == 0
    # Singleflight: the level is resident, a second admission is a no-op.
    assert not admit_ingest_reduction(pyramid, demand, rendered)

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
    assert admit_ingest_reduction(pyramid, demand, rendered)

    # No special cases: presentation never over-reduces with the stale level;
    # it falls back and the ordinary streaming path materializes level 1.
    session.mark_loaded(rendered)
    _state, delta = session.build_tile_presentation({})
    assert session.lod_policy_decision.demand.desired_level == 1
    assert delta.upserts[0].lod.level == 0
    assert len(session.pending_lod_requests) == 1
    tile_number, key, _source = session.pending_lod_requests[0]
    assert tile_number == 0
    assert key.factor_xy == (2, 2)

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
            session.lod_pyramid.end_pending(request[1])
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
            session.lod_pyramid.end_pending(request[1])
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
        key = pyramid_key_for_rendered(rendered, demand=demand, level=level)
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


def test_camera_only_retarget_requests_missing_levels_with_new_viewport_revision():
    pyramid = PyramidCache(max_bytes=1 << 24)
    session = _session(pyramid=pyramid, view_range=ZOOMED_IN_RANGE)
    _present_native(session)
    revision_before = int(session.viewport_revision)

    session.retarget_viewport(view_range=ZOOMED_OUT_RANGE, viewport_shape=VIEWPORT)
    swap_ready = session.refresh_lod_for_viewport()

    # Nothing resident yet: no swap commit, but materializations are queued
    # under a fresh viewport revision so stale zoom targets supersede.
    assert swap_ready is False
    assert sorted(request[0] for request in session.pending_lod_requests) == [0, 1]
    for _tile, key, source in session.pending_lod_requests:
        assert key.factor_xy == (4, 4)
        assert source.shape == (TILE, TILE)
    assert int(session.viewport_revision) > revision_before
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
