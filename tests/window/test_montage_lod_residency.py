"""Qt-free contract tests for resident-LOD montage sessions (ADR 0050)."""

from __future__ import annotations

import numpy as np

from arrayscope.display.lod import LOD_POLICY_NATIVE_ONLY, LOD_POLICY_RESIDENT
from arrayscope.display.montage import MontagePlan, MontageTile, RenderedTile
from arrayscope.display.pyramid import PyramidCache, PyramidLevelKey, reduce_box_mean
from arrayscope.window.montage_session import MontageRenderSession

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
