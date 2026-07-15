"""Golden tests for R2 montage evaluation effects."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.view_state import ViewState
from arrayscope.display.lod import LodDemand, LodInfo
from arrayscope.display.montage import MontageTile, make_montage_plan
from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.pyramid import PyramidCache, PyramidLevelKey
from arrayscope.operations.evaluator import OperationEvaluator
from arrayscope.operations.pipeline import (
    ArrayDocument,
    CenteredFFT,
    CenteredIFFT,
    FFTShift,
    evaluate as evaluate_pipeline,
)
from arrayscope.operations.stage_fanin import StageFanInState
from arrayscope.presentation import ClaimOwner, TileLifecycle, TileTarget
from arrayscope.render import effects
from arrayscope.render.stages import LodAdmissionScope


def _demand(level: int = 0) -> LodDemand:
    factor = 2**level
    return LodDemand(
        desired_level=level,
        desired_factor=factor,
        desired_factor_xy=(factor, factor),
        acceptable_levels=(0, 1, 2),
        source_texels_per_pixel_xy=(float(factor), float(factor)),
        reason="test",
    )


def _session(data=None):
    if data is None:
        data = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
    document = ArrayDocument(data)
    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":")
    )
    plan = make_montage_plan(
        state,
        axis=2,
        indices=(0, 1, 2),
        tile_shape=(4, 6),
        columns=3,
        viewport_shape=(100, 100),
    )
    session = SimpleNamespace(
        document=document,
        view_state=state,
        plan=plan,
        montage_axis=2,
        colormap_lut=None,
        shader_display=False,
        rgb=False,
        lod_preview_level=1,
        stage_fan_in=StageFanInState(),
        lifecycle=TileLifecycle(),
        rendered_tiles={},
        display_tile_payloads={},
        tile_presentation_state=SimpleNamespace(payloads={}),
        pyramid_cache=None,
        tile_semantic_source_id=lambda source_index: ("semantic", int(source_index)),
    )
    from arrayscope.display.model.tile_priority import TilePriorityContext

    session.tile_priority_context = lambda: TilePriorityContext.from_tiles(
        view_range=getattr(session, "view_range", None),
        visible_tiles=getattr(session, "visible_tile_numbers", ()),
    )
    return session


def _assert_optional_array_equal(left, right):
    if left is None or right is None:
        assert left is right
    else:
        np.testing.assert_allclose(np.asarray(left), np.asarray(right))


def _assert_preview_rows_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_allclose(np.asarray(left[1]), np.asarray(right[1]))
    _assert_optional_array_equal(left[2], right[2])
    assert left[3] == right[3]
    assert left[4] == right[4]
    _assert_optional_array_equal(left[5], right[5])
    assert left[6] == right[6]


def test_evaluate_target_tile_level_zero_returns_native_tile_payload():
    session = _session()
    tile = session.plan.tiles[1]
    evaluator = OperationEvaluator(session.document)

    result = effects.evaluate_target_tile(
        session,
        tile,
        level=0,
        demand=_demand(0),
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        stage_cache=evaluator.stage_cache,
        stage_materializer=evaluator.stage_materializer,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    np.testing.assert_allclose(
        result.value.data,
        np.arange(1, 4 * 6 * 3, 3, dtype=np.float32).reshape(4, 6),
    )
    _assert_optional_array_equal(result.value.histogram_data, None)
    assert result.compute_path == "direct"
    assert result.slab_shape == (4, 6)
    assert result.value.level_stats is not None
    assert result.value.level_stats.source_index == tile.source_index


def test_native_tile_discards_result_cancelled_after_evaluation(monkeypatch):
    from arrayscope.operations.cancellation import EvaluationCancelled

    session = _session()
    tile = session.plan.tiles[1]
    evaluator = OperationEvaluator(session.document)
    token = SimpleNamespace(cancelled=False)
    evaluate_image_snapshot = effects.evaluate_image_snapshot

    def cancel_after_evaluation(*args, **kwargs):
        result = evaluate_image_snapshot(*args, **kwargs)
        token.cancelled = True
        return result

    monkeypatch.setattr(effects, "evaluate_image_snapshot", cancel_after_evaluation)

    with pytest.raises(EvaluationCancelled):
        effects.evaluate_target_tile(
            session,
            tile,
            level=0,
            demand=_demand(0),
            semantic_source_id=session.tile_semantic_source_id(tile.source_index),
            stage_cache=evaluator.stage_cache,
            stage_materializer=evaluator.stage_materializer,
            cancellation_token=token,
            shader_display=False,
            evaluation_context=None,
        )


def test_evaluate_target_tile_level_zero_uses_cached_stage_without_waiting_binding(monkeypatch):
    session = _session()
    tile = session.plan.tiles[1]
    stage_key = ("stage", "cached")
    stage_value = object()
    candidate = object()
    plan = SimpleNamespace(region_plan=SimpleNamespace(cache_candidates=(candidate,)))
    session.stage_fan_in = StageFanInState(
        values={stage_key: stage_value},
        tile_stage_plans={int(tile.montage_index): plan},
        tile_stage_candidates={int(tile.montage_index): candidate},
    )

    class Materializer:
        def key_for_candidate(self, _document_key, _candidate):
            assert _candidate is candidate
            return stage_key

    class Cache:
        def get(self, key):
            assert key == stage_key
            return stage_value

    seen = {}

    def fake_evaluate_slab_from_stage(_document, _request, _plan, value, _candidate, **_kwargs):
        seen["value"] = value
        seen["candidate"] = _candidate
        return np.full((4, 6), 7.0, dtype=np.float32)

    monkeypatch.setattr(effects, "evaluate_slab_from_stage", fake_evaluate_slab_from_stage)

    result = effects.evaluate_target_tile(
        session,
        tile,
        level=0,
        demand=_demand(0),
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        stage_cache=Cache(),
        stage_materializer=Materializer(),
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert seen == {"value": stage_value, "candidate": candidate}
    assert result.compute_path == "stage_backed"
    np.testing.assert_allclose(result.value.data, np.full((4, 6), 7.0, dtype=np.float32))
    assert session.stage_fan_in.tile_stage_keys == {}


def test_evaluate_target_tile_non_native_returns_display_payload_not_native_result():
    session = _session()
    tile = session.plan.tiles[1]
    evaluator = OperationEvaluator(session.document)
    demand = _demand(1)

    payload = effects.evaluate_target_tile(
        session,
        tile,
        level=1,
        demand=demand,
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        stage_cache=evaluator.stage_cache,
        stage_materializer=evaluator.stage_materializer,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert not hasattr(payload, "value")
    key, plane, _histogram, _mapping, _kind, _level_data, _level_stats = payload
    assert key.level_xy == (1, 1)
    assert plane.shape == (2, 3)


def test_evaluate_preview_tile_returns_display_only_payload():
    session = _session()
    tile = session.plan.tiles[2]
    demand = _demand(0)
    source_id = session.tile_semantic_source_id(tile.source_index)

    preview = effects.evaluate_preview_tile(
        session,
        tile,
        demand=demand,
        semantic_source_id=source_id,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert preview is not None
    key, plane, histogram, shader_mapping, texture_kind, level_data, level_stats = preview
    assert key.source_id == source_id
    assert key.tile_id == tile.source_index
    assert key.level_xy == (1, 1)
    np.testing.assert_allclose(
        plane,
        np.asarray([[12.5, 18.5, 24.5], [48.5, 54.5, 60.5]], dtype=np.float32),
    )
    assert histogram is None
    assert shader_mapping is not None
    assert texture_kind is not None
    assert level_data is None
    assert level_stats is None


def test_evaluate_preview_tile_uses_requested_rung_level():
    session = _session()
    tile = session.plan.tiles[2]
    demand = _demand(0)
    source_id = session.tile_semantic_source_id(tile.source_index)

    preview = effects.evaluate_preview_tile(
        session,
        tile,
        demand=demand,
        semantic_source_id=source_id,
        level=2,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert preview is not None
    key, plane, *_rest = preview
    assert key.level_xy == (2, 2)
    assert plane.shape == (1, 2)


def test_evaluate_shared_preview_fans_out_display_only_payloads():
    session = _session()
    demand = _demand(0)
    tiles = session.plan.tiles[:2]

    previews = effects.evaluate_shared_preview(
        session,
        tiles[0],
        tiles,
        demand=demand,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert [row[0] for row in previews] == [0, 1]
    for row in previews:
        tile_number, key, plane, histogram, shader_mapping, texture_kind, level_data, level_stats = row
        assert key.source_id == ("semantic", int(tile_number))
        assert key.level_xy == (1, 1)
        assert plane.shape == (2, 3)
        assert histogram is None
        assert shader_mapping is not None
        assert texture_kind is not None
        assert level_data is not None
        assert level_stats is None


def test_shared_preview_candidates_are_limited_to_visible_scope():
    session = _session()
    preview_payload = DisplayTilePayload(
        1,
        1,
        np.ones((2, 3), dtype=np.float32),
        None,
        ("preview", 1),
        lod=LodInfo(level=2, factor=4, source_shape=(4, 6), texture_shape=(2, 3)),
        quality="preview",
    )
    session.display_tile_payloads[1] = preview_payload
    session.tile_presentation_state = SimpleNamespace(payloads={1: preview_payload})
    session.lifecycle.presentation_confirmed((1,))

    first_pixel = tuple(
        effects.shared_transform_candidate_tiles(
            session,
            level=1,
            tile_numbers=(0, 1),
            include_missing=True,
            require_presented_preview=False,
        )
    )
    upgrade = tuple(
        effects.shared_transform_candidate_tiles(
            session,
            level=1,
            tile_numbers=(0, 1),
            include_missing=False,
            require_presented_preview=True,
        )
    )

    assert [int(tile.montage_index) for tile in first_pixel] == [0, 1]
    assert [int(tile.montage_index) for tile in upgrade] == [1]


def test_shared_preview_runs_at_demanded_display_lod():
    session = _session()
    session.lod_preview_level = 1
    demand = _demand(1)
    tiles = session.plan.tiles[:2]

    previews = effects.evaluate_shared_preview(
        session,
        tiles[0],
        tiles,
        demand=demand,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert len(previews) == 2
    assert {row[1].level_xy for row in previews} == {(1, 1)}


def test_reduced_preview_base_samples_display_axes_before_operation_input():
    session = _session()
    demand = _demand(1)
    factor_xy = effects.factor_xy_for_level(demand, 1)

    reduced, preview_state = effects.read_reduced_preview_base_and_state(
        session.document,
        session.plan.tiles[0].view_state,
        factor_xy=factor_xy,
        axis_region_overrides={2: (0, 1, 2)},
        sample_display_axes=True,
    )

    expected = np.asarray(session.document.base_data)[::2, ::2, :]
    np.testing.assert_array_equal(reduced, expected)
    assert reduced.shape == (2, 3, 3)
    assert preview_state.shape == reduced.shape


def test_fft_preview_is_shared_reduced_input_not_per_tile_ladder_input():
    session = _session()
    session.document = ArrayDocument(session.document.base_data, operations=(CenteredFFT(axis=2),))
    tile = session.plan.tiles[0]

    assert effects.can_evaluate_reduced_preview(session, tile) is True
    assert effects.preview_pipeline_commutes_for_display_lod(session, tile) is False
    assert effects.shared_preview_is_useful(session, tile, _demand(1)) is True


def test_shared_fft_shift_ifft_preview_matches_sampled_exact_complex_output():
    data = np.arange(8 * 10 * 8, dtype=np.float32).reshape(8, 10, 8)
    session = _session(data)
    operations = (
        CenteredFFT(axis=2),
        FFTShift(axis=2),
        CenteredIFFT(axis=2),
    )
    session.document = ArrayDocument(data, operations=operations)
    session.shader_display = True
    session.lod_preview_level = 2

    rows = effects.evaluate_shared_preview(
        session,
        session.plan.tiles[0],
        session.plan.tiles,
        demand=_demand(1),
        level=2,
        cancellation_token=None,
        shader_display=True,
        evaluation_context=None,
    )
    exact = evaluate_pipeline(data, operations)

    assert len(rows) == 3
    for tile_number, _key, plane, *_rest in rows:
        np.testing.assert_allclose(
            np.asarray(plane),
            np.asarray(exact)[::4, ::4, int(tile_number)],
            rtol=1e-5,
            atol=1e-5,
        )


def test_shared_fft_preview_maps_shifted_flipped_window_by_source_index():
    data = np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    session = _session(data)
    operations = (
        CenteredFFT(axis=2),
        FFTShift(axis=2),
        CenteredIFFT(axis=2),
    )
    session.document = ArrayDocument(data, operations=operations)
    session.view_state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_axis_flipped(2, True)
        .with_montage_axis(2, columns=3, indices=(5, 6, 7), text="5:8")
    )
    session.plan = make_montage_plan(
        session.view_state,
        axis=2,
        indices=(5, 6, 7),
        tile_shape=(8, 10),
        columns=3,
        viewport_shape=(100, 100),
    )
    session.shader_display = True
    session.lod_preview_level = 2

    rows = effects.evaluate_shared_preview(
        session,
        session.plan.tiles[0],
        session.plan.tiles,
        demand=_demand(1),
        level=2,
        cancellation_token=None,
        shader_display=True,
        evaluation_context=None,
    )
    exact = evaluate_pipeline(data, operations)

    assert [int(row[0]) for row in rows] == [0, 1, 2]
    for row, source_index in zip(rows, (5, 6, 7), strict=True):
        np.testing.assert_allclose(
            np.asarray(row[2]),
            np.asarray(exact)[::4, ::4, int(source_index)],
            rtol=1e-5,
            atol=1e-5,
        )


def test_shared_complex_preview_rows_include_display_histogram():
    data = (
        np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
        + 1j * np.ones((4, 6, 3), dtype=np.float32)
    ).astype(np.complex64)
    session = _session(data)
    session.shader_display = True
    demand = _demand(1)
    tiles = session.plan.tiles[:2]

    rows = effects.evaluate_shared_preview(
        session,
        tiles[0],
        tiles,
        demand=demand,
        cancellation_token=None,
        shader_display=True,
        evaluation_context=None,
    )

    assert len(rows) == 2
    for row in rows:
        _tile_number, _key, plane, histogram, *_rest = row
        assert np.iscomplexobj(plane)
        assert histogram is not None
        assert np.shape(histogram) == np.shape(plane)


def test_reduce_nd_axis_mean_handles_integer_edges():
    values = np.arange(10, dtype=np.uint8)
    reduced = effects.reduce_nd_axis_mean(values, axis=0, factor=4)
    np.testing.assert_array_equal(reduced, np.asarray([2, 6, 8], dtype=np.uint8))


def test_tile_lod_states_reads_lifecycle_and_presented_payload_level():
    session = _session()
    level_key = PyramidLevelKey(
        source_id=("semantic", 0),
        tile_id=0,
        component="scalar",
        level_xy=(2, 2),
    )
    stale_level_key = PyramidLevelKey(
        source_id=("semantic", 99),
        tile_id=99,
        component="scalar",
        level_xy=(3, 3),
    )
    session.lifecycle.level_claimed(0, level_key, ClaimOwner.PREVIEW)
    session.lifecycle.level_resident(0, level_key)
    session.lifecycle.level_claimed(0, stale_level_key, ClaimOwner.PREVIEW)
    session.lifecycle.level_resident(0, stale_level_key)
    session.rendered_tiles[0] = object()
    session.lifecycle.presentation_confirmed((1,))
    session.tile_presentation_state = SimpleNamespace(
        payloads={
            1: DisplayTilePayload(
                1,
                1,
                np.ones((2, 3), dtype=np.float32),
                None,
                ("payload", 1),
                lod=LodInfo(
                    level=1,
                    factor=2,
                    source_shape=(4, 6),
                    texture_shape=(2, 3),
                ),
            )
        }
    )

    states = effects.tile_lod_states(session, _demand(1))

    by_tile = {state.tile_number: state for state in states}
    assert by_tile[0].resident_levels == (2,)
    assert by_tile[1].presented_level == 1


def test_tile_lod_states_ignores_stale_presented_payload_after_slot_retarget():
    session = _session()
    stale_payload = DisplayTilePayload(
        1,
        99,
        np.ones((2, 3), dtype=np.float32),
        None,
        ("payload", "old-source"),
        lod=LodInfo(
            level=0,
            factor=1,
            source_shape=(4, 6),
            texture_shape=(4, 6),
        ),
    )
    session.tile_presentation_state = SimpleNamespace(payloads={1: stale_payload})
    session.lifecycle.backend_presented_snapshot({1: stale_payload.source_id})
    session.lifecycle.presentation_confirmed((1,))

    states = effects.tile_lod_states(session, _demand(0), tile_numbers=(1,))

    assert len(states) == 1
    assert states[0].presented_level is None
    assert states[0].resident_levels == ()


def test_tile_lod_states_prioritizes_screen_distance_in_landscape_viewport():
    session = _session()
    session.plan = SimpleNamespace(
        tiles=(
            MontageTile(0, 0, 0, 0, 69, 4, 2, 2, None),
            MontageTile(1, 1, 0, 1, 49, 8, 2, 2, None),
        )
    )
    session.view_range = ((0.0, 100.0), (0.0, 10.0))
    session.visible_tile_numbers = frozenset({0, 1})
    session.skipped_tiles = set()
    session.active_tile_requests = set()

    states = effects.tile_lod_states(session, _demand(0))

    assert tuple(state.tile_number for state in states) == (0, 1)


def test_tile_lod_states_does_not_treat_resident_level_as_committable_behind_preview():
    session = _session()
    tile = session.plan.tiles[0]
    semantic_source = session.tile_semantic_source_id(tile.source_index)
    level_key = PyramidLevelKey(
        source_id=semantic_source,
        tile_id=int(tile.source_index),
        component="scalar",
        level_xy=(5, 5),
    )
    session.lifecycle.level_claimed(0, level_key, ClaimOwner.PREVIEW)
    session.lifecycle.level_resident(0, level_key)
    preview = DisplayTilePayload(
        0,
        int(tile.source_index),
        np.ones((2, 3), dtype=np.float32),
        None,
        (semantic_source, "preview"),
        lod=LodInfo(1, 2, (4, 6), (2, 3)),
        quality="preview",
    )
    session.lifecycle.fallback_ready(0, preview)
    session.lifecycle.presentation_confirmed((0,))
    session.lifecycle.backend_presented_snapshot({0: preview.source_id})
    session.tile_presentation_state = SimpleNamespace(payloads={0: preview})

    state = effects.tile_lod_states(session, _demand(5), tile_numbers=(0,))[0]

    assert state.presented_quality == "preview"
    assert state.resident_levels == ()
    assert state.target_quality_available is False


def test_tile_lod_states_reads_pyramid_and_preview_floor_residency():
    session = _session()
    demand = _demand(1)
    tile = session.plan.tiles[2]
    session.pyramid_cache = PyramidCache(max_entries=8)
    rendered = effects.rendered_tile_from_evaluation_result(
        tile,
        effects.evaluate_target_tile(
            session,
            tile,
            level=0,
            demand=_demand(0),
            semantic_source_id=session.tile_semantic_source_id(tile.source_index),
            stage_cache=None,
            stage_materializer=None,
            shader_display=False,
            evaluation_context=None,
        ),
    )
    session.rendered_tiles[int(tile.montage_index)] = rendered
    level_key = effects.render_lod.pyramid_key_for(
        session,
        rendered,
        demand=demand,
        level=1,
    )
    session.pyramid_cache.admit(level_key, np.ones((2, 3), dtype=np.float32))
    preview_key = effects.preview_claim_key(
        session,
        tile,
        demand=demand,
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        shader_display=False,
    )
    session.pyramid_cache.admit(preview_key, np.ones((2, 3), dtype=np.float32))
    session.preview_floor_cache = lambda: session.pyramid_cache
    peek_many_calls = []
    peek_many = session.pyramid_cache.peek_many
    session.pyramid_cache.peek_many = lambda keys: (
        peek_many_calls.append(tuple(keys)) or peek_many(keys)
    )

    def unexpected_single_peek(_key):
        raise AssertionError("tile LOD planning must batch floor probes")

    session.pyramid_cache.peek = unexpected_single_peek

    state = {
        state.tile_number: state
        for state in effects.tile_lod_states(session, demand, tile_numbers=(tile.montage_index,))
    }[int(tile.montage_index)]

    # Rendered native tiles are reduction sources, not resident/presented LOD
    # evidence. Lifecycle acknowledgements are the ladder's resident truth.
    assert state.resident_levels == ()
    assert state.floor_available is True
    assert peek_many_calls == [(preview_key,)]


def test_pipeline_effects_tile_states_uses_lifecycle_snapshot():
    from arrayscope.window.frame_effects import FramePipelineEffects

    session = _session()
    level_key = PyramidLevelKey(
        source_id=("semantic", 0),
        tile_id=0,
        component="scalar",
        level_xy=(2, 2),
    )
    session.lifecycle.level_claimed(0, level_key, ClaimOwner.EVALUATION)
    session.lifecycle.level_resident(0, level_key)
    session.rendered_tiles[0] = object()
    bridge = FramePipelineEffects(SimpleNamespace(win=SimpleNamespace()), session)

    states = bridge.tile_states(
        None,
        _demand(2),
        LodAdmissionScope(visible_tile_numbers=frozenset({0, 1, 2})),
    )

    by_tile = {state.tile_number: state for state in states}
    assert by_tile[0].resident_levels == (2,)


def test_pipeline_effects_tile_states_exposes_ready_unacknowledged_fallback():
    session = _session()
    tile = session.plan.tiles[0]
    session.lifecycle.retarget(
        {
            0: TileTarget(
                tile_number=0,
                source_index=int(tile.source_index),
                semantic_source_id=session.tile_semantic_source_id(tile.source_index),
                lod_level=0,
            )
        }
    )
    payload = DisplayTilePayload(
        tile_number=0,
        source_index=int(tile.source_index),
        image=np.zeros((2, 3), dtype=np.float32),
        histogram_data=None,
        source_id=("preview", int(tile.source_index)),
        lod=LodInfo(2, 4, (4, 6), (2, 3), 0),
        quality="preview",
    )
    session.lifecycle.fallback_ready(0, payload)

    state = effects.tile_lod_states(session, _demand(0), tile_numbers=(0,))[0]

    assert state.presented_level is None
    assert state.ready_level == 2
    assert state.ready_quality == "fallback"


def test_cpu_auto_first_commit_allows_progressive_preview_floor():
    from arrayscope.window.frame_effects import FramePipelineEffects

    session = _session()
    session.force_auto = True
    session.user_levels_override = None
    session.display_committed = False
    bridge = FramePipelineEffects(SimpleNamespace(win=SimpleNamespace()), session)

    states = bridge.tile_states(
        None,
        _demand(0),
        LodAdmissionScope(visible_tile_numbers=frozenset({0, 1, 2})),
    )

    assert states
    assert all(state.allow_preview for state in states)


def test_presentation_commit_replays_extent_camera_retarget_after_guard_release():
    from arrayscope.window.frame_effects import _finish_presentation_commit

    scheduled = []
    renderer = SimpleNamespace(
        _montage_presentation_commit_active=True,
        _frame_viewport_retarget_after_commit=True,
        _schedule_frame_viewport_update=lambda *, delay_ms=None: scheduled.append(delay_ms),
    )

    _finish_presentation_commit(renderer)

    assert renderer._montage_presentation_commit_active is False
    assert renderer._frame_viewport_retarget_after_commit is False
    assert scheduled == [1]
