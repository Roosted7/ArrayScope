"""Golden tests for R2 montage evaluation effects."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.display.lod import LodDemand, LodInfo
from arrayscope.display.montage import make_montage_plan
from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.pyramid import PyramidCache, PyramidLevelKey
from arrayscope.operations.evaluator import OperationEvaluator
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.operations.stage_fanin import StageFanInState
from arrayscope.presentation import ClaimOwner, TileLifecycle
from arrayscope.render import effects


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
    return SimpleNamespace(
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
        tile_presentation_state=SimpleNamespace(payloads={}),
        lod_pyramid=None,
        lod_preview_pyramid=None,
        tile_semantic_source_id=lambda source_index: ("semantic", int(source_index)),
    )


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


def test_evaluate_exact_tile_returns_native_tile_payload():
    session = _session()
    tile = session.plan.tiles[1]
    evaluator = OperationEvaluator(session.document)

    result = effects.evaluate_exact_tile(
        session,
        tile,
        stage_cache=evaluator.stage_cache,
        stage_materializer=evaluator.stage_materializer,
        cancellation_token=None,
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
    session.lifecycle.level_claimed(0, level_key, ClaimOwner.PREVIEW)
    session.lifecycle.level_resident(0, level_key)
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


def test_tile_lod_states_reads_pyramid_and_preview_floor_residency():
    session = _session()
    demand = _demand(1)
    tile = session.plan.tiles[2]
    session.lod_pyramid = PyramidCache(max_entries=8)
    session.lod_preview_pyramid = PyramidCache(max_entries=8)
    rendered = effects.rendered_tile_from_evaluation_result(
        tile,
        effects.evaluate_exact_tile(
            session,
            tile,
            stage_cache=None,
            stage_materializer=None,
            evaluation_context=None,
        ),
    )
    session.rendered_tiles[int(tile.montage_index)] = rendered
    level_key = effects.montage_lod.pyramid_key_for(
        session,
        rendered,
        demand=demand,
        level=1,
    )
    session.lod_pyramid.admit(level_key, np.ones((2, 3), dtype=np.float32))
    preview_key = effects.preview_claim_key(
        session,
        tile,
        demand=demand,
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        shader_display=False,
    )
    session.lod_preview_pyramid.admit(preview_key, np.ones((2, 3), dtype=np.float32))
    session.preview_floor_cache = lambda: session.lod_preview_pyramid

    state = {
        state.tile_number: state
        for state in effects.tile_lod_states(session, demand, tile_numbers=(tile.montage_index,))
    }[int(tile.montage_index)]

    assert state.resident_levels == (1,)
    assert state.floor_available is True


def test_pipeline_effects_tile_states_uses_lifecycle_snapshot():
    from arrayscope.window.montage_commit import MontagePipelineEffects

    session = _session()
    level_key = PyramidLevelKey(
        source_id=("semantic", 0),
        tile_id=0,
        component="scalar",
        level_xy=(2, 2),
    )
    session.lifecycle.level_claimed(0, level_key, ClaimOwner.EVALUATION)
    session.lifecycle.level_resident(0, level_key)
    bridge = MontagePipelineEffects(SimpleNamespace(win=SimpleNamespace()), session)

    states = bridge.tile_states(None, _demand(2))

    by_tile = {state.tile_number: state for state in states}
    assert by_tile[0].resident_levels == (2,)
