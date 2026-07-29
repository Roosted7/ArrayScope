"""Golden tests for R2 montage evaluation effects."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.view_state import ViewState
from arrayscope.display.lod import LodDemand, LodInfo
from arrayscope.display.model.frame import DisplayTilePayload, PayloadSourceAnchor
from arrayscope.display.montage import MontageTile, RenderedTile, make_montage_plan
from arrayscope.display.pyramid import (
    REDUCER_NATIVE,
    LodPageCache,
    MaterializedLodPage,
    materialize_lod_page,
    plan_source_grid_pages,
)
from arrayscope.display.shader_mapping import ShaderComponent, ShaderMapping, TexturePlaneKind
from arrayscope.display.source_anchoring import SourceAnchoring
from arrayscope.kernel.task import Lane, Priority
from arrayscope.operations.evaluator import OperationEvaluator
from arrayscope.operations.pipeline import (
    ArrayDocument,
    CenteredFFT,
    CenteredIFFT,
    Conjugate,
    FFTShift,
    Mean,
)
from arrayscope.operations.pipeline import (
    evaluate as evaluate_pipeline,
)
from arrayscope.operations.regions import region_shape
from arrayscope.operations.stage_fanin import StageFanInState
from arrayscope.presentation import ClaimOwner, TileLifecycle, TileTarget
from arrayscope.presentation.prepared_uploads import PreparedUploadMailbox
from arrayscope.render import effects
from arrayscope.render import lod as render_lod
from arrayscope.render.ladder import LadderPolicy, LodLadder, Rung
from arrayscope.render.lod import LodPageSetKey
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
        tile_shape=(int(data.shape[0]), int(data.shape[1])),
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
        lod_page_cache=None,
        tile_semantic_source_id=lambda source_index: ("semantic", int(source_index)),
    )
    from arrayscope.display.model.tile_priority import TilePriorityContext

    session.tile_priority_context = lambda: TilePriorityContext.from_tiles(
        view_range=getattr(session, "view_range", None),
        visible_tiles=getattr(session, "visible_tile_numbers", ()),
    )
    return session


def _page_set(*, tile=0, level=2, source_id=None):
    source_id = ("semantic", int(tile)) if source_id is None else source_id
    reducer = REDUCER_NATIVE if int(level) == 0 else "mean"
    plans = plan_source_grid_pages(
        content_key=("test-page-set", source_id),
        valid_source_rect_yx=(0, 4, 0, 6),
        reduction_yx=(int(level), int(level)),
        stored_page_shape=(256, 256),
        dtype="float32",
        representation="scalar_r32f",
        reducer=reducer,
    )
    return LodPageSetKey(
        source_id=source_id,
        tile_id=int(tile),
        level_xy=(int(level), int(level)),
        reducer=reducer,
        plans=plans,
    )


def _admit_page_set(cache, key, source):
    owner = ("test-page-set", key)
    claimed = cache.claim_plans(key.plans, owner)
    try:
        for plan in claimed:
            page = materialize_lod_page(source, source_origin_yx=(0, 0), plan=plan)
            cache.admit_as(plan.key, page, owner=owner)
    finally:
        cache.release_owner_claims(owner)


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


def _stored_preview_values(pages) -> np.ndarray:
    pages = tuple(pages)
    assert pages
    assert all(isinstance(page, MaterializedLodPage) for page in pages)
    y0 = min(page.plan.stored_rect_yx[0] for page in pages)
    y1 = max(page.plan.stored_rect_yx[1] for page in pages)
    x0 = min(page.plan.stored_rect_yx[2] for page in pages)
    x1 = max(page.plan.stored_rect_yx[3] for page in pages)
    values = np.empty((y1 - y0, x1 - x0), dtype=pages[0].values.dtype)
    for page in pages:
        py0, py1, px0, px1 = page.plan.stored_rect_yx
        values[py0 - y0 : py1 - y0, px0 - x0 : px1 - x0] = page.values
    return values


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
    (
        key,
        pages,
        _histogram,
        _mapping,
        _kind,
        _level_data,
        _level_stats,
        _native_source,
    ) = payload
    assert key.level_xy == (1, 1)
    assert len(pages) == 1
    assert pages[0].values.shape == (2, 3)


def test_reduced_target_currently_carries_speculative_native_plane():
    """Characterize target-pass native warming that the R7 contract forbids."""

    session = _session()
    session.source_anchoring = object()
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
        warm_canonical_plane=True,
    )

    key, pages, *_metadata, native_source = payload
    assert key.level_xy == (1, 1)
    assert pages[0].values.shape == (2, 3)
    np.testing.assert_array_equal(
        native_source,
        np.arange(1, 4 * 6 * 3, 3, dtype=np.float32).reshape(4, 6),
    )


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
        level=1,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert preview is not None
    (
        key,
        pages,
        histogram,
        shader_mapping,
        texture_kind,
        level_data,
        level_stats,
        native_source,
    ) = preview
    assert key.source_id == source_id
    assert key.tile_id == tile.source_index
    assert key.level_xy == (1, 1)
    np.testing.assert_allclose(
        pages[0].values,
        np.asarray([[12.5, 18.5, 24.5], [48.5, 54.5, 60.5]], dtype=np.float32),
    )
    assert histogram is None
    assert shader_mapping is not None
    assert texture_kind is not None
    assert level_data is None
    assert level_stats is not None
    assert not level_stats.refined
    assert native_source is None


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
    key, pages, *_rest = preview
    assert key.level_xy == (2, 2)
    assert pages[0].values.shape == (1, 2)


def test_reduced_preview_keeps_unanchored_stepped_crop_window_local():
    data = np.arange(336 * 336 * 3, dtype=np.float32).reshape(336, 336, 3)
    session = _session(data)
    state = (
        session.view_state.with_axis_range(
            0,
            indices=tuple(range(94, 296, 2)),
            text="94:2:294",
        )
        .with_axis_range(
            1,
            indices=tuple(range(66, 268, 2)),
            text="66:2:266",
        )
        .with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":")
    )
    session.view_state = state
    session.plan = make_montage_plan(
        state,
        axis=2,
        indices=(0, 1, 2),
        tile_shape=(101, 101),
        columns=3,
        viewport_shape=(100, 100),
    )
    session.source_anchoring = SourceAnchoring(
        anchored_starts=(None, None),
        content_key=("stepped-window-local",),
    )
    tile = session.plan.tiles[0]

    preview = effects.evaluate_preview_tile(
        session,
        tile,
        demand=_demand(0),
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        level=4,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert preview is not None
    _key, pages, *_rest = preview
    assert pages[0].plan.valid_source_rect_yx == (0, 101, 0, 101)
    assert _stored_preview_values(pages).shape == (7, 7)


def test_reusable_preview_keeps_captured_route_and_source_anchor():
    """A reused session may retarget while an old reusable rung evaluates."""

    session = _session()
    tile = session.plan.tiles[2]
    demand = _demand(0)
    captured_source_id = session.tile_semantic_source_id(tile.source_index)
    session._payload_source_anchor = lambda _shape: PayloadSourceAnchor(
        content_key=("anchored-content",),
        source_rect=(10, 14, 20, 26),
    )
    session.tile_semantic_source_id = lambda source_index: (
        "new-semantic-route",
        int(source_index),
    )

    preview = effects.evaluate_preview_tile(
        session,
        tile,
        demand=demand,
        semantic_source_id=captured_source_id,
        level=1,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert preview is not None
    key, _pages, *_rest = preview
    assert key.source_id == captured_source_id
    assert key.plans[0].valid_source_rect_yx == (10, 14, 20, 26)


def test_evaluate_shared_preview_fans_out_display_only_payloads():
    session = _session()
    demand = _demand(0)
    tiles = session.plan.tiles[:2]

    previews = effects.evaluate_shared_preview(
        session,
        tiles[0],
        tiles,
        demand=demand,
        level=1,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert [row[0] for row in previews] == [0, 1]
    for row in previews:
        (
            tile_number,
            key,
            pages,
            histogram,
            shader_mapping,
            texture_kind,
            level_data,
            level_stats,
        ) = row
        assert key.source_id == ("semantic", int(tile_number))
        assert key.level_xy == (1, 1)
        assert _stored_preview_values(pages).shape == (2, 3)
        assert histogram is None
        assert shader_mapping is not None
        assert texture_kind is not None
        assert level_data is not None
        assert level_stats is not None
        assert int(level_stats.source_index) == int(tile_number)
        assert level_stats.bounds == (
            float(np.min(np.asarray(session.document.base_data)[..., int(tile_number)])),
            float(np.max(np.asarray(session.document.base_data)[..., int(tile_number)])),
        )
        np.testing.assert_allclose(
            _stored_preview_values(pages),
            effects.reduce_display_payload_axes(
                np.asarray(session.document.base_data)[..., int(tile_number)],
                (2, 2),
            ),
        )


def test_shared_preview_uses_the_same_source_alignment_owner_as_per_tile(monkeypatch):
    session = _session()
    tiles = session.plan.tiles[:2]
    original = effects.read_reduced_preview_base_and_state
    observed = []

    def capture(*args, source_aligned=False, **kwargs):
        observed.append(bool(source_aligned))
        return original(*args, source_aligned=source_aligned, **kwargs)

    monkeypatch.setattr(effects, "read_reduced_preview_base_and_state", capture)
    session.source_anchoring = object()
    assert effects.evaluate_shared_preview(
        session,
        tiles[0],
        tiles,
        demand=_demand(0),
        level=1,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )
    session.source_anchoring = None
    assert effects.evaluate_shared_preview(
        session,
        tiles[0],
        tiles,
        demand=_demand(0),
        level=1,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert observed == [True, False]


def test_shared_preview_uses_round_owned_floor_unchanged():
    session = _session()
    session.lod_preview_level = 3
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
    assert {row[1].level_xy for row in previews} == {(3, 3)}


def test_full_montage_preview_uses_screen_scale_floor_chosen_for_round():
    height = width = 336
    y, x = np.indices((height, width), dtype=np.float32)
    data = np.stack(tuple(y * 1000.0 + x + offset for offset in (0.0, 1.0, 2.0)), axis=2)
    session = _session(data)
    demand = LodDemand(
        desired_level=2,
        desired_factor=4,
        desired_factor_xy=(4, 4),
        acceptable_levels=(1, 2, 3),
        source_texels_per_pixel_xy=(7.58, 7.58),
        reason="captured 272-tile montage scale",
    )
    session.lod_preview_level = 5

    previews = effects.evaluate_shared_preview(
        session,
        session.plan.tiles[0],
        session.plan.tiles,
        demand=demand,
        cancellation_token=None,
        shader_display=True,
        evaluation_context=None,
    )

    assert {row[1].level_xy for row in previews} == {(5, 5)}
    for row in previews:
        values = _stored_preview_values(row[2])
        assert values.shape == (11, 11)
        assert float(np.ptp(values)) > 0.0


def test_shared_preview_slices_the_reduced_volume_without_per_tile_slab_plans(monkeypatch):
    session = _session()
    tiles = session.plan.tiles[:2]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("shared preview must not plan one document slab per tile")

    monkeypatch.setattr(effects, "evaluate_image_snapshot", forbidden)

    previews = effects.evaluate_shared_preview(
        session,
        tiles[0],
        tiles,
        demand=_demand(0),
        level=2,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert len(previews) == 2


def test_shared_preview_plans_only_its_selected_page_identity(monkeypatch):
    session = _session()
    tiles = session.plan.tiles[:2]
    original = effects.plan_source_grid_pages
    planned_content_keys = []

    def capture(*args, **kwargs):
        planned_content_keys.append(kwargs["content_key"])
        return original(*args, **kwargs)

    monkeypatch.setattr(effects, "plan_source_grid_pages", capture)
    monkeypatch.setattr(render_lod, "plan_source_grid_pages", capture)

    previews = effects.evaluate_shared_preview(
        session,
        tiles[0],
        tiles,
        demand=_demand(0),
        level=2,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert len(previews) == 2
    assert len(planned_content_keys) == len(tiles)
    assert all(key[-1][-1] == effects.SHARED_PREVIEW_ROUTE for key in planned_content_keys)


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


def test_montage_axis_fft_is_admitted_only_through_cacheable_shared_stage():
    session = _session()
    session.document = ArrayDocument(session.document.base_data, operations=(CenteredFFT(axis=2),))
    tile = session.plan.tiles[0]

    assert effects.can_evaluate_reduced_preview(session, tile) is True
    assert effects.preview_pipeline_commutes_for_display_lod(session, tile) is True
    # The ladder admits it only because the expanded montage-axis stage is
    # cacheable and the tile requests prove they share one real-document
    # reduced region. This is stricter than merely asking whether it commutes.
    assert effects.preview_pipeline_is_tile_local(session, tile) is True


def test_pyqtgraph_shared_fft_preview_retains_reduced_complex_source_format():
    data = np.arange(8 * 10 * 8, dtype=np.float32).reshape(8, 10, 8)
    session = _session(data)
    session.document = ArrayDocument(data, operations=(CenteredFFT(axis=2),))
    session.rgb = True
    session.shader_display = False
    session.lod_preview_level = 2

    rows = effects.evaluate_shared_preview(
        session,
        session.plan.tiles[0],
        session.plan.tiles,
        demand=_demand(1),
        level=2,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert rows
    for _tile_number, _key, pages, histogram, mapping, texture_kind, *_rest in rows:
        assert texture_kind == TexturePlaneKind.COMPLEX_RG32F
        assert np.iscomplexobj(_stored_preview_values(pages))
        assert histogram is not None
        assert mapping is not None


def test_shared_fft_prepares_native_evidence_without_a_second_full_transform(monkeypatch):
    data = np.arange(128 * 128 * 8, dtype=np.float32).reshape(128, 128, 8)
    session = _session(data)
    session.document = ArrayDocument(data, operations=(CenteredFFT(axis=2),))
    session.shader_display = True
    session.lod_preview_level = 2
    original = effects._evaluate_reduced_preview_volume
    input_shapes = []

    def counted(*args, **kwargs):
        input_shapes.append(tuple(int(size) for size in np.shape(args[1])))
        return original(*args, **kwargs)

    monkeypatch.setattr(effects, "_evaluate_reduced_preview_volume", counted)
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

    assert len(rows) == len(session.plan.tiles)
    assert input_shapes == [(32, 32, 8)]
    assert all(row[-1].bounds is not None for row in rows)


def test_shared_fft_native_floor_reuses_one_input_read_for_pixels_and_bounds(monkeypatch):
    data = np.arange(16 * 16 * 8, dtype=np.float32).reshape(16, 16, 8)
    session = _session(data)
    session.document = ArrayDocument(data, operations=(CenteredFFT(axis=2),))
    session.shader_display = True
    session.lod_preview_level = 0
    original = effects.read_reduced_preview_base_and_state
    factors = []

    def counted(*args, **kwargs):
        factors.append(tuple(kwargs["factor_xy"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(effects, "read_reduced_preview_base_and_state", counted)
    rows = effects.evaluate_shared_preview(
        session,
        session.plan.tiles[0],
        session.plan.tiles,
        demand=_demand(0),
        level=0,
        cancellation_token=None,
        shader_display=True,
        evaluation_context=None,
    )

    assert rows
    assert factors == [(1, 1)]


@pytest.mark.parametrize("extreme_yx", [(8, 8), (9, 9)])
@pytest.mark.parametrize("extreme_value", [-1000.0, 1000.0])
def test_preview_cohort_level_evidence_contains_the_native_extremes(
    extreme_yx,
    extreme_value,
):
    """Round levels from the preview cohort must still contain what is drawn.

    The displayed page is a real box mean, while the cohort bounds come from
    the transformed native plane the same preview worker held before reducing
    those pixels. A lone extreme must therefore affect the mean by the same
    amount and remain verbatim in the evidence regardless of stride alignment.
    FFT is the realistic case: k-space is one sharp DC peak on an otherwise dim
    field, so losing that one source value would mis-scale the whole montage.
    """

    data = np.zeros((16, 16, 3), dtype=np.float32)
    data[extreme_yx[0], extreme_yx[1], :] = extreme_value
    session = _session(data)
    session.lod_preview_level = 2

    rows = effects.evaluate_shared_preview(
        session,
        session.plan.tiles[0],
        session.plan.tiles,
        demand=_demand(0),
        level=2,
        cancellation_token=None,
        shader_display=True,
        evaluation_context=None,
    )

    assert rows
    _tile, key, pages, _histogram, _mapping, _kind, _level_data, stats = rows[0]
    assert stats is not None, "preview cohort must carry level evidence"
    assert key.reducer == "mean"
    stored = _stored_preview_values(pages)
    assert float(
        stored[np.unravel_index(np.argmax(np.abs(stored)), stored.shape)]
    ) == pytest.approx(extreme_value / 16.0)
    assert stats.bounds[0] <= float(np.min(data))
    assert stats.bounds[1] >= float(np.max(data))


@pytest.mark.parametrize("extreme_value", [-1000.0, 1000.0])
def test_preview_cohort_bounds_survive_above_the_evidence_sample_limit(extreme_value):
    """The same guarantee, at a scale that defeats a bounded sample.

    The sibling test above uses a 16x16x3 source: 768 display positions, well
    under ``REFINED_TILE_SAMPLE_LIMIT`` (8192), so every position is visited
    and any subsampling strategy passes it trivially. That made it unable to
    fail for the one defect it exists to catch.

    A field source is 336x336 per slice, so a cohort is millions of positions
    against that same 8192 cap -- three orders of magnitude of subsampling. A
    lone sharp peak is exactly what falls through, and k-space is precisely a
    lone sharp peak on an otherwise dim field, so this is the shape of the real
    data rather than an adversarial one.

    Measured: a bounded-sample implementation reports ``(-0.5, 0.5)`` here for
    a native maximum of 1000.0 -- a degenerate window padded around an all-zero
    sample, mis-scaling the montage by three orders of magnitude, while the
    sibling test stays green.
    """

    data = np.zeros((336, 336, 8), dtype=np.float32)
    data[169, 169, :] = extreme_value  # off every coarse grid, one position only
    session = _session(data)
    session.lod_preview_level = 2

    rows = effects.evaluate_shared_preview(
        session,
        session.plan.tiles[0],
        session.plan.tiles,
        demand=_demand(0),
        level=2,
        cancellation_token=None,
        shader_display=True,
        evaluation_context=None,
    )

    assert rows
    stats = rows[0][-1]
    assert stats is not None, "preview cohort must carry level evidence"
    assert stats.bounds[0] <= float(np.min(data)), (
        f"cohort lower bound {stats.bounds[0]} excludes native minimum {np.min(data)}"
    )
    assert stats.bounds[1] >= float(np.max(data)), (
        f"cohort upper bound {stats.bounds[1]} excludes native maximum {np.max(data)}"
    )


@pytest.mark.parametrize("extreme_value", [-1000.0, 1000.0])
def test_shared_fft_analytic_bounds_cover_a_field_scale_native_peak(extreme_value):
    data = np.zeros((336, 336, 8), dtype=np.float32)
    data[169, 169, :] = extreme_value
    session = _session(data)
    session.document = ArrayDocument(data, operations=(CenteredFFT(axis=2),))
    session.shader_display = True
    session.lod_preview_level = 2

    rows = effects.evaluate_shared_preview(
        session,
        session.plan.tiles[0],
        session.plan.tiles,
        demand=_demand(0),
        level=2,
        cancellation_token=None,
        shader_display=True,
        evaluation_context=None,
    )

    exact = evaluate_pipeline(data, session.document.enabled_operations)
    exact_low = float(np.min(np.real(exact)))
    exact_high = float(np.max(np.real(exact)))
    for row in rows:
        stats = row[-1]
        assert stats is not None
        assert stats.bounds[0] <= exact_low
        assert stats.bounds[1] >= exact_high


@pytest.mark.parametrize(
    ("distribution", "max_looseness"),
    [("non-negative", 1.01), ("zero-mean", 1.30)],
)
def test_shared_fft_analytic_bounds_stay_tight_enough_to_be_worth_using(
    distribution,
    max_looseness,
):
    """Containment alone is not enough: a huge window is legal and useless.

    The analytic envelope replaces an exact scan, so it is conservative by
    construction and every containment test would still pass if it widened to
    ``(-1e30, 1e30)``. That would satisfy R3 and destroy the image, so the
    envelope needs an upper bound on its looseness as well as a guarantee of
    coverage.

    ``max_k|X_k| <= sum_m|x_m| / sqrt(N)`` is exactly attained whenever the DC
    bin dominates, which is the case for the non-negative magnitude data a
    montage-axis FFT is usually taken over -- so the realistic case costs no
    contrast at all. Zero-mean data is the unfavourable direction, where
    cancellation in the DC bin leaves the bound above the true peak.
    """

    rng = np.random.default_rng(7)
    raw = rng.normal(size=(64, 64, 8))
    data = (np.abs(raw) if distribution == "non-negative" else raw).astype(np.float32)
    session = _session(data)
    session.document = ArrayDocument(data, operations=(CenteredFFT(axis=2),))
    session.shader_display = True
    session.lod_preview_level = 2

    rows = effects.evaluate_shared_preview(
        session,
        session.plan.tiles[0],
        session.plan.tiles,
        demand=_demand(0),
        level=2,
        cancellation_token=None,
        shader_display=True,
        evaluation_context=None,
    )

    exact = evaluate_pipeline(data, session.document.enabled_operations)
    exact_peak = float(np.max(np.abs(exact)))
    envelope = float(rows[0][-1].bounds[1])

    assert envelope >= exact_peak, "envelope must still contain the true peak"
    looseness = envelope / exact_peak
    assert looseness <= max_looseness, (
        f"{distribution} envelope is {looseness:.3f}x the true peak "
        f"({envelope:.3f} vs {exact_peak:.3f}); the window is too wide to be useful"
    )


def test_pyqtgraph_per_tile_fft_preview_also_retains_the_complex_source_format():
    """The per-tile reduced route carries the same storm-safety invariant.

    ``can_evaluate_reduced_preview`` used to decline CPU-composited complex
    outright, because a ``(h, w, 3)`` display plane is neither scalar nor
    complex and ``render.lod._reducer_format_for_rendered`` raises on it: the
    rung failed, the pipeline replanned, and it failed again -- a real session
    logged 18314 admissions against 773 completions. That blanket decline is
    gone, so the safety now rests on every preview route emitting a complex
    plane instead. ``evaluate_shared_preview`` is covered above; this is the
    other route, reached through ``evaluate_preview_tile`` when the shared
    batch is not used, and it was previously untested.
    """

    data = np.arange(8 * 10 * 8, dtype=np.float32).reshape(8, 10, 8)
    session = _session(data)
    session.document = ArrayDocument(data, operations=(CenteredFFT(axis=2),))
    session.rgb = True
    session.shader_display = False
    session.lod_preview_level = 2
    tile = session.plan.tiles[0]

    assert effects.display_output_is_composited_rgb(session) is True
    assert effects.can_evaluate_reduced_preview(session, tile) is True

    result = effects.evaluate_preview_tile(
        session,
        tile,
        demand=_demand(1),
        semantic_source_id=("session", 0),
        level=2,
        cancellation_token=None,
        shader_display=False,
    )

    assert result is not None
    _key, pages, _display, mapping, texture_kind, *_rest = result
    values = _stored_preview_values(pages)
    assert texture_kind == TexturePlaneKind.COMPLEX_RG32F
    assert np.iscomplexobj(values), "a composited (h, w, 3) plane would re-open the replan storm"
    assert values.ndim == 2

    # The step the storm actually died on: this must route, not raise.
    reducer, dtype, representation = render_lod._reducer_format_for_rendered(
        SimpleNamespace(shader_mapping=mapping), values
    )
    assert (dtype, representation) == ("complex64", "complex_rg32f")
    assert reducer


@pytest.mark.parametrize("shader_display", [True, False])
@pytest.mark.parametrize(
    ("operations", "admitted"),
    [
        ((), True),
        ((FFTShift(axis=2),), True),
        ((Conjugate(),), True),
        ((FFTShift(axis=2), CenteredFFT(axis=2)), True),
        ((FFTShift(axis=0),), False),
        ((CenteredFFT(axis=0),), False),
    ],
    ids=[
        "raw",
        "montage-axis-reindex",
        "pointwise",
        "reindex-then-montage-fft",
        "display-axis-reindex",
        "display-axis-fft",
    ],
)
def test_montage_axis_reindex_keeps_its_preview_pass(operations, admitted, shader_display):
    """A pipeline that only reindexes the montage axis still previews.

    ``preview_pipeline_is_tile_local`` used to require an expanding, cacheable
    stage on the montage axis before admitting the reduced coarse rung. An
    operation that merely REINDEXES that axis -- ``FFTShift`` along it -- expands
    no request, so it never set that flag and was refused, even though it is
    strictly easier to serve than the shared-stage FFT case that was admitted.

    The visible consequence was total: `_reduced_input_coarse_rung_available`
    returned False, the ladder planned no FLOOR rung, and changing operations or
    reloading the source with any such operation active jumped the montage
    straight to target quality with no preview pass at all -- an R4 violation
    reported from the field on WGPU.

    Display-axis transforms must still be refused: a roll along a display axis
    does not commute with a source-anchored reduction unless the shift is a
    multiple of the reduction factor.
    """

    data = np.arange(16 * 16 * 4, dtype=np.float32).reshape(16, 16, 4)
    session = _session(data)
    session.document = ArrayDocument(data, operations=operations)
    session.shader_display = shader_display
    seed = session.plan.tiles[0]

    assert effects.preview_pipeline_is_tile_local(session, seed) is admitted


def test_display_axis_fft_is_not_admitted_to_the_coarse_ladder():
    session = _session()
    session.document = ArrayDocument(session.document.base_data, operations=(CenteredFFT(axis=0),))
    tile = session.plan.tiles[0]

    assert effects.preview_pipeline_commutes_for_display_lod(session, tile) is False
    assert effects.preview_pipeline_is_tile_local(session, tile) is False

    with pytest.raises(
        ValueError,
        match="commuting tile-local admission",
    ):
        effects.evaluate_shared_preview(
            session,
            tile,
            session.plan.tiles,
            demand=_demand(1),
            level=2,
            cancellation_token=None,
            shader_display=True,
            evaluation_context=None,
        )


@pytest.mark.parametrize("shader_display", [True, False], ids=["wgpu", "pyqtgraph"])
def test_non_reducible_preview_evaluates_once_and_carries_exact_result(monkeypatch, shader_display):
    """R4/R2: native-output FLOOR owns one evaluation on both backends."""

    data = np.arange(16 * 16 * 4, dtype=np.float32).reshape(16, 16, 4)
    session = _session(data)
    session.document = ArrayDocument(data, operations=(CenteredFFT(axis=0),))
    session.shader_display = shader_display
    tile = session.plan.tiles[0]
    calls = 0
    evaluate_native = effects._evaluate_native_tile_result

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return evaluate_native(*args, **kwargs)

    monkeypatch.setattr(effects, "_evaluate_native_tile_result", counted)
    payload = effects.evaluate_preview_tile(
        session,
        tile,
        demand=_demand(1),
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        level=2,
        cancellation_token=None,
        shader_display=shader_display,
        evaluation_context=None,
    )

    assert effects.preview_pipeline_is_tile_local(session, tile) is False
    assert calls == 1
    assert len(payload) == 9
    key, pages, *_metadata, rendered = payload
    assert key.level_xy == (2, 2)
    assert pages
    assert isinstance(rendered, RenderedTile)
    assert rendered.tile == tile


def test_shape_changing_pipeline_can_evaluate_native_output_preview():
    base = np.arange(2 * 3 * 5 * 7, dtype=np.float32).reshape(2, 3, 5, 7)
    session = _session(np.mean(base, axis=1))
    session.document = ArrayDocument(base, operations=(Mean(axis=1),))
    tile = session.plan.tiles[0]

    payload = effects.evaluate_preview_tile(
        session,
        tile,
        demand=_demand(1),
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        level=2,
        cancellation_token=None,
        shader_display=False,
        evaluation_context=None,
    )

    assert payload is not None
    assert isinstance(payload[-1], RenderedTile)
    assert payload[-1].image.shape == (2, 5)


def test_raw_and_pointwise_pipelines_stay_coarse_ladder_admissible():
    session = _session()
    tile = session.plan.tiles[0]

    for operations in ((), (Conjugate(),)):
        session.document = ArrayDocument(session.document.base_data, operations=operations)
        assert effects.preview_pipeline_is_tile_local(session, tile) is True, operations

    # A transform on the scrub/montage axis is not tile-local even where there
    # is no montage axis to narrow: one tile still needs the whole volume.
    session.document = ArrayDocument(session.document.base_data, operations=(CenteredFFT(axis=2),))
    session.montage_axis = None
    assert effects.preview_pipeline_is_tile_local(session, tile) is False


def test_noncommuting_shared_preview_cannot_alias_direct_exact_pages():
    data = np.arange(8 * 10 * 3, dtype=np.float32).reshape(8, 10, 3)
    session = _session(data)
    session.document = ArrayDocument(data, operations=(CenteredFFT(axis=2),))
    session.shader_display = True
    session.lod_preview_level = 2
    demand = _demand(1)
    tile = session.plan.tiles[0]

    rows = effects.evaluate_shared_preview(
        session,
        tile,
        (tile,),
        demand=demand,
        level=2,
        cancellation_token=None,
        shader_display=True,
        evaluation_context=None,
    )
    preview_key = rows[0][1]
    preview_pages = rows[0][2]
    exact_volume = evaluate_pipeline(data, session.document.enabled_operations)
    exact_plane = np.ascontiguousarray(
        exact_volume[..., int(tile.source_index)], dtype=np.complex64
    )
    exact_rendered = RenderedTile(
        tile=tile,
        image=exact_plane,
        histogram_data=np.abs(exact_plane).astype(np.float32),
        eval_ms=0.0,
        slab_shape=exact_plane.shape,
        slab_nbytes=exact_plane.nbytes,
        semantic_data=exact_plane,
        lod_source_data=exact_plane,
    )
    exact_key = render_lod.page_set_key_for_rendered(
        exact_rendered,
        demand=demand,
        level=2,
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        shader_display=True,
    )

    assert preview_key.source_id == exact_key.source_id
    assert preview_key.page_keys != exact_key.page_keys
    assert preview_key.plans[0].key.operation_key != exact_key.plans[0].key.operation_key
    cache = LodPageCache(max_bytes=1 << 20)
    owner = ("noncommuting-preview", 0)
    assert cache.claim_plans(preview_key.plans, owner) == preview_key.plans
    try:
        for page in preview_pages:
            cache.admit_as(page.key, page, owner=owner)
    finally:
        cache.release_owner_claims(owner)
    assert cache.exact_pages(preview_key.plans) is not None
    assert cache.exact_pages(exact_key.plans) is None


def test_shared_fft_shift_ifft_preview_box_means_exact_complex_output():
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
    for tile_number, _key, pages, *_rest in rows:
        np.testing.assert_allclose(
            _stored_preview_values(pages),
            effects.reduce_display_payload_axes(
                np.asarray(exact)[..., int(tile_number)],
                (4, 4),
            ),
            rtol=1e-5,
            atol=1e-5,
        )
        stats = _rest[-1]
        exact_plane = np.real(np.asarray(exact)[..., int(tile_number)])
        assert stats.bounds[0] <= float(np.min(exact_plane))
        assert stats.bounds[1] >= float(np.max(exact_plane))


def test_fft_coarse_rung_materializes_one_reduced_stage_for_272_tiles():
    """Gate 2: sharing is a counter assertion, never a timing inference."""

    data = np.arange(16 * 16 * 272, dtype=np.float32).reshape(16, 16, 272)
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
        .with_montage_axis(2, columns=17, indices=tuple(range(272)), text=":")
    )
    session.plan = make_montage_plan(
        session.view_state,
        axis=2,
        indices=tuple(range(272)),
        tile_shape=(16, 16),
        columns=17,
        viewport_shape=(1000, 1000),
    )
    session.lod_preview_level = 4
    session.shader_display = True
    evaluator = OperationEvaluator(session.document)
    demand = _demand(0)

    rows = [
        effects.evaluate_preview_tile(
            session,
            tile,
            demand=demand,
            semantic_source_id=session.tile_semantic_source_id(tile.source_index),
            level=4,
            cancellation_token=None,
            shader_display=True,
            evaluation_context=None,
            stage_cache=evaluator.stage_cache,
            stage_materializer=evaluator.stage_materializer,
            # A reduced page must never trigger the exact native-plane warm.
            warm_canonical_plane=True,
        )
        for tile in session.plan.tiles
    ]

    stage = evaluator.stage_materialization_diagnostics()
    cache = evaluator.stage_cache_diagnostics()
    assert stage.scheduled == 1
    assert stage.completed == 1
    assert stage.hits + stage.attached == 271
    assert cache.compute_claims == 1
    assert cache.stores == 1
    [(stage_key, stage_value)] = evaluator.stage_cache._resident_snapshot
    assert region_shape(stage_key.shape, stage_value.region) == stage_value.data.shape
    assert stage_value.region.axes[0].value is None
    assert stage_value.region.axes[1].value is None
    assert stage_value.data.shape == (16, 16, 272)
    assert len(rows) == 272
    assert all(row[0].level_xy == (4, 4) for row in rows)
    assert all(row[-1] is None for row in rows), "coarse pages carry no native warm plane"


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
            _stored_preview_values(row[2]),
            effects.reduce_display_payload_axes(
                np.asarray(exact)[..., int(source_index)],
                (4, 4),
            ),
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
        _tile_number, _key, pages, histogram, *_rest = row
        plane = _stored_preview_values(pages)
        assert np.iscomplexobj(plane)
        assert histogram is not None
        assert np.shape(histogram) == np.shape(plane)


def test_reduce_nd_axis_mean_handles_integer_edges():
    values = np.arange(10, dtype=np.uint8)
    reduced = effects.reduce_nd_axis_mean(values, axis=0, factor=4)
    np.testing.assert_array_equal(reduced, np.asarray([2, 6, 8], dtype=np.uint8))


def test_reduce_nd_axis_mean_keeps_partial_bins_aligned_to_source_grid():
    values = np.arange(94, 194, dtype=np.float32)
    reduced = effects.reduce_nd_axis_mean(
        values,
        axis=0,
        factor=16,
        source_start=94,
    )

    np.testing.assert_array_equal(
        reduced,
        np.asarray([94.5, 103.5, 119.5, 135.5, 151.5, 167.5, 183.5, 192.5]),
    )


def test_reduce_nd_axis_mean_keeps_unanchored_crop_bins_local():
    values = np.arange(94, 194, dtype=np.float32)

    reduced = effects.reduce_nd_axis_mean(values, axis=0, factor=16)

    np.testing.assert_array_equal(
        reduced,
        np.asarray([101.5, 117.5, 133.5, 149.5, 165.5, 181.5, 191.5]),
    )


def test_tile_lod_states_reads_lifecycle_and_presented_payload_level():
    session = _session()
    level_key = _page_set(tile=0, level=2)
    stale_level_key = _page_set(tile=99, level=3)
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
    level_key = _page_set(tile=tile.source_index, level=5, source_id=semantic_source)
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


def test_tile_lod_states_reads_page_cache_and_preview_floor_residency():
    session = _session()
    demand = _demand(1)
    tile = session.plan.tiles[2]
    session.lod_page_cache = LodPageCache(max_entries=8)
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
    session.lod_preview_level = 3
    preview_level = effects.preview_evaluation_level(session, demand)
    assert preview_level == 3
    level_key = effects.render_lod.page_set_key_for(
        session,
        rendered,
        demand=demand,
        level=preview_level,
    )
    _admit_page_set(session.lod_page_cache, level_key, rendered.image)
    session.lifecycle.level_claimed(int(tile.montage_index), level_key, ClaimOwner.PREVIEW)
    session.lifecycle.level_resident(int(tile.montage_index), level_key)
    preview_key = effects.preview_claim_key(
        session,
        tile,
        demand=demand,
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        shader_display=False,
    )
    assert preview_key == level_key
    session._best_floor_key = lambda *_args, **_kwargs: (
        level_key,
        level_key.level,
        session.lod_page_cache,
    )

    state = {
        state.tile_number: state
        for state in effects.tile_lod_states(session, demand, tile_numbers=(tile.montage_index,))
    }[int(tile.montage_index)]

    # Lifecycle acknowledgement plus complete exact page residency is the
    # ladder's resident truth; the rendered native tile is only its source.
    assert state.resident_levels == (3,)
    assert state.floor_available is True


def test_unpresented_native_residency_is_not_presented_target_quality():
    """Resident data cannot close the physical presentation obligation."""

    session = _session()
    tile = session.plan.tiles[0]
    source_id = session.tile_semantic_source_id(tile.source_index)
    key = _page_set(tile=tile.source_index, level=0, source_id=source_id)
    session.lod_page_cache = LodPageCache(max_entries=8)
    source = np.asarray(session.document.base_data[:, :, int(tile.source_index)])
    _admit_page_set(session.lod_page_cache, key, source)
    session.lifecycle.level_claimed(0, key, ClaimOwner.EVALUATION)
    session.lifecycle.level_resident(0, key)
    session.rendered_tiles[0] = object()
    session._best_floor_key = lambda *_args, **_kwargs: (
        key,
        key.level,
        session.lod_page_cache,
    )
    session.pending_payload_upserts = {}
    session.dirty_payloads = {}

    state = effects.tile_lod_states(session, _demand(4), tile_numbers=(0,))[0]

    assert state.resident_levels == (0,)
    assert state.presented_level is None
    assert state.target_quality_available is False
    assert state.allow_preview is True
    assert state.floor_available is True
    assert state.presentation_pending is False


def test_resident_floor_from_predecessor_crop_requires_current_window_production():
    """A source match cannot make different source-window pixels presentable."""

    session = _session()
    tile = session.plan.tiles[0]
    session.output_dtype = np.dtype(np.float32)
    session.canonical_orientation = False
    session._lod_page_set_key_cache = {}
    session.lod_page_cache = LodPageCache(max_entries=8)
    demand = _demand(1)
    session.lod_policy_decision = SimpleNamespace(demand=demand)
    source_rect = [0, 4, 0, 6]
    content_key = (
        "src-anchored",
        session.tile_semantic_source_id(tile.source_index),
        ("display-plane",),
    )
    session.payload_source_anchor_for_tile = lambda _tile, _shape: PayloadSourceAnchor(
        content_key=content_key,
        source_rect=tuple(source_rect),
        plane_shape=(8, 6),
    )
    predecessor_key = effects.render_lod.page_set_key_for_tile(
        session,
        tile,
        demand=demand,
        level=1,
    )
    _admit_page_set(
        session.lod_page_cache,
        predecessor_key,
        np.arange(24, dtype=np.float32).reshape(4, 6),
    )
    session.lifecycle.level_claimed(0, predecessor_key, ClaimOwner.EVALUATION)
    session.lifecycle.level_resident(0, predecessor_key)
    source_rect[:] = (2, 6, 0, 6)
    session._best_floor_key = lambda source_index, tile_number=None: (
        effects.render_lod.best_floor_key(
            session,
            source_index,
            tile_number=tile_number,
        )
    )
    session.pending_payload_upserts = {}
    session.dirty_payloads = {}

    state = effects.tile_lod_states(session, demand, tile_numbers=(0,))[0]
    steps = LodLadder(LadderPolicy()).plan(
        (state,),
        demand,
        preview_level=1,
        target_level=1,
    )

    assert effects.render_lod.page_set_source_rect(predecessor_key) == (0, 4, 0, 6)
    assert state.resident_levels == ()
    assert state.floor_available is False
    assert len(steps) == 1
    assert steps[0].rung is Rung.DESIRED
    assert steps[0].presentation_only is False


def test_pipeline_effects_tile_states_uses_lifecycle_snapshot():
    from arrayscope.window.frame_effects import FramePipelineEffects

    session = _session()
    level_key = _page_set(tile=0, level=2)
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


def test_backend_upload_preparation_is_submitted_as_superseded_prefetch():
    """Admission schedules pure preparation; it never runs inline or gates display."""

    from arrayscope.window.frame_effects import FramePipelineEffects

    mailbox = PreparedUploadMailbox()
    submitted = []
    payload = SimpleNamespace(tile_number=2, tile_identity="identity", source_id="source")
    preparation_key = ("identity", "source", "variant")
    view = SimpleNamespace(
        preparedTiledUploads=mailbox,
        tiledPayloadResident=lambda _payload: False,
        tiledUploadPreparations=lambda payloads, **_kwargs: (
            (
                2,
                preparation_key,
                lambda: mailbox.publish(2, preparation_key, np.ones((2, 2), np.float32)),
            ),
        ),
    )
    kernel = SimpleNamespace(submit=submitted.append)
    session = SimpleNamespace(
        session_id=7,
        shader_display=True,
        level_generation=SimpleNamespace(target_levels=(0.0, 1.0)),
        lifecycle=SimpleNamespace(current_presentable_payload=lambda tile: payload),
        display_tile_payloads={},
    )
    effects_bridge = FramePipelineEffects(
        SimpleNamespace(win=SimpleNamespace(img_view=view, kernel=kernel)),
        session,
    )

    effects_bridge._prepare_backend_uploads(
        ((SimpleNamespace(tile_number=2), object()),),
    )

    assert mailbox.counters().published == 0
    assert len(submitted) == 1
    spec = submitted[0]
    # Non-visible is the load-bearing property, not the lane name: it is what
    # subjects preparation to the kernel's speculative governor, so that a
    # started task can no longer hold a worker a pixel-producing task wants.
    assert spec.lane is Lane.SPECULATIVE_RESIDENCY
    assert not spec.visible
    assert spec.priority is Priority.PREFETCH
    assert spec.supersession.family == ("prepared-upload", 7, 2)
    assert spec.supersession.value == preparation_key
    assert spec.session_id == 7
    assert spec.tile_number == 2
    spec.fn()
    assert mailbox.take(2, preparation_key) is not None


def test_backend_upload_preparation_skips_physically_resident_payload():
    """A resident rebind needs no producer and must not inflate visible work."""

    from arrayscope.window.frame_effects import FramePipelineEffects

    submitted = []
    payload = SimpleNamespace(tile_number=2, tile_identity="identity", source_id="source")
    view = SimpleNamespace(
        preparedTiledUploads=PreparedUploadMailbox(),
        tiledPayloadResident=lambda candidate: candidate is payload,
        tiledUploadPreparations=lambda *_args, **_kwargs: pytest.fail(
            "resident payload must be filtered before preparation planning"
        ),
    )
    session = SimpleNamespace(
        session_id=7,
        shader_display=True,
        level_generation=SimpleNamespace(target_levels=(0.0, 1.0)),
        lifecycle=SimpleNamespace(current_presentable_payload=lambda tile: payload),
        display_tile_payloads={},
    )
    effects_bridge = FramePipelineEffects(
        SimpleNamespace(
            win=SimpleNamespace(
                img_view=view,
                kernel=SimpleNamespace(submit=submitted.append),
            )
        ),
        session,
    )

    effects_bridge._prepare_backend_uploads(
        ((SimpleNamespace(tile_number=2), object()),),
    )

    assert submitted == []


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


def test_presentation_commit_holds_camera_replay_while_coverage_is_open():
    """Phase 1 completes before the camera rescope replays (two-phase contract)."""

    from arrayscope.window.frame_effects import _finish_presentation_commit

    scheduled = []
    session = SimpleNamespace(
        scheduling_policy=SimpleNamespace(
            verdict=SimpleNamespace(coverage_open=True),
        ),
    )
    renderer = SimpleNamespace(
        _montage_presentation_commit_active=True,
        _frame_viewport_retarget_after_commit=True,
        _frame_session=session,
        _schedule_frame_viewport_update=lambda *, delay_ms=None: scheduled.append(delay_ms),
    )

    _finish_presentation_commit(renderer)

    assert renderer._montage_presentation_commit_active is False
    assert renderer._frame_viewport_retarget_after_commit is True
    assert scheduled == []

    session.scheduling_policy.verdict = SimpleNamespace(coverage_open=False)
    renderer._montage_presentation_commit_active = True
    _finish_presentation_commit(renderer)

    assert renderer._frame_viewport_retarget_after_commit is False
    assert scheduled == [1]


def test_chunk_summary_levels_share_complex_shader_mapping():
    values = np.asarray(
        [[1.0 + 10.0j, 2.0 + 20.0j], [3.0 + 30.0j, 4.0 + 40.0j]],
        dtype=np.complex64,
    )
    plans = plan_source_grid_pages(
        content_key="complex-summary",
        valid_source_rect_yx=(0, 2, 0, 2),
        reduction_yx=(0, 0),
        stored_page_shape=(2, 2),
        dtype="complex64",
        representation="complex_rg32f",
        reducer="native",
    )
    pages = tuple(
        materialize_lod_page(values, source_origin_yx=(0, 0), plan=plan) for plan in plans
    )

    real = effects.chunk_level_stats_for_pages(
        pages,
        source_index=7,
        mapping=ShaderMapping(component=ShaderComponent.REAL),
    )
    magnitude = effects.chunk_level_stats_for_pages(
        pages,
        source_index=7,
        mapping=ShaderMapping(component=ShaderComponent.ABS),
    )

    assert real.bounds == (1.0, 4.0)
    assert np.allclose(
        magnitude.bounds,
        (float(np.abs(values).min()), float(np.abs(values).max())),
    )
    assert real.sample.size <= 512
    assert magnitude.sample.size <= 512


def test_chunk_summary_levels_reuse_identity_mapped_scalar_summary(monkeypatch):
    values = np.arange(4, dtype=np.float32).reshape(2, 2)
    plans = plan_source_grid_pages(
        content_key="scalar-summary",
        valid_source_rect_yx=(0, 2, 0, 2),
        reduction_yx=(0, 0),
        stored_page_shape=(2, 2),
        dtype="float32",
        representation="scalar_r32f",
        reducer="native",
    )
    pages = tuple(
        materialize_lod_page(values, source_origin_yx=(0, 0), plan=plan) for plan in plans
    )
    monkeypatch.setattr(
        effects,
        "summarize_chunk",
        lambda *_args, **_kwargs: pytest.fail("identity mapping must reuse the resident summary"),
    )

    stats = effects.chunk_level_stats_for_pages(
        pages,
        source_index=3,
        mapping=ShaderMapping(component=ShaderComponent.REAL),
    )

    assert stats.bounds == (0.0, 3.0)


def test_present_tile_delta_declines_when_the_presenter_reported_a_failure():
    """A backend commit that raised is caught and reported by the presenter,
    which returns False.  Declining here is what keeps the caller from
    acknowledging the delta against whatever report the PREVIOUS transaction
    left on the committer."""

    from arrayscope.window.frame_effects import FramePipelineEffects

    session = _session()
    session.window_mode = "auto"
    session.frame_plan = None
    session.key = ("session",)
    session.render_generation = 1
    session.level_key = ("levels",)
    session.applied_level_source = None
    session.defer_side_panels = False

    renderer = SimpleNamespace(
        win=SimpleNamespace(_committed_display_frame=None),
        _apply_full_display_image=lambda *_args, **_kwargs: False,
        _last_montage_tile_layer_apply_ms=0.0,
    )
    effects_bridge = FramePipelineEffects(renderer, session)
    effects_bridge._configure_wgpu_evidence_obligation = lambda *_a, **_k: None
    effects_bridge._commit_direct_delta = lambda *_a, **_k: False

    applied = effects_bridge._present_tile_delta(
        None,
        None,
        tile_state=None,
        base_tile_state=None,
        tile_delta=None,
        semantic_source=None,
        applied_level_source=None,
        histogram_plot_data=None,
        first_display_commit=True,
        explicit_auto=False,
        requested_levels=None,
        semantic_commit=True,
        decision_force_auto=False,
    )

    assert applied is False


def test_present_tile_delta_reports_applied_when_the_presenter_succeeded():
    from arrayscope.window.frame_effects import FramePipelineEffects

    session = _session()
    session.window_mode = "auto"
    session.frame_plan = None
    session.key = ("session",)
    session.render_generation = 1
    session.level_key = ("levels",)
    session.applied_level_source = None
    session.defer_side_panels = False

    renderer = SimpleNamespace(
        win=SimpleNamespace(_committed_display_frame=None),
        _apply_full_display_image=lambda *_args, **_kwargs: True,
        _last_montage_tile_layer_apply_ms=0.0,
    )
    effects_bridge = FramePipelineEffects(renderer, session)
    effects_bridge._configure_wgpu_evidence_obligation = lambda *_a, **_k: None
    effects_bridge._commit_direct_delta = lambda *_a, **_k: False

    applied = effects_bridge._present_tile_delta(
        None,
        None,
        tile_state=None,
        base_tile_state=None,
        tile_delta=None,
        semantic_source=None,
        applied_level_source=None,
        histogram_plot_data=None,
        first_display_commit=True,
        explicit_auto=False,
        requested_levels=None,
        semantic_commit=True,
        decision_force_auto=False,
    )

    assert applied is True


def test_cpu_composited_rgb_display_retains_complex_reduced_coarse_rung():
    """The preview route bypasses PyQtGraph's final CPU-RGB payload shape.

    This deliberately replaces the former deferred-behavior assertion.  The
    compact preview atlas now retains reduced complex source pages and performs
    CPU composition at the round levels, so a CPU-mapped complex view remains
    eligible for the same reduced coarse rung as the shader backend.
    """

    session = _session()
    session.document = ArrayDocument(session.document.base_data, operations=(CenteredFFT(axis=2),))
    tile = session.plan.tiles[0]

    # Shader-mapped complex (wgpu): the payload stays complex, so the rung is
    # legal and this is the configuration ADR 0059 measured.
    session.rgb = True
    session.shader_display = True
    assert effects.display_output_is_composited_rgb(session) is False
    assert effects.can_evaluate_reduced_preview(session, tile) is True

    # CPU-mapped complex (PyQtGraph): final output is composited RGB, while the
    # preview payload remains a reduced complex plane.
    session.shader_display = False
    assert effects.display_output_is_composited_rgb(session) is True
    assert effects.can_evaluate_reduced_preview(session, tile) is True

    # Scalar data on the same CPU-mapping backend is unaffected.
    session.rgb = False
    assert effects.display_output_is_composited_rgb(session) is False
    assert effects.can_evaluate_reduced_preview(session, tile) is True
