"""Canonical G5 page route, reducers, and logical-cache ownership."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.display import pyramid as pyramid_core
from arrayscope.core.view_state import ViewState
from arrayscope.display.pyramid import (
    LodPageCache,
    MaterializedLodPage,
    materialize_lod_page,
    materialize_source_grid_pages,
    plan_source_grid_pages,
    reduce_source_grid,
    reduction_xy_to_yx,
    reduction_yx_to_xy,
)
from arrayscope.display.lod import LodInfo
from arrayscope.display.backends.pyqtgraph.tiles import (
    MontageTileLayer,
    _assemble_page_backed_payload,
    _payload_rgb_already_windowed,
    _resolve_page_backed_payload,
)
from arrayscope.display.backends.vispy.tiles import TextureAtlasPool, _payload_mode
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.montage import MontageTile, RenderedTile
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    PageBackedPresentation,
    TiledValueSource,
    display_tile_payload_has_semantics,
)
from arrayscope.gpu import PageSlot, PageTable
from arrayscope.gpu.keys import COMPLEX_RG32F, RGB8, SCALAR_R32F
from arrayscope.presentation.tile_lifecycle import TileTarget, payload_ref_from_display_payload
from arrayscope.render import lod as render_lod
from arrayscope.display.slice_engine import make_image
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    ShaderMapping,
    TexturePlaneKind,
)
from tests.display.vispy_test_utils import FakeGloo


CONTENT = ("src-anchored", ("doc", 4), ("op", "fft"))


def plan(
    rect=(3, 12, 5, 18),
    *,
    reduction=(1, 2),
    page_shape=(3, 2),
    dtype="float32",
    representation=SCALAR_R32F,
    reducer="mean",
):
    return plan_source_grid_pages(
        content_key=CONTENT,
        valid_source_rect_yx=rect,
        reduction_yx=reduction,
        stored_page_shape=page_shape,
        dtype=dtype,
        representation=representation,
        reducer=reducer,
    )


def source(rect=(3, 12, 5, 18), *, complex_values=False):
    y0, y1, x0, x1 = rect
    yy, xx = np.mgrid[y0:y1, x0:x1]
    values = (yy * 100 + xx).astype(np.float32)
    return values.astype(np.complex64) * (1 + 1j) if complex_values else values


def _rendered_complex_channel(values: np.ndarray, channel: str) -> RenderedTile:
    state = ViewState.from_shape(values.shape).with_channel(channel)
    display = make_image(values, state)
    tile = MontageTile(0, 0, 0, 0, 0, 0, values.shape[1], values.shape[0], state)
    return RenderedTile(
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
    )


def test_plan_is_the_one_owner_of_key_geometry_and_route_lineage():
    plans = plan()

    assert plans
    for page_plan in plans:
        assert page_plan.key.operation_key == ("source-grid-page", 1, ("op", "fft"))
        assert page_plan.key.lod.reduction == (1, 2)
        assert page_plan.key.chunk_origin == (
            page_plan.source_rect_yx[0],
            page_plan.source_rect_yx[2],
        )
        assert page_plan.key.chunk_shape == (
            page_plan.source_rect_yx[1] - page_plan.source_rect_yx[0],
            page_plan.source_rect_yx[3] - page_plan.source_rect_yx[2],
        )
        assert page_plan.source_samples_per_stored_sample_yx == (2, 4)
        assert page_plan.stored_shape[0] <= 3
        assert page_plan.stored_shape[1] <= 2


def test_large_route_planning_keeps_geometry_axis_compact():
    plans = plan(
        rect=(101, 8293, 102, 8294),
        reduction=(3, 2),
        page_shape=(256, 256),
    )

    stored_samples = sum(page.stored_shape[0] * page.stored_shape[1] for page in plans)
    compact_spans = sum(len(page.source_y_bins) + len(page.source_x_bins) for page in plans)
    assert stored_samples > 2_000_000
    assert compact_spans < 20_000
    assert all("sample_source_rects_yx" not in page.__dict__ for page in plans)


def test_page_materialization_uses_vectorized_axis_reduction(monkeypatch):
    rect = (101, 613, 102, 614)
    page_plans = plan(rect=rect, reduction=(3, 2), page_shape=(256, 256))
    values = source(rect)

    def reject_scalar_reduction(*_args, **_kwargs):
        raise AssertionError("materialization must not reduce one stored sample at a time")

    monkeypatch.setattr(pyramid_core, "_reduce_sample", reject_scalar_reduction)
    pages = materialize_source_grid_pages(
        values,
        source_origin_yx=(rect[0], rect[2]),
        plans=page_plans,
    )
    assert sum(page.values.size for page in pages) == 8_385


def test_named_axis_conversion_keeps_asymmetric_routes_distinct():
    assert reduction_yx_to_xy((1, 2)) == (2, 1)
    assert reduction_xy_to_yx((1, 2)) == (2, 1)

    y1_x2 = plan(reduction=(1, 2), page_shape=(8, 8))
    y2_x1 = plan(reduction=(2, 1), page_shape=(8, 8))
    assert y1_x2[0].stored_shape == (5, 4)
    assert tuple(page.stored_shape for page in y2_x1) == ((3, 6), (3, 1))
    assert y1_x2[0].source_y_bins == (
        (3, 4),
        (4, 6),
        (6, 8),
        (8, 10),
        (10, 12),
    )
    assert y1_x2[0].source_x_bins == (
        (5, 8),
        (8, 12),
        (12, 16),
        (16, 18),
    )
    assert y2_x1[0].source_y_bins == ((3, 4), (4, 8), (8, 12))
    assert tuple(
        source_bin
        for page in y2_x1
        for source_bin in page.source_x_bins
    ) == (
        (5, 6),
        (6, 8),
        (8, 10),
        (10, 12),
        (12, 14),
        (14, 16),
        (16, 18),
    )
    assert sum(
        (block.source_rect_yx[1] - block.source_rect_yx[0])
        * (block.source_rect_yx[3] - block.source_rect_yx[2])
        for page in y1_x2
        for block in page.draw_blocks
    ) == 9 * 13
    assert sum(
        (block.source_rect_yx[1] - block.source_rect_yx[0])
        * (block.source_rect_yx[3] - block.source_rect_yx[2])
        for page in y2_x1
        for block in page.draw_blocks
    ) == 9 * 13
    assert y1_x2[0].key != y2_x1[0].key


@pytest.mark.parametrize(
    ("reducer", "expected", "dtype", "representation"),
    [
        ("mean", lambda z: np.mean(z, dtype=np.complex64), np.complex64, COMPLEX_RG32F),
        ("mean_abs", lambda z: np.mean(np.abs(z), dtype=np.float32), np.float32, SCALAR_R32F),
        ("power", lambda z: np.mean(np.abs(z) ** 2, dtype=np.float32), np.float32, SCALAR_R32F),
        ("rms", lambda z: np.sqrt(np.mean(np.abs(z) ** 2, dtype=np.float32)), np.float32, SCALAR_R32F),
    ],
)
def test_reducer_families_match_direct_bin_oracles(reducer, expected, dtype, representation):
    values = np.asarray([[1 + 1j, 3 - 1j], [-1 + 2j, 5 + 0j]], dtype=np.complex64)
    result = reduce_source_grid(
        values,
        source_origin_yx=(0, 0),
        valid_source_rect_yx=(0, 2, 0, 2),
        reduction_yx=(1, 1),
        reducer=reducer,
    )

    assert result.values.dtype == np.dtype(dtype)
    np.testing.assert_allclose(result.values[0, 0], expected(values), rtol=1e-6, atol=1e-6)
    page_plan = plan_source_grid_pages(
        content_key=CONTENT,
        valid_source_rect_yx=(0, 2, 0, 2),
        reduction_yx=(1, 1),
        stored_page_shape=(2, 2),
        dtype=np.dtype(dtype).name,
        representation=representation,
        reducer=reducer,
    )[0]
    page = materialize_lod_page(values, source_origin_yx=(0, 0), plan=page_plan)
    np.testing.assert_allclose(page.values, result.values)


def test_native_route_keeps_samples_unchanged():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = reduce_source_grid(
        values,
        source_origin_yx=(5, 7),
        valid_source_rect_yx=(5, 8, 7, 11),
        reduction_yx=(0, 0),
        reducer="native",
    )
    np.testing.assert_array_equal(result.values, values)


def test_native_uint8_route_keeps_two_dimensional_values_scalar():
    values = np.arange(12, dtype=np.uint8).reshape(3, 4)
    output_dtype, representation = pyramid_core._reducer_output_format("native", values)

    assert output_dtype == np.dtype(np.uint8)
    assert representation == SCALAR_R32F

    result = reduce_source_grid(
        values,
        source_origin_yx=(5, 7),
        valid_source_rect_yx=(5, 8, 7, 11),
        reduction_yx=(0, 0),
        reducer="native",
    )

    assert result.values.dtype == np.dtype(np.uint8)
    np.testing.assert_array_equal(result.values, values)

    rendered = RenderedTile(
        tile=MontageTile(0, 0, 0, 0, 0, 0, 4, 3, ViewState.from_shape(values.shape)),
        image=values,
        histogram_data=values,
        eval_ms=0.0,
        slab_shape=values.shape,
        slab_nbytes=values.nbytes,
        semantic_data=values,
        lod_source_data=values,
    )
    demand = SimpleNamespace(desired_level=0, desired_factor_xy=(1, 1))
    key = render_lod.page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=0,
        semantic_source_id=("uint8-scalar", 0),
    )

    assert {page_plan.key.representation for page_plan in key.plans} == {SCALAR_R32F}
    pages = materialize_source_grid_pages(
        values,
        source_origin_yx=(0, 0),
        plans=key.plans,
    )
    lod = LodInfo(0, 1, values.shape, values.shape, 0)
    payload = DisplayTilePayload(
        0,
        0,
        values,
        values,
        ("uint8-scalar-payload", 0),
        semantic_data=values,
        texture_data=values,
        lod=lod,
        page_backing=PageBackedPresentation(key.plans, pages, (0, 3, 0, 4), lod),
    )
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=512)

    _uvs, stats = pool.update_payloads(
        {0: payload},
        tile_shape=values.shape,
        dirty_tiles=None,
        rgb_already_windowed=False,
        tile_world_regions={0: (0, 0, 4, 3)},
    )

    assert stats.presented_tiles == (0,)
    assert all(page.key in pool.resident_slots for page in pages)
    assert {page.storage_mode for page in pool.pages} == {"scalar"}


def test_materialized_representation_rejects_two_dimensional_values_under_rgb_key():
    values = np.arange(12, dtype=np.uint8).reshape(3, 4)
    bad_plan = plan_source_grid_pages(
        content_key=("bad-uint8-rgb",),
        valid_source_rect_yx=(0, 3, 0, 4),
        reduction_yx=(0, 0),
        stored_page_shape=(3, 4),
        dtype="uint8",
        representation=RGB8,
        reducer="native",
    )[0]

    with pytest.raises(ValueError, match="rgb8.*RGB"):
        MaterializedLodPage(bad_plan, values)


def test_live_three_dimensional_rgb_route_remains_noncanonical():
    values = np.zeros((3, 4, 3), dtype=np.uint8)
    rendered = RenderedTile(
        tile=MontageTile(0, 0, 0, 0, 0, 0, 4, 3, ViewState.from_shape(values.shape)),
        image=values,
        histogram_data=None,
        eval_ms=0.0,
        slab_shape=values.shape,
        slab_nbytes=values.nbytes,
        semantic_data=values,
        lod_source_data=values,
    )
    demand = SimpleNamespace(desired_level=0, desired_factor_xy=(1, 1))

    with pytest.raises(ValueError, match="scalar or complex"):
        render_lod.page_set_key_for_rendered(
            rendered,
            demand=demand,
            level=0,
            semantic_source_id=("rgb-noncanonical", 0),
        )


def test_phase_vector_counts_zero_as_zero_and_cancels_opposed_phase():
    values = np.asarray([[1 + 0j, -1 + 0j], [0 + 0j, 0 + 0j]], dtype=np.complex64)
    result = reduce_source_grid(
        values,
        source_origin_yx=(0, 0),
        valid_source_rect_yx=(0, 2, 0, 2),
        reduction_yx=(1, 1),
        reducer="phase_vector",
    )
    assert result.values.dtype == np.complex64
    assert result.values[0, 0] == 0j

    quarter = reduce_source_grid(
        np.asarray([[1 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]], dtype=np.complex64),
        source_origin_yx=(0, 0),
        valid_source_rect_yx=(0, 2, 0, 2),
        reduction_yx=(1, 1),
        reducer="phase_vector",
    )
    assert quarter.values[0, 0] == pytest.approx(0.25 + 0j)


def test_phase_vector_large_almost_cancelled_bin_keeps_one_unmatched_vector():
    # Independent count oracle: 2048 positive unit vectors, 2047 negative
    # unit vectors, and one counted zero leave exactly 1 / 4096.  Do not
    # compute the expectation through either reducer implementation.
    values = np.empty((64, 64), dtype=np.complex64)
    flat = values.reshape(-1)
    flat[:2048] = np.complex64(1.0 + 0.0j)
    flat[2048:4095] = np.complex64(-1.0 + 0.0j)
    flat[4095] = np.complex64(0.0j)
    expected = np.complex64(1.0 / 4096.0)

    planned = reduce_source_grid(
        values,
        source_origin_yx=(0, 0),
        valid_source_rect_yx=(0, 64, 0, 64),
        reduction_yx=(6, 6),
        reducer="phase_vector",
    )
    sampled = pyramid_core._reduce_sample(values, reducer="phase_vector")

    assert planned.values[0, 0] != 0.0j
    assert sampled != 0.0j
    assert planned.values[0, 0] == expected
    assert sampled == expected


def test_mean_abs_is_not_abs_of_complex_mean():
    values = np.asarray([[1 + 0j, -1 + 0j]], dtype=np.complex64)
    mean = reduce_source_grid(
        values,
        source_origin_yx=(0, 0),
        valid_source_rect_yx=(0, 1, 0, 2),
        reduction_yx=(0, 1),
        reducer="mean",
    )
    mean_abs = reduce_source_grid(
        values,
        source_origin_yx=(0, 0),
        valid_source_rect_yx=(0, 1, 0, 2),
        reduction_yx=(0, 1),
        reducer="mean_abs",
    )
    assert abs(mean.values[0, 0]) == 0.0
    assert mean_abs.values[0, 0] == 1.0


def test_materialized_page_rejects_shape_dtype_and_noncontiguous_values():
    page_plan = plan(rect=(0, 4, 0, 4), reduction=(1, 1), page_shape=(4, 4))[0]
    with pytest.raises(ValueError, match="shape"):
        MaterializedLodPage(page_plan, np.zeros((1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="dtype"):
        MaterializedLodPage(page_plan, np.zeros(page_plan.stored_shape, dtype=np.float64))
    noncontiguous = np.zeros((page_plan.stored_shape[1], page_plan.stored_shape[0]), dtype=np.float32).T
    assert not noncontiguous.flags.c_contiguous
    with pytest.raises(ValueError, match="contiguous"):
        MaterializedLodPage(page_plan, noncontiguous)


def test_shifted_windows_share_byte_identical_interior_page_and_distinct_boundaries():
    first_rect = (100, 104, 101, 113)
    second_rect = (100, 104, 102, 114)
    first_plans = plan(rect=first_rect, reduction=(1, 1), page_shape=(2, 3))
    second_plans = plan(rect=second_rect, reduction=(1, 1), page_shape=(2, 3))
    first_pages = materialize_source_grid_pages(
        source(first_rect), source_origin_yx=(100, 101), plans=first_plans
    )
    second_pages = materialize_source_grid_pages(
        source(second_rect), source_origin_yx=(100, 102), plans=second_plans
    )
    first_by_key = {page.key: page for page in first_pages}
    second_by_key = {page.key: page for page in second_pages}
    shared = set(first_by_key) & set(second_by_key)
    assert len(shared) == 1
    key = next(iter(shared))
    assert key.chunk_origin == (100, 102)
    assert key.chunk_shape == (4, 6)
    assert first_by_key[key].values.tobytes() == second_by_key[key].values.tobytes()
    assert len(set(first_by_key) - shared) == 2
    # Moving the clipped origin onto the global bin boundary removes the old
    # leading fragment and changes only the trailing boundary page.
    assert len(set(second_by_key) - shared) == 1


def test_logical_page_cache_singleflight_and_terminal_cleanup_are_owner_scoped():
    page_plan = plan(rect=(0, 4, 0, 4), reduction=(1, 1), page_shape=(4, 4))[0]
    page = materialize_lod_page(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        source_origin_yx=(0, 0),
        plan=page_plan,
    )
    cache = LodPageCache(max_bytes=1 << 20)

    assert cache.begin_claim(page.key, "request-a")
    assert not cache.begin_claim(page.key, "request-b")
    with pytest.raises(ValueError, match="owner mismatch"):
        cache.admit(page, owner="request-b")
    assert cache.pending(page.key)
    cache.admit(page, owner="request-a")
    assert cache.peek(page.key) is page
    assert not cache.pending(page.key)

    second = plan(rect=(0, 4, 4, 8), reduction=(1, 1), page_shape=(4, 4))[0]
    assert cache.begin_claim(second.key, "request-c")
    assert cache.release_owner_claims("request-c") == (second.key,)
    assert cache.pending_count == 0


def test_logical_page_cache_reuses_resolver_snapshot_until_residency_changes(monkeypatch):
    page_plan = plan(rect=(0, 4, 0, 4), reduction=(1, 1), page_shape=(4, 4))[0]
    page = materialize_lod_page(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        source_origin_yx=(0, 0),
        plan=page_plan,
    )
    cache = LodPageCache(max_bytes=1 << 20)
    assert cache.begin_claim(page.key, "worker")
    cache.admit(page, owner="worker")
    bind_calls = 0
    original_bind = pyramid_core.PageTable.bind

    def counted_bind(table, *args, **kwargs):
        nonlocal bind_calls
        bind_calls += 1
        return original_bind(table, *args, **kwargs)

    monkeypatch.setattr(pyramid_core.PageTable, "bind", counted_bind)
    assert cache.resolve(page.key) is not None
    assert cache.resolve(page.key) is not None
    assert cache.resolved_pages((page_plan,)) == (page,)
    assert bind_calls == 1


def test_exact_page_query_does_not_treat_coarse_ancestor_as_materialized_target():
    values = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    coarse_plan = plan(
        rect=(0, 64, 0, 64),
        reduction=(4, 4),
        page_shape=(256, 256),
    )[0]
    target_plan = plan(
        rect=(0, 64, 0, 64),
        reduction=(2, 2),
        page_shape=(256, 256),
    )[0]
    coarse_page = materialize_lod_page(
        values,
        source_origin_yx=(0, 0),
        plan=coarse_plan,
    )
    cache = LodPageCache(max_bytes=1 << 20)
    owner = ("coarse-page",)
    assert cache.begin_claim(coarse_page.key, owner)
    cache.admit(coarse_page, owner=owner)

    resolution = cache.resolve(target_plan.key)

    assert resolution is not None
    assert resolution.actual_key == coarse_page.key
    assert cache.resolved_pages((target_plan,)) == (coarse_page,)
    assert cache.exact_pages((target_plan,)) is None


def test_running_page_owner_defers_cancellation_release_until_worker_terminal():
    page_plan = plan(rect=(0, 4, 0, 4), reduction=(1, 1), page_shape=(4, 4))[0]
    page = materialize_lod_page(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        source_origin_yx=(0, 0),
        plan=page_plan,
    )
    cache = LodPageCache(max_bytes=1 << 20)
    owner = "running-worker"
    assert cache.begin_claim(page.key, owner)
    assert cache.begin_owner_work(owner)

    assert cache.release_owner_claims(owner) == ()
    assert cache.claimed_by(page.key) == owner
    cache.admit(page, owner=owner)
    assert cache.finish_owner_work(owner) == ()
    assert cache.pending_count == 0


@pytest.mark.parametrize(
    ("first_rect", "shifted_rect", "evicted_origin", "shared_origin"),
    (
        ((0, 4, 0, 8), (0, 4, 4, 12), (0, 0), (0, 4)),
        ((0, 4, 4, 12), (0, 4, 0, 8), (0, 8), (0, 4)),
    ),
)
def test_shifted_exact_set_evicts_outgoing_page_before_shared_interior(
    first_rect,
    shifted_rect,
    evicted_origin,
    shared_origin,
):
    first_plans = plan(
        rect=first_rect,
        reduction=(0, 0),
        page_shape=(4, 4),
        reducer="native",
    )
    shifted_plans = plan(
        rect=shifted_rect,
        reduction=(0, 0),
        page_shape=(4, 4),
        reducer="native",
    )
    page_bytes = 4 * 4 * np.dtype(np.float32).itemsize
    cache = LodPageCache(max_bytes=2 * page_bytes)

    def admit_set(plans, rect, owner):
        claimed = cache.claim_plans(plans, owner)
        for page_plan in claimed:
            page = materialize_lod_page(
                source(rect),
                source_origin_yx=(rect[0], rect[2]),
                plan=page_plan,
            )
            cache.admit_as(page.key, page, owner=owner)
        cache.release_owner_claims(owner)
        return claimed

    assert len(admit_set(first_plans, first_rect, "first-window")) == 2
    claimed = admit_set(shifted_plans, shifted_rect, "shifted-window")

    assert len(claimed) == 1
    assert cache.exact_pages(shifted_plans) is not None
    resident_origins = {page.key.chunk_origin for page in cache.resident_pages()}
    assert shared_origin in resident_origins
    assert evicted_origin not in resident_origins


def test_budget_impossible_exact_set_is_ineligible_until_cache_resize():
    plans = plan(
        rect=(0, 4, 0, 8),
        reduction=(0, 0),
        page_shape=(4, 4),
        reducer="native",
    )
    page_bytes = 4 * 4 * np.dtype(np.float32).itemsize
    cache = LodPageCache(max_bytes=page_bytes)

    assert cache.claim_plans(plans, "impossible") == ()
    assert cache.plan_set_ineligible(plans)
    assert cache.pending_count == 0
    assert cache.claim_plans(plans, "same-budget-replan") == ()
    assert cache.pending_count == 0

    cache.resize(max_bytes=2 * page_bytes)

    assert not cache.plan_set_ineligible(plans)
    assert cache.claim_plans(plans, "resized") == plans
    assert cache.release_owner_claims("resized") == tuple(plan.key for plan in plans)


def test_wrong_key_admission_is_loud_and_releases_the_request_claim():
    first_plan = plan(rect=(0, 4, 0, 4), reduction=(1, 1), page_shape=(4, 4))[0]
    second_plan = plan(rect=(0, 4, 4, 8), reduction=(1, 1), page_shape=(4, 4))[0]
    page = materialize_lod_page(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        source_origin_yx=(0, 0),
        plan=first_plan,
    )
    cache = LodPageCache(max_bytes=1 << 20)
    assert cache.begin_claim(second_plan.key, "worker")
    with pytest.raises(ValueError, match="wrong key"):
        cache.admit_as(second_plan.key, page, owner="worker")
    assert not cache.pending(second_plan.key)


def test_plan_validation_rejects_key_geometry_or_route_drift():
    page_plan = plan(rect=(0, 4, 0, 4), reduction=(1, 1), page_shape=(4, 4))[0]
    wrong_key = replace(page_plan.key, chunk_origin=(1, 0))
    with pytest.raises(ValueError, match="geometry"):
        replace(page_plan, key=wrong_key)


def test_page_backed_payload_validates_cover_and_counts_aliases_once():
    rect = (100, 104, 101, 113)
    plans = plan(rect=rect, reduction=(1, 1), page_shape=(2, 3))
    pages = materialize_source_grid_pages(
        source(rect), source_origin_yx=(100, 101), plans=plans
    )
    lod = LodInfo(
        level=1,
        factor=2,
        source_shape=(4, 12),
        texture_shape=(2, 7),
        gutter=0,
    )
    backing = PageBackedPresentation(plans, pages, rect, lod)
    payload = DisplayTilePayload(
        0,
        0,
        pages[0].values,
        None,
        ("tile", 0),
        texture_data=pages[0].values,
        semantic_data=None,
        lod=lod,
        page_backing=backing,
    )
    expected = sum(page.nbytes for page in pages)
    assert payload.nbytes == expected

    with pytest.raises(ValueError, match="gap"):
        PageBackedPresentation(plans[:-1], pages[:-1], rect, lod)
    with pytest.raises(ValueError, match="duplicate"):
        PageBackedPresentation((*plans, plans[0]), pages, rect, lod)
    with pytest.raises(ValueError, match="source shape"):
        PageBackedPresentation(
            plans,
            pages,
            rect,
            replace(lod, source_shape=(8, 12)),
        )
    with pytest.raises(ValueError, match="texture shape"):
        PageBackedPresentation(
            plans,
            pages,
            rect,
            replace(lod, texture_shape=(2, 8)),
        )
    with pytest.raises(ValueError, match="semantic LOD"):
        PageBackedPresentation(
            plans,
            pages,
            rect,
            replace(lod, level=99),
        )
    with pytest.raises(ValueError, match="source shape"):
        DisplayTilePayload(
            0,
            0,
            pages[0].values,
            None,
            ("wrong-actual-source-shape", 0),
            lod=replace(lod, source_shape=(8, 12)),
            page_backing=backing,
        )
    with pytest.raises(ValueError, match="texture shape"):
        DisplayTilePayload(
            0,
            0,
            pages[0].values,
            None,
            ("wrong-actual-texture-shape", 0),
            lod=replace(lod, texture_shape=(2, 8)),
            page_backing=backing,
        )


def test_page_backed_presentation_probe_uses_exact_clipped_bin_geometry_without_semantic_admission():
    rect = (100, 104, 101, 113)
    plans = plan(rect=rect, reduction=(1, 1), page_shape=(2, 3))
    pages = materialize_source_grid_pages(
        source(rect), source_origin_yx=(100, 101), plans=plans
    )
    lod = LodInfo(1, 2, (4, 12), (2, 7), 0)
    backing = PageBackedPresentation(plans, pages, rect, lod)
    payload = DisplayTilePayload(
        0,
        0,
        pages[0].values,
        None,
        ("page-backed", 0),
        semantic_data=None,
        lod=lod,
        page_backing=backing,
    )
    semantic_values = TiledValueSource({0: payload})

    # x=101 is the one-sample clipped leading bin; x=102 begins the next
    # full factor-two bin. Equal-width arithmetic would map one of these
    # coordinates incorrectly.
    first = backing.sample_presented_value_at_native(100, 101)
    second = backing.sample_presented_value_at_native(100, 102)
    assert first == pytest.approx((10101 + 10201) / 2)
    assert second == pytest.approx((10102 + 10103 + 10202 + 10203) / 4)
    assert semantic_values.value_at(
        SimpleNamespace(tile_number=0, local_y=0, local_x=0)
    ) is None
    assert semantic_values.tile_region(
        SimpleNamespace(tile_number=0),
        (slice(0, 4), slice(0, 12)),
    ) is None
    assert not display_tile_payload_has_semantics(payload)


def test_page_backed_payload_uses_only_explicit_native_planes_for_semantic_reads():
    rect = (100, 104, 101, 113)
    native = source(rect)
    plans = plan(rect=rect, reduction=(1, 1), page_shape=(2, 3))
    pages = materialize_source_grid_pages(
        native,
        source_origin_yx=(rect[0], rect[2]),
        plans=plans,
    )
    lod = LodInfo(1, 2, native.shape, (2, 7), 0)
    semantic_histogram = native + np.float32(1_000_000.0)
    payload = DisplayTilePayload(
        0,
        0,
        pages[0].values,
        None,
        ("page-backed-with-native-semantics", 0),
        semantic_data=native,
        semantic_histogram_data=semantic_histogram,
        lod=lod,
        page_backing=PageBackedPresentation(plans, pages, rect, lod),
    )
    values = TiledValueSource({0: payload})

    assert display_tile_payload_has_semantics(payload)
    assert values.value_at(
        SimpleNamespace(tile_number=0, local_y=3, local_x=11)
    ) == semantic_histogram[3, 11]
    region, histogram, kind = values.tile_region(
        SimpleNamespace(tile_number=0),
        (slice(1, 4), slice(2, 9)),
    )
    np.testing.assert_array_equal(region, native[1:4, 2:9])
    np.testing.assert_array_equal(histogram, semantic_histogram[1:4, 2:9])
    assert kind == "committed_tile_payload"


def test_page_backed_presentation_sample_crosses_clipped_pages_without_becoming_semantic():
    rect = (100, 105, 101, 114)
    plans = plan(rect=rect, reduction=(1, 1), page_shape=(2, 2))
    pages = materialize_source_grid_pages(
        source(rect), source_origin_yx=(100, 101), plans=plans
    )
    assert len(plans) > 2
    texture_shape = (
        max(item.stored_rect_yx[1] for item in plans)
        - min(item.stored_rect_yx[0] for item in plans),
        max(item.stored_rect_yx[3] for item in plans)
        - min(item.stored_rect_yx[2] for item in plans),
    )
    lod = LodInfo(1, 2, (5, 13), texture_shape, 0)
    backing = PageBackedPresentation(plans, pages, rect, lod)
    # Presentation probes assemble every page through planned clipped bins;
    # they do not become committed semantic reads by doing so.
    payload = DisplayTilePayload(
        0,
        0,
        pages[0].values,
        None,
        ("page-backed-presentation-sample", 0),
        semantic_data=None,
        semantic_histogram_data=None,
        lod=lod,
        page_backing=backing,
    )
    region = backing.sample_presented_values_at_native_coordinates(
        np.arange(rect[0], rect[1], dtype=np.int64),
        np.arange(rect[2], rect[3], dtype=np.int64),
    )
    expected = np.empty((5, 13), dtype=np.float32)
    for page in pages:
        for row, (y0, y1) in enumerate(page.plan.source_y_bins):
            for column, (x0, x1) in enumerate(page.plan.source_x_bins):
                expected[
                    y0 - rect[0] : y1 - rect[0],
                    x0 - rect[2] : x1 - rect[2],
                ] = page.values[row, column]

    np.testing.assert_array_equal(region, expected)
    assert region.shape == (5, 13)
    assert TiledValueSource({0: payload}).tile_region(
        SimpleNamespace(montage_index=0),
        (slice(0, 5), slice(0, 13)),
    ) is None


def test_incomplete_page_backing_refuses_presentation_sample_and_semantic_source_stays_empty():
    rect = (0, 8, 0, 8)
    plans = plan(rect=rect, reduction=(1, 1), page_shape=(2, 2))
    first_page = materialize_lod_page(
        source(rect), source_origin_yx=(0, 0), plan=plans[0]
    )
    lod = LodInfo(1, 2, (8, 8), (4, 4), 0)
    payload = DisplayTilePayload(
        0,
        0,
        first_page.values,
        None,
        ("incomplete-presentation-sample", 0),
        semantic_data=None,
        semantic_histogram_data=None,
        lod=lod,
        page_backing=PageBackedPresentation(plans, (first_page,), rect, lod),
    )

    with pytest.raises(RuntimeError, match="incomplete page-backed coverage"):
        payload.page_backing.sample_presented_values_at_native_coordinates(
            np.arange(0, 8, dtype=np.int64),
            np.arange(0, 8, dtype=np.int64),
        )
    assert TiledValueSource({0: payload}).tile_region(
        SimpleNamespace(montage_index=0),
        (slice(0, 8), slice(0, 8)),
    ) is None


def test_page_backed_complex_presentation_sample_maps_all_pages_without_semantic_admission():
    rect = (100, 104, 101, 110)
    plans = plan(
        rect=rect,
        reduction=(1, 1),
        page_shape=(1, 2),
        dtype="complex64",
        representation=COMPLEX_RG32F,
    )
    pages = materialize_source_grid_pages(
        source(rect, complex_values=True),
        source_origin_yx=(100, 101),
        plans=plans,
    )
    texture_shape = (
        max(item.stored_rect_yx[1] for item in plans)
        - min(item.stored_rect_yx[0] for item in plans),
        max(item.stored_rect_yx[3] for item in plans)
        - min(item.stored_rect_yx[2] for item in plans),
    )
    lod = LodInfo(1, 2, (4, 9), texture_shape, 0)
    payload = DisplayTilePayload(
        0,
        0,
        pages[0].values,
        np.zeros(pages[0].values.shape, dtype=np.float32),
        ("page-backed-complex-presentation-sample", 0),
        semantic_data=None,
        semantic_histogram_data=None,
        lod=lod,
        shader_mapping=ShaderMapping(component=ShaderComponent.ABS),
        page_backing=PageBackedPresentation(plans, pages, rect, lod),
    )

    presented = payload.page_backing.sample_presented_values_at_native_coordinates(
        np.arange(rect[0], rect[1], dtype=np.int64),
        np.arange(rect[2], rect[3], dtype=np.int64),
    )

    assert presented.shape == (4, 9)
    assert np.iscomplexobj(presented)
    assert np.count_nonzero(np.abs(presented)) == presented.size
    assert TiledValueSource({0: payload}).tile_region(
        SimpleNamespace(tile_number=0),
        (slice(0, 4), slice(0, 9)),
    ) is None


def test_pyqtgraph_page_assembly_matches_exact_source_grid_nearest_oracle():
    rect = (100, 104, 101, 113)
    plans = plan(rect=rect, reduction=(1, 1), page_shape=(2, 3))
    pages = materialize_source_grid_pages(
        source(rect), source_origin_yx=(100, 101), plans=plans
    )
    lod = LodInfo(1, 2, (4, 12), (2, 7), 0)
    payload = DisplayTilePayload(
        0,
        0,
        pages[0].values,
        None,
        ("page-backed", 0),
        semantic_data=None,
        lod=lod,
        page_backing=PageBackedPresentation(plans, pages, rect, lod),
    )
    assembled = _assemble_page_backed_payload(payload)
    expected = np.empty((4, 12), dtype=np.float32)
    for page in pages:
        for index, (y0, y1, x0, x1) in enumerate(page.plan.sample_source_rects_yx):
            row, column = divmod(index, page.plan.stored_shape[1])
            expected[y0 - rect[0] : y1 - rect[0], x0 - rect[2] : x1 - rect[2]] = (
                page.values[row, column]
            )
    np.testing.assert_array_equal(assembled.image, expected)
    assert assembled.lod == lod


def _expanded_direct_oracle(reduction):
    y0, y1, x0, x1 = reduction.valid_source_rect_yx
    expanded = np.empty((y1 - y0, x1 - x0), dtype=reduction.values.dtype)
    for value, (by0, by1, bx0, bx1) in zip(
        reduction.values.reshape(-1),
        reduction.source_rects,
        strict=True,
    ):
        expanded[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0] = value
    return expanded


def _assert_vispy_scalar_pages_match_oracle(pool, pages, reduction):
    values_by_rect = dict(
        zip(reduction.source_rects, reduction.values.reshape(-1), strict=True)
    )
    for materialized in pages:
        page_index, slot_index = pool.resident_slots[materialized.key]
        atlas = pool.pages[page_index]
        offset = atlas.offset_for_slot(slot_index)
        uploads = [
            values
            for upload_offset, values in atlas.scalar_texture.uploads
            if upload_offset == offset
        ]
        assert uploads, f"VisPy did not upload canonical page {materialized.key!r}"
        expected = np.asarray(
            [values_by_rect[rect] for rect in materialized.plan.sample_source_rects_yx],
            dtype=reduction.values.dtype,
        ).reshape(materialized.plan.stored_shape)
        actual = np.asarray(uploads[-1])
        if actual.ndim == 3 and actual.shape[-1] == 1:
            actual = actual[..., 0]
        np.testing.assert_allclose(
            actual[: expected.shape[0], : expected.shape[1]],
            expected,
        )


def test_clipped_page_backed_backends_share_direct_oracle_and_exact_draw_geometry():
    rect = (100, 105, 101, 114)
    values = source(rect)
    reduction = (1, 1)
    oracle = reduce_source_grid(
        values,
        content_key=CONTENT,
        source_origin_yx=(rect[0], rect[2]),
        valid_source_rect_yx=rect,
        reduction_yx=reduction,
        reducer="mean",
    )
    plans = plan(rect=rect, reduction=reduction, page_shape=(2, 2))
    pages = materialize_source_grid_pages(
        values,
        source_origin_yx=(rect[0], rect[2]),
        plans=plans,
    )
    lod = LodInfo(1, 2, values.shape, oracle.values.shape, 0)
    payload = DisplayTilePayload(
        0,
        0,
        pages[0].values,
        None,
        ("clipped-backend-parity", 0),
        semantic_data=None,
        lod=lod,
        page_backing=PageBackedPresentation(plans, pages, rect, lod),
    )
    expected = _expanded_direct_oracle(oracle)

    pyqtgraph = _resolve_page_backed_payload(payload)
    np.testing.assert_array_equal(pyqtgraph.payload.image, expected)
    assert tuple(item.actual_key for item in pyqtgraph.resolutions) == tuple(
        page_plan.key for page_plan in plans
    )

    world_x, world_y = (7, 11)
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=16)
    _uvs, stats = pool.update_payloads(
        {0: payload},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=1,
        tile_world_regions={0: (world_x, world_y, values.shape[1], values.shape[0])},
    )
    assert stats.presented_tiles == (0,)
    resolutions = pool.tile_page_target_resolutions[0]
    assert tuple(item.target_key for item in resolutions) == tuple(
        page_plan.key for page_plan in plans
    )
    assert all(item.actual_key == item.target_key for item in resolutions)
    _assert_vispy_scalar_pages_match_oracle(pool, pages, oracle)

    parts = pool.tile_draw_parts[0]
    assert parts
    source_y_edges = {edge for source_rect in oracle.source_rects for edge in source_rect[:2]}
    source_x_edges = {edge for source_rect in oracle.source_rects for edge in source_rect[2:]}
    assert sum(
        (part.world_rect[2] - part.world_rect[0])
        * (part.world_rect[3] - part.world_rect[1])
        for part in parts
    ) == pytest.approx(values.size)
    for part in parts:
        native_x0 = part.world_rect[0] - world_x + rect[2]
        native_y0 = part.world_rect[1] - world_y + rect[0]
        native_x1 = part.world_rect[2] - world_x + rect[2]
        native_y1 = part.world_rect[3] - world_y + rect[0]
        assert {native_y0, native_y1}.issubset(source_y_edges)
        assert {native_x0, native_x1}.issubset(source_x_edges)
    for by0, by1, bx0, bx1 in oracle.source_rects:
        expected_world = (
            world_x + bx0 - rect[2],
            world_y + by0 - rect[0],
            world_x + bx1 - rect[2],
            world_y + by1 - rect[0],
        )
        assert sum(
            int(
                part.world_rect[0] <= expected_world[0]
                and part.world_rect[1] <= expected_world[1]
                and expected_world[2] <= part.world_rect[2]
                and expected_world[3] <= part.world_rect[3]
            )
            for part in parts
        ) == 1


def test_mean_abs_page_values_match_both_backends_and_never_alias_complex_mean():
    rect = (0, 4, 0, 4)
    values = np.asarray(
        [
            [1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 2.0j, 0.0 - 2.0j],
            [1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 2.0j, 0.0 - 2.0j],
            [3.0 + 0.0j, -3.0 + 0.0j, 0.0 + 4.0j, 0.0 - 4.0j],
            [3.0 + 0.0j, -3.0 + 0.0j, 0.0 + 4.0j, 0.0 - 4.0j],
        ],
        dtype=np.complex64,
    )
    mean_abs_oracle = reduce_source_grid(
        values,
        content_key=CONTENT,
        source_origin_yx=(0, 0),
        valid_source_rect_yx=rect,
        reduction_yx=(1, 1),
        reducer="mean_abs",
    )
    mean_oracle = reduce_source_grid(
        values,
        content_key=CONTENT,
        source_origin_yx=(0, 0),
        valid_source_rect_yx=rect,
        reduction_yx=(1, 1),
        reducer="mean",
    )
    assert np.any(mean_abs_oracle.values != np.abs(mean_oracle.values))

    mean_abs_plans = plan_source_grid_pages(
        content_key=CONTENT,
        valid_source_rect_yx=rect,
        reduction_yx=(1, 1),
        stored_page_shape=(2, 2),
        dtype="float32",
        representation=SCALAR_R32F,
        reducer="mean_abs",
    )
    mean_plans = plan_source_grid_pages(
        content_key=CONTENT,
        valid_source_rect_yx=rect,
        reduction_yx=(1, 1),
        stored_page_shape=(2, 2),
        dtype="complex64",
        representation=COMPLEX_RG32F,
        reducer="mean",
    )
    mean_abs_pages = materialize_source_grid_pages(
        values,
        source_origin_yx=(0, 0),
        plans=mean_abs_plans,
    )
    mean_pages = materialize_source_grid_pages(
        values,
        source_origin_yx=(0, 0),
        plans=mean_plans,
    )
    assert mean_abs_pages[0].key != mean_pages[0].key
    assert mean_abs_pages[0].key.lod.reducer == "mean_abs"
    assert mean_pages[0].key.lod.reducer == "mean"

    lod = LodInfo(1, 2, values.shape, mean_abs_oracle.values.shape, 0)
    payload = DisplayTilePayload(
        0,
        0,
        mean_abs_pages[0].values,
        None,
        ("mean-abs-backend-parity", 0),
        texture_data=mean_abs_pages[0].values,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=None,
        lod=lod,
        page_backing=PageBackedPresentation(mean_abs_plans, mean_abs_pages, rect, lod),
    )
    expected = _expanded_direct_oracle(mean_abs_oracle)
    pyqtgraph = _resolve_page_backed_payload(payload)
    np.testing.assert_array_equal(pyqtgraph.payload.image, expected)

    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    _uvs, stats = pool.update_payloads(
        {0: payload},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=1,
        tile_world_regions={0: (0, 0, 4, 4)},
    )
    assert stats.presented_tiles == (0,)
    assert all(
        resolution.actual_key == resolution.target_key
        for resolution in pool.tile_page_target_resolutions[0]
    )
    _assert_vispy_scalar_pages_match_oracle(pool, mean_abs_pages, mean_abs_oracle)

    table = PageTable()
    table.bind(mean_pages[0].key, PageSlot("mean-family", 0, 0), nbytes=mean_pages[0].nbytes)
    assert table.resolve(mean_abs_plans[0].key) is None
    assert _vispy_resolve_materialized_pages(mean_abs_plans[0], mean_pages) is None
    with pytest.raises(ValueError, match="do not belong"):
        PageBackedPresentation(mean_abs_plans, mean_pages, rect, lod)


def _anisotropic_page_candidates(*, candidate_reducers=("mean", "mean")):
    rect = (0, 8, 0, 8)
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    target = plan_source_grid_pages(
        content_key=CONTENT,
        valid_source_rect_yx=rect,
        reduction_yx=(1, 1),
        stored_page_shape=(4, 4),
        dtype="float32",
        representation=SCALAR_R32F,
        reducer="mean",
    )[0]
    one_by_three = plan_source_grid_pages(
        content_key=CONTENT,
        valid_source_rect_yx=rect,
        reduction_yx=(1, 3),
        stored_page_shape=(4, 1),
        dtype="float32",
        representation=SCALAR_R32F,
        reducer=candidate_reducers[0],
    )[0]
    two_by_two = plan_source_grid_pages(
        content_key=CONTENT,
        valid_source_rect_yx=rect,
        reduction_yx=(2, 2),
        stored_page_shape=(2, 2),
        dtype="float32",
        representation=SCALAR_R32F,
        reducer=candidate_reducers[1],
    )[0]
    pages = tuple(
        materialize_lod_page(values, source_origin_yx=(0, 0), plan=page_plan)
        for page_plan in (one_by_three, two_by_two)
    )
    return rect, values, target, pages


def _vispy_resolve_materialized_pages(target, pages):
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    payloads = {
        index: DisplayTilePayload(
            index,
            index,
            page.values,
            None,
            page.key,
        )
        for index, page in enumerate(pages)
    }
    pool.update_payloads(
        payloads,
        tile_shape=pages[0].values.shape[:2],
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=len(payloads),
    )
    return pool.resolve_page_targets({17: target.key})[17]


def test_pyqtgraph_anisotropic_resolution_matches_page_table_and_vispy(qt_app):
    """The backend must not own an anisotropic tie-break rank.

    Both candidates have reduction-step sum four.  PyQtGraph's deleted
    ``(sum(reduction), reduction)`` rank selected ``(1, 3)``; the canonical
    componentwise rank minimizes the worst delta and selects ``(2, 2)``.
    """

    rect, _values, target, pages = _anisotropic_page_candidates()
    one_by_three, two_by_two = pages
    table = PageTable()
    for index, page in enumerate(pages):
        table.bind(page.key, PageSlot("test-pages", 0, index), nbytes=page.nbytes)
    canonical = table.resolve(target.key)
    assert canonical is not None
    assert canonical.actual_key == two_by_two.key
    assert canonical.actual_key != one_by_three.key

    requested_lod = LodInfo(1, 2, (8, 8), target.stored_shape, 0)
    payload = DisplayTilePayload(
        0,
        0,
        pages[0].values,
        None,
        ("anisotropic-page-backed", 0),
        semantic_data=None,
        lod=requested_lod,
        page_backing=PageBackedPresentation(
            (target,),
            pages,
            rect,
            requested_lod,
        ),
    )
    assert payload.lod.level == 1
    assert payload.page_backing.requested_lod.level == 1
    assert payload.page_backing.resolved_page_set.actual_levels == (2,)
    pyqtgraph = _resolve_page_backed_payload(payload)
    assert len(pyqtgraph.resolutions) == 1
    assert pyqtgraph.resolutions[0].actual_key == canonical.actual_key
    assert pyqtgraph.resolutions[0].scale == canonical.scale == (0.5, 0.5)

    vispy = _vispy_resolve_materialized_pages(target, pages)
    assert vispy is not None
    assert vispy.actual_key == canonical.actual_key
    assert vispy.scale == canonical.scale

    expected = np.repeat(np.repeat(two_by_two.values, 4, axis=0), 4, axis=1)
    np.testing.assert_array_equal(pyqtgraph.payload.image, expected)

    class Owner:
        def add_tile_item(self, *_args):
            pass

        def remove_tile_item(self, *_args):
            pass

        def move_tile_item(self, *_args):
            pass

    def set_image(item, values, image_levels, **_kwargs):
        item.setImage(values, autoLevels=False, levels=image_levels)

    layer = MontageTileLayer(
        Owner(),
        set_image_item_data=set_image,
        record_upload_timing=lambda *_args: None,
        histogram_levels_for_display=lambda image_levels: image_levels,
        is_rgb_image=lambda values: np.asarray(values).ndim == 3,
    )
    geometry = DisplayGeometry(
        view_state=None,
        display_shape=(8, 8),
        montage=MontageGeometry(
            indices=(0,),
            tile_shape=(8, 8),
            columns=1,
            rows=1,
            gap=0,
        ),
    )
    delta = SimpleNamespace(
        upserts={0: payload},
        active_tiles=(0,),
        target_identities={},
        removals=(),
        near_tile_source_ids={},
        cold_deadline_ms=None,
    )
    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 64.0),
        rgb_already_windowed=False,
        dirty_tiles=None,
        tile_payloads={0: payload},
        tile_delta=delta,
    )
    assert stats.presented_tiles == (0,)
    bindings = layer.tile_truth_physical_rows()[0]["physical_page_bindings"]
    assert len(bindings) == 1
    assert bindings[0]["target_key"] == target.key
    assert bindings[0]["actual_key"] == canonical.actual_key
    assert bindings[0]["actual_lod"].reduction == (2, 2)
    assert bindings[0]["quality"] == "fallback"
    assert bindings[0]["scale"] == (0.5, 0.5)

    warm_layer = MontageTileLayer(
        Owner(),
        set_image_item_data=set_image,
        record_upload_timing=lambda *_args: None,
        histogram_levels_for_display=lambda image_levels: image_levels,
        is_rgb_image=lambda values: np.asarray(values).ndim == 3,
    )
    warm = warm_layer.warm_payloads(
        {0: payload},
        geometry=geometry,
        levels=(0.0, 64.0),
        rgb_already_windowed=False,
        tile_delta=delta,
    )
    assert warm.updated_tiles == (0,)
    assert not warm_layer.states[0].visible
    assert warm_layer.states[0].page_resolutions[0].actual_key == canonical.actual_key
    np.testing.assert_array_equal(warm_layer.states[0].item.image, expected)


def test_two_fine_targets_assemble_from_one_coarse_actual_page():
    rect = (0, 8, 0, 8)
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    targets = plan(
        rect=rect,
        reduction=(1, 1),
        page_shape=(2, 4),
    )
    assert len(targets) == 2
    coarse_plan = plan(
        rect=rect,
        reduction=(2, 2),
        page_shape=(2, 2),
    )[0]
    coarse = materialize_lod_page(
        values,
        source_origin_yx=(0, 0),
        plan=coarse_plan,
    )
    requested_lod = LodInfo(1, 2, values.shape, (4, 4), 0)
    backing = PageBackedPresentation(targets, (coarse,), rect, requested_lod)
    payload = DisplayTilePayload(
        0,
        0,
        coarse.values,
        None,
        ("shared-coarse-page", 0),
        semantic_data=None,
        lod=requested_lod,
        page_backing=backing,
    )

    assembly = _resolve_page_backed_payload(payload)

    assert len(assembly.resolutions) == 2
    assert {resolution.actual_key for resolution in assembly.resolutions} == {coarse.key}
    expected = np.repeat(np.repeat(coarse.values, 4, axis=0), 4, axis=1)
    np.testing.assert_array_equal(assembly.payload.image, expected)
    for y in range(8):
        for x in range(8):
            assert backing.sample_presented_value_at_native(y, x) == expected[y, x]
    assert TiledValueSource({0: payload}).tile_region(
        SimpleNamespace(tile_number=0),
        (slice(1, 7), slice(2, 8)),
    ) is None


def test_heterogeneous_actual_pages_remain_target_aligned_physical_truth(qt_app):
    rect = (0, 16, 0, 32)
    values = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
    targets = plan(rect=rect, reduction=(2, 2), page_shape=(4, 4))
    level_three = plan(rect=rect, reduction=(3, 3), page_shape=(2, 2))
    level_four = plan(rect=rect, reduction=(4, 4), page_shape=(2, 2))
    assert len(targets) == 2 and len(level_three) == 2 and len(level_four) == 1
    pages = (
        materialize_lod_page(values, source_origin_yx=(0, 0), plan=level_three[0]),
        materialize_lod_page(values, source_origin_yx=(0, 0), plan=level_four[0]),
    )
    requested_lod = LodInfo(2, 4, values.shape, (4, 8), 0)
    backing = PageBackedPresentation(targets, pages, rect, requested_lod)
    resolved = backing.resolved_page_set
    assert resolved is not None
    assert resolved.actual_levels == (3, 4)
    assert resolved.target_actual_levels == (
        (targets[0].key, 3),
        (targets[1].key, 4),
    )
    assert resolved.uniform_actual_level is None
    assert resolved.coarsest_actual_level == 4

    payload = DisplayTilePayload(
        0,
        0,
        pages[0].values,
        None,
        ("heterogeneous-page-fallback", 0),
        semantic_data=None,
        lod=requested_lod,
        quality="preview",
        page_backing=backing,
    )
    assert payload.lod == requested_lod
    ref = payload_ref_from_display_payload(payload)
    assert ref.lod_level == 4
    assert not ref.satisfies_target(
        TileTarget(0, 0, ("heterogeneous-page-fallback", 0), lod_level=4)
    )
    assert ref.satisfies_target(
        TileTarget(0, 0, ("heterogeneous-page-fallback", 0), lod_level=5)
    )

    assembly = _resolve_page_backed_payload(payload)
    assert tuple(
        max(item.actual_key.lod.reduction)
        for item in assembly.resolutions
    ) == (3, 4)
    assert assembly.payload.image.shape == values.shape
    assert backing.sample_presented_value_at_native(0, 0) == pages[0].values[0, 0]
    assert backing.sample_presented_value_at_native(0, 31) == pages[1].values[0, 1]

    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    pool.update_payloads(
        {
            index: DisplayTilePayload(index, index, page.values, None, page.key)
            for index, page in enumerate(pages)
        },
        tile_shape=pages[0].values.shape[:2],
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=len(pages),
    )
    vispy = pool.resolve_page_targets(
        {index: target.key for index, target in enumerate(targets)}
    )
    assert tuple(
        max(vispy[index].actual_key.lod.reduction)
        for index in range(len(targets))
    ) == (3, 4)


def test_clipped_fine_targets_map_through_actual_coarse_page_bins():
    rect = (100, 109, 101, 114)
    values = source(rect)
    targets = plan(rect=rect, reduction=(1, 1), page_shape=(2, 2))
    coarse_plan = plan(rect=rect, reduction=(3, 3), page_shape=(4, 4))[0]
    coarse = materialize_lod_page(
        values,
        source_origin_yx=(rect[0], rect[2]),
        plan=coarse_plan,
    )
    requested_texture_shape = (
        max(item.stored_rect_yx[1] for item in targets)
        - min(item.stored_rect_yx[0] for item in targets),
        max(item.stored_rect_yx[3] for item in targets)
        - min(item.stored_rect_yx[2] for item in targets),
    )
    requested_lod = LodInfo(1, 2, (9, 13), requested_texture_shape, 0)
    payload = DisplayTilePayload(
        0,
        0,
        coarse.values,
        None,
        ("clipped-coarse-semantic-region", 0),
        semantic_data=None,
        lod=requested_lod,
        page_backing=PageBackedPresentation(
            targets,
            (coarse,),
            rect,
            requested_lod,
        ),
    )

    presented = payload.page_backing.sample_presented_values_at_native_coordinates(
        np.arange(rect[0], rect[1], dtype=np.int64),
        np.arange(rect[2], rect[3], dtype=np.int64),
    )
    expected = np.empty((9, 13), dtype=np.float32)
    for row, (y0, y1) in enumerate(coarse.plan.source_y_bins):
        for column, (x0, x1) in enumerate(coarse.plan.source_x_bins):
            expected[
                y0 - rect[0] : y1 - rect[0],
                x0 - rect[2] : x1 - rect[2],
            ] = coarse.values[row, column]

    np.testing.assert_array_equal(presented, expected)
    # x=101..103 is the clipped leading factor-eight bin; x=104 starts the
    # next actual coarse sample. Uniform scale/offset flooring aliases it.
    assert presented[0, 2] != presented[0, 3]
    assert TiledValueSource({0: payload}).tile_region(
        SimpleNamespace(tile_number=0),
        (slice(0, 9), slice(0, 13)),
    ) is None


def test_pyqtgraph_incomplete_complex_pages_use_honest_mapped_native_fallback(qt_app):
    yy, xx = np.mgrid[:8, :8]
    values = ((1.0 + xx + yy) * np.exp(1j * (xx - yy) / 3.0)).astype(np.complex64)
    rendered = _rendered_complex_channel(values, "complex")
    targets = plan(
        rect=(0, 8, 0, 8),
        reduction=(1, 1),
        page_shape=(2, 2),
        dtype="complex64",
        representation=COMPLEX_RG32F,
    )
    first_page = materialize_lod_page(
        values,
        source_origin_yx=(0, 0),
        plan=targets[0],
    )
    requested_lod = LodInfo(1, 2, values.shape, (4, 4), 0)
    payload = DisplayTilePayload(
        0,
        0,
        first_page.values,
        None,
        ("incomplete-complex-pages", 0),
        texture_data=first_page.values,
        texture_kind=COMPLEX_RG32F,
        semantic_data=values,
        semantic_histogram_data=np.abs(values),
        lod=requested_lod,
        shader_mapping=rendered.shader_mapping,
        page_backing=PageBackedPresentation(
            targets,
            (first_page,),
            (0, 8, 0, 8),
            requested_lod,
        ),
    )

    assembly = _resolve_page_backed_payload(payload, levels=(0.0, 16.0))

    assert assembly.fallback_reason == "incomplete-page-coverage-native"
    assert assembly.missing
    assert assembly.payload.page_backing is None
    assert assembly.payload.lod.level == 0
    assert assembly.payload.lod.factor == 1
    assert not np.iscomplexobj(assembly.payload.image)
    assert assembly.payload.image.shape[:2] == values.shape

    class Owner:
        def add_tile_item(self, *_args):
            pass

        def remove_tile_item(self, *_args):
            pass

        def move_tile_item(self, *_args):
            pass

    layer = MontageTileLayer(
        Owner(),
        set_image_item_data=lambda item, image, image_levels, **_kwargs: item.setImage(
            image,
            autoLevels=False,
            levels=image_levels,
        ),
        record_upload_timing=lambda *_args: None,
        histogram_levels_for_display=lambda image_levels: image_levels,
        is_rgb_image=lambda image: np.asarray(image).ndim == 3,
    )
    geometry = DisplayGeometry(
        view_state=None,
        display_shape=values.shape,
        montage=MontageGeometry(
            indices=(0,),
            tile_shape=values.shape,
            columns=1,
            rows=1,
            gap=0,
        ),
    )
    delta = SimpleNamespace(
        upserts={0: payload},
        active_tiles=(0,),
        target_identities={},
        removals=(),
        near_tile_source_ids={},
        cold_deadline_ms=None,
    )
    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 16.0),
        rgb_already_windowed=False,
        dirty_tiles=None,
        tile_payloads={0: payload},
        tile_delta=delta,
    )
    assert stats.presented_tiles == (0,)
    row = layer.tile_truth_physical_rows()[0]
    assert row["physical_lod_level"] == 0
    assert row["physical_lod_factor"] == 1
    assert row["physical_quality"] == "exact"
    assert row["physical_page_fallback_reason"] == "incomplete-page-coverage-native"
    assert row["physical_page_candidate_missing"] == assembly.missing

    with pytest.raises(ValueError, match="without a shader mapping"):
        _resolve_page_backed_payload(replace(payload, shader_mapping=None))


def test_pyqtgraph_replaces_coarse_page_pixels_when_fine_pages_arrive(qt_app):
    rect = (0, 8, 0, 8)
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    target = plan(rect=rect, reduction=(1, 1), page_shape=(4, 4))[0]
    coarse_plan = plan(rect=rect, reduction=(2, 2), page_shape=(2, 2))[0]
    fine = materialize_lod_page(values, source_origin_yx=(0, 0), plan=target)
    coarse = materialize_lod_page(values, source_origin_yx=(0, 0), plan=coarse_plan)
    requested_lod = LodInfo(1, 2, values.shape, fine.values.shape, 0)

    def payload(page, lod):
        return DisplayTilePayload(
            0,
            0,
            page.values,
            None,
            ("same-semantic-tile", 0),
            semantic_data=values,
            semantic_histogram_data=values,
            lod=lod,
            page_backing=PageBackedPresentation(
                (target,),
                (page,),
                rect,
                requested_lod,
            ),
        )

    coarse_payload = payload(coarse, requested_lod)
    fine_payload = payload(fine, requested_lod)

    class Owner:
        def add_tile_item(self, *_args):
            pass

        def remove_tile_item(self, *_args):
            pass

        def move_tile_item(self, *_args):
            pass

    layer = MontageTileLayer(
        Owner(),
        set_image_item_data=lambda item, image, image_levels, **_kwargs: item.setImage(
            image,
            autoLevels=False,
            levels=image_levels,
        ),
        record_upload_timing=lambda *_args: None,
        histogram_levels_for_display=lambda image_levels: image_levels,
        is_rgb_image=lambda image: np.asarray(image).ndim == 3,
    )
    geometry = DisplayGeometry(
        view_state=None,
        display_shape=values.shape,
        montage=MontageGeometry(indices=(0,), tile_shape=values.shape, columns=1, rows=1, gap=0),
    )

    def update(one_payload):
        delta = SimpleNamespace(
            upserts={0: one_payload},
            active_tiles=(0,),
            target_identities={},
            removals=(),
            near_tile_source_ids={},
            cold_deadline_ms=None,
        )
        return layer.update_presentation(
            None,
            histogram_data=None,
            geometry=geometry,
            levels=(0.0, 64.0),
            rgb_already_windowed=False,
            dirty_tiles=(0,),
            tile_payloads={0: one_payload},
            tile_delta=delta,
        )

    first = update(coarse_payload)
    coarse_source_id = layer.states[0].source_array_id
    coarse_pixels = np.asarray(layer.states[0].item.image).copy()
    second = update(fine_payload)

    assert first.presented_tiles == second.presented_tiles == (0,)
    assert layer.states[0].source_array_id != coarse_source_id
    assert layer.states[0].page_resolutions[0].actual_key == fine.key
    assert not np.array_equal(layer.states[0].item.image, coarse_pixels)


def test_cross_family_page_is_rejected_by_payload_and_both_resolvers():
    rect, _values, target, pages = _anisotropic_page_candidates(
        candidate_reducers=("mean_abs", "mean_abs")
    )
    table = PageTable()
    for index, page in enumerate(pages):
        table.bind(page.key, PageSlot("cross-family", 0, index), nbytes=page.nbytes)
    assert table.resolve(target.key) is None
    assert _vispy_resolve_materialized_pages(target, pages) is None

    lod = LodInfo(1, 2, (8, 8), target.stored_shape, 0)
    with pytest.raises(ValueError, match="do not belong"):
        PageBackedPresentation((target,), pages, rect, lod)


def test_phase_vector_backend_route_blacks_cancellation_and_rejects_mean_family():
    rect = (0, 2, 0, 2)
    values = np.asarray(
        [[1.0 + 0.0j, -1.0 + 0.0j], [0.0j, 0.0j]],
        dtype=np.complex64,
    )
    target = plan_source_grid_pages(
        content_key=("phase-backend",),
        valid_source_rect_yx=rect,
        reduction_yx=(1, 1),
        stored_page_shape=(1, 1),
        dtype="complex64",
        representation=COMPLEX_RG32F,
        reducer="phase_vector",
    )[0]
    mean_plan = plan_source_grid_pages(
        content_key=("phase-backend",),
        valid_source_rect_yx=rect,
        reduction_yx=(1, 1),
        stored_page_shape=(1, 1),
        dtype="complex64",
        representation=COMPLEX_RG32F,
        reducer="mean",
    )[0]
    phase_page = materialize_lod_page(values, source_origin_yx=(0, 0), plan=target)
    mean_page = materialize_lod_page(values, source_origin_yx=(0, 0), plan=mean_plan)
    assert phase_page.values[0, 0] == 0.0j

    table = PageTable()
    table.bind(mean_page.key, PageSlot("phase-family", 0, 0), nbytes=mean_page.nbytes)
    assert table.resolve(target.key) is None
    assert _vispy_resolve_materialized_pages(target, (mean_page,)) is None

    lod = LodInfo(1, 2, (2, 2), (1, 1), 0)
    with pytest.raises(ValueError, match="do not belong"):
        PageBackedPresentation((target,), (mean_page,), rect, lod)

    mapping = ShaderMapping(
        component=ShaderComponent.ANGLE,
        display_mode=ShaderDisplayMode.PHASE_COLOR,
    )
    payload = DisplayTilePayload(
        0,
        0,
        phase_page.values,
        None,
        ("phase-vector-backend", 0),
        texture_data=phase_page.values,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=values,
        lod=lod,
        shader_mapping=mapping,
        page_backing=PageBackedPresentation((target,), (phase_page,), rect, lod),
    )

    assert _payload_mode(payload, rgb_already_windowed=False) == 5
    resolved = _resolve_page_backed_payload(payload, levels=(-1000.0, 1000.0))
    assert resolved.resolutions[0].actual_key == target.key
    np.testing.assert_array_equal(resolved.payload.image, np.zeros((2, 2, 3), np.uint8))
    vispy = _vispy_resolve_materialized_pages(target, (phase_page,))
    assert vispy is not None and vispy.actual_key == target.key


@pytest.mark.parametrize(
    ("channel", "reducer"),
    (
        ("real", "mean"),
        ("imag", "mean"),
        ("abs", "mean_abs"),
        ("angle", "phase_vector"),
        ("complex", "mean"),
    ),
)
def test_live_complex_display_channels_select_canonical_reducer_family(channel, reducer):
    values = np.asarray([[0.0j, 1.0 + 0.0j], [-1.0 + 0.0j, 0.0j]], dtype=np.complex64)
    rendered = _rendered_complex_channel(values, channel)
    source_values = render_lod.canonical_value_source_for_rendered(
        rendered, shader_display=False
    )
    assert np.iscomplexobj(source_values)
    observed, _dtype, _representation = render_lod._reducer_format_for_rendered(
        rendered, source_values
    )
    assert observed == reducer


def test_live_cpu_angle_route_preserves_zero_magnitude_phase_policy():
    values = np.asarray([[0.0j, 1.0 + 0.0j], [-1.0 + 0.0j, 0.0j]], dtype=np.complex64)
    rendered = _rendered_complex_channel(values, "angle")
    demand = SimpleNamespace(desired_level=1, desired_factor_xy=(2, 2))
    key = render_lod.page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=1,
        semantic_source_id=("angle-source", 0),
        shader_display=False,
    )
    page = materialize_lod_page(
        render_lod.canonical_value_source_for_rendered(rendered, shader_display=False),
        source_origin_yx=(0, 0),
        plan=key.plans[0],
    )
    assert key.reducer == "phase_vector"
    assert page.values[0, 0] == pytest.approx(0.0j)


def test_live_level_zero_route_is_native_even_for_phase_display():
    values = np.asarray([[0.0j, 1.0 + 0.0j], [-1.0 + 0.0j, 1.0j]], dtype=np.complex64)
    rendered = _rendered_complex_channel(values, "complex")
    demand = SimpleNamespace(desired_level=0, desired_factor_xy=(1, 1))

    key = render_lod.page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=0,
        semantic_source_id=("phase-source", 0),
        shader_display=False,
    )
    page = materialize_lod_page(
        render_lod.canonical_value_source_for_rendered(rendered, shader_display=False),
        source_origin_yx=(0, 0),
        plan=key.plans[0],
    )

    assert key.reducer == "native"
    np.testing.assert_array_equal(page.values, values)


def test_page_backed_complex_preview_preserves_pyqtgraph_rewindowing_semantics():
    yy, xx = np.mgrid[:8, :8]
    values = ((1.0 + xx + yy) * np.exp(1j * (xx - yy) / 3.0)).astype(np.complex64)
    rendered = _rendered_complex_channel(values, "complex")
    demand = SimpleNamespace(desired_level=1, desired_factor_xy=(2, 2))
    key = render_lod.page_set_key_for_rendered(
        rendered,
        demand=demand,
        level=1,
        semantic_source_id=("complex-preview", 0),
        shader_display=False,
    )
    pages = tuple(
        materialize_lod_page(values, source_origin_yx=(0, 0), plan=page_plan)
        for page_plan in key.plans
    )
    lod = LodInfo(1, 2, values.shape, pages[0].values.shape[:2], 0)
    payload = DisplayTilePayload(
        0,
        0,
        pages[0].values,
        None,
        ("page-backed-complex-preview", 0),
        semantic_data=None,
        semantic_histogram_data=None,
        texture_data=pages[0].values,
        texture_kind="complex_rg32f",
        lod=lod,
        quality="preview",
        shader_mapping=rendered.shader_mapping,
        page_backing=PageBackedPresentation(key.plans, pages, (0, 8, 0, 8), lod),
    )

    assembled = _assemble_page_backed_payload(payload, levels=(0.0, 16.0))

    assert assembled.texture_kind == "rgb8"
    assert assembled.histogram_data is not None
    assert assembled.histogram_data.shape == values.shape
    assert assembled.semantic_data is None
    assert assembled.semantic_histogram_data is None
    assert not _payload_rgb_already_windowed(
        assembled,
        False,
        levels=(0.0, 16.0),
    ), "reduced complex RGB keeps its magnitude plane for later level drags"


def test_oversized_page_admission_releases_active_owner_claim():
    plans = plan_source_grid_pages(
        content_key=("oversized",),
        valid_source_rect_yx=(0, 8, 0, 8),
        reduction_yx=(0, 0),
        stored_page_shape=(8, 8),
        dtype="float32",
        representation="scalar_r32f",
        reducer="native",
    )
    page = materialize_lod_page(
        np.ones((8, 8), dtype=np.float32),
        source_origin_yx=(0, 0),
        plan=plans[0],
    )
    cache = LodPageCache(max_bytes=8)
    owner = ("oversized-worker", 1)

    assert cache.begin_claim(page.key, owner)
    assert cache.begin_owner_work(owner)
    cache.admit_as(page.key, page, owner=owner)

    assert cache.peek(page.key) is None
    assert cache.pending_count == 0
    assert cache.finish_owner_work(owner) == ()
    assert cache.begin_claim(page.key, owner)
    assert cache.release_owner_claims(owner) == (page.key,)


def test_unrendered_tile_route_is_reused_across_floor_queries(monkeypatch):
    tile = SimpleNamespace(source_index=7, montage_index=0)
    session = SimpleNamespace(
        output_dtype=np.dtype(np.float32),
        view_state=SimpleNamespace(channel="real"),
        plan=SimpleNamespace(tile_shape=(64, 48), tiles=(tile,)),
        tile_semantic_source_id=lambda index: ("tile-source", int(index)),
        _lod_page_set_key_cache={},
    )
    demand = SimpleNamespace(desired_level=2, desired_factor_xy=(4, 4))
    original = render_lod.plan_source_grid_pages
    calls = 0

    def counted(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(render_lod, "plan_source_grid_pages", counted)
    first = render_lod.page_set_key_for_tile(session, tile, demand=demand, level=2)
    second = render_lod.page_set_key_for_tile(session, tile, demand=demand, level=2)

    assert second is first
    assert calls == 1
