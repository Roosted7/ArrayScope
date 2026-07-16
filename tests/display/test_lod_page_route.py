"""Canonical G5 page route, reducers, and logical-cache ownership."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

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
from arrayscope.display.backends.pyqtgraph.tiles import _assemble_page_backed_payload
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    PageBackedPresentation,
    TiledValueSource,
)
from arrayscope.gpu.keys import COMPLEX_RG32F, SCALAR_R32F


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


def test_named_axis_conversion_keeps_asymmetric_routes_distinct():
    assert reduction_yx_to_xy((1, 2)) == (2, 1)
    assert reduction_xy_to_yx((1, 2)) == (2, 1)

    y1_x2 = plan(reduction=(1, 2), page_shape=(8, 8))
    y2_x1 = plan(reduction=(2, 1), page_shape=(8, 8))
    assert y1_x2[0].stored_shape == (5, 4)
    assert tuple(page.stored_shape for page in y2_x1) == ((3, 6), (3, 1))
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


def test_page_backed_value_lookup_uses_exact_clipped_bin_geometry():
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
    values = TiledValueSource({0: payload})

    # x=101 is the one-sample clipped leading bin; x=102 begins the next
    # full factor-two bin. Equal-width arithmetic would map one of these
    # coordinates incorrectly.
    first = values.value_at(SimpleNamespace(tile_number=0, local_y=0, local_x=0))
    second = values.value_at(SimpleNamespace(tile_number=0, local_y=0, local_x=1))
    assert first == pytest.approx((10101 + 10201) / 2)
    assert second == pytest.approx((10102 + 10103 + 10202 + 10203) / 4)


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
