"""Qt-free tests for the LOD pyramid core (ADR 0050 phase 1)."""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.display.lod import (
    LOD_POLICY_RESIDENT,
    LOD_REASON_INVALID_VIEW,
    LOD_REASON_RESIDENT_COARSER,
    LOD_REASON_RESIDENT_FINER,
    LOD_REASON_RESIDENT_MATCH,
    LOD_REASON_RESIDENT_NATIVE_FALLBACK,
    choose_resident_level,
    factor_xy_for_level,
    resident_lod_policy,
    select_lod_demand,
)
from arrayscope.display.pyramid import (
    ALGO_VERSION,
    PyramidCache,
    PyramidLevelKey,
    reduce_box_mean,
    reduce_source_grid_mean,
)


def _key(level_xy=(1, 1), tile=0, component="scalar_r32f", source="src"):
    return PyramidLevelKey(
        source_id=source,
        tile_id=tile,
        component=component,
        level_xy=level_xy,
    )


class TestReduceBoxMean:
    def test_exact_two_by_two_means(self):
        array = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
            ],
            dtype=np.float32,
        )

        reduced = reduce_box_mean(array, (2, 2))

        assert reduced.shape == (2, 2)
        assert reduced.dtype == np.float32
        np.testing.assert_allclose(reduced, [[2.5, 4.5], [10.5, 12.5]])

    def test_float64_input_returns_float32(self):
        array = np.arange(16, dtype=np.float64).reshape(4, 4)

        reduced = reduce_box_mean(array, (2, 2))

        assert reduced.dtype == np.float32
        np.testing.assert_allclose(reduced, [[2.5, 4.5], [10.5, 12.5]])

    def test_partial_edge_boxes_average_without_padding(self):
        array = np.arange(15, dtype=np.float32).reshape(3, 5)

        reduced = reduce_box_mean(array, (2, 2))

        assert reduced.shape == (2, 3)
        # Bottom row and right column average only the samples that exist.
        np.testing.assert_allclose(
            reduced,
            [
                [(0 + 1 + 5 + 6) / 4, (2 + 3 + 7 + 8) / 4, (4 + 9) / 2],
                [(10 + 11) / 2, (12 + 13) / 2, 14.0],
            ],
        )

    def test_anisotropic_factors_reduce_each_axis_independently(self):
        array = np.arange(16, dtype=np.float32).reshape(4, 4)

        reduced_x = reduce_box_mean(array, (4, 1))
        reduced_y = reduce_box_mean(array, (1, 4))

        assert reduced_x.shape == (4, 1)
        assert reduced_y.shape == (1, 4)
        np.testing.assert_allclose(reduced_x[:, 0], [1.5, 5.5, 9.5, 13.5])
        np.testing.assert_allclose(reduced_y[0], [6.0, 7.0, 8.0, 9.0])

    def test_complex_input_reduces_per_component(self):
        array = (np.arange(8, dtype=np.float64) + 1j * np.arange(8, dtype=np.float64)[::-1]).reshape(2, 4)

        reduced = reduce_box_mean(array, (2, 2))

        assert reduced.dtype == np.complex64
        expected_real = np.array([[(0 + 1 + 4 + 5) / 4, (2 + 3 + 6 + 7) / 4]])
        expected_imag = np.array([[(7 + 6 + 3 + 2) / 4, (5 + 4 + 1 + 0) / 4]])
        np.testing.assert_allclose(reduced.real, expected_real)
        np.testing.assert_allclose(reduced.imag, expected_imag)

    def test_two_component_rg_planes_reduce_per_component(self):
        rg = np.stack(
            [
                np.arange(16, dtype=np.float32).reshape(4, 4),
                np.arange(16, dtype=np.float32).reshape(4, 4)[::-1],
            ],
            axis=-1,
        )

        reduced = reduce_box_mean(rg, (2, 2))

        assert reduced.shape == (2, 2, 2)
        np.testing.assert_allclose(reduced[..., 0], reduce_box_mean(rg[..., 0], (2, 2)))
        np.testing.assert_allclose(reduced[..., 1], reduce_box_mean(rg[..., 1], (2, 2)))

    def test_uint8_rgb_reduces_per_channel_and_keeps_dtype(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[..., 0] = [[0, 255], [0, 255]]
        rgb[..., 1] = 10
        rgb[..., 2] = [[1, 2], [3, 4]]

        reduced = reduce_box_mean(rgb, (2, 2))

        assert reduced.dtype == np.uint8
        assert reduced.shape == (1, 1, 3)
        assert int(reduced[0, 0, 0]) == 128  # rounded mean of 0/255/0/255
        assert int(reduced[0, 0, 1]) == 10
        assert int(reduced[0, 0, 2]) == 2  # rint(2.5) rounds half to even

    def test_factor_one_is_identity_values(self):
        array = np.arange(12, dtype=np.float32).reshape(3, 4)

        np.testing.assert_array_equal(reduce_box_mean(array, (1, 1)), array)

    def test_rejects_non_power_of_two_and_invalid_factors(self):
        array = np.zeros((4, 4), dtype=np.float32)

        with pytest.raises(ValueError):
            reduce_box_mean(array, (3, 1))
        with pytest.raises(ValueError):
            reduce_box_mean(array, (0, 2))
        with pytest.raises(ValueError):
            reduce_box_mean(np.zeros(4, dtype=np.float32), (2, 2))
        with pytest.raises(ValueError):
            reduce_box_mean(np.zeros((2, 2, 2, 2), dtype=np.float32), (2, 2))

    def test_level_from_level_matches_direct_reduction_for_divisible_shapes(self):
        rng = np.random.default_rng(7)
        array = rng.normal(size=(64, 64)).astype(np.float32)

        via_levels = reduce_box_mean(reduce_box_mean(array, (2, 2)), (2, 2))
        direct = reduce_box_mean(array, (4, 4))

        np.testing.assert_allclose(via_levels, direct, rtol=1e-5, atol=1e-5)


def _global_plane(rect: tuple[int, int, int, int]) -> np.ndarray:
    """Deterministic source values whose coordinates survive windowing."""

    y0, y1, x0, x1 = rect
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return (yy * 1000 + xx).astype(np.float32)


def _result_by_source_rect(result) -> dict[tuple[int, int, int, int], tuple[object, float]]:
    rects = tuple(result.source_rects)
    identities = tuple(result.identities)
    values = np.asarray(result.values).reshape(-1)
    assert len(rects) == len(identities) == len(values)
    return {
        tuple(int(value) for value in rect): (identity, float(value))
        for rect, identity, value in zip(rects, identities, values, strict=True)
    }


class TestSourceGridMeanReduction:
    """G5 contract for source-anchored, cache-history-independent reduction.

    ``source_origin_yx`` locates input sample ``[0, 0]`` in native source
    coordinates. ``valid_source_rect_yx`` is a half-open native-source rect,
    and ``reduction_vector_xy`` is the absolute per-axis log2 reduction.
    Returned ``source_rects`` and opaque, hashable ``identities`` are row-major
    alongside ``values``. Full global bins must therefore retain identity
    across differently clipped input windows while boundary fragments do not.
    """

    def test_shifted_windows_share_full_global_bins_but_not_clipped_boundaries(self):
        first_rect = (100, 104, 101, 113)
        second_rect = (100, 104, 102, 114)

        first = reduce_source_grid_mean(
            _global_plane(first_rect),
            source_origin_yx=first_rect[:1] + first_rect[2:3],
            valid_source_rect_yx=first_rect,
            reduction_vector_xy=(2, 2),
        )
        second = reduce_source_grid_mean(
            _global_plane(second_rect),
            source_origin_yx=second_rect[:1] + second_rect[2:3],
            valid_source_rect_yx=second_rect,
            reduction_vector_xy=(2, 2),
        )

        first_by_rect = _result_by_source_rect(first)
        second_by_rect = _result_by_source_rect(second)
        for full_rect in ((100, 104, 104, 108), (100, 104, 108, 112)):
            assert first_by_rect[full_rect][0] == second_by_rect[full_rect][0]
            assert first_by_rect[full_rect][1] == second_by_rect[full_rect][1]

        first_boundary = first_by_rect[(100, 104, 101, 104)][0]
        second_boundary = second_by_rect[(100, 104, 102, 104)][0]
        assert first_boundary != second_boundary

    def test_values_match_direct_cpu_global_grid_oracle(self):
        array_rect = (99, 110, 100, 116)
        valid_rect = (101, 109, 102, 115)
        source = _global_plane(array_rect)

        result = reduce_source_grid_mean(
            source,
            source_origin_yx=(array_rect[0], array_rect[2]),
            valid_source_rect_yx=valid_rect,
            reduction_vector_xy=(2, 1),
        )

        expected = []
        for y0, y1, x0, x1 in result.source_rects:
            local = source[
                int(y0) - array_rect[0] : int(y1) - array_rect[0],
                int(x0) - array_rect[2] : int(x1) - array_rect[2],
            ]
            expected.append(float(np.mean(local, dtype=np.float32)))
        np.testing.assert_allclose(np.asarray(result.values).reshape(-1), expected)

    def test_recursive_route_matches_direct_only_for_aligned_input_grid(self):
        rect = (96, 112, 96, 112)
        source = _global_plane(rect)
        level_one = reduce_source_grid_mean(
            source,
            source_origin_yx=(96, 96),
            valid_source_rect_yx=rect,
            reduction_vector_xy=(1, 1),
        )

        recursive = reduce_source_grid_mean(
            level_one.values,
            source_origin_yx=level_one.grid_origin_yx,
            valid_source_rect_yx=rect,
            input_reduction_vector_xy=(1, 1),
            reduction_vector_xy=(2, 2),
        )
        direct = reduce_source_grid_mean(
            source,
            source_origin_yx=(96, 96),
            valid_source_rect_yx=rect,
            reduction_vector_xy=(2, 2),
        )

        assert tuple(recursive.source_rects) == tuple(direct.source_rects)
        assert tuple(recursive.identities) == tuple(direct.identities)
        np.testing.assert_allclose(recursive.values, direct.values, rtol=1e-6, atol=1e-6)

        clipped_rect = (101, 113, 101, 113)
        clipped_level_one = reduce_source_grid_mean(
            _global_plane(clipped_rect),
            source_origin_yx=(101, 101),
            valid_source_rect_yx=clipped_rect,
            reduction_vector_xy=(1, 1),
        )
        with pytest.raises(ValueError, match="align"):
            reduce_source_grid_mean(
                clipped_level_one.values,
                source_origin_yx=clipped_level_one.grid_origin_yx,
                valid_source_rect_yx=clipped_rect,
                input_reduction_vector_xy=(1, 1),
                reduction_vector_xy=(2, 2),
            )

    def test_anisotropic_vector_places_bins_on_each_source_axis_grid(self):
        rect = (5, 12, 9, 20)

        result = reduce_source_grid_mean(
            _global_plane(rect),
            source_origin_yx=(5, 9),
            valid_source_rect_yx=rect,
            reduction_vector_xy=(2, 1),
        )

        assert result.reduction_vector_xy == (2, 1)
        assert result.grid_origin_yx == (4, 8)
        assert np.asarray(result.values).shape == (4, 3)
        assert tuple(result.source_rects)[0] == (5, 6, 9, 12)
        assert tuple(result.source_rects)[-1] == (10, 12, 16, 20)

    def test_reported_coverage_partitions_valid_rect_without_gaps_or_overlaps(self):
        array_rect = (1, 17, 2, 21)
        valid_rect = (3, 14, 5, 18)

        result = reduce_source_grid_mean(
            _global_plane(array_rect),
            source_origin_yx=(array_rect[0], array_rect[2]),
            valid_source_rect_yx=valid_rect,
            reduction_vector_xy=(2, 1),
        )

        coverage = np.zeros((valid_rect[1] - valid_rect[0], valid_rect[3] - valid_rect[2]), dtype=np.uint8)
        for y0, y1, x0, x1 in result.source_rects:
            assert valid_rect[0] <= y0 < y1 <= valid_rect[1]
            assert valid_rect[2] <= x0 < x1 <= valid_rect[3]
            coverage[
                int(y0) - valid_rect[0] : int(y1) - valid_rect[0],
                int(x0) - valid_rect[2] : int(x1) - valid_rect[2],
            ] += 1
        np.testing.assert_array_equal(coverage, np.ones_like(coverage))


class TestPyramidLevelKey:
    def test_key_identity_includes_algorithm_version(self):
        key = _key()

        assert key.algo_version == ALGO_VERSION
        assert key != PyramidLevelKey(
            source_id="src",
            tile_id=0,
            component="scalar_r32f",
            level_xy=(1, 1),
            algo_version=ALGO_VERSION + 1,
        )

    def test_key_reports_factors_and_scalar_level(self):
        key = _key(level_xy=(2, 1))

        assert key.factor_xy == (4, 2)
        assert key.level == 2

    def test_key_rejects_negative_levels(self):
        with pytest.raises(ValueError):
            _key(level_xy=(-1, 0))


class TestPyramidCache:
    def test_lookup_counts_misses_and_hits(self):
        cache = PyramidCache(max_bytes=1 << 20)
        key = _key()

        assert cache.lookup(key) is None
        admitted = cache.admit(key, np.ones((4, 4), dtype=np.float32))
        assert cache.lookup(key) is admitted
        assert cache.misses == 1
        assert cache.hits == 1

    def test_peek_many_returns_resident_levels_without_counting(self):
        cache = PyramidCache(max_bytes=1 << 20)
        first = _key(tile=0)
        second = _key(tile=1)
        first_value = cache.admit(first, np.ones((4, 4), dtype=np.float32))
        second_value = cache.admit(second, np.full((4, 4), 2.0, dtype=np.float32))

        observed = cache.peek_many((second, _key(tile=9), first))

        assert observed.keys() == {second, first}
        assert observed[first] is first_value
        assert observed[second] is second_value
        assert cache.hits == 0 and cache.misses == 0

    def test_bytes_accounting_and_bounded_eviction(self):
        item = np.ones((8, 8), dtype=np.float32)
        cache = PyramidCache(max_bytes=int(item.nbytes * 2))

        cache.admit(_key(tile=0), item)
        cache.admit(_key(tile=1), item)
        assert cache.bytes_used == item.nbytes * 2

        cache.admit(_key(tile=2), item)
        assert cache.bytes_used == item.nbytes * 2
        assert cache.evictions == 1
        assert cache.peek(_key(tile=0)) is None  # LRU evicted
        assert cache.peek(_key(tile=2)) is not None

    def test_oversized_entries_are_not_admitted_but_release_pending(self):
        cache = PyramidCache(max_bytes=8)
        key = _key()
        assert cache.begin_pending(key)

        cache.admit(key, np.ones((64, 64), dtype=np.float32))

        assert cache.peek(key) is None
        assert not cache.pending(key)

    def test_singleflight_claims_coalesce_duplicates(self):
        cache = PyramidCache(max_bytes=1 << 20)
        key = _key()

        assert cache.begin_pending(key) is True
        assert cache.begin_pending(key) is False
        assert cache.pending(key)
        assert cache.pending_count == 1

        cache.admit(key, np.ones((2, 2), dtype=np.float32))
        assert not cache.pending(key)
        # Already cached: no new claim needed.
        assert cache.begin_pending(key) is False

    def test_end_pending_releases_claim_without_admitting(self):
        cache = PyramidCache(max_bytes=1 << 20)
        key = _key()

        assert cache.begin_pending(key)
        cache.end_pending(key)

        assert not cache.pending(key)
        assert cache.begin_pending(key) is True

    def test_resident_level_counts_for_diagnostics(self):
        cache = PyramidCache(max_bytes=1 << 20)
        cache.admit(_key(tile=0, level_xy=(1, 1)), np.ones((2, 2), dtype=np.float32))
        cache.admit(_key(tile=1, level_xy=(1, 1)), np.ones((2, 2), dtype=np.float32))
        cache.admit(_key(tile=0, level_xy=(2, 2)), np.ones((1, 1), dtype=np.float32))

        assert cache.resident_level_counts() == {1: 2, 2: 1}


ZOOMED_OUT_4X = ((0.0, 1024.0), (0.0, 1024.0))
VIEWPORT = (256, 256)
TILE = (64, 64)


class TestResidentLodPolicy:
    def test_native_fallback_when_nothing_is_resident(self):
        decision = resident_lod_policy(ZOOMED_OUT_4X, VIEWPORT, TILE, resident_levels=())

        assert decision.policy == LOD_POLICY_RESIDENT
        assert decision.demand.desired_factor == 4
        assert decision.applied_level == 0
        assert decision.applied_factor == 1
        assert decision.applied_factor_xy == (1, 1)
        assert decision.reason == LOD_REASON_RESIDENT_NATIVE_FALLBACK

    def test_applies_demanded_level_when_resident(self):
        decision = resident_lod_policy(ZOOMED_OUT_4X, VIEWPORT, TILE, resident_levels=(2,))

        assert decision.applied_level == 2
        assert decision.applied_factor == 4
        assert decision.applied_factor_xy == (4, 4)
        assert decision.reason == LOD_REASON_RESIDENT_MATCH

    def test_prefers_closest_resident_level_finer_on_ties(self):
        # Desired level 2; levels 1 and 3 both distance one; finer wins.
        decision = resident_lod_policy(ZOOMED_OUT_4X, VIEWPORT, TILE, resident_levels=(1, 3))

        assert decision.applied_level == 1
        assert decision.applied_factor == 2
        assert decision.reason == LOD_REASON_RESIDENT_FINER

    def test_rejects_coarser_only_resident_level_for_native_fallback(self):
        coarser = resident_lod_policy(ZOOMED_OUT_4X, VIEWPORT, TILE, resident_levels=(3,))
        way_too_coarse = resident_lod_policy(ZOOMED_OUT_4X, VIEWPORT, TILE, resident_levels=(5,))

        assert coarser.applied_level == 0
        assert coarser.reason == LOD_REASON_RESIDENT_NATIVE_FALLBACK
        assert way_too_coarse.applied_level == 0
        assert way_too_coarse.reason == LOD_REASON_RESIDENT_NATIVE_FALLBACK

    def test_invalid_view_selects_native(self):
        decision = resident_lod_policy(None, VIEWPORT, TILE, resident_levels=(1, 2))

        assert decision.applied_level == 0
        assert decision.reason == LOD_REASON_INVALID_VIEW

    def test_hysteresis_flows_through_previous_factor(self):
        # Just past the 4x threshold with a previous factor of 4: demand holds.
        held = resident_lod_policy(
            ((0.0, 300.0 * 4), (0.0, 300.0 * 4)),
            (256, 256),
            TILE,
            previous_factor=4,
            resident_levels=(2,),
        )
        # The same view with no previous factor demands less.
        fresh = resident_lod_policy(
            ((0.0, 300.0 * 4), (0.0, 300.0 * 4)),
            (256, 256),
            TILE,
            previous_factor=None,
            resident_levels=(2,),
        )

        assert held.demand.desired_factor == 4
        assert held.applied_level == 2
        assert fresh.demand.desired_factor == 4

    def test_zoom_in_prefers_native_over_coarser_only_resident_inputs(self):
        # Zoom-in: previously presented coarse level 3, demand drops to 2.
        # Coarser resident data remains useful physical cache, but it must not
        # satisfy the semantic tile target or reintroduce wrong retained pixels.
        demand = select_lod_demand(ZOOMED_OUT_4X, VIEWPORT, TILE)
        assert demand.desired_level == 2

        decision = resident_lod_policy(
            ZOOMED_OUT_4X,
            VIEWPORT,
            TILE,
            resident_levels=(3,),
        )

        assert decision.applied_level == 0
        assert decision.reason == LOD_REASON_RESIDENT_NATIVE_FALLBACK

    def test_zoom_in_equidistant_choice_prefers_native_over_stale_coarse(self):
        # Demand level 1: native and a stale level 2 are equidistant; the
        # finer (native, always resident) side wins.
        demand = select_lod_demand(((0.0, 768.0), (0.0, 768.0)), VIEWPORT, TILE)
        assert demand.desired_level == 1

        decision = resident_lod_policy(
            ((0.0, 768.0), (0.0, 768.0)),
            VIEWPORT,
            TILE,
            resident_levels=(2,),
        )

        assert decision.applied_level == 0
        assert decision.reason == LOD_REASON_RESIDENT_NATIVE_FALLBACK


class TestResidentSelectionHelpers:
    def test_choose_resident_level_defaults_to_native(self):
        demand = select_lod_demand(ZOOMED_OUT_4X, VIEWPORT, TILE)

        assert choose_resident_level(demand, ()) == 0
        assert choose_resident_level(demand, (2,)) == 2
        assert choose_resident_level(demand, (1, 3)) == 1

    def test_factor_xy_for_level_shifts_anisotropy_with_the_level(self):
        demand = select_lod_demand(((0.0, 2048.0), (0.0, 512.0)), (256, 256), TILE)
        assert demand.desired_factor_xy == (8, 1)
        assert demand.desired_level == 3

        assert factor_xy_for_level(demand, 3) == (8, 1)
        assert factor_xy_for_level(demand, 2) == (4, 1)
        assert factor_xy_for_level(demand, 4) == (16, 2)
        assert factor_xy_for_level(demand, 0) == (1, 1)

    def test_factor_xy_for_level_dominant_axis_matches_level(self):
        demand = select_lod_demand(ZOOMED_OUT_4X, VIEWPORT, TILE)

        for level in (1, 2, 3):
            factor_x, factor_y = factor_xy_for_level(demand, level)
            assert max(factor_x, factor_y) == 2 ** level
