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
from arrayscope.display.pyramid import ALGO_VERSION, PyramidCache, PyramidLevelKey, reduce_box_mean


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

    def test_accepts_one_coarser_level_but_never_beyond_acceptable(self):
        coarser = resident_lod_policy(ZOOMED_OUT_4X, VIEWPORT, TILE, resident_levels=(3,))
        way_too_coarse = resident_lod_policy(ZOOMED_OUT_4X, VIEWPORT, TILE, resident_levels=(5,))

        assert coarser.applied_level == 3
        assert coarser.reason == LOD_REASON_RESIDENT_COARSER
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

    def test_retain_presented_until_replacement_resident_inputs(self):
        # Zoom-in: previously presented coarse level 3, demand drops to 2.
        # While level 2 is not yet resident, the policy keeps the resident
        # coarser level (still acceptable and closest) instead of blocking.
        demand = select_lod_demand(ZOOMED_OUT_4X, VIEWPORT, TILE)
        assert demand.desired_level == 2

        decision = resident_lod_policy(
            ZOOMED_OUT_4X,
            VIEWPORT,
            TILE,
            resident_levels=(3,),
        )

        assert decision.applied_level == 3
        assert decision.reason == LOD_REASON_RESIDENT_COARSER

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
