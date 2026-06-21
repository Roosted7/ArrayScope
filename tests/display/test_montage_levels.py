import numpy as np

from arrayscope.display.model.montage_levels import (
    AGGREGATE_SAMPLE_LIMIT,
    PROVISIONAL_TILE_SAMPLE_LIMIT,
    MontageLevelTracker,
)
from arrayscope.display.planning import LevelSourceRank


def test_montage_level_tracker_reuses_overlap_and_excludes_removed_indices():
    tracker = MontageLevelTracker()
    key = "scope"
    tracker.ensure(key, (0, 1, 2))
    for index, value in enumerate((10.0, 20.0, 30.0)):
        tracker.update_from_tile(key, index, np.full((4, 4), value, dtype=np.float32), np.full((4, 4), value, dtype=np.float32))

    first = tracker.stats_for(key)
    assert first.source_indices == frozenset({0, 1, 2})
    assert first.bounds == (9.5, 30.5)

    shifted = tracker.ensure(key, (1, 2, 3))
    assert shifted.source_indices == frozenset({1, 2})
    assert shifted.bounds == (19.5, 30.5)
    assert shifted.rank == LevelSourceRank.MONTAGE_VISIBLE_SUBSET

    tracker.update_from_tile(key, 3, np.full((4, 4), 40.0, dtype=np.float32), np.full((4, 4), 40.0, dtype=np.float32))
    complete = tracker.stats_for(key)
    assert complete.source_indices == frozenset({1, 2, 3})
    assert complete.bounds == (19.5, 40.5)
    assert complete.rank == LevelSourceRank.MONTAGE_SAMPLED_FULL


def test_montage_level_tracker_does_not_downgrade_when_expected_set_shrinks():
    tracker = MontageLevelTracker()
    key = "scope"
    tracker.ensure(key, (0, 1, 2))
    for index in (0, 1, 2):
        tracker.update_from_tile(key, index, np.full((4, 4), index, dtype=np.float32), np.full((4, 4), index, dtype=np.float32))
    complete = tracker.stats_for(key)

    zoomed = tracker.ensure(key, (1,))

    assert complete.rank == LevelSourceRank.MONTAGE_SAMPLED_FULL
    assert zoomed.rank == LevelSourceRank.MONTAGE_SAMPLED_FULL
    assert tracker.ensure(key, (0, 1, 2)).source_indices == frozenset({0, 1, 2})


def test_montage_level_tracker_samples_deterministically_and_caps_aggregate():
    tracker = MontageLevelTracker()
    key = "scope"
    tracker.ensure(key, tuple(range(3)))
    large = np.arange(PROVISIONAL_TILE_SAMPLE_LIMIT * 4, dtype=np.float32)
    for index in range(3):
        tracker.update_from_tile(key, index, large + index, large + index)

    stats = tracker.stats_for(key)
    sample = tracker.histogram_data_for_stats(stats)

    assert sample is not None
    assert sample.size <= AGGREGATE_SAMPLE_LIMIT
    assert np.array_equal(sample[:5], np.asarray([0, 4, 8, 12, 16], dtype=np.float32))


def test_montage_level_key_ignores_requested_coverage_and_layout_but_keeps_selection():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.model.montage_levels import montage_level_key

    state = ViewState.from_shape((8, 8, 6)).with_montage_axis(2, columns=2, indices=(0, 1, 2), text="0:3")
    relaid = state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text="0:3")

    first = montage_level_key("doc", state, (0, 1, 2), None)
    second = montage_level_key("doc", relaid, (0, 1, 2), None)
    changed_coverage = montage_level_key("doc", relaid, (0, 1, 2, 3), None)
    shifted_selection = state.with_montage_axis(2, columns=2, indices=(1, 2, 3), text="1:4")
    shifted = montage_level_key("doc", shifted_selection, (1, 2, 3), None)

    assert first == second
    assert first == changed_coverage
    assert first != shifted


def test_montage_level_key_ignores_colormap_lut():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.model.montage_levels import montage_level_key

    state = ViewState.from_shape((8, 8, 3)).with_montage_axis(2, indices=(0, 1, 2), text=":")

    first = montage_level_key("doc", state, (0, 1, 2), np.zeros((16, 4), dtype=np.uint8))
    second = montage_level_key("doc", state, (0, 1, 2), np.full((16, 4), 255, dtype=np.uint8))

    assert first == second


def test_montage_level_tracker_reports_reusable_source_stats():
    tracker = MontageLevelTracker()
    key = "scope"
    tracker.ensure(key, (0,))

    assert tracker.has_source(key, 0) is False
    tracker.update_from_tile(key, 0, np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32))

    assert tracker.has_source(key, 0) is True
    assert tracker.has_source(key, 0, refined=True) is True


def test_tile_stats_do_not_rescan_values_for_bounds_and_sample(monkeypatch):
    import arrayscope.display.model.montage_levels as montage_levels

    def fail_rescan(*_args, **_kwargs):
        raise AssertionError("tile stats should share one finite-value pass")

    monkeypatch.setattr(montage_levels, "_finite_bounds", fail_rescan)
    monkeypatch.setattr(montage_levels, "_finite_sample", fail_rescan)

    stats = montage_levels._sample_tile_stats(
        np.asarray([0.0, np.nan, 2.0], dtype=np.float32),
        0,
        refined=False,
    )

    assert stats is not None
    assert stats.bounds == (0.0, 2.0)


def test_montage_level_tracker_can_defer_aggregate_rebuild():
    tracker = MontageLevelTracker()
    key = "scope"
    tracker.ensure(key, (0, 1))

    assert tracker.update_from_tile(
        key,
        0,
        np.ones((4, 4), dtype=np.float32),
        np.ones((4, 4), dtype=np.float32),
        aggregate=False,
    ) is None
    stats = tracker.stats_for(key)

    assert stats.source_indices == frozenset({0})
    assert tracker.stats_for(key) is stats


def test_montage_level_tracker_uses_incremental_histogram_accumulator(monkeypatch):
    import arrayscope.display.model.montage_levels as montage_levels

    tracker = MontageLevelTracker()
    key = "scope"
    tracker.ensure(key, (0, 1))
    tracker.update_from_tile(
        key,
        0,
        np.arange(16, dtype=np.float32),
        np.arange(16, dtype=np.float32),
        aggregate=False,
    )

    def fail_rebuild(*_args, **_kwargs):
        raise AssertionError("aggregate sample should be maintained incrementally")

    monkeypatch.setattr(montage_levels, "_aggregate_samples", fail_rebuild)

    first = tracker.stats_for(key)
    assert np.array_equal(tracker.histogram_data_for_stats(first), np.arange(16, dtype=np.float32))

    tracker.update_from_tile(
        key,
        1,
        np.arange(16, 32, dtype=np.float32),
        np.arange(16, 32, dtype=np.float32),
        aggregate=False,
    )
    second = tracker.stats_for(key)

    assert second.source_indices == frozenset({0, 1})
    assert np.array_equal(tracker.histogram_data_for_stats(second), np.arange(32, dtype=np.float32))
