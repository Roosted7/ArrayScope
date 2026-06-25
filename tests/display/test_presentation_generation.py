from hypothesis import given, strategies as st
import pytest

from arrayscope.display.model.level_convergence import ProgressiveTileLevelConvergence, UniformLevelConvergence
from arrayscope.display.model.presentation_generation import PresentationGenerationTracker


def test_generation_tracker_acknowledges_only_latest_revision():
    tracker = PresentationGenerationTracker()

    assert tracker.begin_target((0.0, 1.0), active_tiles=(0, 1))
    first_revision = tracker.revision
    assert tracker.begin_target((2.0, 4.0), active_tiles=(0, 1))

    tracker.acknowledge_upserts(first_revision, (0, 1), levels=(0.0, 1.0))

    snapshot = tracker.snapshot()
    assert snapshot.revision == first_revision + 1
    assert snapshot.stale_count == 2
    assert snapshot.settled is False


def test_generation_tracker_partial_acknowledgement_stays_pending():
    tracker = PresentationGenerationTracker()
    tracker.begin_target((2.0, 4.0), active_tiles=(0, 1, 2))

    tracker.acknowledge_upserts(tracker.revision, (0,), levels=(2.0, 4.0))

    assert tracker.stale_tiles() == (1, 2)
    assert tracker.snapshot(pending_upserts=(1,)).pending_count == 2


def test_generation_tracker_same_settled_target_is_noop():
    tracker = PresentationGenerationTracker()
    assert tracker.begin_target((2.0, 4.0), active_tiles=(0, 1))
    revision = tracker.revision
    tracker.acknowledge_upserts(revision, (0, 1), levels=(2.0, 4.0))

    assert tracker.begin_target((2.0, 4.0), active_tiles=(0, 1)) is False
    assert tracker.revision == revision
    assert tracker.snapshot().settled is True


def test_generation_tracker_active_set_changes_recompute_scope():
    tracker = PresentationGenerationTracker()
    tracker.begin_target((2.0, 4.0), active_tiles=(0, 1))
    tracker.acknowledge_upserts(tracker.revision, (0, 1), levels=(2.0, 4.0))

    tracker.set_active_tiles((1, 2))

    snapshot = tracker.snapshot()
    assert snapshot.active_presented_tile_count == 2
    assert snapshot.stale_count == 1
    assert tracker.stale_active_tiles == {2}
    assert tracker.stale_tiles(priority_order=(2, 1)) == (2,)


def test_generation_tracker_snapshot_uses_incremental_stale_state(monkeypatch):
    tracker = PresentationGenerationTracker()
    tracker.begin_target((2.0, 4.0), active_tiles=(0, 1, 2))
    tracker.acknowledge_upserts(tracker.revision, (0,), levels=(2.0, 4.0))

    monkeypatch.setattr(tracker, "stale_tiles", lambda *args, **kwargs: pytest.fail("snapshot should not enumerate stale tiles"))

    snapshot = tracker.snapshot()
    assert snapshot.stale_count == 2
    assert snapshot.pending_count == 2
    assert snapshot.settled is False


def test_uniform_strategy_settles_same_semantics_as_progressive_strategy():
    progressive = PresentationGenerationTracker()
    uniform = PresentationGenerationTracker()

    assert ProgressiveTileLevelConvergence().begin(progressive, (1.0, 3.0), active_tiles=(0, 1))
    ProgressiveTileLevelConvergence().acknowledge(
        progressive,
        target_revision=progressive.revision,
        accepted_tiles=(0, 1),
        levels=(1.0, 3.0),
    )
    assert UniformLevelConvergence().begin(uniform, (1.0, 3.0), active_tiles=(0, 1)) is False

    assert progressive.snapshot().target_levels == uniform.snapshot().target_levels
    assert progressive.snapshot().settled is True
    assert uniform.snapshot().settled is True
    assert progressive.value_counts() == uniform.value_counts()


@given(
    active=st.lists(st.integers(min_value=0, max_value=20), unique=True),
    acknowledged=st.lists(st.integers(min_value=0, max_value=20), unique=True),
)
def test_generation_tracker_settled_iff_all_active_acknowledged(active, acknowledged):
    tracker = PresentationGenerationTracker()
    tracker.begin_target((0.0, 1.0), active_tiles=active)
    tracker.acknowledge_upserts(tracker.revision, acknowledged, levels=(0.0, 1.0))

    active_set = set(active)
    acknowledged_set = set(acknowledged)
    assert tracker.snapshot().settled is (active_set <= acknowledged_set)
