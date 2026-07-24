import ast
from pathlib import Path

import arrayscope.core.window_levels as window_levels

WINDOW_LEVELS_PATH = Path(window_levels.__file__)


def test_absolute_window_reuses_previous_numeric_levels():
    decision = window_levels.choose_window_levels(
        mode="absolute",
        previous_levels=(10, 20),
        previous_bounds=(0, 100),
        current_bounds=(1000, 2000),
        default_levels=(0, 100),
    )

    assert decision.auto_levels is False
    assert decision.levels == (10.0, 20.0)


def test_relative_window_maps_previous_fractions_to_current_bounds():
    decision = window_levels.choose_window_levels(
        mode="relative",
        previous_levels=(25, 75),
        previous_bounds=(0, 100),
        current_bounds=(200, 300),
    )

    assert decision.auto_levels is False
    assert decision.levels == (225.0, 275.0)


def test_relative_window_auto_levels_without_previous_state():
    decision = window_levels.choose_window_levels(
        mode="relative",
        previous_levels=None,
        previous_bounds=None,
        current_bounds=(200, 300),
        default_levels=(-3.14, 3.14),
    )

    assert decision.auto_levels is True
    assert decision.levels == (-3.14, 3.14)


def test_force_auto_overrides_absolute_window_for_channel_or_scale_changes():
    decision = window_levels.choose_window_levels(
        mode="absolute",
        previous_levels=(10, 20),
        previous_bounds=(0, 100),
        current_bounds=(200, 300),
        default_levels=None,
        force_auto=True,
    )

    assert decision.auto_levels is True
    assert decision.levels is None


def test_controller_relative_same_source_remaps_levels_as_statistics_improve():
    previous = window_levels.LevelSource(
        levels=(25.0, 75.0),
        histogram_range=(0.0, 100.0),
        rank=window_levels.LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        source_count=1,
        expected_count=4,
        semantic_key="same",
    )
    candidate = window_levels.LevelSource(
        levels=(200.0, 400.0),
        histogram_range=(200.0, 400.0),
        rank=window_levels.LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        source_count=2,
        expected_count=4,
        semantic_key="same",
    )

    state = window_levels.WindowLevelController().decide(
        previous=previous, candidate=candidate, mode="relative"
    )

    assert state.display_levels == (100.0, 300.0)
    assert state.histogram_range == (0.0, 400.0)
    assert state.source_count == 2


def test_controller_relative_same_source_does_not_downgrade_when_viewport_coverage_shrinks():
    previous = window_levels.LevelSource(
        levels=(100.0, 300.0),
        histogram_range=(0.0, 400.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=4,
        expected_count=4,
        semantic_key="same",
    )
    candidate = window_levels.LevelSource(
        levels=(100.0, 200.0),
        histogram_range=(100.0, 200.0),
        rank=window_levels.LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        source_count=1,
        expected_count=4,
        semantic_key="same",
    )

    state = window_levels.WindowLevelController().decide(
        previous=previous, candidate=candidate, mode="relative"
    )

    assert state.display_levels == (100.0, 300.0)
    assert state.histogram_range == (0.0, 400.0)
    assert state.source_rank == window_levels.LevelSourceRank.MONTAGE_COMPLETE
    assert state.source_count == 4


def test_controller_keeps_complete_predecessor_until_successor_population_is_complete():
    previous = window_levels.LevelSource(
        levels=(100.0, 300.0),
        histogram_range=(0.0, 400.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=50,
        expected_count=50,
        semantic_key="previous-crop",
    )
    partial_successor = window_levels.LevelSource(
        levels=(200.0, 210.0),
        histogram_range=(200.0, 210.0),
        rank=window_levels.LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        source_count=1,
        expected_count=50,
        semantic_key="successor-crop",
    )

    state = window_levels.WindowLevelController().decide(
        previous=previous,
        candidate=partial_successor,
        mode="relative",
    )

    assert state.semantic_key == "successor-crop"
    assert state.display_levels == (100.0, 300.0)
    assert state.histogram_range == (0.0, 400.0)
    assert state.source_count == 1


def test_controller_switches_semantics_atomically_when_successor_population_is_complete():
    previous = window_levels.LevelSource(
        levels=(100.0, 300.0),
        histogram_range=(0.0, 400.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=50,
        expected_count=50,
        semantic_key="previous-crop",
    )
    complete_successor = window_levels.LevelSource(
        levels=(200.0, 600.0),
        histogram_range=(200.0, 600.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=50,
        expected_count=50,
        semantic_key="successor-crop",
        evidence_quality=2,
    )

    state = window_levels.WindowLevelController().decide(
        previous=previous,
        candidate=complete_successor,
        mode="relative",
    )

    assert state.semantic_key == "successor-crop"
    assert state.display_levels == (300.0, 500.0)
    assert state.histogram_range == (200.0, 600.0)
    assert state.source_count == 50


def test_controller_keeps_predecessor_through_complete_preview_then_switches_at_target_quality():
    controller = window_levels.WindowLevelController()
    previous = window_levels.LevelSource(
        levels=(0.0, 400.0),
        histogram_range=(0.0, 400.0),
        rank=window_levels.LevelSourceRank.PREVIOUS_COMMITTED,
        semantic_key="previous",
    )
    preview = window_levels.LevelSource(
        levels=(100.0, 200.0),
        histogram_range=(100.0, 200.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=50,
        expected_count=50,
        semantic_key="successor",
        evidence_quality=1,
    )
    target = window_levels.LevelSource(
        levels=(0.0, 500.0),
        histogram_range=(0.0, 500.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=50,
        expected_count=50,
        semantic_key="successor",
        evidence_quality=2,
    )

    preview_state = controller.decide(previous=previous, candidate=preview, mode="relative")
    target_state = controller.decide(previous=preview_state, candidate=target, mode="relative")

    assert preview_state.semantic_key == "successor"
    assert preview_state.display_levels == (0.0, 400.0)
    assert preview_state.evidence_quality == 1
    assert target_state.semantic_key == "successor"
    assert target_state.display_levels == (0.0, 500.0)


def test_controller_mature_successor_reanchors_immature_same_content_anchor():
    # A cold load has no incumbent, so the first (immature, reduced-LOD)
    # evidence may anchor provisionally. The mature target-quality evidence
    # for the SAME content must then re-anchor that window — the provisional
    # anchor's averaged extremes are narrower than the data and clip.
    controller = window_levels.WindowLevelController()
    preview = window_levels.LevelSource(
        levels=(45.0, 55.0),
        histogram_range=(45.0, 55.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=1,
        expected_count=1,
        semantic_key="slice-7",
        evidence_quality=1,
    )
    mature = window_levels.LevelSource(
        levels=(0.0, 100.0),
        histogram_range=(0.0, 100.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=1,
        expected_count=1,
        semantic_key="slice-7",
        evidence_quality=2,
    )

    anchored = controller.decide(previous=None, candidate=preview, mode="relative")
    settled = controller.decide(previous=anchored, candidate=mature, mode="relative")

    assert anchored.display_levels == (45.0, 55.0)
    assert settled.display_levels == (0.0, 100.0)
    assert settled.histogram_range == (0.0, 100.0)


def test_controller_settled_levels_are_path_independent_for_mature_evidence():
    # Direct load (no incumbent) and scroll (retained mature predecessor,
    # wider OR narrower than the successor's true range) must settle on the
    # same levels once mature evidence for the same content arrives. The
    # anti-shrink hysteresis may only hold between sources of equal maturity;
    # a strictly more mature successor always re-anchors.
    controller = window_levels.WindowLevelController()
    preview = window_levels.LevelSource(
        levels=(45.0, 55.0),
        histogram_range=(45.0, 55.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=1,
        expected_count=1,
        semantic_key="slice-7",
        evidence_quality=1,
    )
    mature = window_levels.LevelSource(
        levels=(0.0, 100.0),
        histogram_range=(0.0, 100.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=1,
        expected_count=1,
        semantic_key="slice-7",
        evidence_quality=2,
    )

    direct_anchor = controller.decide(previous=None, candidate=preview, mode="relative")
    direct = controller.decide(previous=direct_anchor, candidate=mature, mode="relative")

    wide_predecessor = window_levels.LevelSource(
        levels=(-20.0, 140.0),
        histogram_range=(-20.0, 140.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=1,
        expected_count=1,
        semantic_key="slice-6",
        evidence_quality=2,
    )
    retained = controller.decide(previous=wide_predecessor, candidate=preview, mode="relative")
    scroll = controller.decide(previous=retained, candidate=mature, mode="relative")

    assert retained.display_levels == (-20.0, 140.0)
    assert direct.display_levels == (0.0, 100.0)
    assert scroll.display_levels == (0.0, 100.0)
    assert scroll.display_levels == direct.display_levels


def test_controller_accepts_partial_first_source_instead_of_retaining_fallback():
    fallback = window_levels.WindowLevelController().decide(previous=None, candidate=None)
    partial = window_levels.LevelSource(
        levels=(20.0, 40.0),
        histogram_range=(20.0, 40.0),
        rank=window_levels.LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        source_count=1,
        expected_count=50,
        semantic_key="first-crop",
    )

    state = window_levels.WindowLevelController().decide(
        previous=fallback,
        candidate=partial,
        mode="relative",
    )

    assert state.semantic_key == "first-crop"
    assert state.display_levels == (20.0, 40.0)


def test_controller_absolute_partial_source_updates_histogram_without_changing_levels():
    previous = window_levels.LevelSource(
        levels=(25.0, 75.0),
        histogram_range=(0.0, 100.0),
        rank=window_levels.LevelSourceRank.EXPLICIT_USER,
        semantic_key="same",
        mode=window_levels.LevelMode.USER_LOCKED,
    )
    candidate = window_levels.LevelSource(
        levels=(200.0, 400.0),
        histogram_range=(200.0, 400.0),
        rank=window_levels.LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        source_count=1,
        expected_count=4,
        semantic_key="same",
    )

    state = window_levels.WindowLevelController().decide(
        previous=previous, candidate=candidate, mode="absolute"
    )

    assert state.display_levels == (25.0, 75.0)
    assert state.histogram_range == (0.0, 400.0)


def test_controller_absolute_same_source_keeps_numeric_levels_and_updates_histogram():
    previous = window_levels.LevelSource(
        levels=(25.0, 75.0),
        histogram_range=(0.0, 100.0),
        rank=window_levels.LevelSourceRank.EXPLICIT_USER,
        semantic_key="same",
        mode=window_levels.LevelMode.USER_LOCKED,
    )
    candidate = window_levels.LevelSource(
        levels=(200.0, 400.0),
        histogram_range=(200.0, 400.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=4,
        expected_count=4,
        semantic_key="same",
    )

    state = window_levels.WindowLevelController().decide(
        previous=previous, candidate=candidate, mode="absolute"
    )

    assert state.display_levels == (25.0, 75.0)
    assert state.histogram_range == (0.0, 400.0)


def test_relative_user_edit_is_not_absolute_user_lock():
    source = window_levels.LevelSource(
        levels=(25.0, 75.0),
        histogram_range=(0.0, 100.0),
        rank=window_levels.LevelSourceRank.PREVIOUS_COMMITTED,
        semantic_key="same",
        mode=window_levels.LevelMode.RELATIVE,
    )

    state = window_levels.state_from_source(source, mode="relative")

    assert state is not None
    assert not state.user_locked
    assert state.mode == window_levels.LevelMode.RELATIVE


def test_controller_relative_queued_levels_do_not_outrank_montage_evidence():
    candidate = window_levels.LevelSource(
        levels=(0.0, 4000.0),
        histogram_range=(0.0, 4000.0),
        rank=window_levels.LevelSourceRank.MONTAGE_COMPLETE,
        source_count=8,
        expected_count=8,
        semantic_key="same",
    )

    state = window_levels.WindowLevelController().decide(
        previous=None,
        candidate=candidate,
        user_levels=(0.0, 250.0),
        mode="relative",
    )

    assert state.display_levels == (0.0, 250.0)
    assert state.histogram_range == (0.0, 4000.0)
    assert state.source_rank == window_levels.LevelSourceRank.MONTAGE_COMPLETE
    assert not state.user_locked


def test_window_levels_module_has_no_qt_or_pyqtgraph_imports():
    tree = ast.parse(WINDOW_LEVELS_PATH.read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)
