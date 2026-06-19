import ast
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
WINDOW_LEVELS_PATH = ROOT / "arrayscope" / "core" / "window_levels.py"
SPEC = importlib.util.spec_from_file_location("arrayscope.core.window_levels", WINDOW_LEVELS_PATH)
window_levels = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = window_levels
SPEC.loader.exec_module(window_levels)


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

    state = window_levels.WindowLevelController().decide(previous=previous, candidate=candidate, mode="relative")

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

    state = window_levels.WindowLevelController().decide(previous=previous, candidate=candidate, mode="relative")

    assert state.display_levels == (100.0, 300.0)
    assert state.histogram_range == (0.0, 400.0)
    assert state.source_rank == window_levels.LevelSourceRank.MONTAGE_COMPLETE
    assert state.source_count == 4


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

    state = window_levels.WindowLevelController().decide(previous=previous, candidate=candidate, mode="absolute")

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

    state = window_levels.WindowLevelController().decide(previous=previous, candidate=candidate, mode="absolute")

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
