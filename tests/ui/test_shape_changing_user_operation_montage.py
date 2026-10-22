"""Shape-changing operations, end to end through the window.

Two families live here:

- a *rank-preserving* discovered user-op shape must stay truthful through
  montage tile planning (the original test);
- a *rank-INCREASING* operation applied from the UI must present its result at
  all, which it can only do in a sibling window -- a live window's
  per-dimension widgets are fixed at construction (bundle R6).
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from arrayscope.operations import library, registry
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    frame_session_settled,
    process_events,
    use_pyqtgraph_backend,
)

_DISPOSE_TIMEOUT_MS = min(5000, INTERACTION_SETTLE_HARD_LIMIT_MS)


@pytest.fixture
def isolated_sync(monkeypatch):
    """Keep this module's compare group off any other run's sync bus."""

    monkeypatch.setenv("ARRAYSCOPE_SYNC_NAME", f"arrayscope-r6-uitest-{uuid.uuid4().hex[:12]}")


@pytest.fixture
def sibling_windows(qtbot):
    """Hand out windows and fully dispose every one of them afterwards.

    A rank-increasing apply retains its sibling in the app-global live-window
    list, so closing is not enough: a half-lived window's deferred deletions
    crash a LATER test's nested event loop. Same discipline as
    ``tests/ui/test_compare_difference.py``.
    """

    tracked = []

    def track(win):
        assert win is not None
        tracked.append(win)
        return win

    yield track

    app = QtWidgets.QApplication.instance()
    windows = [win for win in tracked if not _is_destroyed(win)]
    destroyed = {id(win): False for win in windows}
    for win in windows:
        win_key = id(win)
        win.destroyed.connect(lambda *_a, key=win_key: destroyed.__setitem__(key, True))
    for win in windows:
        win.close()
    if app is not None:
        app._arrayscope_live_windows = []
    for win in windows:
        win.deleteLater()
    qtbot.waitUntil(lambda: all(destroyed.values()), timeout=_DISPOSE_TIMEOUT_MS)


def _is_destroyed(win):
    try:
        win.objectName()
    except RuntimeError:
        return True
    return False


def test_decimating_user_operation_settles_truthful_montage_tiles(qtbot, tmp_path, monkeypatch):
    settings = use_pyqtgraph_backend()
    operations_dir = tmp_path / "operations"
    monkeypatch.setattr(library, "user_operations_directory", lambda: str(operations_dir))
    source = tmp_path / "decimate.py"
    source.write_text("def decimate(data):\n    return data[..., ::2]\n")
    operation_id = library.import_custom_operation(str(source), "decimate", changes_shape=True)

    from arrayscope.window import ArrayScopeWindow

    data = np.empty((12, 10, 12), dtype=np.float32)
    for index in range(data.shape[2]):
        data[:, :, index] = index
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        win.show()
        process_events(qtbot)
        win.operation_coordinator.load_operations((registry.create_operation(operation_id),))
        win._set_document(win.operation_coordinator.document)
        assert win.document.current_shape == (12, 10, 6)

        indices = tuple(range(6))
        win._set_view_state(
            win.view_state.with_montage_axis(2, columns=3, indices=indices, text=":")
        )
        win.render(reason="test-discovered-shape-montage")
        qtbot.waitUntil(
            lambda: (
                frame_session_settled(win)
                and win.renderer._frame_session.plan.geometry.indices == indices
            ),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        session = win.renderer._frame_session
        assert tuple(tile.montage_index for tile in session.visible_tiles) == indices
        frame = win._committed_display_frame
        payloads = frame.value_source.payloads
        for index in indices:
            payload = payloads[index]
            np.testing.assert_array_equal(
                np.asarray(payload.image),
                np.full((12, 10), index * 2, dtype=np.float32),
            )
        diagnostics = win.operation_evaluator.stage_cache_diagnostics()
        assert diagnostics.entries >= 1
        assert diagnostics.candidates_seen >= 1
    finally:
        win.close()
        library.remove_user_operation(operation_id)
        settings.clear()
        settings.sync()


# --- Rank increase: the sibling-window presentation (bundle R6) -------------


def _stack_operation_ids(win):
    """The operation ids in this window's stack, in order."""

    return tuple(registry.operation_id_for(step.operation) for step in win.document.steps)


def test_rank_increasing_operation_opens_a_compare_linked_sibling(
    qtbot, isolated_sync, sibling_windows
):
    """Applying `split_axis` from the UI shows the extra axis in a NEW window.

    The origin cannot show it: its per-dimension widgets were built for three
    axes and cannot grow. So the result goes to a sibling constructed at the
    new rank, over the same base array, carrying the whole stack as ordinary
    steps -- and the origin is left untouched.
    """

    settings = use_pyqtgraph_backend()
    from arrayscope.window import ArrayScopeWindow

    data = np.arange(4 * 8 * 5, dtype=np.float32).reshape(4, 8, 5)
    try:
        origin = sibling_windows(ArrayScopeWindow(data))
        origin.show()
        process_events(qtbot)
        # A rank-preserving op still lands in this window, unchanged.
        assert origin.request_operation("reverse", 0) is True
        assert origin.data.shape == (4, 8, 5)

        sibling = sibling_windows(origin._commit_operation("split_axis", 1, {"factor": 4}))

        # The sibling shows the extra axis...
        assert sibling is not origin
        assert sibling.data.shape == (4, 2, 4, 5)
        assert sibling.document.current_shape == (4, 2, 4, 5)
        # ...over the SAME base array, with the inherited stack visible as
        # ordinary, editable steps (reverse first, then the new split).
        assert sibling.base_data is origin.base_data
        assert _stack_operation_ids(sibling) == ("reverse", "split_axis")
        assert all(step.enabled for step in sibling.document.steps)
        # ...and its per-dimension widgets were built for four axes.
        assert sibling.per_dimension_widget_capacity() == 4

        # The origin is exactly as it was.
        assert origin.data.shape == (4, 8, 5)
        assert _stack_operation_ids(origin) == ("reverse",)
        assert not getattr(origin, "_closing", False)

        # Compare-linked, but only on the facet whose meaning survives the rank
        # change: dims would keep dragging the new axis back to the origin's
        # axis of the same number.
        assert (origin.compare_label, sibling.compare_label) == ("A", "B")
        for win in (origin, sibling):
            assert win.sync_controller.facet_enabled("levels"), win.compare_label
            assert not win.sync_controller.facet_enabled("dims"), win.compare_label
        assert "Split axis" in sibling.windowTitle()

        # Scrubbing the NEW axis in the sibling reaches a settled frame.
        for index in (0, 3):
            sibling._set_view_state(sibling.view_state.with_slice(2, index))
            sibling.render(reason="test-rank-increase-scrub")
            qtbot.waitUntil(
                lambda: frame_session_settled(sibling),
                timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
            )
            assert sibling.view_state.slice_indices[2] == index
    finally:
        settings.clear()
        settings.sync()


def test_sibling_view_recipe_round_trips_at_the_increased_rank(
    qtbot, isolated_sync, sibling_windows, tmp_path
):
    """Save the sibling's view, reload it, and get the 4-D state back.

    The recipe is written by a window whose document is rank 4 over a rank-3
    base, which is the shape ``view_state_from_mapping`` used to be unable to
    read back at all. (Its two-directional rank migration is pinned Qt-free in
    ``tests/core/test_view_session.py``.)
    """

    settings = use_pyqtgraph_backend()
    from arrayscope.core.view_recipe import load_view_recipe, save_view_recipe
    from arrayscope.window import ArrayScopeWindow

    data = np.arange(4 * 8 * 5, dtype=np.float32).reshape(4, 8, 5)
    try:
        origin = sibling_windows(ArrayScopeWindow(data))
        origin.show()
        process_events(qtbot)
        sibling = sibling_windows(origin._commit_operation("split_axis", 1, {"factor": 4}))
        sibling._set_view_state(sibling.view_state.with_slice(2, 3))
        process_events(qtbot)

        path = tmp_path / "sibling-view.json"
        save_view_recipe(str(path), sibling._current_view_recipe())

        # The recipe's steps are stated against the BASE array, so that is what
        # a load resolves them against; the view state it carries is at the
        # derived rank the steps produce.
        restored = load_view_recipe(str(path), sibling.base_data.shape, base_data=sibling.base_data)
        assert restored.view_state.ndim == 4
        assert restored.view_state.shape == (4, 2, 4, 5)
        assert restored.view_state.slice_indices[2] == 3
        assert tuple(registry.operation_id_for(step.operation) for step in restored.steps) == (
            "split_axis",
        )
    finally:
        settings.clear()
        settings.sync()


def test_re_enabling_an_inherited_rank_increasing_step_hands_off(
    qtbot, isolated_sync, sibling_windows
):
    """Disabling the split in the sibling drops a rank; re-enabling regains it.

    The drop is survivable in place (surplus widgets just hide), and so is a
    regain that stays within the rank the window was built for. Growing BEYOND
    it must not crash: the document moves to a fresh window at the new rank,
    and the one it left closes.
    """

    settings = use_pyqtgraph_backend()
    from arrayscope.window import ArrayScopeWindow

    data = np.arange(4 * 8 * 5, dtype=np.float32).reshape(4, 8, 5)
    try:
        origin = sibling_windows(ArrayScopeWindow(data))
        origin.show()
        process_events(qtbot)
        sibling = sibling_windows(origin._commit_operation("split_axis", 1, {"factor": 4}))
        process_events(qtbot)

        # Rank DECREASE, in place: the sibling keeps its four widget slots.
        sibling.set_operation_enabled(0, False)
        process_events(qtbot)
        assert sibling.data.shape == (4, 8, 5)
        assert not getattr(sibling, "_closing", False)
        assert sibling.per_dimension_widget_capacity() == 4

        # Rank INCREASE back: still within capacity, still in place.
        sibling.set_operation_enabled(0, True)
        process_events(qtbot)
        assert sibling.data.shape == (4, 2, 4, 5)
        assert not getattr(sibling, "_closing", False)

        # Now grow BEYOND the capacity the window was built for: a second split,
        # of the new step axis. This is the recipe / restore / sync shape, so
        # drive the document directly rather than through the add surfaces.
        grown = sibling.document.with_operation(registry.create_operation("split_axis", axis=2))
        handed_off = sibling_windows(sibling._set_document(grown))

        assert handed_off is not sibling
        assert handed_off.data.shape == (4, 2, 2, 2, 5)
        assert handed_off.per_dimension_widget_capacity() == 5
        assert getattr(sibling, "_closing", False)
        # The window that could not show it kept the document it WAS showing.
        assert sibling.document.current_shape == (4, 2, 4, 5)
        assert sibling.operation_coordinator.document.current_shape == (4, 2, 4, 5)
    finally:
        settings.clear()
        settings.sync()


def test_reducing_to_a_line_stays_in_this_window(qtbot, isolated_sync, sibling_windows):
    """The scalar floor is rank ZERO: a 1-D result is a line plot, not a refusal."""

    settings = use_pyqtgraph_backend()
    from arrayscope.window import ArrayScopeWindow

    data = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    try:
        win = sibling_windows(ArrayScopeWindow(data))
        win.show()
        process_events(qtbot)

        assert win.request_operation("mean", 0) is True
        assert win.data.shape == (8,)
        assert not getattr(win, "_closing", False)
    finally:
        settings.clear()
        settings.sync()
