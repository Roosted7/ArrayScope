"""A dock must never push the main window off the screen.

Dogfood bugs 2026-07-21:

* the profile dock opened at its ``sizeHint`` -- 561 px, 44% of the window,
  because pyqtgraph's ``PlotWidget`` reports 600x480 regardless of content --
  instead of the 23% share the code already had a constant for.  Nothing
  applied that constant on the path that actually opens a dock.
* opening a dock grew the window by the dock's extent to keep the canvas a
  constant size, with NO upper bound, so the window routinely ended up
  larger than the screen: measured 1000x700 -> 1658x700 on a 1400x900 work
  area for the inspection dock, and -> 1000x1267 for the profile dock.

The work area is pinned explicitly here rather than read from the platform:
the offscreen QPA reports 800x800, which is smaller than these windows, so
reading it would make the test measure the platform instead of the clamp.
"""

import numpy as np
from pyqtgraph.Qt import QtCore

from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings
from tests.ui.helpers import process_events as _process_events
from tests.ui.helpers import wait_for_panel_preserve as _wait_for_panel_preserve

_WORK_AREA = QtCore.QSize(1400, 900)


def _window(qtbot, data=None):
    from arrayscope.window import ArrayScopeWindow

    _clear_arrayscope_settings()
    win = ArrayScopeWindow(
        np.arange(64 * 96, dtype=float).reshape(64, 96) if data is None else data
    )
    qtbot.addWidget(win)
    win.layout_manager._available_size_override = QtCore.QSize(_WORK_AREA)
    win.resize(1000, 700)
    _process_events(qtbot, count=20)
    return win


def _open(qtbot, win, dock_name):
    getattr(win.layout_manager, f"set_{dock_name}_dock_visible_from_user")(True)
    _wait_for_panel_preserve(qtbot)
    _process_events(qtbot, count=20)
    return getattr(win, f"{dock_name}_dock")


def test_profile_dock_opens_at_its_intended_share_not_its_size_hint(qtbot):
    win = _window(qtbot)
    try:
        dock = _open(qtbot, win, "profile")
        assert dock.isVisible()

        # The sizeHint is the thing that used to win; it must not be what we
        # get.  Asserting against the hint (not a bare number) keeps this
        # honest if pyqtgraph's default ever changes.
        hint = int(dock.sizeHint().height())
        assert dock.height() < hint, (
            f"profile dock opened at its sizeHint ({dock.height()} px); the "
            "23% share was never applied"
        )
        # 23% of the window, with the body's own usability minimum as a floor.
        floor = int(dock.minimumSizeHint().height())
        expected = max(floor, int(win.height() * 0.23))
        assert abs(dock.height() - expected) <= 8, (
            f"profile dock is {dock.height()} px, expected about {expected}"
        )
    finally:
        win.close()


def test_opening_a_dock_never_pushes_the_window_past_the_work_area(qtbot):
    for dock_name in ("profile", "inspection", "operation"):
        win = _window(qtbot)
        try:
            _open(qtbot, win, dock_name)
            assert win.width() <= _WORK_AREA.width(), (
                f"{dock_name} dock pushed the window to {win.width()} px wide "
                f"on a {_WORK_AREA.width()} px work area"
            )
            assert win.height() <= _WORK_AREA.height(), (
                f"{dock_name} dock pushed the window to {win.height()} px tall "
                f"on a {_WORK_AREA.height()} px work area"
            )
        finally:
            win.close()


def test_the_window_grows_first_and_the_canvas_yields_only_at_the_ceiling(qtbot):
    """The canvas is only allowed to shrink once there is nowhere left to grow."""

    win = _window(qtbot)
    try:
        # Plenty of headroom: the window absorbs the dock, canvas untouched.
        win.layout_manager._available_size_override = QtCore.QSize(4096, 4096)
        before_window = win.width()
        before_canvas = win.centralWidget().width()
        _open(qtbot, win, "inspection")
        assert win.width() > before_window, "the window should grow to fit the dock"
        assert abs(win.centralWidget().width() - before_canvas) <= 2, (
            "the canvas must not shrink while the window still has room to grow"
        )
    finally:
        win.close()

    # No headroom at all: the window cannot grow, so the canvas must yield.
    win = _window(qtbot)
    try:
        win.layout_manager._available_size_override = QtCore.QSize(win.width(), win.height())
        before_window = win.width()
        before_canvas = win.centralWidget().width()
        _open(qtbot, win, "inspection")
        assert win.width() <= before_window + 1, "the window grew past the work area"
        assert win.centralWidget().width() < before_canvas, (
            "at the ceiling the canvas must yield the space instead"
        )
    finally:
        win.close()


def test_a_clamped_show_hide_cycle_gives_the_canvas_its_pixels_back(qtbot):
    """The screen clamp must not ratchet the canvas down.

    A clamped show shrinks the canvas; the matching hide frees exactly the
    room that was missing, so the canvas has to come back.  Without the debt
    bookkeeping the hide faithfully preserved the ALREADY shrunken canvas and
    the window never recovered -- measured 662 -> 392 px across one cycle.
    """

    win = _window(qtbot)
    try:
        win.layout_manager._available_size_override = QtCore.QSize(win.width(), win.height())
        before_canvas = win.centralWidget().width()

        _open(qtbot, win, "inspection")
        assert win.centralWidget().width() < before_canvas, "expected a clamped shrink"

        win.layout_manager.set_inspection_dock_visible_from_user(False)
        _wait_for_panel_preserve(qtbot)
        _process_events(qtbot, count=20)

        assert abs(win.centralWidget().width() - before_canvas) <= 2, (
            f"canvas ratcheted down: {before_canvas} -> {win.centralWidget().width()}"
        )
    finally:
        win.close()


def test_a_manually_resized_dock_keeps_its_size_across_a_reopen(qtbot):
    """The share is a DEFAULT, not a policy: a user's own size must survive."""

    win = _window(qtbot)
    try:
        win.layout_manager._available_size_override = QtCore.QSize(4096, 4096)
        dock = _open(qtbot, win, "profile")
        chosen = int(dock.height()) + 90
        win.resizeDocks([dock], [chosen], QtCore.Qt.Orientation.Vertical)
        _process_events(qtbot, count=20)
        chosen = int(dock.height())

        win.layout_manager.set_profile_dock_visible_from_user(False)
        _wait_for_panel_preserve(qtbot)
        _process_events(qtbot, count=20)
        _open(qtbot, win, "profile")

        assert abs(int(dock.height()) - chosen) <= 8, (
            f"reopened at {dock.height()} px, discarding the user's {chosen} px"
        )
    finally:
        win.close()


def test_a_ceiling_pinned_transition_stops_retrying_instead_of_burning_attempts(qtbot):
    """The correction loop must not retry a target the screen makes unreachable.

    When the window is pinned at the work area and the canvas is still short,
    no number of retries can change anything -- the loop used to spend all
    four of them (4 x 16 ms) re-measuring, then report the generic
    ``best_effort_unsettled``.  Stopping early is both faster and honest
    about WHY the canvas did not keep its size.
    """

    win = _window(qtbot)
    try:
        win.layout_manager._available_size_override = QtCore.QSize(win.width(), win.height())
        _open(qtbot, win, "inspection")

        diagnostics = win.layout_manager.canvas_preserver.diagnostics()
        assert diagnostics.last_result == "clamped", (
            f"expected an honest 'clamped' result, got {diagnostics.last_result!r}"
        )
        assert diagnostics.attempts_used < 4, (
            f"burned {diagnostics.attempts_used} attempts on an unreachable target"
        )
        assert any(event.startswith("clamp") for event in diagnostics.events), (
            "the clamp should be visible in the diagnostics trail"
        )
    finally:
        win.close()
