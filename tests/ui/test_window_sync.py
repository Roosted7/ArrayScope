"""Linked-window sync between two real windows over the local-socket bus.

Both windows live in one process; the transport is the same QLocalServer/
QLocalSocket path that separately started ArrayScope processes use, so this
exercises the full publish -> broker -> apply chain including the sync
toggle buttons, origin/revision echo suppression, and clamping.
"""

import uuid

import numpy as np
import pytest

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings

pytest.importorskip("pytestqt")


@pytest.fixture
def sync_group(monkeypatch):
    name = f"arrayscope-sync-uitest-{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("ARRAYSCOPE_SYNC_NAME", name)
    return name


@pytest.fixture
def make_window(qtbot, sync_group):
    windows = []

    def make(data):
        from arrayscope.window import ArrayScopeWindow

        win = ArrayScopeWindow(data)
        windows.append(win)
        qtbot.addWidget(win)
        return win

    _clear_arrayscope_settings()
    yield make
    for win in windows:
        win.close()


def _enable_all_facets(win):
    win.display_toolbar.sync_window_action.setChecked(True)
    win.sync_dims_button.setChecked(True)
    win.operation_dock.sync_button.setChecked(True)
    win.inspection_dock.sync_button.setChecked(True)


_DEFAULT_SETTLE_TIMEOUT_MS = min(4000, INTERACTION_SETTLE_HARD_LIMIT_MS)


def _settled(
    qtbot,
    predicate,
    timeout_ms=_DEFAULT_SETTLE_TIMEOUT_MS,
):
    qtbot.waitUntil(
        lambda: bool(predicate()),
        timeout=min(int(timeout_ms), INTERACTION_SETTLE_HARD_LIMIT_MS),
    )


def test_sync_buttons_toggle_controller_facets(qtbot, make_window):
    win = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    controller = win.sync_controller
    assert not any(
        controller.facet_enabled(facet) for facet in ("levels", "dims", "operations", "rois")
    )
    _enable_all_facets(win)
    assert all(
        controller.facet_enabled(facet) for facet in ("levels", "dims", "operations", "rois")
    )
    assert controller.bus.is_running()
    win.display_toolbar.sync_window_action.setChecked(False)
    win.sync_dims_button.setChecked(False)
    win.operation_dock.sync_button.setChecked(False)
    win.inspection_dock.sync_button.setChecked(False)
    assert not controller.bus.is_running()


def test_dimension_indexing_syncs_between_windows(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.sync_dims_button.setChecked(True)
    win_b.sync_dims_button.setChecked(True)

    win_a.widgets["spins"]["slice_indices"][2].setValue(3)
    _settled(qtbot, lambda: win_b.view_state.slice_indices[2] == 3)
    # The mirrored spinbox follows the applied state.
    assert win_b.widgets["spins"]["slice_indices"][2].value() == 3


def test_first_change_publishes_on_leading_edge_without_coalesce_wait(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.sync_dims_button.setChecked(True)
    win_b.sync_dims_button.setChecked(True)

    # A discrete change after a quiet period must go out immediately, not
    # sit in the trailing coalesce timer: no publish timer may be pending
    # for the dims facet right after the change.
    win_a.widgets["spins"]["slice_indices"][2].setValue(3)

    timer = win_a.sync_controller._publish_timers.get("dims")
    assert timer is None or not timer.isActive()
    _settled(qtbot, lambda: win_b.view_state.slice_indices[2] == 3)


def test_burst_of_changes_coalesces_through_trailing_timer(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.sync_dims_button.setChecked(True)
    win_b.sync_dims_button.setChecked(True)

    # Leading edge for the first step, trailing coalesce for the burst; the
    # final value must still arrive.
    for value in (1, 2, 3):
        win_a.widgets["spins"]["slice_indices"][2].setValue(value)

    _settled(qtbot, lambda: win_b.view_state.slice_indices[2] == 3)


def test_sustained_changes_publish_periodically_and_trail_final_value(
    qtbot, make_window, monkeypatch
):
    # Drives schedule_publish directly with a fake clock. Going through the
    # spinbox cascade is racy here: event processing inside setValue can fire
    # the real trailing QTimer mid-step, re-stamping the fake-clock publish
    # bookkeeping. The widget -> controller path is covered by the leading-edge
    # and burst-coalesce tests above.
    win = make_window(np.arange(8 * 6 * 8, dtype=float).reshape(8, 6, 8))
    controller = win.sync_controller
    win.sync_dims_button.setChecked(True)

    now = [1000.0]
    published = []
    value = [0]
    # Patch the namespace the controller's methods actually resolve
    # `monotonic` from. Importing arrayscope.sync.controller here can yield a
    # different module object when another test re-imported the package, and
    # patching that copy would leave the controller on the real clock.
    controller_globals = controller.schedule_publish.__func__.__globals__
    monkeypatch.setitem(controller_globals, "monotonic", lambda: now[0])
    monkeypatch.setattr(controller.bus, "publish", published.append)
    monkeypatch.setattr(
        controller, "_build_payload", lambda facet: {"slice_indices": [0, 0, value[0]]}
    )

    def _set(new_value):
        value[0] = new_value
        controller.schedule_publish("dims")

    def _state_values():
        return [
            message["payload"]["slice_indices"][2]
            for message in published
            if message.get("kind") == "state"
        ]

    _set(1)
    assert _state_values() == [1]

    now[0] += 0.060
    _set(2)
    assert _state_values() == [1]

    now[0] += 0.130
    _set(3)
    timer = controller._publish_timers.get("dims")
    assert timer is None or not timer.isActive()
    assert _state_values() == [1, 3]

    now[0] += 0.060
    _set(4)
    timer = controller._publish_timers.get("dims")
    assert timer is not None
    assert timer.isActive()

    qtbot.waitUntil(lambda: len(_state_values()) == 3, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
    assert _state_values() == [1, 3, 4]


def test_dimension_sync_clamps_for_smaller_arrays_and_ignores_extra_dims(qtbot, make_window):
    win_a = make_window(np.arange(9 * 6 * 4, dtype=float).reshape(9, 6, 4))
    win_b = make_window(np.arange(5 * 3, dtype=float).reshape(5, 3))
    win_a.sync_dims_button.setChecked(True)
    win_b.sync_dims_button.setChecked(True)

    win_a.widgets["spins"]["slice_indices"][0].setValue(8)
    _settled(qtbot, lambda: win_b.view_state.slice_indices[0] == 4)  # clamped to size 5
    assert win_b.view_state.ndim == 2


def test_window_levels_sync_between_windows(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.display_toolbar.sync_window_action.setChecked(True)
    win_b.display_toolbar.sync_window_action.setChecked(True)
    _settled(
        qtbot,
        lambda: (
            (
                win_a.sync_controller.bus.role == "broker"
                and win_a.sync_controller.bus.peer_count >= 1
            )
            or (
                win_b.sync_controller.bus.role == "broker"
                and win_b.sync_controller.bus.peer_count >= 1
            )
        ),
    )

    win_a.renderer._apply_display_level_override((10.0, 90.0), emit_user=True)
    _settled(
        qtbot,
        lambda: tuple(round(float(v), 3) for v in win_b.img_view.getLevels()) == (10.0, 90.0),
    )

    index = win_a.display_toolbar.window_combo.findData("absolute")
    win_a.display_toolbar.window_combo.setCurrentIndex(index)
    _settled(qtbot, lambda: win_b._current_window_mode() == "absolute")


def test_operations_sync_between_windows(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.operation_dock.sync_button.setChecked(True)
    win_b.operation_dock.sync_button.setChecked(True)

    assert win_a.request_operation("mean", 2)
    _settled(qtbot, lambda: len(win_b.document.steps) == 1)
    operation = win_b.document.steps[0].operation
    assert type(operation).__name__.lower().startswith("mean")
    assert int(operation.axis) == 2

    win_a.clear_operations()
    _settled(qtbot, lambda: len(win_b.document.steps) == 0)


def test_incompatible_operation_recipe_is_skipped_without_error(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(5 * 3, dtype=float).reshape(5, 3))
    win_a.operation_dock.sync_button.setChecked(True)
    win_b.operation_dock.sync_button.setChecked(True)

    # Axis 2 does not exist on the 2-D receiver; it must skip, not crash.
    assert win_a.request_operation("mean", 2)
    _settled(qtbot, lambda: len(win_a.document.steps) == 1)
    qtbot.wait(600)
    assert len(win_b.document.steps) == 0


def test_rois_sync_between_windows(qtbot, make_window):
    from arrayscope.core.roi import RoiKind

    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.inspection_dock.sync_button.setChecked(True)
    win_b.inspection_dock.sync_button.setChecked(True)

    selection = win_a.img_view.createRoi(RoiKind.RECTANGLE, rect=(1.0, 1.0, 2.0, 2.0))
    _settled(qtbot, lambda: len(win_b.roi_store.selections) == 1)
    mirrored = win_b.roi_store.selections[0]
    assert mirrored.id == selection.id
    assert mirrored.geometry.kind == RoiKind.RECTANGLE

    win_a._clear_rois()
    _settled(qtbot, lambda: len(win_b.roi_store.selections) == 0)


def test_joining_window_pulls_group_state(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.sync_dims_button.setChecked(True)
    win_a.widgets["spins"]["slice_indices"][1].setValue(5)
    qtbot.wait(300)

    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    assert win_b.view_state.slice_indices[1] != 5
    win_b.sync_dims_button.setChecked(True)
    # Joining requests the group's state instead of pushing its own.
    _settled(qtbot, lambda: win_b.view_state.slice_indices[1] == 5)
    assert win_a.view_state.slice_indices[1] == 5


def test_sync_does_not_feedback_loop(qtbot, make_window):
    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.sync_dims_button.setChecked(True)
    win_b.sync_dims_button.setChecked(True)

    win_a.widgets["spins"]["slice_indices"][0].setValue(6)
    _settled(qtbot, lambda: win_b.view_state.slice_indices[0] == 6)
    qtbot.wait(400)
    revisions_a = dict(win_a.sync_controller._revisions)
    revisions_b = dict(win_b.sync_controller._revisions)
    qtbot.wait(600)
    assert dict(win_a.sync_controller._revisions) == revisions_a
    assert dict(win_b.sync_controller._revisions) == revisions_b
    assert win_a.view_state.slice_indices[0] == 6
    assert win_b.view_state.slice_indices[0] == 6


def test_dimension_sync_renders_receiver_even_with_pending_presentation_draw(qtbot, make_window):
    """A received dimension change must re-render the receiver even when it is a
    background window with a pending presentation draw.

    Regression: _apply_dims used the interactive render path, which the render
    coordinator defers behind a pending presentation draw. A window that is not
    actively repainting never clears that flag, so the interactive request was
    starved and the displayed frame never updated even though view_state did --
    dimension sync looked completely broken while levels/operations/ROIs (which
    render directly) worked.
    """

    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.sync_dims_button.setChecked(True)
    win_b.sync_dims_button.setChecked(True)

    # Simulate a background receiver whose presentation draw stays pending.
    win_b.img_view.presentationDrawPending = lambda: True

    rendered_reasons = []
    original_render = win_b.renderer.render

    def _tracking_render(*args, **kwargs):
        rendered_reasons.append(kwargs.get("reason"))
        return original_render(*args, **kwargs)

    win_b.renderer.render = _tracking_render

    win_a.widgets["spins"]["slice_indices"][2].setValue(3)
    _settled(qtbot, lambda: win_b.view_state.slice_indices[2] == 3)
    # The frame must actually render, not just the view_state update.
    _settled(qtbot, lambda: any("sync-dims" in (reason or "") for reason in rendered_reasons))


def test_dimension_role_and_transpose_sync_between_windows(qtbot, make_window):
    win_a = make_window(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    win_b = make_window(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    win_a.sync_dims_button.setChecked(True)
    win_b.sync_dims_button.setChecked(True)
    qtbot.wait(300)

    rendered_reasons = []
    original_render = win_b.renderer.render

    def _tracking_render(*args, **kwargs):
        rendered_reasons.append(kwargs.get("reason"))
        return original_render(*args, **kwargs)

    win_b.renderer.render = _tracking_render

    win_a.set_dimension_role("x", 2)
    _settled(qtbot, lambda: win_b.view_state.image_axes == (0, 2))
    assert win_b.dimension_strip.chip(2).x_button.isChecked()
    assert win_b.view_state.slice_indices == win_a.view_state.slice_indices

    win_a.transposeView(None)
    _settled(qtbot, lambda: win_b.view_state.image_axes == (2, 0))
    assert win_b.dimension_strip.chip(2).y_button.isChecked()
    assert any("sync-dims" in (reason or "") for reason in rendered_reasons)


def test_dimension_axis_flip_syncs_without_slice_change(qtbot, make_window):
    win_a = make_window(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    win_b = make_window(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    win_a.sync_dims_button.setChecked(True)
    win_b.sync_dims_button.setChecked(True)
    qtbot.wait(300)

    rendered_reasons = []
    original_render = win_b.renderer.render

    def _tracking_render(*args, **kwargs):
        rendered_reasons.append(kwargs.get("reason"))
        return original_render(*args, **kwargs)

    win_b.renderer.render = _tracking_render
    before_slices = win_b.view_state.slice_indices

    # Clicking the already-selected X role flips that axis; slice indices stay unchanged.
    win_a.set_dimension_role("x", 1)
    _settled(qtbot, lambda: win_b.view_state.axis_flipped[1])
    qtbot.wait(250)

    assert win_b.view_state.slice_indices == before_slices
    assert any("sync-dims" in (reason or "") for reason in rendered_reasons)


def _view_center(view_box):
    (x0, x1), (y0, y1) = view_box.viewRange()
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def test_camera_pan_zoom_syncs_between_windows(qtbot, make_window):
    from arrayscope.sync.messages import FACET_CAMERA

    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.sync_controller.set_facet_enabled(FACET_CAMERA, True)
    win_b.sync_controller.set_facet_enabled(FACET_CAMERA, True)
    qtbot.wait(300)

    vb_a = win_a.img_view.getView()
    vb_b = win_b.img_view.getView()
    initial_b_center = _view_center(vb_b)

    # Zoom into a sub-region that stays inside the image content, so the
    # viewport-constraint pass does not reshape the target out from under the
    # assertion. Data/world-space coordinates: the two windows must land on the
    # same region regardless of their individual viewport pixel sizes.
    vb_a.setRange(xRange=(1.0, 5.0), yRange=(1.0, 5.0), padding=0)

    def _b_follows_a():
        bx, by = _view_center(vb_b)
        ax, ay = _view_center(vb_a)
        return abs(bx - ax) < 0.75 and abs(by - ay) < 0.75

    _settled(qtbot, _b_follows_a)
    # Oracle can fail: the receiver must actually have moved off its initial
    # fit, not merely match A because nothing changed.
    moved_x = abs(_view_center(vb_b)[0] - initial_b_center[0])
    moved_y = abs(_view_center(vb_b)[1] - initial_b_center[1])
    assert moved_x > 0.5 or moved_y > 0.5


def test_camera_apply_does_not_echo_a_republish(qtbot, make_window):
    """Applying a peer camera state must move this window without republishing.

    Loop prevention is the same layering the other facets rely on: the apply
    runs inside the controller's ``_applying`` window, so the ViewBox range
    signal it triggers is suppressed and the receiver's revision does not
    advance (mirrors test_sync_does_not_feedback_loop for dims).
    """
    from arrayscope.sync.messages import FACET_CAMERA

    win_a = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_b = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win_a.sync_controller.set_facet_enabled(FACET_CAMERA, True)
    win_b.sync_controller.set_facet_enabled(FACET_CAMERA, True)
    qtbot.wait(300)

    win_a.img_view.getView().setRange(xRange=(15.0, 45.0), yRange=(5.0, 35.0), padding=0)
    _settled(
        qtbot,
        lambda: win_b.sync_controller._last_applied.get(FACET_CAMERA) is not None,
    )
    revision_b = win_b.sync_controller._revisions[FACET_CAMERA]
    # Let any (incorrectly) scheduled republish fire.
    qtbot.wait(500)
    assert win_b.sync_controller._revisions[FACET_CAMERA] == revision_b


def test_dimension_role_change_publishes_payload_when_slices_are_unchanged(qtbot, make_window):
    from arrayscope.sync.messages import FACET_DIMS

    win = make_window(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    win.sync_dims_button.setChecked(True)
    qtbot.wait(300)
    controller = win.sync_controller
    before_revision = controller._revisions[FACET_DIMS]
    before_payload = controller._build_payload(FACET_DIMS)

    win.set_dimension_role("x", 2)
    _settled(qtbot, lambda: controller._revisions[FACET_DIMS] > before_revision)
    after_payload = controller._last_payload[FACET_DIMS]

    assert after_payload["slice_indices"] == before_payload["slice_indices"]
    assert after_payload["image_axes"] == [0, 2]
    assert after_payload != before_payload
