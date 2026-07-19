"""Screen-present-path gate for the wgpu view (queue row 3, ceiling program).

Real-Wayland-only oracles for the two known native-child risks plus the
present edge itself:

* INPUT — a paint-less native child (a wl_subsurface) sits above the parent
  surface; the transparent ``ArrayScopeGraphicsView`` must still own every
  pointer/keyboard event.  ``WA_TransparentForMouseEvents`` on the canvas is
  the contract; ``childAt`` + the live drag/close tests are the oracles.
* DRAW-ACK — acknowledgements must key on the REAL present edge
  (``wgpuSurfacePresent``), and ``presentationDrawPending`` must never stay
  armed (the gate-armed-forever family).

Run (serial, opens a window):

    ARRAYSCOPE_GPU_TESTS=1 XDG_RUNTIME_DIR=/run/user/$(id -u) \
    WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland \
    python -m pytest tests/gpu_interaction/test_wgpu_screen_present.py -n 0 -q
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.gpu_interaction.conftest import wait_for_qt_condition

pytestmark = pytest.mark.gpu_interaction

pytest.importorskip("wgpu")


@pytest.fixture()
def screen_view(qt_app):
    from arrayscope.display.wgpu_imageview2d import WgpuImageView2D

    view = WgpuImageView2D(present_method="screen")
    view.resize(420, 320)
    view.show()
    for _ in range(30):
        qt_app.processEvents()
    if view.wgpuPresentMethod() != "screen":
        reason = view.wgpuPresentMethodFallbackReason()
        view.close()
        pytest.fail(f"screen present path did not activate: {reason}")
    try:
        yield view
    finally:
        view.close()
        view.teardown_surface()
        for _ in range(20):
            qt_app.processEvents()


def _commit_ramp(view):
    from tests.display.test_imageview2d import _present_tiled

    canvas = np.linspace(0.0, 1.0, 64 * 96, dtype=np.float32).reshape(64, 96)
    return _present_tiled(
        view,
        canvas,
        histogramData=canvas.copy(),
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
    )


def test_commit_presents_on_the_swapchain_and_drains_the_draw_gate(qt_app, screen_view):
    view = screen_view
    report = _commit_ramp(view)
    assert set(report.presented_tiles) == {0}

    assert wait_for_qt_condition(
        qt_app,
        lambda: view.wgpuPresentationDiagnostics()["wgpu_screen_presents"] >= 1
        and not view.presentationDrawPending(),
        timeout_s=10.0,
    ), f"draw gate never drained: {view.wgpuPresentationDiagnostics()}"

    diagnostics = view.wgpuPresentationDiagnostics()
    assert diagnostics["wgpu_present_method"] == "screen"
    assert diagnostics["wgpu_last_draw_error"] == ""
    # The ack counter must have caught up with every presentation request —
    # the draw-count pair IS the never-armed-forever oracle.
    assert (
        diagnostics["tile_presentation_draw_count"]
        == diagnostics["tile_presentation_request_count"]
    )
    # Present-mode evidence (Fifo acquire blocks ~15 ms/frame, gate-B): the
    # chosen mode is recorded; on Mailbox the steady-state acquire must not
    # pay the vsync block.
    assert diagnostics["wgpu_screen_present_mode"] in ("mailbox", "fifo")
    if diagnostics["wgpu_screen_present_mode"] == "mailbox":
        assert diagnostics["wgpu_screen_acquire_ms_last"] < 10.0


def test_resize_reconfigures_and_keeps_presenting(qt_app, screen_view):
    view = screen_view
    _commit_ramp(view)
    assert wait_for_qt_condition(
        qt_app,
        lambda: view.wgpuPresentationDiagnostics()["wgpu_screen_presents"] >= 1,
        timeout_s=10.0,
    )
    before = view.wgpuPresentationDiagnostics()["wgpu_screen_presents"]

    view.resize(560, 400)
    for _ in range(10):
        qt_app.processEvents()
    view._request_wgpu_canvas_draw()
    assert wait_for_qt_condition(
        qt_app,
        lambda: view.wgpuPresentationDiagnostics()["wgpu_screen_presents"] > before
        and not view.presentationDrawPending(),
        timeout_s=10.0,
    ), f"no present after resize: {view.wgpuPresentationDiagnostics()}"
    diagnostics = view.wgpuPresentationDiagnostics()
    assert diagnostics["wgpu_last_draw_error"] == ""
    # The swapchain follows the widget's physical size.
    context = view._wgpu_canvas._context
    assert (context._wgpu_config.width, context._wgpu_config.height) == tuple(
        view._wgpu_canvas.get_physical_size()
    )


def test_native_child_is_input_transparent(qt_app, screen_view):
    """Qt hit-testing over the canvas must resolve to the interaction view,
    never the native child (WA_TransparentForMouseEvents contract)."""

    from pyqtgraph.Qt import QtCore

    view = screen_view
    _commit_ramp(view)
    canvas = view._wgpu_canvas
    assert canvas.testAttribute(
        QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
    )
    assert canvas.focusPolicy() == QtCore.Qt.FocusPolicy.NoFocus
    center = canvas.geometry().center()
    target = view._display_container.childAt(center)
    assert target is not None and target is not canvas
    # The deepest widget under the pointer is the graphics view's viewport.
    viewport = view.graphicsView.viewport()
    assert target is viewport, f"input target is {target!r}, not the viewport"


def test_close_cancels_active_drag_on_screen_path(qt_app, screen_view):
    """Contract twin of test_widget_close_cancels_active_pointer_interaction,
    pinned here so the input risk is covered even when the contract suite is
    not run on Wayland."""

    from pyqtgraph.Qt import QtCore

    from arrayscope.core.roi import RoiKind
    from arrayscope.display.interaction import PointerPhase
    from tests.display.test_imageview2d import (
        _seed_displayed_image,
        _send_viewport_mouse,
    )

    view = screen_view
    _seed_displayed_image(view, np.zeros((20, 20), dtype=float))
    view.getView().setRange(xRange=(0, 20), yRange=(0, 20), padding=0)
    view.createRoi(RoiKind.RECTANGLE, rect=(2.0, 3.0, 4.0, 5.0))
    assert _send_viewport_mouse(
        view,
        QtCore.QEvent.Type.MouseButtonPress,
        (4.0, 5.0),
        button=QtCore.Qt.MouseButton.LeftButton,
    )
    assert view.interactionState().phase is not PointerPhase.IDLE
    view.close()
    assert view.interactionState().phase is PointerPhase.IDLE


def test_grab_presented_framebuffer_sees_the_committed_content(qt_app, screen_view):
    """The harness capture oracle: a Qt widget grab is blind to swapchain
    pixels (all-red screen journey matrix, 2026-07-19), so the physical
    readback must show the committed ramp where the widget grab cannot."""

    view = screen_view
    _commit_ramp(view)
    assert wait_for_qt_condition(
        qt_app,
        lambda: view.wgpuPresentationDiagnostics()["wgpu_screen_presents"] >= 1,
        timeout_s=10.0,
    )
    frame = view.grabPresentedFramebuffer()
    assert frame is not None and frame.ndim == 3 and frame.shape[2] == 4
    gray = frame[..., :3].astype(np.float32).mean(axis=2)
    # The ramp spans dark to bright; a blind capture would be uniform.
    assert float(gray.max() - gray.min()) > 100.0
    assert float((gray > 10.0).mean()) > 0.05


def test_auto_present_method_activates_screen_on_wayland(qt_app):
    """AUTO's whole point: on a live Wayland session it flips screen on."""

    from arrayscope.display.wgpu_imageview2d import WgpuImageView2D

    view = WgpuImageView2D(present_method="auto")
    view.resize(420, 320)
    view.show()
    try:
        for _ in range(30):
            qt_app.processEvents()
        assert view.wgpuPresentMethod() == "screen", (
            view.wgpuPresentMethodFallbackReason()
        )
        diagnostics = view.wgpuPresentationDiagnostics()
        assert diagnostics["wgpu_present_method_requested"] == "auto"
        assert diagnostics["wgpu_present_method_fallback_reason"] == ""
    finally:
        view.close()
        view.teardown_surface()
        for _ in range(20):
            qt_app.processEvents()
