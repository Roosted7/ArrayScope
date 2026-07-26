"""Feature-parity tests against the ImageSurface contract (roadmap Y2).

Every test here runs on both backends and asserts the shared semantic
contract, not widget internals: a behavior fix in shell logic that only
reaches one backend must fail here.
"""

import contextlib
import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from tests.display.test_imageview2d import (
    _present_tiled,
    _seed_displayed_image,
    _send_viewport_mouse,
    _single_tile_geometry,
    _view_class,
)


def _wgpu_adapter_available() -> bool:
    try:
        import wgpu
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        with contextlib.suppress(RuntimeError):  # instance already exists
            # Vulkan-only instance BEFORE the first adapter request: letting
            # the probe create an all-backends instance makes GL adapter
            # enumeration re-init EGL on some test hosts.
            set_instance_extras(backends=["Vulkan"])
        wgpu.gpu.request_adapter_sync(power_preference="low-power")
        return True
    except Exception:
        return False


def _live_wayland_session() -> bool:
    """True only when the suite actually runs on the wayland QPA.

    The default ring pins ``QT_QPA_PLATFORM=offscreen`` (tests/conftest.py),
    so the screen-path twin below runs only in explicit real-Wayland
    invocations — the same opt-in shape as ``tests/gpu_interaction``.
    """

    return os.environ.get("QT_QPA_PLATFORM") == "wayland" and bool(
        os.environ.get("WAYLAND_DISPLAY")
    )


BACKENDS = (
    "pyqtgraph",
    pytest.param(
        "wgpu",
        marks=pytest.mark.skipif(
            not _wgpu_adapter_available(), reason="no wgpu adapter on this machine"
        ),
    ),
    # Focused twin for the native-Wayland screen present path (queue row 3):
    # the full contract runs against present_method="screen"; the view factory
    # fails loudly if the screen path does not activate.
    pytest.param(
        "wgpu-screen",
        marks=pytest.mark.skipif(
            not (_live_wayland_session() and _wgpu_adapter_available()),
            reason="wgpu screen path needs a live Wayland session and adapter",
        ),
    ),
)


def _shown_view(backend, qt_app):
    view = _view_class(backend)()
    view.resize(320, 260)
    view.show()
    return view


@pytest.mark.parametrize("backend", BACKENDS)
def test_view_satisfies_image_surface_protocol(qt_app, backend):
    from arrayscope.display.backends.base import ImageSurface, surface_for_view

    view = _shown_view(backend, qt_app)
    try:
        surface = surface_for_view(view)
        assert isinstance(surface, ImageSurface)
        assert isinstance(surface.presentation_diagnostics(), dict)
    finally:
        view.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_interaction_event_owner_is_shared_across_backends(qt_app, backend):
    view = _shown_view(backend, qt_app)
    try:
        assert view.interaction_event_owner() == "shared-controller"
    finally:
        view.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_tiled_commit_acknowledges_presented_payloads(qt_app, backend):
    view = _shown_view(backend, qt_app)
    try:
        canvas = np.linspace(0.0, 1.0, 20 * 30, dtype=np.float32).reshape(20, 30)
        report = _present_tiled(
            view,
            canvas,
            histogramData=canvas.copy(),
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
        )
        assert report is not None
        assert set(report.presented_tiles) == {0}
        assert view.montageDisplayMode() != "none"
    finally:
        view.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_montage_commit_acknowledges_each_loaded_tile(qt_app, backend):
    """A 2-tile montage commit must acknowledge both tiles on every backend."""

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState

    view = _shown_view(backend, qt_app)
    try:
        canvas = np.linspace(0.0, 1.0, 20 * 60, dtype=np.float32).reshape(20, 60)
        geometry = DisplayGeometry(
            view_state=ViewState.from_shape((20, 60)).with_image_axes(0, 1),
            display_shape=(20, 60),
            montage=MontageGeometry(indices=(0, 1), tile_shape=(20, 30), columns=2, rows=1, gap=0),
            montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
        )
        report = _present_tiled(
            view,
            canvas,
            histogramData=canvas.copy(),
            geometry=geometry,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
        )
        assert report is not None
        assert set(report.presented_tiles) == {0, 1}
        assert view.montageDisplayMode() != "none"
    finally:
        view.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_hide_tiled_presentation_deactivates_surface(qt_app, backend):
    view = _shown_view(backend, qt_app)
    try:
        canvas = np.zeros((10, 12), dtype=np.float32)
        _present_tiled(
            view, canvas, histogramData=canvas, levels=(0.0, 1.0), histogramRange=(0.0, 1.0)
        )
        assert view.montageDisplayMode() != "none"
        view.hide_tiled_presentation("contract-test")
        assert view.montageDisplayMode() == "none"
    finally:
        view.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_invalidate_tiled_presentation_hides_pixels_but_retains_residency(qt_app, backend):
    view = _shown_view(backend, qt_app)
    try:
        canvas = np.zeros((10, 12), dtype=np.float32)
        _present_tiled(
            view, canvas, histogramData=canvas, levels=(0.0, 1.0), histogramRange=(0.0, 1.0)
        )
        if backend == "pyqtgraph":
            layer = view._montage_tile_layer
            resident_before = len(layer.states)
        else:
            executor = view._wgpu_executor
            resident_before = len(executor.page_table.resident_keys())

        view.invalidate_tiled_presentation("semantic-transition")

        assert view.montageDisplayMode() == "none"
        assert resident_before > 0
        if backend == "pyqtgraph":
            assert len(layer.states) == resident_before
            assert all(
                not state.visible and not state.item.isVisible() for state in layer.states.values()
            )
        else:
            # Residency (page-table entries) survives; no tile is drawn.
            assert len(executor.page_table.resident_keys()) == resident_before
            assert len(executor._tiles) == 0
    finally:
        view.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_reset_tiled_residency_survives_and_recommits(qt_app, backend):
    view = _shown_view(backend, qt_app)
    try:
        canvas = np.zeros((10, 12), dtype=np.float32)
        _present_tiled(
            view, canvas, histogramData=canvas, levels=(0.0, 1.0), histogramRange=(0.0, 1.0)
        )
        view.reset_tiled_residency("contract-test")
        report = _present_tiled(
            view,
            canvas,
            histogramData=canvas,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
        )
        assert set(report.presented_tiles) == {0}
    finally:
        view.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_profile_marker_set_and_hide_share_semantics(qt_app, backend):
    view = _shown_view(backend, qt_app)
    try:
        _seed_displayed_image(view, np.zeros((20, 20), dtype=float))
        moved = []
        view.setProfileMarkerCallback(lambda x, y: moved.append((x, y)))
        view.setProfileMarker(5.0, 7.0, visible=True)
        assert view.profileMarkerPosition() == pytest.approx((5.0, 7.0))
        view.hideProfileMarker()
        # A hidden marker reports no position on any backend.
        assert view.profileMarkerPosition() is None
        assert not view._profile_marker_requested_visible
    finally:
        view.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_preview_levels_route_through_shared_driver(qt_app, backend):
    view = _shown_view(backend, qt_app)
    try:
        _seed_displayed_image(view, np.linspace(0.0, 4.0, 400, dtype=np.float32).reshape(20, 20))
        view._apply_histogram_preview_levels((0.5, 3.5))
        assert view._displayLevels == pytest.approx((0.5, 3.5))
        preview = view._histogram_preview_controller
        assert preview.last_applied_levels == pytest.approx((0.5, 3.5))
    finally:
        view.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_widget_close_cancels_active_pointer_interaction(qt_app, backend):
    """Closing the widget must cancel a live drag on every backend.

    Regression: a backend close override once dropped `_cancel_interaction`,
    so a drag could survive widget close on one surface only.
    """

    from pyqtgraph.Qt import QtCore

    from arrayscope.core.roi import RoiKind
    from arrayscope.display.interaction import PointerPhase

    view = _shown_view(backend, qt_app)
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_viewport_camera_rect_reflects_committed_geometry(qt_app, backend):
    view = _shown_view(backend, qt_app)
    try:
        canvas = np.zeros((16, 24), dtype=np.float32)
        _present_tiled(
            view, canvas, histogramData=canvas, levels=(0.0, 1.0), histogramRange=(0.0, 1.0)
        )
        geometry = _single_tile_geometry(canvas)
        rect = view._current_image_viewport_rect()
        assert rect is not None
        x0, y0, x1, y1 = rect
        assert (x1 - x0, y1 - y0) == pytest.approx(
            (float(geometry.display_shape[1]), float(geometry.display_shape[0])),
            abs=1.0,
        )
    finally:
        view.close()
