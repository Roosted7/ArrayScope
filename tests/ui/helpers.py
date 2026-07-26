import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def process_events(qtbot, count=8):
    for _ in range(count):
        qtbot.wait(10)


def clear_arrayscope_settings():
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings()
    settings.clear()
    settings.sync()


def view_action(win, text):
    for action in win.menuBar().actions():
        if action.text() == "View":
            for child in action.menu().actions():
                if child.text() == text:
                    return child
    raise AssertionError(f"View action not found: {text}")


def panel_body(panel):
    return panel.body


def assert_panel_invariants(win, name, expected_location):
    from pyqtgraph.Qt import QtWidgets

    from arrayscope.window.panels import PanelLocation

    panel = win.panel_manager._panels_by_name[name]
    assert panel.location == expected_location
    if expected_location == PanelLocation.HIDDEN:
        assert panel.dialog is None
        assert not panel.dock.isVisible()
        assert panel.body is not None
    elif expected_location == PanelLocation.DETACHED:
        assert panel.dialog is not None
        assert (
            panel.dialog.findChild(type(panel.body)) is panel.body
            or panel.body.parent() is not None
        )
        assert not panel.dock.isVisible()
    elif expected_location == PanelLocation.DOCKED:
        assert panel.dialog is None
        assert panel.dock.isVisible()
        assert QtWidgets.QDockWidget.widget(panel.dock) is panel.body


def wait_for_panel_preserve(qtbot):
    process_events(qtbot, count=50)


def assert_size_close(actual, expected, tolerance=1):
    assert abs(actual.width() - expected.width()) <= tolerance
    assert abs(actual.height() - expected.height()) <= tolerance


# --- Live backend-window harness --------------------------------------------


def use_pyqtgraph_backend(extra_settings=None):
    """Point QSettings at the PyQtGraph backend; returns the settings object."""

    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice

    clear_arrayscope_settings()
    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.PYQTGRAPH.value)
    for key, value in (extra_settings or {}).items():
        settings.setValue(key, value)
    settings.sync()
    return settings


def use_wgpu_backend(extra_settings=None):
    """Point QSettings at the explicit wgpu backend; returns the settings object."""

    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice

    require_wgpu_adapter()
    clear_arrayscope_settings()
    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.WGPU.value)
    for key, value in (extra_settings or {}).items():
        settings.setValue(key, value)
    settings.sync()
    return settings


def restore_default_backend(settings):
    from arrayscope.app.settings_state import ImageRenderingBackendChoice

    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.PYQTGRAPH.value)
    settings.sync()


def make_backend_window(qtbot, data, *, backend, require_gpu_atlas=False):
    """Build an ArrayScopeWindow on the given backend, skipping if unavailable."""

    import pytest

    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from arrayscope.window import ArrayScopeWindow

    if backend == "wgpu":
        require_wgpu_adapter()
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    capabilities = image_view_backend_capabilities(win.img_view)
    if capabilities.name != backend:
        win.close()
        pytest.skip(f"{backend} backend unavailable in this Qt environment")
    if require_gpu_atlas:
        assert capabilities.tile_residency_kind == "gpu_atlas"
    return win


def require_wgpu_adapter() -> None:
    """Skip an explicit WGPU test when no Vulkan adapter exists locally."""

    import contextlib

    import pytest

    wgpu = pytest.importorskip("wgpu")
    from wgpu.backends.wgpu_native.extras import set_instance_extras

    with contextlib.suppress(RuntimeError):
        set_instance_extras(backends=["Vulkan"])
    try:
        wgpu.gpu.request_adapter_sync(power_preference="low-power")
    except Exception as exc:
        pytest.skip(f"no WGPU Vulkan adapter: {exc}")


def frame_session_settled(win) -> bool:
    """The settle triple every live gate builds on; committed-frame currency
    checks stay with the caller."""

    session = getattr(win.renderer, "_frame_session", None)
    if session is None:
        return False
    return bool(
        session.visible_plan_complete()
        and not win.montage_tile_evaluation_controller.is_busy()
        and session.required_target_settled()
    )


def apply_plane(win, index, *, reason):
    win._set_view_state(win.view_state.with_slice(0, index))
    win.render(reason=reason)


def plane_settled(win, index) -> bool:
    frame = getattr(win, "_committed_display_frame", None)
    if frame is None:
        return False
    if int(frame.geometry.view_state.slice_indices[0]) != int(index):
        return False
    return frame_session_settled(win)


def committed_value(win, view_x, view_y):
    """Probe the committed frame the way live hover does (render.py)."""

    geometry = win.renderer.display_geometry
    if geometry is None:
        return None
    context = geometry.context_for_view_point(float(view_x), float(view_y))
    if context is None:
        return None
    return win.renderer._hover_value_from_display(context.mapping)


def give_generous_work_area(win, width=4096, height=4096):
    """Pin a work area far larger than any fixture window.

    The offscreen QPA reports an 800x800 screen -- SMALLER than several of
    these fixtures' own windows.  The screen clamp (added so an opening dock
    can no longer push the window off the desktop) would therefore fire in
    situations a real desktop cannot produce, and the affected tests would be
    measuring the platform rather than the behaviour they exist to pin.

    Tests about canvas-preservation MECHANICS call this so the clamp stays
    out of the way.  The clamp's own behaviour is pinned separately, with a
    deliberately small work area, in test_dock_screen_fit.py.
    """

    from pyqtgraph.Qt import QtCore

    win.layout_manager._available_size_override = QtCore.QSize(int(width), int(height))
