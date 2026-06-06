import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _artifact_dir():
    path = Path(os.environ.get("ARRAYSCOPE_ARTIFACT_DIR", "tests/artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _process_events(app, count=8):
    for _ in range(count):
        app.processEvents()


def _grab_widget(widget, name, *, min_width=80, min_height=40):
    from PIL import Image

    out = _artifact_dir() / name
    pixmap = widget.grab()
    assert not pixmap.isNull(), f"{name} grab returned a null pixmap"
    assert pixmap.width() >= min_width and pixmap.height() >= min_height
    assert pixmap.save(str(out), "PNG")
    assert out.stat().st_size > 1000

    image = Image.open(out).convert("RGB")
    pixels = list(image.resize((32, 32)).getdata())
    assert len(set(pixels)) > 8, f"{name} has too little pixel diversity"
    return out


def _make_data():
    return np.arange(3 * 5 * 7, dtype=float).reshape(3, 5, 7)


def _clear_arrayscope_settings():
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings("ArrayScope", "ArrayScope")
    settings.clear()
    settings.sync()


def test_main_window_interactions_create_useful_artifacts(qt_app):
    from pyqtgraph.Qt import QtCore, QtGui
    _clear_arrayscope_settings()

    for name in list(sys.modules):
        if name == "arrayscope" or name.startswith("arrayscope."):
            del sys.modules[name]

    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(_make_data())
    try:
        win.resize(900, 720)
        win.show()
        _process_events(qt_app)

        assert win.view_state.image_axes == (0, 1)
        assert not win.operation_dock.isVisible()
        assert not win.profile_dock.isVisible()
        _grab_widget(win, "arrayscope_full_initial.png")

        win.set_dimension_role("x", 2)
        _process_events(qt_app)
        assert win.view_state.image_axes == (0, 2)

        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_T,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        win.keyPressEvent(event)
        _process_events(qt_app)
        assert win.view_state.image_axes == (2, 0)
        _grab_widget(win.dimension_strip, "arrayscope_dimension_controls.png", min_width=150, min_height=30)

        win._append_operation("mean", dim=1)
        _process_events(qt_app)
        assert win.data.shape == (3, 7)
        assert win.operation_dock.isVisible()
        _grab_widget(win.operation_dock.widget(), "arrayscope_operation_dock.png")

        win.set_operation_enabled(0, False)
        _process_events(qt_app)
        assert win.data.shape == (3, 5, 7)
        win.set_operation_enabled(0, True)
        _process_events(qt_app)
        assert win.data.shape == (3, 7)

        win.widgets["buttons"]["display"]["live_profile"].setChecked(True)
        win.img_view.setProfileMarker(1, 1, visible=True)
        win._on_profile_marker_moved(1, 1)
        _process_events(qt_app, count=12)
        assert win.profile_dock.isVisible()
        _grab_widget(win.profile_dock.widget, "arrayscope_profile_dock.png")

        scene_pos = win.img_view.getView().mapViewToScene(QtCore.QPointF(1, 1))
        win.getPixel(scene_pos)
        assert win.widgets["labels"]["pixelValue"].text()
        assert win.pixel_hud.isVisible()
    finally:
        win.close()
        _process_events(qt_app)


def test_progressive_view_configuration_artifacts(qt_app):
    from pyqtgraph.Qt import QtCore
    _clear_arrayscope_settings()

    for name in list(sys.modules):
        if name == "arrayscope" or name.startswith("arrayscope."):
            del sys.modules[name]

    from arrayscope.window import ArrayScopeWindow

    quick = ArrayScopeWindow(np.arange(64 * 64, dtype=float).reshape(64, 64))
    try:
        quick.resize(900, 620)
        quick.show()
        _process_events(qt_app)
        assert not quick.operation_dock.isVisible()
        assert not quick.profile_dock.isVisible()
        _grab_widget(quick, "arrayscope_quick_glance_view.png", min_width=500, min_height=400)

        quick._append_operation("reverse", dim=0)
        _process_events(qt_app)
        assert quick.operation_dock.isVisible()
        _grab_widget(quick, "arrayscope_pipeline_view.png", min_width=500, min_height=400)

        quick.clear_operations()
        _process_events(qt_app)
        assert not quick.operation_dock.isVisible()

        quick.widgets["buttons"]["display"]["live_profile"].setChecked(True)
        quick.img_view.setProfileMarker(12, 10, visible=True)
        quick._on_profile_marker_moved(12, 10)
        _process_events(qt_app, count=12)
        assert quick.profile_dock.isVisible()
        quick.profile_dock.setFloating(False)
        quick.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, quick.profile_dock)
        _process_events(qt_app)
        _grab_widget(quick, "arrayscope_inspect_profile_view.png", min_width=500, min_height=400)
    finally:
        quick.close()
        _process_events(qt_app)

    one_d = ArrayScopeWindow(np.linspace(0, 1, 128))
    try:
        one_d.resize(760, 420)
        one_d.show()
        _process_events(qt_app)
        assert one_d.profile_dock.isVisible()
        _grab_widget(one_d, "arrayscope_1d_profile_fallback.png", min_width=400, min_height=260)
    finally:
        one_d.close()
        _process_events(qt_app)

    complex_win = ArrayScopeWindow((np.ones((16, 16)) + 1j * np.ones((16, 16))))
    try:
        complex_win.show()
        _process_events(qt_app)
        complex_win.display_toolbar.channel_combo.setCurrentIndex(complex_win.display_toolbar.channel_combo.findData("abs"))
        _process_events(qt_app)
        assert complex_win.view_state.channel.value == "abs"
        complex_win.display_toolbar.channel_combo.setCurrentIndex(complex_win.display_toolbar.channel_combo.findData("real"))
        _process_events(qt_app)
        assert complex_win.view_state.channel.value == "real"
    finally:
        complex_win.close()
        _process_events(qt_app)


def test_dimension_strip_wraps_for_many_dimensions(qt_app):
    _clear_arrayscope_settings()

    for name in list(sys.modules):
        if name == "arrayscope" or name.startswith("arrayscope."):
            del sys.modules[name]

    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 4, 5, 6, 7), dtype=float)
    win = ArrayScopeWindow(data)
    try:
        win.resize(560, 520)
        win.show()
        _process_events(qt_app)
        assert win.dimension_strip._columns == 3
        _grab_widget(win.dimension_strip, "arrayscope_dimension_strip_6d_narrow.png", min_width=560, min_height=55)

        win.resize(1180, 620)
        _process_events(qt_app)
        assert win.dimension_strip._columns > 3
        _grab_widget(win.dimension_strip, "arrayscope_dimension_strip_6d_wide.png", min_width=900, min_height=28)

        win._enable_live_profile_for_axis(2)
        _process_events(qt_app, count=12)
        assert win.widgets["buttons"]["display"]["live_profile"].isChecked()
        assert win.profile_dock.isVisible()

        win.profile_dock.hide()
        _process_events(qt_app)
        win.set_profile_axis_from_menu(3)
        _process_events(qt_app)
        assert win.profile_dock.isVisible()
    finally:
        win.close()
        _process_events(qt_app)


def test_pixel_status_label_elides_slice_context_first(qt_app):
    from arrayscope.ui.status_label import PixelStatusLabel

    label = PixelStatusLabel()
    label.resize(180, 24)
    label.set_pixel_status("(101, 64) = 3.445e-1", "d2=123 d3=456 d4=789")

    text = label._text_for_width(180)
    assert "(101, 64)" in text
    assert "..." in text

    narrow = label._text_for_width(45)
    assert "d2" not in narrow
