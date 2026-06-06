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


def test_main_window_interactions_create_useful_artifacts(qt_app):
    from pyqtgraph.Qt import QtCore, QtGui

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
        _grab_widget(win.dim_containers[0], "arrayscope_dimension_controls.png", min_width=40, min_height=80)

        win._append_operation("mean", dim=1)
        _process_events(qt_app)
        assert win.data.shape == (3, 7)
        _grab_widget(win.operation_dock.widget(), "arrayscope_operation_dock.png")

        win.widgets["buttons"]["display"]["live_profile"].setChecked(True)
        win.img_view.setProfileMarker(1, 1, visible=True)
        win._on_profile_marker_moved(1, 1)
        _process_events(qt_app, count=12)
        assert win.profile_dock.isVisible()
        _grab_widget(win.profile_dock.widget, "arrayscope_profile_dock.png")

        scene_pos = win.img_view.getView().mapViewToScene(QtCore.QPointF(1, 1))
        win.getPixel(scene_pos)
        assert "(1, 1)" in win.widgets["labels"]["pixelValue"].text()
    finally:
        win.close()
        _process_events(qt_app)
