import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _process_events(qtbot, count=8):
    for _ in range(count):
        qtbot.wait(10)


def _clear_arrayscope_settings():
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings("ArrayScope", "ArrayScope")
    settings.clear()
    settings.sync()


def test_render_preserves_viewport_for_same_display_shape(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        view = win.img_view.getView()
        view.setRange(xRange=(1, 3), yRange=(1, 3), padding=0)
        before = view.viewRange()

        win._on_slice_index_changed(2, 1)
        _process_events(qtbot, count=20)

        np.testing.assert_allclose(view.viewRange(), before, atol=1e-9)
    finally:
        win.close()


def test_dock_show_hide_preserves_viewport(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        view = win.img_view.getView()
        view.setRange(xRange=(2, 5), yRange=(2, 6), padding=0)
        before = view.viewRange()

        win.toggle_profile_dock()
        _process_events(qtbot, count=12)
        win.toggle_profile_dock()
        _process_events(qtbot, count=12)

        np.testing.assert_allclose(view.viewRange(), before, atol=1e-9)
    finally:
        win.close()


def test_montage_roi_gap_source_is_nan(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=40)

        gap_x = win._current_montage_geometry.tile_width
        source = win._roi_source_image()

        assert np.isnan(source[:, gap_x]).all()
    finally:
        win.close()


def test_strict_ui_mode_raises_callback_exceptions(monkeypatch):
    from arrayscope.app.errors import handle_ui_exception

    monkeypatch.setenv("ARRAYSCOPE_STRICT_UI", "1")

    with pytest.raises(RuntimeError, match="boom"):
        handle_ui_exception("test callback", RuntimeError("boom"))
