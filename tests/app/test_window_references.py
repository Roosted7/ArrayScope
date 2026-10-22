"""Launched viewers release their Qt tree; embedded widgets stay caller-owned."""

import gc
import weakref

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.app.launch import _create_window, _retain_window_reference
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS


@pytest.fixture(autouse=True)
def _preserve_application_state(qapp):
    organization, name = qapp.organizationName(), qapp.applicationName()
    style = qapp.style().objectName()
    palette, stylesheet = qapp.palette(), qapp.styleSheet()
    tokens = qapp.property("_arrayscope_theme_tokens")
    plot_colors = {key: pg.getConfigOption(key) for key in ("background", "foreground")}
    try:
        yield
    finally:
        qapp.setOrganizationName(organization)
        qapp.setApplicationName(name)
        qapp.setStyle(style)
        qapp.setPalette(palette)
        qapp.setStyleSheet(stylesheet)
        qapp.setProperty("_arrayscope_theme_tokens", tokens)
        pg.setConfigOptions(**plot_colors)


def _retained(app):
    return getattr(app, "_arrayscope_live_windows", [])


def test_closed_window_is_released_and_can_be_collected(qtbot, qapp):
    win = QtWidgets.QWidget()
    win.show()
    _retain_window_reference(qapp, win)
    _retain_window_reference(qapp, win)
    assert sum(ref is win for ref in _retained(qapp)) == 1
    alive = weakref.ref(win)

    win.close()
    qtbot.waitUntil(lambda window=win: window not in _retained(qapp))
    del win
    gc.collect()

    assert alive() is None


def test_direct_destruction_releases_a_window_without_a_close_event(qapp):
    win = QtWidgets.QWidget()
    win.show()
    _retain_window_reference(qapp, win)
    alive = weakref.ref(win)
    del win
    gc.collect()
    assert alive() is not None
    assert len(_retained(qapp)) == 1

    # Destruction must clear retention itself: no Close or timer ran here.
    alive().deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert _retained(qapp) == []
    gc.collect()
    assert alive() is None


def test_delete_on_close_viewers_release_widgets_and_sources_repeatedly(
    qtbot, qapp, monkeypatch, tmp_path
):
    # A first matplotlib import with an unwritable home logs an exception.
    # pytest keeps that traceback, including the whole constructing window's
    # frame, until test teardown: that is harness ownership, not a viewer leak.
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))
    for _ in range(2):
        app, win = _create_window(np.arange(16, dtype=float).reshape(4, 4))
        _retain_window_reference(app, win)
        window_ref = weakref.ref(win)
        source_ref = weakref.ref(win.document.base_data)

        win.close()
        qapp.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        assert win not in _retained(qapp)
        del win

        def collected(window_ref=window_ref, source_ref=source_ref):
            # Queued startup callbacks can own their Python arguments until
            # Qt drains them, even after the widget itself has been deleted.
            qapp.processEvents()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            gc.collect()
            return window_ref() is None and source_ref() is None

        qtbot.waitUntil(collected, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)


def test_rejected_close_and_immediate_reopen_keep_the_window_owned(qtbot, qapp):
    class MayRejectClose(QtWidgets.QWidget):
        reject_close = True

        def closeEvent(self, event):
            if self.reject_close:
                event.ignore()
            else:
                super().closeEvent(event)

    win = MayRejectClose()
    qtbot.addWidget(win)
    win.show()
    _retain_window_reference(qapp, win)
    assert not win.close()
    qapp.processEvents()
    assert win.isVisible()
    assert win in _retained(qapp)

    win.reject_close = False
    assert win.close()
    win.show()
    qapp.processEvents()
    assert win in _retained(qapp)

    win.close()
    qtbot.waitUntil(lambda: win not in _retained(qapp))
    win.show()
    assert win in _retained(qapp)
    win.close()
    qtbot.waitUntil(lambda: win not in _retained(qapp))


def test_launched_viewer_deletes_qt_tree_but_preserves_caller_python_attributes(qtbot):
    from shiboken6 import isValid

    app, win = _create_window(np.arange(16, dtype=float).reshape(4, 4))
    _retain_window_reference(app, win)
    assert win in _retained(app)
    try:
        assert win.testAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        win.close()
        app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        assert win not in _retained(app)
        assert not isValid(win)
        # A notebook may still hold and inspect its returned window after close.
        assert win.document.base_data.shape == (4, 4)
    finally:
        if isValid(win):
            win.close()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
