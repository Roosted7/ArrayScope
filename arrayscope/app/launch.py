"""Application launch and process handling for ArrayScope."""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings

import numpy as np

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from arrayscope.window import ArrayScopeWindow

try:
    from IPython import get_ipython
except ImportError:
    def get_ipython():
        return None


def _prepare_qt_environment():
    """Apply conservative Qt workarounds before QApplication creation."""
    # On XWayland/XCB, Qt's MIT-SHM path can fail with:
    #   qt.qpa.xcb: xcb_shm_create_segment() failed
    #   The X11 connection broke
    # This must be set before QApplication / the QPA plugin is initialized.
    if os.environ.get("QT_QPA_PLATFORM") == "xcb":
        os.environ.setdefault("QT_X11_NO_MITSHM", "1")


def _qt_application_exists():
    """Return True if Qt has already created a QApplication in this process."""
    try:
        return QtWidgets.QApplication.instance() is not None
    except Exception:
        return False


def _enable_ipython_qt_event_loop():
    """Try to enable IPython's Qt input hook for responsive inline windows."""
    ip = get_ipython()

    if ip is None:
        return False

    active_eventloop = getattr(ip, "active_eventloop", None)
    if active_eventloop in {"qt", "qt5", "qt6"}:
        return True

    try:
        ip.enable_gui("qt")
    except Exception:
        return False

    active_eventloop = getattr(ip, "active_eventloop", None)
    return active_eventloop in {"qt", "qt5", "qt6"}


def _retain_window_reference(app, win):
    """Keep inline windows alive for as long as they are open."""
    live_windows = app.property("_arrayscope_live_windows")
    if not isinstance(live_windows, list):
        live_windows = []

    live_windows.append(win)
    app.setProperty("_arrayscope_live_windows", live_windows)

    def _release_reference(_=None, w=win, qapp=app):
        refs = qapp.property("_arrayscope_live_windows")
        if not isinstance(refs, list):
            return
        try:
            refs.remove(w)
        except ValueError:
            pass
        qapp.setProperty("_arrayscope_live_windows", refs)

    win.destroyed.connect(_release_reference)


def _create_window(data, title="", complex_dim=None, filepath=None,
                   dataset_path=None, selector_class_name=None):
    _prepare_qt_environment()

    app = pg.mkQApp()
    app.setStyle("Fusion")

    win = ArrayScopeWindow(
        data,
        complex_dim=complex_dim,
        filepath=filepath,
        dataset_path=dataset_path,
        selector_class_name=selector_class_name,
    )
    win.setWindowTitle(title)
    win.show()

    return app, win


def _run_window(data, title="", complex_dim=None, filepath=None,
                dataset_path=None, selector_class_name=None):
    """Open a viewer window in this process and block on the Qt event loop."""
    try:
        app, _win = _create_window(
            data,
            title=title,
            complex_dim=complex_dim,
            filepath=filepath,
            dataset_path=dataset_path,
            selector_class_name=selector_class_name,
        )
        return app.exec()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise


def _show_window_inline(data, title="", complex_dim=None, filepath=None,
                        dataset_path=None, selector_class_name=None):
    """Open a viewer window in this process without starting app.exec()."""
    app, win = _create_window(
        data,
        title=title,
        complex_dim=complex_dim,
        filepath=filepath,
        dataset_path=dataset_path,
        selector_class_name=selector_class_name,
    )

    _retain_window_reference(app, win)

    return win


def arrayscope(data, title="", block=False, complex_dim=None, filepath=None,
               dataset_path=None, selector_class_name=None):
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    if data.ndim < 1:
        raise ValueError("data must have at least 1 dimension")

    kwargs = {
        "filepath": filepath,
        "dataset_path": dataset_path,
        "selector_class_name": selector_class_name,
    }

    if block:
        return _run_window(data, title, complex_dim, **kwargs)

    if _qt_application_exists():
        if _enable_ipython_qt_event_loop():
            return _show_window_inline(data, title, complex_dim, **kwargs)

        warnings.warn(
            "Qt is already initialized, so arrayscope cannot safely fork a child "
            "process for non-blocking display. No active IPython Qt event-loop "
            "integration was detected, so an inline non-blocking window may freeze. "
            "Falling back to blocking mode in the current process. To avoid this, "
            "use arrayscope(..., block=True) explicitly, or in IPython run `%gui qt` "
            "before calling arrayscope(...).",
            RuntimeWarning,
            stacklevel=2,
        )
        return _run_window(data, title, complex_dim, **kwargs)

    process = mp.Process(target=_run_window, args=(data, title, complex_dim), kwargs=kwargs)
    process.start()
    return process
