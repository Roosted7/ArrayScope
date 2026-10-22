"""Application launch and process handling for ArrayScope."""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
import weakref

import numpy as np

from arrayscope.app.qt_binding import pin_wayland_appearance, prefer_pyside6

prefer_pyside6()

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.window import ArrayScopeWindow

try:
    from IPython import get_ipython
except ImportError:

    def get_ipython():
        return None


def _prepare_qt_environment():
    """Apply conservative Qt workarounds before QApplication creation."""
    # Deliberate display-server choice on Linux/Wayland (qt_platform setting).
    # Must run before the QPA plugin is initialized; inert when a
    # QApplication already exists, when QT_QPA_PLATFORM is already set
    # (including by the CLI supervisor), or off Linux/Wayland.
    if not _qt_application_exists():
        from arrayscope.app.qt_platform import (
            apply_qt_platform_env,
            platform_choice_applies,
            read_qt_platform_setting,
            resolve_qt_platform,
        )

        if platform_choice_applies():
            decision = resolve_qt_platform(read_qt_platform_setting(), cli=False)
            apply_qt_platform_env(decision)

    # On XWayland/XCB, Qt's MIT-SHM path can fail with:
    #   qt.qpa.xcb: xcb_shm_create_segment() failed
    #   The X11 connection broke
    # This must be set before QApplication / the QPA plugin is initialized.
    if "xcb" in (os.environ.get("QT_QPA_PLATFORM") or ""):
        os.environ.setdefault("QT_X11_NO_MITSHM", "1")

    # After the platform block, so its offscreen/minimal guard reads the
    # *resolved* platform. Here and not at module import: an import-time pin
    # fired before the test harness could say "offscreen", so a desktop
    # session's WAYLAND_DISPLAY leaked the portal theme into headless test
    # processes — whose settle loops then paid real DBus round-trips
    # (measured: 11.6 s -> 30.5 s on one settle-budget test, deterministic).
    pin_wayland_appearance()

    # Known-harmless terminal noise (wgpu's optional-Vulkan-extension probe,
    # QtWayland text-input leave events): silenced here so the very first
    # adapter request and Wayland surface are already covered.
    from arrayscope.app.log_noise import (
        apply_qt_logging_noise_rules,
        install_wgpu_log_noise_filter,
    )

    apply_qt_logging_noise_rules()
    install_wgpu_log_noise_filter()


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


class _WindowReferenceKeeper(QtCore.QObject):
    """Let the application own visible windows without owning closed ones.

    A close can be rejected, so release only after Qt has dispatched it. The
    window owns the single-shot timer; destroying it cancels that callback.
    Weak references avoid a signal closure pinning the window after release.
    """

    def __init__(self, app, win):
        super().__init__(win)
        self._app = weakref.ref(app)
        self._window = weakref.ref(win)
        self._close_check = QtCore.QTimer(self)
        self._close_check.setSingleShot(True)
        self._close_check.timeout.connect(self._release_if_hidden)
        win.installEventFilter(self)
        # A child QObject receiver is disconnected when its parent starts
        # destruction. Keep this callback receiver-free, and capture weak
        # references only, so destruction itself can release app ownership.
        app_ref, window_ref = self._app, self._window

        def release_destroyed_window(*_):
            app, window = app_ref(), window_ref()
            if app is not None:
                refs = getattr(app, "_arrayscope_live_windows", [])
                app._arrayscope_live_windows = [ref for ref in refs if ref is not window]

        win.destroyed.connect(release_destroyed_window)

    def retain(self):
        app, win = self._app(), self._window()
        if app is None or win is None:
            return
        refs = getattr(app, "_arrayscope_live_windows", None)
        refs = list(refs) if isinstance(refs, list) else []
        if not any(ref is win for ref in refs):
            refs.append(win)
            app._arrayscope_live_windows = refs

    def _release(self, *_):
        app, win = self._app(), self._window()
        if app is None:
            return
        refs = getattr(app, "_arrayscope_live_windows", None)
        if isinstance(refs, list):
            app._arrayscope_live_windows = [ref for ref in refs if ref is not win]

    def _release_if_hidden(self):
        win = self._window()
        if win is not None and not win.isVisible():
            self._release()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Close:
            self._close_check.start(0)
        elif event.type() == QtCore.QEvent.Type.Show:
            self._close_check.stop()
            self.retain()
        return super().eventFilter(obj, event)


def _retain_window_reference(app, win):
    """Keep an inline window alive while open, without taking caller ownership."""
    keeper = getattr(win, "_arrayscope_reference_keeper", None)
    if keeper is None:
        keeper = _WindowReferenceKeeper(app, win)
        win._arrayscope_reference_keeper = keeper
    keeper.retain()


def _create_window(
    data,
    title="",
    complex_dim=None,
    filepath=None,
    dataset_path=None,
    selector_class_name=None,
    axes=None,
    application_name="ArrayScope",
):
    _prepare_qt_environment()

    app = pg.mkQApp()
    app.setOrganizationName("ArrayScope")
    app.setApplicationName(str(application_name))
    app.setStyle("Fusion")

    win = ArrayScopeWindow(
        data,
        complex_dim=complex_dim,
        filepath=filepath,
        dataset_path=dataset_path,
        selector_class_name=selector_class_name,
        axes=axes,
    )
    # Closing a launched viewer is terminal: closeEvent shuts down its
    # workers and renderer. Delete the Qt tree as well so signal closures
    # cannot retain its document and source in a long-lived notebook/CLI app.
    win.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
    win.setWindowTitle(title)
    win.show()

    return app, win


def _run_window(
    data,
    title="",
    complex_dim=None,
    filepath=None,
    dataset_path=None,
    selector_class_name=None,
    axes=None,
):
    """Open a viewer window in this process and block on the Qt event loop."""
    try:
        app, _win = _create_window(
            data,
            title=title,
            complex_dim=complex_dim,
            filepath=filepath,
            dataset_path=dataset_path,
            selector_class_name=selector_class_name,
            axes=axes,
        )
        return app.exec()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise


def _show_window_inline(
    data,
    title="",
    complex_dim=None,
    filepath=None,
    dataset_path=None,
    selector_class_name=None,
    axes=None,
):
    """Open a viewer window in this process without starting app.exec()."""
    app, win = _create_window(
        data,
        title=title,
        complex_dim=complex_dim,
        filepath=filepath,
        dataset_path=dataset_path,
        selector_class_name=selector_class_name,
        axes=axes,
    )

    _retain_window_reference(app, win)

    return win


def arrayscope(
    data,
    title="",
    block=False,
    complex_dim=None,
    filepath=None,
    dataset_path=None,
    selector_class_name=None,
    axes=None,
):
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    if data.ndim < 1:
        raise ValueError("data must have at least 1 dimension")

    kwargs = {
        "filepath": filepath,
        "dataset_path": dataset_path,
        "selector_class_name": selector_class_name,
        "axes": axes,
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
