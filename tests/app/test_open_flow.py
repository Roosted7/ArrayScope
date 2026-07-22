"""Async file-open flow: loading window, streaming viewer, error/cancel paths.

These are offscreen Qt tests; they drive real reader threads against small
files and pump the event loop until the session reaches a terminal state.
"""

import threading
import time
import types

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore


@pytest.fixture(autouse=True)
def _hermetic_backend_and_teardown():
    """Pin the offscreen render backend and guarantee a clean teardown.

    The offscreen contract resolves ``AUTO`` to PyQtGraph, but under a serial
    (``-n 0``) run there is no per-worker QSettings isolation, so a developer's
    persisted ``image_rendering_backend=wgpu`` would leak in and drive these
    async-loading tests through the GPU present path (validated separately in
    the ring-4 GPU suite). Pin PyQtGraph so the flow is exercised
    deterministically regardless of machine settings.

    On teardown, cancel any in-flight load sessions, join their daemon reader
    threads, and close lingering windows so the interpreter exits cleanly
    rather than segfaulting at shutdown with live Qt objects and threads.
    """
    import pyqtgraph as pg

    from arrayscope.app import open_flow

    settings = QtCore.QSettings()
    previous = settings.value("image_rendering_backend")
    settings.setValue("image_rendering_backend", "pyqtgraph")
    settings.sync()
    try:
        yield
    finally:
        app = pg.mkQApp()
        # Cancel and join any daemon reader threads still in flight.
        for session in list(open_flow._ACTIVE_SESSIONS):
            session.cancel()
            session.wait(timeout=5.0)
        open_flow._ACTIVE_SESSIONS.clear()
        # Close windows and release the references retained for inline display.
        for widget in list(app.topLevelWidgets()):
            widget.close()
        app.setProperty("_arrayscope_live_windows", None)
        # Uninstall and delete the app-level file-open event filter installed by
        # ensure_open_app(). Leaving a Python QObject event filter attached to
        # the persistent QApplication segfaults the interpreter at shutdown
        # ("event filter cannot be in a different thread").
        open_filter = app.property("_arrayscope_file_open_filter")
        if open_filter is not None:
            app.removeEventFilter(open_filter)
            app.setProperty("_arrayscope_file_open_filter", None)
            open_filter.setParent(None)
            open_filter.deleteLater()
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if previous is None:
            settings.remove("image_rendering_backend")
        else:
            settings.setValue("image_rendering_backend", previous)
        settings.sync()


@pytest.fixture
def open_app():
    from arrayscope.app.open_flow import ensure_open_app

    return ensure_open_app()


def _pump_until(app, predicate, timeout_s=30.0):
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
    assert predicate(), "condition not reached before timeout"


def _run_session(app, session, timeout_s=30.0):
    terminal = {"done": False}
    session.done.connect(lambda: terminal.__setitem__("done", True))
    _pump_until(app, lambda: terminal["done"], timeout_s)
    return session


def test_open_path_async_opens_viewer_with_loaded_data(open_app, tmp_path):
    from arrayscope.app.open_flow import open_path_async

    arr = np.random.default_rng(0).normal(size=(6, 8, 4)).astype(np.float32)
    path = tmp_path / "vol.npy"
    np.save(path, arr)

    session = _run_session(open_app, open_path_async(path))
    try:
        assert session.window is not None
        assert session.loading_window is None
        np.testing.assert_array_equal(np.asarray(session.window.base_data), arr)
        assert session.window.windowTitle() == "vol.npy [numpy]"
    finally:
        if session.window is not None:
            session.window.close()
        open_app.processEvents()


def test_open_path_async_consume_removes_handoff_file(open_app, tmp_path):
    from arrayscope.app.open_flow import open_path_async

    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = tmp_path / "handoff.npy"
    np.save(path, arr)

    session = _run_session(open_app, open_path_async(path, mmap=True, consume=True))
    try:
        assert session.window is not None
        assert not path.exists()
        np.testing.assert_array_equal(np.asarray(session.window.base_data), arr)
    finally:
        if session.window is not None:
            session.window.close()
        open_app.processEvents()


def test_open_path_async_failure_shows_error_state(open_app, tmp_path):
    from arrayscope.app.open_flow import open_path_async

    path = tmp_path / "broken.npy"
    path.write_bytes(b"not a numpy file at all")

    session = _run_session(open_app, open_path_async(path))
    try:
        assert session.window is None
        assert session.loading_window is not None
        assert session.loading_window._cancel_btn.text() == "Close"
    finally:
        if session.loading_window is not None:
            session.loading_window.close_quietly()
        open_app.processEvents()


def test_open_path_async_cancel_before_read_closes_quietly(open_app, tmp_path):
    from arrayscope.app.open_flow import FileOpenSession

    arr = np.zeros((64, 64), dtype=np.float32)
    path = tmp_path / "vol.npy"
    np.save(path, arr)

    session = FileOpenSession(path)
    session.cancel()
    session.start()
    _run_session(open_app, session)

    assert session.window is None
    assert session.loading_window is None


def test_open_any_path_rejects_unknown_suffix(open_app, tmp_path, capsys):
    from arrayscope.app.open_flow import open_any_path

    path = tmp_path / "data.xyz"
    path.write_bytes(b"?")

    assert not open_any_path(path)
    assert "Unsupported file type" in capsys.readouterr().err


def test_closing_viewer_mid_load_cancels_reader(open_app, tmp_path, monkeypatch):
    """Closing the streaming viewer window mid-load must cancel the reader
    thread (so it stops streaming into a detached array) and must not let any
    terminal handler mutate the closed window.

    Red-first: on the pre-fix code, closing the viewer does not set the cancel
    event, so the reader stays blocked forever and ``session.wait`` returns
    False (the thread never joins). The fix wires the viewer's Close event to
    ``cancel`` so the reader observes cancellation and exits.
    """
    from arrayscope.app.open_flow import open_path_async
    from arrayscope.io import progressive

    arr = np.zeros((32, 32), dtype=np.float32)
    path = tmp_path / "vol.npy"
    np.save(path, arr)

    # Block the reader deterministically right after the streaming probe has
    # opened the viewer: the first cancel poll parks until the cancel event is
    # set, then re-raises through the real check as LoadCancelled.
    reader_parked = threading.Event()
    original_check = progressive._check_cancel

    def blocking_check(cancel):
        reader_parked.set()
        if cancel is not None:
            cancel.wait()  # released only when the load is cancelled
        original_check(cancel)

    monkeypatch.setattr(progressive, "_check_cancel", blocking_check)

    session = open_path_async(path)
    _pump_until(open_app, lambda: session.window is not None and reader_parked.is_set())
    win = session.window

    # Spy on the window mutations a terminal handler would perform.
    title_calls = []
    original_set_title = win.setWindowTitle
    monkeypatch.setattr(
        win, "setWindowTitle", lambda t: (title_calls.append(t), original_set_title(t))[1]
    )
    notify_calls = []
    original_notify = win.notify_data_changed
    monkeypatch.setattr(
        win,
        "notify_data_changed",
        lambda **k: (notify_calls.append(k), original_notify(**k))[1],
    )

    win.close()
    open_app.processEvents()

    # (a) The reader thread is cancelled and joins promptly.
    assert session.wait(timeout=10.0)
    # Let any queued terminal signal (cancelled/finished) be delivered.
    _pump_until(open_app, lambda: session._terminal)
    # (b) No terminal handler ran against the closed window.
    assert title_calls == []
    assert notify_calls == []


def test_finished_handler_ignores_closed_viewer(open_app, tmp_path, monkeypatch):
    """A ``finished`` signal that races in after the user closed the viewer must
    not retitle/refresh the closed window.

    Red-first: pre-fix, ``_on_finished`` unconditionally sets the window title
    and refreshes; here we assert it does neither once the viewer is closed.
    """
    from arrayscope.app.open_flow import open_path_async
    from arrayscope.io import progressive

    arr = np.zeros((32, 32), dtype=np.float32)
    path = tmp_path / "vol.npy"
    np.save(path, arr)

    # Park the reader on a test-controlled gate (independent of cancel) so the
    # session stays non-terminal while we drive the closed-window handler.
    reader_parked = threading.Event()
    release = threading.Event()
    original_check = progressive._check_cancel

    def gated_check(cancel):
        reader_parked.set()
        release.wait()
        original_check(cancel)

    monkeypatch.setattr(progressive, "_check_cancel", gated_check)

    session = open_path_async(path)
    try:
        _pump_until(open_app, lambda: session.window is not None and reader_parked.is_set())
        win = session.window

        title_calls = []
        original_set_title = win.setWindowTitle
        monkeypatch.setattr(
            win, "setWindowTitle", lambda t: (title_calls.append(t), original_set_title(t))[1]
        )
        notify_calls = []
        original_notify = win.notify_data_changed
        monkeypatch.setattr(
            win,
            "notify_data_changed",
            lambda **k: (notify_calls.append(k), original_notify(**k))[1],
        )

        win.close()
        open_app.processEvents()
        assert session._window_closed

        # Simulate the reader's finished signal arriving after the close.
        loaded = types.SimpleNamespace(
            data=np.zeros((32, 32), dtype=np.float32),
            axes=None,
            metadata={"detected_format": "numpy"},
        )
        session._on_finished(loaded)

        assert title_calls == []
        assert notify_calls == []
    finally:
        release.set()
        session.cancel()
        session.wait(timeout=10.0)


def test_streaming_rec_updates_viewer_before_completion(open_app, tmp_path, monkeypatch):
    """The viewer opens on the streaming probe, with a status widget, and the
    status widget disappears once the load completes."""
    from arrayscope.app import open_flow
    from arrayscope.app.open_flow import open_path_async
    from tests.io.test_progressive import _write_rec_pair

    path, expected = _write_rec_pair(tmp_path, size=8, n_slices=4)

    # Slow the reader down so the probe reliably arrives before completion.
    from arrayscope.io.file_interpreters import PhilipsRECLoader

    original = PhilipsRECLoader._next_slice

    def slowed(self, fid, img_idx):
        time.sleep(0.05)
        return original(self, fid, img_idx)

    monkeypatch.setattr(PhilipsRECLoader, "_next_slice", slowed)
    monkeypatch.setattr(open_flow, "_STREAM_REFRESH_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(open_flow, "_STREAM_REFRESH_MIN_FRACTION_STEP", 0.0)

    session = open_path_async(path)
    _pump_until(open_app, lambda: session.window is not None)
    saw_status_widget = session._status_widget is not None
    autolevel_refreshes = []
    original_notify = session.window.notify_data_changed

    def record_notify(*, force_autolevel=False):
        autolevel_refreshes.append(bool(force_autolevel))
        return original_notify(force_autolevel=force_autolevel)

    monkeypatch.setattr(session.window, "notify_data_changed", record_notify)
    _run_session(open_app, session)
    try:
        assert saw_status_widget
        assert session._status_widget is None
        assert autolevel_refreshes
        assert all(autolevel_refreshes)
        np.testing.assert_array_equal(np.asarray(session.window.base_data), expected)
        assert "loading" not in session.window.windowTitle()
    finally:
        if session.window is not None:
            session.window.close()
        open_app.processEvents()
