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

from tests.settle import settle_until


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
        app._arrayscope_live_windows = []
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


@pytest.mark.parametrize("choice", ["dark", "light"])
def test_launcher_restores_saved_theme_before_any_array_is_open(open_app, choice):
    import pyqtgraph as pg

    from arrayscope.app.open_flow import show_launcher_window
    from arrayscope.app.theme import DARK_TOKENS, LIGHT_TOKENS

    settings = QtCore.QSettings()
    previous_choice = settings.value("theme")
    previous_sheet = open_app.styleSheet()
    previous_palette = open_app.palette()
    previous_tokens = open_app.property("_arrayscope_theme_tokens")
    previous_plot_colors = {name: pg.getConfigOption(name) for name in ("background", "foreground")}
    try:
        settings.setValue("theme", choice)
        open_app.setStyleSheet("")
        launcher = show_launcher_window()
        expected = DARK_TOKENS if choice == "dark" else LIGHT_TOKENS
        assert launcher.isVisible()
        assert launcher.palette().window().color().name() == expected.window
        assert expected.window in open_app.styleSheet()
    finally:
        if previous_choice is None:
            settings.remove("theme")
        else:
            settings.setValue("theme", previous_choice)
        open_app.setPalette(previous_palette)
        open_app.setStyleSheet(previous_sheet)
        open_app.setProperty("_arrayscope_theme_tokens", previous_tokens)
        pg.setConfigOptions(**previous_plot_colors)


def _session_progress(session):
    """What the load has published about itself, as one comparable snapshot.

    The reader thread reports a fraction per slice and the session mirrors it
    into ``_latest_fraction``; the window and the status widget appear on their
    own edges.  Any of these moving means the load is running, however slowly.
    """

    return (
        float(getattr(session, "_latest_fraction", 0.0) or 0.0),
        session.window is not None,
        session.loading_window is not None,
        session._status_widget is not None,
        bool(getattr(session, "_terminal", False)),
    )


def _pump_until(app, predicate, session, label):
    del app  # the shared pump owns the event loop
    settle_until(
        predicate,
        progress=lambda: _session_progress(session),
        label=label,
    )


def _run_session(app, session):
    terminal = {"done": False}
    session.done.connect(lambda: terminal.__setitem__("done", True))
    # ``done`` may already have fired: a caller that pumped the loop before
    # handing the session over (the streaming test opens the viewer first) can
    # have consumed the whole load, and connecting afterwards then waits for a
    # signal that will never come again.  The old busy ``processEvents`` spin
    # hid that by holding the GIL and starving the reader thread it was waiting
    # for -- i.e. the test depended on its own wait being slow. ``_terminal`` is
    # the state the signal announces, so ask about the state.
    _pump_until(
        app,
        lambda: terminal["done"] or bool(session._terminal),
        session,
        "load reaches a terminal state",
    )
    return session


def test_open_path_async_opens_viewer_with_loaded_data(open_app, tmp_path):
    from arrayscope.app.open_flow import open_path_async
    from arrayscope.app.recent_files import recent_paths

    arr = np.random.default_rng(0).normal(size=(6, 8, 4)).astype(np.float32)
    path = tmp_path / "vol.npy"
    np.save(path, arr)

    session = _run_session(open_app, open_path_async(path))
    try:
        assert session.window is not None
        assert session.loading_window is None
        np.testing.assert_array_equal(np.asarray(session.window.base_data), arr)
        assert session.window.windowTitle() == "vol.npy [numpy]"
        assert recent_paths(QtCore.QSettings())[0] == path.resolve()
    finally:
        if session.window is not None:
            session.window.close()
        open_app.processEvents()


def test_open_path_async_consume_removes_handoff_file(open_app, tmp_path):
    from arrayscope.app.open_flow import open_path_async
    from arrayscope.app.recent_files import recent_paths

    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = tmp_path / "handoff.npy"
    np.save(path, arr)
    previous_recent = recent_paths(QtCore.QSettings())

    session = _run_session(open_app, open_path_async(path, mmap=True, consume=True))
    try:
        assert session.window is not None
        assert not path.exists()
        np.testing.assert_array_equal(np.asarray(session.window.base_data), arr)
        assert recent_paths(QtCore.QSettings()) == previous_recent
    finally:
        if session.window is not None:
            session.window.close()
        open_app.processEvents()


def test_open_path_async_failure_shows_error_state(open_app, tmp_path):
    from arrayscope.app.open_flow import open_path_async
    from arrayscope.app.recent_files import recent_paths

    path = tmp_path / "broken.npy"
    path.write_bytes(b"not a numpy file at all")
    previous_recent = recent_paths(QtCore.QSettings())

    session = _run_session(open_app, open_path_async(path))
    try:
        assert session.window is None
        assert session.loading_window is not None
        assert session.loading_window._cancel_btn.text() == "Close"
        assert recent_paths(QtCore.QSettings()) == previous_recent
    finally:
        if session.loading_window is not None:
            session.loading_window.close_quietly()
        open_app.processEvents()


def test_open_path_async_cancel_before_read_closes_quietly(open_app, tmp_path):
    from arrayscope.app.open_flow import FileOpenSession
    from arrayscope.app.recent_files import recent_paths

    arr = np.zeros((64, 64), dtype=np.float32)
    path = tmp_path / "vol.npy"
    np.save(path, arr)
    previous_recent = recent_paths(QtCore.QSettings())

    session = FileOpenSession(path)
    session.cancel()
    session.start()
    _run_session(open_app, session)

    assert session.window is None
    assert session.loading_window is None
    assert recent_paths(QtCore.QSettings()) == previous_recent


@pytest.mark.parametrize("sample_id", ["time-series", "complex"])
def test_examples_open_with_committed_values_and_cleanly_close(open_app, sample_id):
    from shiboken6 import isValid

    from arrayscope.app.open_flow import open_sample
    from arrayscope.app.recent_files import recent_paths
    from arrayscope.app.sample_data import SAMPLES, make_sample
    from tests.settle import settle_window
    from tests.ui.helpers import committed_value, frame_session_settled

    expected, axes = make_sample(sample_id)
    sample = next(item for item in SAMPLES if item.id == sample_id)
    previous_recent = recent_paths(QtCore.QSettings())
    win = open_sample(sample_id)
    workers = tuple(win.kernel._backend._threads)
    try:
        settle_window(
            win,
            lambda: (
                getattr(win, "_committed_display_frame", None) is not None
                and frame_session_settled(win)
            ),
            label=f"{sample_id} example commits its first frame",
        )
        assert win.isVisible()
        assert win.windowTitle() == f"ArrayScope example — {sample.title}"
        assert win.document.current_axes == axes
        np.testing.assert_array_equal(win.document.base_data, expected)
        hint = win._arrayscope_status_message_widget
        assert hint.isVisible()
        assert hint.property("arrayscope_status_message_text") == sample.hint

        if sample_id == "time-series":
            win.widgets["spins"]["slice_indices"][2].setValue(11)
            settle_window(
                win,
                lambda: (
                    win._committed_display_frame.geometry.view_state.slice_indices[2] == 11
                    and frame_session_settled(win)
                ),
                label="example time selection commits",
            )
            expected_value = expected[53, 37, 11]
        else:
            channel = win.display_toolbar.channel_combo
            channel.setCurrentIndex(channel.findData("angle"))
            settle_window(
                win,
                lambda: (
                    win._committed_display_frame.geometry.view_state.channel.value == "angle"
                    and frame_session_settled(win)
                ),
                label="example phase channel commits",
            )
            expected_value = np.angle(expected[53, 37])
        assert committed_value(win, 37, 53) == pytest.approx(expected_value)
        assert recent_paths(QtCore.QSettings()) == previous_recent
    finally:
        win.close()
        open_app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert not isValid(win)
    assert win not in getattr(open_app, "_arrayscope_live_windows", [])
    assert not any(worker.is_alive() for worker in workers)


def test_open_any_path_rejects_unknown_suffix(open_app, tmp_path, capsys):
    from arrayscope.app.open_flow import open_any_path

    path = tmp_path / "data.xyz"
    path.write_bytes(b"?")

    assert not open_any_path(path)
    assert "Unsupported file type" in capsys.readouterr().err


@pytest.mark.parametrize("suffix", ["h5", "npz"])
def test_single_dataset_read_failure_closes_its_container(open_app, monkeypatch, tmp_path, suffix):
    import h5py

    from arrayscope.app.open_flow import open_any_path
    from arrayscope.io.selectors import H5DatasetSelector, NpzDatasetSelector

    path = tmp_path / f"single.{suffix}"
    if suffix == "h5":
        with h5py.File(path, "w") as container:
            container["image"] = np.arange(12).reshape(3, 4)
        selector_type = H5DatasetSelector
    else:
        np.savez(path, image=np.arange(12).reshape(3, 4))
        selector_type = NpzDatasetSelector
    selectors = []

    def fail_read(selector, _name):
        # Hold the selector alive: destructor cleanup would mask a leak.
        selectors.append(selector)
        raise OSError("Dataset read failed")

    monkeypatch.setattr(selector_type, "load_data", fail_read)
    try:
        with pytest.raises(OSError, match="Dataset read failed"):
            open_any_path(path)
        assert len(selectors) == 1
        if suffix == "h5":
            assert not selectors[0].h5_file.id.valid
        else:
            assert selectors[0].npz_file.zip is None
    finally:
        for selector in selectors:
            selector.close()


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
    _pump_until(
        open_app,
        lambda: session.window is not None and reader_parked.is_set(),
        session,
        "viewer opens and the reader parks",
    )
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
    _pump_until(open_app, lambda: session._terminal, session, "load reaches a terminal state")
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
        _pump_until(
            open_app,
            lambda: session.window is not None and reader_parked.is_set(),
            session,
            "viewer opens and the reader parks",
        )
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

    # Record the widget's creation instead of sampling for its presence later.
    # A sample cannot express "it appeared and then went away": one pump slice
    # can carry the probe, all four progress events and the terminal handler,
    # and the sample then reads the post-load state and calls it the absence of
    # a streaming phase.  This test used to pass only because the wait it ran
    # first was slow enough to hold the reader thread back.
    from arrayscope.ui import load_status

    status_widgets = []
    original_status_widget = load_status.LoadStatusWidget

    class RecordingStatusWidget(original_status_widget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            status_widgets.append(self)

    monkeypatch.setattr(load_status, "LoadStatusWidget", RecordingStatusWidget)

    # Patched on the class, before the session exists.  Bound to the instance
    # after the viewer opens -- which is where this lived -- it misses every
    # streaming refresh the load already delivered, and a load that finishes
    # inside one pump slice delivers all of them before the recorder is on.
    from arrayscope.window import ArrayScopeWindow

    autolevel_refreshes = []
    original_notify = ArrayScopeWindow.notify_data_changed

    def record_notify(self, *, force_autolevel=False):
        autolevel_refreshes.append(bool(force_autolevel))
        return original_notify(self, force_autolevel=force_autolevel)

    monkeypatch.setattr(ArrayScopeWindow, "notify_data_changed", record_notify)

    session = open_path_async(path)
    _pump_until(open_app, lambda: session.window is not None, session, "streaming viewer opens")
    _run_session(open_app, session)
    try:
        assert status_widgets, "the streaming probe opened the viewer without a status widget"
        assert session._status_widget is None
        assert autolevel_refreshes
        assert all(autolevel_refreshes)
        np.testing.assert_array_equal(np.asarray(session.window.base_data), expected)
        assert "loading" not in session.window.windowTitle()
    finally:
        if session.window is not None:
            session.window.close()
        open_app.processEvents()
