"""Selection-first export reads exact data on workers and owns its cancellation."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.core.array_source import LazySourceArray, read_index_spec
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.settle import settle_window
from tests.ui.helpers import frame_session_settled


def _is_selection_read(spec):
    return all(
        isinstance(item, slice)
        and item.start is not None
        and item.stop is not None
        and start <= item.start <= item.stop <= stop
        for item, (start, stop) in zip(spec, ((3, 9), (7, 14), (2, 5)), strict=True)
    )


class _CountedSource:
    def __init__(self):
        self.values = np.arange(32 * 40 * 9, dtype=np.float64).reshape(32, 40, 9)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.nbytes = self.values.nbytes
        self.chunk_shape = None
        self.label = "counted-export-source"
        self.reads = []
        self.pause_export = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def read_region(self, index_spec, *, cancellation_token=None):
        spec = tuple(index_spec)
        self.reads.append((spec, threading.get_ident()))
        # Display reads may also keep a length-one slice on the hidden axis.
        # Block only this test's selected region, allowing fresh renders after
        # notify_data_changed() to drain normally.
        if self.pause_export and _is_selection_read(spec):
            self.entered.set()
            while not self.release.wait(0.01):
                if cancellation_token is not None and cancellation_token.cancelled:
                    raise EvaluationCancelled()
        return np.array(read_index_spec(self.values, spec), copy=True)

    def close(self):
        pass


@pytest.fixture
def lazy_window(qtbot):
    from shiboken6 import isValid

    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", "pyqtgraph")
    source = _CountedSource()
    win = ArrayScopeWindow(LazySourceArray(source))
    win.show()
    settle_window(win, lambda: frame_session_settled(win), label="lazy export fixture opens")
    source.reads.clear()
    try:
        yield win, source
    finally:
        source.release.set()
        if isValid(win):
            win.close()
            win.deleteLater()
        QtWidgets.QApplication.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def _choose_export(monkeypatch, path, *, cancel_at=None):
    from arrayscope.io import numpy_save_qt

    class RangeDialog:
        def __init__(self, _parent, shape, **_kwargs):
            assert tuple(shape) == (32, 40, 9)

        def exec(self):
            return (
                QtWidgets.QDialog.DialogCode.Rejected
                if cancel_at == "range"
                else QtWidgets.QDialog.DialogCode.Accepted
            )

        def get_ranges(self):
            return ((3, 9), (7, 14), (2, 5))

        def should_squeeze(self):
            return False

    monkeypatch.setattr(numpy_save_qt, "SaveRangeDialog", RangeDialog)
    monkeypatch.setattr(
        numpy_save_qt,
        "get_save_file_name",
        lambda *_args: ("" if cancel_at == "path" else str(path), ""),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *_args: pytest.fail("export unexpectedly opened an error dialog"),
    )


@pytest.mark.parametrize("cancel_at", ["range", "path"])
def test_cancelled_export_dialogs_never_read_the_lazy_source(
    lazy_window, tmp_path, monkeypatch, cancel_at
):
    win, source = lazy_window
    path = tmp_path / "selection.npy"
    win._append_operation("reverse", 0)
    settle_window(win, lambda: frame_session_settled(win), label="export pipeline edit settles")
    source.reads.clear()
    _choose_export(monkeypatch, path, cancel_at=cancel_at)

    assert win._save_current_numpy_file() is None

    assert source.reads == []
    assert not path.exists()


def test_progressively_loading_source_refuses_export_before_dialogs_or_reads(
    lazy_window, tmp_path, monkeypatch
):
    from arrayscope.app import open_flow
    from arrayscope.io import numpy_save_qt

    win, source = lazy_window
    opening = open_flow.FileOpenSession(tmp_path / "loading.npy", parent=win)
    opening.window = win
    monkeypatch.setattr(open_flow, "_ACTIVE_SESSIONS", [opening])
    monkeypatch.setattr(
        numpy_save_qt,
        "SaveRangeDialog",
        lambda *_args, **_kwargs: pytest.fail("a changing source must be refused before options"),
    )
    messages = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: messages.append(args))

    assert win._save_current_numpy_file() is None

    assert messages
    assert source.reads == []
    assert getattr(win, "_numpy_export_session", None) is None
    assert win.kernel.diagnostics().running == 0


@pytest.mark.parametrize("change_kind", ["operation", "new-source"])
def test_selected_export_runs_off_gui_and_keeps_its_document_snapshot(
    lazy_window, qtbot, tmp_path, monkeypatch, change_kind
):
    win, source = lazy_window
    path = tmp_path / "selection.npy"
    _choose_export(monkeypatch, path)
    source.pause_export = True

    session = win._save_current_numpy_file()
    qtbot.waitUntil(source.entered.is_set, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
    assert session.is_running
    # Reopening Save while a job is running reveals the same owned job,
    # rather than submitting a competing export of the source.
    assert win._save_current_numpy_file() is session
    assert session.progress_dialog.isVisible()
    assert not path.exists()
    assert source.reads
    assert all(_is_selection_read(spec) for spec, _thread_id in source.reads)
    heartbeat = []
    QtCore.QTimer.singleShot(0, win, lambda: heartbeat.append(True))
    qtbot.waitUntil(lambda: bool(heartbeat), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

    # Editing or replacing the visible document must not rewrite the already
    # accepted job, whose old source stays alive through its captured document.
    if change_kind == "operation":
        win._append_operation("reverse", 0)
    else:
        win._reset_data(source.values + 100_000)
    source.release.set()
    qtbot.waitUntil(lambda: not session.is_running, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

    np.testing.assert_array_equal(np.load(path), source.values[3:9, 7:14, 2:5])
    export_reads = [
        (spec, thread_id) for spec, thread_id in source.reads if _is_selection_read(spec)
    ]
    assert export_reads
    coverage = np.zeros((6, 7, 3), dtype=bool)
    for spec, thread_id in export_reads:
        assert thread_id != threading.get_ident()
        local = []
        for item, (start, stop) in zip(spec, ((3, 9), (7, 14), (2, 5)), strict=True):
            assert start <= item.start <= item.stop <= stop
            assert item.step in (1, None)
            local.append(slice(item.start - start, item.stop - start))
        coverage[tuple(local)] = True
    assert coverage.all()
    assert session.progress_dialog is None or not session.progress_dialog.isVisible()


@pytest.mark.parametrize("cancel_with", ["user", "source-notification"])
def test_cancelling_export_preserves_previous_file_and_clears_progress(
    lazy_window, qtbot, tmp_path, monkeypatch, cancel_with
):
    win, source = lazy_window
    path = tmp_path / "selection.npy"
    previous = b"previous complete export"
    path.write_bytes(previous)
    _choose_export(monkeypatch, path)
    source.pause_export = True

    session = win._save_current_numpy_file()
    qtbot.waitUntil(source.entered.is_set, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
    if cancel_with == "user":
        session.cancel()
    else:
        win.notify_data_changed()
    qtbot.waitUntil(
        lambda: not session.is_running and win.kernel.diagnostics().running == 0,
        timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
    )

    assert path.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {path}
    assert session.progress_dialog is None or not session.progress_dialog.isVisible()


def test_closing_window_cancels_export_without_publishing_or_late_dialogs(
    lazy_window, qtbot, tmp_path, monkeypatch
):
    from shiboken6 import isValid

    win, source = lazy_window
    win.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
    path = tmp_path / "selection.npy"
    previous = b"previous complete export"
    path.write_bytes(previous)
    _choose_export(monkeypatch, path)
    source.pause_export = True
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args))

    session = win._save_current_numpy_file()
    qtbot.waitUntil(source.entered.is_set, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
    win.close()
    QtWidgets.QApplication.processEvents()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    assert not isValid(win)
    assert not session.is_running
    assert win.kernel.diagnostics().running == 0
    assert path.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {path}
    assert warnings == []


def test_another_window_notifying_a_shared_source_cancels_the_export(
    lazy_window, qtbot, tmp_path, monkeypatch
):
    from arrayscope.window import ArrayScopeWindow

    win, source = lazy_window
    other = ArrayScopeWindow(win.base_data)
    other.show()
    try:
        settle_window(other, lambda: frame_session_settled(other), label="shared source opens")
        path = tmp_path / "shared-selection.npy"
        path.write_bytes(b"previous complete export")
        _choose_export(monkeypatch, path)
        source.pause_export = True
        session = win._save_current_numpy_file()
        qtbot.waitUntil(source.entered.is_set, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

        other.notify_data_changed()
        qtbot.waitUntil(lambda: not session.is_running, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

        assert path.read_bytes() == b"previous complete export"
        assert set(tmp_path.iterdir()) == {path}
        assert session.progress_dialog is None
    finally:
        source.release.set()
        other.close()
        other.deleteLater()


def test_source_notification_while_choosing_options_requires_a_fresh_selection(
    lazy_window, tmp_path, monkeypatch
):
    from arrayscope.io import numpy_save_qt
    from arrayscope.io.numpy_export_qt import notify_export_source_changed

    win, source = lazy_window
    path = tmp_path / "selection.npy"
    _choose_export(monkeypatch, path)

    def changed_before_accepting(*_args):
        notify_export_source_changed(win.base_data)
        return str(path), ""

    monkeypatch.setattr(numpy_save_qt, "get_save_file_name", changed_before_accepting)
    messages = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: messages.append(args[2]))

    assert win._save_current_numpy_file() is None

    assert source.reads == []
    assert not path.exists()
    assert messages
    assert "changed" in messages[0]
    assert getattr(win, "_numpy_export_session", None) is None
