"""Native progress objects must be released by their GUI affinity thread."""

import gc
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from shiboken6 import isValid

from arrayscope.core.array_source import LazySourceArray
from arrayscope.core.compute_policy import ComputeLane, EvaluationContext
from arrayscope.io.numpy_export_qt import ArrayExportSession, _export_worker, _ProgressRelay
from arrayscope.io.region_export import RegionExportRequest, plan_region_export
from arrayscope.kernel.qt_bridge import QtKernelBridge
from arrayscope.kernel.scheduler import Kernel
from arrayscope.kernel.task import CancellationToken, Lane, TaskSpec
from arrayscope.kernel.workers import ThreadWorkerBackend
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS


def _flush_deferred_deletion(qapp):
    qapp.processEvents()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def _observe_destruction(relay):
    threads = []
    relay.destroyed.connect(
        lambda _object=None: threads.append(threading.get_ident()),
        QtCore.Qt.ConnectionType.DirectConnection,
    )
    return threads


def _request(path, data=None):
    if data is None:
        data = np.arange(12).reshape(3, 4)
    return RegionExportRequest(ArrayDocument(data), path, ((0, 3), (0, 4)), squeeze=False)


def _worker(request, relay):
    return _export_worker(
        "evaluating",
        request,
        plan_region_export(request),
        EvaluationContext(ComputeLane.STAGE, None, 1, None),
        CancellationToken(),
        relay,
        1,
    )


def test_worker_releasing_its_last_record_never_destroys_the_native_relay(qapp, qtbot, tmp_path):
    main_thread = threading.get_ident()
    kernel = Kernel(ThreadWorkerBackend(workers=1, name="export-native-owner"))
    bridge = QtKernelBridge(kernel)
    relay = _ProgressRelay()
    destroyed_threads = _observe_destruction(relay)
    reference = weakref.ref(relay)
    request = _request(tmp_path / "values.npy")
    try:
        kernel.submit(
            TaskSpec(
                "native-progress",
                _worker(request, relay),
                lane=Lane.STAGE_MATERIALIZATION,
                pass_token=True,
            )
        )
        del relay
        assert kernel.wait_idle(timeout=INTERACTION_SETTLE_HARD_LIMIT_MS / 1000)
        # Drain the actual bridge, leaving DeferredDelete for the next turn
        # while the worker still owns its last task record.
        qapp.processEvents()
        assert kernel.completions.empty()
        assert reference() is not None

        released_record = threading.Event()
        kernel.submit(
            TaskSpec("release-last-record", released_record.set, lane=Lane.STAGE_MATERIALIZATION)
        )
        assert released_record.wait(INTERACTION_SETTLE_HARD_LIMIT_MS / 1000)
        gc.collect()

        assert all(thread == main_thread for thread in destroyed_threads), (
            "a GUI-affine QObject was natively destroyed by the worker releasing its old task"
        )
        qtbot.waitUntil(lambda: bool(destroyed_threads), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        assert destroyed_threads == [main_thread]
        np.testing.assert_array_equal(np.load(request.path), np.arange(12).reshape(3, 4))
    finally:
        bridge.close()
        kernel.shutdown()
        _flush_deferred_deletion(qapp)


def test_cancelled_pending_relay_can_be_deleted_before_retained_worker_starts(qapp, tmp_path):
    relay = _ProgressRelay()
    destroyed_threads = _observe_destruction(relay)
    request = _request(tmp_path / "never-started.npy")
    worker = _worker(request, relay)

    relay.cancel_pending()
    _flush_deferred_deletion(qapp)
    assert not isValid(relay)
    assert destroyed_threads == [threading.get_ident()]

    # A worker closure may still be retained by a canceled queue record. Even
    # if invoked late, it must examine only its Python lease, never deleted Qt.
    errors = []

    def invoke_retained_worker():
        try:
            worker(CancellationToken())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=invoke_retained_worker, name="late-export-closure")
    thread.start()
    thread.join(INTERACTION_SETTLE_HARD_LIMIT_MS / 1000)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], EvaluationCancelled)
    assert not request.path.exists()
    assert destroyed_threads == [threading.get_ident()]


@pytest.fixture
def export_host(qapp):
    window = QtWidgets.QMainWindow()
    kernel = Kernel(ThreadWorkerBackend(workers=1, name="export-native-lifecycle"))
    bridge = QtKernelBridge(kernel)
    window.kernel = kernel
    window.show()
    try:
        yield window, kernel, bridge
    finally:
        bridge.close()
        kernel.shutdown()
        if isValid(window):
            window.close()
            window.deleteLater()
        _flush_deferred_deletion(qapp)


def _session(window, request, *, budget=512 * 1024 * 1024):
    return ArrayExportSession(
        window,
        request,
        evaluation_context=EvaluationContext(
            ComputeLane.STAGE,
            None,
            1,
            SimpleNamespace(visible_render_budget_bytes=budget),
        ),
    )


def test_queued_cancel_cleans_relay_with_closed_bridge_and_unconsumed_completion(
    qapp, qtbot, export_host, tmp_path, monkeypatch
):
    window, kernel, bridge = export_host
    entered = threading.Event()
    release = threading.Event()

    def occupy_worker():
        entered.set()
        assert release.wait(INTERACTION_SETTLE_HARD_LIMIT_MS / 1000)

    def approve_after_occupying_worker(*_args):
        kernel.submit(
            TaskSpec("hold-export-worker", occupy_worker, lane=Lane.STAGE_MATERIALIZATION)
        )
        assert entered.wait(INTERACTION_SETTLE_HARD_LIMIT_MS / 1000)
        return QtWidgets.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", approve_after_occupying_worker)
    request = _request(tmp_path / "queued.npy")
    session = _session(window, request, budget=1).start()
    try:
        qtbot.waitUntil(
            lambda: session._stage == "evaluating", timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )
        relay = session._progress_relay
        assert relay is not None
        destroyed_threads = _observe_destruction(relay)
        bridge.close()
        window.close()
        window.deleteLater()
        _flush_deferred_deletion(qapp)

        assert destroyed_threads == [threading.get_ident()]
        assert not isValid(relay)
        assert relay not in qapp.children()
        assert not kernel.completions.empty()  # cleanup did not require dispatch
        assert not request.path.exists()
    finally:
        release.set()


def test_active_cancel_and_deleted_receiver_keep_relay_until_worker_finishes(
    qapp, qtbot, export_host, tmp_path
):
    window, kernel, bridge = export_host
    entered = threading.Event()
    release = threading.Event()

    class PausedSource:
        shape = (3, 4)
        dtype = np.dtype("int64")
        nbytes = 96
        chunk_shape = None
        label = "paused-native-export"

        def read_region(self, _spec, *, cancellation_token=None):
            entered.set()
            assert release.wait(INTERACTION_SETTLE_HARD_LIMIT_MS / 1000)
            if cancellation_token.cancelled:
                raise EvaluationCancelled()
            return np.arange(12).reshape(3, 4)

        def close(self):
            pass

    request = _request(tmp_path / "active.npy", LazySourceArray(PausedSource()))
    session = _session(window, request).start()
    try:
        qtbot.waitUntil(entered.is_set, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        relay = session._progress_relay
        destroyed_threads = _observe_destruction(relay)
        bridge.close()
        window.close()
        window.deleteLater()
        _flush_deferred_deletion(qapp)

        assert not isValid(window)
        assert isValid(relay)
        assert destroyed_threads == []
        release.set()
        assert kernel.wait_idle(timeout=INTERACTION_SETTLE_HARD_LIMIT_MS / 1000)
        _flush_deferred_deletion(qapp)

        assert destroyed_threads == [threading.get_ident()]
        assert not isValid(relay)
        assert relay not in qapp.children()
        assert not kernel.completions.empty()
        assert not request.path.exists()
    finally:
        release.set()


def test_evaluation_admission_failure_releases_application_child(
    qapp, qtbot, export_host, tmp_path, monkeypatch
):
    window, kernel, _bridge = export_host
    existing_relays = {id(child) for child in qapp.children() if isinstance(child, _ProgressRelay)}

    def approve_after_shutdown(*_args):
        kernel.shutdown()
        return QtWidgets.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", approve_after_shutdown)
    request = _request(tmp_path / "refused.npy")
    session = _session(window, request, budget=1).start()
    qtbot.waitUntil(lambda: not session.is_running, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
    assert "could not start" in session.progress_dialog.error_label.text()
    _flush_deferred_deletion(qapp)
    assert {
        id(child) for child in qapp.children() if isinstance(child, _ProgressRelay)
    } == existing_relays
    assert not request.path.exists()


def test_cancel_during_signal_emission_cannot_delete_the_active_relay(qapp, tmp_path):
    entered = threading.Event()
    release = threading.Event()
    relay = _ProgressRelay()
    destroyed_threads = _observe_destruction(relay)
    token = CancellationToken()
    request = _request(tmp_path / "emitting.npy")
    kernel = Kernel(ThreadWorkerBackend(workers=1, name="export-active-emission"))

    def hold_signal_emission(_generation, _event):
        entered.set()
        assert release.wait(INTERACTION_SETTLE_HARD_LIMIT_MS / 1000)

    relay.progressed.connect(hold_signal_emission, QtCore.Qt.ConnectionType.DirectConnection)
    worker = _export_worker(
        "evaluating",
        request,
        plan_region_export(request),
        EvaluationContext(ComputeLane.STAGE, None, 1, None),
        token,
        relay,
        1,
    )
    try:
        kernel.submit(
            TaskSpec("emitting", worker, lane=Lane.STAGE_MATERIALIZATION, pass_token=True)
        )
        assert entered.wait(INTERACTION_SETTLE_HARD_LIMIT_MS / 1000)
        token.cancel()
        relay.cancel_pending()
        _flush_deferred_deletion(qapp)
        assert isValid(relay)
        assert destroyed_threads == []

        release.set()
        assert kernel.wait_idle(timeout=INTERACTION_SETTLE_HARD_LIMIT_MS / 1000)
        _flush_deferred_deletion(qapp)
        assert destroyed_threads == [threading.get_ident()]
        assert not isValid(relay)
        assert not request.path.exists()
    finally:
        release.set()
        kernel.shutdown()
        _flush_deferred_deletion(qapp)


def test_relay_cannot_silently_lose_its_application_parent_on_a_worker(qapp):
    errors = []

    def create_on_wrong_thread():
        try:
            _ProgressRelay()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=create_on_wrong_thread, name="invalid-relay-owner")
    thread.start()
    thread.join(INTERACTION_SETTLE_HARD_LIMIT_MS / 1000)
    assert not thread.is_alive()
    assert errors == ["Export progress must be created on the Qt application thread."]
