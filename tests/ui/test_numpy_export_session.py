from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from arrayscope.core.array_source import LazySourceArray, read_index_spec
from arrayscope.core.compute_policy import ComputeLane, EvaluationContext
from arrayscope.io.numpy_export_qt import (
    ArrayExportSession,
    ExportSourceWatch,
    _ProgressEmitter,
    notify_export_source_changed,
)
from arrayscope.io.region_export import RegionExportProgress, RegionExportRequest
from arrayscope.kernel.qt_bridge import QtKernelBridge
from arrayscope.kernel.scheduler import Kernel
from arrayscope.kernel.task import Lane, Priority, TaskSpec
from arrayscope.kernel.workers import ThreadWorkerBackend
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS


class _Source:
    def __init__(self):
        self.values = np.arange(12).reshape(3, 4)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.nbytes = self.values.nbytes
        self.chunk_shape = None
        self.label = "export-only source"
        self.reads = []

    def read_region(self, spec, *, cancellation_token=None):
        self.reads.append(tuple(spec))
        return read_index_spec(self.values, spec)

    def close(self):
        pass


@pytest.fixture
def export_window(qtbot):
    window = QtWidgets.QMainWindow()
    window.kernel = Kernel(ThreadWorkerBackend(workers=1, name="export-ui-test"))
    window.kernel_bridge = QtKernelBridge(window.kernel, window)
    qtbot.addWidget(window)
    window.show()
    try:
        yield window
    finally:
        active = getattr(window, "_test_export", None)
        if active is not None:
            active.cancel()
        window.kernel_bridge.close()
        window.kernel.shutdown()
        window.close()


def _session(window, path, *, budget=512 * 1024 * 1024):
    source = _Source()
    request = RegionExportRequest(
        ArrayDocument(LazySourceArray(source)), path, ((1, 3), (1, 4)), squeeze=False
    )
    context = EvaluationContext(
        ComputeLane.STAGE, None, 1, SimpleNamespace(visible_render_budget_bytes=budget)
    )
    session = ArrayExportSession(window, request, evaluation_context=context, dtype=source.dtype)
    window._test_export = session
    return session, source


def test_declining_processing_cost_does_not_evaluate_or_create_a_file(
    export_window, qtbot, tmp_path, monkeypatch
):
    warnings = []

    def decline(*args):
        warnings.append(args[2])
        return QtWidgets.QMessageBox.StandardButton.No

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", decline)
    path = tmp_path / "declined.npy"
    session, source = _session(export_window, path, budget=1)
    session.start()
    qtbot.waitUntil(lambda: not session.is_running, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

    assert len(warnings) == 1
    assert "selected output" in warnings[0]
    assert "estimated peak memory" in warnings[0]
    assert source.reads == []
    assert not path.exists()
    assert session.request is None
    assert session.progress_dialog is None


def test_cancelled_preflight_has_no_source_work_even_when_the_kernel_is_busy(
    export_window, qtbot, tmp_path
):
    entered = threading.Event()
    release = threading.Event()

    def hold_worker(token):
        entered.set()
        while not release.wait(0.01) and not token.cancelled:
            pass

    export_window.kernel.submit(
        TaskSpec(
            "occupied-worker",
            hold_worker,
            lane=Lane.STAGE_MATERIALIZATION,
            priority=Priority.INTERACTIVE,
            pass_token=True,
        )
    )
    try:
        qtbot.waitUntil(entered.is_set, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        session, source = _session(export_window, tmp_path / "cancelled.npy")
        session.start()
        session.cancel()
        qtbot.waitUntil(lambda: not session.is_running, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        assert source.reads == []
        assert not tuple(tmp_path.iterdir())
        assert session.request is None
    finally:
        release.set()


def test_refused_kernel_admission_is_a_visible_terminal_error(export_window, tmp_path):
    export_window.kernel.shutdown()
    session, source = _session(export_window, tmp_path / "refused.npy")

    session.start()

    assert not session.is_running
    assert "could not start" in session.progress_dialog.error_label.text()
    assert session.progress_dialog.isVisible()
    assert session.request is None
    assert source.reads == []
    # An accepted job is one-shot, including terminal failures.
    assert session.start() is session


def test_write_failure_keeps_a_readable_error_without_retaining_the_source(
    export_window, qtbot, tmp_path
):
    path = tmp_path / "missing-parent" / "result.npy"
    session, _source = _session(export_window, path)
    session.start()
    qtbot.waitUntil(lambda: not session.is_running, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

    assert not path.exists()
    assert session.progress_dialog.error_label.text()
    assert session.progress_dialog.isVisible()
    assert session.progress_dialog.cancel_button.text() == "&Close"
    assert session.request is None


def test_export_progress_queues_a_bounded_number_of_observations():
    observations = []

    class Signal:
        def emit(self, generation, event):
            observations.append((generation, event))

    emitter = _ProgressEmitter(SimpleNamespace(progressed=Signal()), generation=4)
    for count in range(10_001):
        emitter(RegionExportProgress("writing", "Writing", count, 10_000))

    assert len(observations) == 101
    assert {generation for generation, _event in observations} == {4}
    assert observations[-1][1].completed_bytes == 10_000


def test_source_watch_matches_buffer_aliases_only_while_collecting_options():
    original = np.arange(12).reshape(3, 4)
    alias = original[:, :2]
    changes = []
    with ExportSourceWatch(ArrayDocument(original), changes.append) as watch:
        notify_export_source_changed(np.arange(12).reshape(3, 4))
        assert not watch.changed
        notify_export_source_changed(alias)
        assert watch.changed
        assert len(changes) == 1

    notify_export_source_changed(alias)
    assert len(changes) == 1


@pytest.mark.parametrize("changed", ["source", "definition"])
def test_nested_auxiliary_source_or_definition_change_cancels_running_export(
    export_window, qtbot, tmp_path, monkeypatch, changed
):
    from arrayscope.operations import library, plugins
    from arrayscope.operations.cancellation import EvaluationCancelled
    from arrayscope.operations.input_slots import OperationInputSlot, ResolvedSlot, SlotBinding
    from arrayscope.operations.plugins import PluginOperation, PluginOperationSpec

    class PausedSource(_Source):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def read_region(self, spec, *, cancellation_token=None):
            self.entered.set()
            while not self.release.wait(0.01):
                if cancellation_token.cancelled:
                    raise EvaluationCancelled()
            return super().read_region(spec, cancellation_token=cancellation_token)

    source = PausedSource()
    auxiliary = np.ones((3, 4))
    auxiliary_document = ArrayDocument(auxiliary)
    slot = ResolvedSlot(
        SlotBinding("open-document"),
        shape=auxiliary.shape,
        dtype=auxiliary.dtype.str,
        loader="document",
        source=auxiliary_document,
    )
    revision = [1]
    spec = PluginOperationSpec(
        id="export:nested-source-watch",
        label="Add reference",
        build=lambda _axis, _params, slots: lambda data: data + slots["reference"],
        input_slots=(OperationInputSlot("reference", "Reference"),),
        source_identity=lambda: revision[0],
    )
    monkeypatch.setitem(plugins._SPEC_CACHE, spec.id, spec)
    document = ArrayDocument(
        LazySourceArray(source),
        operations=(PluginOperation(spec.id, resolved_slots=(("reference", slot),)),),
    )
    path = tmp_path / "previous.npy"
    path.write_bytes(b"previous export")
    request = RegionExportRequest(document, path, ((1, 3), (1, 4)), squeeze=False)
    context = EvaluationContext(ComputeLane.STAGE, None, 1, None)
    session = ArrayExportSession(export_window, request, evaluation_context=context)
    export_window._test_export = session
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *_args: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    session.start()
    qtbot.waitUntil(source.entered.is_set, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

    if changed == "source":
        notify_export_source_changed(auxiliary.view())
    else:
        revision[0] += 1
        library._notify()
    qtbot.waitUntil(lambda: not session.is_running, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

    assert path.read_bytes() == b"previous export"
    assert set(tmp_path.iterdir()) == {path}
    assert session.request is None
