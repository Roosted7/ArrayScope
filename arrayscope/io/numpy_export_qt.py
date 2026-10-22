"""GUI ownership for an exact export running on the window's execution kernel."""

from __future__ import annotations

import weakref
from dataclasses import replace
from threading import Lock
from uuid import uuid4

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.core.array_source import LazySourceArray
from arrayscope.core.memory_budget import format_bytes
from arrayscope.io.region_export import (
    RegionExportChanged,
    RegionExportProgress,
    capture_export_sources,
    export_region,
    plan_region_export,
    validate_region_export_snapshot,
)
from arrayscope.kernel.task import CancellationToken, Lane, Priority, TaskSpec
from arrayscope.operations import library
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.ui.export_progress import ExportProgressDialog
from arrayscope.ui.toasts import show_status_message

_SOURCE_WATCHES = weakref.WeakSet()


def _source_aliases(source):
    """Follow established buffer owners without reading or comparing pixels."""
    aliases = []
    while source is not None and not any(source is item for item in aliases):
        aliases.append(source)
        if isinstance(source, LazySourceArray):
            source = source.source
        elif isinstance(source, np.ndarray):
            source = source.base
        elif isinstance(source, memoryview):
            source = source.obj
        else:
            break
    return aliases


class ExportSourceWatch:
    """Observe notified changes only while choosing or running one export."""

    def __init__(self, document, on_change=None):
        self._sources = tuple(
            alias
            for source in capture_export_sources(document)
            for alias in _source_aliases(source)
        )
        self._on_change = on_change
        self.changed = False
        self.reason = "An array used by this export changed. Retry with the updated data."

    def __enter__(self):
        _SOURCE_WATCHES.add(self)
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        _SOURCE_WATCHES.discard(self)
        self._sources = ()
        self._on_change = None

    def notify(self, aliases):
        if any(source is alias for source in self._sources for alias in aliases):
            self.changed = True
            if self._on_change is not None:
                self._on_change(self.reason)


def notify_export_source_changed(source):
    """Cancel exports sharing an explicitly mutated primary or auxiliary source."""
    aliases = _source_aliases(source)
    for watch in tuple(_SOURCE_WATCHES):
        watch.notify(aliases)


class _ExportToken:
    """One user cancellation request spans the two kernel-owned task phases."""

    def __init__(self, user_token, task_token):
        self._user_token = user_token
        self._task_token = task_token

    @property
    def cancelled(self):
        return self._user_token.cancelled or self._task_token.cancelled


class _ProgressRelay(QtCore.QObject):
    """Keep GUI ownership until a pending or active worker releases its lease.

    A completed task can remain the worker's last local reference after the
    GUI has dispatched its completion. The application parent prevents that
    worker from destroying this GUI-affine QObject when it takes another task.
    The lease also cleans unstarted tasks when a closing bridge cannot deliver
    their completion, without deleting a relay still emitting from a worker.
    """

    progressed = QtCore.Signal(int, object)

    def __init__(self):
        application = QtCore.QCoreApplication.instance()
        if application is None:
            raise RuntimeError("Export progress requires a running Qt application.")
        if QtCore.QThread.currentThread() != application.thread():
            raise RuntimeError("Export progress must be created on the Qt application thread.")
        super().__init__(application)
        self._lease_lock = Lock()
        self._lease_state = "pending"

    def begin_work(self):
        with self._lease_lock:
            if self._lease_state != "pending":
                return False
            self._lease_state = "running"
            return True

    def cancel_pending(self):
        with self._lease_lock:
            if self._lease_state != "pending":
                return
            self._lease_state = "released"
        # QObject.deleteLater is thread-safe. The application retains native
        # ownership until the GUI processes this deferred deletion.
        self.deleteLater()

    def finish_work(self):
        with self._lease_lock:
            if self._lease_state != "running":
                return
            self._lease_state = "released"
        # All signal emissions have returned before relinquishing the lease.
        # No worker touches this QObject again after requesting deletion.
        self.deleteLater()


class _ProgressEmitter:
    """Bound queued UI observations to phases and whole percentage changes."""

    def __init__(self, relay, generation):
        self._relay = relay
        self._generation = generation
        self._last = None

    def __call__(self, event):
        bucket = (
            min(100, max(0, event.completed_bytes * 100 // event.total_bytes))
            if event.total_bytes
            else event.message
        )
        key = (event.phase, bucket)
        if key != self._last:
            self._last = key
            self._relay.progressed.emit(self._generation, event)


def _export_worker(stage, request, prepared, context, user_token, relay, generation):
    """Create a worker closure containing no window or session references."""
    emit_progress = None if relay is None else _ProgressEmitter(relay, generation)

    def run(task_token):
        if relay is not None and not relay.begin_work():
            raise EvaluationCancelled()
        try:
            token = _ExportToken(user_token, task_token)
            evaluation_context = replace(context, cancellation_token=token)
            if stage == "planning":
                return plan_region_export(
                    request, cancellation_token=token, evaluation_context=evaluation_context
                )
            return export_region(
                request,
                prepared=prepared,
                cancellation_token=token,
                evaluation_context=evaluation_context,
                progress_callback=emit_progress,
            )
        finally:
            if relay is not None:
                relay.finish_work()

    return run


class ArrayExportSession(QtCore.QObject):
    """One accepted selection, independent of render and viewport generations."""

    finished = QtCore.Signal(object)

    def __init__(self, window, request, *, evaluation_context, dtype=None):
        super().__init__(window)
        self.request = request
        self.progress_dialog = ExportProgressDialog(
            window, request.path, request.output_shape, dtype
        )
        self.progress_dialog.cancel_requested.connect(self.cancel)
        self.progress_dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self._window_ref = weakref.ref(window)
        self._kernel = window.kernel
        self._context = evaluation_context
        self._token = CancellationToken()
        self._task_handle = None
        self._progress_relay = None
        self._prepared = None
        self._generation = 0
        self._scope = f"numpy-export:{uuid4().hex}"
        self._stage = "starting"
        self._running = False
        self._started = False
        self._closed = False
        self._cancel_reason = ""
        self._listening = False
        self._source_watch = ExportSourceWatch(request.document, self.cancel)
        window.installEventFilter(self)

    @property
    def is_running(self):
        return self._running

    def start(self):
        if self._closed or self._started:
            return self
        self._started = True
        self._running = True
        self._source_watch.__enter__()
        library.add_library_listener(self._library_changed)
        self._listening = True
        self.progress_dialog.show()
        self._submit_stage("planning")
        return self

    def show_progress(self):
        if self._running and not self._closed and self.progress_dialog is not None:
            self.progress_dialog.show()
            self.progress_dialog.raise_()
            self.progress_dialog.activateWindow()

    def cancel(self, reason=""):
        if not self._running:
            return
        self._cancel_reason = reason or "Export cancelled."
        self._token.cancel()
        if self._progress_relay is not None:
            self._progress_relay.cancel_pending()
        if self._task_handle is not None:
            self._task_handle.cancel()
        if not self._closed and self.progress_dialog is not None:
            self.progress_dialog.set_cancelling(reason)
        if self._stage == "confirmation":
            self._finish()

    def eventFilter(self, obj, event):
        if obj is self._window_ref() and event.type() == QtCore.QEvent.Type.Close:
            self._closed = True
            self.cancel("The viewer was closed.")
            self._finish()
        return super().eventFilter(obj, event)

    def _submit_stage(self, stage):
        if not self._running or self._closed:
            return
        if self._token.cancelled:
            self._finish()
            return
        self._stage = stage
        self._generation += 1
        generation = self._generation
        relay = None
        if stage == "evaluating":
            relay = _ProgressRelay()
            self._progress_relay = relay
            relay.progressed.connect(self._on_progress, QtCore.Qt.ConnectionType.QueuedConnection)
        worker = _export_worker(
            stage,
            self.request,
            self._prepared,
            self._context,
            self._token,
            relay,
            generation,
        )
        reference = weakref.ref(self)

        def done(value):
            session = reference()
            if session is not None:
                session._done(generation, value)

        def failed(error):
            session = reference()
            if session is not None:
                session._failed(generation, error)

        def stale():
            session = reference()
            if session is not None:
                session._stale(generation)

        self.progress_dialog.apply_progress(
            RegionExportProgress(
                stage,
                "Checking the selected export…"
                if stage == "planning"
                else "Evaluating selected scientific values…",
            )
        )
        self._task_handle = self._kernel.submit(
            TaskSpec(
                key=(self._scope, generation),
                fn=worker,
                lane=Lane.STAGE_MATERIALIZATION,
                priority=Priority.INTERACTIVE,
                scope=self._scope,
                pass_token=True,
            ),
            on_done=done,
            on_error=failed,
            on_stale=stale,
        )
        if self._task_handle is None:
            if relay is not None:
                relay.cancel_pending()
            self._finish(error="The viewer is shutting down; export could not start.")

    def _done(self, generation, value):
        if not self._running or self._closed or generation != self._generation:
            return
        self._task_handle = None
        if self._progress_relay is not None:
            self._progress_relay.cancel_pending()
            self._progress_relay = None
        if self._stage == "planning":
            self._prepared = value
            if self._token.cancelled:
                self._finish()
                return
            self.progress_dialog.set_output_summary(value.output_shape, value.output_dtype)
            if value.needs_confirmation:
                self._stage = "confirmation"
                output = (
                    "unknown" if value.output_bytes is None else format_bytes(value.output_bytes)
                )
                source = (
                    "unknown"
                    if value.source_read_bytes is None
                    else format_bytes(value.source_read_bytes)
                )
                peak = (
                    "unknown"
                    if value.estimated_peak_bytes is None
                    else format_bytes(value.estimated_peak_bytes)
                )
                message = (
                    f"The selected output is {value.output_shape}, {value.output_dtype} ({output}).\n"
                    f"Processing requires {source} of source data; estimated peak memory: {peak}."
                )
                if value.warnings:
                    message += "\n\n" + "\n".join(value.warnings)
                message += "\n\nContinue with this export?"
                answer = QtWidgets.QMessageBox.warning(
                    self.progress_dialog,
                    "Confirm expensive export",
                    message,
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if not self._running or self._closed:
                    return
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    self.cancel("Export cancelled before evaluating data.")
                    return
            self._submit_stage("evaluating")
        else:
            # The writer's atomic publication won any late cancellation race.
            # A successfully returned result is a completed export.
            self._finish(result=value)

    @QtCore.Slot(int, object)
    def _on_progress(self, generation, event):
        if self._running and not self._closed and generation == self._generation:
            self.progress_dialog.apply_progress(event)

    def _failed(self, generation, error):
        if self._running and not self._closed and generation == self._generation:
            self._task_handle = None
            self._finish(error=str(error))

    def _stale(self, generation):
        if self._running and not self._closed and generation == self._generation:
            self._task_handle = None
            self._finish()

    def _library_changed(self):
        if not self._running:
            return
        try:
            validate_region_export_snapshot(self.request)
        except (RegionExportChanged, OSError) as error:
            self.cancel(str(error))

    def _finish(self, *, result=None, error=None):
        if not self._running:
            return
        self._running = False
        self._generation += 1
        self._source_watch.close()
        if self._listening:
            library.remove_library_listener(self._library_changed)
            self._listening = False
        window = self._window_ref()
        if window is not None and not self._closed:
            window.removeEventFilter(self)
        dialog = self.progress_dialog
        if error is not None and not self._closed:
            dialog.set_error(error)
        else:
            dialog.close_quietly()
            self.progress_dialog = None
        if window is not None and not self._closed:
            if result is not None:
                show_status_message(
                    window, f"Saved array {list(result.shape)} to {result.paths[0]}"
                )
            elif error is None:
                show_status_message(window, self._cancel_reason or "Export cancelled.")
        self._task_handle = None
        self.request = None
        if self._progress_relay is not None:
            self._progress_relay.cancel_pending()
            self._progress_relay = None
        self._prepared = None
        self._context = None
        self._kernel = None
        self.finished.emit(result)
        self.deleteLater()
