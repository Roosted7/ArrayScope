"""QtKernelBridge: the single GUI fan-in for kernel completions.

Pins: handlers run on the GUI thread via the queued signal path, drains are
budget-bounded, dispatch-time staleness holds across the thread boundary, and
the fallback timer stays a safety net (idle polls back off).
"""

from __future__ import annotations

import threading
import time

from pyqtgraph.Qt import QtCore

from arrayscope.kernel import Kernel, Supersession, TaskSpec, ThreadWorkerBackend
from arrayscope.kernel.qt_bridge import QtKernelBridge
from arrayscope.tools.interaction_budget import (
    INTERACTION_SETTLE_HARD_LIMIT_S,
    bounded_interaction_settle_timeout_s,
)


def _process_until(
    qt_app,
    predicate,
    timeout_s: float = INTERACTION_SETTLE_HARD_LIMIT_S,
) -> bool:
    """Pump the event loop until ``predicate()`` — no fixed sleeps."""

    timeout_s = bounded_interaction_settle_timeout_s(timeout_s)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        qt_app.processEvents()
        time.sleep(0.001)
    return predicate()


def _make(qt_app, **bridge_kwargs):
    kernel = Kernel(
        ThreadWorkerBackend(workers=4, name="test-bridge"),
        handler_error_hook=lambda ctx, exc: None,
    )
    bridge = QtKernelBridge(kernel, **bridge_kwargs)
    return kernel, bridge


def test_handlers_run_on_gui_thread(qt_app):
    kernel, bridge = _make(qt_app)
    try:
        gui_thread = threading.get_ident()
        seen = {}

        def on_done(value):
            seen["thread"] = threading.get_ident()
            seen["value"] = value

        kernel.submit(TaskSpec(key="a", fn=lambda: threading.get_ident()), on_done=on_done)
        assert _process_until(qt_app, lambda: "value" in seen)
        assert seen["thread"] == gui_thread
        assert seen["value"] != gui_thread  # fn really ran on a worker
    finally:
        bridge.close()
        kernel.shutdown()


def test_dispatch_time_staleness_across_thread_boundary(qt_app):
    kernel, bridge = _make(qt_app)
    try:
        done, stale = [], []
        finished = threading.Event()

        def fn():
            finished.set()
            return "pixels"

        kernel.submit(
            TaskSpec(key=("t", "v1"), fn=fn, supersession=Supersession("t", "v1")),
            on_done=done.append,
            on_stale=lambda: stale.append(True),
        )
        assert finished.wait(timeout=5.0)
        kernel.wait_idle(timeout=5.0)
        # The completion is queued but not yet drained; the target moves on.
        kernel.supersede("t", "v2")
        assert _process_until(qt_app, lambda: bool(stale))
        assert done == []
    finally:
        bridge.close()
        kernel.shutdown()


def test_drain_is_item_bounded_but_makes_progress(qt_app):
    kernel, bridge = _make(qt_app, max_items_per_drain=2)
    try:
        done = []
        for index in range(10):
            kernel.submit(
                TaskSpec(key=("n", index), fn=lambda index=index: index), on_done=done.append
            )
        assert _process_until(qt_app, lambda: len(done) == 10)
        assert sorted(done) == list(range(10))
    finally:
        bridge.close()
        kernel.shutdown()


def test_capacity_waiters_fire_after_processed_completions(qt_app):
    kernel, bridge = _make(qt_app)
    try:
        fired = []
        bridge.notify_when_capacity("pump", lambda: fired.append(True))
        kernel.submit(TaskSpec(key="a", fn=lambda: 1))
        assert _process_until(qt_app, lambda: bool(fired))
        assert fired == [True]  # one-shot: cleared after firing
    finally:
        bridge.close()
        kernel.shutdown()


def test_completion_drain_notifies_parent_for_event_driven_governor_updates(qt_app):
    class Parent(QtCore.QObject):
        def __init__(self):
            super().__init__()
            self.calls = []

        def _note_kernel_completion_drain(self):
            self.calls.append(True)

    parent = Parent()
    kernel, bridge = _make(qt_app, parent=parent)
    try:
        kernel.submit(TaskSpec(key="a", fn=lambda: 1))
        assert _process_until(qt_app, lambda: bool(parent.calls))
        assert parent.calls == [True]
    finally:
        bridge.close()
        kernel.shutdown()


def test_fallback_timer_backs_off_when_idle(qt_app):
    kernel, bridge = _make(qt_app)
    try:
        kernel.submit(TaskSpec(key="a", fn=lambda: 1))
        assert _process_until(qt_app, lambda: kernel.completions.empty())
        # After the queue is empty and the kernel is idle, the fallback timer
        # must stop rather than poll forever.
        assert _process_until(qt_app, lambda: not bridge._fallback_timer.isActive())
        assert bridge.fallback_event_polls == 0 or bridge.fallback_event_polls < 3
    finally:
        bridge.close()
        kernel.shutdown()
