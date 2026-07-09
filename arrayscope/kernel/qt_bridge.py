"""Qt adapter: the ONE place kernel completions enter the GUI thread.

Everything Qt about the kernel lives here. The bridge:

- receives `CompletionQueue` notifications from worker threads and converts
  them into a queued signal (never direct calls);
- drains events on the GUI thread under a `GuiCallbackBudget` (item + time
  bounded, observation reported to the resource governor when present);
- keeps a single adaptive fallback timer as an anti-hang safety net for
  missed cross-thread signals — the timer is NOT a scheduling mechanism, and
  `fallback_event_polls` observability is preserved so a busy fallback is a
  visible bug report;
- fires capacity waiters after any processed completion (the ADR 0051
  lost-wakeup rule: a declined admission must always leave a wakeup armed).

The app composes exactly one bridge per kernel; per-purpose controllers are
gone (redesign plan R1).
"""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.gui_callback_budget import GuiCallbackBudget
from arrayscope.kernel.scheduler import Kernel
from arrayscope.kernel.task import TaskOutcome

prefer_pyside6()

import pyqtgraph.Qt as Qt


class QtKernelBridge(Qt.QtCore.QObject):
    eventsReady = Qt.QtCore.Signal()

    def __init__(
        self,
        kernel: Kernel,
        parent=None,
        *,
        max_items_per_drain: int = 8,
        max_events_per_drain: int = 64,
        budget_ms: float | None = None,
    ) -> None:
        super().__init__(parent)
        self.kernel = kernel
        self._max_items_per_drain = max(1, int(max_items_per_drain))
        self._max_events_per_drain = max(1, int(max_events_per_drain))
        self._budget_ms = budget_ms
        self._capacity_waiters: dict[object, object] = {}
        self._closed = False
        self._fallback_min_ms = 10
        self._fallback_max_ms = 100
        self._fallback_interval_ms = self._fallback_min_ms
        self.fallback_event_polls = 0
        self.fallback_idle_polls = 0
        # Timer category: anti-hang fallback. Adaptive poll for missed queued
        # completion signals; event/idle poll counters make activity visible.
        self._fallback_timer = Qt.QtCore.QTimer(self)
        self._fallback_timer.setSingleShot(True)
        self._fallback_timer.setInterval(self._fallback_interval_ms)
        self._fallback_timer.timeout.connect(self._on_fallback)
        self.eventsReady.connect(self._drain, Qt.QtCore.Qt.ConnectionType.QueuedConnection)
        kernel.completions.set_notify(self._notify_from_worker)

    # ------------------------------------------------------------- wiring

    def _notify_from_worker(self) -> None:
        if self._closed:
            return
        try:
            self.eventsReady.emit()
        except RuntimeError:
            pass

    def close(self) -> None:
        self._closed = True
        self.kernel.completions.set_notify(None)
        self._fallback_timer.stop()

    # ---------------------------------------------------------- callbacks

    def notify_when_capacity(self, key, fn) -> None:
        """Arm a one-shot wakeup fired after the next processed completion."""

        self._capacity_waiters[key] = fn

    def set_max_items_per_drain(self, count: int) -> None:
        self._max_items_per_drain = max(1, int(count))

    def set_budget_ms(self, budget_ms: float | None) -> None:
        self._budget_ms = None if budget_ms is None else max(0.25, float(budget_ms))

    # -------------------------------------------------------------- drain

    def _drain(self) -> None:
        if self._closed:
            return
        budget = GuiCallbackBudget(
            channel="kernel_bridge_drain",
            work_class="evaluation_callback",
            backend="qt",
            target_ms=8.0 if self._budget_ms is None else float(self._budget_ms),
            item_cap=self._max_items_per_drain,
            byte_cap=0,
        )
        queue = self.kernel.completions
        processed = 0
        while processed < self._max_events_per_drain:
            event = queue.pop()
            if event is None:
                break
            processed += 1
            outcome = self.kernel.dispatch_event(event)
            if outcome in (TaskOutcome.COMPLETED, TaskOutcome.FAILED, TaskOutcome.STALE_REUSED):
                budget.record_item()
            if budget.should_yield():
                break
        if budget.processed_items > 0 or budget.elapsed_ms >= budget.warning_ms:
            recorder = getattr(
                getattr(self.parent(), "resource_governor", None),
                "record_gui_callback_observation",
                None,
            )
            if callable(recorder):
                recorder(budget.observation())
        if processed:
            notifier = getattr(self.parent(), "_note_kernel_completion_drain", None)
            if callable(notifier):
                notifier()
        if not queue.empty():
            self._notify_from_worker()
            self._schedule_fallback()
        elif not self.kernel.wait_idle(timeout=0.0):
            # Work is still running; keep the safety net armed until quiet.
            self._schedule_fallback()
        else:
            self._fallback_timer.stop()
        if processed:
            self._fire_capacity_waiters()

    def _fire_capacity_waiters(self) -> None:
        if not self._capacity_waiters:
            return
        waiters = tuple(self._capacity_waiters.values())
        self._capacity_waiters.clear()
        for fn in waiters:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - waiter boundary
                handle_ui_exception("kernel capacity waiter", exc)

    def _schedule_fallback(self) -> None:
        if self._closed:
            return
        if not self._fallback_timer.isActive():
            self._fallback_timer.start()

    def _on_fallback(self) -> None:
        # Adapt cadence from what the poll found: events pending mean the
        # signal path missed — snap fast; an empty poll backs off.
        if not self.kernel.completions.empty():
            self.fallback_event_polls += 1
            self._fallback_interval_ms = self._fallback_min_ms
        else:
            self.fallback_idle_polls += 1
            self._fallback_interval_ms = min(self._fallback_max_ms, self._fallback_interval_ms * 2)
        self._fallback_timer.setInterval(self._fallback_interval_ms)
        self._drain()


__all__ = ["QtKernelBridge"]
