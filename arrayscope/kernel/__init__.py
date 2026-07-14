"""ArrayScope execution kernel.

One Qt-free scheduler that **owns execution**: keyed tasks with dependencies,
lanes, priorities, deadlines, supersession, and cooperative cancellation,
running on real worker threads sized to the machine. Completions are delivered
through exactly one bounded fan-in (`CompletionQueue`); the Qt side attaches a
single `QtKernelBridge` that drains on the GUI thread under a
`GuiCallbackBudget`.

Design rules (ADR 0053):

- The kernel never calls user handlers from worker threads. Handlers run only
  during a drain, on the thread that owns the sink (the GUI thread in the app,
  the test thread in tests).
- The GUI thread is a gateway: it submits, drains, and applies commits. All
  evaluation, reduction, statistics, and planning-heavy work belongs in tasks.
- Staleness is decided by the kernel (scopes + supersession families), never
  by ad-hoc comparisons at call sites.
- Timers are anti-hang fallbacks, not scheduling. The bridge's fallback timer
  exists only to survive a missed cross-thread signal.
- Worker backends are swappable (`ThreadWorkerBackend` today; a free-threaded
  or process backend must not require kernel changes).
"""

from arrayscope.kernel.task import (
    Lane,
    Priority,
    Supersession,
    TaskOutcome,
    TaskSpec,
    UNRANKED_SCHEDULING_RANK,
    VISIBLE_LANES,
    WorkItem,
    complete_inline_work,
)
from arrayscope.kernel.completions import CompletionEvent, CompletionQueue, DrainBudget
from arrayscope.kernel.scheduler import Kernel, TaskHandle
from arrayscope.kernel.workers import InlineWorkerBackend, ThreadWorkerBackend, WorkerBackend

__all__ = [
    "CompletionEvent",
    "CompletionQueue",
    "DrainBudget",
    "InlineWorkerBackend",
    "Kernel",
    "Lane",
    "Priority",
    "Supersession",
    "TaskHandle",
    "TaskOutcome",
    "TaskSpec",
    "UNRANKED_SCHEDULING_RANK",
    "ThreadWorkerBackend",
    "VISIBLE_LANES",
    "WorkerBackend",
    "WorkItem",
    "complete_inline_work",
]
