"""Settle waits that fail on a wedged pipeline and never on a slow machine.

``qtbot.waitUntil(predicate, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)`` asks
two questions at once and reports only one answer.  The predicate asks "did the
product reach this state?"; the deadline asks "within five seconds of *this
machine's* wall clock?".  When the box is busy the second question fails first,
and the report says ``waitUntil timed out`` — which is also what a real
regression says.

Measured 2026-08-01 on main: three concurrent full suites made fourteen tests
fail that all pass in 41 s on a quiet machine, and **eleven of those failures
were that one message**.  An agent who did not write those tests cannot tell
that from a genuine red, so the whole session goes into bisecting a phantom.

The fix is not a longer deadline — that is how a repository ends up with gates
that cannot fail.  It is a deadline denominated in something the machine's speed
does not move: **the product's own progress**.  This module pumps the event loop
and watches the counters the kernel and the presentation path already publish.
While any of them advances, or while any work is queued, running or waiting to
be delivered, the wait keeps waiting however long the box takes.  The budget
only starts burning once the pipeline is *idle* with the predicate still false.

That state has a name here: a lost wakeup, the ADR 0051 class the montage
watchdog in ``arrayscope/window/frame_runtime.py`` exists to assert on.  It is
not a speed measurement — an idle kernel, an empty completion queue and no armed
timer mean the same thing on a saturated box as on a quiet one, so this budget
takes exactly as long to fire either way.  That is the point: what remains after
the change is a wedge detector, and a wedge is what these tests are for.

So the constant is unchanged and still owned by
:mod:`arrayscope.tools.interaction_budget`; what changed is what it measures.
Nobody can widen it, because widening it is not what makes a slow run pass.

Wall-clock settlement *is* a real product property — the 2 s target and 5 s hard
limit in ``docs/testing/strategy.md`` are acceptance criteria, not decoration.
They are measured where a measurement means something: the benchmark harness
(``arrayscope.tools.profile_montage_workflow``), on a quiet machine, with the
load line from :mod:`tests.load_report` recorded beside the number.
"""

from __future__ import annotations

from time import perf_counter

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_S


class SettleTimeout(AssertionError):
    """The product went idle without reaching the state, i.e. a lost wakeup.

    A distinct type so a caller can add its own diagnostics to this failure
    without also swallowing an ``assert`` from inside its own predicate.
    """


#: How long one pump slice runs the event loop.  Short enough that a predicate
#: flipping mid-slice is noticed promptly, long enough that the pump is not
#: itself the dominant cost on a saturated box.
_PUMP_SLICE_MS = 20


def _application():
    from pyqtgraph.Qt import QtWidgets

    return QtWidgets.QApplication.instance()


def _pump(*, allow_idle: bool) -> None:
    """Run one slice of the event loop.

    The idle form runs a nested event loop rather than spinning on
    ``processEvents``.  A spin holds the GIL continuously, which on a busy box
    starves the very worker threads the wait is waiting for — the pump would
    then be manufacturing the stall it reports.  A nested loop blocks in the
    platform dispatcher and releases the GIL, which is also what
    ``qtbot.waitUntil`` does.

    ``allow_idle=False`` keeps the old busy pump for the tests that need it: a
    real event loop never idles, so low-priority planning continuations only run
    under sustained queue pressure, and an idle loop dispatches them instantly
    and hides the field scheduling entirely.
    """

    from pyqtgraph.Qt import QtCore

    app = _application()
    if app is None:
        return
    if not allow_idle:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, _PUMP_SLICE_MS)
        return
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(_PUMP_SLICE_MS, loop.quit)
    loop.exec()


def _kernel_diagnostics(win):
    kernel = getattr(win, "kernel", None)
    diagnostics = getattr(kernel, "diagnostics", None)
    return None if diagnostics is None else diagnostics()


def _kernel_counters(win) -> tuple:
    """Every per-lane counter the kernel publishes, plus its queue depths."""

    diag = _kernel_diagnostics(win)
    if diag is None:
        return ()
    lanes = tuple(
        (lane, name, int(value))
        for lane, counters in sorted((diag.lanes or {}).items())
        for name, value in sorted((counters or {}).items())
    )
    return (
        lanes,
        int(diag.queued),
        int(diag.running),
        int(diag.parked_deps),
        int(diag.parked_quota),
        int(diag.completed_keys),
        int(diag.visible_backlog),
    )


def _presentation_counters(win) -> tuple:
    """Commit/draw/republication serials the display path already stamps."""

    view = getattr(win, "img_view", None)
    session = getattr(getattr(win, "renderer", None), "_frame_session", None)
    unsettled = ()
    if session is not None:
        unsettled = tuple(session.required_target_unsettled_tiles())
    presentation = getattr(session, "tile_presentation_state", None)
    return (
        int(getattr(view, "_wgpu_commit_serial", 0) or 0),
        int(getattr(view, "_wgpu_binding_full_republications", 0) or 0),
        int(getattr(session, "commit_batches", 0) or 0),
        int(getattr(presentation, "revision", 0) or 0),
        len(unsettled),
        len(tuple(getattr(session, "visible_tile_numbers", ()) or ())),
        len(tuple(getattr(session, "_last_active_tiles", ()) or ())),
    )


def window_progress(win) -> tuple:
    """A comparable snapshot of what this window has *done* so far.

    Deliberately coarse: the wait only asks whether it changed, never what the
    values mean.  Any advance anywhere is evidence that the pipeline is alive,
    which is the only question the budget is allowed to answer.
    """

    return (_kernel_counters(win), _presentation_counters(win))


def window_work_in_flight(win) -> int:
    """How much work this window still owns: queued, running, or undelivered.

    Non-zero means a slow machine, not a wedged one.  The completion queue and
    the bridge's fallback timer are part of the answer because a result that has
    been computed but not yet handed to the GUI thread is still work in
    progress — the montage watchdog draws the same line.
    """

    in_flight = 0
    diag = _kernel_diagnostics(win)
    if diag is not None:
        in_flight += int(diag.queued) + int(diag.running)
        in_flight += int(diag.parked_deps) + int(diag.parked_quota)
    completions = getattr(getattr(win, "kernel", None), "completions", None)
    if completions is not None and not completions.empty():
        in_flight += 1
    bridge = getattr(win, "kernel_bridge", None)
    fallback = getattr(bridge, "_fallback_timer", None)
    if fallback is not None and fallback.isActive():
        in_flight += 1
    for name in ("montage_tile_evaluation_controller", "visible_evaluation_controller"):
        controller = getattr(win, name, None)
        if controller is not None and controller.is_busy():
            in_flight += 1
    coordinator = getattr(win, "render_coordinator", None)
    if coordinator is not None and coordinator.has_pending_render:
        in_flight += 1
    return in_flight


def _wedge_evidence(win) -> str:
    """What the product itself says about being stuck, for the failure text."""

    renderer = getattr(win, "renderer", None)
    assertions = int(getattr(renderer, "_montage_stall_assertions", 0) or 0)
    if not assertions:
        return "the montage stall watchdog has not asserted"
    return (
        f"the montage stall watchdog asserted {assertions}x "
        f"(last signature {getattr(renderer, '_montage_watchdog_last_stall', None)!r}, "
        f"trace {getattr(renderer, '_montage_watchdog_last_trace_path', None)!r})"
    )


def _call_site_label(depth: int) -> str:
    """Name the waiting line, for reports where no better name was given."""

    import inspect

    frame = inspect.currentframe()
    for _ in range(depth + 1):
        if frame is None:
            return "settle"
        frame = frame.f_back
    if frame is None:
        return "settle"
    return f"{frame.f_code.co_name}:{frame.f_lineno}"


def settle_until(
    predicate,
    *,
    progress,
    label=None,
    in_flight=None,
    evidence=None,
    allow_idle=True,
):
    """Pump until ``predicate`` holds, failing only on a pipeline that stopped.

    ``progress`` returns a comparable snapshot of production-published counters
    and ``in_flight`` returns how much work the product still owns.  The budget
    is re-armed whenever either says the product moved, so it bounds *stalled*
    time, never elapsed time.

    There is deliberately no way to pass a different budget.  A caller who could
    ask for a longer one would have re-created the failure mode this module
    exists to remove, and a shorter one has nothing to mean: what is being
    bounded is how long the product may sit with no work and no wakeup, and that
    number is the same on every machine.
    """

    budget_s = INTERACTION_SETTLE_HARD_LIMIT_S
    label = label or _call_site_label(2)
    started = perf_counter()
    deadline = started + budget_s
    last = progress()
    while True:
        if predicate():
            return
        _pump(allow_idle=allow_idle)
        now = progress()
        if now != last or (in_flight is not None and in_flight()):
            last = now
            deadline = perf_counter() + budget_s
            continue
        if perf_counter() < deadline:
            continue
        if predicate():
            return
        detail = "" if evidence is None else f" Product evidence: {evidence()}."
        raise SettleTimeout(
            f"{label}: no published progress and no work in flight for "
            f"{budget_s:.1f}s, and still not settled after "
            f"{perf_counter() - started:.1f}s. This is a lost wakeup (ADR 0051), "
            f"not a slow machine — every counter that moves re-arms the budget, "
            f"so a busy box does not reach this line.{detail} "
            f"Last snapshot: {_abbreviated(last)}"
        )


def _abbreviated(snapshot, limit: int = 600) -> str:
    """The tail of the state, short enough to read in a failure report.

    The full counter vector runs to several thousand characters; printed in
    full it buries the sentence above it, which is the part that says what
    kind of failure this is.
    """

    text = repr(snapshot)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (+{len(text) - limit} chars)"


def settle_window(win, predicate, *, label=None, allow_idle=True):
    """:func:`settle_until` wired to one window's published progress."""

    label = label or _call_site_label(1)
    settle_until(
        predicate,
        progress=lambda: window_progress(win),
        in_flight=lambda: window_work_in_flight(win),
        evidence=lambda: _wedge_evidence(win),
        label=label,
        allow_idle=allow_idle,
    )
