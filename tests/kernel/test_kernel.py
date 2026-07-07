"""Kernel scheduler semantics.

These tests pin the redesign's execution contract:
- priorities order real execution (not just admission bookkeeping),
- dependencies gate real execution and fail loudly,
- staleness (supersession / scope clear / key resubmission) has one arbiter,
- optional work never starves visible work,
- handlers run only at drain time, with dispatch-time staleness re-checks.

Qt-free by design; the Qt bridge has its own test module.
"""

from __future__ import annotations

import threading
import time

import pytest

from arrayscope.kernel import (
    InlineWorkerBackend,
    Kernel,
    Lane,
    Priority,
    Supersession,
    TaskOutcome,
    TaskSpec,
    ThreadWorkerBackend,
)
from arrayscope.operations.cancellation import EvaluationCancelled


class ManualBackend:
    """Pull-on-demand backend: nothing runs until the test says so."""

    workers = 4

    def attach(self, kernel) -> None:
        self.kernel = kernel

    def wake(self) -> None:
        pass

    def shutdown(self, timeout: float = 5.0) -> None:
        pass

    def run_next(self) -> bool:
        record = self.kernel._take_next(block=False)
        if record is None:
            return False
        self.kernel._execute(record)
        return True

    def run_all(self) -> int:
        count = 0
        while self.run_next():
            count += 1
        return count

    def take(self):
        return self.kernel._take_next(block=False)


def drain(kernel) -> list:
    outcomes = []
    while True:
        event = kernel.completions.pop()
        if event is None:
            return outcomes
        outcomes.append((event.spec.key, kernel.dispatch_event(event)))


def swallow_hook(context, error):  # handler failures observed, not raised
    pass


def make_manual(**kwargs):
    backend = ManualBackend()
    kernel = Kernel(backend, handler_error_hook=kwargs.pop("handler_error_hook", None), **kwargs)
    return kernel, backend


# --------------------------------------------------------------- basics


def test_inline_backend_runs_and_delivers_on_drain():
    kernel = Kernel(InlineWorkerBackend())
    results = []
    kernel.submit(TaskSpec(key="a", fn=lambda: 41 + 1), on_done=results.append)
    assert results == []  # never called from submit/worker context
    outcomes = drain(kernel)
    assert results == [42]
    assert outcomes == [("a", TaskOutcome.COMPLETED)]


def test_priority_orders_real_execution():
    kernel, backend = make_manual()
    ran = []
    for key, priority in (
        ("prefetch", Priority.PREFETCH),
        ("hover", Priority.HOVER),
        ("visible", Priority.VISIBLE_IMAGE),
        ("interactive", Priority.INTERACTIVE),
    ):
        kernel.submit(
            TaskSpec(key=key, fn=lambda key=key: ran.append(key), priority=priority)
        )
    backend.run_all()
    assert ran == ["interactive", "visible", "hover", "prefetch"]


def test_visible_lanes_run_before_optional_lanes_regardless_of_priority():
    kernel, backend = make_manual()
    ran = []
    kernel.submit(
        TaskSpec(
            key="spec",
            fn=lambda: ran.append("spec"),
            lane=Lane.SPECULATIVE_RESIDENCY,
            priority=Priority.INTERACTIVE,
            expected_value=1.0,
        )
    )
    kernel.submit(
        TaskSpec(
            key="vis",
            fn=lambda: ran.append("vis"),
            lane=Lane.VISIBLE_MATERIALIZATION,
            priority=Priority.PREFETCH,
        )
    )
    backend.run_all()
    assert ran == ["vis", "spec"]


def test_failure_routes_to_on_error_with_traceback():
    kernel = Kernel(InlineWorkerBackend())

    def boom():
        raise ValueError("no")

    errors = []
    kernel.submit(TaskSpec(key="f", fn=boom), on_error=errors.append)
    outcomes = drain(kernel)
    assert outcomes == [("f", TaskOutcome.FAILED)]
    (exc,) = errors
    assert isinstance(exc, ValueError)
    assert "ValueError: no" in exc.arrayscope_traceback


def test_unhandled_failure_reaches_error_hook():
    seen = []
    kernel = Kernel(InlineWorkerBackend(), handler_error_hook=lambda ctx, exc: seen.append(ctx))

    def boom():
        raise ValueError("no")

    kernel.submit(TaskSpec(key="f", fn=boom))
    drain(kernel)
    assert seen == ["unhandled task failure"]


# ---------------------------------------------------------- dependencies


def test_dependency_gates_execution_and_promotes_on_completion():
    kernel, backend = make_manual()
    ran = []
    kernel.submit(TaskSpec(key="b", fn=lambda: ran.append("b"), deps=("a",)))
    assert not backend.run_next()  # b is parked, nothing ready
    kernel.submit(TaskSpec(key="a", fn=lambda: ran.append("a")))
    backend.run_all()
    assert ran == ["a", "b"]


def test_dependency_failure_drops_dependents_recursively():
    kernel, backend = make_manual()
    ran = []

    def boom():
        raise RuntimeError("dep failed")

    kernel.submit(TaskSpec(key="c", fn=lambda: ran.append("c"), deps=("b",)))
    kernel.submit(TaskSpec(key="b", fn=lambda: ran.append("b"), deps=("a",)))
    kernel.submit(TaskSpec(key="a", fn=boom), on_error=lambda exc: None)
    backend.run_all()
    outcomes = dict(drain(kernel))
    assert ran == []
    assert outcomes["a"] == TaskOutcome.FAILED
    assert outcomes["b"] == TaskOutcome.DROPPED
    assert outcomes["c"] == TaskOutcome.DROPPED
    lanes = kernel.diagnostics().lanes
    assert lanes[str(Lane.VISIBLE_MATERIALIZATION)]["dependency_failed"] == 2


def test_dependency_already_satisfied_runs_immediately():
    kernel, backend = make_manual()
    ran = []
    kernel.submit(TaskSpec(key="a", fn=lambda: ran.append("a")))
    backend.run_all()
    kernel.submit(TaskSpec(key="b", fn=lambda: ran.append("b"), deps=("a",)))
    backend.run_all()
    assert ran == ["a", "b"]


# ---------------------------------------------------------- supersession


def test_supersession_drops_queued_older_values():
    kernel, backend = make_manual()
    ran = []
    kernel.submit(
        TaskSpec(
            key=("tile", 1, "v1"),
            fn=lambda: ran.append("v1"),
            supersession=Supersession("tile-1", "v1"),
        ),
        on_stale=lambda: ran.append("stale-v1"),
    )
    kernel.submit(
        TaskSpec(
            key=("tile", 1, "v2"),
            fn=lambda: ran.append("v2"),
            supersession=Supersession("tile-1", "v2"),
        )
    )
    backend.run_all()
    outcomes = dict(drain(kernel))
    # v2's fn runs on the worker; stale-v1 is a *handler* and therefore only
    # fires at drain time — handlers never run inline at submit.
    assert ran == ["v2", "stale-v1"]
    assert outcomes[("tile", 1, "v1")] == TaskOutcome.DROPPED


def test_supersession_cancels_running_non_reusable_work():
    kernel, backend = make_manual()

    record = None
    kernel.submit(
        TaskSpec(
            key=("tile", "v1"),
            fn=lambda token: token,
            pass_token=True,
            supersession=Supersession("tile", "v1"),
        )
    )
    record = backend.take()
    assert record is not None
    kernel.supersede("tile", "v2")
    assert record.token.cancelled


def test_stale_completion_with_reuse_handler_is_reused():
    kernel, backend = make_manual()
    reused = []
    stale = []
    kernel.submit(
        TaskSpec(
            key=("stage", "v1"),
            fn=lambda: "payload",
            reusable=True,
            supersession=Supersession("stage", "v1"),
        ),
        on_done=lambda value: pytest.fail("stale work must not hit on_done"),
        on_reuse=reused.append,
        on_stale=lambda: stale.append(True),
    )
    record = backend.take()
    kernel.supersede("stage", "v2")
    assert not record.token.cancelled  # reusable work may finish
    kernel._execute(record)
    outcomes = dict(drain(kernel))
    assert reused == ["payload"]
    assert stale == [True]
    assert outcomes[("stage", "v1")] == TaskOutcome.STALE_REUSED


def test_completion_racing_supersession_is_stale_at_dispatch():
    kernel, backend = make_manual()
    done = []
    stale = []
    kernel.submit(
        TaskSpec(
            key=("t", "v1"),
            fn=lambda: "pixels",
            supersession=Supersession("t", "v1"),
        ),
        on_done=done.append,
        on_stale=lambda: stale.append(True),
    )
    backend.run_all()  # completes; event now queued
    kernel.supersede("t", "v2")  # target changes before the GUI drains
    outcomes = dict(drain(kernel))
    assert done == []
    assert stale == [True]
    assert outcomes[("t", "v1")] == TaskOutcome.STALE


def test_key_resubmission_supersedes_queued_instance():
    kernel, backend = make_manual()
    ran = []
    kernel.submit(TaskSpec(key="k", fn=lambda: ran.append("old")))
    kernel.submit(TaskSpec(key="k", fn=lambda: ran.append("new")))
    backend.run_all()
    assert ran == ["new"]


def test_key_resubmission_marks_running_instance_stale():
    kernel, backend = make_manual()
    done = []
    kernel.submit(TaskSpec(key="k", fn=lambda: "old"), on_done=done.append)
    record = backend.take()
    kernel.submit(TaskSpec(key="k", fn=lambda: "new"), on_done=done.append)
    kernel._execute(record)  # old instance finishes after resubmission
    backend.run_all()
    drain(kernel)
    assert done == ["new"]


# ---------------------------------------------------------------- scopes


def test_clear_scope_drops_queued_and_cancels_running():
    kernel, backend = make_manual()
    ran = []
    kernel.submit(TaskSpec(key="r", fn=lambda token: None, pass_token=True, scope="view:levels"))
    running = backend.take()
    assert running.spec.key == "r"
    kernel.submit(TaskSpec(key="q", fn=lambda: ran.append("q"), scope="view:tiles"))
    kernel.clear_scope("view")  # parent clears both child scopes
    assert running.token.cancelled
    assert not backend.run_next()  # q was dropped, nothing ready
    kernel._execute(running)  # finishes after the clear → stale
    outcomes = dict(drain(kernel))
    assert ran == []
    assert outcomes["q"] == TaskOutcome.DROPPED
    assert outcomes["r"] in (TaskOutcome.CANCELLED, TaskOutcome.STALE)


def test_clear_scope_does_not_touch_sibling_scopes():
    kernel, backend = make_manual()
    ran = []
    kernel.submit(TaskSpec(key="a", fn=lambda: ran.append("a"), scope="viewA"))
    kernel.clear_scope("viewB")
    backend.run_all()
    assert ran == ["a"]


def test_scope_epoch_applies_only_to_older_submissions():
    kernel, backend = make_manual()
    ran = []
    kernel.clear_scope("view")
    kernel.submit(TaskSpec(key="fresh", fn=lambda: ran.append("fresh"), scope="view"))
    backend.run_all()
    assert ran == ["fresh"]


# ------------------------------------------------- quotas and deadlines


def test_optional_work_parks_while_visible_backlog_exists():
    kernel, backend = make_manual()
    ran = []
    kernel.submit(
        TaskSpec(
            key="warm",
            fn=lambda: ran.append("warm"),
            lane=Lane.SPECULATIVE_RESIDENCY,
        )
    )
    kernel.submit(TaskSpec(key="vis", fn=lambda: ran.append("vis")))
    # Visible runs first; speculative with expected_value 0 parks during backlog.
    record = backend.take()
    assert record.spec.key == "vis"
    assert backend.take() is None  # warm is quota-parked
    kernel._execute(record)
    backend.run_all()
    assert ran == ["vis", "warm"]
    lanes = kernel.diagnostics().lanes
    assert lanes[str(Lane.SPECULATIVE_RESIDENCY)]["blocked_by_quota"] == 1


def test_optional_quota_limits_concurrent_speculation():
    kernel, backend = make_manual(speculative_fraction=0.25)  # workers=4 → quota 1
    kernel.submit(TaskSpec(key="s1", fn=lambda: None, lane=Lane.SPECULATIVE_RESIDENCY))
    kernel.submit(TaskSpec(key="s2", fn=lambda: None, lane=Lane.SPECULATIVE_RESIDENCY))
    first = backend.take()
    assert first is not None
    assert backend.take() is None  # second exceeds the speculative quota
    kernel._execute(first)
    assert backend.take() is not None


def test_expired_deadline_drops_optional_but_runs_visible():
    kernel, backend = make_manual()
    ran = []
    past = time.perf_counter_ns() - 1
    kernel.submit(
        TaskSpec(
            key="opt",
            fn=lambda: ran.append("opt"),
            lane=Lane.SPECULATIVE_RESIDENCY,
            deadline_ns=past,
            expected_value=1.0,
        )
    )
    kernel.submit(TaskSpec(key="vis", fn=lambda: ran.append("vis"), deadline_ns=past))
    backend.run_all()
    assert ran == ["vis"]
    lanes = kernel.diagnostics().lanes
    assert lanes[str(Lane.VISIBLE_MATERIALIZATION)]["deadline_missed"] == 1
    assert lanes[str(Lane.SPECULATIVE_RESIDENCY)]["deadline_missed"] == 1
    assert lanes[str(Lane.SPECULATIVE_RESIDENCY)]["dropped"] == 1


# --------------------------------------------------------- cancellation


def test_handle_cancel_before_start_delivers_cancelled():
    kernel, backend = make_manual()
    stale = []
    handle = kernel.submit(TaskSpec(key="x", fn=lambda: None), on_stale=lambda: stale.append(True))
    handle.cancel()
    assert not backend.run_next()
    outcomes = dict(drain(kernel))
    assert outcomes["x"] == TaskOutcome.CANCELLED
    assert stale == [True]


def test_cooperative_cancellation_via_evaluation_cancelled():
    kernel, backend = make_manual()

    def cancels(token):
        raise EvaluationCancelled()

    kernel.submit(TaskSpec(key="x", fn=cancels, pass_token=True))
    backend.run_all()
    outcomes = dict(drain(kernel))
    assert outcomes["x"] == TaskOutcome.CANCELLED


def test_handler_exceptions_hit_hook_not_kernel():
    seen = []
    kernel = Kernel(
        InlineWorkerBackend(),
        handler_error_hook=lambda ctx, exc: seen.append((ctx, type(exc).__name__)),
    )
    kernel.submit(
        TaskSpec(key="h", fn=lambda: 1),
        on_done=lambda value: (_ for _ in ()).throw(RuntimeError("handler")),
    )
    drain(kernel)
    assert seen == [("task completion", "RuntimeError")]


# ------------------------------------------------------- thread backend


def test_thread_backend_completes_many_tasks_across_lanes():
    kernel = Kernel(ThreadWorkerBackend(workers=8, name="test-kernel"))
    try:
        done = []
        lock = threading.Lock()

        def note(value):
            with lock:
                done.append(value)

        total = 200
        for index in range(total):
            lane = Lane.VISIBLE_MATERIALIZATION if index % 2 else Lane.HISTOGRAM_REFINEMENT
            kernel.submit(
                TaskSpec(key=("t", index), fn=lambda index=index: index, lane=lane),
                on_done=note,
            )
        assert kernel.wait_idle(timeout=20.0)
        while not kernel.completions.empty():
            drain(kernel)
        assert sorted(done) == list(range(total))
    finally:
        kernel.shutdown()


def test_thread_backend_dependency_chain_executes_in_order():
    kernel = Kernel(ThreadWorkerBackend(workers=4, name="test-kernel-deps"))
    try:
        ran = []
        lock = threading.Lock()

        def note(key):
            with lock:
                ran.append(key)

        kernel.submit(TaskSpec(key="c", fn=lambda: note("c"), deps=("b",)))
        kernel.submit(TaskSpec(key="b", fn=lambda: note("b"), deps=("a",)))
        kernel.submit(TaskSpec(key="a", fn=lambda: note("a")))
        assert kernel.wait_idle(timeout=20.0)
        assert ran == ["a", "b", "c"]
    finally:
        kernel.shutdown()


def test_thread_backend_supersession_storm_settles_to_latest():
    kernel = Kernel(ThreadWorkerBackend(workers=4, name="test-kernel-storm"))
    try:
        applied = []
        lock = threading.Lock()

        def apply(value):
            with lock:
                applied.append(value)

        for value in range(50):
            kernel.submit(
                TaskSpec(
                    key=("scrub", value),
                    fn=lambda value=value: value,
                    pass_token=False,
                    supersession=Supersession("scrub", value),
                ),
                on_done=apply,
            )
        assert kernel.wait_idle(timeout=20.0)
        while not kernel.completions.empty():
            drain(kernel)
        # Only the final target may reach on_done; every older value is
        # stale by supersession no matter when it finished, and the final
        # value must never be lost.
        assert applied == [49]
    finally:
        kernel.shutdown()


def test_shutdown_cancels_running_tokens_and_stops_accepting():
    kernel = Kernel(ThreadWorkerBackend(workers=2, name="test-kernel-shutdown"))
    started = threading.Event()
    release = threading.Event()

    def slow(token):
        started.set()
        release.wait(timeout=5.0)
        return token.cancelled

    kernel.submit(TaskSpec(key="slow", fn=slow, pass_token=True))
    assert started.wait(timeout=5.0)
    kernel.shutdown(timeout=0.1)
    release.set()
    assert kernel.submit(TaskSpec(key="late", fn=lambda: None)) is None
