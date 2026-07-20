import os
import threading
import time

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _wait_until(qtbot, predicate, *, timeout_ms=INTERACTION_SETTLE_HARD_LIMIT_MS):
    qtbot.waitUntil(predicate, timeout=min(int(timeout_ms), INTERACTION_SETTLE_HARD_LIMIT_MS))


def test_evaluation_controller_ignores_stale_results(qtbot):
    from arrayscope.kernel.eval_adapter import KernelEvaluationController as EvaluationController

    controller = EvaluationController(max_workers=1)
    done = []
    stale = []
    try:
        controller.start(
            lambda: (time.sleep(0.05), "old")[1],
            on_done=done.append,
            on_stale=lambda: stale.append("old"),
        )
        controller.start(lambda: "new", on_done=done.append, on_stale=lambda: stale.append("new"))

        _wait_until(qtbot, lambda: done == ["new"] and stale == ["old"])
    finally:
        controller.shutdown_for_close()


def test_start_latest_uses_kernel_scope_clear_not_pool_clear_luck(qtbot):
    """R1: clear_queued/clear_group are kernel scope clears, not pool.clear()."""

    from arrayscope.kernel import Priority as EvalPriority
    from arrayscope.kernel.eval_adapter import KernelEvaluationController as EvaluationController

    controller = EvaluationController(max_workers=1)
    entered = threading.Event()
    done = []
    stale = []
    try:
        controller.start_latest(
            lambda: entered.set() or (time.sleep(0.05), "old")[1],
            key="old",
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="visible",
            on_done=done.append,
            on_stale=lambda: stale.append("old"),
        )
        _wait_until(qtbot, entered.is_set)
        controller.start_latest(
            lambda: "new",
            key="new",
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="visible",
            on_done=done.append,
            on_stale=lambda: stale.append("new"),
        )

        _wait_until(qtbot, lambda: done == ["new"] and stale == ["old"])
    finally:
        controller.shutdown_for_close()


def test_active_plus_latest_reuses_stale_completion(qtbot):
    from arrayscope.kernel import Priority as EvalPriority
    from arrayscope.kernel.eval_adapter import KernelEvaluationController as EvaluationController

    controller = EvaluationController(max_workers=1)
    entered = threading.Event()
    done = []
    reused = []
    stale = []
    try:
        controller.start_active_plus_latest(
            lambda: entered.set() or (time.sleep(0.05), "active")[1],
            key="active",
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="visible",
            supersession_key="visible-image",
            supersession_value="old",
            on_done=done.append,
            on_stale=lambda: stale.append("active"),
            on_reuse_stale=reused.append,
        )
        _wait_until(qtbot, entered.is_set)
        controller.start_active_plus_latest(
            lambda: "latest",
            key="latest",
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="visible",
            supersession_key="visible-image",
            supersession_value="new",
            on_done=done.append,
            on_stale=lambda: stale.append("latest"),
        )

        _wait_until(
            qtbot, lambda: done == ["latest"] and reused == ["active"] and stale == ["active"]
        )
        assert controller.diagnostics().stale_reused >= 1
    finally:
        controller.shutdown_for_close()


def test_supersession_does_not_advance_group_generation(qtbot):
    from arrayscope.kernel import Priority as EvalPriority
    from arrayscope.kernel.eval_adapter import KernelEvaluationController as EvaluationController

    controller = EvaluationController(max_workers=1)
    done = []
    try:
        before = controller.group_generation("visible")
        controller.start_active_plus_latest(
            lambda: "latest",
            key="latest",
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="visible",
            supersession_key="visible-image",
            supersession_value="target",
            on_done=done.append,
        )

        _wait_until(qtbot, lambda: done == ["latest"])
        assert controller.group_generation("visible") == before
    finally:
        controller.shutdown_for_close()


def test_clear_group_prefix_invalidates_child_group(qtbot):
    from arrayscope.kernel import Priority as EvalPriority
    from arrayscope.kernel.eval_adapter import KernelEvaluationController as EvaluationController

    controller = EvaluationController(max_workers=1)
    entered = threading.Event()
    done = []
    stale = []
    try:
        controller.start_latest(
            lambda: entered.set() or (time.sleep(0.05), "tile")[1],
            key="tile",
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="montage-tile:1:2",
            on_done=done.append,
            on_stale=lambda: stale.append("tile"),
        )
        _wait_until(qtbot, entered.is_set)
        before = controller.group_generation("montage-tile:1:2")

        controller.clear_group("montage-tile")

        assert controller.group_generation("montage-tile:1:2") > before
        _wait_until(qtbot, lambda: stale == ["tile"])
        assert done == []
    finally:
        controller.shutdown_for_close()


def test_start_prefetch_local_gates_then_submits_to_kernel(qtbot):
    from arrayscope.kernel.eval_adapter import KernelEvaluationController as EvaluationController

    controller = EvaluationController(max_workers=1)
    done = []
    try:
        first = controller.start_prefetch(lambda: "a", key=("prefetch", 1), on_done=done.append)
        duplicate = controller.start_prefetch(lambda: "b", key=("prefetch", 1))
        controller.set_max_prefetch(1)
        limited = controller.start_prefetch(lambda: "c", key=("prefetch", 2))
        idle = controller.start_prefetch(lambda: "d", key=("prefetch", 3), idle_elapsed=False)
        cost = controller.start_prefetch(lambda: "e", key=("prefetch", 4), memory_budget_bytes=0)

        _wait_until(qtbot, lambda: done == ["a"])
        assert first.scheduled
        assert not duplicate.scheduled
        assert duplicate.reason == "deduped"
        assert not limited.scheduled
        assert limited.reason == "limited"
        assert not idle.scheduled
        assert idle.reason == "idle"
        assert not cost.scheduled
        assert cost.reason == "cost"
    finally:
        controller.shutdown_for_close()


def test_start_prefetch_forwards_error_and_stale_terminals(qtbot):
    import threading
    import time

    from arrayscope.kernel.eval_adapter import KernelEvaluationController as EvaluationController

    controller = EvaluationController(max_workers=1)
    errors = []
    stale = []
    entered = threading.Event()
    try:
        controller.start_prefetch(
            lambda: (_ for _ in ()).throw(RuntimeError("prefetch failed")),
            key=("prefetch", "error"),
            on_error=errors.append,
            on_stale=lambda: stale.append("error-stale"),
        )
        _wait_until(qtbot, lambda: len(errors) == 1)
        assert str(errors[0]) == "prefetch failed"
        assert stale == []

        controller.start_prefetch(
            lambda: entered.set() or (time.sleep(0.05), "late")[1],
            key=("prefetch", "stale"),
            on_stale=lambda: stale.append("cancelled"),
        )
        _wait_until(qtbot, entered.is_set)
        controller.cancel_prefetch()
        _wait_until(qtbot, lambda: stale == ["cancelled"])
        assert not controller.is_busy()
    finally:
        controller.shutdown_for_close()


def test_start_latest_can_pass_cancellation_token(qtbot):
    from arrayscope.kernel import Priority as EvalPriority
    from arrayscope.kernel.eval_adapter import KernelEvaluationController as EvaluationController

    controller = EvaluationController(max_workers=1)
    seen = []
    try:
        controller.start_latest(
            lambda token: seen.append(hasattr(token, "cancelled")) or "ok",
            key="token",
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="visible",
            on_done=lambda _value: None,
            pass_token=True,
        )

        _wait_until(qtbot, lambda: seen == [True])
    finally:
        controller.shutdown_for_close()


def test_cancelled_evaluation_does_not_call_error(qtbot):
    from arrayscope.kernel import Priority as EvalPriority
    from arrayscope.kernel.eval_adapter import KernelEvaluationController as EvaluationController
    from arrayscope.operations.cancellation import EvaluationCancelled

    controller = EvaluationController(max_workers=1)
    errors = []
    stale = []
    try:
        controller.start_latest(
            lambda _token: (_ for _ in ()).throw(EvaluationCancelled()),
            key="cancel",
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="visible",
            on_done=lambda _value: None,
            on_error=errors.append,
            on_stale=lambda: stale.append(True),
            pass_token=True,
        )

        _wait_until(qtbot, lambda: stale == [True])
        assert errors == []
    finally:
        controller.shutdown_for_close()


def test_drain_fallback_counters_live_on_bridge_once(qtbot):
    """R1: fallback polls are bridge diagnostics, not per-controller state."""

    from arrayscope.kernel.eval_adapter import KernelEvaluationController as EvaluationController

    controller = EvaluationController(max_workers=1)
    try:
        controller.bridge._on_fallback()

        diagnostics = controller.diagnostics()
        assert diagnostics.fallback_idle_polls == controller.bridge.fallback_idle_polls
        assert diagnostics.fallback_event_polls == controller.bridge.fallback_event_polls
    finally:
        controller.shutdown_for_close()
