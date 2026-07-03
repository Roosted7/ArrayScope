import os
import inspect
import time

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


_WAIT_TIMEOUT_MS = 5000


def _wait_until(qtbot, predicate, *, timeout_ms=_WAIT_TIMEOUT_MS):
    qtbot.waitUntil(predicate, timeout=timeout_ms)


def _wait_for_started(qtbot, controller):
    _wait_until(qtbot, lambda: bool(controller._started))


def test_evaluation_controller_ignores_stale_results(qtbot):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController()
    done = []
    stale = []

    controller.start(lambda: (time.sleep(0.12), "old")[1], on_done=done.append, on_stale=lambda: stale.append("old"))
    controller.start(lambda: "new", on_done=done.append, on_stale=lambda: stale.append("new"))

    _wait_until(qtbot, lambda: done == ["new"] and stale == ["old"])

    assert done == ["new"]
    assert stale == ["old"]


def test_evaluation_controller_drains_without_poll_timer(qtbot):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController()
    done = []

    controller.start(lambda: "done", on_done=done.append)

    _wait_until(qtbot, lambda: done == ["done"])

    assert done == ["done"]
    assert not hasattr(controller, "_poll_timer")


def test_drain_fallback_backs_off_while_signal_path_is_healthy(qtbot):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController()

    # Empty polls mean the queued-signal path delivered everything: the
    # safety net must back off instead of waking at 100 Hz for the whole
    # duration of long background work.
    intervals = []
    for _ in range(6):
        controller._on_drain_fallback()
        intervals.append(controller._drain_fallback_interval_ms)

    assert intervals[0] > controller._drain_fallback_min_ms
    assert intervals[-1] == controller._drain_fallback_max_ms
    assert controller.diagnostics().fallback_idle_polls == 6

    # An event-bearing fallback poll snaps back to the fast interval. The
    # counter is intentionally descriptive, not proof that a Qt signal failed.
    controller._queue.put(("started", -1, None))
    controller._on_drain_fallback()

    assert controller._drain_fallback_interval_ms == controller._drain_fallback_min_ms
    assert controller.diagnostics().fallback_event_polls == 1


def test_evaluation_controller_dedupes_and_limits_prefetch(qtbot):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController()
    first = controller.start_prefetch(lambda: "a", key=("prefetch", 1))
    duplicate = controller.start_prefetch(lambda: "b", key=("prefetch", 1))
    controller._max_prefetch = 1
    limited = controller.start_prefetch(lambda: "c", key=("prefetch", 2))

    controller.shutdown_for_close()

    assert first.scheduled
    assert not duplicate.scheduled
    assert duplicate.reason == "deduped"
    assert not limited.scheduled
    assert limited.reason == "limited"


def test_evaluation_controller_drain_yields_on_elapsed_budget(qtbot):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController(max_callback_dispatch_per_drain=99)
    controller.set_callback_budget_ms(0.1)
    done = []

    def callback(value):
        time.sleep(0.01)
        done.append(value)

    for key, value in (("a", 1), ("b", 2), ("c", 3)):
        controller._runnables[key] = object()
        controller._handlers[key] = (callback, None, None, None)
        controller._queue.put(("prefetch_done", key, value))

    controller._drain_queue()

    assert done == [1]
    assert len(controller._pending_queue_events) == 2
    assert controller._drain_continuation_pending is True


def test_evaluation_controller_drain_records_budget_observation(qtbot):
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.compute_policy import compute_policy_from_settings
    from arrayscope.core.resource_governor import ResourceGovernor
    from arrayscope.window.evaluation_controller import EvaluationController
    from pyqtgraph.Qt import QtCore

    parent = QtCore.QObject()
    parent.resource_governor = ResourceGovernor(compute_policy_from_settings(AppSettingsState(), cpu_count=4))

    controller = EvaluationController(parent, max_callback_dispatch_per_drain=2, name="visible")
    seen = []
    controller._runnables["a"] = object()
    controller._handlers["a"] = (seen.append, None, None, None)
    controller._queue.put(("prefetch_done", "a", "done"))

    controller._drain_queue()

    callbacks = parent.resource_governor.diagnostics().feedback_channels
    assert seen == ["done"]
    assert any(channel.channel == "visible_queue_drain" for channel in callbacks)


def test_start_latest_clears_queued_work_and_only_commits_newest(qtbot):
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    done = []
    stale = []

    controller.start_latest(
        lambda: (time.sleep(0.12), "old")[1],
        key="old",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=done.append,
        on_stale=lambda: stale.append("old"),
    )
    controller.start_latest(
        lambda: "new",
        key="new",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=done.append,
        on_stale=lambda: stale.append("new"),
    )

    _wait_until(qtbot, lambda: done == ["new"] and "old" in stale)

    assert done == ["new"]
    assert "old" in stale


def test_active_plus_latest_preserves_started_work_and_collapses_queued(qtbot):
    from arrayscope.core.scheduler import FrameTarget
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    done = []
    stale = []
    active_target = FrameTarget("semantic-old", None, "presentation-old", "exact-visible")
    queued_old_target = FrameTarget("semantic-queued-old", None, "presentation-old", "exact-visible")
    queued_new_target = FrameTarget("semantic-queued-new", None, "presentation-new", "exact-visible")

    controller.start_active_plus_latest(
        lambda: (time.sleep(0.12), "active")[1],
        key="active",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        frame_target=active_target,
        on_done=done.append,
        on_stale=lambda: stale.append("active"),
    )
    _wait_for_started(qtbot, controller)
    assert controller._started
    assert controller.frame_progress("visible").active == active_target

    controller.start_active_plus_latest(
        lambda: "queued-old",
        key="queued-old",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        frame_target=queued_old_target,
        on_done=done.append,
        on_stale=lambda: stale.append("queued-old"),
    )
    controller.start_active_plus_latest(
        lambda: "queued-new",
        key="queued-new",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        frame_target=queued_new_target,
        on_done=done.append,
        on_stale=lambda: stale.append("queued-new"),
    )
    progress = controller.frame_progress("visible")
    assert progress.active == active_target
    assert progress.queued_latest == queued_new_target

    _wait_until(qtbot, lambda: done == ["queued-new"] and "active" in stale and "queued-old" in stale)

    assert done == ["queued-new"]
    assert "active" in stale
    assert "queued-old" in stale
    progress = controller.frame_progress("visible")
    assert progress.presented == queued_new_target
    assert progress.active is None
    assert progress.queued_latest is None
    diagnostics = controller.diagnostics()
    assert diagnostics.active_preserved >= 1
    assert diagnostics.queued_collapsed >= 1
    assert diagnostics.presented_target == queued_new_target


def test_visible_controller_requires_work_item_when_parent_has_work_graph(qtbot):
    import pytest
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.work_graph import WorkGraph
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    parent = QtCore.QObject()
    parent.work_graph = WorkGraph()
    controller = EvaluationController(parent=parent, max_workers=1)

    with pytest.raises(ValueError, match="visible evaluation submissions require"):
        controller.start_latest(
            lambda: "value",
            key="visible",
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="visible",
            on_done=lambda _value: None,
        )


def test_generic_start_is_not_visible_work_when_parent_has_work_graph(qtbot):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.work_graph import WorkGraph
    from arrayscope.window.evaluation_controller import EvaluationController

    parent = QtCore.QObject()
    parent.work_graph = WorkGraph()
    controller = EvaluationController(parent=parent, max_workers=1)
    done = []

    controller.start(lambda: "value", on_done=done.append)

    _wait_until(qtbot, lambda: done == ["value"])

    assert done == ["value"]
    assert parent.work_graph.diagnostics().lanes == {}


def test_controller_reports_work_graph_reusable_stale_completion(qtbot):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.scheduler import FrameTarget
    from arrayscope.core.work_graph import WorkGraph, WorkItem, WorkLane
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    parent = QtCore.QObject()
    parent.work_graph = WorkGraph()
    controller = EvaluationController(parent=parent, max_workers=1)
    old_target = FrameTarget("old", None, "presentation", "exact-visible")
    new_target = FrameTarget("new", None, "presentation", "exact-visible")

    controller.start_active_plus_latest(
        lambda: (time.sleep(0.08), "old")[1],
        key="old",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        frame_target=old_target,
        supersession_key="visible-image",
        supersession_value="old",
        work_item=WorkItem(
            key=("visible", "old"),
            lane=WorkLane.VISIBLE_MATERIALIZATION,
            frame_target=old_target,
            supersession_key="visible-image",
            supersession_value="old",
            reusable_output=True,
        ),
        on_done=lambda _value: None,
        on_reuse_stale=lambda _value: None,
    )
    _wait_for_started(qtbot, controller)

    controller.start_active_plus_latest(
        lambda: "new",
        key="new",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        frame_target=new_target,
        supersession_key="visible-image",
        supersession_value="new",
        work_item=WorkItem(
            key=("visible", "new"),
            lane=WorkLane.VISIBLE_MATERIALIZATION,
            frame_target=new_target,
            supersession_key="visible-image",
            supersession_value="new",
        ),
        on_done=lambda _value: None,
    )

    _wait_until(
        qtbot,
        lambda: parent.work_graph.diagnostics().lanes.get("visible_materialization", {}).get("reusable_finished") == 1
        and parent.work_graph.diagnostics().lanes.get("visible_materialization", {}).get("completed") == 1,
    )

    counters = parent.work_graph.diagnostics().lanes["visible_materialization"]
    assert counters["reusable_finished"] == 1
    assert counters["completed"] == 1


def test_active_plus_latest_reuses_stale_completion_without_on_done(qtbot):
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    done = []
    reused = []
    stale = []

    controller.start_active_plus_latest(
        lambda: (time.sleep(0.08), "active")[1],
        key="active",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=done.append,
        on_stale=lambda: stale.append("active"),
        on_reuse_stale=reused.append,
    )
    _wait_for_started(qtbot, controller)

    controller.start_active_plus_latest(
        lambda: "latest",
        key="latest",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=done.append,
        on_stale=lambda: stale.append("latest"),
    )

    _wait_until(qtbot, lambda: done == ["latest"] and reused == ["active"] and stale == ["active"])

    assert done == ["latest"]
    assert reused == ["active"]
    assert stale == ["active"]
    assert controller.diagnostics().stale_reused == 1


def test_active_plus_latest_supersedes_by_key_value_without_group_epoch(qtbot):
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    done = []
    reused = []
    stale = []
    before = controller.group_generation("visible")

    controller.start_active_plus_latest(
        lambda: (time.sleep(0.08), "active")[1],
        key="active",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        supersession_key="visible-image",
        supersession_value="old-target",
        on_done=done.append,
        on_stale=lambda: stale.append("active"),
        on_reuse_stale=reused.append,
    )
    _wait_for_started(qtbot, controller)

    controller.start_active_plus_latest(
        lambda: "latest",
        key="latest",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        supersession_key="visible-image",
        supersession_value="new-target",
        on_done=done.append,
        on_stale=lambda: stale.append("latest"),
    )

    assert controller.group_generation("visible") == before

    _wait_until(qtbot, lambda: done == ["latest"] and reused == ["active"] and stale == ["active"])

    assert done == ["latest"]
    assert reused == ["active"]
    assert stale == ["active"]


def test_active_plus_latest_keeps_unrelated_supersession_key_queued(qtbot):
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    done = []
    stale = []

    controller.start_active_plus_latest(
        lambda: (time.sleep(0.08), "active-image")[1],
        key="active-image",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        supersession_key="visible-image",
        supersession_value="active-target",
        on_done=done.append,
        on_stale=lambda: stale.append("active-image"),
    )
    _wait_for_started(qtbot, controller)

    controller.start_active_plus_latest(
        lambda: "profile",
        key="profile",
        priority=EvalPriority.LIVE_PROFILE,
        replace_group="visible",
        supersession_key="profile",
        supersession_value="profile-target",
        on_done=done.append,
        on_stale=lambda: stale.append("profile"),
    )
    controller.start_active_plus_latest(
        lambda: "old-image",
        key="old-image",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        supersession_key="visible-image",
        supersession_value="old-target",
        on_done=done.append,
        on_stale=lambda: stale.append("old-image"),
    )
    controller.start_active_plus_latest(
        lambda: "new-image",
        key="new-image",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        supersession_key="visible-image",
        supersession_value="new-target",
        on_done=done.append,
        on_stale=lambda: stale.append("new-image"),
    )

    _wait_until(qtbot, lambda: {"profile", "new-image"}.issubset(done) and "old-image" in stale)

    assert "profile" in done
    assert "new-image" in done
    assert "old-image" not in done
    assert "old-image" in stale
    assert "profile" not in stale
    assert controller.diagnostics().queued_collapsed >= 1


def test_clear_group_preserves_unrelated_prefetch_bookkeeping(qtbot):
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    controller.start_latest(
        lambda: (time.sleep(0.12), "old")[1],
        key="old",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=lambda _value: None,
    )
    prefetch = controller.start_prefetch(lambda: "prefetch", key=("prefetch", 1))
    controller.start_latest(
        lambda: "visible",
        key="visible",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=lambda _value: None,
    )

    assert prefetch.scheduled
    assert ("prefetch", 1) in controller._prefetch_keys
    assert ("prefetch", 1) in controller._runnables
    controller.shutdown_for_close()


def test_clear_group_invalidates_even_without_active_runnable(qtbot):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController(max_workers=1)
    before = controller.group_generation("visible")

    controller.clear_group("visible")

    assert controller.group_generation("visible") > before


def test_clear_group_prefix_invalidates_child_group(qtbot):
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    stale = []
    done = []
    controller.start_latest(
        lambda: (time.sleep(0.12), "tile")[1],
        key="tile",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="montage-tile:1:2",
        on_done=done.append,
        on_stale=lambda: stale.append("tile"),
    )
    _wait_for_started(qtbot, controller)
    assert controller._started
    before = controller.group_generation("montage-tile:1:2")

    controller.clear_group("montage-tile")

    assert controller.group_generation("montage-tile:1:2") > before
    _wait_until(qtbot, lambda: stale == ["tile"])
    assert done == []
    assert stale == ["tile"]
    controller.shutdown_for_close()


def test_clear_group_prefix_preserves_unrelated_groups_and_prefetches(qtbot):
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    visible_done = []
    tile_stale = []
    controller.start_latest(
        lambda: (time.sleep(0.12), "tile")[1],
        key="tile",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="montage-tile:1:2",
        on_done=lambda _value: None,
        on_stale=lambda: tile_stale.append("tile"),
    )
    controller.start_latest(
        lambda: "visible",
        key="visible",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=visible_done.append,
    )
    prefetch = controller.start_prefetch(lambda: "prefetch", key=("prefetch", "keep"))

    controller.clear_group("montage-tile")

    assert prefetch.scheduled
    assert ("prefetch", "keep") in controller._prefetch_keys
    assert "visible" in controller._group_request_generations
    assert controller.group_generation("visible") > 0
    controller.shutdown_for_close()


def test_completed_tile_group_bookkeeping_is_pruned(qtbot):
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    done = []
    controller.start_latest(
        lambda: "tile",
        key="tile",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="montage-tile:1:2",
        on_done=done.append,
    )

    _wait_until(qtbot, lambda: done == ["tile"])

    assert done == ["tile"]
    assert "montage-tile:1:2" not in controller._group_request_generations
    assert "montage-tile:1:2" not in controller._group_generations
    assert controller._group_child_groups.get("montage-tile") in (None, set())


def test_visible_pool_max_thread_count_is_one(qtbot):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController(max_workers=1, name="visible")

    assert controller.pool.maxThreadCount() == 1


def test_shutdown_ignores_late_results(qtbot):
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    done = []

    controller.start_latest(
        lambda: (time.sleep(0.08), "late")[1],
        key="late",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=done.append,
    )
    controller.shutdown_for_close()

    _wait_until(qtbot, lambda: controller.pool.activeThreadCount() == 0)

    assert done == []


def test_shutdown_and_late_runnable_notification_do_not_raise(qtbot):
    from arrayscope.core.scheduler import EvalPriority, EvalRequest
    from arrayscope.window.evaluation_controller import CancellationToken, EvaluationController, _EvaluationRunnable

    controller = EvaluationController(max_workers=1)
    controller.shutdown_for_close()

    controller._notify_queue_event()

    token = CancellationToken()
    token.cancel()
    request = EvalRequest(
        key="late-cancel",
        priority=EvalPriority.VISIBLE_IMAGE,
        generation=1,
        replace_group="visible",
        group_generation=1,
    )
    runnable = _EvaluationRunnable(
        request,
        lambda: None,
        controller._queue,
        token,
        notify_queue=lambda: (_ for _ in ()).throw(RuntimeError("Signal source has been deleted")),
    )

    runnable.run()


def test_start_latest_can_pass_cancellation_token(qtbot):
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    seen = []

    controller.start_latest(
        lambda token: seen.append(hasattr(token, "cancelled")) or "ok",
        key="token",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=lambda _value: None,
        pass_token=True,
    )

    _wait_until(qtbot, lambda: seen == [True])

    assert seen == [True]


def test_cancelled_evaluation_does_not_call_error(qtbot):
    from arrayscope.operations.cancellation import EvaluationCancelled
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    errors = []
    stale = []

    controller.start_latest(
        lambda token: (_ for _ in ()).throw(EvaluationCancelled()),
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
    assert stale == [True]


def test_controller_diagnostics_counts_completed_cancelled_stale(qtbot):
    from arrayscope.operations.cancellation import EvaluationCancelled
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    controller.start_latest(
        lambda: "done",
        key="done",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=lambda _value: None,
    )
    _wait_until(qtbot, lambda: controller.diagnostics().completed == 1)
    controller.start_latest(
        lambda token: (_ for _ in ()).throw(EvaluationCancelled()),
        key="cancel",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=lambda _value: None,
        on_stale=lambda: None,
        pass_token=True,
    )
    _wait_until(qtbot, lambda: controller.diagnostics().cancelled == 1 and controller.diagnostics().stale >= 1)

    diagnostics = controller.diagnostics()
    assert diagnostics.completed == 1
    assert diagnostics.cancelled == 1
    assert diagnostics.stale >= 1


def test_prefetch_can_be_cancelled_separately(qtbot):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController(max_workers=1)
    started = controller.start_prefetch(lambda: "prefetch", key="prefetch")

    controller.cancel_prefetch()

    assert started.scheduled
    assert not controller._prefetch_keys


def test_start_prefetch_idle_elapsed_false_blocks_with_idle_reason(qtbot):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController(max_workers=1)
    started = controller.start_prefetch(lambda: "prefetch", key="prefetch", idle_elapsed=False)

    assert not started.scheduled
    assert started.reason == "idle"
    assert controller.diagnostics().prefetch_idle_blocked == 1


def test_start_prefetch_zero_memory_budget_blocks_with_cost_reason(qtbot):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController(max_workers=1)
    started = controller.start_prefetch(lambda: "prefetch", key="prefetch", memory_budget_bytes=0)

    assert not started.scheduled
    assert started.reason == "cost"
    assert controller.diagnostics().prefetch_cost_blocked == 1


def test_prefetch_local_budget_block_does_not_admit_work_graph_item(qtbot):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.scheduler import FrameTarget
    from arrayscope.core.work_graph import WorkGraph, WorkItem, WorkLane
    from arrayscope.window.evaluation_controller import EvaluationController

    parent = QtCore.QObject()
    parent.work_graph = WorkGraph()
    controller = EvaluationController(parent=parent, max_workers=1)
    item = WorkItem(
        key=("prefetch", "blocked"),
        lane=WorkLane.SPECULATIVE_RESIDENCY,
        frame_target=FrameTarget("near", None, "prefetch", "retained"),
        expected_value=1.0,
    )

    started = controller.start_prefetch(
        lambda: "prefetch",
        key="blocked",
        memory_budget_bytes=0,
        work_item=item,
    )

    assert not started.scheduled
    assert started.reason == "cost"
    diagnostics = parent.work_graph.diagnostics()
    assert diagnostics.active == 0
    assert diagnostics.queued == 0
    assert diagnostics.lanes == {}


def test_prefetch_work_graph_admission_yields_to_visible_backlog(qtbot):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.scheduler import FrameTarget
    from arrayscope.core.work_graph import WorkGraph, WorkItem, WorkLane
    from arrayscope.window.evaluation_controller import EvaluationController

    parent = QtCore.QObject()
    parent.work_graph = WorkGraph()
    target = FrameTarget("visible", None, "presentation", "exact-visible")
    assert parent.work_graph.submit(
        WorkItem(
            key="visible",
            lane=WorkLane.VISIBLE_MATERIALIZATION,
            frame_target=target,
        )
    ).admitted
    controller = EvaluationController(parent=parent, max_workers=1)

    started = controller.start_prefetch(
        lambda: "prefetch",
        key="nearby",
        memory_budget_bytes=1,
        work_item=WorkItem(
            key=("prefetch", "nearby"),
            lane=WorkLane.SPECULATIVE_RESIDENCY,
            frame_target=FrameTarget("nearby", None, "prefetch", "retained"),
            expected_value=1.0,
        ),
    )

    assert not started.scheduled
    assert started.reason == "budget"
    diagnostics = parent.work_graph.diagnostics()
    assert diagnostics.active == 1
    assert diagnostics.lanes["speculative_residency"]["blocked_by_budget"] == 1


def test_start_prefetch_no_longer_accepts_idle_deadline_ms(qtbot):
    from arrayscope.window.evaluation_controller import EvaluationController

    signature = inspect.signature(EvaluationController.start_prefetch)

    assert "idle_deadline_ms" not in signature.parameters


def test_is_busy_reflects_pending_or_running_work(qtbot):
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    controller.start_latest(
        lambda: (time.sleep(0.05), "done")[1],
        key="busy",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=lambda _value: None,
    )

    assert controller.is_busy()
    _wait_until(qtbot, lambda: not controller.is_busy())
    assert not controller.is_busy()
