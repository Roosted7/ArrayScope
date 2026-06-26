from arrayscope.core.scheduler import FrameTarget
from arrayscope.core.work_graph import WorkGraph, WorkItem, WorkLane


def _target(name="semantic"):
    return FrameTarget(name, "viewport", "presentation", "exact-visible")


def test_work_graph_admits_visible_before_budget_blocks_optional():
    graph = WorkGraph()
    visible = WorkItem(
        key="visible",
        lane=WorkLane.VISIBLE_MATERIALIZATION,
        frame_target=_target(),
        supersession_key="visible",
        supersession_value="a",
    )
    optional = WorkItem(
        key="prefetch",
        lane=WorkLane.SPECULATIVE_RESIDENCY,
        frame_target=FrameTarget("near", "viewport", "presentation", "retained"),
        expected_value=0.0,
    )

    assert graph.submit(visible).admitted
    assert not graph.submit(optional).admitted

    diagnostics = graph.diagnostics()
    assert diagnostics.lanes["visible_materialization"]["admitted"] == 1
    assert diagnostics.lanes["speculative_residency"]["blocked_by_budget"] == 1


def test_retained_stage_materialization_is_optional_not_visible_backlog():
    graph = WorkGraph()
    visible = WorkItem(
        key="visible",
        lane=WorkLane.VISIBLE_MATERIALIZATION,
        frame_target=_target(),
    )
    warm_stage = WorkItem(
        key="warm-stage",
        lane=WorkLane.STAGE_MATERIALIZATION,
        frame_target=FrameTarget("stage", None, "stage-warmup", "retained"),
        expected_value=1.0,
        reusable_output=True,
    )

    assert graph.submit(visible).admitted
    decision = graph.submit(warm_stage, available_budget=False, visible_backlog=True)

    assert not decision.admitted
    assert decision.reason == "budget"
    diagnostics = graph.diagnostics()
    assert diagnostics.active == 1
    assert diagnostics.lanes["stage_materialization"]["blocked_by_budget"] == 1


def test_work_graph_supersedes_queued_obsolete_value():
    graph = WorkGraph()
    old = WorkItem(
        key="old",
        lane=WorkLane.VISIBLE_MATERIALIZATION,
        frame_target=_target("old"),
        supersession_key="visible",
        supersession_value="old",
        dependency_keys=("missing",),
    )
    new = WorkItem(
        key="new",
        lane=WorkLane.VISIBLE_MATERIALIZATION,
        frame_target=_target("new"),
        supersession_key="visible",
        supersession_value="new",
    )

    assert not graph.submit(old).admitted
    assert graph.submit(new).admitted

    counters = graph.diagnostics().lanes["visible_materialization"]
    assert counters["queued"] == 1
    assert counters["superseded"] == 1
    assert counters["dropped"] == 1
    assert counters["admitted"] == 1


def test_work_graph_supersession_index_only_touches_matching_queue():
    graph = WorkGraph()
    first = WorkItem(
        key="first",
        lane=WorkLane.VISIBLE_MATERIALIZATION,
        frame_target=_target("first"),
        supersession_key="visible",
        supersession_value="old",
        dependency_keys=("missing",),
    )
    second = WorkItem(
        key="second",
        lane=WorkLane.VISIBLE_MATERIALIZATION,
        frame_target=_target("second"),
        supersession_key="visible",
        supersession_value="old",
        dependency_keys=("missing",),
    )
    unrelated = WorkItem(
        key="profile",
        lane=WorkLane.PROFILE_ROI_HOVER,
        frame_target=_target("profile"),
        supersession_key="profile",
        supersession_value="current",
        dependency_keys=("profile-source",),
    )

    assert not graph.submit(first).admitted
    assert not graph.submit(second).admitted
    assert not graph.submit(unrelated).admitted
    assert graph.submit(
        WorkItem(
            key="latest",
            lane=WorkLane.VISIBLE_MATERIALIZATION,
            frame_target=_target("latest"),
            supersession_key="visible",
            supersession_value="latest",
        )
    ).admitted

    assert set(graph._queued) == {"profile"}
    assert "visible" not in graph._queued_by_supersession
    assert graph._queued_by_supersession == {"profile": {"profile"}}
    counters = graph.diagnostics().lanes["visible_materialization"]
    assert counters["superseded"] == 2
    assert counters["dropped"] == 2


def test_work_graph_running_reusable_stale_completion_is_retained_not_presented():
    graph = WorkGraph()
    old = WorkItem(
        key="old",
        lane=WorkLane.VISIBLE_MATERIALIZATION,
        frame_target=_target("old"),
        supersession_key="visible",
        supersession_value="old",
        reusable_output=True,
    )
    new = WorkItem(
        key="new",
        lane=WorkLane.VISIBLE_MATERIALIZATION,
        frame_target=_target("new"),
        supersession_key="visible",
        supersession_value="new",
    )

    assert graph.submit(old).admitted
    assert graph.submit(new).admitted
    graph.complete("old", stale=True, reusable_output=True)
    graph.complete("new")

    counters = graph.diagnostics().lanes["visible_materialization"]
    assert counters["reusable_finished"] == 1
    assert counters["completed"] == 1
    assert counters["dropped"] == 0


def test_work_graph_dependency_block_reschedules_with_reason():
    graph = WorkGraph()
    item = WorkItem(
        key="tile",
        lane=WorkLane.VISIBLE_MATERIALIZATION,
        frame_target=_target(),
        dependency_keys=("stage",),
    )

    decision = graph.submit(item)

    assert not decision.admitted
    assert decision.reason == "dependencies"
    counters = graph.diagnostics().lanes["visible_materialization"]
    assert counters["queued"] == 1
    assert counters["rescheduled"] == 1

    graph.mark_completed("stage")
    decisions = graph.admit_ready()

    assert tuple(decision.item.key for decision in decisions if decision.admitted) == ("tile",)
    counters = graph.diagnostics().lanes["visible_materialization"]
    assert counters["admitted"] == 1


def test_work_graph_stale_queued_item_does_not_become_current_on_readmission():
    graph = WorkGraph()
    old = WorkItem(
        key="tile",
        lane=WorkLane.VISIBLE_MATERIALIZATION,
        frame_target=_target("old"),
        supersession_key="visible",
        supersession_value="old",
        dependency_keys=("stage",),
    )

    assert not graph.submit(old).admitted
    graph._supersession_values["visible"] = "new"
    graph.mark_completed("stage")
    decisions = graph.admit_ready()

    assert tuple(decision.reason for decision in decisions) == ("stale",)
    assert graph._queued == {}
    counters = graph.diagnostics().lanes["visible_materialization"]
    assert counters["dropped"] == 1
    assert counters["admitted"] == 0


def test_work_graph_budget_blocked_queued_item_is_counted_once_per_block_state():
    graph = WorkGraph()
    item = WorkItem(
        key="prefetch",
        lane=WorkLane.SPECULATIVE_RESIDENCY,
        frame_target=FrameTarget("near", "viewport", "presentation", "retained"),
        dependency_keys=("visible",),
        expected_value=1.0,
    )

    assert not graph.submit(item).admitted
    graph.mark_completed("visible")

    first = graph.admit_ready(available_budget=False)
    second = graph.admit_ready(available_budget=False)

    assert tuple(decision.reason for decision in first) == ("budget",)
    assert tuple(decision.reason for decision in second) == ("budget",)
    counters = graph.diagnostics().lanes["speculative_residency"]
    assert counters["blocked_by_budget"] == 1
    assert counters["admitted"] == 0

    admitted = graph.admit_ready(available_budget=True)

    assert tuple(decision.item.key for decision in admitted if decision.admitted) == ("prefetch",)
    counters = graph.diagnostics().lanes["speculative_residency"]
    assert counters["blocked_by_budget"] == 1
    assert counters["admitted"] == 1


def test_work_graph_diagnostics_does_not_create_empty_lane_counters():
    graph = WorkGraph()

    diagnostics = graph.diagnostics()

    assert diagnostics.lanes == {}
    assert graph._counters == {}


def test_work_graph_counts_failed_and_deadline_missed():
    graph = WorkGraph()
    item = WorkItem(
        key="histogram",
        lane=WorkLane.HISTOGRAM_REFINEMENT,
        frame_target=FrameTarget("semantic", "viewport", "histogram", "retained", deadline_ns=1),
        deadline_ns=1,
        expected_value=1.0,
    )

    decision = graph.submit(item, now_ns=2)

    assert not decision.admitted
    counters = graph.diagnostics().lanes["histogram_refinement"]
    assert counters["deadline_missed"] == 1
    assert counters["dropped"] == 1
