"""Machine-derived dispatch derivation (ADR 0051 P2).

Records in PLANNED/dirty imply scheduled work: the derivation must map
every unsettled session state to at least one pump, and a fully settled
session to none.
"""

from dataclasses import dataclass, field

from arrayscope.presentation.dispatch import derive_montage_dispatch


@dataclass
class _StageFanIn:
    active_requests: set = field(default_factory=set)
    attached_requests: set = field(default_factory=set)
    waiting_tiles: dict = field(default_factory=dict)


@dataclass
class _Session:
    pending_tiles: list = field(default_factory=list)
    active_tile_requests: set = field(default_factory=set)
    loading_tiles: set = field(default_factory=set)
    pending_completed_tiles: list = field(default_factory=list)
    stage_fan_in: _StageFanIn = field(default_factory=_StageFanIn)
    dirty_payloads: dict = field(default_factory=dict)
    dirty_tiles: list = field(default_factory=list)
    pending_payload_upserts: dict = field(default_factory=dict)
    pending_removals: set = field(default_factory=set)
    pending_lod_requests: list = field(default_factory=list)
    stage_planning_deferred: bool = False
    flush_pending: bool = False
    final_commit_pending: bool = False


def test_settled_session_derives_no_work():
    plan = derive_montage_dispatch(_Session())

    assert not plan.any_work
    assert not plan.unsettled


def test_pending_tiles_imply_tile_scheduling():
    plan = derive_montage_dispatch(_Session(pending_tiles=[object()]))

    assert plan.schedule_tiles
    assert plan.unsettled


def test_orphaned_loading_tiles_imply_requeue_and_scheduling():
    # The 2026-07-05 dead-pump field freeze: loading records with nothing
    # in flight anywhere.  The records themselves imply work.
    plan = derive_montage_dispatch(_Session(loading_tiles={3, 4}))

    assert plan.requeue_orphans
    assert plan.schedule_tiles
    assert plan.unsettled


def test_loading_covered_by_active_work_is_not_orphaned():
    plan = derive_montage_dispatch(
        _Session(loading_tiles={3}, active_tile_requests={3})
    )

    assert not plan.requeue_orphans


def test_loading_covered_by_completed_fan_in_is_not_orphaned():
    plan = derive_montage_dispatch(
        _Session(loading_tiles={3}, pending_completed_tiles=[object()])
    )

    assert not plan.requeue_orphans
    assert plan.flush_results


def test_orphans_are_requeued_even_alongside_pending_tiles():
    # The 2026-07-05 zoom-back stall: 50 pending AND 50 orphaned-loading with
    # nothing in flight.  Pending work must not mask the orphan records.
    plan = derive_montage_dispatch(
        _Session(loading_tiles={3}, pending_tiles=[object()])
    )

    assert plan.requeue_orphans
    assert plan.schedule_tiles


def test_stage_waiting_tiles_are_never_requeued_as_orphans():
    # Stage-waiting tiles are `loading` with no active evaluation request by
    # design; requeueing them would bypass the attached stage.
    for fan_in in (
        _StageFanIn(waiting_tiles={"k": [object()]}),
        _StageFanIn(attached_requests={"k"}),
        _StageFanIn(active_requests={"k"}),
    ):
        plan = derive_montage_dispatch(_Session(loading_tiles={3}, stage_fan_in=fan_in))
        assert not plan.requeue_orphans


def test_deferred_planning_owns_unattached_loading_tiles():
    plan = derive_montage_dispatch(
        _Session(loading_tiles={3}, stage_planning_deferred=True)
    )

    assert not plan.requeue_orphans
    assert plan.deferred_planning
    assert plan.unsettled


def test_dirty_presentation_state_implies_commit():
    for kwargs in (
        {"dirty_payloads": {1: None}},
        {"dirty_tiles": [1]},
        {"pending_payload_upserts": {1: object()}},
        {"pending_removals": {1}},
        {"flush_pending": True},
        {"final_commit_pending": True},
    ):
        plan = derive_montage_dispatch(_Session(**kwargs))
        assert plan.commit, kwargs
        assert plan.unsettled, kwargs


def test_commit_is_forced_only_when_evaluation_drained():
    drained = derive_montage_dispatch(_Session(dirty_payloads={1: None}))
    busy = derive_montage_dispatch(
        _Session(dirty_payloads={1: None}, active_tile_requests={2})
    )

    assert drained.force_commit
    assert not busy.force_commit
    assert busy.commit


def test_pending_lod_requests_imply_materialization_pump():
    plan = derive_montage_dispatch(_Session(pending_lod_requests=[object()]))

    assert plan.lod_materializations
    assert plan.unsettled


def test_stage_records_imply_stage_pumps():
    waiting = derive_montage_dispatch(
        _Session(stage_fan_in=_StageFanIn(waiting_tiles={"k": [object()]}))
    )
    attached = derive_montage_dispatch(
        _Session(stage_fan_in=_StageFanIn(attached_requests={"k"}))
    )

    assert waiting.stage_waits and waiting.unsettled
    assert attached.stage_waits and attached.unsettled


def test_every_single_record_kind_marks_unsettled():
    cases = (
        _Session(pending_tiles=[object()]),
        _Session(active_tile_requests={1}),
        _Session(loading_tiles={1}),
        _Session(pending_completed_tiles=[object()]),
        _Session(stage_fan_in=_StageFanIn(active_requests={"k"})),
        _Session(stage_fan_in=_StageFanIn(attached_requests={"k"})),
        _Session(stage_fan_in=_StageFanIn(waiting_tiles={"k": []})),
        _Session(dirty_payloads={1: None}),
        _Session(pending_payload_upserts={1: object()}),
        _Session(pending_removals={1}),
        _Session(pending_lod_requests=[object()]),
        _Session(stage_planning_deferred=True),
        _Session(flush_pending=True),
        _Session(final_commit_pending=True),
    )
    for index, session in enumerate(cases):
        plan = derive_montage_dispatch(session)
        assert plan.unsettled, index
        # Any unsettled record must arm at least one pump or the watchdog
        # alone (active/loading-covered work legitimately waits on
        # completions, which re-derive).
        covered_by_inflight = bool(
            session.active_tile_requests or session.stage_fan_in.active_requests
        )
        assert plan.any_work or covered_by_inflight, index
