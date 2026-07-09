from __future__ import annotations

from arrayscope.presentation import TileLedger, TileLedgerPhase, TilePayloadRef, TileTarget


def _target(tile=0, *, source=10, level=0):
    return TileTarget(tile, source, ("semantic", source), lod_level=level)


def _payload(source_id, *, quality="exact", source=10, level=0, kind="complex_rg32f", shader=("shader", 1)):
    return TilePayloadRef(
        source_id=source_id,
        quality=quality,
        lod_level=level,
        source_index=source,
        texture_kind=kind,
        shader_mapping_key=shader,
        payload=("payload", source_id),
    )


def test_fallback_presented_never_settles_exact_target():
    ledger = TileLedger()
    ledger.retarget({0: _target(level=0)})
    fallback = _payload(("tile", 0, "preview"), quality="fallback", level=4)

    ledger.fallback_ready(0, fallback)
    assert ledger.presentation_changes()[0].payload_ref == fallback
    ledger.commit_emitted({0: fallback})
    assert ledger.backend_ack({0: fallback}) == (0,)

    row = ledger.row(0)
    assert row.phase is TileLedgerPhase.FALLBACK_SHOWN
    assert row.first_pixel_presented
    assert not row.target_settled
    assert not ledger.visible_target_settled()


def test_task_cannot_be_running_without_admitted_key_or_stage_key():
    ledger = TileLedger()
    ledger.retarget({0: _target()})

    ledger.task_admitted(0, None)

    assert ledger.row(0).task_claim is None
    assert ledger.row(0).phase is TileLedgerPhase.NEEDS_FIRST_PIXEL
    assert ledger.snapshot().orphan_running == 0


def test_stage_task_records_producer_key():
    ledger = TileLedger()
    ledger.retarget({0: _target()})

    ledger.task_admitted(0, ("tile", 0), stage_key=("stage", 0), stage_producer_key=("stage", 0))

    assert ledger.row(0).phase is TileLedgerPhase.TARGET_RUNNING
    assert ledger.snapshot().parked_without_producer == 0
    assert ledger.row(0).stage_producer_key == ("stage", 0)


def test_dropped_task_returns_tile_to_schedulable_state():
    ledger = TileLedger()
    ledger.retarget({0: _target()})
    ledger.task_admitted(0, ("task", 0))

    assert ledger.row(0).phase is TileLedgerPhase.TARGET_RUNNING

    ledger.task_released(0, reason="stale")

    assert ledger.row(0).task_claim is None
    assert ledger.row(0).phase is TileLedgerPhase.NEEDS_FIRST_PIXEL
    assert not ledger.visible_target_settled()


def test_stage_ready_wakes_dependents_once():
    ledger = TileLedger()
    ledger.retarget({0: _target(), 1: _target(1, source=11)})
    ledger.stage_waiting(0, ("stage", 1), ("stage-task", 1))
    ledger.stage_waiting(1, ("stage", 1), ("stage-task", 1))

    assert ledger.row(0).phase is TileLedgerPhase.TARGET_WAITING_STAGE

    assert ledger.stage_ready(("stage", 1)) == (0, 1)
    assert ledger.stage_ready(("stage", 1)) == ()
    assert ledger.row(0).stage_key is None
    assert ledger.row(1).stage_key is None


def test_finer_exact_payload_supersedes_fallback_but_coarser_exact_does_not():
    ledger = TileLedger()
    ledger.retarget({0: _target(level=2)})
    fallback = _payload(("tile", 0, "preview"), quality="fallback", level=4)
    coarser = _payload(("tile", 0, "coarse"), quality="exact", level=3)
    finer = _payload(("tile", 0, "fine"), quality="exact", level=1)

    ledger.fallback_ready(0, fallback)
    ledger.target_ready(0, coarser)
    assert ledger.row(0).target_payload is None

    ledger.target_ready(0, finer)
    assert ledger.presentation_changes()[0].payload_ref == finer


def test_backend_ack_rejects_wrong_payload_metadata():
    ledger = TileLedger()
    ledger.retarget({0: _target(level=0)})
    exact = _payload(("tile", 0, "exact"), level=0)
    wrong_kind = _payload(("tile", 0, "exact"), level=0, kind="rgb8")

    ledger.target_ready(0, exact)
    ledger.commit_emitted({0: exact})

    assert ledger.backend_ack({0: wrong_kind}) == ()
    assert not ledger.visible_target_settled()

    assert ledger.backend_ack({0: exact}) == (0,)
    assert ledger.visible_target_settled()
