from __future__ import annotations

from itertools import permutations

from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity
from arrayscope.presentation import Semantic, TileLifecycle, TilePayloadRef, TilePhase, TileTarget


def _target(tile=0, *, source=10, level=0):
    return TileTarget(tile, source, ("semantic", source), lod_level=level)


def _payload(
    source_id, *, quality="exact", source=10, level=0, kind="complex_rg32f", shader=("shader", 1)
):
    return TilePayloadRef(
        source_id=source_id,
        quality=quality,
        lod_level=level,
        source_index=source,
        texture_kind=kind,
        shader_mapping_key=shader,
        payload=("payload", source_id),
    )


def _typed_identity(
    *,
    level: int,
    quality: str = "exact",
    semantic_generation: object = "semantic",
) -> TileIdentity:
    return TileIdentity(
        document_generation="doc",
        operation_key="operation",
        source_index=10,
        image_axes=(0, 1),
        axis_flips=(False, False),
        channel="complex",
        complex_mapping=("phase_color", "abs", "mapped"),
        texture_kind="complex_rg32f",
        semantic_generation=semantic_generation,
        lod=TileLodIdentity(level=level, factor=1 << level),
        quality=quality,
    )


def _typed_target(level: int) -> TileTarget:
    return TileTarget(
        0,
        10,
        ("semantic", 10),
        lod_level=level,
        identity=_typed_identity(level=level),
    )


def _typed_payload(*, level: int, quality: str) -> TilePayloadRef:
    return TilePayloadRef(
        source_id=("tile", level, quality),
        quality=quality,
        lod_level=level,
        source_index=10,
        texture_kind="complex_rg32f",
        identity=_typed_identity(level=level, quality=quality),
        payload=("payload", level, quality),
    )


def test_fallback_presented_never_settles_exact_target():
    ledger = TileLifecycle()
    ledger.retarget({0: _target(level=0)})
    fallback = _payload(("tile", 0, "preview"), quality="fallback", level=4)

    ledger.fallback_ready(0, fallback)
    assert ledger.presentation_changes()[0].payload_ref == fallback
    ledger.commit_emitted({0: fallback})
    assert ledger.backend_ack({0: fallback}) == (0,)

    row = ledger.row(0)
    assert row.phase is TilePhase.FALLBACK_SHOWN
    assert row.first_pixel_presented
    assert not row.target_settled
    assert not ledger.visible_target_settled()


def test_retained_finer_fallback_settles_coarser_target_without_demotion():
    ledger = TileLifecycle()
    fallback = _typed_payload(level=2, quality="fallback")
    ledger.retarget({0: _typed_target(0)})
    ledger.fallback_ready(0, fallback)
    ledger.commit_emitted({0: fallback})
    assert ledger.backend_ack({0: fallback}) == (0,)
    assert not ledger.row(0).target_settled

    # Equal-level fallback still owes exact work.
    ledger.retarget({0: _typed_target(2)})
    assert not ledger.row(0).target_settled
    assert ledger.target_unsettled_tiles((0,)) == (0,)

    # L2 already exceeds a later L6 demand. Keep it physically presented and
    # settle the coarser target instead of scheduling or presenting a demotion.
    ledger.retarget({0: _typed_target(6)})
    assert ledger.row(0).target_settled
    assert ledger.visible_target_settled()
    assert ledger.target_unsettled_tiles((0,)) == ()
    assert ledger.presentation_changes() == ()


def test_task_cannot_be_running_without_admitted_key_or_stage_key():
    ledger = TileLifecycle()
    ledger.retarget({0: _target()})

    ledger.task_admitted(0, None)

    assert ledger.row(0).task_claim is None
    assert ledger.row(0).phase is TilePhase.NEEDS_FIRST_PIXEL
    assert ledger.snapshot().orphan_running == 0


def test_stage_task_records_producer_key():
    ledger = TileLifecycle()
    ledger.retarget({0: _target()})

    ledger.task_admitted(0, ("tile", 0), stage_key=("stage", 0), stage_producer_key=("stage", 0))

    assert ledger.row(0).phase is TilePhase.TARGET_RUNNING
    assert ledger.snapshot().parked_without_producer == 0
    assert ledger.row(0).stage_producer_key == ("stage", 0)


def test_dropped_task_returns_tile_to_schedulable_state():
    ledger = TileLifecycle()
    ledger.retarget({0: _target()})
    ledger.task_admitted(0, ("task", 0))

    assert ledger.row(0).phase is TilePhase.TARGET_RUNNING

    ledger.task_released(0, reason="stale")

    assert ledger.row(0).task_claim is None
    assert ledger.row(0).phase is TilePhase.NEEDS_FIRST_PIXEL
    assert not ledger.visible_target_settled()


def test_stage_ready_wakes_dependents_once():
    ledger = TileLifecycle()
    ledger.retarget({0: _target(), 1: _target(1, source=11)})
    ledger.stage_waiting(0, ("stage", 1), ("stage-task", 1))
    ledger.stage_waiting(1, ("stage", 1), ("stage-task", 1))

    assert ledger.row(0).phase is TilePhase.TARGET_WAITING_STAGE

    assert ledger.stage_ready(("stage", 1)) == (0, 1)
    assert ledger.stage_ready(("stage", 1)) == ()
    assert ledger.row(0).stage_key is None
    assert ledger.row(1).stage_key is None


def test_finer_exact_payload_supersedes_fallback_but_coarser_exact_does_not():
    ledger = TileLifecycle()
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
    ledger = TileLifecycle()
    ledger.retarget({0: _target(level=0)})
    exact = _payload(("tile", 0, "exact"), level=0)
    wrong_kind = _payload(("tile", 0, "exact"), level=0, kind="rgb8")

    ledger.target_ready(0, exact)
    ledger.commit_emitted({0: exact})

    assert ledger.backend_ack({0: wrong_kind}) == ()
    assert not ledger.visible_target_settled()

    assert ledger.backend_ack({0: exact}) == (0,)
    assert ledger.visible_target_settled()


def test_payload_arrival_order_cannot_downgrade_target_choice():
    fallback = _payload(("tile", 0, "preview"), quality="fallback", level=4)
    target = _payload(("tile", 0, "exact"), level=0)

    for arrival_order in permutations((fallback, target)):
        lifecycle = TileLifecycle()
        lifecycle.retarget({0: _target(level=0)})
        for payload in arrival_order:
            if payload.quality == "fallback":
                lifecycle.fallback_ready(0, payload)
            else:
                lifecycle.target_ready(0, payload)

        command = lifecycle.presentation_changes()[0]
        assert command.payload_ref == target
        lifecycle.commit_emitted({0: target})
        assert lifecycle.backend_ack({0: target}) == (0,)
        assert lifecycle.visible_target_settled()


def test_current_payload_does_not_depend_on_native_renderer_materialization_identity():
    lifecycle = TileLifecycle()
    lifecycle.retarget({0: _target(level=2)})
    reduced_shared_target = _payload(("shared-transform", "target", 0), source=10, level=2)

    lifecycle.target_ready(0, reduced_shared_target)

    assert lifecycle.payload_is_current(0, reduced_shared_target)
    assert not lifecycle.payload_is_current(0, _payload(("stale", 0), source=11, level=2))


def test_current_payload_rejects_a_presentable_wrapper_from_an_old_semantic_generation():
    lifecycle = TileLifecycle()
    old = _typed_payload(level=2, quality="exact")
    lifecycle.retarget({0: _typed_target(level=2)})
    lifecycle.target_ready(0, old)
    assert lifecycle.payload_is_current(0, old)

    lifecycle.retarget(
        {
            0: TileTarget(
                0,
                10,
                ("semantic", 10),
                lod_level=2,
                identity=_typed_identity(
                    level=2,
                    semantic_generation="display-axis-crop-successor",
                ),
            )
        }
    )

    assert not lifecycle.payload_is_current(0, old)


def test_source_retarget_preserves_physical_truth_but_rejects_stale_pixels():
    lifecycle = TileLifecycle()
    exact = _payload(("tile", 0, "exact"), source=10)
    lifecycle.retarget({0: _target(source=10)})
    lifecycle.target_ready(0, exact)
    lifecycle.commit_emitted({0: exact})
    lifecycle.backend_ack({0: exact})

    lifecycle.retarget({0: _target(source=11)})

    row = lifecycle.row(0)
    assert row.backend_source_id == exact.source_id
    assert row.presented_source_id is None
    assert not row.first_pixel_presented
    assert not row.target_settled


def test_source_retarget_supersedes_old_evaluation_claims_but_lod_retarget_does_not():
    lifecycle = TileLifecycle()
    lifecycle.retarget({0: _target(source=10, level=1)})
    lifecycle.load_marked(0)
    lifecycle.evaluation_started(0)
    lifecycle.evaluation_claimed(0, ("source", 10))
    lifecycle.task_admitted(0, ("task", 10))

    lifecycle.retarget({0: _target(source=10, level=3)})

    assert lifecycle.evaluating_tiles == frozenset({0})
    assert lifecycle.loading_tiles == frozenset({0})
    assert lifecycle.active_request_tiles == frozenset({0})

    lifecycle.retarget({0: _target(source=11, level=3)})

    row = lifecycle.row(0)
    assert row.semantic is Semantic.PLANNED
    assert row.task_claim is None
    assert lifecycle.evaluating_tiles == frozenset()
    assert lifecycle.loading_tiles == frozenset()
    assert lifecycle.active_request_tiles == frozenset()


def test_retarget_bounds_presentable_history_to_current_and_physical_predecessor():
    lifecycle = TileLifecycle()
    first = _payload(("tile", 10), source=10)
    lifecycle.retarget({0: _target(source=10)})
    lifecycle.target_ready(0, first)
    lifecycle.commit_emitted({0: first})
    lifecycle.backend_ack({0: first})

    for source in range(11, 40):
        successor = _payload(("tile", source), source=source)
        # Index-window remapping can install a successor immediately before
        # the lifecycle receives its new target.
        lifecycle.remember_presentable(0, successor)
        lifecycle.retarget({0: _target(source=source)})
        lifecycle.target_ready(0, successor)

        row = lifecycle.row(0)
        assert set(row.presentable_payloads) == {first.source_id, successor.source_id}
        assert row.target_payload is not None
        assert row.target_payload.source_id == successor.source_id


def test_active_unpresented_successor_cannot_remove_retained_pixels():
    lifecycle = TileLifecycle()
    lifecycle.retarget({0: _target(source=10)})

    assert lifecycle.row(0).phase is TilePhase.NEEDS_FIRST_PIXEL
    assert not lifecycle.may_remove_visible(0)
    assert lifecycle.may_remove_visible(0, memory_pressure=True)

    lifecycle.retarget({})
    assert lifecycle.may_remove_visible(0)


def test_partial_backend_ack_settles_only_confirmed_region():
    lifecycle = TileLifecycle()
    first = _payload(("tile", 0, "exact"), source=10)
    second = _payload(("tile", 1, "exact"), source=11)
    lifecycle.retarget({0: _target(source=10), 1: _target(1, source=11)})
    lifecycle.target_ready(0, first)
    lifecycle.target_ready(1, second)
    lifecycle.commit_emitted({0: first, 1: second})

    assert lifecycle.backend_ack({0: first}) == (0,)
    assert lifecycle.row(0).target_settled
    assert not lifecycle.row(1).target_settled
    assert not lifecycle.visible_target_settled()


def test_retarget_reports_the_tiles_whose_presentable_history_it_pruned():
    """The pruned set is the contract a payload-report memo has to honour.

    Callers memoize "this payload object was already reported for this slot" to
    keep an index-window scrub from reporting every remapped payload twice.  A
    source retarget prunes ``presentable_payloads``, so the return value has to
    name exactly the records whose memo entry is no longer trustworthy — a
    quieter return would let a caller suppress the report that puts the payload
    back.
    """

    lifecycle = TileLifecycle()
    assert lifecycle.retarget({0: _target(source=10), 1: _target(1, source=20)}) == ()

    # An unchanged target and a pure LOD retarget keep the history intact.
    assert lifecycle.retarget({0: _target(source=10), 1: _target(1, source=20)}) == ()
    assert lifecycle.retarget({0: _target(source=10, level=2)}) == ()

    lifecycle.target_ready(0, _payload(("tile", 10), source=10))
    assert lifecycle.row(0).presentable_payloads

    assert lifecycle.retarget({0: _target(source=11, level=2)}) == (0,)
    assert not lifecycle.row(0).presentable_payloads
