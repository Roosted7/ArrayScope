"""Exhaustive Qt-free tests for the single-owner tile lifecycle (ADR 0051).

Each structural rule maps to a fixed ADR 0050 defect; the test names say
which regression they pin.
"""

from collections import namedtuple

import pytest

from arrayscope.presentation import (
    ClaimOwner,
    LevelPhase,
    Presentation,
    ReleaseClaim,
    Semantic,
    TileLifecycle,
)


@pytest.fixture()
def lc() -> TileLifecycle:
    return TileLifecycle()


def _evaluated(lc: TileLifecycle, tile: int) -> None:
    lc.plan_applied([tile])
    lc.evaluation_started(tile)
    lc.evaluation_completed(tile)


# -- semantic axis ----------------------------------------------------------


def test_semantic_progression(lc):
    lc.plan_applied([3])
    assert lc.record(3).semantic is Semantic.PLANNED
    lc.evaluation_started(3)
    assert lc.record(3).semantic is Semantic.EVALUATING
    assert 3 in lc.evaluating_tiles
    lc.evaluation_completed(3)
    assert lc.record(3).semantic is Semantic.EVALUATED
    assert 3 not in lc.evaluating_tiles


def test_declined_evaluation_returns_to_planned_not_loading_forever(lc):
    """ADR 0050 defect: tile evals lost on declined admission (28 tiles
    'loading' forever)."""

    lc.plan_applied([7])
    lc.evaluation_started(7)
    lc.evaluation_declined(7)
    assert lc.record(7).semantic is Semantic.PLANNED
    assert 7 not in lc.evaluating_tiles


def test_plan_applied_does_not_regress_states(lc):
    _evaluated(lc, 1)
    lc.plan_applied([1])
    assert lc.record(1).semantic is Semantic.EVALUATED
    lc.evaluation_started(2)
    lc.plan_applied([2])
    assert lc.record(2).semantic is Semantic.EVALUATING


def test_skip(lc):
    lc.plan_applied([4])
    lc.tile_skipped(4)
    assert lc.record(4).semantic is Semantic.SKIPPED


# -- rule 1: acknowledged presentation only ----------------------------------


def test_presented_only_via_acknowledgement(lc):
    _evaluated(lc, 0)
    lc.upsert_emitted(0, source_id=("src", 0, 1))
    assert lc.record(0).presentation is Presentation.EMITTED
    assert 0 not in lc.presented_tiles
    lc.commit_acknowledged(emitted_tiles=[0], accepted_tiles=[0], active_scope=[0])
    rec = lc.record(0)
    assert rec.presentation is Presentation.PRESENTED
    assert rec.presented_source_id == ("src", 0, 1)
    assert 0 in lc.presented_tiles


def test_stale_report_confirms_nothing(lc):
    _evaluated(lc, 0)
    lc.upsert_emitted(0, source_id="s")
    lc.commit_acknowledged(
        emitted_tiles=[0], accepted_tiles=[0], active_scope=[0], stale=True
    )
    assert lc.record(0).presentation is Presentation.EMITTED
    assert 0 not in lc.presented_tiles


def test_fresh_evaluation_invalidates_stale_emit_identity(lc):
    _evaluated(lc, 5)
    lc.upsert_emitted(5, source_id="old")
    lc.evaluation_completed(5)  # replacement result
    rec = lc.record(5)
    assert rec.presentation is Presentation.UNPRESENTED
    assert rec.emitted_source_id is None


def test_feedback_signature_tracks_lifecycle_owned_work_class(lc):
    LevelKey = namedtuple("LevelKey", "component level_xy")
    preview_key = LevelKey("scalar", (4, 4))

    lc.plan_applied([0])
    initial = lc.feedback_signature([0])
    lc.level_claimed(0, preview_key, ClaimOwner.PREVIEW, request=("preview", preview_key))
    lc.level_resident(0, preview_key)
    preview = lc.feedback_signature([0])
    lc.evaluation_started(0)
    lc.evaluation_completed(0)
    lc.upsert_emitted(0, source_id=("exact", 0))
    lc.commit_acknowledged(emitted_tiles=[0], accepted_tiles=[0], active_scope=[0])
    exact_presented = lc.feedback_signature([0])

    assert initial != preview
    assert preview != exact_presented
    assert "preview" in repr(preview)
    assert "presented" in repr(exact_presented)


# -- rule 3: emit-once, park, re-arm ------------------------------------------


def test_declined_out_of_scope_upsert_parks(lc):
    """ADR 0050 defect: ~120 commits+draws/s idle loop from re-emitting
    upserts a viewport-scoped backend never accepts."""

    _evaluated(lc, 9)
    lc.upsert_emitted(9, source_id="s9")
    lc.commit_acknowledged(emitted_tiles=[9], accepted_tiles=[], active_scope=[1, 2])
    rec = lc.record(9)
    assert rec.presentation is Presentation.PARKED
    assert rec.parked_reason == "declined-out-of-scope"
    assert lc.parked_tiles == frozenset({9})


def test_declined_in_scope_upsert_does_not_park(lc):
    _evaluated(lc, 9)
    lc.upsert_emitted(9, source_id="s9")
    lc.commit_acknowledged(emitted_tiles=[9], accepted_tiles=[], active_scope=[9])
    assert lc.record(9).presentation is Presentation.EMITTED
    assert lc.parked_tiles == frozenset()


def test_declined_without_semantic_result_does_not_park(lc):
    """Nothing to re-present: parking would arm an upsert no build can keep."""

    lc.plan_applied([2])
    lc.upsert_emitted(2)  # e.g. floor payload whose tile was since pruned
    lc.commit_acknowledged(emitted_tiles=[2], accepted_tiles=[], active_scope=[])
    assert lc.record(2).presentation is Presentation.UNPRESENTED
    assert lc.parked_tiles == frozenset()


def test_semantic_axis_decides_park_eligibility(lc):
    """P2: the machine's own EVALUATED state is park eligibility — the P1
    ``parkable_tiles`` crutch is gone."""

    lc.upsert_emitted(2)  # semantic axis never saw this tile
    lc.commit_acknowledged(emitted_tiles=[2], accepted_tiles=[], active_scope=[])
    assert 2 not in lc.parked_tiles
    _evaluated(lc, 3)
    lc.upsert_emitted(3)
    lc.commit_acknowledged(emitted_tiles=[3], accepted_tiles=[], active_scope=[])
    assert 3 in lc.parked_tiles


def test_evaluation_dropped_demotes_semantic_axis(lc):
    _evaluated(lc, 7)
    lc.evaluation_dropped(7)
    lc.upsert_emitted(7)
    lc.commit_acknowledged(emitted_tiles=[7], accepted_tiles=[], active_scope=[])
    # No re-presentable result: nothing to park (rule 3).
    assert 7 not in lc.parked_tiles


def test_identity_mismatch_refuses_acknowledgement(lc):
    """Rule 1, ground-truth edition: a slot holding a different identity did
    not present our upsert, whatever the report's tile numbers claim."""

    _evaluated(lc, 4)
    lc.upsert_emitted(4, source_id="new-level-1")
    confirmed = lc.commit_acknowledged(
        emitted_tiles=[4],
        accepted_tiles=[4],
        active_scope=[4],
        presented_identities={4: "old-level-5"},
    )
    assert confirmed == frozenset()
    assert lc.record(4).presentation is not Presentation.PRESENTED
    assert lc.identity_rejections == 1
    # Matching identity confirms.
    confirmed = lc.commit_acknowledged(
        emitted_tiles=[4],
        accepted_tiles=[4],
        active_scope=[4],
        presented_identities={4: "new-level-1"},
    )
    assert confirmed == frozenset({4})
    assert lc.record(4).presentation is Presentation.PRESENTED


def test_identity_gate_falls_back_without_evidence(lc):
    """No identity map, no emitted identity, or slot absent from the map:
    the accepted set decides, as before."""

    _evaluated(lc, 1)
    lc.upsert_emitted(1)  # no source_id recorded
    assert lc.commit_acknowledged(
        emitted_tiles=[1],
        accepted_tiles=[1],
        active_scope=[1],
        presented_identities={1: "anything"},
    ) == frozenset({1})
    _evaluated(lc, 2)
    lc.upsert_emitted(2, source_id="x")
    assert lc.commit_acknowledged(
        emitted_tiles=[2],
        accepted_tiles=[2],
        active_scope=[2],
        presented_identities={9: "unrelated"},
    ) == frozenset({2})


def test_stale_report_confirms_and_parks_nothing(lc):
    _evaluated(lc, 6)
    lc.upsert_emitted(6)
    assert lc.commit_acknowledged(
        emitted_tiles=[6], accepted_tiles=[6], active_scope=[], stale=True
    ) == frozenset()
    assert lc.record(6).presentation is not Presentation.PRESENTED
    assert 6 not in lc.parked_tiles


def test_rearm_on_entering_active_scope(lc):
    _evaluated(lc, 9)
    lc.upsert_emitted(9)
    lc.commit_acknowledged(emitted_tiles=[9], accepted_tiles=[], active_scope=[])
    assert lc.rearm_for_scope([1, 2]) == ()
    assert lc.rearm_for_scope([9, 1]) == (9,)
    assert lc.parked_tiles == frozenset()
    assert lc.record(9).presentation is Presentation.UNPRESENTED
    # emit-once: re-arming is one-shot until parked again
    assert lc.rearm_for_scope([9]) == ()


def test_acceptance_unparks(lc):
    _evaluated(lc, 9)
    lc.upsert_emitted(9)
    lc.commit_acknowledged(emitted_tiles=[9], accepted_tiles=[], active_scope=[])
    lc.upsert_emitted(9, source_id="retry")
    lc.commit_acknowledged(emitted_tiles=[9], accepted_tiles=[9], active_scope=[9])
    assert lc.record(9).presentation is Presentation.PRESENTED
    assert lc.parked_tiles == frozenset()


def test_removal_clears_presentation_and_park(lc):
    _evaluated(lc, 9)
    lc.upsert_emitted(9)
    lc.commit_acknowledged(emitted_tiles=[9], accepted_tiles=[9], active_scope=[9])
    lc.commit_acknowledged(
        emitted_tiles=[], accepted_tiles=[], active_scope=[], removed_tiles=[9]
    )
    rec = lc.record(9)
    assert rec.presentation is Presentation.UNPRESENTED
    assert rec.presented_source_id is None
    assert 9 not in lc.presented_tiles


def test_presentation_confirmed_is_the_resident_retarget_ack(lc):
    """Resident-retarget commits re-show acknowledged payloads without
    upserts; the report's presented set still confirms them (rule 1)."""

    _evaluated(lc, 4)
    lc.presentation_confirmed([4])
    assert lc.record(4).presentation is Presentation.PRESENTED
    assert 4 in lc.presented_tiles
    # A parked tile confirmed presented leaves parked.
    _evaluated(lc, 5)
    lc.upsert_emitted(5)
    lc.commit_acknowledged(emitted_tiles=[5], accepted_tiles=[], active_scope=[])
    assert 5 in lc.parked_tiles
    lc.presentation_confirmed([5])
    assert 5 not in lc.parked_tiles
    assert 5 in lc.presented_tiles


# -- rule 2: claim balancing ---------------------------------------------------


def test_declined_level_releases_claim(lc):
    """ADR 0050 defect: pyramid singleflight claims leaked on blocked
    admission."""

    lc.level_claimed(3, ("lvl", 1), ClaimOwner.CHAIN)
    effects = lc.level_declined(3, ("lvl", 1))
    assert effects == (ReleaseClaim(3, ("lvl", 1), ClaimOwner.CHAIN),)
    assert lc.dangling_claims() == ()


def test_resident_level_never_released_by_decline(lc):
    lc.level_claimed(3, ("lvl", 1), ClaimOwner.WALK)
    lc.level_resident(3, ("lvl", 1))
    assert lc.level_declined(3, ("lvl", 1)) == ()
    assert lc.record(3).levels[("lvl", 1)].phase is LevelPhase.RESIDENT


def test_session_replacement_releases_all_inflight_claims(lc):
    """ADR 0050 defect: walk claims leaked into the shared pyramid on session
    replacement (wedged wrong LOD on scrub-back)."""

    lc.level_claimed(1, ("lvl", 1), ClaimOwner.WALK)
    lc.level_claimed(1, ("lvl", 2), ClaimOwner.CHAIN)
    lc.level_materializing(1, ("lvl", 2))
    lc.level_claimed(2, ("lvl", 1), ClaimOwner.EVALUATION)
    lc.level_resident(2, ("lvl", 1))
    effects = set(lc.session_replaced())
    assert effects == {
        ReleaseClaim(1, ("lvl", 1), ClaimOwner.WALK),
        ReleaseClaim(1, ("lvl", 2), ClaimOwner.CHAIN),
    }
    assert lc.dangling_claims() == ()
    assert lc.record(2).levels[("lvl", 1)].phase is LevelPhase.RESIDENT


def test_evaluation_decline_releases_only_its_own_claims(lc):
    lc.evaluation_started(4)
    lc.level_claimed(4, ("lvl", 0), ClaimOwner.EVALUATION)
    lc.level_claimed(4, ("lvl", 1), ClaimOwner.WALK)
    effects = lc.evaluation_declined(4)
    assert effects == (ReleaseClaim(4, ("lvl", 0), ClaimOwner.EVALUATION),)
    assert lc.dangling_claims() == (ReleaseClaim(4, ("lvl", 1), ClaimOwner.WALK),)


def test_dangling_claims_is_the_rule2_audit(lc):
    lc.level_claimed(1, "a", ClaimOwner.INGEST)
    lc.level_materializing(1, "a")
    lc.level_resident(1, "a")
    assert lc.dangling_claims() == ()
    lc.level_claimed(2, "b", ClaimOwner.PREVIEW)
    assert lc.dangling_claims() == (ReleaseClaim(2, "b", ClaimOwner.PREVIEW),)


def test_materialization_requests_are_claimed_record_view(lc):
    """P3: pending LOD work is derived from residency records, not a side list."""

    request = type(
        "Req",
        (),
        {"tile_number": 8, "key": "lvl2", "chain": (("lvl1", (2, 2)), ("lvl2", (2, 2)))},
    )()

    lc.materialization_planned(8, request, owner=ClaimOwner.CHAIN)
    assert lc.pending_materializations() == (request,)
    assert set(lc.dangling_claims()) == {
        ReleaseClaim(8, "lvl1", ClaimOwner.CHAIN),
        ReleaseClaim(8, "lvl2", ClaimOwner.CHAIN),
    }

    lc.materialization_started(request)
    assert lc.pending_materializations() == ()
    assert len(lc.dangling_claims()) == 2

    lc.materialization_resident(request)
    assert lc.dangling_claims() == ()


# -- lifecycle end-to-end -------------------------------------------------------


def test_full_tile_story(lc):
    """plan → evaluate → emit → ack → scroll away (park via replan decline) →
    scroll back (re-arm) → re-emit → ack."""

    lc.plan_applied([0, 1])
    lc.evaluation_started(0)
    lc.evaluation_completed(0)
    lc.upsert_emitted(0, source_id="v1")
    lc.commit_acknowledged(emitted_tiles=[0], accepted_tiles=[0], active_scope=[0, 1])
    assert lc.presented_tiles == frozenset({0})

    # viewport moves: level swap emits a new payload, backend declines (tile
    # now outside active scope)
    lc.upsert_emitted(0, source_id="v2")
    lc.commit_acknowledged(emitted_tiles=[0], accepted_tiles=[], active_scope=[1])
    assert lc.parked_tiles == frozenset({0})

    # viewport returns
    assert lc.rearm_for_scope([0, 1]) == (0,)
    lc.upsert_emitted(0, source_id="v2")
    lc.commit_acknowledged(emitted_tiles=[0], accepted_tiles=[0], active_scope=[0, 1])
    rec = lc.record(0)
    assert rec.presentation is Presentation.PRESENTED
    assert rec.presented_source_id == "v2"
    assert lc.counters() == {
        "records": 2,
        "evaluating": 0,
        "parked": 0,
        "presented": 1,
        "loading": 0,
        "active_requests": 0,
        "skipped": 0,
        "stage_blocked": 0,
        "dangling_claims": 0,
        "identity_rejections": 0,
    }


# -- sets-as-views (P2): load intent, requests, skip, stage bindings ----------


def test_confirmed_evaluated_tile_leaves_loading(lc):
    """The 2026-07-05 auto-levels wedge: presented==rendered with the legacy
    loading set stuck at tile_count.  A confirmed EVALUATED tile must leave
    the loading view mechanically."""

    lc.plan_applied([0])
    lc.load_marked(0)
    lc.evaluation_started(0)
    lc.evaluation_completed(0)
    assert lc.loading_tiles == frozenset({0})
    lc.upsert_emitted(0, source_id="v1")
    lc.commit_acknowledged(emitted_tiles=[0], accepted_tiles=[0], active_scope=[0])
    assert lc.loading_tiles == frozenset()


def test_preview_confirmation_keeps_evaluating_tile_loading(lc):
    """A floor/preview acceptance while the exact result is still computing
    must NOT clear the load intent — exact content is still owed."""

    lc.plan_applied([3])
    lc.load_marked(3)
    lc.evaluation_started(3)
    lc.upsert_emitted(3, source_id="floor")
    lc.commit_acknowledged(emitted_tiles=[3], accepted_tiles=[3], active_scope=[3])
    assert lc.loading_tiles == frozenset({3})


def test_parked_tile_is_not_loading(lc):
    lc.plan_applied([5])
    lc.load_marked(5)
    lc.evaluation_started(5)
    lc.evaluation_completed(5)
    lc.upsert_emitted(5, source_id="v1")
    lc.commit_acknowledged(emitted_tiles=[5], accepted_tiles=[], active_scope=[9])
    assert lc.parked_tiles == frozenset({5})
    assert lc.loading_tiles == frozenset()


def test_skip_clears_loading_and_requests(lc):
    lc.load_marked(2)
    lc.evaluation_requested(2)
    lc.tile_skipped(2)
    assert lc.skipped_tiles == frozenset({2})
    assert lc.loading_tiles == frozenset()
    assert lc.active_request_tiles == frozenset()
    lc.tile_unskipped(2)
    assert lc.skipped_tiles == frozenset()
    from arrayscope.presentation import Semantic as _Semantic

    assert lc.record(2).semantic is _Semantic.PLANNED


def test_request_views_follow_semantic_events(lc):
    lc.evaluation_started(7)
    lc.evaluation_requested(7)
    assert lc.active_request_tiles == frozenset({7})
    lc.evaluation_completed(7)
    assert lc.active_request_tiles == frozenset()
    lc.evaluation_started(8)
    lc.evaluation_requested(8)
    lc.evaluation_declined(8)
    assert lc.active_request_tiles == frozenset()


def test_stage_bindings_reconcile(lc):
    lc.stage_bindings_replaced({"stage-a": (1, 2), "stage-b": (3,)})
    assert lc.stage_blocked_tiles == frozenset({1, 2, 3})
    assert lc.record(2).stage_key == "stage-a"
    # Activation batch consumed tile 2; fail dropped stage-b entirely.
    lc.stage_bindings_replaced({"stage-a": (1,)})
    assert lc.stage_blocked_tiles == frozenset({1})
    assert lc.record(2).stage_key is None
    assert lc.record(3).stage_key is None
    # A completed evaluation unbinds mechanically.
    lc.evaluation_started(1)
    lc.evaluation_completed(1)
    assert lc.stage_blocked_tiles == frozenset()
