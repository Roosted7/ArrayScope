"""Which stall-probe rows are a defect and which are just bookkeeping.

The probe dumps one row per suspect tile. Reporting a row that is not a
defect trains people to ignore the dump; failing to report one that is turns
a stranded montage into a silent settle — the failure R2 records, where 137
tiles the ledger considered fine never reached the screen and **no stall
fired**.

So the classifier is pinned directly here rather than only through the
interaction rings, which cannot reliably manufacture either state.
"""

from __future__ import annotations

from arrayscope.window.frame_runtime import _stall_tile_probe_row_actionable


def _row(**overrides):
    """A quiet row: presented, in scope, no live work, no identity conflict."""

    row = {
        "visible_first_pixel_complete": True,
        "required": True,
        "target_unsettled": False,
        "loading": False,
        "active": False,
        "dirty": False,
        "pending_upsert": False,
        "evaluation_claim_source_index": None,
        "evaluation_claim_matches_current_source": True,
        "desired_payload_source_index": None,
        "desired_matches_current_source": True,
        "state_payload_source_index": None,
        "state_matches_current_source": True,
        "backend_source": None,
        "backend_matches_desired": True,
        "backend_matches_state": True,
    }
    row.update(overrides)
    return row


def test_a_settled_presented_tile_is_not_reported():
    assert not _stall_tile_probe_row_actionable(_row())


def test_a_released_tile_that_left_the_required_scope_is_not_reported():
    """The noise this classifier was fixed for.

    A resident tile dropped from the current target keeps a row (it is
    visible-but-unpresented) while owing nothing: no request, no payload, no
    presentation, and no place in the round's required set.
    """

    assert not _stall_tile_probe_row_actionable(
        _row(visible_first_pixel_complete=False, required=False, target_unsettled=False)
    )


def test_a_required_tile_with_no_pixels_is_reported():
    assert _stall_tile_probe_row_actionable(
        _row(visible_first_pixel_complete=False, target_unsettled=True)
    )


def test_a_required_tile_the_ledger_calls_settled_but_shows_nothing_is_reported():
    """The gap that keying on ``target_unsettled`` alone would leave open.

    ``target_unsettled`` is the subset of REQUIRED whose ledger record is not
    yet settled. A record can report settled while nothing plan-matching is on
    screen — that divergence is exactly the stranding R2 documents, and it
    must not be the one shape the probe stays quiet about.
    """

    assert _stall_tile_probe_row_actionable(
        _row(visible_first_pixel_complete=False, target_unsettled=False, required=True)
    )


def test_live_work_is_reported_even_when_pixels_are_current():
    for key in ("loading", "active", "dirty", "pending_upsert"):
        assert _stall_tile_probe_row_actionable(_row(**{key: True})), key


def test_identity_conflicts_are_reported_even_when_out_of_scope():
    """A stale identity is a defect wherever it is found."""

    conflicts = (
        {"evaluation_claim_source_index": 3, "evaluation_claim_matches_current_source": False},
        {"desired_payload_source_index": 3, "desired_matches_current_source": False},
        {"state_payload_source_index": 3, "state_matches_current_source": False},
        {
            "desired_payload_source_index": 3,
            "backend_source": "x",
            "backend_matches_desired": False,
        },
        {
            "state_payload_source_index": 3,
            "backend_source": "x",
            "backend_matches_state": False,
        },
    )
    for conflict in conflicts:
        assert _stall_tile_probe_row_actionable(
            _row(required=False, visible_first_pixel_complete=True, **conflict)
        ), conflict
