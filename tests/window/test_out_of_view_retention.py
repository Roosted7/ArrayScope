"""Out-of-view presentation retention: predicate, ceiling and eviction order.

The live pan behaviour is pinned in ``tests/ui/test_montage_out_of_view_retention.py``
against a real WGPU montage.  These tests own the bookkeeping the live path
cannot make deterministic: which payload is retainable, how much retention may
hold, and which tile it gives up first.

Payloads here carry a real :class:`TileIdentity` minted from the plan tile.
That is deliberate — an identity-less payload skips both the backend's binding
reuse and this module's own staleness guard, so a fixture without one would be
testing a path production never takes.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.display.lod import LodInfo
from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationState
from arrayscope.display.model.tile_identity import TileLodIdentity, tile_ack_identity
from arrayscope.display.montage import make_montage_plan
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.window import frame_session
from arrayscope.window.frame_session import FrameSession

TILE_SHAPE = (4, 4)
SOURCES = (0, 1, 2, 3, 4, 5, 6, 7)


def _session(*, retain: bool = True, budget_bytes: int = 64 * 1024 * 1024) -> FrameSession:
    state = ViewState.from_shape((4, 4, len(SOURCES))).with_montage_axis(
        2, indices=SOURCES, text=":"
    )
    plan = make_montage_plan(state, axis=2, indices=SOURCES, tile_shape=TILE_SHAPE, columns=4)
    return FrameSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=SOURCES,
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(16, 16),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        shader_display=True,
        retain_out_of_view_payloads=retain,
        tile_residency_budget_bytes=budget_bytes,
    )


def _payload(session: FrameSession, tile_number: int, *, pixels=None) -> DisplayTilePayload:
    tile = session.plan.tiles[int(tile_number)]
    image = (
        np.full(TILE_SHAPE, float(tile.source_index), dtype=np.float32)
        if pixels is None
        else np.asarray(pixels, dtype=np.float32)
    )
    identity = session._tile_identity(
        tile,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        complex_mapping=None,
        lod=TileLodIdentity(),
        quality="exact",
    )
    return DisplayTilePayload(
        tile_number=int(tile_number),
        source_index=int(tile.source_index),
        image=image,
        histogram_data=image,
        source_id=("tile", int(tile.source_index), "exact"),
        lod=LodInfo(level=0, factor=1, source_shape=TILE_SHAPE, texture_shape=TILE_SHAPE),
        tile_identity=identity,
    )


def _present(session: FrameSession, *tile_numbers: int) -> None:
    """Put payloads on screen the way an acknowledged commit does."""

    session.sync_lifecycle_scope()
    state_payloads = dict(session.tile_presentation_state.payloads)
    for tile_number in tile_numbers:
        index = int(tile_number)
        payload = _payload(session, index)
        session.display_tile_payloads[index] = payload
        session.record_tile_payload(payload)
        session.lifecycle.commit_emitted({index: payload})
        session.lifecycle.backend_ack({index: payload})
        session.lifecycle.acknowledge_presented(index, tile_ack_identity(payload), "exact", 0)
        state_payloads[index] = payload
    session.tile_presentation_state = TilePresentationState(state_payloads)


def _scope(session: FrameSession, active) -> tuple[int, ...]:
    plan_tiles = {int(tile.montage_index): tile for tile in session.plan.tiles}
    return session.retain_out_of_view_presentation_scope(
        tuple(int(tile) for tile in active),
        plan_tiles_by_number=plan_tiles,
        backend_identities=dict(session.lifecycle.backend_presented_identities),
    )


def _look_at(session: FrameSession, tile_numbers) -> None:
    wanted = {int(tile) for tile in tile_numbers}
    session.visible_tiles = tuple(
        tile for tile in session.plan.tiles if int(tile.montage_index) in wanted
    )
    session.visible_tile_numbers = frozenset(wanted)
    session.sync_lifecycle_scope()


def test_out_of_view_tiles_stay_in_the_presentation_scope():
    session = _session()
    _present(session, *range(8))

    scope = _scope(session, (0, 1))

    # The camera moved off six tiles; nothing about their pixels changed, so
    # they keep the slot rather than being republished on the way back.
    assert set(scope) == set(range(8))
    assert sorted(session.retained_out_of_view) == [2, 3, 4, 5, 6, 7]
    assert session.retained_out_of_view_bytes > 0


def test_retention_is_off_when_the_backend_does_not_ask_for_it():
    session = _session(retain=False)
    _present(session, *range(8))

    assert _scope(session, (0, 1)) == (0, 1)
    assert not session.retained_out_of_view


def test_a_returning_tile_leaves_the_retention_set():
    session = _session()
    _present(session, *range(8))
    _scope(session, (0, 1))
    assert 4 in session.retained_out_of_view

    scope = _scope(session, (4, 5))

    assert 4 not in session.retained_out_of_view
    assert set(scope) == set(range(8))
    assert session.retained_out_of_view_returns == 2


def test_retention_keeps_a_slot_whose_session_payload_moved_on_out_of_view():
    """A better payload arriving out of view must not un-draw the tile.

    This test asserted the opposite until 2026-08-03, on the reading that a
    backend drawing something other than the payload the session holds is "a
    repair obligation".  That reading conflated two questions.  Whether the
    pixels on screen are *stale* is a question about the payload the BACKEND IS
    DRAWING; whether the session has since acquired a *better* one is a
    question about quality, and quality is not a correctness obligation for a
    tile nobody is looking at.

    An LOD refinement to an out-of-view tile is exactly that divergence, and it
    is routine during a zoom.  Refusing retention for it dropped the tile out of
    ``active_tiles`` — it stopped being drawn — and the resulting scope shrink
    broke the WGPU executor's ``previous_commit["tiles"] <= payloads`` fast path
    for a ~150 ms full republication.  Measured in the field trace: 23
    republications correlating 23/23 with the commits over 100 ms.
    """

    session = _session()
    _present(session, *range(8))
    # A better wrapper lands for tile 5 while the backend still draws the old
    # one.  Nothing about the drawn pixels changed.
    session.display_tile_payloads[5] = _payload(session, 5, pixels=np.zeros(TILE_SHAPE))
    session.display_tile_payloads[5] = dataclasses.replace(
        session.display_tile_payloads[5],
        source_id=("tile", 5, "successor"),
        tile_identity=dataclasses.replace(
            session.display_tile_payloads[5].tile_identity, quality="fallback"
        ),
    )

    scope = _scope(session, (0, 1))

    assert 5 in session.retained_out_of_view
    assert 5 in scope
    # And nothing was refused for it: this is the path, not a lucky fallback
    # through some other clause.
    assert session.retained_out_of_view_refusals == {}


def test_retention_refuses_a_slot_whose_drawn_payload_it_cannot_identify():
    """No evidence about the pixels on screen is a refusal, not a default.

    The retention decision is made about the payload the backend is drawing, so
    a drawn identity that resolves to nothing the lifecycle holds leaves the
    session unable to prove those pixels are safe.  In production that state
    means the slot's source moved on — ``TileLifecycle.retarget`` prunes
    ``presentable_payloads`` on a source change — which is precisely when
    retention must let go.
    """

    session = _session()
    _present(session, *range(8))
    session.lifecycle.record(5).presentable_payloads.clear()

    scope = _scope(session, (0, 1))

    assert 5 not in session.retained_out_of_view
    assert 5 not in scope
    assert session.retained_out_of_view_refusals == {"drawn_payload_unknown": 1}
    # Positive control: retention is on, and refuses this tile specifically.
    assert 4 in session.retained_out_of_view


def test_retention_refuses_a_payload_whose_semantic_generation_moved_on():
    session = _session()
    _present(session, *range(8))
    # The displayed slice moved: the retained wrapper's pixels describe the
    # previous window.  ``payload_compatible_with_tile`` is the existing guard
    # against showing them, and retention must not route around it.
    stale = session.display_tile_payloads[6]
    session.display_tile_payloads[6] = dataclasses.replace(
        stale,
        tile_identity=dataclasses.replace(stale.tile_identity, semantic_generation=("moved", "on")),
    )
    session.lifecycle.backend_ack({6: session.display_tile_payloads[6]})
    # A backend-acknowledged payload is, in production, always one the
    # lifecycle also holds as presentable.  Saying so here is what makes the
    # fixture reach the compatibility guard this test is named for: without it
    # the drawn identity resolves to nothing and the refusal comes one clause
    # earlier, for a reason this test is not about.
    session.lifecycle.remember_presentable(6, session.display_tile_payloads[6])

    scope = _scope(session, (0, 1))

    assert 6 not in session.retained_out_of_view
    assert 6 not in scope
    assert session.retained_out_of_view_refusals == {"incompatible": 1}
    assert 4 in session.retained_out_of_view


def test_retention_is_bounded_and_gives_up_the_least_recently_visible_tile():
    session = _session(budget_bytes=0)
    _present(session, *range(8))
    per_tile = session.display_tile_payloads[0].nbytes
    # Room for exactly three retained tiles.
    session.tile_residency_budget_bytes = int(per_tile * 3 / 0.5)

    # Tiles 6 and 7 leave first, then 4 and 5, then 2 and 3: six candidates
    # for three slots.
    _scope(session, (0, 1, 2, 3, 4, 5))
    _scope(session, (0, 1, 2, 3))
    scope = _scope(session, (0, 1))

    assert sorted(session.retained_out_of_view) == [2, 3, 5]
    assert session.retained_out_of_view_evictions == 3
    assert set(scope) == {0, 1, 2, 3, 5}
    # An evicted tile is exactly where it would have been without retention:
    # out of the presentation scope and queued for physical removal.
    assert {4, 6, 7}.issubset(session.pending_removals)
    assert 4 not in session.display_tile_payloads
    assert session.retained_out_of_view_bytes <= session.retained_out_of_view_budget_bytes


def test_retention_is_bounded_by_a_tile_count_as_well_as_by_bytes(monkeypatch):
    """Never removing on pan still has to be bounded by something.

    The byte ceiling does not bound the presentation scope: a retained slot
    costs one entry in every per-commit walk over ``active_tiles`` (the GPU
    instance rebuild is the O(|active|) one), and coarse LOD payloads are
    cheapest in bytes exactly when there are most of them.  So there is a count
    ceiling too, evicting through the same oldest-departure-first path.
    """

    monkeypatch.setattr(frame_session, "RETAINED_OUT_OF_VIEW_MAX_TILES", 3)
    session = _session()
    _present(session, *range(8))

    scope = _scope(session, (0, 1))

    # Six candidates, three slots: the three that left longest ago give way.
    assert sorted(session.retained_out_of_view) == [5, 6, 7]
    assert session.retained_out_of_view_evictions == 3
    assert set(scope) == {0, 1, 5, 6, 7}
    assert {2, 3, 4}.issubset(session.pending_removals)
    assert session.retained_out_of_view_bytes <= session.retained_out_of_view_budget_bytes


def test_retention_is_disabled_without_a_residency_budget():
    session = _session(budget_bytes=0)
    _present(session, *range(8))

    scope = _scope(session, (0, 1))

    assert scope == (0, 1)
    assert not session.retained_out_of_view
    assert session.retained_out_of_view_budget_bytes == 0


def _supersede_out_of_view(session, tile_number: int):
    """Land a better wrapper for a tile the camera is not looking at."""

    superseded = session.display_tile_payloads[int(tile_number)]
    session.display_tile_payloads[int(tile_number)] = dataclasses.replace(
        superseded,
        source_id=("tile", int(tile_number), "successor"),
        tile_identity=dataclasses.replace(superseded.tile_identity, quality="fallback"),
    )
    return superseded


def test_a_retained_tile_survives_the_next_build_unchanged():
    """The identity-repair pass must not undo retention one commit later.

    ``build_tile_presentation`` discards a drawn slot whose identity differs
    from the current payload whenever ``tile not in planned_numbers or
    may_remove_visible(tile)`` — and an out-of-view retained tile satisfies both
    halves of that disjunction (``may_remove_visible`` is unconditionally true
    for a record whose target is not visible, so no ``and`` clause can shield
    anything: only the pass's early continue can).  Get this wrong and every
    retained tile is removed again by the following commit: the behaviour is
    unchanged, the counters are unchanged, and the measurement reads flat for a
    reason nothing reports.

    Run twice over: once for a tile the session and the backend agree about,
    and once for a tile whose session payload has moved on — the second is the
    case the shield actually has to carry, because the first takes the pass's
    identity-match continue on its own.
    """

    session = _session()
    _present(session, *range(8))
    _look_at(session, (0, 1))

    first_state, first_delta = session.build_tile_presentation({})
    assert 5 in session.retained_out_of_view
    assert 5 in first_delta.active_tiles
    assert 5 not in first_delta.removals

    # Tile 5's payload moves on while the camera is elsewhere.
    _supersede_out_of_view(session, 5)

    session.tile_presentation_state = first_state
    second_state, second_delta = session.build_tile_presentation({})

    assert 5 in session.retained_out_of_view
    assert 5 in second_delta.active_tiles
    assert 5 not in second_delta.removals
    assert 5 not in session.pending_removals
    assert 5 in second_state.payloads
    # And the retained tile earned no work: retention is a presentation-scope
    # decision, never an upsert.
    assert not set(second_delta.upserts) & set(session.retained_out_of_view)

    session.tile_presentation_state = second_state
    third_state, third_delta = session.build_tile_presentation({})

    assert 5 in third_delta.active_tiles
    assert 5 not in third_delta.removals
    assert 5 not in session.pending_removals
    assert not set(third_delta.upserts) & set(session.retained_out_of_view)
    # Stable, not merely correct once: the slot keeps drawing the payload the
    # backend acknowledged, and the successor stays parked rather than becoming
    # commit debt no build outside the visible set can discharge.
    assert third_state.payloads[5] is second_state.payloads[5]
    assert session.out_of_view_upgrade_pending == {5}
    assert 5 not in session.dirty_payloads
    assert 5 not in session.pending_payload_upserts


def test_a_retained_tile_whose_payload_moves_on_keeps_drawing_what_it_has():
    """Camera motion cannot make a tile's pixels wrong; a content change can.

    This test asserted the exact opposite until 2026-08-03 — that the tile
    "steps out of the scope" — on the reasoning that retention and the removal
    path must stay mutually exclusive, and that the identity clause was the
    thing keeping them apart.  The exclusivity requirement is real and is now
    met explicitly, by an early continue in the identity-repair pass.  The
    expectation built on top of it was wrong: it made a *better* payload
    arriving for an out-of-view tile un-draw that tile, which is the vanishing
    tiles the field report describes and, through the executor's lost
    ``previous_commit["tiles"] <= payloads`` precondition, the commit p95 too.
    """

    session = _session()
    _present(session, *range(8))
    _look_at(session, (0, 1))
    _state, _delta = session.build_tile_presentation({})
    assert 5 in session.retained_out_of_view

    superseded = _supersede_out_of_view(session, 5)

    state, delta = session.build_tile_presentation({})

    assert 5 in session.retained_out_of_view
    assert 5 in delta.active_tiles
    assert 5 not in delta.removals
    assert 5 not in delta.upserts
    assert not set(delta.active_tiles) & set(delta.removals)
    # The slot still draws the acknowledged payload, and the successor is
    # parked for the drain rather than discarded: the upgrade is deferred, not
    # cancelled, and it costs no recomputation when it runs.
    assert state.payloads[5] is superseded
    assert session.out_of_view_upgrade_pending == {5}
    assert session.lifecycle.peek(5).presentable_payloads[("tile", 5, "successor")] is not None


def test_a_parked_out_of_view_upgrade_drains_nearest_first_and_re_arms():
    session = _session()
    _present(session, *range(8))
    _look_at(session, (0, 1))
    session.build_tile_presentation({})
    for tile in (4, 5, 6, 7):
        _supersede_out_of_view(session, tile)
    session.build_tile_presentation({})
    assert session.out_of_view_upgrade_pending == {4, 5, 6, 7}
    assert session.out_of_view_upgrades_parked == 4

    drained = session.drain_out_of_view_upgrade(2)

    assert len(drained) == 2
    assert set(drained) <= {4, 5, 6, 7}
    assert set(session.out_of_view_upgrade_pending) == {4, 5, 6, 7} - set(drained)
    for tile in drained:
        assert tile in session.dirty_payloads
        assert tile in session.pending_payload_upserts


def test_a_pan_cycle_with_an_interleaved_upgrade_never_shrinks_the_commit_scope():
    """The executor precondition, stated where the session can be held to it.

    WGPU applies a delta instead of republishing the whole binding only while
    ``set(previous_commit["tiles"]) <= set(payloads)``
    (``wgpu_imageview2d.py``), where ``payloads`` is
    ``state.active_payloads(delta)``.  A scope that shrinks between commits
    breaks it and costs a ~150 ms full republication — the mechanism behind the
    148 ms commit p95 in the 2026-08-03 trace.

    Asserting it here rather than through
    ``wgpu_binding_full_republications`` keeps the property machine-independent
    and makes the failure name its own cause; the live counter is asserted over
    a real montage in ``tests/ui/test_montage_out_of_view_retention.py``.
    """

    session = _session()
    _present(session, *range(8))
    legs = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 1))

    previous_keys: set[int] = set()
    for step, leg in enumerate(legs):
        _look_at(session, leg)
        state, delta = session.build_tile_presentation({})
        session.tile_presentation_state = state
        if step == 1:
            # Background work upgrades a tile the camera is not looking at —
            # the LOD refinement that is routine during a zoom.
            _supersede_out_of_view(session, 7)

        keys = set(state.active_payloads(delta))
        assert previous_keys <= keys, (
            f"leg {leg} dropped {sorted(previous_keys - keys)} from the commit scope; "
            "WGPU republishes the whole binding for exactly this"
        )
        assert delta.removals == ()
        assert not session.pending_removals
        previous_keys = keys

    # And the whole montage is still on screen at the end of the cycle.
    assert previous_keys == set(range(8))


def test_a_tile_back_in_view_forgets_its_parked_upgrade():
    """The visible path owns a visible tile's upgrade; the queue must let go."""

    session = _session()
    _present(session, *range(8))
    _look_at(session, (0, 1))
    session.build_tile_presentation({})
    _supersede_out_of_view(session, 5)
    session.build_tile_presentation({})
    assert session.out_of_view_upgrade_pending == {5}

    _look_at(session, (4, 5))
    session.build_tile_presentation({})

    assert session.out_of_view_upgrade_pending == set()
    assert session.out_of_view_upgrades_returned == 1


def test_retention_drops_a_slot_the_plan_repointed_at_another_source():
    """A montage index scroll reuses slots; retained pixels must not follow.

    ``TileLifecycle.retarget`` prunes ``presentable_payloads`` when a slot is
    reused for a different semantic source — the same evidence
    ``may_remove_visible`` reads to decide a visible slot must clear.  Retention
    reads it too rather than deciding staleness for itself.
    """

    session = _session()
    _present(session, *range(8))
    _scope(session, (0, 1))
    assert 6 in session.retained_out_of_view
    session.retained_out_of_view_refusals.clear()

    # Slot 6 now shows a different source. Its retained pixels are the old
    # source's and may never be presented for the new one.
    moved = dataclasses.replace(session.plan.tiles[6], source_index=99)
    plan_tiles = {int(tile.montage_index): tile for tile in session.plan.tiles}
    plan_tiles[6] = moved
    session.lifecycle.retarget(
        {
            6: dataclasses.replace(
                session.lifecycle.peek(6).target, source_index=99, semantic_source_id=("s", 99)
            )
        }
    )

    scope = session.retain_out_of_view_presentation_scope(
        (0, 1),
        plan_tiles_by_number=plan_tiles,
        backend_identities=dict(session.lifecycle.backend_presented_identities),
    )

    assert 6 not in session.retained_out_of_view
    assert 6 not in scope
    assert 4 in session.retained_out_of_view
    # Which source a slot is for is not a quality question, so the
    # camera-motion relaxation never reaches this clause: the retained pixels
    # name source 6 and the plan now names 99.  (The lifecycle prune would
    # refuse it one clause later anyway — ``retarget`` dropped the slot's
    # presentable payloads, so the drawn identity resolves to nothing — which
    # is the second, independent guard.)
    assert session.retained_out_of_view_refusals == {"source_mismatch": 1}
