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


def test_retention_refuses_a_payload_the_backend_is_not_drawing():
    session = _session()
    _present(session, *range(8))
    # A newer wrapper exists for tile 5 but the backend still shows the old
    # one.  That is a repair obligation, not something to retain silently.
    session.display_tile_payloads[5] = _payload(session, 5, pixels=np.zeros(TILE_SHAPE))
    session.display_tile_payloads[5] = dataclasses.replace(
        session.display_tile_payloads[5],
        source_id=("tile", 5, "successor"),
        tile_identity=dataclasses.replace(
            session.display_tile_payloads[5].tile_identity, quality="fallback"
        ),
    )

    scope = _scope(session, (0, 1))

    assert 5 not in session.retained_out_of_view
    assert 5 not in scope
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

    scope = _scope(session, (0, 1))

    assert 6 not in session.retained_out_of_view
    assert 6 not in scope
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


def test_retention_is_disabled_without_a_residency_budget():
    session = _session(budget_bytes=0)
    _present(session, *range(8))

    scope = _scope(session, (0, 1))

    assert scope == (0, 1)
    assert not session.retained_out_of_view
    assert session.retained_out_of_view_budget_bytes == 0


def test_a_retained_tile_survives_the_next_build_unchanged():
    """The identity-repair pass must not undo retention one commit later.

    ``build_tile_presentation`` discards a drawn slot whose identity differs
    from the current payload whenever ``tile not in planned_numbers or
    may_remove_visible(tile)`` — and an out-of-view retained tile satisfies both
    halves of that disjunction.  Retention holds only because it is refused
    unless the backend is drawing exactly the payload the session still has,
    which puts the tile on that pass's early continue.  Get this wrong and every
    retained tile is removed again by the following commit: the behaviour is
    unchanged, the counters are unchanged, and the measurement reads flat for a
    reason nothing reports.
    """

    session = _session()
    _present(session, *range(8))
    _look_at(session, (0, 1))

    first_state, first_delta = session.build_tile_presentation({})
    assert 5 in session.retained_out_of_view
    assert 5 in first_delta.active_tiles
    assert 5 not in first_delta.removals

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


def test_a_retained_tile_whose_payload_moves_on_steps_out_of_the_scope():
    """Retention and the removal path stay mutually exclusive.

    ``build_tile_presentation`` discards a drawn slot whose identity differs
    from its current payload when ``tile not in planned_numbers or
    may_remove_visible(tile)``, and an out-of-view tile satisfies both halves.
    Retention does not widen that predicate; it declines the tile instead.  The
    tile then lands exactly where it would have without retention — removed,
    and out of the presentation scope — rather than being simultaneously
    retained and removed.
    """

    session = _session()
    _present(session, *range(8))
    _look_at(session, (0, 1))
    _state, _delta = session.build_tile_presentation({})
    assert 5 in session.retained_out_of_view

    # A successor wrapper lands for a tile the camera is not looking at.
    superseded = session.display_tile_payloads[5]
    session.display_tile_payloads[5] = dataclasses.replace(
        superseded,
        source_id=("tile", 5, "successor"),
        tile_identity=dataclasses.replace(superseded.tile_identity, quality="fallback"),
    )

    _state, delta = session.build_tile_presentation({})

    assert 5 not in session.retained_out_of_view
    assert 5 not in delta.active_tiles
    assert not set(delta.active_tiles) & set(delta.removals)


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
