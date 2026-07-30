"""Deferred out-of-view invalidation: queue, bound, release and drain.

The live ordering is pinned against real montages in
``tests/ui/test_montage_data_change_invalidation.py``.  These tests own the
bookkeeping that path cannot make deterministic: that the queue is a queue and
not a leak, that a tile the camera comes back to leaves it instead of racing
its own replacement, and that one drain is bounded rather than whole-montage.
"""

from __future__ import annotations

import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.display.montage import make_montage_plan
from arrayscope.window.frame_session import FrameSession

TILE_SHAPE = (4, 4)
SOURCES = tuple(range(16))


def _session(*, defer: bool = True) -> FrameSession:
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
        visible_tiles=plan.tiles[:4],
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        shader_display=True,
        defer_out_of_view_removals=defer,
    )


def test_a_deferred_tile_is_not_a_pending_removal_until_it_drains():
    session = _session()
    for tile in range(4, 16):
        session.defer_out_of_view_invalidation(tile)

    # The whole point: a queued removal re-enters the ``removals`` tuple of
    # every commit until the backend acknowledges it, so while a tile is
    # deferred it must be nowhere near that queue.
    assert not session.pending_removals
    assert not session.dirty_payloads
    assert session.deferred_out_of_view_deferred == 12

    drained = session.drain_deferred_out_of_view_invalidation(12)
    assert set(drained) == set(range(4, 16))
    assert session.pending_removals == set(range(4, 16))
    assert set(session.dirty_payloads) == set(range(4, 16))
    assert not session.deferred_out_of_view_invalidations


def test_deferring_the_same_tile_twice_queues_it_once():
    session = _session()
    session.defer_out_of_view_invalidation(7)
    session.defer_out_of_view_invalidation(7)
    assert session.deferred_out_of_view_invalidations == {7}
    assert session.deferred_out_of_view_deferred == 1


def test_one_drain_is_bounded_and_takes_the_lowest_tiles_first():
    session = _session()
    for tile in range(4, 16):
        session.defer_out_of_view_invalidation(tile)

    assert session.drain_deferred_out_of_view_invalidation(5) == (4, 5, 6, 7, 8)
    assert len(session.deferred_out_of_view_invalidations) == 7
    # Deterministic order matters for reproducing a partially drained montage;
    # a set iteration would make the same fixture drain differently per run.
    assert session.drain_deferred_out_of_view_invalidation(5) == (9, 10, 11, 12, 13)
    assert session.drain_deferred_out_of_view_invalidation(0) == ()
    assert session.deferred_out_of_view_invalidations == {14, 15}


def test_a_tile_back_in_scope_leaves_the_queue_instead_of_being_removed():
    """A pan back onto a deferred tile must not queue its removal.

    The seam already stopped the backend drawing it, so there is nothing left
    to remove — and a removal queued now would race the replacement upsert for
    the same slot.
    """

    session = _session()
    for tile in range(4, 16):
        session.defer_out_of_view_invalidation(tile)

    session.release_deferred_out_of_view_invalidations({5, 6, 99})

    assert session.deferred_out_of_view_invalidations == set(range(4, 16)) - {5, 6}
    assert session.deferred_out_of_view_returned == 2
    assert not session.pending_removals
    drained = session.drain_deferred_out_of_view_invalidation(16)
    assert 5 not in drained
    assert 6 not in drained


def test_the_queue_is_bounded_by_the_montage():
    """No unbounded backlog: a tile can be in the queue at most once.

    Repeated data changes on the same montage — an index scrub is exactly
    that — must not grow the queue past the tile count, or the "bounded"
    claim is only true for a single step.
    """

    session = _session()
    for _ in range(50):
        for tile in range(4, 16):
            session.defer_out_of_view_invalidation(tile)
    assert len(session.deferred_out_of_view_invalidations) == 12
