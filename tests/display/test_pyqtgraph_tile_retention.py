"""Which warm PyQtGraph tile state a hijack or an eviction destroys.

The point of every test here is *which* state survives, so each one builds a
reuse pool holding more than one candidate and hides its members in a known
order.  A pool with a single candidate cannot distinguish head-first from
tail-first and would pass against either policy.
"""

import numpy as np
import pytest

from arrayscope.core.view_state import ViewState
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    TilePresentationDelta,
    TilePresentationState,
)
from arrayscope.display.montage import MontageTileState

TILE = 4
COLUMNS = 10
LEVELS = (0.0, 255.0)


def _geometry():
    indices = tuple(range(COLUMNS))
    return DisplayGeometry(
        view_state=ViewState.from_shape((TILE, TILE, COLUMNS)).with_montage_axis(
            2, columns=COLUMNS, indices=indices, text=":"
        ),
        display_shape=(TILE, TILE * COLUMNS),
        montage=MontageGeometry(
            indices=indices, tile_shape=(TILE, TILE), columns=COLUMNS, rows=1, gap=0
        ),
        montage_tile_states=(MontageTileState.LOADED,) * COLUMNS,
    )


def _payloads():
    # Per-tile gradients, not constant fills: a state that is silently swapped
    # for another tile's pixels still reads as "an image" under a constant
    # fill, so the survivor assertions below compare actual content.
    return {
        tile: DisplayTilePayload(
            tile,
            tile,
            np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE) + tile * 100.0,
            None,
            ("retention", tile),
        )
        for tile in range(COLUMNS)
    }


class _Commits:
    """Drive real tiled commits with an explicit visible set per step."""

    def __init__(self, view, payloads):
        self._view = view
        self._payloads = payloads
        self._revision = 0

    def __call__(self, active, *, upserts=None, near=(), budget=0):
        self._revision += 1
        active = tuple(sorted(int(tile) for tile in active))
        upsert_tiles = active if upserts is None else tuple(sorted(int(t) for t in upserts))
        delta = TilePresentationDelta(
            structure_revision=1,
            payload_revision=self._revision,
            visibility_revision=self._revision,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=self._revision,
            upserts={tile: self._payloads[tile] for tile in upsert_tiles},
            active_tiles=active,
            planned_tiles=active,
            near_tiles=tuple(sorted(int(tile) for tile in near)),
        )
        self._view.setTiledPresentation(
            geometry=_geometry(),
            tile_state=TilePresentationState(self._payloads),
            tile_delta=delta,
            histogramPlotData=None,
            levels=LEVELS,
            histogramRange=LEVELS,
            tile_residency_budget_bytes=int(budget),
        )
        return self._view.lastImageUploadTiming()


@pytest.fixture
def layer_commits(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        yield view, _Commits(view, _payloads())
    finally:
        view.close()


def _pool_tiles(layer):
    return [int(state.tile_number) for state in layer._direct_reuse_pool]


def _seed_pool(commit):
    """States for tiles 0-3, then hide 0, 1, 2 in that order.

    Leaves the pool holding three candidates whose hide order is known, with
    tile 3 still visible.
    """

    commit((0, 1, 2, 3))
    commit((1, 2, 3))
    commit((2, 3))
    commit((3,))


def test_hijack_takes_the_longest_hidden_state_not_the_one_just_hidden(layer_commits):
    view, commit = layer_commits
    layer = view._montage_tile_layer

    _seed_pool(commit)
    assert _pool_tiles(layer) == [0, 1, 2]
    survivor_pixels = np.asarray(layer.states[2].item.image).copy()

    # Tile 9 has never been presented, so it has no state and must hijack one.
    stats = commit((3, 9))

    assert stats.tile_layer_items_created == 0, (
        "a pooled state was available; nothing should be allocated"
    )
    # Tail-first (the old policy) would have taken tile 2 — the tile that just
    # left the viewport and the first one a pan back would ask for.
    assert set(layer.states) == {1, 2, 3, 9}
    assert _pool_tiles(layer) == [1, 2]
    assert np.array_equal(np.asarray(layer.states[2].item.image), survivor_pixels)


def test_hijack_prefers_a_state_no_halo_is_claiming(layer_commits):
    view, commit = layer_commits
    layer = view._montage_tile_layer

    _seed_pool(commit)

    # The coldest state is tile 0, but the delta names it near, so the hijack
    # skips it and takes the coldest stranger instead.
    stats = commit((3, 9), near=(0,))

    assert stats.tile_layer_items_created == 0
    assert set(layer.states) == {0, 2, 3, 9}
    assert _pool_tiles(layer) == [0, 2]


def test_hijack_still_succeeds_when_every_pooled_state_is_near(layer_commits):
    view, commit = layer_commits
    layer = view._montage_tile_layer

    _seed_pool(commit)

    # Every candidate is protected. The halo is a preference, not a pin: a new
    # ImageItem per entering tile would cost more than the warm state it saved,
    # so the coldest protected state is taken anyway.
    stats = commit((3, 9), near=(0, 1, 2))

    assert stats.tile_layer_items_created == 0
    assert set(layer.states) == {1, 2, 3, 9}
    assert _pool_tiles(layer) == [1, 2]


def test_hijack_allocates_only_when_the_pool_is_empty(layer_commits):
    view, commit = layer_commits
    layer = view._montage_tile_layer

    stats = commit((0, 1))
    assert stats.tile_layer_items_created == 2
    assert _pool_tiles(layer) == []

    stats = commit((0, 1, 9))
    assert stats.tile_layer_items_created == 1
    assert set(layer.states) == {0, 1, 9}


def test_budget_prune_evicts_strangers_before_the_halo(layer_commits):
    view, commit = layer_commits
    layer = view._montage_tile_layer

    _seed_pool(commit)
    tile_bytes = int(np.asarray(layer.states[3].item.image).nbytes)

    # Room for the visible tile plus one warm state: two of the three pooled
    # states must go, and tile 1 is the one the halo claims.
    stats = commit((3,), upserts=(), near=(1,), budget=tile_bytes * 2)

    assert stats.tile_layer_storage_evictions == 2
    assert set(layer.states) == {1, 3}
    assert _pool_tiles(layer) == [1]


def test_budget_prune_evicts_a_near_tile_when_it_is_the_last_candidate(layer_commits):
    view, commit = layer_commits
    layer = view._montage_tile_layer

    _seed_pool(commit)
    tile_bytes = int(np.asarray(layer.states[3].item.image).nbytes)

    # Whole halo, hard ceiling of one tile. The budget wins: protection may
    # only reorder victims, never make the ceiling unenforceable.
    stats = commit((3,), upserts=(), near=(0, 1, 2), budget=tile_bytes)

    assert stats.tile_layer_storage_evictions == 3
    assert set(layer.states) == {3}
    assert _pool_tiles(layer) == []
    assert int(layer._resident_bytes) <= tile_bytes


def test_budget_prune_keeps_an_unevictable_state_in_the_reuse_pool(layer_commits):
    view, commit = layer_commits
    layer = view._montage_tile_layer

    _seed_pool(commit)
    # Park a still-visible state in the pool. `_hide_tile` does exactly this on
    # its way out, and `_displace_tile_slot_resident` leaves states there too;
    # the prune used to pop such a state and drop it, freeing nothing and
    # silently retiring a warm-reuse claim that nothing restored.
    visible_state = layer.states[3]
    layer._add_to_direct_reuse_pool(visible_state)
    assert _pool_tiles(layer) == [0, 1, 2, 3]

    # A budget below the whole resident set, so the loop runs out of evictable
    # candidates rather than out of pool entries.
    stats = commit((3,), upserts=(), budget=1)

    assert stats.tile_layer_storage_evictions == 3
    assert set(layer.states) == {3}
    assert layer.states[3] is visible_state
    assert _pool_tiles(layer) == [3], "the unevictable state must keep its reuse claim"
