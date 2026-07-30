"""Ring 1. The maintained presented set must equal the full scan, always.

`MontageTileLayer` used to answer "which tiles are on screen" by walking every
resident state and asking Qt whether each item was showing, on every bounded
commit. Most resident states are retained-but-hidden in the reuse pool, so that
walk was mostly about tiles that were not candidates at all.

It is now answered from a maintained index of the layer's *intent*
(`state.visible` plus still owning its slot), with Qt visibility confirmed per
candidate. That split is the load-bearing part: intent is the layer's own field
and can be remembered, while an item's Qt visibility can change behind the
layer's back — the scene, the owner, or a test can hide it — so it is asked, not
cached.

This module is the gate on that replacement. Every scenario below drives real
visibility mutations and then asserts the maintained answer equals a fresh full
scan. A mutation site that forgets to sync shows up here as a concrete
difference rather than as a tile that silently stops being acknowledged.

The distinctions the layer draws are deliberate and are preserved:

- ``state.visible`` — the layer intends this tile to show
- the item's Qt visibility — what the scene will actually draw
- *active* — selected by this transaction
- *resident* — a state exists (possibly pooled and hidden)
- *presented* — visible intent AND Qt-visible
- *physically acknowledged* — presented AND carrying an accepted identity
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.core.view_state import ChannelMode
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.imageview2d import ImageView2D
from arrayscope.display.lod import LodInfo
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    TilePresentationDelta,
    TilePresentationState,
)
from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity
from arrayscope.display.shader_mapping import TexturePlaneKind

TILE = 8


@pytest.fixture(scope="module")
def qt_app():
    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg

    return pg.mkQApp()


def _geometry(count: int, *, columns: int = 4) -> DisplayGeometry:
    rows = (count + columns - 1) // columns
    return DisplayGeometry(
        view_state=None,
        display_shape=(rows * TILE, columns * TILE),
        montage=MontageGeometry(
            indices=tuple(range(count)),
            tile_shape=(TILE, TILE),
            columns=columns,
            rows=rows,
            gap=0,
        ),
    )


def _identity(tile: int, revision: int) -> TileIdentity:
    return TileIdentity(
        document_generation=("presented-visibility", 1),
        operation_key=("identity",),
        source_index=tile,
        image_axes=(0, 1),
        axis_flips=(False, False),
        channel=ChannelMode.REAL,
        complex_mapping=None,
        texture_kind=TexturePlaneKind.RGB8,
        semantic_generation=("fixture", revision),
        lod=TileLodIdentity(level=0, factor=1),
    )


def _payload(tile: int, *, revision: int = 0) -> DisplayTilePayload:
    values = np.full((TILE, TILE), float(tile + 1 + revision * 100), dtype=np.float32)
    return DisplayTilePayload(
        tile,
        tile,
        values,
        values,
        ("montage-tile", revision, tile),
        semantic_data=values,
        texture_data=values,
        lod=LodInfo(0, 1, values.shape, values.shape, 0),
        tile_identity=_identity(tile, revision),
    )


def _payloads(count: int, *, revision: int = 0) -> dict[int, DisplayTilePayload]:
    return {tile: _payload(tile, revision=revision) for tile in range(count)}


def _delta(payloads, *, revision: int = 1, active=None, targets=None) -> TilePresentationDelta:
    tiles = tuple(sorted(payloads)) if active is None else tuple(sorted(active))
    return TilePresentationDelta(
        structure_revision=revision,
        payload_revision=revision,
        visibility_revision=revision,
        level_revision=revision,
        histogram_revision=revision,
        viewport_revision=revision,
        base_revision=revision - 1,
        target_revision=revision,
        upserts=payloads,
        active_tiles=tiles,
        planned_tiles=tiles,
        target_identities=(
            {tile: payloads[tile].tile_identity for tile in tiles} if targets is None else targets
        ),
    )


def _present(view, payloads, *, revision: int = 1, levels=(0.0, 900.0), delta=None, count=None):
    tiles = count if count is not None else (max(payloads) + 1 if payloads else 1)
    view.setTiledPresentation(
        geometry=_geometry(tiles),
        tile_state=TilePresentationState(payloads, revision=revision),
        tile_delta=_delta(payloads, revision=revision) if delta is None else delta,
        histogramPlotData=None,
        levels=levels,
        histogramRange=levels,
    )


def _check(view) -> set[int]:
    """Assert equivalence and return the agreed presented set."""

    layer = view._montage_tile_layer
    layer.assert_presented_index_matches_scan()
    return layer.presented_tiles


@pytest.fixture
def view(qt_app):
    view = ImageView2D()
    try:
        yield view
    finally:
        view.deleteLater()
        for _ in range(3):
            qt_app.processEvents()


# --- direct presentation ------------------------------------------------------


def test_direct_presentation_agrees_after_the_first_commit(view):
    _present(view, _payloads(6))

    assert _check(view) == set(range(6))


def test_direct_presentation_agrees_after_a_second_commit(view):
    _present(view, _payloads(6))
    _present(view, _payloads(6, revision=1), revision=2)

    assert _check(view) == set(range(6))


# --- hide / show --------------------------------------------------------------


def test_hide_all_leaves_nothing_presented(view):
    _present(view, _payloads(6))
    view._montage_tile_layer.hide_all()

    assert _check(view) == set()


def test_showing_again_after_hide_all_agrees(view):
    _present(view, _payloads(6))
    view._montage_tile_layer.hide_all()
    _present(view, _payloads(6), revision=2)

    assert _check(view) == set(range(6))


def test_an_item_hidden_behind_the_layers_back_is_not_presented(view):
    """Qt visibility is confirmed, never remembered.

    The scene, the owner, or a test can hide an item without telling the layer.
    An index that cached Qt visibility would keep reporting the tile as on
    screen; confirming per candidate cannot.
    """

    _present(view, _payloads(6))
    layer = view._montage_tile_layer
    layer.states[3].item.setVisible(False)

    assert _check(view) == {0, 1, 2, 4, 5}


def test_visible_intent_without_qt_visibility_is_not_presented(view):
    _present(view, _payloads(6))
    layer = view._montage_tile_layer
    state = layer.states[2]
    state.item.setVisible(False)
    state.visible = True  # intent restored, Qt still hidden

    assert 2 not in _check(view)


# --- scope shrink and expansion ----------------------------------------------


def test_scope_shrink_drops_the_tiles_that_left(view):
    _present(view, _payloads(8))
    _present(view, _payloads(3), revision=2)

    assert _check(view) == {0, 1, 2}


def test_scope_expansion_picks_up_the_tiles_that_arrived(view):
    _present(view, _payloads(3))
    _present(view, _payloads(8), revision=2)

    assert _check(view) == set(range(8))


def test_shrink_then_expand_agrees(view):
    _present(view, _payloads(8))
    _present(view, _payloads(2), revision=2)
    _present(view, _payloads(8), revision=3)

    assert _check(view) == set(range(8))


# --- item reuse and eviction --------------------------------------------------


def test_item_reuse_across_a_shrink_and_regrow_agrees(view):
    """States retire into the reuse pool and come back on other slots."""

    _present(view, _payloads(8))
    _present(view, _payloads(2), revision=2)
    # The six retired states are resident and hidden; regrowing must re-home
    # them without the index remembering their old slots.
    _present(view, _payloads(6, revision=1), revision=3)

    assert _check(view) == set(range(6))


def test_eviction_under_a_residency_budget_agrees(view):
    _present(view, _payloads(8))
    _present(view, _payloads(2), revision=2)
    layer = view._montage_tile_layer
    # Force every pooled state out.
    layer._prune_resident_items(budget_bytes=1, active_tiles={0, 1})

    assert _check(view) == {0, 1}


def test_discarding_the_reuse_pool_agrees(view):
    _present(view, _payloads(8))
    _present(view, _payloads(2), revision=2)
    view._montage_tile_layer._discard_direct_reuse_pool()

    assert _check(view) == {0, 1}


# --- residency reset ----------------------------------------------------------


def test_residency_reset_empties_both_answers(view):
    _present(view, _payloads(6))
    view._montage_tile_layer.clear()

    assert _check(view) == set()


def test_presentation_after_a_reset_agrees(view):
    _present(view, _payloads(6))
    view._montage_tile_layer.clear()
    _present(view, _payloads(4), revision=2)

    assert _check(view) == {0, 1, 2, 3}


# --- stale replacement and atomic handoff ------------------------------------


def test_stale_replacement_of_one_slot_agrees(view):
    """A newer payload displaces the state occupying a slot."""

    _present(view, _payloads(4))
    replaced = _payloads(4, revision=1)
    _present(view, replaced, revision=2)

    assert _check(view) == {0, 1, 2, 3}


def test_a_commit_whose_identities_are_not_satisfied_hides_those_tiles(view):
    """Unacknowledged tiles are hidden, which is a visibility mutation."""

    _present(view, _payloads(6))
    payloads = _payloads(6)
    # Ask for identities no payload carries: the layer must hide rather than
    # present pixels it cannot vouch for.
    _present(
        view,
        payloads,
        revision=2,
        delta=_delta(
            payloads,
            revision=2,
            targets={tile: _identity(tile, 99) for tile in payloads},
        ),
    )

    _check(view)


def test_explicit_hides_in_the_delta_agree(view):
    _present(view, _payloads(6))
    payloads = _payloads(6)
    kept = {tile: payloads[tile] for tile in (0, 1, 2)}
    _present(view, kept, revision=2, count=6, delta=_delta(kept, revision=2))

    assert _check(view) == {0, 1, 2}


def test_a_levels_update_does_not_disturb_the_agreement(view):
    _present(view, _payloads(6))
    view._montage_tile_layer.update_levels((0.0, 500.0))

    assert _check(view) == set(range(6))


def test_a_state_must_own_the_slot_that_indexes_it(view):
    """Mapping membership is not ownership when the state's tile disagrees."""

    _present(view, _payloads(3))
    layer = view._montage_tile_layer
    state = layer.states.pop(1)
    layer.states[99] = state

    assert _check(view) == {0, 2}
    assert 99 not in layer._visible_intent_tiles


def test_all_mapping_mutators_keep_the_candidate_set_in_step(view):
    """Inherited ``dict`` mutators must not bypass the maintained index."""

    _present(view, _payloads(3))
    layer = view._montage_tile_layer
    states = layer.states

    removed_tile, state = states.popitem()
    assert removed_tile not in layer._visible_intent_tiles

    state.tile_number = 99
    states |= {99: state}
    assert 99 in layer._visible_intent_tiles
    assert _check(view) == set(states)


# --- the oracle itself --------------------------------------------------------


def test_the_oracle_fails_when_the_index_is_deliberately_corrupted(view):
    """Without this, every assertion above could be passing vacuously."""

    _present(view, _payloads(6))
    layer = view._montage_tile_layer
    layer._visible_intent_tiles.discard(2)

    with pytest.raises(AssertionError, match="scan-only=\\[2\\]"):
        layer.assert_presented_index_matches_scan()


def test_the_oracle_fails_on_a_phantom_index_entry(view):
    _present(view, _payloads(3))
    layer = view._montage_tile_layer
    layer.states[1].item.setVisible(False)
    layer._visible_intent_tiles.add(99)

    # 1 is hidden in Qt so neither side counts it; 99 has no state at all, so
    # the confirmation step drops it too — the index may hold candidates that
    # are not presented, which is exactly why it is not the answer by itself.
    layer.assert_presented_index_matches_scan()
    assert layer.presented_tiles == {0, 2}
