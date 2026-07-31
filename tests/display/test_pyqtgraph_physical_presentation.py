"""Physical-presentation truth gates for the pyqtgraph tile layer.

What the commit stats report must match what the ImageItems physically draw.
"""

from types import SimpleNamespace

import numpy as np

from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.model.frame import DisplayTilePayload


class _Owner:
    def add_tile_item(self, *_args):
        pass

    def remove_tile_item(self, *_args):
        pass

    def move_tile_item(self, *_args):
        pass


def _make_layer(set_image_item_data=lambda *_args, **_kwargs: None):
    from arrayscope.display.backends.pyqtgraph.tiles import MontageTileLayer

    return MontageTileLayer(
        _Owner(),
        set_image_item_data=set_image_item_data,
        record_upload_timing=lambda *_args, **_kwargs: None,
        histogram_levels_for_display=lambda levels: levels,
        is_rgb_image=lambda _image: False,
    )


def _real_image_upload(item, data, levels, **_kwargs):
    """Actually put pixels on the item, as the production setter does.

    Re-showing a retained tile is a claim that its pixels are on screen, so
    the layer refuses to make it for an item that has no image. A stubbed
    upload leaves every item imageless and would turn that guard into the
    thing under test.
    """

    item.setImage(np.asarray(data), autoLevels=False, levels=levels)
    return True


def test_identity_rejected_upserts_are_reported_not_silent(qt_app):
    """Session-148 gate (2026-07-16): typed-target rejection must be loud.

    A delta upsert whose payload identity cannot satisfy that tile's target
    identity is excluded from presentation.  On the pyqtgraph backend that
    exclusion was completely silent (no stat, no skip count), so a presenter
    re-emitting the same dead payload looped forever while the tile stayed
    empty on screen — invisible to the commit_batch trace, immune to the
    presenter's re-commit backoff, and undetectable by trace_verify.  The
    commit stats must name the rejected tiles so diagnostics and traces
    expose the loop on the first commit.
    """

    from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind

    def identity(semantic_generation):
        return TileIdentity(
            document_generation=("doc", 0),
            operation_key=("ops",),
            source_index=0,
            image_axes=(1, 0),
            axis_flips=(False, False),
            channel="real",
            complex_mapping=("scalar", "real", "mapped"),
            texture_kind=TexturePlaneKind.SCALAR_R32F,
            semantic_generation=semantic_generation,
            lod=TileLodIdentity(level=0, factor=1),
        )

    geometry = DisplayGeometry(
        view_state=None,
        display_shape=(4, 4),
        montage=MontageGeometry(indices=(0,), tile_shape=(4, 4), columns=1, rows=1, gap=0),
    )
    image = np.zeros((4, 4), dtype=np.float32)
    payload = DisplayTilePayload(
        0,
        0,
        image,
        None,
        ("tile", 0),
        tile_identity=identity(("stale",)),
    )
    layer = _make_layer()
    delta = SimpleNamespace(
        upserts={0: payload},
        active_tiles=(0,),
        target_identities={0: identity(("current",))},
        removals=(),
        near_tile_source_ids={},
        cold_deadline_ms=None,
    )
    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        rgb_already_windowed=False,
        dirty_tiles=None,
        tile_payloads={0: payload},
        tile_delta=delta,
    )

    assert stats.committed_upserts == ()
    assert stats.presented_tiles == ()
    assert stats.identity_rejected_items == 1
    assert stats.identity_rejected_tiles == (0,)

    # The matching identity presents normally and reports zero rejections.
    delta.target_identities = {0: identity(("stale",))}
    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        rgb_already_windowed=False,
        dirty_tiles=None,
        tile_payloads={0: payload},
        tile_delta=delta,
    )
    assert stats.committed_upserts == (0,)
    assert stats.presented_tiles == (0,)
    assert stats.identity_rejected_items == 0
    assert stats.identity_rejected_tiles == ()

    # A retained payload re-presented outside the delta's upserts is not a
    # rejected upsert: the presenter is not looping on it, so it must not
    # trip the loud path even when a new target has outrun the retained
    # payload's identity.
    delta.upserts = {}
    delta.target_identities = {0: identity(("newer",))}
    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        rgb_already_windowed=False,
        dirty_tiles=None,
        tile_payloads={0: payload},
        tile_delta=delta,
    )
    assert stats.identity_rejected_items == 0
    assert stats.identity_rejected_tiles == ()


def test_atomic_identity_rejection_preserves_complete_predecessor(qt_app):
    """A rejected successor transaction must not punch a physical hole."""

    from arrayscope.display.model.tile_identity import TileIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind

    def identity(generation):
        return TileIdentity(
            document_generation=("doc", 0),
            operation_key=(),
            source_index=0,
            image_axes=(0, 1),
            axis_flips=(False, False),
            channel="real",
            complex_mapping=("scalar", "real", "mapped"),
            texture_kind=TexturePlaneKind.SCALAR_R32F,
            semantic_generation=generation,
        )

    geometry = DisplayGeometry(
        view_state=None,
        display_shape=(4, 4),
        montage=MontageGeometry(indices=(0,), tile_shape=(4, 4), columns=1, rows=1, gap=0),
    )
    image = np.ones((4, 4), dtype=np.float32)
    predecessor = DisplayTilePayload(
        0,
        0,
        image,
        None,
        ("predecessor",),
        tile_identity=identity(("crop", 97, 197)),
    )
    layer = _make_layer()
    initial_delta = SimpleNamespace(
        upserts={0: predecessor},
        active_tiles=(0,),
        target_identities={0: predecessor.tile_identity},
        removals=(),
        near_tile_source_ids={},
        cold_deadline_ms=None,
        atomic_handoff=False,
    )
    initial = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        rgb_already_windowed=False,
        dirty_tiles=None,
        tile_payloads={0: predecessor},
        tile_delta=initial_delta,
    )
    assert initial.presented_tiles == (0,)

    stale_successor = DisplayTilePayload(
        0,
        0,
        image * 2,
        None,
        ("stale-successor",),
        tile_identity=identity(("crop", 96, 196)),
    )
    current_target = identity(("crop", 94, 194))
    rejected_delta = SimpleNamespace(
        upserts={0: stale_successor},
        active_tiles=(0,),
        target_identities={0: current_target},
        removals=(),
        near_tile_source_ids={},
        cold_deadline_ms=None,
        atomic_handoff=True,
    )
    rejected = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        rgb_already_windowed=False,
        dirty_tiles=None,
        tile_payloads={0: stale_successor},
        tile_delta=rejected_delta,
    )

    assert rejected.presented_tiles == (0,)
    assert rejected.committed_upserts == ()
    assert rejected.identity_rejected_tiles == (0,)
    assert layer._states[0].acknowledged_identity == predecessor.tile_identity
    assert layer._states[0].visible is True


def _row_geometry(tiles: int, tile: int = 4):
    return DisplayGeometry(
        view_state=None,
        display_shape=(tile, tile * tiles),
        montage=MontageGeometry(
            indices=tuple(range(tiles)),
            tile_shape=(tile, tile),
            columns=tiles,
            rows=1,
            gap=0,
        ),
    )


def _row_identity(tile_number: int, semantic_generation=("gen", 0)):
    from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind

    return TileIdentity(
        document_generation=("doc", 0),
        operation_key=(),
        source_index=int(tile_number),
        image_axes=(0, 1),
        axis_flips=(False, False),
        channel="real",
        complex_mapping=("scalar", "real", "mapped"),
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_generation=semantic_generation,
        lod=TileLodIdentity(level=0, factor=1),
    )


def _row_payload(tile_number: int, *, tile: int = 4, semantic_generation=("gen", 0)):
    return DisplayTilePayload(
        int(tile_number),
        int(tile_number),
        np.full((tile, tile), float(tile_number), dtype=np.float32),
        None,
        ("tile", int(tile_number)),
        tile_identity=_row_identity(tile_number, semantic_generation),
    )


def _commit(layer, geometry, *, active, upserts, payloads, targets, levels=(0.0, 1.0)):
    delta = SimpleNamespace(
        upserts=dict(upserts),
        active_tiles=tuple(int(tile) for tile in active),
        planned_tiles=tuple(int(tile) for tile in active),
        target_identities=dict(targets),
        removals=(),
        near_tile_source_ids={},
        cold_deadline_ms=None,
        atomic_handoff=False,
        force_refresh=False,
    )
    return layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=levels,
        rgb_already_windowed=False,
        dirty_tiles=tuple(int(tile) for tile in upserts),
        tile_payloads={int(tile): payloads[int(tile)] for tile in active},
        tile_delta=delta,
    )


def test_reentering_retained_tile_is_shown_in_the_commit_that_admits_it(qt_app):
    """The pan-back hole (render-eagerness map 2026-07-30, §2.4).

    ``resolve_items`` is restricted to the admitted upserts whenever a delta
    carries any, so a tile that re-entered the viewport without being an
    upsert had no drawable payload and fell through to the hide sweep — while
    its ``_states`` entry, its ImageItem and its pixels had survived the whole
    excursion. It came back only on a later commit that admitted it as an
    upsert, after ``rearm_for_scope`` re-dirtied it: a hole that blinks on a
    pan back, plus an avoidable round trip through admission.

    The regime is load-bearing and both halves must be present in the SAME
    commit, or the defect cannot occur and the test cannot fail:
      (a) the delta carries at least one upsert (tile 1 refines), and
      (b) tile 0 re-enters ``active`` as a NON-upsert holding a retained,
          identity-satisfying state.
    """

    geometry = _row_geometry(3)
    payloads = {tile: _row_payload(tile) for tile in range(3)}
    targets = {tile: payloads[tile].tile_identity for tile in range(3)}
    layer = _make_layer(_real_image_upload)

    first = _commit(
        layer,
        geometry,
        active=(0, 1),
        upserts={0: payloads[0], 1: payloads[1]},
        payloads=payloads,
        targets=targets,
    )
    assert first.presented_tiles == (0, 1)

    # Pan right: tile 0 leaves the viewport. Its state is retained, hidden,
    # and pooled -- pixels, ImageItem and scene membership all survive.
    panned = _commit(
        layer,
        geometry,
        active=(1, 2),
        upserts={2: payloads[2]},
        payloads=payloads,
        targets=targets,
    )
    assert panned.presented_tiles == (1, 2)
    retained = layer._states[0]
    assert retained.visible is False
    assert retained.item.image is not None
    assert retained.acknowledged_identity == targets[0]

    # Pan back: tile 0 re-enters as a non-upsert while tile 1 is upserted.
    back = _commit(
        layer,
        geometry,
        active=(0, 1),
        upserts={1: payloads[1]},
        payloads=payloads,
        targets=targets,
    )

    assert back.committed_upserts == (1,), "regime (a): the delta must carry an upsert"
    assert 0 not in back.committed_upserts, "regime (b): tile 0 must not be an upsert"
    # The physical claim: pixels on screen in THIS commit, not merely a
    # surviving `_states` entry.
    assert back.presented_tiles == (0, 1)
    assert layer._states[0] is retained
    assert retained.item.isVisible() is True
    assert retained.visible is True
    layer.assert_presented_index_matches_scan()

    # The warm-residency claim is dropped the moment the tile is presented:
    # leaving it pooled lets a later tile in the same transaction pop and move
    # the very same ImageItem, acknowledging two tiles for one physical item.
    assert all(state is not retained for state in layer._direct_reuse_pool)
    assert id(retained) not in layer._direct_reuse_pool_ids
    assert layer._states[0].item is not layer._states[1].item


def test_reentering_tile_with_stale_retained_identity_stays_hidden(qt_app):
    """The stale-pixel guard is not weakened by the re-show above.

    Same regime, one difference: while tile 0 was out of view the target
    moved on. Its retained pixels no longer satisfy the current target, so
    re-showing them would put stale pixels on screen. It must stay hidden and
    wait for an upsert.
    """

    geometry = _row_geometry(3)
    payloads = {tile: _row_payload(tile) for tile in range(3)}
    targets = {tile: payloads[tile].tile_identity for tile in range(3)}
    layer = _make_layer(_real_image_upload)

    _commit(
        layer,
        geometry,
        active=(0, 1),
        upserts={0: payloads[0], 1: payloads[1]},
        payloads=payloads,
        targets=targets,
    )
    _commit(
        layer,
        geometry,
        active=(1, 2),
        upserts={2: payloads[2]},
        payloads=payloads,
        targets=targets,
    )
    assert layer._states[0].visible is False

    outrun = dict(targets)
    outrun[0] = _row_identity(0, semantic_generation=("gen", 1))
    back = _commit(
        layer,
        geometry,
        active=(0, 1),
        upserts={1: payloads[1]},
        payloads=payloads,
        targets=outrun,
    )

    assert back.committed_upserts == (1,)
    assert back.presented_tiles == (1,)
    assert layer._states[0].visible is False
    assert layer._states[0].item.isVisible() is False
    # A payload the target has outrun is not a rejected upsert: the presenter
    # is not looping on it.
    assert back.identity_rejected_tiles == ()
    layer.assert_presented_index_matches_scan()


def test_resolve_budget_defers_unprepared_page_backed_upserts(qt_app, monkeypatch):
    """The commit-cost attribution finding (S3 follow-up, 2026-07-31).

    Page assembly used to run eagerly over every admitted upsert while only
    the item loop honored ``cold_deadline_ms``: one measured commit assembled
    49 FFT payloads (~30 ms each) inside the GUI callback and then applied 5
    of them.  With the budget shared, an exhausted deadline defers the
    remaining unprepared page-backed upserts to the next commit — at least one
    payload always resolves so the commit makes progress, deferred tiles are
    neither hidden nor reported as rejected, and an atomic handoff is never
    cut.
    """

    from arrayscope.display.backends.pyqtgraph import tiles as tiles_module
    from arrayscope.display.backends.pyqtgraph.tiles import (
        _resolve_page_backed_payload,
    )
    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import PageBackedPresentation
    from arrayscope.display.pyramid import materialize_lod_page, plan_source_grid_pages
    from arrayscope.gpu.keys import SCALAR_R32F

    assemblies = []

    def counting_assembly(payload, *, levels=None):
        assemblies.append(int(payload.tile_number))
        return _resolve_page_backed_payload(payload, levels=levels)

    monkeypatch.setattr(tiles_module, "_resolve_page_backed_payload", counting_assembly)

    def page_backed_payload(tile: int) -> DisplayTilePayload:
        native = np.full((8, 8), float(tile + 1), dtype=np.float32)
        plans = plan_source_grid_pages(
            content_key=(("budget", tile), None),
            valid_source_rect_yx=(0, 8, 0, 8),
            reduction_yx=(2, 2),
            stored_page_shape=(256, 256),
            dtype="float32",
            representation=SCALAR_R32F,
            reducer="mean",
        )
        pages = tuple(
            materialize_lod_page(native, source_origin_yx=(0, 0), plan=plan) for plan in plans
        )
        lod = LodInfo(level=2, factor=4, source_shape=(8, 8), texture_shape=(2, 2), gutter=0)
        return DisplayTilePayload(
            tile,
            tile,
            pages[0].values,
            None,
            ("tile", tile),
            texture_data=pages[0].values,
            source_shape=(8, 8),
            lod=lod,
            tile_identity=_row_identity(tile),
            page_backing=PageBackedPresentation(plans, pages, (0, 8, 0, 8), lod),
        )

    geometry = _row_geometry(4, tile=8)
    payloads = {tile: page_backed_payload(tile) for tile in range(4)}
    targets = {tile: _row_identity(tile) for tile in range(4)}
    layer = _make_layer(_real_image_upload)

    def commit(upsert_tiles, *, deadline, atomic=False):
        delta = SimpleNamespace(
            upserts={tile: payloads[tile] for tile in upsert_tiles},
            active_tiles=(0, 1, 2, 3),
            planned_tiles=(0, 1, 2, 3),
            target_identities=dict(targets),
            removals=(),
            near_tile_source_ids={},
            cold_deadline_ms=deadline,
            atomic_handoff=atomic,
            force_refresh=False,
        )
        return layer.update_presentation(
            None,
            histogram_data=None,
            geometry=geometry,
            levels=(0.0, 1.0),
            rgb_already_windowed=False,
            dirty_tiles=tuple(upsert_tiles),
            tile_payloads=dict(payloads),
            tile_delta=delta,
        )

    first = commit((0, 1, 2, 3), deadline=0.0)
    # The budget is already burned when the loop starts, so exactly the
    # progress-guarantee payload assembled; the others deferred — invisible to
    # the rejection channel but visible through committed_upserts.
    assert assemblies == [0]
    assert first.committed_upserts == (0,)
    assert first.identity_rejected_tiles == ()
    assert first.presented_tiles == (0,)

    # Re-admission carries only the still-uncommitted tiles, as the session
    # does once an upsert is acknowledged.
    second = commit((1, 2, 3), deadline=0.0)
    assert assemblies == [0, 1]
    assert set(second.committed_upserts) == {1}
    # The tile committed by the previous round keeps its pixels.
    assert set(second.presented_tiles) == {0, 1}

    # An atomic handoff's RESOLVE is never budget-cut: both remaining tiles
    # assemble even though the deadline is already burned.  (The item loop's
    # own deadline handling is untouched here — production atomic deltas do
    # not carry one.)
    atomic = commit((2, 3), deadline=0.0, atomic=True)
    assert assemblies == [0, 1, 2, 3]
    assert {0, 1, 2}.issubset(set(atomic.presented_tiles))

    final = commit((3,), deadline=None)
    assert set(final.presented_tiles) == {0, 1, 2, 3}
    layer.assert_presented_index_matches_scan()
