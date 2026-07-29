from __future__ import annotations

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui

from arrayscope.display.backends.pyqtgraph.tiles import MontageTileLayer
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.imageview2d import ImageView2D
from arrayscope.display.lod import LodInfo
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    PageBackedPresentation,
    TilePresentationDelta,
)
from arrayscope.display.pyramid import materialize_lod_page, plan_source_grid_pages
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    ShaderMapping,
    cpu_display_rgba,
    default_phase_lut,
)
from arrayscope.gpu.keys import COMPLEX_RG32F, SCALAR_R32F


class _Owner:
    def __init__(self):
        self.preview_items = []
        self.tile_items = {}

    def add_montage_preview_item(self, item):
        self.preview_items.append(item)

    def remove_montage_preview_item(self, item):
        self.preview_items.remove(item)

    def add_tile_item(self, tile_number, item):
        self.tile_items[int(tile_number)] = item

    def remove_tile_item(self, tile_number):
        self.tile_items.pop(int(tile_number), None)

    def move_tile_item(self, old_tile_number, new_tile_number, item):
        self.tile_items.pop(int(old_tile_number), None)
        self.tile_items[int(new_tile_number)] = item

    def unmap_tile_item(self, tile_number, item=None):
        if item is None or self.tile_items.get(int(tile_number)) is item:
            self.tile_items.pop(int(tile_number), None)


def _geometry(count: int, *, columns: int = 17, gap: int = 1) -> DisplayGeometry:
    rows = (int(count) + int(columns) - 1) // int(columns)
    return DisplayGeometry(
        view_state=None,
        display_shape=(rows * 8 + max(0, rows - 1) * gap, columns * 8 + (columns - 1) * gap),
        montage=MontageGeometry(
            indices=tuple(range(count)),
            tile_shape=(8, 8),
            columns=columns,
            rows=rows,
            gap=gap,
        ),
    )


def _preview_payload(tile_number: int) -> DisplayTilePayload:
    native = np.full((8, 8), float(tile_number + 1), dtype=np.float32)
    plans = plan_source_grid_pages(
        content_key=(("preview-atlas", tile_number), None),
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
        tile_number=tile_number,
        source_index=tile_number,
        image=pages[0].values,
        histogram_data=None,
        source_id=("preview", tile_number),
        texture_data=pages[0].values,
        semantic_data=None,
        source_shape=(8, 8),
        lod=lod,
        shader_mapping=ShaderMapping(),
        quality="preview",
        page_backing=PageBackedPresentation(plans, pages, (0, 8, 0, 8), lod),
    )


def _complex_preview_payload(tile_number: int) -> DisplayTilePayload:
    phase = np.float32((tile_number % 16) * (2.0 * np.pi / 16.0) - np.pi)
    magnitude = np.float32(tile_number + 1)
    native = np.full(
        (8, 8),
        magnitude * np.exp(np.complex64(1j) * phase),
        dtype=np.complex64,
    )
    plans = plan_source_grid_pages(
        content_key=(("complex-preview-atlas", tile_number), None),
        valid_source_rect_yx=(0, 8, 0, 8),
        reduction_yx=(2, 2),
        stored_page_shape=(256, 256),
        dtype="complex64",
        representation=COMPLEX_RG32F,
        reducer="mean",
    )
    pages = tuple(
        materialize_lod_page(native, source_origin_yx=(0, 0), plan=plan) for plan in plans
    )
    lod = LodInfo(level=2, factor=4, source_shape=(8, 8), texture_shape=(2, 2), gutter=0)
    mapping = ShaderMapping(
        component=ShaderComponent.ABS,
        display_mode=ShaderDisplayMode.PHASE_COLOR,
        lut_data=default_phase_lut(),
    )
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=pages[0].values,
        histogram_data=np.abs(pages[0].values),
        source_id=("complex-preview", tile_number),
        texture_data=pages[0].values,
        semantic_data=None,
        source_shape=(8, 8),
        lod=lod,
        shader_mapping=mapping,
        quality="preview",
        page_backing=PageBackedPresentation(plans, pages, (0, 8, 0, 8), lod),
    )


def _exact_payload(tile_number: int) -> DisplayTilePayload:
    image = np.full((8, 8), 1000.0 + tile_number, dtype=np.float32)
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=image,
        histogram_data=None,
        source_id=("exact", tile_number),
        semantic_data=image,
        source_shape=image.shape,
        quality="exact",
    )


def _delta(payloads, *, active, planned=None, cold_deadline_ms=None):
    return TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        base_revision=0,
        target_revision=1,
        transaction_generation=7,
        cold_deadline_ms=cold_deadline_ms,
        upserts=payloads,
        active_tiles=tuple(active),
        planned_tiles=tuple(active if planned is None else planned),
    )


def _layer(owner: _Owner) -> MontageTileLayer:
    return MontageTileLayer(
        owner,
        set_image_item_data=lambda item, data, levels, **_kwargs: item.setImage(
            data, autoLevels=False, levels=levels
        ),
        record_upload_timing=lambda *_args, **_kwargs: None,
        histogram_levels_for_display=lambda levels: levels,
        is_rgb_image=lambda image: np.asarray(image).ndim == 3,
    )


def test_large_raw_preview_is_one_compact_physical_item_with_per_tile_truth(qt_app):
    count = 272
    owner = _Owner()
    layer = _layer(owner)
    payloads = {tile: _preview_payload(tile) for tile in range(count)}

    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=_geometry(count),
        levels=(0.0, float(count + 1)),
        rgb_already_windowed=False,
        dirty_tiles=tuple(payloads),
        tile_payloads=payloads,
        tile_delta=_delta(payloads, active=range(count)),
    )

    assert len(owner.preview_items) == 1
    assert owner.tile_items == {}
    assert layer.preview_atlas_page_count <= 2
    assert layer.physically_visible_tile_count == count
    assert stats.items_created == 1
    assert stats.committed_upserts == tuple(range(count))
    assert stats.presented_identities == {
        tile: payload.source_id for tile, payload in payloads.items()
    }
    assert set(layer.tile_truth_physical_rows()) == set(range(count))

    raster = QtGui.QImage(
        int(_geometry(count).display_shape[1]),
        int(_geometry(count).display_shape[0]),
        QtGui.QImage.Format.Format_RGBA8888,
    )
    raster.fill(QtGui.QColor(0, 0, 0, 0))
    painter = QtGui.QPainter(raster)
    owner.preview_items[0].paint(painter, None)
    painter.end()
    assert raster.pixelColor(4, 4).alpha() == 255
    assert raster.pixelColor(8, 4).alpha() == 0, "montage gaps stay physically empty"
    assert raster.pixelColor(13, 4).alpha() == 255


def test_large_complex_preview_is_cpu_composited_at_round_levels_and_rewindowable(qt_app):
    count = 272
    owner = _Owner()
    layer = _layer(owner)
    layer.set_lookup_table(default_phase_lut())
    payloads = {tile: _complex_preview_payload(tile) for tile in range(count)}
    levels = (0.0, float(count + 1))

    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=_geometry(count),
        levels=levels,
        rgb_already_windowed=False,
        dirty_tiles=tuple(payloads),
        tile_payloads=payloads,
        tile_delta=_delta(payloads, active=range(count)),
    )

    assert stats.committed_upserts == tuple(range(count))
    assert len(owner.preview_items) == 1
    assert owner.tile_items == {}
    item = owner.preview_items[0]
    sample_tile = count // 2
    sample_part = item.tiles[sample_tile].parts[0]
    sample_x = int(sample_part.source_rect[0])
    sample_y = int(sample_part.source_rect[1])
    expected = cpu_display_rgba(payloads[sample_tile].texture_data, item.mapping)[0, 0]
    before = item._images[sample_part.page_index].pixelColor(sample_x, sample_y)
    assert (before.red(), before.green(), before.blue(), before.alpha()) == tuple(expected)
    rows = layer.tile_truth_physical_rows()
    assert {row["physical_texture_kind"] for row in rows.values()} == {"complex_rg32f"}
    assert {row["physical_mapping_mode"] for row in rows.values()} == {"cpu_rgb_from_complex_atlas"}
    assert all(row["physical_levels"] == levels for row in rows.values())

    wider_levels = (0.0, 2.0 * levels[1])
    level_stats = layer.update_levels(wider_levels)
    expected_wider = cpu_display_rgba(payloads[sample_tile].texture_data, item.mapping)[0, 0]
    after = item._images[sample_part.page_index].pixelColor(sample_x, sample_y)
    assert level_stats.items_updated == 0
    assert (after.red(), after.green(), after.blue(), after.alpha()) == tuple(expected_wider)
    assert after != before


def test_exact_item_replaces_only_its_atlas_member_after_success(qt_app):
    count = 256
    owner = _Owner()
    layer = _layer(owner)
    previews = {tile: _preview_payload(tile) for tile in range(count)}
    geometry = _geometry(count)
    layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1200.0),
        rgb_already_windowed=False,
        dirty_tiles=tuple(previews),
        tile_payloads=previews,
        tile_delta=_delta(previews, active=range(count)),
    )
    exact = _exact_payload(0)
    mixed = dict(previews)
    mixed[0] = exact

    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1200.0),
        rgb_already_windowed=False,
        dirty_tiles=(0,),
        tile_payloads=mixed,
        tile_delta=_delta({0: exact}, active=range(count)),
    )

    assert set(owner.tile_items) == {0}
    assert layer.preview_atlas_active_tiles == frozenset(range(1, count))
    assert stats.presented_identities[0] == exact.source_id
    assert stats.presented_identities[1] == previews[1].source_id
    assert layer.physically_visible_tile_count == count


def test_large_preview_prefix_uses_staged_items_without_torn_atlas(qt_app):
    count = 256
    owner = _Owner()
    layer = _layer(owner)
    payloads = {tile: _preview_payload(tile) for tile in range(32)}

    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=_geometry(count),
        levels=(0.0, float(count + 1)),
        rgb_already_windowed=False,
        dirty_tiles=tuple(payloads),
        tile_payloads=payloads,
        tile_delta=_delta(
            payloads,
            active=range(32),
            planned=range(count),
        ),
    )

    assert owner.preview_items == []
    assert set(owner.tile_items) == set(range(32))
    assert stats.committed_upserts == tuple(range(32))
    assert set(stats.presented_identities) == set(range(32))
    assert layer.preview_atlas_decline_reason == "awaiting-complete-preview-transaction"


def test_large_preview_completes_around_retained_exact_items(qt_app):
    """A single-slice predecessor may satisfy part of the preview scope.

    Entering a full montage retains the one or two exact ImageItems already on
    screen and evaluates reduced previews only for the missing tiles.  The
    compact transaction is complete when those retained items physically
    satisfy the complement; requiring every active tile to appear in the
    preview-upsert delta rejects the same complete 270+2 frame forever.
    """

    count = 272
    retained_tiles = (135, 136)
    owner = _Owner()
    layer = _layer(owner)
    geometry = _geometry(count)
    exact = {tile: _exact_payload(tile) for tile in retained_tiles}
    layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1400.0),
        rgb_already_windowed=False,
        dirty_tiles=retained_tiles,
        tile_payloads=exact,
        tile_delta=_delta(exact, active=retained_tiles),
    )
    previews = {tile: _preview_payload(tile) for tile in range(count) if tile not in retained_tiles}
    mixed = {**previews, **exact}

    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1400.0),
        rgb_already_windowed=False,
        dirty_tiles=tuple(previews),
        tile_payloads=mixed,
        tile_delta=_delta(previews, active=range(count), planned=range(count)),
    )

    assert stats.committed_upserts == tuple(sorted(previews))
    assert stats.presented_tiles == tuple(range(count))
    assert layer.preview_atlas_active_tiles == frozenset(previews)
    assert set(owner.tile_items) == set(retained_tiles)
    assert all(owner.tile_items[tile].isVisible() for tile in retained_tiles)
    assert stats.presented_identities == {tile: mixed[tile].source_id for tile in range(count)}


def test_compact_preview_paints_through_the_real_pyqtgraph_scene(qtbot):
    count = 256
    geometry = _geometry(count)
    payloads = {tile: _preview_payload(tile) for tile in range(count)}
    view = ImageView2D()
    qtbot.addWidget(view)
    view.resize(640, 520)
    view.show()
    qtbot.waitExposed(view)

    stats = view._montage_tile_layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 257.0),
        rgb_already_windowed=False,
        dirty_tiles=tuple(payloads),
        tile_payloads=payloads,
        tile_delta=_delta(payloads, active=range(count)),
    )
    view.getView().setRange(
        xRange=(0.0, float(geometry.display_shape[1])),
        yRange=(0.0, float(geometry.display_shape[0])),
        padding=0.0,
    )
    view.graphicsView.viewport().repaint()

    assert stats.committed_upserts == tuple(range(count))
    item = view._montage_tile_layer._preview_atlas_item
    assert item is not None
    assert item.scene() is not None
    tile_number = 128
    row, column = divmod(tile_number, 17)
    world = QtCore.QPointF(column * 9 + 4, row * 9 + 4)
    scene = view.getView().mapViewToScene(world)
    viewport_point = view.graphicsView.mapFromScene(scene)
    raster = view.graphicsView.viewport().grab().toImage()
    color = raster.pixelColor(viewport_point)
    expected = round((tile_number + 1) * 255.0 / 257.0)
    assert (
        max(
            abs(color.red() - expected),
            abs(color.green() - expected),
            abs(color.blue() - expected),
        )
        <= 2
    )


def test_failed_exact_upload_keeps_preview_member(qt_app, monkeypatch):
    count = 256
    owner = _Owner()
    layer = _layer(owner)
    previews = {tile: _preview_payload(tile) for tile in range(count)}
    geometry = _geometry(count)
    layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1200.0),
        rgb_already_windowed=False,
        dirty_tiles=tuple(previews),
        tile_payloads=previews,
        tile_delta=_delta(previews, active=range(count)),
    )
    exact = _exact_payload(0)
    mixed = dict(previews)
    mixed[0] = exact
    monkeypatch.setattr(
        layer,
        "_set_tile_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected upload failure")),
    )

    with pytest.raises(RuntimeError, match="injected upload failure"):
        layer.update_presentation(
            None,
            histogram_data=None,
            geometry=geometry,
            levels=(0.0, 1200.0),
            rgb_already_windowed=False,
            dirty_tiles=(0,),
            tile_payloads=mixed,
            tile_delta=_delta({0: exact}, active=range(count)),
        )

    assert 0 in layer.preview_atlas_active_tiles
    assert not owner.tile_items[0].isVisible()
    assert layer.physically_visible_tile_count == count
    assert (
        layer.tile_truth_physical_rows()[0]["physical_acknowledged_identity"]
        == previews[0].source_id
    )


# --- presented-visibility equivalence, compact-preview path -------------------


def _assert_presented_equivalence(layer):
    """The maintained answer must equal a fresh full scan.

    Same gate as `tests/display/test_presented_visibility_equivalence.py`, run
    here because the compact-preview atlas needs 256 tiles to engage and that
    module works at direct-presentation scale.
    """

    layer.assert_presented_index_matches_scan()


def test_presented_index_agrees_while_the_compact_preview_atlas_owns_the_frame(qt_app):
    count = 256
    owner = _Owner()
    layer = _layer(owner)
    previews = {tile: _preview_payload(tile) for tile in range(count)}
    layer.update_presentation(
        None,
        histogram_data=None,
        geometry=_geometry(count),
        levels=(0.0, 1200.0),
        rgb_already_windowed=False,
        dirty_tiles=tuple(previews),
        tile_payloads=previews,
        tile_delta=_delta(previews, active=range(count)),
    )

    _assert_presented_equivalence(layer)
    # The atlas owns every tile, so no direct item is presented — the two
    # populations are disjoint here, which is the case most likely to be
    # double-counted.
    assert layer.presented_tiles == set()
    assert layer.physically_visible_tile_count == count


def test_presented_index_agrees_when_an_exact_item_leaves_the_atlas(qt_app):
    count = 256
    owner = _Owner()
    layer = _layer(owner)
    previews = {tile: _preview_payload(tile) for tile in range(count)}
    geometry = _geometry(count)
    layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1200.0),
        rgb_already_windowed=False,
        dirty_tiles=tuple(previews),
        tile_payloads=previews,
        tile_delta=_delta(previews, active=range(count)),
    )
    exact = _exact_payload(0)
    mixed = dict(previews)
    mixed[0] = exact
    layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1200.0),
        rgb_already_windowed=False,
        dirty_tiles=(0,),
        tile_payloads=mixed,
        tile_delta=_delta({0: exact}, active=range(count)),
    )

    _assert_presented_equivalence(layer)
    # Tile 0 is now a direct item; the rest remain atlas members.
    assert layer.presented_tiles == {0}
    assert layer.preview_atlas_active_tiles == frozenset(range(1, count))


def test_presented_index_agrees_after_hiding_a_compact_preview_frame(qt_app):
    count = 256
    owner = _Owner()
    layer = _layer(owner)
    previews = {tile: _preview_payload(tile) for tile in range(count)}
    layer.update_presentation(
        None,
        histogram_data=None,
        geometry=_geometry(count),
        levels=(0.0, 1200.0),
        rgb_already_windowed=False,
        dirty_tiles=tuple(previews),
        tile_payloads=previews,
        tile_delta=_delta(previews, active=range(count)),
    )
    layer.hide_all()

    _assert_presented_equivalence(layer)
    assert layer.presented_tiles == set()


def test_presented_index_agrees_after_a_residency_reset_of_a_compact_frame(qt_app):
    count = 256
    owner = _Owner()
    layer = _layer(owner)
    previews = {tile: _preview_payload(tile) for tile in range(count)}
    layer.update_presentation(
        None,
        histogram_data=None,
        geometry=_geometry(count),
        levels=(0.0, 1200.0),
        rgb_already_windowed=False,
        dirty_tiles=tuple(previews),
        tile_payloads=previews,
        tile_delta=_delta(previews, active=range(count)),
    )
    layer.clear()

    _assert_presented_equivalence(layer)
    assert layer.presented_tiles == set()
