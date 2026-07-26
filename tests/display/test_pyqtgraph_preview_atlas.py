from __future__ import annotations

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui

from arrayscope.display.backends.pyqtgraph.tiles import MontageTileLayer
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.lod import LodInfo
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    PageBackedPresentation,
    TilePresentationDelta,
)
from arrayscope.display.pyramid import materialize_lod_page, plan_source_grid_pages
from arrayscope.display.shader_mapping import ShaderMapping
from arrayscope.gpu.keys import SCALAR_R32F


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


def test_large_preview_prefix_is_not_acknowledged_as_physical_coverage(qt_app):
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
    assert owner.tile_items == {}
    assert stats.committed_upserts == ()
    assert stats.presented_identities == {}
    assert layer.preview_atlas_decline_reason == "awaiting-complete-preview-transaction"


def test_compact_preview_paints_through_the_real_pyqtgraph_scene(qtbot):
    from arrayscope.display.imageview2d import ImageView2D

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
