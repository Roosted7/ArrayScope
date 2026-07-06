"""PyQtGraph resident-LOD adoption (ADR 0050 phase 3).

Reduced display payloads map onto native texels through a per-item scale
transform; world footprints stay native so montage geometry, ROIs, and
viewport math never see display LOD.
"""

from types import SimpleNamespace

import numpy as np

from arrayscope.display.backends.pyqtgraph.tiles import (
    _apply_item_lod_scale,
    _payload_direct_dims,
)
from arrayscope.display.lod import LodInfo


def _region(width, height, x=0, y=0):
    return SimpleNamespace(x=x, y=y, width=width, height=height, source_index=0)


def _payload(shape, lod):
    return SimpleNamespace(lod=lod)


def test_native_payload_keeps_min_region_image_behavior():
    data = np.zeros((336, 336), dtype=np.float32)
    lod = LodInfo(0, 1, (336, 336), (336, 336), 0)

    dims = _payload_direct_dims(_region(300, 336), data, _payload(data.shape, lod))

    assert dims == (300, 336, 300, 336, 1.0, 1.0)


def test_reduced_payload_occupies_native_world_footprint():
    data = np.zeros((84, 84), dtype=np.float32)
    lod = LodInfo(2, 4, (336, 336), (84, 84), 0)

    width, height, crop_w, crop_h, scale_x, scale_y = _payload_direct_dims(
        _region(336, 336), data, _payload(data.shape, lod)
    )

    assert (width, height) == (336, 336)
    assert (crop_w, crop_h) == (84, 84)
    assert (scale_x, scale_y) == (4.0, 4.0)


def test_reduced_payload_edge_region_crops_in_image_pixels():
    data = np.zeros((84, 84), dtype=np.float32)
    lod = LodInfo(2, 4, (336, 336), (84, 84), 0)

    width, height, crop_w, crop_h, scale_x, scale_y = _payload_direct_dims(
        _region(100, 336), data, _payload(data.shape, lod)
    )

    assert (width, height) == (100, 336)
    assert crop_w == 25  # ceil(100 / 4)
    assert crop_h == 84
    assert (scale_x, scale_y) == (4.0, 4.0)


def test_missing_lod_info_is_native(qt_app):
    data = np.zeros((64, 64), dtype=np.float32)

    dims = _payload_direct_dims(_region(64, 64), data, SimpleNamespace(lod=None))

    assert dims == (64, 64, 64, 64, 1.0, 1.0)


def test_apply_item_lod_scale_sets_and_resets_transform(qt_app):
    from pyqtgraph.graphicsItems.ImageItem import ImageItem

    state = SimpleNamespace(item=ImageItem(axisOrder="row-major"), lod_scale=(1.0, 1.0))

    _apply_item_lod_scale(state, 4.0, 4.0)
    transform = state.item.transform()
    assert (transform.m11(), transform.m22()) == (4.0, 4.0)
    assert state.lod_scale == (4.0, 4.0)

    # Idempotent: same scale leaves the transform object untouched.
    _apply_item_lod_scale(state, 4.0, 4.0)
    assert (state.item.transform().m11(), state.item.transform().m22()) == (4.0, 4.0)

    # A native payload restores identity (reused pool items must not keep
    # a stale LOD transform).
    _apply_item_lod_scale(state, 1.0, 1.0)
    assert state.item.transform().isIdentity()
    assert state.lod_scale == (1.0, 1.0)
