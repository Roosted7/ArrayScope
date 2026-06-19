from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.view_state import ViewState
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.lod import LodInfo
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.display.commit import DisplayCommitter
from arrayscope.display.model.frame import DisplayFrameKey, DisplayTilePayload, TilePresentationDelta, TilePresentationState, TiledValueSource
from arrayscope.display.model.commit import DisplayTiledPresentation


def _presentation():
    state = ViewState.from_shape((2, 2, 1)).with_image_axes(0, 1).with_montage_axis(2, columns=1, indices=(0,))
    geometry = DisplayGeometry(
        state,
        (2, 2),
        montage=MontageGeometry(indices=(0,), tile_shape=(2, 2), columns=1, rows=1, gap=0),
        montage_tile_states=("loaded",),
    )
    image = np.arange(4, dtype=np.float32).reshape(2, 2)
    payload = DisplayTilePayload(
        tile_number=0,
        source_index=0,
        image=image,
        histogram_data=None,
        source_id=("tile", 0),
        texture_data=image,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=image,
        source_shape=image.shape,
        lod=LodInfo(0, 1, image.shape, image.shape, 0),
    )
    state = TilePresentationState({0: payload})
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts={0: payload},
        active_tiles=(0,),
        planned_tiles=(0,),
        near_tiles=(0,),
    )
    return DisplayTiledPresentation(
        geometry=geometry,
        levels=(0.0, 3.0),
        histogram_range=(0.0, 3.0),
        viewport_policy=ViewportPolicy.PRESERVE,
        tile_state=state,
        tile_delta=delta,
        tile_residency_budget_bytes=64 * 1024 * 1024,
    )


class _FakeImageView:
    def __init__(self):
        self.commit = None
        self.bounds = None

    def setTiledMontagePresentation(self, **kwargs):
        self.commit = kwargs

    def setProfileMarkerBoundsRect(self, bounds):
        self.bounds = bounds


def test_tiled_committer_keeps_fake_raster_out_of_committed_frame():
    view = _FakeImageView()
    presentation = _presentation()

    frame = DisplayCommitter(view).commit_tile_layer(
        presentation,
        DisplayFrameKey(("doc",), ("view",), 1),
    )

    assert view.commit["tile_state"] == presentation.tile_state
    assert view.commit["tile_delta"] == presentation.tile_delta
    assert view.commit["geometry"] == presentation.geometry
    assert frame.data is None
    assert frame.is_tiled is True
    assert isinstance(frame.value_source, TiledValueSource)
    assert frame.value_source.payloads == presentation.tile_state.payloads
    assert view.bounds == (0.0, 0.0, 1.0, 1.0)


def test_full_commit_rejects_tiled_presentation():
    with pytest.raises(TypeError, match="raster presentation"):
        DisplayCommitter(_FakeImageView()).commit_full(
            _presentation(),
            DisplayFrameKey(("doc",), ("view",), 1),
        )


def test_tiled_value_source_reads_exact_semantic_data_not_lod_texture():
    semantic = np.arange(16, dtype=np.float32).reshape(4, 4)
    lod_texture = np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32)
    payload = DisplayTilePayload(
        tile_number=0,
        source_index=0,
        image=semantic,
        histogram_data=semantic,
        source_id=("tile", 0, "lod", 2),
        texture_data=lod_texture,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=semantic,
        semantic_histogram_data=semantic,
        source_shape=semantic.shape,
        lod=LodInfo(1, 2, semantic.shape, lod_texture.shape, 0),
    )
    source = TiledValueSource({0: payload})

    value = source.value_at(SimpleNamespace(tile_number=0, local_y=3, local_x=2))
    region, hist, kind = source.tile_region(SimpleNamespace(montage_index=0), (slice(2, 4), slice(1, 3)))

    assert value == semantic[3, 2]
    np.testing.assert_array_equal(region, semantic[2:4, 1:3])
    np.testing.assert_array_equal(hist, semantic[2:4, 1:3])
    assert kind == "committed_tile_payload"
