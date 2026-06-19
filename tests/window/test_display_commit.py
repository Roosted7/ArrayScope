from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.view_state import ViewState
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.window.display_commit import DisplayCommitter
from arrayscope.window.display_frame import DisplayFrameKey, DisplayTilePayload, TilePresentationDelta, TilePresentationState, TiledValueSource
from arrayscope.window.render_model import DisplayTiledPresentation


def _presentation():
    state = ViewState.from_shape((2, 2, 1)).with_image_axes(0, 1).with_montage_axis(2, columns=1, indices=(0,))
    geometry = DisplayGeometry(
        state,
        (2, 2),
        montage=MontageGeometry(indices=(0,), tile_shape=(2, 2), columns=1, rows=1, gap=0),
        montage_tile_states=("loaded",),
    )
    payload = DisplayTilePayload(0, 0, np.arange(4, dtype=np.float32).reshape(2, 2), None, ("tile", 0))
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
