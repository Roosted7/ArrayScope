from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.scheduler import FrameTarget
from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import PYQTGRAPH_CAPABILITIES
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.lod import LodInfo
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.display.commit import DisplayCommitter
from arrayscope.display.model.frame import DisplayFrameKey, DisplayTilePayload, TileCommitReport, TilePresentationDelta, TilePresentationState, TiledValueSource
from arrayscope.display.model.commit import DisplayTiledPresentation


_AUTO_REPORT = object()


def test_commit_report_acceptance_preserves_delta_upsert_order():
    presentation = _presentation()
    sample = presentation.tile_delta.upserts[0]
    upserts = {
        tile: replace(
            sample,
            tile_number=tile,
            source_index=tile,
            source_id=("tile", tile),
        )
        for tile in (3, 1, 2)
    }
    delta = replace(
        presentation.tile_delta,
        upserts=upserts,
        active_tiles=(1, 2, 3),
        planned_tiles=(1, 2, 3),
        near_tiles=(),
    )
    report = TileCommitReport(
        presented_tiles=frozenset((1, 2, 3)),
        committed_upserts=frozenset((1, 3)),
    )

    assert report.accepted_upserts_in_order(delta) == (3, 1)


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
    base_state = TilePresentationState()
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
        base_tile_state=base_state,
        tile_delta=delta,
        tile_residency_budget_bytes=64 * 1024 * 1024,
    )


class _FakeImageView:
    def __init__(self):
        self.surface = self
        self.widget = self
        self.capabilities = PYQTGRAPH_CAPABILITIES
        self.commit = None
        self.bounds = None
        self.report = _AUTO_REPORT

    def present_tiled(self, presentation):
        self.commit = {
            "tile_state": presentation.tile_state,
            "tile_delta": presentation.tile_delta,
            "geometry": presentation.geometry,
        }
        if self.report is _AUTO_REPORT:
            return TileCommitReport(
                presented_tiles=frozenset(presentation.tile_state.active_payloads(presentation.tile_delta)),
                committed_upserts=frozenset(presentation.tile_delta.upserts),
                removed_tiles=frozenset(presentation.tile_delta.removals),
            )
        return self.report

    def hide_tiled_presentation(self, reason):
        self.hide_tiled_reason = str(reason)

    def invalidate_tiled_presentation(self, reason):
        self.invalidate_tiled_reason = str(reason)

    def reset_tiled_residency(self, reason):
        self.reset_tiled_reason = str(reason)

    def set_profile_bounds(self, bounds):
        self.bounds = bounds

    def apply_camera(self, image_shape, viewport_policy, *, image_origin=(0.0, 0.0), content_rect=None):
        self.camera = (tuple(image_shape), viewport_policy, tuple(image_origin), content_rect)

    def map_scene_to_overlay(self, scene_pos):
        return scene_pos

    def current_viewport_rect(self):
        return None

    def presentation_diagnostics(self):
        return {"backend": self.capabilities.name, "interaction_event_owner": self.interaction_event_owner()}

    def interaction_event_owner(self):
        return "fake"

    def sync_interaction_state(self, state):
        self.interaction_state = state

    def reset_surface(self, reason):
        self.reset_reason = str(reason)

    def teardown_surface(self):
        self.teardown_called = True


def test_tiled_committer_keeps_source_pixels_out_of_committed_frame():
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


def test_committer_cache_identity_tracks_shell_not_surface_widget():
    view = _FakeImageView()
    view.widget = object()

    committer = DisplayCommitter(view)

    assert committer.image_view is view
    assert committer.surface is view


def test_tiled_committer_excludes_unpresented_payloads_from_committed_frame():
    view = _FakeImageView()
    view.report = TileCommitReport(presented_tiles=())
    presentation = _presentation()
    committer = DisplayCommitter(view)

    frame = committer.commit_tile_layer(
        presentation,
        DisplayFrameKey(("doc",), ("view",), 1),
    )

    assert frame.data is None
    assert frame.value_source.payloads == {}
    assert committer.last_tile_committed_state.payloads == {}


def test_tiled_committer_rejects_missing_backend_acknowledgement():
    view = _FakeImageView()
    view.report = None

    with pytest.raises(TypeError, match="TileCommitReport"):
        DisplayCommitter(view).commit_tile_layer(
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


def test_single_plane_commits_with_internal_tile_geometry():
    view = _FakeImageView()
    state = ViewState.from_shape((4, 4)).with_image_axes(0, 1)
    frame_plan = FramePlanner(internal_tile_shape=(2, 2)).plan(
        target=FrameTarget("semantic", "viewport", "presentation", "exact-visible"),
        view_state=state,
        display_shape=(4, 4),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    payloads = {
        region.region_id: DisplayTilePayload(
            tile_number=region.region_id,
            source_index=region.region_id,
            image=image[region.data_slices],
            histogram_data=image[region.data_slices],
            source_id=("single", region.region_id),
            semantic_data=image[region.data_slices],
            semantic_histogram_data=image[region.data_slices],
        )
        for region in frame_plan.regions
    }
    base_state = TilePresentationState()
    tile_state = TilePresentationState(payloads)
    tile_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        base_revision=0,
        upserts=payloads,
        active_tiles=frame_plan.active_region_ids,
        planned_tiles=frame_plan.planned_region_ids,
        near_tiles=frame_plan.near_region_ids,
    )
    presentation = DisplayTiledPresentation(
        geometry=frame_plan.geometry,
        levels=(0.0, 15.0),
        histogram_range=(0.0, 15.0),
        viewport_policy=ViewportPolicy.PRESERVE,
        tile_state=tile_state,
        base_tile_state=base_state,
        tile_delta=tile_delta,
        tile_residency_budget_bytes=1024,
        frame_plan=frame_plan,
    )

    frame = DisplayCommitter(view).commit_tile_layer(
        presentation,
        DisplayFrameKey(("doc",), ("single",), 1),
    )

    assert view.commit["geometry"].montage == frame_plan.geometry.montage
    assert frame.scene.layout.value == "single"
    assert frame.scene.resident_region_ids == (0, 1, 2, 3)
    assert frame.value_source.value_at(SimpleNamespace(tile_number=3, local_y=1, local_x=1)) == 15.0
    region, hist, kind = frame.value_source.tile_region(SimpleNamespace(region_id=3), (slice(0, 2), slice(0, 2)))
    np.testing.assert_array_equal(region, image[2:4, 2:4])
    np.testing.assert_array_equal(hist, image[2:4, 2:4])
    assert kind == "committed_tile_payload"


def test_eager_region_source_matches_existing_display_image_slices():
    from arrayscope.display.region_source import EagerDisplayRegionSource
    from arrayscope.display.slice_engine import DisplayImage

    state = ViewState.from_shape((4, 4)).with_image_axes(0, 1)
    frame_plan = FramePlanner(internal_tile_shape=(2, 2)).plan(
        target=FrameTarget("semantic", "viewport", "presentation", "exact-visible"),
        view_state=state,
        display_shape=(4, 4),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    hist = image + 100.0
    source = EagerDisplayRegionSource(DisplayImage(image, histogram_data=hist), source_key=("source",))

    payload = source.read_region(frame_plan.regions[3], quality="exact-visible", deadline_ns=123)

    np.testing.assert_array_equal(payload.image, image[2:4, 2:4])
    np.testing.assert_array_equal(payload.histogram_data, hist[2:4, 2:4])
    assert payload.tile_number == 3
