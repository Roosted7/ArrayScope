from __future__ import annotations

import numpy as np

from arrayscope.core.scheduler import FrameTarget
from arrayscope.core.view_state import ViewState
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.backend_contract import PYQTGRAPH_CAPABILITIES
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.scene import DisplayLayout, display_scene_for_geometry, display_scene_for_presentation
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
from arrayscope.display.model.commit import DisplayTiledPresentation


def test_normal_image_geometry_is_one_tile_scene_without_payloads():
    geometry = DisplayGeometry(ViewState.from_shape((5, 7)).with_image_axes(0, 1), (5, 7))

    scene = display_scene_for_geometry(geometry)

    assert scene.layout is DisplayLayout.SINGLE
    assert scene.bounds == (0.0, 0.0, 6.0, 4.0)
    assert len(scene.regions) == 1
    assert scene.regions[0].bounds == scene.bounds
    assert scene.active_region_ids == (0,)
    assert scene.resident_region_ids == ()


def _montage_geometry():
    state = ViewState.from_shape((3, 4, 3)).with_image_axes(0, 1).with_montage_axis(2, columns=2, indices=(0, 1, 2))
    return DisplayGeometry(
        state,
        (7, 9),
        montage=MontageGeometry(indices=(0, 1, 2), tile_shape=(3, 4), columns=2, rows=2, gap=1),
        montage_tile_states=("loaded", "loading", "loaded"),
    )


def test_montage_scene_separates_visibility_nearness_and_residency():
    geometry = _montage_geometry()
    image = np.zeros((3, 4), dtype=np.float32)
    payload = DisplayTilePayload(0, 0, image, None, ("tile", 0))
    state = TilePresentationState({0: payload})
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=2,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=2,
        active_tiles=(1,),
        planned_tiles=(0, 1, 2),
        near_tiles=(0, 1),
    )
    presentation = DisplayTiledPresentation(
        geometry=geometry,
        levels=(0.0, 1.0),
        histogram_range=(0.0, 1.0),
        viewport_policy=ViewportPolicy.PRESERVE,
        tile_state=state,
        base_tile_state=state,
        tile_delta=delta,
        tile_residency_budget_bytes=1024,
    )

    scene = display_scene_for_presentation(presentation)

    assert scene.layout is DisplayLayout.MONTAGE
    assert scene.bounds == (0.0, 0.0, 8.0, 6.0)
    assert scene.active_region_ids == (1,)
    assert scene.planned_region_ids == (0, 1, 2)
    assert scene.near_region_ids == (0, 1)
    assert scene.resident_region_ids == (0,)
    assert scene.region(2).source_index == 2
    assert scene.region(2).bounds == (0.0, 4.0, 3.0, 6.0)


def test_tiled_single_plane_scene_uses_frame_plan_regions_without_montage_geometry():
    state = ViewState.from_shape((4, 4)).with_image_axes(0, 1)
    target = FrameTarget("semantic", "viewport", "presentation", "exact-visible")
    frame_plan = FramePlanner(internal_tile_shape=(2, 2)).plan(
        target=target,
        view_state=state,
        display_shape=(4, 4),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )
    payloads = {
        region.region_id: DisplayTilePayload(
            region.region_id,
            region.region_id,
            np.zeros((region.height, region.width), dtype=np.float32),
            None,
            ("single", region.region_id),
        )
        for region in frame_plan.regions
    }
    tile_state = TilePresentationState(payloads)
    tile_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=frame_plan.active_region_ids,
        planned_tiles=frame_plan.planned_region_ids,
        near_tiles=frame_plan.near_region_ids,
    )
    presentation = DisplayTiledPresentation(
        geometry=frame_plan.geometry,
        levels=(0.0, 1.0),
        histogram_range=(0.0, 1.0),
        viewport_policy=ViewportPolicy.PRESERVE,
        tile_state=tile_state,
        base_tile_state=TilePresentationState(),
        tile_delta=tile_delta,
        tile_residency_budget_bytes=1024,
        frame_plan=frame_plan,
    )

    scene = display_scene_for_presentation(presentation)

    assert scene.layout is DisplayLayout.SINGLE
    assert scene.bounds == (0.0, 0.0, 3.0, 3.0)
    assert scene.active_region_ids == (0, 1, 2, 3)
    assert [region.bounds for region in scene.regions] == [
        (0.0, 0.0, 1.0, 1.0),
        (2.0, 0.0, 3.0, 1.0),
        (0.0, 2.0, 1.0, 3.0),
        (2.0, 2.0, 3.0, 3.0),
    ]


def test_frame_plan_scene_uses_current_tile_delta_activity():
    state = ViewState.from_shape((4, 4)).with_image_axes(0, 1)
    frame_plan = FramePlanner(internal_tile_shape=(2, 2)).plan(
        target=FrameTarget("semantic", "left", "presentation", "exact-visible"),
        view_state=state,
        display_shape=(4, 4),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        view_range=((0.0, 1.0), (0.0, 1.0)),
    )
    payloads = {
        region.region_id: DisplayTilePayload(
            region.region_id,
            region.region_id,
            np.zeros((region.height, region.width), dtype=np.float32),
            None,
            ("single", region.region_id),
        )
        for region in frame_plan.regions
    }
    tile_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=2,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=2,
        upserts=payloads,
        active_tiles=(3,),
        planned_tiles=frame_plan.planned_region_ids,
        near_tiles=(1, 2, 3),
    )
    presentation = DisplayTiledPresentation(
        geometry=frame_plan.geometry,
        levels=(0.0, 1.0),
        histogram_range=(0.0, 1.0),
        viewport_policy=ViewportPolicy.PRESERVE,
        tile_state=TilePresentationState(payloads),
        base_tile_state=TilePresentationState(),
        tile_delta=tile_delta,
        tile_residency_budget_bytes=1024,
        frame_plan=frame_plan,
    )

    scene = display_scene_for_presentation(presentation)

    assert frame_plan.active_region_ids == (0,)
    assert scene.active_region_ids == (3,)
    assert scene.near_region_ids == (1, 2, 3)
