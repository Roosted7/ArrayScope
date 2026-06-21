from __future__ import annotations

import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.scene import DisplayLayout, DisplayStorage, display_scene_for_geometry, display_scene_for_presentation
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
from arrayscope.display.model.commit import DisplayRasterPresentation, DisplayTiledPresentation


def test_normal_raster_is_one_region_scene():
    geometry = DisplayGeometry(ViewState.from_shape((5, 7)).with_image_axes(0, 1), (5, 7))

    scene = display_scene_for_geometry(geometry)

    assert scene.layout is DisplayLayout.SINGLE
    assert scene.storage is DisplayStorage.RASTER
    assert scene.bounds == (0.0, 0.0, 6.0, 4.0)
    assert len(scene.regions) == 1
    assert scene.regions[0].bounds == scene.bounds
    assert scene.active_region_ids == (0,)


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
    assert scene.storage is DisplayStorage.TILED
    assert scene.bounds == (0.0, 0.0, 8.0, 6.0)
    assert scene.active_region_ids == (1,)
    assert scene.planned_region_ids == (0, 1, 2)
    assert scene.near_region_ids == (0, 1)
    assert scene.resident_region_ids == (0,)
    assert scene.region(2).source_index == 2
    assert scene.region(2).bounds == (0.0, 4.0, 3.0, 6.0)


def test_raster_montage_and_tiled_montage_have_same_region_geometry():
    geometry = _montage_geometry()
    canvas = np.zeros(geometry.display_shape, dtype=np.float32)
    raster = DisplayRasterPresentation(
        data=canvas,
        histogram_data=None,
        histogram_plot_data=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogram_range=(0.0, 1.0),
        viewport_policy=ViewportPolicy.PRESERVE,
    )
    state = TilePresentationState({})
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        active_tiles=(0, 1, 2),
        planned_tiles=(0, 1, 2),
        near_tiles=(0, 1, 2),
    )
    tiled = DisplayTiledPresentation(
        geometry=geometry,
        levels=(0.0, 1.0),
        histogram_range=(0.0, 1.0),
        viewport_policy=ViewportPolicy.PRESERVE,
        tile_state=state,
        base_tile_state=state,
        tile_delta=delta,
        tile_residency_budget_bytes=1024,
    )

    raster_scene = display_scene_for_presentation(raster)
    tiled_scene = display_scene_for_presentation(tiled)

    assert [region.bounds for region in raster_scene.regions] == [region.bounds for region in tiled_scene.regions]
    assert [region.source_index for region in raster_scene.regions] == [region.source_index for region in tiled_scene.regions]
    assert raster_scene.storage is DisplayStorage.RASTER
    assert tiled_scene.storage is DisplayStorage.TILED
