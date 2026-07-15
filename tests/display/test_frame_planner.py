from __future__ import annotations

from arrayscope.core.frame_targets import FrameTarget
from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import PYQTGRAPH_CAPABILITIES
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.scene import DisplayLayout


def _target(semantic="sem", viewport=None):
    return FrameTarget(
        semantic_key=semantic,
        viewport_key=viewport,
        presentation_key=("levels", 0.0, 1.0),
        quality="exact-visible",
    )


def test_small_frame_plans_one_tile_region():
    state = ViewState.from_shape((32, 48)).with_image_axes(0, 1)

    plan = FramePlanner(internal_tile_shape=(64, 64)).plan(
        target=_target(),
        view_state=state,
        display_shape=(32, 48),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )

    assert plan.layout is DisplayLayout.SINGLE
    assert plan.tile_shape == (32, 48)
    assert plan.active_region_ids == (0,)
    assert plan.regions[0].bounds == (0.0, 0.0, 47.0, 31.0)
    assert plan.regions[0].data_slices == (slice(0, 32), slice(0, 48))


def test_huge_single_plane_plans_internal_tiles():
    state = ViewState.from_shape((40, 60)).with_image_axes(0, 1)

    plan = FramePlanner(internal_tile_shape=(16, 16)).plan(
        target=_target(),
        view_state=state,
        display_shape=(40, 60),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )

    assert plan.layout is DisplayLayout.SINGLE
    assert plan.tile_shape == (16, 16)
    assert len(plan.regions) == 12
    assert plan.regions[0].bounds == (0.0, 0.0, 15.0, 15.0)
    assert plan.regions[-1].bounds == (48.0, 32.0, 59.0, 39.0)
    assert plan.active_region_ids == tuple(range(12))


def test_one_tile_montage_matches_single_region_geometry():
    normal = ViewState.from_shape((12, 14, 1)).with_image_axes(0, 1)
    montage = normal.with_montage_axis(2, columns=1, indices=(0,))
    planner = FramePlanner(internal_tile_shape=(16, 16))

    frame_plan = planner.plan(
        target=_target("same-semantic"),
        view_state=normal,
        display_shape=(12, 14),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )
    montage_plan = planner.plan(
        target=_target("same-semantic"),
        view_state=montage,
        display_shape=(12, 14),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )

    assert frame_plan.semantic_key == montage_plan.semantic_key
    assert frame_plan.active_region_ids == montage_plan.active_region_ids == (0,)
    assert frame_plan.regions[0].bounds == montage_plan.regions[0].bounds
    assert frame_plan.tile_shape == montage_plan.tile_shape == (12, 14)


def test_multi_tile_montage_marks_active_and_near_regions_from_viewport():
    state = ViewState.from_shape((10, 10, 6)).with_image_axes(0, 1).with_montage_axis(2, columns=3, indices=tuple(range(6)))

    plan = FramePlanner().plan(
        target=_target(),
        view_state=state,
        display_shape=(21, 32),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        viewport_shape=(10, 10),
        view_range=((11.0, 20.0), (0.0, 9.0)),
    )

    assert plan.layout is DisplayLayout.MONTAGE
    assert plan.planned_region_ids == tuple(range(6))
    assert plan.active_region_ids == (1,)
    assert set(plan.near_region_ids).issuperset({0, 1, 2, 4})
    assert plan.regions[1].source_index == 1


def test_montage_requires_tiles_that_touch_the_viewport_boundary():
    state = ViewState.from_shape((10, 10, 6)).with_image_axes(0, 1).with_montage_axis(2, columns=3, indices=tuple(range(6)))

    plan = FramePlanner().plan(
        target=_target(),
        view_state=state,
        display_shape=(21, 32),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        viewport_shape=(10, 10),
        view_range=((0.0, 10.0), (0.0, 11.0)),
    )

    assert plan.active_region_ids == (0, 3)


def test_camera_only_retarget_changes_active_regions_not_materialization_key():
    state = ViewState.from_shape((40, 60)).with_image_axes(0, 1)
    planner = FramePlanner(internal_tile_shape=(16, 16))

    left = planner.plan(
        target=_target("semantic", viewport="left"),
        view_state=state,
        display_shape=(40, 60),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        view_range=((0.0, 15.0), (0.0, 15.0)),
    )
    right = planner.plan(
        target=_target("semantic", viewport="right"),
        view_state=state,
        display_shape=(40, 60),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        view_range=((32.0, 59.0), (16.0, 39.0)),
    )

    assert left.semantic_key == right.semantic_key == "semantic"
    assert left.materialization_key == right.materialization_key
    assert left.active_region_ids != right.active_region_ids
