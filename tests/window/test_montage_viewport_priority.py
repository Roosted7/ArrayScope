import os

import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _plan():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((10, 10, 9)).with_montage_axis(
        2, indices=tuple(range(9)), text=":"
    )
    return make_montage_plan(
        state, axis=2, indices=tuple(range(9)), tile_shape=(10, 10), columns=3, gap=1
    )


def _plan_with_columns(columns):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((10, 10, 6)).with_montage_axis(
        2, indices=tuple(range(6)), text=":"
    )
    return make_montage_plan(
        state, axis=2, indices=tuple(range(6)), tile_shape=(10, 10), columns=columns, gap=1
    )


def test_montage_tile_priority_orders_from_viewport_center_outward():
    from arrayscope.display.model.tile_priority import TilePriorityContext, prioritize_tiles

    plan = _plan()
    ordered = prioritize_tiles(
        plan.tiles,
        context=TilePriorityContext.from_tiles(view_range=((0, 32), (0, 32))),
    )

    assert ordered[0].montage_index == 4
    assert {tile.montage_index for tile in ordered[:5]} == {1, 3, 4, 5, 7}


def test_montage_tile_priority_normalizes_by_viewport_aspect():
    from arrayscope.display.model.tile_priority import TilePriorityContext, prioritize_tiles

    plan = _plan()
    ordered = prioritize_tiles(
        plan.tiles,
        context=TilePriorityContext.from_tiles(view_range=((10, 20), (0, 32)), focus=(15, 16)),
    )
    first_indices = [tile.montage_index for tile in ordered[:3]]

    assert first_indices == [4, 1, 7]


def test_montage_tile_priority_accepts_array_inputs_and_invalid_focus():
    import numpy as np

    from arrayscope.display.model.tile_priority import TilePriorityContext, prioritize_tiles

    plan = _plan()
    ordered = prioritize_tiles(
        np.asarray(plan.tiles, dtype=object),
        context=TilePriorityContext.from_tiles(
            view_range=((0, 32), (0, 32)), focus=("not-a-number", 16)
        ),
    )

    assert ordered[0].montage_index == 4


def test_montage_viewport_plan_can_return_prioritized_candidates():
    from arrayscope.window.montage_viewport import MontageViewportPlan

    plan = _plan()
    viewport_plan = MontageViewportPlan(
        2,
        tuple(range(9)),
        (100, 100),
        (10, 10),
        plan,
        ((0, 32), (0, 32)),
        True,
        True,
        priority_focus=(15, 16),
    )

    assert viewport_plan.candidate_tiles(margin_tiles=0, prioritize=True)[0].montage_index == 4


def test_effective_montage_columns_overrides_explicit_when_auto_owned():
    from arrayscope.window.montage_viewport import effective_montage_columns

    columns = effective_montage_columns(
        12,
        (10, 10),
        (40, 120),
        requested_columns=2,
        auto_active=True,
    )

    assert columns != 2


def test_effective_montage_columns_preserves_explicit_after_manual_view():
    from arrayscope.window.montage_viewport import effective_montage_columns

    columns = effective_montage_columns(
        12,
        (10, 10),
        (40, 120),
        requested_columns=2,
    )

    assert columns == 2


def test_effective_montage_columns_reflows_explicit_in_stretch_fit():
    from arrayscope.window.montage_viewport import effective_montage_columns

    columns = effective_montage_columns(
        12,
        (10, 10),
        (40, 120),
        requested_columns=2,
        fit_locked=True,
    )

    assert columns != 2


def test_montage_viewport_intent_observes_without_promoting():
    from arrayscope.window.montage_viewport import montage_viewport_intent

    class Controller:
        def __init__(self):
            self.promoted = False

        def is_fit_locked(self):
            return False

        def is_auto_active(self):
            return self.promoted

        def promote_near_auto(self, _view_range):
            self.promoted = True
            return True

    controller = Controller()
    intent = montage_viewport_intent(controller, ((100.0, 120.0), (100.0, 120.0)))

    assert not intent.auto_active
    assert not intent.auto_like
    assert not controller.promoted


def test_remap_montage_view_range_keeps_tile_anchor_and_manual_zoom_scale():
    from arrayscope.window.montage_viewport import remap_montage_view_range

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    previous_tile = previous.tiles[4]
    focus = (previous_tile.x0 + 4.0, previous_tile.y0 + 6.0)
    view_range = ((focus[0] - 20.0, focus[0] + 20.0), (focus[1] - 10.0, focus[1] + 10.0))

    remapped = remap_montage_view_range(
        previous,
        next_plan,
        view_range,
        (50, 100),
        (50, 50),
        focus=focus,
    )

    assert remapped is not None
    assert remapped[0][1] - remapped[0][0] == 20.0
    assert remapped[1][1] - remapped[1][0] == 20.0
    next_tile = next_plan.tiles[4]
    remapped_center = (
        (remapped[0][0] + remapped[0][1]) * 0.5,
        (remapped[1][0] + remapped[1][1]) * 0.5,
    )
    assert remapped_center == (next_tile.x0 + 4.0, next_tile.y0 + 6.0)


def test_remap_montage_view_range_preserves_screen_zoom_without_layout_change():
    from arrayscope.window.montage_viewport import remap_montage_view_range

    plan = _plan_with_columns(3)
    focus = (15.0, 5.0)

    remapped = remap_montage_view_range(
        plan,
        plan,
        ((5.0, 25.0), (0.0, 10.0)),
        (50, 100),
        (50, 50),
        focus=focus,
    )

    assert remapped is not None
    assert remapped == ((10.0, 20.0), (0.0, 10.0))


def test_remap_montage_view_range_shrink_shows_less_content_at_same_zoom():
    from arrayscope.window.montage_viewport import remap_montage_view_range

    plan = _plan_with_columns(3)

    remapped = remap_montage_view_range(
        plan,
        plan,
        ((5.0, 25.0), (-10.0, 30.0)),
        (100, 100),
        (100, 50),
        focus=(15.0, 10.0),
    )

    assert remapped is not None
    assert remapped[0][1] - remapped[0][0] == 10.0
    assert remapped[1][1] - remapped[1][0] == 40.0


def test_remap_montage_view_range_growth_shows_more_content_when_explicitly_remapped():
    from arrayscope.window.montage_viewport import remap_montage_view_range

    plan = _plan_with_columns(3)

    remapped = remap_montage_view_range(
        plan,
        plan,
        ((5.0, 25.0), (-10.0, 30.0)),
        (100, 100),
        (100, 200),
        focus=(15.0, 10.0),
    )

    assert remapped is not None
    assert remapped[0][1] - remapped[0][0] == 40.0
    assert remapped[1][1] - remapped[1][0] == 40.0


def test_repeated_manual_resize_preserves_units_per_viewport_pixel():
    from arrayscope.window.montage_viewport import remap_montage_view_range

    plan = _plan_with_columns(3)
    view_range = ((5.0, 25.0), (-10.0, 30.0))
    viewport_shape = (100, 100)
    initial_x_units = (view_range[0][1] - view_range[0][0]) / viewport_shape[1]
    initial_y_units = (view_range[1][1] - view_range[1][0]) / viewport_shape[0]

    for next_shape in ((100, 90), (100, 80), (120, 80), (120, 140)):
        view_range = remap_montage_view_range(
            plan,
            plan,
            view_range,
            viewport_shape,
            next_shape,
            focus=(
                (view_range[0][0] + view_range[0][1]) * 0.5,
                (view_range[1][0] + view_range[1][1]) * 0.5,
            ),
        )
        viewport_shape = next_shape
        assert view_range is not None
        assert (view_range[0][1] - view_range[0][0]) / viewport_shape[1] == pytest.approx(
            initial_x_units
        )
        assert (view_range[1][1] - view_range[1][0]) / viewport_shape[0] == pytest.approx(
            initial_y_units
        )


def test_remap_montage_view_range_anchors_to_nearest_tile_when_focus_is_outside_tiles():
    from arrayscope.window.montage_viewport import remap_montage_view_range

    two_columns = _plan_with_columns(2)
    three_columns = _plan_with_columns(3)
    view_range = ((-500.0, 1500.0), (-500.0, 1500.0))

    for _ in range(5):
        focus = (
            (view_range[0][0] + view_range[0][1]) * 0.5,
            (view_range[1][0] + view_range[1][1]) * 0.5,
        )
        next_range = remap_montage_view_range(
            two_columns,
            three_columns,
            view_range,
            (600, 800),
            (600, 900),
            focus=focus,
        )
        assert next_range is not None
        next_range = remap_montage_view_range(
            three_columns,
            two_columns,
            view_range,
            (600, 900),
            (600, 800),
            focus=focus,
        )
        assert next_range is not None


def test_remap_montage_view_range_does_not_follow_scrolled_source_window():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.montage_viewport import remap_montage_view_range

    first_state = ViewState.from_shape((10, 10, 8)).with_montage_axis(
        2, indices=(0, 1, 2, 3), text="0:4"
    )
    next_state = ViewState.from_shape((10, 10, 8)).with_montage_axis(
        2, indices=(4, 5, 6, 7), text="4:8"
    )
    first = make_montage_plan(
        first_state, axis=2, indices=(0, 1, 2, 3), tile_shape=(10, 10), columns=2, gap=1
    )
    next_plan = make_montage_plan(
        next_state, axis=2, indices=(4, 5, 6, 7), tile_shape=(10, 10), columns=2, gap=1
    )

    remapped = remap_montage_view_range(
        first,
        next_plan,
        ((0.0, 12.0), (0.0, 12.0)),
        (100, 100),
        (100, 100),
        focus=(5.0, 5.0),
    )

    assert remapped is None


def test_retarget_montage_viewport_plan_preserves_manual_screen_zoom_on_resize():
    from arrayscope.window.montage_viewport import (
        MontageViewportPlan,
        retarget_montage_viewport_plan,
    )

    plan = _plan_with_columns(3)
    current_range = ((5.0, 25.0), (-10.0, 30.0))
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(6)),
        viewport_shape=(100, 50),
        tile_shape=(10, 10),
        plan=plan,
        view_range=current_range,
        shader_display=False,
        persistent_tile_residency=False,
    )

    reflow = retarget_montage_viewport_plan(
        plan,
        viewport_plan,
        (100, 100),
        fit_locked=False,
        focus=(15.0, 10.0),
    )

    assert reflow.viewport_plan.view_range == ((10.0, 20.0), (-10.0, 30.0))
    assert reflow.view_range_to_apply == ((10.0, 20.0), (-10.0, 30.0))
    assert reflow.last_auto_view_range is None


def test_retarget_montage_viewport_plan_preserves_manual_layout_zoom():
    from arrayscope.window.montage_viewport import (
        MontageViewportPlan,
        retarget_montage_viewport_plan,
    )

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    previous_tile = previous.tiles[4]
    focus = (previous_tile.x0 + 4.0, previous_tile.y0 + 6.0)
    current_range = ((focus[0] - 60.0, focus[0] + 60.0), (focus[1] - 40.0, focus[1] + 40.0))
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(6)),
        viewport_shape=(50, 50),
        tile_shape=(10, 10),
        plan=next_plan,
        view_range=current_range,
        shader_display=False,
        persistent_tile_residency=False,
    )

    reflow = retarget_montage_viewport_plan(
        previous,
        viewport_plan,
        (50, 100),
        fit_locked=False,
        focus=focus,
    )

    view_range = reflow.viewport_plan.view_range
    assert view_range[0][1] - view_range[0][0] == 60.0
    assert view_range[1][1] - view_range[1][0] == 80.0
    assert reflow.last_auto_view_range is None


def test_retarget_montage_viewport_plan_near_auto_refits_to_new_auto():
    from arrayscope.window.montage_viewport import (
        MontageViewportPlan,
        retarget_montage_viewport_plan,
        square_montage_fit_view_range,
    )

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(6)),
        viewport_shape=(50, 50),
        tile_shape=(10, 10),
        plan=next_plan,
        view_range=((0.0, 21.0), (0.0, 32.0)),
        shader_display=False,
        persistent_tile_residency=False,
    )
    expected = square_montage_fit_view_range(next_plan, viewport_plan.viewport_shape)

    reflow = retarget_montage_viewport_plan(
        previous,
        viewport_plan,
        (50, 100),
        fit_locked=False,
        auto_active=True,
        focus=(10.5, 16.0),
    )

    assert reflow.viewport_plan.view_range == expected
    assert reflow.last_auto_view_range == expected


def test_retarget_montage_viewport_plan_near_next_auto_refits_and_records_auto_range():
    from arrayscope.window.montage_viewport import (
        MontageViewportPlan,
        retarget_montage_viewport_plan,
        square_montage_fit_view_range,
    )

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    expected = square_montage_fit_view_range(next_plan, (50, 50))
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(6)),
        viewport_shape=(50, 50),
        tile_shape=(10, 10),
        plan=next_plan,
        view_range=expected,
        shader_display=False,
        persistent_tile_residency=False,
    )

    reflow = retarget_montage_viewport_plan(
        previous,
        viewport_plan,
        (50, 100),
        fit_locked=False,
        auto_active=True,
        focus=(16.0, 10.5),
    )

    assert reflow.viewport_plan.view_range == expected
    assert reflow.view_range_to_apply is None
    assert reflow.last_auto_view_range == expected


def test_retarget_montage_viewport_plan_manual_one_edge_far_does_not_refit():
    from arrayscope.window.montage_viewport import (
        MontageViewportPlan,
        retarget_montage_viewport_plan,
        square_montage_fit_view_range,
    )

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    auto_range = square_montage_fit_view_range(next_plan, (50, 50))
    one_edge_far = (
        (auto_range[0][0], auto_range[0][1] + 5.0),
        auto_range[1],
    )
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(6)),
        viewport_shape=(50, 50),
        tile_shape=(10, 10),
        plan=next_plan,
        view_range=one_edge_far,
        shader_display=False,
        persistent_tile_residency=False,
    )

    reflow = retarget_montage_viewport_plan(
        previous,
        viewport_plan,
        (50, 100),
        fit_locked=False,
        focus=(16.0, 10.5),
    )

    assert reflow.viewport_plan.view_range != auto_range
    assert reflow.last_auto_view_range is None


def test_remap_montage_roi_geometry_moves_same_source_rectangle():
    from arrayscope.core.roi import RoiGeometry, RoiKind
    from arrayscope.window.montage_viewport import remap_montage_roi_geometry

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    previous_tile = previous.tiles[4]
    next_tile = next_plan.tiles[4]
    geometry = RoiGeometry(
        RoiKind.RECTANGLE,
        rect=(previous_tile.x0 + 2.0, previous_tile.y0 + 3.0, 4.0, 5.0),
    )

    remapped = remap_montage_roi_geometry(previous, next_plan, geometry)

    assert remapped is not None
    assert remapped.rect == (
        next_tile.x0 + 2.0,
        next_tile.y0 + 3.0,
        4.0,
        5.0,
    )


def test_remap_montage_roi_geometry_keeps_position_for_scrolled_sources():
    from arrayscope.core.roi import RoiGeometry, RoiKind
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.montage_viewport import remap_montage_roi_geometry

    first_state = ViewState.from_shape((10, 10, 8)).with_montage_axis(
        2, indices=(0, 1, 2, 3), text="0:4"
    )
    next_state = ViewState.from_shape((10, 10, 8)).with_montage_axis(
        2, indices=(4, 5, 6, 7), text="4:8"
    )
    first = make_montage_plan(
        first_state, axis=2, indices=(0, 1, 2, 3), tile_shape=(10, 10), columns=2, gap=1
    )
    next_plan = make_montage_plan(
        next_state, axis=2, indices=(4, 5, 6, 7), tile_shape=(10, 10), columns=2, gap=1
    )
    geometry = RoiGeometry(RoiKind.RECTANGLE, rect=(2.0, 3.0, 4.0, 5.0))

    assert remap_montage_roi_geometry(first, next_plan, geometry) is None


def test_remap_montage_roi_geometry_moves_cross_tile_rectangle_by_anchor():
    from arrayscope.core.roi import RoiGeometry, RoiKind
    from arrayscope.window.montage_viewport import remap_montage_roi_geometry

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    anchor_tile = previous.tiles[4]
    next_anchor = next_plan.tiles[4]
    geometry = RoiGeometry(
        RoiKind.RECTANGLE,
        rect=(anchor_tile.x0 + 8.0, anchor_tile.y0 + 1.0, 16.0, 4.0),
    )

    remapped = remap_montage_roi_geometry(previous, next_plan, geometry)

    assert remapped is not None
    assert remapped.rect == (next_anchor.x0 + 8.0, next_anchor.y0 + 1.0, 16.0, 4.0)


def test_remap_montage_roi_geometry_moves_point_rois_by_source_local_position():
    from arrayscope.core.roi import RoiGeometry, RoiKind
    from arrayscope.window.montage_viewport import remap_montage_roi_geometry

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    first = previous.tiles[0]
    fourth = previous.tiles[4]
    next_first = next_plan.tiles[0]
    next_fourth = next_plan.tiles[4]
    geometry = RoiGeometry(
        RoiKind.LINE,
        points=((first.x0 + 1.0, first.y0 + 2.0), (fourth.x0 + 3.0, fourth.y0 + 4.0)),
    )

    remapped = remap_montage_roi_geometry(previous, next_plan, geometry)

    assert remapped is not None
    assert remapped.points == (
        (next_first.x0 + 1.0, next_first.y0 + 2.0),
        (next_fourth.x0 + 3.0, next_fourth.y0 + 4.0),
    )


def test_montage_layout_reflow_updates_roi_mirror_geometry():
    from types import SimpleNamespace

    from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
    from arrayscope.window.frame_controller import FrameControllerMixin

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    previous_tile = previous.tiles[4]
    next_tile = next_plan.tiles[4]
    selection = RoiSelection(
        "roi-1",
        "ROI 1",
        RoiGeometry(
            RoiKind.RECTANGLE, rect=(previous_tile.x0 + 2.0, previous_tile.y0 + 3.0, 4.0, 5.0)
        ),
    )
    selections = {"roi-1": selection}
    changed = []

    def set_geometry(roi_id, geometry, *, emit, sync_item):
        changed.append((roi_id, geometry, emit, sync_item))
        current = selections[str(roi_id)]
        selections[str(roi_id)] = RoiSelection(
            current.id, current.label, geometry, current.enabled, current.color
        )

    img_view = SimpleNamespace(
        roiSelections=lambda: tuple(selections.values()),
        _set_roi_geometry=set_geometry,
    )
    window = SimpleNamespace(img_view=img_view)
    window.win = window

    FrameControllerMixin._remap_montage_rois_for_layout_reflow(window, previous, next_plan)

    assert changed == [("roi-1", selections["roi-1"].geometry, True, True)]
    assert selections["roi-1"].geometry.rect == (next_tile.x0 + 2.0, next_tile.y0 + 3.0, 4.0, 5.0)


def test_committed_tiled_roi_stats_follow_remapped_layout_geometry():
    import numpy as np
    import pytest

    from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry
    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import (
        CommittedDisplayFrame,
        DisplayFrameKey,
        DisplayTilePayload,
        TiledValueSource,
    )
    from arrayscope.display.montage import MontageTileState
    from arrayscope.window.inspection import InspectionWorkflowMixin
    from arrayscope.window.montage_viewport import remap_montage_roi_geometry

    state = ViewState.from_shape((2, 2, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    previous_tile = previous.tiles[4]
    next_tile = next_plan.tiles[4]
    geometry = RoiGeometry(RoiKind.RECTANGLE, rect=(previous_tile.x0, previous_tile.y0, 2.0, 2.0))
    remapped = remap_montage_roi_geometry(previous, next_plan, geometry)
    assert remapped is not None
    assert remapped.rect == (next_tile.x0, next_tile.y0, 2.0, 2.0)
    payloads = {
        int(tile.montage_index): DisplayTilePayload(
            tile_number=int(tile.montage_index),
            source_index=int(tile.source_index),
            image=np.full((2, 2), int(tile.source_index), dtype=np.float32),
            histogram_data=None,
            source_id=("tile", int(tile.source_index)),
            semantic_data=np.full((2, 2), int(tile.source_index), dtype=np.float32),
            source_shape=(2, 2),
            lod=LodInfo(0, 1, (2, 2), (2, 2), 0),
        )
        for tile in next_plan.tiles
    }
    frame = CommittedDisplayFrame(
        data=None,
        histogram_data=None,
        geometry=DisplayGeometry(
            state,
            next_plan.display_shape,
            montage=next_plan.geometry,
            montage_tile_states=tuple(MontageTileState.LOADED for _tile in next_plan.tiles),
        ),
        levels=(0.0, 5.0),
        histogram_range=(0.0, 5.0),
        key=DisplayFrameKey(("doc",), ("view",), 1),
        value_source=TiledValueSource(payloads),
    )
    window = type(
        "Window",
        (InspectionWorkflowMixin,),
        {"_committed_tiled_frame": lambda self: frame},
    )()

    stats_by_roi, _histograms = window._committed_tiled_roi_values(
        (RoiSelection("roi-1", "ROI 1", remapped),),
        collect_histograms=False,
    )

    assert stats_by_roi["roi-1"][1].count == 4
    assert stats_by_roi["roi-1"][1].mean == pytest.approx(4.0)


def test_square_montage_fit_view_range_follows_viewport_aspect():
    from arrayscope.window.montage_viewport import square_montage_fit_view_range

    plan = _plan_with_columns(3)

    fitted = square_montage_fit_view_range(plan, (100, 50))

    assert fitted[0] == (0.0, 32.0)
    assert fitted[1] == (-21.5, 42.5)


def test_montage_autofit_rescues_manual_view_when_new_layout_is_largely_outside():
    from arrayscope.window.frame_runtime import _should_auto_fit_montage_view

    class ManualController:
        def is_near_auto(self, _view_range):
            return False

    manual_zoom = ((0.0, 10.0), (0.0, 10.0))
    full_range = ((0.0, 32.0), (0.0, 21.0))

    assert _should_auto_fit_montage_view(
        manual_zoom,
        full_range,
        viewport_controller=ManualController(),
        visible_count=1,
        tile_count=6,
    )
    assert _should_auto_fit_montage_view(
        manual_zoom,
        full_range,
        viewport_controller=ManualController(),
        visible_count=0,
        tile_count=6,
    )


def test_montage_autofit_promotes_view_close_to_regular_fit():
    from arrayscope.window.frame_runtime import _should_auto_fit_montage_view

    class NearAutoController:
        def is_auto_active(self):
            return False

        def is_near_auto(self, _view_range):
            return True

    assert _should_auto_fit_montage_view(
        ((-0.5, 32.5), (-0.5, 21.5)),
        ((0.0, 32.0), (0.0, 21.0)),
        viewport_controller=NearAutoController(),
        visible_count=5,
        tile_count=6,
    )


def test_montage_autofit_allows_empty_auto_like_view():
    from arrayscope.window.frame_runtime import _should_auto_fit_montage_view

    class AutoController:
        def is_auto_active(self):
            return True

    assert _should_auto_fit_montage_view(
        ((100.0, 110.0), (100.0, 110.0)),
        ((0.0, 32.0), (0.0, 21.0)),
        viewport_controller=AutoController(),
        visible_count=0,
        tile_count=6,
    )


def test_montage_autofit_rejects_far_zoomed_out_manual_view():
    from arrayscope.window.frame_runtime import _should_auto_fit_montage_view

    class ManualController:
        def is_near_auto(self, _view_range):
            return False

    assert not _should_auto_fit_montage_view(
        ((-100.0, 132.0), (-100.0, 121.0)),
        ((0.0, 32.0), (0.0, 21.0)),
        viewport_controller=ManualController(),
        visible_count=6,
        tile_count=6,
    )


def test_montage_autofit_allows_exact_near_full_manual_alignment():
    from arrayscope.window.frame_runtime import _should_auto_fit_montage_view

    class ManualController:
        def is_near_auto(self, _view_range):
            return False

    full_range = ((0.0, 32.0), (0.0, 21.0))

    assert _should_auto_fit_montage_view(
        ((-0.01, 32.01), (0.0, 21.0)),
        full_range,
        viewport_controller=ManualController(),
        visible_count=6,
        tile_count=6,
    )


def test_retarget_layout_reflow_keeps_far_zoomed_out_manual_range_manual():
    from types import SimpleNamespace

    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.montage_viewport import (
        MontageViewportPlan,
        square_montage_fit_view_range,
    )

    class ManualController:
        def is_fit_locked(self):
            return False

        def is_near_auto(self, _view_range):
            return False

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    current_range = ((-100.0, 132.0), (-100.0, 121.0))
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(6)),
        viewport_shape=(50, 50),
        tile_shape=(10, 10),
        plan=next_plan,
        view_range=current_range,
        shader_display=False,
        persistent_tile_residency=False,
    )
    applied = []
    window = SimpleNamespace(
        img_view=SimpleNamespace(viewport_controller=ManualController()),
        _set_montage_view_range=lambda view_range: applied.append(view_range),
    )
    window.win = window
    session = SimpleNamespace(plan=previous, viewport_shape=(50, 100))

    retargeted = FrameControllerMixin._retargeted_montage_viewport_plan(
        window, session, viewport_plan
    )

    assert retargeted.view_range != square_montage_fit_view_range(
        next_plan, viewport_plan.viewport_shape
    )
    assert applied != [square_montage_fit_view_range(next_plan, viewport_plan.viewport_shape)]


def test_released_viewport_continuity_does_not_skip_manual_resize_reflow():
    from types import SimpleNamespace

    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.montage_viewport import MontageViewportPlan
    from arrayscope.window.viewport_continuity import ViewportContinuityTransaction

    class ManualController:
        def is_fit_locked(self):
            return False

        def is_auto_active(self):
            return False

    plan = _plan_with_columns(3)
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(6)),
        viewport_shape=(50, 50),
        tile_shape=(10, 10),
        plan=plan,
        view_range=((0.0, 20.0), (0.0, 20.0)),
        shader_display=False,
        persistent_tile_residency=False,
    )
    applied = []
    continuity = ViewportContinuityTransaction(
        view_range=((-100.0, 100.0), (-100.0, 100.0)),
        range_applied=True,
        released=True,
    )
    window = SimpleNamespace(
        img_view=SimpleNamespace(viewport_controller=ManualController()),
        _set_montage_view_range=lambda view_range: applied.append(view_range),
        _viewport_continuity_transaction=lambda: continuity,
        _frame_session=SimpleNamespace(plan=plan),
    )
    window.win = window
    session = SimpleNamespace(plan=plan, viewport_shape=(50, 100))

    retargeted = FrameControllerMixin._retargeted_montage_viewport_plan(
        window, session, viewport_plan
    )

    assert retargeted.view_range == ((2.5, 12.5), (0.0, 20.0))
    assert applied == [((2.5, 12.5), (0.0, 20.0))]


def test_retarget_layout_reflow_refits_when_near_previous_auto_range():
    from types import SimpleNamespace

    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.montage_viewport import (
        MontageViewportPlan,
        square_montage_fit_view_range,
    )

    class AutoController:
        def __init__(self):
            self.last_auto_view_range = None

        def is_fit_locked(self):
            return False

        def is_auto_active(self):
            return True

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    controller = AutoController()
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(6)),
        viewport_shape=(50, 50),
        tile_shape=(10, 10),
        plan=next_plan,
        view_range=((0.0, 21.0), (0.0, 32.0)),
        shader_display=False,
        persistent_tile_residency=False,
    )
    applied = []
    window = SimpleNamespace(
        img_view=SimpleNamespace(viewport_controller=controller),
        _set_montage_view_range=lambda view_range: applied.append(view_range),
    )
    window.win = window
    session = SimpleNamespace(plan=previous, viewport_shape=(50, 100))
    expected = square_montage_fit_view_range(next_plan, viewport_plan.viewport_shape)

    retargeted = FrameControllerMixin._retargeted_montage_viewport_plan(
        window, session, viewport_plan
    )

    assert retargeted.view_range == expected
    assert applied == [expected]
    assert controller.last_auto_view_range == expected


def test_retarget_layout_reflow_refits_only_when_near_next_auto_range():
    from types import SimpleNamespace

    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.montage_viewport import (
        MontageViewportPlan,
        square_montage_fit_view_range,
    )

    class AutoController:
        def __init__(self):
            self.last_auto_view_range = None

        def is_fit_locked(self):
            return False

        def is_auto_active(self):
            return True

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    controller = AutoController()
    expected = square_montage_fit_view_range(next_plan, (50, 50))
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(6)),
        viewport_shape=(50, 50),
        tile_shape=(10, 10),
        plan=next_plan,
        view_range=expected,
        shader_display=False,
        persistent_tile_residency=False,
    )
    applied = []
    window = SimpleNamespace(
        img_view=SimpleNamespace(viewport_controller=controller),
        _set_montage_view_range=lambda view_range: applied.append(view_range),
    )
    window.win = window
    session = SimpleNamespace(plan=previous, viewport_shape=(50, 100))

    retargeted = FrameControllerMixin._retargeted_montage_viewport_plan(
        window, session, viewport_plan
    )

    assert retargeted.view_range == expected
    assert applied == []
    assert controller.last_auto_view_range == expected


def test_retarget_layout_reflow_refits_when_auto_active_even_if_far_from_next_auto():
    from types import SimpleNamespace

    from arrayscope.window.frame_controller import FrameControllerMixin
    from arrayscope.window.montage_viewport import (
        MontageViewportPlan,
        square_montage_fit_view_range,
    )

    class AutoController:
        def __init__(self):
            self.last_auto_view_range = None

        def is_fit_locked(self):
            return False

        def is_auto_active(self):
            return True

        def is_near_auto(self, _view_range):
            return False

    previous = _plan_with_columns(2)
    next_plan = _plan_with_columns(3)
    controller = AutoController()
    expected = square_montage_fit_view_range(next_plan, (50, 50))
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(6)),
        viewport_shape=(50, 50),
        tile_shape=(10, 10),
        plan=next_plan,
        view_range=((100.0, 120.0), (100.0, 120.0)),
        shader_display=False,
        persistent_tile_residency=False,
    )
    applied = []
    window = SimpleNamespace(
        img_view=SimpleNamespace(viewport_controller=controller),
        _set_montage_view_range=lambda view_range: applied.append(view_range),
    )
    window.win = window
    session = SimpleNamespace(plan=previous, viewport_shape=(50, 100))

    retargeted = FrameControllerMixin._retargeted_montage_viewport_plan(
        window, session, viewport_plan
    )

    assert retargeted.view_range == expected
    assert applied == [expected]
    assert controller.last_auto_view_range == expected


def test_montage_autofit_allows_auto_active_even_when_visible_fraction_is_high():
    from arrayscope.window.frame_runtime import _should_auto_fit_montage_view

    class AutoController:
        def is_auto_active(self):
            return True

        def is_near_auto(self, _view_range):
            return False

    assert _should_auto_fit_montage_view(
        ((100.0, 120.0), (100.0, 120.0)),
        ((0.0, 32.0), (0.0, 21.0)),
        viewport_controller=AutoController(),
        visible_count=6,
        tile_count=6,
    )


def test_montage_live_layout_reflow_skips_autofit_helper():
    from types import SimpleNamespace

    from arrayscope.window.frame_controller import FrameControllerMixin

    geometry = _plan_with_columns(3).geometry
    window = SimpleNamespace(_montage_live_layout_reflow=True)
    window.win = window

    assert not FrameControllerMixin._maybe_auto_fit_montage_tiles(window, geometry)


def test_child_layout_resize_reopens_authoritative_viewport_shape_restore():
    from types import SimpleNamespace

    from arrayscope.window.frame_controller import FrameControllerMixin

    calls = []
    window = SimpleNamespace(
        _closing=False,
        _viewport_continuity_shape_target=lambda: (739, 1247),
        _restore_viewport_continuity_shape_after_layout=lambda: calls.append("restore"),
        _active_viewport_continuity_range=lambda: None,
    )
    window.win = window

    FrameControllerMixin._on_image_viewport_resized(window)

    assert calls == ["restore"]


def test_active_viewport_continuity_skips_autofit_helper():
    from types import SimpleNamespace

    from arrayscope.window.frame_controller import FrameControllerMixin

    geometry = _plan_with_columns(3).geometry
    window = SimpleNamespace(
        _montage_live_layout_reflow=False,
        _pending_viewport_continuity_range=lambda: None,
        _active_viewport_continuity_range=lambda: ((10.0, 20.0), (30.0, 40.0)),
    )
    window.win = window

    assert not FrameControllerMixin._maybe_auto_fit_montage_tiles(window, geometry)


def test_montage_autofit_signature_ignores_layout_only_reflow():
    from arrayscope.window.frame_runtime import (
        _montage_autofit_scope_grew,
        _montage_autofit_signature,
    )

    previous = _plan_with_columns(2).geometry
    next_geometry = _plan_with_columns(3).geometry

    assert not _montage_autofit_scope_grew(
        _montage_autofit_signature(previous),
        _montage_autofit_signature(next_geometry),
    )


def test_montage_priority_focus_uses_semantic_hover_focus():
    from types import SimpleNamespace

    from arrayscope.window.montage_viewport import montage_priority_focus

    frame = SimpleNamespace(key=("frame", 7))
    window = SimpleNamespace(
        _committed_display_frame=frame,
        _last_image_hover_focus=(3.5, 4.25),
        _last_image_hover_focus_frame_key=frame.key,
    )
    window.win = window

    assert montage_priority_focus(window, ((0.0, 10.0), (0.0, 10.0))) == (3.5, 4.25)


def test_montage_priority_focus_falls_back_to_nearest_center_tile():
    from types import SimpleNamespace

    from arrayscope.window.montage_viewport import montage_priority_focus

    plan = _plan_with_columns(3)
    window = SimpleNamespace(_frame_session=SimpleNamespace(plan=plan))
    window.win = window

    assert montage_priority_focus(window, ((0.0, 20.0), (0.0, 20.0))) == (5.0, 5.0)


def test_montage_priority_focus_ignores_backend_priority_policy():
    from types import SimpleNamespace

    from arrayscope.window.montage_viewport import montage_priority_focus

    plan = _plan_with_columns(3)
    backend_controller = SimpleNamespace(priority_focus=lambda _view_range: (19.0, 19.0))
    window = SimpleNamespace(
        _frame_session=SimpleNamespace(plan=plan),
        img_view=SimpleNamespace(viewport_controller=backend_controller),
    )
    window.win = window

    assert montage_priority_focus(window, ((0.0, 20.0), (0.0, 20.0))) == (5.0, 5.0)


def test_montage_priority_focus_rejects_hover_from_previous_frame():
    from types import SimpleNamespace

    from arrayscope.window.montage_viewport import montage_priority_focus

    plan = _plan_with_columns(3)
    window = SimpleNamespace(
        _committed_display_frame=SimpleNamespace(key=("frame", 8)),
        _frame_session=SimpleNamespace(plan=plan),
        _last_image_hover_focus=(19.0, 19.0),
        _last_image_hover_focus_frame_key=("frame", 7),
    )
    window.win = window

    assert montage_priority_focus(window, ((0.0, 20.0), (0.0, 20.0))) == (5.0, 5.0)


def test_frame_session_key_excludes_effective_columns():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.montage_viewport import MontageViewportPlan, frame_session_key

    state = ViewState.from_shape((10, 10, 12)).with_image_axes(0, 1).with_montage_axis(2, columns=2)
    left = make_montage_plan(
        state, axis=2, indices=tuple(range(12)), tile_shape=(10, 10), columns=2
    )
    right = make_montage_plan(
        state, axis=2, indices=tuple(range(12)), tile_shape=(10, 10), columns=4
    )

    def plan(montage_plan):
        return MontageViewportPlan(
            axis=2,
            all_indices=tuple(range(12)),
            viewport_shape=(40, 120),
            tile_shape=(10, 10),
            plan=montage_plan,
            view_range=None,
            shader_display=False,
            persistent_tile_residency=False,
        )

    assert frame_session_key("doc", state, plan(left), None) == frame_session_key(
        "doc", state, plan(right), None
    )
