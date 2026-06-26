import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _plan():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((10, 10, 9)).with_montage_axis(2, indices=tuple(range(9)), text=":")
    return make_montage_plan(state, axis=2, indices=tuple(range(9)), tile_shape=(10, 10), columns=3, gap=1)


def _plan_with_columns(columns):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((10, 10, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    return make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(10, 10), columns=columns, gap=1)


def test_montage_tile_priority_orders_from_viewport_center_outward():
    from arrayscope.window.montage_viewport import prioritize_montage_tiles

    plan = _plan()
    ordered = prioritize_montage_tiles(
        plan.tiles,
        view_range=((0, 32), (0, 32)),
        focus=None,
    )

    assert ordered[0].montage_index == 4
    assert {tile.montage_index for tile in ordered[:5]} == {1, 3, 4, 5, 7}


def test_montage_tile_priority_normalizes_by_viewport_aspect():
    from arrayscope.window.montage_viewport import prioritize_montage_tiles

    plan = _plan()
    ordered = prioritize_montage_tiles(
        plan.tiles,
        view_range=((10, 20), (0, 32)),
        focus=(15, 16),
    )
    first_indices = [tile.montage_index for tile in ordered[:3]]

    assert first_indices == [4, 1, 7]


def test_montage_tile_priority_accepts_array_inputs_and_invalid_focus():
    import numpy as np

    from arrayscope.window.montage_viewport import prioritize_montage_tiles

    plan = _plan()
    ordered = prioritize_montage_tiles(
        np.asarray(plan.tiles, dtype=object),
        view_range=((0, 32), (0, 32)),
        focus=("not-a-number", 16),
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


def test_effective_montage_columns_overrides_explicit_when_near_auto():
    from arrayscope.window.montage_viewport import effective_montage_columns

    columns = effective_montage_columns(
        12,
        (10, 10),
        (40, 120),
        requested_columns=2,
        near_auto=True,
    )

    assert columns != 2


def test_effective_montage_columns_preserves_explicit_after_manual_view():
    from arrayscope.window.montage_viewport import effective_montage_columns

    columns = effective_montage_columns(
        12,
        (10, 10),
        (40, 120),
        requested_columns=2,
        near_auto=False,
    )

    assert columns == 2


def test_effective_montage_columns_preserves_explicit_in_stretch_fit():
    from arrayscope.window.montage_viewport import effective_montage_columns

    columns = effective_montage_columns(
        12,
        (10, 10),
        (40, 120),
        requested_columns=2,
        near_auto=True,
        fit_locked=True,
    )

    assert columns == 2


def test_remap_montage_view_range_keeps_tile_anchor_and_zoom_density():
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


def test_remap_montage_view_range_preserves_zoom_density_without_layout_change():
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


def test_square_montage_fit_view_range_follows_viewport_aspect():
    from arrayscope.window.montage_viewport import square_montage_fit_view_range

    plan = _plan_with_columns(3)

    fitted = square_montage_fit_view_range(plan, (100, 50))

    assert fitted[0] == (0.0, 32.0)
    assert fitted[1] == (-21.5, 42.5)


def test_montage_autofit_skips_manual_zoom_unless_view_is_empty():
    from arrayscope.window.montage_renderer import _should_auto_fit_montage_view

    class ManualController:
        def is_near_auto(self, _view_range):
            return False

    manual_zoom = ((0.0, 10.0), (0.0, 10.0))
    full_range = ((0.0, 32.0), (0.0, 21.0))

    assert not _should_auto_fit_montage_view(
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


def test_montage_session_key_excludes_effective_columns():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.montage_viewport import MontageViewportPlan, montage_session_key

    state = ViewState.from_shape((10, 10, 12)).with_image_axes(0, 1).with_montage_axis(2, columns=2)
    left = make_montage_plan(state, axis=2, indices=tuple(range(12)), tile_shape=(10, 10), columns=2)
    right = make_montage_plan(state, axis=2, indices=tuple(range(12)), tile_shape=(10, 10), columns=4)

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

    assert montage_session_key("doc", state, plan(left), None) == montage_session_key("doc", state, plan(right), None)
