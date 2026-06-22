import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _plan():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((10, 10, 9)).with_montage_axis(2, indices=tuple(range(9)), text=":")
    return make_montage_plan(state, axis=2, indices=tuple(range(9)), tile_shape=(10, 10), columns=3, gap=1)


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
