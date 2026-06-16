import pytest
from hypothesis import given, strategies as st

from arrayscope.core.view_state import ViewState
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.montage import MontageTileState


def state_for(shape, image_axes=(0, 1), line_axis=2, slices=None):
    return ViewState(
        ndim=len(shape),
        shape=tuple(shape),
        image_axes=image_axes,
        line_axis=line_axis,
        slice_indices=tuple(slices if slices is not None else (0,) * len(shape)),
        axis_flipped=(False,) * len(shape),
        axis_fftshifted=(False,) * len(shape),
    )


def test_2d_plain_image_point_maps_to_array_index():
    state = state_for((3, 4), line_axis=0)
    geometry = DisplayGeometry(state, (3, 4))

    mapping = geometry.view_point_to_array_index(2, 1)

    assert mapping.array_index == (1, 2)
    assert mapping.local_x == 2
    assert mapping.local_y == 1


def test_display_point_uses_floor_pixel_cell_mapping():
    state = state_for((3, 4), line_axis=0)
    geometry = DisplayGeometry(state, (3, 4))

    assert geometry.view_point_to_array_index(2.9, 1.8).array_index == (1, 2)
    assert geometry.view_point_to_array_index(-0.1, 1.0) is None


def test_3d_sliced_image_preserves_non_display_slice():
    state = state_for((2, 3, 4), image_axes=(1, 2), line_axis=0, slices=(1, 0, 0))
    geometry = DisplayGeometry(state, (3, 4))

    assert geometry.view_point_to_array_index(3, 2).array_index == (1, 2, 3)


def test_reversed_image_axes_use_y_then_x_axis_roles():
    state = state_for((2, 3, 4, 5), image_axes=(3, 1), line_axis=2, slices=(1, 0, 0, 0))
    geometry = DisplayGeometry(state, (5, 3))

    assert geometry.view_point_to_array_index(2, 4).array_index == (1, 2, 0, 4)


def test_image_axis_subrange_maps_display_index_to_actual_axis_index():
    state = state_for((5, 6), line_axis=0).with_axis_range(0, (0, 2, 4), "0:2:100")
    geometry = DisplayGeometry(state, (3, 6))

    assert geometry.view_point_to_array_index(5, 2).array_index == (4, 5)
    assert geometry.view_point_to_array_index(5, 3) is None


def test_montage_point_maps_tile_and_local_position():
    state = state_for((2, 3, 4), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=(0, 1, 3), text=":")
    montage = MontageGeometry(indices=(0, 1, 3), tile_shape=(2, 3), columns=2, rows=2, gap=1)
    geometry = DisplayGeometry(state, (5, 7), montage=montage)

    mapping = geometry.view_point_to_array_index(5, 1)

    assert mapping.tile_number == 1
    assert mapping.montage_axis == 2
    assert mapping.montage_index == 1
    assert mapping.local_x == 1
    assert mapping.local_y == 1
    assert mapping.array_index == (1, 1, 1)


def test_context_for_montage_point_labels_tiled_axis_once():
    state = state_for((2, 3, 4), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=(0, 1, 3), text=":")
    montage = MontageGeometry(indices=(0, 1, 3), tile_shape=(2, 3), columns=2, rows=2, gap=1)
    geometry = DisplayGeometry(state, (5, 7), montage=montage)

    context = geometry.context_for_view_point(5, 1)

    assert context.value_prefix == "(1, 1)"
    assert context.context_text == "d2=1"


def test_montage_gaps_and_missing_last_tile_return_none():
    state = state_for((2, 3, 4), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=(0, 1, 2), text=":")
    montage = MontageGeometry(indices=(0, 1, 2), tile_shape=(2, 3), columns=2, rows=2, gap=1)
    geometry = DisplayGeometry(state, (5, 7), montage=montage)

    assert geometry.view_point_to_array_index(3, 1) is None
    assert geometry.view_point_to_array_index(5, 4) is None


def test_clamp_view_point_uses_nearest_valid_montage_tile():
    state = state_for((2, 3, 4), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=(0, 1), text=":")
    montage = MontageGeometry(indices=(0, 1), tile_shape=(2, 3), columns=2, rows=1, gap=1)
    geometry = DisplayGeometry(state, (2, 7), montage=montage)

    assert geometry.clamp_view_point(3, 1) in {(2, 1), (4, 1)}


def test_profile_states_under_montage_include_tile_slice_and_local_xy():
    state = state_for((2, 3, 4), image_axes=(0, 1), line_axis=2).with_montage_axis(2, indices=(0, 2), text=":")
    montage = MontageGeometry(indices=(0, 2), tile_shape=(2, 3), columns=2, rows=1, gap=1)
    geometry = DisplayGeometry(state, (2, 7), montage=montage)

    states = geometry.view_point_to_profile_states(5, 1, (1, 2))

    assert tuple(profile_state.line_axis for profile_state in states) == (1, 2)
    assert states[0].slice_indices == (1, 0, 2)
    assert states[1].slice_indices == (1, 1, 2)


def test_montage_canvas_origin_maps_world_point_to_global_source_index():
    state = state_for((2, 3, 20), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=tuple(range(20)), text=":")
    montage = MontageGeometry(indices=tuple(range(20)), tile_shape=(2, 3), columns=5, rows=4, gap=1)
    geometry = DisplayGeometry(state, (2, 11), montage=montage, montage_origin_x=0, montage_origin_y=6)

    mapping = geometry.view_point_to_array_index(1, 7)

    assert mapping.montage_index == 10
    assert mapping.array_index == (1, 1, 10)
    assert mapping.canvas_x == 1
    assert mapping.canvas_y == 1


def test_montage_canvas_origin_change_does_not_change_world_to_array_mapping():
    state = state_for((2, 3, 20), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=tuple(range(20)), text=":")
    montage = MontageGeometry(indices=tuple(range(20)), tile_shape=(2, 3), columns=5, rows=4, gap=1)
    first = DisplayGeometry(state, (2, 11), montage=montage, montage_origin_x=0, montage_origin_y=0)
    shifted = DisplayGeometry(state, (2, 11), montage=montage, montage_origin_x=0, montage_origin_y=6)

    first_mapping = first.view_point_to_array_index(1, 7)
    shifted_mapping = shifted.view_point_to_array_index(1, 7)

    assert first_mapping.array_index == shifted_mapping.array_index == (1, 1, 10)
    assert first_mapping.canvas_y == 7
    assert shifted_mapping.canvas_y == 1


def test_montage_status_for_loaded_tile_allows_mapping():
    state = state_for((2, 3, 3), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=(0, 1, 2), text=":")
    montage = MontageGeometry(indices=(0, 1, 2), tile_shape=(2, 3), columns=3, rows=1, gap=1)
    geometry = DisplayGeometry(state, (2, 11), montage=montage, montage_tile_states=(MontageTileState.LOADED,) * 3)

    status = geometry.view_point_to_tile_point(1, 1)
    mapping = geometry.view_point_to_array_index(1, 1)

    assert status.kind == "loaded"
    assert mapping.array_index == (1, 1, 0)


def test_montage_status_for_loading_tile_blocks_hover_array_mapping_but_allows_demand_mapping():
    state = state_for((2, 3, 3), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=(0, 1, 2), text=":")
    montage = MontageGeometry(indices=(0, 1, 2), tile_shape=(2, 3), columns=3, rows=1, gap=1)
    geometry = DisplayGeometry(
        state,
        (2, 11),
        montage=montage,
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADING, MontageTileState.LOADED),
    )

    assert geometry.view_point_to_tile_point(4, 1).kind == "loading"
    assert geometry.view_point_to_array_index(4, 1) is None
    assert geometry.view_point_to_array_index(4, 1, require_loaded=False).array_index == (1, 0, 1)


def test_montage_status_for_skipped_tile_blocks_array_mapping():
    state = state_for((2, 3, 3), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=(0, 1, 2), text=":")
    montage = MontageGeometry(indices=(0, 1, 2), tile_shape=(2, 3), columns=3, rows=1, gap=1)
    geometry = DisplayGeometry(
        state,
        (2, 11),
        montage=montage,
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.SKIPPED, MontageTileState.LOADED),
    )

    assert geometry.view_point_to_tile_point(4, 1).kind == "skipped"
    assert geometry.view_point_to_array_index(4, 1) is None


def test_montage_status_for_gap_reports_gap():
    state = state_for((2, 3, 3), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=(0, 1, 2), text=":")
    montage = MontageGeometry(indices=(0, 1, 2), tile_shape=(2, 3), columns=3, rows=1, gap=1)
    geometry = DisplayGeometry(state, (2, 11), montage=montage, montage_tile_states=(MontageTileState.LOADED,) * 3)

    assert geometry.view_point_to_tile_point(3, 1).kind == "gap"
    assert geometry.view_point_to_array_index(3, 1) is None


def test_montage_canvas_origin_applies_to_tile_status():
    state = state_for((2, 3, 20), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=tuple(range(20)), text=":")
    montage = MontageGeometry(indices=tuple(range(20)), tile_shape=(2, 3), columns=5, rows=4, gap=1)
    states = tuple(MontageTileState.UNLOADED for _ in range(10)) + (MontageTileState.LOADING,) + tuple(MontageTileState.LOADED for _ in range(9))
    geometry = DisplayGeometry(state, (2, 11), montage=montage, montage_origin_x=0, montage_origin_y=6, montage_tile_states=states)

    status = geometry.view_point_to_tile_point(1, 7)

    assert status.kind == "loading"
    assert status.source_index == 10


def test_montage_canvas_origin_gap_returns_none():
    state = state_for((2, 3, 20), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=tuple(range(20)), text=":")
    montage = MontageGeometry(indices=tuple(range(20)), tile_shape=(2, 3), columns=5, rows=4, gap=1)
    geometry = DisplayGeometry(state, (2, 11), montage=montage, montage_origin_x=0, montage_origin_y=6)

    assert geometry.view_point_to_array_index(3, 7) is None


def test_montage_canvas_origin_profile_state_uses_global_tile_slice():
    state = state_for((2, 3, 20), image_axes=(0, 1), line_axis=2).with_montage_axis(2, indices=tuple(range(20)), text=":")
    montage = MontageGeometry(indices=tuple(range(20)), tile_shape=(2, 3), columns=5, rows=4, gap=1)
    geometry = DisplayGeometry(state, (2, 11), montage=montage, montage_origin_x=0, montage_origin_y=6)

    states = geometry.view_point_to_profile_states(9, 7, (1, 2))

    assert states[0].slice_indices == (1, 0, 12)
    assert states[1].slice_indices == (1, 1, 12)


def test_montage_canvas_clamp_returns_world_point():
    state = state_for((2, 3, 20), image_axes=(0, 1), line_axis=1).with_montage_axis(2, indices=tuple(range(20)), text=":")
    montage = MontageGeometry(indices=tuple(range(20)), tile_shape=(2, 3), columns=5, rows=4, gap=1)
    geometry = DisplayGeometry(state, (2, 7), montage=montage, montage_origin_x=4, montage_origin_y=6)

    assert geometry.clamp_view_point(3, 7) in {(2, 7), (4, 7)}


@given(
    height=st.integers(1, 8),
    width=st.integers(1, 8),
    x=st.integers(-2, 10),
    y=st.integers(-2, 10),
    use_y_range=st.booleans(),
    use_x_range=st.booleans(),
)
def test_display_point_mapping_property_bounds_and_ranges(height, width, x, y, use_y_range, use_x_range):
    state = state_for((height, width), line_axis=0)
    y_range = tuple(range(0, height, 2)) if use_y_range else None
    x_range = tuple(range(0, width, 2)) if use_x_range else None
    if y_range:
        state = state.with_axis_range(0, y_range, "range")
    if x_range:
        state = state.with_axis_range(1, x_range, "range")
    display_shape = (len(y_range) if y_range else height, len(x_range) if x_range else width)
    geometry = DisplayGeometry(state, display_shape)

    mapping = geometry.view_point_to_array_index(x, y)
    if mapping is None:
        assert x < 0 or y < 0 or x >= display_shape[1] or y >= display_shape[0]
        return

    assert 0 <= mapping.array_index[0] < height
    assert 0 <= mapping.array_index[1] < width
    if y_range:
        assert mapping.array_index[0] == y_range[mapping.local_y]
    if x_range:
        assert mapping.array_index[1] == x_range[mapping.local_x]
