import pytest
from hypothesis import given, strategies as st

from arrayscope.gpu import ChunkGrid


def test_grid_shape_and_edge_clipping():
    grid = ChunkGrid(array_shape=(300, 130), chunk_shape=(128, 128))
    assert grid.grid_shape() == (3, 2)
    assert grid.chunk_count() == 6
    assert grid.shape_at((256, 128)) == (44, 2)
    assert grid.shape_at((0, 0)) == (128, 128)


def test_origin_for_index_and_bounds():
    grid = ChunkGrid(array_shape=(300, 130), chunk_shape=(128, 128))
    assert grid.origin_for_index((0, 0)) == (0, 0)
    assert grid.origin_for_index((299, 129)) == (256, 128)
    with pytest.raises(IndexError):
        grid.origin_for_index((300, 0))
    with pytest.raises(IndexError):
        grid.shape_at((5, 0))


def test_window_selects_intersecting_chunks_only():
    grid = ChunkGrid(array_shape=(512,), chunk_shape=(128,))
    assert grid.origins_for_window(((100, 200),)) == ((0,), (128,))
    assert grid.origins_for_window(((128, 256),)) == ((128,),)
    assert grid.origins_for_window(((10, 10),)) == ()


def test_slice_window_shift_requests_at_most_one_boundary_chunk():
    """The ADR 0055 motivating example: X=100:200 -> 101:201."""

    grid = ChunkGrid(array_shape=(1024,), chunk_shape=(128,))
    old = ((100, 200),)
    new = ((101, 201),)
    delta = grid.window_delta(old, new)
    # 100:200 and 101:201 both live inside chunks [0, 128) and [128, 256):
    # a one-step shift is pure reuse, zero uploads.
    assert delta.added == ()
    assert delta.dropped == ()
    assert set(delta.kept) == {(0,), (128,)}

    # Shifting across a chunk boundary requests exactly the one new chunk.
    crossing = grid.window_delta(((100, 200),), ((130, 230),))
    assert crossing.added == ()  # both windows still within chunks 0 and 1
    crossing = grid.window_delta(((100, 200),), ((200, 300),))
    assert crossing.added == ((256,),)
    assert crossing.dropped == ((0,),)
    assert crossing.kept == ((128,),)


def test_window_delta_cold_start_is_all_added():
    grid = ChunkGrid(array_shape=(256, 256), chunk_shape=(128, 128))
    delta = grid.window_delta(None, ((0, 256), (0, 256)))
    assert delta.kept == () and delta.dropped == ()
    assert set(delta.added) == set(grid.origins())


def test_empty_and_degenerate_grids():
    assert ChunkGrid(array_shape=(0, 128), chunk_shape=(64, 64)).origins() == ()
    grid = ChunkGrid(array_shape=(64,), chunk_shape=(128,))
    assert grid.grid_shape() == (1,)
    assert grid.shape_at((0,)) == (64,)


@st.composite
def grid_and_windows(draw):
    rank = draw(st.integers(min_value=1, max_value=3))
    array_shape = tuple(draw(st.integers(min_value=1, max_value=200)) for _ in range(rank))
    chunk_shape = tuple(draw(st.integers(min_value=1, max_value=64)) for _ in range(rank))

    def window(extent):
        start = draw(st.integers(min_value=0, max_value=extent))
        stop = draw(st.integers(min_value=start, max_value=extent))
        return (start, stop)

    old = tuple(window(extent) for extent in array_shape)
    new = tuple(window(extent) for extent in array_shape)
    return ChunkGrid(array_shape, chunk_shape), old, new


@given(grid_and_windows())
def test_window_delta_laws(case):
    grid, old, new = case
    old_set = set(grid.origins_for_window(old))
    new_set = set(grid.origins_for_window(new))
    delta = grid.window_delta(old, new)
    assert set(delta.kept) == old_set & new_set
    assert set(delta.added) == new_set - old_set
    assert set(delta.dropped) == old_set - new_set
    # Every reported origin is a real chunk origin with a valid shape.
    for origin in (*delta.kept, *delta.added, *delta.dropped):
        assert all(dim > 0 for dim in grid.shape_at(origin))


@given(grid_and_windows())
def test_window_chunks_cover_window_exactly(case):
    grid, _old, window = case
    origins = grid.origins_for_window(window)
    if any(stop <= start for start, stop in window):
        assert origins == ()
        return
    # Every index in the window falls inside exactly one selected chunk.
    corners = [
        tuple(start for start, _stop in window),
        tuple(stop - 1 for _start, stop in window),
    ]
    for corner in corners:
        owner = grid.origin_for_index(corner)
        assert owner in origins
    # And every selected chunk genuinely intersects the window.
    for origin in origins:
        shape = grid.shape_at(origin)
        for (start, stop), o, s in zip(window, origin, shape):
            assert o < stop and o + s > start
