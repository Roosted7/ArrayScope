import numpy as np

from arrayscope.display.lod import apply_tile_gutter, build_tile_lod_pyramid, inner_uv_for_gutter, select_lod_factor


def test_lod_selection_keeps_factor_one_when_zoomed_in():
    assert select_lod_factor(((0.0, 64.0), (0.0, 64.0)), (128, 128), (64, 64)) == 1


def test_lod_selection_chooses_power_of_two_for_zoomed_out_view():
    factor = select_lod_factor(((0.0, 1024.0), (0.0, 1024.0)), (128, 128), (64, 64))

    assert factor in {4, 8}
    assert 1.0 <= 8.0 / factor <= 2.0


def test_box_downsampling_handles_odd_shapes_and_nonfinite_values():
    data = np.array(
        [
            [1.0, 3.0, np.nan],
            [np.inf, 5.0, 7.0],
            [9.0, 11.0, 13.0],
        ],
        dtype=np.float32,
    )

    pyramid = build_tile_lod_pyramid(data, max_level=1)

    assert len(pyramid) == 2
    np.testing.assert_allclose(pyramid[1], np.array([[3.0, 7.0], [10.0, 13.0]], dtype=np.float32))


def test_gutter_duplicates_edge_texels_exactly():
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)

    guttered = apply_tile_gutter(data, gutter=1)

    np.testing.assert_array_equal(
        guttered,
        np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ],
            dtype=np.float32,
        ),
    )


def test_inner_uv_for_gutter_excludes_border_pixels():
    assert inner_uv_for_gutter((4, 6), gutter=1) == (1 / 6, 1 / 4, 5 / 6, 3 / 4)
