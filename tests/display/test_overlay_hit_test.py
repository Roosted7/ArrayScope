import pytest

from arrayscope.core.roi import RoiGeometry, RoiKind
from arrayscope.display.overlay_hit_test import hit_test_roi, point_segment_distance, roi_handle_points


def test_point_segment_distance_clamps_to_segment_ends():
    assert point_segment_distance((5.0, 1.0), (0.0, 0.0), (4.0, 0.0)) == pytest.approx(2**0.5)
    assert point_segment_distance((2.0, 3.0), (0.0, 0.0), (4.0, 0.0)) == 3.0


def test_rectangle_prefers_resize_handle_then_outline_then_body():
    geometry = RoiGeometry(RoiKind.RECTANGLE, rect=(2.0, 3.0, 8.0, 6.0))

    handle = hit_test_roi(geometry, (10.1, 9.1), tolerance=0.5)
    outline = hit_test_roi(geometry, (6.0, 3.2), tolerance=0.5)
    body = hit_test_roi(geometry, (6.0, 6.0), tolerance=0.5)

    assert handle is not None and handle.part == "handle" and handle.handle_index == 0
    assert outline is not None and outline.part == "outline"
    assert body is not None and body.part == "body"
    assert hit_test_roi(geometry, (20.0, 20.0), tolerance=0.5) is None


def test_line_exposes_both_end_handles_and_hit_tests_body():
    geometry = RoiGeometry(RoiKind.LINE, points=((1.0, 2.0), (9.0, 2.0)))

    assert roi_handle_points(geometry) == ((1.0, 2.0), (9.0, 2.0))
    assert hit_test_roi(geometry, (1.1, 2.0), tolerance=0.25).handle_index == 0
    assert hit_test_roi(geometry, (8.9, 2.0), tolerance=0.25).handle_index == 1
    assert hit_test_roi(geometry, (5.0, 2.2), tolerance=0.25).part == "outline"


def test_closed_freehand_does_not_duplicate_first_handle():
    geometry = RoiGeometry(
        RoiKind.FREEHAND_POLYGON,
        points=((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 0.0)),
        closed=True,
    )

    assert roi_handle_points(geometry) == ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0))
    assert hit_test_roi(geometry, (2.0, 2.0), tolerance=0.2) is not None
