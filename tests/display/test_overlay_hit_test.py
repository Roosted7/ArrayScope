import pytest

from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
from arrayscope.display.overlay_hit_test import (
    RoiHitIndex,
    hit_test_roi,
    point_segment_distance,
    roi_handle_points,
)


def test_point_segment_distance_clamps_to_segment_ends():
    assert point_segment_distance((5.0, 1.0), (0.0, 0.0), (4.0, 0.0)) == pytest.approx(2**0.5)
    assert point_segment_distance((2.0, 3.0), (0.0, 0.0), (4.0, 0.0)) == 3.0


def test_rectangle_prefers_resize_handle_then_outline_then_body():
    geometry = RoiGeometry(RoiKind.RECTANGLE, rect=(2.0, 3.0, 8.0, 6.0))

    handle = hit_test_roi(geometry, (10.1, 9.1), tolerance=0.5)
    outline = hit_test_roi(geometry, (6.0, 3.2), tolerance=0.5)
    body = hit_test_roi(geometry, (6.0, 6.0), tolerance=0.5)

    assert handle is not None
    assert handle.part == "handle"
    assert handle.handle_index == 0
    assert outline is not None
    assert outline.part == "outline"
    assert body is not None
    assert body.part == "body"
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


def test_roi_hit_index_returns_local_topmost_candidates_and_tracks_updates():
    index = RoiHitIndex(cell_size=4.0)
    bottom = RoiSelection(
        "bottom", "Bottom", RoiGeometry(RoiKind.RECTANGLE, rect=(0.0, 0.0, 3.0, 3.0))
    )
    top = RoiSelection("top", "Top", RoiGeometry(RoiKind.RECTANGLE, rect=(1.0, 1.0, 3.0, 3.0)))
    far = RoiSelection("far", "Far", RoiGeometry(RoiKind.RECTANGLE, rect=(20.0, 20.0, 2.0, 2.0)))

    index.upsert(bottom)
    index.upsert(top)
    index.upsert(far)

    assert [selection.id for selection in index.candidates((2.0, 2.0), tolerance=0.1)] == [
        "top",
        "bottom",
    ]
    assert [selection.id for selection in index.candidates((21.0, 21.0), tolerance=0.1)] == ["far"]

    moved_top = RoiSelection(
        "top", "Top", RoiGeometry(RoiKind.RECTANGLE, rect=(30.0, 30.0, 3.0, 3.0))
    )
    index.upsert(moved_top)

    assert [selection.id for selection in index.candidates((2.0, 2.0), tolerance=0.1)] == ["bottom"]
    assert [selection.id for selection in index.candidates((31.0, 31.0), tolerance=0.1)] == ["top"]


def test_roi_hit_index_stops_cell_collection_at_global_threshold(monkeypatch):
    index = RoiHitIndex(cell_size=1.0, max_cells_per_roi=3)
    visited = []

    def many_cells(_bounds):
        for value in range(10):
            visited.append(value)
            yield (value, 0)

    monkeypatch.setattr(index, "_cell_range", many_cells)

    cells, global_entry = index._cells_for_bounds((0.0, 0.0, 100.0, 100.0))

    assert cells == ()
    assert global_entry is True
    assert visited == [0, 1, 2, 3]
