"""Pure-NumPy gate for the reference oracle's overlay-coverage model.

Ring 1.  The physical-pixel oracles compare a presented frame against CPU
semantic truth, and a presented frame also carries what is drawn *over* the
image: ROI outlines and handles, the profile marker, the rasterized floating
chips.  Those pixels are withheld from the image comparison by a mask built
from where each overlay's own geometry places it, and the ROI-placement report
supplies the other half of that contract.

This pins both halves without a GPU: the mask must cover a stroke drawn where
it belongs and must NOT cover one drawn anywhere else, and the placement
report must fail both on a ROI whose colour appears off its band and on a ROI
that stopped being drawn at all.  An oracle that has never failed on an
injected fault is unproven (testing law 5).
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
from arrayscope.tools.framebuffer_reference import (
    ROI_PLACEMENT_MIN_VISIBLE_LENGTH_PX,
    _add_rect,
    _add_segment,
    _roi_placement_report,
    _semantic_roi_outlines,
    _visible_polyline_length,
)

FRAME = (120, 160)
RED = (230, 60, 30)
GREEN = (40, 150, 90)


class _FakeRoiStore:
    def __init__(self, selections):
        self.selections = tuple(selections)


class _FakeImgView:
    """Only the emphasis-derived style hook the oracle reads."""

    @staticmethod
    def _roi_visual_style(roi_id, color):
        return 2.0, tuple(int(value) for value in color)

    _profile_marker_requested_visible = False


class _FakeWindow:
    def __init__(self, selections):
        self.roi_store = _FakeRoiStore(selections)
        self.img_view = _FakeImgView()


def _rect_roi(roi_id, rect, color):
    return RoiSelection(
        id=roi_id,
        label=roi_id,
        geometry=RoiGeometry(kind=RoiKind.RECTANGLE, rect=rect),
        color=color,
    )


def _identity_project(point):
    return (float(point[0]), float(point[1]))


def _grey_frame(value=90):
    return np.full((*FRAME, 3), value, dtype=np.int16)


def _paint_rect_outline(frame, rect, color, width=2):
    """Stroke a rectangle outline into a frame, the way a backend would."""

    x, y, w, h = rect
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
    stroke = np.zeros(FRAME, dtype=bool)
    for start, end in itertools.pairwise(corners):
        _add_segment(stroke, start, end, width / 2.0)
    frame[stroke] = np.asarray(color, dtype=np.int16)
    return stroke


def _placement(win, frame, overlay_mask):
    return _roi_placement_report(
        win,
        frame_rgb=frame,
        project=_identity_project,
        band_radius=lambda pen: pen + 2.0,
        overlay_mask=overlay_mask,
    )


def test_add_segment_marks_only_its_own_neighbourhood():
    mask = np.zeros(FRAME, dtype=bool)
    _add_segment(mask, (20.0, 60.0), (100.0, 60.0), 2.0)
    assert bool(mask[60, 60])
    assert bool(mask[59, 60])
    assert bool(mask[61, 60])
    # Outside the radius, and beyond either end cap, nothing is claimed.
    assert not bool(mask[55, 60])
    assert not bool(mask[60, 5])
    assert not bool(mask[60, 140])


def test_add_rect_grows_by_its_margin_and_clips_to_the_frame():
    mask = np.zeros(FRAME, dtype=bool)
    _add_rect(mask, -10.0, -10.0, 4.0, 4.0, 1.0)
    assert bool(mask[0, 0])
    assert bool(mask[5, 5])
    assert not bool(mask[7, 7])


def test_visible_length_discounts_the_off_frame_part():
    on_frame = _visible_polyline_length(FRAME, [(10.0, 10.0), (110.0, 10.0)])
    half_off = _visible_polyline_length(FRAME, [(110.0, 10.0), (310.0, 10.0)])
    assert on_frame == pytest.approx(100.0, abs=2.0)
    assert half_off == pytest.approx(50.0, rel=0.1)


def test_semantic_outlines_skip_disabled_rois():
    enabled = _rect_roi("roi-1", (20.0, 20.0, 40.0, 30.0), RED)
    disabled = _rect_roi("roi-2", (60.0, 60.0, 20.0, 20.0), GREEN)
    win = _FakeWindow((enabled, replace(disabled, enabled=False)))
    assert [roi_id for roi_id, _, _, _ in _semantic_roi_outlines(win)] == ["roi-1"]


def test_correctly_placed_roi_passes_and_is_covered_by_its_own_mask():
    rect = (20.0, 20.0, 60.0, 40.0)
    win = _FakeWindow((_rect_roi("roi-1", rect, RED),))
    frame = _grey_frame()
    stroke = _paint_rect_outline(frame, rect, RED)

    mask = np.zeros(FRAME, dtype=bool)
    corners = [(20, 20), (80, 20), (80, 60), (20, 60), (20, 20)]
    for start, end in itertools.pairwise(corners):
        _add_segment(mask, start, end, 4.0)
    # The mask built from geometry covers every pixel the stroke painted.
    assert not np.any(stroke & ~mask)

    report = _placement(win, frame, mask)
    assert [roi.roi_id for roi in report.rois] == ["roi-1"]
    assert not report.failures()
    assert report.rois[0].matched_in_band > 0
    assert report.rois[0].stray_pixels == 0
    assert report.rois[0].stray_checked


def test_misplaced_roi_fails_on_stray_pixels():
    """The fault this whole model exists to keep visible."""

    rect = (20.0, 20.0, 60.0, 40.0)
    win = _FakeWindow((_rect_roi("roi-1", rect, RED),))
    frame = _grey_frame()
    # Drawn a full rectangle-width to the right of where geometry says.
    _paint_rect_outline(frame, (90.0, 20.0, 60.0, 40.0), RED)

    mask = np.zeros(FRAME, dtype=bool)
    corners = [(20, 20), (80, 20), (80, 60), (20, 60), (20, 20)]
    for start, end in itertools.pairwise(corners):
        _add_segment(mask, start, end, 4.0)

    report = _placement(win, frame, mask)
    failures = report.failures()
    assert [roi.roi_id for roi in failures] == ["roi-1"]
    assert failures[0].stray_pixels > 0
    assert failures[0].matched_in_band == 0


def test_undrawn_roi_fails_even_with_no_stray_pixels():
    rect = (20.0, 20.0, 60.0, 40.0)
    win = _FakeWindow((_rect_roi("roi-1", rect, RED),))
    frame = _grey_frame()
    mask = np.zeros(FRAME, dtype=bool)
    corners = [(20, 20), (80, 20), (80, 60), (20, 60), (20, 20)]
    for start, end in itertools.pairwise(corners):
        _add_segment(mask, start, end, 4.0)

    report = _placement(win, frame, mask)
    failures = report.failures()
    assert [roi.roi_id for roi in failures] == ["roi-1"]
    assert failures[0].stray_pixels == 0
    assert failures[0].matched_in_band == 0
    assert failures[0].visible_length_px > ROI_PLACEMENT_MIN_VISIBLE_LENGTH_PX


def test_chromatic_frame_reports_the_stray_half_as_unchecked():
    """A colour LUT makes palette colours undecidable — say so, don't pass."""

    rect = (20.0, 20.0, 60.0, 40.0)
    win = _FakeWindow((_rect_roi("roi-1", rect, RED),))
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 255, (*FRAME, 3)).astype(np.int16)
    _paint_rect_outline(frame, rect, RED)
    mask = np.zeros(FRAME, dtype=bool)
    corners = [(20, 20), (80, 20), (80, 60), (20, 60), (20, 20)]
    for start, end in itertools.pairwise(corners):
        _add_segment(mask, start, end, 4.0)

    report = _placement(win, frame, mask)
    assert not report.rois[0].stray_checked
    assert report.rois[0].stray_pixels == 0
    # The presence half still has to hold.
    assert report.rois[0].matched_in_band > 0
    assert not report.failures()
