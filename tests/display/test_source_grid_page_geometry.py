"""Pure source-grid page materialization and draw geometry (ADR 0056 G5).

Reduced samples own exact native-source rectangles.  Page partitioning may
group those samples for residency, but it must preserve their rectangles for
draw geometry: clipped edge bins are narrower than aligned interior bins and
must never be stretched as though every stored sample covered a full factor.
"""

from __future__ import annotations

import numpy as np

from arrayscope.display import pyramid


FIRST_RECT = (100, 104, 101, 113)
SHIFTED_RECT = (100, 104, 102, 114)
REDUCTION = (1, 1)  # (x, y): factor two on both display axes
STORED_PAGE_SHAPE = (2, 3)


def source_values(rect):
    y0, y1, x0, x1 = rect
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return (yy * 1000 + xx).astype(np.float32)


def reduction_for(rect):
    return pyramid.reduce_source_grid_mean(
        source_values(rect),
        source_origin_yx=(rect[0], rect[2]),
        valid_source_rect_yx=rect,
        reduction_vector_xy=REDUCTION,
    )


def pages_for(rect):
    return pyramid.partition_source_grid_pages(
        reduction_for(rect),
        stored_page_shape=STORED_PAGE_SHAPE,
    )


def test_partition_keeps_values_attached_to_their_exact_draw_spans():
    reduced = reduction_for(FIRST_RECT)
    pages = pages_for(FIRST_RECT)

    assert tuple(page.source_rect_yx for page in pages) == (
        (100, 104, 101, 102),
        (100, 104, 102, 108),
        (100, 104, 108, 113),
    )
    assert tuple(np.asarray(page.values).shape for page in pages) == (
        (2, 1),
        (2, 3),
        (2, 3),
    )

    expected = {
        rect: float(value)
        for rect, value in zip(
            reduced.source_rects,
            np.asarray(reduced.values).reshape(-1),
            strict=True,
        )
    }
    observed = {
        rect: float(value)
        for page in pages
        for rect, value in zip(
            page.draw_source_rects,
            np.asarray(page.values).reshape(-1),
            strict=True,
        )
    }
    assert observed == expected


def test_boundary_draw_spans_have_native_width_and_cover_each_coordinate_once():
    pages = pages_for(FIRST_RECT)
    spans = tuple(rect for page in pages for rect in page.draw_source_rects)

    x_spans = sorted({(x0, x1) for _y0, _y1, x0, x1 in spans})
    assert x_spans == [
        (101, 102),  # clipped first global factor-two bin
        (102, 104),
        (104, 106),
        (106, 108),
        (108, 110),
        (110, 112),
        (112, 113),  # clipped last global factor-two bin
    ]
    assert x_spans[0][1] - x_spans[0][0] == 1
    assert x_spans[-1][1] - x_spans[-1][0] == 1
    assert all(x1 - x0 == 2 for x0, x1 in x_spans[1:-1])

    y0, y1, x0, x1 = FIRST_RECT
    coverage = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for span_y0, span_y1, span_x0, span_x1 in spans:
        coverage[
            span_y0 - y0 : span_y1 - y0,
            span_x0 - x0 : span_x1 - x0,
        ] += 1
    np.testing.assert_array_equal(coverage, np.ones_like(coverage))


def test_shifted_windows_share_only_complete_interior_page_identity():
    first = pages_for(FIRST_RECT)
    shifted = pages_for(SHIFTED_RECT)
    first_by_identity = {page.identity: page for page in first}
    shifted_by_identity = {page.identity: page for page in shifted}

    shared = set(first_by_identity) & set(shifted_by_identity)

    assert len(shared) == 1
    shared_identity = next(iter(shared))
    assert first_by_identity[shared_identity].source_rect_yx == (
        100,
        104,
        102,
        108,
    )
    assert shifted_by_identity[shared_identity].source_rect_yx == (
        100,
        104,
        102,
        108,
    )
    np.testing.assert_array_equal(
        first_by_identity[shared_identity].values,
        shifted_by_identity[shared_identity].values,
    )

