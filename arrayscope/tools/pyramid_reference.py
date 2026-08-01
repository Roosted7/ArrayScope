"""The direct source-grid reducer every reducer family is checked against.

Reduce a whole window in one pass, straight from native source values. The
shipped route plans canonical pages and materializes them
(``plan_source_grid_pages`` + ``materialize_source_grid_pages``); this asks the
same question the short way, so a test can state what a reduction *means*
without restating how the page route computes it.

Nothing in the render path calls it -- an oracle with a production caller would
be the production implementation -- so it lives in the harness package and
reaches for ``pyramid``'s own internals, which is what makes it a mirror of
that module rather than a second opinion.
"""

from __future__ import annotations

import numpy as np

from arrayscope.display.pyramid import (
    REDUCER_MEAN,
    REDUCER_NATIVE,
    SourceGridBinIdentity,
    SourceGridReduction,
    _rect,
    _reducer_output_format,
    _reduction_vector_yx,
    _source_grid_axis_rects,
    materialize_source_grid_pages,
    plan_source_grid_pages,
    reduction_yx_to_xy,
)


def reduce_source_grid(
    array,
    *,
    content_key: object = ("standalone-source-grid",),
    source_origin_yx: tuple[int, int],
    valid_source_rect_yx: tuple[int, int, int, int],
    reduction_yx: tuple[int, int],
    reducer: str,
    stored_page_shape: tuple[int, int] | None = None,
) -> SourceGridReduction:
    """Direct source-grid oracle shared by every reducer family.

    Zero-magnitude phase samples contribute the zero vector and still count
    in the bin denominator. This keeps coverage/count policy deterministic;
    all-zero and exactly opposed-phase bins produce a zero resultant.
    """

    source = np.asarray(array)
    valid_rect = _rect(valid_source_rect_yx, name="valid_source_rect_yx")
    reduction = _reduction_vector_yx(reduction_yx, name="reduction")
    factor_y, factor_x = (1 << reduction[0], 1 << reduction[1])
    if stored_page_shape is None:
        stored_page_shape = (
            len(_source_grid_axis_rects(valid_rect[0], valid_rect[1], factor_y)),
            len(_source_grid_axis_rects(valid_rect[2], valid_rect[3], factor_x)),
        )
    output_dtype, representation = _reducer_output_format(str(reducer), source)
    plans = plan_source_grid_pages(
        content_key=content_key,
        valid_source_rect_yx=valid_rect,
        reduction_yx=reduction,
        stored_page_shape=stored_page_shape,
        dtype=output_dtype.name,
        representation=representation,
        reducer=str(reducer),
    )
    materialized = materialize_source_grid_pages(
        source,
        source_origin_yx=source_origin_yx,
        plans=plans,
    )
    # The oracle's default one-page shape preserves the historical row-major
    # result contract. Explicit smaller pages are concatenated geometrically.
    y_bins = _source_grid_axis_rects(valid_rect[0], valid_rect[1], factor_y)
    x_bins = _source_grid_axis_rects(valid_rect[2], valid_rect[3], factor_x)
    trailing_shape = () if str(reducer) not in (REDUCER_NATIVE, REDUCER_MEAN) else source.shape[2:]
    values = np.empty((len(y_bins), len(x_bins), *trailing_shape), dtype=output_dtype)
    by_rect: dict[tuple[int, int, int, int], object] = {
        rect: value
        for page in materialized
        for rect, value in zip(
            page.plan.sample_source_rects_yx,
            page.values.reshape((-1, *trailing_shape)),
            strict=True,
        )
    }
    rects = tuple((y0, y1, x0, x1) for y0, y1 in y_bins for x0, x1 in x_bins)
    for index, rect in enumerate(rects):
        row, column = divmod(index, len(x_bins))
        values[row, column] = by_rect[rect]
    reduction_xy = reduction_yx_to_xy(reduction)
    return SourceGridReduction(
        values=np.ascontiguousarray(values),
        source_rects=rects,
        identities=tuple(
            SourceGridBinIdentity(
                source_rect_yx=rect,
                reduction_vector_xy=reduction_xy,
                reducer=str(reducer),
            )
            for rect in rects
        ),
        grid_origin_yx=(
            (valid_rect[0] // factor_y) * factor_y,
            (valid_rect[2] // factor_x) * factor_x,
        ),
        reduction_vector_xy=reduction_xy,
        valid_source_rect_yx=valid_rect,
    )
