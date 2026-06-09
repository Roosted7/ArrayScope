import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, CenteredIFFT
from arrayscope.operations.render_plan import (
    RenderCostContext,
    RenderDecisionKind,
    choose_chunk_axis_and_size,
    choose_visible_render_decision,
    degraded_view_state,
    estimate_visible_render_context,
)


def _context(**overrides):
    values = dict(
        view_shape=(1024, 1024),
        display_shape=(1024, 1024),
        image_axes=(0, 1),
        estimated_display_bytes=1024 * 1024 * 4,
        estimated_slab_bytes=1024 * 1024 * 4,
        estimated_pipeline_peak_bytes=1024 * 1024 * 4,
        render_budget_bytes=64 * 1024 * 1024,
        has_operations=False,
        has_transform=False,
        has_reduction=False,
        chunkable=True,
        cache_hit=False,
        pipeline_chunkable_axes=(0, 1),
        pipeline_blocking_axes=(),
        dtype_itemsize=4,
    )
    values.update(overrides)
    return RenderCostContext(**values)


class _ArrayMetadata:
    def __init__(self, shape, dtype):
        self.shape = tuple(int(size) for size in shape)
        self.dtype = np.dtype(dtype)


def test_exact_render_under_budget_chooses_async_exact():
    assert choose_visible_render_decision(_context()).kind == RenderDecisionKind.ASYNC_EXACT


def test_cached_context_chooses_cached():
    assert choose_visible_render_decision(_context(cache_hit=True)).kind == RenderDecisionKind.CACHED


def test_over_budget_chunkable_transform_chooses_async_chunked():
    decision = choose_visible_render_decision(
        _context(
            estimated_slab_bytes=512 * 1024 * 1024,
            estimated_pipeline_peak_bytes=512 * 1024 * 1024,
            has_operations=True,
            has_transform=True,
            pipeline_chunkable_axes=(0,),
            pipeline_blocking_axes=(1,),
        )
    )

    assert decision.kind == RenderDecisionKind.ASYNC_CHUNKED
    assert decision.chunk_axis == 0


def test_over_budget_large_image_chooses_degraded_preview():
    decision = choose_visible_render_decision(
        _context(
            estimated_display_bytes=1024 * 1024 * 1024,
            estimated_slab_bytes=1024 * 1024 * 1024,
            estimated_pipeline_peak_bytes=1024 * 1024 * 1024,
            chunkable=False,
            pipeline_chunkable_axes=(),
        )
    )

    assert decision.kind == RenderDecisionKind.DEGRADED_PREVIEW
    assert decision.degraded_factor in (2, 4, 8, 16)


def test_unpreviewable_over_budget_transform_refuses():
    decision = choose_visible_render_decision(
        _context(
            display_shape=(128, 128),
            estimated_display_bytes=1024 * 1024 * 1024,
            estimated_slab_bytes=1024 * 1024 * 1024,
            estimated_pipeline_peak_bytes=1024 * 1024 * 1024,
            has_transform=True,
            chunkable=False,
            pipeline_chunkable_axes=(),
        )
    )

    assert decision.kind == RenderDecisionKind.REFUSE


def test_degraded_view_state_strides_existing_axis_ranges():
    state = ViewState.from_shape((20, 30)).with_axis_range(0, tuple(range(10))).with_axis_range(1, tuple(range(20)))

    degraded = degraded_view_state(state, factor=4)

    assert degraded.axis_range_indices[0] == (0, 4, 8)
    assert degraded.axis_range_indices[1] == (0, 4, 8, 12, 16)


def test_chunk_axis_never_uses_fft_axis():
    data = np.zeros((64, 64, 16), dtype=np.float32)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=0),))
    state = ViewState.from_shape(document.current_shape)
    context = estimate_visible_render_context(document, state, display_bytes=64 * 64 * 8, render_budget_bytes=1024)

    axis, _size = choose_chunk_axis_and_size(state, context)

    assert axis != 0


def test_simplified_fft_ifft_pair_uses_region_peak_for_render_budget():
    data = _ArrayMetadata((2048, 2048, 128), np.float32)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=2), CenteredIFFT(axis=2)))
    state = ViewState.from_shape(document.current_shape).with_image_axes(0, 1).with_slice(2, 0)

    context = estimate_visible_render_context(
        document,
        state,
        display_bytes=2048 * 2048 * np.dtype(np.complex64).itemsize,
        render_budget_bytes=64 * 1024 * 1024,
    )
    decision = choose_visible_render_decision(context)

    assert context.has_transform is False
    assert context.estimated_pipeline_peak_bytes <= 64 * 1024 * 1024
    assert decision.kind == RenderDecisionKind.ASYNC_EXACT


def test_simplified_fft_ifft_pair_uses_odd_index_region_peak_for_large_2d_matrix():
    data = _ArrayMetadata((10000, 12000), np.float32)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=1), CenteredIFFT(axis=1)))
    rows = tuple(range(1, 10000, 3))
    cols = tuple(range(2, 12000, 5))
    state = (
        ViewState.from_shape(document.current_shape)
        .with_image_axes(0, 1)
        .with_axis_range(0, rows, "odd rows")
        .with_axis_range(1, cols, "odd cols")
    )
    display_bytes = len(rows) * len(cols) * np.dtype(np.complex64).itemsize

    context = estimate_visible_render_context(
        document,
        state,
        display_bytes=display_bytes,
        render_budget_bytes=128 * 1024 * 1024,
    )
    decision = choose_visible_render_decision(context)

    assert context.estimated_pipeline_peak_bytes <= 128 * 1024 * 1024
    assert decision.kind == RenderDecisionKind.ASYNC_EXACT


def test_chunk_axis_prefers_y_then_x():
    axis, _size = choose_chunk_axis_and_size(None, _context(estimated_slab_bytes=1024 * 1024 * 1024))

    assert axis == 0
