"""G6 chunk-summary and ADR 0056 coverage-frontier contracts."""

import numpy as np

from arrayscope.display.pyramid import materialize_lod_page, plan_source_grid_pages
from arrayscope.gpu import (
    ChunkLod,
    DataChunkKey,
    HISTOGRAM_NORMALIZED_L1_TOLERANCE,
    aggregate_chunk_summaries,
    chunk_key_frontier,
    chunk_summary_frontier,
    summarize_chunk,
)


def _key(*, origin, shape, reduction):
    return DataChunkKey(
        document_generation="doc",
        operation_key=("source-grid-page", 1, "op"),
        lod=ChunkLod(
            level=max(reduction),
            factor=1 << max(reduction),
            reduction=reduction,
            reducer="mean",
        ),
        chunk_origin=origin,
        chunk_shape=shape,
        dtype="float32",
    )


def test_materialized_page_retains_weighted_summary_at_admission():
    values = np.arange(35, dtype=np.float32).reshape(5, 7)
    plan = plan_source_grid_pages(
        content_key="doc",
        valid_source_rect_yx=(0, 5, 0, 7),
        reduction_yx=(1, 1),
        stored_page_shape=(8, 8),
        dtype="float32",
        representation="scalar_r32f",
        reducer="mean",
    )[0]

    page = materialize_lod_page(values, source_origin_yx=(0, 0), plan=plan)

    assert page.summary.key == page.key
    assert page.summary.bounds == (4.0, 34.0)
    assert page.summary.stored_finite_count == page.values.size
    assert page.summary.source_weight == 35.0
    assert np.isclose(np.sum(page.summary.counts), 35.0)
    assert page.summary.counts.size == 64


def test_frontier_keeps_parent_until_children_cover_it_then_replaces_atomically():
    parent = summarize_chunk(
        _key(origin=(0, 0), shape=(4, 4), reduction=(2, 2)),
        np.asarray([[100.0]], dtype=np.float32),
        weights=np.asarray([[16.0]], dtype=np.float64),
    )
    children = tuple(
        summarize_chunk(
            _key(origin=origin, shape=(2, 2), reduction=(1, 1)),
            np.asarray([[value]], dtype=np.float32),
            weights=np.asarray([[4.0]], dtype=np.float64),
        )
        for origin, value in (
            ((0, 0), 1.0),
            ((0, 2), 2.0),
            ((2, 0), 3.0),
            ((2, 2), 4.0),
        )
    )

    partial = chunk_summary_frontier((parent, *children[:3]))
    complete = chunk_summary_frontier((parent, *children))

    assert tuple(item.key for item in partial) == (parent.key,)
    assert {item.key for item in complete} == {item.key for item in children}
    assert parent.key not in {item.key for item in complete}
    assert aggregate_chunk_summaries(partial).source_weight == 16.0
    aggregate = aggregate_chunk_summaries(complete)
    assert aggregate.source_weight == 16.0
    assert aggregate.bounds == (1.0, 4.0)


def test_key_frontier_allows_native_root_of_one_reducer_family():
    parent = _key(origin=(0, 0), shape=(4, 4), reduction=(2, 2))
    native_children = tuple(
        DataChunkKey(
            document_generation=parent.document_generation,
            operation_key=parent.operation_key,
            lod=ChunkLod(),
            chunk_origin=origin,
            chunk_shape=(2, 2),
            dtype=parent.dtype,
        )
        for origin in ((0, 0), (0, 2), (2, 0), (2, 2))
    )

    assert set(chunk_key_frontier((parent, *native_children))) == set(native_children)


def test_aggregate_histogram_is_bounded_and_preserves_population_weight():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(64, 64)).astype(np.float32)
    summaries = []
    for row in range(2):
        for column in range(2):
            block = values[row * 32 : (row + 1) * 32, column * 32 : (column + 1) * 32]
            summaries.append(
                summarize_chunk(
                    _key(
                        origin=(row * 32, column * 32),
                        shape=(32, 32),
                        reduction=(1, 1),
                    ),
                    block,
                )
            )

    aggregate = aggregate_chunk_summaries(summaries)

    assert aggregate.bounds == (float(np.min(values)), float(np.max(values)))
    assert aggregate.source_weight == float(values.size)
    assert np.isclose(np.sum(aggregate.counts), values.size)
    assert aggregate.representative_sample.size <= 512
    assert aggregate.counts.size == 64
    cpu_counts, _edges = np.histogram(values, bins=aggregate.bin_edges)
    normalized_l1 = float(np.sum(np.abs(cpu_counts - aggregate.counts))) / float(
        values.size
    )
    assert normalized_l1 <= HISTOGRAM_NORMALIZED_L1_TOLERANCE
    published_counts, _edges = np.histogram(
        aggregate.representative_sample,
        bins=aggregate.bin_edges,
    )
    published_l1 = float(
        np.sum(
            np.abs(
                published_counts / np.sum(published_counts)
                - aggregate.counts / np.sum(aggregate.counts)
            )
        )
    )
    assert published_l1 <= HISTOGRAM_NORMALIZED_L1_TOLERANCE
