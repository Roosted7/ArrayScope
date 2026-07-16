"""Qt-free LOD pyramid materialization core (ADR 0050).

This module owns the reduction algorithm, the pyramid cache key, and the
bounded byte-accounted cache with singleflight request bookkeeping.  It must
stay importable without Qt: workers reduce and admit, the GUI thread only
looks levels up.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np

from arrayscope.core.bounded_cache import BoundedCache
from arrayscope.gpu.keys import (
    COMPLEX_RG32F,
    REDUCER_MEAN,
    REDUCER_MEAN_ABS,
    REDUCER_NATIVE,
    REDUCER_PHASE_VECTOR,
    REDUCER_POWER,
    REDUCER_RMS,
    RGB8,
    SCALAR_R32F,
    ChunkLod,
    DataChunkKey,
)
from arrayscope.gpu.page_table import PageSlot, PageTable


ALGO_VERSION = 1
"""Reduction algorithm version; part of every pyramid cache key."""


ROUTE_ID = "source-grid-page"
"""Canonical operation-identity tag for source-grid page values."""


def reduction_yx_to_xy(reduction_yx: tuple[int, int]) -> tuple[int, int]:
    """Cross the one named boundary into the numeric reducer's ``(x, y)`` API."""

    reduction_y, reduction_x = _reduction_vector_yx(reduction_yx, name="reduction")
    return (reduction_x, reduction_y)


def reduction_xy_to_yx(reduction_xy: tuple[int, int]) -> tuple[int, int]:
    """Convert the legacy numeric reducer convention back to source ``(y, x)``."""

    reduction_x, reduction_y = _reduction_vector_xy(reduction_xy, name="reduction")
    return (reduction_y, reduction_x)


def _content_identity(content_key: object) -> tuple[object, object]:
    if (
        isinstance(content_key, tuple)
        and len(content_key) == 3
        and content_key[0] == "src-anchored"
    ):
        return content_key[1], content_key[2]
    return content_key, None


def _route_operation_key(operation_key: object) -> tuple[object, ...]:
    """Put the reduction algorithm lineage in canonical value identity."""

    return (ROUTE_ID, int(ALGO_VERSION), operation_key)


@dataclass(frozen=True)
class LodPagePlan:
    """Complete immutable route for one canonical source-grid LOD page.

    Geometry is native-source ``(y, x)``. ``stored_rect_yx`` is the page's
    global reduced-sample rectangle; draw-block stored rectangles are local
    to the materialized page array. Boundary bins retain their exact native
    footprint, so a clipped page can never alias a differently clipped page.
    """

    key: DataChunkKey
    source_rect_yx: tuple[int, int, int, int]
    valid_source_rect_yx: tuple[int, int, int, int]
    nominal_source_rect_yx: tuple[int, int, int, int]
    stored_rect_yx: tuple[int, int, int, int]
    stored_page_origin_yx: tuple[int, int]
    source_samples_per_stored_sample_yx: tuple[int, int]
    reduction_yx: tuple[int, int]
    stored_page_shape: tuple[int, int]
    draw_blocks: tuple["SourceGridDrawBlock", ...]
    sample_source_rects_yx: tuple[tuple[int, int, int, int], ...]
    reducer: str
    route_id: str = ROUTE_ID
    algo_version: int = ALGO_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.key, DataChunkKey):
            raise TypeError("LOD page plan key must be a DataChunkKey")
        source_rect = _rect(self.source_rect_yx, name="source_rect_yx")
        valid_rect = _rect(self.valid_source_rect_yx, name="valid_source_rect_yx")
        nominal_rect = _rect(self.nominal_source_rect_yx, name="nominal_source_rect_yx")
        stored_rect = _rect(self.stored_rect_yx, name="stored_rect_yx")
        reduction = _reduction_vector_yx(self.reduction_yx, name="reduction")
        page_shape = _positive_pair(self.stored_page_shape, name="stored page shape")
        samples_per_stored = tuple(1 << step for step in reduction)
        if str(self.route_id) != ROUTE_ID or int(self.algo_version) != int(ALGO_VERSION):
            raise ValueError("page plan route lineage disagrees with the canonical algorithm")
        if tuple(self.source_samples_per_stored_sample_yx) != samples_per_stored:
            raise ValueError("source-sample scale disagrees with reduction vector")
        if source_rect != valid_rect:
            raise ValueError("a page's source footprint must equal its clipped valid footprint")
        if not _rect_contains(nominal_rect, source_rect):
            raise ValueError("nominal source page must contain its valid source footprint")
        if self.key.chunk_origin != (source_rect[0], source_rect[2]) or self.key.chunk_shape != (
            source_rect[1] - source_rect[0],
            source_rect[3] - source_rect[2],
        ):
            raise ValueError("DataChunkKey geometry disagrees with page source footprint")
        if tuple(self.key.lod.reduction) != reduction or self.key.lod.reducer != str(self.reducer):
            raise ValueError("DataChunkKey LOD family disagrees with page route")
        if self.key.operation_key != _route_operation_key(_unwrap_route_operation_key(self.key.operation_key)):
            raise ValueError("DataChunkKey operation identity lacks canonical route lineage")
        stored_shape = (stored_rect[1] - stored_rect[0], stored_rect[3] - stored_rect[2])
        if stored_shape[0] > page_shape[0] or stored_shape[1] > page_shape[1]:
            raise ValueError("valid stored page footprint exceeds its page shape")
        if len(self.sample_source_rects_yx) != stored_shape[0] * stored_shape[1]:
            raise ValueError("sample source rectangles do not match stored page shape")
        _validate_draw_blocks(self.draw_blocks, stored_shape, source_rect)
        object.__setattr__(self, "source_rect_yx", source_rect)
        object.__setattr__(self, "valid_source_rect_yx", valid_rect)
        object.__setattr__(self, "nominal_source_rect_yx", nominal_rect)
        object.__setattr__(self, "stored_rect_yx", stored_rect)
        object.__setattr__(self, "reduction_yx", reduction)
        object.__setattr__(self, "stored_page_shape", page_shape)
        object.__setattr__(self, "source_samples_per_stored_sample_yx", samples_per_stored)
        object.__setattr__(self, "reducer", str(self.reducer))
        object.__setattr__(self, "route_id", str(self.route_id))
        object.__setattr__(self, "algo_version", int(self.algo_version))

    @property
    def stored_shape(self) -> tuple[int, int]:
        y0, y1, x0, x1 = self.stored_rect_yx
        return (y1 - y0, x1 - x0)


@dataclass(frozen=True)
class MaterializedLodPage:
    """Contiguous page values admitted only under the plan that made them."""

    plan: LodPagePlan
    values: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.plan, LodPagePlan):
            raise TypeError("materialized LOD page requires a LodPagePlan")
        values = np.asarray(self.values)
        if tuple(values.shape[:2]) != self.plan.stored_shape:
            raise ValueError(
                f"materialized shape {values.shape[:2]} does not match planned {self.plan.stored_shape}"
            )
        if values.ndim < 2 or values.ndim > 3:
            raise ValueError("materialized page must be 2D plus at most one component axis")
        expected_dtype, expected_representation = _planned_value_format(
            self.plan.reducer,
            self.plan.key.dtype,
            self.plan.key.representation,
        )
        if values.dtype != expected_dtype:
            raise ValueError(
                f"materialized dtype {values.dtype} does not match planned {expected_dtype}"
            )
        if self.plan.key.representation != expected_representation:
            raise ValueError(
                "materialized representation does not match reducer output: "
                f"{self.plan.key.representation!r} != {expected_representation!r}"
            )
        if not values.flags.c_contiguous:
            raise ValueError("materialized page values must be C-contiguous")
        object.__setattr__(self, "values", values)

    @property
    def key(self) -> DataChunkKey:
        return self.plan.key

    @property
    def nbytes(self) -> int:
        return int(self.values.nbytes)


def plan_source_grid_pages(
    *,
    content_key: object,
    valid_source_rect_yx: tuple[int, int, int, int],
    reduction_yx: tuple[int, int],
    stored_page_shape: tuple[int, int],
    dtype: str,
    representation: str,
    reducer: str,
) -> tuple[LodPagePlan, ...]:
    """Plan page identity, reduction slices, and exact draw geometry once."""

    valid_rect = _rect(valid_source_rect_yx, name="valid_source_rect_yx")
    reduction = _reduction_vector_yx(reduction_yx, name="reduction")
    stored_h, stored_w = _positive_pair(stored_page_shape, name="stored page shape")
    reducer = str(reducer)
    if reducer == REDUCER_NATIVE and any(reduction):
        raise ValueError("native pages cannot carry a reduced LOD")
    if reducer != REDUCER_NATIVE and not any(reduction):
        raise ValueError("unreduced pages must use the native reducer")
    source_scale_y, source_scale_x = (1 << reduction[0], 1 << reduction[1])
    source_page_h, source_page_w = stored_h * source_scale_y, stored_w * source_scale_x
    y_bins = _source_grid_axis_rects(valid_rect[0], valid_rect[1], source_scale_y)
    x_bins = _source_grid_axis_rects(valid_rect[2], valid_rect[3], source_scale_x)
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, (bin_y0, _bin_y1) in enumerate(y_bins):
        nominal_y = ((bin_y0 // source_scale_y) // stored_h) * source_page_h
        for column, (bin_x0, _bin_x1) in enumerate(x_bins):
            nominal_x = ((bin_x0 // source_scale_x) // stored_w) * source_page_w
            groups.setdefault((nominal_y, nominal_x), []).append((row, column))

    document_generation, base_operation_key = _content_identity(content_key)
    operation_key = _route_operation_key(base_operation_key)
    level = max(reduction)
    lod = ChunkLod(
        level=level,
        factor=1 << level,
        reduction=reduction,
        reducer=reducer,
    )
    plans: list[LodPagePlan] = []
    for nominal_y, nominal_x in sorted(groups):
        positions = groups[(nominal_y, nominal_x)]
        rows = tuple(sorted({row for row, _column in positions}))
        columns = tuple(sorted({column for _row, column in positions}))
        if len(positions) != len(rows) * len(columns):
            raise ValueError("planned source-grid page samples must form a rectangle")
        page_y_bins = y_bins[rows[0] : rows[-1] + 1]
        page_x_bins = x_bins[columns[0] : columns[-1] + 1]
        source_rect = (
            page_y_bins[0][0],
            page_y_bins[-1][1],
            page_x_bins[0][0],
            page_x_bins[-1][1],
        )
        stored_y0 = page_y_bins[0][0] // source_scale_y
        stored_y1 = page_y_bins[-1][0] // source_scale_y + 1
        stored_x0 = page_x_bins[0][0] // source_scale_x
        stored_x1 = page_x_bins[-1][0] // source_scale_x + 1
        rect_grid = np.asarray(
            [
                (bin_y0, bin_y1, bin_x0, bin_x1)
                for bin_y0, bin_y1 in page_y_bins
                for bin_x0, bin_x1 in page_x_bins
            ],
            dtype=np.int64,
        ).reshape(len(page_y_bins), len(page_x_bins), 4)
        key = DataChunkKey(
            document_generation=document_generation,
            operation_key=operation_key,
            lod=lod,
            chunk_origin=(source_rect[0], source_rect[2]),
            chunk_shape=(source_rect[1] - source_rect[0], source_rect[3] - source_rect[2]),
            dtype=str(dtype),
            representation=str(representation),
        )
        plans.append(
            LodPagePlan(
                key=key,
                source_rect_yx=source_rect,
                valid_source_rect_yx=source_rect,
                nominal_source_rect_yx=(
                    nominal_y,
                    nominal_y + source_page_h,
                    nominal_x,
                    nominal_x + source_page_w,
                ),
                stored_rect_yx=(stored_y0, stored_y1, stored_x0, stored_x1),
                stored_page_origin_yx=(nominal_y // source_scale_y, nominal_x // source_scale_x),
                source_samples_per_stored_sample_yx=(source_scale_y, source_scale_x),
                reduction_yx=reduction,
                stored_page_shape=(stored_h, stored_w),
                draw_blocks=_source_grid_draw_blocks(rect_grid),
                sample_source_rects_yx=tuple(
                    tuple(int(value) for value in rect) for rect in rect_grid.reshape(-1, 4)
                ),
                reducer=reducer,
            )
        )
    return tuple(plans)


@dataclass(frozen=True)
class SourceGridBinIdentity:
    """Value identity of one globally anchored reduced source rectangle."""

    source_rect_yx: tuple[int, int, int, int]
    reduction_vector_xy: tuple[int, int]
    reducer: str = "mean"
    algo_version: int = ALGO_VERSION


@dataclass(frozen=True)
class SourceGridReduction:
    """Reduced values plus the native-source coverage of every sample."""

    values: np.ndarray
    source_rects: tuple[tuple[int, int, int, int], ...]
    identities: tuple[SourceGridBinIdentity, ...]
    grid_origin_yx: tuple[int, int]
    reduction_vector_xy: tuple[int, int]
    valid_source_rect_yx: tuple[int, int, int, int]


@dataclass(frozen=True)
class SourceGridPageIdentity:
    """Identity of one uniform stored-sample page and its valid footprint."""

    source_rect_yx: tuple[int, int, int, int]
    reduction_vector_xy: tuple[int, int]
    reducer: str = "mean"
    algo_version: int = ALGO_VERSION


@dataclass(frozen=True)
class SourceGridDrawBlock:
    """Rectangular stored-sample run with one uniform native mapping."""

    stored_rect_yx: tuple[int, int, int, int]
    source_rect_yx: tuple[int, int, int, int]


@dataclass(frozen=True)
class SourceGridPage:
    """One page of reduced values with exact per-sample draw coverage."""

    identity: SourceGridPageIdentity
    source_rect_yx: tuple[int, int, int, int]
    values: np.ndarray
    draw_source_rects: tuple[tuple[int, int, int, int], ...]
    draw_blocks: tuple[SourceGridDrawBlock, ...]


def partition_source_grid_pages(
    reduction: SourceGridReduction,
    *,
    stored_page_shape: tuple[int, int],
) -> tuple[SourceGridPage, ...]:
    """Partition reduced samples without losing their native draw spans.

    Page alignment is global native-source alignment.  Values are grouped by
    the page containing their bin origin; clipped first/last pages keep their
    exact valid footprint in identity, and every sample retains its own
    source rectangle for later backend geometry construction.
    """

    values = np.asarray(reduction.values)
    if values.ndim < 2:
        raise ValueError("source-grid page partition requires a 2D reduction")
    stored_h, stored_w = (int(value) for value in stored_page_shape)
    if stored_h <= 0 or stored_w <= 0:
        raise ValueError(f"stored page shape must be positive, got {stored_page_shape}")
    level_x, level_y = reduction.reduction_vector_xy
    source_page_h = stored_h * (1 << int(level_y))
    source_page_w = stored_w * (1 << int(level_x))
    rect_grid = np.asarray(reduction.source_rects, dtype=np.int64).reshape(
        values.shape[0], values.shape[1], 4
    )
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            rect = rect_grid[row, column]
            page_origin = (
                (int(rect[0]) // source_page_h) * source_page_h,
                (int(rect[2]) // source_page_w) * source_page_w,
            )
            groups.setdefault(page_origin, []).append((row, column))
    pages: list[SourceGridPage] = []
    for page_origin in sorted(groups):
        positions = groups[page_origin]
        rows = tuple(sorted({row for row, _column in positions}))
        columns = tuple(sorted({column for _row, column in positions}))
        if len(positions) != len(rows) * len(columns):
            raise ValueError("source-grid page samples must form one rectangular block")
        row_slice = slice(rows[0], rows[-1] + 1)
        column_slice = slice(columns[0], columns[-1] + 1)
        page_rects = rect_grid[row_slice, column_slice].reshape(-1, 4)
        source_rect = (
            int(np.min(page_rects[:, 0])),
            int(np.max(page_rects[:, 1])),
            int(np.min(page_rects[:, 2])),
            int(np.max(page_rects[:, 3])),
        )
        identity = SourceGridPageIdentity(
            source_rect_yx=source_rect,
            reduction_vector_xy=tuple(reduction.reduction_vector_xy),
        )
        pages.append(
            SourceGridPage(
                identity=identity,
                source_rect_yx=source_rect,
                values=np.ascontiguousarray(values[row_slice, column_slice]),
                draw_source_rects=tuple(tuple(int(value) for value in rect) for rect in page_rects),
                draw_blocks=_source_grid_draw_blocks(
                    rect_grid[row_slice, column_slice]
                ),
            )
        )
    return tuple(pages)


def _source_grid_draw_blocks(rect_grid: np.ndarray) -> tuple[SourceGridDrawBlock, ...]:
    """Coalesce equal-width sample spans into a bounded Cartesian grid."""

    rows, columns = rect_grid.shape[:2]
    y_spans = tuple((int(rect_grid[row, 0, 0]), int(rect_grid[row, 0, 1])) for row in range(rows))
    x_spans = tuple((int(rect_grid[0, column, 2]), int(rect_grid[0, column, 3])) for column in range(columns))
    y_runs = _equal_width_runs(y_spans)
    x_runs = _equal_width_runs(x_spans)
    return tuple(
        SourceGridDrawBlock(
            stored_rect_yx=(row0, row1, column0, column1),
            source_rect_yx=(source_y0, source_y1, source_x0, source_x1),
        )
        for row0, row1, source_y0, source_y1 in y_runs
        for column0, column1, source_x0, source_x1 in x_runs
    )


def _equal_width_runs(
    spans: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int, int, int], ...]:
    runs: list[tuple[int, int, int, int]] = []
    start = 0
    while start < len(spans):
        width = spans[start][1] - spans[start][0]
        stop = start + 1
        while (
            stop < len(spans)
            and spans[stop][0] == spans[stop - 1][1]
            and spans[stop][1] - spans[stop][0] == width
        ):
            stop += 1
        runs.append((start, stop, spans[start][0], spans[stop - 1][1]))
        start = stop
    return tuple(runs)


def reduce_source_grid_mean(
    array,
    *,
    source_origin_yx: tuple[int, int],
    valid_source_rect_yx: tuple[int, int, int, int],
    reduction_vector_xy: tuple[int, int],
    input_reduction_vector_xy: tuple[int, int] = (0, 0),
) -> SourceGridReduction:
    """Reduce on the global native-source grid, independent of window origin.

    The reduction vectors are absolute log2 steps in ``(x, y)`` order.
    ``source_origin_yx`` locates input sample ``[0, 0]`` in native source
    coordinates. Recursive reduction is accepted only for a fully aligned
    input grid, which makes the result identical to direct source reduction;
    clipped partial parents are rejected instead of making cache history part
    of the values.
    """

    values = np.asarray(array)
    if values.ndim < 2 or values.ndim > 3:
        raise ValueError("source-grid mean requires 2D data plus at most one component axis")
    level_x, level_y = _reduction_vector(reduction_vector_xy, name="reduction")
    input_level_x, input_level_y = _reduction_vector(
        input_reduction_vector_xy,
        name="input reduction",
    )
    if level_x < input_level_x or level_y < input_level_y:
        raise ValueError("output reduction must not be finer than the input reduction")
    factor_x, factor_y = (1 << level_x, 1 << level_y)
    input_factor_x, input_factor_y = (1 << input_level_x, 1 << input_level_y)
    source_y, source_x = (int(source_origin_yx[0]), int(source_origin_yx[1]))
    valid_y0, valid_y1, valid_x0, valid_x1 = (
        int(value) for value in valid_source_rect_yx
    )
    if valid_y1 <= valid_y0 or valid_x1 <= valid_x0:
        raise ValueError("valid source rectangle must be non-empty")
    input_y1 = source_y + int(values.shape[0]) * input_factor_y
    input_x1 = source_x + int(values.shape[1]) * input_factor_x
    if not (
        source_y <= valid_y0 < valid_y1 <= input_y1
        and source_x <= valid_x0 < valid_x1 <= input_x1
    ):
        raise ValueError("valid source rectangle lies outside the input sample coverage")
    if (input_level_x or input_level_y) and (
        source_y % input_factor_y
        or source_x % input_factor_x
        or valid_y0 % input_factor_y
        or valid_y1 % input_factor_y
        or valid_x0 % input_factor_x
        or valid_x1 % input_factor_x
    ):
        raise ValueError("recursive input and valid coverage must align to its source grid")

    y_rects = _source_grid_axis_rects(valid_y0, valid_y1, factor_y)
    x_rects = _source_grid_axis_rects(valid_x0, valid_x1, factor_x)
    output_shape = (len(y_rects), len(x_rects), *values.shape[2:])
    output_dtype = np.complex64 if np.iscomplexobj(values) else np.float32
    reduced = np.empty(output_shape, dtype=output_dtype)
    source_rects: list[tuple[int, int, int, int]] = []
    identities: list[SourceGridBinIdentity] = []
    accumulated = values.astype(output_dtype, copy=False)
    for out_y, (rect_y0, rect_y1) in enumerate(y_rects):
        local_y0 = (rect_y0 - source_y) // input_factor_y
        local_y1 = (rect_y1 - source_y) // input_factor_y
        for out_x, (rect_x0, rect_x1) in enumerate(x_rects):
            local_x0 = (rect_x0 - source_x) // input_factor_x
            local_x1 = (rect_x1 - source_x) // input_factor_x
            source_rect = (rect_y0, rect_y1, rect_x0, rect_x1)
            sample = accumulated[local_y0:local_y1, local_x0:local_x1]
            if sample.size == 0:
                raise ValueError("source-grid bin has no input samples")
            reduced[out_y, out_x] = np.mean(sample, axis=(0, 1), dtype=output_dtype)
            source_rects.append(source_rect)
            identities.append(
                SourceGridBinIdentity(
                    source_rect_yx=source_rect,
                    reduction_vector_xy=(level_x, level_y),
                )
            )
    return SourceGridReduction(
        values=reduced,
        source_rects=tuple(source_rects),
        identities=tuple(identities),
        grid_origin_yx=(
            (valid_y0 // factor_y) * factor_y,
            (valid_x0 // factor_x) * factor_x,
        ),
        reduction_vector_xy=(level_x, level_y),
        valid_source_rect_yx=(valid_y0, valid_y1, valid_x0, valid_x1),
    )


def _reduction_vector(value, *, name: str) -> tuple[int, int]:
    level_x, level_y = (int(value[0]), int(value[1]))
    if level_x < 0 or level_y < 0:
        raise ValueError(f"{name} steps must be non-negative")
    return (level_x, level_y)


def _source_grid_axis_rects(start: int, stop: int, factor: int) -> tuple[tuple[int, int], ...]:
    first = (int(start) // int(factor)) * int(factor)
    return tuple(
        (max(int(start), origin), min(int(stop), origin + int(factor)))
        for origin in range(first, int(stop), int(factor))
    )


def materialize_lod_page(
    array,
    *,
    source_origin_yx: tuple[int, int],
    plan: LodPagePlan,
) -> MaterializedLodPage:
    """Execute exactly one precomputed page route against source coverage."""

    source = np.asarray(array)
    if source.ndim < 2 or source.ndim > 3:
        raise ValueError("LOD page materialization requires 2D data plus at most one component axis")
    source_y, source_x = (int(source_origin_yx[0]), int(source_origin_yx[1]))
    source_stop_y = source_y + int(source.shape[0])
    source_stop_x = source_x + int(source.shape[1])
    if not _rect_contains(
        (source_y, source_stop_y, source_x, source_stop_x),
        plan.valid_source_rect_yx,
    ):
        raise ValueError("materialization source does not cover the planned valid footprint")

    output_dtype, output_representation = _reducer_output_format(plan.reducer, source.dtype)
    if np.dtype(plan.key.dtype) != output_dtype:
        raise ValueError(
            f"planned dtype {plan.key.dtype!r} disagrees with reducer output {output_dtype}"
        )
    if plan.key.representation != output_representation:
        raise ValueError(
            "planned representation disagrees with reducer output: "
            f"{plan.key.representation!r} != {output_representation!r}"
        )
    trailing_shape = () if plan.reducer not in (REDUCER_NATIVE, REDUCER_MEAN) else source.shape[2:]
    values = np.empty((*plan.stored_shape, *trailing_shape), dtype=output_dtype)
    for index, source_rect in enumerate(plan.sample_source_rects_yx):
        row, column = divmod(index, plan.stored_shape[1])
        y0, y1, x0, x1 = source_rect
        sample = source[y0 - source_y : y1 - source_y, x0 - source_x : x1 - source_x]
        if sample.size == 0:
            raise ValueError("planned source-grid bin has no source samples")
        values[row, column] = _reduce_sample(sample, reducer=plan.reducer)
    return MaterializedLodPage(plan=plan, values=np.ascontiguousarray(values))


def materialize_source_grid_pages(
    array,
    *,
    source_origin_yx: tuple[int, int],
    plans: tuple[LodPagePlan, ...],
) -> tuple[MaterializedLodPage, ...]:
    """Materialize a checked set of pages without replanning their geometry."""

    requested = tuple(plans)
    if len({plan.key for plan in requested}) != len(requested):
        raise ValueError("duplicate LOD page targets are not allowed")
    return tuple(
        materialize_lod_page(array, source_origin_yx=source_origin_yx, plan=plan)
        for plan in requested
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
    output_dtype, representation = _reducer_output_format(str(reducer), source.dtype)
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
    by_rect: dict[tuple[int, int, int, int], object] = {}
    for page in materialized:
        for rect, value in zip(
            page.plan.sample_source_rects_yx,
            page.values.reshape((-1, *trailing_shape)),
            strict=True,
        ):
            by_rect[rect] = value
    rects = tuple(
        (y0, y1, x0, x1)
        for y0, y1 in y_bins
        for x0, x1 in x_bins
    )
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


def _reduce_sample(sample: np.ndarray, *, reducer: str):
    reducer = str(reducer)
    if reducer == REDUCER_NATIVE:
        if tuple(sample.shape[:2]) != (1, 1):
            raise ValueError("native route must map one source sample to one stored sample")
        return sample[0, 0]
    if reducer == REDUCER_MEAN:
        dtype = np.complex64 if np.iscomplexobj(sample) else np.float32
        return np.mean(sample, axis=(0, 1), dtype=dtype)
    magnitude = np.abs(sample).astype(np.float32, copy=False)
    if reducer == REDUCER_MEAN_ABS:
        return np.mean(magnitude, axis=(0, 1), dtype=np.float32)
    squared = np.square(magnitude, dtype=np.float32)
    if reducer == REDUCER_POWER:
        return np.mean(squared, axis=(0, 1), dtype=np.float32)
    if reducer == REDUCER_RMS:
        return np.float32(np.sqrt(np.mean(squared, axis=(0, 1), dtype=np.float32)))
    if reducer == REDUCER_PHASE_VECTOR:
        if not np.iscomplexobj(sample):
            raise ValueError("phase_vector requires complex source values")
        vectors = np.zeros(sample.shape, dtype=np.complex64)
        nonzero = magnitude > 0.0
        vectors[nonzero] = sample[nonzero].astype(np.complex64, copy=False) / magnitude[nonzero]
        result = np.mean(vectors, axis=(0, 1), dtype=np.complex64)
        tolerance = np.finfo(np.float32).eps * max(1, int(sample.shape[0]) * int(sample.shape[1]))
        return np.complex64(0.0) if abs(result) <= tolerance else np.complex64(result)
    raise ValueError(f"unsupported source-grid reducer {reducer!r}")


def _reducer_output_format(reducer: str, source_dtype) -> tuple[np.dtype, str]:
    reducer = str(reducer)
    dtype = np.dtype(source_dtype)
    if reducer == REDUCER_NATIVE:
        if np.issubdtype(dtype, np.complexfloating):
            return np.dtype(np.complex64), COMPLEX_RG32F
        if dtype == np.dtype(np.uint8):
            return dtype, RGB8
        return dtype, SCALAR_R32F
    if reducer == REDUCER_MEAN:
        if np.issubdtype(dtype, np.complexfloating):
            return np.dtype(np.complex64), COMPLEX_RG32F
        return np.dtype(np.float32), SCALAR_R32F
    if reducer == REDUCER_PHASE_VECTOR:
        if not np.issubdtype(dtype, np.complexfloating):
            raise ValueError("phase_vector requires complex source values")
        return np.dtype(np.complex64), COMPLEX_RG32F
    if reducer in (REDUCER_MEAN_ABS, REDUCER_POWER, REDUCER_RMS):
        return np.dtype(np.float32), SCALAR_R32F
    raise ValueError(f"unsupported source-grid reducer {reducer!r}")


def _planned_value_format(reducer: str, dtype: str, representation: str) -> tuple[np.dtype, str]:
    planned_dtype = np.dtype(dtype)
    if reducer == REDUCER_NATIVE:
        return planned_dtype, str(representation)
    if reducer in (REDUCER_MEAN, REDUCER_PHASE_VECTOR) and representation == COMPLEX_RG32F:
        expected = np.dtype(np.complex64)
        if planned_dtype != expected:
            raise ValueError("complex reducer output must be planned as complex64")
        return expected, COMPLEX_RG32F
    if reducer in (REDUCER_MEAN, REDUCER_MEAN_ABS, REDUCER_POWER, REDUCER_RMS):
        expected = np.dtype(np.float32)
        if planned_dtype != expected:
            raise ValueError("scalar reducer output must be planned as float32")
        return expected, SCALAR_R32F
    raise ValueError("planned dtype/representation is incompatible with reducer")


def _unwrap_route_operation_key(operation_key: object) -> object:
    if (
        isinstance(operation_key, tuple)
        and len(operation_key) == 3
        and operation_key[0] == ROUTE_ID
        and int(operation_key[1]) == int(ALGO_VERSION)
    ):
        return operation_key[2]
    return object()


def _rect(value, *, name: str) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = (int(part) for part in value)
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"{name} must be a non-empty non-negative rectangle, got {value}")
    return (y0, y1, x0, x1)


def _positive_pair(value, *, name: str) -> tuple[int, int]:
    first, second = (int(part) for part in value)
    if first <= 0 or second <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return (first, second)


def _reduction_vector_yx(value, *, name: str) -> tuple[int, int]:
    reduction_y, reduction_x = (int(part) for part in value)
    if reduction_y < 0 or reduction_x < 0:
        raise ValueError(f"{name} steps must be non-negative")
    return (reduction_y, reduction_x)


def _reduction_vector_xy(value, *, name: str) -> tuple[int, int]:
    reduction_x, reduction_y = (int(part) for part in value)
    if reduction_x < 0 or reduction_y < 0:
        raise ValueError(f"{name} steps must be non-negative")
    return (reduction_x, reduction_y)


def _rect_contains(outer, inner) -> bool:
    return bool(
        outer[0] <= inner[0] < inner[1] <= outer[1]
        and outer[2] <= inner[2] < inner[3] <= outer[3]
    )


def _validate_draw_blocks(
    blocks: tuple[SourceGridDrawBlock, ...],
    stored_shape: tuple[int, int],
    source_rect: tuple[int, int, int, int],
) -> None:
    stored_cover = np.zeros(stored_shape, dtype=np.uint8)
    source_cover = np.zeros(
        (source_rect[1] - source_rect[0], source_rect[3] - source_rect[2]),
        dtype=np.uint8,
    )
    for block in tuple(blocks):
        sy0, sy1, sx0, sx1 = block.stored_rect_yx
        by0, by1, bx0, bx1 = block.source_rect_yx
        if not (0 <= sy0 < sy1 <= stored_shape[0] and 0 <= sx0 < sx1 <= stored_shape[1]):
            raise ValueError("draw block lies outside stored page bounds")
        if not _rect_contains(source_rect, block.source_rect_yx):
            raise ValueError("draw block lies outside source page bounds")
        stored_cover[sy0:sy1, sx0:sx1] += 1
        source_cover[
            by0 - source_rect[0] : by1 - source_rect[0],
            bx0 - source_rect[2] : bx1 - source_rect[2],
        ] += 1
    if not np.all(stored_cover == 1) or not np.all(source_cover == 1):
        raise ValueError("draw blocks must cover stored and source footprints exactly once")


def reduce_box_mean(array, factor_xy: tuple[int, int]) -> np.ndarray:
    """Reduce the first two axes of ``array`` by per-axis box means.

    ``factor_xy`` is ``(factor_x, factor_y)`` where x reduces the width axis
    (axis 1) and y reduces the height axis (axis 0), matching the
    ``desired_factor_xy`` convention in :mod:`arrayscope.display.lod`.

    Rules:

    - factors must be powers of two (level identity is per-axis log2);
    - accumulation runs in float32 for real inputs and complex64 for complex
      inputs (texture-appropriate output precision);
    - non-divisible trailing edges average the partial box, so no padding
      values leak into the result;
    - trailing component axes (RGB(A) or RG two-component planes) are reduced
      per component;
    - integer inputs round back to the input dtype so RGB8 textures stay
      uploadable.
    """

    factor_x, factor_y = (int(factor_xy[0]), int(factor_xy[1]))
    if factor_x < 1 or factor_y < 1:
        raise ValueError("reduction factors must be positive")
    if factor_x & (factor_x - 1) or factor_y & (factor_y - 1):
        raise ValueError(f"reduction factors must be powers of two, got {(factor_x, factor_y)}")
    values = np.asarray(array)
    if values.ndim < 2:
        raise ValueError("box-mean reduction requires at least two axes")
    if values.ndim > 3:
        raise ValueError("box-mean reduction supports 2D data plus one trailing component axis")
    if np.iscomplexobj(values):
        accumulated = values.astype(np.complex64, copy=False)
    else:
        accumulated = values.astype(np.float32, copy=False)
    reduced = _reduce_axis(_reduce_axis(accumulated, factor_y, axis=0), factor_x, axis=1)
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        return np.clip(np.rint(reduced), info.min, info.max).astype(values.dtype)
    if np.iscomplexobj(values):
        return reduced.astype(np.complex64, copy=False)
    return reduced.astype(np.float32, copy=False)


def _reduce_axis(values: np.ndarray, factor: int, *, axis: int) -> np.ndarray:
    if factor <= 1 or values.shape[axis] <= 1:
        return values
    length = int(values.shape[axis])
    starts = np.arange(0, length, factor)
    sums = np.add.reduceat(values, starts, axis=axis, dtype=values.dtype)
    counts = np.diff(np.append(starts, length)).astype(np.float32)
    shape = [1] * values.ndim
    shape[axis] = len(starts)
    return sums / counts.reshape(shape)


class LodPageCache:
    """Renderer-shared, byte-bounded logical page cache with owned claims."""

    def __init__(self, *, max_bytes: int | None = None, max_entries: int | None = None) -> None:
        self._cache = BoundedCache(max_bytes=max_bytes, max_entries=max_entries)
        self._claims: dict[DataChunkKey, object] = {}
        self._lock = RLock()
        self._revision = 0

    @property
    def revision(self) -> int:
        return int(self._revision)

    def lookup(self, key: DataChunkKey) -> MaterializedLodPage | None:
        return self._cache.get(key)

    def peek(self, key: DataChunkKey) -> MaterializedLodPage | None:
        return self._cache.peek(key)

    def peek_many(self, keys) -> dict[DataChunkKey, MaterializedLodPage]:
        return self._cache.peek_many(keys)

    def resolve(self, key: DataChunkKey):
        """Resolve through the shared PageTable ancestry contract."""

        table = PageTable()
        for index, (resident_key, page) in enumerate(self._cache.items()):
            table.bind(
                resident_key,
                PageSlot("cpu-lod-page-cache", 0, index),
                nbytes=page.nbytes,
            )
        return table.resolve(key)

    def resolved_pages(self, plans) -> tuple[MaterializedLodPage, ...] | None:
        """Return the complete actual page set, or None for missing coverage."""

        requested = tuple(plans)
        table = PageTable()
        pages_by_key = dict(self._cache.items())
        for index, (resident_key, page) in enumerate(pages_by_key.items()):
            table.bind(
                resident_key,
                PageSlot("cpu-lod-page-cache", 0, index),
                nbytes=page.nbytes,
            )
        resolved = tuple(table.resolve(plan.key) for plan in requested)
        if any(item is None for item in resolved):
            return None
        return tuple(
            pages_by_key[key]
            for key in dict.fromkeys(item.actual_key for item in resolved)
        )

    def begin_claim(self, key: DataChunkKey, owner: object) -> bool:
        """Own the one producer claim for ``key`` if it is missing."""

        if not isinstance(key, DataChunkKey):
            raise TypeError("LOD page cache claims require DataChunkKey keys")
        with self._lock:
            if self._cache.peek(key) is not None or key in self._claims:
                return False
            self._claims[key] = owner
            return True

    def claim_plans(self, plans, owner: object) -> tuple[LodPagePlan, ...]:
        """Return only missing plans newly owned by this request."""

        requested = tuple(plans)
        if len({plan.key for plan in requested}) != len(requested):
            raise ValueError("duplicate LOD page targets are not allowed")
        return tuple(plan for plan in requested if self.begin_claim(plan.key, owner))

    def admit(self, page: MaterializedLodPage, *, owner: object) -> MaterializedLodPage:
        """Admit the exact claimed page; every failure releases its claim."""

        if not isinstance(page, MaterializedLodPage):
            raise TypeError("LOD page cache stores checked MaterializedLodPage values")
        key = page.key
        with self._lock:
            claimed_owner = self._claims.get(key)
            try:
                if claimed_owner != owner:
                    raise ValueError(
                        f"LOD page admission owner mismatch for {key!r}: "
                        f"claimed by {claimed_owner!r}, admitted by {owner!r}"
                    )
                if key != page.plan.key:
                    raise ValueError("LOD page admission key disagrees with materialized plan")
                if self._cache.would_fit(page.nbytes):
                    self._cache.put(key, page, nbytes=page.nbytes)
                    self._revision += 1
            finally:
                if claimed_owner == owner:
                    self._claims.pop(key, None)
        return page

    def admit_as(
        self,
        key: DataChunkKey,
        page: MaterializedLodPage,
        *,
        owner: object,
    ) -> MaterializedLodPage:
        """Checked admission entrypoint for workers carrying a separate key."""

        if key != page.key:
            self.end_claim(key, owner=owner)
            raise ValueError("worker returned a materialized LOD page under the wrong key")
        return self.admit(page, owner=owner)

    def end_claim(self, key: DataChunkKey, *, owner: object) -> None:
        with self._lock:
            claimed_owner = self._claims.get(key)
            if claimed_owner is None:
                return
            if claimed_owner != owner:
                raise ValueError(
                    f"cannot release LOD page claim owned by {claimed_owner!r} as {owner!r}"
                )
            self._claims.pop(key, None)

    def release_owner_claims(self, owner: object) -> tuple[DataChunkKey, ...]:
        with self._lock:
            released = tuple(key for key, claimed_owner in self._claims.items() if claimed_owner == owner)
            for key in released:
                self._claims.pop(key, None)
            return released

    def pending(self, key: DataChunkKey) -> bool:
        with self._lock:
            return key in self._claims

    def claimed_by(self, key: DataChunkKey) -> object | None:
        with self._lock:
            return self._claims.get(key)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._claims)

    @property
    def max_bytes(self) -> int | None:
        return self._cache.max_bytes

    @property
    def bytes_used(self) -> int:
        return int(self._cache.bytes_used)

    @property
    def hits(self) -> int:
        return int(self._cache.hits)

    @property
    def misses(self) -> int:
        return int(self._cache.misses)

    @property
    def evictions(self) -> int:
        return int(self._cache.evictions)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key) -> bool:
        return key in self._cache

    def resident_pages(self) -> tuple[MaterializedLodPage, ...]:
        return tuple(page for _key, page in self._cache.items())

    def resize(self, *, max_bytes: int | None = None, max_entries: int | None = None) -> None:
        self._cache.resize(max_bytes=max_bytes, max_entries=max_entries)
        with self._lock:
            self._revision += 1

    def resident_lod_reducer_counts(self) -> dict[tuple[tuple[int, ...], str], int]:
        counts: dict[tuple[tuple[int, ...], str], int] = {}
        for key, _page in self._cache.items():
            family = (tuple(key.lod.reduction), str(key.lod.reducer))
            counts[family] = counts.get(family, 0) + 1
        return counts

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._claims.clear()
            self._revision += 1


__all__ = [
    "ALGO_VERSION",
    "ROUTE_ID",
    "LodPageCache",
    "LodPagePlan",
    "MaterializedLodPage",
    "SourceGridBinIdentity",
    "SourceGridDrawBlock",
    "SourceGridPage",
    "SourceGridPageIdentity",
    "SourceGridReduction",
    "partition_source_grid_pages",
    "plan_source_grid_pages",
    "reduce_box_mean",
    "reduce_source_grid",
    "reduce_source_grid_mean",
    "materialize_lod_page",
    "materialize_source_grid_pages",
    "reduction_xy_to_yx",
    "reduction_yx_to_xy",
]


def preview_level_for_tile_shape(tile_shape, *, target_edge: int = 48, min_level: int = 2, max_level: int = 6) -> int:
    """Retained-preview level for one tile shape (ADR 0050).

    Coarse enough that a whole stack stays a few megabytes, fine enough to
    scroll through recognizably: the smallest power-of-two level whose
    reduced edges do not undershoot ``target_edge``.
    """

    edge = max(int(tile_shape[0]), int(tile_shape[1]), 1)
    level = 0
    while (edge >> (level + 1)) >= max(1, int(target_edge)) and level < int(max_level):
        level += 1
    return max(int(min_level), level)
