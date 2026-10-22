"""Budgeted, cancellable base-data reads for slab evaluation.

This is the single seam through which evaluation reads document base data.
Eager ndarrays are indexed directly. Lazy sources (anything exposing a
callable ``read_region``) get an explicit region read that is checked
against a byte budget first, so request planning, cancellation, and memory
budgets stay above the source adapter (roadmap: out-of-core and lazy
sources; ADR 0049).
"""

from __future__ import annotations

from operator import index

import numpy as np

from arrayscope.core.array_source import (
    DEFAULT_SOURCE_READ_BUDGET_BYTES,
    CompositeArraySource,
    LazySourceArray,
    SourceReadRefused,
)
from arrayscope.core.memory_budget import format_bytes
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.regions import (
    RegionSpec,
    apply_region,
    index_spec_from_region,
    region_nbytes,
    region_text,
)


def source_read_budget_bytes(evaluation_context) -> int:
    """Resolve the lane-aware read budget from the evaluation context.

    Prefetch-lane reads use the prefetch budget; everything else uses the
    visible-render budget. The default floor keeps large-but-feasible visible
    slabs readable when a policy budget is smaller than the module default.
    """

    policy = getattr(evaluation_context, "memory_policy", None)
    if policy is None:
        return DEFAULT_SOURCE_READ_BUDGET_BYTES
    lane = getattr(evaluation_context, "lane", None)
    lane_value = str(getattr(lane, "value", lane)).lower()
    if lane_value == "prefetch":
        budget = getattr(policy, "prefetch_budget_bytes", None)
    else:
        budget = getattr(policy, "visible_render_budget_bytes", None)
    if budget is None:
        return DEFAULT_SOURCE_READ_BUDGET_BYTES
    return max(int(budget), DEFAULT_SOURCE_READ_BUDGET_BYTES)


def estimate_source_read_nbytes(base_data, region: RegionSpec) -> int | None:
    """Estimate read working memory from metadata, never physical disk traffic.

    A source may conservatively include decoded chunks and assembly buffers in
    ``estimate_read_nbytes(index_spec)``. An advertised unknown estimate stays
    unknown; legacy sources without the optional method use logical result
    bytes. The declared estimate cannot undercut the result allocation.
    """

    logical = region_nbytes(np.shape(base_data), getattr(base_data, "dtype", None), region)
    estimator = getattr(base_data, "estimate_read_nbytes", None)
    if not callable(estimator):
        if isinstance(base_data, LazySourceArray):
            return estimate_source_read_nbytes(base_data.source, region)
        if isinstance(base_data, CompositeArraySource):
            left = estimate_source_read_nbytes(base_data.a, region)
            right = estimate_source_read_nbytes(base_data.b, region)
            left_result = region_nbytes(base_data.shape, base_data.a.dtype, region)
            right_result = region_nbytes(base_data.shape, base_data.b.dtype, region)
            if any(value is None for value in (left, right, left_result, right_result, logical)):
                return None
            # The left result survives the right read; both survive while the
            # elementwise composite allocates its detached output.
            return max(left, left_result + right, left_result + right_result + logical)
        return logical
    estimated = estimator(index_spec_from_region(region))
    if estimated is None:
        return None
    try:
        estimated = index(estimated)
    except TypeError as exc:
        raise ValueError(
            "source read working estimate must be nonnegative integer bytes or None"
        ) from exc
    if estimated < 0:
        raise ValueError("source read working estimate must be nonnegative integer bytes or None")
    return max(estimated, logical or 0)


def read_base_region(
    base_data,
    region: RegionSpec,
    *,
    cancellation_token=None,
    evaluation_context=None,
    budget_bytes: int | None = None,
):
    """Read ``region`` of ``base_data``, enforcing the read budget for lazy sources."""

    _check_cancelled(cancellation_token)
    reader = getattr(base_data, "read_region", None)
    if not callable(reader):
        return apply_region(base_data, region)
    budget = (
        int(budget_bytes)
        if budget_bytes is not None
        else source_read_budget_bytes(evaluation_context)
    )
    estimated = estimate_source_read_nbytes(base_data, region)
    if estimated is None:
        raise SourceReadRefused(
            f"refusing lazy source read of region {region_text(region)}: "
            "its read working memory estimate is unknown",
            budget_bytes=budget,
        )
    if estimated > budget:
        raise SourceReadRefused(
            f"refusing lazy source read of region {region_text(region)}: "
            f"estimated read working memory {format_bytes(estimated)} exceeds the "
            f"{format_bytes(budget)} read budget",
            requested_nbytes=estimated,
            budget_bytes=budget,
        )
    data = reader(index_spec_from_region(region), cancellation_token=cancellation_token)
    _check_cancelled(cancellation_token)
    return np.asarray(data)


def _check_cancelled(token) -> None:
    if token is not None and getattr(token, "cancelled", False):
        raise EvaluationCancelled()


__all__ = ["estimate_source_read_nbytes", "read_base_region", "source_read_budget_bytes"]
