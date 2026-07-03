"""Budgeted, cancellable base-data reads for slab evaluation.

This is the single seam through which evaluation reads document base data.
Eager ndarrays are indexed directly. Lazy sources (anything exposing a
callable ``read_region``) get an explicit region read that is checked
against a byte budget first, so request planning, cancellation, and memory
budgets stay above the source adapter (roadmap: out-of-core and lazy
sources; ADR 0049).
"""

from __future__ import annotations

import numpy as np

from arrayscope.core.array_source import DEFAULT_SOURCE_READ_BUDGET_BYTES, SourceReadRefused
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
    budget = int(budget_bytes) if budget_bytes is not None else source_read_budget_bytes(evaluation_context)
    estimated = region_nbytes(np.shape(base_data), getattr(base_data, "dtype", None), region)
    if estimated is not None and int(estimated) > budget:
        raise SourceReadRefused(
            f"refusing lazy source read of region {region_text(region)}: "
            f"{format_bytes(int(estimated))} exceeds the {format_bytes(budget)} read budget",
            requested_nbytes=int(estimated),
            budget_bytes=budget,
        )
    data = reader(index_spec_from_region(region), cancellation_token=cancellation_token)
    _check_cancelled(cancellation_token)
    return np.asarray(data)


def _check_cancelled(token) -> None:
    if token is not None and getattr(token, "cancelled", False):
        raise EvaluationCancelled()


__all__ = ["read_base_region", "source_read_budget_bytes"]
