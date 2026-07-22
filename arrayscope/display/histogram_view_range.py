"""Pure histogram value-axis range policy.

The widget/controller layer reports semantic data bounds and manual navigation;
this module decides whether either visible endpoint should move.
"""

from __future__ import annotations

from dataclasses import dataclass

POPULATED_SIDE_MARGIN = 0.10
EMPTY_FLOOR_MARGIN = 0.01
CONTRACTION_THRESHOLD = 0.75
EXPANSION_EDGE_FRACTION = 0.98

ValueRange = tuple[float, float]


@dataclass
class HistogramViewRangePolicy:
    """Stateful, backend-free range and hysteresis owner."""

    manual: bool = False
    data_extents: ValueRange | None = None
    view_range: ValueRange | None = None
    observed_bounds: ValueRange | None = None

    def clear(self) -> None:
        self.manual = False
        self.data_extents = None
        self.view_range = None
        self.observed_bounds = None

    def note_manual_navigation(self) -> None:
        self.manual = True

    def update(self, bounds, *, force: bool = False) -> ValueRange | None:
        """Return a new view range only when the widget must apply one."""

        low, high = sorted((float(bounds[0]), float(bounds[1])))
        observed_bounds = (low, high)
        if force:
            self.manual = False
        elif self.manual or observed_bounds == self.observed_bounds:
            return None

        lower_extent = max(0.0, -low)
        upper_extent = max(0.0, high)
        data_extents = (lower_extent, upper_extent)
        target = range_for_extents(lower_extent, upper_extent)

        previous_extents = self.data_extents
        previous_view = self.view_range
        if force or previous_extents is None or previous_view is None:
            next_extents = data_extents
            next_view = target
        else:
            next_extents_list = [float(previous_extents[0]), float(previous_extents[1])]
            next_view_list = [float(previous_view[0]), float(previous_view[1])]
            side_values = (
                (lower_extent, abs(float(previous_view[0])), float(target[0])),
                (upper_extent, abs(float(previous_view[1])), float(target[1])),
            )
            for side, (extent, visible_extent, target_bound) in enumerate(side_values):
                reference_extent = float(previous_extents[side])
                contracted = extent < reference_extent * CONTRACTION_THRESHOLD
                nearly_full = extent > 0.0 and extent >= visible_extent * EXPANSION_EDGE_FRACTION
                if contracted or nearly_full:
                    next_extents_list[side] = extent
                    next_view_list[side] = target_bound
            next_extents = (next_extents_list[0], next_extents_list[1])
            next_view = (next_view_list[0], next_view_list[1])

        self.data_extents = next_extents
        self.observed_bounds = observed_bounds
        range_changed = previous_view is None or next_view != previous_view
        self.view_range = next_view
        if force or range_changed:
            return next_view
        return None


def range_for_extents(lower_extent: float, upper_extent: float) -> ValueRange:
    """Return an explicit zero-anchored value-axis range."""

    lower_extent = max(0.0, float(lower_extent))
    upper_extent = max(0.0, float(upper_extent))
    populated_span = lower_extent + upper_extent
    reference_span = max(populated_span, lower_extent, upper_extent, 1.0)
    if lower_extent > 0.0:
        low = -lower_extent - reference_span * POPULATED_SIDE_MARGIN
    else:
        low = -reference_span * EMPTY_FLOOR_MARGIN
    high = upper_extent + reference_span * POPULATED_SIDE_MARGIN
    return (float(low), float(high))
