"""Source-anchored tile identity for windowable display windows (ADR 0055 G3).

A displayed-axis subset like ``X=100:200`` is today baked into every tile
key, so shifting it to ``101:201`` re-materializes and re-uploads content
that is almost entirely already resident. When the operation chain commutes
with slicing on a display axis (``pipeline_windowable_display_axes``), tile
identity may instead be anchored to *source* coordinates on that axis: the
tile grid aligns to source-coordinate multiples of the tile shape, and a
tile's content is identified by its source rect plus a window-free view
identity — both invariant under window shifts.

Anchoring is per display axis. A non-windowable axis (e.g. FFT along it)
keeps its window folded into the content key, so shifts along it miss
correctly; the other axis can still anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

from arrayscope.operations.capabilities import pipeline_windowable_display_axes


@dataclass(frozen=True)
class SourceAnchoring:
    """Per-display-axis anchoring decision plus the shift-invariant key.

    ``anchored_starts`` is ``(y_start, x_start)`` in display-axis order:
    the source-coordinate start of that axis's contiguous window, or ``None``
    when the axis is not anchored (window stays part of ``content_key``).
    ``source_starts_yx`` is the same pair in the payload's physical row/column
    order.  They differ only when a backend keeps payloads in canonical
    (sorted-image-axis) orientation and applies an X/Y swap at presentation.
    ``content_key`` identifies tile content independent of anchored-axis
    windows: document identity (data revision + operation steps) plus the
    view identity with anchored axes' ranges stripped.
    """

    anchored_starts: tuple[int | None, int | None]
    content_key: object
    source_starts_yx: tuple[int | None, int | None] | None = None

    def __post_init__(self) -> None:
        display_starts = tuple(self.anchored_starts)
        if len(display_starts) != 2:
            raise ValueError("source anchoring requires two display-axis starts")
        source_starts = (
            display_starts if self.source_starts_yx is None else tuple(self.source_starts_yx)
        )
        if len(source_starts) != 2:
            raise ValueError("source anchoring requires two payload-axis starts")
        object.__setattr__(self, "anchored_starts", display_starts)
        object.__setattr__(self, "source_starts_yx", source_starts)

    @property
    def any_anchored(self) -> bool:
        return any(start is not None for start in self.anchored_starts)


def contiguous_range_start(indices) -> int | None:
    """Start of a contiguous ascending index range, else ``None``."""

    values = tuple(int(value) for value in tuple(indices))
    if not values:
        return None
    start = values[0]
    if values == tuple(range(start, start + len(values))):
        return start
    return None


def source_anchoring_for_view(
    document,
    view_state,
    *,
    canonical_orientation: bool = False,
) -> SourceAnchoring | None:
    """Anchoring decision for a non-montage 2D view, or ``None``.

    Returns ``None`` only when the view has no 2D image axes or is a montage
    (montage tiles anchor per source index already). Otherwise the result
    always carries a stable ``content_key``, even when NO axis is anchored
    (non-windowable chain, e.g. FFT along both display axes): the key then
    keeps every window folded in, which is exactly right — resident chunks
    may be reused only when the identical plane/window is revisited (index
    scroll-back), never across a window shift. Per-axis ``anchored_starts``
    additionally unlock shift reuse where the chain allows it.
    """

    image_axes = getattr(view_state, "image_axes", None)
    if image_axes is None or len(tuple(image_axes)) != 2:
        return None
    if getattr(view_state, "montage_axis", None) is not None:
        return None
    image_axes = tuple(int(axis) for axis in image_axes)
    operations = tuple(getattr(document, "enabled_operations", ()) or ())
    base_data = getattr(document, "base_data", None)
    dtype = getattr(base_data, "dtype", None)
    # pipeline_windowable_display_axes walks the operation chain forward from
    # its INPUT shape, so it must receive the pre-pipeline base-data shape, not
    # ``view_state.shape`` (which is the post-operation display shape). They are
    # equal for shape-preserving chains, but a reduction/reshape on a non-display
    # axis (e.g. Mean over the slider axis) makes view_state.shape lower-rank —
    # and asking a Mean(axis=2) step for output_shape of a 2D shape raised
    # "axis 2 out of bounds for 2D data". With the base shape the chain instead
    # disqualifies every axis on the first shape change, which is the correct
    # windowability answer.
    base_shape = getattr(base_data, "shape", None)
    if base_shape is None:
        base_shape = view_state.shape
    windowable = set(
        pipeline_windowable_display_axes(
            operations,
            tuple(int(size) for size in base_shape),
            dtype,
            display_axes=image_axes,
        )
    )

    anchored_starts: list[int | None] = []
    start_by_axis: dict[int, int | None] = {}
    windowless = view_state
    for axis in image_axes:
        start: int | None = None
        if axis in windowable:
            indices = view_state.axis_range_indices[axis]
            start = 0 if indices is None else contiguous_range_start(indices)
        anchored_starts.append(start)
        start_by_axis[axis] = start
        if start is not None:
            # The slider index of a displayed axis is UI bookkeeping (often
            # moved to the range midpoint by slice editing); the image request
            # keeps that axis, so this index cannot affect its texels.  Leaving
            # it in the window-free key renamed the same source plane between
            # an uncropped view and its first crop even though subsequent
            # same-size shifts happened to reuse correctly.
            windowless = windowless.with_axis_range(axis, None).with_slice(axis, 0)
    source_axes = tuple(sorted(image_axes)) if canonical_orientation else image_axes
    source_starts_yx = tuple(start_by_axis[axis] for axis in source_axes)

    # Deferred import: evaluator pulls in the operations planner stack, and
    # this module is imported by the Qt-free display layer.
    from arrayscope.operations.evaluator import _document_key, _request_key
    from arrayscope.operations.slabs import request_for_image

    return SourceAnchoring(
        anchored_starts=(anchored_starts[0], anchored_starts[1]),
        content_key=(
            "src-anchored",
            _document_key(document),
            _request_key(
                request_for_image(windowless),
                canonical_orientation=canonical_orientation,
            ),
        ),
        source_starts_yx=source_starts_yx,
    )


__all__ = ["SourceAnchoring", "contiguous_range_start", "source_anchoring_for_view"]
