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
    ``content_key`` identifies tile content independent of anchored-axis
    windows: document identity (data revision + operation steps) plus the
    view identity with anchored axes' ranges stripped.
    """

    anchored_starts: tuple[int | None, int | None]
    content_key: object

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


def source_anchoring_for_view(document, view_state) -> SourceAnchoring | None:
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
    dtype = getattr(getattr(document, "base_data", None), "dtype", None)
    windowable = set(
        pipeline_windowable_display_axes(
            operations,
            tuple(int(size) for size in view_state.shape),
            dtype,
            display_axes=image_axes,
        )
    )

    anchored_starts: list[int | None] = []
    windowless = view_state
    for axis in image_axes:
        start: int | None = None
        if axis in windowable:
            indices = view_state.axis_range_indices[axis]
            start = 0 if indices is None else contiguous_range_start(indices)
        anchored_starts.append(start)
        if start is not None:
            windowless = windowless.with_axis_range(axis, None)

    # Deferred import: evaluator pulls in the operations planner stack, and
    # this module is imported by the Qt-free display layer.
    from arrayscope.operations.evaluator import _document_key, _request_key
    from arrayscope.operations.slabs import request_for_image

    return SourceAnchoring(
        anchored_starts=(anchored_starts[0], anchored_starts[1]),
        content_key=(
            "src-anchored",
            _document_key(document),
            _request_key(request_for_image(windowless)),
        ),
    )


__all__ = ["SourceAnchoring", "contiguous_range_start", "source_anchoring_for_view"]
