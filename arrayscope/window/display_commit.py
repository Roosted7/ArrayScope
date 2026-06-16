"""Single gateway from decided display presentations to ImageView2D."""

from __future__ import annotations

import numpy as np

from arrayscope.window.display_frame import CommittedDisplayFrame, DisplayFrameKey
from arrayscope.window.presentation import DisplayPresentation


class DisplayCommitter:
    def __init__(self, image_view):
        self.image_view = image_view

    def commit_full(self, presentation: DisplayPresentation, key: DisplayFrameKey) -> CommittedDisplayFrame:
        self._validate_presentation(presentation)
        self.image_view.setImagePresentation(
            presentation.data,
            histogramData=presentation.histogram_data,
            histogramPlotData=presentation.histogram_plot_data,
            levels=presentation.levels,
            histogramRange=presentation.histogram_range,
            viewport_policy=presentation.viewport_policy,
            rgb_already_windowed=presentation.rgb_already_windowed,
        )
        return self._frame_for(presentation, key)

    def commit_fast(self, presentation: DisplayPresentation, key: DisplayFrameKey) -> CommittedDisplayFrame:
        self._validate_presentation(presentation)
        current = getattr(self.image_view, "image", None)
        if current is None or tuple(np.shape(current)[:2]) != tuple(presentation.geometry.display_shape):
            raise ValueError("fast display commit requires an existing image with the same display shape")
        self.image_view.updateImagePresentationFast(
            presentation.data,
            histogramData=presentation.histogram_data,
            histogramPlotData=presentation.histogram_plot_data,
            levels=presentation.levels,
            histogramRange=presentation.histogram_range,
            rgb_already_windowed=presentation.rgb_already_windowed,
        )
        return self._frame_for(presentation, key)

    def _frame_for(self, presentation: DisplayPresentation, key: DisplayFrameKey) -> CommittedDisplayFrame:
        return CommittedDisplayFrame(
            data=presentation.data,
            histogram_data=presentation.histogram_data,
            geometry=presentation.geometry,
            levels=(float(presentation.levels[0]), float(presentation.levels[1])),
            histogram_range=(float(presentation.histogram_range[0]), float(presentation.histogram_range[1])),
            key=key,
        )

    def _validate_presentation(self, presentation: DisplayPresentation) -> None:
        shape = tuple(np.shape(presentation.data)[:2])
        if shape != tuple(presentation.geometry.display_shape):
            raise ValueError(f"display data shape {shape} does not match geometry {presentation.geometry.display_shape}")
        if presentation.histogram_data is not None and tuple(np.shape(presentation.histogram_data)[:2]) != shape:
            raise ValueError("histogram data shape does not match display data shape")
        if presentation.histogram_plot_data is not None and np.asarray(presentation.histogram_plot_data).size < 1:
            raise ValueError("histogram plot data must not be empty")
        self._validate_bounds("levels", presentation.levels)
        self._validate_bounds("histogram range", presentation.histogram_range)

    def _validate_bounds(self, label: str, bounds) -> None:
        try:
            low, high = bounds
            low = float(low)
            high = float(high)
        except Exception as exc:
            raise ValueError(f"{label} must be a pair of finite floats") from exc
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"{label} must be finite increasing bounds")
