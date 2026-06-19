"""Semantic rendering-backend boundary.

The display orchestration layer commits ArrayScope presentations through this
small protocol. Concrete graphics-library methods stay behind adapters so the
window/controller code does not need to know whether pixels are drawn by
PyQtGraph, VisPy, or a future backend.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from arrayscope.display.backend_contract import ImageViewBackendCapabilities, image_view_backend_capabilities

if TYPE_CHECKING:
    from arrayscope.window.render_model import DisplayRasterPresentation, DisplayTiledPresentation


class RasterCommitMode(Enum):
    """How an already-decided raster presentation should be applied."""

    FULL = "full"
    FAST = "fast"
    TILE_LAYER = "tile_layer"


@runtime_checkable
class ImageRenderBackend(Protocol):
    """Backend operations expressed in ArrayScope display semantics."""

    @property
    def view(self): ...

    @property
    def capabilities(self) -> ImageViewBackendCapabilities: ...

    def current_raster_shape(self) -> tuple[int, int] | None: ...

    def present_raster(self, presentation: "DisplayRasterPresentation", *, mode: RasterCommitMode) -> None: ...

    def present_tiled(self, presentation: "DisplayTiledPresentation") -> None: ...

    def set_profile_bounds(self, bounds: tuple[float, float, float, float]) -> None: ...


class ImageViewMethodBackendAdapter:
    """Migration adapter for the existing ImageView2D public methods.

    This is deliberately the only place where semantic presentations are
    translated to the legacy widget method vocabulary. PyQtGraph and VisPy
    adapters inherit this translation while retaining separate capability and
    implementation modules.
    """

    expected_backend_name: str | None = None

    def __init__(self, view):
        self._view = view
        self._capabilities = image_view_backend_capabilities(view)
        expected = self.expected_backend_name
        if expected is not None and self._capabilities.name != expected:
            raise ValueError(
                f"{type(self).__name__} requires backend {expected!r}, "
                f"got {self._capabilities.name!r}"
            )

    @property
    def view(self):
        return self._view

    @property
    def capabilities(self) -> ImageViewBackendCapabilities:
        return self._capabilities

    def current_raster_shape(self) -> tuple[int, int] | None:
        image = getattr(self._view, "image", None)
        if image is None:
            return None
        shape = tuple(np.shape(image)[:2])
        return (int(shape[0]), int(shape[1])) if len(shape) == 2 else None

    def present_raster(self, presentation: "DisplayRasterPresentation", *, mode: RasterCommitMode) -> None:
        mode = RasterCommitMode(mode)
        if mode is RasterCommitMode.FULL:
            self._present_raster_full(presentation)
            return
        if mode is RasterCommitMode.FAST:
            self._present_raster_fast(presentation)
            return
        if mode is RasterCommitMode.TILE_LAYER:
            self._present_legacy_raster_tile_layer(presentation)
            return
        raise ValueError(f"unsupported raster commit mode: {mode}")

    def present_tiled(self, presentation: "DisplayTiledPresentation") -> None:
        commit = getattr(self._view, "setTiledMontagePresentation", None)
        if not callable(commit):
            raise TypeError("image view does not implement first-class tiled presentation commits")
        commit(
            geometry=presentation.geometry,
            tile_state=presentation.tile_state,
            tile_delta=presentation.tile_delta,
            histogramPlotData=presentation.histogram_plot_data,
            levels=presentation.levels,
            histogramRange=presentation.histogram_range,
            viewport_policy=presentation.viewport_policy,
            rgb_already_windowed=presentation.rgb_already_windowed,
            tile_residency_budget_bytes=presentation.tile_residency_budget_bytes,
        )

    def set_profile_bounds(self, bounds: tuple[float, float, float, float]) -> None:
        setter = getattr(self._view, "setProfileMarkerBoundsRect", None)
        if callable(setter):
            setter(bounds)

    def _present_raster_full(self, presentation: "DisplayRasterPresentation") -> None:
        self._view.setImagePresentation(
            presentation.data,
            histogramData=presentation.histogram_data,
            histogramPlotData=presentation.histogram_plot_data,
            levels=presentation.levels,
            histogramRange=presentation.histogram_range,
            viewport_policy=presentation.viewport_policy,
            rgb_already_windowed=presentation.rgb_already_windowed,
            image_origin=_image_origin(presentation.geometry),
            shader_mapping=presentation.shader_mapping,
            texture_kind=presentation.texture_kind,
            semantic_data=presentation.semantic_data,
            lod=presentation.lod,
        )

    def _present_raster_fast(self, presentation: "DisplayRasterPresentation") -> None:
        self._view.updateImagePresentationFast(
            presentation.data,
            histogramData=presentation.histogram_data,
            histogramPlotData=presentation.histogram_plot_data,
            levels=presentation.levels,
            histogramRange=presentation.histogram_range,
            rgb_already_windowed=presentation.rgb_already_windowed,
            image_origin=_image_origin(presentation.geometry),
            shader_mapping=presentation.shader_mapping,
            texture_kind=presentation.texture_kind,
            semantic_data=presentation.semantic_data,
            lod=presentation.lod,
        )

    def _present_legacy_raster_tile_layer(self, presentation: "DisplayRasterPresentation") -> None:
        commit = getattr(self._view, "setMontageTileLayerPresentation", None)
        if not callable(commit):
            raise TypeError("image view does not implement montage tile-layer presentation commits")
        commit(
            presentation.data,
            histogramData=presentation.histogram_data,
            histogramPlotData=presentation.histogram_plot_data,
            geometry=presentation.geometry,
            levels=presentation.levels,
            histogramRange=presentation.histogram_range,
            viewport_policy=presentation.viewport_policy,
            rgb_already_windowed=presentation.rgb_already_windowed,
            montage_dirty_tiles=presentation.montage_dirty_tiles,
            montage_tile_source_ids=presentation.montage_tile_source_ids,
            montage_tile_payloads=None,
        )


def _image_origin(geometry) -> tuple[float, float]:
    if getattr(geometry, "montage", None) is None:
        return (0.0, 0.0)
    return (
        float(getattr(geometry, "montage_origin_x", 0)),
        float(getattr(geometry, "montage_origin_y", 0)),
    )
