"""Experimental VisPy-backed 2D image view.

This widget intentionally keeps ArrayScope's existing PyQtGraph interaction and
histogram layer while replacing the expensive pixel upload/display path with
VisPy visuals.  That makes the experiment low-risk: ROI/profile/HUD behaviour
continues to use the same ViewBox/world coordinate model, while scalar images can
use VisPy's GPU texture scaling via ``texture_format='auto'`` and ``clim``.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from arrayscope.core.runtime_diagnostics import ImageUploadTiming
from arrayscope.display.backend_contract import VISPY_CAPABILITIES
from arrayscope.display.imageview2d import ImageView2D
from arrayscope.display.imageview2d import _point_inside_view_range
from arrayscope.display.imageview2d import _is_tiled_loading_only_commit
from arrayscope.display.imageview2d import _tiled_montage_placeholder
from arrayscope.display.imageview2d import _tile_commit_report
from arrayscope.display.image_upload import rgb_display_for_levels
from arrayscope.display.interaction import CursorIntent, hit_test_display_overlays
from arrayscope.display.overlay_hit_test import hit_test_roi, roi_handle_points
from arrayscope.display.backends.vispy.raster import (
    GpuMappedImageVisual,
    _coerce_texture_kind,
    _contiguous_display,
    _contiguous_scalar,
    _normalize_levels,
)
from arrayscope.display.shader_mapping import TexturePlaneKind, common_shader_mapping, shader_mapping_with_lut
from arrayscope.display.viewport import ViewportPolicy, coerce_viewport_policy

if TYPE_CHECKING:
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState


@dataclass
class _VisPyTileState:
    image_visual: object | None = None
    windowed_visual: object | None = None
    visual: object | None = None
    source_id: object | None = None
    color_source_id: object | None = None
    scalar_source_id: object | None = None
    levels: tuple[float, float] | None = None
    data_shape: tuple[int, ...] | None = None
    visible: bool = False
    windowed_rgb: bool = False


class VisPyImageView2D(ImageView2D):
    """ImageView2D variant that renders pixels with VisPy.

    The class deliberately preserves the public ImageView2D API.  Existing
    renderer code can switch to it through the image-view factory without a new
    set of shims.  The first experimental version focuses on the hot path:
    scalar image/window-level display and montage tile-layer uploads.  PyQtGraph
    still owns the histogram widget, ROI editing, profile marker, HUD, and mouse
    interaction overlay.
    """

    rendering_backend_name = "vispy"
    rendering_capabilities = VISPY_CAPABILITIES
    supports_direct_montage_tile_payloads = rendering_capabilities.direct_montage_tile_payloads

    def setupUI(self):
        self._vispy_scene, self._vispy_visuals, self._vispy_transforms, self._vispy_panzoom_camera, self._vispy_gloo = _import_vispy()
        try:
            from vispy.app import use_app

            try:
                use_app("pyside6")
            except Exception:
                use_app()
        except Exception:
            pass

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self._display_container = QtWidgets.QWidget()
        self._display_stack = QtWidgets.QStackedLayout(self._display_container)
        self._display_stack.setContentsMargins(0, 0, 0, 0)
        self._display_stack.setStackingMode(QtWidgets.QStackedLayout.StackingMode.StackAll)

        self._vispy_canvas = self._vispy_scene.SceneCanvas(keys=None, bgcolor=(0, 0, 0, 1), show=False)
        self._vispy_view = self._vispy_canvas.central_widget.add_view()
        self._vispy_view.camera = self._vispy_panzoom_camera(aspect=1)
        self._vispy_view.camera.interactive = False
        self._vispy_view.camera.flip = (False, True, False)
        self._vispy_image = self._vispy_visuals.Image(
            np.zeros((1, 1), dtype=np.float32),
            parent=self._vispy_view.scene,
            interpolation="nearest",
            texture_format="auto",
            method="auto",
            clim=(0.0, 1.0),
        )
        self._vispy_image.transform = self._vispy_transforms.STTransform(translate=(0.0, 0.0, 0.0))
        self._vispy_windowed_image = self._vispy_scene.visuals.create_visual_node(GpuMappedImageVisual)(
            parent=self._vispy_view.scene
        )
        self._vispy_windowed_image.visible = False
        self._vispy_windowed_image.transform = self._vispy_transforms.STTransform(translate=(0.0, 0.0, 0.0))
        from arrayscope.display.backends.vispy.tiles import create_gpu_montage_layer

        self._vispy_gpu_montage_layer = create_gpu_montage_layer(
            scene=self._vispy_scene,
            visuals=self._vispy_visuals,
            gloo=self._vispy_gloo,
            transforms=self._vispy_transforms,
            parent=self._vispy_view.scene,
        )
        self._vispy_tile_visuals: dict[int, _VisPyTileState] = {}
        self._vispy_roi_visuals: dict[str, object] = {}
        self._vispy_roi_handle_visuals: dict[str, object] = {}
        self._vispy_selected_roi_id: str | None = None
        self._vispy_hovered_roi_id: str | None = None
        self._vispy_profile_hover_part: str | None = None
        self._vispy_roi_drawing_preview = None
        self._vispy_overlay_visuals: list[object] = []
        self._vispy_overlay_mesh = None
        self._vispy_overlay_lines = None
        self._vispy_overlay_key: tuple[object, ...] = ()
        self._vispy_overlay_count = 0
        self._vispy_profile_visuals: dict[str, object] = {}
        self._vispy_last_levels: tuple[float, float] = (0.0, 1.0)
        self._vispy_warm_tile_timer: QtCore.QTimer | None = None
        self._vispy_pending_warm_tile_payloads: dict[int, object] = {}
        self._vispy_pending_warm_tile_context: dict[str, object] = {}
        self._last_vispy_warm_tile_stats = None
        self._last_vispy_tiled_levels_key = None
        self._last_vispy_tiled_mapping_key = None
        self._last_vispy_tiled_source_shader_mapping = None
        self._last_vispy_tiled_shader_mapping = None
        self._last_vispy_tiled_histogram_key = None
        self._last_vispy_tiled_viewport_key = None
        self._vispy_main_data_id: int | None = None
        self._vispy_main_color_source_id: int | None = None
        self._vispy_main_scalar_source_id: int | None = None
        self._last_vispy_main_source_shader_mapping = None
        self._last_vispy_main_shader_mapping = None
        self._last_vispy_main_texture_kind = None
        self._vispy_display_shape: tuple[int, int] = (1, 1)
        self._vispy_roi_cursor_active = False
        self._vispy_camera_sync_pending = False
        self._vispy_camera_key = None
        self._vispy_canvas_native = self._vispy_canvas.native
        self._vispy_canvas_native.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._display_stack.addWidget(self._vispy_canvas_native)

        self.graphicsView = pg.GraphicsView()
        self.graphicsView.setBackground(None)
        self.graphicsView.setStyleSheet("background: transparent; border: 0px;")
        self.graphicsView.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.graphicsView.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.graphicsView.viewport().setStyleSheet("background: transparent;")
        self._display_stack.addWidget(self.graphicsView)
        self.layout.addWidget(self._display_container, 1)

        self.histogram = pg.HistogramLUTWidget()
        self.layout.addWidget(self.histogram)

        self._vispy_canvas.events.mouse_move.connect(self._on_vispy_mouse_move)

    def __init__(self, parent=None, view=None, imageItem=None):
        super().__init__(parent=parent, view=view, imageItem=imageItem)
        self.imageItem.setVisible(False)
        self.histogramImageItem.setVisible(False)
        self._vispy_bounds_item = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
        self._vispy_bounds_item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self._vispy_bounds_item.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
        self._vispy_bounds_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self._layer_owner.add_bounds_item(self._vispy_bounds_item)
        self.view.sigRangeChanged.connect(lambda *_args: self._request_vispy_camera_sync())
        state_signal = getattr(self.view, "sigStateChanged", None)
        if state_signal is not None:
            state_signal.connect(lambda *_args: self._request_vispy_camera_sync())


    def _display_overlay_parent(self):
        return getattr(self, "_display_container", self.graphicsView)

    def _map_scene_to_display_overlay(self, scene_pos):
        local = self.graphicsView.mapFromScene(scene_pos)
        parent = self._display_overlay_parent()
        if parent is self.graphicsView:
            return local
        return self.graphicsView.mapTo(parent, local)

    def clearMontageTileLayer(self) -> None:
        for state in getattr(self, "_vispy_tile_visuals", {}).values():
            _set_visual_visible(state.image_visual, False)
            _set_visual_visible(state.windowed_visual, False)
            state.visual = None
            state.visible = False
        layer = getattr(self, "_vispy_gpu_montage_layer", None)
        if layer is not None:
            layer.clear()
        self.clearMontageTileOverlays()
        self._last_vispy_tile_payloads = None
        self._last_vispy_tiled_source_key = None
        self._last_vispy_tiled_structure_key = None
        self._last_vispy_tiled_levels_key = None
        self._last_vispy_tiled_mapping_key = None
        self._last_vispy_tiled_source_shader_mapping = None
        self._last_vispy_tiled_shader_mapping = None
        self._last_vispy_tiled_histogram_key = None
        self._last_vispy_tiled_viewport_key = None
        self._montage_display_mode = "canvas"
        self.imageItem.setVisible(False)
        _set_visual_visible(getattr(self, "_vispy_image", None), False)
        _set_visual_visible(getattr(self, "_vispy_windowed_image", None), False)

    def clear(self):
        super().clear()
        self.clearMontageTileLayer()
        self._vispy_main_data_id = None
        self._vispy_main_color_source_id = None
        self._vispy_main_scalar_source_id = None
        self._last_vispy_main_source_shader_mapping = None
        self._last_vispy_main_shader_mapping = None
        self._last_vispy_main_texture_kind = None

    def setColorMap(self, colormap):
        """Update the shared colorbar/render-surface LUT without re-uploading pixels."""

        super().setColorMap(colormap)
        self._apply_vispy_native_colormap()
        texture_kind = getattr(self, "_last_vispy_main_texture_kind", None)
        if texture_kind in {TexturePlaneKind.SCALAR_R32F, TexturePlaneKind.COMPLEX_RG32F}:
            mapping = self._display_shader_mapping(getattr(self, "_last_vispy_main_source_shader_mapping", None))
            self._last_vispy_main_shader_mapping = mapping
            visual = getattr(self, "_vispy_windowed_image", None)
            if visual is not None:
                visual.set_shader_mapping(mapping)
        if getattr(self, "_montage_display_mode", "") == "vispy_tile_layer":
            mapping = self._display_shader_mapping(getattr(self, "_last_vispy_tiled_source_shader_mapping", None))
            self._last_vispy_tiled_shader_mapping = mapping
            self._last_vispy_tiled_mapping_key = _shader_mapping_key(mapping)
            layer = getattr(self, "_vispy_gpu_montage_layer", None)
            if layer is not None:
                layer.set_shader_mapping(mapping)
        canvas = getattr(self, "_vispy_canvas", None)
        if canvas is not None:
            canvas.update()

    def _display_shader_mapping(self, mapping):
        return shader_mapping_with_lut(
            mapping,
            self.displayColorMapLookupTable(),
            lut_identity=self.displayColorMapKey(),
        )

    def _apply_vispy_native_colormap(self) -> None:
        try:
            from vispy.color import Colormap

            lut = np.asarray(self.displayColorMapLookupTable(), dtype=np.float32) / 255.0
            colormap = Colormap(lut)
            self._vispy_native_colormap = colormap
            image = getattr(self, "_vispy_image", None)
            if image is not None:
                image.cmap = colormap
            for state in getattr(self, "_vispy_tile_visuals", {}).values():
                visual = getattr(state, "image_visual", None)
                if visual is not None and len(tuple(getattr(state, "data_shape", ()) or ())) == 2:
                    visual.cmap = colormap
        except Exception:
            pass

    def _apply_vispy_colormap_to_visual(self, visual) -> None:
        colormap = getattr(self, "_vispy_native_colormap", None)
        if colormap is None:
            self._apply_vispy_native_colormap()
            colormap = getattr(self, "_vispy_native_colormap", None)
        if colormap is not None:
            try:
                visual.cmap = colormap
            except Exception:
                pass

    def setImage(
        self,
        img,
        autoRange=None,
        autoLevels=True,
        levels=None,
        pos=None,
        scale=None,
        transform=None,
        autoHistogramRange=True,
        histogramData=None,
        histogramPlotData=None,
        viewport_policy=ViewportPolicy.PRESERVE,
        rgb_already_windowed: bool = False,
        image_origin: tuple[float, float] = (0.0, 0.0),
        shader_mapping=None,
        texture_kind=None,
        semantic_data: np.ndarray | None = None,
        lod=None,
    ):
        if not isinstance(img, np.ndarray):
            raise TypeError("Image must be a numpy array")
        viewport_policy = coerce_viewport_policy(viewport_policy, autoRange)
        self._start_upload_timing("vispy_full")
        applying = self._applying_presentation
        self._applying_presentation = True
        try:
            self.clearMontageTileLayer()
            self.image = img
            self.histogramSource = histogramData
            self.histogramPlotSource = histogramPlotData
            display_levels = _normalize_levels(levels, self._displayLevels or (0.0, 1.0))
            self._upload_vispy_main_image(
                img,
                histogramData=histogramData,
                levels=display_levels,
                image_origin=image_origin,
                rgb_already_windowed=rgb_already_windowed,
                shader_mapping=shader_mapping,
                texture_kind=texture_kind,
                semantic_data=semantic_data,
                lod=lod,
            )
            self._update_histogram_for_vispy(histogramData, histogramPlotData, display_levels)
            self._sync_display_levels(display_levels[0], display_levels[1], update_image=False, emit_user=False)
            if autoHistogramRange:
                bounds = self.getHistogramDataBounds() or display_levels
                self.histogram.setHistogramRange(float(bounds[0]), float(bounds[1]))
            self._update_profile_line_bounds()
            self._updateAspectRatio()
            self._sync_vispy_bounds(tuple(img.shape[:2]), image_origin=image_origin)
            self._apply_viewport_policy(tuple(img.shape[:2]), viewport_policy, image_origin=image_origin)
            self._sync_vispy_camera_to_view()
        finally:
            self._applying_presentation = applying
            self._finish_upload_timing()

    def updateImageDataFast(
        self,
        img: np.ndarray,
        *,
        histogramData: np.ndarray | None = None,
        histogramPlotData: np.ndarray | None = None,
        levels: tuple[float, float] | None = None,
        histogramRange: tuple[float, float] | None = None,
        rgb_already_windowed: bool = False,
        image_origin: tuple[float, float] = (0.0, 0.0),
        shader_mapping=None,
        texture_kind=None,
        semantic_data: np.ndarray | None = None,
        lod=None,
    ) -> None:
        if self.image is None:
            raise RuntimeError("fast image update requires an existing image")
        if tuple(img.shape[:2]) != tuple(self.image.shape[:2]):
            raise ValueError("fast image update requires the same display shape")
        self._start_upload_timing("vispy_fast")
        applying = self._applying_presentation
        self._applying_presentation = True
        try:
            self.clearMontageTileLayer()
            self.image = img
            self.histogramSource = histogramData
            self.histogramPlotSource = histogramPlotData
            if histogramRange is not None:
                self.setHistogramDataBounds(histogramRange)
            display_levels = _normalize_levels(levels, self._displayLevels or (0.0, 1.0))
            self._upload_vispy_main_image(
                img,
                histogramData=histogramData,
                levels=display_levels,
                image_origin=image_origin,
                rgb_already_windowed=rgb_already_windowed,
                shader_mapping=shader_mapping,
                texture_kind=texture_kind,
                semantic_data=semantic_data,
                lod=lod,
            )
            self._update_histogram_for_vispy(histogramData, histogramPlotData, display_levels)
            self._sync_display_levels(display_levels[0], display_levels[1], update_image=False, emit_user=False)
            if histogramRange is not None:
                self.histogram.setHistogramRange(float(histogramRange[0]), float(histogramRange[1]))
            self._update_profile_line_bounds()
            self._sync_vispy_bounds(tuple(img.shape[:2]), image_origin=image_origin)
            self._sync_vispy_camera_to_view()
        finally:
            self._applying_presentation = applying
            self._finish_upload_timing()

    def setMontageTileLayerPresentation(
        self,
        img: np.ndarray,
        *,
        histogramData: np.ndarray | None,
        histogramPlotData: np.ndarray | None,
        geometry,
        levels: tuple[float, float],
        histogramRange: tuple[float, float],
        viewport_policy=ViewportPolicy.PRESERVE,
        rgb_already_windowed: bool = False,
        montage_dirty_tiles: tuple[int, ...] | None = None,
        montage_tile_source_ids: dict[int, object] | None = None,
        montage_tile_payloads: dict[int, "DisplayTilePayload"] | None = None,
        shader_mapping=None,
        tile_delta: "TilePresentationDelta | None" = None,
        tile_residency_budget_bytes: int = 0,
    ) -> None:
        if geometry is None or getattr(geometry, "montage", None) is None:
            raise ValueError("tile-layer presentation requires montage geometry")
        self._apply_vispy_tile_layer_presentation(
            img,
            histogramData=histogramData,
            histogramPlotData=histogramPlotData,
            geometry=geometry,
            levels=levels,
            histogramRange=histogramRange,
            viewport_policy=viewport_policy,
            rgb_already_windowed=rgb_already_windowed,
            montage_dirty_tiles=montage_dirty_tiles,
            montage_tile_source_ids=montage_tile_source_ids,
            montage_tile_payloads=montage_tile_payloads,
            shader_mapping=shader_mapping,
            tile_delta=tile_delta,
            tile_residency_budget_bytes=tile_residency_budget_bytes,
        )

    def _apply_vispy_tile_layer_presentation(
        self,
        img: np.ndarray,
        *,
        histogramData: np.ndarray | None,
        histogramPlotData: np.ndarray | None,
        geometry,
        levels: tuple[float, float],
        histogramRange: tuple[float, float],
        viewport_policy=ViewportPolicy.PRESERVE,
        rgb_already_windowed: bool = False,
        montage_dirty_tiles: tuple[int, ...] | None = None,
        montage_tile_source_ids: dict[int, object] | None = None,
        montage_tile_payloads: dict[int, "DisplayTilePayload"] | None = None,
        shader_mapping=None,
        tile_delta: "TilePresentationDelta | None" = None,
        tile_residency_budget_bytes: int = 0,
    ) -> None:
        self._start_upload_timing("vispy_tile_layer")
        applying = self._applying_presentation
        self._applying_presentation = True
        try:
            if shader_mapping is None and montage_tile_payloads:
                shader_mapping = common_shader_mapping(
                    getattr(payload, "shader_mapping", None)
                    for payload in montage_tile_payloads.values()
                )
            source_shader_mapping = shader_mapping
            shader_mapping = self._display_shader_mapping(source_shader_mapping)
            level_key = (float(levels[0]), float(levels[1]))
            mapping_key = _shader_mapping_key(shader_mapping)
            source_key = _tiled_source_key(montage_tile_payloads, montage_tile_source_ids)
            structure_key = _tiled_structure_key(
                geometry,
                rgb_already_windowed=rgb_already_windowed,
            )
            histogram_key = _tiled_histogram_key(histogramData, histogramPlotData, histogramRange)
            viewport_key = (
                structure_key,
                str(getattr(viewport_policy, "value", viewport_policy)),
            )
            previous_source_key = getattr(self, "_last_vispy_tiled_source_key", None)
            previous_structure_key = getattr(self, "_last_vispy_tiled_structure_key", None)
            previous_levels_key = getattr(self, "_last_vispy_tiled_levels_key", None)
            previous_mapping_key = getattr(self, "_last_vispy_tiled_mapping_key", None)
            previous_histogram_key = getattr(self, "_last_vispy_tiled_histogram_key", None)
            previous_viewport_key = getattr(self, "_last_vispy_tiled_viewport_key", None)
            structure_changed = structure_key != previous_structure_key
            levels_changed = level_key != previous_levels_key
            mapping_changed = mapping_key != previous_mapping_key
            loading_only = _is_tiled_loading_only_commit(
                montage_tile_payloads,
                histogramData=histogramData,
                histogramPlotData=histogramPlotData,
            )
            previous_layer_stats = getattr(getattr(self, "_vispy_gpu_montage_layer", None), "last_stats", None)
            must_clear_visible_pages = bool(
                loading_only
                and (
                    int(getattr(previous_layer_stats, "visible_items", 0) or 0) > 0
                    or int(getattr(previous_layer_stats, "active_pages", 0) or 0) > 0
                )
            )
            data_unchanged = (
                montage_tile_payloads is not None
                and montage_dirty_tiles == ()
                and source_key == previous_source_key
                and not structure_changed
                and not must_clear_visible_pages
            )

            self.image = img
            if not loading_only:
                self.histogramSource = histogramData
                self.histogramPlotSource = histogramPlotData
            self._last_vispy_tile_payloads = montage_tile_payloads
            if not loading_only:
                self.setHistogramDataBounds(histogramRange)
            self._montage_display_mode = "vispy_tile_layer"
            try:
                self._vispy_image.visible = False
            except Exception:
                pass
            _set_visual_visible(getattr(self, "_vispy_windowed_image", None), False)

            if data_unchanged and not levels_changed and not mapping_changed:
                from arrayscope.display.backends.pyqtgraph.tiles import TileLayerUpdateStats

                visible = len(montage_tile_payloads or {})
                stats = TileLayerUpdateStats(
                    visible_items=visible,
                    items_updated=0,
                    items_skipped=visible,
                    rgb_window_tiles=0,
                    resident_items=int(getattr(previous_layer_stats, "resident_items", 0) or 0),
                    storage_capacity=int(getattr(previous_layer_stats, "atlas_capacity", 0) or 0),
                    estimated_gpu_bytes=int(getattr(previous_layer_stats, "estimated_gpu_bytes", 0) or 0),
                    cpu_shadow_bytes=int(getattr(previous_layer_stats, "cpu_shadow_bytes", 0) or 0),
                    page_count=int(getattr(previous_layer_stats, "page_count", 0) or 0),
                    active_pages=int(getattr(previous_layer_stats, "active_pages", 0) or 0),
                    device_max_texture_size=int(getattr(previous_layer_stats, "device_max_texture_size", 0) or 0),
                    budget_bytes=int(getattr(previous_layer_stats, "budget_bytes", 0) or 0),
                    near_resident_items=int(getattr(previous_layer_stats, "near_resident_items", 0) or 0),
                    warm_resident_items=int(getattr(previous_layer_stats, "warm_resident_items", 0) or 0),
                )
            else:
                stats = self._update_vispy_tile_layer(
                    img,
                    histogram_data=histogramData,
                    geometry=geometry,
                    levels=level_key,
                    rgb_already_windowed=rgb_already_windowed,
                    dirty_tiles=montage_dirty_tiles,
                    tile_source_ids=montage_tile_source_ids,
                    tile_payloads=montage_tile_payloads,
                    shader_mapping=shader_mapping,
                    tile_delta=tile_delta,
                    tile_residency_budget_bytes=tile_residency_budget_bytes,
                    force_levels=bool(data_unchanged and levels_changed),
                    force_mapping=bool(data_unchanged and mapping_changed),
                )
            self._record_tile_layer_stats(stats)

            # Histogram, levels, geometry, and viewport are separate concerns.
            # A level-only change must not look like a full structural commit.
            histogram_changed = histogram_key != previous_histogram_key or not data_unchanged
            if histogram_changed and not loading_only:
                self._update_histogram_for_vispy(histogramData, histogramPlotData, level_key)
                self.histogram.setHistogramRange(float(histogramRange[0]), float(histogramRange[1]))
            if levels_changed and not loading_only:
                self._sync_display_levels(level_key[0], level_key[1], update_image=False, emit_user=False)

            montage_shape = None
            if structure_changed:
                self._update_profile_line_bounds()
                self._updateAspectRatio()
                montage_shape = self._sync_vispy_montage_bounds(geometry)
            if viewport_key != previous_viewport_key:
                if montage_shape is None:
                    montage_shape = self._sync_vispy_montage_bounds(geometry)
                self._apply_viewport_policy(montage_shape, viewport_policy, image_origin=(0.0, 0.0))
                self._sync_vispy_camera_to_view()

            self._last_vispy_tiled_source_key = source_key
            self._last_vispy_tiled_structure_key = structure_key
            if not loading_only:
                self._last_vispy_tiled_levels_key = level_key
                self._last_vispy_tiled_mapping_key = mapping_key
                self._last_vispy_tiled_source_shader_mapping = source_shader_mapping
                self._last_vispy_tiled_shader_mapping = shader_mapping
                self._last_vispy_tiled_histogram_key = histogram_key
            self._last_vispy_tiled_viewport_key = viewport_key
            return stats
        finally:
            self._applying_presentation = applying
            self._finish_upload_timing()

    def setTiledMontagePresentation(
        self,
        *,
        geometry,
        tile_state: "TilePresentationState",
        tile_delta: "TilePresentationDelta",
        histogramPlotData: np.ndarray | None,
        levels: tuple[float, float],
        histogramRange: tuple[float, float],
        viewport_policy=ViewportPolicy.PRESERVE,
        rgb_already_windowed: bool = False,
        shader_mapping=None,
        tile_residency_budget_bytes: int = 0,
    ) -> None:
        tile_payloads = tile_state.active_payloads(tile_delta)
        warm_payloads = {
            int(tile): payload
            for tile, payload in tile_state.near_payloads(tile_delta).items()
            if int(tile) not in tile_payloads
        }
        dirty_tiles = None if tile_delta.force_refresh else ()
        placeholder = _tiled_montage_placeholder(geometry.display_shape, tile_payloads)
        stats = self._apply_vispy_tile_layer_presentation(
            placeholder,
            histogramData=None,
            histogramPlotData=histogramPlotData,
            geometry=geometry,
            levels=levels,
            histogramRange=histogramRange,
            viewport_policy=viewport_policy,
            rgb_already_windowed=rgb_already_windowed,
            montage_dirty_tiles=dirty_tiles,
            montage_tile_source_ids={key: payload.source_id for key, payload in tile_payloads.items()},
            montage_tile_payloads=tile_payloads,
            shader_mapping=shader_mapping,
            tile_delta=tile_delta,
            tile_residency_budget_bytes=tile_residency_budget_bytes,
        )
        self._schedule_vispy_warm_tile_residency(
            warm_payloads,
            geometry=geometry,
            rgb_already_windowed=rgb_already_windowed,
            tile_delta=tile_delta,
            tile_residency_budget_bytes=tile_residency_budget_bytes,
        )
        return _tile_commit_report(tile_payloads, tile_delta, stats)

    def _schedule_vispy_warm_tile_residency(
        self,
        payloads,
        *,
        geometry,
        rgb_already_windowed: bool,
        tile_delta,
        tile_residency_budget_bytes: int,
    ) -> None:
        payloads = {int(key): value for key, value in dict(payloads or {}).items()}
        if not payloads:
            return
        self._vispy_pending_warm_tile_payloads = payloads
        self._vispy_pending_warm_tile_context = {
            "geometry": geometry,
            "rgb_already_windowed": bool(rgb_already_windowed),
            "tile_delta": tile_delta,
            "tile_residency_budget_bytes": int(tile_residency_budget_bytes),
        }
        timer = self._vispy_warm_tile_timer
        if timer is None:
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._process_vispy_warm_tile_residency)
            self._vispy_warm_tile_timer = timer
        # Warm uploads are speculative and are queued after the visible commit
        # returns, so they cannot delay the first useful tiled presentation.
        timer.start(0)

    def _process_vispy_warm_tile_residency(self) -> None:
        payloads = dict(getattr(self, "_vispy_pending_warm_tile_payloads", {}) or {})
        context = dict(getattr(self, "_vispy_pending_warm_tile_context", {}) or {})
        if not payloads:
            return
        from arrayscope.display.backends.vispy.tiles import take_payload_batch

        batch, remaining = take_payload_batch(payloads)
        self._vispy_pending_warm_tile_payloads = remaining
        self._vispy_pending_warm_tile_context = context if remaining else {}
        layer = getattr(self, "_vispy_gpu_montage_layer", None)
        if layer is None or not hasattr(layer, "warm_residency"):
            self._vispy_pending_warm_tile_payloads = {}
            self._vispy_pending_warm_tile_context = {}
            return
        try:
            self._last_vispy_warm_tile_stats = layer.warm_residency(
                payloads=batch,
                geometry=context.get("geometry"),
                rgb_already_windowed=bool(context.get("rgb_already_windowed", False)),
                tile_delta=context.get("tile_delta"),
                tile_residency_budget_bytes=int(context.get("tile_residency_budget_bytes", 0) or 0),
            )
        except Exception:
            self._vispy_pending_warm_tile_payloads = {}
            self._vispy_pending_warm_tile_context = {}
            return
        if remaining:
            timer = self._vispy_warm_tile_timer
            if timer is not None:
                timer.start(8)

    def _apply_histogram_preview_levels(self, levels) -> None:
        levels = (float(levels[0]), float(levels[1]))
        started_timing = self._upload_timing is None
        if started_timing:
            self._start_upload_timing("vispy_level_preview")
        try:
            self._displayLevels = levels
            self._vispy_last_levels = levels
            if self._montage_display_mode == "vispy_tile_layer":
                stats = self._update_vispy_tile_layer(
                    self.image,
                    histogram_data=self.histogramSource,
                    geometry=getattr(self, "_last_vispy_geometry", None),
                    levels=levels,
                    rgb_already_windowed=False,
                    dirty_tiles=(),
                    tile_source_ids=None,
                    tile_payloads=getattr(self, "_last_vispy_tile_payloads", None),
                    shader_mapping=getattr(self, "_last_vispy_tiled_shader_mapping", None),
                    tile_delta=None,
                    tile_residency_budget_bytes=0,
                    force_levels=True,
                )
                self._record_tile_layer_stats(stats)
                return
            if self._is_windowed_rgb_vispy_main():
                self._vispy_windowed_image.set_levels(levels)
                self._vispy_canvas.update()
            elif self._is_rgb_image(self.image):
                self._upload_vispy_main_image(self.image, histogramData=self.histogramSource, levels=levels, image_origin=getattr(self, "_last_vispy_origin", (0.0, 0.0)))
            else:
                try:
                    self._vispy_image.clim = levels
                except Exception:
                    pass
        finally:
            if started_timing:
                self._finish_upload_timing()

    def _upload_vispy_main_image(
        self,
        img,
        *,
        histogramData,
        levels,
        image_origin=(0.0, 0.0),
        rgb_already_windowed=False,
        shader_mapping=None,
        texture_kind=None,
        semantic_data=None,
        lod=None,
    ):
        start = perf_counter()
        del lod
        texture_kind = _coerce_texture_kind(texture_kind)
        if texture_kind in {TexturePlaneKind.COMPLEX_RG32F, TexturePlaneKind.SCALAR_R32F} and semantic_data is not None:
            source_shader_mapping = shader_mapping
            shader_mapping = self._display_shader_mapping(source_shader_mapping)
            self._last_vispy_main_source_shader_mapping = source_shader_mapping
            self._last_vispy_main_shader_mapping = shader_mapping
            self._last_vispy_main_texture_kind = texture_kind
            data = self._upload_vispy_mapped_image(
                semantic_data,
                texture_kind=texture_kind,
                levels=levels,
                image_origin=image_origin,
                visual=self._vispy_windowed_image,
                shader_mapping=shader_mapping,
            )
            previous = self._vispy_main_data_id
            same_object = previous == id(data)
            self._vispy_main_data_id = id(data)
            self._vispy_image.visible = False
            self._vispy_windowed_image.visible = True
        elif self._should_use_windowed_rgb(img, histogramData, rgb_already_windowed=rgb_already_windowed):
            self._last_vispy_main_source_shader_mapping = None
            self._last_vispy_main_shader_mapping = None
            self._last_vispy_main_texture_kind = None
            data = self._upload_vispy_windowed_rgb(
                img,
                histogramData,
                levels,
                image_origin=image_origin,
                visual=self._vispy_windowed_image,
            )
            previous = self._vispy_main_data_id
            same_object = previous == id(data)
            self._vispy_main_data_id = id(data)
            self._vispy_image.visible = False
            self._vispy_windowed_image.visible = True
        else:
            self._last_vispy_main_source_shader_mapping = None
            self._last_vispy_main_shader_mapping = None
            self._last_vispy_main_texture_kind = None
            data = self._vispy_display_data(img, histogramData, levels, rgb_already_windowed=rgb_already_windowed)
            previous = self._vispy_main_data_id
            same_object = previous == id(data)
            self._vispy_image.set_data(data, copy=False)
            self._vispy_main_data_id = id(data)
            if data.ndim == 2:
                self._vispy_image.clim = (float(levels[0]), float(levels[1]))
                self._apply_vispy_native_colormap()
            self._vispy_image.transform = self._vispy_transforms.STTransform(translate=(float(image_origin[0]), float(image_origin[1]), 0.0))
            self._vispy_image.visible = True
            _set_visual_visible(self._vispy_windowed_image, False)
        self._last_vispy_origin = (float(image_origin[0]), float(image_origin[1]))
        elapsed = (perf_counter() - start) * 1000.0
        self._record_upload_timing("visible_upload_ms", elapsed)
        timing = self._upload_timing
        if timing is not None:
            array = np.asarray(data)
            timing["visible_bytes"] = int(timing["visible_bytes"]) + int(array.nbytes)
            timing["visible_pixels"] = int(timing["visible_pixels"]) + int(np.prod(array.shape[:2]))
            timing["fast_same_object"] = bool(timing["fast_same_object"] or same_object)

    def _vispy_display_data(self, img, histogramData, levels, *, rgb_already_windowed=False, timing_field="rgb_window_ms"):
        if self._is_rgb_image(img):
            if rgb_already_windowed:
                self.imageDisp = np.asarray(img[..., :3])
                return _contiguous_display(self.imageDisp)
            rgb_start = perf_counter()
            base = np.asarray(img[..., :3], dtype=np.float32)
            source = histogramData if histogramData is not None else self._histogram_data(base)
            self.imageDisp = rgb_display_for_levels(base, source, levels)
            self._record_upload_timing(timing_field, (perf_counter() - rgb_start) * 1000.0)
            return _contiguous_display(self.imageDisp)
        self.imageDisp = np.asarray(img)
        return _contiguous_display(self.imageDisp)

    def _should_use_windowed_rgb(self, img, histogramData, *, rgb_already_windowed: bool) -> bool:
        return self._is_rgb_image(img) and not bool(rgb_already_windowed) and histogramData is not None

    def _upload_vispy_windowed_rgb(self, img, histogramData, levels, *, image_origin, visual):
        color = _contiguous_display(np.asarray(img)[..., :3])
        scalar = _contiguous_scalar(histogramData)
        self.imageDisp = color
        color_source_id = id(color)
        scalar_source_id = id(scalar)
        visual.set_data(
            color,
            scalar,
            levels=levels,
            color_source_id=color_source_id,
            scalar_source_id=scalar_source_id,
            copy=False,
        )
        visual.transform = self._vispy_transforms.STTransform(translate=(float(image_origin[0]), float(image_origin[1]), 0.0))
        self._vispy_main_color_source_id = color_source_id
        self._vispy_main_scalar_source_id = scalar_source_id
        return color

    def _upload_vispy_mapped_image(self, data, *, texture_kind: TexturePlaneKind, levels, image_origin, visual, shader_mapping=None):
        texture_kind = _coerce_texture_kind(texture_kind)
        source_id = (id(data), getattr(texture_kind, "value", texture_kind))
        self.imageDisp = np.asarray(data)
        visual.set_mapped_data(
            data,
            texture_kind=texture_kind,
            levels=levels,
            source_id=source_id,
            shader_mapping=shader_mapping,
            copy=False,
        )
        visual.transform = self._vispy_transforms.STTransform(translate=(float(image_origin[0]), float(image_origin[1]), 0.0))
        self._vispy_main_scalar_source_id = source_id
        return np.asarray(data)

    def _is_windowed_rgb_vispy_main(self) -> bool:
        visual = getattr(self, "_vispy_windowed_image", None)
        return bool(visual is not None and getattr(visual, "visible", False))

    def createRoi(self, kind, *, points=None, rect=None, line_width=1.0, label=None, color=None):
        selection = super().createRoi(kind, points=points, rect=rect, line_width=line_width, label=label, color=color)
        self._upsert_vispy_roi(selection.id, selection.geometry, selection.color)
        return selection

    def removeRoi(self, roi_id):
        removed = super().removeRoi(roi_id)
        if removed:
            if self._vispy_selected_roi_id == str(roi_id):
                self._vispy_selected_roi_id = None
            if self._vispy_hovered_roi_id == str(roi_id):
                self._vispy_hovered_roi_id = None
            self._remove_vispy_roi(roi_id)
        return removed

    def clearRois(self):
        super().clearRois()
        for roi_id in tuple(getattr(self, "_vispy_roi_visuals", {})):
            self._remove_vispy_roi(roi_id)

    def highlightRoi(self, roi_id):
        result = super().highlightRoi(roi_id)
        self._vispy_selected_roi_id = str(roi_id) if result else None
        for current_id, (_item, selection) in self._roi_items.items():
            self._upsert_vispy_roi(current_id, selection.geometry, selection.color)
        return result

    def _on_roi_item_changed(self, roi_id, *, final: bool = True):
        self._sync_roi_item_state(roi_id, emit=True, final=final)

    def _sync_roi_item_state(self, roi_id, *, emit: bool, final: bool = True) -> None:
        item_selection = self._roi_items.get(str(roi_id))
        if item_selection is None:
            self._remove_vispy_roi(roi_id)
            return
        _item, selection = item_selection
        if emit:
            super()._on_roi_item_changed(roi_id, final=final)
            item_selection = self._roi_items.get(str(roi_id))
            if item_selection is None:
                self._remove_vispy_roi(roi_id)
                return
            _item, selection = item_selection
        self._upsert_vispy_roi(selection.id, selection.geometry, selection.color)

    def _upsert_vispy_roi(self, roi_id, geometry, color, *, width: float | None = None) -> None:
        points = _vispy_roi_points(geometry)
        if points is None:
            self._remove_vispy_roi(roi_id)
            return
        roi_id = str(roi_id)
        if width is None:
            width = self._vispy_roi_width(roi_id)
        line_color = self._vispy_roi_color(roi_id, color)
        visual = self._vispy_roi_visuals.get(str(roi_id))
        if visual is None:
            visual = self._vispy_visuals.Line(
                points,
                parent=self._vispy_view.scene,
                color=line_color,
                width=float(width),
                method="agg",
            )
            visual.order = 10_000
            try:
                visual.set_gl_state("translucent", depth_test=False)
            except Exception:
                pass
            self._vispy_roi_visuals[str(roi_id)] = visual
        else:
            visual.set_data(pos=points, color=line_color, width=float(width))
            visual.order = 10_000
        visual.visible = True
        self._upsert_vispy_roi_handles(roi_id, geometry, color)
        self._vispy_canvas.update()

    def _remove_vispy_roi(self, roi_id) -> None:
        visual = self._vispy_roi_visuals.pop(str(roi_id), None)
        handle_visuals = self._vispy_roi_handle_visuals.pop(str(roi_id), ())
        if handle_visuals is None or not isinstance(handle_visuals, (list, tuple)):
            handle_visuals = (handle_visuals,)
        for current in (visual, *tuple(handle_visuals)):
            if current is None:
                continue
            try:
                current.parent = None
            except Exception:
                try:
                    current.visible = False
                except Exception:
                    pass
        self._vispy_canvas.update()

    def _upsert_vispy_roi_handles(self, roi_id, geometry, color) -> None:
        roi_id = str(roi_id)
        points = np.asarray(roi_handle_points(geometry), dtype=np.float32).reshape((-1, 2))
        existing = self._vispy_roi_handle_visuals.get(roi_id, ())
        if existing is None or not isinstance(existing, (list, tuple)):
            existing = (existing,)
        marker = existing[0] if existing else None
        for stale in tuple(existing[1:]):
            if stale is not None:
                try:
                    stale.parent = None
                except Exception:
                    _set_visual_visible(stale, False)
        if not len(points):
            if marker is not None:
                _set_visual_visible(marker, False)
            self._vispy_roi_handle_visuals.pop(roi_id, None)
            return
        if marker is None or not hasattr(marker, "set_data"):
            marker = self._vispy_visuals.Markers(parent=self._vispy_view.scene)
        selected = roi_id == self._vispy_selected_roi_id
        hovered = roi_id == self._vispy_hovered_roi_id
        marker.set_data(
            points,
            symbol="square",
            size=12.0 if selected or hovered else 10.0,
            face_color=(0.05, 0.05, 0.05, 0.75),
            edge_color=_vispy_color((255, 255, 255) if hovered else color),
            edge_width=2.0 if selected or hovered else 1.25,
        )
        marker.order = 10_001
        marker.visible = True
        self._vispy_roi_handle_visuals[roi_id] = [marker]

    def _vispy_roi_width(self, roi_id: str) -> float:
        if str(roi_id) == self._vispy_selected_roi_id:
            return 4.0
        if str(roi_id) == self._vispy_hovered_roi_id:
            return 3.25
        return 2.0

    def _vispy_roi_color(self, roi_id: str, color):
        if str(roi_id) == self._vispy_hovered_roi_id:
            rgb = tuple(min(255, int(value) + 70) for value in tuple(color or (255, 255, 0))[:3])
            return _vispy_color(rgb)
        return _vispy_color(color)

    def _vispy_handle_world_size(self) -> float:
        """World-space radius corresponding to an eight-pixel hit target."""

        try:
            x_range, y_range = self.view.viewRange()
            viewport = self.graphicsView.viewport()
            x_per_pixel = abs(float(x_range[1]) - float(x_range[0])) / max(1, int(viewport.width()))
            y_per_pixel = abs(float(y_range[1]) - float(y_range[0])) / max(1, int(viewport.height()))
            return max(x_per_pixel, y_per_pixel) * 8.0
        except Exception:
            return 2.0

    def setInspectionTool(self, tool):
        result = super().setInspectionTool(tool)
        self._clear_vispy_hover_feedback()
        return result

    def eventFilter(self, obj, event):
        if obj is self.graphicsView.viewport():
            if event.type() == QtCore.QEvent.Type.MouseMove:
                self._update_vispy_roi_cursor(event)
            elif event.type() == QtCore.QEvent.Type.Leave:
                self._clear_vispy_hover_feedback()
        return super().eventFilter(obj, event)

    def _clear_vispy_hover_feedback(self) -> None:
        self.interaction_controller.clear_hover()
        self._set_vispy_profile_hover_part(None)
        self._set_vispy_hovered_roi(None)
        viewport = self.graphicsView.viewport()
        if self._vispy_roi_cursor_active:
            viewport.unsetCursor()
            self._vispy_roi_cursor_active = False

    def _update_vispy_roi_cursor(self, event) -> None:
        if self._pending_roi_draw_tool is not None or self._drawing_active:
            return
        if self._inspection_tool in {"roi_line", "roi_rectangle", "roi_polyline", "roi_freehand"}:
            return
        scene_pos = self.graphicsView.mapToScene(event.pos())
        view_pos = self.view.mapSceneToView(scene_pos)
        point = (float(view_pos.x()), float(view_pos.y()))
        profile_position = self.profileMarkerPosition()
        profile_bounds = self._current_profile_bounds() if profile_position is not None else None
        target = hit_test_display_overlays(
            point,
            roi_selections=self.roiSelections(),
            profile_position=profile_position,
            profile_bounds=profile_bounds,
            tolerance=self._vispy_handle_world_size(),
        )
        interaction = self.interaction_controller.set_hover(target, point=point)
        profile_part = target.part if target is not None and target.kind == "profile" else None
        roi_id = target.object_id if target is not None and target.kind == "roi" else None
        self._set_vispy_profile_hover_part(profile_part)
        self._set_vispy_hovered_roi(roi_id)
        cursor = self._cursor_for_interaction_intent(interaction.cursor_intent)
        viewport = self.graphicsView.viewport()
        if cursor is None:
            if self._vispy_roi_cursor_active:
                viewport.unsetCursor()
                self._vispy_roi_cursor_active = False
            return
        viewport.setCursor(cursor)
        self._vispy_roi_cursor_active = True

    @staticmethod
    def _cursor_for_interaction_intent(intent: CursorIntent):
        shapes = {
            CursorIntent.CROSSHAIR: QtCore.Qt.CursorShape.CrossCursor,
            CursorIntent.MOVE: QtCore.Qt.CursorShape.SizeAllCursor,
            CursorIntent.OPEN_HAND: QtCore.Qt.CursorShape.OpenHandCursor,
            CursorIntent.CLOSED_HAND: QtCore.Qt.CursorShape.ClosedHandCursor,
            CursorIntent.RESIZE_HORIZONTAL: QtCore.Qt.CursorShape.SizeHorCursor,
            CursorIntent.RESIZE_VERTICAL: QtCore.Qt.CursorShape.SizeVerCursor,
            CursorIntent.RESIZE_DIAGONAL: QtCore.Qt.CursorShape.SizeFDiagCursor,
        }
        shape = shapes.get(CursorIntent(intent))
        return None if shape is None else QtGui.QCursor(shape)

    def _vispy_roi_cursor_for_point(self, x: float, y: float):
        result = self._vispy_roi_hit_for_point(float(x), float(y))
        if result is None:
            return None
        _roi_id, hit, geometry = result
        return self._cursor_for_vispy_roi_hit(hit, geometry)

    def _vispy_roi_hit_for_point(self, x: float, y: float):
        tolerance = self._vispy_handle_world_size()
        for roi_id, (_item, selection) in reversed(tuple(self._roi_items.items())):
            hit = hit_test_roi(selection.geometry, (float(x), float(y)), tolerance=tolerance)
            if hit is not None:
                return str(roi_id), hit, selection.geometry
        return None

    def _cursor_for_vispy_roi_hit(self, hit, geometry):
        kind = str(getattr(getattr(geometry, "kind", ""), "value", getattr(geometry, "kind", "")))
        if hit.part == "handle" and kind == "rectangle":
            return QtGui.QCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
        return QtGui.QCursor(QtCore.Qt.CursorShape.SizeAllCursor)

    def _set_vispy_hovered_roi(self, roi_id: str | None) -> None:
        roi_id = None if roi_id is None else str(roi_id)
        previous = self._vispy_hovered_roi_id
        if previous == roi_id:
            return
        self._vispy_hovered_roi_id = roi_id
        for current in (previous, roi_id):
            item_selection = self._roi_items.get(str(current)) if current is not None else None
            if item_selection is None:
                continue
            _item, selection = item_selection
            self._upsert_vispy_roi(selection.id, selection.geometry, selection.color)

    def _vispy_profile_cursor_for_point(self, x: float, y: float):
        return self._cursor_for_vispy_profile_hit(self._vispy_profile_hit_for_point(float(x), float(y)))

    def _vispy_profile_hit_for_point(self, x: float, y: float) -> str | None:
        if not bool(getattr(self, "_profile_marker_requested_visible", False)):
            return None
        position = self.profileMarkerPosition()
        if position is None:
            return None
        px, py = (float(position[0]), float(position[1]))
        tolerance = self._vispy_handle_world_size()
        if abs(float(x) - px) <= tolerance and abs(float(y) - py) <= tolerance:
            return "center"
        x0, y0, x1, y1 = self._current_profile_bounds()
        if min(y0, y1) <= float(y) <= max(y0, y1) and abs(float(x) - px) <= tolerance:
            return "vertical"
        if min(x0, x1) <= float(x) <= max(x0, x1) and abs(float(y) - py) <= tolerance:
            return "horizontal"
        return None

    def _cursor_for_vispy_profile_hit(self, part: str | None):
        if part == "center":
            return QtGui.QCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        if part == "vertical":
            return QtGui.QCursor(QtCore.Qt.CursorShape.SizeHorCursor)
        if part == "horizontal":
            return QtGui.QCursor(QtCore.Qt.CursorShape.SizeVerCursor)
        return None

    def _set_vispy_profile_hover_part(self, part: str | None) -> None:
        part = None if part is None else str(part)
        if part == self._vispy_profile_hover_part:
            return
        self._vispy_profile_hover_part = part
        self._sync_vispy_profile_marker()

    def setProfileMarker(self, x, y, visible=True):
        super().setProfileMarker(x, y, visible=visible)
        self._sync_vispy_profile_marker()

    def hideProfileMarker(self):
        self._vispy_profile_hover_part = None
        super().hideProfileMarker()
        for visual in getattr(self, "_vispy_profile_visuals", {}).values():
            _set_visual_visible(visual, False)
        self._vispy_canvas.update()

    def _sync_profile_marker_visibility(self):
        super()._sync_profile_marker_visibility()
        self._sync_vispy_profile_marker()

    def _on_profile_marker_changed(self, *_args):
        super()._on_profile_marker_changed(*_args)
        self._sync_vispy_profile_marker()

    def _on_profile_handle_changed(self, *_args):
        super()._on_profile_handle_changed(*_args)
        self._sync_vispy_profile_marker()

    def _sync_vispy_profile_marker(self) -> None:
        if self.image is None or not bool(getattr(self, "_profile_marker_requested_visible", False)):
            for visual in getattr(self, "_vispy_profile_visuals", {}).values():
                _set_visual_visible(visual, False)
            return
        position = self.profileMarkerPosition()
        if position is None:
            return
        x, y = (float(position[0]), float(position[1]))
        if not _point_inside_view_range(self.view.viewRange(), x, y):
            for visual in getattr(self, "_vispy_profile_visuals", {}).values():
                _set_visual_visible(visual, False)
            return
        x0, y0, x1, y1 = self._current_profile_bounds()
        hovered = self._vispy_profile_hover_part is not None
        line_color = (255, 125, 55) if hovered else (230, 60, 30)
        line_width = 2.5 if hovered else 1.5
        self._upsert_vispy_line("profile_v", np.asarray([[x, y0], [x, y1]], dtype=np.float32), line_color, width=line_width)
        self._upsert_vispy_line("profile_h", np.asarray([[x0, y], [x1, y]], dtype=np.float32), line_color, width=line_width)
        marker = max(0.8, min(float(x1 - x0), float(y1 - y0)) * 0.025)
        self._upsert_vispy_line("profile_handle_x", np.asarray([[x - marker, y], [x + marker, y]], dtype=np.float32), line_color, width=3.0 if hovered else 2.0)
        self._upsert_vispy_line("profile_handle_y", np.asarray([[x, y - marker], [x, y + marker]], dtype=np.float32), line_color, width=3.0 if hovered else 2.0)
        self._upsert_vispy_profile_dot(x, y, hovered=hovered)
        self._vispy_canvas.update()

    def _upsert_vispy_line(self, key: str, points, color, *, width: float, order: int = 10_000):
        visual = self._vispy_profile_visuals.get(str(key))
        if visual is None:
            visual = self._vispy_visuals.Line(
                points,
                parent=self._vispy_view.scene,
                color=_vispy_color(color),
                width=float(width),
                method="agg",
            )
            self._vispy_profile_visuals[str(key)] = visual
        else:
            visual.set_data(pos=points, color=_vispy_color(color), width=float(width))
        visual.order = int(order)
        visual.visible = True
        return visual

    def _upsert_vispy_profile_dot(self, x: float, y: float, *, hovered: bool = False) -> None:
        visual = self._vispy_profile_visuals.get("profile_handle_dot")
        if visual is None:
            visual = self._vispy_visuals.Markers(parent=self._vispy_view.scene)
            self._vispy_profile_visuals["profile_handle_dot"] = visual
        visual.set_data(
            np.asarray([[float(x), float(y)]], dtype=np.float32),
            symbol="disc",
            size=12.0 if hovered else 9.0,
            face_color=_vispy_color((255, 125, 55) if hovered else (230, 60, 30)),
            edge_color=_vispy_color((255, 255, 255)),
            edge_width=2.0 if hovered else 1.0,
        )
        visual.order = 10_002
        visual.visible = True

    def _set_roi_drawing_preview(self, tool, points) -> None:
        if tool is not None:
            self._clear_vispy_hover_feedback()
        points = np.asarray(tuple(points or ()), dtype=np.float32).reshape((-1, 2))
        visual = getattr(self, "_vispy_roi_drawing_preview", None)
        if tool is None or len(points) < 2:
            _set_visual_visible(visual, False)
            if visual is not None:
                self._vispy_canvas.update()
            return
        if visual is None:
            visual = self._vispy_visuals.Line(
                points,
                parent=self._vispy_view.scene,
                color=_vispy_color((255, 190, 60)),
                width=2.5,
                method="agg",
            )
            visual.order = 10_003
            try:
                visual.set_gl_state("translucent", depth_test=False)
            except Exception:
                pass
            self._vispy_roi_drawing_preview = visual
        else:
            visual.set_data(pos=points, color=_vispy_color((255, 190, 60)), width=2.5)
        visual.visible = True
        self._vispy_canvas.update()

    def setMontageTileOverlays(self, overlays):
        overlays = tuple(overlays or ())
        # Do not mirror these through the PyQtGraph overlay item in the VisPy
        # backend.  The scene already has a transparent PyQtGraph layer for
        # interaction, and painting hundreds of duplicate QGraphics overlays is
        # exactly the kind of UI fan-in that makes large montage commits hang.
        super().clearMontageTileOverlays()
        self._montage_tile_overlay_items = []
        self._set_vispy_montage_tile_overlays(overlays)

    def clearMontageTileOverlays(self):
        super().clearMontageTileOverlays()
        self._vispy_overlay_key = ()
        self._vispy_overlay_count = 0
        for visual in getattr(self, "_vispy_overlay_visuals", ()):
            _set_visual_visible(visual, False)
        self._vispy_canvas.update()

    def montageTileOverlayCount(self) -> int:
        return int(getattr(self, "_vispy_overlay_count", 0) or 0)

    def _set_vispy_montage_tile_overlays(self, overlays) -> None:
        overlays = tuple(overlays or ())
        key = _overlay_batch_key(overlays)
        self._vispy_overlay_count = len(overlays)
        if key == getattr(self, "_vispy_overlay_key", ()):
            for visual in getattr(self, "_vispy_overlay_visuals", ()):
                _set_visual_visible(visual, bool(overlays))
            return
        self._vispy_overlay_key = key
        if not overlays:
            for visual in getattr(self, "_vispy_overlay_visuals", ()):
                _set_visual_visible(visual, False)
            self._vispy_canvas.update()
            return

        mesh = self._ensure_vispy_overlay_mesh()
        lines = self._ensure_vispy_overlay_lines()
        vertices, faces, colors = _overlay_mesh_arrays(overlays)
        line_points, line_colors = _overlay_line_arrays(overlays)
        mesh.set_data(vertices=vertices, faces=faces, vertex_colors=colors)
        lines.set_data(pos=line_points, color=line_colors, width=1.25, connect="segments")
        mesh.visible = True
        lines.visible = bool(len(line_points))
        self._vispy_canvas.update()

    def _ensure_vispy_overlay_mesh(self):
        mesh = getattr(self, "_vispy_overlay_mesh", None)
        if mesh is None:
            mesh = self._vispy_visuals.Mesh(parent=self._vispy_view.scene)
            mesh.order = 11_000
            try:
                mesh.set_gl_state("translucent", depth_test=False)
            except Exception:
                pass
            mesh.visible = False
            self._vispy_overlay_mesh = mesh
            self._refresh_vispy_overlay_visual_list()
        return mesh

    def _ensure_vispy_overlay_lines(self):
        lines = getattr(self, "_vispy_overlay_lines", None)
        if lines is None:
            lines = self._vispy_visuals.Line(parent=self._vispy_view.scene, method="gl", connect="segments")
            lines.order = 11_001
            try:
                lines.set_gl_state("translucent", depth_test=False)
            except Exception:
                pass
            lines.visible = False
            self._vispy_overlay_lines = lines
            self._refresh_vispy_overlay_visual_list()
        return lines

    def _refresh_vispy_overlay_visual_list(self) -> None:
        self._vispy_overlay_visuals = [
            visual
            for visual in (getattr(self, "_vispy_overlay_mesh", None), getattr(self, "_vispy_overlay_lines", None))
            if visual is not None
        ]

    def _sync_vispy_bounds(self, image_shape, *, image_origin=(0.0, 0.0)) -> None:
        if getattr(self, "_vispy_bounds_item", None) is None:
            return
        height, width = tuple(int(value) for value in image_shape[:2])
        self._vispy_display_shape = (max(1, height), max(1, width))
        self._vispy_bounds_item.setRect(
            QtCore.QRectF(
                float(image_origin[0]),
                float(image_origin[1]),
                float(max(1, width)),
                float(max(1, height)),
            )
        )

    def _sync_vispy_montage_bounds(self, geometry) -> tuple[int, int]:
        montage = getattr(geometry, "montage", None)
        if montage is None:
            shape = tuple(getattr(geometry, "display_shape", (1, 1)))[:2]
            self._sync_vispy_bounds(shape)
            return tuple(int(value) for value in shape)
        width = int(montage.columns) * int(montage.tile_width) + max(0, int(montage.columns) - 1) * int(montage.gap)
        height = int(montage.rows) * int(montage.tile_height) + max(0, int(montage.rows) - 1) * int(montage.gap)
        self._sync_vispy_bounds((height, width), image_origin=(0.0, 0.0))
        return (height, width)

    def _current_image_world_rect(self):
        bounds = getattr(self, "_vispy_bounds_item", None)
        if bounds is None:
            return super()._current_image_world_rect()
        rect = bounds.rect()
        return (
            float(rect.left()),
            float(rect.top()),
            float(rect.left() + max(0.0, rect.width() - 1.0)),
            float(rect.top() + max(0.0, rect.height() - 1.0)),
        )

    def _current_image_viewport_rect(self):
        bounds = getattr(self, "_vispy_bounds_item", None)
        if bounds is None:
            return super()._current_image_viewport_rect()
        rect = bounds.rect()
        return (
            float(rect.left()),
            float(rect.top()),
            float(rect.left() + max(1.0, rect.width())),
            float(rect.top() + max(1.0, rect.height())),
        )

    def _updateAspectRatio(self):
        super()._updateAspectRatio()
        camera = getattr(getattr(self, "_vispy_view", None), "camera", None)
        if camera is not None:
            camera.aspect = 1.0 if getattr(self, "displayMode", "square_pixels") == "square_pixels" else None
            self._vispy_canvas.update()

    def setFitLocked(self, enabled):
        super().setFitLocked(enabled)
        if self.image is not None:
            self._sync_vispy_camera_to_view()

    def oneToOne(self):
        self.setDisplayMode("square_pixels")
        self.view.setMouseEnabled(x=True, y=True)
        if self.image is not None:
            self._viewport_applying = True
            try:
                self.viewport_controller.one_to_one(
                    self.view,
                    getattr(self, "_vispy_display_shape", self.image.shape[:2]),
                    self.graphicsView.viewport().size(),
                    display_rect=self._current_image_viewport_rect(),
                )
            finally:
                self._viewport_applying = False
            self._sync_vispy_camera_to_view()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image is not None and self.viewport_controller.is_fit_locked():
            self._sync_vispy_camera_to_view()

    def _update_histogram_for_vispy(self, histogramData, histogramPlotData, levels) -> None:
        previous_plot_source = self.histogramPlotSource
        self.histogramPlotSource = histogramPlotData
        try:
            plot_data = self._histogram_plot_data(histogramData)
        finally:
            self.histogramPlotSource = previous_plot_source
        if plot_data is None:
            return
        self._bind_histogram_item(self.histogramImageItem)
        self._set_image_item_data(self.histogramImageItem, plot_data, self._histogram_levels_for_display(levels), role="histogram")

    def _update_vispy_tile_layer(
        self,
        img,
        *,
        histogram_data,
        geometry,
        levels,
        rgb_already_windowed: bool,
        dirty_tiles,
        tile_source_ids,
        tile_payloads=None,
        shader_mapping=None,
        tile_delta=None,
        tile_residency_budget_bytes: int = 0,
        force_levels: bool = False,
        force_mapping: bool = False,
    ):
        from arrayscope.display.backends.pyqtgraph.tiles import TileLayerUpdateStats

        if geometry is None or getattr(geometry, "montage", None) is None:
            return TileLayerUpdateStats()
        if tile_payloads is not None:
            return self._update_vispy_direct_tile_layer(
                tile_payloads,
                geometry=geometry,
                levels=levels,
                rgb_already_windowed=rgb_already_windowed,
                dirty_tiles=dirty_tiles,
                tile_source_ids=tile_source_ids,
                shader_mapping=shader_mapping,
                tile_delta=tile_delta,
                tile_residency_budget_bytes=tile_residency_budget_bytes,
                force_levels=force_levels,
                force_mapping=force_mapping,
            )
        if img is None:
            return TileLayerUpdateStats()
        self._last_vispy_geometry = geometry
        montage = geometry.montage
        dirty = None if dirty_tiles is None else {int(tile) for tile in dirty_tiles}
        level_tuple = (float(levels[0]), float(levels[1]))
        visible_items = 0
        updated = 0
        skipped = 0
        rgb_tiles = 0
        states = tuple(getattr(geometry, "montage_tile_states", ()) or ())
        canvas = np.asarray(img)
        hist = None if histogram_data is None else np.asarray(histogram_data)
        for tile_number, source_index in enumerate(montage.indices):
            row = tile_number // montage.columns
            col = tile_number % montage.columns
            x0 = int(col * (montage.tile_width + montage.gap))
            y0 = int(row * (montage.tile_height + montage.gap))
            cx0 = x0 - int(geometry.montage_origin_x)
            cy0 = y0 - int(geometry.montage_origin_y)
            cx1 = cx0 + montage.tile_width
            cy1 = cy0 + montage.tile_height
            if cx1 <= 0 or cy1 <= 0 or cx0 >= canvas.shape[1] or cy0 >= canvas.shape[0]:
                self._hide_vispy_tile(tile_number)
                continue
            kind = "loaded"
            if states and tile_number < len(states):
                kind = str(getattr(states[tile_number], "value", states[tile_number]))
            if kind != "loaded":
                self._hide_vispy_tile(tile_number)
                continue
            visible_items += 1
            source_id = None if tile_source_ids is None else tile_source_ids.get(tile_number)
            sx0 = max(0, cx0)
            sy0 = max(0, cy0)
            sx1 = min(canvas.shape[1], cx1)
            sy1 = min(canvas.shape[0], cy1)
            if sx1 <= sx0 or sy1 <= sy0:
                self._hide_vispy_tile(tile_number)
                continue
            tile_img = canvas[sy0:sy1, sx0:sx1, ...]
            tile_hist = None if hist is None else hist[sy0:sy1, sx0:sx1]
            use_windowed_rgb = self._should_use_windowed_rgb(
                tile_img,
                tile_hist,
                rgb_already_windowed=rgb_already_windowed,
            )
            state = self._ensure_vispy_tile(tile_number, windowed_rgb=use_windowed_rgb)
            color_source_id = id(tile_img) if use_windowed_rgb else None
            scalar_source_id = id(tile_hist) if use_windowed_rgb and tile_hist is not None else None
            levels_changed = state.levels != level_tuple
            if force_levels and use_windowed_rgb and state.visible:
                state.visual.set_levels(level_tuple)
                state.levels = level_tuple
                skipped += 1
                continue
            scalar_level_only = not use_windowed_rgb and not self._is_rgb_image(tile_img)
            needs_data = (
                dirty is None
                or tile_number in dirty
                or state.source_id != source_id
                or state.windowed_rgb != bool(use_windowed_rgb)
                or not state.visible
                or (not use_windowed_rgb and levels_changed and not scalar_level_only)
            )
            if not needs_data:
                if use_windowed_rgb and levels_changed:
                    state.visual.set_levels(level_tuple)
                    state.levels = level_tuple
                elif scalar_level_only and levels_changed:
                    try:
                        state.visual.clim = level_tuple
                    except Exception:
                        pass
                    state.levels = level_tuple
                skipped += 1
                continue
            start = perf_counter()
            if use_windowed_rgb:
                tile_data = _contiguous_display(np.asarray(tile_img)[..., :3])
                tile_scalar = _contiguous_scalar(tile_hist)
                state.visual.set_data(
                    tile_data,
                    tile_scalar,
                    levels=level_tuple,
                    color_source_id=color_source_id,
                    scalar_source_id=scalar_source_id,
                    copy=False,
                )
                state.color_source_id = color_source_id
                state.scalar_source_id = scalar_source_id
            else:
                tile_data = self._vispy_display_data(tile_img, tile_hist, level_tuple, rgb_already_windowed=rgb_already_windowed, timing_field="tile_layer_rgb_window_ms")
                if self._is_rgb_image(tile_img) and not rgb_already_windowed:
                    rgb_tiles += 1
                state.visual.set_data(tile_data, copy=False)
                if tile_data.ndim == 2:
                    state.visual.clim = level_tuple
                    self._apply_vispy_colormap_to_visual(state.visual)
            state.visual.transform = self._vispy_transforms.STTransform(translate=(float(x0 + max(0, -cx0)), float(y0 + max(0, -cy0)), 0.0))
            state.visual.visible = True
            state.source_id = source_id
            state.levels = level_tuple
            state.data_shape = tuple(np.shape(tile_data))
            state.visible = True
            state.windowed_rgb = bool(use_windowed_rgb)
            updated += 1
            self._record_upload_timing("tile_layer_upload_ms", (perf_counter() - start) * 1000.0)
        active = set(range(len(montage.indices)))
        for tile_number in tuple(self._vispy_tile_visuals):
            if tile_number not in active:
                self._hide_vispy_tile(tile_number)
        return TileLayerUpdateStats(visible_items=visible_items, items_updated=updated, items_skipped=skipped, rgb_window_tiles=rgb_tiles)

    def _update_vispy_direct_tile_layer(
        self,
        tile_payloads,
        *,
        geometry,
        levels,
        rgb_already_windowed: bool,
        dirty_tiles,
        tile_source_ids,
        shader_mapping=None,
        tile_delta=None,
        tile_residency_budget_bytes: int = 0,
        force_levels: bool = False,
        force_mapping: bool = False,
    ):
        from arrayscope.display.backends.pyqtgraph.tiles import TileLayerUpdateStats

        montage = geometry.montage
        if montage is None:
            return TileLayerUpdateStats()
        self._last_vispy_geometry = geometry
        for state in getattr(self, "_vispy_tile_visuals", {}).values():
            _set_visual_visible(state.image_visual, False)
            _set_visual_visible(state.windowed_visual, False)
            state.visible = False
        states = tuple(getattr(geometry, "montage_tile_states", ()) or ())
        if tile_delta is not None:
            active_set = {int(tile) for tile in tuple(getattr(tile_delta, "active_tiles", ()) or ())}
        else:
            active_set = set(dict(tile_payloads or {}))
        loaded_payloads = {}
        for tile_number, _source_index in enumerate(tuple(montage.indices)):
            if int(tile_number) not in active_set:
                continue
            kind = "loaded"
            if states and tile_number < len(states):
                kind = str(getattr(states[tile_number], "value", states[tile_number]))
            if kind == "loaded" and int(tile_number) in tile_payloads:
                loaded_payloads[int(tile_number)] = tile_payloads[int(tile_number)]
        layer = getattr(self, "_vispy_gpu_montage_layer", None)
        if layer is None:
            return TileLayerUpdateStats()
        if (force_levels or force_mapping) and getattr(layer, "last_stats", None).visible_items:
            stats = layer.set_presentation_uniforms(
                levels=levels,
                shader_mapping=shader_mapping,
            )
        else:
            try:
                stats = layer.update(
                    payloads=loaded_payloads,
                    geometry=geometry,
                    levels=levels,
                    dirty_tiles=dirty_tiles,
                    rgb_already_windowed=rgb_already_windowed,
                    shader_mapping=shader_mapping,
                    tile_delta=tile_delta,
                    tile_residency_budget_bytes=tile_residency_budget_bytes,
                )
            except Exception as exc:
                from arrayscope.display.backends.vispy.tiles import AtlasCapacityError, GpuMontageLayerStats

                if not isinstance(exc, AtlasCapacityError):
                    raise
                previous = getattr(layer, "last_stats", None)
                stats = GpuMontageLayerStats(
                    visible_items=len(loaded_payloads),
                    resident_items=int(getattr(previous, "resident_items", 0) or 0),
                    atlas_capacity=int(getattr(previous, "atlas_capacity", 0) or 0),
                    estimated_gpu_bytes=int(getattr(previous, "estimated_gpu_bytes", 0) or 0),
                    cpu_shadow_bytes=int(getattr(previous, "cpu_shadow_bytes", 0) or 0),
                    page_count=int(getattr(previous, "page_count", 0) or 0),
                    active_pages=int(getattr(previous, "active_pages", 0) or 0),
                    device_max_texture_size=int(getattr(previous, "device_max_texture_size", 0) or 0),
                    budget_bytes=int(tile_residency_budget_bytes),
                    capacity_warning=str(exc),
                )
        self._record_upload_timing("tile_layer_upload_ms", float(stats.upload_ms))
        timing = self._upload_timing
        if timing is not None:
            timing["visible_bytes"] = int(timing["visible_bytes"]) + int(stats.texture_upload_bytes)
        for tile_number in tuple(self._vispy_tile_visuals):
            self._hide_vispy_tile(tile_number)
        return TileLayerUpdateStats(
            visible_items=int(stats.visible_items),
            items_updated=int(stats.items_updated),
            items_skipped=int(stats.items_skipped),
            rgb_window_tiles=0,
            resident_items=int(stats.resident_items),
            storage_capacity=int(stats.atlas_capacity),
            storage_rebuilds=int(stats.atlas_rebuilds),
            storage_evictions=int(stats.atlas_evictions),
            texture_uploads=int(stats.texture_uploads),
            texture_upload_bytes=int(stats.texture_upload_bytes),
            vertex_uploads=int(stats.vertex_uploads),
            level_updates=int(stats.level_updates),
            estimated_gpu_bytes=int(stats.estimated_gpu_bytes),
            cpu_shadow_bytes=int(stats.cpu_shadow_bytes),
            page_count=int(getattr(stats, "page_count", 0)),
            active_pages=int(getattr(stats, "active_pages", 0)),
            device_max_texture_size=int(getattr(stats, "device_max_texture_size", 0)),
            budget_bytes=int(getattr(stats, "budget_bytes", 0)),
            near_resident_items=int(getattr(stats, "near_resident_items", 0)),
            warm_resident_items=int(getattr(stats, "warm_resident_items", 0)),
            evicted_near_items=int(getattr(stats, "evicted_near_items", 0)),
            capacity_warning=str(getattr(stats, "capacity_warning", "")),
            lod_level=int(getattr(stats, "lod_level", 0)),
            lod_factor=int(getattr(stats, "lod_factor", 1)),
            source_texels_per_pixel=float(getattr(stats, "source_texels_per_pixel", 0.0)),
            gutter_pixels=int(getattr(stats, "gutter_pixels", 0)),
            mipmap_updates=int(getattr(stats, "mipmap_updates", 0)),
            mipmap_available=bool(getattr(stats, "mipmap_available", False)),
            complex_texture_uploads=int(getattr(stats, "complex_texture_uploads", 0)),
            shader_uniform_updates=int(getattr(stats, "shader_uniform_updates", 0)),
        )

    def _ensure_vispy_tile(self, tile_number: int, *, windowed_rgb: bool = False) -> _VisPyTileState:
        tile_number = int(tile_number)
        state = self._vispy_tile_visuals.get(tile_number)
        if state is None:
            state = _VisPyTileState()
            self._vispy_tile_visuals[tile_number] = state
        if windowed_rgb:
            if state.windowed_visual is None:
                state.windowed_visual = self._vispy_scene.visuals.create_visual_node(GpuMappedImageVisual)(
                    parent=self._vispy_view.scene
                )
                state.windowed_visual.visible = False
            _set_visual_visible(state.image_visual, False)
            state.visual = state.windowed_visual
        else:
            if state.image_visual is None:
                state.image_visual = self._vispy_visuals.Image(
                    np.zeros((1, 1), dtype=np.float32),
                    parent=self._vispy_view.scene,
                    interpolation="nearest",
                    texture_format="auto",
                    method="auto",
                    clim=self._vispy_last_levels,
                )
                state.image_visual.visible = False
            _set_visual_visible(state.windowed_visual, False)
            state.visual = state.image_visual
        return state

    def _hide_vispy_tile(self, tile_number: int) -> None:
        state = self._vispy_tile_visuals.get(int(tile_number))
        if state is not None:
            _set_visual_visible(state.image_visual, False)
            _set_visual_visible(state.windowed_visual, False)
            state.visible = False

    def _sync_vispy_camera_to_view(self) -> None:
        try:
            x_range, y_range = self.view.viewRange()
            state = getattr(self.view, "state", {}) or {}
            key = (
                (float(x_range[0]), float(x_range[1])),
                (float(y_range[0]), float(y_range[1])),
                bool(state.get("xInverted", False)),
                bool(state.get("yInverted", True)),
            )
            if key == getattr(self, "_vispy_camera_key", None):
                return
            self._vispy_camera_key = key
            self._vispy_view.camera.flip = (key[2], key[3], False)
            self._vispy_view.camera.set_range(x=(float(x_range[0]), float(x_range[1])), y=(float(y_range[0]), float(y_range[1])), margin=0)
            self._vispy_canvas.update()
        except Exception:
            pass

    def _request_vispy_camera_sync(self) -> None:
        # A camera gesture has priority over speculative residency uploads.
        # The next settled tiled presentation will enqueue the relevant near
        # ring again, so discarding stale warm work is both safe and cheaper.
        warm_timer = getattr(self, "_vispy_warm_tile_timer", None)
        if warm_timer is not None and warm_timer.isActive():
            warm_timer.stop()
        self._vispy_pending_warm_tile_payloads = {}
        self._vispy_pending_warm_tile_context = {}
        if getattr(self, "_vispy_camera_sync_pending", False):
            return
        self._vispy_camera_sync_pending = True

        def apply_sync():
            self._vispy_camera_sync_pending = False
            self._sync_vispy_camera_to_view()

        QtCore.QTimer.singleShot(0, apply_sync)

    def _on_vispy_mouse_move(self, event) -> None:
        # The PyQtGraph overlay owns interaction.  This bridge is only useful if
        # the VisPy canvas receives motion events before the transparent overlay.
        try:
            event_pos = _vispy_event_pos(event)
            if event_pos is None:
                return
            mapped = self._map_vispy_canvas_pos_to_world(event_pos)
            self.view.scene().sigMouseMoved.emit(QtCore.QPointF(float(mapped[0]), float(mapped[1])))
        except Exception:
            pass

    def _map_vispy_canvas_pos_to_world(self, pos):
        tr = self._vispy_view.scene.node_transform(self._vispy_canvas.scene)
        return tr.map(pos)[:2]


def _import_vispy():
    try:
        from vispy import scene
        from vispy import gloo
        from vispy.scene import visuals
        from vispy.scene.cameras import PanZoomCamera
        from vispy.visuals import transforms
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("VisPy rendering backend is not available. Install ArrayScope[vispy] or vispy.") from exc
    return scene, visuals, transforms, PanZoomCamera, gloo


def _tiled_source_key(tile_payloads, tile_source_ids):
    if tile_payloads is None:
        return None
    ids = tile_source_ids or {}
    return tuple(
        (
            int(tile),
            ids.get(int(tile), getattr(payload, "source_id", None)),
        )
        for tile, payload in sorted(dict(tile_payloads).items())
    )


def _shader_mapping_key(mapping):
    return None if mapping is None else getattr(mapping, "identity_key", mapping)


def _tiled_structure_key(geometry, *, rgb_already_windowed):
    montage = getattr(geometry, "montage", None)
    if montage is None:
        montage_key = None
    else:
        montage_key = (
            tuple(int(index) for index in tuple(montage.indices)),
            int(montage.tile_height),
            int(montage.tile_width),
            int(montage.columns),
            int(montage.rows),
            int(montage.gap),
            int(getattr(geometry, "montage_origin_x", 0)),
            int(getattr(geometry, "montage_origin_y", 0)),
        )
    return (
        tuple(int(value) for value in tuple(getattr(geometry, "display_shape", ()))[:2]),
        montage_key,
        bool(rgb_already_windowed),
    )


def _tiled_histogram_key(histogram_data, histogram_plot_data, histogram_range):
    return (
        _array_identity_key(histogram_data),
        _array_identity_key(histogram_plot_data),
        (float(histogram_range[0]), float(histogram_range[1])),
    )


def _array_identity_key(data):
    if data is None:
        return None
    array = np.asarray(data)
    return (tuple(int(value) for value in array.shape), str(array.dtype), array.tobytes())


def _set_visual_visible(visual, visible: bool) -> None:
    if visual is None:
        return
    try:
        visual.visible = bool(visible)
    except Exception:
        pass


def _vispy_event_pos(event):
    pos = getattr(event, "pos", None)
    if pos is None:
        return None
    try:
        return (float(pos[0]), float(pos[1]))
    except Exception:
        return None


def _vispy_color(color):
    rgb = tuple(int(value) for value in tuple(color or (255, 255, 0))[:3])
    return tuple(float(value) / 255.0 for value in rgb) + (1.0,)


def _vispy_roi_points(geometry):
    kind = str(getattr(getattr(geometry, "kind", ""), "value", getattr(geometry, "kind", "")))
    if kind == "rectangle":
        rect = getattr(geometry, "rect", None)
        if rect is None:
            return None
        x, y, width, height = (float(value) for value in rect)
        return np.asarray(
            [
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height],
                [x, y],
            ],
            dtype=np.float32,
        )
    points = tuple(getattr(geometry, "points", ()) or ())
    if kind == "line" and len(points) >= 2:
        return np.asarray(points[:2], dtype=np.float32)
    if kind in {"polyline", "freehand_polygon"} and len(points) >= 2:
        return np.asarray(points, dtype=np.float32)
    return None


def _overlay_batch_key(overlays):
    return tuple(
        (
            int(getattr(overlay, "x", 0)),
            int(getattr(overlay, "y", 0)),
            int(getattr(overlay, "width", 1)),
            int(getattr(overlay, "height", 1)),
            str(getattr(overlay, "state", "")),
        )
        for overlay in tuple(overlays or ())
    )


def _overlay_mesh_arrays(overlays):
    vertices = []
    faces = []
    colors = []
    for overlay in tuple(overlays or ()):
        x = float(getattr(overlay, "x", 0.0))
        y = float(getattr(overlay, "y", 0.0))
        width = float(max(1.0, getattr(overlay, "width", 1.0)))
        height = float(max(1.0, getattr(overlay, "height", 1.0)))
        fill, _border, _mark = _overlay_vispy_colors(overlay)
        base = len(vertices)
        vertices.extend(
            (
                (x, y, 0.0),
                (x + width, y, 0.0),
                (x + width, y + height, 0.0),
                (x, y + height, 0.0),
            )
        )
        faces.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
        colors.extend((fill, fill, fill, fill))
    return (
        np.asarray(vertices, dtype=np.float32).reshape((-1, 3)),
        np.asarray(faces, dtype=np.uint32).reshape((-1, 3)),
        np.asarray(colors, dtype=np.float32).reshape((-1, 4)),
    )


def _overlay_line_arrays(overlays):
    points = []
    colors = []
    for overlay in tuple(overlays or ()):
        x = float(getattr(overlay, "x", 0.0))
        y = float(getattr(overlay, "y", 0.0))
        width = float(max(1.0, getattr(overlay, "width", 1.0)))
        height = float(max(1.0, getattr(overlay, "height", 1.0)))
        _fill, border, mark = _overlay_vispy_colors(overlay)
        border_segments = (
            ((x, y), (x + width, y)),
            ((x + width, y), (x + width, y + height)),
            ((x + width, y + height), (x, y + height)),
            ((x, y + height), (x, y)),
        )
        for a, b in border_segments:
            points.extend((a, b))
            colors.extend((border, border))
        mark_points = np.asarray(_overlay_status_mark_points(overlay), dtype=np.float32).reshape((-1, 2))
        for point in mark_points:
            points.append((float(point[0]), float(point[1])))
            colors.append(mark)
    return (
        np.asarray(points, dtype=np.float32).reshape((-1, 2)),
        np.asarray(colors, dtype=np.float32).reshape((-1, 4)),
    )


def _overlay_vispy_colors(overlay):
    if str(getattr(overlay, "state", "")) == "skipped":
        return (
            _rgba255(130, 70, 20, 95),
            _rgba255(210, 130, 60, 180),
            _rgba255(245, 245, 245, 230),
        )
    return (
        _rgba255(35, 35, 35, 95),
        _rgba255(170, 170, 170, 140),
        _rgba255(245, 245, 245, 230),
    )


def _rgba255(r, g, b, a):
    return (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, float(a) / 255.0)


def _overlay_status_mark_points(overlay):
    x = float(getattr(overlay, "x", 0.0))
    y = float(getattr(overlay, "y", 0.0))
    width = float(max(1.0, getattr(overlay, "width", 1.0)))
    height = float(max(1.0, getattr(overlay, "height", 1.0)))
    tile_extent = min(width, height)
    size = max(tile_extent * 0.08, min(tile_extent * 0.18, 0.35))
    cx = x + width * 0.5
    cy = y + height * 0.5
    if str(getattr(overlay, "state", "")) == "skipped":
        points = (
            (cx - size * 0.5, cy - size * 0.5),
            (cx + size * 0.5, cy + size * 0.5),
            (cx - size * 0.5, cy + size * 0.5),
            (cx + size * 0.5, cy - size * 0.5),
        )
    else:
        points = (
            (cx, cy),
            (cx, cy - size * 0.32),
            (cx, cy),
            (cx + size * 0.28, cy),
        )
    return np.asarray(points, dtype=np.float32)
