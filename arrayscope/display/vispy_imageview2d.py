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

import numpy as np

try:  # Keep this module importable enough for factory fallback diagnostics.
    from vispy.visuals import Visual
except Exception:  # pragma: no cover - optional dependency path
    Visual = object

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from arrayscope.core.runtime_diagnostics import ImageUploadTiming
from arrayscope.display.imageview2d import ImageView2D
from arrayscope.display.imageview2d import _point_inside_view_range
from arrayscope.display.image_upload import rgb_display_for_levels
from arrayscope.display.viewport import ViewportPolicy


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

    def setupUI(self):
        self._vispy_scene, self._vispy_visuals, self._vispy_transforms, self._vispy_panzoom_camera = _import_vispy()
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
        self._vispy_windowed_image = self._vispy_scene.visuals.create_visual_node(_WindowedRgbVisual)(
            parent=self._vispy_view.scene
        )
        self._vispy_windowed_image.visible = False
        self._vispy_windowed_image.transform = self._vispy_transforms.STTransform(translate=(0.0, 0.0, 0.0))
        self._vispy_tile_visuals: dict[int, _VisPyTileState] = {}
        self._vispy_roi_visuals: dict[str, object] = {}
        self._vispy_roi_handle_visuals: dict[str, object] = {}
        self._vispy_overlay_visuals: list[object] = []
        self._vispy_profile_visuals: dict[str, object] = {}
        self._vispy_last_levels: tuple[float, float] = (0.0, 1.0)
        self._vispy_main_data_id: int | None = None
        self._vispy_main_color_source_id: int | None = None
        self._vispy_main_scalar_source_id: int | None = None
        self._vispy_display_shape: tuple[int, int] = (1, 1)
        self._vispy_roi_cursor_active = False
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
        self._vispy_bounds_item.setZValue(-1000)
        self.view.addItem(self._vispy_bounds_item)
        self.view.sigRangeChanged.connect(lambda *_args: self._sync_vispy_camera_to_view())
        state_signal = getattr(self.view, "sigStateChanged", None)
        if state_signal is not None:
            state_signal.connect(lambda *_args: self._sync_vispy_camera_to_view())

    def clearMontageTileLayer(self) -> None:
        for state in getattr(self, "_vispy_tile_visuals", {}).values():
            _set_visual_visible(state.image_visual, False)
            _set_visual_visible(state.windowed_visual, False)
            state.visual = None
            state.visible = False
        self.clearMontageTileOverlays()
        self._montage_display_mode = "canvas"
        self.imageItem.setVisible(False)
        try:
            self._vispy_image.visible = True
        except Exception:
            pass
        _set_visual_visible(getattr(self, "_vispy_windowed_image", None), False)

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
    ):
        if not isinstance(img, np.ndarray):
            raise TypeError("Image must be a numpy array")
        viewport_policy = _coerce_viewport_policy(viewport_policy, autoRange)
        self._start_upload_timing("vispy_full")
        applying = self._applying_presentation
        self._applying_presentation = True
        try:
            self.clearMontageTileLayer()
            self.image = img
            self.histogramSource = histogramData
            self.histogramPlotSource = histogramPlotData
            display_levels = _normalize_levels(levels, self._displayLevels or (0.0, 1.0))
            self._upload_vispy_main_image(img, histogramData=histogramData, levels=display_levels, image_origin=image_origin, rgb_already_windowed=rgb_already_windowed)
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
            self._upload_vispy_main_image(img, histogramData=histogramData, levels=display_levels, image_origin=image_origin, rgb_already_windowed=rgb_already_windowed)
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
    ) -> None:
        if geometry is None or getattr(geometry, "montage", None) is None:
            raise ValueError("tile-layer presentation requires montage geometry")
        self._start_upload_timing("vispy_tile_layer")
        applying = self._applying_presentation
        self._applying_presentation = True
        try:
            self.image = img
            self.histogramSource = histogramData
            self.histogramPlotSource = histogramPlotData
            self.setHistogramDataBounds(histogramRange)
            self._montage_display_mode = "vispy_tile_layer"
            try:
                self._vispy_image.visible = False
            except Exception:
                pass
            stats = self._update_vispy_tile_layer(
                img,
                histogram_data=histogramData,
                geometry=geometry,
                levels=(float(levels[0]), float(levels[1])),
                rgb_already_windowed=rgb_already_windowed,
                dirty_tiles=montage_dirty_tiles,
                tile_source_ids=montage_tile_source_ids,
            )
            self._record_tile_layer_stats(stats)
            self._update_histogram_for_vispy(histogramData, histogramPlotData, levels)
            self._sync_display_levels(float(levels[0]), float(levels[1]), update_image=False, emit_user=False)
            self.histogram.setHistogramRange(float(histogramRange[0]), float(histogramRange[1]))
            self._update_profile_line_bounds()
            self._updateAspectRatio()
            montage_shape = self._sync_vispy_montage_bounds(geometry)
            self._apply_viewport_policy(montage_shape, viewport_policy, image_origin=(0.0, 0.0))
            self._sync_vispy_camera_to_view()
        finally:
            self._applying_presentation = applying
            self._finish_upload_timing()

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

    def _upload_vispy_main_image(self, img, *, histogramData, levels, image_origin=(0.0, 0.0), rgb_already_windowed=False):
        start = perf_counter()
        if self._should_use_windowed_rgb(img, histogramData, rgb_already_windowed=rgb_already_windowed):
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
            data = self._vispy_display_data(img, histogramData, levels, rgb_already_windowed=rgb_already_windowed)
            previous = self._vispy_main_data_id
            same_object = previous == id(data)
            self._vispy_image.set_data(data, copy=False)
            self._vispy_main_data_id = id(data)
            if data.ndim == 2:
                self._vispy_image.clim = (float(levels[0]), float(levels[1]))
                try:
                    self._vispy_image.cmap = "grays"
                except Exception:
                    pass
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

    def _vispy_display_data(self, img, histogramData, levels, *, rgb_already_windowed=False):
        if self._is_rgb_image(img):
            if rgb_already_windowed:
                self.imageDisp = np.asarray(img[..., :3])
                return _contiguous_display(self.imageDisp)
            rgb_start = perf_counter()
            base = np.asarray(img[..., :3], dtype=np.float32)
            source = histogramData if histogramData is not None else self._histogram_data(base)
            self.imageDisp = rgb_display_for_levels(base, source, levels)
            self._record_upload_timing("rgb_window_ms", (perf_counter() - rgb_start) * 1000.0)
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

    def _is_windowed_rgb_vispy_main(self) -> bool:
        visual = getattr(self, "_vispy_windowed_image", None)
        return bool(visual is not None and getattr(visual, "visible", False))

    def createRoi(self, kind, *, points=None, rect=None, line_width=1.0, label=None, color=None):
        selection = super().createRoi(kind, points=points, rect=rect, line_width=line_width, label=label, color=color)
        item, _selection = self._roi_items[selection.id]
        item.sigRegionChanged.connect(lambda _item=item, roi_id=selection.id: self._on_roi_item_live_changed(roi_id))
        self._upsert_vispy_roi(selection.id, selection.geometry, selection.color, width=2)
        return selection

    def removeRoi(self, roi_id):
        removed = super().removeRoi(roi_id)
        if removed:
            self._remove_vispy_roi(roi_id)
        return removed

    def clearRois(self):
        super().clearRois()
        for roi_id in tuple(getattr(self, "_vispy_roi_visuals", {})):
            self._remove_vispy_roi(roi_id)

    def highlightRoi(self, roi_id):
        result = super().highlightRoi(roi_id)
        for current_id, (_item, selection) in self._roi_items.items():
            width = 4 if current_id == str(roi_id) else 2
            self._upsert_vispy_roi(current_id, selection.geometry, selection.color, width=width)
        return result

    def _on_roi_item_changed(self, roi_id):
        self._sync_roi_item_state(roi_id, emit=True)

    def _on_roi_item_live_changed(self, roi_id):
        self._sync_roi_item_state(roi_id, emit=True)

    def _sync_roi_item_state(self, roi_id, *, emit: bool) -> None:
        item_selection = self._roi_items.get(str(roi_id))
        if item_selection is None:
            self._remove_vispy_roi(roi_id)
            return
        _item, selection = item_selection
        if emit:
            super()._on_roi_item_changed(roi_id)
            item_selection = self._roi_items.get(str(roi_id))
            if item_selection is None:
                self._remove_vispy_roi(roi_id)
                return
            _item, selection = item_selection
        self._upsert_vispy_roi(selection.id, selection.geometry, selection.color, width=2)

    def _upsert_vispy_roi(self, roi_id, geometry, color, *, width: int = 2) -> None:
        points = _vispy_roi_points(geometry)
        if points is None:
            self._remove_vispy_roi(roi_id)
            return
        visual = self._vispy_roi_visuals.get(str(roi_id))
        if visual is None:
            visual = self._vispy_visuals.Line(
                points,
                parent=self._vispy_view.scene,
                color=_vispy_color(color),
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
            visual.set_data(pos=points, color=_vispy_color(color), width=float(width))
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
        points = _vispy_roi_handle_points(geometry)
        existing = self._vispy_roi_handle_visuals.pop(str(roi_id), ())
        if existing is None or not isinstance(existing, (list, tuple)):
            existing = (existing,)
        for handle in tuple(existing):
            if handle is None:
                continue
            try:
                handle.parent = None
            except Exception:
                _set_visual_visible(handle, False)
        if points is None:
            return
        size = self._vispy_handle_world_size()
        handles = []
        for x, y in np.asarray(points, dtype=np.float32):
            handle = self._vispy_visuals.Rectangle(
                center=(float(x), float(y)),
                width=float(size),
                height=float(size),
                parent=self._vispy_view.scene,
                color=(0.0, 0.0, 0.0, 0.0),
                border_color=_vispy_color((200, 200, 220)),
            )
            handle.order = 10_001
            handle.visible = True
            handles.append(handle)
        self._vispy_roi_handle_visuals[str(roi_id)] = handles

    def _vispy_handle_world_size(self) -> float:
        try:
            x_range, y_range = self.view.viewRange()
            span = min(abs(float(x_range[1]) - float(x_range[0])), abs(float(y_range[1]) - float(y_range[0])))
        except Exception:
            span = 100.0
        return max(1.5, min(5.0, span * 0.025))

    def eventFilter(self, obj, event):
        if obj is self.graphicsView.viewport() and event.type() == QtCore.QEvent.Type.MouseMove:
            self._update_vispy_roi_cursor(event)
        return super().eventFilter(obj, event)

    def _update_vispy_roi_cursor(self, event) -> None:
        if self._pending_roi_draw_tool is not None or self._drawing_active:
            return
        if self._inspection_tool in {"roi_line", "roi_rectangle", "roi_polyline", "roi_freehand"}:
            return
        scene_pos = self.graphicsView.mapToScene(event.pos())
        view_pos = self.view.mapSceneToView(scene_pos)
        cursor = self._vispy_profile_cursor_for_point(float(view_pos.x()), float(view_pos.y()))
        if cursor is None:
            cursor = self._vispy_roi_cursor_for_point(float(view_pos.x()), float(view_pos.y()))
        viewport = self.graphicsView.viewport()
        if cursor is None:
            if self._vispy_roi_cursor_active:
                viewport.unsetCursor()
                self._vispy_roi_cursor_active = False
            return
        viewport.setCursor(cursor)
        self._vispy_roi_cursor_active = True

    def _vispy_roi_cursor_for_point(self, x: float, y: float):
        for _roi_id, (_item, selection) in reversed(tuple(self._roi_items.items())):
            kind = str(getattr(getattr(selection.geometry, "kind", ""), "value", getattr(selection.geometry, "kind", "")))
            if kind != "rectangle" or selection.geometry.rect is None:
                continue
            rect = tuple(float(value) for value in selection.geometry.rect)
            handle_point = _vispy_roi_handle_points(selection.geometry)
            tolerance = self._vispy_handle_world_size()
            if handle_point is not None and len(handle_point):
                hx, hy = (float(handle_point[0][0]), float(handle_point[0][1]))
                if abs(float(x) - hx) <= tolerance and abs(float(y) - hy) <= tolerance:
                    return QtGui.QCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
            rx, ry, width, height = rect
            x0, x1 = sorted((rx, rx + width))
            y0, y1 = sorted((ry, ry + height))
            if x0 <= float(x) <= x1 and y0 <= float(y) <= y1:
                return QtGui.QCursor(QtCore.Qt.CursorShape.SizeAllCursor)
        return None

    def _vispy_profile_cursor_for_point(self, x: float, y: float):
        if not bool(getattr(self, "_profile_marker_requested_visible", False)):
            return None
        position = self.profileMarkerPosition()
        if position is None:
            return None
        px, py = (float(position[0]), float(position[1]))
        tolerance = max(1.5, self._vispy_handle_world_size())
        if abs(float(x) - px) <= tolerance and abs(float(y) - py) <= tolerance:
            return QtGui.QCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        return None

    def setProfileMarker(self, x, y, visible=True):
        super().setProfileMarker(x, y, visible=visible)
        self._sync_vispy_profile_marker()

    def hideProfileMarker(self):
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
        self._upsert_vispy_line("profile_v", np.asarray([[x, y0], [x, y1]], dtype=np.float32), (230, 60, 30), width=1.5)
        self._upsert_vispy_line("profile_h", np.asarray([[x0, y], [x1, y]], dtype=np.float32), (230, 60, 30), width=1.5)
        marker = max(0.8, min(float(x1 - x0), float(y1 - y0)) * 0.025)
        self._upsert_vispy_line("profile_handle_x", np.asarray([[x - marker, y], [x + marker, y]], dtype=np.float32), (230, 60, 30), width=2.0)
        self._upsert_vispy_line("profile_handle_y", np.asarray([[x, y - marker], [x, y + marker]], dtype=np.float32), (230, 60, 30), width=2.0)
        self._upsert_vispy_profile_dot(x, y)
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

    def _upsert_vispy_profile_dot(self, x: float, y: float) -> None:
        visual = self._vispy_profile_visuals.get("profile_handle_dot")
        if visual is None:
            visual = self._vispy_visuals.Markers(parent=self._vispy_view.scene)
            self._vispy_profile_visuals["profile_handle_dot"] = visual
        visual.set_data(
            np.asarray([[float(x), float(y)]], dtype=np.float32),
            symbol="disc",
            size=9.0,
            face_color=_vispy_color((230, 60, 30)),
            edge_color=_vispy_color((255, 255, 255)),
            edge_width=1.0,
        )
        visual.order = 10_002
        visual.visible = True

    def setMontageTileOverlays(self, overlays):
        overlays = tuple(overlays or ())
        super().setMontageTileOverlays(overlays)
        self._set_vispy_montage_tile_overlays(overlays)

    def clearMontageTileOverlays(self):
        super().clearMontageTileOverlays()
        for visual in getattr(self, "_vispy_overlay_visuals", ()):
            try:
                visual.parent = None
            except Exception:
                _set_visual_visible(visual, False)
        self._vispy_overlay_visuals = []
        self._vispy_canvas.update()

    def _set_vispy_montage_tile_overlays(self, overlays) -> None:
        for visual in getattr(self, "_vispy_overlay_visuals", ()):
            try:
                visual.parent = None
            except Exception:
                _set_visual_visible(visual, False)
        self._vispy_overlay_visuals = []
        for overlay in tuple(overlays or ()):
            fill, border, mark = _overlay_vispy_colors(overlay)
            rect = self._vispy_visuals.Rectangle(
                center=(
                    float(getattr(overlay, "x", 0.0)) + float(getattr(overlay, "width", 1.0)) * 0.5,
                    float(getattr(overlay, "y", 0.0)) + float(getattr(overlay, "height", 1.0)) * 0.5,
                ),
                width=float(max(1.0, getattr(overlay, "width", 1.0))),
                height=float(max(1.0, getattr(overlay, "height", 1.0))),
                parent=self._vispy_view.scene,
                color=fill,
                border_color=border,
            )
            rect.order = 11_000
            rect.visible = True
            self._vispy_overlay_visuals.append(rect)

            mark_visual = self._vispy_visuals.Line(
                _overlay_status_mark_points(overlay),
                parent=self._vispy_view.scene,
                color=mark,
                width=1.25,
                method="gl",
                connect="segments",
            )
            mark_visual.order = 11_001
            mark_visual.visible = True
            self._vispy_overlay_visuals.append(mark_visual)
        self._vispy_canvas.update()

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
                    display_rect=self._current_image_world_rect(),
                )
            finally:
                self._viewport_applying = False

    def _update_histogram_for_vispy(self, histogramData, histogramPlotData, levels) -> None:
        plot_data = self._histogram_plot_data(histogramData)
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
        force_levels: bool = False,
    ):
        from arrayscope.display.montage_tile_layer import TileLayerUpdateStats

        if img is None or geometry is None or getattr(geometry, "montage", None) is None:
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
            needs_data = (
                dirty is None
                or tile_number in dirty
                or state.source_id != source_id
                or state.windowed_rgb != bool(use_windowed_rgb)
                or not state.visible
                or (not use_windowed_rgb and levels_changed)
            )
            if not needs_data:
                if use_windowed_rgb and levels_changed:
                    state.visual.set_levels(level_tuple)
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
                    try:
                        state.visual.cmap = "grays"
                    except Exception:
                        pass
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
        self._record_upload_timing("tile_layer_rgb_window_ms", float(rgb_tiles) * 0.0)
        return TileLayerUpdateStats(visible_items=visible_items, items_updated=updated, items_skipped=skipped, rgb_window_tiles=rgb_tiles)

    def _ensure_vispy_tile(self, tile_number: int, *, windowed_rgb: bool = False) -> _VisPyTileState:
        tile_number = int(tile_number)
        state = self._vispy_tile_visuals.get(tile_number)
        if state is None:
            state = _VisPyTileState()
            self._vispy_tile_visuals[tile_number] = state
        if windowed_rgb:
            if state.windowed_visual is None:
                state.windowed_visual = self._vispy_scene.visuals.create_visual_node(_WindowedRgbVisual)(
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
            self._vispy_view.camera.flip = (
                bool(state.get("xInverted", False)),
                bool(state.get("yInverted", True)),
                False,
            )
            self._vispy_view.camera.set_range(x=(float(x_range[0]), float(x_range[1])), y=(float(y_range[0]), float(y_range[1])), margin=0)
            self._vispy_canvas.update()
        except Exception:
            pass

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
        from vispy.scene import visuals
        from vispy.scene.cameras import PanZoomCamera
        from vispy.visuals import transforms
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("VisPy rendering backend is not available. Install ArrayScope[vispy] or vispy.") from exc
    return scene, visuals, transforms, PanZoomCamera


class _WindowedRgbVisual(Visual):
    _vertex_shader = """
    attribute vec2 a_position;
    attribute vec2 a_texcoord;
    varying vec2 v_texcoord;

    void main() {
        v_texcoord = a_texcoord;
        gl_Position = $transform(vec4(a_position, 0.0, 1.0));
    }
    """

    _fragment_shader = """
    uniform sampler2D u_color_texture;
    uniform sampler2D u_scalar_texture;
    uniform vec2 u_levels;
    varying vec2 v_texcoord;

    void main() {
        vec3 color = texture2D(u_color_texture, v_texcoord).rgb;
        float scalar = texture2D(u_scalar_texture, v_texcoord).r;
        float span = max(u_levels.y - u_levels.x, 1e-12);
        float intensity = clamp((scalar - u_levels.x) / span, 0.0, 1.0);
        float alpha = 1.0;
        if (scalar != scalar) {
            intensity = 0.0;
            alpha = 0.0;
        }
        gl_FragColor = vec4(color * intensity, alpha);
    }
    """

    def __init__(self, **kwargs):
        from vispy import gloo

        self._gloo = gloo
        self._vertices = gloo.VertexBuffer(np.zeros((6, 2), dtype=np.float32))
        self._texcoords = gloo.VertexBuffer(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        self._color_texture = None
        self._scalar_texture = None
        self._shape = (0, 0)
        self._levels = (0.0, 1.0)
        self.color_source_id = None
        self.scalar_source_id = None
        self.upload_count = 0
        self.level_update_count = 0
        super().__init__(vcode=self._vertex_shader, fcode=self._fragment_shader, **kwargs)
        self.set_gl_state("translucent", cull_face=False)
        self._draw_mode = "triangles"
        self.freeze()

    @property
    def levels(self) -> tuple[float, float]:
        return self._levels

    def set_data(
        self,
        color,
        scalar,
        *,
        levels,
        color_source_id=None,
        scalar_source_id=None,
        copy: bool = False,
    ) -> None:
        color_array = _contiguous_color_texture(color, copy=copy)
        scalar_array = _contiguous_scalar(scalar, copy=copy)
        if tuple(color_array.shape[:2]) != tuple(scalar_array.shape[:2]):
            raise ValueError("windowed RGB color and scalar textures must have matching image shape")
        self._set_vertices(tuple(color_array.shape[:2]))
        if self._color_texture is None or tuple(self._shape) != tuple(color_array.shape[:2]):
            self._color_texture = self._gloo.Texture2D(color_array, interpolation="nearest", wrapping="clamp_to_edge")
            self._scalar_texture = self._gloo.Texture2D(scalar_array, interpolation="nearest", wrapping="clamp_to_edge")
        else:
            self._color_texture.set_data(color_array, copy=False)
            self._scalar_texture.set_data(scalar_array, copy=False)
        self._shape = tuple(int(size) for size in color_array.shape[:2])
        self.color_source_id = color_source_id
        self.scalar_source_id = scalar_source_id
        self.upload_count += 1
        self.set_levels(levels, count=False)

    def set_levels(self, levels, *, count: bool = True) -> None:
        self._levels = _normalize_levels(levels, self._levels)
        if count:
            self.level_update_count += 1
        self.update()

    def _set_vertices(self, shape: tuple[int, int]) -> None:
        height, width = (int(shape[0]), int(shape[1]))
        vertices = np.array(
            [
                [0.0, 0.0],
                [float(width), 0.0],
                [float(width), float(height)],
                [0.0, 0.0],
                [float(width), float(height)],
                [0.0, float(height)],
            ],
            dtype=np.float32,
        )
        self._vertices.set_data(vertices)

    def _prepare_transforms(self, view) -> None:
        view.view_program.vert["transform"] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        if self._color_texture is None or self._scalar_texture is None:
            return False
        program = view.view_program
        program["a_position"] = self._vertices
        program["a_texcoord"] = self._texcoords
        program["u_color_texture"] = self._color_texture
        program["u_scalar_texture"] = self._scalar_texture
        program["u_levels"] = tuple(float(value) for value in self._levels)
        return True

    def _bounds(self, axis, view):
        del view
        if axis == 0:
            return (0.0, float(self._shape[1]))
        if axis == 1:
            return (0.0, float(self._shape[0]))
        return (0.0, 0.0)


def _normalize_levels(levels, fallback):
    if levels is None:
        levels = fallback
    low, high = levels
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return (0.0, 1.0)
    return (low, high)


def _contiguous_display(data):
    arr = np.asarray(data)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _contiguous_color_texture(data, *, copy: bool = False):
    arr = np.array(data, copy=True) if copy else np.asarray(data)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) > 1.0:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _contiguous_scalar(data, *, copy: bool = False):
    arr = np.array(data, dtype=np.float32, copy=True) if copy else np.asarray(data, dtype=np.float32)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _coerce_viewport_policy(policy, auto_range):
    if auto_range is not None:
        return ViewportPolicy.FIT if bool(auto_range) else ViewportPolicy.PRESERVE
    return policy


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


def _vispy_roi_handle_points(geometry):
    kind = str(getattr(getattr(geometry, "kind", ""), "value", getattr(geometry, "kind", "")))
    if kind == "rectangle":
        rect = getattr(geometry, "rect", None)
        if rect is None:
            return None
        x, y, width, height = (float(value) for value in rect)
        return np.asarray([[x + width, y + height]], dtype=np.float32)
    points = tuple(getattr(geometry, "points", ()) or ())
    if kind == "line" and len(points) >= 2:
        return np.asarray(points[:2], dtype=np.float32)
    if kind in {"polyline", "freehand_polygon"} and len(points) >= 2:
        return np.asarray(points, dtype=np.float32)
    return None


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
