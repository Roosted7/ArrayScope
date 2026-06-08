import numpy as np
from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from pyqtgraph.graphicsItems.ViewBox import ViewBox

from arrayscope.core.roi import (
    DEFAULT_FREEHAND_SIMPLIFY_TOLERANCE,
    MIN_FREEHAND_POINTS,
    MIN_POLYLINE_POINTS,
    RoiGeometry,
    RoiKind,
    RoiSelection,
    close_polygon,
    simplify_polyline,
)
from arrayscope.core.roi_store import DEFAULT_ROI_COLORS
from arrayscope.display.viewport import ViewportController, ViewportIntent, ViewportPolicy


class ImageView2D(QtWidgets.QWidget):
    """
    Simplified widget for displaying 2D image data.
    
    Features:
    - 2D image display via ImageItem
    - Zoom/pan via ViewBox
    - Histogram with level controls
    - Auto-ranging and level adjustment
    """
    
    roiCreated = QtCore.Signal(object)
    roiChanged = QtCore.Signal(str, object)
    roiDeleted = QtCore.Signal(str)
    imageContextMenuRequested = QtCore.Signal(object, object)

    def __init__(self, parent=None, view=None, imageItem=None):
        """
        Parameters
        ----------
        parent : QWidget
            Parent widget
        view : ViewBox
            If specified, this ViewBox will be used for display
        imageItem : ImageItem
            If specified, this ImageItem will be used for display
        """
        super().__init__(parent)
        
        self.image = None
        self.imageDisp = None
        self.levelMin = None
        self.levelMax = None
        self.displayMode = 'square_pixels'  # Default to square pixels
        self.histogramSource = None
        self._rgbBaseImage = None
        self._profile_vline = None
        self._profile_hline = None
        self._profile_handle = None
        self._profile_bounds_rect = None
        self._profile_marker_callback = None
        self._profile_marker_updating = False
        self._hud_widget = None
        self._evaluation_overlay = None
        self._roi_info_panel = None
        self._inspection_tool = "cursor"
        self._roi_items = {}
        self._roi_counter = 0
        self._drawing_points = []
        self._pending_roi_draw_tool = None
        self._drawing_active = False
        self._freehand_spacing = 1.0
        self.viewport_controller = ViewportController()
        self._viewport_applying = False
        
        # Create the UI layout
        self.setupUI()
        self.graphicsView.viewport().installEventFilter(self)
        
        # Create view if not provided
        if view is None:
            self.view = ViewBox()
        else:
            self.view = view
        self.graphicsView.setCentralItem(self.view)
        self.view.setAspectLocked(True)
        self.view.invertY(True)
        
        # Create image item if not provided
        if imageItem is None:
            self.imageItem = ImageItem(axisOrder="row-major")
        else:
            self.imageItem = imageItem
        self.view.addItem(self.imageItem)
        
        # Setup histogram
        self.histogramImageItem = ImageItem(axisOrder="row-major")
        self.histogram.setImageItem(self.histogramImageItem)
        self.histogram.setLevelMode('mono')  # Force mono mode for scalar values
        self.histogram.item.sigLevelsChanged.connect(self._on_histogram_levels_changed)
        
        # Initialize levels
        self.levelMin = 0.0
        self.levelMax = 1.0

        marker_pen = pg.mkPen((230, 60, 30, 180), width=1)
        self._profile_vline = pg.InfiniteLine(angle=90, movable=True, pen=marker_pen)
        self._profile_hline = pg.InfiniteLine(angle=0, movable=True, pen=marker_pen)
        self._profile_handle = pg.TargetItem(
            pos=(0, 0),
            size=14,
            symbol="o",
            pen=pg.mkPen((230, 60, 30, 220), width=2),
            brush=pg.mkBrush(230, 60, 30, 80),
            movable=True,
        )
        self._profile_vline.setVisible(False)
        self._profile_hline.setVisible(False)
        self._profile_handle.setVisible(False)
        self._profile_vline.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
        self._profile_hline.setCursor(QtCore.Qt.CursorShape.SizeVerCursor)
        self._profile_handle.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self._profile_vline.sigPositionChanged.connect(self._on_profile_marker_changed)
        self._profile_hline.sigPositionChanged.connect(self._on_profile_marker_changed)
        self._profile_handle.sigPositionChanged.connect(self._on_profile_handle_changed)
        self.view.addItem(self._profile_vline)
        self.view.addItem(self._profile_hline)
        self.view.addItem(self._profile_handle)
        self.view.sigRangeChanged.connect(self._on_view_range_changed)
        
    def setupUI(self):
        """Create the user interface"""
        # Main layout
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Graphics view for image display
        self.graphicsView = pg.GraphicsView()
        self.layout.addWidget(self.graphicsView, 1)  # Give it most of the space
        
        # Histogram widget
        self.histogram = pg.HistogramLUTWidget()
        self.layout.addWidget(self.histogram)
        
    def setImage(self, img, autoRange=None, autoLevels=True, levels=None, 
                 pos=None, scale=None, transform=None, autoHistogramRange=True,
                 histogramData=None, viewport_policy=ViewportPolicy.PRESERVE):
        """
        Set the image to be displayed.
        
        Parameters
        ----------
        img : np.ndarray
            2D image data to display
        autoRange : bool
            Whether to auto-scale the view to fit the image
        autoLevels : bool
            Whether to auto-adjust the histogram levels
        levels : tuple
            (min, max) levels for the histogram
        pos : tuple
            Position offset for the image
        scale : tuple  
            Scale factors for the image
        transform : QTransform
            Transform to apply to the image
        autoHistogramRange : bool
            Whether to auto-scale the histogram range
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Image must be a numpy array")
        viewport_policy = _coerce_viewport_policy(viewport_policy, autoRange)
            
        is_rgb = self._is_rgb_image(img)
        if img.ndim != 2 and not is_rgb:
            raise ValueError("ImageView2D only supports 2D scalar or RGB images")
            
        previous_shape = None if self.image is None else tuple(self.image.shape[:2])
        self.image = img
        self.imageDisp = None
        self.histogramSource = histogramData
        
        # Update the image display
        self.updateImage(autoHistogramRange=autoHistogramRange)
        self._update_profile_line_bounds()
        
        # Set levels
        self.histogram.setVisible(True)
        if levels is None and autoLevels:
            self.autoLevels()
        elif levels is not None:
            if isinstance(levels, (list, tuple)) and len(levels) == 2:
                self.setLevels(levels[0], levels[1])
            else:
                self.setLevels(*levels)
            
        # Set transform
        if transform is None:
            if pos is not None or scale is not None:
                if pos is None:
                    pos = (0, 0)
                if scale is None:
                    scale = (1, 1)
                transform = QtGui.QTransform()
                transform.translate(pos[0], pos[1])
                transform.scale(scale[0], scale[1])
        
        if transform is not None:
            self.imageItem.setTransform(transform)
            
        # Update aspect ratio based on display mode
        self._updateAspectRatio()

        self._apply_viewport_policy(tuple(img.shape[:2]), viewport_policy)
            
    def updateImage(self, autoHistogramRange=True):
        """Update the displayed image"""
        if self.image is None:
            return
            
        # For 2D images, we can display directly
        self.imageDisp = self.image
        
        is_rgb = self._is_rgb_image(self.imageDisp)
        self._rgbBaseImage = None
        histogram_data = self.histogramSource
        if histogram_data is None:
            histogram_data = self._histogram_data(self.imageDisp)

        # Calculate min/max levels from the image data for histogram
        self._updateImageLevels(histogram_data)
        histogram_levels = (self.levelMin, self.levelMax)
        
        # Set the image data
        if is_rgb:
            self.histogram.setImageItem(self.histogramImageItem)
            self._rgbBaseImage = self.imageDisp[..., :3].astype(float)
            self.imageDisp = self._rgb_display_for_levels(histogram_levels)
            self.imageItem.setImage(self.imageDisp, autoLevels=False, levels=(0, 255))
            self.histogramImageItem.setImage(histogram_data, autoLevels=False, levels=histogram_levels)
        else:
            self.histogram.setImageItem(self.imageItem)
            self.imageItem.setImage(self.imageDisp, autoLevels=False)
        
        # Update histogram range if requested
        if autoHistogramRange:
            self.histogram.setHistogramRange(self.levelMin, self.levelMax)
            
    def autoRange(self):
        """Auto scale and pan the view to fit the image"""
        if self.imageDisp is not None:
            self._viewport_applying = True
            try:
                self.view.autoRange(padding=0)
            finally:
                self._viewport_applying = False
            
    def _updateImageLevels(self, image=None):
        """Update the min/max levels from the current image data"""
        if image is None:
            image = self.imageDisp
        if image is not None:
            # Use the same approach as the original ImageView
            finite_data = image[np.isfinite(image)]
            if len(finite_data) > 0:
                self.levelMin = float(np.min(finite_data))
                self.levelMax = float(np.max(finite_data))
            else:
                self.levelMin = 0.0
                self.levelMax = 1.0

    def _is_rgb_image(self, img):
        return isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] in (3, 4)

    def _histogram_data(self, img):
        if self._is_rgb_image(img):
            rgb = img[..., :3].astype(float)
            return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return img

    def _rgb_display_for_levels(self, levels=None):
        if self._rgbBaseImage is None:
            return self.imageDisp

        histogram_data = self.histogramSource
        if histogram_data is None:
            histogram_data = self._histogram_data(self._rgbBaseImage)

        if levels is None:
            try:
                levels = self.histogram.getLevels()
            except Exception:
                levels = (self.levelMin, self.levelMax)

        low, high = levels
        span = max(float(high) - float(low), 1e-12)
        intensity = np.clip((histogram_data.astype(float) - float(low)) / span, 0.0, 1.0)
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(self._rgbBaseImage * intensity[..., np.newaxis], 0, 255).astype(np.uint8)

    def _on_histogram_levels_changed(self, *args):
        if self._rgbBaseImage is None or self.histogramSource is None:
            return

        self.imageDisp = self._rgb_display_for_levels()
        self.imageItem.setImage(self.imageDisp, autoLevels=False, levels=(0, 255))
                
    def autoLevels(self):
        """Automatically set the histogram levels based on image data"""
        if self.imageDisp is not None:
            if self._rgbBaseImage is not None:
                image = self.histogramSource
                if image is None:
                    image = self._histogram_data(self.imageDisp)
                self._updateImageLevels(image)
            else:
                self._updateImageLevels()
            self.setLevels(self.levelMin, self.levelMax)
                
    def setLevels(self, min_level, max_level):
        """Set the histogram levels"""
        self.histogram.setLevels(min_level, max_level)
        self._on_histogram_levels_changed()
        
    def getLevels(self):
        """Get the current histogram levels"""
        return self.histogram.getLevels()

    def getHistogramDataBounds(self):
        """Get the current histogram source data bounds."""
        if self.levelMin is None or self.levelMax is None:
            return None
        return (float(self.levelMin), float(self.levelMax))
        
    def setHistogramRange(self, min_val, max_val):
        """Set the range of the histogram"""
        self.histogram.setHistogramRange(min_val, max_val)
        
    def getProcessedImage(self):
        """Get the processed image data"""
        return self.imageDisp
        
    def getView(self):
        """Get the ViewBox containing the image"""
        return self.view
        
    def getImageItem(self):
        """Get the ImageItem"""
        return self.imageItem
        
    def getHistogramWidget(self):
        """Get the histogram widget"""
        return self.histogram

    def setProfileMarker(self, x, y, visible=True):
        """Set or hide the image-space profile marker."""
        if self._profile_vline is None or self._profile_hline is None:
            return
        x, y = self._clamp_profile_point(float(x), float(y))
        self._profile_marker_updating = True
        try:
            self._profile_handle.setPos(float(x), float(y))
            self._update_profile_line_bounds()
            self._profile_vline.setValue(float(x))
            self._profile_hline.setValue(float(y))
            self._profile_vline.setVisible(bool(visible))
            self._profile_hline.setVisible(bool(visible))
            self._profile_handle.setVisible(bool(visible))
            self.view.update()
        finally:
            self._profile_marker_updating = False

    def hideProfileMarker(self):
        """Hide the image-space profile marker."""
        if self._profile_vline is not None:
            self._profile_vline.setVisible(False)
        if self._profile_hline is not None:
            self._profile_hline.setVisible(False)
        if self._profile_handle is not None:
            self._profile_handle.setVisible(False)

    def profileMarkerPosition(self):
        """Return the current profile marker position in image coordinates."""
        if self._profile_vline is None or self._profile_hline is None:
            return None
        if not self._profile_handle.isVisible():
            return None
        return (float(self._profile_vline.value()), float(self._profile_hline.value()))

    def setProfileMarkerCallback(self, callback):
        """Set a callback called with image-space x/y when the marker moves."""
        self.set_profile_marker_callback(callback)

    def set_profile_marker_callback(self, callback):
        """Set a callback called with image-space x/y when the marker moves."""
        self._profile_marker_callback = callback

    def clear_profile_marker_callback(self):
        """Clear the callback called when the marker moves."""
        self._profile_marker_callback = None

    def _on_profile_marker_changed(self, *_args):
        if self._profile_marker_updating:
            return
        if self._profile_vline is None or self._profile_hline is None:
            return
        x = float(self._profile_vline.value())
        y = float(self._profile_hline.value())
        self._profile_marker_updating = True
        try:
            self._update_profile_line_bounds()
            x, y = self._clamp_profile_point(x, y)
            self._profile_vline.setValue(x)
            self._profile_hline.setValue(y)
            self._profile_handle.setPos(x, y)
        finally:
            self._profile_marker_updating = False
        if self._profile_marker_callback is not None:
            self._profile_marker_callback(x, y)

    def _on_profile_handle_changed(self, *_args):
        if self._profile_marker_updating:
            return
        if self._profile_handle is None:
            return
        pos = self._profile_handle.pos()
        x = float(pos.x())
        y = float(pos.y())
        self._profile_marker_updating = True
        try:
            self._update_profile_line_bounds()
            x, y = self._clamp_profile_point(x, y)
            self._profile_handle.setPos(x, y)
            self._profile_vline.setValue(x)
            self._profile_hline.setValue(y)
            self.view.update()
        finally:
            self._profile_marker_updating = False
        if self._profile_marker_callback is not None:
            self._profile_marker_callback(x, y)

    def _update_profile_line_bounds(self):
        if self.image is None:
            return
        x0, y0, x1, y1 = self._current_profile_bounds()
        if self._profile_vline is not None:
            self._profile_vline.setBounds((x0, x1))
        if self._profile_hline is not None:
            self._profile_hline.setBounds((y0, y1))
        if self._profile_handle is not None:
            pos = self._profile_handle.pos()
            x, y = self._clamp_profile_point(float(pos.x()), float(pos.y()))
            if (x, y) != (float(pos.x()), float(pos.y())):
                self._profile_handle.setPos(x, y)
            if self._profile_vline is not None:
                self._profile_vline.setValue(x)
            if self._profile_hline is not None:
                self._profile_hline.setValue(y)

    def setProfileMarkerBoundsRect(self, rect):
        if rect is None:
            self._profile_bounds_rect = None
        else:
            x0, y0, x1, y1 = rect
            self._profile_bounds_rect = (float(x0), float(y0), float(x1), float(y1))
        self._update_profile_line_bounds()

    def _clamp_profile_point(self, x, y):
        if self.image is None:
            return (float(x), float(y))
        x0, y0, x1, y1 = self._current_profile_bounds()
        x = max(x0, min(float(x), x1))
        y = max(y0, min(float(y), y1))
        return (x, y)

    def _current_profile_bounds(self):
        height, width = self.image.shape[:2]
        x0, y0, x1, y1 = self._profile_bounds_rect or (0.0, 0.0, float(max(0, width - 1)), float(max(0, height - 1)))
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        return (float(x0), float(y0), float(x1), float(y1))
        
    def clear(self):
        """Clear the displayed image"""
        self.image = None
        self.imageDisp = None
        self.imageItem.clear()
        self.hideProfileMarker()
        
    def setColorMap(self, colormap):
        """Set the color map for the histogram"""
        self.histogram.gradient.setColorMap(colormap)
        
    def setDisplayMode(self, mode):
        """Set the display mode.

        Modes:
        - 'square_pixels': force square pixel display (aspect ratio 1.0)
        - 'fit'          : allow non-uniform scaling so the entire image fits viewport
        """
        if mode not in ('square_pixels', 'fit'):
            raise ValueError(f"Unknown display mode: {mode}")
        self.displayMode = mode
        self._updateAspectRatio()

    def fitToView(self):
        self.setDisplayMode("fit")
        if self.image is not None:
            self._viewport_applying = True
            try:
                self.viewport_controller.fit(self.view)
            finally:
                self._viewport_applying = False

    def oneToOne(self):
        self.setDisplayMode("square_pixels")
        if self.image is not None:
            self._viewport_applying = True
            try:
                self.viewport_controller.one_to_one(self.view, self.image.shape[:2], self.graphicsView.viewport().size())
            finally:
                self._viewport_applying = False

    def autoWindow(self):
        self.autoLevels()

    def setHudWidget(self, widget):
        self._hud_widget = widget
        if widget is not None:
            widget.setParent(self.graphicsView)
            widget.hide()

    def setEvaluationOverlay(self, visible: bool, text: str = ""):
        if self._evaluation_overlay is None:
            overlay = QtWidgets.QLabel(self.graphicsView)
            overlay.setObjectName("EvaluationOverlay")
            overlay.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            overlay.setStyleSheet(
                "QLabel#EvaluationOverlay { background: rgba(20, 20, 20, 170); color: white; "
                "padding: 6px 8px; border-radius: 4px; }"
            )
            self._evaluation_overlay = overlay
        self._evaluation_overlay.setText(str(text))
        self._evaluation_overlay.adjustSize()
        self._evaluation_overlay.move(10, 10)
        self._evaluation_overlay.setVisible(bool(visible))
        if visible:
            self._evaluation_overlay.raise_()

    def setRoiInfoText(self, text):
        text = str(text or "")
        if not text:
            if self._roi_info_panel is not None:
                self._roi_info_panel.hide()
            return
        if self._roi_info_panel is None:
            self._roi_info_panel = _MovableInfoPanel(self.graphicsView)
            self._roi_info_panel.move(12, 44)
        self._roi_info_panel.setText(text)
        self._roi_info_panel.adjustSize()
        self._roi_info_panel.show()
        self._roi_info_panel.raise_()

    def setImageStale(self, stale: bool):
        if self.imageItem is not None:
            self.imageItem.setOpacity(0.55 if stale else 1.0)

    def showHudText(self, text, scene_pos):
        if self._hud_widget is None:
            return
        local = self.graphicsView.mapFromScene(scene_pos)
        self._hud_widget.show_text_near(text, local)

    def hideHud(self):
        if self._hud_widget is not None:
            self._hud_widget.hide()

    def setInspectionTool(self, tool):
        allowed = {"cursor", "profile", "roi_line", "roi_rectangle", "roi_polyline", "roi_freehand"}
        if tool not in allowed:
            raise ValueError(f"unknown inspection tool: {tool}")
        self._inspection_tool = str(tool)
        if tool in {"profile", "roi_line", "roi_rectangle"} or self._pending_roi_draw_tool is not None:
            self.getView().setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.getView().unsetCursor()

    def inspectionTool(self):
        return self._inspection_tool

    def beginRoiDrawingOnce(self, tool):
        if tool not in {"roi_polyline", "roi_freehand"}:
            return False
        self._pending_roi_draw_tool = str(tool)
        self._drawing_active = False
        self._drawing_points = []
        self.getView().setCursor(QtCore.Qt.CursorShape.CrossCursor)
        return True

    def cancelPendingRoiDrawing(self):
        self._pending_roi_draw_tool = None
        self._drawing_active = False
        self._drawing_points = []
        if self._inspection_tool in {"profile", "roi_line", "roi_rectangle"}:
            self.getView().setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.getView().unsetCursor()

    def createRoi(self, kind, *, points=None, rect=None, line_width=1.0, label=None, color=None):
        kind = kind if isinstance(kind, RoiKind) else RoiKind(str(kind))
        points = tuple(points or ())
        if kind == RoiKind.LINE and len(points) < 2:
            points = self._default_line_points()
        if kind == RoiKind.RECTANGLE and rect is None:
            rect = self._default_rect()
        if kind == RoiKind.POLYLINE and len(points) < MIN_POLYLINE_POINTS:
            points = self._default_polyline_points()
        if kind == RoiKind.FREEHAND_POLYGON and len(points) < MIN_FREEHAND_POINTS:
            raise ValueError("freehand ROI requires a drag path")
        if kind == RoiKind.FREEHAND_POLYGON:
            points = close_polygon(simplify_polyline(points, DEFAULT_FREEHAND_SIMPLIFY_TOLERANCE))

        geometry = RoiGeometry(
            kind=kind,
            points=points,
            rect=rect,
            line_width=line_width,
            closed=kind == RoiKind.FREEHAND_POLYGON,
        )
        roi_id = f"roi-{self._roi_counter + 1}"
        self._roi_counter += 1
        color = DEFAULT_ROI_COLORS[(self._roi_counter - 1) % len(DEFAULT_ROI_COLORS)] if color is None else tuple(int(value) for value in color[:3])
        selection = RoiSelection(
            id=roi_id,
            label=label or _default_roi_label(kind, self._roi_counter),
            geometry=geometry,
            color=color,
        )
        item = self._item_for_roi(selection)
        self._roi_items[roi_id] = (item, selection)
        self.view.addItem(item)
        item.sigRegionChangeFinished.connect(lambda _item=item, roi_id=roi_id: self._on_roi_item_changed(roi_id))
        self.roiCreated.emit(selection)
        return selection

    def removeRoi(self, roi_id):
        item_selection = self._roi_items.pop(str(roi_id), None)
        if item_selection is None:
            return False
        item, _selection = item_selection
        self.view.removeItem(item)
        self.roiDeleted.emit(str(roi_id))
        return True

    def clearRois(self):
        for roi_id in tuple(self._roi_items):
            self.removeRoi(roi_id)

    def roiSelections(self):
        return tuple(selection for _item, selection in self._roi_items.values())

    def highlightRoi(self, roi_id):
        roi_id = str(roi_id)
        for current_id, (item, selection) in self._roi_items.items():
            width = 4 if current_id == roi_id else 2
            item.setPen(pg.mkPen(selection.color + (255,), width=width))
        return roi_id in self._roi_items

    def _item_for_roi(self, selection):
        geometry = selection.geometry
        pen = pg.mkPen(selection.color + (220,), width=2)
        hover_pen = pg.mkPen(selection.color + (255,), width=3)
        if geometry.kind == RoiKind.LINE:
            return pg.LineSegmentROI(geometry.points[:2], pen=pen, hoverPen=hover_pen, movable=True)
        if geometry.kind == RoiKind.RECTANGLE:
            x, y, width, height = geometry.rect
            return pg.RectROI((x, y), (width, height), pen=pen, hoverPen=hover_pen, movable=True)
        if geometry.kind in (RoiKind.POLYLINE, RoiKind.FREEHAND_POLYGON):
            return pg.PolyLineROI(
                geometry.points,
                closed=geometry.kind == RoiKind.FREEHAND_POLYGON,
                pen=pen,
                hoverPen=hover_pen,
                movable=True,
            )
        raise ValueError(f"unsupported ROI kind: {geometry.kind}")

    def _on_roi_item_changed(self, roi_id):
        item_selection = self._roi_items.get(str(roi_id))
        if item_selection is None:
            return
        item, selection = item_selection
        geometry = self._geometry_from_item(item, selection.geometry)
        updated = RoiSelection(
            id=selection.id,
            label=selection.label,
            geometry=geometry,
            enabled=selection.enabled,
            color=selection.color,
        )
        self._roi_items[str(roi_id)] = (item, updated)
        self.roiChanged.emit(str(roi_id), geometry)

    def _geometry_from_item(self, item, previous):
        state = item.getState()
        if previous.kind == RoiKind.RECTANGLE:
            pos = state["pos"]
            size = state["size"]
            return RoiGeometry(previous.kind, rect=(float(pos.x()), float(pos.y()), float(size.x()), float(size.y())))
        if previous.kind in (RoiKind.LINE, RoiKind.POLYLINE, RoiKind.FREEHAND_POLYGON):
            pos = state.get("pos")
            dx = 0.0 if pos is None else float(pos.x())
            dy = 0.0 if pos is None else float(pos.y())
            points = tuple((float(point.x()) + dx, float(point.y()) + dy) for point in state.get("points", ()))
            if previous.kind == RoiKind.FREEHAND_POLYGON:
                points = close_polygon(points)
            return RoiGeometry(previous.kind, points=points, line_width=previous.line_width, closed=previous.closed)
        return previous

    def _default_line_points(self):
        x, y, width, height = self._default_rect()
        return ((x, y + height * 0.5), (x + width, y + height * 0.5))

    def _default_polyline_points(self):
        x, y, width, height = self._default_rect()
        return ((x, y), (x + width * 0.5, y + height), (x + width, y))

    def _default_rect(self):
        if self.image is None:
            return (0.0, 0.0, 1.0, 1.0)
        height, width = self.image.shape[:2]
        rect_width = max(1.0, width * 0.25)
        rect_height = max(1.0, height * 0.25)
        return ((width - rect_width) * 0.5, (height - rect_height) * 0.5, rect_width, rect_height)
        
    def _updateAspectRatio(self):
        """Update the aspect ratio based on display mode"""
        if self.image is None:
            return
            
        if self.displayMode == 'square_pixels':
            # Square pixels: maintain 1:1 aspect ratio
            self.view.setAspectLocked(True, ratio=1.0)
        elif self.displayMode == 'fit':
            # Fit: allow free aspect so the whole image fits inside the view box
            self.view.setAspectLocked(False)

    def _apply_viewport_policy(self, image_shape, viewport_policy):
        self._viewport_applying = True
        try:
            self.viewport_controller.apply_after_image(
                self.view,
                image_shape,
                self.graphicsView.viewport().size(),
                policy=viewport_policy,
            )
        finally:
            self._viewport_applying = False

    def _on_view_range_changed(self, *_args):
        if not self._viewport_applying:
            self.viewport_controller.note_user_range_changed()

    # --- Qt Events -----------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self.graphicsView.viewport() and event.type() == QtCore.QEvent.Type.Resize:
            if self.image is not None:
                self._viewport_applying = True
                try:
                    self.viewport_controller.resize(self.view, self.image.shape[:2], event.size())
                finally:
                    self._viewport_applying = False
        if obj is self.graphicsView.viewport() and self._handle_context_menu_event(event):
            return True
        if obj is self.graphicsView.viewport() and self._handle_roi_drawing_event(event):
            return True
        return super().eventFilter(obj, event)

    def _handle_context_menu_event(self, event):
        if event.type() != QtCore.QEvent.Type.MouseButtonPress or event.button() != QtCore.Qt.MouseButton.RightButton:
            return False
        image_point = self._event_image_point(event)
        self.imageContextMenuRequested.emit(event.globalPos(), image_point)
        return True

    def _handle_roi_drawing_event(self, event):
        if self._pending_roi_draw_tool is None and not self._drawing_active:
            return False
        tool = self._pending_roi_draw_tool
        event_type = event.type()
        if event_type == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
            point = self._event_image_point(event)
            if point is None:
                return False
            self._drawing_active = True
            self._drawing_points = [point]
            return True
        if event_type == QtCore.QEvent.Type.MouseMove and self._drawing_active:
            point = self._event_image_point(event)
            if point is None:
                return True
            if not self._drawing_points or _point_distance(self._drawing_points[-1], point) >= self._freehand_spacing:
                self._drawing_points.append(point)
            return True
        if event_type == QtCore.QEvent.Type.MouseButtonRelease and self._drawing_active:
            points = tuple(self._drawing_points)
            self._drawing_active = False
            self._drawing_points = []
            self._pending_roi_draw_tool = None
            if tool == "roi_freehand" and len(points) >= MIN_FREEHAND_POINTS:
                self.createRoi(RoiKind.FREEHAND_POLYGON, points=points)
            elif tool == "roi_polyline" and len(points) >= MIN_POLYLINE_POINTS:
                self.createRoi(RoiKind.POLYLINE, points=points)
            self.setInspectionTool(self._inspection_tool if self._inspection_tool in {"cursor", "profile"} else "cursor")
            return True
        return False

    def _event_image_point(self, event):
        if self.image is None:
            return None
        scene_pos = self.graphicsView.mapToScene(event.pos())
        view_point = self.view.mapSceneToView(scene_pos)
        x = max(0.0, min(float(view_point.x()), max(0.0, float(self.image.shape[1] - 1))))
        y = max(0.0, min(float(view_point.y()), max(0.0, float(self.image.shape[0] - 1))))
        return (x, y)

    def resizeEvent(self, event):
        """On resize, if in 'fit' mode keep the image fully visible."""
        super().resizeEvent(event)


def _coerce_viewport_policy(viewport_policy, auto_range):
    if auto_range is not None:
        viewport_policy = ViewportPolicy.FIT_ONCE if bool(auto_range) else ViewportPolicy.PRESERVE
    if isinstance(viewport_policy, (ViewportPolicy, ViewportIntent)):
        return viewport_policy
    return ViewportPolicy(str(viewport_policy))


def _default_roi_label(kind, index):
    labels = {
        RoiKind.LINE: "Line",
        RoiKind.RECTANGLE: "Rectangle",
        RoiKind.POLYLINE: "Polyline",
        RoiKind.FREEHAND_POLYGON: "Freehand",
    }
    return f"{labels.get(kind, 'ROI')} {int(index)}"


def _point_distance(a, b):
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


class _MovableInfoPanel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_offset = None
        self.setObjectName("RoiInfoPanel")
        self.setWordWrap(False)
        self.setStyleSheet(
            "QLabel#RoiInfoPanel { background: rgba(20, 20, 20, 175); color: white; "
            "border: 1px solid rgba(255, 255, 255, 90); border-radius: 4px; padding: 6px 8px; }"
        )

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_offset = event.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_offset is None:
            return super().mouseMoveEvent(event)
        next_pos = self.mapToParent(event.pos() - self._drag_offset)
        parent = self.parentWidget()
        if parent is not None:
            next_pos.setX(max(0, min(next_pos.x(), parent.width() - self.width())))
            next_pos.setY(max(0, min(next_pos.y(), parent.height() - self.height())))
        self.move(next_pos)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        event.accept()
