import numpy as np
from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from pyqtgraph.graphicsItems.ViewBox import ViewBox


class ImageView2D(QtWidgets.QWidget):
    """
    Simplified widget for displaying 2D image data.
    
    Features:
    - 2D image display via ImageItem
    - Zoom/pan via ViewBox
    - Histogram with level controls
    - Auto-ranging and level adjustment
    """
    
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
        self._profile_marker_callback = None
        self._profile_marker_updating = False
        self._hud_widget = None
        self._evaluation_overlay = None
        
        # Create the UI layout
        self.setupUI()
        
        # Create view if not provided
        if view is None:
            self.view = ViewBox()
        else:
            self.view = view
        self.graphicsView.setCentralItem(self.view)
        self.view.setAspectLocked(True)
        self.view.invertY()
        
        # Create image item if not provided
        if imageItem is None:
            self.imageItem = ImageItem()
        else:
            self.imageItem = imageItem
        self.view.addItem(self.imageItem)
        
        # Setup histogram
        self.histogramImageItem = ImageItem()
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
        
    def setImage(self, img, autoRange=True, autoLevels=True, levels=None, 
                 pos=None, scale=None, transform=None, autoHistogramRange=True,
                 histogramData=None):
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
            
        is_rgb = self._is_rgb_image(img)
        if img.ndim != 2 and not is_rgb:
            raise ValueError("ImageView2D only supports 2D scalar or RGB images")
            
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
        
        # Auto range the view
        if autoRange:
            self.autoRange()
            
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
            self.view.autoRange()
            
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
        if not self._profile_vline.isVisible() or not self._profile_hline.isVisible():
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
        self._profile_marker_updating = True
        try:
            self._update_profile_line_bounds()
            x = float(self._profile_vline.value())
            y = float(self._profile_hline.value())
            if self._profile_handle is not None:
                self._profile_handle.setPos(x, y)
        finally:
            self._profile_marker_updating = False
        self.view.update()
        if self._profile_marker_callback is None:
            return
        position = self.profileMarkerPosition()
        if position is not None:
            self._profile_marker_callback(*position)

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
        height, width = self.image.shape[:2]
        if self._profile_vline is not None:
            self._profile_vline.setBounds((0, max(0, width - 1)))
            self._profile_vline.setSpan(0.0, 1.0)
        if self._profile_hline is not None:
            self._profile_hline.setBounds((0, max(0, height - 1)))
            self._profile_hline.setSpan(0.0, 1.0)
        if self._profile_handle is not None:
            pos = self._profile_handle.pos()
            x = max(0.0, min(float(pos.x()), float(max(0, width - 1))))
            y = max(0.0, min(float(pos.y()), float(max(0, height - 1))))
            if (x, y) != (float(pos.x()), float(pos.y())):
                self._profile_handle.setPos(x, y)
        
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
        - 'square_fov'   : lock aspect ratio to image width/height (field of view square)
        - 'fit'          : allow non-uniform scaling so the entire image fits viewport
        """
        if mode not in ('square_pixels', 'square_fov', 'fit'):
            raise ValueError(f"Unknown display mode: {mode}")
        self.displayMode = mode
        self._updateAspectRatio()

    def fitToView(self):
        self.setDisplayMode("fit")
        self.view.autoRange()

    def oneToOne(self):
        self.setDisplayMode("square_pixels")
        self.view.autoRange()

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
        
    def _updateAspectRatio(self):
        """Update the aspect ratio based on display mode"""
        if self.image is None:
            return
            
        if self.displayMode == 'square_pixels':
            # Square pixels: maintain 1:1 aspect ratio
            self.view.setAspectLocked(True, ratio=1.0)
        elif self.displayMode == 'square_fov':
            # Square FOV: adjust aspect ratio based on image dimensions
            height, width = self.image.shape[:2]
            aspect_ratio = width / height
            self.view.setAspectLocked(True, ratio=aspect_ratio)
        elif self.displayMode == 'fit':
            # Fit: allow free aspect so the whole image fits inside the view box
            self.view.setAspectLocked(False)
            # Ensure view box ranges cover the image exactly
            self.view.autoRange()
        
        # Trigger a refresh of the view
        if hasattr(self, 'imageItem') and self.imageItem is not None:
            self.view.autoRange()

    # --- Qt Events -----------------------------------------------------
    def resizeEvent(self, event):
        """On resize, if in 'fit' mode keep the image fully visible."""
        super().resizeEvent(event)
        if self.displayMode == 'fit' and self.image is not None:
            self.view.autoRange()
