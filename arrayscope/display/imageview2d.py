from time import perf_counter
from typing import TYPE_CHECKING

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
from arrayscope.core.gui_callback_budget import GuiCallbackBudget, GuiCallbackObservation
from arrayscope.core.runtime_diagnostics import ImageUploadTiming
from arrayscope.display.backends import RasterCommitMode
from arrayscope.display.backend_contract import PYQTGRAPH_CAPABILITIES
from arrayscope.display.histogram_controller import HistogramDisplayController, HistogramLevelPreviewController
from arrayscope.display.image_upload import ensure_imageitem_array, rgb_display_for_levels
from arrayscope.display.interaction import (
    CursorIntent,
    DisplayInteractionController,
    DisplayInteractionState,
    DragResult,
    InteractionTarget,
    PointerPhase,
)
from arrayscope.display.levels import finite_bounds
from arrayscope.display.shader_mapping import default_gray_lut, normalize_lut_rgb
from arrayscope.display.layers import ViewLayerOwner
from arrayscope.display.backends.pyqtgraph.histogram_adapter import PyQtGraphHistogramAdapter
from arrayscope.display.backends.pyqtgraph.tiles import MontageTileLayer, TileLayerUpdateStats
from arrayscope.display.model.frame import TileCommitReport
from arrayscope.display.overlay_hit_test import RoiHitIndex
from arrayscope.display.overlays import MontageTileOverlay, MontageTileOverlayItem
from arrayscope.display.profile_marker import ProfileMarkerOwner
from arrayscope.display.pointer_interaction import QtPointerInteractionDriver
from arrayscope.display.roi_items import (
    MovableInfoPanel,
    default_roi_label,
    item_for_roi,
    make_item_passive,
    sync_item_to_roi_geometry,
)
from arrayscope.display.viewport import (
    MIN_VIEWPORT_CONTENT_FRACTION,
    ViewportController,
    ViewportIntent,
    ViewportPolicy,
    coerce_viewport_policy,
    constrain_view_range,
)

if TYPE_CHECKING:
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState


class ImageViewShell(QtWidgets.QWidget):
    # Backends that implement the typed tile-payload method can bypass CPU
    # montage canvas composition.  Renderer orchestration checks this
    # capability rather than branching on a backend name.
    rendering_capabilities = PYQTGRAPH_CAPABILITIES

    # Emitted only for explicit user edits of the histogram/LUT levels.
    userLevelsChanged = QtCore.Signal()
    autoWindowRequested = QtCore.Signal()

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
        self.surface = self
        self._surface_teardown_done = False
        
        self.image = None
        self.imageDisp = None
        self.levelMin = None
        self.levelMax = None
        self._histogramDataBounds = None
        self._displayLevels = None
        self._applying_presentation = False
        self.displayMode = 'square_pixels'  # Default to square pixels
        self.histogramSource = None
        self.histogramPlotSource = None
        self._rgbBaseImage = None
        self._histogram_adapter = None
        self._histogram_preview_controller = None
        self._histogram_display_controller = None
        self._upload_timing = None
        self._last_upload_timing = ImageUploadTiming()
        self._gui_callback_observer = None
        self._gui_callback_budget_provider = None
        self._background_task_submitter = None
        self._level_presentation_change_handler = None
        self._montage_display_mode = "canvas"
        self._montage_tile_layer = None
        self._montage_tile_layer_histogram_key = None
        self._profile_vline = None
        self._profile_hline = None
        self._profile_handle = None
        self._profile_marker = ProfileMarkerOwner()
        self._profile_marker_callback = None
        self._profile_marker_updating = False
        self._profile_marker_requested_visible = False
        self._hud_widget = None
        self._evaluation_overlay = None
        self._roi_info_panel = None
        self.interaction_controller = DisplayInteractionController()
        self._pointer_interaction = QtPointerInteractionDriver(self, self.interaction_controller)
        self._interaction_application = None
        self._last_profile_marker_position: tuple[float, float] | None = None
        self._roi_items = {}
        self._roi_hit_index = RoiHitIndex()
        self._highlighted_roi_id: str | None = None
        self._interaction_visual_roi_id: str | None = None
        self._interaction_visual_profile_part: str | None = None
        self._montage_tile_overlay_item = None
        self._montage_tile_overlay_items = []
        self._roi_counter = 0
        self._freehand_spacing = 1.0
        self.viewport_controller = ViewportController()
        self._viewport_applying = False
        self._viewport_constraining = False
        self._last_accepted_view_range = None
        self._fit_mode_reminder_last_ms = 0.0
        self._display_colormap = None
        self._display_colormap_lut = default_gray_lut()
        self._display_colormap_key = _array_content_key(self._display_colormap_lut)
        
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
        self._layer_owner = ViewLayerOwner(self.view)
        
        # Create image item if not provided
        if imageItem is None:
            self.imageItem = ImageItem(axisOrder="row-major")
        else:
            self.imageItem = imageItem
        self._layer_owner.add_image_item(self.imageItem)
        self._montage_tile_layer = MontageTileLayer(
            self._layer_owner,
            set_image_item_data=self._set_image_item_data,
            record_upload_timing=self._record_upload_timing,
            histogram_levels_for_display=self._histogram_levels_for_display,
            is_rgb_image=self._is_rgb_image,
        )
        
        # Setup histogram
        self.histogramImageItem = ImageItem(axisOrder="row-major")
        self._bind_histogram_item(self.histogramImageItem)
        self._histogram_preview_controller = HistogramLevelPreviewController(self)
        self._histogram_display_controller = HistogramDisplayController(self)
        self.histogram.setLevelMode('mono')  # Force mono mode for scalar values
        self.histogram.item.sigLevelsChanged.connect(self._on_histogram_levels_changed)
        finish_signal = getattr(self.histogram.item, "sigLevelChangeFinished", None)
        if finish_signal is not None:
            finish_signal.connect(self._on_histogram_level_change_finished)
        self._histogram_display_controller.install()
        
        # Initialize levels
        self.levelMin = 0.0
        self.levelMax = 1.0
        self._displayLevels = (0.0, 1.0)
        self._histogramDataBounds = (0.0, 1.0)

        marker_pen = pg.mkPen((230, 60, 30, 180), width=1)
        self._profile_vline = pg.InfiniteLine(angle=90, movable=False, pen=marker_pen)
        self._profile_hline = pg.InfiniteLine(angle=0, movable=False, pen=marker_pen)
        self._profile_handle = pg.TargetItem(
            pos=(0, 0),
            size=14,
            symbol="o",
            pen=pg.mkPen((230, 60, 30, 220), width=2),
            brush=pg.mkBrush(230, 60, 30, 80),
            movable=False,
        )
        self._profile_vline.setVisible(False)
        self._profile_hline.setVisible(False)
        self._profile_handle.setVisible(False)
        self._profile_vline.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
        self._profile_hline.setCursor(QtCore.Qt.CursorShape.SizeVerCursor)
        self._profile_handle.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self._profile_vline.sigPositionChanged.connect(lambda *_args: self._on_profile_marker_changed("vertical"))
        self._profile_hline.sigPositionChanged.connect(lambda *_args: self._on_profile_marker_changed("horizontal"))
        self._profile_handle.sigPositionChanged.connect(lambda *_args: self._on_profile_handle_changed("center"))
        make_item_passive(self._profile_vline)
        make_item_passive(self._profile_hline)
        make_item_passive(self._profile_handle)
        self._layer_owner.add_profile_marker_items(self._profile_vline, self._profile_hline, self._profile_handle)
        self.view.sigRangeChanged.connect(self._on_view_range_changed)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            try:
                app.applicationStateChanged.connect(self._on_application_state_changed)
                self._interaction_application = app
            except Exception:
                pass
    def _histogram_preview_immediate(self) -> bool:
        return True

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
        self._histogram_adapter = PyQtGraphHistogramAdapter(self.histogram)

    def _start_upload_timing(self, mode: str) -> None:
        self._upload_timing = {
            "mode": str(mode),
            "start": perf_counter(),
            "visible_upload_ms": 0.0,
            "histogram_upload_ms": 0.0,
            "histogram_bind_ms": 0.0,
            "histogram_recompute_ms": 0.0,
            "level_sync_ms": 0.0,
            "rgb_window_ms": 0.0,
            "tile_layer_upload_ms": 0.0,
            "tile_layer_rgb_window_ms": 0.0,
            "profile_bounds_ms": 0.0,
            "visible_bytes": 0,
            "visible_pixels": 0,
            "histogram_bytes": 0,
            "histogram_pixels": 0,
            "fast_same_object": False,
            "tile_layer_visible_items": 0,
            "tile_layer_items_created": 0,
            "tile_layer_items_updated": 0,
            "tile_layer_items_skipped": 0,
            "tile_layer_rgb_window_tiles": 0,
            "tile_layer_image_replacements": 0,
            "tile_layer_existing_items_shown": 0,
            "tile_layer_relocated_tiles": 0,
            "tile_layer_resident_items": 0,
            "tile_layer_storage_capacity": 0,
            "tile_layer_storage_rebuilds": 0,
            "tile_layer_storage_evictions": 0,
            "tile_layer_texture_uploads": 0,
            "tile_layer_texture_upload_bytes": 0,
            "tile_layer_texture_prepare_ms": 0.0,
            "tile_layer_texture_submit_ms": 0.0,
            "tile_layer_vertex_uploads": 0,
            "tile_layer_level_updates": 0,
            "tile_layer_level_update_pending_items": 0,
            "tile_layer_estimated_gpu_bytes": 0,
            "tile_layer_cpu_shadow_bytes": 0,
            "tile_layer_page_count": 0,
            "tile_layer_active_pages": 0,
            "tile_layer_device_max_texture_size": 0,
            "tile_layer_budget_bytes": 0,
            "tile_layer_near_resident_items": 0,
            "tile_layer_warm_resident_items": 0,
            "tile_layer_evicted_near_items": 0,
            "tile_layer_capacity_warning": "",
            "tile_layer_lod_level": 0,
            "tile_layer_lod_factor": 1,
            "tile_layer_source_texels_per_pixel": 0.0,
            "tile_layer_gutter_pixels": 0,
            "tile_layer_mipmap_updates": 0,
            "tile_layer_mipmap_available": False,
            "tile_layer_complex_texture_uploads": 0,
            "tile_layer_shader_uniform_updates": 0,
            "cpu_complex_prep_ms": 0.0,
        }

    def _record_upload_timing(self, field: str, ms: float) -> None:
        timing = self._upload_timing
        if timing is not None:
            timing[field] = float(timing.get(field, 0.0) or 0.0) + float(ms)

    def _finish_upload_timing(self) -> None:
        timing = self._upload_timing
        if timing is None:
            return
        self._last_upload_timing = ImageUploadTiming(
            total_ms=(perf_counter() - float(timing["start"])) * 1000.0,
            visible_upload_ms=float(timing["visible_upload_ms"]),
            histogram_upload_ms=float(timing["histogram_upload_ms"]),
            histogram_bind_ms=float(timing["histogram_bind_ms"]),
            histogram_recompute_ms=float(timing["histogram_recompute_ms"]),
            level_sync_ms=float(timing["level_sync_ms"]),
            rgb_window_ms=float(timing["rgb_window_ms"]),
            tile_layer_upload_ms=float(timing["tile_layer_upload_ms"]),
            tile_layer_rgb_window_ms=float(timing["tile_layer_rgb_window_ms"]),
            profile_bounds_ms=float(timing["profile_bounds_ms"]),
            visible_bytes=int(timing["visible_bytes"]),
            visible_pixels=int(timing["visible_pixels"]),
            histogram_bytes=int(timing["histogram_bytes"]),
            histogram_pixels=int(timing["histogram_pixels"]),
            fast_same_object=bool(timing["fast_same_object"]),
            mode=str(timing["mode"]),
            tile_layer_visible_items=int(timing["tile_layer_visible_items"]),
            tile_layer_items_created=int(timing["tile_layer_items_created"]),
            tile_layer_items_updated=int(timing["tile_layer_items_updated"]),
            tile_layer_items_skipped=int(timing["tile_layer_items_skipped"]),
            tile_layer_rgb_window_tiles=int(timing["tile_layer_rgb_window_tiles"]),
            tile_layer_image_replacements=int(timing["tile_layer_image_replacements"]),
            tile_layer_existing_items_shown=int(timing["tile_layer_existing_items_shown"]),
            tile_layer_relocated_tiles=int(timing["tile_layer_relocated_tiles"]),
            tile_layer_resident_items=int(timing["tile_layer_resident_items"]),
            tile_layer_storage_capacity=int(timing["tile_layer_storage_capacity"]),
            tile_layer_storage_rebuilds=int(timing["tile_layer_storage_rebuilds"]),
            tile_layer_storage_evictions=int(timing["tile_layer_storage_evictions"]),
            tile_layer_texture_uploads=int(timing["tile_layer_texture_uploads"]),
            tile_layer_texture_upload_bytes=int(timing["tile_layer_texture_upload_bytes"]),
            tile_layer_texture_prepare_ms=float(timing["tile_layer_texture_prepare_ms"]),
            tile_layer_texture_submit_ms=float(timing["tile_layer_texture_submit_ms"]),
            tile_layer_vertex_uploads=int(timing["tile_layer_vertex_uploads"]),
            tile_layer_level_updates=int(timing["tile_layer_level_updates"]),
            tile_layer_level_update_pending_items=int(timing["tile_layer_level_update_pending_items"]),
            tile_layer_estimated_gpu_bytes=int(timing["tile_layer_estimated_gpu_bytes"]),
            tile_layer_cpu_shadow_bytes=int(timing["tile_layer_cpu_shadow_bytes"]),
            tile_layer_page_count=int(timing["tile_layer_page_count"]),
            tile_layer_active_pages=int(timing["tile_layer_active_pages"]),
            tile_layer_device_max_texture_size=int(timing["tile_layer_device_max_texture_size"]),
            tile_layer_budget_bytes=int(timing["tile_layer_budget_bytes"]),
            tile_layer_near_resident_items=int(timing["tile_layer_near_resident_items"]),
            tile_layer_warm_resident_items=int(timing["tile_layer_warm_resident_items"]),
            tile_layer_evicted_near_items=int(timing["tile_layer_evicted_near_items"]),
            tile_layer_capacity_warning=str(timing["tile_layer_capacity_warning"]),
            tile_layer_lod_level=int(timing["tile_layer_lod_level"]),
            tile_layer_lod_factor=int(timing["tile_layer_lod_factor"]),
            tile_layer_source_texels_per_pixel=float(timing["tile_layer_source_texels_per_pixel"]),
            tile_layer_gutter_pixels=int(timing["tile_layer_gutter_pixels"]),
            tile_layer_mipmap_updates=int(timing["tile_layer_mipmap_updates"]),
            tile_layer_mipmap_available=bool(timing["tile_layer_mipmap_available"]),
            tile_layer_complex_texture_uploads=int(timing["tile_layer_complex_texture_uploads"]),
            tile_layer_shader_uniform_updates=int(timing["tile_layer_shader_uniform_updates"]),
            cpu_complex_prep_ms=float(timing["cpu_complex_prep_ms"]),
        )
        self._upload_timing = None

    def lastImageUploadTiming(self) -> ImageUploadTiming:
        return self._last_upload_timing

    def setGuiCallbackObserver(self, observer) -> None:
        self._gui_callback_observer = observer if callable(observer) else None

    def setGuiCallbackBudgetProvider(self, provider) -> None:
        self._gui_callback_budget_provider = provider if callable(provider) else None

    def setBackgroundTaskSubmitter(self, submitter) -> None:
        self._background_task_submitter = submitter if callable(submitter) else None

    def setLevelPresentationChangeHandler(self, handler) -> None:
        self._level_presentation_change_handler = handler if callable(handler) else None

    def _submit_background_task(self, fn, *, on_done, key):
        submitter = getattr(self, "_background_task_submitter", None)
        if callable(submitter):
            return submitter(fn, on_done=on_done, key=key)
        return None

    def _gui_callback_budget(
        self,
        channel: str,
        *,
        interactive: bool,
        work_class: str,
        item_cap: int | None = None,
        byte_cap: int | None = None,
    ) -> GuiCallbackBudget:
        provider = getattr(self, "_gui_callback_budget_provider", None)
        decision = provider(channel, interactive=interactive) if callable(provider) else None
        return GuiCallbackBudget.for_decision(
            channel,
            decision,
            interactive=interactive,
            work_class=work_class,
            backend=self.capabilities.name,
            item_cap=item_cap,
            byte_cap=byte_cap,
        )

    def _record_gui_budget(self, budget: GuiCallbackBudget) -> None:
        observation = budget.observation()
        if observation.processed_items <= 0 and observation.elapsed_ms < observation.warning_ms:
            return
        observer = getattr(self, "_gui_callback_observer", None)
        if callable(observer):
            observer(observation)

    def _record_gui_callback_observation(
        self,
        *,
        channel: str,
        work_class: str,
        elapsed_ms: float,
        item_count: int = 1,
        byte_count: int = 0,
    ) -> None:
        observer = getattr(self, "_gui_callback_observer", None)
        if not callable(observer):
            return
        observer(
            GuiCallbackObservation(
                channel=str(channel),
                work_class=str(work_class),
                backend=self.capabilities.name,
                target_ms=8.0,
                warning_ms=16.0,
                item_cap=max(1, int(item_count)),
                byte_cap=max(0, int(byte_count)),
                elapsed_ms=max(0.0, float(elapsed_ms)),
                processed_items=max(1, int(item_count)),
                processed_bytes=max(0, int(byte_count)),
            )
        )

    def closeEvent(self, event) -> None:
        self._cancel_interaction("widget-close")
        self.teardown_surface()
        super().closeEvent(event)

    def _bind_histogram_item(self, item) -> None:
        adapter = getattr(self, "_histogram_adapter", None)
        if adapter is None:
            return
        facts = adapter.bind_image_item(item)
        self._record_upload_timing("histogram_bind_ms", facts.elapsed_ms)

    def _refresh_histogram_plot(self, *, auto_level: bool = False) -> None:
        start = perf_counter()
        if self._histogram_display_controller is None or not self._histogram_display_controller.refresh_histogram_plot(auto_level=bool(auto_level)):
            self.histogram.item.imageChanged(autoLevel=bool(auto_level))
        elapsed_ms = (perf_counter() - start) * 1000.0
        self._record_upload_timing("histogram_recompute_ms", elapsed_ms)
        source = self.histogramPlotSource if self.histogramPlotSource is not None else self.histogramSource
        self._record_gui_callback_observation(
            channel="histogram_refresh",
            work_class="histogram_plot",
            elapsed_ms=elapsed_ms,
            item_count=1,
            byte_count=0 if source is None else int(getattr(np.asarray(source), "nbytes", 0) or 0),
        )

    def _set_image_item_data(self, item, data, levels, *, role: str, emit_histogram_change: bool = True) -> bool:
        previous = getattr(item, "image", None)
        same_object = previous is data
        if not same_object and isinstance(previous, np.ndarray) and isinstance(data, np.ndarray):
            same_object = (
                tuple(previous.shape) == tuple(data.shape)
                and np.dtype(previous.dtype) == np.dtype(data.dtype)
                and np.shares_memory(previous, data)
            )
        start = perf_counter()
        array = np.asarray(data)
        image_kwargs = {}
        if str(role) == "visible" and array.ndim == 2:
            image_kwargs["lut"] = self._display_colormap_lut
        if same_object:
            item.setImage(None, autoLevels=False, levels=levels, **image_kwargs)
        else:
            item.setImage(ensure_imageitem_array(data), autoLevels=False, levels=levels, **image_kwargs)
        elapsed = (perf_counter() - start) * 1000.0
        timing = self._upload_timing
        if str(role) == "histogram":
            self._record_upload_timing("histogram_upload_ms", elapsed)
            if timing is not None:
                timing["histogram_bytes"] = int(timing["histogram_bytes"]) + int(array.nbytes)
                timing["histogram_pixels"] = int(timing["histogram_pixels"]) + int(np.prod(array.shape[:2]))
        else:
            self._record_upload_timing("visible_upload_ms", elapsed)
            if timing is not None:
                timing["visible_bytes"] = int(timing["visible_bytes"]) + int(array.nbytes)
                timing["visible_pixels"] = int(timing["visible_pixels"]) + int(np.prod(array.shape[:2]))
                timing["fast_same_object"] = bool(timing["fast_same_object"] or same_object)
        adapter = getattr(self, "_histogram_adapter", None)
        if emit_histogram_change and adapter is not None and adapter.is_bound_item(item):
            self._refresh_histogram_plot(auto_level=False)
        return same_object

    def montageDisplayMode(self) -> str:
        return str(self._montage_display_mode)

    def clearMontageTileLayer(self) -> None:
        if self._montage_tile_layer is not None:
            self._montage_tile_layer.clear()
        self._montage_tile_layer_histogram_key = None
        self._montage_display_mode = "canvas"
        self.imageItem.setVisible(True)

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
        tile_delta=None,
        frame_plan=None,
    ) -> None:
        if geometry is None or getattr(geometry, "montage", None) is None:
            raise ValueError("tile-layer presentation requires montage geometry")
        self._cancel_active_capture_for_frame_replacement()
        self._apply_tile_layer_presentation(
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
            tile_delta=tile_delta,
            frame_plan=frame_plan,
        )

    def _apply_tile_layer_presentation(
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
        tile_delta=None,
        frame_plan=None,
    ) -> None:
        self._start_upload_timing("tile_layer")
        applying = self._applying_presentation
        self._applying_presentation = True
        try:
            self.image = img
            loading_only = _is_tiled_loading_only_commit(
                montage_tile_payloads,
                histogramData=histogramData,
                histogramPlotData=histogramPlotData,
            )
            if not loading_only:
                self.histogramSource = histogramData
                self.histogramPlotSource = histogramPlotData
                self.setHistogramDataBounds(histogramRange)
            self._montage_display_mode = "tile_layer"
            self.imageItem.setVisible(False)
            stats = self._update_montage_tile_layer_items(
                img,
                histogramData=histogramData,
                geometry=geometry,
                levels=levels,
                rgb_already_windowed=rgb_already_windowed,
                montage_dirty_tiles=montage_dirty_tiles,
                montage_tile_source_ids=montage_tile_source_ids,
                montage_tile_payloads=montage_tile_payloads,
                tile_delta=tile_delta,
                frame_plan=frame_plan,
            )
            self._record_tile_layer_stats(stats)
            histogram_key = self._tile_layer_histogram_key(
                histogramData,
                histogramPlotData,
                levels=levels,
                histogramRange=histogramRange,
            )
            skip_histogram_upload = (
                loading_only
                or montage_dirty_tiles == ()
                and self._montage_tile_layer_histogram_key == histogram_key
                and getattr(self.histogramImageItem, "image", None) is not None
            )
            if not skip_histogram_upload:
                plot_data = self._histogram_plot_data(histogramData)
            else:
                plot_data = None
            if plot_data is not None:
                self._bind_histogram_item(self.histogramImageItem)
                self._set_image_item_data(
                    self.histogramImageItem,
                    plot_data,
                    self._histogram_levels_for_display(levels),
                    role="histogram",
                )
            if not loading_only:
                self._montage_tile_layer_histogram_key = histogram_key
                self._sync_display_levels(float(levels[0]), float(levels[1]), update_image=False, emit_user=False)
                self.histogram.setHistogramRange(float(histogramRange[0]), float(histogramRange[1]))
            profile_start = perf_counter()
            self._update_profile_line_bounds()
            self._record_upload_timing("profile_bounds_ms", (perf_counter() - profile_start) * 1000.0)
            self._updateAspectRatio()
            self._apply_viewport_policy(
                tuple(img.shape[:2]),
                viewport_policy,
                image_origin=(geometry.montage_origin_x, geometry.montage_origin_y),
                content_rect=_viewport_rect_for_geometry(
                    geometry,
                    img.shape[:2],
                    (geometry.montage_origin_x, geometry.montage_origin_y),
                ),
            )
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
        frame_plan=None,
    ) -> None:
        """Commit a first-class tiled presentation through this backend.

        A small placeholder remains inside the widget shell because shared
        image-view bookkeeping still expects ``self.image.shape``.  Core
        presentation state and renderer upload decisions use the typed tile
        state and delta, not placeholder pixels.
        """

        del shader_mapping  # PyQtGraph receives already materialized display pixels.
        tile_payloads = tile_state.active_payloads(tile_delta)
        dirty_tiles = None if tile_delta.force_refresh else tuple(tile_delta.upserts)
        placeholder = _tiled_montage_placeholder(geometry.display_shape, tile_payloads)
        stats = self._apply_tile_layer_presentation(
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
            tile_delta=tile_delta,
            frame_plan=frame_plan,
        )
        return _tile_commit_report(tile_payloads, tile_delta, stats)

    def _update_montage_tile_layer_items(self, img, *, histogramData, geometry, levels, rgb_already_windowed: bool, montage_dirty_tiles, montage_tile_source_ids, montage_tile_payloads=None, tile_delta=None, frame_plan=None) -> TileLayerUpdateStats:
        if self._montage_tile_layer is None:
            return TileLayerUpdateStats()
        return self._montage_tile_layer.update_presentation(
            img,
            histogram_data=histogramData,
            geometry=geometry,
            levels=levels,
            rgb_already_windowed=rgb_already_windowed,
            dirty_tiles=montage_dirty_tiles,
            tile_source_ids=montage_tile_source_ids,
            tile_payloads=montage_tile_payloads,
            tile_delta=tile_delta,
            frame_plan=frame_plan,
        )

    def _record_tile_layer_stats(self, stats: TileLayerUpdateStats) -> None:
        timing = self._upload_timing
        if timing is None:
            return
        timing["tile_layer_visible_items"] = int(stats.visible_items)
        timing["tile_layer_items_created"] = int(getattr(stats, "items_created", 0))
        timing["tile_layer_items_updated"] = int(stats.items_updated)
        timing["tile_layer_items_skipped"] = int(stats.items_skipped)
        timing["tile_layer_rgb_window_tiles"] = int(stats.rgb_window_tiles)
        timing["tile_layer_image_replacements"] = int(getattr(stats, "image_replacements", 0))
        timing["tile_layer_existing_items_shown"] = int(getattr(stats, "existing_items_shown", 0))
        timing["tile_layer_relocated_tiles"] = int(getattr(stats, "relocated_tiles", 0))
        timing["tile_layer_resident_items"] = int(stats.resident_items)
        timing["tile_layer_storage_capacity"] = int(stats.storage_capacity)
        timing["tile_layer_storage_rebuilds"] = int(stats.storage_rebuilds)
        timing["tile_layer_storage_evictions"] = int(stats.storage_evictions)
        timing["tile_layer_texture_uploads"] = int(stats.texture_uploads)
        timing["tile_layer_texture_upload_bytes"] = int(stats.texture_upload_bytes)
        timing["tile_layer_texture_prepare_ms"] = float(getattr(stats, "texture_prepare_ms", 0.0) or 0.0)
        timing["tile_layer_texture_submit_ms"] = float(getattr(stats, "texture_submit_ms", 0.0) or 0.0)
        timing["tile_layer_vertex_uploads"] = int(stats.vertex_uploads)
        timing["tile_layer_level_updates"] = int(stats.level_updates)
        timing["tile_layer_level_update_pending_items"] = int(getattr(stats, "level_update_pending_items", 0))
        timing["tile_layer_estimated_gpu_bytes"] = int(stats.estimated_gpu_bytes)
        timing["tile_layer_cpu_shadow_bytes"] = int(stats.cpu_shadow_bytes)
        timing["tile_layer_page_count"] = int(stats.page_count)
        timing["tile_layer_active_pages"] = int(stats.active_pages)
        timing["tile_layer_device_max_texture_size"] = int(stats.device_max_texture_size)
        timing["tile_layer_budget_bytes"] = int(stats.budget_bytes)
        timing["tile_layer_near_resident_items"] = int(stats.near_resident_items)
        timing["tile_layer_warm_resident_items"] = int(stats.warm_resident_items)
        timing["tile_layer_evicted_near_items"] = int(stats.evicted_near_items)
        timing["tile_layer_capacity_warning"] = str(stats.capacity_warning)
        timing["tile_layer_lod_level"] = int(stats.lod_level)
        timing["tile_layer_lod_factor"] = int(stats.lod_factor)
        timing["tile_layer_source_texels_per_pixel"] = float(stats.source_texels_per_pixel)
        timing["tile_layer_gutter_pixels"] = int(stats.gutter_pixels)
        timing["tile_layer_mipmap_updates"] = int(stats.mipmap_updates)
        timing["tile_layer_mipmap_available"] = bool(stats.mipmap_available)
        timing["tile_layer_complex_texture_uploads"] = int(stats.complex_texture_uploads)
        timing["tile_layer_shader_uniform_updates"] = int(stats.shader_uniform_updates)
        elapsed_ms = float(getattr(stats, "upload_ms", 0.0) or 0.0)
        observed_items = (
            int(getattr(stats, "items_created", 0))
            + int(stats.items_updated)
            + int(getattr(stats, "existing_items_shown", 0))
            + int(getattr(stats, "relocated_tiles", 0))
            + int(getattr(stats, "level_updates", 0))
        )
        if elapsed_ms > 0.0 or observed_items > 0:
            self._record_gui_callback_observation(
                channel="tile_layer_commit",
                work_class="presentation_upsert",
                elapsed_ms=elapsed_ms,
                item_count=max(1, observed_items),
                byte_count=int(getattr(stats, "texture_upload_bytes", 0) or 0),
            )

    def _tile_layer_histogram_key(self, histogramData, histogramPlotData, *, levels, histogramRange):
        source = histogramPlotData if histogramPlotData is not None else histogramData
        return (
            id(source),
            tuple(np.shape(source)) if source is not None else None,
            None if source is None else str(np.asarray(source).dtype),
            (float(levels[0]), float(levels[1])),
            (float(histogramRange[0]), float(histogramRange[1])),
        )

    def _update_montage_tile_levels(self, levels) -> TileLayerUpdateStats:
        if self._montage_tile_layer is not None:
            return self._montage_tile_layer.update_levels(
                levels,
                image=self.image,
                histogram_data=self.histogramSource,
            )
        return TileLayerUpdateStats()
        
    def setImage(self, img, autoRange=None, autoLevels=True, levels=None,
                 pos=None, scale=None, transform=None, autoHistogramRange=True,
                 histogramData=None, histogramPlotData=None, viewport_policy=ViewportPolicy.PRESERVE,
                 rgb_already_windowed: bool = False, image_origin: tuple[float, float] = (0.0, 0.0),
                 viewport_content_rect=None, shader_mapping=None, texture_kind=None,
                 semantic_data: np.ndarray | None = None, lod=None):
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
        del shader_mapping, texture_kind, semantic_data, lod
        if not isinstance(img, np.ndarray):
            raise TypeError("Image must be a numpy array")
        viewport_policy = coerce_viewport_policy(viewport_policy, autoRange)
            
        is_rgb = self._is_rgb_image(img)
        if img.ndim != 2 and not is_rgb:
            raise ValueError("ImageView2D only supports 2D scalar or RGB images")

        self._cancel_active_capture_for_frame_replacement()
        self._start_upload_timing("full")
        previous_shape = None if self.image is None else tuple(self.image.shape[:2])
        try:
            self.clearMontageTileLayer()
            self.image = img
            self.imageDisp = None
            self.histogramSource = histogramData
            self.histogramPlotSource = histogramPlotData
            display_levels = None
            if levels is not None:
                if isinstance(levels, (list, tuple)) and len(levels) == 2:
                    display_levels = (float(levels[0]), float(levels[1]))
                else:
                    low, high = levels
                    display_levels = (float(low), float(high))

            applying = self._applying_presentation
            self._applying_presentation = True
            try:
                # Update the image display
                self.updateImage(
                    autoHistogramRange=autoHistogramRange,
                    displayLevels=display_levels,
                    rgb_already_windowed=rgb_already_windowed,
                )
                profile_start = perf_counter()
                self._update_profile_line_bounds()
                self._record_upload_timing("profile_bounds_ms", (perf_counter() - profile_start) * 1000.0)

                # Set levels
                self.histogram.setVisible(True)
                if levels is None and autoLevels:
                    self.autoLevels()
                elif levels is not None:
                    if isinstance(levels, (list, tuple)) and len(levels) == 2:
                        self._displayLevels = (float(levels[0]), float(levels[1]))
                    else:
                        low, high = levels
                        self._displayLevels = (float(low), float(high))
            finally:
                self._applying_presentation = applying
            
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
            self.imageItem.setPos(float(image_origin[0]), float(image_origin[1]))

            # Update aspect ratio based on display mode
            self._updateAspectRatio()

            self._apply_viewport_policy(
                tuple(img.shape[:2]),
                viewport_policy,
                image_origin=image_origin,
                content_rect=viewport_content_rect,
            )
        finally:
            self._finish_upload_timing()

    def setImagePresentation(
        self,
        img: np.ndarray,
        *,
        histogramData: np.ndarray | None,
        histogramPlotData: np.ndarray | None = None,
        levels: tuple[float, float],
        histogramRange: tuple[float, float],
        viewport_policy=ViewportPolicy.PRESERVE,
        rgb_already_windowed: bool = False,
        image_origin: tuple[float, float] = (0.0, 0.0),
        geometry=None,
        shader_mapping=None,
        texture_kind=None,
        semantic_data: np.ndarray | None = None,
        lod=None,
    ) -> None:
        """Set fully decided image pixels, levels, and histogram range."""
        self.setImage(
            img,
            autoLevels=False,
            levels=levels,
            histogramData=histogramData,
            histogramPlotData=histogramPlotData,
            autoHistogramRange=False,
            viewport_policy=viewport_policy,
            rgb_already_windowed=rgb_already_windowed,
            image_origin=image_origin,
            viewport_content_rect=_viewport_rect_for_geometry(geometry, img.shape[:2], image_origin),
            shader_mapping=shader_mapping,
            texture_kind=texture_kind,
            semantic_data=semantic_data,
            lod=lod,
        )
        self.setHistogramDataBounds(histogramRange)
        self.setHistogramRange(histogramRange[0], histogramRange[1])

    def updateImagePresentationFast(
        self,
        img: np.ndarray,
        *,
        histogramData: np.ndarray | None,
        histogramPlotData: np.ndarray | None = None,
        levels: tuple[float, float],
        histogramRange: tuple[float, float],
        rgb_already_windowed: bool = False,
        image_origin: tuple[float, float] = (0.0, 0.0),
        geometry=None,
        shader_mapping=None,
        texture_kind=None,
        semantic_data: np.ndarray | None = None,
        lod=None,
    ) -> None:
        """Fast same-shape update with explicit presentation state."""
        self.updateImageDataFast(
            img,
            histogramData=histogramData,
            histogramPlotData=histogramPlotData,
            levels=levels,
            histogramRange=histogramRange,
            rgb_already_windowed=rgb_already_windowed,
            image_origin=image_origin,
            viewport_content_rect=_viewport_rect_for_geometry(geometry, img.shape[:2], image_origin),
            shader_mapping=shader_mapping,
            texture_kind=texture_kind,
            semantic_data=semantic_data,
            lod=lod,
        )

    @property
    def widget(self):
        return self

    @property
    def capabilities(self):
        return self.rendering_capabilities

    def current_raster_shape(self) -> tuple[int, int] | None:
        image = getattr(self, "image", None)
        if image is None:
            return None
        shape = tuple(np.shape(image))
        if len(shape) < 2:
            return None
        return (int(shape[0]), int(shape[1]))

    def present_raster(self, presentation, *, mode: RasterCommitMode) -> None:
        mode = RasterCommitMode(mode)
        if mode is RasterCommitMode.FULL:
            self.setImagePresentation(
                presentation.data,
                histogramData=presentation.histogram_data,
                histogramPlotData=presentation.histogram_plot_data,
                levels=presentation.levels,
                histogramRange=presentation.histogram_range,
                viewport_policy=presentation.viewport_policy,
                rgb_already_windowed=presentation.rgb_already_windowed,
                image_origin=_image_origin(presentation.geometry),
                geometry=presentation.geometry,
                shader_mapping=presentation.shader_mapping,
                texture_kind=presentation.texture_kind,
                semantic_data=presentation.semantic_data,
                lod=presentation.lod,
            )
            return
        if mode is RasterCommitMode.FAST:
            self.updateImagePresentationFast(
                presentation.data,
                histogramData=presentation.histogram_data,
                histogramPlotData=presentation.histogram_plot_data,
                levels=presentation.levels,
                histogramRange=presentation.histogram_range,
                rgb_already_windowed=presentation.rgb_already_windowed,
                image_origin=_image_origin(presentation.geometry),
                geometry=presentation.geometry,
                shader_mapping=presentation.shader_mapping,
                texture_kind=presentation.texture_kind,
                semantic_data=presentation.semantic_data,
                lod=presentation.lod,
            )
            return
        if mode is RasterCommitMode.TILE_LAYER:
            self.setMontageTileLayerPresentation(
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
            return
        raise ValueError(f"unsupported raster commit mode: {mode}")

    def present_tiled(self, presentation):
        return self.setTiledMontagePresentation(
            geometry=presentation.geometry,
            tile_state=presentation.tile_state,
            tile_delta=presentation.tile_delta,
            histogramPlotData=presentation.histogram_plot_data,
            levels=presentation.levels,
            histogramRange=presentation.histogram_range,
            viewport_policy=presentation.viewport_policy,
            rgb_already_windowed=presentation.rgb_already_windowed,
            shader_mapping=presentation.shader_mapping,
            tile_residency_budget_bytes=presentation.tile_residency_budget_bytes,
            frame_plan=presentation.frame_plan,
        )

    def set_profile_bounds(self, bounds: tuple[float, float, float, float]) -> None:
        self.setProfileMarkerBoundsRect(bounds)

    def apply_camera(
        self,
        image_shape: tuple[int, int],
        viewport_policy,
        *,
        image_origin: tuple[float, float] = (0.0, 0.0),
        content_rect: tuple[float, float, float, float] | None = None,
    ) -> None:
        self._apply_viewport_policy(
            tuple(int(value) for value in image_shape[:2]),
            viewport_policy,
            image_origin=image_origin,
            content_rect=content_rect,
        )

    def map_scene_to_overlay(self, scene_pos):
        return self._map_scene_to_display_overlay(scene_pos)

    def current_viewport_rect(self) -> tuple[float, float, float, float] | None:
        return self._current_image_viewport_rect()

    def presentation_diagnostics(self) -> dict[str, object]:
        timing = self.lastImageUploadTiming()
        return {
            "backend": self.capabilities.name,
            "mode": str(getattr(timing, "mode", "")),
            "interaction_event_owner": self.interaction_event_owner(),
            "native_pointer_interaction": bool(self.capabilities.native_pointer_interaction),
            "montage_display_mode": self.montageDisplayMode(),
            "last_reset_reason": str(getattr(self, "_last_surface_reset_reason", "")),
        }

    def interaction_event_owner(self) -> str:
        return "shared-controller"

    def reset_surface(self, reason: str) -> None:
        self._cancel_interaction("surface-reset")
        self._last_surface_reset_reason = str(reason)
        self.clearMontageTileLayer()
        self.clearMontageTileOverlays()
        overlay = getattr(self, "_evaluation_overlay", None)
        if overlay is not None:
            overlay.hide()
        self.setImageStale(False)

    def teardown_surface(self) -> None:
        if self._surface_teardown_done:
            return
        self._cancel_interaction("surface-teardown")
        self._surface_teardown_done = True
        preview = getattr(self, "_histogram_preview_controller", None)
        if preview is not None:
            preview.cancel()
        display = getattr(self, "_histogram_display_controller", None)
        if display is not None:
            display.cancel()
        app = getattr(self, "_interaction_application", None)
        if app is not None:
            try:
                app.applicationStateChanged.disconnect(self._on_application_state_changed)
            except Exception:
                pass
            self._interaction_application = None
        self.clearMontageTileOverlays()

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
        viewport_content_rect=None,
        shader_mapping=None,
        texture_kind=None,
        semantic_data: np.ndarray | None = None,
        lod=None,
    ) -> None:
        """Replace same-shape pixel data without recomputing levels or viewport."""
        del shader_mapping, texture_kind, semantic_data, lod
        if not isinstance(img, np.ndarray):
            raise TypeError("Image must be a numpy array")
        if self.image is None:
            raise RuntimeError("fast image update requires an existing image")
        if tuple(img.shape[:2]) != tuple(self.image.shape[:2]):
            raise ValueError("fast image update requires the same display shape")
        is_rgb = self._is_rgb_image(img)
        if img.ndim != 2 and not is_rgb:
            raise ValueError("ImageView2D only supports 2D scalar or RGB images")

        self._cancel_active_capture_for_frame_replacement()
        self._start_upload_timing("fast")
        applying = self._applying_presentation
        self._applying_presentation = True
        try:
            self.clearMontageTileLayer()
            self.image = img
            self.histogramSource = histogramData
            self.histogramPlotSource = histogramPlotData
            if histogramRange is not None:
                self.setHistogramDataBounds(histogramRange)

            if is_rgb:
                self._bind_histogram_item(self.histogramImageItem)
                if rgb_already_windowed:
                    self._rgbBaseImage = None
                    self.imageDisp = img[..., :3]
                else:
                    rgb_start = perf_counter()
                    self._rgbBaseImage = img[..., :3].astype(np.float32, copy=False)
                    self.imageDisp = self._rgb_display_for_levels(levels)
                    self._record_upload_timing("rgb_window_ms", (perf_counter() - rgb_start) * 1000.0)
                self._set_image_item_data(self.imageItem, self.imageDisp, (0, 255), role="visible", emit_histogram_change=False)
                plot_data = self._histogram_plot_data(histogramData)
                if plot_data is not None:
                    # Fast progressive updates must not scan the histogram source or
                    # replace missing/NaN data with zeros; missing tile state is
                    # represented separately by the montage overlay.
                    self._set_image_item_data(
                        self.histogramImageItem,
                        plot_data,
                        self._histogram_levels_for_display(levels),
                        role="histogram",
                    )
            else:
                self._rgbBaseImage = None
                self.imageDisp = img
                image_levels = self._histogram_levels_for_display(levels)
                self._set_image_item_data(self.imageItem, self.imageDisp, image_levels, role="visible", emit_histogram_change=False)
                plot_data = self._histogram_plot_data(histogramData)
                if plot_data is not None:
                    self._bind_histogram_item(self.histogramImageItem)
                    self._set_image_item_data(self.histogramImageItem, plot_data, image_levels, role="histogram")
                else:
                    self._bind_histogram_item(self.imageItem)
                    self._refresh_histogram_plot(auto_level=False)
            if levels is not None:
                self._sync_display_levels(float(levels[0]), float(levels[1]), update_image=False, emit_user=False)
            if histogramRange is not None:
                self.histogram.setHistogramRange(float(histogramRange[0]), float(histogramRange[1]))
            self.imageItem.setPos(float(image_origin[0]), float(image_origin[1]))
            self._refresh_viewport_content_rect(tuple(img.shape[:2]), viewport_content_rect, image_origin=image_origin)
            profile_start = perf_counter()
            self._update_profile_line_bounds()
            self._record_upload_timing("profile_bounds_ms", (perf_counter() - profile_start) * 1000.0)
        finally:
            self._applying_presentation = applying
            self._finish_upload_timing()
            
    def updateImage(self, autoHistogramRange=True, displayLevels=None, *, rgb_already_windowed: bool = False):
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
        histogram_plot_data = self._histogram_plot_data(histogram_data)

        # Calculate data bounds independently from display/LUT levels.
        self._updateImageLevels(histogram_data)
        histogram_levels = (self.levelMin, self.levelMax) if displayLevels is None else (float(displayLevels[0]), float(displayLevels[1]))
        
        # Set the image data
        if is_rgb:
            self._bind_histogram_item(self.histogramImageItem)
            if rgb_already_windowed:
                self._rgbBaseImage = None
                self.imageDisp = self.imageDisp[..., :3]
            else:
                rgb_start = perf_counter()
                self._rgbBaseImage = self.imageDisp[..., :3].astype(np.float32, copy=False)
                self.imageDisp = self._rgb_display_for_levels(histogram_levels)
                self._record_upload_timing("rgb_window_ms", (perf_counter() - rgb_start) * 1000.0)
            self._set_image_item_data(self.imageItem, self.imageDisp, (0, 255), role="visible", emit_histogram_change=False)
            histogram_display = histogram_plot_data if histogram_plot_data is not None else histogram_data
            if finite_bounds(histogram_display) is None:
                histogram_display = np.zeros_like(np.asarray(histogram_data), dtype=float)
            self._set_image_item_data(self.histogramImageItem, histogram_display, histogram_levels, role="histogram")
        else:
            self._set_image_item_data(self.imageItem, self.imageDisp, histogram_levels, role="visible", emit_histogram_change=False)
            if histogram_plot_data is not None:
                self._bind_histogram_item(self.histogramImageItem)
                self._set_image_item_data(self.histogramImageItem, histogram_plot_data, histogram_levels, role="histogram")
            elif self.histogramSource is not None:
                self._bind_histogram_item(self.histogramImageItem)
                self._set_image_item_data(
                    self.histogramImageItem,
                    histogram_data,
                    histogram_levels,
                    role="histogram",
                )
            else:
                self._bind_histogram_item(self.imageItem)
                self._refresh_histogram_plot(auto_level=False)
        
        # Update histogram range if requested
        if autoHistogramRange:
            low, high = self.getHistogramDataBounds() or (self.levelMin, self.levelMax)
            self.histogram.setHistogramRange(low, high)
            
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
            bounds = finite_bounds(image)
            if bounds is None:
                self.levelMin = 0.0
                self.levelMax = 1.0
            else:
                self.levelMin, self.levelMax = bounds
            self._histogramDataBounds = (float(self.levelMin), float(self.levelMax))

    def _is_rgb_image(self, img):
        return isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] in (3, 4)

    def _histogram_data(self, img):
        if self._is_rgb_image(img):
            rgb = img[..., :3].astype(np.float32, copy=False)
            return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return img

    def _histogram_plot_data(self, fallback):
        source = self.histogramPlotSource
        if source is None:
            return fallback
        data = np.asarray(source)
        if data.size == 0:
            return fallback
        if data.ndim == 1:
            width = max(1, int(np.ceil(np.sqrt(data.size))))
            height = int(np.ceil(data.size / width))
            padded = np.full(height * width, np.nan, dtype=data.dtype)
            padded[: data.size] = data
            return padded.reshape(height, width)
        return data

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

        return rgb_display_for_levels(self._rgbBaseImage, histogram_data, levels)

    def _on_histogram_levels_changed(self, *args):
        if self._applying_presentation:
            return
        if self._histogram_preview_controller is not None:
            self._histogram_preview_controller.schedule_from_widget()

    def setHistogramPreviewInterval(self, interval_ms: int) -> None:
        if self._histogram_preview_controller is not None:
            self._histogram_preview_controller.interval_ms = max(1, int(interval_ms))
        self._histogram_retry_interval_ms = max(1, int(interval_ms))

    def _apply_histogram_preview_levels(self, levels, *, final: bool = False) -> None:
        levels = (float(levels[0]), float(levels[1]))
        preview = getattr(self, "_histogram_preview_controller", None)
        if preview is not None:
            preview.last_applied_levels = levels
        started_timing = self._upload_timing is None
        if started_timing:
            self._start_upload_timing("level_preview")
        try:
            self._displayLevels = levels
            if self._montage_display_mode == "tile_layer":
                handler = getattr(self, "_level_presentation_change_handler", None)
                if callable(handler) and bool(handler(levels, final=bool(final))):
                    return
                stats = self._update_montage_tile_levels(levels)
                self._record_tile_layer_stats(stats)
                return
            if self._rgbBaseImage is None or self.histogramSource is None:
                if self.imageItem is not None and self.imageDisp is not None and not self._is_rgb_image(self.image):
                    try:
                        self.imageItem.setLevels(levels)
                    except Exception:
                        pass
                return
            rgb_start = perf_counter()
            self.imageDisp = self._rgb_display_for_levels(levels)
            self._record_upload_timing("rgb_window_ms", (perf_counter() - rgb_start) * 1000.0)
            self._set_image_item_data(self.imageItem, self.imageDisp, (0, 255), role="visible", emit_histogram_change=False)
        finally:
            if started_timing:
                self._finish_upload_timing()

    def _finish_histogram_preview_levels(self, levels) -> None:
        """Finalize an already-applied preview without repeating pixel work.

        PyQtGraph may still have deferred CPU tile redraws for this generation;
        the presentation handler is therefore allowed to force-drain the
        existing job.  Raster previews and settled tiled generations require no
        second image conversion merely because the mouse button was released.
        """

        if self._montage_display_mode not in {"tile_layer", "vispy_tile_layer"}:
            return
        handler = getattr(self, "_level_presentation_change_handler", None)
        if callable(handler):
            handler((float(levels[0]), float(levels[1])), final=True)

    def cancelHistogramLevelInteraction(self) -> None:
        if self._histogram_preview_controller is not None:
            self._histogram_preview_controller.cancel()
        if self._histogram_display_controller is not None:
            self._histogram_display_controller.cancel_manual_edit()
                
    def autoLevels(self):
        """Automatically set the histogram levels based on image data"""
        if self.imageDisp is not None:
            bounds = self.getHistogramDataBounds()
            if bounds is None:
                if self._rgbBaseImage is not None:
                    image = self.histogramSource
                    if image is None:
                        image = self._histogram_data(self.imageDisp)
                    self._updateImageLevels(image)
                else:
                    self._updateImageLevels()
                bounds = self.getHistogramDataBounds() or (self.levelMin, self.levelMax)
            self._sync_display_levels(bounds[0], bounds[1], update_image=True, emit_user=False)
                
    def setLevels(self, min_level, max_level):
        """Set levels as an explicit user action.

        Programmatic render/presentation paths must use _apply_display_levels
        with emit_user=False so automatic commits never become user-locked.
        """
        self._sync_display_levels(min_level, max_level, update_image=True, emit_user=True)

    def _apply_display_levels(self, min_level, max_level, *, emit_user: bool) -> None:
        self._sync_display_levels(min_level, max_level, update_image=True, emit_user=emit_user)

    def _sync_display_levels(self, min_level, max_level, *, update_image: bool, emit_user: bool) -> None:
        start = perf_counter()
        low = float(min_level)
        high = float(max_level)
        previous = self._displayLevels
        same_levels = (
            previous is not None
            and float(previous[0]) == low
            and float(previous[1]) == high
        )
        self._displayLevels = (low, high)
        if self._histogram_preview_controller is not None and self._applying_presentation:
            self._histogram_preview_controller.cancel()
        if not same_levels:
            applying = self._applying_presentation
            self._applying_presentation = True
            try:
                self.histogram.setLevels(low, high)
            finally:
                self._applying_presentation = applying
        if update_image:
            self._apply_histogram_preview_levels((low, high), final=True)
        self._record_upload_timing("level_sync_ms", (perf_counter() - start) * 1000.0)
        if emit_user:
            self.userLevelsChanged.emit()

    def _on_histogram_level_change_finished(self, *args):
        if self._applying_presentation:
            return
        if self._histogram_preview_controller is not None:
            self._histogram_preview_controller.finish_from_widget()
            return
        self.userLevelsChanged.emit()
        
    def getLevels(self):
        """Get the current histogram levels"""
        if self._displayLevels is not None:
            return self._displayLevels
        return self.histogram.getLevels()

    def getHistogramDataBounds(self):
        """Get the current histogram source data bounds."""
        if self._histogramDataBounds is None:
            return None
        return (float(self._histogramDataBounds[0]), float(self._histogramDataBounds[1]))

    def setHistogramDataBounds(self, bounds: tuple[float, float] | None) -> None:
        """Set the semantic histogram/data bounds independently from display levels."""
        if bounds is None:
            self._histogramDataBounds = None
            return
        low, high = bounds
        self._histogramDataBounds = (float(low), float(high))
        self.levelMin = float(low)
        self.levelMax = float(high)
        
    def setHistogramRange(self, min_val, max_val):
        """Set the range of the histogram"""
        self.histogram.setHistogramRange(min_val, max_val)

    def _histogram_levels_for_display(self, levels=None):
        if levels is not None:
            return (float(levels[0]), float(levels[1]))
        bounds = self.getHistogramDataBounds()
        if bounds is not None:
            return bounds
        return (float(self.levelMin), float(self.levelMax))
        
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
        self._profile_marker_requested_visible = bool(visible)
        self._profile_marker_updating = True
        try:
            self._profile_handle.setPos(float(x), float(y))
            self._update_profile_line_bounds()
            self._profile_vline.setValue(float(x))
            self._profile_hline.setValue(float(y))
            self._sync_profile_marker_visibility()
            self.view.update()
        finally:
            self._profile_marker_updating = False
        self._last_profile_marker_position = (float(x), float(y))

    def hideProfileMarker(self):
        """Hide the image-space profile marker."""
        self._profile_marker_requested_visible = False
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
        if not self._profile_marker_requested_visible:
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

    def _on_profile_marker_changed(self, part="center"):
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

    def _on_profile_handle_changed(self, part="center"):
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
        self._profile_marker.set_bounds_rect(rect)
        self._update_profile_line_bounds()

    def _clamp_profile_point(self, x, y):
        if self.image is None:
            return (float(x), float(y))
        return self._profile_marker.clamp_point(self.image.shape, x, y)

    def _current_profile_bounds(self):
        return self._profile_marker.current_bounds(self.image.shape)

    def _sync_profile_marker_visibility(self):
        visible = bool(self._profile_marker_requested_visible)
        if visible and self._profile_handle is not None:
            pos = self._profile_handle.pos()
            visible = _point_inside_view_range(self.view.viewRange(), float(pos.x()), float(pos.y()))
        if self._profile_vline is not None:
            self._profile_vline.setVisible(visible)
        if self._profile_hline is not None:
            self._profile_hline.setVisible(visible)
        if self._profile_handle is not None:
            self._profile_handle.setVisible(visible)
        
    def clear(self):
        """Clear the displayed image"""
        self.image = None
        self.imageDisp = None
        self.histogramSource = None
        self.histogramPlotSource = None
        self.imageItem.clear()
        self.hideProfileMarker()

    def valueAtDisplayMapping(self, mapping):
        source = self.histogramSource
        if source is None:
            source = self.image
        if source is None:
            return None
        data = np.asarray(source)
        if self.image is None or tuple(data.shape[:2]) != tuple(self.image.shape[:2]):
            return None
        y_i = int(mapping.canvas_y)
        x_i = int(mapping.canvas_x)
        if y_i < 0 or x_i < 0 or y_i >= data.shape[0] or x_i >= data.shape[1]:
            return None
        return data[y_i, x_i]
        
    def setColorMap(self, colormap):
        """Set one display colormap for the colorbar and all scalar surfaces."""

        lut = normalize_lut_rgb(colormap.getLookupTable(0.0, 1.0, 256, alpha=False))
        self._display_colormap = colormap
        self._display_colormap_lut = lut
        self._display_colormap_key = _array_content_key(lut)
        self.histogram.gradient.setColorMap(colormap)
        image = getattr(self.imageItem, "image", None)
        if image is not None and np.asarray(image).ndim == 2:
            self.imageItem.setLookupTable(lut)
        if self._montage_tile_layer is not None:
            self._montage_tile_layer.set_lookup_table(lut)

    def displayColorMapLookupTable(self) -> np.ndarray:
        """Return the active frame-level RGB lookup table."""

        return self._display_colormap_lut

    def displayColorMapKey(self):
        return self._display_colormap_key
        
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
        self.setFitLocked(True)

    def setFitLocked(self, enabled):
        self.setDisplayMode("fit" if enabled else "square_pixels")
        self.view.setMouseEnabled(x=not bool(enabled), y=not bool(enabled))
        if self.image is not None:
            self._viewport_applying = True
            try:
                self.viewport_controller.set_fit_locked(self.view, bool(enabled))
            finally:
                self._viewport_applying = False

    def oneToOne(self):
        self.setDisplayMode("square_pixels")
        self.view.setMouseEnabled(x=True, y=True)
        if self.image is not None:
            self._viewport_applying = True
            try:
                self.viewport_controller.one_to_one(self.view, self.image.shape[:2], self.graphicsView.viewport().size(), display_rect=self._current_image_viewport_rect())
            finally:
                self._viewport_applying = False
            self._enforce_viewport_constraints()

    def autoWindow(self):
        self.autoLevels()

    def _display_overlay_parent(self):
        """Widget that must remain above the active pixel-rendering surface."""

        return self.graphicsView

    def _map_scene_to_display_overlay(self, scene_pos):
        return self.graphicsView.mapFromScene(scene_pos)

    def setHudWidget(self, widget):
        self._hud_widget = widget
        if widget is not None:
            widget.setParent(self._display_overlay_parent())
            widget.hide()

    def setEvaluationOverlay(self, visible: bool, text: str = ""):
        if self._evaluation_overlay is None:
            overlay = QtWidgets.QLabel(self._display_overlay_parent())
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

    def setMontageTileOverlays(self, overlays):
        overlays = tuple(overlays or ())
        if self._montage_tile_overlay_item is None:
            self._montage_tile_overlay_item = MontageTileOverlayItem()
            self._layer_owner.set_montage_overlay_item(self._montage_tile_overlay_item)
        self._montage_tile_overlay_item.setOverlays(overlays)
        self._montage_tile_overlay_items = [self._montage_tile_overlay_item] if overlays else []

    def clearMontageTileOverlays(self):
        item = getattr(self, "_montage_tile_overlay_item", None)
        if item is not None:
            item.setOverlays(())
        self._montage_tile_overlay_items = []

    def montageTileOverlayCount(self) -> int:
        item = getattr(self, "_montage_tile_overlay_item", None)
        return 0 if item is None else int(item.overlay_count)

    def setRoiInfoText(self, text):
        text = str(text or "")
        if not text:
            if self._roi_info_panel is not None:
                self._roi_info_panel.hide()
            return
        if self._roi_info_panel is None:
            self._roi_info_panel = MovableInfoPanel(self._display_overlay_parent())
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
        local = self._map_scene_to_display_overlay(scene_pos)
        self._hud_widget.show_text_near(text, local)

    def hideHud(self):
        if self._hud_widget is not None:
            self._hud_widget.hide()

    def setInspectionTool(self, tool):
        state = self.interaction_controller.set_tool(tool)
        self._set_roi_drawing_preview(None, ())
        self.sync_interaction_state(state)

    def inspectionTool(self):
        return self.interaction_controller.state.tool

    def interactionState(self):
        return self.interaction_controller.state

    def beginRoiDrawingOnce(self, tool):
        try:
            state = self.interaction_controller.arm_drawing(tool)
        except ValueError:
            return False
        self._set_roi_drawing_preview(state.pending_draw_tool, ())
        self.sync_interaction_state(state)
        return True

    def cancelPendingRoiDrawing(self):
        state = self.interaction_controller.cancel_drawing()
        self._set_roi_drawing_preview(None, ())
        self.sync_interaction_state(state)

    def sync_interaction_state(self, state: DisplayInteractionState) -> None:
        self._apply_interaction_cursor(state.cursor_intent)
        self._sync_pyqtgraph_interaction_visuals(state)

    def _apply_interaction_cursor(self, intent: CursorIntent) -> None:
        cursor_shapes = {
            CursorIntent.CROSSHAIR: QtCore.Qt.CursorShape.CrossCursor,
            CursorIntent.MOVE: QtCore.Qt.CursorShape.SizeAllCursor,
            CursorIntent.OPEN_HAND: QtCore.Qt.CursorShape.OpenHandCursor,
            CursorIntent.CLOSED_HAND: QtCore.Qt.CursorShape.ClosedHandCursor,
            CursorIntent.RESIZE_HORIZONTAL: QtCore.Qt.CursorShape.SizeHorCursor,
            CursorIntent.RESIZE_VERTICAL: QtCore.Qt.CursorShape.SizeVerCursor,
            CursorIntent.RESIZE_DIAGONAL: QtCore.Qt.CursorShape.SizeFDiagCursor,
        }
        shape = cursor_shapes.get(CursorIntent(intent))
        if shape is None:
            self.getView().unsetCursor()
            self.graphicsView.viewport().unsetCursor()
        else:
            self.getView().setCursor(shape)
            self.graphicsView.viewport().setCursor(shape)

    def createRoi(self, kind, *, points=None, rect=None, line_width=1.0, label=None, color=None):
        kind = kind if isinstance(kind, RoiKind) else RoiKind(getattr(kind, "value", kind))
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
            label=label or default_roi_label(kind, self._roi_counter),
            geometry=geometry,
            color=color,
        )
        item = item_for_roi(selection)
        make_item_passive(item)
        self._roi_items[roi_id] = (item, selection)
        self._roi_hit_index.upsert(selection)
        self._layer_owner.add_roi_item(roi_id, item)
        self._sync_roi_item_style(roi_id)
        self.roiCreated.emit(selection)
        return selection

    def removeRoi(self, roi_id):
        roi_id = str(roi_id)
        state = self.interaction_controller.state
        if state.capture is not None and state.capture.kind == "roi" and state.capture.object_id == roi_id:
            self._cancel_interaction("target-removed")
        item_selection = self._roi_items.pop(roi_id, None)
        if item_selection is None:
            return False
        self._roi_hit_index.remove(roi_id)
        if self._highlighted_roi_id == roi_id:
            self._highlighted_roi_id = None
        if self._interaction_visual_roi_id == roi_id:
            self._interaction_visual_roi_id = None
        if state.hover is not None and state.hover.kind == "roi" and state.hover.object_id == roi_id:
            self.sync_interaction_state(self.interaction_controller.clear_hover())
        item, _selection = item_selection
        self._layer_owner.remove_roi_item(roi_id)
        self.roiDeleted.emit(str(roi_id))
        return True

    def clearRois(self):
        for roi_id in tuple(self._roi_items):
            self.removeRoi(roi_id)
        self._roi_hit_index.clear()

    def roiSelections(self):
        return tuple(selection for _item, selection in self._roi_items.values())

    def roiHitCandidates(self, point: tuple[float, float], *, tolerance: float):
        return self._roi_hit_index.candidates(point, tolerance=tolerance)

    def highlightRoi(self, roi_id):
        roi_id = str(roi_id)
        if roi_id not in self._roi_items:
            return False
        previous = self._highlighted_roi_id
        self._highlighted_roi_id = roi_id
        for current_id in {previous, roi_id}:
            if current_id is not None:
                self._sync_roi_item_style(current_id)
        return True

    def _set_roi_geometry(self, roi_id: str, geometry: RoiGeometry, *, emit: bool, sync_item: bool = True) -> bool:
        roi_id = str(roi_id)
        item_selection = self._roi_items.get(str(roi_id))
        if item_selection is None:
            return False
        item, selection = item_selection
        changed = geometry != selection.geometry
        updated = RoiSelection(
            id=selection.id,
            label=selection.label,
            geometry=geometry,
            enabled=selection.enabled,
            color=selection.color,
        )
        self._roi_items[str(roi_id)] = (item, updated)
        self._roi_hit_index.upsert(updated)
        if sync_item:
            self._sync_roi_item_to_geometry(item, geometry)
            self._sync_roi_item_style(roi_id)
        if changed and emit:
            self.roiChanged.emit(str(roi_id), geometry)
        return changed

    def _sync_roi_item_to_geometry(self, item, geometry: RoiGeometry) -> None:
        sync_item_to_roi_geometry(item, geometry)

    def _sync_pyqtgraph_interaction_visuals(self, state: DisplayInteractionState) -> None:
        target = state.capture if state.capture is not None else state.hover
        roi_id = target.object_id if target is not None and target.kind == "roi" else None
        profile_part = target.part if target is not None and target.kind == "profile" else None
        previous_roi_id = self._interaction_visual_roi_id
        self._interaction_visual_roi_id = None if roi_id is None else str(roi_id)
        for current_id in {previous_roi_id, self._interaction_visual_roi_id}:
            if current_id is not None:
                self._sync_roi_item_style(current_id)
        if self._interaction_visual_profile_part != profile_part:
            self._interaction_visual_profile_part = profile_part
            self._sync_profile_interaction_visuals()

    def _sync_roi_item_style(self, roi_id: str) -> None:
        item_selection = self._roi_items.get(str(roi_id))
        if item_selection is None:
            return
        item, selection = item_selection
        highlighted = str(roi_id) == self._highlighted_roi_id
        interactive = str(roi_id) == self._interaction_visual_roi_id
        width = 4.0 if highlighted else 3.25 if interactive else 2.0
        color = tuple(selection.color)
        if interactive:
            color = tuple(min(255, int(value) + 70) for value in color[:3])
        item.setPen(pg.mkPen(color + (255,), width=width))

    def _sync_profile_interaction_visuals(self) -> None:
        marker_pen = (
            pg.mkPen((255, 125, 55, 230), width=2)
            if self._interaction_visual_profile_part
            else pg.mkPen((230, 60, 30, 180), width=1)
        )
        handle_pen = (
            pg.mkPen((255, 255, 255, 240), width=2)
            if self._interaction_visual_profile_part
            else pg.mkPen((230, 60, 30, 220), width=2)
        )
        handle_brush = (
            pg.mkBrush(255, 125, 55, 120)
            if self._interaction_visual_profile_part
            else pg.mkBrush(230, 60, 30, 80)
        )
        for line in (self._profile_vline, self._profile_hline):
            if line is not None and hasattr(line, "setPen"):
                line.setPen(marker_pen)
        if self._profile_handle is not None:
            set_pen = getattr(self._profile_handle, "setPen", None)
            if callable(set_pen):
                set_pen(handle_pen)
            set_brush = getattr(self._profile_handle, "setBrush", None)
            if callable(set_brush):
                set_brush(handle_brush)

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

    def _apply_viewport_policy(self, image_shape, viewport_policy, *, image_origin=(0.0, 0.0), content_rect=None):
        display_rect = _viewport_rect_for_shape(image_shape, image_origin) if content_rect is None else content_rect
        self._viewport_applying = True
        try:
            self.viewport_controller.apply_after_image(
                self.view,
                image_shape,
                self.graphicsView.viewport().size(),
                policy=viewport_policy,
                display_rect=display_rect,
            )
        finally:
            self._viewport_applying = False
        self._enforce_viewport_constraints()

    def _refresh_viewport_content_rect(self, image_shape, content_rect=None, *, image_origin=(0.0, 0.0)) -> None:
        if self.image is None:
            return
        display_rect = _viewport_rect_for_shape(image_shape, image_origin) if content_rect is None else content_rect
        self._viewport_applying = True
        try:
            self.viewport_controller.apply_after_image(
                self.view,
                image_shape,
                self.graphicsView.viewport().size(),
                policy=ViewportPolicy.PRESERVE,
                display_rect=display_rect,
            )
        finally:
            self._viewport_applying = False
        self._enforce_viewport_constraints()

    def _enforce_viewport_constraints(self) -> None:
        if self._viewport_applying or self._viewport_constraining or self.viewport_controller.is_fit_locked():
            if not self._viewport_constraining:
                self._remember_accepted_view_range()
            return
        content_rect = self.viewport_controller.last_display_rect
        if content_rect is None:
            return
        try:
            current = self.view.viewRange()
            previous = self._last_accepted_view_range
            constrained = constrain_view_range(current, content_rect, previous_view_range=previous)
            constrained = self._aspect_safe_view_range(current, constrained, content_rect)
        except Exception:
            return
        if _view_ranges_close(current, constrained):
            self._remember_accepted_view_range(current)
            return
        self._viewport_constraining = True
        self._viewport_applying = True
        try:
            self.view.setRange(xRange=constrained[0], yRange=constrained[1], padding=0)
            self._remember_accepted_view_range()
        finally:
            self._viewport_applying = False
            self._viewport_constraining = False

    def _remember_accepted_view_range(self, view_range=None) -> None:
        try:
            view_range = self.view.viewRange() if view_range is None else view_range
            self._last_accepted_view_range = (
                (float(view_range[0][0]), float(view_range[0][1])),
                (float(view_range[1][0]), float(view_range[1][1])),
            )
        except Exception:
            self._last_accepted_view_range = None

    def _aspect_safe_view_range(self, current, constrained, content_rect):
        if getattr(self, "displayMode", "square_pixels") != "square_pixels":
            return constrained
        try:
            current_x = abs(float(current[0][1]) - float(current[0][0]))
            current_y = abs(float(current[1][1]) - float(current[1][0]))
            x0, y0, x1, y1 = content_rect
            max_x = abs(float(x1) - float(x0)) / MIN_VIEWPORT_CONTENT_FRACTION
            max_y = abs(float(y1) - float(y0)) / MIN_VIEWPORT_CONTENT_FRACTION
            x_range, y_range = constrained
            x_span = abs(float(x_range[1]) - float(x_range[0]))
            y_span = abs(float(y_range[1]) - float(y_range[0]))
            ratio = current_x / current_y
        except Exception:
            return constrained
        if current_x <= 0.0 or current_y <= 0.0 or max_x <= 0.0 or max_y <= 0.0 or ratio <= 0.0:
            return constrained
        x_span = min(x_span, max_x)
        y_span = min(y_span, max_y)
        target_ratio = x_span / y_span if y_span > 0.0 else ratio
        if target_ratio < ratio:
            y_span = min(y_span, x_span / ratio)
        elif target_ratio > ratio:
            x_span = min(x_span, y_span * ratio)
        return (
            _range_with_span(x_range, x_span),
            _range_with_span(y_range, y_span),
        )

    def _on_view_range_changed(self, *_args):
        if not self._viewport_applying:
            self.viewport_controller.note_user_range_changed()
            self._enforce_viewport_constraints()
        self._sync_profile_marker_visibility()

    # --- Qt Events -----------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self.graphicsView.viewport() and event.type() == QtCore.QEvent.Type.Resize:
            if self.image is not None:
                self._viewport_applying = True
                try:
                    self.viewport_controller.resize(self.view, self.image.shape[:2], event.size(), display_rect=self._current_image_viewport_rect())
                finally:
                    self._viewport_applying = False
        if (
            obj is self.graphicsView.viewport()
            and event.type() == QtCore.QEvent.Type.Wheel
            and self.viewport_controller.is_fit_locked()
        ):
            self._show_fit_mode_interaction_reminder()
            event.accept()
            return True
        if obj is self.graphicsView.viewport() and self._handle_context_menu_event(event):
            return True
        if obj is self.graphicsView.viewport() and self._handle_roi_drawing_event(event):
            return True
        if obj is self.graphicsView.viewport() and self._handle_pointer_interaction_event(event):
            return True
        if (
            obj is self.graphicsView.viewport()
            and self.viewport_controller.is_fit_locked()
            and self._is_fit_locked_pan_attempt(event)
        ):
            self._show_fit_mode_interaction_reminder()
        return super().eventFilter(obj, event)

    def event(self, event):
        if event.type() in (
            QtCore.QEvent.Type.WindowDeactivate,
            QtCore.QEvent.Type.FocusOut,
            QtCore.QEvent.Type.Hide,
        ):
            self._cancel_interaction("window-deactivate")
        return super().event(event)

    def _on_application_state_changed(self, state) -> None:
        if state != QtCore.Qt.ApplicationState.ApplicationActive:
            self._cancel_interaction("application-deactivate")

    def _is_fit_locked_pan_attempt(self, event) -> bool:
        event_type = event.type()
        if event_type == QtCore.QEvent.Type.MouseButtonPress:
            return event.button() in (
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.MiddleButton,
            )
        if event_type == QtCore.QEvent.Type.MouseMove:
            buttons = event.buttons()
            return bool(
                buttons
                & (
                    QtCore.Qt.MouseButton.LeftButton
                    | QtCore.Qt.MouseButton.MiddleButton
                )
            )
        return False

    def _show_fit_mode_interaction_reminder(self):
        now_ms = perf_counter() * 1000.0
        if now_ms - float(getattr(self, "_fit_mode_reminder_last_ms", 0.0) or 0.0) < 1000.0:
            return
        self._fit_mode_reminder_last_ms = now_ms
        notify = getattr(self, "_notify_status", None)
        if not callable(notify):
            return
        try:
            notify("Fit mode is enabled; turn off Fit to pan or zoom.", 1000)
        except TypeError:
            notify("Fit mode is enabled; turn off Fit to pan or zoom.")

    def _handle_pointer_interaction_event(self, event) -> bool:
        return self._pointer_interaction.handle_event(event)

    def _begin_pointer_capture(self, target: InteractionTarget, point: tuple[float, float]) -> bool:
        if target.kind == "roi" and target.object_id is not None:
            item_selection = self._roi_items.get(str(target.object_id))
            if item_selection is None:
                return False
            _item, selection = item_selection
            state = self.interaction_controller.begin_capture(target, point, roi_geometry=selection.geometry)
            self.sync_interaction_state(state)
            return True
        if target.kind == "profile":
            position = self.profileMarkerPosition()
            if position is None:
                return False
            state = self.interaction_controller.begin_capture(target, point, profile_position=position)
            self.sync_interaction_state(state)
            return True
        return False

    def _apply_drag_result(self, result: DragResult | None) -> None:
        if result is None:
            return
        target = result.target
        if target.kind == "roi" and target.object_id is not None and result.geometry is not None:
            self._set_roi_geometry(str(target.object_id), result.geometry, emit=True, sync_item=True)
            return
        if target.kind == "profile" and result.profile_position is not None:
            self._apply_profile_drag_position(result.profile_position)

    def _apply_profile_drag_position(self, position: tuple[float, float]) -> None:
        x, y = self._clamp_profile_point(float(position[0]), float(position[1]))
        previous = self.profileMarkerPosition()
        if previous is not None and (
            float(previous[0]),
            float(previous[1]),
        ) == (float(x), float(y)):
            return
        self.setProfileMarker(x, y, visible=True)
        if self._profile_marker_callback is not None:
            self._profile_marker_callback(float(x), float(y))

    def _cancel_interaction(self, reason: str) -> None:
        self._pointer_interaction.cancel(reason)

    def _cancel_active_capture_for_frame_replacement(self) -> None:
        self._pointer_interaction.cancel_active_capture_for_frame_replacement()

    def _current_image_world_rect(self):
        if self.image is None:
            return None
        pos = self.imageItem.pos()
        return _world_rect_for_shape(self.image.shape[:2], (float(pos.x()), float(pos.y())))

    def _current_image_viewport_rect(self):
        if self.image is None:
            return None
        pos = self.imageItem.pos()
        return _viewport_rect_for_shape(self.image.shape[:2], (float(pos.x()), float(pos.y())))

    def _handle_context_menu_event(self, event):
        if event.type() != QtCore.QEvent.Type.MouseButtonPress or event.button() != QtCore.Qt.MouseButton.RightButton:
            return False
        image_point = self._event_image_point(event)
        self.imageContextMenuRequested.emit(event.globalPos(), image_point)
        return True

    def _handle_roi_drawing_event(self, event):
        state = self.interaction_controller.state
        drawing_active = state.phase is PointerPhase.DRAWING
        if state.pending_draw_tool is None and not drawing_active:
            return False
        event_type = event.type()
        if event_type == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
            point = self._event_image_point(event)
            if point is None or not self.interaction_controller.begin_drawing(point):
                return False
            state = self.interaction_controller.state
            self._set_roi_drawing_preview(state.pending_draw_tool, state.drawing_points)
            return True
        if event_type == QtCore.QEvent.Type.MouseMove and drawing_active:
            point = self._event_image_point(event)
            if point is None:
                return True
            if self.interaction_controller.append_drawing_point(point, minimum_distance=self._freehand_spacing):
                state = self.interaction_controller.state
                self._set_roi_drawing_preview(state.pending_draw_tool, state.drawing_points)
            return True
        if event_type == QtCore.QEvent.Type.MouseButtonRelease and drawing_active:
            result = self.interaction_controller.finish_drawing()
            self._set_roi_drawing_preview(None, ())
            if result is not None:
                if result.tool == "roi_freehand" and len(result.points) >= MIN_FREEHAND_POINTS:
                    self.createRoi(RoiKind.FREEHAND_POLYGON, points=result.points)
                elif result.tool == "roi_polyline" and len(result.points) >= MIN_POLYLINE_POINTS:
                    self.createRoi(RoiKind.POLYLINE, points=result.points)
            tool = self.interaction_controller.state.tool
            self.setInspectionTool(tool if tool in {"cursor", "profile"} else "cursor")
            return True
        return False

    def _set_roi_drawing_preview(self, tool, points) -> None:
        """Backend hook for the transient polyline/freehand drawing path."""

        del tool, points

    def _event_image_point(self, event):
        point = self._event_display_point(event)
        if point is None:
            return None
        x, y = point
        x0, y0, x1, y1 = self._current_image_world_rect()
        return (
            max(float(x0), min(float(x), float(x1))),
            max(float(y0), min(float(y), float(y1))),
        )

    def _event_overlay_point(self, event):
        point = self._event_display_point(event)
        if point is None:
            return None
        x, y = point
        x0, y0, x1, y1 = self._current_image_world_rect()
        if float(x) < min(x0, x1) or float(x) > max(x0, x1):
            return None
        if float(y) < min(y0, y1) or float(y) > max(y0, y1):
            return None
        return (float(x), float(y))

    def _event_display_point(self, event):
        if self.image is None:
            return None
        scene_pos = self.graphicsView.mapToScene(event.pos())
        view_point = self.view.mapSceneToView(scene_pos)
        return (float(view_point.x()), float(view_point.y()))

    def resizeEvent(self, event):
        """On resize, if in 'fit' mode keep the image fully visible."""
        super().resizeEvent(event)


class ImageView2D(ImageViewShell):
    """PyQtGraph-backed shell used by the default display factory."""

    pass


def _image_origin(geometry) -> tuple[float, float]:
    if getattr(geometry, "montage", None) is None:
        return (0.0, 0.0)
    return (
        float(getattr(geometry, "montage_origin_x", 0)),
        float(getattr(geometry, "montage_origin_y", 0)),
    )


def _world_rect_for_shape(shape, origin=(0.0, 0.0)) -> tuple[float, float, float, float]:
    """Pixel-center bounds used by semantic hit testing and overlays."""

    height, width = tuple(int(value) for value in shape[:2])
    x0 = float(origin[0])
    y0 = float(origin[1])
    return (x0, y0, x0 + float(max(0, width - 1)), y0 + float(max(0, height - 1)))


def _viewport_rect_for_shape(shape, origin=(0.0, 0.0)) -> tuple[float, float, float, float]:
    """Pixel-edge bounds used to frame the complete rendered surface."""

    height, width = tuple(int(value) for value in shape[:2])
    x0 = float(origin[0])
    y0 = float(origin[1])
    return (x0, y0, x0 + float(max(1, width)), y0 + float(max(1, height)))


def _viewport_rect_for_geometry(geometry, fallback_shape, fallback_origin=(0.0, 0.0)) -> tuple[float, float, float, float]:
    montage = getattr(geometry, "montage", None)
    if montage is None:
        return _viewport_rect_for_shape(fallback_shape, fallback_origin)
    width = int(montage.columns) * int(montage.tile_width) + max(0, int(montage.columns) - 1) * int(montage.gap)
    height = int(montage.rows) * int(montage.tile_height) + max(0, int(montage.rows) - 1) * int(montage.gap)
    return _viewport_rect_for_shape((height, width), (0.0, 0.0))


def _view_ranges_close(first, second, *, atol: float = 1e-9) -> bool:
    try:
        return all(
            abs(float(first[axis][edge]) - float(second[axis][edge])) <= atol
            for axis in (0, 1)
            for edge in (0, 1)
        )
    except Exception:
        return False


def _range_with_span(axis_range, span: float) -> tuple[float, float]:
    start = float(axis_range[0])
    end = float(axis_range[1])
    center = (start + end) * 0.5
    half = max(0.0, float(span)) * 0.5
    if end < start:
        return (center + half, center - half)
    return (center - half, center + half)


def _array_content_key(array: np.ndarray) -> tuple[object, ...]:
    array = np.asarray(array)
    return (tuple(int(value) for value in array.shape), array.dtype.str, array.tobytes())


def _tiled_montage_placeholder(display_shape, tile_payloads) -> np.ndarray:
    height, width = (max(1, int(value)) for value in tuple(display_shape)[:2])
    sample = next(iter(dict(tile_payloads or {}).values()), None)
    sample_image = None if sample is None else np.asarray(sample.image)
    if sample_image is not None and sample_image.ndim == 3 and sample_image.shape[-1] in (3, 4):
        base = np.zeros((1, 1, 3), dtype=np.uint8)
        return np.broadcast_to(base, (height, width, 3))
    base = np.zeros((1, 1), dtype=np.float32)
    return np.broadcast_to(base, (height, width))


def _tile_commit_report(tile_payloads, tile_delta, stats) -> TileCommitReport:
    payloads = dict(tile_payloads or {})
    backend_presented = getattr(stats, "presented_tiles", None)
    if backend_presented is not None:
        presented = frozenset(int(tile) for tile in tuple(backend_presented or ()) if int(tile) in payloads)
    else:
        visible_items = int(getattr(stats, "visible_items", len(payloads)) or 0)
        if visible_items < len(payloads):
            payload_order = tuple(sorted(int(tile) for tile in payloads))
            presented = frozenset(payload_order[:max(0, visible_items)])
        else:
            presented = frozenset(int(tile) for tile in payloads)
    backend_committed = getattr(stats, "committed_upserts", None)
    committed_upserts = (
        None
        if backend_committed is None
        else frozenset(
            int(tile)
            for tile in tuple(backend_committed or ())
            if int(tile) in dict(getattr(tile_delta, "upserts", {}) or {})
        )
    )
    texture_uploads = int(getattr(stats, "texture_uploads", 0) or 0)
    items_created = int(getattr(stats, "items_created", 0) or 0)
    rgb_window_tiles = int(getattr(stats, "rgb_window_tiles", 0) or 0)
    existing_items = int(getattr(stats, "existing_items_shown", 0) or 0)
    relocated = int(getattr(stats, "relocated_tiles", 0) or 0)
    updated = int(getattr(stats, "items_updated", 0) or 0)
    updated_tiles = tuple(int(tile) for tile in tuple(getattr(stats, "updated_tiles", ()) or ()))
    pyqtgraph_data_updates = max(0, len(updated_tiles) - rgb_window_tiles) if texture_uploads <= 0 else 0
    report_uploads = texture_uploads if texture_uploads > 0 else pyqtgraph_data_updates
    report_upload_bytes = int(getattr(stats, "texture_upload_bytes", 0) or 0)
    if report_upload_bytes <= 0 and updated_tiles:
        report_upload_bytes = sum(int(getattr(payloads.get(int(tile)), "nbytes", 0) or 0) for tile in updated_tiles)
    resident = max(0, max(existing_items, relocated))
    return TileCommitReport(
        presented_tiles=presented,
        committed_upserts=committed_upserts,
        removed_tiles=frozenset(getattr(tile_delta, "removals", ()) or ()),
        texture_uploads=report_uploads,
        texture_upload_bytes=report_upload_bytes,
        pyqtgraph_items_created=items_created,
        cpu_windowed_tiles=rgb_window_tiles,
        resident_rebinds=resident,
        existing_items_shown=existing_items,
        relocated_tiles=relocated,
        storage_rebuilds=int(getattr(stats, "storage_rebuilds", 0) or 0),
        cold_work_ms=float(getattr(stats, "upload_ms", 0.0) or 0.0),
    )


def _is_tiled_loading_only_commit(
    montage_tile_payloads,
    *,
    histogramData,
    histogramPlotData,
) -> bool:
    return (
        montage_tile_payloads is not None
        and not dict(montage_tile_payloads)
        and histogramData is None
        and histogramPlotData is None
    )


def _point_inside_view_range(view_range, x: float, y: float) -> bool:
    try:
        x_range, y_range = view_range
        x0, x1 = sorted((float(x_range[0]), float(x_range[1])))
        y0, y1 = sorted((float(y_range[0]), float(y_range[1])))
    except Exception:
        return True
    return x0 <= float(x) <= x1 and y0 <= float(y) <= y1
