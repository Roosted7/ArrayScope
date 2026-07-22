"""Experimental VisPy-backed 2D image view.

This widget intentionally keeps ArrayScope's existing PyQtGraph interaction and
histogram layer while replacing the expensive pixel upload/display path with
VisPy visuals.  That makes the experiment low-risk: ROI/profile/HUD behaviour
continues to use the same ViewBox/world coordinate model, while scalar images can
use VisPy's GPU texture scaling via ``texture_format='auto'`` and ``clim``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import contextlib

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.display.backend_contract import VISPY_CAPABILITIES
from arrayscope.display.backends.pyqtgraph.histogram_adapter import PyQtGraphHistogramAdapter

# The levels-convergence no-op test (P9) and the tile layer must normalize
# levels identically; the single definition lives with the layer.
from arrayscope.display.backends.vispy.tiles import _normalize_levels
from arrayscope.display.imageview2d import (
    ArrayScopeGraphicsView,
    ImageViewShell,
    _is_tiled_loading_only_commit,
    _point_inside_view_range,
)
from arrayscope.display.model.tile_identity import tile_ack_identity
from arrayscope.display.model.tiled_histogram_identity import (
    histogram_data_from_tile_payloads,
    payload_histogram_source,
    tiled_histogram_key,
    tiled_semantic_histogram_identity,
)
from arrayscope.display.overlay_geometry import (
    montage_overlay_rgba,
    montage_overlay_status_segments,
    roi_outline_points,
)
from arrayscope.display.overlay_hit_test import roi_handle_points
from arrayscope.display.shader_mapping import (
    ShaderDisplayMode,
    common_shader_mapping,
    default_phase_lut,
    shader_mapping_with_lut,
)
from arrayscope.display.view_navigation_driver import QtViewNavigationDriver

if TYPE_CHECKING:
    from arrayscope.display.model.frame import (
        DisplayTilePayload,
        TilePresentationDelta,
        TilePresentationState,
    )


VISPY_WARM_RESIDENCY_MAX_PAYLOADS = 64


class VisPyImageView2D(ImageViewShell):
    """ImageView2D variant that renders pixels with VisPy.

    The class deliberately preserves the public ImageView2D API.  Existing
    renderer code can switch to it through the image-view factory without a new
    set of shims.  The first experimental version focuses on the hot path:
    scalar image/window-level display and montage tile-layer uploads.  PyQtGraph
    still owns the histogram widget and mouse event plumbing; VisPy owns the
    pixel and overlay visuals for this backend.
    """

    rendering_capabilities = VISPY_CAPABILITIES
    draws_qgraphics_roi_items = False
    draws_qgraphics_profile_marker_items = False

    def setupUI(self):
        (
            self._vispy_scene,
            self._vispy_visuals,
            self._vispy_transforms,
            self._vispy_panzoom_camera,
            self._vispy_gloo,
        ) = _import_vispy()
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

        self._vispy_canvas = self._vispy_scene.SceneCanvas(
            keys=None, bgcolor=(0, 0, 0, 1), show=False
        )
        self._vispy_view = self._vispy_canvas.central_widget.add_view()
        self._vispy_view.camera = self._vispy_panzoom_camera(aspect=1)
        self._vispy_view.camera.interactive = False
        self._vispy_view.camera.flip = (False, True, False)
        from arrayscope.display.backends.vispy.tiles import create_gpu_montage_layer

        self._vispy_gpu_montage_layer = create_gpu_montage_layer(
            scene=self._vispy_scene,
            visuals=self._vispy_visuals,
            gloo=self._vispy_gloo,
            transforms=self._vispy_transforms,
            parent=self._vispy_view.scene,
        )
        self._vispy_roi_visuals: dict[str, object] = {}
        self._vispy_roi_handle_visuals: dict[str, object] = {}
        # Visual-only cache of the shell-owned interaction emphasis
        # (ImageViewShell._interaction_visual_profile_part); the marker sync
        # resets it when the marker leaves the viewport.
        self._vispy_profile_hover_part: str | None = None
        self._vispy_roi_drawing_preview = None
        self._vispy_overlay_visuals: list[object] = []
        self._vispy_overlay_mesh = None
        self._vispy_overlay_lines = None
        self._vispy_overlay_key: tuple[object, ...] = ()
        self._vispy_overlay_count = 0
        self._vispy_pending_overlay_clear_request_count: int | None = None
        self._vispy_draw_count = 0
        self._vispy_tile_presentation_request_count = 0
        self._vispy_tile_presentation_draw_count = 0
        self._vispy_canvas_update_request_count = 0
        self._vispy_canvas_update_pending = False
        self._vispy_profile_visuals: dict[str, object] = {}
        self._vispy_last_levels: tuple[float, float] = (0.0, 1.0)
        self._vispy_pending_warm_tile_payloads: dict[int, object] = {}
        self._vispy_pending_warm_tile_context: dict[str, object] = {}
        self._vispy_warm_tile_scheduler = None
        self._last_vispy_warm_tile_stats = None
        self._last_vispy_tiled_levels_key = None
        self._last_vispy_tiled_mapping_key = None
        self._last_vispy_tiled_source_shader_mapping = None
        self._last_vispy_tiled_shader_mapping = None
        self._last_vispy_tiled_histogram_key = None
        # ADR 0050 WP: histogram work is keyed by semantic tile content, so a
        # display-LOD level swap must never repaint or recompute the
        # histogram.  `lod_swap` counts key changes whose semantic inputs were
        # unchanged (must stay 0); `cross_level_reuse` counts texture-identity
        # changes (level swaps) that correctly left the histogram untouched.
        self.tile_histogram_lod_swap_recomputes = 0
        self.tile_histogram_cross_level_reuses = 0
        self._last_vispy_tiled_histogram_inputs = None
        self._last_vispy_frame_viewport_key = None
        self._pending_vispy_histogram_update = None
        self._vispy_histogram_update_pending = False
        # Timer category: UI cosmetic. Lower-priority metadata continuation. Pixel commits are synchronous;
        # PyQtGraph histogram/LUT painting is latest-only secondary work and is
        # admitted only when no interactive render is pending.
        self._vispy_histogram_timer = QtCore.QTimer(self)
        self._vispy_histogram_timer.setSingleShot(True)
        self._vispy_histogram_timer.timeout.connect(self._flush_pending_vispy_histogram_update)
        self._vispy_display_shape: tuple[int, int] = (1, 1)
        self._vispy_camera_sync_pending = False
        self._vispy_camera_key = None
        self._vispy_canvas_native = self._vispy_canvas.native
        self._vispy_canvas_native.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._display_stack.addWidget(self._vispy_canvas_native)

        self.graphicsView = ArrayScopeGraphicsView(self)
        self.graphicsView.setBackground(None)
        self.graphicsView.setStyleSheet("background: transparent; border: 0px;")
        self.graphicsView.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.graphicsView.viewport().setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True
        )
        self.graphicsView.viewport().setStyleSheet("background: transparent;")
        self._display_stack.addWidget(self.graphicsView)
        self.layout.addWidget(self._display_container, 1)

        self.histogram = pg.HistogramLUTWidget()
        self.layout.addWidget(self.histogram)
        self._histogram_adapter = PyQtGraphHistogramAdapter(self.histogram)

        self._vispy_canvas.events.mouse_move.connect(self._on_vispy_mouse_move)
        with contextlib.suppress(Exception):
            self._vispy_canvas.events.draw.connect(self._on_vispy_draw)

    def __init__(self, parent=None, view=None, imageItem=None):
        super().__init__(parent=parent, view=view, imageItem=imageItem)
        self._view_navigation = QtViewNavigationDriver(self)
        self.imageItem.setVisible(False)
        self.histogramImageItem.setVisible(False)
        self._vispy_bounds_item = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
        self._vispy_bounds_item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self._vispy_bounds_item.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
        self._vispy_bounds_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self._layer_owner.add_bounds_item(self._vispy_bounds_item)
        self.view.sigRangeChanged.connect(
            lambda *_args: self._request_vispy_camera_sync(immediate=True)
        )
        state_signal = getattr(self.view, "sigStateChanged", None)
        if state_signal is not None:
            state_signal.connect(lambda *_args: self._request_vispy_camera_sync())

    def _cancel_vispy_speculative_work(self) -> None:
        self._vispy_pending_warm_tile_payloads = {}
        self._vispy_pending_warm_tile_context = {}

    def _on_vispy_draw(self, *_args) -> None:
        self._vispy_draw_count = int(getattr(self, "_vispy_draw_count", 0) or 0) + 1
        self._vispy_canvas_update_pending = False
        self._vispy_tile_presentation_draw_count = int(
            getattr(self, "_vispy_tile_presentation_request_count", 0) or 0
        )
        pending_clear = getattr(self, "_vispy_pending_overlay_clear_request_count", None)
        if pending_clear is not None and self._vispy_tile_presentation_draw_count >= int(
            pending_clear
        ):
            self._hide_vispy_montage_tile_overlays_now(request_update=False)
        # Timer category: anti-hang fallback. A presentationDrawn listener may
        # immediately commit the next tile
        # band. Emitting from inside VisPy's draw callback lets that commit
        # call canvas.update() while the canvas is still painting; Qt/VisPy
        # can drop that request and leave the logical draw gate armed forever.
        # Publish the physical acknowledgement after the draw event returns.
        drawn_request_count = int(self._vispy_tile_presentation_draw_count)
        QtCore.QTimer.singleShot(
            0,
            self,
            lambda count=drawn_request_count: self._publish_vispy_draw_ack(count),
        )

    def _publish_vispy_draw_ack(self, drawn_request_count: int) -> None:
        if int(getattr(self, "_vispy_tile_presentation_draw_count", 0) or 0) < int(
            drawn_request_count
        ):
            return
        self._mark_presentation_drawn()

    def vispyPresentationDiagnostics(self) -> dict[str, object]:
        layer = getattr(self, "_vispy_gpu_montage_layer", None)
        snapshot = (
            dict(layer.diagnostics_snapshot())
            if layer is not None and callable(getattr(layer, "diagnostics_snapshot", None))
            else {
                "presented_tiles": (),
                "presented_tile_count": 0,
                "physically_visible_tile_count": 0,
                "tile_visual_visible_pages": 0,
                "physical_visible_page_count": 0,
                "page_candidate_missing_tile_count": 0,
                "page_candidate_missing_key_count": 0,
                "page_table_resident_count": 0,
                "atlas_page_classes": (),
                "atlas_estimated_gpu_bytes": 0,
                "atlas_budget_bytes": 0,
            }
        )
        tile_visuals = tuple(getattr(layer, "_visuals_by_page", ()) or ())
        visible_tile_visuals = tuple(
            visual for visual in tile_visuals if bool(getattr(visual, "visible", False))
        )
        tile_orders = tuple(int(getattr(visual, "order", 0)) for visual in visible_tile_visuals)
        overlay_visuals = tuple(getattr(self, "_vispy_overlay_visuals", ()) or ())
        visible_overlays = tuple(
            visual for visual in overlay_visuals if bool(getattr(visual, "visible", False))
        )
        overlay_orders = tuple(int(getattr(visual, "order", 0)) for visual in visible_overlays)
        tile_min = min(tile_orders) if tile_orders else None
        overlay_max = max(overlay_orders) if overlay_orders else None
        return {
            "draw_count": int(getattr(self, "_vispy_draw_count", 0) or 0),
            "tile_presentation_request_count": int(
                getattr(self, "_vispy_tile_presentation_request_count", 0) or 0
            ),
            "tile_presentation_draw_count": int(
                getattr(self, "_vispy_tile_presentation_draw_count", 0) or 0
            ),
            "tile_presentation_draw_pending": int(
                getattr(self, "_vispy_tile_presentation_draw_count", 0) or 0
            )
            < int(getattr(self, "_vispy_tile_presentation_request_count", 0) or 0),
            "canvas_update_request_count": int(
                getattr(self, "_vispy_canvas_update_request_count", 0) or 0
            ),
            "canvas_update_pending": bool(getattr(self, "_vispy_canvas_update_pending", False)),
            **snapshot,
            "tile_visual_min_order": tile_min,
            "overlay_count": int(getattr(self, "_vispy_overlay_count", 0) or 0),
            "overlay_visual_visible_items": len(visible_overlays),
            "overlay_visual_max_order": overlay_max,
            "overlays_above_tiles": bool(
                tile_min is not None and overlay_max is not None and overlay_max >= tile_min
            ),
        }

    def presentation_diagnostics(self) -> dict[str, object]:
        diagnostics = super().presentation_diagnostics()
        diagnostics.update(self.vispyPresentationDiagnostics())
        diagnostics["interaction_event_owner"] = self.interaction_event_owner()
        return diagnostics

    def presentationDrawPending(self) -> bool:
        return bool(
            super().presentationDrawPending()
            or getattr(self, "_vispy_canvas_update_pending", False)
            or self._vispy_tile_presentation_draw_pending()
        )

    def _paints_qgraphics_scene(self) -> bool:
        return False

    def reset_surface(self, reason: str) -> None:
        self._cancel_vispy_speculative_work()
        super().reset_surface(reason)
        self._vispy_canvas_update_pending = False

    def teardown_surface(self) -> None:
        if getattr(self, "_surface_teardown_done", False):
            return
        self._cancel_vispy_speculative_work()
        canvas = getattr(self, "_vispy_canvas", None)
        if canvas is not None:
            with contextlib.suppress(Exception):
                canvas.close()
        super().teardown_surface()

    def clearMontageTileLayer(self) -> None:
        self.hide_tiled_presentation("surface-reset")

    def invalidate_tiled_presentation(self, reason: str, *, hide_pixels: bool = True) -> None:
        """Hide semantically superseded pixels without discarding residency.

        With ``hide_pixels=False`` the layer keeps drawing the previous
        plane (stale-but-honest preview across a slice-index-only session
        transition).  Nothing here may create evidence for the successor:
        the surface's ``_last_vispy_tiled_*`` keys still describe the OLD
        content, so the successor's first commit computes a different
        source key, takes the full update path, and swaps the drawn layout
        atomically; acknowledgement flows only from that commit's report.
        """

        if not hide_pixels:
            return
        layer = getattr(self, "_vispy_gpu_montage_layer", None)
        if layer is not None:
            layer.clear()
        self._montage_display_mode = "none"
        self.imageItem.setVisible(False)

    def hide_tiled_presentation(self, reason: str) -> None:
        layer = getattr(self, "_vispy_gpu_montage_layer", None)
        if layer is not None:
            layer.clear()
        self.clearMontageTileOverlays()
        self._montage_display_mode = "none"
        self.imageItem.setVisible(False)

    def reset_tiled_residency(self, reason: str) -> None:
        layer = getattr(self, "_vispy_gpu_montage_layer", None)
        if layer is not None and hasattr(layer, "reset_residency"):
            layer.reset_residency()
        else:
            self.hide_tiled_presentation(reason)
        self._last_vispy_tile_payloads = None
        self._last_vispy_tiled_source_key = None
        self._last_vispy_tiled_structure_key = None
        self._last_vispy_tiled_levels_key = None
        self._last_vispy_tiled_mapping_key = None
        self._last_vispy_tiled_source_shader_mapping = None
        self._last_vispy_tiled_shader_mapping = None
        self._last_vispy_tiled_histogram_key = None
        self._last_vispy_tiled_histogram_inputs = None
        self._last_vispy_frame_viewport_key = None
        self._last_vispy_tiled_reset_reason = str(reason)
        self._montage_display_mode = "none"
        self.imageItem.setVisible(False)

    def clear(self):
        super().clear()
        self.reset_tiled_residency("view-clear")

    def setColorMap(self, colormap):
        """Update the shared colorbar/render-surface LUT without re-uploading pixels."""

        super().setColorMap(colormap)
        if getattr(self, "_montage_display_mode", "") == "vispy_tile_layer":
            mapping = self._display_shader_mapping(
                getattr(self, "_last_vispy_tiled_source_shader_mapping", None)
            )
            self._last_vispy_tiled_shader_mapping = mapping
            self._last_vispy_tiled_mapping_key = _shader_mapping_key(mapping)
            layer = getattr(self, "_vispy_gpu_montage_layer", None)
            if layer is not None:
                layer.set_shader_mapping(mapping)
        self._request_vispy_canvas_update()

    def _display_shader_mapping(self, mapping):
        display_mode = getattr(
            getattr(mapping, "display_mode", None),
            "value",
            getattr(mapping, "display_mode", None),
        )
        if display_mode == ShaderDisplayMode.PHASE_COLOR.value:
            explicit_lut = getattr(mapping, "lut_data", None)
            if explicit_lut is not None:
                return shader_mapping_with_lut(
                    mapping,
                    explicit_lut,
                    lut_identity=getattr(mapping, "lut_identity", None),
                )
            # A bare phase-color mapping means the canonical cyclic phase LUT.
            # ImageView2D starts with a scalar grayscale LUT; treating that as
            # explicit phase presentation silently turned acknowledged complex
            # textures grayscale until the outer window happened to switch the
            # colormap family.
            if getattr(self, "_display_colormap", None) is None:
                return shader_mapping_with_lut(
                    mapping,
                    default_phase_lut(),
                    lut_identity=("default-phase",),
                )
        return shader_mapping_with_lut(
            mapping,
            self.displayColorMapLookupTable(),
            lut_identity=self.displayColorMapKey(),
        )

    def _apply_backend_tiled_presentation(
        self,
        img: np.ndarray,
        *,
        histogramPlotData,
        geometry,
        levels: tuple[float, float],
        histogramRange: tuple[float, float],
        viewport_policy,
        rgb_already_windowed: bool,
        montage_dirty_tiles: tuple[int, ...] | None,
        montage_tile_source_ids: dict[int, object] | None,
        montage_tile_payloads: dict[int, DisplayTilePayload] | None,
        shader_mapping,
        tile_delta: TilePresentationDelta,
        tile_residency_budget_bytes: int,
        frame_plan,
    ):
        # ADR 0055 G4c: the prefetch warm hook mirrors the residency knobs of
        # the most recent visible commit instead of inventing its own.  Tiled
        # commits carry histogram evidence only via histogramPlotData.
        self._vispy_last_tile_residency_budget_bytes = int(tile_residency_budget_bytes or 0)
        self._vispy_last_tiled_rgb_already_windowed = bool(rgb_already_windowed)
        histogramData = None
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
                frame_plan=frame_plan,
            )
            histogram_semantic_identity = _tiled_semantic_histogram_identity(montage_tile_payloads)
            histogram_key = _tiled_histogram_key(
                histogramRange,
                histogramPlotData=histogramPlotData,
                tile_delta=tile_delta,
                semantic_identity=histogram_semantic_identity,
            )
            histogram_inputs = (
                histogram_semantic_identity,
                id(histogramPlotData),
                (float(histogramRange[0]), float(histogramRange[1])),
            )
            viewport_key = (
                structure_key,
                str(getattr(viewport_policy, "value", viewport_policy)),
            )
            previous_source_key = getattr(self, "_last_vispy_tiled_source_key", None)
            previous_structure_key = getattr(self, "_last_vispy_tiled_structure_key", None)
            previous_levels_key = getattr(self, "_last_vispy_tiled_levels_key", None)
            previous_mapping_key = getattr(self, "_last_vispy_tiled_mapping_key", None)
            previous_histogram_key = getattr(self, "_last_vispy_tiled_histogram_key", None)
            previous_viewport_key = getattr(self, "_last_vispy_frame_viewport_key", None)
            structure_changed = structure_key != previous_structure_key
            # The completed-transaction cache is an optimization hint, not
            # physical truth. A programmatic/histogram level update can move
            # both the public display state and the page uniforms between two
            # tiled commits without rewriting this cache. Comparing only the
            # cache then lets the next canonical presentation skip its level
            # command and acknowledge a different physical window. Always
            # include the current display and layer states in the no-op test.
            display_level_key = _normalize_levels(self.getLevels(), level_key)
            layer = getattr(self, "_vispy_gpu_montage_layer", None)
            layer_level_key = _normalize_levels(
                getattr(layer, "_levels", None),
                level_key,
            )
            display_levels_changed = level_key != display_level_key
            levels_changed = bool(level_key != previous_levels_key or level_key != layer_level_key)
            mapping_changed = mapping_key != previous_mapping_key
            loading_only = _is_tiled_loading_only_commit(
                montage_tile_payloads,
                histogramData=histogramData,
                histogramPlotData=histogramPlotData,
            )
            previous_layer_stats = getattr(
                getattr(self, "_vispy_gpu_montage_layer", None), "last_stats", None
            )
            must_clear_visible_pages = bool(
                loading_only
                and (
                    int(getattr(previous_layer_stats, "visible_items", 0) or 0) > 0
                    or int(getattr(previous_layer_stats, "active_pages", 0) or 0) > 0
                )
            )
            requested_presented_tiles = _requested_direct_payload_tiles(
                montage_tile_payloads, tile_delta
            )
            previous_presented_tiles = getattr(previous_layer_stats, "presented_tiles", None)
            presentation_incomplete = bool(
                montage_tile_payloads is not None
                and previous_presented_tiles is not None
                and {int(tile) for tile in tuple(previous_presented_tiles or ())}
                != requested_presented_tiles
            )
            data_unchanged = (
                montage_tile_payloads is not None
                and montage_dirty_tiles == ()
                and source_key == previous_source_key
                and not structure_changed
                and not must_clear_visible_pages
                and not presentation_incomplete
            )

            previous_mode = getattr(self, "_montage_display_mode", "none")
            placeholder_key = _vispy_tiled_placeholder_key(img)
            if previous_mode != "vispy_tile_layer" or placeholder_key != getattr(
                self, "_last_vispy_tiled_placeholder_key", None
            ):
                self.image = img
                self._last_vispy_tiled_placeholder_key = placeholder_key
            if not loading_only:
                self.histogramSource = histogramData
                self.histogramPlotSource = histogramPlotData
            self._last_vispy_tile_payloads = montage_tile_payloads
            if not loading_only:
                self.setHistogramDataBounds(histogramRange)
            self._montage_display_mode = "vispy_tile_layer"

            if data_unchanged and not levels_changed and not mapping_changed:
                from arrayscope.display.model.tile_stats import TileLayerUpdateStats

                visible = len(montage_tile_payloads or {})
                presented = getattr(previous_layer_stats, "presented_tiles", None)
                stats = TileLayerUpdateStats(
                    visible_items=len(tuple(presented or ())) if presented is not None else visible,
                    presented_tiles=None
                    if presented is None
                    else tuple(int(tile) for tile in presented),
                    items_updated=0,
                    items_skipped=len(tuple(presented or ())) if presented is not None else visible,
                    rgb_window_tiles=0,
                    resident_items=int(getattr(previous_layer_stats, "resident_items", 0) or 0),
                    storage_capacity=int(getattr(previous_layer_stats, "storage_capacity", 0) or 0),
                    estimated_gpu_bytes=int(
                        getattr(previous_layer_stats, "estimated_gpu_bytes", 0) or 0
                    ),
                    cpu_shadow_bytes=int(getattr(previous_layer_stats, "cpu_shadow_bytes", 0) or 0),
                    page_count=int(getattr(previous_layer_stats, "page_count", 0) or 0),
                    active_pages=int(getattr(previous_layer_stats, "active_pages", 0) or 0),
                    device_max_texture_size=int(
                        getattr(previous_layer_stats, "device_max_texture_size", 0) or 0
                    ),
                    budget_bytes=int(getattr(previous_layer_stats, "budget_bytes", 0) or 0),
                    near_resident_items=int(
                        getattr(previous_layer_stats, "near_resident_items", 0) or 0
                    ),
                    warm_resident_items=int(
                        getattr(previous_layer_stats, "warm_resident_items", 0) or 0
                    ),
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
                    frame_plan=frame_plan,
                )
            stats_presented_tiles = getattr(stats, "presented_tiles", None)
            stats_presented_set = (
                None
                if stats_presented_tiles is None
                else {int(tile) for tile in tuple(stats_presented_tiles or ())}
            )
            tiled_presentation_complete = (
                stats_presented_set is None or stats_presented_set == requested_presented_tiles
            )
            self._record_tile_layer_stats(stats)
            if not (data_unchanged and not levels_changed and not mapping_changed):
                self._request_vispy_tile_layer_redraw()

            # Histogram, levels, geometry, and viewport are separate concerns.
            # Cached visible-tile switches must not repaint the PyQtGraph
            # histogram/axes unless the histogram stream itself changed.  This
            # keeps VisPy pixel commits from inheriting PyQtGraph LUT repaint
            # cost on every scroll step.
            histogram_changed = histogram_key != previous_histogram_key
            previous_histogram_inputs = getattr(self, "_last_vispy_tiled_histogram_inputs", None)
            if montage_tile_payloads and previous_histogram_key is not None:
                if histogram_changed and previous_histogram_inputs == histogram_inputs:
                    # A histogram refresh whose semantic inputs did not change
                    # can only be caused by presentation identity churn (for
                    # example a display-LOD level swap).  ADR 0050 requires
                    # this to be structurally impossible; the counter is the
                    # regression alarm.
                    self.tile_histogram_lod_swap_recomputes += 1
                elif not histogram_changed and source_key != previous_source_key:
                    self.tile_histogram_cross_level_reuses += 1
            if histogram_changed and not loading_only:
                if histogramPlotData is None and montage_tile_payloads:
                    # Defer the tile-payload histogram fallback to the coalescing
                    # histogram timer.  Concatenating every visible tile on every
                    # commit is O(n^2) across a progressive stream; a provider
                    # lets bursts of commits collapse into one materialization,
                    # and its stable identity avoids per-commit repaints.
                    payloads_for_histogram = montage_tile_payloads

                    def histogramPlotData(payloads=payloads_for_histogram):
                        return _histogram_data_from_tile_payloads(payloads)

                self._request_histogram_for_vispy(
                    histogramData,
                    histogramPlotData,
                    level_key,
                    histogramRange=histogramRange,
                )
            if (levels_changed or display_levels_changed) and not loading_only:
                self._set_vispy_display_levels(level_key)
                if not histogram_changed:
                    self._request_histogram_for_vispy(
                        self.histogramSource,
                        self.histogramPlotSource,
                        level_key,
                        histogramRange=None,
                        defer=True,
                    )

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

            if tiled_presentation_complete:
                self._last_vispy_tiled_source_key = source_key
                self._last_vispy_tiled_structure_key = structure_key
            if tiled_presentation_complete and not loading_only:
                self._last_vispy_tiled_levels_key = level_key
                self._last_vispy_tiled_mapping_key = mapping_key
                self._last_vispy_tiled_source_shader_mapping = source_shader_mapping
                self._last_vispy_tiled_shader_mapping = shader_mapping
                self._last_vispy_tiled_histogram_key = histogram_key
                self._last_vispy_tiled_histogram_inputs = histogram_inputs
                self._last_vispy_frame_viewport_key = viewport_key
            return stats
        finally:
            self._applying_presentation = applying
            self._finish_upload_timing()

    def _after_tiled_commit(
        self,
        *,
        tile_state: TilePresentationState,
        tile_delta: TilePresentationDelta,
        tile_payloads: dict[int, DisplayTilePayload],
        geometry,
        rgb_already_windowed: bool,
        tile_residency_budget_bytes: int,
    ) -> None:
        if getattr(tile_delta, "upserts", None) or getattr(tile_delta, "removals", None):
            return
        warm_payloads = {
            int(tile): payload
            for tile, payload in tile_state.near_payloads(tile_delta).items()
            if int(tile) not in tile_payloads
        }
        self._schedule_vispy_warm_tile_residency(
            warm_payloads,
            geometry=geometry,
            rgb_already_windowed=rgb_already_windowed,
            tile_delta=tile_delta,
            tile_residency_budget_bytes=tile_residency_budget_bytes,
        )

    def _schedule_vispy_warm_tile_residency(
        self,
        payloads,
        *,
        geometry,
        rgb_already_windowed: bool,
        tile_delta,
        tile_residency_budget_bytes: int,
    ) -> None:
        if not payloads:
            return
        if getattr(self, "_vispy_pending_warm_tile_payloads", None):
            return
        payloads = dict(payloads or {})
        if len(payloads) > VISPY_WARM_RESIDENCY_MAX_PAYLOADS:
            near_order = tuple(
                int(tile) for tile in tuple(getattr(tile_delta, "near_tiles", ()) or ())
            )
            ordered = [tile for tile in near_order if tile in payloads]
            ordered.extend(tile for tile in payloads if int(tile) not in set(ordered))
            payloads = {
                int(tile): payloads[int(tile)]
                for tile in ordered[:VISPY_WARM_RESIDENCY_MAX_PAYLOADS]
            }
        from arrayscope.display.backends.vispy.tiles import PayloadBatchQueue

        self._vispy_pending_warm_tile_payloads = PayloadBatchQueue(payloads)
        self._vispy_pending_warm_tile_context = {
            "geometry": geometry,
            "rgb_already_windowed": bool(rgb_already_windowed),
            "tile_delta": tile_delta,
            "tile_residency_budget_bytes": int(tile_residency_budget_bytes),
        }
        self._submit_vispy_warm_tile_residency_continuation()

    def warmTiledResidency(
        self,
        *,
        payloads,
        geometry,
        levels: tuple[float, float],
        rgb_already_windowed: bool = False,
        tile_delta=None,
        tile_residency_budget_bytes: int = 0,
        frame_plan=None,
    ):
        """Queue non-presenting montage payloads for bounded atlas warming.

        This is the persistent-residency seam shared with the CPU-item
        backend.  VisPy performs the GL work through its standing GUI-thread
        continuation; warming changes only content-keyed pool residency and
        never acknowledges a presentation or changes an active slot mapping.
        ``levels`` and ``frame_plan`` are accepted as part of the backend
        contract but do not affect raw atlas content identity.
        """

        del levels, frame_plan
        budget_bytes = int(
            tile_residency_budget_bytes
            or getattr(self, "_vispy_last_tile_residency_budget_bytes", 0)
            or 0
        )
        if bool(getattr(tile_delta, "atomic_handoff", False)):
            # The frame coordinator already invokes atomic warming in small,
            # receiver-bound low-priority batches. Queueing those batches a
            # second time lets the coordinator mark them complete before GL
            # residency exists, so the final zero-upload handoff can stall or
            # expose a partially rewritten frame. Perform only the residency
            # mutation here; the layer contract forbids presentation changes.
            layer = getattr(self, "_vispy_gpu_montage_layer", None)
            if layer is None or not hasattr(layer, "warm_residency"):
                return None
            stats = layer.warm_residency(
                payloads=dict(payloads or {}),
                geometry=geometry,
                rgb_already_windowed=bool(rgb_already_windowed),
                tile_delta=tile_delta,
                tile_residency_budget_bytes=budget_bytes,
            )
            self._last_vispy_warm_tile_stats = stats
            return stats
        self._schedule_vispy_warm_tile_residency(
            payloads,
            geometry=geometry,
            rgb_already_windowed=rgb_already_windowed,
            tile_delta=tile_delta,
            tile_residency_budget_bytes=budget_bytes,
        )
        return None

    def _tiled_presentation_layer(self):
        return getattr(self, "_vispy_gpu_montage_layer", None)

    def _submit_vispy_warm_tile_residency_continuation(self) -> None:
        scheduler = getattr(self, "_vispy_warm_tile_scheduler", None)
        if callable(scheduler):
            scheduler(self._process_vispy_warm_tile_residency)

    def _process_vispy_warm_tile_residency(self) -> None:
        pending = getattr(self, "_vispy_pending_warm_tile_payloads", None)
        context = dict(getattr(self, "_vispy_pending_warm_tile_context", {}) or {})
        if not pending:
            return
        from arrayscope.display.backends.vispy.tiles import PayloadBatchQueue

        queue = pending if isinstance(pending, PayloadBatchQueue) else PayloadBatchQueue(pending)
        batch = queue.take()
        self._vispy_pending_warm_tile_payloads = queue if queue else {}
        self._vispy_pending_warm_tile_context = context if queue else {}
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
        if queue:
            self._submit_vispy_warm_tile_residency_continuation()

    def warmPlaneResidency(self, payload) -> bool:
        """Warm one anchored non-montage plane into GPU chunk residency.

        ADR 0055 G4c: the slice prefetcher hands over the exact payload of an
        adjacent plane after its evaluation landed in the CPU display cache;
        the atlas pool uploads the plane's chunks as pure page-table
        residency so a subsequent fixed-index scroll commits upload-free.
        GUI-thread only (GL uploads). Returns True when the plane is warm
        (uploaded now or already resident); False when warming is
        unavailable or was denied (no layer, no atlas layout yet, storage
        mode mismatch, capacity/budget denial) — all silent by design.
        """

        layer = getattr(self, "_vispy_gpu_montage_layer", None)
        if layer is None or not hasattr(layer, "warm_residency"):
            return False
        if payload is None or getattr(payload, "source_anchor", None) is None:
            return False
        try:
            stats = layer.warm_residency(
                payloads={int(payload.tile_number): payload},
                geometry=None,
                rgb_already_windowed=bool(
                    getattr(self, "_vispy_last_tiled_rgb_already_windowed", False)
                ),
                tile_delta=None,
                tile_residency_budget_bytes=int(
                    getattr(self, "_vispy_last_tile_residency_budget_bytes", 0) or 0
                ),
            )
        except Exception:
            return False
        self._last_vispy_warm_tile_stats = stats
        capacity_denied = bool(getattr(stats, "capacity_warning", ""))
        return bool((stats.items_updated or stats.items_skipped) and not capacity_denied)

    _level_preview_timing_channel = "vispy_level_preview"

    def _apply_preview_levels_to_display(self, levels, *, final: bool) -> None:
        self._vispy_last_levels = levels
        if self._montage_display_mode != "vispy_tile_layer":
            return
        layer = getattr(self, "_vispy_gpu_montage_layer", None)
        if layer is None:
            return
        # A level gesture owns only the level uniform. Replaying the
        # separately cached frame mapping here let a stale scalar mapping
        # overwrite an already-correct complex phase mapping during the
        # full-montage -> scroll transition. The next payload commit then
        # appeared to "heal" the psychedelic tiles. Preserve the physical
        # layer's current mapping and update exactly the signal received.
        stats = layer.set_presentation_uniforms(levels=levels)
        self._record_tile_layer_stats(stats)
        self._request_vispy_tile_layer_redraw()
        handler = getattr(self, "_level_presentation_change_handler", None)
        if callable(handler):
            handler(levels, final=bool(final))

    def _backend_roi_visual_upserted(self, selection) -> None:
        self._upsert_vispy_roi(selection.id, selection.geometry, selection.color)

    def _backend_roi_visual_removed(self, roi_id) -> None:
        self._remove_vispy_roi(roi_id)

    def _backend_roi_emphasis_changed(self, roi_id: str) -> None:
        item_selection = self._roi_items.get(str(roi_id))
        if item_selection is None:
            return
        _item, selection = item_selection
        self._upsert_vispy_roi(selection.id, selection.geometry, selection.color)

    def _upsert_vispy_roi(self, roi_id, geometry, color) -> None:
        points = _vispy_roi_points(geometry)
        if points is None:
            self._remove_vispy_roi(roi_id)
            return
        roi_id = str(roi_id)
        width, rgb = self._roi_visual_style(roi_id, color)
        line_color = _vispy_color(rgb)
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
            with contextlib.suppress(Exception):
                visual.set_gl_state("translucent", depth_test=False)
            self._vispy_roi_visuals[str(roi_id)] = visual
        else:
            visual.set_data(pos=points, color=line_color, width=float(width))
            visual.order = 10_000
        visual.visible = True
        self._upsert_vispy_roi_handles(roi_id, geometry, color)
        self._request_vispy_canvas_update()

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
                with contextlib.suppress(Exception):
                    current.visible = False
        self._request_vispy_canvas_update()

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
        highlighted, interactive = self._roi_visual_emphasis(roi_id)
        marker.set_data(
            points,
            symbol="square",
            size=12.0 if highlighted or interactive else 10.0,
            face_color=(0.05, 0.05, 0.05, 0.75),
            edge_color=_vispy_color((255, 255, 255) if interactive else color),
            edge_width=2.0 if highlighted or interactive else 1.25,
        )
        marker.order = 10_001
        marker.visible = True
        self._vispy_roi_handle_visuals[roi_id] = [marker]

    def _vispy_handle_world_size(self) -> float:
        """World-space radius corresponding to an eight-pixel hit target."""

        try:
            x_range, y_range = self.view.viewRange()
            viewport = self.graphicsView.viewport()
            x_per_pixel = abs(float(x_range[1]) - float(x_range[0])) / max(1, int(viewport.width()))
            y_per_pixel = abs(float(y_range[1]) - float(y_range[0])) / max(
                1, int(viewport.height())
            )
            return max(x_per_pixel, y_per_pixel) * 8.0
        except Exception:
            return 2.0

    def _backend_profile_emphasis_changed(self) -> None:
        part = self._interaction_visual_profile_part
        self._vispy_profile_hover_part = None if part is None else str(part)
        self._sync_vispy_profile_marker()

    def setViewportContentExtent(self, extent) -> bool:
        changed = super().setViewportContentExtent(extent)
        if extent is not None:
            # Keep the scene bounds item in step: the ViewBox aspect-relock
            # on fit-unlock ranges around children bounds, which otherwise
            # still hold the pre-plan single-slice rect until a commit lands.
            # Bounds publication is plan-owned geometry, not user camera
            # input. Preserve the committed camera until acknowledgement
            # explicitly replays AUTO/FIT for the successor extent.
            prior_range = self.view.viewRange()
            blocker = QtCore.QSignalBlocker(self.view)
            self._viewport_applying = True
            try:
                self._sync_vispy_bounds(extent)
                self.view.setRange(
                    xRange=prior_range[0],
                    yRange=prior_range[1],
                    padding=0,
                )
            finally:
                blocker.unblock()
                self._viewport_applying = False
            if changed:
                self._remember_accepted_view_range()
        return bool(changed)

    def _viewport_content_shape(self):
        extent = getattr(self, "_viewport_content_extent", None)
        if extent is not None:
            return extent
        return getattr(self, "_vispy_display_shape", None) or self.image.shape[:2]

    def _after_viewport_camera_change(self) -> None:
        self._sync_vispy_camera_to_view()

    def _sync_backend_camera_to_view(self) -> None:
        self._sync_vispy_camera_to_view()

    def _after_profile_marker_sync(self) -> None:
        self._sync_vispy_profile_marker()

    def _sync_vispy_profile_marker(self) -> None:
        if self.image is None or not bool(
            getattr(self, "_profile_marker_requested_visible", False)
        ):
            self._vispy_profile_hover_part = None
            self._hide_vispy_profile_visuals()
            return
        position = self.profileMarkerPosition()
        if position is None:
            return
        x, y = (float(position[0]), float(position[1]))
        if not _point_inside_view_range(self.view.viewRange(), x, y):
            self._hide_vispy_profile_visuals()
            return
        x0, y0, x1, y1 = self._current_profile_bounds()
        hovered = self._vispy_profile_hover_part is not None
        line_color = (255, 125, 55) if hovered else (230, 60, 30)
        line_width = 2.5 if hovered else 1.5
        self._upsert_vispy_line(
            "profile_v",
            np.asarray([[x, y0], [x, y1]], dtype=np.float32),
            line_color,
            width=line_width,
        )
        self._upsert_vispy_line(
            "profile_h",
            np.asarray([[x0, y], [x1, y]], dtype=np.float32),
            line_color,
            width=line_width,
        )
        marker = max(0.8, min(float(x1 - x0), float(y1 - y0)) * 0.025)
        self._upsert_vispy_line(
            "profile_handle_x",
            np.asarray([[x - marker, y], [x + marker, y]], dtype=np.float32),
            line_color,
            width=3.0 if hovered else 2.0,
        )
        self._upsert_vispy_line(
            "profile_handle_y",
            np.asarray([[x, y - marker], [x, y + marker]], dtype=np.float32),
            line_color,
            width=3.0 if hovered else 2.0,
        )
        self._upsert_vispy_profile_dot(x, y, hovered=hovered)
        self._request_vispy_canvas_update()

    def _hide_vispy_profile_visuals(self) -> None:
        for visual in getattr(self, "_vispy_profile_visuals", {}).values():
            _set_visual_visible(visual, False)
        self._request_vispy_canvas_update()

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
            self.sync_interaction_state(self.interaction_controller.clear_hover())
        points = np.asarray(tuple(points or ()), dtype=np.float32).reshape((-1, 2))
        visual = getattr(self, "_vispy_roi_drawing_preview", None)
        if tool is None or len(points) < 2:
            _set_visual_visible(visual, False)
            if visual is not None:
                self._request_vispy_canvas_update()
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
            with contextlib.suppress(Exception):
                visual.set_gl_state("translucent", depth_test=False)
            self._vispy_roi_drawing_preview = visual
        else:
            visual.set_data(pos=points, color=_vispy_color((255, 190, 60)), width=2.5)
        visual.visible = True
        self._request_vispy_canvas_update()

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
        if (
            self._vispy_tile_presentation_draw_pending()
            and int(getattr(self, "_vispy_overlay_count", 0) or 0) > 0
        ):
            self._vispy_pending_overlay_clear_request_count = int(
                getattr(self, "_vispy_tile_presentation_request_count", 0) or 0
            )
            self._request_vispy_canvas_update()
            return
        self._hide_vispy_montage_tile_overlays_now()

    def montageTileOverlayCount(self) -> int:
        return int(getattr(self, "_vispy_overlay_count", 0) or 0)

    def _set_vispy_montage_tile_overlays(self, overlays) -> None:
        overlays = tuple(overlays or ())
        self._vispy_pending_overlay_clear_request_count = None
        key = _overlay_batch_key(overlays)
        self._vispy_overlay_count = len(overlays)
        if key == getattr(self, "_vispy_overlay_key", ()):
            for visual in getattr(self, "_vispy_overlay_visuals", ()):
                _set_visual_visible(visual, bool(overlays))
            return
        self._vispy_overlay_key = key
        if not overlays:
            self._hide_vispy_montage_tile_overlays_now()
            return

        mesh = self._ensure_vispy_overlay_mesh()
        lines = self._ensure_vispy_overlay_lines()
        vertices, faces, colors = _overlay_mesh_arrays(overlays)
        line_points, line_colors = _overlay_line_arrays(overlays)
        mesh.set_data(vertices=vertices, faces=faces, vertex_colors=colors)
        lines.set_data(pos=line_points, color=line_colors, width=1.25, connect="segments")
        mesh.visible = True
        lines.visible = bool(len(line_points))
        self._request_vispy_canvas_update()

    def _hide_vispy_montage_tile_overlays_now(self, *, request_update: bool = True) -> None:
        super().clearMontageTileOverlays()
        self._vispy_overlay_key = ()
        self._vispy_overlay_count = 0
        self._vispy_pending_overlay_clear_request_count = None
        self._set_vispy_overlay_visuals_visible(False, suppress_canvas_update=not request_update)
        if request_update:
            self._request_vispy_canvas_update()

    def _set_vispy_overlay_visuals_visible(
        self, visible: bool, *, suppress_canvas_update: bool = False
    ) -> None:
        visuals = tuple(getattr(self, "_vispy_overlay_visuals", ()))
        if not suppress_canvas_update:
            for visual in visuals:
                _set_visual_visible(visual, visible)
            return
        canvas = getattr(self, "_vispy_canvas", None)
        canvas_update = getattr(canvas, "update", None)
        if canvas is None or not callable(canvas_update):
            for visual in visuals:
                _set_visual_visible(visual, visible)
            return
        try:
            canvas.update = lambda *args, **kwargs: None
            for visual in visuals:
                _set_visual_visible(visual, visible)
        finally:
            canvas.update = canvas_update

    def _vispy_tile_presentation_draw_pending(self) -> bool:
        return int(getattr(self, "_vispy_tile_presentation_draw_count", 0) or 0) < int(
            getattr(self, "_vispy_tile_presentation_request_count", 0) or 0
        )

    def _ensure_vispy_overlay_mesh(self):
        mesh = getattr(self, "_vispy_overlay_mesh", None)
        if mesh is None:
            mesh = self._vispy_visuals.Mesh(parent=self._vispy_view.scene)
            mesh.order = 5
            with contextlib.suppress(Exception):
                mesh.set_gl_state("translucent", depth_test=False)
            mesh.visible = False
            self._vispy_overlay_mesh = mesh
            self._refresh_vispy_overlay_visual_list()
        return mesh

    def _ensure_vispy_overlay_lines(self):
        lines = getattr(self, "_vispy_overlay_lines", None)
        if lines is None:
            lines = self._vispy_visuals.Line(
                parent=self._vispy_view.scene, method="gl", connect="segments"
            )
            lines.order = 6
            with contextlib.suppress(Exception):
                lines.set_gl_state("translucent", depth_test=False)
            lines.visible = False
            self._vispy_overlay_lines = lines
            self._refresh_vispy_overlay_visual_list()
        return lines

    def _refresh_vispy_overlay_visual_list(self) -> None:
        self._vispy_overlay_visuals = [
            visual
            for visual in (
                getattr(self, "_vispy_overlay_mesh", None),
                getattr(self, "_vispy_overlay_lines", None),
            )
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
        width = int(montage.columns) * int(montage.tile_width) + max(
            0, int(montage.columns) - 1
        ) * int(montage.gap)
        height = int(montage.rows) * int(montage.tile_height) + max(0, int(montage.rows) - 1) * int(
            montage.gap
        )
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
            camera.aspect = (
                1.0 if getattr(self, "displayMode", "square_pixels") == "square_pixels" else None
            )
            self._request_vispy_canvas_update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image is not None and self.viewport_controller.is_fit_locked():
            self._sync_vispy_camera_to_view()

    def _update_histogram_for_vispy(self, histogramData, histogramPlotData, levels) -> None:
        if callable(histogramPlotData):
            # Deferred tiled-commit fallback: materialize once at consumption
            # and persist it, matching the eager path's stored plot source.
            histogramPlotData = histogramPlotData()
            self.histogramPlotSource = histogramPlotData
        previous_plot_source = self.histogramPlotSource
        self.histogramPlotSource = histogramPlotData
        try:
            plot_data = self._histogram_plot_data(histogramData)
        finally:
            self.histogramPlotSource = previous_plot_source
        if plot_data is None:
            return
        self._bind_histogram_item(self.histogramImageItem)
        self._set_image_item_data(
            self.histogramImageItem,
            plot_data,
            self._histogram_levels_for_display(levels),
            role="histogram",
        )

    def _set_vispy_display_levels(self, levels) -> None:
        low, high = (float(levels[0]), float(levels[1]))
        self._displayLevels = (low, high)
        if self._histogram_preview_controller is not None and self._applying_presentation:
            self._histogram_preview_controller.cancel()

    def _request_histogram_for_vispy(
        self,
        histogramData,
        histogramPlotData,
        levels,
        *,
        histogramRange=None,
        refresh_curve: bool = True,
        defer: bool = False,
    ) -> None:
        if (
            not bool(defer)
            and getattr(self.histogramImageItem, "image", None) is None
            and not self._vispy_input_interactive()
        ):
            if refresh_curve:
                self._update_histogram_for_vispy(histogramData, histogramPlotData, levels)
            self._sync_vispy_histogram_widget_bounds(levels, histogramRange=histogramRange)
            return
        pending = self._pending_vispy_histogram_update
        if pending is not None:
            # Coalesce: keep the newest levels/range but never let a data-less
            # re-request (for example a levels-only refresh) drop pending
            # histogram data or its deferred provider.
            if histogramData is None:
                histogramData = pending[0]
            if histogramPlotData is None:
                histogramPlotData = pending[1]
        self._pending_vispy_histogram_update = (
            histogramData,
            histogramPlotData,
            tuple(float(value) for value in levels),
            None if histogramRange is None else tuple(float(value) for value in histogramRange),
            bool(refresh_curve),
        )
        if self._vispy_histogram_update_pending:
            return
        self._vispy_histogram_update_pending = True
        self._schedule_pending_vispy_histogram_update()

    def _schedule_pending_vispy_histogram_update(self) -> None:
        timer = getattr(self, "_vispy_histogram_timer", None)
        if timer is None or timer.isActive():
            return
        timer.start(
            0
            if self._vispy_histogram_can_flush_now()
            else self._vispy_histogram_retry_interval_ms()
        )

    def _flush_pending_vispy_histogram_update(self) -> None:
        pending = self._pending_vispy_histogram_update
        if pending is None:
            self._vispy_histogram_update_pending = False
            return
        if not self._vispy_histogram_can_flush_now():
            self._schedule_pending_vispy_histogram_update()
            return
        self._vispy_histogram_update_pending = False
        self._pending_vispy_histogram_update = None
        histogramData, histogramPlotData, levels, histogramRange, refresh_curve = pending
        started_timing = self._upload_timing is None
        if started_timing:
            self._start_upload_timing("vispy_histogram")
        try:
            if refresh_curve:
                self._update_histogram_for_vispy(histogramData, histogramPlotData, levels)
            self._sync_vispy_histogram_widget_bounds(levels, histogramRange=histogramRange)
        finally:
            if started_timing:
                self._finish_upload_timing(merge_with_previous=True)

    def _sync_vispy_histogram_widget_bounds(self, levels, *, histogramRange=None) -> None:
        applying = self._applying_presentation
        self._applying_presentation = True
        try:
            self.histogram.setLevels(float(levels[0]), float(levels[1]))
            if histogramRange is not None:
                self.histogram.setHistogramRange(float(histogramRange[0]), float(histogramRange[1]))
        finally:
            self._applying_presentation = applying

    def _vispy_histogram_can_flush_now(self) -> bool:
        if self._vispy_input_interactive():
            return False
        window = self.window()
        coordinator = getattr(window, "render_coordinator", None)
        return not bool(
            coordinator is not None and getattr(coordinator, "has_pending_render", False)
        )

    def _vispy_histogram_retry_interval_ms(self) -> int:
        window = self.window()
        decision_provider = getattr(window, "_gui_callback_budget_decision", None)
        if callable(decision_provider):
            decision = decision_provider("histogram_refresh", interactive=True)
            if decision is not None:
                return max(8, int(getattr(decision, "interval_ms", 16) or 16))
        return 16

    def _vispy_input_interactive(self) -> bool:
        window = self.window()
        coordinator = getattr(window, "render_coordinator", None)
        return bool(
            getattr(coordinator, "interactive_active", False)
            or getattr(window, "_viewport_interaction_active", False)
        )

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
        frame_plan=None,
    ):
        from arrayscope.display.model.tile_stats import TileLayerUpdateStats

        if geometry is None or (getattr(geometry, "montage", None) is None and frame_plan is None):
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
                frame_plan=frame_plan,
            )
        raise ValueError("VisPy montage presentation requires direct tile payloads")

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
        frame_plan=None,
    ):
        from arrayscope.display.model.tile_stats import TileLayerUpdateStats

        if geometry is None or (getattr(geometry, "montage", None) is None and frame_plan is None):
            return TileLayerUpdateStats()
        self._last_vispy_geometry = geometry
        payloads_by_tile = {
            int(tile): payload for tile, payload in dict(tile_payloads or {}).items()
        }
        if tile_delta is not None:
            active_set = {
                int(tile) for tile in tuple(getattr(tile_delta, "active_tiles", ()) or ())
            }
        else:
            active_set = set(payloads_by_tile)
        layer = getattr(self, "_vispy_gpu_montage_layer", None)
        if layer is None:
            return TileLayerUpdateStats()
        previous_presented = getattr(getattr(layer, "last_stats", None), "presented_tiles", None)
        presentation_complete = (
            previous_presented is None
            or {int(tile) for tile in tuple(previous_presented or ())} == active_set
        )
        # Field defect 2026-07-05 (stale wrong-LOD): the uniforms-only fast
        # path compares presented tile NUMBERS, but a zoom-driven level swap
        # changes payload IDENTITIES under the same numbers while also
        # changing levels (force_levels).  Taking the fast path then skips
        # the swap uploads entirely while the commit report acknowledges
        # them, wedging the layer on the old level until the next real
        # update (user-visible as coarse tiles a pan magically fixes).
        # Any payload work in the delta demands the full update path.
        delta_payload_work = tile_delta is not None and (
            bool(dict(getattr(tile_delta, "upserts", {}) or {}))
            or bool(tuple(getattr(tile_delta, "removals", ()) or ()))
        )
        if (
            (force_levels or force_mapping)
            and not delta_payload_work
            and getattr(layer, "last_stats", None).visible_items
            and presentation_complete
        ):
            stats = layer.set_presentation_uniforms(levels=levels, shader_mapping=shader_mapping)
        else:
            try:
                stats = layer.update(
                    payloads=payloads_by_tile,
                    geometry=geometry,
                    levels=levels,
                    dirty_tiles=dirty_tiles,
                    rgb_already_windowed=rgb_already_windowed,
                    shader_mapping=shader_mapping,
                    tile_delta=tile_delta,
                    tile_residency_budget_bytes=tile_residency_budget_bytes,
                    frame_plan=frame_plan,
                )
            except Exception as exc:
                from arrayscope.display.backends.vispy.tiles import AtlasCapacityError

                if not isinstance(exc, AtlasCapacityError):
                    raise
                previous = getattr(layer, "last_stats", None)
                stats = TileLayerUpdateStats(
                    visible_items=len(active_set),
                    presented_tiles=(),
                    resident_items=int(getattr(previous, "resident_items", 0) or 0),
                    storage_capacity=int(getattr(previous, "storage_capacity", 0) or 0),
                    estimated_gpu_bytes=int(getattr(previous, "estimated_gpu_bytes", 0) or 0),
                    cpu_shadow_bytes=int(getattr(previous, "cpu_shadow_bytes", 0) or 0),
                    page_count=int(getattr(previous, "page_count", 0) or 0),
                    active_pages=int(getattr(previous, "active_pages", 0) or 0),
                    device_max_texture_size=int(
                        getattr(previous, "device_max_texture_size", 0) or 0
                    ),
                    budget_bytes=int(tile_residency_budget_bytes),
                    capacity_warning=str(exc),
                )
        self._record_upload_timing("tile_layer_upload_ms", float(stats.upload_ms))
        return stats

    def _request_vispy_tile_layer_redraw(self) -> None:
        self._vispy_tile_presentation_request_count = (
            int(getattr(self, "_vispy_tile_presentation_request_count", 0) or 0) + 1
        )
        layer = getattr(self, "_vispy_gpu_montage_layer", None)
        visuals = tuple(getattr(layer, "_visuals_by_page", ()) or ())
        changed_pages = tuple(getattr(layer, "changed_page_indices", lambda: ())() or ())
        if changed_pages:
            candidates = (
                visuals[int(index)] for index in changed_pages if 0 <= int(index) < len(visuals)
            )
        else:
            candidates = ()
        for visual in candidates:
            if bool(getattr(visual, "visible", False)) and callable(
                getattr(visual, "update", None)
            ):
                with contextlib.suppress(Exception):
                    visual.update()
        self._request_vispy_canvas_update()

    def _request_vispy_canvas_update(self) -> None:
        canvas = getattr(self, "_vispy_canvas", None)
        if canvas is None:
            return
        # Do not use our draw-pending observation as an update-request latch.
        # Qt/VisPy may coalesce native update calls, but a request made while
        # painting can also be dropped.  Suppressing every later request until
        # a draw then creates an absorbing state: the draw is the only event
        # that can clear the flag, while the flag prevents the request that
        # would produce that draw.
        self._vispy_canvas_update_pending = True
        self._vispy_canvas_update_request_count = (
            int(getattr(self, "_vispy_canvas_update_request_count", 0) or 0) + 1
        )
        try:
            canvas.update()
        except Exception:
            self._vispy_canvas_update_pending = False

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
            self._vispy_view.camera.set_range(
                x=(float(x_range[0]), float(x_range[1])),
                y=(float(y_range[0]), float(y_range[1])),
                margin=0,
            )
            self._request_vispy_canvas_update()
        except Exception:
            pass

    def _request_vispy_camera_sync(self, *, immediate: bool = False) -> None:
        # A camera gesture has priority over speculative residency uploads.
        # The next settled tiled presentation will enqueue the relevant near
        # ring again, so discarding stale warm work is both safe and cheaper.
        self._vispy_pending_warm_tile_payloads = {}
        self._vispy_pending_warm_tile_context = {}
        if immediate:
            self._vispy_camera_sync_pending = False
            self._sync_vispy_camera_to_view()
            return
        if getattr(self, "_vispy_camera_sync_pending", False):
            return
        self._vispy_camera_sync_pending = True

        def apply_sync():
            self._vispy_camera_sync_pending = False
            self._sync_vispy_camera_to_view()

        # Timer category: UI cosmetic. Qt event-turn barrier. Multiple range changes collapse to one camera
        # sync; remove if VisPy exposes a direct safe same-turn camera update.
        QtCore.QTimer.singleShot(0, self, apply_sync)

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
        from vispy import gloo, scene
        from vispy.scene import visuals
        from vispy.scene.cameras import PanZoomCamera
        from vispy.visuals import transforms
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "VisPy rendering backend is not available. Install ArrayScope[vispy] or vispy."
        ) from exc
    return scene, visuals, transforms, PanZoomCamera, gloo


def _tiled_source_key(tile_payloads, tile_source_ids):
    if tile_payloads is None:
        return None
    ids = tile_source_ids or {}
    return tuple(
        (
            int(tile),
            ids.get(int(tile), getattr(payload, "source_id", None)),
            tile_ack_identity(payload),
        )
        for tile, payload in sorted(dict(tile_payloads).items())
    )


def _requested_direct_payload_tiles(tile_payloads, tile_delta) -> set[int]:
    payload_tiles = {int(tile) for tile in dict(tile_payloads or {})}
    if tile_delta is None:
        return payload_tiles
    active_tiles = {int(tile) for tile in tuple(getattr(tile_delta, "active_tiles", ()) or ())}
    return active_tiles


def _shader_mapping_key(mapping):
    return None if mapping is None else getattr(mapping, "identity_key", mapping)


def _vispy_tiled_placeholder_key(img) -> tuple[object, ...]:
    array = np.asarray(img)
    return (tuple(int(value) for value in array.shape), array.dtype.str)


def _tiled_structure_key(geometry, *, rgb_already_windowed, frame_plan=None):
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
        tuple(
            (
                int(region.region_id),
                tuple(float(value) for value in region.bounds),
            )
            for region in tuple(getattr(frame_plan, "regions", ()) or ())
        ),
        bool(rgb_already_windowed),
    )


def _tiled_histogram_key(histogram_range, *, histogramPlotData, tile_delta, semantic_identity):
    return tiled_histogram_key(
        histogram_range,
        histogram_plot_data=histogramPlotData,
        tile_delta=tile_delta,
        semantic_identity=semantic_identity,
    )


_tiled_semantic_histogram_identity = tiled_semantic_histogram_identity
_payload_histogram_source = payload_histogram_source


def _set_visual_visible(visual, visible: bool) -> None:
    if visual is None:
        return
    with contextlib.suppress(Exception):
        visual.visible = bool(visible)


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
    return (*tuple(float(value) / 255.0 for value in rgb), 1.0)


def _vispy_roi_points(geometry):
    points = roi_outline_points(geometry)
    return None if not points else np.asarray(points, dtype=np.float32)


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
        mark_points = np.asarray(_overlay_status_mark_points(overlay), dtype=np.float32).reshape(
            (-1, 2)
        )
        for point in mark_points:
            points.append((float(point[0]), float(point[1])))
            colors.append(mark)
    return (
        np.asarray(points, dtype=np.float32).reshape((-1, 2)),
        np.asarray(colors, dtype=np.float32).reshape((-1, 4)),
    )


def _overlay_vispy_colors(overlay):
    return montage_overlay_rgba(overlay)


_histogram_data_from_tile_payloads = histogram_data_from_tile_payloads


def _overlay_status_mark_points(overlay):
    return np.asarray(
        [point for segment in montage_overlay_status_segments(overlay) for point in segment],
        dtype=np.float32,
    )
