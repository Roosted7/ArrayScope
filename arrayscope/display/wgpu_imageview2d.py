"""Experimental wgpu-backed 2D image view (MVP: scalar 2-D non-montage).

Queue row 3 slice (b): the smallest live rendering backend driven purely by
the renderer command protocol (ADR 0057).  The widget mirrors the VisPy
hybrid exactly at the shell seam — PyQtGraph keeps the histogram widget and
the transparent interaction overlay; a rendercanvas ``QRenderWidget`` in
bitmap present mode owns the pixels — but every pixel decision is expressed
as :class:`~arrayscope.gpu.command_protocol.FrameSubmission` commands into
one :class:`~arrayscope.gpu.wgpu_executor.WgpuPlaneExecutor`.

Scope is deliberately narrow and loud: exactly one scalar 2-D tile (the
non-montage plane the single-tile geometry commits).  Complex, RGB, and
montage (>1 tile) commits raise ``NotImplementedError`` instead of guessing.
Residency is content-keyed (``plane_chunk_key`` with the payload's ack
identity as ``document_generation``), so re-committing identical content is
a physical zero-upload no-op — the executor report is the oracle, and the
commit stats are derived from it, never invented.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from arrayscope.display.backend_contract import WGPU_CAPABILITIES
from arrayscope.display.backends.pyqtgraph.histogram_adapter import PyQtGraphHistogramAdapter
from arrayscope.display.imageview2d import ArrayScopeGraphicsView, ImageViewShell
from arrayscope.display.model.tile_identity import tile_ack_identity
from arrayscope.display.model.tile_stats import TileLayerUpdateStats
from arrayscope.display.shader_mapping import ShaderDisplayMode, common_shader_mapping
from arrayscope.display.view_navigation_driver import QtViewNavigationDriver
from arrayscope.gpu.command_protocol import (
    DisplayMapping,
    EnsureChunkResident,
    EvictChunk,
    FrameSubmission,
    PresentGeneration,
    SetDisplayMapping,
    TileInstance,
    UpdateTileInstances,
)

# One process-wide wgpu device: views (and executor rebuilds on plane growth)
# share it so the canvas context never needs reconfiguration and tests do not
# pay per-view device creation.
_SHARED_WGPU_DEVICE = None


def import_qrenderwidget():
    """Import rendercanvas's Qt widget without leaking its env mutations.

    On Wayland hosts rendercanvas stomps process env vars at import time
    (``QT_QPA_PLATFORM=xcb``, ``GDK_BACKEND=x11``).  With a live QApplication
    that cannot change the running Qt platform, but it silently poisons every
    later env reader in the process — e.g. the AUTO-backend probe's offscreen
    check resolved to VisPy inside offscreen test runs.  Every rendercanvas
    import (view construction AND test availability probes) must go through
    this helper so the snapshot is taken before the first import.
    """

    import os

    keys = ("QT_QPA_PLATFORM", "GDK_BACKEND", "PYGLFW_LIBRARY_VARIANT")
    before = {key: os.environ.get(key) for key in keys}
    try:
        from rendercanvas.pyside6 import QRenderWidget
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return QRenderWidget


def _shared_wgpu_device():
    global _SHARED_WGPU_DEVICE
    if _SHARED_WGPU_DEVICE is None:
        import wgpu
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        try:
            # Vulkan-only instance: the GL backend's EGL re-init is fatal under
            # Wayland (gate-B Tier 0).  Harmless if the instance already exists.
            set_instance_extras(backends=["Vulkan"])
        except RuntimeError:
            pass
        adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
        _SHARED_WGPU_DEVICE = adapter.request_device_sync()
    return _SHARED_WGPU_DEVICE


class WgpuImageView2D(ImageViewShell):
    """ImageView2D variant that renders pixels through the wgpu executor.

    The public ImageView2D API is preserved; the shell owns interaction,
    ROI/profile semantics, histogram plumbing, and the ViewBox camera truth.
    This class owns only how committed tile payloads become protocol commands
    and how the ViewBox range maps to normalized canvas tile geometry.
    """

    rendering_capabilities = WGPU_CAPABILITIES
    draws_qgraphics_roi_items = False
    draws_qgraphics_profile_marker_items = False

    _level_preview_timing_channel = "wgpu_level_preview"

    def setupUI(self):
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self._display_container = QtWidgets.QWidget()
        self._display_stack = QtWidgets.QStackedLayout(self._display_container)
        self._display_stack.setContentsMargins(0, 0, 0, 0)
        self._display_stack.setStackingMode(QtWidgets.QStackedLayout.StackingMode.StackAll)

        # Imported lazily: a QApplication exists by now, so rendercanvas's
        # import-time environment fiddling cannot change the live Qt platform.
        QRenderWidget = import_qrenderwidget()

        self._wgpu_canvas = QRenderWidget(
            parent=self._display_container,
            present_method="bitmap",
            update_mode="ondemand",
        )
        self._wgpu_canvas.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._display_stack.addWidget(self._wgpu_canvas)

        self.graphicsView = ArrayScopeGraphicsView(self)
        self.graphicsView.setBackground(None)
        self.graphicsView.setStyleSheet("background: transparent; border: 0px;")
        self.graphicsView.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.graphicsView.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.graphicsView.viewport().setStyleSheet("background: transparent;")
        self._display_stack.addWidget(self.graphicsView)
        self.layout.addWidget(self._display_container, 1)

        self.histogram = pg.HistogramLUTWidget()
        self.layout.addWidget(self.histogram)
        self._histogram_adapter = PyQtGraphHistogramAdapter(self.histogram)

        # Executor / protocol state.  The context must exist before any draw:
        # rendercanvas cancels draw events while the canvas has no context
        # (configuration is still deferred until an executor provides the
        # device in the first real draw).
        self._wgpu_executor = None
        self._wgpu_generation = 0
        self._wgpu_context = self._wgpu_canvas.get_context("wgpu")
        self._wgpu_context_format = None
        self._wgpu_mapping_state = DisplayMapping(mode="real")
        self._wgpu_committed: dict[str, object] | None = None
        self._wgpu_last_report_uploads = 0
        self._wgpu_last_draw_error: str = ""
        # Draw-ack discipline (mirrors VisPy's request/draw counters exactly).
        self._wgpu_draw_count = 0
        self._wgpu_tile_presentation_request_count = 0
        self._wgpu_tile_presentation_draw_count = 0
        self._wgpu_canvas_update_request_count = 0
        self._wgpu_canvas_update_pending = False
        self._wgpu_display_shape: tuple[int, int] = (1, 1)
        self._wgpu_last_levels: tuple[float, float] = (0.0, 1.0)
        self._last_wgpu_structure_key = None
        self._last_wgpu_viewport_key = None

        self._wgpu_canvas.request_draw(self._on_wgpu_draw)

    def __init__(self, parent=None, view=None, imageItem=None):
        super().__init__(parent=parent, view=view, imageItem=imageItem)
        self._view_navigation = QtViewNavigationDriver(self)
        self.imageItem.setVisible(False)
        self.histogramImageItem.setVisible(False)
        self._wgpu_bounds_item = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
        self._wgpu_bounds_item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self._wgpu_bounds_item.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
        self._wgpu_bounds_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self._layer_owner.add_bounds_item(self._wgpu_bounds_item)
        self.view.sigRangeChanged.connect(lambda *_args: self._request_wgpu_canvas_draw())

    # ---- executor management -------------------------------------------------

    def _ensure_wgpu_executor(self, pixel_shape: tuple[int, int]):
        from arrayscope.gpu.wgpu_executor import PAGE, WgpuPlaneExecutor

        height, width = (max(1, int(value)) for value in pixel_shape)
        padded_h = -(-height // PAGE) * PAGE
        padded_w = -(-width // PAGE) * PAGE
        executor = self._wgpu_executor
        if (
            executor is not None
            and executor.plane_shape[0] >= padded_h
            and executor.plane_shape[1] >= padded_w
        ):
            return executor
        if executor is not None:
            padded_h = max(padded_h, executor.plane_shape[0])
            padded_w = max(padded_w, executor.plane_shape[1])
        grid = (padded_h // PAGE) * (padded_w // PAGE)
        self._wgpu_executor = WgpuPlaneExecutor(
            (padded_h, padded_w),
            max_lod=0,
            pool_layers=2 * grid + 8,
            device=_shared_wgpu_device(),
        )
        # Rebuild discards residency; committed evidence must not survive it.
        self._wgpu_committed = None
        return self._wgpu_executor

    def _next_wgpu_generation(self) -> int:
        self._wgpu_generation += 1
        return self._wgpu_generation

    def _submit_wgpu(self, commands, *, present_to=None, present_format="rgba8unorm"):
        executor = self._wgpu_executor
        if executor is None:
            return None
        generation = self._next_wgpu_generation()
        submission = FrameSubmission(generation, (*commands, PresentGeneration(generation)))
        if present_to is None:
            return executor.submit(submission)
        return executor.submit(submission, present_to=present_to, present_format=present_format)

    # ---- draw-ack discipline -------------------------------------------------

    def _request_wgpu_canvas_draw(self, *, count_presentation: bool = False) -> None:
        if count_presentation:
            self._wgpu_tile_presentation_request_count = int(
                getattr(self, "_wgpu_tile_presentation_request_count", 0) or 0
            ) + 1
        canvas = getattr(self, "_wgpu_canvas", None)
        if canvas is None:
            return
        if bool(getattr(self, "_wgpu_canvas_update_pending", False)) and not count_presentation:
            return
        self._wgpu_canvas_update_pending = True
        self._wgpu_canvas_update_request_count = int(
            getattr(self, "_wgpu_canvas_update_request_count", 0) or 0
        ) + 1
        try:
            canvas.request_draw()
        except Exception:
            self._wgpu_canvas_update_pending = False

    def _on_wgpu_draw(self, *_args) -> None:
        self._wgpu_draw_count = int(getattr(self, "_wgpu_draw_count", 0) or 0) + 1
        self._wgpu_canvas_update_pending = False
        self._wgpu_tile_presentation_draw_count = int(
            getattr(self, "_wgpu_tile_presentation_request_count", 0) or 0
        )
        try:
            self._present_wgpu_frame_to_canvas()
            self._wgpu_last_draw_error = ""
        except Exception as exc:  # keep the Qt paint loop alive; surface in diagnostics
            self._wgpu_last_draw_error = f"{type(exc).__name__}: {exc}"
        # Timer category: anti-hang fallback (same rationale as VisPy's draw
        # ack): a presentationDrawn listener may immediately commit the next
        # band; emitting from inside the draw callback can drop its canvas
        # update and leave the logical draw gate armed forever.  Publish the
        # physical acknowledgement after the draw callback returns.
        drawn_request_count = int(self._wgpu_tile_presentation_draw_count)
        QtCore.QTimer.singleShot(
            0,
            self,
            lambda count=drawn_request_count: self._publish_wgpu_draw_ack(count),
        )

    def _publish_wgpu_draw_ack(self, drawn_request_count: int) -> None:
        if int(getattr(self, "_wgpu_tile_presentation_draw_count", 0) or 0) < int(
            drawn_request_count
        ):
            return
        self._mark_presentation_drawn()

    def _present_wgpu_frame_to_canvas(self) -> None:
        executor = self._wgpu_executor
        if executor is None:
            return
        if self._wgpu_context_format is None:
            preferred = self._wgpu_context.get_preferred_format(None)
            fmt = preferred.removesuffix("-srgb")
            self._wgpu_context.configure(device=executor.device, format=fmt)
            self._wgpu_context_format = fmt
        tiles = self._wgpu_camera_tiles()
        self._submit_wgpu(
            (
                SetDisplayMapping(self._wgpu_mapping_state),
                UpdateTileInstances(tiles),
            ),
            present_to=self._wgpu_context.get_current_texture().create_view(),
            present_format=self._wgpu_context_format,
        )

    def _wgpu_camera_tiles(self) -> tuple[TileInstance, ...]:
        """Map the committed world rect through the ViewBox range to dst space.

        The executor's dst space is normalized [0, 1] with y down.  The
        ViewBox is the camera truth: ``viewRange`` returns sorted world
        bounds; ``yInverted`` (the image default) puts world y-min at the
        top of the canvas, which is already dst-y-down order.
        """

        committed = self._wgpu_committed
        if not committed or self._montage_display_mode != "wgpu_tile_layer":
            return ()
        try:
            (x0, x1), (y0, y1) = self.view.viewRange()
        except Exception:
            return ()
        span_x = float(x1) - float(x0)
        span_y = float(y1) - float(y0)
        if not (span_x > 0.0 and span_y > 0.0):
            return ()
        wx, wy, ww, wh = committed["world_rect"]
        src_w, src_h = committed["src_size"]
        state = getattr(self.view, "state", {}) or {}
        y_inverted = bool(state.get("yInverted", True))
        dst_x = (wx - float(x0)) / span_x
        dst_w = ww / span_x
        dst_h = wh / span_y
        if y_inverted:
            dst_y = (wy - float(y0)) / span_y
            src_origin = (0.0, 0.0)
            src_size = (float(src_w), float(src_h))
        else:
            dst_y = (float(y1) - (wy + wh)) / span_y
            src_origin = (0.0, float(src_h))
            src_size = (float(src_w), -float(src_h))
        return (
            TileInstance((dst_x, dst_y, dst_w, dst_h), src_origin, src_size, 0),
        )

    # ---- tiled presentation --------------------------------------------------

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
        montage_tile_payloads: dict[int, object] | None,
        shader_mapping,
        tile_delta,
        tile_residency_budget_bytes: int,
        frame_plan,
    ):
        from arrayscope.gpu.wgpu_executor import PAGE, plane_chunk_key

        self._start_upload_timing("wgpu_tile_layer")
        applying = self._applying_presentation
        self._applying_presentation = True
        try:
            payloads = {int(tile): payload for tile, payload in dict(montage_tile_payloads or {}).items()}
            if not payloads:
                # Loading-only commit: nothing drawable yet; clear stale tiles.
                self._clear_wgpu_tiles()
                self.image = img
                self._montage_display_mode = "wgpu_tile_layer"
                stats = TileLayerUpdateStats(visible_items=0, presented_tiles=())
                self._record_tile_layer_stats(stats)
                return stats
            self._reject_unsupported_wgpu_commit(payloads, shader_mapping)
            payload = payloads[0]
            plane = np.asarray(payload.image)
            height, width = (int(plane.shape[0]), int(plane.shape[1]))
            identity = tile_ack_identity(payload)
            level_lo, level_hi = (float(levels[0]), float(levels[1]))
            if not level_hi > level_lo:
                level_hi = level_lo + 1e-6

            executor = self._ensure_wgpu_executor((height, width))
            grid_h, grid_w = -(-height // PAGE), -(-width // PAGE)
            plane32 = np.ascontiguousarray(plane, dtype=np.float32)
            commands = []
            page_keys = []
            for chunk_y in range(grid_h):
                for chunk_x in range(grid_w):
                    key = plane_chunk_key(identity, "live", 0, chunk_x, chunk_y)
                    page_keys.append(key)
                    page = np.zeros((PAGE, PAGE), np.float32)
                    block = plane32[
                        chunk_y * PAGE : (chunk_y + 1) * PAGE,
                        chunk_x * PAGE : (chunk_x + 1) * PAGE,
                    ]
                    page[: block.shape[0], : block.shape[1]] = block
                    # Content-keyed: re-ensuring a resident key is a no-op in
                    # the executor (0 uploads); the frame report is the oracle.
                    commands.append(EnsureChunkResident(key, page))

            lut = self._wgpu_resolve_lut_bytes(shader_mapping)
            self._wgpu_mapping_state = DisplayMapping(
                mode="real", level_lo=level_lo, level_hi=level_hi, lut=lut
            )
            display_shape = tuple(int(value) for value in tuple(geometry.display_shape)[:2])
            self._wgpu_committed = {
                "identity": identity,
                "world_rect": (0.0, 0.0, float(width), float(height)),
                "src_size": (float(width), float(height)),
                "page_keys": tuple(page_keys),
                "pixel_shape": (height, width),
            }
            self._montage_display_mode = "wgpu_tile_layer"

            start = perf_counter()
            report = self._submit_wgpu(
                (
                    *commands,
                    SetDisplayMapping(self._wgpu_mapping_state),
                    UpdateTileInstances(self._wgpu_camera_tiles()),
                )
            )
            upload_ms = (perf_counter() - start) * 1000.0
            self._wgpu_last_report_uploads = int(report.uploads)

            # Physical truth: acknowledge only what the page table holds.
            resident = all(executor.page_table.lookup(key) is not None for key in page_keys)
            presented = (0,) if resident else ()
            presented_identities = {0: identity} if resident else {}

            # Shared shell bookkeeping (placeholder image, histogram bounds,
            # display levels) mirrors the VisPy backend's minimal set.
            self.image = img
            self.histogramSource = None
            if not callable(histogramPlotData):
                self.histogramPlotSource = histogramPlotData
            self.setHistogramDataBounds(histogramRange)
            self._displayLevels = (level_lo, level_hi)
            self._wgpu_last_levels = (level_lo, level_hi)
            self._sync_wgpu_histogram_widget_bounds((level_lo, level_hi), histogramRange)

            structure_key = (display_shape, bool(rgb_already_windowed))
            viewport_key = (
                structure_key,
                str(getattr(viewport_policy, "value", viewport_policy)),
            )
            if structure_key != self._last_wgpu_structure_key:
                self._sync_wgpu_bounds(display_shape)
                self._update_profile_line_bounds()
                self._updateAspectRatio()
                self._last_wgpu_structure_key = structure_key
            if viewport_key != self._last_wgpu_viewport_key:
                self._apply_viewport_policy(display_shape, viewport_policy, image_origin=(0.0, 0.0))
                self._last_wgpu_viewport_key = viewport_key

            uploads = int(report.uploads)
            resident_pages = len(executor.page_table.resident_keys())
            stats = TileLayerUpdateStats(
                visible_items=1,
                presented_tiles=presented,
                presented_identities=presented_identities,
                updated_tiles=(0,) if uploads > 0 else (),
                items_created=0,
                items_updated=1 if uploads > 0 else 0,
                items_skipped=0 if uploads > 0 else 1,
                existing_items_shown=0 if uploads > 0 else 1,
                resident_items=1 if resident else 0,
                storage_capacity=len(executor._free_layers) + resident_pages,
                texture_uploads=uploads,
                texture_upload_bytes=uploads * PAGE * PAGE * 8,
                page_count=resident_pages,
                active_pages=len(page_keys),
                estimated_gpu_bytes=resident_pages * PAGE * PAGE * 8,
                budget_bytes=int(tile_residency_budget_bytes or 0),
                shader_uniform_updates=1,
                upload_ms=upload_ms,
            )
            self._record_tile_layer_stats(stats)
            self._record_upload_timing("tile_layer_upload_ms", upload_ms)
            self._request_wgpu_canvas_draw(count_presentation=True)
            return stats
        finally:
            self._applying_presentation = applying
            self._finish_upload_timing()

    def _reject_unsupported_wgpu_commit(self, payloads, shader_mapping) -> None:
        if len(payloads) != 1 or 0 not in payloads:
            raise NotImplementedError(
                "wgpu backend MVP renders exactly one non-montage tile; "
                f"got tiles {sorted(payloads)}"
            )
        payload = payloads[0]
        plane = np.asarray(payload.image)
        if np.iscomplexobj(plane) or plane.ndim != 2:
            raise NotImplementedError(
                "wgpu backend MVP renders scalar 2-D planes only; "
                f"got dtype {plane.dtype} with shape {plane.shape}"
            )
        mapping = shader_mapping
        if mapping is None:
            mapping = common_shader_mapping(
                getattr(candidate, "shader_mapping", None) for candidate in payloads.values()
            )
        display_mode = getattr(getattr(mapping, "display_mode", None), "value", None)
        if display_mode not in (None, ShaderDisplayMode.SCALAR.value):
            raise NotImplementedError(
                f"wgpu backend MVP supports scalar display mode only; got {display_mode!r}"
            )

    def _wgpu_resolve_lut_bytes(self, shader_mapping) -> bytes | None:
        lut = getattr(shader_mapping, "lut_data", None)
        if lut is None:
            lut = getattr(self, "_display_colormap_lut", None)
            if getattr(self, "_display_colormap", None) is None:
                # Default grayscale colormap == the executor's neutral ramp.
                return None
        return _resample_lut_to_rgba256(lut)

    def _sync_wgpu_histogram_widget_bounds(self, levels, histogram_range) -> None:
        applying = self._applying_presentation
        self._applying_presentation = True
        try:
            self.histogram.setLevels(float(levels[0]), float(levels[1]))
            if histogram_range is not None:
                self.histogram.setHistogramRange(
                    float(histogram_range[0]), float(histogram_range[1])
                )
        finally:
            self._applying_presentation = applying

    def _clear_wgpu_tiles(self) -> None:
        if self._wgpu_executor is None:
            return
        self._submit_wgpu((UpdateTileInstances(()),))
        self._request_wgpu_canvas_draw()

    # ---- hide / invalidate / reset -------------------------------------------

    def clearMontageTileLayer(self) -> None:
        self.hide_tiled_presentation("surface-reset")

    def hide_tiled_presentation(self, reason: str) -> None:
        self._wgpu_committed = None
        self._clear_wgpu_tiles()
        self.clearMontageTileOverlays()
        self._montage_display_mode = "none"
        self.imageItem.setVisible(False)

    def invalidate_tiled_presentation(self, reason: str, *, hide_pixels: bool = True) -> None:
        """Hide superseded pixels; executor page-table residency is retained.

        ``hide_pixels=False`` keeps drawing the stale-but-honest previous
        plane; the successor's first commit computes a different content key
        and replaces the drawn tile atomically.
        """

        if not hide_pixels:
            return
        self._wgpu_committed = None
        self._clear_wgpu_tiles()
        self._montage_display_mode = "none"
        self.imageItem.setVisible(False)

    def reset_tiled_residency(self, reason: str) -> None:
        executor = self._wgpu_executor
        if executor is not None:
            evictions = tuple(
                EvictChunk(key) for key in executor.page_table.resident_keys()
            )
            self._submit_wgpu((*evictions, UpdateTileInstances(())))
            self._request_wgpu_canvas_draw()
        self._wgpu_committed = None
        self._last_wgpu_structure_key = None
        self._last_wgpu_viewport_key = None
        self._last_wgpu_tiled_reset_reason = str(reason)
        self._montage_display_mode = "none"
        self.imageItem.setVisible(False)

    def clear(self):
        super().clear()
        self.reset_tiled_residency("view-clear")

    def warmTiledResidency(self, **_kwargs):
        """No persistent-residency contract: warming is declared unavailable."""

        return None

    def warmPlaneResidency(self, payload) -> bool:
        return False

    def _tiled_presentation_layer(self):
        return None

    # ---- levels / LUT ---------------------------------------------------------

    def _apply_preview_levels_to_display(self, levels, *, final: bool) -> None:
        self._wgpu_last_levels = (float(levels[0]), float(levels[1]))
        if self._montage_display_mode != "wgpu_tile_layer" or self._wgpu_executor is None:
            return
        level_lo, level_hi = (float(levels[0]), float(levels[1]))
        if not level_hi > level_lo:
            level_hi = level_lo + 1e-6
        self._wgpu_mapping_state = DisplayMapping(
            mode=self._wgpu_mapping_state.mode,
            level_lo=level_lo,
            level_hi=level_hi,
            lut=self._wgpu_mapping_state.lut,
        )
        start = perf_counter()
        report = self._submit_wgpu((SetDisplayMapping(self._wgpu_mapping_state),))
        upload_ms = (perf_counter() - start) * 1000.0
        committed = self._wgpu_committed
        stats = TileLayerUpdateStats(
            visible_items=1 if committed else 0,
            presented_tiles=(0,) if committed else (),
            items_skipped=1 if committed else 0,
            texture_uploads=int(getattr(report, "uploads", 0) or 0),
            level_updates=1,
            shader_uniform_updates=1,
            upload_ms=upload_ms,
        )
        self._record_tile_layer_stats(stats)
        self._request_wgpu_canvas_draw(count_presentation=True)
        handler = getattr(self, "_level_presentation_change_handler", None)
        if callable(handler):
            handler((level_lo, level_hi), final=bool(final))

    def _backend_display_lut_changed(self, lut: np.ndarray) -> None:
        if self._wgpu_executor is None:
            return
        self._wgpu_mapping_state = DisplayMapping(
            mode=self._wgpu_mapping_state.mode,
            level_lo=self._wgpu_mapping_state.level_lo,
            level_hi=self._wgpu_mapping_state.level_hi,
            lut=_resample_lut_to_rgba256(lut),
        )
        if self._montage_display_mode == "wgpu_tile_layer":
            self._submit_wgpu((SetDisplayMapping(self._wgpu_mapping_state),))
            self._request_wgpu_canvas_draw(count_presentation=True)

    # ---- camera / geometry ----------------------------------------------------

    def _sync_backend_camera_to_view(self) -> None:
        self._request_wgpu_canvas_draw()

    def _after_viewport_camera_change(self) -> None:
        self._request_wgpu_canvas_draw()

    def _sync_wgpu_bounds(self, image_shape, *, image_origin=(0.0, 0.0)) -> None:
        if getattr(self, "_wgpu_bounds_item", None) is None:
            return
        height, width = tuple(int(value) for value in image_shape[:2])
        self._wgpu_display_shape = (max(1, height), max(1, width))
        self._wgpu_bounds_item.setRect(
            QtCore.QRectF(
                float(image_origin[0]),
                float(image_origin[1]),
                float(max(1, width)),
                float(max(1, height)),
            )
        )

    def _viewport_content_shape(self):
        extent = getattr(self, "_viewport_content_extent", None)
        if extent is not None:
            return extent
        return getattr(self, "_wgpu_display_shape", None) or self.image.shape[:2]

    def _current_image_world_rect(self):
        bounds = getattr(self, "_wgpu_bounds_item", None)
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
        bounds = getattr(self, "_wgpu_bounds_item", None)
        if bounds is None:
            return super()._current_image_viewport_rect()
        rect = bounds.rect()
        return (
            float(rect.left()),
            float(rect.top()),
            float(rect.left() + max(1.0, rect.width())),
            float(rect.top() + max(1.0, rect.height())),
        )

    # ---- diagnostics / lifecycle ----------------------------------------------

    def _paints_qgraphics_scene(self) -> bool:
        return False

    def _wgpu_tile_presentation_draw_pending(self) -> bool:
        return int(getattr(self, "_wgpu_tile_presentation_draw_count", 0) or 0) < int(
            getattr(self, "_wgpu_tile_presentation_request_count", 0) or 0
        )

    def presentationDrawPending(self) -> bool:
        return bool(
            super().presentationDrawPending()
            or getattr(self, "_wgpu_canvas_update_pending", False)
            or self._wgpu_tile_presentation_draw_pending()
        )

    def wgpuPresentationDiagnostics(self) -> dict[str, object]:
        executor = self._wgpu_executor
        drawn_tiles = tuple(getattr(executor, "_tiles", ()) or ()) if executor is not None else ()
        resident = len(executor.page_table.resident_keys()) if executor is not None else 0
        return {
            "draw_count": int(getattr(self, "_wgpu_draw_count", 0) or 0),
            "tile_presentation_request_count": int(
                getattr(self, "_wgpu_tile_presentation_request_count", 0) or 0
            ),
            "tile_presentation_draw_count": int(
                getattr(self, "_wgpu_tile_presentation_draw_count", 0) or 0
            ),
            "tile_presentation_draw_pending": self._wgpu_tile_presentation_draw_pending(),
            "canvas_update_request_count": int(
                getattr(self, "_wgpu_canvas_update_request_count", 0) or 0
            ),
            "canvas_update_pending": bool(getattr(self, "_wgpu_canvas_update_pending", False)),
            "physically_visible_tile_count": len(drawn_tiles),
            "page_table_resident_count": resident,
            "wgpu_uploads_total": int(getattr(executor, "uploads_total", 0) or 0),
            "wgpu_last_report_uploads": int(getattr(self, "_wgpu_last_report_uploads", 0) or 0),
            "wgpu_last_draw_error": str(getattr(self, "_wgpu_last_draw_error", "") or ""),
        }

    def presentation_diagnostics(self) -> dict[str, object]:
        diagnostics = super().presentation_diagnostics()
        diagnostics.update(self.wgpuPresentationDiagnostics())
        diagnostics["interaction_event_owner"] = self.interaction_event_owner()
        return diagnostics

    def reset_surface(self, reason: str) -> None:
        super().reset_surface(reason)
        self._wgpu_canvas_update_pending = False

    def teardown_surface(self) -> None:
        if getattr(self, "_surface_teardown_done", False):
            return
        canvas = getattr(self, "_wgpu_canvas", None)
        if canvas is not None:
            try:
                canvas.close()
            except Exception:
                pass
        super().teardown_surface()


def _resample_lut_to_rgba256(lut) -> bytes | None:
    """Resample an (N, 3|4) display LUT to the protocol's 256 RGBA8 entries."""

    if lut is None:
        return None
    from arrayscope.display.shader_mapping import normalize_lut_rgb

    rgb = normalize_lut_rgb(lut)  # (N, 3) uint8, validates shape/dtype
    indices = np.clip(
        np.round(np.linspace(0.0, rgb.shape[0] - 1.0, 256)).astype(np.intp),
        0,
        rgb.shape[0] - 1,
    )
    rgba = np.empty((256, 4), np.uint8)
    rgba[:, :3] = rgb[indices]
    rgba[:, 3] = 255
    return rgba.tobytes()
