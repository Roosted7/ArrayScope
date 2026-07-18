"""Experimental wgpu-backed 2D image view (montage scalar/RGB + complex).

Queue row 3 slice (b): a live rendering backend driven purely by the
renderer command protocol (ADR 0057).  The widget mirrors the VisPy hybrid
exactly at the shell seam — PyQtGraph keeps the histogram widget and the
transparent interaction overlay; a rendercanvas ``QRenderWidget`` in bitmap
present mode owns the pixels — but every pixel decision is expressed as
:class:`~arrayscope.gpu.command_protocol.FrameSubmission` commands into one
:class:`~arrayscope.gpu.wgpu_executor.WgpuPlaneExecutor`.

Committed scope (everything else raises ``NotImplementedError`` loudly
instead of guessing): montages of N scalar 2-D tiles, montages of
display-ready uint8 RGB tiles (``rgb_already_windowed``: levels/LUT
bypassed by the executor's rgb8 pool), and a single complex tile with the
shader-on-read component modes (magnitude/phase/real/imag) including the
phase LUT.  Each tile is one bound :class:`ContentPlane` whose
``document_generation`` is the payload's ack identity, so residency is
content-keyed: re-committing identical content, switching complex modes,
and moving levels are physical zero-upload operations — the executor report
is the oracle, and the commit stats are derived from it, never invented.
Acknowledgement is physical truth per tile: a tile enters
``presented_tiles``/``presented_identities`` only when every one of its
pages is actually resident in the executor page table after the submit.
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
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    ShaderScale,
    TexturePlaneKind,
    common_shader_mapping,
    default_phase_lut,
    pack_texture_data,
)
from arrayscope.display.tile_layout import tile_layout_map, tile_layout_shape
from arrayscope.display.view_navigation_driver import QtViewNavigationDriver
from arrayscope.gpu.command_protocol import (
    BindContentPlanes,
    ContentPlane,
    DisplayMapping,
    EnsureChunkResident,
    EvictChunk,
    FrameSubmission,
    PresentGeneration,
    SetDisplayMapping,
    TileInstance,
    UpdateTileInstances,
)
from arrayscope.gpu.keys import COMPLEX_RG32F, RGB8, SCALAR_R32F

#: Complex shader components → protocol mapping modes.
_WGPU_COMPONENT_MODES = {
    ShaderComponent.REAL.value: "real",
    ShaderComponent.IMAG.value: "imag",
    ShaderComponent.ABS.value: "magnitude",
    ShaderComponent.ANGLE.value: "phase",
    ShaderComponent.COMPLEX_PHASE.value: "phase",
}

_WGPU_REP_BY_KIND = {
    TexturePlaneKind.SCALAR_R32F: SCALAR_R32F,
    TexturePlaneKind.COMPLEX_RG32F: COMPLEX_RG32F,
    TexturePlaneKind.RGB8: RGB8,
}

_WGPU_REP_DTYPES = {SCALAR_R32F: "float32", COMPLEX_RG32F: "complex64", RGB8: "uint8"}
_WGPU_REP_TEXEL_BYTES = {SCALAR_R32F: 4, COMPLEX_RG32F: 8, RGB8: 4}

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

    def _ensure_wgpu_executor(self, required_pages: dict[str, int]):
        """Executor with per-pool budgets covering ``required_pages`` (+headroom)."""

        from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

        executor = self._wgpu_executor
        if executor is not None and all(
            executor.pool_budget(representation) >= needed
            for representation, needed in required_pages.items()
        ):
            return executor
        budgets: dict[str, int] = {}
        for representation in (SCALAR_R32F, COMPLEX_RG32F, RGB8):
            previous = 0 if executor is None else executor.pool_budget(representation)
            needed = int(required_pages.get(representation, 0))
            # 2x headroom keeps recently unbound planes warm (scroll-back).
            budget = max(previous, 2 * needed + 8 if needed else 0)
            if budget:
                budgets[representation] = budget
        self._wgpu_executor = WgpuPlaneExecutor(
            pool_layers=budgets or {SCALAR_R32F: 8},
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
        """Map committed per-tile world rects through the ViewBox to dst space.

        The executor's dst space is normalized [0, 1] with y down.  The
        ViewBox is the camera truth: ``viewRange`` returns sorted world
        bounds; ``yInverted`` (the image default) puts world y-min at the
        top of the canvas, which is already dst-y-down order.  Tile world
        rects come from the shared montage layout (``tile_layout_map``), so
        both drawn geometry and interaction mapping share one owner.
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
        state = getattr(self.view, "state", {}) or {}
        y_inverted = bool(state.get("yInverted", True))
        instances = []
        for tile in sorted(committed["tiles"]):
            info = committed["tiles"][tile]
            wx, wy, ww, wh = info["world_rect"]
            src_w, src_h = info["src_size"]
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
            instances.append(
                TileInstance(
                    (dst_x, dst_y, dst_w, dst_h),
                    src_origin,
                    src_size,
                    0,
                    plane_index=int(info["plane_index"]),
                )
            )
        return tuple(instances)

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
            source_mapping = shader_mapping
            if source_mapping is None:
                source_mapping = common_shader_mapping(
                    getattr(payload, "shader_mapping", None) for payload in payloads.values()
                )
            representation, mode = self._wgpu_commit_plan(
                payloads, source_mapping, rgb_already_windowed
            )
            layout = tile_layout_map(geometry, frame_plan=frame_plan)
            missing = sorted(set(payloads) - set(layout))
            if missing:
                raise NotImplementedError(
                    f"wgpu commit payload tiles {missing} have no montage/frame-plan "
                    "layout region — refusing to guess destination geometry"
                )
            level_lo, level_hi = (float(levels[0]), float(levels[1]))
            if not level_hi > level_lo:
                level_hi = level_lo + 1e-6

            textures = {
                tile: self._wgpu_payload_texture(payloads[tile], representation)
                for tile in payloads
            }
            pages_needed = sum(
                (-(-texture.shape[0] // PAGE)) * (-(-texture.shape[1] // PAGE))
                for texture in textures.values()
            )
            executor = self._ensure_wgpu_executor({representation: pages_needed})

            # One bound content plane per tile: document_generation is the
            # payload's ack identity, so residency is content-keyed and a
            # re-commit of previously seen content (scroll-back across
            # planes) is a physical zero-upload rebind.
            planes = []
            committed_tiles: dict[int, dict[str, object]] = {}
            commands = []
            planned_upload_tiles = []
            for tile in sorted(payloads):
                payload = payloads[tile]
                identity = tile_ack_identity(payload)
                texture = textures[tile]
                height, width = (int(texture.shape[0]), int(texture.shape[1]))
                grid_h, grid_w = -(-height // PAGE), -(-width // PAGE)
                plane_index = len(planes)
                planes.append(
                    ContentPlane(
                        identity,
                        "live",
                        (grid_h * PAGE, grid_w * PAGE),
                        max_lod=0,
                        representation=representation,
                    )
                )
                page_keys = []
                will_upload = False
                for chunk_y in range(grid_h):
                    for chunk_x in range(grid_w):
                        key = plane_chunk_key(
                            identity,
                            "live",
                            0,
                            chunk_x,
                            chunk_y,
                            dtype=_WGPU_REP_DTYPES[representation],
                            representation=representation,
                        )
                        page_keys.append(key)
                        if executor.page_table.lookup(key) is None:
                            will_upload = True
                        # Content-keyed: re-ensuring a resident key is a no-op
                        # in the executor (0 uploads); the report is the oracle.
                        commands.append(
                            EnsureChunkResident(
                                key,
                                self._wgpu_page_block(texture, chunk_y, chunk_x, representation),
                            )
                        )
                if will_upload:
                    planned_upload_tiles.append(tile)
                region = layout[tile]
                committed_tiles[tile] = {
                    "identity": identity,
                    "world_rect": (
                        float(region.x),
                        float(region.y),
                        float(region.width),
                        float(region.height),
                    ),
                    "src_size": (float(width), float(height)),
                    "plane_index": plane_index,
                    "page_keys": tuple(page_keys),
                }

            lut = self._wgpu_resolve_lut_bytes(source_mapping)
            self._wgpu_mapping_state = DisplayMapping(
                mode=mode, level_lo=level_lo, level_hi=level_hi, lut=lut
            )
            display_shape = tile_layout_shape(geometry, frame_plan=frame_plan)
            self._wgpu_committed = {
                "tiles": committed_tiles,
                "representation": representation,
                "display_shape": display_shape,
            }
            self._montage_display_mode = "wgpu_tile_layer"

            start = perf_counter()
            report = self._submit_wgpu(
                (
                    BindContentPlanes(tuple(planes)),
                    *commands,
                    SetDisplayMapping(self._wgpu_mapping_state),
                    UpdateTileInstances(self._wgpu_camera_tiles()),
                )
            )
            upload_ms = (perf_counter() - start) * 1000.0
            self._wgpu_last_report_uploads = int(report.uploads)

            # Physical truth per tile: acknowledge only tiles whose pages the
            # page table actually holds after the submit — partial residency
            # acknowledges the resident subset, never the request.
            presented = tuple(
                tile
                for tile in sorted(committed_tiles)
                if all(
                    executor.page_table.lookup(key) is not None
                    for key in committed_tiles[tile]["page_keys"]
                )
            )
            presented_identities = {
                tile: committed_tiles[tile]["identity"] for tile in presented
            }

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

            structure_key = (display_shape, representation, bool(rgb_already_windowed))
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
            texel_bytes = _WGPU_REP_TEXEL_BYTES[representation]
            updated = tuple(planned_upload_tiles) if uploads > 0 else ()
            stats = TileLayerUpdateStats(
                visible_items=len(committed_tiles),
                presented_tiles=presented,
                presented_identities=presented_identities,
                updated_tiles=updated,
                items_created=0,
                items_updated=len(updated),
                items_skipped=len(committed_tiles) - len(updated),
                existing_items_shown=len(committed_tiles) - len(updated),
                resident_items=len(presented),
                storage_capacity=executor.pool_budget(representation),
                texture_uploads=uploads,
                texture_upload_bytes=uploads * PAGE * PAGE * texel_bytes,
                page_count=resident_pages,
                active_pages=sum(
                    len(info["page_keys"]) for info in committed_tiles.values()
                ),
                estimated_gpu_bytes=resident_pages * PAGE * PAGE * texel_bytes,
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

    def _wgpu_commit_plan(self, payloads, source_mapping, rgb_already_windowed):
        """Validate one commit; return ``(representation, mapping mode)``.

        Everything outside the committed scope raises ``NotImplementedError``
        loudly instead of guessing (montage of N scalar tiles; montage of
        display-ready uint8 RGB tiles; a single complex tile).
        """

        kinds = {tile: _wgpu_payload_kind(payload) for tile, payload in payloads.items()}
        unique = {kind.value for kind in kinds.values()}
        if len(unique) != 1:
            raise NotImplementedError(
                "wgpu backend requires one texture representation per commit; "
                f"got {sorted(unique)}"
            )
        kind = next(iter(kinds.values()))
        representation = _WGPU_REP_BY_KIND[kind]
        display_mode = getattr(
            getattr(source_mapping, "display_mode", None), "value", None
        )
        scale = getattr(getattr(source_mapping, "scale", None), "value", None)
        if scale not in (None, ShaderScale.LINEAR.value):
            raise NotImplementedError(
                f"wgpu backend supports linear shader scale only; got {scale!r}"
            )
        if representation == SCALAR_R32F:
            if display_mode not in (None, ShaderDisplayMode.SCALAR.value):
                raise NotImplementedError(
                    "wgpu backend renders scalar payloads with scalar display "
                    f"mode only; got {display_mode!r}"
                )
            return representation, "real"
        if representation == COMPLEX_RG32F:
            if len(payloads) != 1:
                raise NotImplementedError(
                    "wgpu backend renders complex payloads as a single tile "
                    f"only (montage is row 3c work); got tiles {sorted(payloads)}"
                )
            if display_mode not in (
                None,
                ShaderDisplayMode.COMPLEX.value,
                ShaderDisplayMode.PHASE_COLOR.value,
            ):
                raise NotImplementedError(
                    f"wgpu backend cannot render complex display mode {display_mode!r}"
                )
            component = getattr(
                getattr(source_mapping, "component", None), "value", None
            )
            if component is None:
                component = ShaderComponent.REAL.value
            if component not in _WGPU_COMPONENT_MODES:
                raise NotImplementedError(
                    f"wgpu backend cannot render shader component {component!r}"
                )
            if display_mode == ShaderDisplayMode.PHASE_COLOR.value and component not in (
                ShaderComponent.ANGLE.value,
                ShaderComponent.COMPLEX_PHASE.value,
            ):
                raise NotImplementedError(
                    "wgpu backend renders phase-color for phase components only "
                    f"(magnitude-modulated phase color is unsupported); got {component!r}"
                )
            return representation, _WGPU_COMPONENT_MODES[component]
        # RGB8: display-ready bytes only — the executor pool bypasses
        # levels/LUT, which is honest solely for already-windowed content.
        if not rgb_already_windowed:
            raise NotImplementedError(
                "wgpu backend renders display-ready RGB only "
                "(rgb_already_windowed=False needs shader windowing)"
            )
        if display_mode not in (None, ShaderDisplayMode.RGB_DISPLAY_READY.value):
            raise NotImplementedError(
                f"wgpu backend cannot render RGB display mode {display_mode!r}"
            )
        for tile, payload in payloads.items():
            texture = np.asarray(
                payload.texture_data if payload.texture_data is not None else payload.image
            )
            if texture.dtype != np.uint8 or texture.ndim != 3 or texture.shape[-1] not in (3, 4):
                raise NotImplementedError(
                    f"wgpu RGB tile {tile} payload does not fit rgb8 cleanly "
                    f"(need uint8 (h, w, 3|4), got {texture.dtype} {texture.shape})"
                )
        return representation, "real"

    def _wgpu_payload_texture(self, payload, representation) -> np.ndarray:
        texture = payload.texture_data if payload.texture_data is not None else payload.image
        kind = {
            SCALAR_R32F: TexturePlaneKind.SCALAR_R32F,
            COMPLEX_RG32F: TexturePlaneKind.COMPLEX_RG32F,
            RGB8: TexturePlaneKind.RGB8,
        }[representation]
        return pack_texture_data(texture, kind)

    def _wgpu_page_block(self, texture, chunk_y, chunk_x, representation) -> np.ndarray:
        from arrayscope.gpu.wgpu_executor import PAGE

        if representation == SCALAR_R32F:
            page = np.zeros((PAGE, PAGE), np.float32)
        elif representation == COMPLEX_RG32F:
            page = np.zeros((PAGE, PAGE, 2), np.float32)
        else:
            page = np.zeros((PAGE, PAGE, 3), np.uint8)
        block = texture[
            chunk_y * PAGE : (chunk_y + 1) * PAGE,
            chunk_x * PAGE : (chunk_x + 1) * PAGE,
        ]
        page[: block.shape[0], : block.shape[1]] = block
        return page

    def _wgpu_resolve_lut_bytes(self, shader_mapping) -> bytes | None:
        display_mode = getattr(
            getattr(shader_mapping, "display_mode", None), "value", None
        )
        if display_mode == ShaderDisplayMode.PHASE_COLOR.value:
            explicit = getattr(shader_mapping, "lut_data", None)
            if explicit is not None:
                return _resample_lut_to_rgba256(explicit)
            # A bare phase-color mapping means the canonical cyclic phase LUT
            # (the VisPy _display_shader_mapping template): the view's initial
            # grayscale LUT must not silently turn phase presentation gray.
            if getattr(self, "_display_colormap", None) is None:
                return _resample_lut_to_rgba256(default_phase_lut())
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
        committed_tiles = tuple(sorted((committed or {}).get("tiles", ())))
        stats = TileLayerUpdateStats(
            visible_items=len(committed_tiles),
            presented_tiles=committed_tiles,
            items_skipped=len(committed_tiles),
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


def _wgpu_payload_kind(payload) -> TexturePlaneKind:
    """Payload texture representation (declared kind first, then inference)."""

    kind = getattr(payload, "texture_kind", None)
    if kind is not None:
        return kind if isinstance(kind, TexturePlaneKind) else TexturePlaneKind(
            getattr(kind, "value", kind)
        )
    texture = np.asarray(
        payload.texture_data if getattr(payload, "texture_data", None) is not None else payload.image
    )
    if np.iscomplexobj(texture) or (texture.ndim == 3 and texture.shape[-1] == 2):
        return TexturePlaneKind.COMPLEX_RG32F
    if texture.ndim == 3 and texture.shape[-1] in (3, 4):
        return TexturePlaneKind.RGB8
    if texture.ndim == 2:
        return TexturePlaneKind.SCALAR_R32F
    raise NotImplementedError(
        f"wgpu backend cannot infer a texture representation for payload shape {texture.shape}"
    )


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
