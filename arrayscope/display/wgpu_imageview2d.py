"""Experimental wgpu-backed 2D image view (montage scalar/RGB + complex).

Queue row 3 slice (b): a live rendering backend driven purely by the
renderer command protocol (ADR 0057). The widget composes shared interaction
at the shell seam — PyQtGraph keeps the histogram widget and the
transparent interaction overlay; a rendercanvas ``QRenderWidget`` in bitmap
present mode owns the pixels by default, and the ``wgpu_present_method``
setting can pin the native-Wayland screen path instead
(:mod:`arrayscope.display.backends.wgpu.screen_canvas`: a paint-less native
child driving its own swapchain, bypassing rendercanvas and the per-frame
bitmap readback) — but every pixel decision is expressed as
:class:`~arrayscope.gpu.command_protocol.FrameSubmission` commands into one
:class:`~arrayscope.gpu.wgpu_executor.WgpuPlaneExecutor`.

Committed scope (everything else raises ``NotImplementedError`` loudly
instead of guessing): montages of N scalar, complex, display-ready RGB, or
windowable RGB tiles. Windowable RGB preserves the two-signal semantics:
the color plane is multiplied by one levels-normalized histogram/luminance
plane, packed together in one physical RGBA32F page. Complex tiles use
shader-on-read component modes (magnitude/phase/real/imag), including cyclic
phase hue modulated by normalized magnitude; scalar and complex mappings
support linear/log/symlog display scales. Target-quality tiles bind one
:class:`ContentPlane` each. A complete large-montage preview may instead pack
its reduced values into at most two backend-private page planes; every tile
keeps its own acknowledgement identity and source rectangle, and exact
refinement binds ordinary planes beside the retained preview pages. Ordinary
plane ``document_generation`` is the payload's ack identity, so residency is
content-keyed: re-committing identical content, switching complex modes, and
moving levels are physical zero-upload operations — the executor report is
the oracle, and the commit stats are derived from it, never invented.
Acknowledgement is physical truth per tile: a tile enters
``presented_tiles``/``presented_identities`` only when every one of its
pages is actually resident in the executor page table after the submit.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from time import perf_counter, perf_counter_ns
from typing import ClassVar

import numpy as np

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.core.trace import emit_trace

prefer_pyside6()

import contextlib
import itertools
import threading
import weakref

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.display.backend_contract import WGPU_CAPABILITIES
from arrayscope.display.backends.pyqtgraph.histogram_adapter import PyQtGraphHistogramAdapter
from arrayscope.display.glyph_atlas import GlyphAtlas
from arrayscope.display.imageview2d import ArrayScopeGraphicsView, ImageViewShell
from arrayscope.display.model.tile_identity import TileIdentity, tile_ack_identity
from arrayscope.display.model.tile_stats import TileLayerUpdateStats
from arrayscope.display.overlay_geometry import (
    montage_overlay_rgba,
    montage_overlay_status_segments,
    roi_outline_points,
)
from arrayscope.display.overlay_hit_test import roi_handle_points
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    ShaderScale,
    TexturePlaneKind,
    common_shader_mapping,
    default_phase_lut,
    pack_texture_data,
)
from arrayscope.display.tile_layout import planned_tile_count, tile_layout_map, tile_layout_shape
from arrayscope.display.tile_truth_overlay import (
    tile_truth_overlay_row_text,
    tile_truth_overlay_text,
)
from arrayscope.display.view_navigation_driver import QtViewNavigationDriver
from arrayscope.gpu.chunk_summary import chunk_key_frontier
from arrayscope.gpu.command_protocol import (
    BindContentPlanes,
    ContentPlane,
    DispatchHistogram,
    DisplayMapping,
    EnsureChunkResident,
    EvictChunk,
    FrameSubmission,
    GenerateLodPages,
    OverlayPrimitive,
    PresentGeneration,
    SetDisplayMapping,
    SetOverlayCamera,
    TileInstance,
    UpdateGlyphAtlas,
    UpdateOverlayGeometry,
    UpdateTileInstances,
    UpdateWidgetAtlas,
)
from arrayscope.gpu.keys import (
    COMPLEX_RG32F,
    REDUCER_MEAN,
    REDUCER_PHASE_VECTOR,
    RGB8,
    RGB_WINDOWED_RGBA32F,
    SCALAR_R32F,
)
from arrayscope.presentation.prepared_uploads import prepared_upload_key

#: Complex shader components → protocol mapping modes.
_WGPU_COMPONENT_MODES = {
    ShaderComponent.REAL.value: "real",
    ShaderComponent.IMAG.value: "imag",
    ShaderComponent.ABS.value: "magnitude",
    ShaderComponent.ANGLE.value: "phase",
    ShaderComponent.COMPLEX_PHASE.value: "phase",
}


@dataclass(frozen=True)
class _WgpuPayloadBinding:
    """Physical source-plane binding selected for one payload.

    Anchored payloads bind the complete source plane when every page sampled
    by the window is already resident, or when this payload contains complete
    globally aligned pages that can honestly establish that residency.
    Otherwise the payload keeps its crop-local plane identity.  The latter is
    the correctness fallback for a cold narrow crop: zero padding must never
    masquerade as valid source texels outside the supplied window.
    """

    plane_identity: object
    plane_shape: tuple[int, int]
    source_origin_xy: tuple[float, float]
    page_keys: tuple[object, ...]
    upload_chunks: tuple[tuple[int, int], ...]
    source_anchored: bool
    lod_level: int


@dataclass(frozen=True)
class _WgpuPreviewAtlasSlot:
    """One tile's immutable sub-rectangle in a backend-private preview page."""

    tile_number: int
    payload_identity: object
    binding_identity: object
    page_index: int
    source_origin_xy: tuple[float, float]
    source_size_xy: tuple[float, float]


@dataclass(frozen=True)
class _WgpuPreviewAtlas:
    """Compact physical backing for a large all-coarse montage pass.

    The atlas is deliberately not a semantic source plane.  Its opaque token
    exists only for this view lifetime, its pages never feed resident
    histograms or exact reads, and the per-tile payload identities remain the
    acknowledgement truth.  Exact refinement binds ordinary source planes
    beside these pages while untouched tiles keep sampling their atlas slots.
    """

    token: object
    representation: str
    lod_reducer: str
    pages: tuple[np.ndarray, ...]
    page_keys: tuple[object, ...]
    plane_identities: tuple[object, ...]
    slots: tuple[_WgpuPreviewAtlasSlot, ...]


@dataclass(frozen=True)
class _WgpuTiledBindingSignature:
    """Construction-owned proof of the currently submitted tile binding."""

    payload_identities: tuple[tuple[int, object], ...]
    representation: str
    mapping_mode: str
    layout_identity: object
    transposed: bool
    executor: object
    page_table_generation: int
    histogram_obligation: object


_WGPU_REP_BY_KIND = {
    TexturePlaneKind.SCALAR_R32F: SCALAR_R32F,
    TexturePlaneKind.COMPLEX_RG32F: COMPLEX_RG32F,
    TexturePlaneKind.RGB8: RGB8,
}

_WGPU_REP_DTYPES = {
    SCALAR_R32F: "float32",
    COMPLEX_RG32F: "complex64",
    RGB8: "uint8",
    RGB_WINDOWED_RGBA32F: "float32",
}

_WGPU_POOL_TEXEL_BYTES = {
    SCALAR_R32F: 4,
    COMPLEX_RG32F: 8,
    RGB8: 4,
    RGB_WINDOWED_RGBA32F: 16,
}

# Packing a small preview cohort would only replace cheap ordinary bindings
# with atlas construction.  The product case is the complete 272-tile pass;
# keep this backend optimization large-montage-only and measurable.
_WGPU_PREVIEW_ATLAS_MIN_TILES = 64
_WGPU_PREVIEW_ATLAS_MAX_PAGES = 2
_WGPU_PREVIEW_ATLAS_REPRESENTATIONS = frozenset({SCALAR_R32F, COMPLEX_RG32F})


def _wgpu_rgba(color, alpha: float = 1.0):
    rgb = tuple(int(value) for value in tuple(color or (255, 255, 0))[:3])
    return (*tuple(float(value) / 255.0 for value in rgb), float(alpha))


#: Native tile-truth label styling — mirrors ``tile_truth_overlay._label_style``
#: (the QLabel stylesheet used by the raster backends) in linear RGBA.
_TRUTH_LABEL_FONT_KEY = "monospace"
_TRUTH_LABEL_FONT_PX = 9
_TRUTH_LABEL_BACKGROUND = _wgpu_rgba((8, 18, 24), 210.0 / 255.0)
_TRUTH_LABEL_STYLES = {
    True: (_wgpu_rgba((34, 211, 238)), _wgpu_rgba((165, 243, 252))),  # draw
    False: (_wgpu_rgba((245, 158, 11)), _wgpu_rgba((253, 230, 138))),  # load
}

# One process-wide wgpu device: views (and executor rebuilds on plane growth)
# share it so the canvas context never needs reconfiguration and tests do not
# pay per-view device creation.
_SHARED_WGPU_DEVICE = None
#: Serialises shared-device creation so a background warm-up (started while a
#: file loads) and the GUI thread's first real render never race to build two
#: devices.  Whichever thread arrives second blocks on the same creation and
#: gets the one cached device back.
_SHARED_WGPU_DEVICE_LOCK = threading.Lock()


@dataclass(frozen=True)
class WgpuResidentHistogramEvidence:
    """One fenced GPU histogram over a committed plane's page frontier."""

    evidence_key: object
    tile_number: int
    source_index: int
    frontier_keys: tuple[object, ...]
    readback: object
    wait_completed: object


def import_qrenderwidget():
    """Import rendercanvas's Qt widget without leaking its env mutations.

    On Wayland hosts rendercanvas stomps process env vars at import time
    (``QT_QPA_PLATFORM=xcb``, ``GDK_BACKEND=x11``).  With a live QApplication
    that cannot change the running Qt platform, but it silently poisons every
    later env reader in the process — e.g. the AUTO-backend probe's offscreen
    check selected a different backend inside offscreen test runs. Every rendercanvas
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


# Native texture-codec device features: BC (NVIDIA/AMD and Intel too) and ASTC
# (Intel iGPUs).  Requested whenever the adapter advertises them so the shared
# device can host block-compressed pools; a device without either simply keeps
# the raw path (AUTO degrades silently).
_TEXTURE_CODEC_FEATURES = ("texture-compression-bc", "texture-compression-astc")
_SHARED_WGPU_POWER_PREFERENCE = "low-power"
# Physical framebuffer evidence rejects a lower global threshold: a deterministic
# 39.99 dB page renders at only 39.91 dB under a valid window (see the GPU test).
_WGPU_CODEC_MIN_PSNR_DB = 40.0


def configure_wgpu_adapter_for_profile(power_preference: str) -> None:
    """Select the adapter before device creation for a fresh benchmark process."""

    global _SHARED_WGPU_POWER_PREFERENCE
    value = str(power_preference)
    if value not in {"low-power", "high-performance"}:
        raise ValueError(f"unknown wgpu power preference {power_preference!r}")
    if _SHARED_WGPU_DEVICE is not None and value != _SHARED_WGPU_POWER_PREFERENCE:
        raise RuntimeError("the shared wgpu device already exists; use a fresh benchmark process")
    _SHARED_WGPU_POWER_PREFERENCE = value


def _shared_wgpu_device():
    global _SHARED_WGPU_DEVICE
    # Fast path: the device is created once per process and reused everywhere.
    if _SHARED_WGPU_DEVICE is not None:
        return _SHARED_WGPU_DEVICE
    with _SHARED_WGPU_DEVICE_LOCK:
        # Re-check under the lock: a concurrent caller (e.g. the startup
        # warm-up thread) may have finished creating it while we waited.
        if _SHARED_WGPU_DEVICE is not None:
            return _SHARED_WGPU_DEVICE
        import wgpu
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        with contextlib.suppress(RuntimeError):
            # Vulkan-only instance: the GL backend's EGL re-init is fatal under
            # Wayland (gate-B Tier 0).  Harmless if the instance already exists.
            set_instance_extras(backends=["Vulkan"])
        adapter = wgpu.gpu.request_adapter_sync(power_preference=_SHARED_WGPU_POWER_PREFERENCE)
        # Enable the block-compression features the adapter actually has, so the
        # G7 Phase B BC/ASTC texture pools can be created on this shared device.
        available = {str(f) for f in adapter.features}
        wanted = [f for f in _TEXTURE_CODEC_FEATURES if f in available]
        _SHARED_WGPU_DEVICE = adapter.request_device_sync(required_features=wanted)
    return _SHARED_WGPU_DEVICE


def warm_wgpu_backend():
    """Pre-build the shared device and warm the executor shader cache.

    Runs off the GUI thread (see ``image_view_factory.warm_image_backend_async``)
    so the ~2 s of Vulkan adapter/device init — and the ~165 ms of WGSL
    shader/pipeline compilation behind the first executor — overlap the file
    load instead of stalling the first image.  The throwaway executor exists
    only to push the static shaders through the driver, whose module cache then
    makes each live view's real executor build in a few ms; it owns no residency
    and is discarded immediately.  Best-effort: any failure here is swallowed and
    the normal (lazy) render path re-attempts and reports it.
    """

    device = _shared_wgpu_device()
    with contextlib.suppress(Exception):
        from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

        # A minimal budget on the exact production codec path (OFF): enough to
        # compile every shader module and compute pipeline the real executor
        # shares.  Render pipelines compile lazily per format, so the first
        # framebuffer still pays those, but the compute/shader bulk is warm.
        WgpuPlaneExecutor(
            pool_layers={SCALAR_R32F: 4},
            device=device,
            compressed_textures="off",
        )
    return device


def _device_supports_texture_compression(device) -> bool:
    """Whether the shared device enables a block-compression feature (BC/ASTC)."""

    have = {str(f) for f in device.features}
    return any(f in have for f in _TEXTURE_CODEC_FEATURES)


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

        # Present-method decision (wgpu_present_method setting).  The screen
        # swapchain exists only on a live Wayland session: an explicit
        # ``screen`` pin falls back to bitmap with a recorded reason, and
        # ``auto`` resolves to screen exactly where the measured gate-B
        # recipe applies (bitmap everywhere else — offscreen/xcb runs keep
        # working without configuration changes).
        requested = str(getattr(self, "_wgpu_present_method_requested", "bitmap"))
        self._wgpu_present_method = "bitmap"
        self._wgpu_present_method_fallback_reason = ""
        if requested in ("screen", "auto"):
            from arrayscope.display.backends.wgpu.screen_canvas import (
                screen_present_unavailable_reason,
            )

            reason = screen_present_unavailable_reason()
            if reason is None:
                self._wgpu_present_method = "screen"
            else:
                # For the explicit pin this is a loud fallback; for AUTO it
                # is the resolution rule doing its job.  Recorded either way
                # so diagnostics always answer "why bitmap?".
                self._wgpu_present_method_fallback_reason = reason
                emit_trace(
                    "wgpu_screen_present_fallback",
                    reason=reason,
                    requested=requested,
                )

        if self._wgpu_present_method == "screen":
            # Paint-less native child driving its own swapchain; rendercanvas
            # is bypassed entirely (no import, no env stomping).
            from arrayscope.display.backends.wgpu.screen_canvas import WgpuScreenCanvas

            self._wgpu_canvas = WgpuScreenCanvas(parent=self._display_container)
        else:
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
        self.graphicsView.viewport().setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True
        )
        self.graphicsView.viewport().setStyleSheet("background: transparent;")
        self._display_stack.addWidget(self.graphicsView)
        self.layout.addWidget(self._display_container, 1)

        self.histogram = pg.HistogramLUTWidget()
        self.layout.addWidget(self.histogram)
        self._histogram_adapter = PyQtGraphHistogramAdapter(self.histogram)

        # Executor / protocol state.  Bitmap: the rendercanvas context must
        # exist before any draw (rendercanvas cancels draw events while the
        # canvas has no context; configuration is still deferred until an
        # executor provides the device in the first real draw).  Screen: the
        # swapchain context can only exist once the native window is exposed,
        # so it is created lazily inside the draw.
        self._wgpu_executor = None
        self._wgpu_generation = 0
        # ``tiledPayloadResident`` is a hot scheduling predicate. Binding
        # geometry depends only on the payload, mapping, and page-table
        # generation; rebuilding its nested DataChunkKeys on every pacing
        # pass repeatedly hashes the full semantic identity.
        self._wgpu_residency_binding_generation = -1
        self._wgpu_residency_binding_cache: dict[
            tuple[int, str, str], tuple[weakref.ReferenceType, _WgpuPayloadBinding]
        ] = {}
        self._wgpu_residency_binding_cache_hits = 0
        self._wgpu_residency_binding_cache_misses = 0
        # Hidden atomic successors are warmed across bounded GUI turns. Their
        # pages need a physical owner until the presentation submit transfers
        # ownership to the executor's bound-plane pin set.
        self._wgpu_atomic_warm_pin_owner = object()
        if self._wgpu_present_method == "screen":
            self._wgpu_context = None
        else:
            self._wgpu_context = self._wgpu_canvas.get_context("wgpu")
        self._wgpu_context_format = None
        # Screen-path present-edge diagnostics (Fifo-acquire guard rail).
        self._wgpu_screen_presents = 0
        self._wgpu_screen_acquire_ms_last = 0.0
        self._wgpu_screen_acquire_ms_max = 0.0
        self._wgpu_screen_present_ms_last = 0.0
        self._wgpu_screen_present_ms_max = 0.0
        self._wgpu_mapping_state = DisplayMapping(
            mode="real",
            pixel_grid=self._pixel_grid_enabled,
            clip_indicator=self._clip_indicator_enabled,
            minification_filter=self._minification_filter_enabled,
        )
        self._wgpu_committed: dict[str, object] | None = None
        self._wgpu_tiled_binding_signature: _WgpuTiledBindingSignature | None = None
        self._wgpu_binding_fast_path_commits = 0
        self._wgpu_binding_incremental_commits = 0
        self._wgpu_binding_full_republications = 0
        self._wgpu_tile_instances_cache: (
            tuple[int, tuple[TileInstance, ...], tuple[int, ...]] | None
        ) = None
        # Names each committed record so a successor can point at the record
        # it was derived from without holding a reference to it.
        self._wgpu_commit_serial = 0
        self._wgpu_last_report_uploads = 0
        self._wgpu_last_draw_error: str = ""
        self._wgpu_histogram_evidence_required = False
        self._wgpu_histogram_evidence_obligation = None
        self._wgpu_histogram_evidence: dict[object, WgpuResidentHistogramEvidence] = {}
        self._wgpu_histogram_evidence_ready: set[object] = set()
        # Draw-ack discipline uses explicit request/draw counters.
        self._wgpu_draw_count = 0
        self._wgpu_tile_presentation_request_count = 0
        self._wgpu_tile_presentation_draw_count = 0
        self._wgpu_physical_tile_timeline_enabled = False
        self._wgpu_physical_tile_timeline: list[dict[str, object]] = []
        self._wgpu_canvas_update_request_count = 0
        self._wgpu_canvas_update_pending = False
        self._wgpu_overlay_geometry: tuple[OverlayPrimitive, ...] = ()
        self._wgpu_overlay_geometry_dirty = True
        self._wgpu_montage_tile_overlays: tuple[object, ...] = ()
        # Native text: CPU-baked glyph atlas + tile-truth rows drawn as
        # glyph quads in the same instanced overlay pass (queue row 3 text
        # gap; Qt only bakes the atlas, never touches the frame path).
        self._wgpu_glyph_atlas = GlyphAtlas()
        self._wgpu_glyph_atlas_uploaded_version: int | None = None
        # Floating Qt chips composited into the frame (screen path only).
        from arrayscope.display.backends.wgpu.chip_compositor import (
            FloatingChipCompositor,
        )

        self._wgpu_chip_compositor = FloatingChipCompositor(
            lambda: getattr(self, "_wgpu_canvas", None),
            on_invalidate=self._request_wgpu_canvas_draw,
        )
        self._wgpu_widget_atlas_uploaded_version: int | None = None
        self._wgpu_tile_truth_rows: tuple[dict[str, object], ...] = ()
        self._wgpu_tile_truth_visible_rows: tuple[dict[str, object], ...] = ()
        self._wgpu_text_dpr = 1.0
        self._wgpu_roi_drawing_points: tuple[tuple[float, float], ...] = ()
        self._wgpu_display_shape: tuple[int, int] = (1, 1)
        self._wgpu_last_levels: tuple[float, float] = (0.0, 1.0)
        self._last_wgpu_structure_key = None
        self._last_wgpu_viewport_key = None

        self._wgpu_canvas.request_draw(self._on_wgpu_draw)

    def __init__(
        self,
        parent=None,
        view=None,
        imageItem=None,
        present_method="bitmap",
        texture_codec="off",
        pixel_grid=False,
        clip_indicator=False,
        minification_filter=False,
    ):
        from arrayscope.app.settings_state import (
            normalize_texture_codec_choice,
            normalize_wgpu_present_method_choice,
        )

        # setupUI (called by the shell constructor) reads the request, so it
        # must be normalized and stored before super().__init__ runs.
        self._wgpu_present_method_requested = normalize_wgpu_present_method_choice(
            present_method
        ).value
        # G7 Phase B display-codec choice (AUTO/OFF/BC).  Resolved to an executor
        # ``compressed_textures`` mode string against the device's real BC/ASTC
        # support at executor-build time (``_ensure_wgpu_executor``).
        self._wgpu_texture_codec_choice = normalize_texture_codec_choice(texture_codec)
        # Shader display aids: Stage A's zoom-gated pixel grid and clip
        # markers, plus C1's minification filter.  Pure shader-uniform flags
        # carried on every DisplayMapping this view builds; toggling them live
        # only re-submits SetDisplayMapping (no residency change).  Stored
        # before super().__init__ because setupUI builds the initial mapping.
        self._pixel_grid_enabled = bool(pixel_grid)
        self._clip_indicator_enabled = bool(clip_indicator)
        self._minification_filter_enabled = bool(minification_filter)
        super().__init__(parent=parent, view=view, imageItem=imageItem)
        self._view_navigation = QtViewNavigationDriver(self)
        self.imageItem.setVisible(False)
        self.histogramImageItem.setVisible(False)
        self._wgpu_bounds_item = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
        self._wgpu_bounds_item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self._wgpu_bounds_item.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
        self._wgpu_bounds_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self._layer_owner.add_bounds_item(self._wgpu_bounds_item)
        self.view.sigRangeChanged.connect(lambda *_args: self._on_wgpu_range_changed())
        # Axis inversion (flips) changes ViewBox STATE without necessarily
        # changing the range; without this hook a flip only became visible
        # after the next commit (dogfood bug 2026-07-18).
        state_signal = getattr(self.view, "sigStateChanged", None)
        if state_signal is not None:
            state_signal.connect(lambda *_args: self._request_wgpu_canvas_draw())

    # ---- executor management -------------------------------------------------

    def _ensure_wgpu_executor(
        self,
        required_pages: dict[str, int],
        *,
        preferred_pages: dict[str, int] | None = None,
        residency_budget_bytes: int = 0,
    ):
        """Executor covering active pages with policy-sized retention headroom."""

        from arrayscope.gpu.wgpu_executor import PAGE, WgpuPlaneExecutor

        executor = self._wgpu_executor
        device = _shared_wgpu_device()
        max_layers = int(device.limits["max-texture-array-layers"])
        preferred_pages = dict(preferred_pages or {})
        previous_pages = {
            representation: (0 if executor is None else executor.pool_budget(representation))
            for representation in (
                SCALAR_R32F,
                COMPLEX_RG32F,
                RGB8,
                RGB_WINDOWED_RGBA32F,
            )
        }
        representation_byte_budgets = _wgpu_representation_byte_budgets(
            required_pages=required_pages,
            preferred_pages=preferred_pages,
            previous_pages=previous_pages,
            budget_bytes=int(residency_budget_bytes or 0),
        )
        budgets: dict[str, int] = {}
        for representation in (
            SCALAR_R32F,
            COMPLEX_RG32F,
            RGB8,
            RGB_WINDOWED_RGBA32F,
        ):
            previous = previous_pages[representation]
            needed = int(required_pages.get(representation, 0))
            budget = _wgpu_pool_layer_budget(
                previous=previous,
                needed=needed,
                preferred=int(preferred_pages.get(representation, 0) or 0),
                max_layers=max_layers,
                budget_bytes=int(representation_byte_budgets.get(representation, 0)),
                bytes_per_layer=PAGE * PAGE * _WGPU_POOL_TEXEL_BYTES[representation],
            )
            if budget:
                budgets[representation] = budget
        if executor is not None:
            # The executor's physical arrays already support copy-preserving
            # growth. Raising their logical ceilings in place retains every
            # reusable page and lets the next upload grow only the affected
            # representation. Rebuilding here discarded all residency and
            # forced later index changes to re-upload unchanged chunks.
            executor.ensure_pool_budgets(budgets)
            if not executor.codec_engaged:
                for representation, needed in required_pages.items():
                    if int(needed) > 0:
                        executor.ensure_raw_pool_capacity(representation, int(needed))
            return executor
        # Resolve the explicit display-codec experiment against this device's
        # real block-compression support. OFF is the exact production default.
        from arrayscope.app.settings_state import texture_codec_executor_mode

        codec_mode = texture_codec_executor_mode(
            self._wgpu_texture_codec_choice,
            bc_available=_device_supports_texture_compression(device),
        )
        initial_raw_layers: dict[str, int] = {}
        initial_codec_layers: dict[str, int] = {}
        for representation, budget in budgets.items():
            needed = max(0, int(required_pages.get(representation, 0)))
            # One arriving LOD commonly coexists with its never-black ancestor,
            # so the physical working set is approximately two current rungs,
            # not just the payloads in this commit.  The much larger preferred
            # ladder remains a logical eviction ceiling, not eager allocation.
            physical_target = min(budget, 2 * needed + 8 if needed else 8)
            codec_fraction = 0.0
            if codec_mode != "off" and executor is not None:
                resident = [
                    key
                    for key in executor.page_table.resident_keys()
                    if key.representation == representation
                ]
                if len(resident) >= 8:
                    codec_fraction = sum(
                        executor.page_is_compressed(key) for key in resident
                    ) / len(resident)
            expected_codec = min(physical_target, round(physical_target * codec_fraction))
            # Keep a bounded cushion for a changing quality mix and for LOD
            # destinations. Growth remains the correctness fallback.
            initial_raw_layers[representation] = min(
                budget, max(8, physical_target - expected_codec + 8)
            )
            initial_codec_layers[representation] = min(budget, max(8, expected_codec + 8))
        self._wgpu_executor = WgpuPlaneExecutor(
            pool_layers=budgets or {SCALAR_R32F: 8},
            initial_pool_layers=initial_raw_layers,
            initial_codec_pool_layers=initial_codec_layers,
            device=device,
            compressed_textures=codec_mode,
            codec_min_psnr_db=_WGPU_CODEC_MIN_PSNR_DB,
        )
        # Rebuild discards residency; committed evidence must not survive it.
        self._wgpu_committed = None
        self._wgpu_tiled_binding_signature = None
        self._wgpu_overlay_geometry_dirty = True
        # EVERY atlas upload-currency tracker belongs here: a fresh executor
        # starts with a 1x1 transparent texture, and these versions are what
        # decide whether the atlas is re-uploaded.  Leaving one behind makes
        # its primitives sample transparency forever — the quads still draw,
        # so the geometry looks perfectly healthy.  The widget line was the
        # one missed when chips landed, which is why the floating chips
        # vanished for the rest of a tiled montage fill (the incremental page
        # budget trips a rebuild mid-fill) and only came back when something
        # else happened to bump the compositor's version.
        self._wgpu_glyph_atlas_uploaded_version = None
        self._wgpu_widget_atlas_uploaded_version = None
        self._wgpu_histogram_evidence.clear()
        self._wgpu_histogram_evidence_ready.clear()
        return self._wgpu_executor

    def setResidentHistogramEvidenceRequired(
        self,
        required: bool,
        obligation: object = None,
    ) -> None:
        """Receive the phase owner's current evidence obligation."""

        required = bool(required)
        # Completion belongs to content+mapping, not to the transient coverage
        # phase which asked for it. Closing a phase during a camera gesture
        # must therefore retain accepted evidence. A genuinely new obligation
        # replaces the small current-view cache explicitly.
        if obligation is not None and obligation != self._wgpu_histogram_evidence_obligation:
            self._wgpu_histogram_evidence.clear()
            self._wgpu_histogram_evidence_ready.clear()
            self._wgpu_histogram_evidence_obligation = obligation
        self._wgpu_histogram_evidence_required = required

    def residentHistogramEvidence(self, payloads) -> tuple[WgpuResidentHistogramEvidence, ...]:
        """Return fenced evidence matching the currently committed payloads."""

        committed = self._wgpu_committed or {}
        committed_tiles = dict(committed.get("tiles", {}) or {})
        rows = []
        seen = set()
        for tile, payload in dict(payloads or {}).items():
            info = committed_tiles.get(int(tile))
            if info is None or info.get("identity") != tile_ack_identity(payload):
                continue
            evidence_key = info.get("histogram_evidence_key")
            evidence = self._wgpu_histogram_evidence.get(evidence_key)
            if (
                evidence is not None
                and evidence_key not in self._wgpu_histogram_evidence_ready
                and evidence_key not in seen
            ):
                rows.append(evidence)
                seen.add(evidence_key)
        return tuple(rows)

    def acceptResidentHistogramEvidence(self, evidence_keys) -> None:
        """Mark worker-installed evidence satisfied for this obligation."""

        self._wgpu_histogram_evidence_ready.update(tuple(evidence_keys or ()))

    def _next_wgpu_generation(self) -> int:
        self._wgpu_generation += 1
        return self._wgpu_generation

    def _submit_wgpu(
        self,
        commands,
        *,
        present_to=None,
        present_format="rgba8unorm",
        present_size=None,
    ):
        executor = self._wgpu_executor
        if executor is None:
            return None
        # Overlay-geometry currency is a submission invariant too: the flat
        # overlay buffer (ROI outlines/handles, profile marker, tile-truth
        # labels, floating chips) rides along in any frame drawn while it is
        # stale, before the present.  This is what keeps a freshly rebuilt
        # executor (e.g. warmTiledResidency's mid-fill _ensure_wgpu_executor)
        # from presenting with _overlay_geometry == () until an unrelated
        # present (a dock toggle) happens to flush the dirty flag.  Appended
        # (never prepended) so callers' command indices — the histogram report
        # keys — stay stable; cleared only after a real submission below.
        overlay_dirty = bool(self._wgpu_overlay_geometry_dirty)
        if overlay_dirty:
            commands = (*commands, UpdateOverlayGeometry(self._wgpu_overlay_geometry))
        # Atlas currency is a submission invariant: any frame drawn against a
        # glyph revision the executor has not seen uploads it in the same
        # ordered batch, before the present.  Appended (never prepended) so
        # callers' command indices — the histogram report keys — stay stable.
        # The version compare is one int; frames with fully cached glyphs
        # emit no atlas command (FrameReport.glyph_atlas_uploads stays 0).
        atlas = self._wgpu_glyph_atlas
        if atlas.version != self._wgpu_glyph_atlas_uploaded_version:
            commands = (
                *commands,
                UpdateGlyphAtlas(atlas.size, atlas.size, atlas.image_bytes()),
            )
        # Same currency invariant for the chip atlas: a frame drawn against a
        # revision the executor has not seen uploads it in this batch, before
        # the present.  Unchanged chips emit no command at all
        # (FrameReport.widget_atlas_uploads stays 0).
        chips = self._wgpu_chip_compositor
        chip_atlas = chips.atlas
        if chip_atlas is not None and chips.version != self._wgpu_widget_atlas_uploaded_version:
            commands = (*commands, UpdateWidgetAtlas(*chip_atlas))
        generation = self._next_wgpu_generation()
        submission = FrameSubmission(generation, (*commands, PresentGeneration(generation)))
        if present_to is None:
            report = executor.submit(submission)
        else:
            report = executor.submit(
                submission,
                present_to=present_to,
                present_format=present_format,
                present_size=present_size,
            )
        self._wgpu_glyph_atlas_uploaded_version = atlas.version
        self._wgpu_widget_atlas_uploaded_version = chips.version
        if overlay_dirty:
            self._wgpu_overlay_geometry_dirty = False
        return report

    # ---- draw-ack discipline -------------------------------------------------

    def _request_wgpu_canvas_draw(self, *, count_presentation: bool = False) -> None:
        if count_presentation:
            self._wgpu_tile_presentation_request_count = (
                int(getattr(self, "_wgpu_tile_presentation_request_count", 0) or 0) + 1
            )
        canvas = getattr(self, "_wgpu_canvas", None)
        if canvas is None:
            return
        if bool(getattr(self, "_wgpu_canvas_update_pending", False)) and not count_presentation:
            return
        self._wgpu_canvas_update_pending = True
        self._wgpu_canvas_update_request_count = (
            int(getattr(self, "_wgpu_canvas_update_request_count", 0) or 0) + 1
        )
        # Camera-only descriptor draws have no payload-commit edge, but they
        # are still physical presentation work. Publish them through the same
        # draw-ack signal as tile commits so observers can sample the frame at
        # the real presentation edge.
        self._mark_presentation_draw_pending()
        try:
            canvas.request_draw()
        except Exception:
            self._wgpu_canvas_update_pending = False
            self._presentation_draw_pending = False

    def _on_wgpu_draw(self, *_args) -> None:
        self._wgpu_draw_count = int(getattr(self, "_wgpu_draw_count", 0) or 0) + 1
        self._wgpu_canvas_update_pending = False
        self._wgpu_tile_presentation_draw_count = int(
            getattr(self, "_wgpu_tile_presentation_request_count", 0) or 0
        )
        try:
            self._present_wgpu_frame_to_canvas()
            self._wgpu_last_draw_error = ""
            self._record_wgpu_physical_tile_draw()
        except Exception as exc:  # keep the Qt paint loop alive; surface in diagnostics
            self._wgpu_last_draw_error = f"{type(exc).__name__}: {exc}"
        # Timer category: anti-hang fallback. A presentationDrawn listener may
        # immediately commit the next
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

    def beginPhysicalTileTimeline(self) -> None:
        """Enable a lightweight physical-draw timeline for profiling."""

        self._wgpu_physical_tile_timeline = []
        self._wgpu_physical_tile_timeline_enabled = True

    def endPhysicalTileTimeline(self) -> tuple[dict[str, object], ...]:
        self._wgpu_physical_tile_timeline_enabled = False
        return tuple(dict(row) for row in self._wgpu_physical_tile_timeline)

    def _record_wgpu_physical_tile_draw(self) -> None:
        if not self._wgpu_physical_tile_timeline_enabled:
            return
        rows = self.tileTruthPhysicalRows()
        lod_counts: dict[int, int] = {}
        lossy_tiles = 0
        for row in rows.values():
            lod = int(dict(row or {}).get("physical_lod_level", 0) or 0)
            lod_counts[lod] = lod_counts.get(lod, 0) + 1
            lossy_tiles += str(dict(row or {}).get("physical_quality", "")) != "exact"
        executor = self._wgpu_executor
        presentation_identity = tuple(
            (int(tile), str(row.get("physical_target_identity", "")))
            for tile, row in sorted(rows.items())
        )
        self._wgpu_physical_tile_timeline.append(
            {
                "timestamp_ns": perf_counter_ns(),
                "draw_count": int(self._wgpu_draw_count),
                "request_count": int(self._wgpu_tile_presentation_request_count),
                "tile_count": len(rows),
                "presentation_identity": presentation_identity,
                "exact_tile_count": len(rows) - lossy_tiles,
                "lossy_tile_count": lossy_tiles,
                "lod_counts": {str(level): count for level, count in sorted(lod_counts.items())},
                "resident_page_count": (
                    0 if executor is None else len(executor.page_table.resident_keys())
                ),
                "active_resident_bytes": int(getattr(executor, "active_resident_bytes", 0) or 0),
                "compressed_uploads_total": int(
                    getattr(executor, "compressed_uploads_total", 0) or 0
                ),
            }
        )

    def _present_wgpu_frame_to_canvas(self) -> None:
        executor = self._wgpu_executor
        if executor is None:
            return
        # A chip that moved or repainted (the HUD follows the cursor) changes
        # geometry no shell seam reports, so refresh it before the frame is
        # built.  Deliberately NOT via _sync_wgpu_overlay_geometry: that
        # submits a whole extra frame (re-uploading tile instances) and then
        # asks for yet another draw, so every pointer sample cost two
        # submissions.  Here the new geometry simply rides along in the frame
        # this draw is already about to present.
        if self._wgpu_present_method == "screen" and self._wgpu_chip_compositor.is_dirty:
            self._refresh_wgpu_overlay_geometry()
        # DPR is baked into glyph rasters and label pixel offsets; a monitor
        # move re-lays the text out at the new density before this frame.
        if self._wgpu_tile_truth_rows:
            dpr = float(self._wgpu_canvas.devicePixelRatio() or 1.0)
            if dpr != self._wgpu_text_dpr:
                self._sync_wgpu_overlay_geometry()
        if self._wgpu_present_method == "screen":
            self._present_wgpu_frame_to_screen(executor)
            return
        if self._wgpu_context_format is None:
            preferred = self._wgpu_context.get_preferred_format(None)
            fmt = preferred.removesuffix("-srgb")
            self._wgpu_context.configure(device=executor.device, format=fmt)
            self._wgpu_context_format = fmt
        camera = self._wgpu_camera_command()
        if camera is None:
            return
        texture = self._wgpu_context.get_current_texture()
        self._submit_wgpu(
            (
                SetDisplayMapping(self._wgpu_mapping_state),
                camera,
                UpdateTileInstances(self._wgpu_tile_instances()),
            ),
            present_to=texture.create_view(),
            present_format=self._wgpu_context_format,
            present_size=tuple(int(value) for value in self._wgpu_canvas.get_physical_size()),
        )

    def _present_wgpu_frame_to_screen(self, executor) -> None:
        """Render into the native child's swapchain and present it.

        The draw-ack contract keys on THIS edge: ``context.present()`` is the
        real swapchain present (``wgpuSurfacePresent``), executed inside the
        draw callback before ``_on_wgpu_draw`` publishes the physical
        acknowledgement.  When the surface cannot exist yet (window not
        exposed, zero pixels) the frame returns without presenting; the
        canvas's show/resize/paint hooks schedule a fresh draw once pixels
        are possible, and the ack still publishes so the presentation gate
        can never stay armed forever on a hidden or headless surface.
        """

        canvas = self._wgpu_canvas
        context = canvas.ensure_context(executor.device)
        if context is None:
            return
        if self._wgpu_context is None:
            self._wgpu_context = context
            self._wgpu_context_format = canvas.configured_format
        size = tuple(int(value) for value in canvas.get_physical_size())
        if size[0] < 1 or size[1] < 1:
            return
        # DPR or widget-size change: the context reconfigures the swapchain
        # only when the physical size actually differs.
        context.set_physical_size(*size)
        camera = self._wgpu_camera_command()
        if camera is None:
            return
        acquire_start = perf_counter()
        texture = context.get_current_texture()
        acquire_ms = (perf_counter() - acquire_start) * 1000.0
        # Overlay geometry refreshed for THIS frame (a moved chip) rides along
        # automatically: _submit_wgpu folds a stale overlay buffer into any
        # frame it builds and clears the dirty flag once the submit lands.
        self._submit_wgpu(
            (
                SetDisplayMapping(self._wgpu_mapping_state),
                camera,
                UpdateTileInstances(self._wgpu_tile_instances()),
            ),
            present_to=texture.create_view(),
            present_format=self._wgpu_context_format,
            present_size=size,
        )
        present_start = perf_counter()
        context.present()
        presented_at = perf_counter()
        present_ms = (presented_at - present_start) * 1000.0
        # The canvas owns pacing, so it owns the cadence record: this is the
        # real wgpuSurfacePresent edge the draw-ack contract already keys on.
        canvas.frame_timing.note_presented(
            presented_at, acquire_ms=acquire_ms, present_ms=present_ms
        )
        self._wgpu_screen_presents = int(self._wgpu_screen_presents) + 1
        self._wgpu_screen_acquire_ms_last = acquire_ms
        self._wgpu_screen_acquire_ms_max = max(self._wgpu_screen_acquire_ms_max, acquire_ms)
        self._wgpu_screen_present_ms_last = present_ms
        self._wgpu_screen_present_ms_max = max(self._wgpu_screen_present_ms_max, present_ms)

    def _wgpu_camera_command(self) -> SetOverlayCamera | None:
        """One camera owner for both tile normalization and overlay uniform."""

        try:
            (x0, x1), (y0, y1) = self.view.viewRange()
        except Exception:
            return None
        span_x = float(x1) - float(x0)
        span_y = float(y1) - float(y0)
        if not (span_x > 0.0 and span_y > 0.0):
            return None
        state = getattr(self.view, "state", {}) or {}
        return SetOverlayCamera(
            (float(x0), float(y0), float(x1), float(y1)),
            x_inverted=bool(state.get("xInverted", False)),
            y_inverted=bool(state.get("yInverted", True)),
        )

    def _wgpu_tile_instances(self) -> tuple[TileInstance, ...]:
        """Emit committed per-tile world rects verbatim, camera-free.

        Tile world rects come from the shared montage layout
        (``tile_layout_map``), so drawn geometry and interaction mapping
        share one owner.  The ViewBox is still the camera truth, but it now
        reaches the GPU as a uniform (``_wgpu_camera_command``) rather than
        being folded into every instance here: panning a 272-tile montage
        used to rebuild and re-upload all 272 instances per frame.

        Both inversion axes therefore live in the camera. That is also why
        ``src_size`` is unconditionally positive now — the negative-extent
        mirroring trick only ever compensated for the dst-space flip this
        method used to perform, and dropping it changes no pixels (a flipped
        axis must still mirror the ViewBox, or drags land on mirrored
        features: dogfood bug 2026-07-18).

        Cached on the committed-set identity: ``_wgpu_committed`` is only
        ever replaced wholesale, never mutated in place.

        A bounded delta replaces the whole committed record but changes only
        the descriptors it upserted, so the successor names its predecessor
        and its changed tiles.  When the tile population is unchanged the
        instances are patched at those positions instead of rebuilt, which is
        what keeps this O(delta) rather than O(montage) during a fill.
        """

        committed = self._wgpu_committed
        if not committed or self._montage_display_mode != "wgpu_tile_layer":
            return ()
        serial = committed.get("serial")
        cached = self._wgpu_tile_instances_cache
        if cached is not None and serial is not None and cached[0] == serial:
            return cached[1]
        transposed = bool(committed.get("transposed", False))
        tiles = committed["tiles"]
        base_serial = committed.get("base_serial")
        # Reuse the predecessor's order outright when this commit provably did
        # not change the tile population.  Re-sorting the montage and then
        # comparing the result to prove it equals the cached order are both
        # O(montage) — cheap per element, but they are exactly the kind of
        # whole-montage step this path exists to remove.
        reuse_order = bool(
            cached is not None
            and base_serial is not None
            and cached[0] == base_serial
            and committed.get("population_stable")
        )
        order = cached[2] if reuse_order else tuple(sorted(tiles))

        def instance_for(tile):
            info = tiles[tile]
            return TileInstance(
                tuple(float(value) for value in info["world_rect"]),
                tuple(float(value) for value in info.get("src_origin", (0.0, 0.0))),
                tuple(float(value) for value in info["src_size"]),
                0,
                plane_index=int(info["plane_index"]),
                transposed=transposed,
            )

        changed = committed.get("changed_tiles")
        instances = None
        patched_from_cache = False
        if (
            cached is not None
            and base_serial is not None
            and changed is not None
            and cached[0] == base_serial
            and (reuse_order or cached[2] == order)
        ):
            # ``order`` is sorted, so a changed tile's position is a binary
            # search rather than a montage-sized position index.  The list copy
            # is deferred until a descriptor actually differs: a refinement
            # that re-presents the same geometry then allocates nothing.
            previous = cached[1]
            patched = None
            unplaceable = False
            for tile in changed:
                index = bisect_left(order, int(tile))
                if index >= len(order) or order[index] != int(tile):
                    unplaceable = True
                    break
                replacement = instance_for(int(tile))
                # ``changed`` is a set, so each position is visited once and
                # ``previous[index]`` is always the pre-existing descriptor.
                if replacement != previous[index]:
                    if patched is None:
                        patched = list(previous)
                    patched[index] = replacement
            if not unplaceable:
                patched_from_cache = True
                instances = previous if patched is None else tuple(patched)
        if instances is None:
            # The reused order is only trustworthy for the patch path that
            # proved it. Anything falling through to a full rebuild re-derives
            # it, so a wrong ``population_stable`` can cost work but can never
            # drop or misplace a tile.
            if reuse_order:
                order = tuple(sorted(tiles))
            instances = tuple(instance_for(tile) for tile in order)
        if not patched_from_cache and cached is not None and cached[1] == instances:
            # The executor's identity check deliberately avoids an O(n)
            # equality scan. A full rebuild must compare the full result, but
            # the incremental path above compares only the changed positions.
            # Retain the exact tuple so a target-quality payload replacement
            # with unchanged geometry performs no instance-buffer rewrite.
            instances = cached[1]
        self._wgpu_tile_instances_cache = (serial, instances, order)
        return instances

    # ---- native overlay translation ----------------------------------------

    def _backend_roi_visual_upserted(self, selection) -> None:
        self._sync_wgpu_overlay_geometry()

    def _backend_roi_visual_removed(self, roi_id) -> None:
        self._sync_wgpu_overlay_geometry()

    def _backend_roi_emphasis_changed(self, roi_id: str) -> None:
        self._sync_wgpu_overlay_geometry()

    def _backend_profile_emphasis_changed(self) -> None:
        self._sync_wgpu_overlay_geometry()

    def _after_profile_marker_sync(self) -> None:
        self._sync_wgpu_overlay_geometry()

    def _set_roi_drawing_preview(self, tool, points) -> None:
        if tool is not None:
            self.sync_interaction_state(self.interaction_controller.clear_hover())
        normalized = tuple((float(point[0]), float(point[1])) for point in tuple(points or ()))
        if tool is None or len(normalized) < 2:
            normalized = ()
        if normalized == self._wgpu_roi_drawing_points:
            return
        self._wgpu_roi_drawing_points = normalized
        self._sync_wgpu_overlay_geometry()

    def setMontageTileOverlays(self, overlays):
        # Native geometry replaces the inherited QGraphics painter entirely.
        super().clearMontageTileOverlays()
        self._montage_tile_overlay_items = []
        overlays = tuple(overlays or ())
        if overlays == self._wgpu_montage_tile_overlays:
            return
        self._wgpu_montage_tile_overlays = overlays
        self._sync_wgpu_overlay_geometry()

    def clearMontageTileOverlays(self):
        super().clearMontageTileOverlays()
        self._montage_tile_overlay_items = []
        if not self._wgpu_montage_tile_overlays:
            return
        self._wgpu_montage_tile_overlays = ()
        self._sync_wgpu_overlay_geometry()

    def montageTileOverlayCount(self) -> int:
        return len(self._wgpu_montage_tile_overlays)

    def _prepare_display_overlay_widget(self, widget) -> None:
        """Composite a floating chip inside the frame on the screen path.

        The chip stays an ordinary Qt widget — it is neither reparented nor
        promoted to a native window.  Only its *pixels* are additionally
        drawn inside the wgpu frame, because the swapchain subsurface would
        otherwise hide it over the canvas; see
        :mod:`arrayscope.display.backends.wgpu.chip_compositor` for why the
        two native-window routes cannot work.  Keeping the widget itself
        untouched is what makes input, styling and the part of a chip that
        overhangs the canvas (the hints chip overlaps the histogram) behave
        exactly as on every other backend.
        """

        if widget is None or self._wgpu_present_method != "screen":
            return
        self._wgpu_chip_compositor.register(widget)
        self._request_wgpu_canvas_draw()

    def setTileTruthOverlayRows(self, rows) -> None:
        # Native glyph quads replace the inherited QLabel layer entirely:
        # Qt widgets cannot composite over a native child in screen-present
        # mode, so truth labels must live inside the wgpu frame.
        rows = tuple(dict(row) for row in tuple(rows or ()))
        if rows == self._wgpu_tile_truth_rows:
            return
        self._wgpu_tile_truth_rows = rows
        self._sync_wgpu_overlay_geometry()

    def tileTruthOverlayText(self) -> str:
        return tile_truth_overlay_text(self._wgpu_tile_truth_visible_rows)

    def _on_wgpu_range_changed(self) -> None:
        # Truth-label anchors are world-locked, so pans/zooms move them for
        # free; the resync only rewrites the buffer when the *set* changes
        # (the too-small-to-read visibility rule flips under zoom).
        if self._wgpu_tile_truth_rows:
            self._sync_wgpu_overlay_geometry()
        self._request_wgpu_canvas_draw()

    def _refresh_wgpu_overlay_geometry(self) -> bool:
        """Rebuild the flat overlay buffer WITHOUT submitting a frame.

        Returns whether the geometry changed; the caller folds the resulting
        ``UpdateOverlayGeometry`` into the frame it is already building.
        """

        primitives = self._wgpu_overlay_primitives()
        if primitives == self._wgpu_overlay_geometry:
            return False
        self._wgpu_overlay_geometry = primitives
        self._wgpu_overlay_geometry_dirty = True
        return True

    def _sync_wgpu_overlay_geometry(self) -> None:
        """Rebuild the one flat buffer from shell state after semantic change."""

        evictions_before = self._wgpu_glyph_atlas.evictions
        primitives = self._wgpu_overlay_primitives()
        if self._wgpu_glyph_atlas.evictions != evictions_before:
            # The working set overflowed the bounded atlas mid-build and the
            # cache reset (loudly, via wgpu_glyph_atlas_evicted); rebake once
            # so every UV references the fresh atlas revision.
            primitives = self._wgpu_overlay_primitives()
        if primitives == self._wgpu_overlay_geometry:
            return
        self._wgpu_overlay_geometry = primitives
        self._wgpu_overlay_geometry_dirty = True
        if self._wgpu_present_method == "screen" and bool(
            getattr(self, "_wgpu_canvas_update_pending", False)
        ):
            # A draw is already scheduled and the screen present path folds
            # pending overlay geometry into the frame it presents, so
            # submitting here as well would render the same state twice.
            # Hovering a ROI hit exactly this: the hover-state resync and the
            # chip's own invalidation each drove a submission per sample.
            self._request_wgpu_canvas_draw()
            return
        if self._wgpu_executor is not None:
            # The dirty overlay buffer is a submission invariant now:
            # _submit_wgpu folds UpdateOverlayGeometry into this frame and
            # clears the dirty flag once the submit lands.
            commands = []
            camera = self._wgpu_camera_command()
            if camera is not None:
                commands.extend((camera, UpdateTileInstances(self._wgpu_tile_instances())))
            self._submit_wgpu(tuple(commands))
        self._request_wgpu_canvas_draw()

    def _wgpu_overlay_primitives(self) -> tuple[OverlayPrimitive, ...]:
        primitives: list[OverlayPrimitive] = []

        # Lifecycle truth boxes and geometry-only status marks. Text is
        # deliberately absent: glyph rendering is outside this MVP.
        for overlay in self._wgpu_montage_tile_overlays:
            x = float(getattr(overlay, "x", 0.0))
            y = float(getattr(overlay, "y", 0.0))
            width = float(max(1.0, getattr(overlay, "width", 1.0)))
            height = float(max(1.0, getattr(overlay, "height", 1.0)))
            p0 = (x, y)
            p1 = (x + width, y + height)
            fill, border, mark = montage_overlay_rgba(overlay)
            primitives.append(OverlayPrimitive("world_rect", p0, p1, fill))
            for start, end in (
                ((p0[0], p0[1]), (p1[0], p0[1])),
                ((p1[0], p0[1]), (p1[0], p1[1])),
                ((p1[0], p1[1]), (p0[0], p1[1])),
                ((p0[0], p1[1]), (p0[0], p0[1])),
            ):
                primitives.append(OverlayPrimitive("line", start, end, border, 1.25))
            for start, end in montage_overlay_status_segments(overlay):
                primitives.append(OverlayPrimitive("line", start, end, mark, 2.5))

        for roi_id, (_item, selection) in self._roi_items.items():
            points = roi_outline_points(selection.geometry)
            if len(points) < 2:
                continue
            width, rgb = self._roi_visual_style(roi_id, selection.color)
            color = _wgpu_rgba(rgb)
            for start, end in itertools.pairwise(points):
                primitives.append(OverlayPrimitive("line", start, end, color, width))
            highlighted, interactive = self._roi_visual_emphasis(roi_id)
            handle_size = 12.0 if highlighted or interactive else 10.0
            handle_edge = _wgpu_rgba((255, 255, 255) if interactive else rgb)
            for point in roi_handle_points(selection.geometry):
                center = (float(point[0]), float(point[1]))
                primitives.append(
                    OverlayPrimitive("handle_quad", center, rgba=handle_edge, width=handle_size)
                )
                primitives.append(
                    OverlayPrimitive(
                        "handle_quad",
                        center,
                        rgba=(0.05, 0.05, 0.05, 0.75),
                        width=max(2.0, handle_size - 3.0),
                    )
                )

        if len(self._wgpu_roi_drawing_points) >= 2:
            color = _wgpu_rgba((255, 190, 60))
            for start, end in zip(
                self._wgpu_roi_drawing_points,
                self._wgpu_roi_drawing_points[1:],
                strict=False,
            ):
                primitives.append(OverlayPrimitive("line", start, end, color, 2.5))

        if self.image is not None and bool(
            getattr(self, "_profile_marker_requested_visible", False)
        ):
            position = self.profileMarkerPosition()
            if position is not None:
                x, y = (float(position[0]), float(position[1]))
                x0, y0, x1, y1 = self._current_profile_bounds()
                anchor = (x, y)
                hovered = self._interaction_visual_profile_part is not None
                rgb = (255, 125, 55) if hovered else (230, 60, 30)
                color = _wgpu_rgba(rgb)
                line_width = 2.5 if hovered else 1.5
                primitives.extend(
                    (
                        OverlayPrimitive("line", (x, y0), (x, y1), color, line_width, anchor),
                        OverlayPrimitive("line", (x0, y), (x1, y), color, line_width, anchor),
                        OverlayPrimitive(
                            "handle_quad",
                            anchor,
                            rgba=(1.0, 1.0, 1.0, 1.0),
                            width=12.0 if hovered else 9.0,
                            visibility_anchor=anchor,
                        ),
                        OverlayPrimitive(
                            "handle_quad",
                            anchor,
                            rgba=color,
                            width=8.0 if hovered else 6.0,
                            visibility_anchor=anchor,
                        ),
                    )
                )

        primitives.extend(self._wgpu_tile_truth_primitives())
        # Last: floating chips are window furniture and must sit above every
        # scene overlay, exactly as the Qt widgets they were rasterized from.
        primitives.extend(self._wgpu_chip_primitives())

        return tuple(primitives)

    def _wgpu_chip_primitives(self) -> tuple[OverlayPrimitive, ...]:
        """Screen-anchored quads for the rasterized floating Qt chips."""

        if self._wgpu_present_method != "screen":
            return ()
        self._wgpu_chip_compositor.rebuild_if_needed()
        return tuple(
            OverlayPrimitive(
                "widget_quad",
                (0.0, 0.0),  # unused: widget quads are camera-independent
                screen_offset=placement.offset,
                size=placement.size,
                uv_rect=placement.uv_rect,
            )
            for placement in self._wgpu_chip_compositor.placements
        )

    def _wgpu_tile_truth_primitives(self) -> tuple[OverlayPrimitive, ...]:
        """Tile-truth labels as atlas-textured glyph quads (screen-sized).

        Mirrors the raster backends' QLabel layer: anchored at each tile's
        on-screen top-left corner, constant pixel size under zoom, hidden
        when the tile's screen footprint is too small to read.  All layouts
        are computed before any UV is emitted so mid-build atlas growth
        cannot leave earlier glyphs referencing stale normalized coords.
        """

        rows = self._wgpu_tile_truth_rows
        self._wgpu_tile_truth_visible_rows = ()
        if not rows:
            return ()
        camera = self._wgpu_camera_command()
        canvas = getattr(self, "_wgpu_canvas", None)
        if camera is None or canvas is None:
            return ()
        dpr = float(canvas.devicePixelRatio() or 1.0)
        self._wgpu_text_dpr = dpr
        try:
            target_w, target_h = (max(1, int(value)) for value in canvas.get_physical_size())
        except Exception:
            return ()
        x0, y0, x1, y1 = camera.world_rect
        pixels_per_world_x = target_w / (x1 - x0)
        pixels_per_world_y = target_h / (y1 - y0)
        atlas = self._wgpu_glyph_atlas
        font_px = max(1, round(_TRUTH_LABEL_FONT_PX * dpr))

        labels = []
        visible_rows = []
        for row in rows:
            tile_rect = row.get("tile_rect")
            if tile_rect is None:
                continue
            x, y, width, height = (float(value) for value in tile_rect)
            # Same readability rule as TileTruthOverlayLayer.reposition
            # (16x12 logical px), expressed in physical pixels.
            if width * pixels_per_world_x < 16.0 * dpr or height * pixels_per_world_y < 12.0 * dpr:
                continue
            anchor = (
                x + width if camera.x_inverted else x,
                y + height if not camera.y_inverted else y,
            )
            layout = atlas.layout_text(
                tile_truth_overlay_row_text(row), _TRUTH_LABEL_FONT_KEY, font_px
            )
            labels.append((anchor, bool(row.get("drawable")), layout))
            visible_rows.append(row)
        if not labels:
            return ()

        atlas_size = float(atlas.size)
        inset = 2.0 * dpr
        pad = 3.0 * dpr
        border_px = max(1.0, round(dpr))
        primitives: list[OverlayPrimitive] = []
        for anchor, drawable, layout in labels:
            border_rgba, text_rgba = _TRUTH_LABEL_STYLES[drawable]
            box_w = layout.width + 2.0 * pad
            box_h = layout.height + 2.0 * pad
            primitives.append(
                OverlayPrimitive(
                    "screen_rect",
                    anchor,
                    rgba=_TRUTH_LABEL_BACKGROUND,
                    screen_offset=(inset, inset),
                    size=(box_w, box_h),
                )
            )
            for offset, size in (
                ((inset, inset), (box_w, border_px)),
                ((inset, inset + box_h - border_px), (box_w, border_px)),
                ((inset, inset), (border_px, box_h)),
                ((inset + box_w - border_px, inset), (border_px, box_h)),
            ):
                primitives.append(
                    OverlayPrimitive(
                        "screen_rect",
                        anchor,
                        rgba=border_rgba,
                        screen_offset=offset,
                        size=size,
                    )
                )
            for placement in layout.placements:
                entry = placement.entry
                primitives.append(
                    OverlayPrimitive(
                        "glyph_quad",
                        anchor,
                        rgba=text_rgba,
                        screen_offset=(inset + pad + placement.x, inset + pad + placement.y),
                        size=(float(entry.width), float(entry.height)),
                        uv_rect=(
                            entry.x / atlas_size,
                            entry.y / atlas_size,
                            (entry.x + entry.width) / atlas_size,
                            (entry.y + entry.height) / atlas_size,
                        ),
                    )
                )
        self._wgpu_tile_truth_visible_rows = tuple(visible_rows)
        return tuple(primitives)

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
        from arrayscope.gpu.wgpu_executor import PAGE

        self._start_upload_timing("wgpu_tile_layer")
        applying = self._applying_presentation
        self._applying_presentation = True
        previous_executor = self._wgpu_executor
        pool_growth_ms_before = float(
            getattr(previous_executor, "pool_growth_ms_total", 0.0) or 0.0
        )
        # Prepare/submit attribution for the GUI-thread hand-off split: every
        # command in this callback is built before ``_submit_wgpu`` runs, so
        # the boundary below is exactly the line a worker could take over.
        apply_started = perf_counter()
        self._wgpu_pack_ms = 0.0
        try:
            payloads = {
                int(tile): payload for tile, payload in dict(montage_tile_payloads or {}).items()
            }
            if not payloads:
                # Loading-only commit: nothing drawable yet; clear stale tiles.
                self._clear_wgpu_tiles()
                self.image = img
                self._montage_display_mode = "wgpu_tile_layer"
                stats = TileLayerUpdateStats(visible_items=0, presented_tiles=())
                self._record_tile_layer_stats(stats)
                return stats
            previous_commit = self._wgpu_committed or {}
            delta_upserts = {
                int(tile): payload
                for tile, payload in dict(getattr(tile_delta, "upserts", {}) or {}).items()
            }
            plan_payloads = delta_upserts if previous_commit and delta_upserts else payloads
            source_mapping = shader_mapping
            if source_mapping is None:
                source_mapping = common_shader_mapping(
                    getattr(payload, "shader_mapping", None) for payload in plan_payloads.values()
                )
            (
                representation,
                mode,
                scale,
                symlog_constant,
                phase_color,
            ) = self._wgpu_commit_plan(plan_payloads, source_mapping, rgb_already_windowed)
            level_lo, level_hi = (float(levels[0]), float(levels[1]))
            if not level_hi > level_lo:
                level_hi = level_lo + 1e-6
            mapping_state = DisplayMapping(
                mode=mode,
                level_lo=level_lo,
                level_hi=level_hi,
                lut=self._wgpu_resolve_lut_bytes(source_mapping),
                scale=scale,
                symlog_constant=symlog_constant,
                phase_color=phase_color,
                pixel_grid=self._pixel_grid_enabled,
                clip_indicator=self._clip_indicator_enabled,
                minification_filter=self._minification_filter_enabled,
            )
            layout_identity = _wgpu_layout_identity(geometry, frame_plan=frame_plan)
            transposed = _display_axes_transposed(geometry)
            physical_identities = (
                None if delta_upserts else _wgpu_physical_payload_identities(payloads)
            )
            if not delta_upserts and self._wgpu_tiled_binding_reusable(
                physical_identities=physical_identities,
                representation=representation,
                mapping_mode=mode,
                layout_identity=layout_identity,
                transposed=transposed,
                tile_delta=tile_delta,
            ):
                return self._apply_wgpu_mapping_only_tiled_presentation(
                    img,
                    histogramPlotData=histogramPlotData,
                    histogramRange=histogramRange,
                    levels=(level_lo, level_hi),
                    viewport_policy=viewport_policy,
                    rgb_already_windowed=rgb_already_windowed,
                    tile_delta=tile_delta,
                    mapping_state=mapping_state,
                )

            layout = tile_layout_map(geometry, frame_plan=frame_plan)
            missing = sorted(set(payloads) - set(layout))
            if missing:
                raise NotImplementedError(
                    f"wgpu commit payload tiles {missing} have no montage/frame-plan "
                    "layout region — refusing to guess destination geometry"
                )

            display_shape = tile_layout_shape(geometry, frame_plan=frame_plan)
            incremental_delta = bool(
                previous_commit
                and delta_upserts
                and not tuple(getattr(tile_delta, "removals", ()) or ())
                and not bool(getattr(tile_delta, "force_refresh", False))
                and not bool(getattr(tile_delta, "atomic_handoff", False))
                and previous_commit.get("preview_atlas") is None
                and str(previous_commit.get("representation", "")) == representation
                and tuple(previous_commit.get("display_shape", ())) == tuple(display_shape)
                and previous_commit.get("layout_identity") == layout_identity
                and bool(previous_commit.get("transposed", False)) == transposed
                and previous_commit.get("histogram_obligation")
                == self._wgpu_histogram_evidence_obligation
                and set(dict(previous_commit.get("tiles", {}) or {})) <= set(payloads)
                and set(delta_upserts) <= set(payloads)
            )
            work_payloads = delta_upserts if incremental_delta else payloads
            if incremental_delta:
                signature = self._wgpu_tiled_binding_signature
                identity_rows = (
                    {}
                    if signature is None or signature.payload_identities is None
                    else dict(signature.payload_identities)
                )
                delta_identities = _wgpu_physical_payload_identities(delta_upserts)
                if identity_rows and delta_identities is not None:
                    identity_rows.update(delta_identities)
                    physical_identities = tuple(sorted(identity_rows.items()))
                else:
                    physical_identities = _wgpu_physical_payload_identities(payloads)
            elif physical_identities is None:
                physical_identities = _wgpu_physical_payload_identities(payloads)
            pack_started = perf_counter()
            textures = {
                tile: self._wgpu_payload_texture(work_payloads[tile], representation)
                for tile in work_payloads
            }
            self._wgpu_pack_ms += (perf_counter() - pack_started) * 1000.0
            planned_count = planned_tile_count(
                geometry,
                frame_plan=frame_plan,
                minimum=len(payloads),
            )
            lod_geometry = {
                tile: _wgpu_payload_declared_lod_geometry(work_payloads[tile], textures[tile])
                for tile in work_payloads
            }
            preview_payloads = {} if incremental_delta else _wgpu_preview_payloads(payloads)
            preview_reducers = {
                _wgpu_payload_lod_reducer(
                    payload,
                    representation=representation,
                    mapping_mode=mode,
                )
                for payload in preview_payloads.values()
            }
            preview_atlas = None
            if representation in _WGPU_PREVIEW_ATLAS_REPRESENTATIONS and len(preview_reducers) == 1:
                preview_reducer = next(iter(preview_reducers))
                previous_atlas = (self._wgpu_committed or {}).get("preview_atlas")
                preview_atlas = _wgpu_reusable_preview_atlas(
                    previous_atlas,
                    preview_payloads,
                    representation=representation,
                    lod_reducer=preview_reducer,
                )
                # Construction remains a complete hidden transaction. The
                # production owner can reach this branch only when the
                # governor admitted the whole required set as one chunk;
                # progressive prefixes stay on ordinary page bindings.
                governed_upserts = {
                    int(tile) for tile in dict(getattr(tile_delta, "upserts", {}) or {})
                }
                if (
                    preview_atlas is None
                    and len(preview_payloads) == planned_count
                    and governed_upserts == set(preview_payloads)
                ):
                    preview_atlas = _build_wgpu_preview_atlas(
                        preview_payloads,
                        textures,
                        representation=representation,
                        lod_reducer=preview_reducer,
                    )
            atlas_tiles = (
                frozenset(int(tile) for tile in preview_payloads)
                if preview_atlas is not None
                else frozenset()
            )
            ordinary_tiles = tuple(tile for tile in work_payloads if tile not in atlas_tiles)
            # Size the pool for each declared ladder, not merely the payload
            # arriving in this commit.  Otherwise a coarse-first plane can
            # force an executor rebuild when L0 arrives, discarding the very
            # ancestor pages that make refinement never-black.
            pages_preferred = sum(
                _wgpu_ladder_page_count(lod_geometry[tile][1], max_lod=lod_geometry[tile][0])
                for tile in ordinary_tiles
            )
            pages_preferred += 0 if preview_atlas is None else len(preview_atlas.page_keys)
            capacity_pages_by_tile = {
                tile: _wgpu_payload_capacity_page_count(
                    payloads[tile],
                    texture_shape=tuple(int(value) for value in textures[tile].shape[:2]),
                    representation=representation,
                    selected_lod=lod_geometry[tile][0],
                )
                for tile in ordinary_tiles
            }
            # Progressive commits expose only the payloads materialized so
            # far, while the immutable frame plan already declares the whole
            # same-shaped montage transaction. Reserve that measured visible
            # working set once. Otherwise every small presentation batch
            # grows and copies the texture array again on its way to the same
            # final capacity.
            if preview_atlas is None:
                pages_needed = _wgpu_pre_reservation_page_count(
                    capacity_pages_by_tile,
                    planned_count=planned_count,
                    max_layers=int(_shared_wgpu_device().limits["max-texture-array-layers"]),
                )
            else:
                # The atlas already represents every still-preview tile, so
                # forecasting one ordinary page per planned tile would erase
                # the allocation win before refinement even starts.
                pages_needed = sum(capacity_pages_by_tile.values()) + len(preview_atlas.page_keys)
            executor_ensure_started = perf_counter()
            executor = self._ensure_wgpu_executor(
                {representation: pages_needed},
                preferred_pages={representation: pages_preferred},
                residency_budget_bytes=int(tile_residency_budget_bytes or 0),
            )
            executor_ensure_ms = (perf_counter() - executor_ensure_started) * 1000.0
            resident_by_plane: dict[tuple[object, object, str], list[object]] = {}
            for key in executor.page_table.resident_keys():
                resident_by_plane.setdefault(
                    (
                        key.document_generation,
                        key.operation_key,
                        key.representation,
                    ),
                    [],
                ).append(key)

            # One bound content plane per tile.  The plane key deliberately
            # excludes payload LOD: a coarse and fine acknowledgement are
            # distinct presentation identities but pages of the SAME source
            # plane.  The current payload's level selects the page-table span;
            # tiles continue to request L0 so the shader resolves resident
            # ancestors while refinement is incomplete.
            planes = list(previous_commit.get("planes", ()) or ()) if incremental_delta else []
            previous_tile_map = previous_commit.get("tiles", {}) or {}
            committed_tiles: dict[int, dict[str, object]] = (
                dict(previous_tile_map) if incremental_delta else {}
            )
            commands = []
            planned_resident = set(executor.page_table.resident_keys())
            planned_upload_tiles = []
            histogram_specs = []
            if preview_atlas is not None:
                atlas_plane_indices = {}
                atlas_uploaded = False
                for atlas_page, (plane_identity, page_key, page_values) in enumerate(
                    zip(
                        preview_atlas.plane_identities,
                        preview_atlas.page_keys,
                        preview_atlas.pages,
                        strict=True,
                    )
                ):
                    atlas_plane_indices[atlas_page] = len(planes)
                    planes.append(
                        ContentPlane(
                            plane_identity,
                            "preview-atlas",
                            (PAGE, PAGE),
                            max_lod=0,
                            representation=representation,
                            lod_reducer=preview_atlas.lod_reducer,
                        )
                    )
                    if page_key not in planned_resident:
                        commands.append(EnsureChunkResident(page_key, page_values))
                        planned_resident.add(page_key)
                        atlas_uploaded = True
                    resident_by_plane[(plane_identity, "preview-atlas", representation)] = [
                        page_key
                    ]
                slots = {int(slot.tile_number): slot for slot in preview_atlas.slots}
                for tile in sorted(atlas_tiles):
                    payload = payloads[tile]
                    slot = slots[tile]
                    region = layout[tile]
                    committed_tiles[tile] = {
                        "identity": tile_ack_identity(payload),
                        "world_rect": (
                            float(region.x),
                            float(region.y),
                            float(region.width),
                            float(region.height),
                        ),
                        "src_size": tuple(slot.source_size_xy),
                        "src_origin": tuple(slot.source_origin_xy),
                        "plane_index": int(atlas_plane_indices[slot.page_index]),
                        "page_keys": (preview_atlas.page_keys[slot.page_index],),
                        "lod_level": 0,
                        "plane_identity": preview_atlas.plane_identities[slot.page_index],
                        "lod_reducer": preview_atlas.lod_reducer,
                        "source_index": int(getattr(payload, "source_index", tile)),
                        # Packed pages are physical presentation backing only.
                        # The canonical payload summary remains the evidence
                        # owner; atlas padding/neighbours are never sampled as
                        # semantic histogram input.
                        "prepared_histogram_evidence": bool(
                            getattr(payload, "level_stats", None) is not None
                            or getattr(payload, "level_data", None) is not None
                        ),
                        "histogram_frontier_eligible": False,
                        "histogram_evidence_key": None,
                    }
                if atlas_uploaded:
                    planned_upload_tiles.extend(sorted(atlas_tiles))
            for tile in sorted(work_payloads):
                if tile in atlas_tiles:
                    continue
                payload = payloads[tile]
                identity = tile_ack_identity(payload)
                texture = textures[tile]
                lod_level, source_shape = lod_geometry[tile]
                lod_reducer = _wgpu_payload_lod_reducer(
                    payload,
                    representation=representation,
                    mapping_mode=mode,
                )
                binding = _wgpu_payload_binding(
                    payload,
                    texture,
                    representation=representation,
                    mapping_mode=mode,
                    resident_keys=planned_resident,
                )
                reusable_native = self._wgpu_reusable_native_texture(
                    payload,
                    binding=binding,
                    representation=representation,
                    selected_lod=lod_level,
                )
                upload_texture = texture
                if reusable_native is not None:
                    native_keys = _wgpu_native_prefetch_page_keys(
                        payload,
                        representation=representation,
                        mapping_mode=mode,
                        selected_lod=lod_level,
                    )
                    native_grid_w = -(-int(reusable_native.shape[1]) // PAGE)
                    source_rect = tuple(int(value) for value in payload.source_anchor.source_rect)
                    binding = _WgpuPayloadBinding(
                        plane_identity=(
                            "wgpu-source-plane",
                            payload.source_anchor.content_key,
                        ),
                        plane_shape=tuple(
                            int(value) for value in payload.source_anchor.plane_shape
                        ),
                        source_origin_xy=(float(source_rect[2]), float(source_rect[0])),
                        page_keys=native_keys,
                        upload_chunks=tuple(
                            divmod(index, native_grid_w) for index in range(len(native_keys))
                        ),
                        source_anchored=True,
                        lod_level=0,
                    )
                    upload_texture = reusable_native
                physical_lod_level = int(binding.lod_level)
                plane_identity = binding.plane_identity
                source_height, source_width = binding.plane_shape
                local_source_height, local_source_width = source_shape
                resident_plane_keys = tuple(
                    key
                    for key in resident_by_plane.get((plane_identity, "live", representation), ())
                    if key.lod.is_native or key.lod.reducer == lod_reducer
                )
                resident_lods = (int(key.lod.level) for key in resident_plane_keys)
                max_lod = max((physical_lod_level, *resident_lods))
                content_plane = ContentPlane(
                    plane_identity,
                    "live",
                    (source_height, source_width),
                    max_lod=max_lod,
                    representation=representation,
                    lod_reducer=lod_reducer,
                )
                previous_info = committed_tiles.get(tile) if incremental_delta else None
                if previous_info is None:
                    plane_index = len(planes)
                    planes.append(content_plane)
                else:
                    plane_index = int(previous_info["plane_index"])
                    planes[plane_index] = content_plane
                page_keys = []
                will_upload = False
                for key, (chunk_y, chunk_x) in zip(
                    binding.page_keys,
                    binding.upload_chunks,
                    strict=True,
                ):
                    page_keys.append(key)
                    if key not in planned_resident:
                        generated = _wgpu_plan_lod_page_generation(
                            key,
                            plane_shape=binding.plane_shape,
                            available=planned_resident,
                            commands=commands,
                        )
                        if not generated:
                            will_upload = True
                            # CPU-produced payload remains the fallback for
                            # cold content and non-mean reducer families.
                            page_started = perf_counter()
                            page_block = self._wgpu_page_block(
                                upload_texture,
                                chunk_y,
                                chunk_x,
                                representation,
                            )
                            self._wgpu_pack_ms += (perf_counter() - page_started) * 1000.0
                            commands.append(EnsureChunkResident(key, page_block))
                            planned_resident.add(key)
                if will_upload:
                    planned_upload_tiles.append(tile)
                plane_residency_key = (plane_identity, "live", representation)
                resident_by_plane[plane_residency_key] = list(
                    dict.fromkeys((*resident_plane_keys, *page_keys))
                )
                region = layout[tile]
                committed_tiles[tile] = {
                    "identity": identity,
                    "world_rect": (
                        float(region.x),
                        float(region.y),
                        float(region.width),
                        float(region.height),
                    ),
                    "src_size": (float(local_source_width), float(local_source_height)),
                    "src_origin": tuple(binding.source_origin_xy),
                    "plane_index": plane_index,
                    "page_keys": tuple(page_keys),
                    "lod_level": physical_lod_level,
                    "plane_identity": plane_identity,
                    "lod_reducer": lod_reducer,
                    "source_index": int(getattr(payload, "source_index", tile)),
                    "prepared_histogram_evidence": bool(
                        getattr(payload, "level_stats", None) is not None
                        or getattr(payload, "level_data", None) is not None
                    ),
                    "histogram_frontier_eligible": True,
                }

            histogram_capable = representation != RGB8
            scheduled_evidence_keys = set()
            atlas_missing_prepared_evidence = 0
            histogram_tiles = (
                tuple(sorted(work_payloads)) if incremental_delta else tuple(committed_tiles)
            )
            for tile in histogram_tiles:
                info = committed_tiles[tile]
                if not bool(info["histogram_frontier_eligible"]):
                    info["histogram_evidence_key"] = None
                    if (
                        self._wgpu_histogram_evidence_required
                        and not info["prepared_histogram_evidence"]
                    ):
                        atlas_missing_prepared_evidence += 1
                    continue
                frontier_keys = (
                    chunk_key_frontier(
                        tuple(
                            key
                            for key in resident_by_plane[
                                (info["plane_identity"], "live", representation)
                            ]
                            if key.lod.is_native or key.lod.reducer == info["lod_reducer"]
                        )
                    )
                    if histogram_capable and bool(info["histogram_frontier_eligible"])
                    else ()
                )
                histogram_evidence_key = (
                    "wgpu-resident-histogram",
                    self._wgpu_histogram_evidence_obligation,
                    info["plane_identity"],
                    tuple(frontier_keys),
                    mode,
                    scale,
                    float(symlog_constant),
                )
                info["histogram_evidence_key"] = histogram_evidence_key
                if (
                    self._wgpu_histogram_evidence_required
                    and not info["prepared_histogram_evidence"]
                    and frontier_keys
                    and histogram_evidence_key not in self._wgpu_histogram_evidence
                    and histogram_evidence_key not in scheduled_evidence_keys
                ):
                    histogram_specs.append(
                        (
                            int(tile),
                            int(info["source_index"]),
                            histogram_evidence_key,
                            tuple(frontier_keys),
                        )
                    )
                    scheduled_evidence_keys.add(histogram_evidence_key)
            if atlas_missing_prepared_evidence:
                emit_trace(
                    "wgpu_histogram_queue_bail",
                    reason="preview_atlas_is_nonsemantic",
                    missing_tiles=atlas_missing_prepared_evidence,
                )

            self._wgpu_mapping_state = mapping_state
            commit_serial = int(self._wgpu_commit_serial) + 1
            self._wgpu_commit_serial = commit_serial
            self._wgpu_committed = {
                "tiles": committed_tiles,
                "planes": tuple(planes),
                "representation": representation,
                "display_shape": display_shape,
                "budget_bytes": int(tile_residency_budget_bytes or 0),
                "layout_identity": layout_identity,
                "histogram_obligation": self._wgpu_histogram_evidence_obligation,
                "preview_atlas": preview_atlas,
                "preview_atlas_active_tiles": tuple(sorted(atlas_tiles)),
                # Which descriptors this commit rewrote, and the record they
                # were rewritten from.  A bounded delta only touches the tiles
                # it committed, so downstream per-tile derivations (the
                # instance buffer) patch instead of rebuilding the montage.
                # The predecessor is named by serial rather than held: a
                # reference would chain every commit to all of its ancestors.
                "serial": commit_serial,
                "base_serial": (
                    int(previous_commit.get("serial", -1)) if incremental_delta else None
                ),
                "changed_tiles": (
                    frozenset(int(tile) for tile in (*work_payloads, *atlas_tiles))
                    if incremental_delta
                    else None
                ),
                # Whether this commit rewrote descriptors WITHOUT changing the
                # tile population.  An incremental commit already refuses
                # removals, so it can only add; deciding that from the delta
                # costs O(delta), and it lets the successor reuse the
                # predecessor's tile ORDER instead of re-sorting the montage
                # and comparing the result to prove it is the same order.
                "population_stable": (
                    incremental_delta
                    and not atlas_tiles
                    and all(int(tile) in previous_tile_map for tile in work_payloads)
                ),
                # Payloads are canonical (sorted image axes); an X/Y swap is a
                # display transform the vertex shader applies via a UV axis swap
                # (world rects stay display-oriented, source windows canonical).
                "transposed": transposed,
            }
            self._montage_display_mode = "wgpu_tile_layer"

            start = perf_counter()
            camera = self._wgpu_camera_command()
            previous_page_keys = tuple(
                key
                for info in (previous_commit.get("tiles", {}) or {}).values()
                for key in tuple(info.get("page_keys", ()) or ())
            )
            committed_page_keys = tuple(
                key
                for info in committed_tiles.values()
                for key in tuple(info.get("page_keys", ()) or ())
            )
            atomic_handoff = bool(getattr(tile_delta, "atomic_handoff", False))
            predecessor_transaction_keys = (
                previous_page_keys if incremental_delta or atomic_handoff else ()
            )
            if incremental_delta:
                # Preserve exactly the predecessor's visible backing while
                # this bounded successor delta uploads. Stable plane
                # descriptors cover every resident LOD and cannot define
                # presentation ownership.
                executor.replace_bound_content_pin_set(previous_page_keys)
            submission_commands = [
                BindContentPlanes(
                    tuple(planes),
                    committed_page_keys=predecessor_transaction_keys,
                ),
                *commands,
                SetDisplayMapping(self._wgpu_mapping_state),
            ]
            if camera is not None:
                submission_commands.append(camera)
            submission_commands.append(UpdateTileInstances(self._wgpu_tile_instances()))
            histogram_indices = []
            for _tile, _source_index, _evidence_key, frontier_keys in histogram_specs:
                histogram_indices.append(len(submission_commands))
                submission_commands.append(
                    DispatchHistogram(
                        frontier_keys,
                        bins=64,
                        lo=None,
                        hi=None,
                        mode=mode,
                        scale=scale,
                        symlog_constant=symlog_constant,
                    )
                )
            if histogram_specs:
                emit_trace(
                    "wgpu_histogram_dispatch",
                    evidence_rows=len(histogram_specs),
                    obligation=self._wgpu_histogram_evidence_obligation,
                )
            # BindContentPlanes is the first command and immediately installs
            # the executor's bound-plane pins. Release the temporary hidden
            # transaction owner at this synchronous submission edge: there is
            # no event-loop interval in which successor pages are unowned.
            executor.replace_resident_pin_set(self._wgpu_atomic_warm_pin_owner, ())
            submit_started = perf_counter()
            report = self._submit_wgpu(tuple(submission_commands))
            submit_ms = (perf_counter() - submit_started) * 1000.0
            executor.replace_bound_content_pin_set(committed_page_keys)
            upload_ms = (perf_counter() - start) * 1000.0
            self._wgpu_last_report_uploads = int(report.uploads)
            for command_index, spec in zip(histogram_indices, histogram_specs, strict=False):
                tile, source_index, evidence_key, frontier_keys = spec
                missing = tuple(report.histogram_missing.get(command_index, ()))
                if missing:
                    # Pool pressure inside this very submission evicted part
                    # of the snapshotted frontier; the partial histogram is
                    # not honest evidence.  Drop the spec loudly (ground
                    # rule: silent bails latch the coverage barrier) — it
                    # stays uninstalled, so the normal evidence machinery
                    # re-schedules it on the next commit.
                    emit_trace(
                        "wgpu_histogram_queue_bail",
                        reason="evicted_in_batch",
                        tile=int(tile),
                        source_index=int(source_index),
                        missing_keys=len(missing),
                        frontier_keys=len(frontier_keys),
                    )
                    continue
                self._wgpu_histogram_evidence[evidence_key] = WgpuResidentHistogramEvidence(
                    evidence_key=evidence_key,
                    tile_number=tile,
                    source_index=source_index,
                    frontier_keys=frontier_keys,
                    readback=report.histograms[command_index],
                    wait_completed=report.wait_completed,
                )

            stats = self._finish_wgpu_tiled_presentation(
                img,
                histogramPlotData=histogramPlotData,
                histogramRange=histogramRange,
                levels=(level_lo, level_hi),
                viewport_policy=viewport_policy,
                rgb_already_windowed=rgb_already_windowed,
                tile_delta=tile_delta,
                report=report,
                planned_upload_tiles=tuple(planned_upload_tiles),
                tile_residency_budget_bytes=tile_residency_budget_bytes,
                upload_ms=upload_ms,
                pool_growth_ms=max(
                    0.0,
                    float(getattr(executor, "pool_growth_ms_total", 0.0) or 0.0)
                    - pool_growth_ms_before,
                ),
                executor_initialization_ms=(
                    executor_ensure_ms + upload_ms if previous_executor is None else 0.0
                ),
                binding_fast_path=False,
                binding_incremental=incremental_delta,
                texture_prepare_ms=max(
                    0.0, (submit_started - apply_started) * 1000.0 - executor_ensure_ms
                ),
                texture_submit_ms=submit_ms,
                texture_pack_ms=self._wgpu_pack_ms,
            )
            if physical_identities is not None and layout_identity is not None:
                self._wgpu_tiled_binding_signature = _WgpuTiledBindingSignature(
                    payload_identities=physical_identities,
                    representation=representation,
                    mapping_mode=mode,
                    layout_identity=layout_identity,
                    transposed=transposed,
                    executor=executor,
                    page_table_generation=int(executor.page_table.generation),
                    histogram_obligation=self._wgpu_histogram_evidence_obligation,
                )
            else:
                self._wgpu_tiled_binding_signature = None
            return stats
        finally:
            self._applying_presentation = applying
            self._finish_upload_timing()

    def _wgpu_tiled_binding_reusable(
        self,
        *,
        physical_identities,
        representation: str,
        mapping_mode: str,
        layout_identity,
        transposed: bool,
        tile_delta,
    ) -> bool:
        """Whether this commit can reuse the submitted binding by construction."""

        signature = self._wgpu_tiled_binding_signature
        executor = self._wgpu_executor
        committed = self._wgpu_committed
        if (
            signature is None
            or executor is None
            or committed is None
            or physical_identities is None
            or layout_identity is None
            or signature.executor is not executor
            or signature.layout_identity != layout_identity
            or signature.payload_identities != physical_identities
            or signature.representation != str(representation)
            or signature.mapping_mode != str(mapping_mode)
            or signature.transposed != bool(transposed)
            or signature.page_table_generation != int(executor.page_table.generation)
            or signature.histogram_obligation != self._wgpu_histogram_evidence_obligation
            or bool(dict(getattr(tile_delta, "upserts", {}) or {}))
            or bool(tuple(getattr(tile_delta, "removals", ()) or ()))
        ):
            return False
        if not self._wgpu_histogram_evidence_required:
            return True
        for info in dict(committed.get("tiles", {}) or {}).values():
            if bool(info.get("prepared_histogram_evidence")):
                continue
            evidence_key = info.get("histogram_evidence_key")
            if evidence_key not in self._wgpu_histogram_evidence:
                return False
        return True

    def _apply_wgpu_mapping_only_tiled_presentation(
        self,
        img,
        *,
        histogramPlotData,
        histogramRange,
        levels,
        viewport_policy,
        rgb_already_windowed: bool,
        tile_delta,
        mapping_state: DisplayMapping,
    ) -> TileLayerUpdateStats:
        """Publish uniforms/metadata while preserving the complete tile binding."""

        self._wgpu_mapping_state = mapping_state
        self._montage_display_mode = "wgpu_tile_layer"
        commands = [SetDisplayMapping(mapping_state)]
        camera = self._wgpu_camera_command()
        if camera is not None:
            commands.append(camera)
        start = perf_counter()
        report = self._submit_wgpu(tuple(commands))
        upload_ms = (perf_counter() - start) * 1000.0
        self._wgpu_last_report_uploads = int(report.uploads)
        return self._finish_wgpu_tiled_presentation(
            img,
            histogramPlotData=histogramPlotData,
            histogramRange=histogramRange,
            levels=levels,
            viewport_policy=viewport_policy,
            rgb_already_windowed=rgb_already_windowed,
            tile_delta=tile_delta,
            report=report,
            planned_upload_tiles=(),
            tile_residency_budget_bytes=int(
                (self._wgpu_committed or {}).get("budget_bytes", 0) or 0
            ),
            upload_ms=upload_ms,
            binding_fast_path=True,
            binding_incremental=False,
        )

    def _finish_wgpu_tiled_presentation(
        self,
        img,
        *,
        histogramPlotData,
        histogramRange,
        levels,
        viewport_policy,
        rgb_already_windowed: bool,
        tile_delta,
        report,
        planned_upload_tiles,
        tile_residency_budget_bytes: int,
        upload_ms: float,
        pool_growth_ms: float = 0.0,
        executor_initialization_ms: float = 0.0,
        binding_fast_path: bool,
        binding_incremental: bool,
        texture_prepare_ms: float = 0.0,
        texture_submit_ms: float = 0.0,
        texture_pack_ms: float = 0.0,
    ) -> TileLayerUpdateStats:
        """Finish one tiled commit from either the full or mapping-only path."""

        executor = self._wgpu_executor
        committed = self._wgpu_committed or {}
        committed_tiles = dict(committed.get("tiles", {}) or {})
        representation = str(committed.get("representation", "") or "")
        display_shape = tuple(int(value) for value in committed.get("display_shape", (1, 1)))
        if binding_fast_path:
            # The unchanged page-table generation proves the exact physical
            # residency set is still the one acknowledged by the full commit.
            presented = tuple(int(tile) for tile in committed.get("presented_tiles", ()))
        else:
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
            committed["presented_tiles"] = presented
            self._wgpu_committed = committed
        presented_identities = {tile: committed_tiles[tile]["identity"] for tile in presented}
        delta_upserts = dict(getattr(tile_delta, "upserts", {}) or {})
        committed_upserts = tuple(
            tile
            for tile, payload in delta_upserts.items()
            if presented_identities.get(int(tile)) == tile_ack_identity(payload)
        )

        # Shared shell bookkeeping (placeholder image, histogram bounds,
        # display levels) is the renderer's minimal set.
        self.image = img
        self.histogramSource = None
        # ``None`` means that this commit carries no histogram metadata; it is
        # not a command to erase the last accepted semantic histogram.
        if histogramPlotData is not None and not callable(histogramPlotData):
            previous_histogram = self.histogramPlotSource
            self.histogramPlotSource = histogramPlotData
            if (
                previous_histogram is not histogramPlotData
                or getattr(self.histogramImageItem, "image", None) is None
            ):
                self._bind_histogram_item(self.histogramImageItem)
                self._set_image_item_data(
                    self.histogramImageItem,
                    self._histogram_plot_data(None),
                    self._histogram_levels_for_display(levels),
                    role="histogram",
                )
        self.setHistogramDataBounds(histogramRange)
        level_lo, level_hi = (float(levels[0]), float(levels[1]))
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
        self._sync_wgpu_overlay_geometry()

        uploads = int(report.uploads)
        updated = tuple(planned_upload_tiles) if uploads > 0 else ()
        existing_items = 0 if binding_fast_path else len(committed_tiles) - len(updated)
        if binding_fast_path:
            self._wgpu_binding_fast_path_commits += 1
        elif binding_incremental:
            self._wgpu_binding_incremental_commits += 1
        else:
            self._wgpu_binding_full_republications += 1
        stats = TileLayerUpdateStats(
            visible_items=len(committed_tiles),
            presented_tiles=presented,
            committed_upserts=committed_upserts,
            presented_identities=presented_identities,
            updated_tiles=updated,
            items_created=0,
            items_updated=len(updated),
            items_skipped=len(committed_tiles) - len(updated),
            existing_items_shown=existing_items,
            resident_items=len(presented),
            storage_capacity=executor.pool_budget(representation),
            pool_growth_ms=float(pool_growth_ms),
            executor_initialization_ms=float(executor_initialization_ms),
            texture_uploads=uploads,
            texture_upload_bytes=int(report.upload_bytes),
            page_count=len(executor.page_table.resident_keys()),
            active_pages=len(
                {key for info in committed_tiles.values() for key in tuple(info["page_keys"])}
            ),
            estimated_gpu_bytes=int(executor.active_resident_bytes),
            budget_bytes=int(tile_residency_budget_bytes or 0),
            shader_uniform_updates=1,
            binding_fast_path_commits=int(binding_fast_path),
            binding_incremental_commits=int(binding_incremental),
            binding_full_republications=int(not binding_fast_path and not binding_incremental),
            upload_ms=upload_ms,
            texture_prepare_ms=float(texture_prepare_ms),
            texture_submit_ms=float(texture_submit_ms),
            texture_pack_ms=float(texture_pack_ms),
        )
        self._record_tile_layer_stats(stats)
        self._record_upload_timing("tile_layer_upload_ms", upload_ms)
        self._request_wgpu_canvas_draw(count_presentation=True)
        return stats

    def _wgpu_commit_plan(self, payloads, source_mapping, rgb_already_windowed):
        """Validate one commit; return representation and mapping uniforms.

        Everything outside the committed scope raises ``NotImplementedError``
        loudly instead of guessing (montages of N scalar/complex tiles and
        display-ready uint8 RGB tiles).
        """

        kinds = {tile: _wgpu_payload_kind(payload) for tile, payload in payloads.items()}
        unique = {kind.value for kind in kinds.values()}
        if len(unique) != 1:
            raise NotImplementedError(
                f"wgpu backend requires one texture representation per commit; got {sorted(unique)}"
            )
        kind = next(iter(kinds.values()))
        representation = _WGPU_REP_BY_KIND[kind]
        if kind == TexturePlaneKind.RGB8 and not rgb_already_windowed:
            representation = RGB_WINDOWED_RGBA32F
        display_mode = getattr(getattr(source_mapping, "display_mode", None), "value", None)
        scale = getattr(getattr(source_mapping, "scale", None), "value", None)
        if scale is None:
            scale = ShaderScale.LINEAR.value
        symlog_constant = float(getattr(source_mapping, "symlog_constant", 0.0) or 0.0)
        if representation == SCALAR_R32F:
            if display_mode not in (None, ShaderDisplayMode.SCALAR.value):
                raise NotImplementedError(
                    "wgpu backend renders scalar payloads with scalar display "
                    f"mode only; got {display_mode!r}"
                )
            return representation, "real", scale, symlog_constant, False
        if representation == COMPLEX_RG32F:
            if display_mode not in (
                None,
                ShaderDisplayMode.COMPLEX.value,
                ShaderDisplayMode.PHASE_COLOR.value,
            ):
                raise NotImplementedError(
                    f"wgpu backend cannot render complex display mode {display_mode!r}"
                )
            component = getattr(getattr(source_mapping, "component", None), "value", None)
            if component is None:
                component = ShaderComponent.REAL.value
            if component not in _WGPU_COMPONENT_MODES:
                raise NotImplementedError(
                    f"wgpu backend cannot render shader component {component!r}"
                )
            if display_mode == ShaderDisplayMode.PHASE_COLOR.value and component not in (
                ShaderComponent.ABS.value,
                ShaderComponent.ANGLE.value,
                ShaderComponent.COMPLEX_PHASE.value,
            ):
                raise NotImplementedError(
                    "wgpu backend renders phase-color for phase or magnitude "
                    f"components only; got {component!r}"
                )
            return (
                representation,
                _WGPU_COMPONENT_MODES[component],
                scale,
                symlog_constant,
                display_mode == ShaderDisplayMode.PHASE_COLOR.value,
            )
        expected_mode = (
            ShaderDisplayMode.RGB_DISPLAY_READY.value
            if rgb_already_windowed
            else ShaderDisplayMode.RGB_WINDOWED.value
        )
        if display_mode not in (None, expected_mode):
            raise NotImplementedError(
                f"wgpu backend cannot render RGB display mode {display_mode!r}"
            )
        for tile, payload in payloads.items():
            texture = np.asarray(
                payload.texture_data if payload.texture_data is not None else payload.image
            )
            if texture.ndim != 3 or texture.shape[-1] not in (3, 4):
                raise NotImplementedError(
                    f"wgpu RGB tile {tile} payload must have shape (h, w, 3|4); "
                    f"got {texture.dtype} {texture.shape}"
                )
            if rgb_already_windowed and texture.dtype != np.uint8:
                raise NotImplementedError(
                    f"wgpu display-ready RGB tile {tile} must be uint8; got {texture.dtype}"
                )
        return representation, "real", scale, symlog_constant, False

    def _wgpu_payload_texture(self, payload, representation) -> np.ndarray:
        """Packed upload plane for one payload, from a worker where possible.

        The mailbox hands over a worker-built plane only under the identity and
        representation this commit is submitting; anything else is dropped and
        packed here, exactly as before the hand-off existed.
        """

        prepared = self.preparedTiledUploads.take(
            int(payload.tile_number),
            prepared_upload_key(payload, representation),
        )
        if prepared is not None:
            return prepared
        return wgpu_packed_payload_texture(payload, representation)

    def tiledUploadPreparations(self, payloads, *, levels, rgb_already_windowed: bool = False):
        """Pack each payload into its upload plane ahead of the commit.

        The representation is derived exactly as ``_wgpu_commit_plan`` derives
        it, so a plane prepared here carries the key the commit will ask for. A
        payload whose kind is not packable is skipped rather than guessed at:
        the commit's own validation stays the single place that refuses.

        """

        del levels
        mailbox = self.preparedTiledUploads
        rows = []
        for tile, payload in dict(payloads or {}).items():
            representation = _wgpu_preparable_representation(payload, bool(rgb_already_windowed))
            if representation is None:
                continue
            if not wgpu_pack_saves_work(payload, representation):
                mailbox.note_no_work()
                continue
            slot = int(getattr(payload, "tile_number", tile))
            task = mailbox.plan(slot, prepared_upload_key(payload, representation))
            rows.append((task, _wgpu_pack_preparation(task, payload, representation)))
        return tuple(rows)

    def _wgpu_reusable_native_texture(
        self,
        payload,
        *,
        binding: _WgpuPayloadBinding,
        representation: str,
        selected_lod: int,
    ) -> np.ndarray | None:
        """Return a complete exact native plane worth warming with a coarse draw.

        A zoomed-out full montage commonly presents LOD1/2 while the payload
        still owns its exact semantic plane. Uploading that plane through the
        same bounded tile commit populates the already-budgeted L0 pages once;
        a later displayed-axis crop then becomes only a source-window rebind.
        Display-ready/windowed RGB is skipped because ``semantic_data`` alone
        does not carry its paired luminance representation.

        The qualifying test is the CANONICAL plane shape, which is also what
        ``_wgpu_native_prefetch_page_keys`` keys and what the caller uploads
        into: whether this payload presents all of that plane is a separate
        question, answered by the source origin the caller binds at. A cropped
        payload whose evaluation was widened to the plane
        (``canonical_plane_residency_source``) carries exactly that array while
        presenting a sub-rect. Demanding a whole-plane ``source_rect`` as well
        is what left a born-cropped montage on crop-local pages forever:
        nothing ever presents its whole plane, so nothing ever warmed the
        canonical one. A window-local ``semantic_data`` cannot slip through —
        it matches ``plane_shape`` only when the window IS the plane.
        """

        if representation not in {SCALAR_R32F, COMPLEX_RG32F}:
            return None
        if int(selected_lod) <= 0 and binding.source_anchored:
            # An exact native binding already resolved canonical pages on its
            # own path; replacing it here would only duplicate that geometry.
            return None
        semantic = getattr(payload, "native_residency_data", None)
        if semantic is None:
            semantic = getattr(payload, "semantic_data", None)
        if semantic is None:
            return None
        semantic = np.asarray(semantic)
        anchor = getattr(payload, "source_anchor", None)
        plane_shape = tuple(int(value) for value in (getattr(anchor, "plane_shape", ()) or ()))
        if len(plane_shape) != 2:
            return None
        if tuple(int(value) for value in semantic.shape[:2]) != plane_shape:
            return None
        kind = (
            TexturePlaneKind.COMPLEX_RG32F
            if representation == COMPLEX_RG32F
            else TexturePlaneKind.SCALAR_R32F
        )
        return pack_texture_data(semantic, kind)

    def _wgpu_page_block(self, texture, chunk_y, chunk_x, representation) -> np.ndarray:
        from arrayscope.gpu.wgpu_executor import PAGE

        if representation == SCALAR_R32F:
            page = np.zeros((PAGE, PAGE), np.float32)
        elif representation == COMPLEX_RG32F:
            page = np.zeros((PAGE, PAGE, 2), np.float32)
        elif representation == RGB8:
            page = np.zeros((PAGE, PAGE, 3), np.uint8)
        else:
            page = np.zeros((PAGE, PAGE, 4), np.float32)
        block = texture[
            chunk_y * PAGE : (chunk_y + 1) * PAGE,
            chunk_x * PAGE : (chunk_x + 1) * PAGE,
        ]
        page[: block.shape[0], : block.shape[1]] = block
        return page

    def _wgpu_resolve_lut_bytes(self, shader_mapping) -> bytes | None:
        display_mode = getattr(getattr(shader_mapping, "display_mode", None), "value", None)
        if display_mode == ShaderDisplayMode.PHASE_COLOR.value:
            explicit = getattr(shader_mapping, "lut_data", None)
            if explicit is not None:
                return _resample_lut_to_rgba256(explicit)
            # A bare phase-color mapping means the canonical cyclic phase LUT
            # (the shared display-mapping template): the view's initial
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
                self._apply_presentation_histogram_range(
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
        self._wgpu_tiled_binding_signature = None
        self._wgpu_histogram_evidence.clear()
        self._wgpu_histogram_evidence_ready.clear()
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
        self._wgpu_tiled_binding_signature = None
        self._clear_wgpu_tiles()
        self._montage_display_mode = "none"
        self.imageItem.setVisible(False)

    def reset_tiled_residency(self, reason: str) -> None:
        self.discardPreparedTiledUploads()
        executor = self._wgpu_executor
        if executor is not None:
            executor.replace_resident_pin_set(self._wgpu_atomic_warm_pin_owner, ())
            evictions = tuple(EvictChunk(key) for key in executor.page_table.resident_keys())
            self._submit_wgpu((*evictions, UpdateTileInstances(())))
            self._request_wgpu_canvas_draw()
        self._wgpu_committed = None
        self._wgpu_tiled_binding_signature = None
        self._wgpu_histogram_evidence.clear()
        self._wgpu_histogram_evidence_ready.clear()
        self._last_wgpu_structure_key = None
        self._last_wgpu_viewport_key = None
        self._last_wgpu_tiled_reset_reason = str(reason)
        self._montage_display_mode = "none"
        self.imageItem.setVisible(False)

    def clear(self):
        super().clear()
        self.reset_tiled_residency("view-clear")

    def warmTiledResidency(
        self,
        *,
        payloads,
        rgb_already_windowed: bool = False,
        tile_delta=None,
        tile_residency_budget_bytes: int = 0,
        **_kwargs,
    ):
        """Ensure payload pages without changing the bound presentation."""

        from arrayscope.gpu.wgpu_executor import PAGE

        payloads = {int(tile): payload for tile, payload in dict(payloads or {}).items()}
        if not payloads:
            return None
        source_mapping = common_shader_mapping(
            getattr(payload, "shader_mapping", None) for payload in payloads.values()
        )
        representation, *_mapping = self._wgpu_commit_plan(
            payloads, source_mapping, rgb_already_windowed
        )
        textures = {
            tile: self._wgpu_payload_texture(payload, representation)
            for tile, payload in payloads.items()
        }
        lod_geometry = {
            tile: _wgpu_payload_declared_lod_geometry(payloads[tile], textures[tile])
            for tile in payloads
        }
        # Hidden atomic warming is delivered in small GUI-budget batches, but
        # residency capacity belongs to the whole successor.  Sizing the pool
        # from only this batch lets later batches evict pages the coordinator
        # has already marked warm, stranding the final atomic swap with only
        # its last one or two tiles resident.
        atomic_handoff = bool(getattr(tile_delta, "atomic_handoff", False))
        capacity_payloads = payloads
        transaction_payloads = dict(getattr(tile_delta, "upserts", {}) or {})
        if transaction_payloads:
            # Pool capacity belongs to the complete presentation transaction,
            # even when the coordinator delivers its warm work in small GUI
            # batches. This is equally true for the initial montage fill and
            # for an atomic successor.
            capacity_payloads = transaction_payloads
        capacity_geometry = {}
        for tile, payload in capacity_payloads.items():
            texture = np.asarray(
                payload.texture_data if payload.texture_data is not None else payload.image
            )
            capacity_geometry[int(tile)] = (
                _wgpu_payload_declared_lod_geometry(payload, texture),
                tuple(int(value) for value in texture.shape[:2]),
            )
        pages_needed = sum(
            _wgpu_payload_capacity_page_count(
                capacity_payloads[tile],
                texture_shape=texture_shape,
                representation=representation,
                selected_lod=geometry[0],
            )
            for tile, (geometry, texture_shape) in capacity_geometry.items()
        )
        pages_preferred = sum(
            _wgpu_ladder_page_count(source_shape, max_lod=lod_level)
            for (lod_level, source_shape), _texture_shape in capacity_geometry.values()
        )
        transaction_page_keys = frozenset()
        if atomic_handoff:
            current_executor = self._wgpu_executor
            current_resident = (
                () if current_executor is None else current_executor.page_table.resident_keys()
            )
            transaction_page_keys = frozenset(
                key
                for tile, payload in capacity_payloads.items()
                for key in (
                    _wgpu_native_prefetch_page_keys(
                        payload,
                        representation=representation,
                        mapping_mode=_mapping[0],
                        selected_lod=capacity_geometry[int(tile)][0][0],
                    )
                    or _wgpu_payload_page_keys(
                        payload,
                        representation=representation,
                        mapping_mode=_mapping[0],
                        resident_keys=current_resident,
                    )
                )
            )
            if current_executor is not None:
                # Replace any superseded hidden generation first, then size
                # the physical working set from the canonical page table:
                # bound predecessor coverage plus the complete successor.
                current_executor.replace_resident_pin_set(
                    self._wgpu_atomic_warm_pin_owner,
                    transaction_page_keys,
                )
                protected = frozenset(
                    key
                    for key in current_executor.page_table.resident_keys()
                    if key.representation == representation
                    and current_executor.page_table.is_pinned(key)
                )
                pages_needed = max(pages_needed, len(protected | transaction_page_keys))
        executor = self._ensure_wgpu_executor(
            {representation: pages_needed},
            preferred_pages={representation: pages_preferred},
            residency_budget_bytes=int(tile_residency_budget_bytes or 0),
        )
        if atomic_handoff:
            # A newer atomic generation replaces the old generation's
            # ownership here. Repeated batches for one generation simply
            # refresh the resident subset already uploaded.
            executor.replace_resident_pin_set(
                self._wgpu_atomic_warm_pin_owner,
                transaction_page_keys,
            )
        commands = []
        planned_resident = set(executor.page_table.resident_keys())
        for tile in sorted(payloads):
            texture = textures[tile]
            selected_lod = int(lod_geometry[tile][0])
            binding = _wgpu_payload_binding(
                payloads[tile],
                texture,
                representation=representation,
                mapping_mode=_mapping[0],
                resident_keys=planned_resident,
            )
            reusable_native = self._wgpu_reusable_native_texture(
                payloads[tile],
                binding=binding,
                representation=representation,
                selected_lod=selected_lod,
            )
            if reusable_native is not None:
                native_keys = _wgpu_native_prefetch_page_keys(
                    payloads[tile],
                    representation=representation,
                    mapping_mode=_mapping[0],
                    selected_lod=selected_lod,
                )
                native_grid_w = -(-int(reusable_native.shape[1]) // PAGE)
                for index, key in enumerate(native_keys):
                    if key in planned_resident:
                        continue
                    chunk_y, chunk_x = divmod(index, native_grid_w)
                    commands.append(
                        EnsureChunkResident(
                            key,
                            self._wgpu_page_block(
                                reusable_native,
                                chunk_y,
                                chunk_x,
                                representation,
                            ),
                        )
                    )
                    planned_resident.add(key)
                # Native exact pages are both higher quality and reusable
                # across source-window shifts. Uploading this payload's
                # reduced page as well would duplicate the successor working
                # set and create avoidable eviction pressure during preview
                # handoff.
                continue
            for key, (chunk_y, chunk_x) in zip(
                binding.page_keys,
                binding.upload_chunks,
                strict=True,
            ):
                if key in planned_resident:
                    continue
                commands.append(
                    EnsureChunkResident(
                        key,
                        self._wgpu_page_block(texture, chunk_y, chunk_x, representation),
                    )
                )
                planned_resident.add(key)
        report = self._submit_wgpu(tuple(commands))
        if atomic_handoff:
            executor.replace_resident_pin_set(
                self._wgpu_atomic_warm_pin_owner,
                transaction_page_keys,
            )
        return report

    def warmPlaneResidency(self, payload) -> bool:
        kind = _wgpu_payload_kind(payload)
        texture = np.asarray(
            payload.texture_data if payload.texture_data is not None else payload.image
        )
        rgb = bool(
            kind == TexturePlaneKind.RGB8
            and texture.dtype == np.uint8
            and getattr(payload, "histogram_data", None) is None
        )
        self.warmTiledResidency(
            payloads={int(getattr(payload, "tile_number", 0)): payload},
            rgb_already_windowed=rgb,
        )
        return self.tiledPayloadResident(payload)

    def tiledPayloadResident(self, payload) -> bool:
        executor = self._wgpu_executor
        if executor is None:
            return False
        kind = _wgpu_payload_kind(payload)
        texture_data = np.asarray(
            payload.texture_data if payload.texture_data is not None else payload.image
        )
        representation = _WGPU_REP_BY_KIND[kind]
        if kind == TexturePlaneKind.RGB8 and (
            texture_data.dtype != np.uint8 or getattr(payload, "histogram_data", None) is not None
        ):
            representation = RGB_WINDOWED_RGBA32F
        source_mapping = common_shader_mapping((getattr(payload, "shader_mapping", None),))
        try:
            _representation, mapping_mode, *_mapping = self._wgpu_commit_plan(
                {int(getattr(payload, "tile_number", 0)): payload},
                source_mapping,
                bool(kind == TexturePlaneKind.RGB8 and representation == RGB8),
            )
        except NotImplementedError:
            # This method is a physical-residency predicate, including while a
            # semantic transition temporarily mixes predecessor and successor
            # representations. A payload the current commit path cannot bind
            # is not resident for that transaction; probing it must not turn a
            # normal cold fallback into a swallowed presentation exception.
            return False
        binding = self._cached_wgpu_residency_binding(
            payload,
            representation=representation,
            mapping_mode=mapping_mode,
        )
        return all(key in executor.page_table for key in binding.page_keys)

    def _cached_wgpu_residency_binding(
        self,
        payload,
        *,
        representation: str,
        mapping_mode: str,
    ) -> _WgpuPayloadBinding:
        """Reuse immutable binding geometry until physical residency changes."""

        executor = self._wgpu_executor
        if executor is None:  # pragma: no cover - caller owns this guard
            raise RuntimeError("wgpu residency binding requested without an executor")
        generation = int(executor.page_table.generation)
        if generation != self._wgpu_residency_binding_generation:
            self._wgpu_residency_binding_cache.clear()
            self._wgpu_residency_binding_generation = generation
        cache_key = (id(payload), str(representation), str(mapping_mode))
        cached = self._wgpu_residency_binding_cache.get(cache_key)
        if cached is not None and cached[0]() is payload:
            self._wgpu_residency_binding_cache_hits += 1
            return cached[1]
        self._wgpu_residency_binding_cache_misses += 1
        # Dead weakrefs are cheap, but a long descriptor-only session must
        # not grow this index without bound.
        if len(self._wgpu_residency_binding_cache) >= 1024:
            self._wgpu_residency_binding_cache.clear()
        binding = _wgpu_payload_binding(
            payload,
            np.asarray(payload.texture_data if payload.texture_data is not None else payload.image),
            representation=representation,
            mapping_mode=mapping_mode,
            resident_keys=executor.page_table.resident_keys(),
        )
        self._wgpu_residency_binding_cache[cache_key] = (weakref.ref(payload), binding)
        return binding

    def _tiled_presentation_layer(self):
        return None

    def physicalVisibleTileCount(self) -> int:
        """Count committed tiles whose submitted page bindings still exist.

        This is the allocation-light physical-truth query for high-frequency
        continuity probes. Detailed diagnostics still use
        :meth:`tileTruthPhysicalRows`, but a 1 ms sampler must not construct
        row dictionaries, page-pool snapshots, and compression labels merely
        to learn whether any submitted tile disappeared.
        """

        executor = self._wgpu_executor
        if executor is None:
            return 0
        page_table = executor.page_table
        committed = self._wgpu_committed or {}
        tiles = committed.get("tiles", {}) or {}
        return sum(
            1
            for info in tiles.values()
            if (page_keys := tuple(info.get("page_keys", ()) or ()))
            and all(key in page_table for key in page_keys)
        )

    def tileTruthPhysicalRows(self) -> dict[int, dict[str, object]]:
        """Describe the page-backed tile instances submitted to the executor.

        WGPU deliberately has no Qt tile-layer object, so inheriting the
        shell implementation returned an empty mapping even while the native
        surface drew a complete montage.  The committed command state plus
        current page-table residency is the corresponding physical owner.
        """

        executor = self._wgpu_executor
        committed = self._wgpu_committed or {}
        if executor is None:
            return {}
        representation = str(committed.get("representation", "") or "")
        mapping = self._wgpu_mapping_state
        rows: dict[int, dict[str, object]] = {}
        for tile_number, raw_info in dict(committed.get("tiles", {}) or {}).items():
            info = dict(raw_info or {})
            page_keys = tuple(info.get("page_keys", ()) or ())
            if not page_keys or any(executor.page_table.lookup(key) is None for key in page_keys):
                continue
            x, y, width, height = tuple(info.get("world_rect", (0.0, 0.0, 0.0, 0.0)))
            bounds = (float(x), float(y), float(x + width), float(y + height))
            source_width, source_height = tuple(info.get("src_size", (0.0, 0.0)))
            binding_rows = tuple(
                {
                    "target_key": key,
                    "actual_key": key,
                    "actual_lod": key.lod,
                    "scale": (1.0, 1.0),
                    "offset": (0.0, 0.0),
                    "quality": (
                        f"lossy_{executor.codec_family}"
                        if executor.page_is_compressed(key)
                        else "exact"
                    ),
                }
                for key in page_keys
            )
            physical_quality = (
                "exact"
                if all(row["quality"] == "exact" for row in binding_rows)
                else "lossy_compressed"
            )
            rows[int(tile_number)] = {
                "physical_texture_kind": representation,
                "physical_storage_mode": "wgpu_page_table",
                "physical_texture_dtype": str(_WGPU_REP_DTYPES.get(representation, "")),
                "physical_texture_shape": (int(source_height), int(source_width)),
                "physical_source_origin_xy": tuple(
                    float(value) for value in info.get("src_origin", (0.0, 0.0))
                ),
                "physical_source_size_xy": (float(source_width), float(source_height)),
                "physical_mapping_mode": str(getattr(mapping, "mode", "") or ""),
                "physical_component_mode": None,
                "physical_levels": (
                    float(getattr(mapping, "level_lo", 0.0)),
                    float(getattr(mapping, "level_hi", 1.0)),
                ),
                "physical_acknowledged_identity": info.get("identity"),
                "physical_target_identity": _physical_target_token(info.get("identity")),
                "physical_lod_level": int(info.get("lod_level", 0) or 0),
                "physical_quality": physical_quality,
                "physical_draw_world_rects": (bounds,),
                "physical_draw_uv_rects": ((0.0, 0.0, 1.0, 1.0),),
                "physical_draw_world_bounds": bounds,
                "physical_expected_world_rect": bounds,
                "physical_draw_bounds_match_layout": True,
                "physical_page_bindings": binding_rows,
            }
        return rows

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
            scale=self._wgpu_mapping_state.scale,
            symlog_constant=self._wgpu_mapping_state.symlog_constant,
            phase_color=self._wgpu_mapping_state.phase_color,
            pixel_grid=self._pixel_grid_enabled,
            clip_indicator=self._clip_indicator_enabled,
            minification_filter=self._minification_filter_enabled,
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
            scale=self._wgpu_mapping_state.scale,
            symlog_constant=self._wgpu_mapping_state.symlog_constant,
            phase_color=self._wgpu_mapping_state.phase_color,
            pixel_grid=self._pixel_grid_enabled,
            clip_indicator=self._clip_indicator_enabled,
            minification_filter=self._minification_filter_enabled,
        )
        if self._montage_display_mode == "wgpu_tile_layer":
            self._submit_wgpu((SetDisplayMapping(self._wgpu_mapping_state),))
            self._request_wgpu_canvas_draw(count_presentation=True)

    # ---- shader display aids (Stage A + C1) -----------------------------------

    #: Mapping-field name -> the view attribute that mirrors it.  One table so
    #: a new aid is one row here plus its accessor pair, not another branch.
    _LEGIBILITY_FLAG_ATTRS: ClassVar[dict[str, str]] = {
        "pixel_grid": "_pixel_grid_enabled",
        "clip_indicator": "_clip_indicator_enabled",
        "minification_filter": "_minification_filter_enabled",
    }

    def wgpuPixelGridEnabled(self) -> bool:
        return bool(self._pixel_grid_enabled)

    def wgpuClipIndicatorEnabled(self) -> bool:
        return bool(self._clip_indicator_enabled)

    def wgpuMinificationFilterEnabled(self) -> bool:
        return bool(self._minification_filter_enabled)

    def setWgpuPixelGridEnabled(self, enabled: bool) -> None:
        """Toggle the zoom-gated per-texel pixel grid on the live view."""

        self._set_legibility_flag("pixel_grid", bool(enabled))

    def setWgpuClipIndicatorEnabled(self, enabled: bool) -> None:
        """Toggle the out-of-window clip markers on the live view."""

        self._set_legibility_flag("clip_indicator", bool(enabled))

    def setWgpuMinificationFilterEnabled(self, enabled: bool) -> None:
        """Toggle footprint averaging on minified draws on the live view."""

        self._set_legibility_flag("minification_filter", bool(enabled))

    def _set_legibility_flag(self, name: str, enabled: bool) -> None:
        """Update one display-aid flag and, if live, re-submit the mapping.

        These are pure shader-uniform flags: no residency, no upload — a
        ``SetDisplayMapping`` re-submit plus a redraw is the whole cost, so the
        toggle is felt immediately without rebuilding the view.
        """

        attr = self._LEGIBILITY_FLAG_ATTRS[name]
        if getattr(self, attr) == enabled:
            return
        setattr(self, attr, enabled)
        self._wgpu_mapping_state = replace(self._wgpu_mapping_state, **{name: enabled})
        if self._wgpu_executor is not None and self._montage_display_mode == "wgpu_tile_layer":
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

    def grabPresentedFramebuffer(self) -> np.ndarray | None:
        """Physical-truth harness capture for the screen present path.

        Screen-mode pixels live in the compositor swapchain, which a Qt
        widget grab cannot see (a paint-less native child rasterizes as
        nothing).  Re-render the executor's CURRENT bound state — the exact
        tiles/overlay/mapping/camera the last swapchain present drew — into
        the internal offscreen target and read it back.  Returns ``None`` on
        the bitmap path (the widget grab is already honest there) and when
        no executor exists yet.
        """

        if self._wgpu_present_method != "screen":
            return None
        executor = self._wgpu_executor
        if executor is None:
            return None
        self._submit_wgpu(())
        return executor.read_target()

    def wgpuPresentMethod(self) -> str:
        """Effective present method ("bitmap" or "screen") after fallback."""

        return str(getattr(self, "_wgpu_present_method", "bitmap"))

    def wgpuPresentMethodFallbackReason(self) -> str:
        """Why a requested screen path fell back to bitmap ("" when it did not)."""

        return str(getattr(self, "_wgpu_present_method_fallback_reason", "") or "")

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
        adapter = getattr(getattr(executor, "device", None), "adapter", None)
        adapter_info = dict(getattr(adapter, "info", {}) or {})
        drawn_tiles = tuple(self.tileTruthPhysicalRows())
        committed_tiles = tuple(sorted((self._wgpu_committed or {}).get("tiles", ())))
        preview_atlas = (self._wgpu_committed or {}).get("preview_atlas")
        preview_atlas_active_tiles = tuple(
            (self._wgpu_committed or {}).get("preview_atlas_active_tiles", ()) or ()
        )
        resident = len(executor.page_table.resident_keys()) if executor is not None else 0
        page_pools = () if executor is None else tuple(executor.pool_diagnostics_snapshot())
        diagnostics: dict[str, object] = {
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
            "presented_tile_count": len(committed_tiles),
            "presented_tiles": committed_tiles,
            "wgpu_preview_atlas_tiles": len(preview_atlas_active_tiles),
            "wgpu_preview_atlas_pages": (
                0 if preview_atlas is None else len(preview_atlas.page_keys)
            ),
            "page_table_resident_count": resident,
            "wgpu_uploads_total": int(getattr(executor, "uploads_total", 0) or 0),
            "wgpu_upload_bytes_total": int(getattr(executor, "texture_upload_bytes_total", 0) or 0),
            "wgpu_uploads_by_level": (
                ()
                if executor is None
                else tuple(dict(row) for row in executor.upload_rows_by_level())
            ),
            "wgpu_compressed_uploads_total": int(
                getattr(executor, "compressed_uploads_total", 0) or 0
            ),
            "wgpu_compressed_fallbacks_total": int(
                getattr(executor, "compressed_fallbacks_total", 0) or 0
            ),
            "wgpu_active_resident_bytes": int(getattr(executor, "active_resident_bytes", 0) or 0),
            "wgpu_allocated_pool_bytes": int(getattr(executor, "allocated_pool_bytes", 0) or 0),
            "wgpu_page_pools": page_pools,
            "wgpu_residency_binding_cache_hits": int(self._wgpu_residency_binding_cache_hits),
            "wgpu_residency_binding_cache_misses": int(self._wgpu_residency_binding_cache_misses),
            "wgpu_binding_fast_path_commits": int(self._wgpu_binding_fast_path_commits),
            "wgpu_binding_incremental_commits": int(self._wgpu_binding_incremental_commits),
            "wgpu_binding_full_republications": int(self._wgpu_binding_full_republications),
            "wgpu_atomic_warm_pinned_pages": (
                0
                if executor is None
                else len(executor.resident_pin_set(self._wgpu_atomic_warm_pin_owner))
            ),
            "wgpu_last_pool_exhaustion": str(getattr(executor, "last_pool_exhaustion", "") or ""),
            "wgpu_pool_grows_total": int(getattr(executor, "pool_grows_total", 0) or 0),
            "wgpu_pool_growth_copy_bytes_total": int(
                getattr(executor, "pool_growth_copy_bytes_total", 0) or 0
            ),
            "wgpu_pool_growth_ms_total": float(
                getattr(executor, "pool_growth_ms_total", 0.0) or 0.0
            ),
            "wgpu_codec_family": str(getattr(executor, "codec_family", "none") or "none"),
            "wgpu_codec_min_psnr_db": float(
                getattr(executor, "codec_min_psnr_db", _WGPU_CODEC_MIN_PSNR_DB)
            ),
            "wgpu_adapter": str(adapter_info.get("device", "") or ""),
            "wgpu_adapter_type": str(adapter_info.get("adapter_type", "") or ""),
            "wgpu_backend_type": str(adapter_info.get("backend_type", "") or ""),
            "wgpu_power_preference": _SHARED_WGPU_POWER_PREFERENCE,
            "wgpu_plane_lookup_candidates_total": int(
                getattr(executor, "plane_lookup_candidates_total", 0) or 0
            ),
            "wgpu_last_report_uploads": int(getattr(self, "_wgpu_last_report_uploads", 0) or 0),
            "wgpu_last_draw_error": str(getattr(self, "_wgpu_last_draw_error", "") or ""),
            "wgpu_histogram_evidence_pending": len(
                set(self._wgpu_histogram_evidence) - set(self._wgpu_histogram_evidence_ready)
            ),
            "wgpu_present_method": self.wgpuPresentMethod(),
            "wgpu_present_method_requested": str(
                getattr(self, "_wgpu_present_method_requested", "bitmap")
            ),
            "wgpu_present_method_fallback_reason": self.wgpuPresentMethodFallbackReason(),
            "wgpu_screen_present_mode": str(getattr(self._wgpu_canvas, "present_mode", "") or ""),
            "wgpu_screen_presents": int(getattr(self, "_wgpu_screen_presents", 0) or 0),
            "wgpu_screen_acquire_ms_last": float(
                getattr(self, "_wgpu_screen_acquire_ms_last", 0.0) or 0.0
            ),
            "wgpu_screen_acquire_ms_max": float(
                getattr(self, "_wgpu_screen_acquire_ms_max", 0.0) or 0.0
            ),
            "wgpu_screen_present_ms_last": float(
                getattr(self, "_wgpu_screen_present_ms_last", 0.0) or 0.0
            ),
            "wgpu_screen_present_ms_max": float(
                getattr(self, "_wgpu_screen_present_ms_max", 0.0) or 0.0
            ),
        }
        # Frame cadence: distributions and phase, not just last+max.  Only the
        # screen path paces its own frames, so only it can answer this; the
        # bitmap canvas has no such record and contributes nothing.
        snapshot = getattr(self._wgpu_canvas, "frame_timing_snapshot", None)
        if callable(snapshot):
            diagnostics.update({f"wgpu_screen_{key}": value for key, value in snapshot().items()})
        return diagnostics

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
            teardown = getattr(canvas, "teardown", None)
            if callable(teardown):  # screen path: release the swapchain first
                with contextlib.suppress(Exception):
                    teardown()
                self._wgpu_context = None
                self._wgpu_context_format = None
            with contextlib.suppress(Exception):
                canvas.close()
        super().teardown_surface()


def _physical_target_token(identity) -> str:
    """Stable semantic tile identity, deliberately excluding LOD and payload."""

    return repr(
        tuple(
            getattr(identity, field, None)
            for field in (
                "document_generation",
                "operation_key",
                "source_index",
                "image_axes",
                "axis_flips",
                "channel",
                "complex_mapping",
                "texture_kind",
                "semantic_generation",
            )
        )
    )


def _display_axes_transposed(geometry) -> bool:
    """Whether the committed display is X/Y transposed vs canonical order.

    Canonical (sorted) image axes render as-is; a reversed pair
    (``image_axes[0] > image_axes[1]``) means the swap is applied as a display
    transform (per-tile UV axis swap in the vertex shader).
    """

    image_axes = getattr(getattr(geometry, "view_state", None), "image_axes", None) or ()
    image_axes = tuple(int(axis) for axis in image_axes)
    return len(image_axes) == 2 and image_axes[0] > image_axes[1]


def _wgpu_layout_identity(geometry, *, frame_plan=None) -> object:
    """Canonical immutable identity of tile placement, without rebuilding it."""

    if frame_plan is not None:
        return (
            getattr(frame_plan, "semantic_key", None),
            tuple(getattr(frame_plan, "scene_region_signature", ()) or ()),
        )
    montage = getattr(geometry, "montage", None)
    if montage is None:
        return None
    return (
        tuple(int(value) for value in getattr(montage, "indices", ()) or ()),
        tuple(int(value) for value in getattr(montage, "tile_shape", ()) or ()),
        int(getattr(montage, "columns", 0) or 0),
        int(getattr(montage, "rows", 0) or 0),
        int(getattr(montage, "gap", 0) or 0),
    )


def _wgpu_physical_payload_identities(
    payloads,
) -> tuple[tuple[int, object], ...] | None:
    """Return canonical tile-binding identities, excluding presentation levels.

    ``TileIdentity`` already owns document/operation/source selection, LOD,
    representation, complex mapping, semantic generation, and quality.
    Its real/imag plane provenance is intentionally excluded from normal
    semantic equality, so include those records explicitly here: pointer,
    shape, strides, and dtype are physical binding inputs even when the
    source pixels mean the same thing. ``TilePresentationIdentity`` is not
    consulted; levels/LUT/scale are uniform state and are the reason this
    mapping-only path exists.
    """

    rows = []
    for tile in sorted(payloads):
        identity = tile_ack_identity(payloads[tile])
        if not isinstance(identity, TileIdentity):
            return None
        rows.append(
            (
                int(tile),
                (
                    identity,
                    identity.real_plane,
                    identity.imag_plane,
                ),
            )
        )
    return tuple(rows)


def _wgpu_preview_binding_identity(payload) -> object:
    """Physical identity of bytes admitted to one preview atlas slot.

    Live typed payloads already carry source-plane provenance outside
    ``TileIdentity`` equality.  The array record also keeps legacy/test
    payloads honest: a new buffer under the same opaque ``source_id`` must
    rebuild the atlas rather than reuse stale texels.
    """

    identity = tile_ack_identity(payload)
    texture = np.asarray(
        payload.texture_data if payload.texture_data is not None else payload.image
    )
    pointer = int(texture.__array_interface__["data"][0])
    return (
        identity,
        getattr(identity, "real_plane", None),
        getattr(identity, "imag_plane", None),
        getattr(payload, "source_id", None),
        pointer,
        tuple(int(value) for value in texture.shape),
        tuple(int(value) for value in texture.strides),
        str(texture.dtype),
    )


def _wgpu_preview_payloads(payloads) -> dict[int, object]:
    """Large-montage preview payloads eligible for compact physical backing."""

    return {
        int(tile): payload
        for tile, payload in dict(payloads or {}).items()
        if str(getattr(payload, "quality", "exact") or "exact") == "preview"
    }


def _wgpu_reusable_preview_atlas(
    previous,
    preview_payloads,
    *,
    representation: str,
    lod_reducer: str,
) -> _WgpuPreviewAtlas | None:
    """Return a prior atlas when every still-preview slot names the same bytes."""

    if not isinstance(previous, _WgpuPreviewAtlas):
        return None
    if previous.representation != str(representation) or previous.lod_reducer != str(lod_reducer):
        return None
    slots = {int(slot.tile_number): slot for slot in previous.slots}
    for tile, payload in preview_payloads.items():
        slot = slots.get(int(tile))
        if (
            slot is None
            or slot.payload_identity != tile_ack_identity(payload)
            or slot.binding_identity != _wgpu_preview_binding_identity(payload)
        ):
            return None
    return previous


def _wgpu_preview_page_array(representation: str, page: int) -> np.ndarray:
    """Allocate one zero-padded executor page in the representation's format."""

    if representation == SCALAR_R32F:
        return np.zeros((page, page), np.float32)
    if representation == COMPLEX_RG32F:
        return np.zeros((page, page, 2), np.float32)
    if representation == RGB8:
        return np.zeros((page, page, 3), np.uint8)
    if representation == RGB_WINDOWED_RGBA32F:
        return np.zeros((page, page, 4), np.float32)
    raise ValueError(f"unsupported wgpu preview atlas representation {representation!r}")


def _build_wgpu_preview_atlas(
    preview_payloads,
    textures,
    *,
    representation: str,
    lod_reducer: str,
) -> _WgpuPreviewAtlas | None:
    """Shelf-pack a large preview set into at most two native executor pages."""

    from arrayscope.gpu.wgpu_executor import PAGE, plane_chunk_key

    if len(preview_payloads) < _WGPU_PREVIEW_ATLAS_MIN_TILES:
        return None
    token = object()
    pages: list[np.ndarray] = []
    slots: list[_WgpuPreviewAtlasSlot] = []
    page_index = -1
    cursor_x = cursor_y = row_height = 0

    def start_page() -> bool:
        nonlocal page_index, cursor_x, cursor_y, row_height
        if len(pages) >= _WGPU_PREVIEW_ATLAS_MAX_PAGES:
            return False
        pages.append(_wgpu_preview_page_array(representation, PAGE))
        page_index += 1
        cursor_x = cursor_y = row_height = 0
        return True

    if not start_page():
        return None
    for tile in sorted(preview_payloads):
        texture = np.asarray(textures[tile])
        height, width = (int(value) for value in texture.shape[:2])
        if height < 1 or width < 1 or height > PAGE or width > PAGE:
            return None
        if cursor_x + width > PAGE:
            cursor_x = 0
            cursor_y += row_height
            row_height = 0
        if cursor_y + height > PAGE and not start_page():
            return None
        pages[page_index][
            cursor_y : cursor_y + height,
            cursor_x : cursor_x + width,
            ...,
        ] = texture
        payload = preview_payloads[tile]
        slots.append(
            _WgpuPreviewAtlasSlot(
                tile_number=int(tile),
                payload_identity=tile_ack_identity(payload),
                binding_identity=_wgpu_preview_binding_identity(payload),
                page_index=int(page_index),
                source_origin_xy=(float(cursor_x), float(cursor_y)),
                source_size_xy=(float(width), float(height)),
            )
        )
        cursor_x += width
        row_height = max(row_height, height)

    plane_identities = tuple(("wgpu-preview-atlas", token, index) for index in range(len(pages)))
    page_keys = tuple(
        plane_chunk_key(
            plane_identity,
            "preview-atlas",
            0,
            0,
            0,
            dtype=_WGPU_REP_DTYPES[representation],
            representation=representation,
            plane_shape=(PAGE, PAGE),
            reducer=lod_reducer,
        )
        for plane_identity in plane_identities
    )
    return _WgpuPreviewAtlas(
        token=token,
        representation=str(representation),
        lod_reducer=str(lod_reducer),
        pages=tuple(pages),
        page_keys=page_keys,
        plane_identities=plane_identities,
        slots=tuple(slots),
    )


def _wgpu_payload_plane_identity(payload) -> object:
    """LOD-invariant physical plane identity for executor residency.

    Live payload source ids append a tagged ``lod`` suffix.  Removing only
    that suffix keeps document/operation/source/representation ownership
    intact while allowing every level to populate one executor plane.  Test
    and legacy payloads without the tag remain opaque identities.
    """

    source_id = getattr(payload, "source_id", None)
    if isinstance(source_id, tuple) and len(source_id) >= 4 and source_id[-4] == "lod":
        source_id = source_id[:-4]
    if source_id is None:
        identity = tile_ack_identity(payload)
        source_id = getattr(identity, "semantic_key", identity)
    return ("wgpu-content-plane", source_id)


def _wgpu_payload_local_plane_identity(payload) -> object:
    """Identity for texels whose (0, 0) is local to one source window.

    ``payload.source_id`` deliberately stays invariant across displayed-axis
    crops so canonical source pages can be rebound without transfer.  A
    crop-local fallback cannot use that identity: shifting the source window
    changes which source sample lives at local texel (0, 0).  Keep the
    window-invariant identity as the family prefix, but add the physical
    source rectangle that defines this local coordinate system.
    """

    identity = _wgpu_payload_plane_identity(payload)
    anchor = getattr(payload, "source_anchor", None)
    source_rect = tuple(int(value) for value in tuple(getattr(anchor, "source_rect", ()) or ()))
    plane_shape = tuple(int(value) for value in tuple(getattr(anchor, "plane_shape", ()) or ()))
    if len(source_rect) != 4:
        return identity
    return (
        *identity,
        "local-source-window",
        source_rect,
        plane_shape,
    )


def _wgpu_payload_lod_reducer(payload, *, representation: str, mapping_mode: str) -> str:
    """Canonical derived-value family for one live executor plane."""

    backing = getattr(payload, "page_backing", None)
    reducers = {str(plan.reducer) for plan in tuple(getattr(backing, "requested_plans", ()) or ())}
    if len(reducers) > 1:
        raise ValueError(f"wgpu payload mixes LOD reducer families: {tuple(sorted(reducers))}")
    if reducers:
        reducer = next(iter(reducers))
        if reducer != "native":
            return reducer
    # Native pages are shared input.  The bound plane still names the family
    # its future parents must belong to so phase-vector and component-mean
    # pages can never collide in one flat LOD span.
    if representation == COMPLEX_RG32F and str(mapping_mode) == "phase":
        return REDUCER_PHASE_VECTOR
    return REDUCER_MEAN


def _wgpu_payload_page_keys(
    payload,
    *,
    representation: str,
    mapping_mode: str,
    resident_keys=(),
) -> tuple[object, ...]:
    """Canonical executor keys for one payload without packing its pixels."""

    texture = np.asarray(
        payload.texture_data if payload.texture_data is not None else payload.image
    )
    return _wgpu_payload_binding(
        payload,
        texture,
        representation=representation,
        mapping_mode=mapping_mode,
        resident_keys=resident_keys,
    ).page_keys


def _wgpu_payload_binding(
    payload,
    texture: np.ndarray,
    *,
    representation: str,
    mapping_mode: str,
    resident_keys=(),
) -> _WgpuPayloadBinding:
    """Choose an honest crop-local or reusable source-plane binding."""

    from arrayscope.gpu.wgpu_executor import PAGE, plane_chunk_key

    geometry_error = None
    try:
        lod_level, local_source_shape = _wgpu_payload_lod_geometry(payload, texture)
    except ValueError as exc:
        anchor = getattr(payload, "source_anchor", None)
        lod = getattr(payload, "lod", None)
        if anchor is None or lod is None:
            raise
        # A globally aligned reduced window can contain one extra edge bin
        # compared with ceil(window_extent/factor). Its PageBackedPresentation
        # already validated that geometry. Prefer resident canonical pages
        # below; a cold single-page window can still preserve the global-bin
        # offset through an explicitly padded local plane.
        geometry_error = exc
        lod_level = int(getattr(lod, "level", 0) or 0)
        local_source_shape = tuple(int(value) for value in lod.source_shape)
    lod_reducer = _wgpu_payload_lod_reducer(
        payload,
        representation=representation,
        mapping_mode=mapping_mode,
    )
    local_identity = _wgpu_payload_local_plane_identity(payload)
    local_grid_h = -(-int(texture.shape[0]) // PAGE)
    local_grid_w = -(-int(texture.shape[1]) // PAGE)

    def local_binding() -> _WgpuPayloadBinding:
        if geometry_error is not None:
            raise geometry_error
        chunks = tuple(
            (chunk_y, chunk_x) for chunk_y in range(local_grid_h) for chunk_x in range(local_grid_w)
        )
        return _WgpuPayloadBinding(
            plane_identity=local_identity,
            plane_shape=local_source_shape,
            source_origin_xy=(0.0, 0.0),
            page_keys=tuple(
                plane_chunk_key(
                    local_identity,
                    "live",
                    lod_level,
                    chunk_x,
                    chunk_y,
                    dtype=_WGPU_REP_DTYPES[representation],
                    representation=representation,
                    plane_shape=local_source_shape,
                    reducer=lod_reducer,
                )
                for chunk_y, chunk_x in chunks
            ),
            upload_chunks=chunks,
            source_anchored=False,
            lod_level=lod_level,
        )

    def page_backed_local_binding() -> _WgpuPayloadBinding:
        """Bind cold source-grid pages without pretending their bins are local.

        The executor's plane ladder is uniformly power-of-two reduced.  A
        globally aligned window fits that contract when the local plane starts
        at the first global bin boundary and the requested window is expressed
        as an origin inside the padded plane.  This preserves clipped edge-bin
        geometry and keeps the page's pixels uploadable when no canonical
        native plane is resident yet.
        """

        backing = getattr(payload, "page_backing", None)
        plans = tuple(getattr(backing, "requested_plans", ()) or ())
        pages = tuple(getattr(backing, "materialized_pages", ()) or ())
        if not plans or not pages:
            raise geometry_error
        if any(
            tuple(int(value) for value in plan.reduction_yx) != (int(lod_level), int(lod_level))
            for plan in plans
        ):
            raise geometry_error
        stored_y0 = min(int(plan.stored_rect_yx[0]) for plan in plans)
        stored_y1 = max(int(plan.stored_rect_yx[1]) for plan in plans)
        stored_x0 = min(int(plan.stored_rect_yx[2]) for plan in plans)
        stored_x1 = max(int(plan.stored_rect_yx[3]) for plan in plans)
        stored_shape = (stored_y1 - stored_y0, stored_x1 - stored_x0)
        if stored_shape != tuple(int(value) for value in texture.shape[:2]):
            raise geometry_error
        coverage = tuple(int(value) for value in backing.source_coverage_yx)
        factor = 1 << int(lod_level)
        aligned_y0 = stored_y0 * factor
        aligned_x0 = stored_x0 * factor
        padded_source_shape = (stored_shape[0] * factor, stored_shape[1] * factor)
        source_origin = (coverage[2] - aligned_x0, coverage[0] - aligned_y0)
        if (
            source_origin[0] < 0
            or source_origin[1] < 0
            or source_origin[0] + local_source_shape[1] > padded_source_shape[1]
            or source_origin[1] + local_source_shape[0] > padded_source_shape[0]
        ):
            raise geometry_error
        chunks = tuple(
            (chunk_y, chunk_x) for chunk_y in range(local_grid_h) for chunk_x in range(local_grid_w)
        )
        return _WgpuPayloadBinding(
            plane_identity=local_identity,
            plane_shape=padded_source_shape,
            source_origin_xy=(float(source_origin[0]), float(source_origin[1])),
            page_keys=tuple(
                plane_chunk_key(
                    local_identity,
                    "live",
                    lod_level,
                    chunk_x,
                    chunk_y,
                    dtype=_WGPU_REP_DTYPES[representation],
                    representation=representation,
                    plane_shape=padded_source_shape,
                    reducer=lod_reducer,
                )
                for chunk_y, chunk_x in chunks
            ),
            upload_chunks=chunks,
            source_anchored=False,
            lod_level=lod_level,
        )

    anchor = getattr(payload, "source_anchor", None)
    plane_shape = tuple(getattr(anchor, "plane_shape", ()) or ())
    source_rect = tuple(getattr(anchor, "source_rect", ()) or ())
    if len(plane_shape) != 2 or len(source_rect) != 4:
        return local_binding()
    plane_h, plane_w = (int(value) for value in plane_shape)
    y0, y1, x0, x1 = (int(value) for value in source_rect)
    if not (0 <= y0 < y1 <= plane_h and 0 <= x0 < x1 <= plane_w):
        return local_binding()
    if (y1 - y0, x1 - x0) != tuple(int(value) for value in local_source_shape):
        return local_binding()

    source_page = PAGE << int(lod_level)
    chunk_y0 = y0 // source_page
    chunk_x0 = x0 // source_page
    chunk_y1 = -(-y1 // source_page)
    chunk_x1 = -(-x1 // source_page)
    global_identity = ("wgpu-source-plane", getattr(anchor, "content_key", None))
    global_chunks = tuple(
        (chunk_y, chunk_x)
        for chunk_y in range(chunk_y0, chunk_y1)
        for chunk_x in range(chunk_x0, chunk_x1)
    )
    global_keys = tuple(
        plane_chunk_key(
            global_identity,
            "live",
            lod_level,
            chunk_x,
            chunk_y,
            dtype=_WGPU_REP_DTYPES[representation],
            representation=representation,
            plane_shape=plane_shape,
            reducer=lod_reducer,
        )
        for chunk_y, chunk_x in global_chunks
    )
    resident = set(resident_keys)
    fully_resident = all(key in resident for key in global_keys)
    if not fully_resident and int(lod_level) > 0:
        # A reduced payload is a presentation-quality floor, not a mandate to
        # sample that physical rung.  If the exact native source window is
        # already resident, both preview and exact reduced successors can bind
        # those better pages directly.  This is the fast cropped-axis path:
        # only source origin/extent changes, with no crop upload and no
        # reduced edge-bin geometry to reinterpret.
        native_chunk_y0 = y0 // PAGE
        native_chunk_x0 = x0 // PAGE
        native_chunk_y1 = -(-y1 // PAGE)
        native_chunk_x1 = -(-x1 // PAGE)
        native_chunks = tuple(
            (chunk_y, chunk_x)
            for chunk_y in range(native_chunk_y0, native_chunk_y1)
            for chunk_x in range(native_chunk_x0, native_chunk_x1)
        )
        native_keys = tuple(
            plane_chunk_key(
                global_identity,
                "live",
                0,
                chunk_x,
                chunk_y,
                dtype=_WGPU_REP_DTYPES[representation],
                representation=representation,
                plane_shape=plane_shape,
                reducer=lod_reducer,
            )
            for chunk_y, chunk_x in native_chunks
        )
        if all(key in resident for key in native_keys):
            return _WgpuPayloadBinding(
                plane_identity=global_identity,
                plane_shape=(plane_h, plane_w),
                source_origin_xy=(float(x0), float(y0)),
                page_keys=native_keys,
                upload_chunks=tuple(
                    (chunk_y - native_chunk_y0, chunk_x - native_chunk_x0)
                    for chunk_y, chunk_x in native_chunks
                ),
                source_anchored=True,
                lod_level=0,
            )
    if geometry_error is not None and not fully_resident:
        try:
            return page_backed_local_binding()
        except ValueError:
            resident_levels = tuple(
                sorted(
                    {
                        int(key.lod.level)
                        for key in resident
                        if key.document_generation == global_identity
                        and key.operation_key == "live"
                        and key.representation == representation
                    }
                )
            )
            raise ValueError(
                f"{geometry_error}; canonical source-plane resident levels="
                f"{resident_levels or 'none'}"
            ) from geometry_error
    supplies_complete_pages = (
        str(getattr(payload, "quality", "exact") or "exact") == "exact"
        and y0 % source_page == 0
        and x0 % source_page == 0
        and (y1 == plane_h or y1 % source_page == 0)
        and (x1 == plane_w or x1 % source_page == 0)
        and len(global_chunks) == local_grid_h * local_grid_w
    )
    if not fully_resident and not supplies_complete_pages:
        return local_binding()
    return _WgpuPayloadBinding(
        plane_identity=global_identity,
        plane_shape=(plane_h, plane_w),
        source_origin_xy=(float(x0), float(y0)),
        page_keys=global_keys,
        upload_chunks=tuple(
            (chunk_y - chunk_y0, chunk_x - chunk_x0) for chunk_y, chunk_x in global_chunks
        ),
        source_anchored=True,
        lod_level=lod_level,
    )


def _wgpu_plan_lod_page_generation(
    destination,
    *,
    plane_shape: tuple[int, int],
    available: set,
    commands: list,
) -> bool:
    """Append a topological resident-mean chain, or leave the plan unchanged."""

    from arrayscope.gpu.wgpu_executor import PAGE, plane_chunk_key

    if destination in available:
        return True
    level = int(destination.lod.level)
    if (
        level <= 0
        or destination.lod.reducer != REDUCER_MEAN
        or destination.representation not in (SCALAR_R32F, COMPLEX_RG32F)
    ):
        return False

    command_start = len(commands)
    available_before = set(available)
    child_level = level - 1
    child_extent = PAGE << child_level
    y0, x0 = (int(value) for value in destination.chunk_origin)
    h, w = (int(value) for value in destination.chunk_shape)
    child_keys = [
        plane_chunk_key(
            destination.document_generation,
            destination.operation_key,
            child_level,
            child_x,
            child_y,
            dtype=destination.dtype,
            representation=destination.representation,
            plane_shape=plane_shape,
            reducer=REDUCER_MEAN,
        )
        for child_y in range(y0 // child_extent, -(-(y0 + h) // child_extent))
        for child_x in range(x0 // child_extent, -(-(x0 + w) // child_extent))
    ]
    if not 1 <= len(child_keys) <= 4:
        raise ValueError("wgpu canonical parent must cover one to four immediate child pages")
    for child in child_keys:
        if child in available:
            continue
        if not _wgpu_plan_lod_page_generation(
            child,
            plane_shape=plane_shape,
            available=available,
            commands=commands,
        ):
            del commands[command_start:]
            available.clear()
            available.update(available_before)
            return False
    commands.append(GenerateLodPages(tuple(child_keys), destination))
    available.add(destination)
    return True


def _wgpu_preparable_representation(payload, rgb_already_windowed: bool):
    """The representation a commit would pack ``payload`` into, or None.

    Mirrors the derivation in ``_wgpu_commit_plan``. Returning None means "do
    not prepare this ahead" -- never "this payload is invalid"; the commit's
    validation is unchanged and stays the only place that refuses.
    """

    try:
        kind = _wgpu_payload_kind(payload)
    except Exception:
        return None
    representation = _WGPU_REP_BY_KIND.get(kind)
    if representation == RGB8 and not rgb_already_windowed:
        return RGB_WINDOWED_RGBA32F
    return representation


def _wgpu_multi_page_assembly(payload) -> bool:
    """Whether packing this payload has to assemble more than one page.

    Mirrors the early returns in ``_wgpu_assembled_page_texture``: a single
    plan, or a plan set the page store does not exactly cover, hands the
    fallback texture straight back without touching it.
    """

    backing = getattr(payload, "page_backing", None)
    plans = tuple(getattr(backing, "requested_plans", ()) or ())
    pages = tuple(getattr(backing, "materialized_pages", ()) or ())
    if len(plans) <= 1 or len(pages) != len(plans):
        return False
    return {page.key for page in pages} == {plan.key for plan in plans}


def wgpu_pack_saves_work(payload, representation) -> bool:
    """Whether packing ``payload`` would allocate anything the commit lacks.

    ``pack_texture_data`` is a no-op for a plane that already has the texel
    layout its representation wants: ``np.ascontiguousarray`` on a contiguous
    array of the right dtype hands the same object back. Preparing one of those
    ahead buys nothing and is not free — it takes a scheduler slot and holds a
    worker thread for as long as it runs, which is worker time the round's
    pixel-producing tasks could have had.

    So the hand-off is offered only where there is real work to move: a
    multi-page assembly, a dtype or component conversion, or a channel
    repack. This is a *value* judgement about preparation, never a statement
    about what the commit can upload — the commit packs whatever it is given.
    """

    if _wgpu_multi_page_assembly(payload):
        return True
    if representation == RGB_WINDOWED_RGBA32F:
        # Always builds a fresh RGBA plane and derives or copies the scalar.
        return True
    texture = payload.texture_data if payload.texture_data is not None else payload.image
    texture = np.asarray(texture)
    if representation == SCALAR_R32F:
        return texture.dtype != np.float32 or not texture.flags["C_CONTIGUOUS"]
    if representation == COMPLEX_RG32F:
        if np.iscomplexobj(texture):
            return True
        return (
            texture.ndim < 3
            or texture.shape[-1] != 2
            or texture.dtype != np.float32
            or not texture.flags["C_CONTIGUOUS"]
        )
    if representation == RGB8:
        return (
            texture.ndim != 3
            or texture.shape[-1] != 3
            or texture.dtype != np.uint8
            or not texture.flags["C_CONTIGUOUS"]
        )
    return False


def _wgpu_pack_preparation(task, payload, representation):
    """Build the worker callable that packs one payload's upload plane.

    The ``with`` closes the task's in-flight window even when packing raises.
    """

    def prepare() -> None:
        with task:
            task.publish(wgpu_packed_payload_texture(payload, representation))

    return prepare


def wgpu_packed_payload_texture(payload, representation) -> np.ndarray:
    """Pack one payload into its upload plane, off any thread.

    Pure over an immutable payload: it assembles the payload's own pages and
    converts them to the representation's texel layout, allocating its own
    output. Nothing here touches the view, the executor, or Qt, which is what
    lets a worker run it while the GUI thread is submitting something else.
    """

    texture = payload.texture_data if payload.texture_data is not None else payload.image
    texture = _wgpu_assembled_page_texture(payload, texture)
    if representation == RGB_WINDOWED_RGBA32F:
        base = pack_texture_data(texture, TexturePlaneKind.RGB8)
        scalar = getattr(payload, "histogram_data", None)
        if scalar is None:
            rgb = np.asarray(texture, dtype=np.float32)[..., :3]
            scalar = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        scalar = np.asarray(scalar, dtype=np.float32)
        if scalar.shape != base.shape[:2]:
            raise ValueError(
                "wgpu windowable RGB scalar plane must match the RGB tile; "
                f"got {scalar.shape} versus {base.shape[:2]}"
            )
        packed = np.empty((*base.shape[:2], 4), np.float32)
        packed[..., :3] = base.astype(np.float32) / 255.0
        packed[..., 3] = scalar
        return np.ascontiguousarray(packed)
    kind = {
        SCALAR_R32F: TexturePlaneKind.SCALAR_R32F,
        COMPLEX_RG32F: TexturePlaneKind.COMPLEX_RG32F,
        RGB8: TexturePlaneKind.RGB8,
    }[representation]
    return pack_texture_data(texture, kind)


def _wgpu_assembled_page_texture(payload, fallback) -> np.ndarray:
    """Assemble exact same-rung logical pages for a local WGPU upload.

    ``DisplayTilePayload.texture_data`` historically names the first page
    because most reduced crops fit one 256x256 stored page. A wider cold crop
    legitimately spans several pages. Packing those already-materialized
    values is the bounded upload fallback; it does not create new numeric
    work or pretend the partial crop is canonical source residency.
    """

    texture = np.asarray(fallback)
    backing = getattr(payload, "page_backing", None)
    plans = tuple(getattr(backing, "requested_plans", ()) or ())
    pages = tuple(getattr(backing, "materialized_pages", ()) or ())
    if len(plans) <= 1 or len(pages) != len(plans):
        return texture
    pages_by_key = {page.key: page for page in pages}
    if set(pages_by_key) != {plan.key for plan in plans}:
        # Ancestor-backed floors have different stored geometry and remain on
        # their existing resolution path.
        return texture
    stored_y0 = min(int(plan.stored_rect_yx[0]) for plan in plans)
    stored_y1 = max(int(plan.stored_rect_yx[1]) for plan in plans)
    stored_x0 = min(int(plan.stored_rect_yx[2]) for plan in plans)
    stored_x1 = max(int(plan.stored_rect_yx[3]) for plan in plans)
    shape = (stored_y1 - stored_y0, stored_x1 - stored_x0, *texture.shape[2:])
    assembled = np.empty(shape, dtype=texture.dtype)
    filled = np.zeros(shape[:2], dtype=np.bool_)
    for plan in plans:
        page = pages_by_key[plan.key]
        values = np.asarray(page.values)
        y0, y1, x0, x1 = (int(value) for value in plan.stored_rect_yx)
        target = (slice(y0 - stored_y0, y1 - stored_y0), slice(x0 - stored_x0, x1 - stored_x0))
        if tuple(values.shape) != tuple(assembled[target].shape):
            raise ValueError("wgpu page values do not match their stored upload rectangle")
        assembled[target] = values
        filled[target] = True
    if not bool(np.all(filled)):
        raise ValueError("wgpu page-backed upload has a hole in its stored rectangle")
    return np.ascontiguousarray(assembled)


def _wgpu_representation_byte_budgets(
    *,
    required_pages: dict[str, int],
    preferred_pages: dict[str, int],
    previous_pages: dict[str, int],
    budget_bytes: int,
) -> dict[str, int]:
    """Split one residency byte policy across representation pools."""

    budget_bytes = max(0, int(budget_bytes))
    if budget_bytes <= 0:
        return {}
    from arrayscope.gpu.wgpu_executor import PAGE

    requested: dict[str, int] = {}
    for representation in (
        SCALAR_R32F,
        COMPLEX_RG32F,
        RGB8,
        RGB_WINDOWED_RGBA32F,
    ):
        needed = max(0, int(required_pages.get(representation, 0)))
        preferred = max(0, int(preferred_pages.get(representation, 0)))
        previous = max(0, int(previous_pages.get(representation, 0)))
        working_set = max(preferred, 2 * needed + 8 if needed else 0)
        layers = max(previous, working_set)
        if layers:
            requested[representation] = (
                layers * PAGE * PAGE * _WGPU_POOL_TEXEL_BYTES[representation]
            )
    requested_total = sum(requested.values())
    if requested_total <= budget_bytes:
        return dict(requested)
    shares = {
        representation: max(1, budget_bytes * value // requested_total)
        for representation, value in requested.items()
    }
    # Integer division can leave a few bytes undistributed. Give them to the
    # largest unmet requests deterministically without exceeding the policy.
    remaining = budget_bytes - sum(shares.values())
    for representation in sorted(
        shares,
        key=lambda rep: (requested[rep] - shares[rep], rep),
        reverse=True,
    ):
        if remaining <= 0:
            break
        addition = min(remaining, requested[representation] - shares[representation])
        shares[representation] += addition
        remaining -= addition
    return shares


def _wgpu_pool_layer_budget(
    *,
    previous: int,
    needed: int,
    preferred: int = 0,
    max_layers: int,
    budget_bytes: int = 0,
    bytes_per_layer: int = 0,
) -> int:
    """Size retention from the memory policy, clamped by device limits.

    ``needed`` is correctness and may exceed the retention policy for one
    visible transaction.  ``preferred`` and two-rung headroom are optional
    retention.  A zero byte budget preserves the legacy call contract used
    by low-level tests and explicit executor probes.
    """

    previous = max(0, int(previous))
    needed = max(0, int(needed))
    preferred = max(0, int(preferred))
    max_layers = max(1, int(max_layers))
    budget_bytes = max(0, int(budget_bytes))
    bytes_per_layer = max(0, int(bytes_per_layer))
    if needed > max_layers:
        raise RuntimeError(
            "wgpu active plane pages exceed the device texture-array limit: "
            f"needed={needed}, max_layers={max_layers}"
        )
    working_set = max(preferred, 2 * needed + 8 if needed else 0)
    if budget_bytes and bytes_per_layer and (needed or preferred or previous):
        policy_layers = max(1, budget_bytes // bytes_per_layer)
        # Policy is the retention ceiling, not an admission refusal.  The
        # active transaction still wins when it alone is larger. Previous
        # headroom is a retention request, not a reason to exceed a changed
        # device policy.
        desired = max(needed, min(max(previous, working_set), policy_layers))
    else:
        desired = max(previous, working_set)
    return min(desired, max_layers)


def _wgpu_payload_lod_geometry(payload, texture) -> tuple[int, tuple[int, int]]:
    """Validate the executor's isotropic LOD contract for one payload."""

    texture_shape = tuple(int(value) for value in np.shape(texture)[:2])
    lod = getattr(payload, "lod", None)
    if lod is None:
        rung_level = 0
        factor = 1
        source_shape = tuple(
            int(value) for value in (getattr(payload, "source_shape", None) or texture_shape)[:2]
        )
        declared_texture_shape = texture_shape
    else:
        rung_level = int(getattr(lod, "level", 0) or 0)
        try:
            factor = int(payload.actual_lod_factor)
        except (AttributeError, TypeError, ValueError) as exc:
            raise NotImplementedError(
                "wgpu executor requires one isotropic actual payload reduction factor"
            ) from exc
        gutter = int(getattr(lod, "gutter", 0) or 0)
        # Live ingest-reduced payloads may carry the reduced evaluated plane
        # in DisplayTilePayload.source_shape. LodInfo.source_shape is the
        # canonical native geometry the executor's page ladder addresses.
        source_shape = tuple(int(value) for value in lod.source_shape)
        declared_texture_shape = tuple(int(value) for value in lod.texture_shape)
        if gutter:
            raise NotImplementedError(
                f"wgpu backend does not yet support LOD gutters; got {gutter}"
            )
    if factor < 1 or factor & (factor - 1):
        raise NotImplementedError(
            "wgpu executor requires a power-of-two actual reduction factor; "
            f"got factor {factor} for rung label {rung_level}"
        )
    level = factor.bit_length() - 1
    expected_texture_shape = tuple(-(-extent // factor) for extent in source_shape)
    requested_shape_mismatch = (
        getattr(payload, "page_backing", None) is None and declared_texture_shape != texture_shape
    )
    if requested_shape_mismatch or texture_shape != expected_texture_shape:
        raise ValueError(
            "wgpu payload texture geometry does not match its native LOD ladder: "
            f"source={source_shape}, rung_label={rung_level}, factor={factor}, "
            f"executor_level={level}, declared={declared_texture_shape}, "
            f"actual={texture_shape}, expected={expected_texture_shape}"
        )
    return level, source_shape


def _wgpu_payload_declared_lod_geometry(payload, texture) -> tuple[int, tuple[int, int]]:
    """Capacity geometry for payloads with checked global reduction bins.

    ``PageBackedPresentation`` may validly contain one extra leading/trailing
    stored bin when its source window is not factor-aligned.  Binding still
    decides whether those values can be used honestly; capacity planning only
    needs the declared rung and native window extent and must not reject the
    payload before that decision.
    """

    try:
        return _wgpu_payload_lod_geometry(payload, texture)
    except ValueError:
        lod = getattr(payload, "lod", None)
        if (
            lod is None
            or getattr(payload, "page_backing", None) is None
            or getattr(payload, "source_anchor", None) is None
            or tuple(int(value) for value in np.shape(texture)[:2])
            != tuple(int(value) for value in lod.texture_shape)
        ):
            raise
        factor = int(payload.actual_lod_factor)
        if factor < 1 or factor & (factor - 1):
            raise
        return factor.bit_length() - 1, tuple(int(value) for value in lod.source_shape)


def _wgpu_ladder_page_count(source_shape, *, max_lod: int) -> int:
    height, width = (int(value) for value in source_shape)
    from arrayscope.gpu.wgpu_executor import PAGE

    return sum(
        (-(-height // (PAGE << level))) * (-(-width // (PAGE << level)))
        for level in range(int(max_lod) + 1)
    )


def _wgpu_native_prefetch_page_count(
    payload,
    *,
    representation: str,
    selected_lod: int,
) -> int:
    """Native source pages carried alongside a presentation of part of them.

    Two presentations warm the canonical plane instead of uploading their own
    texture: a REDUCED one (the payload owns finer pixels than it presents) and
    a CROPPED one at any level (the payload owns a wider plane than it
    presents). Both replace their texture upload with the plane's pages, so
    both must be counted as the plane — under-counting turns the pool's
    retention preference into a correctness ceiling.
    """

    if representation not in {SCALAR_R32F, COMPLEX_RG32F}:
        return 0
    source = getattr(payload, "native_residency_data", None)
    if source is None:
        source = getattr(payload, "semantic_data", None)
    anchor = getattr(payload, "source_anchor", None)
    plane_shape = tuple(int(value) for value in (getattr(anchor, "plane_shape", ()) or ()))
    if (
        source is None
        or len(plane_shape) != 2
        or tuple(int(value) for value in np.shape(source)[:2]) != plane_shape
    ):
        return 0
    if int(selected_lod) <= 0 and _wgpu_anchor_covers_plane(anchor):
        # An exact whole-plane payload IS its canonical plane; its ordinary
        # source-anchored binding uploads those pages, and the generic texture
        # count already describes them.
        return 0
    from arrayscope.gpu.wgpu_executor import PAGE

    return -(-int(plane_shape[0]) // PAGE) * -(-int(plane_shape[1]) // PAGE)


def _wgpu_anchor_covers_plane(anchor) -> bool:
    """Does this anchor's window span its whole canonical source plane?"""

    plane_shape = tuple(int(value) for value in (getattr(anchor, "plane_shape", ()) or ()))
    source_rect = tuple(int(value) for value in (getattr(anchor, "source_rect", ()) or ()))
    if len(plane_shape) != 2 or len(source_rect) != 4:
        return False
    return source_rect == (0, plane_shape[0], 0, plane_shape[1])


def _wgpu_payload_capacity_page_count(
    payload,
    *,
    texture_shape: tuple[int, int],
    representation: str,
    selected_lod: int,
) -> int:
    """Physical pages this payload path will submit for admission.

    Reduced payloads can carry an exact reusable source plane.  Both the warm
    path and the visible commit replace the reduced texture with those native
    pages, so capacity must count the replacement rather than the small
    preview texture.  Under-counting here turns a retention preference into a
    correctness ceiling once the already-bound pages fill the pool.
    """

    native_pages = _wgpu_native_prefetch_page_count(
        payload,
        representation=representation,
        selected_lod=selected_lod,
    )
    if native_pages:
        return int(native_pages)
    from arrayscope.gpu.wgpu_executor import PAGE

    height, width = (int(value) for value in texture_shape)
    return -(-height // PAGE) * -(-width // PAGE)


def _wgpu_pre_reservation_page_count(
    capacity_pages_by_tile: dict[int, int],
    *,
    planned_count: int,
    max_layers: int,
) -> int:
    """Bound a progressive frame's one-time capacity forecast.

    The submitted payloads are correctness-critical active admission. The
    remaining planned tiles are only a retention forecast used to avoid
    repeated texture-array growth, so that forecast may be capped at the
    device's texture-array-layer limit. The active batch itself must still
    fail loudly if it cannot fit.
    """

    capacities = tuple(max(0, int(value)) for value in capacity_pages_by_tile.values())
    if not capacities:
        return 0
    max_layers = max(1, int(max_layers))
    active_pages = sum(capacities)
    if active_pages > max_layers:
        raise RuntimeError(
            "wgpu active plane pages exceed the device texture-array limit: "
            f"needed={active_pages}, max_layers={max_layers}"
        )
    planned_count = max(len(capacities), int(planned_count))
    planned_pages = planned_count * max(capacities)
    return max(active_pages, min(planned_pages, max_layers))


def _wgpu_native_prefetch_page_keys(
    payload,
    *,
    representation: str,
    mapping_mode: str,
    selected_lod: int,
) -> tuple[object, ...]:
    """Canonical L0 keys that replace a redundant reduced warm payload."""

    if not _wgpu_native_prefetch_page_count(
        payload,
        representation=representation,
        selected_lod=selected_lod,
    ):
        return ()
    from arrayscope.gpu.wgpu_executor import PAGE, plane_chunk_key

    anchor = payload.source_anchor
    plane_shape = tuple(int(value) for value in anchor.plane_shape)
    plane_identity = ("wgpu-source-plane", anchor.content_key)
    reducer = _wgpu_payload_lod_reducer(
        payload,
        representation=representation,
        mapping_mode=mapping_mode,
    )
    return tuple(
        plane_chunk_key(
            plane_identity,
            "live",
            0,
            chunk_x,
            chunk_y,
            dtype=_WGPU_REP_DTYPES[representation],
            representation=representation,
            plane_shape=plane_shape,
            reducer=reducer,
        )
        for chunk_y in range(-(-plane_shape[0] // PAGE))
        for chunk_x in range(-(-plane_shape[1] // PAGE))
    )


def _wgpu_payload_kind(payload) -> TexturePlaneKind:
    """Payload texture representation (declared kind first, then inference)."""

    kind = getattr(payload, "texture_kind", None)
    if kind is not None:
        return (
            kind
            if isinstance(kind, TexturePlaneKind)
            else TexturePlaneKind(getattr(kind, "value", kind))
        )
    texture = np.asarray(
        payload.texture_data
        if getattr(payload, "texture_data", None) is not None
        else payload.image
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
