"""Image-view backend factory."""

from __future__ import annotations

import contextlib
import os
import platform
import threading

from arrayscope.app.settings_state import (
    ImageRenderingBackendChoice,
    TextureCodecChoice,
    WgpuPresentMethodChoice,
)
from arrayscope.display.backends.pyqtgraph import PyQtGraphSurface

_SOFTWARE_RENDERER_MARKERS = ("llvmpipe", "softpipe", "swrast", "software rasterizer")

# Cached AUTO probe result for this process: (resolved_choice, reason).
_auto_resolution_cache: tuple[ImageRenderingBackendChoice, str] | None = None

# Guards the one-shot startup warm-up so repeated file opens spawn a single
# background thread (the shared device is a process global anyway).
_warm_lock = threading.Lock()
_warm_started = False


def warm_image_backend_async(settings=None) -> None:
    """Start wgpu device + shader warm-up on a background thread, if applicable.

    Called at the start of a file open (before/while the loading window shows):
    when the effective image backend will be wgpu, the ~2 s of device init and
    ~165 ms of shader compilation run concurrently with the file read instead of
    stalling the first image behind it.  A no-op when the backend cannot be wgpu
    (explicit PyQtGraph/VisPy pin, non-Linux, offscreen Qt) so we never spend
    ~2 s building a device the app will not use.  Idempotent per process.
    """

    global _warm_started
    if not _should_warm_wgpu(settings):
        return
    with _warm_lock:
        if _warm_started:
            return
        _warm_started = True
    thread = threading.Thread(
        target=_warm_wgpu_backend,
        name="arrayscope-wgpu-warm",
        daemon=True,
    )
    thread.start()


def _should_warm_wgpu(settings) -> bool:
    """Whether the effective image backend is (likely) wgpu — cheap, no GPU."""

    choice = _image_backend_choice_value(settings)
    if choice == ImageRenderingBackendChoice.WGPU.value:
        return True  # explicit pin: warm regardless of platform heuristics
    if choice != ImageRenderingBackendChoice.AUTO.value:
        return False  # explicit PyQtGraph / VisPy pin
    # AUTO: mirror _probe_auto_backend_choice's wgpu preconditions so the warm
    # matches what create_image_view will resolve.  A device that ultimately
    # fails to build simply leaves the warm a no-op and the GL fallback runs
    # (on the GUI thread, as today).
    if platform.system() != "Linux":
        return False
    return os.environ.get("QT_QPA_PLATFORM", "") not in {"offscreen", "minimal"}


def cpu_display_backend_likely(settings=None) -> bool:
    """Whether the effective image backend uses the CPU display kernels.

    ``cpu_display_rgba`` / ``rgb_display_for_levels`` run only on the pyqtgraph
    backend; wgpu and vispy shade on the GPU.  Consulted at startup so their
    optional numba kernels are not compiled for a wgpu/vispy session.  A
    wgpu->pyqtgraph fallback (device build failure) is still covered by the lazy
    warm on the first CPU display call, so a False here never breaks anything.
    """

    choice = _image_backend_choice_value(settings)
    if choice == ImageRenderingBackendChoice.PYQTGRAPH.value:
        return True
    if choice in {
        ImageRenderingBackendChoice.WGPU.value,
        ImageRenderingBackendChoice.VISPY.value,
    }:
        return False
    # AUTO: pyqtgraph is the CPU fallback chosen when the wgpu preconditions fail.
    return not _should_warm_wgpu(settings)


def wgpu_texture_compression_likely(settings=None) -> bool:
    """Whether bulk BC prewarm is useful for the configured viewer session."""

    if not _should_warm_wgpu(settings):
        return False
    choice = getattr(settings, "texture_codec", None)
    if choice is None:
        with contextlib.suppress(Exception):
            from pyqtgraph.Qt import QtCore

            choice = QtCore.QSettings().value("texture_codec", TextureCodecChoice.OFF.value)
    return str(getattr(choice, "value", choice or TextureCodecChoice.OFF.value)) != str(
        TextureCodecChoice.OFF.value
    )


def _image_backend_choice_value(settings) -> str:
    """The configured image-backend choice value, from ``settings`` or QSettings."""

    choice = getattr(settings, "image_rendering_backend", None)
    if choice is not None:
        return getattr(choice, "value", choice)
    # No settings object at file-open time: read the persisted choice directly.
    with contextlib.suppress(Exception):
        from pyqtgraph.Qt import QtCore

        return str(
            QtCore.QSettings().value(
                "image_rendering_backend", ImageRenderingBackendChoice.AUTO.value
            )
        )
    return ImageRenderingBackendChoice.AUTO.value


def _warm_wgpu_backend() -> None:
    with contextlib.suppress(Exception):
        from arrayscope.display.wgpu_imageview2d import warm_wgpu_backend

        warm_wgpu_backend()


def create_image_view(settings=None, *, notify=None):
    """Create the selected image view implementation.

    ``AUTO`` resolves to **wgpu** (the promotion-candidate backend, 2026-07-22)
    whenever a real GPU device can be created on Linux with a display; it falls
    back to VisPy where a hardware GL context exists but wgpu cannot init, and
    to PyQtGraph for software GL, headless/offscreen, platforms without
    reference traces, or any probe failure. Users can always pin a backend.
    If wgpu init fails at construction, the factory degrades to PyQtGraph.
    """

    choice = getattr(settings, "image_rendering_backend", ImageRenderingBackendChoice.AUTO)
    choice_value = getattr(choice, "value", choice)
    if choice_value == ImageRenderingBackendChoice.AUTO.value:
        resolved, reason = resolve_auto_backend_choice()
        choice_value = resolved.value
        if callable(notify):
            notify(f"Image rendering backend: {choice_value} | {reason}")
    if choice_value == ImageRenderingBackendChoice.WGPU.value:
        present_method = getattr(settings, "wgpu_present_method", WgpuPresentMethodChoice.BITMAP)
        present_method_value = getattr(present_method, "value", present_method)
        texture_codec = getattr(settings, "texture_codec", TextureCodecChoice.OFF)
        texture_codec_value = getattr(texture_codec, "value", texture_codec)
        pixel_grid = bool(getattr(settings, "wgpu_pixel_grid", False))
        clip_indicator = bool(getattr(settings, "wgpu_clip_indicator", False))
        minification_filter = bool(getattr(settings, "wgpu_minification_filter", False))
        try:
            from arrayscope.display.backends.wgpu import WgpuSurface

            view = WgpuSurface(
                present_method=present_method_value,
                texture_codec=texture_codec_value,
                pixel_grid=pixel_grid,
                clip_indicator=clip_indicator,
                minification_filter=minification_filter,
            )
            view._notify_status = notify
            # Only an EXPLICIT screen pin warrants a warning; AUTO resolving
            # to bitmap is the resolution rule working as designed.
            if (
                callable(notify)
                and present_method_value == WgpuPresentMethodChoice.SCREEN.value
                and view.wgpuPresentMethod() != present_method_value
            ):
                notify(
                    "wgpu screen presentation unavailable; using bitmap "
                    f"({view.wgpuPresentMethodFallbackReason()})"
                )
            return view
        except Exception as exc:
            if callable(notify):
                notify(f"wgpu renderer unavailable; using PyQtGraph ({exc})")
        view = PyQtGraphSurface()
        view._notify_status = notify
        return view
    if choice_value == ImageRenderingBackendChoice.VISPY.value:
        try:
            from arrayscope.display.backends.vispy import VisPySurface

            view = VisPySurface()
            view._notify_status = notify
            return view
        except Exception as exc:
            if callable(notify):
                notify(f"VisPy renderer unavailable; using PyQtGraph ({exc})")
    view = PyQtGraphSurface()
    view._notify_status = notify
    return view


def resolve_auto_backend_choice() -> tuple[ImageRenderingBackendChoice, str]:
    """Resolve ``AUTO`` to a concrete backend choice for this process.

    Decision rule (ADR 0047, updated 2026-07-22 — wgpu promotion): on Linux
    with a real display, prefer **wgpu** whenever a GPU device can be created;
    fall back to VisPy where a hardware GL context exists but wgpu cannot init,
    and to PyQtGraph for software GL, headless/offscreen, non-Linux, or any
    probe failure. Users can always pin a specific backend.
    """

    global _auto_resolution_cache
    if _auto_resolution_cache is not None:
        return _auto_resolution_cache
    _auto_resolution_cache = _probe_auto_backend_choice()
    return _auto_resolution_cache


def _probe_auto_backend_choice() -> tuple[ImageRenderingBackendChoice, str]:
    if platform.system() != "Linux":
        return (
            ImageRenderingBackendChoice.PYQTGRAPH,
            f"no reference performance traces for {platform.system()}",
        )
    if os.environ.get("QT_QPA_PLATFORM", "") in {"offscreen", "minimal"}:
        return (ImageRenderingBackendChoice.PYQTGRAPH, "offscreen Qt platform")
    # Prefer wgpu (the promotion-candidate backend) whenever a real GPU device
    # can be created. The probe pins Vulkan (no EGL re-init that would SIGABRT a
    # GL context) and caches the shared device the WgpuImageView2D reuses, so it
    # is not wasted work. wgpu first also means we never create a GL context on
    # the wgpu path.
    wgpu_device = _probe_wgpu_device()
    if wgpu_device is not None:
        return (ImageRenderingBackendChoice.WGPU, f"wgpu device [{wgpu_device}]")
    renderer = _probe_hardware_gl_renderer()
    if renderer is None:
        return (ImageRenderingBackendChoice.PYQTGRAPH, "no usable wgpu or OpenGL device")
    lowered = renderer.lower()
    if any(marker in lowered for marker in _SOFTWARE_RENDERER_MARKERS):
        return (
            ImageRenderingBackendChoice.PYQTGRAPH,
            f"SW OpenGL [{renderer}]",
        )
    return (
        ImageRenderingBackendChoice.VISPY,
        f"HW OpenGL, wgpu unavailable [{renderer}]",
    )


def _probe_wgpu_device() -> str | None:
    """Return a short device label if a wgpu device can be created, else None.

    Reuses the shared Vulkan-pinned device the WgpuImageView2D would use;
    importing the module is side-effect-free (it does not force ``xcb`` and
    imports rendercanvas only lazily via ``import_qrenderwidget``).
    """

    # Import outside the try: a failure to import our own module is a real
    # error the import-health guard must see, not a "no GPU" signal. Only the
    # wgpu runtime device creation is allowed to fail (a machine with no
    # Vulkan adapter).
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device

    try:
        device = _shared_wgpu_device()
    except Exception:
        return None
    if device is None:
        return None
    adapter = getattr(device, "adapter", None)
    info = {}
    if adapter is not None:
        get_info = getattr(adapter, "request_adapter_info", None)
        if callable(get_info):
            with contextlib.suppress(Exception):
                info = get_info() or {}
    return str(info.get("device") or info.get("description") or "gpu")


def _probe_hardware_gl_renderer() -> str | None:
    """Create a short-lived GL context and return its renderer string."""

    from arrayscope.display.backends.vispy.tiles import query_gpu_device_limits

    try:
        from vispy import app as vispy_app
        from vispy import gloo

        canvas = vispy_app.Canvas(show=False, size=(4, 4))
        try:
            canvas.set_current()
            limits = query_gpu_device_limits(gloo)
        finally:
            canvas.close()
        if str(getattr(limits, "source", "")) != "opengl":
            return None
        renderer = str(getattr(limits, "renderer", "") or "")
        return renderer or None
    except Exception:
        return None
