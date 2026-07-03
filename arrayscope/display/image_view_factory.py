"""Image-view backend factory."""

from __future__ import annotations

import os
import platform

from arrayscope.app.settings_state import ImageRenderingBackendChoice
from arrayscope.display.backends.pyqtgraph import PyQtGraphSurface

_SOFTWARE_RENDERER_MARKERS = ("llvmpipe", "softpipe", "swrast", "software rasterizer")

# Cached AUTO probe result for this process: (resolved_choice, reason).
_auto_resolution_cache: tuple[ImageRenderingBackendChoice, str] | None = None


def create_image_view(settings=None, *, notify=None):
    """Create the selected image view implementation.

    ``AUTO`` resolves from measured evidence (X5a, 2026-07): on Linux with a
    real hardware GL context the VisPy backend presented first frames faster
    in every tiled scenario benchmarked on Wayland and XWayland (Intel and
    NVIDIA), and level-only changes are uniform updates that keep resident
    textures instead of re-uploading CPU-windowed pixels.  PyQtGraph remains
    the choice for software GL, headless environments, platforms without
    reference traces, or any probe failure, and users can always pin either
    backend explicitly.
    """

    choice = getattr(settings, "image_rendering_backend", ImageRenderingBackendChoice.AUTO)
    choice_value = getattr(choice, "value", choice)
    if choice_value == ImageRenderingBackendChoice.AUTO.value:
        resolved, reason = resolve_auto_backend_choice()
        choice_value = resolved.value
        if callable(notify):
            notify(f"Image rendering backend: {choice_value} | {reason}")
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

    The decision rule is documented in ADR 0047 and is deliberately
    conservative: VisPy is selected only where the X5a reference traces exist
    (Linux) and only when a live hardware GL context can be created.
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
    renderer = _probe_hardware_gl_renderer()
    if renderer is None:
        return (ImageRenderingBackendChoice.PYQTGRAPH, "no usable OpenGL context")
    lowered = renderer.lower()
    if any(marker in lowered for marker in _SOFTWARE_RENDERER_MARKERS):
        return (
            ImageRenderingBackendChoice.PYQTGRAPH,
            f"SW OpenGL [{renderer}]",
        )
    return (ImageRenderingBackendChoice.VISPY, f"HW OpenGL [{renderer}]")


def _probe_hardware_gl_renderer() -> str | None:
    """Create a short-lived GL context and return its renderer string."""

    try:
        from vispy import app as vispy_app, gloo

        from arrayscope.display.backends.vispy.tiles import query_gpu_device_limits

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
