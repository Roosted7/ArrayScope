"""Live framebuffer gate for VisPy physical presentation truth (P9).

Field defect 2026-07-15: on a montage index scroll (complex data,
phase_color, PAL-relaxed LUT) every presentation was acknowledgement-only
while a page visual physically held a stale ``u_component_mode``/``a_mode``.
Zero-magnitude complex texels then rendered LUT(0) — the PAL-relaxed orange
(249, 127, 16) — where correct state renders black.  The identity layer
cannot see this class by construction, so the gate below samples the actual
VisPy canvas framebuffer:

* after settle, no pixel may be the PAL-relaxed LUT[0] orange;
* after injecting a wrong component uniform into the live page visual, a
  re-present must physically repair the visual (black background again)
  instead of re-acknowledging the stale draw.

Follows the ``test_window_shift_live_path`` harness style; offscreen
compatible (skips when the VisPy backend is unavailable).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.ui.helpers import clear_arrayscope_settings

_WAIT_TIMEOUT_MS = 15_000

# PAL-relaxed LUT[0] (== LUT[-1]; the map is cyclic): the stale-draw color
# for zero-magnitude complex texels.  Verified against
# phase_colormap().getLookupTable(0.0, 1.0, 256).
_PAL_RELAXED_ORANGE = (249, 127, 16)
_ORANGE_TOLERANCE = 16


def _use_vispy_backend():
    from pyqtgraph.Qt import QtCore
    from arrayscope.app.settings_state import ImageRenderingBackendChoice

    clear_arrayscope_settings()
    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.VISPY.value)
    settings.sync()
    return settings


def _restore_default_backend(settings):
    from arrayscope.app.settings_state import ImageRenderingBackendChoice

    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.PYQTGRAPH.value)
    settings.sync()


def _make_window(qtbot, data):
    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    capabilities = image_view_backend_capabilities(win.img_view)
    if capabilities.name != "vispy":
        win.close()
        pytest.skip("VisPy backend unavailable in this Qt environment")
    assert capabilities.tile_residency_kind == "gpu_atlas"
    return win


def _settled(win) -> bool:
    frame = getattr(win, "_committed_display_frame", None)
    if frame is None:
        return False
    session = getattr(win.renderer, "_frame_session", None)
    if session is None:
        return False
    return (
        session.visible_plan_complete()
        and not win.montage_tile_evaluation_controller.is_busy()
        and session.required_target_settled()
    )


def _wait_settled(win, qtbot) -> None:
    qtbot.waitUntil(lambda: _settled(win), timeout=_WAIT_TIMEOUT_MS)


def _render_rgb(win) -> np.ndarray:
    """Render the live VisPy scene to an offscreen FBO and return HxWx3 ints."""

    frame = np.asarray(win.img_view._vispy_canvas.render())
    assert frame.ndim == 3 and frame.shape[-1] in (3, 4)
    return frame[..., :3].astype(np.int16)


def _orange_pixel_count(rgb: np.ndarray) -> int:
    reference = np.array(_PAL_RELAXED_ORANGE, dtype=np.int16)
    return int(np.count_nonzero(np.all(np.abs(rgb - reference) <= _ORANGE_TOLERANCE, axis=-1)))


def _bright_pixel_count(rgb: np.ndarray) -> int:
    return int(np.count_nonzero(rgb.sum(axis=-1) > 90))


def _live_page_visual(win):
    layer = win.img_view._vispy_gpu_montage_layer
    for visual in layer._visuals_by_page:
        if bool(getattr(visual, "visible", False)):
            return layer, visual
    raise AssertionError("no visible VisPy tile page visual")


def test_phase_color_zero_background_never_presents_lut_zero_orange(qtbot):
    """Hard-zero complex background + phase_color + PAL-relaxed must render
    black; an injected wrong component uniform must be physically repaired by
    the next re-present, not re-acknowledged."""

    pytest.importorskip("vispy")
    settings = _use_vispy_backend()
    data = np.zeros((128, 128), dtype=np.complex64)
    data[8:24, 8:24] = 40.0 + 0.0j  # small bright block so levels span > 0
    win = _make_window(qtbot, data)
    try:
        win._on_channel_clicked("complex")  # phase_color display, PAL-relaxed LUT
        win.render(reason="test-phase-initial")
        _wait_settled(win, qtbot)
        layer, visual = _live_page_visual(win)
        assert float(visual._component_mode) == 2.0  # ShaderComponent.ABS
        assert np.all(np.asarray(visual.mode_data) == 4.0)  # phase_color quads

        healthy = _render_rgb(win)
        healthy_orange = _orange_pixel_count(healthy)
        assert healthy_orange == 0, (
            f"settled phase_color frame contains {healthy_orange} PAL-relaxed "
            "LUT[0] orange pixels; zero-magnitude background must render black"
        )
        healthy_bright = _bright_pixel_count(healthy)

        # Inject the field-defect corruption: a stale component uniform on the
        # live page visual (its mapping key still looks fresh, so the ordinary
        # desired-state caches call this page clean).
        visual._component_mode = 3.0
        corrupted = _render_rgb(win)
        corrupted_orange = _orange_pixel_count(corrupted)
        corrupted_bright = _bright_pixel_count(corrupted)
        assert corrupted_orange > 100 and corrupted_bright > 4 * max(1, healthy_bright), (
            "injected wrong component uniform did not visibly corrupt the "
            f"framebuffer (orange={corrupted_orange}, bright={corrupted_bright}, "
            f"healthy bright={healthy_bright}) — the recovery half of this gate "
            "would be vacuous"
        )

        # A levels nudge re-presents through the uniforms-only path; the
        # physical audit must repair the visual rather than acknowledge a
        # no-op from divergent state.
        low, high = (float(value) for value in win.img_view.getLevels())
        win.img_view.setLevels(low, high * 1.02 + 1e-6)
        win.render(reason="test-phase-repair")
        _wait_settled(win, qtbot)

        def _recovered() -> bool:
            return float(visual._component_mode) == 2.0 and _orange_pixel_count(_render_rgb(win)) == 0

        qtbot.waitUntil(_recovered, timeout=_WAIT_TIMEOUT_MS)
        recovered = _render_rgb(win)
        assert _orange_pixel_count(recovered) == 0
        # Background is black again: brightness back to the healthy order of
        # magnitude, not the corrupted full-field wash.
        assert _bright_pixel_count(recovered) < corrupted_bright / 2
        assert float(visual._component_mode) == 2.0
        assert np.all(np.asarray(visual.mode_data) == 4.0)
    finally:
        win.close()
        _restore_default_backend(settings)
