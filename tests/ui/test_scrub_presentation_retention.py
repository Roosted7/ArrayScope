"""Slice-scrub presentation retention gates.

A single-slice (non-montage) scrub step rebirths the frame session; the
transition used to blank the surface (`invalidate_tiled_presentation`)
until the successor's evaluation committed, so scrubbing faster than
per-slice evaluation showed a black canvas 40+% of the time.

The fix splits "pixels stay visible" from "mappings are not evidence"
(ADR 0051): a slice-index-only transition keeps DRAWING the predecessor's
plane (stale-but-honest preview) while the successor's bookkeeping starts
cold, and the first commit replaces the pixels atomically.  Everything
else — document revision change, operations, geometry — still blanks.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.ui.helpers import clear_arrayscope_settings

_WAIT_TIMEOUT_MS = 15_000

HEIGHT = 96
WIDTH = 128


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


def _make_window(qtbot, data, *, backend: str):
    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    capabilities = image_view_backend_capabilities(win.img_view)
    if capabilities.name != backend:
        win.close()
        pytest.skip(f"{backend} backend unavailable in this Qt environment")
    return win


def _apply_plane(win, index: int, *, reason: str) -> None:
    win._set_view_state(win.view_state.with_slice(0, index))
    win.render(reason=reason)


def _plane_settled(win, index: int) -> bool:
    frame = getattr(win, "_committed_display_frame", None)
    if frame is None:
        return False
    if int(frame.geometry.view_state.slice_indices[0]) != int(index):
        return False
    session = getattr(win.renderer, "_frame_session", None)
    if session is None:
        return False
    return (
        session.visible_plan_complete()
        and not win.montage_tile_evaluation_controller.is_busy()
        and session.required_target_settled()
    )


def _committed_value(win, view_x: int, view_y: int):
    geometry = win.renderer.display_geometry
    if geometry is None:
        return None
    context = geometry.context_for_view_point(float(view_x), float(view_y))
    if context is None:
        return None
    return win.renderer._hover_value_from_display(context.mapping)


def _surface_blank(win) -> bool:
    """True when the tiled presentation is hidden (the pre-fix blank)."""

    mode = str(win.img_view.montageDisplayMode())
    if mode == "none":
        return True
    layer = getattr(win.img_view, "_vispy_gpu_montage_layer", None)
    if layer is not None:
        return int(getattr(layer, "_visible_items", 0) or 0) <= 0
    tile_layer = getattr(win.img_view, "_montage_tile_layer", None)
    if tile_layer is not None:
        states = getattr(tile_layer, "_states", {}) or {}
        return not any(
            getattr(getattr(state, "item", None), "isVisible", lambda: False)()
            for state in states.values()
        )
    return False


def _assert_scrub_step_never_blanks(qtbot, win, data):
    win._set_view_state(win.view_state.with_image_axes(1, 2))
    _apply_plane(win, 0, reason="test-retention-initial")
    qtbot.waitUntil(lambda: _plane_settled(win, 0), timeout=_WAIT_TIMEOUT_MS)
    assert not _surface_blank(win)

    # Display-cache MISS for plane 1 (prefetch is off by default).
    state_1 = win.view_state.with_slice(0, 1)
    assert win.operation_evaluator.cached_display_tile(state_1) is None

    _apply_plane(win, 1, reason="test-retention-scrub")

    # The rebirth happened; the successor has not committed yet, and the
    # predecessor's plane must still be on screen (stale-but-honest).
    assert bool(getattr(win.renderer, "_frame_session_transition_retained_pixels", False)), (
        "slice-index-only transition did not take the retained-presentation path"
    )
    assert not _surface_blank(win), (
        "index-scrub with a display-cache miss blanked the canvas while eval runs"
    )

    # Guard: the retained pixels are NOT current for probes/hover.  The
    # committed frame still describes plane 0, and the advanced render
    # generation makes the currency gate reject it until plane 1 commits.
    frame = win._committed_display_frame
    assert int(frame.geometry.view_state.slice_indices[0]) == 0
    assert not win.renderer._is_committed_display_frame_current(frame)
    assert _committed_value(win, WIDTH // 2, HEIGHT // 2) is None

    # The retained plane never blanks at any point until the replacement
    # commit, and the replacement then shows plane-1 pixel truth.
    seen_blank = []

    def _settled_without_blank() -> bool:
        if _surface_blank(win):
            seen_blank.append(True)
        return _plane_settled(win, 1)

    qtbot.waitUntil(_settled_without_blank, timeout=_WAIT_TIMEOUT_MS)
    assert not seen_blank, "surface blanked between the scrub step and the replacement commit"
    assert not _surface_blank(win)
    value = _committed_value(win, WIDTH // 2, HEIGHT // 2)
    assert value == pytest.approx(float(data[1, HEIGHT // 2, WIDTH // 2]))


def _assert_document_change_blanks(qtbot, win):
    from arrayscope.display.backends.base import surface_for_view

    qtbot.waitUntil(lambda: _plane_settled(win, 1), timeout=_WAIT_TIMEOUT_MS)
    # A document revision/steps change is new semantic content: stale pixels
    # from the old document are lies, so the transition must hide them.
    surface = surface_for_view(win.img_view)
    calls = []
    original = surface.invalidate_tiled_presentation

    def recording(reason, *, hide_pixels=True):
        calls.append((str(reason), bool(hide_pixels)))
        return original(reason, hide_pixels=hide_pixels)

    surface.invalidate_tiled_presentation = recording
    try:
        session_before = win.renderer._frame_session
        win.request_operation("reverse", 0)
        qtbot.waitUntil(
            lambda: win.renderer._frame_session is not session_before,
            timeout=_WAIT_TIMEOUT_MS,
        )
    finally:
        delattr(surface, "invalidate_tiled_presentation")
    transitions = [hide for reason, hide in calls if reason == "frame-session-transition"]
    assert transitions, "the operation change never rebirthed the frame session"
    assert transitions[-1] is True, "document change must not retain the stale presentation"
    assert not bool(getattr(win.renderer, "_frame_session_transition_retained_pixels", True))


def test_scrub_step_retains_previous_plane_pyqtgraph(qtbot):
    clear_arrayscope_settings()
    rng = np.random.default_rng(17)
    data = rng.standard_normal((4, HEIGHT, WIDTH)).astype(np.float32)
    win = _make_window(qtbot, data, backend="pyqtgraph")
    try:
        _assert_scrub_step_never_blanks(qtbot, win, data)
        _assert_document_change_blanks(qtbot, win)
    finally:
        win.close()


def test_scrub_step_retains_previous_plane_vispy(qtbot):
    pytest.importorskip("vispy")
    settings = _use_vispy_backend()
    rng = np.random.default_rng(19)
    data = rng.standard_normal((4, HEIGHT, WIDTH)).astype(np.float32)
    win = _make_window(qtbot, data, backend="vispy")
    try:
        _assert_scrub_step_never_blanks(qtbot, win, data)
        _assert_document_change_blanks(qtbot, win)
    finally:
        win.close()
        _restore_default_backend(settings)
