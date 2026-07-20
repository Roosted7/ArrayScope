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

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    apply_plane,
    clear_arrayscope_settings,
    committed_value,
    make_backend_window,
    plane_settled,
    restore_default_backend,
    use_vispy_backend,
)

HEIGHT = 96
WIDTH = 128


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
    apply_plane(win, 0, reason="test-retention-initial")
    qtbot.waitUntil(lambda: plane_settled(win, 0), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
    assert not _surface_blank(win)

    # Display-cache MISS for plane 1 (prefetch is off by default).
    state_1 = win.view_state.with_slice(0, 1)
    assert win.operation_evaluator.cached_display_tile(state_1) is None

    apply_plane(win, 1, reason="test-retention-scrub")

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
    assert committed_value(win, WIDTH // 2, HEIGHT // 2) is None

    # The retained plane never blanks at any point until the replacement
    # commit, and the replacement then shows plane-1 pixel truth.
    seen_blank = []

    def _settled_without_blank() -> bool:
        if _surface_blank(win):
            seen_blank.append(True)
        return plane_settled(win, 1)

    qtbot.waitUntil(_settled_without_blank, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
    assert not seen_blank, "surface blanked between the scrub step and the replacement commit"
    assert not _surface_blank(win)
    value = committed_value(win, WIDTH // 2, HEIGHT // 2)
    assert value == pytest.approx(float(data[1, HEIGHT // 2, WIDTH // 2]))


def _assert_document_change_blanks(qtbot, win):
    from arrayscope.display.backends.base import surface_for_view

    qtbot.waitUntil(lambda: plane_settled(win, 1), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
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
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
    finally:
        delattr(surface, "invalidate_tiled_presentation")
    transitions = [hide for reason, hide in calls if reason == "frame-session-transition"]
    assert transitions, "the operation change never rebirthed the frame session"
    assert transitions[-1] is True, "document change must not retain the stale presentation"
    assert not bool(getattr(win.renderer, "_frame_session_transition_retained_pixels", True))


def _assert_stage_backed_scrub_replaces_retained_plane_while_interactive(qtbot, win, data):
    """Pin the measured field condition instead of racing the 120 ms timer.

    The successor can read native pixels from the already-materialized stage,
    so its only DESIRED rung is correctness/first-pixel work.  Interaction
    parks DISPLAY_PREPARATION but leaves DISPLAY_PREVIEW runnable; the retained
    predecessor must therefore be replaced without waiting for quiet.
    """

    from arrayscope.kernel import Lane
    from arrayscope.window.diagnostics_snapshot import collect_runtime_diagnostics_snapshot

    win._set_view_state(win.view_state.with_image_axes(1, 2))
    # FFT over the scrub axis is non-windowable: the initial plane must
    # materialize one full-array stage, making plane 1 a genuine hot
    # stage-backed successor rather than cold native work.
    win.request_operation("centered_fft", 0)
    apply_plane(win, 0, reason="test-retention-interactive-initial")
    qtbot.waitUntil(lambda: plane_settled(win, 0), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
    assert win.operation_evaluator.stage_cache.diagnostics().entries >= 1

    state_1 = win.view_state.with_slice(0, 1)
    assert win.operation_evaluator.cached_display_tile(state_1) is None

    # Hold the viewport interaction edge open deterministically.  No fixed
    # sleep is involved, and the real governor applies the production quotas.
    timer = getattr(win, "_viewport_interaction_quiet_timer", None)
    if timer is not None:
        timer.stop()
    win._viewport_interaction_active = True
    win._note_interaction_state_changed()
    assert win.kernel._lane_quotas[Lane.DISPLAY_PREVIEW] == 1
    assert win.kernel._lane_quotas[Lane.DISPLAY_PREPARATION] == 0

    before = collect_runtime_diagnostics_snapshot(win).montage
    apply_plane(win, 1, reason="test-retention-interactive-successor")
    during = collect_runtime_diagnostics_snapshot(win).montage
    assert during.slice_retention_transitions == before.slice_retention_transitions + 1
    assert during.slice_retention_replacements == before.slice_retention_replacements
    assert during.slice_retention_active
    assert during.slice_retention_inflight_age_ms >= 0.0

    qtbot.waitUntil(lambda: plane_settled(win, 1), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

    session = win.renderer._frame_session
    assert session is not None
    assert session.tile_compute_stage_backed >= 1
    assert win._interaction_active_now()
    after = collect_runtime_diagnostics_snapshot(win).montage
    assert after.slice_retention_transitions == during.slice_retention_transitions
    assert after.slice_retention_replacements == during.slice_retention_replacements + 1
    assert not after.slice_retention_active
    assert after.slice_retention_inflight_age_ms == 0.0
    assert after.slice_retention_last_replacement_ms > 0.0
    assert after.slice_retention_max_replacement_ms >= after.slice_retention_last_replacement_ms
    value = committed_value(win, WIDTH // 2, HEIGHT // 2)
    from arrayscope.operations.dim_ops import centered_fft

    expected_complex = centered_fft(data, 0)[1, HEIGHT // 2, WIDTH // 2]
    # Backends currently expose either the exact complex probe or its active
    # display mapping here; both must come from the successor plane.
    expected = expected_complex if np.iscomplexobj(value) else abs(expected_complex)
    assert value == pytest.approx(expected)


def test_scrub_step_retains_previous_plane_pyqtgraph(qtbot):
    clear_arrayscope_settings()
    rng = np.random.default_rng(17)
    data = rng.standard_normal((4, HEIGHT, WIDTH)).astype(np.float32)
    win = make_backend_window(qtbot, data, backend="pyqtgraph")
    try:
        _assert_scrub_step_never_blanks(qtbot, win, data)
        _assert_document_change_blanks(qtbot, win)
    finally:
        win.close()


def test_scrub_step_retains_previous_plane_vispy(qtbot):
    pytest.importorskip("vispy")
    settings = use_vispy_backend()
    rng = np.random.default_rng(19)
    data = rng.standard_normal((4, HEIGHT, WIDTH)).astype(np.float32)
    win = make_backend_window(qtbot, data, backend="vispy")
    try:
        _assert_scrub_step_never_blanks(qtbot, win, data)
        _assert_document_change_blanks(qtbot, win)
    finally:
        win.close()
        restore_default_backend(settings)


def test_interactive_stage_backed_scrub_replaces_retained_plane_pyqtgraph(qtbot):
    clear_arrayscope_settings()
    rng = np.random.default_rng(23)
    data = rng.standard_normal((4, HEIGHT, WIDTH)).astype(np.float32)
    win = make_backend_window(qtbot, data, backend="pyqtgraph")
    try:
        _assert_stage_backed_scrub_replaces_retained_plane_while_interactive(
            qtbot,
            win,
            data,
        )
    finally:
        win._viewport_interaction_active = False
        win.close()


def test_interactive_stage_backed_scrub_replaces_retained_plane_vispy(qtbot):
    pytest.importorskip("vispy")
    settings = use_vispy_backend()
    rng = np.random.default_rng(29)
    data = rng.standard_normal((4, HEIGHT, WIDTH)).astype(np.float32)
    win = make_backend_window(qtbot, data, backend="vispy")
    try:
        _assert_stage_backed_scrub_replaces_retained_plane_while_interactive(
            qtbot,
            win,
            data,
        )
    finally:
        win._viewport_interaction_active = False
        win.close()
        restore_default_backend(settings)
