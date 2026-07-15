"""G3 live gate: source-anchored window-shift fast path through the real window.

The pool-level half of this gate (tests/display/test_window_shift_gate.py)
proves the residency arithmetic: anchored plans + content-keyed payloads make
the VisPy atlas skip interior-chunk uploads across a one-pixel window shift.
This module drives the same scenario through the REAL ArrayScopeWindow flow —
view-state window mutation, render scheduling, frame session, tiled commit —
and measures actual texture uploads and pixel truth.

Live status (2026-07-15): the fast path does NOT engage end-to-end. The
divergence, in order:

* window/frame_controller.py:617 — the live frame plan is built by
  ``self._montage_frame_planner().plan(...)`` WITHOUT ``source_anchoring``,
  so ``FramePlanner._plan_single`` (display/frame_planner.py:156) takes the
  unanchored branch: classic (512, 1024) tile shape, ``source_rect=None``,
  ``source_content_key=None``.
* window/frame_effects.py:1987/2024/2073 — every live commit passes
  ``frame_plan=session.frame_plan`` and a prebuilt montage ``tile_state``,
  so the only anchoring wiring — ``_frame_plan_for_display``
  (window/display_presenter.py:348, gated on ``tile_residency_kind ==
  "gpu_atlas"`` at :373) and the region-payload producer
  ``_tile_presentation_for_display_image`` (display_presenter.py:802) — is
  unreachable: the ``frame_plan or ...`` / ``if tile_state is None`` fallbacks
  never fire.
* window/frame_session.py:1827 + window/montage_viewport.py:486 — the payload
  identities that actually reach the atlas are
  ``("montage-tile", montage_tile_semantic_key, source_index)`` where the
  semantic key embeds the windowed ViewState (``axis_range_indices``) and the
  window-sized ``viewport_plan.tile_shape``; a one-pixel shift renames every
  texel, so the whole window re-uploads.

``test_window_shift_live_uploads_only_boundary_strips`` is therefore a strict
xfail: it starts failing (XPASS) the moment the live flow plans with source
anchoring and commits per-region payloads, at which point the marker must be
removed and this docstring updated.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.ui.helpers import clear_arrayscope_settings

_WAIT_TIMEOUT_MS = 15_000

CHUNK = 256  # == display.frame_planner.ANCHORED_CHUNK_SHAPE[1]; asserted below
# Four chunks wide so the shifted window keeps interior chunks strictly
# outnumbering the two boundary strips per row (uploads < total / 2).
EXTENT = 4 * CHUNK
START_A = 100
START_B = 101


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


def _apply_window(win, start: int, *, reason: str) -> None:
    state = win.view_state.with_axis_range(
        1,
        indices=tuple(range(start, start + EXTENT)),
        text=f"{start}:{start + EXTENT}",
    )
    win._set_view_state(state)
    win.render(reason=reason)


def _window_settled(win, start: int) -> bool:
    frame = getattr(win, "_committed_display_frame", None)
    if frame is None:
        return False
    indices = frame.geometry.view_state.axis_range_indices[1]
    if not indices or int(indices[0]) != start or len(indices) != EXTENT:
        return False
    session = getattr(win.renderer, "_frame_session", None)
    if session is None:
        return False
    return (
        session.visible_plan_complete()
        and not win.montage_tile_evaluation_controller.is_busy()
        and session.required_target_settled()
    )


def _wait_for_window(win, qtbot, start: int) -> None:
    qtbot.waitUntil(lambda: _window_settled(win, start), timeout=_WAIT_TIMEOUT_MS)


def _committed_value(win, view_x: int, view_y: int):
    """Probe the committed frame the way live hover does (render.py)."""

    geometry = win.renderer.display_geometry
    if geometry is None:
        return None
    context = geometry.context_for_view_point(float(view_x), float(view_y))
    if context is None:
        return None
    return win.renderer._hover_value_from_display(context.mapping)


def _pool(win):
    return win.img_view._vispy_gpu_montage_layer._pool


def _active_source_ids(pool) -> set:
    return {
        pool.source_ids[key]
        for key in pool.active_resident_keys
        if key in pool.source_ids
    }


def test_window_shift_live_pixels_stay_correct(qtbot):
    """E2E sanity half: a one-pixel window shift commits truthful pixels.

    This must stay green regardless of whether the upload fast path engages —
    it proves the harness (window construction, view-state window mutation,
    settle detection, committed-value probing) that the xfail half stands on.
    """

    pytest.importorskip("vispy")
    settings = _use_vispy_backend()
    rng = np.random.default_rng(23)
    data = rng.standard_normal((2 * CHUNK, 8 * CHUNK)).astype(np.float32)
    win = _make_window(qtbot, data)
    try:
        _apply_window(win, START_A, reason="test-window-initial")
        _wait_for_window(win, qtbot, START_A)

        _apply_window(win, START_B, reason="test-window-shift")
        _wait_for_window(win, qtbot, START_B)

        # Interior probe points, well inside chunk boundaries on both axes.
        probes = ((CHUNK // 2, CHUNK // 2), (EXTENT // 2 + 7, CHUNK + 11), (EXTENT - CHUNK, 2 * CHUNK - 1))
        for view_x, view_y in probes:
            qtbot.waitUntil(
                lambda x=view_x, y=view_y: _committed_value(win, x, y) is not None,
                timeout=_WAIT_TIMEOUT_MS,
            )
            value = _committed_value(win, view_x, view_y)
            expected = data[view_y, START_B + view_x]
            assert value == pytest.approx(float(expected)), (
                f"committed value at view ({view_x}, {view_y}) is {value!r}, "
                f"expected source pixel data[{view_y}, {START_B + view_x}] = {expected!r}"
            )
    finally:
        win.close()
        _restore_default_backend(settings)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ADR 0055 G3 fast path is not wired into the live flow: "
        "frame_controller.py:617 plans without source_anchoring (unanchored "
        "single-tile plan, source_content_key=None); frame_effects.py always "
        "supplies session.frame_plan plus montage tile payloads, so "
        "display_presenter._frame_plan_for_display / "
        "_tile_presentation_for_display_image never run; live source_ids "
        "(frame_session.py:1827 via montage_viewport.py:486) embed the axis "
        "window, so a one-pixel shift renames all resident content and "
        "re-uploads the full window."
    ),
)
def test_window_shift_live_uploads_only_boundary_strips(qtbot, monkeypatch):
    """E2E fast-path half: a one-pixel shift re-uploads only boundary strips.

    Strict xfail — see module docstring. When the live flow starts planning
    with source anchoring this XPASSes; remove the marker then.
    """

    pytest.importorskip("vispy")
    from arrayscope.display.frame_planner import ANCHORED_CHUNK_SHAPE
    import arrayscope.display.backends.vispy.tiles as vispy_tiles

    assert ANCHORED_CHUNK_SHAPE == (CHUNK, CHUNK)

    settings = _use_vispy_backend()

    uploads: list[tuple[int, int]] = []
    original_upload = vispy_tiles._upload_texture_plane

    def counting_upload(texture, plane, *, offset, copy):
        shape = tuple(int(value) for value in np.shape(plane)[:2])
        uploads.append(shape)
        return original_upload(texture, plane, offset=offset, copy=copy)

    monkeypatch.setattr(vispy_tiles, "_upload_texture_plane", counting_upload)

    rng = np.random.default_rng(23)
    data = rng.standard_normal((2 * CHUNK, 8 * CHUNK)).astype(np.float32)
    win = _make_window(qtbot, data)
    try:
        _apply_window(win, START_A, reason="test-window-initial")
        _wait_for_window(win, qtbot, START_A)

        plan_a = win.renderer._frame_session.frame_plan
        # Divergence gate: the live plan must be source-anchored for any of
        # the residency arithmetic below to be meaningful. Today this is the
        # first assertion to fail (see module docstring).
        assert plan_a.source_content_key is not None, (
            "live frame plan is not source-anchored: "
            f"tile_shape={plan_a.tile_shape}, regions={len(plan_a.regions)}, "
            "source_content_key=None (frame_controller.py:617 plans without "
            "source_anchoring; presenter fallback display_presenter.py:348 "
            "is unreachable because frame_effects always passes "
            "session.frame_plan)"
        )
        rects_a = {region.source_rect for region in plan_a.regions}
        assert None not in rects_a

        pool = _pool(win)
        resident_a = _active_source_ids(pool)
        assert len(resident_a) >= len(plan_a.regions)

        uploads.clear()
        _apply_window(win, START_B, reason="test-window-shift")
        _wait_for_window(win, qtbot, START_B)

        plan_b = win.renderer._frame_session.frame_plan
        assert plan_b.source_content_key == plan_a.source_content_key
        rects_b = {region.source_rect for region in plan_b.regions}
        rows = {rect[:2] for rect in rects_b}
        columns = {rect[2:] for rect in rects_b}
        total = len(plan_b.regions)
        # First and last column strips per chunk row are clipped by the
        # window, so their content identity legitimately changes on a shift.
        expected_boundary = 2 * len(rows)
        assert len(columns) > 2, "window must contain interior chunk columns"

        # Residency: interior chunks survive the shift byte-identical.
        resident_b = _active_source_ids(pool)
        overlap = resident_a & resident_b
        assert len(overlap) >= total - expected_boundary, (
            f"interior chunks did not survive the shift: overlap={len(overlap)}, "
            f"total={total}, expected_boundary={expected_boundary}"
        )

        # Uploads: only boundary strips may hit the GPU. Count native-height
        # planes so tiny LOD-preview thumbnails cannot mask a full re-upload.
        native_uploads = [shape for shape in uploads if shape[0] >= CHUNK]
        assert len(native_uploads) <= expected_boundary + 2, (
            f"shift uploaded {len(native_uploads)} native strips "
            f"(expected <= {expected_boundary + 2}); all uploads: {uploads}"
        )
        assert len(native_uploads) < total / 2, (
            f"shift re-uploaded {len(native_uploads)} of {total} tiles — "
            "fast path did not engage"
        )
    finally:
        win.close()
        _restore_default_backend(settings)
