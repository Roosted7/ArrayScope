"""Default-ring smoke for the framebuffer-to-CPU reference oracle.

Ring 1 (default offscreen suite).  Offscreen software GL renders the VisPy
tile shader faithfully for this path (precedent:
tests/ui/test_vispy_phase_framebuffer.py), so this smoke keeps the oracle
itself — geometry mapping, CPU reference, tolerance, vacuity guards — honest
on every push.  It is NOT acceptance for rendering claims: the real-GL gate
is tests/gpu_interaction/test_framebuffer_cpu_reference.py (ring 4).

Pins the oracle mandated by docs/testing/stress-and-trace-strategy.md
(addendum law 2 — intent is not pixels): a settled scene must match the CPU
semantic reference, and an injected wrong levels uniform must make the
oracle FAIL (an oracle that has never failed on an injected fault is
unproven), then pass again after repair.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS

from tests.oracles.framebuffer_reference import assert_frame_matches_cpu_reference
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_vispy_backend,
)

TILE = 32
GRID = 3
COUNT = GRID * GRID


def _gradient_montage_data() -> np.ndarray:
    """(32, 32, 9): frame k = 20*k + smooth x+y gradient.

    Per-tile offsets make swapped/stale content visible; the in-tile gradient
    makes wrong geometry (flip, shifted texcoords, wrong LOD placement)
    visible where constant tiles would hide it.
    """

    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(np.float32)
    gradient = (yy + xx) * (8.0 / (2.0 * (TILE - 1)))
    frames = np.arange(COUNT, dtype=np.float32)[:, None, None] * 20.0 + gradient[None]
    return frames.transpose(1, 2, 0).copy()


def _settled(win) -> bool:
    if getattr(win, "_committed_display_frame", None) is None:
        return False
    return frame_session_settled(win)


def test_settled_montage_matches_cpu_reference_and_fails_on_injected_uniform(qtbot):
    pytest.importorskip("vispy")
    settings = use_vispy_backend()
    win = make_backend_window(qtbot, _gradient_montage_data(), require_gpu_atlas=True)
    try:
        # Offscreen windows never get a layout pass unless shown at an
        # explicit size; a subpixel canvas would starve the oracle's
        # per-tile sample floor (its vacuity guard would fail loudly).
        win.resize(720, 600)
        win.show()
        qtbot.waitExposed(win)
        win._set_view_state(win.view_state.with_montage_axis(2, text=":"))
        win.render(reason="fb-cpu-reference-smoke")
        qtbot.waitUntil(lambda: _settled(win), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

        session = win.renderer._frame_session
        required = set(session.required_tile_numbers())
        assert len(required) == COUNT, (
            f"smoke regime drifted: required tiles {sorted(required)}"
        )
        # Regime guard (strategy law 3): this smoke covers the native-LOD
        # scalar regime only; entering another regime silently must fail.
        for number in sorted(required):
            payload = session.display_tile_payloads[int(number)]
            level = 0 if payload.lod is None else int(payload.lod.level)
            assert level == 0, (
                f"tile {number} presented LOD level {level}; the smoke pins "
                "the native-resolution regime"
            )

        report = assert_frame_matches_cpu_reference(win)
        assert {tile.tile_number for tile in report.tiles} == required
        assert all(
            tile.samples >= report.min_samples_per_tile for tile in report.tiles
        )

        # Fault injection: a wrong levels uniform on the live page visual —
        # CPU-side truth (payloads, UI levels) untouched, so every label
        # stays truthful while the frame is visibly wrong.
        layer = win.img_view._vispy_gpu_montage_layer
        visuals = [
            visual
            for visual in layer._visuals_by_page
            if bool(getattr(visual, "visible", False))
        ]
        assert visuals, "no visible VisPy tile page visual"
        originals = [tuple(visual._levels) for visual in visuals]
        for visual in visuals:
            low, high = visual._levels
            visual.set_levels((low, low + (high - low) * 4.0))
        with pytest.raises(AssertionError, match="diverges from the CPU"):
            assert_frame_matches_cpu_reference(win)

        # Repair: restoring the uniform restores the oracle — the failure
        # above was caused by the injected fault, nothing else.
        for visual, levels in zip(visuals, originals):
            visual.set_levels(levels)
        assert_frame_matches_cpu_reference(win)
    finally:
        win.close()
        restore_default_backend(settings)
