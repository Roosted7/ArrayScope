"""Onscreen pixel + responsiveness assertions for the tile pipeline (ADR 0051).

Every test drives the production window on real hardware.  "Tile shows wrong
content" and "event loop hangs" fail HERE, not in someone's eyes.
See conftest for the opt-in invocation.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu_interaction

# Thomas's bar (gui-responsiveness memory): any >50 ms synchronous step in the
# GUI loop during interaction is a bug to profile.  The heartbeat samples every
# loop iteration, so the max gap IS the longest synchronous step.
MAX_INTERACTION_GAP_MS = 50.0


def test_montage_presents_every_tile_with_its_own_content(montage_window):
    h = montage_window
    h.fit_view()
    h.assert_tile_identity_ramp()
    h.assert_lifecycle_settled()


@pytest.mark.xfail(
    reason="known defect (ADR 0051 context): enabling the montage after the "
    "single-frame view has settled keeps a stale fit, so the session shows "
    "the montage wrongly scaled in a corner; scheduled for the "
    "presentation-pipeline rework",
    strict=False,
)
def test_view_fits_montage_when_enabled_after_settle():
    """Reproduces the field report 'sessions often open with wrongly scaled
    items': the fast path (montage enabled right after open, as in the shared
    fixture) fits correctly, but enabling it after the single-frame view has
    fully settled leaves the old fit in place."""

    from tests.gpu_interaction.conftest import Harness, synthetic_montage_data
    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    from arrayscope.app.launch import _create_window

    app, win = _create_window(
        synthetic_montage_data(), title="gpu-harness-late-montage"
    )
    try:
        h = Harness(app, win)
        h.pump(3.0)  # let the single-frame view settle completely
        win._set_view_state(win.view_state.with_montage_axis(2, text=":"))
        win.render(reason="gpu-harness-late-montage")
        assert h.wait_settled(timeout=20.0)
        (x0, x1), (y0, y1) = h.session.view_range
        height, width = h.session.plan.display_shape
        span_x, span_y = x1 - x0, y1 - y0
        assert span_x <= width * 1.5 and span_y <= height * 1.5, (
            f"fit shows {span_x:.0f}x{span_y:.0f} data units for a "
            f"{width}x{height} montage (wrongly scaled on montage enable)"
        )
    finally:
        win.close()
        for _ in range(50):
            app.processEvents()


def test_pan_keeps_event_loop_responsive_and_content_correct(montage_window):
    h = montage_window
    h.fit_view()
    view = h.win.img_view.getView()
    state = {"n": 0}

    def pan_step():
        (x0, x1), (y0, y1) = h.session.view_range
        dy = (y1 - y0) * (0.15 if state["n"] % 2 == 0 else -0.15)
        state["n"] += 1
        view.setRange(xRange=(x0, x1), yRange=(y0 + dy, y1 + dy), padding=0)

    gaps = h.heartbeat_gaps(3.0, step=pan_step, step_interval=0.1)
    worst = max(gaps)
    assert worst <= MAX_INTERACTION_GAP_MS, (
        f"event loop hung {worst:.0f} ms during pan (bar: {MAX_INTERACTION_GAP_MS} ms)"
    )
    # Back to rest: content must still be each tile's own.
    h.fit_view()
    assert h.wait_settled()
    h.assert_tile_identity_ramp()
    h.assert_lifecycle_settled()


def test_index_scrub_and_return_shows_no_stale_content_or_wedged_claims(montage_window):
    """The scrub-back defect class: session replacement leaked singleflight
    claims, so returning to a previous index window presented wedged/stale
    LOD.  Scrub away and back; every tile must show its own content and the
    lifecycle machine must audit clean."""

    h = montage_window
    h.fit_view()
    axis = 2

    def select(text: str) -> None:
        h.win._set_view_state(h.win.view_state.with_montage_axis(axis, text=text))
        h.win.render(reason="gpu-harness-scrub")

    for text in ("9:27", "18:36", "0:18", ":"):
        select(text)
        assert h.wait_settled(timeout=20.0), f"never settled after scrub to {text!r}"

    h.fit_view()
    h.assert_tile_identity_ramp()
    h.assert_lifecycle_settled()
    pyramid = h.session.lod_pyramid
    if pyramid is not None:
        assert int(getattr(pyramid, "pending_count", 0) or 0) == 0, (
            "pyramid singleflight claims wedged after scrub-back"
        )


def test_idle_stays_settled_after_interaction(montage_window):
    """The idle-loop defect class: parked upserts re-emitted every commit kept
    the app at ~120 commits+draws/s at idle.  After interaction settles, the
    machine must report no re-armable work and the dirty queues must drain."""

    h = montage_window
    h.fit_view()
    assert h.wait_settled()
    h.pump(1.0)
    s = h.session
    assert not s.dirty_payloads, (
        f"idle dirty queue never drains: {sorted(s.dirty_payloads)}"
    )
    assert not s.pending_payload_upserts, (
        f"idle upsert queue never drains: {sorted(s.pending_payload_upserts)}"
    )
    h.assert_lifecycle_settled()
