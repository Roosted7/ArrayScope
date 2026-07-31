"""Screen-canvas draw scheduling: a hidden canvas must never eat a draw.

The residual ``requested>drawn`` wedge (2026-07-31 D1 A/B, control arm):
a counted ``_request_wgpu_canvas_draw(count_presentation=True)`` reached the
canvas, the pacer's one-shot fired while ``isVisible()`` was False, and
``_invoke_draw`` cleared ``_draw_scheduled`` and returned — dropping the
only in-flight draw.  The view's ``_wgpu_canvas_update_pending`` stayed
True, so every later uncounted request coalesced away and no wakeup was
armed anywhere: ``requested=191 drawn=190 draw_pending=True`` until an
expose happened to arrive.  That is ground rule 11's forbidden shape — a
deferral whose resumption owner is "a later show/expose will pick it up"
(docs/redesign/wgpu-frame-pacing-2026-07-21.md).

The contract pinned here: a paced draw on a hidden canvas still runs the
draw callback, so the view's ack publishes ("the presentation gate can
never stay armed forever on a hidden or headless surface" —
``_present_wgpu_frame_to_screen``) — while the swapchain stays untouched,
because ``ensure_context`` answers ``None`` whenever the widget cannot
present.  No new timer exists (ground rule 12): the deferral itself is
gone, not re-owned.

Offscreen-safe: nothing here touches wgpu — the canvas only reaches for a
device inside ``ensure_context``, which these tests never let past its
visibility guard.
"""

import os
import time

import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")


def _pump_until(qt_app, condition, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.001)
    return condition()


class _RecordingContext:
    """Stand-in swapchain: records reconfigures without touching a device."""

    def __init__(self):
        self.sizes = []

    def set_physical_size(self, *size):
        self.sizes.append(tuple(size))


@pytest.fixture
def canvas(qt_app):
    from arrayscope.display.backends.wgpu.screen_canvas import WgpuScreenCanvas

    canvas = WgpuScreenCanvas()
    try:
        yield canvas
    finally:
        canvas._context = None  # never unconfigure a test double
        canvas.teardown()
        canvas.deleteLater()
        qt_app.processEvents()


def test_paced_draw_on_a_hidden_canvas_still_runs_the_callback(qt_app, canvas):
    """The wedge mechanism: the pacer must not drop a requested draw on
    visibility — the callback owns the hidden case (its ack must publish)."""

    draws = []
    assert not canvas.isVisible()
    canvas.request_draw(lambda: draws.append("draw"))

    assert _pump_until(qt_app, lambda: draws, timeout_s=2.0), (
        "the paced draw was dropped on the visibility bail: no callback ran, "
        "no wakeup is armed, and only a show/expose could ever recover it"
    )
    assert canvas._draw_scheduled is False

    # The draw completed, so a new request must re-arm rather than coalesce
    # against state a dropped draw left behind.
    canvas.request_draw()
    assert _pump_until(qt_app, lambda: len(draws) >= 2), (
        "a follow-up request after a hidden draw never ran"
    )


def test_ensure_context_never_hands_out_a_swapchain_while_hidden(qt_app, canvas):
    """Hidden draws run the callback, so the swapchain guard moves here: a
    cached context is only handed out while the widget can present.  (Fifo
    acquire on an unmapped surface can block the GUI thread indefinitely.)"""

    sentinel = object()
    canvas._context = sentinel
    assert not canvas.isVisible()
    assert canvas.ensure_context(device=None) is None

    canvas.show()
    assert canvas.isVisible()
    assert canvas.ensure_context(device=None) is sentinel


def test_surface_resize_stays_synchronous_until_a_swapchain_exists(qt_app, canvas):
    """The cold-establish resize keys on "no swapchain yet", not on the draw
    clock: hidden draws now stamp ``_last_draw_started`` without ever
    configuring, so the clock cannot answer whether the chain exists."""

    draws = []
    canvas.request_draw(lambda: draws.append("draw"))
    assert _pump_until(qt_app, lambda: draws, timeout_s=2.0)

    # No context yet: the establishing resize must stay synchronous (the
    # cold configure+acquire belongs here, off the first content frame).
    before = len(draws)
    canvas._on_surface_resized()
    assert len(draws) == before + 1, (
        "the swapchain-establishing resize was deferred to the paced path"
    )

    # With a swapchain, resizes coalesce onto the paced frame instead.
    canvas._context = _RecordingContext()
    before = len(draws)
    canvas._on_surface_resized()
    assert len(draws) == before, "a post-establish resize drew synchronously"
    assert _pump_until(qt_app, lambda: len(draws) > before), "the coalesced resize never drew"


def test_a_resize_while_hidden_stays_armed_for_the_first_visible_frame(qt_app, canvas):
    """Hidden draws must not reconfigure the swapchain to a size nothing is
    showing — but the resize cannot be dropped either.  Its owner is
    ``showEvent``'s unconditional request_draw, which is what makes deferring
    it legal here where dropping the draw was not."""

    draws = []
    canvas.request_draw(lambda: draws.append("draw"))
    assert _pump_until(qt_app, lambda: draws, timeout_s=2.0)
    context = _RecordingContext()
    canvas._context = context

    canvas._on_surface_resized()
    assert _pump_until(qt_app, lambda: len(draws) >= 2, timeout_s=2.0)
    assert context.sizes == [], "a hidden canvas reconfigured its swapchain"
    assert canvas._resize_pending is True, "the hidden resize was dropped, not deferred"

    canvas.show()
    assert _pump_until(qt_app, lambda: context.sizes, timeout_s=2.0), (
        "showing the canvas never applied the deferred resize"
    )
    assert context.sizes[-1] == canvas.get_physical_size()
    assert canvas._resize_pending is False


def test_counted_presentation_request_on_hidden_screen_canvas_still_acks(qt_app, monkeypatch):
    """The full pair, at the view: the observed wedge was requested=191
    drawn=190 draw_pending=True for the whole 0.5 s settle wait, with the
    canvas's dropped draw on one side and the view's coalescing guard
    (``_wgpu_canvas_update_pending``) on the other.  A counted request on a
    hidden screen canvas must settle without any show/expose — pixels are
    deferred, the acknowledgement is not."""

    import arrayscope.display.backends.wgpu.screen_canvas as screen_canvas_module
    from tests.display.test_imageview2d import _view_class

    monkeypatch.setattr(screen_canvas_module, "screen_present_unavailable_reason", lambda: None)
    view = _view_class("wgpu")(present_method="screen")
    try:
        assert view.wgpuPresentMethod() == "screen"
        canvas = view._wgpu_canvas
        assert isinstance(canvas, screen_canvas_module.WgpuScreenCanvas)
        assert not canvas.isVisible()

        view._request_wgpu_canvas_draw(
            count_presentation=True, presentation_reason="hidden-canvas-oracle"
        )
        # The wild shape's second half: an uncounted rerequest mid-flight
        # coalesces onto the pending draw rather than arming its own.
        view._request_wgpu_canvas_draw()

        def settled():
            diagnostics = view.wgpuPresentationDiagnostics()
            return (
                diagnostics["tile_presentation_draw_count"]
                == diagnostics["tile_presentation_request_count"]
                and not view.presentationDrawPending()
            )

        assert _pump_until(qt_app, settled, timeout_s=2.0), (
            "counted draw request never acked on a hidden screen canvas: "
            f"{view.wgpuPresentationDiagnostics()}"
        )
        diagnostics = view.wgpuPresentationDiagnostics()
        assert diagnostics["tile_presentation_request_count"] == 1
        # No pixels were presented — the surface never existed.
        assert diagnostics["wgpu_screen_presents"] == 0
        assert bool(getattr(view, "_wgpu_canvas_update_pending", False)) is False
    finally:
        view.close()
