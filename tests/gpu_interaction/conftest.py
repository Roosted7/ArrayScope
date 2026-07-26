"""Onscreen GPU interaction harness (ADR 0051).

These tests drive the REAL production window on REAL display hardware and
assert on captured pixels, event-loop heartbeat gaps, and the tile lifecycle
machine.  Xvfb/software-GL runs are not evidence (roadmap X5), so the suite
is opt-in:

    ARRAYSCOPE_GPU_TESTS=1 \
    XDG_RUNTIME_DIR=/run/user/$(id -u) WAYLAND_DISPLAY=wayland-0 \
    QT_QPA_PLATFORM=wayland \
    python -m pytest tests/gpu_interaction -n 0 -q

Run serially (``-n 0``): the tests own a visible window and measure GUI-loop
latency; xdist workers building GL contexts next door would poison both.
"""

from __future__ import annotations

import os
from time import monotonic, perf_counter, sleep

import numpy as np
import pytest

from arrayscope.tools.interaction_budget import (
    INTERACTION_SETTLE_HARD_LIMIT_S,
    bounded_interaction_settle_timeout_s,
)


def _display_available() -> bool:
    if os.environ.get("QT_QPA_PLATFORM", "") == "offscreen":
        return False
    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"))


ENABLED = os.environ.get("ARRAYSCOPE_GPU_TESTS", "") == "1" and _display_available()

pytestmark = pytest.mark.gpu_interaction


def wait_for_qt_condition(
    app,
    predicate,
    *,
    timeout_s: float = INTERACTION_SETTLE_HARD_LIMIT_S,
) -> bool:
    """Pump Qt until one condition holds, bounded by the global hard limit."""

    timeout = bounded_interaction_settle_timeout_s(timeout_s)
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        sleep(0.001)
    app.processEvents()
    return bool(predicate())


def pytest_collection_modifyitems(config, items):
    if ENABLED:
        return
    skip = pytest.mark.skip(
        reason="onscreen GPU harness: set ARRAYSCOPE_GPU_TESTS=1 on a real display"
    )
    for item in items:
        if "gpu_interaction" in item.keywords:
            item.add_marker(skip)


# -- synthetic scene ----------------------------------------------------------

TILE = 64
GRID = 6
COUNT = GRID * GRID


def synthetic_montage_data() -> np.ndarray:
    """(64, 64, 36): frame ``k`` is constant ``k`` — per-tile identity in the
    presented gray ramp is analytically known, so a tile showing another
    tile's content (reused atlas slot, stale mip, wrong window) breaks the
    strictly increasing ramp."""

    frames = np.repeat(np.arange(COUNT, dtype=np.float32), TILE * TILE)
    return frames.reshape(COUNT, TILE, TILE).transpose(1, 2, 0).copy()


@pytest.fixture
def montage_window(qt_app):
    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    from arrayscope.app.launch import _create_window
    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from tests.ui.helpers import use_wgpu_backend

    # Requesting ``qt_app`` activates the suite's isolated QSettings namespace;
    # pin the maintained GPU renderer explicitly for physical ring-4 evidence.
    use_wgpu_backend(extra_settings={"first_run_hints_dismissed": True})
    app = win = None
    try:
        app, win = _create_window(
            synthetic_montage_data(),
            title="gpu-harness",
            application_name="ArrayScopeTests",
        )
        assert image_view_backend_capabilities(win.img_view).name == "wgpu"
        harness = Harness(app, win)
        harness.pump(0.3)
        vs = win.view_state
        win._set_view_state(vs.with_montage_axis(2, text=":"))
        win.render(reason="gpu-harness-montage")
        assert harness.wait_settled(), "montage never settled after open"
        yield harness
    finally:
        if win is not None:
            win.close()
        if app is not None:
            for _ in range(50):
                app.processEvents()


class Harness:
    def __init__(self, app, win) -> None:
        self.app = app
        self.win = win

    # -- session/lifecycle -------------------------------------------------

    @property
    def session(self):
        return self.win.renderer._frame_session

    @property
    def lifecycle(self):
        return self.session.lifecycle

    def settled(self) -> bool:
        s = self.session
        pending_draw = getattr(self.win.img_view, "presentationDrawPending", None)
        physical_drawn = not bool(pending_draw()) if callable(pending_draw) else True
        return bool(s is not None and s.visible_plan_complete() and physical_drawn)

    def assert_lifecycle_settled(self) -> None:
        s = self.session
        counters = s.lifecycle.counters()
        # ADR 0051 P2: the diagnostics-gated stall probe is observational only.
        stall_assertions = int(getattr(self.win.renderer, "_montage_stall_assertions", 0) or 0)
        assert stall_assertions == 0, (
            "stall assertion probe fired: "
            f"{stall_assertions}x, last={getattr(self.win.renderer, '_montage_watchdog_last_stall', None)}"
        )
        assert counters["dangling_claims"] == 0, f"leaked claims: {s.lifecycle.dangling_claims()}"
        assert counters["evaluating"] == 0, (
            f"immortal loading tiles: {sorted(s.lifecycle.evaluating_tiles)}"
        )
        active = {int(tile) for tile in s._last_active_tiles}
        parked_active = s.lifecycle.parked_tiles & active
        assert not parked_active, f"parked tiles inside active scope: {sorted(parked_active)}"
        # Semantic-vs-physical agreement: every active payload identity must be
        # the identity acknowledged by the maintained backend's physical row.
        from arrayscope.display.model.frame import tile_ack_identity

        rows = self.win.img_view.tileTruthPhysicalRows()
        for tile in active:
            payload = s.display_tile_payloads.get(tile)
            assert payload is not None, f"active tile {tile} has no committed payload"
            assert tile in rows, f"active tile {tile} has no physical truth row"
            assert rows[tile]["physical_acknowledged_identity"] == tile_ack_identity(payload)

    def assert_visual_mapping_matches_residency(self) -> None:
        """Every physically drawn tile must resolve all named resident pages."""

        rows = self.win.img_view.tileTruthPhysicalRows()
        for tile, row in rows.items():
            bindings = tuple(row.get("physical_page_bindings", ()) or ())
            assert bindings, f"tile {tile} has no physical page bindings"
            assert all(binding.get("actual_key") is not None for binding in bindings)

    def prepare_image_layer_pixel_sampling(self) -> None:
        """Hide independent composition overlays before sampling image pixels.

        The default restored ROIs/profile line cross tiles 6 and 7 in the GPU
        harness.  They are valid composition pixels, but they must not turn a
        tile-texture identity assertion into an ROI-colour assertion.
        """

        self.win._clear_rois()
        self.win.img_view.setRoiInfoRows(())
        self.win.img_view.clearMontageTileOverlays()
        self.win.img_view.hideProfileMarker()
        self.win._clear_image_hover_state()
        from pyqtgraph.Qt import QtWidgets

        for hints in self.win.img_view.findChildren(
            QtWidgets.QWidget,
            "FirstRunHints",
        ):
            hints.hide()
        self.app.processEvents()
        self.app.processEvents()

    # -- event loop ----------------------------------------------------------

    def pump(self, seconds: float) -> None:
        deadline = monotonic() + seconds
        while monotonic() < deadline:
            self.app.processEvents()

    def wait_settled(self, timeout: float = INTERACTION_SETTLE_HARD_LIMIT_S) -> bool:
        return wait_for_qt_condition(
            self.app,
            self.settled,
            timeout_s=timeout,
        )

    def settlement_diagnostics(self) -> dict[str, object]:
        session = self.session
        image_view = self.win.img_view
        pending_draw = getattr(image_view, "presentationDrawPending", None)
        presentation_diagnostics = getattr(image_view, "presentation_diagnostics", None)
        return {
            "visible_complete": session.visible_plan_complete(),
            "required_unsettled": session.required_target_unsettled_tiles(),
            "active_requests": tuple(session.active_tile_requests),
            "dirty": tuple(session.dirty_payloads),
            "upserts": tuple(session.pending_payload_upserts),
            "level_snapshot": session.level_presentation_snapshot(),
            "draw_pending": bool(pending_draw()) if callable(pending_draw) else False,
            "presentation": (
                dict(presentation_diagnostics()) if callable(presentation_diagnostics) else {}
            ),
            "lifecycle": session.lifecycle.counters(),
        }

    def heartbeat_gaps(self, seconds: float, *, step=None, step_interval: float = 0.1):
        """Pump the loop for ``seconds``; return gap samples (ms) between
        iterations, optionally invoking ``step()`` every ``step_interval``."""

        gaps: list[float] = []
        last = perf_counter()
        next_step = last
        deadline = last + seconds
        while True:
            self.app.processEvents()
            now = perf_counter()
            gaps.append((now - last) * 1000.0)
            last = now
            if now >= deadline:
                return gaps
            if step is not None and now >= next_step:
                step()
                next_step = now + step_interval

    # -- pixels ----------------------------------------------------------------

    def fit_view(self) -> None:
        self.toggle_fit_stretch(True)
        self.pump(0.4)
        self.toggle_fit_stretch(False)
        self.pump(0.2)

    def fit_plan_view(self) -> None:
        """Zoom out to the full applied plan without changing its layout."""

        from arrayscope.window.montage_viewport import square_montage_fit_view_range

        viewport = self.win.img_view.graphicsView.viewport().size()
        x_range, y_range = square_montage_fit_view_range(
            self.session.plan,
            (max(1, viewport.height()), max(1, viewport.width())),
        )
        self.win.img_view.getView().setRange(
            xRange=x_range,
            yRange=y_range,
            padding=0,
        )

    def toggle_fit_stretch(self, enabled: bool) -> None:
        self.win.fit_image_to_view(bool(enabled))
        self.app.processEvents()

    def screenshot(self) -> np.ndarray:
        from pyqtgraph.Qt import QtGui

        image = (
            self.win.img_view.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        )
        w, h = image.width(), image.height()
        buf = np.frombuffer(image.constBits(), dtype=np.uint8, count=w * h * 4)
        return buf.reshape(h, w, 4).copy()

    def tile_centers_px(self) -> list[tuple[float, float]]:
        from pyqtgraph.Qt import QtCore

        vb = self.win.img_view.getView()
        gv = vb.scene().views()[0]

        def to_widget(x: float, y: float) -> tuple[float, float]:
            scene_pt = vb.mapViewToScene(QtCore.QPointF(x, y))
            widget_pt = gv.mapTo(self.win.img_view, gv.mapFromScene(scene_pt))
            return float(widget_pt.x()), float(widget_pt.y())

        return [
            to_widget(t.x0 + t.width / 2.0, t.y0 + t.height / 2.0) for t in self.session.plan.tiles
        ]

    def tile_means(self, shot: np.ndarray | None = None, *, half: int = 5) -> list[float]:
        shot = self.screenshot() if shot is None else shot
        means: list[float] = []
        for x, y in self.tile_centers_px():
            xi, yi = round(x), round(y)
            assert yi - half >= 0, f"tile center off-widget: ({xi}, {yi})"
            assert yi + half < shot.shape[0], f"tile center off-widget: ({xi}, {yi})"
            assert xi - half >= 0, f"tile center off-widget: ({xi}, {yi})"
            assert xi + half < shot.shape[1], f"tile center off-widget: ({xi}, {yi})"
            means.append(
                float(shot[yi - half : yi + half + 1, xi - half : xi + half + 1, 0].mean())
            )
        return means

    def tile_medians(self, shot: np.ndarray | None = None, *, half: int = 5) -> list[float]:
        """Center-patch pixel oracle robust to thin ROI/hover overlay lines."""

        shot = self.screenshot() if shot is None else shot
        medians: list[float] = []
        for x, y in self.tile_centers_px():
            xi, yi = round(x), round(y)
            assert yi - half >= 0
            assert yi + half < shot.shape[0]
            assert xi - half >= 0
            assert xi + half < shot.shape[1]
            medians.append(
                float(
                    np.median(
                        shot[
                            yi - half : yi + half + 1,
                            xi - half : xi + half + 1,
                            0,
                        ]
                    )
                )
            )
        return medians

    def tile_pixel_modes(self, shot: np.ndarray | None = None) -> list[float]:
        """Dominant interior red-channel value for constant synthetic tiles.

        The V1 ramp tiles are analytically constant. Sampling the full tile
        interior makes thin lines and even a large ROI label irrelevant while
        still reading the real framebuffer rather than backend metadata.
        """

        from pyqtgraph.Qt import QtCore

        shot = self.screenshot() if shot is None else shot
        vb = self.win.img_view.getView()
        gv = vb.scene().views()[0]

        def to_widget(x: float, y: float) -> tuple[int, int]:
            scene_pt = vb.mapViewToScene(QtCore.QPointF(x, y))
            widget_pt = gv.mapTo(self.win.img_view, gv.mapFromScene(scene_pt))
            return round(widget_pt.x()), round(widget_pt.y())

        modes: list[float] = []
        for tile in self.session.plan.tiles:
            inset_x = max(1.0, float(tile.width) * 0.08)
            inset_y = max(1.0, float(tile.height) * 0.08)
            p0 = to_widget(tile.x0 + inset_x, tile.y0 + inset_y)
            p1 = to_widget(
                tile.x0 + tile.width - inset_x,
                tile.y0 + tile.height - inset_y,
            )
            x0, x1 = sorted((p0[0], p1[0]))
            y0, y1 = sorted((p0[1], p1[1]))
            assert 0 <= x0 <= x1 < shot.shape[1], (
                f"tile {tile.montage_index} interior is outside the framebuffer: "
                f"x=({x0}, {x1}), width={shot.shape[1]}"
            )
            assert 0 <= y0 <= y1 < shot.shape[0], (
                f"tile {tile.montage_index} interior is outside the framebuffer: "
                f"y=({y0}, {y1}), height={shot.shape[0]}"
            )
            interior = shot[y0 : y1 + 1, x0 : x1 + 1, 0]
            assert interior.size
            modes.append(float(np.bincount(interior.reshape(-1), minlength=256).argmax()))
        return modes

    def assert_tile_matches_cpu_reference(self, **kwargs):
        """Framebuffer vs CPU semantic reference for every required tile.

        The generalization of :meth:`assert_tile_identity_ramp` mandated by
        docs/testing/stress-and-trace-strategy.md (addendum law 2): reads
        either WGPU's executor target or PyQtGraph's painted Qt viewport and
        compares each required tile's interior against ``cpu_display_rgba``
        of the committed payload values (component/scale/levels/LUT applied),
        tolerating only calibrated raster rounding. Returns the per-tile
        :class:`FrameReferenceReport`.
        """

        from arrayscope.tools.framebuffer_reference import (
            assert_qt_raster_matches_cpu_reference,
            assert_wgpu_frame_matches_cpu_reference,
        )

        self.prepare_image_layer_pixel_sampling()
        backend = self.win.img_view.surface.capabilities.name
        if backend == "wgpu":
            return assert_wgpu_frame_matches_cpu_reference(self.win, **kwargs)
        if backend == "pyqtgraph":
            return assert_qt_raster_matches_cpu_reference(self.win, **kwargs)
        raise AssertionError(f"unsupported physical-oracle backend {backend!r}")

    def assert_tile_identity_ramp(self, *, tolerance: float = 12.0) -> list[float]:
        """Every tile must show ITS OWN constant value.

        Use the modal interior pixel rather than the mean: antialiased ROI or
        profile composition can touch an image interior for one frame without
        changing the dominant texture value.  A stale atlas slot changes the
        dominant value and still fails this assertion.
        """

        self.prepare_image_layer_pixel_sampling()
        means = self.tile_pixel_modes()
        expected = [255.0 * k / (COUNT - 1) for k in range(COUNT)]
        for k in range(1, COUNT):
            assert means[k] > means[k - 1] + 1.0, (
                f"tile {k} does not show its own content: means[{k - 1}]={means[k - 1]:.1f} "
                f"means[{k}]={means[k]:.1f} (full ramp: {[round(m) for m in means]})"
            )
        worst = max(abs(m - e) for m, e in zip(means, expected, strict=False))
        assert worst <= tolerance, (
            f"tile gray ramp deviates {worst:.1f} > {tolerance} from analytic values "
            f"(colormap/window drift or wrong LOD content): {[round(m) for m in means]}"
        )
        return means
