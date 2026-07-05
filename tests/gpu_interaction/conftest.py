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
from time import monotonic, perf_counter

import numpy as np
import pytest


def _display_available() -> bool:
    if os.environ.get("QT_QPA_PLATFORM", "") == "offscreen":
        return False
    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"))


ENABLED = os.environ.get("ARRAYSCOPE_GPU_TESTS", "") == "1" and _display_available()

pytestmark = pytest.mark.gpu_interaction


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


@pytest.fixture()
def montage_window():
    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    from arrayscope.app.launch import _create_window

    app, win = _create_window(synthetic_montage_data(), title="gpu-harness")
    try:
        harness = Harness(app, win)
        harness.pump(0.3)
        vs = win.view_state
        win._set_view_state(vs.with_montage_axis(2, text=":"))
        win.render(reason="gpu-harness-montage")
        assert harness.wait_settled(timeout=20.0), "montage never settled after open"
        yield harness
    finally:
        win.close()
        for _ in range(50):
            app.processEvents()


class Harness:
    def __init__(self, app, win) -> None:
        self.app = app
        self.win = win

    # -- session/lifecycle -------------------------------------------------

    @property
    def session(self):
        return self.win.renderer._montage_session

    @property
    def lifecycle(self):
        return self.session.lifecycle

    def settled(self) -> bool:
        s = self.session
        if s is None:
            return False
        return (
            not s.loading_tiles
            and not len(s.pending_tiles)
            and not s.flush_pending
            and not s.final_commit_pending
            and not s.lifecycle.evaluating_tiles
            and len(s.presented_tiles) >= len(s.plan.tiles)
        )

    def assert_lifecycle_settled(self) -> None:
        s = self.session
        counters = s.lifecycle.counters()
        # ADR 0051 P2 (machine-derived dispatch): the stall watchdog is an
        # assertion now — any fire during a scripted interaction means a
        # state mutation escaped the dispatch construction (a lost wakeup).
        stall_repairs = int(getattr(self.win.renderer, "_montage_stall_repairs", 0) or 0)
        assert stall_repairs == 0, (
            "stall watchdog fired (lost wakeup): "
            f"{stall_repairs}x, last={getattr(self.win.renderer, '_montage_watchdog_last_stall', None)}"
        )
        assert counters["dangling_claims"] == 0, (
            f"leaked claims: {s.lifecycle.dangling_claims()}"
        )
        assert counters["evaluating"] == 0, (
            f"immortal loading tiles: {sorted(s.lifecycle.evaluating_tiles)}"
        )
        active = {int(tile) for tile in s._last_active_tiles}
        parked_active = s.lifecycle.parked_tiles & active
        assert not parked_active, (
            f"parked tiles inside active scope: {sorted(parked_active)}"
        )
        # Semantic-vs-backend agreement (field defect 2026-07-05): the layer's
        # last payload map must present the same LOD the session believes is
        # presented — a levels-only commit that falsely acknowledged level
        # swaps left the GPU on the old level until an unrelated pan.
        layer = getattr(self.win.img_view, "_vispy_gpu_montage_layer", None)
        stats = getattr(layer, "last_stats", None)
        active_levels = [
            int(getattr(getattr(p, "lod", None), "level", 0) or 0)
            for number, p in s.display_tile_payloads.items()
            if int(number) in active
        ]
        if stats is not None and active_levels:
            session_level = max(active_levels)
            layer_level = int(getattr(stats, "lod_level", 0) or 0)
            assert layer_level == session_level, (
                f"backend presents level {layer_level} while the session "
                f"believes level {session_level} is presented (stale-LOD desync)"
            )

    # -- event loop ----------------------------------------------------------

    def pump(self, seconds: float) -> None:
        deadline = monotonic() + seconds
        while monotonic() < deadline:
            self.app.processEvents()

    def wait_settled(self, timeout: float = 15.0) -> bool:
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            self.app.processEvents()
            if self.settled():
                return True
        return self.settled()

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
        s = self.session
        height, width = s.plan.display_shape
        self.win.img_view.getView().setRange(
            xRange=(0, width), yRange=(0, height), padding=0
        )
        self.pump(0.4)

    def screenshot(self) -> np.ndarray:
        from pyqtgraph.Qt import QtGui

        image = (
            self.win.img_view.grab()
            .toImage()
            .convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
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
            to_widget(t.x0 + t.width / 2.0, t.y0 + t.height / 2.0)
            for t in self.session.plan.tiles
        ]

    def tile_means(
        self, shot: np.ndarray | None = None, *, half: int = 5
    ) -> list[float]:
        shot = self.screenshot() if shot is None else shot
        means: list[float] = []
        for x, y in self.tile_centers_px():
            xi, yi = int(round(x)), int(round(y))
            assert 0 <= yi - half and yi + half < shot.shape[0], (
                f"tile center off-widget: ({xi}, {yi})"
            )
            assert 0 <= xi - half and xi + half < shot.shape[1], (
                f"tile center off-widget: ({xi}, {yi})"
            )
            means.append(
                float(
                    shot[yi - half : yi + half + 1, xi - half : xi + half + 1, 0].mean()
                )
            )
        return means

    def assert_tile_identity_ramp(self, *, tolerance: float = 12.0) -> list[float]:
        """Every tile must show ITS OWN constant value: the measured gray
        means must be strictly increasing with the tile number and close to
        the analytic ramp.  Wrong-content tiles (previous atlas occupant,
        stale mip/LOD of another window, wrong source index) violate this."""

        means = self.tile_means()
        expected = [255.0 * k / (COUNT - 1) for k in range(COUNT)]
        for k in range(1, COUNT):
            assert means[k] > means[k - 1] + 1.0, (
                f"tile {k} does not show its own content: means[{k - 1}]={means[k - 1]:.1f} "
                f"means[{k}]={means[k]:.1f} (full ramp: {[round(m) for m in means]})"
            )
        worst = max(abs(m - e) for m, e in zip(means, expected))
        assert worst <= tolerance, (
            f"tile gray ramp deviates {worst:.1f} > {tolerance} from analytic values "
            f"(colormap/window drift or wrong LOD content): {[round(m) for m in means]}"
        )
        return means
