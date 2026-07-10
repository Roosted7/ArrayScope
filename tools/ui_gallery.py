#!/usr/bin/env python
"""Offscreen UI gallery for ArrayScope.

Renders the real ArrayScope window across a matrix of scenarios (data shapes,
features, edge cases), themes (dark/light/system), and window sizes, and saves
PNG screenshots for visual review. Uses the PyQtGraph rendering backend only
(VisPy does not work offscreen) and the Qt "offscreen" platform plugin, so it
runs headless and never touches the real user settings (a private QSettings
application name is used).

Usage:
    python tools/ui_gallery.py                # render everything
    python tools/ui_gallery.py --list         # list scenarios
    python tools/ui_gallery.py --only montage --themes dark
    python tools/ui_gallery.py --out /tmp/gallery --jobs 8

Output: <out>/<scenario>/<theme>__<label>.png plus <out>/index.html
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "tests" / "artifacts" / "ui_gallery"
THEMES = ("dark", "light")


# --------------------------------------------------------------------------
# Scenario registry
# --------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {}


def scenario(name, *, themes=THEMES, env=None):
    def register(fn):
        SCENARIOS[name] = {"fn": fn, "themes": tuple(themes), "env": dict(env or {})}
        return fn

    return register


# --------------------------------------------------------------------------
# Synthetic data (deterministic, visually structured)
# --------------------------------------------------------------------------


def _phantom2d(n=384):
    import numpy as np

    y, x = np.mgrid[0:n, 0:n].astype(np.float64) / n
    img = 0.35 * x + 0.15 * y
    for cx, cy, s, a in ((0.32, 0.4, 0.05, 1.0), (0.7, 0.3, 0.02, 0.8), (0.55, 0.68, 0.09, 0.6), (0.8, 0.8, 0.01, 1.4)):
        img += a * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * s)))
    rng = np.random.default_rng(7)
    img += rng.normal(scale=0.02, size=img.shape)
    return img


def _volume3d(nx=96, ny=96, nz=40):
    import numpy as np

    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    z = np.linspace(-3, 3, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    vol = np.exp(-((X**2) / 2 + (Y**2) / 3 + (Z**2) / 1.5))
    vol += 0.4 * np.exp(-(((X - 1.2) ** 2 + (Y + 0.8) ** 2 + Z**2) / 0.4))
    return vol


def _kspace(n=256):
    import numpy as np

    return np.fft.fftshift(np.fft.fft2(_phantom2d(n)))


# --------------------------------------------------------------------------
# Child-process capture context
# --------------------------------------------------------------------------


class Ctx:
    def __init__(self, app, out_dir: Path, theme: str):
        self.app = app
        self.out_dir = out_dir
        self.theme = theme
        self.windows = []

    def window(self, data, size=(960, 720), **kwargs):
        from arrayscope.window import ArrayScopeWindow

        win = ArrayScopeWindow(data, **kwargs)
        win.resize(*size)
        win.show()
        self.windows.append(win)
        self.settle()
        return win

    def pump(self, count=8):
        for _ in range(count):
            self.app.processEvents()
            time.sleep(0.005)

    @staticmethod
    def _window_busy(win) -> bool:
        try:
            if win._resource_governor_work_active():
                return True
        except Exception:
            pass
        overlay = getattr(getattr(win, "img_view", None), "_evaluation_overlay", None)
        if overlay is not None and overlay.isVisible():
            return True
        return False

    def settle(self, timeout=15.0, quiet_checks=6):
        deadline = time.monotonic() + timeout
        quiet = 0
        while time.monotonic() < deadline:
            self.app.processEvents()
            busy = any(self._window_busy(win) for win in self.windows)
            if busy:
                quiet = 0
            else:
                quiet += 1
                if quiet >= quiet_checks:
                    return True
            time.sleep(0.02)
        return False

    def shot(self, widget, label):
        # KNOWN ISSUE on the temp redesign branch: the first frame's native
        # tile materialization can be lost, so the "Updating image frame..."
        # overlay never hides and pixels stay at the preview floor. Hide the
        # stale overlay so chrome screenshots stay reviewable.
        for win in self.windows:
            overlay = getattr(getattr(win, "img_view", None), "_evaluation_overlay", None)
            if overlay is not None and overlay.isVisible():
                overlay.hide()
        self.pump(2)
        path = self.out_dir / f"{self.theme}__{label}.png"
        pixmap = widget.grab()
        if pixmap.isNull():
            raise RuntimeError(f"null pixmap for {label}")
        if not pixmap.save(str(path), "PNG"):
            raise RuntimeError(f"failed to save {path}")
        return path

    def resize_shot(self, win, sizes: dict[str, tuple[int, int]]):
        for label, (w, h) in sizes.items():
            win.resize(w, h)
            self.pump(4)
            self.settle()
            self.shot(win, label)

    def close_all(self):
        for win in self.windows:
            try:
                win.close()
            except Exception:
                pass
        self.pump(4)
        self.windows.clear()


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


@scenario("2d_default", themes=("dark", "light", "system"))
def s_2d_default(ctx: Ctx):
    win = ctx.window(_phantom2d(), size=(960, 720))
    ctx.resize_shot(
        win,
        {
            "main_960x720": (960, 720),
            "narrow_520x680": (520, 680),
            "wide_1600x950": (1600, 950),
            "small_420x420": (420, 420),
        },
    )


@scenario("2d_hidpi", themes=("dark",), env={"QT_SCALE_FACTOR": "2"})
def s_2d_hidpi(ctx: Ctx):
    win = ctx.window(_phantom2d(), size=(960, 720))
    ctx.shot(win, "main_960x720_scale2")


@scenario("3d_volume")
def s_3d_volume(ctx: Ctx):
    win = ctx.window(_volume3d(), size=(960, 720))
    ctx.shot(win, "initial")
    chip = win.dimension_strip.chip(2)
    chip.slice_edit.setText("25")
    win._on_slice_text_changed(2, "25")
    ctx.settle()
    ctx.shot(win, "sliced_z25")
    ctx.shot(win.dimension_strip, "dimension_strip")


@scenario("6d_many_dims")
def s_6d(ctx: Ctx):
    import numpy as np

    shape = (2, 3, 4, 5, 6, 7)
    idx = np.indices(shape).sum(axis=0).astype(float)
    data = np.sin(idx) + idx / idx.max()
    win = ctx.window(data, size=(560, 640))
    ctx.shot(win, "narrow_560x640")
    win.resize(1240, 720)
    ctx.pump(6)
    ctx.settle()
    ctx.shot(win, "wide_1240x720")
    ctx.shot(win.dimension_strip, "dimension_strip_wide")


@scenario("complex_channels")
def s_complex(ctx: Ctx):
    win = ctx.window(_kspace(), size=(960, 720))
    combo = win.display_toolbar.channel_combo
    combo.setCurrentIndex(combo.findData("abs"))
    ctx.settle()
    # log scale suits k-space magnitude
    scale = win.display_toolbar.scale_combo
    scale.setCurrentIndex(scale.findData("log"))
    ctx.settle()
    ctx.shot(win, "abs_log")
    combo.setCurrentIndex(combo.findData("angle"))
    ctx.settle()
    ctx.shot(win, "phase")


@scenario("montage")
def s_montage(ctx: Ctx):
    import numpy as np

    vol = _volume3d(72, 72, 12).transpose(2, 0, 1).copy()
    win = ctx.window(np.ascontiguousarray(vol), size=(1100, 800))
    chip = win.dimension_strip.chip(0)
    chip.slice_edit.setText(":")
    win._on_slice_text_changed(0, ":")
    ctx.pump(12)
    ctx.settle(timeout=25.0)
    ctx.shot(win, "montage_axis0")


@scenario("operations_dock")
def s_operations(ctx: Ctx):
    win = ctx.window(_volume3d(), size=(1100, 760))
    win._append_operation("centered_fft", dim=0)
    ctx.settle()
    win._append_operation("mean", dim=2)
    ctx.settle(timeout=25.0)
    ctx.shot(win, "two_ops")
    ctx.shot(win.operation_dock.widget(), "dock_only")
    win.set_operation_enabled(0, False)
    ctx.settle()
    ctx.shot(win.operation_dock.widget(), "dock_op_disabled")


@scenario("profile_live")
def s_profile(ctx: Ctx):
    win = ctx.window(_volume3d(), size=(1000, 780))
    win.widgets["buttons"]["display"]["live_profile"].setChecked(True)
    win.img_view.setProfileMarker(40, 48, visible=True)
    win._on_profile_marker_moved(40, 48)
    win._update_live_profile_from_pending_pos()
    ctx.pump(20)
    ctx.settle()
    ctx.shot(win, "live_profile")
    ctx.shot(win.profile_dock.widget, "profile_dock")


@scenario("roi_inspection")
def s_roi(ctx: Ctx):
    from arrayscope.core.roi import RoiKind

    win = ctx.window(_phantom2d(192), size=(1200, 800))
    win.img_view.createRoi(RoiKind.RECTANGLE, rect=(30, 40, 60, 50))
    win.img_view.createRoi(RoiKind.POLYLINE, points=((10, 10), (90, 25), (150, 140)))
    win.img_view.createRoi(RoiKind.FREEHAND_POLYGON, points=((100, 100), (170, 100), (170, 170), (100, 170)))
    ctx.pump(12)
    ctx.settle(timeout=25.0)
    ctx.shot(win, "rois_overlay")
    win._show_inspection_dock()
    ctx.pump(8)
    ctx.settle(timeout=25.0)
    ctx.shot(win, "with_inspection_dock")
    ctx.shot(win.inspection_dock.widget(), "inspection_dock_only")


@scenario("1d_line")
def s_1d(ctx: Ctx):
    import numpy as np

    x = np.linspace(0, 8 * np.pi, 512)
    data = np.sin(x) * np.exp(-x / 20) + np.random.default_rng(3).normal(scale=0.05, size=x.size)
    win = ctx.window(data, size=(860, 520))
    ctx.shot(win, "line_plot")


@scenario("edge_nan_inf")
def s_nan(ctx: Ctx):
    import numpy as np

    data = _phantom2d(128)
    data[20:50, 20:50] = np.nan
    data[80:90, 80:90] = np.inf
    data[100:110, 30:40] = -np.inf
    win = ctx.window(data, size=(860, 640))
    ctx.shot(win, "nan_inf")


@scenario("edge_constant")
def s_constant(ctx: Ctx):
    import numpy as np

    win = ctx.window(np.zeros((64, 64)), size=(860, 640))
    ctx.shot(win, "all_zeros")


@scenario("edge_tiny")
def s_tiny(ctx: Ctx):
    import numpy as np

    win = ctx.window(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), size=(860, 640))
    ctx.shot(win, "tiny_3x2")


@scenario("edge_extreme_aspect")
def s_aspect(ctx: Ctx):
    import numpy as np

    data = np.tile(np.sin(np.linspace(0, 20, 2048)), (16, 1))
    win = ctx.window(data, size=(1100, 500))
    ctx.shot(win, "wide_16x2048")


@scenario("dialog_diagnostics")
def s_diagnostics(ctx: Ctx):
    win = ctx.window(_volume3d(48, 48, 8), size=(900, 700))
    win.open_diagnostics_dialog()
    dialog = win._diagnostics_dialog
    dialog.resize(520, 560)
    ctx.pump(8)
    ctx.shot(dialog, "diagnostics_dialog")


@scenario("dialog_command_palette")
def s_palette(ctx: Ctx):
    from pyqtgraph.Qt import QtCore

    win = ctx.window(_volume3d(48, 48, 8), size=(900, 700))
    captured = []

    def grab_palette():
        from arrayscope.ui.command_palette import CommandPaletteDialog

        for widget in ctx.app.topLevelWidgets():
            if isinstance(widget, CommandPaletteDialog):
                ctx.pump(4)
                captured.append(ctx.shot(widget, "command_palette"))
                widget.search_edit.setText("fft") if hasattr(widget, "search_edit") else None
                ctx.pump(4)
                captured.append(ctx.shot(widget, "command_palette_filtered"))
                widget.reject()
                return
        raise RuntimeError("command palette dialog not found")

    QtCore.QTimer.singleShot(300, grab_palette)
    win.open_command_palette()
    if not captured:
        raise RuntimeError("palette capture failed")


# --------------------------------------------------------------------------
# Child runner
# --------------------------------------------------------------------------


def run_child(name: str, theme: str, out_root: Path) -> None:
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Running as `python tools/ui_gallery.py` puts tools/ at sys.path[0]; make
    # sure the working-tree package wins over any installed arrayscope.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    app = pg.mkQApp()
    app.setOrganizationName("ArrayScope")
    # Unique per scenario+theme: children run concurrently and must not share
    # a QSettings file (theme/backend would race).
    app.setApplicationName(f"ArrayScopeUIGallery.{name}.{theme}")
    app.setStyle("Fusion")

    settings = QtCore.QSettings()
    settings.clear()
    settings.setValue("theme", theme)
    settings.setValue("image_rendering_backend", "pyqtgraph")
    settings.sync()

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = Ctx(app, out_dir, theme)
    spec = SCENARIOS[name]
    try:
        spec["fn"](ctx)
    finally:
        ctx.close_all()
        settings.clear()
        settings.sync()


# --------------------------------------------------------------------------
# Parent orchestration
# --------------------------------------------------------------------------


def _spawn(name: str, theme: str, out_root: Path, extra_env: dict) -> tuple[str, bool, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYQTGRAPH_QT_LIB": "PySide6",
            "QT_QPA_PLATFORM": "offscreen",
        }
    )
    env.update(extra_env)
    tag = f"{name}:{theme}"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", name, theme, "--out", str(out_root)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT))
    ok = proc.returncode == 0
    log = (proc.stdout + proc.stderr).strip()
    return tag, ok, log


def write_index(out_root: Path) -> Path:
    rows = []
    for scenario_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        images = sorted(scenario_dir.glob("*.png"))
        if not images:
            continue
        cells = "".join(
            f'<figure><img src="{scenario_dir.name}/{img.name}" loading="lazy">'
            f"<figcaption>{html.escape(img.name)}</figcaption></figure>"
            for img in images
        )
        rows.append(f"<section><h2>{html.escape(scenario_dir.name)}</h2><div class='grid'>{cells}</div></section>")
    doc = (
        "<!doctype html><meta charset='utf-8'><title>ArrayScope UI gallery</title>"
        "<style>body{font-family:sans-serif;background:#202124;color:#e8eaed;margin:2rem}"
        ".grid{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0}"
        "img{max-width:520px;height:auto;border:1px solid #5f6368;display:block}"
        "figcaption{font-size:12px;color:#9aa0a6;padding:2px 0 10px}</style>"
        + "".join(rows)
    )
    index = out_root / "index.html"
    index.write_text(doc)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", nargs=2, metavar=("SCENARIO", "THEME"), help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", default=None, help="substring filter on scenario names")
    parser.add_argument("--themes", default=None, help="comma-separated subset of themes")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.child:
        run_child(args.child[0], args.child[1], args.out)
        return 0

    if args.list:
        for name, spec in SCENARIOS.items():
            print(f"{name}  themes={','.join(spec['themes'])}")
        return 0

    theme_filter = set(args.themes.split(",")) if args.themes else None
    jobs = []
    for name, spec in SCENARIOS.items():
        if args.only and args.only not in name:
            continue
        for theme in spec["themes"]:
            if theme_filter and theme not in theme_filter:
                continue
            jobs.append((name, theme, spec["env"]))

    args.out.mkdir(parents=True, exist_ok=True)
    failures = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_spawn, name, theme, args.out, env): (name, theme) for name, theme, env in jobs}
        for future in concurrent.futures.as_completed(futures):
            tag, ok, log = future.result()
            status = "ok" if ok else "FAIL"
            print(f"[{status}] {tag}")
            if not ok:
                failures.append((tag, log))
    index = write_index(args.out)
    print(f"\n{len(jobs) - len(failures)}/{len(jobs)} scenario runs succeeded in {time.monotonic() - started:.0f}s")
    print(f"gallery: {index}")
    for tag, log in failures:
        print(f"\n--- {tag} ---\n{log[-2000:]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
