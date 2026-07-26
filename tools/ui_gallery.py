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
# ``python tools/ui_gallery.py`` puts tools/ at sys.path[0]. Ensure this
# working tree wins before importing any ArrayScope helper.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import contextlib

from arrayscope.tools.interaction_budget import (
    INTERACTION_SETTLE_HARD_LIMIT_S,
    bounded_interaction_settle_timeout_s,
)
from arrayscope.tools.presentation_settlement import (
    presentation_is_settled,
    presentation_settlement_diagnostic,
)

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
    for cx, cy, s, a in (
        (0.32, 0.4, 0.05, 1.0),
        (0.7, 0.3, 0.02, 0.8),
        (0.55, 0.68, 0.09, 0.6),
        (0.8, 0.8, 0.01, 1.4),
    ):
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
        # A lost wakeup can have no workers and no overlay while the current
        # frame is still physically incomplete. Use the same strict query as
        # release capture and the live probes.
        if not presentation_is_settled(win):
            return True
        try:
            if win._resource_governor_work_active():
                return True
        except Exception:
            pass
        overlay = getattr(getattr(win, "img_view", None), "_evaluation_overlay", None)
        return bool(overlay is not None and overlay.isVisible())

    @staticmethod
    def _window_settle_diagnostics(win) -> dict[str, object]:
        return {"settlement": presentation_settlement_diagnostic(win)}

    def settle(self, timeout=INTERACTION_SETTLE_HARD_LIMIT_S, quiet_checks=6):
        timeout = bounded_interaction_settle_timeout_s(timeout)
        deadline = time.monotonic() + timeout
        quiet = 0
        while time.monotonic() < deadline:
            self.app.processEvents()
            # QWidget::grab drives the offscreen backing store through the
            # exact paint path used to create the artifact. Without this, an
            # offscreen resize can retain draw-pending until the later shot.
            for win in self.windows:
                win.img_view.grab()
            self.app.processEvents()
            busy = any(self._window_busy(win) for win in self.windows)
            if busy:
                quiet = 0
            else:
                quiet += 1
                if quiet >= quiet_checks:
                    return True
            time.sleep(0.02)
        diagnostics = tuple(self._window_settle_diagnostics(win) for win in self.windows)
        raise TimeoutError(
            f"UI gallery interaction did not settle within {timeout:.3f}s: {diagnostics!r}"
        )

    def shot(self, widget, label):
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
            with contextlib.suppress(Exception):
                win.close()
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
    ctx.settle()
    ctx.shot(win, "montage_axis0")


@scenario("operations_dock")
def s_operations(ctx: Ctx):
    win = ctx.window(_volume3d(), size=(1100, 760))
    win._append_operation("centered_fft", dim=0)
    ctx.settle()
    win._append_operation("mean", dim=2)
    ctx.settle()
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
    win.img_view.createRoi(
        RoiKind.FREEHAND_POLYGON, points=((100, 100), (170, 100), (170, 170), (100, 170))
    )
    ctx.pump(12)
    ctx.settle()
    ctx.shot(win, "rois_overlay")
    win._show_inspection_dock()
    ctx.pump(8)
    ctx.settle()
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


@scenario("colormap_designer")
def s_colormap_designer(ctx: Ctx):
    win = ctx.window(_kspace(128), size=(1000, 700))
    ctx.shot(win, "complex_phase_picker")
    from arrayscope.ui.colormap_designer import ColormapDesignerDialog

    dialog = ColormapDesignerDialog(win)
    dialog.show()
    ctx.pump(6)
    ctx.shot(dialog, "designer_dialog")
    dialog.close()


@scenario("first_run_hints")
def s_first_run_hints(ctx: Ctx):
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings()
    settings.setValue("first_run_hints_dismissed", False)
    settings.sync()
    win = ctx.window(_phantom2d(192), size=(1000, 700))
    ctx.pump(8)
    ctx.shot(win, "hints_overlay")


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


def _shot_menu(ctx: Ctx, menu, label):
    """Grab a (possibly popped-up) QMenu, guarding against a blank/unsized grab.

    ``QMenu.popup`` offscreen usually gives the menu a real geometry, but a lost
    wakeup can leave it unsized; force ``sizeHint`` and retry before trusting the
    pixmap, then assert a plausible height so a blank artifact never ships.
    """

    menu.ensurePolished()
    menu.adjustSize()
    menu.resize(menu.sizeHint())
    pixmap = menu.grab()
    if pixmap.isNull() or pixmap.height() < 40:
        raise RuntimeError(
            f"chip menu {label!r} grabbed blank ({pixmap.width()}x{pixmap.height()})"
        )
    path = ctx.out_dir / f"{ctx.theme}__{label}.png"
    if not pixmap.save(str(path), "PNG"):
        raise RuntimeError(f"failed to save {path}")
    return path


@scenario("operation_add_popup")
def s_operation_add_popup(ctx: Ctx):
    win = ctx.window(_volume3d(), size=(1100, 760))
    # Anchor the stage-1 adder just beneath the ops dock's "Add operation"
    # button, exactly like the production open path.
    button = win.operation_dock.add_button
    anchor = button.mapToGlobal(button.rect().bottomLeft())
    win.open_operation_adder(anchor=anchor)
    popup = win._operation_add_popup
    popup.adjustSize()
    ctx.pump(4)
    ctx.shot(popup, "add_popup_collapsed")
    # Unfold the optional backend groups via the popup's public toggle. The list
    # is a fixed-height scroll area, so bring a revealed backend op into view
    # (selecting it auto-scrolls) to actually show the expanded state.
    popup.set_expanded(True)
    for op_id in ("sigpy:soft_thresh", "sigpy:hard_thresh"):
        if popup.select_operation(op_id):
            break
    popup.adjustSize()
    ctx.pump(4)
    ctx.shot(popup, "add_popup_more")
    # BART examples are intentionally unavailable until the shape-discovery
    # bundle can characterize them. Keep one disabled row in-frame so disabled
    # styling and its tooltip-bearing presence are reviewable.
    unavailable_item = next(
        (
            popup._list.item(row)
            for row in range(popup._list.count())
            if popup._list.item(row).data(0x0101) == "bart:ecalib"
        ),
        None,
    )
    if unavailable_item is None:
        raise RuntimeError("unavailable BART example missing from operation add popup")
    popup._list.scrollToItem(unavailable_item)
    ctx.pump(4)
    ctx.shot(popup, "add_popup_unavailable")
    # Highlight an axis-requiring op so the inline axis combo row appears.
    if not popup.select_operation("crop"):
        raise RuntimeError("crop row not selectable in add popup")
    popup.adjustSize()
    ctx.pump(4)
    ctx.shot(popup, "add_popup_axis_row")


@scenario("operation_params_popup")
def s_operation_params_popup(ctx: Ctx):
    win = ctx.window(_volume3d(), size=(1000, 720))
    anchor = win.mapToGlobal(win.rect().center())
    win.request_operation("crop", 0, anchor=anchor)
    popup = win._operation_params_popup
    if popup is None:
        raise RuntimeError("crop params popup was not opened")
    popup.adjustSize()
    ctx.pump(4)
    ctx.shot(popup, "params_crop")

    from arrayscope.operations.registry import all_operations

    if "sigpy:soft_thresh" in {entry.id for entry in all_operations()}:
        # soft_thresh has requires_axis=False -> pass dim=None.
        win.request_operation("sigpy:soft_thresh", None, anchor=anchor)
        sig_popup = win._operation_params_popup
        if sig_popup is None:
            raise RuntimeError("sigpy params popup was not opened")
        sig_popup.adjustSize()
        ctx.pump(4)
        ctx.shot(sig_popup, "params_sigpy")

    from arrayscope.operations.parameter_forms import build_parameter_form
    from arrayscope.operations.registry import get_operation_entry
    from arrayscope.ui.operation_params_popup import OperationParamsPopup

    pics_entry = get_operation_entry("bart:pics")
    pics_form = build_parameter_form(
        pics_entry,
        shape=win.data.shape,
        slot_options=win._slot_source_options(pics_entry),
    )
    pics_popup = OperationParamsPopup(
        pics_entry,
        pics_form,
        lambda _values, _bindings: None,
        parent=win,
    )
    pics_popup._slot_combos["sensitivities"].setCurrentIndex(1)
    pics_popup.adjustSize()
    pics_popup.show()
    ctx.pump(4)
    ctx.shot(pics_popup, "params_input_slot")


@scenario("operation_manager")
def s_operation_manager(ctx: Ctx):
    import tempfile
    from pathlib import Path

    from arrayscope.app import user_dirs
    from arrayscope.operations import library
    from arrayscope.ui.operation_manager import OperationManagerDialog

    # Isolation: the gallery's private QSettings name already nests the user
    # config (and thus the ops dir) under a per-scenario application name, but
    # to remove ANY doubt about touching the real user's config we monkeypatch
    # both resolvers to a scenario-temp dir BEFORE the window is built.
    ops_dir = Path(tempfile.mkdtemp(prefix="ui-gallery-ops-"))
    user_dirs.user_operations_directory = lambda: ops_dir
    library.user_operations_directory = lambda: str(ops_dir)

    # A broken wrapper JSON so the Problems group renders.
    (ops_dir / "broken.json").write_text('{"format": "arrayscope-operation", "version":')
    # A tiny valid user op imported from a source outside the ops dir.
    src_dir = Path(tempfile.mkdtemp(prefix="ui-gallery-src-"))
    (src_dir / "smooth.py").write_text(
        "def smooth(data, axis, width: int = 3):\n"
        '    """Moving-average smooth along an axis."""\n'
        "    return data\n\n\n"
        "def sharpen(data, amount: float = 0.25):\n"
        '    """Sharpen with an editable strength."""\n'
        "    return data\n"
    )
    user_op_id = library.import_custom_operation(str(src_dir / "smooth.py"), "smooth")
    library.update_user_operation(
        user_op_id,
        parameters=[
            {
                "name": "width",
                "label": "Width",
                "kind": "int",
                "default": 3,
                "minimum": 1,
                "maximum": 15,
                "step": 2,
                "description": "Odd smoothing window.",
            }
        ],
    )
    library.set_operation_hidden("fftshift", True)
    command_id = library.create_empty_user_operation()
    library.update_user_operation(
        command_id,
        label="Command reconstruction",
        description="Editable external reconstruction command.",
        runtime="command",
        command_template=("recon-tool --iterations {iterations} {in} {sensitivities} {out}"),
        handoff="npy",
        timeout_s=120,
        environment="bart",
        template=None,
        parameters=[
            {
                "name": "iterations",
                "label": "Iterations",
                "kind": "int",
                "default": 30,
                "minimum": 1,
                "maximum": 500,
            }
        ],
        input_slots=[
            {
                "name": "sensitivities",
                "label": "Sensitivity maps",
                "description": "Second array handed to the reconstruction command.",
                "accepts": ["dimension-set", "open-document", "saved-array"],
            }
        ],
    )
    library.update_execution_environment(
        id="bart",
        name="BART toolbox",
        working_directory="/data/reconstruction",
        variables={
            "BART_TOOLBOX_PATH": "/opt/bart",
            "OMP_NUM_THREADS": "4",
        },
    )
    library.refresh_user_operations()

    win = ctx.window(_volume3d(), size=(900, 760))
    dialog = OperationManagerDialog(win)
    dialog.show()
    ctx.pump(8)
    if not dialog.select_operation("centered_fft"):
        raise RuntimeError("centered_fft not present in operation manager tree")
    ctx.pump(6)
    ctx.shot(dialog, "system_read_only")

    if not dialog.select_operation(user_op_id):
        raise RuntimeError(f"user op {user_op_id!r} not present in operation manager tree")
    ctx.pump(6)
    ctx.shot(dialog, "user_full_parameters")

    dialog.new_button.click()
    ctx.pump(6)
    ctx.shot(dialog, "new_empty")

    if not dialog.select_operation("centered_fft"):
        raise RuntimeError("centered_fft disappeared before duplicate gallery state")
    dialog.duplicate_button.click()
    ctx.pump(6)
    ctx.shot(dialog, "duplicate_prefilled")

    dialog.new_button.click()
    ctx.pump(4)
    dialog._populate_source_file(str(src_dir / "smooth.py"))
    ctx.pump(6)
    ctx.shot(dialog, "source_callable_picker")

    dialog.advanced_button.setChecked(False)
    if not dialog.select_operation(command_id):
        raise RuntimeError("command definition missing from operation manager")
    ctx.pump(6)
    dialog.resize(860, 900)
    ctx.shot(dialog, "input_slot_editor")
    ctx.shot(dialog, "command_template_advanced_collapsed")

    dialog.advanced_button.setChecked(True)
    environment_index = dialog.environment_editor_combo.findData("bart")
    dialog.environment_editor_combo.setCurrentIndex(environment_index)
    dialog.resize(860, 1060)
    ctx.pump(6)
    ctx.shot(dialog, "command_template_environments_expanded")
    dialog.advanced_button.setChecked(False)
    dialog.resize(780, 720)

    problems_item = None
    for index in range(dialog.tree.topLevelItemCount()):
        group = dialog.tree.topLevelItem(index)
        if group.text(0) == "Problems" and group.childCount():
            problems_item = group.child(0)
            break
    if problems_item is None:
        raise RuntimeError("Problems group not present in operation manager tree")
    dialog.tree.setCurrentItem(problems_item)
    ctx.pump(6)
    ctx.shot(dialog, "problems")
    dialog.close()


@scenario("operation_unresolved_slot")
def s_operation_unresolved_slot(ctx: Ctx):
    from arrayscope.operations import registry
    from arrayscope.operations.input_slots import SLOT_ROI_MASK, OperationInputSlot
    from arrayscope.operations.pipeline import ArrayDocument, OperationStep
    from arrayscope.operations.plugins import PluginOperationSpec

    spec = PluginOperationSpec(
        id="gallery:roi-input",
        label="Apply ROI mask",
        build=lambda _axis, _params, slots: lambda data: data * slots["mask"],
        input_slots=(
            OperationInputSlot(
                "mask",
                "ROI mask",
                "One ROI rasterized in the current image plane.",
                accepts=(SLOT_ROI_MASK,),
            ),
        ),
        group="Gallery",
    )
    registry.register_pack_operation(spec)
    operation = registry.create_operation(spec.id)
    reason = operation.current_unavailable_reason()

    win = ctx.window(_volume3d(), size=(1000, 720))
    win._set_document(
        ArrayDocument(
            win.base_data,
            steps=(
                OperationStep(
                    operation,
                    enabled=False,
                    unavailable_reason=reason,
                ),
            ),
        )
    )
    win.operation_dock.widget().resize(430, 600)
    ctx.pump(6)
    ctx.shot(win.operation_dock.widget(), "unresolved_slot_unavailable")


@scenario("operation_chip_menu")
def s_operation_chip_menu(ctx: Ctx):
    from pyqtgraph.Qt import QtWidgets

    win = ctx.window(_volume3d(), size=(1000, 720))
    dim = 0
    chip = win.dimension_strip.chip(dim)
    button = getattr(chip, "ops_button", chip)
    anchor = button.mapToGlobal(button.rect().bottomLeft())
    menu = win._build_operation_context_menu(dim, anchor)
    # A Qt.Popup menu auto-dismisses the instant it is shown over the active
    # offscreen window, so grab the built menu directly rather than popping it
    # up -- the grab drives the same paint path and _shot_menu sizes it from its
    # sizeHint. The "More…" submenu is reached via findChildren: PySide6's
    # QAction.menu() hands back a throwaway wrapper that is GC-reaped, while
    # findChildren returns the live child object.
    submenu = next(iter(menu.findChildren(QtWidgets.QMenu)), None)
    if submenu is not None:
        _shot_menu(ctx, submenu, "chip_menu_more")
    _shot_menu(ctx, menu, "chip_menu")


# --------------------------------------------------------------------------
# Child runner
# --------------------------------------------------------------------------


def run_child(name: str, theme: str, out_root: Path) -> None:
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
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
    # Keep the one-time hints out of regression shots; a dedicated scenario
    # re-enables them explicitly.
    settings.setValue("first_run_hints_dismissed", True)
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
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        name,
        theme,
        "--out",
        str(out_root),
    ]
    try:
        # Whole-child deadlock guard. Each interaction inside the child still
        # hard-fails independently through the shared five-second owner.
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            check=False,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired as exc:
        return tag, False, f"gallery child process watchdog expired: {exc}"
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
        rows.append(
            f"<section><h2>{html.escape(scenario_dir.name)}</h2><div class='grid'>{cells}</div></section>"
        )
    doc = (
        "<!doctype html><meta charset='utf-8'><title>ArrayScope UI gallery</title>"
        "<style>body{font-family:sans-serif;background:#202124;color:#e8eaed;margin:2rem}"
        ".grid{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0}"
        "img{max-width:520px;height:auto;border:1px solid #5f6368;display:block}"
        "figcaption{font-size:12px;color:#9aa0a6;padding:2px 0 10px}</style>" + "".join(rows)
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
        futures = {
            pool.submit(_spawn, name, theme, args.out, env): (name, theme)
            for name, theme, env in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            tag, ok, log = future.result()
            status = "ok" if ok else "FAIL"
            print(f"[{status}] {tag}")
            if not ok:
                failures.append((tag, log))
    index = write_index(args.out)
    print(
        f"\n{len(jobs) - len(failures)}/{len(jobs)} scenario runs succeeded in {time.monotonic() - started:.0f}s"
    )
    print(f"gallery: {index}")
    for tag, log in failures:
        print(f"\n--- {tag} ---\n{log[-2000:]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
