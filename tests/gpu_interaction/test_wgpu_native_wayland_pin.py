"""Pin the undocumented Qt contract behind wgpu native-Wayland presentation.

The gate-B screen path relies on two facts that Qt does not document:
under the wayland QPA, ``QWidget.winId()`` returns the ``wl_surface*``, and
``QNativeInterface.QWaylandApplication.display()`` returns the in-process
``wl_display*`` that surface belongs to.  Both held on Qt 6.11.1
(2026-07-18, tier-0 evidence).  A Qt upgrade may silently change either —
this ring-4 gate exists to make that change LOUD instead of a mystery crash
in a future wgpu backend.

Runs the committed tier-0 probe in a subprocess (its own Qt + wgpu instance;
a compositor protocol error kills only the child) and asserts the
native-child, top-level, and window-container (the production screen-canvas
shape) cases presented every frame successfully.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("wgpu")

PROBE = (
    Path(__file__).resolve().parents[2] / "experiments" / "wgpu_gate_b" / "probe_native_wayland.py"
)


@pytest.mark.skipif(not os.environ.get("WAYLAND_DISPLAY"), reason="needs a live Wayland session")
def test_qt_winid_is_the_wl_surface_and_presentation_succeeds(tmp_path):
    out = tmp_path / "probe.json"
    env = dict(os.environ, QT_QPA_PLATFORM="wayland")
    proc = subprocess.run(
        [sys.executable, str(PROBE), str(out)],
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        "native-Wayland probe crashed — the Qt winId/wl_display contract "
        f"this ring pins has likely changed:\n{proc.stderr[-2000:]}"
    )
    result = json.loads(out.read_text())
    assert result["qt_platform"] == "wayland"
    for case in ("native_child", "top_level", "window_container"):
        frames = result["cases"][case]["frames_presented"]
        statuses = result["cases"][case]["statuses"]
        assert frames == 30, f"{case}: only {frames}/30 frames presented ({statuses})"
        assert set(statuses) == {"1"}, f"{case}: non-optimal statuses {statuses}"
