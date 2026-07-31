"""Pin the Qt contract behind wgpu native-X11 presentation (ring 4).

Sibling of ``test_wgpu_native_wayland_pin.py``, for the X arm of the screen
present path.  The Wayland pin exists because Qt documents neither half of
its recipe; the X11 contract is the opposite — ``winId()`` being the X11
Window XID and ``QX11Application.display()`` being the connection it lives
on are both long-standing documented Qt behaviour — so this gate is NOT
mainly guarding against Qt drift.  It guards the pairing: wgpu-py selecting
``WGPUSurfaceSourceXlibWindow`` for ``platform: "x11"``, and that source
accepting Qt's display/XID pair.  Either side can move under a dependency
bump, and the symptom would otherwise be a mystery crash inside a future
backend rather than a named red here.

Runs the committed tier-0 probe in a subprocess (its own Qt + wgpu instance)
and asserts the native-child, top-level, and window-container (the
production screen-canvas shape) cases presented every frame successfully.

Present modes are recorded, not asserted: Mailbox availability is a driver
and WSI property rather than a Qt contract, and the canvas already downgrades
to Fifo loudly when it is missing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("wgpu")

PROBE = Path(__file__).resolve().parents[2] / "experiments" / "wgpu_gate_b" / "probe_native_x11.py"


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a live X11 display")
def test_qt_winid_is_the_x11_window_and_presentation_succeeds(tmp_path):
    out = tmp_path / "probe.json"
    env = dict(os.environ, QT_QPA_PLATFORM="xcb")
    # A Wayland socket in the environment would let Qt's platform plugin be
    # argued about; the probe asserts its own QPA, but leaving the socket set
    # invites a future Qt to prefer it and turn this into a Wayland run.
    env.pop("WAYLAND_DISPLAY", None)
    proc = subprocess.run(
        [sys.executable, str(PROBE), str(out)],
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        "native-X11 probe crashed — the Qt winId/Display contract or wgpu's "
        f"Xlib surface source has likely changed:\n{proc.stderr[-2000:]}"
    )
    result = json.loads(out.read_text())
    assert result["qt_platform"] == "xcb"
    for case in ("native_child", "top_level", "window_container"):
        frames = result["cases"][case]["frames_presented"]
        statuses = result["cases"][case]["statuses"]
        assert frames == 30, f"{case}: only {frames}/30 frames presented ({statuses})"
        assert set(statuses) == {"1"}, f"{case}: non-optimal statuses {statuses}"
