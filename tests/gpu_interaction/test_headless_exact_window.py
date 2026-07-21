"""Ring 4 — real compositor pixels out of the headless launcher.

Ring 0 can prove which flags we pass; only a real capture can prove what
they produce.  Two traps live here because both stay green while silently
corrupting every downstream pixel oracle:

* **Orientation.** Under the NVIDIA EGL path a headless capture comes back
  y-flipped.  A symmetric probe cannot see that, so this uses a marker in
  one known corner.
* **Capture == window.** ``exact_window`` claims one compositor capture is
  the window image.  That only holds if nothing offsets the window inside
  the output, which is a placement fact, not a flag.

Pins ``arrayscope.tools.headless_display``; recipe and measurements in
``docs/testing/README.md`` ("Headless real rendering").
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytestmark = pytest.mark.gpu_interaction

WIDTH, HEIGHT = 640, 480
MARKER_W, MARKER_H = 80, 40

# Drawn by a child process so the capture sees a real client surface, not a
# widget living in the test process next to a running Qt application.
_CLIENT = textwrap.dedent(
    """
    import subprocess, sys
    from PySide6 import QtCore, QtGui, QtWidgets

    WIDTH, HEIGHT, MARKER_W, MARKER_H = {width}, {height}, {marker_w}, {marker_h}

    class Block(QtWidgets.QWidget):
        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.fillRect(self.rect(), QtGui.QColor(255, 0, 0))
            # Asymmetric on BOTH axes: top-left only, so a y-flip or an
            # x-offset both show up as a moved marker.
            painter.fillRect(QtCore.QRect(0, 0, MARKER_W, MARKER_H),
                             QtGui.QColor(0, 255, 0))

    app = QtWidgets.QApplication([])
    widget = Block()
    widget.resize(WIDTH, HEIGHT)
    widget.show()

    def capture():
        subprocess.run(("weston-screenshooter",), cwd=sys.argv[1], check=True, timeout=10)
        app.quit()

    QtCore.QTimer.singleShot(2000, capture)
    app.exec()
    """
)


def _capture_marker_scene(tmp_path):
    """Run a marker client in an exact-window headless compositor; return pixels."""

    from PIL import Image

    from arrayscope.tools.headless_display import headless_display

    client = tmp_path / "client.py"
    client.write_text(
        _CLIENT.format(width=WIDTH, height=HEIGHT, marker_w=MARKER_W, marker_h=MARKER_H),
        encoding="utf-8",
    )
    shots = tmp_path / "shots"
    shots.mkdir()

    with headless_display(output_size=(WIDTH, HEIGHT), exact_window=True) as display:
        completed = subprocess.run(
            (sys.executable, str(client), str(shots)),
            check=False,
            env=display.child_environment(),
            timeout=120,
        )
    assert completed.returncode == 0, "marker client failed inside the headless compositor"
    captures = sorted(shots.glob("wayland-screenshot*.png"))
    assert len(captures) == 1, f"expected exactly one capture, got {len(captures)}"
    return np.asarray(Image.open(captures[0]).convert("RGB"))


@pytest.fixture(scope="module")
def marker_scene(tmp_path_factory):
    if shutil.which("weston") is None or shutil.which("weston-screenshooter") is None:
        pytest.skip("headless real-rendering ring requires weston")
    return _capture_marker_scene(tmp_path_factory.mktemp("headless-exact"))


def test_capture_is_exactly_the_window(marker_scene):
    """No panel, no decoration: the window fills the output, so capture == window."""

    assert marker_scene.shape[:2] == (HEIGHT, WIDTH)
    body = (marker_scene[:, :, 0] > 200) & (marker_scene[:, :, 1] < 100)
    marker = (marker_scene[:, :, 1] > 200) & (marker_scene[:, :, 0] < 100)
    covered = int(body.sum() + marker.sum())
    assert covered == WIDTH * HEIGHT, (
        f"window covers {covered} of {WIDTH * HEIGHT} captured pixels; "
        "something (panel, decoration, or placement) offset it inside the output"
    )


def test_capture_is_not_y_flipped(marker_scene):
    """The marker is drawn top-left; a flipped capture reports it bottom-left."""

    marker = (marker_scene[:, :, 1] > 200) & (marker_scene[:, :, 0] < 100)
    rows, columns = np.nonzero(marker)
    assert marker.sum() == MARKER_W * MARKER_H
    assert (columns.min(), columns.max()) == (0, MARKER_W - 1)
    assert (rows.min(), rows.max()) == (0, MARKER_H - 1), (
        f"marker occupies rows {rows.min()}..{rows.max()} but was drawn at the "
        f"top of the window; a y-flipped capture would silently invert every "
        f"pixel oracle while leaving it green"
    )


def test_compositor_ran_on_the_real_sessions_gpu(tmp_path):
    """The EGL pin must hold, or performance bars move to the other GPU."""

    from arrayscope.tools.headless_display import headless_display

    if shutil.which("weston") is None:
        pytest.skip("headless real-rendering ring requires weston")
    with headless_display(output_size=(320, 240), log_dir=tmp_path) as display:
        subprocess.run((sys.executable, "-c", "pass"), check=True, env=display.child_environment())
        description = display.renderer_description()

    assert description != "unknown", "compositor never reported a renderer"
    assert "llvmpipe" not in description.lower(), (
        f"headless fell back to software rendering ({description}); "
        "software-GL runs are not ring-4 evidence"
    )
