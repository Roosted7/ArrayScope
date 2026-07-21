"""Ring 0 — the headless-compositor launcher's policy decisions.

These pin the traps documented in ``arrayscope.tools.headless_display``:
the compositor must not inherit a parent display, must pin the same EGL
vendor the real session resolves to (or headless silently renders on the
other GPU), must never use the kiosk shell (it force-fullscreens windows
and changes montage layout), and must fail loudly rather than degrade to
the offscreen platform.  The real-rendering parity itself is ring 4
(``tests/gpu_interaction``), not here.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest


class _FakeProcess:
    """A Weston that binds its socket on the Nth poll, then stays up.

    It binds a real AF_UNIX socket because the launcher deliberately waits
    for ``is_socket()`` — a plain file appearing in XDG_RUNTIME_DIR must not
    be mistaken for a live compositor.
    """

    def __init__(self, socket_path, *, appear_after=0, exit_code=None):
        self._socket_path = socket_path
        self._appear_after = appear_after
        self._polls = 0
        self._exit_code = exit_code
        self._socket = None
        self.returncode = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        self._polls += 1
        if self._exit_code is not None:
            return self._exit_code
        if self._polls > self._appear_after and self._socket is None:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.bind(str(self._socket_path))
        return None

    def terminate(self):
        self.terminated = True
        if self._socket is not None:
            self._socket.close()

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


@pytest.fixture
def runtime_dir(monkeypatch):
    # Deliberately not tmp_path: AF_UNIX paths cap at 108 bytes and pytest's
    # per-test directories are long enough to overflow it with a socket name.
    runtime = Path(tempfile.mkdtemp(prefix="as-hd-"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("ARRAYSCOPE_HEADLESS_DISPLAY", raising=False)
    monkeypatch.setattr(
        "arrayscope.tools.headless_display.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    yield runtime
    shutil.rmtree(runtime, ignore_errors=True)


def _capture_launch(monkeypatch, runtime, **process_kwargs):
    """Run the launcher against a fake Weston; return its argv and env."""

    launched = {}

    def popen(command, env=None, **_kwargs):
        launched["command"] = tuple(command)
        launched["env"] = dict(env or {})
        socket_arg = next(arg for arg in command if str(arg).startswith("--socket="))
        socket_name = str(socket_arg).split("=", 1)[1]
        return _FakeProcess(runtime / socket_name, **process_kwargs)

    monkeypatch.setattr("arrayscope.tools.headless_display.subprocess.Popen", popen)
    return launched


def test_headless_compositor_stands_alone_on_the_real_sessions_gpu(runtime_dir, monkeypatch):
    """Headless must not inherit a display, and must pin the Mesa EGL vendor.

    Without the pin, EGL resolves ``10_nvidia.json`` before ``50_mesa.json``
    and the whole compositor moves to the documented slower GPU — every
    performance bar would shift for a reason nobody would look for.
    """

    from arrayscope.tools.headless_display import MESA_EGL_VENDOR, headless_display

    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(
        "arrayscope.tools.headless_display.MESA_EGL_VENDOR",
        type(MESA_EGL_VENDOR)("/usr/share/glvnd/egl_vendor.d/50_mesa.json"),
    )
    monkeypatch.setattr(
        "arrayscope.tools.headless_display.Path.exists", lambda self: True, raising=False
    )
    launched = _capture_launch(monkeypatch, runtime_dir)

    with headless_display(output_size=(1280, 800)):
        pass

    assert "WAYLAND_DISPLAY" not in launched["env"]
    assert "DISPLAY" not in launched["env"]
    assert launched["env"]["__EGL_VENDOR_LIBRARY_FILENAMES"].endswith("50_mesa.json")


def test_headless_compositor_never_uses_the_kiosk_shell(runtime_dir, monkeypatch):
    """Kiosk force-fullscreens windows, changing viewport aspect and layout."""

    from arrayscope.tools.headless_display import headless_display

    launched = _capture_launch(monkeypatch, runtime_dir)

    with headless_display(output_size=(1920, 1200)):
        pass

    command = launched["command"]
    assert "--shell=kiosk" not in command
    assert "--shell=desktop" in command
    assert "--backend=headless" in command
    assert "--renderer=gl" in command  # real GL, not pixman: law #1 evidence
    assert "--debug" in command  # authorizes weston-screenshooter
    assert "--width=1920" in command
    assert "--height=1200" in command


def test_exact_window_removes_everything_that_would_offset_the_window(runtime_dir, monkeypatch):
    """Screen evidence needs capture == window, without resorting to kiosk.

    A Wayland client cannot query its own global position, so the identity
    only holds when nothing offsets the window inside the output: no
    desktop-shell panel and no Qt client-side decoration.
    """

    from arrayscope.tools.headless_display import headless_display

    launched = _capture_launch(monkeypatch, runtime_dir)

    with headless_display(output_size=(900, 640), exact_window=True) as display:
        command = launched["command"]
        config_arg = next(arg for arg in command if str(arg).startswith("--config="))
        config = Path(str(config_arg).split("=", 1)[1]).read_text(encoding="utf-8")
        child = display.child_environment()

    assert "--no-config" not in command
    assert "--shell=desktop" in command  # still never kiosk
    assert "panel-position=none" in config
    assert child["QT_WAYLAND_DISABLE_WINDOWDECORATION"] == "1"


def test_default_batches_keep_the_stock_compositor_configuration(runtime_dir, monkeypatch):
    """The proven ring-4 parity run is --no-config; exact-window is opt-in."""

    from arrayscope.tools.headless_display import headless_display

    launched = _capture_launch(monkeypatch, runtime_dir)

    with headless_display() as display:
        child = display.child_environment()

    assert "--no-config" in launched["command"]
    assert "QT_WAYLAND_DISABLE_WINDOWDECORATION" not in child


def test_one_compositor_is_shared_by_a_whole_batch(runtime_dir, monkeypatch):
    """A nested call joins the active batch instead of starting a second Weston."""

    from arrayscope.tools.headless_display import headless_display

    launched = _capture_launch(monkeypatch, runtime_dir)

    with headless_display() as outer:
        monkeypatch.setenv("ARRAYSCOPE_HEADLESS_DISPLAY", outer.socket_name)
        starts_before = launched["command"]
        with headless_display() as inner:
            assert inner.socket_name == outer.socket_name
        assert launched["command"] is starts_before, "a second compositor was started"


def test_exact_window_evidence_never_joins_a_batch(runtime_dir, monkeypatch):
    """A batch compositor has its own size and panel; reusing it moves the window.

    Regression pin: this shipped broken and was caught by real pixels in
    tests/gpu_interaction/test_headless_exact_window.py — the marker landed
    at x=103 because the evidence run had joined the batch's compositor.
    """

    from arrayscope.tools.headless_display import headless_display

    launched = _capture_launch(monkeypatch, runtime_dir)
    monkeypatch.setenv("ARRAYSCOPE_HEADLESS_DISPLAY", "arrayscope-headless-someone-elses")

    with headless_display(output_size=(800, 600), exact_window=True) as display:
        assert display.socket_name != "arrayscope-headless-someone-elses"
        assert "--width=800" in launched["command"]


def test_children_render_into_the_batch_compositor(runtime_dir, monkeypatch):
    """Clients get the socket and native Wayland — never a stray XWayland display."""

    from arrayscope.tools.headless_display import headless_display

    monkeypatch.setenv("DISPLAY", ":0")
    _capture_launch(monkeypatch, runtime_dir)

    with headless_display() as display:
        child = display.child_environment()

    assert child["WAYLAND_DISPLAY"] == display.socket_name
    assert child["QT_QPA_PLATFORM"] == "wayland"
    assert child["ARRAYSCOPE_HEADLESS_DISPLAY"] == display.socket_name
    assert "DISPLAY" not in child


def test_missing_weston_fails_loudly_instead_of_degrading_to_offscreen(runtime_dir, monkeypatch):
    """An offscreen run labelled as compositor evidence is a vacuous oracle."""

    from arrayscope.tools.headless_display import headless_display

    monkeypatch.setattr(
        "arrayscope.tools.headless_display.shutil.which",
        lambda name: None if name == "weston" else f"/usr/bin/{name}",
    )

    with pytest.raises(RuntimeError, match="requires weston"), headless_display():
        pass


def test_compositor_that_dies_during_startup_is_reported(runtime_dir, monkeypatch):
    from arrayscope.tools.headless_display import headless_display

    _capture_launch(monkeypatch, runtime_dir, exit_code=1)

    with pytest.raises(RuntimeError, match="exited during startup"), headless_display():
        pass


def test_private_socket_is_removed_after_the_batch(runtime_dir, monkeypatch):
    from arrayscope.tools.headless_display import headless_display

    _capture_launch(monkeypatch, runtime_dir)

    with headless_display() as display:
        socket_path = runtime_dir / display.socket_name
        socket_path.touch()
        (runtime_dir / f"{display.socket_name}.lock").touch()

    assert not tuple(runtime_dir.glob("arrayscope-headless-*"))


def test_child_exit_status_is_propagated(runtime_dir, monkeypatch):
    from arrayscope.tools.headless_display import run_in_headless_display

    _capture_launch(monkeypatch, runtime_dir)
    monkeypatch.setattr(
        "arrayscope.tools.headless_display.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 3),
    )

    assert run_in_headless_display(("pytest", "tests/gpu_interaction")) == 3
