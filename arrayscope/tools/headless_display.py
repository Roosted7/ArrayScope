"""One headless Weston for a whole batch of real-rendering work.

Rings 3-4 and the journey matrix need a REAL compositor and REAL GL: the
offscreen Qt platform cannot see black/stale tiles, stalls, or livelocks
(``docs/testing/README.md`` law #1).  Historically that made those rings
machine-bound — they needed the developer's own logged-in Wayland session,
so CI never ran them and the harness could rot between manual runs.

A headless Weston removes the *attached display* requirement without
weakening the evidence: the compositor's GL renderer and the Wayland client
protocol are the same ones the real session uses. The client renderer is
independent: ArrayScope's WGPU path pins Vulkan, while Weston uses GL only to
composite client surfaces. Measured parity on the
reference laptop (2026-07-21): ``tests/gpu_interaction`` 28/28 here and
28/28 on the real session, and a full-suite run whose failure set is
identical to the real session's.

Two rules this module exists to enforce:

**One compositor per batch.**  Starting a Weston per test would dominate
the run and serialize work that has no reason to serialize.  One compositor
hosts a whole batch, including parallel xdist workers, which all inherit
``WAYLAND_DISPLAY`` from the environment.  Separate *activities* (a test
batch vs. a profiling run) still get separate compositors, so a profiler's
full-output screenshots never photograph another activity's windows.

**No kiosk shell.**  The kiosk shell force-fullscreens every window to the
output size, which silently changes viewport aspect and therefore montage
layout: at 1600x1000 it turned
``test_one_index_boundary_scroll_has_pixels_and_trace_clean[pyqtgraph]``
red purely through geometry.  The desktop shell lets windows take their
natural size, exactly as the real session does.

Environment traps this module pins (both field-proven, 2026-07-21):

1. EGL vendor order.  ``10_nvidia.json`` sorts before ``50_mesa.json``, so
   a headless Weston with no parent compositor picks the NVIDIA GPU while
   the real session uses Intel.  That is the documented *slower* path here
   ("Hybrid GPU: Intel default; NVIDIA offload is slower to first frame"),
   so it would quietly poison every performance bar.  We pin the same EGL
   vendor the real session resolves to.
2. Capture orientation.  Under the NVIDIA EGL path the compositor's screen
   capture comes back y-flipped, which would silently invert every pixel
   oracle.  Pinning Mesa (trap 1) also fixes this; :func:`capture_probe`
   exists so a caller can prove orientation rather than assume it.

Failure is always loud.  If Weston cannot start we raise — we never fall
back to the offscreen platform, because an offscreen run labelled as
compositor evidence is exactly the vacuous oracle law #5 forbids.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

HEADLESS_DISPLAY_ENV = "ARRAYSCOPE_HEADLESS_DISPLAY"
"""Set to the socket name inside a managed headless compositor."""

MESA_EGL_VENDOR = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
EGL_VENDOR_ENV = "__EGL_VENDOR_LIBRARY_FILENAMES"

DEFAULT_OUTPUT_SIZE = (1920, 1200)
STARTUP_TIMEOUT_S = 10.0


def is_headless_display() -> bool:
    """True when we are already running inside a managed headless Weston."""

    return bool(os.environ.get(HEADLESS_DISPLAY_ENV, ""))


@dataclass(frozen=True)
class HeadlessDisplay:
    """A running compositor a batch can attach to."""

    socket_name: str
    log_path: Path
    output_size: tuple[int, int]
    exact_window: bool = False

    def child_environment(self, environ=None) -> dict[str, str]:
        """Environment for a client that should render into this compositor."""

        environ = os.environ if environ is None else environ
        child = dict(environ)
        child["WAYLAND_DISPLAY"] = self.socket_name
        child["QT_QPA_PLATFORM"] = "wayland"
        child[HEADLESS_DISPLAY_ENV] = self.socket_name
        child.pop("DISPLAY", None)  # never let a client silently pick XWayland
        if self.exact_window:
            # Qt's client-side titlebar would offset the window inside the
            # output and break the capture == window identity below.
            child["QT_WAYLAND_DISABLE_WINDOWDECORATION"] = "1"
        _pin_egl_vendor(child)
        return child

    def renderer_description(self) -> str:
        """The device/renderer Weston logged, for evidence records."""

        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "unknown"
        wanted = ("Using rendering device:", "GL renderer:")
        found = [line.split("]", 1)[-1].strip() for line in lines if any(w in line for w in wanted)]
        return "; ".join(found) if found else "unknown"


def _pin_egl_vendor(environ: dict[str, str]) -> None:
    """Pin the EGL vendor so headless picks the real session's GPU (trap 1)."""

    if environ.get(EGL_VENDOR_ENV):
        return  # an explicit caller choice always wins
    if MESA_EGL_VENDOR.exists():
        environ[EGL_VENDOR_ENV] = str(MESA_EGL_VENDOR)


def _require_binaries() -> tuple[str, str]:
    weston = shutil.which("weston")
    screenshooter = shutil.which("weston-screenshooter")
    if weston is None or screenshooter is None:
        missing = "weston" if weston is None else "weston-screenshooter"
        raise RuntimeError(
            f"headless real-rendering ring requires {missing}; "
            "install weston rather than falling back to the offscreen platform"
        )
    return weston, screenshooter


EXACT_WINDOW_CONFIG = """\
# Written by arrayscope.tools.headless_display for exact-window screen
# evidence: with no panel and no window decoration, a window sized to the
# output is placed at (0, 0), so one compositor capture IS the window image.
[shell]
panel-position=none
background-color=0xff000000
animation=none
startup-animation=none
close-animation=none
"""


@contextmanager
def headless_display(
    *,
    output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    log_dir: str | Path | None = None,
    exact_window: bool = False,
):
    """Own one headless Weston for the duration of the block.

    Reuses an already-managed compositor when one is active, so a nested
    invocation joins the batch instead of starting a second compositor.

    ``exact_window`` is for screen-*evidence* activities (the profiler).
    A Wayland client cannot query its own global position, so a compositor
    capture only equals an exact-window image when the window fills the
    output.  Rather than force that with the kiosk shell — which changes
    viewport aspect and therefore montage layout — this removes the two
    things that would offset the window: the desktop-shell panel and Qt's
    client-side decoration.  The caller sizes the output to the window size
    the session asks for, and the capture is then the window, byte for byte.
    Verified on the reference laptop 2026-07-21: window origin (0, 0),
    window size == output size == capture size.
    """

    # Joining a batch is right for ordinary work, but an exact-window request
    # is a screen-EVIDENCE activity: the batch compositor has its own output
    # size and keeps its panel, so reusing it silently offsets the window and
    # breaks the capture == window identity.  Evidence always owns its
    # compositor.  (Caught by tests/gpu_interaction/test_headless_exact_window.)
    active = os.environ.get(HEADLESS_DISPLAY_ENV, "")
    if active and not exact_window:
        runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", "")).resolve()
        yield HeadlessDisplay(active, runtime_dir / f"{active}.log", output_size, exact_window)
        return

    weston, _ = _require_binaries()
    runtime_dir_text = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if not runtime_dir_text:
        raise RuntimeError("headless real-rendering ring requires XDG_RUNTIME_DIR")
    runtime_dir = Path(runtime_dir_text).resolve()

    width, height = (max(1, int(value)) for value in output_size)
    socket_name = f"arrayscope-headless-{os.getpid()}-{uuid4().hex[:10]}"
    socket_path = runtime_dir / socket_name
    lock_path = runtime_dir / f"{socket_name}.lock"
    log_root = Path(log_dir).resolve() if log_dir is not None else runtime_dir
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{socket_name}.log"

    environment = dict(os.environ)
    # A headless compositor must not inherit a parent display, or it renders
    # into the developer's session instead of standing alone.
    environment.pop("WAYLAND_DISPLAY", None)
    environment.pop("DISPLAY", None)
    _pin_egl_vendor(environment)

    config_path = log_root / f"{socket_name}.ini" if exact_window else None
    if config_path is not None:
        config_path.write_text(EXACT_WINDOW_CONFIG, encoding="utf-8")

    command = (
        weston,
        "--backend=headless",
        "--renderer=gl",
        # Natural window sizes: never the kiosk shell (see module docstring).
        "--shell=desktop",
        # Authorizes the screen-capture protocol weston-screenshooter needs.
        "--debug",
        f"--width={width}",
        f"--height={height}",
        f"--socket={socket_name}",
        f"--config={config_path}" if config_path is not None else "--no-config",
        f"--log={log_path}",
    )
    process = subprocess.Popen(command, env=environment)
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if socket_path.is_socket():
                break
            if process.poll() is not None:
                raise RuntimeError(
                    f"headless Weston exited during startup (rc={process.returncode}); "
                    f"log: {log_path}"
                )
            time.sleep(0.05)
        else:
            raise RuntimeError(
                f"headless Weston did not create {socket_path} within "
                f"{STARTUP_TIMEOUT_S:.0f}s; log: {log_path}"
            )
        yield HeadlessDisplay(socket_name, log_path, (width, height), exact_window)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
        socket_path.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)
        if config_path is not None:
            config_path.unlink(missing_ok=True)


def run_in_headless_display(
    child_command: tuple[str, ...],
    *,
    output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    log_dir: str | Path | None = None,
    exact_window: bool = False,
) -> int:
    """Run one command against a private headless compositor."""

    with headless_display(
        output_size=output_size, log_dir=log_dir, exact_window=exact_window
    ) as display:
        completed = subprocess.run(
            tuple(str(part) for part in child_command),
            check=False,
            env=display.child_environment(),
        )
        return int(completed.returncode)


def capture_output(destination: str | Path) -> Path:
    """Capture the compositor output exactly once.

    Under ``exact_window`` the output IS the window, so this is exact-window
    screen evidence.  Otherwise it is the whole output, which may contain
    several windows — callers must not label that as a window capture.

    Pairs with trap 2 in the module docstring: callers that assert on pixels
    should prove orientation with a known asymmetric scene rather than
    trusting the capture path.
    """

    _, screenshooter = _require_binaries()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run((screenshooter,), cwd=destination.parent, check=True, timeout=10.0)
    captures = sorted(destination.parent.glob("wayland-screenshot*.png"))
    if not captures:
        raise RuntimeError("weston-screenshooter produced no image")
    shutil.move(str(captures[-1]), str(destination))
    return destination


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a command inside one private headless Weston (real GL).",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_OUTPUT_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_OUTPUT_SIZE[1])
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--exact-window",
        action="store_true",
        help=(
            "no panel and no decoration, so a window sized to the output is "
            "placed at (0,0) and one capture is exactly the window image"
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = tuple(args.command)
    if command[:1] == ("--",):
        command = command[1:]
    if not command:
        parser.error("a child command is required after --")
    return run_in_headless_display(
        command,
        output_size=(args.width, args.height),
        log_dir=args.log_dir,
        exact_window=args.exact_window,
    )


if __name__ == "__main__":
    raise SystemExit(main())
