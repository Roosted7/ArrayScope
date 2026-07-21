"""Own a private Weston compositor for reproducible screen-path evidence."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

MANAGED_WESTON_ENV = "ARRAYSCOPE_MANAGED_WESTON"


def is_managed_weston() -> bool:
    return os.environ.get(MANAGED_WESTON_ENV, "") == "1"


def run_in_managed_weston(
    child_command: tuple[str, ...],
    *,
    artifact_dir: str | Path,
    output_size: tuple[int, int],
) -> int:
    """Run one command in a private nested Weston and return its status."""

    weston = shutil.which("weston")
    screenshooter = shutil.which("weston-screenshooter")
    if weston is None or screenshooter is None:
        missing = "weston" if weston is None else "weston-screenshooter"
        raise RuntimeError(f"managed screen evidence requires {missing}")
    runtime_dir_text = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if not runtime_dir_text:
        raise RuntimeError("managed screen evidence requires XDG_RUNTIME_DIR")

    width, height = (max(1, int(value)) for value in output_size)
    output = Path(artifact_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime_dir = Path(runtime_dir_text).resolve()
    socket_name = f"arrayscope-profile-{os.getpid()}-{uuid4().hex[:10]}"
    socket_path = runtime_dir / socket_name
    lock_path = runtime_dir / f"{socket_name}.lock"
    log_path = output / "managed-weston.log"
    environment = dict(os.environ)
    environment[MANAGED_WESTON_ENV] = "1"
    environment["QT_QPA_PLATFORM"] = "wayland"

    with tempfile.TemporaryDirectory(prefix="arrayscope-weston-status-") as status_dir:
        status_path = Path(status_dir) / "child-status"
        command = (
            weston,
            "--backend=wayland",
            "--shell=kiosk",
            f"--width={width}",
            f"--height={height}",
            f"--socket={socket_name}",
            "--no-config",
            "--debug",
            f"--log={log_path}",
            "--",
            sys.executable,
            "-m",
            "arrayscope.tools.managed_weston",
            "--status-file",
            str(status_path),
            "--",
            *tuple(str(part) for part in child_command),
        )
        try:
            completed = subprocess.run(command, check=False, env=environment)
            if status_path.exists():
                return int(status_path.read_text(encoding="utf-8").strip())
            return int(completed.returncode or 1)
        finally:
            socket_path.unlink(missing_ok=True)
            lock_path.unlink(missing_ok=True)


def capture_managed_weston_screenshot(destination: str | Path) -> Path:
    """Capture the private compositor's sole kiosk output exactly once."""

    screenshooter = shutil.which("weston-screenshooter")
    if screenshooter is None:
        raise RuntimeError("managed screen evidence requires weston-screenshooter")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arrayscope-weston-shot-") as capture_dir:
        capture_root = Path(capture_dir)
        subprocess.run(
            (screenshooter,),
            cwd=capture_root,
            check=True,
            timeout=5.0,
        )
        captures = tuple(capture_root.glob("wayland-screenshot*.png"))
        if len(captures) != 1:
            raise RuntimeError(
                f"weston-screenshooter produced {len(captures)} images; expected one kiosk output"
            )
        shutil.move(str(captures[0]), str(destination))
    return destination


def _run_child(command: tuple[str, ...], status_path: Path) -> int:
    returncode = 1
    try:
        returncode = int(subprocess.run(command, check=False).returncode)
        return returncode
    finally:
        try:
            status_path.write_text(f"{returncode}\n", encoding="utf-8")
        finally:
            # Weston deliberately survives its kiosk client. This wrapper is
            # launched directly by Weston, so release that owning compositor
            # after every child outcome, including an abort or crash.
            with contextlib.suppress(ProcessLookupError):
                os.kill(os.getppid(), signal.SIGTERM)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = tuple(args.command)
    if command[:1] == ("--",):
        command = command[1:]
    if not command:
        parser.error("a child command is required after --")
    return _run_child(command, Path(args.status_file))


if __name__ == "__main__":
    raise SystemExit(main())
