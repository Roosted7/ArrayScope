"""Qt display-server platform selection (Linux Wayland sessions).

ArrayScope decides its Qt platform DELIBERATELY, once, before QApplication
creation, instead of inheriting whatever the environment or a library
import leaves behind.  Motivation (wgpu renderer gate B, 2026-07-18):

- On a Wayland session Qt can run natively (``wayland``) or through
  XWayland (``xcb``); the renderer experiments proved both paths and each
  has real failure modes the other does not (compositor protocol errors on
  wayland; MIT-SHM / scaling quirks on xcb).
- Some libraries hijack the choice: rendercanvas force-sets
  ``QT_QPA_PLATFORM=xcb`` at *import time* on Wayland systems.  Applying
  our policy first makes that hijack inert.

The persisted ``qt_platform`` setting has three values:

``auto``     Prefer native Wayland.  The CLI supervises the run and
             relaunches once on ``xcb`` if the process dies abnormally
             within ``GRACE_SECONDS`` of spawn (Wayland-hostile setups
             abort during Qt platform init or the first frames).
             Non-CLI embedders get Qt's own plugin-load fallback list
             ``wayland;xcb`` (no crash supervision — we cannot relaunch a
             host process we do not own).
``wayland``  Force native Wayland.
``xcb``      Force X11/XWayland.

The policy is inert unless running on Linux inside a Wayland session, and
an explicit ``QT_QPA_PLATFORM`` already present in the environment always
wins over the setting (the supervised child recognizes itself the same
way).  This module stays Qt-free so the decision logic is ring-0 testable;
only :func:`read_qt_platform_setting` touches QSettings, lazily.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

# "Early crash" window for the auto fallback, wall-clock from child spawn.
# Must comfortably cover interpreter + Qt import time (2-4 s on the
# reference laptop) plus first-window creation, where Wayland-hostile
# environments abort (plugin load failure, compositor protocol errors).
# Only ABNORMAL exits are ever retried, so a generous window is safe:
# clean fast exits (bad CLI args, missing files) never re-run.
GRACE_SECONDS = 20.0
SUPERVISED_ENV = "ARRAYSCOPE_QPA_SUPERVISED"
SETTINGS_KEY = "qt_platform"


class QtPlatformChoice(Enum):
    AUTO = "auto"
    WAYLAND = "wayland"
    XCB = "xcb"


def normalize_qt_platform_choice(value) -> QtPlatformChoice:
    if isinstance(value, QtPlatformChoice):
        return value
    value = getattr(value, "value", value)
    try:
        return QtPlatformChoice(str(value))
    except Exception:
        return QtPlatformChoice.AUTO


def wayland_session_detected(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    if "wayland" in str(environ.get("XDG_SESSION_TYPE", "")).lower():
        return True
    return bool(environ.get("WAYLAND_DISPLAY"))


def platform_choice_applies(environ=None, platform=None) -> bool:
    """The setting is meaningful only on Linux inside a Wayland session."""
    platform = sys.platform if platform is None else platform
    return platform.startswith("linux") and wayland_session_detected(environ)


@dataclass(frozen=True)
class PlatformDecision:
    qt_qpa_platform: str | None  # value to export, or None to leave alone
    supervise: bool  # CLI should run the supervised wayland->xcb retry
    reason: str


def resolve_qt_platform(
    choice, *, environ=None, platform=None, cli: bool = False
) -> PlatformDecision:
    environ = os.environ if environ is None else environ
    choice = normalize_qt_platform_choice(choice)
    if not platform_choice_applies(environ, platform):
        return PlatformDecision(None, False, "not-a-linux-wayland-session")
    existing = environ.get("QT_QPA_PLATFORM")
    if existing:
        return PlatformDecision(None, False, f"env-override:{existing}")
    if choice is QtPlatformChoice.XCB:
        return PlatformDecision("xcb", False, "forced-xcb")
    if choice is QtPlatformChoice.WAYLAND:
        return PlatformDecision("wayland", False, "forced-wayland")
    if cli and not environ.get(SUPERVISED_ENV):
        return PlatformDecision("wayland", True, "auto-supervised")
    # Embedded/API context: Qt's own list handles plugin-load failure
    # (but not post-startup crashes; only the CLI can relaunch).
    return PlatformDecision("wayland;xcb", False, "auto-qt-fallback-list")


def apply_qt_platform_env(decision: PlatformDecision, environ=None) -> None:
    environ = os.environ if environ is None else environ
    if decision.qt_qpa_platform:
        environ["QT_QPA_PLATFORM"] = decision.qt_qpa_platform
        if "xcb" in decision.qt_qpa_platform:
            # MIT-SHM failures on XWayland; harmless otherwise.  Mirrors
            # launch._prepare_qt_environment for paths that bypass it.
            environ.setdefault("QT_X11_NO_MITSHM", "1")


def _qt_application_exists() -> bool:
    """True if a QApplication already exists (without importing Qt anew)."""
    for binding in ("PySide6", "PyQt6", "PySide2", "PyQt5"):
        widgets = sys.modules.get(binding + ".QtWidgets")
        if widgets is not None:
            try:
                if widgets.QApplication.instance() is not None:
                    return True
            except Exception:
                pass
    return False


def read_qt_platform_setting() -> QtPlatformChoice:
    """Read the persisted choice pre-QApplication (lazy Qt import)."""
    try:
        from pyqtgraph.Qt import QtCore

        settings = QtCore.QSettings("ArrayScope", "ArrayScope")
        return normalize_qt_platform_choice(
            settings.value(SETTINGS_KEY, QtPlatformChoice.AUTO.value)
        )
    except Exception:
        return QtPlatformChoice.AUTO


def _run_forwarding_signals(args, env=None):
    """subprocess.run equivalent that keeps the child tied to the parent.

    SIGTERM to the supervisor terminates the child instead of orphaning it;
    Ctrl+C (SIGINT reaches the whole foreground group anyway) waits for the
    child to finish handling it before returning its exit code.
    """
    proc = subprocess.Popen(args, env=env)
    previous = signal.signal(signal.SIGTERM, lambda *_: proc.terminate())
    try:
        while True:
            try:
                return SimpleNamespace(returncode=proc.wait())
            except KeyboardInterrupt:
                continue  # child got the group SIGINT too; wait it out
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_supervised_cli(
    argv,
    *,
    environ=None,
    runner=_run_forwarding_signals,
    monotonic=time.monotonic,
    grace_seconds=GRACE_SECONDS,
    log=None,
) -> int:
    """Run the CLI on wayland; relaunch once on xcb if it dies abnormally fast.

    A clean exit (returncode 0) is never retried, however fast: quick clean
    exits are normal CLI behavior (bad arguments, missing files).  Abnormal
    exits within the grace period are the Wayland-hostility signature
    (platform plugin failure aborts during startup — AFTER several seconds
    of interpreter/Qt imports, hence the generous window; compositor
    protocol errors kill the connection within the first frames).
    """
    environ = os.environ if environ is None else environ
    log = log if log is not None else (lambda msg: print(msg, file=sys.stderr))
    if getattr(sys, "frozen", False):
        # Bundled executable (PyInstaller): the exe IS the CLI; there is no
        # interpreter that understands -m.
        args = [sys.executable, *argv]
    else:
        args = [sys.executable, "-m", "arrayscope", *argv]
    env = dict(environ)
    env["QT_QPA_PLATFORM"] = "wayland"
    env[SUPERVISED_ENV] = "1"
    start = monotonic()
    result = runner(args, env=env)
    elapsed = monotonic() - start
    if result.returncode != 0 and elapsed < grace_seconds:
        log(
            f"ArrayScope: native Wayland session ended abnormally "
            f"(rc={result.returncode}) after {elapsed:.2f}s; "
            f"retrying on X11/XWayland (QT_QPA_PLATFORM=xcb)."
        )
        env["QT_QPA_PLATFORM"] = "xcb"
        env.setdefault("QT_X11_NO_MITSHM", "1")
        result = runner(args, env=env)
    return result.returncode


def supervise_cli_if_needed(
    argv=None,
    *,
    environ=None,
    platform=None,
    choice=None,
    runner=_run_forwarding_signals,
    monotonic=time.monotonic,
    grace_seconds=GRACE_SECONDS,
    log=None,
) -> int | None:
    """CLI entry hook.  Returns an exit code when a supervised child ran the
    whole session (caller should exit with it); returns None when the caller
    should continue in-process (forced/env applied, or policy inert, or we
    ARE the supervised child)."""
    environ = os.environ if environ is None else environ
    if environ.get(SUPERVISED_ENV):
        return None
    if _qt_application_exists():
        # Qt is already up in this process (embedded/interactive use): the
        # platform is fixed and relaunching the process would be wrong.
        return None
    if choice is None:
        if not platform_choice_applies(environ, platform):
            return None  # don't touch QSettings off-Linux/off-Wayland
        choice = read_qt_platform_setting()
    decision = resolve_qt_platform(choice, environ=environ, platform=platform, cli=True)
    if decision.supervise:
        return run_supervised_cli(
            sys.argv[1:] if argv is None else argv,
            environ=environ,
            runner=runner,
            monotonic=monotonic,
            grace_seconds=grace_seconds,
            log=log,
        )
    apply_qt_platform_env(decision, environ)
    return None
