"""Free-threaded CPython (PEP 703, ``python3.14t``) launch policy.

On a free-threaded interpreter build ArrayScope runs with the GIL
disabled by default: the kernel/render workers are the exact workloads
free threading exists for.  Because the GIL state is fixed at interpreter
start (``PYTHON_GIL`` must be in the environment before Python boots),
the policy is applied by the CLI entry point via re-launch, mirroring the
display-server policy in :mod:`arrayscope.app.qt_platform`:

``enabled``         Default.  The CLI relaunches itself once as a
                    supervised child with ``PYTHON_GIL=0``.  If the child
                    dies abnormally within ``GRACE_SECONDS`` of spawn —
                    the same early-crash signature the wayland->xcb
                    fallback uses — the CLI persists ``auto_disabled``
                    and retries once with ``PYTHON_GIL=1``.  If that
                    retry *also* dies abnormally fast, the crash is not
                    free-threading-specific and the persisted setting is
                    reverted to ``enabled``.
``force_disabled``  Menu opt-out: the CLI re-execs itself once with
                    ``PYTHON_GIL=1`` (no supervision needed — a GIL run
                    is the safe configuration, not the experiment).
``auto_disabled``   Written by the crash supervisor, never by the user.
                    Behaves like ``force_disabled`` at launch; the menu
                    shows it distinctly so the user can re-enable.

The policy is inert on regular (with-GIL) builds, and an explicit
``PYTHON_GIL`` already present in the environment always wins over the
setting — which is also what makes the relaunched processes recognize
themselves and stop recursing.  Non-CLI embedders are untouched: we
cannot relaunch a host process we do not own, so they get the
interpreter's own default behavior.

This module stays Qt-free so the decision logic is ring-0 testable; only
the two QSettings helpers touch Qt, lazily.
"""

from __future__ import annotations

import os
import sys
import sysconfig
import time
from dataclasses import dataclass
from enum import Enum

from arrayscope.app.qt_platform import _qt_application_exists, _run_forwarding_signals

# Same early-crash window as the display-server fallback, and for the same
# reason: it must cover interpreter + Qt import time plus first-window
# creation, where incompatible-extension aborts happen.  Only ABNORMAL
# exits are ever acted on, so a generous window is safe.
GRACE_SECONDS = 20.0
SUPERVISED_ENV = "ARRAYSCOPE_GIL_SUPERVISED"
SETTINGS_KEY = "python_free_threading"


class FreeThreadingChoice(Enum):
    ENABLED = "enabled"
    FORCE_DISABLED = "force_disabled"
    AUTO_DISABLED = "auto_disabled"


def normalize_free_threading_choice(value) -> FreeThreadingChoice:
    if isinstance(value, FreeThreadingChoice):
        return value
    value = getattr(value, "value", value)
    try:
        return FreeThreadingChoice(str(value))
    except Exception:
        return FreeThreadingChoice.ENABLED


def interpreter_is_free_threaded() -> bool:
    """True on a free-threaded CPython build (3.13t/3.14t), regardless of
    whether the GIL is currently enabled in this process."""
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def gil_currently_enabled() -> bool:
    is_enabled = getattr(sys, "_is_gil_enabled", None)
    return True if is_enabled is None else bool(is_enabled())


@dataclass(frozen=True)
class FreeThreadingDecision:
    python_gil: str | None  # PYTHON_GIL value for the relaunch, or None
    supervise: bool  # CLI should run the crash-supervised child
    reason: str


def resolve_free_threading(
    choice, *, environ=None, free_threaded_build=None, cli: bool = False
) -> FreeThreadingDecision:
    environ = os.environ if environ is None else environ
    choice = normalize_free_threading_choice(choice)
    if free_threaded_build is None:
        free_threaded_build = interpreter_is_free_threaded()
    if not free_threaded_build:
        return FreeThreadingDecision(None, False, "not-a-free-threaded-build")
    existing = environ.get("PYTHON_GIL")
    if existing is not None:
        return FreeThreadingDecision(None, False, f"env-override:{existing}")
    if choice is FreeThreadingChoice.FORCE_DISABLED:
        return FreeThreadingDecision("1", False, "forced-gil")
    if choice is FreeThreadingChoice.AUTO_DISABLED:
        return FreeThreadingDecision("1", False, "auto-disabled-gil")
    if cli and not environ.get(SUPERVISED_ENV):
        return FreeThreadingDecision("0", True, "free-threading-supervised")
    # Supervised child / embedded context: the free-threaded build already
    # defaults to GIL-off; nothing to relaunch for.
    return FreeThreadingDecision(None, False, "free-threading-default")


def read_free_threading_setting() -> FreeThreadingChoice:
    """Read the persisted choice pre-QApplication (lazy Qt import)."""
    try:
        from pyqtgraph.Qt import QtCore

        settings = QtCore.QSettings("ArrayScope", "ArrayScope")
        return normalize_free_threading_choice(
            settings.value(SETTINGS_KEY, FreeThreadingChoice.ENABLED.value)
        )
    except Exception:
        return FreeThreadingChoice.ENABLED


def write_free_threading_setting(choice) -> None:
    """Persist the choice pre-QApplication (supervisor process; best effort).

    sync() before returning: the relaunched child reads the value back
    immediately, from another process.
    """
    try:
        from pyqtgraph.Qt import QtCore

        settings = QtCore.QSettings("ArrayScope", "ArrayScope")
        settings.setValue(SETTINGS_KEY, normalize_free_threading_choice(choice).value)
        settings.sync()
    except Exception:
        pass


def run_gil_supervised_cli(
    argv,
    *,
    environ=None,
    runner=_run_forwarding_signals,
    monotonic=time.monotonic,
    grace_seconds=GRACE_SECONDS,
    log=None,
    persist=write_free_threading_setting,
) -> int:
    """Run the CLI with the GIL off; on an early abnormal exit persist
    ``auto_disabled`` and retry once with the GIL on.

    A clean exit (returncode 0) is never retried, however fast.  The
    GIL-on retry doubles as the discriminator: if it also dies abnormally
    within the grace window, the crash reproduces without free threading,
    so the auto-disable is reverted rather than left blaming the wrong
    suspect.
    """
    environ = os.environ if environ is None else environ
    log = log if log is not None else (lambda msg: print(msg, file=sys.stderr))
    args = [sys.executable, "-m", "arrayscope", *argv]
    env = dict(environ)
    env["PYTHON_GIL"] = "0"
    env[SUPERVISED_ENV] = "1"
    start = monotonic()
    result = runner(args, env=env)
    elapsed = monotonic() - start
    if result.returncode == 0 or elapsed >= grace_seconds:
        return result.returncode
    log(
        f"ArrayScope: free-threaded session ended abnormally "
        f"(rc={result.returncode}) after {elapsed:.2f}s; auto-disabling "
        f"free threading and retrying with the GIL enabled (PYTHON_GIL=1)."
    )
    persist(FreeThreadingChoice.AUTO_DISABLED)
    env["PYTHON_GIL"] = "1"
    start = monotonic()
    result = runner(args, env=env)
    elapsed = monotonic() - start
    if result.returncode != 0 and elapsed < grace_seconds:
        log(
            f"ArrayScope: the GIL-enabled retry also ended abnormally "
            f"(rc={result.returncode}) after {elapsed:.2f}s; the crash is "
            f"not free-threading-specific — re-enabling free threading."
        )
        persist(FreeThreadingChoice.ENABLED)
    return result.returncode


def _exec_with_gil(argv, environ) -> None:
    env = dict(environ)
    env["PYTHON_GIL"] = "1"
    os.execve(sys.executable, [sys.executable, "-m", "arrayscope", *argv], env)


def supervise_free_threading_if_needed(
    argv=None,
    *,
    environ=None,
    choice=None,
    free_threaded_build=None,
    runner=_run_forwarding_signals,
    monotonic=time.monotonic,
    grace_seconds=GRACE_SECONDS,
    log=None,
    persist=write_free_threading_setting,
    execv=_exec_with_gil,
) -> int | None:
    """CLI entry hook, called before the display-server hook.

    Returns an exit code when a supervised child ran the whole session
    (caller should exit with it); returns None when the caller should
    continue in-process (policy inert, disabled path re-exec'd — which
    does not return — or we ARE the supervised child).  Ordering: this
    hook stays OUTERMOST so the wayland->xcb retry inside the child
    resolves display-server crashes before an early exit reaches this
    supervisor and gets blamed on free threading.
    """
    environ = os.environ if environ is None else environ
    if environ.get(SUPERVISED_ENV):
        return None
    if _qt_application_exists():
        # Qt is already up in this process (embedded/interactive use):
        # relaunching or exec'ing the process would be wrong.
        return None
    if free_threaded_build is None:
        free_threaded_build = interpreter_is_free_threaded()
    if not free_threaded_build:
        return None  # don't touch QSettings on regular builds
    if choice is None:
        choice = read_free_threading_setting()
    decision = resolve_free_threading(
        choice, environ=environ, free_threaded_build=free_threaded_build, cli=True
    )
    argv = sys.argv[1:] if argv is None else argv
    if decision.supervise:
        return run_gil_supervised_cli(
            argv,
            environ=environ,
            runner=runner,
            monotonic=monotonic,
            grace_seconds=grace_seconds,
            log=log,
            persist=persist,
        )
    if decision.python_gil == "1":
        execv(argv, environ)  # replaces the process; returns only in tests
    return None
