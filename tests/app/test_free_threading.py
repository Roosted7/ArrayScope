"""Ring-0 tests for the free-threaded CPython launch policy.

arrayscope.app.free_threading is deliberately Qt-free (only the two
QSettings helpers touch Qt, lazily), so everything here runs without a
QApplication and on any interpreter build: the build flag is injected.
"""

import sys
from dataclasses import dataclass

import pytest

from arrayscope.app.free_threading import (
    GRACE_SECONDS,
    SUPERVISED_ENV,
    FreeThreadingChoice,
    FreeThreadingDecision,
    normalize_free_threading_choice,
    resolve_free_threading,
    run_gil_supervised_cli,
    supervise_free_threading_if_needed,
)


def test_normalize_falls_back_to_enabled():
    assert normalize_free_threading_choice("enabled") is FreeThreadingChoice.ENABLED
    assert (
        normalize_free_threading_choice("force_disabled")
        is FreeThreadingChoice.FORCE_DISABLED
    )
    assert (
        normalize_free_threading_choice("auto_disabled")
        is FreeThreadingChoice.AUTO_DISABLED
    )
    assert normalize_free_threading_choice("nonsense") is FreeThreadingChoice.ENABLED
    assert normalize_free_threading_choice(None) is FreeThreadingChoice.ENABLED


# ---- resolution ------------------------------------------------------------


def test_inert_on_regular_builds():
    for choice in FreeThreadingChoice:
        d = resolve_free_threading(
            choice, environ={}, free_threaded_build=False, cli=True
        )
        assert d == FreeThreadingDecision(None, False, "not-a-free-threaded-build")


def test_explicit_env_always_wins():
    for choice in FreeThreadingChoice:
        d = resolve_free_threading(
            choice, environ={"PYTHON_GIL": "0"}, free_threaded_build=True, cli=True
        )
        assert d.python_gil is None
        assert not d.supervise
        assert d.reason == "env-override:0"


def test_disabled_choices_force_the_gil_on():
    d = resolve_free_threading(
        FreeThreadingChoice.FORCE_DISABLED, environ={}, free_threaded_build=True, cli=True
    )
    assert d == FreeThreadingDecision("1", False, "forced-gil")
    d = resolve_free_threading(
        FreeThreadingChoice.AUTO_DISABLED, environ={}, free_threaded_build=True, cli=True
    )
    assert d == FreeThreadingDecision("1", False, "auto-disabled-gil")


def test_enabled_cli_supervises_and_child_or_embedded_stay_inert():
    d = resolve_free_threading(
        FreeThreadingChoice.ENABLED, environ={}, free_threaded_build=True, cli=True
    )
    assert d == FreeThreadingDecision("0", True, "free-threading-supervised")
    d = resolve_free_threading(
        FreeThreadingChoice.ENABLED,
        environ={SUPERVISED_ENV: "1"},
        free_threaded_build=True,
        cli=True,
    )
    assert not d.supervise
    assert d.python_gil is None
    d = resolve_free_threading(
        FreeThreadingChoice.ENABLED, environ={}, free_threaded_build=True, cli=False
    )
    assert d == FreeThreadingDecision(None, False, "free-threading-default")


# ---- supervisor ------------------------------------------------------------


@dataclass
class FakeResult:
    returncode: int


class FakeRunner:
    """Scripted subprocess.run stand-in; records (args, env) per call."""

    def __init__(self, results, clock):
        self.results = list(results)  # [(returncode, elapsed_seconds), ...]
        self.calls = []
        self.clock = clock

    def __call__(self, args, env=None):
        rc, elapsed = self.results.pop(0)
        self.calls.append((list(args), dict(env)))
        self.clock.now += elapsed
        return FakeResult(rc)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _run(results, argv=("data.npy",)):
    clock = FakeClock()
    runner = FakeRunner(results, clock)
    logs = []
    persisted = []
    rc = run_gil_supervised_cli(
        list(argv),
        environ={"HOME": "/home/user"},
        runner=runner,
        monotonic=clock,
        grace_seconds=GRACE_SECONDS,
        log=logs.append,
        persist=persisted.append,
    )
    return rc, runner.calls, logs, persisted


def test_healthy_free_threaded_run_is_not_retried():
    rc, calls, logs, persisted = _run([(0, 300.0)])
    assert rc == 0
    assert len(calls) == 1
    assert logs == []
    assert persisted == []
    args, env = calls[0]
    assert args[:3] == [sys.executable, "-m", "arrayscope"]
    assert args[3:] == ["data.npy"]
    assert env["PYTHON_GIL"] == "0"
    assert env[SUPERVISED_ENV] == "1"


def test_early_abnormal_exit_auto_disables_and_retries_with_gil():
    rc, calls, logs, persisted = _run([(-6, 0.2), (0, 300.0)])  # SIGABRT then healthy
    assert rc == 0
    assert len(calls) == 2
    assert calls[1][1]["PYTHON_GIL"] == "1"
    assert persisted == [FreeThreadingChoice.AUTO_DISABLED]
    assert len(logs) == 1
    assert "PYTHON_GIL=1" in logs[0]


def test_late_crash_is_not_retried_or_disabled():
    rc, calls, _logs, persisted = _run([(-11, 500.0)])
    assert rc == -11
    assert len(calls) == 1
    assert persisted == []


def test_early_clean_exit_is_not_retried():
    # Fast clean exits are normal CLI behavior (bad file prints and returns).
    rc, calls, _logs, persisted = _run([(0, 0.05)])
    assert rc == 0
    assert len(calls) == 1
    assert persisted == []


def test_crash_reproducing_with_gil_reverts_the_auto_disable():
    # The GIL-on retry is the discriminator: it crashing just as fast means
    # the failure is not free-threading-specific, so the blame is withdrawn.
    rc, calls, logs, persisted = _run([(-6, 0.1), (-6, 0.1)])
    assert rc == -6
    assert len(calls) == 2
    assert persisted == [
        FreeThreadingChoice.AUTO_DISABLED,
        FreeThreadingChoice.ENABLED,
    ]
    assert len(logs) == 2
    assert "re-enabling" in logs[1]


def test_gil_retry_surviving_keeps_the_auto_disable():
    rc, calls, _logs, persisted = _run([(-6, 0.1), (2, 400.0)])
    assert rc == 2
    assert len(calls) == 2
    assert persisted == [FreeThreadingChoice.AUTO_DISABLED]


# ---- CLI entry hook --------------------------------------------------------


@pytest.fixture
def no_existing_qapp(monkeypatch):
    """Other tests in the suite may have created a QApplication; pin the
    embedded-use guard off."""
    import arrayscope.app.free_threading as free_threading

    monkeypatch.setattr(free_threading, "_qt_application_exists", lambda: False)


def test_hook_is_noop_on_regular_builds(no_existing_qapp):
    rc = supervise_free_threading_if_needed(
        [], environ={}, free_threaded_build=False, choice=FreeThreadingChoice.ENABLED
    )
    assert rc is None


def test_hook_is_noop_for_supervised_child(no_existing_qapp):
    rc = supervise_free_threading_if_needed(
        [],
        environ={SUPERVISED_ENV: "1", "PYTHON_GIL": "0"},
        free_threaded_build=True,
        choice=FreeThreadingChoice.ENABLED,
    )
    assert rc is None


def test_hook_is_noop_when_qapplication_exists(monkeypatch):
    import arrayscope.app.free_threading as free_threading

    monkeypatch.setattr(free_threading, "_qt_application_exists", lambda: True)
    rc = free_threading.supervise_free_threading_if_needed(
        [], environ={}, free_threaded_build=True, choice=FreeThreadingChoice.FORCE_DISABLED
    )
    assert rc is None


def test_hook_supervises_enabled_runs(no_existing_qapp):
    clock = FakeClock()
    runner = FakeRunner([(0, 60.0)], clock)
    env = {}
    rc = supervise_free_threading_if_needed(
        ["x.npy"],
        environ=env,
        free_threaded_build=True,
        choice=FreeThreadingChoice.ENABLED,
        runner=runner,
        monotonic=clock,
        log=lambda _msg: None,
        persist=lambda _choice: None,
    )
    assert rc == 0
    assert len(runner.calls) == 1
    assert "PYTHON_GIL" not in env  # parent env untouched


def test_hook_execs_with_gil_for_disabled_choices(no_existing_qapp):
    execs = []
    rc = supervise_free_threading_if_needed(
        ["x.npy"],
        environ={},
        free_threaded_build=True,
        choice=FreeThreadingChoice.FORCE_DISABLED,
        execv=lambda argv, environ: execs.append(list(argv)),
    )
    assert rc is None
    assert execs == [["x.npy"]]


def test_hook_does_not_exec_after_its_own_relaunch(no_existing_qapp):
    # The re-exec'd process sees its own PYTHON_GIL=1 as an env override:
    # that is the recursion stop.
    execs = []
    rc = supervise_free_threading_if_needed(
        ["x.npy"],
        environ={"PYTHON_GIL": "1"},
        free_threaded_build=True,
        choice=FreeThreadingChoice.FORCE_DISABLED,
        execv=lambda argv, environ: execs.append(list(argv)),
    )
    assert rc is None
    assert execs == []


# ---- settings round-trip ---------------------------------------------------


def test_settings_state_round_trips_free_threading():
    from arrayscope.app.settings_state import settings_from_mapping, settings_to_mapping

    state = settings_from_mapping({"python_free_threading": "force_disabled"})
    assert state.python_free_threading is FreeThreadingChoice.FORCE_DISABLED
    assert settings_to_mapping(state)["python_free_threading"] == "force_disabled"
    assert (
        settings_from_mapping({}).python_free_threading is FreeThreadingChoice.ENABLED
    )
    assert (
        settings_from_mapping({"python_free_threading": "junk"}).python_free_threading
        is FreeThreadingChoice.ENABLED
    )
