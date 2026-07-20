"""Ring-0 tests for the Linux/Wayland display-server policy.

arrayscope.app.qt_platform is deliberately Qt-free (only
read_qt_platform_setting touches QSettings, lazily), so everything here
runs without a QApplication.
"""

import sys
from dataclasses import dataclass

import pytest

from arrayscope.app.qt_platform import (
    GRACE_SECONDS,
    SUPERVISED_ENV,
    PlatformDecision,
    QtPlatformChoice,
    apply_qt_platform_env,
    normalize_qt_platform_choice,
    platform_choice_applies,
    resolve_qt_platform,
    run_supervised_cli,
    supervise_cli_if_needed,
    wayland_session_detected,
)

WAYLAND_ENV = {"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0"}


# ---- detection -------------------------------------------------------------


def test_wayland_detected_from_session_type():
    assert wayland_session_detected({"XDG_SESSION_TYPE": "wayland"})
    assert wayland_session_detected({"XDG_SESSION_TYPE": "Wayland"})


def test_wayland_detected_from_wayland_display_alone():
    assert wayland_session_detected({"WAYLAND_DISPLAY": "wayland-1"})


def test_x11_session_not_detected():
    assert not wayland_session_detected({"XDG_SESSION_TYPE": "x11"})
    assert not wayland_session_detected({})


def test_choice_applies_only_on_linux_wayland():
    assert platform_choice_applies(WAYLAND_ENV, "linux")
    assert not platform_choice_applies(WAYLAND_ENV, "darwin")
    assert not platform_choice_applies(WAYLAND_ENV, "win32")
    assert not platform_choice_applies({"XDG_SESSION_TYPE": "x11"}, "linux")


def test_normalize_falls_back_to_auto():
    assert normalize_qt_platform_choice("wayland") is QtPlatformChoice.WAYLAND
    assert normalize_qt_platform_choice("xcb") is QtPlatformChoice.XCB
    assert normalize_qt_platform_choice("nonsense") is QtPlatformChoice.AUTO
    assert normalize_qt_platform_choice(None) is QtPlatformChoice.AUTO


# ---- resolution ------------------------------------------------------------


def test_forced_modes_export_platform():
    d = resolve_qt_platform(QtPlatformChoice.WAYLAND, environ=dict(WAYLAND_ENV), platform="linux")
    assert d == PlatformDecision("wayland", False, "forced-wayland")
    d = resolve_qt_platform(QtPlatformChoice.XCB, environ=dict(WAYLAND_ENV), platform="linux")
    assert d == PlatformDecision("xcb", False, "forced-xcb")


def test_auto_cli_supervises_and_auto_embedded_uses_qt_fallback_list():
    d = resolve_qt_platform(
        QtPlatformChoice.AUTO, environ=dict(WAYLAND_ENV), platform="linux", cli=True
    )
    assert d.supervise
    assert d.qt_qpa_platform == "wayland"
    d = resolve_qt_platform(
        QtPlatformChoice.AUTO, environ=dict(WAYLAND_ENV), platform="linux", cli=False
    )
    assert not d.supervise
    assert d.qt_qpa_platform == "wayland;xcb"


def test_explicit_env_always_wins():
    env = dict(WAYLAND_ENV, QT_QPA_PLATFORM="xcb")
    for choice in QtPlatformChoice:
        d = resolve_qt_platform(choice, environ=env, platform="linux", cli=True)
        assert d.qt_qpa_platform is None
        assert not d.supervise
        assert d.reason == "env-override:xcb"


def test_supervised_child_does_not_re_supervise():
    env = dict(WAYLAND_ENV, QT_QPA_PLATFORM="wayland")
    env[SUPERVISED_ENV] = "1"
    d = resolve_qt_platform(QtPlatformChoice.AUTO, environ=env, platform="linux", cli=True)
    assert not d.supervise  # env-override path: parent already exported


def test_inert_off_linux_or_off_wayland():
    d = resolve_qt_platform(QtPlatformChoice.WAYLAND, environ={}, platform="linux")
    assert d.qt_qpa_platform is None
    d = resolve_qt_platform(QtPlatformChoice.XCB, environ=dict(WAYLAND_ENV), platform="darwin")
    assert d.qt_qpa_platform is None


def test_apply_env_sets_platform_and_xcb_mitshm_guard():
    env = {}
    apply_qt_platform_env(PlatformDecision("xcb", False, "forced-xcb"), env)
    assert env["QT_QPA_PLATFORM"] == "xcb"
    assert env["QT_X11_NO_MITSHM"] == "1"
    env = {}
    apply_qt_platform_env(PlatformDecision("wayland;xcb", False, "auto"), env)
    assert env["QT_QPA_PLATFORM"] == "wayland;xcb"
    assert env["QT_X11_NO_MITSHM"] == "1"
    env = {}
    apply_qt_platform_env(PlatformDecision("wayland", False, "forced-wayland"), env)
    assert env["QT_QPA_PLATFORM"] == "wayland"
    assert "QT_X11_NO_MITSHM" not in env
    env = {"QT_QPA_PLATFORM": "keep"}
    apply_qt_platform_env(PlatformDecision(None, False, "inert"), env)
    assert env["QT_QPA_PLATFORM"] == "keep"


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
    rc = run_supervised_cli(
        list(argv),
        environ=dict(WAYLAND_ENV),
        runner=runner,
        monotonic=clock,
        grace_seconds=GRACE_SECONDS,
        log=logs.append,
    )
    return rc, runner.calls, logs


def test_healthy_wayland_run_is_not_retried():
    rc, calls, logs = _run([(0, 300.0)])
    assert rc == 0
    assert len(calls) == 1
    assert logs == []
    args, env = calls[0]
    assert args[:3] == [sys.executable, "-m", "arrayscope"]
    assert args[3:] == ["data.npy"]
    assert env["QT_QPA_PLATFORM"] == "wayland"
    assert env[SUPERVISED_ENV] == "1"


def test_early_abnormal_exit_relaunches_on_xcb():
    rc, calls, logs = _run([(-6, 0.2), (0, 300.0)])  # SIGABRT then healthy
    assert rc == 0
    assert len(calls) == 2
    assert calls[1][1]["QT_QPA_PLATFORM"] == "xcb"
    assert calls[1][1]["QT_X11_NO_MITSHM"] == "1"
    assert len(logs) == 1
    assert "xcb" in logs[0]


def test_late_crash_is_not_retried():
    rc, calls, _ = _run([(-11, 500.0)])
    assert rc == -11
    assert len(calls) == 1


def test_early_clean_exit_is_not_retried():
    # Fast clean exits are normal CLI behavior (bad file prints and returns).
    rc, calls, _ = _run([(0, 0.05)])
    assert rc == 0
    assert len(calls) == 1


def test_failed_xcb_retry_returns_its_code():
    rc, calls, _ = _run([(-6, 0.1), (3, 0.1)])
    assert rc == 3
    assert len(calls) == 2


# ---- CLI entry hook --------------------------------------------------------


@pytest.fixture
def no_existing_qapp(monkeypatch):
    """The hook is a no-op when a QApplication already exists in-process;
    other tests in the suite may have created one, so pin the guard off."""
    import arrayscope.app.qt_platform as qt_platform

    monkeypatch.setattr(qt_platform, "_qt_application_exists", lambda: False)


def test_supervise_hook_is_noop_for_supervised_child(no_existing_qapp):
    env = dict(WAYLAND_ENV, QT_QPA_PLATFORM="wayland")
    env[SUPERVISED_ENV] = "1"
    assert supervise_cli_if_needed([], environ=env, platform="linux") is None


def test_supervise_hook_is_noop_when_qapplication_exists(monkeypatch):
    import arrayscope.app.qt_platform as qt_platform

    monkeypatch.setattr(qt_platform, "_qt_application_exists", lambda: True)
    env = dict(WAYLAND_ENV)
    rc = qt_platform.supervise_cli_if_needed(
        [], environ=env, platform="linux", choice=QtPlatformChoice.XCB
    )
    assert rc is None
    assert "QT_QPA_PLATFORM" not in env


def test_supervise_hook_applies_forced_choice_in_process(no_existing_qapp):
    env = dict(WAYLAND_ENV)
    rc = supervise_cli_if_needed([], environ=env, platform="linux", choice=QtPlatformChoice.XCB)
    assert rc is None
    assert env["QT_QPA_PLATFORM"] == "xcb"


def test_supervise_hook_runs_supervisor_for_auto(no_existing_qapp):
    clock = FakeClock()
    runner = FakeRunner([(0, 60.0)], clock)
    env = dict(WAYLAND_ENV)
    rc = supervise_cli_if_needed(
        ["x.npy"],
        environ=env,
        platform="linux",
        choice=QtPlatformChoice.AUTO,
        runner=runner,
        monotonic=clock,
        log=lambda _msg: None,
    )
    assert rc == 0
    assert len(runner.calls) == 1
    assert env.get("QT_QPA_PLATFORM") is None  # parent env untouched


def test_supervise_hook_inert_off_wayland(no_existing_qapp):
    env = {"XDG_SESSION_TYPE": "x11"}
    assert supervise_cli_if_needed([], environ=env, platform="linux") is None
    assert "QT_QPA_PLATFORM" not in env


# ---- settings round-trip ---------------------------------------------------


def test_settings_state_round_trips_qt_platform():
    from arrayscope.app.settings_state import settings_from_mapping, settings_to_mapping

    state = settings_from_mapping({"qt_platform": "xcb"})
    assert state.qt_platform is QtPlatformChoice.XCB
    assert settings_to_mapping(state)["qt_platform"] == "xcb"
    assert settings_from_mapping({}).qt_platform is QtPlatformChoice.AUTO
    assert settings_from_mapping({"qt_platform": "junk"}).qt_platform is QtPlatformChoice.AUTO
