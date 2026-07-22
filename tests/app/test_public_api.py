import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_version_uses_canonical_module():
    import arrayscope
    from arrayscope._version import __version__

    assert arrayscope.__version__ == __version__ == "0.8.0"


def test_package_metadata_uses_canonical_version_when_installed():
    from importlib.metadata import PackageNotFoundError, version

    import arrayscope

    try:
        metadata_version = version("ArrayScope")
    except PackageNotFoundError:
        return

    assert metadata_version == arrayscope.__version__


def test_pyproject_uses_dynamic_version_from_canonical_module():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "arrayscope._version.__version__"
    }


def test_changelog_names_arrayscope_v080_release_candidate():
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "## 0.8.0 — ArrayScope v28 release candidate" in changelog
    assert "## Legacy ndslice releases" in changelog
    assert changelog.index("## 0.8.0") < changelog.index("## Legacy ndslice releases")


def test_package_module_is_callable():
    for name in list(sys.modules):
        if name == "arrayscope" or name.startswith("arrayscope."):
            del sys.modules[name]

    import arrayscope as asc

    assert callable(asc)


def test_calling_package_delegates_to_launch(monkeypatch):
    import sys

    for name in list(sys.modules):
        if name == "arrayscope" or name.startswith("arrayscope."):
            del sys.modules[name]

    import arrayscope as asc

    calls = []

    def fake_arrayscope(data, *args, **kwargs):
        calls.append((data, args, kwargs))
        return "window"

    monkeypatch.setattr(asc, "_arrayscope", fake_arrayscope)

    assert asc("data", title="demo") == "window"
    assert calls == [("data", (), {"title": "demo"})]


def test_cli_single_file_opens_async_and_runs_event_loop(monkeypatch, tmp_path):
    from arrayscope import __main__ as cli

    path = tmp_path / "subject.session.npy"
    path.write_bytes(b"placeholder")
    events = []

    monkeypatch.setattr(
        cli,
        "_open_file_async",
        lambda filepath, **kwargs: events.append(("open", filepath.name, kwargs)) or object(),
    )
    monkeypatch.setattr(cli, "_run_cli_event_loop", lambda: events.append(("loop",)))
    monkeypatch.setattr("sys.argv", ["arrayscope", str(path)])

    cli.main()

    assert events == [
        ("open", "subject.session.npy", {"mmap": False, "consume": False, "title": None}),
        ("loop",),
    ]


def test_cli_multi_file_opens_valid_paths_and_survives_errors(monkeypatch, tmp_path):
    from arrayscope import __main__ as cli

    first = tmp_path / "first.npy"
    bad = tmp_path / "bad.npy"
    third = tmp_path / "third.npy"
    for path in (first, bad, third):
        path.write_bytes(b"placeholder")
    events = []

    def fake_open(filepath, **kwargs):
        events.append(("open", filepath.name))
        if filepath == bad:
            raise RuntimeError("broken file")
        return object()

    monkeypatch.setattr(cli, "_open_file_async", fake_open)
    monkeypatch.setattr(cli, "_run_cli_event_loop", lambda: events.append(("loop",)))
    monkeypatch.setattr("sys.argv", ["arrayscope", str(first), str(bad), str(third)])

    cli.main()

    assert events == [
        ("open", "first.npy"),
        ("open", "bad.npy"),
        ("open", "third.npy"),
        ("loop",),
    ]


def test_cli_missing_file_is_reported_and_skipped(monkeypatch, tmp_path, capfd):
    from arrayscope import __main__ as cli

    missing = tmp_path / "nope.npy"
    events = []
    monkeypatch.setattr(cli, "_open_file_async", lambda *a, **k: events.append("open") or object())
    monkeypatch.setattr(cli, "_run_cli_event_loop", lambda: events.append("loop"))
    monkeypatch.setattr("sys.argv", ["arrayscope", str(missing)])

    cli.main()

    assert events == []
    assert "File not found" in capfd.readouterr().out


def test_cli_without_files_opens_launcher(monkeypatch):
    from arrayscope import __main__ as cli

    events = []
    monkeypatch.setattr(cli, "_show_launcher", lambda: events.append("launcher") or object())
    monkeypatch.setattr(cli, "_run_cli_event_loop", lambda: events.append("loop"))
    monkeypatch.setattr("sys.argv", ["arrayscope"])

    cli.main()

    assert events == ["launcher", "loop"]


def test_cli_wrapper_handoff_flags_forwarded(monkeypatch, tmp_path):
    """--mmap/--consume/--title: the language-wrapper invocation contract."""
    from arrayscope import __main__ as cli

    path = tmp_path / "kspace-123.npy"
    path.write_bytes(b"placeholder")
    seen = {}

    def fake_open(filepath, **kwargs):
        seen["filepath"] = filepath
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "_open_file_async", fake_open)
    monkeypatch.setattr(cli, "_run_cli_event_loop", lambda: None)
    monkeypatch.setattr(
        "sys.argv",
        ["arrayscope", "--mmap", "--consume", "--title", "kspace", str(path)],
    )

    cli.main()

    assert seen == {
        "filepath": path,
        "mmap": True,
        "consume": True,
        "title": "kspace",
    }


def test_cli_install_desktop_short_circuits_before_gui(monkeypatch):
    import pytest

    import arrayscope.desktop as desktop
    from arrayscope import __main__ as cli

    events = []

    class FakeReport:
        ok = True
        lines = ("installed",)

    monkeypatch.setattr(
        desktop, "install_desktop_integration", lambda: events.append("install") or FakeReport()
    )
    monkeypatch.setattr(cli, "_run_cli_event_loop", lambda: events.append("loop"))
    monkeypatch.setattr("sys.argv", ["arrayscope", "--install-desktop"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0
    assert events == ["install"]
