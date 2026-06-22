import json


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_release_diagnostics_writes_trace_and_preserves_backend_setting(qt_app, tmp_path):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.diagnostics_trace import summarize_diagnostics_trace
    from arrayscope.tools.release_diagnostics import capture_release_diagnostics

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", "vispy")
    settings.sync()

    path = tmp_path / "release-diagnostics.jsonl"
    written = capture_release_diagnostics(path, backend="pyqtgraph")

    assert written == path
    assert settings.value("image_rendering_backend") == "vispy"

    records = _read_jsonl(path)
    assert records[0]["event"] == "start"
    assert records[0]["app_version"] == "0.8.0"
    assert records[0]["config"]["image_rendering_backend_selected"] == "pyqtgraph"
    assert [record["event"] for record in records[1:]] == ["snapshot", "snapshot", "snapshot"]
    assert records[-1]["diagnostics"]["montage"]["session_id"] is not None

    summary = summarize_diagnostics_trace(path)
    assert summary.backend == "pyqtgraph"
    assert summary.snapshot_count == 3


def test_release_diagnostics_rejects_unknown_backend(tmp_path):
    import pytest

    from arrayscope.tools.release_diagnostics import capture_release_diagnostics

    with pytest.raises(ValueError, match="unsupported backend"):
        capture_release_diagnostics(tmp_path / "diagnostics.jsonl", backend="unknown")
