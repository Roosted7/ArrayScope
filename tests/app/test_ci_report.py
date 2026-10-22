"""Real failing commands must remain failures with downloadable evidence."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import psutil

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "ci_report.py"


def _run(tmp_path, *arguments, **kwargs):
    return subprocess.run(
        [sys.executable, str(_TOOL), *map(str, arguments)],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", **kwargs},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_real_pytest_failure_keeps_exit_status_junit_log_and_action_summary(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "-p no:junitxml -p no:doctest"\n'
    )
    case = tmp_path / "test_example.py"
    case.write_text("def test_values():\n    assert 2 == 3, '<bad & unsafe>'\n")
    reports = tmp_path / "reports"
    output = reports / "unit"
    completed = _run(
        tmp_path,
        "pytest",
        "--name",
        "Real failing tests",
        "--output",
        output,
        "--timeout",
        "8",
        "--",
        case,
        "-q",
    )
    assert completed.returncode == 1, completed.stderr
    status = json.loads((output / "status.json").read_text())
    assert status["exit_code"] == 1
    assert status["junit_present"]
    assert not status["timed_out"]
    assert "test_values" in (output / "command.log").read_text()

    step_summary = tmp_path / "step-summary.md"
    summary = _run(
        tmp_path,
        "summarize",
        reports,
        "--artifact",
        "unit-results",
        GITHUB_STEP_SUMMARY=str(step_summary),
        GITHUB_REPOSITORY="owner/repo",
        GITHUB_RUN_ID="123",
    )
    assert summary.returncode == 0, summary.stderr
    markdown = step_summary.read_text()
    assert "0 passed, 1 failed, 0 skipped" in markdown
    assert "test_example.py::test_values" in markdown
    assert "&lt;bad &amp; unsafe&gt;" in markdown
    assert "actions/runs/123#artifacts" in markdown
    assert markdown == (reports / "summary.md").read_text()
    with (reports / "timings.csv").open(newline="") as handle:
        timing = list(csv.DictReader(handle))
    assert len(timing) == 1
    assert timing[0]["outcome"] == "failed"
    assert float(timing[0]["seconds"]) >= 0


def test_deadline_kills_command_tree_and_preserves_timeout_evidence(tmp_path):
    pid_path = tmp_path / "child-pid"
    code = (
        "import pathlib, signal, subprocess, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)']); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid)); "
        "print('child started', flush=True); time.sleep(60)"
    )
    output = tmp_path / "reports" / "deadline"
    completed = _run(
        tmp_path,
        "run",
        "--name",
        "Hanging command",
        "--output",
        output,
        "--timeout",
        "1",
        "--",
        sys.executable,
        "-c",
        code,
    )
    assert completed.returncode == 124, completed.stderr
    status = json.loads((output / "status.json").read_text())
    assert status["exit_code"] == 124
    assert status["timed_out"]
    assert "child started" in (output / "command.log").read_text()
    try:
        child = psutil.Process(int(pid_path.read_text()))
    except psutil.NoSuchProcess:
        return
    _, alive = psutil.wait_procs([child], timeout=3)
    for process in alive:
        try:
            status = process.status()
        except psutil.NoSuchProcess:
            continue
        assert status == psutil.STATUS_ZOMBIE


def test_stale_junit_cannot_turn_missing_new_report_into_a_pass(tmp_path):
    output = tmp_path / "reports" / "missing"
    output.mkdir(parents=True)
    (output / "junit.xml").write_text('<testsuite><testcase name="stale"/></testsuite>')
    (output / "status.json").write_text('{"name":"stale"}')
    completed = _run(
        tmp_path,
        "pytest",
        "--name",
        "Missing report",
        "--output",
        output,
        "--timeout",
        "8",
        "--",
        "--help",
        "-o",
        "addopts=",
    )
    assert completed.returncode == 2, completed.stderr
    assert not (output / "junit.xml").exists()
    status = json.loads((output / "status.json").read_text())
    assert status["name"] == "Missing report"
    assert not status["junit_present"]
    assert "without a JUnit report" in status["error"]
    report = _run(tmp_path, "summarize", output.parent)
    assert "Report gaps" in report.stdout
    assert "stale" not in report.stdout
    assert "0 reported cases" in report.stdout


def test_setup_failure_and_malformed_xml_are_reported_without_false_passes(tmp_path):
    reports = tmp_path / "reports"
    (reports / "interrupted").mkdir(parents=True)
    (reports / "interrupted" / "junit.xml").write_text("<partial")
    completed = _run(tmp_path, "summarize", reports)
    assert completed.returncode == 0
    assert "Cannot read" in completed.stdout
    assert "No command status was produced" in completed.stdout
    assert "no test pass is established" in completed.stdout
