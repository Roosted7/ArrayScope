"""Concurrent and nested pytest invocations must never share Qt settings."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.filesystem_isolation import install_artifact_directory, isolated_process_root


@pytest.mark.parametrize(
    ("first_uid", "second_uid", "first_pid", "second_pid"),
    [
        pytest.param("first-run", "second-run", 100, 100, id="concurrent-workers"),
        pytest.param("same-run", "same-run", 100, 101, id="inherited-nested-worker"),
        pytest.param(None, None, 100, 100, id="serial-invocations"),
    ],
)
def test_each_invocation_has_its_own_worker_root(
    tmp_path, first_uid, second_uid, first_pid, second_pid
):
    first = isolated_process_root(tmp_path, worker="gw0", run_uid=first_uid, process_id=first_pid)
    second = isolated_process_root(
        tmp_path, worker="gw0", run_uid=second_uid, process_id=second_pid
    )
    assert first != second
    assert first.parent == second.parent == tmp_path
    assert "gw0" in first.name


def test_two_gw0_processes_use_distinct_qsettings_and_children_inherit_them(tmp_path):
    script = """
import json
import os
import subprocess
import sys
import tests.conftest
from pyqtgraph.Qt import QtCore

config = os.environ["XDG_CONFIG_HOME"]
child_config = subprocess.check_output(
    [sys.executable, "-c", "import os; print(os.environ['XDG_CONFIG_HOME'])"], text=True
).strip()
print(json.dumps({
    "config": config,
    "qsettings": QtCore.QSettings("ArrayScope", "IsolationProbe").fileName(),
    "child_config": child_config,
    "artifacts": os.environ["ARRAYSCOPE_ARTIFACT_DIR"],
}))
"""
    env = dict(os.environ)
    env.update(
        TMPDIR=str(tmp_path),
        PYTEST_XDIST_WORKER="gw0",
        PYTEST_XDIST_TESTRUNUID="inherited-outer-run",
        ARRAYSCOPE_ARTIFACT_DIR=str(tmp_path / "outer-artifacts"),
        _ARRAYSCOPE_TEST_ARTIFACT_DIR=str(tmp_path / "outer-artifacts"),
    )
    results = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        results.append(json.loads(result.stdout))
    assert results[0]["config"] != results[1]["config"]
    assert results[0]["artifacts"] != results[1]["artifacts"]
    for result in results:
        assert result["config"] == result["child_config"]
        assert Path(result["qsettings"]).is_relative_to(result["config"])


def test_an_explicit_artifact_destination_survives_worker_setup(tmp_path):
    env = {"ARRAYSCOPE_ARTIFACT_DIR": str(tmp_path / "chosen-output")}
    install_artifact_directory(env, tmp_path / "worker-root")
    assert env["ARRAYSCOPE_ARTIFACT_DIR"] == str(tmp_path / "chosen-output")
    assert "_ARRAYSCOPE_TEST_ARTIFACT_DIR" not in env
