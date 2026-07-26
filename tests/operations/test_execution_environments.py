from __future__ import annotations

import json
import sys

from arrayscope.operations.environments import (
    ExecutionEnvironment,
    load_environments,
    resolve_environment,
    upsert_environment,
)


def test_environment_records_round_trip_and_resolve(tmp_path):
    record = upsert_environment(
        tmp_path,
        {
            "id": "recon",
            "name": "Recon tools",
            "interpreter": sys.executable,
            "working_directory": str(tmp_path),
            "variables": {"RECON_MODE": "careful"},
        },
    )

    assert load_environments(tmp_path) == (record,)
    resolved, reason = resolve_environment(tmp_path, "recon")
    assert reason is None
    assert resolved is not None
    assert resolved.interpreter == sys.executable
    assert resolved.cwd == str(tmp_path)
    assert resolved.env["RECON_MODE"] == "careful"


def test_missing_or_vanished_environment_returns_reason(tmp_path):
    resolved, reason = resolve_environment(tmp_path, "gone")
    assert resolved is None
    assert "not configured" in reason

    upsert_environment(
        tmp_path,
        ExecutionEnvironment(
            id="vanished",
            name="Vanished",
            interpreter=str(tmp_path / "missing-python"),
        ),
    )
    resolved, reason = resolve_environment(tmp_path, "vanished")
    assert resolved is None
    assert "not executable" in reason


def test_conda_environment_resolution_is_bounded_and_checks_membership(tmp_path, monkeypatch):
    conda = tmp_path / "conda"
    conda.write_text("#!/bin/sh\n")
    conda.chmod(0o755)
    upsert_environment(
        tmp_path,
        ExecutionEnvironment(id="named", name="Named", conda_env="research"),
    )
    monkeypatch.setattr(
        "arrayscope.operations.environments.shutil.which",
        lambda *_args, **_kwargs: str(conda),
    )

    class Completed:
        stdout = json.dumps({"envs": ["/opt/conda/envs/other"]})

    monkeypatch.setattr(
        "arrayscope.operations.environments.subprocess.run",
        lambda *_args, **_kwargs: Completed(),
    )
    resolved, reason = resolve_environment(tmp_path, "named")
    assert resolved is None
    assert "no longer exists" in reason
