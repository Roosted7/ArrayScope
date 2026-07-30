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


def _python_environment_operation(tmp_path, monkeypatch, *, handoff, body):
    from arrayscope.operations import library

    monkeypatch.setattr(library, "user_operations_directory", lambda: str(tmp_path))
    upsert_environment(
        tmp_path, ExecutionEnvironment(id="here", name="Here", interpreter=sys.executable)
    )
    source = tmp_path / f"op_{handoff}.py"
    source.write_text(body)
    build = library._make_python_environment_build(
        str(source), "run", environment_id="here", handoff=handoff, timeout=60.0
    )
    return build(None, {}, {})


def test_python_environment_cfl_handoff_hands_the_callable_the_rank_it_was_given(
    tmp_path, monkeypatch
):
    import numpy as np

    # Both cfl readers dropped trailing singletons: the worker's, so the user's
    # callable saw (2, 3) where the document holds (2, 3, 1), and the runtime's,
    # so the result came back a rank lower again.
    bound = _python_environment_operation(
        tmp_path,
        monkeypatch,
        handoff="cfl",
        body="def run(array):\n    assert array.shape == (2, 3, 1), array.shape\n    return array\n",
    )
    data = np.arange(6, dtype=np.float32).reshape(2, 3, 1)

    result = bound(data)

    assert result.shape == (2, 3, 1)
    assert np.array_equal(result, data.astype(np.complex64))


def test_python_environment_operation_round_trips_and_takes_a_cancellation_token(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import numpy as np
    import pytest

    from arrayscope.operations.cancellation import EvaluationCancelled

    bound = _python_environment_operation(
        tmp_path, monkeypatch, handoff="npy", body="def run(array):\n    return array * 3\n"
    )
    data = np.arange(6, dtype=np.float32).reshape(2, 3)

    assert np.array_equal(bound(data), data * 3)

    # The out-of-process runtime is exactly what a token exists for; this
    # builder omitted `cancellation_token=` entirely.
    with pytest.raises(EvaluationCancelled):
        bound(data, cancellation_token=SimpleNamespace(cancelled=True))
