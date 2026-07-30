from __future__ import annotations

import json
import os
import sys

import pytest

from arrayscope.operations import environments
from arrayscope.operations.environments import (
    ExecutionEnvironment,
    load_environments,
    resolve_environment,
    upsert_environment,
)


@pytest.fixture(autouse=True)
def _forget_conda_prefixes(monkeypatch):
    """Prefix resolution is cached per process; give each test an empty cache."""

    monkeypatch.setattr(environments, "_CONDA_PREFIX_CACHE", {})


def _isolated_conda_host(tmp_path, monkeypatch):
    """A fake ``conda`` plus a home directory holding no environments."""

    conda = tmp_path / "conda"
    conda.write_text("#!/bin/sh\n")
    conda.chmod(0o755)
    monkeypatch.setattr(
        "arrayscope.operations.environments.shutil.which",
        lambda *_args, **_kwargs: str(conda),
    )
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_ENVS_PATH", raising=False)
    return conda, home


def _fake_conda_environment(home, name: str, *, with_interpreter: bool) -> str:
    prefix = home / "miniconda3" / "envs" / name
    (prefix / "bin").mkdir(parents=True)
    if with_interpreter:
        interpreter = prefix / "bin" / "python"
        interpreter.write_text("#!/bin/sh\n")
        interpreter.chmod(0o755)
    return str(prefix)


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
    _isolated_conda_host(tmp_path, monkeypatch)
    upsert_environment(
        tmp_path,
        ExecutionEnvironment(id="named", name="Named", conda_env="research"),
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


def test_conda_environment_runs_its_own_interpreter_without_paying_conda_run(tmp_path, monkeypatch):
    """A named conda env resolves to ``<prefix>/bin/python``, not ``conda run``.

    ``conda run`` costs a measured ~1.55 s of conda start-up per call, and an
    operation pays it per characterization probe as well as per apply.
    """

    _conda, home = _isolated_conda_host(tmp_path, monkeypatch)
    prefix = _fake_conda_environment(home, "research", with_interpreter=True)
    upsert_environment(
        tmp_path,
        ExecutionEnvironment(id="named", name="Named", conda_env="research"),
    )

    def _no_subprocess(*_args, **_kwargs):
        raise AssertionError("resolving a conventional conda prefix must not spawn conda")

    monkeypatch.setattr("arrayscope.operations.environments.subprocess.run", _no_subprocess)

    resolved, reason = resolve_environment(tmp_path, "named")

    assert reason is None
    assert resolved is not None
    assert resolved.command_prefix == ()
    assert resolved.interpreter == os.path.join(prefix, "bin", "python")
    assert resolved.env["PATH"].split(os.pathsep)[0] == os.path.join(prefix, "bin")
    assert resolved.env["CONDA_PREFIX"] == prefix
    assert resolved.env["CONDA_DEFAULT_ENV"] == "research"


def test_conda_environment_without_an_interpreter_keeps_conda_run(tmp_path, monkeypatch):
    conda, home = _isolated_conda_host(tmp_path, monkeypatch)
    prefix = _fake_conda_environment(home, "tools", with_interpreter=False)
    upsert_environment(
        tmp_path,
        ExecutionEnvironment(id="named", name="Named", conda_env="tools"),
    )

    class Completed:
        stdout = json.dumps({"envs": [prefix]})

    monkeypatch.setattr(
        "arrayscope.operations.environments.subprocess.run",
        lambda *_args, **_kwargs: Completed(),
    )

    resolved, reason = resolve_environment(tmp_path, "named")

    assert reason is None
    assert resolved is not None
    assert resolved.command_prefix == (str(conda), "run", "--no-capture-output", "-n", "tools")
    assert resolved.interpreter == "python"


def _python_environment_operation(tmp_path, monkeypatch, *, handoff, body, parameters=None):
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
    return build(None, dict(parameters or {}), {})


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
