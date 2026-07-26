"""BART runtime-seam tests after the unary operation pack was demoted.

The arithmetic wrappers are gone.  These tests keep the future command runtime's
load-bearing behavior pinned: cfl I/O, cheap availability, exact argv ordering,
concurrent pipe draining, cancellation, timeout, and scratch cleanup.
"""

from __future__ import annotations

import glob
import os
import stat
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from arrayscope.kernel.task import CancellationToken
from arrayscope.operations import environments, library, plugins, registry
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.packs import bart_pack
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_S

PROBE_SHAPE = (4, 5, 6)
_CHATTY_BYTES = 200_000


@pytest.fixture(autouse=True)
def _clean_pack_state():
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()
    yield
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()


@pytest.mark.parametrize("dtype", ["complex64", "float32", "float64", "int16"])
def test_cfl_round_trips_as_complex64(tmp_path, dtype):
    stem = str(tmp_path / "array")
    source = np.arange(120).reshape(PROBE_SHAPE).astype(dtype)
    if np.dtype(dtype).kind == "c":
        source += 1j * np.flip(source, axis=1)

    bart_pack.write_cfl(stem, source)
    result = bart_pack.read_cfl(stem)

    assert result.shape == source.shape
    assert result.dtype == np.dtype(np.complex64)
    np.testing.assert_allclose(result, source.astype(np.complex64))


def test_bart_pack_registers_only_genuinely_bart_shaped_examples():
    specs = bart_pack.pack_specs()

    assert {spec.id for spec in specs} == {"bart:pics", "bart:ecalib", "bart:walsh"}
    pics = next(spec for spec in specs if spec.id == "bart:pics")
    walsh = next(spec for spec in specs if spec.id == "bart:walsh")
    assert [slot.name for slot in pics.input_slots] == ["sensitivities"]
    assert pics.runtime_config["command_template"].startswith("bart pics -S ")
    assert [parameter.name for parameter in walsh.parameters] == ["calibration_size"]
    assert "covariance" in walsh.label.lower()
    assert all(spec.runtime == "command" for spec in specs)
    assert all(spec.runtime_config["handoff"] == "cfl" for spec in specs)
    registered = []
    assert bart_pack.register(registered.append) is True
    assert [spec.id for spec in registered] == [spec.id for spec in specs]


def test_enumeration_never_spawns_bart(monkeypatch):
    def forbidden_popen(*args, **kwargs):
        raise AssertionError(f"enumeration spawned a process: {args!r} {kwargs!r}")

    monkeypatch.setattr("subprocess.Popen", forbidden_popen)
    registry.all_operations()


def test_bart_executable_uses_effective_environment_path(tmp_path, monkeypatch):
    executable = tmp_path / "bart"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("BART_TOOLBOX_PATH", "/deliberately/not/interpreted")

    assert bart_pack.bart_executable() == str(executable)
    assert bart_pack.bart_available() is True

    monkeypatch.setenv("PATH", "")
    assert bart_pack.bart_available() is False


def _make_executable(script) -> str:
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def _write_recording_bart(tmp_path, marker) -> str:
    script = tmp_path / "recording_bart"
    script.write_text(
        f"#!{sys.executable}\n"
        "import shutil\n"
        "import sys\n"
        f"open({str(marker)!r}, 'w').write('\\n'.join(sys.argv[1:-2]))\n"
        "shutil.copyfile(sys.argv[-2] + '.hdr', sys.argv[-1] + '.hdr')\n"
        "shutil.copyfile(sys.argv[-2] + '.cfl', sys.argv[-1] + '.cfl')\n"
    )
    return _make_executable(script)


def _write_recording_pics_bart(tmp_path, marker) -> str:
    script = tmp_path / "recording_pics_bart"
    script.write_text(
        f"#!{sys.executable}\n"
        "import shutil\n"
        "import sys\n"
        f"open({str(marker)!r}, 'w').write('\\n'.join(sys.argv[1:]))\n"
        "primary, sensitivities, output = sys.argv[-3:]\n"
        "assert primary != sensitivities\n"
        "for suffix in ('.hdr', '.cfl'):\n"
        "    shutil.copyfile(primary + suffix, output + suffix)\n"
    )
    return _make_executable(script)


def test_run_bart_preserves_exact_argv_composition(tmp_path):
    marker = tmp_path / "argv"
    executable = _write_recording_bart(tmp_path, marker)
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    result = bart_pack.run_bart(
        ["pics", "-R", "W:7:0:0.01", "-i", "50"],
        source,
        executable=executable,
    )

    assert marker.read_text().splitlines() == ["pics", "-R", "W:7:0:0.01", "-i", "50"]
    np.testing.assert_allclose(result, source.astype(np.complex64))


def test_pics_definition_preserves_scale_and_hands_inputs_in_exact_order(tmp_path, monkeypatch):
    marker = tmp_path / "argv"
    Path(_write_recording_pics_bart(tmp_path, marker)).rename(tmp_path / "bart")
    monkeypatch.setenv("PATH", str(tmp_path))
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    sensitivities = np.full((2, 3, 4), 7 + 2j, dtype=np.complex64)
    pics = next(spec for spec in bart_pack.pack_specs() if spec.id == "bart:pics")

    result = pics.resolve_fn(
        None,
        {"iterations": 17},
        {"sensitivities": sensitivities},
    )(source)

    argv = marker.read_text().splitlines()
    assert argv[:4] == ["pics", "-S", "-i", "17"]
    assert len(argv) == 7
    assert argv[-3] != argv[-2]
    assert os.path.exists(argv[-3] + ".cfl") is False  # scratch was cleaned
    np.testing.assert_allclose(result, source.astype(np.complex64))


def test_pack_uses_named_bart_execution_environment(tmp_path, monkeypatch):
    marker = tmp_path / "argv"
    Path(_write_recording_bart(tmp_path, marker)).rename(tmp_path / "bart")
    monkeypatch.setenv("PATH", "")
    operations_dir = tmp_path / "operations"
    environments.save_environments(
        operations_dir,
        [
            environments.ExecutionEnvironment(
                id="bart",
                name="BART toolbox",
                variables=(("PATH", str(tmp_path)),),
            )
        ],
    )
    monkeypatch.setattr(library, "user_operations_directory", lambda: str(operations_dir))
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    ecalib = next(spec for spec in bart_pack.pack_specs() if spec.id == "bart:ecalib")

    result = ecalib.resolve_fn(None, {"maps": 2})(source)

    assert marker.read_text().splitlines() == ["ecalib", "-m", "2"]
    np.testing.assert_allclose(result, source.astype(np.complex64))


def _write_blocking_bart(tmp_path, marker) -> str:
    script = tmp_path / "blocking_bart"
    script.write_text(f'#!/bin/bash\necho $$ > "{marker}"\nexec sleep 300\n')
    return _make_executable(script)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_cancel_mid_command_kills_child_under_one_second(tmp_path):
    marker = tmp_path / "child_pid"
    executable = _write_blocking_bart(tmp_path, marker)
    token = CancellationToken()
    before = set(glob.glob(str(tmp_path / "arrayscope-bart-*")))
    outcome: dict[str, object] = {}

    def runner():
        try:
            bart_pack.run_bart(
                ["pics"],
                np.ones((4, 4), dtype=np.complex64),
                cancellation_token=token,
                executable=executable,
                temp_dir=str(tmp_path),
            )
        except BaseException as exc:
            outcome["exc"] = exc

    thread = threading.Thread(target=runner)
    thread.start()
    deadline = time.monotonic() + INTERACTION_SETTLE_HARD_LIMIT_S
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert marker.exists(), "fake bart never started"
    child_pid = int(marker.read_text())

    started = time.monotonic()
    token.cancel()
    thread.join(timeout=3.0)
    elapsed = time.monotonic() - started

    assert not thread.is_alive()
    assert elapsed < 1.0
    assert isinstance(outcome.get("exc"), EvaluationCancelled)
    for _ in range(50):
        if not _pid_alive(child_pid):
            break
        time.sleep(0.01)
    assert not _pid_alive(child_pid)
    assert set(glob.glob(str(tmp_path / "arrayscope-bart-*"))) <= before


def test_already_cancelled_command_never_spawns(tmp_path):
    marker = tmp_path / "child_pid"
    executable = _write_blocking_bart(tmp_path, marker)
    token = CancellationToken()
    token.cancel()

    with pytest.raises(EvaluationCancelled):
        bart_pack.run_bart(
            ["pics"],
            np.ones((2, 2), dtype=np.complex64),
            cancellation_token=token,
            executable=executable,
        )
    assert not marker.exists()


def _write_chatty_copy_bart(tmp_path) -> str:
    script = tmp_path / "chatty_copy_bart"
    script.write_text(
        "#!/bin/bash\n"
        'in="${@: -2:1}"\n'
        'out="${@: -1}"\n'
        f"yes stdout-noise | head -c {_CHATTY_BYTES}\n"
        f"yes stderr-noise | head -c {_CHATTY_BYTES} 1>&2\n"
        'cp "$in.hdr" "$out.hdr"\n'
        'cp "$in.cfl" "$out.cfl"\n'
    )
    return _make_executable(script)


def _write_chatty_hang_bart(tmp_path, marker) -> str:
    script = tmp_path / "chatty_hang_bart"
    script.write_text(
        "#!/bin/bash\n"
        f'echo $$ > "{marker}"\n'
        f"yes stdout-noise | head -c {_CHATTY_BYTES}\n"
        f"yes stderr-noise | head -c {_CHATTY_BYTES} 1>&2\n"
        "exec sleep 300\n"
    )
    return _make_executable(script)


def test_chatty_child_is_drained_without_deadlock(tmp_path):
    executable = _write_chatty_copy_bart(tmp_path)
    source = np.arange(120, dtype=np.complex64).reshape(PROBE_SHAPE)
    outcome: dict[str, object] = {}

    def runner():
        try:
            outcome["result"] = bart_pack.run_bart(
                ["pics"], source, executable=executable, timeout=30.0
            )
        except BaseException as exc:
            outcome["exc"] = exc

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join(timeout=INTERACTION_SETTLE_HARD_LIMIT_S)

    assert not thread.is_alive()
    assert "exc" not in outcome
    np.testing.assert_array_equal(outcome["result"], source)


def test_overall_timeout_kills_stuck_child_and_cleans_scratch(tmp_path):
    marker = tmp_path / "child_pid"
    executable = _write_chatty_hang_bart(tmp_path, marker)
    before = set(glob.glob(str(tmp_path / "arrayscope-bart-*")))

    with pytest.raises(RuntimeError, match="timed out"):
        bart_pack.run_bart(
            ["pics"],
            np.ones((4, 4), dtype=np.complex64),
            executable=executable,
            timeout=0.5,
            temp_dir=str(tmp_path),
        )

    child_pid = int(marker.read_text())
    for _ in range(50):
        if not _pid_alive(child_pid):
            break
        time.sleep(0.01)
    assert not _pid_alive(child_pid)
    assert set(glob.glob(str(tmp_path / "arrayscope-bart-*"))) <= before


def test_bart_timeout_env_config(monkeypatch):
    monkeypatch.delenv(bart_pack.BART_TIMEOUT_ENV, raising=False)
    assert bart_pack.bart_timeout() == bart_pack._DEFAULT_TIMEOUT_S
    monkeypatch.setenv(bart_pack.BART_TIMEOUT_ENV, "12.5")
    assert bart_pack.bart_timeout() == 12.5
    monkeypatch.setenv(bart_pack.BART_TIMEOUT_ENV, "0")
    assert bart_pack.bart_timeout() is None
    monkeypatch.setenv(bart_pack.BART_TIMEOUT_ENV, "malformed")
    assert bart_pack.bart_timeout() == bart_pack._DEFAULT_TIMEOUT_S
