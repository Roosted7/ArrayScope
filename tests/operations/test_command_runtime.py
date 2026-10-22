from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.command_runtime import (
    build_command,
    read_cfl,
    run_command_template,
    write_cfl,
)
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_S
from tests import seams


def _fake_npy_command(tmp_path: Path) -> Path:
    executable = tmp_path / "fake array tool"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, time\n"
        "import numpy as np\n"
        "args = sys.argv[1:]\n"
        # Written before the sleep, so its absence proves the child never ran
        # -- the difference between refusing to spawn and killing what spawned.
        "started = next((a.split('=', 1)[1] for a in args if a.startswith('--started=')), None)\n"
        "if started:\n"
        "    open(started, 'w').write('1')\n"
        "if '--sleep' in args:\n"
        "    time.sleep(float(args[args.index('--sleep') + 1]))\n"
        "input_path, output_path = args[-2:]\n"
        "np.save(output_path, np.load(input_path, allow_pickle=False) * 2, allow_pickle=False)\n"
        "record = next((a.split('=', 1)[1] for a in args if a.startswith('--record=')), None)\n"
        "if record:\n"
        "    open(record, 'w').write(json.dumps(args))\n"
    )
    executable.chmod(0o755)
    return executable


def test_template_tokenization_preserves_spaces_and_flag_like_parameters(tmp_path):
    executable = _fake_npy_command(tmp_path)
    record = tmp_path / "argv record.json"
    template = f'"{executable}" --record="{record}" --label "{{label}}" {{in}} {{out}}'

    result = run_command_template(
        template,
        np.arange(4, dtype=np.float32),
        parameters={"label": "--not-an-option with spaces"},
    )

    assert np.array_equal(result, np.arange(4, dtype=np.float32) * 2)
    argv = json.loads(record.read_text())
    assert argv[1:3] == ["--label", "--not-an-option with spaces"]
    assert not (tmp_path / "not-an-option").exists()


def test_non_numeric_parameters_stay_one_token_each(tmp_path):
    """A string / enum / path / bool value renders as exactly one argument.

    These kinds are what an external toolbox needs (`mode="reflect"`, a model
    path, a device name), and they are also the values most likely to contain a
    space or to start with a dash. Booleans render lowercase so a wrapper author
    can predict the text without knowing the host language.
    """

    executable = _fake_npy_command(tmp_path)
    record = tmp_path / "argv record.json"
    model = tmp_path / "my models" / "weights.pt"
    template = (
        f'"{executable}" --record="{record}" --mode {{mode}} --model {{model}} '
        "--note {note} --verbose {verbose} {in} {out}"
    )

    run_command_template(
        template,
        np.arange(4, dtype=np.float32),
        parameters={
            "mode": "reflect",
            "model": str(model),
            "note": "-- leading dashes and spaces",
            "verbose": True,
        },
    )

    argv = json.loads(record.read_text())
    assert argv[1:9] == [
        "--mode",
        "reflect",
        "--model",
        str(model),
        "--note",
        "-- leading dashes and spaces",
        "--verbose",
        "true",
    ]


def test_a_file_slot_substitutes_its_own_path_without_a_scratch_copy(tmp_path):
    """A ``file`` slot names a file; it is never staged into the scratch dir.

    Copying a model checkpoint or a database per invocation would be wrong as
    well as wasteful -- the point of the kind is that the operation opens the
    file where it lives.
    """

    executable = _fake_npy_command(tmp_path)
    record = tmp_path / "argv.json"
    model = tmp_path / "weights.bin"
    model.write_bytes(b"not an array")
    template = f'"{executable}" --record="{record}" --model {{model}} {{in}} {{out}}'

    run_command_template(
        template,
        np.arange(4, dtype=np.float32),
        inputs={"model": str(model)},
    )

    argv = json.loads(record.read_text())
    assert argv[1:3] == ["--model", str(model)]
    # The scratch directory is gone by now, so the path the child saw can only
    # have been the original one.
    assert model.read_bytes() == b"not an array"


def test_boolean_parameters_render_lowercase(tmp_path):
    command = build_command(
        "tool --flag {flag} {in} {out}",
        {"flag": False, "in": "input", "out": "output"},
    )
    assert command == ["tool", "--flag", "false", "input", "output"]


def test_template_never_interprets_shell_syntax_without_opt_in(tmp_path):
    marker = tmp_path / "shell-was-run"
    command = build_command(
        "tool {value} {in} {out}",
        {"value": f"; touch {marker}", "in": "input", "out": "output"},
    )

    assert command == ["tool", f"; touch {marker}", "input", "output"]
    assert not marker.exists()


def test_command_timeout_and_cancellation_kill_the_child(tmp_path):
    executable = _fake_npy_command(tmp_path)
    template = f'"{executable}" --sleep 5 {{in}} {{out}}'
    with pytest.raises(RuntimeError, match="timed out"):
        run_command_template(
            template,
            np.ones(2),
            timeout=0.05,
            poll_interval=0.005,
            grace=0.02,
        )

    token = SimpleNamespace(cancelled=False)
    outcome = {}

    def run():
        try:
            run_command_template(
                template,
                np.ones(2),
                cancellation_token=token,
                timeout=None,
                poll_interval=0.005,
                grace=0.02,
            )
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.05)
    token.cancelled = True
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), EvaluationCancelled)


def test_command_failure_keeps_a_bounded_stderr_tail(tmp_path):
    stderr_budget = 64 * 1024
    script = tmp_path / "noisy tool.py"
    script.write_text(
        "import os\n"
        "os.write(1, b'output' * 100000)\n"
        f"os.write(2, b'x' * {stderr_budget * 3})\n"
        "os.write(2, b'\\nfinal diagnostic\\n')\n"
        "raise SystemExit(7)\n"
    )

    with pytest.raises(RuntimeError, match="exit 7") as error:
        run_command_template(f'"{sys.executable}" "{script}" {{in}} {{out}}', np.ones(2))

    message = str(error.value)
    assert message.endswith("final diagnostic")
    assert "stderr truncated" in message
    assert len(message.encode()) <= stderr_budget + 1024


@pytest.mark.parametrize("failure", ["termination", "reader start"])
def test_command_closes_stderr_when_runtime_setup_or_cleanup_raises(tmp_path, monkeypatch, failure):
    import subprocess

    from arrayscope.operations import command_runtime

    script = tmp_path / "sleeping tool.py"
    script.write_text("import time\ntime.sleep(30)\n")
    spawned = []
    real_popen = subprocess.Popen
    real_terminate = command_runtime._terminate_child

    def record_spawn(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    def terminate_then_raise(proc, grace):
        real_terminate(proc, grace)
        raise RuntimeError("termination failed")

    monkeypatch.setattr(subprocess, "Popen", record_spawn)
    if failure == "termination":
        monkeypatch.setattr(command_runtime, "_terminate_child", terminate_then_raise)
    else:

        def fail_reader_start(_reader):
            raise RuntimeError("reader start failed")

        monkeypatch.setattr(threading.Thread, "start", fail_reader_start)

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        run_command_template(
            f'"{sys.executable}" "{script}" {{in}} {{out}}',
            np.ones(2),
            timeout=0.05,
            grace=0.02,
        )

    assert len(spawned) == 1
    assert spawned[0].poll() is not None
    assert spawned[0].stderr.closed


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group lifecycle")
@pytest.mark.parametrize("interrupt", ["timeout", "cancel"])
def test_command_supervision_continues_after_leader_exits(tmp_path, interrupt):
    """A grandchild holding stderr open must not disable the command deadline.

    This exercises a real child group, including a grandchild which ignores
    TERM. The final cleanup makes the regression fail without leaving a
    sleeping process or a permanently blocked test thread on the old runtime.
    """

    import psutil

    marker = tmp_path / "child.json"
    script = tmp_path / "forking tool.py"
    script.write_text(
        "import json, os, shutil, signal, sys, time\n"
        "shutil.copyfile(sys.argv[1], sys.argv[2])\n"
        "leader = os.getpid()\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"    open({str(marker)!r}, 'w').write(json.dumps([leader, os.getpid()]))\n"
        "    time.sleep(30)\n"
        "    os._exit(0)\n"
        f"while not os.path.exists({str(marker)!r}):\n"
        "    time.sleep(0.005)\n"
    )
    token = SimpleNamespace(cancelled=False)
    outcome = {}

    def run():
        try:
            outcome["result"] = run_command_template(
                f'"{sys.executable}" "{script}" {{in}} {{out}}',
                np.ones(2),
                cancellation_token=token,
                timeout=1.0 if interrupt == "timeout" else None,
                poll_interval=0.005,
                grace=0.02,
            )
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    child_pid = None
    child_process = None
    try:
        deadline = time.monotonic() + INTERACTION_SETTLE_HARD_LIMIT_S
        while time.monotonic() < deadline:
            if marker.exists() and marker.read_text():
                leader_pid, child_pid = json.loads(marker.read_text())
                with suppress(psutil.NoSuchProcess):
                    child_process = psutil.Process(child_pid)
                leader_exited = True
                with suppress(psutil.NoSuchProcess):
                    leader_exited = psutil.Process(leader_pid).status() == psutil.STATUS_ZOMBIE
                if leader_exited:
                    break
            time.sleep(0.005)
        else:
            pytest.fail("the command leader did not exit")

        token.cancelled = interrupt == "cancel"
        thread.join(timeout=5)
        assert not thread.is_alive(), "inherited stderr bypassed cancellation/timeout"
        expected = EvaluationCancelled if interrupt == "cancel" else RuntimeError
        assert isinstance(outcome.get("error"), expected)
        if interrupt == "timeout":
            assert "timed out" in str(outcome["error"])
        with suppress(psutil.NoSuchProcess):
            assert psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if child_pid is None and marker.exists() and marker.read_text():
            _leader_pid, child_pid = json.loads(marker.read_text())
        if child_pid is not None:
            with suppress(psutil.NoSuchProcess):
                if child_process is None:
                    child_process = psutil.Process(child_pid)
                child_process.kill()
        thread.join(timeout=5)


@pytest.mark.parametrize("handoff", ["npy", "cfl"])
def test_array_handoff_round_trip(tmp_path, handoff):
    executable = tmp_path / "copy_tool.py"
    if handoff == "npy":
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import numpy as np, sys\n"
            "np.save(sys.argv[2], np.load(sys.argv[1], allow_pickle=False), allow_pickle=False)\n"
        )
    else:
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import shutil, sys\n"
            "for suffix in ('.hdr', '.cfl'):\n"
            "    shutil.copyfile(sys.argv[1] + suffix, sys.argv[2] + suffix)\n"
        )
    executable.chmod(0o755)
    source = np.arange(6, dtype=np.float32).reshape(2, 3)

    result = run_command_template(
        f'"{executable}" {{in}} {{out}}',
        source,
        handoff=handoff,
    )

    expected = source.astype(np.complex64) if handoff == "cfl" else source
    assert np.array_equal(result, expected)


def _cfl_copy_tool(tmp_path: Path) -> Path:
    executable = tmp_path / "cfl_copy_tool.py"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "for suffix in ('.hdr', '.cfl'):\n"
        "    shutil.copyfile(sys.argv[1] + suffix, sys.argv[2] + suffix)\n"
    )
    executable.chmod(0o755)
    return executable


def test_cfl_round_trip_preserves_trailing_singleton_axes(tmp_path):
    # read_cfl drops trailing singletons so BART's fixed 16-dimension header
    # reads back sanely; unconditionally, that turned the *transport* into a
    # rank reduction -- (4, 3, 1) came back (4, 3) -- and probe shapes preserve
    # size-1 axes, so a cfl operation was systematically mischaracterized.
    executable = _cfl_copy_tool(tmp_path)
    source = np.arange(12, dtype=np.float32).reshape(4, 3, 1)

    result = run_command_template(f'"{executable}" {{in}} {{out}}', source, handoff="cfl")

    assert result.shape == (4, 3, 1)
    assert np.array_equal(result, source.astype(np.complex64))


def test_cfl_reader_floor_never_invents_axes_a_tool_did_not_write(tmp_path):
    # A tool that genuinely returns a lower rank writes a shorter header; the
    # floor only stops *popping*, it never pads back to the input rank.
    stem = str(tmp_path / "array")
    write_cfl(stem, np.arange(6, dtype=np.complex64).reshape(2, 3))

    assert read_cfl(stem, min_rank=5).shape == (2, 3)
    assert read_cfl(stem).shape == (2, 3)


# --- cancellation reaches the child through the builder path -----------------


def _sleeping_command_spec(operation_id: str, executable: Path, started_marker: Path | None = None):
    """A plugin spec whose bound fn is the real command-runtime builder."""

    from arrayscope.operations import library
    from arrayscope.operations.plugins import PluginOperationSpec

    marker = "" if started_marker is None else f' --started="{started_marker}"'
    build = library._make_command_build(
        "command",
        {
            "command_template": f'"{executable}"{marker} --sleep 5 {{in}} {{out}}',
            "handoff": "npy",
            "shell": False,
            "timeout_s": None,
        },
        environment_id="",
    )
    return PluginOperationSpec(id=operation_id, label=operation_id, build=build)


def test_builder_bound_command_is_cancellable_mid_flight(tmp_path):
    executable = _fake_npy_command(tmp_path)
    spec = _sleeping_command_spec("demo:sleepy-command", executable)
    bound = spec.resolve_fn(None, {})
    token = SimpleNamespace(cancelled=False)
    outcome = {}

    def run():
        try:
            bound(np.ones(2), cancellation_token=token)
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.05)
    token.cancelled = True
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), EvaluationCancelled)


def test_pipeline_evaluation_stops_a_command_op_on_a_pre_cancelled_token(tmp_path):
    from arrayscope.operations import plugins
    from arrayscope.operations.pipeline import evaluate

    executable = _fake_npy_command(tmp_path)
    started_marker = tmp_path / "child-started"
    seams.reset_plugin_cache()
    try:
        spec = _sleeping_command_spec("demo:sleepy-pipeline", executable, started_marker)
        plugins._SPEC_CACHE[spec.id] = spec
        operation = plugins.PluginOperation(plugin_id=spec.id)
        context = SimpleNamespace(cancellation_token=SimpleNamespace(cancelled=True))

        with pytest.raises(EvaluationCancelled):
            evaluate(np.ones((2, 2), dtype=np.float32), (operation,), evaluation_context=context)
    finally:
        seams.reset_plugin_cache()

    # The token used to be dropped on the floor: apply() took no context and
    # apply_to_region deleted the one it was handed, so this waited out the full
    # 5 s sleep instead of refusing before the child was even spawned.
    #
    # That was pinned as ``elapsed < 2.0``, which is the weaker claim and the
    # load-sensitive one: it passes for a spawn that got killed early, and it
    # fails for a correct refusal on a box too busy to return in two seconds.
    # The child writes this marker before it sleeps, so its absence says exactly
    # what the paragraph above says -- nothing was spawned at all.
    assert not started_marker.exists(), "the pre-cancelled token still spawned the child"
