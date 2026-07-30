from __future__ import annotations

import json
import threading
import time
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


def _fake_npy_command(tmp_path: Path) -> Path:
    executable = tmp_path / "fake array tool"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, time\n"
        "import numpy as np\n"
        "args = sys.argv[1:]\n"
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


def _sleeping_command_spec(operation_id: str, executable: Path):
    """A plugin spec whose bound fn is the real command-runtime builder."""

    from arrayscope.operations import library
    from arrayscope.operations.plugins import PluginOperationSpec

    build = library._make_command_build(
        "command",
        {
            "command_template": f'"{executable}" --sleep 5 {{in}} {{out}}',
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
    plugins._reset_plugin_cache()
    try:
        spec = _sleeping_command_spec("demo:sleepy-pipeline", executable)
        plugins._SPEC_CACHE[spec.id] = spec
        operation = plugins.PluginOperation(plugin_id=spec.id)
        context = SimpleNamespace(cancellation_token=SimpleNamespace(cancelled=True))

        started = time.monotonic()
        with pytest.raises(EvaluationCancelled):
            evaluate(np.ones((2, 2), dtype=np.float32), (operation,), evaluation_context=context)
        elapsed = time.monotonic() - started
    finally:
        plugins._reset_plugin_cache()

    # The token used to be dropped on the floor: apply() took no context and
    # apply_to_region deleted the one it was handed, so this waited out the
    # full 5 s sleep instead of refusing before the child was even spawned.
    assert elapsed < 2.0
