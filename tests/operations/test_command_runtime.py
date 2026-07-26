from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.command_runtime import build_command, run_command_template


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
