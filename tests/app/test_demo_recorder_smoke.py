"""Keeps the README demo scenarios recordable as the UI evolves.

tools/demo_recorder.py resolves every interaction target from live widget
geometry at run time. When a widget or window entry point it scripts against
is renamed or removed, these smoke runs fail immediately — long before
anyone next re-renders the README media — so the demo pipeline stays
maintainable instead of rotting silently.

Each case records a scenario in ``--smoke`` mode (high speed factor, low
fps, no encoding), which keeps a full scenario run to a few seconds.
"""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECORDER = ROOT / "tools" / "demo_recorder.py"


def _scenario_names() -> tuple[str, ...]:
    namespace = runpy.run_path(str(RECORDER), run_name="arrayscope_demo_recorder")
    return tuple(namespace["SCENARIOS"])


@pytest.mark.parametrize("name", _scenario_names())
def test_demo_scenario_records_in_smoke_mode(name, tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(RECORDER),
            "--smoke",
            "--only",
            name,
            "--frames",
            str(tmp_path),
            "--jobs",
            "1",
            "--keep-frames",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(ROOT),
        check=False,
    )
    assert proc.returncode == 0, f"{name} failed:\n{proc.stdout}\n{proc.stderr}"
    frames = sorted((tmp_path / name).glob("frame_*.png"))
    assert frames, f"{name} recorded no frames"
    manifest = json.loads((tmp_path / name / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["scenario"] == name
    assert manifest["frame_count"] == len(frames)
    assert manifest["window_size"][0] > 0
    assert manifest["window_size"][1] > 0
    captions = [event for event in manifest["events"] if event["kind"] == "caption"]
    actions = [event for event in manifest["events"] if event["kind"] != "caption"]
    assert captions, f"{name} recorded no caption beats"
    assert actions, f"{name} recorded no action beats"
    assert all(event["end_frame"] is not None for event in manifest["events"])
