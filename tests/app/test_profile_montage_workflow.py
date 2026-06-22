import os
from pathlib import Path

import pytest


def test_profile_montage_workflow_py_spy_command_mentions_external_sampler():
    from arrayscope.tools.profile_montage_workflow import py_spy_command

    command = py_spy_command(("--backend", "all", "--jsonl", "out.jsonl"))

    assert command.startswith("py-spy record")
    assert "arrayscope.tools.profile_montage_workflow" in command
    assert "--backend all" in command


@pytest.mark.skipif(
    os.environ.get("ARRAYSCOPE_RUN_PROFILE_WORKFLOW") != "1",
    reason="opt-in realistic GUI profiling workflow; set ARRAYSCOPE_RUN_PROFILE_WORKFLOW=1",
)
def test_profile_montage_workflow_realistic_dataset_optional(tmp_path):
    from arrayscope.tools.profile_montage_workflow import DEFAULT_DATA_PATH, run_profile_montage_workflow

    data_path = Path(os.environ.get("ARRAYSCOPE_PROFILE_DATA", DEFAULT_DATA_PATH))
    if not data_path.exists():
        pytest.skip(f"profile dataset not found: {data_path}")

    backends = tuple(
        backend.strip()
        for backend in os.environ.get("ARRAYSCOPE_PROFILE_BACKENDS", "pyqtgraph,vispy").split(",")
        if backend.strip()
    )
    if "vispy" in backends:
        pytest.importorskip("vispy")

    timeout_s = float(os.environ.get("ARRAYSCOPE_PROFILE_TIMEOUT_S", "180"))
    max_tiles_raw = int(os.environ.get("ARRAYSCOPE_PROFILE_MAX_TILES", "0"))
    max_tiles = None if max_tiles_raw <= 0 else max_tiles_raw
    jsonl = tmp_path / "profile-workflow.jsonl"
    all_records = []

    for backend in backends:
        records = run_profile_montage_workflow(
            data_path=data_path,
            backend=backend,
            jsonl=jsonl,
            timeout_s=timeout_s,
            max_tiles=max_tiles,
            show_window=os.environ.get("ARRAYSCOPE_PROFILE_HIDE_WINDOW") != "1",
        )
        all_records.extend(records)

    phases = {(record["backend"], record["phase"]) for record in all_records}
    for backend in backends:
        assert (backend, "load_data") in phases
        assert (backend, "raw_full_tiled_montage") in phases
        assert (backend, "fft_full_tiled_montage") in phases
    assert jsonl.exists()
