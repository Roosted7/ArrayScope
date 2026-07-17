"""Opt-in synthetic stress matrix: full workflow x dataset shapes x trace replay.

Every run drives the complete profile-workflow phase sequence (cold raw, FFT,
level refinement, fast/slow scroll, zoom/pan) against a synthetic dataset the
code has never seen, then replays the recorded trace through ``verify_trace``.
The workflow harness is the stressor; the trace invariants are the oracle.

Opt-in because a full matrix run takes minutes:

    ARRAYSCOPE_STRESS=1 pytest tests/stress -q -n 0

Add datasets to ``DATASETS`` when a new feature adds a new input class
(dtype, anisotropy, tiny axes, ...). Add invariants in ``verify_trace``, not
here — this module should stay a thin matrix runner.

KNOWN STATE (2026-07-16, G5 page cutover): native complex64 now converges on
the canonical mean-complex page route. The tiny cell's level settlement
remains a pre-existing non-strict xfail. Do not add retries or widen timeouts
to hide any recurrence; the matrix goes green by fixing convergence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("ARRAYSCOPE_STRESS"),
        reason="stress matrix is opt-in: set ARRAYSCOPE_STRESS=1",
    ),
    # One xdist group: concurrent full-app runs contend for CPU and trip the
    # harness's timing-sensitive stall grace with false positives.  The matrix
    # measures the app, not the load average — always serialize.
    pytest.mark.xdist_group("arrayscope-stress-serial"),
]

ROOT = Path(__file__).parents[2]

# Until production emits `target_satisfied_retained` for targets closed by
# retained compatible payloads, whole-workflow replays legitimately end with
# final targets that have no ack event.  Tolerate exactly that invariant and
# nothing else; tighten by emptying this set once the emitter lands.
TOLERATED_INVARIANTS = frozenset({"final_required_target_acknowledged"})


def _dataset(rng, shape, dtype):
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        data = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        return (data * 100).astype(dtype)
    return (rng.normal(size=shape) * 100).astype(dtype)


DATASETS = (
    pytest.param((64, 64, 12), "float32", 6, id="small-float32"),
    pytest.param((96, 48, 7), "float32", 5, id="anisotropic-odd-depth"),
    pytest.param(
        (32, 32, 3),
        "float32",
        3,
        id="tiny-3-slices",
        marks=pytest.mark.xfail(
            reason="FINDING 2026-07-15: level-presentation settlement is racy"
            " on a 3-tile montage — fft_level_refinement_preview sometimes ends"
            " presentation_settled=False (intermittent across runs)",
            strict=False,
        ),
    ),
    pytest.param((64, 64, 10), "complex64", 6, id="complex64"),
    pytest.param((48, 64, 40), "float64", 12, id="float64-deeper-axis"),
)


@pytest.mark.parametrize(("shape", "dtype", "max_tiles"), DATASETS)
def test_full_workflow_settles_and_trace_verifies(tmp_path, shape, dtype, max_tiles):
    rng = np.random.default_rng(sum(shape))
    data_path = tmp_path / "stress.npy"
    np.save(data_path, _dataset(rng, shape, dtype))
    trace_path = tmp_path / "trace.jsonl"
    jsonl_path = tmp_path / "run.jsonl"

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", TMPDIR=str(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "arrayscope.tools.profile_montage_workflow",
            "--data",
            str(data_path),
            "--load-mode",
            "native",
            "--backend",
            "pyqtgraph",
            "--max-tiles",
            str(max_tiles),
            "--timeout-s",
            "5",
            "--session-fixture",
            "",
            "--jsonl",
            str(jsonl_path),
            "--trace",
            str(trace_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        # Process-deadlock guard for the whole multi-stage child.  Each
        # user-visible stage has already hard-failed at five seconds.
        timeout=60,
    )
    # Exit code 1 alone is tolerated: the R8 certification gates
    # (full-grid-not-capped, presentation-continuity, ...) are calibrated for
    # the canonical fixture geometry and are advisory under synthetic stress.
    # A crash, stall, or timeout is not.
    assert result.returncode in (0, 1) and "Traceback" not in result.stderr, (
        f"workflow crashed/stalled for {shape}/{dtype}:\n"
        f"{result.stdout[-1500:]}\n{result.stderr[-2500:]}"
    )

    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    phases = [record.get("phase") for record in records]
    # Scroll/zoompan phases are skipped for very short index axes; the four
    # core phases must always complete.
    core = {
        "load_data",
        "raw_full_tiled_montage",
        "fft_full_tiled_montage",
        "fft_level_refinement_preview",
    }
    assert core.issubset(set(phases)), f"core phases missing (crash mid-run?): {phases}"
    unsettled = [
        record["phase"]
        for record in records
        if record.get("presentation_settled") is False
    ]
    assert not unsettled, f"phases left unsettled: {unsettled}"

    from arrayscope.tools.trace_verify import verify_trace

    verification = verify_trace(trace_path)
    hard = [
        violation
        for violation in verification["violations"]
        if violation["invariant"] not in TOLERATED_INVARIANTS
    ]
    assert not hard, (
        f"trace invariants violated for {shape}/{dtype}: {hard[:5]} "
        f"(events={verification['event_count']})"
    )
