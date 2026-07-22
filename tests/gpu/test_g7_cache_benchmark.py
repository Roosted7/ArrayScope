"""G7 Phase A cache-benchmark smoke: the tool runs and emits a well-formed report.

A smoke test, not a timing assertion -- it proves the benchmark produces a
structured RAM/eviction/time report on a tiny input, never that a particular
cell wins (timings are machine- and load-dependent).
"""

from __future__ import annotations

import numpy as np

from arrayscope.tools import g7_cache_benchmark as bench


def test_benchmark_runs_and_reports(monkeypatch):
    # Small synthetic volume: enough planes to yield a modest working set.
    volume = (np.random.default_rng(3).standard_normal((40, bench.PAGE, bench.PAGE)) * 500).astype(
        np.float32
    )
    monkeypatch.setattr(bench, "_load_volume", lambda _p: volume)

    result = bench.run_benchmark(
        bench.DEFAULT_DATA_PATH,
        working_set=12,
        sweeps=2,
        budget_chunks=4,
        miss_fft=64,
        miss_fft_sweep=(64, 128),
    )

    assert result["schema"] == "arrayscope.g7-cache-benchmark.v1"
    assert result["topology"]["kind"] in ("integrated", "discrete", "unknown")
    assert result["dtypes"], "expected per-dtype rows"
    for row in result["dtypes"]:
        assert row["dtype"] in ("float32", "complex64", "int16")
        # The RAM win is the invariant: the two-level tier retains at least as
        # many distinct chunks as raw under the same byte budget.
        assert row["two_level"]["fit_chunks"] >= row["raw_only"]["fit_chunks"]
        # Fewer or equal expensive recomputes with the tier engaged.
        assert row["two_level"]["recomputes"] <= row["raw_only"]["recomputes"]
        assert row["ratio"] >= 1.0
        # Formatting must not raise.
    text = bench._format(result)
    assert "RAM win" in text
    assert "End-to-end" in text
