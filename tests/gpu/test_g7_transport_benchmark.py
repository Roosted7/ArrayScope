"""G7 benchmark smoke: the tool runs and emits a matrix on a tiny input.

Law #4: a smoke test, not a timing assertion -- it proves the benchmark
produces a well-formed matrix, never that a particular cell wins.
"""

from __future__ import annotations

import numpy as np

from arrayscope.tools import g7_transport_benchmark as bench


class _FakeNib:
    def __init__(self, array):
        self._array = array

    class _Img:
        def __init__(self, array):
            self.dataobj = array

    def load(self, _path):
        return self._Img(self._array)


def test_benchmark_runs_and_emits_matrix(monkeypatch):
    # Tiny synthetic volume: two 256x256 planes so at least one chunk exists.
    volume = (np.random.default_rng(7).standard_normal((2, bench.PAGE, bench.PAGE)) * 500).astype(
        np.float32
    )
    monkeypatch.setattr(bench, "_load_volume", lambda _p: volume)

    result = bench.run_benchmark(bench.DEFAULT_DATA_PATH, chunk_limit=4)

    assert result["schema"] == "arrayscope.g7-transport-benchmark.v1"
    assert result["cells"], "expected at least one matrix cell"
    for cell in result["cells"]:
        assert cell["dtype"] in ("float32", "complex64", "int16")
        assert cell["raw_bytes"] > 0
        # Every emitted cell must be lossless-exact (default is lossless).
        assert cell["exact"] is True
        assert "break_even_gbps" in cell
    text = bench._format_matrix(result)
    assert "verdict" in text
    assert "break_even" in text


def test_benchmark_main_prints(monkeypatch, capsys):
    volume = np.zeros((1, bench.PAGE, bench.PAGE), dtype=np.float32)
    monkeypatch.setattr(bench, "_load_volume", lambda _p: volume)
    rc = bench.main(("--chunk-limit", "1"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "verdict" in out
