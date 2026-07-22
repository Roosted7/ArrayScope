from __future__ import annotations

import numpy as np

from arrayscope.gpu.keys import SCALAR_R32F
from arrayscope.tools import g7_live_compression_benchmark as bench


def test_host_case_uses_one_total_budget_and_unique_keys():
    tiles = [np.full((256, 256), index, np.float32) for index in range(12)]
    result = bench._host_cache_case(tiles, "auto", budget_chunks=4)

    assert result["reported_max_bytes"] == result["total_budget_bytes"]
    assert result["raw_budget_bytes"] + result["tier_budget_bytes"] == result["total_budget_bytes"]
    assert result["unique_resident_keys"] <= len(tiles)
    assert result["combined_used_bytes"] <= result["total_budget_bytes"]


def test_parser_defaults_to_evidence_gated_modes():
    args = bench._parser().parse_args(())
    assert args.texture_codec == "off"
    assert args.host_codec == "raw"
    assert args.representation == "scalar"
    assert args.scope == "both"
    assert SCALAR_R32F == "scalar_r32f"


def test_host_only_scope_does_not_construct_gpu(monkeypatch, tmp_path):
    tile = np.ones((bench.PAGE, bench.PAGE), dtype=np.float32)
    monkeypatch.setattr(bench, "_load_tiles", lambda *_args: [tile] * 4)
    monkeypatch.setattr(
        bench,
        "_gpu_case",
        lambda *_args: (_ for _ in ()).throw(AssertionError("GPU must remain untouched")),
    )

    result = bench.run_benchmark(
        data_path=tmp_path / "unused.nii",
        power="low-power",
        texture_codec="off",
        host_codec="raw",
        representation=bench.SCALAR_R32F,
        pages=4,
        budget_chunks=2,
        scope="host",
    )

    assert "host" in result
    assert "gpu" not in result
