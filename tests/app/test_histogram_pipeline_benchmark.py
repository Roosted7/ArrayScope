from pathlib import Path


def test_representative_histogram_matrix_covers_dtype_layout_distribution_and_scale():
    from arrayscope.tools.histogram_pipeline_benchmark import benchmark_cases

    cases = benchmark_cases("representative")

    assert {case.dtype for case in cases} == {
        "uint8",
        "int16",
        "uint16",
        "float32",
        "float64",
        "complex64",
    }
    assert {case.layout for case in cases} == {"contiguous", "strided", "npy_mmap"}
    assert {"gradient", "outliers", "nonfinite"} <= {case.distribution for case in cases}
    assert {case.source_count for case in cases} == {1, 60, 272}


def test_cpu_histogram_smoke_runs_production_algorithms_and_checks_continuity(tmp_path):
    from arrayscope.tools.histogram_pipeline_benchmark import run_benchmark

    result = run_benchmark(
        suite="smoke",
        engines=("cpu",),
        shape=(24, 20),
        repetitions=1,
        data_path=Path("unused"),
    )

    assert result["schema"] == "arrayscope.histogram-pipeline-benchmark.v1"
    assert result["correct"] is True
    assert result["gpu"] == {}
    assert len(result["cpu"]["summaries"]) == 3
    assert all(row["correct"] for row in result["cpu"]["summaries"])
    assert {row["covered_sources"] for row in result["cpu"]["measurements"]} == {1, 4}
    assert result["route_contract"]["prepared_page_summary"] == "reuse on every backend"


def test_histogram_benchmark_rejects_unknown_engine():
    import pytest

    from arrayscope.tools.histogram_pipeline_benchmark import run_benchmark

    with pytest.raises(ValueError, match="unknown histogram benchmark engines"):
        run_benchmark(
            suite="smoke",
            engines=("magic",),
            shape=(8, 8),
            repetitions=1,
        )
