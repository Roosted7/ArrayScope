from arrayscope.operations.benchmarks import benchmark_fft_slice, run_foundation_benchmarks


def test_run_foundation_benchmarks_returns_expected_scenarios():
    results = run_foundation_benchmarks()

    assert {result.name for result in results} == {"raw_slice", "fft_slice", "montage_canvas", "roi_stats"}
    for result in results:
        assert result.elapsed_ms >= 0
        assert result.output_shape
        assert result.dtype
        assert result.output_dtype


def test_fft_benchmark_honors_workers_argument():
    result = benchmark_fft_slice(workers=1)

    assert result.name == "fft_slice"
    assert result.peak_estimate_bytes is not None
