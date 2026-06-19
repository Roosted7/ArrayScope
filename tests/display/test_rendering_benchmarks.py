import json

import pytest


pytest.importorskip("vispy")


def test_rendering_backend_benchmarks_report_expected_scenarios(qt_app):
    from arrayscope.display.rendering_benchmarks import assert_optional_perf_gates, benchmark_rendering_backends

    results = benchmark_rendering_backends(measure_presented=False)

    assert {result.name for result in results} == {
        "pyqtgraph_scalar_level_preview",
        "vispy_scalar_level_preview",
        "pyqtgraph_complex_tile_level_preview",
        "vispy_complex_tile_level_preview",
        "pyqtgraph_clean_tile_flush",
        "vispy_clean_tile_flush",
        "pyqtgraph_large_complex_tiled_initial",
        "vispy_large_complex_tiled_initial",
        "pyqtgraph_one_dirty_tile_commit",
        "vispy_one_dirty_tile_commit",
        "pyqtgraph_pan_zoom_no_upload",
        "vispy_pan_zoom_no_upload",
        "pyqtgraph_progressive_tile_stream",
        "vispy_progressive_tile_stream",
    }
    for result in results:
        assert result.elapsed_ms >= 0.0
        assert result.submission_ms == result.elapsed_ms
        assert result.first_frame_ms is None
        assert result.event_loop_drain_ms is None
        assert result.frame_count == 0
        assert result.ui_max_gap_ms is None
        assert result.commit_count >= 1
        assert result.timing.mode
    assert_optional_perf_gates(results)


def test_vispy_complex_tile_preview_uses_less_cpu_work_than_pyqtgraph(qt_app):
    from arrayscope.display.rendering_benchmarks import benchmark_rendering_backends

    results = {result.name: result for result in benchmark_rendering_backends(measure_presented=False)}
    pyqtgraph = results["pyqtgraph_complex_tile_level_preview"].timing
    vispy = results["vispy_complex_tile_level_preview"].timing

    assert pyqtgraph.tile_layer_rgb_window_tiles > 0
    assert pyqtgraph.tile_layer_rgb_window_ms > 0.0
    assert vispy.tile_layer_rgb_window_tiles == 0
    assert vispy.tile_layer_rgb_window_ms == 0.0
    assert vispy.tile_layer_upload_ms == 0.0
    assert vispy.visible_bytes == 0
    assert vispy.tile_layer_items_skipped == vispy.tile_layer_visible_items


def test_vispy_clean_tile_flush_skips_existing_visuals(qt_app):
    from arrayscope.display.rendering_benchmarks import benchmark_rendering_backends

    results = {result.name: result for result in benchmark_rendering_backends(measure_presented=False)}
    vispy = results["vispy_clean_tile_flush"].timing

    assert vispy.tile_layer_visible_items > 0
    assert vispy.tile_layer_items_updated == 0
    assert vispy.tile_layer_items_skipped == vispy.tile_layer_visible_items
    assert vispy.tile_layer_upload_ms == 0.0
    assert vispy.visible_bytes == 0


def test_vispy_dirty_and_pan_scenarios_have_deterministic_upload_counters(qt_app):
    from arrayscope.display.rendering_benchmarks import benchmark_rendering_backends

    results = {result.name: result for result in benchmark_rendering_backends(measure_presented=False)}
    dirty = results["vispy_one_dirty_tile_commit"].timing
    pan = results["vispy_pan_zoom_no_upload"].timing

    assert dirty.tile_layer_items_updated == 1
    assert dirty.tile_layer_items_skipped > 0
    assert dirty.visible_bytes > 0
    assert pan.tile_layer_items_updated == 0
    assert pan.visible_bytes == 0


def test_progressive_tile_stream_reports_aggregate_work(qt_app):
    from arrayscope.display.rendering_benchmarks import benchmark_rendering_backends

    results = {result.name: result for result in benchmark_rendering_backends(measure_presented=False)}
    vispy_result = results["vispy_progressive_tile_stream"]
    timing = vispy_result.timing

    assert vispy_result.commit_count == 12
    assert timing.tile_layer_visible_items == 96
    assert timing.tile_layer_items_updated == 96
    assert timing.tile_layer_storage_rebuilds == 1
    assert timing.tile_layer_resident_items == 96
    assert timing.tile_layer_texture_uploads > 0


def test_benchmark_jsonl_writer_emits_mergeable_sample_records(qt_app, tmp_path):
    from arrayscope.display.rendering_benchmarks import collect_benchmark_samples, write_benchmark_jsonl

    samples = collect_benchmark_samples(runs=1, stress=False, measure_presented=False)
    path = tmp_path / "rendering.jsonl"

    write_benchmark_jsonl(path, samples[:1])

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["run"] == 0
    assert record["environment"]["os"]
    assert "xdg_session_type" in record["environment"]
    assert "gpu_max_texture_size" in record["environment"]
    assert record["result"]["name"]
    assert record["result"]["timing"]["mode"]
