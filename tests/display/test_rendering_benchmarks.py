import json

import pytest


pytest.importorskip("vispy")

@pytest.fixture(scope="module")
def benchmark_results(qt_app):
    from pyqtgraph.Qt import QtWidgets

    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.rendering_benchmarks import benchmark_rendering_backends
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view_types = (ImageView2D, VisPyImageView2D)
    before = sum(
        isinstance(widget, view_types)
        for widget in QtWidgets.QApplication.topLevelWidgets()
    )
    results = benchmark_rendering_backends(measure_presented=False)
    after = sum(
        isinstance(widget, view_types)
        for widget in QtWidgets.QApplication.topLevelWidgets()
    )
    assert after <= before
    return results



def test_rendering_backend_benchmarks_report_expected_scenarios(benchmark_results):
    from arrayscope.display.rendering_benchmarks import assert_optional_perf_gates

    results = benchmark_results

    assert {result.name for result in results} == {
        "pyqtgraph_scalar_level_preview",
        "vispy_scalar_level_preview",
        "pyqtgraph_complex_tile_level_preview",
        "vispy_complex_tile_level_preview",
        "pyqtgraph_tile_level_uniform_update",
        "vispy_tile_level_uniform_update",
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
        "vispy_warm_residency_queue_scaling",
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


def test_vispy_complex_tile_preview_uses_less_cpu_work_than_pyqtgraph(benchmark_results):
    results = {result.name: result for result in benchmark_results}
    pyqtgraph = results["pyqtgraph_complex_tile_level_preview"].timing
    vispy = results["vispy_complex_tile_level_preview"].timing

    assert pyqtgraph.tile_layer_rgb_window_tiles > 0
    assert pyqtgraph.tile_layer_rgb_window_ms > 0.0
    assert vispy.tile_layer_rgb_window_tiles == 0
    assert vispy.tile_layer_rgb_window_ms == 0.0
    assert vispy.tile_layer_upload_ms == 0.0
    assert vispy.visible_bytes == 0
    assert vispy.tile_layer_items_skipped == vispy.tile_layer_visible_items


def test_vispy_clean_tile_flush_skips_existing_visuals(benchmark_results):
    results = {result.name: result for result in benchmark_results}
    vispy = results["vispy_clean_tile_flush"].timing

    assert vispy.tile_layer_visible_items > 0
    assert vispy.tile_layer_items_updated == 0
    assert vispy.tile_layer_items_skipped == vispy.tile_layer_visible_items
    assert vispy.tile_layer_upload_ms == 0.0
    assert vispy.visible_bytes == 0


def test_vispy_dirty_and_pan_scenarios_have_deterministic_upload_counters(benchmark_results):
    results = {result.name: result for result in benchmark_results}
    dirty = results["vispy_one_dirty_tile_commit"].timing
    pan = results["vispy_pan_zoom_no_upload"].timing

    assert dirty.tile_layer_items_updated == 1
    assert dirty.tile_layer_items_skipped > 0
    assert dirty.visible_bytes > 0
    assert pan.tile_layer_items_updated == 0
    assert pan.tile_layer_texture_uploads == 0
    assert pan.tile_layer_texture_upload_bytes == 0
    assert pan.tile_layer_vertex_uploads == 0
    assert pan.tile_layer_level_updates == 0
    assert pan.tile_layer_shader_uniform_updates == 0
    assert pan.visible_bytes == 0


def test_vispy_level_only_tile_commit_updates_uniforms_without_uploads(benchmark_results):
    results = {result.name: result for result in benchmark_results}
    timing = results["vispy_tile_level_uniform_update"].timing

    assert timing.tile_layer_visible_items > 0
    assert timing.tile_layer_items_updated == 0
    assert timing.tile_layer_items_skipped == timing.tile_layer_visible_items
    assert timing.tile_layer_texture_uploads == 0
    assert timing.tile_layer_texture_upload_bytes == 0
    assert timing.tile_layer_level_updates > 0
    assert timing.tile_layer_shader_uniform_updates > 0
    assert timing.visible_bytes == 0


def test_warm_residency_queue_scaling_reports_batched_speculative_uploads(benchmark_results):
    results = {result.name: result for result in benchmark_results}
    result = results["vispy_warm_residency_queue_scaling"]
    timing = result.timing

    assert result.commit_count == 8
    assert timing.tile_layer_items_updated == 32
    assert timing.tile_layer_resident_items == 40
    assert timing.tile_layer_warm_resident_items == 32
    assert timing.tile_layer_texture_uploads > 0
    assert timing.tile_layer_texture_upload_bytes > 0
    assert timing.tile_layer_near_resident_items == 40


def test_progressive_tile_stream_reports_aggregate_work(benchmark_results):
    results = {result.name: result for result in benchmark_results}
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
