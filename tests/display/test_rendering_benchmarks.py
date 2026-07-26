import json
from types import SimpleNamespace

import pytest

from arrayscope.display.backend_contract import ImageViewBackendCapabilities


@pytest.fixture(scope="module")
def benchmark_results(qt_app):
    from pyqtgraph.Qt import QtWidgets

    from arrayscope.display.backends.pyqtgraph.surface import PyQtGraphSurface
    from arrayscope.display.rendering_benchmarks import benchmark_pyqtgraph_rendering

    view_types = (PyQtGraphSurface,)
    before = sum(
        isinstance(widget, view_types) for widget in QtWidgets.QApplication.topLevelWidgets()
    )
    results = benchmark_pyqtgraph_rendering(measure_presented=False)
    after = sum(
        isinstance(widget, view_types) for widget in QtWidgets.QApplication.topLevelWidgets()
    )
    assert after <= before
    return results


def test_rendering_backend_benchmarks_report_expected_scenarios(benchmark_results):
    from arrayscope.display.rendering_benchmarks import assert_optional_perf_gates

    results = benchmark_results

    assert {result.name for result in results} == {
        "pyqtgraph_tiled_small_initial",
        "pyqtgraph_tiled_large_initial",
        "pyqtgraph_one_tile_montage_initial",
        "pyqtgraph_multi_tile_montage_initial",
        "pyqtgraph_scalar_level_preview",
        "pyqtgraph_large_histogram_plot_refresh",
        "pyqtgraph_complex_tile_level_preview",
        "pyqtgraph_large_tile_level_preview",
        "pyqtgraph_tile_level_uniform_update",
        "pyqtgraph_clean_tile_flush",
        "pyqtgraph_large_complex_tiled_initial",
        "pyqtgraph_one_dirty_tile_commit",
        "pyqtgraph_pan_zoom_no_upload",
        "pyqtgraph_progressive_tile_stream",
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
        if result.scenario == "tiled_large_initial":
            assert result.timing.mode == "tile_layer"
            assert result.timing.tile_layer_visible_items == 16
        assert result.lod_policy == "native-only"
        assert result.lod_applied_factor == 1
        assert result.lod_applied_factor_xy == (1, 1)
        assert result.lod_desired_factor == result.lod_applied_factor
        assert result.lod_desired_factor >= 1
        assert len(result.lod_source_texels_per_pixel_xy) == 2
    assert_optional_perf_gates(results)


def test_benchmark_result_does_not_mask_backend_applied_lod():
    from arrayscope.core.runtime_diagnostics import ImageUploadTiming
    from arrayscope.display.rendering_benchmarks import _ActionMeasurement, _result

    view = SimpleNamespace(
        rendering_capabilities=ImageViewBackendCapabilities(name="test"),
        lastImageUploadTiming=lambda: ImageUploadTiming(),
    )
    timing = ImageUploadTiming(
        mode="test",
        tile_layer_lod_factor=4,
        tile_layer_source_texels_per_pixel=8.0,
    )

    result = _result(
        view,
        "non_native_probe",
        _ActionMeasurement(submission_ms=0.0),
        timing=timing,
        commit_count=2,
    )

    assert result.lod_applied_factor == 4
    assert result.lod_applied_factor_xy == (4, 4)
    assert result.lod_policy == "backend-reported"
    assert "non-native applied" in result.lod_reason
    assert result.kernel_counters["backend_commit"]["admitted"] == 2
    assert result.kernel_counters["backend_commit"]["completed"] == 2


def test_large_pyqtgraph_tile_preview_reports_level_work_without_texture_counters(
    benchmark_results,
):
    results = {result.name: result for result in benchmark_results}
    timing = results["pyqtgraph_large_tile_level_preview"].timing

    assert timing.tile_layer_visible_items > 8
    assert timing.tile_layer_rgb_window_tiles == timing.tile_layer_visible_items
    assert timing.tile_layer_level_updates == timing.tile_layer_visible_items
    assert timing.tile_layer_texture_uploads == 0
    assert timing.tile_layer_level_update_pending_items == 0


def test_pyqtgraph_clean_tile_flush_attempts_no_item_work(benchmark_results):
    results = {result.name: result for result in benchmark_results}
    timing = results["pyqtgraph_clean_tile_flush"].timing

    assert timing.tile_layer_visible_items > 0
    assert timing.tile_layer_items_updated == 0
    assert timing.tile_layer_items_skipped == 0
    assert timing.tile_layer_texture_uploads == 0


def test_pyqtgraph_dirty_and_pan_scenarios_have_deterministic_work_counters(benchmark_results):
    results = {result.name: result for result in benchmark_results}
    dirty = results["pyqtgraph_one_dirty_tile_commit"].timing
    pan = results["pyqtgraph_pan_zoom_no_upload"].timing

    assert dirty.tile_layer_items_updated == 1
    # PyQtGraph receives only the dirty delta; unchanged visible items are not
    # attempted and therefore are neither updated nor counted as skipped.
    assert dirty.tile_layer_items_skipped == 0
    assert pan.tile_layer_items_updated == 0
    assert pan.tile_layer_texture_uploads == 0
    assert pan.tile_layer_texture_upload_bytes == 0
    assert pan.tile_layer_vertex_uploads == 0


def test_progressive_tile_stream_reports_aggregate_work(benchmark_results):
    results = {result.name: result for result in benchmark_results}
    result = results["pyqtgraph_progressive_tile_stream"]
    timing = result.timing

    assert result.commit_count == 12
    assert timing.tile_layer_visible_items == 96
    assert timing.tile_layer_items_updated == 96


def test_benchmark_jsonl_writer_emits_mergeable_sample_records(qt_app, tmp_path):
    from arrayscope.display.rendering_benchmarks import (
        collect_benchmark_samples,
        write_benchmark_jsonl,
    )

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
    assert "presentation_revision" in record["result"]
    assert "presentation_stale_count" in record["result"]
    assert "presentation_pending_count" in record["result"]
    assert "presentation_settled" in record["result"]
