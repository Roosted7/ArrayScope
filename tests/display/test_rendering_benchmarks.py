import pytest


pytest.importorskip("vispy")


def test_rendering_backend_benchmarks_report_expected_scenarios(qt_app):
    from arrayscope.display.rendering_benchmarks import benchmark_rendering_backends

    results = benchmark_rendering_backends()

    assert {result.name for result in results} == {
        "pyqtgraph_scalar_level_preview",
        "vispy_scalar_level_preview",
        "pyqtgraph_complex_tile_level_preview",
        "vispy_complex_tile_level_preview",
        "pyqtgraph_clean_tile_flush",
        "vispy_clean_tile_flush",
    }
    for result in results:
        assert result.elapsed_ms >= 0.0
        assert result.timing.mode


def test_vispy_complex_tile_preview_uses_less_cpu_work_than_pyqtgraph(qt_app):
    from arrayscope.display.rendering_benchmarks import benchmark_rendering_backends

    results = {result.name: result for result in benchmark_rendering_backends()}
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

    results = {result.name: result for result in benchmark_rendering_backends()}
    vispy = results["vispy_clean_tile_flush"].timing

    assert vispy.tile_layer_visible_items > 0
    assert vispy.tile_layer_items_updated == 0
    assert vispy.tile_layer_items_skipped == vispy.tile_layer_visible_items
    assert vispy.tile_layer_upload_ms == 0.0
    assert vispy.visible_bytes == 0
