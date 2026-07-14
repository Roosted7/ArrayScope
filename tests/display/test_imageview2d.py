import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")


def _clear_histogram_jobs(view) -> None:
    controller = getattr(view, "_histogram_display_controller", None)
    controller.cancel()
    controller._closed = False


def _viewport_pos_for_image_point(view, x: float, y: float):
    from pyqtgraph.Qt import QtCore

    scene_pos = view.getView().mapViewToScene(QtCore.QPointF(float(x), float(y)))
    return QtCore.QPointF(view.graphicsView.mapFromScene(scene_pos))


def _send_viewport_mouse(view, event_type, image_point, *, button=None, buttons=None):
    from pyqtgraph.Qt import QtCore, QtGui

    if button is None:
        button = QtCore.Qt.MouseButton.NoButton
    if buttons is None:
        buttons = button
    event = QtGui.QMouseEvent(
        event_type,
        _viewport_pos_for_image_point(view, *image_point),
        button,
        buttons,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    return view.eventFilter(view.graphicsView.viewport(), event)


def _view_class(backend):
    if backend == "pyqtgraph":
        from arrayscope.display.imageview2d import ImageView2D

        return ImageView2D
    pytest.importorskip("vispy")
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    return VisPyImageView2D


def _present_tiled(
    view,
    canvas,
    *,
    histogramData=None,
    histogramPlotData=None,
    geometry=None,
    levels,
    histogramRange,
    viewport_policy=None,
    rgb_already_windowed=False,
    montage_dirty_tiles=None,
    montage_tile_source_ids=None,
    montage_tile_payloads=None,
    tile_delta=None,
    tile_residency_budget_bytes=0,
    frame_plan=None,
):
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.viewport import ViewportPolicy

    if geometry is None:
        geometry = _single_tile_geometry(canvas)
    if histogramPlotData is None and histogramData is not None:
        histogramPlotData = histogramData
    payloads = dict(montage_tile_payloads or {})
    if not payloads:
        cache_key = (
            tuple(geometry.montage.indices),
            tuple(geometry.montage.tile_shape),
            int(geometry.montage.columns),
            int(geometry.montage.rows),
            int(geometry.montage.gap),
            int(geometry.montage_origin_x),
            int(geometry.montage_origin_y),
            tuple(str(getattr(state, "value", state)) for state in geometry.montage_tile_states),
            tuple(np.shape(canvas)),
            str(np.asarray(canvas).dtype),
            None if histogramData is None else tuple(np.shape(histogramData)),
            None if histogramData is None else str(np.asarray(histogramData).dtype),
        )
        cache = getattr(view, "_test_tiled_payload_cache", None)
        if cache is None:
            cache = {}
            view._test_tiled_payload_cache = cache
        if montage_dirty_tiles == () and cache_key in cache:
            payloads = dict(cache[cache_key])
        else:
            previous = dict(cache.get(cache_key, {}))
            rebuild_tiles = None if montage_dirty_tiles is None else {int(tile) for tile in montage_dirty_tiles}
            payloads = previous if previous and rebuild_tiles is not None else {}
            serials = getattr(view, "_test_tiled_payload_serials", None)
            if serials is None:
                serials = {}
                view._test_tiled_payload_serials = serials
            def next_source_id(tile_number: int, source_index: int):
                key = (cache_key, int(tile_number))
                serials[key] = int(serials.get(key, 0)) + 1
                return ("test-tile", int(source_index), int(tile_number), serials[key])

            source = np.asarray(canvas)
            hist = None if histogramData is None else np.asarray(histogramData)
            montage = geometry.montage
            tile_h = int(montage.tile_height)
            tile_w = int(montage.tile_width)
            gap = int(montage.gap)
            for tile_number, source_index in enumerate(tuple(montage.indices)):
                if rebuild_tiles is not None and int(tile_number) not in rebuild_tiles and int(tile_number) in payloads:
                    continue
                state = geometry.montage_tile_states[tile_number]
                if str(getattr(state, "value", state)).lower() != "loaded":
                    payloads.pop(int(tile_number), None)
                    continue
                row = tile_number // int(montage.columns)
                column = tile_number % int(montage.columns)
                y0 = row * (tile_h + gap) - int(geometry.montage_origin_y)
                x0 = column * (tile_w + gap) - int(geometry.montage_origin_x)
                if y0 < 0 or x0 < 0 or y0 + tile_h > source.shape[0] or x0 + tile_w > source.shape[1]:
                    payloads.pop(int(tile_number), None)
                    continue
                image = source[y0 : y0 + tile_h, x0 : x0 + tile_w]
                tile_hist = None if hist is None else hist[y0 : y0 + tile_h, x0 : x0 + tile_w]
                payloads[int(tile_number)] = DisplayTilePayload(
                    tile_number,
                    int(source_index),
                    image,
                    tile_hist,
                    next_source_id(tile_number, source_index),
                )
            cache[cache_key] = dict(payloads)
    if tile_delta is None:
        if montage_dirty_tiles is None:
            upserts = payloads
        elif montage_dirty_tiles == ():
            upserts = {}
        else:
            upserts = {int(tile): payloads[int(tile)] for tile in montage_dirty_tiles if int(tile) in payloads}
        revision = 1 if upserts else 2
        tile_delta = TilePresentationDelta(
            structure_revision=revision,
            payload_revision=revision,
            visibility_revision=revision,
            level_revision=revision,
            histogram_revision=revision,
            viewport_revision=revision,
            upserts=upserts,
            active_tiles=tuple(payloads),
            planned_tiles=tuple(payloads),
        )
    return view.setTiledPresentation(
        geometry=geometry,
        tile_state=TilePresentationState(payloads),
        tile_delta=tile_delta,
        histogramPlotData=histogramPlotData,
        levels=levels,
        histogramRange=histogramRange,
        viewport_policy=ViewportPolicy.PRESERVE if viewport_policy is None else viewport_policy,
        rgb_already_windowed=rgb_already_windowed,
        tile_residency_budget_bytes=tile_residency_budget_bytes,
        frame_plan=frame_plan,
    )


def _single_tile_geometry(canvas):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState

    shape = tuple(int(value) for value in np.shape(canvas)[:2])
    return DisplayGeometry(
        view_state=ViewState.from_shape(shape).with_image_axes(0, 1),
        display_shape=shape,
        montage=MontageGeometry(indices=(0,), tile_shape=shape, columns=1, rows=1, gap=0),
        montage_tile_states=(MontageTileState.LOADED,),
    )


def test_profile_marker_callback_replacement_and_programmatic_move(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    calls = []

    view.set_profile_marker_callback(lambda x, y: calls.append(("first", x, y)))
    view.set_profile_marker_callback(lambda x, y: calls.append(("second", x, y)))

    view.setProfileMarker(1, 2, visible=True)
    assert calls == []

    view._profile_vline.setValue(3)
    assert len(calls) == 1
    assert calls[0][0] == "second"

    view.clear_profile_marker_callback()
    view._profile_vline.setValue(4)
    assert len(calls) == 1
    view.close()


def test_evaluation_overlay_and_stale_opacity(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.arange(16, dtype=float).reshape(4, 4))
    view.show()
    qt_app.processEvents()

    view.setImageStale(True)
    view.setEvaluationOverlay(True, "Updating view...")

    assert view.imageItem.opacity() == 0.55
    assert view._evaluation_overlay.isVisible()
    assert view._evaluation_overlay.text() == "Updating view..."

    view.setImageStale(False)
    view.setEvaluationOverlay(False)
    assert view.imageItem.opacity() == 1.0
    assert not view._evaluation_overlay.isVisible()
    view.close()


def test_montage_tile_overlays_reuse_single_graphics_item(qt_app):
    from arrayscope.display.imageview2d import ImageView2D, MontageTileOverlay
    from pyqtgraph.Qt import QtCore

    view = ImageView2D()
    view.setImage(np.zeros((8, 8), dtype=float))
    first = (MontageTileOverlay(0, 0, 4, 4, "loading", "Loading"),)
    second = (
        MontageTileOverlay(0, 0, 4, 4, "loading", "Loading"),
        MontageTileOverlay(4, 0, 4, 4, "skipped", "Skipped"),
    )

    view.setMontageTileOverlays(first)
    item = view._montage_tile_overlay_item
    view.setMontageTileOverlays(second)

    assert view._montage_tile_overlay_item is item
    assert view.montageTileOverlayCount() == 2
    assert len(view._montage_tile_overlay_items) == 1
    assert item.acceptedMouseButtons() == QtCore.Qt.MouseButton.NoButton
    assert not item.acceptHoverEvents()
    view.close()


def test_tile_truth_overlay_shows_backend_neutral_rows(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from pyqtgraph.Qt import QtCore

    view = ImageView2D()
    try:
        view.resize(400, 300)
        view.setImage(np.zeros((4, 4), dtype=np.float32))
        view.show()
        qt_app.processEvents()
        view.setTileTruthOverlayRows(
            (
                {
                    "tile": 4,
                    "tile_rect": (0, 0, 4, 4),
                    "target_source": 17,
                    "acknowledged_source": None,
                    "drawable": False,
                    "target_texture_kind": "complex_rg32f",
                    "acknowledged_texture_kind": None,
                    "target_channel": "real",
                    "target_complex_mapping": ("scalar", "real", "mapped"),
                    "target_lod": {"level": 0},
                    "acknowledged_lod": None,
                    "target_semantic_generation": "('sem', 2)",
                    "acknowledged_semantic_generation": None,
                    "levels_generation": 7,
                },
            )
        )

        layer = view._tile_truth_overlay_layer
        assert len(layer.labels) == 1
        assert not layer.labels[0].isHidden()
        assert "slot 4  LOAD" in layer.labels[0].text()
        assert "src 17 -> None" in layer.labels[0].text()
        assert "real  scalar/real/mapped" in layer.labels[0].text()
        assert layer.labels[0].testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        view.setTileTruthOverlayRows(())
        assert layer.labels[0].isHidden()
    finally:
        view.close()


def test_update_image_data_fast_preserves_levels_and_view_range(qt_app, monkeypatch):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((8, 8), dtype=float), levels=(0.0, 10.0))
    view.getView().setRange(xRange=(1, 5), yRange=(2, 6), padding=0)
    before_range = view.getView().viewRange()
    autolevel_calls = []
    monkeypatch.setattr(view, "autoLevels", lambda: autolevel_calls.append(True))

    view.updateImageDataFast(np.ones((8, 8), dtype=float), histogramData=np.ones((8, 8), dtype=float), levels=(0.0, 10.0))

    assert autolevel_calls == []
    assert tuple(view.image.shape) == (8, 8)
    assert view.getLevels() == (0.0, 10.0)
    assert view.getView().viewRange() == before_range
    view.close()


def test_image_presentation_keeps_levels_and_histogram_range_separate(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    _present_tiled(view,
        np.zeros((4, 4), dtype=float),
        histogramData=np.zeros((4, 4), dtype=float),
        levels=(2.0, 8.0),
        histogramRange=(0.0, 10.0),
    )

    assert tuple(float(value) for value in view.getLevels()) == (2.0, 8.0)
    assert view.getHistogramDataBounds() == (0.0, 10.0)

    _present_tiled(view,
        np.full((4, 4), 1000.0, dtype=float),
        histogramData=np.full((4, 4), 1000.0, dtype=float),
        levels=(2.0, 8.0),
        histogramRange=(0.0, 10.0),
    )

    assert tuple(float(value) for value in view.getLevels()) == (2.0, 8.0)
    assert view.getHistogramDataBounds() == (0.0, 10.0)
    view.close()


def test_tile_commit_report_uses_backend_presented_tile_ids_for_middle_holes():
    from types import SimpleNamespace

    from arrayscope.display.imageview2d import _tile_commit_report
    from arrayscope.display.model.tile_stats import TileLayerUpdateStats

    payloads = {0: object(), 1: object(), 2: object()}
    stats = TileLayerUpdateStats(
        visible_items=2,
        presented_tiles=(0, 2),
    )

    report = _tile_commit_report(payloads, SimpleNamespace(removals=()), stats)

    assert report.presented_tiles == frozenset({0, 2})


def test_tile_commit_report_preserves_backend_presented_ids_outside_delta_payloads():
    from types import SimpleNamespace

    from arrayscope.display.imageview2d import _tile_commit_report
    from arrayscope.display.model.tile_stats import TileLayerUpdateStats

    payloads = {0: object()}
    delta = SimpleNamespace(upserts={0: object()}, removals=(), base_revision=4, target_revision=5)
    stats = TileLayerUpdateStats(
        visible_items=3,
        presented_tiles=(0, 1, 2),
        committed_upserts=(0,),
    )

    report = _tile_commit_report(payloads, delta, stats)

    assert report.presented_tiles == frozenset({0, 1, 2})
    assert report.committed_upserts == frozenset({0})


def test_vispy_requested_presented_tiles_use_active_scope_not_delta_subset():
    from types import SimpleNamespace

    from arrayscope.display.vispy_imageview2d import _requested_direct_payload_tiles

    payloads = {0: object()}
    delta = SimpleNamespace(active_tiles=(0, 1, 2), upserts=payloads)

    assert _requested_direct_payload_tiles(payloads, delta) == {0, 1, 2}


def test_pyqtgraph_tile_commit_report_counts_distinct_updated_tiles():
    from types import SimpleNamespace

    from arrayscope.display.imageview2d import _tile_commit_report, _tile_layer_distinct_work_items
    from arrayscope.display.model.tile_stats import TileLayerUpdateStats

    payloads = {
        index: SimpleNamespace(nbytes=1024)
        for index in range(3)
    }
    stats = TileLayerUpdateStats(
        visible_items=3,
        presented_tiles=(0, 1, 2),
        committed_upserts=(0, 1, 2),
        updated_tiles=(0, 1, 2),
        items_created=3,
        items_updated=3,
    )

    report = _tile_commit_report(payloads, SimpleNamespace(upserts=payloads, removals=()), stats)

    assert _tile_layer_distinct_work_items(stats) == 3
    assert report.cold_count == 3
    assert report.texture_uploads == 3
    assert report.pyqtgraph_items_created == 0


def test_update_image_data_fast_accepts_display_ready_rgb(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((4, 4, 3), dtype=np.uint8), levels=(0.0, 1.0), histogramData=np.zeros((4, 4)))
    rgb = np.full((4, 4, 3), 128, dtype=np.uint8)

    view.updateImageDataFast(rgb, histogramData=np.ones((4, 4)), levels=(0.0, 1.0), rgb_already_windowed=True)

    assert view._rgbBaseImage is None
    np.testing.assert_array_equal(view.imageDisp, rgb)
    view.close()



def test_large_histogram_refresh_uses_background_submitter(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    submitted = []

    def submit(fn, *, on_done, key):
        submitted.append((fn, on_done, key))
        return SimpleNamespace(scheduled=True)

    _present_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    _clear_histogram_jobs(view)
    view.setBackgroundTaskSubmitter(submit)
    submitted.clear()

    assert view._refresh_histogram_plot(auto_level=False) is None
    assert submitted
    assert submitted[-1][2][0] == "histogram_plot"
    view.close()


def test_histogram_result_cache_reuses_completed_signature_with_current_generation(qt_app):
    from arrayscope.display.histogram_controller import HistogramPlotResult, histogram_plot_request_for_view
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    try:
        _present_tiled(view, data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        controller = view._histogram_display_controller
        item = view.histogram.item
        first = histogram_plot_request_for_view(
            item.imageItem(),
            item,
            histogram_bounds=view.getHistogramDataBounds(),
            generation=1,
        )
        assert first is not None
        result = HistogramPlotResult(
            generation=1,
            source_identity=first.source_identity,
            view_signature=first.view_signature,
            x=np.asarray([0.0, 0.5, 1.0]),
            y=np.asarray([1.0, 2.0, 1.0]),
        )
        controller._remember_histogram_result(result)
        second = histogram_plot_request_for_view(
            item.imageItem(),
            item,
            histogram_bounds=view.getHistogramDataBounds(),
            generation=7,
        )

        cached = controller._cached_histogram_result(second)

        assert cached is not None
        assert cached.generation == 7
        np.testing.assert_array_equal(cached.x, result.x)
        np.testing.assert_array_equal(cached.y, result.y)
    finally:
        view.close()


def test_large_histogram_auto_level_applies_bounds_before_refinement(qt_app, monkeypatch):
    from arrayscope.display import histogram_controller
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.linspace(5.0, 15.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    submitted = []

    def submit(fn, *, on_done, key):
        submitted.append((fn, on_done, key))
        return SimpleNamespace(scheduled=True)

    def fail_sync_compute(_request):
        raise AssertionError("large auto-window refinement should not run synchronously")

    monkeypatch.setattr(histogram_controller, "compute_histogram_plot", fail_sync_compute)

    _present_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(5.0, 15.0))
    _clear_histogram_jobs(view)
    view.setBackgroundTaskSubmitter(submit)
    submitted.clear()

    assert view._histogram_display_controller.refresh_histogram_plot(auto_level=True) is True
    assert tuple(float(value) for value in view.getLevels()) == (5.0, 15.0)
    assert submitted
    assert submitted[-1][2][0] == "histogram_plot"
    view.close()


def test_large_histogram_refinement_coalesces_matching_background_request(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    submitted = []

    def submit(fn, *, on_done, key):
        submitted.append((fn, on_done, key))
        return SimpleNamespace(scheduled=True)

    _present_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    _clear_histogram_jobs(view)
    view.setBackgroundTaskSubmitter(submit)
    submitted.clear()

    assert view._histogram_display_controller.refresh_histogram_plot(auto_level=False) is True
    assert view._histogram_display_controller.refresh_histogram_plot(auto_level=False) is True
    assert len(submitted) == 1
    view.close()


def test_large_histogram_refinement_replaces_pending_changed_request(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    submitted = []

    def submit(fn, *, on_done, key):
        submitted.append((fn, on_done, key))
        return SimpleNamespace(scheduled=True)

    _present_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    _clear_histogram_jobs(view)
    view.setBackgroundTaskSubmitter(submit)
    submitted.clear()

    controller = view._histogram_display_controller
    assert controller.refresh_histogram_plot(auto_level=False) is True
    view.setHistogramDataBounds((0.0, 2.0))
    assert controller.refresh_histogram_plot(auto_level=False) is True
    view.setHistogramDataBounds((0.0, 3.0))
    assert controller.refresh_histogram_plot(auto_level=False) is True

    assert len(submitted) == 1

    submitted[0][1](submitted[0][0]())
    qt_app.processEvents()

    assert len(submitted) == 2
    assert controller._running_request_signature == controller._active_request_signature
    view.close()


def test_large_histogram_stale_result_after_newer_range_does_not_update_plot(qt_app, monkeypatch):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    submitted = []

    def submit(fn, *, on_done, key):
        submitted.append((fn, on_done, key))
        return SimpleNamespace(scheduled=True)

    _present_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    _clear_histogram_jobs(view)
    view.setBackgroundTaskSubmitter(submit)
    submitted.clear()
    controller = view._histogram_display_controller

    assert controller.refresh_histogram_plot(auto_level=False) is True
    stale_result = submitted[0][0]()
    view.setHistogramDataBounds((0.0, 2.0))
    assert controller.refresh_histogram_plot(auto_level=False) is True
    monkeypatch.setattr(view.histogram.item.plot, "setData", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("stale histogram result applied")))

    submitted[0][1](stale_result)
    qt_app.processEvents()

    assert len(submitted) == 2
    view.close()


def test_large_histogram_close_ignores_late_result_and_clears_pending(qt_app, monkeypatch):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    submitted = []

    def submit(fn, *, on_done, key):
        submitted.append((fn, on_done, key))
        return SimpleNamespace(scheduled=True)

    _present_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    _clear_histogram_jobs(view)
    view.setBackgroundTaskSubmitter(submit)
    submitted.clear()
    controller = view._histogram_display_controller

    assert controller.refresh_histogram_plot(auto_level=False) is True
    view.setHistogramDataBounds((0.0, 2.0))
    assert controller.refresh_histogram_plot(auto_level=False) is True
    result = submitted[0][0]()
    monkeypatch.setattr(view.histogram.item.plot, "setData", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("late histogram result applied after close")))

    view.close()
    submitted[0][1](result)
    qt_app.processEvents()

    assert controller._closed is True
    assert controller._pending_request is None


def test_repeated_fast_updates_do_not_rebind_same_histogram_item(qt_app, monkeypatch):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.ones((4, 4), dtype=float)
    _present_tiled(view,data, histogramData=data, histogramPlotData=np.arange(16, dtype=float), levels=(0.0, 2.0), histogramRange=(0.0, 2.0))
    calls = []
    monkeypatch.setattr(view.histogram, "setImageItem", lambda item: calls.append(item))

    _present_tiled(view,data, histogramData=data, histogramPlotData=np.arange(16, dtype=float), levels=(0.0, 2.0), histogramRange=(0.0, 2.0))
    _present_tiled(view,data, histogramData=data, histogramPlotData=np.arange(16, dtype=float), levels=(0.0, 2.0), histogramRange=(0.0, 2.0))

    assert calls == []
    view.close()






def test_tiled_single_tile_patch_without_histogram_plot_skips_payload_histogram_aggregation(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    view = ImageView2D()
    try:
        _present_tiled(
            view,
            np.zeros((2, 5), dtype=np.float32),
            histogramData=None,
            histogramPlotData=np.arange(4, dtype=np.float32),
            geometry=_single_tile_geometry(np.zeros((2, 5), dtype=np.float32)),
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
        )

        single_hist = np.full((2, 2), 42.0, dtype=np.float32)
        payloads = {
            0: DisplayTilePayload(0, 0, np.zeros((2, 2), dtype=np.float32), single_hist, ("single", 0)),
        }
        _present_tiled(
            view,
            np.zeros((2, 2), dtype=np.float32),
            histogramData=None,
            histogramPlotData=None,
            geometry=_single_tile_geometry(np.zeros((2, 2), dtype=np.float32)),
            montage_tile_payloads=payloads,
            levels=(40.0, 45.0),
            histogramRange=(40.0, 45.0),
        )

        assert view.histogramSource is None
        assert view.histogramPlotSource is None
    finally:
        view.close()


def test_typed_tiled_single_plane_uses_real_pyqtgraph_items(qt_app):
    from arrayscope.core.scheduler import FrameTarget
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.backend_contract import PYQTGRAPH_CAPABILITIES
    from arrayscope.display.frame_planner import FramePlanner
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.viewport import ViewportPolicy

    state = ViewState.from_shape((4, 4)).with_image_axes(0, 1)
    frame_plan = FramePlanner(internal_tile_shape=(2, 2)).plan(
        target=FrameTarget("semantic", "viewport", "presentation", "exact-visible"),
        view_state=state,
        display_shape=(4, 4),
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
    )
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    payloads = {
        region.region_id: DisplayTilePayload(
            region.region_id,
            region.region_id,
            image[region.data_slices],
            image[region.data_slices],
            ("single", region.region_id),
        )
        for region in frame_plan.regions
    }
    tile_state = TilePresentationState(payloads, revision=1)
    tile_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        base_revision=0,
        target_revision=1,
        upserts=payloads,
        active_tiles=frame_plan.active_region_ids,
        planned_tiles=frame_plan.planned_region_ids,
        near_tiles=frame_plan.near_region_ids,
    )
    view = ImageView2D()
    try:
        report = view.setTiledPresentation(
            geometry=frame_plan.geometry,
            tile_state=tile_state,
            tile_delta=tile_delta,
            histogramPlotData=None,
            levels=(0.0, 15.0),
            histogramRange=(0.0, 15.0),
            viewport_policy=ViewportPolicy.PRESERVE,
            frame_plan=frame_plan,
        )

        assert view.montageDisplayMode() == "tile_layer"
        assert sorted(report.presented_tiles) == [0, 1, 2, 3]
        assert sorted(report.accepted_upserts(tile_delta)) == [0, 1, 2, 3]
        states = view._montage_tile_layer.states
        assert set(states) == {0, 1, 2, 3}
        assert states[0].item.pos().x() == 0.0
        assert states[1].item.pos().x() == 2.0
        assert states[2].item.pos().y() == 2.0
        np.testing.assert_array_equal(states[3].item.image, image[2:4, 2:4])
    finally:
        view.close()


def test_tiled_presentation_does_not_budget_ready_payload_visibility(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 3)).with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"),
        display_shape=(2, 8),
        montage=MontageGeometry(indices=(0, 1, 2), tile_shape=(2, 2), columns=3, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,) * 3,
    )
    payloads = {
        index: DisplayTilePayload(
            index,
            index,
            np.full((2, 2), float(index + 1), dtype=np.float32),
            None,
            ("payload", index),
        )
        for index in range(3)
    }
    state = TilePresentationState(payloads)
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=(0, 1, 2),
        planned_tiles=(0, 1, 2),
    )

    report = view.setTiledPresentation(
        geometry=geometry,
        tile_state=state,
        tile_delta=delta,
        histogramPlotData=None,
        levels=(0.0, 3.0),
        histogramRange=(0.0, 3.0),
    )

    assert report.presented_tiles == frozenset({0, 1, 2})
    view.close()


def test_scalar_tiled_level_delta_acknowledges_without_image_replacement(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    payloads = {
        index: DisplayTilePayload(
            index,
            index,
            np.full((2, 2), float(index + 1), dtype=np.float32),
            None,
            ("payload", index),
        )
        for index in range(2)
    }
    initial_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
    )
    report = view.setTiledPresentation(
        geometry=geometry,
        tile_state=TilePresentationState(payloads),
        tile_delta=initial_delta,
        histogramPlotData=None,
        levels=(0.0, 3.0),
        histogramRange=(0.0, 3.0),
    )
    before_images = {
        tile: state.item.image
        for tile, state in view._montage_tile_layer.states.items()
    }
    level_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=2,
        visibility_revision=1,
        level_revision=2,
        histogram_revision=1,
        viewport_revision=1,
        base_revision=1,
        target_revision=2,
        upserts=payloads,
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
    )

    view.setTiledPresentation(
        geometry=geometry,
        tile_state=TilePresentationState(payloads),
        tile_delta=level_delta,
        histogramPlotData=None,
        levels=(0.5, 2.5),
        histogramRange=(0.0, 3.0),
    )

    timing = view.lastImageUploadTiming()
    assert report.committed_upserts == frozenset({0, 1})
    assert timing.tile_layer_level_updates == 2
    assert timing.tile_layer_items_updated == 0
    assert timing.tile_layer_image_replacements == 0
    assert timing.visible_bytes == 0
    for tile, state in view._montage_tile_layer.states.items():
        assert state.item.image is before_images[tile]
        assert tuple(float(value) for value in state.item.levels) == (0.5, 2.5)
    view.close()



def test_first_typed_tiled_commit_applies_payload_pixels_and_levels_before_autolevel(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    left = np.full((2, 2), 10.0, dtype=np.float32)
    right = np.full((2, 2), 20.0, dtype=np.float32)
    payloads = {
        0: DisplayTilePayload(0, 0, left, left, ("payload", 0)),
        1: DisplayTilePayload(1, 1, right, right, ("payload", 1)),
    }
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
    )

    view.setTiledPresentation(
        geometry=geometry,
        tile_state=TilePresentationState(payloads),
        tile_delta=delta,
        histogramPlotData=np.array([10.0, 20.0], dtype=np.float32),
        levels=(0.0, 25.0),
        histogramRange=(0.0, 25.0),
    )

    states = view._montage_tile_layer.states
    assert set(states) == {0, 1}
    np.testing.assert_array_equal(states[0].item.image, left)
    np.testing.assert_array_equal(states[1].item.image, right)
    assert tuple(float(value) for value in states[0].item.levels) == (0.0, 25.0)
    assert tuple(float(value) for value in states[1].item.levels) == (0.0, 25.0)
    view.close()


def test_pyqtgraph_tiled_retarget_updates_shifted_active_payloads(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    first_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 3)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    shifted_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 3)).with_montage_axis(2, columns=2, indices=(1, 2), text="1:3"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(1, 2), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    images = {
        index: np.full((2, 2), float(index + 1), dtype=np.float32)
        for index in range(3)
    }
    first_payloads = {
        0: DisplayTilePayload(0, 0, images[0], None, ("payload", 0)),
        1: DisplayTilePayload(1, 1, images[1], None, ("payload", 1)),
    }
    shifted_payloads = {
        0: DisplayTilePayload(0, 1, images[1], None, ("payload", 1)),
        1: DisplayTilePayload(1, 2, images[2], None, ("payload", 2)),
    }
    try:
        view.setTiledPresentation(
            geometry=first_geometry,
            tile_state=TilePresentationState(first_payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=first_payloads,
                active_tiles=(0, 1),
                planned_tiles=(0, 1),
            ),
            histogramPlotData=None,
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
        )
        original_overlap_item = view._montage_tile_layer.states[1].item

        report = view.setTiledPresentation(
            geometry=shifted_geometry,
            tile_state=TilePresentationState(shifted_payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=2,
                payload_revision=2,
                visibility_revision=2,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=2,
                upserts=shifted_payloads,
                active_tiles=(0, 1),
                planned_tiles=(0, 1),
            ),
            histogramPlotData=None,
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
        )

        states = view._montage_tile_layer.states
        timing = view.lastImageUploadTiming()
        assert report.presented_tiles == frozenset({0, 1})
        np.testing.assert_array_equal(states[0].item.image, images[1])
        np.testing.assert_array_equal(states[1].item.image, images[2])
        assert states[0].item is original_overlap_item
        assert states[0].source_index == 1
        assert states[1].source_index == 2
        assert timing.tile_layer_items_updated == 1
        assert timing.tile_layer_relocated_tiles == 1
        assert timing.tile_layer_image_replacements == 1
        assert report.cold_count == 1
        assert report.resident_rebinds == 1
    finally:
        view.close()


def test_pyqtgraph_tiled_retarget_shuffles_lower_range_without_overwrite(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    first_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 3)).with_montage_axis(2, columns=2, indices=(1, 2), text="1:3"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(1, 2), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    shifted_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 3)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    images = {
        index: np.full((2, 2), float(index + 1), dtype=np.float32)
        for index in range(3)
    }
    first_payloads = {
        0: DisplayTilePayload(0, 1, images[1], None, ("payload", 1)),
        1: DisplayTilePayload(1, 2, images[2], None, ("payload", 2)),
    }
    shifted_payloads = {
        0: DisplayTilePayload(0, 0, images[0], None, ("payload", 0)),
        1: DisplayTilePayload(1, 1, images[1], None, ("payload", 1)),
    }
    try:
        view.setTiledPresentation(
            geometry=first_geometry,
            tile_state=TilePresentationState(first_payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=first_payloads,
                active_tiles=(0, 1),
                planned_tiles=(0, 1),
            ),
            histogramPlotData=None,
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
        )
        original_overlap_item = view._montage_tile_layer.states[0].item

        report = view.setTiledPresentation(
            geometry=shifted_geometry,
            tile_state=TilePresentationState(shifted_payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=2,
                payload_revision=2,
                visibility_revision=2,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=2,
                upserts=shifted_payloads,
                active_tiles=(0, 1),
                planned_tiles=(0, 1),
            ),
            histogramPlotData=None,
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
        )

        states = view._montage_tile_layer.states
        timing = view.lastImageUploadTiming()
        assert report.presented_tiles == frozenset({0, 1})
        np.testing.assert_array_equal(states[0].item.image, images[0])
        np.testing.assert_array_equal(states[1].item.image, images[1])
        assert states[1].item is original_overlap_item
        assert states[0].source_index == 0
        assert states[1].source_index == 1
        assert timing.tile_layer_items_updated == 1
        assert timing.tile_layer_relocated_tiles == 1
        assert timing.tile_layer_image_replacements == 1
        assert report.cold_count == 1
        assert report.resident_rebinds == 1
    finally:
        view.close()


def test_pyqtgraph_tiled_retarget_reuses_residents_for_cyclic_reorder(qt_app, monkeypatch):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    first_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 3)).with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"),
        display_shape=(2, 8),
        montage=MontageGeometry(indices=(0, 1, 2), tile_shape=(2, 2), columns=3, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED, MontageTileState.LOADED),
    )
    reordered_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 3)).with_montage_axis(2, columns=3, indices=(2, 0, 1), text="2,0,1"),
        display_shape=(2, 8),
        montage=MontageGeometry(indices=(2, 0, 1), tile_shape=(2, 2), columns=3, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED, MontageTileState.LOADED),
    )
    images = {
        index: np.full((2, 2), float(index + 1), dtype=np.float32)
        for index in range(3)
    }
    first_payloads = {
        tile: DisplayTilePayload(tile, tile, images[tile], None, ("payload", tile))
        for tile in range(3)
    }
    reordered_payloads = {
        0: DisplayTilePayload(0, 2, images[2], None, ("payload", 2)),
        1: DisplayTilePayload(1, 0, images[0], None, ("payload", 0)),
        2: DisplayTilePayload(2, 1, images[1], None, ("payload", 1)),
    }
    try:
        view.setTiledPresentation(
            geometry=first_geometry,
            tile_state=TilePresentationState(first_payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=first_payloads,
                active_tiles=(0, 1, 2),
                planned_tiles=(0, 1, 2),
            ),
            histogramPlotData=None,
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
        )
        original_items_by_source = {
            int(state.source_index): state.item
            for state in view._montage_tile_layer.states.values()
        }
        removed_tiles = []
        original_remove = view._layer_owner.remove_tile_item

        def record_remove(tile_number):
            removed_tiles.append(int(tile_number))
            original_remove(tile_number)

        monkeypatch.setattr(view._layer_owner, "remove_tile_item", record_remove)

        report = view.setTiledPresentation(
            geometry=reordered_geometry,
            tile_state=TilePresentationState(reordered_payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=2,
                payload_revision=2,
                visibility_revision=2,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=2,
                upserts=reordered_payloads,
                active_tiles=(0, 1, 2),
                planned_tiles=(0, 1, 2),
            ),
            histogramPlotData=None,
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
        )

        states = view._montage_tile_layer.states
        timing = view.lastImageUploadTiming()
        assert report.presented_tiles == frozenset({0, 1, 2})
        assert states[0].item is original_items_by_source[2]
        assert states[1].item is original_items_by_source[0]
        assert states[2].item is original_items_by_source[1]
        np.testing.assert_array_equal(states[0].item.image, images[2])
        np.testing.assert_array_equal(states[1].item.image, images[0])
        np.testing.assert_array_equal(states[2].item.image, images[1])
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_relocated_tiles == 3
        assert timing.tile_layer_image_replacements == 0
        assert report.cold_count == 0
        assert report.resident_rebinds == 3
        assert removed_tiles == []
    finally:
        view.close()


def test_pyqtgraph_tiled_active_delta_repairs_resident_retarget_without_explicit_upserts(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 4)).with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text=":"),
        display_shape=(2, 11),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=4, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,) * 4,
    )
    view = ImageView2D()
    images = {
        index: np.full((2, 2), float(index), dtype=np.float32)
        for index in range(4)
    }
    initial_payloads = {
        index: DisplayTilePayload(index, index, images[index], None, ("source", index))
        for index in range(4)
    }
    shifted_payloads = {
        0: DisplayTilePayload(0, 2, images[2], None, ("source", 2)),
        1: DisplayTilePayload(1, 3, images[3], None, ("source", 3)),
    }

    def delta(revision: int, payloads, active_tiles, *, upserts=None):
        return TilePresentationDelta(
            structure_revision=revision,
            payload_revision=revision,
            visibility_revision=revision,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=revision,
            upserts=payloads if upserts is None else upserts,
            active_tiles=tuple(active_tiles),
            planned_tiles=(0, 1, 2, 3),
        )

    try:
        view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(initial_payloads),
            tile_delta=delta(1, initial_payloads, (0, 1, 2, 3)),
            histogramPlotData=None,
            levels=(0.0, 4.0),
            histogramRange=(0.0, 4.0),
        )
        original_by_source = {
            int(state.source_index): state.item
            for state in view._montage_tile_layer.states.values()
        }

        report = view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(shifted_payloads),
            tile_delta=delta(2, shifted_payloads, (0, 1), upserts={}),
            histogramPlotData=None,
            levels=(0.0, 4.0),
            histogramRange=(0.0, 4.0),
        )

        states = view._montage_tile_layer.states
        timing = view.lastImageUploadTiming()
        assert report.presented_tiles == frozenset({0, 1})
        assert states[0].item is original_by_source[2]
        assert states[1].item is original_by_source[3]
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_relocated_tiles == 2
        assert report.resident_rebinds == 2
    finally:
        view.close()


def test_pyqtgraph_fast_scroll_budget_keeps_old_slots_visible(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    first_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 7)).with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text=":"),
        display_shape=(2, 11),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=4, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,) * 4,
    )
    jumped_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 7)).with_montage_axis(2, columns=4, indices=(3, 4, 5, 6), text="3:7"),
        display_shape=(2, 11),
        montage=MontageGeometry(indices=(3, 4, 5, 6), tile_shape=(2, 2), columns=4, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,) * 4,
    )
    images = {
        index: np.full((2, 2), float(index + 1), dtype=np.float32)
        for index in range(7)
    }
    first_payloads = {
        tile: DisplayTilePayload(tile, tile, images[tile], None, ("payload", tile))
        for tile in range(4)
    }
    jumped_payloads = {
        tile: DisplayTilePayload(tile, tile + 3, images[tile + 3], None, ("payload", tile + 3))
        for tile in range(4)
    }
    try:
        view.setTiledPresentation(
            geometry=first_geometry,
            tile_state=TilePresentationState(first_payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=first_payloads,
                active_tiles=(0, 1, 2, 3),
                planned_tiles=(0, 1, 2, 3),
            ),
            histogramPlotData=None,
            levels=(0.0, 7.0),
            histogramRange=(0.0, 7.0),
        )
        original_items = {
            tile: view._montage_tile_layer.states[tile].item
            for tile in range(4)
        }

        report = view.setTiledPresentation(
            geometry=jumped_geometry,
            tile_state=TilePresentationState(jumped_payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=2,
                payload_revision=2,
                visibility_revision=2,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=2,
                upserts=jumped_payloads,
                active_tiles=(0, 1, 2, 3),
                planned_tiles=(0, 1, 2, 3),
                cold_deadline_ms=0.0,
            ),
            histogramPlotData=None,
            levels=(0.0, 7.0),
            histogramRange=(0.0, 7.0),
        )

        states = view._montage_tile_layer.states
        timing = view.lastImageUploadTiming()
        assert report.presented_tiles == frozenset({0, 1, 2, 3})
        assert report.committed_upserts == frozenset({0})
        assert timing.tile_layer_items_updated == 1
        assert set(states) == {0, 1, 2, 3}
        for tile in range(4):
            assert states[tile].visible is True
            assert states[tile].item.image is not None
        np.testing.assert_array_equal(states[0].item.image, images[3])
        assert states[1].item is original_items[1]
        assert states[2].item is original_items[2]
        assert states[3].item is original_items[3]
    finally:
        view.close()


def test_pyqtgraph_complex_fast_scroll_budget_keeps_presentable_slots(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.shader_mapping import TexturePlaneKind

    view = ImageView2D()
    first_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 7)).with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text=":"),
        display_shape=(2, 11),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=4, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,) * 4,
    )
    jumped_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 7)).with_montage_axis(2, columns=4, indices=(3, 4, 5, 6), text="3:7"),
        display_shape=(2, 11),
        montage=MontageGeometry(indices=(3, 4, 5, 6), tile_shape=(2, 2), columns=4, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,) * 4,
    )

    def payload(tile: int, source: int) -> DisplayTilePayload:
        rgb = np.full((2, 2, 3), (40 + source * 20) % 255, dtype=np.uint8)
        hist = np.full((2, 2), float(source + 1), dtype=np.float32)
        semantic = np.full((2, 2), complex(source + 1, source + 2), dtype=np.complex64)
        return DisplayTilePayload(
            tile,
            source,
            rgb,
            hist,
            ("complex-source", source, "shader", None),
            texture_data=semantic,
            texture_kind=TexturePlaneKind.COMPLEX_RG32F,
            semantic_data=semantic,
            semantic_histogram_data=hist,
            rgb_windowed_levels=(0.0, 7.0),
        )

    first_payloads = {tile: payload(tile, tile) for tile in range(4)}
    jumped_payloads = {tile: payload(tile, tile + 3) for tile in range(4)}
    try:
        view.setTiledPresentation(
            geometry=first_geometry,
            tile_state=TilePresentationState(first_payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=first_payloads,
                active_tiles=(0, 1, 2, 3),
                planned_tiles=(0, 1, 2, 3),
            ),
            histogramPlotData=None,
            levels=(0.0, 7.0),
            histogramRange=(0.0, 7.0),
            rgb_already_windowed=False,
        )
        original_items = {
            tile: view._montage_tile_layer.states[tile].item
            for tile in range(4)
        }

        report = view.setTiledPresentation(
            geometry=jumped_geometry,
            tile_state=TilePresentationState(jumped_payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=2,
                payload_revision=2,
                visibility_revision=2,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=2,
                upserts=jumped_payloads,
                active_tiles=(0, 1, 2, 3),
                planned_tiles=(0, 1, 2, 3),
                cold_deadline_ms=0.0,
            ),
            histogramPlotData=None,
            levels=(0.0, 7.0),
            histogramRange=(0.0, 7.0),
            rgb_already_windowed=False,
        )

        states = view._montage_tile_layer.states
        timing = view.lastImageUploadTiming()
        assert report.presented_tiles == frozenset({0, 1, 2, 3})
        assert report.committed_upserts == frozenset({0})
        assert timing.tile_layer_items_updated == 1
        assert timing.tile_layer_rgb_window_tiles == 0
        assert set(states) == {0, 1, 2, 3}
        for tile in range(4):
            assert states[tile].visible is True
            assert states[tile].item.image is not None
        assert states[1].item is original_items[1]
        assert states[2].item is original_items[2]
        assert states[3].item is original_items[3]
    finally:
        view.close()


def test_pyqtgraph_rgb8_preview_tile_stays_windowed_after_zoom_without_exact_payload(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.lod import LodInfo
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.shader_mapping import TexturePlaneKind

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((8, 8, 1)).with_montage_axis(2, columns=1, indices=(0,), text=":"),
        display_shape=(8, 8),
        montage=MontageGeometry(indices=(0,), tile_shape=(8, 8), columns=1, rows=1, gap=0),
        montage_tile_states=(MontageTileState.LOADED,),
    )
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    histogram = np.array([[0.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    payload = DisplayTilePayload(
        0,
        0,
        rgb,
        histogram,
        ("preview-rgb", 0),
        texture_kind=TexturePlaneKind.RGB8,
        source_shape=(8, 8),
        lod=LodInfo(level=2, factor=4, source_shape=(8, 8), texture_shape=(2, 2), gutter=0),
        quality="preview",
    )
    try:
        view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState({0: payload}),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts={0: payload},
                active_tiles=(0,),
                planned_tiles=(0,),
            ),
            histogramPlotData=None,
            levels=(0.0, 8.0),
            histogramRange=(0.0, 8.0),
            rgb_already_windowed=False,
        )
        view.getView().setRange(xRange=(0.0, 4.0), yRange=(0.0, 4.0), padding=0)
        before = view._montage_tile_layer.states[0].item.image.copy()

        view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState({0: payload}),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=2,
                histogram_revision=1,
                viewport_revision=2,
                upserts={},
                active_tiles=(0,),
                planned_tiles=(0,),
            ),
            histogramPlotData=None,
            levels=(2.0, 6.0),
            histogramRange=(0.0, 8.0),
            rgb_already_windowed=False,
        )

        state = view._montage_tile_layer.states[0]
        timing = view.lastImageUploadTiming()
        # Contract (field defect 2026-07): a preview/LOD plane that carries
        # its reduced histogram MUST rewindow on level changes — freezing it
        # at evaluation-time (provisional) levels leaves resident-LOD tiles
        # permanently mis-windowed. Cheap: the plane is reduced-size.
        assert timing.tile_layer_rgb_window_tiles == 1
        assert timing.tile_layer_image_replacements == 0
        assert tuple(float(value) for value in state.levels) == (2.0, 6.0)
        assert state.lod_scale == (4.0, 4.0)
        assert state.item.image.shape == (2, 2, 3)
        assert not np.array_equal(state.item.image, before)
    finally:
        view.close()


def test_pyqtgraph_clean_typed_tiled_commit_stays_noop(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2), 1.0, dtype=np.float32), None, ("payload", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2), 2.0, dtype=np.float32), None, ("payload", 1)),
    }
    try:
        view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=payloads,
                active_tiles=(0, 1),
                planned_tiles=(0, 1),
            ),
            histogramPlotData=None,
            levels=(0.0, 2.0),
            histogramRange=(0.0, 2.0),
        )

        report = view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts={},
                active_tiles=(0, 1),
                planned_tiles=(0, 1),
            ),
            histogramPlotData=None,
            levels=(0.0, 2.0),
            histogramRange=(0.0, 2.0),
        )

        timing = view.lastImageUploadTiming()
        assert report.presented_tiles == frozenset({0, 1})
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_image_replacements == 0
        assert timing.visible_bytes == 0
    finally:
        view.close()


def test_pyqtgraph_level_redraw_is_bounded_by_commit_deadline(qt_app, monkeypatch):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.backends.pyqtgraph import tiles as pyqtgraph_tiles
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 3)).with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"),
        display_shape=(2, 8),
        montage=MontageGeometry(indices=(0, 1, 2), tile_shape=(2, 2), columns=3, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,) * 3,
    )
    payloads = {
        tile: DisplayTilePayload(
            tile,
            tile,
            np.full((2, 2, 3), float(tile + 1), dtype=np.float32),
            np.full((2, 2), float(tile + 1), dtype=np.float32),
            ("payload", tile),
        )
        for tile in range(3)
    }
    try:
        view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=payloads,
                active_tiles=(0, 1, 2),
                planned_tiles=(0, 1, 2),
            ),
            histogramPlotData=None,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            rgb_already_windowed=False,
        )

        # Level re-windowing is bounded by a floored refinement deadline
        # (max(8 ms, cold budget)), not by the collapsed cold budget itself.
        # Advance a fake clock 5 ms per call so the floor binds after the
        # first re-windowed tile regardless of machine speed.
        fake_now = {"value": 0.0}

        def advancing_perf_counter():
            fake_now["value"] += 0.005
            return fake_now["value"]

        monkeypatch.setattr(pyqtgraph_tiles, "perf_counter", advancing_perf_counter)

        report = view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=2,
                histogram_revision=1,
                viewport_revision=1,
                upserts=payloads,
                active_tiles=(0, 1, 2),
                planned_tiles=(0, 1, 2),
                cold_deadline_ms=0.0,
            ),
            histogramPlotData=None,
            levels=(0.0, 2.0),
            histogramRange=(0.0, 2.0),
            rgb_already_windowed=False,
        )

        timing = view.lastImageUploadTiming()
        assert report.presented_tiles == frozenset({0, 1, 2})
        assert len(report.committed_upserts or ()) == 1
        assert timing.tile_layer_rgb_window_tiles == 1
        assert timing.tile_layer_level_updates == 1
        assert timing.tile_layer_level_update_pending_items == 2
    finally:
        view.close()


def test_pyqtgraph_clean_typed_tiled_relayout_moves_existing_items(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    payloads = {
        index: DisplayTilePayload(
            index,
            index,
            np.full((2, 2), float(index), dtype=np.float32),
            None,
            ("payload", index),
        )
        for index in range(4)
    }
    first_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 4)).with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text=":"),
        display_shape=(2, 11),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=4, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,) * 4,
    )
    relaid_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 4)).with_montage_axis(2, columns=2, indices=(0, 1, 2, 3), text=":"),
        display_shape=(5, 5),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=2, rows=2, gap=1),
        montage_tile_states=(MontageTileState.LOADED,) * 4,
    )
    try:
        view.setTiledPresentation(
            geometry=first_geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=payloads,
                active_tiles=(0, 1, 2, 3),
                planned_tiles=(0, 1, 2, 3),
            ),
            histogramPlotData=None,
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
        )
        assert view._montage_tile_layer.states[2].item.pos().x() == 6.0
        assert view._montage_tile_layer.states[2].item.pos().y() == 0.0

        report = view.setTiledPresentation(
            geometry=relaid_geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=2,
                payload_revision=1,
                visibility_revision=2,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=2,
                upserts={},
                active_tiles=(0, 1, 2, 3),
                planned_tiles=(0, 1, 2, 3),
            ),
            histogramPlotData=None,
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
        )

        timing = view.lastImageUploadTiming()
        assert report.presented_tiles == frozenset({0, 1, 2, 3})
        assert view._montage_tile_layer.states[2].item.pos().x() == 0.0
        assert view._montage_tile_layer.states[2].item.pos().y() == 3.0
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_relocated_tiles == 2
        assert timing.visible_bytes == 0
    finally:
        view.close()


def test_pyqtgraph_budgeted_tiled_payload_keeps_existing_item_visible(qt_app, monkeypatch):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    first_payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2), 1.0, dtype=np.float32), None, ("first", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2), 2.0, dtype=np.float32), None, ("first", 1)),
    }
    second_payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2), 10.0, dtype=np.float32), None, ("second", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2), 20.0, dtype=np.float32), None, ("second", 1)),
    }
    try:
        view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(first_payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=first_payloads,
                active_tiles=(0, 1),
                planned_tiles=(0, 1),
            ),
            histogramPlotData=None,
            levels=(0.0, 20.0),
            histogramRange=(0.0, 20.0),
        )
        original_item = view._montage_tile_layer.states[1].item
        removed = []
        monkeypatch.setattr(view._layer_owner, "remove_tile_item", lambda tile: removed.append(int(tile)))

        report = view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(second_payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=2,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=second_payloads,
                active_tiles=(0, 1),
                planned_tiles=(0, 1),
                cold_deadline_ms=0.0,
            ),
            histogramPlotData=None,
            levels=(0.0, 20.0),
            histogramRange=(0.0, 20.0),
        )

        assert report.presented_tiles == frozenset({0, 1})
        assert report.committed_upserts == frozenset({0})
        assert 1 in view._montage_tile_layer.states
        np.testing.assert_array_equal(view._montage_tile_layer.states[0].item.image, second_payloads[0].image)
        np.testing.assert_array_equal(view._montage_tile_layer.states[1].item.image, first_payloads[1].image)
        assert view._montage_tile_layer.states[1].item is original_item
        assert view._montage_tile_layer.states[1].visible is True
        assert removed == []
    finally:
        view.close()


def test_pyqtgraph_hidden_tile_reactivates_from_resident_pool_without_upload(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    images = {
        0: np.full((2, 2), 1.0, dtype=np.float32),
        1: np.full((2, 2), 2.0, dtype=np.float32),
    }
    payloads = {
        index: DisplayTilePayload(index, index, images[index], None, ("resident", index))
        for index in range(2)
    }
    try:
        view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=payloads,
                active_tiles=(0, 1),
                planned_tiles=(0, 1),
            ),
            histogramPlotData=None,
            levels=(0.0, 2.0),
            histogramRange=(0.0, 2.0),
        )
        original_item = view._montage_tile_layer.states[0].item

        view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=2,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=2,
                upserts={},
                active_tiles=(1,),
                planned_tiles=(0, 1),
            ),
            histogramPlotData=None,
            levels=(0.0, 2.0),
            histogramRange=(0.0, 2.0),
        )
        assert view._montage_tile_layer.states[0].item is original_item
        assert view._montage_tile_layer.states[0].item.isVisible() is False
        assert original_item.scene() is not None

        report = view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=2,
                visibility_revision=3,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=3,
                upserts={0: payloads[0]},
                active_tiles=(0,),
                planned_tiles=(0, 1),
            ),
            histogramPlotData=None,
            levels=(0.0, 2.0),
            histogramRange=(0.0, 2.0),
        )

        timing = view.lastImageUploadTiming()
        assert report.presented_tiles == frozenset({0})
        assert report.committed_upserts == frozenset({0})
        assert view._montage_tile_layer.states[0].item is original_item
        assert timing.tile_layer_items_created == 0
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_existing_items_shown == 1
        assert report.cold_count == 0
        assert report.resident_rebinds == 1
    finally:
        view.close()


def test_pyqtgraph_warm_residency_prepares_invisible_item_without_committing(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    payload = DisplayTilePayload(1, 1, np.full((2, 2), 9.0, dtype=np.float32), None, ("warm", 1))
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts={},
        active_tiles=(),
        planned_tiles=(0, 1),
        near_tiles=(1,),
    )
    try:
        stats = view.warmTiledResidency(
            payloads={1: payload},
            geometry=geometry,
            levels=(0.0, 10.0),
            tile_delta=delta,
        )

        assert 1 in view._montage_tile_layer.states
        assert len(view._montage_tile_layer._direct_reuse_pool) == 1
        assert view._montage_tile_layer._direct_reuse_pool[0].item.isVisible() is False
        assert stats.items_created == 1
        assert stats.items_updated == 1
        assert stats.visible_items == 0
        assert stats.resident_items == 1
        assert stats.warm_resident_items == 1
    finally:
        view.close()


def test_pyqtgraph_residency_budget_evicts_inactive_fifo_without_evicting_active(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((4, 4, 3)).with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"),
        display_shape=(4, 14),
        montage=MontageGeometry(indices=(0, 1, 2), tile_shape=(4, 4), columns=3, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,) * 3,
    )
    payloads = {
        index: DisplayTilePayload(index, index, np.full((4, 4), float(index), dtype=np.float32), None, ("budget", index))
        for index in range(3)
    }
    try:
        view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=payloads,
                active_tiles=(0, 1, 2),
                planned_tiles=(0, 1, 2),
            ),
            histogramPlotData=None,
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
        )
        active_item = view._montage_tile_layer.states[2].item

        view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=2,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=2,
                upserts={},
                active_tiles=(2,),
                planned_tiles=(0, 1, 2),
            ),
            histogramPlotData=None,
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
            tile_residency_budget_bytes=payloads[2].image.nbytes,
        )

        timing = view.lastImageUploadTiming()
        assert set(view._montage_tile_layer.states) == {2}
        assert view._montage_tile_layer.states[2].item is active_item
        assert timing.tile_layer_storage_evictions == 2
        assert timing.tile_layer_resident_items == 1
        assert timing.tile_layer_warm_resident_items == 0
    finally:
        view.close()



def test_tile_layer_items_use_world_positions_when_display_origin_shifted(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 4)).with_montage_axis(2, columns=2, indices=(0, 1, 2, 3), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=2, rows=2, gap=1),
        montage_origin_x=0,
        montage_origin_y=3,
        montage_tile_states=(MontageTileState.UNLOADED, MontageTileState.UNLOADED, MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.arange(10, dtype=float).reshape(2, 5)

    _present_tiled(view,
        canvas,
        histogramData=canvas.copy(),
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 9.0),
        histogramRange=(0.0, 9.0),
    )

    states = view._montage_tile_layer.states
    assert set(states) == {2, 3}
    assert states[2].item.pos().y() == 3.0
    assert states[3].item.pos().x() == 3.0
    assert states[3].item.pos().y() == 3.0
    view.close()


def test_graphics_layer_z_order_places_tools_above_tiles(qt_app):
    from arrayscope.core.roi import RoiKind
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D, MontageTileOverlay
    from arrayscope.display.layers import Z_MONTAGE_LOADING_OVERLAY, Z_PROFILE_MARKER, Z_ROI, Z_TILE_IMAGE
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 1)).with_montage_axis(2, columns=1, indices=(0,), text=":"),
        display_shape=(2, 2),
        montage=MontageGeometry(indices=(0,), tile_shape=(2, 2), columns=1, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,),
    )
    canvas = np.zeros((2, 2), dtype=float)
    _present_tiled(view,canvas, histogramData=canvas, histogramPlotData=None, geometry=geometry, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    roi = view.createRoi(RoiKind.RECTANGLE, rect=(0, 0, 1, 1))
    view.setProfileMarker(0, 0, visible=True)
    view.setMontageTileOverlays((MontageTileOverlay(0, 0, 1, 1, "loading", "Loading"),))

    tile_z = view._montage_tile_layer.states[0].item.zValue()
    roi_z = view._roi_items[roi.id][0].zValue()
    assert tile_z == Z_TILE_IMAGE
    assert roi_z == Z_ROI
    assert view._profile_handle.zValue() == Z_PROFILE_MARKER
    assert view._montage_tile_overlay_item.zValue() == Z_MONTAGE_LOADING_OVERLAY
    assert tile_z < roi_z < view._profile_handle.zValue() < view._montage_tile_overlay_item.zValue()
    view.close()


def test_inactive_tile_layer_items_are_hidden_residents(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry_loaded = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.zeros((2, 5), dtype=float)
    _present_tiled(view,canvas, histogramData=canvas, histogramPlotData=None, geometry=geometry_loaded, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    removed_item = view._montage_tile_layer.states[1].item
    geometry_one = DisplayGeometry(
        view_state=geometry_loaded.view_state,
        display_shape=(2, 5),
        montage=geometry_loaded.montage,
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.UNLOADED),
    )

    _present_tiled(view,canvas, histogramData=canvas, histogramPlotData=None, geometry=geometry_one, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))

    assert view._montage_tile_layer.states[1].item is removed_item
    assert removed_item.isVisible() is False
    assert removed_item.scene() is not None
    view.close()


def test_tile_layer_clean_commit_skips_tile_and_histogram_uploads(qt_app, monkeypatch):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.arange(10, dtype=np.float32).reshape(2, 5)
    hist = canvas.copy()
    plot = np.arange(16, dtype=np.float32)

    _present_tiled(view,
        canvas,
        histogramData=hist,
        histogramPlotData=plot,
        geometry=geometry,
        levels=(0.0, 9.0),
        histogramRange=(0.0, 9.0),
    )

    calls = []
    for state in view._montage_tile_layer.states.values():
        monkeypatch.setattr(state.item, "setImage", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(view.histogramImageItem, "setImage", lambda *args, **kwargs: calls.append((args, kwargs)))

    _present_tiled(view,
        canvas,
        histogramData=hist,
        histogramPlotData=plot,
        geometry=geometry,
        levels=(0.0, 9.0),
        histogramRange=(0.0, 9.0),
        montage_dirty_tiles=(),
    )

    timing = view.lastImageUploadTiming()
    assert calls == []
    assert timing.tile_layer_visible_items == 2
    assert timing.tile_layer_items_updated == 0
    assert timing.tile_layer_upload_ms == 0.0
    assert timing.visible_bytes == 0
    assert timing.histogram_bytes == 0
    view.close()


def test_tile_layer_dirty_commit_updates_only_dirty_tile(qt_app, monkeypatch):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.arange(10, dtype=np.float32).reshape(2, 5)
    hist = canvas.copy()
    _present_tiled(view,canvas, histogramData=hist, histogramPlotData=None, geometry=geometry, levels=(0.0, 9.0), histogramRange=(0.0, 9.0))

    calls = []
    for tile_number, state in view._montage_tile_layer.states.items():
        monkeypatch.setattr(state.item, "setImage", lambda *args, tile_number=tile_number, **kwargs: calls.append(tile_number))

    canvas[:, 3:5] = 99
    _present_tiled(view,
        canvas,
        histogramData=hist,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 9.0),
        histogramRange=(0.0, 9.0),
        montage_dirty_tiles=(1,),
    )

    assert calls == [1]
    timing = view.lastImageUploadTiming()
    assert timing.tile_layer_items_updated == 1
    view.close()


def test_rgb_tile_layer_clean_commit_does_not_rewindow(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.full((2, 5, 3), 200, dtype=np.uint8)
    hist = np.linspace(0.0, 1.0, 10, dtype=np.float32).reshape(2, 5)
    _present_tiled(view,canvas, histogramData=hist, histogramPlotData=None, geometry=geometry, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))

    for state in view._montage_tile_layer.states.values():
        assert state.rgb_base is not None
        assert state.rgb_base.dtype == np.float32

    _present_tiled(view,
        canvas,
        histogramData=hist,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        montage_dirty_tiles=(),
    )

    timing = view.lastImageUploadTiming()
    assert timing.tile_layer_rgb_window_tiles == 0
    assert timing.rgb_window_ms == 0.0
    assert timing.tile_layer_rgb_window_ms == 0.0
    assert timing.tile_layer_upload_ms == 0.0
    assert timing.tile_layer_items_updated == 0
    view.close()




def test_tile_layer_requires_typed_payloads(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 1)).with_montage_axis(2, columns=1, indices=(0,), text=":"),
        display_shape=(2, 2),
        montage=MontageGeometry(indices=(0,), tile_shape=(2, 2), columns=1, rows=1, gap=0),
        montage_tile_states=(MontageTileState.LOADED,),
    )

    with pytest.raises(ValueError, match="typed tile payloads"):
        view._update_montage_tile_layer_items(
            np.zeros((2, 2), dtype=np.float32),
            histogramData=None,
            geometry=geometry,
            levels=(0.0, 1.0),
            rgb_already_windowed=False,
            montage_dirty_tiles=None,
            montage_tile_source_ids=None,
            montage_tile_payloads=None,
            tile_delta=None,
        )
    view.close()




def test_tile_layer_inactive_tile_is_hidden_without_scene_removal(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    loaded = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.full((2, 5, 3), 200, dtype=np.uint8)
    hist = np.linspace(0.0, 1.0, 10, dtype=np.float32).reshape(2, 5)
    _present_tiled(view,canvas, histogramData=hist, histogramPlotData=None, geometry=loaded, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    assert view._montage_tile_layer.states[1].rgb_base is not None
    removed_item = view._montage_tile_layer.states[1].item

    hidden = DisplayGeometry(
        view_state=loaded.view_state,
        display_shape=loaded.display_shape,
        montage=loaded.montage,
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.UNLOADED),
    )
    _present_tiled(view,canvas, histogramData=hist, histogramPlotData=None, geometry=hidden, levels=(0.0, 1.0), histogramRange=(0.0, 1.0), montage_dirty_tiles=())
    assert view._montage_tile_layer.states[1].item is removed_item
    assert removed_item.isVisible() is False
    assert removed_item.scene() is not None
    view.close()


def test_display_ready_rgb_tile_layer_level_change_keeps_uint8_item_levels(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 1)).with_montage_axis(2, columns=1, indices=(0,), text=":"),
        display_shape=(2, 2),
        montage=MontageGeometry(indices=(0,), tile_shape=(2, 2), columns=1, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,),
    )
    canvas = np.full((2, 2, 3), 128, dtype=np.uint8)
    hist = np.linspace(0.0, 1.0, 4, dtype=np.float32).reshape(2, 2)
    _present_tiled(view,
        canvas,
        histogramData=hist,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=True,
    )

    _present_tiled(view,
        canvas,
        histogramData=hist,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.5, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=True,
        montage_dirty_tiles=(),
    )

    state = view._montage_tile_layer.states[0]
    assert tuple(float(value) for value in state.item.levels) == (0.0, 255.0)
    assert view.lastImageUploadTiming().tile_layer_items_updated == 0
    view.close()



def test_display_ready_rgb_histogram_levels_do_not_rewindow_display(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
    hist = np.linspace(0.0, 1.0, 16, dtype=float).reshape(4, 4)
    _present_tiled(view,
        rgb,
        histogramData=hist,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=True,
    )
    before = np.array(view.imageDisp, copy=True)

    view.setLevels(0.5, 1.0)

    assert view._rgbBaseImage is None
    np.testing.assert_array_equal(view.imageDisp, before)
    view.close()


def test_scalar_image_upload_passes_levels_to_image_item(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(2.0, 12.0))

    assert tuple(float(value) for value in view.imageItem.levels) == (2.0, 12.0)
    view.updateImageDataFast(np.ones((4, 4), dtype=float), levels=(3.0, 9.0))
    assert tuple(float(value) for value in view.imageItem.levels) == (3.0, 9.0)
    view.close()


def test_scalar_histogram_level_drag_updates_display_item(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(0.0, 15.0))

    view.histogram.setLevels(2.0, 8.0)
    view._on_histogram_levels_changed()

    assert tuple(float(value) for value in view.imageItem.levels) == (2.0, 8.0)
    view.close()


def test_value_at_display_mapping_ignores_mismatched_histogram_source(qt_app):
    from arrayscope.display.geometry import ViewPointMapping
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.ones((4, 4), dtype=float), histogramData=np.ones((4, 4), dtype=float), levels=(0.0, 2.0))
    view.histogramSource = np.ones((8, 8), dtype=float)

    value = view.valueAtDisplayMapping(ViewPointMapping(view_x=0, view_y=0, display_x=0, display_y=0, local_x=0, local_y=0, array_index=(0, 0)))

    assert value is None
    view.close()


def test_value_at_display_mapping_uses_display_coordinates_for_montage(qt_app):
    from arrayscope.display.geometry import ViewPointMapping
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.arange(12, dtype=float).reshape(3, 4)
    view.setImage(data, histogramData=data.copy(), levels=(0.0, 12.0))

    value = view.valueAtDisplayMapping(
        ViewPointMapping(
            view_x=3,
            view_y=2,
            display_x=3,
            display_y=2,
            local_x=0,
            local_y=0,
            array_index=(0, 0, 1),
            tile_number=1,
            montage_axis=2,
            montage_index=1,
        )
    )

    assert value == data[2, 3]
    view.close()


def test_profile_marker_bounds_update_when_image_shape_changes(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((8, 10), dtype=float))
    view.setProfileMarker(9, 7, visible=True)
    assert view._profile_vline.maxRange == (0, 9)
    assert view._profile_hline.maxRange == (0, 7)

    view.setImage(np.zeros((4, 5), dtype=float))
    assert view._profile_vline.maxRange == (0, 4)
    assert view._profile_hline.maxRange == (0, 3)
    assert view.profileMarkerPosition() == (4.0, 3.0)
    view.close()


def test_profile_marker_lines_hide_when_marker_center_leaves_view(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((20, 20), dtype=float))
    view.getView().setRange(xRange=(0, 10), yRange=(0, 10), padding=0)
    view.setProfileMarker(5, 5, visible=True)

    assert view._profile_handle.isVisible()
    assert view._profile_vline.isVisible()
    assert view._profile_hline.isVisible()

    view.getView().setRange(xRange=(10, 19), yRange=(10, 19), padding=0)

    assert not view._profile_handle.isVisible()
    assert not view._profile_vline.isVisible()
    assert not view._profile_hline.isVisible()
    assert view.profileMarkerPosition() == (5.0, 5.0)
    view.close()


def test_imageview_creates_polyline_and_freehand_rois(qt_app):
    from arrayscope.core.roi import RoiKind
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((10, 12), dtype=float))
    created = []
    view.roiCreated.connect(lambda selection: created.append(selection))

    polyline = view.createRoi(RoiKind.POLYLINE, points=((1, 1), (5, 2), (8, 7)))
    freehand = view.createRoi(RoiKind.FREEHAND_POLYGON, points=((2, 2), (7, 2), (7, 8), (2, 8)))

    assert len(created) == 2
    assert polyline.geometry.kind.value == RoiKind.POLYLINE.value
    assert freehand.geometry.kind.value == RoiKind.FREEHAND_POLYGON.value
    assert freehand.geometry.points[0] == freehand.geometry.points[-1]
    assert len(view.roiSelections()) == 2

    assert view.removeRoi(polyline.id)
    assert len(view.roiSelections()) == 1
    view.clearRois()
    assert view.roiSelections() == ()
    view.close()


def test_imageview_freehand_requires_points(qt_app):
    from arrayscope.core.roi import RoiKind
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((10, 12), dtype=float))

    with pytest.raises(ValueError, match="freehand ROI requires a drag path"):
        view.createRoi(RoiKind.FREEHAND_POLYGON)
    view.close()


def test_persistent_polyline_and_freehand_tools_do_not_start_drawing(qt_app):
    from pyqtgraph.Qt import QtCore, QtGui

    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((10, 12), dtype=float))
    view.setInspectionTool("roi_freehand")
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPointF(1, 1),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )

    assert not view._handle_roi_drawing_event(event)
    assert view.beginRoiDrawingOnce("roi_freehand")
    assert view._handle_roi_drawing_event(event)
    view.cancelPendingRoiDrawing()
    view.close()


def test_imageview_inspection_tool_validation(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setInspectionTool("roi_freehand")
    assert view.inspectionTool() == "roi_freehand"

    with pytest.raises(ValueError, match="unknown inspection tool"):
        view.setInspectionTool("bad")
    view.close()


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_roi_drag_is_owned_by_shared_pointer_lifecycle(qt_app, backend):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.roi import RoiKind
    from arrayscope.display.interaction import PointerPhase

    view = _view_class(backend)()
    view.resize(320, 260)
    view.show()
    view.setImage(np.zeros((20, 20), dtype=float))
    view.getView().setRange(xRange=(0, 20), yRange=(0, 20), padding=0)
    selection = view.createRoi(RoiKind.RECTANGLE, rect=(2.0, 3.0, 4.0, 5.0))
    changed = []
    view.roiChanged.connect(lambda roi_id, geometry: changed.append((roi_id, geometry)))

    assert _send_viewport_mouse(view, QtCore.QEvent.Type.MouseButtonPress, (4.0, 5.0), button=QtCore.Qt.MouseButton.LeftButton)
    assert _send_viewport_mouse(
        view,
        QtCore.QEvent.Type.MouseMove,
        (6.0, 7.0),
        buttons=QtCore.Qt.MouseButton.LeftButton,
    )

    assert view.interactionState().phase is PointerPhase.DRAGGING
    assert view.interactionState().capture.object_id == selection.id
    assert changed[-1][1].rect == pytest.approx((4.0, 5.0, 4.0, 5.0), abs=0.06)

    assert _send_viewport_mouse(view, QtCore.QEvent.Type.MouseButtonRelease, (6.0, 7.0), button=QtCore.Qt.MouseButton.LeftButton)
    assert view.interactionState().phase is PointerPhase.IDLE
    assert dict((roi.id, roi) for roi in view.roiSelections())[selection.id].geometry.rect == pytest.approx((4.0, 5.0, 4.0, 5.0), abs=0.06)
    view.close()


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_out_of_bounds_roi_can_be_dragged_back_into_content(qt_app, backend):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.roi import RoiKind
    from arrayscope.display.interaction import PointerPhase

    view = _view_class(backend)()
    view.resize(360, 260)
    view.show()
    view.setImage(np.zeros((10, 10), dtype=float))
    view.getView().setRange(xRange=(0, 24), yRange=(0, 10), padding=0)
    selection = view.createRoi(RoiKind.RECTANGLE, rect=(20.0, 2.0, 2.0, 3.0))
    changed = []
    view.roiChanged.connect(lambda roi_id, geometry: changed.append((roi_id, geometry)))

    assert _send_viewport_mouse(view, QtCore.QEvent.Type.MouseButtonPress, (21.0, 3.0), button=QtCore.Qt.MouseButton.LeftButton)
    assert view.interactionState().phase is PointerPhase.DRAGGING
    assert view.interactionState().capture.object_id == selection.id

    assert _send_viewport_mouse(
        view,
        QtCore.QEvent.Type.MouseMove,
        (9.0, 3.0),
        buttons=QtCore.Qt.MouseButton.LeftButton,
    )

    assert changed[-1][1].rect == pytest.approx((7.0, 2.0, 2.0, 3.0), abs=0.06)
    assert _send_viewport_mouse(view, QtCore.QEvent.Type.MouseButtonRelease, (9.0, 3.0), button=QtCore.Qt.MouseButton.LeftButton)
    assert view.interactionState().phase is PointerPhase.IDLE
    assert dict((roi.id, roi) for roi in view.roiSelections())[selection.id].geometry.rect == pytest.approx((7.0, 2.0, 2.0, 3.0), abs=0.06)
    view.close()


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_set_roi_selections_preserves_id_counter_for_next_roi(qt_app, backend):
    from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection

    view = _view_class(backend)()
    view.setImage(np.zeros((20, 20), dtype=float))
    view.setRoiSelections(
        (
            RoiSelection(
                "roi-3",
                "ROI 3",
                RoiGeometry(RoiKind.RECTANGLE, rect=(2.0, 3.0, 4.0, 5.0)),
            ),
        )
    )

    next_roi = view.createRoi(RoiKind.RECTANGLE, rect=(1.0, 1.0, 2.0, 2.0))

    assert next_roi.id == "roi-4"
    view.close()


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_profile_drag_is_owned_by_shared_pointer_lifecycle(qt_app, backend):
    from pyqtgraph.Qt import QtCore

    from arrayscope.display.interaction import PointerPhase

    view = _view_class(backend)()
    view.resize(320, 260)
    view.show()
    view.setImage(np.zeros((20, 20), dtype=float))
    view.getView().setRange(xRange=(0, 20), yRange=(0, 20), padding=0)
    view.setProfileMarker(5.0, 6.0, visible=True)
    moved = []
    view.set_profile_marker_callback(lambda x, y: moved.append((x, y)))

    assert _send_viewport_mouse(view, QtCore.QEvent.Type.MouseButtonPress, (5.0, 8.0), button=QtCore.Qt.MouseButton.LeftButton)
    assert _send_viewport_mouse(
        view,
        QtCore.QEvent.Type.MouseMove,
        (8.0, 8.0),
        buttons=QtCore.Qt.MouseButton.LeftButton,
    )

    state = view.interactionState()
    assert state.phase is PointerPhase.DRAGGING
    assert state.capture.kind == "profile"
    assert state.capture.part == "vertical"
    assert state.drag_profile_position == pytest.approx((8.0, 6.0), abs=0.06)
    assert len(moved) == 1
    assert moved[-1] == pytest.approx((8.0, 6.0), abs=0.06)

    assert _send_viewport_mouse(view, QtCore.QEvent.Type.MouseButtonRelease, (8.0, 8.0), button=QtCore.Qt.MouseButton.LeftButton)
    assert view.interactionState().phase is PointerPhase.IDLE
    assert view.profileMarkerPosition() == pytest.approx((8.0, 6.0), abs=0.06)
    assert len(moved) == 1
    view.close()


@pytest.mark.parametrize(
    ("action", "reason"),
    (
        ("mode-change", "tool-change"),
        ("frame-replacement", "frame-replacement"),
        ("target-removal", "target-removed"),
        ("window-deactivate", "window-deactivate"),
    ),
)
def test_pointer_capture_is_cancelled_by_interruptions(qt_app, action, reason):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.roi import RoiKind
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.interaction import InteractionTarget, PointerPhase

    view = ImageView2D()
    try:
        view.setImage(np.zeros((20, 20), dtype=float))
        selection = view.createRoi(RoiKind.RECTANGLE, rect=(2.0, 3.0, 4.0, 5.0))
        assert view._begin_pointer_capture(
            InteractionTarget("roi", object_id=selection.id, part="body", geometry_kind="rectangle"),
            (4.0, 5.0),
        )
        assert view.interactionState().phase is PointerPhase.DRAGGING

        if action == "mode-change":
            view.setInspectionTool("profile")
        elif action == "frame-replacement":
            view.setImage(np.ones((20, 20), dtype=float))
        elif action == "target-removal":
            view.removeRoi(selection.id)
        elif action == "window-deactivate":
            view.event(QtCore.QEvent(QtCore.QEvent.Type.WindowDeactivate))

        assert view.interactionState().phase is PointerPhase.IDLE
        assert view.interactionState().capture is None
        assert view.interactionState().last_cancel_reason == reason
    finally:
        view.close()


def test_pointer_capture_is_cancelled_by_close(qt_app):
    from arrayscope.core.roi import RoiKind
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.interaction import InteractionTarget, PointerPhase

    view = ImageView2D()
    view.setImage(np.zeros((20, 20), dtype=float))
    selection = view.createRoi(RoiKind.RECTANGLE, rect=(2.0, 3.0, 4.0, 5.0))
    assert view._begin_pointer_capture(
        InteractionTarget("roi", object_id=selection.id, part="body", geometry_kind="rectangle"),
        (4.0, 5.0),
    )

    view.close()

    assert view.interactionState().phase is PointerPhase.IDLE
    assert view.interactionState().capture is None


@pytest.mark.parametrize("phase", ("armed", "drawing"))
def test_frame_replacement_cancels_roi_drawing_lifecycle(qt_app, phase):
    from pyqtgraph.Qt import QtCore

    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.interaction import PointerPhase

    view = ImageView2D()
    try:
        view.resize(320, 260)
        view.show()
        view.setImage(np.zeros((20, 20), dtype=float))
        view.getView().setRange(xRange=(0, 20), yRange=(0, 20), padding=0)
        assert view.beginRoiDrawingOnce("roi_freehand")
        if phase == "drawing":
            assert _send_viewport_mouse(
                view,
                QtCore.QEvent.Type.MouseButtonPress,
                (4.0, 5.0),
                button=QtCore.Qt.MouseButton.LeftButton,
            )
            assert view.interactionState().phase is PointerPhase.DRAWING
        else:
            assert view.interactionState().phase is PointerPhase.DRAWING_ARMED

        view.setImage(np.ones((20, 20), dtype=float))

        assert view.interactionState().phase is PointerPhase.IDLE
        assert view.interactionState().pending_draw_tool is None
        assert view.interactionState().drawing_points == ()
        assert view.interactionState().last_cancel_reason == "frame-replacement"
    finally:
        view.close()


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_pointer_hit_testing_ignores_margin_outside_committed_frame(qt_app, backend):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.roi import RoiKind
    from arrayscope.display.interaction import PointerPhase

    view = _view_class(backend)()
    try:
        view.resize(320, 260)
        view.show()
        view.setImage(np.zeros((20, 20), dtype=float))
        view.getView().setRange(xRange=(0, 20), yRange=(0, 20), padding=0)
        view.createRoi(RoiKind.RECTANGLE, rect=(0.0, 3.0, 4.0, 5.0))

        for image_point in ((-10.0, 5.0), (-0.05, 5.0)):
            handled = _send_viewport_mouse(
                view,
                QtCore.QEvent.Type.MouseButtonPress,
                image_point,
                button=QtCore.Qt.MouseButton.LeftButton,
            )

            assert handled is False
            assert view.interactionState().phase is PointerPhase.IDLE
            assert view.interactionState().capture is None
    finally:
        view.close()


def test_pyqtgraph_roi_and_profile_visuals_mirror_interaction_state(qt_app):
    from arrayscope.core.roi import RoiKind
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.interaction import InteractionTarget

    view = ImageView2D()
    try:
        view.setImage(np.zeros((20, 20), dtype=float))
        selection = view.createRoi(RoiKind.RECTANGLE, rect=(2.0, 3.0, 4.0, 5.0))
        item, _stored = view._roi_items[selection.id]
        base_width = item.pen.widthF()

        state = view.interaction_controller.set_hover(
            InteractionTarget("roi", object_id=selection.id, part="body", geometry_kind="rectangle"),
            point=(4.0, 5.0),
        )
        view.sync_interaction_state(state)
        assert item.pen.widthF() > base_width

        view.setProfileMarker(5.0, 6.0, visible=True)
        state = view.interaction_controller.set_hover(InteractionTarget("profile", part="center"), point=(5.0, 6.0))
        view.sync_interaction_state(state)
        assert view._interaction_visual_profile_part == "center"
    finally:
        view.close()


def test_imageview_preserves_view_range_for_same_shape_by_default(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((8, 10), dtype=float))
    view.getView().setRange(xRange=(2, 5), yRange=(1, 6), padding=0)
    before = view.getView().viewRange()

    view.setImage(np.ones((8, 10), dtype=float))
    after = view.getView().viewRange()

    assert after == before
    view.close()


def test_imageview_resets_view_range_when_shape_changes(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.viewport import ViewportPolicy

    view = ImageView2D()
    view.setImage(np.zeros((8, 10), dtype=float))
    view.getView().setRange(xRange=(2, 5), yRange=(1, 6), padding=0)

    view.setImage(np.ones((4, 5), dtype=float), viewport_policy=ViewportPolicy.RESET_FOR_NEW_SHAPE)
    after = view.getView().viewRange()

    assert after != [[2.0, 5.0], [1.0, 6.0]]
    view.close()


def test_imageview_replays_auto_intent_only_after_content_extent_changes(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.viewport import ViewportMode

    view = ImageView2D()
    try:
        view.resize(400, 240)
        view.show()
        view.setImage(np.zeros((2, 5), dtype=float))
        qt_app.processEvents()

        before = view.getView().viewRange()
        assert view.setViewportContentExtent((4, 10)) is True
        assert view.getView().viewRange() == before
        assert view.refreshViewportContentExtentIntent() is True
        qt_app.processEvents()
        assert view.viewport_controller.mode == ViewportMode.AUTO_UNTOUCHED
        assert view.viewport_controller.is_near_auto(view.getView().viewRange())
        assert view.setViewportContentExtent((4, 10)) is False

        view.getView().setRange(xRange=(2.0, 4.0), yRange=(1.0, 3.0), padding=0)
        qt_app.processEvents()
        assert view.viewport_controller.mode == ViewportMode.USER
        user_range = view.getView().viewRange()
        assert view.setViewportContentExtent((8, 20)) is True
        assert view.refreshViewportContentExtentIntent() is False
        assert view.getView().viewRange() == user_range
    finally:
        view.close()


def test_imageview_manual_resize_preserves_screen_zoom_after_layout(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.viewport import ViewportMode

    view = ImageView2D()
    try:
        view.resize(500, 400)
        view.show()
        qt_app.processEvents()
        view.setImage(np.zeros((100, 100), dtype=float))
        qt_app.processEvents()
        view.getView().setRange(xRange=(-50.0, 150.0), yRange=(-50.0, 150.0), padding=0)
        qt_app.processEvents()
        view.viewport_controller.mode = ViewportMode.USER
        before_size = view.graphicsView.viewport().size()
        before = view.getView().viewRange()

        view.resize(300, 400)
        qt_app.processEvents()

        after_size = view.graphicsView.viewport().size()
        after = view.getView().viewRange()
        before_x_units = (before[0][1] - before[0][0]) / before_size.width()
        before_y_units = (before[1][1] - before[1][0]) / before_size.height()
        after_x_units = (after[0][1] - after[0][0]) / after_size.width()
        after_y_units = (after[1][1] - after[1][0]) / after_size.height()
        assert after_x_units == pytest.approx(before_x_units)
        assert after_y_units == pytest.approx(before_y_units)
    finally:
        view.close()


def test_imageview_limits_zoom_out_to_recoverable_content(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        view.resize(400, 400)
        view.setImage(np.zeros((100, 100), dtype=float))

        view.getView().setRange(xRange=(-1070.0, 1330.0), yRange=(-850.0, 950.0), padding=0)
        x_range, y_range = view.getView().viewRange()

        assert x_range[1] - x_range[0] <= 2000.0 + 1e-6
        assert y_range[1] - y_range[0] <= 2000.0 + 1e-6
        assert (x_range[0] + x_range[1]) * 0.5 == pytest.approx(50.0)
        assert (y_range[0] + y_range[1]) * 0.5 == pytest.approx(50.0)
    finally:
        view.close()


def test_imageview_rejects_extra_zoom_out_at_limit_without_panning(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        view.resize(400, 400)
        view.setImage(np.zeros((100, 100), dtype=float))
        view.getView().setRange(xRange=(-870.0, 1130.0), yRange=(-950.0, 1050.0), padding=0)
        before = view.getView().viewRange()

        view.getView().setRange(xRange=(-2000.0, 4000.0), yRange=(-2200.0, 3800.0), padding=0)
        after = view.getView().viewRange()

        assert after[0] == pytest.approx(before[0])
        assert after[1] == pytest.approx(before[1])
    finally:
        view.close()


def test_imageview_prevents_panning_content_fully_out_of_view(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        view.resize(400, 400)
        view.setImage(np.zeros((100, 100), dtype=float))

        view.getView().setRange(xRange=(200.0, 300.0), yRange=(10.0, 110.0), padding=0)
        x_range, y_range = view.getView().viewRange()

        assert _axis_overlap(x_range, (0.0, 100.0)) >= 5.0 - 1e-6
        assert y_range == pytest.approx((10.0, 110.0))
    finally:
        view.close()


def test_imageview_prevents_vertical_panning_content_fully_out_of_view(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        view.resize(400, 400)
        view.setImage(np.zeros((100, 100), dtype=float))

        view.getView().setRange(xRange=(10.0, 110.0), yRange=(200.0, 300.0), padding=0)
        x_range, y_range = view.getView().viewRange()

        assert x_range[0] < 100.0
        assert _axis_overlap(y_range, (0.0, 100.0)) >= 5.0 - 1e-6
    finally:
        view.close()



def test_programmatic_presentation_does_not_emit_user_level_signal(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    user_calls = []
    view.userLevelsChanged.connect(lambda: user_calls.append(True))

    _present_tiled(view,
        np.zeros((4, 4), dtype=float),
        histogramData=np.zeros((4, 4), dtype=float),
        levels=(2.0, 8.0),
        histogramRange=(0.0, 10.0),
    )
    _present_tiled(view,
        np.ones((4, 4), dtype=float),
        histogramData=np.ones((4, 4), dtype=float),
        levels=(1.0, 9.0),
        histogramRange=(0.0, 10.0),
    )

    assert user_calls == []
    view.close()


def test_explicit_set_levels_emits_user_level_signal(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((4, 4), dtype=float), levels=(0.0, 1.0))
    user_calls = []
    view.userLevelsChanged.connect(lambda: user_calls.append(True))

    view.setLevels(2.0, 8.0)

    assert user_calls == [True]
    assert tuple(float(value) for value in view.getLevels()) == (2.0, 8.0)
    view.close()


def test_histogram_drag_preview_emits_user_level_once_on_finish(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from pyqtgraph.Qt import QtCore

    view = ImageView2D()
    view.setImage(np.zeros((4, 4), dtype=float), levels=(0.0, 1.0))
    user_calls = []
    view.userLevelsChanged.connect(lambda: user_calls.append(True))

    with QtCore.QSignalBlocker(view.histogram.item):
        view.histogram.setLevels(0.1, 0.9)
    view._on_histogram_levels_changed()
    assert user_calls == []
    assert tuple(float(value) for value in view.getLevels()) == (0.1, 0.9)
    with QtCore.QSignalBlocker(view.histogram.item):
        view.histogram.setLevels(0.2, 0.8)
    view._on_histogram_levels_changed()
    assert user_calls == []
    assert tuple(float(value) for value in view.getLevels()) == (0.2, 0.8)
    view._on_histogram_level_change_finished()

    assert user_calls == [True]
    assert tuple(float(value) for value in view.getLevels()) == (0.2, 0.8)
    view.close()


def test_histogram_finish_does_not_repeat_an_already_applied_preview(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from pyqtgraph.Qt import QtCore

    view = ImageView2D()
    view.setImage(np.zeros((4, 4), dtype=float), levels=(0.0, 1.0))
    applied = []
    view._apply_histogram_preview_levels = (
        lambda levels, *, final=False: applied.append((tuple(float(value) for value in levels), bool(final)))
    )

    with QtCore.QSignalBlocker(view.histogram.item):
        view.histogram.setLevels(0.2, 0.8)
    view._on_histogram_levels_changed()
    view._on_histogram_level_change_finished()

    assert applied == [((0.2, 0.8), False)]
    view.close()


def test_histogram_finish_reapplies_target_after_programmatic_level_change(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from pyqtgraph.Qt import QtCore

    view = ImageView2D()
    view.setImage(np.zeros((4, 4), dtype=float), levels=(0.0, 1.0))
    applied = []
    original_apply = view._apply_histogram_preview_levels

    def record(levels, *, final=False):
        applied.append((tuple(float(value) for value in levels), bool(final)))
        original_apply(levels, final=final)

    view._apply_histogram_preview_levels = record
    with QtCore.QSignalBlocker(view.histogram.item):
        view.histogram.setLevels(0.2, 0.8)
    view._on_histogram_levels_changed()
    view._on_histogram_level_change_finished()
    assert applied == [((0.2, 0.8), False)]

    view._apply_display_levels(0.1, 0.9, emit_user=False)
    applied.clear()
    with QtCore.QSignalBlocker(view.histogram.item):
        view.histogram.setLevels(0.2, 0.8)
    # Simulate a finish signal arriving without an intermediate preview signal.
    # The programmatic level update must have invalidated the old preview target.
    view._on_histogram_level_change_finished()

    assert applied == [((0.2, 0.8), True)]
    view.close()


def test_adaptive_histogram_uses_coarser_bins_when_zoomed_out(qt_app):
    from arrayscope.display.histogram_controller import _histogram_value_pixel_height
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        view.resize(480, 520)
        view.show()
        qt_app.processEvents()
        data = np.linspace(0.0, 100.0, 10_000, dtype=float).reshape(100, 100)
        _present_tiled(view,data, histogramData=data, levels=(10.0, 90.0), histogramRange=(0.0, 100.0))
        view.histogram.item.vb.setYRange(0.0, 100.0, padding=0)
        view._refresh_histogram_plot(auto_level=False)

        _x, full_counts = view.histogram.item.plot.getData()
        pixel_height = _histogram_value_pixel_height(view.histogram.item)

        assert len(full_counts) <= int(pixel_height / 5) + 2
    finally:
        view.close()


def test_adaptive_histogram_increases_detail_when_zoomed_in(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        view.resize(480, 520)
        view.show()
        qt_app.processEvents()
        data = np.linspace(0.0, 100.0, 10_000, dtype=float).reshape(100, 100)
        _present_tiled(view,data, histogramData=data, levels=(10.0, 90.0), histogramRange=(0.0, 100.0))

        view.histogram.item.vb.setYRange(0.0, 100.0, padding=0)
        view._refresh_histogram_plot(auto_level=False)
        _x, full_counts = view.histogram.item.plot.getData()

        view.histogram.item.vb.setYRange(45.0, 55.0, padding=0)
        view._refresh_histogram_plot(auto_level=False)
        _x, zoom_counts = view.histogram.item.plot.getData()

        assert len(zoom_counts) > len(full_counts)
        assert len(zoom_counts) <= 500
    finally:
        view.close()


def test_adaptive_histogram_falls_back_for_degenerate_bounds(qt_app):
    from arrayscope.display.histogram_controller import adaptive_histogram_for_view
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        data = np.zeros((8, 8), dtype=float)
        _present_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))

        histogram = adaptive_histogram_for_view(
            view.histogramImageItem,
            view.histogram.item,
            histogram_bounds=(1.0, 1.0),
        )

        assert histogram is not None
        assert len(histogram[0]) >= 2
        assert len(histogram[0]) == len(histogram[1])
    finally:
        view.close()


def test_adaptive_histogram_auto_level_refresh_updates_display_levels(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        data = np.arange(16, dtype=float).reshape(4, 4)
        view.setImage(data, histogramData=data, levels=(2.0, 8.0))

        view._histogram_display_controller.refresh_histogram_plot(auto_level=True)

        assert tuple(float(value) for value in view.getLevels()) == (0.0, 15.0)
        assert tuple(float(value) for value in view.imageItem.levels) == (0.0, 15.0)
    finally:
        view.close()


def test_histogram_native_double_click_between_limits_requests_auto_window(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from pyqtgraph.Qt import QtCore, QtTest

    view = ImageView2D()
    try:
        view.resize(420, 280)
        view.show()
        qt_app.processEvents()
        view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(2.0, 8.0))
        qt_app.processEvents()
        calls = []
        view.autoWindowRequested.connect(lambda: calls.append(True))
        scene_pos = view.histogram.item.vb.mapViewToScene(QtCore.QPointF(0.0, 5.0))
        graphics_view = view.histogram.item.vb.scene().views()[0]
        viewport_pos = graphics_view.mapFromScene(scene_pos)

        QtTest.QTest.mouseDClick(graphics_view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=viewport_pos)
        qt_app.processEvents()

        assert calls == [True]
    finally:
        view.close()


def test_histogram_release_pair_between_limits_requests_auto_window(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from pyqtgraph.Qt import QtCore, QtTest

    view = ImageView2D()

    class _ReleaseEvent:
        def __init__(self, global_pos):
            self._global_pos = QtCore.QPointF(global_pos)
            self.accepted = False

        def type(self):
            return QtCore.QEvent.Type.MouseButtonRelease

        def button(self):
            return QtCore.Qt.MouseButton.LeftButton

        def globalPosition(self):
            return self._global_pos

        def accept(self):
            self.accepted = True

        def isAccepted(self):
            return self.accepted

    try:
        view.resize(420, 280)
        view.show()
        qt_app.processEvents()
        view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(2.0, 8.0))
        qt_app.processEvents()
        calls = []
        view.autoWindowRequested.connect(lambda: calls.append(True))

        first = _ReleaseEvent(QtCore.QPointF(100.0, 200.0))
        assert not view._histogram_display_controller._handle_native_histogram_double_click(
            view.histogram, first
        )
        assert calls == []

        QtTest.QTest.qWait(30)
        second = _ReleaseEvent(QtCore.QPointF(100.0, 200.0))
        assert view._histogram_display_controller._handle_native_histogram_double_click(
            view.histogram, second
        )
        qt_app.processEvents()

        assert second.accepted
        assert calls == [True]
    finally:
        view.close()


def test_histogram_span_edit_waits_for_double_click_window(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

    view = ImageView2D()
    try:
        view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(2.0, 8.0))
        controller = view._histogram_display_controller
        scene_pos = view.histogram.item.vb.mapViewToScene(QtCore.QPointF(0.0, 5.0))

        controller._schedule_span_edit(scene_pos)

        assert controller.active_popup() is None

        controller.request_auto_window()
        QtTest.QTest.qWait(QtWidgets.QApplication.doubleClickInterval() + 20)

        assert controller.active_popup() is None
    finally:
        view.close()


def test_histogram_native_double_click_in_lut_area_requests_auto_window(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from pyqtgraph.Qt import QtCore, QtTest

    view = ImageView2D()
    try:
        view.resize(420, 280)
        view.show()
        qt_app.processEvents()
        view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(2.0, 8.0))
        qt_app.processEvents()
        calls = []
        view.autoWindowRequested.connect(lambda: calls.append(True))
        graphics_view = view.histogram.item.vb.scene().views()[0]
        pos = graphics_view.viewport().rect().center()

        QtTest.QTest.mouseDClick(graphics_view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=pos)
        qt_app.processEvents()

        assert calls == [True]
    finally:
        view.close()


def test_histogram_viewport_double_click_filter_requests_auto_window(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from pyqtgraph.Qt import QtCore, QtGui

    view = ImageView2D()
    try:
        view.resize(420, 280)
        view.show()
        qt_app.processEvents()
        view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(2.0, 8.0))
        qt_app.processEvents()
        calls = []
        view.autoWindowRequested.connect(lambda: calls.append(True))
        viewport = view.histogram.viewport()
        pos = QtCore.QPointF(viewport.rect().center())
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonDblClick,
            pos,
            pos,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        handled = qt_app.sendEvent(viewport, event)
        qt_app.processEvents()

        assert handled
        assert event.isAccepted()
        assert calls == [True]
    finally:
        view.close()


def test_histogram_limit_popup_live_preview_accepts_once(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(0.0, 10.0))
        user_calls = []
        view.userLevelsChanged.connect(lambda: user_calls.append(True))

        view._histogram_display_controller.begin_limit_edit("lower")
        popup = view._histogram_display_controller.active_popup()
        popup.edit.setValue(2.0)

        assert user_calls == []
        assert tuple(float(value) for value in view.getLevels()) == (2.0, 10.0)

        popup.accept()

        assert user_calls == [True]
        assert tuple(float(value) for value in view.getLevels()) == (2.0, 10.0)
    finally:
        view.close()


def test_histogram_limit_popup_escape_restores_without_user_signal(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(0.0, 10.0))
        user_calls = []
        view.userLevelsChanged.connect(lambda: user_calls.append(True))

        view._histogram_display_controller.begin_limit_edit("upper")
        popup = view._histogram_display_controller.active_popup()
        popup.edit.setValue(7.0)
        assert tuple(float(value) for value in view.getLevels()) == (0.0, 7.0)

        popup.reject()

        assert user_calls == []
        assert tuple(float(value) for value in view.getLevels()) == (0.0, 10.0)
    finally:
        view.close()


def test_histogram_span_popup_keeps_clicked_value_anchored(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from pyqtgraph.Qt import QtCore

    view = ImageView2D()
    try:
        view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(0.0, 10.0))
        scene_pos = view.histogram.item.vb.mapViewToScene(QtCore.QPointF(0.0, 2.0))

        view._histogram_display_controller.begin_span_edit(scene_pos)
        popup = view._histogram_display_controller.active_popup()
        popup.edit.setValue(20.0)

        assert tuple(round(float(value), 6) for value in view.getLevels()) == (-2.0, 18.0)
        popup.accept()
    finally:
        view.close()


def _axis_overlap(view_range, content_range) -> float:
    view_start, view_end = sorted((float(view_range[0]), float(view_range[1])))
    content_start, content_end = sorted((float(content_range[0]), float(content_range[1])))
    return max(0.0, min(view_end, content_end) - max(view_start, content_start))



def test_close_cancels_queued_histogram_refresh(qt_app, monkeypatch):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    calls = []
    monkeypatch.setattr(
        view._histogram_display_controller,
        "refresh_histogram_plot",
        lambda **_kwargs: calls.append(True),
    )

    view._histogram_display_controller.schedule_refresh()
    view.close()
    qt_app.processEvents()

    assert calls == []


def test_noop_tile_layer_timing_does_not_arm_physical_draw(qt_app):
    """A skipped tiled commit is not a new frame waiting to be painted."""

    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        view._start_upload_timing("tile_layer")
        view._finish_upload_timing()

        assert not view.presentationDrawPending()
    finally:
        view.close()
