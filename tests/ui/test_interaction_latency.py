import numpy as np
import pytest
from dataclasses import replace

from arrayscope.display.slice_engine import DisplayImage
from arrayscope.operations.evaluator import EvaluationResult
from tests.ui.helpers import clear_arrayscope_settings, process_events


_WAIT_TIMEOUT_MS = 5000


def _tile_result(tile, value):
    image = np.full((tile.height, tile.width), value, dtype=np.float32)
    return EvaluationResult(DisplayImage(image, histogram_data=image.copy()), 0.0, image.shape, int(image.nbytes))


def test_rapid_slice_burst_is_coalesced_and_latest(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=np.float32).reshape(4, 5, 8))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win, "render", lambda **kwargs: calls.append((kwargs, win.view_state.slice_indices[2])))
    try:
        win._on_slice_index_changed(2, 1)
        win._on_slice_index_changed(2, 2)
        win._on_slice_index_changed(2, 3)

        assert calls == []
        qtbot.waitUntil(lambda: bool(calls), timeout=_WAIT_TIMEOUT_MS)

        assert len(calls) < 3
        assert calls[-1][1] == 3
    finally:
        win.close()


def test_slice_control_updates_before_render_completion(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=np.float32).reshape(4, 5, 8))
    qtbot.addWidget(win)
    render_calls = []

    def blocked_render(**kwargs):
        render_calls.append((kwargs, win.view_state.slice_indices[2]))

    monkeypatch.setattr(win, "render", blocked_render)
    try:
        win._on_slice_index_changed(2, 4)

        assert render_calls == []
        assert win.view_state.slice_indices[2] == 4
        assert win.widgets["spins"]["slice_indices"][2].value() == 4
        assert win.dimension_strip.chip(2).slice_edit.text() == "4"

        qtbot.waitUntil(lambda: bool(render_calls), timeout=_WAIT_TIMEOUT_MS)
        assert render_calls[-1][1] == 4
    finally:
        win.close()


def test_rapid_scroll_latest_control_state_not_blocked_by_slow_commit(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=np.float32).reshape(4, 5, 8))
    qtbot.addWidget(win)
    render_calls = []

    def recording_render(**kwargs):
        render_calls.append((kwargs.get("reason"), win.view_state.slice_indices[2]))

    monkeypatch.setattr(win, "render", recording_render)
    try:
        for index in (1, 2, 3, 6):
            win._on_slice_index_changed(2, index)
            assert win.view_state.slice_indices[2] == index
            assert win.widgets["spins"]["slice_indices"][2].value() == index
            assert win.dimension_strip.chip(2).slice_edit.text() == str(index)

        assert render_calls == []
        qtbot.waitUntil(lambda: bool(render_calls), timeout=_WAIT_TIMEOUT_MS)
        assert render_calls[-1] == ("slice", 6)
        assert len(render_calls) < 4
    finally:
        win.close()


def test_hot_cached_montage_schedules_no_tile_evaluation(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        state = win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text=":")
        plan = make_montage_plan(state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2)
        for tile in plan.tiles:
            win.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=2,
                colormap_lut=None,
                result=_tile_result(tile, int(tile.source_index) + 1),
            )
        calls = []
        monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))

        win._set_view_state(state)
        win.update_image_view()

        qtbot.waitUntil(
            lambda: getattr(getattr(win, "_committed_display_frame", None), "scene", None) is not None
            and len(win._committed_display_frame.scene.resident_region_ids) == 2,
            timeout=_WAIT_TIMEOUT_MS,
        )
        assert calls == []
        assert win.renderer._montage_cached_tiles_last_session == 2
        assert win.renderer._montage_missing_tiles_last_session == 0
    finally:
        win.close()


def test_hot_cached_tile_layer_clean_flush_updates_zero_items(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        state = win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text=":")
        plan = make_montage_plan(state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2)
        for tile in plan.tiles:
            win.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=2,
                colormap_lut=None,
                result=_tile_result(tile, int(tile.source_index) + 1),
            )
        calls = []
        monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))

        win._set_view_state(state)
        win.update_image_view()
        qtbot.waitUntil(lambda: win.img_view.montageDisplayMode() == "tile_layer", timeout=_WAIT_TIMEOUT_MS)
        first_sources = {tile: state.source_array_id for tile, state in win.img_view._montage_tile_layer.states.items()}

        win.update_image_view()
        timing = win.img_view.lastImageUploadTiming()
        second_sources = {tile: state.source_array_id for tile, state in win.img_view._montage_tile_layer.states.items()}

        assert calls == []
        assert second_sources == first_sources
        assert timing.tile_layer_visible_items == 2
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_items_skipped == 0
        assert timing.tile_layer_upload_ms == 0.0
        assert timing.visible_bytes == 0

        win.renderer._commit_montage_session_presentation(win._montage_session, force=True)
        timing = win.img_view.lastImageUploadTiming()

        assert calls == []
        assert timing.tile_layer_visible_items == 2
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_items_skipped == 0
        assert timing.tile_layer_upload_ms == 0.0
        assert timing.visible_bytes == 0
    finally:
        win.close()


def test_tile_layer_level_change_uses_governed_presentation_batches(qtbot, monkeypatch):
    """Level changes rewindow visible pixels immediately through the preview
    path, while the semantic level acknowledgement stays governed: each
    presentation commit settles one tile per governed batch."""

    clear_arrayscope_settings()
    from types import SimpleNamespace

    from pyqtgraph.Qt import QtCore
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 2 * 2, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        state = win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":")
        plan = make_montage_plan(state, axis=2, indices=(0, 1, 2), tile_shape=(2, 2), columns=3)
        for tile in plan.tiles:
            rgb = np.full((2, 2, 3), 80 + int(tile.source_index) * 20, dtype=np.uint8)
            hist = np.full((2, 2), float(tile.source_index + 1), dtype=np.float32)
            result = EvaluationResult(DisplayImage(rgb, histogram_data=hist), 0.0, rgb.shape, int(rgb.nbytes))
            win.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=2,
                colormap_lut=None,
                result=result,
            )
        monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: pytest.fail("no tile evaluation expected"))

        win._set_view_state(state)
        win.update_image_view()
        qtbot.waitUntil(lambda: win.img_view.montageDisplayMode() == "tile_layer", timeout=_WAIT_TIMEOUT_MS)

        decision = SimpleNamespace(batch_limit=1, budget_ms=100.0, interval_ms=1000, byte_cap=0)
        monkeypatch.setattr(win, "_ui_work_decision", lambda _channel, *, interactive=False: decision)
        win._montage_session.last_commit_monotonic = 0.0

        win.img_view.setLevels(0.5, 4.0)
        timing = win.img_view.lastImageUploadTiming()

        # Preview rewindow reaches every visible RGB tile immediately.
        assert timing.tile_layer_items_updated == 3
        assert timing.tile_layer_rgb_window_tiles == 3
        # Semantic acknowledgement is governed: only one tile settled so far.
        assert len(win._montage_session.pending_payload_upserts) == 0
        assert win._montage_session.has_stale_level_presentations() is True
        assert win._montage_session.has_pending_level_update() is True
        snapshot = win._montage_session.level_presentation_snapshot()
        assert snapshot.pending_count == 2
        assert snapshot.settled is False

        previous_revision = int(win._montage_session.level_revision)
        with QtCore.QSignalBlocker(win.img_view.histogram.item):
            win.img_view.histogram.setLevels(1.0, 3.5)
        win.img_view._on_histogram_levels_changed()

        assert tuple(float(value) for value in win.img_view.getLevels()) == (1.0, 3.5)
        assert win._montage_session.level_generation.target_levels == (1.0, 3.5)
        assert int(win._montage_session.level_revision) == previous_revision + 1
        assert win._montage_session.has_pending_level_update() is True
        assert win._montage_session.level_presentation_snapshot().pending_count == 3

        # Each governed commit acknowledges exactly one tile until settled.
        pending_counts = []
        for _flush in range(4):
            win._montage_session.last_commit_monotonic = 0.0
            win.renderer._schedule_montage_presentation_commit(win._montage_session, force=False)
            win.renderer._flush_montage_presentation_commit()
            snapshot = win._montage_session.level_presentation_snapshot()
            pending_counts.append(int(snapshot.pending_count))
            if snapshot.settled:
                break

        assert pending_counts == [2, 1, 0]
        assert snapshot.stale_count == 0
        assert snapshot.settled is True
    finally:
        win.close()


def test_scalar_tile_layer_level_change_uses_governed_batches_without_image_replacement(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from types import SimpleNamespace
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 2 * 2, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        state = win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":")
        plan = make_montage_plan(state, axis=2, indices=(0, 1, 2), tile_shape=(2, 2), columns=3)
        for tile in plan.tiles:
            image = np.full((2, 2), float(tile.source_index + 1), dtype=np.float32)
            result = EvaluationResult(DisplayImage(image, histogram_data=image), 0.0, image.shape, int(image.nbytes))
            win.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=2,
                colormap_lut=None,
                result=result,
            )
        monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: pytest.fail("no tile evaluation expected"))

        win._set_view_state(state)
        win.update_image_view()
        qtbot.waitUntil(lambda: win.img_view.montageDisplayMode() == "tile_layer", timeout=_WAIT_TIMEOUT_MS)

        decision = SimpleNamespace(batch_limit=1, budget_ms=100.0, interval_ms=1000, byte_cap=1)
        monkeypatch.setattr(win, "_ui_work_decision", lambda _channel, *, interactive=False: decision)
        win._montage_session.last_commit_monotonic = 0.0

        win.img_view.setLevels(0.5, 4.0)
        timing = win.img_view.lastImageUploadTiming()

        # Scalar tiles take the shader/LUT level path: no image replacement or
        # pixel re-upload for a pure level change.
        assert timing.tile_layer_level_updates == 3
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_image_replacements == 0
        assert timing.visible_bytes == 0
        assert win._montage_session.has_stale_level_presentations() is True
        assert win._montage_session.has_pending_level_update() is True
        snapshot = win._montage_session.level_presentation_snapshot()
        assert snapshot.pending_count == 2
        assert snapshot.settled is False

        # Semantic acknowledgement drains one tile per governed commit.
        before_stale = snapshot.stale_count
        win._montage_session.last_commit_monotonic = 0.0
        win.renderer._schedule_montage_presentation_commit(win._montage_session, force=False)
        win.renderer._flush_montage_presentation_commit()

        snapshot = win._montage_session.level_presentation_snapshot()
        assert snapshot.stale_count < before_stale
    finally:
        win.close()


def test_vispy_montage_pyqtgraph_range_change_schedules_viewport_tile_update(qtbot, monkeypatch):
    pytest.importorskip("vispy")

    clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.window import ArrayScopeWindow

    scheduled = []
    win = None
    try:
        settings = QtCore.QSettings()
        settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.VISPY.value)
        settings.sync()

        win = ArrayScopeWindow(np.arange(2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8))
        qtbot.addWidget(win)
        process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(8)), text=":"))
        win.update_image_view()
        qtbot.waitUntil(lambda: win.img_view.montageDisplayMode() == "vispy_tile_layer", timeout=3000)
        monkeypatch.setattr(
            win.renderer,
            "_schedule_montage_viewport_update",
            lambda **_kwargs: scheduled.append(win.img_view.getView().viewRange()),
        )

        assert win.img_view._vispy_canvas_native.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        win.img_view.getView().setRange(xRange=(0.0, 4.0), yRange=(0.0, 2.0), padding=0)
        process_events(qtbot)

        assert win.img_view.surface.capabilities.name == "vispy"
        assert scheduled
    finally:
        if win is not None:
            win.close()
        clear_arrayscope_settings()


def test_vispy_montage_view_range_change_expands_visible_tile_set(qtbot, monkeypatch):
    pytest.importorskip("vispy")

    clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.window import ArrayScopeWindow

    win = None
    try:
        settings = QtCore.QSettings()
        settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.VISPY.value)
        settings.sync()

        # Keep tiles narrow (width 10): viewport constraints cap zoom-out at a
        # fraction of the content rect, so a very wide tile row could never be
        # fully visible in this window and the expansion would be clamped.
        win = ArrayScopeWindow(np.arange(4 * 10 * 8, dtype=np.float32).reshape(4, 10, 8))
        qtbot.addWidget(win)
        win.resize(360, 240)
        win.show()
        process_events(qtbot)
        monkeypatch.setattr(
            win.montage_tile_evaluation_controller,
            "start_latest",
            lambda _fn, **kwargs: len(getattr(win._montage_session, "active_tile_requests", ())) + 1,
        )
        win._set_view_state(win.view_state.with_montage_axis(2, columns=8, indices=tuple(range(8)), text=":"))
        win.update_image_view()
        plan = win._montage_session.plan
        display_height, display_width = tuple(float(value) for value in plan.display_shape[:2])
        tile_count = len(plan.tiles)
        # Expanded montage ranges auto-fit by design. Narrow explicitly so this
        # test measures viewport retargeting rather than initial fit policy.
        win.img_view.getView().setRange(xRange=(0.0, 10.0), yRange=(0.0, 4.0), padding=0)
        win.renderer._run_montage_viewport_update()
        initial_visible = len(win._montage_session.visible_tiles)

        # The view box is aspect-locked to square pixels, so request an x/y
        # pair that already matches the viewport aspect; otherwise pyqtgraph
        # rewrites the ranges and keeps part of the tile row off screen.
        viewport = win.img_view.graphicsView.viewport().size()
        y_span = display_width * max(1, viewport.height()) / max(1, viewport.width())
        y_center = display_height / 2.0
        win.img_view.getView().setRange(
            xRange=(0.0, display_width),
            yRange=(y_center - y_span / 2.0, y_center + y_span / 2.0),
            padding=0,
        )
        win.renderer._run_montage_viewport_update()
        expanded_visible = len(win._montage_session.visible_tiles)

        assert initial_visible < tile_count
        assert expanded_visible == tile_count
    finally:
        if win is not None:
            win.close()
        clear_arrayscope_settings()


def test_cold_montage_tile_patches_without_side_panel_refresh(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    calls = []
    operation_refreshes = []
    inspection_refreshes = []
    monkeypatch.setattr(win, "_update_operation_dock", lambda: operation_refreshes.append("operation"))
    monkeypatch.setattr(win, "_refresh_inspection_dock", lambda: inspection_refreshes.append("inspection"))
    try:
        process_events(qtbot)
        qtbot.waitUntil(lambda: not win.montage_tile_evaluation_controller.is_busy(), timeout=3000)
        win.operation_evaluator.clear_cache()
        win.renderer._retained_tiled_payload_store().clear_for_document_or_context_change("test-cold-start")
        win.renderer._montage_session = None
        frame = getattr(win, "_committed_display_frame", None)
        payloads = getattr(getattr(frame, "value_source", None), "payloads", None)
        if isinstance(payloads, dict):
            payloads.clear()
        monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_image_view(defer_side_panels=True)
        operation_refreshes.clear()
        inspection_refreshes.clear()

        requested_index = int(calls[0]["key"][-1])
        tile = next(
            tile
            for tile in win._montage_session.plan.tiles
            if int(tile.montage_index) == requested_index
        )
        calls[0]["on_done"](_tile_result(tile, 9))

        def requested_tile_is_patched():
            layer = getattr(win.img_view, "_montage_tile_layer", None)
            state = None if layer is None else layer.states.get(int(tile.montage_index))
            image = None if state is None else getattr(state.item, "image", None)
            return image is not None and np.array_equal(image, np.full((tile.height, tile.width), 9, dtype=np.float32))

        qtbot.waitUntil(requested_tile_is_patched, timeout=_WAIT_TIMEOUT_MS)

        assert operation_refreshes == []
        assert inspection_refreshes == []
    finally:
        win.close()
