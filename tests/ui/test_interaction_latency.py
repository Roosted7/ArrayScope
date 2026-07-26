import numpy as np
import pytest
from pytestqt.exceptions import TimeoutError as QtBotTimeoutError

from arrayscope.display.slice_engine import DisplayImage
from arrayscope.operations.evaluator import EvaluationResult
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import clear_arrayscope_settings, process_events, require_wgpu_adapter


def _tile_result(tile, value):
    image = np.full((tile.height, tile.width), value, dtype=np.float32)
    return EvaluationResult(
        DisplayImage(image, histogram_data=image.copy()), 0.0, image.shape, int(image.nbytes)
    )


def _backend_residency_snapshot(win, backend):
    if backend == "pyqtgraph":
        return {
            int(tile_number): (
                state.source_array_id,
                state.acknowledged_identity,
            )
            for tile_number, state in win.img_view._montage_tile_layer.states.items()
            if state.visible and state.item.isVisible()
        }
    return {
        int(tile_number): (
            tuple(binding.get("actual_key") for binding in row.get("physical_page_bindings", ())),
            row.get("physical_acknowledged_identity"),
        )
        for tile_number, row in win.img_view.tileTruthPhysicalRows().items()
    }


def _settle_initial_tile_state(win, qtbot):
    qtbot.waitUntil(
        lambda: (
            win.renderer._frame_session is not None
            and win.renderer._frame_session.visible_plan_complete()
            and not win.montage_tile_evaluation_controller.is_busy()
        ),
        timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
    )


def _wait_for_backend_residency(win, qtbot, backend):
    try:
        qtbot.waitUntil(
            lambda: (
                win.renderer._frame_session.required_target_settled()
                and len(_backend_residency_snapshot(win, backend)) == 2
            ),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
    except QtBotTimeoutError:
        session = win.renderer._frame_session
        pyqtgraph_states = (
            {
                int(tile): (
                    state.visible,
                    state.item.isVisible(),
                    state.source_index,
                    state.acknowledged_identity,
                )
                for tile, state in win.img_view._montage_tile_layer.states.items()
            }
            if backend == "pyqtgraph"
            else None
        )
        pytest.fail(
            "cached backend residency did not settle: "
            f"required={session.required_tile_numbers()!r}, "
            f"unsettled={session.required_target_unsettled_tiles()!r}, "
            f"backend={_backend_residency_snapshot(win, backend)!r}, "
            f"pyqtgraph_states={pyqtgraph_states!r}, "
            f"lifecycle={session.lifecycle.backend_presented_identities!r}, "
            f"commit={win.renderer._last_montage_commit_outcome!r}, "
            f"timing={win.img_view.lastImageUploadTiming()!r}, "
            f"levels={session.level_presentation_snapshot()!r}, "
            f"kernel={win.kernel.diagnostics()!r}"
        )


def test_rapid_slice_burst_is_coalesced_and_latest(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=np.float32).reshape(4, 5, 8))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(
        win, "render", lambda **kwargs: calls.append((kwargs, win.view_state.slice_indices[2]))
    )
    try:
        win._on_slice_index_changed(2, 1)
        win._on_slice_index_changed(2, 2)
        win._on_slice_index_changed(2, 3)

        assert calls == []
        qtbot.waitUntil(lambda: bool(calls), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

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

        qtbot.waitUntil(lambda: bool(render_calls), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
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
        qtbot.waitUntil(lambda: bool(render_calls), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        assert render_calls[-1] == ("slice", 6)
        assert len(render_calls) < 4
    finally:
        win.close()


@pytest.mark.parametrize("backend", ["pyqtgraph", "wgpu"])
def test_hot_cached_montage_schedules_no_tile_evaluation(qtbot, backend):
    clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", backend)
    settings.sync()
    win = ArrayScopeWindow(np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        _settle_initial_tile_state(win, qtbot)
        state = win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text=":")
        plan = make_montage_plan(state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2)
        for tile in plan.tiles:
            win.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=2,
                colormap_lut=None,
                result=_tile_result(tile, int(tile.source_index) + 1),
                shader_display=backend == "wgpu",
            )
        evaluations_before = win.operation_evaluator.image_evaluations
        win._set_view_state(state)
        win.update_image_view()

        _wait_for_backend_residency(win, qtbot, backend)
        assert win.operation_evaluator.image_evaluations == evaluations_before
        assert len(win._committed_display_frame.scene.resident_region_ids) == 2
        assert set(_backend_residency_snapshot(win, backend)) == {0, 1}
    finally:
        win.close()


@pytest.mark.parametrize("backend", ["pyqtgraph", "wgpu"])
def test_hot_cached_tile_layer_clean_flush_updates_zero_items(qtbot, backend):
    clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", backend)
    settings.sync()
    win = ArrayScopeWindow(np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        _settle_initial_tile_state(win, qtbot)
        state = win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text=":")
        plan = make_montage_plan(state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2)
        for tile in plan.tiles:
            win.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=2,
                colormap_lut=None,
                result=_tile_result(tile, int(tile.source_index) + 1),
                shader_display=backend == "wgpu",
            )
        evaluations_before = win.operation_evaluator.image_evaluations
        win._set_view_state(state)
        win.update_image_view()
        _wait_for_backend_residency(win, qtbot, backend)
        first_residency = _backend_residency_snapshot(win, backend)
        first_timing = win.img_view.lastImageUploadTiming()

        win.update_image_view()
        timing = win.img_view.lastImageUploadTiming()
        second_residency = _backend_residency_snapshot(win, backend)

        assert second_residency == first_residency
        assert win.operation_evaluator.image_evaluations == evaluations_before
        # ADR 0051 P2 (session-rebirth cost): a same-key re-render reuses the
        # live session outright — no rebirth, no flush, the backend is never
        # touched.  The strongest form of "clean": the last upload timing is
        # the untouched record of the first commit.
        assert timing is first_timing
        assert int(getattr(win.renderer, "_frame_session_reuses", 0)) >= 1

        # An explicit forced flush may drain pending level refinement once;
        # the steady state after it must be a true no-op — the backend's
        # upload record does not change again.
        win.renderer.commit_frame_session_presentation(win.renderer._frame_session)
        drained_timing = win.img_view.lastImageUploadTiming()
        win.renderer.commit_frame_session_presentation(win.renderer._frame_session)
        settled_timing = win.img_view.lastImageUploadTiming()

        assert settled_timing is drained_timing
        assert _backend_residency_snapshot(win, backend) == first_residency
        assert settled_timing.tile_layer_visible_items == len(first_residency)
    finally:
        win.close()


def test_wgpu_montage_pyqtgraph_range_change_schedules_viewport_tile_update(qtbot, monkeypatch):
    require_wgpu_adapter()
    clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.window import ArrayScopeWindow

    scheduled = []
    win = None
    try:
        settings = QtCore.QSettings()
        settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.WGPU.value)
        settings.sync()

        win = ArrayScopeWindow(np.arange(2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8))
        qtbot.addWidget(win)
        process_events(qtbot)
        win._set_view_state(
            win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(8)), text=":")
        )
        win.update_image_view()
        qtbot.waitUntil(
            lambda: win.img_view.montageDisplayMode() == "wgpu_tile_layer",
            timeout=min(3000, INTERACTION_SETTLE_HARD_LIMIT_MS),
        )
        monkeypatch.setattr(
            win.renderer,
            "retarget_montage_viewport",
            lambda: scheduled.append(win.img_view.getView().viewRange()),
        )

        assert win.img_view._wgpu_canvas.testAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        win.img_view.getView().setRange(xRange=(0.0, 4.0), yRange=(0.0, 2.0), padding=0)
        process_events(qtbot)

        assert win.img_view.surface.capabilities.name == "wgpu"
        assert scheduled
    finally:
        if win is not None:
            win.close()
        clear_arrayscope_settings()


def test_montage_viewport_continuation_never_retargets_inline(qtbot, monkeypatch):
    from arrayscope.window import ArrayScopeWindow

    clear_arrayscope_settings()
    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        win._set_view_state(
            win.view_state.with_montage_axis(
                2,
                columns=3,
                indices=(0, 1, 2),
                text="0:3",
            )
        )
        calls = []
        monkeypatch.setattr(
            win.renderer,
            "retarget_montage_viewport",
            lambda: calls.append("retarget"),
        )

        win.renderer._schedule_frame_viewport_update(delay_ms=1)
        win.renderer._schedule_frame_viewport_update(delay_ms=1)

        assert calls == []
        qtbot.waitUntil(lambda: calls == ["retarget"], timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
    finally:
        win.close()
        clear_arrayscope_settings()


def test_wgpu_montage_view_range_change_expands_visible_tile_set(qtbot, monkeypatch):
    require_wgpu_adapter()
    clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.window import ArrayScopeWindow

    win = None
    try:
        settings = QtCore.QSettings()
        settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.WGPU.value)
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
            lambda _fn, **kwargs: (
                len(getattr(win.renderer._frame_session, "active_tile_requests", ())) + 1
            ),
        )
        win._set_view_state(
            win.view_state.with_montage_axis(2, columns=8, indices=tuple(range(8)), text=":")
        )
        win.update_image_view()
        plan = win.renderer._frame_session.plan
        tile_count = len(plan.tiles)
        # Expanded montage ranges auto-fit by design. Narrow explicitly so this
        # test measures viewport retargeting rather than initial fit policy.
        win.img_view.getView().setRange(xRange=(0.0, 10.0), yRange=(0.0, 4.0), padding=0)
        win.renderer.apply_montage_viewport_retarget()
        initial_visible = len(win.renderer._frame_session.visible_tiles)

        win.fit_image_to_view(True)
        process_events(qtbot, count=20)
        win.fit_image_to_view(False)
        win.renderer.apply_montage_viewport_retarget()
        expanded_visible = len(win.renderer._frame_session.visible_tiles)

        assert initial_visible < tile_count
        assert expanded_visible == tile_count
    finally:
        if win is not None:
            win.close()
        clear_arrayscope_settings()
