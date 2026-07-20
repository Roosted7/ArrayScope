import numpy as np

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings
from tests.ui.helpers import process_events as _process_events


def test_slice_text_updates_immediately_while_render_is_coalesced(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=float).reshape(4, 5, 8))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(
        win, "render", lambda **kwargs: calls.append((kwargs, win.view_state.slice_indices[2]))
    )
    try:
        _process_events(qtbot, count=2)
        win._on_slice_index_changed(2, 3)

        assert win.view_state.slice_indices[2] == 3
        assert win.dimension_strip.chip(2).slice_edit.text() == "3"
        assert win.widgets["spins"]["slice_indices"][2].value() == 3
        assert calls == []

        qtbot.waitUntil(lambda: len(calls) == 1, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        assert len(calls) == 1
        assert calls[0][0]["reason"] == "slice"
        assert calls[0][0]["defer_side_panels"] is True
        assert calls[0][1] == 3
    finally:
        win.close()


def test_rapid_slice_burst_renders_only_latest_state(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=float).reshape(4, 5, 8))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(
        win,
        "render",
        lambda **kwargs: calls.append((kwargs["reason"], win.view_state.slice_indices[2])),
    )
    try:
        _process_events(qtbot, count=2)
        win._on_slice_index_changed(2, 1)
        win._on_slice_index_changed(2, 2)
        win._on_slice_index_changed(2, 3)

        _process_events(qtbot, count=3)

        assert calls == [("slice", 3)]
        assert win.view_state.slice_indices[2] == 3
        assert win.render_coordinator.requested == 3
        assert win.render_coordinator.coalesced >= 2
    finally:
        win.close()


def test_duplicate_pending_render_request_is_ignored(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=float).reshape(4, 5, 8))
    qtbot.addWidget(win)
    advances = []
    renders = []
    monkeypatch.setattr(
        win.renderer,
        "_advance_render_generation",
        lambda reason: advances.append(reason) or len(advances),
    )
    monkeypatch.setattr(win, "render", lambda **kwargs: renders.append(kwargs))
    try:
        _process_events(qtbot, count=2)
        win.request_render(reason="same-state", interactive=False)
        win.request_render(reason="same-state", interactive=False)

        assert advances == ["request:same-state"]
        assert win.render_coordinator.requested == 1
        assert win.render_coordinator.coalesced == 0

        _process_events(qtbot, count=3)
        assert [call["reason"] for call in renders] == ["same-state"]
    finally:
        win.close()


def test_coalescer_throttles_without_starving_continuous_bursts(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 12, dtype=float).reshape(4, 5, 12))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(
        win, "render", lambda **kwargs: calls.append(win.view_state.slice_indices[2])
    )
    try:
        _process_events(qtbot, count=2)
        for index in range(1, 8):
            win._on_slice_index_changed(2, index)
            qtbot.wait(5)
        _process_events(qtbot, count=4)

        assert 1 <= len(calls) < 7
        assert calls[-1] == 7
    finally:
        win.close()


def test_interactive_slice_preserves_visible_work_and_cancels_side_work(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=float).reshape(4, 5, 8))
    qtbot.addWidget(win)
    cleared = []

    def record_clear(controller_name):
        return lambda group: cleared.append((controller_name, group))

    monkeypatch.setattr(win.visible_evaluation_controller, "clear_group", record_clear("visible"))
    monkeypatch.setattr(
        win.montage_tile_evaluation_controller, "clear_group", record_clear("montage")
    )
    monkeypatch.setattr(win.profile_evaluation_controller, "clear_group", record_clear("profile"))
    monkeypatch.setattr(win.roi_evaluation_controller, "clear_group", record_clear("roi"))
    monkeypatch.setattr(win.pixel_evaluation_controller, "clear_group", record_clear("pixel"))
    monkeypatch.setattr(win, "render", lambda **_kwargs: None)
    try:
        _process_events(qtbot, count=2)
        win._on_slice_index_changed(2, 1)

        assert not any(name in {"visible", "montage"} for name, _group in cleared)
        assert win.render_coordinator.interactive_active
    finally:
        win.close()


def test_deferred_side_panels_refresh_once_after_interaction_quiet(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=float).reshape(4, 5, 8))
    qtbot.addWidget(win)
    calls = {"operation": 0, "inspection": 0, "profile": 0}
    monkeypatch.setattr(
        win,
        "_update_operation_dock",
        lambda: calls.__setitem__("operation", calls["operation"] + 1),
    )
    monkeypatch.setattr(
        win,
        "_refresh_inspection_dock",
        lambda: calls.__setitem__("inspection", calls["inspection"] + 1),
    )
    monkeypatch.setattr(
        win, "update_line_plot", lambda: calls.__setitem__("profile", calls["profile"] + 1)
    )
    try:
        _process_events(qtbot, count=2)
        win.render_coordinator._quiet_interval_ms = 5000
        calls.update({"operation": 0, "inspection": 0, "profile": 0})
        win._on_slice_index_changed(2, 1)
        qtbot.waitUntil(
            lambda: (
                win._deferred_side_panel_refresh_pending
                and not win.render_coordinator.has_pending_render
            ),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        assert calls == {"operation": 0, "inspection": 0, "profile": 0}
        assert win._deferred_side_panel_refresh_pending

        monkeypatch.setattr(win.visible_evaluation_controller, "is_busy", lambda: False)
        win._deferred_side_panel_refresh_pending = True
        win.render_coordinator._quiet_timer.stop()
        win.render_coordinator._quiet_timer_elapsed()
        assert calls["operation"] == 1
        assert calls["inspection"] == 1
        assert calls["profile"] == 0
        assert not win._deferred_side_panel_refresh_pending
    finally:
        win.close()


def test_montage_side_panels_defer_while_viewport_interaction_active():
    from types import SimpleNamespace

    from arrayscope.window.frame_controller import _should_defer_montage_side_panels

    def _renderer(active):
        renderer = SimpleNamespace(win=SimpleNamespace(_viewport_interaction_active=active))
        return renderer

    assert _should_defer_montage_side_panels(
        _renderer(True),
        SimpleNamespace(defer_side_panels=False),
    )
    assert _should_defer_montage_side_panels(
        _renderer(False),
        SimpleNamespace(defer_side_panels=True),
    )
    assert not _should_defer_montage_side_panels(
        _renderer(False),
        SimpleNamespace(defer_side_panels=False),
    )


def test_operation_dock_identical_refresh_does_not_rebuild_rows(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.operations.pipeline import Crop
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5, dtype=float).reshape(4, 5))
    qtbot.addWidget(win)
    try:
        win.operation_coordinator.load_operations((Crop(0, 0, 3),))
        win._set_document(win.operation_coordinator.document)
        win._update_operation_dock()
        _process_events(qtbot, count=2)
        calls = []
        original_row_widget = win.operation_dock._row_widget

        def record_row_widget(index, operation):
            calls.append(index)
            return original_row_widget(index, operation)

        monkeypatch.setattr(win.operation_dock, "_row_widget", record_row_widget)
        win._update_operation_dock()

        assert calls == []
    finally:
        win.close()


def test_direct_render_still_refreshes_side_panels(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5, dtype=float).reshape(4, 5))
    qtbot.addWidget(win)
    calls = {"operation": 0, "inspection": 0}
    monkeypatch.setattr(
        win,
        "_update_operation_dock",
        lambda: calls.__setitem__("operation", calls["operation"] + 1),
    )
    monkeypatch.setattr(
        win,
        "_refresh_inspection_dock",
        lambda: calls.__setitem__("inspection", calls["inspection"] + 1),
    )
    try:
        _process_events(qtbot, count=2)
        calls.update({"operation": 0, "inspection": 0})
        win.operation_evaluator.clear_cache()
        win.render(reason="normal")

        assert calls["operation"] >= 1
        assert calls["inspection"] == 0
    finally:
        win.close()


def test_cached_interactive_render_uses_cadence_and_cancels_side_work(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=float).reshape(4, 5, 8))
    qtbot.addWidget(win)
    renders = []
    cancellations = []
    monkeypatch.setattr(win, "_interactive_frame_cache_hit", lambda: True)
    monkeypatch.setattr(
        win,
        "_cancel_render_dependent_work_for_interactive_change",
        lambda: cancellations.append(True),
    )
    monkeypatch.setattr(win, "render", lambda **kwargs: renders.append(kwargs))
    try:
        before_flushes = int(win.render_coordinator.immediate_cache_flushes)
        win.render_coordinator._quiet_interval_ms = 5000
        win.request_render(reason="cached-slice", interactive=True)

        assert renders == []
        assert cancellations == []
        qtbot.waitUntil(lambda: bool(renders), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        assert renders[-1]["reason"] == "cached-slice"
        assert cancellations == [True]
        assert win.render_coordinator.immediate_cache_flushes == before_flushes
    finally:
        win.close()


def test_cached_interactive_render_skips_intermediate_requests_until_draw_completes(qtbot):
    from pyqtgraph.Qt import QtCore

    from arrayscope.window.render_coordinator import RenderCoordinator

    class DummyImageView(QtCore.QObject):
        presentationDrawn = QtCore.Signal()

        def __init__(self):
            super().__init__()
            self.pending = True

        def presentationDrawPending(self):
            return self.pending

    class DummyWindow(QtCore.QObject):
        def __init__(self):
            super().__init__()
            self.img_view = DummyImageView()
            self.rendered = []
            self.cancelled = 0
            self.render_coordinator = RenderCoordinator(self)

        def _interactive_frame_cache_hit(self):
            return True

        def _cancel_render_dependent_work_for_interactive_change(self):
            self.cancelled += 1

        def render(self, **kwargs):
            self.rendered.append(kwargs)

    win = DummyWindow()
    win.render_coordinator.request(reason="slice-1", interactive=True)
    win.render_coordinator.request(reason="slice-2", interactive=True)
    win.render_coordinator.request(reason="slice-3", interactive=True)
    _process_events(qtbot, count=3)

    assert win.rendered == []
    assert win.render_coordinator.presentation_backpressure_skips == 3

    win.img_view.pending = False
    win.img_view.presentationDrawn.emit()
    qtbot.waitUntil(lambda: bool(win.rendered), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

    assert [call["reason"] for call in win.rendered] == ["slice-3"]
    assert win.cancelled == 1


def test_uncached_interactive_render_supersedes_pending_draw(qtbot):
    from pyqtgraph.Qt import QtCore

    from arrayscope.window.render_coordinator import RenderCoordinator

    class DummyImageView(QtCore.QObject):
        presentationDrawn = QtCore.Signal()

        def __init__(self):
            super().__init__()
            self.pending = True

        def presentationDrawPending(self):
            return self.pending

    class DummyWindow(QtCore.QObject):
        def __init__(self):
            super().__init__()
            self.img_view = DummyImageView()
            self.cancelled = 0
            self.rendered = []
            self.render_coordinator = RenderCoordinator(self)

        def _interactive_frame_cache_hit(self):
            return False

        def _interactive_render_supersedes_presentation(self, *, reason):
            return True

        def _cancel_render_dependent_work_for_interactive_change(self):
            self.cancelled += 1

        def render(self, **kwargs):
            self.rendered.append(kwargs)

    win = DummyWindow()
    win.render_coordinator.request(reason="slice-1", interactive=True)
    win.render_coordinator.request(reason="slice-2", interactive=True)
    qtbot.waitUntil(lambda: len(win.rendered) == 1, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

    assert win.cancelled == 1
    assert win.render_coordinator.presentation_backpressure_skips == 2
    assert [call["reason"] for call in win.rendered] == ["slice-2"]


def test_quiet_timer_flushes_pending_render_if_draw_signal_was_missed(qtbot):
    from pyqtgraph.Qt import QtCore

    from arrayscope.window.render_coordinator import RenderCoordinator

    class DummyImageView(QtCore.QObject):
        presentationDrawn = QtCore.Signal()

        def __init__(self):
            super().__init__()
            self.pending = True

        def presentationDrawPending(self):
            return self.pending

    class DummyWindow(QtCore.QObject):
        def __init__(self):
            super().__init__()
            self.img_view = DummyImageView()
            self.rendered = []
            self.cancelled = 0
            self.render_coordinator = RenderCoordinator(self, quiet_interval_ms=1, busy_retry_ms=1)

        def _interactive_frame_cache_hit(self):
            return True

        def _cancel_render_dependent_work_for_interactive_change(self):
            self.cancelled += 1

        def render(self, **kwargs):
            self.rendered.append(kwargs)

    win = DummyWindow()
    win.render_coordinator.request(reason="slice-latest", interactive=True)
    _process_events(qtbot, count=2)

    assert win.rendered == []
    win.img_view.pending = False
    qtbot.waitUntil(lambda: bool(win.rendered), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

    assert [call["reason"] for call in win.rendered] == ["slice-latest"]
    assert win.cancelled == 1


def test_cached_frame_render_skips_memory_policy_resample(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5, dtype=float).reshape(4, 5))
    qtbot.addWidget(win)
    try:
        qtbot.waitUntil(
            lambda: (
                getattr(win, "_committed_display_frame", None) is not None
                and not win.renderer._montage_render_active()
            ),
            timeout=min(3000, INTERACTION_SETTLE_HARD_LIMIT_MS),
        )
        refreshes = []
        monkeypatch.setattr(
            win.renderer, "_refresh_memory_policy", lambda **kwargs: refreshes.append(kwargs)
        )

        win.render(reason="cached-frame")

        assert refreshes == [{"active_render": False}]
    finally:
        win.close()
