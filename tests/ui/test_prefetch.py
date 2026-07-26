import os
import threading
import time

import numpy as np

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _process_events(qtbot, count=8):
    for _ in range(count):
        qtbot.wait(10)


def _clear_arrayscope_settings():
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings()
    settings.clear()
    settings.sync()


def test_nearby_slice_prefetch_uses_prefetch_state_keys(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True
        )
        win._active_slice_axis = 2
        win.renderer._schedule_prefetch_nearby_slices(win.view_state.with_slice(2, 2), None)
        _process_events(qtbot, count=40)

        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled > 0
        assert win.operation_evaluator.display_cache_diagnostics().prefetch_stored > 0
        assert (
            win.operation_evaluator.cached_display_tile(win.view_state.with_slice(2, 1)) is not None
        )
    finally:
        win.close()


def test_nearby_slice_prefetch_skips_when_setting_disabled(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._active_slice_axis = 2
        before = win.operation_evaluator.display_cache_diagnostics()
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 1), None)
        _process_events(qtbot, count=20)
        after = win.operation_evaluator.display_cache_diagnostics()

        assert after.prefetch_scheduled == before.prefetch_scheduled
        assert after.prefetch_skipped > before.prefetch_skipped
    finally:
        win.close()


def test_prefetch_dispatch_is_queued_but_not_timer_admitted(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True
        )
        win._active_slice_axis = 2
        win.renderer._schedule_prefetch_nearby_slices(win.view_state.with_slice(2, 2), None)
        assert not hasattr(win, "_prefetch_idle_timer")
        _process_events(qtbot, count=10)
        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled > 0
    finally:
        win.close()


def test_prefetch_limits_to_two_neighbors(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.prefetch_policy import SliceScrubMomentum
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 8, dtype=float).reshape(3, 4, 8))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True
        )
        win._active_slice_axis = 2
        momentum = SliceScrubMomentum()
        momentum.observe(2, now=0.0)
        win.renderer._prefetch_momentum = momentum
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 4), None)
        _process_events(qtbot, count=40)

        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled <= 2
    finally:
        win.close()


def test_prefetch_deepens_with_sustained_directional_scrub(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.prefetch_policy import SliceScrubMomentum
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 16, dtype=float).reshape(3, 4, 16))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True
        )
        win._active_slice_axis = 2
        momentum = SliceScrubMomentum()
        base = time.monotonic() - 0.3
        for step, index in enumerate((2, 3, 4, 5)):
            momentum.observe(index, now=base + 0.1 * step)
        win.renderer._prefetch_momentum = momentum
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 6), None)
        _process_events(qtbot, count=40)

        scheduled = win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled
        assert scheduled > 2, (
            "sustained same-direction scrubbing should warm deeper ahead of the motion"
        )
    finally:
        win.close()


def test_montage_prefetch_denial_does_not_cap_slice_prefetch(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.prefetch_policy import SliceScrubMomentum
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 16, dtype=float).reshape(3, 4, 16))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True
        )
        win._active_slice_axis = 2
        win.prefetch_evaluation_controller.set_max_prefetch(32)

        # With no montage stage ready, the montage-prefetch decision denies
        # montage speculation. That local decision must not shrink the shared
        # prefetch controller and accidentally throttle ordinary slice prefetch.
        win._apply_resource_governor_decisions()
        assert win.prefetch_evaluation_controller._max_prefetch == 32

        momentum = SliceScrubMomentum()
        base = time.monotonic() - 0.3
        for step, index in enumerate((2, 3, 4, 5)):
            momentum.observe(index, now=base + 0.1 * step)
        win.renderer._prefetch_momentum = momentum
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 6), None)
        _process_events(qtbot, count=40)

        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled > 2
    finally:
        win.close()


def test_montage_prefetch_completion_uses_real_orchestrator_staleness_guard(
    qtbot,
    monkeypatch,
):
    _clear_arrayscope_settings()
    from arrayscope.core.frame_targets import WorkStart
    from arrayscope.window import ArrayScopeWindow, montage_prefetch

    win = ArrayScopeWindow(np.arange(3 * 4 * 8, dtype=float).reshape(3, 4, 8))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.request_operation("reverse", 0)
        state = win.view_state.with_montage_axis(
            2,
            columns=4,
            indices=tuple(range(8)),
            text=":",
        )
        win._set_view_state(state)
        win.render(reason="test-prefetch-completion-guard")
        qtbot.waitUntil(
            lambda: win.renderer._frame_session is not None,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        session = win.renderer._frame_session
        assert session.document.enabled_operations

        captured = {}

        def capture_prefetch(_evaluate, *, on_done, **_kwargs):
            captured["done"] = on_done
            return WorkStart(True)

        monkeypatch.setattr(montage_prefetch, "_interaction_active", lambda _owner: False)
        monkeypatch.setattr(montage_prefetch, "_busy", lambda _owner, _session: False)
        monkeypatch.setattr(
            montage_prefetch,
            "_candidate_tiles",
            lambda _session: (session.plan.tiles[-1],),
        )
        monkeypatch.setattr(
            montage_prefetch,
            "_stage_for_tile",
            lambda _owner, _session, _tile: (object(), object(), object()),
        )
        monkeypatch.setattr(
            win.operation_evaluator,
            "cached_montage_tile",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            win.prefetch_evaluation_controller,
            "start_prefetch",
            capture_prefetch,
        )

        decisions = montage_prefetch.schedule_near_viewport_montage_prefetch(
            win.renderer,
            session,
            max_tiles=1,
        )
        assert decisions[0].decision == "scheduled"
        assert callable(captured.get("done"))

        win._set_view_state(win.view_state.with_slice(2, 1))
        win.render(reason="test-prefetch-completion-superseded")
        qtbot.waitUntil(
            lambda: win.renderer._frame_session is not session,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        stale_before = win.operation_evaluator.display_cache_diagnostics().prefetch_stale
        replans = []
        monkeypatch.setattr(
            win.renderer,
            "request_montage_replan",
            lambda current: replans.append(current),
        )

        captured["done"](object())

        assert (
            win.operation_evaluator.display_cache_diagnostics().prefetch_stale == stale_before + 1
        )
        assert replans == [win.renderer._frame_session], (
            "stale prefetch claim release must wake the current session that "
            "may have attached to the shared pages"
        )
    finally:
        win.close()


def test_montage_prefetch_completion_warms_gpu_atlas_residency(qtbot, monkeypatch):
    """A stored montage prefetch must cross the persistent GPU-residency seam.

    Pre-fix, the completion path accepted only ``cpu_item`` residency, so the
    same completed payload that warmed PyQtGraph was silently left CPU-only on
    a ``gpu_atlas`` backend.
    """

    _clear_arrayscope_settings()
    from arrayscope.core.frame_targets import WorkStart
    from arrayscope.display.backend_contract import WGPU_CAPABILITIES
    from arrayscope.display.montage import RenderedTile
    from arrayscope.window import ArrayScopeWindow, montage_prefetch

    win = ArrayScopeWindow(np.arange(3 * 4 * 8, dtype=float).reshape(3, 4, 8))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.request_operation("reverse", 0)
        state = win.view_state.with_montage_axis(
            2,
            columns=4,
            indices=tuple(range(8)),
            text=":",
        )
        win._set_view_state(state)
        win.render(reason="test-gpu-montage-prefetch-warm")
        qtbot.waitUntil(
            lambda: win.renderer._frame_session is not None,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        session = win.renderer._frame_session
        tile = session.plan.tiles[-1]
        image = np.full(tuple(session.plan.tile_shape), float(tile.source_index), dtype=np.float32)
        rendered = RenderedTile(
            tile=tile,
            image=image,
            histogram_data=image,
            eval_ms=0.0,
            slab_shape=image.shape,
            slab_nbytes=image.nbytes,
        )
        captured = {}
        warm_calls = []

        def capture_prefetch(_evaluate, *, on_done, **_kwargs):
            captured["done"] = on_done
            return WorkStart(True)

        monkeypatch.setattr(montage_prefetch, "_interaction_active", lambda _owner: False)
        monkeypatch.setattr(montage_prefetch, "_busy", lambda _owner, _session: False)
        monkeypatch.setattr(montage_prefetch, "_candidate_tiles", lambda _session: (tile,))
        monkeypatch.setattr(
            montage_prefetch,
            "_claim_walk_preview",
            lambda _session, _tile: None,
        )
        monkeypatch.setattr(
            montage_prefetch,
            "_stage_for_tile",
            lambda _owner, _session, _tile: (object(), object(), object()),
        )
        monkeypatch.setattr(
            montage_prefetch,
            "image_view_backend_capabilities",
            lambda _view: WGPU_CAPABILITIES,
        )
        monkeypatch.setattr(
            win.operation_evaluator,
            "cached_montage_tile",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            win.operation_evaluator,
            "store_montage_tile_result",
            lambda *_args, **_kwargs: rendered,
        )
        monkeypatch.setattr(
            win.prefetch_evaluation_controller,
            "start_prefetch",
            capture_prefetch,
        )
        monkeypatch.setattr(
            win.img_view,
            "warmTiledResidency",
            lambda **kwargs: warm_calls.append(kwargs),
            raising=False,
        )

        decisions = montage_prefetch.schedule_near_viewport_montage_prefetch(
            win.renderer,
            session,
            max_tiles=1,
        )
        assert decisions[0].decision == "scheduled"
        captured["done"](montage_prefetch._MontagePrefetchWorkerResult(object()))

        assert len(warm_calls) == 1, "gpu_atlas montage prefetch must warm the completed payload"
        call = warm_calls[0]
        assert tuple(call["payloads"]) == (int(tile.montage_index),)
        payload = call["payloads"][int(tile.montage_index)]
        assert payload.source_index == int(tile.source_index)
        assert call["geometry"].montage == session.plan.geometry
    finally:
        win.close()


def test_montage_prefetch_candidates_bias_ahead_of_scroll_direction():
    """Directional speculation stays local to prefetch candidate ordering.

    The canonical viewport-priority order remains the no-momentum fallback.
    Once a montage index-window scrub has direction, candidates ahead of the
    focused visible source are warmed nearest-first, followed by the bounded
    reverse-side guard in the same nearest-first order.
    """

    from types import SimpleNamespace

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.model.tile_priority import TilePriorityContext
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.montage_prefetch import _candidate_tiles

    state = ViewState.from_shape((2, 2, 5)).with_montage_axis(
        2,
        columns=5,
        indices=tuple(range(5)),
        text=":",
    )
    plan = make_montage_plan(
        state,
        axis=2,
        indices=tuple(range(5)),
        tile_shape=(2, 2),
        columns=5,
    )
    context = TilePriorityContext.from_tiles(
        focus=(float(plan.tiles[2].x0 + 1), float(plan.tiles[2].y0 + 1)),
        view_range=((5.0, 9.0), (-1.0, 3.0)),
        visible_tiles=(2,),
        near_tiles=tuple(range(5)),
    )
    session = SimpleNamespace(
        plan=plan,
        visible_tiles=(plan.tiles[2],),
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        tile_priority_context=lambda: context,
    )

    neutral = tuple(tile.source_index for tile in _candidate_tiles(session, direction=0))
    forward = tuple(tile.source_index for tile in _candidate_tiles(session, direction=1))
    reverse = tuple(tile.source_index for tile in _candidate_tiles(session, direction=-1))

    assert neutral == (1, 3, 0, 4)
    assert forward == (3, 4, 1, 0)
    assert reverse == (1, 0, 3, 4)


def test_predicted_future_tiles_lead_the_scroll_beyond_the_window():
    """Speculation targets indices about to arrive, not the resident window.

    The montage plan only ever holds the current index window, so warming the
    next step's incoming tiles requires synthetic tiles for source indices just
    past the window boundary along the scroll direction. They carry no grid
    slot (montage_index == -1) because they warm only the window-independent
    display cache, never the grid-keyed GPU residency.
    """

    from types import SimpleNamespace

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.montage_prefetch import _predicted_future_tiles

    state = ViewState.from_shape((2, 2, 12)).with_montage_axis(
        2, columns=3, indices=tuple(range(3, 9)), text="3:9"
    )
    plan = make_montage_plan(
        state, axis=2, indices=tuple(range(3, 9)), tile_shape=(2, 2), columns=3
    )
    session = SimpleNamespace(plan=plan, montage_axis=2, view_state=state)

    forward = _predicted_future_tiles(session, 1, limit=3)
    reverse = _predicted_future_tiles(session, -1, limit=3)

    # Forward: the three indices just above the window's max (8) -> 9, 10, 11,
    # clamped at the axis size (12).
    assert tuple(t.source_index for t in forward) == (9, 10, 11)
    # Reverse: the three below the window's min (3) -> 2, 1, 0.
    assert tuple(t.source_index for t in reverse) == (2, 1, 0)
    # None of them overlap the resident window, and all are grid-less.
    assert all(t.montage_index == -1 for t in forward + reverse)
    assert all(t.source_index not in set(range(3, 9)) for t in forward + reverse)
    # No direction -> nothing to predict.
    assert _predicted_future_tiles(session, 0, limit=3) == ()


def test_future_index_prefetch_key_matches_the_later_in_window_demand(qtbot):
    """A prefetched future tile is a byte-identical HIT once it scrolls in.

    Requirement: the speculative store must land under the exact key the real
    demand later computes. This drives the true store/lookup round-trip through
    the evaluator so a key-format drift between speculation and demand fails
    here, where it is cheap to diagnose.
    """

    _clear_arrayscope_settings()
    from arrayscope.operations.evaluator import EvaluationResult
    from arrayscope.window import ArrayScopeWindow
    from arrayscope.window.montage_prefetch import _predicted_future_tiles

    win = ArrayScopeWindow(np.arange(3 * 4 * 12, dtype=np.float32).reshape(3, 4, 12))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.request_operation("reverse", 0)
        state = win.view_state.with_montage_axis(2, columns=3, indices=tuple(range(6)), text="0:6")
        win._set_view_state(state)
        win.render(reason="test-future-key")
        qtbot.waitUntil(
            lambda: win.renderer._frame_session is not None,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        session = win.renderer._frame_session

        future = _predicted_future_tiles(session, 1, limit=1)
        assert future, "a mid-axis window must have an index ahead of it"
        tile = future[0]
        assert tile.source_index == 6  # just past window max (5)

        # Store exactly as the prefetch completion does: a real display image
        # wrapped in an EvaluationResult, keyed by the future tile.
        image = np.full(tuple(session.plan.tile_shape), 7.0, dtype=np.float32)
        from arrayscope.display.slice_engine import make_image_from_slab
        from arrayscope.operations.slabs import request_for_image

        request = request_for_image(tile.view_state)
        display_image = make_image_from_slab(image, request, colormap_lut=session.colormap_lut)
        result = EvaluationResult(
            value=display_image, eval_ms=0.0, slab_shape=image.shape, slab_nbytes=image.nbytes
        )
        win.operation_evaluator.store_montage_tile_result(
            tile,
            montage_axis=session.montage_axis,
            colormap_lut=session.colormap_lut,
            result=result,
        )

        # The later demand, when index 6 scrolls into the window, computes its
        # key from tile_state_for_slice(axis, 6) -- the same value the store
        # used. It MUST be a hit.
        demand_state = win.view_state.tile_state_for_slice(2, 6)
        hit = win.operation_evaluator.cached_montage_tile(
            demand_state,
            montage_axis=2,
            source_index=6,
            colormap_lut=session.colormap_lut,
        )
        assert hit is not None, "prefetched future-index key must match the later demand key"
    finally:
        win.close()


def test_busy_visible_permits_a_speculative_share_toward_the_prediction(qtbot, monkeypatch):
    """A busy visible phase no longer starves the speculative lane.

    Red-first: pre-fix, a busy montage returned ``blocked_visible_busy`` and
    scheduled nothing. The guaranteed share now submits a tiny, direction-led
    prefetch on the SPECULATIVE_RESIDENCY lane -- strictly below visible work
    (non-visible lane, so the kernel ready-heap ranks it after every visible
    task) and carrying a positive expected value so it may run under backlog
    without the kernel ever preferring it to a visible task.
    """

    _clear_arrayscope_settings()
    from arrayscope.core.frame_targets import WorkStart
    from arrayscope.core.prefetch_policy import SliceScrubMomentum
    from arrayscope.kernel import Lane as WorkLane
    from arrayscope.kernel.task import VISIBLE_LANES
    from arrayscope.window import ArrayScopeWindow, montage_prefetch

    win = ArrayScopeWindow(np.arange(3 * 4 * 16, dtype=np.float32).reshape(3, 4, 16))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.request_operation("reverse", 0)
        state = win.view_state.with_montage_axis(2, columns=3, indices=tuple(range(6)), text="0:6")
        win._set_view_state(state)
        win.render(reason="test-busy-share")
        qtbot.waitUntil(
            lambda: win.renderer._frame_session is not None,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        session = win.renderer._frame_session

        # A confident forward scrub.
        momentum = SliceScrubMomentum()
        for step, index in enumerate((0, 1, 2, 3)):
            momentum.observe(index, now=0.1 * step)
        win.renderer._montage_prefetch_momentum = momentum
        assert momentum.plan().direction == 1

        captured = {}

        def capture_prefetch(_evaluate, *, on_done, work_item=None, **_kwargs):
            captured["work_item"] = work_item
            return WorkStart(True)

        # Force the busy predicate and a resolvable shared stage; leave the
        # real _predicted_future_tiles / gate logic under test.
        monkeypatch.setattr(montage_prefetch, "_interaction_active", lambda _owner: False)
        monkeypatch.setattr(montage_prefetch, "_busy", lambda _owner, _session: True)
        monkeypatch.setattr(
            montage_prefetch,
            "_stage_for_tile",
            lambda _owner, _session, _tile: (object(), object(), object()),
        )
        monkeypatch.setattr(win.operation_evaluator, "cached_montage_tile", lambda *_a, **_k: None)
        monkeypatch.setattr(win.prefetch_evaluation_controller, "start_prefetch", capture_prefetch)

        decisions = montage_prefetch.schedule_near_viewport_montage_prefetch(win.renderer, session)

        share = [d for d in decisions if d.decision == "scheduled_speculative_share"]
        assert share, f"busy visible must still grant a speculative share, got {decisions}"
        # The share targets an index ahead of the window (a future arrival),
        # not a resident tile.
        assert all(d.source_index >= 6 for d in share)
        item = captured["work_item"]
        assert item.lane == WorkLane.SPECULATIVE_RESIDENCY
        assert item.lane not in VISIBLE_LANES
        assert item.expected_value > 0.0
        # Supersession keyed on the data identity (source index), so distinct
        # predicted arrivals never collapse into one latest-only survivor.
        assert item.supersession_key[-1] == share[0].source_index
    finally:
        win.close()


def _raw_montage_window(qtbot, shape=(3, 4, 16), *, indices=6):
    """A settled montage frame session over a document with no operations."""

    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape))
    qtbot.addWidget(win)
    _process_events(qtbot)
    # Deliberately no request_operation(): this is the raw-viewing workflow.
    assert not win.document.enabled_operations
    state = win.view_state.with_montage_axis(
        2, columns=3, indices=tuple(range(indices)), text=f"0:{indices}"
    )
    win._set_view_state(state)
    win.render(reason="test-raw-prefetch")
    qtbot.waitUntil(
        lambda: win.renderer._frame_session is not None,
        timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
    )
    return win


def _forward_scrub_momentum(win):
    from arrayscope.core.prefetch_policy import SliceScrubMomentum

    momentum = SliceScrubMomentum()
    for step, index in enumerate((0, 1, 2, 3)):
        momentum.observe(index, now=0.1 * step)
    win.renderer._montage_prefetch_momentum = momentum
    assert momentum.plan().direction == 1
    return momentum


def test_raw_montage_earns_the_directional_speculative_share(qtbot, monkeypatch):
    """A document with no operations must still warm predicted arrivals.

    Red-first: pre-fix, ``schedule_near_viewport_montage_prefetch`` returned a
    single ``blocked_no_stage`` decision for every raw montage, so raw viewing --
    a primary workflow -- got zero prefetch no matter how confident the scrub.
    The stage step exists to avoid recomputing an expensive operation pipeline
    per tile; a raw document has no pipeline, so there is nothing to protect and
    the display-payload evaluation is exactly the work worth warming.
    """

    _clear_arrayscope_settings()
    from arrayscope.core.frame_targets import WorkStart
    from arrayscope.kernel import Lane as WorkLane
    from arrayscope.kernel.task import VISIBLE_LANES
    from arrayscope.window import montage_prefetch

    win = _raw_montage_window(qtbot)
    try:
        session = win.renderer._frame_session
        _forward_scrub_momentum(win)

        captured = {}

        def capture_prefetch(_evaluate, *, on_done, work_item=None, **_kwargs):
            captured["work_item"] = work_item
            return WorkStart(True)

        monkeypatch.setattr(montage_prefetch, "_interaction_active", lambda _owner: False)
        monkeypatch.setattr(montage_prefetch, "_busy", lambda _owner, _session: True)
        monkeypatch.setattr(win.operation_evaluator, "cached_montage_tile", lambda *_a, **_k: None)
        monkeypatch.setattr(win.prefetch_evaluation_controller, "start_prefetch", capture_prefetch)

        decisions = montage_prefetch.schedule_near_viewport_montage_prefetch(win.renderer, session)

        assert not [d for d in decisions if d.decision == "blocked_no_stage"], (
            f"raw montages must no longer be rejected outright, got {decisions}"
        )
        share = [d for d in decisions if d.decision == "scheduled_speculative_share"]
        assert share, f"raw busy visible must grant a speculative share, got {decisions}"
        # The share leads the scroll past the window max (5), exactly as it does
        # for an operation-backed document.
        assert all(d.source_index >= 6 for d in share)
        item = captured["work_item"]
        assert item.lane == WorkLane.SPECULATIVE_RESIDENCY
        assert item.lane not in VISIBLE_LANES
        assert item.supersession_key[-1] == share[0].source_index
    finally:
        win.close()


def test_raw_montage_prefetch_skips_the_stage_probe(qtbot, monkeypatch):
    """Raw speculation must not pay for a stage lookup that cannot succeed.

    ``_stage_for_tile`` runs ``plan_slab`` on the GUI thread per candidate. A
    document with no operations has no retainable cache candidates, so that
    probe can only ever return ``None`` -- pure scheduling-boundary overhead on
    the very path we are trying to make cheap. Raw prefetch takes the no-stage
    evaluation branch directly.
    """

    _clear_arrayscope_settings()
    from arrayscope.core.frame_targets import WorkStart
    from arrayscope.window import montage_prefetch

    win = _raw_montage_window(qtbot)
    try:
        session = win.renderer._frame_session
        _forward_scrub_momentum(win)

        def forbidden_stage(*_args, **_kwargs):
            raise AssertionError("raw prefetch must not probe for an operation stage")

        monkeypatch.setattr(montage_prefetch, "_interaction_active", lambda _owner: False)
        monkeypatch.setattr(montage_prefetch, "_busy", lambda _owner, _session: True)
        monkeypatch.setattr(montage_prefetch, "_stage_for_tile", forbidden_stage)
        monkeypatch.setattr(win.operation_evaluator, "cached_montage_tile", lambda *_a, **_k: None)
        monkeypatch.setattr(
            win.prefetch_evaluation_controller,
            "start_prefetch",
            lambda _evaluate, **_kwargs: WorkStart(True),
        )

        decisions = montage_prefetch.schedule_near_viewport_montage_prefetch(win.renderer, session)

        assert [d for d in decisions if d.decision == "scheduled_speculative_share"], (
            f"raw share must schedule without a stage probe, got {decisions}"
        )
        assert not [d for d in decisions if d.decision == "skipped_stage_missing"], (
            f"a missing stage is normal for raw, not a skip reason, got {decisions}"
        )
    finally:
        win.close()


def test_raw_montage_prefetch_yields_a_near_capacity_display_cache(qtbot, monkeypatch):
    """The busy share's budget guard applies to raw exactly as to operations.

    Warming a speculative raw tile must never evict a visible-path entry, so a
    near-full display cache still yields the whole busy share.
    """

    _clear_arrayscope_settings()
    from types import SimpleNamespace

    from arrayscope.window import montage_prefetch

    win = _raw_montage_window(qtbot)
    try:
        session = win.renderer._frame_session
        _forward_scrub_momentum(win)

        monkeypatch.setattr(montage_prefetch, "_interaction_active", lambda _owner: False)
        monkeypatch.setattr(montage_prefetch, "_busy", lambda _owner, _session: True)
        monkeypatch.setattr(
            win.operation_evaluator,
            "_display_cache",
            SimpleNamespace(bytes_used=100, max_bytes=100),
        )
        monkeypatch.setattr(
            win.prefetch_evaluation_controller,
            "start_prefetch",
            lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("near-capacity raw share must schedule nothing")
            ),
        )

        decisions = montage_prefetch.schedule_near_viewport_montage_prefetch(win.renderer, session)

        assert [d.decision for d in decisions] == ["blocked_budget"], decisions
    finally:
        win.close()


def test_raw_future_index_prefetch_key_matches_the_later_in_window_demand(qtbot):
    """A prefetched raw future tile is a byte-identical HIT once it scrolls in.

    The operation-backed round-trip is covered by
    ``test_future_index_prefetch_key_matches_the_later_in_window_demand``. Raw
    documents key through the same ``display_tile_key`` layout, but nothing had
    ever exercised it speculatively, so this drives the real store/lookup
    round-trip for a document with an empty operation pipeline -- the whole
    premise of raw prefetch is that this key matches.
    """

    _clear_arrayscope_settings()
    from arrayscope.display.slice_engine import make_image_from_slab
    from arrayscope.operations.evaluator import EvaluationResult
    from arrayscope.operations.slabs import request_for_image
    from arrayscope.window.montage_prefetch import _predicted_future_tiles

    win = _raw_montage_window(qtbot, shape=(3, 4, 12))
    try:
        session = win.renderer._frame_session

        future = _predicted_future_tiles(session, 1, limit=1)
        assert future, "a mid-axis window must have an index ahead of it"
        tile = future[0]
        assert tile.source_index == 6  # just past window max (5)

        image = np.full(tuple(session.plan.tile_shape), 7.0, dtype=np.float32)
        request = request_for_image(tile.view_state)
        display_image = make_image_from_slab(image, request, colormap_lut=session.colormap_lut)
        result = EvaluationResult(
            value=display_image, eval_ms=0.0, slab_shape=image.shape, slab_nbytes=image.nbytes
        )
        win.operation_evaluator.store_montage_tile_result(
            tile,
            montage_axis=session.montage_axis,
            colormap_lut=session.colormap_lut,
            result=result,
        )

        demand_state = win.view_state.tile_state_for_slice(2, 6)
        hit = win.operation_evaluator.cached_montage_tile(
            demand_state,
            montage_axis=2,
            source_index=6,
            colormap_lut=session.colormap_lut,
        )
        assert hit is not None, "prefetched raw future-index key must match the later demand key"
    finally:
        win.close()


def test_montage_index_window_observation_reuses_scrub_momentum(monkeypatch):
    from types import SimpleNamespace

    import arrayscope.window.render_prefetch as render_prefetch
    from arrayscope.core.view_state import ViewState
    from arrayscope.window.render_prefetch import RenderPrefetchMixin

    times = iter((0.0, 0.1, 0.2))
    monkeypatch.setattr(render_prefetch, "monotonic", lambda: next(times))
    owner = SimpleNamespace()
    initial = ViewState.from_shape((2, 2, 8)).with_montage_axis(
        2, columns=4, indices=(0, 1, 2, 3), text="0:4"
    )
    forward = initial.with_montage_axis(2, columns=4, indices=(1, 2, 3, 4), text="1:5")
    reverse = initial.with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text="0:4")

    observe = RenderPrefetchMixin._observe_montage_prefetch_momentum
    observe(owner, initial, initial)
    observe(owner, initial, forward)
    assert owner._montage_prefetch_momentum.plan().direction == 1

    observe(owner, forward, reverse)
    assert owner._montage_prefetch_momentum.plan().direction == -1


def _hold_real_visible_work(win):
    """Keep one real visible task active until the test releases it."""

    from arrayscope.core.frame_targets import FrameTarget
    from arrayscope.kernel import Lane as WorkLane
    from arrayscope.kernel import WorkItem

    started = threading.Event()
    release = threading.Event()

    def work():
        started.set()
        release.wait(timeout=INTERACTION_SETTLE_HARD_LIMIT_MS / 1000.0)

    target = FrameTarget("busy", None, "presentation", "exact-visible")
    win.visible_evaluation_controller.start_latest(
        work,
        key="busy",
        priority=0,
        replace_group="visible",
        frame_target=target,
        supersession_key="visible-image",
        supersession_value="busy",
        work_item=WorkItem(
            key=("visible", "busy"),
            lane=WorkLane.VISIBLE_MATERIALIZATION,
            frame_target=target,
            supersession_key="visible-image",
            supersession_value="busy",
        ),
        on_done=lambda _value: None,
    )
    return started, release


def test_prefetch_skips_while_visible_controller_busy(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    release_busy = None
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True
        )
        win._active_slice_axis = 2
        started, release_busy = _hold_real_visible_work(win)
        qtbot.waitUntil(started.is_set, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        before_scheduled = win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 2), None)

        # The gate decision is synchronous: blocked, nothing scheduled, and
        # (post re-arm fix) the request is retained for the drain retry
        # rather than dropped.
        assert win.prefetch_evaluation_controller.diagnostics().prefetch_visible_busy_blocked >= 1
        assert (
            win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled
            == before_scheduled
        )
        assert getattr(win.renderer, "_pending_prefetch_request", None) is not None
    finally:
        if release_busy is not None:
            release_busy.set()
        win.close()


def test_prefetch_gated_by_busy_visible_runs_after_drain(qtbot):
    """A visible-busy request is retained and re-arms on the real drain.

    Regression gate: the busy path used to consume the pending request and
    never re-arm, leaving prefetch_scheduled=0 across a whole scrub session.
    Hold an actual visible task so the controller, governor, and wake-up all
    observe one coherent lifecycle instead of forging a contradictory busy
    predicate while no completion exists to release its parked lane.
    """

    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    release_busy = None
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True
        )
        win._active_slice_axis = 2
        # Let startup evaluation drain first so the speculative dispatch is
        # not parked behind real work under parallel test load.
        qtbot.waitUntil(
            lambda: not win.visible_evaluation_controller.is_busy(),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        started, release_busy = _hold_real_visible_work(win)
        qtbot.waitUntil(started.is_set, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

        state = win.view_state.with_slice(2, 2)
        win.renderer._prefetch_nearby_slices(state, None)

        # The request must actually hit the closed gate first...
        assert win.prefetch_evaluation_controller.diagnostics().prefetch_visible_busy_blocked >= 1
        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled == 0
        assert getattr(win.renderer, "_pending_prefetch_request", None) is not None, (
            "the gated dispatch must retain the request, not consume it"
        )

        # ...and the retained request must run once the visible work drains.
        release_busy.set()
        qtbot.waitUntil(
            lambda: win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled >= 1,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        qtbot.waitUntil(
            lambda: (
                win.operation_evaluator.cached_display_tile(win.view_state.with_slice(2, 1))
                is not None
                or win.operation_evaluator.cached_display_tile(win.view_state.with_slice(2, 3))
                is not None
            ),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
    finally:
        if release_busy is not None:
            release_busy.set()
        win.close()


def test_prefetch_burst_coalesces_and_momentum_observes_despite_gating(qtbot):
    """Rapid gated scrub steps collapse to one latest-wins retry.

    Momentum observes at SCHEDULE time, so the gated steps still feed the
    direction model (pre-fix: observe only ran inside the gated body, so a
    busy scrub session left momentum with no data at all).
    """

    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 16, dtype=float).reshape(3, 4, 16))
    qtbot.addWidget(win)
    release_busy = None
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True
        )
        win._active_slice_axis = 2
        qtbot.waitUntil(
            lambda: not win.visible_evaluation_controller.is_busy(),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        started, release_busy = _hold_real_visible_work(win)
        qtbot.waitUntil(started.is_set, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

        for index in (2, 3, 4, 5):
            win.renderer._schedule_prefetch_nearby_slices(win.view_state.with_slice(2, index), None)
        _process_events(qtbot, count=10)

        momentum = getattr(win.renderer, "_prefetch_momentum", None)
        assert momentum is not None, "gated scrub steps must still feed the momentum model"
        assert momentum.direction == 1
        assert momentum.streak >= 3
        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled == 0

        release_busy.set()
        qtbot.waitUntil(
            lambda: win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled >= 1,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        _process_events(qtbot, count=40)
        diagnostics = win.operation_evaluator.display_cache_diagnostics()
        # Latest-wins coalescing: one retained request, one momentum-planned
        # fan-out (max_depth=4) — not one prefetch per gated step.
        assert diagnostics.prefetch_scheduled <= 4
        assert getattr(win.renderer, "_pending_prefetch_request", None) is None
    finally:
        if release_busy is not None:
            release_busy.set()
        win.close()


def test_cost_aware_prefetch_allows_cheap_operation_backed_stack(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True
        )
        win.request_operation("reverse", 0)
        _process_events(qtbot, count=20)
        win._active_slice_axis = 2
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 2), None)
        _process_events(qtbot, count=40)
        after = win.operation_evaluator.display_cache_diagnostics()

        assert after.prefetch_scheduled > 0
        assert after.prefetch_stored > 0
    finally:
        win.close()


def test_cost_aware_prefetch_blocks_expensive_fft_stack(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((96, 96, 96), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True, render_memory_budget_mb=128
        )
        win.request_operation("centered_fft", 2)
        _process_events(qtbot, count=20)
        win._active_slice_axis = 2
        before = win.operation_evaluator.display_cache_diagnostics()
        before_blocked = win.prefetch_evaluation_controller.diagnostics().prefetch_cost_blocked
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 1), None)
        _process_events(qtbot, count=20)
        after = win.operation_evaluator.display_cache_diagnostics()

        assert after.prefetch_scheduled == before.prefetch_scheduled
        assert (
            win.prefetch_evaluation_controller.diagnostics().prefetch_cost_blocked > before_blocked
        )
    finally:
        win.close()


def test_slice_change_schedules_nearby_prefetch_when_enabled(qtbot):
    """The opt-in slice prefetcher rides every live slice change.

    Regression guard: the scheduler call was severed when the legacy
    normal-image update path was deleted, leaving the setting silently dead.
    """

    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 4 * 6, dtype=np.float32).reshape(4, 4, 6))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme, prefetch_nearby_slices=True
        )
        before = win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled
        win._apply_slice_state(
            2,
            win.view_state.with_slice(2, 3),
            reason="test-slice-prefetch-wiring",
            interactive=True,
            immediate_axis_only=True,
        )
        qtbot.waitUntil(
            lambda: win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled > before,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
    finally:
        win.close()


def test_slice_change_without_setting_skips_prefetch(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 4 * 6, dtype=np.float32).reshape(4, 4, 6))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        before = win.operation_evaluator.display_cache_diagnostics().prefetch_skipped
        win._apply_slice_state(
            2,
            win.view_state.with_slice(2, 2),
            reason="test-slice-prefetch-off",
            interactive=True,
            immediate_axis_only=True,
        )
        qtbot.waitUntil(
            lambda: win.operation_evaluator.display_cache_diagnostics().prefetch_skipped > before,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
    finally:
        win.close()


def test_prefetch_menu_action_exists_and_toggles_setting(qtbot):
    """The Performance-menu toggle must stay wired to the live setting.

    Regression guard: the handler survived the legacy-path removal but the
    menu action itself was lost, leaving the setting unreachable in the UI.
    """

    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = getattr(win, "_prefetch_nearby_slices_action", None)
        assert action is not None, "Performance menu lost the prefetch toggle"
        assert action.isCheckable()
        assert not action.isChecked()
        assert not win.app_settings.prefetch_nearby_slices
        action.setChecked(True)
        _process_events(qtbot)
        assert win.app_settings.prefetch_nearby_slices
        action.setChecked(False)
        _process_events(qtbot)
        assert not win.app_settings.prefetch_nearby_slices
    finally:
        win.close()
