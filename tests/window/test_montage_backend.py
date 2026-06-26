from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.app.settings_state import MontageDisplayBackendChoice
from arrayscope.display.backend_contract import ImageViewBackendCapabilities
from arrayscope.window.montage_backend import choose_montage_backend
from arrayscope.window.montage_payload_cache import (
    base_tile_source_id as _base_tile_source_id,
    limited_payload_cache as _limited_payload_cache,
    payload_lod_matches as _payload_lod_matches,
    payload_compatible_with_tile as _payload_compatible_with_tile,
)
from arrayscope.window.montage_viewport import (
    MontageViewportPlan,
    montage_session_key,
    montage_viewport_update_delay_ms as _montage_viewport_update_delay_ms,
)


def _geometry():
    return SimpleNamespace(montage=object())


def test_auto_small_scalar_montage_uses_canvas():
    decision = choose_montage_backend(_geometry(), np.zeros((64, 64), dtype=np.float32))

    assert decision.backend == "canvas"
    assert "small montage" in decision.reason


def test_auto_large_scalar_montage_stays_canvas_until_upload_is_slow():
    data = np.zeros((1500, 1500), dtype=np.float32)

    decision = choose_montage_backend(_geometry(), data)
    slow = choose_montage_backend(_geometry(), data, previous_upload_ms=150.0, very_slow_upload_ms=100.0)

    assert decision.backend == "canvas"
    assert slow.backend == "tile_layer"


def test_auto_large_scalar_vispy_montage_uses_tile_layer_to_avoid_full_uploads():
    data = np.zeros((1500, 1500), dtype=np.float32)

    decision = choose_montage_backend(_geometry(), data, renderer_backend="vispy")

    assert decision.backend == "tile_layer"
    assert decision.expected_tile_layer is True
    assert "prefers tiled montages" in decision.reason


def test_auto_small_scalar_vispy_montage_uses_tile_layer():
    data = np.zeros((64, 64), dtype=np.float32)

    decision = choose_montage_backend(_geometry(), data, renderer_backend="vispy")

    assert decision.backend == "tile_layer"
    assert decision.expected_tile_layer is True
    assert "canvas composition" in decision.reason


def test_vispy_cannot_be_forced_through_montage_canvas():
    data = np.zeros((64, 64), dtype=np.float32)

    decision = choose_montage_backend(
        _geometry(),
        data,
        setting=MontageDisplayBackendChoice.CANVAS,
        renderer_backend="vispy",
    )

    assert decision.backend == "tile_layer"
    assert decision.expected_tile_layer is True
    assert "does not support montage canvas" in decision.reason
    assert "unavailable" in decision.warning


def test_canvas_capability_controls_manual_fallback_without_backend_name_checks():
    data = np.zeros((64, 64), dtype=np.float32)
    capabilities = ImageViewBackendCapabilities(
        name="future-gpu-backend",
        direct_montage_tile_payloads=True,
        prefers_tiled_montages=True,
        supports_montage_canvas=False,
    )

    decision = choose_montage_backend(
        _geometry(),
        data,
        setting=MontageDisplayBackendChoice.CANVAS,
        renderer_capabilities=capabilities,
    )

    assert decision.backend == "tile_layer"
    assert "future-gpu-backend" in decision.reason


def test_first_vispy_display_batch_limit_uses_governed_upsert_limit(monkeypatch):
    import arrayscope.window.montage_renderer as montage_renderer

    session = SimpleNamespace(visible_tiles=tuple(range(100)))
    monkeypatch.setattr(
        montage_renderer,
        "_persistent_tile_upsert_limits",
        lambda _window, _session, *, first_display_commit: {"max_upserts": 3},
    )

    assert montage_renderer._first_vispy_display_batch_limit(object(), session) == 3


def test_auto_policy_uses_capability_instead_of_backend_name():
    data = np.zeros((1500, 1500), dtype=np.float32)
    capabilities = ImageViewBackendCapabilities(
        name="future-gpu-backend",
        direct_montage_tile_payloads=True,
        prefers_tiled_montages=True,
        persistent_tile_residency=True,
    )

    decision = choose_montage_backend(
        _geometry(),
        data,
        renderer_backend="pyqtgraph",
        renderer_capabilities=capabilities,
    )

    assert decision.backend == "tile_layer"
    assert "future-gpu-backend" in decision.reason


def test_auto_prefers_tiled_capability_without_direct_payloads_stays_canvas_until_large():
    data = np.zeros((64, 64), dtype=np.float32)
    capabilities = ImageViewBackendCapabilities(
        name="future-gpu-backend",
        direct_montage_tile_payloads=False,
        prefers_tiled_montages=True,
        persistent_tile_residency=True,
    )

    decision = choose_montage_backend(
        _geometry(),
        data,
        renderer_backend="pyqtgraph",
        renderer_capabilities=capabilities,
    )

    assert decision.backend == "canvas"


def test_auto_preserves_vispy_tile_layer_mode():
    data = np.zeros((64, 64), dtype=np.float32)

    decision = choose_montage_backend(_geometry(), data, current_mode="vispy_tile_layer")

    assert decision.backend == "tile_layer"
    assert "preserving" in decision.reason


def test_interactive_montage_commit_is_timer_coalesced(qt_app, monkeypatch):
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.montage_renderer import MontageRenderMixin
    from arrayscope.window.montage_session import MontageRenderSession

    class Window(QtCore.QObject, MontageRenderMixin):
        def __init__(self):
            super().__init__()
            self.view_state = ViewState.from_shape((2, 2, 1)).with_montage_axis(2, indices=(0,), text=":")
            self._commits = 0

        def _commit_montage_session_canvas(self, session, *, force=False):
            self._commits += 1

    win = Window()
    plan = make_montage_plan(win.view_state, axis=2, indices=(0,), tile_shape=(2, 2), columns=1)
    session = MontageRenderSession(
        session_id=1,
        key=("session",),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=(0,),
        plan=plan,
        view_state=win.view_state,
        document=object(),
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    session.display_committed = True
    win._montage_session = session
    win._viewport_interaction_active = True

    win._schedule_montage_canvas_commit(session, force=True)

    assert win._commits == 0
    assert session.final_commit_pending is True
    assert win._montage_commit_timer.isActive()


def test_interactive_viewport_prunes_stale_montage_tile_work(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import MontageTileState, make_montage_plan
    from arrayscope.window.montage_renderer import MontageRenderMixin
    from arrayscope.window.montage_session import MontageRenderSession

    class Controller:
        def __init__(self):
            self.groups = []

        def clear_group(self, group):
            self.groups.append(group)

    class Window(QtCore.QObject, MontageRenderMixin):
        pass

    state = ViewState.from_shape((2, 2, 8)).with_montage_axis(2, indices=tuple(range(8)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(8)), tile_shape=(2, 2), columns=8, gap=1)
    controller = Controller()
    session = MontageRenderSession(
        session_id=7,
        key=("session",),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=tuple(range(8)),
        plan=plan,
        view_state=state,
        document=object(),
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=((0.0, 2.0), (0.0, 2.0)),
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=(plan.tiles[0],),
        rendered_tiles={},
        loading_tiles={7},
        skipped_tiles=set(),
        pending_tiles=[plan.tiles[1], plan.tiles[7]],
    )
    session.active_tile_requests.add(7)
    session.tile_states = [MontageTileState.UNLOADED for _tile in plan.tiles]
    session.tile_states[7] = MontageTileState.LOADING
    win = Window()
    win._montage_session = session
    win.view_state = state
    win.montage_tile_evaluation_controller = controller
    win._viewport_interaction_active = True

    win._prune_stale_montage_tile_work(session)

    assert [int(tile.montage_index) for tile in session.pending_tiles] == [1]
    assert 7 in session.loading_tiles
    assert 7 in session.active_tile_requests
    assert session.tile_states[7] == MontageTileState.LOADING
    assert controller.groups == []


def test_interactive_viewport_expansion_chunks_cached_tile_resolution(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.montage_renderer import MontageRenderMixin
    from arrayscope.window.montage_session import MontageRenderSession

    class Window(QtCore.QObject, MontageRenderMixin):
        def __init__(self, document, state, viewport_plan):
            super().__init__()
            self.document = document
            self.view_state = state
            self._viewport_plan = viewport_plan
            self._viewport_interaction_active = True
            self._montage_viewport_addition_batch_size = 3
            self.resolved_batches = []
            self.tile_schedules = 0
            self.commits = 0
            self.img_view = SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(
                    name="vispy",
                    direct_montage_tile_payloads=True,
                    persistent_tile_residency=True,
                    shader_windowing=True,
                ),
                montageDisplayMode=lambda: "vispy_tile_layer",
            )

        def _montage_viewport_plan(self, view_state):
            return self._viewport_plan

        def _evaluation_colormap_lut(self, view_state, *, shader_display=None):
            return None

        def _resolve_montage_tiles_from_cache(self, tiles, **_kwargs):
            batch = tuple(tiles)
            self.resolved_batches.append(tuple(int(tile.montage_index) for tile in batch))
            return (), batch

        def _schedule_montage_canvas_commit(self, session, *, force=False):
            self.commits += 1

        def _schedule_montage_tiles(self, session):
            self.tile_schedules += 1

    document = ArrayDocument(np.zeros((2, 2, 10), dtype=np.float32))
    state = ViewState.from_shape(document.current_shape).with_montage_axis(2, columns=10, indices=tuple(range(10)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(10)), tile_shape=(2, 2), columns=10)
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(10)),
        viewport_shape=(4, 40),
        tile_shape=(2, 2),
        plan=plan,
        view_range=((-1.0, 40.0), (-1.0, 4.0)),
        shader_display=True,
        persistent_tile_residency=True,
    )
    session = MontageRenderSession(
        session_id=11,
        key=montage_session_key(_document_key(document), state, viewport_plan, None),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=tuple(range(10)),
        plan=plan,
        view_state=state,
        document=document,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(4, 40),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=(),
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    win = Window(document, state, viewport_plan)
    win._montage_session = session

    assert win._try_update_montage_viewport_only() is True

    assert len(win.resolved_batches) == 3
    resolved = [tile for batch in win.resolved_batches for tile in batch]
    assert len(resolved) == 3
    pending = [int(tile.montage_index) for tile in session.pending_tiles]
    assert pending == resolved
    assert pending[0] in {4, 5}
    assert session.loading_tiles == set()
    assert win.tile_schedules == 0
    assert win._montage_viewport_update_pending is True
    assert win._montage_viewport_continue_immediately is True
    assert win._last_montage_viewport_deferred_additions == 7


def test_quiet_viewport_update_schedules_deferred_missing_tiles(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.montage_renderer import MontageRenderMixin
    from arrayscope.window.montage_session import MontageRenderSession

    class Window(QtCore.QObject, MontageRenderMixin):
        def __init__(self, document, state, viewport_plan):
            super().__init__()
            self.document = document
            self.view_state = state
            self._viewport_plan = viewport_plan
            self.tile_schedules = 0
            self.img_view = SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(
                    name="vispy",
                    direct_montage_tile_payloads=True,
                    persistent_tile_residency=True,
                    shader_windowing=True,
                ),
                montageDisplayMode=lambda: "vispy_tile_layer",
            )

        def _montage_viewport_plan(self, view_state):
            return self._viewport_plan

        def _evaluation_colormap_lut(self, view_state, *, shader_display=None):
            return None

        def _schedule_montage_tiles(self, session):
            self.tile_schedules += 1

        def _schedule_montage_canvas_commit(self, session, *, force=False):
            pass

    document = ArrayDocument(np.zeros((2, 2, 4), dtype=np.float32))
    state = ViewState.from_shape(document.current_shape).with_montage_axis(2, columns=4, indices=tuple(range(4)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(4)), tile_shape=(2, 2), columns=4)
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(4)),
        viewport_shape=(4, 16),
        tile_shape=(2, 2),
        plan=plan,
        view_range=((-1.0, 16.0), (-1.0, 4.0)),
        shader_display=True,
        persistent_tile_residency=True,
    )
    session = MontageRenderSession(
        session_id=12,
        key=montage_session_key(_document_key(document), state, viewport_plan, None),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=tuple(range(4)),
        plan=plan,
        view_state=state,
        document=document,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(4, 16),
        view_range=viewport_plan.view_range,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles={0, 1, 2, 3},
        skipped_tiles=set(),
        pending_tiles=list(plan.tiles),
    )
    win = Window(document, state, viewport_plan)
    win._montage_session = session
    win._viewport_interaction_active = False

    assert win._try_update_montage_viewport_only() is True

    assert win.tile_schedules == 1


def test_nonpersistent_tile_layer_viewport_update_preserves_level_target(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.operations.evaluator import _document_key
    from arrayscope.operations.pipeline import ArrayDocument
    from arrayscope.window.montage_renderer import MontageRenderMixin
    from arrayscope.window.montage_session import MontageRenderSession

    class Window(QtCore.QObject, MontageRenderMixin):
        def __init__(self, document, state, viewport_plan):
            super().__init__()
            self.document = document
            self.view_state = state
            self._viewport_plan = viewport_plan
            self.tile_schedules = 0
            self.commits = 0
            self.img_view = SimpleNamespace(
                rendering_capabilities=ImageViewBackendCapabilities(
                    name="pyqtgraph",
                    direct_montage_tile_payloads=True,
                    persistent_tile_residency=False,
                    shader_windowing=False,
                ),
                montageDisplayMode=lambda: "tile_layer",
            )

        def _montage_viewport_plan(self, view_state):
            return self._viewport_plan

        def _evaluation_colormap_lut(self, view_state, *, shader_display=None):
            return None

        def _schedule_montage_tiles(self, session):
            self.tile_schedules += 1

        def _schedule_montage_canvas_commit(self, session, *, force=False):
            self.commits += 1

    document = ArrayDocument(np.zeros((2, 2, 4), dtype=np.float32))
    state = ViewState.from_shape(document.current_shape).with_montage_axis(2, columns=4, indices=tuple(range(4)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(4)), tile_shape=(2, 2), columns=4)
    viewport_plan = MontageViewportPlan(
        axis=2,
        all_indices=tuple(range(4)),
        viewport_shape=(4, 16),
        tile_shape=(2, 2),
        plan=plan,
        view_range=((-1.0, 16.0), (-1.0, 4.0)),
        shader_display=False,
        persistent_tile_residency=False,
    )
    session = MontageRenderSession(
        session_id=13,
        key=montage_session_key(_document_key(document), state, viewport_plan, None),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=tuple(range(4)),
        plan=plan,
        view_state=state,
        document=document,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(4, 16),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={int(tile.montage_index): SimpleNamespace(tile=tile) for tile in plan.tiles},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    session.level_generation.target_levels = (2.0, 4.0)
    session.set_level_update_pending(True)
    win = Window(document, state, viewport_plan)
    win._montage_session = session
    win._viewport_interaction_active = False

    assert win._try_update_montage_viewport_only() is True

    assert win._montage_session is session
    assert session.level_generation.target_levels == (2.0, 4.0)


def test_hover_priority_retarget_timer_changes_next_pending_tile(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.montage_renderer import MontageRenderMixin
    from arrayscope.window.montage_session import MontageRenderSession

    class Window(QtCore.QObject, MontageRenderMixin):
        def __init__(self, state, viewport_plan):
            super().__init__()
            self.view_state = state
            self._viewport_plan = viewport_plan
            self.scheduled = []
            self.img_view = SimpleNamespace(rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"))

        def _montage_viewport_plan(self, view_state):
            return self._viewport_plan

        def _is_current_montage_session(self, session_id, key):
            return True

        def _schedule_montage_tiles(self, session):
            tile = session.next_tile()
            if tile is not None:
                self.scheduled.append(int(tile.montage_index))

    state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, columns=4, indices=tuple(range(4)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(4)), tile_shape=(2, 2), columns=4)
    viewport_plan = SimpleNamespace(priority_focus=(10.0, 1.0))
    session = MontageRenderSession(
        session_id=12,
        key="key",
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=tuple(range(4)),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(4, 16),
        view_range=((0.0, 12.0), (0.0, 4.0)),
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=list(plan.tiles),
    )
    win = Window(state, viewport_plan)
    win._montage_session = session

    win._run_montage_priority_retarget()

    assert win.scheduled == [3]


def test_tiled_commit_syncs_hover_geometry_after_backend_ack(qt_app):
    from dataclasses import replace
    from pyqtgraph.Qt import QtCore
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import CommittedDisplayFrame, DisplayFrameKey
    from arrayscope.window.montage_renderer import MontageRenderMixin

    class Window(QtCore.QObject, MontageRenderMixin):
        def _set_committed_display_frame(self, frame):
            self._committed_display_frame = frame

    state = ViewState.from_shape((2, 2, 1)).with_montage_axis(2, columns=1, indices=(0,), text=":")
    loading = DisplayGeometry(
        view_state=state,
        display_shape=(2, 2),
        montage=MontageGeometry(indices=(0,), tile_shape=(2, 2), columns=1, rows=1, gap=0),
        montage_tile_states=(MontageTileState.LOADING,),
    )
    loaded = replace(loading, montage_tile_states=(MontageTileState.LOADED,))
    frame = CommittedDisplayFrame(
        data=np.zeros((2, 2), dtype=np.float32),
        histogram_data=None,
        geometry=loading,
        levels=(0.0, 1.0),
        histogram_range=(0.0, 1.0),
        key=DisplayFrameKey(("doc",), ("view",), 1),
    )
    win = Window()
    win.display_geometry = loading
    win._committed_display_frame = frame

    win._sync_committed_montage_geometry(loaded)

    assert win.display_geometry.montage_tile_states == (MontageTileState.LOADED,)
    assert win._committed_display_frame.geometry.montage_tile_states == (MontageTileState.LOADED,)
    assert win._committed_display_frame.scene.geometry == loaded


def test_persistent_tile_residency_defers_tile_discovery_behind_camera_updates():
    from arrayscope.window.montage_viewport import montage_viewport_retarget_policy
    from arrayscope.window.montage_renderer import _persistent_direct_tile_layer_backend

    capabilities = ImageViewBackendCapabilities(
        name="vispy",
        direct_montage_tile_payloads=True,
        persistent_tile_residency=True,
    )
    persistent_nonvispy = ImageViewBackendCapabilities(
        name="future-backend",
        direct_montage_tile_payloads=True,
        persistent_tile_residency=True,
        shader_windowing=True,
    )
    direct_nonpersistent = ImageViewBackendCapabilities(
        name="pyqtgraph",
        direct_montage_tile_payloads=True,
        persistent_tile_residency=False,
    )
    persistent_without_shader = ImageViewBackendCapabilities(
        name="resident-cpu-backend",
        direct_montage_tile_payloads=True,
        persistent_tile_residency=True,
        shader_windowing=False,
    )
    window = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=capabilities,
            montageDisplayMode=lambda: "vispy_tile_layer",
        )
    )
    fallback = SimpleNamespace(
        img_view=SimpleNamespace(
            rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"),
            montageDisplayMode=lambda: "canvas",
        )
    )

    assert _montage_viewport_update_delay_ms(window) == 90
    assert _montage_viewport_update_delay_ms(fallback) == 120
    assert montage_viewport_retarget_policy(capabilities, "vispy_tile_layer").coverage_margin_tiles == 1
    assert (
        _persistent_direct_tile_layer_backend(
            SimpleNamespace(img_view=SimpleNamespace(rendering_capabilities=persistent_nonvispy)),
            SimpleNamespace(),
        )
        is True
    )
    assert (
        _persistent_direct_tile_layer_backend(
            SimpleNamespace(img_view=SimpleNamespace(rendering_capabilities=persistent_without_shader)),
            SimpleNamespace(),
        )
        is False
    )
    assert montage_viewport_retarget_policy(direct_nonpersistent, "tile_layer").enabled is True
    assert montage_viewport_retarget_policy(direct_nonpersistent, "tile_layer").coverage_margin_tiles == 0
    assert montage_viewport_retarget_policy(direct_nonpersistent, "canvas").enabled is False


def test_recent_payload_cache_is_keyed_by_semantic_source_identity():
    payload = SimpleNamespace(source_id=(("montage_tile", "doc", 2), "texture_kind", "complex_rg32f", "shader", None, "lod", 4, 2, 1))
    other = SimpleNamespace(source_id=("plain", 1))

    cache = _limited_payload_cache({}, {0: payload, 1: other}, limit=8)

    assert _base_tile_source_id(payload.source_id) == ("montage_tile", "doc", 2)
    assert cache[("montage_tile", "doc", 2)] is payload
    assert cache[("plain", 1)] is other


def test_recent_payload_cache_requires_matching_lod_factor():
    lod4 = SimpleNamespace(factor=4)
    lod1 = SimpleNamespace(factor=1)

    assert _payload_lod_matches(SimpleNamespace(lod=lod4), 4)
    assert not _payload_lod_matches(SimpleNamespace(lod=lod1), 4)


def test_previous_complex_shader_payload_must_carry_complex_texture():
    from arrayscope.core.view_state import ChannelMode, ViewState
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.shader_mapping import TexturePlaneKind

    state = ViewState.from_shape((2, 2, 4)).with_channel(ChannelMode.COMPLEX)
    stale_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    payload = DisplayTilePayload(
        0,
        0,
        stale_rgb,
        np.zeros((2, 2), dtype=np.float32),
        ("source", 0),
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=stale_rgb,
    )

    assert not _payload_compatible_with_tile(payload, state, shader_display=True)


def test_previous_complex_shader_payload_accepts_complex_texture():
    from arrayscope.core.view_state import ChannelMode, ViewState
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.shader_mapping import TexturePlaneKind

    state = ViewState.from_shape((2, 2, 4)).with_channel(ChannelMode.COMPLEX)
    texture = np.ones((2, 2), dtype=np.complex64)
    payload = DisplayTilePayload(
        0,
        0,
        texture,
        np.ones((2, 2), dtype=np.float32),
        ("source", 0),
        texture_data=texture,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=texture,
    )

    assert _payload_compatible_with_tile(payload, state, shader_display=True)


def test_tiled_payload_source_id_changes_when_texture_content_changes():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import RenderedTile, make_montage_plan
    from arrayscope.window.montage_session import MontageRenderSession

    state = ViewState.from_shape((2, 2, 1)).with_montage_axis(2, columns=1, indices=(0,), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0,), tile_shape=(2, 2), columns=1)
    tile = plan.tiles[0]

    def rendered(value):
        image = np.full((2, 2), float(value), dtype=np.float32)
        return RenderedTile(
            tile=tile,
            image=image,
            histogram_data=image.copy(),
            eval_ms=0.0,
            slab_shape=image.shape,
            slab_nbytes=int(image.nbytes),
        )

    session = MontageRenderSession(
        session_id=1,
        key=("session",),
        render_generation=1,
        level_key=("levels",),
        level_expected_indices=(0,),
        plan=plan,
        view_state=state,
        document=object(),
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode="relative",
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={0: rendered(1.0)},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )

    first = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]
    session.rendered_tiles[0] = rendered(2.0)
    second = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    assert _base_tile_source_id(first.source_id) == ("tile", 0)
    assert _base_tile_source_id(second.source_id) == ("tile", 0)
    assert first.source_id != second.source_id


def test_auto_large_rgb_montage_uses_tile_layer():
    data = np.zeros((1500, 1500, 3), dtype=np.uint8)

    decision = choose_montage_backend(_geometry(), data)

    assert decision.backend == "tile_layer"
    assert decision.expected_tile_layer is True


def test_forced_canvas_warns_for_large_rgb_montage():
    data = np.zeros((1500, 1500, 3), dtype=np.uint8)

    decision = choose_montage_backend(_geometry(), data, setting=MontageDisplayBackendChoice.CANVAS)

    assert decision.backend == "canvas"
    assert "manual" in decision.warning


def test_forced_tile_layer_wins():
    decision = choose_montage_backend(
        _geometry(),
        np.zeros((16, 16), dtype=np.float32),
        setting=MontageDisplayBackendChoice.TILE_LAYER,
    )

    assert decision.backend == "tile_layer"
    assert decision.reason == "user forced tile layer"


def test_stage_wait_release_falls_back_to_direct_tile_evaluation():
    pytest.importorskip("pyqtgraph")
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.window.montage_renderer import MontageRenderMixin

    class _Window(MontageRenderMixin):
        pass

    class _Session:
        def __init__(self):
            self.stage_fan_in = StageFanInState(
                active_requests={"stage-key"},
                attached_requests={"stage-key"},
                waiting_tiles={"stage-key": [SimpleNamespace(montage_index=3)]},
                tile_stage_keys={3: "stage-key"},
            )
            self.rendered_tiles = {}
            self.skipped_tiles = set()
            self.pending_tiles = []
            self.loading_tiles = set()

        def mark_loading(self, tile):
            self.loading_tiles.add(int(tile.montage_index))

    session = _Session()

    _Window()._release_stage_waiting_tiles_to_direct(session, "stage-key")

    assert session.stage_fan_in.active_requests == set()
    assert session.stage_fan_in.attached_requests == set()
    assert session.stage_fan_in.waiting_tiles == {}
    assert session.stage_fan_in.tile_stage_keys == {}
    assert [tile.montage_index for tile in session.pending_tiles] == [3]
    assert session.loading_tiles == {3}


def test_stage_activation_uses_bounded_batches_without_losing_waiting_tiles():
    pytest.importorskip("pyqtgraph")
    from arrayscope.core.gui_callback_budget import GuiCallbackBudget
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.window.montage_renderer import MontageRenderMixin

    class _Window(MontageRenderMixin):
        pass

    class _Session:
        def __init__(self):
            self.stage_fan_in = StageFanInState(
                waiting_tiles={"stage-key": [SimpleNamespace(montage_index=index) for index in range(5)]},
                values={"stage-key": "stage-value"},
            )
            self.rendered_tiles = {}
            self.skipped_tiles = set()
            self.pending_tiles = []
            self.loading_tiles = set()

        def mark_loading(self, tile):
            self.loading_tiles.add(int(tile.montage_index))

    session = _Session()
    budget = GuiCallbackBudget("montage_stage_wait", item_cap=3)

    _Window()._activate_montage_stage_value(session, "stage-key", "stage-value", budget=budget)

    assert [tile.montage_index for tile in session.pending_tiles] == [0, 1, 2]
    assert [tile.montage_index for tile in session.stage_fan_in.waiting_tiles["stage-key"]] == [3, 4]
    assert budget.processed_items == 3


def test_ready_display_commit_refreshes_stale_commit_token_at_source(qt_app):
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtCore
    from arrayscope.window.montage_renderer import MontageRenderMixin, _montage_work_token

    class _Window(QtCore.QObject, MontageRenderMixin):
        def __init__(self):
            super().__init__()
            self.img_view = SimpleNamespace(rendering_capabilities=ImageViewBackendCapabilities(name="pyqtgraph"))

        def _is_current_montage_session(self, session_id, key):
            return True

        def _flush_montage_canvas_commit(self):
            return None

    session = SimpleNamespace(
        session_id=1,
        key="session",
        render_generation=2,
        payload_revision=3,
        level_revision=4,
        viewport_revision=5,
        dirty_tiles=[0],
        dirty_rects=[],
        dirty_payloads={},
        pending_payload_upserts={},
        pending_removals=set(),
        display_committed=False,
        final_commit_pending=False,
        flush_pending=False,
        has_pending_level_update=lambda: False,
        has_stale_level_presentations=lambda: False,
    )
    win = _Window()
    win._montage_session = session
    win._montage_commit_token = ("commit", "stale")

    win._schedule_montage_ready_display_commit(session)

    assert win._montage_commit_token == _montage_work_token(session, "commit")
    assert session.final_commit_pending is True
    assert session.flush_pending is True


def test_montage_commit_token_ignores_viewport_only_revision_changes():
    from arrayscope.window.montage_renderer import _montage_work_token

    session = SimpleNamespace(
        session_id=1,
        key="session",
        render_generation=2,
        payload_revision=3,
        level_revision=4,
        viewport_revision=5,
    )
    commit_token = _montage_work_token(session, "commit")
    priority_token = _montage_work_token(session, "priority_retarget")

    session.viewport_revision += 1

    assert _montage_work_token(session, "commit") == commit_token
    assert _montage_work_token(session, "priority_retarget") != priority_token


def test_native_tiled_payloads_publish_semantic_histogram_for_partial_commit():
    from arrayscope.core.window_levels import LevelSourceRank
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.montage_levels import MontageLevelStats
    from arrayscope.window.montage_renderer import (
        _should_publish_montage_histogram_plot,
        _tiled_payloads_require_semantic_histogram_plot,
    )

    payloads = {
        0: DisplayTilePayload(
            tile_number=0,
            source_index=0,
            image=np.zeros((2, 2), dtype=np.float32),
            histogram_data=None,
            source_id=("native", 0),
        )
    }
    stats = MontageLevelStats(
        bounds=(0.0, 1.0),
        source_indices=frozenset({0}),
        expected_indices=frozenset({0, 1}),
        rank=LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
        sample=np.asarray([0.0, 1.0], dtype=np.float32),
    )

    assert _tiled_payloads_require_semantic_histogram_plot(payloads) is True
    assert (
        _should_publish_montage_histogram_plot(
            False,
            False,
            stats,
            requires_semantic_plot=_tiled_payloads_require_semantic_histogram_plot(payloads),
        )
        is True
    )
    assert _should_publish_montage_histogram_plot(False, False, stats) is False


def test_stage_wait_in_flight_is_not_actionable_timer_work():
    pytest.importorskip("pyqtgraph")
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.window.montage_renderer import MontageRenderMixin

    class _Cache:
        def get(self, key):
            return None

    class _Materializer:
        _in_flight = {"stage-key": object()}

    class _Evaluator:
        stage_cache = _Cache()
        stage_materializer = _Materializer()

    class _Window(MontageRenderMixin):
        operation_evaluator = _Evaluator()

    session = SimpleNamespace(
        stage_fan_in=StageFanInState(waiting_tiles={"stage-key": [SimpleNamespace(montage_index=3)]}),
    )

    assert not _Window()._stage_wait_has_actionable_work(session, release_missing=True)


def test_stage_wait_completed_stage_is_actionable_timer_work():
    pytest.importorskip("pyqtgraph")
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.window.montage_renderer import MontageRenderMixin

    class _Cache:
        def get(self, key):
            return "stage-value" if key == "stage-key" else None

    class _Materializer:
        _in_flight = {}

    class _Evaluator:
        stage_cache = _Cache()
        stage_materializer = _Materializer()

    class _Window(MontageRenderMixin):
        operation_evaluator = _Evaluator()

    session = SimpleNamespace(
        stage_fan_in=StageFanInState(waiting_tiles={"stage-key": [SimpleNamespace(montage_index=3)]}),
    )

    assert _Window()._stage_wait_has_actionable_work(session, release_missing=True)


def test_montage_session_key_excludes_transient_viewport_range():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((4, 4, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=3)
    first = MontageViewportPlan(2, tuple(range(6)), (100, 100), (4, 4), plan, ((0, 10), (0, 10)), True, True)
    second = MontageViewportPlan(2, tuple(range(6)), (100, 100), (4, 4), plan, ((10, 20), (0, 10)), True, True)

    assert montage_session_key("doc", state, first, None) == montage_session_key("doc", state, second, None)


def test_montage_session_key_changes_with_population_but_not_layout_reflow():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((4, 4, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan3 = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=3)
    plan2 = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=2)
    base = MontageViewportPlan(2, tuple(range(6)), (100, 100), (4, 4), plan3, None, True, True)
    changed_population = MontageViewportPlan(2, tuple(range(5)), (100, 100), (4, 4), plan3, None, True, True)
    changed_layout = MontageViewportPlan(2, tuple(range(6)), (100, 100), (4, 4), plan2, None, True, True)

    key = montage_session_key("doc", state, base, None)
    assert key != montage_session_key("doc", state, changed_population, None)
    assert key == montage_session_key("doc", state, changed_layout, None)


def test_direct_tiled_payload_retarget_allows_only_safe_layout_reflow():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.montage_renderer import _safe_tiled_payload_geometry_retarget

    state = ViewState.from_shape((4, 4, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan3 = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=3)
    plan2 = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=2)
    previous = DisplayGeometry(view_state=state, display_shape=plan3.display_shape, montage=plan3.geometry)
    reflow = DisplayGeometry(view_state=state, display_shape=plan2.display_shape, montage=plan2.geometry)
    changed_indices = make_montage_plan(state, axis=2, indices=tuple(range(5)), tile_shape=(4, 4), columns=3)
    incompatible = DisplayGeometry(
        view_state=state,
        display_shape=changed_indices.display_shape,
        montage=changed_indices.geometry,
    )

    assert _safe_tiled_payload_geometry_retarget(previous, reflow)
    assert not _safe_tiled_payload_geometry_retarget(previous, incompatible)
