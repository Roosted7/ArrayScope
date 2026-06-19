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
    assert "full-surface uploads" in decision.reason


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


def test_auto_preserves_vispy_tile_layer_mode():
    data = np.zeros((64, 64), dtype=np.float32)

    decision = choose_montage_backend(_geometry(), data, current_mode="vispy_tile_layer")

    assert decision.backend == "tile_layer"
    assert "preserving" in decision.reason


def test_persistent_tile_residency_defers_tile_discovery_behind_camera_updates():
    capabilities = ImageViewBackendCapabilities(
        name="vispy",
        direct_montage_tile_payloads=True,
        persistent_tile_residency=True,
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
    from arrayscope.window.montage_renderer import MontageRenderMixin

    class _Window(MontageRenderMixin):
        pass

    class _Session:
        def __init__(self):
            self.active_stage_requests = {"stage-key"}
            self.attached_stage_requests = {"stage-key"}
            self.stage_waiting_tiles = {"stage-key": [SimpleNamespace(montage_index=3)]}
            self.tile_stage_keys = {3: "stage-key"}
            self.rendered_tiles = {}
            self.skipped_tiles = set()
            self.pending_tiles = []
            self.loading_tiles = set()

        def mark_loading(self, tile):
            self.loading_tiles.add(int(tile.montage_index))

    session = _Session()

    _Window()._release_stage_waiting_tiles_to_direct(session, "stage-key")

    assert session.active_stage_requests == set()
    assert session.attached_stage_requests == set()
    assert session.stage_waiting_tiles == {}
    assert session.tile_stage_keys == {}
    assert [tile.montage_index for tile in session.pending_tiles] == [3]
    assert session.loading_tiles == {3}


def test_montage_session_key_excludes_transient_viewport_range():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan

    state = ViewState.from_shape((4, 4, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(6)), tile_shape=(4, 4), columns=3)
    first = MontageViewportPlan(2, tuple(range(6)), (100, 100), (4, 4), plan, ((0, 10), (0, 10)), True, True)
    second = MontageViewportPlan(2, tuple(range(6)), (100, 100), (4, 4), plan, ((10, 20), (0, 10)), True, True)

    assert montage_session_key("doc", state, first, None) == montage_session_key("doc", state, second, None)


def test_montage_session_key_changes_with_population_or_layout():
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
    assert key != montage_session_key("doc", state, changed_layout, None)
