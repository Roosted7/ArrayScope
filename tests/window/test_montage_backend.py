from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.app.settings_state import MontageDisplayBackendChoice
from arrayscope.display.backend_contract import ImageViewBackendCapabilities
from arrayscope.window.montage_backend import choose_montage_backend
from arrayscope.window.montage_renderer import _montage_viewport_update_delay_ms


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


def test_persistent_tile_residency_uses_frame_cadence_viewport_updates():
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

    assert _montage_viewport_update_delay_ms(window) == 16
    assert _montage_viewport_update_delay_ms(fallback) == 120

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
