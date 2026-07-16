from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import VISPY_CAPABILITIES
from arrayscope.display.backends import surface_for_view
from arrayscope.display.backends.base import tiled_presentation_visible
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.model.commit import DisplayTiledPresentation
from arrayscope.display.model.frame import DisplayTilePayload, TileCommitReport, TilePresentationDelta, TilePresentationState
from arrayscope.display.shader_mapping import ShaderMapping
from arrayscope.display.viewport import ViewportPolicy


class _FakeSurface:
    def __init__(self):
        self.capabilities = VISPY_CAPABILITIES
        self.widget = object()
        self.calls = []
        self.physically_visible_tile_count = 0

    def present_tiled(self, presentation):
        self.calls.append(("tiled", None, presentation))
        return TileCommitReport(
            presented_tiles=frozenset(presentation.tile_state.active_payloads(presentation.tile_delta)),
            committed_upserts=frozenset(presentation.tile_delta.upserts),
            removed_tiles=frozenset(presentation.tile_delta.removals),
        )

    def invalidate_tiled_presentation(self, reason, *, hide_pixels=True):
        self.calls.append(("invalidate_tiled", str(reason), bool(hide_pixels)))

    def hide_tiled_presentation(self, reason):
        self.calls.append(("hide_tiled", str(reason), None))

    def reset_tiled_residency(self, reason):
        self.calls.append(("reset_tiled", str(reason), None))

    def set_profile_bounds(self, bounds):
        self.calls.append(("bounds", tuple(bounds), None))

    def apply_camera(self, image_shape, viewport_policy, *, image_origin=(0.0, 0.0), content_rect=None):
        self.calls.append(("camera", tuple(image_shape), viewport_policy, tuple(image_origin), content_rect))

    def map_scene_to_overlay(self, scene_pos):
        self.calls.append(("map_scene", scene_pos, None))
        return scene_pos

    def current_viewport_rect(self):
        return None

    def presentation_diagnostics(self):
        return {
            "backend": self.capabilities.name,
            "interaction_event_owner": self.interaction_event_owner(),
            "montage_display_mode": "vispy_tile_layer",
            "physically_visible_tile_count": self.physically_visible_tile_count,
        }

    def interaction_event_owner(self):
        return "fake"

    def sync_interaction_state(self, state):
        self.calls.append(("interaction", state, None))

    def reset_surface(self, reason):
        self.calls.append(("reset", str(reason), None))

    def teardown_surface(self):
        self.calls.append(("teardown", None, None))


def _geometry(*, montage=False):
    state = ViewState.from_shape((2, 3, 1)).with_image_axes(0, 1)
    geometry = DisplayGeometry(state, (2, 3))
    if montage:
        state = state.with_montage_axis(2, columns=1, indices=(0,))
        geometry = DisplayGeometry(
            state,
            (2, 3),
            montage=MontageGeometry(indices=(0,), tile_shape=(2, 3), columns=1, rows=1, gap=0),
            montage_tile_states=("loaded",),
        )
    return geometry


def _tiled_presentation():
    geometry = _geometry(montage=True)
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = DisplayTilePayload(
        tile_number=0,
        source_index=0,
        image=data,
        histogram_data=data,
        source_id=("tile", 0),
    )
    state = TilePresentationState({0: payload})
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts={0: payload},
        active_tiles=(0,),
        planned_tiles=(0,),
        near_tiles=(0,),
    )
    return DisplayTiledPresentation(
        geometry=geometry,
        levels=(0.0, 5.0),
        histogram_range=(0.0, 5.0),
        viewport_policy=ViewportPolicy.PRESERVE,
        tile_state=state,
        base_tile_state=TilePresentationState(),
        tile_delta=delta,
        tile_residency_budget_bytes=1024,
        shader_mapping=ShaderMapping(),
    )


def test_surface_resolver_uses_shell_owned_surface():
    surface = _FakeSurface()
    view = SimpleNamespace(surface=surface)

    assert surface_for_view(view) is surface


def test_surface_resolver_rejects_objects_without_surface_contract():
    with pytest.raises(TypeError, match="SimpleNamespace.*missing \\.surface"):
        surface_for_view(SimpleNamespace())


def test_surface_resolver_explains_nonconforming_surface():
    view = SimpleNamespace(surface=object())

    with pytest.raises(TypeError, match=r"\.surface is object, which does not implement ImageSurface"):
        surface_for_view(view)


def test_tiled_visibility_uses_physical_draw_count_not_selected_mode():
    surface = _FakeSurface()

    assert surface.presentation_diagnostics()["montage_display_mode"] == "vispy_tile_layer"
    assert tiled_presentation_visible(surface) is False

    surface.physically_visible_tile_count = 1
    assert tiled_presentation_visible(surface) is True


def test_surface_contract_commits_tiled_semantics():
    surface = _FakeSurface()
    tiled = _tiled_presentation()

    report = surface.present_tiled(tiled)
    surface.set_profile_bounds((0.0, 0.0, 2.0, 1.0))

    assert [call[0] for call in surface.calls] == ["tiled", "bounds"]
    assert surface.calls[0][2].tile_state == tiled.tile_state
    assert report.committed_upserts == frozenset({0})


def test_uniforms_only_stats_can_never_acknowledge_upserts():
    """Field defect 2026-07-05 (stale wrong-LOD): a levels-only fast path
    reported presented pool slots with committed_upserts=None, and the
    commit report fell back to acknowledging pending level-swap upserts by
    tile-number intersection — payload identities that were never uploaded.
    The GPU stayed on the old level until an unrelated pan.  Rule 1
    (ADR 0051): a path that cannot apply upserts must report an EMPTY
    committed set, so acknowledgement never invents acceptance."""

    from arrayscope.display.imageview2d import _tile_commit_report
    from arrayscope.display.model.tile_stats import TileLayerUpdateStats

    tiled = _tiled_presentation()
    # What set_presentation_uniforms reports: pool still presents tile 0
    # (with its OLD payload identity), no upserts applied.
    uniforms_stats = TileLayerUpdateStats(
        visible_items=1,
        presented_tiles=(0,),
        committed_upserts=(),
        level_updates=1,
    )
    report = _tile_commit_report(dict(tiled.tile_state.payloads), tiled.tile_delta, uniforms_stats)
    assert report.accepted_upserts(tiled.tile_delta) == set()
    # The pending upsert stays armed for the next real commit.
    state = tiled.base_tile_state.acknowledge_delta(tiled.tile_delta, report)
    assert 0 not in state.payloads


def test_pyqtgraph_surface_exposes_lifecycle_contract(qt_app):
    from arrayscope.display.backends.pyqtgraph.surface import PyQtGraphSurface

    view = PyQtGraphSurface()
    try:
        surface = surface_for_view(view)

        assert surface.widget is view
        assert surface.capabilities.name == "pyqtgraph"
        assert surface.interaction_event_owner() == "shared-controller"
        surface.apply_camera((2, 3), ViewportPolicy.PRESERVE)
        assert surface.current_viewport_rect() is None
        diagnostics = surface.presentation_diagnostics()
        assert diagnostics["backend"] == "pyqtgraph"
        assert diagnostics["interaction_event_owner"] == "shared-controller"
        assert diagnostics["physically_visible_tile_count"] == 0

        # Selecting the tiled route before any ImageItem commits must not be
        # mistaken for physical predecessor coverage.
        view._montage_display_mode = "tile_layer"
        assert tiled_presentation_visible(surface) is False

        surface.present_tiled(_tiled_presentation())
        assert surface.presentation_diagnostics()["physically_visible_tile_count"] == 1
        assert tiled_presentation_visible(surface) is True

        surface.reset_surface("test-context-loss")
        assert surface.presentation_diagnostics()["physically_visible_tile_count"] == 0
        assert surface.presentation_diagnostics()["last_reset_reason"] == "test-context-loss"
        surface.teardown_surface()
        surface.teardown_surface()
    finally:
        view.close()
