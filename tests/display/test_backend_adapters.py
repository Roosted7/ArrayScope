from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import VISPY_CAPABILITIES
from arrayscope.display.backends import (
    PyQtGraphBackendAdapter,
    RasterCommitMode,
    VisPyBackendAdapter,
    backend_adapter_for_view,
)
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
from arrayscope.display.model.commit import DisplayRasterPresentation, DisplayTiledPresentation


class _FakeView:
    def __init__(self, *, vispy=False):
        if vispy:
            self.rendering_capabilities = VISPY_CAPABILITIES
        self.image = np.zeros((2, 3), dtype=np.float32)
        self.calls = []

    def setImagePresentation(self, data, **kwargs):
        self.calls.append(("full", data, kwargs))

    def updateImagePresentationFast(self, data, **kwargs):
        self.calls.append(("fast", data, kwargs))

    def setMontageTileLayerPresentation(self, data, **kwargs):
        self.calls.append(("legacy_tiles", data, kwargs))

    def setTiledMontagePresentation(self, **kwargs):
        self.calls.append(("tiles", None, kwargs))

    def setProfileMarkerBoundsRect(self, bounds):
        self.calls.append(("bounds", bounds, {}))


def _raster_presentation(*, montage=False):
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
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    return DisplayRasterPresentation(
        data=data,
        histogram_data=data,
        histogram_plot_data=None,
        geometry=geometry,
        levels=(0.0, 5.0),
        histogram_range=(0.0, 5.0),
        viewport_policy=ViewportPolicy.PRESERVE,
    )


def _tiled_presentation():
    raster = _raster_presentation(montage=True)
    data = raster.data
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
        geometry=raster.geometry,
        levels=raster.levels,
        histogram_range=raster.histogram_range,
        viewport_policy=raster.viewport_policy,
        tile_state=state,
        tile_delta=delta,
        tile_residency_budget_bytes=1024,
    )


def test_factory_selects_and_caches_builtin_adapters_by_capability():
    pyqt_view = _FakeView()
    vispy_view = _FakeView(vispy=True)

    pyqt_adapter = backend_adapter_for_view(pyqt_view)
    vispy_adapter = backend_adapter_for_view(vispy_view)

    assert isinstance(pyqt_adapter, PyQtGraphBackendAdapter)
    assert isinstance(vispy_adapter, VisPyBackendAdapter)
    assert backend_adapter_for_view(pyqt_view) is pyqt_adapter
    assert backend_adapter_for_view(vispy_view) is vispy_adapter


def test_adapters_translate_shared_raster_and_tiled_semantics():
    for view in (_FakeView(), _FakeView(vispy=True)):
        adapter = backend_adapter_for_view(view)
        raster = _raster_presentation(montage=True)
        tiled = _tiled_presentation()

        adapter.present_raster(raster, mode=RasterCommitMode.FULL)
        adapter.present_raster(raster, mode=RasterCommitMode.FAST)
        adapter.present_raster(raster, mode=RasterCommitMode.TILE_LAYER)
        adapter.present_tiled(tiled)
        adapter.set_profile_bounds((0.0, 0.0, 2.0, 1.0))

        assert [call[0] for call in view.calls] == ["full", "fast", "legacy_tiles", "tiles", "bounds"]
        assert view.calls[0][2]["levels"] == raster.levels
        assert view.calls[0][2]["image_origin"] == (0.0, 0.0)
        assert view.calls[3][2]["tile_state"] == tiled.tile_state
        assert view.calls[3][2]["tile_delta"] == tiled.tile_delta


def test_factory_accepts_custom_semantic_backend_without_rewrapping():
    backend = SimpleNamespace(
        view=object(),
        capabilities=SimpleNamespace(name="custom"),
        current_raster_shape=lambda: None,
        present_raster=lambda *_args, **_kwargs: None,
        present_tiled=lambda *_args, **_kwargs: None,
        set_profile_bounds=lambda *_args, **_kwargs: None,
    )
    view = SimpleNamespace(render_backend_adapter=backend)

    assert backend_adapter_for_view(view) is backend
