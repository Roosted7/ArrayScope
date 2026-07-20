"""Shared fakes and payload builders for the VisPy tile/atlas test suites.

Single home for the gloo/scene/visual stand-ins and payload factories that
tests/display/test_vispy_tiled_renderer.py, test_vispy_chunked_residency.py,
test_vispy_physical_presentation.py, test_window_shift_gate.py, and
test_source_anchoring.py previously re-implemented (ADR 0055 G-program).
"""

from __future__ import annotations

import numpy as np

from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    ShaderMapping,
    TexturePlaneKind,
)


class FakeTexture2D:
    """``gloo.Texture2D`` stand-in that records every ``set_data`` call.

    ``updates`` keeps the raw call triples ``(data, offset, copy)``;
    ``uploads`` normalizes the same calls to ``((y, x) offset, copied plane)``
    for tests that index uploaded content by texture offset. Both lists share
    the same copied array objects.
    """

    def __init__(self, data=None, *, shape=None, **kwargs):
        if data is not None and shape is not None:
            raise ValueError("data and shape are mutually exclusive")
        self.initial_data = data
        self.shape = tuple(shape) if shape is not None else tuple(np.shape(data))
        self.kwargs = dict(kwargs)
        self.updates: list[tuple[np.ndarray, tuple[int, int] | None, bool]] = []
        self.uploads: list[tuple[tuple[int, int], np.ndarray]] = []

    def set_data(self, data, *, offset=None, copy=True):
        stored = np.array(data, copy=True)
        self.updates.append((stored, offset, bool(copy)))
        self.uploads.append((tuple(int(value) for value in (offset or (0, 0))), stored))


class FakeGloo:
    Texture2D = FakeTexture2D


class FakeVisual:
    """Call-count-level stand-in for ``GpuWindowedTileVisual``."""

    def __init__(self):
        self.visible = False
        self.levels = []
        self.geometry_calls = 0
        self.vertices = None
        self.texcoords = None
        self.modes = None
        self.mapping_calls = 0
        self.mappings = []
        self.textures = None
        self.texture_calls = 0
        self.mipmap_page = None
        self.mipmap_calls = 0
        self.update_calls = 0

    def set_levels(self, levels):
        levels = tuple(float(value) for value in levels)
        changed = not self.levels or self.levels[-1] != levels
        if changed:
            self.levels.append(levels)
        return changed

    def set_geometry(self, vertices, texcoords, modes):
        self.geometry_calls += 1
        self.vertices = np.asarray(vertices)
        self.texcoords = np.asarray(texcoords)
        self.modes = np.asarray(modes)

    def set_textures(self, scalar, color):
        self.texture_calls += 1
        changed = self.textures != (scalar, color)
        self.textures = (scalar, color)
        return changed

    def set_mipmap_page(self, page):
        self.mipmap_calls += 1
        self.mipmap_page = page

    def set_shader_mapping(self, mapping):
        self.mapping_calls += 1
        key = None if mapping is None else mapping.identity_key
        previous = None if not self.mappings else self.mappings[-1][0]
        changed = not self.mappings or previous != key
        if changed:
            self.mappings.append((key, mapping))
        return changed

    def update(self):
        self.update_calls += 1


class FakeSceneVisuals:
    @staticmethod
    def create_visual_node(_visual_type):
        return lambda parent=None: FakeVisual()


class FakeScene:
    visuals = FakeSceneVisuals()


class FakeDisplayImage:
    """Minimal display-image record for ``EagerDisplayRegionSource``."""

    def __init__(self, data):
        self.data = np.asarray(data)
        self.histogram_data = None


def payload(tile_number: int, value: float, *, source_id=None) -> DisplayTilePayload:
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=np.full((2, 2), value, dtype=np.float32),
        histogram_data=None,
        source_id=("tile", tile_number, value) if source_id is None else source_id,
    )


def color_payload(tile_number: int, value: int, *, window_scalar=True) -> DisplayTilePayload:
    image = np.full((2, 2, 3), int(value), dtype=np.uint8)
    scalar = np.full((2, 2), float(value), dtype=np.float32) if window_scalar else None
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=image,
        histogram_data=scalar,
        source_id=("color", tile_number, value, bool(window_scalar)),
    )


def complex_payload(tile_number: int) -> DisplayTilePayload:
    data = np.array([[1 + 0j, 1j], [-1 + 0j, -1j]], dtype=np.complex64)
    histogram = np.abs(data).astype(np.float32)
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=data,
        histogram_data=histogram,
        source_id=("complex", tile_number),
        texture_data=data,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=data,
        semantic_histogram_data=histogram,
        shader_mapping=ShaderMapping(
            component=ShaderComponent.ABS,
            display_mode=ShaderDisplayMode.PHASE_COLOR,
        ),
    )
