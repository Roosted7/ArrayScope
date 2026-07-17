"""Physical presentation truth for the VisPy GPU tile layer (P9).

Field defect 2026-07-15 (ring-buffer trace of a live montage index scroll):
every tile presentation was acknowledgement-only (retarget remap -> pool skip
path -> lifecycle presented with payload=None) while a page visual physically
held a stale per-quad ``a_mode``/``u_component_mode``.  Zero-magnitude
complex texels then rendered the PAL-relaxed LUT[0] orange instead of black.
The identity layer cannot see this class by construction (it deliberately
excludes levels/LUT/scale and nothing pins the mode vertex buffer), so the
layer itself must compare desired state against each active page visual's
PHYSICAL state before re-presenting, and repair on divergence.

These tests inject exactly that corruption into a committed layer and assert
the clean re-present and uniforms-only paths detect it, repair it, and charge
the repair to the acknowledged stats instead of reporting a no-op.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arrayscope.display.backends.vispy.tiles import (
    GpuDeviceLimits,
    GpuMontageLayer,
    TextureAtlasPool,
    _visual_shader_mapping_key,
)
from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    ShaderMapping,
)
from arrayscope.gpu import ChunkLod, DataChunkKey

from tests.display.vispy_test_utils import FakeGloo, FakeVisual, complex_payload


class PhysicalFakeVisual(FakeVisual):
    """FakeVisual that mirrors GpuWindowedTileVisual's PHYSICAL state.

    The base fake tracks call counts only; this subclass stores the same
    attributes the real visual exposes (``_shader_mapping_key``,
    ``_scale_mode``/``_symlog_constant``/``_component_mode``, ``_levels``,
    ``mode_data``) so injected corruption is observable by the layer's
    physical-divergence audit exactly as on a live canvas.
    """

    def __init__(self):
        super().__init__()
        self._shader_mapping_key = None
        self._scale_mode = 0.0
        self._symlog_constant = 0.0
        self._component_mode = 0.0
        self._levels = (0.0, 1.0)
        self.mode_data = np.zeros((0,), dtype=np.float32)

    def set_geometry(self, vertices, texcoords, modes):
        super().set_geometry(vertices, texcoords, modes)
        self.mode_data = np.asarray(modes, dtype=np.float32).reshape((-1,))

    def set_levels(self, levels):
        levels = tuple(float(value) for value in levels)
        if levels == self._levels:
            return False
        self._levels = levels
        self.levels.append(levels)
        return True

    def set_shader_mapping(self, mapping):
        self.mapping_calls += 1
        key = _visual_shader_mapping_key(mapping)
        if key == self._shader_mapping_key:
            return False
        self._shader_mapping_key = key
        self._scale_mode, self._symlog_constant, self._component_mode = (
            float(key[0]),
            float(key[1]),
            float(key[2]),
        )
        self.mappings.append((None if mapping is None else mapping.identity_key, mapping))
        return True


class PhysicalFakeSceneVisuals:
    @staticmethod
    def create_visual_node(_visual_type):
        return lambda parent=None: PhysicalFakeVisual()


class PhysicalFakeScene:
    visuals = PhysicalFakeSceneVisuals()


PHASE_MAPPING = ShaderMapping(
    component=ShaderComponent.ABS,
    display_mode=ShaderDisplayMode.PHASE_COLOR,
)


def _committed_phase_layer():
    """One committed complex phase_color presentation on physical fakes."""

    layer = GpuMontageLayer(
        scene=PhysicalFakeScene(),
        visuals=None,
        gloo=FakeGloo(),
        transforms=None,
        parent=None,
        limits=GpuDeviceLimits(max_texture_size=4),
    )
    montage = SimpleNamespace(
        indices=tuple(range(4)),
        tile_width=2,
        tile_height=2,
        columns=4,
        rows=1,
        gap=0,
    )
    geometry = SimpleNamespace(montage=montage, montage_tile_states=("loaded",) * 4)
    payloads = {index: complex_payload(index) for index in range(4)}
    delta = SimpleNamespace(
        upserts=payloads,
        removals=(),
        active_tiles=tuple(range(4)),
        planned_tiles=tuple(range(4)),
        near_tiles=(),
        near_tile_source_ids={index: value.source_id for index, value in payloads.items()},
        force_refresh=False,
    )
    first = layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 2.0),
        dirty_tiles=None,
        rgb_already_windowed=False,
        shader_mapping=PHASE_MAPPING,
        tile_delta=delta,
    )
    assert first.physical_repairs == 0
    assert first.presented_tiles == tuple(range(4))
    clean_delta = SimpleNamespace(
        upserts={},
        removals=(),
        active_tiles=tuple(range(4)),
        planned_tiles=tuple(range(4)),
        near_tiles=(),
        near_tile_source_ids={index: value.source_id for index, value in payloads.items()},
        force_refresh=False,
    )
    return layer, geometry, payloads, clean_delta


def _clean_represent(layer, geometry, payloads, clean_delta):
    return layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 2.0),
        dirty_tiles=(),
        rgb_already_windowed=False,
        shader_mapping=PHASE_MAPPING,
        tile_delta=clean_delta,
    )


def _visible_visual(layer):
    for visual in layer._visuals_by_page:
        if visual.visible:
            return visual
    raise AssertionError("no visible page visual")


def test_clean_represent_is_noop_when_physical_state_matches():
    layer, geometry, payloads, clean_delta = _committed_phase_layer()

    clean = _clean_represent(layer, geometry, payloads, clean_delta)

    assert clean.physical_repairs == 0
    assert clean.vertex_uploads == 0
    assert clean.shader_uniform_updates == 0
    assert layer.changed_page_indices() == ()


def test_physical_truth_reports_exact_draw_world_geometry():
    layer, _geometry, _payloads, _clean_delta = _committed_phase_layer()

    rows = layer.tile_truth_physical_rows()

    assert rows[0]["physical_draw_world_rects"] == ((0.0, 0.0, 2.0, 2.0),)
    assert rows[3]["physical_draw_world_bounds"] == (6.0, 0.0, 8.0, 2.0)
    assert rows[3]["physical_expected_world_rect"] == (6.0, 0.0, 8.0, 2.0)
    assert rows[3]["physical_draw_bounds_match_layout"] is True


def test_clean_represent_repairs_stale_component_uniform_behind_fresh_key():
    # The field-defect class: the visual's mapping KEY still looks fresh but
    # the derived uniform diverged (u_component_mode stale -> LUT(0) orange
    # for zero-magnitude complex texels).
    layer, geometry, payloads, clean_delta = _committed_phase_layer()
    visual = _visible_visual(layer)
    expected_component = float(visual._component_mode)
    visual._component_mode = 5.0

    repaired = _clean_represent(layer, geometry, payloads, clean_delta)

    assert repaired.physical_repairs == 1
    assert repaired.shader_uniform_updates == 1
    assert visual._component_mode == expected_component
    assert visual._shader_mapping_key == _visual_shader_mapping_key(PHASE_MAPPING)
    assert layer.changed_page_indices() != ()
    # Repaired state must present clean again (no repair oscillation).
    clean = _clean_represent(layer, geometry, payloads, clean_delta)
    assert clean.physical_repairs == 0
    assert clean.shader_uniform_updates == 0


def test_clean_represent_repairs_wrong_shader_mapping_key():
    layer, geometry, payloads, clean_delta = _committed_phase_layer()
    visual = _visible_visual(layer)
    visual._shader_mapping_key = ("stale", "mapping", "key")

    repaired = _clean_represent(layer, geometry, payloads, clean_delta)

    assert repaired.physical_repairs == 1
    assert repaired.shader_uniform_updates == 1
    assert visual._shader_mapping_key == _visual_shader_mapping_key(PHASE_MAPPING)
    assert visual.mappings[-1][1] is PHASE_MAPPING


def test_clean_represent_repairs_corrupted_mode_vertex_buffer():
    # Stale a_mode 3 (complex-through-LUT, no magnitude modulation) on a
    # phase_color quad is the orange-background draw; nothing but the
    # physical audit pins this buffer.
    layer, geometry, payloads, clean_delta = _committed_phase_layer()
    visual = _visible_visual(layer)
    assert np.all(visual.mode_data == 4.0)
    visual.mode_data = visual.mode_data.copy()
    visual.mode_data[0:6] = 3.0
    geometry_calls_before = visual.geometry_calls

    repaired = _clean_represent(layer, geometry, payloads, clean_delta)

    assert repaired.physical_repairs == 1
    assert repaired.vertex_uploads == 1
    assert visual.geometry_calls == geometry_calls_before + 1
    assert np.all(visual.mode_data == 4.0)
    clean = _clean_represent(layer, geometry, payloads, clean_delta)
    assert clean.physical_repairs == 0
    assert clean.vertex_uploads == 0


def test_clean_represent_repairs_stale_levels_uniform():
    layer, geometry, payloads, clean_delta = _committed_phase_layer()
    visual = _visible_visual(layer)
    visual._levels = (5.0, 9.0)

    repaired = _clean_represent(layer, geometry, payloads, clean_delta)

    assert repaired.physical_repairs == 1
    assert repaired.level_updates == 1
    assert visual._levels == (0.0, 2.0)


def test_divergent_layer_never_acknowledges_a_physical_noop():
    # ADR 0051 rule 1 extension: the acknowledged stats for a divergent
    # re-present must carry the repair work, not read as items_skipped-only.
    layer, geometry, payloads, clean_delta = _committed_phase_layer()
    visual = _visible_visual(layer)
    visual._component_mode = 5.0
    visual.mode_data = visual.mode_data.copy()
    visual.mode_data[:] = 3.0

    repaired = _clean_represent(layer, geometry, payloads, clean_delta)

    assert repaired.physical_repairs == 2
    assert repaired.shader_uniform_updates >= 1
    assert repaired.vertex_uploads == 1
    # The presentation itself is still acknowledged (tiles stay presented) —
    # only the "no physical work happened" claim is withdrawn.
    assert repaired.presented_tiles == tuple(range(4))
    assert repaired.committed_upserts == ()


def test_uniforms_only_path_repairs_divergent_visual_state():
    # A levels/mapping no-op through set_presentation_uniforms must also
    # audit physical state: the level gesture path re-presents without a
    # payload commit.
    layer, geometry, payloads, clean_delta = _committed_phase_layer()
    visual = _visible_visual(layer)
    visual._component_mode = 5.0
    visual.mode_data = visual.mode_data.copy()
    visual.mode_data[6:12] = 3.0

    stats = layer.set_presentation_uniforms(levels=(0.0, 2.0))

    assert stats.physical_repairs == 2
    assert stats.shader_uniform_updates >= 1
    assert stats.vertex_uploads == 1
    assert visual._component_mode == 2.0  # ShaderComponent.ABS
    assert np.all(visual.mode_data == 4.0)
    follow_up = layer.set_presentation_uniforms(levels=(0.0, 2.0))
    assert follow_up.physical_repairs == 0
    assert follow_up.shader_uniform_updates == 0


def test_full_update_repairs_stale_uniform_the_page_sync_cannot_see():
    # A real upsert commit walks the touched pages through
    # visual.set_shader_mapping, but that setter no-ops when the visual's
    # mapping KEY still looks fresh — a corrupted derived uniform slips
    # through.  The end-of-update physical audit must catch it.
    layer, geometry, payloads, clean_delta = _committed_phase_layer()
    visual = _visible_visual(layer)
    visual._component_mode = 5.0

    replacement = complex_payload(3)
    payloads = dict(payloads)
    payloads[3] = replacement
    update_delta = SimpleNamespace(
        upserts={3: replacement},
        removals=(),
        active_tiles=tuple(range(4)),
        planned_tiles=tuple(range(4)),
        near_tiles=(),
        near_tile_source_ids={index: value.source_id for index, value in payloads.items()},
        force_refresh=False,
    )

    stats = layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 2.0),
        dirty_tiles=(3,),
        rgb_already_windowed=False,
        shader_mapping=PHASE_MAPPING,
        tile_delta=update_delta,
    )

    assert stats.physical_repairs == 1
    assert visual._component_mode == 2.0


def test_coarse_page_fallback_reports_actual_physical_identity_and_quality():
    """G5 physical truth names sampled coarse data, never desired fine data."""

    coarse = DataChunkKey(
        document_generation=("doc", 1),
        operation_key=("op", "identity"),
        lod=ChunkLod(reduction=(2, 2), reducer="mean"),
        chunk_origin=(0, 0),
        chunk_shape=(8, 8),
        dtype="float32",
    )
    target = DataChunkKey(
        document_generation=("doc", 1),
        operation_key=("op", "identity"),
        lod=ChunkLod(reduction=(0, 0), reducer="mean"),
        chunk_origin=(2, 2),
        chunk_shape=(2, 2),
        dtype="float32",
    )
    payload = DisplayTilePayload(
        tile_number=0,
        source_index=0,
        image=np.full((2, 2), 4.0, dtype=np.float32),
        histogram_data=None,
        source_id=coarse,
        quality="exact",
    )
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=2)
    _uvs, cold = pool.update_payloads(
        {0: payload},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=1,
    )
    assert cold.texture_uploads == 1

    resolution = pool.resolve_page_targets({7: target})[7]
    row = pool.tile_truth_physical_rows()[7]

    assert resolution.actual_key == coarse
    assert pool.presented_identities()[7] == coarse
    assert pool.presented_identities()[7] != target
    assert row["physical_acknowledged_identity"] == coarse
    assert row["physical_page_target_key"] == target
    assert row["physical_page_actual_key"] == coarse
    assert row["physical_page_lod"] == coarse.lod
    assert row["physical_page_quality"] == "fallback"
    assert row["physical_page_binding_generation"] == resolution.binding_generation


def test_identity_rejected_upserts_are_reported_not_silent():
    """Session-148 gate (2026-07-16): typed-target rejection must be loud.

    A delta upsert whose payload identity cannot satisfy that tile's target
    identity is excluded from presentation.  That exclusion used to be
    completely silent (no stat, no skip count), so a presenter re-emitting
    the same dead payload looped forever while the tile stayed empty on
    screen.  The commit stats must name the rejected tiles so diagnostics
    and traces expose the loop on the first commit.
    """

    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind

    def identity(semantic_generation):
        return TileIdentity(
            document_generation=("doc", 0),
            operation_key=("ops",),
            source_index=3,
            image_axes=(1, 0),
            axis_flips=(False, False),
            channel="real",
            complex_mapping=("scalar", "real", "mapped"),
            texture_kind=TexturePlaneKind.SCALAR_R32F,
            semantic_generation=semantic_generation,
            lod=TileLodIdentity(level=0, factor=1),
        )

    texture = np.zeros((4, 4), dtype=np.float32)
    payload = DisplayTilePayload(
        3,
        3,
        texture,
        None,
        ("tile", 3),
        texture_data=texture,
        lod=LodInfo(level=0, factor=1, source_shape=(4, 4), texture_shape=(4, 4)),
        quality="exact",
        tile_identity=identity(("stale",)),
    )
    delta = SimpleNamespace(
        upserts={3: payload},
        active_tiles=(3,),
        target_identities={3: identity(("current",))},
        removals=(),
        near_tile_source_ids={},
    )
    pool = TextureAtlasPool(FakeGloo())
    _uvs, stats = pool.update_payloads(
        {3: payload},
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        tile_delta=delta,
    )

    assert stats.committed_upserts == ()
    assert stats.texture_uploads == 0
    assert stats.identity_rejected_items == 1
    assert stats.identity_rejected_tiles == (3,)

    # The matching identity presents normally and reports zero rejections.
    delta.target_identities = {3: identity(("stale",))}
    _uvs, stats = pool.update_payloads(
        {3: payload},
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        tile_delta=delta,
    )
    assert stats.committed_upserts == (3,)
    assert stats.identity_rejected_items == 0
    assert stats.identity_rejected_tiles == ()
