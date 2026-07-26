"""Floor payloads for complex textures must carry the current shader mapping.

Field defect 2026-07-16 09:14 (framebuffer-probe proven): when a montage
index window changes, entering tiles present resident complex floor planes
from the persistent pyramid cache — but ``lod_preview_metadata`` (which
recorded each plane's shader mapping) is per-session state and is gone.
``ensure_floor_payloads`` then built COMPLEX_RG32F payloads with
``shader_mapping=None``, so the shader backend interpreted the complex plane
as magnitude through the cyclic LUT instead of phase color: every
zero-magnitude texel rendered the PAL-relaxed LUT[0] orange until exact
evaluation replaced the tile. The physical-divergence audit cannot catch this
by construction — the payload itself IS the desired state.

The mapping is a pure function of the current view state (channel, scale,
LUT), so the floor builder must mint it when the metadata is gone.
"""

from __future__ import annotations

import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.display.lod import LOD_POLICY_RESIDENT, select_lod_demand
from arrayscope.display.montage import MontagePlan, MontageTile, RenderedTile
from arrayscope.display.pyramid import LodPageCache, materialize_lod_page
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    TexturePlaneKind,
)
from arrayscope.window.frame_session import FrameSession

TILE = 64
ZOOMED_OUT_RANGE = ((0.0, 4.0 * 2 * TILE), (0.0, 4.0 * TILE))
VIEWPORT = (TILE, 2 * TILE)


def _tiles(count=2):
    return tuple(
        MontageTile(
            montage_index=index,
            source_index=index,
            row=0,
            col=index,
            x0=index * TILE,
            y0=0,
            width=TILE,
            height=TILE,
            view_state=None,
        )
        for index in range(count)
    )


def _session(*, dtype, view_state, shader_display: bool, count=2):
    tiles = _tiles(count)
    session = FrameSession(
        session_id=1,
        key=("test",),
        render_generation=1,
        level_key=None,
        level_expected_indices=tuple(range(count)),
        plan=MontagePlan(
            axis=0,
            tile_shape=(TILE, TILE),
            grid_shape=(1, count),
            columns=count,
            rows=1,
            gap=0,
            tiles=tiles,
        ),
        view_state=view_state,
        document=None,
        montage_axis=0,
        colormap_lut=None,
        viewport_shape=VIEWPORT,
        view_range=ZOOMED_OUT_RANGE,
        output_dtype=np.dtype(dtype),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        lod_policy_mode=LOD_POLICY_RESIDENT,
        lod_page_cache=LodPageCache(max_bytes=1 << 24),
    )
    session.shader_display = shader_display
    complex_input = bool(np.issubdtype(np.dtype(dtype), np.complexfloating))
    for index, tile in enumerate(tiles):
        image = (np.arange(TILE * TILE).reshape(TILE, TILE) + index).astype(dtype)
        session.rendered_tiles[index] = RenderedTile(
            tile=tile,
            image=image,
            histogram_data=np.abs(image).astype(np.float32),
            eval_ms=0.0,
            slab_shape=image.shape,
            slab_nbytes=image.nbytes,
            texture_kind=(TexturePlaneKind.COMPLEX_RG32F if complex_input else None),
        )
    return session


def _admit_metadata_free_floor(session, tile_number: int):
    """Admit a resident floor plane the way a PREVIOUS session left it:
    resident in the (persistent) pyramid cache, no per-session metadata."""

    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    assert int(demand.desired_level) > 0
    rendered = session.rendered_tiles[tile_number]
    key = session._lod_page_set_key_for(
        rendered,
        demand=demand,
        level=demand.desired_level,
    )
    source_origin_yx = (
        key.plans[0].valid_source_rect_yx[0],
        key.plans[0].valid_source_rect_yx[2],
    )
    pages = tuple(
        materialize_lod_page(
            rendered.image,
            source_origin_yx=source_origin_yx,
            plan=plan,
        )
        for plan in key.plans
    )
    del session.rendered_tiles[tile_number]
    session.dirty_payloads.clear()
    assert session.admit_preview_plane(tile_number, key, pages)
    assert session.preview_floor_metadata(key) is None
    return key


def test_metadata_free_complex_floor_mints_current_phase_mapping():
    view_state = ViewState.from_shape((TILE, 2 * TILE)).with_channel("complex")
    session = _session(dtype=np.complex64, view_state=view_state, shader_display=True)
    _admit_metadata_free_floor(session, 1)

    session._ensure_floor_payloads((1,))
    payload = session.display_tile_payloads[1]

    assert payload.texture_kind == TexturePlaneKind.COMPLEX_RG32F
    mapping = payload.shader_mapping
    assert mapping is not None
    assert mapping.display_mode == ShaderDisplayMode.PHASE_COLOR
    assert mapping.component == ShaderComponent.ABS
    assert mapping.histogram_source_policy == "mapped"


def test_metadata_free_scalar_floor_keeps_no_mapping():
    view_state = ViewState.from_shape((TILE, 2 * TILE))
    session = _session(dtype=np.float32, view_state=view_state, shader_display=False)
    _admit_metadata_free_floor(session, 1)

    session._ensure_floor_payloads((1,))
    payload = session.display_tile_payloads[1]

    assert payload.texture_kind == TexturePlaneKind.SCALAR_R32F
    assert payload.shader_mapping is None
