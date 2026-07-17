"""ADR 0056 G5: source-anchor stamping on floor payloads (c1674beb pin).

``render.lod.ensure_floor_payloads`` stamps
``session._payload_source_anchor(plan.tile_shape)`` on EXACT reduced floor
payloads so the VisPy pool can take the chunked-residency path.  Two laws
must hold or anchors/chunk keys corrupt (field-report hypothesis H3,
2026-07-15):

1. Montage sessions never get an anchor — montage tiles are classic.
2. Non-montage sessions get an anchor whose rect is the anchored window
   start plus the NATIVE plan tile extent (``plan.tile_shape`` — the same
   extent the payload's ``LodInfo.source_shape`` carries), never the
   reduced texture extent.
"""

from __future__ import annotations

import numpy as np

from arrayscope.display.lod import LOD_POLICY_RESIDENT, select_lod_demand
from arrayscope.display.montage import MontagePlan, MontageTile, RenderedTile
from arrayscope.display.pyramid import LodPageCache, materialize_lod_page
from arrayscope.display.source_anchoring import SourceAnchoring
from arrayscope.window.frame_session import FrameSession

TILE = 64
ZOOMED_OUT_RANGE = ((0.0, 4.0 * 2 * TILE), (0.0, 4.0 * TILE))
VIEWPORT = (TILE, 2 * TILE)

ANCHORED_STARTS = (5, 12)
CONTENT_KEY = ("doc-rev-0", "view-no-window")


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


def _session(*, montage_axis, source_anchoring=None, count=2):
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
        view_state=None,
        document=None,
        montage_axis=montage_axis,
        colormap_lut=None,
        viewport_shape=VIEWPORT,
        view_range=ZOOMED_OUT_RANGE,
        output_dtype=np.dtype("float32"),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        lod_policy_mode=LOD_POLICY_RESIDENT,
        lod_page_cache=LodPageCache(max_bytes=1 << 24),
        source_anchoring=source_anchoring,
    )
    for index, tile in enumerate(tiles):
        image = np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE) + index
        session.rendered_tiles[index] = RenderedTile(
            tile=tile,
            image=image,
            histogram_data=image,
            eval_ms=0.0,
            slab_shape=image.shape,
            slab_nbytes=image.nbytes,
        )
        session.dirty_payloads[index] = None
    return session


def _floor_payload_for_exact_reduced_level(session):
    """Admit an EXACT reduced level for tile 1 and build its floor payload."""

    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    assert int(demand.desired_level) > 0
    rendered = session.rendered_tiles[1]
    key = session._lod_page_set_key_for(
        rendered, demand=demand, level=demand.desired_level
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
    del session.rendered_tiles[1]
    session.dirty_payloads.clear()
    assert session.admit_preview_plane(1, key, pages, quality="exact")

    session._ensure_floor_payloads((1,))
    payload = session.display_tile_payloads[1]
    assert payload.quality == "exact"
    assert payload.lod.level == int(demand.desired_level)
    # The LOD's native source extent is the plan tile shape, always.
    assert tuple(payload.lod.source_shape) == (TILE, TILE)
    return payload


def test_montage_floor_payload_never_carries_an_anchor():
    session = _session(
        montage_axis=0,
        source_anchoring=SourceAnchoring(
            anchored_starts=ANCHORED_STARTS, content_key=CONTENT_KEY
        ),
    )
    payload = _floor_payload_for_exact_reduced_level(session)
    assert payload.source_anchor is None


def test_non_montage_floor_payload_anchor_rect_is_native_plan_extent():
    session = _session(
        montage_axis=None,
        source_anchoring=SourceAnchoring(
            anchored_starts=ANCHORED_STARTS, content_key=CONTENT_KEY
        ),
    )
    payload = _floor_payload_for_exact_reduced_level(session)
    anchor = payload.source_anchor
    assert anchor is not None
    assert anchor.content_key == CONTENT_KEY
    y0, x0 = ANCHORED_STARTS
    # NATIVE extent (plan.tile_shape), not the reduced texture extent.
    assert tuple(int(v) for v in anchor.source_rect) == (
        y0,
        y0 + TILE,
        x0,
        x0 + TILE,
    )


def test_non_montage_floor_payload_without_anchoring_stays_classic():
    session = _session(montage_axis=None, source_anchoring=None)
    payload = _floor_payload_for_exact_reduced_level(session)
    assert payload.source_anchor is None


def test_preview_quality_floor_payload_never_carries_an_anchor():
    """Degraded planes do not honor the anchor's pure-function promise."""

    session = _session(
        montage_axis=None,
        source_anchoring=SourceAnchoring(
            anchored_starts=ANCHORED_STARTS, content_key=CONTENT_KEY
        ),
    )
    demand = select_lod_demand(ZOOMED_OUT_RANGE, VIEWPORT, (TILE, TILE))
    rendered = session.rendered_tiles[1]
    key = session._lod_page_set_key_for(
        rendered, demand=demand, level=demand.desired_level
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
    del session.rendered_tiles[1]
    session.dirty_payloads.clear()
    assert session.admit_preview_plane(1, key, pages, quality="preview")

    session._ensure_floor_payloads((1,))
    payload = session.display_tile_payloads[1]
    assert payload.quality == "preview"
    assert payload.source_anchor is None
