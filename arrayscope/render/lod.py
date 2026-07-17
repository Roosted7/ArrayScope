"""Render-owned ADR 0050 montage LOD helpers.

Qt-free by construction.  The ladder owns quality progression and the
pipeline owns scheduling; this module keeps pure identity, policy, cache
lookup, and payload-construction helpers together.

Contract (the defect class behind every stall/loop this code has had is
optimistic bookkeeping — these rules are load-bearing):

* **Singleflight claims are balanced on every path.**  Every page claim is
  matched by checked admission or owner release — on error, cancellation,
  stale-before-start, blocked admission, and session replacement.  A leaked
  claim silently blocks that page forever.
* **Supersession cancels work, never results.**  A rung that admitted its
  level reports through the pipeline completion path so presentation can
  converge without waiting for an unrelated event.
* **Bookkeeping is acknowledge-driven.**  Nothing here marks a tile clean or
  a request satisfied because work was *submitted*; only completions and
  acknowledged commits advance state.  Dirty marks for non-active tiles are
  emitted once and parked by the session on declined acknowledgement.
* **The floor is presentation-only.**  Floor payloads carry
  ``quality="preview"``: they draw pixels but never satisfy semantic reads,
  histograms, level convergence, or LOD convergence.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from arrayscope.display.lod import (
    LOD_POLICY_NATIVE_ONLY,
    LOD_REASON_NATIVE_POLICY,
    LOD_POLICY_RESIDENT,
    LodInfo,
    choose_resident_level,
    factor_xy_for_level,
    native_lod_policy,
    resident_presentation_rank,
    resident_lod_policy,
    select_lod_demand,
)
from arrayscope.display.montage import RenderedTile
from arrayscope.display.model.frame import DisplayTilePayload, PageBackedPresentation
from arrayscope.display.pyramid import (
    LodPageCache,
    LodPagePlan,
    MaterializedLodPage,
    ResolvedLodPageSet,
    materialize_lod_page,
    plan_source_grid_pages,
)
from arrayscope.display.shader_mapping import ShaderComponent, ShaderDisplayMode, TexturePlaneKind
from arrayscope.gpu import DataChunkKey
from arrayscope.gpu.keys import (
    COMPLEX_RG32F,
    REDUCER_MEAN,
    REDUCER_MEAN_ABS,
    REDUCER_NATIVE,
    REDUCER_PHASE_VECTOR,
    RGB8,
    SCALAR_R32F,
)
from arrayscope.presentation import ClaimOwner, LevelPhase


PREVIEW_FLOOR_MIN_LEVEL = 4


def plan_lod_page_targets(
    *,
    content_key: object,
    source_rect: tuple[int, int, int, int],
    reduction: tuple[int, int],
    stored_page_shape: tuple[int, int],
    dtype: str,
    representation: str,
    reducer: str,
) -> tuple[DataChunkKey, ...]:
    """Decompose one desired LOD window into canonical source-grid pages.

    Geometry is native-source ``(y, x)`` space. ``reduction`` follows that
    same axis order, while ``stored_page_shape`` is the uniform physical
    sample extent of a page. Boundary pages carry their clipped footprint;
    aligned interiors therefore share identity across shifted windows.
    """

    return tuple(
        plan.key
        for plan in plan_source_grid_pages(
            content_key=content_key,
            valid_source_rect_yx=source_rect,
            reduction_yx=reduction,
            stored_page_shape=stored_page_shape,
            dtype=dtype,
            representation=representation,
            reducer=reducer,
        )
    )


def texture_requires_display_histogram(texture, texture_kind) -> bool:
    kind = None if texture_kind is None else str(getattr(texture_kind, "value", texture_kind))
    if kind in {TexturePlaneKind.RGB8.value, TexturePlaneKind.COMPLEX_RG32F.value}:
        return True
    shape = tuple(getattr(texture, "shape", ()) or ())
    return len(shape) == 3 and int(shape[-1]) in (3, 4)


def display_histogram_matches_texture(histogram, texture) -> bool:
    if histogram is None:
        return False
    return tuple(np.shape(histogram)[:2]) == tuple(np.shape(texture)[:2])


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def viewport_identity(view_range, viewport_shape: tuple[int, int]) -> tuple[object, ...]:
    shape = tuple(max(1, int(value)) for value in tuple(viewport_shape or (1, 1))[:2])
    if view_range is None:
        return (shape, None)
    try:
        return (
            shape,
            (
                (float(view_range[0][0]), float(view_range[0][1])),
                (float(view_range[1][0]), float(view_range[1][1])),
            ),
        )
    except Exception:
        return (shape, repr(view_range))


def texture_source_for_rendered(
    rendered: RenderedTile,
    *,
    shader_display: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, TexturePlaneKind | None]:
    """Return the texture source plane, histogram, and plane kind of a tile."""

    texture_kind = getattr(rendered, "texture_kind", None)
    if texture_kind is not None and not isinstance(texture_kind, TexturePlaneKind):
        texture_kind = TexturePlaneKind(getattr(texture_kind, "value", texture_kind))
    if (
        bool(shader_display)
        and texture_kind == TexturePlaneKind.COMPLEX_RG32F
        and getattr(rendered, "semantic_data", None) is not None
    ):
        source = np.asarray(rendered.semantic_data)
    else:
        source = np.asarray(rendered.image)
        if not bool(shader_display):
            if source.ndim >= 3 and source.shape[-1] in (3, 4):
                texture_kind = TexturePlaneKind.RGB8
            elif np.iscomplexobj(source) or (
                source.ndim >= 3 and source.shape[-1] == 2
            ):
                texture_kind = TexturePlaneKind.COMPLEX_RG32F
            else:
                texture_kind = TexturePlaneKind.SCALAR_R32F
    histogram = None if rendered.histogram_data is None else np.asarray(rendered.histogram_data)
    return source, histogram, texture_kind


def canonical_value_source_for_rendered(
    rendered: RenderedTile,
    *,
    shader_display: bool = True,
) -> np.ndarray:
    """Return the native semantic values consumed by canonical reducers.

    CPU backends may hold an RGB or scalar display plane after channel
    conversion. ``lod_source_data`` preserves the raw complex plane so mean,
    mean-absolute, and phase-vector families remain backend-independent.
    """

    lod_source = getattr(rendered, "lod_source_data", None)
    if lod_source is not None:
        return np.asarray(lod_source)
    semantic = getattr(rendered, "semantic_data", None)
    if semantic is not None and np.iscomplexobj(semantic):
        return np.asarray(semantic)
    source, _histogram, _kind = texture_source_for_rendered(
        rendered, shader_display=shader_display
    )
    return np.asarray(source)


def component_for_rendered(rendered: RenderedTile, *, shader_display: bool = True) -> str:
    """Pyramid component tag of a rendered tile, without touching its arrays.

    ``texture_source_for_rendered`` normalizes the kind and picks a source
    plane, but the component tag depends on the kind alone — key derivation
    is on hot per-commit paths (tens of thousands of calls per scrub burst)
    and must not pay ``np.asarray`` per call.
    """

    if not bool(shader_display):
        image = getattr(rendered, "image", None)
        if image is not None:
            shape = tuple(getattr(image, "shape", ()) or ())
            if len(shape) == 3 and int(shape[-1]) in (3, 4):
                return str(TexturePlaneKind.RGB8.value)
    texture_kind = getattr(rendered, "texture_kind", None)
    if texture_kind is None:
        return "scalar"
    if TexturePlaneKind(getattr(texture_kind, "value", texture_kind)) == TexturePlaneKind.SCALAR_R32F:
        return "scalar"
    return str(getattr(texture_kind, "value", texture_kind))


def page_set_key_for_rendered(
    rendered: RenderedTile, *, demand, level: int, semantic_source_id, shader_display: bool = True
) -> LodPageSetKey:
    """Lifecycle identity of one canonical logical-page target set.

    ``semantic_source_id`` is the session-owned semantic identity of the tile
    content (session key + source index).  Object identity of the source
    array must never appear in pyramid keys: rendered tiles are rebuilt
    freely across commits and sessions, and cached levels must stay
    addressable without a live ``RenderedTile`` so presentation can floor on
    resident levels for tiles that have not been rendered yet.
    """

    source = canonical_value_source_for_rendered(
        rendered, shader_display=shader_display
    )
    reducer, dtype, representation = _reducer_format_for_rendered(rendered, source)
    factor_x, factor_y = factor_xy_for_level(demand, int(level))
    reduction_yx = (int(factor_y).bit_length() - 1, int(factor_x).bit_length() - 1)
    reducer, dtype, representation = _page_route_format(
        source,
        reduction_yx=reduction_yx,
        reduced_format=(reducer, dtype, representation),
    )
    height, width = (int(value) for value in np.shape(source)[:2])
    content_key = ("src-anchored", semantic_source_id, ("display-plane",))
    plans = plan_source_grid_pages(
        content_key=content_key,
        valid_source_rect_yx=(0, height, 0, width),
        reduction_yx=reduction_yx,
        stored_page_shape=(256, 256),
        dtype=dtype,
        representation=representation,
        reducer=reducer,
    )
    return LodPageSetKey(
        source_id=semantic_source_id,
        tile_id=int(rendered.tile.source_index),
        level_xy=(int(factor_x).bit_length() - 1, int(factor_y).bit_length() - 1),
        reducer=reducer,
        plans=plans,
    )


def page_set_key_for(
    session,
    rendered: RenderedTile,
    *,
    demand,
    level: int,
    semantic_source_id=None,
) -> LodPageSetKey:
    source_id = (
        session.tile_semantic_source_id(rendered.tile.source_index)
        if semantic_source_id is None
        else semantic_source_id
    )
    plans = page_plans_for_rendered(
        session,
        rendered,
        demand=demand,
        level=level,
        semantic_source_id=source_id,
    )
    factor_x, factor_y = factor_xy_for_level(demand, int(level))
    return LodPageSetKey(
        source_id=source_id,
        tile_id=int(rendered.tile.source_index),
        level_xy=(int(factor_x).bit_length() - 1, int(factor_y).bit_length() - 1),
        reducer=plans[0].reducer,
        plans=plans,
    )


def _reducer_format_for_rendered(rendered: RenderedTile, source: np.ndarray) -> tuple[str, str, str]:
    mapping = getattr(rendered, "shader_mapping", None)
    component = getattr(mapping, "component", ShaderComponent.REAL)
    component = ShaderComponent(getattr(component, "value", component))
    if np.iscomplexobj(source):
        display_mode = getattr(mapping, "display_mode", ShaderDisplayMode.SCALAR)
        display_mode = ShaderDisplayMode(getattr(display_mode, "value", display_mode))
        if component == ShaderComponent.ABS:
            if display_mode == ShaderDisplayMode.PHASE_COLOR:
                # Complex presentation uses phase for hue and component
                # magnitude for intensity. Preserve both; phase_vector
                # deliberately discards the amplitude used by levels.
                return REDUCER_MEAN, "complex64", COMPLEX_RG32F
            return REDUCER_MEAN_ABS, "float32", SCALAR_R32F
        if component in (ShaderComponent.ANGLE, ShaderComponent.COMPLEX_PHASE):
            return REDUCER_PHASE_VECTOR, "complex64", COMPLEX_RG32F
        return REDUCER_MEAN, "complex64", COMPLEX_RG32F
    if source.ndim != 2:
        raise ValueError("canonical live LOD pages require scalar or complex source values")
    return REDUCER_MEAN, "float32", SCALAR_R32F


def _page_route_format(
    source: np.ndarray,
    *,
    reduction_yx: tuple[int, int],
    reduced_format: tuple[str, str, str],
) -> tuple[str, str, str]:
    """Select native identity at level zero and reducer families above it."""

    if any(int(step) for step in reduction_yx):
        return reduced_format
    if source.ndim != 2:
        raise ValueError("canonical live LOD pages require scalar or complex source values")
    dtype = np.dtype(source.dtype)
    if np.issubdtype(dtype, np.complexfloating):
        return REDUCER_NATIVE, "complex64", COMPLEX_RG32F
    return REDUCER_NATIVE, dtype.name, SCALAR_R32F


def source_origin_yx_for_session(session, source: np.ndarray) -> tuple[int, int]:
    """Locate a rendered source plane on the canonical native source grid.

    Every live producer must pair its precomputed page plans and numeric
    materialization with this same origin.  Falling back to ``(0, 0)`` is
    correct only when the session has no source anchor (ordinary montage
    tiles); applying it to a shifted source window labels window-local values
    with the wrong canonical bins.
    """

    anchor_fn = getattr(session, "_payload_source_anchor", None)
    anchor = anchor_fn(tuple(np.shape(source)[:2])) if callable(anchor_fn) else None
    if anchor is None:
        return (0, 0)
    return (int(anchor.source_rect[0]), int(anchor.source_rect[2]))


def page_plans_for_rendered(
    session,
    rendered,
    *,
    demand,
    level: int,
    native_source=None,
    semantic_source_id=None,
):
    if native_source is None:
        native_source = canonical_value_source_for_rendered(
            rendered, shader_display=bool(getattr(session, "shader_display", True))
        )
    source = np.asarray(native_source)
    reducer, dtype, representation = _reducer_format_for_rendered(rendered, source)
    factor_x, factor_y = factor_xy_for_level(demand, int(level))
    reduction_yx = (int(factor_y).bit_length() - 1, int(factor_x).bit_length() - 1)
    reducer, dtype, representation = _page_route_format(
        source,
        reduction_yx=reduction_yx,
        reduced_format=(reducer, dtype, representation),
    )
    origin_y, origin_x = source_origin_yx_for_session(session, source)
    height, width = (int(value) for value in source.shape[:2])
    anchor_fn = getattr(session, "_payload_source_anchor", None)
    anchor = anchor_fn((height, width)) if callable(anchor_fn) else None
    source_id = (
        session.tile_semantic_source_id(rendered.tile.source_index)
        if semantic_source_id is None
        else semantic_source_id
    )
    content_key = (
        anchor.content_key
        if anchor is not None
        else (
            "src-anchored",
            source_id,
            ("display-plane",),
        )
    )
    return plan_source_grid_pages(
        content_key=content_key,
        valid_source_rect_yx=(origin_y, origin_y + height, origin_x, origin_x + width),
        reduction_yx=reduction_yx,
        stored_page_shape=(256, 256),
        dtype=dtype,
        representation=representation,
        reducer=reducer,
    )


def page_set_key_for_tile(session, tile, *, demand, level: int) -> LodPageSetKey:
    """Plan an unrendered montage tile from session-owned semantic facts."""

    dtype = np.dtype(getattr(session, "output_dtype", np.float32))
    channel_value = getattr(getattr(session, "view_state", None), "channel", "real")
    channel = str(getattr(channel_value, "value", channel_value))
    if np.issubdtype(dtype, np.complexfloating):
        if channel == "abs":
            reducer, planned_dtype, representation = REDUCER_MEAN_ABS, "float32", SCALAR_R32F
        elif channel == "angle":
            reducer, planned_dtype, representation = REDUCER_PHASE_VECTOR, "complex64", COMPLEX_RG32F
        elif channel == "complex":
            reducer, planned_dtype, representation = REDUCER_MEAN, "complex64", COMPLEX_RG32F
        else:
            reducer, planned_dtype, representation = REDUCER_MEAN, "complex64", COMPLEX_RG32F
    else:
        reducer, planned_dtype, representation = REDUCER_MEAN, "float32", SCALAR_R32F
    factor_x, factor_y = factor_xy_for_level(demand, int(level))
    reduction_yx = (int(factor_y).bit_length() - 1, int(factor_x).bit_length() - 1)
    if not any(reduction_yx):
        reducer = REDUCER_NATIVE
        if np.issubdtype(dtype, np.complexfloating):
            planned_dtype, representation = "complex64", COMPLEX_RG32F
        else:
            # Session tile geometry is two-dimensional.  ``uint8`` is a
            # scalar scientific value here, not evidence of RGB components.
            planned_dtype, representation = dtype.name, SCALAR_R32F
    height, width = (int(value) for value in session.plan.tile_shape[:2])
    semantic_source_id = session.tile_semantic_source_id(int(tile.source_index))
    cache_key = (
        semantic_source_id,
        int(tile.source_index),
        (height, width),
        reduction_yx,
        reducer,
        planned_dtype,
        representation,
    )
    route_cache = getattr(session, "_lod_page_set_key_cache", None)
    if route_cache is not None:
        cached = route_cache.get(cache_key)
        if cached is not None:
            return cached
    plans = plan_source_grid_pages(
        content_key=("src-anchored", semantic_source_id, ("display-plane",)),
        valid_source_rect_yx=(0, height, 0, width),
        reduction_yx=reduction_yx,
        stored_page_shape=(256, 256),
        dtype=planned_dtype,
        representation=representation,
        reducer=reducer,
    )
    key = LodPageSetKey(
        source_id=semantic_source_id,
        tile_id=int(tile.source_index),
        level_xy=(int(factor_x).bit_length() - 1, int(factor_y).bit_length() - 1),
        reducer=reducer,
        plans=plans,
    )
    if route_cache is not None:
        route_cache[cache_key] = key
    return key
# --------------------------------------------------------------------------
# Policy & demand (session-side)
# --------------------------------------------------------------------------


def resident_lod_active(session) -> bool:
    return bool(
        str(session.lod_policy_mode) == LOD_POLICY_RESIDENT
        and session.lod_page_cache is not None
    )


def selected_lod_factor(session) -> int:
    previous = session.lod_policy_decision.demand.desired_factor
    if resident_lod_active(session):
        session.lod_policy_decision = resident_lod_policy(
            session.view_range,
            session.viewport_shape,
            session.plan.tile_shape,
            previous_factor=previous,
            resident_levels=session_resident_levels(session, previous),
        )
        base_preview = int(
            getattr(session, "lod_preview_min_level", 0)
            or getattr(session, "lod_preview_level", 0)
            or 0
        )
        if base_preview > 0:
            desired_level = int(getattr(session.lod_policy_decision.demand, "desired_level", 0) or 0)
            session.lod_preview_level = max(base_preview, desired_level)
    else:
        session.lod_policy_decision = native_lod_policy(
            session.view_range,
            session.viewport_shape,
            session.plan.tile_shape,
            previous_factor=previous,
            deferred_reason=session.lod_native_reason,
        )
    return int(session.lod_policy_decision.applied_factor)


def _demand_key_sig(demand) -> tuple:
    """The demand fields that pyramid keys depend on (via factor_xy_for_level)."""

    return (
        int(demand.desired_level),
        tuple(int(value) for value in demand.desired_factor_xy),
        tuple(int(value) for value in demand.acceptable_levels),
    )


def _page_set_complete(cache: LodPageCache | None, key: LodPageSetKey) -> bool:
    return bool(
        isinstance(cache, LodPageCache)
        and cache.resolve_plans(key.plans) is not None
    )


def _page_set_exact(cache: LodPageCache | None, key: LodPageSetKey) -> bool:
    return bool(
        isinstance(cache, LodPageCache)
        and cache.exact_pages(key.plans) is not None
    )


def _page_set_materialized_pages(
    cache: LodPageCache | None,
    key: LodPageSetKey,
) -> tuple[MaterializedLodPage, ...] | None:
    if not isinstance(cache, LodPageCache):
        return None
    return cache.resolved_pages(key.plans)


def _page_set_resolution(
    cache: LodPageCache | None,
    key: LodPageSetKey,
) -> ResolvedLodPageSet | None:
    """Canonical complete requested -> actual page-set truth."""

    if not isinstance(cache, LodPageCache):
        return None
    return cache.resolved_page_set(key.plans)


def _conservative_actual_level_for_payload(payload) -> int:
    """Coarsest physical page binding used by presentation policy.

    ``DisplayTilePayload.lod`` is the requested semantic page geometry.  It
    cannot describe heterogeneous target bindings, so floor/no-demotion
    decisions consult the canonical resolved page set instead.  Non-page
    payloads retain their ordinary singular LOD meaning.
    """

    return int(
        getattr(
            payload,
            "conservative_actual_lod_level",
            int(getattr(getattr(payload, "lod", None), "level", 0) or 0),
        )
    )


def tile_resident_levels(session, rendered: RenderedTile, *, demand) -> tuple[int, ...]:
    """Resident acceptable levels (>0) for one rendered tile, memoized.

    During a scrub step the same scan runs from the session-wide decision,
    per-tile texture selection, and the presentation commit. The memo
    lives on the session keyed by (source index, component), guarded by the
    pyramid ``revision`` and the demand signature: a hit costs two dict
    probes and is exact because the revision bumps on every admission,
    eviction, resize, and clear.
    """

    pyramid = session.lod_page_cache
    memo = getattr(session, "_lod_resident_levels_memo", None)
    if memo is None:
        memo = {}
        session._lod_resident_levels_memo = memo
    # The demand tuple-signature itself is per-call cost at ~1.4k calls per
    # scrub step; demand objects are immutable (frozen dataclass), so one
    # id()-keyed slot amortizes it.  Safe: the cached entry keeps a strong
    # reference to the demand, so its id cannot be reused while cached.
    sig_cache = getattr(session, "_lod_demand_sig_cache", None)
    if sig_cache is not None and sig_cache[0] is demand:
        demand_sig = sig_cache[1]
    else:
        demand_sig = _demand_key_sig(demand)
        session._lod_demand_sig_cache = (demand, demand_sig)
    guard = (int(pyramid.revision), demand_sig)
    memo_key = (
        int(rendered.tile.source_index),
        component_for_rendered(rendered, shader_display=bool(getattr(session, "shader_display", True))),
    )
    hit = memo.get(memo_key)
    if hit is not None and hit[0] == guard:
        return hit[1]
    levels = set()
    for level in demand.acceptable_levels:
        level = int(level)
        if level <= 0:
            continue
        try:
            key = page_set_key_for(session, rendered, demand=demand, level=level)
        except ValueError:
            continue
        if _page_set_exact(pyramid, key):
            levels.add(level)
    levels = tuple(sorted(levels))
    memo[memo_key] = (guard, levels)
    return levels


def session_resident_levels(session, previous_factor: int) -> tuple[int, ...]:
    """Levels resident for every rendered tile (session-wide decision input).

    Per-tile texture selection probes its own resident set; the
    session-wide decision reports the level that every rendered tile can
    actually present, keeping diagnostics honest for partial residency.
    """

    demand = select_lod_demand(
        session.view_range,
        session.viewport_shape,
        session.plan.tile_shape,
        previous_factor=previous_factor,
    )
    rendered = tuple(session.rendered_tiles.values())
    if not rendered:
        return ()
    common: set[int] | None = None
    for tile in rendered:
        levels = set(tile_resident_levels(session, tile, demand=demand))
        common = levels if common is None else (common & levels)
        if not common:
            return ()
    return tuple(sorted(common or ()))


def presented_lod_summary(session) -> tuple[int, int, tuple[int, int]]:
    """Physical LOD shown by the plurality of target page bindings.

    The session-wide policy decision only claims a level once every
    rendered tile can present it, which reads as "native" while any tile
    is still streaming.  Diagnostics report what the committed
    presentation actually samples, so heterogeneous page ancestors remain
    separate observations instead of becoming one fictional tile level.
    Non-page payloads contribute their ordinary singular LOD.  Ties prefer
    the finer level (ADR 0050).
    """

    payloads = dict(getattr(session.tile_presentation_state, "payloads", {}) or {})
    visible = session.visible_tile_numbers
    if visible:
        scoped = {tile: payload for tile, payload in payloads.items() if int(tile) in visible}
        payloads = scoped or payloads
    if not payloads:
        decision = session.lod_policy_decision
        return (
            int(decision.applied_level),
            int(decision.applied_factor),
            tuple(int(value) for value in decision.applied_factor_xy),
        )
    counts: dict[tuple[int, int, int, int], int] = {}
    for payload in payloads.values():
        backing = getattr(payload, "page_backing", None)
        resolved = getattr(backing, "resolved_page_set", None)
        if isinstance(resolved, ResolvedLodPageSet):
            for reduction_yx in resolved.actual_reductions_yx:
                factor_y = 1 << int(reduction_yx[0])
                factor_x = 1 << int(reduction_yx[1])
                level = max(tuple(int(value) for value in reduction_yx), default=0)
                row = (level, max(factor_x, factor_y), factor_x, factor_y)
                counts[row] = counts.get(row, 0) + 1
            continue
        lod = getattr(payload, "lod", None)
        level = int(getattr(lod, "level", 0) or 0)
        factor = int(getattr(lod, "factor", 1) or 1)
        if lod is None or level <= 0:
            row = (0, 1, 1, 1)
        else:
            source_shape = tuple(getattr(lod, "source_shape", (1, 1)))
            texture_shape = tuple(getattr(lod, "texture_shape", (1, 1)))
            factor_y = max(
                1,
                round(int(source_shape[0]) / max(1, int(texture_shape[0]))),
            )
            factor_x = max(
                1,
                round(int(source_shape[1]) / max(1, int(texture_shape[1]))),
            )
            row = (level, factor, factor_x, factor_y)
        counts[row] = counts.get(row, 0) + 1
    level, factor, factor_x, factor_y = min(
        counts,
        key=lambda candidate: (-counts[candidate], candidate),
    )
    return (level, factor, (factor_x, factor_y))


def ingest_lod_demand(session) -> object | None:
    """Demand snapshot for worker-side reduce-at-ingest (ADR 0050).

    When the resident policy currently wants a reduced level, a cold
    tile's worker should produce that level together with the native
    result so the first upload is the reduced payload.  The returned
    ``LodDemand`` is immutable; a demand change between scheduling and
    completion is corrected by the ordinary streaming path.
    """

    if not resident_lod_active(session):
        return None
    demand = session.lod_policy_decision.demand
    if int(demand.desired_level) <= 0:
        return None
    return demand


def missing_tiles_require_native_target(lod_policy_mode: str, demand) -> bool:
    """Return whether missing tiles owe an immediate native target.

    Resident LOD separates first correct pixels from exact/native refinement:
    if the viewport demands a reduced display level, cold tiles should be
    filled by ladder/pipeline page rungs first. Native-only policy, invalid
    demand, and native-scale resident demand still require native evaluation.
    This is a target-policy query; it owns no queue or scheduling state.
    """

    if str(lod_policy_mode) != LOD_POLICY_RESIDENT:
        return True
    if demand is None:
        return True
    return int(getattr(demand, "desired_level", 0) or 0) <= 0


# --------------------------------------------------------------------------
# Materialization planning (session-side)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LodPageSetKey:
    """One tile/rung lifecycle identity backed by canonical logical pages."""

    source_id: object
    tile_id: int
    level_xy: tuple[int, int]
    reducer: str
    plans: tuple[LodPagePlan, ...]

    def __post_init__(self) -> None:
        plans = tuple(self.plans)
        if not plans or len({plan.key for plan in plans}) != len(plans):
            raise ValueError("LOD page set requires unique canonical plans")
        if any(plan.reducer != self.reducer for plan in plans):
            raise ValueError("LOD page-set reducer disagrees with a canonical plan")
        object.__setattr__(self, "plans", plans)

    @property
    def page_keys(self) -> tuple[DataChunkKey, ...]:
        return tuple(plan.key for plan in self.plans)

    @property
    def level(self) -> int:
        return max(int(value) for value in self.level_xy)

    @property
    def factor_xy(self) -> tuple[int, int]:
        return tuple(1 << int(value) for value in self.level_xy)


@dataclass(frozen=True)
class LodPageMaterializationRequest:
    """One page-set request with only the CPU claims it newly owns."""

    tile_number: int
    key: LodPageSetKey
    source: object
    source_origin_yx: tuple[int, int]
    plans: tuple[LodPagePlan, ...]
    claimed_plans: tuple[LodPagePlan, ...]
    owner: object

    @property
    def chain(self) -> tuple:
        # TileLifecycle owns one tile/rung record. Individual page claims are
        # cache state carried by ``claimed_plans`` and ``owner``.
        return ((self.key, (1, 1)),)

    @property
    def reduce_factor_xy(self) -> tuple[int, int]:
        return self.key.factor_xy


def plan_materialization(
    session,
    rendered: RenderedTile,
    *,
    demand,
    level: int,
    key: LodPageSetKey,
    native_source: np.ndarray | None = None,
) -> LodPageMaterializationRequest:
    """Claim only missing canonical pages; numeric work stays direct-source."""

    if native_source is None:
        native_source = canonical_value_source_for_rendered(
            rendered, shader_display=bool(getattr(session, "shader_display", True))
        )
    tile_number = int(rendered.tile.montage_index)
    cache = session.lod_page_cache
    if not isinstance(cache, LodPageCache):
        raise TypeError("resident LOD requires the logical LodPageCache")
    plans = key.plans
    owner = ("lod-page-request", id(session), tile_number, key)
    claimed = cache.claim_plans(plans, owner)
    return LodPageMaterializationRequest(
        tile_number=tile_number,
        key=key,
        source=native_source,
        source_origin_yx=source_origin_yx_for_session(
            session,
            np.asarray(native_source),
        ),
        plans=plans,
        claimed_plans=claimed,
        owner=owner,
    )


def mark_ladder_swaps_for_viewport(session) -> bool:
    """Re-evaluate LOD demand after a camera-only retarget (ADR 0050).

    Camera changes never restart evaluation. They only request presentation
    work when the current payload is too coarse for the new demand, still a
    preview, or absent. A coarser demand never replaces an exact finer
    presentation. Physical eviction pressure belongs to the backend that can
    first reclaim hidden/speculative residency; a logical payload-byte sum is
    not authority to lower visible quality. This recomputes the decision from the current
    ``view_range``/``viewport_shape`` (demand math plus pyramid peeks; never
    reduction or other bulk work), queues singleflight materializations for
    missing display payloads, and dirties tiles only when a swap improves
    correctness rather than merely matching a coarser preference.

    Returns True when at least one visible tile can present a different
    resident level right now, i.e. a presentation commit is worthwhile
    even though no tile result arrived.
    """

    if not resident_lod_active(session):
        return False
    identity = viewport_identity(session.view_range, session.viewport_shape)
    if identity != session._lod_refresh_viewport_identity:
        session._lod_refresh_viewport_identity = identity
        # Stale zoom targets must cancel: LOD materializations supersede
        # on this dedicated counter.  viewport_revision is owned by the
        # retarget replan — bumping it here changed priority-retarget
        # work identities without replanning, churning work at idle.
        session.lod_target_revision += 1
    selected_lod_factor(session)
    return mark_ladder_swaps_for_current_demand(session)


def mark_ladder_swaps_for_current_demand(session) -> bool:
    """Dirty resident swaps for the policy decision already on ``session``.

    The pipeline retarget boundary recomputes LOD demand before it retargets
    lifecycle identities.  That final decision can differ from the earlier
    viewport callback's decision (for example while the camera is still
    delivering zoom updates).  Marking here keeps the target update and its
    presentation obligation atomic without scheduling or recomputing demand.
    """

    if not resident_lod_active(session):
        return False
    demand = session.lod_policy_decision.demand
    desired = int(demand.desired_level)
    commit_needed = False
    visible_by_number = {int(t.montage_index): t for t in tuple(session.visible_tiles)}
    acknowledged_payloads = dict(
        getattr(getattr(session, "tile_presentation_state", None), "payloads", {}) or {}
    )
    # Priority order, not row order: materializations start immediately
    # for the demanded level, and whatever the workers complete before a
    # newer viewport supersedes the rest is the work nearest the
    # focus/pointer — never wasted, never waited for (Thomas's rule:
    # optimal ordering and cancellation beat debouncing every time).
    for tile_number in session._prioritized_tile_numbers(tuple(session.visible_tile_numbers)):
        rendered = session.rendered_tiles.get(int(tile_number))
        if rendered is None:
            # Unrendered tiles never enter dirty_payloads: the dirty set
            # is consumed by acknowledged upserts, and a build cannot
            # produce an exact upsert without a rendered result — a
            # permanently dirty tile turns the final-commit check into a
            # busy timer loop.  Floor progress (a presentable or closer
            # resident level) only requests a commit; the build's floor
            # pass does the actual work.
            if floor_can_progress(session, int(tile_number), tile=visible_by_number.get(int(tile_number))):
                commit_needed = True
            continue
        payload = acknowledged_payloads.get(
            int(tile_number),
            session.display_tile_payloads.get(int(tile_number)),
        )
        presented_level = _conservative_actual_level_for_payload(payload)
        resident = set(tile_resident_levels(session, rendered, demand=demand))
        if payload is not None and presented_level > 0 and presented_level in demand.acceptable_levels:
            # The presented texture itself is materialized and resident;
            # keep it eligible even when the pyramid cache dropped it so a
            # transient cache miss never forces a native down-swap.
            resident.add(presented_level)
        # Missing levels are scheduled by the ladder's DESIRED rung. This
        # viewport pass only marks immediately swappable resident levels.
        if payload is None:
            continue
        if str(getattr(payload, "quality", "exact")) != "exact":
            # A preview payload can sit at an acceptable level and look
            # converged, but preview never satisfies convergence: the
            # tile has a rendered result, so the exact rebuild is one
            # cheap dirty away — without this, a floored tile whose
            # level matched demand stayed blocky forever next to exact
            # neighbors.
            session.dirty_payloads[int(tile_number)] = None
            commit_needed = True
            continue
        if presented_level <= desired:
            continue
        applied = int(choose_resident_level(demand, tuple(sorted(resident))))
        if presented_level != applied:
            session.dirty_payloads[int(tile_number)] = None
            commit_needed = True
    return commit_needed


def preserve_finer_presented_payload(session, payload) -> bool:
    """Whether an exact finer payload must survive an unrelated rebuild.

    LOD materialization, levels, and lifecycle repair can dirty a tile for
    reasons other than an authorized resolution change. Texture selection
    must not turn those bookkeeping events into quality demotion.
    """

    if payload is None or str(getattr(payload, "quality", "exact")) != "exact":
        return False
    demand = session.lod_policy_decision.demand
    presented = _conservative_actual_level_for_payload(payload)
    desired = int(getattr(demand, "desired_level", 0) or 0)
    return presented < desired


# --------------------------------------------------------------------------
# Worker-side admissions
# --------------------------------------------------------------------------


def admit_retained_preview_level(
    preview_pyramid,
    rendered: RenderedTile,
    *,
    semantic_source_id,
    preview_level: int,
    shader_display: bool = True,
) -> LodPageSetKey | None:
    """Admit the retained preview level for a freshly computed tile.

    Worker-side and opportunistic (ADR 0050 retained preview level): prefetch
    can pin a coarse copy in the shared pyramid cache, so any index ever
    computed re-presents instantly through the floor. When a finer resident
    history never selects its values: it always executes the canonical direct
    source route for any newly claimed page.
    """

    if not isinstance(preview_pyramid, LodPageCache) or int(preview_level) <= 0:
        return None
    shader_display = bool(shader_display)
    level = int(preview_level)
    source = canonical_value_source_for_rendered(
        rendered, shader_display=shader_display
    )
    reducer, dtype, representation = _reducer_format_for_rendered(rendered, source)
    height, width = (int(value) for value in np.shape(source)[:2])
    plans = plan_source_grid_pages(
        content_key=("src-anchored", semantic_source_id, ("display-plane",)),
        valid_source_rect_yx=(0, height, 0, width),
        reduction_yx=(level, level),
        stored_page_shape=(256, 256),
        dtype=dtype,
        representation=representation,
        reducer=reducer,
    )
    key = LodPageSetKey(
        source_id=semantic_source_id,
        tile_id=int(rendered.tile.source_index),
        level_xy=(level, level),
        reducer=reducer,
        plans=plans,
    )
    if _page_set_exact(preview_pyramid, key):
        return None
    owner = ("retained-preview-pages", semantic_source_id, key)
    claimed = preview_pyramid.claim_plans(plans, owner)
    if not claimed:
        return None
    if not preview_pyramid.begin_owner_work(owner):
        preview_pyramid.release_owner_claims(owner)
        raise RuntimeError("retained preview page claims disappeared before admission")
    try:
        for plan in claimed:
            page = materialize_lod_page(source, source_origin_yx=(0, 0), plan=plan)
            preview_pyramid.admit_as(plan.key, page, owner=owner)
    finally:
        preview_pyramid.finish_owner_work(owner)
    if not _page_set_exact(preview_pyramid, key):
        return None
    return key


# --------------------------------------------------------------------------
# Presentation floor (session-side)
# --------------------------------------------------------------------------


def best_floor_key(session, source_index: int, *, tile_number: int | None = None):
    """Best complete floor, ranked by physical rather than requested quality."""

    pyramid = session.lod_page_cache
    demand = session.lod_policy_decision.demand
    desired = int(demand.desired_level)
    semantic_id = session.tile_semantic_source_id(int(source_index))
    candidates = []
    seen: set[LodPageSetKey] = set()

    def add_candidate(
        key: LodPageSetKey,
        resolved: ResolvedLodPageSet,
        *,
        owner=None,
    ) -> None:
        if key in seen:
            return
        seen.add(key)
        coarsest_actual_level = int(resolved.coarsest_actual_level)
        target_level = int(key.level)
        metadata_fn = getattr(session, "preview_floor_metadata", None)
        metadata = metadata_fn(key) if callable(metadata_fn) else None
        if owner is ClaimOwner.PREVIEW:
            semantic_quality = str(getattr(metadata, "quality", "preview") or "preview")
        else:
            semantic_quality = "exact"
        row = (
            (
                resident_presentation_rank(coarsest_actual_level, desired),
                0 if semantic_quality == "exact" else 1,
                0 if resolved.exact else 1,
                abs(target_level - desired),
                target_level,
                len(seen),
            ),
            key,
            coarsest_actual_level,
            pyramid,
        )
        candidates.append(row)

    records = () if tile_number is None else (session.lifecycle.peek(int(tile_number)),)
    for rec in records:
        if rec is None:
            continue
        for key, entry in rec.levels.items():
            if entry.phase is not LevelPhase.RESIDENT or not isinstance(key, LodPageSetKey):
                continue
            if key.source_id != semantic_id or int(key.tile_id) != int(source_index):
                continue
            resolved = _page_set_resolution(pyramid, key)
            if resolved is not None:
                add_candidate(key, resolved, owner=entry.owner)

    tile = next(
        (
            value
            for value in tuple(getattr(session.plan, "tiles", ()) or ())
            if int(value.source_index) == int(source_index)
            and (tile_number is None or int(value.montage_index) == int(tile_number))
        )
        ,
        None,
    )
    if tile is not None:
        preview_level = int(getattr(session, "lod_preview_level", 0) or 0)
        exact_levels = tuple(
            dict.fromkeys(
                (
                    preview_level,
                    desired,
                    *tuple(int(value) for value in demand.acceptable_levels),
                )
            )
        )
        for level in exact_levels:
            if level <= 0:
                continue
            key = page_set_key_for_tile(session, tile, demand=demand, level=level)
            resolved = _page_set_resolution(pyramid, key)
            if resolved is not None and resolved.exact:
                add_candidate(key, resolved)

        # If no exact known rung covers this source, resolve only the semantic
        # target (or the retained preview for native demand). Scanning every
        # acceptable hypothetical target let one physical L4 page masquerade
        # as L1/L2/L3 and made requested identity decide visible quality.
        positive_acceptable = tuple(
            int(value) for value in demand.acceptable_levels if int(value) > 0
        )
        fallback_level = (
            desired
            if desired > 0
            else preview_level
            if preview_level > 0
            else min(positive_acceptable, default=0)
        )
        if fallback_level > 0:
            key = page_set_key_for_tile(
                session,
                tile,
                demand=demand,
                level=fallback_level,
            )
            resolved = _page_set_resolution(pyramid, key)
            if resolved is not None:
                add_candidate(key, resolved)

    if not candidates:
        return None
    _rank, key, coarsest_actual_level, owning_cache = min(
        candidates,
        key=lambda item: item[0],
    )
    return (key, coarsest_actual_level, owning_cache)


def _floor_resolution_and_quality(session, key, cache):
    resolved = _page_set_resolution(cache, key)
    if resolved is None:
        return None, None, "preview"
    metadata_fn = getattr(session, "preview_floor_metadata", None)
    metadata = metadata_fn(key) if callable(metadata_fn) else None
    quality = str(getattr(metadata, "quality", "preview") or "preview")
    if not resolved.exact:
        quality = "preview"
    return resolved, metadata, quality


def floor_can_progress(session, tile_number: int, tile=None) -> bool:
    """True when the floor pass could present or improve this tile."""

    if not resident_lod_active(session):
        return False
    payload = session.display_tile_payloads.get(int(tile_number))
    if payload is not None and int(tile_number) in session.active_tile_requests:
        # Something is on screen and the exact result is in flight: improving
        # the preview now would churn payload identities for no visible win.
        # A BLANK tile is different — see ensure_floor_payloads.
        return False
    if tile is None:
        tile = next(
            (t for t in tuple(session.visible_tiles) if int(t.montage_index) == int(tile_number)),
            None,
        )
    if tile is None:
        return False
    best = best_floor_key(session, int(tile.source_index), tile_number=int(tile_number))
    if best is None:
        return False
    if payload is None:
        return True
    presented_actual_level = _conservative_actual_level_for_payload(payload)
    best_actual_level = int(best[1])
    resolved, _metadata, best_quality = _floor_resolution_and_quality(
        session,
        best[0],
        best[2],
    )
    if resolved is None:
        return False
    if best_actual_level == presented_actual_level:
        current_backing = getattr(payload, "page_backing", None)
        current_resolved = getattr(current_backing, "resolved_page_set", None)
        current_actual_keys = tuple(
            getattr(current_resolved, "actual_keys", ()) or ()
        )
        current_requested_keys = tuple(
            getattr(current_backing, "requested_keys", ()) or ()
        )
        if (
            current_actual_keys != tuple(resolved.actual_keys)
            or current_requested_keys != tuple(best[0].page_keys)
        ):
            return True
    if str(getattr(payload, "quality", "exact")) != "preview":
        # "Exact" describes the reduced target's semantic quality, not native
        # resolution. A level-5 exact target must still swap to an already
        # resident level-1 exact target after zooming in. Refusing every exact
        # floor left the ladder with zero work (target resident) while the
        # backend stayed permanently coarse.
        desired = int(session.lod_policy_decision.demand.desired_level)
        return bool(
            presented_actual_level > desired
            and best_actual_level < presented_actual_level
        )
    return bool(
        best_actual_level != presented_actual_level
        or best_quality != str(getattr(payload, "quality", "preview") or "preview")
    )


def ensure_floor_payloads(session, tile_numbers, *, max_count: int | None = None) -> None:
    """Present the best resident pyramid level for unrendered planned tiles.

    The floor invariant (ADR 0050): a planned tile with any resident
    level never shows a placeholder.  Floor payloads are quality
    "preview" — they draw pixels but refuse semantic reads — and the
    ordinary evaluation path replaces them with exact payloads as tile
    results arrive.  Dictionary probes only; no reduction, no copies.
    """

    if not tile_numbers or not resident_lod_active(session):
        return
    cache = session.lod_page_cache
    if cache is None:
        return
    by_number = {
        int(tile.montage_index): tile
        for tile in tuple(session.visible_tiles)
    }
    preview_pass_scope = (
        set(int(tile) for tile in session.lod_preview_floor_scope)
        if session._lod_preview_floor_first_fill_active(
            session.required_tile_numbers()
        )
        else set()
    )
    built = 0
    for tile_number in tuple(dict.fromkeys(int(number) for number in tuple(tile_numbers))):
        if max_count is not None and built >= int(max_count):
            break
        existing = session.display_tile_payloads.get(tile_number)
        if existing is not None and tile_number in session.active_tile_requests:
            # An exact evaluation is in flight and SOMETHING is on screen:
            # improving the preview now would double payload/identity churn
            # for every tile of a cold fill.  A blank tile is the opposite
            # case — the GUI bar (show something immediately, refine later)
            # outranks churn, and field evidence (2026-07-05) showed slow
            # stage-backed fills leaving black tiles for seconds while their
            # level-2 floor planes sat resident in the pinned cache.
            continue
        tile = by_number.get(tile_number)
        if tile is None:
            continue
        source_index = int(tile.source_index)
        semantic_id = session.tile_semantic_source_id(source_index)
        best = best_floor_key(session, source_index, tile_number=int(tile_number))
        if best is None:
            continue
        key, coarsest_actual_level, owning_cache = best
        resolved, metadata, best_quality = _floor_resolution_and_quality(
            session,
            key,
            owning_cache,
        )
        if resolved is None:
            continue
        # Quality describes the presentation pass as well as the provenance
        # of the cached page. During a declared first-pixel pass, even an exact
        # reduced page is exposed conservatively as preview: its pixels can
        # cover the slot immediately, but it must not become a target-quality
        # island before peer slots have first pixels. The same page can be
        # reclassified/refined after physical preview coverage closes.
        presentation_quality = (
            "preview" if tile_number in preview_pass_scope else best_quality
        )
        if existing is not None:
            presented_actual_level = _conservative_actual_level_for_payload(existing)
            existing_backing = getattr(existing, "page_backing", None)
            existing_resolved = getattr(existing_backing, "resolved_page_set", None)
            existing_actual_keys = tuple(
                getattr(existing_resolved, "actual_keys", ()) or ()
            )
            existing_requested_keys = tuple(
                getattr(existing_backing, "requested_keys", ()) or ()
            )
            if (
                presented_actual_level == int(coarsest_actual_level)
                and str(getattr(existing, "quality", "preview") or "preview")
                == presentation_quality
                and existing_actual_keys == tuple(resolved.actual_keys)
                and existing_requested_keys == tuple(key.page_keys)
            ):
                continue
            if (
                tile_number not in preview_pass_scope
                and str(getattr(existing, "quality", "exact")) != "preview"
            ):
                desired = int(session.lod_policy_decision.demand.desired_level)
                if (
                    presented_actual_level <= desired
                    or int(coarsest_actual_level) >= presented_actual_level
                ):
                    continue
        pages = resolved.materialized_pages
        texture_kind = getattr(metadata, "texture_kind", None)
        if texture_kind is None:
            texture_kind = (
                TexturePlaneKind.COMPLEX_RG32F
                if pages[0].key.representation == COMPLEX_RG32F
                else TexturePlaneKind.SCALAR_R32F
            )
        shader_mapping = getattr(metadata, "shader_mapping", None)
        if (
            shader_mapping is None
            and texture_kind == TexturePlaneKind.COMPLEX_RG32F
            and getattr(session, "view_state", None) is not None
        ):
            # A resident complex floor plane can outlive the session that
            # recorded its preview metadata (the pyramid cache persists;
            # ``lod_preview_metadata`` is per-session).  A complex texture
            # presented WITHOUT its mapping draws magnitude through the
            # cyclic LUT — zero magnitude renders LUT[0] orange (field
            # defect 2026-07-16 09:14: entering tiles of a montage window
            # change flashed orange until exact payloads replaced them).
            # The mapping is a pure function of the current view state.
            from arrayscope.display.slice_engine import complex_texture_shader_mapping

            shader_mapping = complex_texture_shader_mapping(
                session.view_state,
                getattr(session, "colormap_lut", None),
            )
        requested_factor_x = 1 << int(key.level_xy[0])
        requested_factor_y = 1 << int(key.level_xy[1])
        tile_shape = tuple(int(value) for value in session.plan.tile_shape)
        requested_lod = LodInfo(
            level=key.level,
            factor=max(requested_factor_x, requested_factor_y),
            source_shape=tile_shape,
            texture_shape=(
                max(plan.stored_rect_yx[1] for plan in key.plans)
                - min(plan.stored_rect_yx[0] for plan in key.plans),
                max(plan.stored_rect_yx[3] for plan in key.plans)
                - min(plan.stored_rect_yx[2] for plan in key.plans),
            ),
            gutter=0,
        )
        # ADR 0056 G5 slice 1: an EXACT reduced plane covering the whole
        # display window carries the window-invariant source anchor (native
        # rect = window start + native tile extent), so the VisPy pool can
        # take the chunked-residency path with uniform plane-pixel pages.
        # Preview floors stay unanchored: the anchor promises texels that
        # are a pure function of (content key, rect, LOD), which degraded
        # planes do not honor.  ``_payload_source_anchor`` itself returns
        # None for montage sessions and unanchorable chains.
        source_anchor = None
        if presentation_quality == "exact" and resolved.exact:
            anchor_fn = getattr(session, "_payload_source_anchor", None)
            if callable(anchor_fn):
                source_anchor = anchor_fn(tile_shape)
        payload = DisplayTilePayload(
            tile_number=tile_number,
            source_index=source_index,
            image=np.asarray(pages[0].values),
            histogram_data=None,
            source_id=(
                *semantic_id,
                "floor",
                key.reducer,
                ("actual-pages", resolved.actual_keys),
            ),
            texture_data=np.asarray(pages[0].values),
            texture_kind=texture_kind,
            lod=requested_lod,
            quality=presentation_quality,
            shader_mapping=shader_mapping,
            source_anchor=source_anchor,
            page_backing=PageBackedPresentation(
                requested_plans=key.plans,
                materialized_pages=pages,
                source_coverage_yx=(
                    min(plan.valid_source_rect_yx[0] for plan in key.plans),
                    max(plan.valid_source_rect_yx[1] for plan in key.plans),
                    min(plan.valid_source_rect_yx[2] for plan in key.plans),
                    max(plan.valid_source_rect_yx[3] for plan in key.plans),
                ),
                requested_lod=requested_lod,
            ),
            level_data=getattr(metadata, "level_data", None),
            level_stats=getattr(metadata, "level_stats", None),
            tile_identity=session.tile_payload_identity(
                session.plan.tiles[int(tile_number)],
                texture_data=np.asarray(pages[0].values),
                texture_kind=(
                    TexturePlaneKind.RGB8
                    if not bool(getattr(session, "shader_display", True))
                    and texture_kind == TexturePlaneKind.COMPLEX_RG32F
                    else texture_kind
                ),
                shader_mapping=shader_mapping,
                lod=requested_lod,
                quality=presentation_quality,
            ),
            presentation_identity=session.tile_presentation_identity(shader_mapping),
        )
        session.display_tile_payloads[tile_number] = payload
        session.record_tile_payload(payload)
        lifecycle = getattr(session, "lifecycle", None)
        if lifecycle is not None and hasattr(lifecycle, "remember_presentable"):
            lifecycle.remember_presentable(tile_number, payload)
        session.pending_payload_upserts[tile_number] = None
        session.lod_floor_presentations = int(getattr(session, "lod_floor_presentations", 0) or 0) + 1
        built += 1


# --------------------------------------------------------------------------
# Texture selection (session-side)
# --------------------------------------------------------------------------


def resident_texture_for_rendered_tile(
    session,
    rendered: RenderedTile,
    *,
    source: np.ndarray,
    histogram: np.ndarray | None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    LodInfo,
    PageBackedPresentation | None,
    TexturePlaneKind | None,
]:
    """Cache-lookup-only level application (no reduction in this path).

    A demanded level that is not cached is not resident; the tile falls
    back to the nearest resident/native level and the missing level is
    recorded once (singleflight) for the renderer to materialize in the
    background.
    """

    source_shape = tuple(int(value) for value in source.shape[:2])
    native_lod = LodInfo(level=0, factor=1, source_shape=source_shape, texture_shape=source_shape, gutter=0)
    if source.ndim >= 3 and source.shape[-1] in (3, 4):
        native_texture_kind = TexturePlaneKind.RGB8
    elif np.iscomplexobj(source) or (source.ndim >= 3 and source.shape[-1] == 2):
        native_texture_kind = TexturePlaneKind.COMPLEX_RG32F
    else:
        native_texture_kind = TexturePlaneKind.SCALAR_R32F
    demand = session.lod_policy_decision.demand
    pyramid = session.lod_page_cache
    resident_levels = tile_resident_levels(session, rendered, demand=demand)
    # Missing demanded levels are planned by LodLadder. Texture selection is
    # lookup-only so a presentation build cannot create a hidden scheduling
    # queue with its own wakeup rules.
    applied = choose_resident_level(demand, resident_levels)
    if applied <= 0:
        return source, histogram, native_lod, None, native_texture_kind
    key = page_set_key_for(session, rendered, demand=demand, level=applied)
    pages = _page_set_materialized_pages(pyramid, key)
    if not pages:
        return source, histogram, native_lod, None, native_texture_kind
    factor_xy = factor_xy_for_level(demand, applied)
    stored_y0 = min(plan.stored_rect_yx[0] for plan in key.plans)
    stored_y1 = max(plan.stored_rect_yx[1] for plan in key.plans)
    stored_x0 = min(plan.stored_rect_yx[2] for plan in key.plans)
    stored_x1 = max(plan.stored_rect_yx[3] for plan in key.plans)
    lod = LodInfo(
        level=applied,
        factor=max(int(factor_xy[0]), int(factor_xy[1])),
        source_shape=source_shape,
        texture_shape=(stored_y1 - stored_y0, stored_x1 - stored_x0),
        gutter=0,
    )
    page_backing = PageBackedPresentation(
        requested_plans=key.plans,
        materialized_pages=pages,
        source_coverage_yx=(
            min(plan.valid_source_rect_yx[0] for plan in key.plans),
            max(plan.valid_source_rect_yx[1] for plan in key.plans),
            min(plan.valid_source_rect_yx[2] for plan in key.plans),
            max(plan.valid_source_rect_yx[3] for plan in key.plans),
        ),
        requested_lod=lod,
    )
    texture_kind = (
        TexturePlaneKind.COMPLEX_RG32F
        if pages[0].key.representation == COMPLEX_RG32F
        else TexturePlaneKind.SCALAR_R32F
    )
    return np.asarray(pages[0].values), None, lod, page_backing, texture_kind


def release_session_claims(session) -> int:
    """Release every page claim still held by a session's lifecycle records.

    A session can die between planning (claims taken in refresh/build) and
    scheduling (claims handed to work items) — slice scrubbing replaces
    sessions faster than the drain runs.  The pyramid is renderer-shared and
    its keys are semantic, so a claim leaked by a dead session blocks the
    same page when the user scrubs back to that slice: no later request can
    acquire it and the tile presents the wrong LOD forever.  Call
    on every session replacement; in-flight scheduled items are not touched
    (their own release paths balance them).
    """

    if session is None:
        return 0
    pyramid = getattr(session, "lod_page_cache", None)
    lifecycle = getattr(session, "lifecycle", None)
    if lifecycle is not None:
        requests = tuple(lifecycle.active_materializations())
        released = 0
        if isinstance(pyramid, LodPageCache):
            for request in requests:
                released += len(pyramid.release_owner_claims(request.owner))
        for effect in lifecycle.session_replaced():
            if effect.owner is ClaimOwner.PREVIEW:
                getattr(session, "lod_preview_metadata", {}).pop(effect.level_key, None)
        return released
    requests = list(getattr(session, "pending_rung_materializations", ()) or ())
    if pyramid is None:
        return 0
    released = 0
    for request in requests:
        if isinstance(pyramid, LodPageCache):
            released += len(pyramid.release_owner_claims(request.owner))
    session.pending_rung_materializations.clear()
    return released


# --------------------------------------------------------------------------
# Renderer-side: policy resolution, caches, scheduling
# --------------------------------------------------------------------------


def policy_mode_for_renderer(renderer) -> str:
    """Resolve the configured montage LOD policy for the active backend.

    The user setting owns the production policy.  Both tiled backends now use
    the same resident-LOD ladder; backend differences live in presentation
    capabilities, not in policy downgrades.
    """

    choice = getattr(getattr(renderer.win, "app_settings", None), "montage_quality_policy", None)
    value = str(getattr(choice, "value", choice) or LOD_POLICY_NATIVE_ONLY)
    if value != LOD_POLICY_RESIDENT:
        return LOD_POLICY_NATIVE_ONLY
    return LOD_POLICY_RESIDENT


def native_policy_reason_for_renderer(renderer) -> str:
    """Explain *why* native-only applies (honest diagnostics, ADR 0050).

    Distinguishes the user-selected native-only policy from a backend that
    resident LOD has not been adopted on yet; `tile_lod_reason` surfaces this
    verbatim when a desired factor > 1 goes unapplied.
    """

    choice = getattr(getattr(renderer.win, "app_settings", None), "montage_quality_policy", None)
    value = str(getattr(choice, "value", choice) or LOD_POLICY_NATIVE_ONLY)
    if value != LOD_POLICY_RESIDENT:
        return LOD_REASON_NATIVE_POLICY
    return LOD_REASON_NATIVE_POLICY


def lod_page_cache_for_renderer(renderer) -> LodPageCache:
    pyramid = getattr(renderer, "_lod_page_cache_store", None)
    if not isinstance(pyramid, LodPageCache):
        # Zero re-upload zoom cycles (ADR 0050 gate 6) require the CPU
        # pyramid to actually retain the working set across threshold
        # recrossings.  Footprint of the reference montage (272 tiles of
        # 336x336): float32 levels 1+2 are ~38 MiB, complex64 levels 1+2
        # are ~76 MiB, and the raw and FFT scenes share this one cache
        # (~114 MiB together).  The previous max(64 MiB, display/4)
        # budget evicted exactly that set, so each recrossing re-reduced
        # the level and minted a new texture identity, forcing GPU
        # re-uploads on nearly every zoom.
        budget = max(
            256 * 1024 * 1024,
            int(renderer._memory_policy().display_cache_budget_bytes) // 2,
        )
        pyramid = LodPageCache(max_bytes=budget)
        renderer._lod_page_cache_store = pyramid
    return pyramid
