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
    resident_lod_policy,
    select_lod_demand,
)
from arrayscope.display.montage import RenderedTile
from arrayscope.display.model.frame import DisplayTilePayload, PageBackedPresentation
from arrayscope.display.pyramid import (
    LodPageCache,
    LodPagePlan,
    MaterializedLodPage,
    materialize_lod_page,
    plan_source_grid_pages,
)
from arrayscope.display.shader_mapping import ShaderComponent, ShaderDisplayMode, TexturePlaneKind
from arrayscope.gpu import DataChunkKey
from arrayscope.gpu.keys import (
    COMPLEX_RG32F,
    REDUCER_MEAN,
    REDUCER_MEAN_ABS,
    REDUCER_PHASE_VECTOR,
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


def page_set_key_for(session, rendered: RenderedTile, *, demand, level: int) -> LodPageSetKey:
    plans = page_plans_for_rendered(session, rendered, demand=demand, level=level)
    factor_x, factor_y = factor_xy_for_level(demand, int(level))
    return LodPageSetKey(
        source_id=session.tile_semantic_source_id(rendered.tile.source_index),
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
        if display_mode == ShaderDisplayMode.PHASE_COLOR:
            return REDUCER_PHASE_VECTOR, "complex64", COMPLEX_RG32F
        if component == ShaderComponent.ABS:
            return REDUCER_MEAN_ABS, "float32", SCALAR_R32F
        if component in (ShaderComponent.ANGLE, ShaderComponent.COMPLEX_PHASE):
            return REDUCER_PHASE_VECTOR, "complex64", COMPLEX_RG32F
        return REDUCER_MEAN, "complex64", COMPLEX_RG32F
    if source.ndim != 2:
        raise ValueError("canonical live LOD pages require scalar or complex source values")
    return REDUCER_MEAN, "float32", SCALAR_R32F


def _page_source_origin(session, source: np.ndarray) -> tuple[int, int]:
    anchor_fn = getattr(session, "_payload_source_anchor", None)
    anchor = anchor_fn(tuple(np.shape(source)[:2])) if callable(anchor_fn) else None
    if anchor is None:
        return (0, 0)
    return (int(anchor.source_rect[0]), int(anchor.source_rect[2]))


def page_plans_for_rendered(session, rendered, *, demand, level: int, native_source=None):
    if native_source is None:
        native_source = canonical_value_source_for_rendered(
            rendered, shader_display=bool(getattr(session, "shader_display", True))
        )
    source = np.asarray(native_source)
    reducer, dtype, representation = _reducer_format_for_rendered(rendered, source)
    factor_x, factor_y = factor_xy_for_level(demand, int(level))
    origin_y, origin_x = _page_source_origin(session, source)
    height, width = (int(value) for value in source.shape[:2])
    anchor_fn = getattr(session, "_payload_source_anchor", None)
    anchor = anchor_fn((height, width)) if callable(anchor_fn) else None
    content_key = (
        anchor.content_key
        if anchor is not None
        else (
            "src-anchored",
            session.tile_semantic_source_id(rendered.tile.source_index),
            ("display-plane",),
        )
    )
    return plan_source_grid_pages(
        content_key=content_key,
        valid_source_rect_yx=(origin_y, origin_y + height, origin_x, origin_x + width),
        reduction_yx=(int(factor_y).bit_length() - 1, int(factor_x).bit_length() - 1),
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
        elif channel in {"angle", "complex"}:
            reducer, planned_dtype, representation = REDUCER_PHASE_VECTOR, "complex64", COMPLEX_RG32F
        else:
            reducer, planned_dtype, representation = REDUCER_MEAN, "complex64", COMPLEX_RG32F
    else:
        reducer, planned_dtype, representation = REDUCER_MEAN, "float32", SCALAR_R32F
    factor_x, factor_y = factor_xy_for_level(demand, int(level))
    height, width = (int(value) for value in session.plan.tile_shape[:2])
    semantic_source_id = session.tile_semantic_source_id(int(tile.source_index))
    plans = plan_source_grid_pages(
        content_key=("src-anchored", semantic_source_id, ("display-plane",)),
        valid_source_rect_yx=(0, height, 0, width),
        reduction_yx=(int(factor_y).bit_length() - 1, int(factor_x).bit_length() - 1),
        stored_page_shape=(256, 256),
        dtype=planned_dtype,
        representation=representation,
        reducer=reducer,
    )
    return LodPageSetKey(
        source_id=semantic_source_id,
        tile_id=int(tile.source_index),
        level_xy=(int(factor_x).bit_length() - 1, int(factor_y).bit_length() - 1),
        reducer=reducer,
        plans=plans,
    )
# --------------------------------------------------------------------------
# Policy & demand (session-side)
# --------------------------------------------------------------------------


def resident_lod_active(session) -> bool:
    return str(session.lod_policy_mode) == LOD_POLICY_RESIDENT and session.lod_page_cache is not None


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
        and all(cache.resolve(page_key) is not None for page_key in key.page_keys)
    )


def _page_set_exact(cache: LodPageCache | None, key: LodPageSetKey) -> bool:
    if not isinstance(cache, LodPageCache):
        return False
    resolutions = tuple(cache.resolve(page_key) for page_key in key.page_keys)
    return bool(
        all(item is not None for item in resolutions)
        and all(item.actual_key == item.target_key for item in resolutions)
    )


def _page_set_materialized_pages(
    cache: LodPageCache | None,
    key: LodPageSetKey,
) -> tuple[MaterializedLodPage, ...] | None:
    if not isinstance(cache, LodPageCache):
        return None
    return cache.resolved_pages(key.plans)


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
    """(level, factor, (factor_x, factor_y)) shown by the plurality of tiles.

    The session-wide policy decision only claims a level once every
    rendered tile can present it, which reads as "native" while any tile
    is still streaming.  Diagnostics report what the committed
    presentation actually shows, so the JSONL A/B stays truthful during
    partial residency (ADR 0050).  Ties prefer the finer level.
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
    counts: dict[int, int] = {}
    samples: dict[int, object] = {}
    for payload in payloads.values():
        lod = getattr(payload, "lod", None)
        level = int(getattr(lod, "level", 0) or 0)
        counts[level] = counts.get(level, 0) + 1
        samples.setdefault(level, lod)
    level = min(counts, key=lambda candidate: (-counts[candidate], candidate))
    lod = samples.get(level)
    if lod is None or level <= 0:
        return (0, 1, (1, 1))
    source_shape = tuple(getattr(lod, "source_shape", (1, 1)))
    texture_shape = tuple(getattr(lod, "texture_shape", (1, 1)))
    factor_y = max(1, round(int(source_shape[0]) / max(1, int(texture_shape[0]))))
    factor_x = max(1, round(int(source_shape[1]) / max(1, int(texture_shape[1]))))
    return (int(level), int(getattr(lod, "factor", 1) or 1), (int(factor_x), int(factor_y)))


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


def native_missing_tile_queue_required(lod_policy_mode: str, demand) -> bool:
    """Return whether a missing tile belongs in the native evaluation queue.

    Resident LOD separates first correct pixels from exact/native refinement:
    if the viewport demands a reduced display level, cold tiles should be
    filled by ladder/pipeline rungs instead of entering the native target-tile
    queue as the first response. Native-only policy, invalid demand, and
    native-scale resident demand still require native evaluation.
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
class RungMaterializationRequest:
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
) -> RungMaterializationRequest:
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
    return RungMaterializationRequest(
        tile_number=tile_number,
        key=key,
        source=native_source,
        source_origin_yx=_page_source_origin(session, np.asarray(native_source)),
        plans=plans,
        claimed_plans=claimed,
        owner=owner,
    )


def mark_ladder_swaps_for_viewport(session) -> bool:
    """Re-evaluate LOD demand after a camera-only retarget (ADR 0050).

    Camera changes never restart evaluation. They only request presentation
    work when the current payload is too coarse for the new demand, still a
    preview, absent, or an already-resident demanded level can consolidate
    active atlas classes without an upload. This recomputes the decision from the current
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
    demand = session.lod_policy_decision.demand
    pyramid = session.lod_page_cache
    desired = int(demand.desired_level)
    commit_needed = False
    visible_by_number = {int(t.montage_index): t for t in tuple(session.visible_tiles)}
    residency_pressure_demote = _visible_residency_pressure_demands_demote(session, demand)
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
        payload = session.display_tile_payloads.get(int(tile_number))
        presented_level = int(getattr(getattr(payload, "lod", None), "level", 0) or 0)
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
        consolidate_to_resident_demand = bool(
            desired > presented_level and desired in resident
        )
        if (
            presented_level <= desired
            and not residency_pressure_demote
            and not consolidate_to_resident_demand
        ):
            continue
        applied = (
            desired
            if consolidate_to_resident_demand
            else int(choose_resident_level(demand, tuple(sorted(resident))))
        )
        if presented_level != applied:
            session.dirty_payloads[int(tile_number)] = None
            commit_needed = True
    return commit_needed


def _visible_residency_pressure_demands_demote(session, demand) -> bool:
    """Return True when exact/finer active payloads exceed GPU residency budget.

    Demand by itself is not pressure: a fully resident native montage should
    zoom as a camera transform.  This predicate is the explicit escape hatch
    for wider zoomed-out views that reveal enough tiles that keeping active
    exact/finer payloads would exceed the presentation residency budget.
    """

    budget = int(getattr(session, "tile_residency_budget_bytes", 0) or 0)
    desired = int(getattr(demand, "desired_level", 0) or 0)
    if budget <= 0 or desired <= 0:
        return False
    total = 0
    payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    for tile_number in tuple(getattr(session, "visible_tile_numbers", ()) or ()):
        payload = payloads.get(int(tile_number))
        if payload is None:
            continue
        level = int(getattr(getattr(payload, "lod", None), "level", 0) or 0)
        if level > desired:
            continue
        total += int(getattr(payload, "nbytes", 0) or 0)
        if total > budget:
            return True
    return False


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
    if _page_set_complete(preview_pyramid, key):
        return None
    owner = ("retained-preview-pages", semantic_source_id, key)
    claimed = preview_pyramid.claim_plans(plans, owner)
    if not claimed:
        return None
    try:
        for plan in claimed:
            page = materialize_lod_page(source, source_origin_yx=(0, 0), plan=plan)
            preview_pyramid.admit_as(plan.key, page, owner=owner)
    except Exception:
        preview_pyramid.release_owner_claims(owner)
        return None
    if not _page_set_complete(preview_pyramid, key):
        return None
    return key


# --------------------------------------------------------------------------
# Presentation floor (session-side)
# --------------------------------------------------------------------------


def _floor_key_presentable(session, key, cache) -> bool:
    """A floor is presentable only with complete exact/coarse page coverage."""

    return isinstance(key, LodPageSetKey) and _page_set_complete(cache, key)


def best_floor_key(session, source_index: int, *, tile_number: int | None = None):
    """Best *presentable* resident pyramid key: nearest demand, finer ties."""

    pyramid = session.lod_page_cache
    demand = session.lod_policy_decision.demand
    desired = int(demand.desired_level)
    semantic_id = session.tile_semantic_source_id(int(source_index))
    candidates = []
    records = () if tile_number is None else (session.lifecycle.peek(int(tile_number)),)
    for rec in records:
        if rec is None:
            continue
        for key, entry in rec.levels.items():
            if entry.phase is not LevelPhase.RESIDENT or not isinstance(key, LodPageSetKey):
                continue
            if key.source_id != semantic_id or int(key.tile_id) != int(source_index):
                continue
            if not _floor_key_presentable(session, key, pyramid):
                continue
            level = key.level
            candidates.append((_resident_floor_rank(level, desired), key, level, pyramid))
    if not candidates:
        tile = next(
            (
                value
                for value in tuple(getattr(session.plan, "tiles", ()) or ())
                if int(value.source_index) == int(source_index)
                and (tile_number is None or int(value.montage_index) == int(tile_number))
            ),
            None,
        )
        if tile is not None:
            levels = tuple(
                dict.fromkeys(
                    (
                        int(getattr(session, "lod_preview_level", 0) or 0),
                        desired,
                        *tuple(int(value) for value in demand.acceptable_levels),
                    )
                )
            )
            for level in levels:
                if level <= 0:
                    continue
                key = page_set_key_for_tile(session, tile, demand=demand, level=level)
                if _floor_key_presentable(session, key, pyramid):
                    candidates.append(
                        (_resident_floor_rank(level, desired), key, level, pyramid)
                    )
    if not candidates:
        return None
    _rank, key, level, owning_cache = min(candidates, key=lambda item: item[0])
    return (key, level, owning_cache)


def _resident_floor_rank(level: int, desired: int) -> tuple[int, int]:
    level = int(level)
    desired = int(desired)
    if level <= desired:
        return (0, desired - level)
    return (1, level - desired)


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
    presented = int(getattr(getattr(payload, "lod", None), "level", 0) or 0)
    best_level = int(best[1])
    metadata_fn = getattr(session, "preview_floor_metadata", None)
    metadata = metadata_fn(best[0]) if callable(metadata_fn) else None
    best_quality = str(getattr(metadata, "quality", "preview") or "preview")
    if best_level == presented:
        best_pages = _page_set_materialized_pages(best[2], best[0]) or ()
        current_backing = getattr(payload, "page_backing", None)
        current_keys = tuple(
            page.key for page in tuple(getattr(current_backing, "materialized_pages", ()) or ())
        )
        if current_keys != tuple(page.key for page in best_pages):
            return True
    if str(getattr(payload, "quality", "exact")) != "preview":
        # "Exact" describes the reduced target's semantic quality, not native
        # resolution. A level-5 exact target must still swap to an already
        # resident level-1 exact target after zooming in. Refusing every exact
        # floor left the ladder with zero work (target resident) while the
        # backend stayed permanently coarse.
        desired = int(session.lod_policy_decision.demand.desired_level)
        return bool(presented > desired and best_level < presented)
    return bool(
        best_level != presented
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
        key, level, owning_cache = best
        metadata = None
        metadata_fn = getattr(session, "preview_floor_metadata", None)
        if callable(metadata_fn):
            metadata = metadata_fn(key)
        best_quality = str(getattr(metadata, "quality", "preview") or "preview")
        if existing is not None:
            presented = int(getattr(getattr(existing, "lod", None), "level", 0) or 0)
            best_pages = _page_set_materialized_pages(owning_cache, key) or ()
            existing_backing = getattr(existing, "page_backing", None)
            existing_keys = tuple(
                page.key
                for page in tuple(getattr(existing_backing, "materialized_pages", ()) or ())
            )
            if (
                presented == int(level)
                and str(getattr(existing, "quality", "preview") or "preview") == best_quality
                and existing_keys == tuple(page.key for page in best_pages)
            ):
                continue
            if str(getattr(existing, "quality", "exact")) != "preview":
                desired = int(session.lod_policy_decision.demand.desired_level)
                if presented <= desired or int(level) >= presented:
                    continue
        pages = _page_set_materialized_pages(owning_cache, key)
        if not pages:
            continue
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
        factor_x = 1 << int(key.level_xy[0])
        factor_y = 1 << int(key.level_xy[1])
        tile_shape = tuple(int(value) for value in session.plan.tile_shape)
        lod = LodInfo(
            level=level,
            factor=max(factor_x, factor_y),
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
        if best_quality == "exact":
            anchor_fn = getattr(session, "_payload_source_anchor", None)
            if callable(anchor_fn):
                source_anchor = anchor_fn(tile_shape)
        payload = DisplayTilePayload(
            tile_number=tile_number,
            source_index=source_index,
            image=np.asarray(pages[0].values),
            histogram_data=None,
            source_id=(*semantic_id, "floor", key.reducer, key.level_xy),
            texture_data=np.asarray(pages[0].values),
            texture_kind=texture_kind,
            lod=lod,
            quality=str(getattr(metadata, "quality", "preview") or "preview"),
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
                requested_lod=lod,
            ),
            level_data=getattr(metadata, "level_data", None),
            level_stats=getattr(metadata, "level_stats", None),
            tile_identity=session.tile_payload_identity(
                session.plan.tiles[int(tile_number)],
                texture_data=np.asarray(pages[0].values),
                texture_kind=texture_kind,
                shader_mapping=shader_mapping,
                lod=lod,
                quality=str(getattr(metadata, "quality", "preview") or "preview"),
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
    demand = session.lod_policy_decision.demand
    pyramid = session.lod_page_cache
    resident_levels = tile_resident_levels(session, rendered, demand=demand)
    # Missing demanded levels are planned by LodLadder. Texture selection is
    # lookup-only so a presentation build cannot create a hidden scheduling
    # queue with its own wakeup rules.
    applied = choose_resident_level(demand, resident_levels)
    if applied <= 0:
        return source, histogram, native_lod, None, getattr(rendered, "texture_kind", None)
    key = page_set_key_for(session, rendered, demand=demand, level=applied)
    pages = _page_set_materialized_pages(pyramid, key)
    if not pages:
        return source, histogram, native_lod, None, getattr(rendered, "texture_kind", None)
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


def _release_chain_claims(pyramid, chain) -> bool:
    """End every unadmitted claim in a chain; True when any step is resident.

    The single place claim balancing happens for scheduling paths that did
    not run (blocked admission, stale session, supersession) or failed:
    admitted levels are results and stay; everything else releases its
    singleflight claim so the next refresh can re-request it.
    """

    return any(
        isinstance(step_key, LodPageSetKey) and _page_set_complete(pyramid, step_key)
        for step_key, _rel in tuple(chain or ())
    )


def _apply_release_effects(pyramid, effects) -> int:
    # CPU page claims are owner-scoped and released from the request, never
    # inferred from lifecycle's tile/rung key.
    return len(tuple(effects or ())) if pyramid is not None else 0


def _finish_request_claims(session, request, pyramid) -> bool:
    """Mark a drained request's claims resident or released from pyramid truth."""

    if isinstance(pyramid, LodPageCache):
        pyramid.release_owner_claims(request.owner)
    admitted = _page_set_complete(pyramid, request.key)
    if admitted:
        session.lifecycle.level_resident(int(request.tile_number), request.key)
    else:
        session.lifecycle.level_declined(int(request.tile_number), request.key)
    return admitted


def _release_request_claims(session, request, pyramid) -> bool:
    """Release one request through the lifecycle machine and pyramid cache."""

    admitted = _page_set_complete(pyramid, request.key)
    if isinstance(pyramid, LodPageCache):
        pyramid.release_owner_claims(request.owner)
    # `pending_rung_materializations` is always the lifecycle-backed view (ADR 0051
    # P3); the pre-P3 chain fallback was deleted in the redesign.
    _apply_release_effects(pyramid, session.pending_rung_materializations.release(request))
    if admitted:
        session.lifecycle.level_resident(int(request.tile_number), request.key)
    return admitted


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
    pyramid = getattr(renderer, "_montage_lod_page_cache_store", None)
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
        renderer._montage_lod_page_cache_store = pyramid
    return pyramid
