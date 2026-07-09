"""Render-owned ADR 0050 montage LOD helpers.

Qt-free by construction.  The ladder owns quality progression and the
pipeline owns scheduling; this module keeps pure identity, policy, cache
lookup, and payload-construction helpers together.

Contract (the defect class behind every stall/loop this code has had is
optimistic bookkeeping — these rules are load-bearing):

* **Singleflight claims are balanced on every path.**  ``begin_pending`` is
  matched by exactly one of ``admit`` or ``end_pending`` — on error, on
  stale-before-start, on blocked admission (``start_latest`` returning
  ``None``), and on a no-longer-current session.  A leaked claim silently
  blocks that level forever.
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
from typing import NamedTuple

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
from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.pyramid import PyramidCache, PyramidLevelKey, reduce_box_mean
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.presentation import ClaimOwner, LevelPhase


PREVIEW_FLOOR_MIN_LEVEL = 4


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
    histogram = None if rendered.histogram_data is None else np.asarray(rendered.histogram_data)
    return source, histogram, texture_kind


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


def pyramid_key_for_rendered(
    rendered: RenderedTile, *, demand, level: int, semantic_source_id, shader_display: bool = True
) -> PyramidLevelKey:
    """Pyramid identity of one level of a rendered tile (ADR 0050 key contract).

    ``semantic_source_id`` is the session-owned semantic identity of the tile
    content (session key + source index).  Object identity of the source
    array must never appear in pyramid keys: rendered tiles are rebuilt
    freely across commits and sessions, and cached levels must stay
    addressable without a live ``RenderedTile`` so presentation can floor on
    resident levels for tiles that have not been rendered yet.
    """

    factor_x, factor_y = factor_xy_for_level(demand, int(level))
    return PyramidLevelKey(
        source_id=semantic_source_id,
        tile_id=int(rendered.tile.source_index),
        component=component_for_rendered(rendered, shader_display=shader_display),
        level_xy=(int(factor_x).bit_length() - 1, int(factor_y).bit_length() - 1),
    )


def pyramid_key_for(session, rendered: RenderedTile, *, demand, level: int) -> PyramidLevelKey:
    return pyramid_key_for_rendered(
        rendered,
        demand=demand,
        level=level,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
        shader_display=bool(getattr(session, "shader_display", True)),
    )


def histogram_key_for(session, rendered: RenderedTile, *, demand, level: int) -> PyramidLevelKey:
    key = pyramid_key_for(session, rendered, demand=demand, level=level)
    return histogram_key_for_level_key(key)


def histogram_key_for_level_key(key: PyramidLevelKey) -> PyramidLevelKey:
    return PyramidLevelKey(
        source_id=key.source_id,
        tile_id=key.tile_id,
        component=f"{key.component}:histogram",
        level_xy=key.level_xy,
        algo_version=key.algo_version,
    )


# --------------------------------------------------------------------------
# Policy & demand (session-side)
# --------------------------------------------------------------------------


def resident_lod_active(session) -> bool:
    return str(session.lod_policy_mode) == LOD_POLICY_RESIDENT and session.pyramid_cache is not None


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


def tile_resident_levels(session, rendered: RenderedTile, *, demand) -> tuple[int, ...]:
    """Resident acceptable levels (>0) for one rendered tile, memoized.

    During a scrub step the same scan runs from the session-wide decision,
    per-tile texture selection, and the presentation commit — several times
    per tile — and every probe rebuilds a ``PyramidLevelKey``.  The memo
    lives on the session keyed by (source index, component), guarded by the
    pyramid ``revision`` and the demand signature: a hit costs two dict
    probes and is exact because the revision bumps on every admission,
    eviction, resize, and clear.
    """

    pyramid = session.pyramid_cache
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
    levels = set(
        int(level)
        for level in demand.acceptable_levels
        if int(level) > 0
        and pyramid.peek(pyramid_key_for(session, rendered, demand=demand, level=int(level))) is not None
    )
    semantic_id = session.tile_semantic_source_id(int(rendered.tile.source_index))
    component = component_for_rendered(rendered, shader_display=bool(getattr(session, "shader_display", True)))
    coarsest = max(0, int(getattr(demand, "desired_level", 0) or 0))
    for key in pyramid.resident_keys_for(semantic_id, int(rendered.tile.source_index), component):
        if not _floor_key_presentable(session, key, pyramid):
            continue
        level = max(int(value) for value in tuple(getattr(key, "level_xy", ()) or (0,)))
        if 0 < level <= coarsest:
            levels.add(int(level))
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


class RungMaterializationRequest(NamedTuple):
    """One demanded-but-missing pyramid level for background reduction.

    ``source`` is the array the worker reduces (native texture plane or an
    already-resident finer pyramid level) and ``reduce_factor_xy`` is the
    per-axis box-mean factor relative to that source.  ``key.factor_xy``
    always stays relative to the native plane; the two differ exactly when a
    cross-level derivation was chosen (ADR 0050).

    ``chain`` is the level ladder the worker walks: ``(step_key, rel_xy)``
    pairs applied in order, each reducing the previous plane by ``rel_xy``
    (~¼ the texels of its source per doubling) and admitting under
    ``step_key`` when it is not None (``None`` = pass-through: the level is
    resident or claimed elsewhere).  The final step's key is always
    ``key``.  Empty chain = single direct reduction (legacy shape).  Every
    non-None step key holds a singleflight claim taken at plan time; all of
    them are balanced on every scheduling path.
    """

    tile_number: int
    key: PyramidLevelKey
    source: object
    reduce_factor_xy: tuple[int, int]
    cross_level: bool = False
    chain: tuple = ()


def plan_materialization(
    session,
    rendered: RenderedTile,
    *,
    demand,
    level: int,
    key: PyramidLevelKey,
    native_source: np.ndarray | None = None,
) -> RungMaterializationRequest:
    """Plan the cheapest deterministic level ladder ending at ``level``.

    ADR 0050 level-chaining: the worker starts from the finest
    already-resident coarser level (or native) and walks the missing
    acceptable levels in order, reducing each new plane from the previous
    one — every step touches ``relative_factor**2`` fewer texels than
    re-reducing the native plane — and admitting each produced level on
    the way, so the demanded level's neighbors (the hysteresis fallbacks)
    become resident for ~⅓ extra cost instead of one full native
    reduction each when the zoom crosses them later.

    Box means compose exactly only when every box is full, so a level
    joins the chain only when the native plane divides evenly by its
    per-axis factors; non-composing intermediates are left out and the
    target falls back to the single canonical native reduction when it
    cannot compose, keeping level content independent of cache state.
    Intermediate claims are taken here (GUI side, dictionary probes only)
    and balanced by every scheduling path.
    """

    if native_source is None:
        native_source, _histogram, _texture_kind = texture_source_for_rendered(
            rendered,
            shader_display=bool(getattr(session, "shader_display", True)),
        )
    tile_number = int(rendered.tile.montage_index)
    factor_x, factor_y = (int(value) for value in key.factor_xy)
    native_shape = tuple(int(value) for value in np.shape(native_source)[:2])
    pyramid = session.pyramid_cache
    if (
        pyramid is None
        or native_shape[0] % factor_y
        or native_shape[1] % factor_x
    ):
        return RungMaterializationRequest(tile_number, key, native_source, (factor_x, factor_y))
    best = None
    for candidate in demand.acceptable_levels:
        candidate = int(candidate)
        if candidate <= 0 or candidate >= int(level):
            continue
        candidate_x, candidate_y = (
            int(value) for value in factor_xy_for_level(demand, candidate)
        )
        if factor_x % candidate_x or factor_y % candidate_y:
            continue
        if best is not None and candidate <= best[0]:
            continue
        cached = pyramid.peek(pyramid_key_for(session, rendered, demand=demand, level=candidate))
        if cached is None:
            continue
        best = (candidate, cached, (candidate_x, candidate_y))
    if best is None:
        start_level, source, start_x, start_y = 0, native_source, 1, 1
    else:
        start_level, source = best[0], best[1]
        start_x, start_y = (int(value) for value in best[2])
    steps: list[tuple[PyramidLevelKey | None, tuple[int, int]]] = []
    prev_x, prev_y = start_x, start_y
    for lvl in sorted({int(candidate) for candidate in demand.acceptable_levels if start_level < int(candidate) <= int(level)}):
        if lvl == int(level):
            lvl_x, lvl_y = factor_x, factor_y
        else:
            lvl_x, lvl_y = (int(value) for value in factor_xy_for_level(demand, lvl))
        if native_shape[0] % lvl_y or native_shape[1] % lvl_x or lvl_x % prev_x or lvl_y % prev_y:
            if lvl == int(level):
                # Target does not compose through the chain: canonical
                # native reduction (never reachable when the top guard
                # passed and factors are power-of-two monotone; defensive).
                steps = []
                break
            continue
        rel = (lvl_x // prev_x, lvl_y // prev_y)
        if rel == (1, 1):
            continue
        if lvl == int(level):
            steps.append((key, rel))
        else:
            step_key = pyramid_key_for(session, rendered, demand=demand, level=lvl)
            steps.append((step_key if pyramid.begin_pending(step_key) else None, rel))
        prev_x, prev_y = lvl_x, lvl_y
    if not steps:
        return RungMaterializationRequest(tile_number, key, native_source, (factor_x, factor_y))
    claimed_intermediates = sum(1 for step_key, _rel in steps[:-1] if step_key is not None)
    if claimed_intermediates:
        session.lod_chain_planned = int(getattr(session, "lod_chain_planned", 0) or 0) + claimed_intermediates
    if best is not None:
        session.lod_cross_level_reductions += 1
    return RungMaterializationRequest(
        tile_number,
        key,
        source,
        (factor_x // start_x, factor_y // start_y),
        cross_level=best is not None,
        chain=tuple(steps),
    )


def mark_ladder_swaps_for_viewport(session) -> bool:
    """Re-evaluate LOD demand after a camera-only retarget (ADR 0050).

    Camera changes never restart evaluation. They only request presentation
    work when the current payload is too coarse for the new demand, still a
    preview, or absent. A coarser demand does not demote an already-presented
    exact/finer payload: zoom must remain a camera transform once correct
    pixels are on screen. This recomputes the decision from the current
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
    pyramid = session.pyramid_cache
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
        if presented_level <= desired and not residency_pressure_demote:
            continue
        applied = int(choose_resident_level(demand, tuple(sorted(resident))))
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
    reduced: np.ndarray | None = None,
    reduced_level: int | None = None,
    shader_display: bool = True,
) -> bool:
    """Admit the retained preview level for a freshly computed tile.

    Worker-side and opportunistic (ADR 0050 retained preview level): prefetch
    can pin a coarse copy in the shared pyramid cache, so any index ever
    computed re-presents instantly through the floor. When a finer resident
    plane is available and composes cleanly, the preview derives from it;
    otherwise it reduces the native plane once.
    """

    if preview_pyramid is None or int(preview_level) <= 0:
        return False
    shader_display = bool(shader_display)
    component = component_for_rendered(rendered, shader_display=shader_display)
    level = int(preview_level)
    key = PyramidLevelKey(
        source_id=semantic_source_id,
        tile_id=int(rendered.tile.source_index),
        component=component,
        level_xy=(level, level),
    )
    if not preview_pyramid.begin_pending(key):
        return False
    try:
        factor = 1 << level
        source, histogram, kind = texture_source_for_rendered(rendered, shader_display=shader_display)
        if texture_requires_display_histogram(source, kind) and histogram is None:
            preview_pyramid.end_pending(key)
            return False
        if reduced is not None and reduced_level is not None and int(reduced_level) == level:
            # The ingest reduction already produced exactly the preview
            # level: pin the same plane, zero extra reduction or copy.
            preview_pyramid.admit(key, np.asarray(reduced))
        elif (
            reduced is not None
            and reduced_level is not None
            and 0 < int(reduced_level) < level
            and all(int(edge) % (factor >> int(reduced_level)) == 0 for edge in np.shape(reduced)[:2])
        ):
            relative = factor >> int(reduced_level)
            preview_pyramid.admit(key, reduce_box_mean(np.asarray(reduced), (relative, relative)))
        else:
            preview_pyramid.admit(key, reduce_box_mean(source, (factor, factor)))
        if histogram is not None:
            hist_key = histogram_key_for_level_key(key)
            if preview_pyramid.begin_pending(hist_key):
                try:
                    preview_pyramid.admit(hist_key, reduce_box_mean(histogram, hist_key.factor_xy))
                except Exception:
                    preview_pyramid.end_pending(hist_key)
    except Exception:
        preview_pyramid.end_pending(key)
        return False
    return True


# --------------------------------------------------------------------------
# Presentation floor (session-side)
# --------------------------------------------------------------------------


def floor_component_tags(session) -> tuple[str, ...]:
    """Component tags a floor probe may find for this session's tiles."""

    if bool(getattr(session, "shader_display", False)):
        # VisPy/shader presentations own complex windowing on the GPU.  A
        # retained CPU-windowed RGB preview is not compatible evidence for
        # that path, even if it shares the semantic tile source.
        return (str(TexturePlaneKind.COMPLEX_RG32F.value), "scalar")
    return (str(TexturePlaneKind.RGB8.value), "scalar", str(TexturePlaneKind.COMPLEX_RG32F.value))


def _floor_key_presentable(session, key, cache) -> bool:
    """A floor level is usable only if it can actually be drawn.

    The plane must be resident and, for texture kinds that need a display
    histogram (complex / RGB), a matching histogram must be resident too.
    Returning a level whose histogram is missing makes ensure_floor_payloads
    skip the tile — leaving a hole instead of falling back to a coarser level
    that IS presentable (the single-tile complex-floor hole after a scroll).
    """

    if cache is None:
        return False
    plane = cache.peek(key)
    if plane is None:
        return False
    texture_kind = None
    metadata_fn = getattr(session, "preview_floor_metadata", None)
    if callable(metadata_fn):
        texture_kind = getattr(metadata_fn(key), "texture_kind", None)
    if texture_kind is None:
        texture_kind = floor_texture_kind(key.component)
    if texture_requires_display_histogram(plane, texture_kind):
        histogram = cache.peek(histogram_key_for_level_key(key))
        if not display_histogram_matches_texture(histogram, plane):
            return False
    return True


def best_floor_key(session, source_index: int, *, tile_number: int | None = None):
    """Best *presentable* resident pyramid key: nearest demand, finer ties."""

    pyramid = session.pyramid_cache
    demand = session.lod_policy_decision.demand
    desired = int(demand.desired_level)
    semantic_id = session.tile_semantic_source_id(int(source_index))
    if tile_number is not None:
        rec = session.lifecycle.peek(int(tile_number))
        if rec is not None:
            best_preview = None
            for key, entry in rec.levels.items():
                if entry.owner is not ClaimOwner.PREVIEW or entry.phase is not LevelPhase.RESIDENT:
                    continue
                if getattr(key, "source_id", None) != semantic_id or int(getattr(key, "tile_id", -1)) != int(source_index):
                    continue
                cache = session.pyramid_cache
                if not _floor_key_presentable(session, key, cache):
                    continue
                level = max(int(key.level_xy[0]), int(key.level_xy[1]))
                rank = _resident_floor_rank(level, desired)
                if best_preview is None or rank < best_preview[0]:
                    best_preview = (rank, key, level, cache)
            if best_preview is not None:
                return (best_preview[1], best_preview[2], best_preview[3])
    if pyramid is not None:
        best = None
        for component in floor_component_tags(session):
            for key in pyramid.resident_keys_for(semantic_id, int(source_index), component):
                if not _floor_key_presentable(session, key, pyramid):
                    continue
                level = max(int(key.level_xy[0]), int(key.level_xy[1]))
                rank = _resident_floor_rank(level, desired)
                if best is None or rank < best[0]:
                    best = (rank, key, level)
            if best is not None:
                break
        if best is not None:
            return (best[1], best[2], pyramid)
    preview = session.pyramid_cache
    level = int(session.lod_preview_level)
    if preview is None or level <= 0:
        return None
    for component in floor_component_tags(session):
        key = PyramidLevelKey(
            source_id=semantic_id,
            tile_id=int(source_index),
            component=component,
            level_xy=(level, level),
        )
        if preview.peek(key) is not None:
            return (key, level, preview)
    return None


def _resident_floor_rank(level: int, desired: int) -> tuple[int, int]:
    level = int(level)
    desired = int(desired)
    if level <= desired:
        return (0, level)
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
    if payload is not None and str(getattr(payload, "quality", "exact")) != "preview":
        return False
    best = best_floor_key(session, int(tile.source_index), tile_number=int(tile_number))
    if best is None:
        return False
    if payload is None:
        return True
    presented = int(getattr(getattr(payload, "lod", None), "level", 0) or 0)
    return int(best[1]) != presented


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
    cache = session.pyramid_cache
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
        if existing is not None and str(getattr(existing, "quality", "exact")) != "preview":
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
        if existing is not None:
            presented = int(getattr(getattr(existing, "lod", None), "level", 0) or 0)
            if presented == int(level):
                continue
        plane = owning_cache.peek(key)
        if plane is None:
            continue
        histogram = owning_cache.peek(histogram_key_for_level_key(key))
        metadata = None
        metadata_fn = getattr(session, "preview_floor_metadata", None)
        if callable(metadata_fn):
            metadata = metadata_fn(key)
        texture_kind = getattr(metadata, "texture_kind", None)
        if texture_kind is None:
            texture_kind = floor_texture_kind(key.component)
        if texture_requires_display_histogram(plane, texture_kind) and not display_histogram_matches_texture(histogram, plane):
            continue
        factor_x = 1 << int(key.level_xy[0])
        factor_y = 1 << int(key.level_xy[1])
        tile_shape = tuple(int(value) for value in session.plan.tile_shape)
        lod = LodInfo(
            level=level,
            factor=max(factor_x, factor_y),
            source_shape=tile_shape,
            texture_shape=tuple(int(value) for value in np.shape(plane)[:2]),
            gutter=0,
        )
        payload = DisplayTilePayload(
            tile_number=tile_number,
            source_index=source_index,
            image=np.asarray(plane),
            histogram_data=None if histogram is None else np.asarray(histogram),
            source_id=(*semantic_id, "floor", str(key.component), key.level_xy),
            texture_data=np.asarray(plane),
            texture_kind=texture_kind,
            lod=lod,
            quality="preview",
            shader_mapping=getattr(metadata, "shader_mapping", None),
            level_data=getattr(metadata, "level_data", None),
            level_stats=getattr(metadata, "level_stats", None),
        )
        session.display_tile_payloads[tile_number] = payload
        lifecycle = getattr(session, "lifecycle", None)
        if lifecycle is not None and hasattr(lifecycle, "remember_presentable"):
            lifecycle.remember_presentable(tile_number, payload)
        session.pending_payload_upserts[tile_number] = None
        session.lod_floor_presentations = int(getattr(session, "lod_floor_presentations", 0) or 0) + 1
        built += 1


def floor_texture_kind(component: object) -> TexturePlaneKind:
    value = str(getattr(component, "value", component))
    if value == "scalar":
        return TexturePlaneKind.SCALAR_R32F
    return TexturePlaneKind(value)


# --------------------------------------------------------------------------
# Texture selection (session-side)
# --------------------------------------------------------------------------


def _reduced_histogram_for_texture(
    histogram: np.ndarray, texture_shape: tuple[int, int], factor_xy: tuple[int, int]
) -> np.ndarray | None:
    """Reduce a native magnitude histogram to a reduced texture's shape.

    Box-mean by the level factor (the same reduction the texture used), then
    verify the shape matches — returns None when it cannot compose so the
    caller can fall back rather than mis-map magnitude onto pixels.
    """

    factor_x, factor_y = (max(1, int(factor_xy[0])), max(1, int(factor_xy[1])))
    native_shape = tuple(int(value) for value in np.shape(histogram)[:2])
    if native_shape[0] % factor_y or native_shape[1] % factor_x:
        return None
    reduced = np.asarray(reduce_box_mean(np.asarray(histogram), (factor_x, factor_y)))
    if tuple(reduced.shape[:2]) != tuple(int(value) for value in texture_shape):
        return None
    return reduced


def resident_texture_for_rendered_tile(
    session,
    rendered: RenderedTile,
    *,
    source: np.ndarray,
    histogram: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, LodInfo]:
    """Cache-lookup-only level application (no reduction in this path).

    A demanded level that is not cached is not resident; the tile falls
    back to the nearest resident/native level and the missing level is
    recorded once (singleflight) for the renderer to materialize in the
    background.
    """

    source_shape = tuple(int(value) for value in source.shape[:2])
    native_lod = LodInfo(level=0, factor=1, source_shape=source_shape, texture_shape=source_shape, gutter=0)
    demand = session.lod_policy_decision.demand
    pyramid = session.pyramid_cache
    resident_levels = tile_resident_levels(session, rendered, demand=demand)
    desired = int(demand.desired_level)
    # Missing demanded levels are planned by LodLadder. Texture selection is
    # lookup-only so a presentation build cannot create a hidden scheduling
    # queue with its own wakeup rules.
    applied = choose_resident_level(demand, resident_levels)
    if applied <= 0:
        return source, histogram, native_lod
    texture = pyramid.lookup(pyramid_key_for(session, rendered, demand=demand, level=applied))
    if texture is None:
        return source, histogram, native_lod
    texture = np.asarray(texture)
    factor_xy = factor_xy_for_level(demand, applied)
    texture_histogram = histogram
    if histogram is not None and tuple(np.shape(histogram)[:2]) != tuple(np.shape(texture)[:2]):
        hist_key = histogram_key_for(session, rendered, demand=demand, level=applied)
        cached_histogram = pyramid.lookup(hist_key)
        if cached_histogram is not None:
            texture_histogram = np.asarray(cached_histogram)
        elif texture.ndim == 3:
            # RGB/complex base ONLY: the magnitude histogram is NOT optional
            # for a reduced RGB base — without it the phase-hue plane renders
            # at full brightness with no magnitude (field defect 2026-07 —
            # resident-LOD complex tiles showed phase-only, levels had no
            # effect). The on-demand level worker reduces only the texture
            # plane; reduce the native magnitude to match here and cache it so
            # subsequent reads see a level-consistent magnitude. Scalar tiles
            # keep the prior behavior (their histogram is level-stat data, not
            # a brightness channel, and is applied through a different path).
            reduced_histogram = _reduced_histogram_for_texture(
                np.asarray(histogram), tuple(np.shape(texture)[:2]), hist_key.factor_xy
            )
            if reduced_histogram is not None:
                pyramid.admit(hist_key, reduced_histogram)
                texture_histogram = reduced_histogram
            else:
                texture_histogram = np.asarray(histogram)
        else:
            texture_histogram = None
    if texture_requires_display_histogram(texture, getattr(rendered, "texture_kind", None)) and not display_histogram_matches_texture(
        texture_histogram, texture
    ):
        return source, histogram, native_lod
    lod = LodInfo(
        level=applied,
        factor=max(int(factor_xy[0]), int(factor_xy[1])),
        source_shape=source_shape,
        texture_shape=tuple(int(value) for value in texture.shape[:2]),
        gutter=0,
    )
    return texture, texture_histogram, lod


def _release_chain_claims(pyramid, chain) -> bool:
    """End every unadmitted claim in a chain; True when any step is resident.

    The single place claim balancing happens for scheduling paths that did
    not run (blocked admission, stale session, supersession) or failed:
    admitted levels are results and stay; everything else releases its
    singleflight claim so the next refresh can re-request it.
    """

    admitted = False
    for step_key, _rel in chain:
        if step_key is None:
            continue
        if pyramid.peek(step_key) is not None:
            admitted = True
        else:
            pyramid.end_pending(step_key)
    return admitted


def _apply_release_effects(pyramid, effects) -> int:
    if pyramid is None:
        return 0
    released = 0
    for effect in tuple(effects or ()):
        pyramid.end_pending(effect.level_key)
        released += 1
    return released


def _finish_request_claims(session, request, pyramid) -> bool:
    """Mark a drained request's claims resident or released from pyramid truth."""

    admitted = False
    for step_key, _rel in _request_chain(request):
        if step_key is None:
            continue
        if pyramid is not None and pyramid.peek(step_key) is not None:
            admitted = True
            session.lifecycle.level_resident(int(request.tile_number), step_key)
        else:
            _apply_release_effects(
                pyramid,
                session.lifecycle.level_declined(int(request.tile_number), step_key),
            )
    return admitted


def _release_request_claims(session, request, pyramid) -> bool:
    """Release one request through the lifecycle machine and pyramid cache."""

    admitted = False
    for step_key, _rel in _request_chain(request):
        if step_key is not None and pyramid is not None and pyramid.peek(step_key) is not None:
            admitted = True
    # `pending_rung_materializations` is always the lifecycle-backed view (ADR 0051
    # P3); the pre-P3 chain fallback was deleted in the redesign.
    _apply_release_effects(pyramid, session.pending_rung_materializations.release(request))
    if admitted:
        for step_key, _rel in _request_chain(request):
            if step_key is not None and pyramid is not None and pyramid.peek(step_key) is not None:
                session.lifecycle.level_resident(int(request.tile_number), step_key)
    return admitted


def _request_chain(request) -> tuple:
    chain = tuple(getattr(request, "chain", ()) or ())
    if chain:
        return chain
    key = request[1]
    reduce_factor_xy = tuple(int(value) for value in (request[3] if len(request) > 3 else key.factor_xy))
    return ((key, reduce_factor_xy),)


def release_session_claims(session) -> int:
    """Release every pyramid claim still held by a session's lifecycle records.

    A session can die between planning (claims taken in refresh/build) and
    scheduling (claims handed to work items) — slice scrubbing replaces
    sessions faster than the drain runs.  The pyramid is renderer-shared and
    its keys are semantic, so a claim leaked by a dead session blocks the
    SAME level when the user scrubs back to that slice: ``begin_pending``
    never succeeds again and the tile presents the wrong LOD forever.  Call
    on every session replacement; in-flight scheduled items are not touched
    (their own release paths balance them).
    """

    if session is None:
        return 0
    pyramid = getattr(session, "pyramid_cache", None)
    lifecycle = getattr(session, "lifecycle", None)
    if lifecycle is not None:
        released = 0
        for effect in lifecycle.session_replaced():
            cache = session.pyramid_cache if effect.owner is ClaimOwner.PREVIEW else pyramid
            if cache is not None:
                cache.end_pending(effect.level_key)
                released += 1
            if effect.owner is ClaimOwner.PREVIEW:
                getattr(session, "lod_preview_metadata", {}).pop(effect.level_key, None)
        return released
    requests = list(getattr(session, "pending_rung_materializations", ()) or ())
    if pyramid is None:
        return 0
    released = 0
    for request in requests:
        for step_key, _rel in _request_chain(request):
            if step_key is not None:
                pyramid.end_pending(step_key)
                released += 1
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


def pyramid_cache_for_renderer(renderer) -> PyramidCache:
    pyramid = getattr(renderer, "_montage_pyramid_cache_store", None)
    if not isinstance(pyramid, PyramidCache):
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
        pyramid = PyramidCache(max_bytes=budget)
        renderer._montage_pyramid_cache_store = pyramid
    return pyramid
