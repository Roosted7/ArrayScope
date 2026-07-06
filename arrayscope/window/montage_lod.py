"""One home for ADR 0050 montage LOD: policy, identity, materialization, floor.

Qt-free by construction (imported by :mod:`montage_session`).  Session-side
functions take the ``MontageRenderSession`` as their first argument;
renderer-side functions take the ``FrameRenderMixin`` host.  The original
methods remain as delegates so call sites and test names are unchanged.

Contract (the defect class behind every stall/loop this code has had is
optimistic bookkeeping — these rules are load-bearing):

* **Singleflight claims are balanced on every path.**  ``begin_pending`` is
  matched by exactly one of ``admit`` or ``end_pending`` — on error, on
  stale-before-start, on blocked admission (``start_latest`` returning
  ``None``), and on a no-longer-current session.  A leaked claim silently
  blocks that level forever.
* **Supersession cancels work, never results.**  A superseded item that
  already admitted its level must still notify (``on_level_ready``), or the
  tile presents an outdated level until an unrelated event.
* **Bookkeeping is acknowledge-driven.**  Nothing here marks a tile clean or
  a request satisfied because work was *submitted*; only completions and
  acknowledged commits advance state.  Dirty marks for non-active tiles are
  emitted once and parked by the session on declined acknowledgement.
* **The floor is presentation-only.**  Floor payloads carry
  ``quality="preview"``: they draw pixels but never satisfy semantic reads,
  histograms, level convergence, or LOD convergence.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np

from arrayscope.core.scheduler import EvalPriority
from arrayscope.core.work_graph import WorkItem, WorkLane
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.lod import (
    LOD_POLICY_NATIVE_ONLY,
    LOD_REASON_BACKEND_ADOPTION_PENDING,
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
    return str(session.lod_policy_mode) == LOD_POLICY_RESIDENT and session.lod_pyramid is not None


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

    pyramid = session.lod_pyramid
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
    levels = tuple(
        int(level)
        for level in demand.acceptable_levels
        if int(level) > 0
        and pyramid.peek(pyramid_key_for(session, rendered, demand=demand, level=int(level))) is not None
    )
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


# --------------------------------------------------------------------------
# Materialization planning (session-side)
# --------------------------------------------------------------------------


class LodMaterializationRequest(NamedTuple):
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
) -> LodMaterializationRequest:
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
    pyramid = session.lod_pyramid
    if (
        pyramid is None
        or native_shape[0] % factor_y
        or native_shape[1] % factor_x
    ):
        return LodMaterializationRequest(tile_number, key, native_source, (factor_x, factor_y))
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
        return LodMaterializationRequest(tile_number, key, native_source, (factor_x, factor_y))
    claimed_intermediates = sum(1 for step_key, _rel in steps[:-1] if step_key is not None)
    if claimed_intermediates:
        session.lod_chain_planned = int(getattr(session, "lod_chain_planned", 0) or 0) + claimed_intermediates
    if best is not None:
        session.lod_cross_level_reductions += 1
    return LodMaterializationRequest(
        tile_number,
        key,
        source,
        (factor_x // start_x, factor_y // start_y),
        cross_level=best is not None,
        chain=tuple(steps),
    )


def refresh_lod_for_viewport(session) -> bool:
    """Re-evaluate LOD demand after a camera-only retarget (ADR 0050).

    Camera changes never restart evaluation, but they do change which
    pyramid level visible tiles should present.  Demand selection is
    otherwise refreshed only inside presentation builds, so a zoom that
    leaves the active tile set and payload identities untouched would
    keep the old level on screen until an unrelated pan or slice change
    dirtied a payload.  This recomputes the decision from the current
    ``view_range``/``viewport_shape`` (demand math plus pyramid peeks;
    never reduction or other bulk work), queues singleflight
    materializations for the demanded-but-missing level of visible
    rendered tiles, and dirties tiles whose closest resident level
    differs from the payload they currently present so the next commit
    swaps them by payload identity alone.

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
    pyramid = session.lod_pyramid
    desired = int(demand.desired_level)
    commit_needed = False
    visible_by_number = {int(t.montage_index): t for t in tuple(session.visible_tiles)}
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
        if desired > 0 and desired not in resident:
            desired_key = pyramid_key_for(session, rendered, demand=demand, level=desired)
            if pyramid.begin_pending(desired_key):
                session.pending_lod_requests.append(
                    plan_materialization(session, rendered, demand=demand, level=desired, key=desired_key)
                )
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
        applied = int(choose_resident_level(demand, tuple(sorted(resident))))
        if presented_level != applied:
            session.dirty_payloads[int(tile_number)] = None
            commit_needed = True
    return commit_needed


# --------------------------------------------------------------------------
# Worker-side admissions
# --------------------------------------------------------------------------


def admit_ingest_reduction(
    pyramid,
    demand,
    rendered: RenderedTile,
    *,
    semantic_source_id,
    shader_display: bool = True,
) -> bool:
    """Reduce a freshly computed tile to the demanded level, worker-side.

    Runs on the evaluation worker as part of tile materialization (ADR 0041
    gate 1 forbids commit-callback reduction, not worker reduction), so a cold
    tile's first presentation can select the reduced level and never upload a
    native texture.  Exact/semantic/histogram sources stay native; only the
    display texture plane is reduced.  The singleflight claim keeps this from
    duplicating a concurrently scheduled post-hoc materialization.

    Returns the admitted reduced plane (truthy) or None, so the retained
    preview level can derive from it instead of re-reducing native.
    """

    if pyramid is None or demand is None:
        return None
    level = int(demand.desired_level)
    if level <= 0:
        return None
    shader_display = bool(shader_display)
    key = pyramid_key_for_rendered(
        rendered,
        demand=demand,
        level=level,
        semantic_source_id=semantic_source_id,
        shader_display=shader_display,
    )
    if not pyramid.begin_pending(key):
        return None
    try:
        source, histogram, _texture_kind = texture_source_for_rendered(rendered, shader_display=shader_display)
        reduced = pyramid.admit(key, reduce_box_mean(source, key.factor_xy))
        if histogram is not None:
            hist_key = histogram_key_for_level_key(key)
            if pyramid.begin_pending(hist_key):
                try:
                    pyramid.admit(hist_key, reduce_box_mean(histogram, hist_key.factor_xy))
                except Exception:
                    pyramid.end_pending(hist_key)
        return reduced
    except Exception:
        # LOD is a display optimization: a failed reduction must not fail the
        # tile result.  Release the claim so a later commit can retry.
        pyramid.end_pending(key)
        return None


def admit_preview_reduction(
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

    Worker-side and opportunistic (ADR 0050 retained preview level): every
    evaluated tile leaves a coarse copy in the pinned preview cache, so any
    index ever computed re-presents instantly through the floor forever.
    When the ingest reduction already produced a finer level whose shape
    divides evenly, the preview derives from it (relative_factor**2 fewer
    texels); otherwise it reduces the native plane once.
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
        source, histogram, _kind = texture_source_for_rendered(rendered, shader_display=shader_display)
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
        return (str(TexturePlaneKind.COMPLEX_RG32F.value), "scalar")
    return (str(TexturePlaneKind.RGB8.value), "scalar", str(TexturePlaneKind.COMPLEX_RG32F.value))


def best_floor_key(session, source_index: int):
    """Best resident pyramid key for one tile: nearest demand, finer ties."""

    pyramid = session.lod_pyramid
    if pyramid is None:
        return None
    demand = session.lod_policy_decision.demand
    desired = int(demand.desired_level)
    semantic_id = session.tile_semantic_source_id(int(source_index))
    best = None
    for component in floor_component_tags(session):
        for key in pyramid.resident_keys_for(semantic_id, int(source_index), component):
            level = max(int(key.level_xy[0]), int(key.level_xy[1]))
            rank = (abs(level - desired), level)
            if best is None or rank < best[0]:
                best = (rank, key, level)
        if best is not None:
            break
    if best is not None:
        return (best[1], best[2], pyramid)
    preview = session.lod_preview_pyramid
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
    best = best_floor_key(session, int(tile.source_index))
    if best is None:
        return False
    if payload is None:
        return True
    presented = int(getattr(getattr(payload, "lod", None), "level", 0) or 0)
    return int(best[1]) != presented


def ensure_floor_payloads(session, tile_numbers) -> None:
    """Present the best resident pyramid level for unrendered planned tiles.

    The floor invariant (ADR 0050): a planned tile with any resident
    level never shows a placeholder.  Floor payloads are quality
    "preview" — they draw pixels but refuse semantic reads — and the
    ordinary evaluation path replaces them with exact payloads as tile
    results arrive.  Dictionary probes only; no reduction, no copies.
    """

    if not tile_numbers or not resident_lod_active(session):
        return
    pyramid = session.lod_pyramid
    if pyramid is None:
        return
    by_number = {
        int(tile.montage_index): tile
        for tile in tuple(session.visible_tiles)
    }
    for tile_number in sorted(int(number) for number in tile_numbers):
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
        best = best_floor_key(session, source_index)
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
            texture_kind=_floor_texture_kind(key.component),
            lod=lod,
            quality="preview",
        )
        session.display_tile_payloads[tile_number] = payload
        session.pending_payload_upserts[tile_number] = None
        session.lod_floor_presentations = int(getattr(session, "lod_floor_presentations", 0) or 0) + 1


def _floor_texture_kind(component: object) -> TexturePlaneKind:
    value = str(getattr(component, "value", component))
    if value == "scalar":
        return TexturePlaneKind.SCALAR_R32F
    return TexturePlaneKind(value)


# --------------------------------------------------------------------------
# Texture selection (session-side)
# --------------------------------------------------------------------------


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
    pyramid = session.lod_pyramid
    resident_levels = tile_resident_levels(session, rendered, demand=demand)
    desired = int(demand.desired_level)
    if desired > 0 and desired not in resident_levels:
        desired_key = pyramid_key_for(session, rendered, demand=demand, level=desired)
        if pyramid.begin_pending(desired_key):
            session.pending_lod_requests.append(
                plan_materialization(
                    session, rendered, demand=demand, level=desired, key=desired_key, native_source=source
                )
            )
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
        texture_histogram = None if cached_histogram is None else np.asarray(cached_histogram)
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
    view = getattr(session, "pending_lod_requests", None)
    if view is not None and hasattr(view, "release"):
        _apply_release_effects(pyramid, view.release(request))
    else:
        _release_chain_claims(pyramid, _request_chain(request))
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
    pyramid = getattr(session, "lod_pyramid", None)
    lifecycle = getattr(session, "lifecycle", None)
    if lifecycle is not None:
        return _apply_release_effects(pyramid, lifecycle.session_replaced())
    requests = list(getattr(session, "pending_lod_requests", ()) or ())
    if pyramid is None:
        return 0
    released = 0
    for request in requests:
        for step_key, _rel in _request_chain(request):
            if step_key is not None:
                pyramid.end_pending(step_key)
                released += 1
    session.pending_lod_requests.clear()
    return released


# --------------------------------------------------------------------------
# Renderer-side: policy resolution, caches, scheduling
# --------------------------------------------------------------------------


def policy_mode_for_renderer(renderer) -> str:
    """Resolve the configured montage LOD policy for the active backend.

    ADR 0050 scopes the first "resident" slice to VisPy tiled scenes;
    any other backend keeps native-only regardless of the setting.
    """

    choice = getattr(getattr(renderer.win, "app_settings", None), "montage_lod_policy", None)
    value = str(getattr(choice, "value", choice) or LOD_POLICY_NATIVE_ONLY)
    if value != LOD_POLICY_RESIDENT:
        return LOD_POLICY_NATIVE_ONLY
    capabilities = image_view_backend_capabilities(renderer.win.img_view)
    name = str(getattr(capabilities, "name", ""))
    if name == "vispy":
        return LOD_POLICY_RESIDENT
    if name == "pyqtgraph" and _pyqtgraph_resident_lod_enabled():
        # ADR 0050 phase 3: the shared planner/materializer feed reduced
        # payload.image to per-tile ImageItems (scale transform maps image
        # pixels onto native texels).  Opt-in until the A/B evidence gate
        # ("where measured") flips the default.
        return LOD_POLICY_RESIDENT
    return LOD_POLICY_NATIVE_ONLY


def _pyqtgraph_resident_lod_enabled() -> bool:
    return str(os.environ.get("ARRAYSCOPE_PYQTGRAPH_RESIDENT_LOD", "") or "").strip() == "1"


def native_policy_reason_for_renderer(renderer) -> str:
    """Explain *why* native-only applies (honest diagnostics, ADR 0050).

    Distinguishes the user-selected native-only policy from a backend that
    resident LOD has not been adopted on yet; `tile_lod_reason` surfaces this
    verbatim when a desired factor > 1 goes unapplied.
    """

    choice = getattr(getattr(renderer.win, "app_settings", None), "montage_lod_policy", None)
    value = str(getattr(choice, "value", choice) or LOD_POLICY_NATIVE_ONLY)
    if value != LOD_POLICY_RESIDENT:
        return LOD_REASON_NATIVE_POLICY
    capabilities = image_view_backend_capabilities(renderer.win.img_view)
    name = str(getattr(capabilities, "name", ""))
    if name != "vispy" and not (name == "pyqtgraph" and _pyqtgraph_resident_lod_enabled()):
        return LOD_REASON_BACKEND_ADOPTION_PENDING
    return LOD_REASON_NATIVE_POLICY


def shared_pyramid(renderer) -> PyramidCache:
    pyramid = getattr(renderer, "_montage_lod_pyramid_cache", None)
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
        renderer._montage_lod_pyramid_cache = pyramid
    return pyramid


def preview_pyramid(renderer) -> PyramidCache:
    """Pinned whole-stack preview cache (ADR 0050 retained preview level).

    Separate instance = structural eviction exemption: display-churn in
    the main pyramid can never push preview planes out.  At preview
    levels a full 272-tile stack is a few megabytes, so the cap is a
    formality.
    """

    pyramid = getattr(renderer, "_montage_lod_preview_cache", None)
    if not isinstance(pyramid, PyramidCache):
        pyramid = PyramidCache(max_bytes=64 * 1024 * 1024)
        renderer._montage_lod_preview_cache = pyramid
    return pyramid


def schedule_materializations(renderer, session) -> None:
    """Drain demanded-but-missing pyramid levels into background work.

    Reduction never runs in this GUI path: each request becomes a
    low-priority worker item on the montage tile controller, superseded
    per tile by viewport identity.  Completion admits into the pyramid
    cache from the worker and re-enters presentation through the same
    dirty-payload commit path a late tile result uses.
    """

    pending = getattr(session, "pending_lod_requests", None)
    if pending is not None and hasattr(pending, "drain"):
        requests = list(pending.drain())
    else:
        requests = list(pending or ())
        if pending is not None:
            pending.clear()
    if not requests:
        return
    pyramid = getattr(session, "lod_pyramid", None)
    if pyramid is None:
        return
    if not renderer._montage_session_is_current(session):
        for request in requests:
            _release_request_claims(session, request, pyramid)
        return
    controller = getattr(renderer.win, "montage_tile_evaluation_controller", renderer.win.visible_evaluation_controller)
    session_id = int(session.session_id)
    session_key = session.key
    supersession_value = (session_key, session_id, int(getattr(session, "lod_target_revision", 0) or 0))
    blocked_any = False
    for request in requests:
        tile_number = int(request[0])
        key = request[1]
        source = request[2]
        reduce_factor_xy = tuple(int(value) for value in (request[3] if len(request) > 3 else key.factor_xy))
        chain = _request_chain(request)

        def evaluate(chain=chain, source=source, pyramid=pyramid):
            # Level-chaining (ADR 0050): each step reduces the previous
            # plane, so producing the target's neighbors costs a fraction
            # of one native reduction instead of one native read each.
            plane = source
            try:
                for step_key, rel in chain:
                    plane = reduce_box_mean(plane, rel)
                    if step_key is not None:
                        plane = pyramid.admit(step_key, plane)
                return plane
            except BaseException:
                _release_chain_claims(pyramid, chain)
                raise

        def done(_result, request=request, tile_number=tile_number, session_id=session_id, session_key=session_key):
            _finish_request_claims(session, request, pyramid)
            on_level_ready(renderer, session_id, session_key, tile_number)

        def release(request=request, pyramid=pyramid, tile_number=tile_number, session_id=session_id, session_key=session_key):
            # Supersession cancels stale *work*, never a completed
            # result: the worker may have admitted levels before the
            # item went stale, and dropping the notification leaves the
            # tile presenting an outdated level until an unrelated event
            # (user-visible as tiles stuck mid-zoom).  Admitted levels
            # notify; unstarted ones release their singleflight claims.
            if _release_request_claims(session, request, pyramid):
                on_level_ready(renderer, session_id, session_key, tile_number)

        started = controller.start_latest(
            evaluate,
            key=("montage_lod_level", session_key, tile_number, key.level_xy),
            priority=EvalPriority.PREFETCH,
            replace_group=f"montage-lod:{tile_number}",
            on_done=done,
            on_error=lambda _exc, request=request, pyramid=pyramid: _release_request_claims(session, request, pyramid),
            on_stale=release,
            supersession_key=("montage-lod", tile_number),
            supersession_value=supersession_value,
            work_item=WorkItem(
                key=("montage_lod_materialization", session_key, session_id, tile_number, key.level_xy),
                lane=WorkLane.SPECULATIVE_RESIDENCY,
                quality="preview",
                supersession_key=("montage-lod", tile_number),
                supersession_value=supersession_value,
                estimated_bytes=max(
                    1,
                    int(getattr(source, "nbytes", 0) or 0)
                    // max(1, int(reduce_factor_xy[0]) * int(reduce_factor_xy[1])),
                ),
            ),
        )
        if started is None:
            # Admission blocked (bounded speculative lane yielding to
            # visible work): release every singleflight claim immediately,
            # or the next refresh can never re-request these levels and the
            # tile is stuck at its old LOD until the demand changes.
            _release_request_claims(session, request, pyramid)
            renderer._montage_lod_materializations_blocked = (
                int(getattr(renderer, "_montage_lod_materializations_blocked", 0) or 0) + 1
            )
            blocked_any = True
        else:
            renderer._montage_lod_materializations_scheduled = (
                int(getattr(renderer, "_montage_lod_materializations_scheduled", 0) or 0) + 1
            )
    if blocked_any:
        # A blocked admission must leave a wakeup armed (ADR 0051 P2 —
        # before this, released levels waited for an unrelated pan/zoom to
        # trigger the next presentation build; field report 2026-07-05:
        # tiles stuck on a coarser LOD at idle, healed only by panning).
        # When any tracked work finishes, re-derive the demand: the refresh
        # re-claims the released levels and dispatch drains them.
        controller.notify_when_capacity(
            ("montage-lod", session_key),
            lambda: retry_blocked_materializations(renderer),
        )


def retry_blocked_materializations(renderer) -> None:
    """Capacity-waiter wakeup for blocked LOD admissions (ADR 0051 P2).

    Re-evaluates the live demand (re-claiming any released levels into
    ``pending_lod_requests``) and re-derives dispatch.  Cheap and safe to
    run spuriously: demand math plus pyramid peeks, then idempotent
    scheduling.
    """

    session = getattr(renderer, "_montage_session", None)
    if session is None or not renderer._montage_session_is_current(session):
        return
    if session.refresh_lod_for_viewport():
        renderer._schedule_montage_presentation_commit(session, force=False)
    renderer._dispatch_montage_work(session)


def on_level_ready(renderer, session_id, session_key, tile_number) -> None:
    renderer._montage_lod_materializations_completed = (
        int(getattr(renderer, "_montage_lod_materializations_completed", 0) or 0) + 1
    )
    session = getattr(renderer, "_montage_session", None)
    if session is None or not renderer._is_current_montage_session(session_id, session_key):
        return
    if not renderer._is_current_render_generation(session.render_generation):
        return
    session.lod_materializations_completed = int(getattr(session, "lod_materializations_completed", 0) or 0) + 1
    if int(tile_number) in session.rendered_tiles:
        session.dirty_payloads[int(tile_number)] = None
    # Machine-derived dispatch (ADR 0051 P2): a completed level is backend
    # evidence with its own consumer — re-derive instead of hand-picking.
    renderer._dispatch_montage_work(session)
