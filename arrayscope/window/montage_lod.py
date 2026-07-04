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

from typing import NamedTuple

import numpy as np

from arrayscope.core.scheduler import EvalPriority
from arrayscope.core.work_graph import WorkItem, WorkLane
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.lod import (
    LOD_POLICY_NATIVE_ONLY,
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


def texture_source_for_rendered(rendered: RenderedTile) -> tuple[np.ndarray, np.ndarray | None, TexturePlaneKind | None]:
    """Return the texture source plane, histogram, and plane kind of a tile."""

    texture_kind = getattr(rendered, "texture_kind", None)
    if texture_kind is not None and not isinstance(texture_kind, TexturePlaneKind):
        texture_kind = TexturePlaneKind(getattr(texture_kind, "value", texture_kind))
    if texture_kind == TexturePlaneKind.COMPLEX_RG32F and getattr(rendered, "semantic_data", None) is not None:
        source = np.asarray(rendered.semantic_data)
    else:
        source = np.asarray(rendered.image)
    histogram = None if rendered.histogram_data is None else np.asarray(rendered.histogram_data)
    return source, histogram, texture_kind


def pyramid_key_for_rendered(
    rendered: RenderedTile, *, demand, level: int, semantic_source_id
) -> PyramidLevelKey:
    """Pyramid identity of one level of a rendered tile (ADR 0050 key contract).

    ``semantic_source_id`` is the session-owned semantic identity of the tile
    content (session key + source index).  Object identity of the source
    array must never appear in pyramid keys: rendered tiles are rebuilt
    freely across commits and sessions, and cached levels must stay
    addressable without a live ``RenderedTile`` so presentation can floor on
    resident levels for tiles that have not been rendered yet.
    """

    _source, _histogram, texture_kind = texture_source_for_rendered(rendered)
    factor_x, factor_y = factor_xy_for_level(demand, int(level))
    component = "scalar" if texture_kind is None else str(getattr(texture_kind, "value", texture_kind))
    return PyramidLevelKey(
        source_id=semantic_source_id,
        tile_id=int(rendered.tile.source_index),
        component=component,
        level_xy=(int(factor_x).bit_length() - 1, int(factor_y).bit_length() - 1),
    )


def pyramid_key_for(session, rendered: RenderedTile, *, demand, level: int) -> PyramidLevelKey:
    return pyramid_key_for_rendered(
        rendered,
        demand=demand,
        level=level,
        semantic_source_id=session.tile_semantic_source_id(rendered.tile.source_index),
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
        )
    return int(session.lod_policy_decision.applied_factor)


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
    resident = []
    for level in demand.acceptable_levels:
        if int(level) <= 0:
            continue
        if all(
            session.lod_pyramid.peek(pyramid_key_for(session, tile, demand=demand, level=int(level))) is not None
            for tile in rendered
        ):
            resident.append(int(level))
    return tuple(resident)


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
    """

    tile_number: int
    key: PyramidLevelKey
    source: object
    reduce_factor_xy: tuple[int, int]
    cross_level: bool = False


def plan_materialization(
    session,
    rendered: RenderedTile,
    *,
    demand,
    level: int,
    key: PyramidLevelKey,
    native_source: np.ndarray | None = None,
) -> LodMaterializationRequest:
    """Choose the cheapest deterministic reduction source for one level.

    ADR 0050 materializes level *n+1* from level *n* where possible:
    deriving from the finest already-resident coarser level touches
    ``relative_factor**2`` fewer texels than re-reducing the native
    plane.  Box means compose exactly only when every box is full, so the
    cross-level path is taken only when the native plane divides evenly
    by the demanded per-axis factors; partial trailing boxes fall back to
    the single canonical native reduction to keep level content
    independent of cache state.
    """

    if native_source is None:
        native_source, _histogram, _texture_kind = texture_source_for_rendered(rendered)
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
        best = (candidate, cached, (factor_x // candidate_x, factor_y // candidate_y))
    if best is None:
        return LodMaterializationRequest(tile_number, key, native_source, (factor_x, factor_y))
    session.lod_cross_level_reductions += 1
    return LodMaterializationRequest(tile_number, key, best[1], best[2], cross_level=True)


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
        resident = {
            int(level)
            for level in demand.acceptable_levels
            if int(level) > 0
            and pyramid.peek(pyramid_key_for(session, rendered, demand=demand, level=int(level))) is not None
        }
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


def admit_ingest_reduction(pyramid, demand, rendered: RenderedTile, *, semantic_source_id) -> bool:
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
    key = pyramid_key_for_rendered(rendered, demand=demand, level=level, semantic_source_id=semantic_source_id)
    if not pyramid.begin_pending(key):
        return None
    try:
        source, _histogram, _texture_kind = texture_source_for_rendered(rendered)
        return pyramid.admit(key, reduce_box_mean(source, key.factor_xy))
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
    _source, _histogram, texture_kind = texture_source_for_rendered(rendered)
    component = "scalar" if texture_kind is None else str(getattr(texture_kind, "value", texture_kind))
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
            source, _hist, _kind = texture_source_for_rendered(rendered)
            preview_pyramid.admit(key, reduce_box_mean(source, (factor, factor)))
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
    return ("scalar", str(TexturePlaneKind.COMPLEX_RG32F.value))


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
    if int(tile_number) in session.active_tile_requests:
        return False
    if tile is None:
        tile = next(
            (t for t in tuple(session.visible_tiles) if int(t.montage_index) == int(tile_number)),
            None,
        )
    if tile is None:
        return False
    payload = session.display_tile_payloads.get(int(tile_number))
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
        if tile_number in session.active_tile_requests:
            # An exact evaluation is in flight: flooring now would present
            # a preview one commit before its exact replacement, doubling
            # payload/identity churn for every tile of a cold fill.
            continue
        existing = session.display_tile_payloads.get(tile_number)
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
            histogram_data=None,
            source_id=(*semantic_id, "floor", str(key.component), key.level_xy),
            texture_data=np.asarray(plane),
            texture_kind=None if key.component == "scalar" else TexturePlaneKind(key.component),
            lod=lod,
            quality="preview",
        )
        session.display_tile_payloads[tile_number] = payload
        session.pending_payload_upserts[tile_number] = None
        session.lod_floor_presentations = int(getattr(session, "lod_floor_presentations", 0) or 0) + 1


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
    resident_levels = tuple(
        int(level)
        for level in demand.acceptable_levels
        if int(level) > 0
        and pyramid.peek(pyramid_key_for(session, rendered, demand=demand, level=int(level))) is not None
    )
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
    lod = LodInfo(
        level=applied,
        factor=max(int(factor_xy[0]), int(factor_xy[1])),
        source_shape=source_shape,
        texture_shape=tuple(int(value) for value in texture.shape[:2]),
        gutter=0,
    )
    return texture, histogram, lod


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
    if str(getattr(capabilities, "name", "")) != "vispy":
        return LOD_POLICY_NATIVE_ONLY
    return LOD_POLICY_RESIDENT


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

    requests = list(getattr(session, "pending_lod_requests", ()) or ())
    if not requests:
        return
    session.pending_lod_requests.clear()
    pyramid = getattr(session, "lod_pyramid", None)
    if pyramid is None:
        return
    if not renderer._montage_session_is_current(session):
        for request in requests:
            pyramid.end_pending(request[1])
        return
    controller = getattr(renderer.win, "montage_tile_evaluation_controller", renderer.win.visible_evaluation_controller)
    session_id = int(session.session_id)
    session_key = session.key
    supersession_value = (session_key, session_id, int(getattr(session, "lod_target_revision", 0) or 0))
    for request in requests:
        tile_number = int(request[0])
        key = request[1]
        source = request[2]
        reduce_factor_xy = tuple(int(value) for value in (request[3] if len(request) > 3 else key.factor_xy))

        def evaluate(key=key, source=source, reduce_factor_xy=reduce_factor_xy, pyramid=pyramid):
            return pyramid.admit(key, reduce_box_mean(source, reduce_factor_xy))

        def done(_result, tile_number=tile_number, session_id=session_id, session_key=session_key):
            on_level_ready(renderer, session_id, session_key, tile_number)

        def release(key=key, pyramid=pyramid, tile_number=tile_number, session_id=session_id, session_key=session_key):
            # Supersession cancels stale *work*, never a completed
            # result: the worker may have admitted the level before the
            # item went stale, and dropping the notification leaves the
            # tile presenting an outdated level until an unrelated event
            # (user-visible as tiles stuck mid-zoom).  Admitted levels
            # notify; unstarted ones release the singleflight claim.
            if pyramid.peek(key) is not None:
                on_level_ready(renderer, session_id, session_key, tile_number)
            else:
                pyramid.end_pending(key)

        started = controller.start_latest(
            evaluate,
            key=("montage_lod_level", session_key, tile_number, key.level_xy),
            priority=EvalPriority.PREFETCH,
            replace_group=f"montage-lod:{tile_number}",
            on_done=done,
            on_error=lambda _exc, key=key, pyramid=pyramid: pyramid.end_pending(key),
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
            # visible work): release the singleflight claim immediately,
            # or the next refresh can never re-request this level and the
            # tile is stuck at its old LOD until the demand changes.
            pyramid.end_pending(key)
            renderer._montage_lod_materializations_blocked = (
                int(getattr(renderer, "_montage_lod_materializations_blocked", 0) or 0) + 1
            )
        else:
            renderer._montage_lod_materializations_scheduled = (
                int(getattr(renderer, "_montage_lod_materializations_scheduled", 0) or 0) + 1
            )


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
    renderer._schedule_montage_presentation_commit(session, force=False)
