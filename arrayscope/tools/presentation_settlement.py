"""Strict read-only settlement queries for interaction and release gates.

This module does not schedule, commit, or retain anything.  It combines the
canonical ``FrameSession`` target predicate with the physical facts reported by
the active image backend so tools do not grow subtly different definitions of
"the current pixels are ready".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arrayscope.display.backend_contract import image_view_backend_capabilities


@dataclass(frozen=True, repr=False)
class PresentationTargetToken:
    """Immutable identity of the exact frame/viewport target being observed."""

    session_id: int
    render_generation: int
    viewport_revision: int
    session_key: str = field(repr=False)
    required_targets: tuple[tuple[int, int, int, str, str], ...] = field(repr=False)

    def __repr__(self) -> str:
        required = tuple(target[0] for target in self.required_targets)
        return (
            "PresentationTargetToken("
            f"session_id={self.session_id}, render_generation={self.render_generation}, "
            f"viewport_revision={self.viewport_revision}, required_tiles={required!r})"
        )


@dataclass(frozen=True)
class PresentationSettlementSnapshot:
    """One non-owning observation of logical, physical, and cache settlement."""

    target: PresentationTargetToken | None
    expected_target_matches: bool
    backend: str
    required_tiles: tuple[int, ...]
    unsettled_tiles: tuple[int, ...]
    visible_complete: bool
    target_settled: bool
    commit_debt: tuple[str, ...]
    draw_pending: bool
    backend_tiles: tuple[int, ...]
    physical_tiles: tuple[int, ...]
    physical_truth_errors: tuple[str, ...]
    session_work_inflight: int
    page_cache_pending: int
    rung_materializations_pending: int
    scheduler_pending: int
    scheduler_running: int
    stage_materializations_inflight: int

    def is_settled(self, *, require_quiescent: bool = False) -> bool:
        logical_and_physical = bool(
            self.target is not None
            and self.expected_target_matches
            and self.required_tiles
            and self.visible_complete
            and self.target_settled
            and not self.unsettled_tiles
            and not self.commit_debt
            and not self.draw_pending
            and not self.physical_truth_errors
        )
        if not logical_and_physical or not require_quiescent:
            return logical_and_physical
        return bool(
            self.session_work_inflight == 0
            and self.page_cache_pending == 0
            and self.rung_materializations_pending == 0
            and self.scheduler_pending == 0
            and self.scheduler_running == 0
            and self.stage_materializations_inflight == 0
        )


def presentation_target_token(window) -> PresentationTargetToken | None:
    """Return the exact current target through the window's read-only seam."""

    session = getattr(window, "_frame_session", None)
    if session is None:
        return None
    required_fn = _required_callable(session, "required_tile_numbers")
    required = tuple(sorted(int(tile) for tile in required_fn()))
    lifecycle = getattr(session, "lifecycle", None)
    peek = getattr(lifecycle, "peek", None)
    targets = []
    for tile in required:
        record = None if not callable(peek) else peek(tile)
        target = None if record is None else getattr(record, "target", None)
        targets.append(
            (
                int(tile),
                -1 if target is None else int(getattr(target, "source_index", -1)),
                -1 if target is None else int(getattr(target, "lod_level", -1)),
                repr(None if target is None else getattr(target, "semantic_source_id", None)),
                repr(None if target is None else getattr(target, "identity", None)),
            )
        )
    return PresentationTargetToken(
        session_id=int(session.session_id),
        render_generation=int(getattr(session, "render_generation", 0) or 0),
        viewport_revision=int(getattr(session, "viewport_revision", 0) or 0),
        session_key=repr(getattr(session, "key", None)),
        required_targets=tuple(targets),
    )


def presentation_settlement_snapshot(
    window,
    *,
    expected_target: PresentationTargetToken | None = None,
) -> PresentationSettlementSnapshot:
    """Observe canonical target, commit, draw, and cache truth once."""

    session = getattr(window, "_frame_session", None)
    image_view = getattr(window, "img_view", None)
    backend = str(image_view_backend_capabilities(image_view).name or "")
    if backend not in {"pyqtgraph", "vispy"}:
        raise RuntimeError(f"unsupported image backend settlement contract: {backend!r}")
    if session is None:
        return PresentationSettlementSnapshot(
            target=None,
            expected_target_matches=expected_target is None,
            backend=backend,
            required_tiles=(),
            unsettled_tiles=(),
            visible_complete=False,
            target_settled=False,
            commit_debt=("session",),
            draw_pending=True,
            backend_tiles=(),
            physical_tiles=(),
            physical_truth_errors=("session_missing",),
            session_work_inflight=0,
            page_cache_pending=0,
            rung_materializations_pending=0,
            scheduler_pending=0,
            scheduler_running=0,
            stage_materializations_inflight=0,
        )

    visible_complete_fn = _required_callable(session, "visible_plan_complete")
    target_settled_fn = _required_callable(session, "required_target_settled")
    required_fn = _required_callable(session, "required_tile_numbers")
    unsettled_fn = _required_callable(session, "required_target_unsettled_tiles")
    required = tuple(sorted(int(tile) for tile in required_fn()))
    required_set = frozenset(required)
    target = presentation_target_token(window)

    commit_debt = _commit_debt(session)
    draw_pending_fn = _required_callable(image_view, "presentationDrawPending")
    physical_rows_fn = _required_callable(image_view, "tileTruthPhysicalRows")
    rows = {int(tile): dict(row) for tile, row in dict(physical_rows_fn() or {}).items()}
    physical_tiles = tuple(sorted(rows))
    lifecycle = getattr(session, "lifecycle", None)
    backend_identities = {
        int(tile): identity
        for tile, identity in dict(
            getattr(lifecycle, "backend_presented_identities", {}) or {}
        ).items()
    }
    backend_tiles = tuple(sorted(backend_identities))
    physical_errors = _physical_truth_errors(
        backend=backend,
        required=required_set,
        rows=rows,
        backend_identities=backend_identities,
    )

    page_cache = getattr(session, "lod_page_cache", None)
    page_cache_pending = int(getattr(page_cache, "pending_count", 0) or 0)
    rung_pending = len(tuple(getattr(session, "pending_rung_materializations", ()) or ()))
    evaluator = getattr(window, "operation_evaluator", None)
    display_diagnostics_fn = getattr(evaluator, "display_cache_diagnostics", None)
    display_diagnostics = display_diagnostics_fn() if callable(display_diagnostics_fn) else None
    stage_diagnostics_fn = getattr(evaluator, "stage_materialization_diagnostics", None)
    stage_diagnostics = stage_diagnostics_fn() if callable(stage_diagnostics_fn) else None
    stage_fan_in = getattr(session, "stage_fan_in", None)
    stage_waiting = getattr(stage_fan_in, "has_waiting", None)
    session_work_inflight = (
        len(tuple(getattr(session, "active_tile_requests", ()) or ()))
        + len(tuple(getattr(stage_fan_in, "active_requests", ()) or ()))
        + len(tuple(getattr(stage_fan_in, "attached_requests", ()) or ()))
        + int(bool(callable(stage_waiting) and stage_waiting()))
    )

    return PresentationSettlementSnapshot(
        target=target,
        expected_target_matches=expected_target is None or target == expected_target,
        backend=backend,
        required_tiles=required,
        unsettled_tiles=tuple(int(tile) for tile in unsettled_fn()),
        visible_complete=bool(visible_complete_fn()),
        target_settled=bool(target_settled_fn()),
        commit_debt=commit_debt,
        draw_pending=bool(draw_pending_fn()),
        backend_tiles=backend_tiles,
        physical_tiles=physical_tiles,
        physical_truth_errors=physical_errors,
        session_work_inflight=int(session_work_inflight),
        page_cache_pending=page_cache_pending,
        rung_materializations_pending=rung_pending,
        scheduler_pending=int(getattr(display_diagnostics, "scheduler_pending", 0) or 0),
        scheduler_running=int(getattr(display_diagnostics, "scheduler_running", 0) or 0),
        stage_materializations_inflight=int(getattr(stage_diagnostics, "in_flight", 0) or 0),
    )


def presentation_is_settled(
    window,
    *,
    expected_target: PresentationTargetToken | None = None,
    require_quiescent: bool = False,
) -> bool:
    """Return strict current-target settlement without owning any live state."""

    return presentation_settlement_snapshot(
        window,
        expected_target=expected_target,
    ).is_settled(require_quiescent=require_quiescent)


def presentation_settlement_diagnostic(
    window,
    *,
    expected_target: PresentationTargetToken | None = None,
) -> str:
    """Compact diagnostic for a failed strict-settlement deadline."""

    snapshot = presentation_settlement_snapshot(window, expected_target=expected_target)
    return (
        f"target={snapshot.target!r} expected_match={snapshot.expected_target_matches} "
        f"backend={snapshot.backend} required={snapshot.required_tiles!r} "
        f"unsettled={snapshot.unsettled_tiles!r} visible_complete={snapshot.visible_complete} "
        f"target_settled={snapshot.target_settled} commit_debt={snapshot.commit_debt!r} "
        f"draw_pending={snapshot.draw_pending} backend_tiles={snapshot.backend_tiles!r} "
        f"physical_tiles={snapshot.physical_tiles!r} "
        f"physical_errors={snapshot.physical_truth_errors!r} "
        f"session_inflight={snapshot.session_work_inflight} "
        f"page_cache_pending={snapshot.page_cache_pending} "
        f"rungs_pending={snapshot.rung_materializations_pending} "
        f"scheduler=({snapshot.scheduler_pending},{snapshot.scheduler_running}) "
        f"stage_inflight={snapshot.stage_materializations_inflight}"
    )


def _required_callable(owner, name: str):
    method = getattr(owner, name, None)
    if not callable(method):
        raise RuntimeError(f"presentation settlement owner does not expose {name}()")
    return method


def _commit_debt(session) -> tuple[str, ...]:
    debt = [
        name
        for name in (
            "stage_planning_deferred",
            "flush_pending",
            "final_commit_pending",
            "dirty_payloads",
            "pending_payload_upserts",
            "pending_removals",
            "atomic_successor_pending",
        )
        if bool(getattr(session, name, False))
    ]
    pending_levels = getattr(session, "has_pending_level_update", None)
    if callable(pending_levels) and bool(pending_levels()):
        debt.append("pending_level_update")
    return tuple(debt)


def _physical_truth_errors(*, backend, required, rows, backend_identities) -> tuple[str, ...]:
    errors = []
    physical = frozenset(rows)
    missing = tuple(sorted(required - physical))
    extra = tuple(sorted(physical - required))
    if missing:
        errors.append(f"physical_missing={missing!r}")
    if extra:
        errors.append(f"physical_extra={extra!r}")
    missing_backend = tuple(sorted(required - frozenset(backend_identities)))
    if missing_backend:
        errors.append(f"backend_missing={missing_backend!r}")
    for tile in sorted(required & physical & frozenset(backend_identities)):
        row = rows[tile]
        if row.get("physical_acknowledged_identity") != backend_identities[tile]:
            errors.append(f"identity_mismatch={tile}")
        if backend == "vispy":
            if row.get("physical_draw_bounds_match_layout") is not True:
                errors.append(f"vispy_geometry_unproven={tile}")
        elif row.get("physical_storage_mode") != "image_item":
            errors.append(f"pyqtgraph_geometry_unproven={tile}")
    return tuple(errors)
