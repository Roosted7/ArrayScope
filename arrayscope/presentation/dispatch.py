"""Machine-derived montage dispatch (ADR 0051 P2).

The renderer must never depend on an individual completion callback
remembering to reschedule: ``derive_montage_dispatch`` reads the
authoritative session/machine records and returns every pump the state
implies.  The renderer executes the plan through idempotent, coalesced
schedulers, and every montage event edge ends with it — so records in
PLANNED/dirty imply scheduled work and a lost wakeup is impossible by
construction.  The 1 Hz stall watchdog remains only as an assertion
that this construction holds; every fire is a bug report.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MontageDispatchPlan:
    """Everything the session records imply must be scheduled right now."""

    requeue_orphans: bool = False
    deferred_planning: bool = False
    schedule_tiles: bool = False
    lod_materializations: bool = False
    level_evidence: bool = False
    preview_refinements: tuple[int, ...] = ()
    commit: bool = False
    force_commit: bool = False
    unsettled: bool = False

    @property
    def any_work(self) -> bool:
        return bool(
            self.requeue_orphans
            or self.deferred_planning
            or self.schedule_tiles
            or self.lod_materializations
            or self.level_evidence
            or self.preview_refinements
            or self.commit
        )


def derive_montage_dispatch(session) -> MontageDispatchPlan:
    """Derive the complete dispatch plan from session/machine records.

    Pure and cheap (boolean reads plus one orphan predicate); no I/O, no
    Qt, no scheduling.  Notes:

    - Loading tiles with no work attached anywhere are unservable records;
      requeueing them IS the derivation (a record implies work), not a
      repair.  Deferred-planning sessions intentionally hold unattached
      tiles, and in-flight evaluation/stage work may legitimately
      cover loading entries, so the orphan scan applies only when nothing
      is in flight.
    - ``force_commit`` means evaluation has fully drained and presentation
      state remains: the commit must run now, not on the next interval.
    """

    # getattr-defensive: partial session stubs are common in tests, and a
    # missing record field must read as "no such work", never as a crash in
    # the one function every edge runs.
    stage_fan_in = getattr(session, "stage_fan_in", None)
    pending = bool(getattr(session, "pending_tiles", None))
    active = bool(getattr(session, "active_tile_requests", None))
    loading = bool(getattr(session, "loading_tiles", None))
    stage_active = bool(getattr(stage_fan_in, "active_requests", None))
    stage_attached = bool(getattr(stage_fan_in, "attached_requests", None))
    dirty = bool(getattr(session, "dirty_payloads", None)) or bool(getattr(session, "dirty_tiles", None))
    upserts = bool(getattr(session, "pending_payload_upserts", None))
    removals = bool(getattr(session, "pending_removals", None))
    lod_pending = bool(getattr(session, "pending_lod_requests", None))
    deferred = bool(getattr(session, "stage_planning_deferred", False))
    flushish = bool(getattr(session, "flush_pending", False) or getattr(session, "final_commit_pending", False))
    # Level evidence work (cached level stats / session scan) is scheduled
    # state like any other: a parked explicit-auto flush waits on it, so the
    # derivation must pump it (rule 6 — the pyqtgraph+resident auto-levels
    # wedge was a parked flush whose evidence producer nothing re-armed).
    level_evidence = bool(getattr(session, "pending_level_tiles", None)) or (
        int(getattr(session, "level_scan_remaining_tiles", 0) or 0) > 0
    )
    preview_refinements = tuple(
        int(tile)
        for tile in (
            getattr(session, "unrefined_preview_tiles", lambda: ())()
            if callable(getattr(session, "unrefined_preview_tiles", None))
            else ()
        )
    )

    requeue_orphans = (
        loading
        and not deferred
        and not active
        and not stage_active
        and not stage_attached
    )
    evaluation_drained = not (pending or active or stage_active)
    commit = bool(dirty or upserts or removals or flushish or preview_refinements)
    unsettled = bool(
        pending
        or active
        or loading
        or stage_active
        or stage_attached
        or commit
        or lod_pending
        or deferred
        or level_evidence
    )
    return MontageDispatchPlan(
        requeue_orphans=requeue_orphans,
        deferred_planning=deferred,
        schedule_tiles=bool(pending or requeue_orphans),
        lod_materializations=lod_pending,
        level_evidence=level_evidence,
        preview_refinements=preview_refinements,
        commit=commit,
        force_commit=bool(commit and evaluation_drained),
        unsettled=unsettled,
    )


__all__ = ["MontageDispatchPlan", "derive_montage_dispatch"]
