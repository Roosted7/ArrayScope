# Ground rules

Distilled from the 2026-07 course reset
([retro](redesign/retro-2026-07.md)) and the GPU-port sessions. These are
standing law for all lanes, not redesign-era process. AGENTS.md points here.

1. **Pixels are the gate.** A claim of "fixed" requires the relevant
   scenario green on real hardware ([testing/README.md](testing/README.md),
   rings 3–4). An offscreen or unit pass is never acceptance for rendering,
   scheduling, or LOD behavior.
2. **Fix by deleting a duplicate owner, not by adding a gate.** If a fix
   adds a predicate, a generation, or a new set, it is probably wrong. One
   owner per decision: one ranker, one visible set, one completion
   predicate, one key derivation. (Every recurring defect class of 2026-06/07
   — black tiles, wrong order, identity aliasing — was two owners of one
   truth drifting apart.)
3. **GUI thread never hangs.** A synchronous GUI-thread step >50 ms is a
   bug; pan/scrub heartbeat target ~16 ms. Every user-visible open, render,
   zoom, pan, scroll, scrub, level change, or refinement step targets 2 s and
   **hard-fails at 5 s**. The limit is per step, so a multi-step scenario may
   take longer in total. Tests and tools take this value only from
   `arrayscope.tools.interaction_budget`; a local timeout may be shorter but
   must be capped by that owner. Never widen a settlement timeout to make a
   slow path green. A longer whole-process watchdog may only detect a dead
   child and cannot turn late settlement into success.
4. **Tests pin user-visible behavior.** A test that pins an implementation
   detail (upload counts, internal predicate scoping) may be deleted when it
   blocks a user-visible fix — say so in the commit message.
5. **No silent fallbacks — broken must be loud.** `except Exception` around
   an import or lookup that turns a missing symbol into a default is
   forbidden (enforced by `tests/app/test_import_health.py`). A rejected
   payload, a dead code path, or a non-converging tile must surface as a
   counter, a trace event, or a visible diagnostic — silence is how the
   prefetch path stayed dead for a week and the identity-rejection livelock
   stayed invisible.
6. **Repro first; every fix carries its gate.** A failing test (in the ring
   that can see the failure) before the fix; the fix's commit links the
   dossier when one exists. Trace mining before code: stall ring buffers and
   diagnostics JSONLs decode completely.
7. **Speculative work never changes visible outcomes.** Warming, prefetch,
   and priority biasing reorder or pre-place work that was already admitted;
   they never manufacture lifecycle/presentation state.
8. **Residency is not visibility.** GPU/CPU pages may remain resident across an
   operation/value-source change so a later revert can reuse them, but the old
   tile mappings must be hidden immediately. Never-black semantic fallback
   applies only when the complete source identity matches. A slice-only
   predecessor may remain as an explicitly non-semantic preview, but it is
   never acknowledged as current and cannot answer probes or reads. Levels,
   LUT, and other shader-only updates within one encoded presentation identity
   cross with their uniforms atomically. A channel/view-mode change must still
   be encoded in the target identity: it may reuse resident texels only when
   the backend commits the new mapping atomically, otherwise the old mapping is
   hidden. Image-axis, flip, representation, and source changes never retain
   the predecessor mapping.
   An acknowledged finer compatible LOD remains visible for a later coarser
   demand. Neither demand nor a logical payload/cache byte estimate authorizes
   demotion. Only backend-owned physical capacity pressure may replace it,
   after hidden, speculative, superseded, and otherwise less-important
   residency has been reclaimed, and only through an acknowledged complete
   replacement that preserves same-source fallback coverage. An unacknowledged
   candidate has no such protection because it is not physical presentation
   truth.
   A complete same-source target that the backend can already resolve through
   resident pages is a presentation rebind, not cold work: pan/zoom must bind
   the whole set immediately even while the gesture remains active. Only a
   physically cold successor may wait for the interaction-stop edge.
   Backend-safe drawability is not target settlement. A same-source fallback
   may stay visible to prevent black, but it must not suppress the exact
   producer unless the lifecycle's canonical quality/LOD rule says the target
   is satisfied.
   Reduced pages may expose an explicitly presentation-qualified sample so a
   diagnostic can explain the pixel that was drawn. That is not permission to
   answer an exact hover, histogram, ROI, measurement, or export read: those
   require explicit native semantic data or must fall through to exact
   evaluation. Geometry truth and value exactness are separate facts.
   Page-table scale/offset is the nominal aligned-grid transform. It cannot
   represent a clipped boundary bin by itself; exact presentation uses the
   canonical target and actual draw blocks and reports the submitted geometry.
9. **Bounded sessions.** End every working session with the app visibly
   better or the change reverted. Update [`queue.md`](queue.md); a reverted
   experiment gets a [`graveyard.md`](graveyard.md) row. New process
   documents require Thomas's explicit ask.
10. **Backends: both first-class for correctness.** VisPy is the Linux
   certification bar; PyQtGraph targets GPU-headless/remote use and gets 2×
   the performance allowance (Thomas, 2026-07-14).
10. **One scheduler.** No new scheduling systems beside the kernel, no new
    pacing timers, no parallel tile-state collections (ADR 0053). Required
    tile debt belongs to `TileLifecycle`; stage waiters belong to the stage
    fan-in; deferred planning owns an immutable missing set. A session-level
    pending/repair queue must not duplicate any of those facts.
11. **No wait without an owner.** Every deferral, bail, or barrier names the
    event that resumes it, and that event must have a live owner — "a later
    replan will pick it up" is not an owner. Six stall defects in 2026-07
    shared one grammar: a consumer waits for completeness (first pixels, an
    atomic successor, a stage plan) while the producer of the missing piece
    was dropped, superseded, or never scheduled. The mechanical check: a
    commit/scheduler bail repeating with an identical signature and no
    in-flight work is a defect, never pacing — `commit_bail` /
    `commit_gate_no_progress` trace events exist to make this visible on
    the first read of a stall dump.
