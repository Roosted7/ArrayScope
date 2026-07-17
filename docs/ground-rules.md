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
   bug; pan/scrub heartbeat target ~16 ms.
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
8. **Bounded sessions.** End every working session with the app visibly
   better or the change reverted. Update [`queue.md`](queue.md); a reverted
   experiment gets a [`graveyard.md`](graveyard.md) row. New process
   documents require Thomas's explicit ask.
9. **Backends: both first-class for correctness.** VisPy is the Linux
   certification bar; PyQtGraph targets GPU-headless/remote use and gets 2×
   the performance allowance (Thomas, 2026-07-14).
10. **One scheduler.** No new scheduling systems beside the kernel, no new
    pacing timers, no parallel tile-state collections (ADR 0053).
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
