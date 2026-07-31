# The programmatic range-replay edge: 8x measured, and refuted (2026-07-31)

Rank-3 finding of the
[GUI-callback dossier](gui-callback-owners-2026-07-31.md) (§1c, §2 I1/I2):
`viewport_bridge.py:44` runs the whole-plan montage retarget
**synchronously inside the calling dispatch** on
every programmatic (non-pointer) range change, while pointer gestures have
gone through a 16 ms coalescing timer since 2026-07-14. The dossier measured
a 141 ms `viewport_range_bridge` feedback EWMA during wgpu scrub, 64–254 ms
per fit-stretch call three times per fill, and a full-grid zoom-in whose
single input dispatch took up to 542 ms (p95 473) against the 15 ms
warm-input bar.

**The prize is real and was measured: deferring that retarget takes the
caller-blocking cost from 92 ms median / 217 ms p95 to 12 / 25 — 8x, against
a control pass whose own same-code spread is 35 ms.**

**It is also refuted. Deferral is not safe on this edge, and the reason is
not the delay — it is the asynchrony.** Moving the retarget off the caller's
dispatch costs one whole-montage WGPU republication per return-leg pan: the
exact cost [out-of-view retention](out-of-view-retention-2026-07-30.md)
exists to remove. Reproduced at 3/3 and again at 2/2 with the gate set to
0 ms, i.e. merely "next event turn". Trading an ~80 ms caller block for a
whole-montage republication (~90 ms of commit work at 272 tiles per the
per-commit dossier) is not a win.

**What landed: only the O(N^2) hoist**, which is independent of
the deferral and safe. The bounded-discharge work is reverted, with the four
composition defects it exposed recorded below so the next attempt starts
where this one stopped rather than rediscovering them.

## 1. What the V1 obligation actually requires (still believed)

The synchronous edge carries a stated reason at the call site: "the V1
boundary reveal requires its visibility update before the caller observes
pixels", and three broader variants are already buried for violating it
(2026-07-14, [graveyard](../graveyard.md)): pacing every range replay delayed
the reveal on both backends, reusing the frame-viewport timer paced the
*pipeline continuation*, and treating an uncommitted preview session as
viewport truth made ordering matter.

That framing survives this attempt. The obligation genuinely is narrower
than a whole-plan replan, and it can be discharged **by refusing to answer**
rather than by working inline: hold `visible_plan_complete()` false while the
replay is owed, and no settle-waiter can observe a scope the camera has
already invalidated. That mechanism worked — it is not what failed.

## 2. What actually blocks it

`retarget_viewport` derives the retention scope and the presentation edge
from the visible-set **transition**, and the WGPU executor applies a delta
only when `previous_commit["tiles"] <= payloads`. Both are sensitive to
*when* the retarget runs relative to the commit stream, not just to whether
it eventually runs.

Deferring the retarget past the caller's dispatch lets commits from
still-draining work interleave between the camera move and the retarget that
answers it. The return leg — panning back to a region whose pixels never
changed, which must cost **zero** republications — then costs one.
Confirmed against
`tests/ui/test_montage_out_of_view_retention.py::test_wgpu_pan_away_and_back_reuses_the_retained_presentation_scope`,
which fails 3/3 on the deferred tree and passes 3/3 on `main`, interleaved
across a load range of 11–27 so the verdict is not contention.

The decisive experiment: setting the coalescing gate to **0 ms** — same event
loop, next turn — still fails 2/2. So this is not a tuning problem. Any
attempt that moves this retarget off the caller's dispatch has to make the
republication precondition robust to interleaved commits first; that is the
real prerequisite, and it is a bigger piece of work than the edge itself.

## 3. The four composition defects, in the order they appeared

Each was a real defect, fixed, and revealed the next. Recorded because they
are the map for a future attempt — none of them is hypothetical.

1. **The consumed presentation edge.** A partial sync slice that updates
   `visible_tiles` *is* the presentation-change signal, so the deferred pass
   computes `presentation_changed=False` and skips the commit that reveals
   the new draw set. Must be carried forward explicitly.
2. **The swallowed instrumented entry point.** Declining to defer must
   happen *before* any side effect, and the fallback must run through
   `retarget_montage_viewport()` — benchmarks and
   `test_wgpu_montage_pyqtgraph_range_change_schedules_viewport_tile_update`
   wrap that name and count calls on it.
3. **The double transition.** Running `retarget_viewport` once in a sync
   slice and again in the deferred pass makes the second call read the first
   call's result as its predecessor: a home→away move is recorded as
   away→away and the tiles the camera left lose their retention slot
   (observed directly: tiles 4 and 12 visible at home, absent from
   `retained_out_of_view`). **One camera move must stay one retarget.**
4. **The premature discharge.** `apply_montage_presentation` only *requests*
   a commit, so clearing the obligation when the retarget ran let settle be
   observed while the committed active set was still the pre-replay one; and
   clearing it at *any* publication let an unrelated in-flight commit clear
   it before the replay's own retarget had run. The obligation needs two
   stages (retarget done, then publication).

## 4. What did land: the O(N^2) hoist

`_resolve_montage_tiles_from_cache` rebuilt its lookup state on every call —
a full retained-store scan (up to 4096 entries copied), a dict comprehension
over every committed-frame payload, and a fresh semantic key-batch closure.
The budgeted viewport-additions loop calls it **once per tile**, so the loop
was quadratic in the retained-store size. None of those inputs can change
between iterations (commits happen after the loop), so it was pure
recomputation; the state now lives in a `MontageCacheResolutionContext` built
once per pass.

Measured at 650 retained entries:

| quantity | value |
|---|---|
| one resolution-context build | 0.222 ms median (p95 0.372) |
| additions per gesture, deep zoom in | 3 |
| additions per gesture, **zoom out to mid** | **74** |
| additions per gesture, zoom out to full | 38 |
| additions per gesture, pan on settled full grid | 0 |

So the old code spent ~16 ms (74 × 0.222) on a zoom-out rebuilding state it
already had, and nothing on a settled pan. **This is below this machine's
noise floor and the A/B could not see it** — it is quoted from direct
measurement, not claimed from the A/B. It stays because a quadratic inside a
budgeted loop is a shape defect, not because this workload noticed. The
build happens only when the loop has at least one addition, so it is never
more work than before.

## 5. How the 8x was measured (method retained for the next attempt)

**Not** by comparing profiler invocations. Two `profile_montage_workflow`
runs on this machine on the same tree returned `action_fit_stretch_ms` of
665 ms and 4152 ms — a 6x spread, with sibling worktrees holding load at
18–33 on 16 cores against the dossier's 4.6–7.7. A sequential single-shot
A/B in that regime measures the load and inverts sign at will.

Instead: **one process, one settled 272-tile field montage, arms toggled by
instance attributes**, no production flags — `discharge_...= None` restores
the synchronous edge, and wrapping `_resolve_montage_tiles_from_cache` to
drop its `context` kwarg restores the per-tile rebuild. Twelve programmatic
replays per block, each landing on a different rect so the convergence latch
cannot answer one for free. A control pass runs the pre-change arm against
itself, four blocks, before any comparison; blocks are order-balanced in both
directions.

| pair (identical code) | median (ms) | median (ms) | spread |
|---|---|---|---|
| control A vs control B | 134.3 | 99.2 | 35 ms |
| `sync_nohoist` vs `sync_hoist`¹ | 91.8 | 115.5 | 24 ms |

¹ These differ only in code a settled replay does not reach, so this is a
second independent control estimate.

| arm | call med | call p95 | call max | quiet med |
|---|---|---|---|---|
| `sync_nohoist` (pre-change) | 91.8 | 216.7 | 292.2 | 91.8 |
| `sync_hoist` (hoist only) | 115.5 | 199.8 | 251.3 | 115.5 |
| `bounded` (reverted) | **11.6** | **25.1** | 56.7 | 110.7 |

Time-to-quiet for the bounded arm sits inside the same-code spread, so the
work moved off the caller rather than vanishing. An earlier two-arm run
replicates independently: 73.1 / 144.1 synchronous against 11.3 / 16.6
bounded, own control pair 70.0 vs 79.9.

## 6. What this leaves standing

- **The bar stays red on this edge**, and the dossier's ranking is unchanged:
  the retarget still runs inside the wheel handler at up to 542 ms.
- **The prerequisite is now named**: make the WGPU republication precondition
  robust to commits interleaving with a pending retarget. Until then, the
  synchronicity of this edge is load-bearing and should not be touched.
- **S1 (the 1.0–1.3 s cold-entry stage plan) is untouched** and remains the
  largest item on the ranked list — and, unlike this one, it is a chunking
  problem rather than an ordering-invariant problem.

## Repro

The harnesses are scratchpad scripts (no artifacts in the repo). Shape:
build the 272-tile montage, settle, alternate arms in-process; and for the
regime numbers, count `_resolve_montage_tiles_from_cache` calls and time
`_montage_cache_resolution_context` across a zoom/pan lifecycle. A scratchpad
script imports the MAIN checkout unless it pins the worktree on `sys.path`
**and asserts on `arrayscope.__file__`**; a fresh worktree also has no
`data/` (symlink it — it is gitignored).
