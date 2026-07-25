# Trace-proven: shared-transform coverage stall + acknowledgement-only presentation truth gap (2026-07-15, live session)

**Evidence:** `/tmp/arrayscope-stall-{49-1,54-2,63-3}.trace.jsonl` +
`arrayscope-diagnostics-20260715-195337.jsonl` (user session, 19:53–19:54,
complex128 NIfTI 336×336×272, montage axis 2, phase_color, PAL-relaxed,
VisPy). Build: `codex/gpu-engine` at the pre-P9-rebase state (includes
7ee5c74c; predates 5fc43d7a/e19913b6 — verified by absent commit-batch
fields). Authored on `codex/gpu-engine`; dedupe against the live P9 record
at integration.

## Bug 1 — shared-transform coverage stall (P9 lane; NOT fixed on this branch)

Stall signature decoded (`frame_runtime._montage_watchdog_tick`, 11-tuple):
`(session_id, pending, evaluating, active_requests, dirty, upserts,
materializations, level-evidence, level-stale, presented, rendered)` —
sessions 49/54/63 stuck at `presented=100, rendered=0`, everything else 0,
while `required_target_unsettled_tiles()` held 57–62 tiles (presented
level 3, target level 2) and the kernel sat idle.

Mechanism from the ring buffer: every `pipeline_plan` replanned 57–62
DESIRED level-2 steps; exactly one shared-transform fanout ran
(`shared_fanout_batch` covering tiles `[4,3,2,1,0]` only); every later
plan had `submitted=0` because `prepare_rung` (frame_effects.py:337)
claim/coverage gates refused the steps — and nothing releases or refills
claims for the tiles the partial fanout never covered. Planned work
evaporates with no producer and no wakeup; after 2 s the watchdog asserts.
7ee5c74c ("Close shared-transform coverage stalls") was IN the running
build — this is a surviving member of that family at the
partial-fanout-coverage refill edge (frame_effects.py:337 +
`submit_shared_transform_floor` :835).

Secondary: the drain collapsed to one-upsert-per-commit (~12 ms × 22 Hz,
83/88 batches zero-upload) — the diagnostics "UI fan-in" bottleneck;
5fc43d7a's `batch_limit = max(4, …)` addresses this half.

Also observed: `atomic_source_successor_committed` stale-True in every
stall-49 batch (plain bool surviving `retarget_index_window`) — fixed by
e19913b6.

## Bug 2 — acknowledgement-only presentation can present stale GL state (fixed on this branch)

All 3,231 lifecycle `presented` edges in the traces carried
`payload=None` (remap → pool skip → ack), with `vertex_uploads=0` in all
268 commit batches and `shader_uniform_updates=0` throughout — presentation
truth was pure residency-identity equality; no GL command was ever
required to acknowledge. The identity layer excludes levels/LUT/scale by
design and nothing pinned the per-quad `a_mode` buffer, so a physically
divergent visual (stale mode/mapping/uniform) acknowledges cleanly.

Visible symptom pinned: PAL-relaxed `LUT[0] = (249,127,16)` — zero-
magnitude `complex_rg32f` texels render bright orange exactly under stale
`a_mode=3` (component-through-LUT, no magnitude modulation) or stale
`u_component_mode>2.5`; correct phase_color mode renders black. The
per-tile identity probes were fully self-consistent while the screen was
wrong — bookkeeping cannot see below the identity layer.

Fix landed on this branch (extends 5fc43d7a's levels-only physical no-op
check): physical presentation truth in the tile layer — clean re-present
and uniforms-only paths verify per-page shader-mapping key, levels, and
per-quad mode buffer against desired state, repair on divergence, and
never acknowledge from a divergent visual; injected wrong-uniform/mode
tests plus a framebuffer gate (zero-magnitude complex background must
never render LUT[0]).

## Bug 1 update — fresh traces (22:33/22:34) correct the mechanism; branch fix landed

Fresh repros on the CURRENT branch build (includes the commit-progress
watchdog, so genuinely idle): `/tmp/arrayscope-stall-18-1.trace.jsonl`
(scroll down, stall `required_unsettled=[0..16]`, seq 9708) and
`/tmp/arrayscope-stall-65-2.trace.jsonl` (back to 70:170 then scroll up,
stall `required_unsettled=[98,99]`, seq 37585). Direction does not matter.

The claim ledger is NOT the surviving defect on this build: 7ee5c74c's
release/refill held in both traces (every preview fanout released its
claims and armed a replan; candidates re-yielded). The proven cycle is an
**acknowledgement/evidence race around the first-pass histogram barrier**:

1. `retarget_index_window` resets `first_pass_histogram_published`
   (frame_session.py:1062); the scrolled window's first pass is the shared
   L4 preview fanout (`first_pass_quality="preview"`).
2. The DESIRED (L2/L1 exact) shared pass is refused by
   `shared_first_pass_barrier_pending` until the flag is set
   (frame_effects.py:331-335, gate at :892-896) — every later
   `pipeline_plan` shows `submitted=0` (18-1 seq 9299→9706, 65-2 seq
   37172→37579) while per-tile steps are correctly shared-owner-refused.
3. The flag is only set inside an ack commit (`_acknowledge_and_publish`
   :2227-2232), whose flush arm requires `_first_pass_level_evidence_complete`
   AT COMMIT TIME (:2233-2241). In both stalls the LAST ack commit ran
   while the rough-evidence task for the new sources was still in flight
   (18-1: commit_batch 9703 precedes histogram task 228 start 9704; 65-2:
   commit_batch 37576 precedes task 416/417 finishes 37578/37583).
4. When the evidence continuation drained (bridge_drain 9707 / 37584),
   `_maybe_publish_after_level_evidence` (level_stats.py:1334) found no
   parked flush and refused the settled-metadata refresh because
   `_montage_side_work_visible_settled` requires `required_target_settled()`
   — false precisely BY the barred pass in (2). Closed wait cycle, idle
   kernel, watchdog asserts (`flush_pending: false` in both stall dumps).

Slow scrolling "resolves" it because each further retarget produces a new
ack commit that can win the evidence race and arm the flush.

**Fix landed on `codex/gpu-engine` at `fd6b77a6`** ("Arm first-pass
histogram publication when level evidence completes late"):
`_maybe_publish_after_level_evidence` arms the same parked-flush
obligation the ack path arms when it observes a completed, unpublished
first pass; the existing resume path then requests the publication
commit, which sets the flag and replans the shared target (ADR 0053: no
new scheduling, no timers). Tests: deterministic late-evidence unit tests
(tests/window/test_montage_backend.py, fail pre-fix), the dossier's
partial-coverage + dropped-fanout claim-release exit gates
(tests/window/test_montage_lod_residency.py), and an offscreen VisPy
FFT-montage scroll-down-then-up settling regression
(tests/ui/test_montage_scroll_settling.py).

## Queue additions

1. (P9/main) Partial shared-fanout coverage must release/refill claims for
   uncovered tiles; exit gate: injected 5-of-62 fanout coverage converges
   with no stall assertion and no idle kernel while unsettled tiles remain.
   [2026-07-15 late: exit gate now pinned on this branch by
   `test_partial_coverage_shared_fanout_releases_all_claims_and_refills`;
   fresh traces show the surviving live stall was the evidence race above,
   fixed in fd6b77a6 — dedupe both against the P9 record at integration.]

2. (P9/main) Watchdog signature gains a commit-progress term so a live
   22 Hz drain does not assert (done on this branch; port with dedupe).
3. (this branch, done) Physical presentation truth + injected-corruption
   gates as above.
4. (this branch, done — fd6b77a6) First-pass histogram publication
   obligation must survive evidence completing after the last ack commit.

## Bug 1 update — session 50 exact-quality/coarse-LOD candidate hole (fixed 2026-07-16)

Fresh post-evidence-race evidence:
`arrayscope-diagnostics-20260715-235102.jsonl` plus
`/tmp/arrayscope-stall-50-1.trace.jsonl`.  This is a third, distinct shared-
target producer hole.

Session 50 first committed 60 physically valid rows.  Thirteen required rows
(`[33..40,45..49]`) remained acknowledged at L4 while the current demand was
L1.  The final shared-target task (981) selected and fanned out only tiles 81
and 82; after they acknowledged, sequence 36556 still planned the thirteen L1
DESIRED steps but submitted zero work.  The kernel and every evidence/stage
lane were idle, so the commit-progress watchdog correctly fired with
`pending=13`, `dirty=11`, `upserts=11`, `flush_pending=true`, and
`final_commit_pending=true`.

The stranded rows were not native `rendered_tiles`: lifecycle trace rows were
`semantic="planned"`, and the resident-remap path had deliberately removed
their predecessor renderer entries.  They were shared L4 payloads labelled
`quality="exact"` because L4 had satisfied an earlier, coarser demand.  Shared
target admission called `presented_preview_payload()` and therefore rejected
them solely because their historical quality label was not `preview`.  The
ordinary per-tile producer correctly deferred to the non-commuting FFT shared
owner, leaving no producer.

The fix makes shared target candidates follow canonical lifecycle truth:
the tile must have a physically acknowledged current first pixel and
`TileLifecycle.target_unsettled_tiles()` must still name it.  Historical
`preview` versus `exact` labels no longer reinterpret current settlement; an
already exact target row remains excluded.  The trace-shaped mixed regression
pins an L4 preview row and an L4 exact row as candidates while excluding an L2
exact row.  The shared-target/claim slice is 9 passed, the focused render/
window/UI slice is 168 passed, and the real-Wayland
`test_fft_preview_refinement_settles_without_stalls` gate passes.

Rejected path: moving the `rendered_tiles` shortcut did not explain this
trace and would not fix it, because the actual thirteen rows had no renderer
entry and were rejected later by the quality-label check.  Do not add a
watchdog retry or another commit wake; the plan already ran and the missing
fact was candidate ownership.
