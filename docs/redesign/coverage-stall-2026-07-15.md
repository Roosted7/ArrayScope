# Trace-proven: shared-transform coverage stall + acknowledgement-only presentation truth gap (2026-07-15, live session)

**Evidence:** `/tmp/arrayscope-stall-{49-1,54-2,63-3}.trace.jsonl` +
`arrayscope-diagnostics-20260715-195337.jsonl` (user session, 19:53–19:54,
complex128 NIfTI 336×336×272, montage axis 2, phase_color, PAL-relaxed,
VisPy). Build: `codex/gpu-engine` at the pre-P9-rebase state (includes
4646ecf8; predates 1e36084b/eb7f20d2 — verified by absent commit-batch
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
4646ecf8 ("Close shared-transform coverage stalls") was IN the running
build — this is a surviving member of that family at the
partial-fanout-coverage refill edge (frame_effects.py:337 +
`submit_shared_transform_floor` :835).

Secondary: the drain collapsed to one-upsert-per-commit (~12 ms × 22 Hz,
83/88 batches zero-upload) — the diagnostics "UI fan-in" bottleneck;
1e36084b's `batch_limit = max(4, …)` addresses this half.

Also observed: `atomic_source_successor_committed` stale-True in every
stall-49 batch (plain bool surviving `retarget_index_window`) — fixed by
eb7f20d2.

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

Fix landed on this branch (extends 1e36084b's levels-only physical no-op
check): physical presentation truth in the tile layer — clean re-present
and uniforms-only paths verify per-page shader-mapping key, levels, and
per-quad mode buffer against desired state, repair on divergence, and
never acknowledge from a divergent visual; injected wrong-uniform/mode
tests plus a framebuffer gate (zero-magnitude complex background must
never render LUT[0]).

## Queue additions

1. (P9/main) Partial shared-fanout coverage must release/refill claims for
   uncovered tiles; exit gate: injected 5-of-62 fanout coverage converges
   with no stall assertion and no idle kernel while unsettled tiles remain.
2. (P9/main) Watchdog signature gains a commit-progress term so a live
   22 Hz drain does not assert (done on this branch; port with dedupe).
3. (this branch, done) Physical presentation truth + injected-corruption
   gates as above.
