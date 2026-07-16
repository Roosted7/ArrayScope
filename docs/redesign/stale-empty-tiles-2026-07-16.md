# Stale/empty montage tiles — identity-aliasing starvation stall (2026-07-16)

**Status:** root-caused, fixed, gated. Live repro converges post-fix.

## Field evidence

- `/tmp/arrayscope-stall-148-1.trace.jsonl` / `-2` (user session, 2026-07-16
  08:05, after the G5 commit series landed): session 148, full-axis 272-tile
  complex FFT montage (`CenteredFFT+FFTShift+CenteredIFFT` on axis 2, VisPy,
  resident policy). 91 required tiles unsettled, `dirty=91`, `upserts=91`,
  `evaluating=0`, `pending=0`, `flush_pending=final_commit_pending=True`.
- Tile 31's arc in the ring buffer: **never presented anything** for 7.5 s.
  Its exact level-2 payload was `commit_emitted` ~30 times; every
  `commit_batch` (phase `backend_complete`) reported
  `delta_upserts=[31,30,10,9]`, `committed_upserts=[]`, `uploads=0` — the
  backend rejected the same four payloads at ~25 ms each while the bounded
  admission queue kept re-picking them, starving the other 87.
- 272 − 181 presented = 91 stuck (field); the live repro reproduced
  272 − 160/168 = 112/104 stuck — the arithmetic of "everything that ever
  fails the commit gate stays failed".

## Root cause (proven live, identity diff captured)

`TileIdentity.semantic_generation` folds the tile view state's raw
`axis_range_indices`. An explicit range covering the whole axis
(`(0, 1, …, N-1)`, e.g. from typing `0:336` or programmatic full-range
selection) and `None` are the same semantics but compared unequal. The
montage source keys (`montage_tile_semantic_key`, anchoring content keys)
normalize windows away, so retained/seeded payloads survive the spelling flip
with matching *source ids* but dead *typed identities*:

```
payload.semantic_generation = (shape, slices, ((0,1,…,335), None, None), …)
target .semantic_generation = (shape, slices, (None,      None, None), …)
```

`satisfies_target` then fails forever. Three compounding design gaps turned
one aliased payload into a whole-montage stall:

1. **Silent rejection.** `update_payloads` dropped non-satisfying upserts
   with no counter, no trace, not even `items_skipped`.
2. **Bookkeeping counted as coverage.** The shared-transform preview pass
   skipped any tile holding a `quality="preview"` payload at the preview
   level — even one the backend can never acknowledge — so no producer ever
   regenerated it.
3. **Barrier deadlock.** `shared_first_pass_barrier_pending` waits for
   required first pixels; the rejected tiles were exactly the missing first
   pixels, so the desired-level producer stayed barred while per-tile DESIRED
   rungs are (by design) denied for shared-transform pipelines
   (`_shared_transform_owns_display_target`). Result: replan livelock
   (`pipeline_plan` every ~50 ms, `submitted=0`, endless `frame-admission`
   continuations) with the kernel idle.

**Attribution:** the stall reproduces at the pre-series base `2d0f605f` too —
the G5/prefetch/retention series did **not** introduce the root defect; it
changed exposure (more retained/warm coverage surviving transitions, so the
spelling flip had more corpses to leave behind).

## Reproduction

- Live (stalls pre-fix 3/3, converges post-fix): seeded ~12 s gesture mix on
  the real NIfTI — zoom glides, axis-0 range windows **including an explicit
  full range and its `None` reset**, montage slice pokes — then idle.
  Preserved as `tests/stress/test_interaction_convergence.py`
  (`ARRAYSCOPE_STRESS=1`, real display, serial).
- Offscreen does NOT reproduce the scheduling shape (consistent with the
  2026-07 course-reset ground rules); the deterministic gates are unit-level.

## Fixes

1. **Canonical spelling (root).** `ViewState.__post_init__` normalizes an
   explicit full-coverage `axis_range_indices[axis]` (and its text) to
   `None`. One spelling of "whole axis" everywhere, by construction.
   Gate: `tests/core/test_view_state.py::test_full_coverage_axis_range_canonicalizes_to_none`.
2. **Dead payloads are missing coverage (self-heal).**
   `render.effects.payload_identity_dead` + the shared-transform preview pass
   treats an unpresented payload whose typed identity cannot satisfy the
   tile's current lifecycle target as *missing* and regenerates it. Any
   future identity-aliasing bug now degrades to one wasted evaluation
   instead of a permanent stall.
   Gate: `tests/window/test_montage_lod_residency.py::test_unacknowledgeable_payload_counts_as_missing_shared_coverage`.
3. **Loud rejection.** `update_payloads` reports
   `identity_rejected_items`/`identity_rejected_tiles` in
   `TileLayerUpdateStats`, carried into `TileCommitReport` and the
   `commit_batch` trace (`identity_rejected=`). A re-emit loop is now visible
   on the first commit of any future trace.
   Gate: `tests/display/test_vispy_physical_presentation.py::test_identity_rejected_upserts_are_reported_not_silent`.

## Rejected shortcuts

- Normalizing only inside `_tile_identity` (leaves every other ViewState
  consumer aliased; session keys would keep churning).
- Dropping rejected payloads from the retained store at commit time
  (fights retain-until-replace; the transient "target changed, successor in
  flight" case would blank correct pixels).
- Accepting opaque source-id equality at the commit gate when identities
  mismatch (reintroduces the R8A aliasing the typed contract exists to stop).
- A watchdog that "repairs" the loop (V3 rule: watchdogs observe, owners fix).

## Same-day follow-up: transient orange tiles during fill (09:14 field report)

After the fixes above, the user reported ~4-tile clusters rendering
PAL-relaxed orange (full-bright phase color over zero magnitude) transiently
while a complex FFT montage loads, strongest when changing the montage index
window. Framebuffer probe (`probe_orange.py`, Wayland, real data) reproduced
it deterministically on a shrink/grow gesture: 54 orange frames, up to
19,300 orange pixels; the orange pages carried per-quad ``a_mode=3``
(magnitude through the cyclic LUT) with ``physical_page_divergences == {}``
— wrong **desired** state, invisible to the audit by construction.

Root cause: ``ensure_floor_payloads`` presented resident complex floor
planes with ``shader_mapping=None`` because ``lod_preview_metadata`` is
per-session while the pyramid cache persists — entering tiles of a new
montage window reuse the previous session's planes without their mappings,
and ``_payload_mode`` maps an unmapped COMPLEX_RG32F payload to mode 3.

Fix: the mapping is a pure function of the current view state;
``display.slice_engine.complex_texture_shader_mapping`` mints it and the
floor builder uses it whenever the metadata mapping is gone. Probe post-fix:
1 residual frame of 605 scattered pixels (legitimate near-orange phase
content), zero solid tile blocks. Gates:
``tests/window/test_floor_payload_mapping.py``.

The same field session's stall (``/tmp/arrayscope-stall-1-1.trace.jsonl``:
36 ``pending_tiles``, 55 required-unsettled, kernel idle ~3 s during an
interactive fill) belongs to a **deferred-stage-planning lost-wakeup family**,
three members of which are now fixed (churn-harness proven, each moved the
stall to the next member):

1. The interaction-quiet falling edge only replanned when
   ``interactive_native_deferred > 0``; interactive montage retargets defer
   STAGE planning (``stage_planning_deferred`` + tiles parked in
   ``pending_tiles``) without touching that counter, so the deferred plan had
   no owner after the gesture ended. ``replan_deferred_interactive_native_
   quality`` now also triggers on a deferred (non-async) stage plan.
2. ``submit_deferred_stage_fan_in_plan``'s ``done`` callback bailed on a
   stale render generation WITHOUT clearing ``stage_planning_async`` — a
   phantom in-flight planner that ``complete_deferred_stage_fan_in`` deferred
   to forever. The bail now clears the flag and hands ownership back to
   ``retarget_frame_pipeline``.
3. Bounded commits can consume ``flush_pending``/``final_commit_pending``
   while payload upserts remain queued; ``retarget_frame_pipeline`` now
   treats non-empty ``pending_payload_upserts`` as its own flush obligation.

**Still open** (xfail net:
``tests/stress/test_interaction_convergence.py::test_interaction_churn_converges_on_real_data``):
after ~12 s of extreme seeded churn, a compound state remains where ~200
pending upserts stay queued although ``apply_ready_montage_display`` runs
with ``flush=True``, lane quotas are healthy (preview/preparation 7),
``presentationDrawPending`` is False, and a fresh deferred stage plan exists
— the commit path itself bails somewhere between ``apply_montage_
presentation`` and the backend. Next lead: instrument the commit chain's
early returns the way the identity gate was instrumented (loud bail
reasons), then bisect the churn script down to the minimal gesture pair.
The realistic field gestures (fill, window shrink/grow, range flips at human
pace) all converge with the fixes above.

## Follow-ups (all landed 2026-07-16, each repro-first gated)

- `montage_indices` had the same two-spellings hazard (explicit full tuple vs
  `None`). It does not enter `semantic_generation`, only session keys
  (session churn, not correctness). `ViewState.__post_init__` now
  canonicalizes a full-coverage tuple (and `montage_text`) to `None`.
  Gate: `tests/core/test_view_state.py::test_full_coverage_montage_indices_canonicalize_to_none`.
- The per-tile analog of gap 2: `_display_payload_covers_display_target`
  (prepare_rung) trusted payload currency without checking typed-target
  satisfiability; non-shared pipelines could starve the same way. It now
  additionally requires `acknowledged_identity_satisfies_target(payload,
  lifecycle target)` — currency is not satisfiability.
  Gate: `tests/window/test_montage_lod_residency.py::test_dead_identity_display_payload_does_not_cover_direct_tile_target`.
- The presenter re-emitted an unchanged rejected delta at full flush rate
  (~25 ms of wasted geometry sync per cycle). `_finish_commit` now signs an
  all-identity-rejected delta (payload ack identity × delta target identity
  per tile); an identical repeat emits a loud `identity_rejected_recommit`
  trace, stops the flush re-arm from exactly those tiles (both the
  commit-side backlog check and `FrameSession.note_committed`), and requests
  a replan so producers own recovery. One retry is allowed (a retarget can
  race a commit); any payload or target change re-arms normally.
  Gate: `tests/window/test_montage_lod_residency.py::test_identical_identity_rejected_delta_recommit_is_bounded`.
- `trace_verify` now fails any whole-workflow replay whose
  `commit_batch`/`backend_complete` events carry a non-empty
  `identity_rejected` (invariant `no_identity_rejected_commits`), even when
  every target eventually settles.
  Gate: `tests/core/test_trace.py::test_trace_verify_rejects_identity_rejected_commits`.
- The pyqtgraph backend had the silent form of the same rejection: its
  drawable-payload filter dropped identity-dead upserts without filling
  `identity_rejected_items`/`identity_rejected_tiles`, so a dead-payload
  re-emit loop on that backend was invisible to the commit_batch trace,
  immune to the presenter backoff, and undetectable by trace_verify. The
  filter now counts rejected **delta upserts** (retained non-upsert payloads
  a newer target has outrun are deliberately excluded — the presenter is not
  looping on them, and counting them would false-trip the backoff and the
  trace invariant).
  Gate: `tests/display/test_pyqtgraph_physical_presentation.py::test_identity_rejected_upserts_are_reported_not_silent`.
