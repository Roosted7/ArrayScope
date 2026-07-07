# Plan 05 — Preview floor through the lifecycle machine (finish the VisPy preview WIP)

**Status:** landed in code (2026-07-07), validation/docs cleanup in progress. This plan finished
the uncommitted VisPy preview-floor work by fixing its root architecture problem first. Read
`README.md` ground rules; background is Plan 04 and ADR 0051.

## The root cause to fix

The WIP introduces session-side counters — `lod_preview_floor_target_count` /
`lod_preview_floor_level`, `begin_lod_preview_floor()`, `_lod_preview_floor_first_fill_active()`,
`_finish_lod_preview_floor_if_complete()` — to express "present the whole viewport at the preview
level before spending upload budget on exact tiles". This is the exact defect class ADR 0051
exists to forbid: optimistic bookkeeping whose completion is checked only on unrelated events.

Concretely: `begin` only ratchets the target up; completion is evaluated only from
`set_active_tiles` and `build_tile_presentation`; there is **no terminal path** for a preview
evaluation that fails, is declined, goes stale, or for a retarget that shrinks the plan. One
lost preview ⇒ floor-first-fill mode never ends ⇒ exact refinement deferred forever. That is a
wedge by construction, and it cannot be seen by the watchdog (the counters are invisible to the
dispatch signature).

**Rule to record and follow: any counter that must be "finished" by a later event belongs in
the lifecycle records or the dispatch derivation, never on the session.**

## The design

Preview-plane availability is residency-axis data; the machine already owns it since P3.

1. **Claims, not counters.** When preview work is scheduled (per-tile or shared batch), record
   `lifecycle.level_claimed(tile, preview_level_key, owner=PREVIEW, request=batch)` for every
   covered tile (add `ClaimOwner.PREVIEW`). `materialization_started/resident/released` then
   give every terminal path for free: worker failure, admission decline, supersession, and
   `session_replaced` all release mechanically (P3 machinery), and a leak is visible in
   `dangling_claims()`.
2. **Derive the phase; store nothing.** Floor-first-fill is active for a planned tile set iff
   some planned unpresented tile has a preview-level entry in {CLAIMED, MATERIALIZING} or a
   RESIDENT preview entry it has not yet presented. A failed preview releases its claim and the
   tile simply stops holding the gate. Delete `lod_preview_floor_target_count`/`_level`,
   `begin_/…_if_complete` once the derivation is in.
3. **Floor key from records.** `best_floor_key` currently scans two caches for the floor-level
   key. The RESIDENT preview entry in `TileRecord.levels` *is* that key — consult it first and
   drop the dual-cache scan.
4. **Refinement is a dispatch fact.** "Preview presented + exact result rendered" must imply a
   dirty payload inside `derive_montage_dispatch` (rule 6: the evidence has a consumer at
   arrival), replacing the loop in `_finish_lod_preview_floor_if_complete` that marks tiles
   dirty. `has_unrefined_preview_payloads()` should read `tile_presentation_state` only.
5. **One preview-cache seam.** Keep `lod_preview_pyramid` for the floor tier, but route every
   admit through one session method (`admit_preview_plane(key, plane, histogram, metadata...)`).
   Preview display metadata (`shader_mapping`, `texture_kind`, `level_data`) is part of that
   seam: losing it made complex/phase-color preview floors render with the wrong colors even
   though the reduced plane itself was correct.
6. **Name the policy constant.** `preview_level_for_tile_shape(…, min_level=4)` buries policy
   in a call site. Hoist `PREVIEW_FLOOR_MIN_LEVEL` next to the LOD policy constants with a
   comment stating the evidence (or "unmeasured — benchmark levels 3/4/5" until it is).

## Steps

0. **Restore a green base first.** The pacing commit ("Stabilize PyQtGraph LOD tile pacing")
   landed with three suite regressions (bisected 2026-07-06; every earlier commit passes).
   One is resolved (resident-retarget admission — backend-truthful `free_fn` /
   `pace_resident_retargets`, see Plan 04 conclusions). Two remain open on the tip and must
   be fixed before benchmarking anything:
   `tests/ui/test_montage_interactions.py::test_tile_layer_range_scroll_commits_loading_presentation_before_tiles_are_ready`
   (first post-scroll commit now uploads 2 visible items instead of loading placeholders) and
   `tests/ui/test_roi_inspection_interactions.py::test_hidden_montage_roi_overlay_does_not_sample_loading_placeholder`.
   Both point at the same pacing-commit change to first-commit/placeholder presentation
   semantics. Process note: run the full `-n 16` suite before each commit — all three would
   have been caught.
1. Add `ClaimOwner.PREVIEW`; wire claims in `_schedule_montage_preview_tile` and the shared
   batch scheduler (claim at schedule time, `materialization_started` at evaluate, resident on
   admit, released on error/stale/decline — mirror `schedule_materializations`).
2. Implement the derivation (step 2 above) as a lifecycle/session query; switch
   `build_tile_presentation` to it; delete the counters and their methods.
3. Move the refinement kick into `derive_montage_dispatch` (step 4).
4. `best_floor_key` from records (step 3); collapse the preview-cache seam (step 5); hoist the
   constant (step 6).
5. Keep the already-fixed admission-queue invariant (zero-cost items are exempt from upload
   caps — `tile_admission.py`, regression from the pacing commit) and its test.

## Verification

1. Unit: preview claim released on evaluation error / decline / stale / `session_replaced`;
   floor-first-fill derivation goes inactive when the last pending preview claim releases
   (the wedge test the WIP lacks); `dangling_claims()` empty after settle with previews on.
2. Existing WIP tests keep passing (floor batch cap, preview-before-exact ordering,
   preview-floor completion → refinement, debug pass marker, cold-start release,
   `test_resident_retarget_upserts_bypass_cold_priority_cap`).
3. Suite `-n 16`; GPU harness `stall_repairs==0`.
4. Bench loop (Thomas, iterative): the Plan 04 §Verification VisPy runs with
   `--screenshot-dir` + `ARRAYSCOPE_LOD_DEBUG_PASS_MARKER=final-mirror-x`; watch
   `first_preview_floor_fill_ms`, first-complete-fill, settled, heartbeat p95; screenshots must
   show the un-mirrored preview pass first, mirrored exact after.

## 2026-07-07 findings

- Fixed the placeholder regressions called out in step 0 before benchmarking.
- The preview floor is now lifecycle-owned: `ClaimOwner.PREVIEW` records replace the old
  `lod_preview_floor_*` counters, floor-first-fill is derived from lifecycle records, and
  `derive_montage_dispatch` marks preview-exact refinement dirty.
- Preview planes enter through one session seam and carry display metadata. This fixed the
  observed complex FFT preview colormap bug: the plane was correct, but the floor payload had
  lost the phase-color shader mapping and provisional level sample.
- Preview is now queued before exact materialization for cold tiles. Exact still wins if it is
  already resident/evaluated; preview only wins over blank/placeholder.
- VisPy scalar/no-op montage also uses preview when preview LOD is coarser than the target LOD,
  because the reduced first pass lowers GPU upload/commit pressure even without operations.
  Native-scale or equal-preview-level cases skip preview.
- The shared preview evaluator was consolidated after a scalar bug: reusing the non-display
  transform path for scalar/no-op preview originally fanned out the seed slice to every montage
  tile and lost provisional level samples. Shared preview now reads all candidate source indices
  when the pipeline commutes for display LOD, maps each tile to its own reduced slice, and emits
  the same shader/level metadata contract as per-tile preview.
- Exact target work is gated behind the preview floor so FFT/ops do not compete with the first
  visible preview fill. Once the floor is visible, exact admission continues through the normal
  controller/feedback path rather than serializing one tile at a time.
- Pacing reset ownership was split: `TileLifecycle.feedback_signature(...)` owns the lifecycle
  phase/class part of the signature, the tile presentation model owns physical texture/cost
  shape, and `ResourceGovernor` owns detecting opaque signature changes, resetting that channel,
  and fast relearning. The lifecycle signature deliberately ignores progress counts; including
  counts caused reset churn during FFT exact fill.

Visible Wayland evidence (battery state not controlled; do not treat as a backend A/B):

- `tests/artifacts/x5-item1-preview-metadata-vispy.jsonl`: FFT preview floor fully filled at
  ~1.33 s, final exact at ~5.42 s; screenshot
  `tests/artifacts/x5-item1-preview-metadata-screenshots/vispy-fft_full_tiled_montage_preview_floor.png`
  shows the corrected phase-color preview.
- `tests/artifacts/x5-item1-refinement-rearm-vispy.jsonl`: scalar/no-op preview floor fully
  filled at ~1.92 s and final exact at ~1.94 s (`final_exact_payload_count=272`,
  `final_preview_payload_count=0`); FFT preview floor at ~1.17 s and final exact at ~4.64 s.
- `tests/artifacts/x5-item1-preview-unified-screenshot-vispy.jsonl`: after the scalar shared
  preview fix, scalar preview floor was visible and filled at ~0.75 s, scalar final at ~4.02 s;
  FFT preview floor at ~0.98 s, final at ~5.40 s. The screenshots captured the floor event but
  can include early mirrored exact tiles because draw synchronization itself lets refinement
  progress.
- `tests/artifacts/x5-item1-preview-lifecycle-feedback2-vispy.jsonl`: after lifecycle-owned
  feedback signatures stopped resetting on progress-count changes, scalar preview floor filled
  at ~1.82 s and final at ~4.45 s; FFT preview floor filled at ~1.04 s and final at ~6.11 s.
  FFT exact remained stage-backed (`montage_tile_compute_stage_backed=272`, stage-backed total
  ~1.35 s), so the remaining tail is stage/fan-in/presentation pacing rather than repeated FFTs
  per tile.

Follow-up that belongs to roadmap item 2, not this plan:

- Unify preview and desired-LOD compute as one reduced-input ladder: preview LOD 4 is the first
  display rung, desired LOD is the refinement rung, and both should use the same operation-aware
  reduced-input machinery where operations permit it. Operation work should run once per LOD
  rung and feed every tile/presentation at that rung. Today desired LOD and preview LOD still
  travel partly separate paths.
- Consider a `stage_lifecycle` for the unified ladder. Stage cache/fan-in already has the same
  shape as tile levels — claimed, materializing, resident, served, released, failed/stale — and
  desired-LOD operation outputs will need one authoritative owner when they stop using full-size
  operation inputs.
- Reuse preview-derived level samples for the first display and avoid per-tile exact level churn;
  after the desired LOD is fully shown, sample higher-quality histogram/levels once as a
  coordinated refinement.
- Add a lower-priority preview/offscreen warming queue that uses retained preview pyramid planes
  to pre-upload nearby/not-yet-visible tiles after visible work settles. This should not replace
  the stage cache; stage intermediates remain the reusable operation cache, while
  `lod_preview_pyramid` is the retained display-preview tier.

## Docs afterwards

ADR 0051: PREVIEW owner and lifecycle-signature rule are recorded in code/tests; consider a
short ADR addendum only if `stage_lifecycle` becomes a durable API. Plan 04: mark the VisPy
floor slice landed with numbers. Roadmap X5 queue: item 1 is done; item 2 is next.
