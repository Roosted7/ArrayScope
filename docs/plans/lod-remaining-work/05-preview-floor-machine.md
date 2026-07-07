# Plan 05 — Preview floor through the lifecycle machine (finish the VisPy preview WIP)

**Status:** active (2026-07-06). This plan finishes the uncommitted VisPy preview-floor work by
fixing its root architecture problem first. Read `README.md` ground rules; background is
Plan 04 and ADR 0051.

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
5. **One preview-cache seam.** Either keep `lod_preview_pyramid` but route every admit through
   one session method (`admit_preview_plane(key, plane, histogram)`), or fold the preview tier
   into the main `PyramidCache` as a pinned retention class (`BoundedCache.retention_key`
   exists for exactly this). Prefer the retention class unless measurement says otherwise;
   three call sites currently repeat `preview_cache if … else lod_pyramid`.
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

## Docs afterwards

ADR 0051: note PREVIEW owner + the recorded rule above. Plan 04: mark the VisPy floor slice
landed with numbers. Roadmap X5 queue: advance item 1. Memory: new numbers + tip.
