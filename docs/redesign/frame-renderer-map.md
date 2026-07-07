# frame_renderer.py dissolution map

`window/frame_renderer.py` (6,100 lines, ~150 methods on `FrameRenderMixin`)
and `window/montage_lod.py` (1,250 lines of functions taking "the renderer
host") dissolve into the modular pipeline. This table is the port order for
R2/R3: work cluster by cluster, delete the source methods in the same commit
that lands their replacement, and never port a pacing workaround whose job
the kernel already does (mark it `DELETED — kernel` below and remove its
tests in the same commit).

Destination legend:
`ladder` = `render/ladder.py` · `pipeline` = `render/pipeline.py` ·
`effects` = the R2 `PipelineEffects` implementation (Qt-free evaluation
functions + backend commit application) · `lifecycle` =
`presentation/tile_lifecycle.py` · `viewplan` = `window/montage_viewport.py`
(already Qt-free) · `kernel` = obsolete, the kernel/bridge does this
structurally.

## Cluster A — viewport planning and retargeting (mostly keep, relocate)

| methods (frame_renderer) | destination |
|---|---|
| `_montage_viewport_plan`, `_effective_montage_columns`, `_retargeted_montage_viewport_plan` | `viewplan` (pure functions; drop the mixin-host parameter) |
| `_on_image_viewport_resized`, `_retarget_montage_resize_camera/_payloads`, `_commit_montage_resize_presentation_retarget` | `pipeline.retarget` + `viewplan`; camera math stays in `display/viewport.py` |
| `_maybe_auto_fit_montage_tiles`, `_publish_montage_content_extent`, `_set_montage_view_range`, revert/undo helpers | thin GUI controller (`window/`), driven by plan-time data only (see commit a3992c8f for the plan-time principle) |
| `_run_montage_viewport_update`, `_schedule_montage_viewport_update` (+delay) | `pipeline.retarget`; the deferral timer is DELETED — kernel (submit is cheap, supersession dedupes) |
| `_schedule_montage_priority_retarget_from_hover`, `_refresh_montage_priority_targets`, `_run_montage_priority_retarget` (+delay/batch limits) | kernel priorities on resubmission; DELETED — kernel (priority rebuild timers) |

## Cluster B — tile materialization and previews (port to effects + ladder)

| methods | destination |
|---|---|
| `_schedule_montage_tiles`, `_schedule_next_montage_tile`, `_evaluate_montage_tile_snapshot` | `effects.evaluate_rung` (EXACT rung); scheduling loop DELETED — kernel |
| `_schedule_montage_preview_tile`, `_schedule_montage_shared_preview_batch`, `_evaluate_montage_tile_preview_snapshot`, `_evaluate_montage_shared_preview_snapshot`, `_evaluate_montage_tile_native_output_preview_snapshot`, `_read_reduced_preview_base_and_state`, `_reduced_preview_view_state`, reduce helpers (`_reduce_*`), preview region helpers (`_axis_region_for_preview_indices` …) | `effects.evaluate_rung` (PREVIEW/FLOOR rungs), Qt-free module next to `operations/`; rung choice + usefulness checks (`_preview_is_useful_for_current_scheduler`, `_can_evaluate_*`) become `ladder` inputs |
| `_preview_floor_blocks_exact_submission`, `_claim_preview_floor`, floor helpers in `montage_lod` (`best_floor_key`, `floor_can_progress`, `ensure_floor_payloads`) | `ladder` (FLOOR rung planning) + `lifecycle` claims; blocking-exact logic is the ladder's coarse-before-fine invariant |
| `montage_lod.plan_materialization`, `refresh_lod_for_viewport`, `admit_ingest_reduction`, `admit_preview_reduction`, `schedule_materializations`, `retry_blocked_materializations`, `on_level_ready` | `ladder.plan` + `pipeline._submit_step` + lifecycle events; per-request claim plumbing (`_release_request_claims` etc.) stays with `lifecycle`/`PyramidCache` |
| `_store_reusable_montage_tile_result`, `_rendered_tile_from_*` | `effects` (payload adaptation, Qt-free) |

## Cluster C — stage planning and fan-in (port, then delete the pumps)

| methods | destination |
|---|---|
| `_plan_montage_stages`, `_schedule_montage_stage_jobs`, `_merge_montage_stage_plan` | kernel tasks with `deps=` (stage → dependent tiles); singleflight via task keys |
| `_schedule_deferred_montage_planning`, `_complete_deferred_montage_planning`, `_plan_deferred_montage_stages_now` | DELETED — kernel (planning is a task; deferral timers gone) |
| `_on_montage_stage_done/_stale/_error`, `_activate_montage_stage_value`, `_release_stage_waiting_tiles_to_direct`, `_schedule/_process_montage_attached_stage_waits`, `_activate_cached_waiting_stages`, `_activate_or_release_waiting_stage`, `_stage_wait_has_actionable_work` | kernel dependency completion + `lifecycle` events; the wait-pump timer family is DELETED — kernel (deps + capacity waiters) |

## Cluster D — level stats and histograms (move off the GUI thread)

| methods | destination |
|---|---|
| `_montage_level_*` (keys, stats, bounds, tracker, coverage rank ~15 methods), `_schedule/_queue/_process_montage_cached_level_stats`, `_scan_montage_level_stats_from_session`, `_schedule/_on_montage_refined_level_stats` | Qt-free `LevelStatsService` (new, `render/`): scans/refines run as `HISTOGRAM_REFINEMENT` kernel tasks; preview-derived samples stay authoritative until the refinement rung lands (ladder invariant 3). GUI applies published level metadata only |
| `_should_publish_montage_level_metadata`, `_note_montage_level_source_applied`, `_session_requested_levels`, `_tile_layer_auto_levels_wait_for_complete_source`, `_montage_level_evidence_requires_refined` | `LevelStatsService` policy, pinned by its own unit tests |

## Cluster E — presentation commit (port batching; ack already machine-owned)

| methods | destination |
|---|---|
| `_schedule_montage_presentation_commit`, `_start_montage_commit_timer`, `_montage_commit_interval_ms`, `_flush_montage_presentation_commit`, `_queue_montage_presentation_commit_flush` | `pipeline._flush_ready` (bounded per drain); commit-interval timers DELETED — kernel |
| `_commit_montage_session_presentation`, `_commit_montage_session_tile_layer` (435 lines!), `_commit_montage_tile_delta_direct`, `_direct_montage_tile_layer_presentation`, `_accepted_tiled_payloads`, upsert/batch limit functions (`_tile_layer_upsert_limits`, `_persistent_tile_*`, `*_upload_nbytes`) | `effects.apply_commit` consuming `CommitBatch`; split CPU-item vs GPU-atlas mechanics into the two backend adapters behind capabilities |
| `_flush_montage_tile_results`, `_apply_montage_tile_result`, `_schedule_montage_tile_result_flush`, `_on_montage_tile_done/_error/_slow` | kernel bridge drain + `pipeline._on_rung_done`; result-flush timers DELETED — kernel |
| `_montage_watchdog_*` | keep ONLY as an assertion probe (ADR 0051: every rescue is a bug report); rehome beside the pipeline |
| loading overlay / slow-timer family (`_ensure_montage_session_slow_timer` …) | thin GUI controller; `on_slow` equivalent comes from a kernel deadline event, not its own timer |

## Cluster F — budgets, pacing, governor plumbing

| methods | destination |
|---|---|
| `_montage_callback_budget`, `_record_gui_budget`, `_montage_tile_result_batch_limit`, `_montage_viewport_addition_batch_limit`, `_montage_viewport_chunk_delay_ms`, `_presentation_upload_control_budget_ms`, `_tile_presentation_ui_work_decision`, `_montage_commit_budget_ms` | mostly DELETED — kernel (bridge budget + lane quotas). The governor shrinks to telemetry + two knobs: bridge drain budget and commit batch bounds (R4; ADR 0052 G1/G2 re-scoped to those channels) |

## Cluster G — session identity and misc

| methods | destination |
|---|---|
| `_montage_session_key_for_view`, `_is_current_montage_session`, `_montage_session_is_current`, `_maybe_retarget_montage_session` | `RenderIntent` keys + kernel scopes; session reuse/retarget stays in `montage_session.py` (unchanged by R2) |
| hover/profile retry helpers, `_retry_live_profile_after_montage_tile` | capacity waiters on the bridge |
| module-level geometry helpers (`_montage_full_view_range`, `_visible_montage_tile_count`, `_view_range_contains*`, autofit signatures) | `viewplan` |

## Port rules

1. One cluster (or sub-row) per commit; suite + GPU harness after each.
2. The replacement lands WITH the deletion of the old methods and their
   pacing tests. No dual paths, no compatibility shims.
3. Anything found only feeding a deleted pacing mechanism is deleted, not
   ported. When unsure, check whether the kernel already guarantees it
   (staleness, ordering, bounds, wakeups) — usually yes.
