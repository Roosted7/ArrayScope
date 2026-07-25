# PyQtGraph cold_fill demand-freshness red — real latency at the viewport bridge (2026-07-19)

**Status:** adjudicated REAL product latency (not a sampler gap), fixed at the
owner, closed with matrix evidence. This was the last standing journey-matrix
red after the 2026-07-19 perf landing (14/15, wgpu 5/5, vispy 5/5).

## Adjudication method and verdict

The failing oracle was `demand_fresh_within_budget`: the first timeline sample
at which `camera_desired_level == session_desired_level == final camera level`,
measured from gesture start (budget 5 000 ms). Recorded reds: 5 250 ms (perf
run), 5 502 ms (blackoutfix run), 5 400–6 300 ms (v1/v2/v3/v7 incumbents).
First pixels (353–360 ms) and level convergence were green throughout — the
"first-pixel latency" framing of the standing note did not survive the trace
walk.

Two stacked components, separated by walking the trace against the sampled
timeline (the trace previously had **no** event for the demand transition; it
had to be bracketed between a `commit_batch` carrying `desired_level` and the
`scheduling_phase scope_started` of the next generation):

1. **Real latency (dominant):** the session's LOD demand only adopted the
   live camera demand at ~4.9 s (perf run) / ~5.1 s (blackoutfix run) —
   straddling the budget, run-to-run flaky.
2. **Sampler starvation (secondary):** the confirming sample landed 350–800 ms
   after the transition because the GUI thread was inside the 272-tile
   replan + evidence burst. Not fixed by itself — with component 1 fixed the
   margin makes cadence irrelevant. The oracle was not touched.

## Mechanism (offscreen repro, instrumented)

`profile_montage_workflow --backend pyqtgraph --stages raw_full_tiled_montage`
reproduces the whole chain deterministically offscreen:

- ~170 ms after montage entry the product's auto-fit moves the live camera to
  the full 272-tile extent. `ViewportBridge.on_view_range_changed` fires —
  and **drops the intent**: `_owner_has_tiled_scene` is false because the
  committed display frame is still the (non-tiled) predecessor that the
  montage-axis bridge deliberately retains. This is the documented
  "AUTO-camera dead gesture edge" in its live-path form.
- `session.view_range` — and therefore LOD demand (`selected_lod_factor`),
  the lifecycle scope, and fill priorities — stays at the stale entry fit.
- The rescue was the profile driver's `_pulse_fit_stretch` explicit retarget,
  which under load is queued behind the level-1 fill's GUI work: 1.3 s
  offscreen, 4.9–5.8 s under full-matrix Wayland load. In the field nothing
  rescues it. Demand flips within 1 ms of the retarget — the recompute was
  never the problem; the missing retarget was.

All three backends shared the shape (flip exactly when the driver pulse
drains, i.e. at their gen-1 fill completion: wgpu 1.1 s, vispy 3.9 s,
pyqtgraph 4.9 s); only pyqtgraph's fill was slow enough to straddle the
budget.

## Fixes (commit `6fd0c262`, each at its owner)

- **`ViewportBridge`** (`arrayscope/window/viewport_bridge.py`): a
  montage-axis range change without a committed tiled frame records the
  existing replay obligation (`_frame_viewport_retarget_after_commit`)
  instead of being dropped. Camera intent is never lost; nothing replans
  against uncommitted state.
- **`_finish_presentation_commit`** (`arrayscope/window/frame_effects.py`):
  the replay holds while the scheduling policy's coverage pass is open —
  phase 1 fully completes before the camera rescope supersedes the entry
  scope (two-phase contract). Replaying at the *first* commit teardown was
  tried and rejected: it perturbed entry choreography (wgpu first pixels
  pushed over budget offscreen; pyqtgraph tail flake aggravated). With the
  gate, pyqtgraph demand flips ~80 ms after coverage close: 1.9 s offscreen,
  3.2–3.4 s under full-matrix Wayland load.
- **`replan_deferred_interactive_native_quality`**
  (`arrayscope/window/frame_runtime.py`): pumps the wgpu resident-histogram
  evidence queue directly at the interaction-settle edge. The earlier rescope
  exposed a real ownership hole: an `interaction_active`-deferred evidence
  dispatch at the fill tail has no further commits to re-own it (the pump
  lived only in the commit-ack path), so the coverage evidence barrier
  latched with an idle kernel — deterministic offscreen wgpu cold-fill stall
  (2-for-2 red before, 2-for-2 green after; the forced commit alone never
  reaches the ack path when nothing is dirty).
- **`selected_lod_factor`** (`arrayscope/render/lod.py`): permanent
  `lod_demand` trace per desired-level transition, so future freshness
  adjudications read the ground-truth flip timestamp instead of bracketing.

Red-first pins: `test_uncommitted_montage_range_defers_retarget_to_commit_teardown`
(fails on the unfixed bridge), `test_presentation_commit_holds_camera_replay_while_coverage_is_open`,
extended `test_interaction_stop_rearms_deferred_wgpu_histogram_evidence`
(asserts the direct queue pump).

## Non-regressions proven pre-existing (baseline A/B on unfixed main)

- **pyqtgraph cold tail stall under screenshot-flag load** (offscreen, matrix
  driver flags `--screenshot-interval-s 0.1 --timeout-s 5`): unfixed main
  `b7e94879` stalls 1-of-2 with the same frozen-tail signature
  (`level_stale=111`, planned-but-unsubmitted level-2 steps, armed
  presentation gate). The known tile-limbo/levels-tail family; offscreen
  only — the real-Wayland driver rows complete.
- **pyqtgraph zoom_in unbounded-preview-commit flake**: acceptance run v2's
  single red (`bounded_failures`, an early 4-tile preview commit with
  `max_upserts=0`, empty reason) is byte-identical in shape to committed
  incumbent evidence on unfixed trees: `journey-matrix-wgpu-2026-07-18-v19`
  (`ok: false`, same oracle, 12-tile variant at 459 ms) and
  `-v11` (12 @ 555 ms), `rebased-level-identity-final-2026-07-17`
  (59 @ 742 ms). Intermittent, pre-existing, unrelated to this change; the
  ungoverned early preview commit deserves its own owner.

## Follow-up fixes landed during acceptance

- **Ungoverned floor-progress commit closed** (`b30d9940`): the zoom_in
  flake above turned out to be systematic, not racy —
  `tile_layer_upsert_limits` returned `{}` whenever no dirty/pending work
  existed at decision time, but a floor-progress commit materializes
  frontier preview upserts during assembly. The gate now also computes
  limits while unsettled required targets exist. Red-first:
  `test_pyqtgraph_floor_progress_commits_stay_governed`. zoom_in green in
  every subsequent run.
- **Oracle precision** (`f2dbd556`): v6 re-red at 5 178 ms decomposed (via
  the new `lod_demand` trace) into transition 4 276 ms — in budget — plus
  902 ms sampler starvation during the gen-2 replan burst. The oracle now
  uses the transition timestamp as ground truth, honored only when a later
  sample confirms the fresh state stuck. Fault-injection pins: an injected
  transition with no confirming sample stays red; a late transition
  carries a late timestamp. Sample-only artifacts evaluate as before.

## Evidence

- Full offscreen suite at `f2dbd556` (on `b7e94879`): **2488 passed,
  0 failed**, 36 skipped, 1 xfailed.
- Seven real-Wayland full matrices
  (`tests/artifacts/journey-matrix-2026-07-19-v1…v7`), all re-verified
  with the final oracle: v1 **15/15**, v2/v3/v4 14/15 (single red = the
  ungoverned zoom_in commit, pre-dating its fix `b30d9940`; identical
  committed incumbent signature in `journey-matrix-wgpu-2026-07-18-v19`/
  `-v11`), v5 **15/15**, v6 **15/15**, v7 **15/15** — three consecutive
  full-matrix 15/15 runs (v5–v7) executed with the full fix stack. pyqtgraph cold_fill green in every run:
  ground-truth demand freshness 2 601–4 276 ms (was 5 250–6 300 sampled),
  first pixels 337–360 ms. vispy cold margin widened to 1 834–2 975 ms
  (was ~4 728 — one load spike from the same red).
- Offscreen repro: pyqtgraph 272/272 PASS, demand flip 1.9 s; wgpu 272/272
  PASS (stall fixed).
- Residual lane, honestly stated: the transition time equals gen-1
  coverage close (+ one commit teardown), which is pyqtgraph fill-speed
  dependent (2.6 s typical, 4.3 s under heavy load). Bringing that tail
  further down belongs to the fill/perf-bars program, not to freshness
  plumbing.
