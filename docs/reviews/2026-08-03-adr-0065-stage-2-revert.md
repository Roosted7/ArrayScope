# ADR 0065 stage 2 — revert record and trace-4 diagnosis

2026-08-03. Branch `claude/ui-performance-60-144hz-793f93`. Reverted `arrayscope/` and `tests/`
from `8c69b361` to `b0312b70`; `docs/` kept whole. Reverted tip preserved at tag
`ab-stage2-preserved-8c69b361`.

## 1. The decision

Stage 2 (slices B–D3b) degraded interactive frame rate. Two independent measurements, two methods,
same direction.

### (a) Phase-matched live traces

Trace 3 (`arrayscope-diagnostics-20260802-233644.jsonl`, `smooth-ui` @ `6eadeb9c`) vs trace 4
(`arrayscope-diagnostics-20260803-133348.jsonl`, `nervous-satoshi` @ `8c69b361`). Same operator
script, same data (336×336×272 float32), wgpu, ProcessWorkerBackend.

| phase | draws/s before | after | change |
|---|---|---|---|
| P1 slice + small montage | 7.4 | 8.0 | +7 % (noise) |
| **P2 full montage, index scrub** | **6.7** | **5.0** | **−25 %** |
| P3 FFT + interaction | 5.3 | 3.6 | −32 % (**not** a fair pair — trace 4 added a window enlargement and zoom-to-native) |

P2 is a clean pair: **1350 vs 1355 uploads, 354 vs 355 MB**, identical LOD levels and factors,
trace 4 marginally *more* zoomed out, and trace 4 had *fewer* binding republications (28 vs 39).
Same work in, 25 % fewer frames out.

**Confound to respect:** the `montage_retarget` channel did not exist in trace 3 (added by slice A),
so GUI-occupancy comparisons across the two traces are invalid. Draw rate is not confounded — it is
the same counter in both.

### (b) Probe A/B, 29 order-balanced runs with a null control

`tools/render_pass_governor_probe.py` at `4db839cd` (stage 1) vs `8c69b361` (stage 2), plus a third
worktree at the *same* commit as a null arm. Per-arm venvs on `/usr/bin/python3.14`; every run
verified `arrayscope.__file__` resolved inside its own arm (the cwd-leak trap fired live on first
setup and was corrected).

| axis | stage 1 | stage 2 | Δ | p | **null control** |
|---|---|---|---|---|---|
| fill draws/s | 7.68 | 7.02 | **−8.6 %** | 0.0013 | −0.9 % (p=0.84) |
| interaction tile draws/s | 17.12 | 15.05 | **−12.1 %** | <0.0001 | +3.7 % (p=0.59) |
| scroll draw-gap p90 (ms) | 150.7 | 180.1 | +19.5 % | <0.0001 | +8.0 % (p=0.43) |
| longest zero-draw interval while debt exists, fill (ms) | 907 | 1082 | +19.2 % | 0.050 | −5.1 % (p=0.77) |
| uploads / MB | 1389 / 364 | 1382 / 362 | −0.5 % | 0.15 | — |
| target-settled runs | 6/10 | 3/12 | — | — | 0/7 |

**Per-run spread (~±10 %) exceeds the effect**, so the direction is trustworthy only because it is
consistent across 29 order-balanced runs, replicates in an independent three-way batch, and the
same-code null control moves these metrics by ≤3.7 % at p≥0.59. **Any A/B here with fewer than about
five runs per arm is noise.**

Two metrics deliberately not leaned on: `fast_maxgap` (control swung −29.5 %) and the max-gap family
generally. Lock-*wait magnitudes* showed genuine worktree-level bias in the control (p≈0.02–0.03), so
the 0.7 ms → 74 ms figure below is "categorically different", not a calibrated ratio.

### Mechanism

- `staged_clone` + `install_staged` (introduced `1f0bd572`) are **O(all plan tiles) per retarget**.
  Attribution ~22.9 ms of a 27 ms p50 — but see caveat §4.
- The split **relocated blocking onto the GUI thread**: `gui_lock_wait_ms_max` **0.01 → 37–96 ms**,
  `gui_lock_acquisitions` +18 % (fill) / +35 % (scroll), while owner-side wait went to **0**.
- `session_setup_ms` — the one field measured identically by both arms — rose **74.8 → 180.7 ms**.
- It was built to enable admission routing, which its own final commit abandoned.

## 2. What was kept, and why

`3d54934d` (ADR), `4db839cd` (stage 1), `4f62ce7a` (stage-2 design doc), `b0312b70` (slice A).

- **Stage 1 works.** `montage-stage-plan` appears in **0 of 2632** GUI-thread drain observations in
  trace 4, where trace 3 carried 4.4 s at p50 13.7 ms.
- **Slice B, not stage 1, owns the 798 ms owner stall.** At `4db839cd` the stage-plan completion tail
  is a closure posted to the GUI thread; `76a92cd0` deleted the post and left the body inline on the
  dispatching thread. Reverting slice B removes the worst owner-side stall in the trace.
- **Slice A closes stage 1's own named hole** (the unlocked `latency_feedback` window alias across
  five call sites), makes `render_round_commits` monotonic against a watermark (read-then-zero loses
  a commit, and one lost commit trips R6 shed escalation by itself), and adds the `montage_retarget`
  timing channel — correctly classified pressure-telemetry-only, on the rule that *a measurement
  channel which steers scheduling is not a measurement*. That channel is what made this whole
  diagnosis possible.

## 3. Carried over by hand from the reverted range

Tracked as follow-up work, not part of the revert commit:

| item | origin | what |
|---|---|---|
| dirty-set peek/release | slice C | `consume_dirty_tiles()` ran as the first statement of `_commit_tile_layer`, ahead of seven bail sites; every bail permanently dropped the dirty set. A **genuine single-threaded leak pre-dating the series**. Re-implement without the cross-thread `prefix=` parameter, and gate release on `acknowledged` |
| overlay after `mark_presented` | slice C | the tile-truth debug overlay rendered the predecessor commit's truth |
| `geometry.montage is None` guard hoist | slice C | above `decide_presentation` |
| **`continue_warm` lock** | *new* | `frame_effects.py:7006`, posted at `:7080`/`:7087`, runs unlocked via a bare `callback()` and writes `session._atomic_warm_job` / `final_commit_pending` / `flush_pending`. Under stage 1 the owner concurrently writes session state *under* the order lock; an unlocked writer racing a locked one is unprotected. **No stage-2 slice fixed this** |
| probe per-fill draw table | slice D3b | `FillSummary.phase_ms` + `presentation_draws` + `tile_presentation_draws`. All three source fields pre-date stage 2. This is the axis the revert decision turned on and nothing else reports it per fill |
| R6 rebuild-then-replan comment | slice D2b | `frame_controller.py:1479-1491`. Bisected knowledge about **pre-stage-2** code: R6's shed release lives in the retarget's tail, and the round it belongs to is opened by the retarget the index-window rebuild owes — deferring that call by one hop lets the next scrub step supersede it and the round is never released |
| fingerprint oracle + 13 scenario builders | slice D-prep | `test_presentation_plan_split.py:138-496` — `_canonical`/`_scrub`/`_lifecycle_fingerprint`/`_session_fingerprint`/`_delta_fingerprint` over ~50 attributes, plus 13 `FrameSession` state constructors (cold fill, capped fill, settled, out-of-view retention, stale backend identity, pending removals, departed tiles, rehydrated, deferred invalidations). Pure product-state constructors with no staging API — a general differential oracle for `build_tile_presentation` |
| `_reset_commit_timings` completeness | pre-existing | it resets 19 attributes; `_commit_feedback_details` reads 22. Post-revert the three unreset ones are GUI-written, so they degrade from "impossible values" to "stale carry-over across commits" — still a lie on any commit that bails early |

## 4. Claims from the reverted range that did **not** survive audit

Recorded so nobody re-derives the worry.

- **Slice C's four-phase commit is scaffolding, not an independent structural win.** It *adds* a
  duplicated `_decide_direct_presentation` site, a second `capture_residency_snapshot` and a second
  `read_frame_boundary_inputs` per commit. Its headline (`payload_build p50 → 3.2–5.1 ms`) is an
  instrument artifact: pre-commit the timer stopped after `prepare_stats`, the level-decision metadata
  block and `ensure_tile_states()`; post-commit it stops immediately after `payload_state_done`. **The
  endpoint moved strictly earlier — the number had to fall.** Its claimed settle improvement has no
  identified code cause.
- **"A mid-build exception now leaves the session untouched"** is mechanically real but is hygiene,
  not a live defect: `_note_commit_raised` already marks the session terminally dead and refuses to
  re-arm, so nothing re-reads the partial state. No test pins it.
- **`commit_feedback_ms` "redefined for all four channels"** — it was already uniform. The
  redefinition makes the number systematically smaller and feeds the governor's commit-cost fit.
- **Slice B's residency-probe claim** ("can only under-report") — the pre-existing probe ends in a
  live `all(key in executor.page_table ...)` and the geometry memo clears on `page_table.generation`.
  It could not over-report. Nothing was fixed.
- **The `ack-refused-stale` verdict** is unreachable by its own docstring and its counter has no
  reader anywhere in the tree.
- **The 22.9 ms staged-clone attribution is probably inflated.** The prepare seam clones at
  `frame_effects.py:3185`, `build_tile_presentation` clones **again** inside `plan_tile_presentation`,
  and `build_atomic_successor_presentation` clones a **third** time when the atomic fast path declines.
  Before treating 22.9 ms as intrinsic, confirm whether it includes the nested redundant clones.

## 5. Defects in the current tree that this revert does *not* address

All independently verified in trace 4; all survive the revert and remain the live work.

1. **`montage_retarget` is the dominant GUI-thread cost** — 33.7 % (P2) / 36.3 % (P4) of wall clock
   net of nested commits, p50 27 ms, 10–14.5 calls/s (tracks input rate, not work),
   **`submitted=0` on 1308/1362 (96 %)**, max 602 ms (P2) / 960 ms (P3). ~91 % of it is GUI-thread
   (derived from owner lock occupancy: `owner_lock_acquisitions == messages_dispatched +
   owner_batch_acquisitions` exactly, and `session_order_lock` is a null context off the GUI thread).
2. **61 % of P4 retargets are self-generated pairs.** `ProgressiveSchedulingPolicy.retarget`
   re-opens `SchedulingPhase.COVERAGE` on every visible-set change; the next verdict *inside the same
   call* finds coverage complete, flips to REFINE and fires `on_refinement_replan` → 1 ms timer → a
   second full retarget, this one without `prepared_lod_swap_ready` so it also pays the ~20.6 ms
   swap-mark row.
3. **The vanishing tiles and the 148 ms commit p95 are one bug.** `_may_retain_out_of_view`
   (`frame_session.py:4139`) requires the backend to be drawing *exactly* the payload the session
   holds. An LOD refinement to an **out-of-view** tile breaks that, retention is refused, and the tile
   leaves `active_tiles` — it stops being drawn, though its pixels were never stale. The scope shrink
   breaks the executor's `previous_commit["tiles"] ⊆ payloads` fast path, forcing a ~150 ms full
   binding republication. P4: 23 republications correlating **23/23** with the commits over 100 ms,
   which are 100 % of the p95. Current behaviour is pinned by
   `tests/window/test_out_of_view_retention.py:270` — rewrite it deliberately.
4. **The wgpu camera path is already correct — do not rebuild it.** Transform is in the shader,
   `_set_tiles` short-circuits on object identity, `_present` is one 32-byte uniform write plus one
   instanced draw for all 272 tiles, and the canvas paces to `QScreen.refreshRate()`. There is no
   per-draw O(tiles) Python term. The frame clock is a `QTimer.singleShot` losing to 27–104 ms
   planner callbacks.
5. **Compute workers are administratively idle during interaction.** `window/main.py:667` sets the
   `DISPLAY_PREPARATION` lane quota to **0** while `interactive`; **199 of 263 snapshots have
   `kernel.running == 0`** with `parked_quota` peaking at 635.
6. **Resource telemetry is frozen.** `system_cpu_percent` and `process_cpu_percent` are 0.0 in all
   263 snapshots and `load_average_1m` is bit-identical across 139 s, so `pressure.cpu_headroom` is
   permanently 0.5 and the `< 0.15` branch is unreachable in production.
7. **The child-process env pin is inert.** `_spawn` passes a `SlotPoolHandle` as a `Process` argument
   and that module imports numpy, so unpickling imports numpy *before* `_worker_main` applies
   `OMP/OPENBLAS/MKL_NUM_THREADS=1`. Measured in the child: `openblas_get_num_threads` = **16**,
   unchanged after the pin.
8. **Frame-timing instrumentation is built and unexported.** `frame_timing.py` computes
   `schedule_slip_ms`, `phase_lock_r`, `acquire_ms`, `present_ms` and publishes them as
   `wgpu_screen_*`, but `MontageDiagnostics` does not carry them — so the trace cannot say whether the
   present path ran **Mailbox** or **Fifo**, and under Fifo `get_current_texture()` blocks the GUI
   thread ~15 ms per frame.

The FFT-threading hypothesis was tested and **refuted**: `CenteredFFT` declares `blocking_axes=(2,)`
and axis 2 is the montage axis, so the transform already runs as a whole-volume shared stage at
`stage_workers=1, fft_workers_stage=8` — one worker, all native threads. Measured at the traced shape,
raising workers 8 → 14 would save ~11 ms in a 139 s session, and 76 % of the composite is the two
memcpy-bound shift passes sitting at the memory-bandwidth floor.
