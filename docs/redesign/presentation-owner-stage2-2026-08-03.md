# ADR 0065 stage 2 — transaction preparation on the presentation owner

Date: 2026-08-03. Status: design, grounded in three line-verified mapping
passes over `4db839cd`. Companion to `docs/decisions/0065-presentation-owner-thread.md`;
this document is the implementation plan and records the decisions the maps
forced. Line references are against `4db839cd` and drift with edits — verify
before acting on one.

## What stage 2 delivers

The GUI thread's montage commit shrinks to: take a prepared transaction →
validate → backend apply/submit → Qt residue → post AckInputs. The owner
prepares transactions (delta build, level decisions, governor render-pass
decision, priority order), runs the acknowledgement bookkeeping, and owns the
index-window/pipeline retargets. Then — and only then — the per-tile `frame:`
completion families route to the owner (the stage-1 correction's condition).

Adjudication stays draw-gap structure against trace 3
(`arrayscope-diagnostics-20260802-233644.jsonl`: phase medians 8/6/4 draws/s),
plus the probe's fill-health column (3/3 `272/272 within-5s` is the floor —
stage 1's regression was caught there and nowhere else).

## Decisions (each was an open question the maps closed)

1. **Trigger shape: prepare at the tail of the owner's drain batch, coalesced
   by the existing gate-owner flag.** Preparing per `apply_commit` collapses
   the coalescing that turns ~500 admissions into ~6-10 commits; preparing on
   gate-arm doubles cross-thread hops for the seven non-admission arm sites.
   Instead `request_presentation` keeps its one call shape (arm the gate
   flags), and the owner prepares once per drain batch — the freshest and
   most-coalesced point, beside the existing `after_dispatch` hook. The
   single-flight-per-session-generation property (the gate-owner dedup at
   `frame_effects.py:1753`) is reused verbatim as the coalescing key.
   `_rearm_if_backlog`'s progress guard (`_backlog_signature`, 1914-1974)
   moves to the owner with its inputs.

2. **Prepare must not be destructive.** `session.consume_dirty_tiles()`
   (2372) at prepare time strands tiles if the GUI drops the transaction.
   The sharpest correctness edge in the stage: prepare peeks; the dirty set
   is consumed on acknowledgement of the accepted delta, and a take-miss
   re-dirties nothing because nothing was consumed. Pinned by a
   counting test that drops a transaction and proves the tiles re-present.

3. **The transaction's validation tuple generalizes
   `_prepared_atomic_transaction_current` (5402-5453)** to every
   transaction: `(session_id, id(session), level_revision, base_revision
   vs live tile_presentation_state.revision, active scope ⊇ required,
   target identities, payload markers, render_round_key, capabilities
   identity, page-table eviction stamp)`. Both `session_id` AND
   `id(session)` — index-window retargets advance identity on the same
   object; either alone reintroduces `fired-generation-stale` as a
   wrong-transaction take.

4. **`decide_presentation` hoists out of `_commit_direct_delta` onto the
   owner**, carried as a `PresentationDecision` in the transaction; the GUI
   validates `previous_frame is win._committed_display_frame` at take and
   treats mismatch as a miss. `lifecycle.commit_emitted` fires GUI-side
   immediately before submit (the 3017-3021 invariant: a declined apply must
   not strand TARGET_EMITTED) and its effect travels back in AckInputs.

5. **Residency truth for the builder is a posted snapshot of the page
   table, not the lifecycle mirror.** The mirror answers "was this exact
   payload acknowledged"; `tiledPayloadResident` answers "are this
   binding's pages resident" — they diverge on crop-rebinds (new source_id
   over resident pages). Snapshot shape: resident plane index
   (`{plane_identity: resident level/representation keys}`) plus
   `page_table.eviction_stamp` (`gpu/page_table.py:92`); the GUI re-checks
   the stamp at take — bind only adds coverage, so an unchanged stamp
   proves no trusted key was lost, converting staleness into a mailbox
   miss. The stage-1 planner mirror stays as is (its question is the
   mirror's question).

6. **The governor render-pass family runs wholly on the owner** —
   `begin_render_pass` + `decide_render_pass` + recording the render-pass
   observation — because the decision↔observation pairing walks a stored
   marker and is only atomic per-call, not per-pair. The GUI's submit span
   travels in AckInputs and the owner records it. `commit_feedback_ms` is
   redefined ONCE to the GUI submit span (`_present_tile_delta` entry →
   return); all commit channels change definition together, or the
   fitted model is poisoned by alternation. Frame-boundary inputs
   (`presentationDrawPending`, `interactive`) are re-read by the GUI at
   take: if `frame_critical` flipped true, the GUI drops the transaction
   (miss; owner re-prepares) rather than submitting an idle-sized chunk.
   `render_entry_elapsed_ms`/enclosing-callback pricing becomes the GUI's
   own take-time elapsed; the `partial_remap_presentation` path keeps a
   GUI-side synchronous commit in this stage.

7. **The `latency_feedback` alias dies.** `LatencyFeedbackController` has no
   lock; the alias (`main.py:163`) lets four prepare-path sites (5749,
   5775, 6087, 6336) read/write it unlocked. Replace with `@_governed`
   accessors on `ResourceGovernor` (`batch_limit_for(channel, interactive)`,
   `latency_tuning()`); `frame_effects.latency_feedback()` routes through
   the governor or is deleted. Also decorate `decide_ladder_admission` /
   `decide_resident_crop_rebind` before their callers move.

8. **AckInputs** carries: report, `getLevels()`, shader_windowing,
   `residentHistogramEvidence` result (one frozen bool feeding both its
   consumers — they must not disagree within one commit), the residency
   snapshot + stamp, the new committed frame, continuity range, submit
   timings, draw-pending/interactive at submit, and the identity block of
   decision 3. The owner-side ack handler is ONE indivisible owner lock
   section honoring the 13 intra-handler orderings mapped 2026-08-03
   (mirrors before ack; presented-before/after around mark_presented;
   backoff signature before note_committed; etc.). A refused ack emits a
   new `ack-refused-stale` verdict + owner counter — today's synchronous
   code never needed one, and a silent refusal is invisible.

9. **The GUI residue is driven by transaction/ack fields, not live session
   reads**: tile_states snapshot + revision (kills the last live read at
   `frame_runtime.py:179` and the overlay walk at 3610-3617), the wgpu
   evidence obligation tuple, extent/geometry fields. The residue that
   *reads ack results* (extent refresh reads what mark_presented wrote —
   ordering O5) runs from the owner's posted followup message carrying
   `{settle, finish, prefetch_ok}` booleans; the GUI re-checks currency at
   entry as `reconcile_after_phase_close` already does. `applyHistogramMetadata`
   on the level-evidence-wait bail becomes part of that posted residue.

10. **The retarget becomes an owner operation** (map F5 order):
    1. the replan gate `fire()` (highest count, no dependents; delete the
       stage-1 `run_on_gui_thread` + `QTimer.singleShot(1)` hop — arm and
       fire land on one thread again);
    2. the stage-plan tail `retarget_on_gui` becomes an inline owner call
       (it exists only because the retarget was GUI-affine); only chrome
       posts;
    3. the index-window model rebuild (`frame_controller.py:1293-1447`
       minus the img_view probe and `_hide_superseded_presentation`) as an
       owner message; its `commit_frame_session_presentation` tail becomes
       a published transaction + gate post — the gate post IS the wakeup
       the 1058-1065 comment demands, so the lost-wakeup invariant holds;
    4. resize/viewport fire-and-forget callers;
    5. the viewport-only path with its convergence latch moved owner-side
       (the one `submitted` reader);
    6. re-enable `frame:` routing in `montage_completion_route`.
    Interior affinity fixes the maps enumerated: capabilities cached per
    session at build, `_evaluation_context` stamped into the request,
    `applyHistogramMetadata`/`setImageStale`/`setEvaluationOverlay`/
    overlay-clear/status/slow-overlay-timer posted as chrome, the
    crop-rebind seed's `tiledPayloadResident` + `_committed_display_frame`
    reads served by the decision-5 snapshot.

11. **The round latch survives the split via a monotonic commit counter.**
    `open_round` currently reads `render_round_commits > 0` then resets it;
    a GUI increment between read and write is silently lost and one lost
    commit trips R6 shed escalation (`TARGET_PASS_AFTER=1`). The counter
    becomes monotonic; the owner stores a watermark at open and compares
    `commits > watermark_of_predecessor`. `render_round_id` travels inside
    the transaction so the GUI's pass token can never lag the owner's
    round rotation.

12. **`_last_montage_*` timing attributes become a per-commit record**
    assembled from the owner's prepare block + the GUI's submit block and
    published atomically; the `commit_batch` trace is emitted once from
    the merged record. Diagnostics reads (`diagnostics_snapshot.py`) read
    the record, not the attributes. Dual-writer flags named by the maps
    (`_frame_viewport_retarget_after_commit`, `_montage_replan_gate_armed`,
    the gate-owner flags) stay under the order lock until stage 3 assigns
    them one side.

## What stays GUI-side in stage 2 (verified, not assumed)

`commit_tiled_delta` + committed-frame rebuild + axis flips + setImageStale;
`_warm_atomic_successor_residency` (real uploads; the transaction carries
`requires_hidden_warm` + the frozen prepared-transaction dict);
`setResidentHistogramEvidenceRequired` (obligation tuple travels);
`safe_tiled_payload_geometry_retarget` (the GUI's own pre-submit gate);
`_finish_presentation_commit` (Qt timer; its coverage-open read tolerates
the ack landing later — the flag-stays-armed degradation is already
documented as designed); session rebirth's synchronous spine
(`update_image_view`); the same-key-reuse and rebirth initial commits;
prefetch warm (stage 3); `_queue_wgpu_resident_histogram_evidence`'s
accept path (posted); the `montage:*:histogram` / evidence completion
families.

## Migration invariants

- Every step lands under the stage-1 session-order lock and is
  independently revertable; the lock's GUI-acquisition counter is the
  migration's progress gauge.
- Byte-identical presented output pins per step, mutation-checked; the
  probe's fill-health column re-run per step (3/3 within-5s, wgpu 272).
- The tile-truth overlay refresh moves from between-ack to post-ack —
  a real ordering change to overlay chrome, pinned by test rather than
  slipped in (map O15).
- No new staleness concepts: every drop is one of the five existing
  layers or the mailbox's exact-key miss.
- A retarget timing channel (`_record_ui_work("montage_retarget", ...)`)
  lands FIRST, so the move is measured on the axis it was diagnosed on
  (the 89 ms × 52 figure came from lock-hold instrumentation that no
  longer exists).

## Slice C: what landed, and the finding that stopped the rest (2026-08-03)

Landed, against the decisions above, all of it on the GUI thread:

- **The four-phase split** (`window/frame_effects.py`): `_prepare_tile_layer_transaction`
  (Qt-free) -> `_submit_prepared_transaction` -> `_acknowledge_transaction` /
  `_acknowledge_and_publish` -> residue. `_commit_tile_layer` runs the first
  two and is the correctness anchor. The prepare reads no Qt: residency comes
  from `_physical_residency_inputs`, the frame-boundary probes from
  `read_frame_boundary_inputs`, capabilities from the frozen dataclass, and
  `getLevels()` from the last acknowledgement.
- **Decision 2**: `FrameSession.peek_dirty_tiles` / `release_dirty_tiles`
  replace `consume_dirty_tiles`; the release happens on the acknowledgement of
  the accepted delta. Counting tests in
  `tests/presentation/test_prepared_transaction.py`.
- **Decision 3/4**: `TransactionIdentity` carries the validation tuple;
  `_decide_direct_presentation` hoists `decide_presentation` out of
  `_commit_direct_delta` and binds it to the `previous_frame` it was decided
  against.
- **Decision 5**: `_physical_resident_predicate` serves the builder's
  `physical_resident_fn` from the residency snapshot off the GUI thread.
- **Decision 6**: `FrameBoundaryInputs` + `read_frame_boundary_inputs` freeze
  `presentationDrawPending` / `presentationRefreshRate` /
  `render_entry_elapsed_ms`; `commit_feedback_ms` is redefined ONCE, in
  `_record_commit_feedback`, to the GUI submit span for all four commit
  channels together.
- **Decision 8**: `AckInputs`, with `residentHistogramEvidence` probed once and
  travelling as one frozen bool for both consumers.
- **Decision 12**: `CommitTimings` published by one attribute swap;
  `diagnostics_snapshot` reads it through `_commit_timing`.
- **Map O15**: the tile-truth overlay refresh moved from between-ack to
  post-ack. The pin is still owed (it needs a ~40-attribute session double).

**Not landed, and this is the finding: preparation cannot run speculatively
on the owner, because it is not idempotent.**

The trigger was built exactly as decision 1 specifies — `request_presentation`
posts one preparation per armed gate, the owner runs it at the tail of its
drain batch, the gate-owner flag is the coalescing key and the single-flight
interlock — and it was measured to break three montage geometry/hover tests
(`tests/ui/test_montage_interactions.py`: `..._hover_uses_source_index_...`,
`..._panned_montage_hover_...`, `..._shifted_origin_preserves_world_view_range`).
The symptom is `geometry_matches=False` with every tile committed and the last
commit reporting `empty-progressive-settled`: the committed montage geometry
never converges, and the session settles believing the work is done.

The bisect is unambiguous. With the owner post removed the three pass; with it
restored they fail; and with the post kept but the **publish suppressed** —
so the owner prepares and the GUI always re-prepares inline — two of the three
still fail. That last arm is the one that names the cause: the damage is not
in the take, the validation, or the mailbox. It is that **prepare ran twice**.

`FrameSession.build_tile_presentation` is not a pure builder. It calls
`release_deferred_out_of_view_invalidations`, writes
`display_tile_payloads`, retires payloads and pending upserts for slots with
nothing worth keeping. The level block around it advances
`applied_level_source`, `begin_level_presentation_update` and
`bind_payloads_to_level_generation`. Every bail has its own session writes.
Decision 2 named `consume_dirty_tiles` as the destructive step; it is the
*unrecoverable* one, but it is not the only one.

The gate-flag interlock covers one order — GUI first, owner second, owner sees
an unarmed gate and skips. It cannot cover the other: owner prepares, the GUI's
take misses or fails validation, and the GUI prepares again. As long as the
inline fallback re-prepares, the double execution is reachable by construction,
and "a miss is a normal outcome" makes it frequent rather than rare.

### What slice D owes because of this

1. **Make preparation re-entrant, or make the fallback not re-prepare.** The
   two candidates: (a) split `build_tile_presentation` into a pure planner and
   an applier so a prepared-but-dropped transaction writes nothing, which is
   the version the ADR's "the GUI validates and submits" actually implies; or
   (b) let a take-miss re-arm the gate instead of preparing inline, which keeps
   the writes single-flight but gives up the correctness anchor and needs its
   own livelock argument. (a) is the real answer; (b) is the shortcut.
2. **A geometry staleness layer.** `plan.geometry` and `view_state` change on a
   pan without moving the session id, the level round, the base revision or the
   render round, so the identity tuple of decision 3 is incomplete as written.
   Adding it was necessary and not sufficient (the three tests still failed
   with it in place), so it belongs in the tuple whatever else changes.
3. **The acknowledgement post**, unchanged in ambition: `_acknowledge_and_publish`
   already takes a frozen `AckInputs` and makes no live image-view read, which
   was the prerequisite. What is owed is the post, the `CommitFollowup`
   message, and the residue's currency re-checks. The `ack-refused-stale`
   verdict and counter are in the tree ahead of the move.
4. The render-pass **observation** (the recording half of decision 6) is still
   GUI-side; only the decision half moved.

## Slice D: the seam is a planner, and the trigger is refuted a second time (2026-08-03)

### D1 landed: the prepare seam plans, the take writes

Slice C's finding was that `build_tile_presentation` is not a pure builder.
D-prep fixed that for the *builder*. D1 does the same one level out, for the
whole `_prepare_tile_layer_transaction` seam, which is where the level
decisions, the atomic-successor block, the metadata publication decision and
every bail's flags live.

- `_prepare_tile_layer_transaction` opens with `live._staged_presentation_session()`
  and does every session-facing thing against that copy. It returns either a
  `PreparedPresentationTransaction` or a `PreparedCommitBail`, and **both carry
  the `TilePresentationPlan`**. `_take_prepared_outcome` applies it. That is a
  commit's only session mutation.
- A bail is a result too: its renderer-side continuations (`request_montage_replan`,
  `_finish_after_noop_commit`, `_schedule_montage_cached_level_stats`, the
  rough-histogram publication) travel as enumerated flags on `PreparedCommitBail`
  and run *after* the plan is applied, in the inline order. Closures were
  rejected: a closure captures the session it was built against, and the point
  of the record is that the applier binds it to the live session at take time.
- The seam separation the D-prep report demanded is decided by **what a
  renderer call does with the session, not what it reads**. A call that only
  reads (`_montage_level_source_for_session`, `_montage_level_stats_for_session`,
  `_first_pass_level_evidence_complete`) is handed the staged session, which is
  what preserves the inline ordering. A call that *schedules* — and therefore
  captures the session in a kernel closure — is handed the LIVE one:
  `_queue_montage_level_stats_for_payloads` and
  `_montage_histogram_plot_data_for_session`.
- **One deliberate exception to "prepare writes nothing"**:
  `_queue_montage_level_stats_for_payloads` runs live and inline rather than
  deferred, because it merges this delta's evidence into the tracker that
  `_montage_level_stats_for_session` reads two statements later. Deferring it
  moves this commit's evidence into the *next* commit's metadata decision. Its
  session writes are the additive pending-level queue, deduped by
  `queued_sources` / `has_source_quality`.
- `_shader_presented_bounds_clamp` joins `_PRESENTATION_PLAN_CONTAINERS`:
  `shader_commit_level_source` mutates `true_bounds` / `published_bounds` **in
  place**, which is exactly the write the harvest's identity diff cannot see,
  and a dropped plan that advanced `true_bounds` would release the R3 clamp
  against bounds no commit ever presented.
- Cost: the seam keeps calling the composed `build_tile_presentation` (which
  stages again internally) rather than the private builder body, so the anchor
  keeps its production caller. One extra shallow session clone per commit
  (~0.6 ms per 272-record lifecycle clone).

Verification: `tests/window` + `tests/presentation` + `tests/ui` — 1373 passed
against a baseline of 1369, **identical failure list** (the four pre-existing
reds: two `test_resource_governor_integration` session doubles missing `key`,
two load-sensitive `test_session_order_lock` cases). Probe `--backend wgpu
--tiles 272 --repeat 3`, three times: **9/9 `272/272 within-5s`**, target
settle 3.8–4.6 s.

### D2 refuted in the designed form: prepare on the owner starves the GUI

The trigger was built exactly as decision 1 specifies — `request_presentation`
posts one preparation per armed gate, the owner runs it at its drain-batch
tail, the gate-owner flag is the coalescing key and the single-flight
interlock — with the D1 planner underneath, so the prepare-twice corruption
that killed slice C is structurally gone. It still does not ship.

**A new validation layer was required first, and it is not a staleness rule.**
The plan carries private copies of `dirty_payloads`, `pending_payload_upserts`
and `pending_removals`, so applying a plan prepared before an admission
*deletes that admission*. Measured immediately as a settle wedge — 35 stale
tiles, `flush_pending=False`, nothing loading, which is what a lost admission
looks like from outside. `TransactionIdentity.backlog_key` makes an admission
in the prepare→take window a **miss**, not a merge: merging would stop the
applier being mechanical, and mechanical is what makes replaying a plan safe.

With that in place the mechanism works and is measurably used:

| probe run / pass | takes | taken | missed | refused | inline | hit rate |
|---|---:|---:|---:|---:|---:|---:|
| 0 preview | 12 | 5 | 2 | 5 | 7 | 42% |
| 0 target  | 32 | 22 | 2 | 8 | 10 | 69% |
| 1 preview | 13 | 9 | 2 | 2 | 4 | 69% |
| 1 target  | 29 | 20 | 2 | 7 | 9 | 69% |
| 2 preview | 9 | 5 | 2 | 2 | 4 | 56% |
| 2 target  | 29 | 20 | 2 | 7 | 9 | 69% |

Every refusal was `backlog`; no other validation layer fired once. The probe
itself stayed **3/3 `272/272 within-5s`**, and the owner's GUI-side lock waits
stayed small (`gui_lock_wait_ms_max` 3–28 ms).

**The suite is where it fails.** `tests/ui` went from 2 pre-existing reds to
12, and `tests/window` gained one, all of them settle wedges / lost-wakeup
timeouts:

```
test_resident_crop_rebind: test_crop_scrub_without_the_capability_still_evaluates
test_resident_crop_rebind: test_rebind_and_evaluation_paths_settle_identical_levels
test_resident_crop_rebind: test_operation_pipeline_rebind_..._settle_identical_levels
test_montage_interactions: ..._complex_montage_tile_layer_rewindows_rgb_from_histogram_levels
test_montage_data_change_invalidation: test_cpu_item_data_change_leaves_no_out_of_view_tile_drawn
test_montage_data_change_invalidation: test_data_change_serves_the_visible_set_...[pyqtgraph]
test_montage_scroll_settling: test_cropped_display_axis_scroll_keeps_complete_montage[pyqtgraph]
test_montage_settled_quiescence: ..._owes_no_bookkeeping[pyqtgraph, wgpu]
test_window_shift_live_path: test_fixed_index_scroll_forward_hits_warm_wgpu_residency
test_frame_paced_commits: test_a_montage_fill_interleaves_its_draws_with_its_commit_batches
```

A faulthandler dump caught the owner thread **inside the montage build**
(`shader_mapping.__eq__`, under `_prepare_tile_layer_transaction`) while the
GUI waited. That is the stage-1 refutation happening from the other side: the
migration bridge is one lock, and the owner's prepare is a long section under
it — the same shape as the 89 ms `retarget_frame_pipeline` holds that censored
the fill when `frame:` was routed. Preparation is not cheap bookkeeping like
the stage-plan bodies; it is the montage build.

The prepare also runs **twice** whenever the take misses (31–58% of commits
here), because the inline path is the correctness anchor and re-prepares.

### What slice D leaves for the next attempt

1. **The owner's prepare cannot be one lock section against a GUI that still
   holds its own.** Either the migration bridge stops being a single lock for
   this seam, or the prepare has to be interruptible/chunked, or D5's coupled
   pair (per-tile `frame:` routing) has to land *first* so the owner's batches
   are real barriers and the GUI's sections have already shrunk. The stage-1
   note that "the two land together, or neither does" may be stronger than
   written: the trigger may need the admissions routed *before* it, not with it.
2. **The double prepare on a miss is not affordable at this cost.** Either the
   miss must re-arm the gate instead of preparing inline (slice C's option (b),
   which gives up the anchor and needs a livelock argument), or the backlog
   conservation refusal has to become cheap enough to avoid — which means the
   plan must stop carrying whole copies of the admission containers.
3. D3 (ack on the owner), D4 and D5 were not reached. `_acknowledge_and_publish`
   remains AckInputs-driven and free of live image-view reads, so the post is
   still the only thing owed there.
4. `TransactionIdentity`'s extra fields (`state_revision`, `geometry_key`,
   `backlog_key`), `presentation_geometry_key`, `presentation_backlog_key`, the
   `PreparedTransactionMailbox` with its closed counters, and the probe's
   `prepared-transaction` hit-rate column are all *removed from the tree* with
   D2 — they are recorded here rather than left as dead code, and the shape
   above is what to rebuild.

## Slice D2a: the prepare stops being a lock section; the routing is refused a third time (2026-08-03)

Two moves were attempted, and they separate cleanly. The first landed and is a
pure win. The second is refuted again, and this time the counters name the
precondition instead of only the symptom.

### Move 1 landed: prepare is stage-then-plan, and the plan takes no lock

D2's faulthandler caught the owner inside `shader_mapping.__eq__` under
`_prepare_tile_layer_transaction` while the GUI waited: the owner's prepare was
ONE lock section spanning the montage build. `_prepare_tile_layer_transaction`
is now the composition of three phases and nothing else:

1. `_stage_prepare_inputs` (`window/frame_effects.py`) — the lock's business.
   The private session copy plus **every** live, renderer, governor and widget
   read the plan needs, frozen into `_PrepareStaging`: residency inputs and the
   committed frame, the retained payload store's map, backend capabilities,
   `read_frame_boundary_inputs`, `tile_layer_upsert_limits`, the fast-drain
   flag, `getLevels()` and the page-table eviction stamp.
2. `_plan_prepared_outcome` — the ~900-line plan, over the staged copy and
   those frozen values, acquiring no lock.
3. the return — a frozen record, published lock-free.

Two reads that were made **twice** per commit collapse into one:
`_physical_residency_inputs` (the seed probe and the hidden-warm decision) and
the eviction stamp (which now names the generation the residency inputs were
actually taken against, rather than a later one).

`_staged_presentation_session` additionally freezes the live `__dict__` as the
harvest's baseline. Diffing against the live session *at harvest time* is only
correct while nothing writes it mid-plan; against the baseline, a plan carries
exactly what its own planner changed, and the applier cannot revert a live
write no planner ever saw.

The audit Move 1 owed found the plan is **not** free of live reads, and the two
that remain are the ones D1 documented as deliberate:
`_queue_montage_level_stats_for_payloads(live, …)` and
`_montage_histogram_plot_data_for_session(live, …)`. Both are renderer-facing
*scheduling* calls that capture a session in a kernel closure; both take no
order lock; the first also probes `img_view.residentHistogramEvidence`, which
is the one genuinely GPU-affine call left inside a prepare and is the debt the
owner trigger still owes. Everything else audited clean:
`image_view_backend_capabilities` is a frozen-dataclass attribute read,
`tiledUploadPreparations` is numpy plus a self-locking mailbox, and the level
trackers are renderer-owned dictionaries.

Pinned by `test_a_prepare_plans_while_another_thread_holds_the_order_lock`
(`tests/window/test_presentation_owner_wiring.py`): the order lock is held for
the whole middle phase by a third thread and the planner still completes;
mutation-checked by re-adding an acquire inside the plan.

Verification: `tests/window` + `tests/presentation` + `tests/render` +
`tests/core` + `tests/display` 2310 passed; the GPU ring
(`test_montage_interactions`, `test_montage_scroll_settling`) 54 passed twice;
probe `--backend wgpu --tiles 272 --repeat 3` **3/3 `272/272 within-5s`**,
target settle 4.2-4.5 s, cohorts 128/68/55.

### Move 2 refuted: routing `frame:` still censors, and the GUI's holds say why

Both halves were built as specified — `montage_completion_route` widened to
`("frame:", "montage-stage-plan:")`, and `request_montage_replan`'s `fire()`
converted from `run_on_gui_thread` + `singleShot(1)` to `post_to_presentation_owner`.
Measured on the 272-tile wgpu probe:

| arm | run 0 | run 1 | run 2 |
|---|---|---|---|
| Move 1 only (control) | 272/272 within-5s | 272/272 | 272/272 |
| Move 1 + both halves | 90/272 censored | 78/272 censored | 49/272 censored |
| Move 1 + routing only (`fire()` on the timer) | 186/272 censored | 272/272 **late** | 106/272 censored |
| Move 1 + routing only, repeat | 122/272 censored | 271/272 censored | 264/272 censored |

So the coupling claim is *not* what fails: routing alone is refused, with the
`fire()` conversion reverted. And the stage-1 signature is **absent** — commit
cohorts stayed healthy (104, 91, 55 tiles) rather than collapsing to ~10.

The owner counters at teardown, routing-only arm, one fill each:

| counter | run 0 | run 1 | run 2 |
|---|---:|---:|---:|
| `events_dispatched` | 632 | 641 | 651 |
| `owner_lock_wait_ms_total` | 1920.6 | 1897.6 | 1808.8 |
| `owner_lock_wait_ms_max` | 197.8 | 288.9 | 226.2 |
| `gui_lock_wait_ms_total` | 37.5 | 19.0 | 10.0 |
| `owner_yields_to_gui` | 6 | 6 | 6 |

The GUI is never starved — 10-37 ms of waiting across a whole fill, and the
stand-aside fires six times. The **owner** starves: 1.8-1.9 s of an ~8 s fill
spent waiting, across 631 acquisitions.

Instrumented GUI-side lock *holds* for one such fill, against the control arm:

| GUI section | routing arm calls / total ms / max ms | control arm |
|---|---|---|
| `commit_pending_session` | 43 / 2350.2 / 526.2 | 34 / 1913.4 / 520.6 |
| `retarget_frame_pipeline` | 38 / 2009.1 / 193.7 | 28 / 1334.9 / 196.3 |
| `_on_presentation_gate` (encloses the commit) | 41 / 1784.7 / 278.1 | 32 / 1353.7 / 154.6 |

That is the finding. Stage 1 blamed `retarget_frame_pipeline` at ~89 ms × 52,
and **it is still a GUI-side section**, 28-38 times a fill at ~50 ms mean and
~195 ms worst. Slice B moved only the fire-and-forget callers
(`post_frame_pipeline_retarget`); decision 10's items 2-5 — the stage-plan
tail, the index-window model rebuild, the resize/viewport callers, the
viewport-only latch — are all still owed, and the replan gate's own fire is a
fourth caller. Beside it, `commit_pending_session` holds the lock across the
*whole* commit including the backend submit (max 526 ms), which needs no
session ordering at all.

The condition to re-test the `frame:` clause under is therefore not "the
prepare is cheaper" — Move 1 made the prepare a computation with no lock inside
and the probe still refused. It is **GUI-side hold under roughly a frame**.
While two sections hold the bridge for 3-4 s of an 8 s fill, 630 owner
acquisitions cannot fit between them, and no policy on the owner's side changes
that. Next: finish decision 10's retarget items, and take the backend submit
out of the commit's lock section.

## Slice D2b: the commit's hold collapses, the retarget's does not (2026-08-03)

Three moves, and they separate as cleanly as D2a's two: the first is a pure
win, the second is structurally right and measurably beside the point, the
third is half refused by a test.

### Move 1 landed: the backend submit is not a lock section

`commit_pending_session` held the lock for its whole body, and so did every
funnel above it (`_on_presentation_gate`, `_commit_after_presentation_draw`).
The commit is now spans rather than a section — `_gate_admits_commit` is the
gate's own section; the commit takes one for [build + peek + prepare + take],
releases across `_present_tile_delta`, and takes another for the ack and the
backlog re-arm. Nesting still works: reached through `commit_frame_session_presentation`
or `update_image_view` (the rebirth spine, deliberately atomic) the enclosing
section holds the re-entrant lock and the release in the middle is a no-op.

**The audit the move owed — what the ack assumes about apply-to-ack
atomicity — found two reads, and both are re-derived rather than moved.**
`_SubmitWindow` is taken when the lock goes down:

- `release_dirty_tiles(transaction.dirty_tiles)` drops the peeked tiles **by
  value**, so an admission that re-appended one of them (`_admit_evaluation_result`
  is the only other writer, and it appends) would have its append swallowed.
  The peeked entries are the list's prefix, so the release takes a `prefix`
  count and is exact under a concurrent append.
- `acknowledge_tile_presentation` pops `dirty_payloads` / `pending_payload_upserts`
  for every accepted upsert — right for the payload this commit presented,
  wrong for a newer one an admission installed in the window. The window holds
  the delta tiles' payload objects by reference; tiles whose
  `display_tile_payloads` / `rendered_tiles` entry changed are re-dirtied after
  the ack, with a counter and a `commit_interleaved_admission` trace.

Everything else in the ack re-derives live under its own lock (`note_committed`
and `_finish_commit`'s `upload_backlog` recompute the flush flags from the live
containers, so an admission on a tile *outside* the delta was always safe). The
re-entrancy guard `_montage_commit_drain_active` deliberately spans the unlocked
middle — it is what turns a second commit request during the submit into a gate
re-arm — and the gate flags are cleared before the commit, so an admission that
arms the gate mid-submit gets its own wakeup.

### Move 2 landed structurally, and the numbers refuse its premise

`FramePipeline.retarget` is now the composition of `stage_retarget` (adopt the
intent, snapshot the ladder states) -> `plan_retarget` (no lock) ->
`admit_retarget` (claims, kernel submits), and `retarget_frame_pipeline` drives
those three with its own spans, dropping the lock for the middle. The staleness
rule is the pipeline's own admission generation plus a session-currency re-read:
a plan whose staging was superseded admits nothing and counts
`retargets_superseded`.

It works and it buys almost nothing, because **the expensive middle was not the
plan**. Sub-phase timings, same probe, 3 fills:

| phase | calls | total ms | max ms |
|---|---:|---:|---:|
| `retarget.stage.pipeline` (`effects.tile_states`) | 97 | 2223.6 | 50.9 |
| `retarget.stage.mark-swaps` | 88 | 1815.7 | 163.1 |
| `retarget.stage.intent+scope` | 97 | 53.5 | 2.6 |
| `retarget.admit` | 97 | 96.1 | 17.7 |
| `retarget.tail` | 97 | 62.2 | 2.6 |
| `retarget.plan-unlocked` (the middle) | 97 | *not held* | — |

`ladder.plan` is single-digit milliseconds per fill. The hold is
`effects.tile_states` (which releases evaluation claims and seeds crop rebinds
— session **writes**) and `mark_ladder_swaps_for_current_demand` (a write of
the lifecycle's swap marks). Neither can leave the lock by being moved; they
need the planner/applier treatment D1 gave the commit seam.

### Move 3: the hover caller posts, the index-window caller cannot

`apply_montage_priority_retarget`'s tail is now `post_frame_pipeline_retarget`.
`_maybe_retarget_frame_session`'s tail was converted too and **reverted**:
`tests/ui/test_montage_scroll_settling.py::test_wgpu_shed_index_scrub_moves_the_montage_and_still_completes`
reds with `ShedLevel.RETAIN_ONLY, streak=2` against a demanded `NONE`.
R6's shed release lives in the retarget's tail and belongs to the round the
index-window rebuild's own retarget opens; deferring it by one owner hop lets
the next scrub step supersede the post and the round is never released. The
dependent is real and it is not in the callback body — it is the
rebuild-then-replan pair. The replan gate's `fire()` was not touched (slice B).

### Holds, before and after (per fill, 272-tile wgpu probe)

| GUI section | D2a control | D2b |
|---|---|---|
| `commit_pending_session` | 34 / 1913.4 / 520.6 | 31-34 / 605-880 / 44-171 |
| `retarget_frame_pipeline` | 28 / 1334.9 / 196.3 | 31-33 / 1010-1472 / 193 |
| `_on_presentation_gate` | 32 / 1353.7 / 154.6 | 29-37 / 604-870 / 44-171 |

So the slice's target — max holds under ~16 ms — is **not met**. The commit is
where it moved (total -65%, worst case -91% except for outliers to 171 ms,
which are the prepare, not the submit); the retarget is unchanged.

### What the next slice owes, named by the measurement

1. **The retarget's staging is the remaining hold**, and it is two calls:
   `effects.tile_states` and `mark_ladder_swaps_for_current_demand`. Both write
   the lifecycle. Plan/apply them the way D1 planned the commit, or move the
   whole retarget to the owner and leave the GUI none of it (decision 10's
   items 3-5, which the index-window finding above now blocks: that caller
   needs its rebuild and its replan on the *same* thread, whichever thread it
   is).
2. **The commit's remaining 44-171 ms is the prepare**, still under the lock
   inline. D2a made it lock-free by construction, but releasing the lock across
   it here needs the backlog-conservation validation D2 built and removed
   (`TransactionIdentity.backlog_key`): the plan carries private copies of the
   admission containers, so applying one prepared before an admission deletes
   that admission.
3. `frame:` routing was not re-attempted, per the slice's scope.

## Slice D3: the stated condition was met, and the routing is refused a fourth time (2026-08-03)

D2c's whole point was the condition D2a named — "GUI-side hold under roughly a
frame" — so D3 is that condition being cashed in. It was met. The fill censored
anyway, and the counters name a blocker of a different kind, which is why this
entry closes the "shrink one more section" line of attack rather than extending
it.

### What was built (both halves, exactly as the coupling requires)

- `montage_completion_route` widened to `("frame:", "montage-stage-plan:")`.
- the replan gate's `fire()` converted from `run_on_gui_thread` +
  `singleShot(1)` to `post_to_presentation_owner(self, fire)` with an inline
  fallback for a renderer with no owner.
- test-side consequences: the routing table's `frame:` rows flip to routed, the
  timer census in `test_architecture_guards` loses its
  `request_montage_replan` / `singleShot` entry, and
  `test_a_posted_retarget_for_a_superseded_session_does_not_run` needs a new
  oracle — its evidence was "the owner thread never ran a retarget", which
  stops naming a caller once the replan gate's fire is also on the owner. (The
  replacement that worked: mark the post with `force_commit=True`, which the
  gate's fire never passes, and wait on a sentinel message posted behind it,
  since the owner drains messages FIFO.)

Everything above is reverted. What is kept is this section and the ledger entry
in `montage_completion_route`.

### The affinity re-audit: the named debt is not on the routed path

The debt D2a recorded — `_queue_montage_level_stats_for_payloads` reaching
`img_view.residentHistogramEvidence` through
`_queue_wgpu_resident_histogram_evidence`, the one genuinely GPU-affine call
left inside a prepare — **is not reachable from `apply_commit`**. Instrumented
on the routed arm (a thread-name trace at each of the three funnels, three
probe runs), `_queue_montage_level_stats_for_payloads` and
`_schedule_montage_histogram_aggregate` ran on `MainThread` only, and
`_queue_wgpu_resident_histogram_evidence` never ran at all in a cold fill. Its
live callers are the ack (`frame_effects.py` ~4868, GUI-side commit), the
prepare's deliberate live-session exception, and the interaction-stop edge
(`frame_runtime.py` ~758) — none of them a `frame:` body. No fix was owed and
none was made.

The rest of the audit of `apply_commit` -> `_admit_ready_payloads` -> the three
`_admit_*` helpers -> `mark_materialized` came back clean of Qt: `frame_session`
has no Qt reference at all, `render/lod.py` none, the lifecycle and level
trackers are plain dictionaries. Three findings worth keeping:

- `image_view_backend_capabilities(view)` is classified inconsistently in the
  tree — a "live image-view probe" at `frame_effects.py:2280` and "not a Qt
  call, safe from the presentation owner" at `level_stats.py:2422`. The second
  is correct (`display/backend_contract.py` is `getattr` chains onto a frozen
  dataclass); the first is merely guarded incidentally, because `on_gui_thread()`
  is the *first* term of its `and` chain. Worth reconciling, not a defect.
- `_prepare_backend_uploads` is mirror-based as documented, with one unaudited
  edge: `view.preparedTiledUploads` *lazily creates* the mailbox and writes it
  onto the widget (`imageview2d.py:1033`), so a first touch from the owner is an
  unsynchronised attribute write. Benign in practice, not race-free by
  construction.
- `store_montage_tile_result` (reached from `_admit_evaluation_result`) mutates
  `operation_evaluator`'s display cache, counters and `last_status` with no
  lock, and the GUI reads the same cache. Not Qt affinity — genuine shared
  mutable state that only becomes two-threaded when the admissions move.

### The measurement: 6/6 censored against 6/6 green, interleaved

`render_pass_governor_probe --backend wgpu --tiles 272 --repeat 3`, arms
alternated in one loop on one box:

| block | arm | run 0 | run 1 | run 2 |
|---|---|---|---|---|
| 1 | routing + fire | 66/272 censored | 36/272 censored | 64/272 censored |
| 2 | control | 272/272 within-5s | 272/272 | 272/272 |
| 3 | routing + fire | 40/272 censored | 61/272 censored | 125/272 censored |
| 4 | control | 272/272 within-5s | 272/272 | 272/272 |

Control target settle 4.27-4.81 s. Two earlier instrumented triples on the same
arm censored identically (36/51/90, 88/33/74, 70/31/115, 72/38/77), so the
verdict rests on 18 routed fills, not six.

Bisected, as the previous attempts were:

| arm | run 0 | run 1 | run 2 |
|---|---|---|---|
| routing only (`fire()` back on the timer) | 108/272 | 29/272 | 35/272 |
| `fire()` only (routing narrowed) | 18/272 | 5/272 | 5/272 |

The `fire()`-alone arm reproduces slice B's storm unchanged, so **the coupling
claim is still true** — it is the routing that fails, in both of its arms.

### The condition was met, measured on the failing arm itself

GUI-side holds, instrumented per calling funnel (the lock context itself,
attributed to the first caller frame that is not the decorator or `_order_lock`),
three routed fills:

| GUI section | calls | total ms | max ms |
|---|---:|---:|---:|
| `_commit_tile_layer` | 74-132 | 294-604 | 15-20 |
| `_submit_prepared_transaction` | 70-128 | 194-531 | 16-20 |
| `retarget_frame_pipeline` | 12-87 | 16-162 | 4-6 |
| `commit_pending_session` | 74-132 | 19-31 | 2-6 |
| `apply_montage_viewport_retarget` | 3 | 141-224 | 49-126 |

Against D2a's 526 ms and 194 ms worst cases, every per-fill section is now
under a frame except a three-call viewport retarget that is not fill traffic.
**The precondition holds and the fill still censors**, which retires it as the
open question.

### What the counters say instead

| per fill | routed arm | control arm |
|---|---|---|
| `events_dispatched` | 637-662 | 2 |
| `owner_lock_acquisitions` | 692-719 | 3-5 |
| `gui_lock_acquisitions` | 280-356 | 1210-1327 |
| `gui_lock_wait_ms_total` | 984-1407 | 0.7-192 |
| `gui_lock_wait_ms_max` | 135-221 | 0-100 |
| `owner_lock_wait_ms_total` | 147-331 | 0-6 |
| `owner_yields_to_gui` | 31-41 | 0-4 |
| `errors` | 0 | 0 |

Neither side starves the way stage 1 (owner 5.4 s of 8.6) or D2a (owner 1.8 s
of ~8) did, and no wedge is left behind: `pending_events` and
`pending_messages` are zero at teardown, `errors` zero, `messages_dispatched`
52-59 against 637-662 events — so the replan gate's coalescing *works* once the
barrier is real, which is the one thing this attempt proves positively.

The cost is the bridge itself. ~700 owner acquisitions and ~300 GUI
acquisitions per fill on one `RLock`; the GUI's mean wait per acquisition is
3.5-4.2 ms, which is nothing, and its total is 1.0-1.4 s of an ~8 s fill, which
is everything. The stand-aside trades a turn and cannot make 700 alternations
cheaper than 700 alternations.

Two second-order numbers say the same in the fill's own units:

- **Cohorts collapse.** Control `1,1,3,4,25,68,68,68,34` and `1,1,3,91,91,86`
  against routed runs of ones and eights,
  `1,1,1,1,1,1,1,2,128,13,16,8,8,4,68,18` and
  `1,1,1,1,15,1,2,1,1,1,1,1,1,1,8,8,8,8,8,8,8,8,8,8,8,8`. The commit gate's
  batching never came from the gate — it came from admissions being on the
  *commit's own thread*, so none could land inside a commit. Routed, every
  commit re-arms the gate while it runs and the next one presents only what
  arrived during the previous one.
- **Throughput per callback-second** falls 412-472 -> 112-294 items/s, and one
  routed target pass admitted 437 items for 272 tiles — the same tiles
  re-committed. Target chunks go 14-19 (control) -> 20-41.

### Draw-rate structure (informative, per the slice's ask)

`profile_montage_workflow`'s cold-fill record already carries the draw counters,
so the probe was made to print them rather than reconstructing from the trace
ring. Per fill, tile-presentation draws / total presentation draws / wall ms:

| arm | run 0 | run 1 | run 2 |
|---|---|---|---|
| routed | 31 / 40 / 5439 | 35 / 46 / 5543 | 34 / 42 / 5258 |
| control | 32 / 42 / 5072 | 24 / 33 / 4561 | 25 / 35 / 4865 |
| control (2nd block) | 28 / 38 / 4774 | 21 / 32 / 4287 | 25 / 35 / 4665 |

**Draw rate is not what changes**: 5.3-6.4 draws/s on both arms. The control
puts 272 tiles on screen with those draws and the routed arm puts 36-66. The
program's adjudicator therefore says the same thing the settle column does, and
from a direction that rules out frame pacing as the mechanism — each draw
simply carries less.

### What the next attempt owes, named by the measurement

1. **The migration bridge has to stop being one lock for the admission path.**
   Every previous slice shortened a section; this one shows sections are no
   longer the cost, hand-off count is. A per-family lock, a reader/writer split,
   or admissions writing state the commit reads through a snapshot — all of them
   new mechanism, none of them a shrink.
2. **The commit gate needs a barrier of its own.** Its cohorts were an accident
   of single-threading, and any design that moves admissions off the commit's
   thread has to replace that accident deliberately — otherwise cohort size
   becomes "whatever arrived during the last commit", which is what censored
   here.
3. `fire()` on the owner is *correct and coalescing* (52-59 messages against
   637-662 events, zero pending at teardown) and remains blocked only by the
   routing it is coupled to. Nothing new is owed there.
4. Three non-Qt thread-safety items surfaced by the audit above
   (`operation_evaluator`'s unlocked display cache, `MontageLevelTracker`, the
   lazily created `preparedTiledUploads` mailbox) become real the moment
   admissions run off the GUI thread, and none of them is currently defended.

## Slice D3b: both named blockers were built, and the diagnosis behind one of them is refuted (2026-08-03)

D3 closed with two owed mechanisms rather than another shrink. Both were built
and measured together, and the routing is refused a **fifth** time. The value of
this attempt is not the verdict — it is that the counters retire D3's own
explanation of the fourth one.

### What was built

- **M1 — batched owner lock sections** (`arrayscope/presentation/owner.py`).
  `_dispatch_event_batch` replaces the per-event `_dispatch`: one
  `timed_order_lock("owner")` section covers up to `_EVENT_BATCH_LIMIT = 8`
  events or `_EVENT_BATCH_MS = 8.0` of dispatch, `_stop` re-checked per event,
  then release and `_yield_to_gui` between batches. Counters
  `owner_batch_acquisitions` / `owner_events_per_batch_max`. The module
  docstring's "per event, never per batch" rule is rewritten with the reasoning
  that made it right in stage 1 and the measurement that flipped it.
- **M2 — a cohort barrier that does not need thread co-location**
  (`PresentationOwner.defer_to_batch_tail`, called from
  `FramePipelineEffects.request_presentation`). Arming is unchanged — the
  gate-owner flags stay the single-flight key — and only the Qt post is
  deferred, to the point where the event deque empties, ahead of the posted
  messages. Refused (so the caller posts inline, keeping its own wakeup) off the
  owner thread or with nothing queued behind the caller. Counter
  `batch_tail_deferrals`; a new `armed-batch-tail` trace action.
- The D3 change verbatim: `montage_completion_route` widened to
  `("frame:", "montage-stage-plan:")` and `request_montage_replan`'s `fire()`
  converted to `post_to_presentation_owner`, with both test consequences D3
  recorded (the timer census keeps its entry because the no-owner fallback keeps
  the timer; the superseded-retarget oracle moves to a `force_commit=True`
  marker plus a sentinel message).

The probe learned to print what this needs: `presentation_owner` (the whole
counter snapshot) joins the phase record, and the fill table carries wall ms,
tile-presentation draws, total draws and the owner counters.

### The measurement: 9/9 censored against 9/9 complete

`render_pass_governor_probe --backend wgpu --tiles 272 --repeat 3`, arms
alternated in one loop on one box (three routed invocations, three control):

| arm | invocation | run 0 | run 1 | run 2 |
|---|---|---|---|---|
| routed + M1 + M2 | A | 100/272 censored | 75/272 | 108/272 |
| routed + M1 + M2 | B | 59/272 censored | 112/272 | 161/272 |
| routed + M1 + M2 | C | 108/272 censored | 137/272 | 91/272 |
| control (M1+M2 in tree, routing narrowed) | 1 | 261/272 late-milestone | 272/272 within-5s | 272/272 |
| control | 2 | 264/272 late-milestone | 272/272 within-5s | 272/272 |
| control | 3 | 272/272 late-milestone | 272/272 within-5s | 272/272 |

Control settle 4.59-4.99 s. The control's *first fill per process* reports its
target-settled milestone late with 261-272 tiles presented, on every invocation
and on both arms; it is environmental (this box, this week) and is nothing like
the routed arm's 59-161 presented.

Bisected, one invocation each:

| arm | run 0 | run 1 | run 2 |
|---|---|---|---|
| routed, M1 only (M2 refused) | 95/272 | 143/272 | 117/272 |
| routed, M2 only (`_EVENT_BATCH_LIMIT = 1`) | 82/272 | 114/272 | 70/272 |

Neither half rescues anything, and neither half is the one that breaks it.

### What the counters retire

| per fill | routed, batch 8 | routed, batch 1 | control |
|---|---|---|---|
| `events_dispatched` | 636-649 | 634-649 | 2 |
| `messages_dispatched` | 48-53 | 48-53 | 1-3 |
| `owner_lock_acquisitions` | 267-303 | 682-702 | 3-5 |
| `owner_batch_acquisitions` | 216-255 | 634-649 | 2 |
| `batch_tail_deferrals` | 12-17 | 15-16 | 0 |
| `gui_lock_acquisitions` | 228-301 | 261-300 | 1167-1280 |
| `gui_lock_wait_ms_total` | 1010-1577 | 933-1460 | 1-132 |
| `gui_lock_wait_ms_max` | 138-176 | 148-167 | 11-80 |
| `owner_lock_wait_ms_total` | 228-297 | 285-299 | 0-21 |
| `owner_yields_to_gui` | 22-38 | 32-39 | 0-3 |
| `errors` / `pending_*` | 0 | 0 | 0 |

**M1 did exactly what it was designed to do and it bought nothing.** Hand-offs
fell 2.4x and the GUI's total wait did not move. D3 read the fourth refutation
as hand-off count ("there is no policy under which 700 alternations cost less
than the hand-offs themselves"); the correct quantity is the owner's
**occupancy** of the bridge — ~640 admission bodies holding a lock the GUI
needs — and an occupancy is invariant under every batching, priority and
stand-aside policy. Only removing the admission path from this lock changes it.

**M2 failed for a structural reason, not a tuning one: the drain batch is not a
barrier, because the owner is never behind its producers.** The deferral was
accepted 12-17 times per fill against ~640 admissions — every other gate arm
found the deque already empty, so "the tail of the batch" *was* now. What made
the single-threaded predecessor's cohorts large was the GUI thread being busy
inside a 20-45 ms commit while completions queued in the same loop. The owner
has no equivalent occupancy, and 8 ms of batching does not manufacture one.

### The cohort collapse latches through R5, which is new

The governor's own fit, printed per pass by the probe:

| arm | target-pass fit | frame-critical |
|---|---|---|
| control | `fixed=2.4-8.0ms item=0.27-0.79ms` | 0 |
| routed (all three bisect arms) | `fixed=0.00ms item=11.2-16.0ms` | 1 |

That is the documented fill bistability (`arrayscope-fill-bistability`): small
cohorts misattribute the commit's fixed cost to the item term, the model prices
any larger chunk out of the 16.67 ms frame budget, and the chunk stays at 8 for
the rest of the pass — `8,8,8,8,…` in every routed target pass against control
runs of `32,32,32` and `68,68,68`. So cohort size is not merely a symptom of
where the admissions run: **it latches**. A barrier for the next attempt has to
produce a large *first* cohort, not a large average one.

Throughput per callback-second, target pass: control 404-527 items/s, routed
47-197. Draw rate is unchanged for the second time (control 20-32 tile draws per
fill, routed 23-31, comparable wall clock), which rules out frame pacing again.

### What is kept, and what the next attempt owes

Kept: M1, M2, their pins, and the probe's owner/draw instrumentation. Reverted:
the routing clause and the `fire()` conversion, together, as always. With the
clause narrow both mechanisms are **inert and measured so** — the batch holds
the ~6 stage-plan bodies a fill produces, and the deferral is refused 100% of
the time — so keeping them is a decision to hold the scaffolding, not a claim
that either buys anything today.

1. The blocker is unchanged in *shape* and sharper in *statement*: the
   admission path must stop sharing one lock with the commit. What D3b adds is
   that no policy on the drain's side can substitute, because the cost is
   occupancy. A reader/writer or per-family bridge, or admissions writing state
   the commit reads through a snapshot, is the only family left.
2. The commit-gate barrier must be stated on the gate's own terms — a quantity
   of pending admissions, or a deadline — because the drain has no accumulation
   to be the tail of. And it must be measured against R5's fit, not only
   against cohort size, since the collapse is bistable.
3. `fire()` on the owner remains correct and coalescing (48-53 messages against
   ~640 events, zero pending at teardown) and blocked only by the routing.
