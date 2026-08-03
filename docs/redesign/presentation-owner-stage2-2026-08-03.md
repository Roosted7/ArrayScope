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
