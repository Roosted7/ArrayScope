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
