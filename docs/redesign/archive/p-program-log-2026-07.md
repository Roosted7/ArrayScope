# P-program execution log (2026-07-14 → 2026-07-15)

Verbatim record of the measured performance program (P1–P9) that ran on
`main`/`codex/redesign-p8` after V4. Extracted from `docs/roadmap.md` on
2026-07-16 when the roadmap was slimmed; evidence, never direction.
The rejected experiments are indexed in [`../../graveyard.md`](../../graveyard.md).

## Now — measured performance and suite truth (post-redesign)

> **[Codex 2026-07-14 — post-V4 roadmap update; linear-history correction]**
> R1–R7 and V0–V4 are rebased linearly onto `main`; no integration merge
> remains. The fixed viewer passed the V1/V2 real-Wayland
> pixel/trace scenarios on both backends and the V3 loud-stall injection.
> The final pre-integration non-GPU suite remained red at 42 failures and 2
> teardown errors; this is tracked work, not a green-suite claim.

Proceed with the redesign P-program one measured cause at a time against
the frozen T1 baseline, in the order recorded in
[`docs/redesign/marathon-salvage.md`](../marathon-salvage.md):
prefetch-busy → committed `level_source` → viewport intent → background
histogram aggregation → coalesced completion drain → cadence throttle →
stage-cache snapshot/cancellation → governor policy → admission batching →
gate pacing → slot relocation. Every P commit carries before/after trace and
benchmark evidence; the real-display pixel/trace gates must stay green.

> **[Codex 2026-07-14 — P1 result]** Narrowing prefetch-busy was measured on
> both backends, produced no FFT improvement, regressed the scalar elapsed
> sample, and did not change PyQtGraph's 50/60 presentation freeze. The code
> was reverted and the rejected measurements are recorded in the redesign
> README. The active measured cause is now committed-frame `level_source`.

> **[Codex 2026-07-14 — P2 result]** Committing `level_source` removed the
> workflow's first-evidence-quality failure but regressed the real VisPy V2
> priority gate from 14/16 to 4/16 nearest first-cohort tiles and exposed a
> rough-bounds relative-window error. The code was reverted; the redesign
> README records the measurements and missing maturity rule. The active
> measured cause is now viewport-intent replay.

> **[Codex 2026-07-14 — P3 result]** Acknowledged content-extent changes now
> replay AUTO/FIT without moving USER cameras, including VisPy's hidden-bounds
> update. Focused and real-pixel gates pass on both backends. The canonical
> USER-camera workflow remained stalled at the same 7/60 presentation state,
> so P3 carries no performance credit and the stall remains open. The active
> measured cause is now background histogram aggregation.

> **[Codex 2026-07-14 — P4 result]** Aggregate histogram sampling is now
> revision-guarded kernel work; its deterministic 60-tile selector is 9.8×
> faster and the real trace moves up to 36.6 ms per aggregate off the GUI
> thread. Prepared atomic transactions now include level revision, closing a
> stale-level identity defect exposed by the new wake. Broad display/window
> coverage is 840 passed and both physical gates remain green. The unrelated
> 7/60 presentation stall persists. The active measured cause is now the
> coalesced kernel completion drain.

> **[Codex 2026-07-14 — P5 result]** Coalescing 205 completions into 42
> bridge drains bounded the observed drain callbacks, but all three capacity-
> wake variants failed the real VisPy priority gate with 36/36 exact targets
> stranded at preview quality. The runtime experiment was removed and the
> failed designs are recorded in the redesign README. The active measured
> cause is now LOD-plan cadence plus synchronous-title removal.

> **[Codex 2026-07-14 — P6 result]** Wheel/pan-derived replans now have a
> committed-frame-only 16 ms cadence, separate from programmatic replay and
> pipeline continuation; range input no longer performs synchronous title
> layout. Two real traces cut kernel submissions 39–50% and bridge drains
> 61–65%, with a +0.8% two-run first-ack midpoint. The V1/V2 physical matrix
> and 873 focused/broad tests pass. The 7/60 deadlock remains unchanged. The
> active measured cause is now the stage-cache resident snapshot,
> cancellation tokens, and `peek_many`.

> **[Codex 2026-07-14 — P7 result]** Hot stage reuse no longer acquires the
> mutation lock, preview-floor probes batch 60 potential cache locks into one,
> and cancelled render results stop between evaluation/reduction boundaries.
> The deterministic lock-contention regression, 1,297 broad tests, and all
> four physical gates pass. The workflow sample stayed within the ±10% latency
> guard but did not improve the 7/60 deadlock. The active measured cause is now
> governor lane policy.

> **[Codex 2026-07-15 — P8 result; correctness only]** Interaction lane
> quotas and a plan-wide preview barrier prevent exact/ROI work from
> overtaking the visible preview pass; canonical priority reaches execution
> and both backend admission paths. Source-successor and level-generation
> feedback loops now converge, synchronous viewport continuation and VisPy
> draw acknowledgement cross receiver-owned Qt turns, and the full non-GPU
> suite is **1,955 passed, 8 skipped**. Real-Wayland V1/V2 is green on both
> backends and both complete workflow runs reach their final phase, but the
> 16 ms heartbeat and throughput bars remain red. The active measured cause
> is now presentation admission batching; PyQtGraph exposed an untrained
> shared commit-feedback channel and 17-25 ms commit samples.
>
> **[Codex 2026-07-15 — P9 correction after real-VisPy gate]** Two bounded
> admission designs improved some PyQtGraph phases but are rejected and fully
> reverted: one-tile presentation batching regressed both scrolls and stranded
> exact LOD, while completion-owned refill nearly doubled VisPy scalar-scroll
> time and coincided with visibly mixed FFT-scroll tiles. The active step is
> correctness-first physical identity rebinding plus center-out/preview-order
> proof on both backend cache implementations; no further scheduler tuning is
> allowed until that gate is clean.
>
> **[Codex 2026-07-15 — P9 shared-transform checkpoint]** The zero-work
> session-25 stall is closed at the shared-transform owner: retained finer
> previews remain producers after a coarsening retarget, and target work waits
> for unique required-tile physical coverage.  Real PyQtGraph raw and VisPy
> FFT-full each converge 272/272 with clean trace replay, but performance gates
> remain red and the trace cannot certify the user's transient wrong pixels.
> The next single slice is the backend order boundary, not scheduler tuning:
> VisPy currently receives ordered upserts but uploads by numeric active-grid
> order, while the acknowledgement report setifies them.  Preserve command
> order through physical work and its trace first; then replace the harness
> event count with required-identity coverage and add framebuffer comparison
> plus an injected wrong-uniform/page test before any further P9 throughput
> experiment.
>
> **[Codex 2026-07-15 — P9 ordered-presentation checkpoint]** One semantic
> focus now drives current-layout shared fanout, ordered deltas, VisPy physical
> upload, and ordered acknowledgement. Real-display traces changed the stale
> first FFT cohort to current centre-out order. Physical page evidence then
> isolated the broken pixels to stale shader state rather than dtype/source
> corruption: levels-only updates now update levels only, and touched atlas
> pages synchronize their page-local mapping even when the layer-wide key is
> unchanged. `/tmp/arrayscope-vispy-onscreen-page-fix` has 60 coherent preview
> tiles and zero physical identity mismatches. Performance remains vetoed
> (31.75 s scroll, 123 ms heartbeat, worst continuity 1/60), and the immediate
> next slice is the separate 58-ready/2-presented zero-work settlement hole.
> General framebuffer comparison follows that settlement fix. Consolidating
> scalar and batch montage cache-key derivation is tracked separately; its
> runtime parity fallback must remain until one canonical key owner replaces
> both implementations.
>
> **[Codex 2026-07-15 — P9 level-truth/admission checkpoint]** A real trace
> proved the last settlement hole was a VisPy cache/physical-level split, not
> repeated FFT evaluation: exactly one full-volume shared FFT stage was
> submitted. Tiled commits now reassert levels when either public or physical
> layer state disagrees with the command, even if the completed-command cache
> matches. After this fix, the formerly rejected four-item minimum cohort no
> longer livelocks and cuts real FFT refinement from 42.36 s to 8.39 s; trace
> replay is clean. P9 remains active because the scripted scroll is 13.55 s,
> continuity and heartbeat are red, and each source-window step still pays
> for roughly seventeen bounded presentation transactions.
> A 4 MiB minimum byte-cohort follow-up was measured and reverted: the item
> cap remained binding and scroll regressed 13.55 s -> 13.79 s. The next P9
> cause must therefore address item/transaction ownership, not byte tuning.
> Generic eight-item and `not display_committed` eight-item variants were also
> reverted: the first improved cold refinement but regressed scroll/convergence,
> while the second lost the cold win and worsened scroll to 15.63 s. A session
> flag is not an acceptable proxy for the plan phase.
>
> **[Codex 2026-07-15 — P9 source-generation result]** The reused
> `FrameSession` no longer carries a sticky atomic-successor boolean across
> index-window retargets. Completion is bound to the current `session_id` and
> requires a coverage-complete backend acknowledgement. Real VisPy evidence
> changed the ordinary FFT-scroll shape from roughly seventeen partial commits
> per window to one ordered 60-slot transaction, with a clean trace replay and
> coherent final pixels. It did **not** improve the performance gate: staged
> FFT scroll measured **15.75 s** against the accepted **13.55 s** baseline.
> P9 therefore continues at the now-isolated cause: avoid preparing/rebinding/
> acknowledging all 60 slots for a one-index move while preserving atomic
> visible identity. Slot relocation or an equivalent source-keyed remap is the
> relevant ownership boundary; generic cohort tuning is not.

In parallel only where it does not reorder a P-step, migrate stale tests to
the canonical `window.renderer` / `FrameSession` owners and fix the remaining
coalescer, levels, viewport/ROI, cache-rebind, and transition behavior. Do
not weaken user-visible assertions merely to make the suite green.
