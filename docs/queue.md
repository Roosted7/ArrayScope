# The queue — what to do next

**This is the only active queue.** If any other document claims to order
current work, that document is stale — fix it to point here.
[`roadmap.md`](roadmap.md) says *why* this order serves the mission;
this file says *what, in what order, and when it counts as done*.

**Rules for this file** (they exist because the last three queues drowned):

1. Update rows **in place**. When a step lands, move its row to *Done* below
   with one line of result + a link to the evidence. Never append status
   blockquotes or execution logs here — those go in the commit message, a
   dossier under [`redesign/`](redesign/), or a dated review.
2. Every step names its **exit gate in the ring that can actually see the
   failure** ([testing/README.md](testing/README.md)). "Code exists" and
   "offscreen suite green" are not completion.
3. A rejected/reverted attempt gets a [`graveyard.md`](graveyard.md) row in
   the reverting commit.
4. Re-order only with a stated reason in the commit message.

## Now (2026-07-16, in order)

| # | Step | Exit gate |
|---|---|---|
| 1 | **Performance-bars program on the engine** (parked — Thomas 2026-07-17: act only on true stalls/no-progress, never on merely-slow). The bars (below) are the product promise. One measured cause at a time, before/after real-Wayland harness evidence per commit; a step that regresses a bar is reverted and buried in the graveyard. **Histogram evidence follow-up complete 2026-07-23:** normal WGPU frames now reuse materializer summaries and reserve resident GPU histograms for resident-only content; the dtype/storage/device matrix and real-window continuity gate are recorded in the [dossier](redesign/histogram-evidence-pipeline-2026-07-23.md). The independent 50 ms callback bar remains red. **Resident crop rebind is now default ON (2026-07-25):** an index-window retarget that found an existing stage plan left an undrainable `deferred_missing_tiles` backlog, so an operation-pipeline montage never reported `is_complete()` and never refined its auto levels on ANY path; pairing the backlog with its deferral flag closed that, and the rebind and evaluation paths now settle identical levels on raw and op-pipeline montages (102-143 ms/step with zero display producers against 305-770 ms with up to 50 — [dossier](redesign/histogram-evidence-pipeline-2026-07-23.md)). **Cold-fill cohort clamp fixed 2026-07-25:** a field montage load took 15.4 s with no worker active — the wgpu plane-warm clamp collapsed every *idle* commit to two upserts (it qualifies on `lod.level > 0` alone, so a zoomed-out montage is all plane warms), turning a 272-tile fill into 143 whole-montage transactions at ~93 ms each. Byte caps now govern the idle cohort and the item clamp keeps only the interactive arm: **15.4 s → 4.05 s**, with the per-revision LOD resolver rebuild (147 696 binds → 543) taken alongside ([dossier](redesign/montage-cold-fill-cohort-2026-07-25.md)). The 50 ms callback bar is still red on the fill at 551 ms. **Per-commit whole-montage cost diagnosed 2026-07-26, nothing landed:** the preview needs 9 transactions because `_idle_backlog_cohort`'s `min(32, backlog)` ceiling binds exactly, the refinement needs 1–2 because a resident retarget is *free* in `TileAdmissionQueue` and skips every cap, and worker arrival pacing is refuted on the clock (all 340 preview evaluations land 178 ms before the drain starts). The transaction is whole-montage by construction — 63–69% of it is `_apply_backend_tiled_presentation` iterating every presented payload, and an empty-delta commit still costs 90 ms. **The item ceiling is not the lever; the byte cap is** (9→5 batches measures flat, 9→4 with the byte cap measures −32% stage over 6 order-balanced passes), and §4c's "a commit costs 120–170 ms regardless of what it carries" is retracted as a 2-tile-regime extrapolation. Not landed because 512 MB is not a byte cap; the four things that would justify a bound are listed in the [dossier](redesign/per-commit-transaction-count-2026-07-26.md). **Follow-ups the same day (§8):** the 2.32× upload undercharge is fixed (`cbd85384` — measured to change nothing on this workload, because 32 MiB ÷ 1 MiB is the same 32-item cohort the item ceiling already produced, which is exactly why it was invisible); admission now records the cap that *bit* rather than the cap in force (`0d51e136`), so a supply-starved batch reads `limit=""` with `candidates == items` instead of looking like variance — and on its first run it showed the *healthy* 2-batch refinement is supply-bound too, moving the bistable-refinement hunt off the commit caps and onto the retarget/lifecycle side. The "two empty commits" premise was **refuted**: they are level/histogram metadata publications, and the real statement is that a metadata-only commit drags a whole-montage tile republication with it. **Pixel-identical WGPU refinement priced 2026-07-26, nothing landed:** on the zoomed-out raw montage the desired CPU pages never become WGPU pages, but their summaries move level evidence from `ROUGH_PREVIEW`/rank 2 to `ROUGH_TARGET`/rank 4 and their ACKs are the current target-settlement token. An evidence-only probe reproduced the exact normal bounds with zero exact payloads or ACKs; suppression stayed incomplete, while a later physical zoom draw needed zero uploads in either arm. **Mapping-only WGPU publication landed after ADR 0059 (2026-07-26):** the canonical token was already `TileIdentity`, below the fresh level-bearing wrapper. The predicate adds explicit real/imag plane records, representation/mapping mode, layout/transpose, executor, and page-table generation, and refuses every upsert so target settlement remains separate. Across ten order-balanced real-Wayland processes (15 in-process passes per arm), 30/30 metadata commits took the fast path with zero resident rebinds versus 26/26 full republishes and 7,072 rebinds; the load-resistant lower quartile moved 89.0 → 53.8 ms. Scroll/zoom exercised 80 safe fast commits with zero stale or coarse-rung pixel failures ([per-commit dossier §8.5](redesign/per-commit-transaction-count-2026-07-26.md)). Next decouple source/stage evidence and an honest unchanged-binding settlement ACK from display presentation ([dossier](redesign/wgpu-refinement-consumer-price-2026-07-26.md); [retention details](reviews/2026-07-22-compression-live-benefit-review.md)). | Bars trend green in `profile_montage_workflow` on real Wayland, both backends (PyQtGraph at 2× allowance) |
| 3 | **wgpu strangler — promotion evidence** (ADR 0057; slices (a)–(c) LANDED: native GPU overlays + glyph text, screen presentation behind `wgpu_present_method`, G6 compute, dogfood-crash/eviction-shield and codex-review fixes — full status narrative in the [Done ledger](queue-done.md) and dossiers). Open: **(d)** promotion by evidence — wgpu LEADS fast-scroll (p95 77.3 ms vs VisPy ~106–124), matches zoom/pan steady-state, 5/5 in the final matrix. **2026-07-20 correctness blocker — CLEARED 2026-07-21:** compositor screenshots showed native screen-mode overlay boxes not matching the Qt/PyQtGraph presentation, with the histogram/top-right composition clipped. Root cause was promoting the overlay chips to native *child* windows, which Qt never grants an ARGB visual (`alphaBufferSize() == 0`), so translucent rounded chips flattened into opaque boxes and occluded the histogram; restacking the swapchain below was unavailable too (`QWindow.lower()` emits no `place_below`). Chips are now rasterized from Qt's own painter and composited inside the frame (`widget_quad` + `UpdateWidgetAtlas`). Tool-managed Weston full-window evidence vs the bitmap reference went 3.45% -> 0.08% differing, the residue being only a Weston-drawn cursor, one transient live readout, and the bitmap-vs-bitmap noise floor. Tool-managed Weston or manual full-window compositor evidence stays required before AUTO promotion. **2026-07-21 successor review:** physical scroll evidence retains 60/60 tiles without stale/black holes, but WGPU shuffle first-new pixels remain red at 2.55 s (2 s bar). The final successor needed only two missing preview tasks (23 ms finish span) and reused the prepared atomic transaction about 129 ms later, so the remaining delay is in retarget/input trajectory, not a 60-tile cold-data or last-payload barrier. Built-in 0.1 s full-window capture is correctness evidence only: its synchronous compositor/readback load materially perturbs timing. One managed-Weston WGPU scroll run aborted in `wgpuSurfaceConfigure` after `ERROR_SURFACE_LOST_KHR`; an immediate identical rerun passed 60/60, so keep this as promotion reliability evidence if it recurs rather than hiding it as a harness timeout. VisPy's ~9 s comparison is diagnostic only: it reached session 10 where WGPU reached 42 and had a ~402 ms vs ~181 ms maximum event-loop gap. Do not optimize retiring-VisPy latency unless the same signature appears in WGPU/shared code. PyQtGraph's pre-existing 7/10 three-second LOD misses remain a separate standing red. **2026-07-24 cropped-axis follow-up:** the maintained-backend profile now scrolls every dimension at fast and slow cadence with physical CPU-reference checkpoints; stale session anchors, mutable preview anchoring, and predecessor-owned presentation events are fixed. A full 272-plane offscreen Vulkan run completed 272/272 after visible admission counted 1088 submitted native pages instead of the 272 preview pages; one-shot frame-plan capacity cut growth copies from 2.88 GB to 6 MB. **2026-07-25:** with the fixture's viewport gate fixed those two crop stages run again, and their newly visible `display_axis_all_dimension_pixels_match_cpu_reference` red was a reference-model gap, not a crop defect — the mismatching pixels were the fixture's ROI strokes and a composited Qt chip, each verified to sit exactly where its own geometry projects after the crop. The image comparison now withholds overlay-covered pixels (a stroke drawn anywhere else still fails it) and a new `display_axis_roi_overlays_track_geometry` gate projects the semantic ROI store to require each outline be painted in its own band and nowhere else; 12/12 checkpoints green on both stages, wgpu and PyQtGraph ([dossier](redesign/montage-cold-fill-cohort-2026-07-25.md)). Real-Wayland screenshot/journey acceptance is still required before this contributes promotion evidence. Before the AUTO flip: shared row-1 callback bars, dogfood hours (screen mode selectable from Performance → wgpu Presentation), the FFT-scroll 4→17 fps headline on the new tip. VisPy retirement only via the roadmap ladder — never a flag-day switch. **Successor traps:** import rendercanvas ONLY via `import_qrenderwidget()`; every wgpu adapter probe pins `set_instance_extras(backends=["Vulkan"])` BEFORE its first request (`9c115686`). | Promotion gate: journey matrix + perf bars on real data; written verdict in tensor-engine-endpoint.md |
| 4 | **Large-data retention truth before compression.** Instrument a representative lazy 4+ GiB session so source, evaluator display payloads, exact ROI-demand regions, `StageCache`, LOD/retained payloads, and the selected backend's physical storage report owned bytes, alias-adjusted bytes, evictions, costly recomputes prevented/repeated, and RSS/device deltas. Do not merge the evaluator caches: they have different consumers. Do not add PyQtGraph raster and GPU page residency: they are alternative backend mechanics. Use the new `memory_retention_audit` as the static envelope, then decide whether the ROI cache needs its own smaller policy budget and which retained values deserve to exist. **2026-07-26 WGPU input:** the zoomed-out raw montage retained 14.9 MB/604 CPU pyramid entries and 285.2 MB of active level-0 GPU pages; its reduced CPU pages supplied histogram evidence and lifecycle tokens but were never uploaded, and a later physical zoom draw uploaded zero pages with or without the desired CPU rung. Price CPU evidence retention separately from backend display retention; the claim does not yet cover cropped scroll, zoomed-in single image, or PyQtGraph ([dossier](redesign/wgpu-refinement-consumer-price-2026-07-26.md)). **Compression remains parked:** demand-sized pools fixed the full mirror, but AUTO is still slower in physical-tile evidence. Next establish one physical byte cap and measured eviction/re-upload benefit ([details](reviews/2026-07-22-compression-live-benefit-review.md)). | Fresh-process lazy/eager runs on PyQtGraph, VisPy, and wgpu reconcile owner counters with RSS/device allocation; a written keep/evict/resize decision for every owner; no unbounded store; compression remains parked unless the trace identifies its exact bottleneck and owner. |
| 11 | **Operations as a platform** ([proposal](proposals/operations-as-a-platform.md), [ADR 0060](decisions/0060-operation-definitions-runtimes-and-discovered-shapes.md)). **Bundle A DONE 2026-07-26:** the native dtype-honest everyday toolbox covers complex/display maps, scalar maps, thresholds, per-axis normalization/statistics, roll/pad/fractional-resample/transpose/squeeze, difference/gradient/cumulative sum; redundant SigPy/BART specs are gone while their availability/cfl/subprocess seams remain. The [Numba review](reviews/2026-07-26-native-operations-numba.md) landed normalize (6.5–7.0x last-axis, 2.1–2.3x with an axis-copy) behind the shared lazy fallback and rejected otherwise-faster log/threshold kernels because mixed region paths failed the exact oracle by 1–2 ULP. Remaining, in dependency order: **B — DONE 2026-07-26** (`ce2203fe`, `6d273c09`, `a396a16f`): every registration tier renders the shared wrapper-shaped definition; Duplicate creates a selected editable native copy or an explicit provider adapter, while shape-changing copies are loud templates pending D rather than false shape claims; `_OperationImportDialog` is deleted and New/source/callable/copy-link/full parameter metadata live in the single resizable manager editor; all six editor states have reviewed dark/light gallery coverage. **Bundle A gallery input for B:** the nine-item Common section makes the 272×491 collapsed popup scroll immediately, while the captured “More” state is pixel-identical because both optional packs now have zero entries; revisit the fold and Common density with the phase-6 popup redesign, not in A. **C** the hidden subprocess bridge (`run_bart` already takes an arbitrary argv) promoted to a user-editable command-template runtime plus named execution environments (interpreter/conda/venv/cwd/env vars, `BART_TOOLBOX_PATH` stops being special-cased), and BART re-expressed as copyable examples. **D** probe-and-cache shape/dtype/windowability adjudication unified with the Tier-2 conformance harness, lifting the `changes_shape` refusal. **E** input slots for ROIs and second arrays, which is what `pics`/`ecalib`/NUFFT actually need. | Per bundle: the all-ops smoke harness covers the new set on float32/complex64; a duplicated built-in yields a user op field-identical to the original; a command-template op runs end-to-end against a fake binary with proven timeout/cancellation; a shape-changing user op survives tiles/LOD without lying to the planner; one real two-input recon runs from the UI. Every bundle that adds a surface adds its gallery scenario in the same bundle. |

## Product turn — completed 2026-07-22 (rationale in [roadmap.md](roadmap.md) and [reviews/2026-07-19-course-review.md](reviews/2026-07-19-course-review.md))

**The product turn (steps 5–10) is COMPLETE 2026-07-22.** Compare v1 (5/6/7),
plugin ops v1/v2/v3 (8, Tier-2 conformance harness for 9, BART pack for 10),
and the G7 codec groundwork landed — see the [Done ledger](queue-done.md). The
sigpy FFT pack that first accompanied step 9 was removed as redundant with the
built-in FFT (see [graveyard.md](graveyard.md)); the later threshold/resize pack
landed 2026-07-24 as useful plumbing evidence and was demoted by row 11 Bundle A
once native dtype-honest replacements existed. G7's live product-benefit gate
closed with a measured NO. Row 4
audits what should be retained before any codec revival; codec implementation
remains in the review/ideas list unless that audit identifies a real owner-level
bottleneck. `sigpy`, `blosc2`/`zfpy`, and `bart` were installed into the env this
session, so the previously dep-blocked gates ran on real data/tools. Deferred,
with reasons recorded in the ledger: sigpy `nufft`/`espirit`/`fwt`/`iwt` and
`bart:pics` (multi-input / structural-metadata args that don't fit the unary
`fn(ndarray)->ndarray` contract).

## Performance bars (commitments, not history — restored from R2/R4/R8D)

- GUI callbacks < **50 ms** always; event-loop heartbeat gap ≤ **16 ms**
  during fills and scrub; warm scrub input ≤ **15 ms**; settled-idle CPU 0%.
- **#1 throughput target:** fast montage FFT index scroll ~4 fps → toward
  the ~17 fps scalar rate (2026-07-09 measurement, realistic human scroll).
- Benchmark deltas stay within ±10% of the frozen baseline unless a step
  improves them. PyQtGraph gets 2× the VisPy allowance (it targets
  headless/remote use); both backends stay first-class for correctness.

## Standing lane — test hardening & debt (parallel-safe, any order)

Safe to pick up alongside the numbered queue; each is self-contained.

- **ADR 0059 coarse-rung utility policy — OPEN; phase-only shape refuted.**
  The landed per-tile successor lets exact ACKs overlap the remaining coarse
  pass by 2.5–3.2 s. Reclassifying `DESIRED` onto `DISPLAY_PREPARATION`
  enforced global order but regressed the order-balanced whole-fill median
  5400.0→6651.4 ms (+23.2%) and was reverted; do not revisit that shape. The
  policy question is now whether to run the rung at all. Nine order-balanced
  raw-stage passes per arm price current main at 6104.0 ms versus 5102.7 ms on
  `feeea32a` (+1001.3 ms / +19.6%), while three WGPU FFT traces show exact
  starting 2547–3159 ms before coarse completes despite a cheaper FLOOR worker
  total. "Expensive operation" is therefore not the predicate
  ([ADR 0059](decisions/0059-coarse-rung-and-shared-reduced-stage.md)).
  Exit gate: an empirical pipeline/backend/display signature admits coarse
  only when prior evidence shows complete coarse ACK coverage before exact and
  whole-fill within the ±10% bar; unknown signatures skip. Validate both
  backends and do not restore the shared scheduler or multiply bounded
  presentation cohorts.
- **PyQtGraph full complex montage presentation is broken — OPEN.** Short
  prefixes are sufficient; do not hide it behind a long watchdog. Current main
  reached 251/272 operation tiles in 102.8 s while 118 commits spent 84.87 s
  repeatedly CPU-mapping the growing presented set. `feeea32a` reached 163/272
  exact tiles in 27.8 s with 21.88 s in 39 commits, so the pre-ADR exact-only
  path was already broken and ADR 0059's duplicate preview stream made it much
  worse. Reduced-rung admission now hard-refuses CPU-composited RGB, but a
  same-tip exact-only prefix was still far outside the few-second product bar.
  Fix the whole-set republication/mapping owner; gate on bounded commit and ACK
  counters plus a full 272-tile completion within 5 s, never a widened timeout.
- **A ladder rung exception retries without bound — OPEN, separate from ADR
  0059 ordering.** The PyQtGraph RGB-format failure admitted 18,314 preview
  tasks and raised 17,538 times because `_on_rung_error` replanned the same
  target. Make a rung failure terminal for that target with a named outcome,
  analogous to the commit-path `raised` outcome below; do not fold it into a
  coarse-ordering change.
- **An exception on the presentation-commit path is laundered into an
  anonymous stall — DONE 2026-07-26.** Found while root-causing the 272-tile
  FFT montage stall; a `RuntimeError` from page-pool exhaustion and an
  `AttributeError` from a typo'd probe produced **byte-identical** stall
  signatures, so the dump could not tell a lost wakeup from a throw. Resolved
  as a contract amendment to ADR 0051 rather than new machinery
  ([dossier](redesign/wgpu-pool-layer-leak-2026-07-26.md) §5a): **a commit
  that raises is a terminal bail with outcome `raised` and no armed wakeup.**
  The commit path was simply the one place using neither of the two
  vocabularies the repo already had — `_note_commit_bail(outcome, wakeup=…)`
  (nine call sites) and `handle_ui_exception` (fifteen). It now uses both, and
  the profiler reports `COMMIT RAISED` with the exception, its traceback, and
  the session and tiles it was committing, instead of a STALL GUARD.
  **The answer came out narrower than the brief in two places, both argued in
  §5a:** the gate is *not* re-armed after a throw (owner/armed are cleared
  before the commit and the drain flag resets in a `finally`, so a throw
  corrupts no state — re-arming would only replay a delta about to throw
  again), and no poison mark is needed (nothing retries, so the terminal
  record *is* the mark). Evidence:
  `tests/window/test_commit_failure_semantics.py` — five fault-injection
  tests including a live montage window whose commit throws from inside
  `_present_tile_delta`; all five fail on the unfixed tree. Declined as
  separate decisions: partial commits, dropping the offending tile, and any
  transient-vs-permanent (e.g. `SURFACE_LOST`) retry policy.
- **wgpu shader legibility — Stage A (grid / trust signals) — offscreen green,
  ring-4 OWED** (branch `claude/wgpu-shader-stage-a`). Four fragment-shader
- **wgpu shader legibility — Stage A (grid / trust signals) — offscreen green,
  ring-4 OWED** (branch `claude/wgpu-shader-stage-a`). Four fragment-shader
  visuals in `_RENDER_WGSL` + its BC-pool twin (and the CPU mirror in
  `arrayscope/display/shader_mapping.py`): **A1** zoom-gated per-texel pixel
  grid (`fwidth`-based, O(1), no new instances/bandwidth; flag `pixel_grid`);
  **A2** NaN/Inf → fixed 45° black/white hatch (reads on every colormap);
  **A3** missing page → dim -45° hatch, so "not loaded" ≠ "actual zero";
  **A4** clip markers (flag `clip_indicator`). Both flags default **off** →
  default render byte-identical (33 executor oracles + ImageView2D display
  tests pass unchanged, no rebaseline). Grid ships default-off rather than
  default-on-gated because ImageView2D renders small images at ~20 px/texel
  inside the gate band — default-on there would have forced a display-oracle
  rebaseline (forbidden). Flags live in the two spare `Mapping` uniform words
  (no uniform growth; `command_protocol.py` stays backend-neutral, ADR 0057).
  Evidence so far: `tests/gpu/test_wgpu_command_protocol.py` (37 passed;
  4 new paired fault-injection oracles proven red under a 4-way shader
  mutation, 33 baseline green), `tests/render tests/presentation` (131, `-n 0`),
  `tests/gpu tests/display` green. VisPy untouched. **Exit gate (ring 4, real
  Wayland — Thomas drives):** on real hardware, confirm the grid fades in only
  when zoomed past ~12–24 px/texel and never at normal zoom; NaN/Inf and
  missing-page hatches show on injected bad/absent data and are visually
  distinct; A4 markers appear only with `clip_indicator` on. The wgpu
  framebuffer-oracle gap is closed (`arrayscope/tools/framebuffer_reference.py`
  gained `wgpu_frame_matches_cpu_reference`, 2026-07-24) — Stage B–D can grow
  on pixels.
- **Progressive-load publication correctness — DONE 2026-07-22** (core
  `a50247e0`, ring-4 residual `648d00fb`). Full evidence in the
  [Done ledger](queue-done.md). Two adjacent gaps surfaced while writing the
  ring-4 gate remain open (below).
- **`ProgressiveArraySource.write_flat`/`write_bytes` silently no-op on a
  non-contiguous backing array** (found 2026-07-22). `array.ravel(order="K")`
  returns a *copy*, not a view, so the flat writes land nowhere. Latent only —
  every current loader builds the source over a contiguous `np.empty`
  destination and the `.rec` loader writes via in-place `write_transaction`
  indexing — but it is a footgun waiting for the next streaming format. Repro:
  `ProgressiveArraySource(np.zeros((4,4))[:, ::2]).write_flat(0, np.arange(8))`
  then `read_region` returns all zeros. Fix: write through a real flat view or
  reject a non-contiguous destination loudly. Exit gate: a deterministic test
  in `tests/io/test_progressive.py` that a non-contiguous write is either
  visible or refused, never a silent no-op.
- **Closing the streaming viewer mid-load does not cancel the reader thread —
  DONE 2026-07-22** (`2eaf8c2a`). `FileOpenSession` now installs itself as an
  event filter on the viewer window (ArrayScopeWindow emits no close signal and
  has no `WA_DeleteOnClose`, so `destroyed` never fires on user close); a
  `QEvent.Close` sets `_window_closed` + `cancel()`, and every terminal/progress
  handler (`_on_progress`/`_on_finished`/`_on_cancelled`/`_on_failed`/
  `_refresh_viewer_data`) is guarded against touching the closed window. Evidence:
  red-first `tests/app/test_open_flow.py::test_closing_viewer_mid_load_cancels_reader`
  + `::test_finished_handler_ignores_closed_viewer`; full parallel suite 2754 passed.
- **Demand-freshness unit-gate fixture** (live path FIXED 2026-07-19 `6fd0c262`,
  [dossier](redesign/demand-freshness-cold-fill-2026-07-19.md); full history in the
  [Done ledger](queue-done.md)): the unit gate's fixture carries no committed display
  frame, so the deferred camera obligation never replays — red pin stays strict xfail
  with instrumented probes: `tests/ui/test_lod_demand_freshness.py`.
- **PyQtGraph cold-fill tail stall under screenshot-flag load (offscreen
  only).** With the matrix driver's `--screenshot-interval-s 0.1
  --timeout-s 5`, the offscreen pyqtgraph cold driver intermittently
  freezes at the refine tail (all 272 presented, `level_stale=111`,
  planned-but-unsubmitted level-2 steps, armed presentation gate) — 1-of-2
  on unfixed main `b7e94879`, so pre-existing; the known tile-limbo/levels
  family. Real-Wayland rows complete; gate effect is diagnostic-only.
- **Kernel whole-process exit — VERIFIED BOUNDED 2026-07-22; premise was stale.**
  The 2026-07-19 concern was "non-daemon worker evaluations keep the process
  alive," but the app's `ThreadWorkerBackend` workers are `daemon=True`
  (`workers.py:60`) and `Kernel.shutdown()` is already bounded (returns after its
  timeout with an `alive_threads` warning — confirmed: a busy 8 s non-cancellable
  task yields `shutdown(timeout=1.5)` returning at 1.51 s). A real workflow
  process terminates on its own: `profile_montage_workflow --backend wgpu`
  exited with its own code (perf-bar exit 1, NOT a 124/137 timeout-kill) after
  78.6 s, with no leaked-thread diagnostics. The "process won't exit" symptom
  people saw was the **pyqtgraph level-relevel stall** keeping the profiler's
  main-thread wait loop spinning (a separate item — see
  [`redesign/pyqtgraph-level-convergence-2026-07-22.md`](redesign/pyqtgraph-level-convergence-2026-07-22.md)),
  not a leaked worker thread. No cooperative-cancellation change is needed; the
  eval loops already poll `_check_cancelled` and `shutdown()` cancels every
  in-flight token (`scheduler.py:436`). Reopen only if a real non-daemon leak is
  observed on a completing workflow.
- **Remove the `montage_key_batch_fallbacks` runtime guard** once the
  consolidated key owner is proven in the field. 2026-07-17: derivation is
  consolidated — every layout has one owner
  (`_display_tile_key_from_parts`/`_request_key_from_parts`/
  `_view_state_key_with_slices` in `evaluator.py`; the batch's slow path *is*
  `display_tile_key`) and parity + fallback are pinned in
  `tests/operations/test_cache.py`. The runtime guard and counter stay until
  a release cycle shows the counter at zero.
- **R8 continuity gate vs document-changing stages (adjudication needed).**
  With the fill stall and entry blackout fixed, `profile_montage_workflow`'s
  `fft_full_tiled_montage` fails `presentation_continuity` on BOTH vispy
  (first tile 4.6 s) and pyqtgraph (3.4 s), offscreen 2026-07-19: applying
  the FFT pipeline is a document change, ADR 0051 forbids retaining
  old-operation pixels, so entry honestly blanks — and the gate's
  no-blank-sample rule can never pass a document-changing stage slower than
  the sampler's first tick. Either the gate learns a document-change
  transition class (blank legal, successor latency still measured), or the
  FFT successor needs its own first-pixel latency work. Pre-existing on all
  backends; raw-stage entry (same document) now passes via the montage-axis
  bridge.
- **Audit `_resident_source_matches_expected(source, None) → True`**
  (controller-side expected-source coverage during session switches).
  2026-07-22: reviewed — the `None` branch falls through to
  `acknowledged_identity_satisfies_target`, not an unguarded accept, and lives in
  the retiring VisPy backend; no clear defect found, left as a low-priority audit.
- **`tests/ui/test_diagnostics_dialog.py` serial `-n 0` failures — DONE
  2026-07-22** (`0ee00f33`). Root cause was NOT a diagnostics singleton but the
  suite reading the developer's real `~/.config` QSettings under `-n 0` (empty
  pytest-qt org name → org-fallback merge that `.clear()` can't reach →
  `image_rendering_backend=wgpu`). Fixed hermetically in `tests/conftest.py`.
  See the [Done ledger](queue-done.md).
- **Upstream rendercanvas contributions** (from gate B): a native-Wayland
  screen-presentation hook (wl_display via QNativeInterface + winId-as-
  wl_surface, Vulkan-only instance) and making the import-time
  `QT_QPA_PLATFORM=xcb` override opt-out. Until merged upstream, ArrayScope's
  `qt_platform` policy owns the platform decision.
- **Screen-mode follow-ups** (screen LANDED 2026-07-19 behind
  `wgpu_present_method`; Mailbox acquire and the GPU-overlay layer are
  DONE; the 2026-07-20 dogfood glitches — subsurface soup from native-child
  sibling/ancestor promotion, hidden overlay chips, resize flicker — are
  FIXED via the `createWindowContainer` recipe, overlay native promotion,
  and resize-edge immediate present; evidence: nested-weston compositor
  captures + WAYLAND_DEBUG subsurface counts): measure the screen-vs-bitmap
  delta at real 4K — bitmap's measured boundary is ~26 ms readback there,
  the decisive screen case — and decide whether screen becomes the wgpu
  default on capable Wayland sessions.
- **Renderer measurements not yet taken:** NVIDIA/discrete adapter cells for
  Tier 1/4 (PRIME copy changes upload and present arithmetic), real 4K
  swapchain. (The `winId == wl_surface*` per-Qt-minor pin is DONE:
  `tests/gpu_interaction/test_wgpu_native_wayland_pin.py`, ring 4.)

## Done

The completed-step ledger lives in [`queue-done.md`](queue-done.md) (one line per
step, evidence linked, most recent first). When a step lands, move its row there.
