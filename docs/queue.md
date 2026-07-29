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
| 2 | **Close page-backed crop-rebind R3.** A resident rebound wrapper has only predecessor-window `level_stats`/`level_data`; WGPU binds the new `source_anchor` over current resident pages, so the current pixels have no synchronous value-bound proof and predecessor levels can clip before `rearm_crop_rebind_level_evidence` lands. Keep the probe's `page-backed-rebind-no-current-plane` red until the presentation either (a) widens atomically to a proven full-plane/current-page superset, or (b) withholds the new binding until exhaustive current-window evidence is ready. Sparse semantic samples and predecessor page backing are not proof; do not weaken `presented_tile_value_bounds_recorded`. | Managed-Weston `display_x_axis_slice,display_y_axis_slice` has zero `page-backed-rebind-no-current-plane` and zero `commit_levels_contain_presented_tile` rows; a fixture with an extremum outside the semantic sampler proves no intermediate commit can clip; zero new display-preparation producers and the existing rebind/evaluation settled-level parity remain green. |
| 3 | **WGPU renderer — field evidence** (ADR 0057; slices (a)–(c) LANDED: native GPU overlays + glyph text, screen presentation behind `wgpu_present_method`, G6 compute, dogfood-crash/eviction-shield and codex-review fixes — full status narrative in the [Done ledger](queue-done.md) and dossiers). WGPU led the historical fast-scroll comparison (p95 77.3 ms vs VisPy ~106–124), matched zoom/pan steady-state, and reached 5/5 in its final matrix. **2026-07-20 correctness blocker — CLEARED 2026-07-21:** compositor screenshots showed native screen-mode overlay boxes not matching the Qt/PyQtGraph presentation, with the histogram/top-right composition clipped. Root cause was promoting the overlay chips to native *child* windows, which Qt never grants an ARGB visual (`alphaBufferSize() == 0`), so translucent rounded chips flattened into opaque boxes and occluded the histogram; restacking the swapchain below was unavailable too (`QWindow.lower()` emits no `place_below`). Chips are now rasterized from Qt's own painter and composited inside the frame (`widget_quad` + `UpdateWidgetAtlas`). Tool-managed Weston full-window evidence vs the bitmap reference went 3.45% -> 0.08% differing, the residue being only a Weston-drawn cursor, one transient live readout, and the bitmap-vs-bitmap noise floor. Tool-managed Weston or manual full-window compositor evidence stays required. **2026-07-21 successor review:** physical scroll evidence retains 60/60 tiles without stale/black holes, but WGPU shuffle first-new pixels remain red at 2.55 s (2 s bar). The final successor needed only two missing preview tasks (23 ms finish span) and reused the prepared atomic transaction about 129 ms later, so the remaining delay is in retarget/input trajectory, not a 60-tile cold-data or last-payload barrier. Built-in 0.1 s full-window capture is correctness evidence only: its synchronous compositor/readback load materially perturbs timing. One managed-Weston WGPU scroll run aborted in `wgpuSurfaceConfigure` after `ERROR_SURFACE_LOST_KHR`; an immediate identical rerun passed 60/60, so keep this as reliability evidence if it recurs rather than hiding it as a harness timeout. The VisPy comparison is retained as historical diagnostic evidence; current work targets WGPU or shared code. PyQtGraph's pre-existing 7/10 three-second LOD misses remain a separate standing red. **2026-07-24 cropped-axis follow-up:** the maintained-backend profile now scrolls every dimension at fast and slow cadence with physical CPU-reference checkpoints; stale session anchors, mutable preview anchoring, and predecessor-owned presentation events are fixed. A full 272-plane offscreen Vulkan run completed 272/272 after visible admission counted 1088 submitted native pages instead of the 272 preview pages; one-shot frame-plan capacity cut growth copies from 2.88 GB to 6 MB. **2026-07-25:** with the fixture's viewport gate fixed those two crop stages run again, and their newly visible `display_axis_all_dimension_pixels_match_cpu_reference` red was a reference-model gap, not a crop defect — the mismatching pixels were the fixture's ROI strokes and a composited Qt chip, each verified to sit exactly where its own geometry projects after the crop. The image comparison now withholds overlay-covered pixels (a stroke drawn anywhere else still fails it) and a new `display_axis_roi_overlays_track_geometry` gate projects the semantic ROI store to require each outline be painted in its own band and nowhere else; 12/12 checkpoints green on both stages, WGPU and PyQtGraph ([dossier](redesign/montage-cold-fill-cohort-2026-07-25.md)). Real-Wayland screenshot/journey acceptance is still required for field evidence. VisPy is retired by [ADR 0061](decisions/0061-retire-vispy-rendering-backend.md); maintained comparisons are WGPU and PyQtGraph. **Successor traps:** import rendercanvas ONLY via `import_qrenderwidget()`; every WGPU adapter probe pins `set_instance_extras(backends=["Vulkan"])` BEFORE its first request (`9c115686`). | Field gate: maintained-backend journey matrix + perf bars on real data; written verdict in tensor-engine-endpoint.md |
| 4 | **Large-data retention truth before compression.** Instrument a representative lazy 4+ GiB session so source, evaluator display payloads, exact ROI-demand regions, `StageCache`, LOD/retained payloads, and the selected backend's physical storage report owned bytes, alias-adjusted bytes, evictions, costly recomputes prevented/repeated, and RSS/device deltas. Do not merge the evaluator caches: they have different consumers. Do not add PyQtGraph raster and GPU page residency: they are alternative backend mechanics. Use the new `memory_retention_audit` as the static envelope, then decide whether the ROI cache needs its own smaller policy budget and which retained values deserve to exist. **2026-07-26 WGPU input:** the zoomed-out raw montage retained 14.9 MB/604 CPU pyramid entries and 285.2 MB of active level-0 GPU pages; its reduced CPU pages supplied histogram evidence and lifecycle tokens but were never uploaded, and a later physical zoom draw uploaded zero pages with or without the desired CPU rung. Price CPU evidence retention separately from backend display retention; the claim does not yet cover cropped scroll, zoomed-in single image, or PyQtGraph ([dossier](redesign/wgpu-refinement-consumer-price-2026-07-26.md)). **Compression remains parked:** demand-sized pools fixed the full mirror, but AUTO is still slower in physical-tile evidence. Next establish one physical byte cap and measured eviction/re-upload benefit ([details](reviews/2026-07-22-compression-live-benefit-review.md)). | Fresh-process lazy/eager runs on WGPU and PyQtGraph reconcile owner counters with RSS/device allocation; a written keep/evict/resize decision for every owner; no unbounded store; compression remains parked unless the trace identifies its exact bottleneck and owner. |
| 11 | **Operations as a platform — DONE 2026-07-26** ([proposal](proposals/operations-as-a-platform.md), [ADR 0060](decisions/0060-operation-definitions-runtimes-and-discovered-shapes.md)). **A:** the dtype-honest native toolbox replaced NumPy-trivial SigPy/BART wrappers; the [Numba review](reviews/2026-07-26-native-operations-numba.md) landed only the exact normalize win. **B:** every registration tier exports one definition; Duplicate and New/source/callable editing live in one manager with full parameter parity. **C:** command and named-environment runtimes share explicit tokenization, array handoff, cancellation, timeout, availability, and imported-recipe quarantine. **D:** bounded source-aware characterization discovers shape/dtype/windowability, fits only conservative rules, and demotes misfits to OPAQUE whole-array cache stages ([cost review](reviews/2026-07-26-operation-characterization-cost.md)). **E:** definitions/forms/recipes carry dimension-set, Compare-document, one-ROI mask/coordinates, and saved-array input slots with source-aware characterization and fail-closed restoration; `bart:pics` runs from the UI with k-space plus sensitivity maps through the fake-BART argv/cancellation/timeout oracle. **Phase 6:** the command palette is the one search owner and now consumes the same library-backed listing as every add surface, with keyboard result navigation and a visible no-match state. The add popup links to search and browses one taxonomy category at a time instead of flattening 31 non-Common rows. The manager-first [authoring walkthrough](write-your-own-operation.md) covers New/Duplicate through recipes while [plugin operations](plugin-operations.md) remains the single schema reference. The overlapping program tests were consolidated from 68 to 63 across the four audited files without removing hazard coverage; two focused discoverability tests were added outside that set. Dark/light gallery PNGs were inspected for popup browse/search states and the command palette. **Real BART numeric follow-up:** the [one-command harness and dated evidence](reviews/2026-07-26-bart-numeric-validation.md) validate ECALIB's coil subspace, Walsh's actual packed covariance, and scale-preserving PICS against independent NumPy/invariant oracles; its first run found and fixed missing PICS `-S`, the false Walsh sensitivity-map description, and the system-pack named-environment bypass. Dark/light gallery review covers the slot popup, manager editor, and unresolved state. Phase 6 retains only the documented walkthrough and add-popup search/fold polish. | The all-ops smoke harness includes a slot-bearing user op; BART PICS lands through the UI with both inputs; evaluation counters prove ROI invalidation in both directions; recipe and availability paths fail closed; authoring and repair are documented; every introduced surface has inspected dark/light gallery evidence. |

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
bottleneck. The landing session reported installing `sigpy`, `blosc2`/`zfpy`,
and BART, but retained output did not make the broad real-BART claim
reproducible and it did not cover today's three BART definitions. Their current
numeric evidence is the
[2026-07-26 harness review](reviews/2026-07-26-bart-numeric-validation.md).
Deferred at that earlier boundary, with reasons recorded in the ledger: sigpy
`nufft`/`espirit`/`fwt`/`iwt` and `bart:pics` (multi-input /
structural-metadata args did not yet fit the unary `fn(ndarray)->ndarray`
contract).

## Performance bars (commitments, not history — restored from R2/R4/R8D)

- GUI callbacks < **50 ms** always; event-loop heartbeat gap ≤ **16 ms**
  during fills and scrub; warm scrub input ≤ **15 ms**; settled-idle CPU 0%.
- **#1 throughput target:** fast montage FFT index scroll ~4 fps → toward
  the ~17 fps scalar rate (2026-07-09 measurement, realistic human scroll).
- Benchmark deltas stay within ±10% of the frozen baseline unless a step
  improves them. PyQtGraph gets 2× the WGPU allowance (it targets
  headless/remote use); both backends stay first-class for correctness.

## Standing lane — test hardening & debt (parallel-safe, any order)

Safe to pick up alongside the numbered queue; each is self-contained.

- **ADR 0059 exclusive dynamic preview-first — RAW + WGPU FFT MEDIAN GATES DONE
  2026-07-27; PyQtGraph complex preview correctness pass DONE, latency RED
  (the 272-tile post-FFT physical-union row is narrowly xfailed only for a
  timeout at the unchanged 5 s interaction cap; measured 5.38 s serial but
  straddles the cap offscreen, so the deterministic R5 harness remains the
  latency authority);
  product default.**
  Root cause was the successor rule itself: a FLOOR-backed `DESIRED` task still
  used `DISPLAY_PREVIEW`, so phase-only experiments delayed ACK while target
  evaluation consumed coverage workers. Target work now stays on
  `DISPLAY_PREPARATION`, coverage closes on backend-acknowledged first-pass
  identities, and the profile hard-fails both ACK order and worker execution
  order per scheduling generation. Preview quality is always at least two LOD
  levels coarser than the target and follows the smooth zoom so one preview
  texel spans 3–6 screen pixels. Tile count never decides image quality. The
  rejected size surcharge made each 336-square tile one L9 sample and produced
  homogeneous grey boxes; the corrected field scale chooses L5 (11×11,
  4.22 screen pixels/sample) and preserves recognizable anatomy.
  **R5 bulk ownership corrected 2026-07-27:** preview production now uses the
  ordinary governed worker continuation instead of one whole-round task, and
  both preview and target presentation consume
  `ResourceGovernor.decide_render_pass()` caps and deadlines unchanged.
  PyQtGraph prefixes stage as direct items while the compact atlas remains a
  hidden complete transaction; neither backend may enter an atlas build unless
  the governor admitted the entire planned set. Ready payloads suppress
  duplicate FLOOR evaluations. The
  optional JSONL oracle is buffered and its rich identity fields are bounded,
  so evidence collection no longer adds hundreds of synchronous writes before
  T1; the crash-oriented trace ring is unchanged. **Round-level ownership
  recovered 2026-07-27:** preview admission now reserves the one round source,
  the complete cohort installs one 272/272 decision, and the legacy semantic
  source-slab/FFT sweep runs only as the no-preview fallback after coverage
  closes. Fresh real-Wayland raw traces on both PyQtGraph and WGPU contain one
  `source=preview-cohort` evidence event, identical first-visible/final levels,
  and zero stale-level tiles; four-pair order-balanced T1 ratios were
  load-neutral (PyQtGraph +0.6%, WGPU +0.4%). **Native bounds no longer make
  the admitted FFT display transform native (2026-07-27):** the shared cohort
  box-means the display axes before the operation pipeline, while one complete
  scan of the native inputs derives the orthonormal FFT's conservative L1
  envelope; no native FFT or sparse evidence sample runs. The admitted
  FFT/shift/IFFT profile chain uses its exact phase-modulation magnitude
  envelope. The ladder's commuting tile-local predicate, exact operation
  shape/axis, and linear shader scale are executable preconditions, so
  display-axis FFT cannot enter this route and an unproved chain keeps the
  transform-once-native route. On the warm 336×336×60 L5 FFT cohort, 18
  fully order-balanced interleaved Weston passes measured **42.90 ms median**,
  versus the recorded 22.64 ms before native evidence and 108.14 ms for the
  full-native transform: 60.3% below the full-native route, recovering 76.3%
  of the correctness-fix cost without a second transform. The same process
  measured 21.81 and 131.04 ms for those reference revisions. On the bundled
  336×336×272 NIfTI, the envelope and true FFT magnitude maximum were both
  340233.619; its symmetric real-component window was 1.495× the exact span.
  On the first 60 planes they were both 98878.992 and 1.577×.
  **Per-rule state lives in one place: the Status section of
  [the progressive render contract](architecture/progressive-render-contract.md).**
  Update it there, not here — R1–R4 are now enforced, R5 carries measured
  residual debt, and R6/R7 are unimplemented. The historical readings below are
  kept because they name the measurements, not because they are current.
  The independent progressive contract replay was **RED** on all five
  2026-07-27 field traces: R3
  exposes eight frozen/inactive evidence runs across both backends, and R1
  fires 15 (WGPU FFT) and 11 (WGPU scalar) snapshots of production below the
  round's target floor — level 0 uploads of +176, +128 and +256 in rounds that
  demanded level 1 or 2. R1 reads `n/a` on the three PyQtGraph traces: they
  carry no per-level upload counters, so production cannot be told from a page
  cache arrival and the rule is unchecked rather than satisfied.
  **The latency gates are now reclassified (2026-07-27).** `_r8_certification`
  keeps only correctness gates and the profiler's exit status requires an
  in-process R1–R5/R7 verdict; medians, heartbeat gaps and coarse/target
  ordering stay in the record as diagnostics and no longer decide success, and
  the PyQtGraph-complex R4 exemption is gone. Real-NIfTI headless Weston:
  PyQtGraph raw red R1/R5 at 272/272; PyQtGraph FFT red R1/R5, 272/272
  presented but unsettled at 5 s, **R4 now green** after the complex-preview
  atlas landed; WGPU raw red R1/R5 at 272/272 with off-floor physical uploads;
  WGPU FFT red R1/R5 at 113/272 with off-floor uploads and duplicate per-pass
  production. **R2b round identity landed 2026-07-27:** the structural target
  key covers semantic document/operation state, exact camera/viewport, montage
  plan, and display axes; both floors latch to it and every `pipeline_plan`
  carries the id plus `(P, T)`. The in-process profiler now certifies one floor
  pair per derived round (the focused live-Weston synthetic matrix reported
  R2b=0 on all five applicable phases), and only R6 remains **unverifiable** because one
  settled phase cannot exercise sustained-input shedding. The five earlier
  field snapshot traces predate the new identity fields and must be recaptured
  before the stricter standalone replay can attribute their R1 deltas; their
  recorded R1/R3 reds remain standing, not silently converted to green.
  Final-tip low-load AC trace-ACK T1/T2/B medians: WGPU raw
  **972/3874/2130 ms (6/6/3 passes)**, WGPU FFT
  **948/6118/4795 ms (6/5 observed/3 independent processes)**, and PyQtGraph
  raw **853/4374/3358 ms (3/3/3 passes)**. Every applicable A pass passed both
  exclusivity clauses. One WGPU FFT A tail reached only 270/272 target ACKs,
  and every WGPU FFT action exceeded the five-second settlement limit; those
  are reported as target convergence debt, not hidden by the green T1 median.
  The current-main red control reports worker/ACK overlap on WGPU raw and FFT,
  ACK overlap on PyQtGraph raw, and no PyQtGraph complex preview pass.
  Field PyQtGraph 272-tile transitions additionally pinned and fixed two
  no-progress cycles: mixed exact/rough level evidence now publishes from its
  per-source covered set, and a compact preview now accepts a physically
  current retained exact slice as the complement of its 271 reduced tiles.
  PyQtGraph complex preview now retains compact reduced `complex64` pages and
  CPU-composites their derived RGBA8888 atlas through the round levels. A
  272-tile real-Wayland run acknowledged every level-5 preview identity before
  any level-2 target identity and reached physical preview coverage in
  4175 ms. The former 2985 ms atomic atlas build is removed. Final-code Weston
  chunk and wall-time evidence is in the
  [R5 dossier](redesign/r5-bulk-render-governor-2026-07-27.md). The governor
  now fits fixed, item, and byte cost, and minimizes total fill plus continuous
  callback-latency and exponential extrapolation prices;
  50 ms remains a fixed reported requirement, never an adaptive target or a
  hard cohort boundary. Fixed/byte knowledge is shared by backend + commit
  path + montage geometry across passes and reloads; item cost remains
  pass/representation-local and representation changes warm-seed it at high
  uncertainty. Extrapolation is optimistic while sample count/span/residual
  make the fit uncertain, then tightens to the full curve. The former
  quadratic cohort term was removed after the real-backend probe produced no
  repeatable WGPU curvature and only contradictory isolated PyQtGraph fits.
  One Render Responsiveness preset scales the latency price only
  (Responsive/Balanced/Throughput = 2.0/1.0/0.3); remote/software sessions seed
  Throughput unless the user has chosen otherwise. Equal-valued callbacks are
  distinct evidence, robust Cauchy fitting bounds a single high-leverage
  stall, and a separate three-observation load offset follows sustained host
  load and recovery without rewriting item cost. The matched interleaved
  four-repeat Weston run gives WGPU 4.221 s median settlement versus 4.571 s
  at base (7.7% faster, 4/4 versus 3/4 settled). PyQtGraph's valid median is
  4.538 s versus 4.843 s at base, with the same 2/4 censoring. An experimental
  hard-reset arm once reached 3.94 s but also caused 128–185 ms preview jumps
  and a 4.83 s rerun; that result remains a follow-up target, not the landed
  baseline. The first widened WGPU target commit still spends 117–144 ms in
  backend apply while later similar cohorts cost 25–32 ms.
  WGPU now attributes one-time executor initialization and pool growth
  separately: those times stay in full R5 evidence but do not train the steady
  fixed/count/byte model. The reusable Weston
  `render_pass_governor_probe` reports wall-clock completion throughput,
  callback distributions, and component attribution without JSONL artifacts.
  Residual cold/warm outliers fail loudly and remain callback-bar debt; no
  measured preview or target pass completed atomically. **Retained-retarget
  stall fixed 2026-07-28:** the first governed presentation slice is armed
  directly from the retarget edge, so a fully resident successor no longer
  waits for a producer completion that will never occur. WGPU's logically and
  physically retained mappings publish as one visibility transaction while
  cold payloads keep the governor limits; the strict 60-tile zoom-retarget
  oracle passes in three fresh Weston processes and the measured atomic mapping
  cost is 45.2 ms. A post-fix four-repeat WGPU run settled 4/4 with
  1.308 s preview and 4.302 s settlement medians (5.9% faster than the matched
  4.571 s base, 1.9% slower than the prior branch median). PyQtGraph's four
  post-fix runs all censored, so its settlement gate remains red
  ([R5 dossier](redesign/r5-bulk-render-governor-2026-07-27.md)). **Named
  commit failures restored 2026-07-28:** a backend throw now marks its exact
  session generation terminal, so late worker completions cannot retry the
  transaction and overwrite `commit_outcome="raised"`; successor generations
  remain unpoisoned. The six-test failure-semantics module and the paired live
  Weston guards pass. **Resident-but-unpresented floor strand fixed
  2026-07-28:** R2 residency no longer impersonates lifecycle-presented truth.
  An unpresented tile with a suitable resident floor gets a presentation-only
  FLOOR step: the finest suitable resident payload is wrapped and committed
  with no new numeric task, so better-than-demanded pixels appear immediately
  and later rounds skip them after ACK. On the full 336×336×272 NIfTI, the
  272-tile WGPU zoom/pan return changes from a permanent 1/272 presentation
  (271 target-ready, kernel and commit queues idle) to 272/272; final zoom-out
  takes 1.208 s and settles in 0.912 s with zero coarse-target producer starts
  ([R5 dossier](redesign/r5-bulk-render-governor-2026-07-27.md)). **Orphan
  ready-payload strand fixed 2026-07-28:** lifecycle-ready is no longer
  assumed to mean commit-pending. A ready, current payload with no dirty/upsert
  owner gets the same presentation-only handoff; a stale or unrecoverable
  payload falls through to normal production. The field capture moves from
  93/100 presented with seven target-ready orphans to full physical
  presentation in the reproduced interaction. A separate final-refinement
  wedge reproduced identically before this ready-payload change at
  `9ef3d373`: the 272-tile profiler left 6–7 already-visible preview payloads
  dirty with no gate armed. **Fixed 2026-07-28:** concrete dirty/upsert debt
  now keeps its class precedence at both wrapper-build and admission
  boundaries; recurring already-current lifecycle notifications can no longer
  consume the governed cohort with no delta. The real workflow drains and
  completes 272/272, but performance remains red: the zoom/pan scalar phase
  takes 24.326 s with a 513 ms maximum event-loop interval and still fails
  R1/R2/R2b/R5. This closes a permanent wedge, not the governor tuning row.
  **WGPU callback spikes re-attributed 2026-07-29, nothing optimized:** an
  order-balanced managed-Weston A-B-C/C-B-A run compared `fbaf9074`, landed
  `160ede2c`, and the viewport-currency/stall-reporting fix at 336×336×272.
  Same-process warm fills completed 1/2, 2/2, and 2/2 respectively; callback
  min/p50/p95/max was 9.8/42.5/63.3/195.0 ms, 10.2/43.8/62.0/99.5 ms, and
  8.7/41.3/89.5/134.1 ms, with 11/46, 11/42, and 10/48 callbacks above
  50 ms. Fresh-process cold fills remained censored at the five-second gate
  in five of six runs. In every revision the target spikes are
  backend-apply-owned (worst target callbacks: 228.9 ms with 209.2 ms apply;
  212.8/193.4; 243.1/217.0). The fixed revision's worst apply splits between
  130.4 ms texture preparation and 81.6 ms native submission. No new
  post-`fbaf9074` whole-montage Python loop explains more than 25 ms:
  warm `payload_build_ms` maxima were 25.8/24.1/27.1 ms, with the 27.1 ms
  observation a metadata commit and the same inherited owner already present
  at the comparison base. Keep the next optimization on WGPU backend
  preparation/submission and cold native initialization, with deterministic
  physical equivalence; do not retune the scheduler from these noisy tails.
  Preview-first is the explicit default; `--disable-coarse-rung` is the B arm
  ([ADR 0059](decisions/0059-coarse-rung-and-shared-reduced-stage.md)).
  **Non-reducible pipelines keep the pass (2026-07-27):** FLOOR no longer
  depends on reduced-input admission. It evaluates once natively, reduces the
  output for preview presentation, and retains the exact result so target
  refinement performs zero additional evaluations. On a real-Wayland
  displayed-axis `FFTShift` over the 336×336×272 NIfTI, R4 changed red→green
  on both maintained backends: WGPU reached 272/272 preview ACKs in 2452 ms
  and PyQtGraph in 3178 ms, versus no preview ACKs on the parent. Target
  settlement remained beyond 5 s on both and the standing R1/R2 failures did
  not increase ([evidence](redesign/unconditional-native-output-preview-2026-07-27.md)).
- **WGPU odd-origin resident-crop geometry is rejected — OPEN.** The
  managed-Weston 336×336×272 `display_x_axis_slice`/`display_y_axis_slice`
  matrix reproducibly reaches `tiledPayloadResident()` with a globally
  phase-aligned level-1 crop whose native extent is 100 but whose reduced
  extent is 51. `_wgpu_payload_lod_geometry()` derives 50 from the local
  extent alone and raises
  `wgpu payload texture geometry does not match its native LOD ladder`,
  stranding 50 current targets with no work in flight. Exact landed
  `160ede2c` and the viewport-currency fix fail identically. Preserve the
  strict geometry check, but make the expected reduced extent include the
  crop's native origin/phase (or prove that information at the page identity
  owner); pin odd-origin crops in both orientations and require the existing
  physical CPU-reference/continuity matrix to settle. Do not catch and
  silently fall back after the binding error.
- **PyQtGraph full complex montage presentation is broken — OPEN.** Short
  prefixes are sufficient; do not hide it behind a long watchdog. The
  target-only full FFT action failed to return before an 8 s process guard and
  emitted no phase row; an immediately preceding bounded run reached only
  62/272 exact ACKs by its five-second failure. The reduced complex preview is
  now present. The whole-active-set republication is fixed: the backend
  resolves only the governor's admitted upserts and starts at one tile.
  Completion remains red because three individual complex item updates in the
  final Weston run took 63–155 ms and only 55/272 were physically presented by
  five seconds. Split or move that per-item work; gate on bounded commit and ACK
  counters plus a full 272-tile completion within 5 s, never a widened timeout.
- **A ladder rung exception retries without bound — OPEN, separate from ADR
  0059 ordering.** The former PyQtGraph RGB-format trigger is removed by the
  reduced complex preview format, but the generic failure policy still replans
  the same target. Make a rung failure terminal for that target with a named
  outcome, analogous to the commit-path `raised` outcome below; do not fold it
  into a coarse-ordering change.
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
  `tests/gpu tests/display` green. The then-maintained VisPy path was outside
  that experiment and is now retired by ADR 0061. **Exit gate (ring 4, real
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
- **WGPU retarget-mapping stall on a channel change — two strict xfails, ADR 0061
  follow-up (2026-07-27).** Migrating the retired backend's coverage to WGPU
  exposed a stall with no owner (ground rule 11). Retargeting a `COMPLEX_RG32F`
  payload to display mode `'scalar'` makes `WgpuImageView2D.tiledPayloadResident`
  raise `NotImplementedError` out of `_wgpu_commit_plan`; `frame_session.
  _free_retarget_tiles` calls it as a **predicate**, so the throw aborts the whole
  commit, `handle_ui_exception` swallows it, and the session is left with every
  tile dirty, **zero** active requests and no event that can resume it (verified
  stuck for 30 s, not slow). Pre-existing WGPU defect — this branch only made it
  visible. **Guarding the predicate alone is measured NOT sufficient**, so the
  real owner is the retarget path that mints a payload whose texture
  representation and requested display mode disagree. Both cells pass on
  PyQtGraph. Pinned by `tests/ui/test_montage_interactions.py::
  test_semantic_montage_transition_never_leaves_old_tiles_visible`
  (`channel-real-wgpu`, `complex-mode-wgpu`), `strict=True` so the fix un-xfails
  itself. Watch `render/effects.py:696`'s `ComplexWarning` while fixing.
- **WGPU hidden-ROI overlay never fires on a tiled single image — one strict
  xfail, ADR 0061 follow-up (2026-07-27).** With the inspection dock hidden, the
  floating ROI stats panel never appears on WGPU for a tiled **single-image**
  frame. Measured to be the **trigger, not the computation**:
  `_hidden_roi_statistics` returns a row and `_committed_tiled_frame()` is
  non-None on WGPU, yet `setRoiInfoRows` is never called, so `_roi_info_panel`
  stays `None` after 240 event pumps *and* after invoking
  `_refresh_hidden_roi_overlay_from_committed_frame` by hand. The montage twin
  and the PyQtGraph twin both pass, which bounds the gap to the tiled
  single-image WGPU path. Pinned by `tests/ui/test_roi_inspection_interactions.py::
  test_wgpu_hidden_inspection_panel_uses_tiled_frame_payloads` (`strict=True`).
- **R8 continuity gate vs document-changing stages (adjudication needed).**
  With the fill stall and entry blackout fixed, `profile_montage_workflow`'s
  `fft_full_tiled_montage` historically failed `presentation_continuity` on
  VisPy (first tile 4.6 s) and PyQtGraph (3.4 s), offscreen 2026-07-19: applying
  the FFT pipeline is a document change, ADR 0051 forbids retaining
  old-operation pixels, so entry honestly blanks — and the gate's
  no-blank-sample rule can never pass a document-changing stage slower than
  the sampler's first tick. Either the gate learns a document-change
  transition class (blank legal, successor latency still measured), or the
  FFT successor needs its own first-pixel latency work. Pre-existing on all
  backends; raw-stage entry (same document) now passes via the montage-axis
  bridge. Re-adjudicate the gate on the maintained WGPU/PyQtGraph matrix.
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
