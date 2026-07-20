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
| 1 | **Performance-bars program on the engine** (parked — Thomas 2026-07-17: act only on true stalls/no-progress, never on merely-slow). The bars (below) are the product promise. One measured cause at a time, before/after real-Wayland harness evidence per commit; a step that regresses a bar is reverted and buried in the graveyard. | Bars trend green in `profile_montage_workflow` on real Wayland, both backends (PyQtGraph at 2× allowance) |
| 3 | **wgpu strangler — promotion evidence** (ADR 0057; slices (a)–(c) LANDED: native GPU overlays + glyph text, screen presentation behind `wgpu_present_method`, G6 compute, dogfood-crash/eviction-shield and codex-review fixes — full status narrative in the [Done ledger](queue-done.md) and dossiers). Open: **(d)** promotion by evidence — wgpu LEADS fast-scroll (p95 77.3 ms vs VisPy ~106–124), matches zoom/pan steady-state, 5/5 in the final matrix. Before the AUTO flip: shared row-1 callback bars, dogfood hours (screen mode selectable from Performance → wgpu Presentation), the FFT-scroll 4→17 fps headline on the new tip. VisPy retirement only via the roadmap ladder — never a flag-day switch. **Successor traps:** import rendercanvas ONLY via `import_qrenderwidget()`; every wgpu adapter probe pins `set_instance_extras(backends=["Vulkan"])` BEFORE its first request (`8c57a7bf`). | Promotion gate: journey matrix + perf bars on real data; written verdict in tensor-engine-endpoint.md |
| 4 | **G7 — compressed transport.** Codec ladder, measured topology; ZFP-class first. After G6. | Measured end-to-end win on real data |

## Next — the product turn (queued behind rows 3d/4; rationale in [roadmap.md](roadmap.md) and [reviews/2026-07-19-course-review.md](reviews/2026-07-19-course-review.md))

| # | Step | Exit gate |
|---|---|---|
| 5 | **Compare v1a — camera/viewport sync facet.** Add a `camera` facet to `sync/messages.py` + controller (existing revision/loop guards; leading-edge coalesce). | Two linked windows pan/zoom together on real Wayland; sync suite green; no new scheduler/timer (ground rule "one scheduler") |
| 6 | **Compare v1b — "Compare with…" launcher + linked complex cursor.** Second window pre-linked on dims+camera+levels; hover reports value of A and B (mag+phase for complex) at the linked cursor. | Journey-style two-window scenario green on real Wayland; cursor values exact vs NumPy oracle |
| 7 | **Compare v1c — difference as derived source.** `CompositeArraySource(A, B, op=subtract)` array-like flowing through the unchanged unary pipeline/tile engine; opened as a third linked window. Multi-input pipeline *ops* stay rejected. | A−B window renders progressively on real data; values exact vs NumPy oracle; recipes/sessions unaffected |
| 8 | **Plugin ops v1 — Tier-1 registry.** Entry-point group `arrayscope.operations`, namespaced stable ids, lazy import on first use; pure `fn(ndarray)->ndarray` + shape/dtype adapter (OPAQUE, whole-array, cache-stage-able). | Third-party pip package contributes a working reversible op incl. recipe round-trip; import-health guard stays green |
| 9 | **Plugin ops v2 — sigpy pack + Tier-2 conformance harness.** sigpy fft/nufft/espirit as an optional in-process pack; Tier-2 capability claims honored only after property-test `apply(whole)[region] == region-path`. | sigpy ops usable in the dock; harness rejects a deliberately mis-declared capability (red-first) |
| 10 | **Plugin ops v3 — BART subprocess pack.** cfl temp-file handoff at the stage-materialization seam; `CancellationToken` → SIGTERM; honest cost hints for admission. Requires the queue's shutdown/cancellation item closed first. | BART fft/pics run as ops on real data; cancel mid-op kills the child <1 s; process exit stays bounded |

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

- **Progressive-load publication correctness (2026-07-20 integration
  blocker).** The new asynchronous `.npy`/`.cfl`/Philips `.REC` flow must keep
  viewer-before-completion, but may not expose the raw destination while its
  reader thread mutates it. Owner: the progressive array source at the
  existing budgeted `read_region` seam; no renderer-specific locks or cache
  exceptions. Exit gate: deterministic atomic-read test, open-flow suite,
  full combined suite, and real-Wayland visual opens showing zero unread
  regions and correct final pixels/levels on WGPU then PyQtGraph then VisPy.
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
  on unfixed main `b0c3699b`, so pre-existing; the known tile-limbo/levels
  family. Real-Wayland rows complete; gate effect is diagnostic-only.
- **Kernel whole-process exit remains unbounded by current-item work.** The
  2026-07-19 shutdown change closes admission, cancels queued work/tokens and
  bounds the GUI close callback under one five-second join deadline, but the
  final real-Wayland matrix showed current non-daemon worker evaluations can
  keep the process alive after `kernel_shutdown complete`. Diagnose a
  cooperative cancellation boundary inside long slab/evidence evaluations;
  do not daemon-abandon NumPy/FFTW work. Exit gate: a real workflow process
  terminates in <5 s and the suite emits no leaked-thread diagnostics.
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
