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
| 3 | **wgpu strangler — promotion evidence** (ADR 0057; slices (a)–(c) LANDED: native GPU overlays + glyph text, screen presentation behind `wgpu_present_method`, G6 compute, dogfood-crash/eviction-shield and codex-review fixes — full status narrative in the [Done ledger](queue-done.md) and dossiers). Open: **(d)** promotion by evidence — wgpu LEADS fast-scroll (p95 77.3 ms vs VisPy ~106–124), matches zoom/pan steady-state, 5/5 in the final matrix. **2026-07-20 correctness blocker — CLEARED 2026-07-21:** compositor screenshots showed native screen-mode overlay boxes not matching the Qt/PyQtGraph presentation, with the histogram/top-right composition clipped. Root cause was promoting the overlay chips to native *child* windows, which Qt never grants an ARGB visual (`alphaBufferSize() == 0`), so translucent rounded chips flattened into opaque boxes and occluded the histogram; restacking the swapchain below was unavailable too (`QWindow.lower()` emits no `place_below`). Chips are now rasterized from Qt's own painter and composited inside the frame (`widget_quad` + `UpdateWidgetAtlas`). Tool-managed Weston full-window evidence vs the bitmap reference went 3.45% -> 0.08% differing, the residue being only a Weston-drawn cursor, one transient live readout, and the bitmap-vs-bitmap noise floor. Tool-managed Weston or manual full-window compositor evidence stays required before AUTO promotion. **2026-07-21 successor review:** physical scroll evidence retains 60/60 tiles without stale/black holes, but WGPU shuffle first-new pixels remain red at 2.55 s (2 s bar). The final successor needed only two missing preview tasks (23 ms finish span) and reused the prepared atomic transaction about 129 ms later, so the remaining delay is in retarget/input trajectory, not a 60-tile cold-data or last-payload barrier. Built-in 0.1 s full-window capture is correctness evidence only: its synchronous compositor/readback load materially perturbs timing. One managed-Weston WGPU scroll run aborted in `wgpuSurfaceConfigure` after `ERROR_SURFACE_LOST_KHR`; an immediate identical rerun passed 60/60, so keep this as promotion reliability evidence if it recurs rather than hiding it as a harness timeout. VisPy's ~9 s comparison is diagnostic only: it reached session 10 where WGPU reached 42 and had a ~402 ms vs ~181 ms maximum event-loop gap. Do not optimize retiring-VisPy latency unless the same signature appears in WGPU/shared code. PyQtGraph's pre-existing 7/10 three-second LOD misses remain a separate standing red. Before the AUTO flip: shared row-1 callback bars, dogfood hours (screen mode selectable from Performance → wgpu Presentation), the FFT-scroll 4→17 fps headline on the new tip. VisPy retirement only via the roadmap ladder — never a flag-day switch. **Successor traps:** import rendercanvas ONLY via `import_qrenderwidget()`; every wgpu adapter probe pins `set_instance_extras(backends=["Vulkan"])` BEFORE its first request (`8c57a7bf`). | Promotion gate: journey matrix + perf bars on real data; written verdict in tensor-engine-endpoint.md |
| 4 | **G7 — compressed transport — codec + benchmark LANDED 2026-07-22 (`3450b75f`); default OFF.** The ZFP/blosc2 codec and the benchmark exist; the benchmark proved the compression inequality does NOT hold for CPU decode on this hardware (host-RAM win only, not transfer), so the default stays `raw`. **Remaining for a transfer win:** GPU-side decode (nvCOMP / wgpu compute) so decompress leaves the CPU critical path — then re-run the benchmark. See the [Done ledger](queue-done.md). | (transfer win) Measured `compress+transfer+decompress < raw` on real data with GPU decode |

## Next — the product turn (queued behind rows 3d/4; rationale in [roadmap.md](roadmap.md) and [reviews/2026-07-19-course-review.md](reviews/2026-07-19-course-review.md))

**The product turn (steps 5–10) is COMPLETE 2026-07-22.** Compare v1 (5/6/7),
plugin ops v1/v2/v3 (8, sigpy pack + Tier-2 harness for 9, BART pack for 10),
and G7 compressed transport (4) all landed — see the [Done ledger](queue-done.md).
`sigpy`, `blosc2`/`zfpy`, and `bart` were installed into the env this session, so
the previously dep-blocked gates ran on real data/tools. Deferred, with reasons
recorded in the ledger: sigpy `nufft`/`espirit` and `bart:pics` (multi-input /
don't fit the unary `fn(ndarray)->ndarray` contract).

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

- **Progressive-load publication correctness — DONE 2026-07-22** (core
  `447cbe42`, ring-4 residual `db1c5393`). Full evidence in the
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
  DONE 2026-07-22** (`977a86bb`). `FileOpenSession` now installs itself as an
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
  2026-07-22: reviewed — the `None` branch falls through to
  `acknowledged_identity_satisfies_target`, not an unguarded accept, and lives in
  the retiring VisPy backend; no clear defect found, left as a low-priority audit.
- **`tests/ui/test_diagnostics_dialog.py` serial `-n 0` failures — DONE
  2026-07-22** (`aa597939`). Root cause was NOT a diagnostics singleton but the
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
