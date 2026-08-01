# Who eats the frame — ranked GUI-thread callbacks per regime, per backend (2026-07-31)

**Status:** measured and attributed; **no code landed** (deliberately — every
candidate fix either belongs to a sibling chip or is not obviously right, §7).
The ranked lists below are the first published answer to "which callbacks
overrun the 50 ms bar", which `docs/queue.md` row 1 has carried as red without
attribution.

**Environment honesty up front:** every number here was taken on the real
Wayland session with a load average between 4.6 and 7.7 on 16 cores (other
sessions run suites on this machine). Absolute medians are therefore inflated
~20–30% against the queue's quiet-machine numbers — the documented 4.05 s cold
fill measured 5.1–6.7 s here; the documented ~90 ms whole-montage commit
measured 87–95 ms. Attribution (who is on the GUI thread, and roughly how big
each owner is relative to the others) replicated across 2–4 runs per cell and
is load-stable; treat single-cell absolute values as upper bounds, not medians
of record. The heartbeat gap — the metric that matches the complaint — is
measured directly by the harness's 2 ms probe, not reconstructed from callback
sums.

Repro commands and artifact notes: bottom of file. Raw JSONL stayed in the
session scratchpad per the no-artifacts rule.

## 1. The ranked lists

All rows: field montage, 272 tiles, `data/_WIPDelRec-tT2_20260223150234_14.nii`
(336×336×272 int16), `profile_montage_workflow`, real Wayland, wgpu bitmap
present. "budget?" says whether the path consults a work budget it then honors
(§3). Channels `montage_commit` / `montage_present_total` /
`tile_layer_commit(tile_layer_commit)` are the same callback observed at three
layers; they are listed once.

### 1a. Cold fill (raw + FFT full-montage build)

Heartbeat gap during the fill: **max 0.9–1.7 s, p99 0.86–1.6 s** on both
backends (bar: 16 ms). Six fills measured (4 wgpu, 2 pyqtgraph); the ordering
below held in every one.

| rank | callback (site) | wgpu | pyqtgraph | runs/fill | budget? |
|---|---|---|---|---|---|
| 1 | `win.render()` cold entry — `RenderCoordinator._flush_timer` → `window/render.py` render pipeline | 874–1368 ms | 924–1156 ms | 1 | **none** — no budget, no observation; shows up only as an unexplained heartbeat gap |
| 1a | └ of which `last_stage_plan_ms` (synchronous 272-tile stage-plan build) | 851–1336 ms (97%) | 902–1128 ms (97%) | 1 | none |
| 2 | FFT preview first commit (`_apply_backend_tiled_presentation`, whole montage) | 460–708 ms max | **1094–1125 ms** max | 1–2 | R5 `decide_render_pass` budget = 32 ms; overrun 14–35× |
| 3 | fit-stretch pulse total (`action_fit_stretch_ms`; enable/disable/retarget each run `viewport_range_bridge` → synchronous `retarget_montage_viewport`) | 524–959 ms | 308–1426 ms | 3 calls | report-only (recorded at ≥4 ms, never bounded) |
| 4 | steady-state fill commits (`montage_commit`) | med 42–51, max 105 ms @ 120–227 items | med 35–54, max 65 ms @ 1–6 items | 7–11 | feedback budget 2.0 ms (floor); item caps honored, fixed cost is not item-divisible |
| 5 | `kernel_bridge_drain` | med 9.4, max 29.5 ms | med 3.8–5.2, max 30.5 ms | 13–41 | **yes — honored** (`qt_bridge.py:128` breaks) |

The rank-1 entry is why the fill *feels* frozen rather than merely slow: one
callback owns the loop for ~1.3 s before the first commit ever runs. The
`phase_ui_work_observation` epoch max is 63–390 ms in the same fills — i.e. the
instrumented channels **cannot see** the largest callback in the app, because
the render pipeline reports no observation.

### 1b. Index scrub (fast scroll + slow LOD-paced scroll)

wgpu at the full 272-tile scroll grid; pyqtgraph could not build the 272-tile
scroll grid inside the 5.0 s `INTERACTION_SETTLE_HARD_LIMIT_S` (its plain cold
fill already takes 5.4 s here), so pyqtgraph is reported at the harness-default
60-tile grid — read its column as a *best case* and scale mentally by 4.5×.

| metric | wgpu, 272 tiles | pyqtgraph, 60 tiles |
|---|---|---|
| heartbeat gap p95 / p99 / max | **105–111 / 118–140 / 127–143 ms** | 27–65 / 60–89 / **560–847 ms** |
| top callback: `montage_commit` | **med 87–90 ms, max 130, at items_med = 1** (`payload=67–86 ms` in its own details) | **med 499 ms, max 529, at items = 60** |
| 2nd: `fit_stretch_enable/retarget` (synchronous whole-plan retarget) | 92–138 ms per call | 26–45 ms per call |
| input dispatch (`*_input_call_max_ms`) | 1.3–3.5 ms ✅ | 1.1 ms ✅ |
| everything else (`kernel_bridge_drain`, `histogram_refresh`, `roi_refresh`, `tile_layer` presentation) | ≤ 5.4 ms | ≤ 5.3 ms |

Two things matter here. First, the warm-scrub input bar (≤ 15 ms) is **green on
both backends** — input dispatch is innocent, the loop is starved by what runs
between inputs. Second, the wgpu commit costs ~90 ms *while carrying one item*:
this is the whole-montage-by-construction transaction the per-commit dossier
priced, seen live at scrub cadence. It alone accounts for the ~110 ms p95 gap.
On pyqtgraph the same callback is 499 ms at 60 tiles (~8 ms/tile with a fixed
base) — a 272-tile pyqtgraph scrub would spend ~2 s per commit, which is
exactly why it cannot settle inside 5 s.

### 1c. Pan/zoom on a settled montage (wgpu only, 272 tiles)

pyqtgraph produced **no data**: the zoompan montage build (even at the 60-tile
grid) timed out three times in three attempts after the earlier phases
(`loaded=16 target_unsettled=36 …`, `loaded=14 target_unsettled=37 …`, plus one
STALL GUARD firing on the scroll build). That refusal is the pyqtgraph pan/zoom
finding.

wgpu, heartbeat gap p95 per gesture (fft / scalar arm):

| gesture | gap p95 (ms) | input_call max (ms) |
|---|---|---|
| full-grid zoom-in | **924 / 1843** | **542** |
| zoom-in (central span) | 332 / 686 | 179–215 |
| zoom-out return | 393–533 | 192–361 |
| erratic zoom | 258–311 | 96–123 |
| pan (right/down) | 86–101 | 27–38 |
| deep zoom (6% span) | 55–59 | 17–20 |
| opposite pan | 29 | 1.2–3.7 |

The cost scales with **visible tile count**, not gesture speed: deep zoom over
a few tiles is 30× cheaper than full-grid zoom-in. During pan/zoom the
`montage_commit` observation runs med 53.8 / max 90.6 ms at items_med = 272.
And unlike scrub, here the *input dispatch itself* blocks — a full-grid zoom
wheel event takes up to 542 ms (p95 473) because the range-change handler runs
the whole-plan retarget synchronously before returning. The warm-input bar is
broken by 36× on zoom and ~2× even on plain pan.

The 946 ms max gap in the fft arm is **unexplained by any instrumented
callback** (epoch max 318 ms) — same signature as the fill: the retarget/input
path reports nothing.

## 2. Structural vs incidental

**Structural** — the work is genuinely that big and needs splitting:

- **S1. The cold-entry stage-plan build** (rank 1a): ~1.0–1.3 s of real
  planning CPU (`last_hot_stage_total_cpu_ms` ≈ 97% of it) in one callback.
  Nothing recomputes here; it has simply never been chunked. It needs the same
  treatment the commit path got in R5: plan in bounded slices with a
  continuation, or move the plan off-thread and commit it.
- **S2. The whole-montage commit fixed cost** (~90 ms at 272 tiles, `payload=`
  67–86 ms of it): structural per the per-commit dossier — 63–69% is
  `_apply_backend_tiled_presentation` iterating every presented payload. **A
  sibling chip owns this number; measured here, not touched.**
- **S3. pyqtgraph per-tile commit cost** (~8 ms/tile: 499 ms at 60 items,
  105 ms tile-layer apply): the backend rebuilds Qt-side items per tile. At
  field size this backend misses the product's own 5 s settle limit on *any*
  full rebuild. Splitting callbacks can't fix an 8 ms/tile floor; only
  per-tile cost reduction or the documented 2× allowance being made honest
  (queue row 1 says "PyQtGraph at 2× allowance"; measured today it needs ~10×).

**Incidental** — something recomputes what it already knew:

- **I1. Synchronous whole-plan viewport retarget on every programmatic range
  change.** `viewport_bridge.py:44` (`montage and not pointer_gesture` →
  `retarget_montage_viewport()` inline) and the fit-stretch pulse's
  enable/disable/retarget triple each cost 64–254 ms and run up to 3× per
  fill and once per scrub step burst; its EWMA reached **141 ms** during wgpu
  scrub (`viewport_range_bridge` feedback channel). The interactive
  pointer-gesture edge already has a coalescing scheduler
  (`_schedule_interactive_montage_viewport_update`); the programmatic edge
  bypasses it by design ("V1 boundary reveal requires its visibility update
  before the caller observes pixels") — that obligation deserves a narrower
  discharge than a full replan. Inside it, `frame_controller.py:1445–1477`
  rebuilds `previous_payloads` from a full retained-store scan *per budgeted
  item*, making the budgeted loop O(N²) — pure recomputation.
- **I2. The 542 ms zoom input dispatch** (1c): the same retarget running
  inside the wheel handler. The work it does is I1's; the placement (inside
  input dispatch) is the incidental part.
- **I3. Trace-bus argument construction on the commit path.** Verified today:
  `TraceBus._enabled` defaults `False` (`core/trace.py:73`) **but**
  `frame_runtime.py:538` re-enables a ring-only bus whenever the montage
  watchdog arms — i.e. on every montage session. 74 of 76 `emit_trace` sites
  build their kwargs before the enabled check; the `commit_batch` emit
  (`frame_effects.py:3481`) builds ~8 tile-count-proportional tuples per
  commit. Microbenchmark at 272 tiles: 66 µs disabled / 94 µs ring-enabled per
  `commit_batch` emit, plus 0.74 ms per 272 per-tile lifecycle emits —
  **~1 ms per whole-montage commit**, ~1% of the callback. Real, cheap to fix
  (the `if not trace_enabled(): return` shape already exists at 2 of 76
  sites), and *not* the frame-eater. Both prior claims about the trace bus
  ("hot-path tax" and "no longer a tax") were wrong in opposite directions.

## 3. Budgets: who consults them, who ignores them, who bypasses them

Full map from a systematic sweep (verified spot-checks marked ✓):

**Honored (2 sites in the entire app):**
- `kernel_bridge_drain` — `kernel/qt_bridge.py:103→128`, `should_yield()`
  breaks the drain loop; fed by the governor. Its numbers (med 1–9 ms, max 30)
  show the mechanism works where it is wired.
- `montage_viewport_update` — `frame_controller.py:1672→1696`, breaks and
  carries a remainder; but `item_cap=len(additions)` neuters the item cap, and
  the elapsed target only bounds the *loop*, not the per-item O(N) rebuild
  (I1).

**Report-only (observed, never bounded):** `montage_priority_retarget`
(`frame_runtime.py:975` — the entire retarget runs, then the budget is
consulted ✓), `histogram_refresh`, `roi_refresh`, `viewport_range_bridge`
(recorded only when ≥ 4 ms ✓), `resident_crop_rebind` (chunked upstream, no
in-chunk yield).

**Bypasses (no budget anywhere on the path), the ones that own §1:**
- the render pipeline itself: `render_coordinator.py:202 _flush_timer` →
  `win.render()` ✓ — rank 1a's owner;
- the commit apply: `imageview2d.py:923 → _apply_backend_tiled_presentation`
  (both backends) — bounded only by upstream item/byte caps, which two named
  modes drop entirely (`frame_effects.py:2236 limits = {}` on atomic-successor
  handoff ✓, and the `tile_layer_upsert_limits` empty-dict return when no
  dirty/pending work);
- pyqtgraph `update_presentation`'s four whole-montage passes before the
  deadline check (`backends/pyqtgraph/tiles.py:1380–1520`);
- paint-proportional items: `MontageTileOverlayItem.paint`,
  `_CompactPreviewAtlasItem.paint` (sorts every paint);
- `main.py:696 resizeEvent` (synchronous dimension-strip relayout).

**The structural break in the middle:** every window-side budget decision goes
through `window/main.py:665 _gui_callback_budget_decision` ✓, which returns the
**static** vocabulary (`batch_limit=1`, `model="static"`) and never consults
`latency_feedback` — even though the window holds the controller at
`main.py:140`. So half the adaptive machinery's outputs go nowhere by
construction.

## 4. Does the feedback loop converge? Yes — onto a saturated actuator

Live `resource_feedback_channels` snapshot during wgpu scrub:

| channel | EWMA (ms) | budget it computed | batch |
|---|---:|---:|---:|
| `montage_commit` / `montage_present_total` | 94.7 | **2.0 (floor)** | **1 (floor)** |
| `viewport_range_bridge` | **141.3** | 2.0 (floor) | 1 |
| `montage_render_pass_preview` | 94.7 | 32.0 (R5 requirement) | 1 |
| `tile_layer_commit` | 32.8 (274 items, 0.12/item) | 2.0 | 12 |
| `kernel_bridge_drain` | 1.5 | 12.0 | 5 |

The controller is not "just reporting" — it converged exactly where it should:
channels 47× over target sit at the minimum budget and batch = 1. The loop is
closed for `kernel_bridge_drain`, `tile_layer_commit`, `montage_present_total`
and the R5 render-pass cost model, and open (observe-only) for the rest. The
problem is the **plant, not the controller**: at batch = 1 the commit still
costs ~90 ms because the cost is whole-montage-fixed, and `viewport_range_bridge`
isn't batched at all. No tuning of this loop can go below one item; only
removing the fixed cost (S2) or the recomputation (I1) moves these channels.

## 5. Defects found while measuring (not fixed here)

- **D1. wgpu presentation-draw debt can latch.** Twice reproduced at the
  FFT→scalar scroll retarget: `requested==drawn` (261/261, then 314/314) with
  `presentationDrawPending()==True` for > 0.5 s.
  `_mark_presentation_draw_pending` (`imageview2d.py:638`) requests a repaint
  only on the QGraphics path; on wgpu the clearing ack comes solely from the
  wgpu draw callback (`wgpu_imageview2d.py:912`), so a late `visible_changed`
  upload that doesn't itself schedule a frame leaves the debt armed — and,
  worse than the harness timeout, means **changed visible content with no draw
  scheduled**. Likely related: the zoompan_scalar arm failed
  `full_grid_visible_tiles_reach_target_lod` + R1/R2/R5 invariants in the same
  run family. Needs its own chip; touching frame pacing "obviously right" is
  how the resize-coalescing traps happened.
- **D2. pyqtgraph cannot rebuild a field-size montage inside the product's own
  5 s settle limit** under this machine's load — 272-tile scroll grid: never;
  60-tile zoompan build after prior phases: 0 for 3. The queue's "PyQtGraph at
  2× allowance" is off by ~5× at this size.

## 6. What this dossier deliberately leaves to siblings

- The ~90 ms whole-montage commit (S2) — owned by the sibling chip attacking
  `_apply_backend_tiled_presentation`; everything here confirms its target is
  the right one and adds the scrub-cadence evidence (items = 1, payload = 86 ms).
- D1 (draw-debt latch) — needs a frame-pacing-literate owner.
- The trace-guard cleanup (I3) — mechanical, ~1 ms/commit, 74 sites; safe to
  do any time, pointless to do inside this research branch while the commit
  path is under active sibling surgery.
  **Closed 2026-08-01, and it was not the cleanup this line expected**
  ([trace-cost-2026-08-01.md](trace-cost-2026-08-01.md)): guarding the sites
  is a measured no-op because the watchdog arms the bus for every montage
  session (8.1k–11.3k True, zero False, eight sessions), and the default ring
  cannot be narrowed — the field stall dumps that produced three regression
  tests were decoded from the per-tile rows a narrower ring would drop. The
  ~1 ms/commit is one 107-field record whose cost is argument construction;
  removing its recomputation and its redundant re-hashing took ~23% off the
  session trace tax with byte-identical output.

The two fixes that would move the ranked list most, in order of leverage:
chunk/off-thread the cold-entry stage plan (S1, owns the 1.3 s freeze), and
give the programmatic range-replay edge a bounded discharge instead of a
synchronous whole-plan retarget (I1/I2, owns the 542 ms zoom input and the
141 ms EWMA bridge callbacks).

## Repro

Six invocations of `tools/env python -m arrayscope.tools.profile_montage_workflow`
on 2026-07-31, real Wayland session, load avg 4.6–7.7/16 cores, branch at main
`09d60ad1`:

- both-backend all-stage run (`--backend all --scroll-max-tiles 272`) — wgpu
  arm died at the D1 latch in `montage_scroll_scalar`;
- per-backend reruns (`--backend pyqtgraph|wgpu`, same stages) — pyqtgraph died
  at the 272-tile scroll build (5 s limit), wgpu reproduced D1;
- pyqtgraph at `--scroll-max-tiles 60` (scroll phases succeeded, zoompan build
  timed out), wgpu zoompan-only at 272 (both zoompan phases measured; exit 1
  from invariant gates, timings captured).

Per-phase JSONL fields used: `event_loop_*_gap_ms` + per-gesture `*_gap_ms`
(2 ms heartbeat probe), `phase_recent_ui_work_observations` (64-deep, truncated
— per-channel medians are over the tail of each phase),
`recent_over_warning_callbacks`, `resource_feedback_channels`,
`action_render_call_ms` / `last_stage_plan_ms` / `action_fit_stretch_ms`,
`*_input_call_max_ms`. Trace-cost microbenchmark: 2000 × `commit_batch`-shaped
emits + 200 × 272 per-tile emits, `tools/env python`, same session.
