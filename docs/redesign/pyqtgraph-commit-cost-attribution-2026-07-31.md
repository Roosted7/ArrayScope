# PyQtGraph commit cost: attribution of the ~8 ms/tile, and the one cut it authorizes (2026-07-31)

Answers finding **S3** of `gui-callback-owners-2026-07-31.md` (branch
`claude/blissful-chebyshev-a047a5`): during a 60-tile scroll build the
pyqtgraph `montage_commit` callback cost med 499 ms at items = 60 (~8 ms/tile
over a fixed base), pyqtgraph cannot rebuild a 272-tile montage inside
`INTERACTION_SETTLE_HARD_LIMIT_S = 5.0`, and queue row 1's "PyQtGraph at 2×
allowance" measured ~10× at field size. The mandate was measure-first: split
structural (genuine Qt item work) from incidental (recomputing what the delta
already knew) before cutting anything.

## 0. TL;DR

- **The ~8 ms/tile is not Qt item work.** Genuine Qt cost (ImageItem
  create/move/`setImage`/levels) measures **0.3–1.2 ms/tile**. The dominant
  term is `_resolve_page_backed_payload` — nearest-neighbor page expansion
  (`np.repeat`×2) plus complex→display mapping — run **synchronously in the
  commit callback** at ~0.5 ms per scalar payload and **~10–33 ms per FFT
  payload**.
- **The incidental multiplier:** assembly ran eagerly over the whole admitted
  batch while only the item loop honored `cold_deadline_ms`. One measured
  commit assembled 49 FFT payloads (1 446 ms inside a 1 487 ms callback) and
  then applied **5** of them; the 44 discarded assemblies were redone by the
  following commits. Whole-run: 911 payloads resolved for ~640 applied.
- **The off-thread seam exists but is starved:** the prepared-upload mailbox
  hit **94 / missed 765 / took stale 51** during the FFT phases (253 still
  pending when taken) and hit **3 / missed 719 / stale 179** during the scalar
  scroll. The keep-latest hand-off runs systematically one payload behind the
  commit.
- **The dossier's med-499-at-items-60 signature is the atomic shape:** every
  index step of the LOD-paced fft scroll is an `atomic_fast` whole-montage
  handoff assembling + uploading all 60 tiles in one callback (585–867 ms
  measured). All-or-nothing by contract, and `atomic_reuse=0` throughout —
  the prepared-transaction machinery never pre-stages it. That, plus the
  mailbox, is the remaining program (§4, §6).
- **The cut made here:** the resolve pass now spends the same
  `cold_deadline_ms` the item loop already honors; once burned (and ≥1
  payload resolved, so every commit progresses), unprepared page-backed
  upserts defer to the next commit exactly like deadline-cut tiles. Atomic
  handoffs are never cut. Kill switch: `ARRAYSCOPE_PYQTGRAPH_RESOLVE_BUDGET=0`.
- A/B (same binary, flag-only, order-interleaved eager/budget/budget/eager):
  governed-commit p95 527 → 81 ms, max 1553/2066 → 136/169 ms, medians and
  throughput unchanged; atomic/fallback shapes untouched as designed (§5).

## 1. Instruments (new, kept)

- `ARRAYSCOPE_UI_OBSERVATION_LOG=<file>`: the resource governor appends every
  GUI observation as one JSON line at record time. Exists because the two
  standard captures both failed here: the per-phase JSONL keeps only a
  64-observation tail (saturated by `kernel_bridge_drain` within a second),
  and a phase that dies mid-run (the fft-scroll stall, §6) never writes its
  row at all.
- The pyqtgraph tile layer appends a `tile_layer_sections` record per direct
  commit to the same file: per-pass wall time (`resolve`, `active_seed`,
  `reshow`, `hide_inactive`, `tile_order`, `level_scan`, `preclaim`,
  `item_loop` with `item_qt`/`item_upload`/`item_levels` buckets,
  `hide_and_prune`), tile counts, `resolve_deferred`, and prepared-mailbox
  counters.
- `montage_present_total` details now carry `set_image=` (the committer's
  `commit_tiled_delta` span) alongside the existing segment fields.

## 2. Where the 499 ms actually lives

Per-segment decomposition of every ≥100 ms pyqtgraph commit (fills at 272
tiles + fft scroll build at 60; obs_run3):

| commit | items | payload build | apply | …of which set_image | ack |
|---|---|---|---|---|---|
| 1 487 ms | 5 | 10.8 | 1 467.8 | 1 466.3 | 3.7 |
| 461 ms | 6 | 4.1 | 453.2 | 452.0 | 1.5 |
| 372 ms | 2 | 13.0 | 352.6 | 350.8 | 1.5 |
| 258 ms | 5 | 10.9 | 242.7 | 240.0 | 1.5 |
| 250 ms | 6 | 8.6 | 236.9 | 231.3 | 1.5 |

Everything is `apply` → `commit_tiled_delta` → `update_presentation`. Inside
`update_presentation`, the same commits' section records:

| resolve | item_loop | item_upload | item_qt | payloads resolved | applied |
|---|---|---|---|---|---|
| 1 446.4 | 12.5 | 10.8 | 0.1 | 49 | 5 |
| 430.8 | 14.8 | 11.7 | 0.5 | 53 | 6 |
| 324.0 | 16.4 | 15.5 | 0.0 | 34 | 2 |

Run-total (71 commits, fills + fft scroll build): **resolve 3 943 ms** vs
item_loop 483 ms (item_upload 345, item_qt 94). The other whole-montage
passes S3 suspected — `_reshow_retained_active_tiles`, the hide sweeps, the
`levels_match` scan, `_direct_preclaim_specs` — are **noise**: 1–12 ms total
each across the run.

The scalar index scrub is a different regime and is *fine*: 63 commits at the
60-tile grid, med 17 ms, items_med 2, resolve 405 ms / 881 payloads
(~0.46 ms each). The med-499 dossier number is the FFT-payload regime
specifically: ~30 ms per full-res complex assembly, ~4–8 ms per LOD-reduced
one, times the whole admitted batch, in one callback.

**The dossier's "med 499 / max 529 at items = 60" has a name now.** Joining
each commit's `montage_present_total` details to its section record splits
pyqtgraph commits into three shapes:

1. **Governed incremental commits** (deadline set, upserts ≤ batch): the
   1.4 s worst cases above — eager whole-batch assembly. This is the shape
   this change fixes.
2. **Atomic fast whole-montage handoffs** (`atomic_fast=1`, upserts = applied
   = 60): every index step of the LOD-paced fft scroll builds a complete
   successor transaction and assembles + uploads all 60 tiles in one
   callback — measured 585–867 ms each (resolve 420–630 ms + upload
   ~1.9 ms/tile). All-or-nothing by contract (a deadline would punch holes),
   so the resolve budget correctly exempts them. Notably `atomic_reuse=0` on
   every one: the prepared-atomic-transaction path never fires in this
   regime, so the assembly that machinery exists to pre-stage runs on the
   GUI thread every step.
3. **No-upsert fallback commits** (upserts = 0): the resolve restriction
   falls back to assembling *every* active payload — measured 733 ms to
   apply 5 tiles, 100 ms to apply 1. Rare (1–2 per run) but pure
   recomputation; cutting it safely needs dims derivable from
   `page_backing.source_coverage_yx` instead of the assembled array, because
   the staleness checks compare `local_rect` against assembled dims.

## 3. Structural vs incidental split

**Structural** (the work is real):
- Page assembly per applied FFT payload: `np.repeat` expansion + complex
  mapping, ~10–33 ms at full-res tiles, ~1–8 ms at scroll LODs. It is pure
  array work on immutable payloads — off-thread by design via
  `tiledUploadPreparations`/`PreparedUploadMailbox` — so it is structural to
  the *product*, but nothing forces it onto the GUI thread except the mailbox
  starvation (§4).
- Qt item work: 0.3–1.2 ms/tile (create+setPos+setImage+levels). At 272 tiles
  a full rebuild carries ~90–330 ms of genuinely GUI-thread Qt cost — over
  one 60 Hz frame, but far inside the 5 s settle limit.

**Incidental** (recomputing what the delta already knew):
- Eager whole-batch assembly ahead of a deadline-governed loop: the deadline
  cut the loop to 1–10 tiles while resolve had already paid for 12–53.
- Discarded assemblies are not retained anywhere, so the next commit redoes
  them: 911 resolved / ~640 applied across the run — and each redo is the
  expensive kind (the cheap ones commit on first try).

## 4. The mailbox is the real structural lever (not fixed here)

`_prepare_backend_uploads` offers admitted payloads to workers at admission,
and `_resolve_payload` takes a prepared buffer when key+levels match. The
counters say the seam is wired but not paying:

| regime | hits | misses | stale takes | pending at end |
|---|---|---|---|---|
| fills + fft scroll build | 94 | 765 | 51 | 253 |
| scalar scroll (60 tiles) | 3 | 719 | 179 | 1 |

Two shapes: during builds the workers are simply behind (253 preparations
still pending when the commits wanted them); during scrubs the keep-latest
slot hand-off is one payload behind (the slot's prepared buffer describes the
previous index by the time the commit asks — 179 stale takes). Fixing this
hand-off is the follow-up with the leverage: it converts the ~30 ms/payload
structural term into worker time. Deliberately not attempted inside this
change.

## 5. A/B: deadline-bounded resolve

Same binary, managed Weston, order-balanced A,B,B,A;
A = `ARRAYSCOPE_PYQTGRAPH_RESOLVE_BUDGET=0` (old eager behavior),
B = budget on. 60-tile scroll grid, fills at 272.

Per-commit wall time by shape (`montage_present_total`, ms). B1 and A3 hit
the fft-scroll stall (§6) after the fills, so their rows cover fills only —
which is where the governed worst cases live anyway; A2 died to a harness
mislabel (surface came up wgpu) and was re-run as A3.

| arm (order) | governed n / med / p95 / max | atomic n / med / max | fallback |
|---|---|---|---|
| A1 eager (1st) | 92 / 42 / **527** / **1553** | 21 / 204 / 884 | 798 |
| B1 budget (2nd) | 56 / 39 / 129 / 169 | — | — |
| B2 budget (3rd) | 118 / 38 / **81** / **136** | 21 / 150 / 867 | 486, 782 |
| A3 eager (4th) | 33 / 33 / 287 / **2066** | — | — |

Every eager arm produces second-scale governed commits (18 commits ≥300 ms in
A1 alone); no budget arm produces one over 169 ms — a ~6.5× cut at p95 and
~11× at max, with medians unchanged (38–42 ms). Atomic and fallback shapes
are untouched, as designed. Fill-phase heartbeat gap p99 follows: A1 1939 /
A3 1609 vs B1 648 / B2 898 (the residual max-gap is the S1 stage plan, not
commits).

Progress did not regress: total applied tiles A1 1978 vs B2 2002 (spread over
~24% more, far cheaper, commits); total GUI-thread assembly dropped 12.4 s →
7.6 s because deferred tiles assemble once instead of assembling, being
discarded by the item-loop deadline, and assembling again. fft scroll
converged in 53.1 s (A1) vs 55.1 s (B2) — same within run-to-run noise, and
`slow_scroll_converged` fails in both arms (the pacing target itself is
missed on this backend either way, per D2).

Mechanism check: `resolve_deferred` totals — A arms 0 by construction, B1
384, B2 834 — and the deferral test pins the semantics (progress guarantee,
no blink, no rejection reporting, atomic bypass).

## 6. Defects and refusals observed while measuring

- **The fft scroll build stalls under the managed Weston compositor, 3/3
  runs, on unmodified main** (`c1c295e5`): `[profile] STALL GUARD: montage
  frozen (no work in flight) — signature stable 2.5s: (4, 0, 60, 0, 0, 0, 1,
  0, 0, 0)` in `_slow_scroll_lod_paced`, then
  `TimeoutError: coverage pass did not close within 10.000s:
  coverage_open=True`. The scalar scroll phase passes in the same process.
  The gui-callback dossier saw this once on the real Wayland session; under
  Weston it is reproducible. Not touched here (frame-pacing-literate owner
  needed; compare D1's wgpu draw-debt latch).
- The 272-tile fills still measure gap p95 in the hundreds of ms on this
  backend either way — the fill owner is S1 (cold-entry stage plan), not this
  path.

## 7. The honest allowance (queue row 1)

With the incidental term cut, what remains at field size on this machine is
structural today: a 272-tile FFT rebuild costs roughly
`fixed base (~30–60 ms) + 272 × (assembly 10–30 ms + Qt ~1–2 ms)` of
GUI-thread work spread over deadline-bounded commits — roughly 3–8 s of total
GUI work, of which ~90% is assembly that *could* move to workers but today
does not (§4); and each paced-scroll index step remains one atomic ~10 ms/tile
callback (~2.5 s at 272 tiles) until the prepared-atomic path actually
pre-stages it. Until those two hand-offs are fixed, "PyQtGraph at 2×
allowance" should read **"PyQtGraph at 2× on scalar montages; FFT/complex
montages need ~10× at field size, and the deficit is page assembly, not
Qt"**. After the hand-offs are fixed the honest projection is the Qt floor
plus base: ~2–4× wgpu, i.e. the 2× allowance becomes defensible for the
first time.

## Repro

Managed Weston, this worktree, 2026-07-31:

```
ARRAYSCOPE_UI_OBSERVATION_LOG=<scratch>/obs.jsonl \
tools/env python -m arrayscope.tools.headless_display -- \
  python -m arrayscope.tools.profile_montage_workflow --backend pyqtgraph \
  --scroll-max-tiles 60 \
  --stages load_data,raw_full_tiled_montage,fft_full_tiled_montage,montage_scroll_fft,montage_scroll_scalar \
  --jsonl <scratch>/run.jsonl
```

Evidence channels: `tile_layer_sections` + `montage_present_total` details in
the observation log (survives the fft-scroll crash); `event_loop_*_gap_ms`
and per-phase `elapsed_ms` in the run JSONL. The scalar-scroll A/B swaps
`montage_scroll_fft` out of the stage list. Kill switch for bisection:
`ARRAYSCOPE_PYQTGRAPH_RESOLVE_BUDGET=0`.
