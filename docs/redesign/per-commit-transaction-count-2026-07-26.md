# Per-commit whole-montage cost — why 18 transactions, and what one costs (2026-07-26)

**Status:** diagnosed, measured, **no code change landed**. The cause is proven
and the lever is identified; the lever's *safe bound* is not established by this
workload alone, and picking a constant without that would be the fourth
speculative perf fix this repo has reverted. What would justify landing it is
named at the end.

This closes the first open question of
[preview-lod-anatomy](preview-lod-anatomy-2026-07-26.md) ("why 18 transactions
for the preview and 2 for the refinement", §5.1 / Open questions) and re-opens
one of its premises.

Repro, env, and artifact locations: bottom of this file.

## 1. Worker arrival pacing is eliminated, on the clock

Of the three candidates the dossier named — cohort caps, the item clamp's
interactive arm, worker arrival pacing — arrival pacing dies first and cleanly.

In the reference run (`base`, 272-tile montage, wgpu, stage elapsed 4268 ms):

| event | t (ms) |
|---|---:|
| **last** `kernel_finish` for `(rung 0, level 4)` — all 340 preview evaluations | **4702** |
| first cold-carrying commit of the preview drain | **4880** |
| last preview commit returns | 5691 |

Every preview tile is computed and waiting *before* the drain begins. From 4880
to 5691 the backlog is full, no worker is pending, and the commit path moves it
32 tiles at a time in eight transactions. **No commit in the preview drain was
starved of supply.** The pacing is entirely on the presentation side.

## 2. The two numbers have two different causes

**The refinement gets 2 (often 1) transactions because it is *free*.** Its
payloads are already backend-acknowledged, so
`frame_session._free_retarget_tiles` puts them in `free_retarget_tiles`, and
`TileAdmissionQueue.admit` (`display/model/tile_admission.py:74`) short-circuits
every cap for a free item: not the item count, not the byte cap, not the
deadline. That is why the trace shows batches of 63, 209, 268 and 272 upserts
under `max_upserts=32` — those are not cap violations, they are the documented
bypass ("a remap is instant, so a burst of swaps converges in one commit").

**The preview gets 8–10 because it is cold and capped.** Every cold batch in the
drain reads `max_upserts=32` and `delta_upserts=32` — the item cap binds
*exactly*, eight times in a row. That 32 is the ceiling in
`_idle_backlog_cohort` (`window/frame_effects.py:4834`):

```python
if backlog > int(batch_limit):
    return min(32, int(backlog))
```

The byte cap does **not** bind at that point: `wgpu_payload_upload_nbytes`
charges a plane-warm payload 336²×4 = 451 584 B, so 32 items cost 14.4 MB
against the idle cap of 32 MB.

So: **the item ceiling sets the preview's transaction count, and the
resident-free bypass sets the refinement's.** Nothing about the preview pass is
slow; it is paced by a constant.

## 3. Bytes buy transactions — quantified

The §4c control (127 MB upload swing moved the refinement from 2 batches to 15)
asked for the exchange rate. It is a two-stage lock, and the stages have to be
lifted in order:

| arm (idle path only) | preview batches | items/batch | binding cap |
|---|---:|---|---|
| ship | 9 | 32 | **item ceiling 32** |
| item ceiling → 512 | 5 | 74 / 74 / 64 | **byte cap 32 MB** ÷ 451 584 = 74.3 |
| ceiling 512 + byte cap 512 MB | 4 | 212 in one | neither; supply |

The byte-cap prediction lands on the nose (74), which is the cleanest available
proof that the accounting is understood.

**Trap recorded while doing this:** `wgpu_payload_upload_nbytes` charges a warm
payload its whole *logical* plane (451 584 B) but the physical write is four
full 256² pages = 1 048 576 B — a **2.32× undercharge**. A 32 MB cap therefore
admits ~74 MB of `write_texture`. Correcting it today changes nothing (the item
ceiling of 32 binds first, and 32 MB ÷ 1 MiB = 32 by coincidence), but it
becomes load-bearing the moment the ceiling moves. `f450660`'s claim that the
byte cap "bounds hidden source-page warming without help from an item count" is
true in kind and short by 2.32× in degree.

## 4. What one transaction actually costs

cProfile ranked it first (`--repeat 1`, `sort_stats('tottime')`, ranking only:
`_apply_backend_tiled_presentation` cum 3.356 s / 22 calls, the single largest
arrayscope subtree in the commit; the rest of the profile is diffuse
`getattr`/`dict.get`/`len`, the signature of per-payload attribute walking over
the whole montage rather than one hot function). The repo's **own existing
per-commit counters** then confirm it without any new instrumentation — the
`tile_layer_commit` feedback details already carry the full split, so the
dossier's warning about hand-placed `perf_counter` windows in
`_apply_backend_tiled_presentation` never had to be risked:

| backlog `dirty/pending/presented`→delta | payload | prepare | **apply** | ack | state | total |
|---|---:|---:|---:|---:|---:|---:|
| 32/212/60 → 32 | 11.2 | 3.2 | **42.3** | 3.9 | 5.2 | 67.4 |
| 14/180/92 → 32 | 29.1 | 1.7 | **53.4** | 4.2 | 6.7 | 97.2 |
| 9/148/124 → 32 | 23.6 | 1.6 | **53.9** | 4.0 | 9.0 | 94.3 |
| 5/116/156 → 32 | 22.5 | 1.7 | **54.3** | 3.8 | 11.2 | 95.4 |
| 1/84/188 → 32 | 16.9 | 1.6 | **66.4** | 4.1 | 12.7 | 103.5 |
| 0/52/220 → 32 | 12.7 | 1.6 | **71.8** | 4.1 | 14.2 | 106.5 |
| 0/20/252 → 20 | 8.9 | 1.7 | **71.1** | 2.6 | 17.7 | 103.2 |

- **`apply` is 63–69% of every commit and it tracks `presented`, not the delta**
  — 42.3 ms at 60 presented rising to 71.1 ms at 252, ≈ **0.15 ms per presented
  tile**, while the delta stays flat at 32.
- `state` (publication) is the same shape: 5.2 → 17.7 ms, ≈ 0.065 ms/presented.
- `payload` build tracks `pending` the other way: 29.1 ms at 180 down to 8.9 ms
  at 20, ≈ 0.126 ms/pending.

`pending + presented` is the constant montage, so the three cancel — the cohort
dossier's `0.228×pending + 7.4` / `0.232×presented + 12.5` fit reproduces here
at a different slope on a different machine state, and the conclusion survives.

**The natural control is in the trace already:** two commits per fill carry
`delta_upserts=0`, `uploads=0`, `resident_rebinds` unchanged, at
`presented=272` — and cost **93.0 and 89.8 ms**. That is the whole-montage
republication with the delta subtracted out, measured directly rather than
fitted.

**Why it is whole-montage is structural, not incidental.**
`wgpu_imageview2d._apply_backend_tiled_presentation` (line 1390) opens with
`payloads = dict(montage_tile_payloads)` — every presented tile, not the delta —
and then runs it end to end: `textures`, `lod_geometry`,
`capacity_pages_by_tile` dict comprehensions over all payloads; `for tile in
sorted(payloads)` building a `_wgpu_payload_binding`, a `ContentPlane` and a
page-key walk per tile; a second full loop computing `chunk_key_frontier` per
tile for histogram evidence; then `BindContentPlanes(tuple(planes))` and
`UpdateTileInstances(self._wgpu_tile_instances())` over the whole set. It is a
**full-state republication by construction**, not a delta application. Making it
delta-proportional is not a tuning change — the executor's plane binding is a
whole-set replacement.

## 5. The A/B, order-balanced — and the premise it breaks

Probe (env-gated, reverted): idle-arm item ceiling and idle byte cap. Three
in-process passes per process, processes interleaved so `base` holds positions
1/3/5 and each probe arm 2/4 — the arm that would benefit from a favourable
slot is the one being *disadvantaged*, so the win below is conservative.

Median [min–max], 272-tile phase only, direct counters:

| arm | preview batches | preview ms | all commits ms | stage ms |
|---|---:|---:|---:|---:|
| ship | 9 [9–10] | 1006 [778–1199] | 1543 [1258–1816] | 5307 [4077–5966] |
| item ceiling only | 5 [5–6] | 850 [680–901] | 1609 [1408–1988] | 5657 [4970–6750] |
| ceiling **+ byte cap** | **4 [3–4]** | **465 [413–490]** | **860 [776–876]** | **3631 [3394–4119]** |

n = 9 / 6 / 6 passes.

**The item ceiling — the suspect the open question named first, and the one this
investigation set out to fix — is not the lever.** It cuts the batch count 9→5
and buys nothing: −16% on the preview window, *worse* on total commit time and
on the stage. Lifting the byte cap as well is what pays: −54% preview, −44% all
commits, −32% stage, with `base`'s best pass (4077 ms) barely below the probe
arm's *worst* (4119 ms).

The per-batch trace says why, and it corrects a premise:

```
ship            8 upload batches ×  32 items / 33.6 MB   66– 107 ms each
ceiling only    3 upload batches ×  74 items / 77.6 MB  168– 210 ms each
ceiling+cap     1 upload batch   × 212 items /222.3 MB          261 ms
```

**A commit does *not* cost 120–170 ms regardless of what it carries.** That
reading — taken from the 2-tile regime the cohort dossier fitted, and carried
forward into §4c — over-extrapolates. Across the whole range the shape is
`elapsed ≈ 0.5 ms×items + 0.34 ms×presented + 0.63 ms×MB + ~60` (n=97, R²=0.52
pooled across arms; the per-arm reading above is the trustworthy form). The
delta terms are small but not zero, so collapsing 8 transactions into 3 saves
one whole-montage republication three times and pays part of it back; only
collapsing to **one** clears the fixed cost outright. That is why 5 batches
measured flat and 4 measured −32%: the win is not linear in transaction count,
it is won at the last transaction.

The same effect appears unasked-for on the refinement: the ceiling-only arm
*fragmented* the exact pass (2 batches / 281 ms → 3 / 448 ms) because the slower
preview drain delivered exact tiles in waves; the ceiling+cap arm did it in one
(272 upserts, 197 ms).

Cost of the winning arm: worst GUI callback 187–262 ms → 233–295 ms.
`gui_callbacks_below_50ms` is red on this stage in **both** arms and was red
before either.

## 6. Verdict

1. **Preview 9, refinement 2 — answered.** Item ceiling `min(32, backlog)` on
   one side, `TileAdmissionQueue`'s free-item bypass on the other. Worker
   arrival pacing is refuted on the clock (§1). The replan coalescer is not
   involved: it already holds a one-per-event-turn barrier, and 456
   `replan_gate` events produce 22 commits.
2. **The transaction is whole-montage by construction** (§4), 63–69% of it in a
   backend republication that iterates every presented payload. A delta-shaped
   apply is worth more than any cap tuning — and is a redesign of the executor's
   plane binding, not a parameter.
3. **The cap that matters is the byte cap, not the item ceiling** (§5). This is
   the opposite of the standing assumption, including this investigation's own
   opening hypothesis and `f450660`'s "the byte cap stays authoritative" (it is
   authoritative — that is precisely the problem, and it is under-counting by
   2.32× while it does it).
4. **Not landed, deliberately.** A −32% stage win on 6 order-balanced passes is
   real evidence for the *cause*; it is not evidence for a shippable *constant*.
   512 MB is not a byte cap, it is the absence of one, and the idle arm it sits
   on is shared by every persistent-residency backend and every montage size.
   The principled form — "an idle transaction that already pays whole-montage
   cost must not be paced below the montage it is republishing" — is the right
   rule and is bounded by construction, but this workload cannot show what it
   does under pool pressure, and the page-pool layer leak fixed the same day
   ([dossier](wgpu-pool-layer-leak-2026-07-26.md)) is a standing reminder that
   large single-transaction admissions near the residency budget have a failure
   mode. Coordinating with the concurrent FFT-stall work in the same
   commit/replan code, this stops at the diagnosis.

**What would justify landing it**, in order: (a) the same A/B on
`montage_scroll_scalar` / `montage_zoompan_scalar`, whose idle settlement
commits share this arm; (b) a montage large enough that the whole backlog
exceeds the residency budget, to see whether a single transaction evicts inside
itself; (c) the journey matrix's bounded-commit oracle, which is the gate that
caught the last ungoverned batch (pyqtgraph zoom_in, `max_upserts=0`); (d) the
2.32× undercharge fixed first, so whatever bound is chosen means what it says.

## 7. Loose ends worth a line

- **Two empty commits per fill, ~90 ms each, every arm and every pass** —
  `delta_upserts=0`, `uploads=0`, `resident_rebinds` unchanged, and they run the
  full backend republication anyway. The second sets `coverage_pass_closed`, so
  it is doing *something*; the first differs from its predecessor in no traced
  field at all. `commit_gate_no_progress` fires on both — but with
  `outcome="backend-applied"`, i.e. after the 90 ms is already spent. ~180–380 ms
  per fill, 4–9% of the stage.
- **The refinement is bistable.** One baseline run (`base2`, unmodified code)
  refined in **49 batches of 3–8 upserts over 8.0 s** instead of 2 batches over
  0.28 s — `fully_visible_ms` 16 015 vs 4245. `max_upserts` was 12–32 and the
  deltas were 3–8, so neither cap bound: the *supply* of free retargets came in
  dribs. This is `f450660`'s pathology reappearing through a different door and
  it is not reproducible on demand (1 of 16 passes). It is also the single
  largest source of this stage's wall-clock spread, and it means the quoted
  4.0–4.9 s range understates the tail.
- **A/B discipline, paid for again.** The first pass of this investigation ran
  base → probe → probe sequentially and measured the probe **1.4 s worse**,
  which read as a clean refutation. It was process ordering: with in-process
  repeats and interleaved arms the same probe measures 1.7 s *better*. The
  dossier's warning is stronger than "~700 ms" — sequential single-shot
  processes on this machine drift far enough to invert a sign.

## Reproduce

```
ln -sfn /home/thomas/projects/ArrayScope/data data
env -u WAYLAND_DISPLAY python -m arrayscope.tools.headless_display -- \
  env QT_QPA_PLATFORM=wayland python -m arrayscope.tools.profile_montage_workflow \
  --backend wgpu --stages load_data,raw_full_tiled_montage --repeat 3 \
  --jsonl base.jsonl --trace base-trace.jsonl
```

Transactions: `kind == "commit_batch"`, `phase == "backend_complete"`, filter
`required_tile_count == 272`, split passes where `presented_tiles` drops. Fields
`delta_upserts` / `delta_qualities` / `uploads` / `upload_bytes` / `elapsed_ms` /
`max_upserts`. Arrival pacing: `kind == "kernel_finish"`, `(rung, level)` —
`(0, 4)` is the preview rung, `(2, 2)` the exact one. The per-commit split in §4
needs no new code: it is already in the JSONL at
`recent_over_warning_callbacks[*].details` for `channel == "tile_layer_commit"`
(`payload` / `prepare` / `apply` / `ack` / `state` …), emitted by
`_commit_feedback_details` (`window/frame_effects.py:5293`).

The §5 probe was two env flags on the idle arm of
`_persistent_tile_upsert_limits` / `_idle_backlog_cohort` (item ceiling; idle
byte cap), reverted after the runs. Artifacts under
`tests/artifacts/percommit-2026-07-26/` (gitignored).

Use `--repeat` and interleave arms across processes. Do not compare
single-shot processes run back to back — §7 says what that costs.
