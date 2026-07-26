# Per-commit whole-montage cost — why 18 transactions, and what one costs (2026-07-26)

**Status:** diagnosed and measured (§1–§7). The lever's *safe bound* is still not
established by this workload alone, so **the byte cap is untouched** — picking a
constant without those measurements would be the fourth speculative perf fix
this repo has reverted. §6 names what would justify it.

**§8 records three follow-ups since landed or refuted:** the 2.32× upload
undercharge is fixed (`cbd85384`, measured to change nothing here, which is the
point), the "two empty commits" premise is **refuted** — they are metadata
publications, not no-ops (§8.2) — and admission now names the cap that bit, so a
supply-starved batch stops reading as variance (`0d51e136`, §8.3).

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
**(d) is done — §8.1.** (a)–(c) remain open, and until they are, the cap stays.

## 7. Loose ends worth a line

- **Two empty commits per fill, ~90 ms each, every arm and every pass** —
  `delta_upserts=0`, `uploads=0`, `resident_rebinds` unchanged, and they run the
  full backend republication anyway. The second sets `coverage_pass_closed`, so
  it is doing *something*; the first differs from its predecessor in no traced
  field at all. `commit_gate_no_progress` fires on both — but with
  `outcome="backend-applied"`, i.e. after the 90 ms is already spent. ~180–380 ms
  per fill, 4–9% of the stage. **Followed up and reframed in §8.2.**
- **The refinement is bistable.** One baseline run (`base2`, unmodified code)
  refined in **49 batches of 3–8 upserts over 8.0 s** instead of 2 batches over
  0.28 s — `fully_visible_ms` 16 015 vs 4245. `max_upserts` was 12–32 and the
  deltas were 3–8, so neither cap bound: the *supply* of free retargets came in
  dribs. This is `f450660`'s pathology reappearing through a different door and
  it is not reproducible on demand (1 of 16 passes). It is also the single
  largest source of this stage's wall-clock spread, and it means the quoted
  4.0–4.9 s range understates the tail. **Made nameable in §8.3, which also
  narrows where it lives.**

## 8. Follow-ups taken (2026-07-26, same day)

Three, on top of `c9b572a1`. The byte cap itself is left exactly where §6 left
it, pending criteria (a)–(c).

### 8.1 The 2.32× undercharge is fixed — and it changes nothing here, by design

`cbd85384`. `wgpu_payload_upload_nbytes` is documented as "physical bytes for
WGPU" and was not: it charged a plane warm 451 584 B where the commit writes
four whole 256² pages, 1 048 576 B. The non-warm route was wrong by more — a
level-4 tile is 1 764 B logical against one whole page.

Predicted no behaviour change on this workload, because 32 MiB ÷ 1 MiB = 32 is
the same cohort `_idle_backlog_cohort`'s item ceiling already produced.
Measured none: 9–10 preview batches and 1258–1454 ms of commit time over three
in-process passes, inside the unmodified spread of six baseline passes (9–10,
1258–1816 ms). That coincidence *is* the result — it is why the error was
invisible, and why criterion (d) had to be settled before the byte cap's value
could be argued about at all. What changed is that the cap now describes the
transaction it paces.

### 8.2 The empty commits are not empty — premise refuted

The proposal was to gate them out before the republication, on the reading that
they change nothing. Instrumented once (a temporary `probe_empty_delta` trace
carrying every component of `_empty_progressive_commit_settled` plus the
metadata predicates, then removed), both commits in the 272-tile phase report:

```
publish_metadata=True  publish_histogram_plot=True  level_metadata_improved=True
visible_plan_complete=False   force_refresh=False   geometry_changed=False
```

They are **level/histogram metadata publications**. `_empty_progressive_commit_settled`
declines them correctly and on two independent grounds. Gating them out would
drop the published levels, not 180 ms of nothing.

The true statement is narrower and is the same root cause as §4, not a separate
bug: **a metadata-only commit pays a whole-montage tile republication**, because
levels and tiles share one commit path and
`_apply_backend_tiled_presentation` is driven by `montage_tile_payloads` rather
than by the delta.

The fix that *would* be correct is a mapping-only fast path: when the payload
set is identical by object identity to the previous commit's, and
representation, layout, `display_shape`/`transposed` and the resident page set
are all unchanged, every value the per-tile loop derives is a pure function of
those inputs, so `self._wgpu_committed` can be reused and the submission
reduced to `SetDisplayMapping` (+ camera). That is provable by construction
rather than by taste, which is the bar this file wants — but it is a real change
to the most dangerous function in the commit path for ~4% of the stage, and it
is not something to land in the same window as the FFT-stall work. Left for a
session that can own it.

### 8.3 Supply-starvation is now nameable — and it moves the search

`0d51e136`. `TileAdmissionDecision` carries `limit`: the first cap to defer
anything, or `""` when none did (first is the right one — admission walks in
priority order, so whatever stopped the highest-priority candidate set the batch
size). The commit trace carries it beside `admission_deferred` and
`admission_candidates`; `forget_admission_verdict` clears it per commit so the
atomic-successor paths, which never run admission, report `candidates=-1`
("did not run") instead of the previous commit's answer.

The reference fill now reads its own cause:

```
items  cand  defer      limit  maxU      ms   qual
   19    19      0                 2    85.2  preview     <- supply
   41    41      0                32    55.7  preview     <- supply
   32   212    180      items     32    66.7  preview     <- cap
   32   180    148      items     32    83.7  preview
   ...
   20    20      0                20   107.7  preview     <- supply
    0     0      0                12   102.9  -           <- admission ran, nothing offered
  101   101      0                32   133.0  exact       <- supply
  171   171      0                32   157.6  exact       <- supply
```

A 49-batch refinement would now show 49 rows of `limit=""` with
`candidates == items` — supply starvation, stated, not inferred from the absence
of a cap.

**And it already pays a second time.** The *healthy* refinement reads `limit=""`
too: its 2 batches are **supply-bound as well**, not one atomic transaction
admitted past the caps. So the good case and the bistable case differ only in
how the free-retarget supply arrives — which takes the commit caps out of the
frame entirely and puts the next investigation on the retarget/lifecycle side.
That is a narrowing the counter bought on its first run.

Counters only; no caps, no scheduling, no policy. Three in-process passes
measure 9–10 preview batches and 1288–1405 ms of commit time against the same
build without them at 9–10 and 1258–1454.
- **A/B discipline, paid for again.** The first pass of this investigation ran
  base → probe → probe sequentially and measured the probe **1.4 s worse**,
  which read as a clean refutation. It was process ordering: with in-process
  repeats and interleaved arms the same probe measures 1.7 s *better*. The
  dossier's warning is stronger than "~700 ms" — sequential single-shot
  processes on this machine drift far enough to invert a sign.

### 8.4 The literal payload-object predicate does not hold — no fast path landed

Attempted on local `main` at `6ad55232`, then reverted completely. The required
predicate was deliberately literal: the same payload objects, representation,
mapping mode, immutable layout owner, transpose state, and page-table binding
generation. The page-table generation is the honest O(1) resident-set key: it
changes on bind, eviction, re-admission, and slot remap, so it is stronger than
comparing resident keys while avoiding an O(1,088) key walk.

It never opened on the two target commits. In a real-Wayland raw-montage pass,
both final metadata-only transactions presented 272 tiles with zero delta but
**all 272 payload wrapper objects differed** from the preceding committed set.
The sampled mismatches retained the same image object, source ID, and LOD; what
changed was the `DisplayTilePayload` wrapper. This is not accidental churn:
`FrameSession.bind_payloads_to_level_generation()` uses
`replace(payload, presentation_identity=...)`, because
`DisplayCommitter._validate_presentation()` requires every active wrapper to
name the transaction's new level generation. The state builder then rehydrates
the acknowledged payload map from those current wrappers when backend identity
already matches. A level-only publication therefore cannot simultaneously
carry the new presentation identity and retain `DisplayTilePayload` object
identity under the current model.

Direct evidence:

- unmodified `--repeat 3`: the six final zero-delta commits cost
  `88.8, 105.0, 89.5, 87.1, 93.1, 89.9 ms` (median **89.7 ms**), all with
  `resident_rebinds=272`;
- predicate probe, one pass: the two final zero-delta commits cost 89.5 and
  83.7 ms and each reported **272/272 object-identity mismatches**;
- fault injection: forcing the mapping-only branch across a changed one-tile
  payload left the predecessor's 0.2 pixels in place where the CPU oracle
  required 0.8, and the framebuffer oracle failed as intended.

A backend-local substitute based on array identity plus selected wrapper fields
would be the forbidden deep-equivalence predicate under another name. A
durable token would have to become a separate canonical physical-binding
identity owned by payload construction and propagated through level wrappers,
floor reconstruction, resident crop rebind, representation/complex mapping,
transpose, and atomic successor handoff. That crosses the payload/lifecycle and
active ADR 0059 ladder owners excluded from this slice; a quick token attempted
at the level-wrapper seam also failed because floor payload reconstruction
creates fresh wrappers independently.

The future identity has to make every binding-changing case explicit:

- page eviction, re-admission, or slot remap changes the page-table generation;
- LOD changes the physical image/page identity;
- crop/window shift, transpose, sampling, or geometry changes the immutable
  layout identity;
- representation, complex mapping, colormap, or LUT changes the shader/mapping
  identity;
- atomic successor handoff changes the owning session/presentation identity.

The forced stale-pixel case proves that bypassing any one of those guards must
turn the framebuffer oracle red. Because the literal predicate failed before
the fast path could open on the real target, this slice does not claim or land
the rest of that safety matrix as tests.

Therefore **no production code, counter schema, or test change landed**. Scroll
and zoompan timing A/Bs were not run: the candidate took zero fast paths on the
target raw metadata commits, so interaction timings could not answer a shipping
question. Re-run this gate after ADR 0059 lands, because its coarse-rung payload
construction may change both wrapper stability and the transaction population.
Artifacts remain gitignored under
`tests/artifacts/unchanged-binding-fast-path-2026-07-26/`.

### 8.5 TileIdentity is the physical token — mapping-only fast path landed

Retaken after ADR 0059 at local-main tip `fe8bca3a`. The literal wrapper
predicate in §8.4 was too strong: `TileIdentity` already separates pixel and
binding identity from `TilePresentationIdentity`, which owns
`levels_generation`, levels, scale, and LUT identity. The WGPU predicate now
keys the complete presented tile set on each tile's `TileIdentity` plus its
explicit real/imag `ArrayPlaneIdentity` records. The explicit plane records
matter because they are deliberately excluded from normal `TileIdentity`
equality; pointer, shape, strides, and dtype still change a physical upload.

The remaining construction-owned guards are representation, shader mapping
mode, immutable layout/transpose identity, the executor object, and the page
table's binding generation. The generation is the O(1) resident-set proof: it
changes on bind, eviction, re-admission, and slot remap, but not on LRU touches.
The path also refuses removals, **all upserts**, and incomplete histogram
evidence. The all-upsert refusal is intentional: this change publishes only
mapping/metadata. An unchanged-binding target acknowledgement remains
recommendation 2's separate ADR-ladder problem.

When the predicate holds, WGPU submits `SetDisplayMapping` plus the current
camera, reuses the committed tile/page/instance state, and performs the same
shell-level level, histogram, viewport, overlay, and acknowledgement
bookkeeping. Per-commit trace counters state which arm ran:
`binding_fast_path_commits` and `binding_full_republications`.

The safety matrix is explicit:

- LOD/source sampling, representation, complex mapping, crop/window semantic
  generation, and atomic-successor document generation are in `TileIdentity`;
- real/imag buffer replacement is in the explicit plane records;
- tile placement and transpose are in the layout/transpose guards;
- eviction, re-admission, and remap change the page-table generation;
- display minification, levels, scale, colormap, and LUT are mapping state and
  are deliberately updated without rebinding;
- a non-empty upsert stays on the full path, even if its physical identity is
  equal.

The red-first oracle is not a timing assertion. Fault injection forces the fast
path across a changed float plane: the framebuffer retains the predecessor's
0.2 value (51 in RGBA8) where the CPU reference requires 0.8 (204), and the
pixel assertion fails. The positive level and LUT cases update the same
framebuffer to the CPU reference with zero uploads/rebinds. Focused tests also
change every identity dimension above and exercise eviction/re-admission plus
slot remap.

**Direct evidence.** The post-ADR unmodified `--repeat 3` baseline produced six
final 272-tile, zero-delta metadata commits at 62.1–69.5 ms (median 65.2 ms);
all six reported `resident_rebinds=272`. Because system load later rose enough
to push raw-stage wall time from roughly 5 s to 8–10 s, the A/B used ten
order-balanced processes (`base, fast, fast, base, base, fast, fast, base,
base, fast`), each with three in-process repeats: 15 passes per arm. Stage wall
time is not used. The qualifying transaction populations were:

| arm | commits | lower quartile | median | range | fast/full | resident rebinds |
|---|---:|---:|---:|---:|---:|---:|
| full republication | 26 | 89.0 ms | 114.2 ms | 64.6–560.6 ms | 0 / 26 | 7,072 |
| mapping only | 30 | 53.8 ms | 64.4 ms | 42.5–113.7 ms | 30 / 0 | 0 |

The lower quartile is the primary loaded-machine estimate: **35.1 ms removed
per metadata commit (39%)**. The median says 49.7 ms/44%, but is more load
sensitive and is not the headline. The target work counter is exact: 30/30
optimized commits skipped all 272 resident rebinds.

The real-Wayland `montage_scroll_scalar` plus `montage_zoompan_scalar`
`--repeat 3` check exercised 80 mapping-only commits. All 80 had zero upserts
and zero resident rebinds; none crossed the unsafe predicate boundary. Their
lower quartile/median were 10.0/11.3 ms. Under the recorded high load the
standing callback, heartbeat, warm-input, and zoom native-precondition gates
remained red, so this is not a claim that the gesture bars are green. The
correctness counters stayed clean: zero stale presentations, zero stale-level
tiles, and zero coarse-rung pixel failures.

ADR 0059 has already landed and is included in every number above. If its
coarse-rung upload policy changes again, rerun the raw and gesture populations:
that can change how often `TileIdentity`, LOD, or the resident page-table
generation remains stable, even though it does not weaken the predicate.

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
