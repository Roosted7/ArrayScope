# Commit cost vs. what changed — where the 90 ms went, and what is left (2026-07-31)

**Status:** measured and partially landed.  The WGPU backend's remaining
whole-montage per-commit terms are removed (`f495424d`, `c83a934f`); the
session-side per-commit terms are measured and named but deliberately left
for their owners (§6).  All regimes are 272 tiles unless stated.

This continues [per-commit-transaction-count-2026-07-26.md] and closes its
recommendation 2 ("a delta-shaped apply is worth more than any cap tuning")
for the executor's plane binding.  Do not re-derive that file's results; §1
here states how much of its world has already moved.  Two same-day sibling
dossiers cover the adjacent ground: [wgpu-bookkeeping-scaling-2026-07-31.md]
(executor bookkeeping vs. total residency — mostly measured innocent) and
[gui-callback-owners-2026-07-31.md] (the ranked callback census this file's
§6 session-side terms appear in).  This slice's changes compose with the
former's eviction-victim fix: forced eviction still routes through
``unbind``, which is exactly what moves the eviction stamp §2.3 relies on.

## 1. The 2026-07-26 numbers are obsolete — three landings in between

The headline that motivated this slice — *"63–69 % of the commit is
`_apply_backend_tiled_presentation` iterating every presented payload, and an
empty-delta commit still costs ~90 ms"* — was measured before three changes
landed (2026-07-27..29, `bebe9763`, `fb901bde`, `8e9d2cbc`, plus §8.5's
mapping-only fast path):

- the WGPU **incremental-delta apply** (`work_payloads = delta_upserts`, tile
  instances patched, tile order reused);
- the **mapping-only fast path** for zero-delta metadata commits;
- the PyQtGraph resolve restriction to admitted upserts.

Re-measured on main at `09d60ad1` (reference fill, `--repeat 3`, wgpu,
foreign load ~10 of 16 cores — counts are load-proof, milliseconds are not):

- **49 commits at 272 required: 44 incremental, 2 fast-path, 3 full — and
  every full republication was a 1-tile first-commit of a pass.**  The
  whole-montage republication of small deltas is already gone from this
  workload.
- The two metadata publications (the former "~90 ms empty commits") both
  took the mapping-only fast path: **the fast path qualifies 2-for-2 on the
  fill**, and 97 of 258 commits (37 %) on the scroll stage.  It is not a
  rarely-qualifying fix.
- What remains is a **montage-proportional floor around every commit**,
  spread over four layers (session build, level publication, backend entry,
  backend finish).  Under load the two metadata commits cost 92/109 ms
  elapsed — but only ~25 ms of that was backend apply; 45–59 ms was
  session-side payload build and 11–15 ms state publication.  The 90 ms
  problem did not disappear; it moved up the stack and shrank.

## 2. What this slice landed (WGPU) and why that shape

Three backend terms were still O(presented) for a bounded delta.  All three
had clean physical proofs available, which is why they were taken and the
session-side terms were not (§6):

1. **Executor plane rebind → in-place patch** (`_patch_bound_planes`).
   `BindContentPlanes` knew only "identical prefix" (append) or "rebuild
   everything".  A bounded commit replaces a few `ContentPlane` entries in
   place, so a 1-tile rebind re-derived grids, family indices, and the whole
   flat lookup table.  The patch rewrites exactly the changed indices;
   footprint changes (`plane_shape`/`max_lod`) and wide swaps (> ¼ of the
   list) still rebuild, because there the rebuild IS the honest cost.
   Counters `plane_bind_{prefix_reuses,patches,rebuilds}_total` say which
   path ran (also in `wgpuPresentationDiagnostics`).
2. **Pin-set replacement → generation-gated memo.**  The same montage-sized
   key tuple is re-sent up to three times per commit; each send re-hashed
   ~1 088 pages.  The memo is per owner, keyed on the exact tuple and the
   page table's binding generation (moves on every bind/unbind, never on pin
   changes), so any residency change invalidates it.
3. **Presented-set verification → delta-proportional under an eviction
   stamp.**  `_finish_wgpu_tiled_presentation` re-walked every committed
   tile's pages per commit.  `PageTable.eviction_stamp` (per-instance epoch +
   serial bumped only on `unbind`/`rebind_from`) proves "nothing verified
   resident was lost"; an incremental commit whose base serial and stamp
   match re-verifies only its changed tiles plus any the predecessor could
   not present, and is **exactly equivalent** to the full walk (bind only
   adds coverage; unchanged tiles keep their page keys by reference).  Any
   eviction — including pool pressure inside the same submission — falls
   back to the full walk.

Alternatives considered: splitting the commit across frames under
`gui_callback_budget` does not reduce the O(montage) work, only smears it —
the interactive 4 ms target is unreachable while any montage-proportional
scan remains, so the scans had to become delta-proportional first.  Making
the fast path "qualify more often" was not the lever either: it already
qualifies whenever its predicate is honest (§1).

## 3. Measured — interleaved, in-process, order-balanced, with a control

One process, one settled 272-tile offscreen WGPU view, arms alternated per
block (`change, base, control, control, base, change`, reversed on alternate
rounds), n = 48 commits/arm.  *base* = patch off + memo bypassed + presented
cache cleared per commit (pre-change behavior); *control* = change arm run
twice, bounding the noise floor.  Foreign load ~7–12 of 16 cores; the
control arm is what makes these numbers usable anyway.

| 1-item commit, 272 presented | change | base | control |
|---|---:|---:|---:|
| plane **swap** (scroll/crop-rebind shape) | **3.10 ms** [lq 2.88, uq 3.61] | 5.46 ms [lq 5.20, uq 6.45] | 3.24 ms |
| plane **refresh** (same plane, new pixels) | **2.35 ms** [lq 2.08, uq 2.82] | 3.00 ms [lq 2.78, uq 3.37] | 2.32 ms |

Swap distributions do not overlap (change uq 3.61 < base lq 5.20): **−43 %**
for the swap shape, −22 % for the refresh shape.  Classification is proven,
not assumed: the swap A/B recorded 96 patches (change+control arms) against
49 rebuilds (48 base commits + the initial bind), and every measured commit
asserted its binding class.

Absolute per-class costs at 272 presented (same probe, single-arm pass,
near-idle machine): mapping-only ~13.6 ms **of which ~9 ms is the caller's
272-wrapper rebuild pattern and only ~3–4 ms backend**; incremental 1-item
6.3 ms (before this slice's changes); forced full republication 65 ms.  The
65 ms full is within sight of the 07-26 "~90 ms whole-montage" figure and is
now paid only when the delta IS the montage.

Pixel truth: the patch tests draw per-plane value ramps against the CPU
reference (a stale patched plane is a pixel diff); the presented-set test
evicts an unchanged tile's pages between deltas and requires it to leave
`presented_tiles` — deleting the stamp guard turns exactly that test red
(mutant verified).  Fault-injection and eviction/remap cases from the
existing §8.5 suite still pass.

## 4. How often the incremental precondition holds in real interaction

From `commit_batch` traces (counts, load-proof):

- **Fill (272):** 44 incremental / 2 fast / 3 full, fulls all 1-tile
  first-commits.  The precondition effectively always holds during a fill.
- **Scroll (`montage_scroll_scalar`, 60 required, 2 repeats):** 135
  incremental / 97 fast / 26 full.  Of the 26 fulls, **22 carried the whole
  montage as their delta** (60 upserts at 60 presented — an index step
  retargets every slot, so there is nothing incremental to save) and 4 were
  first-display commits.  Small deltas (1–3 items) were 105 of 135
  incremental commits.

So the answer to "what breaks the precondition" today is: **nothing that
matters** — it fails only when the delta is the montage (honest full cost)
or there is no predecessor.  The out-of-view retention work already removed
the viewport-shrink breakage.

## 5. PyQtGraph — measured, one probe defect found, code left alone

Same probe against `cpu_item` at 272:

- **level-only commit: 41.6 ms** — ~275 `ImageItem.setImage` calls: the CPU
  backend re-windows pixels on a level change.  That is real re-bake work,
  paced by the governor's level-rewindow deadline in production; not a
  bookkeeping defect.
- **1-item commit: 9.95 ms**, of which the montage-proportional parts are
  the histogram-mosaic buffer copy (~2 ms — it feeds a real upload and the
  id-keyed change detection requires a fresh array), `presented_tiles`
  computed twice per commit (~1.2 ms each, one Qt `isVisible()` per intended
  tile, and the two calls straddle visibility mutations so they cannot be
  blindly memoized), and `_update_direct_payload_presentation`'s remaining
  O(montage) sweeps (~1.4 ms).

**Probe trap for the record:** `TilePresentationDelta.__post_init__` derives
`target_revision = base + 1` when upserts exist, so a probe that leaves
`base_revision` at 0 mis-keys the payload-histogram cache and measures a
whole-montage histogram rebuild (~4.5 ms/commit) that production — whose
deltas chain revisions — never pays.  First measurements of this slice
overstated the PyQtGraph floor by exactly that term.

The remaining PyQtGraph terms each need their own design (visibility-sweep
memoization has a mutation-ordering hazard; the mosaic copy is load-bearing
for change detection) and none has the clean physical proof the WGPU items
had; they are named debt, not this slice.

## 6. The commit pipeline's remaining O(montage) terms — session side

Per-commit splits from `commit_batch` and `presentation_build_timing`
traces (272-tile fill, loaded ~2×; treat ratios, not absolutes):

| term | 1-item target commit | metadata commit |
|---|---:|---:|
| `build_tile_presentation` (mostly the `candidates` segment: planned/stale-level/identity-repair/rearm scans) | ~8–11 ms | 15–27 ms |
| `payload_level` (shader level source: `_shader_presented_payload_bounds` unions bounds — and builds an observation dict — over every presented payload, per commit) | ~5–7 ms | ~6–12 ms |
| `state_publish` | ~3 ms | 11–15 ms |
| backend apply (after this slice) | ~3–6 ms | ~3–4 ms + wrapper churn |

The largest single remaining item is the session build's `candidates`
segment — a dozen O(planned)/O(payloads) passes in `build_tile_presentation`
(frame_session.py:4194–4463).  Second is `_shader_presented_payload_bounds`,
which builds 272 observation dicts per commit that are only consumed when
the R3 clamp actually fires; a lazy two-pass (bounds first, observations
only when clamping) is the obvious increment, **but the clamp's release edge
is load-bearing (R3 dossier) and sibling chips are currently in
frame_session/frame_effects** — changing it in the same window invites the
composition-seam failure this repo has hit four times.  Left with numbers
attached.

## 7. Verification state

- New tests: `tests/gpu/test_wgpu_plane_bind_patch.py` (5, pixel-oracle),
  `tests/gpu/test_page_table.py` eviction-stamp trio,
  `tests/display/test_wgpu_imageview2d.py::test_incremental_presented_set_drops_a_tile_evicted_between_deltas`
  (mutant-verified red-first).  All pass.
- **Shadow verification is the load-proof evidence.**  Because the machine
  never went quiet (a sibling session held 8–37 of 16 cores all session),
  correctness was settled by a temporary build that recomputes each slow
  path beside its fast path and raises on any difference: the presented-set
  full walk, a from-first-principles reference for the flat lookup table /
  plane rows / family indices, and a fresh pin-set filter on every memo hit.
  It caught a deliberate mutant (a patch that skips refilling its flat span
  → `SHADOW flat-table divergence: 2 entries`), so it has teeth.  Across the
  **full 377-test affected set and 78 live UI tests it recorded zero
  divergences** — the three fast paths are behaviourally identical to the
  code they replace on every path those tests exercise.
- **Baseline diff: indistinguishable from main.**  The affected set ran
  375/377 (the 2 stragglers both pyqtgraph-only, which no part of this diff
  reaches).  Taking the 8 tests that `--since` reported and running them on
  both arms interleaved: branch `1 failed / 8 passed`, main `1 failed /
  1 failed` — same population, same rate.  The most stubborn of them,
  `test_single_slice_to_full_montage...[scalar]` (a declared known flake),
  **fails 8/8 on unmodified main** under current load, having passed 4/4 on
  main twenty minutes earlier.  Its outcome tracks load, not this diff.
- **Mechanism check, not just counts:** the pyqtgraph backend imports only
  `PageResolution` from `page_table` (a frozen dataclass untouched here) and
  never constructs the WGPU executor, so a pyqtgraph-only failure has no
  path to this change.

## Reproduce

Probe (scratch pytest module, untracked, session artifacts dir):
272 typed-payload tiles at the `_present_tiled` seam; arms assert their
binding-class counters; A/B via monkeypatching the three new paths off for
the base arm in one process.  Macro:

```
env -u WAYLAND_DISPLAY tools/env python -m arrayscope.tools.headless_display -- \
  env QT_QPA_PLATFORM=wayland tools/env python -m arrayscope.tools.profile_montage_workflow \
  --backend wgpu --stages load_data,raw_full_tiled_montage,montage_scroll_scalar \
  --repeat 2 --jsonl after.jsonl --trace after-trace.jsonl
```

Commit classes: `commit_batch`/`backend_complete` records,
`binding_{fast_path,incremental,full}_*` fields; plane-bind path split:
`wgpuPresentationDiagnostics()["wgpu_plane_bind_*_total"]`.  Session-side
splits: `payload_*_ms`/`state_publish_ms` on `commit_batch` and the
`presentation_build_timing` records.
