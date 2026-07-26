# The 272-tile FFT montage stall — one leaked pool layer (2026-07-26)

**Status:** root-caused and fixed. The fix is one line in
[`gpu/wgpu_executor.py`](../../arrayscope/gpu/wgpu_executor.py) plus a
regression test; §5 records a second, unfixed defect that turned a loud
`RuntimeError` into an opaque stall and is queued separately.

`arrayscope.tools.profile_montage_workflow --backend wgpu --stages
load_data,fft_full_tiled_montage` died on main (`f9f9229`) with

```
TimeoutError: STALL GUARD: montage frozen (no work in flight) after 4.0s:
  ... active_presented=271/272 ... visible_upserts=1 ... gate_no_progress=1
```

always on tile 0, always at 271/272. It blocked §6 of
[`preview-lod-anatomy-2026-07-26.md`](preview-lod-anatomy-2026-07-26.md) —
the operation-pipeline preview path could not be measured at all.

The stall was a **symptom**. The dump's own reading — "a lost wakeup or a
dropped final upsert" — was correct about the mechanism and wrong about the
level: the wakeup was lost because the commit *raised*, and it raised because
the complex page pool was permanently one layer short of its own budget.

## 1. What the stall dump could not see

The stall dump is printed by the profiler after the fact. Four seconds
earlier the run had already printed, and Qt had already swallowed, this:

```
RuntimeError: Error calling Python override of QObject::event():
  page pool 'complex_rg32f' exhausted and every resident page is pinned:
  pool=wgpu-complex_rg32f-pool budget=1088 allocated=1088
  resident=1087 pinned=1087 free=0
```

with the stack `_on_presentation_gate` → `commit_pending_session` →
`_commit_tile_layer` → … → `WgpuPlaneExecutor.submit` → `_ensure` →
`_ensure_free_pool_layer` → `_evict_one_unpinned`.

Read the counters: `allocated=1088`, `resident=1087`, `free=0`. Those do not
add up. **One physical layer is neither bound to a page nor on the free
list.** A temporary probe named it: layer index **0**, and it confirmed all
1087 resident pages were level 0, pinned by `wgpu-bound-content-planes`.

## 2. Where layer 0 went

Pool construction, `WgpuPlaneExecutor.__init__`:

```python
allocated = max(1, min(max(1, requested), budget)) if budget else 1
...
free_layers=list(range(min(allocated, budget))),
```

A pool built with `budget == 0` still allocates **one** layer — deliberately,
because bind groups require a valid texture even for a representation with no
budget. But `min(allocated, budget)` is then `min(1, 0) == 0`, so
`free_layers` is **empty**. Layer 0 is allocated and unreachable.

Nothing recovers it. `_grow_pool` only ever appends indices at or above the
previous extent (`free_layers.extend(range(old_layers, new_layers))`), and
`ensure_pool_budgets` raises `layer_count` without touching `free_layers`. The
pool is one layer short of its own ceiling for the rest of the process.

For every `budget`, `allocated <= budget` already holds, so `min(allocated,
budget) == allocated` in *every* case except `budget == 0`. The clamp only
ever did harm. The codec pool ten lines below already used
`list(range(allocated))`.

## 3. Why this dataset, and why exactly one tile

`_pool_budgets = {rep: max(0, budgets.get(rep, 0)) for rep in REPRESENTATIONS}`
— a representation absent from `pool_layers` is constructed at budget zero.
The workflow's `load_data` stage is **scalar**, so it builds the executor with
`{scalar_r32f: N}` and `complex_rg32f` is born at budget 0. Layer 0 of the
complex pool is leaked before the FFT stage exists.

The FFT stage then raises the complex ceiling to exactly the visible working
set:

| quantity | value |
|---|---:|
| tiles | 272 |
| L0 pages per 336² tile (`PAGE=256`) | 4 |
| `_wgpu_pre_reservation_page_count` → `pages_needed` | **1088** |
| byte-policy `policy_layers` clamp | ≤ 1088 |
| resulting `budget` (`max(needed, min(working_set, policy))`) | **1088** |
| layers actually usable | **1087** |

Zero slack, by construction. So the leak is exactly fatal: 271 tiles get their
four L0 pages, tile 0 cannot get its fourth, and every resident page is pinned
so eviction has nothing to give.

That also explains the two things the ticket asked about:

- **Same tile, same count, every time** (3/3 runs): the shortfall is exactly
  one page and admission order is deterministic.
- **`--max-tiles 64` does not stall** — verified, completes in 1791 ms. At 64
  tiles `needed` is 256 and the byte policy permits ~520 layers, so the budget
  carries real headroom and one leaked layer is invisible.

## 4. Not pre-existing — first bad is `51b826a` (2026-07-23)

A/B, all with `--session-fixture ""` so the pre-`d52019c` fixture breakage
cannot masquerade as the stall. Every run is
`--backend wgpu --stages load_data,fft_full_tiled_montage` on the same data.

| commit | date | pool | outcome |
|---|---|---|---|
| `e266260` | 07-23 | no exhaustion | **completes** 272/272, full refined 24 924 ms |
| `51b826a` | 07-23 | budget 512, resident 511 | exhaustion ×2 → completion timeout at **127/272** |
| `f3c98bb` | 07-24 | budget 1088, resident 1087 | **STALL GUARD** 271/272 |
| `6c9a602` | 07-25 | budget 1088, resident 1087 | **STALL GUARD** 271/272 |
| `65a9540` | 07-25 | budget 1088, resident 1087 | **STALL GUARD** 271/272 |
| `f9f9229` (main) | 07-26 | budget 1088, resident 1087 | **STALL GUARD** 271/272 |
| `f9f9229` + fix | 07-26 | no exhaustion | **completes** 272/272 (2/2 runs), full refined 4 316 / 7 789 ms |

Two distinct commits shaped this, and neither is the leak itself:

- **`51b826a`** ("size page retention from memory policy") is **first bad**. It
  introduced the byte-policy `policy_layers` clamp in `_wgpu_pool_layer_budget`,
  which lands the ceiling on the working set instead of above it. The leaked
  layer stops being free slack and starts being a shortfall. Only `f1287c1`
  sits between it and the last good commit.
- **`f3c98bb`** ("size residency for visible physical pages") raised the
  complex budget 512 → 1088, i.e. exactly the 272×4 L0 working set. That is
  what converted a messy multi-tile timeout into the deterministic
  single-tile 271/272 stall this ticket reports.

The *leak* is older than both: `min(allocated, budget)` arrived in `c0f086e`
(2026-07-22, "bound compressed residency and encode cost"), replacing
`list(range(self._pool_budgets[rep]))`. It was harmless for a day because the
budget still carried slack.

Note `e266260` and the fixed tip both still fail the unrelated
`gui_callbacks_below_50ms` perf gate; that gate is not part of this defect.

## 5. The amplifier — an exception through the presentation gate is a lost wakeup

`_on_presentation_gate` clears `_montage_presentation_gate_owner` and
`_montage_presentation_gate_armed` *before* calling `commit_pending_session`,
and `commit_pending_session` re-arms only by reaching `_rearm_if_backlog()` on
the way out. Any exception between those two points leaves the gate
disarmed with a live backlog, and the QEvent handler is the outermost Python
frame — Qt prints the traceback and continues. Every wakeup is gone.

This is not hypothetical, and not specific to the pool. While instrumenting, a
typo in the diagnostic raised `AttributeError` at the same depth and produced a
**byte-identical stall signature**: same tile 0, same 271/272, same
`gate_no_progress=1`. Any exception on the commit path is laundered into the
same anonymous four-second freeze.

That inverts ADR 0051's intent. The pool error is *designed* to be loud
("rescues hide bugs"); the gate silently converts it into the least
informative failure the system has. The fix is not to catch and re-arm — that
would hide genuine exhaustion — but to make a commit exception terminate
loudly instead of dissolving into a stall. Queued, not done here: it is a
policy decision about ADR 0051's failure semantics, not a bug fix.

One cheap improvement did land with the fix: `_evict_one_unpinned` now reports
`unaccounted=<n>` — layers neither bound nor free. A genuinely full pool reads
`unaccounted=0`; this defect read `unaccounted=1`. Had that counter existed,
the dump would have named the leak on the first run.

## 6. The fix

```python
# Every allocated layer is free until something binds it.
free_layers=list(range(allocated)),
```

Regression test:
`tests/gpu/test_wgpu_command_protocol.py::test_zero_budget_pool_keeps_every_allocated_layer_usable`
— builds an executor with complex absent (budget 0), asserts the bootstrap
layer is on the free list, raises the ceiling via `ensure_pool_budgets`, and
fills the pool to exactly its budget with pinned pages. It fails on the
unfixed tree (`assert 0 == 1`) and passes on the fixed one.

Measurement is unblocked: the full 272-tile FFT montage reaches **272/272**
on 2/2 runs, full refined at 4 316 ms and 7 789 ms. That spread is this
harness's normal run-to-run variance, not a residue of the defect — treat
neither number as a baseline until §6 of the preview-LOD dossier measures the
op-pipeline path properly.
