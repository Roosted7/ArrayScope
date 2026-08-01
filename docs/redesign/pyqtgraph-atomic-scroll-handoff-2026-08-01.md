# The atomic FFT scroll handoff and the prepared-upload mailbox (2026-08-01)

Closes the two terms `pyqtgraph-commit-cost-attribution-2026-07-31.md`
deliberately left standing: the `atomic_fast` whole-montage handoff (§2 shape 2,
585–867 ms per GUI callback with `atomic_reuse=0` throughout) and the starved
prepared-upload mailbox (94 hits / 765 misses / 179 stale takes).

## 0. TL;DR

- **They are one defect, in two places.** Page assembly is content-addressed
  pure work; both hand-offs that could carry its result were keyed by
  **destination slot**. A montage index scroll moves content between slots —
  after one step slot *i* shows what slot *i+1* showed — so neither hand-off
  could ever hit on the very workload that re-asks for the same pixels every
  step.
- **The upload was the second victim of the same cause.**
  `_direct_payload_source_id` keys item residency on `id(image)` of the
  assembled array. Re-deriving mints a new array, so a fresh assembly *also*
  failed to match the resident ImageItem that was already holding those exact
  pixels one slot over, and paid a full `setImage` for them. The layer's own
  `payload_resident` said the content was resident — which is why the mailbox
  had stopped preparing it — while the same commit re-derived and re-uploaded
  it.
- **Fix: retain assemblies under their content identity** (`AssemblyRetention`,
  byte-bounded LRU, dropped at a residency reset), and **key the mailbox by
  identity instead of slot**. Retention returns the same array object, so item
  residency matches again and the upload disappears too — without touching the
  `id(image)` residency contract.
- **Isolated bench** (60 tiles, 512² complex, one-slot atomic shift,
  order-balanced ABBA in one process with a same-code control pair): per-step
  callback med **64.8 / 69.0 ms → 7.4 / 6.5 ms**; assemblies and uploads
  **420 → 7** over 7 steps (i.e. 60 per step → 1). Zero wrong slots in every
  arm, checked against a per-slot pixel oracle.
- **End-to-end** (managed Weston, `profile_montage_workflow`): the `atomic_fast`
  population now measures **n = 10, med 51.9 ms**, against the dossier's
  585–867 ms for the same shape (§5). `payload` (transaction build) is 6.3 ms
  of that — see §4 for why `atomic_reuse` was never the lever.
- **`atomic_reuse` is still 0, and that is the right answer.** It caches the
  *delta*, not the pixels; it could only ever have removed the ~6 ms build.
- Mailbox: **217 hits / 664 misses** against 94 / 765, and "prepared for
  something else" stops being an outcome at all (§6).
- A latent correctness hole came out with it: `prepared_upload_key` did not
  distinguish a coarse page set from the fine one that replaces it (§3).

## 1. What the callback was actually spending

The 2026-07-31 attribution had the cost right (`resolve` 420–630 ms of a
585–867 ms atomic callback) and the cause one level too shallow: it read the
assembly as *structural* work that merely happened to be on the wrong thread.
It is not structural at all on this workload. Of the 60 payloads an index step
hands over, 59 carry pixels the process already built, still holds, and is still
drawing.

Two independent keys had to be wrong for that to cost anything, and both were:

| seam | keyed by | what a one-slot shift does to it |
|---|---|---|
| `PreparedUploadMailbox` | slot, with key checked on take | buffer for slot *i+1* refused as stale when slot *i* asks |
| page assembly | nothing retained it | re-derived from pages every step |
| item residency (`_states_by_source_key`) | `id(image)` of the assembled array | fresh array ⇒ resident item missed ⇒ full `setImage` |

The third row is why fixing assembly retention alone also removes the upload:
hand back the *same array object* and the existing residency match starts
working, with no change to what `id(image)` promises.

## 2. The instrument said so already

`payload_resident` (used by `_prepare_backend_uploads` to skip planning) scans
every state for a matching identity **and matching page resolutions**,
regardless of slot. On the measured scroll it was answering *yes* — which is
exactly why `submitted` froze at 60 for the rest of the run and the mailbox
looked "starved". The layer knew the pixels were resident. The commit path did
not ask.

So the reported 765 misses decompose into three different things, only one of
which was a hand-off failure:

1. resolves that the deadline then **deferred without assembling** — `take`
   ran before the defer decision, so a commit that packed nothing still
   recorded a fallback. Now `holds` is asked first and no miss is recorded.
2. resolves for payloads planning had **correctly skipped as resident**.
3. genuinely un-prepared content.

## 3. A latent hole in the shared key

`prepared_upload_key` was `(tile_identity, source_id, variant)`, documented as
carrying the physical page distinction via `source_id`. Progressive refinement
is the case it does not cover: the same tile of the same plane keeps one
identity and one `source_id` while its coarse pages are replaced by fine ones,
and the derived pixels change completely. The mailbox survived this only because
`take` consumes and a publish is rarely outstanding across a refinement — a
store that *retains* serves the coarse pixels forever.
`test_pyqtgraph_replaces_coarse_page_pixels_when_fine_pages_arrive` caught it on
the first run. The resolved page set is now part of the key, and PyQtGraph's
item-residency token delegates to the same helper so the two cannot disagree.

## 4. Why `atomic_reuse` was never going to help

The dossier read `atomic_reuse=0` as "the prepared-atomic machinery never
pre-stages this". It cannot: `session._atomic_prepared_transaction` is written
only when `_warm_atomic_successor_residency` reports incomplete residency and
the commit bails, and what it stores is the frozen `(base_tile_state,
tile_state, tile_delta)` triple. It caches the **transaction**, not the pixels.
On the measured atomic callbacks the transaction build (`payload=`) is 6.3 ms.
`atomic_reuse` remains 0 after this change, and the callback is 51.9 ms: the
number the flag was being blamed for was never in its reach.

## 5. Measurements

### 5a. Isolated handoff bench, order-balanced in one process

60 tiles, 512² complex pages, atomic handoff, slots shifted by one per step,
levels fixed. Arms are the kill switch only (`ARRAYSCOPE_PYQTGRAPH_ASSEMBLY_
RETENTION`), same binary, same payload objects, interleaved A,B,B,A with a
same-code control in each pair. Step 0 (cold build) excluded; 7 scroll steps.

| arm (order) | med ms | max ms | total ms | assemblies | uploads | wrong slots |
|---|---|---|---|---|---|---|
| A1 eager (1st) | 64.8 | 119.3 | 509.0 | 420 | 420 | 0 |
| B1 retain (2nd) | **7.4** | 8.2 | 51.7 | **7** | **7** | 0 |
| B2 retain control (3rd) | 6.5 | 6.8 | 44.6 | 7 | 7 | 0 |
| A2 eager control (4th) | 69.0 | 72.9 | 449.1 | 420 | 420 | 0 |

The same-code pairs agree to within 5 ms (A1 64.8 / A2 69.0; B1 7.4 / B2 6.5),
so the ~9× is far outside this box's run-to-run spread. Assemblies and uploads
are counts, not timings, and they are exact: 60 per step becomes 1.

### 5b. End-to-end, real workload

Managed Weston, `profile_montage_workflow --backend pyqtgraph
--scroll-max-tiles 60`, stages `load_data,raw_full_tiled_montage,
fft_full_tiled_montage,montage_scroll_fft`. Same binary, arms are the kill
switch. Commits split by `atomic_fast=` in the `montage_present_total` details
rather than by applied-item counts — with retention an atomic step applies
almost no item updates, so an applied-count heuristic reclassifies exactly the
population under study.

| shape | arm | n | med ms | p95 ms | max ms | Σ set_image ms |
|---|---|---|---|---|---|---|
| `atomic_fast` | control (retention off) | 10 | **534.1** | 714.9 | 714.9 | 5 056.9 |
| `atomic_fast` | retention on | 10 | **51.9** | 391.5 | 391.5 | **518.0** |
| governed / other | control | 126 | 37.0 | 63.6 | 716.6 | 3 848.7 |
| governed / other | retention on | 124 | 29.4 | 63.1 | 165.9 | 2 908.6 |

The control arm's atomic median (534 ms, max 715) sits just under the 2026-07-31
published band for the same shape (585–867 ms per callback) — close enough that
it is the same population, not a new one. **10.3× at the median** on it, and
**9.8× on its uploads** — the second win, from item residency matching again.

Whole-run GUI-thread totals, and the phase that contains the population:

| | control | retention on |
|---|---|---|
| `resolve` (page assembly) | 6 529 ms | **2 017 ms** |
| `item_loop` | 1 683 ms | **801 ms** |
| payloads resolved / deferred | 2 898 / 1 246 | 2 327 / 714 |
| `montage_scroll_fft` wall | 46.7 s | **39.3 s** |

Two sequential processes, so the wall-clock row is the weakest evidence here
(this box has inverted a sign across processes before); it is included because
it agrees with the in-process ABBA bench of §5a and with the counts, which are
exact rather than timed.

Mailbox and retention counters at the end of the run:

| store | hits | misses | evicted | resident |
|---|---|---|---|---|
| retention (fix arm) | 732 | 1 595 | 683 | 198 entries / 64 MiB |
| mailbox (fix arm) | 217 | 664 | 115 | — |
| mailbox (control arm) | 286 | 1 366 | 66 | — |
| mailbox, 2026-07-31 | 94 | 765 (+179 stale) | 0 | — |

The fix arm's mailbox takes *fewer* hits than the control's because retention
answers first; what matters is that both are far above the 94/765 baseline and
that no buffer is refused for being addressed to the wrong slot any more.

**The 64 MiB retention bound is binding.** 683 evictions against 881 stores:
the 272-tile fills evict each other, because the bound is sized for a scroll
grid rather than a whole field (a field's worth of assembled FFT pixels is
~272 MB). §7 records what a larger bound bought.

### 5c. A note on the vehicle

`montage_scroll_fft` aborts on this machine at
`_wait_for_coverage_pass_close` (10 s, `coverage_open=True`) — 2/2 on unmodified
main in this session, consistent with §6 of the 2026-07-31 dossier. Every arm
here runs through a wrapper that relaxes that one journey-coverage oracle to a
warning; nothing else about the workload is changed, and the relaxation is
identical in every arm. The phase then completes (39.3 s), which is how the
`atomic_fast` population above was reachable at all.

## 6. What changed in the contracts

- `PreparedUploadMailbox` is keyed by identity. `publish(key, …)`,
  `take(key)`, `holds(key)`, `discard(key)`; `slot` survives only on
  `plan(slot, key)`, where the scheduler needs it for supersession and
  `tile_number`. The `stale` outcome is **retired**, not renamed: under content
  keying a buffer prepared for other pixels is not refused, it is simply still
  there under its own key. The buffer accounting equation loses that term.
- `misses` now counts only commits that really packed inline.
- `AssemblyRetention` is a separate store on purpose. The mailbox answers "did a
  worker finish this before the commit asked?" and must drop what it hands over;
  retention answers "have we ever built these exact pixels?" and must not. The
  two also want opposite eviction priorities — a consumed entry is retention's
  most valuable and the mailbox's least — so one structure could not serve both.
- Retention stores the *derived field delta*, not the assembled payload: that
  payload names the slot it was built for, and the point is to serve a different
  one. The delta is computed by object identity (`dataclasses.replace` copies
  untouched fields by reference), which is what keeps `tile_number` out of it.

## 7. Sizing the retention bound — 64 MiB is enough, and the evictions are fine

683 evictions looked like thrash worth fixing. It is not. Same run, same
binary, `ARRAYSCOPE_PYQTGRAPH_ASSEMBLY_RETENTION_MB=256`:

| bound | hits | misses | evicted | resident | atomic med | governed med |
|---|---|---|---|---|---|---|
| 64 MiB (default) | 732 | 1 595 | 683 | 198 entries / 64 MiB | 51.9 ms | 29.4 ms |
| 256 MiB | 728 | 2 003 | 129 | 655 entries / 256 MiB | 58.5 ms | 28.4 ms |

**4× the residency buys zero hits** (728 vs 732) and no timing difference the
run-to-run spread can resolve. The evicted entries are ones nothing asks for
again: a fill walks each tile once, so its assemblies have no second reader,
while the scroll grid — which does re-ask, every step — fits comfortably. The
default stays 64 MiB. Recorded because "reduce the eviction count" is an
inviting next step and it is measurably not one.

## 8. Queue row 1: the ~10× allowance is retired

The 2026-07-31 dossier's §7 asked for row 1 to read "PyQtGraph at 2× on scalar
montages; FFT/complex montages need ~10× at field size, and the deficit is page
assembly, not Qt", with the projection that fixing these hand-offs would make 2×
defensible. Measured, same machine, same dataset, same stages, one run per arm:

`montage_present_total` callback wall time over every commit of the run (272-tile
fills plus the 60-tile LOD-paced FFT scroll):

| arm | n | med | p95 | max | >50 ms |
|---|---|---|---|---|---|
| pyqtgraph, retention off | 136 | 37.7 | 470.5 | 716.6 | 14 % |
| pyqtgraph, retention on | 134 | **30.1** | **63.3** | 391.5 | 19 % |
| wgpu | 121 | 26.6 | 85.0 | 563.6 | 18 % |

And the scroll-step population specifically, which is where the ~10× lived:

| arm | shape | n | med ms |
|---|---|---|---|
| pyqtgraph, retention off | `atomic_fast` | 10 | 534.1 |
| pyqtgraph, retention on | `atomic_fast` | 10 | **51.9** |
| wgpu | `atomic_reuse` | 8 | 84.0 |
| wgpu | `atomic_fast` | 2 | 138.9 |

Phase wall time: fft fill 5.1 s (pyqtgraph) vs 6.1 s (wgpu); `montage_scroll_fft`
39.3 s vs 35.9 s.

**On this workload PyQtGraph is now within 1.13× of wgpu at the commit-callback
median, better than wgpu at p95 (0.74×), and faster than wgpu on the paced FFT
scroll step itself.** The FFT/complex deficit that motivated the ~10× wording
was page assembly, and it is gone. Proposed row-1 wording: **"PyQtGraph at 2×
allowance"** stands as written, with no FFT/complex exception.

Caveats, stated rather than buried: one run per backend, one dataset, one
machine, sequential processes; the fills and the scroll are mixed in the
callback population; and **the 50 ms callback bar is still red on both
backends** (19 % / 18 % of commits over 50 ms), so this closes an allowance
question, not the bar.

Also worth recording for whoever reads `atomic_reuse=0` next: the flag fires 8
of 10 scroll steps on **wgpu** and costs 84 ms/step there, while pyqtgraph
never fires it and costs 52 ms/step. Whatever `atomic_reuse` is worth, it was
never this.

## 9. Gate evidence

`pytest` (affected set) is green. `pytest --since` reports 3 failures, all in
`tests/ui`, all passing in a smaller grouping — the documented co-scheduling
shape.

The control that settles it is `tests/ui` run whole on both trees, same venv,
same box:

| tree | reds |
|---|---|
| branch, run 1 | `settle_identical_levels`, `wgpu_pan_away_and_back`, `settled_montage…[wgpu]`, `preview_transition[scalar]` |
| branch, run 2 | `settle_identical_levels` |
| main (`24d2f77f`) | `settle_identical_levels` |

Branch run 2 is byte-for-byte main's red set, so the three extras are the
co-scheduling flakes and not a branch delta.
`test_rebind_and_evaluation_paths_settle_identical_levels` is red on **main** in
the same grouping — the pre-existing R3 presented-bounds clamp latch
(`r3-clamp-stale-true-bounds`), not reachable from this change.

Two things checked by hand because the change makes pixel arrays outlive the
commit that built them:

- **Nothing mutates a retained array.** `_set_tile_data` and
  `_update_tile_levels` only read it; `rgb_display_for_levels` and the fused
  `_rgb_window_kernel` both allocate their output. `state.rgb_base` is an alias
  of the retained array and always has been, across level changes — so the
  immutability this store assumes was already load-bearing.
- **The retention key cannot outlive its content.** It carries
  `document_generation`, `semantic_generation` and `operation_key` (via
  `TileIdentity`), the resolved page set, and the exact CPU mapping including
  levels. A data change, a refinement or a level change all mint a new key, so a
  stale entry is unreachable rather than wrong, and is evicted on budget.

## Repro

```
ARRAYSCOPE_UI_OBSERVATION_LOG=<scratch>/obs.jsonl \
tools/env python -m arrayscope.tools.headless_display -- \
  python -m arrayscope.tools.profile_montage_workflow --backend pyqtgraph \
  --scroll-max-tiles 60 \
  --stages load_data,raw_full_tiled_montage,fft_full_tiled_montage,montage_scroll_fft \
  --jsonl <scratch>/run.jsonl
```

Kill switches for bisection: `ARRAYSCOPE_PYQTGRAPH_ASSEMBLY_RETENTION=0`
(restores per-commit re-derivation), `ARRAYSCOPE_PYQTGRAPH_RESOLVE_BUDGET=0`
(restores eager whole-batch assembly, 2026-07-31),
`ARRAYSCOPE_PYQTGRAPH_ASSEMBLY_RETENTION_MB=<n>` to size the bound.
