# WGPU bookkeeping vs residency, re-measured after retention (2026-07-31)

The suspicion: several executor bookkeeping costs scale with **total
residency** rather than with what changed, and nobody had re-measured them
since [out-of-view retention](out-of-view-retention-2026-07-30.md)
deliberately raised the resident/referenced working set. Six sites were
named from a read of `wgpu_executor.py` / `page_table.py` /
`wgpu_imageview2d.py`.

**Finding: five of the six are noise at post-retention residency, and the
sixth is small.** The only residency-proportional per-upload term that
measured at all is eviction's victim query (a whole-table sort per pressured
upload, ~18 µs at 700 resident pages), and pool pressure itself never arises
in the post-retention 272-tile regime — the pool kept 240 free layers.
That query is now a single-pass min-scan; everything else is deliberately
untouched, with the numbers below as the reason.

## 1. How it was measured

Three probes (scratch pytest modules, deleted after use; no production
instrumentation added — the pre-existing counters are the instruments):

- **App probe**, the regime the chip demanded: offscreen WGPU, 272-source
  montage (17×16, 96 px, per-source gradients, payloads from the real
  pipeline so `TileIdentity` binding reuse is live). Settle, zoom to ~12
  visible tiles, then three pan cycles home → away-x → home → away-y → home,
  then five no-camera-change commits. Executor and page-table class methods
  wrapped with accumulating counters/timers so executor rebuilds don't lose
  instrumentation.
- **Executor churn probe**: a real `WgpuPlaneExecutor` on the shared Vulkan
  device, one 8192² plane, pool filled to capacity (swept 128 / 512 / 1000
  layers), then sustained new-key uploads so **every upload must evict
  first** — the regime the app only reaches on stacks larger than the
  residency budget.
- **Interleaved in-process A/B** for the one change: one process, one
  executor at a full 700-layer pool, arms ABBA/BAAB in 25-upload batches
  (16 samples per arm), plus a B-vs-B control with identical code (8+8) to
  bound the noise floor. Sequential single-shot processes have inverted the
  sign in this repo four times; this comparison never crosses a process
  boundary.

Machine caveat: the runner reported ~4–8 of 16 cores loaded from elsewhere
during these runs. Conclusions rest on counters and on the controlled A/B,
not on raw wall-clock.

## 2. The post-retention regime, measured

At rest after the fill: **312 resident pages, 82 MB, 552 layers allocated
of a 610-layer budget (later raised to 1024 in place, without a copy),
240 layers free.** Whole-session commit classes over settle + 3 pan cycles +
5 no-op commits: 3 full republications, 20 incremental, 13 mapping-only
fast-path.

The load-bearing observation: **once settled, a pan leg costs exactly one
mapping-only fast-path commit and nothing else.** Cycles 1–2 and the no-op
commits show zero table flushes, zero plane binds, zero resolves, zero
uploads, zero pin replacements — the steady state is already O(1) per
commit, at the residency retention now produces. The residency-scaling
sites simply do not run at 60 Hz cadence; they run during fills and
structural changes, where the real work (texture uploads, whole-set
republication) dominates them anyway.

## 3. Verdict per named site

| # | site | measured | verdict |
|---|---|---|---|
| 1 | `_flush_table` rewrites the whole flat buffer | 23 dirty writes in the whole app session, 67 KB cumulative, **0.5 ms total**; the table is 1198–1360 entries ≈ 5 KB (a 96² plane contributes ~5 entries; even one 8192² pyramid is 1365 entries ≈ 5.5 KB, flushed in 0.02–0.03 ms) | **free** — one small write per dirty submit; the per-commit dossier's lesson (transactions, not bytes) holds: this is 1 transaction |
| 2 | `generation` bump clears the binding cache and disqualifies the commit fast path | 477 cache misses / 98 hits across the whole session, ~µs each; at steady state the generation is stable and the fast path fired on **every** leg | **free in practice** — uploads only happen inside commits that republish anyway; splitting the generation (bind-only vs unbind) would be correctness-risky (`_wgpu_payload_binding` legitimately upgrades to *newly* resident predecessor/native pages, so new binds must invalidate too) for no measured win |
| 3 | `_grow_pool` copies every layer; ceilings monotone | **0 grows in the app probe** (pre-reservation sized 552 layers up front); worst case from scratch: 1000 uploads → 7 doublings, 266 MB copied, 88.7 ms total = **0.09 ms/upload amortized** | **amortized free** — the doubling shape is fine. The monotone ceiling is a *memory* concern, not a time concern: a representation switch strands `allocated_layers` × 256 KB (552 layers = 138 MB). Named, untouched |
| 4 | `_bind_planes` re-scatters every resident key on non-prefix change | 8 full rebuilds in the whole session, ~7 ms each at 270 planes / 312 pages; 15 prefix appends ~2 ms; **zero at steady state** | **not the lever** — a full rebuild only happens on a structural change that already republishes the whole presented set |
| 5 | `resolve` does not `touch` → sampled pages age like unused ones | resolve costs 1.2–1.6 µs/call and is silent at steady state; the mis-aging can only change outcomes under pool pressure, which this regime never reaches — and drawn/committed pages are pinned, so the victim set excludes what is being sampled by the presented tiles | **named, untouched** — if a big-stack regime shows pages evicted then re-ensured within a few frames (the honest instrument: per-key re-upload counts, not apply-ms), this is where to look |
| 6 | eviction victim query sorts the whole table per pressured upload | churn upload (evict+bind+write) 0.26–0.65 ms; A/B at 700 resident: sorted **0.416** vs min-scan **0.398** ms/upload median, control split 0.005 | **the one real term — fixed**, see §4; small (~4% of a pressured upload) but the wrong shape, O(R log R) per upload |

Also observed, not on the chip's list: `replace_pin_set` stayed O(delta)
after its earlier fix (223 calls / 44 ms across the session, concentrated in
settle phases), and eviction never fired in the app probe at all —
`_ensure_free_pool_layer` is the only eviction route and free layers never
ran out.

## 4. The change

`PageTable.eviction_candidate(match)` — a single pass over `_entries` taking
the minimum `last_use` among unpinned entries satisfying `match(key, slot)` —
and `_evict_one_unpinned` now asks it instead of building the fully sorted
candidate tuple and scanning for the first match. The victim is identical by
construction: `last_use` stamps are unique and monotone, so "first matching
entry of the last_use-ascending order" *is* "matching entry with minimal
stamp". `eviction_candidates()` (the sorted tuple) remains for diagnostics
and tests, and a unit test pins the two against each other, filtered and
unfiltered.

Interleaved order-balanced A/B, one process, full 700-layer pool,
ms/upload medians (full value lists in the run log shape, n=16 per arm):

| arm | median | spread |
|---|---|---|
| sorted (old) | 0.416 | 0.397–0.475 |
| min-scan (new) | 0.398 | 0.364–0.490 |
| control A = control B | 0.381 / 0.376 | n=8 each |

An earlier isolated timing of `eviction_candidates()` read 0.12 ms/call at
1000 entries; the controlled A/B shows the real in-path delta is ~18 µs —
the isolated number was inflated by tuple construction and machine load.
Recorded so the next reader doesn't repeat the overstatement.

## 5. Correctness evidence

- The changed lines are reached by 3 recorded tests plus the 11
  eviction/residency protocol tests (`tests/gpu/test_wgpu_command_protocol.py`,
  which pin the LRU victim choice and pixel behavior under pool pressure) —
  all green, serially.
- testmon proves the UI retention pixel test
  (`tests/ui/test_montage_out_of_view_retention.py`, framebuffer reference
  oracle at the 272-tile regime) does **not** execute the changed lines —
  which is itself the §2 finding: post-retention panning never evicts.
- New unit: `test_single_candidate_agrees_with_the_sorted_order` +
  `eviction_candidate() is None` on a fully pinned table.
- `pytest --since`: 7 tests reached by the branch, all green.

## 6. Deliberately not touched

Subrange/dirty-range table writes (site 1), a split bind/unbind generation
(site 2), pool shrink or destruction on representation switch (site 3),
incremental plane re-scatter (site 4), touch-on-resolve (site 5): every one
of them is a real design possible, and every one of them would today be an
optimisation of a cost measured in microseconds or in events-per-structural-
change, purchased with correctness risk in exactly the machinery (bindings,
epochs, LRU policy) whose regressions this repo has repeatedly shown to be
silent. The stranded-pool memory figure (138 MB for a 552-layer scalar pool
after a representation switch) is the one number here that could justify
future work, and it is a memory ceiling question, not a 60 Hz question.
