# Index-window retarget cost — anatomy and the delta hypothesis (2026-07-25)

**Status:** measured, two redundancies removed and gated; the rest is
characterised below and deliberately not touched.

The 2026-07-24 perf tier profiled a montage index-window scrub and named
`FrameSession.retarget_index_window`'s model remap as the dominant remaining
synchronous main-thread cost per scroll step (30–54 ms of a ~70–95 ms step,
100 tiles). This is the follow-up: a finer breakdown, what was redundant,
and what is inherent.

## Repro

Offscreen, wgpu, `montage_quality_policy=resident`, a `(336, 336, 272)`
float32 volume, a 100-tile montage (10 columns) in a 1200x1000 window,
scrubbed one index at a time and pumped to first pixels between steps.
`retarget_index_window` is timed directly; the trace bus is left in its
production state — `_ensure_montage_watchdog` arms an 8 MB ring on every
montage session, so lifecycle edges really are encoded during a scrub.

Baseline on that shape: **29.1 ms mean, 30.5 p50, 46.2 worst** — the field's
30–54 ms slice, reproduced.

## The delta hypothesis is wrong

The tier's proposed fix was to make the remap proportional to the delta:
"an index-window shift typically keeps most sources; only entering/leaving
indices change". Measured branch counts for a +1 step over 100 tiles:

```
hits=99  misses=1  unchanged=0  remapped=99
```

**Zero tiles are unchanged.** The hypothesis is true of the *source set* and
false of the *slot set*: montage slot `i` is bound to `window[i]`, so
shifting the window by one moves every source to a different slot. Every
slot therefore owes a genuine lifecycle transition, and the remap is
inherently O(visible tiles). Any delta-shaped saving would require slots to
rotate with the window — a different layout/backend-instance contract, not a
local optimisation.

The redundancy that *does* exist is per-tile constant factor, not tile count.

## Breakdown of the 29.1 ms baseline

Per-callee, instrumented (a pass with the trace bus stubbed out isolates the
observability share):

| segment | ms | calls |
| --- | --- | --- |
| `record_tile_payload` | 11.0 | 198 |
| ↳ of which `lifecycle.fallback_ready` | 9.3 | 198 |
| `mark_ladder_swaps_for_viewport` | 8.2 | 1 |
| `sync_lifecycle_scope` (own + nested) | 8.7 | 1 |
| `_sync_lifecycle_targets` | 4.7 | 2 |
| `tile_target_identity` | 2.6 | 200 |
| `tile_payload_identity` | 2.1 | 99 |
| `lifecycle.retarget` | 1.3 | 1 |
| `_payload_source_anchor` | 1.0 | ~102 |
| trace emission (difference of a trace-off pass) | 7.9 | 298 events |

Two numbers are the whole story: **198 payload reports for 99 remapped
tiles**, and **2 target builds for 1 target set**.

## What was removed

- `perf(render): report a remapped montage payload once, not twice` — the
  retarget installs and reports each remapped payload, and
  `sync_lifecycle_scope`'s safety-net scan then reported all of them again
  because its memo was keyed by `id(payload)` under `id(container)` and the
  retarget had just replaced every payload object. The mutation site now
  primes the memo. Soundness rests on `TileLifecycle.retarget` returning the
  tiles whose `presentable_payloads` it pruned: a prime that survived a
  target adoption would suppress the report that puts the payload back, and
  the tile would lose its first-pixel fallback. The memo is also pinned to
  its mapping object rather than keyed by `id()`, which both stops unbounded
  growth and closes a recycled-`id()` suppression hazard.
- `perf(render): build the retarget's lifecycle targets once, not twice` —
  targets are published before the per-tile remap (a cache hit entering
  `mark_materialized` must find the successor target in place) and were
  rebuilt from scratch by the closing `sync_lifecycle_scope`. The remap
  publishes no new targets; its only reach into a target input is
  `skipped_tiles.discard`. The published set is now handed over behind an
  identity-based guard that declines to a recompute on any doubt.

Interleaved in-process A/B (variants switched per step, round-robin, so
machine drift lands on each equally; three runs of 45–60 samples per
variant):

| variant | min | p50 | mean |
| --- | --- | --- | --- |
| baseline | 24.7–29.7 | 32.8–52.9 | 34.3–53.2 |
| + payload dedupe | 19.5–23.8 | 24.4–39.6 | 26.9–43.8 |
| + target reuse | 18.5–22.0 | 25.6–39.3 | 26.8–39.4 |

**≈ −25%** on the retarget (~7 ms/step at the 100-tile field shape). Counted
work, which does not drift: `record_tile_payload` 198 → 99, payload-ref
normalizations 495 → 297, lifecycle trace edges 298 → 199, target identity
constructions 200 → 100.

The target reuse is worth ~0.4–2.5 ms and sits at this machine's noise floor
at p50. It is kept for the countable work it removes, not for a p50 claim.

## What is left, and why it was not touched

- **Trace encoding, ~5 ms.** `TraceBus.emit` calls `json.dumps` on every
  event even when no JSONL sink is configured — which is the default
  production state, since the watchdog arms a ring-only bus. The encode
  exists solely to size the ring for its byte budget. Deferring it would
  need the ring bound to become approximate, which is an observability
  contract change in a different subsystem; it should be decided on its own
  merits, not smuggled in behind a render perf fix. It is the largest single
  remaining item and it is not specific to the retarget.
- **`mark_ladder_swaps_for_viewport`, ~8 ms.** After the remap, 99 slots have
  no `RenderedTile`, so the ladder pass runs `floor_can_progress` — a real
  pyramid query per tile — for each. That is LOD ladder work, not model
  remap, and it decides visible quality.
- **Per-tile derivations, ~3 ms.** `tile_payload_identity` and
  `_payload_source_anchor` are keyed by source index, so every tile is a
  distinct derivation with nothing to share. Inherent at this slot contract.
- **`_apply_backend_tiled_presentation`, ~25 ms** (field measurement) is a
  separate concern and out of scope here.

## Gates

`tests/window/test_montage_lod_residency.py`,
`tests/window/test_montage_backend.py`,
`tests/ui/test_montage_scroll_settling.py`,
`tests/window/test_presentation.py`,
`tests/ui/test_scrub_presentation_retention.py`, `tests/presentation/`.

New regression nets: `retarget` reports the tiles whose presentable history
it pruned (`tests/presentation/test_tile_lifecycle_transactions.py`); the
remap reports each remapped payload once, reports again when the targets
move underneath the prime, and publishes its targets once
(`tests/window/test_montage_lod_residency.py`).
