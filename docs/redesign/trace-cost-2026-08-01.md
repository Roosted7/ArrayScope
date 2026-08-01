# What the montage trace bus actually costs, and which levers are real (2026-08-01)

**Status:** measured, and a content-preserving cut landed. The two levers the
2026-07-31 dossier named and left (§1) are both **refused here on evidence**;
what replaced them is a ~23% cut to the montage trace tax with byte-identical
output (§4). The residual is the honest price of the flight recorder (§6).

Closes the `I3` line from
[gui-callback-owners-2026-07-31.md](gui-callback-owners-2026-07-31.md) §6, and
answers the lazy-store half of the question
[index-window-retarget-cost-2026-07-25.md](index-window-retarget-cost-2026-07-25.md)
left when it adjudicated the emit-time encode.

## 1. The starting position

`frame_runtime.py:538` arms a ring-only bus when the montage watchdog arms, so
`TraceBus._enabled` is True for every montage session in normal production.
The 2026-07-31 attempt added `if trace_enabled():` guards to hot emit sites
(`39f488f0`) and then counted the branch: 31 True / 0 False at `commit_batch`,
7042 True / 2 False in `tile_lifecycle` — a measured no-op. `330d4eeb` then
added an unconditional per-step `rung_prepare` emit to the same path.

Two levers were named and left:

- **(a)** do not arm the ring by default from the watchdog;
- **(b)** make the ring store lazily, so kwargs are never materialized until
  an event is read.

## 2. Reconfirmed, then both levers refused

**The guard is dead, confirmed eight times.** Every montage session measured
here (272-tile NIfTI, wgpu, `load_data,raw_full_tiled_montage,montage_scroll_scalar`
at `--scroll-max-tiles 272`, headless Weston) counted the guard sites:

| arm | `tile_lifecycle.trace_enabled` | `frame_effects.trace_enabled` |
|---|---|---|
| eight sessions | 8093–11309 True / **0 False** | 105–149 True / **0 False** |

**(a) is refused because the production ring is genuinely read, per-tile rows
included.** The stall dump is not a theoretical artifact: three shipped
regression tests were authored from field dumps — `tests/ui/test_montage_scroll_settling.py`
(`arrayscope-stall-18-1` seq 9703-9708, `arrayscope-stall-65-2` seq
37576-37585) and `tests/ui/test_dimension_control_interactions.py`
(`arrayscope-stall-78-1`, `arrayscope-stall-445-6/7`). The decode in
[coverage-stall-2026-07-15.md](coverage-stall-2026-07-15.md) rests on **"all
3,231 lifecycle `presented` edges"** and on `commit_batch`/kernel ordering.
Suppressing the high-rate kinds from the default ring — the only version of
(a) that keeps a flight recorder at all — would have removed exactly the rows
those diagnoses used.

The tempting supporting argument for suppression does not exist either. A
sliding-window analysis over the recorded stream says an 8192-event ring holds
**20.4 / 23.4 / 29.5 s (min/mean/max)** of history even across the busiest 1 s
buckets (869 events/s). Against a 2 s stall threshold the recorder is not
being crowded out by the hot kinds, so there is no "they destroy the evidence"
case to make.

**(b) is refused because the encode is already deferred and the rest is
unsound.** The ring has stored raw dicts and encoded at `dump` since
2026-07-25; that half of (b) landed then. The only laziness left is deferring
*field construction*, which requires handing the ring live objects and reading
them seconds to minutes later, after the session has moved on. A flight
recorder that reports post-hoc state is worse than none. That is the contract
question the index-scroll memo declined to answer; the answer is no.

## 3. What the cost actually is

In-path accumulators at each site, one live session (medians; means are
GC-tail-inflated and are not the useful statistic here):

| site | n / session | median | what it is |
|---|---:|---:|---|
| `commit_batch` (backend_complete) | 105–149 | **~700 µs** | one 107-key record per commit |
| `lifecycle` | ~5160 | ~15 µs | the app's highest-rate emitter |
| `rung_prepare` | ~1500 | ~8.5 µs | per rung step |
| `backend_ack` (presented) | 664 | ~16 µs | |
| **total** | | **~310 ms** | over a ~46 s session |

Two things this changed about the earlier picture:

1. **The `~1 ms per whole-montage commit` is real and it is the commit record**,
   not the per-tile rows: ~700 µs median in one emit, 100% of it argument
   construction (the bus itself is 8 µs for that event).
2. **The distribution is bimodal.** `commit_batch` runs median 700 µs but p90
   3.6 ms and max 10.5 ms. Those tails are garbage collection landing on trace
   allocations: 700–730 gen-0 collections and 10 gen-1 per session, 230–360 ms
   of pause time total. Anyone timing this path with a mean is measuring the
   process's GC, not the emitter — which is why every number here is a median.

Bisecting the commit record by kwarg group found the single largest
identifiable term: **`session.required_tile_numbers()` was called three times
inside one trace record** — once to bind `required_tiles`, then again for
`required_tile_count` and again inside `preview_missing_tile_count`. It
rebuilds two sets and a sorted tuple over the whole required scope every time:
**31–37 µs per call, measured in-session** (3797 calls in a baseline session,
3426 after).

## 4. The lever that is real, and what it bought

Not "stop recording" and not "record later" — **stop re-deriving, and stop
re-hashing what is already assembled**. Every change below is byte-identical
in output:

1. `TraceBus.emit_fields(kind, mapping)` — an entry point for emitters that
   *build* their fields. `emit()` routes through it, so there is one funnel.
2. The `{str(key): value …}` pass in `emit` is gone. Keyword emission cannot
   produce a non-string key; the coercion ran over every field of every event
   to defend against a caller that cannot exist.
3. `_trace_lifecycle` assembles one mapping in emission order instead of
   kwargs → comprehension → splat → kwargs → event (four dicts, every field
   hashed four times, to record it once).
4. `prepare_rung` hands its mapping over whole.
5. The commit record uses the `required_tiles` it already bound.

**Results.** Eight sessions, alternating arms, same instrumentation on both
(medians, µs):

| site | A (before) ×4 | B (after) ×4 | separation |
|---|---|---|---|
| `lifecycle` | 13.2, 15.7, 16.0, 15.8 | 11.2, 9.3, 8.9, 9.5 | **8/8, −36%** |
| `rung_prepare` | 9.0, 10.3, 7.3, 7.4 | 3.9, 3.1, 3.2, 4.2 | **8/8, −58%** |
| whole-session trace total (ms) | 278, 317, 312, 337 | 263, 249, 245, 256 | **8/8, −23%** |
| `commit_batch` | 702, 651, 863, 760 | 995, 475, 339, 682 | **not separable** |

`commit_batch` is chasing ~80 µs inside a 700 µs median whose p90 is 5× the
median; the end-to-end comparison cannot resolve it, exactly as the
benchmarking memo warns. Its evidence is the **direct work counter** instead:
three whole-scope rebuilds per record became one, at 31–37 µs each, plus one
redundant `int()` per required tile — 244–298 fewer whole-scope rebuilds per
session.

**Output equivalence.** Trace JSONL captured from both arms: 29 of 29 shared
event kinds have identical field-name sets (the three unshared kinds are
condition-triggered and differ by workload, not code). `commit_batch`
backend_complete: 107 keys, identical set *and* identical order.
`lifecycle`: 15 keys, identical order. `rung_prepare`: 11 keys, identical order.

## 5. Traps this run hit (all measurement, none in the product)

- **An asymmetric probe moved an untouched control by 63%.** The first A/B
  wrapped `TraceBus.emit`; after the change the re-routed emitters no longer
  passed through it, so arm B carried less instrumentation on the same events.
  The tell was `backend_ack_presented` — code neither arm touched — moving
  +63% in both B arms. Wrapping the single funnel every event passes through
  in *each* arm fixed it. Without a same-code control site this would have
  been published as a commit-path regression.
- **Absolute per-call medians carry ±30–60% between processes on this box.**
  Only within-arm-consistent, all-runs-separated differences are claimed
  above; anything that overlaps is reported as not separable.
- **A cost mean here is a GC measurement.** See §3.
- `raw_full_tiled_montage` FAILs its 5 s phase gate on this machine under
  load (5.9 s elapsed) on both arms — environment, not the change.

## 6. What is left, and why it stays

~250 ms per session, ~0.6 ms per whole-montage commit, ~1% of that callback.
It buys the only production diagnostic in this codebase that has repeatedly
paid for itself. The remaining terms are irreducible without giving something
up:

- the commit record is 107 fields, several genuinely O(visible tiles)
  (`presented_tiles`, `delta_qualities`, sorted rejection sets) — real values,
  each read by a replay oracle;
- ~5200 lifecycle edges at ~10 µs is close to CPython's floor for a 15-field
  dict plus a bounded identity summary;
- the GC pressure is the ring doing its job: 8192 retained event dicts is a
  large permanently-live tracked set by construction.

If this is ever revisited, the question to ask is **not** "can the guard be
made to fire" (it cannot) but "is any *emitted field* unread by every oracle",
which is a content question with an owner, and needs `trace_verify` /
`trace_latency` consulted first — not a performance question.

## Repro

Eight `profile_montage_workflow` sessions on 2026-08-01, branch
`claude/montage-watchdog-trace-ring-fe5509`, headless Weston:

```
env -u WAYLAND_DISPLAY tools/env python -m arrayscope.tools.headless_display -- \
  python -m arrayscope.tools.profile_montage_workflow --backend wgpu \
  --stages load_data,raw_full_tiled_montage,montage_scroll_scalar \
  --scroll-max-tiles 272 --trace <out>.trace.jsonl
```

Per-site medians came from temporary `perf_counter_ns` accumulators at the four
emit sites plus `FrameSession.required_tile_numbers`, and per-kind counts and
GC pauses from a wrapper around the bus funnel and `gc.callbacks`. Neither is
committed; raw JSONL stayed in the session scratchpad per the no-artifacts rule.
