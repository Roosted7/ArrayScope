# A settled montage that never stopped owing itself work

**Subject.** Chased down as "`pending_removals` does not drain to empty on a
settled montage (248 standing on PyQtGraph), which is why prefetch's `_busy`
predicate never goes quiet on a large montage".

**Verdict.** The premise is wrong in both halves, and a different, real defect
sits underneath it. `pending_removals` drains. `_busy` does stay true forever —
held by `dirty_payloads`, and by a self-sustaining commit loop, not by a removal
backlog.

Regime throughout: 272 tiles at 32 px, nine tiles visible (28 in the visible
margin), one slice step through `retarget_index_window`. Both backends.

---

## 1. What actually stands

Measured at main tip, per phase, with an idle wait between phases so the
event loop is genuinely quiet:

| after a data change, settled + 6 s idle | `pending_removals` | deferred queue | `dirty_payloads` | `_busy` |
|---|---:|---:|---:|---|
| PyQtGraph | **0** | **0** | **13** | true |
| WGPU | **0** | **0** | **13** | true |

`pending_removals` converges. So does `deferred_out_of_view_invalidations`
(244 → 0). The "248" in the original note is the size of the *deferred* queue
mid-drain, not `pending_removals`; the settle-owner drain added in `b0bfabf2`
is what made it converge, so the number was already historical when it was
quoted.

Removing `pending_removals` from `_busy` would therefore have changed nothing
at all, while hiding a queue that is in fact healthy.

## 2. Which entries stand, and why they cannot drain

All thirteen are the same shape on both backends, and they are the same
thirteen tiles:

* in `visible_tile_numbers` (the `VISIBLE_MARGIN_TILES = 1` ring) but **not** in
  `required_tile_numbers()` — so no owner ever evaluates them;
* holding a `display_tile_payloads` entry whose `semantic_generation` names the
  **previous slice** — `(16, 16, 2, 1)` against a target of `(16, 16, 2, 0)`;
* `payload_is_current` false for exactly that reason.

The loop is in `build_tile_presentation`'s reconcile pass. For a dirty slot it
asks the lifecycle for the best presentable payload; outside the required scope
the lifecycle has no target payload and never will, so it returns the
pre-change fallback. Binding it marked the tile *materialized*, which shielded
it from the drop pass immediately below; the wrapper's own staleness kept it out
of `upserts`, so it was never proposed, never acknowledged, and never reached
the parking rule that retires unaccepted upserts; and its dirty entry survived
to be reconciled again by the next commit.

Instrumented directly: **186 rebinds of one tile across three seconds of
idle**. `final_commit_pending` and `flush_pending` stay true throughout, so a
montage the user sees as finished is committing forever.

## 3. `_busy` was not the gate

The chip's model was that `_busy` stays true, so
`schedule_near_viewport_montage_prefetch` reports `blocked_visible_busy` and
speculation is gated off. Measured in that state:

| after a data change, 6 s idle | prefetch invitations | `blocked_visible_busy` recorded |
|---|---:|---:|
| PyQtGraph | **0** | 0 |
| WGPU | **0** | 0 |

Prefetch is not invited at all, so `_busy` is never consulted and
`blocked_visible_busy` is never recorded once. The gate was not load-bearing;
it was not reached.

## 4. The change

`FrameSession._superseded_fallback_is_final`. `payload_compatible_with_tile` is
already the standing owner of "may this wrapper be bound to this tile"
(ADR 0050), and its answer for these payloads was already no. What it does not
settle is how long to wait for a better one, and the required scope is what
answers that: inside it, "no" is a wait (an owner is evaluating the successor
and this wrapper is the floor it stands on); outside it, "no" is final.

Refusing the bind and leaving the tile out of `materialized` hands it to the
existing drop pass — the one owner of "this slot holds nothing worth keeping".
No new drain path, no new flag.

The required-scope half is load-bearing and was not optional: without it,
`test_cropped_display_axis_scroll_keeps_complete_montage` and
`test_indexed_dim_scroll_settle_leaves_committed_frame_current` both fail with
the montage stranded — 37 of 50 tiles unsettled, physically present, nothing
dirty and nothing pending. Those scrolls settle *through* that floor.

### After

| after a data change, settled + idle | `pending_removals` | deferred | `dirty_payloads` | `_busy` | prefetch invitations |
|---|---:|---:|---:|---|---:|
| PyQtGraph | 0 | 0 | **0** | **false** | **~20/turn, 400–3400 decisions** |
| WGPU | 0 | 0 | **0** | **false** | **~25/turn, 550–4400 decisions** |

So prefetch did come unblocked — but by the commit loop ending, not by relaxing
a gate.

## 5. What unblocking cost the visible path — nothing measurable

Interleaved in-process A/B, one process, one window, arms alternated by
patching the predicate, arm order mirrored between passes, both arms measuring
the **same** pan leg from the same home viewport. n = 8 per arm per run, two
runs, plus a **control run in which both arms are the identical code** — the
harness's own asymmetry, which is what any claimed effect has to beat.

| metric | control (on/off) | run 1 | run 2 |
|---|---|---|---|
| PyQtGraph first-pixel ms (median) | 73.7 / 76.6 | 78.0 / 73.7 | 95.5 / 99.4 |
| PyQtGraph commits per pan leg | 3.5 / 3.0 | 3.0 / 2.0 | 3.0 / 2.5 |
| PyQtGraph upserts per pan leg | 13.5 / 13.0 | 13.5 / 13.0 | 13.5 / 13.0 |
| WGPU first-pixel ms (median) | 80.3 / 77.9 | 93.6 / 86.8 | 127.9 / 122.6 |
| WGPU republications per pan leg | 1.0 / 1.0 | 0.0 / 1.0 | 0.0 / 1.0 |
| WGPU commits per pan leg | 2.5 / 1.0 | 2.5 / 1.0 | 2.5 / 1.0 |
| WGPU upserts per pan leg | 15.0 / 12.0 | 15.0 / 12.0 | 15.0 / 12.0 |

Commits and upserts per leg reproduce the control asymmetry **exactly** on both
backends: no effect. First-pixel differences are inside the control's own
asymmetry and are swamped by run-to-run drift (the same arm moves 74 → 96 ms
between runs). WGPU full republications are the one consistent difference and
they go **down** — 1.0 → 0.0 in both runs against a control of 1.0/1.0, which is
what fewer stale bindings should do.

The 2.8× halo-warming regression (`retention-halo-2026-07-30.md` §3) does not
reappear here because nothing widened the warm set; the prefetch that now runs
is the walk that was always meant to run once the montage was quiet.

## 6. Pinned

`tests/ui/test_montage_settled_quiescence.py`, both backends: after a data
change, `pending_removals`, the deferred queue, `dirty_payloads` and
`pending_payload_upserts` all reach zero *and* `_busy` goes false, waited for
and asserted at the same instant. Its own mutants: the viewport must have a
non-empty margin ring outside the required scope (a montage too small has no
defect to observe), and the data change must have left something owed (or the
drain assertions would pass by draining nothing). With
`_superseded_fallback_is_final` neutered it times out on both backends.

`_busy` is deliberately not asserted at a later arbitrary moment: once the gate
opens, prefetch starts speculative work and honestly reports busy again.
