# The cold-entry stage plan no longer owns the event loop (2026-07-31)

**Status:** landed on this branch. Follow-up to the rank-1 finding of
`docs/redesign/gui-callback-owners-2026-07-31.md` (§1a / §2 S1): on a 272-tile
field cold fill, `win.render()` ran 0.9–1.4 s in ONE GUI callback on both
backends, 97% of it `last_stage_plan_ms`, consulting no `GuiCallbackBudget`
and reporting no `GuiCallbackObservation`.

## What the eater actually was — not what S1 guessed

S1 read the cost as "synchronous 272-tile stage-plan build … nothing
recomputes here; it has simply never been chunked", implicating the
`build_stage_fan_in_plan` slab-planning loop. Measured here with a targeted
profiler around the exact call (a whole-program cProfile run changes the
interaction timing and silently reroutes the branch — the expensive path never
executed under it), the cold entry takes the **interaction fast path**:
`defer_stage_planning` is true during the profiled fill (the session fixture
provides a committed same-axis predecessor and the fit pulse keeps
`_viewport_interaction_active` armed), so the branch taken is
`hot_cached_stage_fan_in_plan` — the path that exists *specifically* to avoid
GUI-thread slab planning.

Inside it, the whole-volume full-stage scan is ~0.2 ms and innocent. The cost
is the **per-tile fallback match loop**: O(missing_tiles × cache_entries)
`region_contains` probes, ~14,550 calls per fill at **240 µs each** — 96% of
that inside `_axis_contains`, which materialized both axis sides as numpy
index arrays and asked `np.isin` (29,100 calls; each pays `arange` +
`zeros_like` + reductions construction overhead on tiny inputs). At 272 tiles:
**2.1 s of on-thread CPU**, plus ~1.4 s of same-window off-CPU wall
(GIL/scheduler contention with the workers the fill just started) inside one
callback. And in every measured fill it matched **zero** tiles — the montage
had no partial-region stage to find; 3.5 s bought an empty dict.

The `np.isin` shape was itself a deliberate fix ("answered in C") for a
*different* regime — once per payload per axis during a crop settle — and was
right there. Nobody re-priced it when the stage-plan loop started asking it
tens of thousands of times per callback.

## What landed

Three commits, in dependency order:

1. **`region_contains` answers arithmetically** (`operations/regions.py`).
   ALL/SLICE/POINT membership runs on `range` progressions — endpoint
   membership plus stride alignment, O(1) per probe — INDICES containers
   through one set build; no numpy arrays. Semantics pinned by a matrix test
   against the original implementation kept verbatim as the oracle, including
   the inherited edges (empty INDICES contained by a SLICE but not a POINT;
   out-of-range points still raise; stride-alignment cases where
   endpoint-only containment would lie). Isolated cost: ~0.26 ms per
   *scanned tile* including the full entry scan (see the
   `montage_stage_plan` feedback row below) vs ~7.8 ms before.

2. **The per-tile fallback runs under the window's GUI-callback budget**
   (`window/frame_effects.py`). Both fix shapes from the task were evaluated;
   this is shape (b), and deliberately not (a): bounded GUI-thread slices
   with a continuation would still burn ~2 s of loop occupancy in 40 × 50 ms
   bites, and the off-thread machinery already exists and is battle-tested —
   `submit_deferred_stage_fan_in_plan` plans on a kernel worker
   (`VISIBLE_PLANNING` at `INTERACTIVE`, supersession-keyed, with the
   phantom-flag/livelock lessons of 2026-07-16 already encoded) and its done
   callback commits the plan on the GUI thread under session-currency guards.
   On budget expiry the hot path returns the deferred placeholder — exactly
   the shape all three call sites already pair with the async planner —
   tagged `hot-stage-budget-deferred`. Partial matches are dropped, not
   attached (attaching would make "no match" ambiguous between "scanned,
   none exists" and "never scanned"); the per-tile match cache keeps what a
   bailed slice learned, so consecutive calls resume ahead instead of
   rescanning tile one. The whole-volume full-stage attach — the case that
   makes FFT scrubs instant — runs before the budget and is untouched.
   Preview-first ordering is untouched: deferred stage planning is the
   *existing* cold-entry semantics for every non-native fill; production
   start waits on the worker plan exactly as it did before, minus the 2 s the
   GUI thread spent computing nothing first.

3. **The entry reports itself** (`window/render.py`). `win.render()` records
   a `render_entry` `GuiCallbackObservation` on every call (including
   exception exits) carrying the reason and the stage-plan / frame-update /
   side-panel split in its details; the bounded match loop reports
   `montage_stage_plan` with its item count. Both now appear in
   `resource_feedback_channels` — the governor's loop closes over them
   instead of seeing an unexplained heartbeat gap.

## Numbers

Order-balanced ABBA across separate invocations (A = main `c1c295e5`
untouched checkout, B = this branch), managed offscreen Weston
(`headless_display`, `WAYLAND_DISPLAY` unset so the launcher cannot reuse the
session compositor), 272-tile field NIfTI, stages
`load_data,raw_full_tiled_montage,fft_full_tiled_montage`, backends in
separate invocations. Machine carried 8–13/16 external load throughout —
read absolutes as inflated upper bounds (the same caveat as the parent
dossier); the deltas and the work counters are the evidence.

Raw cold entry (the rank-1 callback):

| arm | `action_render_call_ms` | `last_stage_plan_ms` | hot-stage CPU ms | match-loop tiles scanned | heartbeat max gap ms |
|---|---:|---:|---:|---:|---:|
| wgpu A1 | 2062 | 2010 | 1874 | 272 | 2174 |
| wgpu A2 | 1861 | 1814 | 1728 | 272 | 1953 |
| pyqtgraph A1 | 2137 | 2089 | 1859 | 272 | 2812 |
| pyqtgraph A2 | 2071 | 2027 | 1923 | 272 | 2158 |
| wgpu B1 | **55** | **5** | **5** | **13** | 645 |
| wgpu B2 | **38** | **5** | **4** | **16** | 1367 |
| pyqtgraph B2 | **54** | **5** | **4** | **7** | 1181 |

- The same-config spread (A1 vs A2, B1 vs B2) is ~10%; the A→B delta is
  ~40×. The load-immune counter says the same thing: 272 tiles scanned →
  7–16, then deferred to the worker planner.
- The cold-entry callback now sits at 38–55 ms total (`render_entry` EWMA
  26 ms in the live feedback snapshot) — at or under the 50 ms bar on a
  machine where the *baseline* absolute is inflated ~50% over the dossier's.
- `first_visible_tile_ms` did not regress; direction only, because the A
  arms mostly failed to present at all inside the 5.6 s observation window
  (wgpu A: 56–59/272 presented raw; pyqtgraph A: 0). B arms: wgpu 60–272
  presented, one run completing all 272 with first visible at 2.76 s; first
  *physical* tile 144 ms in the other. pyqtgraph presented 0 on both arms —
  the parent dossier's D2 (pyqtgraph cannot rebuild a field montage inside
  the settle limit on this machine) stands and is untouched here.
- The remaining 0.6–1.4 s heartbeat gaps on B arms belong to the dossier's
  other ranks — the fit-stretch pulse (`action_fit_stretch_ms` 0.7–1.4 s in
  the same records, I1/I2's owner) and the FFT preview commit (rank 2, the
  sibling chip) — not to the stage plan.

Live `resource_feedback_channels` on a B arm confirms the loop is closed:
`montage_stage_plan` budget 12 ms, last 4.2 ms @ 16 items (0.26 ms/item);
`render_entry` EWMA 26.0 ms, last 38.3 ms — over-warning, correctly reported,
attributed by its own details instead of surfacing as a mystery gap.

## Honesty / what this does not claim

- Both arms still exit red on the standing R1/R4/R5 invariant rows and R8
  phase-completion gates at this machine's load — pre-existing on main,
  unchanged in kind by this branch. One pyqtgraph B invocation died before
  measurement at the session-fixture viewport restore (734 vs 739 px, the
  known chrome sensitivity); its repeat measured fine.
- The **native-target cold entry** (`native_target_required`: native-only
  policy or native-scale demand) still builds its stage plan synchronously via
  `build_stage_fan_in_plan` — deliberate inline planning, usually small tile
  sets (deep zoom), but unbudgeted; named residue, not fixed here. Same for
  `complete_deferred_stage_fan_in`'s synchronous fallback when no kernel
  exists.
- The 16 ms heartbeat bar is not met during a fill; this chip removes the
  rank-1 owner. Ranks 2 (whole-montage commit — sibling chip) and 3
  (fit-stretch / synchronous programmatic retarget, I1/I2) now own the gap.

## Gates

- `tests/operations/test_region_axis_contains.py` — fast-path semantics vs
  the verbatim original as oracle (fails on any divergence, including the
  stride-alignment trap).
- `tests/window/test_hot_stage_plan_budget.py` — a zero budget defers with
  the placeholder + counters + observation and still scans exactly one tile;
  a generous budget attaches resident values; a bailed slice's match cache
  carries into the next call.
- `tests/ui/test_window_render_interactions.py::test_render_entry_reports_gui_callback_observation`
  — `render()` must report the entry.
- Suite: the affected-set loop's extra failures against this branch were
  re-run 5× on both checkouts under matched load — worktree 1/5 vs main 2/5
  failing runs on the same two load-flaky montage tests; no regression
  attributable to this change. Raw JSONL stayed in the session scratchpad
  per the no-artifacts rule.
