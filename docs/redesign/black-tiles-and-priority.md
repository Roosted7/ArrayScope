# Root-cause dossier: black tiles & priority order (2026-07-14)

Findings from a full read-only investigation of the `redesign` branch at
18ba39db. These are ranked candidate mechanisms with code evidence — verify
against the harness scenario before and after each fix.

## A. Priority rendering order

There are **two render paths and three viewport-distance rankers** that have
drifted apart:

1. `_tile_priority_rank` — `render/effects.py:564` (VisPy/resident ladder path)
2. `MontageTilePriorityQueue._priority_key` — `display/model/tile_priority.py:215`
   (native-only/pyqtgraph missing-tile queue)
3. `prioritize_montage_tiles` — `window/montage_viewport.py:108` (queue seeding)

Commit e6665315 ("Restore montage priority") touched only #1, verified only by
an isolated sort-order unit test on a `SimpleNamespace` session
(`tests/render/test_effects.py::test_tile_lod_states_prioritizes_screen_distance_in_landscape_viewport`).
No test asserts end-to-end kernel execution or paint order.

### A1 (most likely): per-tile distance never reaches the kernel

The kernel ready-heap key is `(rank, priority, deadline, seq)` —
`kernel/scheduler.py:507`, `kernel/task.py:130`. Per-tile center distance is
**not a field on `TaskSpec`**; it survives only as submission order (`seq`
tiebreak; admitted in comments at `render/effects.py:549-551` and
`render/pipeline.py:204-207`). Meanwhile `LodLadder.plan` emits all tiles'
FLOOR rung, then all PREVIEW, then DESIRED, then EXACT
(`render/ladder.py:262-265`), each rung with a different `Priority`. Priority
sorts before seq, so **a coarse edge tile executes before a fine center tile**
— center-out holds only within one rung. No ranker fix can change this while
distance is absent from kernel ordering.

### A2: chunked admission reorders the sorted tail

`retarget` submits at most `ADMISSION_CHUNK = 24` steps synchronously
(`render/pipeline.py:125,340`), the rest via a continuation task. A concurrent
retarget bumps `_admission_generation` and clears `_pending_admissions`
(`pipeline.py:160-163`), so under scroll churn the sorted tail is discarded
and re-interleaved with new head steps.

### A3: ranker scores against a stale viewport

`_tile_priority_rank` reads `session.view_range`
(`render/effects.py:533,556,570`), but priority retargets deliberately do not
update `view_range` (`frame_session.py:3599-3602,3621` uses a separate
`range_for_priority`). During interaction path-1 ranking can degenerate to
`montage_index` order (`effects.py:565` fallback).

### A4: path-2 queue context staleness

`MontageTilePriorityQueue` pops under `_tile_priority_context`
(`frame_session.py:751-753,3634`), written only by `retarget_tile_priority`.
Enqueues from `_classify_visible_montage_tiles`
(`window/frame_runtime.py:1064-1069`) without a matching retarget pop under a
stale context — the exact hazard warned about at
`display/model/tile_priority.py:52-61`.

### Fix direction (V2)

One ranker function, used by both paths; per-tile distance carried into
kernel ordering (either a fine-priority field in `TaskSpec` or per-tile rung
interleaving at plan time); delete the two other rankers. Acceptance: harness
observes real commit order center-out on cold load and fast scroll.

## B. Persistent black tiles

### B1 (most likely): boundary tiles excluded from admission but still visible

Interaction of four recent commits:

- 4a05cdca made `tiles_intersecting` strict (`>/<`) —
  `display/montage.py:142`; `active_region_ids` is built with
  `margin_tiles=0` (`display/frame_planner.py:208-211`). A tile whose edge
  lands exactly on the viewport boundary is excluded.
- Once display is committed, ladder admission narrows to
  `onscreen_tile_numbers()` (`window/frame_runtime.py:190-195`), and
  `tile_lod_states` only plans tiles in that scope
  (`render/effects.py:453-455,467`). The "coverage ring" is carried but
  **never admitted to the ladder** (`frame_runtime.py:196-199`).
- 359f6184 made `visible_plan_complete()` = `onscreen_target_settled()`
  (`frame_session.py:3274`).
- d6ce5f40 scoped first-pass completion and level evidence to
  `onscreen_tile_numbers()` (`frame_session.py:711-736`,
  `render/level_stats.py:344-364`).

Net: an edge tile is never planned again, yet the frame declares first pass
complete and publishes the histogram → the tile stays black while everything
reports converged. Violates the stated invariants "one-index scrolling must
not create a full-black frame" and "placeholders only for targets with no
compatible source".

Tests that **lock in** this regression (candidates for deletion under ground
rule 4):
`tests/ui/test_frame_session.py::test_first_pass_physical_completion_is_scoped_to_onscreen_targets`,
`…::test_…visible_plan_is_gated_by_onscreen_not_coverage_ring`,
`tests/window/test_montage_backend.py::test_first_pass_rough_evidence_completion_uses_physical_onscreen_scope`,
`tests/display/test_frame_planner.py::test_montage_excludes_tiles_that_only_touch_the_viewport_boundary`.

### B2: level-evidence deadlock after first-pass publication

After `_first_pass_rough_evidence_closed` (`render/level_stats.py:366-371`),
`_schedule_montage_cached_level_stats` clears `pending_level_tiles` /
`pending_level_sources` / `level_scan_remaining_tiles`
(`level_stats.py:625-631`). From then on only the semantic evidence pass can
supply levels — but that pass is gated on
`_montage_side_work_visible_settled` (`level_stats.py:509-516,1447-1464`),
which a single stranded tile (B1) keeps false forever. The black tile blocks
the very pass that would give it levels. Circular; no timeout.

### B3: acknowledgement gating hides pixels when acks never arrive

370e35e6 filters active VisPy tile slots on acknowledged identity
(`display/backends/vispy/tiles.py`); 40ebf8ae defers DESIRED/EXACT rungs
until `first_pass_histogram_published` (`window/frame_effects.py:322-328`).
If B1/B2 prevent publication, deferral is forever and unacknowledged slots
stay hidden → black.

### B4: intended black exists and masks triage

e93737f7 pins that zero-valued complex backgrounds render black (correct
behavior). "Black tile" is therefore sometimes right — check the truth
overlay before assuming a bug.

### Fix direction (V1)

One function owns "tiles that must render"; admission, completion, and
evidence scoping all call it. Boundary rule may be strict for *completion
accounting* only if the same tile is also excluded from *visibility*; a tile
the user can see must be in the admitted set. Remove the rough-scan teardown
until the admitted set is covered, or give the semantic pass a
stranded-work escape. Acceptance: harness one-index-scroll scenario with a
boundary-landing tile, both backends, real Wayland.

## C. Silently dead code paths (fix first — V0)

`window/montage_prefetch.py:258` and `:454` import from the deleted
`arrayscope.window.frame_renderer` inside `try/except Exception`:

- `_interaction_active(window)` always returns `False` → prefetch runs as if
  the user is never interacting (competes with visible work during scroll —
  plausibly contributes to the priority symptom).
- The retained-preview-level admission branch never runs.

The symbols now live at `frame_runtime.py:1291` / `frame_controller.py:1780`.
Also: `_viewport_interaction_active` is defined identically in three modules
(`frame_runtime.py:1291`, `frame_controller.py:1974`, `level_stats.py:1596`)
— consolidate to one. Add the V0 import-health guard so this class of rot is
loud.
