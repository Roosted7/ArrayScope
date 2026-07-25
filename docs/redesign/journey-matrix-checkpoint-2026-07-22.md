# Journey-matrix checkpoint — 2026-07-22 (post Compare-v1 landing)

Regression checkpoint after the 2026-07-22 orchestration session landed Compare
v1 (steps 5/6/7), plugin ops Tier-1 + Tier-2 harness, and several fixes. Also
serves as item-3d (wgpu promotion) evidence.

- **Tree:** `main` at `fd8d830a`.
- **Ring:** headless-weston (`arrayscope.tools.headless_display` → `journey_matrix run`),
  real GL, `wgpu_present_method: bitmap`.
- **Data gotcha:** the matrix needs a local NIfTI under `data/`, which is
  gitignored and NOT shared into git worktrees. First run failed every cell at
  data-load (`FileNotFoundError`); fixed by symlinking the main checkout's `data/`
  into the worktree. (Worth remembering for any worktree-based matrix run.)

## Result

| backend | cold_fill | zoom_in | zoom_out | scroll_shuffle | index_scroll | deep_zoom_far_scroll |
|---|---|---|---|---|---|---|
| **wgpu** | OK | OK | OK | **OK** | **OK** | OK |
| pyqtgraph | OK | OK | OK | RED | RED | OK |
| vispy | OK | OK | OK | RED | RED | OK |

- **wgpu: 6/6 green**, including the fast-scroll journeys (`scroll_shuffle`,
  `index_scroll`) — directly supports the item-3d thesis that wgpu leads
  fast-scroll while the incumbents stall there.
- **pyqtgraph / vispy reds** are on `scroll_shuffle` + `index_scroll` only, with
  reasons `correctness_gate` + `stall_tile_probe`, plus a pyqtgraph `cold`
  `STALL GUARD: montage fully visible but completion gates ['physical_drawn']
  stayed blocked ... active_presented=272/272 fully_visible=True`. These are the
  **documented pre-existing** tile-limbo / cold-fill-tail-stall / fast-scroll
  LOD-miss family (queue.md standing lane + item 1/3d narrative), not new.

## Attribution (why this is a clean checkpoint, not a regression)

1. The session's changed code — `source_anchoring`, `CompositeArraySource`,
   `compare_launcher`, `plugin_conformance` — appears in **zero** red-cell
   tracebacks (`grep` over every `driver.stderr.log` returned nothing).
2. **wgpu shares the one hot-path change** (the `source_anchoring` base-shape
   fix) yet is fully green on exactly the journeys where the incumbents are red —
   so the reds are backend-specific incumbent issues, not shared-code.
3. The red signatures match items already documented as pre-existing.

Conclusion: the Compare-v1 session introduced **no journey regressions**; the
remaining reds are the standing incumbent LOD/stall family. Full AUTO promotion
of wgpu (item 3d) still needs dogfood hours and the VisPy-retirement call, which
are the owner's, not something to flip from one matrix run.

Artifacts: `tests/artifacts/journey-matrix-integrated2-fd8d830a/` (gitignored).
