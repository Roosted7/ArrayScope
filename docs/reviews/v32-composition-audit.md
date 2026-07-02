# v32 composition audit (2026-07-02)

Full-project review answering: why is troubleshooting increasing, what is
structurally wrong, what was fixed in v32, and what remains. Supersedes the
open recommendations of the v30/v31 audits where they overlap.

## Verdict

The semantic core is healthy. The growing bug rate was structural, and it had
one root cause: **the window was a god object of 14 mixins sharing one flat
namespace**, so orchestration state had models but no owners. Every fix that
could not see a sibling mixin's invariants added another token, guard, or
timer — and each of those became the habitat of the next bug.

## Measurements (pre-v32)

- `FrameRenderMixin`: 3,897 lines, ~100 methods, 71 distinct `self.*`
  attributes written onto the shared window.
- 8 revision-counter schemes; 3 session-key constructions; 5 staleness-check
  patterns; 4 tile-priority systems; 3 work-admission paths.
- 11 Qt timers in the render path; several unparented single-shot chains.
- 17 manual `_sync_*` methods mirroring `ViewState` into ~50 widgets.
- ~1,200 duplicated lines between the PyQtGraph and VisPy view classes
  (~40 methods implemented twice); the two `tiles.py` files are
  parallel-divergent.
- 3 cache implementations with near-identical eviction/priority ranking.
- `tools/profile_montage_workflow.py` (2,072 lines) re-implements window
  composition instead of reusing it.
- Tests: 1,255 functions, ~30k lines; heavy monkeypatching of orchestration
  internals; several files replaced production modules in `sys.modules`.

## Bugs found and fixed during the audit

1. **Teardown segfault (canvas preserve).** Unparented
   `QTimer.singleShot` retry chains fired after window close and touched a
   deleted `QMainWindow`. Fixed by scoping the callbacks to the window's
   lifetime (receiver context) and cancelling canvas preservation in
   `closeEvent`. Generation guards protect against *stale* callbacks; receiver
   context protects against *dead* receivers — both are required.
2. **Resize segfault after reduction to line-plot mode.** The montage resize
   retarget read live `view_state` (now `image_axes=None`) against the stale
   montage session and raised inside a C++ `resizeEvent` override, corrupting
   Qt dispatch. Fixed with a semantic guard; additionally, the
   `except TypeError: handler()` retry shim in
   `_notify_viewport_content_resized` re-invoked the failing handler and was
   replaced with the standard UI error path.
3. **FramePlanner montage-layout drift.** `FramePlanner` re-derived montage
   columns from raw view-state, while the viewport planner can override
   requested columns (auto layout with an auto-owned camera). The committed
   geometry disagreed with the applied layout and the camera re-fit to a
   phantom layout, hiding tiles after auto-fit. The applied `MontagePlan` is
   now passed into planning: the applied layout is the single source of truth.
4. **Order-dependent test identity failures.** Several test files loaded
   production modules via `spec_from_file_location` and installed the
   duplicates into `sys.modules`. Production code and tests then held
   different class objects depending on collection order ("fails together,
   passes alone"). Replaced with plain imports.
5. **Python 3.10 support was broken** (`StrEnum` in `core.work_graph`)
   despite `requires-python >= 3.10`.
6. **Host-dependent tests.** Memory-policy budgets derived from sampled host
   RAM, so the same test passed on a 32 GiB workstation and failed on a 4 GiB
   runner. The suite now pins a deterministic snapshot
   (`real_system_memory` marker opts out).

## Dead code removed

"Tiles all the way" (26c768d4) removed the single-image presentation path but
left its support layer: `operations/render_plan.py` (refuse/degraded/chunked
decision machinery, no callers), `window/stage_warmup.py` (idle stage warmup,
caller removed), `resource_governor.decide_stage_warmup`, the
runtime-diagnostics stage-warmup field, and the tests pinning all of it.
Roughly 20 tests pinned removed behavior and were deleted or rewritten against
the tiled pipeline.

## Structural change (ADR 0045)

Render orchestration moved off the window into one composed
`RenderOrchestrator` (`window.renderer`), a `QObject` child of the window.
See the ADR for the ownership contract.

## What remains (now roadmap gates Y1–Y3)

1. **Y1 — one generation contract.** Collapse the parallel revision/staleness
   schemes into one owner on the orchestrator; route the remaining ad-hoc
   admission paths through `WorkGraph`. Timers reschedule; they never encode
   semantic order.
2. **Y2 — backend de-duplication.** Hoist the ~40 twice-implemented shell
   methods into `ImageViewShell`; extract the shared Qt-free tile-model logic
   from the two `tiles.py` files; keep only texture/atlas vs. QGraphicsItem
   mechanics backend-specific.
3. **Y3 — UI state binding and tools reuse.** Replace the 17 manual `_sync_*`
   fan-outs with one observer-style binder keyed on `ViewState` revisions;
   make `tools/profile_montage_workflow.py` drive the production window
   composition instead of re-implementing it. Unify the three cache
   eviction/priority implementations behind one core.
4. **X5 unchanged** — hardware evidence and residency policy remain the gate
   for LOD and backend-default decisions. Note: VisPy-under-Xvfb (llvmpipe)
   is intermittently unstable; software-GL CI results are not evidence for or
   against the VisPy backend.

## Process observations

- The ADR/doc discipline is a real asset; the docs were largely accurate.
- The failure mode was not "no architecture" but "architecture enforced by
  convention across a shared namespace." Ownership boundaries must be
  structural (objects), not documentary.
- Tests that monkeypatch orchestration internals rot quickly; prefer driving
  the real pipeline and asserting on deterministic work counters and
  committed-frame semantics.
