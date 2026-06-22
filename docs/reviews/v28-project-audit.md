# ArrayScope v28 project audit

- **Review date:** 2026-06-22
- **Reviewed ArrayScope baseline:** v28 development line
- **Review branch:** `audit/v28-review`
- **Comparators:** ArrayShow reviewed `develop` branch; supplied ArrayView `v0.26.3` checkout

## Executive assessment

ArrayScope is no longer merely a small Qt viewer. It now has a credible architecture for reversible scientific operations, region-aware evaluation, explicit memory/cost policy, progressive montage rendering, backend-independent presentation state, and diagnostics. Those are real strengths and distinguish it from both comparator projects.

The project is also at an integration-risk boundary. Recent development advanced quickly while normal image versus montage and PyQtGraph versus VisPy still use different orchestration/lifecycle paths. Four of the largest source files are roughly 1,700–2,100 lines, there are many timer-based scheduling edges, and recent changes added hundreds of lines to hot interaction paths in one day.

The correct response is not another broad rewrite or a rush of competitor features. Stabilize correctness and release provenance, enforce callback budgets, preserve visible progress during continuous input, then converge frame planning and backend composition. ArrayShow and ArrayView should influence the product surface—dimension-local immediacy, linked/compare workflows, restraint, command discoverability—not replace ArrayScope’s stronger semantic core.

## Scope and method

The review covered:

- source history and diffs around the current head, with extra scrutiny on the latest slicing, histogram, viewport, and tile-priority changes;
- source ownership, file/function concentration, callbacks, timers, cache/state identity, error handling, and lifecycle;
- pure, backend, window, and selected Qt tests plus deterministic rendering benchmarks;
- existing ADRs, phase notes, reviews, roadmap, ideas, README, changelog, and contributor guidance;
- ArrayShow source/interaction patterns and the complete supplied ArrayView checkout;
- the supplied GPU-rendering issue note and its v25/v27 architectural direction.

The environment supported headless PySide6/PyQtGraph/VisPy contract tests. It did not provide a real display session/GPU matrix, so this review does not claim to validate actual GPU throughput, texture limits, Wayland behavior, frame pacing, or interaction feel.

## Findings by priority

| Priority | Finding | State |
|---|---|---|
| High | Normal render refusal fallback referenced `format_bytes` without importing it, replacing a safety message with `NameError`. | Fixed and tested. |
| High | Viewport max-span clamping returned before minimum-overlap recovery; after crop/content changes, an old distant center could keep all content offscreen. | Fixed and tested. |
| High | Several GUI callbacks can still process complete stage-key/tile collections and sort/requeue many tiles in one turn. | Open; roadmap N1/N3. |
| High | Latest-only/coalesced visible scheduling can repeatedly cancel exact work during continuous interaction instead of retaining active progress. | Open; roadmap N2. |
| Medium | Tile priority is calculated at plan/release points, but mouse motion only records hover/value state; an already-active queue is not dynamically retargeted. | Open; roadmap N3. |
| Medium | Adaptive histogram sampling/binning is bounded but runs in a zero-delay GUI callback. It needs measured callback evidence and off-thread refinement if thresholds are exceeded. | Open; roadmap N4. |
| Medium | Queued histogram callbacks and parentless benchmark views produced lifecycle/teardown hazards; the benchmark suite repeatedly rebuilt the whole matrix. | Fixed and tested. |
| Medium | `VisPyImageView2D` still subclasses the complete PyQtGraph widget, retaining two scene/event/lifecycle systems. | Open; roadmap X2/X3. |
| Medium | Normal-image and montage planning/scheduling remain separate, so semantic fixes and feedback tuning can diverge. | Open; roadmap X1/N2. |
| Medium | Large orchestration/backend modules and many timer/generation interactions make changes difficult to review locally. | Open; refactor only along ownership seams. |
| Medium | Selection text supports Python-like and compatibility repairs/fallbacks that can be surprising without a normalized preview. | Open product issue; idea recorded. |
| Medium | Package/version/changelog/remote provenance do not yet describe one releasable product state. | Open; roadmap N0. |
| Low | Broad `except Exception` remains common in UI/backend paths. Many are intentional fallbacks, but strict diagnostics should expose unexpected cases. | Ongoing cleanup. |
| Low | Historical phase notes and the 593-line cumulative roadmap competed with current guidance. | Fixed through documentation reorganization. |

No evidence of a current silent data-corruption defect was found in the operation planner/cache core after correcting two stale tests that were not changing the axes their names claimed to change.

## Recent Change Scrutiny

### Tile priority around viewport center or hover

**What improved:**

- Visible candidate/missing/additional tile sets are ordered by normalized distance from a focus point.
- Stage-wait release reuses the same ordering.
- Tests cover center/hover ordering and interaction wiring.

**Risks:**

- `prioritize_montage_tiles` materializes the iterable and performs `sorted(...)`, so each call is `O(n log n)` plus tuple allocation.
- `_on_image_mouse_moved` stores the scene point and refreshes pixel information only. The active pending queue is not re-keyed until a later planning/release event.
- Calling the sort directly from mouse move would create a worse regression. Dynamic priority needs coalesced focus updates and a stable heap/bucket/index with aging.
- `_activate_montage_stage_value` and `_release_stage_waiting_tiles_to_direct` can sort and append every waiting tile in one callback.
- `_process_montage_attached_stage_waits` loops all attached keys every 25 ms; each key may release a large tile set.

**Conclusion:** good user-facing intent and deterministic ordering, but only a first step toward dynamic bounded priority. Do not label hover priority complete until N3’s exit gate is met.

### Slicing/range extension and cropped X/Y views

**What improved:**

- Central pure `slice_selection` parsing/normalization and dedicated tests.
- Range selections can affect image axes, enabling cropped X/Y views.
- Dimension strip/state synchronization is more explicit.

**Risks:**

- The grammar accepts multiple conventions/repairs. For example, compatibility forms can interpret colon order differently and clamping can turn out-of-range lists into repeated boundary values.
- Recent code changed core state, controls, strip behavior, and state sync together; interaction combinations deserve focused regression across image/line/montage roles.
- Cropped axes affect geometry, viewport recovery, profiles, cache keys, and montage tile shapes, so tests should keep spanning those boundaries.

**Conclusion:** useful and well-tested core change, but UX must show the normalized selection. Avoid growing another parser in a widget.

### Adaptive/manual histogram and auto-window revert

**What improved:**

- Adaptive histogram bins react to visible range/pixel height.
- Manual level editing, revert behavior, and status feedback are more capable.
- Sampling and bin caps bound data size.

**Risks found:**

- The change added roughly 635 lines to `histogram_controller.py`, which now mixes plotting, range interaction, popups/editing, scheduling, and numerical refinement.
- `schedule_refresh()` uses `QTimer.singleShot(0, ...)`; sampling, finite filtering, min/max, and `np.histogram` execute on the GUI thread. The default sample is bounded around a 200×200 image grid, but dtype/conversion and repeated refresh cost still require traces.
- A queued single-shot callback could fire after graphics objects were deleted. The audit added cancellation/guarding on close.

**Conclusion:** strong functionality; next work is measurement and ownership split, not more controls in the same controller.

### Viewport constraints

**What improved:**

- View-range policy became a pure, testable module shared by backend paths.
- Maximum zoom-out, minimum overlap, preserve/fit, and VisPy synchronization behavior became explicit.

**Defect found:**

- The max-span branch returned immediately after scaling around a previous center. If that center was far outside newly cropped content, the later overlap clamp never ran. The audit removed the early return and added a regression.

**Conclusion:** the extraction was correct; the bug illustrates why viewport policy should remain pure and exhaustively edge-tested.

### Montage auto-fit and preceding tiled-render repairs

Auto-fit of expanded montage ranges improves usability, but fit is semantic UI state and must not cause session recreation or disguise repeated range churn. The preceding repairs materially improved stale rejection, retained tiles, cold-work diagnostics, level coverage, and acknowledged dirty state. Preserve those invariants while unifying planners; do not “simplify” them back into a single rendered boolean.

## Architecture and control-flow assessment

### Strong foundations

1. **View/document separation.** `ViewState` and `ArrayDocument` provide a coherent semantic base.
2. **Operation declarations.** Shape/dtype/region/cost behavior stays with operations rather than renderer type switches.
3. **Runtime planning.** Optimizer, region planner, slab/chunked execution, stage cache, and singleflight are strong and testable.
4. **Display semantics.** Frames, presentations, geometry, level sources, and backend adapters prevent libraries from defining meaning.
5. **Lifecycle distinctions.** Requested/materialized/resident/presented is the correct model for progressive rendering.
6. **Resource policy.** Separate memory caches, lane worker policy, feedback, telemetry, and governor provide the right levers.
7. **Architecture tests.** Guards prevent backsliding into main-window/backend-owned semantics.

### Concentration and seams

At the reviewed baseline:

| File | Approx. lines | Main concern |
|---|---:|---|
| `window/montage_renderer.py` | 2,103 | session planning, stages, workers, presentation updates, priorities, levels, diagnostics, UI status |
| `display/imageview2d.py` | 2,088 after audit | shell, PyQtGraph mechanics, ROI/profile interaction, histogram binding, overlays |
| `display/vispy_imageview2d.py` | 2,095 after audit | inherited shell plus VisPy camera/raster/tile/overlay bridging |
| `display/backends/vispy/tiles.py` | 1,792 | atlas/residency/visual/upload mechanics and diagnostics |
| `display/histogram_controller.py` | 695 after audit | numerical sampling plus UI scheduling/editing/plotting |

Do not split these files by arbitrary line count or create matching PyQtGraph/VisPy copies. Extract along ownership:

- frame/session target and queue policy;
- stage-wait admission/release batches;
- presentation-delta batching/acknowledgement;
- shell lifecycle and shared interaction state;
- concrete raster/atlas/visual mechanics;
- histogram model/refinement versus widget/editor.

### Timer/control-flow risk

The source has dozens of `QTimer`/`singleShot` sites across viewport debounce, render coalescing, slow overlays, upload flushing, warm residency, prefetch, profile refresh, stage waits, and histogram refresh. Timers are acceptable for coalescing/resubmission, but they currently also encode implicit ordering between independent state machines.

Each timer should have:

- one owner and cancellation path;
- explicit target/revision guard;
- bounded callback work;
- diagnostics reason/lane;
- no assumption that elapsed quiet time implies resource availability.

### Error handling

UI/backend paths contain many broad exception fallbacks. They protect optional backends and teardown, but can mask programming errors. Keep graceful production fallback while supporting a strict developer mode that logs traceback, target identity, backend, and recovery action. Narrow exceptions where the failure contract is known.

## Performance and responsiveness assessment

### Likely bottlenecks now

1. **GUI fan-in, not just workers.** A fast producer can overwhelm Qt if one callback accepts many tile/stage results or touches many items.
2. **Repeated queue materialization/sorting.** Focus ordering is cheap for small visible sets but can be expensive when applied to all waiting tiles repeatedly.
3. **PyQtGraph item count.** Persistent items avoid full canvases, but many `ImageItem` visibility/geometry updates still cost Python/Qt scene work.
4. **CPU display preparation.** PyQtGraph complex/window mappings and any repeated RGBA/prepared image work can dominate before upload.
5. **VisPy hybrid lifecycle.** Two scene systems increase event/camera/teardown complexity and can invalidate benchmark interpretation.
6. **Histogram work on the GUI thread.** Bounded is not synonymous with below 4/8 ms.
7. **Cancellation churn.** Latest-only exact image renders can starve under continuous input.
8. **Stage-wait bulk release.** A completed reusable stage can release many tiles in one GUI callback.
9. **Speculation feedback contamination.** Warm rebind and cold upload must remain separate metrics or the controller will grow unsafe batches.

### What is already correct

- Full montage canvases are avoided in the interactive path.
- Native-resolution tile residency is the default; mixed-size CPU LOD is not silently used in fixed atlas slots.
- Levels/LUT are separated from tile materialization identity.
- Pan/zoom retarget sessions/residency rather than automatically rerunning operations.
- Cold upload/rebind/presented work has explicit counters in the repaired paths.
- Memory estimates can refuse/degrade requests before allocation.

### Required evidence

For each reference workload/backend, capture event-to-first-frame, event-to-exact-visible, event-loop gaps, queue delay, preparation ms/item, upload bytes/ms, cold/warm counts, cancellation/reuse, cache/stage hits, RSS, and GPU-residency estimates. Real GPU/device/platform runs are required before selecting a new default backend.

## Documentation assessment and changes

### Before

- The README described the legacy lightweight viewer and omitted most current architecture/features.
- The changelog mixed legacy `0.5–0.7` releases with current `0.0.1` metadata.
- `architecture.md` was a long evolving ownership list without a progressive reading path.
- The roadmap was 593 lines combining completed phases, current implementation notes, and speculative ideas.
- Phase context and old manual checklists sat beside live documentation.
- ArrayShow notes were eight lines with a placeholder source note; ArrayView was absent.
- Dated reviews were useful but competed with current direction.

### After

- `docs/index.md` provides “orient → understand → rationale/evidence” levels.
- Mission, current state, architecture, roadmap, ideas, testing, and comparison have distinct jobs.
- Architecture has focused deep dives for state/operations, rendering, scheduling/memory, and interaction/UI.
- ADRs have a categorized status index.
- Phase notes, old roadmap, and historical checklists are preserved under `docs/archive/`.
- ArrayShow/ArrayView notes record exact revisions and adopt/adapt/avoid lessons.
- This review is the dated evidence; durable action is reflected in the roadmap.

## Comparator conclusions

### ArrayShow

Best lessons: dimension-local immediacy, linked viewers, scripted multi-window inspection, and quick cursor/ROI workflows. Avoid its global workspace/viewer registry, monolithic handle ownership, destructive/direct mutation tendencies, and callback/timing repair mechanisms.

### ArrayView

Best lessons: array-first visual restraint, one command registry, keyboard/help discoverability, invocation quality, focused compare UX, and explicit visual/mode audits. Avoid its 29,561-line frontend concentration, 3,858-line launcher, dual-write migration state, transport duplication, mode combinatorics, global session tendencies, and CPU RGBA/PNG local-render bottleneck.

### ArrayScope’s position

ArrayScope should combine ArrayShow’s dimension awareness and ArrayView’s restraint with its own stronger reversible computation, semantic progressive rendering, and resource policy. It should not combine all three projects’ feature lists.

## Changes Made During The Audit

### Restored Code Fixes

- Centered-slice and prioritized-montage tests, persistent VisPy test settings, and empty Inspection-dock no-op refreshes.
- Bounded evaluation-group invalidation and completed per-tile generation cleanup.
- Render-refusal fallback and single-action auto-window behavior.
- Consolidated CI and release tag/package/runtime guards.
- Concurrent multi-path CLI launches with single-path compatibility.

### Complementary Follow-Up Fixes

- Display resource shutdown and bounded benchmark lifecycle.
- Viewport overlap after max-span clamping.
- Montage priority input hardening and early hover cleanup.

Documentation changes are merged separately so code fixes remain easy to review.

## Test evidence

Validated during the review:

- focused viewport, normal render refusal, operation cache, and complex chunked regressions;
- complete core suite baseline (99 tests before changes);
- application/architecture suite baseline (63 passed, 1 environment-dependent skip);
- combined core/operation/window and targeted regression layers in the earlier audit pass (505 passed, 1 skip);
- rendering benchmark/lifecycle module after the fix (9 passed in approximately 14 seconds headless);
- static undefined-name/syntax checks and Markdown link validation are part of final validation.
- after restoration, the complete headless/offscreen suite passed with `935 passed, 1 skipped`.

Some large combined headless display invocations exceeded the tool’s execution envelope before the lifecycle fix and were subsequently split by module. The restored branch now has a successful full headless/offscreen run; real-display GPU throughput, Wayland, DPI, context-loss, and interaction-feel checks remain manual handoff work.

## Recommended sequence

1. Complete roadmap N0; do not add broad modes before one coherent releasable baseline.
2. Instrument and enforce callback budgets (N1).
3. Implement progress-preserving active-plus-latest scheduling (N2).
4. Implement coalesced indexed priority with aging (N3).
5. Measure/refine histogram/level work (N4).
6. Unify frame planning/storage strategy (X1).
7. Replace backend inheritance with composition and shared pointer lifecycle (X2/X3).
8. Build real hardware/platform evidence and only then decide backend defaults/multi-resolution design (X4).
9. Productize linked viewers and a narrow compare workflow after the foundation gates.

## Bottom line

The project’s hard-won semantic and resource architecture is worth protecting. The main danger is not lack of features; it is adding more interaction and rendering behavior before the existing scheduling/backend paths converge and become measurably bounded. The revised documentation and roadmap are designed to keep that distinction visible.
