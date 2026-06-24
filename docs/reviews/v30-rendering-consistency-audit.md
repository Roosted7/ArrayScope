# ArrayScope v30 rendering-consistency and architecture audit

- **Review date:** 2026-06-24
- **Supplied baseline:** `d35f30d` plus uncommitted v30 histogram/benchmark work
- **Preserved supplied-work commit:** `103ab67`
- **Review branch:** `review/v30-render-consistency`
- **Primary scope:** histogram/window levels, progressive montage presentation, PyQtGraph/VisPy
  integration, LOD, benchmarks, rendering architecture, roadmap, and ADRs

## Executive verdict

ArrayScope does **not** need its operation/evaluation core or display semantics thrown away. Those are
among the strongest parts of the project. It does need a deliberate reroute of the rendering control
plane.

The current failure mode is architectural rather than a collection of random PyQtGraph bugs:

- a global semantic command is spread across session state, widget preview state, presentation
  planning, tile admission, backend deadlines, acknowledgement, timers, and committed-frame history;
- those layers use several overlapping meanings of “done”;
- PyQtGraph and VisPy share semantic intent but have fundamentally different physical level-update
  costs;
- `MontageRenderSession` and `montage_renderer.py` now coordinate too many independent state machines;
- benchmarks originally observed visibility and last callback counters, not end-to-end convergence.

That makes fixes non-local. A change can be individually reasonable yet violate a lifecycle assumption
elsewhere. The observed “most tiles update, a varying subset does not” symptom is a direct example:
retained visibility was accepted as proof that the requested replacement had committed.

The right strategy is:

1. stop feature expansion in this area until presentation-generation gates pass;
2. preserve the latest-target/bounded-progress fixes in this review;
3. extract generation tracking, tile admission, and stage fan-in as Qt-free state machines;
4. keep backend-specific convergence strategies behind one semantic contract;
5. leave production LOD native-only until asynchronous materialization and compatible residency exist;
6. only then continue unified frame planning and backend-shell composition.

This is a substantial refactor, but not a ground-up rewrite. Rewriting the working operation planner,
caches, frame/value semantics, or both renderer backends would increase risk and discard useful,
well-tested code. The unsafe synchronous LOD implementation and implicit timer-ordering patterns should
be discarded.

## Scope and method

The review used the supplied repository including `.git` history. It covered:

- the 28 commits after the `0.8.0` RC baseline `7c34b4b` through the review branch;
- diffs and history for level/histogram, montage session, PyQtGraph tile layer, VisPy atlas/shaders,
  LOD, resource governor, benchmarks, and diagnostics;
- targeted Qt/offscreen tests and deterministic work-counter tests;
- current architecture, roadmap, ADRs, testing guidance, and previous reviews;
- source/file/function concentration and recent churn;
- a small local CPU LOD microbenchmark to establish order of magnitude, not a cross-platform
  performance claim.

The environment supports Python/PySide6/PyQtGraph headless tests. It does not provide the project’s
full conda/direnv setup, `ruff`, or a representative real OpenGL/Wayland/Windows/macOS matrix.
Therefore, this audit does not claim GPU throughput, compositor latency, pointer feel, DPI correctness,
or production platform validation.

## Bottom-line disposition by subsystem

| Subsystem | Verdict | Action |
|---|---|---|
| `ViewState`, document, operation declarations/planner/evaluator | Strong | Keep; continue pure-model tests. |
| Display frame/presentation/geometry/value-source models | Correct direction | Keep; strengthen generation terminology. |
| Resource/memory policy and deterministic counters | Valuable | Keep; correct observations and add end-to-end gates. |
| PyQtGraph tile layer | Viable production fallback | Keep mechanics; always route large committed level work through bounded session admission. |
| VisPy tiled renderer | Promising experimental backend | Keep; isolate from inherited PyQtGraph shell and validate on hardware. |
| `MontageRenderSession` control plane | Overloaded | Incrementally split into focused state machines. |
| `montage_renderer.py` orchestration | Overloaded | Shrink after state extraction; do not split by arbitrary backend copies. |
| Histogram numerical model | Good direction | Keep pure plot/refinement logic. |
| Histogram PyQtGraph binding/editor integration | Brittle | Isolate behind an adapter or replace with owned plot/region shell. |
| Original CPU LOD presentation path | Unsafe | Do not revive; replace with ADR 0041 design. |
| Current LOD selector | Useful as demand diagnostic | Keep, improve per-axis/tile-shape reasoning, apply factor only after prerequisites. |
| Raw widget microbenchmarks | Useful backend probes | Keep, but label separately from application convergence benchmarks. |
| Profile workflow | Useful but too large | Keep outputs; split scenario/action/settlement/reporting modules after stabilization. |

## Recent-work assessment

### What the recent optimization route did well

The last development sequence contains several sound decisions:

- GUI callback budgets turned responsiveness into an observable contract.
- Active-plus-latest visible scheduling avoids repeated cancellation of useful near-complete work.
- Dynamic viewport/hover priority is better than source-index order.
- Direct tile deltas and stable payload wrappers reduce full-state rebuilding.
- Materialization identity no longer includes ordinary level/LUT state.
- VisPy persistent residency and uniform level updates avoid needless texture uploads.
- Stage-plan/candidate caching removes repeated planning from hot commits.
- Cold upload, warm visibility/rebind, texture bytes, and callback observations are increasingly
  separated.
- Semantic montage level/histogram summaries avoid deriving scientific meaning from atlas textures or
  placeholders.
- The new level-loop profile scenario is exactly the kind of integration benchmark the project needs.

These should not be undone.

### Where the method broke down

The optimizations were layered onto a control plane whose ownership was already transitional. The
project then used more counters, flags, revisions, queues, and timers to repair interactions between
those layers. The result is correct in many isolated paths but increasingly difficult to prove as a
whole.

Since RC baseline `7c34b4b`, the reviewed branch contains 28 commits and changes 79 files with roughly
10,404 insertions and 1,085 deletions. Highest churn includes:

| File | Churn | Commits touching it | Current size |
|---|---:|---:|---:|
| `tools/profile_montage_workflow.py` | 2,430 lines | 12 | ~2,028 lines |
| `window/montage_renderer.py` | 1,437 | 13 | ~3,070 lines |
| `window/montage_session.py` | 660 | 13 | ~1,155 lines |
| `display/backends/vispy/tiles.py` | 531 | 6 | ~2,075 lines |
| `display/vispy_imageview2d.py` tests/source combined | high | repeated | source ~2,202 lines |
| `display/imageview2d.py` tests/source combined | high | repeated | source ~2,230 lines |

The issue is not line count alone. Large hot functions also combine many branches and effects:

- VisPy atlas `update_payloads`: ~354 lines and ~50 branch nodes;
- montage tile-layer commit: ~261 lines;
- PyQtGraph direct tile update: ~218 lines;
- session `build_tile_presentation`: ~156 lines and ~33 branch nodes.

This is enough concurrent change in the most timing-sensitive files that integration regressions are
expected unless semantic state-machine tests lead the implementation.

## Finding 1: level targets were globally semantic but incorrectly acknowledged

### Symptom

The level-loop profile showed most tiles updating but a varying subset retaining older levels. Auto
levels could update initial tiles but not all. Direct interactive changes sometimes appeared better,
because timing and target supersession changed which subset was admitted/acknowledged.

### Root cause

The session correctly represented PyQtGraph level convergence as stale per-tile work and fed it through
the existing priority/budget path. The error was in acknowledgement:

- the backend report returned all currently drawable tiles as `presented_tiles`;
- a deadline could defer a requested upsert while retaining the old item on screen;
- the session treated that retained visibility as acceptance of the new upsert;
- it advanced the tile’s level revision/value and removed it from pending work;
- the generation could report completion with old pixels still displayed.

This is a lifecycle collapse:

```text
retained drawable old item != requested replacement committed
Correct backend split
A global target is not required to be physically atomic.

PyQtGraph:

scalar items may call setLevels() per item;

RGB/complex/component views can require CPU re-windowing and image replacement;

every active tile must enter the existing priority/admission queue;

callbacks are bounded and may expose a temporary old/new mix;

convergence is only true when each active tile acknowledges the latest target.

VisPy:

compatible source textures remain resident;

levels are shader uniforms;

all active page visuals can normally update in one backend transaction;

texture-upload counters must remain zero for level-only changes.

Semantic parity means the same final target, source rank, values, and completion—not identical physical
work.

Repairs implemented
TileCommitReport.committed_upserts records exactly which requested upserts were accepted.

presented_tiles remains the drawable set and may include retained old pixels.

Session acknowledgement only advances accepted upserts.

Rapid supersession preserves the latest revision/value and ignores old reports.

A concrete target clears obsolete force_auto work attached to the session.

The latest desired_level_values remains authoritative during progressive commits; persistence
(user_levels_override) remains separate.

Repeating identical numeric levels reuses the current generation. A settled target is a no-op rather
than an unnecessary full redraw.

Obsolete alternative level-acknowledgement fields were removed.

Important invariant
PyQtGraph is allowed to be temporarily inconsistent only while explicitly unsettled. That state
must be visible in diagnostics through target revision, stale active count, pending work, and settled
status. The product must never call it finished merely because all old items remain visible.

Finding 2: auto-window changed the session lifecycle unnecessarily
Auto-window previously could trigger a new montage render/session even when valid materialized tiles,
semantic summaries, and a committed frame already existed. That made a presentation-only command
compete with ongoing render work and created another route for older levels to reappear.

The review keeps auto-window inside the committed presentation:

derive/apply valid semantic bounds immediately;

update the session’s level target/source;

let the backend-specific convergence strategy apply it;

refine histogram bins separately;

retain materialization, residency, viewport, and committed-frame history.

A presentation command should replace a presentation target, not the document/materialization session.

Finding 3: preview-finish duplicated whole generations
Histogram interaction emits previews and then a finish signal. When the finish carried exactly the
same levels, begin_level_presentation_update() still incremented the revision and marked the
montage stale again. On PyQtGraph that can repeat all CPU/item work after the user has already seen the
same result.

The session now:

increments revision only for a different numeric target;

continues an unsettled matching generation without revising it;

treats a settled matching target as a no-op;

retains a new target even when no tiles are currently active so later materialization inherits it.

This reduces redundant work and makes “finish” a durability/notification event rather than an
automatic redraw command.

Finding 4: benchmarks observed the wrong completion condition
The profile workflow originally waited for ordinary montage completion/visibility. During a level
loop, old tiles remain visible by design, so that condition could return before the current level
generation converged. The next loop iteration then superseded a partially drained target, producing
variable subsets and misleading timing.

The benchmark now waits for:

the current level revision;

no stale active tile presentations;

no pending current-generation level update;

active level-value coverage matching the target.

Its record includes settled status, revision, stale count, and active value count. This turns the
profile from a setter/callback benchmark into a semantic convergence benchmark.

Raw display.rendering_benchmarks scenarios still directly exercise backend/widget hot paths. Those
are valuable for cost comparison, but they intentionally bypass application-level admission unless a
handler is installed. Their names/reports should continue to distinguish raw backend work from the
profile workflow’s end-to-end result.

Finding 5: PyQtGraph level work was underreported
The tile stats historically counted an RGB re-window as an image update, but scalar
ImageItem.setLevels() often counted as “skipped.” Consequently a callback could touch many tile
items while tile_layer_level_updates remained zero and resource feedback saw one processed item.

The review now records the actual per-tile level operations for both direct and fallback PyQtGraph
paths. Observation count uses the largest exact overlapping work set rather than summing aliases such
as “created + updated” for the same item. This improves:

profile interpretation;

resource-governor per-item estimates;

detection of unbounded fallback work;

comparison with VisPy’s one/few uniform operations.

Finding 6: histogram refinement had good semantics but risky scheduling/integration
Good design
display/histogram_plot.py contains Qt-free sampling/binning helpers.

plot detail is separate from semantic level coverage;

generation/signature guards reject stale async results;

adaptive bins respond to zoom/range;

manual editing and auto-window are semantically richer than the original widget.

Repairs implemented
large auto-window applies known semantic bounds immediately rather than waiting for sampling/binning
in the GUI path;

detailed refinement is latest-only per view;

a stable background key prevents stale zoom requests from occupying the shared prefetch queue;

fallback futures are canceled/replaced rather than accumulating;

synchronous/auto results no longer get rejected by a stale active-request signature.

Remaining architecture problem
ImageView2D._bind_histogram_item() reaches into PyQtGraph internals:

manually assigns HistogramLUTItem.imageItem as a weak reference;

calls private _setImageLookupTable() when available;

repeatedly disconnects sigImageChanged to prevent duplicate recomputation.

This was introduced for a real reason: PyQtGraph’s public setImageItem() reconnects signals when
called repeatedly. But the current solution is version-sensitive and ties semantic histogram
ownership to widget internals.

Recommended destination:

HistogramModel / HistogramRefinementService  (Qt-free)
                    |
          HistogramInteractionController
                    |
          HistogramWidgetAdapter protocol
             /                         \
  PyQtGraph compatibility       future ArrayScope-owned
       adapter                    plot + level region
Do not “fix” this by keeping a full duplicate hidden image unless memory/CPU evidence supports it.
First isolate the compatibility calls and test supported PyQtGraph versions.

Finding 7: LOD is deliberately disabled, and the old implementation had deeper defects
Exact history
LOD was introduced by:

548901c05d2ab4c157cf7e5c7bcadc26a6af7663

Add LOD and (complex) shaders to VisPy renderer

2026-06-19 17:00 CEST

It was disabled by:

5a1ef86aa61d425ac0cf077b886c577df2f7d437

Remove synchronous LOD work from presentation commits

2026-06-19 18:48 CEST

The current selector still computes desired_tile_lod_factor; _selected_lod_factor() sets the
production policy to native-only, sets tile_lod_factor = 1, and returns one.

Why disabling was correct
The prototype built CPU pyramids from every loaded tile inside
snapshot_display_tile_payloads(), which is called from the UI presentation path. A local Python
3.13 microbenchmark in this review measured building through factor four at approximately:

512×512 float32: median ~5.1 ms per tile;

1024×1024 complex64: median ~38 ms per tile, with roughly 30–48 ms observed across runs.

These are only local CPU timings, but their order of magnitude is sufficient: multiplying by a visible
set before upload violates the GUI budget.

Other prototype defects
Transition identity churn. Changing factor changed every payload/residency identity.

Rebuild on repeat. Reduced levels were not retained as a reusable source cache.

Initial threshold flapping. The first implementation had no hysteresis.

Incompatible atlas storage. Reduced tiles/gutters changed dimensions while fixed slots assumed
one shape; padding/rebuilds could waste memory or expose wrong UV regions.

Backend mismatch. PyQtGraph consumes payload.image; changing primarily texture_data did not
guarantee a display benefit while shared session churn remained.

Wasted histogram work. A reduced histogram pyramid was built and then not used as the semantic
source.

Isotropic selector. The current function deletes tile_shape, takes the worse world/pixel axis,
and returns one factor, which can over-reduce an extreme aspect ratio.

Transition cost. Without adjacent-level retention, zooming across a threshold repeatedly causes
materialization/upload churn.

Selection quality
The basic goal—roughly one to two source texels per screen pixel and power-of-two levels—is reasonable.
The later hysteresis is also a reasonable start. The problem is not the concept; it is coupling demand
selection to synchronous materialization and incompatible residency.

Required replacement
ADR 0041 separates:

LodPlanner (desired quality)
    -> LodMaterializer / source pyramid cache (background or source-provided)
    -> LOD-compatible ResidencyStore (page/array class by level/shape/format)
    -> backend presentation transition retaining the previous usable level
Production should remain native-only until the acceptance gates pass. The review makes that policy
observable through desired factor, applied factor, policy, and reason instead of leaving users to
infer that LOD “never activates.”

Architectural diagnosis
The project’s fundamental choices are mostly sound
These choices should remain:

immutable/explicit semantic state;

operation-declared capabilities and region behavior;

Qt-free planning/evaluation models;

semantic display frames and value sources;

separate materialization and presentation identity;

explicit memory/cost/resource policy;

retained last valid frame;

progressive visible work with stale-result guards;

backend adapters/mechanics separated from scientific meaning.

The control plane has crossed the complexity threshold
MontageRenderSession currently owns or tracks:

pending/active tile compute;

stage keys, waits, attached requests, and stage values;

rendered/materialized/presented tiles;

dirty payloads, pending upserts/removals, and tile source identities;

viewport/near-residency sets;

semantic level statistics and pending level tiles;

desired/applied level values, per-tile revisions/value counts, stale count;

structure/payload/visibility/histogram/viewport revisions;

LOD desired/applied policy;

canvas and dirty rectangles;

display committed/final commit state.

That is several state machines sharing one mutable record. The large renderer then decides when each
one advances. This is why adding an acknowledgement field or timer can affect auto levels, viewport,
benchmarks, or residency.

Target extraction
The next refactor should introduce focused, Qt-free owners:

class PresentationGenerationTracker:
    def request(self, target, active_ids) -> GenerationDecision: ...
    def acknowledge(self, revision, accepted_ids) -> None: ...
    def retarget_active(self, active_ids) -> None: ...
    def snapshot(self) -> GenerationSnapshot: ...

class TileAdmissionQueue:
    def upsert(self, work_items) -> None: ...
    def retarget(self, viewport, hover, revision) -> None: ...
    def admit(self, *, item_cap, byte_cap, deadline_ns) -> AdmissionBatch: ...

class LevelConvergenceStrategy(Protocol):
    def build_work(self, target, frame, backend_state) -> Iterable[TileWork]: ...
    def acknowledge(self, report) -> AcceptedWork: ...
Implementations:

PyQtGraphProgressiveLevelStrategy: stale active tiles become ordinary priority upserts;

VisPyUniformLevelStrategy: update active visuals/pages and acknowledge a backend transaction.

Stage waits/result fan-in should be another owner; LOD planning/materialization/residency should not be
added to the generation tracker.

Migration rule
Do not rewrite montage_renderer.py in one change. Extract one model with characterization/property
tests, route the existing code through it, then remove the old fields. At every commit:

at least PyQtGraph remains runnable;

retained frame behavior remains;

the same semantic conformance tests pass;

deterministic work counters do not regress;

compatibility shims contain no behavior.

Recommended test strategy
Semantic conformance matrix
Run each scenario against a small model fixture and both backend strategies:

Scenario	Required semantic result	PyQtGraph physical expectation	VisPy physical expectation
Initial target	all active tiles at target	bounded item/image work	payload + uniforms
Level-only target	latest target settles	progressive CPU/item operations	uniform update, zero texture upload
Rapid A→B→C	C settles; A/B reports ignored	unfinished work reprioritized/superseded	latest uniform wins
Deferred visible upsert	remains stale/pending	old item retained	old visual retained
Auto then user	user source/target wins	drain user generation	user uniform wins
User then auto/revert	source-rank rules respected	bounded convergence	uniform convergence
Viewport change mid-generation	only current active coverage gates settled	queue retargeted	active visual set retargeted
Tile enters active set	inherits latest target	admitted at latest target	visual receives latest uniform
Tile leaves active set	no longer gates settled	retained/hidden by policy	visibility/residency policy
Close/context loss	no late mutation	callbacks canceled	context recovery explicit
Property/state-machine tests
Generate sequences of:

target changes;

active-set additions/removals;

partial/empty/stale acknowledgements;

source replacements;

cancellations and session retargets.

Assert:

acknowledged current-revision coverage is a subset of active materialized coverage;

settled implies every active presented tile has the current target;

stale reports never advance a newer target;

no target revision goes backward;

reissuing the same settled target performs no physical work;

removing an active tile cannot increase stale current coverage;

adding a tile cannot be considered current until it is presented at the target.

Performance gates
Use deterministic counters first, real timings second:

max admitted items/bytes per callback;

actual processed items, not aliases or visibility count;

event-loop max/p95/p99 gaps;

target-to-first-change and target-to-settled;

stale count over time;

CPU window ms/tile for PyQtGraph;

uniform and texture-upload counts for VisPy;

threshold transition materialization/upload/residency for future LOD;

cancellation work and reusable completed work.

Roadmap recommendation
The roadmap is updated in this review:

N5 backend-aware presentation convergence is the immediate correctness gate.

N4 histogram refinement discipline remains active, with PyQtGraph adapter isolation added.

N6 control-plane extraction precedes unified planner work.

N7 explicit native-only LOD policy/design prevents accidental revival of synchronous work.

X1 unified frame planning, X2 backend composition, and X3 interaction ownership continue only after
the state machines stabilize.

X4 includes real hardware/residency evidence and the eventual ADR 0041 multi-resolution path.

No new renderer rewrite or product feature should run in parallel with N5/N6 unless it is isolated from
these contracts.

ADR changes
ADR 0040 records one semantic target
with backend-specific convergence and explicit acknowledgement.

ADR 0041 records native-only
production LOD and the required three-stage replacement.

ADR 0039 is refined so “presentation-only does not invalidate residency” is not misread as
“PyQtGraph never needs redraw.”

Code changes made during this review
Commit	Change
103ab67	Preserve the supplied uncommitted v30 histogram/benchmark work as an auditable checkpoint.
e3eee7b	Distinguish accepted tile upserts from retained visibility.
4476104	Keep auto-window updates inside the committed presentation/session.
ffc6c80	Wait for level-generation convergence in the profile loop.
ba758f5	Coalesce histogram refinement and keep semantic auto-window off the GUI refinement path.
e944609	Expose desired/applied LOD policy and reason.
0eb19ff	Keep progressive commits on the latest level target and supersede stale auto work.
a523f14	Remove obsolete duplicate level acknowledgement fields.
516ab03	Report actual PyQtGraph per-tile level work to diagnostics/feedback.
a31ff0f	Reuse identical active generations and avoid a duplicate finish redraw.
Validation evidence
Focused validation completed during the review includes:

98 targeted montage/backend/session regressions passed before later cleanup;

25 histogram-focused tests passed;

53 LOD/runtime/session/profile tests passed with 2 environment-dependent skips;

14 profile-workflow tests passed with 2 skips in an earlier focused run;

additional regressions for identical generation reuse and PyQtGraph work counters passed;

python -m compileall -q arrayscope and git diff --check passed during commits.

Final broad-suite results and environment limitations are recorded in the handoff accompanying the
review artifacts. Real Qt/VisPy/OpenGL/platform manual regression was not available here and remains a
release gate.

Unfinished work and implementation guidance
The review intentionally does not perform the high-risk extraction or enable LOD. The required logic
is fully specified in ADRs 0040/0041 and the roadmap. In particular:

move current level fields/counts from MontageRenderSession into a tested generation tracker;

move priority/budget admission out of semantic session state;

keep the old frame drawable until accepted replacement acknowledgement;

implement separate PyQtGraph and VisPy strategies rather than capability branches spread through
widgets/renderers;

isolate histogram widget compatibility calls;

split the profile tool by scenario action, settlement probe, record schema, and report rendering;

design/source an async LOD pyramid cache and compatible storage pools before applying factor >1;

retain native and adjacent LOD through transitions and keep scientific values exact;

run the complete test matrix and real-platform manual traces before merging as a release candidate.

Final recommendation
Do not throw away ArrayScope. Throw away the assumption that one mutable session plus timers can safely
coordinate every rendering lifecycle, and throw away the original synchronous LOD path.

The best route is a controlled internal rewrite of the control plane, preserving the semantic core,
backend mechanics, caches, and tests. Make generation/acknowledgement explicit first. Once local fixes
are local again, unified tiled planning and a composed backend shell become much safer and more likely
to deliver the project’s actual goal: fast, understandable array inspection from low-power systems to
high-end render hardware.
