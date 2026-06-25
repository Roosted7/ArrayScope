# Current state

**Snapshot:** ArrayScope v30 rendering-consistency review branch, reviewed on 2026-06-24. The supplied
v30 histogram/benchmark work is preserved in commit `103ab67`; review fixes are separate commits on
top.

ArrayScope has a strong semantic/evaluation foundation and a recently extracted rendering control
plane. The project is not on the wrong overall path; the immediate v30 risk was that recent
optimization work crossed too many queues, timers, backend contracts, and presentation identities.
N6 moved presentation-generation, tile-admission, level-convergence, and stage-fan-in state machines
into Qt-free models so local rendering fixes are easier to reason about.

## Maturity map

| Area | State | Notes |
|---|---|---|
| Basic launch, slicing, image/line display | Established | Broad automated coverage; platform/Qt integration still needs real-system checks. |
| Dimension roles, ranges, flips/FFT shift | Established with recent change | Keep interaction regressions across cropped/ranged axes. |
| Reversible operation document/recipes | Established | Optimizer preserves public step history. |
| Region planning, stage cache, cost/memory estimates | Substantial | Strong Qt-free coverage; workload heuristics need field evidence. |
| Profiles and ROI inspection | Substantial | Shared semantics exist; pointer/drag lifecycle is not fully backend-neutral. |
| Histogram and window/level | Substantial, under stabilization | Semantic auto bounds and latest-only refinement exist; PyQtGraph binding remains brittle. |
| Progressive montage | Advanced, stabilizing | Core control-plane state machines are extracted; renderer/session orchestration is still large. |
| PyQtGraph backend | Production fallback | Correctly requires progressive CPU/item convergence for some level changes; large item counts remain costly. |
| VisPy backend | Experimental | Persistent textures/shader levels are promising; hybrid widget inheritance and real-hardware evidence remain gaps. |
| LOD | Explicit native-only production policy | Demand selection records desired/applied factor, per-axis texels, policy, and reason; applied factor remains 1 until async compatible residency exists. |
| Diagnostics/benchmarks | Good internal base, recently corrected | Completion and PyQtGraph level-work counters now reflect convergence/work rather than visibility/image replacement. |
| Documentation/ADRs | Updated for v30 findings | ADR 0040 and 0041 define level convergence and LOD prerequisites. |

## What is working well

### Semantic and physical identities are mostly explicit

`ViewState`, `ArrayDocument`, operation plans, display geometry, committed frames, level sources, tile
payloads, and memory policy are mostly Qt-free. Materialization identity is separated from ordinary
levels/LUT state. Requested, materialized, resident, and presented are named lifecycle states.

### Expensive work has real policy levers

The project has operation capabilities, region plans, cost estimates, stage materialization and
singleflight, separate caches, cancellation/supersession keys, lane worker policy, GUI callback
budgets, latency feedback, and a resource governor. These are worth preserving.

### Recent performance work improved important hot paths

Active-plus-latest scheduling preserves useful visible progress. Stage-plan/candidate caching, direct
tile deltas, stable texture identity, retained residency, dynamic tile priority, and separation of
cold upload from warm visibility/rebind are sensible optimizations. VisPy level changes can remain
uniform-only, while PyQtGraph can reuse the same priority/admission queue for CPU redraws.

### Tests increasingly protect lifecycle contracts

The suite now covers stale delta rejection, accepted-upsert acknowledgement, rapid level
supersession, one-tile batches, auto-window within a committed session, zero-upload VisPy level
updates, native-only LOD diagnostics, and benchmark convergence state.

## Correctness repairs in this review

- Backend reports now distinguish drawable retained tiles from upserts actually accepted in a commit.
- PyQtGraph progressive level generations retry deferred tiles and settle only after every active tile
  acknowledges the latest target.
- Auto-window no longer replaces a useful committed montage session.
- A concrete level command supersedes older automatic work still attached to that session.
- Benchmark completion uses target revision/stale coverage rather than “all tiles are visible.”
- Large auto-window bounds apply immediately while detailed histogram refinement remains latest-only.
- Per-view histogram background requests are coalesced instead of accumulating stale work.
- LOD diagnostics expose desired versus applied factor and the native-only reason.
- PyQtGraph scalar/RGB per-tile level work is counted accurately for diagnostics and resource feedback.
- Obsolete duplicate level-acknowledgement fields were removed.
- Montage level convergence now has a single session snapshot for target revision, stale active tiles,
  pending target work, and settled state. Profile and rendering benchmark records expose those fields
  beside backend-specific physical work counters.

## Material risks

### 1. Renderer/session orchestration remains large

`window/montage_renderer.py` and `window/montage_session.py` are still substantial orchestration
modules. N6 removed ownership of level generation, convergence strategy, admission caps, and stage
fan-in from the session, but the renderer still coordinates Qt timers, committed frames, overlays,
side panels, diagnostics, and backend commits. Future X1/X2 work should reuse the extracted models
rather than growing another scheduler.

### 2. Semantic parity is being confused with mechanical uniformity

PyQtGraph and VisPy must agree on target levels, values, source ranks, revisions, and completion. They
must not be forced into one physical update method. PyQtGraph may need many bounded CPU/item updates;
VisPy can update resident visuals through uniforms. ADR 0040 makes that distinction durable.

### 3. Histogram ownership is adapter-isolated but still version-sensitive

PyQtGraph histogram rebinding is isolated in `display/backends/pyqtgraph/histogram_adapter.py`, which
owns `HistogramLUTWidget.setImageItem`, private `HistogramLUTItem.imageItem` rebinding, lookup-table
refreshes, region refreshes, and `sigImageChanged` cleanup. This keeps `ImageView2D` out of PyQtGraph
internals, but the adapter still depends on private API shape and needs explicit coverage when
PyQtGraph changes.

### 4. LOD is intentionally unavailable, not merely failing to trigger

The selector runs, records per-axis source-texel demand, and the native-only policy applies factor
one. The old implementation built CPU pyramids in a GUI commit path and mixed incompatible tile
dimensions with fixed atlas assumptions; the production callable path is gone and guarded. ADR 0041
defines the required split before non-native LOD can be enabled.

### 5. Timer interactions still need audit discipline

Debounce, commit, warm-residency, prefetch, histogram, stage-wait, and overlay timers are useful
rescheduling tools. Montage commit/result fan-in/stage-wait/priority-retarget callbacks now carry
explicit session/revision work tokens; future timer paths must keep that pattern and avoid owning
semantic order.

### 6. Hardware evidence remains incomplete

Headless tests prove models and deterministic work counters. They do not prove OpenGL upload latency,
texture limits, Wayland behavior, high-DPI pointer mapping, frame pacing, or interaction feel.

## Current direction

Do not discard the operation/evaluation core, display models, resource policy, extracted
control-plane models, or backend mechanics. Do discard the unsafe synchronous LOD route and stop adding
cross-cutting behavior to the session and renderer. The next architecture steps are unified frame
planning and backend composition; non-native LOD waits for the ADR 0041 async materialization and
compatible-residency gates.

The ordered acceptance gates are in the [roadmap](roadmap.md). Full evidence and recommendations are
in [the v30 rendering-consistency audit](reviews/v30-rendering-consistency-audit.md).
