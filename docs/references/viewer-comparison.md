# ArrayScope, ArrayShow, and ArrayView

**Comparison date:** 2026-06-22. Sources are pinned in the companion reference documents.

## Different aims

| Project | Primary aim | Natural environment |
|---|---|---|
| ArrayShow (`arrShow` 0.35) | Immediate, scriptable MATLAB workspace inspection with strong complex/MRI conventions | MATLAB figures/workspace |
| ArrayView 0.26.3 | A polished multidimensional viewer available across shell, scripts, notebooks, VS Code, local, and remote workflows | Browser canvas backed by Python server/stdio |
| ArrayScope v28 | A native local Python/Qt scientific array viewer with explicit operation semantics, bounded lazy evaluation, and direct rendering | Python/NumPy desktop and IPython Qt |

They overlap in slice browsing and scientific inspection, but they are not substitutes with identical
constraints. ArrayShow optimizes MATLAB immediacy, ArrayView optimizes reach and mode breadth, and
ArrayScope currently optimizes local semantic correctness and large-array orchestration.

## Capability and architecture matrix

| Area | ArrayScope | ArrayShow | ArrayView |
|---|---|---|---|
| One-call array view | Good | Excellent | Good |
| Public scriptable viewer handle | Internal window exists; stable public API is incomplete | Strong handle methods | `ViewHandle` and multi-environment launch API |
| Native direct array display | Yes | Yes, MATLAB graphics | Server extracts/renders and frontend displays transported pixels |
| N-dimensional slices | Yes | Yes | Yes |
| Complex/MRI conventions | Strong and operation-aware | Strong, mature | Broad display modes |
| Lazy operation pipeline | Strong differentiator: region plans, stages, optimizer, budgets | Primarily direct data operations | Server render pipeline, no equivalent visible operation document |
| Montage/large view | Progressive bounded tiled/raster strategies | Traditional figure image arrays | Mosaic and many modes in frontend/server |
| Compare/multiview | Limited scaffolding; roadmap gap | Linked relatives, not modern compare breadth | Major strength: compare, diff, overlay, multiview, qMRI |
| ROI/profile/statistics | Strong core, still polishing backend parity | Mature basics | Broad tools and contextual islands |
| Memory/scheduler diagnostics | Strong differentiator | Minimal by modern standards | Caches/tests, less explicit native GPU/Qt scheduling need |
| Jupyter/editor/remote reach | Basic IPython Qt; significant gap | MATLAB workspace | Major strength |
| UI minimalism/discoverability | Progressive docks and command palette, still mixed shell | Dense traditional MATLAB UI | Major strength |
| Architecture risk | Large transition-era Qt/render classes; dual backend migration | Monolithic handle/global registry/fixed callbacks | Giant frontend/launcher, mode combinatorics, global/dual state |

## Where ArrayScope is ahead

### Explicit scientific and presentation semantics

ArrayScope has first-class `ViewState`, operation steps, axis metadata, region plans, stage identities,
semantic frame/value sources, and raster/tiled presentation contracts. This makes correctness under
shape-changing operations, stale async work, complex display, ROI/profile demand, and backend changes
more tractable than deriving state from GUI objects.

### Bounded large-array work

Memory policy, conservative cost estimates, lane-specific scheduling, stage singleflight, bounded
caches, progressive presentation, stale guards, and diagnostics are deeper than the inspected alternatives.
This is valuable only if end-to-end latency is continuously measured; sophistication is not a license
for a slower first frame.

### Direct native path

For a local NumPy workflow, ArrayScope can avoid server serialization and PNG transport. PyQtGraph and
VisPy can receive array/texture data directly, and semantic values remain available without a round
trip.

## Where ArrayScope is behind

### Invocation and public API

ArrayView is much easier to reach from editors, notebooks, remote sessions, and other languages.
ArrayShow offers a mature scriptable object and linked instances. ArrayScope's callable module is
convenient, but the public window/session control contract is underdeveloped.

### Compare, linked views, and workflow breadth

ArrayView's comparison modes and ArrayShow's relatives expose common scientific workflows that
ArrayScope mostly leaves to separate windows or internal scaffolding. A focused compare/linked-session
MVP is more valuable than adding another isolated display effect.

### Product polish and command consistency

ArrayView's command registry, help, mode badges, dynamic islands, and consistency matrix are a better
model for discoverability. ArrayScope has progressive docks and a command palette, but commands/menus/
help are not yet one generated system.

### Release discipline

ArrayView's inspected source is tagged and packaged as 0.26.3. ArrayScope's package metadata remains
0.0.1 despite a v0.7.0 tag and extensive later history. This is a basic trust gap and a release blocker.

## Concrete adaptations

### Adopt now

- One command registry for palette, keyboard help, menus, and compatibility/cost metadata.
- A documented mode/transition matrix before compare and linked-view expansion.
- A stable scriptable `ArrayScopeSession`/window handle.
- Minimal canvas-first presentation and contextual controls derived from state.
- Reproducible competitor/source references and release/version guards.

### Build after the current performance gate

- Linked session groups with explicit fields and compatibility rules.
- Two-array compare: linked cursor/slice, side-by-side, difference, phase difference, and overlay.
- Jupyter/VS Code adapters that control the native session or launch a local process through a small
  protocol.
- Axis labels/physical coordinates and metadata-aware profiles/ROI/export.

### Do not copy blindly

- ArrayShow's global `asObjs`, monolithic figure ownership, fixed panel geometry, or callback timing
  workarounds.
- ArrayView's giant frontend/launcher, indefinite dual state, server/browser transport for local-only
  display, or dozens of boolean-composable modes before lifecycle tests exist.
- Feature parity as a roadmap. ArrayScope should be the most dependable native Python scientific array
  inspector, not the union of two other viewers.

## Strategic recommendation

Finish release hygiene, first-frame/event-loop evidence, and backend composition first. Then invest in
the two gaps that most improve real workflows: a public linked-session API and a focused compare mode.
Use ArrayShow's immediacy and ArrayView's reach/discoverability as product standards while preserving
ArrayScope's strongest technical differentiators: explicit semantics, direct local rendering, and
bounded operation-aware evaluation.
