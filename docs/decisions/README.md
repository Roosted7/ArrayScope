# Architecture decisions

Decision records capture durable rationale. They are not a chronological backlog. Read the [architecture overview](../architecture.md) first, then the smallest relevant group below.

Status terminology:

- **Implemented**: the decision’s central contract is present and tested.
- **Partly implemented**: key pieces exist, but the stated destination is incomplete.
- **Historical foundation**: still explains the code, though later decisions refine the mechanism.
- **Experiment**: records an evaluated direction rather than a final default.

## State, slicing, operations, and UI boundary

| ADR | Topic | Current status |
|---|---|---|
| [0001](0001-view-state.md) | View state | Implemented; expanded since acceptance. |
| [0002](0002-slice-engine.md) | Slice engine | Historical foundation. |
| [0003](0003-dimension-operations.md) | Dimension operations | Implemented and generalized into the operation stack. |
| [0004](0004-ui-boundary.md) | UI/state boundary | Implemented principle; ongoing enforcement. |
| [0005](0005-operation-pipeline.md) | Operation pipeline | Implemented. |
| [0006](0006-operation-ui-and-recipes.md) | Operation UI/recipes | Implemented. |
| [0007](0007-window-level-mode.md) | Window/level mode | Historical foundation, refined by later presentation ADRs. |
| [0008](0008-live-image-profiles.md) | Live profiles | Implemented. |
| [0009](0009-profile-dock.md) | Profile panel | Implemented, later folded into managed panels. |
| [0010](0010-ui-polish-settings-theme.md) | Settings/theme | Implemented. |
| [0011](0011-progressive-interaction-polish.md) | Interaction polish | Historical foundation; parts remain roadmap work. |

## Inspection, geometry, panels, and viewport

| ADR | Topic | Current status |
|---|---|---|
| [0013](0013-roi-inspection-workflows.md) | ROI inspection | Implemented core workflow. |
| [0014](0014-profiles-and-montage.md) | Profiles/montage roles | Implemented; range behavior recently extended. |
| [0015](0015-coordinate-geometry-and-viewport.md) | Geometry/viewport | Implemented contract. |
| [0017](0017-managed-panels-and-wayland.md) | Managed panels/Wayland | Implemented with platform regression needs. |
| [0018](0018-viewport-fit-toggle.md) | Fit/1:1 behavior | Implemented and recently refined. |

## Evaluation, planning, cache, and resource policy

| ADR | Topic | Current status |
|---|---|---|
| [0012](0012-lazy-slab-evaluation.md) | Lazy slab evaluation | Implemented and superseded in detail by planner/stage work. |
| [0016](0016-evaluation-scheduler-and-memory-budget.md) | Evaluation scheduling/memory | Historical foundation. |
| [0020](0020-operation-cost-and-fft-backend.md) | Cost model/FFT backend | Implemented. |
| [0021](0021-scheduler-v2-cost-aware-rendering.md) | Cost-aware rendering | Partly implemented; visible scheduling still converging. |
| [0023](0023-memory-policy-and-developer-diagnostics.md) | Memory/diagnostics | Implemented foundation. |
| [0025](0025-operation-capabilities-and-region-contracts.md) | Operation contracts | Implemented. |
| [0026](0026-runtime-region-planner.md) | Region planner | Implemented. |
| [0027](0027-in-memory-stage-cache.md) | Stage cache | Implemented. |
| [0028](0028-runtime-operation-optimizer.md) | Optimizer | Implemented. |
| [0029](0029-stage-first-rendering.md) | Stage-first rendering | Partly implemented. |
| [0034](0034-compute-policy-and-stage-warmup.md) | Compute/stage warmup | Implemented foundation. |
| [0035](0035-resource-governor-feedback-control.md) | Resource governor | Implemented foundation; evidence/tuning ongoing. |

## Presentation, montage, and backends

| ADR | Topic | Current status |
|---|---|---|
| [0019](0019-tiled-montage-renderer.md) | Tiled montage | Implemented foundation. |
| [0022](0022-stable-progressive-montage-rendering.md) | Stable progressive montage | Implemented, refined by repair work. |
| [0024](0024-canvas-preserve-transaction.md) | Canvas preservation | Implemented. |
| [0030](0030-display-presentation-boundaries.md) | Presentation boundaries | Implemented. |
| [0031](0031-render-presentation-model.md) | Presentation model | Implemented. |
| [0032](0032-semantic-montage-histograms.md) | Semantic levels/histograms | Implemented contract. |
| [0033](0033-responsive-montage-display-upload.md) | Responsive upload | Partly implemented; callback budgets remain active work. |
| [0036](0036-vispy-rendering-backend-experiment.md) | VisPy experiment | Experiment completed; led to 0037/0038. |
| [0037](0037-first-class-vispy-tiled-renderer.md) | VisPy tiled renderer | Implemented experimental backend. |
| [0038](0038-render-backend-composition.md) | Backend composition | Partly implemented via adapters; inheritance remains. |
| [0039](0039-unified-image-surface-and-deadline-scheduler.md) | Unified surface/scheduler | Current target, partly implemented. |

## Adding or changing a decision

Create an ADR only when future contributors need to preserve a choice about architecture, public API, packaging, testing strategy, or major UX behavior. Record context, decision, alternatives, consequences, and migration. When a later ADR supersedes part of an earlier one, update this index and cross-link both records rather than rewriting history.
