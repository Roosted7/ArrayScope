# ArrayScope documentation

ArrayScope documentation is organized so a reader can stop at the first useful level instead of reading a chronological development diary.

## Level 1 — orient yourself

- [Project README](../README.md): install, launch, and feature overview.
- [Mission](mission.md): who ArrayScope serves, what it promises, and what it will not become.
- [Current state](current-state.md): a candid maturity map and the most important risks.
- [Project status supplement](project-status.md): supplemental v28 status snapshot restored from the audit notes.
- [Roadmap](roadmap.md): the active sequence of measurable gates.

These documents should answer most product and planning questions.

## Level 2 — understand the system

- [Architecture overview](architecture.md): ownership, identities, data flow, and non-negotiable invariants.
- [State and operations](architecture/state-and-operations.md): `ViewState`, document revisions, operation planning, stages, and caches.
- [Rendering](architecture/rendering.md): semantic presentations, frames, geometry, raster/tiled storage, and backends.
- [Scheduling and memory](architecture/scheduling-and-memory.md): visible work, montage sessions, budgets, feedback, and cancellation.
- [Interaction and UI](architecture/interaction-and-ui.md): widget/state boundaries, viewport behavior, ROI/profile interaction, and panels.

Read only the deep dive related to the change being made.

## Level 3 — rationale and evidence

- [Architecture decisions](decisions/README.md): accepted decisions, grouped by topic and implementation status.
- [Testing strategy](testing/strategy.md): what each test layer proves and what it cannot prove.
- [Manual regression](testing/manual-regression.md): compact release/hardware checks.
- [Reviews](reviews/README.md): dated audits and trace analyses.
- [Proposals](proposals/README.md): work that is designed but not accepted as current direction.

## Product references

- [ArrayShow reference](references/ArrayShow.md)
- [ArrayView reference](references/ArrayView.md)
- [Viewer comparison reference](references/viewer-comparison.md)
- [Comparative assessment](comparison.md)

References are sources of lessons, not specifications. ArrayScope should adopt useful interaction patterns without inheriting another project’s global state, feature sprawl, or rendering bottlenecks.

## Historical material

[`archive/`](archive/README.md) contains phase context, old roadmap snapshots, and dated manual checklists. It remains searchable because it explains why code exists, but it must not be used as the current backlog.

## Source-of-truth order

When documents disagree, use this order:

1. Current tested behavior and public code contracts.
2. Accepted ADRs that have not been explicitly superseded.
3. `current-state.md`, `architecture.md`, and the active roadmap.
4. Dated reviews/proposals.
5. Archived phase notes.

A contradiction in the first three levels is a documentation defect and should be fixed with the code change that exposes it.
