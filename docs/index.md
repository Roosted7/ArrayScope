# Documentation index

Organized by the question you're asking. Stop at the first useful level.

## "What should I work on, and when is it done?"

- [**The queue**](queue.md) — the only active, ordered work list, with exit
  gates and the performance bars. If another doc claims to order work, that
  doc is stale.
- [Roadmap](roadmap.md) — why this order: Now / Next / Later / explicitly
  not now.
- [Mission](mission.md) — who ArrayScope serves, the product promise, and
  what it will not become.

## "How do we work here?"

- [Ground rules](ground-rules.md) — standing law: pixels are the gate, one
  owner per decision, no silent fallbacks, repro-first.
- [Testing](testing/README.md) — the rings, what runs in CI vs by hand, the
  defect→ring law, environment facts and harness commands.
- [Areas](areas.md) — how to split parallel work, the chokepoint files, ADR
  numbering and integration conventions.
- [Graveyard](graveyard.md) — rejected approaches with evidence. **Read
  before starting any performance or scheduling experiment.**

## "How does the system work?"

- [Architecture overview](architecture.md) — ownership, identities, data
  flow, non-negotiable invariants.
- Deep dives: [state and operations](architecture/state-and-operations.md),
  [rendering](architecture/rendering.md),
  [scheduling and memory](architecture/scheduling-and-memory.md),
  [interaction and UI](architecture/interaction-and-ui.md).
- [Invocation](invocation.md) — launch routes and wrappers.
- [Current state](current-state.md) — maturity snapshot and material risks.

## "Why is it this way?"

- [Architecture decisions](decisions/README.md) — accepted ADRs (0001–0056).
- [Proposals](proposals/README.md) — designed but not (yet) accepted
  direction; includes the
  [tensor-engine endpoint](proposals/tensor-engine-endpoint.md) and the
  [GPU program plan](proposals/gpu-engine-plan.md).
- [Redesign record](redesign/README.md) — the closed 2026-07 program, its
  [retrospective](redesign/retro-2026-07.md), and the live defect dossiers.
- [Reviews](reviews/README.md) — dated audits and trace analyses.
- [References](references/) and [comparison](comparison.md) — ArrayShow/
  ArrayView lessons; sources of patterns, not specifications.

## Historical material

[`archive/`](archive/README.md) and
[`redesign/archive/`](redesign/archive/) explain why code exists; never use
them as the backlog. [Ideas](ideas.md) holds exploratory/future material
only.

## Source-of-truth order

When documents disagree: (1) tested behavior and public code contracts;
(2) accepted ADRs not explicitly superseded; (3) `queue.md`, `roadmap.md`,
`architecture.md`, `ground-rules.md`; (4) dated reviews/proposals/dossiers;
(5) archives. A contradiction in the first three levels is a documentation
defect — fix it with the code change that exposes it.
