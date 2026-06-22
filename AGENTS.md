# AGENTS.md

ArrayScope is a Python/Qt n-dimensional scientific-array viewer. Preserve its defining property: a small user action should produce an understandable view quickly, while expensive work remains bounded and observable.

## Read the minimum useful context

Use this order rather than scanning every historical note:

1. `docs/mission.md` for scope and product principles.
2. `docs/current-state.md` for maturity and known risks.
3. `docs/roadmap.md` for active work and exit gates.
4. `docs/architecture.md` for ownership and invariants.
5. The relevant deep dive in `docs/architecture/`.
6. `docs/decisions/README.md` and the specific ADR when a decision’s rationale matters.
7. `docs/ideas.md` only for exploratory/future work.

`docs/archive/` is historical evidence, not live direction. `docs/reviews/` contains dated assessments that may be superseded.

## Environment

The maintained local workflow uses the `arrayscope` conda environment through direnv:

```bash
PATH=~/miniconda3/bin:$PATH direnv exec . <command>
```

When dependencies change:

```bash
PATH=~/miniconda3/bin:$PATH direnv exec . conda env update -f environment.yml --prune
```

Headless GUI tests normally use `QT_QPA_PLATFORM=offscreen`. VisPy/OpenGL tests still need separate real-hardware runs before performance or Wayland claims are accepted.

## Architecture rules

- `ViewState` and document objects own semantic state; widgets mirror state and emit intent.
- Keep GUI callbacks thin. Do bounded work, publish progress, and reschedule the remainder.
- Keep authoritative identities separate: document, semantic target, viewport, presentation, and physical residency.
- Separate materialization identity from presentation identity. Levels/LUT changes must not imply new source pixels.
- Camera-only changes must not restart array evaluation.
- Requested, materialized, resident, and presented are distinct lifecycle states.
- The committed frame owns coordinate/value semantics; placeholders never do.
- Operations declare capabilities and region behavior. Do not add registered-operation type switches to render/slab code.
- Background workers consume immutable snapshots and return values. They do not mutate live Qt-owned coordinators.
- Backend code owns scene/texture mechanics, not the meaning of ROI hits, levels, frame identity, or viewport intent.
- Compatibility shims, wrappers, or other "quick fixes" must be avoided. Problems must be solved at the source.

## Change policy

Prefer small, reviewable fixes. Add an ADR only for a durable architecture, API, packaging, test-strategy, or major UX decision. Update live docs when behavior, maturity, ownership, or roadmap status changes; move obsolete process notes to the archive instead of layering another contradictory section on top.

Every new array operation needs shape/value coverage. Every visible feature needs an interaction/smoke test where practical. Performance work needs deterministic work counters plus real timing evidence; wall-clock headless timings alone are not a GPU claim.

## Validation

Run focused tests first, then the broadest affordable layer:

```bash
pytest -q tests/core tests/operations
QT_QPA_PLATFORM=offscreen pytest -q tests/display tests/window
QT_QPA_PLATFORM=offscreen pytest -q tests/ui tests/app
```

Also run:

```bash
python -m compileall -q arrayscope
ruff check arrayscope tests --select F821,E9
git diff --check
```

For rendering/UI changes, perform the relevant manual checks from `docs/testing/manual-regression.md` and record the backend, OS/session type, dataset shape/dtype, and diagnostics trace.

## Handoff

State what changed, tests and manual checks run, remaining risks, and any follow-up that belongs in `docs/roadmap.md` or `docs/ideas.md`.
