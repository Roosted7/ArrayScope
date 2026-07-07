# AGENTS.md

ArrayScope is a Python/Qt n-dimensional scientific-array viewer. Preserve its defining property: a small user action should produce an understandable view quickly, while expensive work remains bounded and observable.

## Read the minimum useful context

Use this order rather than scanning every historical note:

1. `docs/mission.md` for scope and product principles.
2. `docs/current-state.md` for maturity and known risks.
3. `docs/roadmap.md` for active work and exit gates. While the redesign is
   in flight, `docs/redesign/README.md` owns the active queue and ground
   rules — read it before touching scheduling, rendering, or LOD code.
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
- Rendering orchestration state lives on `window.renderer` (`RenderOrchestrator`), never flat on the
  window. Orchestration code reaches window services via `self.win`; the window exposes only the thin
  rendering API and read-only presentation properties (ADR 0045).
- Deferred callbacks and timers carry both a generation guard and a Qt receiver context; a timer must
  not be able to outlive its window, and a guard must not be able to accept a stale generation.
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

Test-suite rules:

- Never load a production module with `spec_from_file_location` and install it in `sys.modules`;
  plain imports only. Duplicated module objects cause order-dependent identity failures.
- Memory-policy budgets are pinned to a deterministic system snapshot in `tests/conftest.py`; use the
  `real_system_memory` marker only when a test intentionally exercises host sampling.
- Fake windows used with orchestrator methods model the composition (`fake.win = fake`); prefer real
  `MontageRenderSession`/`ViewState` objects over `SimpleNamespace` stand-ins.
- Prefer driving the real pipeline and asserting deterministic work counters and committed-frame
  semantics over monkeypatching orchestration internals.
- The suite runs in parallel by default (`pytest-xdist`, configured in `pyproject.toml`). Never assert
  on a *fixed* wait window — e.g. `QTest.qWait(220)` after launching background work, or a short
  `qtbot.waitUntil(..., timeout=250)`. Those pass only on an idle CPU and flake under parallel load.
  Wait on the actual signal/condition with a generous timeout instead. See
  `docs/testing/strategy.md` (Parallel execution) for the worker model and per-worker isolation.

## Validation

The suite is parallel by default (`-n auto`, capped at half the cores; see `docs/testing/strategy.md`).
Run focused tests first, then the broadest affordable layer:

```bash
pytest -q tests/core tests/operations
QT_QPA_PLATFORM=offscreen pytest -q tests/display tests/window
QT_QPA_PLATFORM=offscreen pytest -q tests/ui tests/app
```

Add `-n 0` to any command to run serially when debugging (clean tracebacks, working `-s`/`pdb`):
`pytest -q -n 0 tests/ui/test_foo.py::test_bar`.

Also run:

```bash
python -m compileall -q arrayscope
ruff check arrayscope tests --select F821,E9
git diff --check
```

For rendering/UI changes, perform the relevant manual checks from `docs/testing/manual-regression.md` and record the backend, OS/session type, dataset shape/dtype, and diagnostics trace.

## Handoff

State what changed, tests and manual checks run, remaining risks, and any follow-up that belongs in `docs/roadmap.md` or `docs/ideas.md`.
