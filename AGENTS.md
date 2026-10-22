# Working on ArrayScope

ArrayScope is a Python/Qt viewer for n-dimensional scientific arrays. Help a
new user understand an array quickly, while keeping demanding workflows correct,
responsive, and explicit about resource costs.

## Direction and autonomy

The user's current goals take precedence over historical plans and local process
rules. Use engineering judgment to improve architecture, dependencies, UX, or
language boundaries when the evidence supports it. Explain material decisions
and verify the resulting behavior; these files are guidance, not an approval
workflow or a reason to leave authorized work unfinished.

Prefer coherent, reviewable changes. A substantial redesign is welcome when it
solves a demonstrated problem and has a migration and validation path. Native
code (including Rust) should earn its boundary with measured benefit, correct
array/dtype semantics, and a practical packaging story. Heavy NumPy, SciPy, FFT,
and WGPU work already runs in native code; account for conversion and call costs.

## Find useful context

Start with the README and the code/tests involved. Consult `docs/queue.md` for
current priorities and unresolved issues, `docs/architecture.md` for ownership,
and `docs/testing/README.md` for relevant validation. Read deeper documents when
they answer a question about the change, rather than treating the entire history
as a prerequisite.

Before performance or scheduling experiments, search `docs/graveyard.md` for
previous measurements. Earlier failures are evidence to learn from, not a
permanent ban on a better approach. `docs/agent-briefing.md` records recurring
review mistakes; `docs/areas.md` helps assign disjoint files to parallel agents.
Archive material describes history, not current instructions.

## Environment

Use a per-checkout environment pinned by `uv.lock`:

```bash
uv sync
tools/env python -m arrayscope data.npy
tools/env pytest
```

`tools/env <command>` selects this checkout's environment and source without
shell activation. Keep `uv.lock` in sync when dependencies change. Do not share
editable-install pointers across checkouts.

GUI tests normally use `QT_QPA_PLATFORM=offscreen`. Hardware/compositor tests
require an appropriate environment; report what was actually available.

## Design principles

- Semantic state belongs to `ViewState` and document objects; widgets mirror
  state and emit intent. Rendering orchestration belongs to `window.renderer`.
- Keep GUI callbacks responsive. Expensive work uses immutable worker snapshots,
  bounded resources, progress reporting, cancellation, and an owned completion.
- One authority owns each lifecycle/identity decision. Distinguish document,
  target, viewport, presentation, and physical residency; camera or LUT changes
  must not accidentally restart array evaluation.
- The committed frame owns coordinate and value semantics. A preview or reduced
  page cannot silently answer an exact ROI, hover, histogram, or export request.
- Backend implementations own textures and scenes, not scientific meaning.
  Maintained backends are WGPU and PyQtGraph; test the backend a claim names.
- Operations declare shape, dtype, capability, and region behavior. Preserve
  empty, scalar, complex, strided, and lazy-array behavior when generalizing.
- Every allocation, open file, worker, timer, and callback needs a clear owner
  and cleanup path, including cancellation, failure, and window closure.
- Deferred Qt callbacks need a receiver lifetime and protection from stale work.
- Surface failures with useful errors, diagnostics, or counters. Intentional
  fallbacks should explain themselves; broken imports must not disappear.
- Fix the responsible owner rather than layering compatibility shims or a
  second scheduling/state system over a defect.

## Verification proportional to the change

Reproduce a defect before fixing it when practical. Prefer a few regression
tests that expose its real failure mode to many tests that mirror internals.
New operations need shape/value coverage. Visible features need interaction
coverage and an inspected screenshot where useful (`tools/ui_gallery.py`).

Use change-driven selection when its dependency map is trustworthy:

```bash
tools/env pytest
tools/env pytest --since HEAD
tools/env python tools/test_selection.py
```

`--since HEAD` includes current uncommitted changes; use a
specific merge base for a committed branch. Read the reported skipped/unmapped
and known-red tests before claiming coverage. If the map is stale or explicitly
under-scoped, use `pytest --testmon-noselect` to re-record it. Focused node IDs
are useful for reproductions; a green scoped run is not a whole-suite verdict.

Coordinate test worker counts across concurrent agents. Respect machine-pressure
refusals rather than overriding them. Wait for actual conditions/signals in Qt
tests, not a fixed sleep. User-visible interactions target 2 seconds and normally
fail at 5 seconds via `arrayscope.tools.interaction_budget`; whole-process
watchdogs are separate from interaction success criteria.

Choose the ring that can observe the defect:

- Numerical, I/O, process, allocator, and resource-ownership fixes need direct
  tests of values, failures, cancellation, and cleanup. A file's directory alone
  does not require a GPU run.
- Qt interactions and appearance need offscreen interaction/artifact checks.
- Changes to rendering, scheduling, LOD, presentation, or viewport behavior need
  stress and real compositor/pixel checks before claiming rendering acceptance.
  Offscreen tests cannot substitute for physical-pixel or GPU timing evidence.
- Performance claims need deterministic work/resource counters and repeatable
  before/after measurements. State dataset, backend, environment, and limits.

Finish with the appropriate checks:

```bash
tools/env ruff check .
tools/env ruff format --check .
tools/env python -m compileall -q arrayscope
tools/env python tools/dead_code.py
git diff --check
```

Do not broaden lint/dead-code exceptions to hide a defect. Ordinary imports only:
do not load a duplicate production module into `sys.modules` in tests. Fixtures
pin memory budgets; use `real_system_memory` only when host sampling is intended.

## Communication and handoff

Explain what changed, why it helps, what was verified, and material limitations.
Distinguish reproduced baseline failures from new regressions with concrete
commands and revisions. Update user-facing docs when behavior changes and keep
`docs/queue.md` current; summarize completed work in `docs/queue-done.md` rather
than appending execution logs. Add an ADR for a durable architecture/API decision,
not every fix. Record rejected experiments with evidence in the graveyard.
