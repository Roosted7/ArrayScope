# AGENTS.md

This is a Python/Qt n-dimensional array viewer.

## Before changing code

Read only the files relevant to the task:

* For project scope and feature direction, read `docs/mission.md` and `docs/roadmap.md`.
* For architecture-sensitive changes, read `docs/architecture.md`.
* For accepted design decisions and rationale, see `docs/decisions/*.md`.
* For ArrayShow comparisons, read `docs/references/ArrayShow.md`.
* For non-blocking future ideas, check `docs/ideas.md` only when the task asks for planning, feature design, or follow-up suggestions.

## While changing code

* Update docs when changing architecture, public behavior, roadmap status, or ArrayShow-derived design decisions.
* Add concise notes to `docs/ideas.md` when useful future ideas or technical debt are discovered.
* Add or update decision documents only for choices that affect future architecture, public API, packaging, testing strategy, or major UX behavior.
* Do not mark roadmap items as done, or move ideas into the roadmap, unless explicitly requested or clearly part of the assigned task.

## After changing code

* Add tests for new behavior.
* Run relevant existing tests and update them only when behavior intentionally changed.
* Do a small coherence pass if architecture, public behavior, or documentation changed.
* End with a handoff note containing:

  * what changed;
  * tests run;
  * manual checks performed, if any;
  * important follow-up ideas or risks.

## Rules

* Do not add major behavior directly to the main window if it belongs in view state, slicing, dimension operations, or another focused module.
* Prefer pure NumPy functions for array transformations.
* GUI callbacks should be thin: update state, call focused logic, refresh the view.
* Every new array operation needs at least a shape/value test.
* Every visible UI feature should have a smoke test or rendering snapshot where practical.
* Keep refactors behavior-preserving unless the task explicitly asks for behavior changes.
* Prefer small, reviewable changes over broad rewrites.
