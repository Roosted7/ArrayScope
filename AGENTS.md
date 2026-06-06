# AGENTS.md

This is a Python/Qt n-dimensional array viewer.

## Before changing code

Read only the files relevant to the task:

* For project scope and feature direction, read `docs/mission.md` and `docs/roadmap.md`.
* For architecture-sensitive changes, read `docs/architecture.md`.
* For accepted design decisions and rationale, see `docs/decisions/*.md`.
* For ArrayShow comparisons, read `docs/references/ArrayShow.md`.
* For non-blocking future ideas, check `docs/ideas.md` only when the task asks for planning, feature design, or follow-up suggestions.

## Environment

Use the project conda environment through direnv for local commands:

```bash
direnv exec . <command>
```

The `.envrc` activates the conda environment named `arrayscope`. If a non-interactive agent shell cannot find `conda`, add the local conda binary to `PATH` before running `direnv exec`, for example:

```bash
PATH=~/miniconda3/bin:$PATH direnv exec . <command>
```

When dependencies change, update `environment.yml` and apply it to the conda environment:

```bash
PATH=~/miniconda3/bin:$PATH direnv exec . conda env update -f environment.yml --prune
```

## While changing code

* Update docs when changing architecture, public behavior, roadmap status, or ArrayShow-derived design decisions.
* Add concise notes to `docs/ideas.md` when useful future ideas or technical debt are discovered.
* Add or update decision documents only for choices that affect future architecture, public API, packaging, testing strategy, or major UX behavior.
* Do not mark roadmap items as done, or move ideas into the roadmap, unless explicitly requested or clearly part of the assigned task.

## After changing code

* Add tests for new behavior.
* Run relevant existing tests and update them only when behavior intentionally changed.
* Check and ensure coherence across architecture, public behavior, and documentation.
  * Through file inspection, small tests, full UI simulations (simulate interactions, and check rendered results)
  * Iterate when needed (from updating docs, to bugs, all the way to improving looks and feels)
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
