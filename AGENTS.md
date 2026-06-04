# AGENTS.md

This is a Python/Qt n-dimensional array viewer.

Before changing code:
- For feature direction, read `docs/mission.md` and `docs/roadmap.md`.
- For architecture-sensitive changes, read `docs/architecture.md`.
- For dimension operations, read `docs/decisions/0002-dimension-ops.md`.
- For ArrayShow comparisons, read `docs/references/arrayshow.md`.
Update docs only when changing architecture, public behavior, roadmap status, or ArrayShow-derived design decisions.

Rules:
- Do not add major behavior directly to the main window if it belongs in view state, slicing, or dimension operations.
- Prefer pure NumPy functions for array transformations.
- GUI callbacks should be thin: update state, call services, refresh view.
- Every new operation needs at least a shape/value test.
- Every visible UI feature needs either a smoke test or a rendering snapshot.