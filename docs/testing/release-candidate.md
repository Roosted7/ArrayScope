# Release-candidate evidence

Use this developer checklist when preparing an ArrayScope release candidate or
when an agent needs to reproduce the N0 baseline artifacts. The user-facing
README intentionally does not include these commands.

## Version identity

The `0.8.0` RC baseline uses one canonical runtime version:

```bash
python - <<'PY'
from importlib.metadata import metadata, version
import arrayscope

assert metadata("ArrayScope")["Name"] == "ArrayScope"
assert version("ArrayScope") == arrayscope.__version__ == "0.8.0"
print(f"ArrayScope {arrayscope.__version__}")
PY
```

Release publication should use tag `v0.8.0`, and the package should be pushed
from the dedicated ArrayScope repository rather than the historical ndslice
fork.

## Automated gate

Run the normal local gate before publishing:

```bash
pytest -q tests/core tests/operations
QT_QPA_PLATFORM=offscreen pytest -q tests/display tests/window
QT_QPA_PLATFORM=offscreen pytest -q tests/ui tests/app
python -m compileall -q arrayscope
ruff check arrayscope tests --select F821,E9
git diff --check
python -m build
python -m twine check dist/*
python -m arrayscope --help
```

## RC artifacts

Runtime diagnostics traces and rendering benchmarks are separate JSONL schemas.
Generate and summarize them with separate commands:

```bash
mkdir -p tests/artifacts
QT_QPA_PLATFORM=offscreen python -m arrayscope.tools.release_diagnostics \
  --jsonl tests/artifacts/v0.8.0-diagnostics-pyqtgraph.jsonl \
  --backend pyqtgraph
python -m arrayscope.core.diagnostics_trace \
  tests/artifacts/v0.8.0-diagnostics-pyqtgraph.jsonl \
  > tests/artifacts/v0.8.0-diagnostics-pyqtgraph-summary.md
QT_QPA_PLATFORM=offscreen python -m arrayscope.tools.release_diagnostics \
  --jsonl tests/artifacts/v0.8.0-diagnostics-vispy.jsonl \
  --backend vispy
python -m arrayscope.core.diagnostics_trace \
  tests/artifacts/v0.8.0-diagnostics-vispy.jsonl \
  > tests/artifacts/v0.8.0-diagnostics-vispy-summary.md
QT_QPA_PLATFORM=offscreen python -m arrayscope.display.rendering_benchmarks \
  --runs 1 \
  --jsonl tests/artifacts/v0.8.0-rendering-benchmark-linux.jsonl
```

Do not feed rendering benchmark JSONL into
`arrayscope.core.diagnostics_trace`; benchmark samples and runtime diagnostics
traces have different schemas.

## Evidence to record

Record the commit, clean/dirty state, CI run URL, platform skips, artifact
paths, OS/session type, Python/Qt/PySide/PyQtGraph/VisPy versions, backend,
dataset shape/dtype, and any diagnostics warning observed during manual checks.
