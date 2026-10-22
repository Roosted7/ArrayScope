# ArrayScope

[![CI](https://github.com/Roosted7/ArrayScope/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Roosted7/ArrayScope/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Roosted7/ArrayScope/branch/main/graph/badge.svg)](https://codecov.io/gh/Roosted7/ArrayScope)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://github.com/Roosted7/ArrayScope/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ArrayScope is a Python/Qt viewer for quickly understanding n-dimensional NumPy arrays. It is aimed at scientific and reconstruction workflows where the useful first questions are usually: *which dimensions matter, what does this slice contain, how do values change, and what happens after a small operation such as an FFT, crop, reduction, or axis change?*

The current repository is the ArrayScope `0.8.0` release-candidate line. It has moved well beyond the original lightweight ndslice viewer: the implementation now contains a staged operation evaluator, bounded caches, progressive montage rendering, ROI/profile inspection, runtime diagnostics, and maintained WGPU and PyQtGraph rendering backends. See [Current state](docs/current-state.md) before treating every advanced path as production-stable.

<picture>
  <source srcset="docs/media/showcase.avif" type="image/avif">
  <img src="docs/media/showcase.gif" alt="ArrayScope walkthrough: hovering values, scrubbing dimensions, re-mapping image axes, zooming and panning" width="880">
</picture>

<sub>Every demo on this page is [rendered automatically](docs/media/README.md) from a scripted walkthrough of the real application — higher-quality MP4: [showcase.mp4](docs/media/showcase.mp4).</sub>

## Start here

Run `arrayscope` without arguments to open your data, resume a recent file,
or explore a built-in volume, time series, or complex field. Examples are
generated locally, include named dimensions and hints, and need no download.

HDF5, NPZ, and MAT containers have a searchable dataset browser with shape,
type, and memory estimates. To transfer a processing workflow between arrays,
use **Copy Operation Recipe to Clipboard** and **Paste Operation Recipe
(Replace Stack)** from File, Operations → More, or the `Ctrl+K` palette.

```python
import numpy as np
import arrayscope as asc

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
z = np.linspace(-5, 5, 50)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
data = np.exp(-(X**2 + Y**2 + Z**2) / 10)

asc(data, title="3D Gaussian")
```

For a file:

```bash
python -m arrayscope data.npy
# or, after installation
arrayscope data.npy
```

By default, a call from a normal Python process opens a separate viewer process. Use `block=True` when the caller must wait for the window to close:

```python
asc(data, block=True)
```

When Qt already exists in the current process, ArrayScope avoids an unsafe fork. In IPython/Jupyter with `%gui qt`, it opens inline; without a running Qt event loop it falls back to blocking mode with a warning.

From Julia or MATLAB, dependency-free wrappers in [`wrappers/`](wrappers/README.md) hand the array to a detached viewer process through a raw, memory-mapped `.npy` file — no in-process Python bridge, no compression, and the host session stays responsive:

```julia
arrayscope(vol; name="kspace")          # Julia
```

```matlab
arrayscope(vol, 'name', 'kspace');      % MATLAB
```

See [Invocation](docs/invocation.md) for all routes and the handoff efficiency contract.

## What it does today

### Navigate and reshape the view

- Select two image axes and slice remaining dimensions.
- Enter explicit index/range selections, including cropped X/Y views.
- Flip axes and apply centered FFT shift semantics per dimension.
- Use normal image, line-profile, or tiled montage presentation.
- Preserve, fit, or restore the viewport, including true 1:1 display.
- Choose scalar, diverging, scientific, or custom colormaps independently of source data.

<picture>
  <source srcset="docs/media/colormap.avif" type="image/avif">
  <img src="docs/media/colormap.gif" alt="Opening the Color control and changing the displayed image from gray to viridis" width="880">
</picture>

<sub>[colormap.mp4](docs/media/colormap.mp4)</sub>

### Inspect values

- Hover pixels using the committed frame’s coordinate/value model.
- Draw ROI, line, polyline, and freehand inspection geometry.
- View live profiles and ROI statistics/histograms.
- Adjust window/level from the histogram, use automatic levels, or enter values directly.

<picture>
  <source srcset="docs/media/roi.avif" type="image/avif">
  <img src="docs/media/roi.gif" alt="Drawing ROIs, live statistics and histograms, and pointer-following line profiles" width="880">
</picture>

<sub>[roi.mp4](docs/media/roi.mp4)</sub>

### Transform data without replacing the source

The operation stack supports reversible, ordered steps such as crop, reverse, reductions, centered FFT/IFFT, FFT shift, and complex-axis conversion. The document keeps the source array and operation history separate; runtime optimization does not rewrite the visible operation stack.

**Edit → Undo/Redo** restores additions, removals, parameters, ordering,
enabled state, and recipe replacements. Standard platform shortcuts and the
Operations dock use the same history; text fields retain their own editing
shortcuts. History keeps up to 64 edits on the current source, with a separate
limit on extra input arrays retained only for recovery.

The operation manager browses the same shared library and keeps New, Duplicate,
runtime selection, parameters, and implementation settings in one editor.

<picture>
  <source srcset="docs/media/operations.avif" type="image/avif">
  <img src="docs/media/operations.gif" alt="Opening the operation manager from the View menu and browsing its shared-library authoring controls" width="880">
</picture>

<sub>[operations.mp4](docs/media/operations.mp4)</sub>

<picture>
  <source srcset="docs/media/fft.avif" type="image/avif">
  <img src="docs/media/fft.gif" alt="Adding centered FFT steps, viewing magnitude on a log scale and phase, and toggling a step in the operation stack" width="880">
</picture>

<sub>[fft.mp4](docs/media/fft.mp4)</sub>

### Work with larger views

- Visible image evaluation can run asynchronously and in chunks.
- Montage tiles are evaluated and presented progressively.
- Image, tile, profile, and reusable operation-stage caches have separate budgets.
- Runtime memory policy, latency feedback, a resource governor, and diagnostics expose why work was admitted, delayed, degraded, or refused.
- WGPU is the GPU/rendering certification path, with shader mapping, compute, and persistent tiled residency. PyQtGraph is the CPU/remote fallback.

<picture>
  <source srcset="docs/media/montage.avif" type="image/avif">
  <img src="docs/media/montage.gif" alt="Typing ':' to montage a dimension, tiles rendering progressively, and zooming into the montage" width="880">
</picture>

<sub>[montage.mp4](docs/media/montage.mp4)</sub>

These mechanisms substantially improve bounded behavior. The unified frame planner, work-graph
admission, backend composition, and the v32 render-orchestrator extraction are in place; the live
plan (token unification, backend de-duplication, hardware evidence) is in the
[roadmap](docs/roadmap.md).

### Export and load data

Choose **File → Export Selected Array…**, or press **Ctrl+S** (**Cmd+S** on macOS),
to choose an exact array region before
processing it. The selection dialog previews shape, dtype, and payload size;
evaluation and writing run in a worker with progress and cancellation. Export
uses scientific values in array axis order, including complex values, rather
than display colors or scaling. Small selections read only the required source
region when the operations support it; FFTs and other whole-stage operations
show their larger processing cost before execution.
**Export Full Derived Array…** remains available for the complete processed
array with embedded recipes in NPZ or separate NPY recipe files.

Video/PNG-frame export is available from a dimension action.

<picture>
  <source srcset="docs/media/workflow.avif" type="image/avif">
  <img src="docs/media/workflow.gif" alt="Choosing a parameterized crop, editing its bounds, applying it to the reversible operation stack, and opening dimension video/PNG export" width="880">
</picture>

<sub>[workflow.mp4](docs/media/workflow.mp4)</sub>

Supported command-line inputs include:

| Format | Suffix or input | Notes |
|---|---|---|
| NumPy | `.npy`, `.npz` | Multiple arrays open a selector |
| MATLAB | `.mat` | SciPy, with HDF5 fallback for v7.3-style files |
| HDF5 | `.h5`, `.hdf5` | Numeric datasets and common complex layouts |
| BART | `.cfl` + `.hdr` | Paired files |
| Philips | `.REC` + `.xml` | Paired files |
| NIfTI | `.nii`, `.nii.gz` | Via nibabel |
| DICOM | `.dcm` | Single-file loading via pydicom |
| DICOM directory | directory | Converted through `dcm2niix` on `PATH` |

When a container contains several datasets, ArrayScope shows a selector and highlights supported numeric arrays.

![Dataset selector](docs/images/selector.png)

Files load asynchronously: a loading window with progress appears immediately, and for `.npy`, `.cfl`, and `.REC` the viewer opens while the file is still streaming in, with a status-bar indicator of how much of the file is available. Running `arrayscope` without arguments opens a launcher window; files can also be dropped onto any viewer window.

### Desktop integration

Register ArrayScope with your desktop shell (application menu entry, icons, and file-type associations, per-user, no root/admin needed):

```bash
arrayscope --install-desktop     # register; --uninstall-desktop reverses it
```

On Linux this installs an XDG desktop entry, MIME types, and icons under `~/.local/share`; on Windows per-user file associations and a Start Menu shortcut; on macOS an `ArrayScope.app` bundle in `~/Applications`. See [docs/desktop-integration.md](docs/desktop-integration.md).

## Installation

### Just want a viewer? (no Python required)

Standalone builds bundle everything — download, run, open files. Grab them from the [releases page](https://github.com/Roosted7/ArrayScope/releases):

- **Windows** — `ArrayScope-Setup-<version>.exe`: a conventional installer (Start Menu entry, optional desktop icon and file associations, uninstaller). A `...-portable.zip` is also available if you prefer no installation at all.
- **Linux** — `ArrayScope-<version>-x86_64.AppImage`: `chmod +x` and run; no installation, no root.
- **macOS** — `ArrayScope-<version>-macos-<arch>.dmg` (Apple Silicon and Intel): drag to Applications. The bundle is unsigned for now, so the first launch is right-click → Open.

Launching without a file shows an open dialog; `packaging/` has the build scripts and [packaging/README.md](packaging/README.md) the details.

### Using Python? (integrates with your environment)

Once published on PyPI, any of the standard routes work — pick the one matching how you manage environments:

```bash
pip install arrayscope        # into the active environment (importable + CLI)
pipx install arrayscope       # isolated CLI tool on your PATH
uv tool install arrayscope    # same idea, via uv
```

The Python install is the full-featured route: `from arrayscope import arrayscope as asc` in scripts and notebooks, plus the `arrayscope` CLI.

### From source

ArrayScope identifies this release-candidate baseline as version `0.8.0`. Until the first PyPI upload is published, install from the dedicated ArrayScope source checkout:

```bash
git clone <repository-url>
cd ArrayScope
python -m pip install -e ".[dev,wgpu]"
python -m arrayscope path/to/data.npy
```

The project’s maintained development environment is [uv](https://docs.astral.sh/uv/) plus a per-checkout `.venv`, described by `pyproject.toml` and pinned by the committed `uv.lock`. One command builds it, and `.envrc`/direnv puts it on PATH:

```bash
uv sync                       # creates .venv with every maintained extra
tools/env pytest -q tests/core
```

`tools/env <command>` runs one command against `.venv` without needing direnv or an activated shell — useful for scripts and CI-like invocations. Each checkout gets its own venv deliberately: a shared editable install lets one checkout’s pointer decide what every other process imports.

`pytest` runs the tests your working tree affects and skips the ones it provably does not — pytest-testmon records which source lines each test executed, so a change selects the tests that reach it wherever they live. `pytest --since` is the pre-merge command: it names everything the branch changed, reports which of it the map already verified against the current tree, and runs only the gaps. `python tools/test_selection.py` reports the blast radius without running anything. See [change-driven test selection](docs/testing/test-selection.md).

The suite runs in parallel by default (`pytest-xdist`); append `-n 0` to run serially when debugging. See [testing strategy](docs/testing/strategy.md#parallel-execution) for details.

Pull requests run a dedicated coverage job in CI. To reproduce it locally:

```bash
tools/env pytest tests/ -q --cov=arrayscope --cov-report=term-missing:skip-covered --cov-report=xml
```

Core runtime dependencies include NumPy, PySide6, PyQtGraph, SciPy, h5py, pydicom, nibabel, imageio, Pillow, and psutil. WGPU support is available through the optional `wgpu` extra.

## Project documentation

- [Documentation index](docs/index.md) — the progressive entry point.
- [Mission](docs/mission.md) — product boundaries and success criteria.
- [Current state](docs/current-state.md) — what is mature, transitional, or experimental.
- [Architecture](docs/architecture.md) — ownership and invariants.
- [Roadmap](docs/roadmap.md) — current gates rather than historical phases.
- [ArrayShow and ArrayView comparison](docs/comparison.md) — product and technical lessons.
- [v32 composition audit](docs/reviews/v32-composition-audit.md) — findings behind the current plan.

Historical phase notes and manual checklists remain under [`docs/archive/`](docs/archive/README.md); they are evidence, not current instructions.

## Scope

ArrayScope is deliberately not a napari replacement, a MATLAB clone, a medical workstation, or a general registration/segmentation platform. It should remain quick to invoke and easy to understand while supporting serious array inspection and a bounded path for larger scientific data.

## License

MIT. See [LICENSE](LICENSE).
