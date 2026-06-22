# Changelog

This file records user-visible release changes. Detailed development history and architecture decisions live under `docs/` and in Git.

## Unreleased — ArrayScope development line

### Added

- Reversible dimension-operation stack with recipes, runtime optimization, cost estimates, and reusable operation-stage caching.
- Live profiles, ROI inspection, ROI histograms, comparison helpers, and managed inspection panels.
- Progressive, viewport-bounded montage rendering with typed tile payloads and explicit requested/materialized/resident/presented state.
- Runtime memory policy, lane-aware compute policy, latency feedback, resource governance, diagnostics snapshots, trace logging, and rendering benchmarks.
- Experimental VisPy raster/tiled backend with shader-based scalar and complex display mapping.
- Explicit viewport modes, fit lock, 1:1 view, cropped image-axis ranges, and adaptive/manual histogram controls.

### Changed

- PySide6 is the default Qt binding through PyQtGraph’s abstraction.
- Display semantics, backend mechanics, operation planning, caching, and UI orchestration have been split into focused packages.
- Documentation now separates live guidance from archived phase notes and provides a progressive architecture/roadmap path.

### Fixed in the v28 audit

- The normal-image over-budget fallback now imports and uses the byte formatter instead of raising `NameError`.
- Viewport zoom-out constraints still enforce recoverable content overlap when an old viewport center is far outside new content.
- Display widgets cancel queued histogram refreshes and close VisPy resources during shutdown.
- Rendering benchmarks release parentless Qt/VisPy object graphs and share one module result set, avoiding very slow test-process teardown.
- Two stale evaluator tests now vary actual sliced axes, and test package bootstrapping is collection-order independent.

## Legacy ndslice releases

The entries below predate the ArrayScope rebrand. Their version numbers do not describe the current package maturity.

### 0.7.0

- Added DICOM directory conversion through `dcm2niix`.
- Added Ctrl+S NumPy export with range selection and optional singleton squeezing.

### 0.6.1

- Real arrays default to the real component; complex arrays default to magnitude.
- Fixed macOS emoji rendering.

### 0.6.0

- Added file monitoring/live reload and cross-platform CI.
- Adopted the Fusion Qt style.
- Closed HDF5/NPZ containers promptly to avoid file-lock issues.

### 0.5.1 — 2026-04-09

- Fixed switching back to the gray colormap without matplotlib.

### 0.5.0 — 2026-02-18

- Added PyQt6/HiDPI support in the legacy line, more colormaps, video export, and MATLAB v7.3 fallback loading.
- Fixed window/level reset when reselecting linear or symlog scale.
