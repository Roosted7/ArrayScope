# Current project status

**Snapshot:** ArrayScope v28 audit branch, reviewed on 2026-06-22. The audit adds focused correctness,
performance, CI, and documentation fixes without changing the intended product direction.

## What is solid today

- Native PySide6/PyQtGraph application with a callable `arrayscope` module and CLI.
- N-dimensional slice selection with centered defaults, image-axis ranges, montage ranges/lists, and
  axis flips.
- Real, complex, RGB, 1D, linear/log/symlog, colormap, histogram, and window-level presentation.
- Operation document with immutable steps, enable/disable state, recipes, shape/dtype prediction,
  runtime optimization, region planning, lazy slab evaluation, and materialization.
- Profiles, hover values, ROI geometry/statistics/histograms, and demand evaluation outside the
  visible montage viewport.
- Bounded caches, a reusable in-memory stage cache, compute/memory policy, resource feedback,
  diagnostics, and JSONL traces.
- Progressive montage rendering with stable world geometry, semantic level tracking, stale-commit
  guards, priority around the viewport center or hover point, and persistent backend residency.
- Production PyQtGraph backend and an experimental VisPy raster/tiled backend behind shared
  presentation adapters.
- A substantial automated suite covering pure models, evaluation, rendering policy, and Qt behavior.

## Recent change audit

The most recent main-branch work concentrated on viewport constraints, adaptive/manual histograms,
centered slice/range selection, and center/hover-prioritized montage tiles. The direction is good, but
the changes crossed several timing-sensitive contracts at once.

The audit found and addressed these concrete issues:

- Tests paired the first scheduled callback with tile zero even though the scheduler now intentionally
  reorders work. Tests now bind callbacks to tile identity and still verify stale-result rejection.
- Centered-slice behavior was implemented but several tests still asserted index zero.
- A VisPy viewport test could skip after leaving the backend preference in persistent `QSettings`,
  contaminating later tests.
- Empty inspection refreshes rebuilt the table and histogram during montage viewport updates.
- Replacing an evaluation group invalidated the same hierarchy once per outstanding request and kept
  historical per-tile generations. Prefix generations now make invalidation bounded and completed
  tile state is pruned.
- The normal-image over-budget fallback referenced `format_bytes` without importing it, causing a
  latent crash only when the planner supplied no custom status text.
- Auto-windowing while a manual histogram popup was open accepted the preview and emitted a user-level
  change before requesting auto levels, producing two render paths. It now rejects the preview first.
- CI ran overlapping complete suites, omitted the optional VisPy backend, did not test Python 3.10 or
  3.11 despite declaring support, and could publish a mismatched release tag/package version.
- Follow-up restoration kept the display-resource cleanup, benchmark lifecycle bounding, viewport
  overlap recovery, montage priority input hardening, and early hover cleanup that remained
  complementary after the six restored fixes.

See [`reviews/project-audit-v28.md`](reviews/project-audit-v28.md) for the full findings and fixes.

## Main risks

### Rendering and UI concentration

The project has good semantic boundaries, but several transition-era classes still own too much:

| File | Approx. lines | Concentrated responsibilities |
|---|---:|---|
| `window/montage_renderer.py` | 2,101 | planning, stage warmup, tile scheduling, composition, levels, presentation updates, overlays |
| `display/imageview2d.py` | 2,079 | widget shell, PyQtGraph pixels, histogram, ROI, viewport, signals |
| `display/vispy_imageview2d.py` | 2,080 | inherited shell plus VisPy raster/tile/camera/overlay bridging |
| `display/backends/vispy/tiles.py` | 1,736 | atlas allocation, uploads, residency, visuals, diagnostics |
| `display/histogram_controller.py` | 691 | sampling, bins, interaction interception, popup UI, level math |

Large files are not automatically wrong, but these ones sit directly on event-loop and rendering
boundaries. New behavior should move toward the owners described in the architecture docs rather than
adding another branch to these classes.

### Timing and responsiveness

The existing timers are mostly bounded and stop when idle, but a cold interaction can still accumulate
coalescing, queue-poll, viewport-settle, evaluation, and presentation latency. Setter timing alone is not a
useful performance measure. The project still needs automated request-to-first-frame, frame-age,
event-loop-gap, no-upload-pan, level-change, and queue-scaling benchmarks.

### Backend migration

The shared semantic presentation and backend adapter direction is correct. The migration is incomplete:
`VisPyImageView2D` still subclasses `ImageView2D`, pointer capture/drag lifecycle are not fully owned by
the shared interaction controller, and real OpenGL/Wayland/DPI validation is limited. PyQtGraph must
remain the safe default until conformance and manual evidence say otherwise.

### Slice text ambiguity

Python slicing is the default and MATLAB ranges are accepted as fallback. Ambiguous three-part ranges
and silently bounded explicit index lists can hide user mistakes. This needs an explicit UX decision,
not more parser heuristics.

## Release blockers

1. Package/runtime metadata, repository tags, changelog history, and the intended distribution name do
   not yet describe one authoritative release. No project wheel is currently published on PyPI. The next
   release must choose one versioning scheme, update the source of truth, and verify the intended
   package name. The release workflow now rejects mismatches.
2. Run the complete CI matrix and strict UI suite from a clean checkout after merging the audit branch.
3. Perform a real-platform manual pass for PyQtGraph and VisPy on at least Linux Wayland/X11 plus one
   high-DPI macOS or Windows system.
4. Capture current screenshots and verify installation from the built wheel, not only editable source.
5. Decide whether the next published release advertises VisPy as experimental or keeps it out of the
   normal installation path.

## Scale and test posture

The source tree contains about 40,098 Python lines across the package and 90 `test_*.py` files. At the
start of the audit, the full local baseline reported 906 passed, 17 failed, and 2 skipped; the failures
were investigated rather than normalized away. Final validation and environment-dependent skips are
recorded in the audit report.
