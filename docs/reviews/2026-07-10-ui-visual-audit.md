# UI visual audit — 2026-07-10 (offscreen gallery, PyQtGraph backend)

Method: `tools/ui_gallery.py` renders the real `ArrayScopeWindow` offscreen across
scenarios (2D/3D/6D/1D, complex, montage, operations, profiles, ROIs, dialogs,
NaN/Inf, constant, tiny, extreme aspect), themes (dark/light/system), window
sizes (420×420 → 1600×950) and HiDPI (scale 2). Screenshots reviewed per theme.

## Systemic

- **S1 Light theme is half-applied.** Chrome follows the palette but every
  pyqtgraph surface stays black: image canvas, histogram/levels pane, profile
  plot, inspection histogram. The app looks broken in light mode.
- **S2 Hardcoded dark styles in docks.** In light mode the operations dock
  renders dark rows and dark buttons inside a white panel; secondary metadata
  text has poor contrast in both modes.
- **S3 Crude level handles.** Yellow triangles/lines on the histogram look
  like debug graphics and clash with both themes; evidence markers (blue/
  orange triangles) read as visual noise.
- **S4 No consistent accent color** across selection, checked states, ROI
  defaults, level handles.
- **S5 Evaluation overlay** ("Updating image frame...") is a hardcoded dark
  chip, wrong for light mode, and shows no progress indication.

## Chrome / layout

- **C1 Menu bar clutter:** `Performance`, `Developer`, `Theme` as top-level
  menus. Theme belongs in View; Performance/Developer under a single place.
- **C2 Toolbar:** mixed label+combo+icon-button groups without visual
  grouping; at small widths (≤520 px) actions clip into slivers with no
  overflow affordance.
- **C3 Dimension strip is cryptic** — the core control of the app. Chips read
  `0 [384] ↑ ← P : ⁺₋ +`; nothing communicates that ↑/← assign image axes,
  `P` toggles profile, or that the text field takes slice syntax. Micro
  spin buttons; orphaned link icon at the far right; no scrub affordance for
  stepping through slices (the workhorse gesture).
- **C4 Operations dock:** wall of 9 equally-weighted buttons (Undo, Clear,
  Delete, Save/Load Recipe, Materialize, Export, Save/Load View). Rare
  actions crowd the common ones.
- **C5 Inspection dock:** cryptic mini-toolbar (bare combo + 3 glyph
  buttons); cramped table headers; unlabeled histogram axes.

## Behavior / edge cases (UI-visible)

- **B1 1D input renders an empty profile plot** (no curve) and wastes the
  whole top half on an empty image area with one floating chip.
- **B2 Non-finite data destroys autolevels:** ±Inf pushes the window to
  ±4e38, hiding all structure; NaN regions are not visually distinguished.
- **B3 Phase channel inherits abs/log display state:** levels stay at the
  magnitude range and log scale persists → near-black screen. Phase should
  force linear scale with [-π, π] levels and a cyclic map.
- **B4 Operation labels duplicate the word axis:** "Mean over axis axis 2".

## Status (end of 2026-07-10 redesign pass)

Addressed in this pass: S1–S5 (theme engine in `arrayscope/app/theme.py`:
semantic tokens, palette, app stylesheet, pyqtgraph propagation, runtime
re-theme of open windows), C1 (Theme menu folded into View), C3 (chip role
buttons now show Y↑/X←/P with accent-checked state, montage chips get an
accent border), C4 (ops dock: compact action row + "More" menu; "axis axis"
label fixed), C5 (histogram axes labeled), B2 (`finite_bounds` masks ±Inf;
`slice_engine` clamps non-finite into the finite range), B3 (phase channel
forces linear scale). Colorbar gradient ticks are hidden (colormaps stay
editable via the gradient context menu).

Still open: C2 (toolbar clips below ~460 px width — it lives in a plain
layout, not a QMainWindow toolbar, so no overflow chevron), B1 (empty 1D
profile plot — appears linked to the pipeline stall below), and the wasted
top area in the 1D layout.

Known interaction: styling `HistogramLUTItem.region` brushes breaks offscreen
VisPy grabs (`test_vispy_direct_tiled_complex_display_images_render_nonblank`);
only the region line pens are themed.

## Render pipeline (out of scope here — owned by the main-repo work)

- First frame stalls at the preview floor offscreen: one tile stays in
  `loading_tiles`, `dirty_payloads` keeps one entry, and
  `pipeline.counters.interactive_native_deferred == 1` with no active
  requests anywhere. The native rung was deferred against an "interaction
  active" snapshot after the last interaction-stop edge, so no wakeup
  remains (ADR 0051 violation). On-screen this is masked by mouse-move
  edges; offscreen it never recovers. `retarget_frame_pipeline(...,
  force_commit=True)`, `requeue_orphaned_loading_tiles()` (returns 0) and
  `replan_deferred_interactive_native_quality()` (returns False — counter
  already consumed) all fail to rescue. `montage_quality_policy=native-only`
  yields a fully black canvas.
