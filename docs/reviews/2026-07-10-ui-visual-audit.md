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

Second pass (same day, from on-system screenshots + feedback):

- Icons are now explicitly tinted from the palette (raw Material SVGs are
  black — invisible on dark systems), with disabled and checked-state
  variants, re-tinted on theme switch (`icons.refresh_icon_tints`).
- Native theme derives full ArrayScope tokens from the OS palette instead
  of dropping all styling (readable chips/HUD/checked states, system colors).
- Selected list rows use a surface fill + accent edge instead of a full
  accent fill (ops rows stay readable when selected).
- Dimension chips: index badge bubble on the left, size without brackets,
  `|` separators, elastic widths (slice input shrinks before rows reflow;
  chip widths stay consistent across wrapped rows; sync-button footprint
  reserved so nothing clips), native-style spin arrows.
- C2 resolved: adaptive toolbar — iconed group labels and combo entries;
  degradation order: hide label text → hide label icons → icon-only combo
  entries; toolbar takes stretch priority over the eliding status labels.
- Size grips only on detached panels; inspection table/histogram split is a
  QSplitter; ops edit button hidden for non-editable operations.
- ROI right-click menu (rename/recolor/delete); hover HUD gains iconed
  context rows for the hovered ROI (label/kind, mean/n, min/max) and the
  profile marker; the HUD follows the cursor during ROI/profile drags (drag
  moves are re-emitted as scene mouse-moves — the pointer driver consumes
  them otherwise).
- 1D input caps the central-area height so the profile dock gets the space.

Still open: B1 (empty 1D profile curve — pipeline-linked).

Third pass (same day, user feedback on interactive session):

- Dimension strip: index badge is the profile toggle (P button removed);
  badges share one width sized to the largest index, un-highlight while the
  profile dock is hidden, and reopen it on click. Slice input defaults to
  fitting `100:2:200` (min ~3.5 chars); size label centered (5→3 chars);
  inter-chip spacing 12 px preferred shrinking to 4 px before chips shrink;
  sync button pinned right with fixed gaps; native spin arrows restored
  (the QSS subcontrol rules had hidden them).
- Profile semantics: closing the dock keeps the live crosshair and its
  value readout; reopening restores the remembered state; badge clicks and
  live-profile enable auto-open the dock; crosshair moves mirror the value
  into the toolbar status.
- Toolbar: left controls / centered status text (pixel value + muted
  shape·dtype, both eliding with full-text tooltips) / right-aligned
  window-mode + auto-levels + sync.
- ROI: default names are plain numbers; renameable inline in the
  inspection table; the overlay renders bold name + italic kind left-aligned
  with n/mean column-aligned right; the hover HUD separates ROI/profile
  context from the pixel row with a thin rule.
- Inspection dock: auto split sizes the table to its rows plus half an
  empty row (histogram clamped to 25–75%), until the user drags the handle.
- Operations dock: accent axis chip (d0/d2…) above the enable checkbox
  opens a change-dimension menu; rows show name / parameters / shape·dtype
  on separate lines; the context menu gained enable-disable, change
  dimension, edit parameters — all iconed; Materialize/More collapse to
  icons below ~360 px.
- Image context menu: icons on every entry plus a stubbed "Save viewport"
  submenu (with overlays default/bold, without overlays, full content).

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
