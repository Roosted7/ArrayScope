# README walkthrough audit — 2026-07-30

## Scope and evidence

This audit covers the four scripted README walkthroughs in
`tools/demo_recorder.py` and all twelve shipped files in `docs/media/`.
It does not treat “the child process exited successfully” as visual evidence.
The baseline was decoded with:

```bash
tools/env python tools/demo_inspector.py --strict
```

The resulting `tests/artifacts/demo-inspection/` bundle contains a
timestamped storyboard for every final MP4, AVIF, and GIF, a JSON probe report,
and an HTML playback gallery. The structural gate passed for all twelve files.
Durations and native MP4 sizes were:

| Scenario | Duration | MP4 frame | What the README promises |
|---|---:|---:|---|
| `showcase` | 21.60 s | 1100×780 | Hover, scrub, axis remap, zoom, pan |
| `fft` | 16.90 s | 1280×800 | Add FFTs, magnitude/log, phase, stack toggle |
| `montage` | 10.43 s | 1100×780 | Enter montage, progressive tiles, zoom |
| `roi` | 14.50 s | 1280×900 | Draw ROIs, inspect statistics, line/live profiles |

Every GIF is encoded at 760 px but embedded at 880 px. The browser therefore
enlarges it 1.16×, magnifying palette loss while making already-small UI text
less distinct. AVIF is 880 px and MP4 retains the recording size.

## Baseline findings

### The inspection loop still misses the most important instants

The new contact sheets make the final encodes visible to agents, but uniform
sampling can skip a 100 ms press or a short-lived menu. The recorder emits no
timeline manifest, so an inspector cannot ask for “the frame before, during,
and after every action.” There is also no durable mapping from a caption to
the action and target it describes.

**Acceptance:** every scenario writes an event manifest with caption and action
frame ranges, target geometry, cursor position, and the resulting visual
checkpoint. The inspector renders an event sheet in addition to the uniform
storyboard and reports actions whose target/cursor is outside the captured
frame.

### Script success is weaker than workflow success

The smoke test asserts only that at least one PNG exists. It cannot prove that
all declared captions/actions ran, that a promised visible state was reached,
or that a popup was captured. Several scenarios call internal handlers while
the cursor merely gestures at a plausible place. A UI path can therefore
break while the demo remains green.

**Acceptance:** scenario contracts name their promised beats; smoke mode
requires each beat in the manifest and scenario-specific visible postconditions.
Common clicks use a real Qt widget action where possible, with an explicit
documented presentation seam only for offscreen popups.

### Popups and menus are absent

`fft` moves to each dimension “+” button and then invokes
`_append_operation()` directly. Channel and scale combos similarly jump by
calling `setCurrentIndex()`. The viewer never sees the operation menu,
“Magnitude,” “Log,” or “Phase” choices. This is both a discoverability failure
and the main reason clicks look disconnected from their consequences.

Qt offscreen popups are separate top-level surfaces and are not included by
`win.grab()`; `tools/ui_gallery.py` already documents that a `QMenu` can also
auto-dismiss immediately offscreen.

**Acceptance:** the recorder can composite a grabbed popup at its real anchor,
move the synthetic pointer within that same coordinate system, visibly
highlight the chosen row, and trigger the row’s production action. The FFT
walkthrough shows both operation menus and the channel/scale choices.

### Cursor ownership is inconsistent

The ordinary widget path resolves geometry live and is sound. The failures are
the bypasses:

- `roi` calls `click(None, invoke=win._show_inspection_dock)`, so a click ripple
  remains at the rectangle corner while the dock opens elsewhere.
- The polyline appears atomically, then the pointer moves to its endpoint; no
  drawing gesture is shown.
- Live profile is enabled with a direct `setChecked(True)` while its control is
  hidden, so a profile panel appears without any activation gesture.
- `fft` toggles an operation through `set_operation_enabled()` after clicking a
  hand-chosen fraction of the dock body rather than the checkbox itself.

**Acceptance:** an action either targets a visible live widget/overlay and
records that target, or is introduced as a non-click transition. Drawing
gestures keep the cursor on the geometry being created. No click ripple may be
used to imply an action at another location.

### Captions compete with the application

Captions are drawn over the bottom of the window, exactly where the dimension
chips, status text, profile controls, and dock footer live. They use one
single-line 11.5-point font with no width bound or wrapping. In the GIF the
whole frame is reduced to 760 px and then enlarged by the README, making the
text harder to read. A new caption fades from zero while the action can already
be underway, so the montage-progress caption is nearly invisible during the
first progressive frames.

**Acceptance:** captions occupy a dedicated, non-overlapping band; use a
pixel-sized font and bounded wrapping that remains legible at the smallest
encoded width; and establish the caption before its action starts. Tests cover
long text, the band bounds, and the first actionable frame.

### Important UI is too small at README scale

The two 1280-wide scenarios shrink to 880 px (0.69×) in AVIF and 760 px
(0.59×) in GIF. Operation rows, inspection values, histogram axes, toolbar
choices, and the dimension strip become difficult to read. The uniform
full-window framing never emphasizes a small control when it is the subject.

**Acceptance:** the GIF is never enlarged by the README; source windows use the
smallest layout that keeps their panels usable; and control-heavy beats receive
enough hold time and framing to read the relevant label. The inspector reports
the exact final scale for every format.

### Scenario-specific clarity defects

| Scenario | Baseline defect | Visible acceptance |
|---|---|---|
| `showcase` | “true 1:1” is only caption text; the action is never shown. Hover coordinates are tiny and the scripted slice changes do not visibly type. | Show the 1:1 control and return to fit; type/scrub against the real edit; hold a hover readout long enough to inspect. |
| `fft` | Menus and combo choices are skipped; operation enable uses an approximate dock coordinate; the linear-magnitude near-black beat reads as a rendering failure. | Show/highlight actual choices, target the real stack checkbox, and label or remove the ambiguous black beat. |
| `montage` | The progressive caption fades in after work has already begun; the `:` field is tiny; no tile identity or non-square layout behavior is demonstrated. | Caption before triggering, make the typed field readable, and preserve the progressive incomplete-to-complete sequence plus zoom/fit. |
| `roi` | Inspection opens at the wrong cursor location; polyline creation is atomic; live profile has no visible activation; the 1280-wide split panels are too small. | Use honest non-click/panel transitions or visible controls, animate every drawn segment, and frame the inspection/profile results readably. |

## Feature coverage against the README

The existing media should not become an exhaustive product tour, but every
claim adjacent to an animation must either be shown or clearly remain prose.
The current coverage is:

| README claim | Current evidence | Action |
|---|---|---|
| Select image axes; slice dimensions | `showcase` | Keep and make input/action visible |
| Explicit ranges/cropped X/Y | None | Add a short range/crop beat |
| Flip axes / FFT-shift semantics | None | Leave as prose; FFT already owns transform time |
| Fit / true 1:1 | Fit only | Add one real 1:1 action |
| Rectangle, line, polyline, freehand | Rectangle and atomic polyline | Animate polyline; leave line/freehand as prose |
| ROI stats/histogram and live profile | Present but small/bypassed | Repair the ROI walkthrough |
| Adjust window/level | None | Add a visible histogram/window adjustment if it stays concise |
| Reversible ordered operation stack | FFT/toggle only | Show actual chooser and exact toggle; add one parameterized crop/reduction beat if readable |
| Progressive montage | Present | Preserve the incomplete→complete pixels |
| Resource budgets/diagnostics/WGPU | None; recorder is PyQtGraph/offscreen | Keep as prose; a headless README demo is not hardware evidence |
| Video/PNG export | None | Add a short visible export-menu beat without writing user files |
| Asynchronous file loading/selectors | Static selector PNG only | Keep separate; deterministic file-stream capture is a different workflow |

## Bounded landing sequence

Each item below is independently reviewable and must land with focused tests,
updated documentation, and an explanatory commit body:

1. Agent-readable final-format inspection (landed as `afc10b81`).
2. Event manifests and action-centered inspection sheets.
3. Recorder-owned popup/menu composition with coordinate-correct targeting.
4. Non-overlapping, wrapped, scale-stable captions.
5. Honest widget clicks/drawing gestures and target/postcondition smoke gates.
6. Encoding/layout scale correction, including an 880 px GIF.
7. One scenario commit per materially changed README animation, with refreshed
   MP4/AVIF/GIF files and inspected before/after evidence.
8. A concise additional workflow only where the coverage table calls for a
   visible action (parameterized operation and export menu), rather than
   stretching the four existing narratives beyond readability.

The final gate is the affected pytest selection plus `pytest --since`, Ruff,
the dead-code/pre-commit hook, `demo_inspector.py --strict`, and direct review
of every changed final-format storyboard. Offscreen PyQtGraph media is visual
documentation, not WGPU or real-compositor certification.

## Landed outcome

The baseline above is retained as the before-state. The completed media set has
seven scenarios and twenty-one final encodes:

| Scenario | Final MP4 duration | Final source frame | Closed findings |
|---|---:|---:|---|
| `showcase` | 26.20 s | 1100×858 | Live hover HUD, pointer-coupled window/level, middle-drag slice scrub, asymmetric crop/restore, meaningful 1:1 and fit actions |
| `fft` | 26.90 s | 1100×838 | Readable menus/combos, exact stack checkbox, display mapping before the first transform |
| `montage` | 12.67 s | 1100×858 | Caption precedes trigger, semantic `:` action, incomplete→complete presentation, real middle-drag zoom and fit |
| `colormap` | 5.37 s | 1100×858 | Real Color control, readable choice list, and visible gray-to-viridis result |
| `operations` | 7.37 s | 1100×858 | Real View-menu action and composited production operation-manager dialog |
| `roi` | 24.87 s | 1100×898 | Dock-owned rectangle creation, command-palette polyline drawing, live statistics, and per-frame profile tracking |
| `workflow` | 13.27 s | 1100×858 | Real parameterized Crop form and highlighted dimension video/PNG export entry |

All inline AVIF and GIF files are 880 px wide, so the README renders them at
native width rather than enlarging a smaller GIF. Every format has a companion
timeline. Strict inspection produced no structural, scale, duration,
cursor-target, caption-timing, inline-byte-budget, or companion-file finding.
Every one of the twenty-one final-format storyboards and all seven event sheets was
also reviewed directly; the structural report was not treated as pixel
acceptance.

The bounded implementation history, stated without commit hashes so it remains
valid across the required media-history rewrite, is:

1. Final-format storyboard, report, and HTML inspection.
2. This baseline audit and explicit acceptance gates.
3. Action timelines and before/act/after event sheets.
4. Faithful FFT popup/combo composition and targets.
5. Honest click ownership and animated ROI gestures.
6. Dedicated, wrapped caption band.
7. Native-width inline media and readable source layouts.
8. Showcase crop and real 1:1 behavior.
9. Unambiguous logarithmic-magnitude FFT sequence.
10. Parameterized-operation and export-menu workflow.
11. Semantic montage trigger and readable hold.
12. Direct coverage of the production export-menu seam.
13. Fail-closed inspection of timelines, targets, captions, labels, and inline
    dimensions.
14. Structured, slice-varying volume fixtures for the three volume tours.
15. Display mapping before FFT, with human-readable choice rows.
16. Manifest-backed detail insets for control-heavy actions.
17. A real histogram window/level gesture.
18. A strict inline-media byte budget.
19. Focused magnification of ROI statistics.

Some README claims deliberately remain prose: flip/FFT-shift variants, every
ROI shape, resource diagnostics/WGPU behavior, and asynchronous file loading.
Adding those to these tours would either duplicate an existing narrative or
misrepresent offscreen PyQtGraph capture as hardware/file-stream evidence.
The new workflow previews export discoverability but intentionally does not
open a modal dialog or write a user file.

## Closure of the second pixel review

The second review decoded the shipped media and found failures that the first
structural gate could not see. Each finding was closed in code, re-recorded,
and then checked in the decoded pixels:

| Review item | Landed closure | Pixel or gate evidence |
|---|---|---|
| A — black FFT beat | Select logarithmic magnitude before the first centered FFT. | The transform now remains visible from its first result; the former ~4.5–11.5 s near-black hold is absent from all three FFT storyboards. |
| B — invisible crop | Use a structured 384×320×36 volume and crop axis 0 asymmetrically to `72:352`. | The edge landmark, field of view, and refit visibly change before the restore. |
| C — unreadable choices | Give composited combo rows scenario-supplied human names, with accessible text and tooltips as fallbacks. | Channel and scale rows read as full names; the event sheet shows them in a 1.9× detail inset rather than as 40 px glyph menus. |
| D — featureless volumes | Replace the Gaussian-only fixture with ellipses, an edge landmark, a striped insert, and z-varying structure. | Showcase, montage, and workflow now have recognizable spatial and slice-to-slice evidence; the montage's final row reads as valid data. |
| E — broken-looking 1:1 | Use the 384×320 showcase source. | At 1:1 the image remains a meaningful pixel-exact view instead of a 128 px speck. |
| F — missing framing | Add a bordered DETAIL inset that follows the active live or composited target while retaining the full window as context. Record and validate its source rectangle. | Control-heavy beats decode at roughly 1.9×; the wider ROI statistics table uses a focused 1.3× inset. |
| G — omitted window/level row | Drive a real histogram range drag through the production level-setting path. | The histogram handles and image contrast visibly change in the showcase event sheet and encodes. |
| H — media weight/history | Enforce a 3.5 MB limit for every inline AVIF/GIF and place all final generated files in one recording commit after the bounded code/docs commits. | `showcase.gif` fell from 3.73 MB to 1.91 MB; linked MP4 files remain exempt from the inline budget. |
| I — stale recorded labels | Re-record all five scenarios after the widget-label correction. | Timelines use control names rather than current values or bare `QComboBox`; event sheets contain readable multiplication text rather than the former tofu glyph. |

Luminance, semantic salience, and whether an operation makes an understandable
pixel change remain human visual-review questions. The inspector deliberately
does not pretend that geometry and timing alone prove those properties; its
storyboards and event sheets make that review repeatable.

## Closure of the third pixel review

The next review identified eight remaining places where a caption described a
capability but the recorder bypassed its shipped interaction. Each now follows
the same production entry point a user does:

| Review item | Landed closure | Pixel or gate evidence |
|---|---|---|
| 1 — ROI creation bypassed its tool | Choose Rectangle in the Inspection dock and press its Add button; enter polyline drawing through the production command-palette action. | The ROI event sheet shows both real entry points before either geometry appears. The dock intentionally exposes Cursor, Profile, Line, and Rectangle; polyline and freehand remain command-palette tools. |
| 2 — live profile jumped | Convert every animated cursor point into image coordinates and feed the production marker/profile update path during the move. | The profile curve changes continuously with the cursor, rather than only at three endpoints. |
| 3 — hover caption had no HUD | Send the synthetic scene move through the production hover signal after the viewport event filter. | The real `PixelHud` chip follows the pointer and displays coordinates and a value during the showcase hover beat. |
| 4 — histogram gesture was independent of levels | Map the cursor through the histogram value axis and move the upper handle while retaining the lower level. | Cursor, upper handle, histogram range, and image contrast now move together. |
| 5 — montage navigation was bypassed | Send a real middle-button press, vertical move stream, and release to the viewport. | The montage visibly zooms under the pointer; the manifest records the middle button for the complete gesture range. |
| 6 — slice navigation was bypassed | Send the same production gesture horizontally over the showcase viewport. | The highlighted dimension and pixels scrub together while the middle button is held. |
| 7 — colormaps had no animation | Add a topic-scoped clip that opens the real Color control and chooses viridis from the readable composited menu. | All three formats show both the choice and the changed image/LUT; the 5.37 s clip keeps the lead tour bounded. |
| 8 — the operation manager had no animation | Add a topic-scoped clip that triggers the real View-menu action, constructs the production dialog, and composites it without entering a modal loop. | The full manager, shipped operation tree, editor, and New/Duplicate controls remain visible for inspection. |

DETAIL framing now starts from the subject widget or action rectangle plus a
margin, chooses a clear destination corner, and is omitted below 1.15×. Slice
edits promote the subject to the complete dimension chip, popup focus is
cleared with the popup, and icon-only Fit framing is omitted. The manifest and
strict inspector record and validate both source geometry and achieved scale.

The GIF fallback remains deliberately supported for legacy README clients,
while AVIF stays the primary `<picture>` source. This is a bounded compatibility
choice rather than a reason to stretch a clip: the 3.5 MB inline limit decides
how long any scenario may become. Palette-limited ordered dithering leaves the
largest final GIF, `showcase.gif`, at 1.91 MB; every other GIF is 1.23 MB or
smaller.

The final re-record completed seven of seven scenarios in one run. Strict
inspection decoded all twenty-one files with no findings. Direct review covered
all twenty-one storyboards and seven action-centred event sheets; two suspicious
FFT GIF storyboard samples were additionally decoded at their exact timestamps
and contained one current DETAIL inset, matching MP4, rather than retained UI.
