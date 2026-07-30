# README walkthrough audit — 2026-07-30

## Scope and evidence

This audit covers the four scripted README walkthroughs in
`tools/demo_recorder.py` and all twelve shipped files in `docs/media/`.
It does not treat “the child process exited successfully” as visual evidence.
The baseline was decoded with:

```bash
PATH=~/miniconda3/bin:$PATH direnv exec . \
  python tools/demo_inspector.py --strict
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

1. Agent-readable final-format inspection (landed as `bf5d69b4`).
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
