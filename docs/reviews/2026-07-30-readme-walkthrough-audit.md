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
five scenarios and fifteen final encodes:

| Scenario | Final MP4 duration | Final source frame | Closed findings |
|---|---:|---:|---|
| `showcase` | 30.00 s | 1100×858 | Visible slice scrub, explicit crop/restore, real 1:1 and fit actions |
| `fft` | 27.63 s | 1100×838 | Real menus/combos, exact stack checkbox, no ambiguous linear-magnitude failure beat |
| `montage` | 12.50 s | 1100×858 | Caption precedes trigger, semantic `:` action, incomplete→complete presentation, zoom/fit |
| `roi` | 17.37 s | 1100×898 | Honest panel transitions, animated rectangle/polyline, visible live-profile activation |
| `workflow` | 13.27 s | 1100×858 | Real parameterized Crop form and highlighted dimension video/PNG export entry |

All inline AVIF and GIF files are 880 px wide, so the README renders them at
native width rather than enlarging a smaller GIF. Every format has a companion
timeline, and strict inspection produced no structural, scale, duration,
cursor-target, or companion-file finding.

The bounded implementation history is:

1. `afc10b81` — final-format storyboard, report, and HTML inspection.
2. `c5834ade` — this baseline audit and explicit acceptance gates.
3. `ef73fa3f` — action timelines and before/act/after event sheets.
4. `61c46fd6` — faithful FFT popup/combo composition and targets.
5. `b2b98eba` — honest click ownership and animated ROI gestures.
6. `0a070ce5` — dedicated, wrapped caption band.
7. `b8d1ac31` — native-width inline media and readable source layouts.
8. `b15b571f` — showcase crop and real 1:1 behavior.
9. `836f855a` — unambiguous logarithmic-magnitude FFT sequence.
10. `d36c1fe1` — parameterized-operation and export-menu workflow.
11. `47ee9a22` — semantic montage trigger and readable hold.
12. `df4623b4` — direct coverage of the production export-menu seam.

Some README claims deliberately remain prose: flip/FFT-shift variants, every
ROI shape, resource diagnostics/WGPU behavior, and asynchronous file loading.
Adding those to these tours would either duplicate an existing narrative or
misrepresent offscreen PyQtGraph capture as hardware/file-stream evidence.
The new workflow previews export discoverability but intentionally does not
open a modal dialog or write a user file.

## Still open after the closure

Reviewing the decoded storyboards frame by frame — rather than the structural
report — leaves five findings that the acceptance criteria above do not yet
hold the media to. They are recorded here so the next pass starts from
evidence rather than from a fresh reading of the same files.

| # | Finding | Why the current gate misses it |
|---|---|---|
| 1 | `fft` still spends roughly 4.5 s–11.5 s on a near-black frame: the Real channel of a linear-scaled complex FFT. The ambiguous-failure beat was moved, not removed. | Nothing measures decoded luminance, so "does this read as broken?" is still a human question. |
| 2 | The `showcase` crop beat is invisible. Cropping a centred Gaussian to `16:112` removes near-zero margins and the view refits, so the pixels before and after are indistinguishable. | The timeline proves the action ran and hit its target; no check asks whether the frame changed. |
| 3 | The composited channel/scale choice lists render the combo's own abbreviations (`R`, `|x|`, `∠`) at ~40 px wide. The menu is visible but unreadable, so "show the real choice" is only half met. | Target geometry is validated; legibility is not. |
| 4 | `showcase`, `montage`, and `workflow` all use `_volume3d`, a sum of Gaussians with no structure. Zoom, pan, 1:1, montage tiles, and the crop all act on a featureless blob; `_phantom2d` (used by `fft`/`roi`) reads far better. | Dataset choice is not part of any acceptance criterion. |
| 5 | The `true 1:1` beat shows a 128 px image in an ~830 px viewport — a speck in black. Correct, but it reads as the same kind of failure the FFT black beat did. | Same gap as #1. |

Three process findings sit alongside them:

- **One coverage row was neither shown nor declared prose.** "Adjust
  window/level → add a visible histogram/window adjustment if it stays
  concise" appears in the coverage table above, is absent from the landed
  outcome, and is missing from the deliberately-prose paragraph. Either
  demonstrate it or say why it stays prose.

- **Framing was never implemented.** "Control-heavy beats receive enough hold
  time *and framing* to read the relevant label" was met only by widening the
  window and holding longer. No beat crops or magnifies its subject, so a
  popup row is still ~1.5 % of the frame.
- **Media weight.** The thirteen commits write 38.0 MB of blobs to reach a
  12.1 MB final `docs/media/`; each re-record commits another full set. Squash
  the media so history carries one copy of each final file. `showcase.gif`
  alone is 3.73 MB and is the largest file in the repository's documentation.
