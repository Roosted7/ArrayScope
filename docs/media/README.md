# README demo media

Every animation in the top-level [README](../../README.md) is rendered by an
automated pipeline — nothing here is screen-recorded by hand, so the demos can
be regenerated at any time and never drift from the current UI.

The `workflow` scenario deliberately stops on the highlighted dimension-export
entry. It proves discoverability without opening a modal save dialog or writing
files during documentation generation.

## How it works

[`tools/demo_recorder.py`](../../tools/demo_recorder.py) drives the real
ArrayScope window headless (Qt `offscreen` platform, PyQtGraph backend,
private `QSettings`), exactly like `tools/ui_gallery.py`. Each scenario is a
short scripted walkthrough; a large synthetic mouse cursor (white with a black
outline, amber click ripples) and caption pills are composited onto every
frame, because the offscreen platform renders no native pointer. All
interaction targets are resolved from live widget geometry at run time — there
are no hard-coded pixel coordinates — and the frame timeline is captured at a
fixed 30 fps while asynchronous evaluation advances in roughly real time, so
progressive rendering shows up naturally.

Each scenario is then encoded three ways:

| Format | Purpose | Settings |
|---|---|---|
| `.avif` | Primary inline animation (all current browsers; near-MP4 sizes) | AV1, 20 fps, 880 px wide, CRF 34 |
| `.gif` | Inline fallback for viewers without AVIF | 12 fps, 880 px wide, diff palette + bayer dither, `gifsicle -O3 --lossy` |
| `.mp4` | Full-quality video (link/download) | H.264 CRF 21, native size and fps |

The README embeds `<picture>` blocks with the AVIF as primary source and the
GIF as fallback, plus an MP4 link. (Animated WebP was tried and rejected: on
the noisy k-space segments it came out at 10+ MB, an order of magnitude
larger than AVIF at comparable quality.)

## Regenerating

```bash
tools/env python tools/demo_recorder.py
```

Useful flags: `--list`, `--only <substring>`, `--formats gif,avif,mp4`,
`--keep-frames` (keep the PNG frame dumps under `tests/artifacts/demos/`),
`--smoke` (fast tiny run, no encoding), and `--no-inspect` (skip the normally
automatic final-format inspection). Requires `ffmpeg` (and optionally
`gifsicle`) on `PATH`. A successful full recording writes each scenario's
timeline next to its media and runs the strict inspector before returning.

Commit the refreshed files in this directory together with the change that
made them stale.

## Inspecting what viewers actually receive

Encoding success is not a visual acceptance gate. The recorder runs this step
automatically after encoding. To re-inspect existing MP4, AVIF, and GIF files
without recording them again:

```bash
tools/env python tools/demo_inspector.py --strict
```

The report is written under `tests/artifacts/demo-inspection/` (gitignored):

- `*-storyboard.png` is a timestamped, evenly sampled contact sheet of each
  final encoded file. Image-capable agents can open these PNGs directly.
- `*-events.png` shows the frame before, during, and after every recorded
  action, so a short click, popup, or state change cannot fall between uniform
  samples.
- `report.json` records dimensions, duration, frame rate, file size, README
  display scale, companion-format parity, action timelines, cursor/target
  validation, and absolute artifact paths.
- `index.html` plays every final format beside its storyboard for human
  review.

Use `--only <substring>` for one scenario and increase `--samples` when an
interaction needs closer inspection. `--strict` fails when a companion format
is missing, encoded durations/aspect ratios disagree, an action's cursor
misses (or a click lacks) its recorded target, or an action starts before its
caption can be read. It also refuses any inline format narrower than the
README's 880 px slot, because browser enlargement damages small text and
palette edges. Every recorded scenario writes `<scenario>.timeline.json`
beside its final media, and `--strict` requires it: without a timeline nothing
can adjudicate cursor targets or caption timing, so accepting one would be a
gate that cannot fail. Media without a timeline stays inspectable through
uniform samples in a non-strict run.
These structural checks complement, but do not replace, reviewing cursor
alignment, click/menu visibility, captions, readable scale, and whether each
promised feature is actually visible. The dated
[walkthrough audit](../reviews/2026-07-30-readme-walkthrough-audit.md) records
the baseline findings, README feature coverage, and bounded acceptance order.

Qt's offscreen plugin does not include separate popup surfaces in
`QWidget.grab()`, and can dismiss a `QMenu` immediately. The recorder handles
that limitation explicitly: it composites a grab of the real built menu at
the live control anchor, moves the synthetic cursor using the menu action's
real geometry, and triggers the real `QAction`. Combo choices use the same
presentation seam with the live combo model and `setCurrentIndex()` signal
path. This keeps menus visible without pretending an unrelated cursor click
caused the change.

The same honesty rule applies when no visible control exists in the captured
layout: the timeline records a named `transition` without drawing a click
ripple. Drawing interactions are recorded as gestures with their own
press-to-result frame range. A click is reserved for a live widget or
composited action whose target contains the cursor.

Captions are rendered in a dedicated 78 px band below the application capture,
never over the dimension strip, status text, dock footer, or plotted data.
They use a 22 px font, wrap or elide within two bounded lines, and receive a
lead-in before the described action begins. The timeline distinguishes the
application `window_size` from the final `capture_size`, so target checks
remain in the application's coordinate system.

All five walkthrough windows are 1100 px wide and both inline encodes are
880 px. This gives every scenario the same 0.8× application-to-README scale;
control-heavy FFT and ROI scenes no longer shrink from 1280 px, and the GIF
fallback is never enlarged.

## Staying maintainable

`tests/app/test_demo_recorder_smoke.py` records every scenario in `--smoke`
mode as part of the suite. If a widget or window entry point a scenario scripts against is
renamed or removed, the suite fails at once — the demos break loudly in the
PR that changed the UI, not silently in the README.

`tests/app/test_demo_inspector.py` creates a small encoded video and proves the
inspection path decodes the final artifact, writes a usable storyboard/JSON/
HTML bundle, and fails closed when an expected companion format is absent. It
also loads the inspector in-process to exercise its findings directly — a
subprocess is invisible to test selection, so without that the gate's own
tests would never re-run when the gate changes.
