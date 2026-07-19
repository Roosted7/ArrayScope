# README demo media

Every animation in the top-level [README](../../README.md) is rendered by an
automated pipeline — nothing here is screen-recorded by hand, so the demos can
be regenerated at any time and never drift from the current UI.

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
| `.gif` | Inline fallback for viewers without AVIF | 12 fps, 760 px wide, diff palette + bayer dither, `gifsicle -O3 --lossy` |
| `.mp4` | Full-quality video (link/download) | H.264 CRF 21, native size and fps |

The README embeds `<picture>` blocks with the AVIF as primary source and the
GIF as fallback, plus an MP4 link. (Animated WebP was tried and rejected: on
the noisy k-space segments it came out at 10+ MB, an order of magnitude
larger than AVIF at comparable quality.)

## Regenerating

```bash
PATH=~/miniconda3/bin:$PATH direnv exec . python tools/demo_recorder.py
```

Useful flags: `--list`, `--only <substring>`, `--formats gif,avif,mp4`,
`--keep-frames` (keep the PNG frame dumps under `tests/artifacts/demos/`),
`--smoke` (fast tiny run, no encoding). Requires `ffmpeg` (and optionally
`gifsicle`) on `PATH`.

Commit the refreshed files in this directory together with the change that
made them stale.

## Staying maintainable

`tests/app/test_demo_recorder_smoke.py` records every scenario in `--smoke`
mode in CI. If a widget or window entry point a scenario scripts against is
renamed or removed, the suite fails at once — the demos break loudly in the
PR that changed the UI, not silently in the README.
