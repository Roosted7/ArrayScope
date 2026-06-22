# Manual regression checklist — VisPy v24 rendering review

Run on at least Wayland/Intel integrated graphics and one discrete-GPU system.  Repeat the performance
section with both PyQtGraph and VisPy selected.

## Basic presentation

- Open scalar, RGB, and complex 2D arrays; verify orientation, axis flips, fit, and 1:1.
- Change channel/component, colormap, brightness/contrast, and histogram levels.
- Verify level-only changes do not increment VisPy texture-upload counters.
- Switch render backends repeatedly without stale pixels, levels, geometry, or crashes.

## Tiled montage

- Load a 272×336×336 or similarly large montage progressively.
- Verify every tile identity while batches arrive and after panning away/back.
- Confirm atlas rebuild count remains bounded for a stable planned visible set and that page count
  only grows when the visible/near plan exceeds the current page capacity.
- Zoom into a small region, pan elsewhere, then return; resident tiles should reappear without upload
  unless pressure caused a reported eviction.
- Narrow the montage index range to an overlapping subset and scroll that range; sources that remain
  resident should redraw in their new tile positions with zero texture uploads.
- Pan between nearby regions under pressure; viewport-near resident tiles should survive before
  farther inactive tiles.
- Verify loading/skipped overlays and tile gaps at all zoom levels.
- Exercise scalar, already-windowed RGB, CPU display-ready RGB, and raw complex `RG32F` tile modes.
- For raw complex raster and tiled montage, verify phase color diversity, LUT changes, and
  linear/log/symlog level changes without texture re-upload.
- Zoom far out until tiled LOD is selected, then pan and zoom repeatedly. Verify diagnostics keep the
  expected nonzero LOD level/factor and do not intermittently fall back to level 0 unless the view is
  actually zoomed in.
- Check tile seams under linear filtering; gutters should prevent neighboring-tile color bleed.
- Pan across an already resident montage region. Tiles that were visible or warmed recently should
  appear immediately without dropping out, showing loading overlays, or scheduling redundant renders.
- Confirm diagnostics show expected storage mode, resident/capacity, GPU bytes, zero CPU shadow bytes,
  texture submissions, vertex submissions, pages, active pages, derived budget, runtime max texture
  size, near/warm resident counts, rebuilds, evictions, and capacity warnings.
  Also check tile payload build time, LOD level/factor, gutter pixels, mipmap fallback status, complex
  texture upload count, and shader uniform update count.

## Interaction parity

- Live profile crosshair: center dot visible, hover highlight/cursor, real-time motion, aligned profile.
- Rectangle ROI: body drag, bottom-right resize handle, handle hover/cursor/highlight, real-time outline.
- Line ROI: both endpoint handles, line drag, endpoint resize, real-time outline and statistics.
- Polyline/freehand: live drawing preview, vertex hover and drag where supported.
- Verify a handle wins hit testing where it overlaps an ROI line/body.
- Verify hover state clears when changing tools or leaving the canvas.
- Verify pixel/status HUD follows the mouse and reports value plus dimensions/tile context.
- Verify ROI information box remains above the VisPy canvas and updates type, mean, N, and other stats.
- Pan/zoom while overlays exist; verify alignment and no lagging duplicate mirror.

## Responsiveness

- During initial progressive load, continuously open menus, drag docks, pan, zoom, and move an ROI.
- Record diagnostics for maximum UI gap, first frame, commit time, upload bytes, and atlas rebuilds.
- Run:

  ```bash
  ARRAYSCOPE_RUN_STRESS=1 \
  ARRAYSCOPE_BENCH_PRESENTED=1 \
  python -m arrayscope.display.rendering_benchmarks --presented --stress --runs 3 \
    --jsonl tests/artifacts/rendering-stress-local.jsonl
  ```

- For baseline scenarios, run:

  ```bash
  python -m arrayscope.display.rendering_benchmarks --presented --runs 5 \
    --jsonl tests/artifacts/rendering-baseline-local.jsonl
  ```

- Repeat on each target OS/compositor/GPU class; merge JSONL samples and compare medians and tail
  latency, not one run.
- Check for frame pacing at monitor refresh rates with vsync on/off where the platform supports it.
- Treat GPU utilization as supporting evidence only; correlate it with missed frames and submission
  stalls.

## Memory and lifetime

- Repeat large montage load, backend switch, data reload, and window close cycles.
- Observe process RSS and GPU allocation; verify no monotonic atlas/surface leak.
- Test low GPU-memory pressure and oversized tile counts; failure must be graceful and diagnostics
  must explain budget, page, device-limit, or capacity limitations.
- Modify/reload source data and verify source identities force the correct dirty tile uploads.

## Platform-specific

- Wayland: HUD/ROI panel must remain above the GL surface; no black/transparent stacking artifacts.
- X11: verify mouse-transparent interaction surface and cursor changes.
- Windows/macOS: verify OpenGL context recreation after hide/show, dock changes, and screen changes.
- HiDPI: handles and hit targets remain screen-sized and crisp at multiple scale factors.
