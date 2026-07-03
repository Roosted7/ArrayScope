# X5a hardware telemetry — Linux/Wayland reference traces (2026-07-03)

First real-hardware evidence pass for the X5 gates. All runs executed on a
live GNOME Wayland session (no Xvfb, no software GL) on a hybrid-graphics
laptop, using the in-repo harnesses:

- micro/stress matrix: `python -m arrayscope.display.rendering_benchmarks --runs 5 --presented --jsonl ...`
- real-app workflow: `python -m arrayscope.tools.profile_montage_workflow --backend all --jsonl ...`
- interaction cadence: `python -m arrayscope.tools.profile_scroll_input --backend {pyqtgraph,vispy}`

Raw JSONL, logs, and cProfile artifacts live in `tests/artifacts/x5a-20260702/`
(gitignored; regenerate with the commands above).

## Environment

| | |
|---|---|
| OS / session | Arch Linux, kernel 7.0.13-zen1, GNOME on Wayland (`wayland-0`) |
| Qt | PySide6 6.11.1, pyqtgraph 0.14.0, VisPy 0.16.2, Python 3.14.6 |
| iGPU | Intel TigerLake-H GT1 UHD (Mesa 26.1.3), `GL_MAX_TEXTURE_SIZE` 16384 |
| dGPU | NVIDIA RTX A2000 Mobile (PRIME offload), `GL_MAX_TEXTURE_SIZE` 32768 |
| Configs | wayland+Intel, wayland+NVIDIA, xcb(XWayland)+Intel, xcb+NVIDIA |

Telemetry fixes made during this pass (all X5a exit-gate items):

- `query_gpu_device_limits` never worked against VisPy ≥ 0.14 (`gloo.gl` has no
  `glGetString`); every environment record silently reported the 4096
  fallback. Real limits are 16384 (Intel) / 32768 (NVIDIA) — the fallback was
  wrong by 4–8×. Fixed to query through `glGetParameter`, to reject a
  contextless 0 answer, and to open a short-lived context when the
  benchmark views are already closed.
- Environment records now carry the GPU vendor/renderer/version strings and a
  correct Qt version on PySide6.

## Micro benchmark matrix (medians of 5 presented runs)

`pg` = PyQtGraph, `vp` = VisPy; `submit` = CPU submission ms, `ff` =
first presented frame ms. Full per-config tables are in the JSONL artifacts;
the wayland+Intel table is representative (the backend ordering is identical
in all four configs):

| scenario | pg submit | vp submit | pg ff | vp ff | first-frame winner |
|---|---:|---:|---:|---:|---|
| tiled_small_initial (128²) | 2.6 | 5.4 | 9.6 | 6.6 | vispy 1.5× |
| one_tile_montage_initial | 1.9 | 5.3 | 12.5 | 6.8 | vispy 1.9× |
| multi_tile_montage_initial (16) | 7.9 | 6.4 | 21.7 | 7.7 | vispy 2.8× |
| tiled_large_initial (1024²) | 16.7 | 9.2 | 32.0 | 10.7 | vispy 3.0× |
| large_complex_tiled_initial (128 tiles) | 43.4 | 14.2 | 63.3 | 15.6 | vispy 4.1× |
| progressive_tile_stream (96 tiles) | 41.9 | 26.3 | 59.3 | 27.7 | vispy 2.1× |
| tile_level_uniform_update | 3.5 | 0.5 | 19.8 | 2.4 | vispy 8.2× |
| large_tile_level_preview (96 tiles) | 11.0 | 0.7 | 29.1 | 2.2 | vispy 13.3× |
| pan_zoom_no_upload | 0.2 | 2.2 | 8.0 | 3.9 | vispy 2.0× |
| stress 272-tile progressive montage | 458–512 | 127–166 | 570–624 | 129–168 | vispy 3.5–4.4× |

Observations:

- **VisPy presents first frames earlier in every scenario measured**, on both
  GPUs and both Qt platforms. Its advantage is largest exactly where the
  physical mechanics differ: level-only changes are shader-uniform updates
  (no CPU re-window, no re-upload, resident textures stay valid), and large
  tiled commits scale with the batch instead of with CPU pixel conversion.
- PyQtGraph still has the cheapest CPU submission for tiny single-tile
  commits (~2 ms vs ~5 ms) and for pan/zoom without uploads, but its
  presented first frame still lands later than VisPy's in those cases.
- VisPy's worst metric is event-loop gap on small fresh canvases
  (one-time canvas/shader initialization per view, ~30–60 ms); in the real
  application this is a per-window cost, not a per-commit cost.
- On this hybrid laptop the NVIDIA PRIME offload path is *slower* to first
  frame than the Intel iGPU for every scenario (offload copy overhead);
  neither backend showed instability on either GPU. Do not force dGPU
  offload by default.

## Real-application workflow (272-tile montage + FFT + level drag)

`profile_montage_workflow --backend all`, production `ArrayScopeWindow`:

| phase | pyqtgraph | vispy |
|---|---:|---:|
| raw_full_tiled_montage | 1054 ms (max gap 87 ms) | 1140 ms (max gap 85 ms) |
| fft_full_tiled_montage | 2936 ms (max gap 409 ms) | 1797 ms (max gap 140 ms) |
| fft_level_refinement_preview | 8114 ms (max action 1788 ms) | **257 ms (max action 18 ms)** |

The level-refinement phase is the decisive real-life difference: a window/level
drag over a 272-tile FFT montage settles in ~0.26 s on VisPy (uniform updates)
versus ~8 s on PyQtGraph (CPU re-window of every tile) — **after** the fixes
below; before them the PyQtGraph phase never converged inside the 180 s
timeout.

## Interaction cadence (60 Hz FFT slice scrolling)

`profile_scroll_input`, 180 ticks: VisPy presented 123 renders
(mean 6.3 ms, p95 7.9 ms, timer p95 24 ms) vs PyQtGraph 96 renders
(mean 7.8 ms, p95 9.2 ms, timer p95 31 ms). Moderate VisPy advantage; both
stay interactive.

## Defects found and fixed during this pass

1. **GPU limit query broken** (above) — every policy consumer saw a fixed
   4096 texture limit, violating the "no fixed assumed max texture size"
   exit gate.
2. **VisPy per-commit histogram concatenation.** `setTiledPresentation`
   concatenated every visible tile's histogram data on every commit
   (O(n²) across a progressive stream — 47 % of stress submit CPU) and the
   fresh array identity forced a histogram repaint per commit. The fallback
   is now materialized lazily through the coalescing histogram timer, keyed
   by payload histogram identity. Stress submit: 1180 ms → ~130 ms.
3. **VisPy resident-key recomputation.** `_resident_key` was recomputed up to
   four times per payload per commit, dominated by `str(np.dtype)`; it is a
   pure function of immutable payload identity and is now memoized on the
   payload (update_payloads cost −70 %).
4. **PyQtGraph level-refinement starvation (pre-existing, real-hardware
   only).** After a level change on a large montage, each governed commit
   re-windowed ~2 tiles: the latency-feedback budget collapses to its 2 ms
   floor because whole-commit elapsed time is dominated by an O(visible)
   fixed pipeline cost the batch size cannot control, and the backend's cold
   deadline then cut re-window work after ~2 tiles. A 272-tile montage
   drained at ~6 tiles/s and never converged inside the profiler timeout.
   Level re-windowing now uses a floored refinement deadline
   (`max(8 ms, cold budget)`) in the PyQtGraph tile layer, and
   `build_tile_presentation` prioritizes only the stale candidates instead
   of re-ordering every active tile per commit. Settle time: never → 4.3 s.
   The remaining fixed cost (~40 ms per commit, O(visible tiles) rebuild in
   the commit pipeline) is X5c/X5d work and is documented there.
5. **`mark_presented` was O(n²)** (per-tile frozenset extension); now batches
   the level-scope update.

## Decision input

See [ADR 0047](../decisions/0047-auto-image-backend-selection.md): with these
traces, `image_rendering_backend=auto` resolves to VisPy on Linux when a real
hardware GL context is available, and to PyQtGraph everywhere else (software
GL, offscreen, probe failure, platforms without reference traces). The
explicit `pyqtgraph`/`vispy` settings remain available and unchanged.

Not yet earned by this pass (stays on the X5 roadmap): Windows/macOS traces,
context-loss/allocation-failure conformance (X5b), viewport-scoped normal
images (X5c), region-first materialization and the singleton/direct strategy
policy (X5d), LOD enablement (X5e).
