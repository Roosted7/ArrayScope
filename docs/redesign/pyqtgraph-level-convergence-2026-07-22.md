# The montage-relevel "red" — diagnosed 2026-07-22 (NOT a bug; a pyqtgraph throughput fork)

Answers "is the level_stale red accurate?" — yes, but the premise that it is a
convergence *bug* is wrong. This is a definitive diagnosis (headless-Weston
trace-only, instrumented per-commit); no engine change was committed.

## Symptom (reproduces trace-only, pre-existing on de5862d4)
`profile_montage_workflow --backend pyqtgraph` on the FFT montage times out:
`active_presented=272/272 fully_visible=True … level_pending=True level_stale=~250 level_values=2`.
The pixels are fully up; the per-tile window/level never finishes converging
within `INTERACTION_SETTLE_HARD_LIMIT_S = 5 s`.

## Root cause: pyqtgraph CPU-windowing throughput, not a convergence defect
Ruled out (with per-commit trace evidence): (b) bookkeeping — `acknowledge_upserts`
records `tile_values`/`tile_revisions` correctly, committed tiles converge and
leave the stale set; (c) unreachable target — the target is reached by every
committed tile (`level_values=2` is just old+target during crossover); (a)
never-rescheduled remainder — `stale_level_tiles` is recomputed every drain, the
gate re-arms each commit, and `level_stale` decreases monotonically (272→269→266…).

What actually happens: pyqtgraph **bakes levels into pixels**, so each level
refinement commit re-windows complex FFT tiles to RGB on the CPU. Measured:
~180 ms/commit, converting only ~1–4 of 12 emitted tiles, dominated by *fixed
O(N) per-commit overhead* (`set_image` ~100 ms + `payload_build` ~60 ms:
re-prioritizing all 272 stale candidates + the backend's full-residency
geometry walk), largely independent of tiles actually converted. Net
≈ 272 × 180 ms ÷ ~4 ≈ **45 s** to drain vs the 5 s budget.

**The GPU backends do not have this.** `--backend vispy` passes the
`fft_level_refinement_preview` phase in ~447 ms (levels are one GPU shader
uniform — `acknowledge_uniform_level_presentation`, no per-tile CPU rebake);
vispy's own `exit 1` is from *unrelated* phases (gui-callback/heartbeat/zoompan
LOD — the pre-existing 4 fps-scroll perf items). wgpu is 6/6 green in the
2026-07-22 journey matrix and uses the same GPU-uniform level path. So the red
is confined to the **CPU-windowing pyqtgraph backend** — the headless/remote
target — and matches the standing note *"PyQtGraph bakes levels into pixels, so
its auto-levels crawl tile-by-tile… pyqtgraph set aside; VisPy only for now."*

## The design fork (why no fix was committed)
Hard tension between two live invariants:
- R8 `gui_callbacks_below_50ms` wants *small* per-commit work (pyqtgraph level
  commits already breach it at ~180 ms).
- The 5 s settle budget wants *few, large* commits to drain 272 tiles.

Options (ADR-level):
1. **Level-only commit fast-path** (recommended by the diagnosis, lowest-regression
   because the bottleneck is the fixed O(N) overhead, not the rewindow itself):
   when a commit only drains `stale_level_tiles` (no new payloads, all required
   pixels present), skip re-prioritizing all 272 candidates (cache/window the
   order) and skip the full-residency geometry walk (iterate only the requested
   upsert slice). Keeps each callback <50 ms AND cuts commit count enough to
   settle in ~2 s. Substantial, delicate restructuring of `build_tile_presentation`
   + the pyqtgraph `update_direct` path.
2. **Accept slow-but-correct pyqtgraph relevel** and give the montage-relevel
   phase its own (wider) harness budget on the CPU-windowing backend only.
3. **Treat pyqtgraph complex-montage relevel as VisPy/wgpu-only** — aligns with
   the roadmap's VisPy-retirement / wgpu-promotion / pyqtgraph-for-remote verdict.

Key code (for whoever implements option 1): `frame_session.py`
`build_tile_presentation` (~2642), stale-candidate build (~2744–2759);
`frame_effects.py` `_commit_tile_layer` (~1517) + `tile_layer_upsert_limits`
budget knobs (~4242); `display/backends/pyqtgraph/tiles.py`
`level_rewindow_deadline_ms` (~628), the resident-tile walk + rewindow branch
(~660, ~941–979), `_update_tile_levels` (~1512).
