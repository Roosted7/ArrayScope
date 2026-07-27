# R5 bulk render governor evidence — 2026-07-27

## Ownership

`ResourceGovernor.decide_render_pass()` owns the item cap, byte cap, 32 ms
target, 50 ms hard evidence threshold, and the preview/target feedback
channels. `FramePipeline` produces preview tiles through its ordinary governed
continuation instead of one whole-round worker task. Presentation effects pass
the governor decision through unchanged. Backends may stop earlier at the
deadline; they cannot widen the cap or deadline.

PyQtGraph prefixes use ordinary hidden-per-item staging. The compact atlas is
still a complete hidden transaction and is never exposed as partial truth, but
production cannot enter that atomic branch unless the governor admits the
entire planned set in one measured chunk. WGPU has the same full-set admission
guard for its preview atlas.

Every backend completion emits `pass_kind`, `pass_chunk_items`,
`pass_chunk_budget_ms`, `pass_chunk_within_50ms`, and
`pass_completed_atomically`, plus payload-build, backend, acknowledgement,
state-publication, and geometry timings. `presentation_build_timing` further
attributes governed payload construction.

## Weston measurements

All runs used the managed private Weston compositor, the deterministic geometry
scene, a 272-tile plan, a 5 s hard interaction timeout, and the final code in
this branch. The table reports every observed pass chunk before the hard
timeout; slower governed passes intentionally do not finish all 272 tiles
within five seconds.

| Backend/pass | Observed chunks/items | min / p50 / p95 / max (ms) | chunks > 50 ms | atomic pass completions |
|---|---:|---:|---:|---:|
| PyQtGraph scalar preview | 25 / 260 | 8.963 / 21.021 / 25.026 / 29.462 | 0 | 0 |
| PyQtGraph scalar target | 122 / 122 | 8.598 / 13.945 / 20.425 / 24.806 | 0 | 0 |
| WGPU scalar preview, screen | 73 / 188 | 12.442 / 42.482 / 53.235 / 61.955 | 10 | 0 |
| WGPU scalar target, screen | 118 / 118 | 12.554 / 27.480 / 45.590 / 47.842 | 0 | 0 |
| PyQtGraph FFT target | 63 / 65 | 14.537 / 20.501 / 47.405 / 154.923 | 3 | 0 |

Commands were `python -m arrayscope.tools.headless_display --width 600
--height 800 -- python -m arrayscope.tools.profile_montage_workflow
--synthetic-scene geometry --synthetic-shape 336x336x272 --max-tiles 272
--timeout-s 5 --stages raw_full_tiled_montage`, selecting each backend and
`--enable-coarse-rung` or `--disable-coarse-rung`; WGPU used
`--wgpu-present-method screen --texture-codec off`. The FFT row substituted
`fft_full_tiled_montage`.

## Result and remaining red evidence

No preview or target pass completed atomically in these runs. The former
PyQtGraph compact-preview whole-atlas burst and FFT whole-active-set
republication are gone: the backend resolves only the admitted upserts and
retains already drawn items.

R5 is green for the scalar PyQtGraph target and for all observed WGPU scalar
target chunks. It remains honestly red for two indivisible per-tile costs:
WGPU full-binding republication grows to 50–62 ms late in the preview, and
three PyQtGraph FFT `ImageItem` updates took 63–155 ms even at the governor's
one-item cold-start cap. The governor records and exposes these failures; it
cannot subdivide one backend item update. Closing those residuals requires an
incremental WGPU binding publication and moving/splitting PyQtGraph's
per-item complex window/upload work, not a wider deadline.

