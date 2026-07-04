# LOD resident-policy A/B traces — 2026-07-04

First hardware evidence for ADR 0050's opt-in `resident` montage LOD policy on the VisPy
backend. Live Wayland session, `arrayscope` conda env, dataset
`data/_WIPDelRec-tT2_20260223150234_14.nii` (336×336×272 f64), full 272-tile montage of
axis 2 in a 1400×900 window via `arrayscope.tools.profile_montage_workflow`
(`--backend vispy --montage-lod-policy {native-only,resident}`), branch
`feature/lod-residency` @ 82573ba1. Desired LOD factor is 4 in every montage phase.

| Metric | native-only | resident |
|---|---:|---:|
| Applied factor (presented plurality) | 1 | 4 |
| Resident tile levels | 272 @ L0 | 272 @ L2 |
| Ingest reductions (raw phase) | 0 | 272 |
| Raw montage settle | 1216 ms | 1419 ms |
| Raw montage upload bytes | 2.26 MB | 0.08 MB |
| Raw montage est. tile GPU bytes | 135.5 MB | 70.6 MB |
| FFT montage settle | 1630 ms | 1875 ms |
| FFT montage est. tile GPU bytes | 280.0 MB | 34.8 MB |
| FFT 10× level-drag preview | ~294 ms | ~249 ms |
| Event-loop gap p95 (raw / FFT) | 84 / 112 ms | 75 / 101 ms |
| Direct tile compute (raw) | 512 ms | 920 ms |

Reading:

- The resident path presents every cold tile at the demanded level on first upload
  (ingest reduction on the evaluation worker), so texture uploads collapse and tiled GPU
  residency drops 2–8×. Event-loop gap p95 improves slightly.
- Settle time regresses ~15 %: worker-side reduction adds ~1.5 ms per tile of measured
  compute (raw box-mean is ~0.32 ms per 336² f64 tile on this machine; the rest is
  admit/key/bookkeeping plus GIL contention). Acceptable for an opt-in policy whose
  target is zoomed-out interaction and residency, not cold-fill latency; revisit before
  making resident the default.
- An earlier build without ingest reduction (streamed second-pass refinement only)
  regressed settle 1216→1562 ms *and* grew GPU bytes; ingest reduction is what turned
  residency and uploads into wins. Traces: `/tmp/lod-baseline/ab2_*.jsonl` (rerunnable).

Not yet evidence for: PyQtGraph reduced-image adoption (phase 3), ops-input LOD
(`lod-commuting`, phase 4), zoom-in refinement latency, Windows/macOS.
