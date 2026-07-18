# wgpu field stalls 259-1 and 1-1 — physical first-pass quality drift (2026-07-18)

**Status:** root-caused, fixed, and gated on
`codex/wgpu-field-stalls-259-1` (`43287f8`).

## Field evidence and disambiguation

Thomas hit two real-app, real-data wgpu watchdog stalls on 2026-07-18:

- `tests/artifacts/field-stalls-wgpu-2026-07-18/arrayscope-stall-259-1.trace.jsonl`
  (19:15 session, before the 19:19 axis-flip fix): `trace_verify` sees 4,435
  events, 20 final targets, 19 exact acknowledgements, and stall session 259.
  The final owner chain has one required target unsettled (`tile 10`), six
  dirty/pending upserts, `flush_pending=final_commit_pending=True`, and no
  kernel/stage work.
- `tests/artifacts/field-stalls-wgpu-2026-07-18/arrayscope-stall-1-1.trace.jsonl`
  (19:21 session, after the axis-flip fix): 1,029 events, 60 final targets,
  zero exact acknowledgements, and stall session 1. Its final owner chain is
  fully idle: no dirty/upserts, no evidence, no draw/commit gate, but all 60
  exact targets remain unsettled.

These are **not** the standing backend-agnostic 272-tile raw-fill stall. That
stall presents 0/272 with a large commit backlog on PyQtGraph, VisPy, and
wgpu. Here wgpu physically presented every required tile before the watchdog:
`1-1` reached 60/60 physical tiles (mixed L6 floors and L2 previews), while
`259-1` reached 12/12 required first pixels and left only one exact target
open. Neither trace contains identity rejection or phase-2 work admitted
during coverage.

The older `/tmp/arrayscope-stall-{19-2,24-3,29-4}.trace.jsonl` files are
timestamped 17:54–17:55. They include the 17:45 late-evidence absorption
change (`9f8b3970`) but predate loud queue bails (`7567fb3a`, 17:56), retained
fallback coverage (`14633cd0`, 18:18), and the final first-pass aggregation
fix (`6670b9df`, 18:38). They are therefore pre-fix evidence and were not used
to classify the two current field stalls.

## One mechanism, two terminal signatures

Both traces stop after the same edge:

1. `ProgressiveSchedulingPolicy` emits `coverage_evidence_pending`.
2. The wgpu resident-histogram tasks complete. Later queue attempts bail
   loudly as either `evidence_inflight` or `no_waiting_evidence_rows`; these
   reasons are healthy consequences, not the defect.
3. The physical required set changes from all `quality="exact"` payloads to
   a mixture of retained exact payloads and `quality="preview"` fallbacks.
   `FrameSession.first_pass_quality` was write-once, so it stayed `exact`.
4. `first_pass_pixels_presented()` rejects the preview member of that mixed
   set. Therefore `_first_pass_level_evidence_complete()` can never publish
   the already-completed histogram evidence, the policy never emits
   `coverage_evidence_ready`, and exact/refinement work remains barred.

`1-1` then exhausts every ordinary owner and goes idle. `259-1` has the same
barrier plus a same-topology atomic successor: six off-viewport upserts keep
the flush armed, producing empty backend commits with
`atomic_fast_reject_reason="payload:2"`. That loop is an amplifier, not a
second root defect; opening the first-pass barrier lets the existing owner
finish the successor.

## Root fix and regression

The acknowledged backend snapshot remains the single first-pass quality
owner. `FrameSession.observe_physically_presented_first_pass_quality()` now
widens an earlier exact latch to preview when a later physically acknowledged
snapshot contains preview/fallback payloads. This is conservative and
monotonic: a preview pass accepts exact overlap, but exact never silently
accepts preview. The transition emits a loud
`first_pass_quality/event=widened_to_preview` trace row.

Red-first gate:
`tests/ui/test_frame_session.py::test_physical_preview_widens_latched_exact_first_pass`.
It first acknowledges a complete exact frame, physically replaces one tile
with a preview, and requires the canonical owner to widen the pass, report
completion, and emit the trace edge. Pre-fix it fails with
`first_pass_quality == "exact"`.

## Reproduction and validation

- Existing offscreen wgpu scroll-back driver settles after the fix:
  `tests/ui/test_montage_scroll_settling.py::test_wgpu_scalar_scroll_back_settles_retained_fallbacks_to_exact`.
- Field-shaped offscreen profile (real NIfTI, wgpu, FFT and scalar scroll)
  physically presented every tile and its trace had no invariant violation
  other than the deliberately mismatched `--expect-targets` probe. The
  profile still reports its separate scalar `presentation_continuity` red and
  an FFT step over the 5 s budget; neither is claimed fixed here.
- Offscreen journey smoke after the fix: all five wgpu rows green (cold fill,
  zoom in, zoom out, scroll shuffle, ten index-scroll gestures). The overall
  three-backend smoke remains red only in pre-existing/diagnostic VisPy and
  PyQtGraph rows. Independent `trace_verify --expect-targets 60` is green for
  both wgpu scroll and zoom traces with zero violations.
- Focused offscreen owner/backend coverage: 101 passed
  (`test_frame_session`, wgpu scroll settling, wgpu view, and the resident
  histogram/publication tests).
- Full offscreen suite: **2,424 passed, 36 skipped, 1 xfailed, 0 failed**.
- Real Wayland (`wayland-0`, Vulkan-only wgpu instance): the retained-fallback
  scroll regression passed; the real NIfTI FFT+scalar scroll profile produced
  physical screenshot timelines with complete tile grids and a 15,053-event
  trace. `trace_verify --expect-targets 20` reports 20/20 exact targets,
  zero identity rejection, zero phase-2 submission during coverage, zero
  stalls, and no violations.

The real-data profile remains red on the pre-existing performance bars
(`presentation_continuity` for FFT; GUI callback/heartbeat timing for both
stages, with 20/20 tiles ultimately presented). This stall fix makes no
performance claim and does not close queue row 3(d).
