# R8 status

Updated: 2026-07-14

## Scope

This checkpoint covers only R8A viewer truth. It does not change scheduling,
throughput policy, benchmark code, or LOD admission policy.

The work started from clean R7+UI commit `906e5c3c`. The dirty
`redesign-r8-marathon` worktree was inspected as reference only and was not
modified.

## Confirmed causes

1. Tile lifecycle acceptance and backend reports used an opaque `source_id`.
   That identity did not express document/operation/source/axes/flips/channel,
   complex mapping, texture kind, semantic generation, or LOD, so incompatible
   complex tiles could be retained as if current.
2. VisPy replaced a bare `PHASE_COLOR` mapping's canonical cyclic LUT with the
   image view's initial scalar grayscale LUT. The complex texture identity and
   shader mode were correct, but the final presentation mapping was not.

Both hypotheses were confirmed; neither experiment was reverted.

## Landed invariant slices

- `500bcdc1` restores R7 frame API coherence needed to run the viewer path.
- `44b77892` adds typed tile target and backend-acknowledgement identities.
- `6228a4f9` aligns the diagnostics snapshot with its ledger field contract.
- `257e5be2` enforces typed target/acknowledgement truth immediately before
  either backend draws a tile.
- `bc8b7477` removes the stale paced-admission keyword that crashed the live
  presentation gate.
- `adf81fd4` attaches the live presentation effects before level side work can
  schedule a commit.
- `b49f67c3` aligns predicted and acknowledged texture kinds with the storage
  each backend physically draws. Scalar source data uses scalar storage;
  scalar components of complex source data use RG32F only with an explicit
  complex component shader mode.
- `db2aaa3f` adds the six-pattern adversarial complex fixture.
- `a78adbf` adds the opt-in, backend-neutral tile truth HUD sourced directly
  from lifecycle diagnostic rows.

The current R8A slice adds the pre-draw truth firewall:

- `TilePresentationDelta` carries lifecycle-owned typed targets for every
  active tile.
- PyQtGraph and VisPy draw only an acknowledged identity that satisfies that
  target. An incompatible or opaque acknowledgement is hidden, exposing the
  zero/loading placeholder beneath it.
- Exact same-semantic content and explicitly compatible coarser LOD remain
  valid progressive fallbacks. Presentation levels/LUT generation stays
  separate from pixel identity.
- Per-tile diagnostics record target identity, acknowledged identity, texture
  kind, real/imag plane provenance, complex mapping, LOD, levels generation,
  and drawable/placeholder state.
- VisPy preserves the canonical phase LUT for a bare phase-color mapping and
  still honors an explicit mapping LUT or an explicitly applied view colormap.

## Evidence

Synthetic complex fixture:

- Four deterministic complex phase/magnitude tiles are presented, followed by
  a back-to-back semantic source transition where only one successor is ready.
- PyQtGraph and VisPy both present only tile 0; tiles 1-3 are placeholders.
- After all successor payloads arrive, both backends present all four exact
  typed identities.
- `tests/display/test_complex_tile_truth.py` records and asserts all R8A truth
  fields for each tile.

Real fixture standalone:

- File: `/home/thomas/projects/UMC/scan2go/ClinicalScans/MrT/_WIPDelRec-tT2_20260223150234_14.nii`
- Loaded shape/dtype: `(336, 336, 272)`, `float64`; smoke window: first eight
  slices transformed with centered FFT, four tiles displayed at a time.
- Session: Wayland, Python 3.14.6, PySide6 6.11.1, PyQtGraph 0.14.0,
  VisPy 0.16.2, NumPy 2.5.1, NVIDIA RTX A2000 Laptop GPU, driver 610.43.03.
- Typed partial transition: `presented=[0]`, placeholders `[1,2,3]` on both
  backends; final transition: `presented=[0,1,2,3]` on both backends.
- Visual inspection shows equivalent cyclic phase color and source layout in
  `tests/artifacts/r8a-real-fixture-screens/pyqtgraph-complex-truth-final.png`
  and `tests/artifacts/r8a-real-fixture-screens/vispy-complex-truth-final.png`.

Automated validation:

- `tests/display`: 515 passed.
- R8A-relevant `tests/window` slice: 293 passed.
- `tests/presentation` plus runtime diagnostics: 60 passed.
- Focused FrameSession/diagnostics UI slice: 15 passed.

## Known validation limitation outside R8A

The full profiling workflow is not acceptance evidence for this slice. After a
narrow diagnostics-name repair, its R7 viewport harness materialized all eight
tiles but timed out with `active_presented=0/0` and no pending/loading/dirty
work. Three existing window tests describe the same out-of-scope scheduling or
viewport drift and were deliberately not repaired here:

- `test_vispy_persistent_upsert_limits_use_governed_upload_limit`
- `test_retarget_index_window_demotes_misses_with_immediate_invalidation`
- `test_tile_presentation_limits_cap_resident_retarget_upserts`

Changing those policies would violate the R8A truth-first scope. The direct
real-backend standalone smoke above is the current visual and semantic gate.

## R8B checkpoint after reverted experiment

The committed adversarial fixture now covers constant-magnitude phase ramp,
constant-phase magnitude ramp, real-only, imaginary-only, zeros, and a known
complex source signature. PyQtGraph's accepted RGB pixels match the CPU
reference; VisPy's accepted RG32F uploads match the exact real/imag planes.
Committed-value tests verify native complex data and exact magnitude values,
and the same truth HUD is exercised on both backend surfaces.

One uncommitted full-`ArrayScopeWindow` synthetic test was attempted and then
removed on 2026-07-14. Both PyQtGraph and VisPy timed out waiting for
`visible_plan_complete()` plus zero visible presentation obligations. This is
the already recorded R7 active-viewport/full-window settlement limitation, not
evidence that the six-pattern backend fixture failed. The VisPy offscreen run
also lacked a supported `QOpenGLWidget`, so exact framebuffer sampling remains
a real-hardware gate.

Per the R8 working rule, work stopped here after that reverted experiment. The
next hypothesis must address full-window convergence/settlement as an R8C
truth-and-convergence slice; the benchmark and scheduling/throughput policy
remain untouched.

### Rejected overlay presentation

The first `a78adbf` overlay presentation is not accepted as the R8B debug
overlay. It renders all lifecycle rows in one global HUD, so the identity is
not spatially attached to the tile whose pixels it describes. That makes it
too difficult to correlate a flash with one slot and fails the stated
per-tile requirement. The lifecycle rows remain valid evidence, but the UI
must be replaced by one overlay on each visible tile for both PyQtGraph and
VisPy.

Work stopped again at this checkpoint before changing the overlay. Two other
confirmed issues are intentionally kept separate from that replacement:

- `seed_display_tile_payloads()` retargets a reused wrapper's
  `tile_number`/`source_index` without rebuilding its typed `tile_identity`.
  A slot-3 payload can therefore carry acknowledged source 0 and is correctly
  rejected forever by the truth firewall.
- `montage_prefetch.py` still calls the removed
  `_is_current_montage_session()` API from a completion callback; the
  canonical method is `_is_current_frame_session()`.

## Remaining R8 work

- Exercise the typed firewall through the full ArrayScopeWindow real-file path
  once the existing R7 active-viewport harness contract is repaired in its own
  scoped slice.
- Run exact selected-pixel framebuffer comparison for the six-pattern fixture
  on real OpenGL hardware for VisPy.
- Repair full-window settlement from lifecycle evidence before using that
  harness for the R8B hover/ROI and R8C transition gates.
- Add the R8C semantic/quality/presentation/viewport transition matrix only
  after the full-window convergence gate is green.
