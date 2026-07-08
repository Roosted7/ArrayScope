# Operation reduced-input (display-LOD) compatibility audit

Reduced-input display preview (ADR 0050) evaluates the operation pipeline on
box-mean-reduced display axes to produce cheap first pixels, instead of
computing native output and downsampling. Two gates decide eligibility
(`arrayscope/operations/capabilities.py`):

- `pipeline_commutes_for_display_lod` — the strict per-tile path: **every** op
  must be shape-preserving *and* declare `lod_commuting`.
- `pipeline_supports_reduced_display_lod` — the broader path (used by the
  shared transform): a non-`lod_commuting` op is still fine if its affected
  axes (`blocking_axes ∪ expands_request_axes ∪ declared axes`) lie entirely
  outside the display axes — **but any shape-changing op (`output_shape !=
  input_shape`) is rejected unconditionally**, whatever axis it changes.

## Current op status

| Op | Kind | Shape | `lod_commuting` | Reduced-input display? |
|---|---|---|---|---|
| Conjugate | elementwise | preserves | yes | ✅ per-tile |
| CenteredFFT / CenteredIFFT | transform | preserves | no (blocks its axis) | ✅ when axis ∉ display (shared transform) |
| FFTShift | view | preserves | no (blocks its axis) | ✅ when axis ∉ display |
| ReverseAxis | view | preserves | no (blocks its axis) | ✅ when axis ∉ display |
| Mean / Sum / RootSumSquares / Maximum / Minimum | reduction | **removes axis** | n/a | ❌ blocked (shape change) |
| Crop | view | **changes length** | no | ❌ blocked (shape change) |
| CombineRealImagAxis / SplitComplexAxis | view | **changes length** | n/a | ❌ blocked (shape change) |

The common FFT-over-montage-axis workflow (FFT/shift/iFFT with x/y as display
axes) **is** reduced-input compatible via the shared transform — that path is
correct today.

## Gaps (defer improvements to R4 / focused follow-up)

1. **Shape-change is over-rejected.** `pipeline_supports_reduced_display_lod`
   rejects *any* shape change, so reducing over a **non-display** axis
   (e.g. `Mean`/`RootSumSquares` over echoes/channels while x/y stay display,
   or `Crop` on the montage axis) needlessly forfeits the cheap reduced path
   and falls back to native. The op already declares its affected axes; the
   gate could allow shape-changing ops whose delta axes ∉ display, **with
   display-axis-index remapping** through the removed/added axis (the reason
   for today's conservatism — the display-axis identity below the reduction
   must be re-derived).

2. **No proportional crop.** `Crop(axis, start, stop)` is an absolute
   index range, which cannot map onto reduced input (indices don't correspond
   after box-mean). A "center X% / N samples" crop *is* proportionally
   map-able on display axes and would be reduced-input compatible — worth
   adding as a distinct op rather than teaching `Crop` two behaviours.

3. **"Op per tile" safety.** When a pipeline is not reduced-input capable,
   `evaluate_preview_tile` → `_evaluate_tile_native_output_preview` runs a
   **full native evaluation per tile** and downsamples the output. That is
   cheap for a lone reduction/crop, but a shape-changer stacked on an
   expensive transform (e.g. `FFT + Crop`) drops the whole pipeline off both
   reduced paths and can evaluate the FFT per tile through a retained-floor
   rung. R4's preview-level selection should take op cost/capabilities as
   input (already noted in `r4-timer-and-governor-audit.md`) and route
   expensive+shape-changing pipelines to the shared transform or a single
   native pass — never per-tile native output.

## Now

The most-used ops (elementwise maps and the FFT chain over a non-display axis)
are compatible. No code change is required for correctness — the fallbacks are
safe (native/shared, never silently wrong). The three gaps above are
performance/coverage improvements tracked for R4.
