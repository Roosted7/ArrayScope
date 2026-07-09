# ADR 0054: Montage level evidence phasing

- **Status:** Implemented for montage level/histogram evidence ordering
- **Date:** 2026-07-09
- **Related:** ADR 0007, ADR 0032, ADR 0040, ADR 0050, ADR 0051, ADR 0053

## Context

Montage display can now show payloads at several quality points before all
analysis work is finished:

- a coarser-than-target preview floor, used as first pixels when target-quality
  work is behind;
- a reduced target pass, where the requested LOD itself is reduced and there is
  no separate preview rung;
- a native/exact target pass;
- later refined histogram/level sampling admitted after visible presentation
  settles.

Before this ADR, montage level state mostly distinguished "has stats" from
"refined stats". That was too coarse. A low-LOD preview sample could suppress a
better target sample for the same semantic source, while a metadata-only
histogram improvement could be skipped because no tile payload delta was ready
to upload. In the user-visible failure mode, tiles often displayed with stale or
default levels and the histogram could remain empty until an unrelated source
reload forced a fresh first commit.

Backend mechanics differ. VisPy can update compatible levels through shader
uniforms without changing resident pixels. PyQtGraph may need CPU-windowed
payload updates, so it must avoid crawling auto levels tile-by-tile during a
large fill.

## Decision

Montage level and histogram evidence is ordered explicitly:

1. **Rough preview**: provisional evidence sampled from a coarser-than-target
   first-pixel payload.
2. **Rough target**: evidence sampled from the currently requested target data,
   including a reduced target LOD when that is the final displayed target.
3. **Refined**: bounded higher-quality sampling run after visible presentation
   has settled.

`MontageLevelTracker` stores this evidence quality per source. Lower-quality
evidence never replaces higher-quality evidence for the same semantic source.
Equal-or-better evidence is reused; lower evidence is not resampled just because
a viewport or payload commit revisits the same tile.

The full selected montage index population remains the semantic expected set.
Visible tiles may seed early evidence, but viewport culling cannot shrink the
semantic histogram domain or downgrade the applied level source.

Presentation uses the best available evidence, with backend-specific
application:

- **VisPy** may publish rough preview levels/histogram data immediately and
  update relative levels through shader uniforms as rough target or refined
  evidence arrives.
- **PyQtGraph** treats rough full/target evidence as the stable source for
  CPU-windowed commits and applies refined evidence later through the normal
  bounded level-presentation path.
- A metadata-only improvement is a valid presentation update even when there is
  no tile data delta. It may refresh uniforms, histogram plot data, and the
  committed level source without uploading new source pixels.

Explicit Auto Window is not a new absolute numeric mode. It clears user override
intent and returns to the default relative automatic mapping using the best
available semantic source. User-locked absolute levels preserve numeric
low/high values while histogram metadata is still allowed to improve.

## Consequences

Positive:

- first visible pixels can use provisional levels without freezing the
  histogram at preview quality;
- target-quality or refined evidence supersedes preview evidence even when the
  source index is already known;
- histogram/level metadata can improve after visible rendering without waiting
  for an unrelated reload or tile upload;
- VisPy keeps its uniform-only level advantage while PyQtGraph keeps bounded
  CPU-windowed convergence;
- tests can assert the semantic evidence ordering independently from backend
  upload mechanics.

Costs:

- level evidence now has a third ordering dimension beyond coverage rank and
  source count;
- metadata-only presentation commits must still pass through the normal
  centralized window-level policy, otherwise user locks and relative mapping can
  diverge;
- benchmark probes can expose remaining settlement/performance issues even when
  level and histogram evidence is correct.

## Rejected alternatives

- **Treat all preview-labelled payloads as provisional forever.** Reduced
  target LODs are sometimes the requested final target, not a temporary preview.
- **Promote preview evidence to refined once visible.** That hides the need for
  a later target/refined pass and can suppress better histogram data.
- **Refresh histogram only when tile payloads upload.** That recreates the empty
  histogram/reload-only failure because level evidence may finish after the
  visible payload delta is already acknowledged.
