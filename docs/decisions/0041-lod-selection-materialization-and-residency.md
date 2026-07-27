# ADR 0041: Separate LOD selection, materialization, and residency

- **Status:** Accepted and implemented through ADR 0050/0056 for the
  maintained WGPU and PyQtGraph backends. VisPy examples below are historical
  after [ADR 0061](0061-retire-vispy-rendering-backend.md).
- **Date:** 2026-06-24
- **Related:** ADR 0037, ADR 0039, ADR 0040

## Context

Tile LOD was introduced by commit `26a63afe492afd6239c36ae207eaa6b975e5349d` on 2026-06-19 at
17:00 CEST. The first implementation selected a factor, built a complete CPU reduction pyramid from
loaded tiles, added gutters, changed payload identity, and presented the result from
`snapshot_display_tile_payloads()`.

That snapshot function runs in the GUI presentation path. Commit
`eb324c75837f1fd28624302bf970fa94b9aedc12`, *Remove synchronous LOD work from presentation commits*,
disabled application of the selected factor at 18:48 CEST the same day. The selector remained active,
but the applied factor was forced to one.

The disablement was correct. The prototype also had structural problems beyond placement:

- every threshold transition rebuilt pyramids for loaded tiles and changed payload/residency identity;
- the first implementation had no transition hysteresis;
- zooming back discarded reduced work, so repeated threshold crossings rebuilt and re-uploaded it;
- reduced tiles plus gutters have different texture dimensions, while the VisPy atlas assumed one
  compatible slot shape;
- mixed dimensions could force storage rebuilds, padding, or incorrect sampling of padded regions;
- PyQtGraph displays `payload.image`, while the prototype primarily changed `texture_data`, so shared
  session churn did not guarantee a rendering benefit there;
- histogram reduction work was built along with texture LOD even though semantic histogram/value
  sources must remain independent of display approximation;
- the current scalar selector uses one isotropic factor from the worst axis and ignores tile shape,
  which is not sufficient for extreme aspect ratios.

Measured in the v30 review environment, building through factor four took roughly 5.5 ms for one
512×512 float tile and roughly 27 ms for one 1024×1024 complex tile. Multiplying that by a visible tile
set before upload is incompatible with an interactive GUI callback.

## Decision

LOD is not one operation. ArrayScope will separate **demand selection**, **materialization**, and
**physical residency**. Native-resolution tiles remain the production policy until all three stages
are implemented and validated.

### 1. LOD demand selection

A Qt-free `LodPlanner` computes desired quality from:

- world units/source texels per screen pixel on each axis;
- viewport dimensions and device-pixel ratio;
- tile/source shape and available source pyramid levels;
- backend filtering/storage capabilities;
- current and adjacent resident levels;
- memory pressure and interaction state.

The output is a demand, not a promise that the factor is available:

```python
@dataclass(frozen=True)
class LodDemand:
    desired_level: int
    desired_factor_xy: tuple[int, int]
    acceptable_levels: tuple[int, ...]
    source_texels_per_pixel_xy: tuple[float, float]
    reason: str
```

Power-of-two levels and the current 1–2 source-texels-per-screen-pixel target are reasonable starting
points. Promotion and demotion use asymmetric hysteresis. Selection is allowed to report a desired
factor greater than the applied factor.

### 2. LOD materialization

A `LodMaterializer` creates or reads reduced source payloads outside GUI commit callbacks. Its cache
key contains semantic source identity, tile/region identity, component/scale representation required
for reduction, level, and reduction algorithm version.

Materialization rules:

- workers consume immutable source snapshots;
- duplicate requests are singleflight;
- work is cancellable/supersedable but reusable completed levels enter a bounded cache;
- source-provided pyramids are preferred over rebuilding them;
- adjacent levels may be retained to avoid threshold thrash;
- semantic values, exact histogram summaries, hover, ROI, and profiles continue to use exact or
  explicitly qualified sources rather than silently reading approximate display textures.

### 3. LOD-compatible residency

Physical storage identity includes LOD level, texture dimensions, gutter, format, and backend context.
Different dimensions are never placed into a fixed slot class that assumes one tile shape.

Compatible implementations include:

- separate atlas pages/pools per `(level, tile shape, format, gutter)`;
- texture arrays grouped by identical dimensions and format;
- a virtual-texture/page-table design;
- a backend-native mipmap path only when edge handling, complex/component mapping, memory accounting,
  and update behavior are proven.

The residency manager keeps the currently presented level usable until the requested replacement is
resident. It may retain one adjacent level when budget allows. A transition must not clear a valid
native tile merely because a coarser request is pending.

### Backend strategy

VisPy is the primary beneficiary because reduced source textures can reduce sampling and residency
cost. PyQtGraph may use reduced CPU images only when measured scene/update savings exceed reduction
and replacement cost. The semantic planner is shared; physical allocation and admission remain
backend-specific.

### Current production policy and diagnostics

Until the above exists:

- applied LOD factor is exactly `1`;
- hardware/backend filtering handles zoomed-out native textures;
- diagnostics report desired factor, applied factor, policy (`native-only`), and the reason that
  asynchronous compatible residency is required;
- no UI or benchmark may imply that a computed desired factor was actually presented.

## Acceptance gates for enabling non-native LOD

All of the following are required:

1. no pyramid construction or bulk reduction in a Qt/OpenGL commit callback;
2. deterministic cache/singleflight tests and bounded memory accounting;
3. separate compatible storage classes with no mixed-slot padding assumption;
4. hysteresis tests across repeated zoom threshold crossings;
5. retained-frame behavior while a new level materializes/resides;
6. no source-pixel upload when only the selected already-resident level changes;
7. exact hover/profile/ROI/histogram semantics or an explicit approximation label;
8. request-to-frame, event-loop-gap, upload-byte, residency, and transition traces on real hardware;
9. evidence that enabling LOD improves a reference workload rather than merely reducing a counter.

## Consequences

Positive:

- the disabled state is explicit rather than appearing broken;
- selection policy can evolve without running expensive work;
- source pyramids and GPU residency can be reused across viewport changes;
- transitions avoid rebuild storms and atlas shape corruption;
- exact semantic inspection remains independent of display quality.

Costs:

- native textures may consume more memory and bandwidth until the full path exists;
- a real multi-resolution cache/residency design is more work than reviving the prototype;
- anisotropic and backend-specific decisions need additional diagnostics/tests.

## Rejected alternatives

- **Re-enable the old synchronous pyramid code.** It predictably restores GUI stalls and repeated
  rebuild/upload transitions.
- **Mix arbitrary LOD sizes in the existing fixed-shape atlas.** Padding does not repair incompatible
  UV/storage assumptions and wastes memory.
- **Put levels in ordinary presentation identity only.** LOD changes physical source content and
  storage class; it is not a window/level uniform.
- **Use approximate textures for all semantic inspection.** Display quality must not silently change
  scientific values.
- **Claim factor one means the selector did not run.** Desired and applied quality are separate states
  and must both be visible.
