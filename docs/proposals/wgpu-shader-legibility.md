# wgpu shader legibility and filtering

## Status

Proposed (2026-07-21), authored on `claude/wgpu-shader-enhancements-f3a22d`.
Recovered into the repo 2026-07-25 during branch cleanup; it had only ever
existed as an untracked file in that worktree.

**Stage A is implemented on main** — `f18cdfdb` (A1, zoom-gated pixel grid),
`7cb49e44` (A2–A4, NaN / missing-page / clip trust signals), `631db92a`
(settings + Performance menu wiring). Both `pixel_grid` and `clip_indicator`
ship **default off**; see the note in Stage A on why.

**Stages B, C, and D are unbuilt** and remain the live part of this proposal.

**Current renderer note (2026-07-27):** VisPy was retired by
[ADR 0061](../decisions/0061-retire-vispy-rendering-backend.md). Its behavior
below is historical comparison evidence, not a source tree or fallback to
extend.

The original ADR 0058 reservation is void: a parallel branch claimed 0058 for
[canonical tile orientation](../decisions/0058-canonical-tile-orientation-and-display-transpose.md)
— exactly the collision the reservation anticipated. Stage C is the only
remaining stage that could change the renderer command protocol; it should take
the next free number (**0059**) if it gets there.

This proposal covers four separable stages. Stages A and B are independent of
each other and of C. Stage D depends on B's text helper. Each stage is a
landable commit with its own ring-4 gate; the program can stop after any stage
without leaving a half-built seam.

## Problem

The wgpu tile shader renders data faithfully but tells the user nothing about
the *frame* they are looking at. Concretely, five gaps:

1. **No pixel structure at high zoom.** A texel magnified to 40 screen pixels
   is an anonymous flat square. There is no way to see where one sample ends
   and the next begins — which is exactly the question at that zoom level.
   `docs/ideas.md:107` already sanctions "Optional pixel grid and crosshair at
   high zoom."
2. **No montage orientation when zoomed out.** With small tiles and gaps, a
   montage of mostly-empty slices is a field of dark rectangles with no cue
   for which row/column a tile sits in, or which slice index it is. The grid
   structure exists (`MontageTile.row/col`, `source_index`) and is discarded
   before it reaches the GPU.
3. **Minification aliasing.** The fragment shader point-samples exactly one
   texel per fragment (`textureLoad`, `wgpu_executor.py:295`). At the LOD
   policy's own steady-state target of 1–2 texels per pixel
   (`display/lod.py:85-86`), and worse during the transient before a finer
   level lands, that discards most of the footprint the pyramid already paid
   to load. The visible symptom is shimmer under pan.
4. **Three silent-lie cases in the fragment shader.** NaN/Inf, missing pages,
   and out-of-window values are all rendered as ordinary data. Detail below;
   this is the part with the strongest mission claim.
5. **No LOD cross-fade.** When refinement lands a finer level, the image
   snaps. Progressive refinement is the product's core behavior and its most
   visible moment is an unexplained pop.

## What exists today

Established by reading, not assumed. Line numbers are as of 2026-07-21 and
have since drifted; the claims below were re-verified against main on
2026-07-25, with the four marked *(updated)* corrected.

- **No sampler is used on the tile path.** *(updated — was "zero hits".)*
  `create_sampler` now has exactly one hit,
  `gpu/wgpu_executor.py:1661`, and it belongs to the texture-codec path
  (`_codec_sampler`), not tile rendering. Every tile texture read is still
  `textureLoad` at integer coordinates, mip 0, so Stage C's premise holds.
- **No mipmaps.** Every `create_texture` omits `mip_level_count`
  (`wgpu_executor.py:925, 971, 982, 988, 998`). The retired VisPy renderer
  had them, but WGPU never grew a counterpart, so
  `tile_layer_mipmap_available` is permanently `False` for it
  (`display/imageview2d.py:465-466`).
- **Resolution adaptation is CPU-side.** `display/pyramid.py` is a NumPy box
  reduction pyramid with semantic reducer families (`REDUCER_MEAN_ABS`,
  `REDUCER_RMS`, `REDUCER_PHASE_VECTOR`, `pyramid.py:20-27`). The shader's
  `lod_info`/`PlaneInfo.lod_base` span is page-table addressing, not GPU mips.
- **The shader has no zoom awareness.** *(updated — Stage A resolved this.)*
  `Mapping` carried no scale; `fwidth(in.src)` on the non-flat interpolated
  `VOut.src` varying supplies per-fragment texels-per-pixel with no new
  uniform, and that is what shipped. **The two spare padding words this
  proposal earmarked are now spent**: `_pad2`/`_pad3` became
  `pixel_grid: u32` and `clip_indicator: u32`. Any further flag needs a new
  word or a bitfield, in *both* copies of the `Mapping` struct — the WGSL is
  duplicated across the two shader variants in `gpu/wgpu_executor.py`.
- **`lod_req` is a `resolve()` parameter, always passed 0.** *(updated.)* It no
  longer appears in `display/wgpu_imageview2d.py`; it survives only as the
  third argument of `fn resolve(...)` in the shaders. The loop still walks up
  to whatever coarser page is resident, so level is decided by residency, not
  by the shader.
- **Overlays have no capacity ceiling.** `_set_overlay_geometry`
  (`wgpu_executor.py:1159-1195`) doubles `_overlay_cap` (initially 256) and
  rebuilds bind groups. Overflow costs a realloc, never dropped geometry. The
  512 cap is on *tiles*, not overlays.
- **Text infrastructure exists but has no reusable entry point.**
  `GlyphAtlas.layout_text` (`display/glyph_atlas.py:128`) returns placements;
  turning them into `glyph_quad` primitives is hand-rolled at exactly one call
  site, `_wgpu_tile_truth_primitives` (`wgpu_imageview2d.py:1025-1129`).
- **Overlay world space already matches montage display-pixel space.** Tile
  world rects come straight from `tile_layout_regions` in integer display
  pixels (`wgpu_imageview2d.py:1180, 1315-1323`), and `SetOverlayCamera` uses
  the same `viewRange()` as tile dst normalization. Per-tile rects can be
  emitted as `world_rect` primitives with no transform.
- **`MontageTileOverlay.text` is carried but rendered by no backend**
  (`display/overlays.py:14-21`; `wgpu_imageview2d.py:913-914` says "glyph
  rendering is outside this MVP").

### Mipmaps: the answer is no

WGPU should **not** grow generic hardware mipmaps. GPU mipmap generation averages
texels in the texture's own format. The pyramid's reducer families exist
because a coarse level of complex data is not the arithmetic mean of its
components — `REDUCER_PHASE_VECTOR` and `REDUCER_RMS` are the semantically
correct answers, and hardware mip generation cannot express either. Adding
mipmaps would give a *second, wrong* reduction ladder next to the correct one,
violating ground rule 2 (one owner per decision). The correct wgpu answer is
to keep the CPU pyramid as the only ladder and improve how the shader
*samples within* a level (Stage C).

## Design principles for this program

1. **Default render stays byte-identical.** Every new visual is either
   zoom-gated well outside the range existing oracles exercise, or behind a
   setting that is off by default. This keeps
   `tests/gpu/test_wgpu_command_protocol.py` and the framebuffer oracle
   meaningful instead of needing a wholesale rebaseline.
2. **Nearest stays the magnification default.** Bilinear magnification invents
   values that are not in the array. For an inspection tool whose mission is
   "quickest *trustworthy*", smoothing on magnification is a lie, and it would
   directly defeat Stage A's pixel grid. Offer it; never default to it.
3. **Every shader branch gets a CPU mirror.** `display/shader_mapping.py` is
   the pure-NumPy shader mirror the framebuffer oracle compares against, and
   `Scene.reference` (`tests/gpu/test_wgpu_command_protocol.py:468`) is the
   wgpu executor's mirror. A shader change that skips either is untested by
   construction.
4. **No new scheduler, timer, or tile-state collection** (ground rules 11–12).
   Everything here is either fragment-local or folds into the existing
   `_wgpu_overlay_primitives()` rebuild.
5. **Backend-neutral protocol.** ADR 0057 forbids naming WGSL in
   `gpu/command_protocol.py:110-111`. New per-tile facts travel as neutral
   fields; shader detail stays in `gpu/wgpu_executor.py`.

## Stage A — in-shader pixel grid and trust signals — **IMPLEMENTED**

Landed on main as `f18cdfdb` (A1), `7cb49e44` (A2–A4), `631db92a` (settings +
menu). Kept here as the design record; the text below is as proposed, with the
one significant deviation noted at the end of the section.

Entirely inside `_RENDER_WGSL.fs_main`. No new uniforms for the grid; one new
`Mapping` word for the feature flags (two spare padding words exist:
`_pad2`, `_pad3` at `wgpu_executor.py:194-195`).

**A1 — zoom-gated pixel grid.** Compute `let fw = fwidth(in.src);` at the top
of `fs_main`, before any non-uniform control flow (WGSL derivatives require
uniform control flow; `resolve()` returns early, so the derivative must be
taken first). Pixels-per-texel is `1.0 / max(fw.x, fw.y)`. Texel-local
distance to the nearest edge in screen pixels is
`min(f, 1.0 - f) / fw` where `f = fract(in.src)`. Blend a subtle darkening
with `1.0 - smoothstep(0.0, line_px, d)`, multiplied by a global fade that is
0 below ~12 px/texel and 1 above ~24. Anti-aliased and resolution-independent
by construction; costs no bandwidth and no instances.

**A2 — NaN/Inf made visible.** Today a NaN flows through
`clamp((x - lo) / (hi - lo))` into an arbitrary LUT entry and is
indistinguishable from real data. Render non-finite values as a fixed
diagonal hatch keyed off `@builtin(position)` so it reads against every
colormap, rather than a flat color that some LUT will match.

**A3 — missing pages distinguished from black.** `resolve()` returning
`layer < 0` currently yields `vec4(0,0,0,1)` (`wgpu_executor.py:274, 283`),
which is also what a legitimate zero renders as. "Not loaded yet" and
"actually zero" must not look identical. A very low-contrast hatch at a
different angle from A2 keeps the frame calm while making the distinction
available. This one interacts with progressive loading and is the most
valuable of the three.

**A4 — clipping indication.** Values at or beyond the level window flatten
silently, so an over-tight window looks like flat structure. Mark
`x < level_lo` and `x > level_hi` distinctly (off by default; a windowing aid,
not a default view).

*Exit gate:* ring 4 (`tests/gpu_interaction`) pixel evidence at high zoom on
real Wayland showing the grid fading in, plus a `Scene.reference` mirror
branch and a paired **fault-injection** test per new visual (testing law 5 —
an oracle must be proven able to fail). Default-off state must render
byte-identical to the pre-change target.

**Deviation as shipped: the pixel grid defaults to OFF.** Design principle 1
assumed the 12–24 px/texel gate band sat outside what the oracles exercise.
It does not: `ImageView2D` renders small images at roughly 20 px/texel, inside
the band. Defaulting the grid on would therefore have forced a display-oracle
rebaseline, which principle 1 exists to prevent. Both flags are user-opt-in
under the Performance menu, and the CPU mirror lives in
`display/shader_mapping.py`. Ring 4 evidence is still owed.

## Stage B — montage guides and slice labels

Python-side only, in `_wgpu_overlay_primitives()`
(`wgpu_imageview2d.py:910-1006`). No shader change.

**B0 — extract a reusable text helper.** `_wgpu_tile_truth_primitives` is
currently the only code that turns `layout_text` placements into
`glyph_quad`s. Stages B and D both need it. Extract
`_text_primitives(anchor_world, text, style, dpr, ...)` and reduce the tile
truth overlay to a caller. This is a ground-rule-2 move (one owner), and it
should land as its own commit with the tile-truth pixels unchanged.

**B1 — per-tile slice number.** `source_index` already survives to draw time
in `_wgpu_committed["tiles"]` (`wgpu_imageview2d.py:1329`) and is discarded by
`_wgpu_camera_tiles`. Emit one small label per tile, anchored at the tile's
world corner, reusing the tile-truth visibility rule (hide below ~16x12 px)
so labels vanish rather than crowd. Bounded by tile count.

**B2 — row/column tile guides.** One `world_rect` per tile cell at very low
alpha, alternating by `(row + col) & 1` or by row parity, drawn beneath the
data. This is what makes empty space legible: an all-zero slice still shows
its cell. Tile rects come from `tile_layout_regions` with no transform.

**B3 — render `MontageTileOverlay.text`.** The field is already populated
("Skipped"/"Loading", `window/frame_runtime.py:721`) and drawn by no backend.
With B0 in place this is nearly free, and it removes a carried-but-ignored
field.

Both B1 and B2 are zoom-gated: labels above a tile-size threshold, guides
below one, so they are never on screen together at full strength.

*Exit gate:* journey matrix on real Wayland with a montage fixture, showing
labels and guides appearing at the intended zoom bands and no regression in
the trajectory gate; overlay instance count reported in `FrameReport`.

## Stage C — honest minification filtering

**C1 is implemented** (2026-07-26), default off, under View ▸ Display Aids ▸
"Smooth When Zoomed Out". C2 and C3 remain unbuilt. Measurements and the two
deviations from the text below are in [C1 as built](#c1-as-built).

The substantive image-quality work, and the only stage with a real
performance risk. Land it last among A–C and gate it on the perf bars.

**The gutter blocker does not apply.** `wgpu_imageview2d.py:2306-2309` rejects
LOD gutters, which would block a *hardware sampler* doing bilinear across
256x256 page boundaries. Doing the taps manually and calling `resolve()`
**per tap** sidesteps this entirely: each tap independently walks the page
table, so a footprint spanning four pages resolves correctly with zero
gutters, zero samplers, and zero mipmaps. The cost is 4 page-table lookups
instead of 1.

**C1 — footprint box filter on minification only.** When
`max(fw.x, fw.y) > 1.0` (minifying), take a 2x2 tap pattern across the
fragment's footprint and average. Skip entirely when magnifying, so the
magnification path stays exactly nearest and Stage A's grid stays crisp.
Complex planes must average the *components* before the mode reduction, not
the magnitudes after it — otherwise the filter contradicts the pyramid's own
`REDUCER_PHASE_VECTOR` semantics.

**C2 — optional LOD cross-fade.** `resolve()` can already reach level *n* and
*n+1*. Blending them by the fractional part of the ideal level removes the
snap when refinement lands. Flagged honestly as a *display* smoothing of
already-approximate preview data, and therefore opt-in: a blended value is
neither level's value, which is in tension with principle "semantically
stable". Worth building, worth defaulting off, worth Thomas's call.

**C3 — opt-in bilinear magnification.** For completeness and for users who
want it. Default off, per design principle 2.

*Exit gate:* `profile_montage_workflow` on real Wayland showing pan shimmer
gone on a fixture that exhibits it, **and** benchmark deltas within ±10% of
the frozen baseline (`docs/queue.md` performance bars). A measured regression
means C1 ships behind a setting or not at all. Plus `Scene.reference` mirror
and fault-injection tests.

### C1 as built

`DisplayMapping.minification_filter` (backend-neutral, ADR 0057) → the
`AID_MINIFY_FILTER` bit of `Mapping.aids` in both render shaders. **Default
off, so the default render is byte-identical and no display-oracle rebaseline
is needed** — the pre-existing `Scene.reference` oracles all render minified
tiles and still pass unchanged.

Two deviations from the text above, both measured rather than argued:

**The cap is 2x2, not adaptive up to 3x3.** On the zoomed-out 272-tile montage
(1400x948 window, 5.9 source texels per screen pixel — the case the
[preview-LOD dossier](../redesign/preview-lod-anatomy-2026-07-26.md) §4 opens
with), against a point-sampled baseline:

| taps/axis | frame-time delta | aliasing energy | of the 2x2 gain |
|---|---:|---:|---:|
| 2 (shipped) | **+1.7 ms** | 9.36 → 7.47 (−20.2%) | 100% |
| 3 | +3.3 ms | 9.36 → 7.21 (−23.0%) | 114% |

The third ring of taps doubles the cost for one seventh more benefit; its
samples are too correlated with the ones already taken. Aliasing energy is the
mean absolute neighbour difference over the screenshot, which is what
shimmer is; 22.5% of the frame's pixels change, run-to-run noise is 0.0%.

**The cost is not free, and the "no extra bandwidth" argument was only half
right.** Residency and upload traffic really are unchanged — but nine (now
four) texel reads per fragment are not nine-for-the-price-of-one. A full-window
draw at the shipped cap goes **1.63 → 3.28 ms** (+1.7 ms, +105%) in a tight
executor loop. That is why it ships default off. On the montage *stage* it is
invisible (off 4194/4611 ms vs on 3748/3855 ms across an order-balanced pair —
the arms overlap, and the stage is scheduling-bound exactly as the dossier
says), but it is a fifth of a 60 fps budget on the interactive pan path.

Implementation notes worth keeping:

- **Resolve-per-tap, with a same-page shortcut.** Doing the taps manually and
  calling `resolve()` per tap sidesteps the gutter blocker as designed. Naively
  that is `taps²` page-table walks; since the footprint is a few texels wide,
  a tap landing in the centre tap's own 256² page is that page's texel *by
  construction*. Taking that shortcut (only when the centre resolved at the
  requested level, so it is bit-exact) is worth 29% of the filter's cost.
- **The trust rules had to be decided explicitly.** Residency stays owned by
  the centre tap, so A3's missing-page hatch keeps one owner and a tap on a
  non-resident page is dropped rather than counted as zero. A tap set
  containing *any* non-finite value stays non-finite — averaging a NaN into
  three good neighbours would launder it into a plausible number, which is what
  A2 exists to prevent. The filter therefore makes a lone bad texel *more*
  visible: at 4 texels/px a point sample can miss it entirely.
- **The tile-source-rect clamp is a guard, not a hot path.** Taps are clamped
  to the tile's own source window so one can never average in a neighbouring
  montage cell. Under the protocol's affine tile contract the clamp is provably
  inert (the extreme tap sits `fw/(2n)` inside the edge), and removing it turns
  no shader oracle red — so the rule is locked at the mirror level in
  `tests/display/test_shader_mapping.py` instead. Kept because a wrong colour
  at a tile border would be worse than the aliasing this removes.
- **Dropping a non-resident tap does re-colour a sliver during a partial
  fill, and that is accepted.** A fragment whose tap crosses into a page that
  is absent *at every level* averages fewer taps than it eventually will, so
  its colour changes when the neighbour lands. The band is bounded: a tap sits
  `fw/4` from the fragment centre, so only centres within `fw/4` of a page
  boundary can cross it — **half a screen pixel per 256-texel page boundary,
  independent of zoom**, and only while the neighbour is drawing the A3
  missing-page hatch, i.e. immediately beside a region that is about to change
  far more visibly anyway. Both alternatives are worse: counting a missing tap
  as zero invents data and darkens the band, and promoting any missing tap to
  "fragment missing" would grow the hatch over data we actually have.

### Rejected: gating the filter on the coarse rung instead of a user flag

Proposed 2026-07-26 — filter only where the payload is coarser than the
demanded level, so the cost lands on the placeholder frame and the refined
draw keeps its exact single `textureLoad`, making refinement read as
blurry-then-sharp. **Refuted on three independent legs; the filter stays gated
on minification alone.**

**1. There is nothing to gate on: on wgpu the two rungs are the same draw.**
Every tile the wgpu view commits is built with `lod_level=0`
(`_wgpu_camera_tiles`), and `wgpu_uploads_by_level` on this stage reports
exactly one row — `level 0, scalar_r32f, 1088 uploads, 285 MB`, no level-2 or
level-4 row at all — because the native-plane warm (`e266260`) deliberately
uploads the exact semantic plane. So `resolve()` lands on level 0 for every
fragment of every frame, coarse rung and refined rung alike. The rungs differ
in *when* tiles arrive, not in what is drawn. The harness's own FLOOR and FINAL
screenshots of this stage are byte-identical.

**2. The pixels agree: the coarse phase is not blurrier, it is if anything
rougher.** Aliasing energy through the fill, 0.25 s timelapse:

| t (s) | 2.8 | 3.5 | 4.0 | settled |
|---|---:|---:|---:|---:|
| filter off | 7.98 | 9.39 | 9.43 | 9.28 |
| filter on | 6.46 | 7.51 | 7.54 | 7.39 |

There is no blurry-then-sharp progression to preserve — the "preview" is
already a native-texel point sample. Composing the two measured arms, a
coarse-gated filter would run 7.4 during the fill and then **9.28 at rest**:
the upgrade would visibly get *rougher*, which re-creates the field report's
complaint rather than fixing it. (Composition, not a third run — with legs 1
and 3 the gate has no other reachable output.)

**3. Where a genuinely reduced coarse payload does exist, the filter is
already inert — at today's floor.** On this dataset the coarse rung is level 4:
21×21 texels drawn at ~57 px, **0.37 texels/px, magnifying 2.7×**, which the
minification gate declines outright. Following
[ADR 0059](../decisions/0059-coarse-rung-and-shared-reduced-stage.md), state
what fixes that level: it is the `PREVIEW_FLOOR_MIN_LEVEL = 4` clamp, not
`target_edge=48` — the unclamped formula would choose level 2 here (84 texels;
42 undershoots 48). So this leg is conditional on the floor, and worth
re-checking if the floor ever moves finer. It does not flip even then: at the
unclamped level 2 the coarse draw is 84 texels over ~57 px = **1.47
texels/px**, a 2-tap draw at the very bottom of the filter's range, while the
native draw it precedes sits at **5.94**. Gating on "coarse" would still point
the filter away from most of the aliasing, and at today's floor it disarms it
completely.

The cost argument for the gate inverts too. Because both rungs draw
identically, a coarse-gated filter costs the same per frame as an always-on one
*during* the fill and saves the pan-path cost only by giving up the resting
view — and on a zoomed-out montage the resting view and the pan view are the
same view at the same 5.9 texels/px.

### The app default is ON (2026-07-26)

`AppSettingsState.wgpu_minification_filter` ships `True`; the protocol
dataclass default stays `False`, because that is what the executor oracles
construct and it should keep constructing the unfiltered baseline. The menu
item is now an override — checked by default, with a tooltip that reads as
"what unchecking costs you". Authorised on the +1.7 ms measurement.

**The gate went in first, because the honest reading of "all four suites stay
green" was "not blocked", not "covered".** Nothing exercised a minified wgpu
draw through app settings — the display oracles all magnify, where this filter
is inert by design. `test_app_settings_default_renders_a_minified_montage_filtered`
closes that: it builds a view the way the app does (`settings_from_mapping` →
`create_image_view`), commits a per-texel checkerboard montage under a (0, 1)
window, and draws it at 1.82 source texels per screen pixel. A point sample of
binary data is binary — `g` is 0 or 1, so the LUT index is 0 or 255 — so every
intermediate grey is proof the footprint was averaged, and nothing else in the
frame can produce one. Filtered: 99.4% midtones. Point-sampled: zero.

It fails on each link of the chain independently — dataclass default,
`settings_from_mapping` fallback, factory forwarding, and the view's mapping
rebuild.

Two things that cost time and are worth not rediscovering:

- **The default has two owners.** `AppSettingsState`'s field default and the
  literal `settings_from_mapping` falls back to are separate (the convention
  every sibling setting follows). The first draft of the oracle asserted only
  the second, and flipping the *dataclass* default back to `False` left it
  green. The test now asserts both.
- **Offscreen, `viewRange()` is not a way to ask for a zoom.** The executor's
  read-back target is a fixed 768², independent of widget size, and
  pyqtgraph's fit with no real window geometry put the montage somewhere that
  made `fwidth` read as magnification. `_rerender_internal` inherits that.
  The oracle pins an explicit `SetOverlayCamera` instead and submits through
  the view, so the mapping still comes from settings and only the camera is
  the test's. This is why no display oracle had covered a minified wgpu draw:
  the harness does not make one easy to ask for.

## Stage D — per-pixel value labels

Feasible but expensive; explicitly last.

Values are available CPU-side via
`display/model/frame.py:170 sample_presented_value_at_native`, so no GPU
readback is required. The cost is instance count: a 1080p viewport at 40
px/texel is roughly 1300 labels; at ~5 glyphs each that is ~6500 overlay
instances rebuilt whenever the camera moves. The overlay buffer will grow to
fit (no cap), but the *rebuild* is on the GUI thread and ground rule 3 caps a
synchronous GUI step at 50 ms.

Therefore: hard gate at a high px/texel threshold (only when a label
comfortably fits inside a texel), a hard instance budget that refuses rather
than degrades (ground rule 5 — loud, via a `FrameReport` counter), and
formatting that respects the value's dtype and magnitude. Semi-transparent,
corner-anchored, as the original request describes.

*Exit gate:* ring 4 at extreme zoom plus a GUI-callback measurement proving
the rebuild stays under the 50 ms bar with the budget saturated; refusal path
covered by a test.

## Rejected alternatives

- **GPU mipmaps for the wgpu path.** Rejected: cannot express the pyramid's
  semantic reducers for complex/phase data, and would create a second
  reduction ladder beside the correct one. See above.
- **A hardware sampler with page gutters.** Rejected for now: gutters are
  already refused by the wgpu presenter, they inflate every page upload, and
  resolve-per-tap achieves correct cross-page filtering without them. Revisit
  only if tap cost measures badly.
- **Grid lines as overlay `line` primitives.** Rejected: one instance per
  texel edge is unbounded in the zoom range where the grid is wanted, whereas
  the in-shader derivative approach is O(1) and anti-aliases for free.
- **A new uniform carrying zoom to the tile shader.** Unnecessary:
  `fwidth(in.src)` is already available and is *per-fragment*, which a uniform
  is not.
- **Bilinear magnification as the default.** Rejected on mission grounds; see
  design principle 2.

## Risks

- **Oracle churn.** Any visible default change forces a rebaseline of the
  framebuffer and `Scene.reference` mirrors. Mitigated by principle 1.
- **WGPU physical-reference path — closed.** `framebuffer_reference.py` reads
  the WGPU executor target as well as PyQtGraph raster output. Stage B–D must
  extend that path and its fault-shaped executor oracles rather than create a
  new renderer-specific reference.
- **Derivative correctness at tile seams.** `fwidth` across a tile-quad edge
  can spike on the boundary fragment. Needs visual checking at seams, not just
  in tile interiors.
- **Stage C tap cost** on the fast-scroll path, which remains WGPU field
  evidence in `docs/queue.md` row 3. Stage C must not
  disturb that measurement — run it after, or behind a default-off setting.

## Related

- ADR 0055 (view tiles / data chunks / residency pages), ADR 0056 (sparse
  virtual multiresolution pyramid), ADR 0057 (renderer command protocol)
- `docs/proposals/wgpu-renderer-experiment.md`,
  `docs/proposals/lod-multires-implementation-plan.md`
- `docs/ideas.md:138` (pixel grid and crosshair at high zoom),
  `docs/ideas.md:160` (surfacing the LOD/interpolation actually used) — both
  moved from :107 and :129 since this was written
