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
  (`wgpu_executor.py:925, 971, 982, 988, 998`). VisPy *does* have them
  (`display/backends/vispy/tiles.py:364`); wgpu never grew a counterpart, so
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

wgpu should **not** grow VisPy's mipmaps. GPU mipmap generation averages
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
- **`framebuffer_reference.py` has no wgpu path today** — it covers VisPy and
  PyQtGraph only. Stage A's ring-4 gate either adds one or leans on the
  executor-level `Scene` oracle plus managed-Weston bitmap evidence. Adding
  the wgpu path is the more durable choice and may deserve its own commit.
- **Derivative correctness at tile seams.** `fwidth` across a tile-quad edge
  can spike on the boundary fragment. Needs visual checking at seams, not just
  in tile interiors.
- **Stage C tap cost** on the fast-scroll path, which is the wgpu promotion
  evidence currently in flight (`docs/queue.md` row 3d). Stage C must not
  disturb that measurement — run it after, or behind a default-off setting.

## Related

- ADR 0055 (view tiles / data chunks / residency pages), ADR 0056 (sparse
  virtual multiresolution pyramid), ADR 0057 (renderer command protocol)
- `docs/proposals/wgpu-renderer-experiment.md`,
  `docs/proposals/lod-multires-implementation-plan.md`
- `docs/ideas.md:138` (pixel grid and crosshair at high zoom),
  `docs/ideas.md:160` (surfacing the LOD/interpolation actually used) — both
  moved from :107 and :129 since this was written
