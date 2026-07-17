"""General framebuffer-to-CPU reference oracle.

Closes the "visibly wrong but every label truthful" gap named in
docs/testing/stress-and-trace-strategy.md (addendum law 2: *intent is not
pixels*): the tile-truth overlay and the trace report upload-intent only, so
a frame can present stale or swapped physical texels while every CPU-side
label stays correct.  This oracle reads the REAL VisPy canvas framebuffer and
compares it, pixel by sampled pixel, against a CPU-computed reference of the
same semantic values — component/scale/levels/LUT applied through
``arrayscope.display.shader_mapping`` (the pure-NumPy shader mirror) — using
the live camera transform for geometry.  Nothing is taken from the backend's
draw bookkeeping: values come from the session's committed payloads, geometry
from the montage plan and the camera, mapping state from the payload's
semantic ``ShaderMapping`` plus the UI levels/LUT owners.

Tolerances exist only for GPU rounding (float raster arithmetic and the
half-texel difference between GL's texel-center LUT sampling and the CPU
``(N-1)``-index convention), never for content: a wrong uniform, a stale
atlas page, or a swapped tile changes whole sampled populations and fails
loudly.  Vacuity guards (ground-rules law: *a count is not coverage* /
testing law 5) are built in: the compared tile set must equal the required
set exactly, and every tile must contribute a minimum sample population —
a clamped rectangle or an off-screen tile cannot silently pass.

Ring placement: the oracle itself is ring-agnostic (it needs a live VisPy
canvas).  ``tests/gpu_interaction`` runs it on real GL (ring 4, the only
acceptance evidence); ``tests/ui/test_framebuffer_cpu_reference.py`` runs a
default-ring smoke on offscreen software GL, which is faithful for this
shader path (precedent: tests/ui/test_vispy_phase_framebuffer.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.display.shader_mapping import (
    ShaderDisplayMode,
    ShaderMapping,
    TexturePlaneKind,
    apply_phase_lut,
    cpu_display_rgba,
    normalize_lut_rgb,
)
from arrayscope.gpu.keys import REDUCER_PHASE_VECTOR


# Per-channel 8-bit tolerance for GPU-vs-CPU rounding: float raster rounding
# plus the <=0.5-texel LUT sampling-convention offset (GL samples the LUT at
# ``intensity`` over N texel centers; the CPU mirror indexes ``intensity *
# (N-1)``).  Content faults produce differences far above this.
DEFAULT_TOLERANCE = 6
# Fraction of sampled pixels allowed to exceed the tolerance: rasterization
# may pick the neighbouring texel for pixel centers that survive the texel
# guard only marginally.  A stale page or swapped tile mismatches its whole
# sample population, so a small fraction keeps the oracle strict.
DEFAULT_MAX_MISMATCH_FRACTION = 0.01
# Vacuity floor: a tile compared over fewer samples is a failed comparison,
# never a pass (testing law 5 — oracles must be proven able to fail).
DEFAULT_MIN_SAMPLES_PER_TILE = 24
# Pixel centers closer than this (in texels) to a texel boundary are skipped:
# GPU nearest-sampling and the CPU floor() may legitimately disagree there.
DEFAULT_TEXEL_GUARD = 0.25
# Framebuffer-pixel inset from each tile edge: keeps montage gap/seam pixels
# and antialiased tile borders out of the interior comparison.
DEFAULT_EDGE_INSET_PX = 2.0


@dataclass(frozen=True)
class TileComparison:
    tile_number: int
    samples: int
    mismatched: int
    worst_diff: int
    mean_diff: float
    detail: str = ""

    @property
    def mismatch_fraction(self) -> float:
        return self.mismatched / self.samples if self.samples else 1.0


@dataclass(frozen=True)
class FrameReferenceReport:
    tiles: tuple[TileComparison, ...]
    frame_shape: tuple[int, int]
    tolerance: int
    max_mismatch_fraction: float
    min_samples_per_tile: int

    @property
    def total_samples(self) -> int:
        return sum(tile.samples for tile in self.tiles)

    def failures(self) -> tuple[TileComparison, ...]:
        return tuple(
            tile
            for tile in self.tiles
            if tile.samples < self.min_samples_per_tile
            or tile.mismatch_fraction > self.max_mismatch_fraction
        )


def payload_display_kind(payload) -> str:
    """Semantic display-mode classification from payload facts alone.

    Deliberately independent of the backend's ``_payload_mode`` so a backend
    mode-classification bug is a divergence this oracle can catch, not a
    shared assumption.
    """

    kind = payload.texture_kind
    if kind == TexturePlaneKind.COMPLEX_RG32F:
        mapping = payload.shader_mapping
        display_mode = getattr(mapping, "display_mode", None)
        if display_mode == ShaderDisplayMode.PHASE_COLOR:
            plans = tuple(
                getattr(payload.page_backing, "requested_plans", ()) or ()
            )
            if plans and all(
                plan.reducer == REDUCER_PHASE_VECTOR for plan in plans
            ):
                return "phase_vector"
            return "phase_color"
        return "complex"
    if kind == TexturePlaneKind.RGB8:
        raise NotImplementedError(
            "framebuffer CPU-reference oracle does not support RGB payloads "
            f"yet (tile {payload.tile_number}); extend the oracle rather than "
            "letting an RGB scene pass unverified"
        )
    return "scalar"


def payload_semantic_values(payload) -> np.ndarray:
    """The exact per-texel values the GPU samples, as a NumPy array."""

    values = np.asarray(payload.texture_data)
    if payload.texture_kind == TexturePlaneKind.COMPLEX_RG32F and not np.iscomplexobj(values):
        if values.ndim < 3 or values.shape[-1] != 2:
            raise AssertionError(
                f"tile {payload.tile_number}: complex payload texture has "
                f"shape {values.shape}, expected trailing real/imag planes"
            )
        values = values[..., 0] + 1j * values[..., 1]
    return values


def resolve_reference_mapping(win, payload) -> ShaderMapping:
    """Payload mapping with levels/LUT resolved from their semantic owners.

    Levels: the payload mapping when it pins them, else the UI histogram
    owner (``img_view.getLevels()``) — the same truth the backend feeds
    ``set_levels``.  LUT: the payload mapping's LUT when present; for
    LUT-mapped scalar modes the frame display colormap
    (``displayColorMapLookupTable``); phase modes keep the canonical phase
    wheel default.
    """

    mapping = payload.shader_mapping or ShaderMapping()
    levels = mapping.levels
    if levels is None:
        ui_levels = win.img_view.getLevels()
        if ui_levels is not None:
            levels = (float(ui_levels[0]), float(ui_levels[1]))
    lut_data = mapping.lut_data
    phase_mode = mapping.display_mode == ShaderDisplayMode.PHASE_COLOR
    if lut_data is None and not phase_mode:
        display_lut = win.img_view.displayColorMapLookupTable()
        if display_lut is not None:
            lut_data = normalize_lut_rgb(display_lut)
    from dataclasses import replace

    return replace(mapping, levels=levels, lut_data=lut_data)


def cpu_reference_tile_image(payload, mapping) -> tuple[np.ndarray, np.ndarray]:
    """(expected RGB uint8 (th, tw, 3), background mask) for one payload."""

    values = payload_semantic_values(payload)
    kind = payload_display_kind(payload)
    if kind == "phase_vector":
        # Mode 5: reduced circular-mean pages — hue from phase, intensity is
        # the resultant magnitude already in [0, 1]; levels bypassed
        # (tiles.py fragment shader, phase_vector branch).
        color, _magnitude = apply_phase_lut(values, mapping.lut_data)
        intensity = np.clip(np.abs(values).astype(np.float32), 0.0, 1.0)
        rgb = np.clip(
            color.astype(np.float32) * intensity[..., np.newaxis], 0.0, 255.0
        ).astype(np.uint8)
        background = ~np.isfinite(values)
        return rgb, background
    rgba = cpu_display_rgba(values, mapping)
    return rgba[..., :3], rgba[..., 3] == 0


def frame_matches_cpu_reference(
    win,
    *,
    tiles=None,
    tolerance: int = DEFAULT_TOLERANCE,
    max_mismatch_fraction: float = DEFAULT_MAX_MISMATCH_FRACTION,
    min_samples_per_tile: int = DEFAULT_MIN_SAMPLES_PER_TILE,
    texel_guard: float = DEFAULT_TEXEL_GUARD,
    edge_inset_px: float = DEFAULT_EDGE_INSET_PX,
) -> FrameReferenceReport:
    """Compare the live VisPy framebuffer against the CPU reference.

    Returns a report; use :func:`assert_frame_matches_cpu_reference` for the
    asserting form.  ``tiles`` defaults to the session's
    ``required_tile_numbers()`` — the current viewport obligation, the set an
    accepted frame must physically show.
    """

    img_view = win.img_view
    canvas = getattr(img_view, "_vispy_canvas", None)
    if canvas is None:
        raise AssertionError(
            "framebuffer CPU-reference oracle needs the VisPy backend "
            "(no _vispy_canvas on this image view)"
        )
    session = win.renderer._frame_session
    required = (
        {int(number) for number in session.required_tile_numbers()}
        if tiles is None
        else {int(number) for number in tiles}
    )
    if not required:
        raise AssertionError(
            "framebuffer CPU-reference oracle invoked with an empty required "
            "tile set — a vacuous comparison is not evidence"
        )
    payloads = dict(session.display_tile_payloads)
    missing = sorted(required - {int(key) for key in payloads})
    if missing:
        raise AssertionError(
            f"required tiles have no committed display payload: {missing}"
        )
    plan_tiles = {int(t.montage_index): t for t in session.plan.tiles}
    missing_plan = sorted(required - set(plan_tiles))
    if missing_plan:
        raise AssertionError(
            f"required tiles missing from the montage plan: {missing_plan}"
        )

    frame = np.asarray(canvas.render())
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise AssertionError(f"unexpected framebuffer shape: {frame.shape}")
    frame_rgb = frame[..., :3].astype(np.int16)
    canvas_w, canvas_h = (int(value) for value in canvas.size)
    scale_x = frame.shape[1] / max(1, canvas_w)
    scale_y = frame.shape[0] / max(1, canvas_h)
    background = np.clip(
        np.round(np.asarray(canvas.bgcolor.rgb, dtype=np.float32) * 255.0),
        0,
        255,
    ).astype(np.int16)

    # Direction verified empirically against the live camera: ``.map`` takes
    # view-scene (world) coordinates to canvas coordinates, ``.imap`` back.
    transform = img_view._vispy_view.scene.node_transform(canvas.scene)

    reports: list[TileComparison] = []
    for tile_number in sorted(required):
        tile = plan_tiles[tile_number]
        payload = payloads[tile_number]
        mapping = resolve_reference_mapping(win, payload)
        expected_rgb, background_mask = cpu_reference_tile_image(payload, mapping)
        expected_rgb = expected_rgb.astype(np.int16)
        tex_h, tex_w = expected_rgb.shape[:2]
        gutter = int(payload.lod.gutter) if payload.lod is not None else 0
        inner_w = max(1e-9, float(tex_w - 2 * gutter))
        inner_h = max(1e-9, float(tex_h - 2 * gutter))

        corners_world = np.asarray(
            [
                [float(tile.x0), float(tile.y0)],
                [float(tile.x0 + tile.width), float(tile.y0 + tile.height)],
            ],
            dtype=np.float64,
        )
        corners_canvas = np.asarray(transform.map(corners_world))[:, :2]
        xs_fb = np.sort(corners_canvas[:, 0] * scale_x)
        ys_fb = np.sort(corners_canvas[:, 1] * scale_y)
        x_first = int(np.ceil(xs_fb[0] + edge_inset_px - 0.5))
        x_last = int(np.floor(xs_fb[1] - edge_inset_px - 0.5))
        y_first = int(np.ceil(ys_fb[0] + edge_inset_px - 0.5))
        y_last = int(np.floor(ys_fb[1] - edge_inset_px - 0.5))
        x_first = max(0, x_first)
        y_first = max(0, y_first)
        x_last = min(frame.shape[1] - 1, x_last)
        y_last = min(frame.shape[0] - 1, y_last)
        if x_last < x_first or y_last < y_first:
            reports.append(
                TileComparison(
                    tile_number=tile_number,
                    samples=0,
                    mismatched=0,
                    worst_diff=0,
                    mean_diff=0.0,
                    detail="tile rect is off-framebuffer or degenerate",
                )
            )
            continue

        px = np.arange(x_first, x_last + 1, dtype=np.int64)
        py = np.arange(y_first, y_last + 1, dtype=np.int64)
        grid_x, grid_y = np.meshgrid(px, py)
        grid_x = grid_x.ravel()
        grid_y = grid_y.ravel()
        centers_canvas = np.column_stack(
            (
                (grid_x + 0.5) / scale_x,
                (grid_y + 0.5) / scale_y,
            )
        )
        world = np.asarray(transform.imap(centers_canvas))[:, :2]
        frac_x = (world[:, 0] - float(tile.x0)) / float(tile.width)
        frac_y = (world[:, 1] - float(tile.y0)) / float(tile.height)
        inside = (frac_x > 0.0) & (frac_x < 1.0) & (frac_y > 0.0) & (frac_y < 1.0)
        texel_x = gutter + frac_x * inner_w
        texel_y = gutter + frac_y * inner_h
        frac_tx = texel_x - np.floor(texel_x)
        frac_ty = texel_y - np.floor(texel_y)
        guarded = (
            (frac_tx >= texel_guard)
            & (frac_tx <= 1.0 - texel_guard)
            & (frac_ty >= texel_guard)
            & (frac_ty <= 1.0 - texel_guard)
        )
        select = inside & guarded
        if not np.any(select):
            reports.append(
                TileComparison(
                    tile_number=tile_number,
                    samples=0,
                    mismatched=0,
                    worst_diff=0,
                    mean_diff=0.0,
                    detail="no pixel centers survive the interior/texel guards",
                )
            )
            continue
        index_x = np.clip(np.floor(texel_x[select]).astype(np.int64), 0, tex_w - 1)
        index_y = np.clip(np.floor(texel_y[select]).astype(np.int64), 0, tex_h - 1)
        expected = expected_rgb[index_y, index_x]
        is_background = background_mask[index_y, index_x]
        if np.any(is_background):
            expected = expected.copy()
            expected[is_background] = background
        actual = frame_rgb[grid_y[select], grid_x[select]]
        diff = np.abs(actual - expected).max(axis=-1)
        mismatched = diff > tolerance
        worst_index = int(np.argmax(diff))
        detail = ""
        if np.any(mismatched):
            fb_x = int(grid_x[select][worst_index])
            fb_y = int(grid_y[select][worst_index])
            detail = (
                f"worst at framebuffer ({fb_x}, {fb_y}) texel "
                f"({int(index_x[worst_index])}, {int(index_y[worst_index])}): "
                f"actual={tuple(int(v) for v in actual[worst_index])} "
                f"expected={tuple(int(v) for v in expected[worst_index])}"
            )
        reports.append(
            TileComparison(
                tile_number=tile_number,
                samples=int(diff.size),
                mismatched=int(np.count_nonzero(mismatched)),
                worst_diff=int(diff[worst_index]),
                mean_diff=float(diff.mean()),
                detail=detail,
            )
        )

    return FrameReferenceReport(
        tiles=tuple(reports),
        frame_shape=(int(frame.shape[0]), int(frame.shape[1])),
        tolerance=int(tolerance),
        max_mismatch_fraction=float(max_mismatch_fraction),
        min_samples_per_tile=int(min_samples_per_tile),
    )


def assert_frame_matches_cpu_reference(win, **kwargs) -> FrameReferenceReport:
    """Asserting form of :func:`frame_matches_cpu_reference`.

    Fails when any required tile's framebuffer interior deviates from the
    CPU-computed reference beyond GPU-rounding tolerance, or when a tile
    contributes too few samples for the comparison to be evidence.
    """

    report = frame_matches_cpu_reference(win, **kwargs)
    failures = report.failures()
    if failures:
        lines = [
            "framebuffer diverges from the CPU semantic reference "
            f"(tolerance={report.tolerance}/255, "
            f"max_mismatch_fraction={report.max_mismatch_fraction}, "
            f"min_samples={report.min_samples_per_tile}):"
        ]
        for tile in failures:
            lines.append(
                f"  tile {tile.tile_number}: samples={tile.samples} "
                f"mismatched={tile.mismatched} "
                f"({tile.mismatch_fraction:.1%}) worst={tile.worst_diff} "
                f"mean={tile.mean_diff:.2f}"
                + (f" [{tile.detail}]" if tile.detail else "")
            )
        healthy = len(report.tiles) - len(failures)
        lines.append(
            f"  ({healthy}/{len(report.tiles)} required tiles within "
            "tolerance)"
        )
        raise AssertionError("\n".join(lines))
    return report
