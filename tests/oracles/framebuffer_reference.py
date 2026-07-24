"""General physical-pixels-to-CPU reference oracle.

Closes the "visibly wrong but every label truthful" gap named in
docs/testing/stress-and-trace-strategy.md (addendum law 2: *intent is not
pixels*): the tile-truth overlay and the trace report upload-intent only, so
a frame can present stale or swapped physical texels while every CPU-side
label stays correct. This oracle reads the REAL VisPy canvas framebuffer or
the PyQtGraph Qt-raster viewport and compares it, pixel by sampled pixel,
against a CPU-computed reference of the
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

Ring placement: the oracle itself is ring-agnostic. ``tests/gpu_interaction``
runs the VisPy path on real GL and the PyQtGraph path on a real Qt display
(ring 4, the only acceptance evidence); default-ring smokes keep both
readback/mapping paths honest offscreen without constituting acceptance.
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

# Qt's raster conversion and the NumPy shader mirror differ only at integer
# rounding boundaries. Keep this independently calibrated from the GL path:
# widening the GPU tolerance must not silently weaken PyQtGraph's CPU-LUT
# gate.
QT_RASTER_TOLERANCE = 2
QT_RASTER_MAX_MISMATCH_FRACTION = 0.0


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
            plans = tuple(getattr(payload.page_backing, "requested_plans", ()) or ())
            if plans and all(plan.reducer == REDUCER_PHASE_VECTOR for plan in plans):
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
        rgb = np.clip(color.astype(np.float32) * intensity[..., np.newaxis], 0.0, 255.0).astype(
            np.uint8
        )
        background = ~np.isfinite(values)
        return rgb, background
    rgba = cpu_display_rgba(values, mapping)
    return rgba[..., :3], rgba[..., 3] == 0


def qt_scalar_reference_tile_image(payload, mapping) -> tuple[np.ndarray, np.ndarray, int]:
    """Return expected PyQtGraph scalar RGB, alpha mask, and gutter.

    Page-backed Qt presentation expands resolved page samples over their
    exact native bins before QImage rasterization. Reconstruct that expansion
    through the backend-neutral ``PageBackedPresentation`` sampler rather
    than importing PyQtGraph's assembly code. Incomplete page coverage uses
    the payload's native semantic fallback, matching the declared contract.
    """

    backing = payload.page_backing
    if backing is None:
        rgb, background = cpu_reference_tile_image(payload, mapping)
        gutter = int(payload.lod.gutter) if payload.lod is not None else 0
        return rgb, background, gutter

    if backing.resolved_page_set is None:
        values = payload.semantic_data
        if values is None:
            raise AssertionError(
                f"tile {payload.tile_number}: incomplete page-backed Qt "
                "presentation has no native semantic fallback"
            )
        values = np.asarray(values)
    else:
        y0, y1, x0, x1 = backing.source_coverage_yx
        values = backing.sample_presented_values_at_native_coordinates(
            np.arange(y0, y1, dtype=np.int64),
            np.arange(x0, x1, dtype=np.int64),
        )
    rgba = cpu_display_rgba(values, mapping)
    return rgba[..., :3], rgba[..., 3] == 0, 0


def _required_payloads_and_plan(win, tiles=None):
    session = win.renderer._frame_session
    required = {int(number) for number in session.required_tile_numbers()}
    if not required:
        raise AssertionError(
            "physical-pixel CPU-reference oracle invoked with an empty "
            "required tile set -- a vacuous comparison is not evidence"
        )
    compared = required if tiles is None else {int(number) for number in tiles}
    if compared != required:
        raise AssertionError(
            "physical-pixel CPU-reference oracle requires an exact tile set: "
            f"required={sorted(required)}, requested={sorted(compared)}"
        )
    payloads = dict(session.display_tile_payloads)
    missing = sorted(required - {int(key) for key in payloads})
    if missing:
        raise AssertionError(f"required tiles have no committed display payload: {missing}")
    plan_tiles = {int(t.montage_index): t for t in session.plan.tiles}
    missing_plan = sorted(required - set(plan_tiles))
    if missing_plan:
        raise AssertionError(f"required tiles missing from the montage plan: {missing_plan}")
    return required, payloads, plan_tiles


def _assert_exact_tile_coverage(required, reports) -> None:
    compared = {int(report.tile_number) for report in reports}
    if compared != set(required) or len(reports) != len(required):
        raise AssertionError(
            "physical-pixel CPU-reference oracle compared a non-exact tile "
            f"set: required={sorted(required)}, compared={sorted(compared)}, "
            f"records={len(reports)}"
        )


def _qimage_rgba_array(image) -> np.ndarray:
    """Copy a QImage into a tightly packed ``(h, w, 4)`` RGBA array."""

    from pyqtgraph.Qt import QtGui

    converted = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    width = int(converted.width())
    height = int(converted.height())
    if width <= 0 or height <= 0:
        raise AssertionError(f"Qt raster readback returned an empty image: {width}x{height}")
    stride = int(converted.bytesPerLine())
    raw = np.frombuffer(
        converted.constBits(),
        dtype=np.uint8,
        count=int(converted.sizeInBytes()),
    ).reshape(height, stride)
    return raw[:, : width * 4].reshape(height, width, 4).copy()


def _view_to_viewport_affine(img_view) -> tuple[np.ndarray, np.ndarray]:
    """Return ``viewport_xy = matrix @ view_xy + offset``.

    PyQtGraph's ViewBox transform is affine for this 2-D image path. Sampling
    three points through public Qt/PyQtGraph mappings avoids sharing any tile
    item transform or cached-pixmap assumption with the renderer.
    """

    from pyqtgraph.Qt import QtCore

    view_box = img_view.getView()
    viewport_transform = img_view.graphicsView.viewportTransform()

    def mapped(x: float, y: float) -> np.ndarray:
        scene = view_box.mapViewToScene(QtCore.QPointF(float(x), float(y)))
        point = viewport_transform.map(scene)
        return np.asarray((float(point.x()), float(point.y())), dtype=np.float64)

    origin = mapped(0.0, 0.0)
    matrix = np.column_stack((mapped(1.0, 0.0) - origin, mapped(0.0, 1.0) - origin))
    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise AssertionError(f"degenerate PyQtGraph view transform: {matrix!r}")
    return matrix, origin


def qt_raster_matches_cpu_reference(
    win,
    *,
    tiles=None,
    tolerance: int = QT_RASTER_TOLERANCE,
    max_mismatch_fraction: float = QT_RASTER_MAX_MISMATCH_FRACTION,
    min_samples_per_tile: int = DEFAULT_MIN_SAMPLES_PER_TILE,
    texel_guard: float = DEFAULT_TEXEL_GUARD,
    edge_inset_px: float = DEFAULT_EDGE_INSET_PX,
) -> FrameReferenceReport:
    """Compare the live PyQtGraph Qt-raster viewport with the CPU mirror.

    The readback is the graphics viewport's painted pixels. Expected values
    come only from committed payloads and semantic levels/LUT owners; the
    backend's ``ImageItem.image`` and cached ``ImageItem.qimage`` are never
    consulted, so a stale pixmap remains observable as a divergence.

    This first PyQtGraph gate deliberately covers scalar payloads, where Qt
    owns levels/LUT rasterization. Unsupported RGB/complex modes fail loudly
    instead of passing outside the oracle's calibrated regime.
    """

    img_view = win.img_view
    if getattr(img_view, "_vispy_canvas", None) is not None:
        raise AssertionError(
            "Qt-raster CPU-reference oracle needs the PyQtGraph backend (found a VisPy canvas)"
        )
    layer = getattr(img_view, "_montage_tile_layer", None)
    if layer is None:
        raise AssertionError(
            "Qt-raster CPU-reference oracle needs the PyQtGraph montage tile layer"
        )
    required, payloads, plan_tiles = _required_payloads_and_plan(win, tiles)
    viewport = img_view.graphicsView.viewport()
    pixmap = viewport.grab()
    if pixmap.isNull():
        raise AssertionError("PyQtGraph viewport grab returned a null pixmap")
    frame = _qimage_rgba_array(pixmap.toImage())
    frame_rgb = frame[..., :3].astype(np.int16)
    background_color = img_view.graphicsView.backgroundBrush().color()
    background = np.asarray(
        (
            int(background_color.red()),
            int(background_color.green()),
            int(background_color.blue()),
        ),
        dtype=np.int16,
    )
    viewport_width = max(1, int(viewport.width()))
    viewport_height = max(1, int(viewport.height()))
    scale_x = frame.shape[1] / viewport_width
    scale_y = frame.shape[0] / viewport_height
    view_matrix, view_offset = _view_to_viewport_affine(img_view)
    inverse_view_matrix = np.linalg.inv(view_matrix)

    reports: list[TileComparison] = []
    for tile_number in sorted(required):
        tile = plan_tiles[tile_number]
        payload = payloads[tile_number]
        if payload_display_kind(payload) != "scalar":
            raise NotImplementedError(
                "Qt-raster CPU-reference oracle currently supports scalar "
                f"payloads only (tile {tile_number}: {payload.texture_kind})"
            )
        mapping = resolve_reference_mapping(win, payload)
        expected_rgb, background_mask, gutter = qt_scalar_reference_tile_image(payload, mapping)
        expected_rgb = expected_rgb.astype(np.int16)
        tex_h, tex_w = expected_rgb.shape[:2]
        inner_w = max(1e-9, float(tex_w - 2 * gutter))
        inner_h = max(1e-9, float(tex_h - 2 * gutter))

        corners_world = np.asarray(
            [
                [float(tile.x0), float(tile.y0)],
                [float(tile.x0 + tile.width), float(tile.y0)],
                [float(tile.x0), float(tile.y0 + tile.height)],
                [float(tile.x0 + tile.width), float(tile.y0 + tile.height)],
            ],
            dtype=np.float64,
        )
        corners_viewport = corners_world @ view_matrix.T + view_offset
        xs_fb = np.sort(corners_viewport[:, 0] * scale_x)
        ys_fb = np.sort(corners_viewport[:, 1] * scale_y)
        x_first = max(0, int(np.ceil(xs_fb[0] + edge_inset_px - 0.5)))
        x_last = min(
            frame.shape[1] - 1,
            int(np.floor(xs_fb[-1] - edge_inset_px - 0.5)),
        )
        y_first = max(0, int(np.ceil(ys_fb[0] + edge_inset_px - 0.5)))
        y_last = min(
            frame.shape[0] - 1,
            int(np.floor(ys_fb[-1] - edge_inset_px - 0.5)),
        )
        if x_last < x_first or y_last < y_first:
            reports.append(
                TileComparison(
                    tile_number=tile_number,
                    samples=0,
                    mismatched=0,
                    worst_diff=0,
                    mean_diff=0.0,
                    detail="tile rect is off-viewport or degenerate",
                )
            )
            continue

        px = np.arange(x_first, x_last + 1, dtype=np.int64)
        py = np.arange(y_first, y_last + 1, dtype=np.int64)
        grid_x, grid_y = np.meshgrid(px, py)
        grid_x = grid_x.ravel()
        grid_y = grid_y.ravel()
        centers_viewport = np.column_stack(
            (
                (grid_x + 0.5) / scale_x,
                (grid_y + 0.5) / scale_y,
            )
        )
        world = (centers_viewport - view_offset) @ inverse_view_matrix.T
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
            detail = (
                f"worst at viewport ({int(grid_x[select][worst_index])}, "
                f"{int(grid_y[select][worst_index])}) texel "
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

    _assert_exact_tile_coverage(required, reports)
    return FrameReferenceReport(
        tiles=tuple(reports),
        frame_shape=(int(frame.shape[0]), int(frame.shape[1])),
        tolerance=int(tolerance),
        max_mismatch_fraction=float(max_mismatch_fraction),
        min_samples_per_tile=int(min_samples_per_tile),
    )


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
    required, payloads, plan_tiles = _required_payloads_and_plan(win, tiles)

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

    _assert_exact_tile_coverage(required, reports)
    return FrameReferenceReport(
        tiles=tuple(reports),
        frame_shape=(int(frame.shape[0]), int(frame.shape[1])),
        tolerance=int(tolerance),
        max_mismatch_fraction=float(max_mismatch_fraction),
        min_samples_per_tile=int(min_samples_per_tile),
    )


def wgpu_frame_matches_cpu_reference(
    win,
    *,
    tiles=None,
    tolerance: int = DEFAULT_TOLERANCE,
    max_mismatch_fraction: float = DEFAULT_MAX_MISMATCH_FRACTION,
    min_samples_per_tile: int = DEFAULT_MIN_SAMPLES_PER_TILE,
    texel_guard: float = DEFAULT_TEXEL_GUARD,
    edge_inset_px: float = DEFAULT_EDGE_INSET_PX,
) -> FrameReferenceReport:
    """Compare WGPU's physical render target with current semantic payloads.

    This deliberately reads the executor target, not ``tileTruthPhysicalRows``:
    a page-table key can be current while its texels still belong to an older
    crop window.  The expected image is derived from the frame session's
    payloads and montage plan; only the public camera state is shared with the
    renderer.
    """

    from arrayscope.gpu.command_protocol import SetDisplayMapping, UpdateTileInstances

    img_view = win.img_view
    executor = getattr(img_view, "_wgpu_executor", None)
    if executor is None:
        raise AssertionError("WGPU CPU-reference oracle needs a live WGPU executor")
    required, payloads, plan_tiles = _required_payloads_and_plan(win, tiles)
    camera = img_view._wgpu_camera_command()
    if camera is None:
        raise AssertionError("WGPU CPU-reference oracle needs a non-degenerate camera")

    # Render the current committed instances into the readback target.  Normal
    # screen/bitmap presentation may target a swapchain texture instead; this
    # zero-upload submit makes the oracle independent of present method.
    img_view._submit_wgpu(
        (
            SetDisplayMapping(img_view._wgpu_mapping_state),
            camera,
            UpdateTileInstances(img_view._wgpu_tile_instances()),
        )
    )
    frame = executor.read_target()
    if frame.ndim != 3 or frame.shape[-1] != 4:
        raise AssertionError(f"unexpected WGPU target shape: {frame.shape}")
    frame_rgb = frame[..., :3].astype(np.int16)
    frame_h, frame_w = frame.shape[:2]
    x0, y0, x1, y1 = (float(value) for value in camera.world_rect)
    span_x = x1 - x0
    span_y = y1 - y0
    if not (span_x > 0.0 and span_y > 0.0):
        raise AssertionError(f"degenerate WGPU camera: {camera!r}")

    transposed = bool(getattr(img_view, "_wgpu_committed", {}).get("transposed", False))
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

        tile_x0 = float(tile.x0)
        tile_y0 = float(tile.y0)
        tile_x1 = tile_x0 + float(tile.width)
        tile_y1 = tile_y0 + float(tile.height)

        def world_to_frame_x(value: float) -> float:
            fraction = (value - x0) / span_x
            if camera.x_inverted:
                fraction = 1.0 - fraction
            return fraction * frame_w

        def world_to_frame_y(value: float) -> float:
            fraction = (value - y0) / span_y
            if not camera.y_inverted:
                fraction = 1.0 - fraction
            return fraction * frame_h

        xs_fb = np.sort((world_to_frame_x(tile_x0), world_to_frame_x(tile_x1)))
        ys_fb = np.sort((world_to_frame_y(tile_y0), world_to_frame_y(tile_y1)))
        x_first = max(0, int(np.ceil(xs_fb[0] + edge_inset_px - 0.5)))
        x_last = min(frame_w - 1, int(np.floor(xs_fb[1] - edge_inset_px - 0.5)))
        y_first = max(0, int(np.ceil(ys_fb[0] + edge_inset_px - 0.5)))
        y_last = min(frame_h - 1, int(np.floor(ys_fb[1] - edge_inset_px - 0.5)))
        if x_last < x_first or y_last < y_first:
            reports.append(
                TileComparison(
                    tile_number=tile_number,
                    samples=0,
                    mismatched=0,
                    worst_diff=0,
                    mean_diff=0.0,
                    detail="tile rect is off-target or degenerate",
                )
            )
            continue

        px = np.arange(x_first, x_last + 1, dtype=np.int64)
        py = np.arange(y_first, y_last + 1, dtype=np.int64)
        grid_x, grid_y = np.meshgrid(px, py)
        grid_x = grid_x.ravel()
        grid_y = grid_y.ravel()
        frac_frame_x = (grid_x + 0.5) / frame_w
        frac_frame_y = (grid_y + 0.5) / frame_h
        world_x = x1 - frac_frame_x * span_x if camera.x_inverted else x0 + frac_frame_x * span_x
        world_y = y0 + frac_frame_y * span_y if camera.y_inverted else y1 - frac_frame_y * span_y
        frac_x = (world_x - tile_x0) / float(tile.width)
        frac_y = (world_y - tile_y0) / float(tile.height)
        inside = (frac_x > 0.0) & (frac_x < 1.0) & (frac_y > 0.0) & (frac_y < 1.0)
        if transposed:
            frac_x, frac_y = frac_y, frac_x
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
            expected[is_background] = 0
        actual = frame_rgb[grid_y[select], grid_x[select]]
        diff = np.abs(actual - expected).max(axis=-1)
        mismatched = diff > tolerance
        worst_index = int(np.argmax(diff))
        detail = ""
        if np.any(mismatched):
            detail = (
                f"worst at target ({int(grid_x[select][worst_index])}, "
                f"{int(grid_y[select][worst_index])}) texel "
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

    _assert_exact_tile_coverage(required, reports)
    return FrameReferenceReport(
        tiles=tuple(reports),
        frame_shape=(int(frame_h), int(frame_w)),
        tolerance=int(tolerance),
        max_mismatch_fraction=float(max_mismatch_fraction),
        min_samples_per_tile=int(min_samples_per_tile),
    )


def _assert_report(report: FrameReferenceReport, *, label: str) -> FrameReferenceReport:
    failures = report.failures()
    if not failures:
        return report
    lines = [
        f"{label} diverges from the CPU semantic reference "
        f"(tolerance={report.tolerance}/255, "
        f"max_mismatch_fraction={report.max_mismatch_fraction}, "
        f"min_samples={report.min_samples_per_tile}):"
    ]
    lines.extend(
        f"  tile {tile.tile_number}: samples={tile.samples} "
        f"mismatched={tile.mismatched} "
        f"({tile.mismatch_fraction:.1%}) worst={tile.worst_diff} "
        f"mean={tile.mean_diff:.2f}" + (f" [{tile.detail}]" if tile.detail else "")
        for tile in failures
    )
    healthy = len(report.tiles) - len(failures)
    lines.append(f"  ({healthy}/{len(report.tiles)} required tiles within tolerance)")
    raise AssertionError("\n".join(lines))


def assert_frame_matches_cpu_reference(win, **kwargs) -> FrameReferenceReport:
    """Asserting form of :func:`frame_matches_cpu_reference`.

    Fails when any required tile's framebuffer interior deviates from the
    CPU-computed reference beyond GPU-rounding tolerance, or when a tile
    contributes too few samples for the comparison to be evidence.
    """

    return _assert_report(
        frame_matches_cpu_reference(win, **kwargs),
        label="framebuffer",
    )


def assert_wgpu_frame_matches_cpu_reference(win, **kwargs) -> FrameReferenceReport:
    """Asserting form of :func:`wgpu_frame_matches_cpu_reference`."""

    return _assert_report(
        wgpu_frame_matches_cpu_reference(win, **kwargs),
        label="WGPU target",
    )


def assert_qt_raster_matches_cpu_reference(win, **kwargs) -> FrameReferenceReport:
    """Asserting form of :func:`qt_raster_matches_cpu_reference`."""

    return _assert_report(
        qt_raster_matches_cpu_reference(win, **kwargs),
        label="Qt raster",
    )
