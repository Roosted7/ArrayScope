"""General physical-pixels-to-CPU reference oracle for tools and tests.

Closes the "visibly wrong but every label truthful" gap named in
docs/testing/stress-and-trace-strategy.md (addendum law 2: *intent is not
pixels*): the tile-truth overlay and the trace report upload-intent only, so
a frame can present stale or swapped physical texels while every CPU-side
label stays correct. This oracle reads the REAL WGPU render target or the
PyQtGraph Qt-raster viewport and compares it, pixel by sampled pixel, against
a CPU-computed reference of the
same semantic values — component/scale/levels/LUT applied through
``arrayscope.display.shader_mapping`` (the pure-NumPy shader mirror) — using
the live camera transform for geometry.  Nothing is taken from the backend's
draw bookkeeping: values come from the session's committed payloads, geometry
from the montage plan and the camera, mapping state from the payload's
semantic ``ShaderMapping`` plus the UI levels/LUT owners.

Tolerances exist only for GPU rounding (float raster arithmetic and the
half-texel difference between GPU texel-center LUT sampling and the CPU
``(N-1)``-index convention), never for content: a wrong uniform, a stale
atlas page, or a swapped tile changes whole sampled populations and fails
loudly.  Vacuity guards (ground-rules law: *a count is not coverage* /
testing law 5) are built in: the compared tile set must equal the required
set exactly, and every tile must contribute a minimum sample population —
a clamped rectangle or an off-screen tile cannot silently pass.

Ring placement: the oracle itself is ring-agnostic. ``tests/gpu_interaction``
runs the WGPU path on a real GPU and the PyQtGraph path on a real Qt display
(ring 4, the only acceptance evidence); default-ring smokes keep both
readback/mapping paths honest offscreen without constituting acceptance.
"""

from __future__ import annotations

import itertools
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
# plus the <=0.5-texel LUT sampling-convention offset (the GPU samples the LUT at
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

# Slack (frame pixels) added around an overlay's geometric footprint before
# its pixels are withheld from the image comparison.  Covers antialiasing,
# the half-pixel between a quad edge and the pixel centers it tints, and Qt's
# cosmetic-pen rounding.  Deliberately small: the mask is built from where an
# overlay is *supposed* to be, so a stroke drawn in the wrong place stays
# inside the compared population and still fails the oracle.
OVERLAY_COVERAGE_MARGIN_PX = 2.0
# Per-channel distance at which a frame pixel counts as "this ROI's colour".
# The montage image is greyscale, so no image pixel can come within this of a
# saturated palette entry, and the palette's own entries stay far apart.
ROI_COLOR_MATCH_TOLERANCE = 40
# Minimum max-minus-min channel spread for a pixel to be a candidate ROI
# stroke.  The palette's least saturated entry spreads 110, and a match may
# shift each channel by the tolerance in opposite directions, so anything a
# ROI could have painted still clears this; greyscale image pixels spread 0.
ROI_COLOR_MIN_CHROMA = 24
# A ROI whose on-target outline is at least this long (frame pixels) must
# show its colour somewhere inside its own band; shorter slivers may be
# swallowed whole by antialiasing against a bright tile.
ROI_PLACEMENT_MIN_VISIBLE_LENGTH_PX = 12.0
# Above this share of chromatic un-masked pixels the frame's own image is
# coloured (a phase/colour LUT), a palette colour no longer identifies an
# overlay, and the stray half of the placement check stops being evidence.
# Set far above any misplacement: a whole ROI outline drawn in the wrong place
# is a fraction of a percent of a frame, while a colour LUT tints nearly all of
# it -- so the guard cannot swallow the fault it exists beside.
ROI_PLACEMENT_MAX_CHROMATIC_FRACTION = 0.25


@dataclass(frozen=True)
class TileComparison:
    tile_number: int
    samples: int
    mismatched: int
    worst_diff: int
    mean_diff: float
    detail: str = ""
    overlay_excluded: int = 0

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
    overlay_excluded_samples: int = 0
    #: Placement verdict for the same captured frame -- the presence half of
    #: the overlay contract, computed from the capture this report compared so
    #: the two verdicts can never describe different frames.
    roi_placement: RoiPlacementReport | None = None

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


def _bounded_sample_positions(
    select: np.ndarray,
    *,
    max_samples: int | None,
    sample_seed: int | None,
    tile_number: int,
) -> np.ndarray:
    """Return a deterministic bounded subset of selected flat positions."""

    positions = np.flatnonzero(np.asarray(select, dtype=bool))
    if max_samples is None or positions.size <= int(max_samples):
        return positions
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("max_samples_per_tile must be positive")
    seed = int(sample_seed or 0) & ((1 << 64) - 1)
    rng = np.random.default_rng(
        np.random.SeedSequence(
            (
                seed & 0xFFFFFFFF,
                seed >> 32,
                int(tile_number) & 0xFFFFFFFF,
            )
        )
    )
    return np.sort(rng.choice(positions, size=max_samples, replace=False))


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
    (``displayColorMapLookupTable``). Phase modes use that same UI owner when
    a display colormap is selected, and keep the canonical phase-wheel
    default only when no UI colormap owns the mapping.
    """

    mapping = payload.shader_mapping or ShaderMapping()
    levels = mapping.levels
    if levels is None:
        ui_levels = win.img_view.getLevels()
        if ui_levels is not None:
            levels = (float(ui_levels[0]), float(ui_levels[1]))
    lut_data = mapping.lut_data
    phase_mode = mapping.display_mode == ShaderDisplayMode.PHASE_COLOR
    display_colormap = getattr(win.img_view, "_display_colormap", None)
    if lut_data is None and (not phase_mode or display_colormap is not None):
        display_lut = win.img_view.displayColorMapLookupTable()
        if display_lut is not None:
            lut_data = normalize_lut_rgb(display_lut)
    from dataclasses import replace

    return replace(mapping, levels=levels, lut_data=lut_data)


def cpu_reference_tile_image(payload, mapping, *, values=None) -> tuple[np.ndarray, np.ndarray]:
    """(expected RGB uint8 (th, tw, 3), background mask) for one payload."""

    if values is None:
        values = payload_semantic_values(payload)
    else:
        values = np.asarray(values)
        if payload.texture_kind == TexturePlaneKind.COMPLEX_RG32F and not np.iscomplexobj(values):
            if values.ndim < 3 or values.shape[-1] != 2:
                raise AssertionError(
                    f"tile {payload.tile_number}: complex reference values have "
                    f"shape {values.shape}, expected trailing real/imag planes"
                )
            values = values[..., 0] + 1j * values[..., 1]
    kind = payload_display_kind(payload)
    if kind == "phase_vector":
        # Mode 5: reduced circular-mean pages — hue from phase, intensity is
        # the resultant magnitude already in [0, 1]; levels bypassed
        # (``map_value``'s ``phase_vector`` branch in gpu/wgpu_executor.py).
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


def _native_reference_values(win, session, tile, payload) -> np.ndarray:
    """Return independent CPU values for a native page-backed WGPU draw."""

    native = getattr(payload, "native_residency_data", None)
    if native is None:
        native = payload.semantic_data
    if native is not None:
        # ``native_residency_data`` describes the whole CANONICAL plane, which
        # is wider than what a cropped payload draws (the crop-warm path
        # uploads the plane and binds the window's source origin into it).  The
        # oracle compares the payload's own window, so take that window out of
        # the plane; the values themselves stay independent worker output.
        return _window_of_plane(np.asarray(native), payload)

    # A resident native source plane can outlive the exact RenderedTile that
    # populated it.  A later reduced page-backed payload may therefore draw
    # that better plane without retaining a native array in its wrapper.  For
    # a physical-truth check, recompute the immutable tile snapshot on CPU
    # rather than reading the GPU page back and using the backend as its own
    # oracle.
    from arrayscope.operations.evaluator import (
        evaluate_image_snapshot,
        stage_document_key,
    )

    evaluator = win.operation_evaluator
    result = evaluate_image_snapshot(
        session.document,
        tile.view_state,
        colormap_lut=session.colormap_lut,
        shader_display=bool(getattr(session, "shader_display", False)),
        provisional_histogram=False,
        stage_cache=getattr(evaluator, "_stage_cache", None),
        stage_document_key=stage_document_key(session.document),
        canonical_orientation=bool(getattr(session, "canonical_orientation", False)),
    )
    value = result.value
    semantic = getattr(value, "semantic_data", None)
    return np.asarray(value.data if semantic is None else semantic)


def _window_of_plane(values: np.ndarray, payload) -> np.ndarray:
    """The payload's own source window out of a whole-plane value array."""

    anchor = getattr(payload, "source_anchor", None)
    plane_shape = tuple(int(size) for size in (getattr(anchor, "plane_shape", ()) or ()))
    source_rect = tuple(int(edge) for edge in (getattr(anchor, "source_rect", ()) or ()))
    if len(plane_shape) != 2 or len(source_rect) != 4:
        return values
    if tuple(int(size) for size in values.shape[:2]) != plane_shape:
        return values
    y0, y1, x0, x1 = source_rect
    if (y0, x0) == (0, 0) and (y1, x1) == plane_shape:
        return values
    return values[y0:y1, x0:x1]


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


# --- Overlay coverage -------------------------------------------------------
# A presented frame is image pixels *plus* everything drawn over them: ROI
# outlines and handles, the profile marker, montage lifecycle boxes, tile-truth
# glyphs, and (on the WGPU screen path) the rasterized floating Qt chips.  The
# CPU reference models semantic image values only, so those pixels are not the
# oracle's to compare -- but they are also not the oracle's to ignore blindly.
#
# The mask below is built from where each overlay's own geometry says it
# belongs, projected through the same camera the tiles use.  Two properties
# follow, and both are load-bearing:
#
#   * an overlay drawn where it belongs is withheld from the image comparison,
#     so a stroke crossing a tile is no longer read as a stale texel;
#   * an overlay drawn anywhere else lands OUTSIDE the mask, stays in the
#     compared population, and still fails the oracle.
#
# ``roi_placement_report`` adds the other half -- that each ROI's colour is
# actually present inside its own band -- so a silently undrawn overlay cannot
# pass by leaving a hole that nothing looks at.


@dataclass(frozen=True)
class RoiPlacementComparison:
    roi_id: str
    color: tuple[int, int, int]
    visible_length_px: float
    band_pixels: int
    matched_in_band: int
    stray_pixels: int
    stray_extent: tuple[int, int, int, int] | None = None
    #: False when the frame's own image pixels are chromatic enough that a
    #: palette colour cannot be told from image content.  The stray half of
    #: the check is then not evidence and is not counted; the presence half
    #: still is.  Reported so a skipped half is never silently a pass.
    stray_checked: bool = True

    @property
    def passed(self) -> bool:
        if self.stray_checked and self.stray_pixels:
            return False
        if self.visible_length_px < ROI_PLACEMENT_MIN_VISIBLE_LENGTH_PX:
            return True
        return self.matched_in_band > 0


@dataclass(frozen=True)
class RoiPlacementReport:
    rois: tuple[RoiPlacementComparison, ...]
    frame_shape: tuple[int, int]

    def failures(self) -> tuple[RoiPlacementComparison, ...]:
        return tuple(roi for roi in self.rois if not roi.passed)


def _add_segment(mask: np.ndarray, p0, p1, radius: float) -> None:
    """OR every pixel within ``radius`` of the ``p0``-``p1`` segment into ``mask``.

    Accumulating in place, over the segment's own bounding box only, keeps the
    whole coverage build off the GUI thread's critical path: a per-primitive
    full-frame temporary cost ~34 ms per checkpoint and tripped the profile
    harness's own 50 ms callback budget.
    """

    height, width = mask.shape
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    radius = max(0.5, float(radius))
    lo_x = max(0, int(np.floor(min(x0, x1) - radius)))
    hi_x = min(width - 1, int(np.ceil(max(x0, x1) + radius)))
    lo_y = max(0, int(np.floor(min(y0, y1) - radius)))
    hi_y = min(height - 1, int(np.ceil(max(y0, y1) + radius)))
    if hi_x < lo_x or hi_y < lo_y:
        return
    xs = np.arange(lo_x, hi_x + 1, dtype=np.float64)[None, :] + 0.5
    ys = np.arange(lo_y, hi_y + 1, dtype=np.float64)[:, None] + 0.5
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        distance_sq = (xs - x0) ** 2 + (ys - y0) ** 2
    else:
        t = np.clip(((xs - x0) * dx + (ys - y0) * dy) / length_sq, 0.0, 1.0)
        distance_sq = (xs - (x0 + t * dx)) ** 2 + (ys - (y0 + t * dy)) ** 2
    mask[lo_y : hi_y + 1, lo_x : hi_x + 1] |= distance_sq <= radius * radius


def _add_rect(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float, margin: float) -> None:
    """OR an axis-aligned rect grown by ``margin`` into ``mask``."""

    height, width = mask.shape
    lo_x = max(0, int(np.floor(min(x0, x1) - margin)))
    hi_x = min(width - 1, int(np.ceil(max(x0, x1) + margin)))
    lo_y = max(0, int(np.floor(min(y0, y1) - margin)))
    hi_y = min(height - 1, int(np.ceil(max(y0, y1) + margin)))
    if hi_x < lo_x or hi_y < lo_y:
        return
    mask[lo_y : hi_y + 1, lo_x : hi_x + 1] = True


def _polyline_coverage(shape, points, radius: float) -> np.ndarray:
    mask = np.zeros((int(shape[0]), int(shape[1])), dtype=bool)
    for start, end in itertools.pairwise(points):
        _add_segment(mask, start, end, radius)
    return mask


def _polyline_bounds(shape, points, radius: float) -> tuple[int, int, int, int] | None:
    """Clipped ``(lo_x, lo_y, hi_x, hi_y)`` covering a dilated polyline."""

    height, width = int(shape[0]), int(shape[1])
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    if not xs:
        return None
    lo_x = max(0, int(np.floor(min(xs) - radius)))
    hi_x = min(width - 1, int(np.ceil(max(xs) + radius)))
    lo_y = max(0, int(np.floor(min(ys) - radius)))
    hi_y = min(height - 1, int(np.ceil(max(ys) + radius)))
    if hi_x < lo_x or hi_y < lo_y:
        return None
    return (lo_x, lo_y, hi_x, hi_y)


def _visible_polyline_length(shape, points) -> float:
    """Total polyline length (frame px) clipped to the frame rectangle."""

    height, width = float(shape[0]), float(shape[1])
    total = 0.0
    for start, end in itertools.pairwise(points):
        x0, y0 = float(start[0]), float(start[1])
        x1, y1 = float(end[0]), float(end[1])
        # Cheap conservative clip: sample the segment and count the fraction
        # of it that lies on-frame.  Exact Liang-Barsky is unnecessary here --
        # this number only decides whether a ROI is long enough to demand
        # visible evidence of itself.
        steps = 64
        ts = np.linspace(0.0, 1.0, steps)
        xs = x0 + ts * (x1 - x0)
        ys = y0 + ts * (y1 - y0)
        on_frame = (xs >= 0.0) & (xs <= width) & (ys >= 0.0) & (ys <= height)
        total += float(np.hypot(x1 - x0, y1 - y0) * on_frame.mean())
    return total


def _wgpu_world_projector(camera, frame_h: int, frame_w: int):
    """Return ``project(world_xy) -> frame_xy`` for the WGPU camera."""

    x0, y0, x1, y1 = (float(value) for value in camera.world_rect)
    span_x, span_y = x1 - x0, y1 - y0

    def project(point) -> tuple[float, float]:
        fraction_x = (float(point[0]) - x0) / span_x
        if camera.x_inverted:
            fraction_x = 1.0 - fraction_x
        fraction_y = (float(point[1]) - y0) / span_y
        if not camera.y_inverted:
            fraction_y = 1.0 - fraction_y
        return (fraction_x * frame_w, fraction_y * frame_h)

    return project


def _wgpu_overlay_coverage(img_view, *, frame_shape, camera) -> np.ndarray:
    """Mask of every non-image primitive this frame drew.

    Mirrors ``_OVERLAY_WGSL``'s vertex expansion: ``line`` widths are a
    half-extent in target pixels, ``handle_quad`` is a ``width``-sided square
    centred on its world point, screen-anchored kinds offset from a projected
    anchor in y-down pixels, and ``widget_quad`` ignores the camera entirely.

    Reads ``_wgpu_overlay_geometry`` -- the tuple last handed to the executor,
    which the oracle's own submit folds into the frame it reads back -- rather
    than rebuilding the list.  That is the geometry the target actually
    contains, and it costs nothing.  Independence is not lost by reading the
    backend here: the ROI *placement* half projects ``win.roi_store`` instead,
    so an outline published in the wrong place still has an oracle above it.
    """

    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    project = _wgpu_world_projector(camera, frame_h, frame_w)
    margin = OVERLAY_COVERAGE_MARGIN_PX
    mask = np.zeros((frame_h, frame_w), dtype=bool)
    for primitive in tuple(img_view._wgpu_overlay_geometry or ()):
        anchor = getattr(primitive, "visibility_anchor", None)
        if anchor is not None:
            anchor_x, anchor_y = project(anchor)
            if not (0.0 <= anchor_x <= frame_w and 0.0 <= anchor_y <= frame_h):
                # The shader flings anchor-hidden geometry off-clip, so it
                # tints nothing and must not be withheld from the comparison.
                continue
        kind = str(primitive.kind)
        width = float(primitive.width)
        if kind == "line":
            _add_segment(mask, project(primitive.p0), project(primitive.p1), width + margin)
        elif kind == "world_rect":
            (rx0, ry0), (rx1, ry1) = project(primitive.p0), project(primitive.p1)
            _add_rect(mask, rx0, ry0, rx1, ry1, margin)
        elif kind == "handle_quad":
            cx, cy = project(primitive.p0)
            half = width / 2.0
            _add_rect(mask, cx - half, cy - half, cx + half, cy + half, margin)
        elif kind == "widget_quad":
            ox, oy = (float(value) for value in primitive.screen_offset)
            sx, sy = (float(value) for value in primitive.size)
            _add_rect(mask, ox, oy, ox + sx, oy + sy, margin)
        elif kind in ("screen_rect", "glyph_quad"):
            ax, ay = project(primitive.p0)
            ox, oy = (float(value) for value in primitive.screen_offset)
            sx, sy = (float(value) for value in primitive.size)
            _add_rect(mask, ax + ox, ay + oy, ax + ox + sx, ay + oy + sy, margin)
        else:
            raise AssertionError(f"overlay coverage does not model primitive kind {kind!r}")
    return mask


def _qt_overlay_coverage(win, *, frame_shape, project, scale: float) -> np.ndarray:
    """Mask of the Qt-raster viewport's ROI and profile-marker strokes.

    PyQtGraph draws these as QGraphics items rather than as a published
    primitive list, so the footprint is derived from the same semantic ROI
    geometry and the same ``_roi_visual_style`` width the items were pen'd
    with -- one owner, read twice, never a second geometry.
    """

    from arrayscope.display.overlay_geometry import roi_outline_points
    from arrayscope.display.overlay_hit_test import roi_handle_points

    img_view = win.img_view
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    margin = OVERLAY_COVERAGE_MARGIN_PX
    mask = np.zeros((frame_h, frame_w), dtype=bool)
    for _roi_id, geometry, _color, pen_width in _semantic_roi_outlines(win):
        points = [project(point) for point in roi_outline_points(geometry)]
        for start, end in itertools.pairwise(points):
            _add_segment(mask, start, end, pen_width * scale / 2.0 + margin)
        for handle in roi_handle_points(geometry):
            hx, hy = project(handle)
            # pyqtgraph's scale handle is a fixed-size device-pixel glyph;
            # this half-extent covers it and its hover halo.
            half = 9.0 * scale
            _add_rect(mask, hx - half, hy - half, hx + half, hy + half, margin)
    if bool(getattr(img_view, "_profile_marker_requested_visible", False)):
        position = img_view.profileMarkerPosition()
        if position is not None:
            x, y = (float(position[0]), float(position[1]))
            bx0, by0, bx1, by1 = img_view._current_profile_bounds()
            radius = 2.0 * scale + margin
            _add_segment(mask, project((x, by0)), project((x, by1)), radius)
            _add_segment(mask, project((bx0, y)), project((bx1, y)), radius)
            hx, hy = project((x, y))
            half = 8.0 * scale
            _add_rect(mask, hx - half, hy - half, hx + half, hy + half, margin)
    return mask


def _semantic_roi_outlines(win):
    """Yield ``(roi_id, geometry, rgb, pen_width)`` for every drawn ROI.

    Geometry comes from ``win.roi_store`` -- the semantic owner -- not from a
    backend's mirrored item, so a backend that drew a stale outline is a
    divergence this oracle can see rather than one it inherits.
    """

    store = getattr(win, "roi_store", None)
    selections = tuple(getattr(store, "selections", ()) or ())
    img_view = win.img_view
    for selection in selections:
        if not bool(selection.enabled):
            continue
        width, rgb = img_view._roi_visual_style(str(selection.id), selection.color)
        yield str(selection.id), selection.geometry, tuple(int(v) for v in rgb), float(width)


def _roi_placement_report(
    win,
    *,
    frame_rgb: np.ndarray,
    project,
    band_radius,
    overlay_mask: np.ndarray,
) -> RoiPlacementReport:
    """Check each ROI's colour is present in its band and nowhere else.

    ``band_radius(pen_width)`` returns the stroke half-extent in frame pixels
    for the backend under test.  Stray pixels are counted only outside the
    *whole* overlay mask, so one ROI's colour appearing under another overlay
    (the profile marker shares the first palette entry) is not miscounted as
    misplacement.

    The stray search runs over non-grey, un-masked pixels only.  A greyscale
    LUT emits ``r == g == b``, so under one the candidate set is a handful of
    antialiased overlay fringes and the scan stays inside the harness's
    GUI-callback budget on a full-size frame.  Under a colour LUT the set
    explodes and a palette colour stops being distinguishable from image
    content -- ``stray_checked`` then records that this half of the check was
    not evidence, rather than reporting a pass it did not earn.
    """

    from arrayscope.display.overlay_geometry import roi_outline_points

    frame_shape = (int(frame_rgb.shape[0]), int(frame_rgb.shape[1]))
    red, green, blue = frame_rgb[..., 0], frame_rgb[..., 1], frame_rgb[..., 2]
    candidates = ((red != green) | (green != blue)) & ~overlay_mask
    candidate_y, candidate_x = np.nonzero(candidates)
    candidate_rgb = frame_rgb[candidate_y, candidate_x]
    if candidate_rgb.size:
        chroma = candidate_rgb.max(axis=-1) - candidate_rgb.min(axis=-1)
        keep = chroma >= ROI_COLOR_MIN_CHROMA
        candidate_y, candidate_x = candidate_y[keep], candidate_x[keep]
        candidate_rgb = candidate_rgb[keep]
    stray_checked = candidate_y.size <= int(
        ROI_PLACEMENT_MAX_CHROMATIC_FRACTION * frame_shape[0] * frame_shape[1]
    )
    comparisons: list[RoiPlacementComparison] = []
    for roi_id, geometry, rgb, pen_width in _semantic_roi_outlines(win):
        points = [project(point) for point in roi_outline_points(geometry)]
        if len(points) < 2:
            continue
        target = np.asarray(rgb, dtype=np.int16)
        stray_count = 0
        extent = None
        if stray_checked and candidate_rgb.size:
            stray = np.abs(candidate_rgb - target).max(axis=-1) <= ROI_COLOR_MATCH_TOLERANCE
            stray_count = int(np.count_nonzero(stray))
            if stray_count:
                xs, ys = candidate_x[stray], candidate_y[stray]
                extent = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        radius = band_radius(pen_width)
        bounds = _polyline_bounds(frame_shape, points, radius)
        band_pixels = 0
        matched_in_band = 0
        if bounds is not None:
            lo_x, lo_y, hi_x, hi_y = bounds
            window = np.zeros((hi_y - lo_y + 1, hi_x - lo_x + 1), dtype=bool)
            for start, end in itertools.pairwise(points):
                _add_segment(
                    window,
                    (start[0] - lo_x, start[1] - lo_y),
                    (end[0] - lo_x, end[1] - lo_y),
                    radius,
                )
            band_pixels = int(np.count_nonzero(window))
            patch = frame_rgb[lo_y : hi_y + 1, lo_x : hi_x + 1]
            in_band = np.abs(patch - target).max(axis=-1) <= ROI_COLOR_MATCH_TOLERANCE
            matched_in_band = int(np.count_nonzero(in_band & window))
        comparisons.append(
            RoiPlacementComparison(
                roi_id=roi_id,
                color=rgb,
                visible_length_px=_visible_polyline_length(frame_shape, points),
                band_pixels=band_pixels,
                matched_in_band=matched_in_band,
                stray_pixels=stray_count,
                stray_extent=extent,
                stray_checked=bool(stray_checked),
            )
        )
    return RoiPlacementReport(rois=tuple(comparisons), frame_shape=frame_shape)


def qt_raster_matches_cpu_reference(
    win,
    *,
    tiles=None,
    tolerance: int = QT_RASTER_TOLERANCE,
    max_mismatch_fraction: float = QT_RASTER_MAX_MISMATCH_FRACTION,
    min_samples_per_tile: int = DEFAULT_MIN_SAMPLES_PER_TILE,
    texel_guard: float = DEFAULT_TEXEL_GUARD,
    edge_inset_px: float = DEFAULT_EDGE_INSET_PX,
    max_samples_per_tile: int | None = None,
    sample_seed: int | None = None,
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
    layer = getattr(img_view, "_montage_tile_layer", None)
    if layer is None:
        raise AssertionError(
            "Qt-raster CPU-reference oracle needs the PyQtGraph montage tile layer"
        )
    required, payloads, plan_tiles = _required_payloads_and_plan(win, tiles)
    session = win.renderer._frame_session
    image_axes = tuple(int(axis) for axis in (session.view_state.image_axes or ()))
    transposed = bool(
        getattr(session, "canonical_orientation", False)
        and len(image_axes) == 2
        and image_axes[0] > image_axes[1]
    )
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

    def project(point) -> tuple[float, float]:
        viewport_xy = np.asarray(point, dtype=np.float64) @ view_matrix.T + view_offset
        return (float(viewport_xy[0]) * scale_x, float(viewport_xy[1]) * scale_y)

    overlay_mask = _qt_overlay_coverage(
        win,
        frame_shape=frame.shape[:2],
        project=project,
        scale=float(max(scale_x, scale_y)),
    )
    reports: list[TileComparison] = []
    overlay_excluded = 0
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
        if transposed:
            expected_rgb = np.swapaxes(expected_rgb, 0, 1)
            background_mask = np.swapaxes(background_mask, 0, 1)
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
        drawn_over = overlay_mask[grid_y, grid_x]
        overlay_excluded += int(np.count_nonzero(inside & guarded & drawn_over))
        select = inside & guarded & ~drawn_over
        sample = _bounded_sample_positions(
            select,
            max_samples=max_samples_per_tile,
            sample_seed=sample_seed,
            tile_number=tile_number,
        )
        if not sample.size:
            reports.append(
                TileComparison(
                    tile_number=tile_number,
                    samples=0,
                    mismatched=0,
                    worst_diff=0,
                    mean_diff=0.0,
                    detail=(
                        "no pixel centers survive the interior/texel guards "
                        "and the overlay coverage mask"
                    ),
                )
            )
            continue
        index_x = np.clip(np.floor(texel_x[sample]).astype(np.int64), 0, tex_w - 1)
        index_y = np.clip(np.floor(texel_y[sample]).astype(np.int64), 0, tex_h - 1)
        expected = expected_rgb[index_y, index_x]
        is_background = background_mask[index_y, index_x]
        if np.any(is_background):
            expected = expected.copy()
            expected[is_background] = background
        actual = frame_rgb[grid_y[sample], grid_x[sample]]
        diff = np.abs(actual - expected).max(axis=-1)
        mismatched = diff > tolerance
        worst_index = int(np.argmax(diff))
        detail = ""
        if np.any(mismatched):
            detail = (
                f"worst at viewport ({int(grid_x[sample][worst_index])}, "
                f"{int(grid_y[sample][worst_index])}) texel "
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
        overlay_excluded_samples=int(overlay_excluded),
        # Qt pens are centred on the path, so half the pen width each side.
        roi_placement=_roi_placement_report(
            win,
            frame_rgb=frame_rgb,
            project=project,
            band_radius=lambda pen: pen * max(scale_x, scale_y) / 2.0 + OVERLAY_COVERAGE_MARGIN_PX,
            overlay_mask=overlay_mask,
        ),
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
    max_samples_per_tile: int | None = None,
    sample_seed: int | None = None,
) -> FrameReferenceReport:
    """Compare WGPU's physical render target with current semantic payloads.

    This deliberately reads the executor target, not ``tileTruthPhysicalRows``:
    a page-table key can be current while its texels still belong to an older
    crop window.  The expected image is derived from the frame session's
    payloads and montage plan; only the public camera state is shared with the
    renderer.

    Pixels the frame draws over the image -- ROI strokes, the profile marker,
    lifecycle boxes, tile-truth glyphs, floating chips -- are withheld through
    ``_wgpu_overlay_coverage``, which masks only where each overlay's own
    geometry places it.  See that function for why misplaced overlays are
    still caught, and ``roi_placement_matches_geometry`` for the presence half
    of the contract.
    """

    from arrayscope.gpu.command_protocol import SetDisplayMapping, UpdateTileInstances

    img_view = win.img_view
    executor = getattr(img_view, "_wgpu_executor", None)
    if executor is None:
        raise AssertionError("WGPU CPU-reference oracle needs a live WGPU executor")
    required, payloads, plan_tiles = _required_payloads_and_plan(win, tiles)
    ui_levels = tuple(float(value) for value in img_view.getLevels())
    physical_levels = (
        float(img_view._wgpu_mapping_state.level_lo),
        float(img_view._wgpu_mapping_state.level_hi),
    )
    if not np.allclose(ui_levels, physical_levels, rtol=0.0, atol=1e-6):
        session = win.renderer._frame_session
        level_snapshot = session.level_presentation_snapshot()
        raise AssertionError(
            "WGPU display mapping is stale relative to the semantic level "
            f"owner: ui_levels={ui_levels}, physical_levels={physical_levels}, "
            f"level_snapshot={level_snapshot!r}"
        )
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

    overlay_mask = _wgpu_overlay_coverage(img_view, frame_shape=(frame_h, frame_w), camera=camera)
    transposed = bool(getattr(img_view, "_wgpu_committed", {}).get("transposed", False))
    committed_tiles = dict(getattr(img_view, "_wgpu_committed", {}).get("tiles", {}) or {})
    reports: list[TileComparison] = []
    overlay_excluded = 0
    for tile_number in sorted(required):
        tile = plan_tiles[tile_number]
        payload = payloads[tile_number]
        mapping = resolve_reference_mapping(win, payload)
        committed_info = dict(committed_tiles.get(tile_number, {}) or {})
        physical_lod = int(committed_info.get("lod_level", 0) or 0)
        payload_lod = int(getattr(getattr(payload, "lod", None), "level", 0) or 0)
        # A reduced candidate is allowed to bind already-resident native
        # source pages.  In that case the GPU samples the payload's exact
        # semantic plane, not its coarser presentation texture.  Comparing
        # against the latter makes a sharper frame look wrong and, worse,
        # cannot detect a stale native crop.  The semantic plane is immutable
        # worker output and therefore remains independent of backend storage.
        reference_values = (
            _native_reference_values(win, win.renderer._frame_session, tile, payload)
            if physical_lod == 0 and payload_lod > 0
            else None
        )
        expected_rgb, background_mask = cpu_reference_tile_image(
            payload,
            mapping,
            values=reference_values,
        )
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
        drawn_over = overlay_mask[grid_y, grid_x]
        overlay_excluded += int(np.count_nonzero(inside & guarded & drawn_over))
        select = inside & guarded & ~drawn_over
        sample = _bounded_sample_positions(
            select,
            max_samples=max_samples_per_tile,
            sample_seed=sample_seed,
            tile_number=tile_number,
        )
        if not sample.size:
            reports.append(
                TileComparison(
                    tile_number=tile_number,
                    samples=0,
                    mismatched=0,
                    worst_diff=0,
                    mean_diff=0.0,
                    detail=(
                        "no pixel centers survive the interior/texel guards "
                        "and the overlay coverage mask"
                    ),
                )
            )
            continue

        index_x = np.clip(np.floor(texel_x[sample]).astype(np.int64), 0, tex_w - 1)
        index_y = np.clip(np.floor(texel_y[sample]).astype(np.int64), 0, tex_h - 1)
        expected = expected_rgb[index_y, index_x]
        is_background = background_mask[index_y, index_x]
        if np.any(is_background):
            expected = expected.copy()
            expected[is_background] = 0
        actual = frame_rgb[grid_y[sample], grid_x[sample]]
        diff = np.abs(actual - expected).max(axis=-1)
        mismatched = diff > tolerance
        worst_index = int(np.argmax(diff))
        detail = ""
        if np.any(mismatched):
            detail = (
                f"worst at target ({int(grid_x[sample][worst_index])}, "
                f"{int(grid_y[sample][worst_index])}) texel "
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
        overlay_excluded_samples=int(overlay_excluded),
        # WGSL expands a line by ``width`` on EACH side of its centreline.
        roi_placement=_roi_placement_report(
            win,
            frame_rgb=frame_rgb,
            project=_wgpu_world_projector(camera, frame_h, frame_w),
            band_radius=lambda pen: pen + OVERLAY_COVERAGE_MARGIN_PX,
            overlay_mask=overlay_mask,
        ),
    )


def roi_placement_matches_geometry(win, *, backend: str) -> RoiPlacementReport:
    """Check every drawn ROI outline sits where its semantic geometry says.

    The complement of the image comparison: that oracle proves no overlay
    stroke tinted a pixel it had no business tinting, this one proves each
    ROI's colour really is on the frame, inside its own band, after whatever
    the stage just did to the displayed axis.  Together they keep a cropped
    montage's ROI coverage without letting overlay pixels masquerade as image
    divergence -- and without letting an overlay that stopped being drawn at
    all pass unnoticed.

    Standalone form.  Callers that also want the image verdict should read
    ``FrameReferenceReport.roi_placement`` instead: both verdicts then come
    from one capture, which is both cheaper and the only way to be sure they
    describe the same frame.
    """

    backend = str(backend)
    if backend == "wgpu":
        report = wgpu_frame_matches_cpu_reference(win)
    elif backend == "pyqtgraph":
        report = qt_raster_matches_cpu_reference(win)
    else:
        raise AssertionError(f"ROI placement oracle does not cover backend {backend!r}")
    if report.roi_placement is None:
        raise AssertionError("frame reference report carried no ROI placement verdict")
    return report.roi_placement


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


def assert_wgpu_frame_matches_cpu_reference(win, **kwargs) -> FrameReferenceReport:
    """Asserting form of :func:`wgpu_frame_matches_cpu_reference`."""

    report = wgpu_frame_matches_cpu_reference(win, **kwargs)
    try:
        return _assert_report(report, label="WGPU target")
    except AssertionError as exc:
        rows = dict(win.img_view.tileTruthPhysicalRows() or {})
        physical_lods: dict[int, int] = {}
        for row in rows.values():
            level = int(row.get("physical_lod_level", 0) or 0)
            physical_lods[level] = physical_lods.get(level, 0) + 1
        session = win.renderer._frame_session
        committed = dict(getattr(win.img_view, "_wgpu_committed", {}) or {})
        committed_tiles = dict(committed.get("tiles", {}) or {})
        sample_tile = min(committed_tiles, default=None)
        sample_info = (
            {} if sample_tile is None else dict(committed_tiles.get(sample_tile, {}) or {})
        )
        sample_payload = (
            None
            if sample_tile is None
            else dict(session.display_tile_payloads).get(int(sample_tile))
        )
        sample_anchor = getattr(sample_payload, "source_anchor", None)
        sample_rendered = (
            None if sample_tile is None else dict(session.rendered_tiles).get(int(sample_tile))
        )
        sample_plan_tile = next(
            (
                tile
                for tile in tuple(session.plan.tiles)
                if int(tile.montage_index) == int(sample_tile)
            ),
            None,
        )
        raise AssertionError(
            f"{exc}\n"
            f"  physical_lods={physical_lods}, "
            f"desired_lod={getattr(session, 'desired_lod_level', None)}, "
            f"target_settled={session.required_target_settled()}, "
            f"levels={tuple(float(value) for value in win.img_view.getLevels())}, "
            f"sample_committed_world={sample_info.get('world_rect')}, "
            f"sample_plan_world={None if sample_plan_tile is None else (sample_plan_tile.x0, sample_plan_tile.y0, sample_plan_tile.width, sample_plan_tile.height)}, "
            f"sample_src_origin={sample_info.get('src_origin')}, "
            f"sample_src_size={sample_info.get('src_size')}, "
            f"sample_texture_shape={None if sample_payload is None else np.asarray(sample_payload.texture_data).shape}, "
            f"sample_rendered_shape={None if sample_rendered is None else np.asarray(sample_rendered.image).shape}, "
            f"sample_rendered_semantic_shape={None if sample_rendered is None or sample_rendered.semantic_data is None else np.asarray(sample_rendered.semantic_data).shape}, "
            f"sample_native_residency_shape={None if sample_payload is None or sample_payload.native_residency_data is None else np.asarray(sample_payload.native_residency_data).shape}, "
            f"sample_anchor={sample_anchor!r}, "
            f"bound_plane_shapes={tuple(plane.plane_shape for plane in win.img_view._wgpu_executor._bound_planes)}"
        ) from exc


def assert_qt_raster_matches_cpu_reference(win, **kwargs) -> FrameReferenceReport:
    """Asserting form of :func:`qt_raster_matches_cpu_reference`."""

    return _assert_report(
        qt_raster_matches_cpu_reference(win, **kwargs),
        label="Qt raster",
    )
