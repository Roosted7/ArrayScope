"""Stateful per-tile montage display items."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from math import ceil
from time import perf_counter

import numpy as np
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from pyqtgraph.Qt import QtGui

from arrayscope.display.image_upload import rgb_display_for_levels
from arrayscope.display.lod import LodInfo
from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.model.presentation_generation import levels_match
from arrayscope.display.model.tile_identity import (
    TileLodIdentity,
    acknowledged_identity_satisfies_target,
    array_plane_identities,
    plane_identity_record,
)
from arrayscope.display.model.tile_stats import TileLayerUpdateStats
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    TexturePlaneKind,
    apply_phase_lut,
    cpu_display_rgba,
    mapped_scalar,
)
from arrayscope.display.tile_layout import tile_layout_map
from arrayscope.gpu.keys import REDUCER_PHASE_VECTOR, DataChunkKey
from arrayscope.gpu.page_table import PageResolution

RGB_SOURCE_CACHE_BUDGET_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class _PageAssembly:
    payload: DisplayTilePayload
    resolutions: tuple[PageResolution, ...] = ()
    missing: tuple[DataChunkKey, ...] = ()
    fallback_reason: str | None = None


def _resolve_page_backed_payload(
    payload: DisplayTilePayload,
    *,
    levels: tuple[float, float] | None = None,
) -> _PageAssembly:
    """Resolve supplied CPU pages through the canonical page table once."""

    backing = payload.page_backing
    if backing is None:
        return _PageAssembly(payload)
    resident_pages = tuple(backing.materialized_pages)
    pages_by_key = {page.key: page for page in resident_pages}
    resolutions = backing.candidate_resolutions
    missing = tuple(
        target
        for target, resolution in zip(backing.requested_keys, resolutions, strict=True)
        if resolution is None
    )
    if missing:
        if payload.semantic_data is None:
            raise ValueError(
                "PyQtGraph cannot present incomplete page-backed coverage and no native fallback exists"
            )
        native = np.ascontiguousarray(payload.semantic_data)
        return _PageAssembly(
            _native_page_fallback(
                payload,
                native,
                marker="native-page-fallback",
                levels=levels,
            ),
            missing=missing,
            fallback_reason="incomplete-page-coverage-native",
        )
    resolved_page_set = backing.resolved_page_set
    if resolved_page_set is None:
        raise RuntimeError("complete page targets have no resolved page-set record")
    resolved = resolved_page_set.resolutions
    # Assembly is target-aligned even when several targets share one coarse
    # actual page. Storage stays deduplicated in ``resolved_page_set``.
    resolved_pages = tuple(pages_by_key[resolution.actual_key] for resolution in resolved)

    y0, y1, x0, x1 = backing.source_coverage_yx
    sample = resolved_pages[0].values
    trailing = tuple(np.shape(sample)[2:])
    assembled = np.empty((y1 - y0, x1 - x0, *trailing), dtype=np.asarray(sample).dtype)
    coverage = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for target_plan, page in zip(backing.requested_plans, resolved_pages, strict=True):
        plan = page.plan
        target_y0, target_y1, target_x0, target_x1 = target_plan.valid_source_rect_yx
        sy0, sy1 = max(y0, target_y0), min(y1, target_y1)
        sx0, sx1 = max(x0, target_x0), min(x1, target_x1)
        if sy0 >= sy1 or sx0 >= sx1:
            continue
        first = plan.stored_index_for_source(sy0, sx0)
        last = plan.stored_index_for_source(sy1 - 1, sx1 - 1)
        if first is None or last is None:
            raise ValueError("resolved page does not cover its requested target geometry")
        row0, column0 = first
        row1, column1 = last[0] + 1, last[1] + 1
        y_counts = tuple(end - start for start, end in plan.source_y_bins[row0:row1])
        x_counts = tuple(end - start for start, end in plan.source_x_bins[column0:column1])
        expanded = np.repeat(
            np.repeat(page.values[row0:row1, column0:column1], y_counts, axis=0),
            x_counts,
            axis=1,
        )
        expanded_y0 = plan.source_y_bins[row0][0]
        expanded_x0 = plan.source_x_bins[column0][0]
        assembled[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0, ...] = expanded[
            sy0 - expanded_y0 : sy1 - expanded_y0,
            sx0 - expanded_x0 : sx1 - expanded_x0,
            ...,
        ]
        coverage[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] += 1
    if not np.all(coverage == 1):
        raise ValueError("page-backed PyQtGraph assembly has incomplete or overlapping geometry")
    assembled = np.ascontiguousarray(assembled)
    if np.iscomplexobj(assembled):
        return _PageAssembly(
            _map_complex_cpu_payload(payload, assembled, levels=levels),
            resolutions=resolved,
        )
    return _PageAssembly(
        replace(payload, image=assembled, texture_data=assembled),
        resolutions=resolved,
    )


def _native_page_fallback(
    payload: DisplayTilePayload,
    native: np.ndarray,
    *,
    marker: str,
    levels: tuple[float, float] | None,
) -> DisplayTilePayload:
    """Return an honestly native payload when CPU page assembly is unsafe."""

    native = np.ascontiguousarray(native)
    if np.iscomplexobj(native):
        mapped = _map_complex_cpu_payload(payload, native, levels=levels)
    else:
        if native.ndim >= 3 and native.shape[-1] in (3, 4):
            kind = TexturePlaneKind.RGB8
        else:
            kind = TexturePlaneKind.SCALAR_R32F
        histogram = getattr(payload, "semantic_histogram_data", None)
        mapped = replace(
            payload,
            image=native,
            texture_data=native,
            histogram_data=None if histogram is None else np.asarray(histogram),
            texture_kind=kind,
        )
    native_lod = LodInfo(
        level=0,
        factor=1,
        source_shape=tuple(int(value) for value in native.shape[:2]),
        texture_shape=tuple(int(value) for value in native.shape[:2]),
        gutter=0,
    )
    identity = getattr(payload, "tile_identity", None)
    if identity is not None:
        identity = replace(
            identity,
            lod=TileLodIdentity(level=0, factor=1, gutter=0),
            quality="exact",
        )
    return replace(
        mapped,
        source_id=(payload.source_id, marker),
        source_shape=tuple(int(value) for value in native.shape[:2]),
        lod=native_lod,
        quality="exact",
        tile_identity=identity,
        page_backing=None,
    )


def _map_complex_cpu_payload(
    payload: DisplayTilePayload,
    values: np.ndarray,
    *,
    levels: tuple[float, float] | None,
) -> DisplayTilePayload:
    """Apply the payload's complex view semantics to CPU-resident values."""

    mapping = payload.shader_mapping
    if mapping is None:
        raise ValueError("PyQtGraph cannot safely present complex values without a shader mapping")
    if levels is not None:
        mapping = replace(mapping, levels=levels)
    if mapping.display_mode != ShaderDisplayMode.PHASE_COLOR:
        scalar = np.ascontiguousarray(mapped_scalar(values, mapping))
        return replace(
            payload,
            image=scalar,
            texture_data=scalar,
            histogram_data=scalar,
            texture_kind=TexturePlaneKind.SCALAR_R32F,
        )
    if mapping.component not in {
        ShaderComponent.ANGLE,
        ShaderComponent.COMPLEX_PHASE,
    }:
        display, _magnitude = apply_phase_lut(values, mapping.lut_data)
        histogram = np.ascontiguousarray(mapped_scalar(values, mapping))
        return replace(
            payload,
            image=np.ascontiguousarray(display),
            texture_data=np.ascontiguousarray(display),
            histogram_data=histogram,
            texture_kind=TexturePlaneKind.RGB8,
            rgb_windowed_levels=None,
        )
    backing = payload.page_backing
    plans = tuple(getattr(backing, "requested_plans", ()) or ())
    phase_vector = bool(plans and all(plan.reducer == REDUCER_PHASE_VECTOR for plan in plans))
    if phase_vector:
        # Match VisPy's phase-vector shader mode exactly: page magnitude is
        # circular-resultant coherence in [0, 1], not native amplitude, and
        # hue spans the canonical phase range rather than the amplitude level
        # window.  Opposed/all-zero bins therefore stay visibly undefined
        # (black) on both backends instead of acquiring an arbitrary hue.
        phase_mapping = replace(mapping, levels=(-np.pi, np.pi))
        rgba = cpu_display_rgba(values, phase_mapping)
        coherence = np.clip(np.abs(values), 0.0, 1.0).astype(np.float32, copy=False)
        display = np.ascontiguousarray(
            np.clip(
                rgba[..., :3].astype(np.float32) * coherence[..., np.newaxis],
                0.0,
                255.0,
            ).astype(np.uint8)
        )
    else:
        display = np.ascontiguousarray(cpu_display_rgba(values, mapping)[..., :3])
    return replace(
        payload,
        image=display,
        texture_data=display,
        histogram_data=None,
        texture_kind=TexturePlaneKind.RGB8,
        rgb_windowed_levels=levels,
    )


def _assemble_page_backed_payload(
    payload: DisplayTilePayload,
    *,
    levels: tuple[float, float] | None = None,
) -> DisplayTilePayload:
    """Assemble exact native geometry for PyQtGraph's one-item CPU path.

    Reduced samples are repeated over their exact planned source rectangles,
    not uniformly stretched. This is nearest-neighbour presentation with the
    same clipped-bin geometry as VisPy.
    """

    return _resolve_page_backed_payload(payload, levels=levels).payload


def _payload_direct_dims(region, tile_data, payload):
    """World footprint + image crop for a possibly LOD-reduced payload.

    Returns ``(world_w, world_h, crop_w, crop_h, scale_x, scale_y)``:
    ``world_*`` are native texels (the item's on-screen footprint),
    ``crop_*`` are image pixels to keep, and ``scale_*`` map image pixels
    to world units (ADR 0050 phase 3 — PyQtGraph adoption).  Native
    payloads return the historical min(region, image) behavior with an
    identity scale.
    """

    img_h = int(tile_data.shape[0])
    img_w = int(tile_data.shape[1])
    lod = getattr(payload, "lod", None)
    factor = int(getattr(lod, "factor", 1) or 1) if lod is not None else 1
    if factor <= 1 or img_w <= 0 or img_h <= 0:
        width = min(int(region.width), img_w)
        height = min(int(region.height), img_h)
        return width, height, width, height, 1.0, 1.0
    src_h, src_w = (int(value) for value in lod.source_shape)
    scale_x = float(src_w) / float(img_w)
    scale_y = float(src_h) / float(img_h)
    world_w = min(int(region.width), src_w)
    world_h = min(int(region.height), src_h)
    crop_w = min(img_w, ceil(world_w / scale_x))
    crop_h = min(img_h, ceil(world_h / scale_y))
    return world_w, world_h, crop_w, crop_h, scale_x, scale_y


def _apply_item_lod_scale(state: TileLayerItemState, scale_x: float, scale_y: float) -> None:
    """Set the item transform mapping image pixels onto native texels.

    Idempotent: only touches the QGraphicsItem transform when the scale
    actually changes (transform churn invalidates the paint cache).
    """

    scale = (float(scale_x), float(scale_y))
    if tuple(getattr(state, "lod_scale", (1.0, 1.0))) == scale:
        return
    state.lod_scale = scale
    transform = QtGui.QTransform()
    if scale != (1.0, 1.0):
        transform.scale(scale[0], scale[1])
    state.item.setTransform(transform)


def _payload_rgb_already_windowed(
    payload: DisplayTilePayload,
    fallback: bool,
    *,
    levels: tuple[float, float] | None = None,
) -> bool:
    """Per-payload windowing from payload METADATA, never from dtype sniffing.

    A plane is immutably pre-windowed ONLY when it carries nothing to
    re-window against (``histogram_data is None``). Everything else — exact
    complex tiles AND preview/LOD planes with their reduced histograms —
    keeps the rewindow-on-level-change contract via rgb_base/hist_source.

    Field defect that fixed this rule: resident-LOD complex planes on
    PyQtGraph are windowed at provisional levels at evaluation time; a
    "preview planes never rewindow" shortcut froze them at those garbage
    levels forever (saturated tiles in exactly the refined, center-out
    region). Rewindowing a reduced plane against its reduced histogram is
    cheap and correct; skipping it is neither.
    """

    if bool(fallback):
        return True
    image = getattr(payload, "image", None)
    is_rgb_plane = bool(
        image is not None
        and getattr(image, "ndim", 0) == 3
        and int(getattr(image, "shape", (0, 0, 0))[-1]) in (3, 4)
    )
    # Non-RGB planes: the flag is meaningless — keep it at the commit-global
    # value so it never perturbs resident-state identity matching.
    if not is_rgb_plane:
        return False
    if getattr(payload, "histogram_data", None) is None:
        return True
    payload_levels = getattr(payload, "rgb_windowed_levels", None)
    return bool(
        levels is not None and payload_levels is not None and levels_match(payload_levels, levels)
    )


@dataclass
class TileLayerItemState:
    tile_number: int
    source_index: int
    item: ImageItem
    local_rect: tuple[int, int, int, int]
    world_rect: tuple[int, int, int, int]
    source_array_id: object
    histogram_array_id: object | None
    levels: tuple[float, float]
    rgb_already_windowed: bool
    visible: bool
    rgb_base: np.ndarray | None = None
    hist_source: np.ndarray | None = None
    display_cache: np.ndarray | None = None
    source_cache_serial: int = 0
    source_cache_nbytes: int = 0
    resident_serial: int = 0
    resident_nbytes: int = 0
    # Image-pixel → world-texel scale for LOD-reduced payloads (ADR 0050
    # phase 3): (1.0, 1.0) means native and an identity item transform.
    lod_scale: tuple[float, float] = (1.0, 1.0)
    acknowledged_identity: object = None
    page_resolutions: tuple[PageResolution, ...] = ()
    page_candidate_missing: tuple[DataChunkKey, ...] = ()
    physical_lod: LodInfo | None = None
    physical_quality: str | None = None
    page_fallback_reason: str | None = None


class MontageTileLayer:
    def __init__(
        self,
        layer_owner,
        *,
        set_image_item_data: Callable,
        record_upload_timing: Callable[[str, float], None],
        histogram_levels_for_display: Callable,
        is_rgb_image: Callable[[object], bool],
    ):
        self.layer_owner = layer_owner
        self._set_image_item_data = set_image_item_data
        self._record_upload_timing = record_upload_timing
        self._histogram_levels_for_display = histogram_levels_for_display
        self._is_rgb_image = is_rgb_image
        self._states: dict[int, TileLayerItemState] = {}
        self._states_by_source_key: dict[object, TileLayerItemState] = {}
        self._direct_reuse_pool: list[TileLayerItemState] = []
        self._direct_reuse_pool_ids: set[int] = set()
        self._source_cache_serial = 0
        self._rgb_source_cache_bytes = 0
        self._resident_serial = 0
        self._resident_bytes = 0
        self._rgb_source_cache_budget_bytes = RGB_SOURCE_CACHE_BUDGET_BYTES

    @property
    def states(self) -> dict[int, TileLayerItemState]:
        return self._states

    @property
    def physically_visible_tile_count(self) -> int:
        """Number of tile ImageItems that can contribute pixels right now.

        The shell's selected montage mode is desired routing state, not
        physical evidence: it is assigned before an update and can remain
        ``tile_layer`` after a zero-item candidate.  Derive visibility from
        the concrete item and its installed image instead.
        """

        return sum(
            1
            for state in self._states.values()
            if _state_is_physically_visible(state)
            and getattr(state.item, "image", None) is not None
            and int(np.size(state.item.image)) > 0
        )

    def tile_truth_physical_rows(self) -> dict[int, dict[str, object]]:
        """Describe the arrays and mapping the visible ImageItems draw now."""

        rows: dict[int, dict[str, object]] = {}
        for tile_number, state in self._states.items():
            image = getattr(state.item, "image", None)
            if not _state_is_physically_visible(state) or image is None:
                continue
            values = np.asarray(image)
            if values.ndim >= 3 and values.shape[-1] in (3, 4):
                kind = TexturePlaneKind.RGB8
                mapping_mode = "cpu_rgb"
            elif np.iscomplexobj(values) or (values.ndim >= 3 and values.shape[-1] == 2):
                kind = TexturePlaneKind.COMPLEX_RG32F
                mapping_mode = "complex_array"
            else:
                kind = TexturePlaneKind.SCALAR_R32F
                mapping_mode = "scalar_levels"
            real_plane, imag_plane = array_plane_identities(values)
            rows[int(tile_number)] = {
                "physical_texture_kind": kind.value,
                "physical_storage_mode": "image_item",
                "physical_texture_dtype": str(values.dtype),
                "physical_texture_shape": tuple(int(value) for value in values.shape),
                "physical_real_plane_identity": plane_identity_record(real_plane),
                "physical_imag_plane_identity": plane_identity_record(imag_plane),
                "physical_mapping_mode": mapping_mode,
                "physical_component_mode": None,
                "physical_levels": tuple(float(value) for value in state.levels),
                "physical_acknowledged_identity": state.acknowledged_identity,
                "physical_lod_level": (
                    None if state.physical_lod is None else int(state.physical_lod.level)
                ),
                "physical_lod_factor": (
                    None if state.physical_lod is None else int(state.physical_lod.factor)
                ),
                "physical_quality": state.physical_quality,
            }
            if state.page_resolutions:
                rows[int(tile_number)]["physical_page_bindings"] = tuple(
                    {
                        "target_key": resolution.target_key,
                        "actual_key": resolution.actual_key,
                        "actual_lod": resolution.actual_key.lod,
                        "scale": tuple(float(value) for value in resolution.scale),
                        "offset": tuple(float(value) for value in resolution.offset),
                        "quality": (
                            "exact"
                            if resolution.actual_key == resolution.target_key
                            else "fallback"
                        ),
                    }
                    for resolution in state.page_resolutions
                )
            if state.page_candidate_missing:
                rows[int(tile_number)]["physical_page_candidate_missing"] = (
                    state.page_candidate_missing
                )
            if state.page_fallback_reason is not None:
                rows[int(tile_number)]["physical_page_fallback_reason"] = state.page_fallback_reason
        return rows

    def set_lookup_table(self, lut) -> None:
        """Apply the frame colormap to every resident scalar tile item."""

        for state in self._states.values():
            image = getattr(state.item, "image", None)
            if image is not None and np.asarray(image).ndim == 2:
                state.item.setLookupTable(lut)

    def clear(self) -> None:
        for state in tuple(self._states.values()):
            self.layer_owner.remove_tile_item(state.tile_number)
        self._states.clear()
        self._states_by_source_key.clear()
        self._direct_reuse_pool.clear()
        self._direct_reuse_pool_ids.clear()
        self._rgb_source_cache_bytes = 0
        self._resident_bytes = 0

    def hide_all(self) -> None:
        """Hide every mapped tile while retaining its physical residency."""

        for tile_number in tuple(self._states):
            self._hide_tile(int(tile_number))

    def update_presentation(
        self,
        img,
        *,
        histogram_data,
        geometry,
        levels: tuple[float, float],
        rgb_already_windowed: bool,
        dirty_tiles: tuple[int, ...] | None,
        tile_source_ids: dict[int, object] | None = None,
        tile_payloads: dict[int, DisplayTilePayload] | None = None,
        tile_delta=None,
        tile_residency_budget_bytes: int = 0,
        frame_plan=None,
        transposed: bool = False,
    ) -> TileLayerUpdateStats:
        if tile_payloads is not None:
            return self._update_direct_payload_presentation(
                tile_payloads,
                geometry=geometry,
                levels=levels,
                rgb_already_windowed=rgb_already_windowed,
                dirty_tiles=dirty_tiles,
                tile_source_ids=tile_source_ids,
                tile_delta=tile_delta,
                tile_residency_budget_bytes=tile_residency_budget_bytes,
                frame_plan=frame_plan,
                transposed=transposed,
            )
        raise ValueError("PyQtGraph tiled presentation requires typed tile payloads")

    def _update_direct_payload_presentation(
        self,
        tile_payloads: dict[int, DisplayTilePayload],
        *,
        geometry,
        levels: tuple[float, float],
        rgb_already_windowed: bool,
        dirty_tiles: tuple[int, ...] | None,
        tile_source_ids: dict[int, object] | None = None,
        tile_delta=None,
        tile_residency_budget_bytes: int = 0,
        frame_plan=None,
        transposed: bool = False,
    ) -> TileLayerUpdateStats:
        layout = tile_layout_map(geometry, frame_plan=frame_plan)
        if not layout:
            return TileLayerUpdateStats()
        requested_active = (
            {
                int(tile)
                for tile in tuple(getattr(tile_delta, "active_tiles", ()) or ())
                if int(tile) in tile_payloads
            }
            if tile_delta is not None
            else set()
        )
        target_identities = dict(getattr(tile_delta, "target_identities", {}) or {})
        requested_upserts = (
            {int(tile) for tile in tile_payloads}
            if tile_delta is None
            else {int(tile) for tile in dict(getattr(tile_delta, "upserts", {}) or {})}
        )
        drawable_payloads: dict[int, DisplayTilePayload] = {}
        page_assemblies: dict[int, _PageAssembly] = {}
        identity_rejected_tiles: list[int] = []
        # Level-only drain fast path: a commit whose every emitted upsert is a
        # level rewindow of an already-resident, already-presented tile does not
        # touch the other resident tiles' pixels.  Resolving (page-assembling +
        # re-windowing) every active payload here is the whole per-commit cost
        # on a large complex montage (measured: ~1200 ms for 272 tiles vs
        # ~55 ms for the 12 actually re-levelled).  Restrict resolution to the
        # requested upsert slice; the remaining resident tiles keep their drawn
        # pixels and stay in ``active`` via the resident-visibility seed below,
        # exactly as they would have through the general (skip) path.
        level_only_drain = bool(getattr(tile_delta, "level_only_drain", False))
        if level_only_drain:
            resolve_items = tuple(
                (int(tile), tile_payloads[int(tile)])
                for tile in requested_upserts
                if int(tile) in tile_payloads
            )
        else:
            resolve_items = tuple(tile_payloads.items())
        for tile, payload in resolve_items:
            if acknowledged_identity_satisfies_target(
                getattr(payload, "tile_identity", None) or payload.source_id,
                target_identities.get(int(tile)),
            ):
                assembly = _resolve_page_backed_payload(payload, levels=levels)
                drawable_payloads[int(tile)] = assembly.payload
                page_assemblies[int(tile)] = assembly
            elif int(tile) in requested_upserts:
                # Not presentable for this delta's typed target; the payload
                # is dropped from this commit.  Report it loudly — a payload
                # that can NEVER satisfy its target re-appears here on every
                # flush, and silence turned that into a starvation stall
                # (2026-07-16, session 148).  Retained non-upsert payloads a
                # newer target has outrun are excluded: the presenter is not
                # looping on them, so counting them would false-trip the
                # re-commit backoff and the trace invariant.
                identity_rejected_tiles.append(int(tile))
        identity_rejected_tiles.sort()
        active = {
            int(tile)
            for tile in requested_active
            if int(tile) in self._states
            and bool(getattr(self._states[int(tile)], "visible", False))
            and acknowledged_identity_satisfies_target(
                self._states[int(tile)].acknowledged_identity,
                target_identities.get(int(tile)),
            )
        }
        for tile in requested_active - active:
            state = self._states.get(int(tile))
            if state is not None and bool(getattr(state, "visible", False)):
                self._hide_tile(int(tile))
        states = tuple(getattr(geometry, "montage_tile_states", ()) or ())
        dirty_set = None if dirty_tiles is None else {int(tile) for tile in dirty_tiles}
        cold_deadline_ms = (
            None if tile_delta is None else getattr(tile_delta, "cold_deadline_ms", None)
        )
        # Level-only re-windowing refreshes already-resident pixels.  The cold
        # budget is a feedback value that collapses to its floor when the
        # commit pipeline's fixed cost dominates, which throttled level
        # convergence to ~2 tiles per commit on large montages (measured on a
        # 272-tile montage: ~45 s to settle after a level drag).  Refinement
        # gets its own floor so each commit makes real progress while staying
        # within roughly one frame of UI-thread work.
        level_rewindow_deadline_ms = (
            None if cold_deadline_ms is None else max(8.0, float(cold_deadline_ms))
        )
        if level_only_drain:
            # The session already bounded this commit to its upsert cap, and we
            # resolved (page-assembled + re-windowed) exactly that slice above.
            # A mid-loop rewindow deadline would then commit only the first few
            # of the already-paid-for tiles and discard the rest, forcing the
            # discarded tiles to be re-resolved on the next commit — pure wasted
            # CPU that stretches the drain.  Boundedness comes from the upsert
            # cap here, not the time deadline, so let every resolved tile land.
            level_rewindow_deadline_ms = None
        cold_start = perf_counter()
        cold_tiles_committed = 0
        update_start = perf_counter()
        levels = (float(levels[0]), float(levels[1]))
        items_created = 0
        items_updated = 0
        items_skipped = 0
        rgb_window_tiles = 0
        image_replacements = 0
        existing_items_shown = 0
        relocated_tiles = 0
        level_updates = 0
        storage_evictions = 0
        updated_tiles: list[int] = []
        committed_upserts: set[int] = set()
        tile_order = _direct_tile_order(
            layout,
            drawable_payloads,
            tile_delta,
            self._states,
            tile_states=states,
            tile_source_ids=tile_source_ids,
            rgb_already_windowed=bool(rgb_already_windowed),
        )
        level_update_pending_items = sum(
            1
            for tile in requested_active
            if int(tile) in self._states
            and not levels_match(self._states[int(tile)].levels, levels)
        )
        level_update_scope = tuple(tile_order) if requested_upserts else requested_active
        level_update_tiles = tuple(
            int(tile)
            for tile in level_update_scope
            if (not requested_upserts or int(tile) in requested_upserts)
            and int(tile) in self._states
            and not levels_match(self._states[int(tile)].levels, levels)
        )
        if level_update_tiles:
            tile_order = tuple(dict.fromkeys(tuple(tile_order) + tuple(level_update_tiles)))
        # Match the VisPy atlas path's ordering: resolve active payloads to
        # resident identities, bind tile placement to resident storage, then
        # decide whether any data upload is needed.  For PyQtGraph the
        # resident storage is the ImageItem state itself.
        preclaim_specs = _direct_preclaim_specs(
            layout,
            tile_order,
            drawable_payloads,
            states=states,
            tile_source_ids=tile_source_ids,
        )
        cold_holes = _direct_cold_hole_count(preclaim_specs, self._states_by_source_key)
        # Moving an ImageItem is destructive for its old slot.  Unlike VisPy's
        # coherent atlas remap, a backend-local cold deadline could stop before
        # the displaced slot is replaced.  The unconstrained range-shift path
        # still preclaims all resident items; deadline-capped callbacks keep
        # old pixels visible unless the move can be completed safely.
        allow_resident_reuse = cold_deadline_ms is None or cold_holes <= 1
        if allow_resident_reuse:
            for tile_number, spec in preclaim_specs.items():
                item_state = self._states.get(int(tile_number))
                spec_rgb_already_windowed = bool(spec[3])
                if _direct_state_matches(
                    item_state,
                    source_id=spec[0],
                    histogram_id=spec[1],
                    local_rect=spec[2],
                    rgb_already_windowed=spec_rgb_already_windowed,
                ):
                    continue
                self._take_resident_direct_state(
                    int(tile_number),
                    source_id=spec[0],
                    histogram_id=spec[1],
                    local_rect=spec[2],
                    rgb_already_windowed=spec_rgb_already_windowed,
                )
        for tile_number in tile_order:
            region = layout.get(int(tile_number))
            if region is None:
                continue
            source_index = (
                int(region.source_index) if region.source_index is not None else int(tile_number)
            )
            state_value = "loaded"
            if states and tile_number < len(states):
                state_value = str(getattr(states[tile_number], "value", states[tile_number]))
            payload = None if state_value == "skipped" else drawable_payloads.get(int(tile_number))
            if payload is None:
                self._hide_tile(tile_number)
                continue

            if not isinstance(payload, DisplayTilePayload):
                raise TypeError("typed tile-layer payloads must be DisplayTilePayload instances")
            tile_img = payload.image
            if tile_img is None:
                self._hide_tile(tile_number)
                continue
            tile_data = np.asarray(tile_img)
            if tile_data.ndim < 2:
                self._hide_tile(tile_number)
                continue
            if transposed:
                # Canonical payload + X/Y display swap: a cheap transposed VIEW
                # (no copy, shares the canonical buffer) makes the item read
                # display-oriented, so all downstream dims/crop/world-rect logic
                # is unchanged -- the swap costs nothing to materialize.
                tile_data = np.swapaxes(tile_data, 0, 1)
            width, height, crop_w, crop_h, scale_x, scale_y = _payload_direct_dims(
                region, tile_data, payload
            )
            if width <= 0 or height <= 0:
                self._hide_tile(tile_number)
                continue
            if crop_w != int(tile_data.shape[1]) or crop_h != int(tile_data.shape[0]):
                tile_data = tile_data[:crop_h, :crop_w, ...]
            # Histogram/level planes stay native-resolution regardless of the
            # display LOD (ADR 0050 semantic identity): crop in world texels.
            if payload.histogram_data is None:
                tile_hist = None
            else:
                hist_plane = np.asarray(payload.histogram_data)
                if transposed:
                    hist_plane = np.swapaxes(hist_plane, 0, 1)
                tile_hist = hist_plane[:height, :width]

            world_x = int(region.x)
            world_y = int(region.y)
            world_rect = (int(world_x), int(world_y), int(width), int(height))
            base_source_id = (
                tile_source_ids.get(int(tile_number), payload.source_id)
                if tile_source_ids is not None
                else payload.source_id
            )
            payload_rgb_already_windowed = _payload_rgb_already_windowed(
                payload,
                bool(rgb_already_windowed),
                levels=levels,
            )
            source_id = _direct_payload_source_id(base_source_id, payload)
            if transposed:
                # The transposed VIEW is a distinct display array over the same
                # canonical source, so key item/source residency on it: an X/Y
                # swap that keeps the source id (canonical) still re-images the
                # item with the swapped view instead of reusing the stale one.
                source_id = ("axes-transposed", source_id)
            hist_id = ("tile-source", source_id) if tile_hist is not None else None
            local_rect = (0, 0, int(crop_w), int(crop_h))
            item_state = self._states.get(tile_number)
            reused_source = False
            resident_current = _direct_state_matches(
                item_state,
                source_id=source_id,
                histogram_id=hist_id,
                local_rect=local_rect,
                rgb_already_windowed=payload_rgb_already_windowed,
            )
            if allow_resident_reuse and (item_state is None or not resident_current):
                reused = self._take_resident_direct_state(
                    tile_number,
                    source_id=source_id,
                    histogram_id=hist_id,
                    local_rect=local_rect,
                    rgb_already_windowed=payload_rgb_already_windowed,
                )
                if reused is not None:
                    item_state = reused
                    reused_source = True
                    resident_current = True
            existing_item = item_state is not None
            identity_current = bool(
                item_state is not None
                and acknowledged_identity_satisfies_target(
                    item_state.acknowledged_identity,
                    target_identities.get(int(tile_number)),
                )
            )
            geometry_changed = (
                item_state is None
                or tuple(item_state.local_rect) != local_rect
                or tuple(getattr(item_state, "world_rect", (-1, -1, -1, -1))) != world_rect
            )
            source_changed = (
                item_state is None
                or item_state.source_array_id != source_id
                or item_state.histogram_array_id != hist_id
                or tuple(item_state.local_rect) != local_rect
                or bool(item_state.rgb_already_windowed) != payload_rgb_already_windowed
            )
            dirty = dirty_set is None or int(tile_number) in dirty_set
            if resident_current:
                dirty = False
            levels_changed = item_state is None or not levels_match(item_state.levels, levels)
            is_rgb_tile = self._is_rgb_image(tile_data)
            missing_display = (
                item_state is not None
                and getattr(item_state.item, "image", None) is None
                and is_rgb_tile
            )
            needs_source_rewindow = (
                item_state is not None
                and levels_changed
                and is_rgb_tile
                and not payload_rgb_already_windowed
                and tile_hist is not None
                and (item_state.rgb_base is None or item_state.hist_source is None)
            )
            level_update_admitted = not requested_upserts or int(tile_number) in requested_upserts
            should_upload = bool(
                item_state is None
                or source_changed
                or not identity_current
                or dirty
                or (not item_state.visible and not resident_current)
                or missing_display
            )
            cold_candidate = bool(
                item_state is None
                or item_state.source_array_id == 0
                or source_changed
                or not identity_current
                or dirty
                or (not item_state.visible and not resident_current)
                or missing_display
            )
            rewindow_only = bool(
                existing_item
                and not source_changed
                and not dirty
                and not missing_display
                and needs_source_rewindow
                and level_update_admitted
            )
            item_deadline_ms = level_rewindow_deadline_ms if rewindow_only else cold_deadline_ms
            if level_only_drain:
                # PyQtGraph bakes levels into the payload source identity, so a
                # re-level presents as a full-upload ``cold_candidate`` rather
                # than the lightweight rewindow path.  The resolve pass above
                # already page-assembled and re-windowed exactly the capped
                # upsert slice; letting the cold deadline drop the tail here
                # would discard that paid-for work and force a re-resolve next
                # commit.  The upsert cap already bounds the work, so commit the
                # whole resolved slice.
                item_deadline_ms = None
            if (
                item_deadline_ms is not None
                and cold_candidate
                and cold_tiles_committed > 0
                and (perf_counter() - cold_start) * 1000.0 >= float(item_deadline_ms)
            ):
                if item_state is not None and item_state.visible:
                    active.add(int(tile_number))
                continue

            created_item = False
            if item_state is None:
                item_state = self._pop_direct_reuse_pool() if self._direct_reuse_pool else None
                if item_state is None:
                    item = ImageItem(axisOrder="row-major")
                    item_state = TileLayerItemState(
                        tile_number=int(tile_number),
                        source_index=int(source_index),
                        item=item,
                        local_rect=(-1, -1, -1, -1),
                        world_rect=(-1, -1, -1, -1),
                        source_array_id=0,
                        histogram_array_id=None,
                        levels=levels,
                        rgb_already_windowed=payload_rgb_already_windowed,
                        visible=False,
                    )
                    created_item = True
                    items_created += 1
                else:
                    old_tile = int(item_state.tile_number)
                    if self._states.get(old_tile) is item_state:
                        self._states.pop(old_tile, None)
                    self._unregister_source_state(item_state)
                self._displace_tile_slot_resident(int(tile_number), item_state)
                item_state.tile_number = int(tile_number)
                move_item = getattr(self.layer_owner, "move_tile_item", None)
                if item_state is not None and not created_item and callable(move_item):
                    move_item(old_tile, int(tile_number), item_state.item)
                else:
                    self.layer_owner.add_tile_item(tile_number, item_state.item)
                self._states[int(tile_number)] = item_state

            # A state can remain assigned to its tile while hidden in the
            # direct-reuse pool. Once this transaction selects it for an
            # active slot, remove that warm-residency claim immediately.
            # Otherwise a later tile in the same commit can pop and move the
            # very same ImageItem, leaving lifecycle acknowledgement for two
            # tiles but only the last physical item on screen.
            self._remove_from_direct_reuse_pool(item_state)
            self._touch_resident_state(item_state)
            item_state.item.setPos(float(world_x), float(world_y))
            _apply_item_lod_scale(item_state, scale_x, scale_y)
            if should_upload:
                previous_source_id = item_state.source_array_id
                updated, windowed = self._set_tile_data(
                    item_state,
                    tile_data,
                    tile_hist,
                    levels,
                    source_index=int(source_index),
                    source_array_id=source_id,
                    histogram_array_id=hist_id,
                    local_rect=local_rect,
                    rgb_already_windowed=payload_rgb_already_windowed,
                )
                item_state.world_rect = world_rect
                item_state.acknowledged_identity = (
                    getattr(payload, "tile_identity", None) or payload.source_id
                )
                assembly = page_assemblies.get(int(tile_number), _PageAssembly(payload))
                item_state.page_resolutions = assembly.resolutions
                item_state.page_candidate_missing = assembly.missing
                item_state.physical_lod = getattr(assembly.payload, "lod", None)
                item_state.physical_quality = str(
                    getattr(assembly.payload, "quality", "exact") or "exact"
                )
                item_state.page_fallback_reason = assembly.fallback_reason
                items_updated += int(updated)
                if updated:
                    updated_tiles.append(int(tile_number))
                rgb_window_tiles += int(windowed)
                image_replacements += int(
                    updated
                    and not created_item
                    and not _direct_base_source_matches(previous_source_id, source_id)
                )
                cold_tiles_committed += int(cold_candidate)
                if updated and int(tile_number) in requested_upserts:
                    committed_upserts.add(int(tile_number))
            elif levels_changed and level_update_admitted:
                if (
                    level_rewindow_deadline_ms is not None
                    and level_updates > 0
                    and (perf_counter() - cold_start) * 1000.0 >= float(level_rewindow_deadline_ms)
                ):
                    if item_state is not None and item_state.visible:
                        active.add(int(tile_number))
                    continue
                updated, windowed = self._update_tile_levels(
                    item_state,
                    levels,
                    image=tile_data,
                    histogram_data=tile_hist,
                )
                item_state.world_rect = world_rect
                self._touch_resident_state(item_state)
                level_updates += int(existing_item)
                items_updated += int(updated)
                if updated:
                    updated_tiles.append(int(tile_number))
                rgb_window_tiles += int(windowed)
                if not updated:
                    items_skipped += 1
                if int(tile_number) in requested_upserts:
                    committed_upserts.add(int(tile_number))

            else:
                items_skipped += 1
                if not levels_changed:
                    item_state.levels = levels
                item_state.visible = True
                item_state.source_index = int(source_index)
                item_state.world_rect = world_rect
                self._touch_resident_state(item_state)
                existing_items_shown += 1
                relocated_tiles += int(geometry_changed or reused_source)
                if int(tile_number) in requested_upserts:
                    committed_upserts.add(int(tile_number))

            if acknowledged_identity_satisfies_target(
                item_state.acknowledged_identity,
                target_identities.get(int(tile_number)),
            ):
                item_state.item.setVisible(True)
                item_state.visible = True
                active.add(int(tile_number))
            else:
                self._hide_tile(int(tile_number))

        for tile_number in tuple(self._states):
            if int(tile_number) not in active:
                self._hide_tile(tile_number)
        self._prune_rgb_source_cache()
        storage_evictions += self._prune_resident_items(
            budget_bytes=int(tile_residency_budget_bytes or 0),
            active_tiles=active,
        )
        resident_items = self._resident_count()
        resident_bytes = int(self._resident_bytes)
        physically_presented = tuple(
            sorted(
                int(state.tile_number)
                for state in self._states.values()
                if _state_is_physically_visible(state)
            )
        )
        committed_upserts.intersection_update(physically_presented)

        return TileLayerUpdateStats(
            visible_items=len(physically_presented),
            presented_tiles=physically_presented,
            presented_identities=_direct_presented_identities(self._states, drawable_payloads),
            committed_upserts=tuple(int(tile) for tile in sorted(committed_upserts)),
            identity_rejected_items=len(identity_rejected_tiles),
            identity_rejected_tiles=tuple(identity_rejected_tiles),
            updated_tiles=tuple(int(tile) for tile in updated_tiles),
            items_created=int(items_created),
            items_updated=int(items_updated),
            items_skipped=int(items_skipped),
            rgb_window_tiles=int(rgb_window_tiles),
            image_replacements=int(image_replacements),
            existing_items_shown=int(existing_items_shown),
            relocated_tiles=int(relocated_tiles),
            resident_items=int(resident_items),
            storage_capacity=int(resident_items),
            storage_evictions=int(storage_evictions),
            cpu_shadow_bytes=int(resident_bytes),
            budget_bytes=int(tile_residency_budget_bytes or 0),
            warm_resident_items=max(0, int(resident_items) - len(physically_presented)),
            level_updates=int(level_updates),
            level_update_processed_items=int(level_updates),
            upload_ms=(perf_counter() - update_start) * 1000.0,
            level_update_pending_items=max(0, int(level_update_pending_items) - int(level_updates)),
        )

    def update_levels(
        self,
        levels,
        *,
        image=None,
        histogram_data=None,
    ) -> TileLayerUpdateStats:
        levels = (float(levels[0]), float(levels[1]))
        image_array = None if image is None else np.asarray(image)
        hist_array = None if histogram_data is None else np.asarray(histogram_data)
        items_updated = 0
        items_skipped = 0
        rgb_window_tiles = 0
        update_start = perf_counter()
        processed = 0
        for state in tuple(self._states.values()):
            if not state.visible:
                continue
            updated, windowed = self._update_tile_levels(
                state, levels, image=image_array, histogram_data=hist_array
            )
            processed += 1
            items_updated += int(updated)
            rgb_window_tiles += int(windowed)
            if not updated:
                items_skipped += 1
        self._prune_rgb_source_cache()
        resident_items = self._resident_count()
        physically_presented = tuple(
            sorted(
                int(state.tile_number)
                for state in self._states.values()
                if _state_is_physically_visible(state)
            )
        )
        return TileLayerUpdateStats(
            visible_items=len(physically_presented),
            presented_tiles=physically_presented,
            presented_identities=_direct_presented_identities(self._states),
            items_updated=items_updated,
            items_skipped=items_skipped,
            rgb_window_tiles=rgb_window_tiles,
            level_updates=processed,
            resident_items=int(resident_items),
            storage_capacity=int(resident_items),
            cpu_shadow_bytes=int(self._resident_bytes),
            warm_resident_items=max(0, int(resident_items) - len(physically_presented)),
            level_update_processed_items=processed,
            upload_ms=(perf_counter() - update_start) * 1000.0,
        )

    def warm_payloads(
        self,
        payloads: dict[int, DisplayTilePayload],
        *,
        geometry,
        levels: tuple[float, float],
        rgb_already_windowed: bool,
        tile_residency_budget_bytes: int = 0,
        tile_delta=None,
        frame_plan=None,
    ) -> TileLayerUpdateStats:
        """Prepare non-visible PyQtGraph tile items without committing semantics."""

        if not payloads:
            resident_items = self._resident_count()
            visible_items = sum(
                1 for state in self._states.values() if _state_is_physically_visible(state)
            )
            return TileLayerUpdateStats(
                resident_items=int(resident_items),
                storage_capacity=int(resident_items),
                cpu_shadow_bytes=int(self._resident_bytes),
                budget_bytes=int(tile_residency_budget_bytes or 0),
                warm_resident_items=max(0, int(resident_items) - int(visible_items)),
            )
        layout = tile_layout_map(geometry, frame_plan=frame_plan)
        if not layout:
            return TileLayerUpdateStats()
        start = perf_counter()
        levels = (float(levels[0]), float(levels[1]))
        items_created = 0
        items_updated = 0
        items_skipped = 0
        rgb_window_tiles = 0
        image_replacements = 0
        updated_tiles: list[int] = []
        storage_evictions = 0
        near_source_ids = dict(getattr(tile_delta, "near_tile_source_ids", {}) or {})
        for tile_number in tuple(sorted(int(tile) for tile in payloads)):
            if int(tile_number) in self._states and self._states[int(tile_number)].visible:
                items_skipped += 1
                continue
            region = layout.get(int(tile_number))
            payload = payloads.get(int(tile_number))
            if region is None or not isinstance(payload, DisplayTilePayload):
                items_skipped += 1
                continue
            assembly = _resolve_page_backed_payload(payload, levels=levels)
            payload = assembly.payload
            tile_data = np.asarray(payload.image)
            if tile_data.ndim < 2:
                items_skipped += 1
                continue
            width, height, crop_w, crop_h, scale_x, scale_y = _payload_direct_dims(
                region, tile_data, payload
            )
            if width <= 0 or height <= 0:
                items_skipped += 1
                continue
            if crop_w != int(tile_data.shape[1]) or crop_h != int(tile_data.shape[0]):
                tile_data = tile_data[:crop_h, :crop_w, ...]
            tile_hist = (
                None
                if payload.histogram_data is None
                else np.asarray(payload.histogram_data)[:height, :width]
            )
            base_source_id = near_source_ids.get(int(tile_number), payload.source_id)
            payload_rgb_already_windowed = _payload_rgb_already_windowed(
                payload,
                bool(rgb_already_windowed),
                levels=levels,
            )
            source_id = _direct_payload_source_id(base_source_id, payload)
            hist_id = ("tile-source", source_id) if tile_hist is not None else None
            local_rect = (0, 0, int(crop_w), int(crop_h))
            item_state = self._states.get(int(tile_number))
            if not _direct_state_matches(
                item_state,
                source_id=source_id,
                histogram_id=hist_id,
                local_rect=local_rect,
                rgb_already_windowed=payload_rgb_already_windowed,
            ):
                item_state = self._take_resident_direct_state(
                    int(tile_number),
                    source_id=source_id,
                    histogram_id=hist_id,
                    local_rect=local_rect,
                    rgb_already_windowed=payload_rgb_already_windowed,
                )
            if item_state is None:
                item_state = self._pop_direct_reuse_pool() if self._direct_reuse_pool else None
                created_item = False
                if item_state is None:
                    item_state = TileLayerItemState(
                        tile_number=int(tile_number),
                        source_index=int(
                            getattr(region, "source_index", tile_number) or tile_number
                        ),
                        item=ImageItem(axisOrder="row-major"),
                        local_rect=(-1, -1, -1, -1),
                        world_rect=(-1, -1, -1, -1),
                        source_array_id=0,
                        histogram_array_id=None,
                        levels=levels,
                        rgb_already_windowed=payload_rgb_already_windowed,
                        visible=False,
                    )
                    created_item = True
                    items_created += 1
                else:
                    old_tile = int(item_state.tile_number)
                    if self._states.get(old_tile) is item_state:
                        self._states.pop(old_tile, None)
                    self._unregister_source_state(item_state)
                self._displace_tile_slot_resident(int(tile_number), item_state)
                item_state.tile_number = int(tile_number)
                move_item = getattr(self.layer_owner, "move_tile_item", None)
                if not created_item and callable(move_item):
                    move_item(old_tile, int(tile_number), item_state.item)
                else:
                    self.layer_owner.add_tile_item(int(tile_number), item_state.item)
                self._states[int(tile_number)] = item_state
            matches = _direct_state_matches(
                item_state,
                source_id=source_id,
                histogram_id=hist_id,
                local_rect=local_rect,
                rgb_already_windowed=payload_rgb_already_windowed,
            )
            if not matches or tuple(item_state.levels) != levels:
                previous_source_id = item_state.source_array_id
                updated, windowed = self._set_tile_data(
                    item_state,
                    tile_data,
                    tile_hist,
                    levels,
                    source_index=int(getattr(region, "source_index", tile_number) or tile_number),
                    source_array_id=source_id,
                    histogram_array_id=hist_id,
                    local_rect=local_rect,
                    rgb_already_windowed=payload_rgb_already_windowed,
                )
                items_updated += int(updated)
                rgb_window_tiles += int(windowed)
                image_replacements += int(
                    updated
                    and item_state.source_array_id != 0
                    and not _direct_base_source_matches(previous_source_id, source_id)
                )
                if updated:
                    updated_tiles.append(int(tile_number))
                item_state.acknowledged_identity = (
                    getattr(payload, "tile_identity", None) or payload.source_id
                )
                item_state.page_resolutions = assembly.resolutions
                item_state.page_candidate_missing = assembly.missing
                item_state.physical_lod = getattr(assembly.payload, "lod", None)
                item_state.physical_quality = str(
                    getattr(assembly.payload, "quality", "exact") or "exact"
                )
                item_state.page_fallback_reason = assembly.fallback_reason
            else:
                items_skipped += 1
            item_state.item.setVisible(False)
            item_state.visible = False
            _apply_item_lod_scale(item_state, scale_x, scale_y)
            item_state.world_rect = (
                int(region.x),
                int(region.y),
                int(width),
                int(height),
            )
            # Hidden warm residency is owned by the successor transaction,
            # not by the generic reusable-item pool.  Putting the holder back
            # in that pool lets the next warm batch pop and retarget it while
            # the coordinator still records the old payload as resident.  A
            # 272-tile logical warm can then collapse to one physical holder.
            # The next presentation either claims this exact state or calls
            # ``_hide_tile``/``_displace_tile_slot_resident`` to release it
            # back to ordinary reuse.
            self._remove_from_direct_reuse_pool(item_state)
            self._touch_resident_state(item_state)
        self._prune_rgb_source_cache()
        storage_evictions += self._prune_resident_items(
            budget_bytes=int(tile_residency_budget_bytes or 0),
            active_tiles={int(tile) for tile, state in self._states.items() if state.visible},
        )
        resident_items = self._resident_count()
        physically_presented = tuple(
            sorted(
                int(state.tile_number)
                for state in self._states.values()
                if _state_is_physically_visible(state)
            )
        )
        return TileLayerUpdateStats(
            visible_items=len(physically_presented),
            presented_tiles=physically_presented,
            presented_identities=_direct_presented_identities(self._states, payloads),
            updated_tiles=tuple(updated_tiles),
            items_created=int(items_created),
            items_updated=int(items_updated),
            items_skipped=int(items_skipped),
            rgb_window_tiles=int(rgb_window_tiles),
            image_replacements=int(image_replacements),
            resident_items=int(resident_items),
            storage_capacity=int(resident_items),
            storage_evictions=int(storage_evictions),
            cpu_shadow_bytes=int(self._resident_bytes),
            budget_bytes=int(tile_residency_budget_bytes or 0),
            near_resident_items=len(
                tuple(
                    1
                    for key in dict(getattr(tile_delta, "near_tile_source_ids", {}) or {}).values()
                    if key in self._states_by_source_key
                )
            ),
            warm_resident_items=max(0, int(resident_items) - len(physically_presented)),
            upload_ms=(perf_counter() - start) * 1000.0 if items_updated or items_created else 0.0,
        )

    def payload_resident(self, payload: DisplayTilePayload) -> bool:
        """Report exact physical ImageItem residency without changing state."""

        source_id = _direct_payload_source_id(payload.source_id, payload)
        identity = getattr(payload, "tile_identity", None) or payload.source_id
        backing = getattr(payload, "page_backing", None)
        for state in self._states.values():
            if (
                (
                    state.acknowledged_identity != identity
                    and not acknowledged_identity_satisfies_target(
                        state.acknowledged_identity, identity
                    )
                )
                or getattr(state.item, "image", None) is None
                or self._states.get(int(state.tile_number)) is not state
            ):
                continue
            if backing is None:
                if _direct_physical_payload_source_matches(
                    state.source_array_id,
                    source_id,
                ):
                    return True
                continue
            missing = tuple(
                target
                for target, resolution in zip(
                    backing.requested_keys,
                    backing.candidate_resolutions,
                    strict=True,
                )
                if resolution is None
            )
            if missing:
                if (
                    state.page_candidate_missing == missing
                    and state.page_fallback_reason == "incomplete-page-coverage-native"
                ):
                    return True
                continue
            resolved = backing.resolved_page_set
            if (
                resolved is not None
                and state.page_resolutions == resolved.resolutions
                and not state.page_candidate_missing
            ):
                return True
        return False

    def payload_commit_slot_owned(self, payload: DisplayTilePayload) -> bool:
        """Return whether an onscreen holder owns this tile's atomic swap."""

        state = self._states.get(int(payload.tile_number))
        return bool(state is not None and _state_is_physically_visible(state))

    def _take_resident_direct_state(
        self,
        tile_number: int,
        *,
        source_id: object,
        histogram_id: object | None,
        local_rect: tuple[int, int, int, int],
        rgb_already_windowed: bool,
    ) -> TileLayerItemState | None:
        tile_number = int(tile_number)
        key = _direct_state_key(
            source_id=source_id,
            histogram_id=histogram_id,
            local_rect=local_rect,
            rgb_already_windowed=rgb_already_windowed,
        )
        state = self._states_by_source_key.get(key)
        if state is None:
            return None
        old_tile = int(state.tile_number)
        was_assigned = self._states.get(old_tile) is state
        if was_assigned and old_tile == tile_number:
            return None
        self._remove_from_direct_reuse_pool(state)
        existing = self._states.get(tile_number)
        if existing is not None and existing is not state:
            self._displace_tile_slot_resident(tile_number, state)
        if was_assigned:
            self._states.pop(old_tile, None)
        self._states[tile_number] = state
        move_item = getattr(self.layer_owner, "move_tile_item", None)
        if was_assigned and callable(move_item):
            move_item(old_tile, tile_number, state.item)
        else:
            self.layer_owner.add_tile_item(tile_number, state.item)
        state.tile_number = tile_number
        self._touch_resident_state(state)
        return state

    def _hide_tile(self, tile_number: int) -> None:
        state = self._states.get(int(tile_number))
        if state is None:
            return
        state.item.setVisible(False)
        state.visible = False
        self._add_to_direct_reuse_pool(state)
        self._touch_resident_state(state)

    def _displace_tile_slot_resident(
        self, tile_number: int, replacement: TileLayerItemState
    ) -> None:
        existing = self._states.get(int(tile_number))
        if existing is None or existing is replacement:
            return
        self._states.pop(int(tile_number), None)
        existing.item.setVisible(False)
        existing.visible = False
        unmap = getattr(self.layer_owner, "unmap_tile_item", None)
        if callable(unmap):
            unmap(int(tile_number), existing.item)
        self._add_to_direct_reuse_pool(existing)
        self._touch_resident_state(existing)

    def _remove_tile(self, tile_number: int) -> None:
        state = self._states.pop(int(tile_number), None)
        if state is None:
            return
        self._unregister_source_state(state)
        self._remove_from_direct_reuse_pool(state)
        with contextlib.suppress(Exception):
            self.layer_owner.remove_tile_item(int(tile_number))
        self._resident_bytes -= int(getattr(state, "resident_nbytes", 0) or 0)
        self._rgb_source_cache_bytes -= int(getattr(state, "source_cache_nbytes", 0) or 0)
        state.resident_nbytes = 0
        state.source_cache_nbytes = 0

    def _set_tile_data(
        self,
        state: TileLayerItemState,
        tile_data,
        tile_hist,
        levels: tuple[float, float],
        *,
        source_index: int,
        source_array_id: object,
        histogram_array_id: object | None,
        local_rect: tuple[int, int, int, int],
        rgb_already_windowed: bool,
    ) -> tuple[bool, bool]:
        self._unregister_source_state(state)
        is_rgb = self._is_rgb_image(tile_data)
        windowed = False
        if is_rgb:
            if rgb_already_windowed or tile_hist is None:
                state.rgb_base = None
                state.hist_source = None
                self._refresh_source_cache_nbytes(state)
                display = np.asarray(tile_data)[..., :3]
            else:
                rgb_start = perf_counter()
                base = np.asarray(tile_data)[..., :3].astype(np.float32, copy=False)
                hist = np.asarray(tile_hist, dtype=np.float32)
                display = rgb_display_for_levels(base, hist, levels)
                rgb_ms = (perf_counter() - rgb_start) * 1000.0
                self._record_upload_timing("rgb_window_ms", rgb_ms)
                self._record_upload_timing("tile_layer_rgb_window_ms", rgb_ms)
                state.rgb_base = base
                state.hist_source = hist
                self._touch_rgb_source_cache(state)
                windowed = True
            state.display_cache = display
            upload_start = perf_counter()
            self._set_image_item_data(
                state.item, display, (0, 255), role="visible", emit_histogram_change=False
            )
            self._record_upload_timing(
                "tile_layer_upload_ms", (perf_counter() - upload_start) * 1000.0
            )
        else:
            state.rgb_base = None
            state.hist_source = None
            state.display_cache = None
            self._refresh_source_cache_nbytes(state)
            upload_start = perf_counter()
            self._set_image_item_data(
                state.item,
                tile_data,
                self._histogram_levels_for_display(levels),
                role="visible",
                emit_histogram_change=False,
            )
            self._record_upload_timing(
                "tile_layer_upload_ms", (perf_counter() - upload_start) * 1000.0
            )
        state.source_index = int(source_index)
        state.source_array_id = source_array_id
        state.histogram_array_id = histogram_array_id
        state.local_rect = tuple(int(value) for value in local_rect)
        state.levels = levels
        state.rgb_already_windowed = bool(rgb_already_windowed)
        state.visible = True
        self._touch_resident_state(state)
        self._register_source_state(state)
        self._refresh_resident_nbytes(state)
        return True, windowed

    def _update_tile_levels(
        self,
        state: TileLayerItemState,
        levels: tuple[float, float],
        *,
        image=None,
        histogram_data=None,
    ) -> tuple[bool, bool]:
        if state.rgb_base is None and state.hist_source is None and state.display_cache is not None:
            rebuilt = self._set_tile_data_from_current_source(
                state, levels, image=image, histogram_data=histogram_data
            )
            if rebuilt is not None:
                return rebuilt
        if state.rgb_base is not None and state.hist_source is not None:
            rgb_start = perf_counter()
            display = rgb_display_for_levels(state.rgb_base, state.hist_source, levels)
            rgb_ms = (perf_counter() - rgb_start) * 1000.0
            self._record_upload_timing("rgb_window_ms", rgb_ms)
            self._record_upload_timing("tile_layer_rgb_window_ms", rgb_ms)
            state.display_cache = display
            self._touch_rgb_source_cache(state)
            state.levels = levels
            upload_start = perf_counter()
            self._set_image_item_data(
                state.item, display, (0, 255), role="visible", emit_histogram_change=False
            )
            self._record_upload_timing(
                "tile_layer_upload_ms", (perf_counter() - upload_start) * 1000.0
            )
            return True, True
        if state.display_cache is not None:
            if state.rgb_already_windowed:
                state.item.setLevels((0, 255))
                state.levels = levels
            return False, False
        state.item.setLevels(self._histogram_levels_for_display(levels))
        state.levels = levels
        return False, False

    def _set_tile_data_from_current_source(
        self,
        state: TileLayerItemState,
        levels: tuple[float, float],
        *,
        image,
        histogram_data,
    ) -> tuple[bool, bool] | None:
        if state.rgb_already_windowed or image is None or histogram_data is None:
            return None
        x, y, width, height = tuple(int(value) for value in state.local_rect)
        if width <= 0 or height <= 0:
            return None
        image_array = np.asarray(image)
        hist_array = np.asarray(histogram_data)
        if y < 0 or x < 0 or y + height > image_array.shape[0] or x + width > image_array.shape[1]:
            return None
        if y + height > hist_array.shape[0] or x + width > hist_array.shape[1]:
            return None
        tile_data = image_array[y : y + height, x : x + width, ...]
        if not self._is_rgb_image(tile_data):
            return None
        tile_hist = hist_array[y : y + height, x : x + width]
        return self._set_tile_data(
            state,
            tile_data,
            tile_hist,
            levels,
            source_index=state.source_index,
            source_array_id=state.source_array_id,
            histogram_array_id=state.histogram_array_id,
            local_rect=state.local_rect,
            rgb_already_windowed=False,
        )

    def _touch_rgb_source_cache(self, state: TileLayerItemState) -> None:
        if state.rgb_base is None and state.hist_source is None:
            self._refresh_resident_nbytes(state)
            self._refresh_source_cache_nbytes(state)
            state.source_cache_serial = 0
            return
        self._source_cache_serial += 1
        state.source_cache_serial = int(self._source_cache_serial)
        self._refresh_source_cache_nbytes(state)
        self._refresh_resident_nbytes(state)

    def _touch_resident_state(self, state: TileLayerItemState) -> None:
        self._resident_serial += 1
        state.resident_serial = int(self._resident_serial)

    def _refresh_source_cache_nbytes(self, state: TileLayerItemState) -> None:
        old = int(getattr(state, "source_cache_nbytes", 0) or 0)
        new = _source_cache_nbytes(state)
        if new != old:
            self._rgb_source_cache_bytes += int(new) - int(old)
            state.source_cache_nbytes = int(new)

    def _prune_rgb_source_cache(self) -> None:
        budget = int(self._rgb_source_cache_budget_bytes)
        if budget <= 0:
            for state in self._direct_reuse_pool:
                if state.rgb_base is None and state.hist_source is None:
                    continue
                state.rgb_base = None
                state.hist_source = None
                state.source_cache_serial = 0
                self._refresh_source_cache_nbytes(state)
                self._refresh_resident_nbytes(state)
            return
        if int(self._rgb_source_cache_bytes) <= budget:
            return
        for state in self._direct_reuse_pool:
            if int(self._rgb_source_cache_bytes) <= budget:
                break
            if state.rgb_base is None and state.hist_source is None:
                continue
            state.rgb_base = None
            state.hist_source = None
            state.source_cache_serial = 0
            self._refresh_source_cache_nbytes(state)
            self._refresh_resident_nbytes(state)

    def _register_source_state(self, state: TileLayerItemState) -> None:
        key = _source_key_for_state(state)
        if key is not None:
            self._states_by_source_key[key] = state

    def _unregister_source_state(self, state: TileLayerItemState) -> None:
        key = _source_key_for_state(state)
        if key is not None and self._states_by_source_key.get(key) is state:
            self._states_by_source_key.pop(key, None)

    def _remove_from_direct_reuse_pool(self, state: TileLayerItemState) -> None:
        for index, candidate in enumerate(tuple(self._direct_reuse_pool)):
            if candidate is state:
                self._direct_reuse_pool.pop(index)
                self._direct_reuse_pool_ids.discard(id(state))
                return

    def _add_to_direct_reuse_pool(self, state: TileLayerItemState) -> None:
        state_id = id(state)
        if state_id in self._direct_reuse_pool_ids:
            return
        self._direct_reuse_pool.append(state)
        self._direct_reuse_pool_ids.add(state_id)

    def _pop_direct_reuse_pool(self, index: int = -1) -> TileLayerItemState:
        state = self._direct_reuse_pool.pop(index)
        self._direct_reuse_pool_ids.discard(id(state))
        return state

    def _discard_direct_reuse_pool(self) -> None:
        for state in tuple(self._direct_reuse_pool):
            self._unregister_source_state(state)
            state.visible = False
            state.rgb_base = None
            state.hist_source = None
            state.display_cache = None
            state.source_cache_nbytes = 0
            state.resident_nbytes = 0
        self._direct_reuse_pool.clear()
        self._direct_reuse_pool_ids.clear()
        self._rgb_source_cache_bytes = 0
        self._resident_bytes = sum(
            int(getattr(state, "resident_nbytes", 0) or 0) for state in self._states.values()
        )

    def _resident_count(self) -> int:
        return len(self._states)

    def _refresh_resident_nbytes(self, state: TileLayerItemState) -> None:
        old = int(getattr(state, "resident_nbytes", 0) or 0)
        new = _state_resident_nbytes(state)
        if new != old:
            self._resident_bytes += int(new) - int(old)
            state.resident_nbytes = int(new)

    def _prune_resident_items(self, *, budget_bytes: int, active_tiles: set[int]) -> int:
        if int(budget_bytes or 0) <= 0:
            return 0
        evicted = 0
        while int(self._resident_bytes) > int(budget_bytes) and self._direct_reuse_pool:
            state = self._pop_direct_reuse_pool(0)
            if bool(state.visible) or int(state.tile_number) in active_tiles:
                continue
            self._evict_resident_state(state)
            evicted += 1
        return evicted

    def _evict_resident_state(self, state: TileLayerItemState) -> None:
        self._unregister_source_state(state)
        tile_number = int(state.tile_number)
        if self._states.get(tile_number) is state:
            if state.visible:
                return
            self._states.pop(tile_number, None)
        self._remove_from_direct_reuse_pool(state)
        with contextlib.suppress(Exception):
            self.layer_owner.remove_tile_item(tile_number)
        self._resident_bytes -= int(getattr(state, "resident_nbytes", 0) or 0)
        self._rgb_source_cache_bytes -= int(getattr(state, "source_cache_nbytes", 0) or 0)
        state.resident_nbytes = 0
        state.source_cache_nbytes = 0
        state.visible = False
        state.rgb_base = None
        state.hist_source = None
        state.display_cache = None
        state.source_array_id = 0
        state.acknowledged_identity = None
        state.histogram_array_id = None
        state.page_resolutions = ()
        state.page_candidate_missing = ()
        state.physical_lod = None
        state.physical_quality = None
        state.page_fallback_reason = None


def _source_cache_nbytes(state: TileLayerItemState) -> int:
    total = 0
    if state.rgb_base is not None:
        total += int(getattr(state.rgb_base, "nbytes", 0) or 0)
    if state.hist_source is not None:
        total += int(getattr(state.hist_source, "nbytes", 0) or 0)
    return total


def _state_resident_nbytes(state: TileLayerItemState) -> int:
    total = 0
    image = getattr(state.item, "image", None)
    if image is not None:
        total += int(getattr(np.asarray(image), "nbytes", 0) or 0)
    if state.rgb_base is not None:
        total += int(getattr(state.rgb_base, "nbytes", 0) or 0)
    if state.hist_source is not None:
        total += int(getattr(state.hist_source, "nbytes", 0) or 0)
    if state.display_cache is not None and state.display_cache is not image:
        total += int(getattr(np.asarray(state.display_cache), "nbytes", 0) or 0)
    return max(0, int(total))


def _direct_state_matches(
    state: TileLayerItemState | None,
    *,
    source_id: object,
    histogram_id: object | None,
    local_rect: tuple[int, int, int, int],
    rgb_already_windowed: bool,
) -> bool:
    if state is None:
        return False
    return (
        state.source_array_id == source_id
        and state.histogram_array_id == histogram_id
        and tuple(state.local_rect) == tuple(int(value) for value in local_rect)
        and bool(state.rgb_already_windowed) == bool(rgb_already_windowed)
    )


def _source_key_for_state(state: TileLayerItemState) -> object | None:
    if state.source_array_id == 0:
        return None
    return _direct_state_key(
        source_id=state.source_array_id,
        histogram_id=state.histogram_array_id,
        local_rect=state.local_rect,
        rgb_already_windowed=state.rgb_already_windowed,
    )


def _direct_presented_identities(
    states: Mapping[int, TileLayerItemState],
    payloads: Mapping[int, DisplayTilePayload] | None = None,
) -> dict[int, object]:
    payloads = dict(payloads or {})
    identities: dict[int, object] = {}
    for state in tuple(dict(states).values()):
        if not _state_is_physically_visible(state) or state.source_array_id == 0:
            continue
        tile_number = int(state.tile_number)
        payload = payloads.get(tile_number)
        if payload is not None:
            expected = _direct_payload_source_id(payload.source_id, payload)
            if state.source_array_id == expected:
                identities[tile_number] = state.acknowledged_identity or payload.source_id
                continue
        identities[tile_number] = state.acknowledged_identity or state.source_array_id
    return identities


def _state_is_physically_visible(state: TileLayerItemState) -> bool:
    return bool(
        getattr(state, "visible", False)
        and getattr(state, "item", None) is not None
        and state.item.isVisible()
    )


def _direct_state_key(
    *,
    source_id: object,
    histogram_id: object | None,
    local_rect: tuple[int, int, int, int],
    rgb_already_windowed: bool,
) -> tuple[object, object | None, tuple[int, int, int, int], bool]:
    return (
        source_id,
        histogram_id,
        tuple(int(value) for value in local_rect),
        bool(rgb_already_windowed),
    )


def _direct_base_source_id(source_id: object) -> object:
    if isinstance(source_id, tuple) and len(source_id) >= 2 and source_id[1] == "pyqtgraph_display":
        return _direct_payload_semantic_source_token(source_id)
    return _canonical_direct_base_source_id(source_id)


def _canonical_direct_base_source_id(base_source_id: object) -> object:
    if isinstance(base_source_id, tuple) and "texture_kind" in base_source_id:
        base_source_id = base_source_id[: base_source_id.index("texture_kind")]
    if (
        isinstance(base_source_id, tuple)
        and len(base_source_id) >= 3
        and base_source_id[0] == "montage-tile"
    ):
        return ("montage-tile", base_source_id[1], int(base_source_id[2]))
    if (
        isinstance(base_source_id, tuple)
        and len(base_source_id) >= 8
        and base_source_id[0] == "display_tile"
    ):
        request_key = base_source_id[5]
        slice_indices = _display_tile_slice_indices(request_key)
        if slice_indices:
            return (
                "display-tile-semantic",
                base_source_id[1],
                tuple(int(value) for value in slice_indices),
            )
    if (
        isinstance(base_source_id, tuple)
        and len(base_source_id) >= 2
        and base_source_id[1] == "pyqtgraph_display"
    ):
        return _direct_base_source_id(base_source_id)
    return base_source_id


def _direct_base_source_matches(left: object, right: object) -> bool:
    return _direct_base_source_id(left) == _direct_base_source_id(right)


def _direct_physical_payload_source_matches(left: object, right: object) -> bool:
    """Compare exact direct payload storage while ignoring route-key spelling."""

    if left == right:
        return True
    return bool(
        isinstance(left, tuple)
        and isinstance(right, tuple)
        and len(left) > 1
        and len(right) > 1
        and left[1:] == right[1:]
    )


def _direct_payload_semantic_source_token(source_id: tuple[object, ...]) -> object:
    try:
        rgb_marker = source_id.index("rgb_windowed_levels")
    except ValueError:
        rgb_marker = len(source_id)
    # The base key spelling can legitimately change between cache/evaluator
    # and session-owned identities.  The semantic payload planes are the
    # ownership boundary for PyQtGraph base replacement accounting.
    return ("pyqtgraph-semantic-payload", *source_id[2:rgb_marker])


def _display_tile_slice_indices(request_key: object) -> tuple[int, ...] | None:
    if not (isinstance(request_key, tuple) and len(request_key) >= 4 and request_key[0] == "image"):
        return None
    for part in reversed(request_key):
        if not isinstance(part, tuple) or not part:
            continue
        try:
            return tuple(int(value) for value in part)
        except (TypeError, ValueError):
            continue
    return None


def _direct_payload_source_id(
    base_source_id: object, payload: DisplayTilePayload
) -> tuple[object, ...]:
    image = payload.semantic_data if payload.semantic_data is not None else payload.image
    histogram = (
        payload.semantic_histogram_data
        if payload.semantic_histogram_data is not None
        else payload.histogram_data
    )
    texture_kind = getattr(payload, "texture_kind", None)
    return (
        base_source_id,
        "pyqtgraph_display",
        tuple(int(value) for value in np.shape(image)),
        np.asarray(image).dtype,
        id(image),
        None if histogram is None else tuple(int(value) for value in histogram.shape),
        None if histogram is None else np.dtype(histogram.dtype),
        None if histogram is None else id(histogram),
        "texture_kind",
        None if texture_kind is None else getattr(texture_kind, "value", texture_kind),
        "rgb_windowed_levels",
        getattr(payload, "rgb_windowed_levels", None),
        "page_bindings",
        _page_binding_source_token(payload),
    )


def _page_binding_source_token(payload: DisplayTilePayload) -> object | None:
    """Canonical CPU residency identity for PyQtGraph item reuse."""

    backing = getattr(payload, "page_backing", None)
    if backing is None:
        return None
    return tuple(
        (
            target,
            None if resolution is None else resolution.actual_key,
        )
        for target, resolution in zip(
            backing.requested_keys,
            backing.candidate_resolutions,
            strict=True,
        )
    )


def _direct_tile_order(
    layout: dict[int, object],
    tile_payloads: dict[int, DisplayTilePayload],
    tile_delta,
    state_map: dict[int, TileLayerItemState] | None = None,
    *,
    tile_states: tuple[object, ...] = (),
    tile_source_ids: dict[int, object] | None = None,
    rgb_already_windowed: bool = False,
) -> tuple[int, ...]:
    if tile_delta is None:
        return tuple(sorted(int(tile) for tile in layout))
    candidates: list[int] = []
    candidates.extend(int(tile) for tile in tuple(getattr(tile_delta, "removals", ()) or ()))
    candidates.extend(int(tile) for tile in tuple(getattr(tile_delta, "upserts", ()) or ()))
    active_tiles = tuple(int(tile) for tile in tuple(getattr(tile_delta, "active_tiles", ()) or ()))
    if bool(getattr(tile_delta, "force_refresh", False)):
        candidates.extend(active_tiles)
    else:
        state_map = state_map or {}
        candidates.extend(
            int(tile)
            for tile in active_tiles
            if int(tile) in tile_payloads
            and (
                int(tile) not in state_map
                or not bool(getattr(state_map[int(tile)], "visible", False))
                or _direct_tile_geometry_changed(
                    state_map[int(tile)], layout, int(tile), tile_payloads[int(tile)]
                )
                or _direct_tile_binding_stale(
                    state_map.get(int(tile)),
                    layout,
                    int(tile),
                    tile_payloads[int(tile)],
                    tile_states=tile_states,
                    tile_source_ids=tile_source_ids,
                    rgb_already_windowed=_payload_rgb_already_windowed(
                        tile_payloads[int(tile)],
                        bool(rgb_already_windowed),
                    ),
                )
            )
        )
    seen: set[int] = set()
    ordered: list[int] = []
    for tile in candidates:
        if int(tile) < 0 or int(tile) in seen:
            continue
        seen.add(int(tile))
        ordered.append(int(tile))
    return tuple(ordered)


def _direct_tile_binding_stale(
    state: TileLayerItemState | None,
    layout: dict[int, object],
    tile_number: int,
    payload: DisplayTilePayload,
    *,
    tile_states: tuple[object, ...],
    tile_source_ids: dict[int, object] | None,
    rgb_already_windowed: bool,
) -> bool:
    tile_number = int(tile_number)
    state_value = "loaded"
    if tile_states and tile_number < len(tile_states):
        state_value = str(getattr(tile_states[tile_number], "value", tile_states[tile_number]))
    if state_value == "skipped":
        return False
    region = layout.get(tile_number)
    if region is None:
        return False
    tile_data = np.asarray(payload.image)
    if tile_data.ndim < 2:
        return False
    width, height, crop_w, crop_h, _scale_x, _scale_y = _payload_direct_dims(
        region, tile_data, payload
    )
    if width <= 0 or height <= 0:
        return False
    base_source_id = (
        tile_source_ids.get(tile_number, payload.source_id)
        if tile_source_ids is not None
        else payload.source_id
    )
    source_id = _direct_payload_source_id(base_source_id, payload)
    histogram_id = ("tile-source", source_id) if payload.histogram_data is not None else None
    return not _direct_state_matches(
        state,
        source_id=source_id,
        histogram_id=histogram_id,
        local_rect=(0, 0, int(crop_w), int(crop_h)),
        rgb_already_windowed=bool(rgb_already_windowed),
    )


def _direct_preclaim_specs(
    layout: dict[int, object],
    tile_order: tuple[int, ...],
    tile_payloads: dict[int, DisplayTilePayload],
    *,
    states: tuple[object, ...],
    tile_source_ids: dict[int, object] | None,
) -> dict[int, tuple[object, object | None, tuple[int, int, int, int], bool]]:
    specs: dict[int, tuple[object, object | None, tuple[int, int, int, int], bool]] = {}
    for tile_number in tile_order:
        tile_number = int(tile_number)
        region = layout.get(tile_number)
        if region is None:
            continue
        state_value = "loaded"
        if states and tile_number < len(states):
            state_value = str(getattr(states[tile_number], "value", states[tile_number]))
        payload = None if state_value == "skipped" else tile_payloads.get(tile_number)
        if not isinstance(payload, DisplayTilePayload):
            continue
        tile_data = np.asarray(payload.image)
        if tile_data.ndim < 2:
            continue
        width, height, crop_w, crop_h, _scale_x, _scale_y = _payload_direct_dims(
            region, tile_data, payload
        )
        if width <= 0 or height <= 0:
            continue
        base_source_id = (
            tile_source_ids.get(tile_number, payload.source_id)
            if tile_source_ids is not None
            else payload.source_id
        )
        source_id = _direct_payload_source_id(base_source_id, payload)
        histogram_id = ("tile-source", source_id) if payload.histogram_data is not None else None
        specs[tile_number] = (
            source_id,
            histogram_id,
            (0, 0, int(crop_w), int(crop_h)),
            _payload_rgb_already_windowed(payload, False),
        )
    return specs


def _direct_cold_hole_count(
    specs: dict[int, tuple[object, object | None, tuple[int, int, int, int], bool]],
    states_by_source_key: dict[object, TileLayerItemState],
) -> int:
    cold = 0
    for source_id, histogram_id, local_rect, rgb_already_windowed in specs.values():
        key = _direct_state_key(
            source_id=source_id,
            histogram_id=histogram_id,
            local_rect=local_rect,
            rgb_already_windowed=bool(rgb_already_windowed),
        )
        if key not in states_by_source_key:
            cold += 1
    return cold


def _direct_tile_geometry_changed(
    state: TileLayerItemState,
    layout: dict[int, object],
    tile_number: int,
    payload: DisplayTilePayload,
) -> bool:
    data = np.asarray(payload.image)
    if data.ndim < 2:
        return False
    region = layout.get(int(tile_number))
    if region is None:
        return False
    width, height, crop_w, crop_h, _scale_x, _scale_y = _payload_direct_dims(region, data, payload)
    if width <= 0 or height <= 0:
        return False
    expected_world = (int(region.x), int(region.y), int(width), int(height))
    expected_local = (0, 0, int(crop_w), int(crop_h))
    return (
        tuple(getattr(state, "local_rect", (-1, -1, -1, -1))) != expected_local
        or tuple(getattr(state, "world_rect", (-1, -1, -1, -1))) != expected_world
    )
