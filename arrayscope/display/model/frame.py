"""Committed visible display frame state and value sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.lod import LodInfo
from arrayscope.display.scene import DisplayScene, display_scene_for_geometry
from arrayscope.display.shader_mapping import ShaderMapping, TexturePlaneKind
from arrayscope.display.model.tile_identity import TileIdentity, TilePresentationIdentity
from arrayscope.display.pyramid import LodPagePlan, MaterializedLodPage


def array_value_at(data, y_i: int, x_i: int):
    value = np.asarray(data)[int(y_i), int(x_i)]
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if np.isscalar(value):
        try:
            return value.item()
        except AttributeError:
            return value
    return value


@dataclass(frozen=True)
class PayloadSourceAnchor:
    """Window-invariant content identity of a payload's plane (ADR 0055 G3).

    ``content_key`` identifies the evaluated-value space the plane samples
    (document revision + operation steps + window-free view identity);
    ``source_rect`` is the plane's ``(y0, y1, x0, x1)`` in that space at
    native resolution. A backend may key sub-plane residency on
    ``(content_key, chunk rect, lod, texture kind)`` — equal keys mean equal
    texels regardless of the display window that produced the payload.
    """

    content_key: object
    source_rect: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_rect", tuple(int(value) for value in self.source_rect)
        )


@dataclass(frozen=True)
class PageBackedPresentation:
    """Backend-neutral logical pages and supplied CPU residency values."""

    requested_plans: tuple[LodPagePlan, ...]
    materialized_pages: tuple[MaterializedLodPage, ...]
    source_coverage_yx: tuple[int, int, int, int]
    requested_lod: LodInfo

    def __post_init__(self) -> None:
        plans = tuple(self.requested_plans)
        pages = tuple(self.materialized_pages)
        if not plans:
            raise ValueError("page-backed presentation requires at least one target plan")
        if any(not isinstance(plan, LodPagePlan) for plan in plans):
            raise TypeError("page-backed targets must be LodPagePlan instances")
        if any(not isinstance(page, MaterializedLodPage) for page in pages):
            raise TypeError("page-backed supplied values must be MaterializedLodPage instances")
        keys = tuple(plan.key for plan in plans)
        if len(set(keys)) != len(keys):
            raise ValueError("page-backed presentation has duplicate targets")
        page_keys = tuple(page.key for page in pages)
        if len(set(page_keys)) != len(page_keys):
            raise ValueError("page-backed presentation has duplicate materialized pages")
        unknown = tuple(key for key in page_keys if key not in set(keys))
        if unknown:
            raise ValueError(f"materialized pages do not belong to requested targets: {unknown!r}")
        coverage = tuple(int(value) for value in self.source_coverage_yx)
        if len(coverage) != 4 or coverage[0] < 0 or coverage[2] < 0 or coverage[1] <= coverage[0] or coverage[3] <= coverage[2]:
            raise ValueError("page-backed source coverage must be a non-empty native-source rectangle")
        _validate_exact_rect_cover(
            coverage,
            tuple(plan.valid_source_rect_yx for plan in plans),
        )
        if not isinstance(self.requested_lod, LodInfo):
            raise TypeError("page-backed requested_lod must be LodInfo")
        requested_reductions = {tuple(plan.reduction_yx) for plan in plans}
        if len(requested_reductions) != 1:
            raise ValueError("page-backed targets must share one requested reduction")
        reduction_y, reduction_x = next(iter(requested_reductions))
        if int(self.requested_lod.factor) != 1 << max(reduction_y, reduction_x):
            raise ValueError("requested semantic LOD disagrees with target page reduction")
        object.__setattr__(self, "requested_plans", plans)
        object.__setattr__(self, "materialized_pages", pages)
        object.__setattr__(self, "source_coverage_yx", coverage)

    @property
    def requested_keys(self) -> tuple[object, ...]:
        return tuple(plan.key for plan in self.requested_plans)

    def materialized_by_key(self) -> dict[object, MaterializedLodPage]:
        return {page.key: page for page in self.materialized_pages}

    def value_at_native(self, y_i: int, x_i: int):
        """Map native coordinates through exact clipped-bin geometry."""

        y_i, x_i = int(y_i), int(x_i)
        pages = self.materialized_by_key()
        for plan in self.requested_plans:
            y0, y1, x0, x1 = plan.valid_source_rect_yx
            if not (y0 <= y_i < y1 and x0 <= x_i < x1):
                continue
            page = pages.get(plan.key)
            if page is None:
                return None
            for index, (sy0, sy1, sx0, sx1) in enumerate(plan.sample_source_rects_yx):
                if sy0 <= y_i < sy1 and sx0 <= x_i < sx1:
                    row, column = divmod(index, plan.stored_shape[1])
                    return array_value_at(page.values, row, column)
            raise RuntimeError("planned page source coverage has no containing stored sample")
        return None


@dataclass(frozen=True)
class DisplayTilePayload:
    tile_number: int
    source_index: int
    image: np.ndarray
    histogram_data: np.ndarray | None
    source_id: object
    texture_data: np.ndarray | None = None
    texture_kind: TexturePlaneKind | None = None
    semantic_data: np.ndarray | None = None
    semantic_histogram_data: np.ndarray | None = None
    source_shape: tuple[int, int] | None = None
    lod: LodInfo | None = None
    shader_mapping: ShaderMapping | None = None
    level_data: np.ndarray | None = None
    level_stats: object | None = None
    rgb_windowed_levels: tuple[float, float] | None = None
    quality: str = "exact"
    tile_identity: TileIdentity | None = None
    presentation_identity: TilePresentationIdentity | None = None
    # ADR 0055 G3: optional window-invariant anchor; lets a backend keep
    # sub-plane residency across display-window shifts. Never part of tile
    # semantic identity.
    source_anchor: PayloadSourceAnchor | None = None
    # ADR 0056 G5: logical page targets and checked materialized values.
    # Tile/presentation identity remains separate from every page key.
    page_backing: PageBackedPresentation | None = None

    def __post_init__(self) -> None:
        quality = str(self.quality or "exact")
        if quality not in ("exact", "preview"):
            raise ValueError(f"display tile payload quality must be 'exact' or 'preview', got {quality!r}")
        object.__setattr__(self, "quality", quality)
        image = np.asarray(self.image)
        if image.ndim < 2:
            raise ValueError("display tile payload image must be at least 2D")
        texture = image if self.texture_data is None else np.asarray(self.texture_data)
        if texture.ndim < 2:
            raise ValueError("display tile payload texture data must be at least 2D")
        if self.histogram_data is not None:
            histogram = np.asarray(self.histogram_data)
            if tuple(histogram.shape[:2]) != tuple(image.shape[:2]):
                raise ValueError("display tile payload histogram shape must match image shape")
        if quality == "preview":
            if self.semantic_data is not None or self.semantic_histogram_data is not None:
                raise ValueError("preview display tile payloads must not carry exact semantic planes")
            semantic = None
            semantic_histogram = None
        else:
            semantic = (
                None
                if self.semantic_data is None and self.page_backing is not None
                else (image if self.semantic_data is None else np.asarray(self.semantic_data))
            )
            semantic_histogram = (
                None
                if self.semantic_histogram_data is None and self.page_backing is not None
                else (
                    self.histogram_data
                    if self.semantic_histogram_data is None
                    else self.semantic_histogram_data
                )
            )
            semantic_histogram = None if semantic_histogram is None else np.asarray(semantic_histogram)
        source_shape = tuple(int(value) for value in (self.source_shape or image.shape[:2])[:2])
        texture_kind = self.texture_kind
        if texture_kind is None:
            if texture.ndim == 3 and texture.shape[-1] in (3, 4):
                texture_kind = TexturePlaneKind.RGB8
            elif np.iscomplexobj(texture) or (texture.ndim == 3 and texture.shape[-1] == 2):
                texture_kind = TexturePlaneKind.COMPLEX_RG32F
            else:
                texture_kind = TexturePlaneKind.SCALAR_R32F
        elif not isinstance(texture_kind, TexturePlaneKind):
            texture_kind = TexturePlaneKind(getattr(texture_kind, "value", texture_kind))
        object.__setattr__(self, "tile_number", int(self.tile_number))
        object.__setattr__(self, "source_index", int(self.source_index))
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "texture_data", texture)
        object.__setattr__(self, "texture_kind", texture_kind)
        object.__setattr__(self, "semantic_data", semantic)
        object.__setattr__(self, "semantic_histogram_data", semantic_histogram)
        object.__setattr__(self, "source_shape", source_shape)
        if self.histogram_data is not None:
            object.__setattr__(self, "histogram_data", np.asarray(self.histogram_data))
        if self.level_data is not None:
            object.__setattr__(self, "level_data", np.asarray(self.level_data))
        if self.rgb_windowed_levels is not None:
            try:
                low, high = self.rgb_windowed_levels
                object.__setattr__(self, "rgb_windowed_levels", (float(low), float(high)))
            except Exception as exc:
                raise ValueError("rgb_windowed_levels must be a 2-tuple of finite levels") from exc
        page_backing = self.page_backing
        if page_backing is not None:
            if not isinstance(page_backing, PageBackedPresentation):
                raise TypeError("display tile page_backing must be PageBackedPresentation")
            if self.lod is None:
                raise ValueError("page-backed payload requires requested semantic LOD")
            if page_backing.requested_lod != self.lod:
                raise ValueError("payload LOD disagrees with page-backed requested LOD")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.shape(self.image))

    @property
    def dtype(self) -> np.dtype:
        return np.asarray(self.image).dtype

    @property
    def nbytes(self) -> int:
        arrays = [
            self.texture_data if self.texture_data is not None else self.image,
            self.histogram_data,
            self.semantic_data,
            self.semantic_histogram_data,
            self.level_data,
        ]
        if self.page_backing is not None:
            arrays.extend(page.values for page in self.page_backing.materialized_pages)
        seen: set[int] = set()
        total = 0
        for value in arrays:
            if value is None or id(value) in seen:
                continue
            seen.add(id(value))
            total += int(np.asarray(value).nbytes)
        return total


def display_tile_payload_has_semantics(payload) -> bool:
    """Return whether a tiled payload can update committed semantic state."""

    return str(getattr(payload, "quality", "exact") or "exact") == "exact"


def _validate_exact_rect_cover(
    coverage: tuple[int, int, int, int],
    rectangles: tuple[tuple[int, int, int, int], ...],
) -> None:
    """Reject page target gaps/overlaps without allocating a source-sized mask."""

    y_edges = sorted({coverage[0], coverage[1], *(edge for rect in rectangles for edge in rect[:2])})
    x_edges = sorted({coverage[2], coverage[3], *(edge for rect in rectangles for edge in rect[2:])})
    for y0, y1 in zip(y_edges, y_edges[1:]):
        for x0, x1 in zip(x_edges, x_edges[1:]):
            if not (
                coverage[0] <= y0 < y1 <= coverage[1]
                and coverage[2] <= x0 < x1 <= coverage[3]
            ):
                continue
            owners = sum(
                int(ry0 <= y0 and y1 <= ry1 and rx0 <= x0 and x1 <= rx1)
                for ry0, ry1, rx0, rx1 in rectangles
            )
            if owners != 1:
                reason = "gap" if owners == 0 else "overlap"
                raise ValueError(f"page-backed target cover has a {reason} at {(y0, y1, x0, x1)}")


@dataclass(frozen=True)
class TileCommitReport:
    """Backend acknowledgement for a tiled presentation commit.

    Counts distinguish expensive cold work from cheap resident rebinds or
    geometry-only changes, so scheduling feedback does not throttle already
    resident tiles as though they all required uploads.
    """

    presented_tiles: frozenset[int] = field(default_factory=frozenset)
    committed_upserts: frozenset[int] | None = None
    # Upserts the backend refused because their typed identity cannot satisfy
    # the delta's target identity.  Loud by contract: re-emitting the same
    # rejected payloads never converges (field stall 2026-07-16).
    identity_rejected_tiles: frozenset[int] = field(default_factory=frozenset)
    removed_tiles: frozenset[int] = field(default_factory=frozenset)
    texture_uploads: int = 0
    texture_upload_bytes: int = 0
    pyqtgraph_items_created: int = 0
    cpu_windowed_tiles: int = 0
    resident_rebinds: int = 0
    existing_items_shown: int = 0
    relocated_tiles: int = 0
    storage_rebuilds: int = 0
    vertex_uploads: int = 0
    cold_work_ms: float = 0.0
    visibility_work_ms: float = 0.0
    total_ms: float = 0.0
    stale: bool = False
    clear_reason: str = ""
    # Causal binding to the delta this report acknowledges: (base_revision,
    # target_revision) of the committed TilePresentationDelta.  Skipped or
    # superseded commits leave the committer's last report pointing at an
    # OLDER delta; acknowledging a new delta against it invents acceptance
    # (ADR 0051 rule 1; field defect 2026-07-05, JSONL 112841).  None means
    # unbound (legacy constructions/tests): the causality check is skipped.
    delta_key: tuple[int, int] | None = None
    # Presentation revisions are local to one FrameSession and can repeat
    # after a retarget/rebirth. Bind acknowledgement to that session too.
    transaction_generation: int | None = None
    # Ground truth from the backend: the payload identity each drawn tile
    # ACTUALLY holds after this commit (tile -> source_id).  The session
    # converges its presentation against this map (ADR 0051 rule 1); its own
    # acknowledgement records repeatedly proved capable of lying.  None =
    # backend does not report identities.
    presented_identities: Mapping[int, object] | None = None

    def __post_init__(self) -> None:
        if self.delta_key is not None:
            object.__setattr__(
                self, "delta_key", (int(self.delta_key[0]), int(self.delta_key[1]))
            )
        if self.transaction_generation is not None:
            object.__setattr__(self, "transaction_generation", int(self.transaction_generation))
        if self.presented_identities is not None:
            object.__setattr__(
                self,
                "presented_identities",
                {int(tile): identity for tile, identity in dict(self.presented_identities).items()},
            )
        object.__setattr__(self, "presented_tiles", frozenset(int(tile) for tile in self.presented_tiles))
        if self.committed_upserts is not None:
            object.__setattr__(
                self,
                "committed_upserts",
                frozenset(int(tile) for tile in self.committed_upserts),
            )
        object.__setattr__(self, "removed_tiles", frozenset(int(tile) for tile in self.removed_tiles))
        for name in (
            "texture_uploads",
            "texture_upload_bytes",
            "pyqtgraph_items_created",
            "cpu_windowed_tiles",
            "resident_rebinds",
            "existing_items_shown",
            "relocated_tiles",
            "storage_rebuilds",
        ):
            object.__setattr__(self, name, max(0, int(getattr(self, name))))
        for name in ("cold_work_ms", "visibility_work_ms", "total_ms"):
            object.__setattr__(self, name, max(0.0, float(getattr(self, name) or 0.0)))
        object.__setattr__(self, "stale", bool(self.stale))
        object.__setattr__(self, "clear_reason", str(self.clear_reason or ""))

    @property
    def cold_count(self) -> int:
        return int(self.texture_uploads + self.pyqtgraph_items_created + self.cpu_windowed_tiles)

    def acknowledges(self, delta: "TilePresentationDelta") -> bool:
        """Whether this report was produced by committing exactly ``delta``."""

        if (
            self.transaction_generation is not None
            and delta.transaction_generation is not None
            and int(self.transaction_generation) != int(delta.transaction_generation)
        ):
            return False
        if self.delta_key is None:
            return True
        return self.delta_key == (int(delta.base_revision), int(delta.target_revision))

    def accepted_upserts(self, delta: "TilePresentationDelta") -> set[int]:
        if not self.acknowledges(delta):
            return set()
        if self.committed_upserts is not None:
            return set(self.committed_upserts.intersection(delta.upserts))
        return set(self.presented_tiles.intersection(delta.upserts))

    def accepted_upserts_in_order(self, delta: "TilePresentationDelta") -> tuple[int, ...]:
        """Accepted membership in the producer's presentation order."""

        accepted = self.accepted_upserts(delta)
        return tuple(int(tile) for tile in delta.upserts if int(tile) in accepted)


@dataclass(frozen=True)
class TilePresentationDelta:
    structure_revision: int
    payload_revision: int
    visibility_revision: int
    level_revision: int
    histogram_revision: int
    viewport_revision: int
    base_revision: int = 0
    target_revision: int = 0
    transaction_generation: int | None = None
    cold_deadline_ms: float | None = None
    upserts: Mapping[int, DisplayTilePayload] = field(default_factory=dict)
    removals: tuple[int, ...] = ()
    active_tiles: tuple[int, ...] = ()
    planned_tiles: tuple[int, ...] = ()
    near_tiles: tuple[int, ...] = ()
    near_tile_source_ids: Mapping[int, object] = field(default_factory=dict)
    target_identities: Mapping[int, TileIdentity] = field(default_factory=dict)
    force_refresh: bool = False
    clear_reason: str = ""

    def __post_init__(self) -> None:
        upserts = {int(key): _coerce_tile_payload(value) for key, value in dict(self.upserts).items()}
        for key, payload in upserts.items():
            if int(payload.tile_number) != int(key):
                raise ValueError("tile delta upsert key must match payload tile_number")
        removals = _unique_int_tuple(self.removals, "removals")
        if set(removals).intersection(upserts):
            raise ValueError("tile delta cannot remove and upsert the same tile")
        active = _unique_int_tuple(self.active_tiles, "active_tiles")
        planned = _unique_int_tuple(self.planned_tiles, "planned_tiles")
        near = _unique_int_tuple(self.near_tiles, "near_tiles")
        near_sources = {int(key): value for key, value in dict(self.near_tile_source_ids or {}).items()}
        target_identities = {
            int(key): value for key, value in dict(self.target_identities or {}).items()
        }
        if any(not isinstance(value, TileIdentity) for value in target_identities.values()):
            raise TypeError("tile delta target identities must be TileIdentity instances")
        object.__setattr__(self, "structure_revision", int(self.structure_revision))
        object.__setattr__(self, "payload_revision", int(self.payload_revision))
        object.__setattr__(self, "visibility_revision", int(self.visibility_revision))
        object.__setattr__(self, "level_revision", int(self.level_revision))
        object.__setattr__(self, "histogram_revision", int(self.histogram_revision))
        object.__setattr__(self, "viewport_revision", int(self.viewport_revision))
        object.__setattr__(self, "base_revision", int(self.base_revision))
        target = int(self.target_revision) if int(self.target_revision) else int(self.base_revision) + (1 if upserts or removals else 0)
        object.__setattr__(self, "target_revision", target)
        if self.transaction_generation is not None:
            object.__setattr__(self, "transaction_generation", int(self.transaction_generation))
        deadline = self.cold_deadline_ms
        object.__setattr__(self, "cold_deadline_ms", None if deadline is None else max(0.0, float(deadline)))
        object.__setattr__(self, "upserts", upserts)
        object.__setattr__(self, "removals", removals)
        object.__setattr__(self, "active_tiles", active)
        object.__setattr__(self, "planned_tiles", planned)
        object.__setattr__(self, "near_tiles", near)
        object.__setattr__(self, "near_tile_source_ids", near_sources)
        object.__setattr__(self, "target_identities", target_identities)
        object.__setattr__(self, "force_refresh", bool(self.force_refresh))
        object.__setattr__(self, "clear_reason", str(self.clear_reason or ""))


@dataclass(frozen=True)
class TilePresentationState:
    payloads: Mapping[int, DisplayTilePayload] = field(default_factory=dict)
    revision: int = 0

    def __post_init__(self) -> None:
        typed = {int(key): _coerce_tile_payload(value) for key, value in dict(self.payloads).items()}
        for key, payload in typed.items():
            if int(payload.tile_number) != int(key):
                raise ValueError("tile state payload key must match tile_number")
        object.__setattr__(self, "payloads", typed)
        object.__setattr__(self, "revision", int(self.revision))

    def apply_delta(self, delta: TilePresentationDelta) -> "TilePresentationState":
        if not isinstance(delta, TilePresentationDelta):
            raise TypeError("tile presentation state requires a TilePresentationDelta")
        if int(delta.base_revision) != int(self.revision):
            return self
        payloads = dict(self.payloads)
        for tile_number in delta.removals:
            payloads.pop(int(tile_number), None)
        payloads.update(delta.upserts)
        return TilePresentationState(payloads, revision=int(delta.target_revision))

    def acknowledge_delta(self, delta: TilePresentationDelta, report: TileCommitReport) -> "TilePresentationState":
        """Apply only the portion of a proposed delta accepted by the backend."""

        if not isinstance(report, TileCommitReport):
            raise TypeError("tile presentation acknowledgement requires a TileCommitReport")
        if bool(report.stale) or not report.acknowledges(delta) or int(delta.base_revision) != int(self.revision):
            return self
        accepted_upserts = {
            int(tile): payload
            for tile, payload in dict(delta.upserts).items()
            if int(tile) in report.accepted_upserts(delta)
        }
        removals = set(int(tile) for tile in report.removed_tiles)
        if not accepted_upserts and not removals:
            return self
        acknowledged = replace(
            delta,
            upserts=accepted_upserts,
            removals=tuple(sorted(removals)),
            target_revision=int(delta.target_revision),
        )
        return self.apply_delta(acknowledged)

    def active_payloads(self, delta: TilePresentationDelta) -> dict[int, DisplayTilePayload]:
        return {int(tile): self.payloads[int(tile)] for tile in delta.active_tiles if int(tile) in self.payloads}

    def near_payloads(self, delta: TilePresentationDelta) -> dict[int, DisplayTilePayload]:
        return {int(tile): self.payloads[int(tile)] for tile in delta.near_tiles if int(tile) in self.payloads}


class FrameValueSource:
    def value_at(self, mapping):
        raise NotImplementedError

    def tile_region(self, tile, region: tuple[slice, slice]):
        raise NotImplementedError


@dataclass(frozen=True)
class TiledValueSource(FrameValueSource):
    payloads: dict[int, DisplayTilePayload] = field(default_factory=dict)

    def __post_init__(self) -> None:
        typed = {int(key): _coerce_tile_payload(value) for key, value in dict(self.payloads).items()}
        for key, payload in typed.items():
            if int(payload.tile_number) != int(key):
                raise ValueError("display tile payload key must match tile_number")
        object.__setattr__(self, "payloads", typed)

    def value_at(self, mapping):
        tile_number = getattr(mapping, "tile_number", None)
        if tile_number is None:
            return None
        payload = self.payloads.get(int(tile_number))
        if payload is None:
            return None
        if payload.quality != "exact":
            # Preview payloads (coarser-LOD floor while the exact plane
            # computes) draw pixels but never provide semantic values.
            return None
        source = (
            payload.semantic_histogram_data
            if payload.semantic_histogram_data is not None
            else (payload.semantic_data if payload.semantic_data is not None else payload.image)
        )
        page_backing = payload.page_backing
        if (
            page_backing is not None
            and payload.semantic_histogram_data is None
            and payload.semantic_data is None
        ):
            local_y = int(getattr(mapping, "local_y", -1))
            local_x = int(getattr(mapping, "local_x", -1))
            coverage_y0, _coverage_y1, coverage_x0, _coverage_x1 = page_backing.source_coverage_yx
            return page_backing.value_at_native(
                coverage_y0 + local_y,
                coverage_x0 + local_x,
            )
        data = np.asarray(source)
        y_i = int(getattr(mapping, "local_y", -1))
        x_i = int(getattr(mapping, "local_x", -1))
        if y_i < 0 or x_i < 0 or y_i >= data.shape[0] or x_i >= data.shape[1]:
            return None
        return array_value_at(data, y_i, x_i)

    def tile_region(self, tile, region: tuple[slice, slice]):
        if tile is None:
            return None
        tile_number = getattr(tile, "montage_index", None)
        if tile_number is None:
            tile_number = getattr(tile, "region_id", None)
        if tile_number is None:
            tile_number = getattr(tile, "tile_number", None)
        if tile_number is None:
            return None
        payload = self.payloads.get(int(tile_number))
        if payload is None:
            return None
        if payload.quality != "exact":
            return None
        y_slice, x_slice = region
        semantic = payload.semantic_data if payload.semantic_data is not None else payload.image
        data = np.asarray(semantic)[y_slice, x_slice, ...]
        hist_source = payload.semantic_histogram_data if payload.semantic_histogram_data is not None else payload.histogram_data
        hist = None if hist_source is None else np.asarray(hist_source)[y_slice, x_slice]
        return data, hist, "committed_tile_payload"


def _coerce_tile_payload(payload) -> DisplayTilePayload:
    if not isinstance(payload, DisplayTilePayload):
        raise TypeError("tiled display presentations require DisplayTilePayload values")
    return payload


def _unique_int_tuple(values, label: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in tuple(values or ()))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"tile delta {label} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class DisplayFrameKey:
    document_key: object
    request_key: object
    render_generation: int
    semantic_key: object | None = None


@dataclass(frozen=True)
class CommittedDisplayFrame:
    data: np.ndarray | None
    histogram_data: np.ndarray | None
    geometry: DisplayGeometry
    levels: tuple[float, float]
    histogram_range: tuple[float, float]
    key: DisplayFrameKey
    value_source: FrameValueSource | None = None
    scene: DisplayScene | None = None

    def __post_init__(self) -> None:
        if self.value_source is None:
            raise ValueError("committed display frames require a tiled value source")
        if not isinstance(self.value_source, TiledValueSource):
            raise ValueError("committed display frames require a tiled value source")
        if self.scene is None:
            object.__setattr__(
                self,
                "scene",
                display_scene_for_geometry(self.geometry, payloads=self.value_source.payloads),
            )
        elif self.scene.geometry != self.geometry:
            raise ValueError("committed display frame scene geometry must match frame geometry")

    @property
    def is_tiled(self) -> bool:
        return isinstance(self.value_source, TiledValueSource)
