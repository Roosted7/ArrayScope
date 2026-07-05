"""Committed visible display frame state and value sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.lod import LodInfo
from arrayscope.display.scene import DisplayScene, display_scene_for_geometry
from arrayscope.display.shader_mapping import ShaderMapping, TexturePlaneKind


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
    quality: str = "exact"

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
        semantic = image if self.semantic_data is None else np.asarray(self.semantic_data)
        semantic_histogram = self.histogram_data if self.semantic_histogram_data is None else self.semantic_histogram_data
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

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.shape(self.image))

    @property
    def dtype(self) -> np.dtype:
        return np.asarray(self.image).dtype

    @property
    def nbytes(self) -> int:
        total = int(np.asarray(self.texture_data if self.texture_data is not None else self.image).nbytes)
        if self.histogram_data is not None:
            total += int(np.asarray(self.histogram_data).nbytes)
        if self.semantic_data is not None and self.semantic_data is not self.image and self.semantic_data is not self.texture_data:
            total += int(np.asarray(self.semantic_data).nbytes)
        if (
            self.semantic_histogram_data is not None
            and self.semantic_histogram_data is not self.histogram_data
            and self.semantic_histogram_data is not self.semantic_data
        ):
            total += int(np.asarray(self.semantic_histogram_data).nbytes)
        if (
            self.level_data is not None
            and self.level_data is not self.image
            and self.level_data is not self.histogram_data
            and self.level_data is not self.semantic_data
            and self.level_data is not self.semantic_histogram_data
        ):
            total += int(np.asarray(self.level_data).nbytes)
        return total




@dataclass(frozen=True)
class TileCommitReport:
    """Backend acknowledgement for a tiled presentation commit.

    Counts distinguish expensive cold work from cheap resident rebinds or
    geometry-only changes, so scheduling feedback does not throttle already
    resident tiles as though they all required uploads.
    """

    presented_tiles: frozenset[int] = field(default_factory=frozenset)
    committed_upserts: frozenset[int] | None = None
    removed_tiles: frozenset[int] = field(default_factory=frozenset)
    texture_uploads: int = 0
    texture_upload_bytes: int = 0
    pyqtgraph_items_created: int = 0
    cpu_windowed_tiles: int = 0
    resident_rebinds: int = 0
    existing_items_shown: int = 0
    relocated_tiles: int = 0
    storage_rebuilds: int = 0
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

        if self.delta_key is None:
            return True
        return self.delta_key == (int(delta.base_revision), int(delta.target_revision))

    def accepted_upserts(self, delta: "TilePresentationDelta") -> set[int]:
        if not self.acknowledges(delta):
            return set()
        if self.committed_upserts is not None:
            return set(self.committed_upserts.intersection(delta.upserts))
        return set(self.presented_tiles.intersection(delta.upserts))


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
    cold_deadline_ms: float | None = None
    upserts: Mapping[int, DisplayTilePayload] = field(default_factory=dict)
    removals: tuple[int, ...] = ()
    active_tiles: tuple[int, ...] = ()
    planned_tiles: tuple[int, ...] = ()
    near_tiles: tuple[int, ...] = ()
    near_tile_source_ids: Mapping[int, object] = field(default_factory=dict)
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
        object.__setattr__(self, "structure_revision", int(self.structure_revision))
        object.__setattr__(self, "payload_revision", int(self.payload_revision))
        object.__setattr__(self, "visibility_revision", int(self.visibility_revision))
        object.__setattr__(self, "level_revision", int(self.level_revision))
        object.__setattr__(self, "histogram_revision", int(self.histogram_revision))
        object.__setattr__(self, "viewport_revision", int(self.viewport_revision))
        object.__setattr__(self, "base_revision", int(self.base_revision))
        target = int(self.target_revision) if int(self.target_revision) else int(self.base_revision) + (1 if upserts or removals else 0)
        object.__setattr__(self, "target_revision", target)
        deadline = self.cold_deadline_ms
        object.__setattr__(self, "cold_deadline_ms", None if deadline is None else max(0.0, float(deadline)))
        object.__setattr__(self, "upserts", upserts)
        object.__setattr__(self, "removals", removals)
        object.__setattr__(self, "active_tiles", active)
        object.__setattr__(self, "planned_tiles", planned)
        object.__setattr__(self, "near_tiles", near)
        object.__setattr__(self, "near_tile_source_ids", near_sources)
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
