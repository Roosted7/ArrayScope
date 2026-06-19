"""Committed visible display frame state and value sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.lod import LodInfo
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

    def __post_init__(self) -> None:
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
        return total


@dataclass(frozen=True)
class TilePresentationDelta:
    structure_revision: int
    payload_revision: int
    visibility_revision: int
    level_revision: int
    histogram_revision: int
    viewport_revision: int
    upserts: Mapping[int, DisplayTilePayload] = field(default_factory=dict)
    removals: tuple[int, ...] = ()
    active_tiles: tuple[int, ...] = ()
    planned_tiles: tuple[int, ...] = ()
    near_tiles: tuple[int, ...] = ()
    near_tile_source_ids: Mapping[int, object] = field(default_factory=dict)
    force_refresh: bool = False

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
        object.__setattr__(self, "upserts", upserts)
        object.__setattr__(self, "removals", removals)
        object.__setattr__(self, "active_tiles", active)
        object.__setattr__(self, "planned_tiles", planned)
        object.__setattr__(self, "near_tiles", near)
        object.__setattr__(self, "near_tile_source_ids", near_sources)
        object.__setattr__(self, "force_refresh", bool(self.force_refresh))


@dataclass(frozen=True)
class TilePresentationState:
    payloads: Mapping[int, DisplayTilePayload] = field(default_factory=dict)

    def __post_init__(self) -> None:
        typed = {int(key): _coerce_tile_payload(value) for key, value in dict(self.payloads).items()}
        for key, payload in typed.items():
            if int(payload.tile_number) != int(key):
                raise ValueError("tile state payload key must match tile_number")
        object.__setattr__(self, "payloads", typed)

    def apply_delta(self, delta: TilePresentationDelta) -> "TilePresentationState":
        if not isinstance(delta, TilePresentationDelta):
            raise TypeError("tile presentation state requires a TilePresentationDelta")
        payloads = dict(self.payloads)
        for tile_number in delta.removals:
            payloads.pop(int(tile_number), None)
        payloads.update(delta.upserts)
        return TilePresentationState(payloads)

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
class CanvasValueSource(FrameValueSource):
    data: np.ndarray
    histogram_data: np.ndarray | None
    geometry: DisplayGeometry

    def value_at(self, mapping):
        source = self.histogram_data if self.histogram_data is not None else self.data
        if source is None:
            return None
        data = np.asarray(source)
        if tuple(data.shape[:2]) != tuple(self.geometry.display_shape):
            return None
        y_i = int(getattr(mapping, "canvas_y", -1))
        x_i = int(getattr(mapping, "canvas_x", -1))
        if y_i < 0 or x_i < 0 or y_i >= data.shape[0] or x_i >= data.shape[1]:
            return None
        return array_value_at(data, y_i, x_i)

    def tile_region(self, tile, region: tuple[slice, slice]):
        geometry = self.geometry
        if tile is None or getattr(geometry, "montage", None) is None:
            return None
        y_slice, x_slice = region
        x0 = int(0 if x_slice.start is None else x_slice.start)
        x1 = int(tile.width if x_slice.stop is None else x_slice.stop)
        y0 = int(0 if y_slice.start is None else y_slice.start)
        y1 = int(tile.height if y_slice.stop is None else y_slice.stop)
        canvas_x0 = int(tile.x0 + x0 - geometry.montage_origin_x)
        canvas_y0 = int(tile.y0 + y0 - geometry.montage_origin_y)
        canvas_x1 = int(tile.x0 + x1 - geometry.montage_origin_x)
        canvas_y1 = int(tile.y0 + y1 - geometry.montage_origin_y)
        data = np.asarray(self.data)
        if canvas_x0 < 0 or canvas_y0 < 0 or canvas_x1 > data.shape[1] or canvas_y1 > data.shape[0]:
            return None
        hist = None if self.histogram_data is None else np.asarray(self.histogram_data)
        hist_region = None if hist is None else hist[canvas_y0:canvas_y1, canvas_x0:canvas_x1]
        return data[canvas_y0:canvas_y1, canvas_x0:canvas_x1, ...], hist_region, "committed_canvas"


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
        payload = self.payloads.get(int(tile.montage_index))
        if payload is None:
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

    def __post_init__(self) -> None:
        if self.value_source is None:
            if self.data is None:
                raise ValueError("a committed raster frame requires display data")
            object.__setattr__(
                self,
                "value_source",
                CanvasValueSource(
                    data=self.data,
                    histogram_data=self.histogram_data,
                    geometry=self.geometry,
                ),
            )
        elif self.data is None and not isinstance(self.value_source, TiledValueSource):
            raise ValueError("data-less committed frames require a tiled value source")

    @property
    def is_tiled(self) -> bool:
        return isinstance(self.value_source, TiledValueSource)
