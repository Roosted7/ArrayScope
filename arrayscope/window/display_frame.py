"""Committed visible display frame state and value sources."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arrayscope.display.geometry import DisplayGeometry


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

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        if image.ndim < 2:
            raise ValueError("display tile payload image must be at least 2D")
        if self.histogram_data is not None:
            histogram = np.asarray(self.histogram_data)
            if tuple(histogram.shape[:2]) != tuple(image.shape[:2]):
                raise ValueError("display tile payload histogram shape must match image shape")
        object.__setattr__(self, "tile_number", int(self.tile_number))
        object.__setattr__(self, "source_index", int(self.source_index))
        object.__setattr__(self, "image", image)
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
        total = int(np.asarray(self.image).nbytes)
        if self.histogram_data is not None:
            total += int(np.asarray(self.histogram_data).nbytes)
        return total


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
        source = payload.histogram_data if payload.histogram_data is not None else payload.image
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
        data = np.asarray(payload.image)[y_slice, x_slice, ...]
        hist = None if payload.histogram_data is None else np.asarray(payload.histogram_data)[y_slice, x_slice]
        return data, hist, "committed_tile_payload"


def _coerce_tile_payload(payload) -> DisplayTilePayload:
    if not isinstance(payload, DisplayTilePayload):
        raise TypeError("tiled display presentations require DisplayTilePayload values")
    return payload


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
