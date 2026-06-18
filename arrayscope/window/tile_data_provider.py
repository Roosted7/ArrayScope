"""Window-side demand access to visible/cached/evaluated tile regions."""

from __future__ import annotations

import numpy as np

from arrayscope.operations.tile_regions import TileRegionRequest, TileRegionResult


class TileDataProvider:
    def __init__(
        self,
        *,
        operation_evaluator,
        document,
        committed_frame=None,
        montage_plan=None,
        colormap_lut=None,
        evaluation_context=None,
    ):
        self.operation_evaluator = operation_evaluator
        self.document = document
        self.committed_frame = committed_frame
        self.montage_plan = montage_plan
        self.colormap_lut = colormap_lut
        self.evaluation_context = evaluation_context

    def request_tile_region(self, request: TileRegionRequest, *, priority=None, cancellation_token=None) -> TileRegionResult:
        del priority
        _check_cancelled(cancellation_token)
        cached = _call_cache(self.operation_evaluator, "cached_tile_region_silent", "cached_tile_region", request)
        if cached is not None:
            return TileRegionResult(request, cached[0], cached[1], "region_cache")

        tile = self._tile_for_request(request)
        region = self._normalized_region(request, tile)
        visible = self._from_committed_canvas(request, tile, region)
        if visible is not None:
            _store_region(self.operation_evaluator, request, (visible.image, visible.histogram_data))
            return visible

        _check_cancelled(cancellation_token)
        if tile is not None and request.montage_axis is not None:
            payload = _call_cache(
                self.operation_evaluator,
                "cached_montage_tile_silent",
                "cached_montage_tile",
                tile.view_state,
                montage_axis=request.montage_axis,
                source_index=tile.source_index,
                colormap_lut=self.colormap_lut,
            )
            if payload is not None:
                result = _slice_payload(request, payload.image, payload.histogram_data, region, "tile_cache")
                _store_region(self.operation_evaluator, request, (result.image, result.histogram_data))
                return result

        _check_cancelled(cancellation_token)
        view_state = tile.view_state if tile is not None else request.view_state
        display_image = self.operation_evaluator.evaluate_image_snapshot_silent(
            self.document,
            view_state,
            colormap_lut=self.colormap_lut,
            evaluation_context=self.evaluation_context,
        ).value
        result = _slice_payload(request, display_image.data, display_image.histogram_data, region, "computed")
        _store_region(self.operation_evaluator, request, (result.image, result.histogram_data))
        return result

    def _tile_for_request(self, request: TileRegionRequest):
        plan = self.montage_plan
        if plan is None or request.tile_number is None:
            return None
        tile_number = int(request.tile_number)
        if tile_number < 0 or tile_number >= len(plan.tiles):
            return None
        return plan.tiles[tile_number]

    def _normalized_region(self, request: TileRegionRequest, tile) -> tuple[slice, slice]:
        region = request.tile_local_region
        if region is not None:
            return region
        if tile is not None:
            return (slice(0, int(tile.height)), slice(0, int(tile.width)))
        shape = getattr(request.view_state, "shape", ())
        if getattr(request.view_state, "image_axes", None) is not None:
            y_axis, x_axis = request.view_state.image_axes
            return (slice(0, int(shape[y_axis])), slice(0, int(shape[x_axis])))
        return (slice(None), slice(None))

    def _from_committed_canvas(self, request: TileRegionRequest, tile, region: tuple[slice, slice]) -> TileRegionResult | None:
        frame = self.committed_frame
        if frame is None or tile is None:
            return None
        geometry = getattr(frame, "geometry", None)
        if geometry is None or getattr(geometry, "montage", None) is None:
            return None
        if tuple(getattr(geometry, "montage_tile_states", ()) or ()):
            state = geometry.view_point_to_tile_point(tile.x0, tile.y0, require_loaded=True)
            if state is None or state.kind != "loaded":
                return None
        value_source = getattr(frame, "value_source", None)
        if value_source is None:
            return None
        committed = value_source.tile_region(tile, region)
        if committed is None:
            return None
        image, histogram, source = committed
        return TileRegionResult(request, image, histogram, source)


def _slice_payload(request, image, histogram_data, region: tuple[slice, slice], source: str) -> TileRegionResult:
    y_slice, x_slice = region
    data = np.asarray(image)[y_slice, x_slice, ...]
    hist = None if histogram_data is None else np.asarray(histogram_data)[y_slice, x_slice]
    return TileRegionResult(request, data, hist, source)


def _check_cancelled(token) -> None:
    if token is not None and getattr(token, "cancelled", False):
        from arrayscope.operations.cancellation import EvaluationCancelled

        raise EvaluationCancelled()


def _call_cache(evaluator, silent_name: str, status_name: str, *args, **kwargs):
    method = getattr(evaluator, silent_name, None)
    if callable(method):
        return method(*args, **kwargs)
    return getattr(evaluator, status_name)(*args, **kwargs)


def _store_region(evaluator, request, result):
    method = getattr(evaluator, "store_tile_region_result_silent", None)
    if callable(method):
        return method(request, result)
    return evaluator.store_tile_region_result(request, result)
