"""Single gateway from decided display presentations to ImageView2D."""

from __future__ import annotations

import numpy as np

from arrayscope.display.backends import RasterCommitMode, backend_adapter_for_view
from arrayscope.display.scene import DisplayScene, display_scene_for_presentation
from arrayscope.display.model.frame import CanvasValueSource, CommittedDisplayFrame, DisplayFrameKey, TileCommitReport, TiledValueSource
from arrayscope.display.model.commit import DisplayPresentation, DisplayRasterPresentation, DisplayTiledPresentation


class DisplayCommitter:
    def __init__(self, image_view):
        self.backend = backend_adapter_for_view(image_view)
        self.image_view = self.backend.view
        self.last_tile_commit_report: TileCommitReport | None = None
        self.last_tile_committed_state = None

    def commit_full(self, presentation: DisplayPresentation, key: DisplayFrameKey) -> CommittedDisplayFrame:
        self.last_tile_commit_report = None
        self.last_tile_committed_state = None
        presentation = self._require_raster(presentation, "full")
        self._validate_presentation(presentation)
        scene = display_scene_for_presentation(presentation)
        self.backend.present_raster(presentation, mode=RasterCommitMode.FULL)
        self.backend.set_profile_bounds(scene.bounds)
        return self._frame_for(presentation, key, scene)

    def commit_fast(self, presentation: DisplayPresentation, key: DisplayFrameKey) -> CommittedDisplayFrame:
        self.last_tile_commit_report = None
        self.last_tile_committed_state = None
        presentation = self._require_raster(presentation, "fast")
        self._validate_presentation(presentation)
        if self.backend.current_raster_shape() != tuple(presentation.geometry.display_shape):
            raise ValueError("fast display commit requires an existing image with the same display shape")
        scene = display_scene_for_presentation(presentation)
        self.backend.present_raster(presentation, mode=RasterCommitMode.FAST)
        self.backend.set_profile_bounds(scene.bounds)
        return self._frame_for(presentation, key, scene)

    def commit_tile_layer(self, presentation: DisplayPresentation, key: DisplayFrameKey) -> CommittedDisplayFrame:
        self._validate_presentation(presentation)
        scene = display_scene_for_presentation(presentation)
        if isinstance(presentation, DisplayTiledPresentation):
            report = self.backend.present_tiled(presentation)
            if not isinstance(report, TileCommitReport):
                report = TileCommitReport(
                    presented_tiles=presentation.tile_state.active_payloads(presentation.tile_delta),
                    removed_tiles=presentation.tile_delta.removals,
                )
            tile_state = presentation.base_tile_state.acknowledge_delta(presentation.tile_delta, report)
            self.last_tile_commit_report = report
            self.last_tile_committed_state = tile_state
        else:
            self.last_tile_commit_report = None
            self.last_tile_committed_state = None
            self.backend.present_raster(presentation, mode=RasterCommitMode.TILE_LAYER)
        self.backend.set_profile_bounds(scene.bounds)
        return self._frame_for(presentation, key, scene, tile_state=self.last_tile_committed_state)

    def _frame_for(
        self,
        presentation: DisplayPresentation,
        key: DisplayFrameKey,
        scene: DisplayScene,
        *,
        tile_state=None,
    ) -> CommittedDisplayFrame:
        if isinstance(presentation, DisplayTiledPresentation):
            data = None
            histogram_data = None
            committed_state = tile_state or presentation.tile_state
            value_source = TiledValueSource(committed_state.payloads)
        else:
            data = presentation.data
            histogram_data = presentation.histogram_data
            value_source = CanvasValueSource(
                data=presentation.data,
                histogram_data=presentation.histogram_data,
                geometry=presentation.geometry,
            )
        return CommittedDisplayFrame(
            data=data,
            histogram_data=histogram_data,
            geometry=presentation.geometry,
            levels=(float(presentation.levels[0]), float(presentation.levels[1])),
            histogram_range=(float(presentation.histogram_range[0]), float(presentation.histogram_range[1])),
            key=key,
            value_source=value_source,
            scene=scene,
        )

    def _validate_presentation(self, presentation: DisplayPresentation) -> None:
        if isinstance(presentation, DisplayTiledPresentation):
            if getattr(presentation.geometry, "montage", None) is None:
                raise ValueError("tiled display presentation requires montage geometry")
            for tile_number, payload in dict(presentation.tile_state.payloads).items():
                if int(tile_number) != int(payload.tile_number):
                    raise ValueError("tile payload key must match tile_number")
            for tile_number, payload in dict(presentation.tile_delta.upserts).items():
                if int(tile_number) != int(payload.tile_number):
                    raise ValueError("tile delta upsert key must match tile_number")
            if presentation.histogram_plot_data is not None and np.asarray(presentation.histogram_plot_data).size < 1:
                raise ValueError("histogram plot data must not be empty")
        else:
            shape = tuple(np.shape(presentation.data)[:2])
            if shape != tuple(presentation.geometry.display_shape):
                raise ValueError(f"display data shape {shape} does not match geometry {presentation.geometry.display_shape}")
            if presentation.histogram_data is not None and tuple(np.shape(presentation.histogram_data)[:2]) != shape:
                raise ValueError("histogram data shape does not match display data shape")
            if presentation.histogram_plot_data is not None and np.asarray(presentation.histogram_plot_data).size < 1:
                raise ValueError("histogram plot data must not be empty")
        self._validate_bounds("levels", presentation.levels)
        self._validate_bounds("histogram range", presentation.histogram_range)

    def _require_raster(self, presentation: DisplayPresentation, commit_kind: str) -> DisplayRasterPresentation:
        if not isinstance(presentation, DisplayRasterPresentation):
            raise TypeError(f"{commit_kind} display commit requires a raster presentation")
        return presentation

    def _validate_bounds(self, label: str, bounds) -> None:
        try:
            low, high = bounds
            low = float(low)
            high = float(high)
        except Exception as exc:
            raise ValueError(f"{label} must be a pair of finite floats") from exc
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"{label} must be finite increasing bounds")
