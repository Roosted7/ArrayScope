"""Single gateway from decided display presentations to ImageView2D."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.backends import surface_for_view
from arrayscope.display.model.commit import DisplayTiledPresentation
from arrayscope.display.model.frame import (
    CommittedDisplayFrame,
    DisplayFrameKey,
    TileCommitReport,
    TiledValueSource,
)
from arrayscope.display.scene import DisplayScene, display_scene_for_presentation


def _presentation_display_transposed(image_view, presentation) -> bool:
    """Whether a committed tiled presentation stores canonical tiles that the
    backend displays X/Y transposed.

    The swap only applies on a backend that renders canonical tiles
    (``display_axis_transpose``); a reversed image-axis pair then marks the
    display transpose so hover/ROI index the canonical array with swapped
    coordinates.  Legacy backends reorder pixels at materialization, so their
    display and array coordinates already align (never transposed here).
    """

    if not bool(image_view_backend_capabilities(image_view).display_axis_transpose):
        return False
    axes = getattr(getattr(presentation.geometry, "view_state", None), "image_axes", None) or ()
    axes = tuple(int(axis) for axis in axes)
    return len(axes) == 2 and axes[0] > axes[1]


class DisplayCommitter:
    def __init__(self, image_view):
        self.surface = surface_for_view(image_view)
        self.image_view = image_view
        self.last_tile_commit_report: TileCommitReport | None = None
        self.last_tile_committed_state = None

    def commit_tile_layer(
        self, presentation: DisplayTiledPresentation, key: DisplayFrameKey
    ) -> CommittedDisplayFrame:
        self._validate_presentation(presentation)
        self.commit_tiled_delta(presentation)
        committed_state = self.last_tile_committed_state or presentation.base_tile_state
        committed_presentation = replace(presentation, tile_state=committed_state)
        scene = display_scene_for_presentation(committed_presentation)
        self.surface.set_profile_bounds(scene.bounds)
        return self._frame_for(committed_presentation, key, scene, tile_state=committed_state)

    def commit_tiled_delta(self, presentation: DisplayTiledPresentation) -> TileCommitReport:
        """Present a tiled delta and return the backend acknowledgement.

        This is the hot-path primitive for progressive tile layers. It still
        commits through the backend and records exactly what was accepted, but
        it does not build a DisplayScene or CommittedDisplayFrame.
        """

        self._validate_presentation(presentation)
        # A raised backend commit must not leave the previous transaction's
        # acknowledgement readable.  Callers read ``last_tile_commit_report``
        # after the fact, so a stale report would be acknowledged as if it
        # described this delta.
        self.last_tile_commit_report = None
        self.last_tile_committed_state = None
        report = self.surface.present_tiled(presentation)
        if not isinstance(report, TileCommitReport):
            raise TypeError("tiled presentation commits require a TileCommitReport acknowledgement")
        tile_state = presentation.base_tile_state.acknowledge_delta(presentation.tile_delta, report)
        self.last_tile_commit_report = report
        self.last_tile_committed_state = tile_state
        return report

    def _frame_for(
        self,
        presentation: DisplayTiledPresentation,
        key: DisplayFrameKey,
        scene: DisplayScene,
        *,
        tile_state=None,
    ) -> CommittedDisplayFrame:
        data = None
        histogram_data = None
        committed_state = tile_state or presentation.tile_state
        value_source = TiledValueSource(
            committed_state.payloads,
            transposed=_presentation_display_transposed(self.image_view, presentation),
        )
        return CommittedDisplayFrame(
            data=data,
            histogram_data=histogram_data,
            geometry=presentation.geometry,
            levels=(float(presentation.levels[0]), float(presentation.levels[1])),
            histogram_range=(
                float(presentation.histogram_range[0]),
                float(presentation.histogram_range[1]),
            ),
            key=key,
            value_source=value_source,
            scene=scene,
        )

    def _validate_presentation(self, presentation: DisplayTiledPresentation) -> None:
        for tile_number, payload in dict(presentation.tile_state.payloads).items():
            if int(tile_number) != int(payload.tile_number):
                raise ValueError("tile payload key must match tile_number")
        transaction_payloads = presentation.tile_state.active_payloads(presentation.tile_delta)
        transaction_payloads.update(presentation.tile_delta.upserts)
        for tile_number, payload in transaction_payloads.items():
            if int(tile_number) != int(payload.tile_number):
                raise ValueError("tile transaction payload key must match tile_number")
            identity = getattr(payload, "presentation_identity", None)
            if (
                identity is None
                or int(identity.levels_generation) != int(presentation.tile_delta.level_revision)
                or identity.levels != tuple(float(value) for value in presentation.levels)
            ):
                raise ValueError(
                    "tile delta payload must name the transaction's accepted level generation"
                )
        if (
            presentation.histogram_plot_data is not None
            and np.asarray(presentation.histogram_plot_data).size < 1
        ):
            raise ValueError("histogram plot data must not be empty")
        self._validate_bounds("levels", presentation.levels)
        self._validate_bounds("histogram range", presentation.histogram_range)

    def _validate_bounds(self, label: str, bounds) -> None:
        try:
            low, high = bounds
            low = float(low)
            high = float(high)
        except Exception as exc:
            raise ValueError(f"{label} must be a pair of finite floats") from exc
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"{label} must be finite increasing bounds")
