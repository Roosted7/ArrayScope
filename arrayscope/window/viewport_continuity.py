"""Viewport continuity transactions shared by restore, reload, and layout settle."""

from __future__ import annotations

from dataclasses import dataclass

from arrayscope.core.view_session import ViewportSession
from arrayscope.display.viewport import ViewportMode


@dataclass(init=False)
class ViewportContinuityTransaction:
    """Single owner for preserved viewport intent.

    The transaction is Qt-free. Window controllers may resize, commit, or
    retarget in separate event turns, but they all consume this object instead
    of carrying parallel pending viewport flags.
    """

    reason: str
    generation: int
    view_range: tuple[tuple[float, float], tuple[float, float]] | None
    viewport_shape: tuple[int, int] | None
    montage_columns: int | None
    mode: str
    semantic_key: object | None
    profile_visible: bool
    defaults: dict[str, object]
    shape_settled: bool
    range_applied: bool
    released: bool
    message_enabled: bool

    def __init__(
        self,
        *,
        reason: str = "file-session-restore",
        generation: int = 0,
        viewport: ViewportSession | None = None,
        view_range=None,
        viewport_shape: tuple[int, int] | None = None,
        montage_columns: int | None = None,
        mode: str | None = None,
        semantic_key: object | None = None,
        profile_visible: bool = False,
        defaults: dict[str, object] | None = None,
        applied: bool = False,
        shape_settled: bool = False,
        range_applied: bool | None = None,
        released: bool | None = None,
        message_enabled: bool = True,
    ) -> None:
        if viewport is not None:
            if view_range is None:
                view_range = viewport.view_range
            if viewport_shape is None:
                viewport_shape = viewport.viewport_shape
            if montage_columns is None:
                montage_columns = viewport.montage_columns
            if mode is None:
                mode = str(viewport.mode)
        self.reason = str(reason)
        self.generation = int(generation)
        self.view_range = normalize_view_range(view_range)
        self.viewport_shape = normalize_viewport_shape(viewport_shape)
        self.montage_columns = None if montage_columns is None else max(1, int(montage_columns))
        self.mode = str(mode or ViewportMode.AUTO_UNTOUCHED.value)
        self.semantic_key = semantic_key
        self.profile_visible = bool(profile_visible)
        self.defaults = dict(defaults or {})
        self.shape_settled = bool(shape_settled or self.viewport_shape is None)
        self.range_applied = bool(applied if range_applied is None else range_applied)
        self.released = bool(False if released is None else released)
        self.message_enabled = bool(message_enabled)

    @property
    def viewport(self) -> ViewportSession | None:
        if self.view_range is None and self.viewport_shape is None and self.montage_columns is None:
            return None
        return ViewportSession(
            mode=self.mode,
            view_range=self.view_range,
            viewport_shape=self.viewport_shape,
            montage_columns=self.montage_columns,
        )

    @property
    def applied(self) -> bool:
        return bool(self.range_applied)

    @applied.setter
    def applied(self, value: bool) -> None:
        self.range_applied = bool(value)


def normalize_view_range(view_range) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if view_range is None:
        return None
    try:
        if len(view_range) != 2 or len(view_range[0]) != 2 or len(view_range[1]) != 2:
            return None
        return (
            (float(view_range[0][0]), float(view_range[0][1])),
            (float(view_range[1][0]), float(view_range[1][1])),
        )
    except Exception:
        return None


def normalize_viewport_shape(viewport_shape) -> tuple[int, int] | None:
    if viewport_shape is None:
        return None
    try:
        if len(viewport_shape) != 2:
            return None
        return (max(1, int(viewport_shape[0])), max(1, int(viewport_shape[1])))
    except Exception:
        return None
