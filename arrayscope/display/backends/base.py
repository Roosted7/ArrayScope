"""Semantic image-surface boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from arrayscope.display.backend_contract import ImageViewBackendCapabilities

if TYPE_CHECKING:
    from arrayscope.display.interaction import DisplayInteractionState
    from arrayscope.display.model.commit import DisplayTiledPresentation
    from arrayscope.display.model.frame import TileCommitReport


@runtime_checkable
class ImageSurface(Protocol):
    """Concrete pixel surface expressed in ArrayScope display semantics."""

    @property
    def capabilities(self) -> ImageViewBackendCapabilities: ...

    @property
    def widget(self): ...

    def present_tiled(self, presentation: "DisplayTiledPresentation") -> "TileCommitReport": ...

    def invalidate_tiled_presentation(self, reason: str) -> None: ...

    def hide_tiled_presentation(self, reason: str) -> None: ...

    def reset_tiled_residency(self, reason: str) -> None: ...

    def set_profile_bounds(self, bounds: tuple[float, float, float, float]) -> None: ...

    def apply_camera(
        self,
        image_shape: tuple[int, int],
        viewport_policy,
        *,
        image_origin: tuple[float, float] = (0.0, 0.0),
        content_rect: tuple[float, float, float, float] | None = None,
    ) -> None: ...

    def map_scene_to_overlay(self, scene_pos): ...

    def current_viewport_rect(self) -> tuple[float, float, float, float] | None: ...

    def presentation_diagnostics(self) -> dict[str, object]: ...

    def interaction_event_owner(self) -> str: ...

    def sync_interaction_state(self, state: "DisplayInteractionState") -> None: ...

    def reset_surface(self, reason: str) -> None: ...

    def teardown_surface(self) -> None: ...


def surface_for_view(view) -> ImageSurface:
    """Return the concrete surface owned by an image-view shell."""

    missing = object()
    surface = getattr(view, "surface", missing)
    if isinstance(surface, ImageSurface):
        return surface
    if isinstance(view, ImageSurface):
        return view
    view_type = _qualified_type_name(view)
    if surface is missing:
        detail = "missing .surface"
    elif surface is None:
        detail = ".surface is None"
    else:
        detail = f".surface is {_qualified_type_name(surface)}, which does not implement ImageSurface"
    raise TypeError(f"{view_type} does not expose an ImageSurface ({detail})")


def _qualified_type_name(value) -> str:
    cls = type(value)
    module = getattr(cls, "__module__", "")
    name = getattr(cls, "__qualname__", getattr(cls, "__name__", str(cls)))
    return name if module in {"", "builtins"} else f"{module}.{name}"
