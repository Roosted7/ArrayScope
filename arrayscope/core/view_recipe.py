"""Serializable full-view recipes for ArrayScope."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Tuple

from arrayscope.core.view_state import ViewState
from arrayscope.operations.recipes import operation_to_recipe_item, steps_from_recipe


VIEW_RECIPE_VERSION = 1


@dataclass(frozen=True)
class DisplaySettings:
    channel: str
    scale: str
    aspect_mode: str
    window_mode: str
    levels: Tuple[float, float] | None = None
    colormap: str | None = None
    profile_visible: bool = False
    live_profile: bool = False


@dataclass(frozen=True)
class ViewRecipe:
    view_state: ViewState
    display: DisplaySettings
    steps: tuple = ()
    version: int = VIEW_RECIPE_VERSION


def view_state_to_mapping(state: ViewState):
    return {
        "shape": list(state.shape),
        "image_axes": list(state.image_axes) if state.image_axes is not None else None,
        "line_axis": state.line_axis,
        "slice_indices": list(state.slice_indices),
        "channel": state.channel.value,
        "scale": state.scale.value,
        "axis_flipped": list(state.axis_flipped),
        "axis_fftshifted": list(state.axis_fftshifted),
    }


def view_state_from_mapping(mapping, base_shape):
    if not isinstance(mapping, dict):
        raise ValueError("view_state must be an object")
    state = ViewState(
        ndim=len(tuple(base_shape)),
        shape=tuple(base_shape),
        image_axes=tuple(mapping["image_axes"]) if mapping.get("image_axes") is not None else None,
        line_axis=mapping.get("line_axis"),
        slice_indices=tuple(mapping.get("slice_indices", (0,) * len(tuple(base_shape)))),
        channel=mapping.get("channel", "real"),
        scale=mapping.get("scale", "linear"),
        axis_flipped=tuple(mapping.get("axis_flipped", (False,) * len(tuple(base_shape)))),
        axis_fftshifted=tuple(mapping.get("axis_fftshifted", (False,) * len(tuple(base_shape)))),
    )
    return state.for_shape(base_shape, preserve_flags=True)


def display_settings_to_mapping(settings: DisplaySettings):
    return {
        "channel": settings.channel,
        "scale": settings.scale,
        "aspect_mode": settings.aspect_mode,
        "window_mode": settings.window_mode,
        "levels": list(settings.levels) if settings.levels is not None else None,
        "colormap": settings.colormap,
        "profile_visible": bool(settings.profile_visible),
        "live_profile": bool(settings.live_profile),
    }


def display_settings_from_mapping(mapping):
    if not isinstance(mapping, dict):
        raise ValueError("display settings must be an object")
    levels = mapping.get("levels")
    return DisplaySettings(
        channel=str(mapping.get("channel", "real")),
        scale=str(mapping.get("scale", "linear")),
        aspect_mode=str(mapping.get("aspect_mode", "square_pixels")),
        window_mode=str(mapping.get("window_mode", "relative")),
        levels=tuple(float(value) for value in levels) if levels is not None else None,
        colormap=mapping.get("colormap"),
        profile_visible=bool(mapping.get("profile_visible", False)),
        live_profile=bool(mapping.get("live_profile", False)),
    )


def recipe_to_mapping(recipe: ViewRecipe):
    return {
        "version": recipe.version,
        "view_state": view_state_to_mapping(recipe.view_state),
        "display": display_settings_to_mapping(recipe.display),
        "operations": [operation_to_recipe_item(step) for step in recipe.steps],
    }


def recipe_from_mapping(mapping, base_shape):
    if not isinstance(mapping, dict):
        raise ValueError("view recipe must be a JSON object")
    if mapping.get("version") != VIEW_RECIPE_VERSION:
        raise ValueError(f"unsupported view recipe version: {mapping.get('version')!r}")
    operations_recipe = {"version": 2, "operations": mapping.get("operations", [])}
    steps = steps_from_recipe(operations_recipe, base_shape)
    shape = tuple(base_shape)
    for step in steps:
        if step.enabled:
            shape = step.operation.output_shape(shape)
    return ViewRecipe(
        view_state=view_state_from_mapping(mapping.get("view_state", {}), shape),
        display=display_settings_from_mapping(mapping.get("display", {})),
        steps=steps,
    )


def dumps_view_recipe(recipe: ViewRecipe, **kwargs):
    options = {"indent": 2, "sort_keys": True}
    options.update(kwargs)
    return json.dumps(recipe_to_mapping(recipe), **options)


def loads_view_recipe(text: str, base_shape):
    try:
        mapping = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON view recipe: {exc}") from exc
    return recipe_from_mapping(mapping, base_shape)


def save_view_recipe(path, recipe: ViewRecipe):
    with open(path, "w", encoding="utf-8") as recipe_file:
        recipe_file.write(dumps_view_recipe(recipe))
        recipe_file.write("\n")


def load_view_recipe(path, base_shape):
    with open(path, "r", encoding="utf-8") as recipe_file:
        return loads_view_recipe(recipe_file.read(), base_shape)
