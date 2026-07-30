"""Serializable full-view recipes for ArrayScope."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass

from arrayscope.core.view_state import ViewState
from arrayscope.operations.pipeline import evaluate_shape
from arrayscope.operations.recipes import operation_to_recipe_item, steps_from_recipe

VIEW_RECIPE_VERSION = 1


@dataclass(frozen=True)
class DisplaySettings:
    channel: str
    scale: str
    aspect_mode: str
    window_mode: str
    levels: tuple[float, float] | None = None
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
        "montage_axis": state.montage_axis,
        "montage_columns": state.montage_columns,
        "montage_indices": list(state.montage_indices)
        if state.montage_indices is not None
        else None,
        "montage_text": state.montage_text,
        "axis_range_indices": [
            list(value) if value is not None else None for value in state.axis_range_indices
        ],
        "axis_range_text": list(state.axis_range_text),
    }


def _per_axis(values, ndim, fill):
    """``values`` resized to exactly ``ndim`` entries, padding with ``fill``.

    A recipe's per-axis tuples are only ever read back at the rank they were
    written at (see :func:`view_state_from_mapping`); this trims/pads the one
    case that rank cannot fix -- a hand-edited or truncated payload.
    """

    values = () if values is None else tuple(values)
    if len(values) >= ndim:
        return values[:ndim]
    return values + (fill,) * (ndim - len(values))


def view_state_from_mapping(mapping, base_shape):
    """Rebuild a :class:`ViewState` from ``mapping`` and migrate it to ``base_shape``.

    The state is constructed at *its own* recorded rank and only then handed to
    :meth:`ViewState.for_shape`. Constructing it directly at ``base_shape``
    -- what this used to do -- fed per-axis tuples of the recorded rank into a
    ``ViewState`` declared at the target rank, so a recipe saved against a 3-D
    document and restored against a 4-D one (or the reverse) raised
    ``slice_indices length must match ndim`` before any migration could run.
    That is live the moment operations are edited between save and restore, and
    it is exactly what a rank-increasing operation makes ordinary.

    ``for_shape`` stays the single owner of rank migration: everything below
    only reconstitutes the saved state faithfully at the rank it was saved at.
    """

    if not isinstance(mapping, dict):
        raise ValueError("view_state must be an object")
    base_shape = tuple(int(size) for size in base_shape)
    recorded_shape = tuple(int(size) for size in (mapping.get("shape") or base_shape))
    ndim = len(recorded_shape)
    state = ViewState(
        ndim=ndim,
        shape=recorded_shape,
        image_axes=tuple(mapping["image_axes"]) if mapping.get("image_axes") is not None else None,
        line_axis=mapping.get("line_axis"),
        slice_indices=_per_axis(mapping.get("slice_indices"), ndim, 0),
        channel=mapping.get("channel", "real"),
        scale=mapping.get("scale", "linear"),
        axis_flipped=_per_axis(mapping.get("axis_flipped"), ndim, False),
        axis_fftshifted=_per_axis(mapping.get("axis_fftshifted"), ndim, False),
        montage_axis=mapping.get("montage_axis"),
        montage_columns=mapping.get("montage_columns"),
        montage_indices=tuple(mapping["montage_indices"])
        if mapping.get("montage_indices") is not None
        else None,
        montage_text=mapping.get("montage_text"),
        axis_range_indices=tuple(
            None if value is None else tuple(value)
            for value in _per_axis(mapping.get("axis_range_indices"), ndim, None)
        ),
        axis_range_text=_per_axis(mapping.get("axis_range_text"), ndim, None),
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


def recipe_from_mapping(mapping, base_shape, *, imported: bool = False, base_data=None):
    """Rebuild a :class:`ViewRecipe` against ``base_shape``.

    ``base_data`` is optional and only ever used to *predict a shape*: an
    operation whose output shape is discovered rather than declared (an
    ADR-0060 plugin op) can only answer from a real input, and
    :func:`~arrayscope.operations.pipeline.evaluate_shape` is the one place
    that funnel lives. Walking ``step.operation.output_shape`` directly -- what
    this used to do -- raised ``CharacterizationUnavailable`` for exactly those
    operations, which surfaced as "Skipped saved view" on restore.
    """

    if not isinstance(mapping, dict):
        raise ValueError("view recipe must be a JSON object")
    if mapping.get("version") != VIEW_RECIPE_VERSION:
        raise ValueError(f"unsupported view recipe version: {mapping.get('version')!r}")
    operations_recipe = {"version": 2, "operations": mapping.get("operations", [])}
    steps = steps_from_recipe(operations_recipe, base_shape, imported=imported)
    shape = evaluate_shape(
        tuple(base_shape),
        tuple(step.operation for step in steps if step.enabled),
        base_dtype=getattr(base_data, "dtype", None),
        base_data=base_data,
    )
    return ViewRecipe(
        view_state=view_state_from_mapping(mapping.get("view_state", {}), shape),
        display=display_settings_from_mapping(mapping.get("display", {})),
        steps=steps,
    )


def dumps_view_recipe(recipe: ViewRecipe, **kwargs):
    options = {"indent": 2, "sort_keys": True}
    options.update(kwargs)
    return json.dumps(recipe_to_mapping(recipe), **options)


def loads_view_recipe(text: str, base_shape, *, imported: bool = False, base_data=None):
    try:
        mapping = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON view recipe: {exc}") from exc
    return recipe_from_mapping(mapping, base_shape, imported=imported, base_data=base_data)


def save_view_recipe(path, recipe: ViewRecipe):
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, delete=False
    ) as recipe_file:
        temporary_path = recipe_file.name
        recipe_file.write(dumps_view_recipe(recipe))
        recipe_file.write("\n")
    os.replace(temporary_path, path)


def load_view_recipe(path, base_shape, *, base_data=None):
    with open(path, encoding="utf-8") as recipe_file:
        return loads_view_recipe(recipe_file.read(), base_shape, imported=True, base_data=base_data)
