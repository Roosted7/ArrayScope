"""JSON operation recipe serialization for ArrayScope."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass

from arrayscope.operations.pipeline import evaluate_shape
from arrayscope.operations.registry import create_operation, get_operation_entry, operation_id_for


RECIPE_VERSION = 1


def recipe_from_operations(operations):
    return {
        "version": RECIPE_VERSION,
        "operations": [operation_to_recipe_item(operation) for operation in operations],
    }


def operation_to_recipe_item(operation):
    if not is_dataclass(operation):
        raise ValueError(f"operation is not serializable: {operation!r}")

    operation_id = operation_id_for(operation)
    entry = get_operation_entry(operation_id)
    item = {"id": operation_id}

    if entry.requires_axis:
        item["axis"] = int(getattr(operation, "axis"))

    parameters = {}
    for parameter in entry.parameters:
        parameters[parameter.name] = getattr(operation, parameter.name)
    if parameters:
        item["parameters"] = parameters

    known_names = {"axis", *(parameter.name for parameter in entry.parameters)}
    extra_names = {field.name for field in fields(operation)} - known_names
    if extra_names:
        raise ValueError(f"operation has unsupported recipe fields: {sorted(extra_names)}")

    return item


def operations_from_recipe(recipe, base_shape):
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be a JSON object")
    if recipe.get("version") != RECIPE_VERSION:
        raise ValueError(f"unsupported recipe version: {recipe.get('version')!r}")

    raw_operations = recipe.get("operations")
    if not isinstance(raw_operations, list):
        raise ValueError("recipe operations must be a list")

    operations = []
    shape = tuple(int(size) for size in base_shape)
    for index, item in enumerate(raw_operations):
        if not isinstance(item, dict):
            raise ValueError(f"operation {index} must be an object")
        operation_id = item.get("id")
        if not isinstance(operation_id, str):
            raise ValueError(f"operation {index} is missing an id")
        try:
            operation = create_operation(
                operation_id,
                axis=item.get("axis"),
                parameters=item.get("parameters", {}),
            )
            shape = operation.output_shape(shape)
        except Exception as exc:
            raise ValueError(f"operation {index} ({operation_id}) is incompatible: {exc}") from exc
        operations.append(operation)

    return tuple(operations)


def dumps_recipe(operations, **kwargs) -> str:
    options = {"indent": 2, "sort_keys": True}
    options.update(kwargs)
    return json.dumps(recipe_from_operations(operations), **options)


def loads_recipe(text: str, base_shape):
    try:
        recipe = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON recipe: {exc}") from exc
    return operations_from_recipe(recipe, base_shape)


def save_recipe(path, operations):
    with open(path, "w", encoding="utf-8") as recipe_file:
        recipe_file.write(dumps_recipe(operations))
        recipe_file.write("\n")


def load_recipe(path, base_shape):
    with open(path, "r", encoding="utf-8") as recipe_file:
        return loads_recipe(recipe_file.read(), base_shape)
