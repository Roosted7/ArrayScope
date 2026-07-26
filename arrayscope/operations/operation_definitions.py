"""Qt-free declarative definitions for every operation registration tier.

The runtime registry remains the execution owner.  This module provides the
shared, wrapper-shaped representation used to *read* an operation and to seed
an editable user copy without turning the declarative format into a second
execution mechanism (ADR 0060, decision 1).
"""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from typing import Any

from arrayscope.operations import plugins, registry
from arrayscope.operations.registry import OperationEntry, OperationParameter


def export_operation_definition(operation: str | OperationEntry) -> dict[str, Any]:
    """Render any registered operation into the shared wrapper-shaped schema.

    Built-ins identify their native implementation class, in-tree packs identify
    their providing module, entry-point plugins retain the entry-point identity,
    and user operations retain their existing runtime/source block.
    """

    entry = registry.get_operation_entry(operation) if isinstance(operation, str) else operation
    operation_id = str(entry.id)

    if operation_id.startswith("user:"):
        from arrayscope.operations import library

        payload = library.user_operation_wrapper(operation_id)
        if payload is None:
            raise ValueError(f"user operation has no wrapper: {operation_id}")
        definition = copy.deepcopy(payload)
        definition.setdefault("format", library.WRAPPER_FORMAT)
        definition.setdefault("version", 1)
        definition["tier"] = "user"
        definition["parameters"] = _parameter_payloads(entry.parameters)
        definition["input_slots"] = _input_slot_payloads(entry.input_slots)
        definition["unavailable_reason"] = entry.unavailable_reason
        return definition

    registry.load_operation_packs()
    pack_spec = registry._PACK_SPECS.get(operation_id)
    if pack_spec is not None:
        provider = _callable_module(pack_spec.build or pack_spec.fn)
        definition = _definition_from_entry(
            entry,
            tier="pack",
            runtime=str(getattr(pack_spec, "runtime", "python") or "python"),
            source={
                "mode": "pack",
                "id": operation_id,
                "provider": operation_id.split(":", 1)[0],
                "module": provider,
            },
        )
        runtime_config = dict(getattr(pack_spec, "runtime_config", None) or {})
        definition.update(runtime_config)
        environment_id = str(getattr(pack_spec, "environment_id", "") or "")
        if environment_id:
            definition["environment"] = environment_id
        return definition

    if operation_id in registry.OPERATION_REGISTRY:
        operation_type = entry.operation_type
        return _definition_from_entry(
            entry,
            tier="builtin",
            runtime="native",
            source={
                "mode": "native",
                "module": operation_type.__module__,
                "class": operation_type.__qualname__,
            },
        )

    entry_points = plugins.discover_plugin_entry_points()
    entry_point = entry_points.get(operation_id)
    if entry_point is not None:
        distribution = getattr(getattr(entry_point, "dist", None), "name", None)
        return _definition_from_entry(
            entry,
            tier="plugin",
            runtime="python",
            source={
                "mode": "entry-point",
                "group": plugins.PLUGIN_ENTRY_POINT_GROUP,
                "name": entry_point.name,
                "value": entry_point.value,
                "distribution": str(distribution or ""),
            },
        )

    raise ValueError(f"operation registration tier is unknown: {operation_id}")


def native_editable_source(entry: OperationEntry, callable_name: str) -> str:
    """Copy a built-in ``apply`` implementation into an editable function.

    References to pipeline globals are qualified through ``_native`` and
    dataclass fields become ordinary function arguments.  The resulting file is
    standalone user-editable code while continuing to reuse low-level ArrayScope
    helpers such as the FFT backend owner.
    """

    method = entry.operation_type.apply
    method_source = textwrap.dedent(inspect.getsource(method))
    parsed = ast.parse(method_source)
    original = parsed.body[0]
    if not isinstance(original, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise TypeError(f"{entry.operation_type.__qualname__}.apply is not a function")

    arguments = [ast.arg(arg="data")]
    if entry.requires_axis:
        arguments.append(ast.arg(arg="axis"))
    arguments.extend(ast.arg(arg=parameter.name) for parameter in entry.parameters)
    defaults = [_literal_default(parameter) for parameter in entry.parameters]
    original.name = callable_name
    original.args = ast.arguments(
        posonlyargs=[],
        args=arguments,
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=defaults,
    )
    original.decorator_list = []
    original.returns = None

    transformed = _NativeApplyRewriter(method.__globals__).visit(original)
    ast.fix_missing_locations(transformed)
    implementation = ast.unparse(transformed)
    class_name = entry.operation_type.__qualname__
    return (
        '"""Editable copy of an ArrayScope native operation.\n\n'
        f"Source implementation: {entry.operation_type.__module__}.{class_name}\n"
        '"""\n\n'
        "from arrayscope.operations import pipeline as _native\n\n\n"
        f"{implementation}\n"
    )


def adapter_template_source(entry: OperationEntry, callable_name: str) -> str:
    """Return a working, editable adapter for a pack or entry-point operation."""

    args = ["data"]
    if entry.requires_axis:
        args.append("axis")
    for parameter in entry.parameters:
        if parameter.default is None:
            args.append(parameter.name)
        else:
            args.append(f"{parameter.name}={parameter.default!r}")

    parameter_pairs = ", ".join(
        f"{parameter.name!r}: {parameter.name}" for parameter in entry.parameters
    )
    axis_expr = "axis" if entry.requires_axis else "None"
    return (
        '"""Editable adapter for an externally provided ArrayScope operation.\n\n'
        f"The starting implementation still depends on {entry.id!r}. Replace the\n"
        "body with your own implementation to make the copy independent.\n"
        '"""\n\n'
        "from arrayscope.operations.registry import create_operation as _create_operation\n\n\n"
        f"def {callable_name}({', '.join(args)}):\n"
        f"    parameters = {{{parameter_pairs}}}\n"
        f"    operation = _create_operation({entry.id!r}, axis={axis_expr}, "
        "parameters=parameters)\n"
        "    return operation.apply(data)\n"
    )


def empty_template_source(callable_name: str) -> str:
    """Source for a new, deliberately unfinished user operation."""

    message = "This new operation is empty. Open its code file and implement it before running."
    return (
        '"""New ArrayScope operation. Replace this docstring and implementation."""\n\n\n'
        f"def {callable_name}(data):\n"
        f"    raise NotImplementedError({message!r})\n"
    )


def _definition_from_entry(
    entry: OperationEntry, *, tier: str, runtime: str, source: dict[str, Any]
) -> dict[str, Any]:
    return {
        "format": "arrayscope-operation",
        "version": 1,
        "id": entry.id,
        "label": entry.label,
        "description": entry.description,
        "group": entry.group,
        "icon": entry.icon,
        "runtime": runtime,
        "source": source,
        "requires_axis": bool(entry.requires_axis),
        "changes_shape": bool(entry.changes_shape),
        "parameters": _parameter_payloads(entry.parameters),
        "input_slots": _input_slot_payloads(entry.input_slots),
        "unavailable_reason": entry.unavailable_reason,
        "tier": tier,
    }


def _parameter_payloads(parameters: tuple[OperationParameter, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": parameter.name,
            "label": parameter.label,
            "kind": parameter.kind,
            "default": parameter.default,
            "minimum": parameter.minimum,
            "maximum": parameter.maximum,
            "step": parameter.step,
            "description": parameter.description,
        }
        for parameter in parameters
    ]


def _input_slot_payloads(input_slots) -> list[dict[str, Any]]:
    return [
        {
            "name": slot.name,
            "label": slot.label,
            "description": slot.description,
            "accepts": list(slot.accepts),
        }
        for slot in input_slots
    ]


def _callable_module(fn) -> str:
    return str(getattr(fn, "__module__", "") or "")


def _literal_default(parameter: OperationParameter) -> ast.expr:
    if parameter.default is None:
        return ast.Constant(value=None)
    return ast.parse(repr(parameter.default), mode="eval").body


class _NativeApplyRewriter(ast.NodeTransformer):
    def __init__(self, method_globals: dict[str, Any]) -> None:
        self._globals = method_globals

    def visit_Attribute(self, node: ast.Attribute):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node

    def visit_Name(self, node: ast.Name):
        if (
            isinstance(node.ctx, ast.Load)
            and node.id != "self"
            and node.id in self._globals
            and not inspect.isbuiltin(self._globals[node.id])
        ):
            return ast.copy_location(
                ast.Attribute(
                    value=ast.Name(id="_native", ctx=ast.Load()),
                    attr=node.id,
                    ctx=node.ctx,
                ),
                node,
            )
        return node
