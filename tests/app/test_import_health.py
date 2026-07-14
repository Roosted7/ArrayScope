import ast
import importlib
import pkgutil
from pathlib import Path

import arrayscope


ROOT = Path(__file__).parents[2]


def test_every_arrayscope_module_imports():
    failures = []
    modules = sorted(
        module.name
        for module in pkgutil.walk_packages(
            arrayscope.__path__,
            prefix=f"{arrayscope.__name__}.",
        )
    )
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except BaseException as exc:  # noqa: BLE001 - the guard reports every broken module
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
    assert failures == []


def test_internal_imports_are_not_silently_swallowed():
    """An internal import failure must either propagate or be reported.

    Broad exception handlers may still protect an explicitly reported UI or
    optional-backend boundary.  A handler with no call and no re-raise is a
    silent fallback: it hid both dead ``frame_renderer`` imports after R7.
    """

    offenders = []
    for path in sorted((ROOT / "arrayscope").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            internal_imports = []
            for statement in node.body:
                for child in ast.walk(statement):
                    if isinstance(child, ast.ImportFrom) and str(child.module or "").startswith("arrayscope"):
                        internal_imports.append(child)
                    elif isinstance(child, ast.Import) and any(
                        alias.name.startswith("arrayscope") for alias in child.names
                    ):
                        internal_imports.append(child)
            if not internal_imports:
                continue
            for handler in node.handlers:
                broad = handler.type is None or (
                    isinstance(handler.type, ast.Name)
                    and handler.type.id in {"Exception", "BaseException"}
                )
                if not broad:
                    continue
                reported_or_raised = any(
                    isinstance(child, (ast.Call, ast.Raise))
                    for statement in handler.body
                    for child in ast.walk(statement)
                )
                if reported_or_raised:
                    continue
                for imported in internal_imports:
                    offenders.append(f"{path.relative_to(ROOT)}:{imported.lineno}")
    assert offenders == []
