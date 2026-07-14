import ast
import importlib
import pkgutil
from pathlib import Path

import arrayscope


ROOT = Path(__file__).parents[2]

# Handler types that can swallow a dead internal import.  ImportError and
# ModuleNotFoundError are included deliberately: `except ImportError: pass`
# around an internal import is exactly the deleted-module hazard this guard
# exists for (both dead ``frame_renderer`` imports after R7 would have been
# equally silent behind ImportError).
_SWALLOWING_HANDLER_NAMES = {
    "Exception",
    "BaseException",
    "ImportError",
    "ModuleNotFoundError",
}


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


def _handler_swallows_imports(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(
        isinstance(node, ast.Name) and node.id in _SWALLOWING_HANDLER_NAMES
        for node in types
    )


def swallowed_internal_imports(root: Path) -> list[str]:
    """Find internal imports whose failure a handler can hide without a report.

    A handler may still protect an explicitly reported UI or optional-backend
    boundary.  A handler with no call and no re-raise is a silent fallback: it
    hid both dead ``frame_renderer`` imports after R7.
    """

    offenders = []
    for path in sorted(root.rglob("*.py")):
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
                if not _handler_swallows_imports(handler):
                    continue
                reported_or_raised = any(
                    isinstance(child, (ast.Call, ast.Raise))
                    for statement in handler.body
                    for child in ast.walk(statement)
                )
                if reported_or_raised:
                    continue
                for imported in internal_imports:
                    offenders.append(f"{path.relative_to(root.parent)}:{imported.lineno}")
    return offenders


def test_internal_imports_are_not_silently_swallowed():
    assert swallowed_internal_imports(ROOT / "arrayscope") == []


def test_guard_rejects_import_error_and_tuple_handlers(tmp_path):
    """The scanner must flag every handler form that can hide a dead import."""

    package = tmp_path / "arrayscope"
    package.mkdir()
    (package / "swallow_import_error.py").write_text(
        "def probe():\n"
        "    try:\n"
        "        from arrayscope.window.frame_renderer import gone\n"
        "        return gone\n"
        "    except ImportError:\n"
        "        return None\n"
    )
    (package / "swallow_tuple.py").write_text(
        "def probe():\n"
        "    try:\n"
        "        import arrayscope.window.frame_renderer as gone\n"
        "        return gone\n"
        "    except (RuntimeError, Exception):\n"
        "        return None\n"
    )
    (package / "swallow_bare.py").write_text(
        "def probe():\n"
        "    try:\n"
        "        from arrayscope.window.frame_renderer import gone\n"
        "        return gone\n"
        "    except:\n"
        "        return None\n"
    )
    (package / "reported_boundary.py").write_text(
        "import logging\n"
        "def probe():\n"
        "    try:\n"
        "        from arrayscope.window.frame_renderer import gone\n"
        "        return gone\n"
        "    except ImportError:\n"
        "        logging.getLogger(__name__).warning('optional path missing')\n"
        "        return None\n"
    )
    (package / "narrow_handler.py").write_text(
        "def probe():\n"
        "    try:\n"
        "        from arrayscope.core.trace import TRACE\n"
        "        return TRACE\n"
        "    except KeyError:\n"
        "        return None\n"
    )

    offenders = swallowed_internal_imports(package)

    flagged_files = {entry.split(":")[0].split("/")[-1] for entry in offenders}
    assert flagged_files == {
        "swallow_import_error.py",
        "swallow_tuple.py",
        "swallow_bare.py",
    }
