import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
PACKAGE = types.ModuleType("arrayscope")
PACKAGE.__path__ = [str(ROOT / "arrayscope")]
sys.modules.setdefault("arrayscope", PACKAGE)


def load_module(name):
    path = ROOT / "arrayscope" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"arrayscope.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


operation_pipeline = load_module("operation_pipeline")
operation_stack = load_module("operation_stack")

Crop = operation_pipeline.Crop
Mean = operation_pipeline.Mean
ReverseAxis = operation_pipeline.ReverseAxis


def test_delete_operation_validates_resulting_stack():
    operations = (Crop(axis=1, start=0, stop=2), ReverseAxis(axis=0))

    updated = operation_stack.delete_operation(operations, 0, base_shape=(3, 4))

    assert updated == (ReverseAxis(axis=0),)


def test_move_operation_rejects_invalid_reorder_without_returning_bad_stack():
    operations = (ReverseAxis(axis=1), Mean(axis=0))

    with pytest.raises(ValueError, match="out of bounds"):
        operation_stack.move_operation(operations, 1, -1, base_shape=(2, 3))


def test_move_operation_returns_same_stack_at_bounds():
    operations = (ReverseAxis(axis=0), Crop(axis=1, start=0, stop=2))

    assert operation_stack.move_operation(operations, 0, -1, base_shape=(3, 4)) == operations


def test_reorder_operations_validates_complete_order():
    operations = (ReverseAxis(axis=0), Crop(axis=1, start=0, stop=2))

    assert operation_stack.reorder_operations(operations, (1, 0), base_shape=(3, 4)) == (
        Crop(axis=1, start=0, stop=2),
        ReverseAxis(axis=0),
    )

    with pytest.raises(ValueError, match="each operation index exactly once"):
        operation_stack.reorder_operations(operations, (0, 0), base_shape=(3, 4))
