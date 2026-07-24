"""Contract tests for the operation library (grouping / layout / user ops)."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from arrayscope.operations import library, recipes, registry


@pytest.fixture(autouse=True)
def _isolated_ops_dir(tmp_path, monkeypatch):
    ops_dir = tmp_path / "operations"
    monkeypatch.setattr(library, "user_operations_directory", lambda: str(ops_dir))
    library.refresh_user_operations()
    yield
    library.refresh_user_operations()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_source(tmp_path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(body)
    return str(path)


_DOUBLE_SRC = '''
def double(data):
    """Double every sample."""
    return data * 2
'''

_SHIFT_SRC = '''
import numpy as np


def shift(data, axis, amount: int = 1):
    """Roll one axis by amount."""
    return np.roll(data, amount, axis=axis)
'''


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def test_grouped_operations_default_order_follows_group_taxonomy():
    groups = [group for group, _entries in library.grouped_operations()]
    # Reduce precedes Transform precedes Complex per DEFAULT_GROUP_ORDER.
    assert groups.index("Reduce") < groups.index("Transform") < groups.index("Complex")


def test_layout_round_trip_group_and_op_order():
    library.apply_library_layout(
        group_order=["Complex", "Reduce", "Transform"],
        op_groups={"mean": "Transform"},
        op_order={"sum": 0, "mean": 1},
    )
    groups = library.grouped_operations()
    order = [group for group, _entries in groups]
    assert order[0] == "Complex"
    # mean was reassigned into Transform.
    transform = dict(groups)["Transform"]
    assert "mean" in {entry.id for entry in transform}


def test_effective_common_and_more_round_trip_and_reset():
    assert library.effective_common_ids() == tuple(registry.COMMON_OPERATION_IDS)
    assert library.effective_more_groups() == library.DEFAULT_MORE_GROUPS

    library.apply_library_layout(common_ids=["crop", "mean"], more_groups=["User", "Other"])
    assert library.effective_common_ids() == ("crop", "mean")
    assert library.effective_more_groups() == ("User", "Other")

    library.reset_layout()
    assert library.effective_common_ids() == tuple(registry.COMMON_OPERATION_IDS)
    assert library.effective_more_groups() == library.DEFAULT_MORE_GROUPS


# ---------------------------------------------------------------------------
# hidden ops
# ---------------------------------------------------------------------------


def test_hidden_operations_excluded_and_reset_unhides():
    library.set_operation_hidden("mean", True)
    assert "mean" in library.hidden_operations()
    visible = {entry.id for _group, entries in library.grouped_operations() for entry in entries}
    assert "mean" not in visible
    # include_hidden surfaces it again.
    with_hidden = {
        entry.id
        for _group, entries in library.grouped_operations(include_hidden=True)
        for entry in entries
    }
    assert "mean" in with_hidden

    assert library.reset_operation("mean") is True
    assert "mean" not in library.hidden_operations()
    assert library.reset_operation("mean") is False  # already un-hidden


# ---------------------------------------------------------------------------
# listeners
# ---------------------------------------------------------------------------


def test_listeners_notified_on_every_mutation(tmp_path):
    calls = []
    library.add_library_listener(lambda: calls.append(1))
    try:
        library.set_operation_hidden("mean", True)
        library.apply_library_layout(group_order=["Reduce"])
        library.reset_layout()
        src = _write_source(tmp_path, "d.py", _DOUBLE_SRC)
        op_id = library.import_custom_operation(src, "double")
        library.remove_user_operation(op_id)
        library.refresh_user_operations()
    finally:
        library.remove_library_listener(lambda: None)  # idempotent no-op
    assert len(calls) >= 6

    library._listeners.clear()


def test_dead_qt_listener_is_self_healed():
    def boom():
        raise RuntimeError("underlying C++ object deleted")

    library.add_library_listener(boom)
    library.set_operation_hidden("mean", True)  # triggers _notify -> drops boom
    assert boom not in library._listeners


# ---------------------------------------------------------------------------
# user ops: end to end
# ---------------------------------------------------------------------------


def test_import_double_registers_and_applies(tmp_path):
    src = _write_source(tmp_path, "d.py", _DOUBLE_SRC)
    op_id = library.import_custom_operation(src, "double")
    assert op_id == "user:double"

    entry = registry.get_operation_entry(op_id)
    assert entry.requires_axis is False
    assert entry.group == "User"
    assert op_id in {entry.id for entry in registry.all_operations()}

    operation = registry.create_operation(op_id)
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert np.allclose(operation.apply(data), data * 2)


def test_import_shift_with_axis_and_param(tmp_path):
    src = _write_source(tmp_path, "s.py", _SHIFT_SRC)
    op_id = library.import_custom_operation(src, "shift")

    entry = registry.get_operation_entry(op_id)
    assert entry.requires_axis is True
    assert [p.name for p in entry.parameters] == ["amount"]
    assert entry.parameters[0].kind == "int"
    assert entry.parameters[0].default == 1

    operation = registry.create_operation(op_id, axis=2, parameters={"amount": 2})
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert np.allclose(operation.apply(data), np.roll(data, 2, axis=2))


def test_user_op_recipe_round_trip(tmp_path):
    src = _write_source(tmp_path, "s.py", _SHIFT_SRC)
    op_id = library.import_custom_operation(src, "shift")
    operation = registry.create_operation(op_id, axis=1, parameters={"amount": 2})

    recipe = recipes.recipe_from_operations([operation])
    data = np.zeros((2, 3, 4), dtype=np.float32)
    restored = recipes.operations_from_recipe(recipe, data.shape)
    assert len(restored) == 1
    probe = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert np.allclose(restored[0].apply(probe), operation.apply(probe))

    described = registry.describe_operation(operation)
    assert "amount=2" in described


def test_import_mode_copies_code_file(tmp_path):
    src = _write_source(tmp_path, "orig.py", _DOUBLE_SRC)
    op_id = library.import_custom_operation(src, "double")
    ops_dir = library.user_operations_directory()
    assert os.path.exists(os.path.join(ops_dir, "double.py"))
    assert os.path.exists(os.path.join(ops_dir, "double.json"))

    # Deleting the original source does not break an imported op (self-contained).
    os.remove(src)
    library.refresh_user_operations()
    operation = registry.create_operation(op_id)
    assert np.allclose(operation.apply(np.ones((2, 2), dtype=np.float32)), 2.0)


def test_link_mode_picks_up_edited_file(tmp_path):
    src = _write_source(tmp_path, "linked.py", "def scale(data):\n    return data * 3\n")
    op_id = library.import_custom_operation(src, "scale", link=True)
    data = np.ones((2, 2), dtype=np.float32)
    assert np.allclose(registry.create_operation(op_id).apply(data), 3.0)

    # Edit the linked file and bump its mtime -> re-import picks up the change.
    with open(src, "w") as handle:
        handle.write("def scale(data):\n    return data * 5\n")
    os.utime(src, (time.time() + 2, time.time() + 2))
    assert np.allclose(registry.create_operation(op_id).apply(data), 5.0)


def test_broken_code_file_is_skipped_and_recorded(tmp_path):
    # A wrapper whose code file has a syntax error is skipped, recorded, and
    # never registered -- and the rest of the library still loads.
    ops_dir = tmp_path / "operations"
    ops_dir.mkdir(parents=True, exist_ok=True)
    (ops_dir / "broken.py").write_text("def oops(data)\n    return data\n")  # syntax error
    (ops_dir / "broken.json").write_text(
        json.dumps(
            {
                "format": "arrayscope-operation",
                "version": 1,
                "id": "user:broken",
                "source": {"mode": "import", "path": "broken.py", "callable": "oops"},
            }
        )
    )
    library.refresh_user_operations()

    assert "user:broken" not in {entry.id for entry in registry.all_operations()}
    problems = library.user_operation_problems()
    assert any("broken.json" in path for path, _message in problems)
    # A good op alongside the broken one still loads.
    good = _write_source(tmp_path, "g.py", _DOUBLE_SRC)
    library.import_custom_operation(good, "double")
    assert "user:double" in {entry.id for entry in registry.all_operations()}


def test_reserved_runtime_is_skipped_with_problem(tmp_path):
    ops_dir = tmp_path / "operations"
    ops_dir.mkdir(parents=True, exist_ok=True)
    (ops_dir / "jl.json").write_text(
        json.dumps(
            {
                "format": "arrayscope-operation",
                "version": 1,
                "id": "user:jl",
                "runtime": "julia",
                "source": {"mode": "import", "path": "jl.py", "callable": "run"},
            }
        )
    )
    library.refresh_user_operations()

    assert "user:jl" not in {entry.id for entry in registry.all_operations()}
    problems = library.user_operation_problems()
    assert any("not yet supported" in message and "julia" in message for _path, message in problems)


def test_unsupported_parameter_kind_is_skipped_with_problem(tmp_path):
    # A parameter whose kind is neither "int" nor "float" would crash later in
    # form building / coercion, so the wrapper is skipped and recorded instead.
    ops_dir = tmp_path / "operations"
    ops_dir.mkdir(parents=True, exist_ok=True)
    (ops_dir / "named.py").write_text(_DOUBLE_SRC)
    (ops_dir / "named.json").write_text(
        json.dumps(
            {
                "format": "arrayscope-operation",
                "version": 1,
                "id": "user:named",
                "source": {"mode": "import", "path": "named.py", "callable": "double"},
                "parameters": [{"name": "tag", "kind": "str"}],
            }
        )
    )
    library.refresh_user_operations()

    assert "user:named" not in {entry.id for entry in registry.all_operations()}
    problems = library.user_operation_problems()
    assert any(
        "unsupported parameter kind" in message and "named.json" in path
        for path, message in problems
    )


def test_changes_shape_wrapper_is_skipped_with_problem(tmp_path):
    # A wrapper cannot predict the output shape, so a shape-changing user op
    # would lie to the evaluator: it is skipped, recorded, and never registered.
    ops_dir = tmp_path / "operations"
    ops_dir.mkdir(parents=True, exist_ok=True)
    (ops_dir / "grow.py").write_text(_DOUBLE_SRC)
    (ops_dir / "grow.json").write_text(
        json.dumps(
            {
                "format": "arrayscope-operation",
                "version": 1,
                "id": "user:grow",
                "changes_shape": True,
                "source": {"mode": "import", "path": "grow.py", "callable": "double"},
            }
        )
    )
    library.refresh_user_operations()

    assert "user:grow" not in {entry.id for entry in registry.all_operations()}
    problems = library.user_operation_problems()
    assert any(
        "shape-changing" in message and "grow.json" in path for path, message in problems
    )


def test_import_custom_operation_rejects_changes_shape(tmp_path):
    src = _write_source(tmp_path, "d.py", _DOUBLE_SRC)
    with pytest.raises(ValueError, match="shape-changing"):
        library.import_custom_operation(src, "double", changes_shape=True)
    # Nothing was written to disk for the rejected op.
    assert not os.path.exists(os.path.join(library.user_operations_directory(), "double.json"))


def test_remove_user_operation_deletes_files(tmp_path):
    src = _write_source(tmp_path, "d.py", _DOUBLE_SRC)
    op_id = library.import_custom_operation(src, "double")
    ops_dir = library.user_operations_directory()
    assert os.path.exists(os.path.join(ops_dir, "double.py"))

    assert library.remove_user_operation(op_id) is True
    assert not os.path.exists(os.path.join(ops_dir, "double.json"))
    assert not os.path.exists(os.path.join(ops_dir, "double.py"))
    assert op_id not in {entry.id for entry in registry.all_operations()}
    assert library.remove_user_operation(op_id) is False  # already gone


def test_update_user_operation_rewrites_wrapper(tmp_path):
    src = _write_source(tmp_path, "d.py", _DOUBLE_SRC)
    op_id = library.import_custom_operation(src, "double")

    assert library.update_user_operation(op_id, label="Twice", description="x2") is True
    entry = registry.get_operation_entry(op_id)
    assert entry.label == "Twice"
    assert entry.description == "x2"


def test_slug_collision_suffixes(tmp_path):
    src = _write_source(tmp_path, "d.py", _DOUBLE_SRC)
    id1 = library.import_custom_operation(src, "double", label="Boost")
    id2 = library.import_custom_operation(src, "double", label="Boost")
    assert id1 != id2
    assert {id1, id2} <= {entry.id for entry in registry.all_operations()}


# ---------------------------------------------------------------------------
# introspection (ast-only)
# ---------------------------------------------------------------------------


def test_introspection_reads_kinds_defaults_and_axis(tmp_path):
    src = _write_source(
        tmp_path,
        "m.py",
        (
            "def op(data, axis, count: int = 3, weight: float = 0.5, ratio=1.5):\n"
            '    """One-liner doc.\n\n    More.\n    """\n'
            "    return data\n"
        ),
    )
    infos = {info.name: info for info in library.introspect_python_source(src)}
    info = infos["op"]
    assert info.has_axis is True
    assert info.doc == "One-liner doc."
    by_name = {p.name: p for p in info.params}
    assert set(by_name) == {"count", "weight", "ratio"}
    assert by_name["count"].kind == "int"
    assert by_name["count"].default == 3
    assert by_name["weight"].kind == "float"
    assert by_name["weight"].default == 0.5
    # kind guessed from the default value when there is no annotation.
    assert by_name["ratio"].kind == "float"
    assert by_name["ratio"].default == 1.5


def test_introspection_works_on_file_that_raises_at_import(tmp_path):
    src = _write_source(
        tmp_path,
        "raiser.py",
        "raise RuntimeError('boom at import')\n\n\ndef good(data, threshold: float = 0.1):\n    return data\n",
    )
    # ast-only: never executes the module, so the top-level raise is irrelevant.
    infos = {info.name: info for info in library.introspect_python_source(src)}
    assert "good" in infos
    assert infos["good"].params[0].name == "threshold"


# ---------------------------------------------------------------------------
# smoke-harness compatibility
# ---------------------------------------------------------------------------


def test_user_op_passes_smoke_harness_checks(tmp_path):
    """A registered user op behaves like any op the smoke harness iterates."""

    from arrayscope.operations.parameter_forms import build_parameter_form

    src = _write_source(tmp_path, "s.py", _SHIFT_SRC)
    op_id = library.import_custom_operation(src, "shift")

    entries = {entry.id: entry for entry in registry.all_operations()}
    assert op_id in entries
    entry = entries[op_id]

    shape = (6, 8, 10)
    axis = 1
    form = build_parameter_form(entry, shape=shape, axis=axis)
    parameters = form.values() if form is not None else {}
    if form is not None:
        assert form.validate() is None

    operation = registry.create_operation(op_id, axis=axis, parameters=parameters)
    data = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    predicted_shape = tuple(operation.output_shape(shape))
    predicted_dtype = operation.output_dtype(np.dtype(np.float32))
    capabilities = operation.capabilities(shape, np.dtype(np.float32))
    result = np.asarray(operation.apply(data))

    assert capabilities is not None
    assert result.shape == predicted_shape
    if predicted_dtype is not None:
        assert result.dtype == np.dtype(predicted_dtype)
