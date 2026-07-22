"""Tier-1 plugin-operation registry: discovery, laziness, recipe round-trip.

The exit gate for queue item 8 is "a third-party pip package contributes a
working reversible op including recipe round-trip".  We prove that here without
a real pip install by simulating the third-party package: a tiny module is
written to ``tmp_path`` and placed on ``sys.path``, and real
``importlib.metadata.EntryPoint`` objects pointing at it are fed to the
discovery layer via a monkeypatched ``entry_points``.  ``EntryPoint.load()``
therefore does a genuine import, so the laziness assertions exercise the real
import machinery, not a mock.
"""

from __future__ import annotations

import sys
from importlib.metadata import EntryPoint

import numpy as np
import pytest

from arrayscope.operations import plugins, recipes, registry
from arrayscope.operations.pipeline import ArrayDocument, OperationStep

GROUP = plugins.PLUGIN_ENTRY_POINT_GROUP

# A distinctive module name so ``sys.modules`` membership is an unambiguous
# import side-effect flag.
PLUGIN_MODULE = "arrayscope_demo_ops_fixture"

PLUGIN_SOURCE = '''
"""Simulated third-party ArrayScope operations package (test fixture)."""

import numpy as np

from arrayscope.operations.plugins import OperationParameter, PluginOperationSpec


def make_reverse_rows():
    def fn(array):
        return array[::-1]

    # Reversing axis 0 twice is the identity -> a self-inverse reversible op.
    return PluginOperationSpec(
        id="demo-ops:reverse_rows",
        label="Reverse rows (demo)",
        fn=fn,
    )


def make_roll():
    def build(axis, params):
        shift = int(params["shift"])
        target_axis = 0 if axis is None else int(axis)
        return lambda array: np.roll(array, shift, axis=target_axis)

    return PluginOperationSpec(
        id="demo-ops:roll",
        label="Roll (demo)",
        build=build,
        parameters=(OperationParameter("shift", "Shift"),),
        requires_axis=True,
    )


def make_drop_first():
    def fn(array):
        return array[1:]

    def output_shape(shape, axis, params):
        return (int(shape[0]) - 1, *tuple(int(size) for size in shape[1:]))

    return PluginOperationSpec(
        id="demo-ops:drop_first",
        label="Drop first row (demo)",
        fn=fn,
        output_shape=output_shape,
        changes_shape=True,
    )
'''


def _entry_point(name: str, attr: str) -> EntryPoint:
    return EntryPoint(name=name, value=f"{PLUGIN_MODULE}:{attr}", group=GROUP)


@pytest.fixture
def demo_plugin(tmp_path, monkeypatch):
    """Install a fake third-party plugin package and route discovery to it.

    Yields the tuple of namespaced ids the fixture advertises.
    """

    (tmp_path / f"{PLUGIN_MODULE}.py").write_text(PLUGIN_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))

    entry_points = (
        _entry_point("demo-ops:reverse_rows", "make_reverse_rows"),
        _entry_point("demo-ops:roll", "make_roll"),
        _entry_point("demo-ops:drop_first", "make_drop_first"),
    )

    def fake_entry_points(*, group=None):
        return list(entry_points) if group == GROUP else []

    monkeypatch.setattr(plugins, "entry_points", fake_entry_points)
    plugins._reset_plugin_cache()
    sys.modules.pop(PLUGIN_MODULE, None)

    yield tuple(sorted(ep.name for ep in entry_points))

    plugins._reset_plugin_cache()
    sys.modules.pop(PLUGIN_MODULE, None)


def test_plugin_ids_are_discovered_without_importing_the_plugin(demo_plugin):
    ids = plugins.plugin_operation_ids()
    assert ids == demo_plugin
    # The op is visible in the registry (get_operation_entry resolves it) but
    # nothing was loaded merely by enumerating names.
    assert PLUGIN_MODULE not in sys.modules, "discovery must not import the plugin module"

    # First actual use loads the module; the assertion above can therefore fail
    # if laziness regresses (see the direct-load guard below).
    operation = registry.create_operation("demo-ops:reverse_rows")
    assert PLUGIN_MODULE in sys.modules
    assert isinstance(operation, plugins.PluginOperation)


def test_lazy_load_is_falsifiable(demo_plugin):
    # Prove the laziness assertion is meaningful: touching load() DOES import.
    assert PLUGIN_MODULE not in sys.modules
    plugins.load_plugin_spec("demo-ops:reverse_rows")
    assert PLUGIN_MODULE in sys.modules


def test_plugin_operation_applies_to_ndarray(demo_plugin):
    data = np.arange(4 * 3).reshape(4, 3)
    operation = registry.create_operation("demo-ops:reverse_rows")
    np.testing.assert_array_equal(operation.apply(data), data[::-1])


def test_reverse_rows_is_reversible(demo_plugin):
    data = np.arange(4 * 3).reshape(4, 3)
    reverse = registry.create_operation("demo-ops:reverse_rows")
    round_tripped = reverse.apply(reverse.apply(data))
    np.testing.assert_array_equal(round_tripped, data)


def test_parametric_forward_inverse_round_trips_data(demo_plugin):
    data = np.arange(5 * 2).reshape(5, 2)
    forward = registry.create_operation("demo-ops:roll", axis=0, parameters={"shift": 2})
    inverse = registry.create_operation("demo-ops:roll", axis=0, parameters={"shift": -2})
    np.testing.assert_array_equal(inverse.apply(forward.apply(data)), data)


def test_recipe_round_trip_reconstructs_equal_plugin_step(demo_plugin):
    steps = (
        OperationStep(registry.create_operation("demo-ops:reverse_rows")),
        OperationStep(registry.create_operation("demo-ops:roll", axis=0, parameters={"shift": 2})),
    )

    text = recipes.dumps_recipe(steps)
    loaded = recipes.loads_recipe_steps(text, base_shape=(5, 3))

    assert tuple(step.operation for step in loaded) == tuple(step.operation for step in steps)
    # The serialized item carries the namespaced id + params, nothing else.
    recipe = recipes.recipe_from_steps(steps)
    assert recipe["operations"][0]["id"] == "demo-ops:reverse_rows"
    assert recipe["operations"][1]["parameters"] == {"shift": 2}


def test_shape_changing_plugin_op_uses_its_adapter(demo_plugin):
    data = np.arange(4 * 3).reshape(4, 3)
    document = ArrayDocument(data).with_operation(registry.create_operation("demo-ops:drop_first"))
    assert document.current_shape == (3, 3)
    np.testing.assert_array_equal(document.materialize(), data[1:])


def test_plugin_op_flows_through_the_opaque_region_engine(demo_plugin):
    # Drive the operation through the display/slab engine (OperationEvaluator),
    # which exercises required_input_region + apply_to_region -- the existing
    # opaque whole-array materialization path, not a parallel one.  The result
    # must match materializing the same transform into the base array (a
    # convention-independent oracle), proving the region path is faithful.
    from arrayscope.core.view_state import ViewState
    from arrayscope.operations.evaluator import OperationEvaluator

    data = np.arange(3 * 4 * 5).reshape(3, 4, 5).astype(float)
    plugin_document = ArrayDocument(data).with_operation(
        registry.create_operation("demo-ops:roll", axis=0, parameters={"shift": 1})
    )
    oracle_document = ArrayDocument(np.roll(data, 1, axis=0))
    state = ViewState.from_shape(plugin_document.current_shape)

    plugin_image = OperationEvaluator(plugin_document).image(state)
    oracle_image = OperationEvaluator(oracle_document).image(state)
    np.testing.assert_allclose(np.asarray(plugin_image.data), np.asarray(oracle_image.data))


def test_uninstalled_plugin_recipe_raises_clear_error(demo_plugin):
    recipe = {
        "version": recipes.RECIPE_VERSION,
        "operations": [{"id": "not-installed:mystery", "axis": 0, "enabled": True}],
    }
    with pytest.raises(ValueError, match="uninstalled plugin operation"):
        recipes.steps_from_recipe(recipe, base_shape=(3, 3))


def test_plugin_cannot_hijack_a_builtin_id(monkeypatch, caplog):
    # A plugin advertising a built-in's flat id ("crop") is rejected -- the
    # namespace rule already forbids it -- and the built-in still resolves.
    hijack = EntryPoint(name="crop", value=f"{PLUGIN_MODULE}:make_reverse_rows", group=GROUP)

    def fake_entry_points(*, group=None):
        return [hijack] if group == GROUP else []

    monkeypatch.setattr(plugins, "entry_points", fake_entry_points)
    plugins._reset_plugin_cache()

    with caplog.at_level("WARNING"):
        assert plugins.plugin_operation_ids() == ()
    assert any("crop" in record.message for record in caplog.records)
    assert registry.get_operation_entry("crop").operation_type.__name__ == "Crop"


def test_namespaced_id_colliding_with_a_builtin_is_rejected_loudly(monkeypatch, caplog):
    # Defense in depth: even a *namespaced* id is rejected if it collides with
    # a built-in id (should the built-in id space ever grow a separator).
    monkeypatch.setattr(plugins, "_builtin_operation_ids", lambda: frozenset({"vendor:op"}))
    collide = EntryPoint(name="vendor:op", value=f"{PLUGIN_MODULE}:make_reverse_rows", group=GROUP)

    def fake_entry_points(*, group=None):
        return [collide] if group == GROUP else []

    monkeypatch.setattr(plugins, "entry_points", fake_entry_points)
    plugins._reset_plugin_cache()

    with caplog.at_level("WARNING"):
        assert plugins.plugin_operation_ids() == ()
    assert any("shadows a built-in" in record.message for record in caplog.records)


def test_un_namespaced_entry_point_is_rejected_loudly(monkeypatch, caplog):
    flat = EntryPoint(name="reverse_rows", value=f"{PLUGIN_MODULE}:make_reverse_rows", group=GROUP)

    def fake_entry_points(*, group=None):
        return [flat] if group == GROUP else []

    monkeypatch.setattr(plugins, "entry_points", fake_entry_points)
    plugins._reset_plugin_cache()

    with caplog.at_level("WARNING"):
        assert plugins.plugin_operation_ids() == ()
    assert any("un-namespaced" in record.message for record in caplog.records)
