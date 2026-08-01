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
from arrayscope.operations.operation_definitions import export_operation_definition
from arrayscope.operations.pipeline import ArrayDocument, OperationStep
from tests import seams

GROUP = plugins.PLUGIN_ENTRY_POINT_GROUP

# A distinctive module name so ``sys.modules`` membership is an unambiguous
# import side-effect flag.
PLUGIN_MODULE = "arrayscope_demo_ops_fixture"

PLUGIN_SOURCE = '''
"""Simulated third-party ArrayScope operations package (test fixture)."""

import numpy as np

from arrayscope.operations.plugins import OperationParameter, PluginOperationSpec

from tests import seams


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


def make_offset_by_mode():
    """An op whose result depends on an enum, a bool, and a string parameter."""

    STEPS = {"none": 0.0, "one": 1.0, "ten": 10.0}

    def build(axis, params):
        step = STEPS[params["mode"]]
        sign = -1.0 if params["negate"] else 1.0
        marker = float(len(params["note"]))
        return lambda array: array + sign * step + marker

    return PluginOperationSpec(
        id="demo-ops:offset_by_mode",
        label="Offset by mode (demo)",
        build=build,
        parameters=(
            OperationParameter(
                "mode", "Mode", kind="enum", choices=("none", "one", "ten"), default="one"
            ),
            OperationParameter("negate", "Negate", kind="bool", default=False),
            OperationParameter("note", "Note", kind="string", default=""),
        ),
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
        _entry_point("demo-ops:offset_by_mode", "make_offset_by_mode"),
    )

    def fake_entry_points(*, group=None):
        return list(entry_points) if group == GROUP else []

    monkeypatch.setattr(plugins, "entry_points", fake_entry_points)
    seams.reset_plugin_cache()
    sys.modules.pop(PLUGIN_MODULE, None)

    yield tuple(sorted(ep.name for ep in entry_points))

    seams.reset_plugin_cache()
    sys.modules.pop(PLUGIN_MODULE, None)


def test_plugin_ids_are_discovered_without_importing_the_plugin(demo_plugin):
    ids = tuple(sorted(plugins.discover_plugin_entry_points()))
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


def test_entry_point_plugin_exports_provider_and_interface(demo_plugin):
    definition = export_operation_definition("demo-ops:roll")

    assert definition["tier"] == "plugin"
    assert definition["source"] == {
        "mode": "entry-point",
        "group": GROUP,
        "name": "demo-ops:roll",
        "value": f"{PLUGIN_MODULE}:make_roll",
        "distribution": "",
    }
    assert definition["requires_axis"] is True
    assert definition["parameters"][0]["name"] == "shift"


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


def test_recipe_round_trips_enum_bool_and_string_parameters(demo_plugin):
    """Non-numeric parameter values survive JSON serialization unconverted.

    A recipe stores parameters by name, so nothing about the kinds is written
    down: a bool that came back as the string ``"True"``, or an enum silently
    floated, would rebuild a *different* operation from the same file.
    """

    data = np.zeros((3, 2))
    operation = registry.create_operation(
        "demo-ops:offset_by_mode",
        parameters={"mode": "ten", "negate": True, "note": "k-space"},
    )
    np.testing.assert_allclose(operation.apply(data), data - 10.0 + 7.0)

    text = recipes.dumps_recipe((OperationStep(operation),))
    assert '"negate": true' in text
    assert '"mode": "ten"' in text

    (rebuilt,) = recipes.loads_recipe_steps(text, base_shape=(3, 2))
    assert rebuilt.operation == operation
    assert dict(rebuilt.operation.params) == {
        "mode": "ten",
        "negate": True,
        "note": "k-space",
    }
    np.testing.assert_allclose(rebuilt.operation.apply(data), operation.apply(data))


def test_recipe_with_an_unknown_enum_choice_is_refused_not_clamped(demo_plugin):
    recipe = {
        "version": recipes.RECIPE_VERSION,
        "operations": [
            {"id": "demo-ops:offset_by_mode", "parameters": {"mode": "hundred"}, "enabled": True}
        ],
    }
    with pytest.raises(ValueError, match="none, one, ten"):
        recipes.steps_from_recipe(recipe, base_shape=(3, 2))


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
    seams.reset_plugin_cache()

    with caplog.at_level("WARNING"):
        assert plugins.discover_plugin_entry_points() == {}
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
    seams.reset_plugin_cache()

    with caplog.at_level("WARNING"):
        assert plugins.discover_plugin_entry_points() == {}
    assert any("shadows a built-in" in record.message for record in caplog.records)


def test_un_namespaced_entry_point_is_rejected_loudly(monkeypatch, caplog):
    flat = EntryPoint(name="reverse_rows", value=f"{PLUGIN_MODULE}:make_reverse_rows", group=GROUP)

    def fake_entry_points(*, group=None):
        return [flat] if group == GROUP else []

    monkeypatch.setattr(plugins, "entry_points", fake_entry_points)
    seams.reset_plugin_cache()

    with caplog.at_level("WARNING"):
        assert plugins.discover_plugin_entry_points() == {}
    assert any("un-namespaced" in record.message for record in caplog.records)


# --- slab-mapped operations ("apply per slab, broadcast over the rest") -------
#
# The specs are installed straight into the cache through the documented
# ``seams.reset_plugin_cache`` seam: the subject here is execution and capability
# declaration, not entry-point discovery, which the fixture above already owns.


@pytest.fixture
def spec_cache():
    seams.reset_operation_packs()
    seams.reset_plugin_cache()
    yield None
    seams.reset_operation_packs()
    seams.reset_plugin_cache()


def _install(spec_cache, spec):
    """Register an in-process spec the way a first-party pack does."""

    del spec_cache
    registry.register_pack_operation(spec)
    return spec


def test_a_slab_mapped_op_runs_per_slab_and_broadcasts_over_the_rest(spec_cache):
    seen = []

    def build(axis, params, slots):
        del axis, params, slots

        def fn(slab):
            # The callable sees a SLAB spanning exactly the selected axes -- a
            # 2-D image here -- never the volume.
            seen.append(tuple(slab.shape))
            return slab - slab.mean()

        return fn

    _install(
        spec_cache,
        plugins.PluginOperationSpec(
            id="demo-ops:demean_slab",
            label="Demean each slab",
            build=build,
            requires_axes=True,
            maps_over_slabs=True,
        ),
    )
    operation = registry.create_operation("demo-ops:demean_slab", axes=(1, 2), ndim=3)

    rng = np.random.default_rng(0x51AB)
    data = rng.standard_normal((4, 5, 6)).astype(np.float32)
    result = operation.apply(data)

    assert operation.output_shape(data.shape) == (4, 5, 6)
    expected = data - data.mean(axis=(1, 2), keepdims=True)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    # Four slabs of the two selected axes, each handed over whole.
    assert seen[-4:] == [(5, 6)] * 4


def test_a_slab_mapped_op_maps_non_adjacent_axes_in_the_arrays_own_order(spec_cache):
    _install(
        spec_cache,
        plugins.PluginOperationSpec(
            id="demo-ops:transpose_slab",
            label="Transpose each slab",
            # A slab operation that is *not* symmetric, so a lost or reversed
            # moveaxis shows up as wrong values rather than a lucky pass.
            fn=lambda slab: slab.T,
            requires_axes=True,
            maps_over_slabs=True,
        ),
    )
    operation = registry.create_operation("demo-ops:transpose_slab", axes=(0, 2), ndim=3)

    data = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)
    result = operation.apply(data)

    expected = np.stack([data[:, index, :].T for index in range(4)], axis=1)
    np.testing.assert_array_equal(result, expected)


def test_a_shape_preserving_slab_op_blocks_only_its_own_axes(spec_cache):
    _install(
        spec_cache,
        plugins.PluginOperationSpec(
            id="demo-ops:slab_identity",
            label="Slab identity",
            fn=lambda slab: slab * 2.0,
            requires_axes=True,
            maps_over_slabs=True,
        ),
    )
    operation = registry.create_operation("demo-ops:slab_identity", axes=(1, 2), ndim=3)

    capabilities = operation.capabilities((4, 5, 6), np.float32)

    assert capabilities.blocking_axes == (1, 2)
    assert capabilities.expands_request_axes == (1, 2)
    # The complement is chunkable -- this is the whole difference from the
    # incumbent Tier-1 declaration, which blocked every axis.
    assert capabilities.chunkable_axes == (0,)
    assert capabilities.cache_stage is True


def test_a_slab_resizing_op_falls_back_to_the_whole_array_declaration(spec_cache):
    """The honest fallback branch: still correct, just not chunkable.

    A callable that resizes its slab still produces a well-defined whole-array
    result -- the applied axes change length and nothing else does -- so it
    runs. What it does not get is the broadcast-chunkable declaration, because
    chunk placement would then depend on the fitted shape rule.
    """

    _install(
        spec_cache,
        plugins.PluginOperationSpec(
            id="demo-ops:crop_slab",
            label="Crop each slab",
            fn=lambda slab: slab[1:, 1:],
            requires_axes=True,
            maps_over_slabs=True,
            changes_shape=True,
        ),
    )
    operation = registry.create_operation("demo-ops:crop_slab", axes=(1, 2), ndim=3)

    capabilities = operation.capabilities((4, 5, 6), np.float32)
    assert capabilities.blocking_axes == (0, 1, 2)
    assert capabilities.chunkable_axes == ()
    assert capabilities.cache_stage is True

    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    assert operation.output_shape((4, 5, 6)) == (4, 4, 5)
    np.testing.assert_array_equal(operation.apply(data), data[:, 1:, 1:])


def test_a_slab_mapped_op_that_changes_the_slab_rank_is_refused(spec_cache):
    from arrayscope.operations.plugin_conformance import CharacterizationUnavailable

    _install(
        spec_cache,
        plugins.PluginOperationSpec(
            id="demo-ops:flatten_slab",
            label="Flatten each slab",
            fn=lambda slab: slab.reshape(-1),
            requires_axes=True,
            maps_over_slabs=True,
            changes_shape=True,
        ),
    )
    operation = registry.create_operation("demo-ops:flatten_slab", axes=(1, 2), ndim=3)

    # There is no position in the array for a slab that lost an axis -- refuse
    # rather than invent one.
    with pytest.raises(CharacterizationUnavailable, match="slab rank"):
        operation.output_shape((4, 5, 6))


def test_maps_over_slabs_without_an_axis_set_is_an_authoring_error():
    with pytest.raises(ValueError, match="maps_over_slabs without requires_axes"):
        plugins.PluginOperationSpec(
            id="demo-ops:bad_slab", label="Bad", fn=lambda a: a, maps_over_slabs=True
        )


def test_a_slab_mapped_op_round_trips_its_axis_set_through_a_recipe(spec_cache):
    _install(
        spec_cache,
        plugins.PluginOperationSpec(
            id="demo-ops:slab_recipe",
            label="Slab recipe",
            fn=lambda slab: slab + 1.0,
            requires_axes=True,
            maps_over_slabs=True,
        ),
    )
    operation = registry.create_operation("demo-ops:slab_recipe", axes=(2, 0), ndim=3)

    text = recipes.dumps_recipe([operation])

    assert '"axes"' in text
    assert recipes.loads_recipe(text, base_shape=(4, 5, 6)) == (operation,)
    assert operation.axes == (0, 2)
