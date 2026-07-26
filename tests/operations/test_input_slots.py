"""Auxiliary input slots: schema, characterization, recipes, and ROI invalidation."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.roi import (
    RoiGeometry,
    RoiKind,
    RoiSelection,
    roi_coordinates,
    roi_mask,
)
from arrayscope.core.roi_store import RoiStore
from arrayscope.operations import plugins, recipes, registry
from arrayscope.operations.capabilities import OperationClass
from arrayscope.operations.evaluator import OperationEvaluator
from arrayscope.operations.input_slots import (
    SLOT_DIMENSION_SET,
    SLOT_OPEN_DOCUMENT,
    SLOT_ROI_COORDINATES,
    SLOT_ROI_MASK,
    SLOT_SAVED_ARRAY,
    OperationInputSlot,
    ResolvedSlot,
    SlotBinding,
)
from arrayscope.operations.pipeline import ArrayDocument, OperationStep
from arrayscope.operations.plugin_conformance import (
    characterization_stats,
    characterize_operation,
    reset_characterization_cache,
)
from arrayscope.operations.plugins import PluginOperationSpec
from arrayscope.window.operation_slots import OperationSlotSourcesMixin


@pytest.fixture(autouse=True)
def _clean_plugin_state():
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()
    reset_characterization_cache()
    yield
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()
    reset_characterization_cache()


def _slot_spec(operation_id="test-slots:add"):
    return PluginOperationSpec(
        id=operation_id,
        label="Add auxiliary input",
        build=lambda _axis, _params, slots: lambda data: np.asarray(data) + slots["reference"],
        input_slots=(
            OperationInputSlot(
                "reference",
                "Reference",
                accepts=(SLOT_DIMENSION_SET, SLOT_ROI_MASK),
            ),
        ),
        # Multi-input operations do not earn a region path by inheriting a
        # single-input claim.
        region_capable=True,
    )


def _resolved(
    *,
    source_id="source-a",
    shape=(3, 4),
    dtype="float32",
    source_identity=("revision", 1),
    value=None,
):
    binding = SlotBinding(
        SLOT_DIMENSION_SET,
        source_id=source_id,
        indices=tuple(None for _ in shape),
    )
    array = (
        np.ones(shape, dtype=dtype)
        if value is None
        else np.asarray(value, dtype=dtype).reshape(shape)
    )
    return ResolvedSlot(
        binding=binding,
        shape=shape,
        dtype=np.dtype(dtype).str,
        source_identity=source_identity,
        source=array,
    )


@pytest.mark.parametrize(
    "changed",
    [
        _resolved(source_id="source-b"),
        _resolved(shape=(1, 4)),
        _resolved(dtype=np.dtype("float64")),
        _resolved(source_identity=("revision", 2)),
    ],
    ids=["binding", "shape", "dtype", "source-identity"],
)
def test_every_slot_identity_component_partitions_characterization_cache(changed):
    calls = 0

    def build(_axis, _params, _slots):
        nonlocal calls
        calls += 1
        return lambda data: data

    spec = replace(_slot_spec("test-slots:cache"), build=build)
    initial = _resolved()

    characterize_operation(spec, (3, 4), "float32", slots={"reference": initial})
    characterize_operation(spec, (3, 4), "float32", slots={"reference": initial})
    characterize_operation(spec, (3, 4), "float32", slots={"reference": changed})

    assert calls == 2
    assert characterization_stats()["cache_hits"] == 1


def test_multi_input_is_opaque_even_when_definition_claims_regions():
    spec = _slot_spec()
    plugins._SPEC_CACHE[spec.id] = spec
    source = _resolved()
    operation = plugins.create_plugin_operation(
        spec.id,
        slot_bindings={"reference": source.binding},
        resolved_slots={"reference": source},
    )

    assert operation.execution_class is OperationClass.OPAQUE
    assert operation.capabilities((3, 4), np.dtype("float32")).blocking_axes == (0, 1)
    np.testing.assert_array_equal(
        operation.apply(np.zeros((3, 4), dtype=np.float32)),
        np.ones((3, 4), dtype=np.float32),
    )


def test_slot_bindings_round_trip_and_unresolved_load_is_quarantined():
    spec = _slot_spec("test-slots:recipe")
    registry.register_pack_operation(spec)
    source = _resolved()
    operation = registry.create_operation(
        spec.id,
        slot_bindings={"reference": source.binding},
        resolved_slots={"reference": source},
    )
    text = recipes.dumps_recipe((operation,))

    restored = recipes.loads_recipe_steps(
        text,
        base_shape=(3, 4),
        slot_resolver=lambda _slot, _binding: source,
    )
    assert restored[0].enabled is True
    assert dict(restored[0].operation.slot_bindings) == {"reference": source.binding}

    unresolved = recipes.loads_recipe_steps(text, base_shape=(3, 4))
    assert unresolved[0].enabled is False
    assert "cannot resolve" in unresolved[0].unavailable_reason


@pytest.mark.parametrize(
    ("primary", "secondary"),
    [
        (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)),
        (np.ones((1,), dtype=np.float32), np.ones((1,), dtype=np.float32)),
        (np.ones((2, 2), dtype=np.int16), np.ones((2, 2), dtype=np.float32)),
    ],
    ids=["zero-length-axis", "single-element-slot", "mismatched-dtypes"],
)
def test_single_and_multi_input_paths_agree_on_boundary_inputs(primary, secondary):
    unary = PluginOperationSpec(
        id="test-slots:unary",
        label="Unary identity",
        fn=lambda data: data,
    )
    multi = _slot_spec("test-slots:boundary")
    plugins._SPEC_CACHE.update({unary.id: unary, multi.id: multi})
    binding = SlotBinding(
        SLOT_DIMENSION_SET,
        source_id="boundary",
        indices=tuple(None for _ in secondary.shape),
    )
    source = ResolvedSlot(
        binding=binding,
        shape=secondary.shape,
        dtype=secondary.dtype.str,
        source_identity=("boundary", secondary.shape, secondary.dtype.str),
        source=secondary,
    )
    unary_operation = plugins.create_plugin_operation(unary.id)
    multi_operation = plugins.create_plugin_operation(
        multi.id,
        slot_bindings={"reference": binding},
        resolved_slots={"reference": source},
    )

    np.testing.assert_array_equal(unary_operation.apply(primary), primary)
    np.testing.assert_array_equal(
        multi_operation.apply(primary),
        primary + secondary,
    )


def test_empty_roi_representations_are_well_formed():
    empty = RoiGeometry(RoiKind.FREEHAND_POLYGON, points=())

    assert roi_coordinates(empty).shape == (0, 2)
    assert roi_coordinates(empty).dtype == np.dtype(np.float64)
    assert roi_mask((0, 4), empty).shape == (0, 4)
    assert roi_mask((3, 4), empty).sum() == 0


def test_window_resolves_dimension_document_roi_and_saved_sources(tmp_path):
    geometry = RoiGeometry(
        RoiKind.FREEHAND_POLYGON,
        points=((0, 0), (2, 0), (1, 2)),
    )
    primary = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    window = _SlotWindow(
        primary,
        RoiStore(selections=(RoiSelection("roi-a", "A", geometry),)),
    )
    window.view_state = SimpleNamespace(image_axes=(0, 1), slice_indices=(0, 0, 2))

    dimension_binding = window._current_dimension_set_binding()
    dimension = window._resolve_operation_slot(SimpleNamespace(label="Plane"), dimension_binding)
    np.testing.assert_array_equal(dimension.materialize(), primary[:, :, 2])

    other = np.full((2, 3), 5, dtype=np.float64)
    member = SimpleNamespace(
        _operation_source_id="other-document",
        document=ArrayDocument(other),
        data=other,
    )
    window._compare_group = SimpleNamespace(members=lambda: (window, member))
    document = window._resolve_operation_slot(
        SimpleNamespace(label="Document"),
        SlotBinding(SLOT_OPEN_DOCUMENT, source_id="other-document"),
    )
    np.testing.assert_array_equal(document.materialize(), other)

    coordinates = window._resolve_operation_slot(
        SimpleNamespace(label="Coordinates"),
        SlotBinding(SLOT_ROI_COORDINATES, source_id="roi-a"),
    )
    np.testing.assert_array_equal(
        coordinates.materialize(),
        roi_coordinates(geometry),
    )

    saved_value = np.arange(6, dtype=np.int16).reshape(2, 3)
    saved_path = tmp_path / "reference.npy"
    np.save(saved_path, saved_value)
    saved = window._resolve_operation_slot(
        SimpleNamespace(label="Saved"),
        SlotBinding(SLOT_SAVED_ARRAY, path=str(saved_path)),
    )
    np.testing.assert_array_equal(saved.materialize(), saved_value)


class _Coordinator:
    def __init__(self, owner):
        self.owner = owner

    def _document(self, *, steps):
        return ArrayDocument(
            self.owner.base_data,
            steps=steps,
            revision=self.owner.document.revision,
        )


class _SlotWindow(OperationSlotSourcesMixin):
    def __init__(self, data, roi_store):
        self.base_data = data
        self.data = data
        self.document = ArrayDocument(data)
        self.operation_evaluator = OperationEvaluator(self.document)
        self.roi_store = roi_store
        self.img_view = SimpleNamespace(image=data)
        self.view_state = SimpleNamespace(image_axes=(0, 1), slice_indices=(0, 0))
        self._operation_source_id = "primary"
        self._refreshing_operation_slots = False
        self._slot_file_watcher = None
        self.operation_coordinator = _Coordinator(self)

    def _set_document(self, document):
        self.document = document
        self.operation_evaluator.set_document(document)

    def render(self, **_kwargs):
        pass


def test_roi_geometry_invalidates_only_the_dependent_operation_evaluation():
    spec = _slot_spec("test-slots:roi")
    registry.register_pack_operation(spec)
    first_geometry = RoiGeometry(RoiKind.RECTANGLE, rect=(0, 0, 1, 1))
    unrelated_geometry = RoiGeometry(RoiKind.RECTANGLE, rect=(2, 2, 1, 1))
    window = _SlotWindow(
        np.zeros((4, 4), dtype=np.float32),
        RoiStore(
            selections=(
                RoiSelection("roi-a", "A", first_geometry),
                RoiSelection("roi-b", "B", unrelated_geometry),
            )
        ),
    )
    binding = SlotBinding(SLOT_ROI_MASK, source_id="roi-a", label="A (mask)")
    operation = registry.create_operation(
        spec.id,
        slot_bindings={"reference": binding},
        slot_resolver=window._resolve_operation_slot,
    )
    window._set_document(ArrayDocument(window.base_data, steps=(OperationStep(operation),)))

    initial = window.operation_evaluator.current_data().copy()
    assert window.operation_evaluator.derived_evaluations == 1

    # An unrelated ROI geometry update is not even rebuilt, so the cached
    # derived array remains the exact same evaluation.
    window.roi_store = window.roi_store.upsert(
        RoiSelection(
            "roi-b",
            "B",
            RoiGeometry(RoiKind.RECTANGLE, rect=(1, 1, 2, 2)),
        )
    )
    assert window._refresh_operation_slot_bindings(roi_id="roi-b", render=False) is False
    window.operation_evaluator.current_data()
    assert window.operation_evaluator.derived_evaluations == 1

    # Label/color are presentation metadata: rebuilding the referenced source
    # compares equal and must also preserve the cached evaluation.
    window.roi_store = window.roi_store.upsert(
        RoiSelection("roi-a", "Renamed", first_geometry, color=(1, 2, 3))
    )
    assert window._refresh_operation_slot_bindings(roi_id="roi-a", render=False) is False
    window.operation_evaluator.current_data()
    assert window.operation_evaluator.derived_evaluations == 1

    window.roi_store = window.roi_store.upsert(
        RoiSelection(
            "roi-a",
            "Renamed",
            RoiGeometry(RoiKind.RECTANGLE, rect=(0, 0, 3, 3)),
        )
    )
    assert window._refresh_operation_slot_bindings(roi_id="roi-a", render=False) is True
    moved = window.operation_evaluator.current_data()
    assert window.operation_evaluator.derived_evaluations == 2
    assert moved.sum() > initial.sum()

    window.roi_store = window.roi_store.remove("roi-a")
    assert window._refresh_operation_slot_bindings(roi_id="roi-a", render=False) is True
    assert window.document.steps[0].enabled is False
    assert "no longer exists" in window.document.steps[0].unavailable_reason
