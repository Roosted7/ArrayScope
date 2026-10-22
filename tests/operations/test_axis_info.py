import numpy as np
import pytest

from arrayscope.core.axis_info import (
    AxisInfo,
    axis_display_name,
    axis_metadata_summary,
    default_axes,
    output_axes_for_operation,
    output_axes_for_operations,
)
from arrayscope.operations.pipeline import (
    ArrayDocument,
    CenteredFFT,
    CombineRealImagAxis,
    Crop,
    FFTShift,
    Mean,
    ReverseAxis,
    RootSumSquares,
    SplitComplexAxis,
)
from tests import seams


def test_default_axes_have_stable_ids_labels_and_sizes():
    axes = default_axes((3, 4))

    assert tuple(axis.id for axis in axes) == ("axis-0", "axis-1")
    assert tuple(axis.label for axis in axes) == ("Dim 0", "Dim 1")
    assert tuple(axis.size for axis in axes) == (3, 4)


def test_empty_scientific_array_axes_and_document_keep_zero_extent():
    axes = default_axes((0, 2))
    document = ArrayDocument(np.empty((0, 2), dtype=np.complex128), axes=axes)

    assert tuple(axis.size for axis in axes) == (0, 2)
    assert document.current_shape == (0, 2)
    assert document.current_axes == axes
    assert document.materialize().shape == (0, 2)
    assert document.materialize().dtype == np.dtype("complex128")


def test_axis_metadata_rejects_negative_extent():
    with pytest.raises(ValueError, match="axis size"):
        AxisInfo("invalid", "Invalid", -1)


def test_axis_info_crop_updates_size_and_preserves_identity():
    axes = default_axes((3, 4))

    updated = output_axes_for_operation(axes, Crop(axis=1, start=1, stop=3))

    assert updated[1].id == axes[1].id
    assert updated[1].size == 2


def test_axis_info_reduction_removes_axis():
    axes = default_axes((3, 4, 5))

    updated = output_axes_for_operations(axes, (Mean(axis=1), RootSumSquares(axis=1)))

    assert tuple(axis.id for axis in updated) == ("axis-0",)


def test_axis_info_fft_and_reverse_preserve_identity():
    axes = default_axes((3, 4))

    updated = output_axes_for_operations(axes, (ReverseAxis(axis=0), CenteredFFT(axis=1)))

    assert updated == axes


def test_axis_info_combine_and_split_update_size_and_coordinate():
    axes = default_axes((3, 2))

    combined = output_axes_for_operation(axes, CombineRealImagAxis(axis=1))
    split = output_axes_for_operation(combined, SplitComplexAxis(axis=1))

    assert combined[1].size == 1
    assert combined[1].coordinate == "complex"
    assert split[1].size == 2
    assert split[1].coordinate == "real-imag"


def test_array_document_exposes_current_axes_matching_current_shape():
    axes = (AxisInfo("readout", "Readout", 3), AxisInfo("coil", "Coil", 2))
    document = ArrayDocument([[1, 2], [3, 4], [5, 6]], operations=(Mean(axis=1),), axes=axes)

    assert document.current_shape == (3,)
    assert tuple(axis.id for axis in document.current_axes) == ("readout",)
    assert tuple(axis.size for axis in document.current_axes) == document.current_shape


def test_axis_info_coerces_spacing_and_origin_to_float():
    axis = AxisInfo("slice", "Slice", 8, unit="mm", spacing="1.5", origin="-4")

    assert axis.spacing == 1.5
    assert axis.origin == -4.0


def test_crop_shifts_origin_by_start_times_spacing():
    axes = (AxisInfo("slice", "Slice", 10, unit="mm", spacing=2.0, origin=5.0),)

    updated = output_axes_for_operation(axes, Crop(axis=0, start=3, stop=8))

    assert updated[0].size == 5
    assert updated[0].spacing == 2.0
    assert updated[0].origin == 11.0
    assert updated[0].unit == "mm"


def test_reverse_negates_spacing_and_moves_origin_to_last_sample():
    axes = (AxisInfo("slice", "Slice", 4, unit="mm", spacing=2.0, origin=1.0),)

    updated = output_axes_for_operation(axes, ReverseAxis(axis=0))

    assert updated[0].spacing == -2.0
    assert updated[0].origin == 7.0
    assert updated[0].id == "slice"


def test_centered_fft_clears_physical_metadata_on_transformed_axis():
    axes = (
        AxisInfo("readout", "Readout", 4, unit="mm", spacing=1.0, origin=0.0),
        AxisInfo("phase", "Phase", 4, unit="mm", spacing=1.0, origin=0.0),
    )

    updated = output_axes_for_operation(axes, CenteredFFT(axis=0))

    assert updated[0].unit is None
    assert updated[0].spacing is None
    assert updated[0].origin is None
    assert updated[0].id == "readout"
    assert updated[1] == axes[1]


def test_fftshift_clears_affine_mapping_but_keeps_unit():
    axes = (AxisInfo("readout", "Readout", 4, unit="mm", spacing=1.0, origin=0.0),)

    updated = output_axes_for_operation(axes, FFTShift(axis=0))

    assert updated[0].unit == "mm"
    assert updated[0].spacing is None
    assert updated[0].origin is None


def test_reduction_and_combine_drop_or_clear_physical_metadata():
    axes = (
        AxisInfo("readout", "Readout", 3, unit="mm", spacing=1.0),
        AxisInfo("coil", "Coil", 2, spacing=1.0, origin=0.0),
    )

    reduced = output_axes_for_operations(axes, (Mean(axis=1),))
    combined = output_axes_for_operation(axes, CombineRealImagAxis(axis=1))

    assert tuple(axis.id for axis in reduced) == ("readout",)
    assert combined[1].coordinate == "complex"
    assert combined[1].spacing is None
    assert combined[1].origin is None


def test_array_document_propagates_spacing_through_operations():
    axes = (
        AxisInfo("readout", "Readout", 3, unit="mm", spacing=0.5, origin=0.0),
        AxisInfo("coil", "Coil", 2),
    )
    document = ArrayDocument(
        [[1, 2], [3, 4], [5, 6]],
        operations=(Crop(axis=0, start=1, stop=3), Mean(axis=1)),
        axes=axes,
    )

    assert document.current_shape == (2,)
    assert document.current_axes[0].spacing == 0.5
    assert document.current_axes[0].origin == 0.5
    assert document.current_axes[0].unit == "mm"


def test_operation_coordinator_passes_axes_to_document():
    import numpy as np

    from arrayscope.operations.coordinator import OperationCoordinator

    axes = (AxisInfo("slice", "Slice", 3, unit="mm", spacing=2.0), AxisInfo("coil", "Coil", 2))
    coordinator = OperationCoordinator(np.zeros((3, 2)), axes=axes)

    assert coordinator.document.base_axes[0].label == "Slice"
    assert coordinator.document.current_axes[0].spacing == 2.0


def test_axis_display_name_prefers_custom_labels():
    assert axis_display_name(None, 2) == "2"
    assert axis_display_name(AxisInfo("axis-2", "Dim 2", 4), 2) == "2"
    assert axis_display_name(AxisInfo("slice", "Slice", 4), 2) == "Slice"


def test_axis_metadata_summary_includes_unit_spacing_and_origin():
    summary = axis_metadata_summary(
        AxisInfo("slice", "Slice", 8, unit="mm", spacing=1.5, origin=-4.0)
    )

    assert "Slice [8]" in summary
    assert "unit: mm" in summary
    assert "spacing: 1.5 mm" in summary
    assert "origin: -4 mm" in summary


# --- rank-changing operations the name table does not know -------------------
#
# ``output_axes_for_operation`` fell through to ``return tuple(axes)`` for any
# unnamed operation, so a rank change reached ``axes_for_shape`` with the wrong
# length and ``ArrayDocument.__init__`` raised "axis metadata length must match
# shape length" -- a hard failure at document construction for exactly the class
# of operation ADR 0060 exists to allow.


class _DropMiddleAxis:
    """An operation with no ``output_axes`` hook that reduces rank."""

    def apply(self, data):
        return np.asarray(data).mean(axis=1)

    def output_shape(self, shape):
        return (int(shape[0]), *tuple(int(size) for size in shape[2:]))


class _AppendAxis:
    """An operation with no ``output_axes`` hook that grows rank."""

    def apply(self, data):
        return np.asarray(data)[..., None].repeat(3, axis=-1)

    def output_shape(self, shape):
        return (*tuple(int(size) for size in shape), 3)


class _PrependAxis:
    """Rank growth whose new axis lands where nothing can be paired with it."""

    def apply(self, data):
        return np.stack([np.asarray(data)] * 3, axis=0)

    def output_shape(self, shape):
        return (3, *tuple(int(size) for size in shape))


def test_unknown_rank_reducing_operation_keeps_the_surviving_axes():
    axes = (
        AxisInfo("slice", "Slice", 4, unit="mm", spacing=2.0),
        AxisInfo("coil", "Coil", 5),
        AxisInfo("echo", "Echo", 6, unit="ms"),
    )

    updated = output_axes_for_operation(axes, _DropMiddleAxis())

    assert tuple(axis.id for axis in updated) == ("slice", "echo")
    assert updated[0].spacing == 2.0
    assert updated[1].unit == "ms"


def test_unknown_rank_growing_operation_appends_default_axes():
    axes = default_axes((4, 5))

    updated = output_axes_for_operation(axes, _AppendAxis())

    assert tuple(axis.id for axis in updated) == ("axis-0", "axis-1", "axis-2")
    assert tuple(axis.label for axis in updated) == ("Dim 0", "Dim 1", "Dim 2")
    assert tuple(axis.size for axis in updated) == (4, 5, 3)


def test_unknown_rank_change_that_cannot_be_paired_mints_plain_defaults():
    # Sizes do not line up positionally, so pinning "Slice"/2 mm onto whichever
    # axis happens to sit there would be a guess. Plain defaults are the honest
    # answer; an operation that knows better implements ``output_axes``.
    axes = (AxisInfo("slice", "Slice", 4, unit="mm", spacing=2.0), AxisInfo("coil", "Coil", 5))

    updated = output_axes_for_operation(axes, _PrependAxis())

    assert tuple(axis.id for axis in updated) == ("axis-0", "axis-1", "axis-2")
    assert all(axis.unit is None and axis.spacing is None for axis in updated)


def test_array_document_constructs_over_unknown_rank_changing_operations():
    grown = ArrayDocument(np.zeros((4, 5), dtype=np.float32), operations=(_AppendAxis(),))
    dropped = ArrayDocument(np.zeros((4, 5, 6), dtype=np.float32), operations=(_DropMiddleAxis(),))

    assert grown.current_shape == (4, 5, 3)
    assert dropped.current_shape == (4, 6)
    assert tuple(axis.size for axis in grown.current_axes) == grown.current_shape
    assert tuple(axis.size for axis in dropped.current_axes) == dropped.current_shape


def test_rank_increasing_plugin_operation_labels_every_axis_uniquely():
    from arrayscope.operations import plugins
    from arrayscope.operations.plugins import PluginOperation, PluginOperationSpec

    seams.reset_plugin_cache()
    try:
        spec = PluginOperationSpec(
            id="demo:stack-3",
            label="Stack 3",
            fn=lambda a: np.stack([a, a, a], axis=0),
            changes_shape=True,
        )
        plugins._SPEC_CACHE[spec.id] = spec
        document = ArrayDocument(
            np.zeros((4, 5), dtype=np.float32),
            operations=(PluginOperation(plugin_id=spec.id),),
        )
    finally:
        seams.reset_plugin_cache()

    # The created axis 0 and the surviving input axis 0 both claimed the
    # positional default, producing ['Dim 0', 'Dim 0', 'Dim 1'].
    labels = tuple(axis.label for axis in document.current_axes)
    identifiers = tuple(axis.id for axis in document.current_axes)
    assert labels == ("Dim 0", "Dim 1", "Dim 2")
    assert len(set(identifiers)) == len(identifiers)


def test_a_custom_label_survives_a_rank_change_unrenumbered():
    from arrayscope.operations import plugins
    from arrayscope.operations.plugins import PluginOperation, PluginOperationSpec

    seams.reset_plugin_cache()
    try:
        spec = PluginOperationSpec(
            id="demo:stack-2",
            label="Stack 2",
            fn=lambda a: np.stack([a, a], axis=0),
            changes_shape=True,
        )
        plugins._SPEC_CACHE[spec.id] = spec
        document = ArrayDocument(
            np.zeros((4, 5), dtype=np.float32),
            operations=(PluginOperation(plugin_id=spec.id),),
            axes=(AxisInfo("slice", "Slice", 4), AxisInfo("axis-1", "Dim 1", 5)),
        )
    finally:
        seams.reset_plugin_cache()

    labels = tuple(axis.label for axis in document.current_axes)
    assert labels == ("Dim 0", "Slice", "Dim 2")
