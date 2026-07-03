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


def test_default_axes_have_stable_ids_labels_and_sizes():
    axes = default_axes((3, 4))

    assert tuple(axis.id for axis in axes) == ("axis-0", "axis-1")
    assert tuple(axis.label for axis in axes) == ("Dim 0", "Dim 1")
    assert tuple(axis.size for axis in axes) == (3, 4)


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
    document = ArrayDocument([[1, 2], [3, 4], [5, 6]], operations=(Crop(axis=0, start=1, stop=3), Mean(axis=1)), axes=axes)

    assert document.current_shape == (2,)
    assert document.current_axes[0].spacing == 0.5
    assert document.current_axes[0].origin == 0.5
    assert document.current_axes[0].unit == "mm"


def test_operation_coordinator_passes_axes_to_document():
    from arrayscope.operations.coordinator import OperationCoordinator
    import numpy as np

    axes = (AxisInfo("slice", "Slice", 3, unit="mm", spacing=2.0), AxisInfo("coil", "Coil", 2))
    coordinator = OperationCoordinator(np.zeros((3, 2)), axes=axes)

    assert coordinator.document.base_axes[0].label == "Slice"
    assert coordinator.document.current_axes[0].spacing == 2.0


def test_axis_display_name_prefers_custom_labels():
    assert axis_display_name(None, 2) == "2"
    assert axis_display_name(AxisInfo("axis-2", "Dim 2", 4), 2) == "2"
    assert axis_display_name(AxisInfo("slice", "Slice", 4), 2) == "Slice"


def test_axis_metadata_summary_includes_unit_spacing_and_origin():
    summary = axis_metadata_summary(AxisInfo("slice", "Slice", 8, unit="mm", spacing=1.5, origin=-4.0))

    assert "Slice [8]" in summary
    assert "unit: mm" in summary
    assert "spacing: 1.5 mm" in summary
    assert "origin: -4 mm" in summary

