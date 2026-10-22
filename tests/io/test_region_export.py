"""Exact numerical, bounded-source, and publication contracts for export."""

from __future__ import annotations

import json
from math import prod
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.array_source import SourceReadRefused
from arrayscope.core.roi import RoiGeometry
from arrayscope.io.numpy_save import save_derived_array
from arrayscope.io.region_export import (
    RegionExportChanged,
    RegionExportRequest,
    capture_export_definitions,
    capture_export_sources,
    export_region,
    plan_region_export,
    validate_region_export_snapshot,
)
from arrayscope.operations import plugin_conformance, plugins
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.input_slots import OperationInputSlot, ResolvedSlot, SlotBinding
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, Crop, Mean, SplitAxis
from arrayscope.operations.plugins import PluginOperation, PluginOperationSpec


class _VirtualArray:
    """A many-GiB source whose data exist only for an explicitly bounded read."""

    dtype = np.dtype("float32")

    def __init__(self, shape=(2_000_000, 1000)):
        self.shape = shape
        self.reads = []

    @property
    def nbytes(self):
        return prod(self.shape) * self.dtype.itemsize

    def __array__(self, *args, **kwargs):
        raise AssertionError("export must not implicitly materialize its source")

    def read_region(self, spec, *, cancellation_token=None):
        self.reads.append(spec)
        rows, columns = (
            np.arange(*item.indices(size)) for item, size in zip(spec, self.shape, strict=True)
        )
        return (rows[:, None] * 1000 + columns[None, :]).astype(self.dtype)


def test_preflight_reads_nothing_and_large_lazy_export_reads_only_selected_values(tmp_path):
    source = _VirtualArray()
    request = RegionExportRequest(
        ArrayDocument(source), tmp_path / "selection.npy", ((1, 3), (4, 7))
    )

    prepared = plan_region_export(request)

    assert source.reads == []
    assert prepared.source_read_bytes == 24
    assert prepared.output_bytes == 24
    assert prepared.estimated_peak_bytes == 24
    result = export_region(request, prepared=prepared)

    np.testing.assert_array_equal(
        np.load(result.paths[0], allow_pickle=False),
        np.array([[1004, 1005, 1006], [2004, 2005, 2006]], dtype=np.float32),
    )
    assert source.reads == [(slice(1, 3, 1), slice(4, 7, 1))]
    assert result.shape == (2, 3)
    assert result.dtype == np.dtype("float32")


def test_read_budget_refuses_huge_lazy_regions_without_integer_overflow_or_reads(tmp_path):
    source = _VirtualArray((2**40, 2**30))
    request = RegionExportRequest(
        ArrayDocument(source), tmp_path / "huge.npy", ((0, 2**40), (0, 2**30))
    )
    prepared = plan_region_export(request)

    assert prepared.source_read_bytes == prepared.output_bytes == 2**72
    assert prepared.needs_confirmation
    with pytest.raises(SourceReadRefused) as caught:
        export_region(request, prepared=prepared)
    assert caught.value.requested_nbytes == 2**72
    assert source.reads == []
    assert list(tmp_path.iterdir()) == []


def test_complex_strided_crop_fft_export_matches_independent_numpy_values(tmp_path):
    values = np.arange(4 * 8 * 12, dtype=np.float64).reshape(4, 8, 12)
    source = (values + 1j * values[::-1])[:, ::-1, ::2]
    assert not source.flags.c_contiguous
    document = ArrayDocument(source, operations=(Crop(1, 1, 7), CenteredFFT(2), Mean(0)))
    request = RegionExportRequest(
        document, tmp_path / "complex.npz", ((1, 3), (2, 4)), squeeze=False
    )
    # ArrayScope's established MRI-oriented "FFT" uses the inverse DFT.
    expected = np.fft.ifftshift(
        np.fft.ifft(np.fft.fftshift(source[:, 1:7, :], axes=2), axis=2, norm="ortho"), axes=2
    ).mean(axis=0)[1:3, 2:4]

    result = export_region(request)

    with np.load(result.paths[0], allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["array"], expected, rtol=1e-12, atol=1e-12)
        assert archive["array"].dtype == np.dtype("complex128")
        selection = json.loads(str(archive["selection_json"]))
    assert selection["derived_shape"] == [6, 6]
    assert selection["ranges"] == [[1, 3], [2, 4]]
    assert selection["output_axes"] == [0, 1]


def test_selection_is_in_output_axis_space_after_rank_increase_and_reduction(tmp_path):
    source = np.arange(8 * 6 * 4, dtype=np.int16).reshape(8, 6, 4)
    document = ArrayDocument(source, operations=(SplitAxis(1, 3), Mean(0)))
    request = RegionExportRequest(
        document, tmp_path / "rank.npy", ((0, 2), (1, 3), (0, 2)), squeeze=False
    )

    result = export_region(request)

    expected = source.reshape(8, 2, 3, 4).mean(axis=0, dtype=np.float32)[0:2, 1:3, 0:2]
    np.testing.assert_array_equal(np.load(result.paths[0], allow_pickle=False), expected)
    assert result.shape == (2, 2, 2)
    assert result.dtype == np.dtype("float32")


@pytest.mark.parametrize(
    ("shape", "ranges", "squeeze", "expected_shape"),
    [
        ((1, 2, 1), ((0, 1), (1, 2), (0, 1)), True, ()),
        ((1, 2, 1), ((0, 1), (1, 2), (0, 1)), False, (1, 1, 1)),
        ((0, 2), ((0, 0), (0, 2)), True, (0, 2)),
        ((), (), True, ()),
    ],
)
def test_scalar_singleton_and_empty_exports_keep_numpy_shape_semantics(
    tmp_path, shape, ranges, squeeze, expected_shape
):
    source = np.arange(prod(shape), dtype=np.complex128).reshape(shape)
    request = RegionExportRequest(
        ArrayDocument(source), tmp_path / "shape.npy", ranges, squeeze=squeeze
    )

    result = export_region(request)

    expected = source[tuple(slice(start, stop) for start, stop in ranges)]
    if squeeze:
        expected = np.squeeze(expected)
    np.testing.assert_array_equal(np.load(result.paths[0], allow_pickle=False), expected)
    assert result.shape == expected_shape
    assert result.dtype == source.dtype


def test_empty_lazy_selection_never_reads_source(tmp_path):
    source = _VirtualArray()
    result = export_region(
        RegionExportRequest(ArrayDocument(source), tmp_path / "empty.npy", ((5, 5), (2, 4)))
    )
    assert source.reads == []
    assert result.shape == (0, 2)
    assert np.load(result.paths[0], allow_pickle=False).shape == (0, 2)


def test_small_fft_selection_prices_full_expanded_transform_output(tmp_path):
    source = np.ones((4, 1024), dtype=np.float64)
    request = RegionExportRequest(
        ArrayDocument(source, operations=(CenteredFFT(1),)),
        tmp_path / "fft.npy",
        ((1, 2), (2, 3)),
    )

    prepared = plan_region_export(request)

    assert prepared.output_bytes == 16
    assert prepared.source_read_bytes == 1024 * 8
    assert prepared.estimated_peak_bytes == 1024 * 16 * 6
    assert prepared.needs_confirmation


def test_opaque_operation_runs_once_on_its_full_input_before_output_selection(
    tmp_path, monkeypatch
):
    calls = []

    def opaque(data):
        calls.append(data.shape)
        return np.square(data)

    spec = PluginOperationSpec(id="export:opaque", label="Opaque", fn=opaque)
    monkeypatch.setitem(plugins._SPEC_CACHE, spec.id, spec)
    source = np.arange(8 * 6, dtype=np.float64).reshape(8, 6)
    document = ArrayDocument(source, operations=(PluginOperation(spec.id),))
    request = RegionExportRequest(document, tmp_path / "opaque.npy", ((2, 4), (1, 3)))
    prepared = plan_region_export(request)
    calls.clear()

    result = export_region(request, prepared=prepared)

    np.testing.assert_array_equal(
        np.load(result.paths[0], allow_pickle=False), np.square(source)[2:4, 1:3]
    )
    assert calls == [source.shape]
    assert prepared.source_read_bytes == source.nbytes
    assert prepared.needs_confirmation


def test_definition_change_before_preflight_or_publication_never_replaces_file(
    tmp_path, monkeypatch
):
    revision = [1]
    spec = PluginOperationSpec(
        id="export:changing",
        label="Changing",
        fn=lambda data: data + 1,
        source_identity=lambda: revision[0],
    )
    monkeypatch.setitem(plugins._SPEC_CACHE, spec.id, spec)
    document = ArrayDocument(np.arange(12).reshape(3, 4), operations=(PluginOperation(spec.id),))
    definitions = capture_export_definitions(document)
    request = RegionExportRequest(
        document, tmp_path / "previous.npy", ((0, 2), (0, 2)), definitions=definitions
    )
    request.path.write_bytes(b"previous export")
    revision[0] = 2
    with pytest.raises(RegionExportChanged, match="changed during export"):
        plan_region_export(request)
    revision[0] = 1
    prepared = plan_region_export(request)

    def change_after_evaluation(progress):
        if progress.phase == "writing":
            revision[0] = 2

    with pytest.raises(RegionExportChanged, match="changed during export"):
        export_region(request, prepared=prepared, progress_callback=change_after_evaluation)
    assert request.path.read_bytes() == b"previous export"
    assert set(tmp_path.iterdir()) == {request.path}


def test_deleted_definition_reports_changed_snapshot(tmp_path, monkeypatch):
    spec = PluginOperationSpec(id="export:removed", label="Removed", fn=lambda data: data)
    monkeypatch.setitem(plugins._SPEC_CACHE, spec.id, spec)
    request = RegionExportRequest(
        ArrayDocument(np.arange(3), operations=(PluginOperation(spec.id),)),
        tmp_path / "removed.npy",
        ((0, 3),),
    )
    monkeypatch.delitem(plugins._SPEC_CACHE, spec.id)
    monkeypatch.setattr(plugins, "discover_plugin_entry_points", dict)

    with pytest.raises(RegionExportChanged, match="changed during export"):
        validate_region_export_snapshot(request)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "source",
    [
        None,
        ResolvedSlot(SlotBinding("roi-mask"), loader="roi-mask", source=SimpleNamespace()),
        ResolvedSlot(SlotBinding("open-document"), loader="document", source=lambda: None),
        ResolvedSlot(SlotBinding("dimension-set"), source=lambda: np.ones(2)),
        ResolvedSlot(SlotBinding("dimension-set"), unavailable_reason="Source closed"),
    ],
    ids=["missing", "live-roi", "live-document", "provider", "unresolved"],
)
def test_capture_refuses_unresolved_or_live_provider_slots_without_evaluating(monkeypatch, source):
    spec = PluginOperationSpec(
        id="export:unsafe-slot",
        label="Unsafe slot",
        build=lambda _axis, _params, _slots: lambda data: data,
        input_slots=(OperationInputSlot("reference", "Reference"),),
    )
    monkeypatch.setitem(plugins._SPEC_CACHE, spec.id, spec)
    operation = PluginOperation(
        spec.id, resolved_slots=() if source is None else (("reference", source),)
    )

    with pytest.raises(RegionExportChanged, match=r"[Ii]nput"):
        capture_export_definitions(SimpleNamespace(enabled_operations=(operation,)))


def test_capture_accepts_frozen_roi_geometry_without_materializing(monkeypatch):
    spec = PluginOperationSpec(
        id="export:roi-slot",
        label="ROI slot",
        build=lambda _axis, _params, _slots: lambda data: data,
        input_slots=(OperationInputSlot("reference", "Reference", accepts=("roi-mask",)),),
    )
    monkeypatch.setitem(plugins._SPEC_CACHE, spec.id, spec)
    operation = PluginOperation(
        spec.id,
        resolved_slots=(
            (
                "reference",
                ResolvedSlot(
                    SlotBinding("roi-mask"),
                    loader="roi-mask",
                    source=RoiGeometry("rectangle", rect=(0, 0, 1, 1)),
                ),
            ),
        ),
    )

    snapshots = capture_export_definitions(SimpleNamespace(enabled_operations=(operation,)))

    assert len(snapshots) == 1


def test_cold_auxiliary_plugin_preflight_never_materializes_inputs_before_consent(
    tmp_path, monkeypatch
):
    class LazyAuxiliary:
        shape = (3, 4)
        dtype = np.dtype("float32")
        reads = 0

        def __array__(self, dtype=None, copy=None):
            self.reads += 1
            return np.arange(12, dtype=dtype or self.dtype).reshape(self.shape)

    auxiliary = LazyAuxiliary()
    auxiliary_document = ArrayDocument(auxiliary)
    slot = ResolvedSlot(
        SlotBinding("open-document"),
        shape=auxiliary.shape,
        dtype=auxiliary.dtype.str,
        loader="document",
        source=auxiliary_document,
    )
    spec = PluginOperationSpec(
        id="export:cold-auxiliary",
        label="Auxiliary",
        build=lambda _axis, _params, slots: lambda data: data + slots["reference"],
        input_slots=(OperationInputSlot("reference", "Reference"),),
    )
    monkeypatch.setitem(plugins._SPEC_CACHE, spec.id, spec)
    document = ArrayDocument(
        np.zeros((3, 4), dtype=np.float32),
        operations=(PluginOperation(spec.id, resolved_slots=(("reference", slot),)),),
    )
    request = RegionExportRequest(document, tmp_path / "auxiliary.npy", ((1, 3), (2, 4)))
    captured_sources = capture_export_sources(document)
    assert len(captured_sources) == 2
    assert captured_sources[0] is document.base_data
    assert captured_sources[1] is auxiliary
    plugin_conformance._CHARACTERIZATION_CACHE.clear()
    auxiliary.reads = 0

    prepared = plan_region_export(request)

    assert auxiliary.reads == 0
    assert prepared.region_plan is None
    assert prepared.output_shape == (2, 2)
    assert (
        prepared.output_bytes is prepared.source_read_bytes is prepared.estimated_peak_bytes is None
    )
    assert prepared.needs_confirmation
    assert "materialize auxiliary inputs" in prepared.warnings[0]
    result = export_region(request, prepared=prepared)
    assert auxiliary.reads > 0
    np.testing.assert_array_equal(
        np.load(result.paths[0], allow_pickle=False),
        np.arange(12, dtype=np.float32).reshape(3, 4)[1:3, 2:4],
    )


def test_bound_file_change_is_detected_before_preflight_or_publication(tmp_path, monkeypatch):
    auxiliary_path = tmp_path / "auxiliary.npy"
    auxiliary = np.arange(6, dtype=np.float64).reshape(2, 3)
    np.save(auxiliary_path, auxiliary)
    stat = auxiliary_path.stat()
    slot = ResolvedSlot(
        SlotBinding("saved-array", path=str(auxiliary_path)),
        shape=auxiliary.shape,
        dtype=auxiliary.dtype.str,
        source_identity=(str(auxiliary_path), stat.st_mtime_ns, stat.st_size),
        loader="saved-array",
        source=str(auxiliary_path),
    )
    spec = PluginOperationSpec(
        id="export:file-input",
        label="File input",
        build=lambda _axis, _params, slots: lambda data: data + slots["reference"],
        input_slots=(OperationInputSlot("reference", "Reference"),),
    )
    monkeypatch.setitem(plugins._SPEC_CACHE, spec.id, spec)
    request = RegionExportRequest(
        ArrayDocument(
            np.ones((2, 3)),
            operations=(PluginOperation(spec.id, resolved_slots=(("reference", slot),)),),
        ),
        tmp_path / "previous.npy",
        ((0, 2), (0, 3)),
    )
    request.path.write_bytes(b"previous export")
    prepared = plan_region_export(request)

    def change_after_evaluation(progress):
        if progress.phase == "writing":
            auxiliary_path.write_bytes(b"changed file")

    with pytest.raises(RegionExportChanged, match="changed during export"):
        export_region(request, prepared=prepared, progress_callback=change_after_evaluation)
    assert request.path.read_bytes() == b"previous export"
    assert set(tmp_path.iterdir()) == {request.path, auxiliary_path}
    with pytest.raises(RegionExportChanged):
        plan_region_export(request)


def test_cancel_before_evaluation_never_reads_or_creates_file(tmp_path):
    source = _VirtualArray()
    request = RegionExportRequest(ArrayDocument(source), tmp_path / "cancel.npy", ((0, 2), (0, 3)))

    with pytest.raises(EvaluationCancelled):
        export_region(request, cancellation_token=SimpleNamespace(cancelled=True))
    assert source.reads == []
    assert list(tmp_path.iterdir()) == []


def test_queued_request_freezes_relative_destination_directory(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    request = RegionExportRequest(ArrayDocument(np.arange(3)), "result.npy", ((0, 3),))
    monkeypatch.chdir(second)

    result = export_region(request)

    assert result.paths == (first / "result.npy",)
    assert list(second.iterdir()) == []


def test_completion_observer_cannot_cancel_an_already_published_export(tmp_path, caplog):
    request = RegionExportRequest(ArrayDocument(np.arange(3)), tmp_path / "complete.npy", ((0, 3),))

    def fail_when_complete(progress):
        if progress.phase == "complete":
            raise EvaluationCancelled()

    result = export_region(request, progress_callback=fail_when_complete)

    np.testing.assert_array_equal(np.load(result.paths[0], allow_pickle=False), np.arange(3))
    assert "Export committed, but its completion observer failed" in caplog.text


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
@pytest.mark.parametrize("order", ["C", "F", "strided"])
def test_cancellable_writer_preserves_array_order_shape_and_native_complex_dtype(
    tmp_path, suffix, order
):
    values = np.arange(30).reshape(5, 6) + 1j * np.arange(30).reshape(5, 6)[::-1]
    source = values[:, ::-2] if order == "strided" else np.array(values, order=order)
    path = tmp_path / f"ordered{suffix}"

    save_derived_array(path, source, cancellation_token=SimpleNamespace(cancelled=False))

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            saved = archive["array"]
    else:
        saved = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(saved, source)
    assert saved.dtype == source.dtype
    if order == "F":
        assert saved.flags.f_contiguous


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_cancellation_after_partial_array_serialization_preserves_previous_export(tmp_path, suffix):
    path = tmp_path / f"previous{suffix}"
    path.write_bytes(b"previous export")
    token = SimpleNamespace(cancelled=False)
    written = []

    def cancel_after_data(count):
        written.append(count)
        if count > 1024:
            token.cancelled = True

    with pytest.raises(EvaluationCancelled):
        save_derived_array(
            path,
            np.arange(20 * 1024**2, dtype=np.uint8),
            cancellation_token=token,
            progress_callback=cancel_after_data,
        )
    assert max(written) > 1024
    assert max(written) < 20 * 1024**2
    assert path.read_bytes() == b"previous export"
    assert set(tmp_path.iterdir()) == {path}


def test_cancel_after_staging_before_publish_preserves_file_but_late_cancel_is_success(
    tmp_path, monkeypatch
):
    from arrayscope.io import numpy_save

    path = tmp_path / "previous.npy"
    path.write_bytes(b"previous export")
    token = SimpleNamespace(cancelled=False)
    source = np.arange(6)
    monkeypatch.setattr(numpy_save.os, "fsync", lambda _fd: setattr(token, "cancelled", True))
    with pytest.raises(EvaluationCancelled):
        save_derived_array(path, source, cancellation_token=token)
    assert path.read_bytes() == b"previous export"
    token.cancelled = False
    monkeypatch.undo()
    replace = numpy_save.os.replace

    def cancel_at_commit(staged, target):
        replace(staged, target)
        token.cancelled = True

    monkeypatch.setattr(numpy_save.os, "replace", cancel_at_commit)
    assert save_derived_array(path, source, cancellation_token=token) == (path,)
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), source)


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_object_exports_are_refused_without_pickle_or_destination_changes(tmp_path, suffix):
    path = tmp_path / f"previous{suffix}"
    path.write_bytes(b"previous export")
    with pytest.raises(ValueError, match="without pickle"):
        save_derived_array(path, np.array([{"unsafe": 1}], dtype=object))
    assert path.read_bytes() == b"previous export"
    assert set(tmp_path.iterdir()) == {path}
