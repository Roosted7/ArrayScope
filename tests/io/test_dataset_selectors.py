import io
import zipfile

import h5py
import numpy as np
import pytest

from arrayscope.io.selectors import H5DatasetSelector, MatDatasetSelector, NpzDatasetSelector


@pytest.mark.parametrize("dataset_path", ["image", "acquisition/image", "scan/session/image"])
def test_hdf5_numeric_datasets_load_at_any_group_depth(tmp_path, dataset_path):
    path = tmp_path / "scan.h5"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    with h5py.File(path, "w") as output:
        output.create_dataset(dataset_path, data=expected)

    selector = H5DatasetSelector(path)
    try:
        assert selector.compatible_datasets == [(dataset_path, expected.shape)]
        name, data = selector.get_single_data()
        assert name == dataset_path
        np.testing.assert_array_equal(data, expected)
    finally:
        selector.close()


@pytest.mark.parametrize("records", [1, 3])
def test_hdf5_nested_compound_fields_report_the_shape_they_load(tmp_path, records):
    path = tmp_path / "records.h5"
    values = np.zeros(records, dtype=[("samples", "f4", (2, 4)), ("id", "i4")])
    values["samples"] = np.arange(records * 8).reshape(records, 2, 4)
    expected = values["samples"][0] if records == 1 else values["samples"]
    with h5py.File(path, "w") as output:
        output.create_dataset("scan/session/records", data=values)

    selector = H5DatasetSelector(path)
    try:
        assert selector.compatible_datasets == [("scan/session/records/samples", expected.shape)]
        np.testing.assert_array_equal(selector.load_data("scan/session/records/samples"), expected)
    finally:
        selector.close()


def test_hdf5_complex_encoding_is_selectable_alongside_other_datasets(tmp_path, qapp):
    path = tmp_path / "complex.h5"
    encoded = np.zeros((2, 3), dtype=[("realdata", "f4"), ("imagdata", "f4")])
    encoded["realdata"] = np.arange(6).reshape(2, 3)
    encoded["imagdata"] = -2
    with h5py.File(path, "w") as output:
        output.create_dataset("encoded", data=encoded)
        output.create_dataset("reference", data=np.ones((2, 3)))

    selector = H5DatasetSelector(path)
    try:
        dialog = selector._create_dialog()
        item = selector.tree.topLevelItem(0)
        assert item.text(0) == "encoded"
        assert item.data(0, selector.QtCore.Qt.ItemDataRole.UserRole + 1) is True
        selector._on_item_double_clicked(item, 0, dialog)
        assert selector.selected_dataset == "encoded"
        assert dialog.result() == selector.QtWidgets.QDialog.DialogCode.Accepted
        np.testing.assert_array_equal(
            selector.load_data(selector.selected_dataset),
            encoded["realdata"] + 1j * encoded["imagdata"],
        )
    finally:
        selector.close()


def test_hdf5_browsing_compound_array_fields_does_not_read_them(tmp_path, qapp, monkeypatch):
    path = tmp_path / "compound.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("records", (3,), dtype=[("image", "f4", (5, 6))])
        output.create_dataset("reference", data=np.ones((2, 3)))

    def unexpected_read(*args, **kwargs):
        pytest.fail("Browsing metadata must not materialize a compound array field")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", unexpected_read)
    selector = H5DatasetSelector(path)
    try:
        selector._create_dialog()
        field = selector.tree.topLevelItem(0).child(0)
        assert field.text(1) == "[3 × 5 × 6]"
    finally:
        selector.close()


@pytest.mark.parametrize("encoding", ["split_dimension", "native", "compound"])
def test_hdf5_auto_complex_dimension_only_applies_to_real_arrays(tmp_path, encoding):
    path = tmp_path / "complex.h5"
    if encoding == "split_dimension":
        values = np.zeros((4, 2), dtype=np.float32)
    elif encoding == "native":
        values = np.zeros((4, 2), dtype=np.complex64)
    else:
        values = np.zeros((4, 2), dtype=[("realdata", "f4"), ("imagdata", "f4")])
    with h5py.File(path, "w") as output:
        output.create_dataset("complex_data", data=values)

    selector = H5DatasetSelector(path)
    try:
        assert selector.interpret_as_complex == (1 if encoding == "split_dimension" else None)
    finally:
        selector.close()


@pytest.mark.parametrize("shape", [None, (), (0,)])
@pytest.mark.parametrize("encoding", ["array_field", "complex_fields"])
def test_hdf5_null_scalar_and_empty_records_do_not_hide_valid_arrays(tmp_path, shape, encoding):
    path = tmp_path / "mixed.h5"
    dtype = np.dtype(
        [("samples", "f4", (3, 4))]
        if encoding == "array_field"
        else [("realdata", "f4"), ("imagdata", "f4")]
    )
    with h5py.File(path, "w") as output:
        if shape is None:
            output.create_dataset("metadata", data=h5py.Empty(dtype))
        else:
            output.create_dataset("metadata", shape, dtype=dtype)
        output.create_dataset("image", data=np.arange(6).reshape(2, 3))

    selector = H5DatasetSelector(path)
    try:
        expected = [("image", (2, 3))]
        if encoding == "array_field" and shape == ():
            expected.append(("metadata/samples", (3, 4)))
        assert selector.compatible_datasets == expected
    finally:
        selector.close()


@pytest.mark.parametrize("record_shape", [(), (1,), (3,)])
def test_hdf5_complex_array_fields_publish_the_shape_and_dtype_they_load(tmp_path, record_shape):
    path = tmp_path / "complex-records.h5"
    encoded = np.zeros(record_shape, dtype=[("realdata", "f4", (3, 4)), ("imagdata", "f4", (1, 4))])
    encoded["realdata"] = 3
    encoded["imagdata"] = -2
    expected = encoded["realdata"] + 1j * encoded["imagdata"]
    with h5py.File(path, "w") as output:
        output.create_dataset("acquisition/encoded", data=encoded)

    selector = H5DatasetSelector(path)
    try:
        assert selector.compatible_datasets == [("acquisition/encoded", expected.shape)]
        np.testing.assert_array_equal(selector.load_data("acquisition/encoded"), expected)
    finally:
        selector.close()


def test_hdf5_real_and_imaginary_names_do_not_make_string_fields_numeric(tmp_path):
    path = tmp_path / "strings.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("labels", (2, 3), dtype=[("realdata", "S3"), ("imagdata", "S3")])
        output.create_dataset("image", data=np.ones((2, 3)))

    selector = H5DatasetSelector(path)
    try:
        assert selector.compatible_datasets == [("image", (2, 3))]
    finally:
        selector.close()


def test_dataset_browser_filters_full_paths_keeps_groups_and_reports_size(
    tmp_path, qtbot, monkeypatch
):
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("acquisition/t1", (3, 4), dtype="f4")
        output.create_dataset("acquisition/t2", (3, 4), dtype="f4")
        output.create_dataset("reference/image", (4, 5), dtype="c8")
        output.create_dataset("metadata", data=h5py.Empty(np.dtype([("samples", "f4", (3, 4))])))

    def unexpected_read(*_args, **_kwargs):
        pytest.fail("Browsing and filtering must use metadata without source reads")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", unexpected_read)
    selector = H5DatasetSelector(path)
    try:
        dialog = selector._create_dialog()
        qtbot.addWidget(dialog)
        dialog.show()
        selector.filter_edit.setText("ACQUISITION/T2")
        acquisition = selector.tree.topLevelItem(0)
        assert not acquisition.isHidden()
        assert acquisition.child(0).isHidden()
        target = acquisition.child(1)
        assert not target.isHidden()
        assert selector.tree.topLevelItem(2).isHidden()
        assert selector.tree.currentItem() is target
        assert selector.match_count.text() == "1 of 3 datasets"
        assert target.text(3) == "48 B"
        assert target.toolTip(0) == "acquisition/t2"
        assert selector.open_button.isEnabled()

        selector.filter_edit.setText("missing")
        assert selector.match_count.text() == "0 of 3 datasets"
        assert selector.empty_state.isVisible()
        assert not selector.open_button.isEnabled()

        selector.filter_edit.clear()
        assert selector.match_count.text() == "3 datasets"
        assert not selector.empty_state.isVisible()
        assert selector.open_button.isEnabled()
    finally:
        selector.close()


@pytest.mark.parametrize("activation", ["enter", "open_button", "double_click"])
def test_dataset_browser_opens_the_filtered_choice_with_keyboard_or_pointer(
    tmp_path, qtbot, activation
):
    path = tmp_path / "arrays.npz"
    np.savez(path, first=np.ones((2, 3)), selected=np.ones((3, 4)))
    selector = NpzDatasetSelector(path)
    try:
        dialog = selector._create_dialog()
        qtbot.addWidget(dialog)
        dialog.show()
        selector.filter_edit.setFocus()
        qtbot.keyClicks(selector.filter_edit, "selected")
        with qtbot.waitSignal(dialog.accepted):
            if activation == "enter":
                qtbot.keyClick(selector.filter_edit, selector.QtCore.Qt.Key.Key_Return)
            elif activation == "open_button":
                qtbot.mouseClick(selector.open_button, selector.QtCore.Qt.MouseButton.LeftButton)
            else:
                item = selector.tree.currentItem()
                position = selector.tree.visualItemRect(item).center()
                qtbot.mouseClick(
                    selector.tree.viewport(),
                    selector.QtCore.Qt.MouseButton.LeftButton,
                    pos=position,
                )
                qtbot.mouseDClick(
                    selector.tree.viewport(),
                    selector.QtCore.Qt.MouseButton.LeftButton,
                    pos=position,
                )
        assert selector.selected_dataset == "selected"
    finally:
        selector.close()


def test_dataset_browser_empty_archive_cannot_open_and_can_cancel(tmp_path, qtbot):
    path = tmp_path / "metadata.npz"
    np.savez(path, scalar=np.array(1), empty=np.empty((0, 3)), text=np.array("description"))
    selector = NpzDatasetSelector(path)
    try:
        dialog = selector._create_dialog()
        qtbot.addWidget(dialog)
        dialog.show()
        assert selector.match_count.text() == "0 datasets"
        assert selector.empty_state.isVisible()
        assert not selector.open_button.isEnabled()
        qtbot.keyClick(selector.filter_edit, selector.QtCore.Qt.Key.Key_Return)
        assert selector.selected_dataset is None
        with qtbot.waitSignal(dialog.rejected):
            qtbot.mouseClick(selector.cancel_button, selector.QtCore.Qt.MouseButton.LeftButton)
        assert selector.selected_dataset is None
    finally:
        selector.close()


def test_dataset_browser_can_open_an_unnamed_npz_member(tmp_path, qtbot, monkeypatch):
    path = tmp_path / "unnamed.npz"
    expected = np.arange(6).reshape(2, 3)
    np.savez(path, other=np.ones((3, 4)), **{"": expected})
    selector = NpzDatasetSelector(path)
    try:
        dialog = selector._create_dialog()
        qtbot.addWidget(dialog)
        dialog.show()
        selector.filter_edit.setText("unnamed")
        qtbot.keyClick(selector.filter_edit, selector.QtCore.Qt.Key.Key_Return)
        assert dialog.result() == selector.QtWidgets.QDialog.DialogCode.Accepted
        assert selector.selected_dataset == ""
        monkeypatch.setattr(selector, "_create_dialog", lambda: dialog)
        monkeypatch.setattr(dialog, "exec", lambda: dialog.result())
        assert selector.show() == ""
        np.testing.assert_array_equal(selector.load_data(""), expected)
    finally:
        selector.close()


@pytest.mark.parametrize("layout", ["metadata", "struct", "nested_struct", "struct_array"])
def test_mat_single_numeric_leaf_opens_with_metadata_or_structs(tmp_path, layout):
    from scipy.io import savemat

    path = tmp_path / "scan.mat"
    expected = np.arange(6).reshape(2, 3)
    if layout == "metadata":
        variables = {"image": expected, "description": "Example scan"}
        expected_path = "image"
    elif layout == "struct":
        variables = {"scan": {"image": expected, "description": "Example scan"}}
        expected_path = "scan/image"
    elif layout == "nested_struct":
        variables = {"scan": {"derived": {"image": expected}}}
        expected_path = "scan/derived/image"
    else:
        records = np.empty((1, 2), dtype=[("id", object)])
        records["id"][0, 0] = 1
        records["id"][0, 1] = 2
        variables = {"image": expected, "records": records}
        expected_path = "image"
    savemat(path, variables)

    selector = MatDatasetSelector(path)
    assert selector.compatible_datasets == [(expected_path, expected.shape, expected.dtype)]
    assert not selector.requires_gui()
    name, actual = selector.get_single_data()
    assert name == expected_path
    np.testing.assert_array_equal(actual, expected)


def test_mat_browser_lists_nested_numeric_leaves_and_marks_struct_arrays(tmp_path, qtbot):
    from scipy.io import savemat

    path = tmp_path / "scan.mat"
    records = np.empty((1, 2), dtype=[("id", object)])
    records["id"][0, 0] = 1
    records["id"][0, 1] = 2
    savemat(
        path,
        {
            "scan": {"image": np.ones((2, 3)), "derived": {"phase": np.ones((3, 4))}},
            "records": records,
        },
    )
    selector = MatDatasetSelector(path)
    assert selector.requires_gui()
    assert selector.get_single_data() is None
    assert {path for path, _shape, _dtype in selector.compatible_datasets} == {
        "scan/image",
        "scan/derived/phase",
    }
    dialog = selector._create_dialog()
    qtbot.addWidget(dialog)
    dialog.show()
    assert selector.match_count.text() == "2 datasets"
    unsupported = selector.tree.topLevelItem(0)
    assert unsupported.text(0) == "records"
    assert unsupported.text(2) == "Unsupported"
    assert not unsupported.data(0, selector.QtCore.Qt.ItemDataRole.UserRole + 1)
    selector.filter_edit.setText("scan/derived/phase")
    assert selector.match_count.text() == "1 of 2 datasets"
    assert selector.tree.currentItem().text(0) == "phase"
    np.testing.assert_array_equal(selector.load_data("scan/derived/phase"), np.ones((3, 4)))


def test_invalid_npz_header_closes_container_during_discovery(tmp_path, monkeypatch):
    path = tmp_path / "invalid.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("invalid.npy", b"\x93NUMPY\x01\x00\x10\x00bad")
    opened = []
    original_load = np.load

    def capture_load(*args, **kwargs):
        container = original_load(*args, **kwargs)
        opened.append(container)
        return container

    monkeypatch.setattr(np, "load", capture_load)
    with pytest.raises(ValueError, match="EOF"):
        NpzDatasetSelector(path)
    assert opened[0].zip is None


def test_truncated_npz_payload_is_rejected_from_metadata_and_closes_container(
    tmp_path, monkeypatch
):
    path = tmp_path / "truncated.npz"
    member = io.BytesIO()
    np.save(member, np.arange(8))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("image.npy", member.getvalue()[:-8])
    opened = []
    original_load = np.load

    def capture_load(*args, **kwargs):
        container = original_load(*args, **kwargs)
        opened.append(container)
        return container

    monkeypatch.setattr(np, "load", capture_load)
    with pytest.raises(ValueError, match=r"image.*truncated"):
        NpzDatasetSelector(path)
    assert opened[0].zip is None


def test_npz_browsing_reads_headers_only_and_ignores_non_numeric_members(
    tmp_path, qapp, monkeypatch
):
    path = tmp_path / "archive.npz"
    expected = np.arange(24, dtype=np.complex64).reshape(2, 3, 4)
    np.savez_compressed(
        path,
        **{
            "session/image": expected,
            "volume.npy": expected.real,
            "scalar": np.array(2),
            "empty": np.empty((0, 3)),
            "description": np.array("An archive with numeric and non-numeric members"),
            "objects": np.array([{"unsafe": "do not unpickle"}], dtype=object),
        },
    )
    # NpzFile also supports raw member names without .npy and arbitrary bytes.
    with zipfile.ZipFile(path, "a") as archive:
        member = io.BytesIO()
        np.save(member, expected)
        archive.writestr("raw-member", member.getvalue())
        archive.writestr("notes.txt", "Not an array")

    with np.load(path) as archive:
        archive_type = type(archive)

    def unexpected_decode(*args, **kwargs):
        pytest.fail("Browsing an NPZ archive must not decode array payloads")

    with monkeypatch.context() as guard:
        guard.setattr(archive_type, "__getitem__", unexpected_decode)
        selector = NpzDatasetSelector(path)
        selector._create_dialog()
        assert [name for name, _shape, _dtype in selector.compatible_datasets] == [
            "session/image",
            "volume.npy",
            "raw-member",
        ]
    try:
        np.testing.assert_array_equal(selector.load_data("session/image"), expected)
        np.testing.assert_array_equal(selector.load_data("volume.npy"), expected.real)
        np.testing.assert_array_equal(selector.load_data("raw-member"), expected)
    finally:
        selector.close()


@pytest.mark.parametrize("kind", ["h5", "npz"])
@pytest.mark.parametrize("outcome", ["cancel", "read_error"])
def test_dataset_selection_closes_the_container_on_cancel_or_error(
    tmp_path, qapp, monkeypatch, kind, outcome
):
    path = tmp_path / f"archive.{kind}"
    hdf5_handles = h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE)
    if kind == "h5":
        with h5py.File(path, "w") as output:
            output.create_dataset("a", data=np.ones((2, 3)))
            output.create_dataset("b", data=np.ones((3, 4)))
        selector = H5DatasetSelector(path)
    else:
        np.savez(path, a=np.ones((2, 3)), b=np.ones((3, 4)))
        selector = NpzDatasetSelector(path)
    if outcome == "cancel":
        monkeypatch.setattr(selector, "show", lambda: None)
        assert selector.view(block=True) is False
    else:
        monkeypatch.setattr(selector, "show", lambda: "a")

        def fail_read(_path):
            raise OSError("container read failed")

        monkeypatch.setattr(selector, "load_selected", fail_read)
        with pytest.raises(OSError, match="container read failed"):
            selector.view(block=True)
    if kind == "h5":
        assert h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE) == hdf5_handles
    else:
        assert selector.npz_file.zip is None
