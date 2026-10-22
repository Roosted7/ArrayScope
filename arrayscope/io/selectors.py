"""
arrayscope GUI for choosing datasets from multi-dataset files.
Handles mat, h5, and npz files.
"""

import logging
from contextlib import closing
from math import prod

import numpy as np

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.core.memory_budget import format_bytes

_LOGGER = logging.getLogger(__name__)


def _show_selector(filepath, selector_class_name, interpret_as_complex):
    prefer_pyside6()

    from arrayscope.app.launch import arrayscope

    # Import appropriate selector class (not checking filename twice with this approach)
    if selector_class_name == "H5DatasetSelector":
        from arrayscope.io.selectors import H5DatasetSelector

        selector_class = H5DatasetSelector
    elif selector_class_name == "NpzDatasetSelector":
        from arrayscope.io.selectors import NpzDatasetSelector

        selector_class = NpzDatasetSelector
    elif selector_class_name == "MatDatasetSelector":
        from arrayscope.io.selectors import MatDatasetSelector

        selector_class = MatDatasetSelector
    else:
        return

    # Create QApplication
    import pyqtgraph as pg

    pg.mkQApp()

    # Create selector
    selector = selector_class(filepath)
    selector.interpret_as_complex = interpret_as_complex

    with closing(selector):
        selected_path = selector.show()
        data = selector.load_data(selected_path) if selected_path is not None else None
    if selected_path is not None:
        arrayscope(
            data=data,
            title=f"{filepath.name} - {selected_path}",
            block=True,
            complex_dim=interpret_as_complex,
            filepath=filepath,
            dataset_path=selected_path,
            selector_class_name=selector_class_name,
        )


class DatasetSelector:
    """Base class for GUI dialog to select datasets from multi-dataset files."""

    COLOR_INCOMPATIBLE = (180, 180, 180)  # Light grey for incompatible items

    def __init__(self, filepath, compatible_datasets=None):
        prefer_pyside6()

        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

        self.QtWidgets = QtWidgets
        self.QtCore = QtCore
        self.QtGui = QtGui

        self.filepath = filepath
        self.compatible_datasets = compatible_datasets or []
        self.selected_dataset = None
        self.interpret_as_complex = None  # Dimension to interpret as complex (real/imag split)

    def requires_gui(self):
        """Check if GUI selector is needed (more than one compatible dataset)."""
        return len(self.compatible_datasets) > 1

    def get_single_data(self):
        """Get data if there's only one compatible dataset. Returns (name, data) or None."""
        raise NotImplementedError("Subclasses must implement get_single_data()")

    def load_data(self, path):
        """Load data for a given dataset path/name. Returns numpy array."""
        raise NotImplementedError("Subclasses must implement load_data()")

    def close(self):
        """Close any open file handles. Subclasses override if needed."""

    def _build_tree(self):
        """Build the tree widget. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_tree()")

    def show(self):
        """Show the selector dialog and return the selected dataset name."""
        dialog = self._create_dialog()
        result = dialog.exec()

        if (
            result == self.QtWidgets.QDialog.DialogCode.Accepted
            and self.selected_dataset is not None
        ):
            return self.selected_dataset
        return None

    def view(self, block=False):
        """
        Select dataset (with GUI if needed) and open in arrayscope.
        Returns True if successful, False otherwise.
        """
        from arrayscope.app.launch import arrayscope

        # The container belongs to this selection attempt, including cancel,
        # no-compatible-data, and read/process-launch failure paths. Loaded
        # arrays are detached before the viewer opens.
        with closing(self):
            if not self.requires_gui():
                result = self.get_single_data()
            elif block:
                import pyqtgraph as pg

                pg.mkQApp()
                selected_path = self.show()
                result = (
                    (selected_path, self.load_data(selected_path))
                    if selected_path is not None
                    else None
                )
            else:
                import multiprocessing as mp
                import multiprocessing.process as _mp_process

                p = mp.Process(
                    target=_show_selector,
                    args=(self.filepath, self.__class__.__name__, self.interpret_as_complex),
                )
                p.start()
                _mp_process._children.discard(p)
                return True

        if result is None:
            return False
        name, data = result
        arrayscope(
            data=data,
            title=f"{self.filepath.name} - {name}",
            block=block,
            complex_dim=self.interpret_as_complex,
            filepath=self.filepath,
            dataset_path=name,
            selector_class_name=self.__class__.__name__,
        )
        return True

    def _create_dialog(self):
        """Create and configure the dialog widget."""
        self.selected_dataset = None
        dialog = self.QtWidgets.QDialog()
        dialog.setWindowTitle(f"Select Dataset - {self.filepath.name}")
        dialog.resize(760, 460)
        dialog.setMinimumSize(520, 340)

        layout = self.QtWidgets.QVBoxLayout()

        self.filter_edit = self.QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter datasets by name or path…")
        self.filter_edit.setAccessibleName("Search datasets")
        self.filter_edit.setClearButtonEnabled(True)
        layout.addWidget(self.filter_edit)

        # Tree widget
        self.tree = self.QtWidgets.QTreeWidget()
        self.tree.setAccessibleName("Datasets")
        self.tree.setHeaderLabels(["Name", "Shape", "Type", "Memory"])
        self.tree.headerItem().setToolTip(
            3, "Estimated array size in memory; compressed file size may differ."
        )
        self.tree.setUniformRowHeights(True)
        self.tree.setColumnWidth(0, 280)
        self.tree.setColumnWidth(1, 175)
        self.tree.setColumnWidth(2, 110)
        self.tree.itemDoubleClicked.connect(
            lambda item, col: self._on_item_double_clicked(item, col, dialog)
        )
        self.tree.itemActivated.connect(
            lambda item, col: self._on_item_double_clicked(item, col, dialog)
        )

        # Build tree (implemented by subclasses)
        self._build_tree()

        layout.addWidget(self.tree, 1)

        self.empty_state = self.QtWidgets.QLabel()
        self.empty_state.setWordWrap(True)
        layout.addWidget(self.empty_state)

        footer = self.QtWidgets.QHBoxLayout()
        self.match_count = self.QtWidgets.QLabel()
        self.match_count.setAccessibleName("Dataset count")
        footer.addWidget(self.match_count)
        footer.addStretch()
        buttons = self.QtWidgets.QDialogButtonBox(
            self.QtWidgets.QDialogButtonBox.StandardButton.Open
            | self.QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.open_button = buttons.button(self.QtWidgets.QDialogButtonBox.StandardButton.Open)
        self.cancel_button = buttons.button(self.QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.open_button.setDefault(True)
        buttons.accepted.connect(lambda: self._accept_dataset(dialog))
        buttons.rejected.connect(dialog.reject)
        footer.addWidget(buttons)
        layout.addLayout(footer)

        dialog.setLayout(layout)
        self.filter_edit.textChanged.connect(self._filter_dataset_tree)
        self.tree.currentItemChanged.connect(lambda _current, _previous: self._update_open_button())
        self._filter_dataset_tree("")
        self.filter_edit.setFocus()
        return dialog

    def _on_item_double_clicked(self, item, column, dialog):
        """Route keyboard activation and double-click through the Open action."""
        self.tree.setCurrentItem(item)
        self._accept_dataset(dialog)

    def _accept_dataset(self, dialog):
        item = self.tree.currentItem()
        if (
            self.selected_dataset is None
            and item is not None
            and not item.isHidden()
            and item.data(0, self.QtCore.Qt.ItemDataRole.UserRole + 1)
        ):
            self.selected_dataset = item.data(0, self.QtCore.Qt.ItemDataRole.UserRole)
            dialog.accept()

    def _update_open_button(self):
        item = self.tree.currentItem()
        self.open_button.setEnabled(
            item is not None
            and not item.isHidden()
            and bool(item.data(0, self.QtCore.Qt.ItemDataRole.UserRole + 1))
        )

    def _filter_dataset_tree(self, text):
        query = text.strip().casefold()
        matches = []
        total = 0
        role = self.QtCore.Qt.ItemDataRole.UserRole

        def filter_item(item):
            nonlocal total
            child_matches = [filter_item(item.child(index)) for index in range(item.childCount())]
            path = item.data(0, role) or ""
            own_match = query in str(path).casefold() or query in item.text(0).casefold()
            visible = any(child_matches) if child_matches else own_match
            item.setHidden(not visible)
            if query and child_matches and visible:
                item.setExpanded(True)
            if item.data(0, role + 1):
                total += 1
                if visible:
                    matches.append(item)
            return visible

        for index in range(self.tree.topLevelItemCount()):
            filter_item(self.tree.topLevelItem(index))
        if self.tree.currentItem() not in matches:
            self.tree.setCurrentItem(matches[0] if matches else None)
        count = len(matches)
        noun = "dataset" if total == 1 else "datasets"
        self.match_count.setText(f"{count} of {total} {noun}" if query else f"{total} {noun}")
        self.empty_state.setText(
            "No datasets match this search. Try a different name or path."
            if total
            else "No compatible arrays found in this file."
        )
        self.empty_state.setVisible(count == 0)
        self._update_open_button()

    def _add_group(self, parent, name, path=None, item_type="Group"):
        """Add a group/struct item to the tree (non-selectable, greyed out, expanded)."""
        item = self.QtWidgets.QTreeWidgetItem([name, "", item_type])
        item.setData(0, self.QtCore.Qt.ItemDataRole.UserRole, path)
        item.setData(0, self.QtCore.Qt.ItemDataRole.UserRole + 1, False)  # Not compatible
        if path is not None:
            item.setToolTip(0, str(path))

        # Customize group appearance
        item.setFlags(item.flags() & ~self.QtCore.Qt.ItemFlag.ItemIsSelectable)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setFont(1, font)
        item.setFont(2, font)

        if isinstance(parent, self.QtWidgets.QTreeWidget):
            parent.addTopLevelItem(item)
        else:
            parent.addChild(item)

        item.setExpanded(True)
        return item

    def _get_shape_display(self, shape, data=None):
        """Get display string for shape, or scalar value if applicable."""
        if shape is None:
            return "Empty dataset"
        # Show scalar value if multiple datasets and data is scalar
        if (
            data is not None
            and self.requires_gui()
            and isinstance(shape, tuple)
            and np.prod(shape) == 1
        ):
            try:
                arr = np.asarray(data)
                if arr.size == 1:
                    value = arr.flat[0]
                    # Format string values (truncate to 100 chars)
                    if isinstance(value, bytes):
                        value_str = value.decode("utf-8", errors="replace")
                    elif isinstance(value, str):
                        value_str = value
                    else:
                        value_str = str(value)

                    if len(value_str) > 100:
                        value_str = value_str[:97] + "..."
                    return value_str
            except Exception:
                pass

        # Format shape normally
        if isinstance(shape, tuple):
            if not shape:
                return "Scalar"
            return f"[{' × '.join(map(str, shape))}]"
        else:
            return f"[{shape}]"

    def _add_item(self, parent, name, shape, dtype, path, compatible=True, data=None):
        display_text = self._get_shape_display(shape, data)

        # Create item
        prefix = ""  # "✓ " if compatible else "  "
        item = self.QtWidgets.QTreeWidgetItem([f"{prefix}{name or '(unnamed)'}"])
        item.setText(1, display_text)
        item.setText(2, str(dtype))
        item.setData(0, self.QtCore.Qt.ItemDataRole.UserRole, path)
        item.setData(0, self.QtCore.Qt.ItemDataRole.UserRole + 1, compatible)
        if path is not None:
            item.setToolTip(0, str(path))
        if isinstance(shape, tuple) and np.dtype(dtype).kind in "biufc":
            nbytes = prod(shape) * np.dtype(dtype).itemsize
            item.setText(3, format_bytes(nbytes))
            item.setToolTip(3, f"{nbytes:,} bytes in memory")
        else:
            item.setText(3, "—")

        if not compatible:  # Mark incompatible items
            item.setFlags(item.flags() & ~self.QtCore.Qt.ItemFlag.ItemIsEnabled)
            item.setForeground(0, self.QtGui.QBrush(self.QtGui.QColor(*self.COLOR_INCOMPATIBLE)))
            item.setForeground(1, self.QtGui.QBrush(self.QtGui.QColor(*self.COLOR_INCOMPATIBLE)))
            item.setForeground(2, self.QtGui.QBrush(self.QtGui.QColor(*self.COLOR_INCOMPATIBLE)))

        if isinstance(parent, self.QtWidgets.QTreeWidget):
            parent.addTopLevelItem(item)
        else:
            parent.addChild(item)

        return item


class H5DatasetSelector(DatasetSelector):
    def __init__(self, filepath):
        try:
            import h5py as h5
        except ImportError as err:
            raise ImportError(
                "h5py is required to read HDF5 files. Install it with: pip install h5py"
            ) from err

        self.h5 = h5
        self.h5_file = h5.File(filepath, "r")
        try:
            self.compatible_datasets = self._find_compatible_datasets()
            self.compatible_paths = {path for path, _ in self.compatible_datasets}
            super().__init__(filepath, self.compatible_datasets)
            self._detect_complex_interpretation()
        except BaseException:
            self.close()
            raise

    def __del__(self):
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

    def close(self):
        if hasattr(self, "h5_file") and self.h5_file:
            self.h5_file.close()

    def _detect_complex_interpretation(self):
        """
        If there's a single compatible dataset with 'complex' in name and a size-2 dimension,
        auto-set interpret_as_complex to that dimension.
        This is how ISMRMD stores complex data.
        """
        if len(self.compatible_datasets) != 1:
            return

        dataset_path, shape = self.compatible_datasets[0]

        if "complex" not in dataset_path.lower():
            return

        dataset, field_name = self._dataset_and_field(dataset_path)
        dtype = (
            dataset.dtype
            if field_name is None
            else self._get_compound_field_info(dataset, field_name)[1]
        )
        # Native complex arrays and real/imag record encodings already carry
        # both components. Their size-two image axis is still an image axis.
        if dtype.kind == "c" or dtype.names:
            return

        # Find first dimension with size 2
        for dim_idx, size in enumerate(shape):
            if size == 2:
                self.interpret_as_complex = dim_idx
                _LOGGER.info(
                    "Auto-detected complex interpretation: dimension %s of '%s' (shape %s)",
                    dim_idx,
                    dataset_path,
                    shape,
                )
                break

    def _is_compound_dataset(self, dataset):
        """Check if dataset is compound type with array fields."""
        return hasattr(dataset, "dtype") and dataset.dtype.names is not None

    def _get_compound_field_info(self, dataset, field_name):
        """Get shape and dtype for a compound field. Returns (shape, dtype, is_array)."""
        field_dtype = dataset.dtype.fields[field_name][0]
        if dataset.shape is None:
            return (None, field_dtype.base, False)

        # Check if field has subdtype (array within compound)
        if hasattr(field_dtype, "subdtype") and field_dtype.subdtype is not None:
            base_dtype, shape = field_dtype.subdtype
            is_array = (
                isinstance(shape, tuple)
                and len(shape) >= 1
                and np.issubdtype(base_dtype, np.number)
            )
            # load_data removes a single record axis for the common ISMRMRD
            # encoding; multiple records remain part of the returned array.
            record_shape = () if dataset.shape == (1,) else dataset.shape
            return (record_shape + shape, base_dtype, is_array)
        # Scalar field
        return (dataset.shape, field_dtype, False)

    def _complex_encoding_info(self, dataset):
        """Return real/imag field names and actual result metadata, without reads."""
        if dataset.shape is None:
            return None
        names = dataset.dtype.names or ()
        real_name = next((name for name in ("real", "realdata", "r") if name in names), None)
        imag_name = next((name for name in ("imag", "imagdata", "i") if name in names), None)
        if real_name is None or imag_name is None:
            return None
        real_dtype = dataset.dtype.fields[real_name][0]
        imag_dtype = dataset.dtype.fields[imag_name][0]
        if not all(np.issubdtype(dtype.base, np.number) for dtype in (real_dtype, imag_dtype)):
            return None
        try:
            result_shape = np.broadcast_shapes(
                dataset.shape + real_dtype.shape, dataset.shape + imag_dtype.shape
            )
        except ValueError:
            return None
        # A zero-length array models NumPy's Python-scalar promotion for
        # 1j * imag; it never samples source values.
        imag_complex_dtype = np.result_type(np.empty((0,), dtype=imag_dtype.base), 1j)
        result_dtype = np.result_type(real_dtype.base, imag_complex_dtype)
        return real_name, imag_name, result_shape, result_dtype

    def _find_compatible_datasets(self):
        """Find all compatible datasets and compound array fields."""
        import h5py as h5

        compatible = []

        def visit_func(name, obj):
            if isinstance(obj, h5.Dataset):
                if obj.shape is None:
                    return
                # Only include numeric arrays with more than 1 element (strings are always incompatible)
                is_compatible = (
                    obj.ndim >= 1 and np.issubdtype(obj.dtype, np.number) and np.prod(obj.shape) > 1
                )

                if self._is_compound_dataset(obj):
                    complex_info = self._complex_encoding_info(obj)
                    if complex_info is not None and prod(complex_info[2]) > 1:
                        compatible.append((name, complex_info[2]))
                    else:
                        # Still allow array fields (e.g. for ISMRMRD style)
                        for field_name in obj.dtype.names:
                            shape, _dtype, is_array = self._get_compound_field_info(obj, field_name)
                            if is_array and np.prod(shape) > 1:
                                field_path = f"{name}/{field_name}"
                                compatible.append((field_path, shape))
                elif is_compatible:
                    compatible.append((name, obj.shape))

        self.h5_file.visititems(visit_func)
        return compatible

    def get_single_data(self):
        if len(self.compatible_datasets) == 1:
            path = self.compatible_datasets[0][0]
            data = self.load_data(path)
            return (path, data)
        return None

    def _dataset_and_field(self, path):
        # A slash usually separates HDF5 groups, not a compound field.
        # Resolve the actual object first; only a dataset parent can own a
        # field, at any group depth.
        dataset_path, separator, field_name = path.rpartition("/")
        parent = self.h5_file.get(dataset_path) if separator else None
        if isinstance(parent, self.h5.Dataset) and field_name in (parent.dtype.names or ()):
            return parent, field_name
        return self.h5_file[path], None

    def load_data(self, path):
        dset, field_name = self._dataset_and_field(path)
        if field_name is not None:
            field_data = dset[field_name]
            # Remove singleton dataset dimension if present
            if dset.shape == (1,):
                field_data = field_data[0]
            return np.array(field_data)

        # Handle native complex arrays
        if np.iscomplexobj(dset):
            return np.array(dset)

        if complex_info := self._complex_encoding_info(dset):
            real_name, imag_name, _shape, _dtype = complex_info
            return dset[real_name] + 1j * dset[imag_name]

        return np.array(dset)

    def _build_tree(self):

        def add_items(parent_item, h5_group, parent_path=""):
            for key in h5_group:
                item = h5_group[key]
                full_path = f"{parent_path}/{key}" if parent_path else key

                if isinstance(item, self.h5.Group):
                    group_item = self._add_group(parent_item, key, full_path, "Group")
                    add_items(group_item, item, full_path)

                elif isinstance(item, self.h5.Dataset):
                    is_compatible = full_path in self.compatible_paths

                    # Add compound datasets as groups (like MATLAB structs)
                    if self._is_compound_dataset(item) and not is_compatible:
                        dataset_item = self._add_group(parent_item, key, full_path, "Compound")
                        dataset_item.setExpanded(True)

                        # Add all compound fields as children
                        for field_name in item.dtype.names:
                            field_path = f"{full_path}/{field_name}"
                            shape, dtype, _is_array = self._get_compound_field_info(
                                item, field_name
                            )
                            field_compatible = field_path in self.compatible_paths

                            # Show all fields (arrays as enabled, scalars as disabled)
                            self._add_item(
                                dataset_item,
                                field_name,
                                shape,
                                dtype,
                                field_path,
                                field_compatible,
                                data=None,
                            )
                    else:
                        complex_info = self._complex_encoding_info(item)
                        shape = item.shape if complex_info is None else complex_info[2]
                        dtype = item.dtype if complex_info is None else complex_info[3]
                        self._add_item(
                            parent_item,
                            key,
                            shape,
                            dtype,
                            full_path,
                            is_compatible,
                            data=None,
                        )

        add_items(self.tree, self.h5_file)


class NpzDatasetSelector(DatasetSelector):
    def __init__(self, filepath):
        self.npz_file = np.load(filepath)
        try:
            compatible_datasets = self._find_compatible_datasets()
            super().__init__(filepath, compatible_datasets)
        except BaseException:
            self.close()
            raise

    def __del__(self):
        """Close the NPZ file when selector is destroyed."""
        if hasattr(self, "npz_file"):
            self.npz_file.close()

    def close(self):
        if hasattr(self, "npz_file"):
            self.npz_file.close()

    def _find_compatible_datasets(self):
        """Inspect member headers without decompressing their array payloads."""
        from numpy.lib.format import MAGIC_PREFIX

        from arrayscope.io.progressive import _read_npy_header

        compatible = []
        member_names = set(self.npz_file.zip.namelist())
        for key in self.npz_file.files:
            # Match NpzFile lookup: an exact member name wins over adding
            # .npy. This also preserves slash-containing and raw-member keys.
            member_name = key if key in member_names else key + ".npy"
            with self.npz_file.zip.open(member_name) as member:
                if member.read(len(MAGIC_PREFIX)) != MAGIC_PREFIX:
                    continue
                member.seek(0)
                shape, _fortran_order, dtype = _read_npy_header(member)
                payload_bytes = self.npz_file.zip.getinfo(member_name).file_size - member.tell()
            # Object members are safely classified from their header; never
            # unpickle metadata just to offer the numeric arrays beside it.
            if len(shape) >= 1 and np.issubdtype(dtype, np.number) and prod(shape) > 1:
                expected_bytes = prod(shape) * dtype.itemsize
                if payload_bytes < expected_bytes:
                    raise ValueError(
                        f"NPZ array {key!r} is truncated: expected {expected_bytes} data bytes, "
                        f"found {payload_bytes}"
                    )
                compatible.append((key, shape, dtype))
        return compatible

    def get_single_data(self):
        if len(self.compatible_datasets) == 1:
            name = self.compatible_datasets[0][0]
            data = self.load_data(name)
            return (name, data)
        return None

    def load_data(self, name):
        """Load data for a given array name."""
        return np.asarray(self.npz_file[name])

    def _build_tree(self):
        for name, shape, dtype in self.compatible_datasets:  # flat and simple
            self._add_item(self.tree, name, shape, dtype, name, compatible=True)


class MatDatasetSelector(DatasetSelector):
    def __init__(self, filepath):
        try:
            import scipy.io
        except ImportError as err:
            raise ImportError(
                "scipy is required to read .mat files. Install it with: pip install scipy"
            ) from err

        try:
            mat_data = scipy.io.loadmat(filepath)
            self.mat_dict = {k: v for k, v in mat_data.items() if not k.startswith("__")}

            compatible_datasets = self._find_compatible_datasets()
            super().__init__(filepath, compatible_datasets)
        except NotImplementedError:
            # MATLAB v7.3 files are HDF5-based, fall back to H5DatasetSelector
            try:
                self.__class__ = H5DatasetSelector
                H5DatasetSelector.__init__(self, filepath)
            except Exception as e:
                raise NotImplementedError(
                    f"Failed to load {filepath}. MATLAB v7.3 files are HDF5-based and require h5py. "
                    f"Details: {e}"
                ) from e

    def _is_struct(self, val):
        return (
            isinstance(val, np.ndarray)
            and hasattr(val.dtype, "names")
            and val.dtype.names is not None
        )

    def _is_numeric_array(self, val):
        return (
            isinstance(val, np.ndarray)
            and np.issubdtype(val.dtype, np.number)
            and val.ndim >= 1
            and np.prod(val.shape) > 1
        )

    def _is_string_array(self, val):
        return isinstance(val, np.ndarray) and val.dtype.kind in ("U", "S", "O") and val.ndim >= 1

    def _is_any_array(self, val):
        """Check if value is any kind of array (for display in tree)."""
        return isinstance(val, np.ndarray) and val.ndim >= 1

    def _struct_fields(self, value):
        # Scalar MATLAB structs wrap each field in a one-element object
        # array. A record array has no unambiguous field-to-array mapping;
        # leave it visible as unsupported instead of calling .item() on it.
        if value.size == 1:
            for field_name in value.dtype.names:
                yield field_name, value[field_name].item()

    def _numeric_shape(self, value):
        return tuple(size for size in value.shape if size != 1) or (1,)

    def _find_compatible_datasets(self):
        compatible = []

        def visit(path, value):
            if self._is_numeric_array(value):
                compatible.append((path, self._numeric_shape(value), value.dtype))
            elif self._is_struct(value):
                for field_name, field_value in self._struct_fields(value):
                    visit(f"{path}/{field_name}", field_value)

        for key, var in self.mat_dict.items():
            visit(key, var)
        return compatible

    def get_single_data(self):
        if len(self.compatible_datasets) == 1:
            name = self.compatible_datasets[0][0]
            return name, self.load_data(name)
        return None

    def load_data(self, path):
        """Doesn't actually load, because that's done in ctor. Just navigate the struct.""" ""
        parts = path.split("/")
        data = self.mat_dict[parts[0]]

        # Navigate through struct hierarchy
        for field_name in parts[1:]:
            data = data[field_name].item()  # .item() extracts from (1,1) struct array

        return np.atleast_1d(np.squeeze(data))  # Ensure at least 1D

    def _build_tree(self):
        for name in sorted(self.mat_dict.keys()):
            self._add_variable(name, self.mat_dict[name], self.tree)

    def _add_variable(self, name, val, parent, path=None):
        """Add variable to tree (handles structs recursively)."""
        if not isinstance(val, np.ndarray):
            return
        path = name if path is None else path

        if self._is_struct(val):
            if val.size != 1:
                item = self._add_group(parent, name, path, "Unsupported")
                item.setText(1, self._get_shape_display(val.shape))
                reason = "MATLAB struct arrays with multiple or zero records cannot be opened."
                item.setToolTip(0, f"{path}\n{reason}")
                item.setToolTip(2, reason)
                return
            item = self._add_group(parent, name, path, "struct")
            for field_name, field_val in self._struct_fields(val):
                self._add_variable(field_name, field_val, item, f"{path}/{field_name}")

        elif self._is_any_array(val):
            # Add any array (numeric compatible, strings incompatible)
            is_compatible = self._is_numeric_array(val)
            self._add_item(
                parent,
                name,
                self._numeric_shape(val) if is_compatible else val.shape,
                val.dtype,
                path,
                compatible=is_compatible,
                data=val,
            )
