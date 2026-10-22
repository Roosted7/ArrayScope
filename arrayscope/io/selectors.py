"""
arrayscope GUI for choosing datasets from multi-dataset files.
Handles mat, h5, and npz files.
"""

from contextlib import closing
from math import prod

import numpy as np

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.core.memory_budget import format_bytes


def _prepare_selector_app():
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.open_flow import ensure_open_app
    from arrayscope.app.theme import apply_theme_to_qapplication

    app = ensure_open_app()
    apply_theme_to_qapplication(app, QtCore.QSettings().value("theme", "system"))
    return app


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

    _prepare_selector_app()

    # Create selector
    selector = selector_class(filepath)
    if interpret_as_complex is not None:
        selector.interpret_as_complex = interpret_as_complex

    with closing(selector):
        if not selector.requires_gui():
            result = selector.get_single_data()
        else:
            selected_path = selector.show()
            result = (
                (selected_path, selector.load_selected(selected_path))
                if selected_path is not None
                else None
            )
    if result is not None:
        selected_path, data = result
        arrayscope(
            data=data,
            title=f"{filepath.name} - {selected_path}",
            block=True,
            complex_dim=selector.interpret_as_complex,
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

    def load_selected(self, path):
        """Open a selected dataset for viewing; subclasses may return a lazy source."""
        return self.load_data(path)

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
        # arrays and lazy descriptors remain valid after the selector closes.
        with closing(self):
            if not self.requires_gui():
                result = self.get_single_data()
            elif block:
                _prepare_selector_app()
                selected_path = self.show()
                result = (
                    (selected_path, self.load_selected(selected_path))
                    if selected_path is not None
                    else None
                )
            else:
                return self._start_selector_process()

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

    def _start_selector_process(self):
        import multiprocessing as mp
        import multiprocessing.process as _mp_process

        process = mp.Process(
            target=_show_selector,
            args=(self.filepath, self.__class__.__name__, self.interpret_as_complex),
        )
        process.start()
        _mp_process._children.discard(process)
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

        self.selection_hint = self.QtWidgets.QLabel()
        self.selection_hint.setObjectName("DatasetAccessHint")
        self.selection_hint.setTextFormat(self.QtCore.Qt.TextFormat.PlainText)
        self.selection_hint.setWordWrap(True)
        self.selection_hint.hide()
        layout.addWidget(self.selection_hint)

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
        description = self._selection_description(item)
        self.selection_hint.setText(description)
        self.selection_hint.setVisible(bool(description))
        self.open_button.setAccessibleDescription(description)

    def _selection_description(self, item):
        return ""

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
    """Metadata-only HDF5 browser with independently owned selected sources."""

    def __init__(self, filepath):
        from arrayscope.io.hdf5_catalog import catalog_hdf5

        self.catalog = catalog_hdf5(filepath)
        self.descriptors = {descriptor.key: descriptor for descriptor in self.catalog.datasets}
        self.compatible_paths = {
            key for key, descriptor in self.descriptors.items() if self._can_view(descriptor)
        }
        # A recognized real/imag record is one logical array. Its component
        # fields remain addressable by the catalog but do not duplicate GUI rows.
        complex_paths = {
            key for key in self.compatible_paths if self.descriptors[key].encoding == "complex"
        }
        self.compatible_paths = {
            key
            for key in self.compatible_paths
            if not any(key.startswith(parent + "/") for parent in complex_paths)
        }
        compatible = [(key, self.descriptors[key].shape) for key in sorted(self.compatible_paths)]
        super().__init__(self.catalog.path, compatible)
        self._detect_complex_interpretation()

    @staticmethod
    def _can_view(descriptor):
        shape = descriptor.shape
        if (
            shape is None
            or len(shape) < 1
            or prod(shape) <= 1
            or not np.issubdtype(descriptor.dtype, np.number)
            or not (descriptor.lazy_supported or descriptor.eager_supported)
        ):
            return False
        if descriptor.encoding == "field":
            # Preserve the viewer's compound-field policy: array fields are
            # scientific arrays; scalar record fields remain metadata rows.
            field_dtype = descriptor.storage_dtype.fields[descriptor.field_name][0]
            return field_dtype.subdtype is not None
        return True

    def _detect_complex_interpretation(self):
        # This historical viewer convenience is deliberately absent from the
        # Qt-free loader and batch API, where raw shape/value semantics win.
        if len(self.compatible_datasets) != 1:
            return
        key, shape = self.compatible_datasets[0]
        descriptor = self.descriptors[key]
        if "complex" not in key.lower() or descriptor.dtype.kind == "c":
            return
        for axis, size in enumerate(shape):
            if size == 2:
                self.interpret_as_complex = axis
                break

    def close(self):
        # Catalog construction and each read own their own short-lived file.
        # The browser retains immutable descriptors, never h5py objects.
        pass

    def requires_gui(self):
        # A legacy eager fallback needs its visible size/access explanation,
        # including when it is the only compatible array in the container.
        return super().requires_gui() or any(
            not self.descriptors[key].lazy_supported for key in self.compatible_paths
        )

    def get_single_data(self):
        if len(self.compatible_datasets) == 1:
            key = self.compatible_datasets[0][0]
            return key, self.load_selected(key)
        return None

    def load_selected(self, path):
        descriptor = self.catalog.dataset(path)
        if descriptor.lazy_supported:
            from arrayscope.io.hdf5_source import open_hdf5_source

            return open_hdf5_source(self.filepath, descriptor)
        if descriptor.eager_supported:
            return self.load_data(path)
        raise ValueError(descriptor.unsupported_reason or f"Unsupported HDF5 dataset: {path}")

    def load_data(self, path):
        """Preserve the explicit eager API, detached from a selector lifetime."""
        import h5py

        from arrayscope.io.hdf5_catalog import (
            HDF5SourceChanged,
            _resolve_selection,
            dataset_identity,
            file_identity,
        )

        descriptor = self.catalog.dataset(path)
        if not descriptor.eager_supported:
            raise ValueError(descriptor.unsupported_reason or "Unsupported HDF5 dataset")

        def check_file():
            try:
                unchanged = file_identity(self.filepath) == descriptor.file_identity
            except (OSError, ValueError) as exc:
                raise HDF5SourceChanged("HDF5 source is unavailable; open it again") from exc
            if not unchanged:
                raise HDF5SourceChanged("HDF5 source changed; open the dataset browser again")

        check_file()
        with h5py.File(self.filepath, "r") as container:
            check_file()
            try:
                dataset, _key, field = _resolve_selection(container, descriptor.dataset_path)
                if field is not None or dataset_identity(dataset) != descriptor.dataset_identity:
                    raise HDF5SourceChanged("HDF5 dataset metadata changed; open it again")
            except (KeyError, ValueError) as exc:
                raise HDF5SourceChanged("HDF5 dataset is unavailable; open it again") from exc
            if descriptor.field_name is not None:
                data = dataset[descriptor.field_name]
                if dataset.shape == (1,):
                    data = data[0]
                data = np.array(data)
            elif descriptor.real_field is not None and descriptor.imag_field is not None:
                data = dataset[descriptor.real_field] + 1j * dataset[descriptor.imag_field]
            else:
                data = np.array(dataset)
        check_file()
        return data

    def view(self, block=False):
        if not block:
            # Pass a file/key intent to a child, not a live adapter or its lock.
            if not self.compatible_datasets:
                return False
            return self._start_selector_process()
        return super().view(block=True)

    def _selection_description(self, item):
        if item is None:
            return ""
        key = item.data(0, self.QtCore.Qt.ItemDataRole.UserRole)
        descriptor = self.descriptors.get(key)
        if descriptor is None:
            return ""
        if key not in self.compatible_paths:
            if descriptor.lazy_supported or descriptor.eager_supported:
                return (
                    "This entry is metadata in the browser. Explicit dataset loading can read "
                    "supported scalar, empty, boolean, or scalar-record arrays."
                )
            return descriptor.unsupported_reason or "This dataset cannot be displayed as an array."
        if descriptor.lazy_supported:
            return "Opens lazily: only requested array regions are read. The array size is not a RAM allocation."
        if descriptor.eager_supported and key in self.compatible_paths:
            size = prod(descriptor.shape) * descriptor.dtype.itemsize
            return (
                f"Loads the entire selected array ({format_bytes(size)} of values), "
                "plus decoding buffers. "
                "Compound array fields do not yet support lazy reads."
            )
        return descriptor.unsupported_reason

    def _create_dialog(self):
        dialog = super()._create_dialog()
        preferred = next(
            (key for key in sorted(self.compatible_paths) if self.descriptors[key].lazy_supported),
            None,
        )
        if preferred is not None:
            self.tree.setCurrentItem(self._dataset_items[preferred])
        if missing := getattr(self, "missing_dataset_path", None):
            notice = self.QtWidgets.QLabel(
                f"The previous dataset '{missing}' is no longer available. "
                "Choose a replacement, or cancel to keep the current view."
            )
            notice.setTextFormat(self.QtCore.Qt.TextFormat.PlainText)
            notice.setWordWrap(True)
            notice.setObjectName("MissingHDF5DatasetNotice")
            dialog.layout().insertWidget(0, notice)
        return dialog

    def _build_tree(self):
        self._dataset_items = {}
        self.tree.setHeaderLabels(["Name", "Shape", "Type", "Array size", "Access"])
        self.tree.headerItem().setToolTip(
            3, "Logical uncompressed array bytes, not current RAM use or compressed file size."
        )
        self.tree.setColumnWidth(0, 230)
        self.tree.setColumnWidth(1, 150)
        self.tree.setColumnWidth(2, 100)
        self.tree.setColumnWidth(3, 100)
        self.tree.setColumnWidth(4, 85)
        nodes = dict.fromkeys(self.catalog.groups)
        hidden_parents = {
            key for key in self.compatible_paths if self.descriptors[key].encoding == "complex"
        }
        for key, descriptor in self.descriptors.items():
            if not any(key.startswith(parent + "/") for parent in hidden_parents):
                nodes[key] = descriptor
        children = {}
        for key in nodes:
            parent, _separator, _name = key.rpartition("/")
            children.setdefault(parent, []).append(key)

        def add_children(parent_item, parent_key):
            for key in sorted(children.get(parent_key, ())):
                descriptor = nodes[key]
                name = key.rsplit("/", 1)[-1]
                if descriptor is None:
                    item = self._add_group(parent_item, name, key, "Group")
                elif descriptor.dtype.names and key not in self.compatible_paths:
                    item = self._add_group(parent_item, name, key, "Compound")
                else:
                    item = self._add_item(
                        parent_item,
                        name,
                        descriptor.shape,
                        descriptor.dtype,
                        key,
                        key in self.compatible_paths,
                    )
                    if key in self.compatible_paths:
                        access = "Lazy" if descriptor.lazy_supported else "Load all"
                    elif descriptor.lazy_supported or descriptor.eager_supported:
                        access = "Metadata"
                    else:
                        access = "Unsupported"
                    item.setText(4, access)
                    item.setToolTip(4, self._selection_description(item))
                    self._dataset_items[key] = item
                add_children(item, key)

        add_children(self.tree, "")


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
