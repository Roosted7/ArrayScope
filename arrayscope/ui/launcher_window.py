"""Empty-state launcher window: open files via dialog or drag-and-drop.

Shown when ArrayScope starts without file arguments (e.g. from a desktop
shell launcher). On macOS this window also keeps the app alive to receive
Finder open-document events.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.ui.file_dialogs import get_existing_directory


class LauncherWindow(QtWidgets.QWidget):
    """Open data, resume recent work, or explore a built-in example."""

    def __init__(
        self,
        open_callback,
        *,
        supported_suffixes,
        name_filter,
        recent_paths=(),
        samples=(),
        sample_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self._open_callback = open_callback
        self._supported_suffixes = tuple(s.lower() for s in supported_suffixes)
        self._name_filter = name_filter
        self._sample_callback = sample_callback
        self.setWindowTitle("ArrayScope")
        self.setAcceptDrops(True)
        self.setMinimumSize(520, 480)
        self.resize(640, 600)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("ArrayScope")
        font = title.font()
        font.setPointSize(font.pointSize() + 10)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel("Explore images, volumes, time series and complex data.")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        open_btn = QtWidgets.QPushButton("&Open Files…")
        open_btn.setObjectName("openFilesButton")
        open_btn.setAccessibleName("Open array files")
        open_btn.setToolTip("Choose one or more array files to view.")
        open_btn.setMinimumHeight(36)
        open_btn.setDefault(True)
        open_btn.clicked.connect(self._open_dialog)
        folder_btn = QtWidgets.QPushButton("Open &DICOM Folder…")
        folder_btn.setObjectName("openFolderButton")
        folder_btn.setAccessibleName("Open DICOM folder")
        folder_btn.setToolTip("Choose a folder containing a DICOM series.")
        folder_btn.setMinimumHeight(36)
        folder_btn.clicked.connect(self._open_folder)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(open_btn, 1)
        button_row.addWidget(folder_btn, 1)
        layout.addLayout(button_row)

        self._open_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Open, self)
        self._open_shortcut.activated.connect(self._open_dialog)
        open_btn.setFocus()

        shortcut = QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Open).toString(
            QtGui.QKeySequence.SequenceFormat.NativeText
        )
        hint = self._muted_label(f"Drop files or a DICOM folder anywhere.  {shortcut} opens files.")
        layout.addWidget(hint)

        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(10)

        recent_paths = tuple(Path(path) for path in islice(recent_paths, 8))
        self.recent_list = QtWidgets.QListWidget()
        self.recent_list.setObjectName("recentFilesList")
        self.recent_list.setAccessibleName("Recent files and folders")
        self.recent_list.setAccessibleDescription(
            "Use the arrow keys to select a path and Enter to open it."
        )
        self.recent_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.recent_list.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        self.recent_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.recent_list.setFixedHeight(110)
        self.recent_list.setSpacing(3)
        self.recent_list.itemActivated.connect(self._open_recent)
        if recent_paths:
            body_layout.addWidget(self._heading("Continue working"))
            for path in recent_paths:
                available = path.exists()
                title = path.name or str(path)
                suffix = "" if available else " — Not found"
                item = QtWidgets.QListWidgetItem(f"{title}{suffix}\n{path.parent}")
                item.setData(QtCore.Qt.ItemDataRole.UserRole, str(path))
                item.setData(
                    QtCore.Qt.ItemDataRole.AccessibleTextRole, f"{title}{suffix}, {path.parent}"
                )
                item.setToolTip(
                    str(path) if available else f"{path}\nThis path is no longer available."
                )
                if not available:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
                    item.setForeground(self.palette().color(QtGui.QPalette.ColorRole.Mid))
                self.recent_list.addItem(item)
            body_layout.addWidget(self.recent_list)
        else:
            self.recent_list.hide()

        samples = tuple(samples)
        if samples:
            body_layout.addWidget(self._heading("Explore without a file"))
            body_layout.addWidget(
                self._muted_label("Built-in examples. Real tools. No download needed.")
            )
            examples = QtWidgets.QHBoxLayout()
            examples.setSpacing(10)
            for sample in samples:
                examples.addWidget(self._sample_card(sample), 1)
            body_layout.addLayout(examples)

        if not recent_paths and not samples:
            body_layout.addWidget(self._heading("Start with your data"))
            body_layout.addWidget(
                self._muted_label(
                    "Open an array to browse dimensions, inspect values and try reversible operations."
                )
            )
        body_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("launcherBody")
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setObjectName("launcherStatus")
        self.status_label.setAccessibleName("Open status")
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        formats = self._muted_label(
            "Formats: " + ", ".join(s.lstrip(".") for s in sorted(self._supported_suffixes))
        )
        formats.setToolTip("Supported file extensions and DICOM folders")
        formats.setWordWrap(True)
        layout.addWidget(formats)

    @staticmethod
    def _heading(text):
        label = QtWidgets.QLabel(text)
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        font = label.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        label.setFont(font)
        return label

    @staticmethod
    def _muted_label(text):
        label = QtWidgets.QLabel(text)
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setStyleSheet("color: palette(mid);")
        return label

    def _sample_card(self, sample):
        card = QtWidgets.QFrame()
        card.setObjectName("LauncherExampleCard")
        card.setStyleSheet(
            "QFrame#LauncherExampleCard { border: 1px solid palette(light); border-radius: 8px; }"
        )
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._heading(sample.title))
        description = QtWidgets.QLabel(sample.description)
        description.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        description.setWordWrap(True)
        layout.addWidget(description)
        layout.addWidget(self._muted_label(sample.hint))
        layout.addStretch(1)
        button = QtWidgets.QPushButton("Explore")
        button.setObjectName(f"sample_{sample.id}")
        button.setAccessibleName(f"Explore {sample.title}")
        button.setAccessibleDescription(f"{sample.description} {sample.hint}")
        button.setEnabled(self._sample_callback is not None)
        button.clicked.connect(
            lambda _checked=False, sample_id=sample.id: self._open_sample(sample_id)
        )
        layout.addWidget(button)
        return card

    def _open_sample(self, sample_id):
        if self._sample_callback is not None and self._sample_callback(sample_id):
            self.close()
        else:
            self.status_label.setText("Could not open this example. Please try again.")
            self.status_label.show()

    def _open_recent(self, item):
        if item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled:
            self._open_paths((Path(item.data(QtCore.Qt.ItemDataRole.UserRole)),))

    def _open_dialog(self):
        dialog = QtWidgets.QFileDialog(self, "Open array files", "", self._name_filter)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._open_paths(Path(name) for name in dialog.selectedFiles())

    def _open_folder(self):
        name = get_existing_directory(self, "Open DICOM folder")
        if name:
            self._open_paths((Path(name),))

    def _open_paths(self, paths):
        opened_any = False
        for path in paths:
            opened_any = bool(self._open_callback(path)) or opened_any
        if opened_any:
            self.close()
        else:
            self.status_label.setText(
                "Could not open the selected data. Check that it is available and supported."
            )
            self.status_label.show()

    def _droppable_paths(self, mime):
        if not mime.hasUrls():
            return []
        paths = [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]
        return [
            p
            for p in paths
            if p.is_dir() or any(str(p).lower().endswith(s) for s in self._supported_suffixes)
        ]

    def dragEnterEvent(self, event):
        if self._droppable_paths(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = self._droppable_paths(event.mimeData())
        if not paths:
            return
        event.acceptProposedAction()
        self._open_paths(paths)
