"""Shared resolution of the per-user config / data directories.

ArrayScope stores small pieces of user state (saved view sessions, and now
user-defined operations) *next to the settings file* so a portable / scoped
install keeps everything together.  The resolution rule is subtle -- it prefers
the ``QSettings`` file's directory (which honors a ``-n <name>`` scoped run or a
custom ``QSettings`` path) and only falls back to the platform ``AppConfig``
location -- so it lives here once instead of being copied per feature.

Qt-only imports (``QtCore``): this module never touches widgets, so it stays
importable in headless / worker contexts, matching the discipline of
``arrayscope.display.colormap_library``.
"""

from __future__ import annotations

from pathlib import Path


def user_config_directory(application_name: str | None = None) -> Path:
    """Directory that holds this install's user config (settings file's parent).

    Mirrors the historic behavior of the file-view-session config dir: prefer
    the directory of the active ``QSettings`` file (so a scoped ``-n <name>``
    run or a custom settings path is honored), otherwise the platform
    ``AppConfig`` location, otherwise the home directory.  When the running
    application carries a non-default scoped name that the base does not already
    encode, the returned path is nested under that name so scoped runs never
    share the default install's state.

    ``application_name`` defaults to the live ``QCoreApplication.applicationName``
    so callers get the current scoping automatically; pass an explicit value to
    resolve a specific scope.
    """

    from pyqtgraph.Qt import QtCore

    def scoped_for_application(base: Path) -> Path:
        name = application_name
        if name is None:
            name = str(QtCore.QCoreApplication.applicationName() or "")
        if name and name != "ArrayScope" and base.name != name:
            return base / name
        return base

    try:
        settings_path = Path(QtCore.QSettings().fileName())
        if str(settings_path):
            return scoped_for_application(settings_path.parent)
    except Exception:
        pass
    location = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.AppConfigLocation
    )
    if not location:
        location = QtCore.QDir.homePath()
    return scoped_for_application(Path(location))


def user_operations_directory() -> Path:
    """Directory that holds user-defined operations (wrapper JSON + code files).

    Lives next to the user config so user operations travel with the rest of an
    install's state.  See :mod:`arrayscope.operations.library` for the on-disk
    schema and loader.
    """

    return user_config_directory() / "operations"
