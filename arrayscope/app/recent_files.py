"""Bounded recent-data history, using the application's existing settings store."""

from __future__ import annotations

import os
from pathlib import Path

RECENT_FILES_KEY = "recent_data_files"
RECENT_FILES_LIMIT = 10


def recent_paths(settings) -> tuple[Path, ...]:
    """Read normalized paths; retain missing entries so the UI can explain them."""
    stored = settings.value(RECENT_FILES_KEY, [])
    if isinstance(stored, str):
        stored = [stored]
    if not isinstance(stored, (tuple, list)):
        return ()
    paths = []
    seen = set()
    for entry in stored:
        if not isinstance(entry, str) or not entry or "\0" in entry:
            continue
        path = Path(entry).expanduser().absolute()
        key = os.path.normcase(str(path))
        if key not in seen:
            paths.append(path)
            seen.add(key)
        if len(paths) == RECENT_FILES_LIMIT:
            break
    return tuple(paths)


def remember_path(settings, path) -> None:
    """Move a successfully opened path to the front without growing the store."""
    path = Path(path).expanduser().resolve()
    previous = recent_paths(settings)
    key = os.path.normcase(str(path))
    paths = [path, *(p for p in previous if os.path.normcase(str(p)) != key)]
    settings.setValue(RECENT_FILES_KEY, [str(p) for p in paths[:RECENT_FILES_LIMIT]])
