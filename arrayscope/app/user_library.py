"""One storage protocol behind every user-editable library.

ArrayScope ships two libraries the user can rearrange, extend and break:
colormaps (:mod:`arrayscope.display.colormap_library`) and operations
(:mod:`arrayscope.operations.library`).  They grew independently and converged
on the same three mechanisms by copy-paste -- a listener registry with
dead-Qt self-healing, a ``layout.json`` of group order / per-item group /
per-item order, and a ``hidden-*.json`` set -- which then drifted in the
details that matter (one had a format header, the other did not; one honored a
scoped ``-n <name>`` install, the other did not; one recorded what it could not
load, the other swallowed it).

This module is that mechanism, once, parameterized by the key names each side
already writes to disk.  ADR 0063.

Qt-free by discipline: the only Qt that can reach here is the lazy ``QtCore``
import inside :mod:`arrayscope.app.user_dirs`, so both libraries stay
importable in headless / worker contexts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from typing import NamedTuple

_LOGGER = logging.getLogger(__name__)

LAYOUT_VERSION = 1


# ---------------------------------------------------------------------------
# The problems channel
# ---------------------------------------------------------------------------


class LibraryProblem(NamedTuple):
    """One stored file the loader could not accept, and enough to repair it.

    **Keyed by ``path``, deliberately.**  The obvious key is the item's id (an
    operation id, a colormap name) and it is the wrong one: the id lives
    *inside* the payload, so every failure class that stops the payload parsing
    has no id at all, and the classes that reject the id itself have one that
    cannot be looked up.  The file path is the one identity a stored definition
    always has.  ADR 0062 established this for operations; ADR 0063 makes it the
    shape both libraries answer in.

    ``payload`` is the raw JSON document when it parsed as an object, so a
    repair surface can drive an ordinary form from the stored fields instead of
    demanding hand-edited JSON.  It is ``None`` only when the file is not
    parseable JSON or is not a JSON object.  ``item_id`` is whatever the payload
    claims, unvalidated, for display.
    """

    path: str
    message: str
    payload: dict | None
    item_id: str | None


def unique_name(base: str, existing, *, copy_suffix: bool = True) -> str:
    """``"<base> copy"``, then ``"<base> copy 2"``, … -- the one copy convention.

    The operation library already spelled a duplicate this way
    (``_unique_copy_label``) and the colormap designer spelled it
    ``"<base>-2"``; a user who learns one should not have to learn the other.
    ``copy_suffix=False`` is the *new*-item case: take the plain name when it is
    free.
    """

    taken = {str(name).casefold() for name in existing}
    root = str(base).strip()
    candidate = f"{root} copy" if copy_suffix else root
    if candidate.casefold() not in taken:
        return candidate
    suffix = 2
    while f"{candidate} {suffix}".casefold() in taken:
        suffix += 1
    return f"{candidate} {suffix}"


def problem_record(path: str, message: str, *, id_field: str = "id") -> LibraryProblem:
    """Build a :class:`LibraryProblem`, salvaging whatever of the payload survives."""

    payload: dict | None = None
    item_id: str | None = None
    try:
        with open(path) as handle:
            raw = json.load(handle)
    except Exception:
        raw = None
    if isinstance(raw, dict):
        payload = dict(raw)
        claimed = raw.get(id_field)
        if isinstance(claimed, str) and claimed:
            item_id = claimed
    return LibraryProblem(path, message, payload, item_id)


# ---------------------------------------------------------------------------
# Listeners
# ---------------------------------------------------------------------------


class ListenerRegistry:
    """Process-global change callbacks for one library, with dead-Qt self-heal."""

    def __init__(self, description: str):
        self._description = str(description)
        self._callbacks: list = []

    def add(self, callback) -> None:
        if callable(callback) and callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove(self, callback) -> None:
        """Unregister a callback (idempotent).

        Widget-bound listeners MUST call this on teardown: the registry is
        process-global, so a listener that outlives its widget turns the next
        library mutation into a call on a dead Qt object.
        """

        with contextlib.suppress(ValueError):
            self._callbacks.remove(callback)

    def notify(self) -> None:
        for callback in tuple(self._callbacks):
            try:
                callback()
            except RuntimeError:
                # Listener bound to a Qt widget whose C++ object is gone (a
                # window closed without unregistering).  Drop it so the registry
                # self-heals instead of retrying the dead wrapper every mutation.
                self.remove(callback)
            except Exception:
                # No repr of the callback in the message: repr of a dead Qt
                # wrapper raises RuntimeError itself.
                _LOGGER.exception("%s listener failed", self._description)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def write_json_atomically(path: str, payload) -> None:
    """Serialize ``payload``, replacing ``path`` only once it is fully written.

    Write-in-place turns an interrupted save into a *new* broken file, which is
    the failure class both libraries exist to shrink.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def migrate_directory(old: str, new: str) -> bool:
    """Copy ``old`` into ``new`` once, when ``new`` holds nothing yet.

    Used when a library's storage location moves (the colormap library moved
    from the unscoped ``AppDataLocation`` to the scoped config directory that
    operations already honored).  Deliberately a copy rather than a move: an
    older ArrayScope on the same machine keeps working, and a migration that
    goes wrong has not destroyed the only copy.  Returns whether anything moved.
    """

    old, new = str(old), str(new)
    if os.path.abspath(old) == os.path.abspath(new) or not os.path.isdir(old):
        return False
    if os.path.isdir(new) and any(os.scandir(new)):
        return False
    names = [name for name in sorted(os.listdir(old)) if name.endswith(".json")]
    if not names:
        return False
    os.makedirs(new, exist_ok=True)
    moved = False
    for name in names:
        source = os.path.join(old, name)
        if not os.path.isfile(source):
            continue
        try:
            shutil.copyfile(source, os.path.join(new, name))
            moved = True
        except OSError:
            _LOGGER.warning("could not migrate %s to %s", source, new)
    return moved


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class LayoutStore:
    """``layout.json``: group order, per-item group, per-item order.

    Parameterized by the key names each library already writes (``map_groups``
    / ``map_order`` for colormaps, ``op_groups`` / ``op_order`` for operations)
    so neither side's on-disk file has to be rewritten to share the code.

    ``optional_lists`` are extra list-valued keys that are persisted *only when
    set* (operations' ``common_ids`` and ``more_groups``), so an unset override
    keeps tracking the code default rather than freezing today's.

    A file with no ``format``/``version`` header still loads: the colormap
    library wrote header-less layouts for two years.  The next save adds the
    header, which makes the upgrade one-way and invisible.
    """

    def __init__(
        self,
        directory,
        *,
        format_name: str,
        group_key: str,
        order_key: str,
        optional_lists: tuple[str, ...] = (),
        listeners: ListenerRegistry | None = None,
        file_name: str = "layout.json",
    ):
        self._directory = directory
        self._format = str(format_name)
        self.group_key = str(group_key)
        self.order_key = str(order_key)
        self.optional_lists = tuple(optional_lists)
        self._listeners = listeners
        self._file_name = str(file_name)
        self._cache: dict | None = None

    def path(self) -> str:
        return os.path.join(self._directory(), self._file_name)

    def empty(self) -> dict:
        layout: dict = {"group_order": [], self.group_key: {}, self.order_key: {}}
        for key in self.optional_lists:
            layout[key] = None
        return layout

    def load(self) -> dict:
        if self._cache is None:
            try:
                with open(self.path()) as handle:
                    payload = json.load(handle)
                layout = {
                    "group_order": [str(group) for group in payload.get("group_order", [])],
                    self.group_key: {
                        str(key): str(value)
                        for key, value in payload.get(self.group_key, {}).items()
                    },
                    self.order_key: {
                        str(key): int(value)
                        for key, value in payload.get(self.order_key, {}).items()
                    },
                }
                for key in self.optional_lists:
                    layout[key] = [str(value) for value in payload[key]] if key in payload else None
                self._cache = layout
            except Exception:
                self._cache = self.empty()
        return self._cache

    def save(self, layout: dict) -> None:
        payload: dict = {
            "format": self._format,
            "version": LAYOUT_VERSION,
            "group_order": [str(group) for group in layout.get("group_order", [])],
            self.group_key: {
                str(key): str(value) for key, value in layout.get(self.group_key, {}).items()
            },
            self.order_key: {
                str(key): int(value) for key, value in layout.get(self.order_key, {}).items()
            },
        }
        for key in self.optional_lists:
            if layout.get(key) is not None:
                payload[key] = [str(value) for value in layout[key]]
        write_json_atomically(self.path(), payload)
        self._cache = None
        self._notify()

    def group_of(self, item_id: str, declared: str) -> str:
        """The group ``item_id`` lands in: the layout override, else ``declared``."""

        return self.load()[self.group_key].get(str(item_id), declared)

    def position_of(self, item_id: str, default: int) -> int:
        return self.load()[self.order_key].get(str(item_id), default)

    def ordered_groups(self, present, defaults) -> list[str]:
        """Group order: persisted first, then the code default, then the rest."""

        ordered = [group for group in self.load()["group_order"] if group in present]
        ordered += [group for group in defaults if group in present and group not in ordered]
        ordered += [group for group in sorted(present) if group not in ordered]
        return ordered

    def rename_group(self, old: str, new: str, members) -> bool:
        """Reassign every member of ``old`` to ``new`` and rename the order entry.

        ``members`` is the iterable of item ids currently *in* ``old`` (each
        library resolves that itself, because a group membership is the layout
        override *or* the declared group).  Returns whether anything changed.
        """

        old, new = str(old), str(new).strip()
        if not new or old == new:
            return False
        layout = self.load()
        groups = dict(layout[self.group_key])
        for item_id in members:
            groups[str(item_id)] = new
        self.save(
            {
                "group_order": [new if group == old else group for group in layout["group_order"]],
                self.group_key: groups,
                self.order_key: layout[self.order_key],
                **{key: layout[key] for key in self.optional_lists},
            }
        )
        return True

    def invalidate(self) -> None:
        self._cache = None

    def remove(self) -> None:
        """Discard the persisted layout (revert to code defaults)."""

        path = self.path()
        if os.path.exists(path):
            os.remove(path)
        self._cache = None
        self._notify()

    def _notify(self) -> None:
        if self._listeners is not None:
            self._listeners.notify()


# ---------------------------------------------------------------------------
# Hidden sets
# ---------------------------------------------------------------------------


class HiddenSet:
    """The persisted "the user removed this system definition" set.

    Hiding rather than deleting is what makes a built-in's removal reversible;
    both libraries store it as a sorted JSON array of ids beside the library.
    """

    def __init__(self, directory, *, file_name: str, listeners: ListenerRegistry | None = None):
        self._directory = directory
        self._file_name = str(file_name)
        self._listeners = listeners

    def path(self) -> str:
        return os.path.join(self._directory(), self._file_name)

    def values(self) -> frozenset[str]:
        try:
            with open(self.path()) as handle:
                return frozenset(str(value) for value in json.load(handle))
        except Exception:
            return frozenset()

    def set(self, item_id: str, hidden: bool) -> None:
        ids = set(self.values())
        if bool(hidden):
            ids.add(str(item_id))
        else:
            ids.discard(str(item_id))
        write_json_atomically(self.path(), sorted(ids))
        if self._listeners is not None:
            self._listeners.notify()
