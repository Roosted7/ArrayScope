"""Pure helpers for saving displayed arrays as NumPy files."""

from __future__ import annotations

import errno
import os
import stat
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np


def default_numpy_filename(source_filepath=None):
    if source_filepath is None:
        return "arrayscope.npy"

    source_path = Path(source_filepath)
    source_name = source_path.name
    if source_name.lower().endswith(".nii.gz"):
        return source_name[:-7] + ".npy"
    return f"{source_path.stem}.npy"


def selected_numpy_data(data, ranges, *, squeeze=True):
    sliced_data = data[tuple(slice(start, stop) for start, stop in ranges)]
    return np.squeeze(sliced_data) if squeeze else sliced_data


def _stage_export(stack, path, write):
    """Serialize beside the resolved destination without modifying it.

    Following symlinks preserves the semantics of a normal file write. New
    files use the caller's umask; replacing an existing file preserves its
    mode and first checks that opening it for writing would be permitted.
    """
    target = path.resolve()
    mode = None
    try:
        current = target.stat()
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISDIR(current.st_mode):
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), str(path))
        if not stat.S_ISREG(current.st_mode):
            raise ValueError(f"Export destination must be a regular file: {path}")
        # Atomic replacement requires directory write permission but does not
        # itself honor a target's read-only mode. Do not bypass that refusal.
        descriptor = os.open(target, os.O_WRONLY)
        os.close(descriptor)
        mode = stat.S_IMODE(current.st_mode)

    directory = stack.enter_context(
        tempfile.TemporaryDirectory(prefix=".arrayscope-save-", dir=target.parent)
    )
    staged = Path(directory) / "payload"
    with staged.open("wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    if mode is not None:
        staged.chmod(mode)
    return staged, target


def save_derived_array(path, data, *, recipe_json=None, view_recipe_json=None, sidecar=True):
    """Save to exact paths, publishing each file only after serialization.

    All output is staged before any destination is replaced, so serialization
    errors preserve previous exports and a mapped input can safely overwrite
    its own path. Each replacement is atomic; several sidecars are not one
    filesystem transaction. A single ``.npz`` keeps data and recipes together.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        payload = {"array": data}
        if recipe_json is not None:
            payload["recipe_json"] = np.array(recipe_json)
        if view_recipe_json is not None:
            payload["view_recipe_json"] = np.array(view_recipe_json)
        exports = [(path, lambda stream: np.savez(stream, **payload))]
    else:
        if suffix != ".npy":
            path = path.with_suffix(".npy")
        exports = [(path, lambda stream: np.save(stream, data))]
        if sidecar:
            for suffix, recipe in ((".recipe.json", recipe_json), (".view.json", view_recipe_json)):
                if recipe is not None:
                    encoded = (recipe.rstrip() + "\n").encode("utf-8")
                    exports.append(
                        (path.with_suffix(suffix), lambda stream, text=encoded: stream.write(text))
                    )

    written = tuple(path for path, _write in exports)
    if len({path.resolve() for path in written}) != len(written):
        raise ValueError("Array and recipe export destinations must refer to different files")
    with ExitStack() as stack:
        staged = [_stage_export(stack, path, write) for path, write in exports]
        for temporary, target in staged:
            os.replace(temporary, target)
    return written
