"""Pure helpers for saving displayed arrays as NumPy files."""

from __future__ import annotations

import errno
import os
import stat
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from arrayscope.operations.cancellation import EvaluationCancelled


def _check_cancelled(token):
    if token is not None and getattr(token, "cancelled", False):
        raise EvaluationCancelled()


class _ExportStream:
    """File-like writes keep NumPy on its bounded, cancellable stream path."""

    def __init__(self, stream, token, record_bytes):
        self._stream = stream
        self._token = token
        self._record_bytes = record_bytes

    def write(self, value):
        _check_cancelled(self._token)
        written = self._stream.write(value)
        self._record_bytes(written)
        _check_cancelled(self._token)
        return written

    def tell(self):
        return self._stream.tell()

    def read(self, *args):
        # NumPy uses this capability marker to distinguish file-like archive
        # output from a path. Its ZIP writer does not read a write-only file.
        return self._stream.read(*args)

    def seek(self, *args):
        return self._stream.seek(*args)

    def flush(self):
        return self._stream.flush()


def default_numpy_filename(source_filepath=None):
    if source_filepath is None:
        return "arrayscope.npy"

    source_path = Path(source_filepath)
    source_name = source_path.name
    if source_name.lower().endswith(".nii.gz"):
        return source_name[:-7] + ".npy"
    return f"{source_path.stem}.npy"


def _stage_export(stack, path, write, *, cancellation_token=None, record_bytes=None):
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
        _check_cancelled(cancellation_token)
        write(
            stream
            if cancellation_token is None and record_bytes is None
            else _ExportStream(stream, cancellation_token, record_bytes or (lambda _count: None))
        )
        stream.flush()
        os.fsync(stream.fileno())
    if mode is not None:
        staged.chmod(mode)
    return staged, target


def save_derived_array(
    path,
    data,
    *,
    recipe_json=None,
    view_recipe_json=None,
    selection_json=None,
    sidecar=True,
    cancellation_token=None,
    progress_callback=None,
    before_publish=None,
):
    """Save to exact paths, publishing each file only after serialization.

    All output is staged before any destination is replaced, so serialization
    errors preserve previous exports and a mapped input can safely overwrite
    its own path. Each replacement is atomic; several sidecars are not one
    filesystem transaction. A single ``.npz`` keeps data and recipes together.

    Cancellation is honored during stream writes and immediately before the
    publication begins. Once the first replacement starts, all staged files
    are published and success is returned even if cancellation arrives late.
    ``progress_callback`` receives cumulative serialized bytes; archive header
    rewrites count as work, so this is not necessarily the final file size.
    """
    data = np.asanyarray(data)
    if data.dtype.hasobject or not getattr(type(data.dtype), "_legacy", True):
        # NumPy's archive writer pickles object and non-legacy custom dtypes.
        # Refuse those explicitly, including on NumPy versions whose savez
        # did not yet expose an allow_pickle argument.
        raise ValueError("Export requires a NumPy dtype that can be saved without pickle")
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        payload = {"array": data}
        if recipe_json is not None:
            payload["recipe_json"] = np.array(recipe_json)
        if view_recipe_json is not None:
            payload["view_recipe_json"] = np.array(view_recipe_json)
        if selection_json is not None:
            payload["selection_json"] = np.array(selection_json)
        if any(value.dtype.hasobject for value in payload.values()):
            raise ValueError("Export metadata must not require pickle")
        exports = [(path, lambda stream: np.savez(stream, **payload))]
    else:
        if suffix != ".npy":
            path = path.with_suffix(".npy")
        exports = [(path, lambda stream: np.save(stream, data, allow_pickle=False))]
        if sidecar:
            for suffix, recipe in (
                (".recipe.json", recipe_json),
                (".view.json", view_recipe_json),
                (".selection.json", selection_json),
            ):
                if recipe is not None:
                    encoded = (recipe.rstrip() + "\n").encode("utf-8")
                    exports.append(
                        (path.with_suffix(suffix), lambda stream, text=encoded: stream.write(text))
                    )

    written = tuple(path for path, _write in exports)
    if len({path.resolve() for path in written}) != len(written):
        raise ValueError("Array and recipe export destinations must refer to different files")
    serialized = 0

    def record_bytes(count):
        nonlocal serialized
        serialized += int(count)
        if progress_callback is not None:
            progress_callback(serialized)

    _check_cancelled(cancellation_token)
    with ExitStack() as stack:
        staged = [
            _stage_export(
                stack,
                path,
                write,
                cancellation_token=cancellation_token,
                record_bytes=record_bytes if progress_callback is not None else None,
            )
            for path, write in exports
        ]
        _check_cancelled(cancellation_token)
        if before_publish is not None:
            before_publish()
        _check_cancelled(cancellation_token)
        for temporary, target in staged:
            os.replace(temporary, target)
    return written
