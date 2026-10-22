"""Pure helpers for saving displayed arrays as NumPy files."""

from __future__ import annotations

import errno
import operator
import os
import stat
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

try:
    # NumPy 2 moved the implementation while retaining the public format API;
    # its automatic header writer is needed for Unicode (NPY v3) field names.
    from numpy.lib._format_impl import _write_array_header
except ImportError:  # NumPy 1.x, supported by this package's lower bound.
    from numpy.lib.format import _write_array_header

from arrayscope.operations.cancellation import EvaluationCancelled

_CHUNK_WRITE_BYTES = 1024 * 1024


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


def _publish_exports(
    exports, *, cancellation_token=None, progress_callback=None, before_publish=None
):
    """One staging and publication owner for materialized and streamed output."""
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


def _check_export_dtype(dtype):
    if dtype.hasobject or not getattr(type(dtype), "_legacy", True):
        raise ValueError("Export requires a NumPy dtype that can be saved without pickle")


class _ChunkExportStream:
    """Bound header and payload writes even for unusually large dtype records."""

    def __init__(self, stream, token):
        self._stream = stream
        self._token = token

    def write(self, value):
        view = memoryview(value).cast("B")
        total = 0
        while total < len(view):
            _check_cancelled(self._token)
            count = self._stream.write(view[total : total + _CHUNK_WRITE_BYTES])
            if count is None or count <= 0:
                raise OSError("Export stream did not accept the next bytes")
            total += count
            _check_cancelled(self._token)
        return total


def _write_chunk_c_order(stream, chunk):
    if not chunk.size or not chunk.dtype.itemsize:
        return
    # NumPy's buffered external loop traverses C order without copying an
    # entire F-contiguous or strided chunk. A contiguous operand permits a
    # byte view, including dtypes unsupported by Python's buffer format.
    with np.nditer(
        chunk,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly", "contig"],
        buffersize=max(1, _CHUNK_WRITE_BYTES // chunk.dtype.itemsize),
        order="C",
    ) as iterator:
        for block in iterator:
            stream.write(block.view(np.uint8))


def save_derived_array_chunks(
    path,
    *,
    shape,
    dtype,
    chunks,
    cancellation_token=None,
    progress_callback=None,
    before_publish=None,
):
    """Atomically write one C-order NPY from leading-axis ndarray chunks.

    The declared dtype, rank and trailing dimensions are exact; no implicit
    casts or broadcasts occur. Scalar output requires one scalar chunk. An
    empty output may omit chunks; if supplied, they must cover its full row
    count. Each write is at most 1 MiB, with cancellation checked around it.
    Conversion buffers target 1 MiB, or one dtype element if it is larger.
    Progress counts cumulative serialized bytes, including the single header.

    All validation and ``before_publish`` run before atomic replacement. Once
    replacement starts, success wins any late cancellation. No sidecars or
    archive output are created.
    """
    path = Path(path)
    if path.suffix.lower() != ".npy":
        raise ValueError("Streamed array export requires an explicit .npy destination")
    try:
        dimensions = tuple(shape)
        if any(isinstance(size, (bool, np.bool_)) for size in dimensions):
            raise ValueError
        shape = tuple(operator.index(size) for size in dimensions)
        if any(size < 0 for size in shape):
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError("Export shape must contain nonnegative integer dimensions") from error
    dtype = np.dtype(dtype)
    _check_export_dtype(dtype)
    if dtype.subdtype is not None:
        raise ValueError("Export dtype must describe array elements, not a subarray shape")

    def write(stream):
        stream = _ChunkExportStream(stream, cancellation_token)
        header = {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        }
        # NumPy's own automatic selection preserves v2 large headers and v3
        # Unicode field names, rather than pinning a lossy descriptor/version.
        _write_array_header(stream, header)
        rows = 0
        seen = False
        for chunk in chunks:
            _check_cancelled(cancellation_token)
            if not isinstance(chunk, np.ndarray):
                raise TypeError("Export chunks must be NumPy ndarrays")
            if chunk.dtype != dtype:
                raise ValueError(f"Export chunk dtype {chunk.dtype} does not match {dtype}")
            if chunk.ndim != len(shape) or chunk.shape[1:] != shape[1:]:
                raise ValueError(f"Export chunk shape {chunk.shape} is incompatible with {shape}")
            count = chunk.shape[0] if shape else 1
            if rows + count > (shape[0] if shape else 1):
                raise ValueError("Export chunks contain more rows than the declared shape")
            rows += count
            seen = True
            _write_chunk_c_order(stream, np.asarray(chunk))
            # Release the previous output before the producer allocates its
            # next block; the iterator itself owns any longer source lease.
            del chunk
        _check_cancelled(cancellation_token)
        if rows != (shape[0] if shape else 1) and not (0 in shape and not seen):
            raise ValueError("Export chunks contain fewer rows than the declared shape")

    return _publish_exports(
        [(path, write)],
        cancellation_token=cancellation_token,
        progress_callback=progress_callback,
        before_publish=before_publish,
    )


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
    _check_export_dtype(data.dtype)
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

    return _publish_exports(
        exports,
        cancellation_token=cancellation_token,
        progress_callback=progress_callback,
        before_publish=before_publish,
    )
