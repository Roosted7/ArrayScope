"""Module-level entry points the ``RemoteCall`` transport tests resolve.

Deliberately Qt-free, pyqtgraph-free and wgpu-free, and free of fixture state:
a spawned worker imports this module and the standard library, which is the
whole claim ``child_gui_modules`` exists to check.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import numpy as np

from arrayscope.operations.cancellation import EvaluationCancelled


def doubled(request):
    """One array in, one array out — the smallest transport round trip."""

    return np.asarray(request) * 2


def concatenated(request):
    """A tuple of arrays in, one array out — the aggregate families' shape."""

    return np.concatenate([np.asarray(item) for item in request])


def own_pid(request):
    """Which process actually ran this. The only proof of isolation."""

    del request
    return os.getpid()


def child_gui_modules(request):
    """Whatever GUI/GPU modules the child imported. Must be empty."""

    del request
    return sorted(
        name for name in sys.modules if name.startswith(("PySide", "PyQt", "wgpu", "pyqtgraph"))
    )


def stage_attachments(request):
    """How many stage segments this process has mapped. Zero means it recomputed."""

    del request
    from arrayscope.operations import stage_segments

    return int(stage_segments.attachments)


def stage_segment_probe(request):
    """Attach a published stage segment and report what the child sees.

    The pid proves which process read it, the checksum proves it read the
    parent's bytes, and ``writeable`` proves the parent's copy is safe from it.
    """

    from arrayscope.operations.stage_segments import attach_stage_array

    array = attach_stage_array(request)
    return {
        "pid": os.getpid(),
        "shape": tuple(int(size) for size in array.shape),
        "dtype": str(array.dtype),
        "checksum": complex(np.asarray(array).sum()),
        "writeable": bool(array.flags.writeable),
    }


def child_environment(request):
    """The child's value for each named environment variable."""

    return {name: os.environ.get(name) for name in request}


def native_thread_team_sizes(request):
    """What the loaded native libraries actually sized their thread teams to.

    ``child_environment`` reads the *intent* — that ``os.environ`` carries the
    pin.  This reads the *effect*.  OpenBLAS sizes its team in a loader
    constructor that reads the environment exactly once, so any import landing
    ahead of ``_worker_main``'s pin leaves ``os.environ`` looking correct while
    the team is already ``os.cpu_count()`` wide.  That is precisely the state a
    ``SlotPoolHandle`` process argument used to produce.
    """

    del request
    return {"openblas": _loaded_openblas_threads(), "env": os.environ.get("OPENBLAS_NUM_THREADS")}


def _loaded_openblas_threads():
    """Read ``*_get_num_threads`` from the BLAS this process already loaded.

    Returns ``None`` when the library or symbol cannot be found, which is a
    skip for the caller rather than a failure: the pin is what is under test,
    not which BLAS a given wheel happens to bundle.
    """

    import ctypes
    import re

    try:
        with open("/proc/self/maps") as handle:
            candidates = [
                match.group(1)
                for match in (
                    re.search(r"(\S*(?:openblas|blas)\S*\.so\S*)", line) for line in handle
                )
                if match
            ]
    except OSError:
        return None
    for path in dict.fromkeys(candidates):
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        for symbol in (
            "openblas_get_num_threads",
            "openblas_get_num_threads64_",
            "scipy_openblas_get_num_threads64_",
            "scipy_openblas_get_num_threads_64_",
        ):
            reader = getattr(library, symbol, None)
            if reader is not None:
                reader.restype = ctypes.c_int
                return int(reader())
    return None


def cancellation_flag(request, token):
    """Read the token once. ``pass_token`` is what routes it here."""

    del request
    return bool(token.cancelled)


def block_until_cancelled(request, token):
    """Poll the shared cancellation byte until it trips, or give up."""

    limit = time.monotonic() + float(request)
    while not token.cancelled and time.monotonic() < limit:
        time.sleep(0.005)
    if token.cancelled:
        raise EvaluationCancelled("cancelled in the child")
    return "never cancelled"


def called(request):
    """Invoke the request itself — an unpicklable payload's in-process fallback."""

    return request()


def raise_value_error(request):
    """Fail the way ordinary code fails, so the parent must rebuild the type."""

    raise ValueError(str(request))


def kill_own_process(request):
    """Die mid-task without a reply, the way a segfault does."""

    del request
    os.kill(os.getpid(), signal.SIGKILL)
