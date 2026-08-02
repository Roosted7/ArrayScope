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


def child_environment(request):
    """The child's value for each named environment variable."""

    return {name: os.environ.get(name) for name in request}


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
