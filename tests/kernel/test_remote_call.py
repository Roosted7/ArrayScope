"""``RemoteCall`` as a task body: resolvable, callable, and loud when wrong.

The point of the descriptor is that it behaves identically wherever it runs,
so these tests never build a process — they pin the in-process half, which is
what every backend but ``ProcessWorkerBackend`` executes and what that backend
falls back to when the pool is gone.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.kernel import InlineWorkerBackend, Kernel, TaskSpec
from arrayscope.kernel.remote_call import RemoteCall, remote_dispatch, resolve_entry

DOUBLED = "tests.kernel.remote_entries:doubled"
CANCELLATION_FLAG = "tests.kernel.remote_entries:cancellation_flag"


def test_resolves_and_runs_in_process():
    call = RemoteCall(DOUBLED, np.arange(4))

    assert np.array_equal(call(), np.arange(4) * 2)


def test_runs_as_a_kernel_task_body():
    kernel = Kernel(InlineWorkerBackend())
    seen = []
    kernel.submit(
        TaskSpec(key="doubled", fn=RemoteCall(DOUBLED, np.arange(3))), on_done=seen.append
    )

    event = kernel.completions.pop()
    kernel.dispatch_event(event)

    assert len(seen) == 1
    assert np.array_equal(seen[0], np.arange(3) * 2)


def test_pass_token_reaches_the_function():
    call = RemoteCall(CANCELLATION_FLAG, None)

    class Token:
        cancelled = True

    assert call(Token()) is True
    # Without a token the same entry is called with the request alone, which
    # is the arity `pass_token=False` promises.
    with pytest.raises(TypeError):
        call()


@pytest.mark.parametrize(
    "entry",
    [
        "tests.kernel.remote_entries.doubled",
        "tests.kernel.remote_entries:",
        ":doubled",
        "tests.kernel.remote_entries:missing",
        "arrayscope.kernel.remote_call:RemoteCall",
    ],
)
def test_bad_entries_fail_at_construction(entry):
    with pytest.raises(ValueError):
        RemoteCall(entry, None)


def test_unimportable_module_fails_at_construction():
    with pytest.raises(ImportError):
        RemoteCall("arrayscope.kernel.no_such_module:doubled", None)


def test_resolution_is_cached_by_entry():
    from tests.kernel import remote_entries

    assert resolve_entry(DOUBLED) is remote_entries.doubled
    assert resolve_entry(DOUBLED) is remote_entries.doubled


def test_installed_dispatcher_takes_over_and_unwinds():
    call = RemoteCall(DOUBLED, np.arange(2))
    seen = []

    def dispatcher(remote, token):
        seen.append((remote.entry, token))
        return "dispatched"

    with remote_dispatch(dispatcher):
        assert call() == "dispatched"
    assert np.array_equal(call(), np.arange(2) * 2)
    assert seen == [(DOUBLED, None)]
