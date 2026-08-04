"""The process backend end to end: transport, isolation, and containment.

Spawn is slow, so one module-scoped pool serves every case that only needs a
worker; the two cases that are *about* pool lifetime build their own. What
these pin is the part a unit test of the parent cannot: that the child really
is another process, that it never imports the GUI stack, that a cancellation
byte crosses, and that killing a worker costs exactly one task.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from arrayscope.kernel import Kernel, Lane, Priority, TaskOutcome, TaskSpec
from arrayscope.kernel.process_backend import (
    CHILD_ENVIRONMENT,
    ProcessPool,
    ProcessWorkerBackend,
    RemoteWorkerError,
)
from arrayscope.kernel.remote_call import RemoteCall

BLOCK_UNTIL_CANCELLED = "tests.kernel.remote_entries:block_until_cancelled"
CHILD_ENVIRONMENT_ENTRY = "tests.kernel.remote_entries:child_environment"
CHILD_GUI_MODULES = "tests.kernel.remote_entries:child_gui_modules"
CONCATENATED = "tests.kernel.remote_entries:concatenated"
DOUBLED = "tests.kernel.remote_entries:doubled"
KILL_OWN_PROCESS = "tests.kernel.remote_entries:kill_own_process"
NATIVE_THREAD_TEAMS = "tests.kernel.remote_entries:native_thread_team_sizes"
OWN_PID = "tests.kernel.remote_entries:own_pid"
RAISE_VALUE_ERROR = "tests.kernel.remote_entries:raise_value_error"


@pytest.fixture(scope="module")
def pool():
    pool = ProcessPool(processes=2).acquire()
    yield pool
    pool.release()


# Worker-process startup, not an interaction settlement budget: nothing
# user-visible waits on it, no gesture can be greened by widening it, and
# exceeding it fails the test rather than degrading anything.
PROCESS_START_LIMIT_S = 20.0


def await_completion(kernel):
    limit = time.monotonic() + PROCESS_START_LIMIT_S
    while time.monotonic() < limit:
        event = kernel.completions.pop()
        if event is not None:
            return event.spec.key, kernel.dispatch_event(event)
        time.sleep(0.01)
    raise AssertionError("no completion arrived")


def await_running(kernel):
    """Block until one task is on a worker, so cancellation has a target."""

    limit = time.monotonic() + PROCESS_START_LIMIT_S
    while time.monotonic() < limit:
        if kernel.diagnostics().running >= 1:
            return
        time.sleep(0.01)
    raise AssertionError("no task ever reached a worker")


def test_a_remote_call_runs_in_another_process(pool):
    assert pool.run(RemoteCall(OWN_PID, None)) != os.getpid()
    assert np.array_equal(pool.run(RemoteCall(DOUBLED, np.arange(5))), np.arange(5) * 2)


def test_large_payloads_travel_both_ways_through_slots(pool):
    before = pool._slots.fallbacks
    samples = (np.arange(200_000, dtype=np.float32), np.arange(4, dtype=np.float32))

    value = pool.run(RemoteCall(CONCATENATED, samples))

    assert value.shape == (200_004,)
    assert np.array_equal(value, np.concatenate(samples))
    assert pool._slots.fallbacks == before


def test_the_child_never_imports_the_gui_stack(pool):
    assert pool.run(RemoteCall(CHILD_GUI_MODULES, None)) == []


def test_the_child_pins_its_native_thread_teams(pool):
    names = sorted(CHILD_ENVIRONMENT)

    assert pool.run(RemoteCall(CHILD_ENVIRONMENT_ENTRY, names)) == dict.fromkeys(names, "1")


def test_the_childs_blas_really_sized_itself_to_the_pin(pool):
    # The environment assertion above passed for the whole life of the bug: it
    # reads the intent, and `os.environ` did carry "1". OpenBLAS sizes its
    # thread team in a loader constructor that reads the environment once, so
    # the only honest check is the team size the child ended up with.
    reading = pool.run(RemoteCall(NATIVE_THREAD_TEAMS, None))

    assert reading["env"] == "1"
    if reading["openblas"] is None:
        pytest.skip("no OpenBLAS with a readable get_num_threads in the child")
    assert reading["openblas"] == 1


def test_no_process_argument_reaches_the_child_through_an_arrayscope_module():
    # `spawn` unpickles the target and its arguments before `_bootstrap` calls
    # `_worker_main`, so every module an argument names is imported while the
    # environment is still the parent's. `kernel.shm_slots` imports numpy,
    # which loads OpenBLAS -- passing a `SlotPoolHandle` here is what made the
    # pin inert. Guard the arguments `_spawn` actually builds, not a helper's
    # return value: the defect lived in the call, so the call is the subject.
    recorded = {}

    class _RecordingProcess:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def start(self):
            pass

    probe = ProcessPool(processes=1)
    try:
        real_context = probe._context

        class _RecordingContext:
            def Pipe(self, duplex=True):
                return real_context.Pipe(duplex=duplex)

            def Process(self, **kwargs):
                return _RecordingProcess(**kwargs)

        probe._context = _RecordingContext()
        try:
            worker = probe._spawn(0)
        finally:
            probe._context = real_context
        worker.conn.close()
    finally:
        probe.close()

    offenders = {
        type(argument).__module__
        for argument in recorded["args"]
        if type(argument).__module__.startswith("arrayscope")
    }
    assert recorded["args"], "no process arguments were recorded"
    assert offenders == set()


def test_a_child_exception_arrives_typed_with_the_childs_traceback(pool):
    with pytest.raises(ValueError) as caught:
        pool.run(RemoteCall(RAISE_VALUE_ERROR, "child said no"))

    assert str(caught.value) == "child said no"
    assert "raise_value_error" in caught.value.arrayscope_child_traceback


def test_an_unpicklable_request_runs_in_process_and_keeps_the_worker(pool):
    """Serialization fails before any byte reaches the pipe, so the worker is
    healthy — a pickling problem must degrade to a counted in-process run, not
    retire a process (observed live via a duplicate-class ViewState, which is
    exactly the payload shape this stands in for)."""

    fallbacks = pool._slots.fallbacks
    value = pool.run(RemoteCall("tests.kernel.remote_entries:called", lambda: 41 + 1))

    assert value == 42
    assert pool._slots.fallbacks == fallbacks + 1
    assert pool.run(RemoteCall(OWN_PID, None)) != os.getpid()


def test_a_killed_worker_fails_one_task_and_the_pool_recovers(pool):
    with pytest.raises(RemoteWorkerError, match="died"):
        pool.run(RemoteCall(KILL_OWN_PROCESS, None))

    assert pool.run(RemoteCall(OWN_PID, None)) != os.getpid()


def test_the_pilot_families_survive_the_round_trip(pool):
    """ADR 0064's two stage-1 entries, run for real in a worker process.

    Spelled out rather than imported: both owning modules are Qt-free, but the
    submit sites that hold these constants are not, and this module must stay
    importable without a GUI stack. The constants themselves are pinned beside
    their submit sites.
    """

    from arrayscope.display.histogram_plot import HistogramPlotRequest, compute_histogram_plot
    from arrayscope.display.model.montage_levels import aggregate_histogram_samples

    data = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    request = HistogramPlotRequest(
        data=data,
        source_identity=(id(data), data.shape, str(data.dtype)),
        histogram_bounds=(0.0, 1.0),
        visible_value_span=1.0,
        pixel_extent=200.0,
    )
    samples = (np.linspace(0.0, 1.0, 90_000, dtype=np.float32),)

    plot = pool.run(RemoteCall("arrayscope.display.histogram_plot:compute_histogram_plot", request))
    aggregate = pool.run(
        RemoteCall(
            "arrayscope.display.model.montage_levels:aggregate_histogram_samples",
            samples,
        )
    )

    assert np.array_equal(plot.y, compute_histogram_plot(request).y)
    assert plot.source_identity == request.source_identity
    assert np.array_equal(aggregate, aggregate_histogram_samples(samples))


def test_the_kernel_splits_remote_calls_from_closures(pool):
    kernel = Kernel(ProcessWorkerBackend(workers=2, pool=pool))
    seen = {}
    kernel.submit(
        TaskSpec(key="remote", fn=RemoteCall(OWN_PID, None), lane=Lane.HISTOGRAM_REFINEMENT),
        on_done=lambda value: seen.__setitem__("remote", value),
    )
    assert await_completion(kernel)[1] == TaskOutcome.COMPLETED
    kernel.submit(
        TaskSpec(key="closure", fn=os.getpid, lane=Lane.HISTOGRAM_REFINEMENT),
        on_done=lambda value: seen.__setitem__("closure", value),
    )
    assert await_completion(kernel)[1] == TaskOutcome.COMPLETED

    kernel.shutdown()

    assert seen["remote"] != os.getpid()
    assert seen["closure"] == os.getpid()


def test_a_remote_call_nested_in_a_closure_routes_to_a_child(pool):
    """A family migrates body-first: the pure core becomes a ``RemoteCall``
    while the closure around it still holds unpicklable state (ADR 0064
    family 2). The dispatcher therefore covers every record, not only tasks
    whose ``fn`` is itself a ``RemoteCall``."""

    kernel = Kernel(ProcessWorkerBackend(workers=2, pool=pool))
    seen = {}
    kernel.submit(
        TaskSpec(
            key="nested",
            fn=lambda: (os.getpid(), RemoteCall(OWN_PID, None)()),
            lane=Lane.HISTOGRAM_REFINEMENT,
        ),
        on_done=lambda value: seen.__setitem__("nested", value),
    )
    assert await_completion(kernel)[1] == TaskOutcome.COMPLETED
    kernel.shutdown()

    closure_pid, nested_pid = seen["nested"]
    assert closure_pid == os.getpid()
    assert nested_pid != os.getpid()


def test_the_child_observes_the_kernels_cancellation(pool):
    kernel = Kernel(ProcessWorkerBackend(workers=1, pool=pool))
    handle = kernel.submit(
        TaskSpec(
            key="blocking",
            fn=RemoteCall(BLOCK_UNTIL_CANCELLED, 20.0),
            lane=Lane.HISTOGRAM_REFINEMENT,
            priority=Priority.HISTOGRAM,
            pass_token=True,
        )
    )
    await_running(kernel)

    handle.cancel()

    assert await_completion(kernel)[1] == TaskOutcome.CANCELLED
    kernel.shutdown()


def test_shutdown_leaves_no_worker_and_no_segment():
    pool = ProcessPool(processes=1).acquire()
    segment = pool._slots.handle().name
    pids = {worker.process.pid for worker in pool._workers}
    assert pool.run(RemoteCall(OWN_PID, None)) in pids

    pool.release()

    assert not [child for child in multiprocessing.active_children() if child.pid in pids]
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=segment, create=False, track=False)


def test_shutdown_with_a_task_in_flight_releases_its_thread():
    backend = ProcessWorkerBackend(workers=1, processes=1)
    kernel = Kernel(backend)
    kernel.submit(
        TaskSpec(
            key="blocking",
            fn=RemoteCall(BLOCK_UNTIL_CANCELLED, 60.0),
            lane=Lane.HISTOGRAM_REFINEMENT,
            pass_token=True,
        )
    )
    await_running(kernel)

    kernel.shutdown(timeout=15.0)

    assert backend._pool is None
    assert threading.active_count() >= 1
