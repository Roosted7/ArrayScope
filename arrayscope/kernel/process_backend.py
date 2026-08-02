"""A worker backend that runs ``RemoteCall`` bodies in separate processes.

The kernel is untouched: this backend pulls tasks on its own threads exactly
like ``ThreadWorkerBackend``, and a thread that pulls a ``RemoteCall`` hands it
to a persistent spawn-context pool and blocks on the reply. Blocking is the
point — the kernel's worker slot stays occupied for as long as the work runs,
so lane quotas and backlog stay truthful. Everything that is not a
``RemoteCall`` (closures, continuation no-ops) runs in-thread as today.

This module must stay import-light and numpy-free: a spawned child imports it
before ``_worker_main`` gets to pin the native thread teams, and the whole
point of the child is that it never touches Qt, pyqtgraph, or wgpu.
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
import pickle
import queue
import threading
import traceback
import warnings
from contextlib import suppress
from importlib import import_module
from time import monotonic

from arrayscope.core.trace import emit_trace
from arrayscope.kernel.remote_call import RemoteCall, remote_dispatch, resolve_entry
from arrayscope.kernel.workers import default_worker_count
from arrayscope.operations.cancellation import EvaluationCancelled

# N worker processes with uncapped native thread teams is an oversubscription
# bomb (numba kernels here are `parallel=True`). One team each; raising a cap
# is a measured decision, not a default.
CHILD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}
CHILD_NICENESS = 5
_REPLY_POLL_SECONDS = 0.02


def default_process_count() -> int:
    """Compute processes sized to cores, leaving the GUI and driver a core."""

    return max(1, (os.cpu_count() or 4) - 2)


class RemoteWorkerError(RuntimeError):
    """A worker process failed in a way with no parent-side counterpart."""


# --------------------------------------------------------------- child side


class _ChildCancellationToken:
    """The child's view of the parent's cooperative flag: one shared byte."""

    __slots__ = ("_control", "_index")

    def __init__(self, control, index: int) -> None:
        self._control = control
        self._index = int(index)

    @property
    def cancelled(self) -> bool:
        return bool(self._control[self._index])


def _worker_main(conn, index, control, slot_handle):
    os.environ.update(CHILD_ENVIRONMENT)
    with suppress(AttributeError, OSError):
        os.nice(CHILD_NICENESS)
    from arrayscope.kernel.shm_slots import SlotPool

    slots = SlotPool.attach(slot_handle)
    token = _ChildCancellationToken(control, index)
    owner = int(index) + 1
    try:
        while True:
            try:
                message = conn.recv()
            except (EOFError, OSError):
                return
            if message is None:
                return
            try:
                conn.send(_run_child_job(message, token, slots, owner))
            except (BrokenPipeError, OSError):
                return
    finally:
        slots.close()
        conn.close()


def _run_child_job(message, token, slots, owner):
    from arrayscope.kernel.shm_slots import unwrap_payload, wrap_payload

    entry, payload, pass_token = message
    try:
        request, _ = unwrap_payload(payload, slots)
        function = resolve_entry(entry)
        value = function(request, token) if pass_token else function(request)
    except EvaluationCancelled as exc:
        return ("cancelled", str(exc))
    except BaseException as exc:
        return (
            "error",
            f"{type(exc).__module__}:{type(exc).__qualname__}",
            str(exc),
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
    # The leases stay held: the slots are READY until the parent takes and
    # releases them, which is the whole one-writer guarantee.
    result, _ = wrap_payload(value, slots, owner=owner)
    return ("ok", result)


# -------------------------------------------------------------- parent side


class _PoolWorker:
    __slots__ = ("conn", "index", "process")

    def __init__(self, index: int, process, conn) -> None:
        self.index = int(index)
        self.process = process
        self.conn = conn


class ProcessPool:
    """Persistent spawn-context workers plus the slot pool they write into."""

    def __init__(self, *, processes: int | None = None, slot_pool=None) -> None:
        from arrayscope.kernel.shm_slots import SlotPool

        self._context = multiprocessing.get_context("spawn")
        self.processes = default_process_count() if processes is None else max(1, int(processes))
        self._slots = SlotPool(context=self._context) if slot_pool is None else slot_pool
        self._owns_slots = slot_pool is None
        self._control = self._context.Array("b", self.processes, lock=False)
        self._idle: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._references = 0
        self._closed = False
        self._noted_fallbacks = 0
        self._workers = []
        try:
            for index in range(self.processes):
                self._workers.append(self._spawn(index))
        except BaseException:
            # A half-built pool owns a segment and possibly live children, and
            # nobody else has a reference to close it. `spawn` refuses outright
            # when the caller's `__main__` has no import guard, which is the
            # commonest way to arrive here.
            self.close()
            raise
        for worker in self._workers:
            self._idle.put(worker)

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    def acquire(self) -> ProcessPool:
        with self._lock:
            if self._closed:
                raise RemoteWorkerError("compute pool is shut down")
            self._references += 1
        return self

    def release(self) -> None:
        with self._lock:
            self._references = max(0, self._references - 1)
            if self._references > 0:
                return
        self.close()

    def run(self, call: RemoteCall, token=None, *, pass_token: bool = False):
        """Execute one ``RemoteCall`` on a worker process and return its value."""

        from arrayscope.kernel.shm_slots import wrap_payload

        worker = self._acquire()
        leases = ()
        try:
            payload, leases = wrap_payload(call.request, self._slots, owner=worker.index + 1)
            fallbacks = self._slots.fallbacks
            if fallbacks != self._noted_fallbacks:
                # The pool ran dry and a payload pickled instead. Edge-only, so
                # a saturated pool does not turn the trace into its own load.
                self._noted_fallbacks = fallbacks
                emit_trace("compute_slot_fallback", entry=call.entry, fallbacks=int(fallbacks))
            self._control[worker.index] = 0
            try:
                worker.conn.send((call.entry, payload, bool(pass_token)))
            except (pickle.PicklingError, TypeError, AttributeError) as exc:
                # Serialization fails before any byte reaches the pipe, so the
                # worker is healthy — retiring it would kill a process for a
                # payload problem. Degrade to in-process, counted, like every
                # other transport fallback.
                self._slots.count_fallback()
                emit_trace("compute_pickle_fallback", entry=call.entry, error=repr(exc))
                self._idle.put(worker)
                worker = None
                return call.invoke(token if pass_token else None)
            reply = self._await_reply(worker, token)
        except BaseException:
            # The worker's state is unknown after a transport failure; a child
            # error arrives as an ordinary reply and leaves it healthy.
            if worker is not None:
                self._retire(worker)
            raise
        finally:
            for lease in leases:
                lease.release()
        self._idle.put(worker)
        return self._value(reply)

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._references = 0
            workers = tuple(self._workers)
            self._workers = []
        for worker in workers:
            with suppress(BrokenPipeError, OSError, ValueError):
                worker.conn.send(None)
        deadline = monotonic() + max(0.0, float(timeout))
        for worker in workers:
            worker.process.join(timeout=max(0.0, deadline - monotonic()))
        for worker in workers:
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=1.0)
            with suppress(OSError):
                worker.conn.close()
        if self._owns_slots:
            self._slots.close()

    def _spawn(self, index: int) -> _PoolWorker:
        parent_conn, child_conn = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main,
            args=(child_conn, int(index), self._control, self._slots.handle()),
            name=f"arrayscope-compute-{index}",
            daemon=True,
        )
        process.start()
        child_conn.close()
        return _PoolWorker(index, process, parent_conn)

    def _acquire(self) -> _PoolWorker:
        while True:
            if self._closed:
                raise RemoteWorkerError("compute pool is shut down")
            try:
                return self._idle.get(timeout=0.1)
            except queue.Empty:
                continue

    def _await_reply(self, worker: _PoolWorker, token):
        while True:
            if token is not None and token.cancelled:
                self._control[worker.index] = 1
            if worker.conn.poll(_REPLY_POLL_SECONDS):
                # A dead worker's pipe also reads ready — at EOF, which is the
                # crash arriving as data.
                try:
                    return worker.conn.recv()
                except (EOFError, OSError) as exc:
                    raise self._died(worker) from exc
            if not worker.process.is_alive():
                # A worker can exit between writing its reply and this check,
                # so the reply wins over the corpse.
                if worker.conn.poll(0):
                    try:
                        return worker.conn.recv()
                    except (EOFError, OSError):
                        pass
                raise self._died(worker)
            if self._closed:
                raise RemoteWorkerError("compute pool is shut down")

    def _died(self, worker: _PoolWorker) -> RemoteWorkerError:
        worker.process.join(timeout=1.0)
        return RemoteWorkerError(
            f"compute worker {worker.index} died (exit code {worker.process.exitcode})"
        )

    def _retire(self, worker: _PoolWorker) -> None:
        with suppress(OSError):
            worker.conn.close()
        if worker.process.is_alive():
            worker.process.terminate()
        worker.process.join(timeout=1.0)
        self._slots.reclaim(worker.index + 1)
        self._control[worker.index] = 0
        with self._lock:
            if self._closed:
                return
            replacement = self._spawn(worker.index)
            self._workers[worker.index] = replacement
        self._idle.put(replacement)

    def _value(self, reply):
        from arrayscope.kernel.shm_slots import unwrap_payload

        kind = reply[0]
        if kind == "ok":
            value, taken = unwrap_payload(reply[1], self._slots, copy=True)
            for index in taken:
                self._slots.release(index)
            return value
        if kind == "cancelled":
            raise EvaluationCancelled(reply[1])
        error = _rebuild_exception(reply[1], reply[2])
        # The scheduler overwrites `arrayscope_traceback` with the parent-side
        # frames when it catches this, so the child's text also gets its own
        # attribute — that copy is the only place the real failure survives.
        error.arrayscope_child_traceback = reply[3]
        error.arrayscope_traceback = reply[3]
        raise error


def _rebuild_exception(type_name: str, message: str) -> BaseException:
    module_name, _, qualname = str(type_name).partition(":")
    try:
        candidate = getattr(import_module(module_name), qualname, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            return candidate(message)
    except Exception:
        pass
    return RemoteWorkerError(f"{type_name}: {message}")


_shared_pool: ProcessPool | None = None
_shared_pool_lock = threading.Lock()


def shared_process_pool(*, processes: int | None = None) -> ProcessPool:
    """The one compute pool per GUI process; every window borrows this one."""

    global _shared_pool
    with _shared_pool_lock:
        if _shared_pool is None or _shared_pool.closed:
            _shared_pool = ProcessPool(processes=processes)
        return _shared_pool.acquire()


def _close_shared_pool() -> None:
    global _shared_pool
    with _shared_pool_lock:
        pool, _shared_pool = _shared_pool, None
    if pool is not None:
        pool.close()


atexit.register(_close_shared_pool)


class ProcessWorkerBackend:
    """Kernel worker threads that route ``RemoteCall`` bodies to the pool.

    Same three-method contract and the same thread behaviour as
    ``ThreadWorkerBackend``; the only difference is where a ``RemoteCall``
    physically runs. With no pool (or after shutdown) a ``RemoteCall`` still
    executes in-thread, so the backend never has a state in which work simply
    stops happening.
    """

    def __init__(
        self,
        workers: int | None = None,
        *,
        processes: int | None = None,
        pool: ProcessPool | None = None,
        name: str = "arrayscope-kernel",
    ) -> None:
        self.workers = default_worker_count() if workers is None else max(1, int(workers))
        self._name = str(name)
        self._processes = processes
        self._pool = pool
        self._borrowed_pool = pool is None
        self._kernel = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def attach(self, kernel) -> None:
        if self._kernel is not None:
            raise RuntimeError("backend is already attached")
        self._kernel = kernel
        if self._pool is None:
            try:
                self._pool = shared_process_pool(processes=self._processes)
            except Exception as exc:
                # No pool is a degraded mode, not a dead viewer: every
                # `RemoteCall` still runs in-thread. Loud, because a session
                # that silently lost its isolation would be measured as if it
                # had it.
                warnings.warn(
                    f"compute process pool unavailable, running in-thread: {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        for index in range(self.workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"{self._name}-{index}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def wake(self) -> None:
        kernel = self._kernel
        if kernel is not None:
            kernel._notify_workers()

    def shutdown(self, timeout: float = 5.0) -> tuple[str, ...]:
        self._stop.set()
        pool, self._pool = self._pool, None
        if pool is not None and self._borrowed_pool:
            # Releasing first is what unblocks a thread parked on a reply.
            pool.release()
        kernel = self._kernel
        if kernel is not None:
            kernel._notify_workers(all_workers=True)
        deadline = monotonic() + max(0.0, float(timeout))
        for thread in self._threads:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
        return tuple(thread.name for thread in self._threads if thread.is_alive())

    def _worker_loop(self) -> None:
        kernel = self._kernel
        while not self._stop.is_set():
            record = kernel._take_next(stop=self._stop)
            if record is None:
                continue
            # Installed for every record, not only ``spec.fn is RemoteCall``:
            # a task family migrates by building a ``RemoteCall`` wherever its
            # body is pure, and that is routinely *inside* a closure the rest
            # of the family still needs (ADR 0064 family 2). The caller's own
            # token argument tells the child's arity; the record token backs
            # cancellation when the caller passed none.
            token = record.token
            with remote_dispatch(
                lambda call, call_token, token=token: self._dispatch(call, call_token, token)
            ):
                kernel._execute(record)

    def _dispatch(self, call: RemoteCall, call_token, record_token):
        pool = self._pool
        if pool is None or pool.closed:
            return call.invoke(call_token)
        return pool.run(
            call,
            call_token if call_token is not None else record_token,
            pass_token=call_token is not None,
        )


__all__ = [
    "CHILD_ENVIRONMENT",
    "CHILD_NICENESS",
    "ProcessPool",
    "ProcessWorkerBackend",
    "RemoteWorkerError",
    "default_process_count",
    "shared_process_pool",
]
