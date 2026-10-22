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
#
# `OMP_NUM_THREADS=1` IS A FLOOR, NOT A POLICY, AND IT IS A LANDMINE FOR ANY
# FUTURE FFT-BEARING FAMILY. Measured 2026-08-04 inside a real pool child:
# it CAPS `scipy.fft`'s own `workers=` argument, because scipy now dispatches
# to ducc, whose binary carries the string "by the `OMP_NUM_THREADS`
# environment variable". `fft(..., workers=8)` on (336,336,272) complex64 ran
# 145.6 ms at 1.00x parallelism pinned, against 47.6 ms at 6.84x unpinned --
# 3.06x slower. Bisected against all five names: OMP alone does it, the other
# four are innocent, and "all except OMP" is clean.
#
# Latent today, because no routed family runs an FFT: the four `RemoteCall`
# entries are histogram plot, histogram aggregate, preview page fan-out and
# image snapshot, none of which touch scipy.fft. It stops being latent the
# moment stage materialization or tile evaluation routes -- a 600 ms
# whole-volume stage would become ~1.8 s and adding an operation would get
# VISIBLY WORSE. A whole-volume transform that nothing else waits behind wants
# many threads, which is the opposite of what this dict says; the shape the
# product already uses in-process (`stage_workers=1, fft_workers_stage=8`) is
# the correct one and must survive any such move.
#
# So: before routing an FFT-bearing family, this must become a per-family
# thread budget rather than a constant. It is not fixed here because it is not
# adjudicable until such a family routes, and a blind change would trade a
# measured oversubscription bomb for an unmeasured one.
#
# TRAP for anyone A/B-ing this: patching `CHILD_ENVIRONMENT` in the PARENT is
# inert. `_worker_main` reads the module global in the child, which re-imports
# this module under `spawn`, so the parent's mutation is never seen -- an
# intended A/B silently runs A/A. Change the tracked source.
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


def _slot_handle_fields(slots) -> tuple:
    """Flatten a ``SlotPoolHandle`` into members a child can unpickle numpy-free.

    The shared state has to reach the child through `Process(args=...)`: a
    `multiprocessing.sharedctypes` object refuses to travel any other way
    ("should only be shared between processes through inheritance"), so it
    cannot be sent over the pipe after the child has pinned its environment.
    Sending the *fields* instead keeps the inheritance path while keeping
    `kernel.shm_slots` -- and therefore numpy, and therefore OpenBLAS -- out of
    the child's pre-pin import set. `_worker_main` rebuilds the handle.
    """

    handle = slots.handle()
    return (
        handle.name,
        handle.slot_bytes,
        handle.slots,
        handle.states,
        handle.owners,
        handle.fallbacks,
    )


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


def _worker_main(conn, index, control, slot_fields):
    # FIRST, before anything drags in a native library. `spawn` unpickles this
    # function and its arguments before `_bootstrap` calls it, so every module
    # those arguments name is imported while the environment is still the
    # parent's. OpenBLAS and libgomp size their thread teams in a loader
    # constructor, reading the environment exactly once, so an import that
    # lands ahead of this line makes the pin below inert -- `os.environ` says
    # 1 and `openblas_get_num_threads()` says `os.cpu_count()`.
    #
    # This is why `slot_fields` is a plain tuple rather than a `SlotPoolHandle`:
    # unpickling the dataclass would import `kernel.shm_slots`, which imports
    # numpy, which loads OpenBLAS. The tuple's members unpickle through
    # `multiprocessing.sharedctypes` alone, which is numpy-free.
    os.environ.update(CHILD_ENVIRONMENT)
    with suppress(AttributeError, OSError):
        os.nice(CHILD_NICENESS)
    from arrayscope.kernel.shm_slots import SlotPool, SlotPoolHandle

    slots = SlotPool.attach(SlotPoolHandle(*slot_fields))
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
        self._active: set[_PoolWorker] = set()
        self._slots_close_pending = False
        try:
            for index in range(self.processes):
                worker = self._spawn(index)
                self._workers.append(worker)
                self._idle.put(worker)
        except BaseException:
            # A half-built pool owns a segment and possibly live children, and
            # nobody else has a reference to close it. `spawn` refuses outright
            # when the caller's `__main__` has no import guard, which is the
            # commonest way to arrive here.
            self.close()
            raise

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    def acquire(self) -> ProcessPool:
        with self._lock:
            if self._closed:
                raise RemoteWorkerError("compute pool is shut down")
            self._references += 1
        return self

    def release(self, timeout: float = 5.0) -> None:
        deadline = monotonic() + max(0.0, float(timeout))
        with self._lock:
            self._references = max(0, self._references - 1)
            if self._references > 0:
                return
            workers = self._detach_workers_locked()
        if workers is not None:
            self._close_workers(workers, deadline)

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
            try:
                worker.conn.send((call.entry, payload, bool(pass_token)))
            except (pickle.PicklingError, TypeError, AttributeError) as exc:
                # Serialization fails before any byte reaches the pipe, so the
                # worker is healthy — retiring it would kill a process for a
                # payload problem. Degrade to in-process, counted, like every
                # other transport fallback.
                self._slots.count_fallback()
                emit_trace("compute_pickle_fallback", entry=call.entry, error=repr(exc))
                self._return_worker(worker)
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
        try:
            return self._value(reply)
        finally:
            self._return_worker(worker)

    def _return_worker(self, worker: _PoolWorker) -> None:
        with self._lock:
            self._active.discard(worker)
            if self._closed:
                # This thread has finished its pipe I/O. Closing an active
                # pipe from the shutdown thread could race a read/write on a
                # file descriptor the OS then reuses.
                with suppress(OSError):
                    worker.conn.close()
                self._close_slots_if_unused_locked()
            else:
                self._idle.put(worker)

    def _close_slots_if_unused_locked(self) -> None:
        if self._slots_close_pending and not self._active:
            self._slots_close_pending = False
            self._slots.close()

    def close(self, timeout: float = 5.0) -> None:
        """Allow one total graceful wait, then hard-stop and reap survivors.

        A zero or negative timeout skips graceful waiting. Hard-stop/reap
        still pays the OS scheduling latency required to leave no children.
        """

        deadline = monotonic() + max(0.0, float(timeout))
        with self._lock:
            workers = self._detach_workers_locked()
        if workers is not None:
            self._close_workers(workers, deadline)

    def _detach_workers_locked(self):
        # The last release claims closure under the same lock as acquire;
        # another borrower cannot acquire between reference count zero and
        # the closed flag becoming visible.
        if self._closed:
            return None
        self._closed = True
        self._references = 0
        workers = tuple(self._workers)
        self._workers = []
        return workers

    def _close_workers(self, workers, deadline: float) -> None:
        try:
            for worker in workers:
                self._control[worker.index] = 1
            # EOF wakes idle children. Active calls close their own pipes
            # after transport unwinds; neither a blocking sentinel write nor
            # a cross-thread close can interrupt an active pipe operation.
            with self._lock:
                while True:
                    try:
                        idle = self._idle.get_nowait()
                    except queue.Empty:
                        break
                    with suppress(OSError):
                        idle.conn.close()
            for worker in workers:
                worker.process.join(timeout=max(0.0, deadline - monotonic()))
            survivors = [worker for worker in workers if worker.process.is_alive()]
            for worker in survivors:
                # SIGTERM can be ignored by user code. Grace has expired, so
                # stop every survivor before reaping any of them; do not
                # grant each child another independent timeout.
                with suppress(ProcessLookupError):
                    worker.process.kill()
            for worker in survivors:
                worker.process.join()
        finally:
            with self._lock:
                # Reaping a child does not finish the parent's transport: a
                # thread can still be copying a request or decoding an
                # already-received result. Its borrow keeps the map alive.
                self._slots_close_pending = self._owns_slots
                self._close_slots_if_unused_locked()

    def _spawn(self, index: int) -> _PoolWorker:
        parent_conn, child_conn = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main,
            args=(child_conn, int(index), self._control, _slot_handle_fields(self._slots)),
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
                worker = self._idle.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                if self._closed:
                    with suppress(OSError):
                        worker.conn.close()
                    raise RemoteWorkerError("compute pool is shut down")
                self._control[worker.index] = 0
                self._active.add(worker)
                return worker

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
        with self._lock:
            if self._closed:
                # Pool shutdown owns all children and their storage now. A
                # reply waiter must not add its own retirement wait or spawn
                # a replacement while that teardown is in progress.
                with suppress(OSError):
                    worker.conn.close()
                self._active.discard(worker)
                self._close_slots_if_unused_locked()
                return
        with suppress(OSError):
            worker.conn.close()
        if worker.process.is_alive():
            worker.process.terminate()
        worker.process.join(timeout=1.0)
        if worker.process.is_alive():
            # A broken transport does not imply cooperative user code. Do
            # not reclaim slots or lose the handle while an old writer lives.
            with suppress(ProcessLookupError):
                worker.process.kill()
            worker.process.join()
        self._slots.reclaim(worker.index + 1)
        with self._lock:
            self._active.discard(worker)
            if self._closed:
                self._close_slots_if_unused_locked()
                return
            self._control[worker.index] = 0
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
        deadline = monotonic() + max(0.0, float(timeout))
        self._stop.set()
        pool, self._pool = self._pool, None
        if pool is not None and self._borrowed_pool:
            # Releasing first is what unblocks a thread parked on a reply.
            pool.release(timeout=max(0.0, deadline - monotonic()))
        kernel = self._kernel
        if kernel is not None:
            kernel._notify_workers(all_workers=True)
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
