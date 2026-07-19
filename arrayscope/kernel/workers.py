"""Worker backends: where task functions physically run.

The kernel owns *what* runs next; a backend owns *threads*. Swapping the
backend must never change scheduling semantics — this is the seam for a
free-threaded (3.14t) or process-based executor later.
"""

from __future__ import annotations

import os
import threading
from time import monotonic
from typing import Protocol


def default_worker_count() -> int:
    """Compute workers sized to cores, leaving headroom for GUI + GPU driver."""

    return max(2, (os.cpu_count() or 4) - 2)


class WorkerBackend(Protocol):
    """Contract between the kernel and an execution substrate."""

    workers: int

    def attach(self, kernel) -> None:
        """Bind to a kernel. Called exactly once, before any submit."""

    def wake(self) -> None:
        """Signal that ready work may be available. Cheap, thread-safe."""

    def shutdown(self, timeout: float = 5.0) -> tuple[str, ...]:
        """Stop pulling work and report threads alive after one total deadline."""


class ThreadWorkerBackend:
    """N daemon threads pulling from the kernel under its condition variable.

    NumPy/FFT/compression release the GIL, so this backend already provides
    real multi-core execution for array work. Pure-Python-heavy tasks are the
    kernel's quota problem, not this backend's.
    """

    def __init__(self, workers: int | None = None, *, name: str = "arrayscope-kernel") -> None:
        self.workers = default_worker_count() if workers is None else max(1, int(workers))
        self._name = str(name)
        self._kernel = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def attach(self, kernel) -> None:
        if self._kernel is not None:
            raise RuntimeError("backend is already attached")
        self._kernel = kernel
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
            kernel._execute(record)


class InlineWorkerBackend:
    """Deterministic synchronous execution for tests and tools.

    ``wake()`` drains every ready task on the calling thread before
    returning. Reentrant wakes (a task submitting more tasks) are flattened
    into the outer drain loop, so submission order plus priority fully
    determines execution order.
    """

    workers = 1

    def __init__(self) -> None:
        self._kernel = None
        self._draining = False

    def attach(self, kernel) -> None:
        if self._kernel is not None:
            raise RuntimeError("backend is already attached")
        self._kernel = kernel

    def wake(self) -> None:
        if self._draining:
            return
        self._draining = True
        try:
            while True:
                record = self._kernel._take_next(block=False)
                if record is None:
                    return
                self._kernel._execute(record)
        finally:
            self._draining = False

    def shutdown(self, timeout: float = 5.0) -> tuple[str, ...]:
        self._kernel = None
        return ()


__all__ = [
    "InlineWorkerBackend",
    "ThreadWorkerBackend",
    "WorkerBackend",
    "default_worker_count",
]
