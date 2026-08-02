"""Process-portable task bodies: ``(entry, request)`` instead of a closure.

A closure cannot cross a process boundary; a ``"module:function"`` string plus
picklable data can. ``RemoteCall`` is the form a submit site uses when it wants
the option of running off-process, and it stays callable in-process, so
``InlineWorkerBackend`` and ``ThreadWorkerBackend`` execute it exactly as they
executed the lambda it replaced. Nothing here knows about processes: a backend
that owns one installs its dispatcher for the duration of a task, and a
``RemoteCall`` invoked on that thread routes through it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from inspect import isfunction
from typing import Any

_resolved: dict[str, Callable[..., Any]] = {}
_dispatch = threading.local()


def resolve_entry(entry: str) -> Callable[..., Any]:
    """Import ``"module:function"`` and return the module-level function.

    Resolution is cached per process: the parent pays it once at submit (which
    is what makes a typo fail at the call site instead of in a child), the
    child pays it once per entry for the life of the worker.
    """

    cached = _resolved.get(entry)
    if cached is not None:
        return cached
    module_name, separator, attribute = str(entry).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"remote entry must be 'module:function', got {entry!r}")
    module = import_module(module_name)
    function = getattr(module, attribute, None)
    if function is None:
        raise ValueError(f"remote entry {entry!r} names no attribute of {module_name}")
    if not isfunction(function) or getattr(function, "__qualname__", "") != attribute:
        raise ValueError(f"remote entry {entry!r} is not a module-level function")
    _resolved[entry] = function
    return function


@dataclass(frozen=True, eq=False)
class RemoteCall:
    """One task body a worker process can execute: entry point plus payload.

    Identity comparison is deliberate — ``request`` routinely holds ndarrays,
    which have neither a usable ``__eq__`` nor a hash.
    """

    entry: str
    request: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry", str(self.entry))
        resolve_entry(self.entry)

    def __call__(self, token=None):
        dispatcher = getattr(_dispatch, "dispatch", None)
        if dispatcher is None:
            return self.invoke(token)
        return dispatcher(self, token)

    def invoke(self, token=None):
        """Resolve and run in this process — the always-available fallback."""

        function = resolve_entry(self.entry)
        return function(self.request) if token is None else function(self.request, token)


@contextmanager
def remote_dispatch(dispatcher):
    """Route ``RemoteCall`` bodies run on this thread through ``dispatcher``.

    Thread-local and nesting-safe: a backend installs it around one task, so a
    ``RemoteCall`` reached through a wrapper is routed too, while every other
    thread keeps the in-process path.
    """

    previous = getattr(_dispatch, "dispatch", None)
    _dispatch.dispatch = dispatcher
    try:
        yield
    finally:
        _dispatch.dispatch = previous


__all__ = ["RemoteCall", "remote_dispatch", "resolve_entry"]
