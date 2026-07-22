"""Shared runtime for ArrayScope's optional numba accelerators.

Every numba accelerator in the codebase (the pyqtgraph display kernels, the RSS
operation reduction, the LOD pyramid bin reduction, ...) wants the same four
things, and got them by copy-pasting the machinery three times.  This module
centralises them so there is one implementation to reason about:

1. **A cheap availability probe.**  ``NUMBA_AVAILABLE`` is resolved with
   ``importlib.util.find_spec`` -- it does *not* ``import numba`` (a bare import
   is ~0.4-0.6 s here), so importing an accelerator module, and therefore
   whatever imports it, never pays that cost.

2. **A deferred, off-the-hot-path compile.**  The heavy ``import numba`` and the
   JIT compilation happen inside a *builder* callback that only runs on a
   background prewarm thread (or an explicit blocking prewarm).  The visible
   path never blocks on either.

3. **A numpy fallback contract.**  Until a group's kernels are compiled,
   :meth:`KernelGroup.get` returns ``None`` and the caller uses its
   always-correct numpy reference.

4. **Selective prewarming.**  Groups register by name, so a caller can warm
   exactly the kernels a session will use -- ``prewarm_async("pyramid")`` when a
   file opens, ``prewarm_async("display")`` only on the CPU display backend --
   instead of compiling everything unconditionally.

A *group* owns one builder.  The builder imports numba, defines and compiles
(force-warms) its ``@njit`` kernels, and returns an opaque object (typically a
dict of compiled kernels) that the accelerator module then calls.  If numba is
absent or the build raises (e.g. a broken LLVM), the group is marked
unavailable and its ``get()`` returns ``None`` forever -- the app stays on numpy.
"""

from __future__ import annotations

import importlib
import importlib.util
import threading
from collections.abc import Callable
from typing import Any

# Cheap: does not import numba. The real import happens inside a builder, on the
# prewarm thread.
NUMBA_AVAILABLE: bool = importlib.util.find_spec("numba") is not None

# The optional accelerator modules this runtime knows how to warm. Importing a
# module runs its module-level ``register()`` (cheap -- numba is not imported
# until a builder runs), after which the group can be compiled in the
# background. Keeping the list here is the whole point: the "what to prewarm"
# decision lives in one place, and the app makes a single startup call
# (:func:`prewarm_all_async`) instead of scattering per-accelerator warm-ups
# across window creation, file opens, and so on.
_ACCELERATOR_MODULES: tuple[str, ...] = (
    "arrayscope.operations._numba_reductions",
    "arrayscope.display.shader_kernels",
    "arrayscope.display._numba_pyramid",
    "arrayscope.gpu.bc_numba",
)

# Builder: zero-arg callable that imports numba, compiles kernels, and returns
# an opaque kernels object (dict, tuple, single callable -- the group does not
# care). Runs exactly once per process, off the visible path.
Builder = Callable[[], Any]


class KernelGroup:
    """One named set of numba kernels sharing a single lazy compile + prewarm.

    Not constructed directly -- use :func:`register`, which is idempotent by
    name so repeated imports of an accelerator module return the same group.
    """

    def __init__(
        self, name: str, builder: Builder, should_prewarm: Callable[[], bool] | None = None
    ) -> None:
        self.name = name
        self._builder = builder
        self._should_prewarm = should_prewarm
        self._ready = threading.Event()
        self._compile_lock = threading.Lock()
        self._thread_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._available = NUMBA_AVAILABLE
        self.kernels: Any = None

    @property
    def available(self) -> bool:
        """False if numba is missing or the build has failed for this group."""

        return self._available

    def wanted(self) -> bool:
        """Whether a *bulk* prewarm should compile this group in this session.

        Encapsulated by the accelerator (e.g. the display kernels return False
        for a wgpu session, since they run only on the pyqtgraph backend), so
        the runtime never has to know backend/settings semantics.  Defaults to
        True and is fail-open: a predicate that raises is treated as wanted.
        On-demand callers (:meth:`get`) ignore this -- if a kernel is actually
        needed it is always warmed and used.
        """

        if self._should_prewarm is None:
            return True
        try:
            return bool(self._should_prewarm())
        except Exception:
            return True

    def ready(self) -> bool:
        """True once the kernels are compiled and usable."""

        return self._ready.is_set()

    def prewarm(self) -> None:
        """Import numba and compile this group's kernels. **Blocks.**

        Callers on the visible path must use :meth:`prewarm_async` (or rely on
        the lazy warm inside :meth:`get`) instead. Idempotent and safe to call
        repeatedly; a group whose build failed stays permanently unavailable.
        """

        if not self._available or self._ready.is_set():
            return
        with self._compile_lock:
            if self._ready.is_set():
                return
            try:
                self.kernels = self._builder()
            except Exception:
                # numba present but unusable (broken LLVM, incompatible
                # version, ...); pin this group off and stay on numpy.
                self._available = False
                return
            self._ready.set()

    def prewarm_async(self) -> None:
        """Kick :meth:`prewarm` on a background daemon thread (non-blocking).

        Idempotent: at most one warm thread per group, and a no-op once ready.
        """

        if not self._available or self._ready.is_set():
            return
        with self._thread_lock:
            if self._ready.is_set():
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self.prewarm,
                name=f"arrayscope-numba-{self.name}-warm",
                daemon=True,
            )
            self._thread.start()

    def get(self) -> Any:
        """Return the compiled kernels, or ``None`` for the numpy fallback.

        When not yet warm, a background prewarm is kicked off and ``None`` is
        returned so the caller stays on numpy for this call.
        """

        if self._ready.is_set():
            return self.kernels
        self.prewarm_async()
        return None


_REGISTRY: dict[str, KernelGroup] = {}
_REGISTRY_LOCK = threading.Lock()


def register(
    name: str, builder: Builder, *, should_prewarm: Callable[[], bool] | None = None
) -> KernelGroup:
    """Register (or fetch) the kernel group named ``name``.

    ``should_prewarm`` is an optional predicate the accelerator supplies to gate
    *bulk* prewarming to sessions that will use it (e.g. the display kernels
    only on the pyqtgraph backend); omit it for always-relevant kernels.

    Idempotent: the first call for a name wins the builder and predicate; later
    calls return the existing group (the arguments are ignored), so module
    re-import is safe and every caller shares one compiled instance.
    """

    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(name)
        if existing is not None:
            return existing
        group = KernelGroup(name, builder, should_prewarm)
        _REGISTRY[name] = group
        return group


def get_group(name: str) -> KernelGroup | None:
    return _REGISTRY.get(name)


def registered_names() -> tuple[str, ...]:
    with _REGISTRY_LOCK:
        return tuple(_REGISTRY)


def _select(names: tuple[str, ...]) -> list[KernelGroup]:
    with _REGISTRY_LOCK:
        if not names:
            return list(_REGISTRY.values())
        return [_REGISTRY[n] for n in names if n in _REGISTRY]


def prewarm(*names: str) -> None:
    """Blocking selective prewarm. Warms the named groups (all if none named)."""

    for group in _select(names):
        group.prewarm()


def prewarm_async(*names: str) -> None:
    """Background selective prewarm. Warms the named groups (all if none named).

    This is the app-layer entry point: e.g. ``prewarm_async("pyramid")`` when a
    file opens, or ``prewarm_async()`` to warm every registered accelerator.
    Never blocks; a no-op for groups that are unavailable or already warm.
    """

    for group in _select(names):
        group.prewarm_async()


def ready(name: str) -> bool:
    """True if the named group is registered and its kernels are compiled."""

    group = _REGISTRY.get(name)
    return group is not None and group.ready()


def prewarm_all_async() -> None:
    """Register and background-compile every known optional numba accelerator.

    The single startup entry point: it imports each module in
    :data:`_ACCELERATOR_MODULES` (running its ``register()``) and kicks off
    compilation on daemon threads -- but only for groups whose
    :meth:`KernelGroup.wanted` predicate passes, so e.g. a wgpu session does not
    compile the pyqtgraph-only display kernels.  Never blocks and never raises;
    a no-op when numba is unavailable.  Individual accelerators also self-warm
    lazily on first use, so correctness never depends on this call (nor on the
    predicate) -- it just moves the one-time JIT cost off the first interaction.
    """

    if not NUMBA_AVAILABLE:
        return
    for module in _ACCELERATOR_MODULES:
        try:
            importlib.import_module(module)
        except Exception:
            continue
    for group in _select(()):
        if group.wanted():
            group.prewarm_async()
