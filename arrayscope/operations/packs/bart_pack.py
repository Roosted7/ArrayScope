"""Optional in-process BART operation pack (out-of-process ``bart`` subprocess).

This pack ships BART (the Berkeley Advanced Reconstruction Toolbox) operations as
an **optional** in-process pack that runs the ``bart`` binary as a **subprocess**
with a **cfl temp-file handoff**.  It is optional in exactly the sense the sigpy
pack is: it self-guards on whether ``bart`` is runnable and contributes *nothing*
when it is not, so ``import arrayscope`` never spawns ``bart`` and import-health
stays green.  Availability is decided with a cheap, lazy filesystem check
(:func:`bart_available`) -- an executable ``bart`` on ``PATH`` or in the
``BART_TOOLBOX_PATH`` toolbox, with that env var set.  We deliberately do *not*
run ``bart version`` at import/enumeration time; the binary is only ever
*executed* when an op is applied.

Ops shipped (all clean unary ``fn(ndarray) -> ndarray``):

- ``bart:fft``  -- centered FFT via ``bart fft <bitmask>`` (the primary
  demonstrator).  BART's ``fft`` is **centered but unnormalized**: empirically it
  equals ``fftshift(fft(ifftshift(x, ax), ax), ax)`` (verified against NumPy in
  the tests).  An axis is mapped to a dimension bitmask (``1 << axis``); the cfl
  handoff preserves axis order, so numpy axis ``a`` is BART dim ``a``.
- ``bart:ifft`` -- centered inverse FFT via ``bart fft -i <bitmask>``, likewise
  **unnormalized** (BART's forward/inverse are both unnormalized, so
  ``ifft(fft(x)) == N * x``; this matches BART, not NumPy's 1/N convention, and
  is documented as such).
- ``bart:cabs`` -- pointwise complex magnitude via ``bart cabs``, a non-FFT
  demonstrator that the same cfl+subprocess mechanism serves.
- ``bart:carg`` -- pointwise complex **phase** (argument, ``atan2(Im, Re)``) via
  ``bart carg``.  The natural companion to ``cabs``: magnitude and phase are the
  two halves of a complex sample, and neither has a built-in ArrayScope op.
- ``bart:scale`` -- multiply every sample by a real scalar ``factor`` via
  ``bart scale <factor>``.  Additive: no built-in scalar-multiply op.
- ``bart:spow`` -- raise every sample to a (complex-principal-branch) power
  ``exponent`` via ``bart spow <exponent>``.  Additive: no built-in power op.
- ``bart:normalize`` -- scale by the reciprocal L2 norm computed *along one axis*
  via ``bart normalize <bitmask>`` (shape-preserving; the norm broadcasts back
  over the axis).  Additive: no built-in normalize.
- ``bart:std`` / ``bart:var`` -- **reductions** along one axis: standard deviation
  / variance via ``bart std <bitmask>`` / ``bart var <bitmask>``.  Additive: the
  built-in reductions are ``mean`` / ``rss`` / ``sum`` / ``max`` / ``min`` -- there
  is no built-in second-moment reduction.  BART reduces the axis to a singleton;
  we reshape the axis *out* so the output ndim drops by one, matching the built-in
  reductions' convention (and sidestepping ``read_cfl``'s trailing-singleton
  strip, which is otherwise axis-position-dependent).

Every op here is **OPAQUE / Tier-1** (see the capability note below): even the
pointwise ones (``cabs`` / ``carg`` / ``scale`` / ``spow``) stay whole-array,
because a per-tile out-of-process subprocess round-trip is never the right plan
for an expensive backend -- here the *cost model*, not correctness, forbids
windowing.

Deferred (mirroring how the sigpy pack deferred ESPIRiT / NUFFT rather than
shipping a fragile multi-input hack):

- ``bart:pics`` -- parallel-imaging compressed sensing is **multi-input**: it
  needs a k-space array *and* a coil-sensitivity map array
  (``bart pics kspace sens out``).  That does not fit the unary
  ``fn(ndarray) -> ndarray`` + scalar-parameter plugin contract -- there is no
  honest way to bind the second ndarray through the recipe/dock parameter model.
  A self-contained ``ecalib``->``pics`` variant would have to hard-code a coil
  axis and calibration semantics and would *change dimensionality*, which the
  scalar-param shape adapter cannot predict.  It is deferred with this reason
  rather than forced through the unary pipeline.  Correctness over coverage.

**Everything is complex64.** cfl is a complex64 container, so every op takes and
returns complex64: real/integer inputs are promoted on write.  The ops therefore
declare ``output_dtype = complex64`` unconditionally -- an honest cost signal
(the admission model sees the larger output for real inputs, below).

**Capability / admission (honest by design).**  Every BART op here is
**OPAQUE / Tier-1** (``region_capable=False``).  ``bart:fft``/``bart:ifft`` are
global transforms along the axis and genuinely not windowable.  ``bart:cabs`` is
*mathematically* pointwise, but it stays OPAQUE anyway: a per-region execution
would mean one out-of-process cfl round-trip *per tile*, which is never the right
plan for an expensive subprocess op.  Here the *cost model*, not correctness,
dictates whole-array-only.  The plugin path already classifies every pack op as
an OPAQUE whole-array ``TRANSFORM`` (blocks and expands every axis, not
chunkable, not fusable, a legitimate cache-stage boundary) -- the heaviest class
the admission cost model has -- and the forced complex64 output raises the
estimated bytes for real inputs.  That OPAQUE/TRANSFORM classification *is* the
admission cost hint; see :func:`bart_admission_notes`.

**Cancellation is independent of the kernel cooperative-cancellation item.**
The subprocess runner (:func:`run_bart`) honors a cancellation token by
``SIGTERM``-ing the child (then ``SIGKILL`` after a short grace) so a mid-op
cancel kills ``bart`` in well under a second, with no orphaned process and the
temp dir always cleaned.  The runner also drains stdout/stderr concurrently
(reader threads) for the whole run so a chatty child that writes past the
~64 KB pipe buffer can never deadlock, and applies a generous, env-configurable
overall timeout (:func:`bart_timeout`, ``ARRAYSCOPE_BART_TIMEOUT_S``) that bounds
a *stuck* child without capping an honest multi-minute recon.  This machinery is
self-contained: it does not depend on
the separate kernel work that threads a token into the plugin ``fn`` call path.
Until that lands, the engine plugin path applies the op with no token (the sync
whole-array path), so the runner is exercised with ``cancellation_token=None``
there; the token-driven kill is proven directly against :func:`run_bart` (and is
ready to be forwarded the moment the engine threads a token to plugin ops).
"""

from __future__ import annotations

import os
import shutil
import signal
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.plugins import PluginOperationSpec
from arrayscope.operations.registry import OperationParameter, register_pack_operation

# Namespaced, collision-safe ids (must contain the plugin namespace separator).
FFT_ID = "bart:fft"
IFFT_ID = "bart:ifft"
CABS_ID = "bart:cabs"
CARG_ID = "bart:carg"
SCALE_ID = "bart:scale"
SPOW_ID = "bart:spow"
NORMALIZE_ID = "bart:normalize"
STD_ID = "bart:std"
VAR_ID = "bart:var"

# Environment variable BART uses to locate its toolbox.  Its presence is part of
# the (cheap) availability check: without a toolbox path a bare ``bart`` on PATH
# still would not run the reconstruction commands reliably.
BART_TOOLBOX_ENV = "BART_TOOLBOX_PATH"

# Overall wall-clock ceiling for a single ``bart`` invocation, overridable via the
# same env-var config surface the toolbox path uses.  BART reconstructions
# (``pics``/``ecalib`` on real k-space) legitimately run for *minutes*, so the
# default is deliberately generous -- this ceiling exists to bound a *stuck* child
# (a hang, a deadlock, a runaway), not to cap honest long recons.  A non-positive
# value disables the ceiling entirely (rely on cancellation alone); a malformed
# value falls back to the default.
BART_TIMEOUT_ENV = "ARRAYSCOPE_BART_TIMEOUT_S"
_DEFAULT_TIMEOUT_S = 600.0  # 10 minutes -- generous headroom over a real recon.

# Subprocess-run tuning.  ``_POLL_INTERVAL`` is how often the run loop checks the
# cancellation token; ``_TERM_GRACE`` is how long we wait after SIGTERM before
# escalating to SIGKILL.  Both are small so a cancel resolves far under 1 s.
_POLL_INTERVAL = 0.02
_TERM_GRACE = 0.25

# Sentinel distinguishing "caller did not pass a timeout" (-> use the configured
# ceiling) from an explicit ``timeout=None`` (-> caller wants no ceiling).
_USE_CONFIGURED_TIMEOUT: object = object()


# --- availability (cheap, lazy: never runs bart) -----------------------------


def bart_executable() -> str | None:
    """Absolute path to a runnable ``bart``, or ``None`` -- without executing it.

    Looks in the ``BART_TOOLBOX_PATH`` toolbox first (the canonical install
    layout ships the binary at ``$BART_TOOLBOX_PATH/bart``), then falls back to a
    ``bart`` on ``PATH``.  Pure filesystem checks -- no subprocess, so enumeration
    and import-health never pay to spawn ``bart``.
    """

    toolbox = os.environ.get(BART_TOOLBOX_ENV)
    if toolbox:
        candidate = os.path.join(toolbox, "bart")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("bart")
    return found if found else None


def bart_available() -> bool:
    """Whether ``bart`` is runnable, without running it.

    Requires both that the toolbox env var is set and that an executable ``bart``
    is locatable.  A cheap metadata/filesystem check only -- the same discipline
    the sigpy pack uses with ``find_spec``: a mere availability check (register,
    enumerate) never spawns the backend.
    """

    return bool(os.environ.get(BART_TOOLBOX_ENV)) and bart_executable() is not None


def bart_timeout(default: float = _DEFAULT_TIMEOUT_S) -> float | None:
    """Configured wall-clock ceiling (seconds) for one ``bart`` run, or ``None``.

    Reads ``ARRAYSCOPE_BART_TIMEOUT_S`` from the environment -- the same env-var
    config surface :func:`bart_executable` uses for the toolbox path.  A
    non-positive override disables the ceiling (returns ``None``); a malformed
    override falls back to ``default``.  Pure env read -- never runs ``bart``.
    """

    raw = os.environ.get(BART_TIMEOUT_ENV)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else None


def bart_admission_notes() -> tuple[str, ...]:
    """Human-readable cost hints attached to the cost note stream (advisory).

    The concrete admission signal is the OPAQUE/TRANSFORM classification the
    plugin path already assigns plus the forced complex64 output; these notes
    make the *out-of-process* expense legible to anyone reading the cost model
    output.
    """

    return (
        "BART op: out-of-process subprocess + cfl temp-file round-trip (expensive).",
        "OPAQUE whole-array: never run per-region (a per-tile subprocess is never the right plan).",
    )


# --- cfl I/O (self-contained; matches BART's on-disk format) -----------------
#
# We roll our own minimal cfl reader/writer rather than importing
# ``$BART_TOOLBOX_PATH/python/cfl.py``.  Justification: it keeps the pack
# self-contained (no sys.path mutation into the BART source tree, no import of a
# module that lives outside the arrayscope package), and cfl is trivially simple
# -- a ``.hdr`` text file listing the dimensions plus a ``.cfl`` blob of raw
# complex64 in column-major (Fortran) order.  The format below is byte-for-byte
# what BART writes/reads (verified end-to-end by the fft-correctness test), so
# interop is exact.


def write_cfl(stem: str, array: np.ndarray) -> None:
    """Write ``array`` as a BART cfl pair ``<stem>.hdr`` + ``<stem>.cfl``.

    The data blob is complex64 in column-major order (BART's convention).  Real
    or integer inputs are promoted to complex64, exactly as BART's own writer
    does -- cfl has no real dtype.
    """

    array = np.asarray(array)
    if array.dtype != np.complex64:
        array = array.astype(np.complex64)
    with open(stem + ".hdr", "w") as handle:
        handle.write("# Dimensions\n")
        handle.write(" ".join(str(int(size)) for size in array.shape))
        handle.write("\n")
    # Column-major blob: writing the C-contiguous transpose lays the axes out in
    # Fortran order, which read_cfl reverses with reshape(order="F").
    with open(stem + ".cfl", "wb") as handle:
        handle.write(np.ascontiguousarray(array.T))


def read_cfl(stem: str) -> np.ndarray:
    """Read a BART cfl pair ``<stem>.hdr`` + ``<stem>.cfl`` into a complex64 array."""

    with open(stem + ".hdr") as handle:
        next(handle)  # "# Dimensions"
        dims = [int(token) for token in next(handle).split()]
    # BART drops trailing singleton dims in its own reader; mirror that so shapes
    # round-trip predictably.
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    with open(stem + ".cfl", "rb") as handle:
        data = np.frombuffer(handle.read(), dtype=np.complex64)
    return np.reshape(data[: int(np.prod(dims))], dims, order="F")


# --- subprocess runner (the load-bearing seam: cfl handoff + cancellation) ---


def bart_env(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Child environment: inherit ``os.environ``, overlay any explicit overrides.

    The caller (or the test fixture) is responsible for ensuring
    ``BART_TOOLBOX_PATH`` and, on this machine, an MKL ``LD_LIBRARY_PATH`` are
    present; ``bart_available`` already requires the toolbox path in the parent
    env, so the common case needs no overrides.
    """

    env = dict(os.environ)
    if overrides:
        env.update({str(key): str(value) for key, value in overrides.items()})
    return env


def _terminate_child(proc, grace: float = _TERM_GRACE) -> None:
    """SIGTERM the child's process group, then SIGKILL after ``grace`` seconds.

    The child is started in its own session (``start_new_session=True``) so any
    grandchildren ``bart`` spawns die with it -- no orphaned reconstruction.
    """

    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    deadline = time.monotonic() + grace
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    proc.wait()


def run_bart(
    argv: Sequence[str],
    array: np.ndarray,
    *,
    cancellation_token: object | None = None,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
    poll_interval: float = _POLL_INTERVAL,
    grace: float = _TERM_GRACE,
    timeout: float | None = _USE_CONFIGURED_TIMEOUT,  # type: ignore[assignment]
    temp_dir: str | None = None,
) -> np.ndarray:
    """Run ``bart <argv> in out`` on ``array`` via a cfl temp-file handoff.

    Writes ``array`` to a temp ``in.cfl``/``.hdr``, spawns ``bart`` with an
    ``out`` target, polls the cancellation token while it runs, reads ``out.cfl``
    back, and *always* cleans up the temp directory -- on success, error, cancel,
    or timeout.

    Draining: ``bart`` can be chatty (progress lines on stderr, banners on
    stdout).  Both pipes are drained *concurrently* by reader threads for the
    whole run, so a child that writes past the OS pipe buffer (~64 KB) can never
    deadlock waiting for us to read -- the classic ``Popen(PIPE)`` + late-read
    hang.  The captured stderr is still available for the failure message.

    Cancellation: if ``cancellation_token.cancelled`` becomes truthy mid-run the
    child (and its process group) is SIGTERM'd, then SIGKILL'd after ``grace``
    seconds, and :class:`EvaluationCancelled` is raised -- the same signal the
    rest of the operations engine uses.  The kill returns in well under a second.

    Timeout: an overall wall-clock ceiling bounds a *stuck* child (hang/deadlock/
    runaway).  ``timeout`` defaults to :func:`bart_timeout` (env-configurable,
    generous by default because real recons run minutes); pass an explicit float
    to override or ``None`` to disable.  On expiry the child is killed exactly
    like a cancel and a :class:`RuntimeError` mentioning the timeout is raised.

    Temp root: each run gets its own ``arrayscope-bart-*`` :class:`TemporaryDirectory`
    that is always cleaned up.  ``temp_dir`` overrides *where* that scratch dir is
    created (default: the system temp dir); it mainly lets a test isolate a run's
    scratch dir under its own ``tmp_path`` so a leak check cannot race a concurrent
    run's dir in the shared system temp.
    """

    import subprocess
    import tempfile
    import threading

    if timeout is _USE_CONFIGURED_TIMEOUT:
        timeout = bart_timeout()

    binary = executable or bart_executable()
    if binary is None:
        raise RuntimeError("bart executable not found (is BART_TOOLBOX_PATH set?)")
    child_env = bart_env(env)

    def _cancelled() -> bool:
        return cancellation_token is not None and getattr(cancellation_token, "cancelled", False)

    with tempfile.TemporaryDirectory(prefix="arrayscope-bart-", dir=temp_dir) as tmp:
        in_stem = os.path.join(tmp, "in")
        out_stem = os.path.join(tmp, "out")
        write_cfl(in_stem, array)

        # Bail before spawning if we were already cancelled.
        if _cancelled():
            raise EvaluationCancelled()

        proc = subprocess.Popen(
            [binary, *[str(token) for token in argv], in_stem, out_stem],
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # own process group -> we can kill the whole tree
        )

        # Reader threads drain both pipes for the lifetime of the child so a
        # chatty ``bart`` never blocks on a full pipe buffer while we poll.
        # stdout is drained-and-discarded; only stderr is captured for the
        # failure message.
        stderr_chunks: list[bytes] = []
        readers: list[threading.Thread] = []

        def _drain(stream, sink: list[bytes] | None) -> None:
            try:
                for chunk in iter(lambda: stream.read(65536), b""):
                    if sink is not None:
                        sink.append(chunk)
            except (ValueError, OSError):
                # Stream was closed underneath us during teardown -- benign.
                pass

        for stream, sink in ((proc.stdout, None), (proc.stderr, stderr_chunks)):
            if stream is not None:
                thread = threading.Thread(target=_drain, args=(stream, sink), daemon=True)
                thread.start()
                readers.append(thread)

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        try:
            while True:
                if _cancelled():
                    _terminate_child(proc, grace)
                    raise EvaluationCancelled()
                if deadline is not None and time.monotonic() >= deadline:
                    _terminate_child(proc, grace)
                    raise RuntimeError(
                        f"bart {' '.join(str(a) for a in argv)} timed out after {float(timeout):g}s"
                    )
                try:
                    proc.wait(timeout=poll_interval)
                except subprocess.TimeoutExpired:
                    continue
                break
            # Child has exited: its write ends are closed, so the readers reach
            # EOF and finish -- join them before we touch the captured bytes.
            for thread in readers:
                thread.join()
            stderr = b"".join(stderr_chunks)
            if proc.returncode != 0:
                message = stderr.decode("utf-8", "replace").strip()
                raise RuntimeError(
                    f"bart {' '.join(str(a) for a in argv)} failed "
                    f"(exit {proc.returncode}): {message}"
                )
            return read_cfl(out_stem)
        finally:
            # Never leave a child holding the temp dir open when we unwind (the
            # cancel/timeout paths already killed it; this covers error/GC too).
            if proc.poll() is None:
                _terminate_child(proc, grace)
            # Join readers (the now-dead child hit EOF) before closing the pipes
            # so we do not close a fd out from under a mid-read reader thread.
            for thread in readers:
                thread.join(timeout=grace + 1.0)
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()


# --- op fn builders ----------------------------------------------------------


def _axis_bitmask(axis: int, ndim: int) -> str:
    """BART dimension bitmask selecting a single (possibly negative) numpy axis."""

    resolved = int(axis) % int(ndim)
    return str(1 << resolved)


def _build_fft(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    del params
    resolved_axis = int(axis)  # requires_axis=True guarantees a bound axis

    def fn(data):
        data = np.asarray(data)
        bitmask = _axis_bitmask(resolved_axis, data.ndim)
        return run_bart(["fft", bitmask], data)

    return fn


def _build_ifft(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    del params
    resolved_axis = int(axis)

    def fn(data):
        data = np.asarray(data)
        bitmask = _axis_bitmask(resolved_axis, data.ndim)
        return run_bart(["fft", "-i", bitmask], data)

    return fn


def _build_cabs(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    del axis, params

    def fn(data):
        return run_bart(["cabs"], np.asarray(data))

    return fn


def _build_carg(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    del axis, params

    def fn(data):
        return run_bart(["carg"], np.asarray(data))

    return fn


def _build_scale(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    del axis
    factor = float(params["factor"])  # defensive coercion (recipe/CLI robustness)

    def fn(data):
        # ``bart scale <factor>`` multiplies every sample by the real scalar.
        return run_bart(["scale", repr(factor)], np.asarray(data))

    return fn


def _build_spow(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    del axis
    exponent = float(params["exponent"])

    def fn(data):
        # ``bart spow <exponent>`` raises every sample to the power (complex
        # principal branch for complex input).
        return run_bart(["spow", repr(exponent)], np.asarray(data))

    return fn


def _build_normalize(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
    del params
    resolved_axis = int(axis)  # requires_axis=True guarantees a bound axis

    def fn(data):
        data = np.asarray(data)
        bitmask = _axis_bitmask(resolved_axis, data.ndim)
        # ``bart normalize <bitmask>`` scales by the reciprocal L2 norm computed
        # along the selected axis; the norm broadcasts back, so shape is preserved.
        return run_bart(["normalize", bitmask], data)

    return fn


def _build_reduce(tool: str) -> Callable[[int | None, Mapping[str, object]], Callable]:
    """Return a ``build(axis, params)`` for a BART ``std`` / ``var`` reduction.

    BART reduces the selected axis to a *singleton* (size 1).  We reshape that
    singleton axis away so the output ndim drops by one -- matching the built-in
    reductions (``mean`` / ``rss`` / ...), and making the realized shape
    independent of ``read_cfl``'s trailing-singleton strip (which would otherwise
    keep or drop the reduced axis depending on its position).  The reshape is
    exact: a reduction-to-one has ``prod(shape) / shape[axis]`` elements, which is
    precisely ``prod(shape without axis)``.
    """

    def build(axis: int | None, params: Mapping[str, object]) -> Callable[[object], object]:
        del params
        resolved_axis = int(axis)

        def fn(data):
            data = np.asarray(data)
            ax = resolved_axis % data.ndim
            bitmask = _axis_bitmask(ax, data.ndim)
            result = np.asarray(run_bart([tool, bitmask], data))
            out_shape = data.shape[:ax] + data.shape[ax + 1 :]
            return result.reshape(out_shape)

        return fn

    return build


def _complex64_dtype(input_dtype):
    """Every BART op returns complex64 -- cfl has no other element type."""

    del input_dtype
    return np.dtype(np.complex64)


def _reduce_output_shape(shape, axis, params):
    """Reduction output shape: drop ``axis`` (ndim - 1), matching built-in reductions."""

    del params
    ax = int(axis) % len(shape)
    out = [int(s) for s in shape]
    del out[ax]
    return tuple(out)


def fft_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=FFT_ID,
        label="Centered FFT (BART)",
        build=_build_fft,
        output_dtype=_complex64_dtype,
        requires_axis=True,
        changes_shape=False,
        # Global transform along the axis, run out-of-process -> OPAQUE / Tier-1.
        region_capable=False,
        group="BART",
        description="Centered forward FFT along one axis (via the BART binary).",
        icon="waves",
    )


def ifft_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=IFFT_ID,
        label="Centered iFFT (BART, unnormalized)",
        build=_build_ifft,
        output_dtype=_complex64_dtype,
        requires_axis=True,
        changes_shape=False,
        region_capable=False,
        group="BART",
        description="Centered inverse FFT (unnormalized) along one axis (via BART).",
        icon="waves",
    )


def cabs_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=CABS_ID,
        label="Complex magnitude (BART)",
        build=_build_cabs,
        output_dtype=_complex64_dtype,
        requires_axis=False,
        changes_shape=False,
        # Pointwise, but kept OPAQUE: a per-region subprocess round-trip is never
        # the right execution for an expensive out-of-process op.
        region_capable=False,
        group="BART",
        description="Elementwise complex magnitude |x| (via the BART binary).",
        icon="filter_alt",
    )


def carg_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=CARG_ID,
        label="Complex phase (BART)",
        build=_build_carg,
        output_dtype=_complex64_dtype,
        requires_axis=False,
        changes_shape=False,
        # Pointwise, but kept OPAQUE for the same cost reason as cabs.
        region_capable=False,
        group="BART",
        description="Elementwise complex phase / argument atan2(Im, Re) (via the BART binary).",
        icon="rotate_right",
    )


def scale_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=SCALE_ID,
        label="Scale (BART)",
        build=_build_scale,
        output_dtype=_complex64_dtype,
        parameters=(
            OperationParameter(
                "factor",
                "Factor",
                kind="float",
                default=1.0,
                step=0.1,
                description="Multiply every sample by this real scalar.",
            ),
        ),
        requires_axis=False,
        changes_shape=False,
        region_capable=False,
        group="BART",
        description="Multiply every sample by a real scalar factor (via the BART binary).",
        icon="close_fullscreen",
    )


def spow_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=SPOW_ID,
        label="Power (BART)",
        build=_build_spow,
        output_dtype=_complex64_dtype,
        parameters=(
            OperationParameter(
                "exponent",
                "Exponent",
                kind="float",
                default=1.0,
                step=0.1,
                description="Raise every sample to this power (complex principal branch).",
            ),
        ),
        requires_axis=False,
        changes_shape=False,
        region_capable=False,
        group="BART",
        description="Raise every sample to a scalar power (via the BART binary).",
        icon="functions",
    )


def normalize_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=NORMALIZE_ID,
        label="Normalize along axis (BART)",
        build=_build_normalize,
        output_dtype=_complex64_dtype,
        requires_axis=True,
        changes_shape=False,
        # The scale factor depends on a whole-axis norm -> not windowable; and an
        # out-of-process op stays OPAQUE regardless.
        region_capable=False,
        group="BART",
        description="Scale by the reciprocal L2 norm computed along one axis (via BART).",
        icon="straighten",
    )


def std_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=STD_ID,
        label="Std along axis (BART)",
        build=_build_reduce("std"),
        output_shape=_reduce_output_shape,
        output_dtype=_complex64_dtype,
        requires_axis=True,
        changes_shape=True,
        # A reduction is never windowable, and it is out-of-process -> OPAQUE.
        region_capable=False,
        group="BART",
        description="Standard deviation along one axis, collapsing it (via the BART binary).",
        icon="show_chart",
    )


def var_spec() -> PluginOperationSpec:
    return PluginOperationSpec(
        id=VAR_ID,
        label="Variance along axis (BART)",
        build=_build_reduce("var"),
        output_shape=_reduce_output_shape,
        output_dtype=_complex64_dtype,
        requires_axis=True,
        changes_shape=True,
        region_capable=False,
        group="BART",
        description="Variance along one axis, collapsing it (via the BART binary).",
        icon="show_chart",
    )


def pack_specs() -> tuple[PluginOperationSpec, ...]:
    """The specs this pack contributes (independent of bart being installed)."""

    return (
        fft_spec(),
        ifft_spec(),
        cabs_spec(),
        carg_spec(),
        scale_spec(),
        spow_spec(),
        normalize_spec(),
        std_spec(),
        var_spec(),
    )


def register(register_fn=register_pack_operation) -> bool:
    """Register the BART ops iff ``bart`` is runnable.

    Returns ``True`` when the pack contributed its ops, ``False`` when ``bart`` is
    absent (the pack silently contributes nothing).  Idempotent.  Called by
    :func:`arrayscope.operations.registry.load_operation_packs`.
    """

    if not bart_available():
        return False
    for spec in pack_specs():
        register_fn(spec)
    return True
