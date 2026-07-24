"""Tests for the optional in-process BART operation pack.

Covers: ``bart:fft`` correctness against a NumPy centered-FFT reference (BART's
own centered/unnormalized convention, verified end-to-end through the real
subprocess + cfl handoff); a cfl write/read round-trip; the load-bearing
**cancellation <1 s** gate (SIGTERM kills the child promptly, no orphan, temp dir
cleaned) driven deterministically by a fake ``bart`` shim + a startup barrier;
optionality (bart-absent -> the pack registers nothing) and laziness
(enumeration never spawns ``bart``).

The real-BART assertions ``pytest.skip`` cleanly when ``bart`` is not runnable so
CI without BART stays green; on a machine where BART is installed they execute.
"""

from __future__ import annotations

import glob
import os
import stat
import tempfile
import threading
import time

import numpy as np
import pytest

from arrayscope.kernel.task import CancellationToken
from arrayscope.operations import plugins, registry
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.packs import bart_pack
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_S

PROBE_SHAPE = (6, 5, 4)

# Fallback locations for the real-BART tests.  Prefer explicit env overrides
# (``ARRAYSCOPE_BART_TOOLBOX`` / ``ARRAYSCOPE_BART_MKL_LIB``) so the recipe is not
# tied to one contributor's machine; the hardcoded paths remain only as a
# documented local convenience for this workstation.  Used only to populate the
# env when the caller has not already; if they are absent the real-BART tests
# skip.
_DEFAULT_TOOLBOX = os.environ.get("ARRAYSCOPE_BART_TOOLBOX", "/home/thomas/projects/bart")
_DEFAULT_MKL_LIB = os.environ.get(
    "ARRAYSCOPE_BART_MKL_LIB", "/home/thomas/miniconda3/pkgs/mkl-2025.3.0-h0e700b2_462/lib"
)


@pytest.fixture(autouse=True)
def _clean_pack_state():
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()
    yield
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()


@pytest.fixture
def bart_env(monkeypatch):
    """Ensure BART_TOOLBOX_PATH (+ MKL lib path) are set, else skip cleanly."""

    if not os.environ.get(bart_pack.BART_TOOLBOX_ENV):
        if os.path.isfile(os.path.join(_DEFAULT_TOOLBOX, "bart")):
            monkeypatch.setenv(bart_pack.BART_TOOLBOX_ENV, _DEFAULT_TOOLBOX)
        else:
            pytest.skip("BART_TOOLBOX_PATH not set and no default toolbox present")
    if "mkl" not in os.environ.get("LD_LIBRARY_PATH", "") and os.path.isdir(_DEFAULT_MKL_LIB):
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        monkeypatch.setenv(
            "LD_LIBRARY_PATH", f"{_DEFAULT_MKL_LIB}:{existing}" if existing else _DEFAULT_MKL_LIB
        )
    if not bart_pack.bart_available():
        pytest.skip("bart is not runnable in this environment")


def _reference_fft(x, axis):
    ax = axis % x.ndim
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)


# --- cfl round-trip (self-contained I/O) -------------------------------------


@pytest.mark.parametrize("dtype", ["complex64", "float32", "float64", "int16"])
def test_cfl_round_trips(tmp_path, dtype):
    rng = np.random.default_rng(0)
    real = rng.standard_normal(PROBE_SHAPE)
    x = real.astype(dtype)
    stem = str(tmp_path / "probe")
    bart_pack.write_cfl(stem, x)
    got = bart_pack.read_cfl(stem)
    assert got.dtype == np.complex64
    assert got.shape == PROBE_SHAPE
    # complex64 promotion is exact for the real part we wrote.
    np.testing.assert_allclose(got, x.astype(np.complex64))


# --- discovery / enumeration -------------------------------------------------


def test_bart_ops_appear_when_available(bart_env):
    ids = {entry.id for entry in registry.all_operations()}
    assert {"bart:fft", "bart:ifft", "bart:cabs"} <= ids
    fft_entry = registry.get_operation_entry("bart:fft")
    assert fft_entry.label == "Centered FFT (BART)"
    assert fft_entry.requires_axis is True
    assert fft_entry.changes_shape is False


def test_enumeration_never_spawns_bart(monkeypatch):
    """Laziness / import-health: listing ops must not execute bart."""

    import subprocess

    def _boom(*args, **kwargs):
        raise AssertionError("enumerating operations must not spawn bart")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    registry.all_operations()  # must not raise


# --- optionality: bart-absent contributes nothing ----------------------------


def test_pack_contributes_nothing_when_bart_absent(monkeypatch):
    monkeypatch.setattr(bart_pack, "bart_available", lambda: False)
    registry._reset_operation_packs()
    assert bart_pack.register() is False
    ids = {entry.id for entry in registry.all_operations()}
    assert not any(op_id.startswith("bart:") for op_id in ids)
    assert "centered_fft" in ids  # built-ins untouched


def test_pack_specs_exist_independently_of_installation():
    ids = {spec.id for spec in bart_pack.pack_specs()}
    assert ids == {"bart:fft", "bart:ifft", "bart:cabs"}


# --- correctness against a NumPy reference (real bart subprocess) -------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_bart_fft_matches_numpy_centered_reference(bart_env, axis):
    rng = np.random.default_rng(axis)
    x = (rng.standard_normal(PROBE_SHAPE) + 1j * rng.standard_normal(PROBE_SHAPE)).astype(
        np.complex64
    )
    op = registry.create_operation("bart:fft", axis=axis)
    got = np.asarray(op.apply(x))
    assert got.dtype == np.complex64
    ref = _reference_fft(x, axis).astype(np.complex64)
    # BART is single precision; compare relative to the transform magnitude.
    assert np.max(np.abs(got - ref)) <= 1e-3 * np.max(np.abs(ref))
    assert np.dtype(op.output_dtype(np.dtype("float32"))) == np.complex64


def test_bart_fft_ifft_round_trips_up_to_bart_normalization(bart_env):
    # BART's fft/ifft are both unnormalized -> ifft(fft(x)) == N * x along the axis.
    x = (np.arange(120).reshape(PROBE_SHAPE)).astype(np.complex64)
    fwd = registry.create_operation("bart:fft", axis=1)
    inv = registry.create_operation("bart:ifft", axis=1)
    y = np.asarray(inv.apply(np.asarray(fwd.apply(x))))
    n = PROBE_SHAPE[1]
    np.testing.assert_allclose(y, x * n, rtol=0, atol=1e-2 * np.max(np.abs(x * n)))


def test_bart_cabs_matches_numpy_abs(bart_env):
    rng = np.random.default_rng(7)
    x = (rng.standard_normal(PROBE_SHAPE) + 1j * rng.standard_normal(PROBE_SHAPE)).astype(
        np.complex64
    )
    op = registry.create_operation("bart:cabs")
    got = np.asarray(op.apply(x))
    np.testing.assert_allclose(got.real, np.abs(x), atol=1e-4)
    np.testing.assert_allclose(got.imag, 0.0, atol=1e-5)


# --- admission: the op is classified OPAQUE / heavy TRANSFORM -----------------


def test_bart_fft_is_opaque_and_costed_heavy(bart_env):
    from arrayscope.operations.capabilities import OperationClass
    from arrayscope.operations.cost import estimate_operation_cost

    op = registry.create_operation("bart:fft", axis=0)
    assert op.execution_class is OperationClass.OPAQUE
    # Real (float32) input -> forced complex64 output = 2x the bytes: an honest,
    # heavier admission signal, on a non-chunkable whole-array TRANSFORM stage.
    cost = estimate_operation_cost(PROBE_SHAPE, np.dtype("float32"), op)
    assert cost.kind == "transform"
    assert cost.can_chunk is False
    assert cost.chunkable_axes == ()
    assert cost.estimated_output_bytes == int(np.prod(PROBE_SHAPE)) * 8  # complex64


# --- cancellation <1 s (the load-bearing gate) -------------------------------


def _write_fake_bart(tmp_path, marker):
    """A ``bart`` shim that records its pid to ``marker`` then blocks forever.

    ``exec sleep`` keeps the same pid/process-group so a SIGTERM to the group we
    started kills exactly the process whose pid we observe -- the barrier
    (waiting for the marker) makes the cancel deterministic, not time-based.
    """

    script = tmp_path / "fake_bart"
    script.write_text(f'#!/bin/bash\necho $$ > "{marker}"\nexec sleep 300\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_cancel_mid_op_kills_child_under_one_second(tmp_path):
    marker = tmp_path / "child_pid"
    fake_bart = _write_fake_bart(tmp_path, marker)
    token = CancellationToken()

    temp_root = tempfile.gettempdir()
    before = set(glob.glob(os.path.join(temp_root, "arrayscope-bart-*")))

    result: dict[str, object] = {}

    def _runner():
        try:
            bart_pack.run_bart(
                ["fft", "0"],
                np.ones((4, 4), dtype=np.complex64),
                cancellation_token=token,
                executable=fake_bart,
            )
        except BaseException as exc:
            result["exc"] = exc

    thread = threading.Thread(target=_runner)
    thread.start()

    # Barrier: wait until the child has actually started (marker written).
    # The barrier deadline is owned by the one interaction-budget owner, not a
    # local literal (architecture guard: one bounded timeout owner).
    deadline = time.monotonic() + INTERACTION_SETTLE_HARD_LIMIT_S
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert marker.exists(), "fake bart never started"
    child_pid = int(marker.read_text().strip())
    assert _pid_alive(child_pid)

    cancel_at = time.monotonic()
    token.cancel()
    thread.join(timeout=3.0)
    elapsed = time.monotonic() - cancel_at

    assert not thread.is_alive(), "run_bart did not return after cancel"
    assert elapsed < 1.0, f"cancel took {elapsed:.3f}s (>1s)"
    assert isinstance(result.get("exc"), EvaluationCancelled)

    # The child (and its process group) is dead -- no orphan.
    for _ in range(50):
        if not _pid_alive(child_pid):
            break
        time.sleep(0.01)
    assert not _pid_alive(child_pid), "bart child survived the cancel"

    # The temp dir was always cleaned up, even on cancel.
    after = set(glob.glob(os.path.join(temp_root, "arrayscope-bart-*")))
    assert after <= before, f"leaked temp dirs: {after - before}"


def test_already_cancelled_never_spawns(tmp_path):
    marker = tmp_path / "child_pid"
    fake_bart = _write_fake_bart(tmp_path, marker)
    token = CancellationToken()
    token.cancel()
    with pytest.raises(EvaluationCancelled):
        bart_pack.run_bart(
            ["fft", "0"],
            np.ones((2, 2), dtype=np.complex64),
            cancellation_token=token,
            executable=fake_bart,
        )
    assert not marker.exists(), "run_bart spawned bart despite pre-cancel"


# --- pipe draining + overall timeout (no real BART toolbox needed) ------------


# Comfortably past the ~64 KB OS pipe buffer, on BOTH stdout and stderr: an
# undrained ``Popen(PIPE)`` child that writes this much blocks on write forever.
_CHATTY_BYTES = 200_000


def _make_executable(script) -> str:
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def _write_chatty_then_exit_bart(tmp_path) -> str:
    """A ``bart`` shim that floods both pipes, echoes its input to ``out``, exits 0.

    ``run_bart`` appends ``in`` and ``out`` stems as the last two argv, so with an
    empty command they land as ``$1``/``$2``.  Copying the cfl pair lets the real
    ``read_cfl`` round-trip succeed, so a green result *proves* the flood was
    drained rather than deadlocking.
    """

    script = tmp_path / "chatty_bart"
    script.write_text(
        "#!/bin/bash\n"
        'in="$1"\n'
        'out="$2"\n'
        f"yes 'stdout-noise' | head -c {_CHATTY_BYTES}\n"
        f"yes 'stderr-noise' | head -c {_CHATTY_BYTES} 1>&2\n"
        'cp "$in.cfl" "$out.cfl"\n'
        'cp "$in.hdr" "$out.hdr"\n'
    )
    return _make_executable(script)


def _write_chatty_then_hang_bart(tmp_path, marker) -> str:
    """A ``bart`` shim that floods both pipes, records its pid, then blocks forever."""

    script = tmp_path / "chatty_hang_bart"
    script.write_text(
        "#!/bin/bash\n"
        f'echo $$ > "{marker}"\n'
        f"yes 'stdout-noise' | head -c {_CHATTY_BYTES}\n"
        f"yes 'stderr-noise' | head -c {_CHATTY_BYTES} 1>&2\n"
        "exec sleep 300\n"
    )
    return _make_executable(script)


def test_bart_timeout_env_config(monkeypatch):
    """The overall-timeout ceiling is env-driven with a documented fallback."""

    monkeypatch.delenv(bart_pack.BART_TIMEOUT_ENV, raising=False)
    assert bart_pack.bart_timeout() == bart_pack._DEFAULT_TIMEOUT_S
    monkeypatch.setenv(bart_pack.BART_TIMEOUT_ENV, "12.5")
    assert bart_pack.bart_timeout() == 12.5
    monkeypatch.setenv(bart_pack.BART_TIMEOUT_ENV, "0")  # non-positive disables
    assert bart_pack.bart_timeout() is None
    monkeypatch.setenv(bart_pack.BART_TIMEOUT_ENV, "nonsense")  # malformed -> default
    assert bart_pack.bart_timeout() == bart_pack._DEFAULT_TIMEOUT_S


def test_chatty_child_is_drained_and_does_not_deadlock(tmp_path):
    """A child flooding both pipes past the buffer must not hang the runner."""

    fake_bart = _write_chatty_then_exit_bart(tmp_path)
    x = (np.arange(120).reshape(PROBE_SHAPE)).astype(np.complex64)

    result: dict[str, object] = {}

    def _runner():
        try:
            # A bounded ceiling so a *regression* (undrained -> blocked child)
            # fails as a timeout rather than wedging the whole suite forever.
            result["out"] = bart_pack.run_bart([], x, executable=fake_bart, timeout=30.0)
        except BaseException as exc:
            result["exc"] = exc

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join(timeout=INTERACTION_SETTLE_HARD_LIMIT_S)

    assert not thread.is_alive(), "run_bart deadlocked on an undrained chatty child"
    assert "exc" not in result, f"unexpected error: {result.get('exc')!r}"
    np.testing.assert_allclose(np.asarray(result["out"]), x)


def test_overall_timeout_kills_stuck_child(tmp_path):
    """A child that floods then hangs is killed by the overall timeout, no orphan."""

    marker = tmp_path / "child_pid"
    fake_bart = _write_chatty_then_hang_bart(tmp_path, marker)

    temp_root = tempfile.gettempdir()
    before = set(glob.glob(os.path.join(temp_root, "arrayscope-bart-*")))

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        bart_pack.run_bart(
            [],
            np.ones((4, 4), dtype=np.complex64),
            executable=fake_bart,
            timeout=0.5,
        )
    elapsed = time.monotonic() - started
    assert elapsed < INTERACTION_SETTLE_HARD_LIMIT_S, f"timeout was not prompt ({elapsed:.2f}s)"

    # The flooding child recorded its pid; it (and its group) must be dead.
    assert marker.exists(), "chatty child never started"
    child_pid = int(marker.read_text().strip())
    for _ in range(50):
        if not _pid_alive(child_pid):
            break
        time.sleep(0.01)
    assert not _pid_alive(child_pid), "stuck bart child survived the timeout"

    # The temp dir was cleaned up even though we bailed on a timeout.
    after = set(glob.glob(os.path.join(temp_root, "arrayscope-bart-*")))
    assert after <= before, f"leaked temp dirs: {after - before}"
