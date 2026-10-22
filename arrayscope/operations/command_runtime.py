"""Out-of-process array runtimes shared by command and Python operations.

The runtime owns four things and nothing operation-specific:

* explicit command-template tokenization (``shell=False`` by default),
* array handoff through a named format,
* discarded stdout and a bounded diagnostic tail from stderr, and
* bounded cancellation/timeout teardown (the whole child group on POSIX).

``cfl`` and ``npy`` are the first two handoffs.  The small registry at the
bottom is deliberately the extension seam for later NIfTI/raw support.
"""

from __future__ import annotations

import os
import shlex
import signal
import string
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arrayscope.operations.cancellation import EvaluationCancelled

DEFAULT_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 0.02
TERM_GRACE_S = 0.25
MAX_STDERR_BYTES = 64 * 1024


@dataclass(frozen=True)
class ArrayHandoff:
    """Filesystem representation used for one array subprocess round trip.

    ``rank_floor_aware`` marks a ``read`` that accepts a ``min_rank`` keyword.
    A handoff registered without it keeps the plain ``read(path)`` contract, so
    a third-party entry in :data:`ARRAY_HANDOFFS` is unaffected.
    """

    name: str
    input_name: str
    output_name: str
    write: Callable[[str, np.ndarray], None]
    read: Callable[..., np.ndarray]
    rank_floor_aware: bool = False


def write_cfl(stem: str, array: np.ndarray) -> None:
    """Write BART's complex64, Fortran-ordered ``.hdr``/``.cfl`` pair.

    The cast to complex64 is lossy and unconditional because that is the only
    thing BART's format stores: a float64 or int array loses precision here and
    comes back complex. See ``docs/plugin-operations.md``.
    """

    value = np.asarray(array)
    if value.dtype != np.complex64:
        value = value.astype(np.complex64)
    with open(stem + ".hdr", "w", encoding="utf-8") as handle:
        handle.write("# Dimensions\n")
        handle.write(" ".join(str(int(size)) for size in value.shape))
        handle.write("\n")
    with open(stem + ".cfl", "wb") as handle:
        handle.write(np.ascontiguousarray(value.T))


def read_cfl(stem: str, *, min_rank: int = 1) -> np.ndarray:
    """Read BART's ``.hdr``/``.cfl`` pair into a complex64 array.

    Trailing singleton dimensions are dropped down to ``min_rank`` because BART
    writes a fixed 16-dimension header and reading that literally would return
    rank-16 arrays. Dropping them *unconditionally*, however, silently changes
    an operation's rank: ``(4, 3, 1)`` round-tripped to ``(4, 3)``. Probe shapes
    preserve size-1 axes, so a cfl-handoff operation was then characterized as
    rank-reducing when it was the transport, not the operation, that reduced.

    ``min_rank`` is therefore a floor, not a promise: the runtime passes the
    rank of the array it just *wrote*, so a rank-preserving tool round-trips
    exactly, while a tool that genuinely returns fewer meaningful axes keeps
    them as explicit singletons -- which is BART's own convention anyway.
    """

    with open(stem + ".hdr", encoding="utf-8") as handle:
        next(handle)
        dims = [int(token) for token in next(handle).split()]
    floor = max(1, int(min_rank))
    while len(dims) > floor and dims[-1] == 1:
        dims.pop()
    with open(stem + ".cfl", "rb") as handle:
        data = np.frombuffer(handle.read(), dtype=np.complex64)
    return np.reshape(data[: int(np.prod(dims))], dims, order="F")


def _write_npy(path: str, array: np.ndarray) -> None:
    np.save(path, np.asarray(array), allow_pickle=False)


def _read_npy(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


ARRAY_HANDOFFS: dict[str, ArrayHandoff] = {
    "cfl": ArrayHandoff("cfl", "input", "output", write_cfl, read_cfl, rank_floor_aware=True),
    "npy": ArrayHandoff("npy", "input.npy", "output.npy", _write_npy, _read_npy),
}


def array_handoff(name: str) -> ArrayHandoff:
    try:
        return ARRAY_HANDOFFS[str(name)]
    except KeyError as exc:
        supported = ", ".join(sorted(ARRAY_HANDOFFS))
        raise ValueError(
            f"unsupported array handoff {name!r}; expected one of: {supported}"
        ) from exc


class _StrictFields(dict):
    def __missing__(self, key):
        raise ValueError(f"command template references unknown placeholder {{{key}}}")


def _rendered(value: object) -> str:
    """One parameter value as a command-line token.

    A boolean renders lowercase ``true`` / ``false``: Python's ``"True"`` is not
    what any command-line tool, JSON reader, or shell test expects, and the
    wrapper author has to be able to predict the text without knowing the host
    language. Every other value is its plain ``str``; argv boundaries are the
    tokenizer's job, not this function's.
    """

    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def template_fields(template: str) -> frozenset[str]:
    """Return simple named placeholders and reject attribute/index expressions."""

    names: set[str] = set()
    try:
        parsed = string.Formatter().parse(str(template))
        for _literal, field_name, _format_spec, _conversion in parsed:
            if field_name is None:
                continue
            if not field_name or any(char in field_name for char in ".["):
                raise ValueError(
                    "command placeholders must be simple names such as {in}, {out}, or {iters}"
                )
            names.add(field_name)
    except ValueError:
        raise
    return frozenset(names)


def build_command(
    template: str,
    values: Mapping[str, object],
    *,
    shell: bool = False,
    prefix: Sequence[str] = (),
) -> list[str] | str:
    """Render a template without allowing values to change argv boundaries.

    With the safe default, the authored template is tokenized first and each
    token is formatted second.  A value containing spaces or looking like a
    flag therefore remains one literal argument.  Only the explicit ``shell``
    opt-in returns a single command string.
    """

    template = str(template).strip()
    if not template:
        raise ValueError("command template is empty")
    fields = template_fields(template)
    missing = sorted(fields - set(values))
    if missing:
        raise ValueError(f"command template is missing values for: {', '.join(missing)}")
    rendered_values = _StrictFields({str(key): _rendered(value) for key, value in values.items()})
    if shell:
        rendered = template.format_map(rendered_values)
        if prefix:
            raise ValueError("interpreter prefixes are not supported with shell execution")
        return rendered
    try:
        tokens = shlex.split(template, posix=True)
    except ValueError as exc:
        raise ValueError(f"invalid command template quoting: {exc}") from exc
    return [
        *[str(token) for token in prefix],
        *[token.format_map(rendered_values) for token in tokens],
    ]


def _terminate_child(proc, grace: float = TERM_GRACE_S) -> None:
    """Stop the dedicated child group, even after its leader has exited.

    ``start_new_session=True`` makes the child's PID the POSIX group ID. A
    lookup through ``getpgid`` after reaping the leader loses that identity
    while descendants can still be alive and holding our pipes open.
    """

    group = os.name == "posix"

    def send(*, kill: bool = False) -> None:
        with suppress(ProcessLookupError):
            if group:
                os.killpg(proc.pid, signal.SIGKILL if kill else signal.SIGTERM)
            elif proc.poll() is None:
                # Windows has no killpg; terminate/kill are its supported
                # direct-process operations and do not use POSIX signals.
                if kill:
                    proc.kill()
                else:
                    proc.terminate()

    send()
    deadline = time.monotonic() + max(0.0, float(grace))
    while time.monotonic() < deadline:
        if group:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
        elif proc.poll() is not None:
            break
        time.sleep(0.005)
    # A descendant can ignore TERM after the leader has exited. Polling only
    # the leader here used to leave that descendant running indefinitely.
    send(kill=True)
    proc.wait(timeout=max(1.0, float(grace)))


def _child_environment(overrides: Mapping[str, object] | None) -> dict[str, str]:
    environment = dict(os.environ)
    if overrides:
        environment.update({str(key): str(value) for key, value in overrides.items()})
    return environment


def run_array_command(
    command: Sequence[str] | str,
    array: np.ndarray,
    *,
    inputs: Mapping[str, np.ndarray | str] | None = None,
    handoff: str = "npy",
    shell: bool = False,
    cancellation_token: object | None = None,
    env: Mapping[str, object] | None = None,
    cwd: str | None = None,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    poll_interval: float = POLL_INTERVAL_S,
    grace: float = TERM_GRACE_S,
    temp_dir: str | None = None,
) -> np.ndarray:
    """Execute an already-built command with ``{in}``/``{out}`` handoff paths.

    ``command`` must already contain the concrete input/output paths.  Most
    callers should use :func:`run_command_template`, which creates them and
    preserves argv boundaries.

    An ``inputs`` value that is a ``str`` is a *file* slot: its path substitutes
    as-is, with no copy into the scratch directory.  Copying would be wrong as
    well as wasteful -- a model checkpoint or a database is named, not handed
    over, and may be far larger than the array itself.
    """

    import subprocess
    import tempfile
    import threading

    format_spec = array_handoff(handoff)

    def cancelled() -> bool:
        return bool(
            cancellation_token is not None and getattr(cancellation_token, "cancelled", False)
        )

    with tempfile.TemporaryDirectory(prefix="arrayscope-command-", dir=temp_dir) as scratch:
        input_path = os.path.join(scratch, format_spec.input_name)
        output_path = os.path.join(scratch, format_spec.output_name)
        written = np.asarray(array)
        format_spec.write(input_path, written)
        # The runtime is the only layer that knows the rank it handed over, so
        # it is the only one that can stop a transport-level rank drop from
        # being read back as an operation's shape rule (see ``read_cfl``).
        read_options = {"min_rank": written.ndim} if format_spec.rank_floor_aware else {}
        slot_paths: dict[str, str] = {}
        for name, value in sorted(dict(inputs or {}).items()):
            if isinstance(value, str):
                slot_paths[str(name)] = value
                continue
            safe_name = "".join(
                character if character.isalnum() or character in "_-" else "_"
                for character in str(name)
            )
            suffix = ".npy" if format_spec.name == "npy" else ""
            path = os.path.join(scratch, f"input-{safe_name}{suffix}")
            format_spec.write(path, np.asarray(value))
            slot_paths[str(name)] = path
        if cancelled():
            raise EvaluationCancelled()

        # Callers pass sentinel path tokens so this lower-level function can
        # still own scratch lifetime without reparsing a template.
        replacements = {
            "{in}": input_path,
            "{out}": output_path,
            **{f"{{slot:{name}}}": path for name, path in slot_paths.items()},
        }
        if isinstance(command, str):
            child_command: str | list[str] = command
            for placeholder, path in replacements.items():
                child_command = child_command.replace(placeholder, shlex.quote(path))
        else:
            child_command = []
            for token in command:
                resolved = str(token)
                for placeholder, path in replacements.items():
                    resolved = resolved.replace(placeholder, path)
                child_command.append(resolved)

        stderr_tail = bytearray()
        stderr_bytes = 0
        stderr_done = threading.Event()
        stop_reader = threading.Event()

        def drain() -> None:
            nonlocal stderr_bytes
            try:
                while not stop_reader.is_set():
                    try:
                        chunk = os.read(stderr_fd, 65536)
                    except BlockingIOError:
                        stop_reader.wait(max(0.001, float(poll_interval)))
                        continue
                    if not chunk:
                        break
                    stderr_bytes += len(chunk)
                    stderr_tail.extend(chunk)
                    del stderr_tail[: max(0, len(stderr_tail) - MAX_STDERR_BYTES)]
            except (OSError, ValueError):
                pass
            finally:
                stderr_done.set()

        reader = threading.Thread(target=drain, daemon=True)
        proc = subprocess.Popen(
            child_command,
            cwd=cwd or None,
            env=_child_environment(env),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            shell=bool(shell),
            # Raw FileIO avoids BufferedReader.close waiting for a blocked
            # reader's lock when a descendant inherited the pipe.
            bufsize=0,
        )
        try:
            stderr_fd = proc.stderr.fileno()
            with suppress(OSError, NotImplementedError):
                # POSIX pipes, and Windows pipes on Python >=3.12, support
                # nonblocking reads. Older Windows versions use the bounded
                # join and raw close below, but cannot interrupt a pipe read
                # held open by a descendant outside direct-process cleanup.
                os.set_blocking(stderr_fd, False)
            reader.start()
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            # Keep an exited POSIX leader unreaped while descendants hold the
            # pipe: its reserved PID also reserves our process-group identity,
            # so timeout cleanup cannot signal a later group reusing that PID.
            while not stderr_done.is_set() or proc.poll() is None:
                if cancelled():
                    raise EvaluationCancelled()
                if deadline is not None and time.monotonic() >= deadline:
                    raise RuntimeError(f"command timed out after {float(timeout):g}s")
                if not stderr_done.is_set():
                    stderr_done.wait(max(0.001, float(poll_interval)))
                    continue
                try:
                    proc.wait(timeout=max(0.001, float(poll_interval)))
                except subprocess.TimeoutExpired:
                    continue
            stderr = stderr_tail.decode("utf-8", "replace").strip()
            if stderr_bytes > MAX_STDERR_BYTES:
                stderr = f"[stderr truncated; last {MAX_STDERR_BYTES} bytes]\n{stderr}"
            if proc.returncode != 0:
                label = child_command if isinstance(child_command, str) else " ".join(child_command)
                raise RuntimeError(f"command {label!r} failed (exit {proc.returncode}): {stderr}")
            return format_spec.read(output_path, **read_options)
        finally:
            try:
                if not stderr_done.is_set() or proc.poll() is None:
                    _terminate_child(proc, grace)
            finally:
                stop_reader.set()
                if reader.ident is not None:
                    reader.join(timeout=max(0.0, float(grace)) + 1.0)
                proc.stderr.close()


def run_command_template(
    template: str,
    array: np.ndarray,
    *,
    parameters: Mapping[str, object] | None = None,
    inputs: Mapping[str, np.ndarray | str] | None = None,
    handoff: str = "npy",
    shell: bool = False,
    prefix: Sequence[str] = (),
    cancellation_token: object | None = None,
    env: Mapping[str, object] | None = None,
    cwd: str | None = None,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    poll_interval: float = POLL_INTERVAL_S,
    grace: float = TERM_GRACE_S,
    temp_dir: str | None = None,
) -> np.ndarray:
    """Build and run a user-authored command template over one array."""

    fields = template_fields(template)
    if not {"in", "out"}.issubset(fields):
        raise ValueError("command template must contain both {in} and {out}")
    values = dict(parameters or {})
    slot_inputs = dict(inputs or {})
    overlap = sorted(set(slot_inputs) & set(values))
    if overlap:
        raise ValueError(f"input slot names overlap parameter names: {', '.join(overlap)}")
    values.update(
        {
            "in": "{in}",
            "out": "{out}",
            **{str(name): f"{{slot:{name}}}" for name in slot_inputs},
        }
    )
    command = build_command(template, values, shell=shell, prefix=prefix)
    return run_array_command(
        command,
        array,
        inputs=slot_inputs,
        handoff=handoff,
        shell=shell,
        cancellation_token=cancellation_token,
        env=env,
        cwd=cwd,
        timeout=timeout,
        poll_interval=poll_interval,
        grace=grace,
        temp_dir=temp_dir,
    )


def command_executable(command: Sequence[str] | str, *, shell: bool = False) -> str | None:
    """First executable token for availability checks, or the platform shell."""

    if shell:
        return os.environ.get("SHELL") or "/bin/sh"
    if isinstance(command, str):
        tokens = shlex.split(command, posix=True)
    else:
        tokens = [str(token) for token in command]
    return tokens[0] if tokens else None


def is_executable(path_or_name: str, *, env: Mapping[str, str] | None = None) -> bool:
    """Resolve an absolute/relative executable without running it."""

    import shutil

    candidate = os.path.expanduser(str(path_or_name))
    if os.path.dirname(candidate):
        path = Path(candidate)
        return path.is_file() and os.access(path, os.X_OK)
    search_path = None if env is None else env.get("PATH")
    return shutil.which(candidate, path=search_path) is not None
