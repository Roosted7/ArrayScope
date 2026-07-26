"""Out-of-process array runtimes shared by command and Python operations.

The runtime owns four things and nothing operation-specific:

* explicit command-template tokenization (``shell=False`` by default),
* array handoff through a named format,
* concurrent stdout/stderr draining, and
* bounded cancellation/timeout teardown of the whole child process group.

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arrayscope.operations.cancellation import EvaluationCancelled

DEFAULT_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 0.02
TERM_GRACE_S = 0.25


@dataclass(frozen=True)
class ArrayHandoff:
    """Filesystem representation used for one array subprocess round trip."""

    name: str
    input_name: str
    output_name: str
    write: Callable[[str, np.ndarray], None]
    read: Callable[[str], np.ndarray]


def write_cfl(stem: str, array: np.ndarray) -> None:
    """Write BART's complex64, Fortran-ordered ``.hdr``/``.cfl`` pair."""

    value = np.asarray(array)
    if value.dtype != np.complex64:
        value = value.astype(np.complex64)
    with open(stem + ".hdr", "w", encoding="utf-8") as handle:
        handle.write("# Dimensions\n")
        handle.write(" ".join(str(int(size)) for size in value.shape))
        handle.write("\n")
    with open(stem + ".cfl", "wb") as handle:
        handle.write(np.ascontiguousarray(value.T))


def read_cfl(stem: str) -> np.ndarray:
    """Read BART's ``.hdr``/``.cfl`` pair into a complex64 array."""

    with open(stem + ".hdr", encoding="utf-8") as handle:
        next(handle)
        dims = [int(token) for token in next(handle).split()]
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    with open(stem + ".cfl", "rb") as handle:
        data = np.frombuffer(handle.read(), dtype=np.complex64)
    return np.reshape(data[: int(np.prod(dims))], dims, order="F")


def _write_npy(path: str, array: np.ndarray) -> None:
    np.save(path, np.asarray(array), allow_pickle=False)


def _read_npy(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


ARRAY_HANDOFFS: dict[str, ArrayHandoff] = {
    "cfl": ArrayHandoff("cfl", "input", "output", write_cfl, read_cfl),
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
    rendered_values = _StrictFields({str(key): str(value) for key, value in values.items()})
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
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    deadline = time.monotonic() + max(0.0, float(grace))
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.005)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    proc.wait()


def _child_environment(overrides: Mapping[str, object] | None) -> dict[str, str]:
    environment = dict(os.environ)
    if overrides:
        environment.update({str(key): str(value) for key, value in overrides.items()})
    return environment


def run_array_command(
    command: Sequence[str] | str,
    array: np.ndarray,
    *,
    inputs: Mapping[str, np.ndarray] | None = None,
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
        format_spec.write(input_path, np.asarray(array))
        slot_paths: dict[str, str] = {}
        for name, value in sorted(dict(inputs or {}).items()):
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

        proc = subprocess.Popen(
            child_command,
            cwd=cwd or None,
            env=_child_environment(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            shell=bool(shell),
        )
        stderr_chunks: list[bytes] = []
        readers: list[threading.Thread] = []

        def drain(stream, sink: list[bytes] | None) -> None:
            try:
                for chunk in iter(lambda: stream.read(65536), b""):
                    if sink is not None:
                        sink.append(chunk)
            except (OSError, ValueError):
                pass

        for stream, sink in ((proc.stdout, None), (proc.stderr, stderr_chunks)):
            if stream is not None:
                reader = threading.Thread(target=drain, args=(stream, sink), daemon=True)
                reader.start()
                readers.append(reader)

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        try:
            while True:
                if cancelled():
                    _terminate_child(proc, grace)
                    raise EvaluationCancelled()
                if deadline is not None and time.monotonic() >= deadline:
                    _terminate_child(proc, grace)
                    raise RuntimeError(f"command timed out after {float(timeout):g}s")
                try:
                    proc.wait(timeout=max(0.001, float(poll_interval)))
                except subprocess.TimeoutExpired:
                    continue
                break
            for reader in readers:
                reader.join()
            stderr = b"".join(stderr_chunks).decode("utf-8", "replace").strip()
            if proc.returncode != 0:
                label = child_command if isinstance(child_command, str) else " ".join(child_command)
                raise RuntimeError(f"command {label!r} failed (exit {proc.returncode}): {stderr}")
            return format_spec.read(output_path)
        finally:
            if proc.poll() is None:
                _terminate_child(proc, grace)
            for reader in readers:
                reader.join(timeout=grace + 1.0)
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()


def run_command_template(
    template: str,
    array: np.ndarray,
    *,
    parameters: Mapping[str, object] | None = None,
    inputs: Mapping[str, np.ndarray] | None = None,
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
