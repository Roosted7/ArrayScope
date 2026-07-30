"""Named execution-environment records for user operations.

Records live in ``operations/environments.json`` beside operation wrappers.
Resolution is lazy and side-effect free apart from the bounded ``conda env
list`` query needed to distinguish a real named env from a vanished one.

A named conda environment resolves to its *prefix directory*, and the child is
then launched with ``<prefix>/bin/python`` and ``<prefix>/bin`` prepended to
``PATH``.  ``conda run`` remains only as the fallback for a prefix that holds no
interpreter: it costs a measured ~1.55 s of conda start-up per call, and an
operation pays that per characterization probe (7-9x for one new signature) as
well as per apply.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENVIRONMENTS_FORMAT = "arrayscope-operation-environments"
ENVIRONMENTS_VERSION = 1
ENVIRONMENTS_FILE = "environments.json"


@dataclass(frozen=True)
class ExecutionEnvironment:
    id: str
    name: str
    interpreter: str = ""
    conda_env: str = ""
    venv_path: str = ""
    working_directory: str = ""
    variables: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExecutionEnvironment:
        environment_id = str(payload.get("id") or "").strip()
        if not environment_id:
            raise ValueError("environment is missing an id")
        variables = payload.get("variables") or {}
        if not isinstance(variables, dict):
            raise ValueError(f"environment {environment_id!r} variables must be an object")
        locators = [
            bool(str(payload.get(field) or "").strip())
            for field in ("interpreter", "conda_env", "venv_path")
        ]
        if sum(locators) > 1:
            raise ValueError(
                f"environment {environment_id!r} must use only one of interpreter, conda_env, venv_path"
            )
        return cls(
            id=environment_id,
            name=str(payload.get("name") or environment_id),
            interpreter=str(payload.get("interpreter") or "").strip(),
            conda_env=str(payload.get("conda_env") or "").strip(),
            venv_path=str(payload.get("venv_path") or "").strip(),
            working_directory=str(payload.get("working_directory") or "").strip(),
            variables=tuple(sorted((str(key), str(value)) for key, value in variables.items())),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "interpreter": self.interpreter,
            "conda_env": self.conda_env,
            "venv_path": self.venv_path,
            "working_directory": self.working_directory,
            "variables": dict(self.variables),
        }


@dataclass(frozen=True)
class ResolvedEnvironment:
    record: ExecutionEnvironment
    env: dict[str, str]
    cwd: str | None
    command_prefix: tuple[str, ...] = ()
    interpreter: str | None = None


def environments_path(directory: str | os.PathLike[str]) -> Path:
    return Path(directory) / ENVIRONMENTS_FILE


def load_environments(directory: str | os.PathLike[str]) -> tuple[ExecutionEnvironment, ...]:
    path = environments_path(directory)
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("format") != ENVIRONMENTS_FORMAT:
        raise ValueError(f"unrecognized execution-environments file: {path}")
    if payload.get("version") != ENVIRONMENTS_VERSION:
        raise ValueError(f"unsupported execution-environments version: {payload.get('version')!r}")
    raw_records = payload.get("environments")
    if not isinstance(raw_records, list):
        raise ValueError("execution-environments file is missing an environments list")
    records = tuple(ExecutionEnvironment.from_payload(item) for item in raw_records)
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("execution-environment ids must be unique")
    return records


def save_environments(
    directory: str | os.PathLike[str],
    records: tuple[ExecutionEnvironment, ...] | list[ExecutionEnvironment],
) -> None:
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": ENVIRONMENTS_FORMAT,
        "version": ENVIRONMENTS_VERSION,
        "environments": [record.to_payload() for record in records],
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory_path, delete=False
    ) as temporary:
        json.dump(payload, temporary, indent=2)
        temporary.write("\n")
        temporary_path = temporary.name
    os.replace(temporary_path, environments_path(directory_path))


def environment_by_id(
    directory: str | os.PathLike[str], environment_id: str
) -> ExecutionEnvironment | None:
    wanted = str(environment_id)
    return next((record for record in load_environments(directory) if record.id == wanted), None)


def upsert_environment(
    directory: str | os.PathLike[str], payload: ExecutionEnvironment | dict[str, Any]
) -> ExecutionEnvironment:
    record = (
        payload
        if isinstance(payload, ExecutionEnvironment)
        else ExecutionEnvironment.from_payload(payload)
    )
    records = list(load_environments(directory))
    for index, existing in enumerate(records):
        if existing.id == record.id:
            records[index] = record
            break
    else:
        records.append(record)
    save_environments(directory, records)
    return record


def remove_environment(directory: str | os.PathLike[str], environment_id: str) -> bool:
    records = list(load_environments(directory))
    retained = [record for record in records if record.id != str(environment_id)]
    if len(retained) == len(records):
        return False
    save_environments(directory, retained)
    return True


def _expanded_directory(value: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(value)))


def _interpreter_in(prefix: str) -> str:
    return os.path.join(
        prefix,
        "Scripts" if os.name == "nt" else "bin",
        "python.exe" if os.name == "nt" else "python",
    )


# Prefix directories keyed by (conda executable, requested name). Only *found*
# prefixes are cached: a miss stays uncached so an environment created after the
# first probe still resolves without restarting ArrayScope.
_CONDA_PREFIX_CACHE: dict[tuple[str, str], str] = {}

# Installation directories that hold an ``envs`` folder on a default install.
_CONDA_INSTALLATIONS: tuple[str, ...] = (
    "miniforge3",
    "mambaforge",
    "miniconda3",
    "anaconda3",
    ".conda",
)


def _conda_prefix_candidates(name_or_path: str, env: dict[str, str]) -> tuple[str, ...]:
    """Conventional prefix directories for a named env, cheapest checks first."""

    if os.path.sep in name_or_path or (os.path.altsep and os.path.altsep in name_or_path):
        return (_expanded_directory(name_or_path),)
    roots: list[str] = [
        _expanded_directory(raw)
        for raw in str(env.get("CONDA_ENVS_PATH") or "").split(os.pathsep)
        if raw.strip()
    ]
    active = str(env.get("CONDA_PREFIX") or "").strip()
    if active:
        # An active env normally lives at ``<installation>/envs/<name>``; its
        # siblings are the other environments of the same installation.
        roots.append(os.path.dirname(_expanded_directory(active)))
    home = os.path.expanduser("~")
    roots.extend(os.path.join(home, installation, "envs") for installation in _CONDA_INSTALLATIONS)
    seen: set[str] = set()
    candidates: list[str] = []
    for root in roots:
        candidate = os.path.join(root, name_or_path)
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return tuple(candidates)


def _conda_environment_paths(conda: str) -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            [conda, "env", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        payload = json.loads(completed.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return ()
    return tuple(str(path) for path in payload.get("envs", ()))


def _conda_environment_prefix(conda: str, name_or_path: str, env: dict[str, str]) -> str | None:
    """The prefix directory for a conda environment, or ``None`` when it is gone.

    Conventional locations are stat-ed first so the common case never spawns
    conda at all; only an environment in an unusual envs directory falls through
    to the bounded ``conda env list --json`` query.
    """

    key = (str(conda), str(name_or_path))
    cached = _CONDA_PREFIX_CACHE.get(key)
    if cached is not None:
        if os.path.isdir(cached):
            return cached
        _CONDA_PREFIX_CACHE.pop(key, None)
    for candidate in _conda_prefix_candidates(name_or_path, env):
        if os.path.isfile(_interpreter_in(candidate)):
            _CONDA_PREFIX_CACHE[key] = candidate
            return candidate
    expanded = _expanded_directory(name_or_path)
    for path in _conda_environment_paths(conda):
        if path == expanded or Path(path).name == name_or_path:
            _CONDA_PREFIX_CACHE[key] = path
            return path
    return None


def resolve_environment(
    directory: str | os.PathLike[str], environment_id: str
) -> tuple[ResolvedEnvironment | None, str | None]:
    """Resolve a record into child-process settings, returning a visible reason."""

    try:
        record = environment_by_id(directory, environment_id)
    except (OSError, ValueError) as exc:
        return None, f"Execution environments could not be read: {exc}"
    if record is None:
        return None, f"Execution environment {environment_id!r} is not configured."

    child_env = dict(os.environ)
    child_env.update(dict(record.variables))
    cwd = None
    if record.working_directory:
        cwd = _expanded_directory(record.working_directory)
        if not os.path.isdir(cwd):
            return None, f"Environment {record.name!r} working directory does not exist: {cwd}"

    if record.interpreter:
        interpreter = _expanded_directory(record.interpreter)
        if not os.path.isfile(interpreter) or not os.access(interpreter, os.X_OK):
            return None, f"Environment {record.name!r} interpreter is not executable: {interpreter}"
        return ResolvedEnvironment(record, child_env, cwd, interpreter=interpreter), None

    if record.venv_path:
        venv = _expanded_directory(record.venv_path)
        interpreter = _interpreter_in(venv)
        if not os.path.isfile(interpreter) or not os.access(interpreter, os.X_OK):
            return (
                None,
                f"Environment {record.name!r} virtualenv has no executable Python: {interpreter}",
            )
        bin_dir = os.path.dirname(interpreter)
        child_env["PATH"] = os.pathsep.join(
            part for part in (bin_dir, child_env.get("PATH", "")) if part
        )
        return ResolvedEnvironment(record, child_env, cwd, interpreter=interpreter), None

    if record.conda_env:
        conda = shutil.which("conda", path=child_env.get("PATH"))
        if conda is None:
            return None, f"Environment {record.name!r} requires conda, but conda is not on PATH."
        prefix = _conda_environment_prefix(conda, record.conda_env, child_env)
        if prefix is None:
            return None, f"Conda environment {record.conda_env!r} no longer exists."
        interpreter = _interpreter_in(prefix)
        if os.path.isfile(interpreter) and os.access(interpreter, os.X_OK):
            # Launch the environment's own interpreter instead of paying conda's
            # start-up per call. Prepending the prefix's bin directory and
            # exporting CONDA_PREFIX reproduces what activation gives a child
            # that only needs to find the environment's executables.
            bin_dir = os.path.dirname(interpreter)
            child_env["PATH"] = os.pathsep.join(
                part for part in (bin_dir, child_env.get("PATH", "")) if part
            )
            child_env["CONDA_PREFIX"] = prefix
            child_env["CONDA_DEFAULT_ENV"] = Path(prefix).name
            return ResolvedEnvironment(record, child_env, cwd, interpreter=interpreter), None
        # No interpreter in the prefix (a tools-only environment, or a layout we
        # do not recognize): keep the incumbent ``conda run`` path.
        command_prefix = (conda, "run", "--no-capture-output", "-n", record.conda_env)
        return (
            ResolvedEnvironment(record, child_env, cwd, command_prefix, interpreter="python"),
            None,
        )

    return ResolvedEnvironment(record, child_env, cwd), None
