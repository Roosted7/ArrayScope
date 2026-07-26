"""Named execution-environment records for user operations.

Records live in ``operations/environments.json`` beside operation wrappers.
Resolution is lazy and side-effect free apart from the bounded ``conda env
list`` query needed to distinguish a real named env from a vanished one.
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


def _conda_environment_exists(conda: str, name_or_path: str) -> bool:
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
        return False
    paths = tuple(str(path) for path in payload.get("envs", ()))
    expanded = _expanded_directory(name_or_path)
    return expanded in paths or any(Path(path).name == name_or_path for path in paths)


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
        interpreter = os.path.join(
            venv,
            "Scripts" if os.name == "nt" else "bin",
            "python.exe" if os.name == "nt" else "python",
        )
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
        if not _conda_environment_exists(conda, record.conda_env):
            return None, f"Conda environment {record.conda_env!r} no longer exists."
        prefix = (conda, "run", "--no-capture-output", "-n", record.conda_env)
        return ResolvedEnvironment(record, child_env, cwd, prefix, interpreter="python"), None

    return ResolvedEnvironment(record, child_env, cwd), None
