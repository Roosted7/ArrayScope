"""BART examples and compatibility bridge over the general command runtime.

Bundle A removed the NumPy-trivial BART operations.  Bundle C keeps
``run_bart`` as a compatibility API while moving its cfl/process machinery to
``arrayscope.operations.command_runtime`` and exposes only genuinely
BART-shaped reconstruction examples.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from arrayscope.operations import command_runtime

BART_ENVIRONMENT_ID = "bart"
BART_TIMEOUT_ENV = "ARRAYSCOPE_BART_TIMEOUT_S"
_DEFAULT_TIMEOUT_S = command_runtime.DEFAULT_TIMEOUT_S
_POLL_INTERVAL = command_runtime.POLL_INTERVAL_S
_TERM_GRACE = command_runtime.TERM_GRACE_S
_USE_CONFIGURED_TIMEOUT: object = object()

write_cfl = command_runtime.write_cfl
read_cfl = command_runtime.read_cfl


def bart_executable(env: Mapping[str, object] | None = None) -> str | None:
    """Resolve ``bart`` from the effective PATH without running it.

    ``BART_TOOLBOX_PATH`` is intentionally not interpreted here.  Toolbox paths
    and library variables now belong to a named execution-environment record.
    """

    search_path = None
    if env is not None:
        search_path = str(env.get("PATH") or "") or None
    return shutil.which("bart", path=search_path)


def bart_available(env: Mapping[str, object] | None = None) -> bool:
    return bart_executable(env) is not None


@dataclass(frozen=True)
class _BartRuntime:
    executable: str
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    prefix: tuple[str, ...] = ()


def _pack_runtime() -> tuple[_BartRuntime | None, str | None]:
    """Resolve the named ``bart`` record, falling back to the process PATH."""

    # Import lazily because library imports the pack registry.
    from arrayscope.operations.library import resolve_execution_environment

    resolved, reason = resolve_execution_environment(BART_ENVIRONMENT_ID)
    if reason:
        if "is not configured" not in reason:
            return None, reason
    elif resolved is not None:
        prefix = tuple(resolved.command_prefix)
        if prefix:
            # The environment resolver has already proved that the conda
            # environment exists. ``conda run`` owns lookup inside it.
            return _BartRuntime("bart", resolved.env, resolved.cwd, prefix), None
        executable = bart_executable(resolved.env)
        if executable is None:
            return (
                None,
                "BART is unavailable because the bart executable was not found "
                f"in execution environment {BART_ENVIRONMENT_ID!r}.",
            )
        return _BartRuntime(executable, resolved.env, resolved.cwd), None

    executable = bart_executable()
    if executable is not None:
        return _BartRuntime(executable), None
    return None, "BART is unavailable because the bart executable was not found."


def bart_timeout(default: float = _DEFAULT_TIMEOUT_S) -> float | None:
    raw = os.environ.get(BART_TIMEOUT_ENV)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else None


def bart_admission_notes() -> tuple[str, ...]:
    return (
        "BART op: out-of-process subprocess + cfl temp-file round-trip (expensive).",
        "OPAQUE whole-array: never run per-region (a per-tile subprocess is never the right plan).",
    )


def bart_env(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    if overrides:
        environment.update({str(key): str(value) for key, value in overrides.items()})
    return environment


def run_bart(
    argv: Sequence[str],
    array: np.ndarray,
    *,
    inputs: Mapping[str, np.ndarray] | None = None,
    cancellation_token: object | None = None,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
    poll_interval: float = _POLL_INTERVAL,
    grace: float = _TERM_GRACE,
    timeout: float | None = _USE_CONFIGURED_TIMEOUT,  # type: ignore[assignment]
    temp_dir: str | None = None,
    cwd: str | None = None,
    prefix: Sequence[str] = (),
) -> np.ndarray:
    """Run ``bart <argv> in out`` through the shared cfl command runtime."""

    child_env = bart_env(env)
    binary = executable or bart_executable(child_env)
    if binary is None:
        raise RuntimeError("bart executable not found in the selected environment")
    if timeout is _USE_CONFIGURED_TIMEOUT:
        timeout = bart_timeout()
    slot_inputs = dict(inputs or {})
    return command_runtime.run_array_command(
        [
            *[str(token) for token in prefix],
            binary,
            *[str(token) for token in argv],
            "{in}",
            *[f"{{slot:{name}}}" for name in slot_inputs],
            "{out}",
        ],
        array,
        inputs=slot_inputs,
        handoff="cfl",
        cancellation_token=cancellation_token,
        env=child_env,
        timeout=timeout,
        poll_interval=poll_interval,
        grace=grace,
        temp_dir=temp_dir,
        cwd=cwd,
    )


def pack_specs() -> tuple:
    """Readable BART-native examples; shape discovery must unlock execution."""

    from arrayscope.operations.input_slots import (
        SLOT_DIMENSION_SET,
        SLOT_OPEN_DOCUMENT,
        SLOT_SAVED_ARRAY,
        OperationInputSlot,
    )
    from arrayscope.operations.plugins import PluginOperationSpec
    from arrayscope.operations.registry import OperationParameter

    def availability():
        _runtime, reason = _pack_runtime()
        return reason

    def invoke(argv, data, *, inputs=None):
        runtime, reason = _pack_runtime()
        if reason or runtime is None:
            raise RuntimeError(reason or "BART execution environment is unavailable.")
        return run_bart(
            argv,
            data,
            inputs=inputs,
            env=runtime.env,
            executable=runtime.executable,
            cwd=runtime.cwd,
            prefix=runtime.prefix,
        )

    def build(argv):
        def factory(_axis, params, _slots):
            tokens = [str(token).format_map(dict(params)) for token in argv]
            return lambda data: invoke(tokens, data)

        return factory

    def build_pics(_axis, params, slots):
        tokens = ["pics", "-S", "-i", str(int(params["iterations"]))]
        return lambda data: invoke(tokens, data, inputs=slots)

    return (
        PluginOperationSpec(
            id="bart:pics",
            label="BART PICS reconstruction",
            build=build_pics,
            parameters=(
                OperationParameter(
                    "iterations",
                    "Iterations",
                    kind="int",
                    default=30,
                    minimum=1,
                    maximum=10_000,
                    description="Maximum PICS solver iterations.",
                ),
            ),
            input_slots=(
                OperationInputSlot(
                    "sensitivities",
                    "Sensitivity maps",
                    "Coil sensitivity maps paired with the primary k-space array.",
                    accepts=(
                        SLOT_DIMENSION_SET,
                        SLOT_OPEN_DOCUMENT,
                        SLOT_SAVED_ARRAY,
                    ),
                ),
            ),
            group="BART",
            description="Iterative parallel-imaging reconstruction from k-space and coil maps.",
            icon="hub",
            availability=availability,
            runtime="command",
            runtime_config={
                "command_template": ("bart pics -S -i {iterations} {in} {sensitivities} {out}"),
                "handoff": "cfl",
                "timeout_s": _DEFAULT_TIMEOUT_S,
                "shell": False,
            },
            environment_id=BART_ENVIRONMENT_ID,
        ),
        PluginOperationSpec(
            id="bart:ecalib",
            label="BART ESPIRiT calibration",
            build=build(("ecalib", "-m", "{maps}")),
            parameters=(
                OperationParameter(
                    "maps",
                    "Maps",
                    kind="int",
                    default=1,
                    minimum=1,
                    description="Number of sensitivity-map sets.",
                ),
            ),
            group="BART",
            description="Estimate ESPIRiT sensitivity maps from calibration data.",
            icon="hub",
            availability=availability,
            runtime="command",
            runtime_config={
                "command_template": "bart ecalib -m {maps} {in} {out}",
                "handoff": "cfl",
                "timeout_s": _DEFAULT_TIMEOUT_S,
                "shell": False,
            },
            environment_id=BART_ENVIRONMENT_ID,
        ),
        PluginOperationSpec(
            id="bart:walsh",
            label="BART Walsh calibration covariance",
            build=build(("walsh", "-r", "{calibration_size}")),
            parameters=(
                OperationParameter(
                    "calibration_size",
                    "Calibration size",
                    kind="int",
                    default=5,
                    minimum=1,
                    description="Centered k-space calibration-region size.",
                ),
            ),
            group="BART",
            description=("Estimate packed Hermitian coil-covariance matrices for BART ecaltwo."),
            icon="blur_on",
            availability=availability,
            runtime="command",
            runtime_config={
                "command_template": ("bart walsh -r {calibration_size} {in} {out}"),
                "handoff": "cfl",
                "timeout_s": _DEFAULT_TIMEOUT_S,
                "shell": False,
            },
            environment_id=BART_ENVIRONMENT_ID,
        ),
    )


def register(register_fn=None) -> bool:
    if register_fn is None:
        from arrayscope.operations.registry import register_pack_operation

        register_fn = register_pack_operation
    registered = False
    for spec in pack_specs():
        register_fn(spec)
        registered = True
    return registered
