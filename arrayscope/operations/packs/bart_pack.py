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

import numpy as np

from arrayscope.operations import command_runtime

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
    cancellation_token: object | None = None,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
    poll_interval: float = _POLL_INTERVAL,
    grace: float = _TERM_GRACE,
    timeout: float | None = _USE_CONFIGURED_TIMEOUT,  # type: ignore[assignment]
    temp_dir: str | None = None,
) -> np.ndarray:
    """Run ``bart <argv> in out`` through the shared cfl command runtime."""

    child_env = bart_env(env)
    binary = executable or bart_executable(child_env)
    if binary is None:
        raise RuntimeError("bart executable not found in the selected environment")
    if timeout is _USE_CONFIGURED_TIMEOUT:
        timeout = bart_timeout()
    return command_runtime.run_array_command(
        [binary, *[str(token) for token in argv], "{in}", "{out}"],
        array,
        handoff="cfl",
        cancellation_token=cancellation_token,
        env=child_env,
        timeout=timeout,
        poll_interval=poll_interval,
        grace=grace,
        temp_dir=temp_dir,
    )


def pack_specs() -> tuple:
    """Readable BART-native examples; shape discovery must unlock execution."""

    from arrayscope.operations.plugins import PluginOperationSpec
    from arrayscope.operations.registry import OperationParameter

    blocked = (
        "BART reconstruction example — duplicate it to edit. "
        "Execution remains unavailable until Bundle D discovers its output shape."
    )

    def build(argv):
        def factory(_axis, params):
            tokens = [str(token).format_map(dict(params)) for token in argv]
            return lambda data: run_bart(tokens, data)

        return factory

    return (
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
            unavailable_reason=blocked,
            runtime="command",
            runtime_config={
                "command_template": "bart ecalib -m {maps} {in} {out}",
                "handoff": "cfl",
                "timeout_s": _DEFAULT_TIMEOUT_S,
                "shell": False,
            },
            environment_id="bart",
        ),
        PluginOperationSpec(
            id="bart:walsh",
            label="BART Walsh sensitivity maps",
            build=build(("walsh", "-r", "{radius}")),
            parameters=(
                OperationParameter(
                    "radius",
                    "Radius",
                    kind="int",
                    default=5,
                    minimum=1,
                    description="Local covariance smoothing radius.",
                ),
            ),
            group="BART",
            description="Estimate coil sensitivities with BART's Walsh method.",
            icon="blur_on",
            unavailable_reason=blocked,
            runtime="command",
            runtime_config={
                "command_template": "bart walsh -r {radius} {in} {out}",
                "handoff": "cfl",
                "timeout_s": _DEFAULT_TIMEOUT_S,
                "shell": False,
            },
            environment_id="bart",
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
