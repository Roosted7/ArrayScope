"""Validate ArrayScope's exposed BART operations against a real toolbox.

The fake-BART unit tests own argv, cancellation, timeout, pipe draining, and
scratch cleanup. This harness owns the missing numeric evidence. It executes
the registered ``bart:ecalib``, ``bart:walsh``, and ``bart:pics`` definitions
on one deterministic, fully sampled four-coil problem and compares their
outputs with independent NumPy references or mathematically required
invariants.

Example:

    conda run -n arrayscope python tools/validate_bart_numerics.py \
        --bart-toolbox-path ~/projects/bart

Use ``--library-path`` when a locally built BART needs runtime libraries that
are not already visible to the dynamic loader.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

# ``python tools/validate_bart_numerics.py`` otherwise searches an editable
# install before this checkout. Numeric evidence must name the code it ran.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arrayscope.operations.packs import bart_pack

GRID_SIZE = 24
COIL_COUNT = 4
PICS_ITERATIONS = 30
ECALIB_NORM_ERROR_LIMIT = 2e-5
ECALIB_SUBSPACE_CORRELATION_LIMIT = 0.999
PICS_RELATIVE_L2_LIMIT = 1e-5
PICS_MAX_ABS_LIMIT = 5e-5
WALSH_RELATIVE_PSD_FLOOR = -1e-6
WALSH_DOMINANT_FRACTION_LIMIT = 0.999
WALSH_SUBSPACE_CORRELATION_LIMIT = 0.999
EXIT_UNAVAILABLE = 2


@dataclass(frozen=True)
class SyntheticCase:
    image: np.ndarray
    sensitivities: np.ndarray
    kspace: np.ndarray
    support: np.ndarray
    coil_vector: np.ndarray


@dataclass(frozen=True)
class OperationOutput:
    ecalib: np.ndarray
    walsh: np.ndarray
    pics: np.ndarray


@dataclass(frozen=True)
class CheckResult:
    operation: str
    oracle: str
    observed: str
    tolerance: str
    passed: bool


@dataclass(frozen=True)
class ValidationReport:
    executable: str
    version: str
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)


class BartUnavailable(RuntimeError):
    """The requested real toolbox cannot be found or started."""


def _centered_fft2(array: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(array, axes=(0, 1))
    transformed = np.fft.fftn(shifted, axes=(0, 1), norm="ortho")
    return np.fft.fftshift(transformed, axes=(0, 1)).astype(np.complex64)


def _centered_ifft2(array: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(array, axes=(0, 1))
    transformed = np.fft.ifftn(shifted, axes=(0, 1), norm="ortho")
    return np.fft.fftshift(transformed, axes=(0, 1)).astype(np.complex64)


def synthetic_case() -> SyntheticCase:
    """A small rank-one coil problem with an analytic SENSE inverse."""

    coordinate = (np.arange(GRID_SIZE, dtype=np.float32) - GRID_SIZE / 2) / (GRID_SIZE / 2)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")
    image = np.exp(-3.0 * (xx**2 + yy**2))
    image += 0.45 * np.exp(-22.0 * ((xx - 0.35) ** 2 + (yy + 0.22) ** 2))
    image = image.astype(np.complex64)

    coil_vector = np.asarray([1.0, 0.8j, -0.6, 0.5 - 0.2j], dtype=np.complex64)
    coil_vector /= np.linalg.norm(coil_vector)
    sensitivities = np.broadcast_to(coil_vector, (GRID_SIZE, GRID_SIZE, 1, COIL_COUNT)).copy()
    coil_images = image[:, :, None, None] * sensitivities
    kspace = _centered_fft2(coil_images)
    support = np.abs(image) > 0.08
    return SyntheticCase(image, sensitivities, kspace, support, coil_vector)


def _registered_specs():
    return {spec.id: spec for spec in bart_pack.pack_specs()}


def execute_operations(case: SyntheticCase) -> OperationOutput:
    """Execute the canonical registered definitions, not copied command lines."""

    specs = _registered_specs()
    ecalib = specs["bart:ecalib"].resolve_fn(None, {"maps": 1})(case.kspace)
    walsh = specs["bart:walsh"].resolve_fn(None, {"calibration_size": 5})(case.kspace)
    pics = specs["bart:pics"].resolve_fn(
        None,
        {"iterations": PICS_ITERATIONS},
        {"sensitivities": case.sensitivities},
    )(case.kspace)
    return OperationOutput(ecalib=ecalib, walsh=walsh, pics=pics)


def _shape_is(array: np.ndarray, expected: tuple[int, ...]) -> bool:
    return tuple(np.asarray(array).shape) == expected


def _ecalib_check(case: SyntheticCase, output: np.ndarray) -> CheckResult:
    expected_shape = (GRID_SIZE, GRID_SIZE, 1, COIL_COUNT)
    if not _shape_is(output, expected_shape):
        return CheckResult(
            "bart:ecalib",
            "unit coil norm and known rank-one coil subspace",
            f"shape={tuple(np.asarray(output).shape)}",
            f"shape={expected_shape}",
            False,
        )
    value = np.asarray(output)
    rss = np.sqrt(np.sum(np.abs(value) ** 2, axis=3))
    foreground_rss = rss[:, :, 0][case.support]
    norm_error = float(np.max(np.abs(foreground_rss - 1.0)))
    normalized = value / np.where(rss[..., None] > 0, rss[..., None], 1.0)
    correlation = np.abs(
        np.sum(
            np.conj(case.coil_vector) * normalized[:, :, 0, :],
            axis=-1,
        )
    )
    minimum_correlation = float(np.min(correlation[case.support]))
    finite = bool(np.all(np.isfinite(value)))
    passed = (
        finite
        and norm_error <= ECALIB_NORM_ERROR_LIMIT
        and minimum_correlation >= ECALIB_SUBSPACE_CORRELATION_LIMIT
    )
    return CheckResult(
        "bart:ecalib",
        "unit coil norm and known rank-one coil subspace",
        (
            f"max_norm_error={norm_error:.3e}; "
            f"min_correlation={minimum_correlation:.9f}; finite={finite}"
        ),
        (
            f"norm_error<={ECALIB_NORM_ERROR_LIMIT:.1e}; "
            f"correlation>={ECALIB_SUBSPACE_CORRELATION_LIMIT:.3f}"
        ),
        passed,
    )


def _unpack_walsh_covariance(output: np.ndarray) -> np.ndarray:
    packed = np.asarray(output)
    expected_shape = (
        GRID_SIZE,
        GRID_SIZE,
        1,
        COIL_COUNT * (COIL_COUNT + 1) // 2,
    )
    if not _shape_is(packed, expected_shape):
        raise ValueError(f"shape={packed.shape}, expected={expected_shape}")
    matrices = np.zeros((GRID_SIZE, GRID_SIZE, COIL_COUNT, COIL_COUNT), dtype=np.complex64)
    packed_index = 0
    for row in range(COIL_COUNT):
        for column in range(row + 1):
            item = packed[:, :, 0, packed_index]
            matrices[:, :, row, column] = item
            matrices[:, :, column, row] = np.conj(item)
            packed_index += 1
    return matrices


def _walsh_check(case: SyntheticCase, output: np.ndarray) -> CheckResult:
    try:
        matrices = _unpack_walsh_covariance(output)
    except ValueError as exc:
        return CheckResult(
            "bart:walsh",
            "packed Hermitian PSD covariance and dominant coil subspace",
            str(exc),
            f"packed coil axis={COIL_COUNT * (COIL_COUNT + 1) // 2}",
            False,
        )
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    maximum = eigenvalues[..., -1]
    minimum = eigenvalues[..., 0]
    positive_sum = np.sum(np.maximum(eigenvalues, 0.0), axis=-1)
    relative_floor = float(
        np.min(minimum[case.support])
        / max(float(np.max(maximum[case.support])), np.finfo(np.float32).tiny)
    )
    dominant_fraction = float(
        np.min(
            maximum[case.support]
            / np.maximum(positive_sum[case.support], np.finfo(np.float32).tiny)
        )
    )
    # BART packs the covariance convention whose principal vector is the
    # conjugated sensitivity vector. Absolute correlation ignores phase gauge.
    principal = eigenvectors[..., -1]
    expected = np.conj(case.coil_vector)
    correlation = np.abs(np.sum(np.conj(expected) * principal, axis=-1))
    minimum_correlation = float(np.min(correlation[case.support]))
    finite = bool(np.all(np.isfinite(matrices)))
    passed = (
        finite
        and relative_floor >= WALSH_RELATIVE_PSD_FLOOR
        and dominant_fraction >= WALSH_DOMINANT_FRACTION_LIMIT
        and minimum_correlation >= WALSH_SUBSPACE_CORRELATION_LIMIT
    )
    return CheckResult(
        "bart:walsh",
        "packed Hermitian PSD covariance and dominant coil subspace",
        (
            f"relative_lambda_min={relative_floor:.3e}; "
            f"min_dominant_fraction={dominant_fraction:.9f}; "
            f"min_correlation={minimum_correlation:.9f}; finite={finite}"
        ),
        (
            f"lambda_min/lambda_max>={WALSH_RELATIVE_PSD_FLOOR:.1e}; "
            f"dominant_fraction>={WALSH_DOMINANT_FRACTION_LIMIT:.3f}; "
            f"correlation>={WALSH_SUBSPACE_CORRELATION_LIMIT:.3f}"
        ),
        passed,
    )


def _pics_check(case: SyntheticCase, output: np.ndarray) -> CheckResult:
    coil_images = _centered_ifft2(case.kspace)
    reference = np.sum(np.conj(case.sensitivities) * coil_images, axis=3)[:, :, 0]
    value = np.squeeze(np.asarray(output))
    if not _shape_is(value, (GRID_SIZE, GRID_SIZE)):
        return CheckResult(
            "bart:pics",
            "NumPy centered unitary inverse FFT plus direct SENSE combine",
            f"shape={value.shape}",
            f"shape={(GRID_SIZE, GRID_SIZE)}",
            False,
        )
    difference = value - reference
    relative_l2 = float(np.linalg.norm(difference) / np.linalg.norm(reference))
    maximum_error = float(np.max(np.abs(difference)))
    finite = bool(np.all(np.isfinite(value)))
    passed = (
        finite and relative_l2 <= PICS_RELATIVE_L2_LIMIT and maximum_error <= PICS_MAX_ABS_LIMIT
    )
    return CheckResult(
        "bart:pics",
        "NumPy centered unitary inverse FFT plus direct SENSE combine",
        (f"relative_l2={relative_l2:.3e}; max_abs={maximum_error:.3e}; finite={finite}"),
        (f"relative_l2<={PICS_RELATIVE_L2_LIMIT:.1e}; max_abs<={PICS_MAX_ABS_LIMIT:.1e}"),
        passed,
    )


def validate_outputs(case: SyntheticCase, outputs: OperationOutput) -> tuple[CheckResult, ...]:
    """Evaluate the three independent numeric contracts."""

    return (
        _ecalib_check(case, outputs.ecalib),
        _walsh_check(case, outputs.walsh),
        _pics_check(case, outputs.pics),
    )


def _resolve_bart(toolbox_path: Path | None) -> Path:
    if toolbox_path is not None:
        executable = toolbox_path.expanduser().resolve() / "bart"
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise BartUnavailable(f"no executable bart found under {toolbox_path.expanduser()}")
        return executable
    discovered = shutil.which("bart")
    if discovered is None:
        raise BartUnavailable(
            "bart executable not found on PATH; pass --bart-toolbox-path /path/to/bart"
        )
    return Path(discovered).resolve()


def _probe_version(executable: Path) -> str:
    try:
        completed = subprocess.run(
            [str(executable), "version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise BartUnavailable(f"{executable} could not run: {str(detail).strip()}") from exc
    lines = [
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip()
    ]
    return next((line for line in lines if line.startswith("v")), lines[0] if lines else "unknown")


def _print_report(report: ValidationReport, stream: TextIO) -> None:
    print(f"BART executable: {report.executable}", file=stream)
    print(f"BART version:    {report.version}", file=stream)
    print(file=stream)
    rows = [
        (
            check.operation,
            "PASS" if check.passed else "FAIL",
            check.oracle,
            check.observed,
            check.tolerance,
        )
        for check in report.checks
    ]
    headers = ("operation", "status", "reference / invariant", "observed", "acceptance")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)), file=stream
    )
    print("  ".join("-" * width for width in widths), file=stream)
    for row in rows:
        print(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)),
            file=stream,
        )
    print(file=stream)
    print(
        "BART numeric validation: " + ("PASS" if report.passed else "FAIL"),
        file=stream,
    )


def run_validation(
    *,
    toolbox_path: Path | None = None,
    library_paths: tuple[Path, ...] = (),
    stream: TextIO = sys.stdout,
) -> ValidationReport:
    executable = _resolve_bart(toolbox_path)
    prior_path = os.environ.get("PATH")
    prior_library_path = os.environ.get("LD_LIBRARY_PATH")
    try:
        if library_paths:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                [
                    *(str(path.expanduser().resolve()) for path in library_paths),
                    *([prior_library_path] if prior_library_path else []),
                ]
            )
        os.environ["PATH"] = os.pathsep.join([str(executable.parent), prior_path or ""])
        version = _probe_version(executable)
        case = synthetic_case()
        try:
            outputs = execute_operations(case)
            checks = validate_outputs(case, outputs)
        except Exception as exc:
            checks = (
                CheckResult(
                    "BART operations",
                    "all registered definitions complete",
                    f"{type(exc).__name__}: {exc}",
                    "no command/runtime error",
                    False,
                ),
            )
    finally:
        if prior_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = prior_path
        if prior_library_path is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = prior_library_path
    report = ValidationReport(str(executable), version, checks)
    _print_report(report, stream)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bart-toolbox-path",
        type=Path,
        help="Directory containing the real bart executable (defaults to PATH).",
    )
    parser.add_argument(
        "--library-path",
        type=Path,
        action="append",
        default=[],
        help="Prepend a directory to the BART child's LD_LIBRARY_PATH; repeatable.",
    )
    args = parser.parse_args(argv)
    try:
        report = run_validation(
            toolbox_path=args.bart_toolbox_path,
            library_paths=tuple(args.library_path),
        )
    except BartUnavailable as exc:
        print(f"BART numeric validation: UNAVAILABLE: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
