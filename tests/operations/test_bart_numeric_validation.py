"""Real-toolbox numeric gate for the exposed BART operation definitions."""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools import validate_bart_numerics


def test_harness_reports_a_missing_toolbox(tmp_path, capsys):
    exit_code = validate_bart_numerics.main(
        ["--bart-toolbox-path", str(tmp_path / "missing-toolbox")]
    )

    assert exit_code == validate_bart_numerics.EXIT_UNAVAILABLE
    assert "BART numeric validation: UNAVAILABLE" in capsys.readouterr().err


def test_numeric_oracles_fail_closed_on_corrupt_outputs():
    case = validate_bart_numerics.synthetic_case()
    outputs = validate_bart_numerics.OperationOutput(
        ecalib=np.zeros((24, 24, 1, 4), dtype=np.complex64),
        walsh=np.zeros((24, 24, 1, 10), dtype=np.complex64),
        pics=np.zeros((24, 24), dtype=np.complex64),
    )

    checks = validate_bart_numerics.validate_outputs(case, outputs)

    assert {check.operation for check in checks} == {
        "bart:ecalib",
        "bart:walsh",
        "bart:pics",
    }
    assert all(not check.passed for check in checks)


def test_real_bart_numeric_validation_harness():
    executable = shutil.which("bart")
    if executable is None:
        pytest.skip(
            "real BART numeric validation requires a BART toolbox "
            "(bart executable not found on PATH)"
        )

    output = io.StringIO()
    report = validate_bart_numerics.run_validation(
        toolbox_path=Path(executable).resolve().parent,
        stream=output,
    )

    assert report.passed, output.getvalue()
