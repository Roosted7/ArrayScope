"""Self-contained child entry point for environment-backed Python operations.

This file is executed directly by the selected interpreter.  Keep it free of
ArrayScope imports: the target environment only needs Python and NumPy.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np


def _read_cfl(stem: str) -> np.ndarray:
    with open(stem + ".hdr", encoding="utf-8") as handle:
        next(handle)
        dims = [int(token) for token in next(handle).split()]
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return np.reshape(
        np.fromfile(stem + ".cfl", dtype=np.complex64)[: int(np.prod(dims))],
        dims,
        order="F",
    )


def _write_cfl(stem: str, array: np.ndarray) -> None:
    value = np.asarray(array, dtype=np.complex64)
    with open(stem + ".hdr", "w", encoding="utf-8") as handle:
        handle.write("# Dimensions\n")
        handle.write(" ".join(str(int(size)) for size in value.shape) + "\n")
    np.ascontiguousarray(value.T).tofile(stem + ".cfl")


def _load_array(path: str, handoff: str) -> np.ndarray:
    return np.load(path, allow_pickle=False) if handoff == "npy" else _read_cfl(path)


def _save_array(path: str, handoff: str, value) -> None:
    if handoff == "npy":
        np.save(path, np.asarray(value), allow_pickle=False)
    else:
        _write_cfl(path, np.asarray(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--callable", required=True)
    parser.add_argument("--handoff", choices=("npy", "cfl"), required=True)
    parser.add_argument("--parameters", default="{}")
    parser.add_argument("--axis", default="null")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    source = str(Path(args.source).resolve())
    spec = importlib.util.spec_from_file_location("arrayscope_external_user_operation", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import operation source {source!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, args.callable)
    signature = inspect.signature(function)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    accepted = set(signature.parameters)
    parameters = json.loads(args.parameters)
    kwargs = {key: value for key, value in parameters.items() if accepts_kwargs or key in accepted}
    axis = json.loads(args.axis)
    if axis is not None and (accepts_kwargs or "axis" in accepted):
        kwargs["axis"] = axis
    result = function(_load_array(args.input, args.handoff), **kwargs)
    _save_array(args.output, args.handoff, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
