"""Registry for ArrayScope dimension operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from arrayscope.operations.pipeline import (
    CenteredFFT,
    CenteredIFFT,
    CombineRealImagAxis,
    Conjugate,
    Crop,
    FFTShift,
    Maximum,
    Mean,
    Minimum,
    ReverseAxis,
    RootSumSquares,
    SplitComplexAxis,
    Sum,
)


@dataclass(frozen=True)
class OperationParameter:
    name: str
    label: str
    kind: str = "int"


@dataclass(frozen=True)
class OperationEntry:
    id: str
    label: str
    operation_type: type
    parameters: Tuple[OperationParameter, ...] = ()
    changes_shape: bool = False
    requires_axis: bool = True


OPERATION_REGISTRY = {
    "crop": OperationEntry(
        id="crop",
        label="Crop axis...",
        operation_type=Crop,
        parameters=(
            OperationParameter("start", "Start"),
            OperationParameter("stop", "Stop"),
        ),
        changes_shape=True,
    ),
    "reverse": OperationEntry(
        id="reverse",
        label="Reverse / flip axis",
        operation_type=ReverseAxis,
    ),
    "conjugate": OperationEntry(
        id="conjugate",
        label="Conjugate",
        operation_type=Conjugate,
        requires_axis=False,
    ),
    "mean": OperationEntry(
        id="mean",
        label="Mean over axis",
        operation_type=Mean,
        changes_shape=True,
    ),
    "rss": OperationEntry(
        id="rss",
        label="Root-sum-squares over axis",
        operation_type=RootSumSquares,
        changes_shape=True,
    ),
    "sum": OperationEntry(
        id="sum",
        label="Sum over axis",
        operation_type=Sum,
        changes_shape=True,
    ),
    "max": OperationEntry(
        id="max",
        label="Maximum over axis",
        operation_type=Maximum,
        changes_shape=True,
    ),
    "min": OperationEntry(
        id="min",
        label="Minimum over axis",
        operation_type=Minimum,
        changes_shape=True,
    ),
    "centered_fft": OperationEntry(
        id="centered_fft",
        label="Centered FFT",
        operation_type=CenteredFFT,
    ),
    "centered_ifft": OperationEntry(
        id="centered_ifft",
        label="Centered iFFT",
        operation_type=CenteredIFFT,
    ),
    "fftshift": OperationEntry(
        id="fftshift",
        label="FFT shift",
        operation_type=FFTShift,
    ),
    "combine_real_imag": OperationEntry(
        id="combine_real_imag",
        label="Combine real/imag axis",
        operation_type=CombineRealImagAxis,
        changes_shape=True,
    ),
    "split_complex": OperationEntry(
        id="split_complex",
        label="Split complex axis",
        operation_type=SplitComplexAxis,
        changes_shape=True,
    ),
}


def operation_entries():
    return tuple(OPERATION_REGISTRY.values())


def get_operation_entry(operation_id: str) -> OperationEntry:
    try:
        return OPERATION_REGISTRY[operation_id]
    except KeyError as exc:
        raise ValueError(f"unknown operation id: {operation_id}") from exc


def create_operation(operation_id: str, axis=None, parameters: Mapping[str, object] | None = None):
    entry = get_operation_entry(operation_id)
    parameters = dict(parameters or {})
    kwargs = {}

    if entry.requires_axis:
        if axis is None:
            raise ValueError(f"operation {operation_id} requires an axis")
        kwargs["axis"] = int(axis)

    for parameter in entry.parameters:
        if parameter.name not in parameters:
            raise ValueError(f"operation {operation_id} requires parameter {parameter.name}")
        value = parameters[parameter.name]
        if parameter.kind == "int":
            value = int(value)
        kwargs[parameter.name] = value

    return entry.operation_type(**kwargs)


def operation_id_for(operation) -> str:
    operation_type = type(operation)
    for entry in OPERATION_REGISTRY.values():
        if entry.operation_type is operation_type:
            return entry.id
    operation_module = getattr(operation_type, "__module__", "")
    operation_name = getattr(operation_type, "__name__", "")
    for entry in OPERATION_REGISTRY.values():
        entry_type = entry.operation_type
        if getattr(entry_type, "__module__", "") == operation_module and getattr(entry_type, "__name__", "") == operation_name:
            return entry.id
    raise ValueError(f"operation type is not registered: {operation_type.__name__}")


def describe_operation(operation) -> str:
    operation_id = operation_id_for(operation)
    entry = get_operation_entry(operation_id)
    parts = [entry.label.rstrip(".")]
    if entry.requires_axis:
        parts.append(f"axis {getattr(operation, 'axis')}")
    for parameter in entry.parameters:
        parts.append(f"{parameter.name}={getattr(operation, parameter.name)}")
    return " ".join(parts)
