"""Qt-free input-slot definitions, bindings, and resolved source snapshots.

The primary operation chain remains linear.  A slot binding names an auxiliary
source, while :class:`ResolvedSlot` carries the immutable source snapshot used
by worker-side execution and characterization.  Bindings are recipe data;
resolved slots are process-local and are never serialized.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SLOT_DIMENSION_SET = "dimension-set"
SLOT_OPEN_DOCUMENT = "open-document"
SLOT_ROI_MASK = "roi-mask"
SLOT_ROI_COORDINATES = "roi-coordinates"
SLOT_SAVED_ARRAY = "saved-array"

SLOT_SOURCE_KINDS = frozenset(
    {
        SLOT_DIMENSION_SET,
        SLOT_OPEN_DOCUMENT,
        SLOT_ROI_MASK,
        SLOT_ROI_COORDINATES,
        SLOT_SAVED_ARRAY,
    }
)
_SLOT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class OperationInputSlot:
    """One declared auxiliary input beyond the primary array."""

    name: str
    label: str
    description: str = ""
    accepts: tuple[str, ...] = (
        SLOT_DIMENSION_SET,
        SLOT_OPEN_DOCUMENT,
        SLOT_SAVED_ARRAY,
    )

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("input slot name cannot be empty")
        if not _SLOT_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"input slot name {name!r} must be a Python-style identifier")
        accepts = tuple(dict.fromkeys(str(kind) for kind in self.accepts))
        unknown = sorted(set(accepts) - SLOT_SOURCE_KINDS)
        if unknown:
            raise ValueError(f"input slot {name!r} has unknown source kinds: {unknown}")
        if not accepts:
            raise ValueError(f"input slot {name!r} must accept at least one source kind")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "label", str(self.label or name.replace("_", " ").title()))
        object.__setattr__(self, "description", str(self.description or ""))
        object.__setattr__(self, "accepts", accepts)


@dataclass(frozen=True)
class SlotBinding:
    """Serializable reference to one auxiliary source."""

    kind: str
    source_id: str = ""
    path: str = ""
    indices: tuple[int | None, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        kind = str(self.kind or "")
        if kind and kind not in SLOT_SOURCE_KINDS:
            raise ValueError(f"unknown input-slot binding kind: {kind!r}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_id", str(self.source_id or ""))
        object.__setattr__(self, "path", str(self.path or ""))
        object.__setattr__(
            self,
            "indices",
            tuple(None if value is None else int(value) for value in self.indices),
        )
        object.__setattr__(self, "label", str(self.label or ""))

    @property
    def is_bound(self) -> bool:
        return bool(self.kind)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.source_id:
            payload["source_id"] = self.source_id
        if self.path:
            payload["path"] = self.path
        if self.indices:
            payload["indices"] = list(self.indices)
        if self.label:
            payload["label"] = self.label
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> SlotBinding:
        if isinstance(payload, SlotBinding):
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError("input-slot binding must be an object")
        return cls(
            kind=str(payload.get("kind") or ""),
            source_id=str(payload.get("source_id") or ""),
            path=str(payload.get("path") or ""),
            indices=tuple(payload.get("indices") or ()),
            label=str(payload.get("label") or ""),
        )


@dataclass(frozen=True)
class SlotSourceOption:
    """One source choice rendered by the Qt-free operation form."""

    binding: SlotBinding
    label: str
    description: str = ""
    available: bool = True
    unavailable_reason: str = ""


@dataclass(frozen=True)
class ResolvedSlot:
    """Process-local immutable snapshot for one bound auxiliary source.

    ``source`` is deliberately excluded from equality.  Operation/document cache
    identity is the explicit ``source_identity`` plus binding, shape, and dtype;
    large arrays must never participate in dataclass equality.
    """

    binding: SlotBinding
    shape: tuple[int, ...] = ()
    dtype: str = ""
    source_identity: object = None
    loader: str = "array"
    source: object = field(default=None, compare=False, repr=False)
    unavailable_reason: str = ""

    @property
    def available(self) -> bool:
        return not bool(self.unavailable_reason)

    @property
    def signature(self) -> tuple:
        return (
            self.binding,
            tuple(int(size) for size in self.shape),
            str(self.dtype),
            _freeze(self.source_identity),
        )

    def materialize(self) -> np.ndarray:
        if self.unavailable_reason:
            raise RuntimeError(self.unavailable_reason)
        if self.loader == "document":
            value = self.source.materialize()
        elif self.loader == "dimension-set":
            value = _read_dimension_set(self.source, self.binding.indices)
        elif self.loader == "saved-array":
            value = _load_saved_array(self.binding.path)
        elif self.loader == "roi-mask":
            from arrayscope.core.roi import roi_mask

            value = roi_mask(self.shape, self.source)
        elif self.loader == "roi-coordinates":
            from arrayscope.core.roi import roi_coordinates

            value = roi_coordinates(self.source)
        else:
            value = self.source
        result = np.asarray(value)
        if tuple(result.shape) != tuple(self.shape) or result.dtype.str != str(self.dtype):
            raise RuntimeError(
                f"slot source changed after binding: expected {tuple(self.shape)} / "
                f"{np.dtype(self.dtype)}, got {tuple(result.shape)} / {result.dtype}"
            )
        return result


def unresolved_slot(binding: SlotBinding, reason: str) -> ResolvedSlot:
    return ResolvedSlot(binding=binding, unavailable_reason=str(reason))


def inspect_saved_array(path: str) -> tuple[tuple[int, ...], np.dtype]:
    """Read saved ``.npy`` metadata through a memory map."""

    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if not path.lower().endswith(".npy"):
        raise ValueError("saved-array bindings currently require a .npy file")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    return tuple(int(size) for size in value.shape), value.dtype


def _load_saved_array(path: str) -> np.ndarray:
    path = os.path.abspath(os.path.expanduser(str(path)))
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _read_dimension_set(source, indices: tuple[int | None, ...]):
    index = tuple(slice(None) if value is None else int(value) for value in indices)
    reader = getattr(source, "read_region", None)
    return reader(index) if callable(reader) else source[index]


def _freeze(value):
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    if isinstance(value, np.generic):
        return value.item()
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value
