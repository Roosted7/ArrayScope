"""JSONL serialization helpers for runtime diagnostics snapshots."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import json
import math
import os

from arrayscope.core.runtime_diagnostics import WindowRuntimeDiagnostics


DIAGNOSTICS_JSONL_SCHEMA_VERSION = 1


def diagnostics_to_jsonable(value):
    """Convert diagnostics snapshots to JSON-compatible primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: diagnostics_to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {_json_key(key): diagnostics_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostics_to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [diagnostics_to_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return f"<{value.__class__.__module__}.{value.__class__.__qualname__}>"


def diagnostics_start_record(
    snapshot: WindowRuntimeDiagnostics,
    *,
    recorded_at: str,
    app_version: str,
    cwd: str,
    pid: int,
    python_version: str,
    platform: str,
    interval_ms: int,
) -> dict[str, object]:
    policy = snapshot.memory_policy
    return {
        "schema_version": DIAGNOSTICS_JSONL_SCHEMA_VERSION,
        "event": "start",
        "recorded_at": recorded_at,
        "app_version": str(app_version),
        "cwd": str(cwd),
        "pid": int(pid),
        "python_version": str(python_version),
        "platform": str(platform),
        "interval_ms": int(interval_ms),
        "config": {
            "derived_shape": diagnostics_to_jsonable(snapshot.derived_shape),
            "derived_dtype": snapshot.derived_dtype,
            "operation_count": int(snapshot.operation_count),
            "fft_backend_choice": snapshot.fft_backend_choice,
            "fft_backend_resolved": snapshot.fft_backend_resolved,
            "fft_workers_choice": snapshot.fft_workers_choice,
            "fft_workers_resolved": int(snapshot.fft_workers_resolved),
            "memory_profile": policy.profile.value,
            "memory_budgets": {
                "visible_render_bytes": int(policy.visible_render_budget_bytes),
                "montage_canvas_bytes": int(policy.montage_canvas_budget_bytes),
                "single_tile_bytes": int(policy.single_tile_budget_bytes),
                "image_cache_bytes": int(policy.image_cache_budget_bytes),
                "tile_cache_bytes": int(policy.tile_cache_budget_bytes),
                "profile_cache_bytes": int(policy.profile_cache_budget_bytes),
                "stage_cache_bytes": int(policy.stage_cache_budget_bytes),
                "prefetch_bytes": int(policy.prefetch_budget_bytes),
            },
            "image_rendering_backend_selected": snapshot.image_rendering_backend_selected,
            "image_rendering_backend_actual": snapshot.image_rendering_backend_actual,
        },
    }


def diagnostics_snapshot_record(
    snapshot: WindowRuntimeDiagnostics,
    *,
    sequence: int,
    recorded_at: str,
) -> dict[str, object]:
    return {
        "schema_version": DIAGNOSTICS_JSONL_SCHEMA_VERSION,
        "event": "snapshot",
        "sequence": int(sequence),
        "recorded_at": recorded_at,
        "diagnostics": diagnostics_to_jsonable(snapshot),
    }


def diagnostics_jsonl_line(record: dict[str, object]) -> str:
    return json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"


def _json_key(value) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return repr(value)
