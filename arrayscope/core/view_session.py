"""Per-file viewer session persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
from arrayscope.core.view_recipe import (
    ViewRecipe,
    recipe_from_mapping,
    recipe_to_mapping,
)
from arrayscope.display.viewport import ViewportMode


VIEW_SESSION_VERSION = 1
SETTINGS_GROUP = "file_view_sessions"


@dataclass(frozen=True)
class ViewportSession:
    mode: str
    view_range: tuple[tuple[float, float], tuple[float, float]] | None
    viewport_shape: tuple[int, int] | None = None
    montage_columns: int | None = None


@dataclass(frozen=True)
class PanelSession:
    operation_visible: bool = False
    inspection_visible: bool = False
    window_size: tuple[int, int] | None = None
    window_maximized: bool | None = None


@dataclass(frozen=True)
class FileViewSession:
    metadata: dict[str, object]
    recipe: ViewRecipe
    viewport: ViewportSession | None = None
    rois: tuple[RoiSelection, ...] = ()
    selected_roi_id: str | None = None
    panels: PanelSession | None = None
    version: int = VIEW_SESSION_VERSION


def metadata_for_file(path, *, dataset_path=None, selector_class_name=None, data=None) -> dict[str, object]:
    path = Path(path)
    stat = path.stat()
    dtype = None if data is None else str(getattr(data, "dtype", ""))
    shape = None if data is None else [int(value) for value in tuple(getattr(data, "shape", ()))]
    return {
        "path": str(path.resolve()),
        "dataset_path": None if dataset_path is None else str(dataset_path),
        "selector_class_name": None if selector_class_name is None else str(selector_class_name),
        "shape": shape,
        "ndim": None if shape is None else len(shape),
        "dtype": dtype,
        "file_size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def metadata_matches(saved: dict[str, object], current: dict[str, object]) -> bool:
    keys = ("path", "dataset_path", "selector_class_name", "shape", "ndim", "dtype", "file_size", "mtime_ns")
    return all(saved.get(key) == current.get(key) for key in keys)


def settings_key_for_metadata(metadata: dict[str, object]) -> str:
    identity = json.dumps(
        {
            "path": metadata.get("path"),
            "dataset_path": metadata.get("dataset_path"),
            "selector_class_name": metadata.get("selector_class_name"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{SETTINGS_GROUP}/{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def session_filename_for_metadata(metadata: dict[str, object]) -> str:
    key = settings_key_for_metadata(metadata).split("/", 1)[-1][:12]
    raw_name = Path(str(metadata.get("path") or "file")).name
    if not raw_name:
        raw_name = "file"
    stem = Path(raw_name).stem or raw_name
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("._-") or "file"
    if len(safe_stem) > 48:
        safe_stem = safe_stem[:48].rstrip("._-") or "file"
    return f"{safe_stem}--{key}.json"


def session_path_for_metadata(config_dir, metadata: dict[str, object]) -> Path:
    return Path(config_dir) / SETTINGS_GROUP / session_filename_for_metadata(metadata)


def session_path_for_filename(config_dir, filename) -> Path:
    return Path(config_dir) / SETTINGS_GROUP / Path(os.fspath(filename)).name


def save_session_file(config_dir, session: FileViewSession) -> Path:
    path = session_path_for_metadata(config_dir, session.metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as session_file:
        temporary_path = Path(session_file.name)
        session_file.write(dumps_session(session))
        session_file.write("\n")
    os.replace(temporary_path, path)
    return path


def load_session_file(config_dir, metadata: dict[str, object], base_shape, *, filename=None) -> FileViewSession | None:
    path = session_path_for_filename(config_dir, filename) if filename else session_path_for_metadata(config_dir, metadata)
    if not path.exists():
        return None
    session = loads_session(path.read_text(encoding="utf-8"), base_shape)
    if not metadata_matches(session.metadata, metadata):
        return None
    return session


def session_to_mapping(session: FileViewSession) -> dict[str, object]:
    return {
        "version": int(session.version),
        "metadata": dict(session.metadata),
        "recipe": recipe_to_mapping(session.recipe),
        "viewport": None if session.viewport is None else viewport_to_mapping(session.viewport),
        "rois": [roi_to_mapping(roi) for roi in session.rois],
        "selected_roi_id": session.selected_roi_id,
        "panels": None if session.panels is None else panel_session_to_mapping(session.panels),
    }


def session_from_mapping(mapping, base_shape) -> FileViewSession:
    if not isinstance(mapping, dict):
        raise ValueError("file view session must be a JSON object")
    if mapping.get("version") != VIEW_SESSION_VERSION:
        raise ValueError(f"unsupported file view session version: {mapping.get('version')!r}")
    viewport = mapping.get("viewport")
    return FileViewSession(
        metadata=dict(mapping.get("metadata", {})),
        recipe=recipe_from_mapping(mapping.get("recipe", {}), base_shape),
        viewport=None if viewport is None else viewport_from_mapping(viewport),
        rois=tuple(roi_from_mapping(item) for item in tuple(mapping.get("rois", ()) or ())),
        selected_roi_id=mapping.get("selected_roi_id"),
        panels=(
            None
            if mapping.get("panels") is None
            else panel_session_from_mapping(mapping.get("panels"))
        ),
    )


def dumps_session(session: FileViewSession) -> str:
    return json.dumps(session_to_mapping(session), sort_keys=True, separators=(",", ":"))


def loads_session(text: str, base_shape) -> FileViewSession:
    return session_from_mapping(json.loads(str(text)), base_shape)


def viewport_to_mapping(viewport: ViewportSession) -> dict[str, object]:
    return {
        "mode": str(viewport.mode),
        "view_range": None if viewport.view_range is None else [list(axis) for axis in viewport.view_range],
        "viewport_shape": None if viewport.viewport_shape is None else [int(value) for value in viewport.viewport_shape],
        "montage_columns": None if viewport.montage_columns is None else int(viewport.montage_columns),
    }


def viewport_from_mapping(mapping) -> ViewportSession:
    if not isinstance(mapping, dict):
        raise ValueError("viewport session must be an object")
    view_range = mapping.get("view_range")
    normalized = None
    if view_range is not None:
        if (
            not isinstance(view_range, (list, tuple))
            or len(view_range) != 2
            or any(not isinstance(axis, (list, tuple)) or len(axis) != 2 for axis in view_range)
        ):
            raise ValueError("viewport.view_range must be [[x0, x1], [y0, y1]]")
        normalized = (
            (float(view_range[0][0]), float(view_range[0][1])),
            (float(view_range[1][0]), float(view_range[1][1])),
        )
    viewport_shape = mapping.get("viewport_shape")
    normalized_shape = None
    if viewport_shape is not None:
        if not isinstance(viewport_shape, (list, tuple)) or len(viewport_shape) != 2:
            raise ValueError("viewport.viewport_shape must be [height, width]")
        normalized_shape = (max(1, int(viewport_shape[0])), max(1, int(viewport_shape[1])))
    mode = str(mapping.get("mode", ViewportMode.AUTO_UNTOUCHED.value))
    montage_columns = mapping.get("montage_columns")
    normalized_columns = None if montage_columns is None else max(1, int(montage_columns))
    return ViewportSession(
        mode=mode,
        view_range=normalized,
        viewport_shape=normalized_shape,
        montage_columns=normalized_columns,
    )


def panel_session_to_mapping(panels: PanelSession) -> dict[str, object]:
    return {
        "operation_visible": bool(panels.operation_visible),
        "inspection_visible": bool(panels.inspection_visible),
        "window_size": (
            None
            if panels.window_size is None
            else [int(panels.window_size[0]), int(panels.window_size[1])]
        ),
        "window_maximized": panels.window_maximized,
    }


def panel_session_from_mapping(mapping) -> PanelSession:
    if not isinstance(mapping, dict):
        raise ValueError("session panels must be an object")
    window_size = mapping.get("window_size")
    normalized_size = None
    if window_size is not None:
        if not isinstance(window_size, (list, tuple)) or len(window_size) != 2:
            raise ValueError("session panels.window_size must be [width, height]")
        normalized_size = (
            max(1, int(window_size[0])),
            max(1, int(window_size[1])),
        )
    maximized = mapping.get("window_maximized")
    return PanelSession(
        operation_visible=bool(mapping.get("operation_visible", False)),
        inspection_visible=bool(mapping.get("inspection_visible", False)),
        window_size=normalized_size,
        window_maximized=None if maximized is None else bool(maximized),
    )


def roi_to_mapping(selection: RoiSelection) -> dict[str, object]:
    geometry = selection.geometry
    return {
        "id": str(selection.id),
        "label": str(selection.label),
        "enabled": bool(selection.enabled),
        "color": [int(value) for value in tuple(selection.color)[:3]],
        "geometry": {
            "kind": geometry.kind.value if isinstance(geometry.kind, RoiKind) else str(geometry.kind),
            "points": [[float(x), float(y)] for x, y in tuple(geometry.points)],
            "rect": None if geometry.rect is None else [float(value) for value in tuple(geometry.rect)],
            "line_width": float(geometry.line_width),
            "closed": bool(geometry.closed),
            "image_axes": [int(axis) for axis in tuple(geometry.image_axes)],
        },
    }


def roi_from_mapping(mapping) -> RoiSelection:
    if not isinstance(mapping, dict):
        raise ValueError("ROI session item must be an object")
    raw_geometry = mapping.get("geometry", {})
    raw_points = tuple(raw_geometry.get("points", ()) or ())
    raw_rect = raw_geometry.get("rect")
    raw_axes = tuple(raw_geometry.get("image_axes", (0, 1)))
    if any(not isinstance(point, (list, tuple)) or len(point) != 2 for point in raw_points):
        raise ValueError("ROI geometry points must be [x, y] pairs")
    if raw_rect is not None and (not isinstance(raw_rect, (list, tuple)) or len(raw_rect) != 4):
        raise ValueError("ROI geometry rect must be [x, y, width, height]")
    if len(raw_axes) != 2:
        raise ValueError("ROI geometry image_axes must contain exactly two axes")
    geometry = RoiGeometry(
        kind=raw_geometry.get("kind", RoiKind.RECTANGLE.value),
        points=tuple((float(point[0]), float(point[1])) for point in raw_points),
        rect=None if raw_rect is None else tuple(float(value) for value in tuple(raw_rect)),
        line_width=float(raw_geometry.get("line_width", 1.0)),
        closed=bool(raw_geometry.get("closed", False)),
        image_axes=(int(raw_axes[0]), int(raw_axes[1])),
    )
    color = tuple(int(value) for value in tuple(mapping.get("color", (230, 60, 30)))[:3])
    if len(color) != 3:
        raise ValueError("ROI color must contain exactly 3 RGB channels")
    return RoiSelection(
        id=str(mapping.get("id", "")),
        label=str(mapping.get("label", "")),
        geometry=geometry,
        enabled=bool(mapping.get("enabled", True)),
        color=color,
    )


def canonical_file_exists(path) -> bool:
    try:
        return bool(path is not None and os.path.exists(os.fspath(path)))
    except Exception:
        return False
