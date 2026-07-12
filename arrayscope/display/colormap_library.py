"""First-class colormap registry: built-ins, persistent user maps, import/export.

Colormaps carry a *kind* — ``sequential``, ``diverging`` or ``cyclic`` — so the
UI can offer only maps that make sense for the active channel: cyclic maps for
phase/complex display, everything else for scalar channels. Using a
non-cyclic map as a phase LUT produces a hard seam at ±π, which is exactly
the breakage this module exists to prevent.

User-defined maps persist as small JSON files in the application data
directory and can be imported from common native formats (MATLAB ``.mat``,
``.csv``/``.txt`` tables, ``.npy`` arrays, or this JSON schema).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

SEQUENTIAL = "sequential"
DIVERGING = "diverging"
CYCLIC = "cyclic"
KINDS = (SEQUENTIAL, DIVERGING, CYCLIC)


@dataclass(frozen=True)
class ColormapInfo:
    name: str
    kind: str
    source: str  # "builtin" | "user"
    # None for built-ins resolved through pyqtgraph / factories.
    stops: tuple[tuple[float, tuple[int, int, int]], ...] | None = None


# Curated built-in library. Kind assignments follow the CET taxonomy
# (L=linear, D=diverging, C=cyclic); ArrayScope's own maps keep their roles.
_BUILTIN_SPECS = (
    # Sequential / linear
    ("gray", SEQUENTIAL),
    ("viridis", SEQUENTIAL),
    ("plasma", SEQUENTIAL),
    ("inferno", SEQUENTIAL),
    ("magma", SEQUENTIAL),
    ("cividis", SEQUENTIAL),
    ("turbo", SEQUENTIAL),
    ("d3-warm", SEQUENTIAL),
    ("d3-cool", SEQUENTIAL),
    ("CET-L1", SEQUENTIAL),
    ("CET-L3", SEQUENTIAL),
    ("CET-L8", SEQUENTIAL),
    ("CET-L16", SEQUENTIAL),
    ("CET-L17", SEQUENTIAL),
    ("CET-CBL1", SEQUENTIAL),
    # Diverging
    ("CET-D1", DIVERGING),
    ("CET-D3", DIVERGING),
    ("CET-D7", DIVERGING),
    ("CET-D13", DIVERGING),
    ("CET-CBD1", DIVERGING),
    # Cyclic (phase-safe)
    ("PAL-relaxed", CYCLIC),
    ("PAL-relaxed_bright", CYCLIC),
    ("hsv-phase", CYCLIC),
    ("CET-C1", CYCLIC),
    ("CET-C2", CYCLIC),
    ("CET-C3", CYCLIC),
    ("CET-C6", CYCLIC),
    ("CET-C7", CYCLIC),
    ("CET-CBC1", CYCLIC),
)

_user_cache: dict[str, ColormapInfo] | None = None
_listeners: list = []


# ---------------------------------------------------------------------------
# Listing / lookup
# ---------------------------------------------------------------------------


def builtin_colormaps() -> tuple[ColormapInfo, ...]:
    return tuple(ColormapInfo(name, kind, "builtin") for name, kind in _BUILTIN_SPECS)


def user_colormaps() -> tuple[ColormapInfo, ...]:
    return tuple(_load_user_cache().values())


def list_colormaps(kind: str | None = None) -> tuple[ColormapInfo, ...]:
    """User maps first (they shadow built-ins by name), then built-ins."""
    seen = set()
    result = []
    for info in (*user_colormaps(), *builtin_colormaps()):
        if info.name in seen:
            continue
        seen.add(info.name)
        if kind is None or info.kind == kind:
            result.append(info)
    return tuple(result)


def kinds_for_family(family: str) -> tuple[str, ...]:
    """Which colormap kinds are valid for a channel family."""
    return (CYCLIC,) if str(family) == "phase" else (SEQUENTIAL, DIVERGING)


def colormaps_for_family(family: str) -> tuple[ColormapInfo, ...]:
    kinds = kinds_for_family(family)
    return tuple(info for info in list_colormaps() if info.kind in kinds)


def find_colormap(name: str) -> ColormapInfo | None:
    for info in list_colormaps():
        if info.name == str(name):
            return info
    return None


def get_colormap(name: str):
    """Resolve a name to a pyqtgraph ColorMap (user maps shadow built-ins)."""
    info = _load_user_cache().get(str(name))
    if info is not None and info.stops:
        return _colormap_from_stops(info.stops)
    from arrayscope.display.colormaps import named_colormap

    return named_colormap(str(name))


def colormap_stops(name: str, points: int = 9) -> tuple[tuple[float, tuple[int, int, int]], ...]:
    """Sampled stops for a named map (used to seed the designer)."""
    info = _load_user_cache().get(str(name))
    if info is not None and info.stops:
        return info.stops
    colormap = get_colormap(name)
    positions = np.linspace(0.0, 1.0, int(points))
    lut = colormap.getLookupTable(0.0, 1.0, int(points), alpha=False)
    return tuple(
        (float(pos), (int(rgb[0]), int(rgb[1]), int(rgb[2])))
        for pos, rgb in zip(positions, lut)
    )


# ---------------------------------------------------------------------------
# User map persistence
# ---------------------------------------------------------------------------


def add_library_listener(callback) -> None:
    if callable(callback) and callback not in _listeners:
        _listeners.append(callback)


def _notify() -> None:
    for callback in tuple(_listeners):
        try:
            callback()
        except Exception:
            pass


def user_colormap_directory() -> str:
    try:
        from pyqtgraph.Qt import QtCore

        base = QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.AppDataLocation
        )
    except Exception:
        base = ""
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".local", "share", "ArrayScope")
    return os.path.join(base, "colormaps")


def _load_user_cache() -> dict[str, ColormapInfo]:
    global _user_cache
    if _user_cache is None:
        _user_cache = {}
        directory = user_colormap_directory()
        if os.path.isdir(directory):
            for file_name in sorted(os.listdir(directory)):
                if not file_name.endswith(".json"):
                    continue
                try:
                    info = _info_from_json_file(os.path.join(directory, file_name))
                    _user_cache[info.name] = info
                except Exception:
                    continue
    return _user_cache


def refresh_user_colormaps() -> None:
    global _user_cache
    _user_cache = None
    _load_user_cache()
    _notify()


def save_user_colormap(name: str, kind: str, stops) -> ColormapInfo:
    name = str(name).strip()
    if not name:
        raise ValueError("colormap name must not be empty")
    if kind not in KINDS:
        raise ValueError(f"unknown colormap kind: {kind}")
    normalized = _normalize_stops(stops)
    if len(normalized) < 2:
        raise ValueError("a colormap needs at least two stops")
    info = ColormapInfo(name, str(kind), "user", normalized)
    directory = user_colormap_directory()
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{_safe_file_name(name)}.json"), "w") as handle:
        json.dump(_info_to_payload(info), handle, indent=2)
    _load_user_cache()[name] = info
    _notify()
    return info


def delete_user_colormap(name: str) -> bool:
    cache = _load_user_cache()
    info = cache.pop(str(name), None)
    path = os.path.join(user_colormap_directory(), f"{_safe_file_name(str(name))}.json")
    removed = False
    if os.path.exists(path):
        os.remove(path)
        removed = True
    if info is not None or removed:
        _notify()
    return info is not None or removed


def export_colormap(name: str, path: str) -> None:
    info = find_colormap(name)
    if info is None:
        raise ValueError(f"unknown colormap: {name}")
    stops = info.stops or colormap_stops(name, points=17)
    payload = _info_to_payload(ColormapInfo(info.name, info.kind, info.source, stops))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


# ---------------------------------------------------------------------------
# Import from native formats
# ---------------------------------------------------------------------------


def import_colormap_file(path: str, *, name: str | None = None, kind: str = SEQUENTIAL) -> ColormapInfo:
    """Import a colormap from .json (ours), .mat (MATLAB), .csv/.txt or .npy.

    Tabular formats hold an N×3 (or N×4, alpha ignored) array in 0–1 floats
    or 0–255 ints. The map is saved as a persistent user colormap.
    """
    lower = str(path).lower()
    if lower.endswith(".json"):
        info = _info_from_json_file(path)
        return save_user_colormap(name or info.name, info.kind, info.stops)
    if lower.endswith(".mat"):
        table = _table_from_mat(path)
    elif lower.endswith(".npy"):
        table = np.load(path)
    else:  # csv / txt / anything np.loadtxt can read
        table = _loadtxt_any(path)
    stops = _stops_from_table(np.asarray(table))
    default_name = os.path.splitext(os.path.basename(path))[0]
    return save_user_colormap(name or default_name, kind, stops)


def _table_from_mat(path: str) -> np.ndarray:
    from scipy.io import loadmat

    contents = loadmat(path)
    candidates = []
    for key, value in contents.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[1] in (3, 4) and array.shape[0] >= 2:
            candidates.append((array.shape[0], key, array))
    if not candidates:
        raise ValueError("no N×3 colormap array found in the MATLAB file")
    candidates.sort(reverse=True)
    return candidates[0][2]


def _loadtxt_any(path: str) -> np.ndarray:
    for delimiter in (",", None, ";", "\t"):
        try:
            table = np.loadtxt(path, delimiter=delimiter)
            if table.ndim == 2 and table.shape[1] in (3, 4):
                return table
        except Exception:
            continue
    raise ValueError("could not parse a numeric N×3 table from the file")


def _stops_from_table(table: np.ndarray):
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[1] not in (3, 4) or table.shape[0] < 2:
        raise ValueError(f"expected an N×3 color table, got shape {table.shape}")
    rgb = table[:, :3]
    if rgb.max() <= 1.0 + 1e-9:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0, 255)
    count = rgb.shape[0]
    # Dense tables are resampled to a manageable set of editing stops.
    if count > 33:
        indices = np.linspace(0, count - 1, 33).round().astype(int)
        rgb = rgb[indices]
        count = rgb.shape[0]
    positions = np.linspace(0.0, 1.0, count)
    return tuple(
        (float(pos), (int(round(r)), int(round(g)), int(round(b))))
        for pos, (r, g, b) in zip(positions, rgb)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_stops(stops):
    normalized = []
    for position, color in stops:
        rgb = tuple(int(max(0, min(255, round(component)))) for component in tuple(color)[:3])
        normalized.append((float(max(0.0, min(1.0, position))), rgb))
    normalized.sort(key=lambda item: item[0])
    return tuple(normalized)


def _colormap_from_stops(stops):
    import pyqtgraph as pg

    positions = [position for position, _color in stops]
    colors = [tuple(color) + (255,) for _position, color in stops]
    return pg.ColorMap(positions, colors)


def _info_to_payload(info: ColormapInfo) -> dict:
    return {
        "format": "arrayscope-colormap",
        "version": 1,
        "name": info.name,
        "kind": info.kind,
        "stops": [[position, list(color)] for position, color in (info.stops or ())],
    }


def _info_from_json_file(path: str) -> ColormapInfo:
    with open(path) as handle:
        payload = json.load(handle)
    name = str(payload.get("name") or os.path.splitext(os.path.basename(path))[0])
    kind = payload.get("kind", SEQUENTIAL)
    if kind not in KINDS:
        kind = SEQUENTIAL
    stops = _normalize_stops(
        (float(position), tuple(color)) for position, color in payload.get("stops", ())
    )
    if len(stops) < 2:
        raise ValueError(f"colormap file has fewer than two stops: {path}")
    return ColormapInfo(name, kind, "user", stops)


def _safe_file_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name) or "colormap"
