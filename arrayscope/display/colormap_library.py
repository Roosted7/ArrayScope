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

# The empty group renders ungrouped at the top of pickers ("Favorites").
FAVORITES_GROUP = ""
FAVORITES_LABEL = "Favorites"


@dataclass(frozen=True)
class ColormapInfo:
    name: str
    kind: str
    source: str  # "builtin" | "user"
    # None for built-ins resolved through pyqtgraph / factories.
    stops: tuple[tuple[float, tuple[int, int, int]], ...] | None = None
    group: str = "Other"
    hidden: bool = False


_LIPARI_STOPS = (
    (0.0, (3, 19, 38)),
    (0.0312, (6, 29, 53)),
    (0.0625, (9, 40, 68)),
    (0.0938, (14, 51, 83)),
    (0.125, (24, 62, 97)),
    (0.1562, (37, 72, 109)),
    (0.1875, (54, 81, 118)),
    (0.2188, (69, 88, 122)),
    (0.25, (82, 91, 122)),
    (0.2812, (92, 93, 121)),
    (0.3125, (101, 94, 119)),
    (0.3438, (110, 95, 117)),
    (0.375, (120, 95, 114)),
    (0.4062, (130, 96, 112)),
    (0.4375, (141, 97, 109)),
    (0.4688, (152, 97, 106)),
    (0.5, (165, 98, 103)),
    (0.5312, (176, 99, 100)),
    (0.5625, (190, 101, 97)),
    (0.5938, (203, 104, 95)),
    (0.625, (216, 110, 94)),
    (0.6562, (226, 119, 96)),
    (0.6875, (232, 130, 101)),
    (0.7188, (234, 142, 108)),
    (0.75, (233, 153, 115)),
    (0.7812, (231, 163, 122)),
    (0.8125, (229, 173, 130)),
    (0.8438, (229, 184, 140)),
    (0.875, (231, 195, 152)),
    (0.9062, (235, 207, 167)),
    (0.9375, (240, 219, 183)),
    (0.9688, (246, 232, 201)),
    (1.0, (253, 245, 218)),
)
_NAVIA_STOPS = (
    (0.0, (3, 19, 39)),
    (0.0312, (5, 28, 54)),
    (0.0625, (5, 37, 70)),
    (0.0938, (6, 47, 86)),
    (0.125, (7, 57, 102)),
    (0.1562, (10, 67, 116)),
    (0.1875, (14, 77, 129)),
    (0.2188, (20, 87, 138)),
    (0.25, (27, 96, 143)),
    (0.2812, (32, 104, 145)),
    (0.3125, (38, 111, 144)),
    (0.3438, (42, 116, 142)),
    (0.375, (47, 121, 139)),
    (0.4062, (51, 125, 137)),
    (0.4375, (55, 129, 134)),
    (0.4688, (60, 134, 131)),
    (0.5, (65, 138, 128)),
    (0.5312, (70, 143, 125)),
    (0.5625, (76, 148, 122)),
    (0.5938, (82, 154, 118)),
    (0.625, (89, 160, 114)),
    (0.6562, (98, 168, 110)),
    (0.6875, (107, 176, 106)),
    (0.7188, (120, 185, 104)),
    (0.75, (135, 194, 105)),
    (0.7812, (154, 204, 112)),
    (0.8125, (174, 213, 125)),
    (0.8438, (193, 220, 141)),
    (0.875, (209, 227, 159)),
    (0.9062, (223, 232, 176)),
    (0.9375, (234, 237, 191)),
    (0.9688, (244, 241, 205)),
    (1.0, (252, 244, 217)),
)


# Curated built-in library, grouped for navigable pickers — an opinionated,
# high-quality set rather than an exhaustive one. Kind assignments follow
# the CET taxonomy (L=linear, D=diverging, C=cyclic). "crameri:<file>"
# entries load Fabio Crameri's Scientific colour maps from the cmcrameri
# package at runtime; Lipari and Navia keep embedded samples as fallback so
# they exist even without cmcrameri installed.
_BUILTIN_SPECS = (
    # (name, kind, group, stops | "crameri:<file>" | None=pyqtgraph)
    # Group "" = Favorites: a handful of go-to maps shown ungrouped at the
    # top of pickers. Groups are collections; the *kind* is separate
    # metadata (and a filter in the designer).
    ("gray", SEQUENTIAL, FAVORITES_GROUP, None),
    ("viridis", SEQUENTIAL, FAVORITES_GROUP, None),
    ("Batlow", SEQUENTIAL, FAVORITES_GROUP, "crameri:batlow"),
    ("Vik", DIVERGING, FAVORITES_GROUP, "crameri:vik"),
    ("PAL-relaxed", CYCLIC, FAVORITES_GROUP, None),
    ("RomaO", CYCLIC, FAVORITES_GROUP, "crameri:romaO"),
    ("plasma", SEQUENTIAL, "Perceptual", None),
    ("inferno", SEQUENTIAL, "Perceptual", None),
    ("cividis", SEQUENTIAL, "Perceptual", None),
    ("turbo", SEQUENTIAL, "Perceptual", None),
    ("Lipari", SEQUENTIAL, "Scientific", "crameri:lipari"),
    ("Navia", SEQUENTIAL, "Scientific", "crameri:navia"),
    ("Davos", SEQUENTIAL, "Scientific", "crameri:davos"),
    ("LaJolla", SEQUENTIAL, "Scientific", "crameri:lajolla"),
    ("Oslo", SEQUENTIAL, "Scientific", "crameri:oslo"),
    ("Roma", DIVERGING, "Scientific", "crameri:roma"),
    ("Broc", DIVERGING, "Scientific", "crameri:broc"),
    ("VikO", CYCLIC, "Scientific", "crameri:vikO"),
    ("CorkO", CYCLIC, "Scientific", "crameri:corkO"),
    ("CET-L1", SEQUENTIAL, "CET", None),
    ("CET-L17", SEQUENTIAL, "CET", None),
    ("CET-CBL1", SEQUENTIAL, "CET", None),
    ("CET-D1", DIVERGING, "CET", None),
    ("CET-CBD1", DIVERGING, "CET", None),
    ("CET-C2", CYCLIC, "CET", None),
    ("CET-C6", CYCLIC, "CET", None),
    ("CET-CBC1", CYCLIC, "CET", None),
    ("d3-warm", SEQUENTIAL, "Classic", None),
    ("d3-cool", SEQUENTIAL, "Classic", None),
    ("hsv-phase", CYCLIC, "Classic", None),
    ("PAL-relaxed_bright", CYCLIC, "Classic", None),
)

_CRAMERI_FALLBACK_STOPS = {"lipari": _LIPARI_STOPS, "navia": _NAVIA_STOPS}
_crameri_stops_cache: dict = {}


def _resolve_builtin_stops(spec_stops):
    """None => pyqtgraph name; tuple => embedded; 'crameri:x' => cmcrameri."""
    if spec_stops is None or isinstance(spec_stops, tuple):
        return spec_stops
    file_stem = str(spec_stops).partition(":")[2]
    if file_stem in _crameri_stops_cache:
        return _crameri_stops_cache[file_stem]
    stops = None
    try:
        from pathlib import Path

        import cmcrameri

        table = np.loadtxt(Path(cmcrameri.__file__).parent / "cmaps" / f"{file_stem}.txt")
        stops = _stops_from_table(table)
    except Exception:
        stops = _CRAMERI_FALLBACK_STOPS.get(file_stem)
    _crameri_stops_cache[file_stem] = stops
    return stops


def _builtin_spec_available(spec_stops) -> bool:
    if spec_stops is None or isinstance(spec_stops, tuple):
        return True
    return _resolve_builtin_stops(spec_stops) is not None


GROUP_ORDER = (FAVORITES_GROUP, "Perceptual", "Scientific", "CET", "Classic", "User")

_user_cache: dict[str, ColormapInfo] | None = None
_listeners: list = []


# ---------------------------------------------------------------------------
# Listing / lookup
# ---------------------------------------------------------------------------


def builtin_colormaps() -> tuple[ColormapInfo, ...]:
    hidden = hidden_builtins()
    result = []
    for name, kind, group, spec_stops in _BUILTIN_SPECS:
        if not _builtin_spec_available(spec_stops):
            continue
        stops = spec_stops if isinstance(spec_stops, tuple) else None
        result.append(ColormapInfo(name, kind, "builtin", stops, group, name in hidden))
    return tuple(result)


def builtin_group_for(name: str) -> str | None:
    for spec_name, _kind, group, _stops in _BUILTIN_SPECS:
        if spec_name == str(name):
            return group
    return None


def user_colormaps() -> tuple[ColormapInfo, ...]:
    return tuple(_load_user_cache().values())


def list_colormaps(kind: str | None = None, *, include_hidden: bool = False) -> tuple[ColormapInfo, ...]:
    """User maps first (they shadow built-ins by name), then built-ins."""
    seen = set()
    result = []
    for info in (*user_colormaps(), *builtin_colormaps()):
        if info.name in seen:
            continue
        seen.add(info.name)
        if info.hidden and not include_hidden:
            continue
        if kind is None or info.kind == kind:
            result.append(info)
    return tuple(result)


def _layout_file() -> str:
    return os.path.join(user_colormap_directory(), "layout.json")


_layout_cache: dict | None = None


def _load_layout() -> dict:
    global _layout_cache
    if _layout_cache is None:
        try:
            with open(_layout_file()) as handle:
                payload = json.load(handle)
            _layout_cache = {
                "group_order": [str(g) for g in payload.get("group_order", [])],
                "map_groups": {str(k): str(v) for k, v in payload.get("map_groups", {}).items()},
                "map_order": {str(k): int(v) for k, v in payload.get("map_order", {}).items()},
            }
        except Exception:
            _layout_cache = {"group_order": [], "map_groups": {}, "map_order": {}}
    return _layout_cache


def _save_layout(layout: dict) -> None:
    global _layout_cache
    os.makedirs(user_colormap_directory(), exist_ok=True)
    with open(_layout_file(), "w") as handle:
        json.dump(layout, handle, indent=2)
    _layout_cache = None
    _notify()


def effective_group(info: ColormapInfo) -> str:
    return _load_layout()["map_groups"].get(info.name, info.group)


def apply_library_layout(group_order, map_groups, map_order) -> None:
    """Persist a user arrangement (from the designer's drag & drop)."""
    _save_layout(
        {
            "group_order": [str(g) for g in group_order],
            "map_groups": {str(k): str(v) for k, v in map_groups.items()},
            "map_order": {str(k): int(v) for k, v in map_order.items()},
        }
    )


def rename_group(old: str, new: str) -> None:
    """Rename a group: reassign every member map and the order entry."""
    old, new = str(old), str(new).strip()
    if not new or old == new or old == FAVORITES_GROUP:
        return
    layout = dict(_load_layout())
    map_groups = dict(layout["map_groups"])
    for info in list_colormaps(include_hidden=True):
        if effective_group(info) == old:
            map_groups[info.name] = new
    group_order = [new if g == old else g for g in layout["group_order"]]
    _save_layout({"group_order": group_order, "map_groups": map_groups, "map_order": layout["map_order"]})


def grouped_colormaps(family: str | None = None, *, include_hidden: bool = False):
    """Ordered [(group, [infos])], honoring the persisted user layout."""
    kinds = None if family is None else kinds_for_family(family)
    layout = _load_layout()
    by_group: dict[str, list[ColormapInfo]] = {}
    default_positions = {name: index for index, (name, *_rest) in enumerate(_BUILTIN_SPECS)}
    for info in list_colormaps(include_hidden=include_hidden):
        if kinds is not None and info.kind not in kinds:
            continue
        by_group.setdefault(effective_group(info), []).append(info)
    for group, infos in by_group.items():
        infos.sort(
            key=lambda info: (
                layout["map_order"].get(info.name, 10_000 + default_positions.get(info.name, 20_000)),
                info.name.lower(),
            )
        )
    ordered = [group for group in layout["group_order"] if group in by_group]
    ordered += [group for group in GROUP_ORDER if group in by_group and group not in ordered]
    ordered += [group for group in sorted(by_group) if group not in ordered]
    return [(group, by_group[group]) for group in ordered]


def kinds_for_family(family: str) -> tuple[str, ...]:
    """Which colormap kinds are valid for a channel family."""
    return (CYCLIC,) if str(family) == "phase" else (SEQUENTIAL, DIVERGING)


def colormaps_for_family(family: str) -> tuple[ColormapInfo, ...]:
    kinds = kinds_for_family(family)
    return tuple(info for info in list_colormaps() if info.kind in kinds)


def find_colormap(name: str) -> ColormapInfo | None:
    for info in list_colormaps(include_hidden=True):
        if info.name == str(name):
            return info
    return None


def builtin_stops_colormap(name: str):
    """ColorMap for a built-in that carries its own stops (embedded or
    cmcrameri-backed); None when the name resolves elsewhere."""
    for spec_name, _kind, _group, spec_stops in _BUILTIN_SPECS:
        if spec_name == str(name) and spec_stops is not None:
            stops = _resolve_builtin_stops(spec_stops)
            if stops is not None:
                return _colormap_from_stops(stops)
    return None


def get_colormap(name: str):
    """Resolve a name to a pyqtgraph ColorMap (user maps shadow built-ins)."""
    info = _load_user_cache().get(str(name))
    if info is not None and info.stops:
        return _colormap_from_stops(info.stops)
    for spec_name, _kind, _group, spec_stops in _BUILTIN_SPECS:
        if spec_name == str(name) and spec_stops is not None:
            stops = _resolve_builtin_stops(spec_stops)
            if stops is not None:
                return _colormap_from_stops(stops)
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
    global _user_cache, _layout_cache
    _user_cache = None
    _layout_cache = None
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
    builtin_group = builtin_group_for(name)
    group = "User" if builtin_group is None else builtin_group
    info = ColormapInfo(name, str(kind), "user", normalized, group)
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


def import_colormap_file(path: str, *, name: str | None = None, kind: str | None = None) -> ColormapInfo:
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
    if kind is None:
        detected, confidence = detect_colormap_kind(stops)
        kind = detected if confidence >= 0.9 else SEQUENTIAL
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
# Hidden built-ins (deletable, restorable)
# ---------------------------------------------------------------------------


def _hidden_file() -> str:
    return os.path.join(user_colormap_directory(), "hidden-builtins.json")


def hidden_builtins() -> frozenset[str]:
    try:
        with open(_hidden_file()) as handle:
            return frozenset(str(name) for name in json.load(handle))
    except Exception:
        return frozenset()


def set_builtin_hidden(name: str, hidden: bool) -> None:
    names = set(hidden_builtins())
    if bool(hidden):
        names.add(str(name))
    else:
        names.discard(str(name))
    os.makedirs(user_colormap_directory(), exist_ok=True)
    with open(_hidden_file(), "w") as handle:
        json.dump(sorted(names), handle)
    _notify()


def overrides_builtin(name: str) -> bool:
    return str(name) in _load_user_cache() and builtin_group_for(name) is not None


def reset_builtin(name: str) -> bool:
    """Remove a user override and/or un-hide the built-in of this name."""
    changed = False
    if overrides_builtin(name):
        changed = delete_user_colormap(name) or changed
    if str(name) in hidden_builtins():
        set_builtin_hidden(name, False)
        changed = True
    return changed


# ---------------------------------------------------------------------------
# Kind detection for imported tables
# ---------------------------------------------------------------------------


def detect_colormap_kind(stops) -> tuple[str, float]:
    """Best-guess (kind, confidence 0..1) from stop colors.

    Cyclic: endpoints nearly identical. Diverging: endpoint lightness
    similar with a strong lightness extremum near the middle. Otherwise
    sequential, scored by lightness monotonicity.
    """
    colors = np.asarray([color for _pos, color in stops], dtype=float)
    if len(colors) < 3:
        return SEQUENTIAL, 0.0
    lightness = colors @ np.array([0.299, 0.587, 0.114])
    span = max(1.0, float(lightness.max() - lightness.min()))

    endpoint_rgb_distance = float(np.linalg.norm(colors[0] - colors[-1]))
    cyclic_score = max(0.0, 1.0 - endpoint_rgb_distance / 90.0)

    endpoint_l_delta = abs(float(lightness[0] - lightness[-1]))
    mid = len(lightness) // 2
    center_zone = lightness[max(1, mid - max(1, len(lightness) // 4)) : mid + max(2, len(lightness) // 4)]
    center_extreme = max(
        float(center_zone.max() - max(lightness[0], lightness[-1])),
        float(min(lightness[0], lightness[-1]) - center_zone.min()),
    )
    diverging_score = max(0.0, min(1.0, center_extreme / (0.35 * span))) * max(
        0.0, 1.0 - endpoint_l_delta / (0.3 * span)
    )
    # A cyclic map also has matching endpoints; prefer cyclic when both fire.
    diverging_score *= 1.0 - 0.5 * cyclic_score

    diffs = np.diff(lightness)
    monotonic_score = float(max((diffs >= -1e-9).mean(), (diffs <= 1e-9).mean()))
    sequential_score = monotonic_score * max(0.0, 1.0 - cyclic_score)

    scores = {CYCLIC: cyclic_score, DIVERGING: diverging_score, SEQUENTIAL: sequential_score}
    best = max(scores, key=scores.get)
    return best, float(scores[best])


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
    builtin_group = builtin_group_for(name)
    return ColormapInfo(name, kind, "user", stops, "User" if builtin_group is None else builtin_group)


def _safe_file_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name) or "colormap"
