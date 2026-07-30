"""First-class colormap registry: built-ins, persistent user maps, import/export.

Colormaps carry a *kind* — ``sequential``, ``diverging`` or ``cyclic`` — so the
UI can offer only maps that make sense for the active channel: cyclic maps for
phase/complex display, everything else for scalar channels. Using a
non-cyclic map as a phase LUT produces a hard seam at ±π, which is exactly
the breakage this module exists to prevent.

User-defined maps persist as small JSON files *next to this install's config*
(see :func:`user_colormap_directory`) and can be imported from common native
formats (MATLAB ``.mat``, ``.csv``/``.txt`` tables, ``.npy`` arrays, or this
JSON schema).

Storage, layout, hidden built-ins, listeners and the problems channel are the
shared :mod:`arrayscope.app.user_library` protocol -- the same one
:mod:`arrayscope.operations.library` answers (ADR 0063). A map file this module
cannot read is *recorded*, never swallowed: it shows up in
:func:`user_colormap_problems` and stays addressable in the manager (ADR 0062).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import numpy as np

from arrayscope.app import user_dirs, user_library

_LOGGER = logging.getLogger(__name__)

LAYOUT_FORMAT = "arrayscope-colormap-layout"
COLORMAP_FORMAT = "arrayscope-colormap"

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
    # Non-empty when the definition exists but cannot be resolved right now --
    # a cmcrameri built-in whose package is not installed, say. Such a map used
    # to be dropped from the listing entirely, which made a curated name simply
    # vanish with no explanation; operations have said "(unavailable)" out loud
    # since ADR 0060 and colormaps now use the same word (ADR 0063).
    unavailable_reason: str = ""


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


def _builtin_unavailable_reason(spec_stops) -> str:
    if spec_stops is None or isinstance(spec_stops, tuple):
        return ""
    if _resolve_builtin_stops(spec_stops) is not None:
        return ""
    return (
        f"Scientific colour map “{str(spec_stops).partition(':')[2]}” needs the "
        "cmcrameri package, which is not installed."
    )


GROUP_ORDER = (FAVORITES_GROUP, "Perceptual", "Scientific", "CET", "Classic", "User")

_user_cache: dict[str, ColormapInfo] | None = None
_user_problems: list[user_library.LibraryProblem] = []
_listeners = user_library.ListenerRegistry("colormap library")
# Resolved through a lambda, never captured: tests monkeypatch
# ``user_colormap_directory`` on this module.
_layout = user_library.LayoutStore(
    lambda: user_colormap_directory(),
    format_name=LAYOUT_FORMAT,
    group_key="map_groups",
    order_key="map_order",
    listeners=_listeners,
)
_hidden = user_library.HiddenSet(
    lambda: user_colormap_directory(),
    file_name="hidden-builtins.json",
    listeners=_listeners,
)
# Library bookkeeping that shares the directory with the colormap files; never a
# colormap, and never a "problem" when it does not parse as one.
_RESERVED_FILES = ("layout.json", "hidden-builtins.json")


# ---------------------------------------------------------------------------
# Listing / lookup
# ---------------------------------------------------------------------------


def builtin_colormaps() -> tuple[ColormapInfo, ...]:
    hidden = hidden_builtins()
    result = []
    for name, kind, group, spec_stops in _BUILTIN_SPECS:
        stops = spec_stops if isinstance(spec_stops, tuple) else None
        result.append(
            ColormapInfo(
                name,
                kind,
                "builtin",
                stops,
                group,
                name in hidden,
                _builtin_unavailable_reason(spec_stops),
            )
        )
    return tuple(result)


def builtin_group_for(name: str) -> str | None:
    for spec_name, _kind, group, _stops in _BUILTIN_SPECS:
        if spec_name == str(name):
            return group
    return None


def user_colormaps() -> tuple[ColormapInfo, ...]:
    return tuple(_load_user_cache().values())


def list_colormaps(
    kind: str | None = None, *, include_hidden: bool = False, include_unavailable: bool = False
) -> tuple[ColormapInfo, ...]:
    """User maps first (they shadow built-ins by name), then built-ins."""
    seen = set()
    result = []
    for info in (*user_colormaps(), *builtin_colormaps()):
        if info.name in seen:
            continue
        seen.add(info.name)
        if info.hidden and not include_hidden:
            continue
        if info.unavailable_reason and not include_unavailable:
            continue
        if kind is None or info.kind == kind:
            result.append(info)
    return tuple(result)


def effective_group(info: ColormapInfo) -> str:
    return _layout.group_of(info.name, info.group)


def apply_library_layout(group_order, map_groups, map_order) -> None:
    """Persist a user arrangement (from the manager's drag & drop)."""
    _layout.save(
        {
            "group_order": [str(g) for g in group_order],
            "map_groups": {str(k): str(v) for k, v in map_groups.items()},
            "map_order": {str(k): int(v) for k, v in map_order.items()},
        }
    )


def rename_group(old: str, new: str) -> bool:
    """Rename a group: reassign every member map and the order entry."""
    if str(old) == FAVORITES_GROUP:
        # Favorites is the unnamed section pickers render ungrouped at the top;
        # renaming it would make it an ordinary group and lose that meaning.
        return False
    members = [
        info.name
        for info in list_colormaps(include_hidden=True, include_unavailable=True)
        if effective_group(info) == str(old)
    ]
    return _layout.rename_group(old, new, members)


def reset_layout() -> None:
    """Discard the persisted arrangement (revert to the curated defaults)."""

    _layout.remove()


def grouped_colormaps(
    family: str | None = None, *, include_hidden: bool = False, include_unavailable: bool = False
):
    """Ordered [(group, [infos])], honoring the persisted user layout."""
    kinds = None if family is None else kinds_for_family(family)
    by_group: dict[str, list[ColormapInfo]] = {}
    default_positions = {name: index for index, (name, *_rest) in enumerate(_BUILTIN_SPECS)}
    for info in list_colormaps(
        include_hidden=include_hidden, include_unavailable=include_unavailable
    ):
        if kinds is not None and info.kind not in kinds:
            continue
        by_group.setdefault(effective_group(info), []).append(info)
    for infos in by_group.values():
        infos.sort(
            key=lambda info: (
                _layout.position_of(info.name, 10_000 + default_positions.get(info.name, 20_000)),
                info.name.lower(),
            )
        )
    ordered = _layout.ordered_groups(by_group, GROUP_ORDER)
    return [(group, by_group[group]) for group in ordered]


def kinds_for_family(family: str) -> tuple[str, ...]:
    """Which colormap kinds are valid for a channel family."""
    return (CYCLIC,) if str(family) == "phase" else (SEQUENTIAL, DIVERGING)


def colormaps_for_family(family: str) -> tuple[ColormapInfo, ...]:
    kinds = kinds_for_family(family)
    return tuple(info for info in list_colormaps() if info.kind in kinds)


def find_colormap(name: str) -> ColormapInfo | None:
    for info in list_colormaps(include_hidden=True, include_unavailable=True):
        if info.name == str(name):
            return info
    return None


def gray_colormap():
    """Grayscale colormap matching pyqtgraph's built-in default."""
    import pyqtgraph as pg

    return pg.ColorMap(pos=[0.0, 1.0], color=[[0, 0, 0, 255], [255, 255, 255, 255]])


def phase_colormap():
    """pyqtgraph's PAL-relaxed phase map, falling back to cyclic HSV."""
    import pyqtgraph as pg

    try:
        colormap = pg.colormap.get("PAL-relaxed")
        if colormap is not None:
            return colormap
    except Exception:
        pass
    return hsv_phase_colormap()


def hsv_phase_colormap():
    """Cyclic HSV phase colormap."""
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui

    positions = np.linspace(0.0, 1.0, 257)
    colors = []
    for pos in positions:
        qcolor = QtGui.QColor()
        qcolor.setHsvF(pos % 1.0, 1.0, 1.0, 1.0)
        colors.append([qcolor.red(), qcolor.green(), qcolor.blue(), 255])
    return pg.ColorMap(pos=positions, color=np.array(colors))


def d3_warm_colormap():
    """D3.js interpolateWarm colormap."""
    return _d3_cubehelix_colormap(start_hue=-100, end_hue=80)


def d3_cool_colormap():
    """D3.js interpolateCool colormap."""
    return _d3_cubehelix_colormap(start_hue=260, end_hue=80)


def _d3_cubehelix_colormap(start_hue, end_hue):
    # D3 uses cubehelix interpolation with long linear hue interpolation.
    import pyqtgraph as pg

    colors = []
    positions = []
    n_samples = 256

    a_const = -0.14861
    b_const = +1.78277
    c_const = -0.29227
    d_const = -0.90649
    e_const = +1.97294

    for i in range(n_samples):
        t = i / (n_samples - 1)
        hue = start_hue + t * (end_hue - start_hue)
        saturation = 0.75 + t * (1.50 - 0.75)
        lightness = 0.35 + t * (0.8 - 0.35)

        h_rad = (hue + 120) * np.pi / 180
        amp = saturation * lightness * (1 - lightness)
        cosh = np.cos(h_rad)
        sinh = np.sin(h_rad)

        red = lightness + amp * (a_const * cosh + b_const * sinh)
        green = lightness + amp * (c_const * cosh + d_const * sinh)
        blue = lightness + amp * (e_const * cosh)

        colors.append(
            (
                int(np.clip(red * 255, 0, 255)),
                int(np.clip(green * 255, 0, 255)),
                int(np.clip(blue * 255, 0, 255)),
            )
        )
        positions.append(t)

    return pg.ColorMap(pos=np.array(positions), color=np.array(colors))


_FACTORY_COLORMAPS = {
    "gray": gray_colormap,
    "PAL-relaxed": phase_colormap,
    "hsv-phase": hsv_phase_colormap,
    "d3-warm": d3_warm_colormap,
    "d3-cool": d3_cool_colormap,
}


def get_colormap(name: str):
    """Resolve a name to a pyqtgraph ColorMap (user maps shadow built-ins)."""
    import pyqtgraph as pg

    info = _load_user_cache().get(str(name))
    if info is not None and info.stops:
        return _colormap_from_stops(info.stops)
    for spec_name, _kind, _group, spec_stops in _BUILTIN_SPECS:
        if spec_name == str(name) and spec_stops is not None:
            stops = _resolve_builtin_stops(spec_stops)
            if stops is not None:
                return _colormap_from_stops(stops)
    factory = _FACTORY_COLORMAPS.get(str(name))
    if factory is not None:
        return factory()
    return pg.colormap.get(str(name))


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
        for pos, rgb in zip(positions, lut, strict=False)
    )


# ---------------------------------------------------------------------------
# User map persistence
# ---------------------------------------------------------------------------


def add_library_listener(callback) -> None:
    _listeners.add(callback)


def remove_library_listener(callback) -> None:
    """Unregister a listener added with :func:`add_library_listener` (idempotent).

    Widget-bound listeners MUST call this on teardown (see the main window's
    ``closeEvent`` and the manager dialog's); the registry is process-global, so
    a listener that outlives its widget is a leak that turns the next library
    mutation into a call on a dead Qt object.
    """
    _listeners.remove(callback)


def _notify() -> None:
    _listeners.notify()


def legacy_user_colormap_directory() -> str:
    """Where user colormaps lived before they became install-scoped.

    ``AppDataLocation`` is per-*user*, not per-install, so a scoped ``-n <name>``
    run shared one colormap library with every other scope while its operations,
    settings and sessions were correctly separate. Kept only so the one-time
    migration in :func:`user_colormap_directory` can find the old files.
    """

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


_migrated_legacy_directory = False


def user_colormap_directory() -> str:
    """Directory holding user colormap JSON (tests monkeypatch this).

    Beside this install's config, exactly where user operations live, so a
    scoped run keeps its own library. The first call per process copies an old
    unscoped library across when the new location is still empty.
    """

    global _migrated_legacy_directory
    directory = str(user_dirs.user_config_directory() / "colormaps")
    if not _migrated_legacy_directory:
        _migrated_legacy_directory = True
        try:
            user_library.migrate_directory(legacy_user_colormap_directory(), directory)
        except Exception:
            _LOGGER.warning("could not migrate the legacy colormap directory", exc_info=True)
    return directory


def user_colormap_problems() -> list[user_library.LibraryProblem]:
    """One record per colormap file this module could not read.

    A parse error used to be swallowed by a bare ``continue``, so a map the user
    had saved simply stopped existing with no message anywhere. ADR 0062's
    invariant -- every stored definition stays listable, editable and deletable
    -- applies to colormaps too (ADR 0063).
    """

    _load_user_cache()
    return list(_user_problems)


def _load_user_cache() -> dict[str, ColormapInfo]:
    global _user_cache, _user_problems
    if _user_cache is None:
        _user_cache = {}
        problems: list[user_library.LibraryProblem] = []
        directory = user_colormap_directory()
        if os.path.isdir(directory):
            for file_name in sorted(os.listdir(directory)):
                if not file_name.endswith(".json") or file_name in _RESERVED_FILES:
                    continue
                path = os.path.join(directory, file_name)
                try:
                    info = _info_from_json_file(path)
                    _user_cache[info.name] = info
                except Exception as exc:
                    _LOGGER.warning("skipping colormap %s: %s", path, exc)
                    problems.append(user_library.problem_record(path, str(exc), id_field="name"))
        _user_problems = problems
    return _user_cache


def refresh_user_colormaps() -> None:
    global _user_cache
    _user_cache = None
    _layout.invalidate()
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
    user_library.write_json_atomically(user_colormap_path(name), _info_to_payload(info))
    _load_user_cache()[name] = info
    _notify()
    return info


def user_colormap_path(name: str) -> str:
    """Where the user map called ``name`` is (or would be) stored."""

    return os.path.join(user_colormap_directory(), f"{_safe_file_name(str(name))}.json")


def delete_user_colormap(name: str) -> bool:
    cache = _load_user_cache()
    info = cache.pop(str(name), None)
    path = user_colormap_path(name)
    removed = False
    if os.path.exists(path):
        os.remove(path)
        removed = True
    if info is not None or removed:
        _notify()
    return info is not None or removed


def remove_user_colormap_file(path: str) -> bool:
    """Delete one stored colormap by *file path*, however broken it is.

    The path-keyed sibling of :func:`delete_user_colormap`: a file the loader
    could not read has no usable name, so this is the only delete that reaches
    the map the library is complaining about. Refuses paths outside the colormap
    directory, and refuses the library's own bookkeeping files.
    """

    path = os.path.abspath(path)
    directory = os.path.abspath(user_colormap_directory())
    if os.path.dirname(path) != directory or not os.path.isfile(path):
        return False
    if os.path.basename(path) in _RESERVED_FILES:
        return False
    os.remove(path)
    refresh_user_colormaps()
    return True


def update_user_colormap_file(path: str, payload: dict) -> bool:
    """Replace one stored colormap's whole payload by *file path*, then rescan.

    A whole replacement rather than a merge, for the reason ADR 0062 gives for
    operations: a repair surface owns the entire document, and merging into a
    payload the loader could not understand preserves exactly the keys the user
    is trying to remove.
    """

    path = os.path.abspath(path)
    directory = os.path.abspath(user_colormap_directory())
    if os.path.dirname(path) != directory or not os.path.isfile(path):
        return False
    if not isinstance(payload, dict):
        raise TypeError("colormap payload must be a JSON object")
    user_library.write_json_atomically(path, payload)
    refresh_user_colormaps()
    return True


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


def import_colormap_file(
    path: str, *, name: str | None = None, kind: str | None = None
) -> ColormapInfo:
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
        (float(pos), (round(r), round(g), round(b)))
        for pos, (r, g, b) in zip(positions, rgb, strict=False)
    )


# ---------------------------------------------------------------------------
# Hidden built-ins (deletable, restorable)
# ---------------------------------------------------------------------------


def hidden_builtins() -> frozenset[str]:
    return _hidden.values()


def set_builtin_hidden(name: str, hidden: bool) -> None:
    _hidden.set(name, hidden)


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
    center_zone = lightness[
        max(1, mid - max(1, len(lightness) // 4)) : mid + max(2, len(lightness) // 4)
    ]
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
    colors = [(*tuple(color), 255) for _position, color in stops]
    return pg.ColorMap(positions, colors)


def _info_to_payload(info: ColormapInfo) -> dict:
    return {
        "format": COLORMAP_FORMAT,
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
    return ColormapInfo(
        name, kind, "user", stops, "User" if builtin_group is None else builtin_group
    )


def _safe_file_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name) or "colormap"
