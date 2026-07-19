"""Linux (XDG) desktop integration: desktop entry, MIME types, icons.

Everything is installed per-user under ``$XDG_DATA_HOME`` (default
``~/.local/share``) following the freedesktop.org specs, so no root is
needed and ``uninstall()`` can remove exactly what was written.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from arrayscope.desktop import IntegrationReport
from arrayscope.desktop.assets import ICON_SIZES, icon_png, icon_svg
from arrayscope.desktop.filetypes import all_mime_types, owned_types
from arrayscope.desktop.launcher_cmd import launcher_argv

DESKTOP_FILE_NAME = "arrayscope.desktop"
MIME_PACKAGE_NAME = "arrayscope.xml"
ICON_NAME = "arrayscope"


def data_home():
    return Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))


def _desktop_exec_quote(part):
    """Quote one Exec argument per the desktop-entry spec."""
    if not any(c in part for c in " \t\n\"'\\><~|&;$*?#()`"):
        return part
    escaped = part.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    return f'"{escaped}"'


def desktop_entry_text():
    argv = launcher_argv()
    exec_line = " ".join(_desktop_exec_quote(part) for part in argv) + " %F"
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        "Name=ArrayScope",
        "GenericName=Array Viewer",
        "Comment=Interactive N-dimensional array viewer",
        f"TryExec={argv[0]}",
        f"Exec={exec_line}",
        "Terminal=false",
        f"Icon={ICON_NAME}",
        "Categories=Science;DataVisualization;Viewer;",
        "MimeType=" + ";".join(all_mime_types()) + ";",
        "StartupNotify=true",
        "StartupWMClass=ArrayScope",
        "Keywords=numpy;array;viewer;fft;mri;nifti;dicom;bart;",
    ]
    return "\n".join(lines) + "\n"


def mime_package_text():
    """shared-mime-info package defining the MIME types ArrayScope owns."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">',
    ]
    for file_type in owned_types():
        parts.append(f'  <mime-type type="{escape(file_type.mime)}">')
        parts.append(f"    <comment>{escape(file_type.description)}</comment>")
        parts.append('    <sub-class-of type="application/octet-stream"/>')
        parts.append(f'    <icon name="{ICON_NAME}"/>')
        for extension in file_type.extensions:
            parts.append(f'    <glob pattern="*{escape(extension)}"/>')
            if extension != extension.upper():
                parts.append(f'    <glob pattern="*{escape(extension.upper())}"/>')
        parts.append("  </mime-type>")
    parts.append("</mime-info>")
    return "\n".join(parts) + "\n"


def _install_paths(base):
    paths = {
        "desktop": base / "applications" / DESKTOP_FILE_NAME,
        "mime": base / "mime" / "packages" / MIME_PACKAGE_NAME,
        "icon_svg": base / "icons" / "hicolor" / "scalable" / "apps" / f"{ICON_NAME}.svg",
    }
    for size in ICON_SIZES:
        paths[f"icon_{size}"] = (
            base / "icons" / "hicolor" / f"{size}x{size}" / "apps" / f"{ICON_NAME}.png"
        )
    return paths


def _run_quietly(report, argv):
    tool = argv[0]
    if shutil.which(tool) is None:
        report.add(f"note: {tool} not found; skipped (associations may need a re-login)")
        return
    try:
        subprocess.run(argv, check=False, capture_output=True, timeout=60)
    except Exception as exc:
        report.add(f"note: {tool} failed: {exc}")


def install(base=None) -> IntegrationReport:
    report = IntegrationReport()
    base = Path(base) if base is not None else data_home()
    paths = _install_paths(base)

    paths["desktop"].parent.mkdir(parents=True, exist_ok=True)
    paths["desktop"].write_text(desktop_entry_text())
    report.add(f"Wrote {paths['desktop']}")

    paths["mime"].parent.mkdir(parents=True, exist_ok=True)
    paths["mime"].write_text(mime_package_text())
    report.add(f"Wrote {paths['mime']}")

    svg = icon_svg()
    if svg is not None:
        paths["icon_svg"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(svg, paths["icon_svg"])
        report.add(f"Wrote {paths['icon_svg']}")
    installed_png = 0
    for size in ICON_SIZES:
        png = icon_png(size)
        if png is None:
            continue
        target = paths[f"icon_{size}"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(png, target)
        installed_png += 1
    report.add(f"Installed {installed_png} PNG icon sizes into {base / 'icons' / 'hicolor'}")

    _refresh_databases(report, base)

    # Only claim to be the *default* opener for formats ArrayScope owns.
    # xdg-mime edits ~/.config/mimeapps.list (not under `base`), so skip it
    # for prefix installs (tests, staging).
    if base != data_home():
        report.add("note: prefix install; default-handler registration skipped")
    elif shutil.which("xdg-mime") is not None:
        for file_type in owned_types():
            _run_quietly(report, ["xdg-mime", "default", DESKTOP_FILE_NAME, file_type.mime])
        report.add(
            "Set ArrayScope as default handler for: " + ", ".join(t.mime for t in owned_types())
        )
    else:
        report.add("note: xdg-mime not found; defaults not set")

    report.add(
        "Desktop integration installed. You may need to restart your "
        "file manager or re-login for icons to appear everywhere."
    )
    return report


def uninstall(base=None) -> IntegrationReport:
    report = IntegrationReport()
    base = Path(base) if base is not None else data_home()
    removed = 0
    for path in _install_paths(base).values():
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            report.fail(f"could not remove {path}: {exc}")
    report.add(f"Removed {removed} installed files from {base}")
    if base == data_home():
        _remove_default_handler_entries(report)
    _refresh_databases(report, base)
    return report


def _remove_default_handler_entries(report):
    """Drop `<mime>=arrayscope.desktop` lines from ~/.config/mimeapps.list."""
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    mimeapps = config_home / "mimeapps.list"
    try:
        lines = mimeapps.read_text().splitlines(keepends=True)
    except OSError:
        return
    kept = [line for line in lines if f"={DESKTOP_FILE_NAME}" not in line]
    if len(kept) != len(lines):
        try:
            mimeapps.write_text("".join(kept))
            report.add(f"Removed {len(lines) - len(kept)} default-handler entries from {mimeapps}")
        except OSError as exc:
            report.fail(f"could not update {mimeapps}: {exc}")


def _refresh_databases(report, base):
    _run_quietly(report, ["update-desktop-database", str(base / "applications")])
    _run_quietly(report, ["update-mime-database", str(base / "mime")])
    _run_quietly(report, ["gtk-update-icon-cache", "-f", "-t", str(base / "icons" / "hicolor")])
