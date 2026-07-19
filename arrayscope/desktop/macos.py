"""macOS desktop integration: an ArrayScope.app bundle in ~/Applications.

The bundle wraps the current Python environment's ArrayScope entry point
and declares document types, so Finder offers "Open With → ArrayScope"
for supported files and the app shows up in Spotlight and the Dock.
Finder document opens arrive as Apple open-document events, which the Qt
application layer translates into QFileOpenEvent (handled by
``arrayscope.app.open_flow``).
"""

from __future__ import annotations

import plistlib
import shutil
import stat
import subprocess
from pathlib import Path

from arrayscope.desktop import IntegrationReport
from arrayscope.desktop.assets import icon_png
from arrayscope.desktop.filetypes import FILE_TYPES
from arrayscope.desktop.launcher_cmd import launcher_argv

BUNDLE_IDENTIFIER = "io.github.roosted7.arrayscope"


def bundle_path(applications_dir=None):
    base = Path(applications_dir) if applications_dir is not None else Path.home() / "Applications"
    return base / "ArrayScope.app"


def _document_types():
    return [
        {
            "CFBundleTypeName": file_type.description,
            "CFBundleTypeRole": "Viewer",
            "CFBundleTypeExtensions": [ext.lstrip(".") for ext in file_type.extensions],
            "LSHandlerRank": "Default" if file_type.owned else "Alternate",
        }
        for file_type in FILE_TYPES
    ]


def info_plist():
    from arrayscope._version import __version__

    return {
        "CFBundleName": "ArrayScope",
        "CFBundleDisplayName": "ArrayScope",
        "CFBundleIdentifier": BUNDLE_IDENTIFIER,
        "CFBundleVersion": __version__,
        "CFBundleShortVersionString": __version__,
        "CFBundlePackageType": "APPL",
        "CFBundleExecutable": "ArrayScope",
        "CFBundleIconFile": "arrayscope.icns",
        "CFBundleDocumentTypes": _document_types(),
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
    }


def launcher_script_text():
    argv = launcher_argv()
    quoted = " ".join(f'"{part}"' for part in argv)
    return f'#!/bin/sh\nexec {quoted} "$@"\n'


def install(applications_dir=None) -> IntegrationReport:
    report = IntegrationReport()
    bundle = bundle_path(applications_dir)
    contents = bundle / "Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(parents=True, exist_ok=True)

    with open(contents / "Info.plist", "wb") as plist_file:
        plistlib.dump(info_plist(), plist_file)
    report.add(f"Wrote {contents / 'Info.plist'}")

    executable = contents / "MacOS" / "ArrayScope"
    executable.write_text(launcher_script_text())
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    report.add(f"Wrote {executable}")

    _write_icns(report, contents / "Resources" / "arrayscope.icns")
    _register_bundle(report, bundle)
    report.add(f"Installed {bundle}. Finder may need a moment to index it.")
    return report


def uninstall(applications_dir=None) -> IntegrationReport:
    report = IntegrationReport()
    bundle = bundle_path(applications_dir)
    if bundle.exists():
        shutil.rmtree(bundle)
        report.add(f"Removed {bundle}")
    else:
        report.add(f"Nothing to remove at {bundle}")
    return report


def _write_icns(report, target):
    """Best-effort .icns via iconutil (ships with macOS)."""
    if shutil.which("iconutil") is None:
        report.add("note: iconutil not found; app installed without an icon")
        return
    import tempfile

    iconset_sources = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        iconset = Path(temp_dir) / "arrayscope.iconset"
        iconset.mkdir()
        for name, size in iconset_sources.items():
            source = icon_png(size)
            if source is not None:
                shutil.copyfile(source, iconset / name)
        try:
            subprocess.run(
                ["iconutil", "-c", "icns", str(iconset), "-o", str(target)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            report.add(f"Wrote {target}")
        except Exception as exc:
            report.add(f"note: iconutil failed ({exc}); app installed without an icon")


def _register_bundle(report, bundle):
    lsregister = (
        "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
        "LaunchServices.framework/Support/lsregister"
    )
    if not Path(lsregister).exists():
        return
    try:
        subprocess.run(
            [lsregister, "-f", str(bundle)], check=False, capture_output=True, timeout=60
        )
        report.add("Registered the bundle with LaunchServices")
    except Exception:
        pass
