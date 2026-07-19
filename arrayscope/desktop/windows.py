"""Windows desktop integration: HKCU file associations + Start Menu shortcut.

Everything lives under the per-user hive (``HKEY_CURRENT_USER\\Software\\
Classes``) so no elevation is required. For formats ArrayScope owns the
ProgID becomes the default handler; for shared formats (DICOM, HDF5)
ArrayScope is only added to the "Open with" list via ``OpenWithProgids``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from arrayscope.desktop import IntegrationReport
from arrayscope.desktop.assets import icon_ico
from arrayscope.desktop.filetypes import FILE_TYPES
from arrayscope.desktop.launcher_cmd import launcher_argv

_PROGID_PREFIX = "ArrayScope"


def _progid(file_type):
    return f"{_PROGID_PREFIX}.{file_type.key}"


def _open_command():
    argv = launcher_argv()
    quoted = " ".join(f'"{part}"' for part in argv)
    return f'{quoted} "%1"'


def _simple_extensions(file_type):
    # Registry associations are keyed by final extension only; compound
    # suffixes like .nii.gz resolve to .gz and are skipped.
    return [ext for ext in file_type.extensions if ext.count(".") == 1]


def install() -> IntegrationReport:
    report = IntegrationReport()
    if sys.platform != "win32":
        report.fail("Windows integration requested on a non-Windows platform")
        return report
    import winreg

    command = _open_command()
    ico = icon_ico()
    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r"Software\Classes",
        0,
        winreg.KEY_CREATE_SUB_KEY | winreg.KEY_SET_VALUE,
    ) as classes:
        for file_type in FILE_TYPES:
            progid = _progid(file_type)
            with winreg.CreateKey(classes, progid) as key:
                winreg.SetValue(key, "", winreg.REG_SZ, file_type.description)
                if ico is not None:
                    with winreg.CreateKey(key, "DefaultIcon") as icon_key:
                        winreg.SetValue(icon_key, "", winreg.REG_SZ, str(ico))
                with winreg.CreateKey(key, r"shell\open\command") as cmd_key:
                    winreg.SetValue(cmd_key, "", winreg.REG_SZ, command)
            for extension in _simple_extensions(file_type):
                with winreg.CreateKey(classes, extension) as ext_key:
                    if file_type.owned:
                        winreg.SetValue(ext_key, "", winreg.REG_SZ, progid)
                    with winreg.CreateKey(ext_key, "OpenWithProgids") as open_with:
                        winreg.SetValueEx(open_with, progid, 0, winreg.REG_NONE, b"")
            report.add(f"Registered {progid} for {', '.join(_simple_extensions(file_type))}")

    _create_start_menu_shortcut(report)
    _notify_shell()
    report.add("Windows integration installed (per-user, no elevation needed).")
    return report


def uninstall() -> IntegrationReport:
    report = IntegrationReport()
    if sys.platform != "win32":
        report.fail("Windows integration requested on a non-Windows platform")
        return report
    import winreg

    def delete_tree(root, subkey):
        try:
            with winreg.OpenKey(root, subkey, 0, winreg.KEY_ALL_ACCESS) as key:
                while True:
                    try:
                        child = winreg.EnumKey(key, 0)
                    except OSError:
                        break
                    delete_tree(key, child)
            winreg.DeleteKey(root, subkey)
        except FileNotFoundError:
            pass

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, r"Software\Classes", 0, winreg.KEY_ALL_ACCESS
    ) as classes:
        for file_type in FILE_TYPES:
            progid = _progid(file_type)
            delete_tree(classes, progid)
            for extension in _simple_extensions(file_type):
                try:
                    with winreg.OpenKey(classes, extension, 0, winreg.KEY_ALL_ACCESS) as ext_key:
                        if file_type.owned:
                            try:
                                if winreg.QueryValue(ext_key, "") == progid:
                                    winreg.SetValue(ext_key, "", winreg.REG_SZ, "")
                            except OSError:
                                pass
                        try:
                            with winreg.OpenKey(
                                ext_key, "OpenWithProgids", 0, winreg.KEY_ALL_ACCESS
                            ) as open_with:
                                winreg.DeleteValue(open_with, progid)
                        except FileNotFoundError:
                            pass
                except FileNotFoundError:
                    pass
        report.add("Removed ArrayScope ProgIDs and associations")

    shortcut = _start_menu_shortcut_path()
    try:
        shortcut.unlink()
        report.add(f"Removed {shortcut}")
    except FileNotFoundError:
        pass
    _notify_shell()
    return report


def _start_menu_shortcut_path():
    appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "ArrayScope.lnk"


def _create_start_menu_shortcut(report):
    shortcut = _start_menu_shortcut_path()
    shortcut.parent.mkdir(parents=True, exist_ok=True)
    argv = launcher_argv()
    target = argv[0]
    arguments = " ".join(argv[1:])
    ico = icon_ico()
    script = (
        "$shell = New-Object -ComObject WScript.Shell; "
        f"$link = $shell.CreateShortcut('{shortcut}'); "
        f"$link.TargetPath = '{target}'; "
        f"$link.Arguments = '{arguments}'; "
        "$link.Description = 'Interactive N-dimensional array viewer'; "
        + (f"$link.IconLocation = '{ico}'; " if ico is not None else "")
        + "$link.Save()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            capture_output=True,
            timeout=60,
        )
        report.add(f"Created Start Menu shortcut: {shortcut}")
    except Exception as exc:
        report.add(f"note: could not create Start Menu shortcut: {exc}")


def _notify_shell():
    """Tell Explorer the associations changed (SHCNE_ASSOCCHANGED)."""
    try:
        import ctypes

        ctypes.windll.shell32.SHChangeNotify(0x08000000, 0x0000, None, None)
    except Exception:
        pass
