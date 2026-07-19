"""Desktop-shell integration: generated artifacts and install/uninstall."""

import xml.etree.ElementTree as ET
from pathlib import Path

from arrayscope.desktop import linux, macos, windows
from arrayscope.desktop.assets import application_icon_path, icon_ico, icon_png, icon_svg
from arrayscope.desktop.filetypes import FILE_TYPES, all_mime_types, owned_types
from arrayscope.desktop.launcher_cmd import launcher_argv


def test_icon_assets_are_packaged():
    assert icon_svg() is not None
    assert icon_ico() is not None
    for size in (16, 32, 256):
        assert icon_png(size) is not None
    assert application_icon_path().suffix == ".png"


def test_launcher_argv_is_absolute_or_module_fallback():
    argv = launcher_argv()
    assert argv
    assert Path(argv[0]).is_absolute()
    if len(argv) > 1:
        assert argv[1:] == ["-m", "arrayscope"]


def test_file_type_catalogue_covers_supported_formats():
    extensions = {ext for t in FILE_TYPES for ext in t.extensions}
    assert {
        ".npy",
        ".npz",
        ".cfl",
        ".rec",
        ".nii",
        ".nii.gz",
        ".mat",
        ".dcm",
        ".h5",
        ".hdf5",
    } <= extensions
    assert len(set(all_mime_types())) == len(FILE_TYPES)
    assert all(t.mime.startswith("application/x-") for t in owned_types())


# --- Linux -----------------------------------------------------------------


def test_linux_desktop_entry_content():
    text = linux.desktop_entry_text()
    assert text.startswith("[Desktop Entry]\n")
    assert "Exec=" in text
    assert " %F" in text
    assert "Icon=arrayscope" in text
    assert "MimeType=" + ";".join(all_mime_types()) + ";" in text
    assert "Terminal=false" in text


def test_linux_mime_package_is_valid_xml_with_globs():
    root = ET.fromstring(linux.mime_package_text())
    ns = "{http://www.freedesktop.org/standards/shared-mime-info}"
    types = {el.get("type") for el in root.findall(f"{ns}mime-type")}
    assert types == {t.mime for t in owned_types()}
    globs = {g.get("pattern") for g in root.iter(f"{ns}glob")}
    assert {"*.npy", "*.cfl", "*.rec", "*.nii", "*.nii.gz"} <= globs


def test_linux_install_and_uninstall_roundtrip(tmp_path):
    report = linux.install(tmp_path)
    assert report.ok
    desktop_file = tmp_path / "applications" / "arrayscope.desktop"
    mime_file = tmp_path / "mime" / "packages" / "arrayscope.xml"
    assert desktop_file.exists()
    assert mime_file.exists()
    assert (tmp_path / "icons" / "hicolor" / "scalable" / "apps" / "arrayscope.svg").exists()
    assert (tmp_path / "icons" / "hicolor" / "256x256" / "apps" / "arrayscope.png").exists()
    # Prefix installs must not touch the real ~/.config/mimeapps.list.
    assert any("prefix install" in line for line in report.lines)

    report = linux.uninstall(tmp_path)
    assert report.ok
    assert not desktop_file.exists()
    assert not mime_file.exists()


def test_linux_exec_quoting_handles_spaces():
    assert linux._desktop_exec_quote("/plain/path") == "/plain/path"
    quoted = linux._desktop_exec_quote("/path with space/arrayscope")
    assert quoted.startswith('"')
    assert quoted.endswith('"')


# --- Windows / macOS artifact generation (platform-independent parts) ------


def test_windows_open_command_quotes_argv():
    command = windows._open_command()
    assert command.endswith(' "%1"')
    assert command.startswith('"')


def test_windows_compound_suffixes_are_skipped():
    nii = next(t for t in FILE_TYPES if t.key == "nii")
    assert windows._simple_extensions(nii) == [".nii"]


def test_macos_info_plist_declares_document_types():
    plist = macos.info_plist()
    assert plist["CFBundleIdentifier"] == macos.BUNDLE_IDENTIFIER
    extensions = {
        ext for entry in plist["CFBundleDocumentTypes"] for ext in entry["CFBundleTypeExtensions"]
    }
    assert {"npy", "cfl", "rec", "nii", "dcm"} <= extensions


def test_macos_launcher_script_is_posix_shell():
    text = macos.launcher_script_text()
    assert text.startswith("#!/bin/sh\n")
    assert 'exec "' in text
    assert text.rstrip().endswith('"$@"')
