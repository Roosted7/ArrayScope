"""Native desktop-shell integration (menu entries, icons, file types).

``arrayscope --install-desktop`` registers ArrayScope with the current
platform's shell so array files can be opened from the file manager and
the app appears in menus/launchers:

- **Linux**: XDG desktop entry, shared-mime-info definitions for the
  formats ArrayScope owns, and hicolor icons under ``$XDG_DATA_HOME``.
- **Windows**: per-user (HKCU) file associations, ProgIDs with icons,
  and a Start Menu shortcut. No administrator rights needed.
- **macOS**: an ``ArrayScope.app`` bundle in ``~/Applications`` declaring
  document types, launched through the current Python environment.

``--uninstall-desktop`` reverses the registration. Both operations are
per-user and idempotent.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field


@dataclass
class IntegrationReport:
    """Human-readable outcome of an install/uninstall run."""

    ok: bool = True
    lines: list = field(default_factory=list)

    def add(self, message):
        self.lines.append(str(message))

    def fail(self, message):
        self.ok = False
        self.lines.append(f"ERROR: {message}")


def _backend():
    system = platform.system()
    if system == "Linux":
        from arrayscope.desktop import linux as backend
    elif system == "Windows":
        from arrayscope.desktop import windows as backend
    elif system == "Darwin":
        from arrayscope.desktop import macos as backend
    else:
        raise RuntimeError(f"Desktop integration is not supported on {system}")
    return backend


def install_desktop_integration() -> IntegrationReport:
    return _backend().install()


def uninstall_desktop_integration() -> IntegrationReport:
    return _backend().uninstall()


__all__ = [
    "IntegrationReport",
    "install_desktop_integration",
    "uninstall_desktop_integration",
]
