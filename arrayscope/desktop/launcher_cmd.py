"""Resolve how the desktop shell should launch ArrayScope.

Desktop entries and file associations need an absolute command that works
outside any activated environment (login shells don't source conda/venv
activation). Prefer the installed ``arrayscope`` console script next to
the current interpreter; fall back to ``<python> -m arrayscope``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def launcher_argv():
    """Absolute argv (list of str) that starts the ArrayScope CLI."""
    exe_dir = Path(sys.executable).parent
    names = ("arrayscope.exe", "arrayscope") if sys.platform == "win32" else ("arrayscope",)
    for candidate_dir in (exe_dir, exe_dir / "Scripts"):
        for name in names:
            script = candidate_dir / name
            if script.exists():
                return [str(script)]
    found = shutil.which("arrayscope")
    if found:
        return [found]
    return [sys.executable, "-m", "arrayscope"]
