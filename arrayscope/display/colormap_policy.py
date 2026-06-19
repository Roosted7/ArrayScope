"""Qt-free policy for selecting the active display colormap.

The rendering backends consume a LUT; this module only decides which named LUT
owns the presentation.  Keeping that choice outside Qt widgets prevents channel
changes, recipe restore, and backend switches from silently selecting different
maps.
"""

from __future__ import annotations


SCALAR_DEFAULT_COLORMAP = "gray"
PHASE_DEFAULT_COLORMAP = "PAL-relaxed"
_PHASE_CHANNELS = frozenset({"complex", "angle"})


def default_colormap_name(channel) -> str:
    """Return the deterministic default colormap for a semantic channel."""

    value = getattr(channel, "value", channel)
    return PHASE_DEFAULT_COLORMAP if str(value).lower() in _PHASE_CHANNELS else SCALAR_DEFAULT_COLORMAP


def resolved_colormap_name(channel, current_name, *, user_selected: bool) -> str:
    """Keep an explicit user choice; otherwise follow the channel default."""

    if user_selected and isinstance(current_name, str) and current_name.strip():
        return current_name.strip()
    return default_colormap_name(channel)


__all__ = [
    "PHASE_DEFAULT_COLORMAP",
    "SCALAR_DEFAULT_COLORMAP",
    "default_colormap_name",
    "resolved_colormap_name",
]
