"""Pure helpers for line profile display policy."""

from __future__ import annotations


def profile_y_range(mode: str, image_levels):
    if mode == "match_image" and image_levels is not None:
        low, high = image_levels
        return (float(low), float(high))
    return None

