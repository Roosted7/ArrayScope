"""Viewport update policy for 2D image display."""

from __future__ import annotations

from enum import Enum


class ViewportPolicy(Enum):
    PRESERVE = "preserve"
    FIT_ONCE = "fit_once"
    RESET_FOR_NEW_SHAPE = "reset_for_new_shape"

