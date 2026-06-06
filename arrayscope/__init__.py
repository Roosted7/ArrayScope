"""Callable public package API for ArrayScope."""

from __future__ import annotations

import sys
import types

__version__ = "0.0.1"
_arrayscope = None


class _CallableArrayScopeModule(types.ModuleType):
    def __call__(self, data, *args, **kwargs):
        global _arrayscope
        if _arrayscope is None:
            from arrayscope.app.launch import arrayscope as _arrayscope
        return _arrayscope(data, *args, **kwargs)


sys.modules[__name__].__class__ = _CallableArrayScopeModule
