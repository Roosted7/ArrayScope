"""Callable public package API for ArrayScope."""

from __future__ import annotations

import sys
import types

from arrayscope.app.launch import arrayscope as _arrayscope

__version__ = "0.0.1"


class _CallableArrayScopeModule(types.ModuleType):
    def __call__(self, data, *args, **kwargs):
        return _arrayscope(data, *args, **kwargs)


sys.modules[__name__].__class__ = _CallableArrayScopeModule
