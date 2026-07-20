"""Main window package."""

from arrayscope.window.domain import Domain

__all__ = ["ArrayScopeWindow", "Domain"]


def __getattr__(name):
    if name == "ArrayScopeWindow":
        from arrayscope.window.main import ArrayScopeWindow

        return ArrayScopeWindow
    raise AttributeError(name)
