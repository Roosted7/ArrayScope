"""Declarative ViewState → widget bindings (roadmap Y3).

One binder owns view-state mirroring. Each control registers a binding once,
where the control is created; sync entry points stop enumerating widgets.
Controls emit intent through their own signals only — the binder is the single
place that writes widget state from ``ViewState``, always with the bound
widgets' signals blocked, and only when the derived value actually changed.

A binding's ``read`` returns a comparable snapshot of everything the ``apply``
callable mirrors. Code that mutates a bound widget outside the binder must
call :meth:`ViewStateBinder.forget` so change detection cannot skip the next
re-apply.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

_UNSET = object()


@dataclass(frozen=True)
class StateBinding:
    name: str
    read: Callable[[object], object]
    apply: Callable[[object], None]
    widgets: tuple = ()
    on_demand: bool = False


class ViewStateBinder:
    def __init__(self) -> None:
        self._bindings: dict[str, StateBinding] = {}
        self._applied: dict[str, object] = {}

    def bind(self, name: str, *, read, apply, widgets=(), on_demand: bool = False) -> None:
        """Register (or replace) one control's ViewState mirror.

        ``on_demand`` bindings are interactive fast paths: they run only when
        named explicitly, never in a full pass (a broader binding covers them
        there).
        """

        name = str(name)
        self._bindings[name] = StateBinding(
            name=name, read=read, apply=apply, widgets=tuple(widgets), on_demand=bool(on_demand)
        )
        self._applied.pop(name, None)

    def forget(self) -> None:
        """Drop change-detection state; the next sync re-applies everything."""

        self._applied.clear()

    def sync(self, window, *, names=None, force: bool = False) -> int:
        """Apply registered bindings whose derived value changed.

        ``names`` limits the pass to specific bindings (interactive fast
        paths); ``force`` reapplies even when the value is unchanged.
        Returns the number of bindings applied.
        """

        if names is None:
            selected = [binding for binding in self._bindings.values() if not binding.on_demand]
        else:
            selected = [self._bindings[name] for name in names if name in self._bindings]
        applied = 0
        for binding in selected:
            value = binding.read(window)
            if not force and self._applied.get(binding.name, _UNSET) == value:
                continue
            blocked = []
            for widget in binding.widgets:
                try:
                    widget.blockSignals(True)
                except RuntimeError:
                    continue
                blocked.append(widget)
            try:
                binding.apply(value)
            finally:
                for widget in blocked:
                    with contextlib.suppress(RuntimeError):
                        widget.blockSignals(False)
            self._applied[binding.name] = value
            applied += 1
        return applied


__all__ = ["StateBinding", "ViewStateBinder"]
