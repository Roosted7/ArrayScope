"""Visible-output render generation guard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderGeneration:
    current: int = 0
    last_reason: str = ""

    def advance(self, reason: str = "") -> int:
        self.current += 1
        self.last_reason = str(reason or "")
        return self.current

    def capture(self) -> int:
        return int(self.current)

    def is_current(self, generation: int) -> bool:
        return int(generation) == int(self.current)
