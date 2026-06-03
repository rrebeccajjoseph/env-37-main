"""Shared data contracts for Forge-2D.

Two things every module agrees on:
  * the action vocabulary (move/pickup/use/wait), and
  * CheckResult / VerificationReport — the uniform output of every verifier.

Positions are [x, y] with x = column, y = row; grids are indexed grid[y][x].
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
# Action vocabulary
# --------------------------------------------------------------------------- #
MOVE_UP = "move_up"
MOVE_DOWN = "move_down"
MOVE_LEFT = "move_left"
MOVE_RIGHT = "move_right"
PICKUP = "pickup"
USE = "use"
WAIT = "wait"

ACTIONS = [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, USE, WAIT]

MOVE_DELTAS = {
    MOVE_UP: (0, -1),
    MOVE_DOWN: (0, 1),
    MOVE_LEFT: (-1, 0),
    MOVE_RIGHT: (1, 0),
}


# --------------------------------------------------------------------------- #
# Verification result types (used by spec_verifier, predicates, spec_exploit)
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def line(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        extra = f" — {self.detail}" if self.detail else ""
        return f"[{tag}] {self.name}{extra}"


@dataclass
class VerificationReport:
    """A bundle of CheckResults with an overall pass flag."""
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> CheckResult:
        c = CheckResult(name, passed, detail)
        self.checks.append(c)
        return c

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def render(self) -> str:
        return "\n".join(c.line() for c in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}
