"""Optional LLM high-level policy for the generic spec path.

The LLM **never emits grid moves**. It only selects which remaining sub-goal of
the task's predicate goal the agent should pursue *next*; the spec oracle's BFS
then executes the actual movement (`spec._actions_for_subgoal`). Because the
choices are restricted to the task's real sub-goals, the policy cannot
hallucinate an illegal action — it can only (re)order legitimate ones.

With no `ANTHROPIC_API_KEY` (or no `anthropic` SDK) it falls back to a
deterministic mock that picks the next unsatisfied sub-goal in order, so
`--agent llm` always runs, keyless and offline.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from . import predicates as P

_SYSTEM = """You are the HIGH-LEVEL policy for a 2D gridworld agent.
You do NOT control movement — a BFS path planner handles that. You only choose
which remaining sub-goal the agent should pursue NEXT, to satisfy the task in the
intended order. Reply with ONLY the number of the sub-goal to do next."""


def ordered_subgoals(goal: Dict[str, Any]) -> List[Dict[str, Any]]:
    pred = P.goal_predicate(goal)
    if pred.get("op") == "ordered":
        return list(pred["of"])
    if pred.get("op") == "all":
        for c in pred["of"]:
            if c.get("op") == "ordered":
                return list(c["of"])
        return list(pred["of"])
    return [pred]


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader so ANTHROPIC_API_KEY can live in a .env file."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


def _observation(env, remaining: List[Dict[str, Any]]) -> str:
    lines = [f"Agent at {tuple(env.agent_pos)}. Inventory: {env.inventory or 'empty'}.",
             "Objects:"]
    for o in env.objects.values():
        if not o.visible():
            continue
        lines.append(f"  - {o.id} at {tuple(o.position)} [{', '.join(sorted(o.components))}]")
    lines.append("Remaining sub-goals (choose the number to do NEXT):")
    for i, s in enumerate(remaining, 1):
        lines.append(f"  {i}. {P.describe(s)}")
    return "\n".join(lines)


class LLMHighLevelPolicy:
    """Chooses the next sub-goal; Anthropic-backed if a key is present, else mock."""

    def __init__(self, model: str = "claude-opus-4-8", allow_api: bool = True,
                 verbose: bool = False):
        self.model = model
        self._client = None
        self.mode = "mock"
        if allow_api:
            _load_dotenv()
            if os.environ.get("ANTHROPIC_API_KEY"):
                try:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                    self.mode = "anthropic"
                except Exception:  # noqa: BLE001 - SDK missing / init failure
                    self._client = None
        if verbose:
            print(f"[llm-agent] {'using Anthropic ' + self.model if self._client else 'no ANTHROPIC_API_KEY — keyless mock high-level policy'}")

    def next_subgoal(self, env, goal: Dict[str, Any],
                     record: "P.RunRecord") -> Optional[Dict[str, Any]]:
        remaining = [s for s in ordered_subgoals(goal) if not P.evaluate(record, s)]
        if not remaining:
            return None
        if self._client is None:
            return remaining[0]                      # mock: next unsatisfied, in order
        try:
            resp = self._client.messages.create(
                model=self.model, max_tokens=8, system=_SYSTEM,
                messages=[{"role": "user", "content": _observation(env, remaining)}],
            )
            idx = int(re.search(r"\d+", resp.content[0].text).group()) - 1
            if 0 <= idx < len(remaining):
                return remaining[idx]
        except Exception:  # noqa: BLE001 - any API/parse error -> safe fallback
            pass
        return remaining[0]
