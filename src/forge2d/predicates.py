"""The predicate verification DSL and its trusted interpreter.

Task success is no longer a hardcoded "ordered subsequence of task strings".
It is a **predicate tree** authored in the spec file and evaluated by a small,
whitelisted interpreter over the run record (the object model + the trajectory).
There is no `eval`, no `exec`, no arbitrary callables — every operator is a named
Python function in `_OPS`, so a spec authored by an LLM can define *what counts
as success* without being able to execute code. That is the "trusted
interpreter": data in, boolean out.

Predicate node shape (JSON/dict):

    {"op": "<name>", <args...>}

Supported operators
-------------------
    exists(object)                      object is declared in the world
    has_component(object, component)    object declares that component
    reachable(agent, target)            target cell is reachable in principle
    acquired(object)                    object entered the inventory
    placed_in(object, receptacle)       object was placed into the receptacle
    state_changed(object, attr,         object.attr transitioned from->to
                  **from**, to)
    at(agent, target)                   agent ended on target cell / region
    no_illegal_actions()                no blocked / invalid actions occurred
    ordered(of=[...])                   sub-predicates became true in order

Composition
    all(of=[...])  any(of=[...])  not(of=<pred>)

Top-level goal wrapper
    reward_iff(of=<pred>)   success is defined as exactly <pred>. Independent
                            loose-reward checks live in spec_verifier.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .generic import WALL, GenericEnvironment, fmt_value
from .schemas import VerificationReport


# --------------------------------------------------------------------------- #
# Run record: the immutable evidence a predicate is evaluated against
# --------------------------------------------------------------------------- #
class RunRecord:
    """Bundles the world + per-step trajectory so predicates can be evaluated
    against any *prefix* of the run (needed for `ordered`)."""

    def __init__(self, env: GenericEnvironment, trace: List[Dict[str, Any]],
                 spec: Dict[str, Any]):
        self.env = env
        self.spec = spec
        self.start = list(env.start)
        self.step_events: List[List[str]] = [s.get("events", []) or [] for s in trace]
        self.step_inventory: List[List[str]] = [s.get("inventory", []) or [] for s in trace]
        self.step_position: List[List[int]] = [list(s.get("position", [])) for s in trace]
        self.n = len(trace)
        # A pristine copy of the world for static spatial reachability checks.
        self._initial_env = GenericEnvironment(spec)

    def events_upto(self, horizon: int) -> List[str]:
        flat: List[str] = []
        for evs in self.step_events[:horizon]:
            flat.extend(evs)
        return flat

    def position_at(self, horizon: int) -> List[int]:
        if horizon <= 0:
            return list(self.start)
        return list(self.step_position[horizon - 1])

    def object_cell(self, obj_id: str) -> Optional[List[int]]:
        o = self.env.objects.get(obj_id)
        return list(o.position) if o is not None else None


# --------------------------------------------------------------------------- #
# Operator implementations  (ctx=RunRecord, node=dict, horizon=int) -> bool
# --------------------------------------------------------------------------- #
def _op_exists(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    return node["object"] in ctx.env.objects


def _op_has_component(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    o = ctx.env.objects.get(node["object"])
    return o is not None and o.has(node["component"])


def _op_reachable(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    """Is `target` reachable from the agent start in principle? Conditional
    barriers (doors/bridges that can be opened) are treated as passable;
    permanent barriers, walls and hazards are not."""
    env = ctx._initial_env
    cell = env.objects.get(node["target"])
    if cell is None:
        return False
    goal = {cell.cell}

    def passable(x: int, y: int) -> bool:
        if env.is_wall(x, y):
            return False
        o = env.object_at(x, y)
        if o is None:
            return True
        if o.has("hazard"):
            return False
        if o.has("barrier") and "while" not in o.components.get("barrier", {}):
            return False  # permanent wall-like barrier
        return True

    return env.bfs(tuple(env.start), goal, passable=passable) is not None


def _op_acquired(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    return f"pickup {node['object']}" in ctx.events_upto(horizon)


def _op_placed_in(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    return f"place {node['object']} in {node['receptacle']}" in ctx.events_upto(horizon)


def _op_state_changed(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    obj, attr = node["object"], node["attr"]
    to = node["to"]
    frm = node.get("from")
    # If an initial value is asserted, it must match the declared world state.
    if frm is not None:
        o = ctx.env.objects.get(obj)
        declared = next((d for d in ctx.spec.get("objects", []) if d["id"] == obj), {})
        initial = declared.get("state", {}).get(attr)
        if initial != frm:
            return False
    return f"state {obj} {attr}={fmt_value(to)}" in ctx.events_upto(horizon)


def _op_at(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    pos = ctx.position_at(horizon)
    if not pos:
        return False
    fx, fy = pos[0], pos[1]
    target = node["target"]
    if isinstance(target, str):
        cell = ctx.object_cell(target)
        return cell is not None and [fx, fy] == cell
    if len(target) == 2:
        return [fx, fy] == list(target)
    x0, y0, x1, y1 = target  # region
    return min(x0, x1) <= fx <= max(x0, x1) and min(y0, y1) <= fy <= max(y0, y1)


def _op_no_illegal_actions(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    for ev in ctx.events_upto(horizon):
        if ev.startswith("blocked") or ev.startswith("invalid"):
            return False
    return True


def _target_cell(ctx: RunRecord, target) -> Optional[List[int]]:
    return ctx.object_cell(target) if isinstance(target, str) else list(target)


def _op_near(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    """Two objects (or an object and a cell) are within Manhattan `within` tiles.
    Grounds relations like 'the can is on/at the table'."""
    a = ctx.object_cell(node["object"])
    b = _target_cell(ctx, node["target"])
    if a is None or b is None:
        return False
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= node.get("within", 1)


def _op_near_wall(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    """An object lies within Manhattan `within` tiles of a wall cell. Grounds
    'near the wall'."""
    c = ctx.object_cell(node["object"])
    if c is None:
        return False
    within = node.get("within", 1)
    env = ctx.env
    for y in range(env.height):
        for x in range(env.width):
            if env.terrain[y][x] == WALL and abs(c[0] - x) + abs(c[1] - y) <= within:
                return True
    return False


def _first_true(ctx: RunRecord, node: Dict) -> Optional[int]:
    """Smallest horizon (1..n) at which `node` evaluates true, or None."""
    for h in range(1, ctx.n + 1):
        if evaluate(ctx, node, h):
            return h
    return None


def _op_ordered(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    """Each sub-predicate must first become true strictly after the previous
    one (within the considered horizon)."""
    last = 0
    for child in node["of"]:
        # Restrict first-true search to the active horizon.
        h = next((k for k in range(1, horizon + 1) if evaluate(ctx, child, k)), None)
        if h is None or h <= last:
            return False
        last = h
    return True


def _op_all(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    return all(evaluate(ctx, c, horizon) for c in node["of"])


def _op_any(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    return any(evaluate(ctx, c, horizon) for c in node["of"])


def _op_not(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    return not evaluate(ctx, node["of"], horizon)


def _op_reward_iff(ctx: RunRecord, node: Dict, horizon: int) -> bool:
    return evaluate(ctx, node["of"], horizon)


_OPS = {
    "exists": _op_exists,
    "has_component": _op_has_component,
    "reachable": _op_reachable,
    "acquired": _op_acquired,
    "placed_in": _op_placed_in,
    "state_changed": _op_state_changed,
    "at": _op_at,
    "near": _op_near,
    "near_wall": _op_near_wall,
    "no_illegal_actions": _op_no_illegal_actions,
    "ordered": _op_ordered,
    "all": _op_all,
    "any": _op_any,
    "not": _op_not,
    "reward_iff": _op_reward_iff,
}


def evaluate(ctx: RunRecord, node: Dict[str, Any], horizon: Optional[int] = None) -> bool:
    """Evaluate a predicate node against `ctx` up to `horizon` trace steps
    (default: the whole run)."""
    if horizon is None:
        horizon = ctx.n
    op = node.get("op")
    if op not in _OPS:
        raise ValueError(f"unknown predicate op {op!r}; valid ops: {sorted(_OPS)}")
    return _OPS[op](ctx, node, horizon)


# --------------------------------------------------------------------------- #
# Goal handling + human-readable reporting
# --------------------------------------------------------------------------- #
def goal_predicate(goal: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap a `reward_iff` wrapper to the underlying success predicate."""
    return goal["of"] if goal.get("op") == "reward_iff" else goal


def describe(node: Dict[str, Any]) -> str:
    op = node.get("op")
    if op in ("ordered", "all", "any"):
        inner = ", ".join(describe(c) for c in node["of"])
        return f"{op}({inner})"
    if op == "not":
        return f"not({describe(node['of'])})"
    if op == "reward_iff":
        return f"reward_iff({describe(node['of'])})"
    args = {k: v for k, v in node.items() if k != "op"}
    parts = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{op}({parts})"


def verify(record: RunRecord, goal: Dict[str, Any]) -> VerificationReport:
    """Evaluate the goal predicate tree and bundle a console-friendly report.

    Each top-level conjunct of the success predicate becomes its own check.
    Reward-hacking checks are reported by spec_verifier.verify_task, where the
    strict predicate is compared against loose outcome-only baselines.
    """
    report = VerificationReport()
    success_pred = goal_predicate(goal)

    # Decompose the success predicate into per-sub-task checks so the report
    # reads as a curriculum: an `all([...])` becomes its conjuncts, an
    # `ordered([...])` becomes its ordered sub-goals (each a checkable sub-task).
    op = success_pred.get("op")
    if op in ("all", "ordered"):
        conjuncts = success_pred["of"]
    else:
        conjuncts = [success_pred]
    for pred in conjuncts:
        ok = evaluate(record, pred)
        report.add(describe(pred), ok, "" if ok else "predicate false over trace")

    overall = evaluate(record, success_pred)
    if goal.get("op") == "reward_iff":
        report.add(
            "reward_iff: strict predicate success",
            overall,
            "" if overall else "strict success predicate is false",
        )
    return report
