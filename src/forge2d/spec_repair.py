"""Curriculum repair for the generic component/predicate path.

When a generated (or authored) component world is NOT solvable, localize the
first failing ordered sub-goal, identify the barrier, and **repair the
generation** — free a target spawned inside a wall, or carve a corridor toward
an unreachable target — then re-solve until the oracle completes the goal.
Deterministic; no RL, no LLM.

    solve world -> goal predicate satisfied?  -> done
                -> else: first failing sub-goal -> target cell
                      -> free target / carve corridor toward it -> retry
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from . import predicates as P
from . import spec_verifier
from .components import is_permanent_barrier
from .generic import FLOOR, WALL, GenericEnvironment

Cell = Tuple[int, int]


# --------------------------------------------------------------------------- #
# Terrain helpers (operate on a spec dict + a built env)
# --------------------------------------------------------------------------- #
def _is_border(env: GenericEnvironment, c: Cell) -> bool:
    x, y = c
    return x == 0 or y == 0 or x == env.width - 1 or y == env.height - 1


def _protected(env: GenericEnvironment) -> Set[Cell]:
    cells: Set[Cell] = {(env.start[0], env.start[1])}
    cells |= {(o.position[0], o.position[1]) for o in env.objects.values()}
    return cells


def _free_cell(spec: Dict[str, Any], env: GenericEnvironment, c: Cell) -> None:
    """Turn a wall cell into floor, in both the live env and the spec."""
    x, y = c
    env.terrain[y][x] = FLOOR
    walls = spec.setdefault("walls", [])
    if [x, y] in walls:
        walls.remove([x, y])
    if (x, y) in walls:           # tolerate tuple form
        walls.remove((x, y))


def _bfs_through_walls(env: GenericEnvironment, start: Cell,
                       target: Cell) -> Optional[List[Cell]]:
    """Shortest interior route start->target allowed to pass through walls."""
    if start == target:
        return []
    came: Dict[Cell, Optional[Cell]] = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nb = (cur[0] + dx, cur[1] + dy)
            if nb in came or not env.in_bounds(*nb) or _is_border(env, nb):
                continue
            came[nb] = cur
            if nb == target:
                path = [nb]
                while came[path[-1]] != start:
                    path.append(came[path[-1]])  # type: ignore
                path.reverse()
                return path
            q.append(nb)
    return None


def _reachable(env: GenericEnvironment, cell: Cell) -> bool:
    passable = spec_verifier._passable(env)
    start = (env.start[0], env.start[1])
    return cell == start or env.bfs(start, {cell}, passable) is not None


# --------------------------------------------------------------------------- #
# Locate the failing sub-goal's target cell
# --------------------------------------------------------------------------- #
def _ordered_steps(goal: Dict) -> List[Dict]:
    gp = P.goal_predicate(goal)
    if gp.get("op") == "ordered":
        return list(gp["of"])
    if gp.get("op") == "all":
        for c in gp["of"]:
            if c.get("op") == "ordered":
                return list(c["of"])
        return list(gp["of"])
    return [gp]


def _target_cell(env: GenericEnvironment, pred: Dict) -> Optional[Cell]:
    from .spec import _find_state_trigger
    op = pred.get("op")
    o = None
    if op == "acquired":
        o = env.objects.get(pred["object"])
    elif op == "at":
        t = pred["target"]
        if isinstance(t, str):
            o = env.objects.get(t)
        elif len(t) >= 2:
            return (t[0], t[1])
    elif op == "placed_in":
        o = env.objects.get(pred["receptacle"])
    elif op == "state_changed":
        o = _find_state_trigger(env, pred["object"], pred["attr"], pred["to"])
    return (o.position[0], o.position[1]) if o is not None else None


# --------------------------------------------------------------------------- #
# Repair loop
# --------------------------------------------------------------------------- #
def _solvable(spec: Dict[str, Any]) -> bool:
    from .spec import solve
    env = GenericEnvironment(spec)
    trace = solve(env, spec["goal"], max_steps=env.max_steps)
    record = P.RunRecord(env, trace, spec)
    return P.evaluate(record, P.goal_predicate(spec["goal"]))


def curriculum_repair_loop(spec: Dict[str, Any], max_iters: int = 20) -> Dict[str, Any]:
    """Repair a component spec until the oracle can complete its predicate goal."""
    from .spec import normalize_spec, solve
    spec = normalize_spec(dict(spec))          # work on a copy
    log: List[str] = []

    for it in range(max_iters):
        env = GenericEnvironment(spec)
        trace = solve(env, spec["goal"], max_steps=env.max_steps)
        record = P.RunRecord(env, trace, spec)
        if P.evaluate(record, P.goal_predicate(spec["goal"])):
            log.append(f"iter {it}: valid and oracle-solvable")
            break

        failing = next((s for s in _ordered_steps(spec["goal"])
                        if not P.evaluate(record, s)), None)
        if failing is None:
            log.append(f"iter {it}: no failing sub-goal isolated; stopping")
            break
        cell = _target_cell(env, failing)
        if cell is None:
            log.append(f"iter {it}: sub-goal {failing.get('op')} has no target cell")
            break

        changed = False
        # free the target if it was generated inside a wall
        x, y = cell
        if env.terrain[y][x] == WALL and not _is_border(env, cell):
            _free_cell(spec, env, cell)
            changed = True
        # carve one interior wall along the straightest route toward the target
        if not _reachable(env, cell):
            path = _bfs_through_walls(env, (env.start[0], env.start[1]), cell)
            if path:
                protected = _protected(env)
                for c in path:
                    cx, cy = c
                    if env.terrain[cy][cx] == WALL and not _is_border(env, c) \
                            and c not in protected:
                        _free_cell(spec, env, c)
                        changed = True
                        break
        log.append(f"iter {it}: sub-goal FAIL [{P.describe(failing)}] "
                   f"barrier=reachability -> {'carved generation' if changed else 'unrepairable'}")
        if not changed:
            break

    return {"spec": spec, "log": log, "final_solvable": _solvable(spec)}
