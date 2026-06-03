"""Code-level environment validation + symbolic task verification, surfaced
under the assignment's named checks, for the generic component/predicate world.

All checks are pure code over the components, predicate goal, and trajectory,
never pixels:

  Environment checks (BEFORE rollout) — is the generated world valid?
    check_objects_exist, check_collision_map_valid, check_no_illegal_overlap,
    check_topology_connected, check_goal_reachable, check_required_sequence_possible

  Task checks (AFTER rollout) — did the agent really complete the task?
    verify_inventory, verify_state_transition, verify_sequence,
    verify_position, verify_no_illegal_actions, verify_reward_matches_predicates
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from . import predicates as P
from .components import is_goal, is_hazard, is_permanent_barrier
from .generic import GenericEnvironment
from .schemas import VerificationReport

Cell = Tuple[int, int]


# --------------------------------------------------------------------------- #
# Goal-tree helpers
# --------------------------------------------------------------------------- #
def _walk(node: Any):
    """Yield every predicate dict in the goal tree."""
    if not isinstance(node, dict):
        return
    yield node
    of = node.get("of")
    if isinstance(of, list):
        for c in of:
            yield from _walk(c)
    elif isinstance(of, dict):
        yield from _walk(of)


def _collect(goal: Dict, ops: Tuple[str, ...]) -> List[Dict]:
    return [n for n in _walk(goal) if n.get("op") in ops]


def _goal_object_ids(goal: Dict) -> Set[str]:
    ids: Set[str] = set()
    for n in _walk(goal):
        for k in ("object", "receptacle"):
            if isinstance(n.get(k), str):
                ids.add(n[k])
        if n.get("op") == "at" and isinstance(n.get("target"), str):
            ids.add(n["target"])
    return ids


def _passable(env: GenericEnvironment):
    """Walls + permanent barriers + hazards block; openable barriers (doors,
    bridges) are treated as eventually passable for connectivity analysis."""
    def ok(x: int, y: int) -> bool:
        if env.is_wall(x, y):
            return False
        o = env.object_at(x, y)
        if o is None:
            return True
        if o.has("hazard"):
            return False
        if o.has("barrier"):
            return not is_permanent_barrier(o)
        return True
    return ok


# --------------------------------------------------------------------------- #
# Environment validation (code-level, before rollout)
# --------------------------------------------------------------------------- #
def _objective_cells(env: GenericEnvironment, goal: Dict) -> Set[Cell]:
    """The cells that constitute 'the goal': any goal-component objects, else the
    object referenced by the final ordered step (e.g. the receptacle of a
    place-in task, or the target of a reach task)."""
    cells = {(o.position[0], o.position[1]) for o in env.objects.values() if is_goal(o)}
    if cells:
        return cells
    steps = [n for n in _walk(P.goal_predicate(goal))
             if n.get("op") in ("at", "placed_in", "acquired", "state_changed")]
    if steps:
        last = steps[-1]
        oid = last.get("target") if last.get("op") == "at" else \
            last.get("receptacle") or last.get("object")
        o = env.objects.get(oid) if isinstance(oid, str) else None
        if o is not None:
            cells.add((o.position[0], o.position[1]))
    return cells


def _loose_outcome_reward(record: "P.RunRecord", goal: Dict) -> bool:
    """Naive reward baseline: did the agent finish on an objective cell?

    This intentionally ignores inventory, state transitions, and order. It is
    the sort of outcome-only reward that can be hacked; task verification checks
    that the strict predicate can reject it.
    """
    pos = record.position_at(record.n)
    if not pos:
        return False
    return tuple(pos[:2]) in _objective_cells(record.env, goal)


def verify_environment(env: GenericEnvironment, goal: Dict,
                       sequence_possible: bool) -> VerificationReport:
    r = VerificationReport()
    start = (env.start[0], env.start[1])
    passable = _passable(env)

    # check_objects_exist: every object the goal references is present, and the
    # task actually defines at least one objective step.
    referenced = _goal_object_ids(goal)
    missing = sorted(i for i in referenced if i not in env.objects)
    has_objective = bool(_collect(P.goal_predicate(goal),
                                  ("at", "placed_in", "acquired", "state_changed")))
    r.add("check_objects_exist", not missing and has_objective,
          f"missing {missing}" if missing else
          ("goal defines no objective steps" if not has_objective else ""))

    # check_collision_map_valid
    bad = next((o.id for o in env.objects.values()
                if not env.in_bounds(*o.position) or env.is_wall(*o.position)), None)
    r.add("check_collision_map_valid", bad is None,
          "" if bad is None else f"{bad} is out of bounds or inside a wall")

    # check_no_illegal_overlap
    seen: Dict[Cell, str] = {}
    overlap = None
    for o in env.objects.values():
        c = (o.position[0], o.position[1])
        if c in seen:
            overlap = (seen[c], o.id)
            break
        seen[c] = o.id
    start_blocked = env.blocks_movement(*start)
    r.add("check_no_illegal_overlap", overlap is None and not start_blocked,
          (f"{overlap[0]} and {overlap[1]} share a cell" if overlap else
           ("agent starts on a blocking cell" if start_blocked else "")))

    # check_topology_connected
    unreachable = [o.id for o in env.objects.values()
                   if not is_permanent_barrier(o)
                   and not is_hazard(o)
                   and (o.position[0], o.position[1]) != start
                   and env.bfs(start, {(o.position[0], o.position[1])}, passable) is None]
    r.add("check_topology_connected", not unreachable,
          "" if not unreachable else f"unreachable: {sorted(unreachable)}")

    # check_goal_reachable: the objective cell(s) are reachable from the start.
    objectives = _objective_cells(env, goal)
    reachable = bool(objectives) and any(
        g == start or env.bfs(start, {g}, passable) is not None for g in objectives)
    r.add("check_goal_reachable", reachable,
          "" if reachable else "no objective cell reachable from start")

    # check_required_sequence_possible: a planner witness that the generated
    # task is executable. This is not independent proof of policy competence;
    # adversarial negative controls live in spec_exploit.
    r.add("check_required_sequence_possible", sequence_possible,
          "" if sequence_possible else "ordered goal is not achievable in order")
    return r


# --------------------------------------------------------------------------- #
# Symbolic task verification (after rollout)
# --------------------------------------------------------------------------- #
def verify_task(record: "P.RunRecord", goal: Dict,
                env: GenericEnvironment) -> VerificationReport:
    r = VerificationReport()
    gp = P.goal_predicate(goal)

    def check(name: str, ops: Tuple[str, ...]) -> None:
        nodes = _collect(gp, ops)
        if not nodes:
            r.add(name, True, "(no such step in this task)")
            return
        results = [(P.describe(n), P.evaluate(record, n)) for n in nodes]
        ok = all(p for _, p in results)
        fails = [d for d, p in results if not p]
        r.add(name, ok, "" if ok else "failed: " + "; ".join(fails))

    check("verify_inventory", ("acquired",))
    check("verify_state_transition", ("state_changed", "placed_in"))
    check("verify_position", ("at",))

    # verify_sequence: the whole ordered/all goal happened, in order
    seq_ok = P.evaluate(record, gp)
    r.add("verify_sequence", seq_ok,
          "" if seq_ok else "goal steps did not occur in the required order")

    # verify_no_illegal_actions
    clean = P.evaluate(record, {"op": "no_illegal_actions"})
    r.add("verify_no_illegal_actions", clean,
          "" if clean else "blocked/invalid actions occurred in the trajectory")

    # verify_reward_matches_predicates: compare strict predicate success against
    # a deliberately loose outcome-only reward and adversarial synthetic traces.
    # This deliberately avoids comparing env.success back to the same strict
    # predicate used to assign it in the runner.
    loose_ok = _loose_outcome_reward(record, goal)
    try:
        from . import spec_exploit
        controls = spec_exploit.predicate_negative_controls(
            GenericEnvironment(record.spec), goal)
    except Exception:  # noqa: BLE001 - integrity check should report, not crash
        controls = {"leaky": True, "failures": ["predicate controls crashed"]}
    integrity = not (loose_ok and not seq_ok) and not controls.get("leaky", False)
    detail = ""
    if loose_ok and not seq_ok:
        detail = "loose outcome-only reward would fire but strict predicate is false"
    elif controls.get("leaky"):
        detail = "adversarial predicate controls accepted: " + \
            ", ".join(controls.get("failures", []))
    r.add("verify_reward_matches_predicates", integrity, detail)
    return r
