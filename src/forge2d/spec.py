"""Spec-file workflow: the bridge between spec authoring and the engine.

Forge-2D runs JSON/YAML worlds described in terms of the generic component
object model plus a predicate goal. A spec can be authored by a human, by the
deterministic parser fallback, or by the optional LLM authoring path. The engine
then builds the world, drives a rollout agent, and verifies the result with the
trusted predicate interpreter.

    text/spec  --(authoring)-->  spec.json  --(Forge-2D)-->  run + verify + render

The whole path is deterministic and **requires no API key**: the oracle reads
the goal predicate to decide what to do, so the core demo runs offline.

    python -m forge2d.cli run examples/generated_spec.json

Spec shape
----------
    {
      "name": "...",
      "grid": {"width": W, "height": H},      # optional; derived if absent
      "walls": [[x, y], ...],
      "agent": {"start": [x, y]},
      "objects": [
        {"id": "...", "position": [x, y],
         "components": {"pickupable": {}, ...},   # or ["pickupable", ...]
         "state": {"locked": true}}
      ],
      "goal": { <predicate tree, usually reward_iff(ordered([...])) > }
    }
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from . import components as C
from . import controller
from . import predicates
from . import spec_diversity
from . import spec_exploit
from . import llm_agent
from . import spec_repair
from . import spec_verifier
from .generic import GenericEnvironment
from .schemas import MOVE_DELTAS, PICKUP, USE

Cell = Tuple[int, int]
_DELTA_TO_ACTION = {delta: action for action, delta in MOVE_DELTAS.items()}


# --------------------------------------------------------------------------- #
# Loading & normalization
# --------------------------------------------------------------------------- #
def load_spec(path: str) -> Dict[str, Any]:
    """Load a spec from JSON (always) or YAML (if PyYAML is installed)."""
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # optional dependency
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to read .yaml specs; use JSON or `pip install pyyaml`"
            ) from e
        with open(path) as f:
            return yaml.safe_load(f)
    with open(path) as f:
        return json.load(f)


def normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Fill defaults and derive the grid size from content if not given."""
    spec = dict(spec)
    spec.setdefault("name", "unnamed_spec")
    spec.setdefault("walls", [])
    spec.setdefault("objects", [])
    spec.setdefault("agent", {"start": [1, 1]})
    spec.setdefault("goal", {"op": "no_illegal_actions"})

    if "grid" not in spec:
        max_x = max_y = 2
        for o in spec["objects"]:
            px, py = o["position"]
            max_x, max_y = max(max_x, px + 1), max(max_y, py + 1)
        for wx, wy in spec["walls"]:
            max_x, max_y = max(max_x, wx + 1), max(max_y, wy + 1)
        sx, sy = spec["agent"]["start"]
        max_x, max_y = max(max_x, sx + 1), max(max_y, sy + 1)
        spec["grid"] = {"width": max_x + 1, "height": max_y + 1}
    return spec


# --------------------------------------------------------------------------- #
# Spec-derived oracle: read the goal predicate, act to satisfy it
# --------------------------------------------------------------------------- #
def _path_to_actions(path: List[Cell]) -> List[str]:
    actions: List[str] = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        actions.append(_DELTA_TO_ACTION[(x1 - x0, y1 - y0)])
    return actions


def _navigate_onto(env: GenericEnvironment, cell: Cell) -> Optional[List[str]]:
    start = tuple(env.agent_pos)
    if start == cell:
        return []
    path = env.bfs(start, {cell})
    if path is None:
        return None
    return _path_to_actions([start] + path)


def _navigate_adjacent(env: GenericEnvironment, cell: Cell) -> Optional[List[str]]:
    """Stand on a passable orthogonal neighbour of `cell` (for USE)."""
    start = tuple(env.agent_pos)
    cx, cy = cell
    neighbours = {(cx + dx, cy + dy) for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))}
    if start in neighbours:
        return []
    path = env.bfs(start, neighbours)
    if path is None:
        return None
    return _path_to_actions([start] + path)


def _acquire(env: GenericEnvironment, obj_id: str) -> Optional[List[str]]:
    o = env.objects.get(obj_id)
    if o is None:
        return None
    if obj_id in env.inventory or o.picked:
        return []
    moves = _navigate_onto(env, o.cell)
    if moves is None:
        return None
    return moves + [PICKUP]


def _find_state_trigger(env: GenericEnvironment, obj_id: str, attr: str, to: Any):
    for o in env.objects.values():
        if not o.has(C.TRIGGER):
            continue
        cfg = o.components[C.TRIGGER]
        for eff in cfg.get("effects", []):
            tid = eff.get("target", "self")
            tid = o.id if tid in ("self", o.id) else tid
            if tid == obj_id and eff.get("set", {}).get(attr) == to:
                return o
    return None


def _actions_for_subgoal(env: GenericEnvironment, sub: Dict[str, Any]) -> List[str]:
    """Greedy action list (executed against the live env) to satisfy one
    sub-predicate. Returns [] for declarative predicates and best-effort actions
    for behavioural ones; an unreachable target yields a partial/empty plan so
    verification fails honestly rather than the planner pretending success."""
    op = sub.get("op")

    if op == "acquired":
        return _acquire(env, sub["object"]) or []

    if op == "at":
        target = sub["target"]
        cell = tuple(env.objects[target].cell) if isinstance(target, str) else tuple(target[:2])
        return _navigate_onto(env, cell) or []

    if op == "placed_in":
        obj_id, recep_id = sub["object"], sub["receptacle"]
        actions = _acquire(env, obj_id) or []
        recep = env.objects.get(recep_id)
        if recep is None:
            return actions
        nav = _navigate_adjacent(env, recep.cell)
        return actions + (nav + [USE] if nav is not None else [])

    if op == "state_changed":
        trig = _find_state_trigger(env, sub["object"], sub["attr"], sub["to"])
        if trig is None:
            return []
        cfg = trig.components[C.TRIGGER]
        prefix: List[str] = []
        for req in cfg.get("requires", []) or []:
            prefix += _acquire(env, req) or []
        if cfg.get("on", "enter") == "use":
            nav = _navigate_adjacent(env, trig.cell)
            return prefix + (nav + [USE] if nav is not None else [])
        nav = _navigate_onto(env, trig.cell)
        return prefix + (nav or [])

    # exists / has_component / reachable / no_illegal_actions: nothing to do
    return []


def _subgoals(goal: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The ordered list of sub-predicates that drive the oracle."""
    pred = predicates.goal_predicate(goal)
    if pred.get("op") == "ordered":
        return list(pred["of"])
    if pred.get("op") == "all":
        # drive on the first ordered conjunct if present, else all conjuncts
        for c in pred["of"]:
            if c.get("op") == "ordered":
                return list(c["of"])
        return list(pred["of"])
    return [pred]


def solve(env: GenericEnvironment, goal: Dict[str, Any], max_steps: int = 200
          ) -> List[Dict[str, Any]]:
    """Drive the spec-derived oracle and record a trajectory."""
    trace: List[Dict[str, Any]] = []

    def record(action: str, info: Dict[str, Any]) -> None:
        trace.append({
            "t": env.t, "action": action,
            "position": list(env.agent_pos),
            "inventory": list(env.inventory),
            "events": info.get("events", []),
        })

    for sub in _subgoals(goal):
        # Recompute each sub-goal against the live env (prior unlocks/pickups
        # have already mutated it), executing one action at a time.
        for action in _actions_for_subgoal(env, sub):
            if env.done or env.t >= max_steps:
                break
            _, _, _, info = env.step(action)
            record(action, info)
        if env.done:
            break
    return trace


def solve_with_llm(env: GenericEnvironment, goal: Dict[str, Any], spec: Dict[str, Any],
                   max_steps: int = 200, allow_api: bool = True,
                   verbose: bool = False) -> List[Dict[str, Any]]:
    policy = llm_agent.LLMHighLevelPolicy(allow_api=allow_api, verbose=verbose)
    trace: List[Dict[str, Any]] = []

    def record(action: str, info: Dict[str, Any]) -> None:
        trace.append({
            "t": env.t, "action": action,
            "position": list(env.agent_pos),
            "inventory": list(env.inventory),
            "events": info.get("events", []),
        })

    while not env.done and env.t < max_steps:
        record_view = predicates.RunRecord(env, trace, spec)
        subgoal = policy.next_subgoal(env, goal, record_view)
        if subgoal is None:
            break
        actions = _actions_for_subgoal(env, subgoal)
        if not actions:
            break
        for action in actions:
            if env.done or env.t >= max_steps:
                break
            _, _, _, info = env.step(action)
            record(action, info)
    return trace


def solve_with_controller(env: GenericEnvironment, goal: Dict[str, Any],
                          max_steps: int = 200) -> List[Dict[str, Any]]:
    """Run through the live rendered-observation/controller-action interface.

    The baseline policy receives an RGB frame each tick and emits a GI-style
    controller command. A symbolic scaffold provides the next intended sub-goal
    action so this remains a simple scripted policy, not a learned visual model.
    """
    policy = controller.ScriptedControllerPolicy()
    trace: List[Dict[str, Any]] = []

    def record(engine_action: str, controller_action: str,
               obs: Dict[str, Any], info: Dict[str, Any]) -> None:
        trace.append({
            "t": env.t,
            "action": engine_action,
            "controller_action": controller_action,
            "observation": {
                "frame_width": obs["frame_width"],
                "frame_height": obs["frame_height"],
                "channels": obs["channels"],
                "controller_actions": obs["controller_actions"],
            },
            "position": list(env.agent_pos),
            "inventory": list(env.inventory),
            "events": info.get("events", []),
        })

    for sub in _subgoals(goal):
        plan = _actions_for_subgoal(env, sub)
        for planned in plan:
            if env.done or env.t >= max_steps:
                break
            obs = controller.observation(env)
            cmd = policy.next_action(obs, planned)
            engine_action = controller.controller_to_engine(cmd)
            _, _, _, info = env.step(engine_action)
            record(engine_action, cmd.primary(), obs, info)
        if env.done:
            break
    return trace


# --------------------------------------------------------------------------- #
# Top-level run
# --------------------------------------------------------------------------- #
def run_spec(spec: Dict[str, Any], auto_repair: bool = True,
             env_quality: bool = True,
             diversity_buffer: "Optional[spec_diversity.DiversityBuffer]" = None,
             agent_name: str = "oracle",
             allow_llm_api: bool = True,
             verbose_llm: bool = False,
             ) -> Dict[str, Any]:
    """Build the world, run the oracle, and verify with the predicate DSL.

    If `auto_repair` and the world is not solvable as generated, the curriculum
    repair loop carves the generation until the oracle can complete the goal.
    If `env_quality`, run diversity (anti-collapse) + exploit testing (shadow
    verifier + search adversary, sealing any instruction-skipping shortcut).
    The (repaired/sealed) world is what we then run, verify and render.
    """
    spec = normalize_spec(spec)
    goal = spec["goal"]
    repair_log: Optional[List[str]] = None
    if auto_repair and not spec_repair._solvable(spec):
        rep = spec_repair.curriculum_repair_loop(spec)
        if rep["final_solvable"]:
            spec = rep["spec"]
            repair_log = rep["log"]

    # Exploit testing: detect (and, with auto_repair, seal) shortcuts that reach
    # the goal without following the intended sequence.
    exploit: Optional[Dict[str, Any]] = None
    if env_quality:
        exploit = spec_exploit.shadow_audit(GenericEnvironment(spec), goal)
        if exploit["leaky"] and auto_repair:
            exploit = {**exploit, **spec_exploit.seal_bypass(spec, goal)}

    env_for_checks = GenericEnvironment(spec)
    check_trace = solve(GenericEnvironment(spec), goal, max_steps=env_for_checks.max_steps)
    check_env = GenericEnvironment(spec)
    check_record = predicates.RunRecord(check_env, check_trace, spec)
    sequence_possible = predicates.evaluate(check_record, predicates.goal_predicate(goal))
    env_checks = spec_verifier.verify_environment(check_env, goal, sequence_possible)

    env = GenericEnvironment(spec)
    ascii_initial = env.render_ascii()

    inv_record = predicates.RunRecord(GenericEnvironment(spec), [], spec)
    inv_report = predicates.VerificationReport()
    for inv in spec.get("invariants", []):
        ok = predicates.evaluate(inv_record, inv)
        inv_report.add(predicates.describe(inv), ok,
                       "" if ok else "invariant false on initial world")

    if agent_name == "llm":
        trace = solve_with_llm(env, goal, spec, max_steps=env.max_steps,
                               allow_api=allow_llm_api, verbose=verbose_llm)
    elif agent_name == "controller":
        trace = solve_with_controller(env, goal, max_steps=env.max_steps)
    else:
        trace = solve(env, goal, max_steps=env.max_steps)

    # The strict predicate goal decides success; task verification separately
    # checks it against loose outcome-only reward baselines.
    record = predicates.RunRecord(env, trace, spec)
    success_pred = predicates.goal_predicate(goal)
    env.success = predicates.evaluate(record, success_pred)
    report = predicates.verify(record, goal)

    task_checks = spec_verifier.verify_task(record, goal, env)

    # Diversity (anti-collapse): reduce the world to a layout embedding and
    # compare it against the buffer of worlds already generated.
    diversity: Optional[Dict[str, Any]] = None
    if env_quality:
        feats = spec_diversity.layout_features(
            env, sequence_depth=len(spec_repair._ordered_steps(goal)),
            path_length=env.t)
        vec = spec_diversity.feature_vector(feats)
        if diversity_buffer is not None:
            similarity, novel = diversity_buffer.assess(vec)
            diversity_buffer.add(vec)
        else:
            similarity, novel = 0.0, True
        diversity = {"features": feats, "similarity": similarity, "novel": novel}

    return {
        "name": env.name,
        "spec": spec,
        "goal": goal,
        "ascii_initial": ascii_initial,
        "ascii_final": env.render_ascii(),
        "trace": trace,
        "events": list(env.event_log),
        "steps": env.t,
        "success": env.success,
        "dead": env.dead,
        "agent": agent_name,
        "report": report,
        "invariants": inv_report,
        "env_checks": env_checks,
        "task_checks": task_checks,
        "repair": repair_log,
        "diversity": diversity,
        "exploit": exploit,
    }


def run_spec_file(path: str, auto_repair: bool = True, env_quality: bool = True,
                  diversity_buffer: "Optional[spec_diversity.DiversityBuffer]" = None,
                  agent_name: str = "oracle",
                  allow_llm_api: bool = True,
                  verbose_llm: bool = False,
                  ) -> Dict[str, Any]:
    return run_spec(load_spec(path), auto_repair=auto_repair,
                    env_quality=env_quality, diversity_buffer=diversity_buffer,
                    agent_name=agent_name, allow_llm_api=allow_llm_api,
                    verbose_llm=verbose_llm)


# --------------------------------------------------------------------------- #
# Console rendering
# --------------------------------------------------------------------------- #
def format_report(result: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("=" * 64)
    L.append("FORGE-2D  " + result.get("source", "generic component world + predicate goal"))
    L.append("=" * 64)
    L.append(f"Spec: {result['name']}")
    L.append("")
    L.append("Initial world (component-derived glyphs):")
    L.append(result["ascii_initial"])
    L.append("  A agent  G goal  X hazard  B/D barrier(blocking)  b/d barrier(open)  "
             "S trigger  R receptacle  P pickupable  o object")
    L.append("")
    if result.get("repair"):
        L.append("Curriculum Repair (closed loop — world was unsolvable, repaired):")
        for line in result["repair"]:
            L.append(f"    - {line}")
        L.append("")
    # --- (1) code-level environment validation, BEFORE the agent acts --------
    L.append("Environment Verification — code-level validation (before rollout):")
    L.append(result["env_checks"].render())
    inv = result.get("invariants")
    if inv is not None and inv.checks:
        L.append("  spec invariants (author-asserted):")
        for line in inv.render().splitlines():
            L.append("  " + line)
    L.append("")
    L.append(f"Goal predicate: {predicates.describe(result['goal'])}")
    L.append("")
    L.append(f"Agent rollout ({result.get('agent', 'oracle')}): {result['steps']} steps")
    if result.get("agent") == "controller" and result.get("trace"):
        obs = result["trace"][0].get("observation", {})
        ctrl = result["trace"][0].get("controller_action")
        L.append("  Observation/action interface:")
        L.append(f"    - RGB frame: {obs.get('frame_width')}x{obs.get('frame_height')}x{obs.get('channels')}")
        L.append(f"    - controller action vocabulary: {', '.join(obs.get('controller_actions', []))}")
        L.append(f"    - first controller action: {ctrl}")
    if result["events"]:
        L.append("  Events:")
        for e in result["events"]:
            L.append(f"    - {e}")
    L.append("")
    # --- (2) symbolic task verification, AFTER the agent acts ----------------
    L.append("Task Verification — symbolic task checks (after rollout):")
    L.append(result["task_checks"].render())
    L.append("")
    # --- Environment quality: diversity (anti-collapse) + exploit testing -----
    div = result.get("diversity")
    exp = result.get("exploit")
    if div is not None or exp is not None:
        L.append("Environment Quality:")
        if div is not None:
            f = div["features"]
            L.append(f"  diversity: similarity_to_buffer={div['similarity']:.2f}  "
                     f"[{'PASS' if div['novel'] else 'FLAG'}] "
                     + ("structurally novel" if div["novel"]
                        else "TOO SIMILAR (template collapse)"))
            L.append(f"    layout embedding: objects={f['objects']} "
                     f"barriers={f['lockable_barriers']}+{f['static_barriers']} "
                     f"goals={f['goals']} triggers={f['triggers']} "
                     f"receptacles={f['receptacles']} hazards={f['hazards']} "
                     f"depth={f['sequence_depth']} path={f['path_length']} "
                     f"adjacency_edges={f['adjacency_edges']} "
                     f"junctions={f.get('junctions', 0)} "
                     f"dead_ends={f.get('dead_ends', 0)} "
                     f"rooms={f.get('rooms', 0)} open_area={f.get('open_area', 0)}")
        if exp is not None:
            leaky = exp.get("leaky_after", exp.get("leaky"))
            if exp.get("log"):
                L.append(f"  exploit: leaky {exp.get('leaky_before')} -> {leaky} "
                         "(shadow verifier + spatial search adversary)")
                for line in exp["log"]:
                    L.append(f"    - {line}")
                L.append("    official verifier: strict predicate goal (not outcome-only)")
            else:
                L.append(f"  exploit: [{'FLAG' if exp.get('leaky') else 'PASS'}] "
                         + (exp.get("exploit") or
                            "no instruction-skipping shortcut to the goal"))
            controls = exp.get("predicate_controls")
            if controls and controls.get("tested"):
                status = "PASS" if not controls.get("leaky") else "FLAG"
                L.append(f"    predicate negative controls: [{status}] "
                         f"{len(controls['tested'])} tested")
        L.append("")
    L.append("Final world:")
    L.append(result["ascii_final"])
    L.append("")
    env_ok = result["env_checks"].passed
    task_ok = result["task_checks"].passed
    exp = result.get("exploit")
    exploit_ok = not (exp and exp.get("leaky_after", exp.get("leaky")))
    verdict = "SUCCESS" if result["success"] and env_ok and task_ok and exploit_ok else "FAILURE"
    L.append(f"=> {verdict}  (env_valid={env_ok}, task_verified={task_ok}, "
             f"success={result['success']}, exploit_safe={exploit_ok})")
    L.append("=" * 64)
    return "\n".join(L)
