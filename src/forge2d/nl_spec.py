"""Generic natural-language -> component-spec mapping.

This layer is **purely verb-driven**: an object's role comes from what the agent
is told to *do* with it, never from recognising the noun. A noun is just an id.

    pick up X    -> X: pickupable        ; acquired(X)
    unlock X     -> X: barrier + trigger ; state_changed(X.locked, true->false)
    press X      -> X: trigger (controls the next gate)
    raise X      -> X: barrier(while raised)  (a gate)
    reach X      -> X: goal               ; at(agent, X)
    place X in Y -> X: placeable, Y: receptacle ; placed_in(X, Y)
    ...on Y      -> Y: surface ; near(X, Y)
    ...near wall -> near_wall(X)
    ...behind Y  -> Y: barrier (placed between agent and the host)

There is deliberately NO noun lexicon (no "door means a barrier"): that would be
exactly the scenario-specific hardcoding the engine avoids. A bare noun-list
prompt ("a key, a door, a switch") therefore yields no objects — write the task
as actions instead ("pick up the key, unlock the door, ...").
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

GRID_HEIGHT = 5
SPINE_ROW = 2
FIRST_OBJECT_X = 3
OBJECT_X_STEP = 2
NAME_LIMIT = 48

_PHRASES = {
    "pick up": "pickup", "picks up": "pickup", "go to": "goto", "get to": "goto",
    "step on": "stepon", "navigate to": "goto", "walk to": "goto",
}

_VERB = {
    "pickup": "acquire", "grab": "acquire", "collect": "acquire", "take": "acquire",
    "retrieve": "acquire", "obtain": "acquire", "fetch": "acquire",
    "unlock": "unlock", "open": "unlock",
    "place": "place", "put": "place", "drop": "place", "dispose": "place",
    "throw": "place", "deposit": "place",
    "reach": "reach", "goto": "reach", "arrive": "reach", "find": "reach",
    "press": "trigger", "flip": "trigger", "activate": "trigger", "toggle": "trigger",
    "push": "trigger", "pull": "trigger", "hit": "trigger", "stepon": "trigger",
    "raise": "gate", "lower": "gate", "cross": "gate", "extend": "gate", "lift": "gate",
}

_PREPS = {"in", "into", "inside", "on", "onto", "atop", "near", "by", "against", "behind"}

_ARTICLES = {"a", "an", "the", "some", "another"}
_SKIP = {
    "is", "are", "be", "been", "being", "sits", "sit", "sitting", "rests", "rest",
    "resting", "lies", "lying", "located", "placed", "must", "then", "where",
    "which", "that", "this", "with", "agent", "it", "them", "and", "to", "of",
    "create", "build", "make", "generate", "world", "task", "should",
}
_NONOBJECT = {"wall", "walls", "room", "rooms"}
_PRONOUNS = {"it", "them"}

_PUNCT = re.compile(r"[.,;]")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

AUTHOR_CHOICES = ("auto", "parser", "llm")
DEFAULT_LLM_MODEL = "claude-opus-4-8"

_AUTHOR_SYSTEM = """You author Forge-2D component specs from natural-language tasks.
Return ONLY valid JSON. No markdown.

Schema:
{
  "name": "short_slug",
  "grid": {"width": int, "height": int},
  "walls": [[x, y], ...],
  "agent": {"start": [x, y]},
  "objects": [
    {
      "id": "snake_case",
      "position": [x, y],
      "components": {"pickupable": {}, "goal": {}, "barrier": {...}, ...},
      "state": {"locked": true}
    }
  ],
  "invariants": [{"op": "exists", "object": "id"}, ...],
  "goal": {"op": "reward_iff", "of": {"op": "ordered", "of": [...]}}
}

Supported components:
- pickupable
- placeable
- receptacle: {"accepts": ["placeable"]}
- goal
- hazard
- surface
- stateful
- barrier: {"while": {"attr": "locked", "equals": true}} or {"while": {"attr": "raised", "equals": false}} or {}
- trigger: {"on": "use"|"enter", "requires": ["object_id"], "effects": [{"target": "self"|"object_id", "set": {"locked": false}}], "event": "unlock"}

Supported predicates:
- exists(object)
- has_component(object, component)
- reachable(agent, target)
- near(object, target, within)
- near_wall(object, within)
- acquired(object)
- placed_in(object, receptacle)
- state_changed(object, attr, from, to)
- at(agent, target)
- ordered(of)
- reward_iff(of)

Rules:
- Make a small 2D grid with border walls omitted; the engine supplies bounds.
- Keep all coordinates in bounds and avoid overlaps.
- Natural-language support relations are spatial, not overlapping: for "X on Y",
  "X on a shelf/table", or similar, put X adjacent to Y, make Y a surface, and
  add a near(X, Y, within=1) invariant.
- For disposal/placement tasks, make the carried item pickupable + placeable and
  make the destination a receptacle that accepts placeable.
- Put the agent at [1, 2] unless the task needs another start.
- Use walls/barriers to express spatial constraints, but keep the task solvable.
- Encode the intended objective in the goal predicate, not in prose.
- Prefer component semantics over object-name special cases.
- IMPORTANT gating semantics: a trigger's `requires` is checked against the
  agent's INVENTORY only (carried pickupables) -- it CANNOT test whether another
  switch was pressed. So a single barrier can be controlled by at most ONE
  trigger. For a multi-step lock ("flip A then B to open the vault"), express it
  as a CHAIN of single-trigger gates the agent must pass through in order, or as
  carried-key prerequisites (trigger.requires = [carried_key_id]) -- never as two
  independent switches both gating the same door."""


def _tokenize(command: str) -> List[str]:
    text = command.lower()
    for phrase, tok in _PHRASES.items():
        text = text.replace(phrase, tok)
    text = _PUNCT.sub(" . ", text)
    return re.findall(r"[a-z]+|\.", text)


def _load_dotenv(path: str = ".env") -> None:
    candidates = [path, os.path.join(os.path.dirname(__file__), "..", "..", ".env")]
    for candidate in candidates:
        try:
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except OSError:
            pass


def _slug(command: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", command.lower()).strip("_")[:NAME_LIMIT] or "world"


def _extract_json(text: str) -> Dict[str, Any]:
    match = _JSON_BLOCK.search(text)
    if match is None:
        raise ValueError("LLM author did not return a JSON object")
    return json.loads(match.group(0))


def _validate_authored_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Cheap trust-boundary validation for LLM-authored specs.

    Deep checks still happen in the normal verifier/repair/exploit pipeline.
    This only ensures the JSON can be built as a Forge-2D world and contains a
    predicate goal, so bad LLM output fails at authoring time instead of
    crashing during rollout.
    """
    if not isinstance(spec.get("objects", []), list):
        raise ValueError("LLM-authored spec has invalid `objects`; expected a list")
    if "goal" not in spec or not isinstance(spec["goal"], dict):
        raise ValueError("LLM-authored spec is missing `goal`")
    from .spec import normalize_spec
    from .generic import GenericEnvironment
    GenericEnvironment(normalize_spec(spec))
    return spec


def _evaluate_authored(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run an authored spec through the trusted harness evaluators once.

    Returns the full run_spec result (including the normalized/repaired `spec`
    and the structural `diversity` features) when the world is playable,
    task-verified, and exploit-safe. Raises with a compact failure summary
    otherwise, so the LLM can repair its draft.
    """
    from .spec import run_spec

    spec = _validate_authored_spec(spec)
    result = run_spec(spec, auto_repair=True, env_quality=True)
    exp = result.get("exploit")
    exploit_ok = not (exp and exp.get("leaky_after", exp.get("leaky")))
    ok = result["success"] and result["env_checks"].passed \
        and result["task_checks"].passed and exploit_ok
    if ok:
        result["spec"]["_author"] = "llm"
        return result

    failures: List[str] = []
    failures.extend(c.line() for c in result["env_checks"].failures())
    failures.extend(c.line() for c in result["task_checks"].failures())
    if exp and not exploit_ok:
        failures.append("[FAIL] exploit audit: " + (exp.get("exploit") or "leaky task"))
    if not failures:
        failures.append("[FAIL] generated spec did not complete successfully")
    raise ValueError("; ".join(failures[:6]))


def _validate_authored_run(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate + run an authored spec; return the trusted/repaired spec."""
    return _evaluate_authored(spec)["spec"]


# Distinct structural targets so the diversity loop can ask the LLM author for a
# genuinely DIFFERENT layout per variant -- the LLM analogue of the parser's
# topology families. Variant 0 is the natural rendering; higher variants push the
# embedding (path length, junctions, dead ends, rooms, open area) apart so
# anti-collapse can actually select a novel world instead of only flagging one.
_VARIANT_STYLES = (
    "",  # 0: natural layout
    "a single long winding corridor with several turns (high path length, few junctions)",
    "many branching junctions with dead-end side passages (high junctions and dead ends)",
    "two or three separate rooms connected by narrow one-tile doorways",
    "a large open hall with scattered free-standing obstacle walls",
    "a tight maze of many short corridors",
    "a nested / spiral structure that wraps toward the goal",
    "an asymmetric layout forcing one long detour around a big central obstacle",
)


def _variant_directive(variant: int) -> str:
    if not variant:
        return ""
    style = _VARIANT_STYLES[variant % len(_VARIANT_STYLES)]
    return ("\n\n[Generation variant "
            f"{variant}: author a STRUCTURALLY DIFFERENT world shaped as {style}. "
            "Vary the grid size, wall topology, and object placement accordingly. "
            "Keep the SAME task semantics, objects' roles, and solvability.]")


def _generate_llm(command: str, model: str = DEFAULT_LLM_MODEL,
                  max_attempts: int = 4, variant: int = 0) -> Dict[str, Any]:
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for --author llm")
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("Install the optional `anthropic` SDK for --author llm") from e

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": command + _variant_directive(variant)}]
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        resp = client.messages.create(
            model=model,
            max_tokens=3000,
            system=_AUTHOR_SYSTEM,
            messages=messages,
        )
        text = "".join(getattr(block, "text", "") for block in resp.content)
        try:
            spec = _extract_json(text)
            spec.setdefault("name", _slug(command))
            spec.setdefault("walls", [])
            spec.setdefault("objects", [])
            spec.setdefault("agent", {"start": [1, 2]})
            spec["_author"] = "llm"
            return _validate_authored_run(spec)
        except Exception as e:  # noqa: BLE001 - feed validation failure back
            last_error = str(e)
            if attempt == max_attempts:
                break
            messages.extend([
                {"role": "assistant", "content": text},
                {"role": "user", "content":
                 "That spec failed the Forge-2D verifier: "
                 f"{last_error}. Return a corrected JSON spec only. "
                 "Avoid overlapping objects; make every required object reachable; "
                 "ensure the ordered goal can be completed; and avoid outcome-only shortcuts."},
            ])
    raise ValueError(f"LLM-authored spec failed validation after {max_attempts} attempts: {last_error}")


def _finalize_authored(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Fill the structural defaults Forge-2D expects on an authored spec."""
    spec = dict(raw)
    spec.setdefault("name", name)
    spec.setdefault("walls", [])
    spec.setdefault("objects", [])
    spec.setdefault("agent", {"start": [1, 2]})
    spec["_author"] = "llm"
    return spec


_IMPROVE_SYSTEM = _AUTHOR_SYSTEM + """

You are now EXTENDING an existing, working spec into a STRICTLY HARDER one.
- Keep the existing objective and objects; APPEND new structure (more objects,
  barriers, triggers, hazards) and lengthen the ordered goal so the required
  action sequence grows.
- The world must stay fully solvable, with no outcome-only shortcut.
- Return ONLY the complete new JSON spec."""


def _llm_proposer(model: str = DEFAULT_LLM_MODEL):
    """Build a proposer that asks Anthropic to extend a spec into a harder one."""
    _load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for LLM improvement")
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("Install the optional `anthropic` SDK for LLM improvement") from e
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def propose(spec: Dict[str, Any], score: float, feedback: str) -> Dict[str, Any]:
        msg = ("Current spec (difficulty score "
               f"{score:.0f}):\n{json.dumps(spec)}\n\n"
               "Return a NEW complete spec that is strictly harder: a longer "
               "required action sequence and/or more interacting objects, still "
               "solvable.")
        if feedback:
            msg += f"\n\nYour previous attempt was rejected: {feedback}"
        resp = client.messages.create(model=model, max_tokens=3000,
                                       system=_IMPROVE_SYSTEM,
                                       messages=[{"role": "user", "content": msg}])
        text = "".join(getattr(b, "text", "") for b in resp.content)
        return _extract_json(text)

    return propose


def improve(spec: Dict[str, Any], rounds: int = 3,
            model: str = DEFAULT_LLM_MODEL, propose=None,
            max_tries: int = 3) -> List[Dict[str, Any]]:
    """Iteratively extend a world into a strictly-harder, still-valid curriculum.

    Starting from `spec`, each round asks the author (the LLM by default) to
    append structure and lengthen the task. A proposed world is ACCEPTED only
    when it (a) passes the same trusted harness as any authored world -- playable,
    task-verified, and exploit-safe -- AND (b) scores strictly higher on
    `complexity_score` than the last accepted world. So "better" means genuinely
    harder *and still correct*, never merely bigger or broken; a proposal that
    breaks verification or fails to raise the difficulty is rejected and the LLM
    is asked again with that reason as feedback.

    `propose(current_spec, current_score, feedback) -> raw_spec_dict` is the only
    LLM touch point; tests inject a deterministic stub. Returns the curriculum as
    a list of ``{"spec", "score", "features", "round"}`` entries with strictly
    increasing scores, the seed first.
    """
    from . import spec_diversity

    propose = propose or _llm_proposer(model)
    seed_result = _evaluate_authored(dict(spec))
    seed_feats = seed_result["diversity"]["features"]
    curriculum = [{"spec": seed_result["spec"],
                   "score": spec_diversity.complexity_score(seed_feats),
                   "features": seed_feats, "round": 0}]

    for rnd in range(1, rounds + 1):
        cur = curriculum[-1]
        feedback = ""
        for _ in range(max_tries):
            try:
                raw = propose(cur["spec"], cur["score"], feedback)
                result = _evaluate_authored(
                    _finalize_authored(raw, cur["spec"].get("name", "world")))
            except Exception as e:  # noqa: BLE001 - rejection feedback for next try
                feedback = f"invalid: {e}"
                continue
            feats = result["diversity"]["features"]
            score = spec_diversity.complexity_score(feats)
            if score <= cur["score"]:
                feedback = (f"not harder: score {score:.0f} <= {cur['score']:.0f}; "
                            "add required steps / interacting objects / longer path")
                continue
            curriculum.append({"spec": result["spec"], "score": score,
                               "features": feats, "round": rnd})
            break
    return curriculum


class _Obj:
    def __init__(self, oid: str, head: str):
        self.id = oid
        self.head = head
        self.components: Dict[str, Dict[str, Any]] = {}
        self.state: Dict[str, Any] = {}
        self.on_target: Optional[str] = None
        self.near_wall = False
        self.behind_host: Optional[str] = None
        self.spine = False

    def add(self, name: str, config: Optional[Dict] = None) -> None:
        self.components[name] = dict(config or {})


class _Parser:
    def __init__(self) -> None:
        self.objs: Dict[str, _Obj] = {}
        self.by_head: Dict[str, _Obj] = {}
        self.order: List[Dict[str, Any]] = []
        self.invariants: List[Dict[str, Any]] = []
        self._inv_seen: set = set()
        self.last_obj: Optional[_Obj] = None
        self.last_item: Optional[str] = None
        self.pending_trigger: Optional[_Obj] = None

    def resolve(self, words: List[str]) -> Optional[_Obj]:
        words = [w for w in words
                 if w not in _ARTICLES and w not in _SKIP and w not in _NONOBJECT]
        if not words:
            return None
        head = words[-1]
        oid = "_".join(words)
        if len(words) > 1:
            # qualified noun phrase ("blue key"): a distinct object unless this
            # exact phrase was seen before. This keeps "red key" and "blue key"
            # separate instead of collapsing them on the shared head "key".
            if oid in self.objs:
                return self.objs[oid]
        elif head in self.by_head:
            # bare noun ("the key"): coreference back to the last such object.
            return self.by_head[head]
        o = _Obj(oid, head)
        self.objs[oid] = o
        self.by_head[head] = o
        return o

    def _inv(self, node: Dict[str, Any]) -> None:
        key = tuple(sorted((k, str(v)) for k, v in node.items()))
        if key not in self._inv_seen:
            self._inv_seen.add(key)
            self.invariants.append(node)

    def acquire(self, o: _Obj) -> None:
        o.add("pickupable")
        o.spine = True
        self.order.append({"op": "acquired", "object": o.id})
        self._inv({"op": "exists", "object": o.id})
        if self.last_item is None:
            self._inv({"op": "reachable", "agent": "agent", "target": o.id})
        self.last_item = o.id

    def unlock(self, o: _Obj) -> None:
        o.add("stateful")
        o.add("barrier", {"while": {"attr": "locked", "equals": True}})
        o.add("trigger", {"on": "use",
                          "requires": [self.last_item] if self.last_item else [],
                          "effects": [{"target": "self", "set": {"locked": False}}],
                          "event": "unlock"})
        o.state["locked"] = True
        o.spine = True
        self.order.append({"op": "state_changed", "object": o.id,
                           "attr": "locked", "from": True, "to": False})
        self._inv({"op": "exists", "object": o.id})
        self._inv({"op": "has_component", "object": o.id, "component": "barrier"})

    def place(self, item: _Obj, recep: _Obj) -> None:
        item.add("pickupable")
        item.add("placeable")
        item.spine = True
        recep.add("receptacle", {"accepts": ["placeable"]})
        recep.spine = True
        if not any(p.get("op") == "acquired" and p.get("object") == item.id
                   for p in self.order):
            self.order.append({"op": "acquired", "object": item.id})
            if self.last_item is None:
                self._inv({"op": "reachable", "agent": "agent", "target": item.id})
            self.last_item = item.id
        self.order.append({"op": "placed_in", "object": item.id, "receptacle": recep.id})
        self._inv({"op": "exists", "object": item.id})
        self._inv({"op": "exists", "object": recep.id})
        self._inv({"op": "reachable", "agent": "agent", "target": recep.id})

    def reach(self, o: _Obj) -> None:
        o.add("goal")
        o.spine = True
        self.order.append({"op": "at", "agent": "agent", "target": o.id})
        self._inv({"op": "exists", "object": o.id})
        self._inv({"op": "reachable", "agent": "agent", "target": o.id})

    def trigger(self, o: _Obj) -> None:
        o.add("trigger", {"on": "enter", "effects": []})
        o.spine = True
        self.pending_trigger = o

    def gate(self, o: _Obj) -> None:
        o.add("stateful")
        o.add("barrier", {"while": {"attr": "raised", "equals": False}})
        o.state["raised"] = False
        o.spine = True
        if self.pending_trigger is not None:
            self.pending_trigger.components["trigger"]["effects"] = [
                {"target": o.id, "set": {"raised": True}}
            ]
            self.order.append({"op": "state_changed", "object": o.id,
                               "attr": "raised", "from": False, "to": True})
            self.pending_trigger = None
        self._inv({"op": "exists", "object": o.id})
        self._inv({"op": "has_component", "object": o.id, "component": "barrier"})

    _APPLY = {"acquire": acquire, "unlock": unlock, "reach": reach,
              "trigger": trigger, "gate": gate}

    def parse(self, command: str) -> None:
        tokens = _tokenize(command)
        buf: List[str] = []
        pending_verb: Optional[str] = None
        pending_role: Optional[str] = None
        place_item: Optional[_Obj] = None
        last_verb: Optional[str] = None

        def flush() -> None:
            nonlocal pending_verb, pending_role, place_item, last_verb
            np = list(buf)
            buf.clear()
            o = self.resolve(np)
            if pending_role == "near" and (not o):
                # "near the wall" -> placement hint + a near_wall invariant
                if self.last_obj is not None and any(w in _NONOBJECT for w in np):
                    self.last_obj.near_wall = True
                    self._inv({"op": "near_wall", "object": self.last_obj.id,
                               "within": 1})
                pending_role = None
                return
            if o is None:
                return
            if pending_role == "anchor":
                o.add("surface")
                if self.last_obj is not None:
                    self.last_obj.on_target = o.id
                    self._inv({"op": "exists", "object": o.id})
                    self._inv({"op": "near", "object": self.last_obj.id,
                               "target": o.id, "within": 1})
                pending_role = None
            elif pending_role == "receptacle":
                if place_item is not None:
                    self.place(place_item, o)
                    place_item = None
                pending_role = None
            elif pending_role == "behind":
                o.add("barrier")
                if self.last_obj is not None:
                    o.behind_host = self.last_obj.id
                pending_role = None
            elif pending_role == "near":
                if self.last_obj is not None:
                    self._inv({"op": "near", "object": self.last_obj.id,
                               "target": o.id, "within": 1})
                pending_role = None
            elif pending_verb is not None:
                verb = pending_verb
                pending_verb = None
                if verb == "place":
                    place_item = o
                else:
                    self._APPLY[verb](self, o)
                    last_verb = verb
            self.last_obj = o

        for tok in tokens:
            if tok == ".":                      # sentence boundary: reset all
                flush()
                pending_verb = pending_role = last_verb = None
                continue
            if tok == "and":                    # coordination: distribute the verb
                flush()
                pending_role = None
                pending_verb = last_verb        # "pick up X and Y" -> Y acquired too
                continue
            if tok in _PRONOUNS and pending_verb == "place" and self.last_item:
                place_item = self.objs.get(self.last_item)
                pending_verb = None
                continue
            if tok in _VERB:
                flush()
                pending_verb = _VERB[tok]
                pending_role = None
                continue
            if tok in _PREPS:
                flush()
                if tok in ("in", "into", "inside"):
                    pending_role = "receptacle" if place_item is not None else None
                elif tok in ("on", "onto", "atop"):
                    pending_role = "anchor"
                elif tok == "behind":
                    pending_role = "behind"
                else:
                    pending_role = "near"
                continue
            if tok in _SKIP:                     # a skip-word ENDS the current NP
                if buf:
                    flush()
                continue
            if tok in _ARTICLES:
                continue
            buf.append(tok)
        flush()


def _layout_params(variant: int) -> tuple[int, int, int, int]:
    height = GRID_HEIGHT + 2 * (variant % 2)
    spine_row = min(height - 2, SPINE_ROW + (variant % 2))
    first_x = FIRST_OBJECT_X + (variant % 3)
    step = OBJECT_X_STEP + (1 if variant >= 3 else 0)
    return height, spine_row, first_x, step


def _created_objects(parser: _Parser) -> tuple[List[_Obj], List[_Obj], Dict[str, List[_Obj]]]:
    created = list(parser.objs.values())
    spine_objects = [o for o in created if o.spine and o.behind_host is None]
    host_extras: Dict[str, List[_Obj]] = {}
    for o in created:
        if o.behind_host:
            host_extras.setdefault(o.behind_host, []).append(o)
    return created, spine_objects, host_extras


def _objects_from_positions(created: List[_Obj], pos: Dict[str, List[int]]) -> List[Dict[str, Any]]:
    objects = []
    for o in created:
        if o.id not in pos:
            continue
        objects.append({"id": o.id, "position": pos[o.id],
                        "components": o.components, "state": o.state})
    return objects


def _place_non_gate(o: _Obj, x: int, row: int, height: int,
                    pos: Dict[str, List[int]], parser: _Parser,
                    host_extras: Dict[str, List[_Obj]]) -> None:
    anchor = parser.objs.get(o.on_target) if o.on_target else None
    extras = host_extras.get(o.id, [])
    if anchor is not None:
        arow = 1 if anchor.near_wall else row
        pos[anchor.id] = [x, arow]
        orow = arow + 1 if arow + 1 <= height - 2 else arow - 1
        pos[o.id] = [x, orow]
    elif extras:
        pos[o.id] = [x, max(1, row - 1)]
        pos[extras[0].id] = [x, row]
        for k, ex in enumerate(extras[1:], start=1):
            pos[ex.id] = [x, min(row + k, height - 2)]
    else:
        pos[o.id] = [x, 1 if o.near_wall else row]


def _gate_wall_column(x: int, pass_row: int, height: int, walls: List[List[int]]) -> None:
    for y in range(1, height - 1):
        if y != pass_row:
            walls.append([x, y])


def _layout_spine(parser: _Parser, variant: int = 0) -> Dict[str, Any]:
    height, spine_row, x, step = _layout_params(variant)
    walls: List[List[int]] = []
    pos: Dict[str, List[int]] = {}
    created, spine_objects, host_extras = _created_objects(parser)

    for o in spine_objects:
        is_gate = "barrier" in o.components and "stateful" in o.components
        if is_gate:
            _gate_wall_column(x, spine_row, height, walls)
            pos[o.id] = [x, spine_row]
        else:
            _place_non_gate(o, x, spine_row, height, pos, parser, host_extras)
        x += step

    width = x + 1
    return {"grid": {"width": width, "height": height},
            "walls": walls, "agent": {"start": [1, spine_row]},
            "objects": _objects_from_positions(created, pos)}


def _layout_branch(parser: _Parser, variant: int = 0) -> Dict[str, Any]:
    created, spine_objects, host_extras = _created_objects(parser)
    if len(spine_objects) < 3:
        return _layout_spine(parser, variant=variant)
    height = 9
    width = max(13 + (variant % 2) * 2, 2 * len(spine_objects) + 5)
    main_row = 4
    branch_row = 2 if variant % 2 == 0 else 6
    walls: List[List[int]] = []
    pos: Dict[str, List[int]] = {}
    for i, o in enumerate(spine_objects):
        x = 3 + 2 * i
        row = main_row if i == 0 or i == len(spine_objects) - 1 else (
            branch_row if i % 2 else main_row)
        is_gate = "barrier" in o.components and "stateful" in o.components
        if is_gate:
            _gate_wall_column(x, main_row, height, walls)
            pos[o.id] = [x, main_row]
        else:
            _place_non_gate(o, x, row, height, pos, parser, host_extras)
    # A central divider with two openings makes topology branch/merge instead of
    # a single corridor on each side of the required gate. Keep object cells clear.
    divider_y = main_row + (1 if branch_row < main_row else -1)
    for x in range(2, width - 2):
        if x % 4 not in (1, 3):
            walls.append([x, divider_y])
    walls = [w for w in walls if w not in pos.values()]
    return {"grid": {"width": width, "height": height},
            "walls": walls, "agent": {"start": [1, main_row]},
            "objects": _objects_from_positions(created, pos)}


def _layout_rooms(parser: _Parser, variant: int = 0) -> Dict[str, Any]:
    created, spine_objects, host_extras = _created_objects(parser)
    if len(spine_objects) < 3:
        return _layout_spine(parser, variant=variant)
    width, height = max(15, 2 * len(spine_objects) + 5), 9
    door_row = 4
    walls: List[List[int]] = []
    for x in range(2, width - 2):
        if x not in (3, 11):
            walls.append([x, 2])
    pos: Dict[str, List[int]] = {}
    for i, o in enumerate(spine_objects):
        x = 3 + 2 * i
        row = door_row if i == 0 or i == len(spine_objects) - 1 else (
            6 if i % 2 else door_row)
        is_gate = "barrier" in o.components and "stateful" in o.components
        if is_gate:
            _gate_wall_column(x, door_row, height, walls)
            pos[o.id] = [x, door_row]
        else:
            _place_non_gate(o, x, row, height, pos, parser, host_extras)
    walls = [w for w in walls if w not in pos.values()]
    return {"grid": {"width": width, "height": height},
            "walls": walls, "agent": {"start": [1, 4]},
            "objects": _objects_from_positions(created, pos)}


def _layout(parser: _Parser, variant: int = 0) -> Dict[str, Any]:
    family = variant % 3
    if family == 1:
        return _layout_branch(parser, variant=variant)
    if family == 2:
        return _layout_rooms(parser, variant=variant)
    return _layout_spine(parser, variant=variant)


def _generate_parser(command: str, variant: int = 0) -> Dict[str, Any]:
    parser = _Parser()
    parser.parse(command)
    # Verb-driven only: drop any bare noun that never received a role, so nothing
    # leaks into the world from a noun-list phrasing.
    parser.objs = {oid: o for oid, o in parser.objs.items() if o.components}
    spec = _layout(parser, variant=variant)
    spec["name"] = _slug(command)
    spec["_author"] = "parser"
    spec["invariants"] = parser.invariants
    if parser.order:
        spec["goal"] = {"op": "reward_iff", "of": {"op": "ordered", "of": parser.order}}
    else:
        # Empty parser output should be an honest failed generation, not a
        # vacuous no-op success. This makes noun-list prompts fail visibly while
        # `--author llm/auto` remains the open-ended text path.
        spec["goal"] = {"op": "reward_iff", "of": {"op": "all", "of": [
            {"op": "exists", "object": "__no_objective_generated__"},
            {"op": "no_illegal_actions"},
        ]}}
    return spec


def generate(command: str, variant: int = 0, author: str = "parser",
             model: str = DEFAULT_LLM_MODEL) -> Dict[str, Any]:
    """Generate a component spec from text.

    `parser` is the deterministic offline semantic compiler. `llm` asks an
    LLM to author the JSON spec and lets the engine/verifier sandbox it. `auto`
    uses LLM authoring when the key and SDK are available, else falls back to
    the parser so the challenge remains reproducible without secrets.
    """
    if author not in AUTHOR_CHOICES:
        raise ValueError(f"unknown author {author!r}; expected one of {AUTHOR_CHOICES}")
    if author == "parser":
        return _generate_parser(command, variant=variant)
    if author == "llm":
        return _generate_llm(command, model=model, variant=variant)
    try:
        return _generate_llm(command, model=model, variant=variant)
    except Exception:
        return _generate_parser(command, variant=variant)
