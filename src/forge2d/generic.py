"""The generic, component-driven 2D gridworld.

This engine has no knowledge of keys, doors, switches, bridges, cans or any
other named mechanic.
Every behaviour is derived from the **components** an object declares
(`components.py`):

    * BARRIER     -> the cell blocks movement (optionally only while a state
                     condition holds, e.g. a door while `locked`, a bridge while
                     not `raised`)
    * PICKUPABLE  -> PICKUP on the cell moves the object into the inventory
    * GOAL        -> entering the cell emits a `reach <id>` event
    * HAZARD      -> entering the cell ends the episode
    * TRIGGER     -> entering (or USE-ing adjacent, per config) applies effects
                     to object state, emitting `state <id> <attr>=<value>` events
    * RECEPTACLE  -> a held PLACEABLE item can be USE-placed here

Because mechanics are data, the *same* engine runs a key/door world, a
switch/bridge world, or a place-the-trash world with no code change — the
difference lives entirely in the spec file.

Coordinates are (x, y); the grid is indexed terrain[y][x].

Event vocabulary:
    "pickup <id>", "place <item> in <recep>", "state <id> <attr>=<value>",
    "trigger <id>", "reach <id>", "hazard <id>", "blocked x,y", "invalid <a>".
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from . import components as C
from .components import normalize_components
from .schemas import MOVE_DELTAS, PICKUP, USE, WAIT

WALL = "#"
FLOOR = "."

Cell = Tuple[int, int]

# Orthogonal neighbourhood including the agent's own cell (for USE / placement).
_ADJ = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))


def fmt_value(v: Any) -> str:
    """Canonical string form of a state value so engine events and the predicate
    DSL agree (`True` -> "true", so `state door locked=false` is comparable)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


# --------------------------------------------------------------------------- #
# Shared component mechanics
# --------------------------------------------------------------------------- #
# These pure functions are the SINGLE source of truth for how components behave.
# There is zero `if type == ...` branching in the simulation.


def barrier_active(o: Any) -> bool:
    """A BARRIER blocks unless its optional `while` state-condition is unmet.
    `{"while": {"attr": "raised", "equals": false}}` blocks only while
    `state["raised"] == false`. No `while` -> always blocks (wall-like prop)."""
    cfg = o.components.get(C.BARRIER, {})
    cond = cfg.get("while")
    if cond is None:
        return True
    return o.state.get(cond.get("attr")) == cond.get("equals")


def trigger_ready(o: Any, inventory: List[str], when: str) -> bool:
    """The trigger is wired for this interaction (`enter`/`use`) and its required
    items are all held."""
    cfg = o.components.get(C.TRIGGER, {})
    if cfg.get("on", "enter") != when:
        return False
    return all(req in inventory for req in (cfg.get("requires") or []))


def trigger_would_change(o: Any, objects: Dict[str, Any]) -> bool:
    """True if firing the trigger would actually mutate some state (or it is a
    pure event trigger with no effects). Prevents a settled trigger — an
    already-unlocked door, an already-raised bridge — from re-firing."""
    effects = o.components.get(C.TRIGGER, {}).get("effects", [])
    if not effects:
        return True
    for eff in effects:
        tid = eff.get("target", "self")
        target = o if tid in ("self", o.id) else objects.get(tid)
        if target is None:
            continue
        for attr, value in eff.get("set", {}).items():
            if target.state.get(attr) != value:
                return True
    return False


def can_fire(o: Any, inventory: List[str], objects: Dict[str, Any], when: str) -> bool:
    return trigger_ready(o, inventory, when) and trigger_would_change(o, objects)


def apply_effects(o: Any, objects: Dict[str, Any]) -> List[str]:
    """Apply a trigger's effects, returning events: the trigger's event label
    (default "trigger", but e.g. "unlock" for a door) followed by one
    `state <id> <attr>=<value>` per real change."""
    cfg = o.components.get(C.TRIGGER, {})
    state_evs: List[str] = []
    for eff in cfg.get("effects", []):
        tid = eff.get("target", "self")
        target = o if tid in ("self", o.id) else objects.get(tid)
        if target is None:
            continue
        for attr, value in eff.get("set", {}).items():
            if target.state.get(attr) != value:
                target.state[attr] = value
                state_evs.append(f"state {target.id} {attr}={fmt_value(value)}")
    label = cfg.get("event", "trigger")
    return [f"{label} {o.id}"] + state_evs


def receptacle_accepts(recep: Any, item: Any) -> bool:
    accepts = recep.components.get(C.RECEPTACLE, {}).get("accepts")
    if not accepts:
        return True  # any placeable
    if item.id in accepts:
        return True
    return any(comp in accepts for comp in item.components)


@dataclass
class GenericObject:
    id: str
    position: List[int]
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    picked: bool = False
    placed: bool = False

    @property
    def cell(self) -> Cell:
        return (self.position[0], self.position[1])

    def has(self, component: str) -> bool:
        return component in self.components

    def visible(self) -> bool:
        """On the grid (not carried in inventory, not consumed into a bin)."""
        return not self.picked and not self.placed


class GenericEnvironment:
    """A deterministic gridworld whose rules are read from object components.

    Success is intentionally NOT decided here — the spec's predicate goal
    (`predicates.py`) decides it from the trace. The engine only simulates
    physics and emits events.
    """

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.name = spec.get("name", "world")
        grid = spec["grid"]
        self.width = grid["width"]
        self.height = grid["height"]

        self.terrain: List[List[str]] = [
            [FLOOR] * self.width for _ in range(self.height)
        ]
        for x in range(self.width):
            self.terrain[0][x] = WALL
            self.terrain[self.height - 1][x] = WALL
        for y in range(self.height):
            self.terrain[y][0] = WALL
            self.terrain[y][self.width - 1] = WALL
        for wx, wy in spec.get("walls", []):
            if 0 <= wy < self.height and 0 <= wx < self.width:
                self.terrain[wy][wx] = WALL

        self._initial_objects: List[GenericObject] = []
        for o in spec.get("objects", []):
            self._initial_objects.append(
                GenericObject(
                    id=o["id"],
                    position=list(o["position"]),
                    components=normalize_components(o.get("components", {})),
                    state=dict(o.get("state", {})),
                )
            )
        self.start: Cell = tuple(spec.get("agent", {}).get("start", [1, 1]))  # type: ignore
        self.max_steps = int(spec.get("max_steps", 200))

        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self) -> Dict:
        self.objects: Dict[str, GenericObject] = {
            o.id: copy.deepcopy(o) for o in self._initial_objects
        }
        self.agent_pos: List[int] = list(self.start)
        self.inventory: List[str] = []
        self.t = 0
        self.done = False
        self.dead = False
        self.success = False          # set by the runner from the predicate goal
        self.last_reward = 0.0
        self.event_log: List[str] = []
        self.illegal_actions = 0
        return self.get_state()

    # ----------------------------------------------------------- terrain query
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, x: int, y: int) -> bool:
        return not self.in_bounds(x, y) or self.terrain[y][x] == WALL

    def object_at(self, x: int, y: int) -> Optional[GenericObject]:
        for o in self.objects.values():
            if o.visible() and o.position[0] == x and o.position[1] == y:
                return o
        return None

    def objects_with(self, component: str) -> List[GenericObject]:
        return [o for o in self.objects.values() if o.visible() and o.has(component)]

    # -------------------------------------------------- component-derived rules
    def blocks_movement(self, x: int, y: int) -> bool:
        if self.is_wall(x, y):
            return True
        o = self.object_at(x, y)
        if o is not None and o.has(C.BARRIER) and barrier_active(o):
            return True
        return False

    def is_hazard(self, x: int, y: int) -> bool:
        o = self.object_at(x, y)
        return o is not None and o.has(C.HAZARD)

    # ------------------------------------------------------------ pathfinding
    def bfs(self, start: Cell, goals: Set[Cell],
            passable: Optional[Callable[[int, int], bool]] = None) -> Optional[List[Cell]]:
        """Shortest path (excluding start) to the nearest goal cell, or None."""
        if passable is None:
            passable = lambda x, y: (not self.blocks_movement(x, y)
                                     and not self.is_hazard(x, y))
        if start in goals:
            return []
        came: Dict[Cell, Optional[Cell]] = {start: None}
        q = deque([start])
        while q:
            cx, cy = q.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nb = (cx + dx, cy + dy)
                if nb in came or not self.in_bounds(*nb):
                    continue
                if not passable(*nb):
                    continue
                came[nb] = (cx, cy)
                if nb in goals:
                    path = [nb]
                    while came[path[-1]] != start:
                        path.append(came[path[-1]])  # type: ignore
                    path.reverse()
                    return path
                q.append(nb)
        return None

    # -------------------------------------------------------------------- step
    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        events: List[str] = []
        reward = 0.0
        if self.done:
            return self.get_state(), 0.0, True, {"events": events}
        self.t += 1

        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if self.blocks_movement(nx, ny):
                self.illegal_actions += 1
                events.append(f"blocked {nx},{ny}")
            else:
                self.agent_pos = [nx, ny]
                o = self.object_at(nx, ny)
                if o is not None and o.has(C.HAZARD):
                    self.dead = True
                    self.done = True
                    reward = -1.0
                    events.append(f"hazard {o.id}")
                else:
                    if o is not None and o.has(C.GOAL):
                        events.append(f"reach {o.id}")
                    if o is not None and o.has(C.TRIGGER) and \
                            can_fire(o, self.inventory, self.objects, "enter"):
                        events.extend(apply_effects(o, self.objects))

        elif action == PICKUP:
            o = self.object_at(*self.agent_pos)
            if o is not None and o.has(C.PICKUPABLE) and not o.picked:
                o.picked = True
                self.inventory.append(o.id)
                events.append(f"pickup {o.id}")
            else:
                self.illegal_actions += 1

        elif action == USE:
            used = self._use()
            if used:
                events.extend(used)
            else:
                self.illegal_actions += 1

        elif action == WAIT:
            pass
        else:
            self.illegal_actions += 1
            events.append(f"invalid {action}")

        self.event_log.extend(events)
        self.last_reward = reward
        return self.get_state(), reward, self.done, {"events": events}

    # ----------------------------------------------------------- interactions
    def _use(self) -> List[str]:
        """USE resolves to the first applicable interaction adjacent to the agent:
        fire a use-trigger, then try to place a held item into a receptacle."""
        ax, ay = self.agent_pos
        # 1) a use-trigger whose requirements are met and would change state
        for dx, dy in _ADJ:
            o = self.object_at(ax + dx, ay + dy)
            if o is not None and o.has(C.TRIGGER) and \
                    can_fire(o, self.inventory, self.objects, "use"):
                return apply_effects(o, self.objects)
        # 2) place a held placeable into an adjacent receptacle
        placed = self._try_place()
        if placed is not None:
            return placed
        return []

    def _try_place(self) -> Optional[List[str]]:
        held = [self.objects[i] for i in self.inventory
                if i in self.objects and self.objects[i].has(C.PLACEABLE)]
        if not held:
            return None
        ax, ay = self.agent_pos
        for dx, dy in _ADJ:
            recep = self.object_at(ax + dx, ay + dy)
            if recep is None or not recep.has(C.RECEPTACLE):
                continue
            for item in held:
                if receptacle_accepts(recep, item):
                    self.inventory.remove(item.id)
                    item.placed = True
                    return [f"place {item.id} in {recep.id}"]
        return None

    # ------------------------------------------------------------------ state
    def get_state(self) -> Dict:
        return {
            "t": self.t,
            "agent_pos": list(self.agent_pos),
            "inventory": list(self.inventory),
            "objects": {
                o.id: {
                    "position": list(o.position),
                    "components": sorted(o.components),
                    "state": dict(o.state),
                    "picked": o.picked,
                    "placed": o.placed,
                }
                for o in self.objects.values()
            },
            "events": list(self.event_log),
            "done": self.done,
            "dead": self.dead,
            "success": self.success,
            "reward": self.last_reward,
        }

    # --------------------------------------------------------------- rendering
    def _glyph(self, o: GenericObject) -> str:
        """A single ASCII glyph chosen from the object's components (priority
        order keeps the most salient behaviour visible)."""
        if o.has(C.HAZARD):
            return "X"
        if o.has(C.GOAL):
            return "G"
        if o.has(C.BARRIER):
            return "B" if barrier_active(o) else "b"
        if o.has(C.TRIGGER):
            return "S"
        if o.has(C.RECEPTACLE):
            return "R"
        if o.has(C.PICKUPABLE):
            return "P"
        return "o"

    def render_ascii(self) -> str:
        grid = [row[:] for row in self.terrain]
        for o in self.objects.values():
            if not o.visible():
                continue
            ox, oy = o.position
            if self.in_bounds(ox, oy):
                grid[oy][ox] = self._glyph(o)
        ax, ay = self.agent_pos
        if self.in_bounds(ax, ay):
            grid[ay][ax] = "A"
        return "\n".join("".join(row) for row in grid)
