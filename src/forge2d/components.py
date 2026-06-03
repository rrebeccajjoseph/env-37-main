"""The generic component object model for Forge-2D.

Forge-2D worlds are no longer described by a closed set of hardcoded object
*types* (key / door / switch / bridge / can / trash ...). Instead an object is a
bag of **components**, and *all* environment behaviour is derived from the
components an object declares — never from its name or a per-scenario `if`.

This is the substrate the text/spec authoring layers target: a new mechanic is
a new *combination of components + a predicate goal* in a spec file, not new
Python in the engine.

Component vocabulary
--------------------
    pickupable   the agent can PICK UP this object (it enters the inventory)
    barrier      blocks movement (optionally only while a state condition holds)
    receptacle   a held `placeable` can be USE-placed into/next to this
    trigger      firing it (on ENTER or USE) applies effects to object state
    goal         entering this cell emits a `reach` event
    hazard        entering this cell ends the episode (death)
    surface      things can sit on top of it (declarative relation anchor)
    container    can hold other objects (declarative relation anchor)
    relation     declares a relation to another object (`on`, `in`, `near`)
    movable      the object itself can be pushed/relocated (declarative)
    placeable    can be placed into a matching `receptacle`
    stateful     carries an arbitrary `state` dict that predicates can read and
                 triggers can mutate

The first six components are *behavioural* (the generic engine acts on them).
The rest are *declarative*: they enrich the object model and are read by the
predicate DSL (`placed_in`, `at`, `state_changed`, `has_component`) but do not
themselves drive engine logic. Keeping the full vocabulary in one place means a
spec author — human or LLM — has a single, stable contract to write against.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

# --------------------------------------------------------------------------- #
# Component names (the stable contract)
# --------------------------------------------------------------------------- #
PICKUPABLE = "pickupable"
BARRIER = "barrier"
RECEPTACLE = "receptacle"
TRIGGER = "trigger"
GOAL = "goal"
HAZARD = "hazard"
SURFACE = "surface"
CONTAINER = "container"
RELATION = "relation"
MOVABLE = "movable"
PLACEABLE = "placeable"
STATEFUL = "stateful"

#: Components the generic engine actively simulates.
BEHAVIOURAL_COMPONENTS = {PICKUPABLE, BARRIER, RECEPTACLE, TRIGGER, GOAL, HAZARD}

#: Components that are declarative metadata read by predicates.
DECLARATIVE_COMPONENTS = {SURFACE, CONTAINER, RELATION, MOVABLE, PLACEABLE, STATEFUL}

#: Every legal component name.
COMPONENTS = BEHAVIOURAL_COMPONENTS | DECLARATIVE_COMPONENTS


# A component map is {name: config_dict}; an author may also write a bare list
# of names (configs default to {}). `ComponentSpec` captures both input shapes.
ComponentSpec = Union[List[str], Dict[str, Dict[str, Any]]]


def normalize_components(spec: ComponentSpec) -> Dict[str, Dict[str, Any]]:
    """Normalize a components declaration to ``{name: config}``.

    Accepts either a list of component names (``["pickupable", "placeable"]``)
    or a mapping of name -> config (``{"barrier": {"while": {...}}}``). Unknown
    component names raise, so a typo in a spec fails loudly instead of silently
    producing an inert object.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(spec, dict):
        items = spec.items()
    elif isinstance(spec, list):
        items = ((name, {}) for name in spec)
    else:
        raise TypeError(
            f"components must be a list or dict, got {type(spec).__name__}"
        )
    for name, config in items:
        if name not in COMPONENTS:
            raise ValueError(
                f"unknown component {name!r}; valid components are "
                f"{sorted(COMPONENTS)}"
            )
        out[name] = dict(config or {})
    return out


# --------------------------------------------------------------------------- #
# Component-role predicates
# --------------------------------------------------------------------------- #
# These let every downstream module reason about an object's ROLE purely from
# its components — no `obj.type == TYPE_DOOR`. A "door" is just a lockable
# barrier; an "obstacle" is just a permanent barrier. They duck-type on any
# object exposing `.has`, `.components` and `.state`.


def is_pickupable(o: Any) -> bool:
    return o.has(PICKUPABLE)


def is_goal(o: Any) -> bool:
    return o.has(GOAL)


def is_hazard(o: Any) -> bool:
    return o.has(HAZARD)


def is_barrier(o: Any) -> bool:
    return o.has(BARRIER)


def is_locked(o: Any) -> bool:
    return o.state.get("locked") is True


def is_lockable_barrier(o: Any) -> bool:
    """A "door": a barrier whose passability is gated by a `locked` state attr."""
    return o.has(BARRIER) and "locked" in o.state


def is_permanent_barrier(o: Any) -> bool:
    """An "obstacle": an always-active barrier (no conditional `while`)."""
    return o.has(BARRIER) and "while" not in o.components.get(BARRIER, {})


def required_item(o: Any) -> Optional[str]:
    """The item id a use-trigger requires (e.g. the key a door needs), or None."""
    reqs = o.components.get(TRIGGER, {}).get("requires") or []
    return reqs[0] if reqs else None
