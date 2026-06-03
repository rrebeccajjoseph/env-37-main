"""Controller-style 2D policy harness.

General Intuition's policy interface is controller-shaped: rendered observation
in, low-level action out. This module exposes the same idea for Forge-2D:

    observation = RGB frame of the live gridworld
    action      = move_forward/move_backward/move_left/move_right + look deltas

The built-in scripted policy is deliberately simple. It is not a learned vision
policy; it is a baseline driver for the interface. The important boundary is
that rollout happens step-by-step through observations and controller actions,
not by replaying a finished trace through the renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import components as C
from .generic import FLOOR, WALL, GenericEnvironment, barrier_active
from .schemas import MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, PICKUP, USE, WAIT

MOVE_FORWARD = "move_forward"
MOVE_BACKWARD = "move_backward"
CTRL_MOVE_LEFT = "move_left"
CTRL_MOVE_RIGHT = "move_right"
LOOK_DELTA_X = "mouse_delta_x"
LOOK_DELTA_Y = "mouse_delta_y"
INTERACT = "interact"
CTRL_PICKUP = "pickup"
CTRL_WAIT = "wait"

CONTROLLER_ACTIONS = [
    MOVE_FORWARD,
    MOVE_BACKWARD,
    CTRL_MOVE_LEFT,
    CTRL_MOVE_RIGHT,
    LOOK_DELTA_X,
    LOOK_DELTA_Y,
    INTERACT,
    CTRL_PICKUP,
    CTRL_WAIT,
]

# Compact RGB contract for tests/policies. The renderer can be pretty; this is
# intentionally stable and machine-readable.
C_WALL = (38, 42, 51)
C_FLOOR = (224, 227, 232)
C_AGENT = (54, 110, 230)
C_PICKUPABLE = (240, 200, 40)
C_BARRIER = (165, 84, 44)
C_OPEN_BARRIER = (96, 186, 96)
C_GOAL = (190, 70, 200)
C_HAZARD = (214, 64, 64)
C_TRIGGER = (240, 150, 40)
C_RECEPTACLE = (90, 150, 200)
C_OBJECT = (150, 150, 158)


@dataclass
class ControllerCommand:
    move_forward: float = 0.0
    move_backward: float = 0.0
    move_left: float = 0.0
    move_right: float = 0.0
    mouse_delta_x: float = 0.0
    mouse_delta_y: float = 0.0
    pickup: bool = False
    interact: bool = False

    def primary(self) -> str:
        if self.pickup:
            return CTRL_PICKUP
        if self.interact:
            return INTERACT
        moves = [
            (abs(self.move_forward), MOVE_FORWARD),
            (abs(self.move_backward), MOVE_BACKWARD),
            (abs(self.move_left), CTRL_MOVE_LEFT),
            (abs(self.move_right), CTRL_MOVE_RIGHT),
        ]
        strength, action = max(moves)
        return action if strength > 0 else CTRL_WAIT


def command(action: str) -> ControllerCommand:
    if action == MOVE_FORWARD:
        return ControllerCommand(move_forward=1.0)
    if action == MOVE_BACKWARD:
        return ControllerCommand(move_backward=1.0)
    if action == CTRL_MOVE_LEFT:
        return ControllerCommand(move_left=1.0)
    if action == CTRL_MOVE_RIGHT:
        return ControllerCommand(move_right=1.0)
    if action == CTRL_PICKUP:
        return ControllerCommand(pickup=True)
    if action == INTERACT:
        return ControllerCommand(interact=True)
    return ControllerCommand()


def engine_to_controller(action: str) -> str:
    """Map grid moves to a fixed-camera 2D controller.

    The camera faces north in this 2D harness, so forward/back/left/right map to
    up/down/left/right. Mouse deltas are exposed but unused by the baseline.
    """
    return {
        MOVE_UP: MOVE_FORWARD,
        MOVE_DOWN: MOVE_BACKWARD,
        MOVE_LEFT: CTRL_MOVE_LEFT,
        MOVE_RIGHT: CTRL_MOVE_RIGHT,
        PICKUP: CTRL_PICKUP,
        USE: INTERACT,
        WAIT: CTRL_WAIT,
    }.get(action, CTRL_WAIT)


def controller_to_engine(cmd: ControllerCommand) -> str:
    action = cmd.primary()
    return {
        MOVE_FORWARD: MOVE_UP,
        MOVE_BACKWARD: MOVE_DOWN,
        CTRL_MOVE_LEFT: MOVE_LEFT,
        CTRL_MOVE_RIGHT: MOVE_RIGHT,
        CTRL_PICKUP: PICKUP,
        INTERACT: USE,
        CTRL_WAIT: WAIT,
    }[action]


def _object_color(o) -> Tuple[int, int, int]:
    if o.has(C.HAZARD):
        return C_HAZARD
    if o.has(C.GOAL):
        return C_GOAL
    if o.has(C.BARRIER):
        return C_BARRIER if barrier_active(o) else C_OPEN_BARRIER
    if o.has(C.TRIGGER):
        return C_TRIGGER
    if o.has(C.RECEPTACLE):
        return C_RECEPTACLE
    if o.has(C.PICKUPABLE):
        return C_PICKUPABLE
    return C_OBJECT


def render_frame(env: GenericEnvironment, scale: int = 8) -> List[List[List[int]]]:
    """Return an RGB frame as nested Python lists: frame[y][x] = [r, g, b]."""
    cell_colors: List[List[Tuple[int, int, int]]] = []
    for y in range(env.height):
        row = []
        for x in range(env.width):
            row.append(C_WALL if env.terrain[y][x] == WALL else C_FLOOR)
        cell_colors.append(row)
    for o in env.objects.values():
        if o.visible():
            x, y = o.position
            if env.in_bounds(x, y):
                cell_colors[y][x] = _object_color(o)
    ax, ay = env.agent_pos
    if env.in_bounds(ax, ay):
        cell_colors[ay][ax] = C_AGENT

    frame: List[List[List[int]]] = []
    for row in cell_colors:
        expanded_rows = []
        for _ in range(scale):
            pixels: List[List[int]] = []
            for color in row:
                pixels.extend([list(color) for _ in range(scale)])
            expanded_rows.append(pixels)
        frame.extend(expanded_rows)
    return frame


def observation(env: GenericEnvironment, scale: int = 8) -> Dict[str, Any]:
    frame = render_frame(env, scale=scale)
    return {
        "frame": frame,
        "frame_width": len(frame[0]) if frame else 0,
        "frame_height": len(frame),
        "channels": 3,
        "controller_actions": list(CONTROLLER_ACTIONS),
    }


class ScriptedControllerPolicy:
    """Baseline policy for the controller interface.

    It consumes the live rendered observation every tick. The next intended
    engine action is supplied by the harness's symbolic scaffold so the baseline
    can demonstrate the low-level interface without pretending to be a learned
    visual navigator.
    """

    def next_action(self, obs: Dict[str, Any], planned_engine_action: Optional[str]) -> ControllerCommand:
        if not obs.get("frame"):
            return command(CTRL_WAIT)
        if planned_engine_action is None:
            return command(CTRL_WAIT)
        return command(engine_to_controller(planned_engine_action))

