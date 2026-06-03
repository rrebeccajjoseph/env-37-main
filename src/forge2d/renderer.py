"""Pygame renderer for Forge-2D.

Rendering is a *view* over the component world — it never decides anything; the
engine and verifiers stay fully headless.

Design notes:
  * pygame is imported lazily (inside functions), so importing forge2d or
    running the verifiers headlessly never requires pygame or a display.
  * `render_spec_run` replays the EXACT actions recorded in the run's trace on a
    fresh GenericEnvironment, so the animation always matches the verified result.
  * Colours and glyphs are derived from object COMPONENTS, never names/types.
  * If no display is available it falls back to SDL's dummy driver rather than
    crashing, so it is safe in CI.
"""
from __future__ import annotations

import os
from typing import Dict

from .components import (
    is_barrier, is_goal, is_hazard, is_lockable_barrier, is_locked,
    is_permanent_barrier, is_pickupable,
)
from .generic import WALL, barrier_active

CELL = 44
STATUS_H = 56

# RGB palette
C_WALL = (38, 42, 51)
C_FLOOR = (224, 227, 232)
C_GRID = (196, 200, 206)
C_PICKUPABLE = (240, 200, 40)
C_LOCKED_BARRIER = (165, 84, 44)
C_OPEN_BARRIER = (96, 186, 96)
C_GOAL = (190, 70, 200)
C_HAZARD = (214, 64, 64)
C_STATIC_BARRIER = (140, 120, 96)
C_AGENT = (54, 110, 230)
C_STATUS_BG = (28, 30, 36)
C_STATUS_FG = (232, 234, 238)
C_OK = (80, 210, 110)
C_FAIL = (226, 76, 76)
C_TRIGGER = (240, 150, 40)
C_RECEPTACLE = (90, 150, 200)
C_PROP = (150, 150, 158)


def available() -> bool:
    try:
        import pygame  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _glyph(pygame, screen, font, ch, x, y, color) -> None:
    label = font.render(ch, True, color)
    rect = label.get_rect(center=(x * CELL + CELL // 2, y * CELL + CELL // 2))
    screen.blit(label, rect)


def _open(pygame, env, caption: str):
    w, h = env.width * CELL, env.height * CELL + STATUS_H
    try:
        screen = pygame.display.set_mode((w, h))
    except pygame.error:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.display.quit()
        pygame.display.init()
        screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(caption)
    return screen


# --------------------------------------------------------------------------- #
# Generic spec animation (component world + predicate goal)
# --------------------------------------------------------------------------- #
def _obj_color_glyph(o):
    """Colour + glyph chosen purely from an object's COMPONENTS (its role).

    Barriers are handled generically via `barrier_active`, so a locked door, an
    un-raised bridge, or a static obstacle all read as 'currently blocking'
    (brown) vs 'open/passable' (green) — no per-state special-casing.
    """
    if is_hazard(o):
        return C_HAZARD, "X"
    if is_goal(o):
        return C_GOAL, "G"
    if is_barrier(o):
        if barrier_active(o):
            # brown; static obstacles use B, openable barriers (door/bridge) D
            return C_LOCKED_BARRIER, ("B" if is_permanent_barrier(o) else "D")
        return C_OPEN_BARRIER, "d"          # an opened door / raised bridge
    if o.has("receptacle"):
        return C_RECEPTACLE, "R"
    if o.has("trigger"):
        return C_TRIGGER, "S"
    if is_pickupable(o):
        return C_PICKUPABLE, "P"
    return C_PROP, "o"


def _draw_generic(pygame, screen, font, big, env, subtitle, last_action, banner):
    screen.fill(C_FLOOR)
    for y in range(env.height):
        for x in range(env.width):
            rect = (x * CELL, y * CELL, CELL, CELL)
            pygame.draw.rect(screen, C_WALL if env.terrain[y][x] == WALL else C_FLOOR, rect)
            pygame.draw.rect(screen, C_GRID, rect, 1)

    for o in env.objects.values():
        if not o.visible():
            continue
        ox, oy = o.position
        color, ch = _obj_color_glyph(o)
        pygame.draw.rect(screen, color,
                         (ox * CELL + 5, oy * CELL + 5, CELL - 10, CELL - 10))
        _glyph(pygame, screen, font, ch, ox, oy, (30, 30, 30))

    ax, ay = env.agent_pos
    pygame.draw.circle(screen, C_AGENT,
                       (ax * CELL + CELL // 2, ay * CELL + CELL // 2), CELL // 3)

    bar_y = env.height * CELL
    pygame.draw.rect(screen, C_STATUS_BG, (0, bar_y, env.width * CELL, STATUS_H))
    line2 = f"t={env.t}  action={last_action}  inv={env.inventory or '[]'}"
    screen.blit(font.render(subtitle, True, C_STATUS_FG), (8, bar_y + 7))
    screen.blit(font.render(line2, True, C_STATUS_FG), (8, bar_y + 30))

    if banner:
        color = C_OK if banner == "SUCCESS" else C_FAIL
        overlay = pygame.Surface((env.width * CELL, env.height * CELL), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 90))
        screen.blit(overlay, (0, 0))
        text = big.render(banner, True, color)
        screen.blit(text, text.get_rect(center=(env.width * CELL // 2, env.height * CELL // 2)))


def render_spec_run(result: Dict, fps: int = 6, hold_ms: int = 1800) -> None:
    """Animate a `spec.run_spec` result: replays the recorded oracle trace on a
    fresh GenericEnvironment and draws component-derived frames."""
    try:
        import pygame
    except Exception:  # noqa: BLE001
        print("[render] pygame not installed — skipping window "
              "(`pip install pygame`, or pass --no-render)")
        return
    from .generic import GenericEnvironment

    try:
        pygame.init()
        font = pygame.font.SysFont("monospace", 15)
        big = pygame.font.SysFont("monospace", 44, bold=True)

        env = GenericEnvironment(result["spec"])
        screen = _open(pygame, env, f"Forge-2D — {result['name']}")
        actions = [s["action"] for s in result.get("trace", []) if s.get("action")]
        subtitle = f"{result['name']}  |  generic component spec"
        banner = "SUCCESS" if result.get("success") else "FAILURE"

        clock = pygame.time.Clock()

        def pump() -> bool:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False
            return True

        _draw_generic(pygame, screen, font, big, env, subtitle, "(start)", None)
        pygame.display.flip()
        pygame.time.delay(400)
        for a in actions:
            if not pump():
                break
            env.step(a)
            _draw_generic(pygame, screen, font, big, env, subtitle, a, None)
            pygame.display.flip()
            clock.tick(max(1, fps))
        _draw_generic(pygame, screen, font, big, env, subtitle, "(done)", banner)
        pygame.display.flip()
        waited = 0
        while waited < hold_ms:
            if not pump():
                break
            pygame.time.delay(50)
            waited += 50
    finally:
        try:
            pygame.quit()
        except Exception:  # noqa: BLE001
            pass
