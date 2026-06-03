"""Diversity-aware generation for the generic component path (anti-collapse).

Generators tend to emit shallow variants of the same world. We defend against
that with a **buffer of layout embeddings**: each world is reduced to a
scale-normalized feature vector built from its *components* plus its object
**adjacency** (how many object pairs sit next to each other). `generate` probes
layout variants against this buffer and selects the first structurally novel
candidate before running it. If every candidate is still too similar, the final
report flags the collapse. Everything here is component-derived, never
names/types.
"""
from __future__ import annotations

from collections import deque
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from .components import (
    is_goal, is_hazard, is_lockable_barrier, is_permanent_barrier, is_pickupable,
)
from .generic import GenericEnvironment

DEFAULT_BUFFER_PATH = os.environ.get(
    "FORGE2D_DIVERSITY_BUFFER",
    os.path.abspath(".forge2d_diversity_buffer.json"),
)


# --------------------------------------------------------------------------- #
# Feature extraction (the "embedding")
# --------------------------------------------------------------------------- #
def adjacency_matrix(env: GenericEnvironment) -> List[List[int]]:
    """Symmetric object-adjacency matrix: 1 where two objects are within one
    tile (Manhattan). Object order is by sorted id for determinism."""
    objs = sorted(env.objects.values(), key=lambda o: o.id)
    n = len(objs)
    m = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            ax, ay = objs[i].position
            bx, by = objs[j].position
            if abs(ax - bx) + abs(ay - by) <= 1:
                m[i][j] = m[j][i] = 1
    return m


def layout_features(env: GenericEnvironment, sequence_depth: int,
                    path_length: int) -> Dict[str, Any]:
    """A structural summary of the world, all derived from components."""
    objs = list(env.objects.values())
    adj = adjacency_matrix(env)
    edges = sum(sum(row) for row in adj) // 2
    free = [(x, y) for y in range(1, env.height - 1)
            for x in range(1, env.width - 1)
            if not env.is_wall(x, y)]
    degree_hist = {1: 0, 2: 0, 3: 0, 4: 0}
    junctions = dead_ends = 0
    for x, y in free:
        deg = sum(1 for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
                  if not env.is_wall(x + dx, y + dy))
        degree_hist[min(4, max(1, deg))] += 1
        if deg >= 3:
            junctions += 1
        if deg == 1:
            dead_ends += 1
    rooms = _connected_floor_components(env)
    return {
        "objects": len(objs),
        "pickupable": sum(1 for o in objs if is_pickupable(o)),
        "lockable_barriers": sum(1 for o in objs if is_lockable_barrier(o)),
        "static_barriers": sum(1 for o in objs if is_permanent_barrier(o)),
        "goals": sum(1 for o in objs if is_goal(o)),
        "triggers": sum(1 for o in objs if o.has("trigger")),
        "receptacles": sum(1 for o in objs if o.has("receptacle")),
        "hazards": sum(1 for o in objs if is_hazard(o)),
        "width": env.width,
        "height": env.height,
        "sequence_depth": sequence_depth,
        "path_length": path_length,
        "adjacency_edges": edges,
        "junctions": junctions,
        "dead_ends": dead_ends,
        "rooms": rooms,
        "open_area": len(free),
    }


def _connected_floor_components(env: GenericEnvironment) -> int:
    seen = set()
    comps = 0
    for y in range(1, env.height - 1):
        for x in range(1, env.width - 1):
            if env.is_wall(x, y) or (x, y) in seen:
                continue
            comps += 1
            q = deque([(x, y)])
            seen.add((x, y))
            while q:
                cx, cy = q.popleft()
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    nb = (cx + dx, cy + dy)
                    if nb in seen or env.is_wall(*nb):
                        continue
                    seen.add(nb)
                    q.append(nb)
    return comps


_VECTOR_KEYS = ("objects", "pickupable", "lockable_barriers", "static_barriers",
                "goals", "triggers", "receptacles", "hazards", "width", "height",
                "sequence_depth", "path_length", "adjacency_edges", "junctions",
                "dead_ends", "rooms", "open_area")

# Per-dimension scale so no single feature (e.g. width/path) dominates the
# distance. Each scaled feature is ~O(1), so structurally different worlds are
# genuinely far apart instead of all pointing the same direction.
_SCALE = {"objects": 3, "pickupable": 2, "lockable_barriers": 2,
          "static_barriers": 2, "goals": 1, "triggers": 2, "receptacles": 2,
          "hazards": 2, "width": 10, "height": 6, "sequence_depth": 3,
          "path_length": 10, "adjacency_edges": 3, "junctions": 8,
          "dead_ends": 4, "rooms": 2, "open_area": 40}


def feature_vector(features: Dict[str, Any]) -> List[float]:
    """Scale-normalized layout embedding (so distance is meaningful)."""
    return [float(features.get(k, 0)) / _SCALE[k] for k in _VECTOR_KEYS]


# Difficulty weights for the iterative "make it harder" curriculum. A world is
# "better" when it demands a longer required action sequence over more
# interacting parts and a longer/branchier solution path -- NOT when it merely
# has more empty floor. `open_area` is deliberately excluded so "bigger empty
# room" never counts as "harder".
_DIFFICULTY = {
    "sequence_depth": 5, "objects": 2, "pickupable": 1, "lockable_barriers": 2,
    "static_barriers": 1, "triggers": 2, "receptacles": 2, "hazards": 1,
    "path_length": 1, "junctions": 1,
}


def complexity_score(features: Dict[str, Any]) -> float:
    """Scalar difficulty of a world derived from its structural features.

    Used to enforce strictly-increasing curricula: an "improved" world must
    score higher than the one it extends, on top of staying playable,
    task-verified, and exploit-safe. Higher = longer required sequence, more
    interacting objects, longer solution path.
    """
    return float(sum(features.get(k, 0) * w for k, w in _DIFFICULTY.items()))


def _similarity(a: List[float], b: List[float]) -> float:
    """1.0 for identical layouts, decreasing with structural distance."""
    if len(a) != len(b):
        return 0.0
    dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return max(0.0, 1.0 - dist / math.sqrt(len(a)))


# --------------------------------------------------------------------------- #
# The buffer
# --------------------------------------------------------------------------- #
class DiversityBuffer:
    """Stores layout embeddings; flags worlds that are too similar to past ones.

    `path=None` keeps it in memory (deterministic for tests). The CLI passes a
    persistent path so collapse is detected across separate invocations.
    """

    def __init__(self, path: Optional[str] = None, threshold: float = 0.9):
        self.path = path
        self.threshold = threshold
        self.vectors: List[List[float]] = self._load()

    def _load(self) -> List[List[float]]:
        if not self.path or not os.path.exists(self.path):
            return []
        try:
            with open(self.path) as f:
                return [list(map(float, v)) for v in json.load(f)]
        except Exception:  # noqa: BLE001 - corrupt buffer -> start fresh
            return []

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.vectors, f)
        except OSError:
            pass

    def assess(self, vec: List[float]) -> Tuple[float, bool]:
        """Return (max similarity to buffer, is_novel)."""
        sim = max((_similarity(vec, v) for v in self.vectors), default=0.0)
        return sim, sim < self.threshold

    def add(self, vec: List[float]) -> None:
        self.vectors.append(vec)
        self._save()
