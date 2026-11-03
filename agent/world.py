"""Static world model for Pokemon Red, built from ROM-derived data.

The previous system *learned* its map live and the learned map kept poisoning
itself (phantom gaps, invented portals, purge tooling to undo the damage).
This model is the opposite: it is frozen from the ROM extraction in
``data/maps/`` (walkability grids, map connections, warp tables produced by
``data/maps/build_walkability.py``) and is never mutated at runtime.

Two levels:
  * tile level  - BFS over one map's walkability grid. Warp tiles are
                  treated as destinations, never as corridors.
  * map level   - BFS over connections + warps between the 246 maps, with
                  capability gates (Surf for sea routes and the Power Plant,
                  Fly as virtual edges between towns).

Walkability grids mark water tiles as traversable (they are, with Surf);
capability gating happens at the map level, which is where HM checks belong.

One data caveat, learned by testing (not by trusting the generator): the
JSON grids store one quadrant sample per tile (see build_walkability.py),
which leaves them checkerboarded and disconnected as-is - Pallet Town
fragments into 25 components and its own lab is unreachable. The game's
collision unit is the 2x2-tile BLOCK, so at load time each block is
collapsed: a block is walkable iff any of its sampled quadrants is
passable, and warp tiles are forced walkable (the generator intended
this but never applied it). That yields the connected town the game
actually has. It remains an approximation - on real hardware the live
RAM collision window is the final arbiter, and the executor's stuck
detector exists precisely for the places where the static model lies.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

Tile = Tuple[int, int]

# Maps that are open water: entering them on foot is impossible.
WATER_MAP_NAMES = {"Route19", "Route20", "Route21"}
# Warp destinations that sit across water even though the source map is land.
SURF_WARP_DESTINATIONS = {"PowerPlant"}
# Places Fly can take you (towns/cities with a Pokemon Center, plus Indigo).
FLY_DESTINATIONS = {
    "PalletTown", "ViridianCity", "PewterCity", "CeruleanCity",
    "LavenderTown", "VermilionCity", "CeladonCity", "FuchsiaCity",
    "CinnabarIsland", "SaffronCity", "IndigoPlateau",
}

DIRS = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}


@dataclass
class Warp:
    x: int
    y: int
    dest_id: Optional[int]  # None = "LAST_MAP" exit warp (dest is dynamic)
    idx: int  # which warp in the destination map you land on (1-based)

    @property
    def is_exit(self) -> bool:
        return self.dest_id is None


@dataclass
class GameMap:
    map_id: int
    name: str
    w: int
    h: int
    walk: List[List[int]]
    ledges: Set[Tile] = field(default_factory=set)
    warps: List[Warp] = field(default_factory=list)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def walkable(self, x: int, y: int, dx: int = 0, dy: int = 0) -> bool:
        """Is tile (x, y) enterable when moving in direction (dx, dy)?

        Ledges are recorded in `ledges` but deliberately NOT enforced
        here: the extracted ledge set is broad (it includes fence/face
        tiles, e.g. 323 tiles in Viridian City) and enforcing it seals
        city entrances that are open in the real game. One-way ledge
        behaviour belongs to the live-collision layer on hardware; if the
        static model routes over a ledge the wrong way here, the
        executor's stuck detector catches it honestly.
        """
        if not self.in_bounds(x, y):
            return False
        return self.walk[y][x] == 1


@dataclass
class Leg:
    """One map-to-map step of a route."""
    src: int
    dst: int
    via: str                      # "connection north", "warp", "fly"
    requires: Tuple[str, ...] = ()  # capabilities this leg needs


@dataclass
class Route:
    legs: List[Leg]

    @property
    def maps(self) -> List[int]:
        if not self.legs:
            return []
        return [self.legs[0].src] + [leg.dst for leg in self.legs]

    def __len__(self) -> int:
        return len(self.legs)


class World:
    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        walk_raw = json.loads((data_dir / "pokered_walkability.json").read_text())
        conn_raw = json.loads((data_dir / "pokered_connections.json").read_text())

        self.maps: Dict[int, GameMap] = {}
        for mid_s, m in walk_raw.items():
            mid = int(mid_s)
            warps = [
                Warp(wp["x"], wp["y"],
                     int(wp["dest_id"]) if wp["dest_id"] not in (None, -1)
                     else None,
                     int(wp["idx"]))
                for wp in m.get("warps", [])
            ]
            self.maps[mid] = GameMap(
                map_id=mid,
                name=m["name"],
                w=m["w"],
                h=m["h"],
                walk=self._collapse_blocks(m["walk"], m["w"], m["h"], warps),
                ledges={(l["x"], l["y"]) for l in m.get("ledges", [])},
                warps=warps,
            )
        # Exit warps (dest = LAST_MAP) lead back to wherever warps into this
        # map. Precompute entrance maps for graph edges.
        self._entrances: Dict[int, Set[int]] = {}
        for mid, gm in self.maps.items():
            for wp in gm.warps:
                if wp.dest_id is not None:
                    self._entrances.setdefault(wp.dest_id, set()).add(mid)
        self._name_to_id = {m.name: mid for mid, m in self.maps.items()}
        self.connections: Dict[int, List[Tuple[str, int]]] = {
            int(src): [(d, int(dst)) for d, dst in conns]
            for src, conns in conn_raw.items()
        }

    @staticmethod
    def _collapse_blocks(grid: List[List[int]], w: int, h: int,
                         warps: List[Warp]) -> List[List[int]]:
        """Collapse the quadrant-sampled grid to block resolution (see the
        module docstring) and force warp tiles walkable."""
        out = [[0] * w for _ in range(h)]
        for by in range(0, h - 1, 2):
            for bx in range(0, w - 1, 2):
                v = 1 if any(grid[by + dy][bx + dx] == 1
                             for dy in (0, 1) for dx in (0, 1)) else 0
                for dy in (0, 1):
                    for dx in (0, 1):
                        out[by + dy][bx + dx] = v
        for wp in warps:
            if 0 <= wp.x < w and 0 <= wp.y < h:
                out[wp.y][wp.x] = 1
        return out

    # -- lookups -----------------------------------------------------------
    def map_id(self, name: str) -> int:
        return self._name_to_id[name]

    def map_name(self, map_id: int) -> str:
        return self.maps[map_id].name

    def warp_landing(self, warp: Warp) -> Optional[Tile]:
        """Tile you arrive on when taking `warp`."""
        dest = self.maps.get(warp.dest_id)
        if dest is None or not dest.warps:
            return None
        idx = min(max(warp.idx, 1), len(dest.warps))
        w = dest.warps[idx - 1]
        return (w.x, w.y)

    # -- tile-level pathfinding ---------------------------------------------
    def tile_path(self, map_id: int, start: Tile, goal: Tile) -> Optional[List[Tile]]:
        """BFS shortest path within one map. Includes both endpoints;
        None when unreachable. Warp tiles are destinations, not corridors:
        a path never steps *through* a door (it would teleport), though it
        may end on one."""
        gm = self.maps[map_id]
        if start == goal:
            return [start]
        warp_tiles = {(wp.x, wp.y) for wp in gm.warps}
        prev: Dict[Tile, Optional[Tile]] = {start: None}
        q = deque([start])
        while q:
            cx, cy = q.popleft()
            for dx, dy in DIRS.values():
                n = (cx + dx, cy + dy)
                if n in prev:
                    continue
                if not gm.walkable(n[0], n[1], dx, dy):
                    continue
                if n in warp_tiles and n != goal:
                    continue
                prev[n] = (cx, cy)
                if n == goal:
                    path = [n]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                q.append(n)
        return None

    # -- map-level routing --------------------------------------------------
    def _edges(self, map_id: int, capabilities: FrozenSet[str]) -> List[Leg]:
        gm = self.maps[map_id]
        edges: List[Leg] = []
        for direction, dst in self.connections.get(map_id, []):
            dst_name = self.maps[dst].name if dst in self.maps else ""
            req: Tuple[str, ...] = ()
            if dst_name in WATER_MAP_NAMES and "surf" not in capabilities:
                req = ("surf",)
            edges.append(Leg(map_id, dst, f"connection {direction}", req))
        seen_warp_dests = set()
        for wp in gm.warps:
            if wp.is_exit:
                for entrance in self._entrances.get(map_id, set()):
                    if entrance not in seen_warp_dests:
                        seen_warp_dests.add(entrance)
                        edges.append(Leg(map_id, entrance, "warp (exit)", ()))
                continue
            if wp.dest_id in seen_warp_dests or wp.dest_id not in self.maps:
                continue
            seen_warp_dests.add(wp.dest_id)
            req = ()
            if self.maps[wp.dest_id].name in SURF_WARP_DESTINATIONS \
                    and "surf" not in capabilities:
                req = ("surf",)
            edges.append(Leg(map_id, wp.dest_id, "warp", req))
        if "fly" in capabilities and gm.name in FLY_DESTINATIONS:
            for name in FLY_DESTINATIONS:
                dst = self._name_to_id.get(name)
                if dst is not None and dst != map_id:
                    edges.append(Leg(map_id, dst, "fly", ()))
        return edges

    def route(self, src: int, dst: int,
              capabilities: FrozenSet[str] = frozenset()) -> Optional[Route]:
        """Shortest map-level route honouring capability gates. Legs that
        need a missing capability are simply not traversable."""
        if src == dst:
            return Route([])
        prev: Dict[int, Optional[Leg]] = {src: None}
        q = deque([src])
        while q:
            cur = q.popleft()
            for leg in self._edges(cur, capabilities):
                if leg.requires:
                    continue
                if leg.dst in prev:
                    continue
                prev[leg.dst] = leg
                if leg.dst == dst:
                    legs = []
                    node = dst
                    while prev[node] is not None:
                        legs.append(prev[node])
                        node = prev[node].src
                    return Route(list(reversed(legs)))
                q.append(leg.dst)
        return None

    def missing_capabilities(self, src: int, dst: int,
                             capabilities: FrozenSet[str] = frozenset()
                             ) -> Set[str]:
        """If `dst` is unreachable, which capabilities would unlock a route?
        (Empty set means unreachable for some other reason.)"""
        all_caps = capabilities | {"surf", "fly", "cut", "strength"}
        if self.route(src, dst, all_caps) is None:
            return set()
        missing = set()
        for cap in ("surf", "fly", "cut", "strength"):
            if cap not in capabilities and \
                    self.route(src, dst, capabilities | {cap}) is not None:
                missing.add(cap)
        return missing or {"surf", "fly", "cut", "strength"} - set(capabilities)


def default_world() -> World:
    """Load the world from this repo's data/maps directory."""
    here = Path(__file__).resolve().parent.parent
    return World(here / "data" / "maps")
