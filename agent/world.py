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

Data provenance, learned the hard way (2026-11-03): the grids were
originally generated with the passability test INVERTED - the engine's
coll_tiles lists are PASSABLE tiles (home/overworld.asm
CheckTilePassable), not blockers. The inverted grids looked plausible
in the simulator (which shared the same wrong data) and only failed
against the real game, where Red's bedroom floor read as solid wall.
The generator (data/maps/build_walkability.py) has been fixed and the
grids regenerated from the pokered source: passable iff the cell's
sampled tile id is in its tileset's list, warp cells forced passable,
and a coherence fallback (open grid) for the two maps whose warps are
genuinely unreachable on foot (teleporter floors). What remains
approximate: tile-PAIR collisions (the engine's second system) are not
modelled beyond ledges, so on hardware the stuck detector is still the
safety net.
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
    warps: List[Warp] = field(default_factory=list)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def walkable(self, x: int, y: int, dx: int = 0, dy: int = 0) -> bool:
        """Is tile (x, y) enterable? Ledges need no rule here: their face
        tiles are absent from the passable lists, so the grid already
        treats them as walls and BFS routes around via the side paths -
        a jump is only a shortcut. (Two earlier attempts at ledge
        modelling were built on misread data and sealed whole towns;
        see build_walkability.py.)"""
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
                walk=self._force_warps(m["walk"], m["w"], m["h"], warps),
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
        # Connections carry the header offset (in blocks): crossing
        # src's edge at column/row t enters dst at t - 2*offset
        # (see build_walkability.py).
        self.connections: Dict[int, List[Tuple[str, int, int]]] = {
            int(src): [(c["dir"], int(c["dst"]), int(c["offset"]))
                       for c in conns]
            for src, conns in conn_raw.items()
        }

    @staticmethod
    def _force_warps(grid: List[List[int]], w: int, h: int,
                     warps: List[Warp]) -> List[List[int]]:
        """Warp cells are step-on triggers; force them passable so a data
        quirk can never seal a door. (The regenerated data already does
        this; kept as belt-and-braces for older files.)"""
        out = [row[:] for row in grid]
        for wp in warps:
            if 0 <= wp.x < w and 0 <= wp.y < h:
                out[wp.y][wp.x] = 1
        return out

    def is_outside(self, map_id: int) -> bool:
        """Outside maps (towns/routes) vs indoor maps (houses, gates,
        dungeons). The engine's wLastMap - the destination of every
        LAST_MAP exit warp - is the last OUTSIDE map visited; leaving an
        outside map sets it, indoor maps never do (home/overworld.asm
        WarpFound). Connection-bearing maps are exactly the outside set.
        """
        return len(self.connections.get(map_id, [])) > 0

    def default_last_outside(self, map_id: int) -> Optional[int]:
        """Best static guess for wLastMap when starting indoors: walk the
        entrance graph outward through the indoor cluster (house floors,
        gatehouses) and return the outside map it hangs off - if exactly
        one exists. (During a journey the tracked value takes over; this
        is only the cold-start guess, e.g. Red's house -> Pallet Town.)"""
        if self.is_outside(map_id):
            return map_id
        seen = {map_id}
        frontier = [map_id]
        outside_found: Set[int] = set()
        while frontier:
            cur = frontier.pop()
            for ent in self._entrances.get(cur, set()):
                if ent in seen:
                    continue
                seen.add(ent)
                if self.is_outside(ent):
                    outside_found.add(ent)
                else:
                    frontier.append(ent)
        return outside_found.pop() if len(outside_found) == 1 else None

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
    def tile_path(self, map_id: int, start: Tile, goal: Tile,
                  blocked: FrozenSet[Tile] = frozenset()) -> Optional[List[Tile]]:
        """BFS shortest path within one map. Includes both endpoints;
        None when unreachable. Warp tiles are destinations, not corridors:
        a path never steps *through* a door (it would teleport), though it
        may end on one. `blocked` tiles are treated as walls for this call
        only (an NPC standing in the way is a transient obstacle, not a
        fact about the map); the goal itself is never blocked."""
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
                if n in blocked and n != goal:
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

    def tile_paths_from(self, map_id: int, start: Tile,
                        targets: Set[Tile]) -> Dict[Tile, List[Tile]]:
        """One BFS from `start`; returns a path for every reachable target.
        Warp tiles are blocked unless they are targets (doors are
        destinations, not corridors)."""
        gm = self.maps[map_id]
        warp_tiles = {(wp.x, wp.y) for wp in gm.warps}
        remaining = set(targets) - {start}
        found: Dict[Tile, List[Tile]] = {}
        if start in targets:
            found[start] = [start]
        prev: Dict[Tile, Optional[Tile]] = {start: None}
        q = deque([start])
        while q and remaining:
            cx, cy = q.popleft()
            for dx, dy in DIRS.values():
                n = (cx + dx, cy + dy)
                if n in prev:
                    continue
                if not gm.walkable(n[0], n[1], dx, dy):
                    continue
                if n in warp_tiles and n not in targets:
                    continue
                prev[n] = (cx, cy)
                if n in remaining:
                    remaining.discard(n)
                    path = [n]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    found[n] = list(reversed(path))
                q.append(n)
        return found

    # -- journey planning (portal level) --------------------------------------
    def journey(self, src_map: int, src_xy: Tile, dst_map: int,
                dst_xy: Optional[Tile] = None,
                capabilities: FrozenSet[str] = frozenset()) -> Optional[list]:
        """Plan a full journey as executable steps, chaining within-map
        walks with warps and connection crossings. Map-level route() can
        promise a connection the map's interior geometry cannot honour
        (Route 2's north edge is unreachable without passing through the
        forest gates); a journey only exists if every walk in it does.

        Steps: ("walk", map, tile) - walk within a map (the tile may be a
        warp, which fires on arrival); ("cross", direction, dst) - one
        more step across a connection edge; ("fly", dst).
        """
        from collections import deque as _dq
        # Journey state includes wLastMap: LAST_MAP exit warps return to
        # the last OUTSIDE map, so where an exit leads depends on where
        # the journey has been (the forest gates are the famous case).
        start_lo = src_map if self.is_outside(src_map) \
            else self.default_last_outside(src_map)
        start_state = (src_map, src_xy[0], src_xy[1], start_lo)
        # state -> (prev_state, [steps from prev])
        parents: Dict[Tuple[int, int, int, object], Optional[tuple]] = \
            {start_state: None}
        q = _dq([start_state])

        def reconstruct(state):
            steps = []
            while parents[state] is not None:
                prev_state, seg = parents[state]
                steps = seg + steps
                state = prev_state
            return steps

        while q:
            state = q.popleft()
            mid, mx, my, lo = state
            gm = self.maps[mid]
            if mid == dst_map and dst_xy is None:
                return reconstruct(state)
            targets: Dict[Tile, tuple] = {}
            for wp in gm.warps:
                targets[(wp.x, wp.y)] = ("warp", wp)
            for direction, dst, off in self.connections.get(mid, []):
                if self.maps[dst].name in WATER_MAP_NAMES \
                        and "surf" not in capabilities:
                    continue
                if dy_dx := DIRS[direction]:
                    dx, dy = dy_dx
                if dy != 0:
                    ey = 0 if dy < 0 else gm.h - 1
                    exits = [(x, ey) for x in range(gm.w)
                             if gm.walkable(x, ey, dx, dy)]
                else:
                    ex = 0 if dx < 0 else gm.w - 1
                    exits = [(ex, y) for y in range(gm.h)
                             if gm.walkable(ex, y, dx, dy)]
                for ex_tile in exits:
                    targets[ex_tile] = ("cross", direction, dst, off)
            if mid == dst_map and dst_xy is not None:
                targets[dst_xy] = ("goal",)
            if "fly" in capabilities and gm.name in FLY_DESTINATIONS \
                    and self.maps[dst_map].name in FLY_DESTINATIONS \
                    and mid != dst_map:
                fly_state = (dst_map, -1, -1, dst_map)
                parents[fly_state] = (state, [("fly", dst_map)])
                return reconstruct(fly_state)
            paths = self.tile_paths_from(mid, (mx, my), set(targets))
            for tile, action in targets.items():
                if tile not in paths:
                    continue
                walk_step = ("walk", mid, tile)
                if action[0] == "goal":
                    goal_state = (mid, tile[0], tile[1], lo)
                    parents[goal_state] = (state, [walk_step])
                    return reconstruct(goal_state)
                if action[0] == "warp":
                    wp = action[1]
                    arrivals = []
                    if wp.is_exit:
                        if lo is not None and 1 <= wp.idx <= len(self.maps[lo].warps):
                            lw = self.maps[lo].warps[wp.idx - 1]
                            arrivals = [(lo, lw.x, lw.y, lo)]
                    else:
                        if self.maps[wp.dest_id].name in SURF_WARP_DESTINATIONS \
                                and "surf" not in capabilities:
                            continue
                        landing = self.warp_landing(wp)
                        if landing:
                            new_lo = wp.dest_id if self.is_outside(wp.dest_id) else lo
                            arrivals = [(wp.dest_id, landing[0], landing[1], new_lo)]
                    for arr in arrivals:
                        if arr not in parents:
                            parents[arr] = (state, [walk_step])
                            q.append(arr)
                elif action[0] == "cross":
                    _, direction, dst, off = action
                    dx, dy = DIRS[direction]
                    dgm = self.maps[dst]
                    if dy != 0:
                        lx = tile[0] - 2 * off
                        ly = dgm.h - 1 if dy < 0 else 0
                    else:
                        lx = dgm.w - 1 if dx < 0 else 0
                        ly = tile[1] - 2 * off
                    if 0 <= lx < dgm.w and 0 <= ly < dgm.h \
                            and dgm.walkable(lx, ly, dx, dy):
                        arr = (dst, lx, ly, dst)
                        if arr not in parents:
                            parents[arr] = (state, [walk_step,
                                                    ("cross", direction, dst)])
                            q.append(arr)
        return None

    # -- map-level routing --------------------------------------------------
    def _edges(self, map_id: int, capabilities: FrozenSet[str]) -> List[Leg]:
        gm = self.maps[map_id]
        edges: List[Leg] = []
        for direction, dst, _offset in self.connections.get(map_id, []):
            dst_name = self.maps[dst].name if dst in self.maps else ""
            req: Tuple[str, ...] = ()
            if dst_name in WATER_MAP_NAMES and "surf" not in capabilities:
                req = ("surf",)
            edges.append(Leg(map_id, dst, f"connection {direction}", req))
        seen_warp_dests = set()
        for wp in gm.warps:
            if wp.is_exit:
                # LAST_MAP warps return to the last OUTSIDE map visited.
                for entrance in self._entrances.get(map_id, set()):
                    if not self.is_outside(entrance):
                        continue
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
