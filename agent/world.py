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

# Map scripts that assign wLastMap on entry (pokered scripts/*.asm:
# each map's _Script does `ld a, <MAP>` / `ld [wLastMap], a`). The
# generic engine rule (home/overworld.asm WarpFound) only sets
# wLastMap when warping FROM an outside map, so without these scripts
# a two-mouthed indoor complex (a cave or tunnel with an entrance
# lobby on each side) could never exit on its far side: the exit warp
# would return to whichever outside map you entered from. Each
# lobby's script re-points wLastMap at its OWN outside map the moment
# you enter the lobby - from either direction. That is the whole
# mechanism behind Diglett's Cave and the Underground Path working.
# Name -> name; resolved to ids at load, missing names skipped.
LAST_MAP_SCRIPT_MAPS = {
    "DiglettsCaveRoute2": "Route2",
    "DiglettsCaveRoute11": "Route11",
    "UndergroundPathRoute5": "Route5",
    "UndergroundPathRoute6": "Route6",
    "UndergroundPathRoute7": "Route7",
    "UndergroundPathRoute7Copy": "Route7",
    "UndergroundPathRoute8": "Route8",
}
# Route22Gate's script assigns wLastMap by position instead
# (scripts/Route22Gate.asm: player y < 4 -> ROUTE_23, else ROUTE_22),
# so it is handled tile-wise in World._script_last_map.


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
    # Ledge face tile -> jump direction. A face is not standable
    # (walk == 0); stepping into it from behind, in the jump
    # direction, hops the player onto the tile beyond it. Modelled in
    # the BFS step generator (_bfs_steps), not here.
    ledges: Dict[Tile, Tuple[int, int]] = field(default_factory=dict)
    tileset: str = ""
    # Cut-tree FACE tiles (build_walkability.py: the cell of the
    # tree's 2x2 block whose sampled tile id is the tileset's cut id -
    # $3D OVERWORLD / $50 GYM, engine/overworld/cut.asm UsedCut).
    # Every other cell of the block reads as generic tree graphic
    # (walk == 0); the engine replaces the WHOLE block when the tree
    # is cut (CutTreeBlockSwaps), so passability is a block property:
    # `cut_blocks` maps every cell of a tree block to its face tile.
    cut_trees: FrozenSet[Tile] = frozenset()
    cut_blocks: Dict[Tile, Tile] = field(default_factory=dict)
    # Tile-pair collisions (TilePairCollisionsLand, CAVERN and
    # FOREST tilesets only): the engine refuses a step whose
    # standing/front sampled tile ids form a forbidden pair, in
    # either direction (an elevation rule - CAVERN $20 floor may
    # not cross to $05 raised floor). `tiles` is the per-cell
    # sampled (bottom-left) tile id grid the passability rule and
    # the pair rule both read; None on maps whose tileset has no
    # pair table. Modelled in _bfs_steps like ledges and Cut.
    tiles: Optional[List[List[int]]] = None
    pair_forbidden: FrozenSet[FrozenSet[int]] = frozenset()

    def cut_block_cells(self, face: Tile) -> FrozenSet[Tile]:
        """All cells of the tree block whose face is `face`."""
        return frozenset(t for t, f in self.cut_blocks.items() if f == face)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def walkable(self, x: int, y: int, dx: int = 0, dy: int = 0) -> bool:
        """Is tile (x, y) enterable as a standing tile? Ledge faces are
        absent from the passable lists, so they read as walls here;
        jumping one is a movement rule, not a property of the face
        tile, and lives in _bfs_steps (see build_walkability.py for
        why the face ids, never the standing-tile ids, are the data)."""
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
            ledges = {}
            for key, delta in m.get("ledges", {}).items():
                lx, ly = key.split(",")
                ledges[(int(lx), int(ly))] = (int(delta[0]), int(delta[1]))
            cut_trees = frozenset(
                (int(kx), int(ky))
                for kx, ky in (k.split(",") for k in m.get("cut_trees", [])))
            cut_blocks: Dict[Tile, Tile] = {}
            for fx, fy in cut_trees:
                # A tree is one 2x2 block; its face is one of the four
                # cells. Expand to the block so the BFS can leave the
                # block by any side once the tree is cut.
                ox, oy = fx - fx % 2, fy - fy % 2
                for bx in (ox, ox + 1):
                    for by in (oy, oy + 1):
                        if 0 <= bx < m["w"] and 0 <= by < m["h"]:
                            cut_blocks[(bx, by)] = (fx, fy)
            self.maps[mid] = GameMap(
                map_id=mid,
                name=m["name"],
                w=m["w"],
                h=m["h"],
                walk=self._force_warps(m["walk"], m["w"], m["h"], warps),
                warps=warps,
                ledges=ledges,
                tileset=m.get("tileset", ""),
                cut_trees=cut_trees,
                cut_blocks=cut_blocks,
                tiles=m.get("tiles"),
                pair_forbidden=frozenset(
                    frozenset(p) for p in m.get("pair_forbidden", [])),
            )
        # Exit warps (dest = LAST_MAP) lead back to wherever warps into this
        # map. Precompute entrance maps for graph edges.
        self._entrances: Dict[int, Set[int]] = {}
        for mid, gm in self.maps.items():
            for wp in gm.warps:
                if wp.dest_id is not None:
                    self._entrances.setdefault(wp.dest_id, set()).add(mid)
        # wLastMap assignments made by map scripts (see
        # LAST_MAP_SCRIPT_MAPS): map id -> the outside map id its
        # entry script stores in wLastMap.
        self._last_map_script: Dict[int, int] = {}
        # Name lookup must resolve to the canonical map, not a _COPY.
        # pokered ships duplicate maps that reuse the original's header
        # (CERULEAN_TRASHED_HOUSE_COPY $45, UNDERGROUND_PATH_ROUTE_6_COPY
        # $4B, an unused Saffron duplicate at $0B), so several ids share
        # one name in the extracted data. The canonical map is always
        # the lowest id (the copy constants come later in
        # map_constants.asm order); a last-wins dict resolved
        # "CeruleanTrashedHouse" to the $45 copy, whose id never matches
        # the game's current-map byte - executor walks inside the real
        # house then silently no-oped, reading the mismatch as a warp
        # exit (found live, 2026-11-04).
        self._name_to_id: Dict[str, int] = {}
        for mid in sorted(self.maps):
            self._name_to_id.setdefault(self.maps[mid].name, mid)
        for mname, lname in LAST_MAP_SCRIPT_MAPS.items():
            if mname in self._name_to_id and lname in self._name_to_id:
                self._last_map_script[self._name_to_id[mname]] = \
                    self._name_to_id[lname]
        self._route22_gate_id = self._name_to_id.get("Route22Gate")
        self._route23_id = self._name_to_id.get("Route23")
        self._route22_id = self._name_to_id.get("Route22")
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

    def _script_last_map(self, map_id: int, tile: Tile) -> Optional[int]:
        """The wLastMap value map_id's entry script assigns, if any.
        Route22Gate's script is position-dependent (y < 4 -> Route 23,
        else Route 22; scripts/Route22Gate.asm), the rest are fixed."""
        if map_id == self._route22_gate_id:
            return self._route23_id if tile[1] < 4 else self._route22_id
        return self._last_map_script.get(map_id)

    def _enter_last_map(self, map_id: int, tile: Tile,
                        carried: Optional[int]) -> Optional[int]:
        """wLastMap after arriving in map_id at `tile`, having carried
        `carried` in: a map-script assignment wins, an outside map
        replaces it with itself (WarpFound's from-outside rule), and
        anywhere else it carries over unchanged."""
        scripted = self._script_last_map(map_id, tile)
        if scripted is not None:
            return scripted
        if self.is_outside(map_id):
            return map_id
        return carried

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
    def _bfs_steps(self, gm: GameMap, cur: Tile, blocked,
                   warp_tiles, exempt, capabilities=frozenset(),
                   cut_done=frozenset()):
        """Tiles reachable from `cur` in one step. `exempt` tiles (the
        goal / the target set) override the blocked and warp-corridor
        rules. Ledge jumps appear as two unit steps through a virtual
        node: entering a face tile is allowed only in its jump
        direction and only when the landing beyond is standable, and
        the face node's only exit is that landing - so paths stay
        unit-step and the follower can treat the face entry as "press
        toward it and land two tiles on" (see Executor._follow_tiles).

        Cut trees (gm.cut_blocks) follow the engine's rules: the
        block's non-walkable cells are solid while the tree stands
        (a block cell whose own tile is passable ground - some tree
        blocks have one - stays ordinary ground throughout; the
        walkable branch above has already yielded those). With the
        "cut" capability a block may be ENTERED only through its
        face tile (the one cell that reports the cut tile id - that
        entry is an action, not a walk: journey() splits it out as a
        ("cut", ...) step and the Executor performs the field move);
        from the face, the block's other cells are passable (cutting
        replaces the whole block with ground), and a block whose face
        is in `cut_done` (cut earlier this map visit) is plain ground.
        Without the capability, tree cells are walls, full stop.

        Tile-pair collisions (gm.pair_forbidden, CAVERN/FOREST
        only) refuse a plain step whose two cells' sampled tile
        ids form a forbidden pair, either direction - the engine's
        CheckForTilePairCollisions compares wTilePlayerStandingOn
        with wTileInFrontOfPlayer, both bottom-left samples. The
        rule is applied to the walkable branch only: ledge jumps
        skip the pair check in the engine too (HandleLedges runs
        first and sets BIT_LEDGE_OR_FISHING), and Cut tilesets
        (OVERWORLD/GYM) have no pair entries at all.
        """
        cx, cy = cur
        if cur in gm.ledges:
            dx, dy = gm.ledges[cur]
            b = (cx + dx, cy + dy)
            if gm.in_bounds(*b) and gm.walkable(b[0], b[1], dx, dy) \
                    and b not in blocked \
                    and (b not in warp_tiles or b in exempt):
                yield b
            return
        for dx, dy in DIRS.values():
            n = (cx + dx, cy + dy)
            if n in blocked and n not in exempt:
                continue
            if gm.walkable(n[0], n[1], dx, dy):
                if n in warp_tiles and n not in exempt:
                    continue
                if gm.pair_forbidden:
                    t_cur = gm.tiles[cy][cx]
                    t_n = gm.tiles[n[1]][n[0]]
                    if t_cur != t_n and \
                            frozenset((t_cur, t_n)) in gm.pair_forbidden:
                        continue
                yield n
                continue
            face = gm.cut_blocks.get(n)
            if face is not None:
                if face in cut_done:
                    yield n                      # cut earlier: ground
                elif "cut" in capabilities and \
                        (n == face or cur == face):
                    yield n                      # enter via the face,
                    # or step from the face into the rest of its block
                continue
            if gm.ledges.get(n) == (dx, dy):
                b = (n[0] + dx, n[1] + dy)
                if gm.in_bounds(*b) and gm.walkable(b[0], b[1], dx, dy) \
                        and b not in blocked \
                        and (b not in warp_tiles or b in exempt):
                    yield n

    def tile_path(self, map_id: int, start: Tile, goal: Tile,
                  blocked: FrozenSet[Tile] = frozenset(),
                  capabilities: FrozenSet[str] = frozenset(),
                  cut_done: FrozenSet[Tile] = frozenset()
                  ) -> Optional[List[Tile]]:
        """BFS shortest path within one map. Includes both endpoints;
        None when unreachable. Warp tiles are destinations, not corridors:
        a path never steps *through* a door (it would teleport), though it
        may end on one. `blocked` tiles are treated as walls for this call
        only (an NPC standing in the way is a transient obstacle, not a
        fact about the map); the goal itself is never blocked.
        `capabilities` / `cut_done` gate Cut-tree blocks (see _bfs_steps)."""
        gm = self.maps[map_id]
        if start == goal:
            return [start]
        warp_tiles = {(wp.x, wp.y) for wp in gm.warps}
        prev: Dict[Tile, Optional[Tile]] = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            for n in self._bfs_steps(gm, cur, blocked, warp_tiles, {goal},
                                     capabilities, cut_done):
                if n in prev:
                    continue
                prev[n] = cur
                if n == goal:
                    path = [n]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                q.append(n)
        return None

    def tile_paths_from(self, map_id: int, start: Tile,
                        targets: Set[Tile],
                        blocked: FrozenSet[Tile] = frozenset(),
                        capabilities: FrozenSet[str] = frozenset(),
                        cut_done: FrozenSet[Tile] = frozenset()
                        ) -> Dict[Tile, List[Tile]]:
        """One BFS from `start`; returns a path for every reachable target.
        Warp tiles are blocked unless they are targets (doors are
        destinations, not corridors). `blocked` tiles are walls for
        this call (learned walls on guessed grids - see the Executor);
        targets are exempt. `capabilities` / `cut_done` gate Cut-tree
        blocks (see _bfs_steps)."""
        gm = self.maps[map_id]
        warp_tiles = {(wp.x, wp.y) for wp in gm.warps}
        remaining = set(targets) - {start}
        found: Dict[Tile, List[Tile]] = {}
        if start in targets:
            found[start] = [start]
        prev: Dict[Tile, Optional[Tile]] = {start: None}
        q = deque([start])
        while q and remaining:
            cur = q.popleft()
            for n in self._bfs_steps(gm, cur, blocked, warp_tiles,
                                     targets, capabilities, cut_done):
                if n in prev:
                    continue
                prev[n] = cur
                if n in remaining:
                    remaining.discard(n)
                    path = [n]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    found[n] = list(reversed(path))
                q.append(n)
        return found

    def _walk_steps(self, mid: int, path: List[Tile],
                    cut_done: FrozenSet[Tile]) -> list:
        """Expand one within-map tile path into journey steps, splitting
        out a ("cut", map, face) step wherever the path enters a
        Cut-tree block through its face tile: walk to the stance tile
        beside the tree, cut, then continue. The Executor performs the
        cut (party-menu field move) and steps onto the face tile, so
        the following walk resumes from inside the (now ground) block.
        Faces in `cut_done` were cut earlier this visit and need no
        action. A path may cross several trees; each gets its own cut
        step, in path order."""
        gm = self.maps[mid]
        steps: list = []
        for i in range(1, len(path)):
            t = path[i]
            if t in gm.cut_trees and t not in cut_done:
                steps.append(("walk", mid, path[i - 1]))
                steps.append(("cut", mid, t))
        steps.append(("walk", mid, path[-1]))
        return steps

    # -- journey planning (portal level) --------------------------------------
    def journey(self, src_map: int, src_xy: Tile, dst_map: int,
                dst_xy: Optional[Tile] = None,
                capabilities: FrozenSet[str] = frozenset(),
                blocked=None, cut_done=None) -> Optional[list]:
        """Plan a full journey as executable steps, chaining within-map
        walks with warps and connection crossings. Map-level route() can
        promise a connection the map's interior geometry cannot honour
        (Route 2's north edge is unreachable without passing through the
        forest gates); a journey only exists if every walk in it does.

        Steps: ("walk", map, tile) - walk within a map (the tile may be a
        warp, which fires on arrival); ("cut", map, tile) - use Cut on
        the tree whose face is `tile` (requires the "cut" capability;
        see _walk_steps); ("cross", direction, dst) - one more step
        across a connection edge; ("fly", dst).

        `cut_done`: {map_id: {face tiles already cut this visit}} -
        trees respawn when a map is reloaded, so this is session state
        owned by the Executor, never a fact about the world.
        """
        from collections import deque as _dq
        # Journey state includes wLastMap: LAST_MAP exit warps return to
        # the last OUTSIDE map, so where an exit leads depends on where
        # the journey has been (the forest gates are the famous case).
        # Map scripts can assign it on entry (the two-mouthed cave and
        # tunnel lobbies) - _enter_last_map applies those assignments
        # at every arrival, and to the start state here.
        start_lo = self._script_last_map(src_map, src_xy)
        if start_lo is None:
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
            paths = self.tile_paths_from(
                mid, (mx, my), set(targets),
                blocked=(blocked or {}).get(mid, frozenset()),
                capabilities=capabilities,
                cut_done=(cut_done or {}).get(mid, frozenset()))
            for tile, action in targets.items():
                if tile not in paths:
                    continue
                walk_steps = self._walk_steps(
                    mid, paths[tile],
                    (cut_done or {}).get(mid, frozenset()))
                if action[0] == "goal":
                    goal_state = (mid, tile[0], tile[1], lo)
                    parents[goal_state] = (state, walk_steps)
                    return reconstruct(goal_state)
                if action[0] == "warp":
                    wp = action[1]
                    arrivals = []
                    if wp.is_exit:
                        # The exit's destination is wLastMap *at the
                        # moment of stepping on it*: this map's own
                        # script may have assigned it on entry (the
                        # cave/tunnel lobbies), which is what lets a
                        # through-journey leave by the far mouth.
                        eff_lo = self._script_last_map(mid, tile)
                        if eff_lo is None:
                            eff_lo = lo
                        if eff_lo is not None and \
                                1 <= wp.idx <= len(self.maps[eff_lo].warps):
                            lw = self.maps[eff_lo].warps[wp.idx - 1]
                            arrivals = [(eff_lo, lw.x, lw.y, eff_lo)]
                    else:
                        if wp.dest_id not in self.maps:
                            # A warp into a map the data does not model
                            # (the Underground Path interior) is a dead
                            # end for routing, not a crash.
                            continue
                        if self.maps[wp.dest_id].name in SURF_WARP_DESTINATIONS \
                                and "surf" not in capabilities:
                            continue
                        landing = self.warp_landing(wp)
                        if landing:
                            new_lo = self._enter_last_map(
                                wp.dest_id, landing, lo)
                            arrivals = [(wp.dest_id, landing[0], landing[1], new_lo)]
                    for arr in arrivals:
                        if arr not in parents:
                            parents[arr] = (state, walk_steps)
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
                        arr = (dst, lx, ly,
                               self._enter_last_map(dst, (lx, ly), lo))
                        if arr not in parents:
                            parents[arr] = (state, walk_steps +
                                            [("cross", direction, dst)])
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

    def journey_missing_capabilities(
            self, src_map: int, src_xy: Tile, dst_map: int,
            dst_xy: Optional[Tile] = None,
            capabilities: FrozenSet[str] = frozenset()) -> Set[str]:
        """Tile-level counterpart of missing_capabilities: if no
        journey exists from (src_map, src_xy), which single capability
        would create one? (Empty set: a journey exists, or no single
        capability unlocks one.) This is where a Cut diagnosis comes
        from: a map-level route() can exist while every tile-level
        journey is sealed by a tree row (Route 2's cave-mouth pocket),
        so the map-level check alone can never see it."""
        if self.journey(src_map, src_xy, dst_map, dst_xy,
                        capabilities) is not None:
            return set()
        missing = set()
        for cap in ("cut", "surf", "fly", "strength"):
            if cap not in capabilities and \
                    self.journey(src_map, src_xy, dst_map, dst_xy,
                                 capabilities | {cap}) is not None:
                missing.add(cap)
        return missing

    def missing_capabilities(self, src: int, dst: int,
                             capabilities: FrozenSet[str] = frozenset(),
                             src_xy: Optional[Tile] = None,
                             dst_xy: Optional[Tile] = None
                             ) -> Set[str]:
        """If `dst` is unreachable, which capabilities would unlock a route?
        (Empty set means unreachable for some other reason.) With
        `src_xy` given, a tile-level journey diagnosis runs first: it
        is the only way to see obstacles (Cut trees) that seal a
        destination whose map-level route exists."""
        if src_xy is not None:
            jm = self.journey_missing_capabilities(
                src, src_xy, dst, dst_xy, capabilities)
            if jm:
                return jm
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
