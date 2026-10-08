"""Global map graph for Pokémon Red.

Two data sources, both cheap and already on disk:

1. ``v2/map_data.json`` - the SB3 training env's visual map. Each of the 226
   regions carries a global ``[x, y]`` in "map tile" units and a ``[w, h]``
   size. Rectangles that touch edge-to-edge are geographically adjacent, so
   the whole of Kanto becomes a graph without any hand-written table.

2. ``portals.json`` (written here) - warp destinations the agent has actually
   read out of game RAM (``wWarpEntries``). A door in Pallet Town whose
   ``dest_map`` is 1 *is* the edge "Pallet Town -> Viridian City"; recording
   it as the agent walks turns the warp table into a persistent navigation
   graph that survives restarts.

The graph answers one question the local 10x9 collision grid cannot:
"which way is Viridian City from here, and through which maps?" Route
finding is BFS over map nodes; the returned hop list names each intermediate
map and the compass direction to take on the CURRENT map to start the route.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

from pokemon_agent.collision import (PLAYER_ROW, PLAYER_COL,
                                     COL_LABELS as _CELL_COLS)

logger = logging.getLogger("pokemon-agent.map_graph")

DEFAULT_MAP_DATA = "/home/vlad/pokemon-agent/v2/map_data.json"
DEFAULT_PORTALS = os.path.expanduser("~/.pokemon-agent/portals.json")

# Compass deltas in map-tile space. Screen y grows SOUTH, x grows EAST.
_DIR_NAMES = {(-1, 0): "north", (1, 0): "south", (0, -1): "west", (0, 1): "east"}


def _rect(coord: Tuple[int, int], size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """(x, y, w, h) -> (x0, y0, x1, y1) with x1/y1 exclusive."""
    x, y = coord
    w, h = size
    return x, y, x + w, y + h


def _canon(name: str) -> str:
    """Normalize a map name for cross-source matching.

    Lowercase plus whitespace collapse so "Cinnabar Island" (MAP_NAMES) and
    "Cinnabar island" (map_data.json) resolve to the same key. Punctuation is
    left alone - apostrophes in "Oak's Lab" are meaningful.
    """
    return " ".join(str(name).lower().split())


def _touch(a: Tuple[int, int, int, int],
           b: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
    """Compass direction from rect *a* to rect *b* if they share an edge.

    Returns (dy, dx) in map-tile steps, or None when the rectangles are not
    edge-adjacent. Only strict edge adjacency counts - corner touches and
    overlaps are ignored, because they do not describe a walkable connection.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    # Vertical: a directly above b (and x ranges overlap)
    if ay1 == by0 and not (ax1 <= bx0 or bx1 <= ax0):
        return (1, 0)   # b is SOUTH of a
    if by1 == ay0 and not (ax1 <= bx0 or bx1 <= ax0):
        return (-1, 0)  # b is NORTH of a
    # Horizontal: a directly left of b (and y ranges overlap)
    if ax1 == bx0 and not (ay1 <= by0 or by1 <= ay0):
        return (0, 1)   # b is EAST of a
    if bx1 == ax0 and not (ay1 <= by0 or by1 <= ay0):
        return (0, -1)  # b is WEST of a
    return None


class MapGraph:
    """Map-level navigation graph: nodes are maps, edges are adjacencies."""

    def __init__(self, map_data_path: str = DEFAULT_MAP_DATA,
                 portals_path: str = DEFAULT_PORTALS):
        self.map_data_path = map_data_path
        self.portals_path = portals_path
        self.maps: Dict[str, dict] = {}          # name -> {coord, size, id}
        self.adj: Dict[str, Dict[str, Tuple[int, int]]] = {}   # name -> {neighbor: dir}
        # Portals discovered from RAM: name -> {cell: {dest_map, dest_map_name, row, col}}
        self.portals: Dict[str, Dict[str, dict]] = {}
        self._load_maps()
        self._build_adjacency()
        self.load_portals()

    # --- load ---------------------------------------------------------------

    def _load_maps(self) -> None:
        try:
            with open(self.map_data_path, "r") as f:
                regions = json.load(f).get("regions", [])
        except Exception as e:  # pragma: no cover - file presence is checked live
            logger.warning("map graph: cannot read %s: %s", self.map_data_path, e)
            return
        for e in regions:
            name = e.get("name")
            coord = e.get("coordinates") or [0, 0]
            size = e.get("tileSize") or [0, 0]
            if not name or e.get("id") == "-1":
                continue          # skip the whole-region "Kanto" pseudo-entry
            self.maps[name] = {
                "coord": (int(coord[0]), int(coord[1])),
                "size": (int(size[0]), int(size[1])),
                "id": e.get("id"),
            }
        # Case/spacing-insensitive index: MAP_NAMES says "Cinnabar Island"
        # while map_data.json says "Cinnabar island". Without this the two
        # name spaces silently fail every lookup.
        self._index: Dict[str, str] = {_canon(n): n for n in self.maps}
        logger.info("map graph: %d maps loaded", len(self.maps))

    def resolve(self, name: Optional[str]) -> Optional[str]:
        """Canonical graph key for a map name, ignoring case and spacing."""
        if not name:
            return None
        if name in self.maps:
            return name
        return self._index.get(_canon(name))


    def _build_adjacency(self) -> None:
        rects = {n: _rect(m["coord"], m["size"]) for n, m in self.maps.items()}
        names = list(rects)
        self.adj = {n: {} for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = _touch(rects[a], rects[b])
                if d is None:
                    continue
                self.adj[a][b] = d
                self.adj[b][a] = (-d[0], -d[1])
        edges = sum(len(v) for v in self.adj.values()) // 2
        logger.info("map graph: %d adjacency edges", edges)

    # A real Gen 1 door is one or two tiles wide, so a map never has more
    # than a couple of cells leading to the same destination. Count alone is
    # too blunt (a city legitimately has many doors); a big group pointing at
    # ONE map is the signature of a stale-frame read, where the garbage warp
    # table decodes to whichever map the agent just came from.
    MAX_SAME_DEST = 4

    def load_portals(self) -> None:
        try:
            with open(self.portals_path, "r") as f:
                raw = json.load(f)
        except FileNotFoundError:
            self.portals = {}
            return
        except Exception as e:
            logger.warning("map graph: bad portals file: %s", e)
            self.portals = {}
            return
        clean: Dict[str, Dict[str, dict]] = {}
        dropped_maps = 0
        dropped_cells = 0
        for name, cells in (raw or {}).items():
            if not isinstance(cells, dict):
                continue
            grouped: Dict[str, Dict[str, dict]] = {}
            for cell, entry in cells.items():
                if not isinstance(entry, dict) or not entry.get("dest_map_name"):
                    dropped_cells += 1
                    continue        # unnamed: never routable, see record_warp
                grouped.setdefault(entry["dest_map_name"], {})[cell] = entry
            kept: Dict[str, dict] = {}
            for dst, group in grouped.items():
                if len(group) > self.MAX_SAME_DEST:
                    # A legitimate gate interior (e.g. Viridian Forest South
                    # Gate) has many cells all pointing at ONE destination -
                    # that's what a connector IS. Drop only when the group is
                    # implausibly huge (>12) AND the destination matches the
                    # previously visited map pattern (stale-frame signature).
                    if len(group) > 12:
                        logger.warning(
                            "map graph: dropping %d portals %s -> %s "
                            "(implausible door width)", len(group), name, dst)
                        dropped_cells += len(group)
                        continue
                    # Otherwise it's a connector: keep the first 4 cells (the
                    # door is 1-2 tiles wide; extras are frame noise).
                    for c in list(group)[:self.MAX_SAME_DEST]:
                        kept[c] = group[c]
                    dropped_cells += len(group) - self.MAX_SAME_DEST
                    continue
                kept.update(group)
            if kept:
                clean[name] = kept
            else:
                dropped_maps += 1
        self.portals = clean
        if dropped_cells or dropped_maps:
            self.save_portals()
        n = sum(len(v) for v in self.portals.values())
        logger.info("map graph: %d recorded portals from %s "
                    "(%d cells / %d maps dropped as implausible)",
                    n, self.portals_path, dropped_cells, dropped_maps)

    # --- record -------------------------------------------------------------

    def record_warp(self, cur_map: str, warp: dict) -> bool:
        """Persist one warp read from RAM. Returns True if it was new/updated.

        ``warp`` is a collision.py warp dict: it has row/col/cell plus
        dest_map and (usually) dest_map_name. Destination name is resolved
        from dest_map when absent so the portal is usable immediately.
        """
        if not cur_map or not isinstance(warp, dict):
            return False
        # Store under the canonical graph name so portal lookups and the
        # geography adjacency share one key space (MAP_NAMES vs map_data.json
        # differ in case for e.g. Cinnabar Island).
        cur_map = self.resolve(cur_map) or cur_map
        dest_id = warp.get("dest_map")
        if dest_id is None:
            return False
        dest_name = warp.get("dest_map_name")
        if not dest_name or dest_name.startswith("Map "):
            try:
                from pokemon_agent.memory.red import MAP_NAMES
                dest_name = MAP_NAMES.get(int(dest_id))
            except Exception:
                dest_name = None
        dest_name = self.resolve(dest_name) or dest_name
        # Drop warps we cannot name. During a map transition wNumberOfWarps
        # briefly reads garbage and the table yields dozens of phantom cells
        # (all dest_map=255, no name); persisting them marks half the screen
        # as "doors", which then blocks route paths and misleads
        # leave_building. 255 is Gen 1's "return to the last outdoor map"
        # sentinel and is handled by the exit logic, not by name routing.
        if not dest_name:
            return False
        cell = warp.get("cell")
        if not cell:
            return False
        entry = {
            "dest_map": int(dest_id),
            "dest_map_name": dest_name,
            "row": warp.get("row"),
            "col": warp.get("col"),
        }
        bucket = self.portals.setdefault(cur_map, {})
        if bucket.get(cell) == entry:
            return False
        bucket[cell] = entry
        return True

    def save_portals(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.portals_path), exist_ok=True)
            tmp = self.portals_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.portals, f, indent=1, sort_keys=True)
            os.replace(tmp, self.portals_path)
        except Exception as e:  # pragma: no cover
            logger.warning("map graph: cannot save portals: %s", e)

    # --- query --------------------------------------------------------------

    def neighbors(self, name: str) -> Dict[str, Tuple[int, int]]:
        """Adjacent maps merged from geography and recorded portals."""
        name = self.resolve(name) or name
        out = dict(self.adj.get(name) or {})
        # Outgoing portals: a warp we read on *name* tells us where its doors go.
        for cell, p in (self.portals.get(name) or {}).items():
            dst = self.resolve(p.get("dest_map_name"))
            if dst and dst not in out:
                # Derive a compass direction from global coordinates.
                d = self._direction_between(name, dst)
                if d is not None:
                    out[dst] = d
        # Incoming portals, treated as bidirectional. A portal is recorded
        # against the map that OWNS the warp, so this edge is otherwise
        # one-way: Route 2 -> Viridian Forest gets learned, Viridian Forest ->
        # Route 2 does not, and routing out of the forest fails even though
        # the agent has walked it repeatedly. Doors and map connections are
        # two-way in this game; the rare one-way warp is not worth losing the
        # whole connection over.
        for src, cells in (self.portals or {}).items():
            if src == name or src in out:
                continue
            for entry in cells.values():
                if self.resolve(entry.get("dest_map_name")) == name:
                    d = self._direction_between(name, src)
                    if d is not None:
                        out[src] = d
                    break
        return out

    def _portal_direction(self, owner: str, dest: str) -> Optional[Tuple[int, int]]:
        """Direction to walk on *owner* to leave for *dest*, from door cells.

        A recorded portal cell is where the door sits ON owner's screen, so its
        offset from the player IS the direction to walk. This beats rectangle
        geometry whenever both exist: on Route 2 the door to Viridian Forest
        sits at cells D2/E4, i.e. NORTH, while the rectangles put the forest to
        the EAST - and that error is what made the agent loop
        Route 2 <-> Viridian Forest instead of progressing.
        """
        owner = self.resolve(owner) or owner
        dest = self.resolve(dest) or dest
        votes: Dict[Tuple[int, int], int] = {}
        for cell, entry in ((self.portals or {}).get(owner) or {}).items():
            if self.resolve(entry.get("dest_map_name")) != dest:
                continue
            try:
                col = _CELL_COLS.index(cell[0].upper())
                row = int(cell[1:]) - 1
            except (ValueError, IndexError):
                continue
            dr, dc = row - PLAYER_ROW, col - PLAYER_COL
            if dr == 0 and dc == 0:
                continue
            if abs(dr) >= abs(dc):
                key = (-1, 0) if dr < 0 else (1, 0)
            else:
                key = (0, -1) if dc < 0 else (0, 1)
            votes[key] = votes.get(key, 0) + 1
        if not votes:
            return None
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def _direction_between(self, a: str, b: str) -> Optional[Tuple[int, int]]:
        a = self.resolve(a) or a
        b = self.resolve(b) or b
        # A door we have actually seen on `a` outranks the rectangle guess.
        # Only the owner side is used: the reverse would be an assumption about
        # relative layout, which is the very thing that is unreliable here.
        d = self._portal_direction(a, b)
        if d is not None:
            return d
        if a not in self.maps or b not in self.maps:
            return None
        d = _touch(_rect(self.maps[a]["coord"], self.maps[a]["size"]),
                   _rect(self.maps[b]["coord"], self.maps[b]["size"]))
        if d is not None:
            return d
        # Fall back to the dominant axis between the two coordinates.
        ax, ay = self.maps[a]["coord"]
        bx, by = self.maps[b]["coord"]
        dx, dy = bx - ax, by - ay
        if abs(dy) >= abs(dx):
            return (1, 0) if dy > 0 else (-1, 0)
        return (0, 1) if dx > 0 else (0, -1)

    def route(self, start: str, goal: str,
              max_hops: int = 12) -> Optional[List[str]]:
        """Shortest list of maps from *start* to *goal* (inclusive), or None."""
        start = self.resolve(start) or start
        goal = self.resolve(goal) or goal
        if start == goal:
            return [start]
        if start not in self.maps or goal not in self.maps:
            return None
        seen = {start}
        q = deque([(start, [start])])
        while q:
            node, path = q.popleft()
            if len(path) > max_hops:
                continue
            for nb in self.neighbors(node):
                if nb in seen:
                    continue
                if nb == goal:
                    return path + [nb]
                seen.add(nb)
                q.append((nb, path + [nb]))
        return None

    def first_hop_direction(self, start: str, goal: str) -> Optional[str]:
        """Compass direction to take on *start* to begin the route to *goal*."""
        r = self.route(start, goal)
        if not r or len(r) < 2:
            return None
        d = self.neighbors(r[0]).get(r[1])
        return _DIR_NAMES.get(d) if d is not None else None

    def neighbor_in_direction(self, name: str,
                              compass: str) -> Optional[str]:
        """The map reached by leaving *name* toward *compass*, if any.

        Used to record which map a route hop is *supposed* to reach, so the
        caller can tell a successful transition from wandering elsewhere.
        """
        for nb, delta in self.neighbors(name).items():
            if _DIR_NAMES.get(delta) == compass:
                return nb
        return None
