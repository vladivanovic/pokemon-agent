#!/usr/bin/env python3
"""Von - clean two-tier execution engine and driver.

    Tier 1  planner (VLM | rules)  -> ONE goal string        (planner_client)
    Tier 2  this file (Von)        -> execute it             (BFS + scorer)

Execution contract
------------------
    1. goal text      -> destination map     (_graph_destination)
    2. destination    -> compass             (map_graph.first_hop_direction)
    3. compass        -> concrete walk path  (_map_edge_path, BFS over the
                                              STITCHED map = winding-safe)
    4. path offered as the SOLE option       (one option cannot oscillate)

That is the whole routing strategy. There are deliberately NO per-map
`if cur_map == ...` overrides, NO compass locks, NO stall cooldowns and NO
pinball detectors: the planner owns strategy, BFS owns topology, and a
single-option choice set owns determinism.

Battles stay deterministic (gen1_battle): an exact type chart at ~0ms beats
a language model at ~2s/turn, every time.

Ported verbatim from the pre-split engine because it took hundreds of live
runs to get right: the BFS/ledge/map-walk primitives, the driver I/O, heal,
scorer and preflight methods, the dialog and battle option assembly, the
battle-policy override, and the action dispatch table.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
# Maps whose interior ROM geometry DOES NOT match the pokered
# ground-truth decode (proven live: 241/445/114 conflicts on Mt Moon
# cave floors; the fold and bumps are authoritative). Used to gate
# the static/gt merge: on these maps a cell opens ONLY by physical
# occupancy (visited/player_on), never by static/gt claims.
_CAVE_MAP_RE = re.compile(r"Mt Moon (B?\dF)|Rock Tunnel|Victory Road", re.I)

# MT MOON CROSSING SCRIPT (2026-10-26, user design): the cave crossing as
# an explicit, uniquely-numbered hop chain. Each hop names ONE cell on ONE
# floor; the driver commits that cell as the sole waypoint and advances
# only when the floor transition completes. No per-turn re-derivation, no
# equal-cost tie-breaks, no goal-text drift - the exact anti-flip model
# the user asked for ("go to B2F via ladder 2, go to B1F via ladder 1...").
# Ladder choice corrected 08:35 from the flight recorder: the agent has
# PHYSICALLY OCCUPIED (21,17) and the whole y6 corridor out to x20, so
# the (25,9) ladder (2 steps from the (25,11) corridor) is the proven
# climb - not the far-west (5,7), which is 50+ phantom steps away.
#   h0 1F   (17,11) descend -> B1F south pocket
#   h1 B2F  (25,9)  climb   -> B1F south pocket
#   h2 B1F  (13,27) descend -> B2F west (5,7)
#   h3 B2F  (5,7)   climb   -> B1F NE strip
#   h4 B1F  (27,3)  exit    -> Route 4 east
# from a window reading only on the second consecutive open vote.
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from pokemon_agent.scorer import get_scorer
from pokemon_agent.planner_client import PlannerClient, GOAL_VOCAB, rule_plan
from pokemon_agent.planner_rules import derive_goal
from pokemon_agent.map_graph import MapGraph
from pokemon_agent.state.builder import PC_CENTERS
from pokemon_agent.collision import render_ascii_map
from pokemon_agent import gen1_battle
from pokemon_agent.gen1_data import MOVE_BASE_PP   # max PP per move (pokered)

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pokemon-agent.von")
# urllib3 logs every HTTP request to /state (thousands per hour) - keep it
# quiet so the human log stays readable.
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)



# ---------------------------------------------------------------------------
# Constants (verbatim from the pre-split engine)
# ---------------------------------------------------------------------------

TRACE_PATH = os.environ.get(
    "POKEMON_TRACE",
    os.path.expanduser("~/.pokemon-agent/flight_recorder.jsonl"))

# The collision grid is screen-relative and the player is always at E5,
# i.e. (row 4, col 4) in the 9x10 walkability grids.
PLAYER_ROW, PLAYER_COL = 4, 4
GRID_ROWS, GRID_COLS = 9, 10
_DIRS: Dict[str, Tuple[int, int]] = {
    "walk_up": (-1, 0), "walk_down": (1, 0),
    "walk_left": (0, -1), "walk_right": (0, 1),
}
_REVERSE: Dict[str, str] = {
    "up": "down", "down": "up", "left": "right", "right": "left",
}
# The map graph speaks compass (north/south/east/west); the button actions
# speak up/down/left/right. Translating here is the difference between a
# working route hop and a handler that silently falls through to "any legal
# direction" (which is what made the agent alternate on the spot).
COMPASS_TO_WALK: Dict[str, str] = {
    "north": "up", "south": "down", "east": "right", "west": "left",
}
# (dx, dy) -> walk action. Note _DIRS above is ordered (dy, dx); the Pokecenter
# fields are (dx, dy) like every other delta in the state. Keeping a separate
# map is deliberate - using one for the other silently mirrors the step.
_DELTA_TO_WALK: Dict[Tuple[int, int], str] = {
    (0, -1): "walk_up", (0, 1): "walk_down",
    (-1, 0): "walk_left", (1, 0): "walk_right",
}
# (dy, dx) map-tile delta -> compass, for the geographic fallback below.
_DELTA_TO_COMPASS: Dict[Tuple[int, int], str] = {
    (-1, 0): "north", (1, 0): "south", (0, -1): "west", (0, 1): "east",
}
# Bare direction -> the one that undoes it.
_OPPOSITE_WALK: Dict[str, str] = {
    "up": "down", "down": "up", "left": "right", "right": "left",
}
COL_LABELS_LOCAL = "ABCDEFGHIJ"




SHOP_ACTIONS: List[str] = [
    "press_a", "wait_60",   # talk -> "How may I serve you?" (BUY/SELL/EXIT)
    "press_a", "wait_60",   # BUY (cursor starts on BUY)
    "press_a", "wait_60",   # first item (Poke Ball)
    "press_up", "wait_30",  # quantity -> maximum
    "press_a", "wait_60",   # confirm quantity
    "press_a", "wait_60",   # YES on "That will be $X. OK?"
    "wait_60",
    "press_b", "wait_30",   # leave the "Anything else?" prompt
    "press_b", "wait_30",   # back to BUY/SELL/EXIT
    "press_b", "wait_30",   # EXIT
]

# Mart inventories - pokered data/items/marts.asm ground truth. Keyed by
# the map the clerk stands on. The planner reads these so it never issues
# an order the shelf can't fill (Viridian Mart sells Poke Balls from the
# start - no Pokedex gate - the real gate is money: $200/ball).
MART_STOCK: Dict[str, List[str]] = {
    "Viridian Mart": ["POKE_BALL", "ANTIDOTE", "PARLYZ_HEAL", "BURN_HEAL"],
    "Pewter Mart": ["POKE_BALL", "POTION", "ESCAPE_ROPE", "ANTIDOTE",
                    "BURN_HEAL", "AWAKENING", "PARLYZ_HEAL"],
    "Cerulean Mart": ["POKE_BALL", "POTION", "REPEL", "ANTIDOTE", "BURN_HEAL",
                      "AWAKENING", "PARLYZ_HEAL"],
    "Vermilion Mart": ["POKE_BALL", "SUPER_POTION", "ICE_HEAL", "AWAKENING",
                       "PARLYZ_HEAL", "REPEL"],
}

# Clerk ground truth from pokered data/maps/objects/*.asm: every mart
# clerk is STAY at (0,5) facing RIGHT. The talk mechanic extends range
# across ONE counter tile (wTilesetTalkingOverTiles), so the player
# parks EAST of the clerk on the counter row and faces LEFT - the game
# resolves the talk over the counter. NO clerk-sprite chasing: the tile
# is a property of the map, not of a moving sprite read.
MART_CLERK: Dict[str, Dict[str, Any]] = {
    "Viridian Mart": {"x": 0, "y": 5, "face": "left"},
    "Pewter Mart": {"x": 0, "y": 5, "face": "left"},
    "Cerulean Mart": {"x": 0, "y": 5, "face": "left"},
    "Vermilion Mart": {"x": 0, "y": 5, "face": "left"},
}

# ROM ground-truth walkability per map id (data/maps/pokered_walkability.json,
# built from the pret blocksets). Keyed by decimal map id string. Small
# interiors (marts, pokecenters) are 8x8 - the live 10x9 window reads
# off-map padding as walls at their edges, which is what trapped the
# park path in the Viridian Mart. GT wins inside known interiors.
_GT_WALK_PATH = os.path.expanduser(
    "~/pokemon-agent/data/maps/pokered_walkability.json")
_GT_WALK: Dict[str, Dict[str, Any]] = {}


def _gt_walk(map_id: int) -> Optional[Dict[str, Any]]:
    """ROM walkability grid for a map id, lazily loaded once."""
    if not _GT_WALK:
        try:
            with open(_GT_WALK_PATH) as f:
                _GT_WALK.update(json.load(f))
        except Exception as exc:  # missing file: fall back to live grid
            logger.warning("gt walkability unavailable: %s", exc)
            return None
    return _GT_WALK.get(str(map_id))



def _compact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Trim the full state dict to what a turn actually needs.

    The ASCII map is passed separately in the prompt body, and raw tile_ids /
    timestamps are dropped - they burn tokens without informing a decision.
    """
    p = state.get("player") or {}
    dialog = state.get("dialog") or {}
    battle = state.get("battle") or {}
    enemy = battle.get("enemy") or {}
    in_battle = battle.get("in_battle")
    dialog_active = dialog.get("text_active") or dialog.get("input_locked")

    party = []
    for m in state.get("party") or []:
        party.append({
            "nickname": m.get("nickname"), "species": m.get("species"),
            "level": m.get("level"), "hp": m.get("hp"), "max_hp": m.get("max_hp"),
            "status": m.get("status"), "types": m.get("types"),
            # PP included so Hermes never picks a 0-PP move in battle -
            # names alone left it blind (wedged menu-looping vs Weedle).
            "moves": [{"name": mv.get("name"), "pp": mv.get("pp", 0)}
                      if isinstance(mv, dict) else mv
                      for mv in m.get("moves") or []],
        })

    out: Dict[str, Any] = {
        "map": (state.get("map") or {}).get("map_name"),
        "position": p.get("position"),
        "facing": p.get("facing"),
        "money": p.get("money"),
        "badges": p.get("badges"),
        "party": party if in_battle else [],   # drop full party outside battle
        "active_mon": state.get("active_mon") if in_battle else None,
        "text_active": dialog.get("text_active") if dialog_active else None,
        "input_locked": dialog.get("input_locked") if dialog_active else None,
        "in_battle": in_battle,
        "enemy": ({"species": enemy.get("species"), "level": enemy.get("level"),
                   "hp": enemy.get("hp"), "max_hp": enemy.get("max_hp"),
                   "types": enemy.get("types")}
                  if in_battle else None),
        "status": state.get("status"),
        "dialog_text": state.get("dialog") if dialog_active else None,
    }
    # Bag: the VLM needs item counts to avoid impossible orders (a catch
    # goal with 0 Poke balls can never succeed). Compact form only.
    bag = state.get("bag") or []
    if bag:
        out["bag"] = [{"item": b.get("item"), "qty": b.get("quantity")}
                      for b in bag]
    # Mart stock: what THIS city's Mart sells (pokered marts.asm ground
    # truth). The VLM can then see e.g. that Viridian Mart sells Poke
    # Balls from the start - and never issue an order the shelf can't fill.
    mart_stock = MART_STOCK.get((state.get("map") or {}).get("map_name", ""))
    if mart_stock:
        out["mart_sells"] = mart_stock
    col = state.get("collision") or {}
    people = [{"cell": s.get("cell"), "who": s.get("who")}
              for s in col.get("sprites") or []]
    if people:
        out["people"] = people
    exits = [{"cell": w.get("cell"), "to": w.get("dest_map_name")}
             for w in col.get("warps") or []]
    if exits:
        out["exits"] = exits
    return out

def _progress_fingerprint(state: Dict[str, Any]) -> tuple:
    flags = state.get("flags") or {}
    party = state.get("party") or []
    battle = state.get("battle") or {}
    enemy = battle.get("enemy") or {}
    return (
        (state.get("map") or {}).get("map_id"),
        len(party),
        sum(m.get("level", 0) for m in party),
        flags.get("badge_count", 0),
        bool(flags.get("has_pokedex")),
        bool(flags.get("has_oaks_parcel")),
        flags.get("pokedex_owned", 0),
        len(state.get("bag") or []),
        # Dealing damage is progress: a gym battle can run minutes without
        # changing anything else.
        enemy.get("hp") if battle.get("in_battle") else None,
    )

# ---------------------------------------------------------------------------
# Grid helpers - screen-relative pathing over verified walkability
# ---------------------------------------------------------------------------

def _grid_open(passable: List[List[bool]], r: int, c: int) -> bool:
    return bool(passable
                and 0 <= r < len(passable)
                and 0 <= c < len(passable[r])
                and passable[r][c])


def _legal_dirs(passable: List[List[bool]]) -> Dict[str, Tuple[int, int]]:
    """Directions the collision map says are actually possible from E5."""
    return {name: d for name, d in _DIRS.items()
            if _grid_open(passable, PLAYER_ROW + d[0], PLAYER_COL + d[1])}


def _path_to(passable: List[List[bool]], target: Tuple[int, int],
             limit: int = 12,
             ledges: Optional[List[List[bool]]] = None) -> List[str]:
    """BFS from E5 to *target*, returning walk actions.

    Trivially cheap on a 10x9 grid, and it converts "one LLM call per tile"
    into "one call per destination" - the single biggest speed win available
    when the decision model is the bottleneck.

    A ledge can only be entered moving DOWN, so a route that would climb one
    is rejected rather than silently walked into a wall.
    """
    start = (PLAYER_ROW, PLAYER_COL)
    if target == start:
        return []
    seen = {start}
    q = deque([(start, [])])
    while q:
        (r, c), path = q.popleft()
        if len(path) >= limit:
            continue
        for name, (dr, dc) in _DIRS.items():
            nr, nc = r + dr, c + dc
            if (nr, nc) in seen:
                continue
            if not (0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS):
                continue
            # A ledge is one-directional: enterable only from above.
            if (ledges and nr < len(ledges) and nc < len(ledges[nr])
                    and ledges[nr][nc] and name != "walk_down"):
                continue
            # The destination itself may be a door/NPC tile that reads as
            # blocked; allow stepping onto it as the final move.
            if not _grid_open(passable, nr, nc) and (nr, nc) != target:
                continue
            seen.add((nr, nc))
            if (nr, nc) == target:
                return path + [name]
            q.append(((nr, nc), path + [name]))
    return []

MAP_WALK_PATH = os.path.expanduser("~/.pokemon-agent/map_walk.json")
MAP_BOUNDS_PATH = os.path.expanduser(
    os.environ.get("POKEMON_MAP_BOUNDS",
                   os.path.join(os.path.dirname(MAP_WALK_PATH),
                                "map_bounds.json")))


def _load_map_bounds() -> Dict[str, Tuple[int, int]]:
    """Map name -> (W, H) in walking-grid tiles, from pokered constants.

    BFS and edge/frontier paths must never commit a target outside these
    bounds: a camera-edge fold at a map's border stamps phantom cells
    (x=-1 etc.) walkable, and an unbounded BFS commits them as targets
    the game can never reach (proven live: Viridian School House target
    (-1,2) wedged the agent 2600+ turns bouncing (0,1)<->(0,2))."""
    try:
        with open(MAP_BOUNDS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        return {str(k): (int(v[0]), int(v[1])) for k, v in raw.items()}
    except (OSError, ValueError, TypeError, IndexError):
        logger.warning("map_bounds.json missing/invalid - no bounds clamp")
        return {}


MAP_BOUNDS: Dict[str, Tuple[int, int]] = _load_map_bounds()

WALK_EDGES_PATH = os.path.expanduser(
    os.path.join(os.path.dirname(MAP_WALK_PATH), "walk_edges.json"))


def _load_walk_edges() -> Dict[str, List[Dict[str, Any]]]:
    """Pokered map-connection walk edges: map -> [{dir, dest, cell}].
    Map connections are NOT warps - the player just walks across the
    border - so neither the gt warp table nor learned portals record
    them. Without them the router starves on the one true exit (proven
    live: Viridian City's Route 2 north exit (7,0) invisible while the
    wrong-direction Route 1 south warp-cells (learned with drifted
    coordinates) outranked everything)."""
    try:
        with open(WALK_EDGES_PATH, encoding="utf-8") as f:
            return {str(k): v for k, v in json.load(f).items()}
    except (OSError, ValueError, TypeError):
        logger.warning("walk_edges.json missing/invalid - no walk edges")
        return {}


WALK_EDGES: Dict[str, List[Dict[str, Any]]] = _load_walk_edges()

# ROM-direct walkability - THE map authority (2026-10-26): decoded from the
# ROM's own map headers + blocksets + collision lists (see
# references/mt-moon-crossing.md). Per map:
#   edges: {'north'/'south'/'west'/'east': grid[ty][tx]} where 1 means
#          "entering this cell moving that direction is legal" (Gen 1
#          collision is directional - it tests the 2 tiles facing travel);
#   walk:  cell-stick grid - 1 when all four tiles under the player are free;
#   hops:  ledge transitions, keyed "x,y,walk_dir" -> "x2,y2".
# Everything else that used to claim walkability (static_walk.json from the
# old decoder, the fold's stitched grid) is now a live-window correction
# layer on top of this, never a competing authority.
ROM_WALK_PATH = os.path.join(os.path.dirname(MAP_WALK_PATH),
                             "rom_walk.json")
try:
    with open(ROM_WALK_PATH, encoding="utf-8") as _f:
        _ROM_RAW: Dict[str, dict] = json.load(_f)
except (OSError, ValueError, TypeError):
    _ROM_RAW = {}
ROM_EDGES: Dict[str, Dict[str, list]] = {
    n: r["edges"] for n, r in _ROM_RAW.items()
    if isinstance(r, dict) and "edges" in r}
# Cell-walkability seed for the live-window overlay, straight off the same
# decode (this replaced static_walk.json, the older decoder's output).
STATIC_WALK: Dict[str, set] = {
    n: {(x, y) for y, row in enumerate(r["walk"])
        for x, w in enumerate(row) if w}
    for n, r in _ROM_RAW.items() if isinstance(r, dict) and "walk" in r}
# Ledge hops, in the shape the BFS wants: {(x, y, walk_dir): (x2, y2)}.
ROM_HOPS: Dict[str, Dict[tuple, tuple]] = {}
for _n, _r in _ROM_RAW.items():
    if not isinstance(_r, dict) or not _r.get("hops"):
        continue
    _h = {}
    for _k, _v in _r["hops"].items():
        try:
            _x, _y, _d = _k.split(",")
            _lx, _ly = _v.split(",")
            _h[(int(_x), int(_y), _d)] = (int(_lx), int(_ly))
        except (ValueError, AttributeError):
            continue
    ROM_HOPS[_n] = _h


def _in_bounds(map_name: str, cell: Tuple[int, int]) -> bool:
    """True when *cell* lies inside *map_name*'s walking grid. Unknown
    maps are unbounded (no clamp) - the clamp only ever REJECTS."""
    b = MAP_BOUNDS.get(map_name)
    if not b:
        return True
    return 0 <= cell[0] < b[0] and 0 <= cell[1] < b[1]
# CONFIRMED-BLOCK SHIELD registry note: cells physically proven blocked
# by the zero-displacement watchdog (2x bump-no-move) OR hand-pinned from
# ROM/live evidence. The live-window fold reads camera-edge rows open
# every turn and would re-stamp the lie over any correction; these cells
# stay False until the PLAYER IS SEEN STANDING ON ONE (the strongest
# possible refutation - visited beats everything). Loaded once at module
# scope from confirmed_blocks.json - there is NO second assignment.


def _load_map_walk() -> Dict[str, Dict[tuple, bool]]:
    """Restore the stitched per-map grids.

    Without this the grid restarts empty every run, so a map the agent has
    already walked (Viridian Forest) has to be re-learned screen by screen -
    and if it is wedged it never moves, so it never re-learns it. Persisting
    turns minutes of exploration into permanent knowledge.
    """
    try:
        with open(MAP_WALK_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    out: Dict[str, Dict[tuple, bool]] = {}
    for m, cells in (raw or {}).items():
        if m == "__ledges__":
            continue          # ledge edges ride in the same file
        d: Dict[tuple, bool] = {}
        for c in cells or []:
            try:
                d[(int(c[0]), int(c[1]))] = bool(c[2])
            except (TypeError, ValueError, IndexError):
                continue
        if d:
            out[m] = d
    logger.info("map_walk: restored %d maps (%d cells)", len(out),
                sum(len(v) for v in out.values()))
    return out


def _save_map_walk(mw: Dict[str, Dict[tuple, bool]]) -> None:
    """Persist the live-window-corrected grid (crash insurance only)."""
    try:
        os.makedirs(os.path.dirname(MAP_WALK_PATH), exist_ok=True)
        raw = {m: [[x, y, bool(ok)] for (x, y), ok in cells.items()]
               for m, cells in mw.items()}
        tmp = MAP_WALK_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        os.replace(tmp, MAP_WALK_PATH)
    except OSError:
        logger.exception("map_walk save failed")


def _stamp_screen(map_walk: Dict[tuple, bool], player_xy: Tuple[int, int],
                  passable: List[List[bool]],
                  warps: Optional[List[Dict[str, Any]]] = None,
                  visited: Optional[set] = None,
                  map_name: str = "",
                  npc_cells: Optional[set] = None) -> None:
    """Fold the current 10x9 screen into a per-map global walkable grid.

    The local grid is only ever one screen. Viridian Forest is 34x48 and the
    route through it winds, so a compass rule ("walk north") cannot follow it
    - the agent walks into a tree, the map never changes, and after four
    attempts the direction gets blacklisted. Stitching every screen we see
    into a persistent per-map grid gives the map the shape it actually has.

    Keyed by MAP coordinates (x, y), which the collision warps already carry
    as map_x/map_y, so doors can be targets even while off-screen.

    Two rules keep the accumulated grid honest:

    * Walkability is STICKY. A cell seen walkable stays walkable even if a
      later screen reports it blocked. Blocked readings are the unreliable
      ones: an NPC standing on a tile shadows it (the Pewter Mart Super Nerd
      walls off the counter aisle), and screen borders read as solid. The
      game's geometry never changes, so the first "walkable" is the truth.
    * Warp tiles are forced walkable. The collision reader marks door tiles
      blocked, which severs every building and gate from the stitched graph -
      BFS could never path to or through a door.

    Ledge edges are NOT recorded here - the collision reader's ledge grid is
    unreliable (it does not flag the Route 3 ledge). They are inferred from
    observed 2-tile down hops in _laya_turn and stored in self.ledge_edges.
    """
    px, py = int(player_xy[0]), int(player_xy[1])
    # MINIMAL WINDOW OVERLAY. The ROM decode is the map; the live window
    # only contributes what the ROM cannot know:
    #   * the cell the player physically occupies is open (the body
    #     cannot stand in a wall) - this is the only authority that
    #     overrides the decode;
    #   * a live NPC cell is blocked while the NPC stands there;
    #   * a live OPEN reading opens a cell (the decode has false walls);
    #   * a live BLOCKED reading closes a cell only when the decode does
    #     not call it open (camera borders and battle frames read as
    #     walls) and the player has never stood on it.
    player_on = (px, py)
    _static = STATIC_WALK.get(map_name, set())
    _n_rows = len(passable or [])
    _n_cols = len((passable or [[]])[0]) if _n_rows else 0
    for r, row in enumerate(passable or []):
        for c, ok in enumerate(row or []):
            key = (px + (c - PLAYER_COL), py + (r - PLAYER_ROW))
            _border = (r == 0 or c == 0 or r == _n_rows - 1
                       or c == _n_cols - 1)
            if key == player_on or (visited is not None and key in visited):
                map_walk[key] = True
            elif npc_cells and key in npc_cells:
                map_walk[key] = False
            elif ok:
                map_walk[key] = True
            elif not _border:
                # LIVE CLOSE. The window is the game's own collision output,
                # so a blocked reading inside the window (not on its
                # artifact-prone border ring) closes the cell even when the
                # ROM decode claims it open. Route 4's ledge terraces are
                # exactly this: the decode reads the terrace rows open, the
                # game refuses them, and the BFS kept routing to Cerulean
                # through a wall (proven live 2026-10-27, agent parked at
                # (19,14) dispatching walk_right into (20,14) forever).
                # A close is per-turn and self-correcting: once the window
                # reads the cell open again, the next turn reopens it.
                map_walk[key] = False
    for w in (warps or []):
        mx, my = w.get("map_x"), w.get("map_y")
        if mx is not None and my is not None:
            map_walk[(int(mx), int(my))] = True


def _map_step_toward(map_walk: Dict[tuple, bool],
                     start_xy: Tuple[int, int],
                     goal_xy: Tuple[int, int],
                     limit: int = 400,
                     ledge_edges: Optional[Dict[tuple, tuple]] = None,
                     avoid_warps: Optional[set] = None,
                     avoid_cells: Optional[set] = None,
                     map_name: str = "",
                     dir_edges: Optional[Dict[str, list]] = None) -> List[str]:
    """BFS over the accumulated per-map grid; return walk actions to *goal*.

    Empty list means the goal is not reachable in what we have seen so far -
    which is expected while the map is still being explored, and is exactly
    the signal that we should explore rather than keep retrying a compass hop.

    Warp/door tiles are recorded as unwalkable in the stitched grid (the
    collision reader marks them blocked), so a door coordinate can never be a
    BFS goal. When the exact goal is not walkable, BFS for the nearest
    walkable cell ORTHOGONALLY ADJACENT to it instead - standing next to a
    door and stepping into it is how every Gen 1 warp fires.

    Ledge edges are also traversable: a ledge at (x, y) allows a
    2-tile DOWN hop to (x, y+2). The agent must walk down onto the ledge
    tile and the engine auto-hops them to the landing cell.

    avoid_cells: cells proven live-blocked (pokered walkable, emulator
    blocked - e.g. Route 3 (64,0)). BFS routes around them.

    dir_edges (ROM-direct decode 2026-10-26): per-direction entry grids
    {walk_name: grid[ty][tx]} from rom_walk.json - Gen 1 collision is
    DIRECTIONAL (entering a cell checks the 2-tile pair facing the direction
    of travel). When present for the map, the edge test REPLACES the
    cell-walkability test for every step; validated 920/920 walked cells.
    """
    start = (int(start_xy[0]), int(start_xy[1]))
    goal = (int(goal_xy[0]), int(goal_xy[1]))
    if start == goal:
        return []
    if dir_edges is None:
        # Auto-wire the ROM-direct directional decode by map name: every
        # call site inherits it with zero plumbing.
        dir_edges = ROM_EDGES.get(map_name)
    if ledge_edges is None:
        # Same for the ROM ledge hops (the runtime-learned store is gone).
        ledge_edges = ROM_HOPS.get(map_name)

    def _enterable(cell: Tuple[int, int]) -> bool:
        # ROM-direct directional decode: a cell is enterable iff ANY of its
        # four directional edges admits entry (and the cell-level grid, when
        # the ROM edges do not cover the map, stays the authority).
        if dir_edges is not None:
            try:
                return any(dir_edges[d][cell[1]][cell[0]]
                           for d in ("north", "south", "west", "east"))
            except IndexError:
                return False
        return bool(map_walk.get(cell))

    if dir_edges is None and not map_walk.get(start):
        return []
    # Warp cells are FORCED walkable in the ground-truth grid (stairs and
    # doors), but walking ACROSS an unrelated warp tile fires it and
    # teleports the agent somewhere unplanned (proven live: the Pokecenter
    # exit door sat on the path back to the cave mouth and re-entered the
    # center on every pass). Exclude warp cells from the path unless one IS
    # the goal or the start.
    warp_block: set = set()
    if avoid_warps:
        warp_block = {w for w in avoid_warps if w != goal and w != start}
    cell_block: set = set()
    if avoid_cells:
        cell_block = {c for c in avoid_cells if c != goal and c != start}
    # Goal not walkable (door tile): aim at a walkable orthogonal neighbour.
    if not _enterable(goal):
        adjacent = [g for g in ((goal[0] + 1, goal[1]), (goal[0] - 1, goal[1]),
                                (goal[0], goal[1] + 1), (goal[0], goal[1] - 1))
                    if _enterable(g)]
        if not adjacent:
            return []
        goal = min(adjacent,
                   key=lambda g: abs(g[0] - start[0]) + abs(g[1] - start[1]))
        if start == goal:
            return []
    seen = {start}
    q: deque = deque([(start, [])])
    while q:
        (mx, my), path = q.popleft()
        if (mx, my) == goal:
            return path
        if len(path) >= limit:
            continue
        for name, (dr, dc) in _DIRS.items():
            nxt = (mx + dc, my + dr)          # map x=col, y=row
            # LEDGE HOPS (ROM-direct, 2026-10-26): when the standing cell's
            # sampled tile is a ledge-capable tile (0x2C/0x39) and the tile
            # ahead is the matching ledge tile for this direction, the
            # engine auto-hops to the cell 2 tiles ahead. rom_walk.json
            # precomputes these per direction; the walk onto the ledge
            # tile never fires (the engine jumps it), so the BFS treats
            # source -> landing as a single edge for THAT direction only.
            if ledge_edges:
                _hop = ledge_edges.get((mx, my, name))
                if _hop is not None and _hop not in seen:
                    seen.add(_hop)
                    q.append((_hop, path + [name]))
                    continue
            if dir_edges is not None:
                # ROM-direct directional collision: entering nxt moving
                # `name` checks nxt's facing tile pair (rom_walk.json).
                _dname = {"walk_up": "north", "walk_down": "south",
                          "walk_left": "west", "walk_right": "east"}.get(name)
                try:
                    _step_ok = bool(dir_edges[_dname][nxt[1]][nxt[0]])
                except (KeyError, IndexError):
                    _step_ok = False
                # LIVE CLOSE beats the decode (Game Boy is the arbiter): the
                # window close in _apply_live_window sets the cell False in
                # map_walk; honour it here or the ROM edges would keep
                # routing into a wall the game refuses (Route 4 terraces).
                if _step_ok and map_walk.get(nxt) is False:
                    _step_ok = False
                if nxt in seen or not _step_ok:
                    if (name == "walk_down" and ledge_edges
                            and (mx, my) in ledge_edges):
                        nxt = ledge_edges[(mx, my)]
                        if nxt in seen:
                            continue
                    else:
                        continue
            elif nxt in seen or not map_walk.get(nxt):
                # Check ledge hop: if current cell is a ledge source and we're
                # walking down, the destination is the landing cell.
                if (name == "walk_down" and ledge_edges
                        and (mx, my) in ledge_edges):
                    nxt = ledge_edges[(mx, my)]
                    if nxt in seen or not map_walk.get(nxt):
                        continue
                else:
                    continue
            if nxt in warp_block or nxt in cell_block \
                    or not _in_bounds(map_name, nxt):
                continue
            seen.add(nxt)
            q.append((nxt, path + [name]))
    # F6 DIAGNOSTIC: primary BFS exhausted without reaching the goal.
    try:
        _g_reach = any((goal[0] + dx, goal[1] + dy) in seen
                       for dx, dy in ((0, 0), (1, 0), (-1, 0),
                                      (0, 1), (0, -1)))
        _g_blocked = goal in cell_block or goal in warp_block
        logger.info(
            "BFS-FAIL %s->%s seen_n=%d goal_reach=%s goal_blocked=%s "
            "grid_goal=%s avoid_w=%d avoid_c=%d",
            start, goal, len(seen), _g_reach, _g_blocked,
            map_walk.get(goal), len(warp_block), len(cell_block))
    except Exception:
        logger.exception("bfs-fail diag failed")
    return []


def _map_frontier_step(map_walk: Dict[tuple, bool],
                       start_xy: Tuple[int, int],
                       limit: int = 400) -> List[str]:
    """BFS to the nearest reachable cell that borders UNEXPLORED space.

    The escape hatch for a wedged agent. When a compass hop keeps failing
    (Pewter City's top-right pocket: 5,100 turns ping-ponging between two
    tiles) the grid holds a large connected component plus a severed pocket
    holding the goal. No amount of re-pathing within the component reaches
    it - the map has to GROW. Walking to the nearest cell adjacent to an
    unseen tile makes the agent stamp new geometry every turn, which is what
    eventually welds the components together.
    """
    start = (int(start_xy[0]), int(start_xy[1]))
    if not map_walk.get(start):
        return []
    # reverse lookup: (dx, dy) -> walk name  (_DIRS is name -> (dr, dc))
    step_name = {(dc, dr): name for name, (dr, dc) in _DIRS.items()}
    seen = {start}
    q: deque = deque([(start, [])])
    while q:
        (mx, my), path = q.popleft()
        # Frontier test: standing here, can we step into an unstamped cell?
        for (dx, dy), name in step_name.items():
            if (mx + dx, my + dy) not in map_walk:
                return path + [name]
        if len(path) >= limit:
            continue
        for name, (dr, dc) in _DIRS.items():
            nxt = (mx + dc, my + dr)
            if nxt in seen or not map_walk.get(nxt):
                continue
            seen.add(nxt)
            q.append((nxt, path + [name]))
    return []


def _map_edge_path(map_walk: Dict[tuple, bool], start_xy: Tuple[int, int],
                   direction: str, limit: int = 400,
                   ledge_edges: Optional[Dict[tuple, tuple]] = None,
                   avoid_cells: Optional[set] = None) -> List[str]:
    """BFS the stitched map; return the path to the furthest reachable cell
    along *direction*.

    This is the fix for winding maps. ``_path_toward_edge`` works on the
    single 10x9 screen, so in Viridian Forest "go north" walks into a tree and
    the map never changes - the edge is 40 tiles away around three corners.
    Over the accumulated per-map grid the same query follows the real route.

    Ledge edges (2-tile down hops) are traversable so a ledge can be used to
    cross a row that is otherwise a solid wall.

    avoid_cells: cells proven live-blocked - BFS routes around them.
    """
    dr, dc = _DIRS[f"walk_{direction}"]
    start = (int(start_xy[0]), int(start_xy[1]))
    if not map_walk.get(start):
        return []
    cell_block: set = set(avoid_cells or set())
    seen = {start}
    q: deque = deque([(start, [])])
    best: Tuple[int, List[str]] = (0, [])
    while q:
        (mx, my), path = q.popleft()
        proj = (my - start[1]) * dr + (mx - start[0]) * dc
        if proj > best[0]:
            best = (proj, path)
        if len(path) >= limit:
            continue
        for name, (ddr, ddc) in _DIRS.items():
            nxt = (mx + ddc, my + ddr)
            if nxt in seen or not map_walk.get(nxt):
                if (name == "walk_down" and ledge_edges
                        and (mx, my) in ledge_edges):
                    nxt = ledge_edges[(mx, my)]
                    if nxt in seen or not map_walk.get(nxt):
                        continue
                else:
                    continue
            if nxt in cell_block:
                continue
            seen.add(nxt)
            q.append((nxt, path + [name]))
    return best[1]


def _path_toward_edge(passable: List[List[bool]], direction: str,
                      avoid: Optional[set] = None,
                      ledges: Optional[List[List[bool]]] = None,
                      limit: int = 14) -> List[str]:
    """BFS to the reachable cell furthest along *direction* from E5.

    A blind ``walk_up`` x8 run is what walked the agent into Blue's House on
    the way out of Pallet Town: the door sits on the route to the north edge
    and stepping onto it triggers the warp. Following a path lets us route
    around door tiles (``avoid``) while still heading the right way.

    Returns walk actions, or [] when no progress can be made.
    """
    dr, dc = _DIRS[f"walk_{direction}"]
    back = f"walk_{_REVERSE[direction]}"
    avoid = avoid or set()
    start = (PLAYER_ROW, PLAYER_COL)
    seen: set = {start}
    q: deque = deque([(start, [])])
    best: Tuple[int, List[str]] = (0, [])   # (projection, path)
    while q:
        (r, c), path = q.popleft()
        # Projection along the travel axis: how far in *direction* we got.
        proj = (r - PLAYER_ROW) * dr + (c - PLAYER_COL) * dc
        if proj > best[0]:
            best = (proj, path)
        if len(path) >= limit:
            continue
        for name, (ddr, ddc) in _DIRS.items():
            if name == back:
                # Never expand backwards. Recomputing each turn then yields a
                # monotonic path; allowing the reverse is what made the agent
                # alternate walk_up/walk_down on the spot forever.
                continue
            nr, nc = r + ddr, c + ddc
            if (nr, nc) in seen:
                continue
            if not (0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS):
                continue
            if ledges and nr < len(ledges) and nc < len(ledges[nr]) \
                    and ledges[nr][nc] and name != "walk_down":
                continue
            if not _grid_open(passable, nr, nc):
                continue
            if (nr, nc) in avoid:
                # Never step on a door: route_toward_ means walk off the map
                # edge, so any warp tile is a wrong turn.
                continue
            seen.add((nr, nc))
            q.append(((nr, nc), path + [name]))
    return best[1]


def _frontier_targets(passable: List[List[bool]], visits: Dict[tuple, int],
                      cur_map: str, px: int, py: int,
                      union_global: Optional[set] = None,
                      map_offset: Optional[tuple] = None,
                      warp_cells: Optional[set] = None) -> List[Tuple[int, int]]:
    """On-screen cells the agent has never stood on, reachable from E5.

    Screen cells map to world coords by offsetting from the player: the player
    is always at (PLAYER_ROW, PLAYER_COL) and at world (px, py).

    When union_global (SB3 explored cells, global (y, x)) and map_offset
    ((mx, my) of the current map) are provided, each candidate's local
    (wx, wy) is converted to global (gy, gx) and checked against the union
    set too — RL-explored terrain is treated as known and stays out of the
    frontier.

    warp_cells: door tiles excluded from the frontier - doors are entered
    only as EXPLICIT route targets, never as exploration destinations.
    """
    out = []
    _wc = warp_cells or set()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if not _grid_open(passable, r, c):
                continue
            wx = px + (c - PLAYER_COL)
            wy = py + (r - PLAYER_ROW)
            if visits.get((cur_map, wx, wy), 0) != 0:
                continue
            # WARP-CELL EXCLUSION (pattern 47 corollary): a frontier target
            # on a door tile re-entered the building the agent had just
            # left (proven live: Pewter Gym exit at 10:36-10:42,
            # explore_H1 targeting the entrance door (16,17) seconds
            # after exiting).
            if (wx, wy) in _wc:
                continue
            if union_global and map_offset is not None:
                mx, my = map_offset
                # GLOBAL = local + map_offset + PAD(20), matching the SB3 env's
                # local_to_global(): gy = row + map_y + 20, gx = col + map_x + 20
                gy = wy + my + 20
                gx = wx + mx + 20
                if (gy, gx) in union_global:
                    continue
            out.append((r, c))
    return out

def _pinballed(recent: List[tuple], window: int = 12) -> bool:
    """True when the recent position trail is a back-and-forth pinball.

    Two shapes, because both were observed live and neither implies the other:

    * FEW DISTINCT TILES - Route 2's y=60 strip cycled five tiles
      ((2,60)..(11,60)) for 605 left/right reversals;
    * EXPLICIT REVERSALS (A -> B -> A) - Viridian City alternated
      goto_E1/goto_E2 at (32,11)/(32,10), and drifted across two adjacent
      pairs (4 tiles) for good measure.

    A tile count alone missed the 4-tile drift when the bar was <=3; reversals
    alone miss the Route 2 cycle, which never returns in a single step. The
    caller resets the trail on map change, so this only ever sees one map.
    """
    if len(recent) < window:
        return False
    flips = sum(1 for k in range(1, len(recent) - 1)
                if recent[k + 1] == recent[k - 1])
    return len(set(recent)) <= 5 or flips >= 6


def cell_label_local(col: int, row: int) -> str:
    """0-indexed (col,row) -> 'E5'. Mirrors collision.cell_label."""
    return f"{COL_LABELS_LOCAL[col]}{row + 1}"

def _bearing(row: int, col: int) -> str:
    """Compass direction of a screen cell relative to the player at E5.

    Laya cannot connect "B2" to an objective that says "go north", but it can
    connect "north" to it.
    """
    dr, dc = row - PLAYER_ROW, col - PLAYER_COL
    vert = "north" if dr < 0 else ("south" if dr > 0 else "")
    horiz = "west" if dc < 0 else ("east" if dc > 0 else "")
    if vert and horiz:
        return f"{vert}-{horiz}" if abs(dr) >= abs(dc) else f"{horiz}-{vert}"
    return vert or horiz or "here"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------



class VonDriver:
    # empirical edge-blacklist tunables (verbatim from the engine)
    EDGE_FAIL_LIMIT = 4
    EDGE_FAIL_TTL = 200


    def __init__(self, server: str, model: Optional[str] = None,
                 provider: Optional[str] = None,
                 turn_delay: float = 1.5, save_every: int = 20,
                 turn_timeout: int = 240, use_laya: bool = False,
                 laya_narrate_every: int = 0,
                 stall_seconds: float = 60.0, hermes_turns: int = 12,
                 vision: bool = True, replan_every: int = 120,
                 scorer: Optional[str] = None,
                 planner: str = "rules", plan_interval: int = 15):
        self.server = server.rstrip("/")
        self.model = model
        self.provider = provider
        self.turn_delay = turn_delay
        self.save_every = save_every
        self.turn_timeout = turn_timeout
        self.vision = vision
        self.vision_capable = vision
        # Single-brain design: Von drives, the planner plans. --laya is still
        # accepted so existing launch commands keep working, but Laya is not
        # loaded and not consulted.
        self.use_laya = False
        if use_laya:
            logger.warning("--laya accepted but unused: Von is the only driver")
        self.laya_narrate_every = laya_narrate_every
        self.stall_seconds = stall_seconds
        self.hermes_turns = hermes_turns
        self.replan_every = replan_every

        # --- session / loop state ---
        self.game_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.turn = 0
        self.last_pos: Optional[Any] = None
        self.stuck = 0
        self.consecutive_fail = 0
        self.last_confidence = 0.0
        self.recent: deque = deque(maxlen=8)
        self.prev_map: Optional[str] = None
        self.arrived_from: Optional[str] = None
        self.visits: Dict[tuple, int] = {}
        self.special_tiles: Dict[tuple, str] = {}
        self.union_global: set = set()
        self.map_offsets: Dict[str, tuple] = {}
        self.explore_union_path = os.environ.get(
            "POKEMON_EXPLORE_UNION",
            "/home/vlad/pokemon-agent/explore_map_union.json")
        self._load_map_offsets()
        try:
            self.map_graph: Optional[MapGraph] = MapGraph()
        except Exception:
            logger.exception("map graph init failed - continuing without it")
            self.map_graph = None
        self.map_changed_at: int = -99
        self.map_changed_time: float = 0.0
        self.progress_fp: Optional[tuple] = None
        self.progress_at: float = time.perf_counter()
        self._last_objs: Optional[list] = None
        self.failed_targets: Dict[tuple, int] = {}
        self.last_explore_target: Optional[tuple] = None
        self.last_explore_from: Optional[tuple] = None
        self._recent_pos: list = []
        self._recent_map: Optional[str] = None
        self.session_started_at: int = int(time.time())
        self.edge_fails: Dict[tuple, int] = {}
        self._edge_fail_turn: Dict[tuple, int] = {}
        # Pokered ground truth: verified warp tables + walkability grids.
        # Warps are exact (cross-validated against the emulator's own warp
        # reader); they seed the portal knowledge the router's tier 1 needs,
        # so Mt Moon / Rock Tunnel / every building works on run 1.
        from pokemon_agent.ground_truth import (load_ground_truth,
                                                warps_by_name,
                                                walk_grid_by_name,
                                                ledges_by_name)
        from pokemon_agent.gen1_data import MAP_NAMES as _GT_MAP_NAMES
        _gt = load_ground_truth()
        self.gt = _gt
        self.gt_warps: Dict[str, list] = warps_by_name(_gt, _GT_MAP_NAMES)
        self.gt_loaded: bool = bool(_gt)
        # PRIMARY map_walk = pokered ground truth. The stitched accumulator
        # only fills in what ground truth does not cover (it previously was
        # the only source and guessed geometry from screenshots - the source
        # of every wrong-edge patch this file used to carry).
        self.map_walk: Dict[str, Dict[tuple, bool]] = walk_grid_by_name(
            _gt, _GT_MAP_NAMES)
        for m, cells in _load_map_walk().items():
            if m in self.map_walk:
                # Stitched-wins-when-observed: the stitched overlay holds
                # cells the agent has PHYSICALLY walked on this ROM. pokered
                # geometry for tiny interiors can contradict the live ROM
                # (Reds House 2F: pokered 8x8 says the player's own tile is
                # a wall; the real room is wider). Where stitched has
                # observations, they override the static grid; gt only
                # fills cells stitched has never seen.
                for k, v in cells.items():
                    self.map_walk[m][k] = v
            else:
                dst = self.map_walk.setdefault(m, {})
                for k, v in cells.items():
                    dst.setdefault(k, v)
        logger.info("map_walk: %d ground-truth maps + stitched overlay (%d maps)",
                    len(_gt), len(self.map_walk))
        # MAP SEED: the ROM decode (rom_walk.json) is the only
        # walkability authority; the live window corrects it per turn in
        # _apply_live_window. Ledge hops come from the same decode.
        for _m, _cells in STATIC_WALK.items():
            _mw_m = self.map_walk.setdefault(_m, {})
            for _c in _cells:
                _mw_m.setdefault(_c, True)
        self.ledge_edges: Dict[str, Dict[tuple, tuple]] = dict(ROM_HOPS)
        logger.info("map seed: %d maps, %d with ROM ledge hops",
                    len(STATIC_WALK), len(ROM_HOPS))
        self._map_walk_saved_at: int = 0
        self._prev_xy: Optional[Tuple[int, int]] = None
        self._prev_xy_map: str = ""
        self._route_pending: Optional[tuple] = None
        self._route_last_pos: Optional[tuple] = None
        self._pending_dead_edge: Optional[tuple] = None

        # --- mode/stats fields the /stats endpoint and event stream read ---
        # (no arbitration any more, but the dashboard still queries these)
        self.mode: str = "von"
        self.mode_turns: int = 0
        self.hermes_budget: int = 0
        self.hermes_budget_max: int = 0
        self.hermes_goal: str = ""
        self.hermes_time_total: float = 0.0
        self.laya_time_total: float = 0.0
        self.last_mode_switch: float = time.perf_counter()
        self._hermes_last_pos: Optional[Any] = None

        # --- low-level scorer: von | expert | laya | llm ---
        self.scorer_name: str = ((scorer
                                  or os.environ.get("POKEMON_SCORER", "von"))
                                 .lower())
        self.local_label: str = self.scorer_name
        self.scorer: Optional[Any] = None
        from pokemon_agent.llm_direct import (DEFAULT_MODEL as _LLM_MODEL,
                                              DEFAULT_TIMEOUT as _LLM_TO,
                                              DEFAULT_URL as _LLM_URL)
        self.llm_url: str = os.environ.get("POKEMON_LLM_URL", _LLM_URL)
        self.llm_model: str = os.environ.get("POKEMON_LLM_MODEL", _LLM_MODEL)
        self.llm_timeout: float = float(
            os.environ.get("POKEMON_LLM_TIMEOUT", _LLM_TO))

        # --- planner (tier 1) ---
        self.planner_mode: str = (planner or "rules").lower()
        self.plan_interval: int = plan_interval
        self._planner: Optional[Any] = None
        if self.planner_mode == "vlm":
            try:
                self._planner = PlannerClient()
            except Exception:
                logger.exception("planner client init failed - using rules")
                self.planner_mode = "rules"
        self._plan_goal_text: str = ""
        self._plan_turn: int = -10 ** 9
        self._plan_map: str = ""
        # Dataset flywheel bookkeeping: what the planner decided + the screen
        # it saw this turn. Written by _plan_goal/_turn, read by step() when
        # the collector records a sample.
        self._latest_goal_text: str = ""
        self._latest_frame_png: bytes = b""

        # --- live training-data collector (dataset flywheel) ---
        # Records (frame, state, executed goal) every N turns while playing.
        # OFF by default; enable with POKEMON_COLLECT=1. The goal recorded is
        # the one actually executed, whatever planner produced it, so a
        # rules-planner run yields teacher labels for the next VLM retrain.
        self._collector = None
        self._latest_ascii: str = ""
        if os.environ.get("POKEMON_COLLECT") == "1":
            try:
                from pokemon_agent.collect_live import LiveCollector
                self._collector = LiveCollector(
                    every_n_turns=int(
                        os.environ.get("POKEMON_COLLECT_EVERY", "5")))
                logger.info("live dataset collection ON (every %s turns)",
                            os.environ.get("POKEMON_COLLECT_EVERY", "5"))
            except Exception:
                logger.exception("live collector init failed - disabled")

        # --- waypoint cache ---
        self._waypoint_path: List[str] = []
        self._last_xy: Optional[Tuple[int, int]] = None
        self._waypoint_map: str = ""
        # The COMMITTED target cell. Held until reached or unreachable so the
        # tier that picked it cannot flip-flop on the next turn.
        self._waypoint_target: Optional[Tuple[int, int]] = None
        self._waypoint_dest: str = ""
        # Stall tracking for empirical map correction
        self._last_wp_pos: Optional[Tuple[int, int]] = None
        self._wp_stall: int = 0
        # Where we landed on the current map (avoid re-using the stair we just
        # came through - see _resolve_route tier 1)
        self._arrival_map: str = ""
        self._arrival_xy: Optional[Tuple[int, int]] = None
        # (map, cell) targets the agent stalled on - avoid re-committing to
        # them. Cleared per map. Self-healing, unlike editing map_walk.
        self._failed_targets: set = set()
        self._failed_at: dict = {}   # target -> turn added
        self._walk_edge_cells: set = set()
        self._visited_heals: set = set()
        # Map we arrived on this one from (for LAST_MAP warp resolution).
        self.arrived_from_map: Optional[str] = None
        # Latched retreat: once a heal reason fires it stays on until the
        # party is actually healed - prevents heal/exploration flapping.
        self._heal_latched: bool = False
        self.active_plan: str = ""

        self.laya_router: Optional[Any] = None
        self._load_special_tiles()

    def _needs_heal(self, state: Dict[str, Any]) -> str:
        """Why we should go to a Pokecenter, or "" when the party is fine.

        Sources, all from the PARTY struct (which carries HP and PP outside
        battle - `active_mon.moves` is empty out of battle, which is what
        made the old PP check fire on every healthy turn; see the long note
        this replaces).

        Retreat policy (Vlad): with zero damaging PP the smart play is to
        RUN from wilds and keep moving; but a TRAINER battle cannot be fled
        and Struggle costs our own HP. So "cannot damage anything AND a
        trainer could pin us" => retreat before attrition wins. HP low is
        the other hard trigger.

        LATCHING: once a heal reason fires, it stays latched until the party
        is actually healed (a full-HP lead). Without the latch the stall
        cooldown (10 stalled turns -> 40 turns of exploration) alternates
        heal-routes with normal goals, and the normal goals walk straight
        back INTO the cave we are trying to leave - the 1F<->Route 4 flapping
        seen live. While latched the heal watchdog may still stand down, but
        the planner goal is overridden for those turns (see _rules_goal).
        """
        party = [p for p in (state.get("party") or []) if isinstance(p, dict)]
        for p in party:
            if (p.get("hp") or 0) <= 0:
                self._heal_latched = True
                return "a party member has fainted"
        am = state.get("active_mon") or {}
        hp, mx = am.get("hp"), am.get("max_hp")
        lead = party[0] if party else {}
        lead_full = bool(lead) and (lead.get("hp") == lead.get("max_hp"))
        if hp is not None and mx and (hp / mx) < 0.7:
            self._heal_latched = True
            return f"HP is low ({hp}/{mx})"
        # Damaging-PP check from the PARTY struct. If the lead has no
        # damaging move with PP and no reliable flee (trainer risk ahead),
        # the run is on borrowed time - go heal while walking back is cheap.
        if party:
            from pokemon_agent.gen1_data import MOVE_TYPE_POWER
            damaging = any(
                (mv.get("pp", 0) or 0) > 0
                and MOVE_TYPE_POWER.get(str(mv.get("name", "")).lower(),
                                        (None, 0))[1] > 0
                for mv in (lead.get("moves") or []) if isinstance(mv, dict))
            if not damaging:
                # Latch. The latch clears only when the lead is at full HP
                # (i.e. the nurse did her job) AND at least one damaging
                # move has PP again - the heal restores both.
                if not lead_full:
                    self._heal_latched = True
                    return ("no damaging move has PP left - Struggle recoil "
                            "will grind us down in trainer battles")
                self._heal_latched = False
                return ""
        if getattr(self, "_heal_latched", False) and lead_full:
            self._heal_latched = False
        return ""


    def _load_special_tiles(self):
        """Load special tile knowledge from special_tiles.json if present."""
        try:
            with open('special_tiles.json', 'r') as f:
                data = json.load(f)
            # Convert nested dict to flat dict with keys (map_id, x, y)
            for map_str, inner in data.items():
                map_id = int(map_str)
                for x_str, ydict in inner.items():
                    x = int(x_str)
                    for y_str, purpose in ydict.items():
                        y = int(y_str)
                        self.special_tiles[(map_id, x, y)] = purpose
            logger.info(f"Loaded special tiles: {len(self.special_tiles)} entries")
        except FileNotFoundError:
            logger.info("No special_tiles.json found - continuing without semantic tile knowledge.")
        except Exception as e:
            logger.warning(f"Failed to load special_tiles.json: {e}")

    # --- LEARNED HEAL-SITE TABLE ----------------------------------------------
    # File: ~/.pokemon-agent/heal_sites.json
    # {map_name: {"status": "good"|"bad", "attempts": int, "last_ok": int, "last_fail": int}}
    # Pewter Pokecenter is seeded GOOD (we healed there successfully).
    # On successful heal (HP actually increased): mark GOOD, reset attempts.
    # On failed _heal_talk (returns False): increment attempts; >=3 -> BAD.
    # When choosing a heal destination: prefer GOOD centers; if current
    # target is BAD, pick the next-nearest GOOD center.
    HEAL_SITES_PATH = os.path.expanduser("~/.pokemon-agent/heal_sites.json")
    _HEAL_SITE_ATTEMPT_LIMIT = 3

    def _load_heal_sites(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.HEAL_SITES_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_heal_sites(self, data: Dict[str, Dict[str, Any]]) -> None:
        try:
            with open(self.HEAL_SITES_PATH, "w") as f:
                json.dump(data, f, indent=1)
        except Exception:
            logger.exception("heal_sites: failed to save")

    def _mark_heal_site(self, map_name: str, ok: bool) -> None:
        sites = self._load_heal_sites()
        rec = sites.get(map_name, {"status": "unknown", "attempts": 0,
                                    "last_ok": 0, "last_fail": 0})
        if ok:
            rec["status"] = "good"
            rec["attempts"] = 0
            rec["last_ok"] = self.turn
        else:
            rec["attempts"] = rec.get("attempts", 0) + 1
            rec["last_fail"] = self.turn
            if rec["attempts"] >= self._HEAL_SITE_ATTEMPT_LIMIT:
                rec["status"] = "bad"
        sites[map_name] = rec
        self._save_heal_sites(sites)

    def _best_heal_site(self, cur_map: str, state: Dict[str, Any]) -> Optional[str]:
        """Nearest GOOD heal site, or None if none known."""
        if self.map_graph is None:
            return None
        sites = self._load_heal_sites()
        good = {n for n, r in sites.items() if r.get("status") == "good"}
        if not good:
            return None
        mg = self.map_graph
        start = mg.resolve(cur_map)
        if not start:
            return None
        best = None
        for city in good:
            target = mg.resolve(city)
            if not target:
                continue
            seen, frontier, hops, found = {start}, [start], 0, None
            while frontier and hops <= 8:
                if target in frontier:
                    found = hops
                    break
                nxt = []
                for node in frontier:
                    for nb in (mg.neighbors(node) or {}):
                        if nb not in seen:
                            seen.add(nb)
                            nxt.append(nb)
                frontier, hops = nxt, hops + 1
            if found is not None and (best is None or found < best[0]):
                best = (found, city)
            elif found is None:
                # No path from current map to this heal site - skip it
                continue
        return best[1] if best else None

    def _load_explore_union(self):
        """Reload union explored-cells JSON exported by the SB3 training env.

        Merges into self.visits with a minimum count of 1 so the frontier
        treats RL-explored cells as known terrain. File may not exist yet -
        that is fine, we just keep what we have.

        Union cells are GLOBAL (y, x) on the padded grid; the autopilot
        frontier works in per-map LOCAL coords. They are stored in BOTH
        forms: self.visits under the map=-1 sentinel (audit trail) and
        self.union_global as a plain (y, x) set that _frontier_targets
        checks after converting local->global via the map offset table.
        """
        # Per-rank files from the 12 parallel SB3 envs (plus the legacy single
        # file). Each env writes its own map; the union is the merge of all.
        import glob as _glob
        paths = sorted(_glob.glob(
            "/home/vlad/pokemon-agent/explore_map_union_rank*.json"))
        if not paths:
            paths = [self.explore_union_path]   # legacy fallback
        loaded_any = False
        max_step = 0
        for path in paths:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
                continue
            loaded_any = True
            max_step = max(max_step, int(data.get("updated_at_step", 0)))
            before = len(self.union_global)
            for y, x in data.get("cells", []):
                # Union cells are global coords; store both forms:
                # (-1, x, y) in visits (audit) and (y, x) in union_global
                # (frontier check after local->global conversion).
                self.union_global.add((int(y), int(x)))
                key = (-1, int(x), int(y))
                if self.visits.get(key, 0) == 0:
                    self.visits[key] = 1
            added_global = len(self.union_global) - before
            if added_global:
                logger.info("Loaded %d new cells from %s", added_global,
                            os.path.basename(path))
        if loaded_any:
            logger.info("SB3 union: %d global cells total (step %s) from "
                        "%d file(s)", len(self.union_global), max_step, len(paths))


    def _load_map_offsets(self):
        """Load map_name -> (mx, my) from the SB3 training env's map_data.json.

        Needed to convert the frontier's per-map local cells to the global
        coords the SB3 union map uses, so RL-explored terrain actually
        suppresses Laya's frontier (previously loaded but inert).
        """
        try:
            with open("/home/vlad/pokemon-agent/v2/map_data.json", "r") as f:
                regions = json.load(f).get("regions", [])
            for e in regions:
                name = e.get("name")
                coords = e.get("coordinates") or [0, 0]
                if name:
                    self.map_offsets[name] = (int(coords[0]), int(coords[1]))
            logger.info("Loaded %d map offsets from map_data.json",
                        len(self.map_offsets))
        except Exception as e:
            logger.warning(f"Failed to load map offsets: {e}")

    # --- server helpers ----------------------------------------------------



    def _post_stats(self) -> None:
        try:
            self._post("/stats", {
                "hermes_time_total": self.hermes_time_total,
                "laya_time_total": self.laya_time_total,
                "turn": self.turn,
                "mode": self.mode,
                "session_started_at": self.session_started_at,
                "hermes_budget": self.hermes_budget,
            })
        except Exception:
            pass  # best effort


    def _get(self, path: str):
        r = requests.get(self.server + path, timeout=15)
        r.raise_for_status()
        return r


    def _post(self, path: str, payload: dict, timeout: int = 15):
        r = requests.post(self.server + path, json=payload, timeout=timeout)
        r.raise_for_status()
        return r


    def control_state(self) -> str:
        try:
            return self._get("/control").json().get("state", "stopped")
        except Exception:
            return "stopped"


    def emulator_state(self) -> str:
        """idle | booting | ready | error | unknown."""
        try:
            return self._get("/health").json().get("emulator_state", "unknown")
        except Exception:
            return "unknown"


    def _apply_live_window(self, cur_map: str, grid: Dict[Tuple[int, int], bool],
                           pos: Dict[str, Any],
                           live_warps: set) -> Dict[Tuple[int, int], bool]:
        """Override the planning grid's near field with the emulator's live
        collision window (/map/ascii).

        The fold merges three imperfect sources (the pokered decode, the
        static corrections, the seen windows), so it can claim corridors
        that do not exist in this ROM - the BFS then confidently routes
        into a wall and the wall learner pays for the map's lie one bump
        at a time (B2F (32,11): hours). The live window is small (10x10)
        but TRUE, so the path's first steps become game-truth and phantom
        shortcuts near the agent stop wasting turns.

        Guard: when the window shows exit (D) cells, at least one must
        coincide with a live warp cell before the override is trusted -
        the window can render offset (the (24,25) pocket lesson).
        """
        try:
            import re as _re
            r = self._get("/map/ascii")
            data = r.json() if hasattr(r, "json") else None
            if not isinstance(data, dict):
                return grid
            txt = None
            for v in data.values():
                if isinstance(v, str) and "@" in v:
                    txt = v
                    break
            if not txt:
                return grid
            rows = []
            agent_col = agent_row = None
            for line in txt.splitlines():
                parts = line.split()
                if not parts or not parts[0].isdigit():
                    continue
                ry = int(parts[0]) - 1
                cells = parts[1:]
                for ci, ch in enumerate(cells):
                    if ch == "@":
                        agent_col, agent_row = ci, ry
                rows.append((ry, cells))
            if agent_col is None or not rows:
                return grid
            sx, sy = pos.get("x"), pos.get("y")
            if sx is None or sy is None:
                return grid
            x0 = sx - agent_col
            y0 = sy - agent_row
            # CENTERING GUARD: if the window shows D cells, at least one
            # must land on a live warp cell, else the rendering is offset
            # and the override would poison the near field.
            d_cells = [(x0 + ci, y0 + ry) for ry, cells in rows
                       for ci, ch in enumerate(cells) if ch == "D"]
            if d_cells and not any(c in live_warps for c in d_cells):
                logger.info("live window skipped: D cells %s match no live "
                            "warp (offset window)", d_cells[:3])
                return grid
            g2 = dict(grid)
            # PIN IMMUNITY (proven live 19:31): the reader's window can
            # transiently report '.' for cells the wall learner already
            # proved blocked by physical bumps - overwriting those pins
            # re-opened (32,12)/(33,12) and the BFS curled paths back east
            # (the batch literally contained walk_right after walk_left).
            # A bump-confirmed wall is the strongest evidence that exists;
            # the live window may CLOSE cells but never re-open a pin.
            # CLOSINGS ONLY (proven live 19:49): the window's OPENINGS are
            # the unreliable half - the reader transiently reports '.' for
            # real walls, and opening a phantom cell makes the BFS emit
            # left-then-right curls that net-move and so escape the wall
            # learner entirely ((32,11) batches: walk_left,down,down,down,
            # walk_right,down for 20 minutes). The window may ADD walls
            # (a '#' costs at most a detour) but may NEVER remove one, and
            # bump-pinned cells are immune either way.
            for ry, cells in rows:
                for ci, ch in enumerate(cells):
                    cx, cy = x0 + ci, y0 + ry
                    if ch in "#N":
                        g2[(cx, cy)] = False
            return g2
        except Exception:
            return grid


    def act(self, actions: List[str]) -> bool:
        """Send actions to the EMULATOR. /event is narration only and moves
        nothing - this is the endpoint that actually presses buttons."""
        if not actions:
            return False
        try:
            self._post("/action", {"actions": actions}, timeout=90)
            return True
        except Exception as exc:
            body = getattr(getattr(exc, "response", None), "text", "")
            logger.warning("action %s failed: %s %s", actions, exc, body[:200])
            return False


    def _select_move(self, idx: int, moves: List[Any]) -> bool:
        """Open the move menu and select slot *idx* (0-based in ``moves``).

        Gen 1's move menu opens on the PREVIOUSLY SELECTED slot and the cursor
        WRAPS, so a fixed "press down * idx" from an unknown start cannot land
        correctly - that wedged the agent on a 0-PP Vine Whip, looping on
        "No PP left for this move!" for 370+ turns.

        It is also not safe to assume the game is on any particular screen when
        we start: a "No PP" dialog from the previous attempt may be showing, or
        the move menu may already be open. So drive it off the observed screen
        text - the loop reads the state, decides what screen we are on, and
        takes one step. Every path is bounded, so it always terminates.
        """
        names = {str((m or {}).get("name", "")).upper()
                 for m in moves if isinstance(m, dict)}
        names.discard("")
        target = idx + 1                     # move menu positions are 1-based

        for _ in range(12):
            try:
                st = self._get("/state").json()
            except Exception:
                return False
            b = st.get("battle") or {}
            if not b.get("in_battle"):
                return False
            text = str((st.get("dialog") or {}).get("text") or "").upper()
            cur = b.get("menu_index")

            if "NO PP" in text:
                # A failed selection left this dialog up; it eats one press.
                if not self.act(["press_a", "wait_20"]):
                    return False
                continue

            on_move_menu = bool(names) and any(n in text for n in names)
            if on_move_menu:
                if cur == target:
                    return self.act(["press_a", "wait_60"])   # confirm
                if not self.act(["press_down", "wait_20"]):
                    return False
                continue

            # Not on the move menu: get to FIGHT and open it.
            if not self.act(["press_b", "wait_30", "press_up", "press_left",
                             "press_a", "wait_30"]):
                return False

        logger.warning("move select: gave up on slot %d of %d", idx, len(moves))
        return False


    def event(self, **kw) -> None:
        """Push narration to the dashboard. Never affects the game."""
        try:
            self._post("/event", kw)
        except Exception:
            pass


    def sync_active_game(self) -> None:
        """Adopt the active game's id and its Hermes brain id.

        This is how 'load game' on the dashboard takes effect: the driver
        resumes the SAME Hermes session that game was played with.
        """
        try:
            cur = self._get("/games/current").json().get("active")
        except Exception:
            cur = None
        if not cur:
            self.game_id = None
            return
        if cur.get("id") != self.game_id:
            self.game_id = cur.get("id")
            self.session_id = cur.get("hermes_session_id")  # None for a new game
            print(f"[driver] active game: {self.game_id} (hermes={self.session_id})")


    def bind_hermes(self) -> None:
        if self.game_id and self.session_id:
            try:
                self._post(f"/games/{self.game_id}/hermes",
                           {"hermes_session_id": self.session_id})
            except Exception as exc:
                logger.warning("failed to bind hermes session: %s", exc)


    def save_game(self) -> None:
        try:
            self._post("/save", {"name": f"turn_{self.turn:06d}"}, timeout=30)
            logger.info("autosaved at turn %d", self.turn)
        except Exception as exc:
            logger.warning("autosave failed: %s", exc)


    def preflight(self) -> bool:
        """Fail loudly at startup rather than silently timing out each turn."""
        if os.environ.get("POKEMON_SKIP_PREFLIGHT"):
            logger.warning("preflight SKIPPED (POKEMON_SKIP_PREFLIGHT set)")
            return True
        if shutil.which("hermes") is None:
            logger.error("`hermes` not found on PATH")
            return False
        cmd = ["hermes", "chat", "-Q", "--yolo"]
        if self.model:
            cmd += ["-m", self.model]
        if self.provider:
            cmd += ["--provider", self.provider]
        # Same reasoning flag as the turn cmd: a thinking model (Qwen3.5)
        # loops on <think> forever and fills the 65k context, which kills the
        # preflight and the whole autopilot before the first turn.
        reasoning = os.environ.get("POKEMON_HERMES_REASONING", "none")
        if reasoning and reasoning != "default":
            cmd += ["--reasoning", reasoning]
        cmd += ["-q", "Reply with exactly: OK"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=300)
        except subprocess.TimeoutExpired:
            logger.error("preflight timed out - model or gateway not responding")
            return False
        blob = (r.stdout or "") + (r.stderr or "")
        bad = ("turned off", "not found", "auth failed", "Primary auth failed")
        if r.returncode != 0 or any(b in blob for b in bad):
            logger.error("preflight failed (rc=%s): %s", r.returncode,
                         blob[-800:].strip())
            return False
        logger.info("preflight OK: %s", (r.stdout or "").strip()[:120])
        return True


    def _sync_objectives(self, state: Dict[str, Any]) -> None:
        """Recompute objective completion from game state.

        Objectives are display-only, but a stale list showing a finished task
        is worse than none - it misleads both the viewer and any LLM that
        reads the dashboard.
        """
        flags = state.get("flags") or {}
        party = state.get("party") or []
        objs = [
            {"tier": "primary",
             "text": "Get a starter Pokémon from Oak's Lab",
             "done": len(party) > 0},
            {"tier": "primary",
             "text": "Deliver Oak's Parcel · get the Pokédex",
             "done": bool(flags.get("has_pokedex"))},
            {"tier": "secondary",
             "text": "Reach Pewter City · Boulder Badge",
             "done": flags.get("badge_count", 0) >= 1},
        ]
        if objs != self._last_objs:
            self._last_objs = None
            try:
                self._post("/objectives", {"objectives": objs})
            except Exception:
                pass

    # --- frames ------------------------------------------------------------


    def _fetch_frame(self, endpoint: str, path: str) -> bool:
        """Download a PNG to *path*. Returns True on success."""
        try:
            shot = self._get(endpoint).content
            if not shot.startswith(b"\x89PNG"):
                raise ValueError(f"not a PNG ({shot[:60]!r})")
            with open(path, "wb") as f:
                f.write(shot)
            return True
        except Exception as exc:
            body = getattr(getattr(exc, "response", None), "text", "")
            logger.warning("screenshot %s failed: %s %s", endpoint, exc, body[:200])
            return False


    def _graph_destination(self, goal: str,
                           exclude: Optional[str] = None) -> Optional[str]:
        """Extract the map a goal names, using the map graph's own keys.

        'Reach Pewter City' -> 'Pewter City'; 'Walk to Oak's Lab in Pallet
        Town' -> 'Pallet Town'. When a goal names several maps - the badge
        hint says "Route from Viridian City: north through Viridian Forest to
        Pewter" - the one right after a goal verb (reach/to/at/enter/...) is
        the destination; the others are waypoints and must not win.

        *exclude*: a map to never return (the map the agent is standing on -
        a destination equal to the current map means "no route" and the agent
        falls back to blind wandering; a stale standing plan naming the
        current map triggered exactly that, proven live on Route 3).
        """
        if not goal or self.map_graph is None:
            return None
        low = goal.lower()
        hits = [n for n in self.map_graph.node_names() if n and n.lower() in low]
        if exclude:
            hits = [n for n in hits
                    if n.lower() != str(exclude).lower()]
        if not hits:
            return None

        # A map whose name is preceded by a goal-intent word is the target.
        intent = re.compile(
            r"(?:reach|reaching|to|toward|towards|at|enter|entering|into|"
            r"visit|arrive|arrive at|head to|walk to|walk into|get to|"
            r"go to|return to)\s+$")
        best: Optional[str] = None
        best_pos = -1
        for n in hits:
            # Use rfind to get the LAST occurrence of this map name,
            # then check if it's preceded by an intent word.
            idx = low.rfind(n.lower())
            prefix = low[:idx]
            if intent.search(prefix) and idx > best_pos:
                best, best_pos = n, idx
        if best is not None:
            return best
        return max(hits, key=len)


    def _edge_dead(self, key: tuple) -> bool:
        """True when *key*'s edge is currently blacklisted (and not expired)."""
        n = self.edge_fails.get(key, 0)
        if n < self.EDGE_FAIL_LIMIT:
            return False
        age = self.turn - self._edge_fail_turn.get(key, 0)
        if age > self.EDGE_FAIL_TTL:
            logger.info("edge %s blacklist expired after %d turns - retrying",
                        key, age)
            self.edge_fails.pop(key, None)
            self._edge_fail_turn.pop(key, None)
            return False
        return True


    def _edge_bump(self, key: tuple) -> int:
        """Record another failure for *key*; returns the new count."""
        n = self.edge_fails.get(key, 0) + 1
        self.edge_fails[key] = n
        self._edge_fail_turn[key] = self.turn
        return n


    def _laya_choose(self, laya_state: Dict[str, Any],
                     criteria: Dict[str, str],
                     instructions: str) -> Optional[str]:
        """Ask the configured scorer (Laya or Von) to pick one key from
        *criteria*. Returns the key or None.

        Laya API:
            router.predict(state: dict, questions: dict) -> dict
            questions = {"action": {"type": "choice",
                                    "instructions": str,
                                    "criteria": {key: description}}}
            result["action"]["choice"] == one of the criteria keys

        Von API:
            client.choice(prompt: str, options: List[str]) ->
                {"choice": int, "scores": List[float]}
        """
        scorer_name = self.scorer_name
        logger.info("options offered: %s", sorted(criteria.keys()))
        if scorer_name == "laya":
            return self._laya_router_choose(laya_state, criteria,
                                            instructions)
        if scorer_name == "llm":
            # Schema-constrained local model: the reply is grammar-limited to
            # the criteria keys for THIS turn, so it cannot emit an option that
            # was not offered (the malformed-JSON retry loop the subprocess
            # path suffered). See pokemon_agent/llm_direct.py.
            return self._llm_choose(criteria, laya_state, instructions)
        # Von or expert - both take a flat prompt + option list.
        if self.scorer is None:
            self.scorer = get_scorer(scorer_name)
            logger.info("scorer initialized: %s", scorer_name)
        prompt = json.dumps(laya_state, sort_keys=True) + "||" + instructions
        options = sorted(criteria)
        try:
            result = self.scorer.choice(prompt, options)
        except Exception as exc:
            if self.consecutive_fail == 0:
                logger.exception("scorer %s failed", scorer_name)
            else:
                logger.error("scorer %s failed: %s", scorer_name, exc)
            return None
        idx = result.get("choice")
        try:
            choice = options[int(idx)]
        except (TypeError, ValueError, IndexError):
            logger.warning("scorer %s returned bad index %r",
                           scorer_name, idx)
            return None
        scores = result.get("scores") or []
        if scores and 0 <= int(idx) < len(scores):
            total = sum(s for s in scores if s > 0) or 1.0
            self.last_confidence = float(scores[int(idx)]) / total
        return choice


    def _llm_choose(self, criteria: Dict[str, str],
                    laya_state: Dict[str, Any],
                    instructions: str) -> Optional[str]:
        """Pick among *criteria* with a grammar-constrained local model.

        The enum in the response schema IS the criteria key set, so the model
        physically cannot return an option that was not offered - the failure
        mode that made the subprocess path retry until it timed out.

        Descriptions are included (not just the keys): the keys are dense
        (route_A9, goto_B4, leave_toward_down) and carry no meaning on their
        own, so a bare enum makes the choice a coin flip.
        """
        from pokemon_agent.llm_direct import constrained_choice

        options = sorted(criteria)
        if not options:
            return None
        if len(options) == 1:
            self.last_confidence = 1.0
            return options[0]
        listing = "\n".join(f"  - {k}: {(criteria.get(k) or '').strip()}"
                            for k in options)
        user = (
            f"{instructions}\n\n"
            f"AVAILABLE ACTIONS (choose exactly one key):\n{listing}\n\n"
            f"Reply with the single best key."
        )
        system = ("You are controlling Pokemon Red. Choose the one best action "
                  "from the allowed list. The 'choice' field must be one of the "
                  "listed keys.")
        choice, conf = constrained_choice(
            options, system, user,
            url=self.llm_url, model=self.llm_model,
            timeout=self.llm_timeout)
        if choice:
            self.last_confidence = conf
        return choice


    def _laya_router_choose(self, laya_state: Dict[str, Any],
                            criteria: Dict[str, str],
                            instructions: str) -> Optional[str]:
        try:
            result = self.laya_router.predict(  # type: ignore[union-attr]
                laya_state,
                {"action": {"type": "choice",
                            "instructions": instructions,
                            "criteria": criteria}},
            )
        except Exception as exc:
            # Full traceback once, then just the message - this fires every
            # turn when the schema is wrong and drowns the log otherwise.
            if self.consecutive_fail == 0:
                logger.exception("laya predict failed")
            else:
                logger.error("laya predict failed: %s", exc)
            return None

        # Response shape: {"answers": {"action": {"choice": ..., "probabilities": {...},
        #                                        "answer_confidence": float}}, ...}
        answers = result.get("answers") if isinstance(result, dict) else None
        ans = (answers or {}).get("action") if isinstance(answers, dict) else None
        if not isinstance(ans, dict):
            logger.warning("unexpected laya response shape: %r", result)
            return None

        choice = ans.get("choice")
        logger.debug("Laya turn {self.turn}: got choice \"{choice}\" from Laya")
        if choice is not None:
            logger.debug("Laya turn {self.turn}: chose action \"{choice}\" from criteria: {list(criteria.keys())}")
            if choice != "back_out":
                logger.debug("Laya turn {self.turn}: executing \"{choice}\" (reason: {criteria.get(choice, \"no reason\")})")
        if choice not in criteria:
            logger.warning("laya returned %r which is not in %s",
                           choice, sorted(criteria))
            return None

        conf = ans.get("answer_confidence")
        if isinstance(conf, (int, float)):
            self.last_confidence = float(conf)
        return choice


    # ================================================================
    # Tier 1 bridge - planner goal + waypoint resolution
    # ================================================================

    def _current_frame_png(self) -> bytes:
        """Raw PNG of the current screen (for the VLM planner)."""
        try:
            shot = self._get("/screenshot").content
            if shot.startswith(b"\x89PNG"):
                return shot
        except Exception as exc:
            logger.warning("screenshot fetch failed: %s", exc)
        return b""

    def _rules_goal(self, state: Dict[str, Any]) -> str:
        flags = state.get("flags") or {}
        party = state.get("party") or []
        hp = [int(m.get("hp") or 0) for m in party]
        mx = [int(m.get("max_hp") or 0) for m in party]
        types = [str(m.get("type1", "")).capitalize()
                 for m in party if m.get("type1")]
        levels = [int(m.get("level") or 0) for m in party]
        # PP EXHAUSTION IS "HURT" (proven live 04:41): a lead with no
        # damaging move left (Ivysaur, Vine Whip 0 PP) cannot win a single
        # fight, so every Mt Moon wild Zubat became a flee-loop - and the
        # heal branch never fired because HP was 67/88. A Pokecenter visit
        # restores PP, so treat "no attack PP" as needing a center.
        no_attack_pp = False
        if party:
            try:
                from pokemon_agent.gen1_data import MOVE_TYPE_POWER
                lead = party[0]
                moves = lead.get("moves") or []
                has = any(int(mv.get("pp") or 0) > 0 and
                          (MOVE_TYPE_POWER.get(str(mv.get("name", "")).lower(),
                                                (None, 0))[1] or 0) > 0
                          for mv in moves)
                no_attack_pp = bool(moves) and not has
            except Exception:
                no_attack_pp = False
        # Bag as {ITEM: count} for ball/potion guards in derive_goal - the
        # catch goals must never fire without balls, else Von hunts grass
        # with an empty bag forever.
        bag_dict: Dict[str, int] = {}
        for e in (state.get("bag") or []):
            name = str(e.get("item", "")).upper()
            if name:
                bag_dict[name] = bag_dict.get(name, 0) + int(e.get("quantity") or 0)
        map_name = (state.get("map") or {}).get("map_name", "")
        # HEAL-SITE HINT (concrete, routable): the rules' plain "Heal the
        # party at the nearest Pokemon Center." names NO map, so
        # _graph_destination returns None and the route falls through to
        # the free-walk explorer. Compute the actual nearest center map
        # and hand it to derive_goal so the heal goal is routable (the
        # PP-exhaustion heal trigger below depends on this).
        heal_sites = None
        try:
            _hd = self._nearest_heal_destination(state, map_name)
            if _hd:
                heal_sites = {_hd: {"status": "good"}}
        except Exception:
            logger.exception("rules goal: heal destination failed")
        # (The Mt Moon half/third suffixing and the crossing goal script are
        # gone: with the ROM decode verified against the game's own RAM grid
        # (100% on every map, 177 real ledge hops on Route 4) the generic
        # goal derivation routes the cave and Route 4 correctly.)
        if False and "Mt Moon" in map_name and "B2F" not in map_name:
            y = ((state.get("player") or {}).get("position") or {}).get("y")
            if y is not None:
                if "B1F" in map_name:
                    # B1F is THREE regions, not two (measured 02:59-04:20):
                    #   north (y0-6)  the NE strip: (23,3) B2F ladder + the
                    #                 (27,3) exit to Route 4 east.
                    #   mid   (y7-14) the (14-18, 8-11) pocket: warps to
                    #                 B2F only; cannot reach (13,27).
                    #   south (y15+)  the (25,15) pocket: reaches (13,27).
                    # The BFS from (14,9) reaches 48 cells and NO exit, so
                    # a two-way split sent every mid-pocket visit to a
                    # goal it could not reach and it flipped back to B2F
                    # forever. The mid rule now descends via (17,11).
                    if y <= 6:
                        map_name = f"{map_name} north"
                    elif y <= 14:
                        map_name = f"{map_name} mid"
                    else:
                        map_name = f"{map_name} south"
                else:
                    map_name = f"{map_name} {'north' if y < 16 else 'south'}"
        _hurt = any(0 < h < m * 0.7 for h, m in zip(hp, mx) if m > 0)

        return derive_goal(
            map_name,
            int(flags.get("badge_count") or 0),
            bool((state.get("battle") or {}).get("in_battle")),
            bool((state.get("dialog") or {}).get("text_active")),
            hp, mx,
            party_types=types,
            party_levels=levels,
            bag_items=bag_dict,
            has_oaks_parcel=bool(flags.get("has_oaks_parcel")),
            oak_got_parcel=bool(flags.get("oak_got_parcel")),
            money=int((state.get("player") or {}).get("money") or 0),
            no_attack_pp=no_attack_pp,
            heal_sites=heal_sites)

    _HEAL_CITIES = ("Viridian City", "Pewter City", "Cerulean City")

    def _nearest_heal_destination(self, state: Dict[str, Any],
                                  cur_map: str) -> str:
        """The concrete map to route to when healing.

        Returns ``"<City> Pokecenter"`` when already inside a city (the
        Center is a map-internal door the router BFSes straight to),
        otherwise the reachable city nearest by graph hops.

        This exists because the ROUTER requires a map NAME in the goal
        text: `_graph_destination` returns None for any goal that does
        not name one. The VLM's vocabulary offers the map-less string
        "Heal the party at the nearest Pokemon Center.", which is
        therefore unroutable - the router returns None, the driver
        falls through to the free-walk explorer, and Von wanders
        (proven live: repeatedly into the Pewter Museum, one tile above
        his restore position, instead of south to the Pokecenter).
        """
        if cur_map and cur_map.endswith(" City"):
            return f"{cur_map.split(' City')[0]} Pokecenter"
        mg = self.map_graph
        if mg is None or not cur_map:
            return "Pewter City"
        start = mg.resolve(cur_map)
        if not start:
            return "Pewter City"
        # CANDIDATES = the heal cities PLUS real Pokecenter MAPS. Caves and
        # mountains have their OWN Center map (Mt Moon Pokecenter sits on
        # the Route 4 side of the cave); searching cities alone sent the
        # agent DEEPER INTO Mt Moon toward Cerulean (2 hops) instead of
        # back out to the Center 2 hops away, and he wedged inside the
        # cave at HP 35/71 (proven live: Mt Moon 1F, 18:21).
        _cands: list = list(self._HEAL_CITIES)
        try:
            for n in mg.node_names():
                low = n.lower()
                if ("pokecenter" in low or "pokemon center" in low) \
                        and n not in _cands:
                    _cands.append(n)
        except Exception:
            logger.exception("heal-destination: center scan failed")
        best = None
        for city in _cands:
            target = mg.resolve(city)
            if not target:
                continue
            seen, frontier, hops, found = {start}, [start], 0, None
            while frontier and hops <= 8:
                if target in frontier:
                    found = hops
                    break
                nxt = []
                for node in frontier:
                    for nb in (mg.neighbors(node) or {}):
                        if nb not in seen:
                            seen.add(nb)
                            nxt.append(nb)
                frontier, hops = nxt, hops + 1
            if found is None:
                continue
            # PREFER AN ACTUAL CENTER on equal hops: an entry tile in a
            # Center map heals; a city still needs an extra walk to its
            # Center door.
            _low = city.lower()
            _is_center = "pokecenter" in _low or "pokemon center" in _low
            score = (found, 0 if _is_center else 1)
            if best is None or score < best[0]:
                best = (score, city)
        return best[1] if best else "Pewter City"

    def _plan_goal_text_offers(self) -> bool:
        """F1 helper: does the current latched goal name a destination?

        True when the planner goal text contains 'walk to <dest>' (the
        destination-guard rewrite) or names a Pokecenter. When True in VLM
        mode, the free-walk single-step producer is suppressed: the
        planner owns WHERE, Von computes HOW (README rule 8).

        'Leave <city> via its <dir> exit to <route>' goals are also
        routable - they name the exit compass, and free-walk single steps
        immediately walked the agent back into the building he just left
        (proven live: Pewter Gym exit at 10:24, walk_up onto the door
        tile 5 seconds after exiting).
        """
        g = (self._plan_goal_text or "").lower()
        if "leave " in g and " exit" in g:
            return True
        return ("walk to " in g) or ("pokecenter" in g)

    def _plan_goal(self, state: Dict[str, Any], cur_map: str) -> str:
        """Tier 1: ONE goal string, cached for plan_interval turns.

        RETREAT OVERRIDE: while the heal latch is on, the normal goal is
        replaced by an explicit "exit toward the Pokecenter" goal. Without
        this the cached normal goal (e.g. "Descend into Mt Moon") keeps
        firing during the heal watchdog's stand-down windows and walks the
        agent back into the cave it is trying to leave (1F<->Route 4
        flapping, proven live).
        """
        if getattr(self, "_heal_latched", False):
            # In VLM mode, let the VLM plan the heal strategy - it can see
            # the screenshot and context. Only use hardcoded fallback if
            # VLM is unavailable or not in VLM mode.
            if self.planner_mode == "vlm" and self._planner is not None:
                # Don't override - let VLM plan the heal strategy
                pass
            else:
                # Name a concrete city: _graph_destination needs a map name in
                # the goal text, else it returns None and routing falls through
                # to frontier wandering (proven live). The city must be the one
                # whose Pokecenter is actually REACHABLE from here - a hardcoded
                # "Cerulean City" sent the Viridian agent north through the
                # Cut-gated Gym/Forest side while its own Pokecenter was next
                # door. Pick the nearest of Viridian/Pewter/Cerulean by BFS hop
                # count over portal edges.
                latch_goal = ("Heal the party at the nearest Pokemon Center - "
                              "walk to Viridian City.")
                try:
                    mg = self.map_graph
                    if mg is not None:
                        start = mg.resolve(cur_map)
                        best = None
                        for city in ("Viridian City", "Pewter City",
                                     "Cerulean City"):
                            target = mg.resolve(city)
                            if not start or not target:
                                continue
                            seen = {start}
                            frontier = [start]
                            hops = 0
                            found = False
                            while frontier and hops <= 8:
                                nxt = []
                                for node in frontier:
                                    if node == target:
                                        found = True
                                        break
                                    for nb in mg.neighbors(node):
                                        if nb not in seen:
                                            seen.add(nb)
                                            nxt.append(nb)
                                if found:
                                    break
                                frontier = nxt
                                hops += 1
                            if found and (best is None or hops < best[0]):
                                best = (hops, city)
                        # HOP-TIE BREAK (pure geometry): the collapsed
                        # graph gives cities on one long route identical
                        # hops (Route 2 serves Pewter AND Viridian). Pick
                        # the tied city whose exit on the CURRENT map is
                        # physically nearest: gt warps, learned portals,
                        # and walk-edge border cells all count as exits,
                        # each scored by its distance to the player
                        # (proven live: from Route 2's north pocket the
                        # Pewter exit sits 9 tiles north while the
                        # Viridian walk-off is 62 tiles south - the
                        # extent/heuristic layers kept picking Viridian).
                        if best is not None:
                            # The player's position for exit-distance
                            # scoring: the collision layer's settled pair
                            # when coherent, else the fused position.
                            _col_p = (state.get("collision") or {})
                            _sett_p = _col_p.get("settled_position") or {}
                            if (_col_p.get("valid") and _sett_p
                                    and _col_p.get("position_coherent")
                                    is False and _sett_p.get("x") is not None):
                                xy = (int(_sett_p["x"]), int(_sett_p["y"]))
                            else:
                                _pp = ((state.get("player") or {})
                                       .get("position") or {})
                                xy = (_pp.get("x") or 0, _pp.get("y") or 0)
                            _tied = []
                            for city in ("Viridian City", "Pewter City",
                                         "Cerulean City"):
                                target = mg.resolve(city)
                                if not start or not target:
                                    continue
                                _h = 0
                                _seen = {start}
                                _frontier = [start]
                                _f = False
                                while _frontier and _h <= 8:
                                    _nxt = []
                                    for node in _frontier:
                                        if node == target:
                                            _f = True
                                            break
                                        for nb in mg.neighbors(node):
                                            if nb not in _seen:
                                                _seen.add(nb)
                                                _nxt.append(nb)
                                    if _f:
                                        break
                                    _frontier = _nxt
                                    _h += 1
                                if _f and _h == best[0]:
                                    _tied.append(city)
                            if len(_tied) > 1:
                                _exits = {}
                                for w in (self.gt_warps.get(cur_map) or []):
                                    _exits.setdefault(
                                        (w["x"], w["y"]), w.get("dest_name"))
                                for info in (self.map_graph.portals.get(
                                        cur_map) or {}).values():
                                    c = self._portal_map_xy(info)
                                    if c is not None:
                                        _exits.setdefault(
                                            c, (info or {}).get(
                                                "dest_map_name"))
                                for we in WALK_EDGES.get(cur_map, []):
                                    _exits.setdefault(
                                        (we["cell"][0], we["cell"][1]),
                                        we.get("dest"))
                                _best_d = None
                                for cell_xy, dest_name in _exits.items():
                                    # NOTE: failed-target status does NOT
                                    # exclude an exit from RANKING - a
                                    # target fails when BFS couldn't path
                                    # to it at the time (poisoned grid
                                    # stretches); the exit itself may be
                                    # the right direction and the BFS at
                                    # offer time still avoids blocked
                                    # cells (proven live: the Route 2
                                    # north walk-off (9,0) was failed
                                    # during the pocket wedge, which kept
                                    # the latch pinned on Viridian).
                                    for city in _tied:
                                        _hops = (0 if dest_name == city else
                                                 self._warp_leads_to(
                                                     cur_map, dest_name,
                                                     city))
                                        if _hops is None:
                                            continue
                                        _d = (_hops * 100
                                              + abs(cell_xy[0] - xy[0])
                                              + abs(cell_xy[1] - xy[1]))
                                        if _best_d is None or _d < _best_d[0]:
                                            _best_d = (_d, city)
                                if _best_d:
                                    logger.info("heal tie-break: exits=%s "
                                                "-> city %s (was %s)",
                                                sorted(_exits.items())[:6],
                                                _best_d[1], best[1])
                                    best = (best[0], _best_d[1])
                        if best:
                            # Already inside the chosen city? Then the
                            # Pokecenter is a MAP-INTERNAL portal: name the
                            # Pokecenter map itself so the router BFSes to its
                            # door instead of "arriving" (the latch city == the
                            # current map resolves to no destination and the
                            # agent frontier-wanders the south-east pocket).
                            if best[1] == cur_map:
                                pc = f"{best[1].split(' City')[0]} Pokecenter"
                                latch_goal = ("Heal the party at the nearest "
                                              f"Pokemon Center - walk to {pc}.")
                            else:
                                latch_goal = ("Heal the party at the nearest "
                                              "Pokemon Center - walk to "
                                              f"{best[1]}.")
                except Exception:
                    logger.exception("RETREAT OVERRIDE city pick failed")
                if (self._plan_goal_text != latch_goal):
                    self._plan_goal_text = latch_goal
                    self._plan_turn = self.turn
                    self._plan_map = cur_map
                    self.active_plan = latch_goal
                    logger.info("planner (RETREAT OVERRIDE): %r (map=%s)",
                                latch_goal, cur_map)
                return latch_goal

        # TERMINATION CHECK: if the cached goal is already COMPLETE in
        # state, drop it and replan NOW - a goal that stays cached after
        # its completion condition fires loops forever (proven live: the
        # starter goal kept firing with Bulbasaur in the party; the
        # catch goal would hunt grass with the target type already
        # owned; the parcel goal would keep returning to the Mart).
        if self._plan_goal_text and self._goal_done(self._plan_goal_text,
                                                    state):
            logger.info("planner: goal COMPLETE - %r (map=%s)",
                        self._plan_goal_text, cur_map)
            self._plan_goal_text = ""
            self._plan_turn = 0
            # REPLAN NOW, not next turn: a cleared goal lets the free-walk
            # fallback act for one turn ungated - which is how the agent
            # walked back into the earned gym between completions
            # (complete -> cleared -> free-walk walk_up -> gym, 10:36).
            # Fall through to the fresh-plan block immediately.

        # When heal is latched, FORCE a fresh VLM plan - the cached goal
        # (e.g. "Progress toward Pewter City") keeps routing us back into
        # the cave we're trying to leave. The VLM must choose the heal
        # strategy (which center, route out).
        if getattr(self, "_heal_latched", False):
            logger.info("planner: heal latched - forcing fresh VLM plan (turn=%d _heal_latched=%s)",
                        self.turn, getattr(self, "_heal_latched", False))
        elif (self._plan_goal_text
                and self.turn - self._plan_turn < self.plan_interval
                and not (self._plan_map and self._plan_map != cur_map
                         and str(self._plan_map).lower()
                         in self._plan_goal_text.lower())):
            # CACHED-GOAL RULE. Default: no map-change clause - _goal_done
            # already drops transition-completed goals, and a goal with no
            # map name (e.g. "Exit this house...") must survive a boundary
            # to avoid the Pewter-Gym enter/exit ping-pong (live 09:27).
            # EXCEPTION: a cached goal that NAMES THE PREVIOUS MAP is
            # invalid on the new map - "Descend from Mt Moon 1F north half
            # ..." planned on 1F resolves its destination to 'Mt Moon 1F'
            # once standing on B1F (B1F excluded), the interior override
            # routes back UP the (5,5) ladder and the agent 5-second-dips
            # 1F<->B1F forever (proven live 03:35). Map-naming goals
            # re-plan immediately on a boundary; map-agnostic ones ride.
            logger.debug("planner: returning cached goal (turn=%d _plan_turn=%d interval=%d)",
                         self.turn, self._plan_turn, self.plan_interval)
            return self._plan_goal_text

        logger.debug("planner: calling VLM/rule_plan (mode=%s _planner=%s available=%s)",
                     self.planner_mode, self._planner is not None,
                     self._planner.available() if self._planner else False)
        # LEARNED HEAL-SITE HINT: expose good/bad centers in state so the
        # VLM can SEE which centers work. Information only - the VLM still
        # picks the goal (which center, or to press on).
        _sites = self._load_heal_sites()
        if _sites:
            state["heal_sites"] = _sites
        goal = ""
        if self.planner_mode == "vlm" and self._planner is not None:
            try:
                if self._planner.available():
                    frame = self._current_frame_png()
                    if frame:
                        self._latest_frame_png = frame
                        # LOCAL MAP: symbolic 9x10 view for grounding -
                        # walls/doors/exits in text the VLM reads natively
                        # (its GB-frame vision is weak). Also recorded by
                        # the collector so v4 trains on the same field.
                        try:
                            _ascii = render_ascii_map(
                                state.get("collision") or {}, legend=False)
                        except Exception:
                            _ascii = ""
                        goal = self._planner.plan(frame, state,
                                                  ascii_map=_ascii or None)
                        if _ascii:
                            self._latest_ascii = _ascii
                        logger.info("planner VLM returned: %r", goal)
                        # Dynamic heal-site goals name a specific center and
                        # are NOT in the training vocab - accept them.
                        if goal not in GOAL_VOCAB and not (
                                goal.startswith("Heal the party at")
                                and "Pokecenter" in goal):
                            logger.warning("planner VLM goal not in vocab: %r", goal)
                            goal = ""
                    else:
                        logger.warning("planner: no frame from _current_frame_png")
                else:
                    logger.warning("planner: VLM not available")
            except Exception:
                logger.exception("VLM planner failed - falling back to rules")
                goal = ""
        if not goal:
            logger.info("planner falling back to rule_plan")
            goal = rule_plan(state)
            logger.info("planner rule_plan returned: %r", goal)
        # Dynamic heal-site goals ("Heal the party at <X> - walk to <X>.")
        # are NOT in GOAL_VOCAB by design - the VLM path accepts them
        # explicitly (above); the rules path must too, or the heal goal is
        # discarded here and _rules_goal re-derives WITHOUT heal_sites,
        # emitting the descend goal instead (proven live 16:17: heal at
        # :23, descend at :35, the agent yo-yoing at the 1F/Route 4
        # boundary instead of walking the 2-hop trip to the center).
        if goal not in GOAL_VOCAB and not (
                goal.startswith("Heal the party at")
                and "Pokecenter" in goal):
            goal = self._rules_goal(state)

        # ------------------------------------------------------------------
        # GOAL DESTINATION GUARD - "the goal must name a map".
        #
        # The ROUTER resolves a goal to a map name (`_graph_destination`);
        # a goal with no map name returns None, `_resolve_route` returns
        # None, no route_to_waypoint is offered, and the driver falls
        # through to the free-walk explorer. That is precisely how Von
        # kept ending up in the Pewter Museum instead of walking south to
        # the Pokecenter: the VLM's vocabulary contains the perfectly
        # reasonable-sounding but UNROUTABLE string
        #
        #     "Heal the party at the nearest Pokemon Center."
        #
        # which `_graph_destination` maps to None (verified: that exact
        # string -> None, while "...walk to Pewter Pokecenter." ->
        # 'Pewter Pokecenter'). The heal latch's RETREAT OVERRIDE would
        # have named a city, but it is skipped in VLM mode ("let the VLM
        # plan the heal strategy") - so in VLM mode nothing supplied a
        # destination at all.
        #
        # Keep the VLM's DECISION (heal vs. do something else) and supply
        # the DESTINATION deterministically: that is the intended split -
        # the VLM steers WHAT, the driver computes HOW.
        # ------------------------------------------------------------------
        if goal and "pokemon center" in goal.lower() \
                and self._graph_destination(goal, cur_map) is None:
            # The VLM's dynamic heal goal may name a center with a
            # non-canonical map name ('Pewter City Pokecenter' vs the real
            # 'Pewter Pokecenter'). Try to normalize BEFORE rewriting to
            # the nearest center - the nearest rewrite is what sent him
            # back into the BAD Mt Moon center (proven live 06:10).
            _norm = None
            _mg = self.map_graph
            if _mg is not None:
                for n in _mg.node_names():
                    if ("pokecenter" in n.lower()
                            and n.split(" Pokecenter")[0].lower()
                            in goal.lower()):
                        _norm = n
                        break
            if _norm:
                logger.info("goal destination guard: normalized center to %s",
                            _norm)
                goal = (f"Heal the party at {_norm} - walk to {_norm}.")
            else:
                _hdest = self._nearest_heal_destination(state, cur_map)
                if _hdest:
                    logger.info("goal destination guard: %r names no map - "
                                "rewriting to walk to %s", goal[:48], _hdest)
                    goal = ("Heal the party at the nearest Pokemon Center - "
                            f"walk to {_hdest}.")

        self._plan_goal_text = goal
        self._plan_turn = self.turn
        self._plan_map = cur_map
        self.active_plan = goal
        self._latest_goal_text = goal
        if goal:
            logger.info("planner: %r (map=%s mode=%s)",
                        goal, cur_map, self.planner_mode)
        return goal

    @staticmethod
    def _goal_done(goal: str, state: Dict[str, Any]) -> bool:
        """State-based completion predicate for the cached strategic goal.

        Mirrors the goal vocabulary: each goal family checks the RAM fact
        that makes it moot. Returns True when the goal is ACHIEVED (replan
        immediately), False while it is still actionable.
        """
        g = (goal or "").lower()
        flags = state.get("flags") or {}
        party = state.get("party") or []
        types = {str(m.get("type1", "")).lower() for m in party}
        bag: Dict[str, int] = {}
        for e in (state.get("bag") or []):
            name = str(e.get("item", "")).upper()
            if name:
                bag[name] = bag.get(name, 0) + int(e.get("quantity") or 0)
        balls = (bag.get("POKE BALL", 0) + bag.get("GREAT BALL", 0)
                 + bag.get("ULTRA BALL", 0))

        # Starter: a non-empty party means the starter is taken.
        if "starter" in g:
            return bool(party)
        # Gym goals: done when the badge count meets the gym ordinal
        # ('first Gym' -> 1). Without this the latched goal kept steering
        # the agent back into the earned gym - the city<->gym bounce of
        # 09:29-09:40 - because the goal never expired (proven live).
        if "gym" in g:
            _flags_gd = state.get("flags") or {}
            _badges_gd = int(_flags_gd.get("badge_count") or 0)
            if "first" in g or "1st" in g:
                return _badges_gd >= 1
            if "second" in g or "2nd" in g:
                return _badges_gd >= 2
            if "third" in g or "3rd" in g:
                return _badges_gd >= 3
            return _badges_gd >= 1
        # Catch goals: done when the party has the requested type.
        if "catch" in g:
            for t in ("electric", "grass", "water", "fire", "psychic",
                      "flying", "bug", "rock", "ground"):
                if t in g and t in types:
                    return True
            # "Water- or Grass-type" phrasing: either satisfies
            if "water- or grass" in g and (types & {"water", "grass"}):
                return True
            if "electric- or grass" in g and (types & {"electric", "grass"}):
                return True
            if "fire/psychic/flying" in g and (types & {"fire", "psychic",
                                                        "flying"}):
                return True
            return False
        # Parcel goals are two-phase:
        #   "Collect Oak's Parcel" -> done when carrying it.
        #   "Return/Deliver Oak's Parcel" -> done only when Oak RECEIVED
        #   it (EVENT_OAK_GOT_PARCEL). Checking has_oaks_parcel marked
        #   the delivery done instantly while the parcel was still in
        #   the bag (proven live: 'goal COMPLETE' fired every turn in
        #   the Pokecenter with the parcel carried).
        if "parcel" in g:
            if "return" in g or "deliver" in g or "oak's lab" in g \
                    or "oaks lab" in g:
                return bool(flags.get("oak_got_parcel"))
            return bool(flags.get("has_oaks_parcel"))
        # Heal: done when party is fully healed (all mons at max HP).
        # The 30% threshold was meant to stop the "walk to center" goal from
        # stalling, but it caused false completion when the nurse was
        # unreachable (Mt Moon Pokecenter - nurse blocked by Team Rocket NPC).
        # Only mark complete when ALL mons are at max HP - the VLM will
        # re-issue heal goals until the party is actually restored.
        if ("heal" in g and ("pokecenter" in g or "pokémon center" in g)) or ("heal" in g and "center" in g and "pokecenter" in g):
            hp = [int(m.get("hp") or 0) for m in party]
            mx = [int(m.get("max_hp") or 0) for m in party]
            return all(h >= m for h, m in zip(hp, mx) if m > 0)
        # Buy goals: done when we have balls (and potions for the mart goal).
        if "buy" in g and "ball" in g:
            return balls > 0
        # Travel-to-city goals: done on arrival. A goal naming the city the
        # player is STANDING IN has no remaining travel leg (any gym intent
        # in the same line is covered by the badge branch above). Without
        # this the v3 planner re-issued 'Travel north toward Pewter City'
        # inside Pewter with the badge held - live 19:11 - and the driver
        # loop-escaped around the museum forever.
        # Shape note: this receives the driver's COMPACT state, where "map"
        # is a plain string and badges is a list of names - handle both.
        m = state.get("map")
        if isinstance(m, dict):
            cur = m.get("map_name") or ""
        else:
            cur = m or ""
        if cur:
            for city in ("Pewter City", "Viridian City", "Cerulean City",
                         "Vermilion City", "Celadon City", "Fuchsia City",
                         "Saffron City", "Cinnabar Island", "Pallet Town",
                         "Lavender Town"):
                if city.lower() in g and cur == city:
                    return True
        return False

    @staticmethod
    def _is_interior(map_name: str) -> bool:
        """True for a building or cave interior (not a city/route/forest)."""
        if not map_name:
            return False
        outdoors = ("City", "Town", "Route", "Forest", "Sea", "League")
        return not any(t in map_name for t in outdoors)

    def _local_action(self, state: Dict[str, Any], cur_map: str,
                      pos: Dict[str, Any], goal: str) -> Optional[Dict[str, str]]:
        """A goal finishable WITHOUT leaving this map.

        The planner says WHAT ("Beat the Gym Leader"); this decides whether
        that is doable here and hands back the single key that does it.
        Keyword-matched against the closed GOAL_VOCAB, so there is no per-map
        `if cur_map == ...`. Heal and battle are handled by their own branches
        before this is reached.
        """
        g = (goal or "").lower()
        if not g:
            return None
        # Gym badge: walking into the leader starts the badge battle.
        # Broadened trigger: the VLM progress goal says '... and the first
        # Gym' - no 'gym leader'/'beat the gym' phrase - so the plain
        # phrase match starved inside the gym and the stale city-door
        # waypoint dragged the agent back outside (bounce door<->gym).
        # Inside a Gym, ANY gym goal means: walk into the leader.
        # BADGE COMPLETION: once the badge is won, the gym goal is DONE -
        # re-offering challenge_leader re-talks the leader forever (his
        # post-battle speech re-opens on contact) and the agent never
        # walks out (proven live: 300+ turns at (3,1) re-talking Brock
        # with the Boulder Badge already in the pocket). Return None so
        # the router paths back out of the gym.
        if "gym" in g and "gym" in cur_map.lower():
            _flags = state.get("flags") or {}
            _badges = int(_flags.get("badge_count") or 0)
            # First gym goal (Pewter) satisfied by any badge; later cities
            # need more - this mapping grows with the goal vocabulary.
            _need = {"Pewter": 1}.get(
                (cur_map or "").replace(" Gym", "").strip(), 1)
            if _badges >= _need:
                logger.info("gym goal COMPLETE: %d badge(s) in %s - "
                            "walking out", _badges, cur_map)
                return None
            return {"challenge_leader":
                    f"Walk into the gym leader to earn the badge ({goal})"}
        if ("gym leader" in g or "beat the gym" in g) and "gym" in cur_map.lower():
            return {"challenge_leader":
                    f"Walk into the gym leader to earn the badge ({goal})"}
        # GYM DOOR ROUTING: a gym goal while in the gym's CITY routes to the
        # gym door via the ordinary waypoint router. The VLM's progress goal
        # ('Progress toward Pewter City and the first Gym') resolves to
        # dest == cur_map - the router then has nothing to offer (already
        # there) and free-walk explores at random instead of walking the
        # 20 tiles to the door. The ground-truth warp table names the door
        # cell exactly (Pewter (16,17), Cerulean (30,19), ...), so route to
        # it the same way the heal branch routes to a Pokecenter.
        if "gym" in g:
            # GLOBAL COMPLETION: the badge gate applies in the city too -
            # with the badge in the pocket, do not route to the gym door
            # (that re-created the city<->gym bounce once the in-gym
            # completion walked him out; proven live 09:29-09:33).
            _flags_g = state.get("flags") or {}
            if int(_flags_g.get("badge_count") or 0) >= 1:
                return None
            _city = re.sub(r"\bthe first\b|\bfirst\b|\bgym\b|\bbadge\b|"
                           r"\bleader\b|\bbeat\b|\bprogress\b|\btoward\b|"
                           r"\band\b|\bthe\b|\bto\b|\bin\b|\bchalleng\w*",
                           "", g).strip()
            _city_map = None
            for known in ("Pewter City", "Cerulean City", "Viridian City",
                          "Vermilion City", "Celadon City", "Fuchsia City",
                          "Saffron City", "Cinnabar Island"):
                if known.split(" City")[0].lower() in g:
                    _city_map = known
                    break
            if _city_map and _city_map == cur_map:
                _door = next(((w["x"], w["y"]) for w in
                              (self.gt_warps.get(cur_map) or [])
                              if "Gym" in str(w.get("dest_name", ""))), None)
                if _door is not None:
                    self._waypoint_target = _door
                    self._waypoint_map = cur_map
                    self._waypoint_dest = _door and f"{_city_map} Gym"
                    self._waypoint_path = []
                    logger.info("gym goal in own city - routing to gym "
                                "door %s", _door)
                    return {"route_to_waypoint":
                            (f"Walk to the gym door to challenge the "
                             f"leader ({goal})")}
        # Starter: the lab table holds three Pokeballs (pokered OaksLab
        # (6,3),(7,3),(8,3) - 1-based = 0-based (5,2),(6,2),(7,2)). Walk to
        # the tile BELOW a ball and press A to take it.
        if "starter" in g and "lab" in cur_map.lower():
            return {"take_starter":
                    f"Walk to the starter table and press A on a Pokeball ({goal})"}
        # Delivery: a named-NPC talk action. GENERALIZED (Vlad): not
        # Oak-specific - any goal that says "deliver/talk to <Name>"
        # resolves to talk_npc_<Name>, and the executor finds the NPC
        # via the live sprite reader (name match), paths to the tile
        # beside it, faces it and presses A. Oak/Gary/clerk/whoever.
        m_npc = re.search(r"(?:deliver (?:the )?parcel to|talk to)"
                          r"\s+(?:prof\.?\s*)?([a-z]+)", g)
        if m_npc and "parcel" in g:
            who = m_npc.group(1).title()
            return {"talk_npc_" + who:
                    f"Walk to {who} and press A to hand over the parcel"}
        # Mart: the goal is to buy supplies at the counter. ROM GATE:
        # while the parcel is carried but undelivered (EVENT_OAK_GOT_
        # PARCEL clear), the Viridian Mart clerk only replays the parcel
        # cutscene - the shop menu never opens. The right action is
        # EXITING the mart to deliver, not shop_buy.
        if "buy supplies" in g:
            if (getattr(self, "_last_flags", {}).get("has_oaks_parcel")
                    and not getattr(self, "_last_flags", {})
                    .get("oak_got_parcel")
                    and "Mart" in cur_map):
                return {"exit_building":
                        "The clerk won't sell until Oak's parcel is "
                        "delivered - exit the Mart and head to Oak's Lab"}
            return {"shop_buy": "Walk to the counter and press A to buy"}
        # Leaving an interior back to the map outside. CAVE FLOORS are
        # excluded ("Mt Moon 1F"/"B1F"/"B2F" - you do not exit a cave by
        # rotating walk_down; you BFS to the next floor's stair) but the
        # Pokecenter/Mart INSIDE a cave ("Mt Moon Pokecenter") is a normal
        # building with a normal exit door. The exclusion is CAVE-NAME
        # based: a plain digit-suffix also matched house floors
        # ("Reds House 1F") and caused an endless 1F<->2F stair ping-pong
        # (the router's only known portal from 1F was the stairs back up).
        if self._is_interior(cur_map) and not re.search(
                r"Mt Moon (B?\dF)|Rock Tunnel|Victory Road", cur_map) and (
                g.startswith("exit") or "leave " in g or "outside" in g):
            return {"exit_building":
                    f"Walk to the door and step outside ({goal})"}
        return None

    @staticmethod
    def _portal_map_xy(info: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Map-space (x, y) of a portal's DOOR tile.

        A persisted portal record stores the player's map position at the
        moment the door was seen (``player_x``/``player_y``) plus the door's
        SCREEN cell (``col``/``row``). The door is not at the player's tile -
        it is offset by (col - PLAYER_COL, row - PLAYER_ROW). Decoding it
        correctly is what puts the agent on the cave mouth instead of on the
        tile where it once stood: all three Mt Moon portals on Route 4
        (F2/G4/J3) resolve to the same door at (18, 5).
        """
        px, py = info.get("player_x"), info.get("player_y")
        col, row = info.get("col"), info.get("row")
        if px is None or py is None:
            return None
        if col is None or row is None:
            return (int(px), int(py))
        return (int(px) + int(col) - PLAYER_COL,
                int(py) + int(row) - PLAYER_ROW)

    @staticmethod
    def _path_target(xy: Tuple[int, int], path: List[str]) -> Tuple[int, int]:
        """The cell a walk-action path ends on."""
        x, y = xy
        for a in path:
            dy, dx = _DIRS.get(a, (0, 0))
            x += dx
            y += dy
        return (x, y)

    def _offer_path(self, path: List[str], cur_map: str, dest: str,
                    how: str) -> Dict[str, str]:
        """Stash a concrete walk path and offer it as the SOLE option."""
        self._waypoint_path = path[:8]
        self._waypoint_map = cur_map
        logger.info("waypoint route: %s -> %s via %s (%d-step path)",
                    cur_map, dest, how, len(path))
        return {"route_to_waypoint":
                f"Follow the pre-computed path {how} out of {cur_map} "
                f"toward {dest} ({len(path)} steps)"}

    def _waypoint_stall_check(self, cur_map: str, xy: Tuple[int, int]) -> None:
        """Stop re-committing to a target we cannot actually reach.

        The BFS plans over the ACCUMULATED grid, which is stitched from
        screens and can therefore contain cells marked walkable that the game
        actually blocks. When the agent stops moving with a target committed,
        record that (map, target) as failed and drop it so the router picks a
        different route.

        Deliberately does NOT edit `map_walk`: an earlier version marked the
        first step unwalkable, and one false positive silently cut the only
        route to the cave mouth (the blocks were in-memory, so the damage
        only showed up live, never in an offline replay from disk). Avoidance
        is self-healing; mutation is not.
        """
        if self._waypoint_target is None:
            self._last_wp_pos = xy
            if self._wp_stall:
                logger.info("stall reset: target is None (was %d)",
                            self._wp_stall)
            self._wp_stall = 0
            return
        if xy == self._last_wp_pos:
            self._wp_stall += 1
        else:
            logger.info("stall reset: xy moved %s -> %s",
                        self._last_wp_pos, xy)
            self._wp_stall = 0
        self._last_wp_pos = xy
        # NET-PROGRESS STALL: oscillation (A<->B, proven live: Viridian
        # City (8,30)<->(12,30) for 2500+ turns) defeats per-turn stall
        # counting because xy 'moves' every other turn. Track distance
        # to the committed target over a 20-turn window: if it hasn't
        # SHRUNK, the target is unreachable from here - drop it and let
        # the frontier explore laterally.
        if self._waypoint_target is not None:
            tgt = self._waypoint_target
            dist = abs(tgt[0] - xy[0]) + abs(tgt[1] - xy[1])
            n = getattr(self, "_wp_progress_n", 0) + 1
            best = getattr(self, "_wp_progress_best", None)
            best = dist if best is None else min(best, dist)
            if n >= 20:
                if best >= dist - 1 and dist > 2:
                    logger.info("net-progress stall: %d turns, best dist "
                                "%d, now %d - dropping target %s",
                                n, best, dist, tgt)
                    self._fail_target(cur_map, tgt)
                    # F3: if this target came from a tier-2 compass walk,
                    # cool that heading so the drop isn't followed by the
                    # identical re-commit (the 20-turn west-loop period).
                    _how3 = getattr(self, "_waypoint_how", "")
                    if "along the stitched map" in _how3:
                        _compass3 = _how3.split(" along")[0].strip()
                        _cd3 = getattr(self, "_compass_cooldown", {}) or {}
                        _cd3[(cur_map, _compass3)] = self.turn + 60
                        self._compass_cooldown = _cd3
                    self._waypoint_target = None
                    self._wp_stall = 0
                    self._wp_progress_n = 0
                    self._wp_progress_best = None
                    return
                self._wp_progress_n = 0
                self._wp_progress_best = None
            else:
                self._wp_progress_n = n
                self._wp_progress_best = best
        if self._wp_stall < 3:
            return
        # The target itself may be fine - the STALL is usually caused by a
        # blocked cell on the path. Blacklist the target only if we are
        # standing next to it (then it is genuinely blocked); otherwise the
        # grid is lying about an intermediate cell in an all-open cavern,
        # and dropping the target just restarts the same plan forever
        # (proven live: heal waypoint (0,35) dropped 3x in a row from the
        # same tile). Instead, walk one step along the path and mark THAT
        # cell as failed so the next BFS routes around it.
        tgt = self._waypoint_target
        if abs(tgt[0] - xy[0]) + abs(tgt[1] - xy[1]) <= 1:
            logger.info("target %s on %s unreachable (adjacent, stalled %d) "
                        "- avoiding it", tgt, cur_map, self._wp_stall)
            self._fail_target(cur_map, tgt)
            self._waypoint_target = None
            self._wp_stall = 0
            return
        # Far from the target: the stall is from a mid-path cell the all-open
        # cavern grid believes walkable. Keep the target. DO NOT reset the
        # counter here: the route_to_waypoint dispatch reads _wp_stall to
        # validate the first step against the LIVE window - resetting before
        # dispatch meant the check could never fire (proven live: 800+ turns
        # at Route 3 (63,0) where (64,0) is live-blocked and every east
        # target's path starts through it). The dispatch check itself resets
        # the counter after blacklisting the blocked cell.

    def _fail_target(self, m: str, c: Tuple[int, int]) -> None:
        """Mark a target failed WITH a turn stamp; entries expire after
        300 turns so verdicts formed under a poisoned/immature grid
        self-heal once the grid matures (proven live: the Pewter PC door
        (13,25) failed during the door-approach era stayed failed after
        the grid healed, starving tier-1 and pinning the agent in the
        west pocket oscillation)."""
        self._failed_targets.add((m, c))
        self._failed_at[(m, c)] = self.turn

    def _failed_now(self, m: str, c: Tuple[int, int]) -> bool:
        """True when (m, c) is on the failed list AND not expired.

        TTL by static verdict: a target the ROM says is walkable failed
        for CIRCUMSTANTIAL reasons (severed path, mid-stitch grid) -
        40 turns, same as the arrival shelf. A target the ROM says is
        blocked failed for REAL reasons - 300 turns (it will re-fail
        anyway if re-tried)."""
        t = self._failed_at.get((m, c))
        if t is None:
            return (m, c) in self._failed_targets
        ttl = 300
        try:
            if self.map_walk.get(m, {}).get(c):
                ttl = 40
        except Exception:
            pass
        if self.turn - t > ttl:
            self._failed_targets.discard((m, c))
            self._failed_at.pop((m, c), None)
            return False
        return True

    def _warp_leads_to(self, cur_map: str, wdest: Optional[str],
                       dest: str) -> Optional[int]:
        """Graph hops from *wdest* to *dest* (0 when equal, None when no
        bounded path). Cands sort by this - the exit whose destination is
        CLOSEST to the goal wins, so in a cyclic graph the north gate
        (1 hop to Route 2 north -> Pewter) outranks the south gate (which
        reaches Pewter only the long way round). None = no credit."""
        if not wdest or not dest or self.map_graph is None:
            return None
        mg = self.map_graph
        start = mg.resolve(wdest)
        target = mg.resolve(dest)
        if not start or not target:
            return None
        # INTERIOR LEAF RULE: buildings (houses, marts, gyms, centers) are
        # leaf nodes - entering one always means leaving by the same door,
        # so a path through an interior to a DIFFERENT map is never on the
        # shortest route. Without this, house doors on the city map qualify
        # as 'next hops toward Pewter' (house->city->Route 2->Pewter, 3
        # hops) and the agent walks into houses instead of out of town
        # (proven live: Viridian City -> Nickname/School House hopping).
        #
        # CORRIDORS STAY ELIGIBLE - and the test is STRUCTURAL, not a name
        # list. A cave floor, gate house or store floor connects >=2 maps,
        # so traffic may legitimately pass THROUGH it; only a genuine
        # one-door leaf is disqualified. The old test used _is_interior()
        # alone, which classified every cave as a building (Mt Moon 1F has
        # no 'City'/'Route'/'Forest' in its name) - so Mt Moon 1F's three
        # ladders to B1F got NO credit toward Cerulean, tier-1 came back
        # with cands=[], and the agent oscillated at the cave mouth
        # (proven live: Mt Moon 1F (14,34)/(14,35), 17:04).
        if start != target and self._is_interior(wdest):
            try:
                _nbrs = len(self.map_graph.neighbors(wdest) or {})
            except Exception:
                _nbrs = 0
            if _nbrs < 2:
                return None
        if start == target:
            return 0
        seen = {start}
        frontier = [start]
        for hops in range(1, 7):
            nxt = []
            for node in frontier:
                for nb in mg.neighbors(node):
                    if nb == target:
                        return hops
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
            if not frontier:
                return None
        return None

    def _resolve_route(self, state: Dict[str, Any], cur_map: str,
                       pos: Dict[str, Any], goal: str) -> Optional[Dict[str, str]]:
        col = (state or {}).get("collision") or {}
        passable = col.get("passable") or col.get("walkable") or []
        """goal -> dest map -> a COMMITTED target cell -> BFS path.

        The target is sticky: once a tier picks a destination cell it is held
        until the agent stands on it or it becomes unreachable. Without this
        the tiers flip-flop (tier 2 says east, tier 3 says unexplored, next
        turn back again) and the agent oscillates even with a single option.

        Tiers, ALL generic - there is no per-map code anywhere:
          1. a LEARNED PORTAL toward the destination (sharper every session);
          2. the stitched-map BFS to the boundary in the compass direction;
          3. the FRONTIER STEP toward unexplored space - which discovers a
             boundary the stitched map has not learned yet (winding maps like
             Route 4's down-then-east). Generic replacement for the old
             hand-targeted "gap cell" hack.
        """
        if self.map_graph is None or not goal:
            return None
        # INSIDE-AN-INTERIOR OVERRIDE: when the agent is in a building and
        # the goal points outside it, the ONLY meaningful first move is the
        # front door. Graph routing across maps ("Pokecenter->Oak's Lab:
        # map not stitched at all") fails without portal records for every
        # intermediate map; gt's LAST_MAP warp always exists. Route to the
        # door; the next map re-resolves the goal from there.
        if self._is_interior(cur_map):
            g_out = goal.lower()
            # HEAL GOALS EXEMPT (17:50, live): the heal machinery routes
            # through the map graph itself (B1F -> (5,5) hole -> 1F ->
            # Pokecenter). The exit-door override hijacked a 14%-HP heal
            # goal to the (27,3) exit - the OPPOSITE direction - while a
            # KO was one Zubat away. A heal goal that names a Pokecenter
            # never takes the exit-door detour.
            if 'pokecenter' in g_out or 'heal' in g_out:
                return None
            _gdest = self._graph_destination(goal, exclude=cur_map)
            if (cur_map not in g_out and _gdest):
                door = None
                # DEST-MATCHED WARP FIRST: when the goal names a specific
                # map that a gt warp on this map leads to (Mt Moon 1F has
                # THREE ladders with dest_name='Mt Moon B1F'), route to
                # that warp. The old LAST_MAP-first pick chose (14,35) -
                # which is the Route 4 EXIT, not a B1F ladder - and the
                # agent walked 58 steps to it and back out of the cave
                # (proven live 22:59-23:02, the ladder churn).
                # GOAL-NAMED LADDER CELL: a goal may carry a "(x,y)" that
                # names WHICH ladder to prefer among several leading to the
                # same map. Mt Moon needs it: B1F has four ladders to B2F
                # and only the (17,11) -> B2F (21,17) -> (15,27) -> B1F
                # (23,3) -> strip -> (27,3) chain is wart-free (every hop
                # a clean step-onto; verified against the live warp table
                # and the fold components, 2026-10-25). Without it the
                # first-match pick laps forever.
                _pref = None
                _m = re.search(r"\((\d{1,3})\s*,\s*(\d{1,3})\)", goal or "")
                if _m:
                    _pref = (int(_m.group(1)), int(_m.group(2)))
                # LIVE WARPS FIRST (proven live 01:41): the gt warp table is
                # the pokered disassembly and does NOT label this ROM's
                # ladders - on Mt Moon 1F it has no (17,11)->B1F entry, so
                # the dest-matched scan found nothing, fell through to the
                # LAST_MAP branch and picked the (14,35) WEST door. The
                # agent then exited west onto the sealed Route 4 pocket
                # instead of descending to cross the cave, forever. The
                # emulator's own warp table (state.collision.warps) has the
                # true dest_map_name per cell - consult it first.
                # An explicitly named goal cell that IS a live warp is
                # authoritative on its own - the dest labels are the
                # unreliable part (pokered vs this ROM).
                if _pref is not None and door is None:
                    for _w in ((state or {}).get("collision") or {}).get(
                            "warps", []):
                        if (_w.get("map_x") is not None
                                and _w.get("map_y") is not None
                                and (int(_w["map_x"]), int(_w["map_y"]))
                                == _pref):
                            door = _pref
                            logger.info("interior override: goal-named "
                                        "ladder %s is a live warp - "
                                        "using it", _pref)
                            break
                for _w in ((state or {}).get("collision") or {}).get(
                        "warps", []):
                    _wx, _wy = _w.get("map_x"), _w.get("map_y")
                    _wd = str(_w.get("dest_map_name") or "")
                    if _wx is None or _wy is None or not _wd or not _gdest:
                        continue
                    if _wd.lower() != str(_gdest).lower():
                        continue
                    if _pref is not None and (int(_wx), int(_wy)) == _pref:
                        door = _pref
                        break
                    if door is None:
                        door = (int(_wx), int(_wy))
                for w in (self.gt_warps.get(cur_map) or []):
                    if w.get("dest_name") and _gdest and \
                            str(w["dest_name"]).lower() == str(_gdest).lower():
                        if _pref is not None and (w["x"], w["y"]) == _pref:
                            door = _pref
                            break
                        if door is None:
                            door = (w["x"], w["y"])
                if door is None:
                    for w in (self.gt_warps.get(cur_map) or []):
                        if w.get("dest_name") is None:  # LAST_MAP = front door
                            door = (w["x"], w["y"])
                            break
                if door:
                    logger.info("interior override: %s goal needs the exit "
                                "door first -> %s", cur_map, door)
                    # COMMIT the door as the sticky target: without this
                    # the override re-BFSes every turn and the agent
                    # oscillates between equal-cost route candidates
                    # (proven live: Viridian Pokecenter (9,4)<->(13,4)
                    # for 2h). The sticky block below holds the target
                    # until the agent stands on the door tile.
                    self._waypoint_target = door
                    self._waypoint_map = cur_map
                    self._waypoint_dest = self._graph_destination(
                        goal, exclude=cur_map) or goal
                    self._wp_stall = 0
                    # NO pathing here - mw/warp_cells don't exist yet at
                    # this point in the function. The sticky block below
                    # (after the interior fold) re-BFSes to the committed
                    # door and offers the path.
        dest = self._graph_destination(goal, exclude=cur_map)
        if not dest:
            return None

        mw = self.map_walk.get(cur_map) or {}
        # INTERIOR GEOMETRY: this ROM's building interiors do NOT match
        # pokered's (proven live: Viridian Pokecenter - the ROM is a wider
        # room with the counter east; gt says 14x8 with a west aisle). The
        # LIVE collision window is the only authority inside buildings, so
        # fold the current screen into the BFS grid before pathing: gt
        # walls that the live reader calls open yield to the live reading
        # (never the reverse - the reader is never more permissive than
        # the game).
        if col.get("valid") and passable:
            # FOLD FOR ALL MAPS (was interior-only): the live window is the
            # only reliable geometry source everywhere. Outdoor maps have
            # the same coordinate-skew risk at edges (proven live: Viridian
            # City south fence read as solid wall while the connection
            # corridor exists). Fold open readings unconditionally; blocked
            # readings need 2 consecutive strikes (flicker guard).
            px_i, py_i = (pos.get("x") or 0), (pos.get("y") or 0)
            # Flicker guard: some interior cells read blocked/open
            # alternately (reader flakiness near counters - proven live:
            # Viridian Pokecenter (5,5) flips, bouncing the exit path).
            # A blocked reading needs 2 consecutive observations of the
            # same cell before the fold records it as a wall; open
            # readings clear the strike counter.
            pending = getattr(self, "_fold_pending", {})
            for r_i, row_i in enumerate(passable):
                for c_i, ok_i in enumerate(row_i):
                    mx_i = px_i + (c_i - PLAYER_COL)
                    my_i = py_i + (r_i - PLAYER_ROW)
                    if ok_i:
                        mw[mx_i, my_i] = True
                        pending.pop((mx_i, my_i), None)
                    else:
                        n = pending.get((mx_i, my_i), 0) + 1
                        pending[(mx_i, my_i)] = n
                        if n >= 2:
                            mw[mx_i, my_i] = False
            self._fold_pending = pending
        # COHERENT-ROUTE GATE: prefer the collision layer's settled
        # wXCoord/wYCoord when it disagrees with the fused position - a
        # mid-step race makes the fused value the destination while the
        # window (and the grid BFS plans over) still centers on the
        # source tile. Planning from the settled pair keeps BFS start and
        # the grid in the same instant.
        _col = (state.get("collision") or {})
        _settled = _col.get("settled_position") or {}
        if (_col.get("valid") and _settled
                and _col.get("position_coherent") is False
                and _settled.get("x") is not None):
            xy = (int(_settled["x"]), int(_settled["y"]))
        else:
            xy = (pos.get("x") or 0, pos.get("y") or 0)
        ledges = self.ledge_edges.get(cur_map, {})
        # Warp cells on this map (ground truth + live), used to keep BFS
        # paths from walking ACROSS unrelated doors/stairs.
        warp_cells = {(w["x"], w["y"])
                      for w in (self.gt_warps.get(cur_map) or [])}
        warp_cells.update((int(w.get("map_x")), int(w.get("map_y")))
                          for w in ((state or {}).get("collision") or {})
                          .get("warps", [])
                          if w.get("map_x") is not None)

        # Learn from being stuck: a committed target we cannot reach means the
        # stitched map lies about a cell. Fix it, then re-plan.
        self._waypoint_stall_check(cur_map, xy)

        # --- sticky: hold the committed target ---------------------------
        # Cells proven live-blocked on this map: every BFS (commit-side and
        # dispatch-side) must route around them, else the committed path
        # goes through a blocked cell and the dispatch truncates to zero
        # every turn (proven live: Route 3 (3,8), the 66-step 'east' path
        # started through blacklisted (4,8) and never dispatched a step).
        # BFS AVOID SET: physical evidence only. Failed-target verdicts
        # (path-severed-under-an-immature-grid) stay OUT of pathing - the
        # ROM-native static grid is the authority for what is walkable;
        # failed targets only gate target SELECTION (anti-thrash).
        failed_cells: set = set()
        # F4 (stand-on-tile authority), router side: same rule as the
        # dispatch - the player's own cell is the strongest walkability
        # evidence there is. A fold artifact reading the standing tile as
        # blocked must not sever every BFS in _resolve_route the way it
        # severed the dispatch-side BFS. Force it walkable in a BFS COPY.
        _mw_bfs = mw
        if xy and not mw.get(xy):
            _mw_bfs = dict(mw)
            _mw_bfs[xy] = True
            logger.info("router start-cell authority: standing tile %s "
                        "reads blocked - forcing walkable for BFS", xy)

        # WALK-EDGE OVERRIDE: standing ON a map-connection border cell means
        # the very next action is the OFF-MAP step - a committed waypoint
        # must not short-circuit it. Without this the border commit fires
        # ONCE and then the sticky target ping-pongs the agent either side
        # of the border forever: Route 3 (62,0) <-> (63,0) at 17:24, after a
        # perfectly good Route 4 crossing at 16:56. Clearing the waypoint
        # lets tier-1 re-commit the off-map step (the walk-edge branch).
        if xy is not None and any(
                (we.get("cell") and (we["cell"][0], we["cell"][1]) == xy)
                for we in WALK_EDGES.get(cur_map, [])):
            if self._waypoint_target is not None:
                logger.info("walk-edge override: standing on a map-connection "
                            "border cell %s - clearing the committed waypoint "
                            "(%s) so the off-map step can run",
                            xy, self._waypoint_target)
            self._waypoint_target = None
        if (self._waypoint_target is not None
                and self._waypoint_map == cur_map
                and self._waypoint_dest == dest):
            if xy == self._waypoint_target:
                # ARRIVED. Park the target on the recent-arrivals shelf: a
                # target we just reached must not be re-picked as a NEW
                # destination on the next re-plan, or the agent ping-pongs
                # (proven live: Viridian Forest (16,1) re-committed 544
                # times - reach, re-pick, walk away, walk back, forever).
                # Shelf entries EXPIRE (frontier needs to reuse corridor
                # cells as targets once the maze is mapped); only the
                # immediate re-pick is suppressed. This is target-selection
                # only: the cell stays walkable for BFS paths.
                _how = getattr(self, "_waypoint_how", "")
                if "along the stitched map" in _how:
                    # DEAD-END COMPASS END: a tier-2 edge-path target that
                    # we REACHED and that is not a portal/walk-edge is a
                    # corner of the map, not an exit. A 40-turn shelf only
                    # delays the re-pick (proven live: the Forest's NE
                    # corner (31,1) cycled arrive/shelve/expire/re-commit
                    # forever). Fail it permanently - the compass end
                    # cannot become an exit later.
                    self._failed_targets.add(
                        (cur_map, self._waypoint_target))
                    # F3: cool the compass DOWN for 60 turns so the drop
                    # is not immediately followed by the identical commit.
                    _compass = _how.split(" along")[0].strip()
                    _cd = getattr(self, "_compass_cooldown", {}) or {}
                    _cd[(cur_map, _compass)] = self.turn + 60
                    self._compass_cooldown = _cd
                else:
                    shelf = getattr(self, "_arrived_targets", {})
                    shelf[(cur_map, self._waypoint_target)] = self.turn
                    self._arrived_targets = shelf
                self._waypoint_target = None            # arrived
            else:
                path = _map_step_toward(_mw_bfs, xy, self._waypoint_target,
                                        limit=400, ledge_edges=ledges,
                                        avoid_warps=warp_cells,
                                        avoid_cells=failed_cells,
                                        map_name=cur_map)
                if path:
                    return self._offer_path(path, cur_map, dest,
                                            "to the committed target")
                # The sticky BFS failed - but the live-window dispatch check
                # may be about to blacklist a blocked cell (its stall counter
                # needs a target to count against). KEEP the target and fall
                # through: the dispatch re-BFS + live check is the authority.
                # Nulling here reset the stall counter every turn, so the
                # blocked cell was never blacklisted (proven live: Route 3
                # (63,0), 600+ turns replaying a stale 6-step path whose
                # first step is live-blocked).

        # --- pick a target -------------------------------------------------
        target: Optional[Tuple[int, int]] = None
        how = ""
        path: List[str] = []

        # Remember where we LANDED on this map. Mt Moon 1F's stairs all lead
        # to B1F, so once we descend, the portal nearest our arrival point is
        # the one we just came through - re-using it yo-yos between floors.
        # Also record WHICH map we arrived from, for LAST_MAP warp resolution.
        if self._arrival_map != cur_map:
            self._arrival_map = cur_map
            self._arrival_xy = xy
            self.arrived_from_map = self.prev_map
            # A new map means the old map's unreachable-target verdicts say
            # nothing about this one.
            self._failed_targets = {
                (m, c) for (m, c) in self._failed_targets if m == cur_map}
            # ANTI-RE-ENTRY: park the door/walk-off cell we emerged from
            # on the arrivals shelf - the free-walk scorer then refuses
            # to step back through it for 40 turns (proven live: Von
            # exited the Pewter Museum and walked straight back in
            # through the same door twice).
            shelf = getattr(self, "_arrived_targets", {})
            shelf[(cur_map, xy)] = self.turn
            self._arrived_targets = shelf

        # tier 1: a learned portal that leads toward dest, preferring one we
        # did NOT just walk through. Sources: the pokered ground-truth warp
        # table (exact, known on run 1) plus any portals learned live.
        # LAST_MAP warps (dest None) resolve from the arrival record: they
        # lead back where we came from, so only offer them when dest is the
        # map we entered this one from.
        # NEXT-HOP MATCHING: a warp qualifies when its destination IS the
        # goal OR lies on the map-graph path TOWARD the goal. In a winding
        # maze the exit toward Pewter is a GATE, not Pewter itself - exact
        # matching starved tier-1 and the compass edge path kept picking
        # dead-end corridor ends instead (proven live: Viridian Forest, the
        # North Gate doors (1,0)/(2,0) sit 17 tiles west through the maze
        # while tier-2 pinned (16,1)). Cands carry their hop count so the
        # closest-to-goal exit wins the sort (the cyclic graph would
        # otherwise rank both forest gates equal and yo-yo between them).
        cands: List[Tuple[int, int]] = []
        cand_hops: Dict[Tuple[int, int], int] = {}
        # NO-IMMEDIATE-BACKTRACK GUARD (proven live 01:09): a portal that
        # leads back to the map we JUST came from is not a route. On
        # Route 4 the only portal is the Mt Moon cave mouth, so tier-1
        # re-entered the cave 2 seconds after every crossing (23:48 and
        # 01:09 - the whole cave traverse undone each time). Applies for
        # ~40 turns after a map change only, so a genuine dead-end
        # backtrack is still allowed later.
        _block_prev = None
        if (getattr(self, "prev_map", None)
                and (getattr(self, "turn", 0)
                     - getattr(self, "_prev_map_turn", -9999)) <= 40):
            _block_prev = str(self.prev_map).lower()
        for w in (self.gt_warps.get(cur_map) or []):
            wdest = w.get("dest_name")
            if _block_prev and wdest and str(wdest).lower() == _block_prev:
                continue
            if wdest is None:
                # LAST_MAP: resolve the dest by REVERSE WARP PAIRING -
                # pokered pairs a warp with the dest map's same-idx warp.
                # Any map whose same-idx warp points BACK at cur_map is
                # the other end of this door. Covers gates (the Forest
                # North Gate's (4,0)/(5,0) idx 2 pairs with Route 2's
                # (3,11) idx 2) and interiors (the Pewter Museum's
                # (10,7) idx 1 pairs with Pewter City's (14,7) idx 1)
                # without special-casing either - the museum exit was
                # invisible under the arrived_from/other-neighbor rule
                # when the museum was entered from the city.
                _pair_maps = []
                for om, ows in (self.gt_warps or {}).items():
                    if om == cur_map:
                        continue
                    for w2 in ows:
                        if (w2.get("idx") == w.get("idx")
                                and w2.get("dest_name") == cur_map):
                            _pair_maps.append(om)
                            break
                _adj = (self.map_graph.neighbors(
                    self.map_graph.resolve(cur_map) or "")
                    if self.map_graph else {})
                for _pm in _pair_maps:
                    if _pm not in _adj:
                        continue
                    if _block_prev and str(_pm).lower() == _block_prev:
                        continue
                    _hops = self._warp_leads_to(cur_map, _pm, dest)
                    if _hops is None:
                        continue
                    c = (w["x"], w["y"])
                    if not self._failed_now(cur_map, c) \
                            and _in_bounds(cur_map, c):
                        cands.append(c)
                        cand_hops[c] = _hops
                continue
            hops_w = (0 if wdest == dest
                      else self._warp_leads_to(cur_map, wdest, dest))
            if hops_w is None:
                continue
            c = (w["x"], w["y"])
            if c is not None and not self._failed_now(cur_map, c) \
                    and _in_bounds(cur_map, c):
                cands.append(c)
                cand_hops[c] = hops_w
        for info in (self.map_graph.portals.get(cur_map) or {}).values():
            wd = (info or {}).get("dest_map_name")
            if _block_prev and wd and str(wd).lower() == _block_prev:
                continue
            hops_p = (0 if wd == dest
                      else self._warp_leads_to(cur_map, wd, dest)) \
                if wd else None
            if hops_p is None:
                continue
            c = self._portal_map_xy(info)
            if c is not None and not self._failed_now(cur_map, c) \
                    and _in_bounds(cur_map, c):
                cands.append(c)
                cand_hops[c] = hops_p
        # WALK-EDGE CANDIDATES: pokered map connections (border walk-offs
        # to other maps). Ranked exactly like warps: dest graph-hops to
        # the goal. This is how the city's true north exit (Route 2, 1
        # hop to Pewter) outranks the drifted Route 1 south cells (3
        # hops through Viridian again).
        for we in WALK_EDGES.get(cur_map, []):
            c_w = (we["cell"][0], we["cell"][1])
            wd_w = we.get("dest")
            if _block_prev and wd_w and str(wd_w).lower() == _block_prev:
                continue
            # Walk-off border cells are NEVER skipped for past failures:
            # they are the authoritative map connection, and the BFS fold
            # reads border rows as blocked, so a "failed path" verdict on
            # them is a fold artifact, not a real failure. Blacklisting
            # (62,0) on Route 3 left tier-1 with cands=[] and the agent
            # wandering the border row for 35+ min (proven live 14:05,
            # TIER1 cands=[] while standing ON the border cell).
            if False and (cur_map, c_w) in self._failed_targets:
                continue
            hops_we = (0 if wd_w == dest
                       else self._warp_leads_to(cur_map, wd_w, dest)) \
                if wd_w else None
            if hops_we is None:
                continue
            if not _in_bounds(cur_map, c_w):
                continue
            cands.append(c_w)
            cand_hops[c_w] = hops_we
            # remember that this cand is a walk-off border cell: on BFS
            # the goal-blocked fallback stops one tile short (the border
            # row reads blocked in the fold), so the dispatch needs the
            # final step appended explicitly (same as portal approaches).
            self._walk_edge_cells.add((cur_map, c_w, we['dir']))

        # TIER-1 DEBUG: what did the candidate scan actually collect, and
        # why is the destination door not among them?
        try:
            logger.info(
                "TIER1 dest=%r pos=%s cands=%s hops=%s pc_warp=%s "
                "pc_failed_now=%s",
                dest, xy, sorted(cands)[:8],
                {k: cand_hops.get(k) for k in sorted(cands)[:8]},
                [w for w in (self.gt_warps.get(cur_map) or [])
                 if w.get("dest_name") == dest],
                [self._failed_now(cur_map, (w["x"], w["y"]))
                 for w in (self.gt_warps.get(cur_map) or [])
                 if w.get("dest_name") == dest])
        except Exception:
            logger.exception("tier1 debug log failed")

        def _is_arrival_stair(c: Tuple[int, int]) -> bool:
            if self._arrival_xy is None:
                return False
            return (abs(c[0] - self._arrival_xy[0])
                    + abs(c[1] - self._arrival_xy[1])) <= 2

        # Arrival stair goes last; among equals, the exit whose destination
        # is graph-closest to the goal goes first, then the physically
        # nearest door (kills the gate yo-yo: from Route 2 south, the North
        # Gate door (3,11) is 33 tiles north - the South Gate door (3,43)
        # is 1 tile behind - equal graph hops, but walking BACK through the
        # door you just entered is never the shortest route).
        cands.sort(key=lambda c: (_is_arrival_stair(c),
                                  cand_hops.get(c, 9),
                                  abs(c[0] - xy[0]) + abs(c[1] - xy[1])))
        for cand in cands:
            # STANDING ON A WALK-OFF BORDER CELL: the map transition fires
            # when stepping OFF this tile in the connection direction. BFS
            # returns [] (start == goal) and every fallback skips a falsy
            # path - the agent stood ON the Pewter City east border for
            # minutes with Route 3 one step away (proven live: (39,17),
            # 'edge path rejected proj=5', 'no route', 11:11). Commit the
            # off-map step directly.
            _we_dir0 = next((d for (m, c, d) in self._walk_edge_cells
                             if m == cur_map and c == cand), None)
            if xy == cand and _we_dir0:
                _off0 = {"north": "walk_up", "south": "walk_down",
                         "west": "walk_left", "east": "walk_right"}[
                             _we_dir0]
                self._waypoint_how = "off the border into the next map"
                target, how, path = cand, self._waypoint_how, [_off0] * 2
                logger.info("tier-1: standing on border cell %s (%s) - "
                            "stepping off into %s", cand, cur_map,
                            _we_dir0)
                break
            p = _map_step_toward(_mw_bfs, xy, cand, limit=400, ledge_edges=ledges,
                                 avoid_warps=warp_cells,
                                 avoid_cells=failed_cells,
                                 map_name=cur_map)
            if not p:
                # DOOR-APPROACH FALLBACK: door tiles read blocked in the
                # fold (they are recorded unwalkable by design), and when
                # EVERY neighbor reads blocked too (bad camera-edge reads),
                # the door BFS fails and the exit starves - proven live:
                # Viridian City S1/S2 (36,31)/(38,31) unreachable while
                # (35,30), one tile NW, sits on a 49-step path. Retry to
                # any walkable cell within 2 of the door; arriving there
                # puts the door on-screen and the interact/walk-through
                # logic takes over.
                near = [(cand[0] + dx, cand[1] + dy)
                        for dx in range(-2, 3) for dy in range(-2, 3)
                        if (dx, dy) != (0, 0) and mw.get(
                            (cand[0] + dx, cand[1] + dy))
                        and (cur_map, (cand[0] + dx, cand[1] + dy))
                        not in self._failed_targets]
                near.sort(key=lambda c: abs(c[0] - cand[0])
                          + abs(c[1] - cand[1]))
                for approach in near[:4]:
                    p = _map_step_toward(
                        _mw_bfs, xy, approach, limit=400, ledge_edges=ledges,
                        avoid_warps=warp_cells, avoid_cells=failed_cells,
                        map_name=cur_map)
                    if p:
                        # COMPLETE THE STEP: the approach cell is 1-2 off
                        # the door; append the final step(s) onto the door
                        # tile itself. The grid records doors unwalkable
                        # (the MW override dispatches them - that is its
                        # whole job), and stopping AT the approach leaves
                        # the agent one tile short forever (proven live:
                        # (35,30) approach, S1 door (36,31) never entered).
                        cx_a, cy_a = approach
                        guard = 0
                        while (cx_a, cy_a) != cand and guard < 4:
                            guard += 1
                            dx_a, dy_a = cand[0] - cx_a, cand[1] - cy_a
                            if abs(dx_a) >= abs(dy_a) and dx_a:
                                step_a = "walk_right" if dx_a > 0 else "walk_left"
                                cx_a += 1 if dx_a > 0 else -1
                            elif dy_a:
                                step_a = "walk_down" if dy_a > 0 else "walk_up"
                                cy_a += 1 if dy_a > 0 else -1
                            else:
                                break
                            p = p + [step_a]
                        target, how, path = approach, \
                            "to the portal approach", p
                        break
            _we_dir = next((d for (m, c, d) in self._walk_edge_cells
                            if m == cur_map and c == cand), None)
            _p_end = self._path_target(xy, p) if p else None
            if p and _we_dir is not None and _p_end is not None \
                    and (abs(_p_end[0] - cand[0]) + abs(_p_end[1] - cand[1])
                         <= 1):
                # COMPLETE THE BORDER STEP: the path stopped at the last
                # walkable cell before the border row; append steps onto
                # the border cell itself (the MW override dispatches it).
                cx_e, cy_e = self._path_target(xy, p)
                guard_e = 0
                while (cx_e, cy_e) != cand and guard_e < 4:
                    guard_e += 1
                    dx_e, dy_e = cand[0] - cx_e, cand[1] - cy_e
                    if abs(dx_e) >= abs(dy_e) and dx_e:
                        step_e = "walk_right" if dx_e > 0 else "walk_left"
                        cx_e += 1 if dx_e > 0 else -1
                    elif dy_e:
                        step_e = "walk_down" if dy_e > 0 else "walk_up"
                        cy_e += 1 if dy_e > 0 else -1
                    else:
                        break
                    p = p + [step_e]
                # ONE MORE: the border cell is the last city tile; the
                # map transition fires when stepping OFF it in the
                # connection direction.
                _off = {"north": "walk_up", "south": "walk_down",
                        "west": "walk_left", "east": "walk_right"}.get(
                            _we_dir)
                if _off:
                    p = p + [_off]
            if p:
                # PLAIN TIER-1 SUCCESS: this is the normal case - BFS found
                # a path to the candidate door. Commit cand as the target
                # and label the route honestly. The commit 44735c3 rewrite
                # made this 'target, how, path = target, how, p', which
                # self-assigned the still-None target: every plain BFS
                # success was silently discarded (break with target=None)
                # and the router fell through to 'no route', so ONLY the
                # fallback branches (approach cells, walk-edges, portal
                # chains, greedy) ever produced offers - the 41-step path
                # to the Pokecenter door existed and was thrown away on
                # every single turn (proven live: TIER1 cands non-empty,
                # BFS-FAIL absent, 'no route' 5ms later).
                if target is None:
                    target = cand
                    if not how:
                        how = "to the portal (tier-1 BFS)"
                self._waypoint_how = how
                path = p
                break
            # Multi-floor caves: no single-map BFS reaches a warp in another
            # half/floor (proven live: 1F north -> the y=35 south exits needs
            # 1F->B1F->1F-south). Chain through portals: BFS to the nearest
            # warp leading ANYWHERE, walk it, and let the next map's turn
            # re-route toward dest. Greedy (no global search) but monotone:
            # every leg crosses a real warp, so the agent always changes map.
            if p is None and target is None:
                chain = [(w["x"], w["y"]) for w in
                         (self.gt_warps.get(cur_map) or [])
                         if (w["x"], w["y"]) != self._arrival_xy
                         and (cur_map, (w["x"], w["y"]))
                         not in self._failed_targets]
                # Prefer warps whose dest is dest itself, then any warp.
                chain.sort(key=lambda c: 0 if any(
                    w.get("dest_name") == dest and (w["x"], w["y"]) == c
                    for w in (self.gt_warps.get(cur_map) or [])) else 1)
                for c2 in chain:
                    p2 = _map_step_toward(_mw_bfs, xy, c2, limit=400,
                                          ledge_edges=ledges,
                                          avoid_warps=warp_cells,
                                          avoid_cells=failed_cells,
                                          map_name=cur_map)
                    if p2:
                        target, how, path = c2, "portal chain (toward dest)", p2
                        break
            if target is not None:
                break
            # The stitched map may be wrong about the corridor (it is a
            # guess accumulated from screens; the warp table is not). Fall
            # back to dead-reckoning: greedy steps that reduce Manhattan
            # distance, taking unknown ('?') cells as walkable. The stall
            # check below still guards against a genuinely walled target.
            # DEAD-RECKONING AVOIDANCE: the greedy walk must respect the
            # same blacklist as BFS - stepping into a CONFIRMED-block cell
            # (2x bump-no-move) or known-failed cell just replays the
            # doomed batch (proven live: walk_left x8 into (15,1) after
            # three watchdog blacklists, every 15s, forever). When the
            # greedy path's FIRST step enters an avoided cell, this cand
            # is not greedy-reachable: skip it (the chain/edge/frontier
            # tiers handle routing around).
            if p is None and mw.get(cand) is not None and not mw.get(cand):
                continue          # target itself known-blocked: skip
            greedy: List[str] = []
            cx, cy = xy
            greedy_dead = False
            for _ in range(64):
                if (cx, cy) == cand:
                    break
                dx, dy = cand[0] - cx, cand[1] - cy
                step = ("walk_right" if dx > 0 else "walk_left") \
                    if abs(dx) >= abs(dy) and dx else \
                    ("walk_down" if dy > 0 else "walk_up")
                dr_g, dc_g = _DIRS.get(step, (0, 0))
                nxt_g = (cx + dc_g, cy + dr_g)
                if nxt_g in failed_cells \
                        or not _in_bounds(cur_map, nxt_g) \
                        or mw.get(nxt_g) is False \
                        or (nxt_g in warp_cells and nxt_g != cand):
                    # mw False counts too: the watchdog/shield correct the
                    # grid to False for physically-confirmed walls; a
                    # greedy walk that only checks the avoid lists replays
                    # the bump whenever the lists and the grid disagree
                    # mid-cycle (proven live: Route 2 (9,10), 14k turns).
                    greedy_dead = True
                    break
                greedy.append(step)
                if step == "walk_right":
                    cx += 1
                elif step == "walk_left":
                    cx -= 1
                elif step == "walk_down":
                    cy += 1
                else:
                    cy -= 1
                if len(greedy) >= 64:
                    break
            if greedy_dead:
                continue
            if greedy:
                target, how, path = cand, "through the portal (dead reckoning)", \
                    greedy
                break

        # tier 2: BFS to the boundary in the compass direction. SKIPPED when
        # leaving a cave/multi-floor interior for an outdoors map: floors
        # STACK vertically, so the rectangle graph's compass is meaningless -
        # proven live, first_hop(Mt Moon 1F -> Route 4) = 'south' (the (14,35)
        # warps back toward Route 3) while the real exit is the north walk-off
        # at (18,5). A compass walk there ping-pongs the agent between floors
        # forever. Portals (tier 1) + frontier (tier 3) carry the truth.
        skip_compass = (self._is_interior(cur_map)
                        and not self._is_interior(dest))
        # F2 (tier-2 subordination): the compass is derived from
        # screen-relative portal positions and is often simply wrong on
        # multi-exit maps - 'via west' fired 1971 times vs 'via east' 7
        # while the Pokecenter door sat SE of the agent. Two gates:
        #
        #   (a) TIER-1 PRECEDENCE - if tier-1 PRODUCED A ROUTE (target
        #       committed), the compass may not override it. Candidates
        #       alone are NOT enough: a lone candidate whose BFS failed
        #       (door unreachable from here) used to suppress the compass
        #       and starve the real exit (proven live: Pewter City ->
        #       Route 3, 1 candidate, BFS failed, compass 'east'
        #       suppressed, 'no route', free-walk wandered back into the
        #       gym - 10:55).
        #
        #   (b) DOOR-BEARING CORRECTION - when the destination IS a known
        #       door on this map (gt_warps dest_name == dest), derive the
        #       compass from the DOOR's global position relative to the
        #       agent, not from the portal graph. Manhattan-dominant axis
        #       only, so it stays a compass.
        if target is not None:
            skip_compass = True
            logger.info("tier-2 compass suppressed: tier-1 committed a "
                        "route to %s - data outranks the compass guess",
                        dest)
        if target is None and not skip_compass:
            compass = None
            try:
                compass = self.map_graph.first_hop_direction(cur_map, dest)
            except Exception:
                logger.exception("first_hop_direction(%s, %s) failed",
                                 cur_map, dest)
            # F2b: door-bearing correction. A destination door known on
            # THIS map overrides the graph's compass with geometry.
            if compass:
                _door = next(((w["x"], w["y"]) for w in
                              (self.gt_warps.get(cur_map) or [])
                              if w.get("dest_name") == dest), None)
                if _door is not None:
                    _dx, _dy = _door[0] - xy[0], _door[1] - xy[1]
                    _geo = (("east" if _dx > 0 else "west")
                            if abs(_dx) >= abs(_dy)
                            else ("south" if _dy > 0 else "north"))
                    if _geo != compass:
                        logger.info("tier-2 compass corrected by door "
                                    "bearing: graph said %s, door %s is "
                                    "%s - using %s",
                                    compass, _door, _geo, _geo)
                    compass = _geo
            # graph speaks COMPASS, BFS speaks BUTTON directions
            walk_dir = COMPASS_TO_WALK.get(compass) if compass else None
            if walk_dir:
                p = _map_edge_path(mw, xy, walk_dir, limit=400,
                                   ledge_edges=ledges,
                                   avoid_cells=failed_cells)
                # MINIMUM-PROGRESS GATE: an edge path that only advances
                # 1-2 tiles along the compass direction is a dead end
                # (proven live: Viridian City 'south' path hits the fence
                # after 2 steps while the real exit is 14 tiles east).
                # Fall through to frontier exploration so the agent walks
                # laterally and EXPANDS the known grid.
                end_cell = self._path_target(xy, p) if p else xy
                dr, dc = _DIRS[f"walk_{walk_dir}"]
                proj = (end_cell[1] - xy[1]) * dr + (end_cell[0] - xy[0]) * dc
                # RECENT-ARRIVALS SHELF: an end cell we JUST reached (within
                # 40 turns) is not a destination - re-picking it is the
                # (16,1) ping-pong. Expire after 40 turns so the frontier
                # can legitimately re-use corridor cells once the maze is
                # mapped and real progress demands it.
                shelf = getattr(self, "_arrived_targets", {}) or {}
                shelf_hit = shelf.get((cur_map, end_cell))
                shelf_fresh = (shelf_hit is not None
                               and self.turn - shelf_hit < 40)
                if p and proj >= 3 and (cur_map, end_cell) \
                        not in self._failed_targets and not shelf_fresh \
                        and _in_bounds(cur_map, end_cell):
                    # F3 (compass cooldown): a compass heading whose target
                    # was dropped by the net-progress stall must not be
                    # re-picked immediately - that re-pick is the 20-turn
                    # west-loop period observed live (drop, re-commit the
                    # same 'via west', drag east, repeat). 60 turns of
                    # cooldown per (map, compass) gives tier-1/frontier
                    # room to route properly.
                    _cd = getattr(self, "_compass_cooldown", {}) or {}
                    _cd_key = (cur_map, compass)
                    _cd_until = _cd.get(_cd_key, 0)
                    if self.turn < _cd_until:
                        logger.info(
                            "tier-2 compass %s on %s in cooldown (%d more "
                            "turns) - falling to frontier",
                            compass, cur_map, _cd_until - self.turn)
                    else:
                        target, how, path = (
                            end_cell,
                            f"{compass} along the stitched map", p)
                        self._waypoint_how = how
                elif p:
                    logger.info("edge path rejected: proj=%d < 3 along %s "
                                "(end %s) - falling to frontier",
                                proj, compass, end_cell)

        # tier 3: discover the boundary (winding / unstitched)
        if target is None:
            p = _map_frontier_step(mw, xy, limit=400)
            if p and (cur_map, self._path_target(xy, p)) \
                    not in self._failed_targets \
                    and _in_bounds(cur_map, self._path_target(xy, p)):
                target, how, path = (self._path_target(xy, p),
                                     "toward the unexplored edge", p)

        if target is None:
            logger.info("no route %s->%s: map not stitched at all",
                        cur_map, dest)
            return None

        # NULL-TARGET GUARD: a committed target equal to the current cell
        # is not a route - the sticky block 'arrives' instantly, re-picks
        # the same cell, and the agent trembles on the spot forever
        # (proven live: Viridian Forest (18,1) re-committed every turn
        # with here=87 visits while the real north gate sat 17 tiles west
        # behind a poison wall). Reject it; the frontier tier below (or a
        # later turn with fresher folds) picks a real destination.
        if target == xy:
            logger.info("null target: picked cell == current cell - "
                        "rejecting and exploring instead")
            self._fail_target(cur_map, target)
            p = _map_frontier_step(mw, xy, limit=400)
            if p and (cur_map, self._path_target(xy, p)) \
                    not in self._failed_targets \
                    and _in_bounds(cur_map, self._path_target(xy, p)):
                target, how, path = (self._path_target(xy, p),
                                     "toward the unexplored edge "
                                     "(null-target rejected)", p)
            else:
                return None

        self._waypoint_target = target
        self._waypoint_map = cur_map
        self._waypoint_dest = dest
        return self._offer_path(path, cur_map, dest, how)


    def _heal_talk(self) -> bool:
        """Talk to the Pokecenter nurse for real, and VERIFY the heal.

        ROM GROUND TRUTH (pret/pokered data/maps/objects/*.asm): EVERY
        Pokecenter has the nurse at (3,1) facing DOWN behind the counter
        row (y=2). She is talked to OVER the counter: stand at (3,3),
        face UP, press A - exactly like a Mart clerk. Adjacent-tile search
        never finds her because her only 'adjacent' tile IS the counter
        (blocked), which is what starved Mt Moon for hours (live 18:33 -
        06:10, proven fixed 06:12 by standing at (3,3) and pressing A).

        Procedure: clear any text box, walk to (3,3), face up, mash A
        through the yes-prompt, VERIFY by party HP signature change.
        Fallback for non-standard layouts: the old ascii-NPC sweep.

        An open text box makes the ascii come back EMPTY, so any dialog
        must be dismissed before the screen can be read. Success is the
        party HP signature CHANGED - the only signal that means "healed".
        """
        def _sig(st: Dict[str, Any]):
            return tuple((m.get("hp"), m.get("max_hp"))
                         for m in (st.get("party") or [])
                         if isinstance(m, dict))

        def _st() -> Dict[str, Any]:
            try:
                return self._get("/state").json()
            except Exception:
                return {}

        def _pos_of(st: Dict[str, Any]):
            q = ((st.get("player") or {}).get("position") or {})
            return q.get("x"), q.get("y")

        start = _st()
        before = _sig(start)
        if not before:
            return False
        if all((mx or 0) > 0 and (hp or 0) >= (mx or 0)
               for hp, mx in before):
            return True                        # nothing to heal

        # --- ROM-native counter talk: nurse at (3,1), stand (3,3) face up ---
        for _attempt in range(3):
            st = _st()
            if _sig(st) != before:
                logger.info("heal_talk: HEALED")
                return True
            if (st.get("dialog") or {}).get("text_active"):
                self.act(["press_b", "wait_30", "press_b", "wait_30"])
                st = _st()
                if _sig(st) != before:
                    return True
            px, py = _pos_of(st)
            if px is None:
                return False
            # Walk to (3,3) - BFS on the live ascii, blind retries as backup.
            if (px, py) != (3, 3):
                col = st.get("collision") or {}
                loc: Dict[tuple, bool] = {}
                for line in (col.get("ascii") or "").split("\n"):
                    m = re.match(r"\s*(\d)\s+(.*)$", line)
                    if not m:
                        continue
                    srow = int(m.group(1))
                    for i, ch in enumerate(m.group(2).replace(" ", "")):
                        wx = px + (i - PLAYER_COL)
                        wy = py + (srow - PLAYER_ROW - 1)
                        loc[(wx, wy)] = ch == "." or ch == "N" or ch == "D" \
                            or (wx == px and wy == py)
                loc[(px, py)] = True
                if loc:
                    path = _map_step_toward(loc, (px, py), (3, 3), limit=40)
                    if path:
                        self.act(list(path[:8]))
                cx, cy = _pos_of(_st())
                if (cx, cy) != (3, 3):
                    # Blind approach: standard Centers are open floor in the
                    # counter lobby; x=3 col then up to y=3 usually lands.
                    self.act(["walk_down", "wait_20"] if cy is not None and cy < 3 else
                             ["walk_up", "wait_20"] if cy is not None and cy > 3 else
                             ["wait_20"])
                    cx, cy = _pos_of(_st())
                    if (cx, cy) != (3, 3):
                        continue
            # ON the counter tile: face up, talk over the counter, mash A
            self.act(["press_up", "press_a", "wait_60"])
            self.act(["press_a", "wait_150", "press_a", "wait_150"])
            self.act(["press_a", "wait_150", "press_a", "wait_150"])
            self.act(["press_a", "wait_150", "press_a", "wait_150"])
            st = _st()
            if _sig(st) != before:
                logger.info("heal_talk: HEALED via the counter nurse at (3,1)")
                return True
            # Not healed: dismiss whatever box opened and retry
            if (st.get("dialog") or {}).get("text_active"):
                self.act(["press_b", "wait_30", "press_b", "wait_30"])
        logger.warning("heal_talk: counter-nurse procedure failed - "
                       "falling back to the ascii NPC sweep")

        _HEAL_WORDS = ("heal", "pokemon center", "pokémon center",
                       "fighting fit", "perfect health")

        tried: set = set()
        for _round in range(10):
            st = _st()
            if _sig(st) != before:
                logger.info("heal_talk: HEALED")
                return True
            # An open box blanks the ascii - clear it first.
            if (st.get("dialog") or {}).get("text_active"):
                self.act(["press_b", "wait_30"])
                st = _st()
            px, py = _pos_of(st)
            if px is None:
                return False
            # Inline ascii parse (world coords): walkable cells + NPC cells.
            loc: Dict[tuple, bool] = {}
            npcs: list = []
            for line in ((st.get("collision") or {}).get("ascii") or "").split("\n"):
                m = re.match(r"\s*(\d)\s+(.*)$", line)
                if not m:
                    continue
                srow = int(m.group(1))
                for i, ch in enumerate(m.group(2).replace(" ", "")):
                    wx = px + (i - PLAYER_COL)
                    wy = py + (srow - PLAYER_ROW - 1)
                    if ch == "N":
                        npcs.append((wx, wy))
                    loc[(wx, wy)] = ch in (".", "N", "D") or (wx == px and wy == py)
            loc[(px, py)] = True
            todo = [c for c in npcs if c not in tried]
            if not todo:
                break
            todo.sort(key=lambda c: abs(c[0] - px) + abs(c[1] - py))
            nx, ny = todo[0]
            tried.add((nx, ny))
            adj = [c for c in ((nx + 1, ny), (nx - 1, ny),
                               (nx, ny + 1), (nx, ny - 1)) if loc.get(c)]
            adj.sort(key=lambda c: abs(c[0] - px) + abs(c[1] - py))
            if not adj:
                continue
            # Try each stand-tile for this NPC; talk only once we are ON it.
            for tgt in adj:
                for _step in range(6):
                    cx, cy = _pos_of(_st())
                    if cx is None:
                        break
                    if (cx, cy) == tgt:
                        break
                    path = _map_step_toward(loc, (cx, cy), tgt, limit=40)
                    if not path:
                        break
                    self.act(list(path[:6]))
                cx, cy = _pos_of(_st())
                if (cx, cy) != tgt:
                    continue
                dx, dy = nx - tgt[0], ny - tgt[1]
                d = ("walk_right" if dx > 0 else
                     "walk_left" if dx < 0 else
                     "walk_down" if dy > 0 else "walk_up")
                self.act([d, "press_a", "wait_45"])
                st = _st()
                if _sig(st) != before:
                    logger.info("heal_talk: HEALED by the NPC at (%d,%d)",
                                nx, ny)
                    return True
                if any(w in str((st.get("dialog") or {}).get("text") or "").lower()
                       for w in _HEAL_WORDS):
                    self.act(["press_a", "wait_150", "press_a", "wait_150"])
                    if _sig(_st()) != before:
                        logger.info("heal_talk: HEALED via the prompt at "
                                    "(%d,%d)", nx, ny)
                        return True
                if (st.get("dialog") or {}).get("text_active"):
                    self.act(["press_b", "wait_20", "press_b", "wait_20"])
                break            # this NPC is not the healer - next
        logger.warning("heal_talk: no healing NPC found (%d tried) - "
                       "leaving the heal to the next turn", len(tried))
        return False

    # --- gym-battle auto-video -------------------------------------------
    GYM_VIDEO_DIR = "/home/vlad/.hermes/cron/output/pokemon_video"
    GYM_VIDEO_COOLDOWN = 600   # seconds between gym recordings

    def _maybe_gym_video(self, state: Dict[str, Any]) -> None:
        """Record a 30s clip when a GYM battle starts (leader fights only).

        Fires on the trainer battle's first seen turn while the player is on
        a Gym map (wild battles on gyms are not possible, so type==trainer +
        gym map == leader). The cooldown stops re-recording on later turns
        of the same fight and keeps adjacent gym trainers from stacking
        clips. Never affects the game: recording is a server screenrecord.
        """
        battle = state.get("battle") or {}
        if str(battle.get("type") or "").lower() != "trainer":
            return
        if "gym" not in (self._plan_map or "").lower():
            return
        now = time.time()
        if now - getattr(self, "_gym_video_at", 0.0) < self.GYM_VIDEO_COOLDOWN:
            return
        self._gym_video_at = now
        os.makedirs(self.GYM_VIDEO_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.GYM_VIDEO_DIR, f"gym_battle_{ts}.mp4")
        enemy = battle.get("enemy") or {}
        logger.info("GYM BATTLE detected (%s Lv%s) - recording 30s video",
                    enemy.get("species", "?"), enemy.get("level", "?"))
        self.event(type="alert",
                   text=f"GYM BATTLE vs {enemy.get('species', '?')} - "
                        f"recording")
        try:
            r = self._post("/screenrecord",
                           {"duration": 30, "fps": 20, "scale": 2},
                           timeout=45)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
                logger.info("gym video saved: %s (%d bytes)",
                            path, len(r.content))
            else:
                logger.warning("screenrecord HTTP %s - no gym video",
                               r.status_code)
        except Exception as exc:
            logger.warning("gym video failed: %s", exc)



    def _turn(self, state: Dict[str, Any], intro: bool) -> None:

        col = state.get("collision") or {}
        passable = col.get("passable") or col.get("walkable") or []
        ledge_grid = col.get("ledges") or None
        dlg = state.get("dialog") or {}
        battle = state.get("battle") or {}
        p = state.get("player") or {}
        # Fold this screen into the per-map grid BEFORE anything reads it, so
        # the map-route option below sees the newest cells.
        _mp = p.get("position") or {}
        if (col.get("valid") and _mp.get("x") is not None
                and _mp.get("y") is not None):
            _mk = (state.get("map") or {}).get("map_name", "")
            _new_xy = (int(_mp["x"]), int(_mp["y"]))
            # (Ledge observation by movement is gone: the ROM decode gives
            # every ledge hop exactly.)
            self._prev_xy = _new_xy
            self._prev_xy_map = _mk
            self._last_xy = _new_xy
            # Ping-pong detector: a short ring of recent map-positions. When
            # only one or two tiles appear in the whole ring the agent is
            # oscillating (Pewter City north strip: (34,2)/(34,3) for hours)
            # and any single-step "route" must yield to frontier exploration.
            _ring = getattr(self, "_pos_ring", None)
            if _ring is None:
                _ring = self._pos_ring = deque(maxlen=10)
                self._pos_ring_map = _mk
            if self._pos_ring_map != _mk:
                _ring.clear()
                self._pos_ring_map = _mk
            _ring.append(self._last_xy)
            # Warp tiles are forced walkable while stamping, so hand the
            # frame's door list in (only on frames the collision layer vouches
            # for - stale frames carry a bad warp table).
            _stamp_warps = (col.get("warps") or []) if col.get("valid") else []
            # OCCUPANCY OVERLAY REMOVED (user directive 2026-10-26): the
            # static decode is the map; walked cells no longer force-open
            # anything (the body cannot stand on a wall, so any mismatch
            # is a static error to fix in the data, not to paper over).
            # The BFS keeps its single-cell standing-tile force for the
            # position-read race. Bump evidence stays log-only.
            _stamp_visits = None
            # COHERENT-STAMP GATE: stamp only when the window and the
            # position come from the same instant. build_collision_grid
            # re-reads wXCoord/wYCoord right after the tilemap; if that
            # settled pair disagrees with the (fused) player position,
            # a step completed mid-read - stamping now would write cells
            # offset by one tile (the drift class). Use the settled pair
            # as the stamp anchor instead of the fused position.
            _settled = col.get("settled_position") or {}
            if col.get("valid") and _settled and not col.get(
                    "position_coherent", True):
                _anchor = (int(_settled.get("x", 0)),
                           int(_settled.get("y", 0)))
            else:
                _anchor = self._last_xy
            _stamp_npcs = set()
            _px_n = (_anchor or (0, 0))
            for _sp in (col.get("sprites") or []):
                if _sp.get("row") is None or _sp.get("col") is None:
                    continue
                _stamp_npcs.add((_px_n[0] + int(_sp["col"]) - PLAYER_COL,
                                 _px_n[1] + int(_sp["row"]) - PLAYER_ROW))
            _stamp_screen(self.map_walk.setdefault(_mk, {}),
                          _anchor, passable, warps=_stamp_warps,
                          visited=_stamp_visits, map_name=_mk,
                          npc_cells=_stamp_npcs or None)
            # Persist every 25 turns so a restart keeps what we have learned.
            # RUNTIME WRITES DISABLED (user directive 2026-10-26): the map
            # file stays exactly the static decode - runtime window merges
            # are session-ephemeral, so transient phantom openings can
            # never bake into the data again. Re-enabled only by setting
            # POKEMON_PERSIST_FOLD=1.
            if (os.environ.get("POKEMON_PERSIST_FOLD", "0") == "1"
                    and self.turn - self._map_walk_saved_at >= 25):
                self._map_walk_saved_at = self.turn
                _save_map_walk(self.map_walk)
        # Only trust the warp list on a frame the collision layer vouches for.
        # An invalid frame (mid-transition, battle, dialog) carries a stale
        # warp table, so `goto_`/`leave_building` would path to a phantom door
        # and never arrive - the agent kept re-entering buildings this way.
        warps = (col.get("warps") or []) if col.get("valid") else []
        # Quest flags so Laya knows what's done and skips unnecessary buildings
        flags = state.get("flags") or {}
        has_parcel = bool(flags.get("has_oaks_parcel"))
        has_dex = bool(flags.get("has_pokedex"))

        cur_map = (state.get("map") or {}).get("map_name", "")
        pos = p.get("position") or {}
        if cur_map != self.prev_map:
            logger.info("map changed: %s -> %s", self.prev_map, cur_map)
            # Recency stamp for the no-immediate-backtrack portal guard
            # (proven live 01:09): on Route 4 the tier-1 router's only
            # candidate was the Mt Moon cave mouth and it walked straight
            # back in 2s after every crossing. A portal that leads to the
            # map we JUST left is never the route for the next ~40 turns.
            self._prev_map_turn = getattr(self, "turn", 0)
            # A waypoint is only meaningful for the map it was committed
            # on. A commit made on the ARRIVAL turn of a new map is
            # meaningless (it was computed against the old map's context)
            # and, worse, can point straight back through the door we just
            # came through: Route 4 committed (18,5) - the Mt Moon cave
            # mouth - and re-entered the cave in 2 seconds, twice. Drop it;
            # the planner re-commits within the same turn.
            self._waypoint_target = None
            self._waypoint_map = None
            self._waypoint_path = None
            self._waypoint_dest = None
            # Which way did we come in? The hop that landed us here recorded the
            # bare direction it walked; its reverse is the boundary we just
            # crossed. Used to stop an immediate walk back out (see the
            # selection below). _route_pending is cleared later in this turn, so
            # it still holds the PREVIOUS turn's hop here.
            self.arrived_from = None
            if self._route_pending:
                _, rdir, rnext = self._route_pending
                if rnext is None or rnext == cur_map:
                    self.arrived_from = rdir
            self.prev_map = cur_map
            self.map_changed_at = self.turn
            self.map_changed_time = time.perf_counter()
            self.recent.clear()          # a real transition is not a loop
            # Re-arm edge conclusions for the map we just entered. A node can
            # be reached from more than one side - Route 2 is split by Viridian
            # Forest, and the south edge to Pewter City is a fake adjacency
            # while the north one is real. Without this the blacklist earned
            # on the south segment would suppress the real north edge too.
            entered = self.map_graph.resolve(cur_map) if self.map_graph else None
            if entered:
                for k in [k for k in self.edge_fails if k[0] == entered]:
                    self.edge_fails.pop(k, None)
        key = (cur_map, pos.get("x"), pos.get("y"))
        self.visits[key] = self.visits.get(key, 0) + 1
        # Position history for the anti-pinball guard (reset on map change).
        if cur_map != self._recent_map:
            self._recent_map = cur_map
            self._recent_pos = []
        self._recent_pos.append((pos.get("x"), pos.get("y")))
        del self._recent_pos[:-12]

        # --- patch gate tiles as walkable in collision grid ---
        # Gate/warp tiles are often marked as walls in the collision grid
        # but are actually passable (stepping on them triggers map transition).
        # NOTE: build_collision_grid() already lifts warp tiles to walkable
        # (the engine's CheckWarpsNoCollision runs before the collision check),
        # but only for warps VISIBLE in the 10x9 window. Once the warp-address
        # off-by-one was fixed (see collision.py), visible doors show up in
        # warps[] and are handled there - no extra patching needed here.

        # --- empirical edge verification -----------------------------------
        # Resolve the last route hop: if the map changed the edge is real and
        # restores goto_/explore_ so the agent can find a gate instead.
        if self._route_pending:
            rmap, rdir, rnext = self._route_pending
            moved_since = (pos.get("x"), pos.get("y")) != self._route_last_pos
            if rnext is not None and cur_map == rnext:
                # The hop worked: this edge is real and any count clears.
                self.edge_fails.pop((rmap, rdir), None)
            elif cur_map != rmap and cur_map != rnext:
                # Transitioned somewhere else (e.g. wandered back into the
                # previous town). That neither proves nor disproves the edge,
                # so keep the count - resetting here is what let the blacklist
                # clear itself on every Viridian City round trip.
                pass
            elif cur_map == rmap and not moved_since:
                # Distinguish "the walk was blocked by an NPC/ledge in the
                # middle of the map" (transient - the agent can path around)
                # from "the map edge truly does not open" (dead edge). A
                # stall only counts as an edge failure when the agent is
                # within 2 tiles of the map boundary in the walk direction;
                # mid-map stalls just mean the blind walk needs a re-path
                # (proven live: Route 3 stall at x=22 of ~80 blacklisted
                # ('Route 3','right') and dropped the agent onto travel_lr
                # pinball for hundreds of turns).
                _near_edge = (
                    (rdir == "right" and (pos.get("x") or 0) >= 76) or
                    (rdir == "left" and (pos.get("x") or 0) <= 2) or
                    (rdir == "down" and (pos.get("y") or 0) >= 34) or
                    (rdir == "up" and (pos.get("y") or 0) <= 2))
                if _near_edge:
                    n = self._edge_bump((rmap, rdir))
                else:
                    n = 0
                    logger.info("route hop stalled mid-map at (%s,%s) - not "
                                "blacklisting edge %s/%s, needs re-path",
                                pos.get("x"), pos.get("y"), rmap, rdir)
                if n >= 4 and self.map_graph is not None and rnext:
                    # Queue the proof for the GRAPH. Applied below, once the
                    # goal is known - dropping an edge is only safe if the
                    # graph can STILL route to the goal without it. Merely
                    # suppressing the hop here left the agent with NO route at
                    # all, which dropped it onto blind travel_left/travel_right
                    # headings and started the Route 2 y=60 pinball
                    # (605 reversals = 26.6% of all transitions).
                    self._pending_dead_edge = (rmap, rnext)
            self._route_pending = None

        # --- global map graph: learn portals as we read them ---------------
        # `warps` is already empty on an invalid frame (see above), so this
        # only sees trustworthy reads. The length cap is a second guard: a
        # 10x9 screen cannot show more than a handful of doors, so a longer
        # list means the read is still suspect.
        # Portal cells are SCREEN-relative at record time. We also store the
        # player's global position at record time so the door can be found
        # again from anywhere on the map: global_door = screen_cell - player_at
        # + player_now. Without this, route_B8 has no on-screen target when the
        # player is at the other end of a big map (Viridian Forest is 24 rows
        # tall) and the handler falls to a blind single step - the wedge that
        # bounced the agent up/down at (32,1)/(32,2) for 100+ turns.
        if self.map_graph is not None and warps and len(warps) <= 12:
            learned = False
            player_at = (pos.get("x"), pos.get("y"))
            for w in warps:
                if self.map_graph.record_warp(cur_map, w,
                                              player_at=player_at):
                    learned = True
            if learned:
                self.map_graph.save_portals()

        # Did the previous explore path actually arrive? If not, the target is
        # unreachable - a table or NPC the collision grid thinks is walkable -
        # and must be retired or Laya will offer it forever.
        if self.last_explore_target is not None and self.last_explore_from:
            want_r, want_c = self.last_explore_target
            fx = self.last_explore_from[0] + (want_c - PLAYER_COL)
            fy = self.last_explore_from[1] + (want_r - PLAYER_ROW)
            if (pos.get("x"), pos.get("y")) != (fx, fy):
                fk = (cur_map, want_r, want_c)
                self.failed_targets[fk] = self.failed_targets.get(fk, 0) + 1
                logger.info("explore target %s not reached (%d failures)",
                            self.last_explore_target, self.failed_targets[fk])
            self.last_explore_target = None

        # Wall-clock, not turn count: Laya runs several turns/sec, so a turn
        # based window expires before the agent steps away from the door.
        recent_transition = (time.perf_counter() - self.map_changed_time) < 8.0
        on_warp = bool(col.get("player_on_warp")) or any(
            (w.get("row"), w.get("col")) == (PLAYER_ROW, PLAYER_COL)
            for w in warps)

        criteria: Dict[str, str] = {}
        # Defined here so the interact-spam guard (below) can read it on
        # battle/dialog turns too - it lives in BRANCH C otherwise, and the
        # guard crashed with UnboundLocalError on every battle turn
        # (1932 occurrences in one run), which is why the video showed 30s
        # of no action: every battle turn crashed before dispatching.
        directed = False
        # Track consecutive interact choices regardless of position (micro-moves
        # reset the per-tile counter but not the intent-spam).
        self._consecutive_interact = getattr(self, "_consecutive_interact", 0)

        # ------------------------------------------------------------------
        # BRANCH A - DIALOG
        # ------------------------------------------------------------------
        # BRANCH ORDER: branch A (dialog) yields to the battle branch
        # whenever a battle is live. The in-battle move menu renders as a
        # 'text box' to the reader, so text_active is True during battles
        # and this arm used to claim every turn - A-spam at the FIGHT
        # menu froze both HP bars (proven live: Route 3 trainer Caterpie
        # 30/30 at 12:03-12:08, and a 3-hour stall on the Route 3
        # Youngster 13:12-16:27).
        #
        # ALSO FIXES MY OWN BUG (24ac014): I made the battle CONDITION the
        # first arm but left the DIALOG body under it, making the real
        # battle branch an unreachable `elif` - strictly worse than before.
        # The condition is now `dialog AND NOT battle` so the battle
        # branch below actually runs.
        if (intro or dlg.get("text_active")) and not battle.get("in_battle"):
            # Consecutive dialog-active turns: the flag can go stale while the
            # model keeps offering movement instead of pressing A. After 2
            # stalled turns, force advance_text and suppress everything else -
            # walking away mid-conversation is why the mart clerk never finished.
            self._dialog_turns = getattr(self, "_dialog_turns", 0) + 1
            force_advance = self._dialog_turns > 2

            instructions = ("A text box or menu is on screen. Press A to advance it. "
                            "Do NOT walk away - the NPC will not finish the interaction.")
            criteria["advance_text"] = "Press A to advance dialog"
            if force_advance:
                criteria["advance_text"] += " (REQUIRED - dialog is stalled)"
            else:
                criteria["back_out"] = "Press B to dismiss or cancel"
                for name in _legal_dirs(passable):
                    criteria[name] = f"Move one tile {name.split('_')[1]}"
                if intro:
                    criteria["menu_down_a"] = "Move the menu cursor down, then confirm"
        elif battle.get("in_battle"):
            # Gym-battle auto-video (see _maybe_gym_video docstring).
            try:
                self._maybe_gym_video(state)
            except Exception:
                logger.exception("gym video hook failed - continuing")
            enemy = battle.get("enemy") or {}
            mine = state.get("active_mon") or {}
            hp_frac = (mine.get("hp") or 0) / (mine.get("max_hp") or 1)
            # PP-aware move list so the AI never picks a move it cannot use.
            # All four slots at 0 PP means the game forces STRUGGLE (typeless,
            # 1/4 recoil) - pressing FIGHT still works, the game picks the
            # move for us. Growl has PP but deals NO damage - it cannot win a
            # battle alone, so when no damaging move has PP, attack anyway
            # and let the game's STRUGGLE handle it.
            moves = mine.get("moves", [])
            pps = [mv.get("pp", 0) if isinstance(mv, dict) else 0
                   for mv in moves]
            any_pp = any(pp > 0 for pp in pps)
            instructions = (
                f"You are in a battle against {enemy.get('species','?')} "
                f"Lv{enemy.get('level','?')} "
                f"({enemy.get('hp','?')}/{enemy.get('max_hp','?')} HP). "
                f"Your {mine.get('nickname','?')} is at "
                f"{mine.get('hp','?')}/{mine.get('max_hp','?')} HP. "
                f"Move PP: {', '.join(f'{m.get('name', '?')}={m.get('pp', 0)}' for m in moves) if moves else 'none'}. "
                "The menu is FIGHT / PKMN / ITEM / RUN. Choose what to do."
            )
            if not any_pp:
                # Every move slot is out of PP - selecting FIGHT makes the
                # game use STRUGGLE automatically. This is the ONLY way to
                # deal damage now; menu-looping between SWITCH and FIGHT
                # achieves nothing (proven live: wedged vs Weedle).
                criteria["attack"] = (
                    "Select FIGHT - all moves are out of PP so the game will "
                    "use STRUGGLE (the only remaining way to deal damage)"
                )
            else:
                # Offer EVERY move with PP, not just slot 0: the Gen 1 battle
                # policy picks on merit (type effectiveness + STAB + power),
                # and a fallback scorer needs the set to choose from. Slot
                # index is the displayed-menu position - the reader compacts
                # empty slots and so does the game's move list.
                for i, mv in enumerate(moves):
                    pp = mv.get("pp", 0) if isinstance(mv, dict) else 0
                    if pp > 0:
                        move_name = mv.get("name", f"move_{i}")
                        criteria[f"move_{i}"] = (
                            f"Select FIGHT, navigate to {move_name} (PP={pp}), "
                            "and confirm"
                        )
            # RUN is offered when we are hurt, OR when no damaging move has PP
            # (a wild battle is then unwinnable by attrition). The battle policy
            # returns "flee" for that second case, but this criterion used to
            # exist only for low HP - so the policy's own choice was rejected as
            # "not offered" (141 times in one run) and the scorer picked a
            # useless status move instead. Match the criterion to the policy so
            # the two agree.
            _ranked_b = gen1_battle.score_moves(state) if moves else []
            _no_damaging_pp = not any(e[1] > 0 for e in _ranked_b)
            _is_wild = str(battle.get("type") or "").lower() != "trainer"
            # TRIVIAL-WILD (matches gen1_battle.choose's skip): a wild enemy
            # far below our level is not worth PP - the policy flees and
            # this criterion must offer it or the choice is rejected as
            # "not offered" and the scorer burns PP instead (live 07:53).
            _en_b = battle.get("enemy") or {}
            _my_lv = int((state.get("active_mon") or {}).get("level") or 0)
            _en_lv = int(_en_b.get("level") or 0)
            _trivial = (_is_wild and _en_lv > 0 and _my_lv >= _en_lv + 8
                        and _my_lv >= 2 * _en_lv)
            if hp_frac < 0.35 or (_no_damaging_pp and _is_wild) or _trivial:
                criteria["flee"] = "Select RUN and escape this battle"
            criteria["advance_text"] = "Press A to advance battle text"
            criteria["pkmn"] = "Select PKMN to switch Pokémon"
            criteria["item"] = "Select ITEM to use an item"
        # ------------------------------------------------------------------

        else:
            # ------------------------------------------------------------
            # ROUTING - VLM-first architecture.
            # ------------------------------------------------------------
            # The planner (VLM in vlm mode, rules otherwise) ALWAYS speaks
            # first: it decides WHAT (the goal, including WHICH Pokecenter).
            # The driver only decides HOW: route via BFS/portal-chain, or
            # run an action primitive (heal_party = counter-talk the nurse).
            # No driver-side heuristic may override or pre-empt the goal
            # (the old heal-first branch ignored the VLM entirely and
            # wedged Mt Moon for 12h - proven live 18:33-06:12).
            goal = self._plan_goal(state, cur_map)
            g = (goal or "").lower()
            heal_goal = ("heal" in g and "pokemon center" in g) \
                or ("heal" in g and "pokecenter" in g)

            # 1. HEAL GOAL EXECUTION:
            #    - At the named center (or any center when the goal says
            #      'nearest'): run the heal_party primitive (ROM-native
            #      counter talk - stand (3,3), face up, A; verified by HP).
            #    - Elsewhere: BFS/portal-chain route to the named center.
            if heal_goal:
                _gdest = self._graph_destination(goal, cur_map)
                _at_center = "pokecenter" in cur_map.lower() \
                    or _gdest == cur_map
                if _at_center:
                    # WAYPOINT COMPLETION: arriving at the center finishes
                    # any committed door waypoint.
                    if self._waypoint_target is not None:
                        logger.info("heal waypoint COMPLETE - clearing %s",
                                    self._waypoint_target)
                        self._waypoint_target = None
                        self._waypoint_path = []
                    # heal_party PRIMITIVE - the one and only way to heal.
                    # The primitive drives inputs directly (like _heal_talk
                    # always did); this turn offers wait-only options so the
                    # scorer has something legal while the inputs land.
                    healed = self._heal_talk()
                    criteria = {"wait": "Hold position - the heal just ran"}
                    if healed:
                        logger.info("heal_party: primitive returned healed")
                    # Primitive failed: stand down this turn. The goal stays
                    # latched; the VLM sees the unchanged HP next turn and
                    # may pick a different center (its call, not ours).
                    if not healed:
                        logger.warning("heal_party: primitive failed - "
                                       "standing down, goal stays %r",
                                       goal[:60])
                else:
                    # Not at the named center: route there (pure HOW).
                    route = self._resolve_route(state, cur_map, pos, goal)
                    if route:
                        criteria = dict(route)
                        directed = True
                        instructions = f"OBJECTIVE: {goal}"
                    else:
                        criteria = {}
                    logger.info("heal route: %s (at_center=%s dest=%s)",
                                list(criteria), _at_center, _gdest)
            else:
                # 2. Non-heal goal: local action, then route, then frontier.
                local = self._local_action(state, cur_map, pos, goal)
                # STICKY-WAYPOINT PRECEDENCE: a committed waypoint for this
                # map+dest is authoritative. Without this, the free-walk
                # scorer (8 direction options) fights route_to_waypoint on
                # alternating turns and the agent oscillates on the spot -
                # 5-10 waggles on the Pokecenter front door before finally
                # exiting (proven live, turns 1634-1642).
                # GOAL-NAMED LADDER OVERRIDE (proven live 23:55): when the
                # goal names a specific ladder "(x,y)" for the committed
                # destination and the sticky waypoint is a DIFFERENT ladder
                # to that same destination, the commit is stale - it was
                # picked before the goal named the correct ladder. On Mt
                # Moon B2F the goal says "(5,7)" (the only ladder landing
                # in the exit strip) while the commit was (25,9) (nearer,
                # but it lands in B1F SOUTH, whose rule says "descend to
                # B2F" - a stable B1F<->B2F cycle, 23:53-23:55). Re-target
                # to the named ladder and drop the cached path.
                if self._waypoint_target is not None and goal:
                    _m2 = re.search(r"\((\d{1,3})\s*,\s*(\d{1,3})\)", goal)
                    if _m2:
                        _want = (int(_m2.group(1)), int(_m2.group(2)))
                        if _want != tuple(self._waypoint_target):
                            _wdest = self._graph_destination(
                                goal, exclude=cur_map)
                            if (_wdest and self._waypoint_dest
                                    and str(_wdest).lower()
                                    == str(self._waypoint_dest).lower()):
                                logger.info(
                                    "goal-named ladder override: waypoint "
                                    "%s -> %s (goal names a different "
                                    "ladder to the same destination)",
                                    self._waypoint_target, _want)
                                self._waypoint_target = _want
                                self._waypoint_map = cur_map
                                self._waypoint_path = None
                                    # STALE EXIT WAYPOINT (proven live 17:08): the waypoint
                # (14,35) = the Mt Moon 1F west door was set to exit to
                # Route 4. On Route 4 the waypoint's map no longer matches,
                # so it can never "arrive" - and its target is only
                # reachable back INSIDE the cave, so the driver re-entered
                # and descended to B2F again 2 seconds after exiting. A
                # waypoint committed for map A is dead the moment the agent
                # stands on a different map: clear it and let the planner
                # set the next objective.
                if (self._waypoint_target is not None
                        and self._waypoint_map != cur_map
                        and cur_map not in (goal or '')):
                    logger.info("stale waypoint cleared: target=%s was for "
                                "map %s, now on %s (goal=%s)",
                                self._waypoint_target, self._waypoint_map,
                                cur_map, (goal or '')[:50])
                    self._waypoint_target = None
                    self._waypoint_map = None
                    self._waypoint_path = None
                    self._waypoint_dest = None
                if (self._waypoint_target is not None
                        and self._waypoint_map == cur_map
                        and self._waypoint_dest == self._graph_destination(
                            goal, cur_map)):
                    criteria = {"route_to_waypoint":
                                ("Follow the pre-computed path toward the "
                                 "objective. It is the only offered option.")}
                    directed = True
                    instructions = ("Follow the committed route. Do NOT "
                                    "free-walk.")
                    logger.info("sticky waypoint: route_to_waypoint only "
                                "(target=%s)", self._waypoint_target)
                elif local:
                    criteria = dict(local)
                    directed = True
                    instructions = f"OBJECTIVE: {goal}"
                    logger.info("local objective: %s", list(local))
                else:
                    # 3. Planner -> goal -> destination map -> compass -> BFS.
                    route = self._resolve_route(state, cur_map, pos, goal)
                    if route:
                        criteria = dict(route)
                        directed = True
                        instructions = ("Follow the single offered route. It "
                                        "is the pre-computed path out of this "
                                        "map.")
                    else:
                        # 4. No graph route: explore the frontier. Still
                        #    offered as few, named destinations - not a wall
                        #    of legal moves - so the scorer has something
                        #    meaningful to rank rather than a coin flip.
                        # F1 (PLANNER-MODE FREE-WALK GATE): in planner
                        # mode the planner owns WHERE the agent goes and
                        # Von computes HOW. Free-walk single steps are
                        # the pre-planner exploration fallback - the
                        # producer that walked him into the museum - and
                        # they must never compete with a routed plan.
                        # README rule 8 documented the intent; this
                        # enforces it. Frontier targets below STAY
                        # offered (they are named destinations, not
                        # blind steps); the 8-direction single-step
                        # wall is dropped entirely.
                        _vlm_gate = (self.planner_mode == "vlm"
                                     and not self._plan_goal_text_offers())
                        frontier = _frontier_targets(
                            passable, self.visits, cur_map,
                            pos.get("x") or 0, pos.get("y") or 0,
                            union_global=self.union_global or None,
                            map_offset=self.map_offsets.get(cur_map),
                            warp_cells={(w["x"], w["y"])
                                        for w in (self.gt_warps.get(cur_map)
                                                  or [])}
                                        | set(getattr(
                                            self,
                                            "_active_warp_cells",
                                            set()) or set()))
                        # SHELF-DISCIPLINED FRONTIER: explore targets
                        # on recently-arrived-at cells (the door we
                        # just exited) are the re-entry ping-pong -
                        # drop them from the offered set.
                        shelf_f = getattr(self, "_arrived_targets", {}) \
                            or {}
                        frontier = [t for t in frontier
                                    if (cur_map, t) not in shelf_f
                                    or self.turn - shelf_f.get(
                                        (cur_map, t), 0) >= 40]
                        if frontier:
                            for tgt in frontier[:4]:
                                label = cell_label_local(tgt[1], tgt[0])
                                criteria[f"explore_{label}"] = (
                                    f"Walk to the unexplored cell {label}")
                            directed = True
                        if on_warp:
                            criteria["interact"] = (
                                "Press A to use the door/exits")
                        # SHELF-DISCIPLINED FREE WALK: single-step moves
                        # that ENTER a recently-arrived-at cell are the
                        # ping-pong the router just refused - don't
                        # offer them. The cell stays reachable for real
                        # routes; only the immediate re-entry option is
                        # suppressed (expires with the shelf, 40 turns).
                        shelf = getattr(self, "_arrived_targets", {}) \
                            or {}
                        # WARP-CELL DISCIPLINE for the free-walk producer.
                        # Doors are entered only as an EXPLICIT route
                        # target; an aimless single step must never walk
                        # onto one. Without this the explorer strolls into
                        # buildings it has no business in - proven live:
                        # the Pewter Museum, re-entered 25s after walking
                        # out, because the entrance sits right next to
                        # where the driver started. Same avoidance source
                        # as the BFS and the greedy (pattern 47: every
                        # producer consults the same set).
                        _wc_free = {(w["x"], w["y"])
                                    for w in (self.gt_warps.get(cur_map)
                                              or [])}
                        _wc_free |= set(getattr(
                            self, "_active_warp_cells", set()) or set())
                        _me_free = (pos.get("x") or 0, pos.get("y") or 0)
                        _standing_on_door = _me_free in _wc_free
                        if not _vlm_gate:
                            for name in _legal_dirs(passable):
                                dr_s, dc_s = _DIRS.get(name, (0, 0))
                                nxt_s = (_me_free[0] + dc_s,
                                         _me_free[1] + dr_s)
                                shelf_hit_s = shelf.get((cur_map, nxt_s))
                                if (shelf_hit_s is not None
                                        and self.turn - shelf_hit_s < 40):
                                    continue
                                # Do not step ONTO a door from ordinary
                                # ground (stepping OFF one is how you
                                # leave).
                                if nxt_s in _wc_free \
                                        and not _standing_on_door:
                                    continue
                                criteria[name] = (
                                    f"Move one tile {name.split('_')[1]}")
                        if recent_transition:
                            criteria["back_out"] = "Press B to dismiss"
                        instructions = ("Choose the best way to make "
                                        "progress: a named destination beats "
                                        "a single step.")

        # EMPTY-CRITERIA FALLBACK: every producer can decline (all-visited
        # screen, doors excluded, route unstitched, gate closed) and the
        # driver then died with 'options offered: []' x5 -> 'Laya failed 5
        # turns in a row' (proven live: Route 3 (11,6), 12:45 AND again at
        # 13:12 -> a 3-hour stall). Last-resort safety net: it must NEVER
        # be empty, because an empty criteria set kills the driver and a
        # dead driver loses hours. One bad step is strictly cheaper.
        if not criteria:
            if battle.get("in_battle"):
                # FIGHT always works: with no PP the game runs Struggle.
                criteria["attack"] = ("Select FIGHT - fallback, the battle "
                                      "branch offered nothing")
                criteria["advance_text"] = "Press A to advance battle text"
            elif dlg.get("text_active") or intro:
                criteria["advance_text"] = "Press A to advance dialog"
                criteria["back_out"] = "Press B to dismiss"
            else:
                for name in _legal_dirs(passable):
                    criteria[name] = f"Move one tile {name.split('_')[1]}"
            if not criteria:
                criteria["advance_text"] = "Press A to advance anything"
                criteria["back_out"] = "Press B to dismiss anything"
            logger.warning("empty criteria - last-resort fallback (%s)",
                           sorted(criteria))

        has_badges = int((flags.get("badge_count") or 0))
        laya_state = {
            "map_name": cur_map,
            "x": pos.get("x"),
            "y": pos.get("y"),
            "facing": p.get("facing"),
            "on_door": on_warp,
            "in_battle": bool(battle.get("in_battle")),
            "text_active": bool(dlg.get("text_active")),
            "phase": (state.get("context") or {}).get("phase"),
            "map_ascii": col.get("ascii") or "",
            "exits": [w.get("cell") for w in warps],
            "stuck_turns": self.stuck,
            "times_on_this_tile": self.visits.get(key, 0),
            "tiles_seen_this_map": sum(1 for (m, _, _) in self.visits
                                       if m == cur_map),
            "maps_seen": len({m for (m, _, _) in self.visits}),
            "plan": self.active_plan or None,
            "plan_age": (self.turn - self._plan_turn
                         if self.active_plan else None),
            "has_oaks_parcel": has_parcel,
            "has_pokedex": has_dex,
            "badge_count": has_badges,
        }

        # Oscillation guard. Do NOT clear the window on dialog turns - a text
        # box mid-cycle is part of the cycle, and clearing means the window
        # never fills. Only append while exploring, since a static position is
        # normal in battle and dialog.
        exploring = not (intro or dlg.get("text_active")
                         or battle.get("in_battle"))
        stall_defer = (self._waypoint_target is not None
                       and self._wp_stall >= 2)
        if exploring and not stall_defer:
            self.recent.append((cur_map, pos.get("x"), pos.get("y")))
        if len(self.recent) == self.recent.maxlen:
            maps = {m for m, _, _ in self.recent}
            cells = set(self.recent)
            # Ping-ponging between two maps is a loop even though the cells
            # differ, so a plain cell-count test misses it.
            if len(cells) <= 3 or (len(maps) == 2 and len(cells) <= 6):
                # On a door, the escape IS the exit - try it before wandering.
                if on_warp:
                    logger.warning("loop detected on a door - forcing exit")
                    self.recent.clear()
                    self.act(["walk_down", "walk_down"])
                    return
                legal = list(_legal_dirs(passable))
                escape = legal[self.turn % len(legal)] if legal else "press_b"
                logger.warning("loop detected (%d map(s), %d cell(s)) - "
                               "escaping via %s", len(maps), len(cells), escape)
                self.recent.clear()
                self.act([escape] * 4)
                return

        # Interact spam guard. Von/Laya rate "interact" highest whenever the
        # option set is small, so the agent stands on one tile pressing A 40+
        # turns in a row (proven live: here=54 on Route 2). When we have been
        # on this tile for a while and NOT on a door/exit, interacting again
        # cannot be right - take a step instead and demote interact.
        # Run regardless of exploring state or directed state - trainer NPCs
        # trigger dialog cycles that reset exploring, but the spam is still real.
        # The key insight: ALWAYS suppress interact after 6 turns, but only
        # ADD travel options when not directed (so heal route isn't competed with).
        # Intent-level counter: micro-moves reset the per-tile counter but not
        # _consecutive_interact - a trainer on an edge ate 200+ turns this way.
        if (not on_warp and (laya_state["times_on_this_tile"] >= 6
                             or self._consecutive_interact >= 8)):
            logger.warning("interact spam (tile=%d, consecutive=%d) - "
                           "suppressing interact",
                           laya_state["times_on_this_tile"],
                           self._consecutive_interact)
            criteria.pop("interact", None)
            # Only add travel options if we're not following a route/heal
            if not directed:
                for name, (dr, dc) in _DIRS.items():
                    d = name.split("_")[1]
                    if _grid_open(passable, PLAYER_ROW + dr, PLAYER_COL + dc):
                        criteria[f"travel_{d}"] = (
                            f"Head {d} for several tiles - you have stood here "
                            "too long, move on")

        # Dialog cleared - reset the stall counter (after the whole if/elif
        # chain so explore still runs when no dialog is active)
        if not intro and not dlg.get("text_active") and not battle.get("in_battle"):
            self._dialog_turns = 0

        # Gen 1 battle policy: a deterministic move-selection rule beats any
        # model here (exact RAM state, exact pokered type chart). It runs
        # BEFORE the scorer so battles cost ~0ms and never need escalation.
        # When it declines (no damaging move, no PP) the scorer decides.
        policy_choice: Optional[str] = None
        if battle.get("in_battle"):
            try:
                policy_choice = gen1_battle.choose(state)
            except Exception:
                logger.exception("battle policy failed - deferring to scorer")
                policy_choice = None
            if (policy_choice is not None
                    and policy_choice not in criteria
                    and policy_choice.startswith("move_")
                    and "attack" in criteria):
                policy_choice = "attack"       # 0-PP set: FIGHT = Struggle
            # POLICY-TRUST FIX: the policy read the same state and chose a
            # move (e.g. move_0 = Poison Powder PP=14) - if the criteria
            # build saw an empty/partial move list (reader race during
            # battle), TRUST THE POLICY and offer its choice anyway. The
            # deferral left only advance_text, whose A-spam could not
            # navigate the FIGHT menu - 40 minutes frozen vs a Caterpie
            # (proven live 12:03-12:28).
            if (policy_choice is not None and policy_choice not in criteria
                    and policy_choice.startswith("move_")):
                logger.warning("battle policy picked %r not in criteria "
                               "(%s) - reader race, TRUSTING policy and "
                               "offering it", policy_choice, list(criteria))
                criteria[policy_choice] = (
                    "Select FIGHT and choose the policy's move")
            if policy_choice is not None and policy_choice not in criteria:
                logger.warning("battle policy picked %r which is not offered "
                               "(%s) - deferring", policy_choice,
                               list(criteria))
                policy_choice = None

        started = time.perf_counter()
        if policy_choice is not None:
            choice = policy_choice
            self.last_confidence = 1.0
        else:
            choice = self._laya_choose(laya_state, criteria, instructions)
        took_ms = (time.perf_counter() - started) * 1000.0
        if policy_choice is not None:
            logger.info("battle policy=%s (from %d options) took=%.1fms",
                        policy_choice, len(criteria), took_ms)

        # A failed decision must not consume a turn, and must not loop forever.
        if choice is None:
            self.consecutive_fail += 1
            logger.error("laya returned no choice (%d consecutive)",
                         self.consecutive_fail)
            if self.consecutive_fail >= 5:
                # PANIC RECOVERY, not suicide. This used to raise
                # SystemExit and the driver then stayed DEAD until a human
                # noticed - the 13:12 crash left the game frozen on the
                # Route 3 Youngster's dialog for 3.2 hours (13:12-16:27).
                # A stuck decision layer is never worth losing hours: press
                # B to clear any text box/menu, poke A, reset the counter
                # and keep running. The stall/zero-displacement watchdogs
                # police genuine dead ends, and an empty criteria set can
                # no longer happen (see the last-resort fallback above).
                logger.error("laya failed %d turns in a row - PANIC "
                             "RECOVERY: clearing UI and continuing "
                             "(NOT exiting)", self.consecutive_fail)
                try:
                    self.act(["press_b", "wait_30", "press_a", "wait_30",
                              "press_b", "wait_30"])
                except Exception:
                    logger.exception("panic recovery act failed - "
                                     "continuing anyway")
                self.consecutive_fail = 0
                time.sleep(1.0)
                return
            time.sleep(1.0)
            return
        self.consecutive_fail = 0
        # Intent-level interact spam: micro-moves reset the per-tile counter
        # but not this one. A trainer on an edge ate 200+ turns of press_a
        # here (proven live: here=30, pos bouncing between tiles, interact
        # chosen every turn regardless of position).
        if choice == "interact":
            self._consecutive_interact += 1
        else:
            self._consecutive_interact = 0

        # ------------------------------------------------------------------
        # DISPATCH - map the chosen key to emulator actions
        # ------------------------------------------------------------------
        actions: List[str] = []
        # A heal route IS an ordinary map-graph edge hop - normalise it into
        # route_toward_ so it reuses that handler's door-avoiding path and
        # edge-failure tracking instead of duplicating both.
        if choice.startswith("heal_route_"):
            choice = "route_toward_" + choice[len("heal_route_"):]

        if choice == "route_to_waypoint":
            # The unified router already computed the concrete path; execute
            # it verbatim. No cooldown, no compass recomputation, no fallback
            # to "first legal direction" - the path IS the decision. This is
            # what makes oscillation structurally impossible.
            #
            # STALE-PATH GUARD: the stashed path starts at the position
            # WHERE IT WAS COMPUTED. Battles interrupt between compute and
            # dispatch, and the first steps of a stale path walk the wrong
            # direction (proven live: path committed at (20,10) replayed at
            # (20,15) walked LEFT into the wall instead of DOWN to the
            # exit). Re-BFS from the CURRENT position every dispatch; the
            # committed target is the contract, the path is just a hint.
            actions: List[str] = []
            if self._waypoint_target is not None:
                mw_wp = self.map_walk.get(cur_map) or {}
                # Same coherent gate as the router: when the settled
                # wXCoord/wYCoord disagrees with the fused position, the
                # grid was stamped from the settled instant - dispatch
                # BFS must start from the same cell or the path's first
                # step is offset by the race.
                _col_d = (state.get("collision") or {})
                _sett_d = _col_d.get("settled_position") or {}
                if (_col_d.get("valid") and _sett_d
                        and _col_d.get("position_coherent") is False
                        and _sett_d.get("x") is not None):
                    xy_wp = (int(_sett_d["x"]), int(_sett_d["y"]))
                else:
                    xy_wp = (getattr(self, "_last_xy", None)
                             or ((pos.get("x"), pos.get("y"))
                                 if pos.get("x") is not None else None))
                if mw_wp and xy_wp:
                    # F4 (stand-on-tile authority): the player's OWN cell is
                    # the highest-grade walkability evidence that exists -
                    # they are physically standing on it. A camera-edge
                    # artifact that reads the standing tile as blocked used
                    # to veto the ENTIRE fresh BFS here (the route fell
                    # through to the stale tier-2 'via west' stash and the
                    # agent ping-ponged all night, 1971 west routes vs 7
                    # east). Force the standing tile walkable in a BFS COPY
                    # and re-run; live-blocked neighbours are still avoided
                    # via failed_cells below, so this cannot route into a
                    # real wall - and if the first step genuinely bumps,
                    # the zero-displacement watchdog blacklists it as
                    # usual. The game remains the final arbiter.
                    if not mw_wp.get(xy_wp):
                        _mw_bfs = dict(mw_wp)
                        _mw_bfs[xy_wp] = True
                        logger.info(
                            "BFS start-cell authority: standing tile %s "
                            "reads blocked in the fold - forcing walkable "
                            "for routing (stand-on-tile beats stale read)",
                            xy_wp)
                    else:
                        _mw_bfs = mw_wp
                    ledges_wp = self.ledge_edges.get(cur_map or "", {})
                    warp_cells_wp = {(w["x"], w["y"])
                                     for w in (self.gt_warps.get(cur_map)
                                               or [])}
                    warp_cells_wp.update(
                        (int(w.get("map_x")), int(w.get("map_y")))
                        for w in warps if w.get("map_x") is not None)
                    # Cells proven live-blocked on this map: BFS must route
                    # around them (blacklisting targets alone just burns one
                    # target per stall cycle - all their paths share the
                    # same blocked first cell).
                    failed_cells_wp: set = set()
                    self._active_warp_cells = warp_cells_wp
                    # DOOR STEP-THROUGH: when the committed target IS a warp
                    # door, the path must END on it - stepping onto a door
                    # fires the warp. Excluding doors from the path (the old
                    # avoid_warps) made "route to the door" impossible: from
                    # the door tile the only "path" that avoids the door is
                    # walk away and circle back - the (2,7)<->(3,7) bounce,
                    # proven live 07:11-07:15. The avoid set now applies to
                    # every door EXCEPT the target itself.
                    _avoid_wp = ({c for c in warp_cells_wp
                                  if c != tuple(self._waypoint_target)}
                                 if self._waypoint_target is not None
                                 else warp_cells_wp)
                    # NEAR-FIELD TRUTH: overlay the emulator's live 10x10
                    # collision window onto the planning grid before every
                    # dispatch BFS. The fold can claim phantom corridors
                    # (the pokered decode does not match this ROM); the
                    # live window is ground truth, so the path's first
                    # steps can no longer be invented.
                    _live_warp_cells = set()
                    for _w in ((state or {}).get("collision") or {}).get(
                            "warps", []):
                        if _w.get("map_x") is not None \
                                and _w.get("map_y") is not None:
                            _live_warp_cells.add(
                                (int(_w["map_x"]), int(_w["map_y"])))
                    _mw_bfs = self._apply_live_window(
                        cur_map, _mw_bfs, pos, _live_warp_cells)
                    actions = list(_map_step_toward(
                        _mw_bfs, xy_wp, self._waypoint_target, limit=400,
                        ledge_edges=ledges_wp,
                        avoid_warps=_avoid_wp,
                        avoid_cells=failed_cells_wp,
                        map_name=cur_map))
                    # WARP SETTLE (proven live 23:30): standing on a
                    # door/exit tile means the floor transition is
                    # animating. In Gen 1 the warp fires on STEP-ON and
                    # takes ~50 frames; any direction press during the
                    # fade cancels it. The probe mode (single steps) kept
                    # firing walk_down on the Mt Moon 1F west door (14,35)
                    # and the agent stood on the exit for two minutes
                    # without crossing. On a warp tile: press nothing.
                    if (tuple(xy_wp) in warp_cells_wp
                            and self._waypoint_target is not None
                            and tuple(self._waypoint_target) == tuple(xy_wp)):
                        logger.info("warp bounce: on door %s without a "
                                    "transition - stepping off and back on "
                                    "to re-trigger the step-on warp", xy_wp)
                        actions = ["walk_up", "wait_45",
                                   "walk_down", "wait_90"]
                    # WAYPOINT ARRIVAL DETECTION.
                    #
                    # An empty path while standing ON the target is
                    # ARRIVAL, not a stall. Clear the waypoint here instead
                    # of recomputing an empty path: otherwise the driver
                    # re-commits the same target every turn, gets [] back,
                    # dispatches wait_30, and spins forever while tier-1
                    # (the real destination) never gets a look-in. Proven
                    # live: target=(4,8) while pos=(4,8), wait_30 repeated
                    # 1000+ turns in Pewter City's west pocket after a
                    # tier-2 compass target had been committed.
                    #
                    # Skipped for walk-off border cells and warp doors:
                    # standing on those is mid-exit, not arrival, and the
                    # border-completion logic below owns that case.
                    # ------------------------------------------------------
                    if (not actions
                            and self._waypoint_target is not None
                            and not any(m == cur_map
                                        and c == tuple(self._waypoint_target)
                                        for (m, c, _d)
                                        in self._walk_edge_cells)
                            and (xy_wp[0], xy_wp[1])
                            == tuple(self._waypoint_target)
                            and (xy_wp[0], xy_wp[1])
                            not in warp_cells_wp):
                        logger.info(
                            "waypoint arrived: %s on %s - clearing "
                            "(empty path ON the target = arrival)",
                            self._waypoint_target, cur_map)
                        _shelf_a = getattr(self, "_arrived_targets", {}) or {}
                        _shelf_a[(cur_map,
                                  tuple(self._waypoint_target))] = self.turn
                        self._arrived_targets = _shelf_a
                        self._waypoint_target = None
                        self._waypoint_map = None
                        self._waypoint_path = []
                    # BORDER-STEP COMPLETION at the dispatch choke point:
                    # when the committed target is a walk-off border cell,
                    # make sure the batch ends by stepping ONTO it and then
                    # OFF the map in the connection direction. Every
                    # producer (sticky re-offer, fresh BFS, stash) flows
                    # through here, so the completion can't be bypassed
                    # (proven live: the sticky re-offer kept truncating the
                    # batch one tile short of the Route 2 walk-off).
                    _we_d = next((d for (m, c, d)
                                  in self._walk_edge_cells
                                  if m == cur_map
                                  and c == tuple(self._waypoint_target)), None)
                    _appended = False
                    if _we_d and (xy_wp[0], xy_wp[1]) == tuple(self._waypoint_target):
                        # Standing ON a walk-off border cell with a committed
                        # target: the BFS fold reads the border row blocked
                        # and returns garbage detours instead of [] - so the
                        # arrival check never fires and the off-map step
                        # never gets appended. Dispatch it directly (proven
                        # live 12:29, Route 3 (62,0): 8-step detour offered
                        # while standing on the target, bounce (62,0)->
                        #(63,0) for 10 min).
                        _off_m0 = {"north": "walk_up", "south": "walk_down",
                                   "west": "walk_left",
                                   "east": "walk_right"}.get(_we_d)
                        if _off_m0:
                            actions = [_off_m0, "wait_30", _off_m0,
                                       "wait_30", _off_m0]
                    if actions and _we_d:
                        _cx, _cy = xy_wp
                        for _a in actions:
                            _dr_a, _dc_a = _DIRS.get(_a, (0, 0))
                            _cx += _dc_a
                            _cy += _dr_a
                        _tx, _ty = self._waypoint_target
                        _g = 0
                        while ((_cx, _cy) != (_tx, _ty)) and _g < 4:
                            _g += 1
                            _dx, _dy = _tx - _cx, _ty - _cy
                            if abs(_dx) >= abs(_dy) and _dx:
                                _s = "walk_right" if _dx > 0 else "walk_left"
                                _cx += 1 if _dx > 0 else -1
                            elif _dy:
                                _s = "walk_down" if _dy > 0 else "walk_up"
                                _cy += 1 if _dy > 0 else -1
                            else:
                                break
                            actions.append(_s)
                        _off_m = {"north": "walk_up", "south": "walk_down",
                                  "west": "walk_left",
                                  "east": "walk_right"}.get(_we_d)
                        if _off_m and (_cx, _cy) == (_tx, _ty):
                            # Triple-press with waits: a single off-map press
                            # races the edge transition and can be eaten
                            # mid-step (proven live 13:54: ['walk_left',
                            # 'walk_up'] dispatched, the up vanished, the
                            # walker stayed on the border row). One of three
                            # always lands after the step animation ends.
                            actions.extend([_off_m, "wait_30", _off_m,
                                            "wait_30", _off_m])
                            _appended = True
                    # The cap must never truncate the border-completion step:
                    # an 8-step border detour pushed the appended walk_off
                    # past actions[:8], the batch ended ON the border row,
                    # and the crossing never fired (proven live 12:15,
                    # Route 3 (62,0) <-> (63,0) bounce, 10+ min lost).
                    if _appended and len(actions) > 8:
                        actions = actions[:7] + [actions[-1]]
                    else:
                        actions = actions[:8]
            # LIVE-WINDOW VALIDATION: the first DISPATCHED action must step
            # into a cell the emulator's own collision grid calls passable.
            # Static grids (pokered) disagree with live ledges/NPCs on some
            # cells (proven: Route 4 (14,16) tile $1e - walkable per pokered,
            # blocked live). Validate whatever actions ended up dispatched -
            # fresh BFS or the stale-path fallback (the stale path's first
            # step is exactly what got replayed 600+ times live without the
            # check seeing it, because it only ran on the fresh BFS result).
            if not actions:
                actions = list(self._waypoint_path or [])
            # ------------------------------------------------------------------
            # DIALOG DISPATCH GATE - the last line of defence before dispatch.
            #
            # A Gen 1 text box that is WAITING FOR PLAYER INPUT does not set
            # d730 bit 5 (_JOY_IGNORE). The text engine reads the joypad
            # itself and consumes only A/B; it never looks at the d-pad. So
            # direction presses vanish silently, the position never moves,
            # and every "is there a dialog?" signal we used said no - while
            # the agent went on dispatching BFS walk batches computed from a
            # coordinate that had been frozen for 1054 turns (proven live:
            # the Pewter Museum, "Come again!" box on screen, d-pad did
            # nothing, one A press moved the player instantly).
            #
            # Whatever produced this batch - fresh BFS, the sticky waypoint,
            # the stale stash, greedy dead-reckoning, or the _waypoint_path
            # fallback directly above - a live text box makes pressing A the
            # only legal action. Gate it here so no producer can leak one.
            #
            # SWALLOWED-INPUT WATCHDOG (second, decoding-independent arm):
            # if the previous turn dispatched a movement batch and NEITHER
            # the position NOR the facing changed, the game is not reading
            # the d-pad. That is the signature of a text box regardless of
            # whether its text decodes, and of any other silent input lock.
            # Two dead turns in a row is enough to stop trusting movement.
            _facing_now = p.get("facing")
            _obs = ((pos.get("x"), pos.get("y")), _facing_now, cur_map)
            _prev_actions = getattr(self, "_input_probe_actions", None) or []
            _prev_obs = getattr(self, "_input_probe_obs", None)
            _was_move = bool(_prev_actions) \
                and str(_prev_actions[0]).startswith("walk_")
            if _prev_obs is not None and _was_move and _obs == _prev_obs:
                self._input_dead = getattr(self, "_input_dead", 0) + 1
            else:
                self._input_dead = 0
            # BATTLE EXEMPTION: during a battle the player POSITION cannot
            # change - frozen pos is the NORMAL battle signature, not a
            # swallowed-input signal. Without this the gate fired every
            # battle turn (dead_input>=2), replaced the flee/attack batch
            # with A/B, and fought the battle policy's own sequence into a
            # soft-lock (proven live: Mt Moon 1F, 40+ wild-battle cycles,
            # 14 flee turns, zero escapes, player pinned at (10,22) while
            # Zubat healed off Leech Life).
            if battle.get("in_battle"):
                # STALE-BATTLE BREAKER (proven live 04:51): the in_battle
                # flag can stay true while the game actually sits on a
                # post-battle TEXT BOX - the flee batch then does nothing
                # (40+ identical flee turns, HP and enemy frozen) and a
                # plain press_a ends it instantly. Track the battle
                # signature; if it is unchanged for several turns of
                # dispatched battle batches, stop trusting the read and
                # let the A-gate below fire.
                _bsig = (str(battle.get("type")),
                         str((battle.get("enemy") or {}).get("species")),
                         (battle.get("enemy") or {}).get("hp"),
                         (battle.get("active") or {}).get("hp"))
                if getattr(self, "_battle_sig", None) == _bsig and _prev_actions:
                    self._battle_stale = getattr(self, "_battle_stale", 0) + 1
                else:
                    self._battle_stale = 0
                self._battle_sig = _bsig
                self._input_dead = 2 if getattr(self, "_battle_stale", 0) >= 3 else 0
                self._dialog_gate_turns = 0
            _box_up = bool(dlg.get("text_active") or dlg.get("box_visible")
                           or dlg.get("input_locked"))
            # WALL LEARNER (proven live 2026-10-25): a dispatched walk whose
            # cell did not change WHILE THE FACING NOW POINTS THAT WAY is a
            # physical wall - the game read the input and refused the step
            # (a net-zero multi-step batch leaves the facing on its LAST
            # action, so it cannot false-positive here). The fold may show
            # the cell open (ROM/window lie) and the static seed re-opens it
            # every init, so confirmed_blocks is the only authority that
            # sticks - 1F (21-23,26) and B2F (26,12) each wedged the agent
            # for 100+ turns until pinned by hand. The dispatch veto and its
            # veto-escape (4 stand-downs then lift) already guard against a
            # rare false positive.
            if (_was_move and _prev_obs is not None
                    and _prev_obs[0] == _obs[0]
                    and _prev_obs[2] == _obs[2]
                    and not battle.get("in_battle") and not _box_up
                    and p.get("facing")):
                _wd = _DIRS.get(str(_prev_actions[0]))
                if _wd and str(p.get("facing")) == \
                        str(_prev_actions[0]).replace("walk_", ""):
                    _wall = (_obs[0][0] + _wd[1], _obs[0][1] + _wd[0])
                    logger.info("wall learner: %s %s bumped (no move) - "
                                "log only; the ROM decode is the map",
                                cur_map, _wall)
            if (_box_up
                    or self._input_dead >= 2
                    or (battle.get("in_battle")
                        and (battle.get("enemy") or {}).get("hp") == 0)) \
                    and actions \
                    and (not battle.get("in_battle")
                         or (battle.get("enemy") or {}).get("hp") == 0
                         or getattr(self, "_battle_stale", 0) >= 3):
                _gate_key = ((pos.get("x"), pos.get("y")),
                             (dlg.get("text") or "")[:40])
                if _gate_key != getattr(self, "_dialog_gate_key", None):
                    self._dialog_gate_key = _gate_key
                    self._dialog_gate_turns = 1
                else:
                    self._dialog_gate_turns = getattr(
                        self, "_dialog_gate_turns", 0) + 1
                # A first (advances most boxes); after 4 presses that change
                # nothing, switch to B - it closes nearly every Gen 1 box and
                # is the escape for a prompt A cannot advance.
                _press = "press_a" if self._dialog_gate_turns <= 4 \
                    else "press_b"
                # DEAD-INPUT ARM (17:07): when walks stopped moving but no
                # box is reported, the likely culprit is an invisible menu
                # (proven live: the START menu ate every walk at Route 4
                # (19,6) for 150 turns - the dialog flags lag the render).
                # press_b closes menus; press_a would interact with
                # whatever opened it. Only the dead-input arm forces B.
                if not _box_up and self._input_dead >= 2:
                    _press = "press_b"
                # NO-SPEND GUARD. On a Yes/No prompt A means YES - so the
                # blanket press_a above would cheerfully BUY whatever the
                # box is asking. That is not theoretical: Von's wallet went
                # $121 -> $71 (exactly the Museum's 50 ticket) after the
                # attendant's prompt. Gen 1 price prompts carry the amount
                # ("It costs ¥50 for a child ticket."), so when the on-screen
                # text shows a price and the current goal is not itself a
                # purchase, decline with B instead.
                _txt_now = dlg.get("text") or ""
                _spend_prompt = bool(re.search(r"[$¥]\s*\d", _txt_now))
                _goal_l = str(goal or self._plan_goal_text or "").lower()
                _wants_buy = any(w in _goal_l for w in
                                 ("buy", "purchase", "shop", "mart",
                                  "supplies", "poké ball", "poke ball"))
                if _spend_prompt and not _wants_buy and _press == "press_a":
                    logger.info(
                        "no-spend guard: price prompt %r but goal %r is not "
                        "a purchase - pressing B instead of A",
                        _txt_now[:40], _goal_l[:40])
                    _press = "press_b"
                if not any(a.startswith(_press) for a in actions):
                    logger.info(
                        "dialog dispatch gate: box=%s dead_input=%d (n=%d) - "
                        "replacing %s with %s",
                        _box_up, self._input_dead, self._dialog_gate_turns,
                        actions[:3], _press)
                    actions = [_press, "wait_30"]
            else:
                self._dialog_gate_turns = 0
            self._input_probe_obs = _obs
            # NOTE: _input_probe_actions is recorded at the real dispatch
            # point (just before self.act) - the veto and the passability
            # prefix-truncation below both rewrite `actions`, and the
            # swallowed-input watchdog must judge what was ACTUALLY sent.
            # CONFIRMED-BLOCK DISPATCH VETO: whatever produced the batch
            # (fresh BFS, stale stash, greedy), a first step entering a
            # 2x-confirmed physically-blocked cell is refused here - the
            # game will bump, the watchdog will fire, and the loop repeats.
            # Break it at the dispatch gate instead of chasing which
            # producer leaked (proven live: Viridian City (35,13), walk_down
            # x8 replayed 6x after watchdog blacklists).
            if not actions:
                legal = list(_legal_dirs(passable))
                actions = [legal[0]] if legal else ["wait_30"]
            if not actions:
                actions = ["wait_30"]
        elif choice == "advance_text":
            actions = (["a_until_dialog_end"] if dlg.get("text_active")
                       else ["press_a", "wait_30", "press_a"])
        elif choice.startswith("talk_npc_"):
            # GENERALIZED NPC TALK (Vlad's design): find the NPC by name
            # via the live sprite reader, compute its map tile from the
            # player delta, path to the tile BELOW the NPC (Gen 1 NPCs
            # are talked to from the south by default; the sprite reader
            # settles only when the camera is still), face it, press A.
            # The trailing A completes the talk in the same turn when the
            # path is short.
            who = choice[len("talk_npc_"):]
            col_st = (state.get("collision") or {})
            sprites = col_st.get("sprites") or []
            tgt_sprite = None
            for sp in sprites:
                if who.lower() in str(sp.get("who", "")).lower():
                    tgt_sprite = sp
                    break
            if tgt_sprite is None:
                # NPC not on screen: walk toward the map's center of
                # population (the room's open middle) until it appears.
                logger.info("talk_npc: %s not on screen - stepping toward "
                            "room center", who)
                actions = ["walk_up", "wait_30", "walk_left", "wait_30"]
            else:
                npc_r, npc_c = tgt_sprite["row"], tgt_sprite["col"]
                approach = (npc_r + 1, npc_c)   # tile below the NPC
                p_npc = _path_to(passable, approach, ledges=ledge_grid)
                if p_npc:
                    actions = p_npc[:6] + ["press_up", "wait_30",
                                           "press_a", "wait_60"]
                else:
                    actions = (["press_down"] if npc_r == PLAYER_ROW - 1
                               else ["press_a"])
                    actions += ["wait_60"]
        elif choice == "take_starter":
            # The lab table holds three balls at pokered OaksLab 1-based
            # (6,3),(7,3),(8,3) = 0-based (5,2),(6,2),(7,2). Walk to the
            # tile BELOW the nearest ball ((5,3),(6,3),(7,3) 0-based) and
            # press A (the ball sits on the table above the approach tile).
            mw_st = self.map_walk.get(cur_map) or {}
            xy_st = (getattr(self, "_last_xy", None)
                     or ((pos.get("x"), pos.get("y"))
                         if pos.get("x") is not None else None))
            # Live-verified: the player STANDS at y=4 (open walkway); the
            # balls render one row ABOVE (y=3). Approach = y=4, press UP+A.
            balls = [(6, 3), (7, 3), (8, 3)]
            approach = [(6, 4), (7, 4), (8, 4)]
            # If ALREADY standing on an approach tile, face the ball and
            # press A - do NOT path first (pathing away and back caused
            # A-presses on whatever the player happened to face, e.g. the
            # lab Pokedex 'SAVE option is on the MENU' loop).
            if xy_st in approach:
                actions = ["press_a", "wait_60"]
                return
            best: List[str] = []
            if mw_st and xy_st:
                for tgt in approach:
                    mw_st.setdefault(tgt, True)   # table tile may be unseen
                    p_st = _map_step_toward(
                        mw_st, xy_st, tgt, limit=100,
                        ledge_edges=self.ledge_edges.get(cur_map or "", {}))
                    if p_st and (not best or len(p_st) < len(best)):
                        best = p_st
            if best:
                # Path only; the A press happens next turn ON the approach
                # tile (see above) - no blind A presses mid-path.
                actions = best[:8]
            else:
                actions = ["walk_down", "walk_down", "walk_up", "press_a",
                           "wait_60"]
        elif choice == "challenge_leader":
            # Leader is OFF-SCREEN (deep in the gym). Path via the stitched
            # map to the tile in front of the leader's known position, then
            # walk up into them. Brock: pokered PewterGym 1-based (4,1) =
            # 0-based map (3,0); the approach tile is (3,1).
            mw = self.map_walk.get(cur_map) or {}
            xy = getattr(self, "_last_xy", None)
            target = (3, 1) if "pewter" in (cur_map or "").lower() else None
            if mw and xy and target:
                # ARRIVED: standing ON the approach tile. Brock's actual tile
                # comes from the live sprite reader when visible ('N' at
                # (4,1) - directly RIGHT of the approach (3,1), not above:
                # pokered PewterGym Brock is 1-based (5,2) = 0-based (4,1)
                # and the approach is his WEST side). Walk toward the
                # sprite when on screen; fall back to RIGHT (the verified
                # live geometry). The old 'walk_up' guess pressed into the
                # wall for 100+ turns with Brock one tile east.
                if xy == target:
                    _leader_dir = "right"
                    for sp in ((state.get("collision") or {})
                               .get("sprites") or []):
                        sr, sc = sp.get("row"), sp.get("col")
                        if sr is None or sc is None:
                            continue
                        _lx = xy[0] + (sc - PLAYER_COL)
                        _ly = xy[1] + (sr - PLAYER_ROW)
                        if (_lx, _ly) != xy and abs(_lx - xy[0]) \
                                + abs(_ly - xy[1]) <= 1:
                            if _lx > xy[0]:
                                _leader_dir = "right"
                            elif _lx < xy[0]:
                                _leader_dir = "left"
                            elif _ly > xy[1]:
                                _leader_dir = "down"
                            else:
                                _leader_dir = "up"
                            break
                    actions = [f"press_{_leader_dir}", "wait_30",
                               "press_a", "wait_60"]
                    logger.info("challenge_leader: at approach tile - "
                                "facing %s and talking (adjacent NPC: "
                                "walking into a trainer sprite is "
                                "blocked; contact is made by facing "
                                "and pressing A)", _leader_dir)
                else:
                    # Seed the approach tile as walkable even if unseen.
                    mw.setdefault(target, True)
                    path = _map_step_toward(
                        mw, xy, target,
                        ledge_edges=self.ledge_edges.get(cur_map or "", {}))
                    if path:
                        actions = path[:8]
                        logger.info("challenge_leader: %d steps via stitched map",
                                    len(path))
                    else:
                        # No path yet - walk toward the leader's compass direction
                        # to keep stamping new geometry.
                        dx = target[0] - xy[0]
                        dy = target[1] - xy[1]
                        d = ("up" if abs(dy) >= abs(dx) and dy < 0 else
                             "down" if abs(dy) >= abs(dx) else
                             "left" if dx < 0 else "right")
                        actions = [f"walk_{d}"] * 4
            else:
                actions = ["walk_up"] * 4
        elif choice.startswith("challenge_"):
            # Walk INTO the trainer/leader in the named direction. Contact
            # triggers their dialog and then the battle. Cap at 8 steps.
            # If a blind walk stalls (blocked by gym geometry), re-path via
            # the stitched map toward the trainer's MAP position, seeded as
            # walkable so BFS can finish at the tile in front of them.
            d = choice[len("challenge_"):]
            xy = getattr(self, "_last_xy", None)
            if self.turn - getattr(self, "_challenge_started_at", -99) > 3 \
                    and xy is not None:
                # Stalled: find the nearest battle sprite and path to it.
                sprites = col.get("sprites") or []
                _TH = ("super nerd", "cooltrainer", "brock", "rocker",
                       "burglar", "juggler")
                best = None
                for sp in sprites:
                    who = str(sp.get("who", "")).lower()
                    if not any(h in who for h in _TH):
                        continue
                    mx = sp.get("map_x")
                    my = sp.get("map_y")
                    if mx is None or my is None:
                        # derive map pos from screen cell + player offset
                        sr, sc = sp.get("row"), sp.get("col")
                        if sr is None or sc is None:
                            continue
                        mx = xy[0] + (sc - PLAYER_COL)
                        my = xy[1] + (sr - PLAYER_ROW)
                    dist = abs(mx - xy[0]) + abs(my - xy[1])
                    if best is None or dist < best[0]:
                        best = (dist, mx, my)
                if best:
                    _, mx, my = best
                    mw = self.map_walk.get(cur_map) or {}
                    # approach tile: the cell on OUR side of the trainer
                    if mx > xy[0]:
                        approach = (mx - 1, my)
                    elif mx < xy[0]:
                        approach = (mx + 1, my)
                    elif my > xy[1]:
                        approach = (mx, my - 1)
                    else:
                        approach = (mx, my + 1)
                    mw.setdefault(approach, True)
                    mw.setdefault((mx, my), True)
                    path = _map_step_toward(
                        mw, xy, approach,
                        ledge_edges=self.ledge_edges.get(cur_map or "", {}))
                    if path:
                        actions = path[:8]
                        logger.info("challenge: re-path %d steps to trainer "
                                    "at (%d,%d)", len(path), mx, my)
                        self._challenge_started_at = self.turn
                    elif approach == xy:
                        # Already standing in front of the trainer - walk
                        # INTO them (the step from us to their tile).
                        if mx > xy[0]:
                            into = "right"
                        elif mx < xy[0]:
                            into = "left"
                        elif my > xy[1]:
                            into = "down"
                        else:
                            into = "up"
                        actions = [f"walk_{into}", "wait_30", f"walk_{into}"]
                        logger.info("challenge: adjacent to trainer at "
                                    "(%d,%d) - walking %s into them",
                                    mx, my, into)
                        self._challenge_started_at = self.turn
                    else:
                        actions = [f"walk_{d}"] * 8
                else:
                    actions = [f"walk_{d}"] * 8
            else:
                if "_challenge_started_at" not in dir(self) or \
                        self._challenge_started_at < 0:
                    self._challenge_started_at = self.turn
                actions = [f"walk_{d}"] * 8
        elif choice == "menu_down_a":
            actions = ["press_down", "press_a"]
        elif choice == "back_out":
            actions = ["press_b"]
        elif choice == "interact":
            actions = ["press_a"]
        elif choice == "attack":
            # STATE-GATED ATTACK (proven live 18:44): the fixed press chain
            # below assumes the main battle menu is live. When a text box is
            # up instead (the move message, "Got away safely!", the fainted
            # box), its presses leak: after the box clears mid-batch the
            # direction presses become overworld movement and shove the
            # agent off-route (3h of B2F: 1,976 raw x-steps, zero net
            # displacement). If a prose box is up, just advance it and let
            # the next turn run the real attack.
            _txt_a = ((dlg or {}).get("text") or "").upper()
            if ((dlg or {}).get("text_active")
                    and "FIGHT" not in _txt_a):
                actions = ["press_a", "wait_40"]
            else:
                # B first: the cursor persists between turns and may be
                # sitting in ITEM or PKMN. Up+Left drives it to FIGHT from
                # anywhere in the 2x2 menu, since cursor movement clamps at
                # the edges.
                actions = ["press_b", "wait_30", "press_up", "press_left",
                           "press_a", "wait_30", "press_a", "wait_60"]
        elif choice.startswith("move_") and choice[5:].isdigit():
            # Specific move selection (e.g. move_1 = second move in the list).
            # This one cannot be expressed as a fixed action list: the move
            # menu opens on the previously selected slot and wraps, so the
            # right number of presses depends on where the cursor actually is.
            # _select_move opens the menu, reads the cursor and navigates, so
            # it runs the buttons itself and leaves `actions` empty.
            idx = int(choice[5:])
            _mine = (state.get("active_mon") or {})
            self._select_move(idx, _mine.get("moves") or [])
            actions = []
        elif choice == "heal_center":
            # Record the visit: the heal-city tie-break prefers centers
            # not used this run (backtracking through the whole forest to
            # re-heal is never the shorter walk).
            try:
                self._visited_heals.add(((state.get("map") or {})
                                         .get("map_name", ""), "pokecenter"))
            except Exception:
                pass
            # Walk toward the Pokecenter and talk to the nurse. The direction
            # is read from /state each turn rather than baked into the action
            # list, so it stays correct as the agent moves and re-aims.
            src = state.get("nearest_pokecenter_src")
            d = state.get("nearest_pokecenter_dir") or [0, 0]
            try:
                dx, dy = int(d[0]), int(d[1])
            except (TypeError, ValueError, IndexError):
                dx, dy = 0, 0
            dist = state.get("nearest_pokecenter_dist") or 0
            walk = _DELTA_TO_WALK.get((dx, dy))
            if walk is None and (dx or dy):
                # Diagonal delta (e.g. Viridian City's door sits down-right of
                # the arrival point). Take the dominant axis - the next turn
                # recomputes the direction from the new position. Returning {}
                # here made heal_center emit the INSIDE fallback (walk_up) with
                # the agent still in the street, walking into walls forever.
                # On a perfect diagonal (|dx| == |dy|) the dominant axis is a
                # tie; always picking dx made the agent walk right into the
                # wall and back forever. Alternate by turn parity instead.
                if abs(dx) >= abs(dy):
                    if abs(dx) == abs(dy) and self.turn % 2 == 0:
                        walk = "walk_down" if dy > 0 else "walk_up"
                    else:
                        walk = "walk_right" if dx > 0 else "walk_left"
                else:
                    walk = "walk_down" if dy > 0 else "walk_up"
            if walk is None:
                # dx == dy == 0 with dist <= 0: inside the center, nurse at the
                # counter (north of the entrance in every Gen 1 Pokecenter).
                # dist > 0 with a zero delta means the door read was garbage -
                # fall back to the heal route instead of walking up blindly.
                if dist > 0:
                    actions = ["walk_down", "walk_down", "walk_left",
                               "walk_right"]   # reposition; next turn re-reads
                elif self._heal_talk():
                    actions = ["wait_60"]         # healed - next turn re-reads
                else:
                    self._mark_heal_site(cur_map, False)
                    actions = ["walk_down", "wait_30"]
            elif dist <= 1 and src in ("nurse", "inside"):
                # Beside/at the nurse per the builder - still VERIFY, because
                # that reading has been wrong on crowded maps.
                if self._heal_talk():
                    self._mark_heal_site(cur_map, True)
                    actions = ["wait_60"]
                else:
                    self._mark_heal_site(cur_map, False)
                    actions = ["walk_down", "wait_30"]
            elif dist <= 1:
                # The door is adjacent but we are still OUTSIDE - step onto it.
                # Pressing A out in the street does nothing in Gen 1, which is
                # exactly what wedged the agent at (23,26) with Viridian's
                # Pokecenter door 1 tile away (heal_center looping, 45-72 turns
                # on one tile). Only "already beside the nurse" justifies A.
                actions = [(walk or "walk_down"), "wait_30", "wait_30"]
            else:
                # Obstacle-aware walk, not a straight-line guess: Viridian
                # City's door sits up-right of the arrival point with a wall
                # row between, so walk_up x3 hit the wall forever while the
                # flowers swayed. Path to the door's on-screen cell (warps
                # know it, e.g. F2), or to a tile in the delta direction when
                # the door is off-screen. Capped at 3 steps; the next turn
                # re-aims from the new position. No trailing press_a: when the
                # guess is blocked, press_a talks to whoever is in front - the
                # Gambler on the city edge ate the approach with dialogs.
                door_cell = next(((w["row"], w["col"]) for w in warps
                                  if "pokecenter" in (w.get("dest_map_name")
                                                      or "").lower()), None)
                actions = []
                if door_cell is not None:
                    path = _path_to(passable, door_cell, ledges=ledge_grid)
                    if path:
                        actions = path[:4]   # step onto the door is next turn
                if not actions:
                    # Off-screen door: walk toward the delta direction on the
                    # collision grid (open cells only). If the direct step is
                    # blocked - Viridian City has an unbroken wall row between
                    # the agent and its Pokecenter door - fall through to an
                    # open neighbour instead of standing still forever.
                    dr, dc = (0, 0)
                    if "up" in walk: dr = -1
                    elif "down" in walk: dr = 1
                    if "left" in walk: dc = -1
                    elif "right" in walk: dc = 1
                    steps = 0
                    r, c = PLAYER_ROW, PLAYER_COL
                    while steps < 3:
                        nr, nc = r + dr, c + dc
                        if not _grid_open(passable, nr, nc):
                            break
                        actions.append(f"walk_{walk.split('_')[1]}")
                        r, c = nr, nc
                        steps += 1
                if not actions:
                    # Still blocked: try each neighbour, biggest delta axis
                    # first, and take the first direction that is actually open.
                    sy = 0 if dy == 0 else (1 if dy > 0 else -1)
                    sx = 0 if dx == 0 else (1 if dx > 0 else -1)
                    if abs(dy) >= abs(dx):
                        cands = [(sy, 0), (0, sx), (0, -sx), (-sy, 0)]
                    else:
                        cands = [(0, sx), (sy, 0), (-sy, 0), (0, -sx)]
                    for cdr, cdc in cands:
                        if cdr == 0 and cdc == 0:
                            continue
                        name = ("walk_up" if cdr < 0 else "walk_down" if cdr > 0
                                else "walk_left" if cdc < 0 else "walk_right")
                        r, c, run = PLAYER_ROW, PLAYER_COL, []
                        while len(run) < 3:
                            nr, nc = r + cdr, c + cdc
                            if not _grid_open(passable, nr, nc):
                                break
                            run.append(name)
                            r, c = nr, nc
                        if run:
                            actions = run
                            break
                if not actions:
                    actions = [walk] * min(dist - 1, 3)
        elif choice == "flee":
            # STATE-GATED RUN (proven live 18:44): the same leak as the
            # attack gate - on a prose box ("Got away safely!" is the worst
            # because the enemy is ALIVE so the fainted-enemy gate never
            # fires) the down/right presses land on the overworld after the
            # box clears mid-batch, displacing the agent and re-rolling
            # encounters. Advance the text; run the menu chain only when
            # the main battle menu is actually on screen.
            _txt_f = ((dlg or {}).get("text") or "").upper()
            if ((dlg or {}).get("text_active")
                    and "FIGHT" not in _txt_f):
                actions = ["press_a", "wait_40"]
            else:
                # GEN 1 RUN SEQUENCE with settle timing. The main battle
                # menu is 2x2 (FIGHT|ITEM / PKMN|RUN); the cursor returns
                # to FIGHT (top-left) every turn, so RUN is down+right.
                # The old batch had NO waits between the presses: the
                # battle text engine animates for ~30-40 frames after B
                # before the menu is live, so down/right/A landed on
                # text - A advanced the text instead of confirming RUN,
                # and the flee never executed (proven live: 40+ wild-
                # battle cycles at Mt Moon 1F, 14 flee dispatches, ZERO
                # escapes, the loop burning turns while Zubat healed off
                # Leech Life). Each press now gets its own settle wait.
                actions = ["press_b", "wait_45",
                           "press_down", "wait_30",
                           "press_right", "wait_30",
                           "press_a", "wait_90"]
        elif choice == "exit_building":
            # Route to THIS map's exit door via ground truth: BFS to the
            # LAST_MAP warp (the front door of every building). The old
            # blind down/up rotation walked Reds House 1F's up-leg onto
            # the (7,1) stairs and ping-ponged 1F<->2F forever.
            mw_xb = self.map_walk.get(cur_map) or {}
            xy_xb = (getattr(self, "_last_xy", None)
                     or ((pos.get("x"), pos.get("y"))
                         if pos.get("x") is not None else None))
            door = None
            # LIVE door first: this ROM's interiors do NOT match pokered's
            # geometry (the Viridian Pokecenter lesson: gt said west aisle,
            # the ROM has the counter east). The reader's live D cells are
            # the ROM-true exit - gt's door coords are shifted 3 columns
            # here (gt (3,7)/(4,7) vs live (6,7)/(7,7), proven live 17:29:
            # the BFS reached the gt cell, the walk_down stepped into a
            # wall, and the agent oscillated inside the center).
            for w in ((state or {}).get("collision") or {}).get("warps", []):
                if w.get("map_x") is not None and w.get("map_y") is not None:
                    door = (int(w["map_x"]), int(w["map_y"]))
                    break
            if door is None:
                for w in (self.gt_warps.get(cur_map) or []):
                    if w.get("dest_name") is None:  # LAST_MAP = front door
                        door = (w["x"], w["y"])
                        break
            actions: List[str] = []
            if door and mw_xb and xy_xb and mw_xb.get(xy_xb):
                p_xb = _map_step_toward(
                    mw_xb, xy_xb, door, limit=200,
                    ledge_edges=self.ledge_edges.get(cur_map or "", {}),
                    avoid_cells={c for (m, c) in self._failed_targets
                                 if m == cur_map})
                if p_xb:
                    actions = p_xb[:8] + ["walk_down"]
            if not actions:
                # Ground truth lacks the door: fall back to the old
                # rotation (down first covers most Gen 1 exits).
                order = ["walk_down", "walk_down", "walk_up",
                         "walk_left", "walk_right"]
                actions = [order[self.turn % len(order)]] * 2
        elif choice == "catch_pokemon":
            # Strategic wild catch: the goal names a target type and zone.
            # Parse "Catch an Electric-type in Viridian Forest" -> target_type="Electric", zone="Viridian Forest"
            # Walk to the zone's encounter grass, then battle loop with ball throws.
            goal = self._plan_goal_text.lower()
            target_type = None
            for t in ("Electric", "Grass", "Water", "Fire", "Psychic", "Flying", "Bug", "Rock", "Ground", "Poison", "Normal", "Fighting", "Ghost", "Dragon", "Ice"):
                if f"{t.lower()}-type" in goal or f"{t.lower()} type" in goal:
                    target_type = t
                    break
            # Walk to encounter grass in the named zone using route_to_waypoint
            # The planner will set "route_to_waypoint" as the option, then this
            # executor handles the battle + ball throw when a wild mon appears.
            # For now, delegate to frontier exploration which finds grass.
            # The battle policy handles ball throws when the target type matches.
            mw_cp = self.map_walk.get(cur_map) or {}
            xy_cp = (getattr(self, "_last_xy", None)
                     or ((pos.get("x"), pos.get("y"))
                         if pos.get("x") is not None else None))
            if mw_cp and xy_cp:
                # Frontier step toward unexplored grass tiles in this map
                p_cp = _map_frontier_step(mw_cp, xy_cp, limit=200,
                                          ledge_edges=self.ledge_edges.get(cur_map or "", {}))
                if p_cp:
                    actions = p_cp[:6]
                else:
                    actions = ["walk_right"] * 4  # fallback
            else:
                actions = ["walk_right"] * 4
        elif choice == "shop_buy":
            # Poké Mart purchase. Two phases:
            #   1. not at the park tile -> path there, face the clerk, A
            #   2. at the park tile -> run the scripted buy (BUY is the
            #      default menu entry; the wanted item is the first list
            #      entry in every Gen 1 mart; UP maxes the quantity; A
            #      confirms; B backs out)
            # PARK-AND-PRESS (Vlad's design): the talk tile is DETERMINED
            # BY THE MAP (pokered object data), never chased from the
            # sprite read. Every mart clerk is STAY at (0,5) facing
            # RIGHT; the player parks at (1,5) facing LEFT and the game
            # talks across the counter (wTilesetTalkingOverTiles
            # extends the facing range over one counter tile). No
            # dependence on the (biased, flappy) sprite reader.
            sprites = col.get("sprites") or []
            clerk = next((s for s in sprites
                          if "clerk" in str(s.get("who", "")).lower()), None)
            mc = MART_CLERK.get(cur_map)
            if (state.get("dialog") or {}).get("active"):
                # The counter A-press opened a menu/dialog - run the buy
                # script (BUY is the default cursor position).
                actions = list(SHOP_ACTIONS)
            elif mc:
                # Park tile: map (clerk.x + 1, clerk.y) - east of the
                # clerk, on the counter row (screen-agnostic).
                park_map = (mc["x"] + 1, mc["y"])
                px_i, py_i = (pos.get("x") or 0), (pos.get("y") or 0)
                if (px_i, py_i) == park_map:
                    # On the park tile: face LEFT at the clerk and A.
                    actions = ["press_left", "wait_30", "press_a", "wait_60"]
                else:
                    # WINDOW BFS to the park tile: the ROM gt walk grid
                    # for small interiors has a block-orientation bug
                    # (map 42 claims walls where the ROM's own warp
                    # doors sit open - the live window agrees with the
                    # warps). The live window reads the ACTUAL tilemap,
                    # so it wins inside known interiors too.
                    wtr = PLAYER_ROW + (park_map[1] - py_i)
                    wtc = PLAYER_COL + (park_map[0] - px_i)
                    p2 = None
                    if (0 <= wtr < len(passable)
                            and 0 <= wtc < len(passable[wtr] or [])
                            and passable[wtr][wtc]):
                        p2 = _path_to(passable, (wtr, wtc),
                                      ledges=ledge_grid)
                    if p2:
                        actions = p2[:4]
                    elif py_i == park_map[1] and px_i > park_map[0]:
                        # Same row as the clerk, east of the park tile:
                        # the window calls the counter row blocked but
                        # the ROM objects file proves the clerk row is
                        # open floor (the clerk STANDS there). Walk
                        # toward the clerk - the game will stop us at
                        # the park tile.
                        # FROZEN-READ COMPENSATION: one walk_left from
                        # the row east of the park IS the park tile -
                        # the reader may not report the step (coords
                        # lag inside small interiors). After the first
                        # unmoved walk_left, act as if parked: face the
                        # clerk and press A (the game ignores an A with
                        # nothing in front, so it is safe to try).
                        n_left = getattr(self, "_park_lefts", 0)
                        if pos == getattr(self, "_park_prev_pos", None):
                            n_left += 1
                        else:
                            n_left = 0
                        self._park_lefts = n_left
                        self._park_prev_pos = pos
                        if n_left >= 1:
                            actions = ["press_left", "wait_30",
                                       "press_a", "wait_60"]
                            self._park_lefts = 0
                        else:
                            actions = ["walk_left", "wait_30"]
                    else:
                        # Park tile unreachable on the window grid (edge
                        # padding or a stale frame): take the legal step
                        # that shrinks the manhattan distance to it.
                        legal = list(_legal_dirs(passable))
                        if legal:
                            def _d(a):
                                nr = PLAYER_ROW + {"walk_up": -1,
                                                   "walk_down": 1}.get(a, 0)
                                nc = PLAYER_COL + {"walk_left": -1,
                                                   "walk_right": 1}.get(a, 0)
                                tx = px_i + (nc - PLAYER_COL)
                                ty = py_i + (nr - PLAYER_ROW)
                                return (abs(ty - park_map[1])
                                        + abs(tx - park_map[0]))
                            legal.sort(key=_d)
                            actions = [legal[0]]
                        else:
                            actions = ["wait_30"]
            elif clerk:
                cr, cc = clerk.get("row"), clerk.get("col")
                # FALLBACK for marts without ground-truth clerk data:
                # approach the sprite-read clerk tile from below.
                committed = getattr(self, "_shop_approach", None)
                if committed and committed[0] == cur_map:
                    tr, tc = committed[1]
                    p2 = _path_to(passable, (tr, tc), ledges=ledge_grid)
                    if p2:
                        actions = p2[:4]
                    else:
                        # p2 is empty = already standing ON the approach
                        # tile (the path_to start==goal returns []). The
                        # approach is BELOW the clerk, so face UP and
                        # press A to open the shop dialog.
                        actions = ["press_up", "wait_30", "press_a",
                                   "wait_60"]
                else:
                    best = None
                    best_tile = None
                    for tr, tc in [(cr + 1, cc), (cr, cc - 1), (cr, cc + 1),
                                   (cr - 1, cc)]:
                        if not (0 <= tr < len(passable)
                                and 0 <= tc < len(passable[tr] or [])):
                            continue
                        if not passable[tr][tc]:
                            continue
                        p2 = _path_to(passable, (tr, tc), ledges=ledge_grid)
                        if p2 and (best is None or len(p2) < len(best)):
                            best = p2
                            best_tile = (tr, tc)
                    if best:
                        self._shop_approach = (cur_map, best_tile)
                        actions = (best + ["press_a", "wait_60"])
                    else:
                        actions = []
            elif "Mart" in cur_map:
                # Clerk off-screen. Gen 1 marts put the counter on the WEST
                # wall (pokered PewterMart: clerk at x=0, y=5) - walk left
                # until the clerk sprite enters the window, then the branch
                # above takes over. Blind left steps are safe: the counter
                # row blocks before any wall can trap us.
                actions = ["walk_left", "walk_left", "walk_left", "walk_left"]
            else:
                actions = []
            if not actions:
                legal = list(_legal_dirs(passable))
                actions = [legal[0]] if legal else ["wait_30"]
        elif choice.startswith("leave_toward_"):
            # Off-screen exit: walk in the compass direction of the door
            # until it enters the 9x10 window, then the on-screen
            # leave_building_ option takes over.
            d = choice[len("leave_toward_"):]
            actions = [f"walk_{d}"] * 4
        elif choice.startswith("leave_building_"):
            # Interior with no business here: walk to the exit and step on it.
            label = choice[len("leave_building_"):]
            target = next(((w["row"], w["col"]) for w in warps
                           if w.get("cell") == label), None)
            if target is not None:
                actions = _path_to(passable, target, ledges=ledge_grid)
                if actions:
                    actions = actions + ["walk_down"]
            if not actions:
                legal = list(_legal_dirs(passable))
                actions = [legal[0]] if legal else ["wait_30"]
        elif choice == "route_toward_gap":
            # Route 4 upper-half descent: the concrete actions were computed by
            # the stitched-map BFS when the option was offered (the only path
            # from the upper half to the lower half is the x=10..13 gap).
            actions = list(getattr(self, "_route4_gap_path", []) or [])
            if not actions:
                legal = list(_legal_dirs(passable))
                actions = [legal[0]] if legal else ["wait_30"]
        elif choice.startswith("route_toward_"):
            # Map-graph edge hop: follow a path to the far side of the screen
            # in that compass direction, routing around door tiles. A blind
            # walk x8 run steps straight into any building on the way.
            # "route_toward_north" -> walk_up; without this translation the
            # handler fell through to an arbitrary legal direction and the
            # agent alternated walk_up/walk_down on the spot.
            d = COMPASS_TO_WALK.get(choice[len("route_toward_"):])
            if d:
                avoid = {(w.get("row"), w.get("col")) for w in warps
                         if w.get("row") is not None}
                # Prefer the stitched per-map grid. On a winding map the local
                # screen path stops at the first tree and the hop never fires;
                # over the accumulated map it follows the real route to the
                # boundary. Only fall back to the screen path when the map has
                # no better answer.
                mw = self.map_walk.get(cur_map) or {}
                xy = getattr(self, "_last_xy", None)
                map_path = _map_edge_path(
                    mw, xy, d,
                    ledge_edges=self.ledge_edges.get(cur_map or "", {})
                ) if xy else []
                local_path = _path_toward_edge(passable, d, avoid=avoid,
                                               ledges=ledge_grid)
                if len(map_path) > len(local_path):
                    # Cap the batch: sending a 90-step path in one turn makes
                    # the agent overshoot and wander (proven live: the Route 3
                    # ledge route hopped down then the tail walked it back up
                    # into the pocket). Execute a few steps, re-evaluate.
                    actions = map_path[:6]
                    logger.info("map-route %s: %d steps via stitched map "
                                "(local would give %d)", d, len(map_path),
                                len(local_path))
                else:
                    actions = local_path or [f"walk_{d}"] * 8
                # DOOR FALLBACK. In a gate or building the way out is a warp
                # tile that often sits SIDEWAYS of the travel direction - the
                # Route 2 Gate Viridian exit is two tiles RIGHT of the player
                # while the route to Viridian City is "south". Compass
                # walking then pushes into the wall row below and the agent
                # bounces forever. If the compass path is blocked but an
                # on-screen door exists, walk onto the door instead: stepping
                # on a warp tile is what actually changes the map.
                if len(actions) <= 1 and warps:
                    best = None
                    for w in warps:
                        wr, wc = w.get("row"), w.get("col")
                        if wr is None or wc is None:
                            continue
                        p2 = _path_to(passable, (wr, wc), ledges=ledge_grid,
                                      limit=20)
                        if p2 and (best is None or len(p2) < len(best[1])):
                            best = (w, p2)
                    if best is not None:
                        w, p2 = best
                        actions = p2 + ["wait_30"]
                        logger.info("route %s blocked - taking the on-screen "
                                    "door at %s instead", d, w.get("cell"))
                # Remember the hop so the next turn can tell whether this
                # edge actually opens (see the empirical edge check).
                rkey = (self.map_graph.resolve(cur_map) or cur_map
                        if self.map_graph else cur_map)
                # rnext: the map this hop is supposed to reach. The edge check
                # only clears a failure count when we actually land there, not
                # when we wander off somewhere else.
                rnext = None
                if self.map_graph:
                    compass = {v: k for k, v in COMPASS_TO_WALK.items()}.get(d)
                    if compass:
                        rnext = self.map_graph.neighbor_in_direction(
                            cur_map, compass)
                self._route_pending = (rkey, d, rnext)
                self._route_last_pos = (pos.get("x"), pos.get("y"))
                # Mid-map stall guard (proven live: heal route west on Route 3
                # pinned the agent at (18,10) for 100+ turns - the stitched
                # grid showed walk_left open but a ledge row blocked it for
                # real). After 3 consecutive stalls on the same heading,
                # stand the route down and let compass-biased exploration
                # find a way around the obstacle.
                # Pinball-aware stall tracking: a horizontal walk_right
                # ping-pong alternates between two x values with the SAME y
                # ((18,10)<->(22,10) proven live), so exact-position equality
                # resets the counter every other turn and the guard never
                # fires. Track (d, y) instead: same heading + same y + no
                # real progress = stall.
                if (getattr(self, "_rt_stall_d", None) == d
                        and getattr(self, "_rt_stall_y", None) ==
                        pos.get("y")):
                    self._rt_stall_n = getattr(self, "_rt_stall_n", 0) + 1
                else:
                    self._rt_stall_d = d
                    self._rt_stall_y = pos.get("y")
                    self._rt_stall_n = 1
                if self._rt_stall_n >= 4:
                    logger.info("route_toward_%s stalled %dx at (%s,%s) - "
                                "standing down for exploration",
                                d, self._rt_stall_n,
                                pos.get("x"), pos.get("y"))
                    self._rt_stall_n = 0
                    # Cooldown this specific route direction so the agent
                    # explores to find a way around the obstacle.
                    self._route_dir_cooldown = getattr(self, "_route_dir_cooldown", {})
                    # Key by COMPASS name - _graph_travel_options checks the
                    # cooldown with the compass direction, while here d is
                    # the walk direction. Store both to be safe.
                    compass = {v: k for k, v in COMPASS_TO_WALK.items()}.get(d, d)
                    # Remember WHERE the obstacle is so returning to this
                    # spot re-arms the cooldown instantly (proven live: the
                    # 60-turn cooldown expired while the pinball continued,
                    # and the route walked straight back into the wall).
                    self._rt_stall_sites = getattr(self, "_rt_stall_sites", {})
                    self._rt_stall_sites[(rkey, d)] = (pos.get("y"), self.turn)
                    self._route_dir_cooldown[d] = self.turn + 25
                    self._route_dir_cooldown[compass] = self.turn + 25
                    self._heal_cooldown_until = self.turn + 15
                    self._route_pending = None
                    actions = []
                else:
                    # NOTE: the "re-arm cooldown when returning to a known
                    # stall site" idea was removed - the agent returns to the
                    # stall row every turn, so it re-armed forever and the
                    # route was permanently suppressed (proven live: 96 turns
                    # left, forever, at (22,10)). The short cooldown above is
                    # the whole guard: retry the route every 25 turns, and if
                    # a ledge/route is known the BFS path walks it.
                    self._rt_stall_sites = getattr(self, "_rt_stall_sites", {})
                    self._rt_stall_sites[(rkey, d)] = (pos.get("y"), self.turn)
                    self._route_pending = (rkey, d, rnext)
                    self._route_last_pos = (pos.get("x"), pos.get("y"))
            else:
                legal = list(_legal_dirs(passable))
                actions = [legal[0]] if legal else ["wait_30"]
        elif choice.startswith("route_"):
            # Map-graph portal hop: walk to the named door cell and step on it.
            label = choice[len("route_"):]
            target = next(((w["row"], w["col"]) for w in warps
                           if w.get("cell") == label), None)
            if target is None:
                # Door not on this screen (big map, player far away). Project
                # the recorded door back onto the screen using the player
                # position at record time: global_door = screen_cell_at_record
                # - player_at + player_now, then screen_now = global_door
                # - player_now + PLAYER. Falls back to a blind step only when
                # we never recorded where we stood.
                entry = None
                if self.map_graph:
                    okey = self.map_graph.resolve(cur_map) or cur_map
                    entry = ((self.map_graph.portals.get(okey) or {})
                             .get(label))
                if entry and entry.get("player_x") is not None:
                    rec_row = int(entry["row"])
                    rec_col = int(entry["col"])
                    px = int(entry["player_x"])
                    py = int(entry["player_y"])
                    gx = pos.get("x") or 0
                    gy = pos.get("y") or 0
                    # Screen cell and map position relate by
                    #   screen = PLAYER + (map - player)
                    # so the recorded cell implies
                    #   map = recorded_screen - PLAYER + player_at_record
                    # and the same door now sits at
                    #   screen_now = PLAYER + (map - player_now)
                    #              = recorded_screen + (player_at_record - player_now)
                    # The old code dropped the PLAYER offset and computed
                    # rec_row - py + PLAYER_ROW, which is off by PLAYER_ROW
                    # (4) - the projected door landed on a wall, so the agent
                    # ping-ponged on a tile it could never step through.
                    tr = rec_row + (py - gy)
                    tc = rec_col + (px - gx)
                    h = len(passable)
                    wdw = len(passable[0]) if h else 0
                    if 0 <= tr < h and 0 <= tc < wdw:
                        target = (tr, tc)
                    else:
                        # Door is off-screen this turn. Portal cells are
                        # recorded SCREEN-relative at record time, so project
                        # each back to the current screen with the same
                        #   now = recorded + (player_at_record - player_now)
                        # rule before using it as an avoid tile or target.
                        def _proj(p_entry):
                            try:
                                rr = int(p_entry["row"])
                                rc = int(p_entry["col"])
                                ppx = int(p_entry["player_x"])
                                ppy = int(p_entry["player_y"])
                            except (TypeError, ValueError, KeyError):
                                return None
                            return (rr + (ppy - gy), rc + (ppx - gx))
                        if self.map_graph:
                            compass = self.map_graph._portal_direction(
                                cur_map, entry.get("dest_map_name"))
                            if compass:
                                d = COMPASS_TO_WALK.get(compass)
                                if d:
                                    # Avoid doors that lead SOMEWHERE ELSE
                                    # (stepping on one teleports us away), but
                                    # never the target or its sibling cells -
                                    # a gate is recorded as several cells to
                                    # the SAME destination, and avoiding them
                                    # walled off the very approach we need.
                                    tgt_dest = entry.get("dest_map_name")
                                    avoid = set()
                                    for cell, p in (self.map_graph.portals
                                                    .get(cur_map) or {}).items():
                                        if p.get("cell") == label:
                                            continue
                                        if p.get("dest_map_name") == tgt_dest:
                                            continue
                                        pr = _proj(p)
                                        if pr:
                                            avoid.add(pr)
                                    offscreen_actions = _path_toward_edge(
                                        passable, d, avoid=avoid,
                                        ledges=ledge_grid)
                                    if offscreen_actions:
                                        actions = offscreen_actions
                                        target = None
                                # STITCHED-MAP DOOR PATH. The compass walk above
                                # only reaches the screen edge; the real door
                                # may be many tiles away around buildings
                                # (Pewter Mart at (23,17) from (14,10)). The
                                # stitched grid knows the door's MAP cell (the
                                # projection map_door = rec_screen - PLAYER +
                                # player_at is exact), so BFS to it directly.
                                xy2 = getattr(self, "_last_xy", None)
                                if (len(actions) <= 1 and entry
                                        and entry.get("player_x") is not None
                                        and xy2):
                                    door_map = (int(entry["col"]) - 4
                                                + int(entry["player_x"]),
                                                int(entry["row"]) - 4
                                                + int(entry["player_y"]))
                                    mw = self.map_walk.get(cur_map) or {}
                                    if mw:
                                        dp = _map_step_toward(
                                            mw, xy2, door_map,
                                            ledge_edges=self.ledge_edges.get(
                                                cur_map or "", {}))
                                        if dp:
                                            actions = dp
                                            logger.info(
                                                "route %s: %d steps to door "
                                                "map%s via stitched map",
                                                label, len(dp), door_map)
                        # Mark the target door's CURRENT cell walkable so a
                        # later turn paths straight onto it once it is visible.
                        if self.map_graph and target is None:
                            p = (self.map_graph.portals.get(cur_map) or {}).get(label)
                            pr = _proj(p) if p else None
                            if pr:
                                prr, prc = pr
                                if (0 <= prr < len(passable)
                                        and 0 <= prc < len(passable[prr])):
                                    passable[prr][prc] = True
                                    if (ledge_grid and 0 <= prr < len(ledge_grid)
                                            and 0 <= prc < len(ledge_grid[prr])):
                                        ledge_grid[prr][prc] = False
                if target is not None:
                    actions = _path_to(passable, target, ledges=ledge_grid)
                    if actions:
                        actions = actions + ["walk_down"]   # step onto the door
                    # OTHER-DOOR GUARD. The screen path to the projected door
                    # can cross a DIFFERENT building's warp (Pewter Mart door
                    # (23,17) sits between the city centre and the gym door
                    # (16,17)); walking it enters the wrong building. If the
                    # stitched grid has a path to the target door's MAP cell
                    # and it avoids that trap, prefer it.
                    if actions and self.map_graph:
                        entry_q = ((self.map_graph.portals.get(cur_map) or {})
                                   .get(label)) or entry
                        if entry_q and entry_q.get("player_x") is not None:
                            door_map = (int(entry_q["col"]) - 4
                                        + int(entry_q["player_x"]),
                                        int(entry_q["row"]) - 4
                                        + int(entry_q["player_y"]))
                            xy3 = getattr(self, "_last_xy", None)
                            mw = self.map_walk.get(cur_map) or {}
                            if mw and xy3:
                                dp = _map_step_toward(
                                    mw, xy3, door_map,
                                    ledge_edges=self.ledge_edges.get(
                                        cur_map or "", {}))
                                # The screen path enters a warp iff it ends
                                # with a walk onto a cell the reader flags as
                                # a door that is NOT ours. Compare lengths:
                                # take the stitched path when it is longer -
                                # it routes around obstacles by construction.
                                if dp and len(dp) > len(actions) + 2:
                                    actions = dp
                                    logger.info(
                                        "route %s: screen path would cross "
                                        "another door - using %d-step stitched "
                                        "path to map%s", label, len(dp),
                                        door_map)
            # Wedged? Two or fewer distinct tiles across the whole ring means
            # the agent is oscillating, so a 1-step "route" is a ping-pong.
            # Horizontal pinball (Pewter Gym row 2: bouncing x=1..8 with y
            # frozen) also wedges - detect a y-span of 0 across the ring.
            _pr = getattr(self, "_pos_ring", None) or ()
            _ys = {y for (_, y) in _pr}
            _wedged = len(_pr) >= 8 and (len(set(_pr)) <= 2 or len(_ys) <= 1)
            if not actions or (len(actions) <= 1 and _wedged):
                # Dead end. The old fallback took an arbitrary legal step,
                # which on a two-tile pocket just ping-pongs forever (Pewter
                # City's north strip cost 5,100 turns). Walk toward the
                # nearest UNEXPLORED cell instead: every step stamps new
                # geometry, which is the only thing that can weld a severed
                # region back onto the component holding the goal.
                xyf = getattr(self, "_last_xy", None)
                mw = self.map_walk.get(cur_map) or {}
                frontier = _map_frontier_step(mw, xyf) if (mw and xyf) else []
                if frontier:
                    logger.info("route %s unreachable - exploring frontier "
                                "(%d steps to unknown)", label, len(frontier))
                    actions = frontier
                else:
                    logger.warning("no path to route door %s; single step", label)
                    legal = list(_legal_dirs(passable))
                    actions = [legal[0]] if legal else ["wait_30"]
        elif choice.startswith("travel_"):
            # A long run in one direction - enough to cross most of a screen and
            # trigger the map connection at the edge.
            d = choice[7:]
            actions = [f"walk_{d}"] * 6
        elif choice in _DIRS:
            actions = [choice]
        elif choice.startswith("explore_"):
            label = choice[8:]
            try:
                fc = COL_LABELS_LOCAL.index(label[0])
                fr = int(label[1:]) - 1
                actions = _path_to(passable, (fr, fc), ledges=ledge_grid)
                # Remember the attempt so next turn can tell whether it worked.
                self.last_explore_target = (fr, fc)
                self.last_explore_from = (pos.get("x"), pos.get("y"))
            except (ValueError, IndexError):
                actions = []
            if not actions:
                legal = list(_legal_dirs(passable))
                actions = [legal[0]] if legal else ["wait_30"]
        elif choice.startswith("goto_"):
            # One decision, many tiles: BFS over verified walkability turns a
            # per-tile decision loop into a per-destination one.
            label = choice[5:]
            target = next(((w["row"], w["col"]) for w in warps
                           if w.get("cell") == label), None)
            if target is not None:
                actions = _path_to(passable, target, ledges=ledge_grid)
            if not actions:
                logger.warning("no path to %s; taking a single step instead",
                               label)
                legal = list(_legal_dirs(passable))
                actions = [legal[0]] if legal else ["wait_30"]
        elif choice.startswith("talk_oak_"):
            # Walk to the tile beside Oak, then press A to hand over the
            # parcel. The walk covers the approach; the trailing A completes
            # the handoff in the same turn when the path is short.
            label = choice[len("talk_oak_"):]
            try:
                fc = COL_LABELS_LOCAL.index(label[0])
                fr = int(label[1:]) - 1
                actions = _path_to(passable, (fr, fc), ledges=ledge_grid)
            except (ValueError, IndexError):
                actions = []
            if actions:
                actions = actions + ["wait_30", "press_a", "wait_30"]
            else:
                legal = list(_legal_dirs(passable))
                actions = [legal[0]] if legal else ["wait_30"]
        elif choice == "pkmn" or choice == "item":
            # PKMN/ITEM submenu selection. B first (cursor may be in a
            # different menu quadrant), then the submenu position:
            # FIGHT=up+left, PKMN=up+right, ITEM=down+left, RUN=down+right.
            # From anywhere in the 2x2 menu, cursor movement clamps at the
            # edges - so drive to the submenu's corner, then confirm.
            if choice == "pkmn":
                actions = ["press_b", "wait_30", "press_up", "press_right",
                           "press_a", "wait_60"]
            else:
                actions = ["press_b", "wait_30", "press_down", "press_left",
                           "press_a", "wait_60"]
        else:
            logger.warning("unhandled laya choice %r - waiting", choice)
            actions = ["wait_30"]

        logger.info("turn %d: %s=%s (p=%.2f) in %.0fms -> %s (from %d options) "
                    "map=%s pos=(%s,%s) door=%s here=%d stuck=%d",
                    self.turn + 1, self.scorer_name, choice,
                    self.last_confidence, took_ms,
                    actions, len(criteria), cur_map, pos.get("x"), pos.get("y"),
                    on_warp, laya_state["times_on_this_tile"], self.stuck)

        # ---- flight recorder: one machine-readable line per decision --------
        # "Why did it do that at turn N?" should be a jq query, not a grep
        # through millions of chars of interleaved log. This records the whole
        # decision: the option set, the raw PP both sources reported (the
        # battle struct and the party struct), the enemy, and the rationale.
        try:
            _battle = state.get("battle") or {}
            _am = state.get("active_mon") or {}
            _party0 = (state.get("party") or [{}])[0]
            _rec = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "turn": self.turn + 1,
                "scorer": self.scorer_name,
                "choice": choice,
                "confidence": round(self.last_confidence, 3),
                "took_ms": round(took_ms),
                "actions": actions,
                "policy_driven": bool(policy_choice is not None),
                "n_options": len(criteria),
                "options": list(criteria.keys()),
                "map": cur_map,
                "pos": [pos.get("x"), pos.get("y")],
                "door": bool(on_warp),
                "here": laya_state["times_on_this_tile"],
                "stuck": self.stuck,
                "battle": {
                    "in_battle": _battle.get("in_battle"),
                    "type": _battle.get("type"),
                    "sub_type": _battle.get("sub_type"),
                    "menu_index": _battle.get("menu_index"),
                    "menu_max": _battle.get("menu_max"),
                    # --- full state for battle-model training ---------------
                    # The reader (memory/red.py read_battle) already exposes
                    # species, level, HP, STATUS and TYPES for both sides; only
                    # the legacy flat fields were being recorded, so the SFT
                    # harness could not label a decision without re-deriving
                    # enemy types from the species name. Record the structs.
                    "player": {
                        "species": _am.get("species"),
                        "dex_number": _am.get("dex_number"),
                        "level": _am.get("level"),
                        "types": list(_am.get("types") or []),
                        "status": _am.get("status"),
                    },
                    "enemy_full": {
                        "species": (_battle.get("enemy") or {}).get("species"),
                        "dex_number": (_battle.get("enemy") or {}).get("dex_number"),
                        "level": (_battle.get("enemy") or {}).get("level"),
                        "types": list((_battle.get("enemy") or {}).get("types") or []),
                        "status": (_battle.get("enemy") or {}).get("status"),
                        "moves": [
                            [m.get("name"), m.get("pp")]
                            for m in ((_battle.get("enemy") or {}).get("moves") or [])],
                    },
                    # legacy flat fields, kept so existing tooling keeps working
                    "my_hp": [_am.get("hp"), _am.get("max_hp")],
                    # raw PP from the battle struct (what the policy scores)
                    "active_moves": [
                        [m.get("name"), m.get("pp")]
                        for m in (_am.get("moves") or [])],
                    # raw PP from the party struct (the durable copy)
                    "party_moves": [
                        [m.get("name"), m.get("pp")]
                        for m in (_party0.get("moves") or [])],
                    "enemy": [
                        (_battle.get("enemy") or {}).get("species"),
                        (_battle.get("enemy") or {}).get("level"),
                        (_battle.get("enemy") or {}).get("hp"),
                        (_battle.get("enemy") or {}).get("max_hp")],
                },
            }
            with open(TRACE_PATH, "a") as _fh:
                _fh.write(json.dumps(_rec) + "\n")
        except Exception:
            logger.exception("flight recorder write failed")

        # Record what is ACTUALLY being sent, for the swallowed-input
        # watchdog: next turn compares the resulting (position, facing) to
        # this turn's observation. A movement batch that changes neither
        # means the game is not reading the d-pad.
        self._input_probe_actions = list(actions)
        self.act(actions)
        # ZERO-DISPLACEMENT WATCHDOG (outdoors): when a routed batch's first
        # step does not move the player (confirmed next turn with fresh
        # state), the sticky grid lied about the cell that step ENTERED -
        # blacklist THAT cell so the next BFS routes around. This closes the
        # re-commit loop the net-progress stall cannot catch (it drops the
        # TARGET, but tier-2 re-picks a new target whose path starts through
        # the SAME blocked cell; proven live: Route 2 (6,48), 84 visits,
        # walk_up x7 through (6,47) every time). Interior maps keep
        # truncation and need no watchdog.
        if (actions and not self._is_interior(cur_map)
                and actions[0].startswith("walk_")):
            try:
                px_zw, py_zw = (pos.get("x"), pos.get("y"))
                if px_zw is not None and py_zw is not None:
                    dr_zw, dc_zw = _DIRS.get(actions[0], (0, 0))
                    self._zw_pending = {
                        "map": cur_map,
                        # PLAYER ANCHOR: the player's own tile at dispatch
                        # time - the check compares this against the next
                        # turn's position (no move = batch's first step was
                        # live-blocked). The entry cell is derived below.
                        "xy": (int(px_zw), int(py_zw)),
                        "enter": (int(px_zw) + dc_zw, int(py_zw) + dr_zw),
                        "first": actions[0],
                        "at": time.time(),
                    }
                else:
                    self._zw_pending = None
            except Exception:
                self._zw_pending = None
        else:
            self._zw_pending = None
        self.event(type="decision",
                   text=f"[{self.scorer_name} {took_ms:.0f}ms] "
                        f"{choice} → {' · '.join(actions)}")
        # Feed the decision trace so an expert scorer can learn from it.
        scorer_obj = getattr(self, "scorer", None)
        if scorer_obj is not None and hasattr(scorer_obj, "record"):
            scorer_obj.record(laya_state, choice)

        


    # --- one turn ----------------------------------------------------------

    def step(self) -> None:
        emu = self.emulator_state()
        if emu != "ready":
            logger.info("emulator %s - waiting (press START on the dashboard)",
                        emu)
            time.sleep(3)
            return
        try:
            state = self._get("/state").json()
        except Exception as exc:
            logger.error("state read failed: %s", exc)
            time.sleep(2)
            return
        if state.get("status") == "not_ready":
            logger.info("state not ready - waiting")
            time.sleep(2)
            return

        ctx = state.get("context") or {}
        intro = not ctx.get("in_game", False)
        # ZERO-DISPLACEMENT WATCHDOG (check side): did the outdoor batch
        # dispatched LAST turn move the player? The state here is FRESH
        # (read above, after last turn's animation completed), so a equal
        # position means the batch's first step was live-blocked and the
        # sticky grid lied about it. Two consecutive no-move turns on the
        # same cell (animation-timing false-positive guard), then blacklist
        # that cell and drop a target whose path starts there. Trust order
        # is the same as the flicker guard: live beats stale.
        _zw = getattr(self, "_zw_pending", None)
        if _zw and not intro:
            now_xy = (((state.get("player") or {}).get("position")) or {})
            now_map = (state.get("map") or {}).get("map_name")
            moved = (now_map != _zw["map"]
                     or (now_xy.get("x"), now_xy.get("y")) != _zw["xy"])
            if moved:
                self._zw_pending = None
            else:
                # Blacklist the cell the batch tried to ENTER, and correct
                # the sticky grid (the live-window fold re-stamps camera-
                # edge cells walkable every turn, so an in-memory blacklist
                # alone loses to the fold; with 2x zero-displacement
                # confirmation live evidence beats stale).
                cell = _zw["enter"]
                n = getattr(self, "_zw_count", 0) + 1
                self._zw_count = n
                if n >= 2:
                    # OCCUPANCY REFUTATION (F5): the player's CURRENT cell
                    # can never be blacklisted as blocked - they are
                    # standing on it. During the interleaved-driver
                    # incident the watchdog judged batch-vs-position pairs
                    # from TWO different drivers, so "no move" was an
                    # artifact, and 24 false blocks accumulated - including
                    # (5,5), occupied seconds later. Standing on a cell is
                    # a physical refutation stronger than any no-move
                    # inference; also lift any existing block on it.
                    _cur_cell = (((state.get("player") or {})
                                  .get("position")) or {})
                    _cur_xy = (_cur_cell.get("x"), _cur_cell.get("y"))
                    if cell == _cur_xy:
                        logger.info(
                            "zero-displacement watchdog REFUTED: entry-cell "
                            "%s == current position (driver-interleave "
                            "artifact) - not blacklisting, lifting any "
                            "existing block", cell)
                        if _zw["map"] in self.map_walk:
                            self.map_walk[_zw["map"]][cell] = True
                        self._zw_pending = None
                        self._zw_count = 0
                    else:
                        logger.info("zero-displacement watchdog: %s batch from "
                                    "%s produced no move - blacklisting "
                                    "entry-cell %s", _zw["first"], _zw["xy"],
                                    cell)
                        self._failed_targets.add((_zw["map"], cell))
                    # ALSO correct the sticky grid + register the confirmed
                    # block so the live-window fold cannot re-stamp the lie
                    # (the y=0 camera-edge artifact reads the row open every
                    # turn; the shield keeps the correction in place). The
                    # block lifts only if the player is later SEEN standing
                    # on the cell (visited beats everything).
                    if _zw["map"] in self.map_walk:
                        self.map_walk[_zw["map"]][cell] = False
                    # PINS REMOVED (user directive 2026-10-26): the
                    # zero-displacement watchdog no longer persists pins
                    # (they severed the map); the in-session fold close
                    # above + the stall guard handle the block.
                    # ALWAYS drop the stashed path: it was computed through
                    # the now-blacklisted cell, and the dispatch falls back
                    # to it when fresh BFS fails - replaying the poisoned
                    # batch forever (proven live: walk_left x8 through
                    # (15,1) after 3 watchdog blacklists). The target may
                    # stay (BFS reroutes around the cell next offer), but
                    # the stale path must go.
                    self._waypoint_path = []
                    if (self._waypoint_map == _zw["map"]
                            and self._waypoint_target is not None):
                        # Drop the target too when its path STARTS here;
                        # otherwise keep it (the blacklist reroutes it).
                        self._waypoint_target = None
                    self._zw_pending = None
                    self._zw_count = 0
        elif not _zw:
            self._zw_count = 0
        if self.control_state() != "running":
            logger.info("control changed mid-turn - skipping")
            return

        pos = (state.get("player") or {}).get("position")
        # CAMERA-SETTLED GUARD: wYCoord/wXCoord (0xD361/62) only update
        # when a step COMPLETES; mid-step polls see the previous tile.
        # Trusting unsettled reads made the position flip between the
        # old and new tiles across turns (the Mart (0,2)<->(1,2) bounce
        # and every 'soft-lock' where coords froze mid-walk). While the
        # camera is moving, keep the last SETTLED position.
        col_cs = ((state.get("collision") or {}).get("camera_settled", True))
        if not col_cs and self.last_pos is not None:
            logger.debug("camera unsettled - holding last settled pos")
            pos = self.last_pos
        self.stuck = self.stuck + 1 if (pos is not None and pos == self.last_pos) else 0
        self.last_pos = pos
        if self.stuck >= 5:
            logger.warning("stuck %d turns - forcing B to clear any text box",
                           self.stuck)
            self.act(["press_b", "wait_30"])
            self.stuck = 0
            return

        fp = _progress_fingerprint(state)
        if fp != self.progress_fp:
            if self.progress_fp is not None:
                logger.info("progress: %s -> %s", self.progress_fp, fp)
            self.progress_fp = fp
            self.progress_at = time.perf_counter()
            self._sync_objectives(state)

        self._turn(state, intro)

        self.turn += 1
        # NOTE: the zero-displacement watchdog check runs at the TOP of the
        # NEXT step() (see below _zw_pending set in _laya_turn dispatch) -
        # comparing against the state read at THAT point, which reflects the
        # movement caused by the dispatched batch. Checking here would read
        # the same stale pre-dispatch state and fire falsely every turn.
        # Dataset flywheel: after the action phase, record what this turn did.
        # latest_goal is whatever the planner produced this turn (VLM or
        # rules) - that IS the teacher label for the next retrain. When the
        # goal was served from cache (no fresh VLM call this turn) or we are
        # in rules mode, fetch the screen directly so samples still carry a
        # frame - a state+goal pair without the matching image is useless.
        if self._collector is not None:
            try:
                frame = self._latest_frame_png
                if not frame:
                    frame = self._current_frame_png()
                    self._latest_frame_png = frame
                # Flatten the raw server state to the training schema here -
                # _record stores it verbatim, and nested dicts train the VLM
                # on the wrong shape. Same normalization state_to_text uses.
                # facing arrives as a string ('left') or int depending on the
                # reader build - the trainer only saw ints (0=down etc.), so
                # map strings back to the same encoding.
                _facing = (state.get("player") or {}).get("facing") or 0
                if isinstance(_facing, str):
                    _facing = {"down": 0, "up": 4,
                               "left": 8, "right": 12}.get(_facing, 0)
                flat = {
                    "map": ((state.get("map") or {}).get("map_name")
                            or "?"),
                    "badges": int(((state.get("player") or {})
                                   .get("badge_count")) or 0),
                    "x": int((((state.get("player") or {})
                               .get("position")) or {}).get("x") or 0),
                    "y": int((((state.get("player") or {})
                               .get("position")) or {}).get("y") or 0),
                    "facing": int(_facing),
                    "in_battle": bool((state.get("battle") or {})
                                      .get("in_battle", False)),
                    "dialog_open": bool((state.get("dialog") or {})
                                        .get("text_active", False)),
                    "party_hp": [int(m.get("hp") or 0)
                                 for m in (state.get("party") or [])],
                    "party_max_hp": [int(m.get("max_hp") or 0)
                                     for m in (state.get("party") or [])],
                    # Same fields the VLM prompt renders at inference
                    # (planner_client.state_to_text) - training/inference
                    # state shapes must match or the model conditions on a
                    # world it never sees live.
                    "money": int(((state.get("player") or {})
                                  .get("money")) or 0),
                    "bag": [{"item": b.get("item"),
                             "quantity": b.get("quantity")}
                            for b in (state.get("bag") or [])],
                    # Symbolic local map: same block the VLM prompt gets at
                    # inference (planner_client.plan ascii_map arg).
                    "ascii": self._latest_ascii,
                }
                # Teacher label, not the executed goal: a wedged VLM goal
                # recorded as its own training label teaches v4 the wedge.
                # rule_plan is the reference the VLM is meant to imitate.
                try:
                    teacher = rule_plan(flat)
                except Exception:
                    teacher = ""
                self._collector.maybe_record(
                    self.turn, frame, flat,
                    teacher or self._latest_goal_text or "")
            except Exception:
                logger.exception("live collect failed (non-fatal)")
        if self.save_every and self.turn % self.save_every == 0:
            self.save_game()
            self._load_explore_union()

    # --- main loop ---------------------------------------------------------

    def run(self) -> None:
        # SINGLETON LOCK: prevent multiple driver instances from fighting over
        # the same emulator port. Only one driver per server should run.
        import fcntl, os
        # extract port from server URL
        try:
            port = int(self.server.split(":")[-1].split("/")[0])
        except Exception:
            port = 8765
        lock_path = f"/tmp/von_driver_{port}.lock"
        self._lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.error(f"Another driver is already locked on port {port}; exiting.")
            sys.exit(1)
        except Exception as e:
            logger.warning(f"Could not acquire singleton lock ({e}); continuing anyway.")

        if not self.preflight():
            self.event(type="alert", text="preflight failed - see driver log.")
            sys.exit(1)
        brain = f"{self.local_label} + {self.planner_mode} planner"
        print(f"[von] engine up. server={self.server} brain={brain}")
        print("[von] waiting for control=running + an active game...")
        self.event(type="alert", text=f"Von online ({brain}).")

        idle_logged = False
        no_game_logged = False
        while True:
            st = self.control_state()
            if st == "stopped":
                if not idle_logged:
                    print("[von] stopped - idling.")
                    idle_logged = True
                self.last_pos, self.stuck = None, 0
                self.recent.clear()
                time.sleep(2)
                continue
            if st == "paused":
                time.sleep(1.5)
                continue
            idle_logged = False

            self.sync_active_game()
            if not self.game_id:
                if not no_game_logged:
                    print("[von] running but no active game.")
                    self.event(type="alert", text="No active game.")
                    no_game_logged = True
                time.sleep(2)
                continue
            no_game_logged = False
            if self.control_state() != "running":
                continue

            t0 = time.perf_counter()
            try:
                self.step()
            except SystemExit:
                raise
            except Exception:
                logger.exception("turn failed - continuing")
                self.event(type="alert", text="Driver error - see log.")
                time.sleep(3)
            self.laya_time_total += time.perf_counter() - t0
            time.sleep(self.turn_delay)


def run_autopilot(server: str = "http://localhost:8765",
                  model: Optional[str] = None,
                  turn_delay: float = 1.5,
                  turn_timeout: int = 240,
                  save_every: int = 20,
                  debug: bool = False,
                  use_laya: bool = False,
                  laya_narrate_every: int = 0,
                  stall_seconds: float = 60.0,
                  hermes_turns: int = 12,
                  vision: bool = True,
                  replan_every: int = 120,
                  scorer: Optional[str] = None,
                  planner: str = "rules",
                  plan_interval: int = 15) -> None:
    if debug:
        logging.getLogger("pokemon-agent").setLevel(logging.DEBUG)
        logger.info("debug logging enabled")
    model = model or os.environ.get("POKEMON_HERMES_MODEL")
    provider = os.environ.get("POKEMON_HERMES_PROVIDER")
    VonDriver(server, model, provider,
              turn_delay=turn_delay, turn_timeout=turn_timeout,
              save_every=save_every, use_laya=use_laya,
              laya_narrate_every=laya_narrate_every,
              stall_seconds=stall_seconds,
              hermes_turns=hermes_turns,
              vision=vision,
              replan_every=replan_every,
              scorer=scorer,
              planner=planner,
              plan_interval=plan_interval).run()


HermesDriver = VonDriver   # back-compat for existing imports
