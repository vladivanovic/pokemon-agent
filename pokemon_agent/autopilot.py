"""Standalone driver that plays Pokemon through a game session.

Two brains are supported:

  Hermes (default) - a real Hermes Agent session with the `pokemon-player`
    skill, terminal tool, memory, driven one turn at a time via
    `hermes chat --resume`. Slow (seconds to minutes per turn on local
    hardware) but can reason, narrate and set objectives. The web toolset is
    DISABLED - prompts never mention web search and the model does not loop
    on tool_search looking for removed tools.

  Laya (--laya)    - an in-process decision model. Milliseconds per turn,
    returns a structured choice from a fixed candidate set, no text
    generation, no regex parsing, no hallucinated actions. Bypasses Hermes
    entirely. Cannot narrate or plan.

Normal Hermes turns are TEXT ONLY. The ASCII collision map in /state is
ground truth read from game RAM, so it beats asking a vision model to read
pixel art - and it keeps the prompt small, which matters a lot on local
hardware. Hermes can fetch a frame itself (curl /screenshot + its vision
tool) when the map is not enough: menus, dialog text, battle screens. The
driver only pushes an image for the intro screens, where no map exists yet,
and when the player appears wedged.

The loop is gated by the server's /control state (Start/Pause/Stop buttons).

Config (env, optional):
  POKEMON_HERMES_MODEL     model override passed to `hermes chat -m`
  POKEMON_HERMES_PROVIDER  provider override passed to `hermes chat --provider`
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from pokemon_agent.scorer import get_scorer  # pluggable low-level scorer
from pokemon_agent.map_graph import MapGraph    # global map graph + portals
from pokemon_agent.state.builder import PC_CENTERS   # cities with a Pokecenter
from pokemon_agent import gen1_battle           # deterministic Gen 1 battle policy
from pokemon_agent.gen1_data import MOVE_BASE_PP  # max PP per move (pokered)

# Laya decision model - optional. Import failure must be loud at use time,
# not silently equivalent to "feature disabled".
try:
    from laya import Router  # type: ignore
    LAYA_AVAILABLE = True
    _LAYA_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover
    Router = None  # type: ignore
    LAYA_AVAILABLE = False
    _LAYA_IMPORT_ERROR = _exc

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pokemon-agent.autopilot")
# urllib3's DEBUG output logs every single HTTP request to /state (thousands
# per hour), which buried the actual decisions. The replay trace below is the
# real "log everything" mechanism - keep urllib3 at WARNING so the human log
# stays readable.
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Flight recorder - one JSON object per turn, append-only.
#
# The human log is good for a tail, but troubleshooting a specific turn means
# grepping 3.9M characters of interleaved output. This writes a compact,
# machine-readable record of EVERY decision with the full rationale, so a
# question like "why did it flee at turn 1345?" is one jq call.
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


# ---------------------------------------------------------------------------
# Prompts
#
# Keep these byte-stable between turns. llama.cpp reuses the KV cache for a
# shared prefix, so a constant preamble with the volatile parts (map, state)
# at the END is significantly cheaper than a prompt that changes at the top.
# ---------------------------------------------------------------------------

INTRO_VISION_OK = "A screenshot of the current screen is attached - look at it."
INTRO_VISION_NONE = ("No screenshot available. Press A to advance and check "
                     "the result next turn.")

INTRO_NUDGE = """You are booting Pokémon Red on the Hermes Plays Pokémon dashboard.

Server: {server}

The game is NOT in play yet - it is at the title screen, Oak's intro, or a
name-entry menu. Game state values are uninitialised garbage right now, so
IGNORE them entirely. {vision}

Your only job this turn: advance the intro. POST one of these to
{server}/action with -H 'Content-Type: application/json':

  Title screen / NEW GAME       {{"actions":["press_a"]}}
  Oak talking / any text box    {{"actions":["a_until_dialog_end"]}}
  Name menu (NEW NAME/RED/...)  {{"actions":["press_down","press_a"]}}
  Options screen (went too far) {{"actions":["press_b"]}}
  Unsure                        {{"actions":["press_a"]}}

On the name menu do NOT press A on "NEW NAME" - that opens letter-by-letter
entry. Press down first to take a preset.

Do not narrate, do not set objectives yet. Current phase: {phase}

Reply with one short sentence saying what you pressed.

If you need information about the world, read the MAP - it is ground truth
read from game RAM. Do not search the web; the web toolset is disabled."""
TURN_NUDGE = """You are playing Pokémon Red on the Hermes Plays Pokémon dashboard.

Server: {server}

Take ONE short turn:
1. POST {server}/event  {{"type":"reasoning","text":"..."}}     what you see
2. POST {server}/action {{"actions":["walk_down","walk_down"]}}  2-4 moves
3. Reply with ONE short sentence. Be brief.

On a real beat (new town, badge, catch) also POST {server}/event
{{"type":"key_moment","description":"...","category":"milestone|badge|catch"}}

All POSTs need -H 'Content-Type: application/json'.

The MAP below is ground truth read from game memory - trust it over any image.
You are always at @ (cell E5). Columns A-J left to right, rows 1-9 top to
bottom. `.` walkable, `#` blocked, `N` a person blocking you, `D` a door/exit.

If the map says "unavailable", or you are in a menu/battle/dialog and cannot
tell what is on screen, you may look at the frame:
  curl -s '{server}/screenshot' -o /tmp/look.png
then use the vision tool on /tmp/look.png. Only do this when the map and state
are not enough - it costs an extra round trip.

MAP:
{map_ascii}

STATE:
{state}

If you need information about the world, read the MAP - it is ground truth
read from game RAM. Do not search the web; the web toolset is disabled."""
ESCALATION_NUDGE = """You are playing Pokémon Red. {brain} (a fast movement scorer) has
been driving, but nothing has advanced for a while, so YOU have control for the
next {budget} turns. This is turn {n} of {budget}.

GOAL: {goal}

Server: {server}

You know Pokémon Red. {brain} does not - it only picks directions. Use that
knowledge: work out what the game is waiting for, and do it.

NEVER call /load or /save. NEVER load a save state. If movement seems not to
work, it is because a text box is open or an NPC is in the way - not because
the emulator is broken. Press B to clear text, or walk around the obstacle.

Each turn:
1. POST {server}/event {{"type":"reasoning","text":"..."}}  what is blocking us
2. POST {server}/action {{"actions":[...]}}  up to 8 actions - you may send a
   longer sequence than usual since you have the context to plan it
3. If the blocker is cleared and only movement remains, write HANDBACK in your
   reply and {brain} will resume.

EXACT curl commands (copy this shape - do not invent new JSON):
  curl -s -X POST {server}/event -H 'Content-Type: application/json' \\
       -d '{{"type":"reasoning","text":"a wall blocks north, gate is east"}}'
  curl -s -X POST {server}/action -H 'Content-Type: application/json' \\
       -d '{{"actions":["walk_right","walk_right","walk_right"]}}'
The -d body is ONE line of strict JSON: double quotes only, no trailing
commas, no nested quotes. If a call fails, do NOT retry with different
formats - re-read this example.

Warps (doors, stairs, exits) trigger when you WALK onto or off them - pressing
A on a warp tile does nothing. Building exits in this game are on the SOUTH
edge: to leave, walk DOWN off the doormat. `exits` in STATE gives the cell of
each warp; "outside" means it leads out of the building.
'v' is a LEDGE: you can hop DOWN over it, sometimes left or right too
but never climb up. If a ledge blocks, your way north, walk around it - do not keep pressing up.

All POSTs need -H 'Content-Type: application/json'.

COORDINATES - two different systems, do not mix them:
  `position` (x,y) is your absolute location on the map. Larger y is SOUTH.
  Grid cells (A1..J9) are SCREEN positions RELATIVE to you. You are ALWAYS at
  E5. A cell like C3 means "2 columns left, 2 rows up from me". The grid is
  always 10x9 regardless of your position. The grid is NOT misaligned.

BEFORE YOU FINISH, always write one line starting with PLAN: giving the next
concrete sub-goal for the fast movement model to execute. Use compass
directions and named places, not grid cells. Examples:

  PLAN: leave this building, then head north out of Pallet Town onto Route 1
  PLAN: follow Route 1 north, going around any ledges, until Viridian City
  PLAN: inside the Viridian Poke Mart, walk to the counter and talk to the clerk

The fast model cannot read dialog or reason about the story - it only picks
directions. Your PLAN is the only steering it gets, so make it unambiguous.
Write PLAN: on every turn, updating it as the situation changes.

MAP:
{map_ascii}

STATE:
{state}

If you need information about the world, read the MAP - it is ground truth
read from game RAM. Do not search the web; the web toolset is disabled."""

# ---------------------------------------------------------------------------
# State trimming
# ---------------------------------------------------------------------------

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
    # Drop bag entirely - only relevant in menus which we avoid
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
                      map_offset: Optional[tuple] = None) -> List[Tuple[int, int]]:
    """On-screen cells the agent has never stood on, reachable from E5.

    Screen cells map to world coords by offsetting from the player: the player
    is always at (PLAYER_ROW, PLAYER_COL) and at world (px, py).

    When union_global (SB3 explored cells, global (y, x)) and map_offset
    ((mx, my) of the current map) are provided, each candidate's local
    (wx, wy) is converted to global (gy, gx) and checked against the union
    set too — RL-explored terrain is treated as known and stays out of the
    frontier.
    """
    out = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if not _grid_open(passable, r, c):
                continue
            wx = px + (c - PLAYER_COL)
            wy = py + (r - PLAYER_ROW)
            if visits.get((cur_map, wx, wy), 0) != 0:
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

class HermesDriver:
    def __init__(self, server: str, model: Optional[str], provider: Optional[str],
                 turn_delay: float = 1.5, save_every: int = 20,
                 turn_timeout: int = 240, use_laya: bool = False,
                 laya_narrate_every: int = 0,
                 stall_seconds: float = 60.0, hermes_turns: int = 12,
                 vision: bool = True, replan_every: int = 120,
                 scorer: Optional[str] = None):
        self.server = server.rstrip("/")
        self.model = model
        self.provider = provider
        self.turn_delay = turn_delay
        self.save_every = save_every
        self.turn_timeout = turn_timeout
        self.use_laya = use_laya
        self.laya_narrate_every = laya_narrate_every
        self.game_id: Optional[str] = None      # active game session id
        self.session_id: Optional[str] = None   # bound Hermes session id
        self.turn = 0
        self.last_pos: Optional[Any] = None     # stuck detection
        self.stuck = 0
        self.consecutive_fail = 0               # consecutive laya failures
        self.last_confidence = 0
        self.recent: deque = deque(maxlen=8)     # oscillation detection
        self.prev_map: Optional[str] = None
        # Bare direction ("up"/"down"/"left"/"right") we walked to get into the
        # current map. Its reverse is suppressed briefly after arrival so the
        # agent cannot walk straight back through the boundary it just crossed.
        self.arrived_from: Optional[str] = None
        self.visits: Dict[tuple, int] = {}      # (map, x, y) -> times seen
        self.special_tiles: Dict[tuple, str] = {}   # (map, x, y) -> tile purpose
        self.explore_union_path = os.environ.get(
            "POKEMON_EXPLORE_UNION",
            "/home/vlad/pokemon-agent/explore_map_union.json")
        self.union_global: set = set()          # (gy, gx) global coords from SB3
        self.map_offsets: Dict[str, tuple] = {}  # map_name -> (mx, my) for global conv
        self._load_map_offsets()
        # --- global map graph: which way is <city>, through which maps -----
        # Adjacency from map_data.json; portals recorded from RAM warps and
        # persisted. Gives Laya a map-level goal ("head north toward Pewter")
        # the local 10x9 grid can never express.
        try:
            self.map_graph: Optional[MapGraph] = MapGraph()
        except Exception:
            logger.exception("map graph init failed - continuing without it")
            self.map_graph = None
        self.map_changed_at: int = -99
        self.map_changed_time: float = 0.0
        # --- brain arbitration ---
        self.stall_seconds = stall_seconds
        self.hermes_turns = hermes_turns
        self.mode: str = "laya"
        self.mode_turns: int = 0
        self.hermes_budget: int = 0
        self.hermes_goal: str = ""
        self.progress_fp: Optional[tuple] = None
        self.progress_at: float = time.perf_counter()
        self._last_objs: Optional[list] = None
        self.turn_timeout = turn_timeout
        self.use_laya = use_laya
        self.vision = vision          # False for text-only models
        self.failed_targets: Dict[tuple, int] = {}   # (map, r, c) -> failures
        self.last_explore_target: Optional[tuple] = None
        self.last_explore_from: Optional[tuple] = None
        # Anti-pinball: recent positions on the current map. A tiny tile set
        # revisited over many turns with no map change means the agent is
        # bouncing between two boundaries - Route 2's y=60 strip did exactly
        # that for 605 left/right reversals (26.6% of all transitions). Used to
        # withhold the blind travel_ headings so named destinations drive.
        self._recent_pos: list = []
        self._recent_map: Optional[str] = None
        self.session_started_at: int = 0
        self.hermes_time_total: float = 0.0      # seconds spent in Hermes mode
        self.laya_time_total: float = 0.0       # seconds spent in Laya mode
        self.last_mode_switch: float = time.perf_counter()
        self._hermes_last_pos: Optional[Any] = None
        self.hermes_budget_max: int = 30
        # --- empirical edge verification ------------------------------------
        # map_data.json adjacency is geometric, not gameplay truth: Route 2's
        # rectangle touches Pewter City's, but in-game Viridian Forest sits
        # between them (the gates are drawn a tile off Route 2, so the real
        # gate edges never appear). Rather than hand-patch one case, track
        # route hops that never produce a transition and stop offering them -
        # which drops route_active and lets local navigation find the gate.
        self.edge_fails: Dict[tuple, int] = {}
        self._route_pending: Optional[tuple] = None
        self._route_last_pos: Optional[tuple] = None
        # An edge proven dead in play, queued until the goal is known so the
        # graph is only mutated when it can still route without that edge.
        self._pending_dead_edge: Optional[tuple] = None
        # --- low-level scorer: laya | von | expert[:base] ----------------
        # Selectable via --scorer, $POKEMON_SCORER, or config. The scorer
        # replaces Laya's raw router for pick-one-of-N decisions; "expert"
        # wraps the base scorer with a frequency table learned from the
        # agent's own decision log.
        self.scorer_name: str = ((scorer
                                  or os.environ.get("POKEMON_SCORER", "laya"))
                                 .lower())
        # The local (non-Hermes) brain is whichever scorer we were STARTED with,
        # so every user-facing label says "von" when Von is driving. The internal
        # mode value stays "laya" to avoid churning comparisons all over the
        # file, but logs/events must not claim Laya is in control when it is not.
        self.local_label: str = self.scorer_name
        self.scorer: Optional[Any] = None       # lazy-init on first use
        # Vision-capable models get the vision toolset; text-only models (e.g.
        # Qwen GGUF) must not - vision_analyze 500s and the retry loop blows
        # the turn timeout.
        self.vision_capable: bool = vision
        # --- planning: Hermes sets intent, Laya executes it ---
        self.standing_plan: str = ""
        self.plan_set_at: int = -999
        self.plan_from_map: str = ""
        self.plan_ttl: int = 150
        self.replan_every: int = replan_every
        
        self.laya_router: Optional[Any] = None
        if self.use_laya:
            if not LAYA_AVAILABLE:
                raise SystemExit(
                    f"--laya requested but `import laya` failed: {_LAYA_IMPORT_ERROR!r}\n"
                    f"check you are in the venv where laya is installed")
            try:
                logger.info("loading Laya (first run downloads ~1GB)…")
                t0 = time.perf_counter()
                self.laya_router = Router(preload=True)  # type: ignore[misc]
                logger.info("Laya ready in %.1fs", time.perf_counter() - t0)
            except Exception:
                logger.exception("Laya router preload failed")
                raise SystemExit(
                    "Laya could not load. Pre-download it with:\n"
                    "  rm -rf ~/.cache/huggingface/hub/models--convaiinnovations--laya\n"
                    "  HF_HUB_ENABLE_HF_TRANSFER=1 hf download convaiinnovations/laya")
        self._load_special_tiles()
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

    def _needs_heal(self, state: Dict[str, Any]) -> str:
        """Why we should go to a Pokecenter, or "" when the party is fine."""
        for p in (state.get("party") or []):
            if isinstance(p, dict) and (p.get("hp") or 0) <= 0:
                return "a party member has fainted"
        am = state.get("active_mon") or {}
        hp, mx = am.get("hp"), am.get("max_hp")
        if hp is not None and mx and (hp / mx) < 0.35:
            return f"HP is low ({hp}/{mx})"
        if am.get("moves"):
            # Reuse the battle policy's own ranking: if it finds no damaging
            # move with PP, the party cannot win a fight and 15 turns of Growl
            # is not a strategy. (A blackout would "fix" this for free, but it
            # costs money and warps the agent backwards.)
            ranked = gen1_battle.score_moves(state)
            damaging = [e for e in ranked if e[1] > 0]
            if not damaging:
                return "no damaging move has PP left"
            # Top up BEFORE the tank runs dry. Healing only at 0 PP meant the
            # agent always fought down to its last PP, so a wild battle ended
            # "one move, then RUN" (proven live: watched vs a level-4 Pidgey)
            # and the walk back started from empty. Keep a 25% reserve of the
            # move's TRUE max PP, compiled from pokered (MOVE_BASE_PP); fall
            # back to the highest PP ever observed for that move, which is the
            # max whenever the party has been healed at least once.
            best = damaging[0][2] or {}
            best_name = str(best.get("name") or "")
            best_pp = int(best.get("pp", 0) or 0)
            maxima = getattr(self, "_move_max_pp", None)
            if maxima is None:
                maxima = self._move_max_pp = {}
            if best_name and best_pp > maxima.get(best_name, 0):
                maxima[best_name] = best_pp          # fresh PP is the new max
            top = MOVE_BASE_PP.get(best_name.lower()) or maxima.get(best_name, 0)
            if top and best_pp <= max(3, top // 4):
                return (f"{best_name} PP is nearly exhausted "
                        f"({best_pp}/{top})")
        return ""

    def _heal_options(self, state: Dict[str, Any],
                      cur_map: Optional[str] = None) -> Dict[str, str]:
        """Criteria entry for going to heal, or {} when not warranted.

        Only offered when there is a real reason, so it cannot crowd out normal
        play. Two shapes:

        * a center on THIS map -> ``heal_center``, walk to it and talk to the
          nurse (sources come from annotate_nearest_pokecenter: the nurse
          sprite is screen-relative, the pc_centers.json door is map-relative
          and works off-screen, "inside" means we have already arrived);
        * otherwise ``heal_route_<compass>``, heading for the nearest city that
          has one. Without this a low-HP party in Viridian Forest has nowhere
          to go and simply wanders until it blacks out.
        """
        reason = self._needs_heal(state)
        if not reason:
            # Goal satisfied (or no longer needed) - reset the stall watchdog.
            self._heal_best_dist = None
            self._heal_stall = 0
            return {}
        # A heal goal that stops making progress is a WEDGE, not patience.
        # Proven live: the agent cycled heal_center for 130+ of 272 turns in
        # Viridian City (45-72 turns on a single tile, position frozen at
        # (23,12)) because the Pokecenter door is 13 tiles SOUTH with an
        # unbroken wall row between us, and the local 3-step walk cannot route
        # around it. After N turns with no reduction in distance, stand down
        # for a while and let normal exploration run - exploration covers
        # ground, and the walk can finish from a position where it works.
        if self.turn < getattr(self, "_heal_cooldown_until", 0):
            return {}
        src = state.get("nearest_pokecenter_src")
        dist = state.get("nearest_pokecenter_dist")

        if isinstance(dist, (int, float)):
            best = getattr(self, "_heal_best_dist", None)
            if best is None or dist < best:
                self._heal_best_dist = dist
                self._heal_stall = 0
            else:
                self._heal_stall = getattr(self, "_heal_stall", 0) + 1
                if self._heal_stall >= 10:
                    self._heal_cooldown_until = self.turn + 40
                    self._heal_best_dist = None
                    self._heal_stall = 0
                    logger.warning(
                        "heal goal stalled after 10 turns with no progress "
                        "(still %s tiles away) - standing down for 40 turns "
                        "and letting exploration run", dist)
                    return {}

        if src is not None and dist is not None:
            d = state.get("nearest_pokecenter_dir") or [0, 0]
            try:
                dx, dy = int(d[0]), int(d[1])
            except (TypeError, ValueError, IndexError):
                dx, dy = 0, 0
            where = {"nurse": "the nurse is on screen",
                     "door": "a Pokecenter door is known on this map",
                     "inside": "you are already inside a Pokecenter"}.get(src, src)
            if dist <= 0:
                return {"heal_center":
                        f"HEAL - {reason}. {where}: step up to the counter and "
                        f"press A to talk to the nurse, then accept the heal."}
            walk = _DELTA_TO_WALK.get((dx, dy))
            if walk is None and (dx or dy):
                # A diagonal delta (e.g. (1,1) for Viridian City's door, which
                # sits down-right of the arrival point) has no single walk.
                # Take the dominant axis - the next turn recomputes the
                # direction from the new position, so the other axis follows.
                # Returning {} here instead made heal_center vanish and the
                # heal_route fallback fire forever.
                if abs(dx) >= abs(dy):
                    walk = "walk_right" if dx > 0 else "walk_left"
                else:
                    walk = "walk_down" if dy > 0 else "walk_up"
            if walk is None:
                return {}
            return {"heal_center":
                    f"HEAL - {reason}. {where}, about {dist} tiles away. Walk "
                    f"{walk.split('_')[1]} toward it and press A at the nurse."}

        # No center on this map - head for the nearest city that has one.
        if not (self.map_graph and cur_map):
            return {}
        best = None
        for city in PC_CENTERS:
            if not self.map_graph.resolve(city):
                continue
            r = self.map_graph.route(cur_map, city)
            if r and (best is None or len(r) < len(best[1])):
                best = (city, r)
        if not best:
            return {}
        city, chain = best
        compass = self.map_graph.first_hop_direction(cur_map, city)
        if not compass or compass not in COMPASS_TO_WALK:
            return {}
        # Respect the empirical edge blacklist. A direction we have proven
        # does not open (e.g. Forest/south, the fake edge) must not be
        # re-offered forever - the heal route would just repeat it.
        cur_key = self.map_graph.resolve(cur_map) or cur_map
        walk = COMPASS_TO_WALK.get(compass)
        if self.edge_fails.get((cur_key, walk), 0) >= 4:
            logger.info("heal route %s->%s via %s is a proven dead edge - "
                        "trying an alternative", cur_key, city, compass)
            # Find another route that starts in a different direction.
            for city2 in PC_CENTERS:
                if city2 == city or not self.map_graph.resolve(city2):
                    continue
                r2 = self.map_graph.route(cur_map, city2)
                if not r2 or len(r2) < 2:
                    continue
                c2 = self.map_graph.first_hop_direction(cur_map, city2)
                w2 = COMPASS_TO_WALK.get(c2) if c2 else None
                if (c2 and w2 and self.edge_fails.get((cur_key, w2), 0) < 4):
                    best = (city2, r2)
                    compass, walk = c2, w2
                    break
            else:
                return {}
        return {f"heal_route_{compass}":
                f"HEAL - {reason}. There is no Pokecenter on {cur_map}: head "
                f"{compass} toward {city} to heal. Route: {' -> '.join(chain)}."}

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
                               stdin=subprocess.DEVNULL, timeout=120)
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

    def _enter_mode(self, mode: str) -> None:
        now = time.perf_counter()
        elapsed = now - self.last_mode_switch
        if self.mode == "hermes":
            self.hermes_time_total += elapsed
        elif self.mode == "laya":
            self.laya_time_total += elapsed
        self.mode, self.mode_turns = mode, 0
        self.recent.clear()
        self.progress_at = now   # grace period in the new mode
        self.last_mode_switch = now
        self.event(type="decision",
                   text=f"[brain] switched to "
                        f"{self.local_label if mode == 'laya' else mode}")
        self._post_stats()

    def _infer_goal(self, state: Dict[str, Any]) -> str:
        """A coarse objective for the escalated turns, from game state alone.

        Only the current blocker is described, never the whole walkthrough -
        the escalated model has limited context and a long plan dilutes the
        one thing it needs to do right now.
        """
        flags = state.get("flags") or {}
        party = state.get("party") or []
        badges = flags.get("badge_count", 0)
        map_name = (state.get("map") or {}).get("map_name", "?")

        if not party:
            if map_name == "Oak's Lab":
                return (
                    "Take a starter Pokemon from the table in Oak's Lab. "
                    "The three Pokeballs sit ON THE TABLE - stand so you are "
                    "facing a ball and press A, then confirm YES. "
                    "Your rival Gary is nearby and gives you nothing; the "
                    "`people` field in STATE shows who is where. "
                    "If Oak is blocking or still talking, press A to let him "
                    "finish first."
                )
            return ("Walk to Oak's Lab in Pallet Town and get your first "
                    "Pokemon. The lab is the large building in the south of town.")

        if not flags.get("has_pokedex"):
            if flags.get("has_oaks_parcel"):
                return ("You are carrying Oak's Parcel. Return to Oak's Lab in "
                        "Pallet Town and give it to Oak to receive the Pokedex. "
                        "Follow the walkable path shown on the MAP - do not "
                        "assume a compass direction is open.")
            return ("Reach Viridian City and collect Oak's Parcel from the Poke "
                    "Mart clerk. Viridian lies beyond Route 1, north of Pallet "
                    "Town, but the path is not a straight line: read the MAP "
                    "and follow whichever directions are actually open, even "
                    "if that means going west or east first.")

        if badges == 0:
            # Shopping first: the forest road to Pewter is a meat grinder with
            # an empty bag - no Poke Balls means wild encounters are flee-only,
            # and no Potions means every Caterpie bite sticks. Only trigger in
            # Viridian City (the Mart is THERE); buying is handled by the
            # dedicated shop handler once inside.
            bag = state.get("bag") or []
            ball_count = sum(e.get("quantity", 0) for e in bag
                             if "ball" in str(e.get("item", "")).lower())
            potion_count = sum(e.get("quantity", 0) for e in bag
                               if "potion" in str(e.get("item", "")).lower())
            if (ball_count == 0 or potion_count == 0) and "Viridian" in map_name:
                need = []
                if ball_count == 0:
                    need.append("Poke Balls")
                if potion_count == 0:
                    need.append("Potions")
                return ("You have no " + " or ".join(need) + ". Enter the Viridian "
                        "Poke Mart, face the clerk behind the counter, press A, "
                        "and buy " + " and ".join(need) +
                        " (you have enough money). Then leave the Mart and "
                        "head north to Pewter City through Viridian Forest. "
                        "Brock uses Rock types - a Grass or Water Pokemon helps.")
            return ("Reach Pewter City and beat Brock at the Gym for the Boulder "
                    "Badge. Route from Viridian City: north through Viridian "
                    "Forest. Brock uses Rock types - a Grass or Water Pokemon "
                    "helps. Train your party to about level 12 or higher first.")

        # General case: no scripted hint. Tell it where it has already been so
        # it can pick somewhere new, rather than writing a walkthrough branch
        # for every stage of the game.
        seen = sorted({m for (m, _, _) in self.visits if m})
        hint = f" Maps visited so far: {', '.join(seen)}." if seen else ""
        return (f"You have {badges} badge(s) and appear stuck in {map_name}. "
                f"Work out what the game is waiting for and do it.{hint}")

    def _graph_destination(self, goal: str) -> Optional[str]:
        """Extract the map a goal names, using the map graph's own keys.

        'Reach Pewter City' -> 'Pewter City'; 'Walk to Oak's Lab in Pallet
        Town' -> 'Pallet Town'. When a goal names several maps - the badge
        hint says "Route from Viridian City: north through Viridian Forest to
        Pewter" - the one right after a goal verb (reach/to/at/enter/...) is
        the destination; the others are waypoints and must not win.
        """
        if not goal or self.map_graph is None:
            return None
        low = goal.lower()
        hits = [n for n in self.map_graph.maps if n and n.lower() in low]
        if not hits:
            return None

        # A map whose name is preceded by a goal-intent word is the target.
        intent = re.compile(
            r"(?:reach|reaching|to|toward|towards|at|enter|visit|arrive|"
            r"arrive at|head to|walk to|get to|go to|return to)\s+$")
        best: Optional[str] = None
        best_pos = -1
        for n in hits:
            idx = low.find(n.lower())
            prefix = low[:idx]
            if intent.search(prefix) and idx > best_pos:
                best, best_pos = n, idx
        if best is not None:
            return best
        return max(hits, key=len)

    def _graph_travel_options(self, cur_map: str, goal: str,
                              passable: List[List[bool]],
                              ledge_grid: Optional[List[List[bool]]]) -> Dict[str, str]:
        """Map-level travel candidates for the current map.

        Two flavours, both named so the scorer sees a clear goal rather than
        an anonymous direction:
          - route edge : the next map is a neighbour you leave by walking off
            the map boundary in a compass direction ("head north out of Pallet
            Town toward Pewter City").
          - portal     : the next map is reached through a door on this map;
            the candidate points at that door's cell.
        The full remaining route is spelled out so the model can keep going
        after each transition instead of re-deciding from scratch.
        """
        out: Dict[str, str] = {}
        if not cur_map or self.map_graph is None:
            return out
        dest = self._graph_destination(goal)
        cur_key = self.map_graph.resolve(cur_map) or cur_map
        if not dest or dest == cur_key:
            return out
        route = self.map_graph.route(cur_map, dest)
        if not route or len(route) < 2:
            # No graph path. The map may still be reachable in game but
            # disconnected in the graph: Viridian Forest's only known
            # neighbours are two dead-end gates, and its real exits are edge
            # connections with no warp cells to learn from. Fall back to
            # geography - head in the compass direction of the destination and
            # let the empirical edge check retire a direction that never opens.
            if not dest or dest not in self.map_graph.maps:
                return out
            geo = self.map_graph._direction_between(cur_key, dest)
            compass = (_DELTA_TO_COMPASS.get((geo[0], geo[1]))
                       if geo else None)
            walk = COMPASS_TO_WALK.get(compass) if compass else None
            if compass and walk and self.edge_fails.get((cur_key, walk), 0) < 4:
                out[f"route_toward_{compass}"] = (
                    f"Head {compass} toward {dest}. No mapped route exists from "
                    f"{cur_map} (its only known links are dead ends), but "
                    f"{dest} lies {compass} of here - keep going {compass}.")
            return out
        nxt = route[1]
        chain = " -> ".join(route)
        # Prefer a door if one exists: stepping onto a known warp cell is
        # deterministic, whereas walking off an edge can miss the seam.
        for cell, portal in (self.map_graph.portals.get(cur_key) or {}).items():
            if (self.map_graph.resolve(portal.get("dest_map_name")) or
                    portal.get("dest_map_name")) == nxt:
                out[f"route_{cell}"] = (
                    f"Walk to the door at {cell} on the MAP, step onto it to "
                    f"enter {nxt}. Route: {chain}.")
                break
        else:
            d = self.map_graph.first_hop_direction(cur_map, dest)
            # Skip an edge we have empirically proven does not open. The
            # blacklist is keyed by WALK direction (up/down/left/right) - the
            # handler records it that way - while first_hop_direction speaks
            # compass, so translate before the lookup.
            if d:
                d_walk = COMPASS_TO_WALK.get(d)
                if d_walk and self.edge_fails.get((cur_key, d_walk), 0) >= 4:
                    logger.info(
                        "route %s->%s via %s is a proven dead edge - falling "
                        "back to local navigation", cur_key, dest, d)
                    d = None
            if d:
                out[f"route_toward_{d}"] = (
                    f"Head {d}, walking off the edge of {cur_map} to reach "
                    f"{nxt}. Route: {chain}. Keep moving {d} - do not turn "
                    f"back into {cur_map}.")
        return out

    def _active_plan(self, state: Dict[str, Any]) -> str:
        """The current standing plan, or '' if it has expired.

        A plan is retired when it gets old, or when the map changes enough that
        it no longer describes the situation - "head north out of Pallet Town"
        is actively misleading once you are in Viridian.
        """
        if not self.standing_plan:
            return ""
        if self.turn - self.plan_set_at > self.plan_ttl:
            logger.info("plan expired after %d turns: %s",
                        self.turn - self.plan_set_at, self.standing_plan)
            self.standing_plan = ""
            return ""
        cur = (state.get("map") or {}).get("map_name", "")
        # Two map changes past the issuing map means the plan probably
        # succeeded and is now stale.
        if cur != self.plan_from_map and cur not in self.standing_plan:
            hops = len({m for (m, _, _) in self.visits
                        if m and m != self.plan_from_map})
            if hops >= 2:
                logger.info("plan superseded by map change: %s",
                            self.standing_plan)
                self.standing_plan = ""
                return ""
        return self.standing_plan

    def _arbitrate(self, state: Dict[str, Any], made_progress: bool) -> str:
        """Decide which brain drives this turn.

        Laya is the default: ~500x cheaper and good at movement. Hermes takes
        over when nothing has advanced for a while and KEEPS control for a
        budget of turns, so it can finish a multi-step errand instead of being
        cut off mid-sequence.
        """
        stalled_for = time.perf_counter() - self.progress_at

        # Mechanical problems have mechanical solutions - but not if Laya has
        # been failing at this one for a while. Don't let the shortcut become
        # a deadlock.
        if (self.mode == "laya"
                and (state.get("collision") or {}).get("player_on_warp")
                and stalled_for < self.stall_seconds * 2):
            return "laya"

        if self.mode == "hermes":
            self.mode_turns += 1
            cur_pos = (state.get("player") or {}).get("position")
            moved = cur_pos is not None and cur_pos != self._hermes_last_pos
            self._hermes_last_pos = cur_pos

            if self.mode_turns >= 2 and not (state.get("battle") or {}).get("in_battle"):
                logger.info("hermes yielded after %d turns (battle over) - "
                            "back to %s", self.mode_turns, self.local_label)
                self._enter_mode("laya")
            elif (moved and self.mode_turns >= self.hermes_budget - 2
                    and self.hermes_budget < self.hermes_budget_max):
                # Crossing a route changes no fingerprint field, so a long trek
                # would otherwise be cut off mid-journey. Reward movement.
                self.hermes_budget = min(self.hermes_budget + 4,
                                         self.hermes_budget_max)
                logger.info("hermes still moving - budget extended to %d",
                            self.hermes_budget)
            elif self.mode_turns >= self.hermes_budget:
                logger.warning("hermes budget (%d turns) exhausted - back to %s",
                               self.hermes_budget, self.local_label)
                self._enter_mode("laya")
            return self.mode

        if (state.get("battle") or {}).get("in_battle"):
            # Battles are resolved by the deterministic Gen 1 policy
            # (gen1_battle) behind the scorer: exact RAM state + exact
            # pokered type chart, ~0ms. A language model adds ~60s per turn
            # and cannot beat a correct lookup table here, so battles no
            # longer escalate to Hermes - it stalls out on tool calls and
            # wedges the game. If Hermes was driving, hand back. progress_at
            # is refreshed because being in a battle is not a stall.
            self.progress_at = time.perf_counter()
            if self.mode == "hermes":
                logger.info("battle started - handing control back to the "
                            "local scorer (Gen 1 policy)")
                self._enter_mode("laya")
            return "laya"

        self.mode_turns += 1
        # Two escalation triggers historically: a stall (Laya wedged) and a
        # stale plan (re-planning on a timer). The small local model (Qwen3.5)
        # burns its whole budget on tool-retry loops during WALK turns - each
        # escalation was 50-200s of mostly-wasted wall clock - so normal
        # navigation no longer escalates at all. Von handles movement; a stale
        # plan only rotates the standing-plan text, it does not hand over.
        # Hermes/Qwen is reserved for BATTLES (handled by the deterministic
        # Gen 1 policy behind the scorer anyway) and a future manual
        # escalate/deescalate button in the Web UI.
        plan_age = self.turn - self.plan_set_at
        if self.standing_plan and plan_age > self.replan_every:
            logger.info("plan stale after %d turns - awaiting next escalation "
                        "(no auto handover)", plan_age)
        return self.mode

    # --- Laya --------------------------------------------------------------
    #
    # ALL Laya-specific API usage is confined to _laya_choose(). If the real
    # signature differs from what is assumed here, this is the only method
    # that needs changing.

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
        if scorer_name == "laya":
            return self._laya_router_choose(laya_state, criteria,
                                            instructions)
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

    def _laya_turn(self, state: Dict[str, Any], intro: bool) -> None:
        """One Laya-driven turn: pick from a legal candidate set and execute.

        Branch layout (referred to by name elsewhere):
          BRANCH A - DIALOG : a text box or the intro is on screen
          BRANCH B - BATTLE : in a battle menu
          BRANCH C - EXPLORE: free movement in the overworld
            C1 - on-warp     : standing on a door; leaving takes priority
            C2 - frontier    : multi-tile paths to unvisited cells
            C3 - single steps: fallback when the frontier is empty
        """
        col = state.get("collision") or {}
        passable = col.get("passable") or col.get("walkable") or []
        ledge_grid = col.get("ledges") or None
        dlg = state.get("dialog") or {}
        battle = state.get("battle") or {}
        p = state.get("player") or {}
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
                n = self.edge_fails.get((rmap, rdir), 0) + 1
                self.edge_fails[(rmap, rdir)] = n
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
        if intro or dlg.get("text_active"):
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
            if hp_frac < 0.35 or (_no_damaging_pp and _is_wild):
                criteria["flee"] = "Select RUN and escape this battle"
            criteria["advance_text"] = "Press A to advance battle text"
            criteria["pkmn"] = "Select PKMN to switch Pokémon"
            criteria["item"] = "Select ITEM to use an item"
        # ------------------------------------------------------------------
        # BRANCH C - EXPLORE
        # ------------------------------------------------------------------
        else:
            times_here = self.visits.get(key, 0)
            # Hermes' plan beats the static hint: it was written with knowledge
            # of what actually blocked us, and in compass terms Laya can use.
            plan = self._active_plan(state)
            goal = plan or self._infer_goal(state)
            seen_maps = {m for (m, _, _) in self.visits if m}

            # --- apply a proven-dead edge, but never strand the route --------
            # An edge that repeatedly fails to transition is fake (Route 2 ->
            # Pewter City looks adjacent by rectangle, but in-game you go via
            # the gates and Viridian Forest). Dropping it lets route() find the
            # real chain. Only do it if the graph can STILL reach the goal
            # without the edge: removing a load-bearing edge turned route()
            # into None and dumped the agent onto blind travel headings, which
            # is the pinball we are fixing.
            if self._pending_dead_edge and self.map_graph is not None:
                src_m, dst_m = self._pending_dead_edge
                self._pending_dead_edge = None
                self.map_graph.dead_edges.add((src_m, dst_m))
                if goal and self.map_graph.route(src_m, goal):
                    self.map_graph.save_dead_edges()
                    logger.warning(
                        "edge %s -> %s proven dead; graph still reaches %s "
                        "without it - dropping the edge and re-planning",
                        src_m, dst_m, goal)
                    for k in [k for k in self.edge_fails if k[0] == src_m]:
                        self.edge_fails.pop(k, None)
                else:
                    self.map_graph.dead_edges.discard((src_m, dst_m))
                    logger.warning(
                        "edge %s -> %s is dead but removing it would strand "
                        "the route to %s - keeping the old suppress-the-hop "
                        "behaviour instead", src_m, dst_m, goal)

            # --- C1: on-warp ---
            # Standing on a door. The tile you must step toward is the
            # building's OUTER WALL, correctly marked non-walkable, so
            # _legal_dirs will never offer it. The engine checks warps before
            # collision, so terrain does not apply here - offer every
            # direction, and keep the option set small so leaving is not
            # competing against six exploration choices.
            if on_warp and not recent_transition:
                instructions = (
                    f"OBJECTIVE: {goal}\n\n"
                    "You are STANDING ON a door or exit. To leave, step OFF it "
                    "- for a building exit that means walking DOWN, into what "
                    "the map shows as a wall. The engine moves you through. "
                    "Pressing A on a door does nothing."
                )
                criteria["exit_building"] = (
                    "Step DOWN off this door to leave the building")
                for name in _DIRS:
                    criteria[name] = f"Step one tile {name.split('_')[1]}"
                criteria["interact"] = "Press A to talk to whoever is in front"

            else:
                # The objective is the single most important thing Laya was
                # missing: without it, "walk back into Oak's Lab" and "head
                # north to Viridian" look equally reasonable.
                instructions = (
                    f"OBJECTIVE: {goal}\n\n"
                    "You are exploring Pokemon Red. The MAP shows what is "
                    "around you: '.' walkable, '#' wall, 'N' a person blocking "
                    "you, 'D' a door or exit, 'v' a ledge you can only hop "
                    "DOWN over, '@' you. You have stood on this exact tile "
                    f"{times_here} time(s). Choose the option that best serves "
                    "the OBJECTIVE above - not merely the nearest unexplored "
                    "tile.")

                # Suppress the warp we just came through. Blocking only the
                # named goto_ option is not enough: walking onto the tile
                # triggers it too.
                if recent_transition and warps:
                    passable = [list(r) for r in passable]   # copy before mutating
                    for w in warps:
                        wr, wc = w.get("row"), w.get("col")
                        if (wr is not None and wc is not None
                                and 0 <= wr < len(passable)
                                and 0 <= wc < len(passable[wr])):
                            passable[wr][wc] = False

                # --- C2c: map-graph route (computed first) ---
                # A named, multi-map goal ("head north toward Pewter City,
                # route Pallet -> Route 1 -> Viridian -> Route 2 -> Pewter")
                # outranks anonymous local wandering. This is the one thing
                # the local grid cannot express, and the reason Laya/Von
                # circled town instead of leaving it. When a route exists we
                # suppress the interchangeable local options below: Von's
                # probabilities are flat (~0.15 across 11 choices), so
                # leaving "explore B2" / "enter Blue's House" in the set is
                # what makes it wander into buildings instead of leaving.
                # --- C2h: heal (outranks the route) ---
                # A crippled party cannot finish the route, so going to heal
                # beats making progress toward the goal. Offered only when
                # _needs_heal finds a reason and the state located a center on
                # THIS map. Sets `directed` so the interchangeable local
                # options below get suppressed exactly as they are for a named
                # route - Von's probabilities are flat, so leaving "explore B2"
                # in the set is what makes it wander instead of leaving.
                heal_opts = self._heal_options(state, cur_map)

                try:
                    route_opts = self._graph_travel_options(
                        cur_map, goal, passable, ledge_grid)
                except Exception:
                    logger.exception("graph travel options failed")
                    route_opts = {}
                route_active = bool(route_opts)
                # Heal outranks route: if we need to heal, we do NOT offer the
                # normal route. The route handler would just send us toward the
                # goal with a crippled party. Instead we ONLY offer the heal
                # route, which includes the full chain to the nearest center.
                if heal_opts:
                    criteria.update(heal_opts)
                    logger.info("heal active: %s", list(heal_opts))
                elif route_active:
                    criteria.update(route_opts)
                    logger.info("route active: %s", list(route_opts))
                # Either a named route or an errand to heal means the local
                # wandering options are noise, not choices.
                directed = route_active or bool(heal_opts)

                # --- C2: frontier ---
                # Multi-tile destinations beat interchangeable single steps:
                # Laya scores ~0.33 on symmetric directions and ~0.99 on named
                # destinations, and one decision covers 5-10 tiles.
                frontier = _frontier_targets(
                    passable, self.visits, cur_map,
                    pos.get("x") or 0, pos.get("y") or 0,
                    union_global=self.union_global or None,
                    map_offset=self.map_offsets.get(cur_map))
                # Farthest first - a neighbouring unvisited tile is barely a
                # choice; "walk to the far corner" is a plan.
                frontier.sort(key=lambda rc: -(abs(rc[0] - PLAYER_ROW)
                                               + abs(rc[1] - PLAYER_COL)))
                added = 0
                for fr, fc in frontier:
                    if directed:
                        break             # route supersedes local wandering
                    if added >= 3:
                        break
                    if self.failed_targets.get((cur_map, fr, fc), 0) >= 2:
                        continue          # proven unreachable
                    path = _path_to(passable, (fr, fc), ledges=ledge_grid)
                    if not path:
                        continue
                    label = cell_label_local(fc, fr)
                    # Name the compass direction: "walk 7 tiles north" serves an
                    # objective in a way that "walk 7 tiles to B2" does not.
                    bearing = _bearing(fr, fc)
                    criteria[f"explore_{label}"] = (
                        f"Walk {len(path)} tiles {bearing} to {label}, "
                        f"never visited")
                    added += 1

                # --- C2b: travel ---
                # Route transitions are NOT warps - you leave a town or cross a
                # route by walking off the map edge. Without these options Laya
                # literally cannot express "go north", so it can only shuffle
                # within the current map.
                #
                # PINBALL GUARD: these headings are interchangeable by design
                # (each reads "head <dir> for several tiles"), so when the agent
                # is bouncing between two boundaries nothing distinguishes them
                # and it never stops. Route 2's y=60 strip produced 605
                # left/right reversals - 26.6% of every transition recorded.
                # If the last 12 turns covered <=3 tiles, withdraw them and let
                # the frontier options, which name a destination, drive instead.
                #
                # Detect REVERSALS (A -> B -> A), not just a small tile count:
                # the Viridian City pinball drifted across two adjacent pairs,
                # 4 distinct tiles, and slipped straight through a <=3 test.
                recent = self._recent_pos
                pinned = _pinballed(recent)
                if pinned and not directed:
                    logger.info(
                        "pinball on %s (%d tiles across %d turns) - "
                        "withholding blind travel headings and door targets",
                        cur_map, len(set(recent)), len(recent))
                # Which heading actually leads to the goal, when the graph
                # knows. Without this every travel_ option reads identically
                # and the scorer has no reason to pick the useful one.
                goal_compass = None
                if self.map_graph is not None and goal:
                    try:
                        goal_compass = self.map_graph.first_hop_direction(
                            cur_map, goal)
                    except Exception:
                        goal_compass = None
                for name, (dr, dc) in _DIRS.items():
                    if directed or pinned:
                        break       # named route / pinball guard supersedes
                    d = name.split("_")[1]
                    if not _grid_open(passable, PLAYER_ROW + dr,
                                      PLAYER_COL + dc):
                        continue
                    toward = (f" - this heads toward {goal}"
                              if goal and d == goal_compass else "")
                    criteria[f"travel_{d}"] = (
                        f"Head {d} for several tiles, continuing in that "
                        f"direction to reach a new area{toward}")

                # --- C2d: leave building ---
                # An interior (not in the map graph) where the goal does not
                # live is a dead end, but the seen_maps guard below suppresses
                # its exit door, so the agent wanders room-to-room until it
                # stumbles onto the warp. Offer the exit explicitly: path to
                # the nearest warp cell and step onto it.
                in_graph = bool(self.map_graph and
                                self.map_graph.resolve(cur_map))
                # If the goal names THIS building, its business is here -
                # leaving would abandon the errand (Oak's Lab / the parcel).
                goal_here = bool(cur_map and goal
                                 and cur_map.lower() in goal.lower())
                if (not directed and not on_warp and warps
                        and not in_graph and not goal_here):
                    best = None
                    for w in warps:
                        wr, wc = w.get("row"), w.get("col")
                        if wr is None or wc is None:
                            continue
                        path = _path_to(passable, (wr, wc), ledges=ledge_grid)
                        if path and (best is None or len(path) < len(best[1])):
                            best = (w, path)
                    if best is not None:
                        w, path = best
                        dest = (w.get("dest_map_name") or
                                f"map {w.get('dest_map')}")
                        criteria[f"leave_building_{w.get('cell')}"] = (
                            f"Walk to the exit at {w.get('cell')} and step "
                            f"onto it to return to {dest} - this building "
                            f"has nothing you need for the objective")

                # --- C3: single steps ---
                # Always offered, not only when the frontier is empty: if every
                # frontier path is blocked mid-route the agent still needs a way
                # to nudge past the obstruction.
                # EXCEPTION: when a named route or heal route is active, single
                # steps are noise that lets Von pick walk_down over the route.
                # The route handler already emits the full walk sequence.
                if not directed:
                    ledge_grid = col.get("ledges") or [[False]*10 for _ in range(9)]
                    for name in _legal_dirs(passable):
                        dr, dc = _DIRS[name]
                        # Going up (north) through a ledge is blocked; going down (south)
                        # through a ledge is allowed (engine handles the wrap-through).
                        nr, nc = PLAYER_ROW + dr, PLAYER_COL + dc
                        if dr < 0 and 0 <= nr < 9 and 0 <= nc < 10 and ledge_grid[nr][nc]:
                            continue  # skip moving up through a ledge
                        criteria[name] = f"Move one tile {name.split('_')[1]}"

                # Do not offer the boundary we just crossed. In a map we have
                # to TRAVERSE (Viridian Forest) the reverse walk was exactly
                # what the agent kept choosing - it entered and fell straight
                # back out, forever, which is the "shy" behaviour. Bounded by
                # the recent_transition window so a real dead end is not a
                # trap, and skipped if it is the only legal move left.
                if recent_transition and self.arrived_from:
                    back = "walk_" + _OPPOSITE_WALK.get(self.arrived_from, "")
                    others = [k for k in criteria if k in _DIRS and k != back]
                    # Healing outranks the arrival guard: if the walk back is
                    # toward a heal target (heal_center or heal_route walking
                    # through that boundary), keep it. A crippled party that
                    # entered Viridian City from the north has its Pokecenter
                    # to the north-west; suppressing north leaves it nowhere
                    # to go but back out south, forever.
                    heal_keys = [k for k in criteria
                                 if k.startswith("heal_")]
                    if back in criteria and others and not heal_keys:
                        criteria.pop(back, None)

                # Do not offer interact when a named route or heal route is active.
                # The heal route already includes the full walk sequence; pressing A
                # on a trainer is never correct when we're fleeing to heal.
                if not directed:
                    criteria["interact"] = "Press A to talk to or examine what you face"

                # Named exits elsewhere on screen (not the one underfoot).
                # Suppressed while a map-graph route is active: doors into
                # buildings are the exact distraction that made the agent
                # wander (and its own named route may already be a door).
                if not recent_transition and not directed:
                    for w in warps[:3]:
                        cell = w.get("cell")
                        if not cell:
                            continue
                        if (w.get("row"), w.get("col")) == (PLAYER_ROW, PLAYER_COL):
                            continue          # handled by C1
                        dest = w.get("dest_map_name") or f"map {w.get('dest_map')}"
                        # A door into somewhere already explored is a trap: it is
                        # a named, distinct option so it scores well, and it
                        # undoes progress. Oak's Lab was being re-entered
                        # repeatedly for exactly this reason.
                        if dest in seen_maps:
                            continue
                        # Quest-aware skip: don't offer doors to buildings whose
                        # business is already done. Carrying the parcel means the
                        # Mart is pointless; having the Pokedex means the Lab is.
                        # Names must match MAP_NAMES in memory/red.py exactly:
                        if dest == "Viridian Mart" and has_parcel and not has_dex:
                            continue   # deliver the parcel to Oak FIRST
                        if dest == "Oak's Lab" and has_dex:
                            continue   # lab business complete
                        if pinned:
                            # Ping-ponging beside doors means we cannot
                            # actually reach them: Viridian City produced a
                            # goto_E1 <-> goto_E2 alternation at (32,11)/(32,10).
                            # Withhold door targets so a reachable frontier
                            # destination drives instead.
                            continue
                        criteria[f"goto_{cell}"] = (
                            f"Enter the door at {cell} leading to {dest}")

                # NPC-targeted handoff: inside the lab with the parcel, the
                # frontier treats "wander" and "approach Oak" as equal, so the
                # agent loops in and out without ever facing him. Offer a named
                # walk-to-Oak option aimed at the tile BESIDE the Prof. Oak
                # sprite - standing on him is impossible (he blocks), so the
                # target is the passable cell adjacent to his position.
                if (cur_map == "Oak's Lab" and has_parcel
                        and not has_dex and not on_warp):
                    sprites = col.get("sprites") or []
                    oak = next((s for s in sprites
                                if s.get("who") == "Prof. Oak"), None)
                    if oak:
                        orow, oc = oak.get("row"), oak.get("col")
                        # Adjacent passable cell, prefer facing-up approach
                        # (Oak reads dialogs from the tile below him).
                        candidates = [(orow + 1, oc), (orow, oc - 1),
                                      (orow, oc + 1), (orow - 1, oc)]
                        placed = False
                        for tr, tc in candidates:
                            if not (0 <= tr < len(passable)
                                    and 0 <= tc < len(passable[tr] or [])):
                                continue
                            if not passable[tr][tc]:
                                continue
                            path = _path_to(passable, (tr, tc),
                                            ledges=ledge_grid)
                            if not path:
                                continue
                            label = cell_label_local(tc, tr)
                            criteria[f"talk_oak_{label}"] = (
                                f"Walk {len(path)} tiles to {label}, the tile "
                                "beside Prof. Oak, then press A to hand over "
                                "the parcel")
                            placed = True
                            break
                        if placed:
                            # The handoff outranks generic exploration: drop
                            # explore_ options so Laya scores the Oak option
                            # against movement only.
                            criteria = {k: v for k, v in criteria.items()
                                        if not k.startswith("explore_")}
        if not criteria:
            instructions = "Nothing is possible right now. Press B."
            criteria["back_out"] = "Press B"

        has_badges = int((state.get("flags") or {}).get("badge_count") or 0)

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
            "plan": self._active_plan(state) or None,
            "plan_age": (self.turn - self.plan_set_at
                         if self.standing_plan else None),
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
        if exploring:
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
                raise SystemExit(
                    "Laya failed 5 turns in a row - see the traceback above. "
                    "The question schema is probably wrong.")
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

        if choice == "advance_text":
            actions = (["a_until_dialog_end"] if dlg.get("text_active")
                       else ["press_a", "wait_30", "press_a"])
        elif choice == "menu_down_a":
            actions = ["press_down", "press_a"]
        elif choice == "back_out":
            actions = ["press_b"]
        elif choice == "interact":
            actions = ["press_a"]
        elif choice == "attack":
            # B first: the cursor persists between turns and may be sitting in
            # ITEM or PKMN. Up+Left drives it to FIGHT from anywhere in the 2x2
            # menu, since cursor movement clamps at the edges.
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
            self._select_move(idx, mine.get("moves") or [])
            actions = []
        elif choice == "heal_center":
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
                else:
                    actions = ["walk_up", "press_a", "wait_60"]
            elif dist <= 1 and src in ("nurse", "inside"):
                actions = ["press_a", "wait_60"]        # already beside her
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
            actions = ["press_b", "wait_30", "press_down", "press_right",
                       "press_a", "wait_60"]
        elif choice == "exit_building":
            # South covers nearly every Gen 1 building exit. Rotate on repeat
            # attempts so a north/side exit is eventually found.
            order = ["walk_down", "walk_down", "walk_up",
                     "walk_left", "walk_right"]
            actions = [order[self.turn % len(order)]] * 2
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
                path = _path_toward_edge(passable, d, avoid=avoid,
                                         ledges=ledge_grid)
                actions = path or [f"walk_{d}"] * 8
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
                    # global door position (x=col axis, y=row axis)
                    gdx = rec_col - px + gx
                    gdy = rec_row - py + gy
                    # back to screen-relative for the CURRENT position
                    tr = gdy - gy + PLAYER_ROW
                    tc = gdx - gx + PLAYER_COL
                    h = len(passable)
                    wdw = len(passable[0]) if h else 0
                    if 0 <= tr < h and 0 <= tc < wdw:
                        target = (tr, tc)
                    else:
                        # Door is off-screen. Use _path_toward_edge with the
                        # correct compass direction instead of projecting.
                        # Figure out which compass direction this door is in.
                        if self.map_graph:
                            compass = self.map_graph._portal_direction(cur_map, entry.get("dest_map_name"))
                            if compass:
                                d = COMPASS_TO_WALK.get(compass)
                                if d:
                                    # Use map_graph portals for avoid set (persistent),
                                    # but EXCLUDE the target door so we can step on it.
                                    avoid = set()
                                    for map_name, portals in self.map_graph.portals.items():
                                        if map_name != cur_map:
                                            continue
                                        for cell, p in portals.items():
                                            if p.get("cell") == label:
                                                continue  # don't avoid the target door
                                            try:
                                                wr = int(p.get("row", 0))
                                                wc = int(p.get("col", 0))
                                            except (TypeError, ValueError):
                                                continue
                                            avoid.add((wr, wc))
                                    offscreen_actions = _path_toward_edge(passable, d, avoid=avoid,
                                                                        ledges=ledge_grid)
                                    if offscreen_actions:
                                        # Use these actions directly - they lead toward the off-screen gate
                                        actions = offscreen_actions
                                        target = None  # ensure we don't try _path_to
                                # Also patch the target gate tiles as walkable in passable grid
                                # so future turns can path through the gate once reached
                                if self.map_graph:
                                    for cell, p in self.map_graph.portals.get(cur_map, {}).items():
                                        if p.get("cell") == label:
                                            try:
                                                wr = int(p.get("row", 0))
                                                wc = int(p.get("col", 0))
                                            except (TypeError, ValueError):
                                                continue
                                            if 0 <= wr < len(passable) and 0 <= wc < len(passable[0]):
                                                passable[wr][wc] = True
                                                if ledge_grid and 0 <= wr < len(ledge_grid) and 0 <= wc < len(ledge_grid[0]):
                                                    ledge_grid[wr][wc] = False
                if target is not None:
                    actions = _path_to(passable, target, ledges=ledge_grid)
                    if actions:
                        actions = actions + ["walk_down"]   # step onto the door
            if not actions:
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
                    "menu_index": _battle.get("menu_index"),
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

        self.act(actions)
        self.event(type="decision",
                   text=f"[{self.scorer_name} {took_ms:.0f}ms] "
                        f"{choice} → {' · '.join(actions)}")
        # Feed the decision trace so an expert scorer can learn from it.
        scorer_obj = getattr(self, "scorer", None)
        if scorer_obj is not None and hasattr(scorer_obj, "record"):
            scorer_obj.record(laya_state, choice)

        
    # --- Hermes ------------------------------------------------------------

    def _hermes_turn(self, state: Dict[str, Any], intro: bool,
                     goal: str = "", budget: int = 0, n: int = 0) -> None:
        """One Hermes-driven turn: build a prompt, shell out, let it act."""
        # Replaying a 1000-message history dominates latency and buys nothing:
        # the escalation goal is regenerated from live state every turn.
        # Rotation at 4 turns: the CLI repairs message-alternation violations
        # on EVERY resume, and the violation count grows with history
        # (1 -> 3 proven live), so a long session gets slower each turn.
        if self.session_id and (self.turn - self.session_started_at) > 4:
            logger.info("rotating hermes session (was %d turns old)",
                        self.turn - self.session_started_at)
            self.session_id = None
            self.session_started_at = self.turn

        ctx = state.get("context") or {}
        col = state.get("collision") or {}
        map_ascii = col.get("ascii") or (
            f"(map unavailable: {col.get('reason', 'not built')})")

        # Push an image only when there is no usable map: the intro screens, or
        # when we appear wedged. Ordinary turns are text-only and cheap; Hermes
        # can curl a frame itself when it decides it needs one.
        img_path = str(Path(tempfile.gettempdir()) / "pokemon_turn.png")
        have_img = False
        if self.vision and intro:
            have_img = self._fetch_frame("/screenshot", img_path)
        elif self.vision and self.stuck >= 2:
            logger.info("position unchanged for %d turns - attaching frame", self.stuck)
            have_img = (self._fetch_frame("/screenshot/grid?scale=2", img_path)
                        or self._fetch_frame("/screenshot", img_path))

        if intro:
            prompt = INTRO_NUDGE.format(
                server=self.server,
                phase=ctx.get("phase", "unknown"),
                vision=INTRO_VISION_OK if have_img else INTRO_VISION_NONE,
            )
        elif goal:
            prompt = ESCALATION_NUDGE.format(
                server=self.server, goal=goal, budget=budget, n=n,
                brain=self.local_label,
                map_ascii=map_ascii,
                state=json.dumps(_compact_state(state), indent=2),
            )
        else:
            prompt = TURN_NUDGE.format(
                server=self.server,
                map_ascii=map_ascii,
                state=json.dumps(_compact_state(state), indent=2),
            )

        #cmd = ["hermes", "chat", "-Q", "--yolo", "--pass-session-id",
        #       "-s", "pokemon-player", "-t", "file,terminal,web,vision"]
        # Vision toolset only for vision-capable models. A text-only model +
        # vision tool = a retry loop: vision_analyze 500s ("image input is not
        # supported"), the model retries, each retry is an extra API call, and
        # the turn blows its timeout (proven live: 18 API calls x 3s + errors).
        # Small local models (Qwen-2B) also over-search: 10+ web_search calls x
        # 2.5s each + API latency > 120s. A $POKEMON_HERMES_TOOLS override can
        # narrow the toolset further (e.g. "terminal" to drop web).
        toolset = os.environ.get("POKEMON_HERMES_TOOLS", "terminal")
        cmd = ["hermes", "chat", "-Q", "--yolo", "--pass-session-id",
                "-s", "pokemon-player", "-t", toolset]
        if self.session_id and not goal:
            cmd += ["--resume", self.session_id]
        if self.model:
            cmd += ["-m", self.model]
        if self.provider:
            cmd += ["--provider", self.provider]
        # Thinking models (Qwen3.5 etc.) can loop on  forever and fill the
        # whole 65k context ("n_tokens = 65535, truncated = 1" - proven live).
        # --reasoning none disables the loop; $POKEMON_HERMES_REASONING
        # overrides (e.g. "low" if you want bounded thinking).
        reasoning = os.environ.get("POKEMON_HERMES_REASONING", "none")
        if reasoning and reasoning != "default":
            cmd += ["--reasoning", reasoning]
        # Small models loop on tool_search when a wanted tool is missing
        # (proven live: 95 tool_search calls x 2.4s = 228s > timeout). Cap
        # the tool-calling iterations and the wall clock per turn.
        # 6 turns: enough for reasoning + action + a retry or two, without
        # the model burning its whole budget on tool_search loops (12 was
        # proven live to exhaust on 5KB-state prompts).
        cmd += ["--max-turns", "6",
                "--run-budget", str(max(self.turn_timeout - 30, 60))]
        if have_img:
            cmd += ["--image", img_path]
        cmd += ["-q", prompt]

        logger.info("turn %d: phase=%s img=%s stuck=%d prompt=%dB",
                    self.turn + 1, ctx.get("phase"), have_img, self.stuck,
                    len(prompt))
        logger.debug("prompt:\n%s", prompt)

        started = time.perf_counter()
        try:
            # stdin=DEVNULL so an interactive prompt from hermes fails fast
            # instead of blocking until the turn timeout.
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 stdin=subprocess.DEVNULL,
                                 timeout=self.turn_timeout)
            stdout, stderr = out.stdout or "", out.stderr or ""
            logger.info("turn took %.1fs (rc=%s)",
                        time.perf_counter() - started, out.returncode)
            if stdout.strip():
                logger.info("hermes: %s", stdout.strip()[:400])
                # The plan is the handover: it is what Laya steers by for the
                # next few hundred cheap turns, so an escalation that produces
                # one has earned its cost even if it made no direct progress.
                pm = re.search(r"^\s*PLAN:\s*(.+)$", stdout, re.M)
                if pm:
                    self.standing_plan = pm.group(1).strip()[:240]
                    self.plan_set_at = self.turn
                    self.plan_from_map = (state.get("map") or {}).get("map_name", "")
                    logger.warning("NEW PLAN: %s", self.standing_plan)
                    self.event(type="decision",
                               text=f"[plan] {self.standing_plan}")
                if goal and "HANDBACK" in stdout.upper():
                    logger.info("hermes requested handback")
                    self._enter_mode("laya")
            if stderr.strip():
                logger.debug("hermes stderr: %s", stderr[-2000:])
        except subprocess.TimeoutExpired as exc:
            def _tail(v):
                if isinstance(v, bytes):
                    v = v.decode(errors="replace")
                return (v or "")[-1500:]
            logger.error("hermes timed out after %ss", self.turn_timeout)
            logger.error("last stdout: %s", _tail(exc.stdout))
            logger.error("last stderr: %s", _tail(exc.stderr))
            self.event(type="alert", text="Turn timed out - retrying.")
            return
        except Exception as exc:
            logger.error("hermes invocation failed: %s", exc)
            self.event(type="alert", text=f"Driver error: {exc}")
            time.sleep(3)
            return

        # Capture the session id from the first run so later turns resume it.
        # Without this every turn is a brand-new session with no memory.
        if self.session_id is None:
            combined = stdout + "\n" + stderr
            m = (re.search(r"session_id:\s*(\S+)", combined)
                 or re.search(r"hermes --resume (\S+)", combined)
                 or re.search(r"Session:\s*(\S+)", combined))
            if m:
                self.session_id = m.group(1)
                logger.info("hermes session: %s", self.session_id)
                self.bind_hermes()
                self.event(type="key_moment",
                           description="Hermes session started",
                           category="milestone")
            else:
                logger.warning("could not extract session_id - this run will have "
                               "NO memory across turns")
                logger.debug("searched output: %s", combined[:800])

    # --- one turn ----------------------------------------------------------

    def step(self) -> None:
        emu = self.emulator_state()
        if emu != "ready":
            logger.info("emulator %s - waiting (press START on the dashboard)", emu)
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

        # Transition debounce: mid-battle/mid-map-change reads return garbage -
        # species_id 0 ("???(0)") while the game swaps Pokemon data, or dialog
        # text like "999999999..." while the text buffer redraws. Acting on a
        # garbage read wedges the AI on the battle-end screen. Re-read once
        # after a short settle; keep the cleaner of the two.
        def _state_garbage(s: Dict[str, Any]) -> bool:
            am = s.get("active_mon") or {}
            # species_id 0 mid-swap: a real swap rewrites the WHOLE battle
            # struct, so level/max_hp go blank too. A live battle where only
            # the species byte reads 0 while level/hp are real is NOT garbage
            # - calling it one skip-loops forever (proven live: 15+ min wedged
            # at the battle menu, /stats frozen at one turn).
            if am and (s.get("battle") or {}).get("in_battle"):
                if (not am.get("species_id") and not am.get("level")
                        and not am.get("max_hp")):
                    return True
            elif am and am.get("species_id") == 0 and (s.get("party") or []):
                # Party non-empty but active mon reads species 0 outside
                # battle: transition leftover. Only trust when a party exists
                # (pre-starter intros legitimately read 0).
                return True
            txt = ((s.get("dialog") or {}).get("text")) or ""
            nonspace = txt.replace(" ", "")
            if len(nonspace) > 10:
                common = max(set(nonspace), key=nonspace.count)
                if nonspace.count(common) / len(nonspace) > 0.8:
                    return True      # buffer redraw garbage (all-9s etc.)
            return False

        if _state_garbage(state):
            # A read that is dirty twice (1.5s apart) is normally a transition,
            # but a STABLE dirty read would skip forever and wedge the run.
            # Cap the skips: after 5 consecutive dirty reads, treat the state
            # as authoritative and act on it.
            self._dirty_skips = getattr(self, "_dirty_skips", 0) + 1
            if self._dirty_skips > 5:
                logger.warning("state dirty for %d consecutive reads - "
                               "acting anyway", self._dirty_skips)
            else:
                time.sleep(1.5)
                try:
                    retry = self._get("/state").json()
                    if not _state_garbage(retry):
                        logger.info("garbage state read (transition) - re-read clean")
                        state = retry
                    else:
                        logger.info("garbage state read (transition) - still dirty, "
                                    "skipping this turn (%d/5)", self._dirty_skips)
                        time.sleep(1.5)
                        return
                except Exception:
                    time.sleep(1.5)
                    return
        else:
            self._dirty_skips = 0

        ctx = state.get("context") or {}
        intro = not ctx.get("in_game", False)

        if self.control_state() != "running":
            logger.info("control changed mid-turn - skipping Hermes call")
            return
        
        # Stuck detection: the map says we can move but the position is not
        # changing. Escalates to vision (Hermes) which usually reveals an
        # unnoticed text box or a sprite the grid missed.
        pos = (state.get("player") or {}).get("position")
        self.stuck = self.stuck + 1 if (pos is not None and pos == self.last_pos) else 0
        self.last_pos = pos

        if self.stuck >= 5:
            logger.warning("stuck %d turns - forcing B to clear any text box",
                           self.stuck)
            self.act(["press_b", "wait_30"])
            self.stuck = 0
            return

        fp = _progress_fingerprint(state)
        made_progress = fp != self.progress_fp
        if made_progress:
            if self.progress_fp is not None:
                logger.info("progress: %s -> %s", self.progress_fp, fp)
            self.progress_fp = fp
            self.progress_at = time.perf_counter()
            self._sync_objectives(state)

        mode = self._arbitrate(state, made_progress) if self.use_laya else "hermes"
        if mode == "hermes":
            self._hermes_turn(state, intro, goal=self.hermes_goal,
                              budget=self.hermes_budget, n=self.mode_turns)
        else:
            self._laya_turn(state, intro)

        self.turn += 1
        if self.save_every and self.turn % self.save_every == 0:
            self.save_game()
            self._load_explore_union()

    # --- main loop ---------------------------------------------------------

# --- main loop ---------------------------------------------------------

    def run(self) -> None:
        # Hermes must always be reachable: escalation can fire at any time when
        # Laya stalls, so a broken hermes is fatal even in --laya mode.
        if not self.preflight():
            self.event(type="alert",
                       text="Hermes preflight failed - see driver log.")
            sys.exit(1)

        brain = (f"{self.local_label} + hermes escalation"
                 if self.use_laya else "hermes")
        print(f"[driver] autopilot. server={self.server} brain={brain} "
              f"model={self.model or 'config default'}")
        if self.use_laya:
            print(f"[driver] escalating to hermes after {self.stall_seconds:.0f}s "
                  f"without progress, for up to {self.hermes_turns} turns")
        print("[driver] waiting for control=running + an active game…")
        self.event(type="alert",
                   text=f"Driver online ({brain}) - load a game, then press START.")

        idle_logged = False
        no_game_logged = False
        while True:
            st = self.control_state()
            if st == "stopped":
                if not idle_logged:
                    print("[driver] stopped - idling.")
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
                    print("[driver] running but no active game - start/load one "
                          "on the dashboard.")
                    self.event(type="alert",
                               text="No active game - click New Game or load one.")
                    no_game_logged = True
                time.sleep(2)
                continue
            no_game_logged = False

            # Re-check immediately before acting: a hermes turn can take tens of
            # seconds and span a STOP press, and state is only read at the top.
            if self.control_state() != "running":
                continue

            # A bad turn must not kill the run. SystemExit is deliberate
            # (repeated laya failures) and must propagate.
            try:
                self.step()
            except SystemExit:
                raise
            except Exception:
                logger.exception("turn failed - continuing")
                self.event(type="alert", text="Driver error - see log.")
                time.sleep(3)

            # Laya turns are ~70ms; hermes turns are tens of seconds. Only the
            # fast path needs throttling.
            if self.mode == "laya":
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
                  scorer: Optional[str] = None) -> None:
    if debug:
        logging.getLogger("pokemon-agent").setLevel(logging.DEBUG)
        logger.info("debug logging enabled")
    model = model or os.environ.get("POKEMON_HERMES_MODEL")
    provider = os.environ.get("POKEMON_HERMES_PROVIDER")
    HermesDriver(server, model, provider,
                 turn_delay=turn_delay, turn_timeout=turn_timeout,
                 save_every=save_every, use_laya=use_laya,
                 laya_narrate_every=laya_narrate_every,
                 stall_seconds=stall_seconds,
                 hermes_turns=hermes_turns,
                 vision=vision,
                 replan_every=replan_every,
                 scorer=scorer).run()