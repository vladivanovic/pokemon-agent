"""Standalone driver that plays Pokemon through a game session.

Two brains are supported:

  Hermes (default) - a real Hermes Agent session with the `pokemon-player`
    skill, vision, memory and the terminal tool, driven one turn at a time via
    `hermes chat --resume`. Slow (seconds to minutes per turn on local
    hardware) but can reason, narrate, search the web and set objectives.

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

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

# The collision grid is screen-relative and the player is always at E5,
# i.e. (row 4, col 4) in the 9x10 walkability grids.
PLAYER_ROW, PLAYER_COL = 4, 4
GRID_ROWS, GRID_COLS = 9, 10
_DIRS: Dict[str, Tuple[int, int]] = {
    "walk_up": (-1, 0), "walk_down": (1, 0),
    "walk_left": (0, -1), "walk_right": (0, 1),
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

If you need information about the world (e.g., what a building does), you can use the web search tool."""

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

If you need information about the world (e.g., what a building does), you can use the web search tool."""

ESCALATION_NUDGE = """You are playing Pokémon Red. Laya (a fast movement model) has
been driving, but nothing has advanced for a while, so YOU have control for the
next {budget} turns. This is turn {n} of {budget}.

GOAL: {goal}

Server: {server}

You know Pokémon Red. Laya does not - it only picks directions. Use that
knowledge: work out what the game is waiting for, and do it.

NEVER call /load or /save. NEVER load a save state. If movement seems not to
work, it is because a text box is open or an NPC is in the way - not because
the emulator is broken. Press B to clear text, or walk around the obstacle.

Each turn:
1. POST {server}/event {{"type":"reasoning","text":"..."}}  what is blocking us
2. POST {server}/action {{"actions":[...]}}  up to 8 actions - you may send a
   longer sequence than usual since you have the context to plan it
3. If the blocker is cleared and only movement remains, write HANDBACK in your
   reply and Laya will resume.

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

If you need information about the world (e.g., what a building does), you can use the web search tool."""


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
        "party": party,
        "active_mon": state.get("active_mon"),
        "text_active": dialog.get("text_active"),
        "input_locked": dialog.get("input_locked"),
        "in_battle": battle.get("in_battle"),
        # battle.enemy is a DICT, not a list - one active enemy at a time.
        "enemy": ({"species": enemy.get("species"), "level": enemy.get("level"),
                   "hp": enemy.get("hp"), "max_hp": enemy.get("max_hp"),
                   "types": enemy.get("types")}
                  if battle.get("in_battle") else None),
        "status": state.get("status"),
        "dialog_text": (state.get("dialog")) or None,
    }
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
                 vision: bool = True, replan_every: int = 120):
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
        self.recent: deque = deque(maxlen=12)    # oscillation detection
        self.prev_map: Optional[str] = None
        self.visits: Dict[tuple, int] = {}      # (map, x, y) -> times seen
        self.special_tiles: Dict[tuple, str] = {}   # (map, x, y) -> tile purpose
        self.explore_union_path = os.environ.get(
            "POKEMON_EXPLORE_UNION",
            "/home/vlad/pokemon-agent/explore_map_union.json")
        self.union_global: set = set()          # (gy, gx) global coords from SB3
        self.map_offsets: Dict[str, tuple] = {}  # map_name -> (mx, my) for global conv
        self._load_map_offsets()
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
        self.session_started_at: int = 0
        self.hermes_time_total: float = 0.0      # seconds spent in Hermes mode
        self.laya_time_total: float = 0.0       # seconds spent in Laya mode
        self.last_mode_switch: float = time.perf_counter()
        self._hermes_last_pos: Optional[Any] = None
        self.hermes_budget_max: int = 30
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
        self.event(type="decision", text=f"[brain] switched to {mode}")
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

            if made_progress and self.mode_turns >= 2 and not (state.get("battle") or {}).get("in_battle"):
                logger.info("hermes made progress after %d turns - back to laya", self.mode_turns)
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
                logger.warning("hermes budget (%d turns) exhausted - back to laya",
                               self.hermes_budget)
                self._enter_mode("laya")
            return self.mode

        if (state.get("battle") or {}).get("in_battle"):
            self.progress_at = time.perf_counter()
            # Ensure Hermes is driver for battle turns so stats reflect correctly
            self._enter_mode("hermes")
            return self.mode

        self.mode_turns += 1
        # Two reasons to escalate. A stall means Laya is wedged. A stale plan
        # means it is executing intent formed long ago - re-planning on a timer
        # keeps steering fresh instead of waiting for a wedge.
        plan_age = self.turn - self.plan_set_at
        plan_stale = self.standing_plan and plan_age > self.replan_every
        if stalled_for > self.stall_seconds or plan_stale:
            why = "stalled" if stalled_for > self.stall_seconds else "plan stale"
            self.hermes_goal = self._infer_goal(state)
            self.hermes_budget = self.hermes_turns
            logger.warning("escalating to hermes (%s, %.0fs since progress, "
                           "plan %d turns old, budget %d): %s",
                           why, stalled_for, plan_age, self.hermes_budget,
                           self.hermes_goal)
            self._enter_mode("hermes")
            self.event(type="alert", text=f"Escalating ({why}): {self.hermes_goal}")
        return self.mode

    # --- Laya --------------------------------------------------------------
    #
    # ALL Laya-specific API usage is confined to _laya_choose(). If the real
    # signature differs from what is assumed here, this is the only method
    # that needs changing.

    def _laya_choose(self, laya_state: Dict[str, Any],
                     criteria: Dict[str, str],
                     instructions: str) -> Optional[str]:
        """Ask Laya to pick one key from *criteria*. Returns the key or None.

        ASSUMED API:
            router.predict(state: dict, questions: dict) -> dict
            questions = {"action": {"type": "choice",
                                    "instructions": str,
                                    "criteria": {key: description}}}
            result["action"]["choice"] == one of the criteria keys
        """
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
        warps = col.get("warps") or []
        # Quest flags so Laya knows what's done and skips unnecessary buildings
        flags = state.get("flags") or {}
        has_parcel = bool(flags.get("has_oaks_parcel"))
        has_dex = bool(flags.get("has_pokedex"))

        cur_map = (state.get("map") or {}).get("map_name", "")
        pos = p.get("position") or {}
        if cur_map != self.prev_map:
            logger.info("map changed: %s -> %s", self.prev_map, cur_map)
            self.prev_map = cur_map
            self.map_changed_at = self.turn
            self.map_changed_time = time.perf_counter()
            self.recent.clear()          # a real transition is not a loop
        key = (cur_map, pos.get("x"), pos.get("y"))
        self.visits[key] = self.visits.get(key, 0) + 1

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
            elif pps and pps[0] > 0:
                criteria["attack"] = "Select FIGHT and use the first move"
            else:
                # First move has no PP but others do - offer specific moves
                # with PP > 0 so the AI can choose one (e.g., Growl to burn
                # PP toward Struggle). Each move becomes its own criterion.
                for i, mv in enumerate(moves):
                    pp = mv.get("pp", 0) if isinstance(mv, dict) else 0
                    if pp > 0:
                        move_name = mv.get("name", f"move_{i}")
                        criteria[f"move_{i}"] = (
                            f"Select FIGHT, navigate to {move_name} (PP={pp}), "
                            "and confirm"
                        )
            if hp_frac < 0.35:
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
                for name, (dr, dc) in _DIRS.items():
                    d = name.split("_")[1]
                    if not _grid_open(passable, PLAYER_ROW + dr,
                                      PLAYER_COL + dc):
                        continue
                    criteria[f"travel_{d}"] = (
                        f"Head {d} for several tiles, continuing in that "
                        f"direction to reach a new area")

                # --- C3: single steps ---
                # Always offered, not only when the frontier is empty: if every
                # frontier path is blocked mid-route the agent still needs a way
                # to nudge past the obstruction.
                # Exclude upward ledge crossing: ledges are passable DOWNWARD only.
                ledge_grid = col.get("ledges") or [[False]*10 for _ in range(9)]
                for name in _legal_dirs(passable):
                    dr, dc = _DIRS[name]
                    # Going up (north) through a ledge is blocked; going down (south)
                    # through a ledge is allowed (engine handles the wrap-through).
                    nr, nc = PLAYER_ROW + dr, PLAYER_COL + dc
                    if dr < 0 and 0 <= nr < 9 and 0 <= nc < 10 and ledge_grid[nr][nc]:
                        continue  # skip moving up through a ledge
                    criteria[name] = f"Move one tile {name.split('_')[1]}"

                criteria["interact"] = "Press A to talk to or examine what you face"

                # Named exits elsewhere on screen (not the one underfoot).
                if not recent_transition:
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

        # Dialog cleared - reset the stall counter (after the whole if/elif
        # chain so explore still runs when no dialog is active)
        if not intro and not dlg.get("text_active") and not battle.get("in_battle"):
            self._dialog_turns = 0

        started = time.perf_counter()
        choice = self._laya_choose(laya_state, criteria, instructions)
        took_ms = (time.perf_counter() - started) * 1000.0

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

        # ------------------------------------------------------------------
        # DISPATCH - map the chosen key to emulator actions
        # ------------------------------------------------------------------
        actions: List[str] = []
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
            # FIGHT, then navigate down (i-1) entries to the move, confirm.
            # Cursor starts at the first move after FIGHT is chosen.
            idx = int(choice[5:])
            actions = ["press_b", "wait_30", "press_up", "press_left",
                       "press_a", "wait_30"]
            actions += ["press_down"] * idx
            actions += ["press_a", "wait_60"]
        elif choice == "flee":
            actions = ["press_b", "wait_30", "press_down", "press_right",
                       "press_a", "wait_60"]
        elif choice == "exit_building":
            # South covers nearly every Gen 1 building exit. Rotate on repeat
            # attempts so a north/side exit is eventually found.
            order = ["walk_down", "walk_down", "walk_up",
                     "walk_left", "walk_right"]
            actions = [order[self.turn % len(order)]] * 2
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
        else:
            logger.warning("unhandled laya choice %r - waiting", choice)
            actions = ["wait_30"]

        logger.info("turn %d: laya=%s (p=%.2f) in %.0fms -> %s (from %d options) "
                    "map=%s pos=(%s,%s) door=%s here=%d stuck=%d",
                    self.turn + 1, choice, self.last_confidence, took_ms,
                    actions, len(criteria), cur_map, pos.get("x"), pos.get("y"),
                    on_warp, laya_state["times_on_this_tile"], self.stuck)

        self.act(actions)
        self.event(type="decision",
                   text=f"[laya {took_ms:.0f}ms] {choice} → {' · '.join(actions)}")

        
    # --- Hermes ------------------------------------------------------------

    def _hermes_turn(self, state: Dict[str, Any], intro: bool,
                     goal: str = "", budget: int = 0, n: int = 0) -> None:
        """One Hermes-driven turn: build a prompt, shell out, let it act."""
        # Replaying a 1000-message history dominates latency and buys nothing:
        # the escalation goal is regenerated from live state every turn.
        if self.session_id and (self.turn - self.session_started_at) > 10:
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
        cmd = ["hermes", "chat", "-Q", "--yolo", "--pass-session-id",
                "-s", "pokemon-player", "-t", "terminal,web,vision"]
        if self.session_id and not goal:
            cmd += ["--resume", self.session_id]
        if self.model:
            cmd += ["-m", self.model]
        if self.provider:
            cmd += ["--provider", self.provider]
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

        brain = "laya + hermes escalation" if self.use_laya else "hermes"
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
                  replan_every: int = 120) -> None:
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
                 replan_every=replan_every).run()