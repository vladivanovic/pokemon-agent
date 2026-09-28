"""Standalone driver that plays Pokemon through a game session.

Two brains are supported:

  Hermes (default) — a real Hermes Agent session with the `pokemon-player`
    skill, vision, memory and the terminal tool, driven one turn at a time via
    `hermes chat --resume`. Slow (seconds to minutes per turn on local
    hardware) but can reason, narrate, search the web and set objectives.

  Laya (--laya)    — an in-process decision model. Milliseconds per turn,
    returns a structured choice from a fixed candidate set, no text
    generation, no regex parsing, no hallucinated actions. Bypasses Hermes
    entirely. Cannot narrate or plan.

Normal Hermes turns are TEXT ONLY. The ASCII collision map in /state is
ground truth read from game RAM, so it beats asking a vision model to read
pixel art — and it keeps the prompt small, which matters a lot on local
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

# Laya decision model — optional. Import failure must be loud at use time,
# not silently equivalent to "feature disabled".
try:
    from laya import Router  # type: ignore
    LAYA_AVAILABLE = True
    _LAYA_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover
    Router = None  # type: ignore
    LAYA_AVAILABLE = False
    _LAYA_IMPORT_ERROR = _exc

logging.basicConfig(level=logging.INFO,
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

INTRO_VISION_OK = "A screenshot of the current screen is attached — look at it."
INTRO_VISION_NONE = ("No screenshot available. Press A to advance and check "
                     "the result next turn.")

INTRO_NUDGE = """You are booting Pokémon Red on the Hermes Plays Pokémon dashboard.

Server: {server}

The game is NOT in play yet — it is at the title screen, Oak's intro, or a
name-entry menu. Game state values are uninitialised garbage right now, so
IGNORE them entirely. {vision}

Your only job this turn: advance the intro. POST one of these to
{server}/action with -H 'Content-Type: application/json':

  Title screen / NEW GAME       {{"actions":["press_a"]}}
  Oak talking / any text box    {{"actions":["a_until_dialog_end"]}}
  Name menu (NEW NAME/RED/...)  {{"actions":["press_down","press_a"]}}
  Options screen (went too far) {{"actions":["press_b"]}}
  Unsure                        {{"actions":["press_a"]}}

On the name menu do NOT press A on "NEW NAME" — that opens letter-by-letter
entry. Press down first to take a preset.

Do not narrate, do not set objectives yet. Current phase: {phase}

Reply with one short sentence saying what you pressed.
"""

TURN_NUDGE = """You are playing Pokémon Red on the Hermes Plays Pokémon dashboard.

Server: {server}

Take ONE short turn:
1. POST {server}/event  {{"type":"reasoning","text":"..."}}     what you see
2. POST {server}/action {{"actions":["walk_down","walk_down"]}}  2-4 moves
3. Reply with ONE short sentence. Be brief.

On a real beat (new town, badge, catch) also POST {server}/event
{{"type":"key_moment","description":"...","category":"milestone|badge|catch"}}

All POSTs need -H 'Content-Type: application/json'.

The MAP below is ground truth read from game memory — trust it over any image.
You are always at @ (cell E5). Columns A-J left to right, rows 1-9 top to
bottom. `.` walkable, `#` blocked, `N` a person blocking you, `D` a door/exit.

If the map says "unavailable", or you are in a menu/battle/dialog and cannot
tell what is on screen, you may look at the frame:
  curl -s '{server}/screenshot' -o /tmp/look.png
then use the vision tool on /tmp/look.png. Only do this when the map and state
are not enough — it costs an extra round trip.

MAP:
{map_ascii}

STATE:
{state}
"""

ESCALATION_NUDGE = """You are playing Pokémon Red. Laya (a fast movement model) has
been driving, but nothing has advanced for a while, so YOU have control for the
next {budget} turns. This is turn {n} of {budget}.

GOAL: {goal}

Server: {server}

You know Pokémon Red. Laya does not — it only picks directions. Use that
knowledge: work out what the game is waiting for, and do it.

NEVER call /load or /save. NEVER load a save state. If movement seems not to
work, it is because a text box is open or an NPC is in the way — not because
the emulator is broken. Press B to clear text, or walk around the obstacle.

Each turn:
1. POST {server}/event {{"type":"reasoning","text":"..."}}  what is blocking us
2. POST {server}/action {{"actions":[...]}}  up to 8 actions — you may send a
   longer sequence than usual since you have the context to plan it
3. If the blocker is cleared and only movement remains, write HANDBACK in your
   reply and Laya will resume.

All POSTs need -H 'Content-Type: application/json'.

MAP:
{map_ascii}

STATE:
{state}
"""


# ---------------------------------------------------------------------------
# State trimming
# ---------------------------------------------------------------------------

def _compact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Trim the full state dict to what a turn actually needs.

    The ASCII map is passed separately in the prompt body, and raw tile_ids /
    timestamps are dropped — they burn tokens without informing a decision.
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
            "moves": [mv.get("name") if isinstance(mv, dict) else mv
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
        # battle.enemy is a DICT, not a list — one active enemy at a time.
        "enemy": ({"species": enemy.get("species"), "level": enemy.get("level"),
                   "hp": enemy.get("hp"), "max_hp": enemy.get("max_hp"),
                   "types": enemy.get("types")}
                  if battle.get("in_battle") else None),
        "status": state.get("status"),
        "dialog_text": (state.get("dialog")),
    }
    # Only surface failures when there are some — silence is the normal case.
    if state.get("errors"):
        out["errors"] = state["errors"]
    exits = [{"cell": w.get("cell"), "to": w.get("dest_map_name")}
             for w in (state.get("collision") or {}).get("warps") or []]
    if exits:
        out["exits"] = exits
    return out

def _progress_fingerprint(state: Dict[str, Any]) -> tuple:
    """State that only changes on genuine advancement.

    Deliberately excludes position: wandering changes x/y every turn but is
    not progress. Map transitions, party growth, badges and story flags are.
    """
    flags = state.get("flags") or {}
    party = state.get("party") or []
    return (
        (state.get("map") or {}).get("map_id"),
        len(party),
        sum(m.get("level", 0) for m in party),
        flags.get("badge_count", 0),
        bool(flags.get("has_pokedex")),
        bool(flags.get("has_oaks_parcel")),
        flags.get("pokedex_owned", 0),
        len(state.get("bag") or []),
    )

# ---------------------------------------------------------------------------
# Grid helpers — screen-relative pathing over verified walkability
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
             limit: int = 12) -> List[str]:
    """BFS from E5 to *target*, returning walk actions.

    Trivially cheap on a 10x9 grid, and it converts "one LLM call per tile"
    into "one call per destination" — the single biggest speed win available
    when the decision model is the bottleneck.
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
                      cur_map: str, px: int, py: int) -> List[Tuple[int, int]]:
    """On-screen cells the agent has never stood on, reachable from E5.

    Screen cells map to world coords by offsetting from the player: the player
    is always at (PLAYER_ROW, PLAYER_COL) and at world (px, py).
    """
    out = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if not _grid_open(passable, r, c):
                continue
            wx = px + (c - PLAYER_COL)
            wy = py + (r - PLAYER_ROW)
            if visits.get((cur_map, wx, wy), 0) == 0:
                out.append((r, c))
    return out

def cell_label_local(col: int, row: int) -> str:
    """0-indexed (col,row) -> 'E5'. Mirrors collision.cell_label."""
    return f"{COL_LABELS_LOCAL[col]}{row + 1}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class HermesDriver:
    def __init__(self, server: str, model: Optional[str], provider: Optional[str],
                 turn_delay: float = 1.5, save_every: int = 20,
                 turn_timeout: int = 240, use_laya: bool = False,
                 laya_narrate_every: int = 0,
                 stall_seconds: float = 60.0, hermes_turns: int = 12):
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
        self.map_changed_at: int = -99
        self.map_changed_time: float = 0.0
        # --- brain arbitration ---
        self.stall_seconds = stall_seconds
        self.hermes_turns = hermes_turns
        self.mode: str = "laya" if use_laya else "hermes"
        self.mode_turns: int = 0
        self.hermes_budget: int = 0
        self.hermes_goal: str = ""
        self.progress_fp: Optional[tuple] = None
        self.progress_at: float = time.perf_counter()
        

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

    # --- server helpers ----------------------------------------------------

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
        nothing — this is the endpoint that actually presses buttons."""
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
            logger.error("preflight timed out — model or gateway not responding")
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
        is worse than none — it misleads both the viewer and any LLM that
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
        self.mode, self.mode_turns = mode, 0
        self.recent.clear()
        self.progress_at = time.perf_counter()   # grace period in the new mode
        self.event(type="decision", text=f"[brain] switched to {mode}")

    def _infer_goal(self, state: Dict[str, Any]) -> str:
        """A coarse objective for the escalated turns, from game state alone."""
        flags = state.get("flags") or {}
        party = state.get("party") or []
        map_name = (state.get("map") or {}).get("map_name", "?")
        if not party:
            return "Get your first Pokemon from Oak's Lab"
        if not flags.get("has_pokedex"):
            return ("Deliver Oak's Parcel from the Viridian City mart, "
                    "then get the Pokedex")
        if flags.get("badge_count", 0) == 0:
            return "Reach Pewter City Gym and beat Brock for the Boulder Badge"
        return f"Make story progress; you appear stuck in {map_name}"

    def _arbitrate(self, state: Dict[str, Any], made_progress: bool) -> str:
        """Decide which brain drives this turn.

        Laya is the default: ~500x cheaper and good at movement. Hermes takes
        over when nothing has advanced for a while and KEEPS control for a
        budget of turns, so it can finish a multi-step errand instead of being
        cut off mid-sequence.
        """
        stalled_for = time.perf_counter() - self.progress_at

        if self.mode == "hermes":
            self.mode_turns += 1
            if made_progress and self.mode_turns >= 2:
                logger.info("hermes made progress after %d turns — back to laya",
                            self.mode_turns)
                self._enter_mode("laya")
            elif self.mode_turns >= self.hermes_budget:
                logger.warning("hermes budget (%d turns) exhausted without "
                               "progress — back to laya", self.hermes_budget)
                self._enter_mode("laya")
            return self.mode

        self.mode_turns += 1
        if stalled_for > self.stall_seconds:
            self.hermes_goal = self._infer_goal(state)
            self.hermes_budget = self.hermes_turns
            logger.warning("no progress for %.0fs — escalating to hermes "
                           "(budget %d turns, goal: %s)",
                           stalled_for, self.hermes_budget, self.hermes_goal)
            self._enter_mode("hermes")
            self.event(type="alert",
                       text=f"Escalating to Hermes: {self.hermes_goal}")
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
            # Full traceback once, then just the message — this fires every
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
        if choice not in criteria:
            logger.warning("laya returned %r which is not in %s",
                           choice, sorted(criteria))
            return None

        conf = ans.get("answer_confidence")
        if isinstance(conf, (int, float)):
            self.last_confidence = float(conf)
        return choice

    def _laya_turn(self, state: Dict[str, Any], intro: bool) -> None:
        """One Laya-driven turn: pick from a legal candidate set and execute."""
        col = state.get("collision") or {}
        passable = col.get("passable") or col.get("walkable") or []
        dlg = state.get("dialog") or {}
        battle = state.get("battle") or {}
        p = state.get("player") or {}
        warps = col.get("warps") or []

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

        # Wall-clock, not turn count: Laya runs ~10 turns/sec, so a 6-turn
        # window expires before the agent has taken a single step away from
        # the door it just came through.
        recent_transition = (time.perf_counter() - self.map_changed_time) < 8.0
        on_warp = bool(col.get("player_on_warp")) or any(
            (w.get("row"), w.get("col")) == (PLAYER_ROW, PLAYER_COL)
            for w in warps)

        # The candidate set is built from ground truth, so an illegal move is
        # not merely discouraged — it is never offered.
        criteria: Dict[str, str] = {}
        if intro or dlg.get("text_active"):
            instructions = ("A text box or menu may be on screen. Advance it, or "
                            "move if you think it has already closed.")
            criteria["advance_text"] = "Press A to advance dialog, menu or intro"
            criteria["back_out"] = "Press B to dismiss or cancel"
            # Movement too: the dialog flag can be stale, and walking is the
            # only way to find out. Never offer a single-option question.
            for name in _legal_dirs(passable):
                criteria[name] = f"Move one tile {name.split('_')[1]}"
            if intro:
                criteria["menu_down_a"] = "Move the menu cursor down, then confirm"
        elif battle.get("in_battle"):
            instructions = ("You are in a Pokemon battle. Choose the single best "
                            "menu action for this turn.")
            criteria["advance_text"] = "Press A to confirm the highlighted option"
            criteria["menu_down_a"] = "Move down one option, then confirm"
            criteria["back_out"] = "Press B to go back"
        else:
            times_here = self.visits.get(key, 0)
            instructions = (
                "You are exploring Pokemon Red. The MAP shows what is around you: "
                "'.' is walkable, '#' is a wall, 'N' is a person blocking you, "
                "'D' is a door or exit, '@' is you. You have stood on this exact "
                f"tile {times_here} time(s) — prefer moves that lead somewhere "
                "new, and leave the building once the room is explored.")

            # Suppress the warp we just came through. Blocking only the named
            # goto_ option is not enough: walking onto the tile triggers it too.
            if recent_transition and warps:
                passable = [list(r) for r in passable]   # copy before mutating
                for w in warps:
                    wr, wc = w.get("row"), w.get("col")
                    if (wr is not None and wc is not None
                            and 0 <= wr < len(passable)
                            and 0 <= wc < len(passable[wr])):
                        passable[wr][wc] = False

            # Frontier exploration: offer multi-tile destinations the agent has
            # never stood on, rather than four interchangeable single steps.
            # Laya scores ~0.33 on symmetric directions and ~0.99 on named
            # destinations, so distinct options are worth far more than nudges.
            frontier = _frontier_targets(passable, self.visits, cur_map,
                                         pos.get("x") or 0, pos.get("y") or 0)
            # Farthest first — a neighbouring unvisited tile is barely a choice,
            # "walk to the far corner" is a plan.
            frontier.sort(key=lambda rc: -(abs(rc[0] - PLAYER_ROW)
                                           + abs(rc[1] - PLAYER_COL)))
            added = 0
            for fr, fc in frontier:
                if added >= 4:
                    break
                path = _path_to(passable, (fr, fc))
                if not path:
                    continue
                label = cell_label_local(fc, fr)
                criteria[f"explore_{label}"] = (
                    f"Walk {len(path)} tiles to {label}, never visited")
                added += 1

            # Single steps remain as a fallback: when every reachable cell has
            # been seen, the frontier is empty and the agent still needs to move.
            if added == 0:
                for name in _legal_dirs(passable):
                    criteria[name] = f"Move one tile {name.split('_')[1]}"

            criteria["interact"] = "Press A to talk to or examine what you face"
            # Standing ON a doormat: "goto" it is a no-op, so offer the exit as
            # its own action instead.
            if on_warp and not recent_transition:
                criteria["exit_building"] = ("Walk out through the door you are "
                                             "standing on")

            if not recent_transition:
                for w in warps[:4]:
                    cell = w.get("cell")
                    if not cell:
                        continue
                    if (w.get("row"), w.get("col")) == (PLAYER_ROW, PLAYER_COL):
                        continue          # covered by exit_building
                    dest = w.get("dest_map_name") or f"map {w.get('dest_map')}"
                    criteria[f"goto_{cell}"] = (
                        f"Walk to the exit at {cell} leading to {dest}")

        if not criteria:
            instructions = "Nothing is possible right now. Press B."
            criteria["back_out"] = "Press B"

        laya_state = {
            "map_name": cur_map,
            "x": pos.get("x"),
            "y": pos.get("y"),
            "facing": p.get("facing"),
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
        }

        # Oscillation guard. Do NOT clear the window on dialog turns — a text
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
                legal = list(_legal_dirs(passable))
                escape = legal[self.turn % len(legal)] if legal else "press_b"
                logger.warning("loop detected (%d map(s), %d cell(s)) — "
                               "escaping via %s", len(maps), len(cells), escape)
                self.recent.clear()
                self.act([escape] * 4)
                return

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
                    "Laya failed 5 turns in a row — see the traceback above. "
                    "The question schema is probably wrong.")
            time.sleep(1.0)
            return
        self.consecutive_fail = 0

        actions: List[str] = []
        if choice == "advance_text":
            actions = (["a_until_dialog_end"] if dlg.get("text_active")
                       else ["press_a"])
        elif choice == "menu_down_a":
            actions = ["press_down", "press_a"]
        elif choice == "back_out":
            actions = ["press_b"]
        elif choice == "interact":
            actions = ["press_a"]
        elif choice == "exit_building":
            # House exits in Gen 1 are on the south edge, so walking down off
            # the mat triggers the warp. Verified by hand on Red's House 1F.
            actions = ["walk_down", "walk_down"]
        elif choice in _DIRS:
            actions = [choice]
        elif choice in _DIRS:
            actions = [choice]
        # ---- add ----
        elif choice.startswith("explore_"):
            label = choice[8:]
            try:
                fc = COL_LABELS_LOCAL.index(label[0])
                fr = int(label[1:]) - 1
                actions = _path_to(passable, (fr, fc))
            except (ValueError, IndexError):
                actions = []
            if not actions:
                legal = list(_legal_dirs(passable))
                actions = [legal[0]] if legal else ["wait_30"]
        # ---- end ----
        elif choice.startswith("goto_"):
            # One decision, many tiles: BFS over verified walkability turns a
            # per-tile decision loop into a per-destination one.
            label = choice[5:]
            target = next(((w["row"], w["col"]) for w in warps
                           if w.get("cell") == label), None)
            if target is not None:
                actions = _path_to(passable, target)
            if not actions:
                logger.warning("no path to %s; taking a single step instead",
                               label)
                legal = list(_legal_dirs(passable))
                actions = [legal[0]] if legal else ["wait_30"]
        else:
            logger.warning("unhandled laya choice %r — waiting", choice)
            actions = ["wait_30"]

        logger.info("turn %d: laya=%s (p=%.2f) in %.0fms -> %s (from %d options) "
                    "map=%s pos=(%s,%s) here=%d stuck=%d",
                    self.turn + 1, choice, self.last_confidence, took_ms,
                    actions, len(criteria), cur_map, pos.get("x"), pos.get("y"),
                    laya_state["times_on_this_tile"], self.stuck)

        self.act(actions)
        self.event(type="decision",
                   text=f"[laya {took_ms:.0f}ms] {choice} → {' · '.join(actions)}")

        
    # --- Hermes ------------------------------------------------------------

    def _hermes_turn(self, state: Dict[str, Any], intro: bool,
                     goal: str = "", budget: int = 0, n: int = 0) -> None:
        """One Hermes-driven turn: build a prompt, shell out, let it act."""
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
            logger.info("position unchanged for %d turns — attaching frame", self.stuck)
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
                "-s", "pokemon-player", "-t", "terminal,webex,vision"]
        if self.session_id:
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
            self.event(type="alert", text="Turn timed out — retrying.")
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
                logger.warning("could not extract session_id — this run will have "
                               "NO memory across turns")
                logger.debug("searched output: %s", combined[:800])

    # --- one turn ----------------------------------------------------------

    def step(self) -> None:
        emu = self.emulator_state()
        if emu != "ready":
            logger.info("emulator %s — waiting (press START on the dashboard)", emu)
            time.sleep(3)
            return

        try:
            state = self._get("/state").json()
        except Exception as exc:
            logger.error("state read failed: %s", exc)
            time.sleep(2)
            return

        if state.get("status") == "not_ready":
            logger.info("state not ready — waiting")
            time.sleep(2)
            return

        ctx = state.get("context") or {}
        intro = not ctx.get("in_game", False)

        if self.control_state() != "running":
            logger.info("control changed mid-turn — skipping Hermes call")
            return
        
        # Stuck detection: the map says we can move but the position is not
        # changing. Escalates to vision (Hermes) which usually reveals an
        # unnoticed text box or a sprite the grid missed.
        pos = (state.get("player") or {}).get("position")
        self.stuck = self.stuck + 1 if (pos is not None and pos == self.last_pos) else 0
        self.last_pos = pos

        if self.stuck >= 5:
            logger.warning("stuck %d turns — forcing B to clear any text box",
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

    # --- main loop ---------------------------------------------------------

# --- main loop ---------------------------------------------------------

    def run(self) -> None:
        # Hermes must always be reachable: escalation can fire at any time when
        # Laya stalls, so a broken hermes is fatal even in --laya mode.
        if not self.preflight():
            self.event(type="alert",
                       text="Hermes preflight failed — see driver log.")
            sys.exit(1)

        brain = "laya + hermes escalation" if self.use_laya else "hermes"
        print(f"[driver] autopilot. server={self.server} brain={brain} "
              f"model={self.model or 'config default'}")
        if self.use_laya:
            print(f"[driver] escalating to hermes after {self.stall_seconds:.0f}s "
                  f"without progress, for up to {self.hermes_turns} turns")
        print("[driver] waiting for control=running + an active game…")
        self.event(type="alert",
                   text=f"Driver online ({brain}) — load a game, then press START.")

        idle_logged = False
        no_game_logged = False
        while True:
            st = self.control_state()
            if st == "stopped":
                if not idle_logged:
                    print("[driver] stopped — idling.")
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
                    print("[driver] running but no active game — start/load one "
                          "on the dashboard.")
                    self.event(type="alert",
                               text="No active game — click New Game or load one.")
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
                logger.exception("turn failed — continuing")
                self.event(type="alert", text="Driver error — see log.")
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
                  hermes_turns: int = 12) -> None:
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
                 hermes_turns=hermes_turns).run()