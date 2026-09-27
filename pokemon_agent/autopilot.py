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
    }
    # Only surface failures when there are some — silence is the normal case.
    if state.get("errors"):
        out["errors"] = state["errors"]
    exits = [{"cell": w.get("cell"), "to": w.get("dest_map_name")}
             for w in (state.get("collision") or {}).get("warps") or []]
    if exits:
        out["exits"] = exits
    return out


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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class HermesDriver:
    def __init__(self, server: str, model: Optional[str], provider: Optional[str],
                 turn_delay: float = 1.5, save_every: int = 20,
                 turn_timeout: int = 240, use_laya: bool = False,
                 laya_narrate_every: int = 0):
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
        self.map_changed_at: int = -99

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
            self.prev_map, self.map_changed_at = cur_map, self.turn
        key = (cur_map, pos.get("x"), pos.get("y"))
        self.visits[key] = self.visits.get(key, 0) + 1

        # The candidate set is built from ground truth, so an illegal move is
        # not merely discouraged — it is never offered.
        criteria: Dict[str, str] = {}
        if intro or dlg.get("text_active"):
            instructions = ("A text box or menu is on screen. Choose the single "
                            "best button press to advance it.")
            criteria["advance_text"] = "Press A to advance dialog, menu or intro"
            if intro:
                criteria["menu_down_a"] = "Move the menu cursor down, then confirm"
                criteria["back_out"] = "Press B to leave this menu"
        elif battle.get("in_battle"):
            instructions = ("You are in a Pokemon battle. Choose the single best "
                            "menu action for this turn.")
            criteria["advance_text"] = "Press A to confirm the highlighted option"
            criteria["menu_down_a"] = "Move down one option, then confirm"
            criteria["back_out"] = "Press B to go back"
        else:
            instructions = (
                "You are exploring Pokemon Red. The MAP shows what is around you: "
                "'.' is walkable, '#' is a wall, 'N' is a person blocking you, "
                "'D' is a door or exit, '@' is you. Choose the single best move to "
                "make progress — head for an exit when you have explored the room.")
            recent_transition = (self.turn - self.map_changed_at) < 6
            # Block the warp we just came through. Suppressing only the named
            # goto_ option is not enough — walking onto the tile triggers it too.
            if recent_transition and warps:
                passable = [list(r) for r in passable]      # copy before mutating
                for w in warps:
                    r, c = w.get("row"), w.get("col")
                    if r is not None and 0 <= r < len(passable) and 0 <= c < len(passable[r]):
                        passable[r][c] = False
            for name in _legal_dirs(passable):
                criteria[name] = f"Move one tile {name.split('_')[1]}"
            criteria["interact"] = "Press A to talk to or examine what you face"
            for w in warps[:4]:
                cell = w.get("cell")
                if not cell or (w.get("row"), w.get("col")) == (PLAYER_ROW, PLAYER_COL):
                    continue
                if recent_transition:
                    continue      # just arrived; don't offer the way back
                dest = w.get("dest_map_name") or f"map {w.get('dest_map')}"
                criteria[f"goto_{cell}"] = f"Walk to the exit at {cell} leading to {dest}"

        if not criteria:
            instructions = "Nothing is possible right now. Press B."
            criteria["back_out"] = "Press B"

        laya_state = {
            "map_name": (state.get("map") or {}).get("map_name", ""),
            "x": (p.get("position") or {}).get("x"),
            "y": (p.get("position") or {}).get("y"),
            "facing": p.get("facing"),
            "in_battle": bool(battle.get("in_battle")),
            "text_active": bool(dlg.get("text_active")),
            "phase": (state.get("context") or {}).get("phase"),
            "map_ascii": col.get("ascii") or "",
            "exits": [w.get("cell") for w in warps],
            "stuck_turns": self.stuck,
        }

        # Oscillation guard: cycling through the same couple of positions means
        # the candidate set is wrong for this situation, not that Laya is
        # unlucky. Only meaningful while exploring — a static position is
        # normal and expected in battle and during dialog.
        exploring = not (intro or dlg.get("text_active") or battle.get("in_battle"))
        if exploring:
            self.recent.append((laya_state["map_name"],
                                laya_state["x"], laya_state["y"]))
        # Do NOT clear on dialog turns — a text box mid-cycle is part of the
        # cycle, and clearing means the window never fills.
        if len(self.recent) == self.recent.maxlen:
            maps = {m for m, _, _ in self.recent}
            cells = set(self.recent)
            # Ping-ponging between two maps is a loop even though the cells differ.
            if len(cells) <= 3 or (len(maps) == 2 and len(cells) <= 6):
                legal = list(_legal_dirs(passable))
                escape = legal[self.turn % len(legal)] if legal else "press_b"
                logger.warning("loop detected (%d maps, %d cells) — escaping via %s",
                               len(maps), len(cells), escape)
                self.recent.clear()
                self.act([escape] * 4)
                return
        else:
            self.recent.clear()

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
            actions = ["a_until_dialog_end"] if dlg.get("text_active") else ["press_a"]
        elif choice == "menu_down_a":
            actions = ["press_down", "press_a"]
        elif choice == "back_out":
            actions = ["press_b"]
        elif choice == "interact":
            actions = ["press_a"]
        elif choice in _DIRS:
            actions = [choice]
        elif choice.startswith("goto_"):
            label = choice[5:]
            target = next(((w["row"], w["col"]) for w in warps
                           if w.get("cell") == label), None)
            if target == (PLAYER_ROW, PLAYER_COL):
                actions = ["press_a"]     # already on the warp tile
            elif target is not None:
                actions = _path_to(passable, target)
            if not actions:
                logger.warning("no path to %s; taking a single step instead", label)
                legal = list(_legal_dirs(passable))
                actions = [legal[0]] if legal else ["wait_30"]
        else:
            logger.warning("unhandled laya choice %r — waiting", choice)
            actions = ["wait_30"]

        logger.info("turn %d: laya=%s (p=%.2f) in %.0fms -> %s (from %d options) "
                    "phase=%s stuck=%d",
                    self.turn + 1, choice, self.last_confidence, took_ms, actions,
                    len(criteria), laya_state["phase"], self.stuck)

        self.act(actions)
        self.event(type="decision",
                   text=f"[laya {took_ms:.0f}ms] {choice} → {' · '.join(actions)}")

    # --- Hermes ------------------------------------------------------------

    def _hermes_turn(self, state: Dict[str, Any], intro: bool) -> None:
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
        if intro:
            have_img = self._fetch_frame("/screenshot", img_path)
        elif self.stuck >= 2:
            logger.info("position unchanged for %d turns — attaching frame", self.stuck)
            have_img = (self._fetch_frame("/screenshot/grid?scale=2", img_path)
                        or self._fetch_frame("/screenshot", img_path))

        if intro:
            prompt = INTRO_NUDGE.format(
                server=self.server,
                phase=ctx.get("phase", "unknown"),
                vision=INTRO_VISION_OK if have_img else INTRO_VISION_NONE,
            )
        else:
            prompt = TURN_NUDGE.format(
                server=self.server,
                map_ascii=map_ascii,
                state=json.dumps(_compact_state(state), indent=2),
            )

        cmd = ["hermes", "chat", "-Q", "--yolo", "--pass-session-id",
               "-s", "pokemon-player", "-t", "file,terminal,web,vision"]
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

        if self.use_laya:
            narrate = (self.laya_narrate_every
                       and (self.turn + 1) % self.laya_narrate_every == 0)
            if narrate:
                self._hermes_turn(state, intro)
            else:
                self._laya_turn(state, intro)
        else:
            self._hermes_turn(state, intro)

        self.turn += 1
        if self.save_every and self.turn % self.save_every == 0:
            self.save_game()

    # --- main loop ---------------------------------------------------------

    def run(self) -> None:
        # Laya does not shell out to hermes, so a hermes preflight failure
        # must not kill a Laya run. Narration turns still need it, though.
        needs_hermes = (not self.use_laya) or self.laya_narrate_every > 0
        if needs_hermes:
            if not self.preflight():
                self.event(type="alert",
                           text="Hermes preflight failed — see driver log.")
                sys.exit(1)
        else:
            logger.info("Laya-only mode — skipping Hermes preflight")

        brain = "laya" if self.use_laya else "hermes"
        if self.use_laya and self.laya_narrate_every:
            brain = f"laya + hermes every {self.laya_narrate_every} turns"
        print(f"[driver] autopilot. server={self.server} brain={brain} "
              f"model={self.model or 'config default'}")
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

            # Re-check immediately before acting: a long turn can span a STOP.
            if self.control_state() != "running":
                continue
            try:
                self.step()
            except Exception as exc:
                if self.consecutive_fail == 0:
                    logger.exception("laya predict failed")
                else:
                    logger.error("laya predict failed: %s", exc)
                return None


def run_autopilot(server: str = "http://localhost:8765",
                  model: Optional[str] = None,
                  turn_delay: float = 1.5,
                  turn_timeout: int = 240,
                  save_every: int = 20,
                  debug: bool = False,
                  use_laya: bool = False,
                  laya_narrate_every: int = 0) -> None:
    if debug:
        logging.getLogger("pokemon-agent").setLevel(logging.DEBUG)
        logger.info("debug logging enabled")
    model = model or os.environ.get("POKEMON_HERMES_MODEL")
    provider = os.environ.get("POKEMON_HERMES_PROVIDER")
    HermesDriver(server, model, provider,
                 turn_delay=turn_delay, turn_timeout=turn_timeout,
                 save_every=save_every, use_laya=use_laya,
                 laya_narrate_every=laya_narrate_every).run()