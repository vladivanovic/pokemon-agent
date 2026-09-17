"""Standalone driver that lets **Hermes Agent** play Pokemon through a session.

This is NOT a raw-LLM loop. The brain is a real Hermes Agent session — with
the `pokemon-player` skill, vision, memory, and the terminal tool — driven one
turn at a time. The driver is intentionally thin:

  loop while /control == "running":
      hermes chat --resume <session> --yolo -s pokemon-player \\
        --image <grid screenshot> -q "<turn nudge + compact state + ascii map>"

Hermes itself does the work each turn: it reads the state/map we hand it (and
can curl the server for more), looks at the grid screenshot with its own
vision, decides, then calls the game server's HTTP API with its terminal tool
to POST /action and POST /event (narration) and POST /objectives. Because we
pass --resume with a single persistent session id, Hermes keeps memory and
context across the whole playthrough — it is "running through a session."

The loop is gated by the server's /control state (Start/Pause/Stop buttons).

Config (env, optional):
  POKEMON_HERMES_MODEL     model override passed to `hermes chat -m`
  POKEMON_HERMES_PROVIDER  provider override passed to `hermes chat --provider`
"""

from __future__ import annotations

import logging
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pokemon-agent.autopilot")

# What Hermes is told once at the start of the session, then nudged each turn.
TURN_NUDGE = """You are playing Pokémon Red live on the Hermes Plays Pokémon dashboard.

The game server is at {server}. Take ONE short turn now, then stop and reply.

This turn:
1. Look at the attached screenshot (it is the current game view).
2. Use the game state to decide a move.
3. Narrate to the stream, then act, using the terminal tool with curl:
   - POST {server}/event  body {{"type":"reasoning","text":"..."}}  (what you see)
   - POST {server}/event  body {{"type":"decision","text":"..."}}   (your plan)
   - POST {server}/action body {{"actions":["walk_down","walk_down"]}} (2-4 moves)
   - On a real beat (new town/badge/item/catch): POST {server}/event
     body {{"type":"key_moment","description":"...","category":"milestone|badge|catch"}}
   - If your goals change: POST {server}/objectives body
     {{"objectives":[{{"tier":"primary","text":"...","done":false}}, ...]}}
   All POSTs need  -H 'Content-Type: application/json'.
4. Keep it to 2-4 game actions this turn — you'll get another turn next.

CURRENT STATE:
{state}

Take your turn now."""

FIRST_TURN_PREFIX = """This is the start of your Pokémon Red run. First, set your objectives by
POSTing to {server}/objectives (primary/secondary/tertiary tiers), then take
your first turn as described below.

"""


def _compact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    p = state.get("player", {}) or {}
    party = []
    for m in state.get("party", []) or []:
        party.append({
            "nickname": m.get("nickname"), "species": m.get("species"),
            "level": m.get("level"), "hp": m.get("hp"), "max_hp": m.get("max_hp"),
            "status": m.get("status"), "types": m.get("types"),
            "moves": [mv.get("name") if isinstance(mv, dict) else mv for mv in m.get("moves", [])],
        })
    battle = state.get("battle") or {}
    enemy = battle.get("enemy") or {}
    return {
        "map": (state.get("map") or {}).get("map_name"),
        "position": p.get("position"), "facing": p.get("facing"),
        "cell": (state.get("collision") or {}).get("player_cell", "E5"),
        "money": p.get("money"), "badges": p.get("badges"),
        "party": party,
        "dialog_active": (state.get("dialog") or {}).get("active"),
        "in_battle": battle.get("in_battle"),
        "enemy": ({"species": enemy.get("species"), "level": enemy.get("level"),
                   "hp": enemy.get("hp"), "max_hp": enemy.get("max_hp")}
                  if battle.get("in_battle") else None),
        "context": (state.get("context") or {}).get("phase"),
        "status": state.get("status"),
        "errors": state.get("errors") or None,
        "active_mon": state.get("active_mon"),
        "map_ascii": (state.get("collision") or {}).get("ascii"),
    }


class HermesDriver:
    def __init__(self, server: str, model: Optional[str], provider: Optional[str],
                 turn_delay: float = 1.5, save_every: int = 20,
                 turn_timeout: int = 240):
        self.server = server.rstrip("/")
        self.model = model
        self.provider = provider
        self.turn_delay = turn_delay
        self.save_every = save_every
        self.turn_timeout = turn_timeout
        self.game_id: Optional[str] = None        # active game session id
        self.session_id: Optional[str] = None     # bound Hermes session id
        self.turn = 0

    # --- server helpers ---
    def _get(self, path: str):
        r = requests.get(self.server + path, timeout=15)
        r.raise_for_status()
        return r

    def control_state(self) -> str:
        try:
            return self._get("/control").json().get("state", "stopped")
        except Exception:
            return "stopped"

    def sync_active_game(self) -> None:
        """Read the active game session and adopt its id + Hermes brain id.

        This is how 'load game' on the dashboard takes effect: the driver
        resumes the SAME Hermes session that game was played with, and scopes
        its work to that game.
        """
        try:
            cur = self._get("/games/current").json().get("active")
        except Exception:
            cur = None
        if not cur:
            self.game_id = None
            return
        if cur.get("id") != self.game_id:
            # switched to a different game (new or loaded) — adopt its brain
            self.game_id = cur.get("id")
            self.session_id = cur.get("hermes_session_id")  # may be None for a new game
            print(f"[driver] active game: {self.game_id} (hermes={self.session_id})")

    def event(self, **kw):
        try:
            requests.post(self.server + "/event", json=kw, timeout=15)
        except Exception:
            pass

    def bind_hermes(self):
        if self.game_id and self.session_id:
            try:
                requests.post(self.server + f"/games/{self.game_id}/hermes",
                              json={"hermes_session_id": self.session_id}, timeout=15)
            except Exception:
                pass

    def save_game(self) -> None:
        """Autosave into the active session's dir."""
        try:
            r = requests.post(self.server + "/save",
                              json={"name": f"turn_{self.turn:06d}"}, timeout=30)
            r.raise_for_status()
            logger.info("autosaved at turn %d", self.turn)
        except Exception as exc:
            logger.warning("autosave failed: %s", exc)

    def check_health(self) -> bool:
        """Check if server and emulator are ready."""
        try:
            return self._get("/health").json().get("status") == "ok"
        except Exception:
            return False

    def step(self) -> None:
            if not self.check_health():
                logger.error("Server down or emulator not ready, pausing driver.")
                time.sleep(5)
                return

            try:
                state = self._get("/state").json()
            except Exception as e:
                logger.error(f"State read failed: {e}")
                time.sleep(2)
                return

            # Readiness: not_ready or not in_game
            if state.get("status") == "not_ready" or not (
                    state.get("context") or {}).get("in_game"):
                logger.info("Not in game (%s), waiting...",
                            (state.get("context") or {}).get("phase"))
                time.sleep(2)
                return

            logger.debug(f"State: {state}")

            # Grab the full screenshot for vision model analysis.
            img_path = str(Path(tempfile.gettempdir()) / "pokemon_turn.png")
            try:
                shot = self._get("/screenshot/grid?scale=4").content
                if not shot.startswith(b"\x89PNG"):
                    raise ValueError(f"not a PNG ({shot[:40]!r})")
                with open(img_path, "wb") as f:
                    f.write(shot)
                have_img = True
                logger.debug(f"Screenshot taken, saved to {img_path}")
            except Exception as e:
                logger.warning(f"Screenshot failed: {e}")
                have_img = False

            prompt = TURN_NUDGE.format(
                server=self.server,
                state=json.dumps(_compact_state(state), indent=2),
            )
            if self.session_id is None:
                prompt = FIRST_TURN_PREFIX.format(server=self.server) + prompt

            cmd = ["hermes", "chat", "-Q", "--yolo", "--pass-session-id",
                   "-s", "pokemon-player",
                   "-t", "file,terminal,web,vision"]
            if self.session_id:
                cmd += ["--resume", self.session_id]
            if self.model:
                cmd += ["-m", self.model]
            if self.provider:
                cmd += ["--provider", self.provider]
            if have_img:
                cmd += ["--image", img_path]
            cmd += ["-q", prompt]

            logger.info(f"Triggering Hermes: {' '.join(cmd)}")

            try:
                out = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=self.turn_timeout)
                stdout = out.stdout or ""
                stderr = out.stderr or ""
                logger.debug(f"Hermes stdout: {stdout}")
                logger.debug(f"Hermes stderr: {stderr}")
            except subprocess.TimeoutExpired:
                logger.error("Hermes turn timed out")
                self.event(type="alert", text="Turn timed out — retrying.")
                return
            except Exception as e:
                logger.error(f"Hermes invocation failed: {e}")
                self.event(type="alert", text=f"Driver error: {e}")
                time.sleep(3)
                return

            # Capture the session id from the first run so later turns resume it.
            if self.session_id is None:
                # Session ID is printed to stderr in format: "session_id: <id>"
                combined = stdout + "\n" + stderr
                m = re.search(r"session_id:\s*(\S+)", combined) or \
                    re.search(r"hermes --resume (\S+)", combined) or \
                    re.search(r"Session:\s*(\S+)", combined)
                if m:
                    self.session_id = m.group(1)
                    logger.info(f"Hermes session: {self.session_id}")
                    self.bind_hermes()
                    self.event(type="key_moment",
                               description="Hermes session started",
                               category="milestone")
                else:
                    logger.warning("Failed to extract session_id from Hermes output")
                    logger.debug(f"Combined output searched: {combined[:500]}")

            self.turn += 1
            if self.save_every and self.turn % self.save_every == 0:
                self.save_game()

    def run(self):
        model_note = self.model or "config default"
        print(f"[driver] Hermes-driven autopilot. server={self.server} model={model_note}")
        print("[driver] waiting for control=running + an active game…")
        self.event(type="alert", text="Hermes online — start or load a game, then press START.")
        idle_logged = False
        no_game_logged = False
        while True:
            st = self.control_state()
            if st == "stopped":
                if not idle_logged:
                    print("[driver] stopped — idling.")
                    idle_logged = True
                time.sleep(2)
                continue
            if st == "paused":
                time.sleep(1.5)
                continue
            idle_logged = False
            self.sync_active_game()
            if not self.game_id:
                if not no_game_logged:
                    print("[driver] running but no active game — start/load one on the dashboard.")
                    self.event(type="alert", text="No active game — click New Game or load one.")
                    no_game_logged = True
                time.sleep(2)
                continue
            no_game_logged = False
            self.step()
            time.sleep(self.turn_delay)


def run_autopilot(server: str = "http://localhost:8765", model: Optional[str] = None,
                  turn_delay: float = 1.5, debug: bool = False):
    if debug:
        logging.getLogger("pokemon-agent").setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    model = model or os.environ.get("POKEMON_HERMES_MODEL")
    provider = os.environ.get("POKEMON_HERMES_PROVIDER")
    HermesDriver(server, model, provider, turn_delay=turn_delay).run()
