"""Standalone driver that lets **Hermes Agent** play Pokemon through a session.

This is NOT a raw-LLM loop. The brain is a real Hermes Agent session — with
the `pokemon-player` skill, vision, memory, and the terminal tool — driven one
turn at a time. The driver is intentionally thin:

  loop while /control == "running":
      hermes chat --resume <session> --yolo -s pokemon-player \
        -q "<turn nudge + ascii map + compact state>"

Normal turns are TEXT ONLY. The ASCII collision map in /state is ground truth
read from game RAM, so it beats asking a vision model to read pixel art — and
it keeps the prompt small, which matters a lot on local hardware. Hermes can
fetch a frame itself (curl /screenshot + its vision tool) when the map is not
enough: menus, dialog text, battle screens. The intro/title screens are the
one case where the driver pushes an image, because no map exists there yet.

Because we pass --resume with a single persistent session id, Hermes keeps
memory and context across the whole playthrough.

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
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pokemon-agent.autopilot")


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

  Title screen / NEW GAME      {{"actions":["press_a"]}}
  Oak talking / any text box   {{"actions":["a_until_dialog_end"]}}
  Name menu (NEW NAME/RED/...) {{"actions":["press_down","press_a"]}}
  Options screen (went too far) {{"actions":["press_b"]}}
  Unsure                       {{"actions":["press_a"]}}

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
        self.game_id: Optional[str] = None      # active game session id
        self.session_id: Optional[str] = None   # bound Hermes session id
        self.turn = 0
        self.last_pos: Optional[Any] = None     # stuck detection
        self.stuck = 0

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

    def event(self, **kw) -> None:
        try:
            self._post("/event", kw)
        except Exception:
            pass

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

    # --- one turn ----------------------------------------------------------

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

    def step(self) -> None:
        emu = self.emulator_state()
        if emu != "ready":
            logger.info("emulator %s — waiting", emu)
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

        # Stuck detection: the map says we can move but the position is not
        # changing. Escalates to vision, which usually reveals an unnoticed
        # text box or a sprite the grid missed.
        pos = (state.get("player") or {}).get("position")
        self.stuck = self.stuck + 1 if (pos is not None and pos == self.last_pos) else 0
        self.last_pos = pos

        col = state.get("collision") or {}
        map_ascii = col.get("ascii") or (
            f"(map unavailable: {col.get('reason', 'not built')})")

        # Push an image only when there is no usable map: the intro screens,
        # or when we appear wedged. Ordinary turns are text-only and cheap;
        # Hermes can curl a frame itself when it decides it needs one.
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
                    self.turn + 1, ctx.get("phase"), have_img, self.stuck, len(prompt))
        logger.debug("prompt:\n%s", prompt)

        started = time.perf_counter()
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 stdin=subprocess.DEVNULL,
                                 timeout=self.turn_timeout)
            stdout, stderr = out.stdout or "", out.stderr or ""
            logger.info("turn took %.1fs (rc=%s)", time.perf_counter() - started,
                        out.returncode)
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

        self.turn += 1
        if self.save_every and self.turn % self.save_every == 0:
            self.save_game()

    # --- main loop ---------------------------------------------------------

    def run(self) -> None:
        if not self.preflight():
            self.event(type="alert", text="Hermes preflight failed — see driver log.")
            sys.exit(1)

        print(f"[driver] Hermes-driven autopilot. server={self.server} "
              f"model={self.model or 'config default'}")
        print("[driver] waiting for control=running + an active game…")
        self.event(type="alert",
                   text="Hermes online — start or load a game, then press START.")

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

            # A bad turn must not kill the run.
            try:
                self.step()
            except Exception:
                logger.exception("turn failed — continuing")
                self.event(type="alert", text="Driver error — see log.")
                time.sleep(3)
            time.sleep(self.turn_delay)


def run_autopilot(server: str = "http://localhost:8765",
                  model: Optional[str] = None,
                  turn_delay: float = 1.5,
                  turn_timeout: int = 240,
                  save_every: int = 20,
                  debug: bool = False) -> None:
    if debug:
        logging.getLogger("pokemon-agent").setLevel(logging.DEBUG)
        logger.info("debug logging enabled")
    model = model or os.environ.get("POKEMON_HERMES_MODEL")
    provider = os.environ.get("POKEMON_HERMES_PROVIDER")
    HermesDriver(server, model, provider, turn_delay=turn_delay,
                 turn_timeout=turn_timeout, save_every=save_every).run()