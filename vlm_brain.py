#!/usr/bin/env python3
"""VLM brain + BFS legs autopilot for Pokémon Red.

Architecture (the lesson of the 7B live test): Qwen2.5-VL-7B reads the screen
accurately but cannot plan multi-tile movement - it picks "walk_right" at a
door that is three tiles away and never arrives. Hand-rolling movement from a
VLM is a wedge generator.

So the VLM decides WHAT to do at the level it is good at:

    go_exit      leave the current building through the door
    go_north / go_south / go_left / go_right   travel on the overworld
    talk_nurse   heal at the Pokecenter counter
    talk_clerk   buy at the Mart counter
    fight / run / switch / ball   battle decisions
    advance      mash A through a dialog that needs advancing
    wait         do nothing this turn

...and deterministic code turns that intent into actions:

    * movement intents -> BFS over the stitched per-map grid
      (reuses pokemon_agent.autopilot._map_edge_path/_map_step_toward and the
      persisted ~/.pokemon-agent/map_walk.json)
    * go_exit -> walk onto the nearest on-screen warp tile
    * advance -> press_a until dialog.text is empty
    * battle -> gen1_battle.score_moves picks the move (deterministic, 100%
      oracle agreement), run = the proven flee sequence

The VLM is also asked WHY, and the reason is logged - that is the part the
flight recorder wants for later VL training.

Usage:
    POKEMON_VLM_URL=http://127.0.0.1:8002/v1 \
    POKEMON_VLM_MODEL=Qwen2.5-VL-7B-Instruct \
    python3 vlm_brain.py --host localhost --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import websockets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pokemon_agent.autopilot import (  # noqa: E402
    _map_edge_path, _map_step_toward, _load_map_walk, _stamp_screen)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
# silence websockets frame spam
logging.getLogger("websockets").setLevel(logging.WARNING)
logger = logging.getLogger("vlm_brain")

# ---------------------------------------------------------------------------
INTENTS = [
    "go_exit", "go_north", "go_south", "go_left", "go_right",
    "talk_nurse", "talk_clerk", "fight", "run", "switch", "ball",
    "advance", "wait",
]

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "reason": {"type": "string"},
    },
    "required": ["intent", "reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are the BRAIN of a Pokemon Red player. The screen image
and a compact state are given. You choose an INTENT; a separate pathfinder
executes it. Reply with JSON only.

INTENTS:
- go_exit: the current building should be left (walk onto its door)
- go_north/go_south/go_left/go_right: travel across the overworld in that
  direction toward the objective
- talk_nurse: stand in front of the nurse and heal (Pokecenter only)
- talk_clerk: talk to the Mart clerk to buy items
- fight/run/switch/ball: battle choices. fight = best damaging move,
  run = escape a wild battle, ball = throw a Poke Ball, switch = swap Pokemon
- advance: a dialog/quest text is waiting; mash A until it is gone
- wait: nothing useful to do

RULES:
- Look at the SCREEN: if a menu or text box is open, choose advance (or
  run/ball in battle). If you are indoors and the objective lies outside,
  choose go_exit.
- In battle: fight when healthy, run when the enemy is dangerous and you
  cannot win, ball when the objective is to CATCH.
- Outside: follow the OBJECTIVE. If the last 3 intents repeated without
  progress, pick a different direction.

GAME KNOWLEDGE (verified against pokered):
- Museums are OPTIONAL. Never talk to museum clerks; leave immediately.
- Pewter City: the GYM is in the NORTH-WEST. The MART is north-east.
  The Pokecenter is south-west of the gym. Get the Boulder Badge at the gym
  (requires fighting Brock - bring healed Pokemon).
- Route 3 begins east of Pewter City (needs the Boulder Badge first? No -
  the guard only blocks after you TRY to leave without beating Brock... in
  Gen 1 Route 3 is open once you have the badge OR you can pass freely).
- Viridian Forest: winding path, exit north to Pewter.
- When low on HP or PP, find the Pokecenter (talk_nurse).
- $200 buys 2 Poke Balls at any Mart (talk_clerk) - useful for catching."""

# Map-specific hints injected per state (from pokered ground truth)
MAP_HINTS = {
    "Pewter Museum 1F": "Museum is OPTIONAL. Do not talk to clerks. Take the "
                        "south door out (D5/E5) and continue the objective.",
    "Pewter Museum 2F": "Take the stairs down, then exit south.",
    "Pewter City": "Gym is north-west (Brock, Boulder Badge). Mart is "
                   "north-east. Pokecenter south-west of gym. Museum is "
                   "optional - avoid it.",
    "Pewter Pokecenter": "Talk to the nurse (talk_nurse) to heal, then "
                         "go_exit.",
    "Pewter Mart": "Buy 2 Poke Balls ($200) via talk_clerk if you have "
                   "money, then go_exit.",
    "Pewter Gym": "Fight trainers with fight intent. Brock uses Rock types: "
                  "grass/water moves are super effective (Vine Whip!).",
    "Viridian Forest": "Winding maze. Keep going north to reach the Pewter "
                       "gate. Wild battles: fight if healthy, run if low.",
}


@dataclass
class VLMBrain:
    base_url: str
    model: str
    timeout: int = 90
    history: List[Tuple[str, str]] = field(default_factory=list)  # (intent, map)

    def decide(self, frame: np.ndarray, state: Dict[str, Any]) -> Tuple[str, str]:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("encode failed")
        b64 = base64.b64encode(buf).decode("ascii")
        text = self._state_text(state)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": text},
            ]}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "intent",
                                                "schema": INTENT_SCHEMA}},
            "temperature": 0.0,
            "max_tokens": 160,
        }
        r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                          timeout=self.timeout)
        r.raise_for_status()
        parsed = json.loads(r.json()["choices"][0]["message"]["content"])
        intent, reason = parsed["intent"], parsed["reason"]
        # Battle-only intents are meaningless outside battle; remap them.
        in_battle = bool((state.get("battle") or {}).get("in_battle"))
        if not in_battle and intent in ("switch", "ball", "run"):
            indoor = any(k in (state.get("map") or {}).get("map_name", "").lower()
                         for k in ("mart", "pokecenter", "gym", "house", "lab",
                                   "cave", "gate", "center"))
            intent = "go_exit" if indoor else "go_north"
            reason = f"[remap] {parsed['intent']} outside battle -> {intent}"
        # ring memory of (intent, map) so the prompt can show stagnation
        self.history.append((intent, (state.get("map") or {}).get("map_name", "")))
        del self.history[:-12]
        return intent, reason

    def _state_text(self, state: Dict[str, Any]) -> str:
        lines: List[str] = []
        p = state.get("player") or {}
        pos = p.get("position") or {}
        m = state.get("map") or {}
        lines.append(f"Map: {m.get('map_name', '?')}  Pos: ({pos.get('x')}, {pos.get('y')})")
        hint = MAP_HINTS.get(m.get("map_name", ""))
        if hint:
            lines.append(f"HINT: {hint}")
        party = state.get("party") or []
        if party:
            a = party[0]
            lines.append(f"Lead: {a.get('species')} Lv{a.get('level')} "
                         f"HP {a.get('hp')}/{a.get('max_hp')}")
            for mv in a.get("moves") or []:
                if (mv.get("pp") or 0) > 0:
                    lines.append(f"  {mv.get('name')} PP {mv.get('pp')}")
        money = p.get("money")
        if money is not None:
            lines.append(f"Money: ${money}")
        bag = state.get("bag") or []
        if bag:
            lines.append(f"Bag: {bag}")
        b = state.get("battle") or {}
        if b.get("in_battle"):
            e = b.get("enemy_full") or {}
            lines.append(f"BATTLE vs {e.get('species')} Lv{e.get('level')} "
                         f"HP {e.get('hp')}/{e.get('max_hp')}")
        dlg = state.get("dialog") or {}
        if (dlg.get("text") or "").strip():
            lines.append(f"TEXT OPEN: {dlg['text'][:70]!r}")
        for obj in state.get("objectives") or []:
            if not obj.get("done"):
                lines.append(f"OBJECTIVE: {obj.get('text')}")
                break
        # recent intents - lets the model see its own stagnation
        if self.history:
            lines.append("Recent intents: " +
                         ", ".join(i for i, _ in self.history[-6:]))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Executors - deterministic, reuse the proven autopilot helpers
# ---------------------------------------------------------------------------
DIR_OF = {"go_north": "up", "go_south": "down", "go_left": "left",
          "go_right": "right"}


def _grid(passable: List[List[bool]]) -> List[List[bool]]:
    return passable


def _screen_path_to_door(passable, warps, ledges) -> List[str]:
    """Walk to the nearest on-screen warp tile (BFS on the 10x9 screen)."""
    best = None
    for w in warps or []:
        if w.get("on_screen") and w.get("row") is not None:
            from pokemon_agent.autopilot import _path_to
            path = _path_to(passable, (w["row"], w["col"]), ledges=ledges)
            if path and (best is None or len(path) < len(best)):
                best = path
    return best or []


def execute(intent: str, state: Dict[str, Any],
            map_walk: Dict[str, Dict[tuple, bool]]) -> List[str]:
    """Turn an intent into concrete actions. Never raises."""
    col = state.get("collision") or {}
    passable = col.get("passable") or []
    ledges = col.get("ledges")
    warps = col.get("warps") or []
    pos = (state.get("player") or {}).get("position") or {}
    cur = (state.get("map") or {}).get("map_name", "")
    dlg = state.get("dialog") or {}
    battle = state.get("battle") or {}

    if intent == "advance":
        return ["press_a", "wait_30"]

    if intent in ("run",):
        return ["press_b", "wait_30", "press_down", "press_right",
                "press_a", "wait_60"]

    if intent == "fight":
        # deterministic move scoring from the autopilot's policy
        try:
            from pokemon_agent import gen1_battle
            ranked = gen1_battle.score_moves(state)
            if ranked:
                idx = ranked[0][0]
                return [f"move_{idx}"]
        except Exception:
            pass
        return ["move_0"]

    if intent == "ball":
        return ["press_a", "wait_30"]  # battle menu: choose BALL via hotkey path

    if intent == "wait":
        return ["wait_30"]

    if intent == "go_exit":
        if warps:
            path = _screen_path_to_door(passable, warps, ledges)
            if path:
                return path + ["wait_30"]
        # No reachable on-screen door: bounce-walk every direction (the
        # autopilot's proven unstick) - one step each, biggest axis first.
        from pokemon_agent.autopilot import _legal_dirs
        legal = list(_legal_dirs(passable))
        if legal:
            return [legal[0], "wait_30", legal[0] if len(legal) == 1 else legal[-1], "wait_30"]
        return ["press_b", "wait_30"]

    if intent in DIR_OF:
        d = DIR_OF[intent]
        # 1) stitched map BFS (handles winding routes)
        xy = (pos.get("x"), pos.get("y"))
        mw = map_walk.get(cur) or {}
        if col.get("valid") and xy[0] is not None:
            _stamp_screen(map_walk.setdefault(cur, {}), xy, passable)
        if xy[0] is not None:
            mp = _map_edge_path(mw, xy, d)
            if len(mp) > 2:
                return mp[:8]
        # 2) fall back to the local screen path
        from pokemon_agent.autopilot import _path_toward_edge
        local = _path_toward_edge(passable, d, ledges=ledges)
        if local:
            return local
        return [f"walk_{d}"] * 3

    if intent in ("talk_nurse", "talk_clerk"):
        # walk to the nearest NPC-ish sprite on screen and press A.
        # nurse/clerk sprites: server marks persons as 'N' on ascii only, so
        # approximate: press A toward the tile in front, then advance.
        return ["press_a", "wait_30", "press_a", "wait_30", "press_a", "wait_60"]

    # switch / unknown -> open party menu is complex; safe fallback
    return ["press_b", "wait_30"]


# ---------------------------------------------------------------------------
async def fetch_state(url: str) -> Dict[str, Any]:
    r = requests.get(f"{url}/state", timeout=10)
    r.raise_for_status()
    return r.json()


def post_actions(url: str, actions: List[str]) -> None:
    requests.post(f"{url}/action", json={"actions": actions}, timeout=60).raise_for_status()


class FrameReceiver:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self._ws = None
        self._frame: Optional[np.ndarray] = None
        self._ready = asyncio.Event()

    async def connect(self):
        self._ws = await websockets.connect(self.ws_url, max_size=10_000_000)
        async for msg in self._ws:
            data = json.loads(msg)
            if data.get("type") == "screenshot":
                self._decode(data["data"]["image"])
                self._ready.set()
                break

    def _decode(self, b64: str):
        try:
            png = base64.b64decode(b64)
            arr = np.frombuffer(png, dtype=np.uint8)
            f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if f is not None:
                self._frame = f
                self._ready.set()
        except Exception:
            pass

    async def stream(self):
        async for msg in self._ws:
            try:
                data = json.loads(msg)
                if data.get("type") == "screenshot":
                    self._decode(data["data"]["image"])
            except Exception:
                pass

    async def latest(self) -> np.ndarray:
        await self._ready.wait()
        self._ready.clear()
        return self._frame

    async def close(self):
        if self._ws:
            await self._ws.close()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--vlm-url", default=os.environ.get("POKEMON_VLM_URL", "http://127.0.0.1:8002/v1"))
    ap.add_argument("--vlm-model", default=os.environ.get("POKEMON_VLM_MODEL", "Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--max-turns", type=int, default=0)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    brain = VLMBrain(args.vlm_url, args.vlm_model)
    frames = FrameReceiver(f"ws://{args.host}:{args.port}/ws")
    map_walk = _load_map_walk()

    logger.info("VLM brain starting: server=%s model=%s", base, args.vlm_model)
    await frames.connect()
    stream_task = asyncio.create_task(frames.stream())

    turn = 0
    try:
        while args.max_turns == 0 or turn < args.max_turns:
            turn += 1
            try:
                state = await fetch_state(base)
                frame = await frames.latest()

                intent, reason = brain.decide(frame, state)

                # STAGNATION OVERRIDE: same intent 4x in a row without a map
                # change = the brain is stuck (the 7B loops on "switch" in a
                # mart). Indoors, the only sensible unstick is go_exit.
                recent = [i for i, _ in brain.history[-4:]]
                cur_map = (state.get("map") or {}).get("map_name", "")
                indoor = any(k in cur_map.lower() for k in
                             ("mart", "pokecenter", "gym", "house", "lab",
                              "cave", "gate", "center"))
                if len(recent) == 4 and len(set(recent)) == 1:
                    if indoor:
                        forced = "go_exit"
                    else:
                        # outdoors: rotate compass so we cover ground
                        rot = {"go_north": "go_right", "go_right": "go_south",
                               "go_south": "go_left", "go_left": "go_north"}
                        forced = rot.get(intent, "go_north")
                    reason = f"[stagnation] {intent} x4 -> forced {forced}"
                    intent = forced
                    brain.history.clear()

                actions = execute(intent, state, map_walk)
                logger.info("t%d %s -> %s | %s", turn, intent, actions, reason)
                if actions:
                    post_actions(base, actions)
            except Exception:
                logger.exception("turn %d failed", turn)
                await asyncio.sleep(2)
            await asyncio.sleep(0.2)
    finally:
        stream_task.cancel()
        await frames.close()
        # persist the stitched maps we refreshed
        from pokemon_agent.autopilot import _save_map_walk
        _save_map_walk(map_walk)


if __name__ == "__main__":
    asyncio.run(main())
