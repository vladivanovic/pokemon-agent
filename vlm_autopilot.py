#!/usr/bin/env python3
"""Standalone VLM autopilot for Pokémon Red.

Replaces the Von/Laya/Hermes hybrid with a single Qwen2-VL-2B call per turn.
No Hermes, no subprocess, no JSON retry loops — native llama.cpp with
json_schema constrained decoding.

Usage:
    POKEMON_VLM_URL=http://127.0.0.1:8002/v1 \
    POKEMON_VLM_MODEL=Qwen2-VL-2B-Instruct \
    python3 vlm_autopilot.py --host localhost --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vlm_autopilot")

# ---------------------------------------------------------------------------
# Action space (must match server's /control expectations)
# ---------------------------------------------------------------------------
WALK_ACTIONS = ["walk_up", "walk_down", "walk_left", "walk_right"]
MENU_ACTIONS = ["press_a", "press_b", "press_start", "press_select"]
WAIT_ACTIONS = ["wait_30", "wait_60"]
ALL_ACTIONS = WALK_ACTIONS + MENU_ACTIONS + WAIT_ACTIONS

# JSON schema for constrained decoding
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ALL_ACTIONS},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# VLM Client
# ---------------------------------------------------------------------------
@dataclass
class VLMClient:
    base_url: str
    model: str
    timeout: int = 30
    # Minimal memory to break loops
    _recent_actions: List[str] = field(default_factory=list)
    _max_recent: int = 8

    def _build_action_schema(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build action schema, filtering out invalid actions for current menu state."""
        actions = ALL_ACTIONS.copy()
        dlg = state.get("dialog") or {}
        # A non-empty text field means SOMETHING is on screen reading input:
        # dialog, START menu, battle menu, save screen. The server's active
        # flags miss the START menu ("OPTION EXIT"), which is how the 2B
        # model got wedged toggling it with press_start forever.
        menu_open = bool((dlg.get("text") or "").strip())
        if menu_open:
            # In a menu: START toggles it, B cancels, arrows navigate, A selects
            for a in ["press_start"]:
                if a in actions:
                    actions.remove(a)
        return {"type": "object", "properties": {"action": {"type": "string", "enum": actions}, "reason": {"type": "string"}}, "required": ["action", "reason"], "additionalProperties": False}

    def decide(self, frame_bgr: np.ndarray, state: Dict[str, Any]) -> Tuple[str, str]:
        """Return (action, reason) from VLM."""
        # Encode frame as base64 JPEG
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Failed to encode frame")
        frame_b64 = base64.b64encode(buf).decode("ascii")

        # Build action schema for current state
        action_schema = self._build_action_schema(state)

        # Build compact state summary
        state_text = self._compact_state(state)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the player in Pokémon Red. You see the current screen "
                    "and a compact game state. Choose ONE action from the legal "
                    "set, then respond with JSON matching the schema.\n\n"
                    "RULES:\n"
                    "1. OBJECTIVE FIRST: you are on a journey (see OBJECTIVE). "
                    "Mostly WALK toward exits/doors; menus are for buying and "
                    "healing only.\n"
                    "2. NEVER press A while facing an NPC unless the objective "
                    "requires talking (nurse, clerk for shopping). Pressing A "
                    "at a random NPC starts a dialog you must then escape with "
                    "B several times.\n"
                    "3. If MENU/DIALOG is open: close it with press_b unless you "
                    "deliberately opened it to buy/heal. Never press_start.\n"
                    "4. Walk into doors (marked D on the ascii map) to exit "
                    "buildings. To go north, walk_up repeatedly."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                    {"type": "text", "text": state_text},
                ],
            },
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": {"name": "action", "schema": action_schema}},
            "temperature": 0.0,
            "max_tokens": 128,
        }

        r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        action = parsed["action"]
        reason = parsed["reason"]

        # Update recent actions memory
        self._recent_actions.append(action)
        if len(self._recent_actions) > self._max_recent:
            self._recent_actions.pop(0)

        return action, reason

    def _compact_state(self, state: Dict[str, Any]) -> str:
        """Tiny state summary for the prompt."""
        lines = []
        player = state.get("player", {})
        pos = player.get("position", {})
        lines.append(f"Map: {(state.get('map') or {}).get('map_name', 'unknown')}")
        lines.append(f"Pos: ({pos.get('x', '?')}, {pos.get('y', '?')})")

        party = state.get("party") or []
        if party:
            active = party[0]
            hp = active.get("hp", "?")
            max_hp = active.get("max_hp", "?")
            lines.append(f"Lead: {active.get('species', '?')} HP {hp}/{max_hp}")
            moves = active.get("moves") or []
            for m in moves:
                if m.get("pp", 0) > 0:
                    lines.append(f"  {m.get('name', '?')} PP {m.get('pp')}")

        bag = state.get("bag") or []
        if bag:
            lines.append(f"Bag: {', '.join(bag[:5])}")
        money = player.get("money")
        if money is not None:
            lines.append(f"Money: ${money}")

        battle = state.get("battle") or {}
        if battle.get("in_battle"):
            enemy = battle.get("enemy_full") or {}
            lines.append(f"BATTLE: vs {enemy.get('species', '?')} Lv{enemy.get('level', '?')}")

        # Menu/dialog awareness - the model must know a menu is open
        dlg = state.get("dialog") or {}
        if (dlg.get("text") or "").strip():
            lines.append(f"⚠️ MENU/DIALOG OPEN (text: {dlg['text'][:60]!r}) — "
                         f"navigate with arrows, select with press_a, close with press_b. "
                         f"Do NOT press_start (it would toggle the menu).")

        # ASCII map - the model can see doors (D), NPCs (N), walls (#)
        col = state.get("collision") or {}
        if col.get("ascii"):
            lines.append("ASCII MAP (@=you, D=door, N=person, #=wall, .=walkable). "
                         "Row 1 is the TOP (north); row 9 is the BOTTOM (south). "
                         "walk_up moves UP a row, walk_down DOWN a row, walk_left "
                         "LEFT a column, walk_right RIGHT a column.")
            lines.append(col["ascii"])
            # Explicit door guidance: list reachable doors with their compass
            # direction from the player, so "go to the door" is actionable.
            doors = []
            for w in col.get("warps") or []:
                if w.get("on_screen") and w.get("row") is not None:
                    wr, wc = w["row"], w["col"]
                    dr, dc = wr - 4, wc - 4  # player is at row 4, col 4 (E5)
                    dirs = []
                    if dr < 0: dirs.append("up")
                    if dr > 0: dirs.append("down")
                    if dc < 0: dirs.append("left")
                    if dc > 0: dirs.append("right")
                    doors.append(f"door at {'+'.join(dirs) if dirs else 'your tile'}")
            if doors:
                lines.append("Doors on screen: " + "; ".join(doors) +
                             ". WALK ONTO the door tile to exit the building.")

        # Recent actions memory - critical for breaking loops
        if self._recent_actions:
            lines.append(f"Recent actions: {', '.join(self._recent_actions)}")
            # Loop breaker hint
            if len(self._recent_actions) >= 4:
                last = self._recent_actions[-1]
                if all(a == last for a in self._recent_actions[-4:]):
                    lines.append("⚠️ LAST 4 ACTIONS IDENTICAL — try something DIFFERENT")

        # Current objective from server
        objectives = state.get("objectives") or []
        for obj in objectives:
            if not obj.get("done"):
                lines.append(f"OBJECTIVE: {obj.get('text', '')}")
                break

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local BFS executor (turns "walk to door at (x,y)" into walk_* sequence)
# ---------------------------------------------------------------------------
DIRS = {
    "walk_up": (-1, 0),
    "walk_down": (1, 0),
    "walk_left": (0, -1),
    "walk_right": (0, 1),
}
REVERSE = {"walk_up": "walk_down", "walk_down": "walk_up",
           "walk_left": "walk_right", "walk_right": "walk_left"}
PLAYER_ROW, PLAYER_COL = 4, 4
GRID_ROWS, GRID_COLS = 9, 10


def _grid_open(grid: List[List[bool]], r: int, c: int) -> bool:
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c]


def _path_to(passable: List[List[bool]], target: Tuple[int, int],
             ledges: Optional[List[List[bool]]] = None,
             limit: int = 20) -> List[str]:
    """BFS from player (E5) to target cell on the 10x9 screen."""
    start = (PLAYER_ROW, PLAYER_COL)
    if target == start:
        return []
    if not _grid_open(passable, *target):
        return []
    seen = {start}
    q = [(start, [])]
    for (r, c), path in q:
        if len(path) >= limit:
            continue
        for name, (dr, dc) in DIRS.items():
            if name == REVERSE.get(path[-1]) if path else False:
                continue
            nr, nc = r + dr, c + dc
            if (nr, nc) in seen or not _grid_open(passable, nr, nc):
                continue
            if ledges and nr < len(ledges) and nc < len(ledges[nr]) and ledges[nr][nc] and name != "walk_down":
                continue
            seen.add((nr, nc))
            new_path = path + [name]
            if (nr, nc) == target:
                return new_path
            q.append(((nr, nc), new_path))
    return []


# ---------------------------------------------------------------------------
# WebSocket frame receiver
# ---------------------------------------------------------------------------
class FrameReceiver:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ready = asyncio.Event()

    async def connect(self):
        self._ws = await websockets.connect(self.ws_url, max_size=10_000_000)
        # Consume initial messages until we get a frame
        async for msg in self._ws:
            data = json.loads(msg)
            if data.get("type") == "screenshot":
                self._decode_frame(data["data"]["image"])
                self._frame_ready.set()
                break

    def _decode_frame(self, b64_png: str):
        try:
            png = base64.b64decode(b64_png)
            arr = np.frombuffer(png, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self._latest_frame = frame
        except Exception as e:
            logger.warning("Frame decode failed: %s", e)

    async def get_latest_frame(self) -> np.ndarray:
        await self._frame_ready.wait()
        frame = self._latest_frame
        self._frame_ready.clear()
        return frame

    async def keep_alive(self):
        """Background task to keep receiving frames."""
        if not self._ws:
            return
        async for msg in self._ws:
            try:
                data = json.loads(msg)
                if data.get("type") == "screenshot":
                    self._decode_frame(data["data"]["image"])
                    self._frame_ready.set()
            except Exception:
                pass

    async def close(self):
        if self._ws:
            await self._ws.close()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def fetch_state(url: str) -> Dict[str, Any]:
    r = requests.get(f"{url}/state", timeout=10)
    r.raise_for_status()
    return r.json()


def post_control(url: str, actions: List[str]) -> None:
    r = requests.post(f"{url}/action", json={"actions": actions}, timeout=10)
    r.raise_for_status()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--vlm-url", default=os.environ.get("POKEMON_VLM_URL", "http://127.0.0.1:8002/v1"))
    ap.add_argument("--vlm-model", default=os.environ.get("POKEMON_VLM_MODEL", "Qwen2-VL-2B-Instruct"))
    ap.add_argument("--max-turns", type=int, default=0, help="0 = infinite")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    ws_url = f"ws://{args.host}:{args.port}/ws"
    vlm = VLMClient(args.vlm_url, args.vlm_model)
    frame_recv = FrameReceiver(ws_url)

    logger.info("VLM autopilot starting: server=%s ws=%s vlm=%s model=%s",
                base, ws_url, args.vlm_url, args.vlm_model)

    # Connect to WebSocket and start frame receiver
    await frame_recv.connect()
    keep_alive_task = asyncio.create_task(frame_recv.keep_alive())

    turn = 0
    # Hard loop-breaker state
    _recent: List[str] = []
    _LOOP_N = 4  # after this many identical actions, override
    _OVERRIDE = {  # deterministic substitute for a stuck action
        "press_start": "press_b",
        "press_a": "press_b",
        "press_b": "walk_down",
        "wait_30": "walk_right",
        "wait_60": "walk_left",
    }
    try:
        while args.max_turns == 0 or turn < args.max_turns:
            turn += 1
            try:
                state = await fetch_state(base)
                frame = await frame_recv.get_latest_frame()

                # VLM decides
                action, reason = vlm.decide(frame, state)

                # HARD LOOP-BREAKER: identical action N times in a row
                # triggers a deterministic override. The 2B model reads the
                # "try something different" hint but does not act on it.
                _recent.append(action)
                del _recent[:-_LOOP_N]
                if len(_recent) == _LOOP_N and len(set(_recent)) == 1:
                    stuck = _recent[-1]
                    alt = _OVERRIDE.get(stuck, "walk_right")
                    reason = f"[loop-breaker] {stuck} x{_LOOP_N} -> forced {alt}"
                    action = alt
                    _recent.clear()

                logger.info("Turn %d: %s — %s", turn, action, reason)

                # Execute
                post_control(base, [action])

            except Exception as e:
                logger.exception("Turn %d failed: %s", turn, e)
                await asyncio.sleep(1)

            await asyncio.sleep(0.1)  # polite delay

    finally:
        keep_alive_task.cancel()
        await frame_recv.close()


if __name__ == "__main__":
    asyncio.run(main())