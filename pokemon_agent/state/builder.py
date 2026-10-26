"""Game-state orchestrator.

:func:`build_game_state` calls every reader method and assembles the
results into a single JSON-serialisable dictionary.

:func:`build_state_summary` renders that dict as a concise text summary
suitable for injection into an LLM prompt.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Optional, Tuple
import logging
from typing import Any, Dict

from pokemon_agent.memory.reader import GameMemoryReader
from pokemon_agent.gen1_data import SPRITE_NURSE
from pokemon_agent.collision import PLAYER_ROW, PLAYER_COL

logger = logging.getLogger("pokemon-agent.state")


def _nurse_on_screen(collision: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    """(dx, dy, dist) to the nearest nurse **on the current screen**, or None.

    Screen-relative: the collision grid is a 10x9 window with the player
    locked to its centre, and sprites are reported in those coordinates.
    `dist` is the BFS step count, so real walking distance, not Manhattan.
    """
    sprites = collision.get("sprites") or []
    nurse_cells = {
        (s.get("row"), s.get("col"))
        for s in sprites
        if s.get("picture_id") == SPRITE_NURSE or s.get("who") == "Nurse"
    }
    nurse_cells.discard((None, None))
    if not nurse_cells:
        return None

    grid = collision.get("walkable") or collision.get("passable") or []
    if not grid:
        return None

    from collections import deque
    start: Tuple[int, int] = (PLAYER_ROW, PLAYER_COL)
    q: "deque[Tuple[int, int, Optional[Tuple[int, int]], int]]" = deque(
        [(start[0], start[1], None, 0)])          # row, col, first step, dist
    seen: set = {start}
    while q:
        y, x, first, dist = q.popleft()
        if (y, x) in nurse_cells:
            if first is None:
                return (0, 0, 0)                 # already on the nurse tile
            return (first[1], first[0], dist)    # dx, dy, dist
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < len(grid) and 0 <= nx < len(grid[0])):
                continue
            if (ny, nx) in seen:
                continue
            if _cell_walkable(grid, ny, nx):
                seen.add((ny, nx))
                q.append((ny, nx, first if first else (dy, dx), dist + 1))
    return None


def _cell_walkable(grid, y: int, x: int) -> bool:
    """The grid mixes '1'/'0' strings and ints depending on the reader."""
    try:
        v = grid[y][x]
    except (IndexError, TypeError):
        return False
    if isinstance(v, str):
        return v not in ("0", "", " ")
    return bool(v)


def annotate_nearest_pokecenter(state: Dict[str, Any],
                                collision: Dict[str, Any]) -> None:
    """Set nearest_pokecenter_* on a state dict that already has collision.

    Called AFTER the collision grid is attached, because that grid is the
    only source for nurse sprites. The previous version lived inside
    build_game_state and read `read_map_info()["collision"]`, which does not
    exist - so it silently returned None on every map.
    """
    # Nurse FIRST, even inside a Pokecenter: knowing we are inside is not
    # enough to walk to the counter, and the nurse is usually on screen once
    # you are in the door. The "inside" early-return used to swallow that and
    # reported dist 0 / dir (0,0), which gave the caller nothing to act on.
    nurse = _nurse_on_screen(collision) if collision.get("valid") else None
    if nurse is not None:
        dx, dy, dist = nurse
        state["nearest_pokecenter_dist"] = dist
        state["nearest_pokecenter_dir"] = (dx, dy)
        state["nearest_pokecenter_src"] = "nurse"
        return

    # Inside a Pokecenter but no nurse visible (looking at the door, or a
    # frame the collision layer rejected): distance 0 means "you have arrived".
    map_name = (state.get("map") or {}).get("map_name") or ""
    if "pokecenter" in map_name.lower():
        state["nearest_pokecenter_dist"] = 0
        state["nearest_pokecenter_dir"] = (0, 0)
        state["nearest_pokecenter_src"] = "inside"
        return

    door = _nearest_pokecenter_door(state)
    if door is not None:
        dx, dy, dist, door_x, door_y = door
        state["nearest_pokecenter_dist"] = dist
        state["nearest_pokecenter_dir"] = (dx, dy)
        state["nearest_pokecenter_src"] = "door"
        state["nearest_pokecenter_x"] = door_x
        state["nearest_pokecenter_y"] = door_y
        return

    state["nearest_pokecenter_dist"] = None
    state["nearest_pokecenter_dir"] = None
    state["nearest_pokecenter_src"] = None


def _load_pc_centers() -> Dict[str, Dict[str, int]]:
    """Load the PC-center door lookup table from pc_centers.json."""
    path = os.path.join(os.path.dirname(__file__), "..", "pc_centers.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Failed to load pc_centers.json: %s", e)
        return {}


PC_CENTERS: Dict[str, Dict[str, int]] = _load_pc_centers()


def _nearest_pokecenter_door(state: Dict[str, Any]) -> Optional[Tuple[int, int, int, int, int]]:
    """Fallback: on a city we have a door coordinate for, return (dx, dy, dist, door_x, door_y).

    Uses MAP coordinates, not screen coordinates - the player position and
    the pc_centers.json door tiles are both map-local, so the direction
    stays correct even when the Pokecenter is off-screen.
    """
    map_name = (state.get("map") or {}).get("map_name")
    if not map_name or map_name not in PC_CENTERS:
        return None
    pos = (state.get("player") or {}).get("position") or {}
    px, py = pos.get("x"), pos.get("y")
    if px is None or py is None:
        return None
    door = PC_CENTERS[map_name]
    dx = door["door_x"] - px
    dy = door["door_y"] - py
    dist = abs(dx) + abs(dy)
    if dist == 0:
        return (0, 0, 0, door["door_x"], door["door_y"])  # standing on the door
    step_dx = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_dy = 0 if dy == 0 else (1 if dy > 0 else -1)
    return (step_dx, step_dy, dist, door["door_x"], door["door_y"])


def build_game_state(
    reader: GameMemoryReader,
    frame_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Read all game data and assemble a complete state snapshot.

    Parameters
    ----------
    reader : GameMemoryReader
        An initialised memory reader bound to a running emulator.
    frame_count : int, optional
        Current emulator frame count (injected into metadata).

    Returns
    -------
    dict
        A JSON-serialisable game-state dictionary.  Sections that fail
        to read are ``None`` with an ``"errors"`` key.
    """
    errors: Dict[str, str] = {}

    def _safe(name: str, fn, default=None):
        try:
            return fn()
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            logger.debug("read %s failed", name, exc_info=True)
            return default

    state: Dict[str, Any] = {
        "metadata": {
            "game": _safe("game_name", lambda: reader.game_name, "unknown"),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "frame_count": frame_count,
        },
    }

    state["context"] = _safe("context", reader.read_context,
                             {"phase": "unknown", "in_game": False})

    for key, attr in (
        ("player", "read_player"), ("party", "read_party"),
        ("bag", "read_bag"), ("battle", "read_battle"),
        ("dialog", "read_dialog"), ("map", "read_map_info"),
        ("flags", "read_flags"),
    ):
        fn = getattr(reader, attr, None)
        if fn is None:
            errors[key] = f"AttributeError: reader has no {attr}"
            state[key] = None
            continue
        state[key] = _safe(key, fn)

    # Mid-battle, wBattleMon is authoritative for the active player mon;
    # wPartyMon only syncs after the fight. Surface the preferred source.
    battle = state.get("battle") or {}
    if battle.get("in_battle") and battle.get("active"):
        state["active_mon"] = battle["active"]
    elif state.get("party"):
        state["active_mon"] = state["party"][0]
    else:
        state["active_mon"] = None

    # ----------------------------------------------------------------------
    #  Nearest Pokémon Center
    # ----------------------------------------------------------------------
    # Deliberately left unset here. Detection needs the collision grid (nurse
    # sprites), and the server attaches that only after this function returns;
    # the previous in-line version read a key that never existed and silently
    # produced None on every map. The server calls
    # annotate_nearest_pokecenter() once collision is available.
    state.setdefault("nearest_pokecenter_dist", None)
    state.setdefault("nearest_pokecenter_dir", None)
    state.setdefault("nearest_pokecenter_src", None)

    state["errors"] = errors
    state["status"] = "degraded" if errors else "ok"
    return state


# -----------------------------------------------------------------------
# Text summary
# -----------------------------------------------------------------------
def build_state_summary(state: Dict[str, Any]) -> str:
    """Render a game state dict as a concise text summary for an LLM prompt.

    Parameters
    ----------
    state : dict
        A dict produced by :func:`build_game_state`.

    Returns
    -------
    str
        Multi-line plain-text summary.
    """
    lines: list[str] = []
    _hr = "=" * 50

    lines.append(_hr)
    lines.append("GAME STATE SNAPSHOT")
    lines.append(_hr)

    # -- metadata --
    meta = state.get("metadata", {})
    ctx = state.get("context") or {}
    lines.append(f"Game      : {meta.get('game', '?')}")
    lines.append(f"Status    : {state.get('status', '?')}")
    lines.append(f"Context   : {ctx.get('phase', 'unknown')}")
    if meta.get("frame_count") is not None:
        lines.append(f"Frame     : {meta['frame_count']}")

    if state.get("errors"):
        for k, v in state["errors"].items():
            lines.append(f"  ! {k}: {v}")

    if not ctx.get("in_game", False):
        lines.append(f"\nNot in game ({ctx.get('phase', 'unknown')}) — "
                     f"gameplay fields are unreliable.")
        lines.append(_hr)
        return "\n".join(lines)

    # -- map --
    map_info = state.get("map")
    if map_info is not None:
        if map_info:
            lines.append(f"Location  : {map_info.get('map_name', '?')} (id={map_info.get('map_id')})")
        else:
            lines.append("Location  : (unknown map)")

    # -- player --
    player = state.get("player")
    if player:
        lines.append("")
        lines.append("--- PLAYER ---")
        lines.append(f"Name    : {player.get('name', '?')}")
        lines.append(f"Rival   : {player.get('rival_name', '?')}")
        money = player.get("money")
        lines.append(f"Money   : ${money:,}" if isinstance(money, int) else "Money   : ?")
        badges = player.get("badges", [])
        lines.append(f"Badges  : {len(badges)} — {', '.join(badges) if badges else 'none'}")
        pos = player.get("position", {})
        lines.append(f"Position: ({pos.get('x', '?')}, {pos.get('y', '?')})  facing {player.get('facing', '?')}")
        lines.append(f"Playtime: {player.get('play_time', '?')}")
    elif state.get("player_error"):
        lines.append(f"\n[Player read error: {state['player_error']}]")

    # -- party --
    party = state.get("party")
    if party is not None:
        lines.append("")
        lines.append("--- PARTY ---")
        if not party:
            lines.append("  (empty)")
        for i, mon in enumerate(party, 1):
            moves_str = ", ".join(
                m["name"] if isinstance(m, dict) else str(m) for m in mon.get("moves", [])
            )
            lines.append(
                f"  {i}. {mon.get('nickname', '?')} "
                f"({mon.get('species', '?')} Lv{mon.get('level', '?')})  "
                f"HP {mon.get('hp', '?')}/{mon.get('max_hp', '?')}  "
                f"Status: {mon.get('status', '?')}"
            )
            lines.append(f"     Moves: {moves_str}")
    elif state.get("party_error"):
        lines.append(f"\n[Party read error: {state['party_error']}]")

    # -- battle --
    battle = state.get("battle")
    if battle and battle.get("in_battle"):
        lines.append("")
        lines.append("--- BATTLE ---")
        lines.append(f"Type: {battle.get('type', '?')}")
        enemy = battle.get("enemy")
        if enemy:
            lines.append(
                f"Enemy: {enemy.get('species', '?')} Lv{enemy.get('level', '?')}  "
                f"HP {enemy.get('hp', '?')}/{enemy.get('max_hp', '?')}  "
                f"Status: {enemy.get('status', '?')}"
            )
            enemy_moves = enemy.get("moves", [])
            if enemy_moves:
                names = [m.get("name", "?") if isinstance(m, dict) else str(m)
                         for m in enemy_moves]
                lines.append(f"Enemy moves: {', '.join(names)}")
    elif battle and not battle.get("in_battle"):
        lines.append("\nNot in battle.")
    elif state.get("battle_error"):
        lines.append(f"\n[Battle read error: {state['battle_error']}]")

    # -- dialog --
    dialog = state.get("dialog")
    if dialog and dialog.get("active"):
        lines.append("")
        lines.append("--- DIALOG ---")
        lines.append("Text box is active.")
    elif state.get("dialog_error"):
        lines.append(f"\n[Dialog read error: {state['dialog_error']}]")

    # -- bag --
    bag = state.get("bag")
    if bag is not None:
        lines.append("")
        lines.append("--- BAG ---")
        if not bag:
            lines.append("  (empty)")
        for entry in bag:
            lines.append(f"  {entry.get('item', '?')} x{entry.get('quantity', '?')}")
    elif state.get("bag_error"):
        lines.append(f"\n[Bag read error: {state['bag_error']}]")

    # -- flags --
    flags = state.get("flags")
    if flags is not None:
        lines.append("")
        lines.append("--- FLAGS ---")
        lines.append(f"Has Pokedex   : {flags.get('has_pokedex', '?')}")
        lines.append(f"Pokedex owned : {flags.get('pokedex_owned', '?')}")
        lines.append(f"Pokedex seen  : {flags.get('pokedex_seen', '?')}")
        lines.append(f"Badges        : {flags.get('badge_count', 0)}")
    elif state.get("flags_error"):
        lines.append(f"\n[Flags read error: {state['flags_error']}]")

    lines.append(_hr)
    return "\n".join(lines)