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

logger = logging.getLogger("pokemon-agent.state")


def _nearest_pokecenter_nurse(reader: GameMemoryReader) -> Optional[Tuple[int, int, int]]:
    """
    Returns (dx, dy, dist) where (dx,dy) is one step toward the closest
    nurse tile, or None if no nurse on the current map.
    Uses the same collision/walkability data that the Laya router uses.
    """
    try:
        player = reader.read_player()
        px, py = player.get("position", {}).get("x", 0), player.get("position", {}).get("y", 0)
        collision = reader.read_map_info().get("collision", {})
        walkable = collision.get("walkability", [])   # list of strings, '1'=walkable
        if not walkable:
            logger.debug("[PC] nurse check: no walkable tiles")
            return None

        # Nurse tile id – in Red/Blue the nurse sprite uses tile 0x6B in the
        # second sprite bank (value may differ by ROM; we rely on the fact
        # that the nurse is the only NPC with sprite id 0x1B in most maps).
        sprites = collision.get("sprites", [])
        logger.debug(f"[PC] nurse check: sprites={sprites}")
        nurse_positions = []
        for s in sprites:
            if s.get("who") == 0x1B:          # nurse sprite id
                nurse_positions.append((s.get("cell", {}).get("y", 0),
                                        s.get("cell", {}).get("x", 0)))
        if not nurse_positions:
            logger.debug("[PC] nurse check: no nurse sprites found")
            return None

        # BFS over walkable tiles to find the closest nurse.
        from collections import deque
        q = deque()
        q.append((py, px, 0, 0))   # (y, x, dist_y, dist_x)
        seen = {(py, px)}
        while q:
            y, x, dy, dx = q.popleft()
            if (y, x) in nurse_positions:
                logger.debug(f"[PC] nurse check: found nurse at ({y},{x}) dist={abs(dy)+abs(dx)}")
                return (dy, dx, abs(dy) + abs(dx))   # one-step direction + distance
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= ny < len(walkable) and 0 <= nx < len(walkable[0]):
                    if walkable[ny][nx] == '1' and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        # store the step that got us here
                        q.append((ny, nx,
                                   ny - py if abs(ny-py) > abs(nx-px) else 0,
                                   nx - px if abs(nx-px) > abs(ny-py) else 0))
        logger.debug("[PC] nurse check: BFS completed without finding nurse")
        return None   # should not happen on maps that have a nurse
    except Exception as e:
        logger.debug(f"[PC] nurse check exception: {e}")
        return None


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


def _nearest_pokecenter_door(reader: GameMemoryReader) -> Optional[Tuple[int, int, int]]:
    """Fallback: if the map is a known town and we have a door coordinate,
    return (dx, dy, dist) to that door."""
    map_info = reader.read_map_info()
    map_name = map_info.get("map_name") if map_info else None
    if not map_name or map_name not in PC_CENTERS:
        return None
    # Player position in map coordinates
    player = reader.read_player()
    px = player.get("position", {}).get("x", 0)
    py = player.get("position", {}).get("y", 0)
    door = PC_CENTERS[map_name]
    dx = door["door_x"] - px
    dy = door["door_y"] - py
    dist = abs(dx) + abs(dy)
    # Return one-step direction (sign of dx/dy, zero if already aligned)
    step_dx = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_dy = 0 if dy == 0 else (1 if dy > 0 else -1)
    return (step_dx, step_dy, dist)


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
    #  Nearest Pokémon Center (Manhattan distance in tiles)
    # ----------------------------------------------------------------------
    dxdydist = _nearest_pokecenter(reader)
    if dxdydist is not None:
        dx, dy, dist = dxdydist
        state["nearest_pokecenter_dist"] = dist
        state["nearest_pokecenter_dir"]  = (dx, dy)   # one step toward it
    else:
        state["nearest_pokecenter_dist"] = None
        state["nearest_pokecenter_dir"]  = None

    state["errors"] = errors
    state["status"] = "degraded" if errors else "ok"
    return state


def _nearest_pokecenter(reader: GameMemoryReader) -> Optional[Tuple[int, int, int]]:
    """
    Primary detection: nurse sprite (inside or adjacent to the door).
    Fallback: PC-center door lookup.
    If inside a Pokecenter interior map, return (0,0,0).
    Returns (dx, dy, dist) or None.
    """
    nurse = _nearest_pokecenter_nurse(reader)
    if nurse is not None:
        return nurse
    # If we are inside a Pokecenter map, consider the center reached
    map_info = reader.read_map_info()
    map_name = map_info.get("map_name") if map_info else None
    if map_name and "pokecenter" in map_name.lower():
        return (0, 0, 0)
    return _nearest_pokecenter_door(reader)


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