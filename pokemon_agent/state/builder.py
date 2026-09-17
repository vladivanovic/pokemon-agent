"""Game-state orchestrator.

:func:`build_game_state` calls every reader method and assembles the
results into a single JSON-serialisable dictionary.

:func:`build_state_summary` renders that dict as a compact, human-readable
text block suitable for injection into an LLM prompt.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Optional

from pokemon_agent.memory.reader import GameMemoryReader

logger = logging.getLogger("pokemon-agent.state")


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