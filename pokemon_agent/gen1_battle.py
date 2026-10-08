"""Deterministic Gen 1 battle policy.

Gen 1 battles are mostly a pure move-selection problem: pick the damaging
move with the best expected damage. That needs three things, all of which we
have exactly:

  * the enemy's types          - read from RAM (battle.enemy.types)
  * our moves + PP             - read from RAM (active_mon.moves)
  * move type / power and the type chart - compiled from pokered
    (pokemon_agent/gen1_data.py)

No model is required, and no model trained on a later generation can beat
this: Gen 9 data would teach abilities, held items, Tera and entry hazards
that do not exist here, plus a corrected type chart (Ghost really is 0x vs
Psychic in this game, and Bug really is 2x vs Poison).

Deliberately out of scope in v1: switching and item use. Both need menu
sequences the autopilot does not implement yet, and neither matters for the
early-game fights this is built for. Return None in those cases and let the
caller fall back to the configured scorer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pokemon_agent.gen1_data import MOVE_TYPE_POWER, TYPE_CHART

logger = logging.getLogger("pokemon-agent.gen1_battle")

# Gen 1 splits physical/special by TYPE, not per move. Kept for the score
# tiebreak (our own attacking stat), unused when stats are absent.
SPECIAL_TYPES = frozenset({
    "fire", "water", "grass", "electric", "psychic", "ice", "dragon",
})


def _type_multiplier(atk_type: str, defenders: List[str]) -> float:
    """Product of the type chart over a (possibly dual-typed) defender."""
    mult = 1.0
    for d in defenders:
        mult *= TYPE_CHART.get((atk_type, d), 1.0)
    return mult


def score_moves(state: Dict[str, Any]) -> List[Tuple[int, float, Dict[str, Any]]]:
    """Rank our usable moves best-first as (slot index, score, detail).

    Slot index is the position in the *displayed* move menu: the reader
    compacts empty slots away, and so does the game's move list, so the
    compacted index is what ``press_down`` steps count.
    """
    mine = state.get("active_mon") or {}
    battle = state.get("battle") or {}
    enemy = battle.get("enemy") or {}
    my_types = [str(t).lower() for t in (mine.get("types") or [])]
    def_types = [str(t).lower() for t in (enemy.get("types") or [])]

    ranked: List[Tuple[int, float, Dict[str, Any]]] = []
    for i, mv in enumerate(mine.get("moves") or []):
        if not isinstance(mv, dict):
            continue
        pp = mv.get("pp", 0) or 0
        if pp <= 0:
            continue
        name = str(mv.get("name") or "").lower()
        mtype, power = MOVE_TYPE_POWER.get(name, (None, 0))
        detail: Dict[str, Any] = {"name": mv.get("name"), "pp": pp}
        if not mtype or power <= 0:
            detail["why"] = "status move"
            ranked.append((i, 0.0, detail))
            continue
        mult = _type_multiplier(mtype, def_types) if def_types else 1.0
        stab = 1.5 if mtype in my_types else 1.0
        score = power * mult * stab
        detail.update({"type": mtype, "power": power,
                       "mult": mult, "stab": stab})
        ranked.append((i, score, detail))

    ranked.sort(key=lambda t: -t[1])
    return ranked


def choose(state: Dict[str, Any]) -> Optional[str]:
    """Best battle option for this turn, or None to defer to the scorer.

    Returns ``move_<i>`` naming the move menu slot, which the autopilot's
    dispatch already understands, or None when no move with PP can deal
    damage (nothing to gain over letting the configured scorer decide).
    """
    ranked = score_moves(state)
    if not ranked:
        return None
    slot, score, detail = ranked[0]
    if score <= 0:
        # Every usable move is a status move or fully resisted. Falling back
        # keeps this deterministic rule out of the one case where a model
        # might actually help (and drains PP toward Struggle naturally).
        logger.info("battle policy: no damaging move (%s) - deferring",
                    detail.get("name"))
        return None
    logger.info("battle policy: move_%d %s (power=%s x%s stab x%s)",
                slot, detail.get("name"), detail.get("power"),
                detail.get("mult"), detail.get("stab"))
    return f"move_{slot}"
