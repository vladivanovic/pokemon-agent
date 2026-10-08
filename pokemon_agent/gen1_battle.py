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

from pokemon_agent.gen1_data import MOVE_TYPE_POWER, MOVE_EFFECT, TYPE_CHART

logger = logging.getLogger("pokemon-agent.gen1_battle")

# Gen 1 splits physical/special by TYPE, not per move. Kept for the score
# tiebreak (our own attacking stat), unused when stats are absent.
SPECIAL_TYPES = frozenset({
    "fire", "water", "grass", "electric", "psychic", "ice", "dragon",
})

# Ranking for the no-damaging-move case, keyed by the game's own effect
# constant (compiled from pokered, so it cannot drift from the ROM).
# (category, tier) - higher tier is the better use of a turn.
EFFECT_RANK: Dict[str, Tuple[str, int]] = {
    "SLEEP_EFFECT":       ("inflict", 90),
    "HEAL_EFFECT":        ("heal",    85),
    "PARALYZE_EFFECT":    ("inflict", 70),
    "CONFUSION_EFFECT":   ("inflict", 60),
    "POISON_EFFECT":      ("inflict", 55),
    "LEECH_SEED_EFFECT":  ("drain",   50),
    "ATTACK_UP2_EFFECT":  ("boost",   40),
    "DEFENSE_UP2_EFFECT": ("boost",   40),
    "SPECIAL_UP2_EFFECT": ("boost",   40),
    "SPEED_UP2_EFFECT":   ("boost",   38),
    "EVASION_UP1_EFFECT": ("boost",   35),
    "ATTACK_DOWN1_EFFECT":    ("drop", 20),
    "DEFENSE_DOWN1_EFFECT":   ("drop", 20),
    "SPEED_DOWN1_EFFECT":     ("drop", 18),
    "SPECIAL_DOWN1_EFFECT":   ("drop", 20),
    "ACCURACY_DOWN1_EFFECT":  ("drop", 22),
}

# Which condition each inflicting effect sets, matching the status byte
# decoded by memory/red.py _decode_status (so we never re-apply one).
EFFECT_STATUS: Dict[str, Optional[str]] = {
    "SLEEP_EFFECT": "SLP",
    "PARALYZE_EFFECT": "PAR",
    "POISON_EFFECT": "PSN",
    "CONFUSION_EFFECT": None,   # volatile; not in the status byte
}


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


def _choose_status(state: Dict[str, Any]) -> Optional[str]:
    """Best usable status move, or None when nothing sensible remains.

    Reached only when no move with PP can deal damage. Rather than deferring
    to a model that is not type-aware, rank the status moves by the game's
    own effect constant and skip ones that cannot do anything: a sleep/para/
    poison move on an already-statused enemy fails, and healing at high HP
    wastes the turn.
    """
    mine = state.get("active_mon") or {}
    enemy = (state.get("battle") or {}).get("enemy") or {}
    enemy_status = str(enemy.get("status") or "OK").upper()
    hp = mine.get("hp") or 0
    max_hp = mine.get("max_hp") or 1
    hp_frac = hp / max_hp

    cands: List[Tuple[int, int, str]] = []
    for i, mv in enumerate(mine.get("moves") or []):
        if not isinstance(mv, dict) or (mv.get("pp", 0) or 0) <= 0:
            continue
        name = str(mv.get("name") or "")
        effect = MOVE_EFFECT.get(name.lower())
        if not effect:
            continue
        category, tier = EFFECT_RANK.get(effect, (None, 0))
        if category is None:
            continue
        if category == "inflict":
            want = EFFECT_STATUS.get(effect)
            if want and enemy_status != "OK":
                continue        # already suffering it - re-applying fails
        if category == "heal" and hp_frac > 0.5:
            continue            # nothing to heal
        cands.append((tier, i, name))

    if not cands:
        return None
    cands.sort(key=lambda t: -t[0])
    tier, slot, name = cands[0]
    logger.info("battle policy: status move_%d %s (tier %d, enemy %s)",
                slot, name, tier, enemy_status)
    return f"move_{slot}"


def choose(state: Dict[str, Any]) -> Optional[str]:
    """Best battle option for this turn, or None to defer to the scorer.

    Returns ``move_<i>`` (a move menu slot the autopilot's dispatch already
    understands) or ``flee``. Damaging moves win on merit; with none usable
    a status move is chosen - except when we are hurt, where a status move
    just burns turns while the enemy attacks, so leaving is better.
    """
    mine = state.get("active_mon") or {}
    battle = state.get("battle") or {}
    hp_frac = (mine.get("hp") or 0) / (mine.get("max_hp") or 1)

    ranked = score_moves(state)
    if ranked and ranked[0][1] > 0:
        slot, score, detail = ranked[0]
        logger.info("battle policy: move_%d %s (power=%s x%s stab x%s)",
                    slot, detail.get("name"), detail.get("power"),
                    detail.get("mult"), detail.get("stab"))
        return f"move_{slot}"

    # No damaging move with PP. If we are also hurt, a status move loses the
    # war of attrition - leave instead. Only in a wild battle: RUN fails
    # against a trainer and would waste the turn.
    btype = str(battle.get("type") or "").lower()
    if hp_frac < 0.35 and btype != "trainer":
        logger.info("battle policy: no damaging PP and hp %.0f%% - fleeing",
                    hp_frac * 100)
        return "flee"

    return _choose_status(state)
