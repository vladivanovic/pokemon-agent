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
    btype = str(battle.get("type") or "").lower()

    # TRIVIAL-WILD RULE REVERSED (user directive 2026-10-26, "do it your
    # way"): fight trivial wilds instead of fleeing them. The flee was a
    # speed lottery (each failed attempt burns a turn and a successful one
    # still leaves the position unchanged, so the same tile immediately
    # re-rolls a new encounter - proven live 17:01: 26 turns parked at B2F
    # (20,24) in a flee loop). A Lv 36 lead one-shots a Lv 5 Zubat in one
    # deterministic hit; the Pokecenter loop keeps PP topped up, so the
    # PP-conservation argument no longer outweighs the pace. Fleeing stays
    # reserved for genuinely dangerous situations below (weak move + hurt,
    # or no damaging PP at critical HP).
    enemy = battle.get("enemy") or {}
    _my_lvl = int(mine.get("level") or 0)
    _en_lvl = int(enemy.get("level") or 0)

    ranked = score_moves(state)
    # HURTED-CHIP GUARD: a weak (type-disadvantaged) attack at critical HP
    # is a stalemate - the enemy out-damages our chip and heals (Zubat's
    # Leech Life). Running is better than dying: flee when hurt and the
    # best move cannot reasonably win. Proven live 12:27: 0.25x Vine Whip
    # (score 13) at 9/68 HP vs a Zubat, 10+ min stalled, then blackout.
    # Only wild battles: RUN fails vs trainers and burns the turn.
    btype = str(battle.get("type") or "").lower()
    if (ranked and 0 < ranked[0][1] < 20 and hp_frac < 0.30
            and btype != "trainer"):
        logger.info("battle policy: best move score %.1f at hp %.0f%% - "
                    "too weak to win, fleeing wild battle",
                    ranked[0][1], hp_frac * 100)
        return "flee"
    if ranked and ranked[0][1] > 0:
        slot, score, detail = ranked[0]
        logger.info("battle policy: move_%d %s (power=%s x%s stab x%s)",
                    slot, detail.get("name"), detail.get("power"),
                    detail.get("mult"), detail.get("stab"))
        return f"move_{slot}"

    # No damaging move with PP: PUSH THROUGH (user direction 2026-10-25 -
    # "push through instead of looping on PP being out for a single move").
    # Leech Seed drains 1/8 of the enemy's max HP every turn and ends wild
    # battles by itself; once ALL PP are 0 the game auto-fires STRUGGLE
    # (50-power typeless). Fleeing here just produced an endless
    # flee -> Pokecenter -> re-enter -> Zubat grind. Only flee when HP is
    # critical (Struggle's recoil plus the enemy's damage could KO us
    # before the battle ends).
    btype = str(battle.get("type") or "").lower()
    if btype != "trainer" and hp_frac < 0.35:
        logger.info("battle policy: no damaging PP (hp %.0f%%) - fleeing "
                    "this wild battle (critical HP)", hp_frac * 100)
        return "flee"

    # Trainer battle - RUN is impossible, so we must attack. With every
    # damaging move at 0 PP, choosing FIGHT triggers STRUGGLE: 50-power
    # typeless damage each turn. Proven live 13:09: stalling with Growl
    # vs a Lv 11 Bellsprout burned 10+ min while the enemy whittled us
    # down - Struggle ends the battle. The Growl stall only "wins" if the
    # enemy also cannot damage us; when it can, attrition loses.
    mine_moves = mine.get("moves") or []
    damaging_left = any(
        isinstance(mv, dict) and (mv.get("pp", 0) or 0) > 0
        and MOVE_TYPE_POWER.get(str(mv.get("name") or "").lower(), (None, 0))[1] > 0
        for mv in mine_moves)
    if not damaging_left:
        # Struggle only auto-fires when ALL PP are 0 (the game skips the
        # menu); selecting a 0-PP slot just spams "No PP left for this
        # move!" forever (proven live 13:20: move_0 = Poison Powder PP=0
        # while Growl/Leech Seed had PP - frozen 10+ min). Prefer LEECH
        # SEED: it drains the enemy every turn and ends the battle. Growl
        # cuts incoming damage while the stall runs.
        pp_moves = [(i, str((mv or {}).get("name") or "").lower())
                    for i, mv in enumerate(mine_moves)
                    if isinstance(mv, dict) and (mv.get("pp", 0) or 0) > 0]
        if pp_moves:
            seed = next((i for i, n in pp_moves if n == "leech seed"), None)
            slot = seed if seed is not None else pp_moves[0][0]
            logger.info("battle policy: trainer battle, no damaging PP - "
                        "stalling with %s (PP>0) until Struggle is forced",
                        mine_moves[slot].get("name"))
            return f"move_{slot}"
    for i, mv in enumerate(mine_moves):
        if isinstance(mv, dict) and (mv.get("pp", 0) or 0) > 0:
            logger.info("battle policy: trainer battle, no damaging PP - "
                        "stalling with %s (PP=%s)", mv.get("name"),
                        mv.get("pp"))
            return f"move_{i}"
    logger.info("battle policy: trainer battle, ALL PP empty - using "
                "FIGHT to trigger Struggle")
    return "move_0"
