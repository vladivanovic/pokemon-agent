#!/usr/bin/env python3
"""Gen 1 battle oracle, parsed from pret/pokered (the ground truth for OUR ROM).

Why parse the disassembly instead of using a public dataset:
  * Gen 1 has quirks that later gens fixed, and training on Gen 7 data would
    teach the model WRONG matchups. Examples straight out of
    data/types/type_matchups.asm:
        POISON -> BUG     SUPER_EFFECTIVE  (neutral in Gen 2+)
        BUG    -> POISON  SUPER_EFFECTIVE  (not-very-effective in Gen 2+)
        PSYCHIC-> GHOST   NO_EFFECT        (the famous 0-damage "super effective" bug)
  * It is byte-exact to the cartridge in the emulator.

Type encoding (constants/type_constants.asm), which is ALSO the encoding of
the type bytes in RAM -- engine/battle/core.asm compares wMoveType directly
against the table's "attacking type" byte:
    NORMAL=0 FIGHTING=1 FLYING=2 POISON=3 GROUND=4 ROCK=5 BIRD=6 BUG=7 GHOST=8
    FIRE=20 WATER=21 GRASS=22 ELECTRIC=23 PSYCHIC_TYPE=24 ICE=25 DRAGON=26
Multiplier is applied as damage * mult / 10, so SUPER_EFFECTIVE=20 (x2),
NOT_VERY_EFFECTIVE=5 (x0.5), NO_EFFECT=0 (x0).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

POKERED = os.environ.get("POKERED_DIR", "/tmp/pokered")

# constants/type_constants.asm
TYPE_IDS: Dict[str, int] = {
    "NORMAL": 0, "FIGHTING": 1, "FLYING": 2, "POISON": 3, "GROUND": 4,
    "ROCK": 5, "BIRD": 6, "BUG": 7, "GHOST": 8,
    "FIRE": 20, "WATER": 21, "GRASS": 22, "ELECTRIC": 23,
    "PSYCHIC_TYPE": 24, "ICE": 25, "DRAGON": 26,
}
ID_TYPES: Dict[int, str] = {v: k for k, v in TYPE_IDS.items()}

# Physical/special split in Gen 1 is BY TYPE, not per-move.
SPECIAL_TYPES = {TYPE_IDS[t] for t in
                 ("FIRE", "WATER", "GRASS", "ELECTRIC", "PSYCHIC_TYPE",
                  "ICE", "DRAGON")}

MULT = {"SUPER_EFFECTIVE": 20, "NOT_VERY_EFFECTIVE": 5, "NO_EFFECT": 0}


def _strip_comment(line: str) -> str:
    return line.split(";", 1)[0].strip()


@lru_cache(maxsize=1)
def load_type_chart() -> Dict[Tuple[int, int], int]:
    """(attacker_id, defender_id) -> multiplier in tenths (20/5/0). Absent = 10."""
    path = os.path.join(POKERED, "data/types/type_matchups.asm")
    chart: Dict[Tuple[int, int], int] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = _strip_comment(raw)
            if not line.startswith("db "):
                continue
            body = line[3:].strip()
            if body.startswith("-1"):
                break
            parts = [p.strip() for p in body.split(",")]
            if len(parts) != 3:
                continue
            atk, dfn, m = parts
            if atk not in TYPE_IDS or dfn not in TYPE_IDS or m not in MULT:
                continue
            chart[(TYPE_IDS[atk], TYPE_IDS[dfn])] = MULT[m]
    return chart


def effectiveness(move_type: int, def_types: Sequence[Optional[int]]) -> int:
    """Gen 1 type multiplier in tenths for a move type vs 1-2 defender types.

    Dual types MULTIPLY (so 20*20 = x4). Immune (0) wins outright.
    """
    chart = load_type_chart()
    out = 10
    for dt in def_types:
        if dt is None:
            continue
        out = out * chart.get((move_type, dt), 10) // 10
    return out


@lru_cache(maxsize=1)
def load_moves() -> Dict[str, Dict]:
    """move name -> {power, type_id, type, accuracy, pp, effect}."""
    path = os.path.join(POKERED, "data/moves/moves.asm")
    moves: Dict[str, Dict] = {}
    pat = re.compile(r"^\s*move\s+([A-Z0-9_]+)\s*,\s*([A-Z0-9_]+)\s*,\s*"
                     r"(\d+)\s*,\s*([A-Z_]+)\s*,\s*(\d+)\s*,\s*(\d+)")
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = _strip_comment(raw)
            m = pat.match(line)
            if not m:
                continue
            name, effect, power, mtype, acc, pp = m.groups()
            if mtype not in TYPE_IDS:
                continue
            moves[name.replace("_", " ")] = {
                "name": name,
                "effect": effect,
                "power": int(power),
                "type_id": TYPE_IDS[mtype],
                "type": mtype,
                "accuracy": int(acc),
                "pp": int(pp),
            }
    return moves


@lru_cache(maxsize=1)
def load_base_stats() -> Dict[str, Dict]:
    """pokemon name -> {hp, atk, def, spd, spc, types:[id], type_names:[str]}."""
    d = os.path.join(POKERED, "data/pokemon/base_stats")
    out: Dict[str, Dict] = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".asm"):
            continue
        name = fn[:-4]
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            txt = f.read()
        sm = re.search(r"db\s+(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)", txt)
        tm = re.search(r"db\s+([A-Z_]+),\s*([A-Z_]+)\s*;\s*type", txt)
        if not sm or not tm:
            continue
        hp, atk, dfn, spd, spc = (int(x) for x in sm.groups())
        t1, t2 = tm.group(1), tm.group(2)
        tids = [TYPE_IDS.get(t1)]
        if t2 != t1 and t2 in TYPE_IDS:
            tids.append(TYPE_IDS[t2])
        out[name] = {
            "hp": hp, "atk": atk, "def": dfn, "spd": spd, "spc": spc,
            "types": [t for t in tids if t is not None],
            "type_names": [ID_TYPES.get(t, str(t)) for t in tids if t is not None],
        }
    return out


# Display spelling (as the ROM prints it, e.g. wram TYPE_NAMES) -> type id.
# Needed because the flight recorder stores live-read type NAMES ("Grass"),
# while the pokered constants are UPPER_SNAKE ("GRASS"). Psychic is the trap:
# the constant is PSYCHIC_TYPE but the display name is just "Psychic".
DISPLAY_TO_TYPE: Dict[str, int] = {
    "NORMAL": 0, "FIGHTING": 1, "FLYING": 2, "POISON": 3, "GROUND": 4,
    "ROCK": 5, "BIRD": 6, "BUG": 7, "GHOST": 8,
    "FIRE": 20, "WATER": 21, "GRASS": 22, "ELECTRIC": 23,
    "PSYCHIC": 24, "PSYCHIC_TYPE": 24, "ICE": 25, "DRAGON": 26,
}


def type_id_from_name(name: Any) -> Optional[int]:
    """Resolve a type given as an id, a pokered constant, or a display name."""
    if name is None:
        return None
    if isinstance(name, int):
        return name if name in ID_TYPES else None
    key = str(name).strip().upper().replace(" ", "_")
    if key in TYPE_IDS:
        return TYPE_IDS[key]
    return DISPLAY_TO_TYPE.get(key)


def _norm_move(name: str) -> str:
    """Normalize a move name for lookup.

    pokered keys are UPPER_SNAKE ('VINE_WHIP'); the emulator/recorder emits
    display text ('Vine Whip'). Fold to a single canonical form.
    """
    return str(name).upper().replace("_", " ").strip()


def move_score(move_name: str, user_types: Sequence[int],
               def_types: Sequence[int], atk_stat: int, spc_stat: int) -> float:
    """Expected-power heuristic score for one move in Gen 1.

    Gen 1 damage uses Attack/Defense for physical TYPES and Special for
    special TYPES (no per-move split), so the relevant stat depends on the
    move's type, not the move.
    """
    mv = load_moves().get(_norm_move(move_name))
    if not mv or mv["power"] <= 0:
        return 0.0
    eff = effectiveness(mv["type_id"], def_types) / 10.0
    if eff == 0:
        return -1.0                      # immune: never pick
    stab = 1.5 if mv["type_id"] in set(user_types) else 1.0
    acc = mv["accuracy"] / 100.0
    stat = spc_stat if mv["type_id"] in SPECIAL_TYPES else atk_stat
    return mv["power"] * eff * stab * acc * (stat / 100.0)


def best_move(moves_with_pp: Sequence[Tuple[str, int]],
              user_types: Sequence[int], def_types: Sequence[int],
              atk_stat: int = 100, spc_stat: int = 100) -> Optional[str]:
    """Oracle choice: highest expected-power move with PP remaining."""
    best, best_s = None, None
    for name, pp in moves_with_pp:
        if pp is None or int(pp) <= 0:
            continue
        s = move_score(name, user_types, def_types, atk_stat, spc_stat)
        if s < 0:
            continue
        if best_s is None or s > best_s:
            best, best_s = name, s
    return best


if __name__ == "__main__":
    chart = load_type_chart()
    moves = load_moves()
    stats = load_base_stats()
    print(f"type chart rows : {len(chart)}")
    print(f"moves parsed    : {len(moves)}")
    print(f"base stats      : {len(stats)}")
    # Sanity-check the three Gen 1 quirks.
    checks = [
        ("POISON", "BUG", 20, "Gen1-only super effective"),
        ("BUG", "POISON", 20, "Gen1-only super effective"),
        # The famous Gen 1 bug is GHOST -> PSYCHIC = NO_EFFECT (line 78 of the
        # table). PSYCHIC -> GHOST is not listed at all, i.e. genuinely
        # neutral in Gen 1.
        ("GHOST", "PSYCHIC_TYPE", 0, "famous Ghost-vs-Psychic 0-damage bug"),
        ("PSYCHIC_TYPE", "GHOST", 10, "Psychic->Ghost is neutral (not listed)"),
        ("ELECTRIC", "GROUND", 0, "standard immunity"),
        ("WATER", "GRASS", 5, "standard resist"),
        ("NORMAL", "GHOST", 0, "Normal cannot touch Ghost"),
    ]
    ok = True
    for a, d, want, why in checks:
        got = chart.get((TYPE_IDS[a], TYPE_IDS[d]), 10)
        mark = "OK " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{mark}] {a:13s}->{d:8s} = {got:3d} (want {want:3d})  {why}")
    print("\nbulbasaur:", stats.get("bulbasaur"))
    print("VINE WHIP:", moves.get("VINE WHIP"))
    print("\nSELF-CHECK:", "PASS" if ok else "FAIL")
