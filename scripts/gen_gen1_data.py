"""Compile the pokered disassembly tables into a static Gen 1 data module.

Inputs (fetched from pret/pokered):
  /tmp/pokered_moves.asm    - data/moves/moves.asm
  /tmp/pokered_types.asm    - data/types/type_matchups.asm
  /tmp/pokered_sprites.asm  - constants/sprite_constants.asm

Output: pokemon_agent/gen1_data.py  (static, no runtime fetch)

The battle policy needs exactly two things the game state does not carry:
which type and how much power each move has, and the Gen 1 type chart.
Both are canonical in the disassembly, so we compile rather than hand-type.
The sprite table is here for the same reason: a hand-typed version had the
wrong id for the Pokecenter nurse.
"""
import logging
import re

logger = logging.getLogger("gen-gen1-data")

MOVES_SRC = "/tmp/pokered_moves.asm"
TYPES_SRC = "/tmp/pokered_types.asm"
SPRITES_SRC = "/tmp/pokered_sprites.asm"
OUT = "/home/vlad/pokemon-agent/pokemon_agent/gen1_data.py"

# db WATER, FIRE, SUPER_EFFECTIVE
EFFECT_MULT = {
    "SUPER_EFFECTIVE": 2.0,
    "NOT_VERY_EFFECTIVE": 0.5,
    "NO_EFFECT": 0.0,
}


def parse_sprites(path):
    """constants/sprite_constants.asm -> {id: display name}.

    The overworld sprite table is a flat ``const_def`` run, so the ordinal is
    the value; the trailing ``; $XX`` comment is checked against it. This
    replaces a hand-written 3-entry table that had the wrong id for the
    Pokecenter nurse (0x1B is SILPH_WORKER_F, not NURSE - 0x29 is), which
    silently broke every nurse lookup.
    """
    out = {}
    seen_const_def = False
    ordinal = 0
    pat = re.compile(r"^\s*const\s+SPRITE_([A-Z0-9_]+)\s*(?:;\s*\$([0-9A-Fa-f]+))?")
    for line in open(path):
        if "const_def" in line and not line.strip().startswith(";"):
            seen_const_def = True
            ordinal = 0
            continue
        if not seen_const_def:
            continue
        m = pat.match(line)
        if not m:
            if "const_skip" in line or "const_next" in line:
                ordinal += 1
            continue
        name, raw = m.groups()
        if raw is not None:
            val = int(raw, 16)
            if val != ordinal:
                logger.warning("sprite %s: comment $%02X != ordinal $%02X",
                               name, val, ordinal)
            ordinal = val
        out[ordinal] = name.replace("_", " ").title()
        ordinal += 1
    return out


def norm_type(t: str) -> str:
    """ROM type constant -> the display spelling used by /state."""
    t = t.strip().upper()
    if t.endswith("_TYPE"):
        t = t[: -len("_TYPE")]
    return t.lower()


def norm_move(m: str) -> str:
    """VINE_WHIP -> 'vine whip' (matches MOVE_NAMES display, lowercased)."""
    return m.strip().lower().replace("_", " ")


def parse_moves(path):
    """move NAME, EFFECT, POWER, TYPE, ACC, PP  ->  two dicts.

    Returns (type_power, effects) where
      type_power[move] = (type, power)
      effects[move]    = EFFECT_NAME   (e.g. SLEEP_EFFECT, HEAL_EFFECT)
    The effect name is what lets the battle policy rank status moves - it is
    the game's own classification, so no hand-written move list to drift.
    """
    type_power = {}
    effects = {}
    pat = re.compile(
        r"^\s*move\s+([A-Z0-9_]+)\s*,\s*([A-Z0-9_]+)\s*,\s*(\d+)\s*,"
        r"\s*([A-Z_]+)\s*,\s*(\d+)\s*(?:percent)?\s*,\s*(\d+)")
    for line in open(path):
        m = pat.match(line)
        if not m:
            continue
        move, effect, power, mtype, _acc, _pp = m.groups()
        key = norm_move(move)
        type_power[key] = (norm_type(mtype), int(power))
        effects[key] = effect
    return type_power, effects


def parse_types(path):
    """db ATK, DEF, EFFECT  ->  {(atk, def): mult}"""
    out = {}
    pat = re.compile(
        r"^\s*db\s+([A-Z_]+)\s*,\s*([A-Z_]+)\s*,\s*([A-Z_]+)")
    for line in open(path):
        m = pat.match(line)
        if not m:
            continue
        atk, dfn, eff = m.groups()
        if eff not in EFFECT_MULT:
            continue
        key = (norm_type(atk), norm_type(dfn))
        out[key] = EFFECT_MULT[eff]
    return out


def main():
    moves, effects = parse_moves(MOVES_SRC)
    types = parse_types(TYPES_SRC)
    sprites = parse_sprites(SPRITES_SRC)
    print(f"parsed {len(moves)} moves, {len(effects)} effects, "
          f"{len(types)} type matchups, {len(sprites)} sprites")

    # Assert the facts the policy depends on. A silent drift here is what
    # broke nurse detection before, so fail loudly instead.
    assert moves.get("vine whip") == ("grass", 35), moves.get("vine whip")
    assert moves.get("tackle") == ("normal", 35), moves.get("tackle")
    assert types[("ghost", "psychic")] == 0.0, "Gen 1 Ghost vs Psychic must be 0x"
    assert types[("bug", "poison")] == 2.0, "Gen 1 Bug vs Poison must be 2x"
    assert sprites.get(0x29) == "Nurse", sprites.get(0x29)
    assert sprites.get(0x1B) == "Silph Worker F", sprites.get(0x1B)

    lines = [
        '"""Gen 1 move + type data, compiled from the pokered disassembly.',
        "",
        "Generated by scripts/gen_gen1_data.py - do not hand-edit.",
        "  MOVE_TYPE_POWER: {move name (lowercase) -> (type, base power)}",
        "  MOVE_EFFECT:     {move name -> the game's effect constant}",
        "  TYPE_CHART:      {(attacking type, defending type) -> multiplier}",
        "Absent keys mean a neutral 1.0 multiplier / unknown move.",
        '"""',
        "",
        "MOVE_TYPE_POWER: dict = {",
    ]
    for name in sorted(moves):
        mtype, power = moves[name]
        lines.append(f"    {name!r}: ({mtype!r}, {power}),")
    lines.append("}")
    lines.append("")
    lines.append("MOVE_EFFECT: dict = {")
    for name in sorted(effects):
        lines.append(f"    {name!r}: {effects[name]!r},")
    lines.append("}")
    lines.append("")
    lines.append("TYPE_CHART: dict = {")
    for (atk, dfn) in sorted(types):
        lines.append(f"    ({atk!r}, {dfn!r}): {types[(atk, dfn)]},")
    lines.append("}")
    lines.append("")
    lines.append("# Overworld sprite ids, keyed by wSpriteStateData1 picture byte.")
    lines.append("SPRITE_NAMES: dict = {")
    for sid in sorted(sprites):
        lines.append(f"    0x{sid:02X}: {sprites[sid]!r},")
    lines.append("}")
    lines.append("")
    lines.append("# Sprite ids the agent interacts with to heal / shop.")
    lines.append("SPRITE_NURSE = 0x29      # Pokecenter nurse - talk to heal")
    lines.append("SPRITE_CLERK = 0x26      # Mart clerk")
    lines.append("SPRITE_LINK_RECEPTIONIST = 0x2A")
    lines.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
