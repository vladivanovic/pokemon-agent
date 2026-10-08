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
MAPS_SRC = "/tmp/pokered_maps.asm"
OUT = "/home/vlad/pokemon-agent/pokemon_agent/gen1_data.py"

# ROM constant -> the display spelling the rest of the codebase matches on.
# Everything else is Title Cased from the constant. Keys here are the ones
# referenced by name elsewhere (goal text, pc_centers.json, the autopilot),
# so changing one silently breaks a lookup.
MAP_ALIASES = {
    "OAKS_LAB": "Oak's Lab",
    "BLUES_HOUSE": "Blue's House",
    "MUSEUM_1F": "Pewter Museum 1F",
    "MUSEUM_2F": "Pewter Museum 2F",
    # Viridian Forest South Gate (id 50) is named 'Viridian Forest South Gate'
    # by MAP_NAMES but 'Route 2 Gate Viridian' in map_data.json. They are the
    # same physical map. The geographic graph uses the map_data.json name, so
    # we alias the MAP_NAMES spelling to the map_data.json spelling so the
    # portal and region key spaces unify.
    "VIRIDIAN_FOREST_SOUTH_GATE": "Route 2 Gate Viridian",
}

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


def parse_maps(path):
    """constants/map_constants.asm -> {id: display name}.

    The map table is one flat ``const_def`` run, so the ordinal IS the ROM map
    id. red.py's hand-maintained MAP_NAMES was missing the Viridian Forest
    South Gate entry, which shifted every id from 50 up by one - so map 51
    (the real Viridian Forest) was reported as "Pewter Museum 1F" and the
    agent believed it had reached Pewter City, while map 50 (a 5x4 gate with
    no encounters) was reported as "Viridian Forest". Compiling removes that
    whole class of error.
    """
    out = {}
    started = False
    ordinal = 0
    for line in open(path):
        if "const_def" in line and not line.strip().startswith(";"):
            started = True
            ordinal = 0
            continue
        if not started:
            continue
        m = re.match(r"\s*map_const\s+([A-Z0-9_]+)\s*,", line)
        if not m:
            continue
        const = m.group(1)
        # Display spellings the rest of the codebase already matches on.
        out[ordinal] = MAP_ALIASES.get(const, const.replace("_", " ").title())
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
    maps = parse_maps(MAPS_SRC)
    print(f"parsed {len(moves)} moves, {len(effects)} effects, "
          f"{len(types)} type matchups, {len(sprites)} sprites, "
          f"{len(maps)} maps")

    # Assert the facts the policy depends on. A silent drift here is what
    # broke nurse detection before, so fail loudly instead.
    assert moves.get("vine whip") == ("grass", 35), moves.get("vine whip")
    assert moves.get("tackle") == ("normal", 35), moves.get("tackle")
    assert types[("ghost", "psychic")] == 0.0, "Gen 1 Ghost vs Psychic must be 0x"
    assert types[("bug", "poison")] == 2.0, "Gen 1 Bug vs Poison must be 2x"
    assert sprites.get(0x29) == "Nurse", sprites.get(0x29)
    assert sprites.get(0x1B) == "Silph Worker F", sprites.get(0x1B)
    # The ids that were mislabelled by the old off-by-one table.
    assert maps.get(0) == "Pallet Town", maps.get(0)
    assert maps.get(1) == "Viridian City", maps.get(1)
    assert maps.get(13) == "Route 2", maps.get(13)
    assert maps.get(50) == "Route 2 Gate Viridian", maps.get(50)
    assert maps.get(51) == "Viridian Forest", maps.get(51)
    assert maps.get(52) == "Pewter Museum 1F", maps.get(52)
    assert maps.get(54) == "Pewter Gym", maps.get(54)
    assert maps.get(58) == "Pewter Pokecenter", maps.get(58)

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
    lines.append("# ROM map id -> display name. The ordinal position in")
    lines.append("# constants/map_constants.asm IS the map id used by wCurMap.")
    lines.append("# A hand-maintained version was off by one from id 50 up, which")
    lines.append("# reported the real Viridian Forest (51) as Pewter Museum 1F.")
    lines.append("MAP_NAMES: dict = {")
    for mid in sorted(maps):
        lines.append(f"    0x{mid:02X}: {maps[mid]!r},")
    lines.append("}")
    lines.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
