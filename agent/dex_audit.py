"""Pokédex reachability audit for Pokémon Red.

Ported from probe168 (post-game leg 1, 2026-11-13/17), where it
compiled the ledger the dex-completion sweep ran against.

Parses a pret/pokered checkout (all paths relative to it):
- constants/pokemon_constants.asm  -> internal species ids
- constants/pokedex_constants.asm  -> national dex numbers
- data/pokemon/names.asm           -> display names
- data/pokemon/evos_moves.asm      -> evolution edges
- data/wild/maps/*.asm             -> grass + water encounters (_RED)
- data/wild/super_rod.asm          -> fishing groups per map
- data/wild/good_rod.asm           -> good rod table
- data/events/prizes.asm           -> Game Corner prizes (_RED)
- data/events/prize_mon_levels.asm -> prize levels (_RED)
- data/events/trades.asm           -> in-game trades (used ones)

Then computes Red-cartridge reachability by closure over sources
+ level/item evolutions (trade evolutions deliberately do NOT
propagate) and classifies every species: obtainable (with its
source list), unobtainable (with a reason), or choice-locked
(the three Eeveelutions: exactly one is obtainable from the
single gift Eevee). The ceiling counts obtainable + 1.

CLI: python3 -m agent.dex_audit <pokered_dir> <out.json>
"""
import json
import os
import re
import sys

SLOT_CHANCES = [19.9, 19.9, 15.2, 9.8, 9.8, 9.8, 5.1, 5.1, 4.3, 1.2]


def parse_consts(path):
    """Return {NAME: value} counting const / const_skip / const_next."""
    vals = {}
    cur = 0
    for line in open(path):
        line = line.split(";")[0].strip()
        m = re.match(r"const_def\s+(\d+)$", line)
        if m:
            cur = int(m.group(1))
            continue
        m = re.match(r"const\s+([A-Z0-9_]+)$", line)
        if m:
            vals[m.group(1)] = cur
            cur += 1
            continue
        m = re.match(r"const_skip(?:\s+(\d+))?$", line)
        if m:
            cur += int(m.group(1) or 1)
            continue
        m = re.match(r"const_next\s+(.+)$", line)
        if m:
            expr = m.group(1).replace("$", "0x")
            cur = eval(expr, {}, {})
            continue
    return vals



def build_ledger(pokered_dir):
    """Compile the ledger from a pokered checkout.

    Returns {"species": ..., "trades": ..., "prizes": ...,
    "ceiling": int, "anomalies": [...]}.
    """
    PK = str(pokered_dir)
    # ---------- species ----------
    sp_id = {}          # CONST_NAME -> internal id
    for name, v in parse_consts(f"{PK}/constants/pokemon_constants.asm").items():
        if v > 0:
            sp_id[name] = v
    dex_of = {}         # CONST_NAME -> dex number
    for name, v in parse_consts(f"{PK}/constants/pokedex_constants.asm").items():
        if name.startswith("DEX_") and v > 0:
            dex_of[name[4:]] = v
    by_dex = {v: k for k, v in dex_of.items()}
    names = []          # internal-id order from names.asm (index 0 == id 1)
    for line in open(f"{PK}/data/pokemon/names.asm"):
        m = re.search(r'db\s+"([^"]+)"', line)
        if m:
            names.append(m.group(1).replace("@", "").strip())

    SPECIES = {}        # const name -> dict
    for const_name, iid in sp_id.items():
        if const_name not in dex_of:
            continue    # MissingNo. slots etc.
        SPECIES[const_name] = {
            "const": const_name, "id": iid, "dex": dex_of[const_name],
            "name": names[iid - 1] if iid - 1 < len(names) else const_name,
        }
    assert len(SPECIES) == 151, len(SPECIES)

    # ---------- evolutions ----------
    evos = {}           # const -> list of (method, param, target_const)
    txt = open(f"{PK}/data/pokemon/evos_moves.asm").read()
    blocks = re.split(r"\n(?=[A-Za-z0-9_]+EvosMoves:)", txt)
    ptr = re.findall(r"dw\s+([A-Za-z0-9_]+)EvosMoves", txt)
    id_by_ptr_index = {}
    inv_id = {v: k for k, v in sp_id.items()}
    for idx, label in enumerate(ptr):
        # pointer table order == internal id order (id = idx+1)
        id_by_ptr_index[label] = inv_id.get(idx + 1)
    for b in blocks:
        m = re.match(r"([A-Za-z0-9_]+)EvosMoves:", b)
        if not m:
            continue
        base = id_by_ptr_index.get(m.group(1))
        if base is None or base not in SPECIES:
            continue
        edges = []
        for line in b.splitlines()[1:]:
            line = line.split(";")[0].strip()
            if line == "db 0":
                break
            mm = re.match(r"db\s+EVOLVE_(LEVEL|ITEM|TRADE),\s*([^,]+),\s*(?:(\d+),\s*)?([A-Z0-9_]+)$", line)
            if mm:
                kind, p1, p2, tgt = mm.groups()
                if kind == "LEVEL":
                    edges.append(("level", int(p1), tgt))
                elif kind == "ITEM":
                    edges.append(("item", p1.strip(), tgt))
                else:
                    edges.append(("trade", None, tgt))
        if edges:
            evos[base] = edges

    # ---------- version-filtered line stream ----------
    def red_lines(path):
        """Yield code lines (comments stripped) with _BLUE blocks removed."""
        stack = []      # True == currently emitting
        emit = True
        for raw in open(path):
            line = raw.split(";")[0].rstrip()
            s = line.strip()
            if s.startswith("IF DEF(_RED)"):
                stack.append(emit)
                emit = emit and True
                continue
            if s.startswith("IF DEF(_BLUE)"):
                stack.append(emit)
                emit = False
                continue
            if s.startswith("IF "):
                stack.append(emit)
                continue
            if s == "ELSE":
                if stack:
                    emit = stack[-1] and not emit if False else emit
                    # ELSE of IF DEF(_BLUE) means Red branch: flip
                    emit = stack[-1]
                continue
            if s == "ENDC":
                if stack:
                    emit = stack.pop()
                continue
            if emit:
                yield line


    def parse_wild_file(path):
        """-> (grass slots [(lvl, sp)], water slots [(lvl, sp)])"""
        grass, water = [], []
        cur = None
        for line in red_lines(path):
            s = line.strip()
            if s.startswith("def_grass_wildmons"):
                cur = grass
                continue
            if s.startswith("def_water_wildmons"):
                cur = water
                continue
            if s.startswith("end_grass_wildmons") or s.startswith("end_water_wildmons"):
                cur = None
                continue
            m = re.match(r"db\s+(\d+),\s*([A-Z0-9_]+)$", s)
            if m and cur is not None:
                cur.append((int(m.group(1)), m.group(2)))
        return grass, water


    wild = {}           # map file stem -> {"grass": [...], "water": [...]}
    wdir = f"{PK}/data/wild/maps"
    for fn in sorted(os.listdir(wdir)):
        if fn.endswith(".asm"):
            g, w = parse_wild_file(os.path.join(wdir, fn))
            if g or w:
                wild[fn[:-4]] = {"grass": g, "water": w}

    # ---------- rods ----------
    super_rod = {}      # map const -> group name
    groups = {}
    cur_group = None
    for line in open(f"{PK}/data/wild/super_rod.asm"):
        s = line.split(";")[0].strip()
        m = re.match(r"dbw\s+([A-Z0-9_]+),\s*\.Group(\d+)", s)
        if m:
            super_rod[m.group(1)] = int(m.group(2))
            continue
        m = re.match(r"\.Group(\d+):", s)
        if m:
            cur_group = int(m.group(1))
            groups[cur_group] = []
            continue
        m = re.match(r"db\s+(\d+),\s*([A-Z0-9_]+)$", s)
        if m and cur_group is not None and m.group(2) in SPECIES:
            groups[cur_group].append((int(m.group(1)), m.group(2)))
    good_rod = [(10, "GOLDEEN"), (10, "POLIWAG")]
    old_rod = [(5, "MAGIKARP")]     # engine/items/item_effects.asm lb bc, 5, MAGIKARP

    # ---------- prizes (Red) ----------
    prize_mons, prize_costs, prize_levels = [], [], {}
    cost_lines = []
    in_cost = False
    for line in red_lines(f"{PK}/data/events/prizes.asm"):
        s = line.strip()
        if s.startswith("PrizeMenuMon1Entries"):
            sec = "m1"
            continue
        if s.startswith("PrizeMenuMon1Cost"):
            sec = "c1"
            continue
        if s.startswith("PrizeMenuMon2Entries"):
            sec = "m2"
            continue
        if s.startswith("PrizeMenuMon2Cost"):
            sec = "c2"
            continue
        if s.startswith("PrizeMenuTMs"):
            sec = None
        m = re.match(r"db\s+([A-Z0-9_]+)$", s)
        if m and sec in ("m1", "m2") and m.group(1) in SPECIES:
            prize_mons.append(m.group(1))
        m = re.match(r"bcd2\s+(\d+)$", s)
        if m and sec in ("c1", "c2"):
            prize_costs.append(int(m.group(1)))
    for line in red_lines(f"{PK}/data/events/prize_mon_levels.asm"):
        m = re.match(r"\s*db\s+([A-Z0-9_]+),\s*(\d+)", line)
        if m and m.group(1) in SPECIES:
            prize_levels[m.group(1)] = int(m.group(2))
    prizes = {sp: {"coins": c, "level": prize_levels.get(sp)}
              for sp, c in zip(prize_mons, prize_costs)}

    # ---------- trades (used only) ----------
    trades = []
    for line in open(f"{PK}/data/events/trades.asm"):
        m = re.search(r"npctrade\s+([A-Z0-9_]+),\s*([A-Z0-9_]+),.*;\s*used in\s+(\S+)", line)
        if m:
            trades.append({"give": m.group(1), "get": m.group(2), "where": m.group(3)})

    # ---------- gifts / statics seeds (source-verified) ----------
    GIFTS = {
        "SQUIRTLE": "starter (chosen)",
        "EEVEE": "gift ball, CeladonMansionRoofHouse (4,3), Lv25",
        "MAGIKARP": "Route 4 PC salesman, 500 yen, EVENT_BOUGHT_MAGIKARP",
        "OMANYTE": "revive HELIX FOSSIL at Cinnabar Lab (fossil in PC storage)",
        "AERODACTYL": "OLD AMBER from Pewter Museum back room -> Cinnabar Lab",
        "LAPRAS": "Silph Co gift (owned)",
        "HITMONLEE": "Fighting Dojo pick (taken)",
        "SNORLAX": "Route 16 static (caught)",
        "ZAPDOS": "Power Plant static (deferred leg)",
        "ARTICUNO": "Seafoam B4F static -- VLAD FENCED: leave alone",
        "MOLTRES": "Victory Road 2F static -- VLAD FENCED: leave alone",
        "MEWTWO": "Cerulean Cave B1F static -- fenced this leg",
    }
    BLUE_EXCLUSIVES = {"SANDSHREW", "SANDSLASH", "VULPIX", "NINETALES",
                       "MEOWTH", "PERSIAN", "BELLSPROUT", "WEEPINBELL",
                       "VICTREEBEL", "PINSIR", "MAGMAR"}
    UNCHOSEN_STARTERS = {"BULBASAUR", "IVYSAUR", "VENUSAUR",
                         "CHARMANDER", "CHARMELEON", "CHARIZARD"}
    UNCHOSEN_FOSSIL = {"KABUTO", "KABUTOPS"}
    TRADE_EVOS = {"ALAKAZAM", "MACHAMP", "GOLEM", "GENGAR"}

    # ---------- reachability closure ----------
    sources = {}        # const -> list of source strings

    def add(sp, src):
        if sp in SPECIES:
            sources.setdefault(sp, []).append(src)

    for mapname, tab in wild.items():
        for i, (lvl, sp) in enumerate(tab["grass"]):
            add(sp, f"wild grass {mapname} (slot {i}, {SLOT_CHANCES[i]}%, Lv{lvl})")
        for i, (lvl, sp) in enumerate(tab["water"]):
            add(sp, f"wild surf {mapname} (slot {i}, {SLOT_CHANCES[i]}%, Lv{lvl})")
    for mapconst, gnum in super_rod.items():
        for lvl, sp in groups[gnum]:
            add(sp, f"Super Rod group {gnum} at {mapconst} (Lv{lvl})")
    for lvl, sp in good_rod:
        add(sp, f"Good Rod (Lv{lvl})")
    for lvl, sp in old_rod:
        add(sp, f"Old Rod (Lv{lvl})")
    for sp, p in prizes.items():
        add(sp, f"Game Corner prize ({p['coins']} coins, Lv{p['level']})")
    for t in trades:
        add(t["get"], f"in-game trade at {t['where']} (give {t['give']})")
    for sp, note in GIFTS.items():
        add(sp, note)

    obtainable = set(sources)
    changed = True
    while changed:
        changed = False
        for base, edges in evos.items():
            if base not in obtainable:
                continue
            for kind, param, tgt in edges:
                if kind in ("level", "item") and tgt not in obtainable:
                    how = f"L{param}" if kind == "level" else f"{param}"
                    sources.setdefault(tgt, []).append(f"evolve {base} ({how})")
                    obtainable.add(tgt)
                    changed = True

    # ---------- classification ----------
    def unobtainable_reason(sp):
        if sp in BLUE_EXCLUSIVES:
            return "Blue version exclusive"
        if sp in UNCHOSEN_STARTERS:
            return "unchosen starter line (Squirtle chosen)"
        if sp in UNCHOSEN_FOSSIL:
            return "unchosen Dome fossil line (Helix chosen)"
        if sp in TRADE_EVOS:
            return "trade evolution (no trading partner)"
        if sp == "HITMONCHAN":
            return "Dojo either/or spent on Hitmonlee"
        if sp in ("VAPOREON", "JOLTEON", "FLAREON"):
            return "Eevee evolves into only one of three (two unobtainable)"
        if sp == "MEW":
            return "event-only; not in a solo cartridge"
        return None

    ledger = {}
    anomalies = []
    for const_name, info in SPECIES.items():
        entry = dict(info)
        entry["sources"] = sources.get(const_name, [])
        entry["evos"] = [{"kind": k, "param": str(p), "to": t}
                         for k, p, t in evos.get(const_name, [])]
        if const_name in obtainable:
            entry["status"] = "obtainable"
        else:
            entry["status"] = "unobtainable"
            entry["reason"] = unobtainable_reason(const_name)
            if entry["reason"] is None:
                anomalies.append(const_name)
            # Eevee exception: exactly ONE eeveelution is obtainable; the
            # closure marked all three unobtainable because Eevee itself is
            # obtainable but stone-evos from a one-shot gift still count.
        ledger[const_name] = entry

    # Eevee's stone evolutions ARE reachable (one of them): mark all three
    # obtainable-with-caveat instead of unobtainable; the two not chosen
    # become unobtainable only after the choice is spent. For the ceiling:
    # count Eevee family as +1 species slot beyond Eevee itself.
    for ee in ("VAPOREON", "JOLTEON", "FLAREON"):
        ledger[ee]["status"] = "choice-locked"
        ledger[ee]["reason"] = ("one of the three Eeveelutions is obtainable; "
                                "the other two are not (single gift Eevee)")

    ceiling = sum(1 for e in ledger.values()
                  if e["status"] in ("obtainable",)) + 1  # +1 eeveelution slot
    unobt = [e for e in ledger.values() if e["status"] == "unobtainable"]
    choice = [e for e in ledger.values() if e["status"] == "choice-locked"]

    return {"species": ledger, "trades": trades,
            "prizes": prizes, "ceiling": ceiling,
            "anomalies": anomalies}


if __name__ == "__main__":
    result = build_ledger(sys.argv[1])
    with open(sys.argv[2], "w") as fh:
        json.dump(result, fh, indent=1)
    unobt = [e for e in result["species"].values()
             if e["status"] == "unobtainable"]
    choice = [e for e in result["species"].values()
              if e["status"] == "choice-locked"]
    print(f"species={len(result['species'])} "
          f"ceiling={result['ceiling']} "
          f"unobtainable={len(unobt)} "
          f"choice_locked={len(choice)}")
    print("anomalies:", result["anomalies"])
