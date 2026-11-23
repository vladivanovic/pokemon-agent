#!/usr/bin/env python3
"""quests.py - the side-quest library for the Pokemon Red agent.

Vlad's 2026-11-18 design, built after the dex programme
completed (docs S40): "revise that code as actionable side
quests that can be called based upon a command". Every quest
is a named, self-contained action over the ported primitives
in run_real.py with the programme's verify-first discipline
baked in:

* declared arguments (name / kind / default), validated
  BEFORE the emulator is touched - a bad call is a structured
  refusal, never a traceback;
* RAM-verified preconditions - an unmet precondition refuses
  cleanly with the reason (quest, precondition, detail) and
  nothing moves;
* execution by composition of the proven drivers (Fly,
  league, Mansion grind, careful catch, Safari machinery,
  Game Corner, NPC trades, stone evolution, fossil revival,
  mart buy/sell, post-HoF PC helpers);
* RAM-verified postconditions - a quest reports ok only when
  the effect it promised is readable in RAM afterwards.

Two invocation surfaces:

1. CLI:
       python3 quests.py --list
       python3 quests.py --state work.state --out done.state \
           travel "Fuchsia City"
       python3 quests.py --state work.state grind lapras 60
   (--rom / POKERED_ROM choose the ROM, as in
   play_full_game.py; --out defaults to --state, the
   working-state convention of the probe stages.)

2. The live redirect channel: with play_full_game.py
   --directives, append
       {"quest": "heal", "args": {}}
   to the directives file. Quest directives are validated and
   audit-logged exactly like goal directives (directives.py);
   the runner's Ctx.execute_quest is the host endpoint.

Data (species / learnsets / habitats / trades / prizes /
marts / prices) is parsed from the pret/pokered checkout -
the same source agent/dex_audit.py compiles the ledger from
- located via POKERED_DIR or ~/workspace/projects/pokered.
Quests that need data refuse cleanly when it is absent; the
pure-machinery quests (heal, travel, box_switch) do not.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_real as rr
from agent.world import default_world

# ------------------------------------------------------------ data layer

_POKERED_CANDIDATES = (
    os.environ.get("POKERED_DIR"),
    str(Path.home() / "workspace" / "projects" / "pokered"),
)


def pokered_dir():
    for cand in _POKERED_CANDIDATES:
        if cand and (Path(cand) / "constants").is_dir():
            return Path(cand)
    return None


class Data:
    """Everything the quests need from the pokered checkout,
    parsed once per process: the dex_audit ledger (species,
    trades, prizes, ceiling), learnsets, item ids, wild
    encounter tables, mart stocks and item prices."""

    def __init__(self, pk: Path):
        from agent.dex_audit import build_ledger, parse_consts
        self.pk = pk
        self.ledger = build_ledger(str(pk))
        self.species = self.ledger["species"]     # const -> info
        self.by_id = {v["id"]: k for k, v in self.species.items()}
        self.items = parse_consts(
            str(pk / "constants" / "item_constants.asm"))
        self.moves = parse_consts(
            str(pk / "constants" / "move_constants.asm"))
        self.learnsets = self._parse_learnsets()
        self.wild = self._parse_wild()            # stem -> grass/water
        self.marts = self._parse_marts()          # map name -> [items]
        self.prices = self._parse_prices()        # item const -> yen

    # -- species / items --------------------------------------
    def species_const(self, text):
        """Resolve a user spelling to a species const name:
        const ('MR_MIME'), display name ('Mr. Mime'), dex
        number ('122' / '#122'), case/space flexible."""
        if text is None:
            return None
        t = str(text).strip()
        if t.startswith("#"):
            t = t[1:]
        if t.isdigit():
            dex = int(t)
            for const, info in self.species.items():
                if info["dex"] == dex:
                    return const
            return None
        norm = re.sub(r"[^A-Z0-9]", "", t.upper())
        for const, info in self.species.items():
            if re.sub(r"[^A-Z0-9]", "", const) == norm:
                return const
            if re.sub(r"[^A-Z0-9]", "", info["name"].upper()) == norm:
                return const
        return None

    def item_const(self, text):
        if text is None:
            return None
        norm = re.sub(r"[^A-Z0-9]", "", str(text).upper())
        for const in self.items:
            if re.sub(r"[^A-Z0-9]", "", const) == norm:
                return const
        return None

    def species_id(self, const):
        return self.species[const]["id"]

    def dex_owned(self, emu, const) -> bool:
        """The wPokedexOwned bit for this species' dex number."""
        dex = self.species[const]["dex"]
        mem = emu.pb.memory
        return bool(mem[rr.ADDR_POKEDEX_OWNED + (dex - 1) // 8]
                    >> ((dex - 1) % 8) & 1)

    # -- learnsets (evos_moves.asm: evo entries, db 0, then
    #    `db level, MOVE` lines to a closing db 0) -------------
    def _parse_learnsets(self):
        from agent.dex_audit import parse_consts
        pk = self.pk
        sp_id = {}
        for name, v in parse_consts(
                str(pk / "constants" / "pokemon_constants.asm")).items():
            if v > 0:
                sp_id[name] = v
        inv_id = {v: k for k, v in sp_id.items()}
        txt = open(pk / "data" / "pokemon" / "evos_moves.asm").read()
        blocks = re.split(r"\n(?=[A-Za-z0-9_]+EvosMoves:)", txt)
        ptr = re.findall(r"dw\s+([A-Za-z0-9_]+)EvosMoves", txt)
        by_label = {}
        for idx, label in enumerate(ptr):
            by_label[label] = inv_id.get(idx + 1)
        out = {}
        for b in blocks:
            m = re.match(r"([A-Za-z0-9_]+)EvosMoves:", b)
            if not m:
                continue
            const = by_label.get(m.group(1))
            if const is None or const not in self.species:
                continue
            phase = "evo"
            learn = []
            for line in b.splitlines()[1:]:
                s = line.split(";")[0].strip()
                if phase == "evo":
                    if s == "db 0":
                        phase = "learn"
                    continue
                mm = re.match(r"db\s+(\d+),\s*([A-Z0-9_]+)$", s)
                if mm:
                    learn.append((int(mm.group(1)), mm.group(2)))
                elif s == "db 0":
                    break
            out[const] = learn
        return out

    # -- wild tables (grass + water per map file) -------------
    def _parse_wild(self):
        def red_lines(path):
            stack, emit = [], True
            for raw in open(path):
                line = raw.split(";")[0].rstrip()
                s = line.strip()
                if s.startswith("IF DEF(_RED)"):
                    stack.append(emit)
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
                        emit = stack[-1]
                    continue
                if s == "ENDC":
                    if stack:
                        emit = stack.pop()
                    continue
                if emit:
                    yield line

        wild = {}
        wdir = self.pk / "data" / "wild" / "maps"
        for fn in sorted(os.listdir(wdir)):
            if not fn.endswith(".asm"):
                continue
            grass, water, cur = [], [], None
            for line in red_lines(wdir / fn):
                s = line.strip()
                if s.startswith("def_grass_wildmons"):
                    cur = grass
                    continue
                if s.startswith("def_water_wildmons"):
                    cur = water
                    continue
                if s.startswith(("end_grass_wildmons",
                                  "end_water_wildmons")):
                    cur = None
                    continue
                m = re.match(r"db\s+(\d+),\s*([A-Z0-9_]+)$", s)
                if m and cur is not None:
                    cur.append((int(m.group(1)), m.group(2)))
            if grass or water:
                wild[fn[:-4]] = {"grass": grass, "water": water}
        return wild

    # -- mart stocks (data/items/marts.asm) -------------------
    _MART_LABEL2MAP = {
        "ViridianMartClerkText": "ViridianMart",
        "PewterMartClerkText": "PewterMart",
        "CeruleanMartClerkText": "CeruleanMart",
        "VermilionMartClerkText": "VermilionMart",
        "LavenderMartClerkText": "LavenderMart",
        "CeladonMart2FClerk1Text": "CeladonMart2F",
        "CeladonMart4FClerkText": "CeladonMart4F",
        "CeladonMart5FClerk1Text": "CeladonMart5F",
        "FuchsiaMartClerkText": "FuchsiaMart",
        "CinnabarMartClerkText": "CinnabarMart",
        "SaffronMartClerkText": "SaffronMart",
        "IndigoPlateauLobbyClerkText": "IndigoPlateauLobby",
    }
    # buy_mart_item clerk stances proven live (its docstring /
    # the legs); every other mart takes the Cerulean defaults.
    MART_STANCE = {
        "CeladonMart2F": ((5, 5), "up"),
        "CeladonMart4F": ((5, 5), "down"),
    }
    MART_TOWN = {
        "ViridianMart": "VIRIDIAN CITY",
        "PewterMart": "PEWTER CITY",
        "CeruleanMart": "CERULEAN CITY",
        "VermilionMart": "VERMILION CITY",
        "LavenderMart": "LAVENDER TOWN",
        "CeladonMart2F": "CELADON CITY",
        "CeladonMart4F": "CELADON CITY",
        "CeladonMart5F": "CELADON CITY",
        "FuchsiaMart": "FUCHSIA CITY",
        "CinnabarMart": "CINNABAR ISLAND",
        "SaffronMart": "SAFFRON CITY",
        "IndigoPlateauLobby": "INDIGO PLATEAU",
    }

    def _parse_marts(self):
        out = {}
        label = None
        for raw in open(self.pk / "data" / "items" / "marts.asm"):
            s = raw.split(";")[0].strip()
            m = re.match(r"([A-Za-z0-9_]+)::?$", s)
            if m:
                label = m.group(1)
                continue
            m = re.match(r"script_mart\s+(.+)$", s)
            if m and label in self._MART_LABEL2MAP:
                out[self._MART_LABEL2MAP[label]] = [
                    it.strip() for it in m.group(1).split(",")]
        return out

    def _parse_prices(self):
        out = {}
        for raw in open(self.pk / "data" / "items" / "prices.asm"):
            m = re.search(r"bcd3\s+(\d+)\s*;\s*([A-Z0-9_]+)", raw)
            if m:
                out[m.group(2)] = int(m.group(1))
        return out


_DATA = None


def data() -> Data:
    """The parsed pokered Data, or None when no checkout is
    available (data-dependent quests refuse on None)."""
    global _DATA
    if _DATA is None:
        pk = pokered_dir()
        if pk is not None:
            _DATA = Data(pk)
    return _DATA


def require_data():
    d = data()
    if d is None:
        raise QuestRefusal(
            "data", "pokered checkout not found (set POKERED_DIR "
            "or place it at ~/workspace/projects/pokered); this "
            "quest's tables are parsed from it")
    return d


# ------------------------------------------------------------ machinery

class QuestRefusal(Exception):
    """A precondition (or argument) failure: the quest did not
    run. Carries the structured refusal the callers report."""

    def __init__(self, precondition: str, reason: str):
        super().__init__(reason)
        self.precondition = precondition
        self.reason = reason


class QuestFailure(Exception):
    """An execution failure: the quest ran (and may have
    spent money or moved the player) but its postcondition
    does not hold. Reported as outcome 'failed', distinct
    from a refusal, where nothing was attempted."""


@dataclass
class Arg:
    name: str
    kind: str = "str"          # str | int | bool | species | item
                               # | town | species_list
    required: bool = False
    default: object = None
    help: str = ""


@dataclass
class Quest:
    name: str
    summary: str
    args: list
    run: object                # fn(Qtx, clean_args) -> detail str
    read_only: bool = False


@dataclass
class QuestResult:
    quest: str
    outcome: str               # ok | failed | refused
    detail: str
    args: dict = field(default_factory=dict)
    pre: dict = field(default_factory=dict)
    post: dict = field(default_factory=dict)
    refusal: str = ""          # the unmet precondition, if refused


QUESTS: dict = {}


def quest(name, summary, args=(), read_only=False):
    def deco(fn):
        QUESTS[name] = Quest(name, summary, list(args), fn,
                             read_only)
        return fn
    return deco


class Qtx:
    """One quest invocation's context: emulator, executor,
    world, logger, checkpoint directory."""

    def __init__(self, emu, ex, log=print, checkpoint_dir=None):
        self.emu = emu
        self.ex = ex
        self.world = ex.world if ex is not None else default_world()
        self.log = log
        self.checkpoint_dir = checkpoint_dir

    def state(self):
        return self.emu.read_state()

    def snap(self) -> dict:
        st = self.state()
        return {"map": self.world.map_name(st.map_id),
                "money": st.money,
                "dex": rr._dex_owned_count(self.emu)}

    def refuse(self, precondition, reason):
        raise QuestRefusal(precondition, reason)

    def checkpoint(self, tag: str):
        """Save a named state into the checkpoint directory
        (a quest's own pre-engagement insurance)."""
        if self.checkpoint_dir is None:
            return None
        d = Path(self.checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{tag}.state"
        self.emu.save_state(str(path))
        return path


def validate_args(name: str, raw: dict):
    """Validate + coerce a raw arg dict against the quest's
    schema. Pure (no emulator, no data): returns the clean
    args dict, or raises QuestRefusal('args', ...)."""
    q = QUESTS.get(name)
    if q is None:
        raise QuestRefusal("quest", f"unknown quest {name!r} "
                           f"(see --list)")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise QuestRefusal("args", "args must be an object")
    known = {a.name: a for a in q.args}
    for key in raw:
        if key not in known:
            raise QuestRefusal(
                "args", f"quest {name}: unknown argument {key!r} "
                f"(want: {', '.join(known) or 'none'})")
    clean = {}
    for a in q.args:
        if a.name in raw and raw[a.name] is not None:
            v = raw[a.name]
        elif a.required:
            raise QuestRefusal(
                "args", f"quest {name}: missing required "
                f"argument {a.name!r}")
        else:
            v = a.default
        clean[a.name] = _coerce(name, a, v)
    return clean


def _coerce(qname, a: Arg, v):
    if v is None:
        return None
    if a.kind == "int":
        try:
            return int(v)
        except (TypeError, ValueError):
            raise QuestRefusal(
                "args", f"quest {qname}: argument {a.name!r} "
                f"must be an integer, got {v!r}")
    if a.kind == "bool":
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.lower() in ("true", "false",
                                                "yes", "no"):
            return v.lower() in ("true", "yes")
        raise QuestRefusal(
            "args", f"quest {qname}: argument {a.name!r} must "
            f"be a boolean, got {v!r}")
    if a.kind == "species_list":
        if isinstance(v, str):
            parts = [p for p in re.split(r"[,\s]+", v) if p]
        elif isinstance(v, (list, tuple)):
            parts = [str(p) for p in v]
        else:
            raise QuestRefusal(
                "args", f"quest {qname}: argument {a.name!r} "
                f"must be a species name or list, got {v!r}")
        if a.required and not parts:
            raise QuestRefusal(
                "args", f"quest {qname}: argument {a.name!r} "
                "must name at least one species")
        return parts
    return str(v)


def run_quest(emu, ex, name, raw_args=None, log=print,
              checkpoint_dir=None) -> QuestResult:
    """The one entry point both surfaces call. Validates args,
    runs the quest, verifies via its own postconditions, and
    returns a structured QuestResult. A refusal or failure is
    a result, never an exception."""
    try:
        clean = validate_args(name, raw_args or {})
    except QuestRefusal as r:
        log(f"QUEST {name} -> REFUSED [{r.precondition}] "
            f"{r.reason}")
        return QuestResult(name, "refused", r.reason,
                           refusal=r.precondition)
    q = QUESTS[name]
    ctx = Qtx(emu, ex, log=log, checkpoint_dir=checkpoint_dir)
    pre = ctx.snap() if emu is not None else {}
    try:
        detail = q.run(ctx, clean)
    except QuestRefusal as r:
        post = ctx.snap() if emu is not None else {}
        log(f"QUEST {name} {clean} -> REFUSED "
            f"[{r.precondition}] {r.reason}")
        return QuestResult(name, "refused", r.reason, clean,
                           pre, post, refusal=r.precondition)
    except QuestFailure as exc:
        post = ctx.snap() if emu is not None else {}
        log(f"QUEST {name} {clean} -> FAILED {exc}")
        return QuestResult(name, "failed", str(exc), clean,
                           pre, post)
    except Exception as exc:                       # machinery broke
        post = ctx.snap() if emu is not None else {}
        log(f"QUEST {name} {clean} -> FAILED "
            f"(exception {exc!r})")
        return QuestResult(name, "failed",
                           f"exception during execution: {exc!r}",
                           clean, pre, post)
    post = ctx.snap() if emu is not None else {}
    log(f"QUEST {name} {clean} -> OK: {detail} "
        f"[money {pre.get('money')}->{post.get('money')}, "
        f"dex {pre.get('dex')}->{post.get('dex')}, "
        f"at {post.get('map')}]")
    return QuestResult(name, "ok", detail, clean, pre, post)


# ------------------------------------------------------- shared travel

FLY_TOWNS = rr.FLY_TOWN_MAP_NAMES          # fly name -> map name
TOWN_PC = {
    "PalletTown": None,
    "ViridianCity": "ViridianPokecenter",
    "PewterCity": "PewterPokecenter",
    "CeruleanCity": "CeruleanPokecenter",
    "LavenderTown": "LavenderPokecenter",
    "VermilionCity": "VermilionPokecenter",
    "CeladonCity": "CeladonPokecenter",
    "FuchsiaCity": "FuchsiaPokecenter",
    "CinnabarIsland": "CinnabarPokecenter",
    "SaffronCity": "SaffronPokecenter",
    "IndigoPlateau": "IndigoPlateauLobby",
}
PC_MAP_NAMES = {v for v in TOWN_PC.values() if v}


def resolve_town(text, world):
    """'Fuchsia', 'FUCHSIA CITY' or 'FuchsiaCity' -> the
    FLY_TOWN_MAP_NAMES key, or None."""
    if text is None:
        return None
    t = str(text).strip()
    if t.upper() in FLY_TOWNS:
        return t.upper()
    norm = re.sub(r"[^A-Z0-9]", "", t.upper())
    for key, map_name in FLY_TOWNS.items():
        if re.sub(r"[^A-Z0-9]", "", key) == norm:
            return key
        if re.sub(r"[^A-Z0-9]", "", map_name.upper()) == norm:
            return key
    return None


def party_has_move(emu, move_id) -> bool:
    from agent.state import ADDR_PARTY_MONS
    mem = emu.pb.memory
    for i, mon in enumerate(emu.read_state().party):
        moves = [mem[ADDR_PARTY_MONS + 44 * i + 8 + j]
                 for j in range(4)]
        if move_id in moves:
            return True
    return False


class _Watchdog(Exception):
    """A timeboxed primitive exceeded its wall-clock budget."""


def _timeboxed(seconds, fn, *a, **k):
    """Run fn under a wall-clock budget (SIGALRM; main thread
    only - on any other thread fn simply runs unboxed). The
    ported menu rides can spin forever on a state they do not
    recognise (the S40 fund_e4 home-flight hang: fly_to looped
    for 23 minutes on a fainted Fly carrier); a quest must
    degrade to an honest failure, never hang. Raises
    _Watchdog on expiry."""
    import signal
    import threading
    if threading.current_thread() is not \
            threading.main_thread():
        return fn(*a, **k)

    def handler(signum, frame):
        raise _Watchdog(f"exceeded {seconds}s budget")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*a, **k)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _fly_carrier(ctx):
    """(party index, mon) of the first party member knowing
    Fly, or None."""
    from agent.state import ADDR_PARTY_MONS
    mem = ctx.emu.pb.memory
    for i, mon in enumerate(ctx.state().party):
        moves = [mem[ADDR_PARTY_MONS + 44 * i + 8 + j]
                 for j in range(4)]
        if rr.MOVE_FLY in moves:
            return i, mon
    return None


def ensure_outdoors(ctx) -> bool:
    """Exit buildings until the player is outside (Fly and
    the journeys need it); the probe178 home-loop pattern."""
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    for _ in range(4):
        st = ctx.state()
        if world.is_outside(st.map_id):
            return True
        if not rr._exit_building(emu, ex, st.map_id):
            return False
    return world.is_outside(ctx.state().map_id)


def fly(ctx, town_key: str) -> bool:
    if not ensure_outdoors(ctx):
        return False
    # a state saved right after a flight can carry held-input
    # residue that eats fly_to's start-menu press (the leg-11
    # lesson, relearned by the S40 travel smoke): release and
    # settle first
    rr.release_held_buttons(ctx.emu)
    ctx.emu.tick(120)
    try:
        return _timeboxed(120, rr.fly_to, ctx.emu, ctx.ex,
                          town_key)
    except _Watchdog:
        return False


def go_pc(ctx, fly_town: str = None) -> int:
    """Get the player inside a Pokecenter (or the Indigo
    lobby) and return its map id. Prefers the center of the
    town the player is already in; otherwise flies to
    fly_town (default Fuchsia). Raises QuestRefusal on
    travel failure."""
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    st = ctx.state()
    here = world.map_name(st.map_id)
    if here in PC_MAP_NAMES:
        return st.map_id
    pc_name = TOWN_PC.get(here)
    if pc_name:
        pc = world.map_id(pc_name)
        if rr._ride_battling(emu, ex, pc):
            return pc
    town = fly_town or "FUCHSIA CITY"
    key = resolve_town(town, world)
    if key is None or not party_has_move(emu, rr.MOVE_FLY):
        ctx.refuse("travel", f"cannot reach a Pokecenter from "
                   f"{here} (no local center, no Fly carrier)")
    if not fly(ctx, key):
        ctx.refuse("travel", f"Fly to {key} failed")
    pc = world.map_id(TOWN_PC[FLY_TOWNS[key]])
    if not rr._ride_battling(emu, ex, pc):
        ctx.refuse("travel", f"ride into {TOWN_PC[FLY_TOWNS[key]]} "
                   "failed after landing")
    return pc


def heal_party(ctx, pc: int) -> bool:
    """Heal at the given center/lobby and verify EVERY party
    member is at full HP in RAM."""
    emu, ex = ctx.emu, ctx.ex
    if pc == ctx.world.map_id("IndigoPlateauLobby"):
        rr._heal_at_pc(emu, ex, pc)
    else:
        rr.heal_at_pokecenter(emu, ex, pc)
    st = ctx.state()
    return bool(st.party) and all(
        m.hp == m.max_hp for m in st.party)


def ensure_species_in_party(ctx, species_id: int, pc: int):
    """Withdraw `species_id` from whichever box holds it
    (depositing the last party member that is not the sole
    Fly carrier when the party is full). True when it is in
    the party afterwards."""
    emu, ex = ctx.emu, ctx.ex
    if rr._party_index_of(emu, species_id) is not None:
        return True
    census = rr._box_census(emu)
    hits = rr._box_find_species(census, species_id)
    if not hits:
        return False
    box_no = hits[0][0]
    if rr._current_box_num(emu) != box_no:
        if not rr.pc_change_box(emu, ex, pc, box_no):
            return False
    st = ctx.state()
    if len(st.party) < 6:
        return rr.pc_withdraw_mon(emu, ex, pc, species_id)
    bench = None
    flyers = [m for m in st.party
              if rr.MOVE_FLY in m.moves]
    for m in reversed(st.party):
        if m.species_id == species_id:
            continue
        if len(flyers) == 1 and m is flyers[0]:
            continue
        bench = m.species_id
        break
    if bench is None:
        return False
    return rr.pc_swap_mon(emu, ex, pc, bench, species_id)


def box_headroom(ctx) -> bool:
    """A catch/receive can file: party not full, or the
    active box has a free slot (the S30 full-box lesson)."""
    if len(ctx.state().party) < 6:
        return True
    return rr._box_count(ctx.emu) < 20



# ------------------------------------------------------------ quests

@quest("dex_audit",
       "Read-only Pokedex audit: owned vs the solo-Red "
       "ceiling, with the obtainable species still missing.",
       read_only=True)
def q_dex_audit(ctx, args):
    d = require_data()
    owned = [c for c in d.species if d.dex_owned(ctx.emu, c)]
    missing = [info["name"] for c, info in d.species.items()
               if info["status"] == "obtainable"
               and c not in owned]
    locked = [info["name"] for c, info in d.species.items()
              if info["status"] == "choice-locked"
              and c not in owned]
    ceiling = d.ledger["ceiling"]
    detail = (f"dex {len(owned)}/{ceiling} owned; "
              f"{len(missing)} obtainable species missing")
    if missing:
        detail += ": " + ", ".join(sorted(missing))
    if locked:
        detail += (f"; choice-locked and unchosen on this "
                   f"save: {', '.join(sorted(locked))}")
    ctx.log(f"DEX_AUDIT {detail}")
    return detail


@quest("heal", "Heal the party at the nearest Pokecenter "
       "(postcondition: every member at full HP).")
def q_heal(ctx, args):
    pc = go_pc(ctx)
    if not heal_party(ctx, pc):
        raise QuestFailure("party not at full HP after the "
                           "Pokecenter visit")
    return f"party healed at {ctx.world.map_name(pc)}"


@quest("travel", "Fly to a visited town (postcondition: RAM "
       "map is the destination).",
       args=[Arg("town", "town", required=True,
                 help="town name, e.g. 'Fuchsia City'")])
def q_travel(ctx, args):
    emu, world = ctx.emu, ctx.world
    key = resolve_town(args["town"], world)
    if key is None:
        ctx.refuse("args", f"unknown Fly destination "
                   f"{args['town']!r} (want one of: "
                   f"{', '.join(FLY_TOWNS)})")
    dest = world.map_id(FLY_TOWNS[key])
    st = ctx.state()
    if st.map_id == dest:
        return f"already at {key}"
    if st.in_battle:
        ctx.refuse("state", "cannot travel mid-battle")
    if not party_has_move(emu, rr.MOVE_FLY):
        ctx.refuse("fly", "no party member knows Fly")
    if not fly(ctx, key):
        ctx.refuse("travel", f"Fly to {key} failed")
    if ctx.state().map_id != dest:
        raise QuestFailure(
            f"landed on {world.map_name(ctx.state().map_id)}, "
            f"not {key}")
    return f"flew to {key}"


# --- fund_e4 ------------------------------------------------

_LEARN_GUARD_WINDOW = 4     # levels of headroom (docs S39)


def _learn_prompt_risks(ctx, d):
    """Party members whose next learnset level falls within
    the guard window - the S39 failure (Snorlax's Hyper Beam
    prompt wedging the league ride at L55->56) made a class:
    detect it BEFORE the gauntlet, never inside it."""
    risks = []
    for mon in ctx.state().party:
        const = d.by_id.get(mon.species_id)
        if const is None:
            continue
        for lv, move in d.learnsets.get(const, []):
            if mon.level < lv <= mon.level + _LEARN_GUARD_WINDOW:
                risks.append(
                    f"{const} L{mon.level} would try to learn "
                    f"{move} at L{lv}")
                break
    return risks


@quest("fund_e4", "Elite Four money run(s): assemble at the "
       "Indigo lobby, run the gauntlet, come home healed. "
       "Refuses when a party member would hit a learn prompt "
       "inside the guard window (the S39 lesson).",
       args=[Arg("runs", "int", default=1,
                 help="number of full gauntlet runs (1-3)")])
def q_fund_e4(ctx, args):
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    runs = args["runs"]
    if not 1 <= runs <= 3:
        ctx.refuse("args", f"runs must be 1-3, got {runs}")
    st = ctx.state()
    if len(st.badges) < 8:
        ctx.refuse("badges", "the Elite Four needs all 8 "
                   f"badges (have {len(st.badges)})")
    if len(st.party) < 6:
        ctx.refuse("party", "a funding run takes a full party "
                   f"of six (have {len(st.party)}) - the S31 "
                   "attrition lesson")
    d = data()
    if d is not None:
        risks = _learn_prompt_risks(ctx, d)
        if risks:
            ctx.refuse("learn_prompt_risk",
                       "; ".join(risks) + " - grind the named "
                       "member past the prompt level first "
                       "(the grind quest's driver is "
                       "prompt-aware)")
    lobby = world.map_id("IndigoPlateauLobby")
    money0 = st.money
    done = 0
    for run in range(runs):
        if ctx.state().map_id != lobby:
            if not fly(ctx, "INDIGO PLATEAU"):
                ctx.refuse("travel", "Fly to INDIGO PLATEAU "
                           "failed")
            if not rr._ride_battling(emu, ex, lobby):
                ctx.refuse("travel", "ride into the Indigo "
                           "lobby failed")
        # the proven runs both carried a Revive; top up at the
        # lobby mart when the bag has none and money allows
        revive = d.items["REVIVE"] if d else 0x28
        if rr._bag_count(emu, revive) == 0 \
                and ctx.state().money >= 1500:
            idx = None
            if d is not None:
                stock = d.marts.get("IndigoPlateauLobby", [])
                if "REVIVE" in stock:
                    idx = stock.index("REVIVE")
            if idx is not None:
                rr.buy_mart_item(emu, ex, lobby, idx, revive,
                                 rr._bag_count(emu, revive) + 1)
        if not heal_party(ctx, lobby):
            ctx.refuse("heal", "lobby heal did not bring the "
                       "party to full HP")
        if not rr.elite_four(emu, ex):
            raise QuestFailure(
                f"E4 run {run + 1} did not complete ({done} "
                f"completed first); money {money0} -> "
                f"{ctx.state().money}")
        done += 1
        # bank the win immediately: a later hang or kill must
        # not lose a completed gauntlet (the S40 lesson)
        ctx.checkpoint(f"fund_e4-run{done}")
    # home to Fuchsia, healed (the probe178 loop). Two S40
    # failure modes are handled explicitly: the Fly carrier
    # can be FAINTED after a hard gauntlet (fly_to then spins
    # forever - it hung 23 minutes in the first smoke), and
    # every ride home is timeboxed so a hang degrades to an
    # honest failure with the money already banked.
    money_now = ctx.state().money
    carrier = _fly_carrier(ctx)
    if carrier is not None and carrier[1].hp == 0:
        idx = carrier[0]
        if rr._bag_count(emu, rr.ITEM_REVIVE) > 0 and \
                rr._use_bag_item_on_mon(emu, rr.ITEM_REVIVE,
                                        idx):
            ctx.log(f"fund_e4: revived the Fly carrier "
                    f"(party slot {idx}) for the trip home")
        if ctx.state().party[idx].hp == 0:
            raise QuestFailure(
                f"{done} E4 run(s) complete and the money "
                f"banked ({money0} -> {money_now}), but the "
                f"trip home is impossible: the Fly carrier "
                f"is fainted and no Revive is available - "
                f"the saved state is post-gauntlet; heal at "
                f"a Pokecenter and re-run to finish the trip")
    fuchsia_pc = world.map_id("FuchsiaPokecenter")
    try:
        for _ in range(4):
            st = ctx.state()
            if st.map_id == fuchsia_pc:
                break
            name = world.map_name(st.map_id)
            if name.endswith("Pokecenter") or "House" in name \
                    or "Lobby" in name:
                rr._exit_building(emu, ex, st.map_id)
            fly(ctx, "FUCHSIA CITY")
            if ctx.state().map_id != fuchsia_pc \
                    and world.map_name(ctx.state().map_id) \
                    == "FuchsiaCity":
                # ride only within Fuchsia: from anywhere
                # else the ride is a cross-map journey, the
                # spin-prone path - fail banked instead
                _timeboxed(120, rr._ride_battling, emu, ex,
                           fuchsia_pc)
    except _Watchdog:
        raise QuestFailure(
            f"{done} E4 run(s) complete and the money banked "
            f"({money0} -> {money_now}), but the trip home "
            f"hung and was timeboxed out - the saved state "
            f"is post-gauntlet; heal at a Pokecenter and "
            f"re-run to finish the trip")
    if ctx.state().map_id != fuchsia_pc \
            or not heal_party(ctx, fuchsia_pc):
        raise QuestFailure(
            f"{done} E4 run(s) complete and the money banked "
            f"({money0} -> {money_now}), but the trip home "
            f"did not complete (ended at "
            f"{world.map_name(ctx.state().map_id)}) - the "
            f"saved state holds the banked money; finish "
            f"the trip from a fresh boot")
    money1 = ctx.state().money
    return (f"{done} E4 run(s) complete; money {money0} -> "
            f"{money1} (net {money1 - money0:+d})")


# --- grind ----------------------------------------------------

@quest("grind", "Prompt-aware Mansion switch-grind of one "
       "party/box member to a target level. All move "
       "replacements are DECLINEd; free-slot auto-learns are "
       "the engine's own and are audited, not hidden.",
       args=[Arg("target", "species", required=True,
                 help="party slot number (0-5) or species name"),
             Arg("to_level", "int", required=True,
                 help="target level (current+1 .. 100)"),
             Arg("area", "str", default="mansion",
                 help="only 'mansion' is ported")])
def q_grind(ctx, args):
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    d = require_data()
    if str(args["area"]).lower() != "mansion":
        ctx.refuse("args", "only area=mansion is ported "
                   "(the Mansion 1F grind)")
    to_level = args["to_level"]
    if not 2 <= to_level <= 100:
        ctx.refuse("args", f"to_level must be 2-100, "
                   f"got {to_level}")
    # resolve the target: a slot number, or a species
    target = str(args["target"])
    sp = None
    if target.isdigit() and int(target) < 6:
        st = ctx.state()
        if int(target) >= len(st.party):
            ctx.refuse("target", f"party slot {target} is "
                       "empty")
        sp = st.party[int(target)].species_id
    else:
        const = d.species_const(target)
        if const is None:
            ctx.refuse("target", f"unknown species "
                       f"{target!r}")
        sp = d.species_id(const)
    idx, mon = rr._mon_of(emu, sp)
    if mon is None:
        # boxed: bring it in at the Cinnabar PC
        pc = go_pc(ctx, fly_town="CINNABAR ISLAND")
        if not ensure_species_in_party(ctx, sp, pc):
            ctx.refuse("target", "species is neither in the "
                       "party nor in any box")
        idx, mon = rr._mon_of(emu, sp)
    if mon.level >= to_level:
        ctx.refuse("level", f"{d.by_id.get(sp, sp)} is already "
                   f"L{mon.level} (>= {to_level})")
    const = d.by_id[sp]
    # the learn plan: simulate the engine - a learnset move
    # auto-appends while a slot is free, otherwise the prompt
    # is DECLINEd (Vlad's standing grind policy)
    learn = {}
    expected = [m for m in mon.moves if m]
    count = len(expected)
    for lv, move_const in d.learnsets.get(const, []):
        if mon.level < lv <= to_level:
            move_id = d.moves[move_const]
            if count < 4:
                learn[lv] = (move_id, ("append",))
                expected.append(move_id)
                count += 1
            else:
                learn[lv] = (move_id, ("decline",))
    moves0 = [m for m in mon.moves if m]
    if not fly(ctx, "CINNABAR ISLAND"):
        ctx.refuse("travel", "Fly to CINNABAR ISLAND failed")
    if not rr.party_lead(emu, sp):
        ctx.refuse("party", "could not lead the grind target")
    m1 = world.map_id("PokemonMansion1F")
    if ctx.state().map_id != m1 and not rr._enter_mansion(emu, ex):
        ctx.refuse("travel", "could not enter the Mansion")
    ok, fights, trips, blackouts = rr._league_grind_one(
        emu, ex, sp, to_level, learn, tag=f"quest-grind-{const}")
    if not ok:
        raise QuestFailure(
            f"Mansion grind stalled at "
            f"L{rr._mon_of(emu, sp)[1].level} after {fights} "
            f"fights ({blackouts} blackouts)")
    idx, mon2 = rr._mon_of(emu, sp)
    got_moves = [m for m in mon2.moves if m]
    if mon2.level < to_level:
        raise QuestFailure(f"level {mon2.level} < {to_level}")
    if got_moves != expected:
        raise QuestFailure(
            f"moveset drift: expected {expected}, RAM shows "
            f"{got_moves} (start {moves0})")
    rr._exit_mansion(emu, ex)
    pc = world.map_id("CinnabarPokecenter")
    if rr._ride_battling(emu, ex, pc):
        heal_party(ctx, pc)
    return (f"{const} ground to L{mon2.level} "
            f"({fights} fights, {trips} heal trips, "
            f"{blackouts} blackouts); moves {moves0} -> "
            f"{got_moves}")


# --- catch_wild ------------------------------------------------

SLOT_CHANCES = [19.9, 19.9, 15.2, 9.8, 9.8, 9.8, 5.1, 5.1, 4.3, 1.2]
_GATEWAYS = (
    ("CeruleanCave", "CERULEAN CITY"),
    ("SeafoamIslands", "FUCHSIA CITY"),
    ("PokemonMansion", "CINNABAR ISLAND"),
    ("PowerPlant", "CERULEAN CITY"),
    ("VictoryRoad", "VIRIDIAN CITY"),
    ("RockTunnel", "CERULEAN CITY"),
    ("MtMoon", "PEWTER CITY"),
    ("DiglettsCave", "VERMILION CITY"),
    ("PokemonTower", "LAVENDER TOWN"),
    ("SafariZone", "FUCHSIA CITY"),
    ("SilphCo", "SAFFRON CITY"),
    ("RocketHideout", "CELADON CITY"),
)


def _go_map(ctx, map_id: int) -> bool:
    """Ride the world journey to a map; on failure, Fly to
    the map's gateway town and ride again."""
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    if ctx.state().map_id == map_id:
        return True
    if rr._ride_battling(emu, ex, map_id) \
            and ctx.state().map_id == map_id:
        return True
    name = world.map_name(map_id)
    town = next((t for p, t in _GATEWAYS if name.startswith(p)),
                None)
    if town and party_has_move(emu, rr.MOVE_FLY) and fly(ctx, town):
        if rr._ride_battling(emu, ex, map_id):
            return ctx.state().map_id == map_id
    return ctx.state().map_id == map_id


def _pair_candidates(ctx, map_id: int, limit: int = 3):
    """Adjacent walkable pairs on a map, best first: pairs
    whose tiles match the map's modal walkable tile (grass /
    cave floor) outrank path tiles. Spread out so a sterile
    pair (the S37 lesson) costs one rotation, not the hunt."""
    from collections import Counter
    gm = ctx.world.maps[map_id]
    warps = {(w.x, w.y) for w in gm.warps}
    modal = None
    if gm.tiles:
        counts = Counter()
        for y in range(gm.h):
            for x in range(gm.w):
                if gm.tile_walkable(x, y):
                    counts[gm.tiles[y][x]] += 1
        if counts:
            modal = counts.most_common(1)[0][0]
    scored = []
    for y in range(gm.h):
        for x in range(gm.w):
            for dx, dy in ((1, 0), (0, 1)):
                a, b = (x, y), (x + dx, y + dy)
                if a in warps or b in warps:
                    continue
                if gm.tile_walkable(*a) and gm.tile_walkable(*b):
                    score = 0
                    if gm.tiles and modal is not None:
                        score = int(gm.tiles[a[1]][a[0]] == modal) \
                            + int(gm.tiles[b[1]][b[0]] == modal)
                    scored.append((score, a, b))
    scored.sort(key=lambda c: -c[0])
    picked = []
    for _score, a, b in scored:
        if all(abs(a[0] - p[0][0]) + abs(a[1] - p[0][1]) > 6
               for p in picked):
            picked.append((a, b))
        if len(picked) >= limit:
            break
    return picked


def _best_ball(ctx, d):
    for const in ("ULTRA_BALL", "GREAT_BALL", "POKE_BALL"):
        iid = d.items[const]
        if rr._bag_count(ctx.emu, iid) > 0:
            return iid
    return None


def _pace_hunt(ctx, pair, want_sp, d, max_enc, sterile_steps=50):
    """Pace a pair hunting one species with the careful
    thrower. Returns (encounters, caught). A pair that
    produces nothing in `sterile_steps` paces is abandoned
    (returns (0, False)) so the caller can rotate pairs."""
    emu = ctx.emu
    mem = emu.pb.memory
    encounters = 0
    steps = 0
    toggle = False
    while encounters < max_enc:
        st = ctx.state()
        if st.in_battle:
            encounters += 1
            for _ in range(200):
                if mem[rr.ADDR_ENEMY_SPECIES] != 0 and \
                        mem[rr.ADDR_ENEMY_MAX_HP] != 0:
                    break
                emu.tick(10)
            sp = mem[rr.ADDR_ENEMY_SPECIES]
            if sp == want_sp:
                ball = _best_ball(ctx, d)
                if ball is None:
                    return encounters, False
                if rr._catch_current_careful(emu, ball):
                    return encounters, True
            elif not rr._run_away(emu) \
                    and ctx.state().in_battle:
                rr._clear_wild_battle(emu)
            continue
        if encounters == 0 and steps >= sterile_steps:
            return 0, False
        tgt = pair[1] if toggle else pair[0]
        toggle = not toggle
        cur = (st.x, st.y)
        if cur == tgt:
            continue
        for dname, (dx, dy) in (("up", (0, -1)), ("down", (0, 1)),
                                ("left", (-1, 0)),
                                ("right", (1, 0))):
            if (cur[0] + dx, cur[1] + dy) == tgt:
                emu.press(dname)
                break
        else:
            return encounters, False     # knocked off the pair
        emu.tick(24)
        steps += 1
    return encounters, False


@quest("catch_wild", "Travel to a species' verified grass "
       "habitat (pokered encounter tables) and careful-catch "
       "one specimen.",
       args=[Arg("species", "species", required=True),
             Arg("area", "str", default=None,
                 help="restrict to this map (e.g. 'Route 24')"),
             Arg("allow_owned", "bool", default=False,
                 help="catch a duplicate of an owned species"),
             Arg("max_encounters", "int", default=120)])
def q_catch_wild(ctx, args):
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    d = require_data()
    const = d.species_const(args["species"])
    if const is None:
        ctx.refuse("args", f"unknown species {args['species']!r}")
    sp = d.species_id(const)
    if d.dex_owned(emu, const) and not args["allow_owned"]:
        ctx.refuse("owned", f"{const} is already owned "
                   "(allow_owned=true catches a duplicate, "
                   "e.g. trade fodder)")
    if _best_ball(ctx, d) is None:
        ctx.refuse("balls", "no Poke/Great/Ultra Balls in the "
                   "bag")
    if not box_headroom(ctx):
        ctx.refuse("box", "party full and the active box is "
                   "full (20/20) - box_switch to an emptier "
                   "box first (the S30 lesson)")
    # habitat: best grass slot across the Red tables
    area_norm = re.sub(r"[^A-Z0-9]", "",
                       str(args["area"]).upper()) \
        if args["area"] else None
    best = None          # (chance, map_name, level)
    water_only = False
    for stem, tab in d.wild.items():
        try:
            map_id = world.map_id(stem)
        except (KeyError, TypeError):
            continue
        map_name = world.map_name(map_id)
        if area_norm and re.sub(r"[^A-Z0-9]", "",
                                map_name.upper()) != area_norm:
            continue
        for i, (lvl, spc) in enumerate(tab["grass"]):
            if spc == const and (best is None
                                 or SLOT_CHANCES[i] > best[0]):
                best = (SLOT_CHANCES[i], map_name, lvl)
        if any(spc == const for _l, spc in tab["water"]):
            water_only = True
    if best is None:
        why = "a surf/fishing-only species in the Red tables " \
              "(v1 hunts grass)" if water_only else \
              "no grass habitat in the Red tables"
        ctx.refuse("habitat", f"{const}: {why}")
    chance, map_name, lvl = best
    map_id = world.map_id(map_name)
    if not _go_map(ctx, map_id):
        raise QuestFailure(f"could not reach habitat "
                           f"{map_name}")
    specimens0 = rr._specimen_count(emu, sp)
    pairs = _pair_candidates(ctx, map_id)
    if not pairs:
        raise QuestFailure(f"no pacing pair found on "
                           f"{map_name}")
    total_enc = 0
    caught = False
    for pair in pairs:
        rr._walk_fighting(emu, ex, map_id, pair[0],
                          attempts=30, flee_wild=True)
        if (ctx.state().x, ctx.state().y) != pair[0]:
            continue
        enc, caught = _pace_hunt(ctx, pair, sp, d,
                                 args["max_encounters"])
        total_enc += enc
        if caught or total_enc >= args["max_encounters"]:
            break
        if ctx.state().map_id != map_id:
            raise QuestFailure("left the habitat mid-hunt "
                               "(blackout?) at "
                               f"{world.map_name(ctx.state().map_id)}")
    if not caught or rr._specimen_count(emu, sp) <= specimens0:
        raise QuestFailure(
            f"{const} not caught on {map_name} after "
            f"{total_enc} encounters (best slot {chance}% "
            f"L{lvl})")
    return (f"{const} caught on {map_name} "
            f"({total_enc} encounters, best slot {chance}%)")


# --- catch_static ----------------------------------------------

_STATICS = {
    "SNORLAX": {"kind": "snorlax", "map": "Route16",
                "stance": (27, 10)},
    "ZAPDOS": {"kind": "zapdos", "map": "PowerPlant",
               "stance": (4, 10)},
    "ARTICUNO": {"kind": "legendary", "map": "SeafoamIslandsB4F",
                 "stance": (6, 2), "face": "up", "level": 50,
                 "hp": (150, 180), "dfn": (80, 140),
                 "recover": False},
    "MOLTRES": {"kind": "legendary", "map": "VictoryRoad2F",
                "stance": (11, 6), "face": "up", "level": 50,
                "hp": (140, 170), "dfn": (80, 140),
                "recover": False},
    "MEWTWO": {"kind": "legendary", "map": "CeruleanCaveB1F",
               "stance": (27, 14), "face": "up", "level": 70,
               "hp": (225, 255), "dfn": (105, 165),
               "recover": True},
}


@quest("catch_static", "Catch a static encounter (Snorlax, "
       "the three birds, Mewtwo) from its stance on its own "
       "map, with a pre-engagement checkpoint and up to 3 "
       "attempts. Inter-map travel is NOT included - the "
       "bespoke descents (Seafoam boulders, Victory Road "
       "loop, cave ladders) remain leg work; this quest owns "
       "verification, checkpointing and the catch.",
       args=[Arg("species", "species", required=True)])
def q_catch_static(ctx, args):
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    d = require_data()
    const = d.species_const(args["species"])
    if const is None or const not in _STATICS:
        ctx.refuse("species", f"{args['species']!r} is not a "
                   "quest-supported static (want: "
                   f"{', '.join(sorted(_STATICS))}; item-ball "
                   "statics are not supported)")
    spec = _STATICS[const]
    sp = d.species_id(const)
    if d.dex_owned(emu, const):
        ctx.refuse("owned", f"{const} is already owned - the "
                   "static encounter is consumed")
    map_id = world.map_id(spec["map"])
    if ctx.state().map_id != map_id:
        ctx.refuse("location", f"{const} stands on "
                   f"{spec['map']}; the player is on "
                   f"{world.map_name(ctx.state().map_id)} "
                   "(catch_static does not travel between "
                   "maps)")
    if not box_headroom(ctx):
        ctx.refuse("box", "party full and the active box is "
                   "full (20/20)")
    import tempfile
    ck_dir = ctx.checkpoint_dir or Path(
        tempfile.mkdtemp(prefix="quest-ckpt-"))
    if spec["kind"] == "snorlax":
        return _catch_snorlax(ctx, d, const, sp, spec, ck_dir)
    # the birds + Mewtwo run the Lapras Sing discipline
    lapras = d.species_id("LAPRAS")
    idx = rr._party_index_of(emu, lapras)
    if idx is None:
        ctx.refuse("party", "Lapras (the Sing + Strength "
                   "carrier) is not in the party")
    _i, mon = rr._mon_of(emu, lapras)
    if rr.MOVE_SING not in mon.moves \
            or rr.MOVE_STRENGTH not in mon.moves:
        ctx.refuse("party", "Lapras must know Sing + Strength "
                   f"(has {list(mon.moves)})")
    if mon.hp < mon.max_hp:
        ctx.refuse("party", f"Lapras is at {mon.hp}/"
                   f"{mon.max_hp} HP - heal before engaging "
                   "a legendary")
    ultra = d.items["ULTRA_BALL"]
    if rr._bag_count(emu, ultra) < 5:
        ctx.refuse("balls", "fewer than 5 Ultra Balls in the "
                   "bag for a catch-rate-3 static")
    if not rr.party_lead(emu, lapras):
        ctx.refuse("party", "could not lead Lapras")
    rr._walk_fighting(emu, ex, map_id, spec["stance"],
                      attempts=40, flee_wild=True)
    st = ctx.state()
    if (st.x, st.y) != spec["stance"]:
        raise QuestFailure(f"could not reach the stance "
                           f"{spec['stance']} on {spec['map']} "
                           f"(at {(st.x, st.y)})")
    ck = Path(ck_dir) / f"{const.lower()}-pre.state"
    ck.parent.mkdir(parents=True, exist_ok=True)
    emu.save_state(str(ck))
    specimens0 = rr._specimen_count(emu, sp)
    result = "failed"
    for attempt in range(1, 4):
        if attempt > 1:
            emu.load_state(str(ck))
            emu.tick(30)
        if spec["kind"] == "zapdos":
            result = rr.catch_zapdos(emu, ex, log=ctx.log)
        else:
            result = rr.catch_static_legendary(
                emu, ex, map_id=map_id, stance=spec["stance"],
                face_dir=spec["face"], species=sp,
                level=spec["level"], max_hp_range=spec["hp"],
                dfn_range=spec["dfn"],
                recover_aware=spec["recover"], log=ctx.log)
        if result == "caught":
            break
    if result != "caught" \
            or rr._specimen_count(emu, sp) <= specimens0:
        emu.load_state(str(ck))
        emu.tick(30)
        raise QuestFailure(
            f"{const}: {result} after 3 attempts; state "
            f"restored to the pre-engagement checkpoint "
            f"{ck}")
    return (f"{const} CAUGHT on {spec['map']} "
            f"(checkpoint {ck})")


def _catch_snorlax(ctx, d, const, sp, spec, ck_dir):
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    if rr._event_flag(emu, rr.EV_BEAT_R16_SNORLAX):
        ctx.refuse("event", "the Route 16 Snorlax has "
                   "already been battled (event flag set)")
    flute = d.items["POKE_FLUTE"]
    if rr._bag_count(emu, flute) == 0:
        ctx.refuse("item", "the Poke Flute is not in the bag")
    oddish = d.species_id("ODDISH")
    if rr._party_index_of(emu, oddish) is None:
        ctx.refuse("party", "snorlax_catch's proven policy "
                   "needs Oddish (Sleep Powder) in the party")
    map_id = world.map_id(spec["map"])
    rr._walk_fighting(emu, ex, map_id, spec["stance"],
                      attempts=40, flee_wild=True)
    st = ctx.state()
    if (st.x, st.y) != spec["stance"]:
        raise QuestFailure(f"could not reach the Snorlax "
                           f"stance {spec['stance']} "
                           f"(at {(st.x, st.y)})")
    ck = Path(ck_dir) / "snorlax-pre.state"
    ck.parent.mkdir(parents=True, exist_ok=True)
    emu.save_state(str(ck))
    specimens0 = rr._specimen_count(emu, sp)
    result = rr.snorlax_catch(emu, ex)
    if result != "caught" \
            or rr._specimen_count(emu, sp) <= specimens0:
        emu.load_state(str(ck))
        emu.tick(30)
        raise QuestFailure(
            f"SNORLAX: {result}; state restored to the "
            f"pre-engagement checkpoint {ck} (the quest "
            "promises a catch - the storyline's KO fallback "
            "is not spent here)")
    return f"SNORLAX CAUGHT on Route 16 (checkpoint {ck})"


# --- safari_hunt -------------------------------------------------
#
# The S37/S38 composition with Vlad's standing policy baked
# in: UNBAITED throws only (the legs' A/B verdict: unbaited
# 11/23 caught vs baited 5/17). The battle/settle/pace below
# are probe168 hunt.py's, parameterised by target set.

def _safari_area_ids():
    return {"SafariZoneCenter": rr.MAP_SZ_CENTER,
            "SafariZoneEast": rr.MAP_SZ_EAST,
            "SafariZoneNorth": rr.MAP_SZ_NORTH,
            "SafariZoneWest": rr.MAP_SZ_WEST}


def _safari_battle_u(ctx, targets, owned, slog):
    """One Safari battle, unbaited-only: RUN from owned
    non-targets, reserve balls on bycatch, throw at targets
    (cap 12) and unowned bycatch (cap 3)."""
    emu = ctx.emu
    mem = emu.pb.memory
    sp = mem[rr.ADDR_ENEMY_SPECIES]
    lv = mem[rr.ADDR_ENEMY_LEVEL]
    name = rr.SAFARI_SPECIES.get(sp, f"0x{sp:02x}")
    rec = {"species": name, "level": lv, "balls": 0,
           "outcome": "?"}
    is_target = sp in targets
    if sp in owned and not is_target:
        rr._run_away(emu)
        rec["outcome"] = "fled(owned)"
        slog.append(rec)
        return rec
    if not is_target and rr._safari_balls(emu) < 8:
        rr._run_away(emu)
        rec["outcome"] = "fled(reserve)"
        slog.append(rec)
        return rec
    slog.append(rec)
    cap = 12 if is_target else 3
    stored0 = rr._stored_count(emu)
    guard = 0
    while ctx.state().in_battle and guard < 40:
        guard += 1
        if rr._safari_balls(emu) == 0:
            rec["outcome"] = "balls-out"
            break
        if rec["balls"] >= cap:
            rec["outcome"] = "fled(cap)"
            rr._run_away(emu)
            break
        if not rr._safari_menu_pick(emu, 0, 1):   # BALL
            break
        rec["balls"] += 1
        caught = False
        for _ in range(160):
            rr._press10(emu, "a")
            emu.tick(10)
            if rr._stored_count(emu) > stored0:
                caught = True
                break
            if not ctx.state().in_battle:
                break
        if caught:
            ok = rr._catch_epilogue(emu, stored0 + 1)
            rec["outcome"] = "CAUGHT" if ok else "caught?"
            if ok:
                owned.add(sp)
            return rec
    if rec["outcome"] == "?":
        rec["outcome"] = "fled(enemy)" \
            if not ctx.state().in_battle else "stuck"
    return rec


def _safari_settle_u(ctx, targets, owned, slog):
    emu = ctx.emu
    for _ in range(300):
        st = ctx.state()
        if st.in_battle:
            _safari_battle_u(ctx, targets, owned, slog)
            continue
        if st.in_dialog:
            emu.close_dialog(max_presses=2)
            continue
        emu.tick(6)
        st2 = ctx.state()
        if (st2.x, st2.y) == (st.x, st.y) \
                and not st2.in_dialog and not st2.in_battle:
            return


def _safari_pace_u(ctx, pair, targets, owned, slog):
    emu = ctx.emu
    toggle = False
    while True:
        st = ctx.state()
        if st.map_id == rr.MAP_SAFARI_GATE:
            return "ejected"
        if st.map_id not in rr.SAFARI_AREAS:
            return "lost"
        if targets <= owned:
            return "done"
        if rr._safari_balls(emu) == 0:
            return "balls-out"
        if rr._safari_steps(emu) <= 12:
            return "steps-out"
        tgt = pair[1] if toggle else pair[0]
        toggle = not toggle
        cur = (st.x, st.y)
        if cur == tgt:
            continue
        for dname, (dx, dy) in rr._SZ_DIRS.items():
            if (cur[0] + dx, cur[1] + dy) == tgt:
                emu.press(dname)
                break
        else:
            return "off-pair"
        _safari_settle_u(ctx, targets, owned, slog)


def _safari_burn_out_u(ctx, targets, owned, slog):
    emu = ctx.emu
    for i in range(900):
        if ctx.state().map_id == rr.MAP_SAFARI_GATE:
            return True
        emu.press("left" if i % 2 == 0 else "right")
        _safari_settle_u(ctx, targets, owned, slog)
    return ctx.state().map_id == rr.MAP_SAFARI_GATE


def _safari_enter(ctx):
    """Pay at the gate and land on the Center spawn."""
    emu, ex = ctx.emu, ctx.ex
    if not rr._ride_battling(emu, ex, rr.MAP_SAFARI_GATE):
        return False
    money0 = ctx.state().money
    rr._walk_fighting(emu, ex, rr.MAP_SAFARI_GATE, (4, 2),
                      attempts=20)
    if (ctx.state().x, ctx.state().y) != (4, 2):
        return False
    emu.press("right")
    emu.tick(20)
    rr._press10(emu, "a")
    paid = False
    for _ in range(200):
        if rr._safari_balls(emu) == 30 and \
                rr._event_flag(emu, rr.EV_IN_SAFARI_ZONE):
            paid = True
            break
        if ctx.state().in_dialog:
            rr._press10(emu, "a")
        emu.tick(25)
    if not paid or ctx.state().money != money0 - 500:
        return False
    for _ in range(80):
        if not ctx.state().in_dialog:
            break
        rr._press10(emu, "a")
        emu.tick(30)
    for _ in range(10):
        if ctx.state().map_id == rr.MAP_SZ_CENTER:
            break
        emu.press("up")
        rr._explore_settle(emu)
        rr._safari_settle(emu, [])
    return ctx.state().map_id == rr.MAP_SZ_CENTER


def _safari_one_entry(ctx, area_id, targets, owned, slog):
    """One paid entry hunted in one area; ends back at the
    Fuchsia Pokecenter, healed. West rides the recorded
    route (S37); other areas ride _sz_nav (the hunt-entry
    pattern)."""
    emu, ex = ctx.emu, ctx.ex
    if not _safari_enter(ctx):
        return False
    pair = rr._safari_grass_pair(area_id)
    if pair is None:
        return False
    if area_id == rr.MAP_SZ_WEST:
        orig = rr._safari_settle
        rr._safari_settle = lambda e, seen: _safari_settle_u(
            ctx, targets, owned, slog)
        try:
            ok = rr._safari_replay_west_route(emu, ex, [])
        finally:
            rr._safari_settle = orig
        if ok:
            # walk the last hop west in the grass to the pair
            for _ in range(12):
                st = ctx.state()
                if (st.x, st.y) == pair[0] \
                        and st.map_id == rr.MAP_SZ_WEST:
                    break
                if st.map_id != rr.MAP_SZ_WEST:
                    break
                emu.press("left")
                _safari_settle_u(ctx, targets, owned, slog)
            if (ctx.state().x, ctx.state().y) == pair[0]:
                _safari_pace_u(ctx, pair, targets, owned, slog)
    else:
        if rr._sz_nav(emu, ex, area_id, pair[0], [],
                      budget_guard=10):
            # pace with the U policy (the hunt-entry loop)
            toggle = False
            while True:
                st = ctx.state()
                if st.map_id == rr.MAP_SAFARI_GATE:
                    break
                if st.map_id != area_id:
                    break
                if targets <= owned \
                        or rr._safari_balls(emu) == 0 \
                        or rr._safari_steps(emu) <= 12:
                    break
                tgt = pair[1] if toggle else pair[0]
                toggle = not toggle
                cur = (st.x, st.y)
                if cur == tgt:
                    continue
                for dname, (dx, dy) in rr._SZ_DIRS.items():
                    if (cur[0] + dx, cur[1] + dy) == tgt:
                        emu.press(dname)
                        break
                else:
                    break
                _safari_settle_u(ctx, targets, owned, slog)
    if ctx.state().map_id != rr.MAP_SAFARI_GATE:
        _safari_burn_out_u(ctx, targets, owned, slog)
    for _ in range(12):
        emu.press("down")
        rr._explore_settle(emu)
        rr._safari_settle(emu, [])
        if ctx.state().map_id != rr.MAP_SAFARI_GATE:
            break
    if ctx.state().map_id == rr.MAP_SAFARI_GATE:
        return False
    pc = ctx.world.map_id("FuchsiaPokecenter")
    if not rr._ride_battling(emu, ex, pc):
        return False
    heal_party(ctx, pc)
    return True


@quest("safari_hunt", "Safari Zone hunt for the named "
       "species: paid entries, unbaited throws only "
       "(Vlad's standing policy), recorded West route + "
       "live pairs, stop the moment every target is caught.",
       args=[Arg("species", "species_list", required=True,
                 help="one or more species names"),
             Arg("entries", "int", default=6,
                 help="max paid entries (1-6, 500 yen each)")])
def q_safari_hunt(ctx, args):
    emu, world = ctx.emu, ctx.world
    d = require_data()
    entries = args["entries"]
    if not 1 <= entries <= 6:
        ctx.refuse("args", f"entries must be 1-6, "
                   f"got {entries}")
    consts = []
    for name in args["species"]:
        const = d.species_const(name)
        if const is None:
            ctx.refuse("args", f"unknown species {name!r}")
        consts.append(const)
    for const in consts:
        if d.dex_owned(emu, const):
            ctx.refuse("owned", f"{const} is already owned - "
                       "the hunt stops at owned targets")
    area_ids = _safari_area_ids()
    area_chance = {}       # area map name -> {const: chance}
    for stem, area_id in area_ids.items():
        tab = d.wild.get(stem)
        if not tab:
            continue
        area_chance[stem] = {}
        for i, (_lvl, spc) in enumerate(tab["grass"]):
            if spc in consts:
                area_chance[stem][spc] = SLOT_CHANCES[i]
    huntable = {c for m in area_chance.values() for c in m}
    for const in consts:
        if const not in huntable:
            ctx.refuse("habitat", f"{const} does not appear "
                       "in any Safari Zone area table")
    money0 = ctx.state().money
    if money0 < 500 * entries:
        ctx.refuse("funds", f"money {money0} < {500 * entries} "
                   f"for {entries} entries at 500 each")
    here = world.map_name(ctx.state().map_id)
    if here not in ("FuchsiaCity", "FuchsiaPokecenter",
                    "SafariZoneGate"):
        if not party_has_move(emu, rr.MOVE_FLY):
            ctx.refuse("travel", "not in Fuchsia and no Fly "
                       "carrier in the party")
        if not fly(ctx, "FUCHSIA CITY"):
            ctx.refuse("travel", "Fly to FUCHSIA CITY failed")
    targets = {d.species_id(c) for c in consts}
    owned = {m.species_id for m in ctx.state().party}
    owned |= rr._all_box_species(emu)
    slog = []
    used = 0
    for n in range(1, entries + 1):
        if targets <= owned:
            break
        missing = [c for c in consts
                   if d.species_id(c) not in owned]
        best_area, best_score = None, -1.0
        for stem, m in area_chance.items():
            score = sum(m.get(c, 0.0) for c in missing)
            if score > best_score:
                best_area, best_score = stem, score
        used = n
        if not _safari_one_entry(ctx, area_ids[best_area],
                                 targets, owned, slog):
            raise QuestFailure(
                f"Safari entry {n} failed mechanically "
                f"({used - 1} entries completed); money "
                f"{money0} -> {ctx.state().money}")
        ctx.checkpoint(f"safari-e{n}")
    sightings = sum(1 for r in slog
                    if r["species"] in
                    {d.species[c]['name'] for c in consts})
    caught_names = [c for c in consts
                    if d.species_id(c) in owned]
    if targets <= owned:
        return (f"all targets caught in {used} entr"
                f"{'y' if used == 1 else 'ies'}: "
                f"{', '.join(caught_names)} "
                f"({sightings} sightings, money {money0} -> "
                f"{ctx.state().money})")
    raise QuestFailure(
        f"hunt spent its {used} entries: caught "
        f"{caught_names or 'nothing'} of "
        f"{[c for c in consts]} ({sightings} sightings, "
        f"money {money0} -> {ctx.state().money})")


# --- game_corner -----------------------------------------------

_GC_MENU2 = ["DRATINI", "SCYTHER", "PORYGON"]
_GC_MENU1 = ["ABRA", "CLEFAIRY", "NIDORINA"]
_GC_SPIN_CAP = 40000


def _buy_coins(ctx, min_coins):
    """Buy 50-coin lots at the Game Corner clerk (stance
    (5,8), face up; S36) until the case holds min_coins or
    the money runs out. Returns the coin balance."""
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    gc = world.map_id("GameCorner")
    while rr.player_coins(emu) < min_coins \
            and ctx.state().money >= 1000:
        rr._walk_fighting(emu, ex, gc, (5, 8), attempts=30,
                          flee_wild=True)
        if (ctx.state().x, ctx.state().y) != (5, 8):
            break
        rr._face(emu, "up")
        emu.tick(30)
        before_c, before_m = rr.player_coins(emu), \
            ctx.state().money
        for _ in range(120):
            if rr.player_coins(emu) >= before_c + 50:
                break
            if ctx.state().in_dialog:
                rr._press10(emu, "a")
            emu.tick(25)
        if rr.player_coins(emu) < before_c + 50 \
                or ctx.state().money != before_m - 1000:
            break
    return rr.player_coins(emu)


@quest("game_corner", "Grind the Game Corner slots and "
       "redeem a prize Pokemon (vendor 2: Dratini 2,800 / "
       "Scyther 5,500 / Porygon 9,999). Coin-Case cap aware; "
       "a single redemption per call, so the S36 bankroll "
       "trap does not apply.",
       args=[Arg("prize", "species", required=True)])
def q_game_corner(ctx, args):
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    d = require_data()
    const = d.species_const(args["prize"])
    if const is None or const not in d.ledger["prizes"]:
        ctx.refuse("prize", f"{args['prize']!r} is not a "
                   "Game Corner prize species")
    if const in _GC_MENU1:
        ctx.refuse("prize", f"{const} is a vendor-1 prize - "
                   "the ported redemption is calibrated on "
                   "vendor 2 only (Dratini/Scyther/Porygon); "
                   "vendor 1's stance is unproven")
    if const not in _GC_MENU2:
        ctx.refuse("prize", f"{const}: prize vendor unknown")
    if d.dex_owned(emu, const):
        ctx.refuse("owned", f"{const} is already owned")
    if not box_headroom(ctx):
        ctx.refuse("box", "party full and the active box is "
                   "full (20/20)")
    price = d.ledger["prizes"][const]["coins"]
    coin_case = d.items["COIN_CASE"]
    if rr._bag_count(emu, coin_case) == 0:
        if rr._pc_count(emu, coin_case) == 0:
            ctx.refuse("item", "no Coin Case in the bag or "
                       "PC item storage")
        pc = go_pc(ctx, fly_town="CELADON CITY")
        if rr.pc_withdraw_item(emu, ex, pc, coin_case, 1) < 1:
            ctx.refuse("item", "could not withdraw the "
                       "Coin Case from PC storage")
    if ctx.state().map_id != world.map_id("GameCorner"):
        if not fly(ctx, "CELADON CITY"):
            ctx.refuse("travel", "Fly to CELADON CITY failed")
        if not _go_map(ctx, world.map_id("GameCorner")):
            raise QuestFailure("could not reach the Game "
                               "Corner")
    gc = world.map_id("GameCorner")
    coins = rr.player_coins(emu)
    spins = 0
    spin_log = []
    while coins < price:
        if coins < 3:
            coins = _buy_coins(ctx, 50)
            if coins < 3:
                raise QuestFailure(
                    f"bust at {coins} coins with "
                    f"{ctx.state().money} yen - cannot "
                    "re-seed a 50-coin lot (1,000 yen)")
        idx = rr.lucky_slot_index(emu)
        if idx is None:
            raise QuestFailure("no lucky machine identified "
                               "in RAM this visit")
        stance, face = rr.slot_machine_stance(idx)
        if not rr.slots_open_session(emu, ex, gc, stance, face):
            raise QuestFailure("could not open a slot "
                               "session on the lucky machine")
        n = rr.slots_play_spins(
            emu, 2000, spin_log,
            should_stop=lambda: rr.player_coins(emu) >= price)
        spins += n
        rr.slots_close_session(emu)
        coins = rr.player_coins(emu)
        if spins >= _GC_SPIN_CAP:
            raise QuestFailure(
                f"spin cap {_GC_SPIN_CAP} reached at "
                f"{coins}/{price} coins (balance kept)")
    pr = world.map_id("GameCornerPrizeRoom")
    if not _go_map(ctx, pr):
        raise QuestFailure("could not reach the Prize Room")
    sp = d.species_id(const)
    specimens0 = rr._specimen_count(emu, sp)
    try:
        where = rr.redeem_prize_mon(
            emu, ex, pr, _GC_MENU2.index(const),
            d.species[const]["name"], price)
    except AssertionError as exc:
        raise QuestFailure(f"redemption failed: {exc}")
    if rr._specimen_count(emu, sp) <= specimens0:
        raise QuestFailure("redemption left no specimen "
                           "in RAM")
    return (f"{const} redeemed for {price} coins after "
            f"{spins} spins (filed: {where}); coins left "
            f"{rr.player_coins(emu)}")


# --- trade --------------------------------------------------------

_TRADES = [
    # id, give, get, map name, proven live (legs 2/5)
    ("mr_mime", "ABRA", "MR_MIME", "Route2TradeHouse", True),
    ("farfetchd", "SPEAROW", "FARFETCHD",
     "VermilionTradeHouse", True),
    ("jynx", "POLIWHIRL", "JYNX", "CeruleanTradeHouse", True),
    ("lickitung", "SLOWBRO", "LICKITUNG",
     "Route18Gate2F", True),
    ("nidorina", "NIDORINO", "NIDORINA",
     "Route11Gate2F", False),
    ("seel", "PONYTA", "SEEL", "CinnabarLabFossilRoom", False),
    ("electrode", "RAICHU", "ELECTRODE",
     "CinnabarLabTradeRoom", False),
    ("tangela", "VENONAT", "TANGELA",
     "CinnabarLabTradeRoom", False),
    ("nidoran_f", "NIDORAN_M", "NIDORAN_F",
     "UndergroundPathRoute5", False),
]


@quest("trade", "An in-game NPC trade via the live-cell "
       "trader driver. The give species is brought into the "
       "party first (from a box if needed).",
       args=[Arg("trade_id", "str", required=True,
                 help="trade id or received species: "
                 + ", ".join(t[0] for t in _TRADES))])
def q_trade(ctx, args):
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    d = require_data()
    want = str(args["trade_id"]).lower()
    row = next((t for t in _TRADES
                if t[0] == want
                or t[2] == (d.species_const(args["trade_id"])
                            or "")), None)
    if row is None:
        ctx.refuse("args", f"unknown trade {args['trade_id']!r} "
                   f"(want: {', '.join(t[0] for t in _TRADES)})")
    _tid, give_c, get_c, map_name, _proven = row
    give_id, get_id = d.species_id(give_c), d.species_id(get_c)
    if rr._specimen_count(emu, get_id) > 0:
        ctx.refuse("owned", f"a {get_c} specimen is already "
                   "held - the trade's effect would be "
                   "unverifiable")
    if rr._specimen_count(emu, give_id) == 0:
        ctx.refuse("give", f"no {give_c} specimen in party "
                   "or boxes to offer")
    if rr._party_index_of(emu, give_id) is None:
        pc = go_pc(ctx)
        if not ensure_species_in_party(ctx, give_id, pc):
            ctx.refuse("give", f"could not bring {give_c} "
                       "into the party")
    map_id = world.map_id(map_name)
    if not _go_map(ctx, map_id):
        raise QuestFailure(f"could not reach {map_name}")
    give0 = rr._specimen_count(emu, give_id)
    if not rr._do_trade_live(emu, ex, map_id, give_id, get_id,
                             world):
        raise QuestFailure(f"the {map_name} trade did not "
                           "complete")
    # specimen arithmetic is the ground truth (a pre-trade
    # specimen was refused above, so any specimen now is
    # the traded one)
    if rr._specimen_count(emu, get_id) < 1:
        raise QuestFailure("no received specimen in RAM "
                           "after the trade dialog")
    return (f"traded {give_c} -> {get_c} at {map_name} "
            f"(dex {rr._dex_owned_count(emu)})")


# --- evolve_stone -------------------------------------------------

@quest("evolve_stone", "Stone-evolve a species (the stone "
       "may come from the bag or PC item storage; the base "
       "may come from a box).",
       args=[Arg("species", "species", required=True),
             Arg("stone", "item", required=True,
                 help="e.g. THUNDER_STONE")])
def q_evolve_stone(ctx, args):
    emu, ex = ctx.emu, ctx.ex
    d = require_data()
    const = d.species_const(args["species"])
    stone = d.item_const(args["stone"])
    if const is None:
        ctx.refuse("args", f"unknown species "
                   f"{args['species']!r}")
    if stone is None:
        ctx.refuse("args", f"unknown item {args['stone']!r}")
    edge = next((e for e in d.species[const]["evos"]
                 if e["kind"] == "item" and e["param"] == stone),
                None)
    if edge is None:
        ctx.refuse("evolution", f"{const} does not evolve "
                   f"with {stone} (pokered evos_moves data)")
    to_const = edge["to"]
    from_id, to_id = d.species_id(const), d.species_id(to_const)
    item_id = d.items[stone]
    if rr._specimen_count(emu, from_id) == 0:
        ctx.refuse("species", f"no {const} specimen held")
    if rr._bag_count(emu, item_id) == 0 \
            and rr._pc_count(emu, item_id) == 0:
        ctx.refuse("stone", f"no {stone} in the bag or PC "
                   "item storage")
    pc = None
    if rr._bag_count(emu, item_id) == 0 \
            or rr._party_index_of(emu, from_id) is None:
        pc = go_pc(ctx)
    if rr._bag_count(emu, item_id) == 0:
        if rr.pc_withdraw_item(emu, ex, pc, item_id, 1) < 1:
            ctx.refuse("stone", f"could not withdraw "
                       f"{stone} from PC storage")
    if rr._party_index_of(emu, from_id) is None:
        if not ensure_species_in_party(ctx, from_id, pc):
            ctx.refuse("species", f"could not bring {const} "
                       "into the party")
    dex0 = rr._dex_owned_count(emu)
    if not rr.evolve_with_stone(emu, item_id, from_id, to_id):
        raise QuestFailure(f"stone evolution {const} -> "
                           f"{to_const} did not land")
    if rr._party_index_of(emu, to_id) is None:
        raise QuestFailure("evolved specimen not in the "
                           "party afterwards")
    return (f"{const} evolved into {to_const} with {stone} "
            f"(dex {dex0} -> {rr._dex_owned_count(emu)})")


# --- revive_fossil --------------------------------------------------

_FOSSILS = {"HELIX_FOSSIL": "OMANYTE",
            "DOME_FOSSIL": "KABUTO",
            "OLD_AMBER": "AERODACTYL"}


def _talk_fossil_scientist(ctx, lab_map):
    """probe168's approach: the scientist is sprite 1 in the
    Fossil Room; stand adjacent (re-reading his live cell),
    face him, talk."""
    emu, ex = ctx.emu, ctx.ex
    dirs = {"up": (0, -1), "down": (0, 1), "left": (-1, 0),
            "right": (1, 0)}
    face = {(0, -1): "up", (0, 1): "down", (-1, 0): "left",
            (1, 0): "right"}
    for _round in range(24):
        cell = rr._boulder_cell(emu, 1)
        st = ctx.state()
        if abs(st.x - cell[0]) + abs(st.y - cell[1]) == 1:
            delta = (cell[0] - st.x, cell[1] - st.y)
            emu.press(face[delta])
            emu.tick(15)
            rr._press10(emu, "a")
            emu.tick(45)
            if ctx.state().in_dialog:
                return True
            continue
        for (dx, dy) in dirs.values():
            stance = (cell[0] + dx, cell[1] + dy)
            rr._walk_fighting(emu, ex, lab_map, stance,
                              attempts=8)
            if (ctx.state().x, ctx.state().y) == stance:
                break
        emu.tick(10)
    return ctx.state().in_dialog


@quest("revive_fossil", "Revive a fossil at the Cinnabar "
       "Lab (Helix -> Omanyte, Dome -> Kabuto, Old Amber -> "
       "Aerodactyl). The fossil may start in PC storage.",
       args=[Arg("fossil", "item", required=True)])
def q_revive_fossil(ctx, args):
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    d = require_data()
    fossil = d.item_const(args["fossil"])
    if fossil is None or fossil not in _FOSSILS:
        ctx.refuse("args", f"{args['fossil']!r} is not a "
                   "revivable fossil (want: "
                   f"{', '.join(_FOSSILS)})")
    want_const = _FOSSILS[fossil]
    want_id = d.species_id(want_const)
    if rr._specimen_count(emu, want_id) > 0:
        ctx.refuse("owned", f"{want_const} is already held")
    item_id = d.items[fossil]
    if rr._bag_count(emu, item_id) == 0 \
            and rr._pc_count(emu, item_id) == 0:
        ctx.refuse("fossil", f"no {fossil} in the bag or PC "
                   "item storage")
    if not box_headroom(ctx):
        ctx.refuse("box", "party full and the active box is "
                   "full (20/20)")
    pc = go_pc(ctx, fly_town="CINNABAR ISLAND")
    if rr._bag_count(emu, item_id) == 0:
        if rr.pc_withdraw_item(emu, ex, pc, item_id, 1) < 1:
            ctx.refuse("fossil", f"could not withdraw "
                       f"{fossil} from PC storage")
    lab = world.map_id("CinnabarLabFossilRoom")
    if not _go_map(ctx, lab):
        raise QuestFailure("could not reach the Cinnabar "
                           "Lab Fossil Room")
    if not _talk_fossil_scientist(ctx, lab):
        raise QuestFailure("the fossil scientist's dialog "
                           "never opened")
    for _ in range(240):
        if rr._bag_count(emu, item_id) == 0:
            break
        if ctx.state().in_dialog:
            rr._press10(emu, "a")
        emu.tick(25)
    emu.wait_free(max_rounds=8, stable_needed=2)
    if rr._bag_count(emu, item_id) != 0:
        raise QuestFailure("the fossil was not handed over")
    island = world.map_id("CinnabarIsland")
    if not _go_map(ctx, island) or not _go_map(ctx, lab):
        raise QuestFailure("the leave-and-return step failed")
    if not _talk_fossil_scientist(ctx, lab):
        raise QuestFailure("the scientist's second dialog "
                           "never opened")
    specimens0 = rr._specimen_count(emu, want_id)
    for _ in range(240):
        if rr._specimen_count(emu, want_id) > specimens0:
            break
        if ctx.state().in_dialog:
            rr._press10(emu, "a")
        emu.tick(25)
    # the box/party write fires BEFORE the nickname prompt;
    # answer NO with B (an A here lands on the keyboard)
    for _ in range(30):
        if not ctx.state().in_dialog:
            break
        rr._press10(emu, "b")
        emu.tick(30)
    emu.wait_free(max_rounds=8, stable_needed=2)
    rr._close_menus(emu)
    if rr._specimen_count(emu, want_id) <= specimens0:
        raise QuestFailure(f"{want_const} was not received")
    return (f"{fossil} revived into {want_const} at the "
            f"Cinnabar Lab (dex {rr._dex_owned_count(emu)})")


# --- buy / sell -----------------------------------------------------

_MART_PREF = ["FuchsiaMart", "CeruleanMart", "CeladonMart4F",
              "IndigoPlateauLobby", "CinnabarMart",
              "SaffronMart", "LavenderMart", "VermilionMart",
              "ViridianMart", "PewterMart", "CeladonMart2F",
              "CeladonMart5F"]


def _resolve_mart(ctx, d, text):
    if text is None:
        return None
    norm = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    for map_name in d.marts:
        if re.sub(r"[^A-Z0-9]", "", map_name.upper()) == norm:
            return map_name
    return "?"


@quest("buy", "Buy an item at a mart that stocks it "
       "(stock + prices parsed from pokered). Travels by "
       "Fly + journey; verifies bag count and money in RAM.",
       args=[Arg("item", "item", required=True),
             Arg("qty", "int", required=True),
             Arg("mart", "str", default=None,
                 help="mart map name (default: nearest/"
                 "preferred stocking mart)")])
def q_buy(ctx, args):
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    d = require_data()
    const = d.item_const(args["item"])
    if const is None:
        ctx.refuse("args", f"unknown item {args['item']!r}")
    qty = args["qty"]
    if qty < 1:
        ctx.refuse("args", f"qty must be >= 1, got {qty}")
    price = d.prices.get(const)
    if price is None:
        ctx.refuse("price", f"{const} has no mart price "
                   "(TMs and key items are not mart goods)")
    stocking = [m for m, stock in d.marts.items()
                if const in stock]
    if not stocking:
        ctx.refuse("stock", f"no mart stocks {const}")
    mart = _resolve_mart(ctx, d, args["mart"]) \
        if args["mart"] else None
    if args["mart"] and (mart == "?" or mart not in stocking):
        ctx.refuse("mart", f"{args['mart']!r} does not stock "
                   f"{const} (stocking: {stocking})")
    if mart is None:
        here = world.map_name(ctx.state().map_id)
        if here in stocking:
            mart = here
        else:
            mart = next((m for m in _MART_PREF
                         if m in stocking), stocking[0])
    money = ctx.state().money
    if money < price * qty:
        ctx.refuse("funds", f"money {money} < {price * qty} "
                   f"for {qty} x {const} at {price}")
    item_id = d.items[const]
    have = rr._bag_count(emu, item_id)
    if have == 0 and len(rr._bag_items(emu)) >= 20:
        ctx.refuse("bag", "the bag is full (20 stacks) and "
                   f"{const} is not already in it")
    mart_id = world.map_id(mart)
    if ctx.state().map_id != mart_id:
        town = d.MART_TOWN[mart]
        here_town = world.map_name(ctx.state().map_id)
        if here_town != FLY_TOWNS[town]:
            if not party_has_move(emu, rr.MOVE_FLY):
                ctx.refuse("travel", f"not in {town} and no "
                           "Fly carrier in the party")
            if not fly(ctx, town):
                ctx.refuse("travel", f"Fly to {town} failed")
        if not rr._ride_battling(emu, ex, mart_id):
            raise QuestFailure(f"ride into {mart} failed")
    stock = d.marts[mart]
    stand, face = d.MART_STANCE.get(mart, ((2, 5), "left"))
    got = rr.buy_mart_item(emu, ex, mart_id,
                           stock.index(const), item_id,
                           have + qty, stand=stand, face=face)
    if got < have + qty:
        raise QuestFailure(f"bought {got - have} of {qty} "
                           f"x {const} at {mart}")
    return (f"bought {qty} x {const} at {mart} for "
            f"{price * qty} yen (bag now {got})")


@quest("sell", "Sell items at a mart clerk (Fuchsia by "
       "default - the proven clerk stance). Verifies the "
       "money delta against the pokered price (sell = half).",
       args=[Arg("item", "item", required=True),
             Arg("qty", "int", required=True),
             Arg("mart", "str", default=None,
                 help="FuchsiaMart (default) or SaffronMart")])
def q_sell(ctx, args):
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    d = require_data()
    const = d.item_const(args["item"])
    if const is None:
        ctx.refuse("args", f"unknown item {args['item']!r}")
    qty = args["qty"]
    if qty < 1:
        ctx.refuse("args", f"qty must be >= 1, got {qty}")
    price = d.prices.get(const)
    if price is None:
        ctx.refuse("price", f"{const} has no price - it "
                   "cannot be sold (key items included)")
    item_id = d.items[const]
    have = rr._bag_count(emu, item_id)
    if have < qty:
        ctx.refuse("bag", f"only {have} x {const} in the "
                   f"bag, cannot sell {qty}")
    mart = _resolve_mart(ctx, d, args["mart"]) \
        if args["mart"] else "FuchsiaMart"
    if mart not in ("FuchsiaMart", "SaffronMart"):
        ctx.refuse("mart", "the sell clerk stance is proven "
                   "at FuchsiaMart and SaffronMart only")
    mart_id = world.map_id(mart)
    if ctx.state().map_id != mart_id:
        town = d.MART_TOWN[mart]
        if not party_has_move(emu, rr.MOVE_FLY):
            ctx.refuse("travel", f"not in {town} and no Fly "
                       "carrier in the party")
        if not fly(ctx, town):
            ctx.refuse("travel", f"Fly to {town} failed")
        if not rr._ride_battling(emu, ex, mart_id):
            raise QuestFailure(f"ride into {mart} failed")
    if not rr._mart_clerk_menu(emu, ex, mart_id):
        raise QuestFailure(f"the {mart} clerk menu never "
                           "opened")
    money0 = ctx.state().money
    delta = rr._mart_sell_qty(emu, item_id, qty)
    rr._close_menus(emu)
    rr._drain_dialog(emu)
    want_delta = qty * (price // 2)
    if delta != want_delta \
            or rr._bag_count(emu, item_id) != have - qty:
        raise QuestFailure(
            f"sell mismatch: money delta {delta} (want "
            f"{want_delta}), bag {have} -> "
            f"{rr._bag_count(emu, item_id)}")
    return (f"sold {qty} x {const} at {mart} for {delta} "
            f"yen (money {money0} -> {ctx.state().money})")


# --- box_switch -----------------------------------------------------

@quest("box_switch", "Change the active PC box (post-HoF "
       "text-verified PC flow; the game saves on a switch).",
       args=[Arg("box", "int", required=True,
                 help="box number 1-12")])
def q_box_switch(ctx, args):
    emu, ex = ctx.emu, ctx.ex
    n = args["box"]
    if not 1 <= n <= 12:
        ctx.refuse("args", f"box must be 1-12, got {n}")
    cur = rr._current_box_num(emu)
    if cur == n:
        return f"already on box {n}"
    pc = go_pc(ctx)
    if not rr.pc_change_box(emu, ex, pc, n):
        raise QuestFailure(f"CHANGE BOX {cur} -> {n} did "
                           "not land")
    if rr._current_box_num(emu) != n:
        raise QuestFailure("RAM box number did not change")
    return f"active box {cur} -> {n}"


# ------------------------------------------------------------ CLI

def _catalogue_text() -> str:
    lines = ["Side quests (python3 quests.py --state S "
             "<quest> [args]):", ""]
    for q in QUESTS.values():
        lines.append(f"{q.name}: {q.summary}")
        for a in q.args:
            req = "required" if a.required else \
                f"default {a.default!r}"
            lines.append(f"    {a.name} ({a.kind}, {req})"
                         + (f" - {a.help}" if a.help else ""))
        lines.append("")
    return "\n".join(lines)


def _parse_cli_args(q: Quest, tokens: list) -> dict:
    """Positional tokens fill the schema in order (a
    species_list swallows the remaining bare tokens);
    key=value tokens set named args."""
    raw = {}
    specs = list(q.args)
    pos = []
    for tok in tokens:
        if "=" in tok and not tok.startswith("="):
            k, v = tok.split("=", 1)
            raw[k] = v
        else:
            pos.append(tok)
    i = 0
    for si, spec in enumerate(specs):
        if spec.name in raw:
            continue
        if i >= len(pos):
            break
        if spec.kind == "species_list":
            rest = specs[si + 1:]
            take = len(pos) - i - sum(
                1 for s in rest if s.name not in raw)
            raw[spec.name] = pos[i:i + max(take, 1)]
            i += max(take, 1)
        else:
            raw[spec.name] = pos[i]
            i += 1
    return raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run one side quest against a save state.")
    ap.add_argument("--rom", help="Pokemon Red ROM (or "
                    "POKERED_ROM)")
    ap.add_argument("--state", help="input save state")
    ap.add_argument("--out", help="output save state "
                    "(default: --state)")
    ap.add_argument("--list", action="store_true",
                    help="print the quest catalogue and exit")
    ap.add_argument("quest", nargs="?", help="quest name")
    ap.add_argument("qargs", nargs="*", help="quest arguments")
    args = ap.parse_args(argv)
    if args.list or not args.quest:
        print(_catalogue_text())
        return 0 if args.list else 2
    if args.quest not in QUESTS:
        print(f"unknown quest {args.quest!r}\n")
        print(_catalogue_text())
        return 2
    if not args.state:
        print("--state is required", file=sys.stderr)
        return 2
    rom = args.rom or os.environ.get("POKERED_ROM") or str(
        Path.home() / "workspace" / "pokemon-play" / "pokered.gbc")
    from agent.pyboy_emu import PyBoyEmulator
    from agent.executor import Executor, SmartFightPolicy
    from agent.planner import ChecklistPlanner
    from agent.memory import EpisodicMemory
    world = default_world()
    emu = PyBoyEmulator(rom)
    emu.load_state(args.state)
    emu.tick(30)
    ex = Executor(emu, world, ChecklistPlanner(),
                  EpisodicMemory(),
                  battle_policy=SmartFightPolicy())
    raw = _parse_cli_args(QUESTS[args.quest], args.qargs)
    out = args.out or args.state
    ck_dir = Path(out).resolve().parent / "quest-checkpoints"
    result = run_quest(emu, ex, args.quest, raw,
                       log=lambda m: print(m, flush=True),
                       checkpoint_dir=ck_dir)
    print(json.dumps({"quest": result.quest,
                      "outcome": result.outcome,
                      "detail": result.detail,
                      "pre": result.pre, "post": result.post},
                     indent=1))
    if result.outcome in ("ok", "failed") \
            and not QUESTS[args.quest].read_only:
        emu.save_state(out)
        print(f"state saved -> {out}")
    return {"ok": 0, "failed": 1, "refused": 2}[result.outcome]


if __name__ == "__main__":
    sys.exit(main())
