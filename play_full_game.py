#!/usr/bin/env python3
"""play_full_game.py - the whole game, one process, from power-on.

    python3 play_full_game.py --rom /path/to/pokered.gbc
    python3 play_full_game.py --rom ... --web 8000        # + live viewer
    python3 play_full_game.py --out run --resume          # continue a run
    python3 play_full_game.py --list-phases               # the phase table

This is a thin orchestration layer over run_real.py: every phase calls
the SAME ported leg functions that produced the completed save line
(docs/supervisor-training.md sections 1-28, chain head
probe167/post-league.state). The runner adds what the probes did by
hand: sequencing, a save-state checkpoint after every phase
(run/checkpoints/NN-phase.state), a progress log (run/progress.log)
with RAM-verified evidence per phase, and bounded retries - a failed
phase is retried from its start checkpoint at most --max-retries
times, then the runner STOPS and reports. It never skips a phase and
never thrashes forever.

Decision branches are the completed save line's, because the ported
functions encode them: Squirtle starter, Fire Stone -> Arcanine,
Hitmonlee at the Dojo, Surf + Strength on Lapras, the Route 16
Snorlax CAUGHT (Master Ball never thrown), the Mansion grinds, and
the League team (Blastoise / Lapras / Snorlax / Arcanine + bench).

Honesty: each phase is individually proven and most segments were
cold-replayed from checkpoints during the storyline build, but no
single-process power-on-to-credits soak has been run yet - that is
the remaining hardening step, and exactly what this runner is for.
The two hand-written journey glues (fly-journey, snorlax-journey)
are the least-replayed segments; both are bounded and fail loudly.

The optional web viewer (web_viewer.py, stdlib only) serves the live
frame + state JSON on localhost; --host 0.0.0.0 opts into LAN access
(e.g. watching an Orin run from another machine on your network).
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import io
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Importing run_real first: it sets the same SDL dummy drivers and
# carries every ported phase function this runner sequences.
import run_real as rr
from agent.executor import Executor, SmartFightPolicy
from agent.memory import EpisodicMemory
from agent.planner import ChecklistPlanner, LLMPlanner
from agent.pyboy_emu import PyBoyEmulator
from agent.world import default_world

SPECIES_SQUIRTLE = 0xB1   # pokered pokemon_constants.asm


# ---------------------------------------------------------------- ctx

class Ctx:
    """Everything a phase needs: emulator, executor, logging, states."""

    def __init__(self, rom: str, out_dir, supervisor: str = "off",
                 player: str = "VLAD", rival: str = "GARY"):
        self.rom = rom
        self.out_dir = Path(out_dir)
        self.supervisor = supervisor
        self.player = player
        self.rival = rival
        self.world = default_world()
        self.emu: Optional[PyBoyEmulator] = None
        self.ex: Optional[Executor] = None
        self.memory: Optional[EpisodicMemory] = None
        self.recent: list[str] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.out_dir / "progress.log"

    # -- lifecycle ---------------------------------------------------
    def _build_executor(self) -> None:
        # Mirrors run_real.main()'s construction exactly.
        if self.supervisor == "adapter":
            planner = LLMPlanner(
                endpoint="inline", model="inline:adapter",
                adapter=str(Path(__file__).parent / "training_output"
                            / "qwen_supervisor_v2" / "final"),
                base_model="/home/vlad/models/Qwen3.5-4B")
        elif self.supervisor == "stock":
            planner = LLMPlanner(
                endpoint="inline", model="inline:stock", adapter=None,
                base_model="/home/vlad/models/Qwen3.5-4B")
        else:
            planner = ChecklistPlanner()
        self.memory = EpisodicMemory()
        self.ex = Executor(self.emu, self.world, planner, self.memory,
                           battle_policy=SmartFightPolicy())

    def boot_fresh(self) -> None:
        self.emu = PyBoyEmulator(self.rom)
        self._build_executor()

    def reboot(self) -> None:
        """Fresh power-on (phase-0 retries have no checkpoint)."""
        self.boot_fresh()

    def load_state(self, path) -> None:
        if self.emu is None:
            self.emu = PyBoyEmulator(self.rom)
            self._build_executor()
        self.emu.load_state(str(path))
        self.emu.tick(30)

    def save_state(self, path) -> None:
        self.emu.save_state(str(path))

    # -- reads / logging ----------------------------------------------
    def state(self):
        return self.emu.read_state()

    def note(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.recent.append(line)
        self.recent = self.recent[-40:]
        with open(self.progress_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ------------------------------------------------------- verify tools

def _badges(ctx: Ctx, *names: str):
    st = ctx.state()
    missing = [n for n in names if n not in st.badges]
    if missing:
        return False, f"badges {sorted(st.badges)} missing {missing}"
    return True, f"badges={sorted(st.badges)}"


def _bag_has(ctx: Ctx, item: int, label: str):
    n = rr._bag_count(ctx.emu, item)
    return (n >= 1), f"{label} x{n}"


def _party_species(ctx: Ctx, species: int):
    st = ctx.state()
    for m in st.party:
        if m.species_id == species:
            return m
    return None


def _map_is(ctx: Ctx, *names: str):
    st = ctx.state()
    ids = {ctx.world.map_id(n) for n in names}
    here = ctx.world.map_name(st.map_id)
    return (st.map_id in ids), f"map={here} ({st.x},{st.y})"


# ------------------------------------------------------ opening loop

def _opening_loop(ctx: Ctx, want_badges: set, max_rounds: int = 800) -> bool:
    """run_real.main()'s executor-round loop, parameterised by the
    badge set to stop at. Scene hooks fire exactly as in main():
    parcel handover in Oak's Lab, Brock in Pewter Gym, the fossil on
    Mt. Moon B2F, Misty in Cerulean Gym."""
    emu, ex, memory = ctx.emu, ctx.ex, ctx.memory
    world = ctx.world
    gym = world.map_id("CeruleanGym")
    flags = {"handover": False, "brock": False,
             "fossil": False, "misty": False}
    for round_no in range(max_rounds):
        st = emu.read_state()
        if want_badges <= set(st.badges):
            return True
        result = ex.run(max_goals=1)
        st = emu.read_state()
        ctx.note(f"  round {round_no}: {result.status} at "
                 f"{world.map_name(st.map_id)} ({st.x},{st.y}) "
                 f"badges={sorted(st.badges)}")
        if want_badges <= set(st.badges):
            return True
        delivered = any(
            e.get("goal") == "goto:40:None"
            and "Deliver the parcel" in e.get("reason", "")
            and "error" not in e and "problems" not in e
            for e in result.log)
        if delivered and not flags["handover"] \
                and st.map_id == rr.MAP_OAKS_LAB:
            flags["handover"] = bool(rr.deliver_parcel_scene(emu, ex))
            ctx.note(f"  parcel handover scene: {flags['handover']}")
            st = emu.read_state()
        if not flags["brock"] and st.map_id == rr.MAP_PEWTER_GYM \
                and "Boulder" not in st.badges:
            try:
                flags["brock"] = bool(rr.fight_brock(emu, ex))
            except Exception as exc:  # keep the loop's shape; verify rules
                ctx.note(f"  fight_brock raised {exc!r}")
                flags["brock"] = False
            st = emu.read_state()
        if not flags["fossil"] and st.map_id == rr.MAP_MT_MOON_B2F:
            try:
                flags["fossil"] = bool(rr.take_fossil_scene(emu, ex))
            except Exception as exc:
                ctx.note(f"  take_fossil_scene raised {exc!r}")
                flags["fossil"] = False
            st = emu.read_state()
        if not flags["misty"] and st.map_id == gym \
                and "Cascade" not in st.badges:
            try:
                flags["misty"] = bool(rr.fight_misty(emu, ex))
            except Exception as exc:
                ctx.note(f"  fight_misty raised {exc!r}")
                flags["misty"] = False
            st = emu.read_state()
        if want_badges <= set(st.badges):
            return True
        if result.status in ("stuck", "blocked"):
            emu.wait_free()
            if result.log:
                goal_key = result.log[-1].get("goal")
                if goal_key:
                    memory.forgive(goal_key)
    return want_badges <= set(emu.read_state().badges)


# ----------------------------------------------------- journey glues
# The two segments the storyline probes drove from intermediate
# checkpoints rather than one ported function. Assembled here from
# ported primitives and the legs' documented anchor cells
# (docs/supervisor-training.md sections 23-24). Bounded throughout:
# any anchor mismatch returns False and the runner stops loudly.

def _settle(ctx: Ctx, ticks: int = 30) -> None:
    ctx.emu.tick(ticks)
    for _ in range(6):
        if not ctx.emu.read_state().in_battle:
            break
        ctx.emu.wait_free()
        ctx.emu.tick(30)


def _step(ctx: Ctx, map_id: int, cell, expect_maps=None) -> bool:
    ok = rr._step_to_live(ctx.emu, ctx.ex, map_id, cell[0], cell[1],
                          max_steps=400)
    _settle(ctx)
    if expect_maps is not None:
        return ctx.state().map_id in expect_maps
    return bool(ok)


def glue_fuchsia_to_cerulean(ctx: Ctx) -> bool:
    """Fly leg, part 1: Fuchsia -> Seafoam ladder transit -> Cinnabar
    -> Pallet -> Mt. Moon -> Cerulean PC (the section-24 transit;
    seafoam_crossing cannot be re-run - its boulder checks assume a
    pre-cascade cave)."""
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    fuchsia = world.map_id("FuchsiaCity")
    fpc = world.map_id("FuchsiaPokecenter")
    r19 = world.map_id("Route19")
    r20 = world.map_id("Route20")
    sf1 = world.map_id("SeafoamIslands1F")
    b1f = world.map_id("SeafoamIslandsB1F")
    b2f = world.map_id("SeafoamIslandsB2F")
    b3f = world.map_id("SeafoamIslandsB3F")
    b4f = world.map_id("SeafoamIslandsB4F")
    if ctx.state().map_id == fpc:
        if not rr._hop_ride(emu, ex, ["FuchsiaCity"]):
            return False
    if ctx.state().map_id != fuchsia:
        return False
    if not rr.surf_to(emu, ex, r19):
        return False
    if not rr.surf_to(emu, ex, r20):
        return False
    # Ashore on the north plateau, in the west door (as the crossing's
    # first leg does it).
    if not rr._surf_walk_to(emu, ex, r20, (48, 5)):
        return False
    _settle(ctx)
    if ctx.state().map_id != sf1:
        return False
    # Ladder transit, section-24 anchors.
    if not _step(ctx, sf1, (7, 5), {b1f}):
        return False
    if not _step(ctx, b1f, (4, 2), {b2f}):
        return False
    # Step off the landing and back on to ride the ladder back up.
    rr._step_to_live(emu, ex, b2f, 5, 4, max_steps=60)
    _settle(ctx)
    if not _step(ctx, b2f, (5, 3), {b1f}):
        return False
    if not _step(ctx, b1f, (13, 7), {b2f}):
        return False
    if not _step(ctx, b2f, (5, 13), {b3f}):
        return False
    if not _step(ctx, b3f, (15, 7), {b3f}):
        return False
    if not rr._surf_board(emu, "down"):
        return False
    _settle(ctx, 120)
    if ctx.state().map_id != b4f:
        return False
    if not _step(ctx, b4f, (23, 9), {b4f}):
        return False
    if not _step(ctx, b4f, (25, 14), {b2f}):
        return False
    if not _step(ctx, b2f, (25, 11), {b1f}):
        return False
    if not _step(ctx, b1f, (23, 15), {sf1}):
        return False
    if not _step(ctx, sf1, (26, 17), {r20}):
        # Door cell reached but warp not taken: press out east.
        for _ in range(6):
            if ctx.state().map_id == r20:
                break
            emu.press("right")
            _settle(ctx)
    if ctx.state().map_id != r20:
        return False
    if not rr.surf_to(emu, ex, world.map_id("CinnabarIsland")):
        return False
    if not rr.surf_to(emu, ex, world.map_id("PalletTown")):
        return False
    hops = ["Route1", "ViridianCity", "Route2", "ViridianForest",
            "PewterCity", "Route3", "MtMoon1F", "MtMoonB1F",
            "MtMoonB2F", "Route4", "CeruleanCity", "CeruleanPokecenter"]
    if not rr._hop_ride(emu, ex, hops):
        return False
    rr.heal_at_pokecenter(emu, ex, world.map_id("CeruleanPokecenter"))
    return ctx.state().map_id == world.map_id("CeruleanPokecenter")


def glue_cerulean_to_snorlax(ctx: Ctx) -> bool:
    """Fly leg, part 2: Cerulean -> R9/R10 -> Rock Tunnel ->
    Lavender -> Underground Path -> Celadon -> R16 stance (27, 10),
    where snorlax_catch takes over."""
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    cerpc = world.map_id("CeruleanPokecenter")
    r10pc = world.map_id("Route10Pokecenter")
    r16 = world.map_id("Route16")
    if ctx.state().map_id == cerpc:
        if not rr._hop_ride(emu, ex, ["CeruleanCity"]):
            return False
    if not rr._hop_ride(emu, ex, ["Route9", "Route10"]):
        return False
    if not rr._hop_ride(emu, ex, ["Route10Pokecenter"]):
        return False
    rr.heal_at_pokecenter(emu, ex, r10pc)
    if not rr.traverse_rock_tunnel(emu, ex):
        return False
    hops = ["Route8", "UndergroundPathRoute8",
            "UndergroundPathWestEast", "UndergroundPathRoute7",
            "Route7", "CeladonPokecenter"]
    if not rr._hop_ride(emu, ex, hops):
        return False
    rr.heal_at_pokecenter(emu, ex, world.map_id("CeladonPokecenter"))
    if not rr._hop_ride(emu, ex, ["Route16Gate1F", "Route16"]):
        if not rr._hop_ride(emu, ex, ["Route16"]):
            return False
    if not _step(ctx, r16, (27, 10), {r16}):
        return False
    st = ctx.state()
    return (st.x, st.y) == (27, 10)


def glue_teach_fly(ctx: Ctx) -> bool:
    """Fly leg, part 3 (probe161's teachfly segment): out of the Fly
    House, through the R16 Cut tree, ride to Celadon PC, teach Fly
    to the boxed Pidgey, fly to Fuchsia."""
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    house = world.map_id("Route16FlyHouse")
    r16 = world.map_id("Route16")
    celpc = world.map_id("CeladonPokecenter")
    celadon = world.map_id("CeladonCity")
    if ctx.state().map_id != house:
        return False
    if not rr._step_to_live(emu, ex, house, 2, 6, max_steps=120):
        return False
    for _ in range(8):
        if ctx.state().map_id == r16:
            break
        emu.press("down")
    _settle(ctx)
    if ctx.state().map_id != r16:
        return False
    if not rr._step_to_live(emu, ex, r16, 27, 10, max_steps=300):
        return False
    if not rr._step_to_live(emu, ex, r16, 34, 9, max_steps=300):
        return False
    if not ex._cut_tree(r16, (34, 9)):
        return False
    if not rr._ride_battling(emu, ex, celpc):
        return False
    if not rr.teach_fly(emu, ex, celpc):
        return False
    st = ctx.state()
    if st.map_id == celpc:
        if not rr._step_to_live(emu, ex, celpc, 4, 6, max_steps=120):
            return False
        for _ in range(8):
            if ctx.state().map_id == celadon:
                break
            emu.press("down")
        _settle(ctx)
    return bool(rr.fly_to(emu, ex, "FUCHSIA CITY"))


# --------------------------------------------------------- phases

@dataclass
class Phase:
    name: str
    title: str
    run: Callable[[Ctx], object]
    verify: Optional[Callable[[Ctx], tuple]] = None


def _v_party_move(ctx: Ctx, species: int, move: int, label: str):
    m = _party_species(ctx, species)
    if m is None:
        return False, f"{label}: species 0x{species:02x} not in party"
    return (move in m.moves), f"{label}: moves={m.moves}"


def _v_party_level(ctx: Ctx, species: int, min_level: int, label: str):
    m = _party_species(ctx, species)
    if m is None:
        return False, f"{label}: species 0x{species:02x} not in party"
    return (m.level >= min_level), f"{label}: Lv{m.level}"


def _v_owned(ctx: Ctx, species: int, label: str):
    if _party_species(ctx, species) is not None:
        return True, f"{label}: in party"
    try:
        boxed = rr._box_species(ctx.emu)
    except Exception as exc:
        return False, f"{label}: box read failed {exc!r}"
    return (species in boxed), f"{label}: box={len(boxed)} species"


def _ph_intro(ctx: Ctx):
    rr.boot_and_intro(ctx.emu, ctx.player, ctx.rival)
    return True


def _v_intro(ctx: Ctx):
    st = ctx.state()
    return (st.map_id == rr.MAP_REDS_HOUSE_2F), \
        f"map={ctx.world.map_name(st.map_id)}"


def _ph_starter(ctx: Ctx):
    rr.trigger_oak_intercept(ctx.emu, ctx.ex)
    rr.pick_starter(ctx.emu, ctx.ex, "squirtle")
    ctx.memory.record_visit(rr.MAP_OAKS_LAB)
    return True


def _v_starter(ctx: Ctx):
    st = ctx.state()
    if not st.party:
        return False, "party empty"
    first = st.party[0]
    return (first.species_id == SPECIES_SQUIRTLE), \
        f"starter species=0x{first.species_id:02x} Lv{first.level}"


def _ph_vermilion(ctx: Ctx):
    """Executor rounds until one ends in Cerulean with the Cascade
    Badge (run_real.main's eastward trigger), then the ported leg."""
    emu, ex, world = ctx.emu, ctx.ex, ctx.world
    cer = world.map_id("CeruleanCity")
    cerpc = world.map_id("CeruleanPokecenter")
    for _ in range(60):
        st = emu.read_state()
        if "Cascade" in st.badges and st.map_id in (cer, cerpc):
            if rr.eastward_from_cerulean(emu, ex):
                break
        result = ex.run(max_goals=1)
        st = emu.read_state()
        if "Cascade" in st.badges and st.map_id in (cer, cerpc):
            if rr.eastward_from_cerulean(emu, ex):
                break
        if result.status in ("stuck", "blocked"):
            emu.wait_free()
    else:
        return False
    return rr.vermilion_leg(emu, ex)


def _ph_hideout(ctx: Ctx):
    if not rr.rocket_hideout(ctx.emu, ctx.ex):
        return False
    return rr.beat_giovanni_hq(ctx.emu, ctx.ex)


def _ph_fuchsia(ctx: Ctx):
    emu, ex = ctx.emu, ctx.ex
    if not rr.fuchsia_road(emu, ex):
        return False
    if not rr.safari_zone_hms(emu, ex):
        return False
    if not rr.give_teeth_get_strength(emu, ex):
        return False
    return rr.beat_koga(emu, ex)


def _ph_lapras_surf(ctx: Ctx):
    """The probe155 replay call, verbatim: make bag room (TM24/29/46
    to the PC), deposit Nidoran-M, withdraw Lapras, teach HM03."""
    emu, ex = ctx.emu, ctx.ex
    for tm in (rr.ITEM_TM24, rr.ITEM_TM29, rr.ITEM_TM46):
        if rr._bag_count(emu, tm):
            rr.deposit_item(emu, ex, rr.MAP_FUCHSIA_PC, tm)
    return rr.teach_hm_to_boxed(
        emu, ex, rr.MAP_FUCHSIA_PC, rr.SPECIES_NIDORAN_M,
        rr.SPECIES_LAPRAS, rr.ITEM_HM03_SURF, rr.MOVE_SURF,
        rr._heal_fuchsia_pc)


def _ph_safari_hunt(ctx: Ctx):
    return rr.safari_rare_hunt(ctx.emu, ctx.ex,
                                ["east", "north", "east", "north"])


def _ph_mansion_blaine(ctx: Ctx):
    if not rr.mansion_secret_key(ctx.emu, ctx.ex):
        return False
    return rr.beat_blaine(ctx.emu, ctx.ex)


def _v_super_rod(ctx: Ctx):
    ok1, e1 = _bag_has(ctx, rr.ITEM_SUPER_ROD, "Super Rod")
    ok2, e2 = _bag_has(ctx, rr.ITEM_BIKE_VOUCHER, "Bike Voucher")
    return (ok1 and ok2), f"{e1}; {e2}"


def _ph_dratini(ctx: Ctx):
    ok, log = rr.safari_dratini(ctx.emu, ctx.ex)
    for line in log[-6:]:
        ctx.note(f"  dratini: {line}")
    return ok


def _ph_snorlax(ctx: Ctx):
    res = rr.snorlax_catch(ctx.emu, ctx.ex)
    ctx.note(f"  snorlax_catch -> {res}")
    return res == "caught"


def _ph_prep_grind(ctx: Ctx):
    spend = rr.route22_prep(ctx.emu, ctx.ex)
    if spend is None or spend is False:
        return False
    ctx.note(f"  route22_prep spend={spend}")
    rr.mansion_grind(ctx.emu, ctx.ex)
    return True


def _v_prep_grind(ctx: Ctx):
    ok1, e1 = _v_party_level(ctx, rr.SPECIES_BLASTOISE, 66, "Blastoise")
    ok2, e2 = _v_party_level(ctx, rr.SPECIES_ARCANINE, 48, "Arcanine")
    return (ok1 and ok2), f"{e1}; {e2}"


def _v_rival2(ctx: Ctx):
    return bool(rr._event_flag(ctx.emu, rr.EV_BEAT_RIVAL2_R22)), \
        "EV_BEAT_RIVAL2_R22"


def _ph_victory_road(ctx: Ctx):
    if not rr.route23_passage(ctx.emu, ctx.ex):
        return False
    return rr.victory_road(ctx.emu, ctx.ex)


def _v_league_grind(ctx: Ctx):
    parts = [
        _v_party_level(ctx, rr.SPECIES_LAPRAS, 50, "Lapras"),
        _v_party_level(ctx, rr.SPECIES_SNORLAX, 48, "Snorlax"),
        _v_party_level(ctx, rr.SPECIES_ARCANINE, 52, "Arcanine"),
    ]
    return all(p[0] for p in parts), "; ".join(p[1] for p in parts)


def build_phases() -> list:
    """The phase table: storyline order (docs sections 1-28), each
    phase a ported run_real function or a documented glue above."""
    B = lambda *n: (lambda ctx: _badges(ctx, *n))  # noqa: E731
    M = lambda *n: (lambda ctx: _map_is(ctx, *n))   # noqa: E731
    return [
        Phase("intro", "Boot, Oak intro, naming (VLAD/GARY)",
              _ph_intro, _v_intro),
        Phase("starter", "Oak intercept, Squirtle, first rival fight",
              _ph_starter, _v_starter),
        Phase("boulder", "Parcel, Pokedex, Viridian Forest, Brock",
              lambda ctx: _opening_loop(ctx, {"Boulder"}),
              B("Boulder")),
        Phase("cascade", "Mt. Moon + Helix Fossil, Misty",
              lambda ctx: _opening_loop(ctx, {"Boulder", "Cascade"}),
              B("Cascade")),
        Phase("vermilion", "S.S. Anne, Cut (Oddish), Diglett, Surge",
              _ph_vermilion, B("Thunder")),
        Phase("rock-tunnel", "Flash (Voltorb), Rock Tunnel, Lavender",
              lambda ctx: rr.rock_tunnel_leg(ctx.emu, ctx.ex),
              M("LavenderTown")),
        Phase("celadon", "Growlithe, Fire Stone -> Arcanine, Erika",
              lambda ctx: rr.celadon_leg(ctx.emu, ctx.ex),
              B("Rainbow")),
        Phase("hideout", "Rocket Hideout, Giovanni HQ, Silph Scope",
              _ph_hideout,
              lambda ctx: _bag_has(ctx, rr.ITEM_SILPH_SCOPE,
                                   "Silph Scope")),
        Phase("tower", "Pokemon Tower, Mr. Fuji, Poke Flute",
              lambda ctx: rr.pokemon_tower(ctx.emu, ctx.ex),
              lambda ctx: _bag_has(ctx, rr.ITEM_POKE_FLUTE,
                                   "Poke Flute")),
        Phase("gates", "Saffron gate guards (drinks)",
              lambda ctx: rr.saffron_gates(ctx.emu, ctx.ex),
              M("SaffronCity")),
        Phase("silph", "Silph Co.: Card Key, rival, Lapras, Giovanni, "
              "Master Ball",
              lambda ctx: rr.silph_co(ctx.emu, ctx.ex),
              lambda ctx: _bag_has(ctx, rr.ITEM_MASTER_BALL,
                                   "Master Ball")),
        Phase("sabrina", "Sabrina (psychic answer: Arcanine's Bite)",
              lambda ctx: rr.beat_sabrina(ctx.emu, ctx.ex),
              B("Marsh")),
        Phase("dojo", "Fighting Dojo - Hitmonlee (Vlad's pick)",
              lambda ctx: rr.fighting_dojo(ctx.emu, ctx.ex,
                                           "hitmonlee"),
              lambda ctx: _v_owned(ctx, rr.SPECIES_HITMONLEE,
                                   "Hitmonlee")),
        Phase("fuchsia", "Cycling Road south, Safari HMs, teeth, Koga",
              _ph_fuchsia, B("Soul")),
        Phase("lapras-surf", "Teach HM03 Surf to Lapras (boxed swap)",
              _ph_lapras_surf,
              lambda ctx: _v_party_move(ctx, rr.SPECIES_LAPRAS,
                                        rr.MOVE_SURF, "Lapras Surf")),
        Phase("safari-hunt", "Safari rare hunt (Paras..Kangaskhan)",
              _ph_safari_hunt, M("FuchsiaPokecenter", "FuchsiaCity")),
        Phase("teach-strength", "Teach HM04 Strength to Lapras",
              lambda ctx: rr.teach_strength(ctx.emu, ctx.ex),
              lambda ctx: _v_party_move(ctx, rr.SPECIES_LAPRAS,
                                        rr.MOVE_STRENGTH,
                                        "Lapras Strength")),
        Phase("seafoam", "Seafoam boulder cascade to Cinnabar",
              lambda ctx: rr.seafoam_crossing(ctx.emu, ctx.ex),
              M("CinnabarIsland")),
        Phase("mansion-blaine", "Mansion Secret Key, Blaine",
              _ph_mansion_blaine, B("Volcano")),
        Phase("super-rod", "Bike Voucher detour, Super Rod (Fuchsia)",
              lambda ctx: rr.super_rod_run(ctx.emu, ctx.ex),
              _v_super_rod),
        Phase("dratini", "Safari fishing: Dratini (+ Chansey on exit)",
              _ph_dratini,
              lambda ctx: _v_owned(ctx, 0x58, "Dratini")),
        Phase("fly-journey", "GLUE: Seafoam transit, Cinnabar, "
              "Pallet, Mt. Moon, Cerulean",
              glue_fuchsia_to_cerulean, M("CeruleanPokecenter")),
        Phase("bicycle", "Bike Voucher -> Bicycle (Cerulean)",
              lambda ctx: rr.bicycle_exchange(ctx.emu, ctx.ex),
              lambda ctx: _bag_has(ctx, rr.ITEM_BICYCLE, "Bicycle")),
        Phase("snorlax-journey", "GLUE: R9/R10, Rock Tunnel, "
              "Underground Path, Celadon, R16 stance",
              glue_cerulean_to_snorlax, M("Route16")),
        Phase("snorlax", "Route 16 Snorlax - CATCH (never Master Ball)",
              _ph_snorlax,
              lambda ctx: _v_owned(ctx, rr.SPECIES_SNORLAX, "Snorlax")),
        Phase("fly-house", "Route 16 Fly House - HM02 Fly",
              lambda ctx: rr.route16_fly_house(ctx.emu, ctx.ex),
              lambda ctx: _bag_has(ctx, rr.ITEM_HM02_FLY, "HM02")),
        Phase("teach-fly", "GLUE: Cut tree, Celadon PC, Fly on "
              "Pidgey, fly to Fuchsia",
              glue_teach_fly, M("FuchsiaCity")),
        Phase("viridian-gym", "Fly to Viridian, Giovanni, Earth Badge",
              lambda ctx: rr.viridian_gym(ctx.emu, ctx.ex),
              B("Earth")),
        Phase("r22-prep-grind", "Stock-up + Mansion grind "
              "(Blastoise 66, Arcanine 48)",
              _ph_prep_grind, _v_prep_grind),
        Phase("r22-rival", "Route 22 rival, second attempt profile",
              lambda ctx: rr.route22_rival(ctx.emu, ctx.ex),
              _v_rival2),
        Phase("victory-road", "Route 23 badge gates, Victory Road "
              "(Moltres left alone)",
              _ph_victory_road, M("IndigoPlateau")),
        Phase("league-grind", "League team grind (Lapras 50, "
              "Snorlax 48, Arcanine 52)",
              lambda ctx: rr.league_grind(ctx.emu, ctx.ex),
              _v_league_grind),
        Phase("league-assemble", "Assemble six at Viridian, stock "
              "up, walk to Indigo",
              lambda ctx: rr.league_assemble(ctx.emu, ctx.ex),
              M("IndigoPlateau", "IndigoPlateauLobby")),
        Phase("elite-four", "Lorelei, Bruno, Agatha, Lance, Champion "
              "- Hall of Fame",
              lambda ctx: rr.elite_four(ctx.emu, ctx.ex),
              M("PalletTown")),
    ]


# ---------------------------------------------------------- runner

class Runner:
    """Sequences phases with per-phase checkpoints and bounded
    retries. Duck-typed ctx (note/state/save_state/load_state/reboot)
    so the logic is unit-testable without an emulator."""

    def __init__(self, ctx, phases, out_dir, max_retries: int = 2,
                 start_index: int = 0, provider=None):
        self.ctx = ctx
        self.phases = phases
        self.out_dir = Path(out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.start_index = start_index
        self.last_good: Optional[Path] = None
        self.provider = provider

    def checkpoint_path(self, index: int) -> Path:
        return self.ckpt_dir / f"{index + 1:02d}-{self.phases[index].name}.state"

    def _stop_requested(self) -> bool:
        return self.provider is not None \
            and self.provider.stop_flag.is_set()

    def run(self) -> bool:
        total = len(self.phases)
        for i in range(self.start_index, total):
            if self._stop_requested():
                self.ctx.note("STOP requested via web control; "
                              f"halting before phase {i + 1}")
                return False
            if not self._run_one(i):
                self.ctx.note(
                    f"RUN STOPPED at phase {i + 1}/{total} "
                    f"{self.phases[i].name} after "
                    f"{self.max_retries} retries - see progress log; "
                    f"resume with --resume")
                return False
        self.ctx.note(f"RUN COMPLETE: all {total} phases verified")
        return True

    def _run_one(self, i: int) -> bool:
        ph = self.phases[i]
        total = len(self.phases)
        t0 = time.time()
        if self.provider is not None:
            self.provider.info.update({
                "phase": ph.name, "phase_index": i,
                "phase_total": total, "run_status": "running"})
        for attempt in range(self.max_retries + 1):
            if attempt:
                self.ctx.note(f"PHASE {ph.name}: retry {attempt} "
                              f"from checkpoint {self.last_good}")
                if self.last_good is not None:
                    self.ctx.load_state(self.last_good)
                else:
                    self.ctx.reboot()
            res = None
            try:
                res = ph.run(self.ctx)
            except Exception as exc:
                self.ctx.note(f"PHASE {ph.name}: raised {exc!r}")
                res = False
            okv, ev = (res is not False), ""
            if ph.verify is not None:
                try:
                    okv, ev = ph.verify(self.ctx)
                except Exception as exc:
                    okv, ev = False, f"verify raised {exc!r}"
            if res is not False and okv:
                path = self.checkpoint_path(i)
                self.ctx.save_state(path)
                self.last_good = path
                st = self.ctx.state()
                self.ctx.note(
                    f"PHASE {i + 1:02d}/{total} {ph.name} OK "
                    f"wall={time.time() - t0:.1f}s "
                    f"badges={sorted(st.badges)} money={st.money} "
                    f"evidence: {ev} ckpt={path.name}")
                return True
            self.ctx.note(f"PHASE {ph.name} attempt {attempt + 1} "
                          f"FAILED (run={res!r}; verify: {ev})")
        return False


def discover_checkpoints(out_dir) -> list:
    """[(phase_index0, path)] for run/checkpoints/NN-name.state."""
    ck = Path(out_dir) / "checkpoints"
    found = []
    if ck.is_dir():
        for p in ck.glob("??-*.state"):
            try:
                found.append((int(p.name[:2]) - 1, p))
            except ValueError:
                continue
    return sorted(found)


# ----------------------------------------------------- web provider

class WebProvider:
    """Feeds web_viewer from the live run; also carries the page's
    start/stop controls back to the runner."""

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.info = {"run_status": "idle", "phase": None,
                     "phase_index": -1, "phase_total": 0}
        self.start_event = threading.Event()
        self.stop_flag = threading.Event()
        self._frame_cache = (0.0, None)

    def frame_png(self):
        now = time.time()
        ts, data = self._frame_cache
        if data is not None and now - ts < 0.25:
            return data
        emu = self.ctx.emu
        if emu is None:
            return None
        try:
            buf = io.BytesIO()
            emu.pb.screen.image.save(buf, format="PNG")
            data = buf.getvalue()
        except Exception:
            return self._frame_cache[1]
        self._frame_cache = (now, data)
        return data

    def state_dict(self):
        from web_viewer import state_to_dict
        info = dict(self.info)
        info["recent_log"] = self.ctx.recent[-12:]
        if self.ctx.emu is None:
            return {**info, "map_id": -1, "map": None, "x": 0, "y": 0,
                    "in_battle": False, "money": 0, "badges": [],
                    "party": []}
        d = state_to_dict(self.ctx.state(), self.ctx.world, info)
        try:
            d["dex"] = rr._dex_owned_count(self.ctx.emu)
        except Exception:
            pass
        return d

    def control(self, action: str) -> dict:
        if action == "start":
            self.start_event.set()
            return {"ok": True, "action": "start"}
        if action == "stop":
            self.stop_flag.set()
            return {"ok": True, "action": "stop"}
        return {"ok": False, "action": action}


# ------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Play the whole of Pokemon Red with the "
                    "deterministic agent (see module docstring).")
    ap.add_argument("--rom", help="path to a Pokemon Red ROM "
                                  "(or set POKERED_ROM)")
    ap.add_argument("--out", default="./run",
                    help="run dir: checkpoints + progress.log "
                         "(default ./run)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the latest checkpoint in --out")
    ap.add_argument("--from-phase", dest="from_phase",
                    help="start at this phase (needs --state or "
                         "checkpoints in --out)")
    ap.add_argument("--state", help="save state file to start from")
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--supervisor", choices=["off", "adapter", "stock"],
                    default="off",
                    help="opening executor planner (default off: "
                         "the deterministic checklist)")
    ap.add_argument("--web", type=int, metavar="PORT",
                    help="serve the live viewer on this port")
    ap.add_argument("--host", default="127.0.0.1",
                    help="viewer bind address (default localhost; "
                         "0.0.0.0 = LAN, e.g. on an Orin)")
    ap.add_argument("--web-manual", action="store_true",
                    help="with --web: wait for the page's Start "
                         "button before running")
    args = ap.parse_args(argv)

    phases = build_phases()
    if args.list_phases:
        for i, ph in enumerate(phases):
            print(f"{i + 1:02d} {ph.name:<16} {ph.title}")
        return 0

    rom = args.rom or os.environ.get("POKERED_ROM")
    if not rom:
        ap.error("no ROM: pass --rom PATH or set POKERED_ROM "
                 "(the ROM is not distributed with this repo)")

    names = [ph.name for ph in phases]
    start_index = 0
    if args.from_phase:
        if args.from_phase not in names:
            ap.error(f"unknown phase {args.from_phase!r}; "
                     f"--list-phases shows all {len(names)}")
        start_index = names.index(args.from_phase)

    ctx = Ctx(rom, args.out, supervisor=args.supervisor)
    runner = Runner(ctx, phases, args.out,
                    max_retries=args.max_retries,
                    start_index=start_index)

    start_state = args.state
    ckpts = discover_checkpoints(args.out)
    if start_state is None and (args.resume or args.from_phase) and ckpts:
        idx, path = ckpts[-1]
        start_state = str(path)
        if args.resume:
            start_index = idx + 1
            runner.start_index = start_index
    if start_state is not None:
        ctx.load_state(start_state)
        runner.last_good = Path(start_state)
        ctx.note(f"START from state {start_state} at phase "
                 f"{start_index + 1}/{len(phases)} "
                 f"{phases[start_index].name if start_index < len(phases) else '-'}")
    elif start_index == 0:
        ctx.boot_fresh()
        ctx.note("START fresh boot")
    else:
        ap.error("starting mid-game needs --state or existing "
                 "checkpoints in --out (see --resume)")

    if start_index >= len(phases):
        ctx.note("nothing to do: all phases already checkpointed")
        return 0

    provider = None
    server = None
    if args.web:
        from web_viewer import make_server
        provider = WebProvider(ctx)
        runner.provider = provider
        server = make_server(args.host, args.web, provider)
        threading.Thread(target=server.serve_forever,
                         daemon=True).start()
        ctx.note(f"WEB viewer on http://{args.host}:{args.web}/ "
                 f"(frame + state; Start/Stop on the page)")

    ok = True
    try:
        if provider is not None and args.web_manual:
            provider.info["run_status"] = "waiting for Start"
            ctx.note("WEB manual mode: waiting for Start on the page")
            provider.start_event.wait()
        if provider is not None:
            provider.info["run_status"] = "running"
        ok = runner.run()
        if provider is not None:
            provider.info["run_status"] = "complete" if ok else "stopped"
    except KeyboardInterrupt:
        ctx.note("interrupted by user")
        ok = False
    if server is not None:
        ctx.note("run finished; viewer still serving - Ctrl-C to exit")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
