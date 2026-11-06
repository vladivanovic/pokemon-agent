#!/usr/bin/env python3
"""Play the real game: boot a Pokemon Red ROM in PyBoy and let the agent
drive from the title screen through the opening.

    python3 run_real.py --rom /path/to/pokered.gbc [--workdir DIR]
                        [--player VLAD] [--rival GARY] [--starter squirtle]
                        [--max-rounds 12]

The ROM is not included in this repo. Building pret/pokered from source
(with RGBDS) produces a byte-identical retail Red ROM; any legally
obtained Red ROM with the same WRAM layout works.

What is deterministic here, and what is a scripted macro:

- Boot, Oak's intro speech and the two naming screens are a *macro*
  (_type_name below): a fresh boot is deterministic, so the speech is
  advanced by counted presses at a measured cadence (no RAM signal
  marks the naming screens - their tilemap shadow stays blank), names
  are typed on the letter grid, and the macro only hands over once a
  test step proves the player is free in the bedroom. Everything after
  the bedroom is the agent proper.
- The Oak intercept (walk north out of Pallet without a Pokemon and Oak
  walks you to the lab) and the starter pick are also macros, because
  they are cutscenes with their own rules: the lab's Poke Balls are inert
  flavor text until the intercept arms them, and the pick involves a
  full-screen PokeDex page that is not a standard text box. Both macros
  are short, use only the backend's verified primitives (walk_to,
  close_dialog, wait_free), and hand control back to the Executor.
- Everything else - leaving the lab, the rival battle, Route 1, the trip
  to Viridian Mart - is the Executor + ChecklistPlanner, unscripted.

Verified end-to-end on 2026-11-03 against a from-source pokered build
(SHA1 ea9bcae617fdf159b045185467ae58b2e4a48b9a): the agent took Squirtle,
won the rival battle, fought through Route 1 and reached Viridian Mart,
where the parcel scene played. Known open edge: the parcel *return* leg.
After the Mart's handover scene the player ends up in a spot where the
static interior model and the real game disagree (see README), so the
runner stops at the Mart for now instead of thrashing.

Speed: PyBoy headless runs this at roughly 2,000-2,500 fps, about
35-40x real-time; the whole opening takes a few seconds of wall clock.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# PyBoy wants SDL; headless machines need the dummy drivers, and they
# must be set before pyboy is imported (PyBoyEmulator imports it).
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from agent.executor import Executor, SmartFightPolicy, StuckError
from agent.goals import Goal
from agent.memory import EpisodicMemory
from agent.planner import ChecklistPlanner, LLMPlanner
from agent.pyboy_emu import PyBoyEmulator
from agent.world import default_world

MAP_REDS_HOUSE_2F = 38
MAP_PALLET = 0
MAP_OAKS_LAB = 40

# Event flags live in wEventFlags (0xD747), one bit per event constant
# in pokered's constants/event_constants.asm order - counting the
# const_skip gaps, which a naive line count misses (the PokeDex flag
# read as never-set until the skips were counted).
ADDR_EVENT_FLAGS = 0xD747
EVENT_GOT_POKEDEX = 37


def _event_flag(emu: PyBoyEmulator, event_id: int) -> bool:
    byte = emu.pb.memory[ADDR_EVENT_FLAGS + event_id // 8]
    return bool(byte & (1 << (event_id % 8)))

# Gen 1 naming-screen letter grid: (column, row). ED (confirm) sits at
# (8, 4). Navigation sequences were verified against the real screen.
_LETTER_ROWS = ["ABCDEFGHI", "JKLMNOPQR", "STUVWXYZ"]
LETTER_POS = {ch: (col, row) for row, letters in enumerate(_LETTER_ROWS)
              for col, ch in enumerate(letters)}
ED_POS = (8, 4)

# Ball tiles in Oak's Lab and the tile to stand on in front of each.
STARTERS = {
    "charmander": (6, 3),
    "squirtle": (7, 3),
    "bulbasaur": (8, 3),
}


def _raw(emu: PyBoyEmulator, button: str, hold: int = 5,
         settle: int = 45) -> None:
    """A raw joypad press at the intro's measured cadence. The
    backend's press() must NOT be used on menu/naming screens: for
    directions it applies tile-step semantics (hold until the RAM
    coordinates change), and on a naming screen coordinates never
    change, so every press would burn its full timeout."""
    emu.pb.button_press(button)
    emu.tick(hold)
    emu.pb.button_release(button)
    emu.tick(settle)


def _type_name(emu: PyBoyEmulator, name: str) -> None:
    """Type a name on the naming screen. Assumes the screen is up and
    the cursor starts on 'A'. Clears any pre-typed letters first (the A
    presses that advance Oak's speech can land in the name field).
    Verified live: 'VLAD' typed this way reads back on screen and Oak
    confirms 'So your name is VLAD!'."""
    for _ in range(7):
        _raw(emu, "b")
    col = row = 0

    def move_to(tcol: int, trow: int) -> None:
        nonlocal col, row
        while col < tcol:
            _raw(emu, "right"); col += 1
        while col > tcol:
            _raw(emu, "left"); col -= 1
        while row < trow:
            _raw(emu, "down"); row += 1
        while row > trow:
            _raw(emu, "up"); row -= 1

    for ch in name.upper()[:10]:
        if ch not in LETTER_POS:
            continue
        move_to(*LETTER_POS[ch])
        _raw(emu, "a")
    move_to(*ED_POS)
    _raw(emu, "a")
    emu.tick(120)


_OPPOSITE = {"right": "left", "left": "right", "up": "down", "down": "up"}


def verify_mobility(emu: PyBoyEmulator, rounds: int = 40) -> bool:
    """Scripted scenes can report 'free' (no box, stable position)
    while input is still locked for a few more beats. The only proof
    of control is a step that actually moves the player: try the four
    directions (the tile east of Oak's Lab deposit holds a Poke Ball,
    so no single direction suffices), step back, and only then hand
    over. (This exact test is what un-stuck the lab scene in the
    original session.)"""
    for _ in range(rounds):
        st = emu.read_state()
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
            emu.tick(30)
            continue
        if st.in_battle:
            return False
        before = (st.x, st.y)
        for button in ("down", "right", "left", "up"):
            emu.press(button)
            st2 = emu.read_state()
            if (st2.x, st2.y) != before:
                emu.press(_OPPOSITE[button])
                return True
        emu.tick(30)
    return False


def boot_and_intro(emu: PyBoyEmulator, player: str, rival: str) -> None:
    """Title screen -> NEW GAME -> Oak's speech -> both naming screens
    -> the bedroom in Red's house. Dialog-driven where possible: text
    boxes are advanced one verified press at a time, and a screen that
    stops changing under A presses (and is not a text box) is treated
    as a naming screen."""
    emu.tick(1500)                       # copyright + title animation
    _raw(emu, "start")
    emu.tick(60)
    _raw(emu, "a")                       # NEW GAME (no save -> top item)
    emu.tick(120)
    # Oak's speech, part 1: a fresh boot is deterministic, and at this
    # cadence the "YOUR NAME?" screen is up after 29 A presses
    # (measured; a stray press or two just types A's the typer clears).
    # No RAM signal marks the naming screens - the tilemap shadow stays
    # blank for them - which is why this phase is counted, not sensed.
    for _ in range(29):
        _raw(emu, "a")
    _type_name(emu, player)
    if not emu.read_state().in_dialog:
        raise RuntimeError("player naming did not confirm "
                           "(expected Oak's 'So your name is...' line)")
    # Speech, part 2: the rival walks in and Oak asks his name. Ride
    # the dialogs; the naming screen is the first screen that stays
    # box-free, and extra A presses only type A's (cleared as before).
    for _ in range(16):
        _raw(emu, "a")
    _type_name(emu, rival)
    # Speech tail + shrink scene: advance dialogs until the player is
    # verifiably free in the bedroom. (The map id alone proves nothing:
    # the engine preloads the spawn map before the speech, so it reads
    # 38 the whole time.)
    for _ in range(300):
        st = emu.read_state()
        if st.in_dialog:
            emu.close_dialog(max_presses=1)
            continue
        if st.map_id == MAP_REDS_HOUSE_2F and verify_mobility(emu, rounds=2):
            return
        _raw(emu, "a", settle=20)
    raise RuntimeError("intro did not reach a free bedroom")


def trigger_oak_intercept(emu: PyBoyEmulator, ex: Executor) -> None:
    """Walk out of the house, head for Pallet's north exit and step
    toward Route 1: Oak intercepts, walks the player to the lab and
    arms the starter event. Uses the executor's own journey planner to
    reach the exit tile; only the final step and the cutscene watch
    are scripted."""
    goal = Goal(kind="goto", target_map=MAP_PALLET, target_xy=(10, 1),
                reason="trigger the Oak intercept")
    ex._goto(goal, emu.read_state())
    emu.press("up")
    # The cutscene alternates dialog beats with scripted walking (the
    # instrumented trace: Pallet dialogs, a walk down through town, in
    # the lab door, up to (5, 3), then Oak's long speech). Ride it out:
    # clear dialogs as they come, and only call it done after the
    # player has stood still in the lab, box-free, for ~360 frames -
    # the scene pauses mid-way with the player briefly able to move,
    # so a quick mobility check declares victory too early - and then
    # prove mobility with a test step anyway.
    quiet = 0
    last_pos = None
    for _ in range(300):
        st = emu.read_state()
        if st.in_dialog:
            emu.close_dialog(max_presses=1)
            quiet = 0
            continue
        pos = (st.map_id, st.x, st.y)
        if pos == last_pos and st.map_id == MAP_OAKS_LAB:
            quiet += 1
            if quiet >= 12 and verify_mobility(emu, rounds=2):
                return
        else:
            quiet = 0
            last_pos = pos
        emu.tick(30)
    raise RuntimeError("Oak intercept did not settle in the lab")


def pick_starter(emu: PyBoyEmulator, ex: Executor, starter: str) -> None:
    """Take a starter ball: stand in front of it, face it, examine.
    The flow crosses a full-screen PokeDex preview (not a text box, so
    in_dialog misses it): press A when no box is up, close boxes when
    one is. Then decline the nickname and ride out the rival's pick."""
    ball = STARTERS[starter]
    # Oak walks back to his desk as a free NPC after the cutscene and
    # can body-block the aisle for a few seconds; NPC blocks are
    # transient, so wait and retry rather than failing the pick.
    for attempt in range(8):
        try:
            ex._walk_to(MAP_OAKS_LAB, (ball[0], ball[1] + 1))
            break
        except Exception:
            if attempt == 7:
                raise
            emu.tick(150)
    emu.press("up")                      # face the ball
    emu.press("a")                       # examine -> dex preview
    for _ in range(80):
        st = emu.read_state()
        # A non-empty party read can be a mid-write transient (count
        # byte set, struct still zeros); only a sane struct counts.
        if st.party and st.party[0].level > 0:
            break
        if st.in_dialog:
            emu.close_dialog(max_presses=1)   # "...You want X?" -> YES
        else:
            emu.press("a")                    # advance the dex page
            emu.tick(20)
    st = emu.read_state()
    if not (st.party and st.party[0].level > 0):
        raise RuntimeError(f"starter {starter} never joined the party")
    # Nickname prompt: YES is the default and a too-early "down" gets
    # eaten by the receive text, landing on YES - which opens the
    # naming screen, where dialog-mashing then types AAAAAAA (seen
    # live). Press down twice with a beat between, then confirm.
    emu.press("down")
    emu.tick(30)
    emu.press("down")
    emu.tick(10)
    emu.press("a")
    # Ride out the rival's pick. If the naming screen DID open (no
    # dialog, no battle, yet the player never becomes mobile), recover
    # properly: clear the field and type the species name on the grid.
    # (The mobility test's four direction presses are cursor-neutral
    # from the grid's top-left start, so the typer's bearings hold.)
    static = 0
    for _ in range(240):
        st = emu.read_state()
        if st.in_dialog:
            emu.close_dialog(max_presses=1)
            static = 0
            continue
        if st.in_battle:
            break
        if verify_mobility(emu, rounds=1):
            break
        static += 1
        if static == 12:
            for _ in range(10):
                _raw(emu, "b")
            _type_name(emu, starter)
            static = 0
        emu.tick(30)


def deliver_parcel_scene(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The real delivery: walk to Oak (he stands at (5, 2) facing
    down), talk to him from (5, 3) with the parcel in the bag. His
    script takes the parcel and starts the rival-arrives scene, at the
    end of which both trainers get the PokeDex (EVENT_GOT_POKEDEX).
    The checklist counts delivery by arrival; the *game* counts this
    scene - Viridian's north road stays blocked until it has played.
    Returns True when the PokeDex flag is set and the player is free."""
    if _event_flag(emu, EVENT_GOT_POKEDEX):
        return True
    for attempt in range(8):
        try:
            ex._walk_to(MAP_OAKS_LAB, (5, 3))
            break
        except Exception:
            if attempt == 7:
                raise
            emu.tick(150)
    emu.press("up")                      # face Oak
    emu.press("a")                       # talk -> parcel handover
    quiet = 0
    last_pos = None
    for _ in range(400):
        st = emu.read_state()
        if st.in_dialog:
            emu.close_dialog(max_presses=1)
            quiet = 0
            continue
        pos = (st.map_id, st.x, st.y)
        if pos == last_pos:
            quiet += 1
            if quiet >= 12 and _event_flag(emu, EVENT_GOT_POKEDEX) \
                    and verify_mobility(emu, rounds=2):
                return True
        else:
            quiet = 0
            last_pos = pos
        emu.tick(30)
    return _event_flag(emu, EVENT_GOT_POKEDEX)


MAP_PEWTER_GYM = 54


def fight_brock(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Win the Boulder Badge. Brock stands at (4, 1) facing down;
    talk to him from (4, 2). The Jr. Trainer at (3, 6) guards the
    aisle with a line-of-sight battle that interrupts the walk - the
    executor's battle policy fights it and the walk resumes. Brock
    himself (Geodude Lv12, Onix Lv14) against the A-mash policy is a
    level check more than a tactics check: if the party faints, the
    game blacks out home, the executor re-runs the journey, and the
    rematch happens levels higher. Verified by the badge bit itself
    (state.badges), never by dialog progress."""
    for _attempt in range(40):
        st = emu.read_state()
        if "Boulder" in st.badges:
            return True
        if st.in_battle:
            try:
                ex._battle()
            except StuckError:
                return "Boulder" in emu.read_state().badges
            continue
        if st.map_id != MAP_PEWTER_GYM:
            return "Boulder" in emu.read_state().badges  # blacked out
        if st.in_dialog:
            emu.close_dialog()
            continue
        try:
            ex._walk_to(MAP_PEWTER_GYM, (4, 2))
        except Exception:
            emu.tick(60)                 # trainer battle / NPC shuffle
            continue
        emu.press("up")                  # face Brock
        emu.press("a")                   # talk -> challenge dialog
        for _ in range(240):
            st = emu.read_state()
            if "Boulder" in st.badges:
                break
            if st.in_battle:
                try:
                    ex._battle()
                except StuckError:
                    break
                continue
            if st.in_dialog:
                emu.close_dialog(max_presses=1)
                continue
            emu.tick(30)
        emu.wait_free(max_rounds=60, stable_needed=6)
        if "Boulder" in emu.read_state().badges:
            return True
    return "Boulder" in emu.read_state().badges


MAP_CERULEAN_GYM = 65

# The bag's item pocket (wNumBagItems / wBagItems in pokered WRAM).
ADDR_NUM_BAG_ITEMS = 0xD31D
ADDR_BAG_ITEMS = 0xD31E
ITEM_SS_TICKET = 0x3F
ITEM_TM28_DIG = 0xE4          # TM ids are 0xC9 + (n - 1) in the pocket


def _bag_items(emu: PyBoyEmulator) -> list:
    n = emu.pb.memory[ADDR_NUM_BAG_ITEMS]
    return [emu.pb.memory[ADDR_BAG_ITEMS + 2 * i] for i in range(n)]


class _FixedPlanner:
    """Scene-local planner: propose one fixed goto until swapped back.
    The scene functions below drive cross-map rides with the executor's
    own run() loop (which fights battles and retries goals) by
    temporarily replacing the checklist planner."""

    def __init__(self, map_id: int, xy=None):
        self.map_id = map_id
        self.xy = xy

    def propose(self, state, world, memory):
        return Goal("goto", self.map_id, self.xy, "scene ride")


def _ride_to(emu: PyBoyEmulator, ex: Executor, map_id: int,
             xy=None, rounds: int = 200) -> bool:
    """Ride the executor to a map (optionally a tile), forgiving each
    round's goal the way main()'s loop does. Scripted scenes on the
    way (sight battles, the Cerulean rival) are fought by the battle
    policy and ridden out by the retry."""
    saved = ex.planner
    ex.planner = _FixedPlanner(map_id, xy)
    try:
        for _ in range(rounds):
            result = ex.run(max_goals=1)
            if result.log and result.log[-1].get("goal"):
                ex.memory.forgive(result.log[-1]["goal"])
            if emu.read_state().map_id == map_id:
                return True
        return emu.read_state().map_id == map_id
    finally:
        ex.planner = saved


def _step(emu: PyBoyEmulator, ex: Executor, button: str,
          retries: int = 5) -> bool:
    """One press-until-move tile step. A bare emu.press() only turns
    the player when it starts a new direction (Gen 1 semantics), so a
    step is proven by the RAM coordinates changing, with retries for
    NPC shuffles. Used inside the small house interiors, where the
    probes proved step sequences rather than _walk_to paths."""
    for _ in range(retries):
        before = emu.read_state()
        if before.in_battle:
            ex._battle()
            emu.wait_free(max_rounds=10, stable_needed=3)
            before = emu.read_state()
        for _ in range(4):
            emu.press(button)
            emu.tick(20)
            after = emu.read_state()
            if after.in_dialog:
                emu.close_dialog(max_presses=3)
                after = emu.read_state()
            if (after.x, after.y) != (before.x, before.y) \
                    or after.map_id != before.map_id:
                return True
        emu.tick(45)
    return False


def heal_at_pokecenter(emu: PyBoyEmulator, ex: Executor,
                       pc_map: int) -> bool:
    """Heal at a Pokecenter counter: stand at (3, 3), talk up across
    the counter, and verify from RAM that HP and PP are fully
    restored before leaving by the exit mat. Proven at the Cerulean
    center (probe49/52) and again at Vermilion (probe99). Assumes the
    player is already inside the center."""
    try:
        ex._walk_to(pc_map, (3, 3))
    except StuckError:
        return False
    emu.press("up")                      # face the nurse
    emu.press("a")                       # talk -> heal dialog
    for _ in range(6):
        emu.tick(30)
        ex._clear_dialog(max_presses=80)
        emu.wait_free(max_rounds=15, stable_needed=3)
        st = emu.read_state()
        if st.party and st.party[0].hp == st.party[0].max_hp:
            break
    st = emu.read_state()
    healed = bool(st.party) and st.party[0].hp == st.party[0].max_hp
    try:
        ex._walk_to(pc_map, (3, 7))      # exit mat
    except StuckError:
        pass
    emu.wait_free(max_rounds=10, stable_needed=3)
    return healed


def fight_misty(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Win the Cascade Badge: fight_brock's pattern, one room east.
    Misty stands at (4, 2) facing down; the proven talk spot is
    (5, 2), facing west (probe51, from the probe47 save). The tile
    south of her, (4, 3), is genuinely walkable - probe50 called it
    impossible, but that was a single-tap artifact (a bare press only
    turns the player); probe54 stood on it via _walk_to, and its four
    quadrants are all GYM floor ($11) in pokered's block data, so the
    world model needs no change. Her gym trainers sight-fight on the
    way in, like Brock's Jr. Trainer. Verified by the badge bit."""
    gym = MAP_CERULEAN_GYM
    for _attempt in range(40):
        st = emu.read_state()
        if "Cascade" in st.badges:
            return True
        if st.in_battle:
            try:
                ex._battle()
            except StuckError:
                return "Cascade" in emu.read_state().badges
            continue
        if st.map_id != gym:
            return "Cascade" in emu.read_state().badges  # blacked out
        if st.in_dialog:
            emu.close_dialog()
            continue
        try:
            ex._walk_to(gym, (5, 2))
        except Exception:
            emu.tick(60)                 # trainer battle / NPC shuffle
            continue
        emu.press("left")                # face Misty
        emu.press("a")                   # talk -> challenge dialog
        for _ in range(240):
            st = emu.read_state()
            if "Cascade" in st.badges:
                break
            if st.in_battle:
                try:
                    ex._battle()
                except StuckError:
                    break
                continue
            if st.in_dialog:
                emu.close_dialog(max_presses=1)
                continue
            emu.tick(30)
        emu.wait_free(max_rounds=60, stable_needed=6)
        if "Cascade" in emu.read_state().badges:
            return True
    return "Cascade" in emu.read_state().badges


def get_ss_ticket(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Bill's House: turn Bill back into a human and take the S.S.
    Ticket he hands over. Assumes the player is inside the house.
    The sequence is scripted by the game, and every beat is a
    different interaction kind (probe86-89):
    1. Talk to Pokemon-Bill (the monster sprite at (6, 5)) from
       (5, 5), facing right. Either answer to his yes/no proceeds;
       he walks up into the teleporter and is hidden.
    2. Run the Cell Separator from his PC: a *hidden event* at tile
       (1, 4), fired by facing UP from (1, 5) and pressing A
       (engine/events/hidden_events/bills_house_pc.asm). It sets
       EVENT_USED_CELL_SEPARATOR_ON_BILL, and Bill (human) exits the
       machine and walks to his object tile (4, 4).
    3. Talk to him there from (4, 5), facing up: he gives the ticket.
    Verified by the ticket itself (item $3F) in the bag, never by
    dialog progress."""
    bh = ex.world.map_id("BillsHouse")
    if ITEM_SS_TICKET in _bag_items(emu):
        return True
    try:
        ex._walk_to(bh, (5, 5))
    except StuckError:
        return False
    emu.press("right")                   # face Pokemon-Bill
    emu.press("a")
    emu.tick(40)
    ex._clear_dialog(max_presses=100)
    for _ in range(12):                  # he walks to the machine
        emu.tick(60)
    try:
        ex._walk_to(bh, (1, 5))
    except StuckError:
        return False
    emu.press("up")                      # face the PC (hidden event)
    emu.tick(20)
    emu.press("a")
    emu.tick(60)
    ex._clear_dialog(max_presses=60)
    for _ in range(20):                  # separator runs; Bill exits
        emu.tick(60)
        if emu.read_state().in_dialog:
            ex._clear_dialog(max_presses=40)
    try:
        ex._walk_to(bh, (4, 5))
    except StuckError:
        return False
    emu.press("up")                      # face Bill at (4, 4)
    emu.tick(20)
    emu.press("a")
    emu.tick(40)
    ex._clear_dialog(max_presses=120)
    for _ in range(8):
        emu.tick(60)
    return ITEM_SS_TICKET in _bag_items(emu)


def leave_cerulean_south(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Cerulean -> Route 5, the long way the game actually allows.
    Cerulean's southern tree line is sealed scenery (proven by an
    exhaustive in-game DFS, probe83: 360 reachable cells, none at
    y >= 29), and the Trashed House's front doorstep (27, 12) is
    occupied by a guard until the S.S. Ticket is in the bag - with
    the ticket he steps aside (probe90). From there: through the
    house, out the hole in its back wall into the yard, fight the
    Rocket thief (his script triggers from (30, 7)/(30, 9)), then
    around the house's east side, down the river bank (x=33-36) and
    west along the southern terrace to the (12, 35) crossing.
    Assumes the player is in Cerulean City with the ticket. Returns
    True standing on Route 5 (probe94/96)."""
    world = ex.world
    cc = world.map_id("CeruleanCity")
    house = world.map_id("CeruleanTrashedHouse")
    r5 = world.map_id("Route5")
    if emu.read_state().map_id == r5:
        return True
    if ITEM_SS_TICKET not in _bag_items(emu):
        return False
    if not _ride_to(emu, ex, house):
        return False
    # Interior: from the entry (2, 7) to the hole approach (3, 1),
    # then up through the hole to the yard at Cerulean (27, 9). The
    # centre table blocks the direct column; this is probe94's
    # proven step sequence.
    for button in ("up", "up", "up", "up", "up", "right", "up", "up"):
        if not _step(emu, ex, button):
            return False
    emu.wait_free(max_rounds=10, stable_needed=3)
    if emu.read_state().map_id != cc:
        return False
    # The Rocket's script: stepping on (30, 9) makes him walk over,
    # talk, and fight. The walk itself can be interrupted by his
    # approach; ride out dialog and battle until TM28 is in the bag.
    try:
        ex._walk_to(cc, (30, 9))
    except StuckError:
        pass
    for _ in range(60):
        if ITEM_TM28_DIG in _bag_items(emu):
            break
        st = emu.read_state()
        if st.in_battle:
            ex._battle()
            emu.wait_free(max_rounds=10, stable_needed=3)
            continue
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
            continue
        emu.tick(40)
    if ITEM_TM28_DIG not in _bag_items(emu):
        return False
    # Yard -> east side -> bank -> terrace -> crossing (the model
    # path proven cell-by-cell in probe96).
    for xy in ((33, 15), (36, 25), (20, 31), (12, 35)):
        try:
            ex._walk_to(cc, xy)
        except StuckError:
            return False
    if not _step(emu, ex, "down"):       # cross the south edge
        return False
    emu.wait_free(max_rounds=10, stable_needed=3)
    return emu.read_state().map_id == r5


def ride_underground_to_vermilion(emu: PyBoyEmulator,
                                  ex: Executor) -> bool:
    """Route 5 -> Vermilion City via the Underground Path. The
    Saffron gates are a dead end (thirsty guard), and the planner's
    naive journey from Route 5 walks into the Route 5 gatehouse, so
    the tunnel houses are explicit waypoints (probe97/99). Map 119
    (the tunnel itself) was missing from the extracted data until
    the 2026-11-04 build fix - without it no journey through the
    tunnel exists at all. Assumes the player is on Route 5; returns
    True healed at the Vermilion Pokecenter."""
    world = ex.world
    h5 = world.map_id("UndergroundPathRoute5")
    h6 = world.map_id("UndergroundPathRoute6")
    r6 = world.map_id("Route6")
    vc = world.map_id("VermilionCity")
    vpc = world.map_id("VermilionPokecenter")
    if not _ride_to(emu, ex, h5):
        return False
    if not _ride_to(emu, ex, h6):
        return False
    # The exit mat is (4, 7); the tunnel warp lands at (4, 4) and
    # the interior walk was a probe-proven step sequence, not a
    # _walk_to (probe99).
    for _ in range(3):
        if emu.read_state().map_id == r6:
            break
        _step(emu, ex, "down")
    emu.wait_free(max_rounds=10, stable_needed=3)
    if emu.read_state().map_id != r6:
        return False
    if not _ride_to(emu, ex, vc):
        return False
    if not _ride_to(emu, ex, vpc):
        return False
    return heal_at_pokecenter(emu, ex, vpc)


def eastward_from_cerulean(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The whole post-Cascade leg, as proven by probes 85-99: heal,
    ride north through the rival fight and Nugget Bridge to Bill's
    House, get the S.S. Ticket, come back, heal again, leave
    Cerulean south through the robbed house, and take the
    Underground Path to Vermilion. Each stage verifies itself from
    RAM; a stage that fails stops the chain rather than pressing on
    unverified."""
    world = ex.world
    cc = world.map_id("CeruleanCity")
    cpc = world.map_id("CeruleanPokecenter")
    r24 = world.map_id("Route24")
    r25 = world.map_id("Route25")
    bh = world.map_id("BillsHouse")
    if not _ride_to(emu, ex, cpc):
        return False
    heal_at_pokecenter(emu, ex, cpc)
    if not _ride_to(emu, ex, r24):
        return False
    if not _ride_to(emu, ex, r25):
        return False
    if not _ride_to(emu, ex, bh):
        return False
    if not get_ss_ticket(emu, ex):
        return False
    if not _ride_to(emu, ex, cc):
        return False
    if not _ride_to(emu, ex, cpc):
        return False
    heal_at_pokecenter(emu, ex, cpc)
    if not leave_cerulean_south(emu, ex):
        return False
    return ride_underground_to_vermilion(emu, ex)


# ----------------------------------------------------------------------
# The Vermilion leg: SS Anne -> HM01 Cut -> Lt. Surge.
# Proven in probes 100-138 (~/workspace/pokemon-play/), November 2026.
# Two hard-won lessons shape the code below:
#
# * The stock SmartFightPolicy cannot be trusted inside Vermilion
#   Gym. It never reads the battle menu's cursor COLUMN, so it can
#   open the party screen and A-press the active mon's "already
#   out" text forever (probe132); and the executor's 4-frame taps
#   fall between the battle engine's joypad polls during animation
#   phases. _BattleDriver fixes both: column-aware steering, B to
#   escape the party screen, a stale-signature guard for the text
#   phases (the menu bytes freeze mid-animation - the same stale
#   echo the policy's own probe logic exists for), 10-frame holds,
#   and a forced-switch branch for after a faint.
#
# * The Squirtle line CANNOT learn Cut in Gen 1 (pokered
#   base_stats: of the HMs it takes only Surf and Strength). Cut
#   goes on a caught Oddish. And Surge's Raichu (Thunderbolt,
#   2x into Wartortle) is answered by a cave Diglett: Ground is
#   immune to every Electric move and Dig is super-effective
#   into his whole team.

ITEM_POKE_BALL = 0x04
ITEM_HM01 = 0xC4
ITEM_HM05 = 0xC8               # HM ids are 0xC4 + (n - 1) in the pocket
MOVE_CUT = 15
MOVE_DIG = 91
SPECIES_ODDISH = 0xB9
SPECIES_DIGLETT = 0x3B
# Species ids are Gen 1 INTERNAL indices (party struct byte 0),
# not National Dex numbers - pokered constants/pokemon_constants.
SPECIES_NIDORAN_M = 0x03
SPECIES_SPEAROW = 0x05
SPECIES_NIDORAN_F = 0x0F
SPECIES_PIDGEY = 0x24
SPECIES_PIKACHU = 0x54
SPECIES_WEEDLE = 0x70
SPECIES_KAKUNA = 0x71
SPECIES_CATERPIE = 0x7B
SPECIES_METAPOD = 0x7C
SPECIES_RATTATA = 0xA5

# wPokedexOwned: 151 flags in DEX order (probe142 read it raw).
ADDR_POKEDEX_OWNED = 0xD2F7
# The current PC box's count / species list in WRAM (the box data
# is banked in; box 1 is current until someone uses a PC).
ADDR_BOX_COUNT = 0xDA80        # wBoxCount
ADDR_BOX_SPECIES = 0xDA81      # wBoxSpecies, 0xFF-terminated

ADDR_MENU_TOP_Y = 0xCC24
ADDR_MENU_ITEM = 0xCC26
ADDR_MENU_COL = 0xCC25
ADDR_MENU_SCROLL = 0xCC36
ADDR_ENEMY_SPECIES = 0xCFE5
ADDR_ENEMY_LEVEL = 0xCFF3
ADDR_ENEMY_MAX_HP = 0xCFF4
ADDR_BATTLE_MON_SPECIES = 0xD014
ADDR_BATTLE_MON_HP = 0xD015
ADDR_TILE_IN_FRONT = 0xCFC6
ADDR_FIRST_LOCK_CAN = 0xD743
ADDR_SECOND_LOCK_CAN = 0xD744

# Base powers (pokered data/moves/moves.asm) for the moves this
# party carries; the battle driver ranks damaging moves by power
# because "last slot" is wrong here (Bubblebeam 65 sits in slot 1,
# Water Gun is only 40). Unknown damaging moves rank 30.
_MOVE_POWER = {61: 65, 55: 40, 145: 20, 10: 40, 91: 100, 71: 25,
               15: 50}


def _press10(emu: PyBoyEmulator, button: str, settle: int = 25):
    """A press the battle/menu engines reliably register."""
    _raw(emu, button, hold=10, settle=settle)
    emu.tick(20)


def _bag_index(emu: PyBoyEmulator, item_id: int):
    mem = emu.pb.memory
    for i in range(mem[0xD31D]):
        if mem[ADDR_BAG_ITEMS + 2 * i] == item_id:
            return i
    return None


def _bag_count(emu: PyBoyEmulator, item_id: int) -> int:
    i = _bag_index(emu, item_id)
    if i is None:
        return 0
    return emu.pb.memory[ADDR_BAG_ITEMS + 2 * i + 1]


def _party_index_of(emu: PyBoyEmulator, species_id: int):
    for i, mon in enumerate(emu.read_state().party):
        if mon.species_id == species_id:
            return i
    return None


def _dex_owned_count(emu: PyBoyEmulator) -> int:
    """Species registered as owned in the Pokedex - the Route 2
    aide's HM05 gate, and the cheapest proof a catch was new."""
    mem = emu.pb.memory
    return sum(bin(mem[ADDR_POKEDEX_OWNED + i]).count("1")
               for i in range(19))


def _box_count(emu: PyBoyEmulator) -> int:
    """Mons in the current PC box (wBoxCount)."""
    return emu.pb.memory[ADDR_BOX_COUNT]


def _stored_count(emu: PyBoyEmulator) -> int:
    """Party + box: everywhere a fresh catch can land. Catch
    verification must use this, not the party size alone - with
    a full party the engine sends the catch straight to the box
    (item_effects.asm SendNewMonToBox) and the party never grows."""
    return len(emu.read_state().party) + _box_count(emu)


def _ranked_move_fn(emu: PyBoyEmulator):
    """fn(party idx) -> damaging move indices, best power first."""
    from agent.executor import STATUS_MOVE_IDS

    def fn(idx):
        st = emu.read_state()
        mon = st.party[idx]
        cands = []
        for i, mv in enumerate(mon.moves):
            if mv and mv not in STATUS_MOVE_IDS \
                    and i < len(mon.pp) and mon.pp[i] > 0:
                cands.append((_MOVE_POWER.get(mv, 30), i))
        cands.sort(reverse=True)
        return [i for _p, i in cands] or [0]
    return fn


def _best_move_fn(emu: PyBoyEmulator):
    ranked = _ranked_move_fn(emu)

    def fn(idx):
        return ranked(idx)[0]
    return fn


class _BattleDriver:
    """Screen-aware battle driver for scripted fights the stock
    policy wedges on (see the leg header). Handles: intro/send-out
    text, the main menu (row AND column), the move menu, the party
    screen (B escapes an accidental opening; after a faint it
    picks the first healthy replacement), and stale menu bytes
    during animations (three identical signatures in a row means
    we are really in a text phase - advance with A)."""

    def __init__(self, emu: PyBoyEmulator):
        self.emu = emu
        self.mem = emu.pb.memory
        # Disable/rejection tracking, per battle (the stock
        # policy's probe45 lesson, ported): a confirmed move whose
        # PP has not moved when the move menu reopens on it was
        # rejected (disabled); twice, and it is blacklisted for
        # the rest of the battle.
        self._rejected = set()
        self._pending = None
        self._rejects = 0

    def sig(self):
        m = self.mem
        return (m[ADDR_MENU_TOP_Y], m[ADDR_MENU_ITEM],
                m[ADDR_MENU_COL])

    def fight_until_end(self, move_rank, switch_first=None,
                        max_actions: int = 400) -> bool:
        """move_rank: fn(active party idx) -> candidate move
        indices, best first."""
        emu, mem = self.emu, self.mem
        switched = switch_first is None
        actions = 0
        last_sig = None
        stale = 0
        while emu.read_state().in_battle and actions < max_actions:
            actions += 1
            my, item, col = self.sig()
            if (my, item, col) == last_sig:
                stale += 1
            else:
                stale = 0
                last_sig = (my, item, col)
            if not switched and my == 14:
                want_sp = emu.read_state().party[switch_first] \
                    .species_id
                for _try in range(3):
                    self._do_switch(switch_first)
                    emu.tick(30)
                    if mem[ADDR_BATTLE_MON_SPECIES] == want_sp:
                        break
                switched = True
                continue
            if stale >= 3:
                _press10(emu, "a")
                emu.tick(15)
                continue
            if my == 14:                       # main menu
                if col == 15:
                    _press10(emu, "left")
                elif item == 1:
                    _press10(emu, "up")
                else:
                    _press10(emu, "a")        # FIGHT
                continue
            if my == 12:                       # move menu
                active = self._active_party_idx()
                mon = emu.read_state().party[active]
                # Judge a pending confirmation (see __init__).
                if self._pending is not None:
                    pidx, ppp = self._pending
                    if pidx < len(mon.pp) and mon.pp[pidx] != ppp:
                        self._pending = None
                        self._rejects = 0
                    elif item == pidx + 1 \
                            and not emu.read_state().in_dialog:
                        self._rejects += 1
                        if self._rejects >= 2:
                            self._rejected.add(pidx)
                            self._pending = None
                            self._rejects = 0
                cands = [c for c in move_rank(active)
                         if c not in self._rejected]
                want = (cands[0] if cands else
                        move_rank(active)[0]) + 1
                if item == want:
                    if self._pending != (want - 1,
                                         mon.pp[want - 1]) \
                            and not emu.read_state().in_dialog:
                        self._pending = (want - 1, mon.pp[want - 1])
                        self._rejects = 0
                    _press10(emu, "a")
                else:
                    _press10(emu, "down" if item < want else "up")
                continue
            if my == 1:                        # party screen
                if (mem[ADDR_BATTLE_MON_HP] << 8
                        | mem[ADDR_BATTLE_MON_HP + 1]) == 0:
                    repl = self._replacement()
                    if item == repl:
                        _press10(emu, "a")
                    else:
                        _press10(emu, "down" if item < repl
                                 else "up")
                else:
                    _press10(emu, "b")
                continue
            _press10(emu, "a")
            emu.tick(15)
        over = not emu.read_state().in_battle
        if over:
            self._rejected = set()
            self._pending = None
            self._rejects = 0
        return over

    def _replacement(self) -> int:
        for i, mon in enumerate(self.emu.read_state().party):
            if mon.hp > 0:
                return i
        return 0

    def _active_party_idx(self) -> int:
        sp = self.mem[ADDR_BATTLE_MON_SPECIES]
        for i, mon in enumerate(self.emu.read_state().party):
            if mon.species_id == sp and mon.hp > 0:
                return i
        return 0

    def _do_switch(self, party_idx: int) -> None:
        emu = self.emu
        for _ in range(200):
            if not emu.read_state().in_battle:
                return
            my, item, col = self.sig()
            if my == 14:
                if col == 9:
                    _press10(emu, "right")
                elif item == 1:
                    _press10(emu, "up")
                else:
                    _press10(emu, "a")        # PKMN
                continue
            if my == 1:                        # party screen
                if item == party_idx:
                    _press10(emu, "a")
                    emu.tick(40)
                    return
                _press10(emu, "down" if item < party_idx else "up")
                continue
            _press10(emu, "a")
            emu.tick(15)


# -- wild catching (probe115/126 machinery) --------------------------

def _advance_battle(emu: PyBoyEmulator, rounds: int = 140):
    """Ride a wild battle to its next decision point: returns
    'menu' with the main battle menu live, or 'over'."""
    mem = emu.pb.memory

    def press(button):
        _raw(emu, button, hold=4, settle=22)
        emu.tick(8)

    for _ in range(rounds):
        st = emu.read_state()
        if not st.in_battle:
            return "over"
        if mem[ADDR_MENU_TOP_Y] == 12:
            press("b")
            emu.tick(40)
            continue
        if mem[ADDR_MENU_TOP_Y] == 14:
            r0 = mem[ADDR_MENU_ITEM]
            if r0 in (0, 1):
                press("down" if r0 == 0 else "up")
                emu.tick(35)
                if mem[ADDR_MENU_ITEM] != r0:
                    press("up" if r0 == 0 else "down")
                    emu.tick(35)
                    return "menu"
            if st.in_dialog:
                press("a")
                emu.tick(65)
            else:
                emu.tick(30)
            continue
        if st.in_dialog:
            press("a")
            emu.tick(65)
            continue
        emu.tick(30)
    return None


def _steer_byte(emu: PyBoyEmulator, getter, want, rounds: int = 14):
    mem = emu.pb.memory

    def press(button):
        _raw(emu, button, hold=4, settle=22)
        emu.tick(8)

    for _ in range(rounds):
        cur = getter()
        if cur == want:
            return True
        press("down" if cur < want else "up")
        emu.tick(30)
    return getter() == want


def _run_away(emu: PyBoyEmulator) -> bool:
    mem = emu.pb.memory
    for _ in range(12):
        if _advance_battle(emu) != "menu":
            return not emu.read_state().in_battle
        _steer_byte(emu, lambda: mem[ADDR_MENU_ITEM], 1)
        # RUN is the bottom-right cell: row 1, column 15
        for _ in range(14):
            if mem[ADDR_MENU_COL] == 15:
                break
            _raw(emu, "right", hold=4, settle=22)
            emu.tick(30)
        _raw(emu, "a", hold=4, settle=22)
        emu.tick(60)
        if _advance_battle(emu) == "over":
            return True
    return not emu.read_state().in_battle


def _throw_ball(emu: PyBoyEmulator) -> bool:
    mem = emu.pb.memory
    idx = _bag_index(emu, ITEM_POKE_BALL)
    if idx is None:
        return False
    before = _bag_count(emu, ITEM_POKE_BALL)
    if _advance_battle(emu) != "menu":
        return False
    _steer_byte(emu, lambda: mem[ADDR_MENU_ITEM], 1)
    for _ in range(14):                        # ITEM: row 1, col 9
        if mem[ADDR_MENU_COL] == 9:
            break
        _raw(emu, "left", hold=4, settle=22)
        emu.tick(30)
    _raw(emu, "a", hold=4, settle=22)
    emu.tick(70)
    for _ in range(160):
        st = emu.read_state()
        if not st.in_battle:
            return _bag_count(emu, ITEM_POKE_BALL) < before
        if _bag_count(emu, ITEM_POKE_BALL) < before:
            return True
        cur = mem[ADDR_MENU_SCROLL] + mem[ADDR_MENU_ITEM]
        if cur == idx:
            _raw(emu, "a", hold=4, settle=22)
            emu.tick(70)
            return True
        _raw(emu, "down" if cur < idx else "up", hold=4, settle=22)
        emu.tick(30)
    return _bag_count(emu, ITEM_POKE_BALL) < before


def _catch_epilogue(emu: PyBoyEmulator, stored_target: int) -> bool:
    """Ride the catch text and the first-catch Pokedex page (and,
    for a box catch, the transfer text), then decline the
    nickname: the yes/no cursor starts on YES, so a *verified*
    down + A picks NO (probe115's blind As typed a junk
    nickname; probe126's verified version is this). Completion
    is judged by the STORED count (party + box): the original
    version keyed on party size, which a box catch never moves -
    SendNewMonToBox files the mon without asking for a nickname
    (item_effects.asm), so only the party path shows the prompt
    this declines."""
    mem = emu.pb.memory

    def press(button):
        _raw(emu, button, hold=4, settle=22)
        emu.tick(8)

    for _ in range(160):
        st = emu.read_state()
        if not st.in_battle and _stored_count(emu) >= stored_target \
                and mem[ADDR_MENU_TOP_Y] == 0 and not st.in_dialog:
            return True
        if st.in_battle and mem[ADDR_MENU_TOP_Y] not in (0, 12, 14) \
                and mem[ADDR_MENU_ITEM] == 0:
            press("down")
            emu.tick(35)
            if mem[ADDR_MENU_ITEM] == 1:
                press("a")
                emu.tick(60)
                continue
        press("a")
        emu.tick(65)
    st = emu.read_state()
    return not st.in_battle and _stored_count(emu) >= stored_target


def _catch_any(emu: PyBoyEmulator, ex: Executor,
               want_species, max_encounters: int = 60):
    """Pace in place for wild encounters; run from everything
    that is not in want_species, throw balls at a want until it
    is caught. Returns the species id caught, or None.

    Generalised from probe115/126's single-species hunter
    (probe143): a SET of wanted species (a whole map's uncaught
    pool in one grass session), and box-aware success detection -
    the catch is proven by the stored count (party + box)
    ticking up inside the battle, so hunts keep working after
    the party fills at 6. Balls at full HP: nothing in this
    party can weaken a Lv2-6 wild without one-shotting it, and
    the Gen 1 formula gives ~1/3 per ball at catch rate 255
    regardless (item_effects.asm: X = 85 at full HP), ~3 balls
    per catch in practice. A battle that cannot be fled (a
    trainer who found us mid-hunt) is fought out with the
    battle driver instead of wedging the hunt."""
    mem = emu.pb.memory
    wants = set(want_species)
    for _enc in range(max_encounters):
        for i in range(150):
            if emu.read_state().in_battle:
                break
            emu.press("left" if i % 2 == 0 else "right")
            emu.tick(8)
        if not emu.read_state().in_battle:
            continue
        for _ in range(200):
            if mem[ADDR_ENEMY_SPECIES] != 0 \
                    and mem[ADDR_ENEMY_MAX_HP] != 0:
                break
            emu.tick(10)
        species = mem[ADDR_ENEMY_SPECIES]
        if species not in wants:
            if not _run_away(emu) and emu.read_state().in_battle:
                drv = _BattleDriver(emu)
                drv.fight_until_end(_ranked_move_fn(emu),
                                    max_actions=300)
                emu.wait_free(max_rounds=8, stable_needed=2)
            continue
        if _bag_count(emu, ITEM_POKE_BALL) == 0:
            return None
        if _catch_current(emu):
            return species
    return None


def _catch_current(emu: PyBoyEmulator) -> bool:
    """Throw balls at the wild mon already in front of us until
    it is caught (proven by the stored count ticking up, then
    the epilogue) or the battle ends some other way. Assumes a
    battle is live. Split out of _catch_any (probe143): a hunt
    interrupted mid-walk by exactly the species it came for
    should catch THAT mon, not run and re-roll the grass.

    The throw needs an explicit ride (probe143's first hunt
    wedged on exactly this): in the item effect's order -
    catch text, set the dex flag, ShowPokedexData, AddPartyMon /
    SendNewMonToBox, remove the ball - a first-of-species catch
    parks on the full-screen Pokedex page BEFORE the mon is
    filed. That page is not a dialog (in_dialog reads False) and
    only a plain A/B press dismisses it, while the stored count
    cannot tick until it is gone - so a caller that waits for
    the count before pressing anything deadlocks with the catch
    already made. The ride below presses A through everything
    that is not a live decision menu (the throw text, the
    shakes, the enemy's turn, the dex page) and stops the
    moment the main battle menu (menuY 14) is genuinely back,
    which is the broke-free case."""
    mem = emu.pb.memory

    def press_a():
        _raw(emu, "a", hold=4, settle=22)
        emu.tick(30)

    before = _stored_count(emu)
    for _attempt in range(20):
        if _bag_count(emu, ITEM_POKE_BALL) == 0:
            return False
        _throw_ball(emu)
        for _ride in range(240):
            if _stored_count(emu) > before:
                return _catch_epilogue(emu, before + 1)
            st = emu.read_state()
            if not st.in_battle:
                return False
            if mem[ADDR_MENU_TOP_Y] == 14 and not st.in_dialog:
                break                # broke free; menu is live
            if st.in_dialog or mem[ADDR_MENU_TOP_Y] not in (12, 14):
                press_a()
            else:
                emu.tick(30)
        else:
            return False             # the throw never resolved
        if _advance_battle(emu) == "over":
            return False
    return False


def _catch_species(emu: PyBoyEmulator, ex: Executor,
                   want_species: int, max_encounters: int = 40
                   ) -> bool:
    """Single-species wrapper over _catch_any (the probe115/126
    call shape the Vermilion scenes were built against)."""
    return _catch_any(emu, ex, {want_species},
                      max_encounters) == want_species


# -- bag item use: the menuY stage machine (probes 122-124) ---------

def _wait_menu_y(emu: PyBoyEmulator, want: int,
                 rounds: int = 80) -> bool:
    mem = emu.pb.memory
    for _ in range(rounds):
        if mem[ADDR_MENU_TOP_Y] == want:
            return True
        if emu.read_state().in_dialog:
            _press10(emu, "a")
            continue
        emu.tick(30)
    return mem[ADDR_MENU_TOP_Y] == want


def _steer_menu(emu: PyBoyEmulator, getter, want: int,
                rounds: int = 30) -> bool:
    mem = emu.pb.memory
    for _ in range(rounds):
        cur = getter()
        if cur == want:
            return True
        _press10(emu, "down" if cur < want else "up")
        emu.tick(20)
    return getter() == want


def _close_menus(emu: PyBoyEmulator) -> None:
    mem = emu.pb.memory
    for _ in range(12):
        if mem[ADDR_MENU_TOP_Y] == 0:
            return
        _press10(emu, "b")
        emu.tick(20)


def _use_bag_item_on_mon(emu: PyBoyEmulator, item_id: int,
                         mon_idx: int) -> bool:
    """START -> ITEM -> the item -> USE -> party member. Used for
    HM01: teaches the move into a free slot (the full-party TM
    delete flow is deliberately NOT ported - probe125's delete
    misclick cost TM28; see the leg report)."""
    mem = emu.pb.memory
    bag_idx = _bag_index(emu, item_id)
    if bag_idx is None:
        return False
    _press10(emu, "start")
    if not _wait_menu_y(emu, 2):
        return False
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 2):
        return False
    _press10(emu, "a")                         # ITEM
    if not _wait_menu_y(emu, 4):
        return False
    if not _steer_menu(
            emu, lambda: mem[ADDR_MENU_SCROLL] + mem[ADDR_MENU_ITEM],
            bag_idx):
        return False
    _press10(emu, "a")
    if not _wait_menu_y(emu, 11):
        return False
    _press10(emu, "a")                         # USE
    if not _wait_menu_y(emu, 1):
        return False
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], mon_idx):
        return False
    _press10(emu, "a")
    return True


# -- the scenes -------------------------------------------------------

def board_ss_anne(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Vermilion dock -> SS Anne -> the captain's cabin. The rival
    ambushes on 2F (the executor's rides fight him; his post-
    battle move-learning prompt plays out inside the ride's
    dialog handling, as in probe101). HARD GATE: HM01 must be in
    the bag before we leave - the ship departs for good."""
    world = ex.world
    dock = world.map_id("VermilionDock")
    a1 = world.map_id("SSAnne1F")
    a2 = world.map_id("SSAnne2F")
    capt = world.map_id("SSAnneCaptainsRoom")
    vc = world.map_id("VermilionCity")
    vpc = world.map_id("VermilionPokecenter")
    if ITEM_HM01 in _bag_items(emu):
        return True
    if not _ride_to(emu, ex, dock):
        return False
    if not _ride_to(emu, ex, a1):
        return False
    if not _ride_to(emu, ex, a2):
        return False
    # The 2F corridor is crossed on foot, not by ride: the rival
    # ambush mid-corridor can wedge the stock battle policy for
    # hundreds of presses (his Pidgeotto's Sand-Attack spam did
    # exactly that in the replay, probe140 - the policy ground
    # through 40 ride rounds without finishing the fight, while
    # _BattleDriver ended it in seconds). Walk legs + the driver
    # for the ambush, then step onto the captain's door warp.
    drv = _BattleDriver(emu)

    def walk_leg(map_id, tile):
        for _ in range(40):
            try:
                ex._walk_to(map_id, tile)
            except StuckError:
                pass
            st = emu.read_state()
            if st.in_battle:
                # Burst calls, not one long one: a single
                # multi-hundred-action call was seen to wedge
                # where 60-action bursts win the same fight
                # (probe140 - the driver's stale-signature state
                # interacts badly with the mid-battle
                # move-learning prompt's menus over long runs).
                drv.fight_until_end(_ranked_move_fn(emu),
                                    max_actions=120)
                emu.wait_free(max_rounds=8, stable_needed=2)
                continue
            if st.map_id != map_id:
                return True                    # a warp fired
            if (st.x, st.y) == tile:
                return True
            if st.in_dialog:
                emu.close_dialog(max_presses=1)
                emu.tick(20)
                continue
            # The rival's walk-up scene moves the player/NPCs on
            # rails for a while; let it settle before retrying.
            emu.wait_free(max_rounds=6, stable_needed=3)
        st = emu.read_state()
        return st.map_id != map_id or (st.x, st.y) == tile

    if not walk_leg(a2, (36, 4)):
        return False
    if emu.read_state().map_id != capt:
        if not _ride_to(emu, ex, capt):
            return False
    try:
        ex._walk_to(capt, (4, 3))
    except StuckError:
        return False
    emu.press("up")                            # face the captain
    emu.tick(20)
    emu.press("a")
    for _ in range(40):
        if ITEM_HM01 in _bag_items(emu):
            break
        if emu.read_state().in_dialog:
            emu.press("a")
        emu.tick(20)
    if ITEM_HM01 not in _bag_items(emu):
        return False
    if not _ride_to(emu, ex, a1):
        return False
    if not _ride_to(emu, ex, vc):
        return False
    if not _ride_to(emu, ex, vpc):
        return False
    return heal_at_pokecenter(emu, ex, vpc)


def buy_poke_balls(emu: PyBoyEmulator, ex: Executor,
                   want: int = 10, mart: int = None) -> bool:
    """Buy Poke Balls, verifying the bag count after every press.
    The shop's stages all read dialog=True and A-riding walks
    them (welcome -> item list, cursor starts on the ball ->
    quantity 1 -> confirm), so a press loop gated on the RAM
    count buys one ball per cycle without the runaway probe104
    hit when its loop-riding bought 31. Proven at Vermilion Mart
    (probe104 lineage); every mart interior shares the counter
    layout (clerk at (0,5), stance (2,5) facing left across the
    counter), so `mart` just retargets the ride - Viridian Mart
    uses it from probe143 on."""
    world = ex.world
    if mart is None:
        mart = world.map_id("VermilionMart")
    mem = emu.pb.memory
    if _bag_count(emu, ITEM_POKE_BALL) >= want:
        return True
    if not _ride_to(emu, ex, mart):
        return False
    try:
        ex._walk_to(mart, (2, 5))
    except StuckError:
        return False
    _step(emu, ex, "left")                     # face the clerk
    _press10(emu, "a")
    emu.tick(60)
    for _ in range(200):
        if _bag_count(emu, ITEM_POKE_BALL) >= want:
            break
        _press10(emu, "a")
        emu.tick(30)
    got = _bag_count(emu, ITEM_POKE_BALL)
    # exit: B back to the BUY/SELL/QUIT menu (menuY 1), QUIT = 2
    for _ in range(10):
        if mem[ADDR_MENU_TOP_Y] == 1:
            break
        _press10(emu, "b")
        emu.tick(20)
    _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 2)
    _press10(emu, "a")
    for _ in range(10):
        if not emu.read_state().in_dialog:
            break
        _press10(emu, "a")
        emu.tick(20)
    return got >= want


def catch_oddish_and_teach_cut(emu: PyBoyEmulator,
                               ex: Executor) -> bool:
    """Route 6's grass holds Oddish (the only Cut-capable family
    on the nearby routes in Red). Catch one, then teach it HM01
    Cut - it has a free move slot, so no deletion is involved.
    Wartortle cannot learn Cut in Gen 1; see the leg header."""
    world = ex.world
    r6 = world.map_id("Route6")
    st = emu.read_state()
    oddish_idx = _party_index_of(emu, SPECIES_ODDISH)
    if oddish_idx is None:
        if not _ride_to(emu, ex, r6):
            return False
        try:
            ex._walk_to(r6, (9, 15))           # the mid-route field
        except StuckError:
            return False
        if not _catch_species(emu, ex, SPECIES_ODDISH):
            return False
        oddish_idx = _party_index_of(emu, SPECIES_ODDISH)
        if oddish_idx is None:
            return False
    mon = emu.read_state().party[oddish_idx]
    if MOVE_CUT in mon.moves:
        return True
    if not _use_bag_item_on_mon(emu, ITEM_HM01, oddish_idx):
        return False
    for _ in range(60):
        mon = emu.read_state().party[oddish_idx]
        if MOVE_CUT in mon.moves:
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
            continue
        emu.tick(30)
    _close_menus(emu)
    mon = emu.read_state().party[oddish_idx]
    return MOVE_CUT in mon.moves


def catch_diglett(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Diglett in Diglett's Cave (via Route 11): Ground type,
    already knows Dig at cave levels. The Surge answer."""
    world = ex.world
    r11 = world.map_id("Route11")
    cave = world.map_id("DiglettsCave")
    if _party_index_of(emu, SPECIES_DIGLETT) is not None:
        return True
    if not _ride_to(emu, ex, r11):
        return False
    if not _ride_to(emu, ex, cave):
        return False
    return _catch_species(emu, ex, SPECIES_DIGLETT)


def collect_flash(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Route 2 Gate: Professor Oak's aide hands over HM05 Flash
    once the Pokedex holds 10 owned species (scripts/
    Route2Gate.asm: hOaksAideRequirement = 10, reward HM_FLASH).
    The aide stands at (1,4) facing left; talk to him from (2,4).
    Verified by HM05 appearing in the bag - the aide's refusal
    path (fewer than 10 species) leaves the bag untouched, so
    this returns False instead of a false positive. Idempotent:
    EVENT_GOT_HM05 makes later talks explanation-only, and the
    bag check short-circuits them anyway."""
    world = ex.world
    gate = world.map_id("Route2Gate")
    if ITEM_HM05 in _bag_items(emu):
        return True
    if _dex_owned_count(emu) < 10:
        return False
    if not _ride_to(emu, ex, gate):
        return False
    try:
        ex._walk_to(gate, (2, 4))
    except StuckError:
        return False
    emu.press("left")                        # face the aide
    emu.tick(20)
    _press10(emu, "a")
    for _ in range(120):
        if ITEM_HM05 in _bag_items(emu):
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(25)
    for _ in range(30):                      # ride out the talk
        if not emu.read_state().in_dialog:
            break
        _press10(emu, "a")
        emu.tick(20)
    return ITEM_HM05 in _bag_items(emu)


def _cut_tree_at_gym(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The gym door's tree stands at Vermilion City (15, 18),
    faced from (14, 18) (front tile 0x3D, pokered cut.asm). The
    world model marks the tree tile a static wall, so no journey
    into the gym yard exists - step the last tiles by hand."""
    world = ex.world
    vc = world.map_id("VermilionCity")
    mem = emu.pb.memory
    try:
        ex._walk_to(vc, (14, 18))
    except StuckError:
        return False
    _step(emu, ex, "right")                    # blocked: turns us
    if mem[ADDR_TILE_IN_FRONT] == 0x3D:
        oddish_idx = _party_index_of(emu, SPECIES_ODDISH)
        if oddish_idx is None:
            return False
        _press10(emu, "start")
        if not _wait_menu_y(emu, 2):
            return False
        if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 1):
            return False
        _press10(emu, "a")                     # POKeMON
        emu.tick(80)
        if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM],
                           oddish_idx):
            return False
        _press10(emu, "a")                     # mon submenu
        emu.tick(80)
        # A mon whose only field move is Cut lists it at item 0.
        _press10(emu, "a")                     # CUT
        emu.tick(150)
        for _ in range(40):
            if emu.read_state().in_dialog:
                emu.close_dialog(max_presses=1)
                continue
            emu.tick(30)
        _close_menus(emu)
    return _step(emu, ex, "right")             # through the gap


# Trash-can stands (can index -> stand tile, facing). Sight lines
# use the gym's real trainer ranges (scripts/VermilionGym.asm):
# Gentleman (9,6) LEFT range 3, Rocker (3,8) LEFT range 2,
# Sailor (0,10) RIGHT range 3 - every stand below is outside all
# three, reached through the corridor nodes.
_GYM_CANS = {0: (1, 7), 1: (1, 9), 2: (1, 11), 3: (3, 7),
             4: (3, 9), 5: (3, 11), 6: (5, 7), 7: (5, 9),
             8: (5, 11), 9: (7, 7), 10: (7, 9), 11: (7, 11),
             12: (9, 7), 13: (9, 9), 14: (9, 11)}
_GYM_STANDS = {
    0: ((1, 6), "down"), 1: ((0, 9), "right"), 2: ((1, 12), "up"),
    3: ((4, 7), "left"), 4: ((4, 9), "left"), 5: ((3, 12), "up"),
    6: ((5, 6), "down"), 7: ((5, 8), "down"), 8: ((5, 12), "up"),
    9: ((8, 7), "left"), 10: ((7, 8), "down"),
    11: ((7, 12), "up"), 12: ((9, 8), "up"), 13: ((9, 8), "down"),
    14: ((9, 12), "up"),
}
_GYM_NODES = {"A": (4, 12), "B": (4, 8), "C": (4, 6), "D": (0, 6),
              "E": (0, 9)}
_GYM_EDGES = {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"),
              ("C", "D"), ("D", "C"), ("D", "E"), ("E", "D")}
_GYM_STAND_NODE = {
    (1, 6): ("D", []), (0, 9): ("E", []),
    (1, 12): ("A", []), (3, 12): ("A", []), (5, 12): ("A", []),
    (7, 12): ("A", []), (9, 12): ("A", []),
    (4, 7): ("B", []), (4, 9): ("B", []), (5, 8): ("B", []),
    (7, 8): ("B", []), (9, 8): ("B", []),
    (6, 7): ("B", [(6, 8)]), (6, 9): ("B", [(6, 8)]),
    (8, 7): ("B", [(8, 8)]), (5, 6): ("C", []),
}


def beat_surge(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Heal, cut the gym tree, read both trash locks straight
    from RAM (wFirstLockTrashCanIndex / wSecondLockTrashCanIndex
    - with them known, the puzzle is two deterministic examines;
    note the game can even roll the second index onto the same
    can via its mask bug, probe136), then fight Surge with the
    Diglett switch line. Ends RAM-verified on the Thunder Badge.
    """
    world = ex.world
    vc = world.map_id("VermilionCity")
    vpc = world.map_id("VermilionPokecenter")
    gym = world.map_id("VermilionGym")
    mem = emu.pb.memory
    drv = _BattleDriver(emu)

    def clear_interruptions():
        for _ in range(40):
            st = emu.read_state()
            if st.in_battle:
                drv.fight_until_end(_ranked_move_fn(emu))
                emu.wait_free(max_rounds=8, stable_needed=2)
                continue
            if st.in_dialog:
                _press10(emu, "a")
                emu.tick(20)
                continue
            return True
        return False

    def walk_leg(map_id, tile):
        for _ in range(6):
            try:
                ex._walk_to(map_id, tile)
            except StuckError:
                pass
            st = emu.read_state()
            if st.map_id == map_id and (st.x, st.y) == tile \
                    and not st.in_battle:
                return True
            clear_interruptions()
        st = emu.read_state()
        return st.map_id == map_id and (st.x, st.y) == tile

    # Exit Diglett's Cave on foot if that's where we are: the
    # journey planner cannot chain LAST_MAP exits mid-route
    # (probe128), so step the ladder and the exit mat by hand.
    cave = world.map_id("DiglettsCave")
    entry = world.map_id("DiglettsCaveRoute11")
    if emu.read_state().map_id == cave:
        for _ in range(4):
            if emu.read_state().map_id != cave:
                break
            _step(emu, ex, "right")
            emu.tick(20)
    if emu.read_state().map_id == entry:
        walk_leg(entry, (2, 7))
        emu.wait_free(max_rounds=10, stable_needed=3)
    if not _ride_to(emu, ex, vc):
        return False
    if not _ride_to(emu, ex, vpc):
        return False
    if not heal_at_pokecenter(emu, ex, vpc):
        return False
    if not _cut_tree_at_gym(emu, ex):
        return False
    if not _ride_to(emu, ex, gym):
        return False

    cur_node = [None]

    def node_path(a, b):
        if a == b:
            return []
        from collections import deque
        prev = {a: None}
        q = deque([a])
        while q:
            n = q.popleft()
            if n == b:
                break
            for (x, y) in _GYM_EDGES:
                if x == n and y not in prev:
                    prev[y] = n
                    q.append(y)
        if b not in prev:
            return []
        path = []
        n = b
        while n != a:
            path.append(_GYM_NODES[n])
            n = prev[n]
        return path[::-1]

    def goto_stand(stand):
        node, extra = _GYM_STAND_NODE[stand]
        legs = []
        if cur_node[0] is None:
            legs.append(_GYM_NODES["A"])
            legs += node_path("A", node)
        else:
            legs += node_path(cur_node[0], node)
        legs += extra
        legs.append(stand)
        for leg in legs:
            if not walk_leg(gym, leg):
                return False
        cur_node[0] = node
        return True

    def examine_can(idx):
        stand, facing = _GYM_STANDS[idx]
        if not goto_stand(stand):
            return False
        _step(emu, ex, facing)
        if (emu.read_state().x, emu.read_state().y) != stand:
            return False
        _press10(emu, "a")
        emu.tick(60)
        clear_interruptions()
        return True

    first = mem[ADDR_FIRST_LOCK_CAN]
    if first not in _GYM_CANS:
        return False
    if not examine_can(first):
        return False
    second = mem[ADDR_SECOND_LOCK_CAN]
    if second not in _GYM_CANS:
        return False
    if second != first:
        if not examine_can(second):
            return False
    else:
        # The game's mask bug can put the second lock in the same
        # can (probe136) - examine it once more.
        stand, facing = _GYM_STANDS[first]
        _step(emu, ex, facing)
        _press10(emu, "a")
        emu.tick(60)
        clear_interruptions()
    legs = node_path(cur_node[0] or "A", "C")
    for leg in legs:
        if not walk_leg(gym, leg):
            return False
    if not walk_leg(gym, (5, 2)):
        return False
    # Talk to Surge and fight: switch to Diglett on turn one;
    # Diglett uses Dig, anything else its best-power move.
    _step(emu, ex, "up")
    _press10(emu, "a")
    for _ in range(40):
        if emu.read_state().in_battle:
            break
        _press10(emu, "a")
        emu.tick(20)
    if not emu.read_state().in_battle:
        return False
    diglett_idx = _party_index_of(emu, SPECIES_DIGLETT)
    ranked = _ranked_move_fn(emu)

    def surge_moves(idx):
        mon = emu.read_state().party[idx]
        rest = ranked(idx)
        if MOVE_DIG in mon.moves:
            dig = mon.moves.index(MOVE_DIG)
            return [dig] + [c for c in rest if c != dig]
        return rest

    drv.fight_until_end(surge_moves, switch_first=diglett_idx,
                        max_actions=500)
    for _ in range(40):
        if emu.read_state().in_dialog:
            _press10(emu, "a")
            emu.tick(20)
            continue
        emu.tick(25)
    return "Thunder" in emu.read_state().badges


def vermilion_leg(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The whole post-Cascade Vermilion chain, as proven by
    probes 100-138: board the SS Anne for HM01, stock up on
    balls, catch Oddish and teach it Cut, catch a Diglett, then
    take the Thunder Badge. Each stage verifies itself from RAM
    and is skipped when its effect is already in place."""
    if "Thunder" in emu.read_state().badges:
        return True
    if not board_ss_anne(emu, ex):
        return False
    if not buy_poke_balls(emu, ex):
        return False
    if not catch_oddish_and_teach_cut(emu, ex):
        return False
    if not catch_diglett(emu, ex):
        return False
    return beat_surge(emu, ex)


MAP_MT_MOON_B2F = 61


def take_fossil_scene(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Mt. Moon B2F's ladder pocket is reachable only through the
    Super Nerd's fossil chamber, and the two fossil objects block
    its only door: after his battle you must TAKE one (he takes the
    other) before the passage opens. Stand below the Helix Fossil,
    face it, take it, then prove the door by walking through the
    tile it stood on. Idempotent: with the fossil already taken the
    verification walk simply succeeds."""
    b2 = MAP_MT_MOON_B2F
    try:
        ex._walk_to(b2, (13, 7))
    except StuckError:
        return False
    emu.press("up")              # face the Helix Fossil at (13, 6)
    for i in range(40):
        st = emu.read_state()
        if not st.in_dialog and i > 2:
            break
        emu.press("a")
        emu.tick(10)
    emu.wait_free(max_rounds=6, stable_needed=6)
    try:
        ex._walk_to(b2, (13, 5))
        return True
    except StuckError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rom", required=True, help="path to a Pokemon Red ROM")
    ap.add_argument("--workdir", default="run",
                    help="where save states and screenshots go")
    ap.add_argument("--player", default="VLAD")
    ap.add_argument("--rival", default="GARY")
    ap.add_argument("--starter", default="squirtle", choices=STARTERS)
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--supervisor", default="adapter",
                    choices=["adapter", "stock", "off"],
                    help="LLM supervisor mode: adapter (trained LoRA, "
                         "default) / stock (bare base model) / off "
                         "(checklist only). Adapter failures degrade to "
                         "stock, then to the checklist - never a hard "
                         "failure.")
    ap.add_argument("--supervisor-adapter", default=str(
        Path(__file__).parent / "training_output" / "qwen_supervisor_v2"
        / "final"), help="LoRA adapter directory for --supervisor adapter")
    ap.add_argument("--supervisor-base", default="/home/vlad/models/"
                    "Qwen3.5-4B", help="base model directory")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    world = default_world()
    emu = PyBoyEmulator(args.rom)
    memory = EpisodicMemory()
    if args.supervisor == "adapter":
        planner = LLMPlanner(endpoint="inline", model="inline:adapter",
                             adapter=args.supervisor_adapter,
                             base_model=args.supervisor_base)
        print(f"supervisor: adapter {args.supervisor_adapter} "
              f"(inline; degrades to stock, then checklist)")
    elif args.supervisor == "stock":
        planner = LLMPlanner(endpoint="inline", model="inline:stock",
                             adapter=None, base_model=args.supervisor_base)
        print("supervisor: stock base model (inline)")
    else:
        planner = ChecklistPlanner()
        print("supervisor: off (checklist only)")
    ex = Executor(emu, world, planner, memory,
                  battle_policy=SmartFightPolicy())

    def shot(name: str) -> None:
        emu.screenshot(os.path.join(args.workdir, name))

    try:
        boot_and_intro(emu, args.player, args.rival)
        st = emu.read_state()
        print(f"intro done: {world.map_name(st.map_id)} ({st.x},{st.y})")
        shot("01-bedroom.png")

        trigger_oak_intercept(emu, ex)
        print("intercept done: in Oak's Lab")
        shot("02-lab-intercept.png")

        pick_starter(emu, ex, args.starter)
        st = emu.read_state()
        print(f"starter taken: {st.party}")
        shot("03-starter.png")

        memory.record_visit(MAP_OAKS_LAB)   # the lab step is done by hand
        t0 = time.time()
        f0 = emu.pb.frame_count
        handover_done = False
        brock_done = False
        fossil_done = False
        misty_done = False
        east_done = False
        surge_done = False
        for round_no in range(args.max_rounds):
            result = ex.run(max_goals=1)
            st = emu.read_state()
            print(f"round {round_no}: {result.status} "
                  f"at {world.map_name(st.map_id)} ({st.x},{st.y}) "
                  f"hp={st.party[0].hp if st.party else '-'} "
                  f"log={result.log}", flush=True)
            shot(f"round-{round_no}-{world.map_name(st.map_id)}.png")
            if result.status == "complete":
                break
            # The executor's delivery goal completes on *arrival* at
            # the Lab; the game wants the handover scene itself. Play
            # it once, right after the delivery round lands us there.
            delivered = any(
                e.get("goal") == "goto:40:None"
                and "Deliver the parcel" in e.get("reason", "")
                and "error" not in e and "problems" not in e
                for e in result.log)
            if delivered and not handover_done \
                    and emu.read_state().map_id == MAP_OAKS_LAB:
                handover_done = deliver_parcel_scene(emu, ex)
                print(f"pokedex handover scene: {handover_done}",
                      flush=True)
                shot("04-pokedex-handover.png")
            # Same pattern at the next spine step: the gym goal
            # completes on arrival, but the badge itself is an event.
            if not brock_done and st.map_id == MAP_PEWTER_GYM \
                    and "Boulder" not in st.badges:
                try:
                    brock_done = fight_brock(emu, ex)
                except Exception as e:
                    print(f"brock fight errored: {e}", flush=True)
                    brock_done = False
                print(f"brock fight: {brock_done}", flush=True)
                shot("05-boulder-badge.png")
            # Same pattern at Cerulean Gym: the cascade goal
            # completes on arrival; the badge is a scene.
            if not misty_done and st.map_id == MAP_CERULEAN_GYM \
                    and "Cascade" not in st.badges:
                try:
                    misty_done = fight_misty(emu, ex)
                except Exception as e:
                    print(f"misty fight errored: {e}", flush=True)
                    misty_done = False
                print(f"misty fight: {misty_done}", flush=True)
                shot("07-cascade-badge.png")
            # With the Cascade Badge held, the road east is a chain
            # of scenes (Bill's ticket, the robbed house, the tunnel)
            # that the checklist's naive goto cannot sequence - its
            # Cerulean->Vermilion route runs at the Saffron gates.
            # Play the chain once, the first round that ends in
            # Cerulean with the badge.
            if not east_done and "Cascade" in st.badges \
                    and st.map_id in (MAP_CERULEAN_GYM,
                                      world.map_id("CeruleanCity")):
                try:
                    east_done = eastward_from_cerulean(emu, ex)
                except Exception as e:
                    print(f"eastward leg errored: {e}", flush=True)
                    east_done = False
                print(f"eastward leg: {east_done}", flush=True)
                shot("08-vermilion.png")
            # Vermilion's own chain (SS Anne, Cut, Surge) is the
            # same shape: scenes the checklist cannot sequence.
            # Play it once, the first round that ends in Vermilion
            # with the Cascade Badge and no Thunder Badge.
            if not surge_done and "Cascade" in st.badges \
                    and "Thunder" not in st.badges \
                    and st.map_id in (world.map_id("VermilionCity"),
                                      world.map_id(
                                          "VermilionPokecenter")):
                try:
                    surge_done = vermilion_leg(emu, ex)
                except Exception as e:
                    print(f"vermilion leg errored: {e}", flush=True)
                    surge_done = False
                print(f"vermilion leg: {surge_done}", flush=True)
                shot("09-thunder-badge.png")
            # Mt. Moon B2F's exit ladder sits behind the fossil
            # chamber: the first round that ends on B2F, play the
            # fossil scene (it is on the way out regardless).
            if not fossil_done and st.map_id == MAP_MT_MOON_B2F:
                try:
                    fossil_done = take_fossil_scene(emu, ex)
                except Exception as e:
                    print(f"fossil scene errored: {e}", flush=True)
                    fossil_done = False
                print(f"fossil scene: {fossil_done}", flush=True)
                shot("06-fossil.png")
            if result.status in ("stuck", "blocked"):
                emu.wait_free()          # ride out any scripted scene
                if result.log:
                    goal_key = result.log[-1].get("goal")
                    if goal_key:
                        memory.forgive(goal_key)
        frames = emu.pb.frame_count - f0
        dt = time.time() - t0
        print(f"executor phase: {frames} frames in {dt:.1f}s "
              f"= {frames / max(dt, 0.01):.0f} fps "
              f"({frames / max(dt, 0.01) / 60:.0f}x real-time), "
              f"{emu.press_count} presses total")
        emu.save_state(os.path.join(args.workdir, "final.state"))
    finally:
        emu.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
