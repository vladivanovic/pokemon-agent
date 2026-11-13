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
ITEM_POTION = 0x14
ITEM_ANTIDOTE = 0x0B
ITEM_REPEL = 0x1E
MOVE_CUT = 15
MOVE_DIG = 91
MOVE_FLASH = 148
SPECIES_VOLTORB = 0x06
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
ADDR_MAP_PAL_OFFSET = 0xD35D     # wMapPalOffset: 6 in an unlit
                                 # dark cave, 0 once Flash is used
ADDR_MAX_MENU_ITEM = 0xCC28      # wMaxMenuItem: tells the PC main
                                 # menu (max 3) from Bill's (max 4)
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
        any_pp = None
        for i, mv in enumerate(mon.moves):
            if not mv:
                continue
            if i < len(mon.pp) and mon.pp[i] > 0 \
                    and any_pp is None:
                any_pp = i
            if mv not in STATUS_MOVE_IDS \
                    and i < len(mon.pp) and mon.pp[i] > 0:
                cands.append((_MOVE_POWER.get(mv, 30), i))
        cands.sort(reverse=True)
        # The fallback must NEVER be a 0-PP move: selecting
        # one just prints "no PP left" and reopens the menu
        # forever (the probe145 Exeggcute wedge). Fall back
        # to any PP'd move, status included; [0] only when
        # literally nothing has PP (the game force-Struggles
        # then and the menu never really opens).
        return [i for _p, i in cands] or \
            ([any_pp] if any_pp is not None else [0])
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
        # Faint-pick screen bookkeeping: how many steering
        # presses in a row failed to move the party cursor
        # (a text box is eating them) and where the cursor
        # last was.
        self._pick_stuck = 0
        self._pick_last_item = -1

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
            if stale >= 3 and not (
                    my == 1
                    and (mem[ADDR_BATTLE_MON_HP] << 8
                         | mem[ADDR_BATTLE_MON_HP + 1]) == 0):
                # Never blind-A the faint-replacement pick
                # screen: the cursor starts on the FAINTED
                # mon, A there only prints "There's no will
                # to fight!" and the sig never changes, so
                # the stale branch would re-fire forever and
                # the steering branch below never runs (the
                # gym14/grind wedge: a fainted trainee in a
                # wild battle froze the driver for 15+
                # drives). The party branch must steer.
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
                        self._pick_stuck = 0
                        _press10(emu, "a")
                    else:
                        # Steer to the replacement - but if
                        # the cursor will not move, a text
                        # box ("There's no will to fight!"
                        # after an accidental pick of the
                        # fainted mon, or the faint message
                        # itself) is eating the presses:
                        # dismiss it with A, then steer on.
                        if item == self._pick_last_item:
                            self._pick_stuck += 1
                        else:
                            self._pick_stuck = 0
                        self._pick_last_item = item
                        if self._pick_stuck >= 2:
                            self._pick_stuck = 0
                            _press10(emu, "a")
                        else:
                            _press10(emu,
                                     "down" if item < repl
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
            self._pick_stuck = 0
            self._pick_last_item = -1
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


def _ride_battling(emu: PyBoyEmulator, ex: Executor, map_id: int,
                   xy=None, rounds: int = 300) -> bool:
    """_ride_to with a battle escalation: ride executor slices,
    but whenever a slice leaves us in a battle, finish the fight
    with _BattleDriver bursts (the board_ss_anne pattern), then
    keep riding. The stock SmartFightPolicy wedges on
    trainer-dense routes - probe144's diagnostics watched it burn
    500-press battle caps repeatedly against Route 9's first
    trainers while Wartortle bled - so every ride in the Rock
    Tunnel leg goes through here."""
    saved = ex.planner
    ex.planner = _FixedPlanner(map_id, xy)
    drv = _BattleDriver(emu)
    try:
        for _ in range(rounds):
            st = emu.read_state()
            if st.map_id == map_id and not st.in_battle:
                return True
            if st.in_battle:
                for _burst in range(15):
                    if not emu.read_state().in_battle:
                        break
                    drv.fight_until_end(_ranked_move_fn(emu),
                                        max_actions=120)
                emu.wait_free(max_rounds=8, stable_needed=2)
                continue
            result = ex.run(max_goals=1)
            if result.log and result.log[-1].get("goal"):
                ex.memory.forgive(result.log[-1]["goal"])
            if emu.read_state().map_id == map_id:
                return True
        return emu.read_state().map_id == map_id
    finally:
        ex.planner = saved


def buy_mart_item(emu: PyBoyEmulator, ex: Executor, mart: int,
                  list_index: int, item_id: int, want: int,
                  stand=(2, 5), face: str = "left") -> int:
    """Buy up to `want` of one mart item; returns the final bag
    count. STAGE-GATED on the shop's menu signatures (calibrated
    live at Cerulean Mart, probe144; the first blind-steering
    version bought 10 extra Poke Balls because the cursor bytes
    change meaning per stage):
      menuY 1 = BUY/SELL/QUIT (steer item to 0 = BUY, press A;
                index 1 at this stage is SELL, not an item)
      menuY 4 = the item list (steer scroll+item to the item's
                list index, press A)
      menuY 8 = quantity/confirm (A-ride until the bag count
                ticks in RAM - the count is the ground truth)
    A unit of the WRONG item moving (the Poke Ball count is the
    index-0 canary) aborts the buy with what it has.

    `stand`/`face` locate the clerk (generalised in probe145
    from the Cerulean-only original for Celadon Mart, whose
    clerks stand elsewhere): the defaults keep the exact
    probe144-proven Cerulean behaviour (stand (2,5), step left
    into the clerk); other shops pass their own pair and are
    faced with _face instead (probe145's proven Celadon path -
    Mart 2F wants stand (5,5) facing up, Mart 4F (5,5) facing
    down). Callers buying on an upper floor must chain the
    floors first: a single journey ride into an elevator
    building spins (see celadon_leg)."""
    mem = emu.pb.memory
    have = _bag_count(emu, item_id)
    if have >= want:
        return have
    if not _ride_to(emu, ex, mart):
        return have
    try:
        ex._walk_to(mart, stand)
    except StuckError:
        return have
    if stand == (2, 5) and face == "left":
        _step(emu, ex, "left")                 # face the clerk
    else:
        _face(emu, face)
    _press10(emu, "a")
    emu.tick(60)
    balls0 = _bag_count(emu, ITEM_POKE_BALL)

    def wait_menu_y(want_y, rounds=60):
        for _ in range(rounds):
            if mem[ADDR_MENU_TOP_Y] == want_y:
                return True
            emu.tick(20)
        return mem[ADDR_MENU_TOP_Y] == want_y

    if not wait_menu_y(1):                     # BUY/SELL/QUIT
        return _bag_count(emu, item_id)
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 0):
        return _bag_count(emu, item_id)
    _press10(emu, "a")                         # BUY -> item list
    for _cycle in range(want * 2 + 2):
        if _bag_count(emu, item_id) >= want:
            break
        for _ in range(60):                    # the item list:
            if mem[ADDR_MENU_TOP_Y] == 4:      # post-buy text
                break                          # needs an A to
            _press10(emu, "a")                 # reopen it
            emu.tick(20)
        if mem[ADDR_MENU_TOP_Y] != 4:
            break
        if not _steer_menu(
                emu, lambda: mem[ADDR_MENU_SCROLL]
                + mem[ADDR_MENU_ITEM], list_index):
            break
        before = _bag_count(emu, item_id)
        _press10(emu, "a")                     # select the item
        ok = False
        for _ in range(40):                    # quantity/confirm
            _press10(emu, "a")
            emu.tick(25)
            if _bag_count(emu, item_id) > before:
                ok = True
                break
            if item_id != ITEM_POKE_BALL and \
                    _bag_count(emu, ITEM_POKE_BALL) != balls0:
                ok = False                     # WRONG ITEM moved
                break
        if not ok:
            break
        emu.tick(40)                           # back to the list
    for _ in range(10):                        # exit the shop
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
    return _bag_count(emu, item_id)


# Cerulean Mart's shelf, in list order (probe144, verified live
# against the game's own list menu). buy_mart_item takes the
# index from the caller so other marts can pass their own.
CERULEAN_MART_LIST = [ITEM_POKE_BALL, ITEM_POTION, ITEM_REPEL,
                      ITEM_ANTIDOTE, 0x0C, 0x0E, 0x0F]


def _field_rank(emu: PyBoyEmulator, mon_idx: int,
                move_id: int) -> int:
    """Submenu index of a field move in a mon's party submenu:
    its field moves in its own slot order (FIELD_MOVE_IDS)."""
    from agent.executor import FIELD_MOVE_IDS
    mon = emu.read_state().party[mon_idx]
    slot = mon.moves.index(move_id)
    return sum(1 for mv in mon.moves[:slot] if mv in FIELD_MOVE_IDS)


def teach_flash(emu: PyBoyEmulator, mon_idx: int) -> bool:
    """Teach HM05 Flash to the party mon at `mon_idx`. FREE SLOT
    ONLY: the full-moveset delete flow is deliberately not used
    (this save line already lost TM28 Dig to a move-delete
    misclick in the Anne leg), so a mon with four moves returns
    False untouched. Verified from RAM afterwards.

    The pupil matters: the leg brief's original plan, Oddish,
    CANNOT learn Flash in Gen 1 (its base-stats learnset in
    pokered lists Cut, not Flash - the game's own CanLearnTM
    screen reads NOT ABLE for the whole arriving party). The
    carrier is Voltorb, caught for the purpose on Route 10."""
    mon = emu.read_state().party[mon_idx]
    if MOVE_FLASH in mon.moves:
        return True
    if len([m for m in mon.moves if m]) >= 4:
        return False
    if not _use_bag_item_on_mon(emu, ITEM_HM05, mon_idx):
        return False
    for _ in range(60):
        mon = emu.read_state().party[mon_idx]
        if MOVE_FLASH in mon.moves:
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
            continue
        emu.tick(30)
    _close_menus(emu)
    mon = emu.read_state().party[mon_idx]
    return MOVE_FLASH in mon.moves


def use_flash(emu: PyBoyEmulator) -> bool:
    """Use Flash from the party submenu of whichever mon knows
    it, and prove it in RAM: wMapPalOffset goes 6 (dark) -> 0
    (lit). The menu driver is PyBoyEmulator.field_flash (the
    shared _field_move behind field_cut)."""
    st = emu.read_state()
    holder = next((i for i, m in enumerate(st.party)
                   if MOVE_FLASH in m.moves), None)
    if holder is None:
        return False
    sub = _field_rank(emu, holder, MOVE_FLASH)
    if not emu.field_flash(holder, sub):
        return False
    for _ in range(40):
        if emu.pb.memory[ADDR_MAP_PAL_OFFSET] == 0:
            break
        if emu.read_state().in_dialog:
            emu.close_dialog(max_presses=1)
        emu.tick(30)
    return emu.pb.memory[ADDR_MAP_PAL_OFFSET] == 0


def _use_bag_item(emu: PyBoyEmulator, item_id: int) -> bool:
    """Use a no-target bag item (Repel): START -> ITEM -> the
    item -> USE, ride the effect text, close the menus. Verified
    by the bag count dropping exactly one."""
    mem = emu.pb.memory
    idx = _bag_index(emu, item_id)
    if idx is None:
        return False
    before = _bag_count(emu, item_id)
    _press10(emu, "start")
    if not _wait_menu_y(emu, 2):
        return False
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 2):
        return False
    _press10(emu, "a")                         # ITEM
    if not _wait_menu_y(emu, 4):
        return False
    if not _steer_menu(
            emu, lambda: mem[ADDR_MENU_SCROLL]
            + mem[ADDR_MENU_ITEM], idx):
        return False
    _press10(emu, "a")
    if not _wait_menu_y(emu, 11):
        return False
    _press10(emu, "a")                         # USE
    for _ in range(40):
        if not emu.read_state().in_dialog:
            break
        _press10(emu, "a")
        emu.tick(25)
    _close_menus(emu)
    return _bag_count(emu, item_id) == before - 1


def use_repel(emu: PyBoyEmulator) -> bool:
    """Burn one Repel (100 steps of no wild encounters while the
    lead outlevels the locals - the tunnel wilds top out far
    below the party lead)."""
    return _use_bag_item(emu, ITEM_REPEL)


def _field_heal_lead(emu: PyBoyEmulator) -> None:
    """Potion the party lead when it drops under 45%: the tunnel
    crossing is an attrition problem (~a dozen forced battles),
    and the bag machinery is item-generic. A fainted lead is
    left alone - Potions do not revive."""
    st = emu.read_state()
    if not st.party:
        return
    lead = st.party[0]
    if lead.hp == 0 or lead.hp_frac >= 0.45:
        return
    if _bag_count(emu, ITEM_POTION) <= 0:
        return
    if _use_bag_item_on_mon(emu, ITEM_POTION, 0):
        for _ in range(30):
            if not emu.read_state().in_dialog:
                break
            _press10(emu, "a")
            emu.tick(25)
        _close_menus(emu)


def _fight_now(emu: PyBoyEmulator, drv: _BattleDriver) -> None:
    """Finish the current battle with driver bursts, then
    top up the lead."""
    for _burst in range(15):
        if not emu.read_state().in_battle:
            break
        drv.fight_until_end(_ranked_move_fn(emu), max_actions=120)
    emu.wait_free(max_rounds=8, stable_needed=2)
    _field_heal_lead(emu)


def _box_species(emu: PyBoyEmulator) -> list:
    """Species ids in Box 1, in box order (0xFF-terminated)."""
    mem = emu.pb.memory
    out = []
    for i in range(20):
        s = mem[ADDR_BOX_SPECIES + i]
        if s == 0xFF:
            break
        out.append(s)
    return out


def catch_voltorb(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Catch a Voltorb on Route 10's grass (the patch at the
    Route 9 junction; hunt tile (8,7)) - the Flash carrier.
    Voltorb is ~35% of that grass at Lv14-17 with a 190 catch
    rate; with a full party it files to Box 1, which is where
    the PC swap picks it up. Idempotent: already boxed or in
    the party means done. Ends back at the Rock Tunnel
    Pokecenter."""
    world = ex.world
    rtpc = world.map_id("RockTunnelPokecenter")
    r10 = world.map_id("Route10")
    grass = (8, 7)
    if SPECIES_VOLTORB in _box_species(emu) or \
            _party_index_of(emu, SPECIES_VOLTORB) is not None:
        return True
    if not _ride_battling(emu, ex, rtpc):
        return False
    if not heal_at_pokecenter(emu, ex, rtpc):
        return False
    if not _ride_battling(emu, ex, r10):
        return False
    mem = emu.pb.memory
    for _try in range(60):
        st = emu.read_state()
        if st.in_battle:
            sp = mem[ADDR_ENEMY_SPECIES]
            if sp == SPECIES_VOLTORB and _catch_current(emu):
                break
            if not _run_away(emu) and emu.read_state().in_battle:
                _BattleDriver(emu).fight_until_end(
                    _ranked_move_fn(emu), max_actions=300)
                emu.wait_free(max_rounds=8, stable_needed=2)
            continue
        if (st.x, st.y) == grass and st.map_id == r10:
            break
        try:
            ex._walk_to(r10, grass)
        except StuckError:
            pass
        if emu.read_state().in_dialog:
            emu.close_dialog(max_presses=2)
    if SPECIES_VOLTORB not in _box_species(emu) and \
            _party_index_of(emu, SPECIES_VOLTORB) is None:
        _catch_any(emu, ex, {SPECIES_VOLTORB}, max_encounters=40)
    got = SPECIES_VOLTORB in _box_species(emu) or \
        _party_index_of(emu, SPECIES_VOLTORB) is not None
    if not got:
        return False
    return _ride_battling(emu, ex, rtpc)


def _open_pc(emu: PyBoyEmulator, ex: Executor, pc_map: int) -> bool:
    """Walk to the PC stance and boot the PC. The hidden event
    sits at (13,3) facing UP in every Pokecenter, so the stance
    is (13,4) facing up; if that does not take, scan the tiles
    around the room's right side. Success = the PC answered
    (dialog or a menu cursor up)."""
    stances = [(13, 4, "up"), (13, 2, "up"), (12, 2, "up"),
               (12, 3, "right"), (11, 3, "right"), (13, 3, "up")]
    for (x, y, face) in stances:
        try:
            ex._walk_to(pc_map, (x, y))
        except StuckError:
            continue
        emu.press(face)                        # turn toward the PC
        emu.tick(20)
        _press10(emu, "a")
        for _ in range(30):
            emu.tick(20)
            st = emu.read_state()
            if st.in_dialog or \
                    emu.pb.memory[ADDR_MENU_TOP_Y] != 0:
                return True
        _close_menus(emu)
    return False


def _pc_press_a_ride(emu: PyBoyEmulator, done, rounds: int = 120):
    """Press A through PC text until `done()` reads True."""
    for _ in range(rounds):
        if done():
            return True
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(25)
    return done()


def pc_swap(emu: PyBoyEmulator, ex: Executor, pc_map: int,
            deposit_species: int, withdraw_species: int) -> bool:
    """Bill's PC swap at a Pokecenter: deposit `deposit_species`
    from the party, withdraw `withdraw_species` from Box 1.
    Menu stages are told apart by wMaxMenuItem (the PC main
    menu caps at 3, Bill's menu at 4 - both share menuY 2, the
    trap a stage-blind driver falls into). Every effect is
    verified in RAM (party/box membership and sizes); the swap
    ends with a heal so the withdrawn mon is ready. Proven live
    in probe144 (Spearow out, Voltorb in; the story party's
    first three slots were asserted untouched there)."""
    if _party_index_of(emu, withdraw_species) is not None and \
            _party_index_of(emu, deposit_species) is None:
        return True                            # already swapped
    dep_idx = _party_index_of(emu, deposit_species)
    if dep_idx is None or withdraw_species not in _box_species(emu):
        return False
    if not _open_pc(emu, ex, pc_map):
        return False
    mem = emu.pb.memory

    def sig():
        return (mem[ADDR_MENU_TOP_Y], mem[ADDR_MENU_ITEM],
                mem[ADDR_MAX_MENU_ITEM])

    def wait_sig(pred, rounds=120, ride_a=False):
        for _ in range(rounds):
            if pred(sig()):
                return True
            if ride_a and emu.read_state().in_dialog:
                _press10(emu, "a")
            emu.tick(25)
        return pred(sig())

    def bills_menu():
        return wait_sig(lambda s: s[2] == 4, ride_a=True)

    if not bills_menu():
        if not wait_sig(lambda s: s[2] == 3):
            return False
        if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 0):
            return False
        _press10(emu, "a")                     # BILL'S PC
        if not bills_menu():
            return False

    def pick_from_list(index, done):
        emu.tick(60)
        if not _steer_menu(
                emu, lambda: mem[ADDR_MENU_SCROLL]
                + mem[ADDR_MENU_ITEM], index):
            return False
        _press10(emu, "a")
        emu.tick(90)
        if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 0):
            return False
        _press10(emu, "a")
        return _pc_press_a_ride(emu, done, rounds=160)

    n_party = len(emu.read_state().party)
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 1):
        return False
    _press10(emu, "a")                         # DEPOSIT
    if not pick_from_list(
            dep_idx,
            lambda: len(emu.read_state().party) == n_party - 1):
        return False
    if deposit_species not in _box_species(emu):
        return False
    if not bills_menu():
        return False
    box = _box_species(emu)
    v_idx = box.index(withdraw_species)
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 0):
        return False
    _press10(emu, "a")                         # WITHDRAW
    if not pick_from_list(
            v_idx,
            lambda: len(emu.read_state().party) == n_party):
        return False
    if _party_index_of(emu, withdraw_species) is None:
        return False
    if not bills_menu():
        return False
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 4):
        return False
    _press10(emu, "a")                         # SEE YA!
    if not wait_sig(lambda s: s[2] == 3, ride_a=True):
        return False
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 3):
        return False
    _press10(emu, "a")                         # LOG OFF
    emu.tick(90)
    _close_menus(emu)
    emu.wait_free(max_rounds=10, stable_needed=3)
    return heal_at_pokecenter(emu, ex, pc_map)


def _walk_tile_leg(emu: PyBoyEmulator, ex: Executor, map_id: int,
                   tile, ok_maps, repel_every: int = 0,
                   max_rounds: int = 240) -> str:
    """Tile-level walk inside one map toward `tile`, for
    interiors where journey-level rides fail. Returns "warp"
    (a warp carried us to a map in `ok_maps`), "tile" (standing
    on the target), "blackout" (a warp to an unexpected map -
    the party wiped and respawned; reported as its own outcome
    instead of masquerading as leg success), "sealed" or
    "stuck".

    The blocked set is PERSISTENT for the whole leg: the
    executor's own learned-wall forgetting kept erasing Rock
    Tunnel B1F's refusals and replanning through them, orbiting
    the same trainers while the party bled (probe144 attempts
    1-2). A refused tile stays refused here; battles go to the
    driver; if the persistent set seals the map, the fallback
    steps beside a sealing tile and talks to whatever stands
    on it (a trainer battle frees the corridor's sight
    geometry). The lead is Potion-topped between fights, and
    Repel upkeep runs on a round cadence when repel_every > 0.
    """
    world = ex.world
    drv = _BattleDriver(emu)
    blocked = set()
    engaged = set()
    for rnd in range(max_rounds):
        st = emu.read_state()
        if st.map_id != map_id:
            if st.map_id in ok_maps:
                return "warp"
            return "blackout"
        if (st.x, st.y) == tile:
            return "tile"
        if st.in_battle:
            _fight_now(emu, drv)
            continue
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
            continue
        _field_heal_lead(emu)
        if repel_every and rnd > 0 and rnd % repel_every == 0:
            use_repel(emu)
        path = world.tile_path(
            map_id, (st.x, st.y), tile,
            blocked=frozenset(blocked),
            capabilities=st.capabilities(),
            cut_done=frozenset(ex._cut_done.get(map_id, ())))
        if path is None:
            cands = [t for t in blocked if t not in engaged]
            if not cands:
                return "sealed"
            tgt = cands[0]
            engaged.add(tgt)
            stood = False
            for n in [(tgt[0] + dx, tgt[1] + dy)
                      for dx, dy in ((0, 1), (0, -1),
                                     (1, 0), (-1, 0))]:
                p2 = world.tile_path(
                    map_id, (st.x, st.y), n,
                    blocked=frozenset(blocked),
                    capabilities=st.capabilities())
                if p2:
                    try:
                        ex._follow_tiles(map_id, p2)
                        stood = True
                        break
                    except StuckError:
                        if ex._blocked_tile is not None:
                            blocked.add(ex._blocked_tile)
                            ex._blocked_tile = None
            if not stood:
                continue
            st = emu.read_state()
            if st.map_id != map_id:
                continue
            dx, dy = tgt[0] - st.x, tgt[1] - st.y
            btn = {(0, -1): "up", (0, 1): "down",
                   (-1, 0): "left", (1, 0): "right"}.get((dx, dy))
            if btn is None:
                continue
            emu.press(btn)
            emu.tick(15)
            _press10(emu, "a")
            emu.tick(40)
            if emu.read_state().in_battle:
                _fight_now(emu, drv)
            elif emu.read_state().in_dialog:
                emu.close_dialog(max_presses=4)
            continue
        try:
            ex._follow_tiles(map_id, path)
        except StuckError:
            if ex._blocked_tile is not None:
                blocked.add(ex._blocked_tile)
                ex._blocked_tile = None
            if emu.read_state().in_battle:
                _fight_now(emu, drv)
            emu.tick(10)
    st = emu.read_state()
    if st.map_id == map_id and (st.x, st.y) == tile:
        return "tile"
    if st.map_id in ok_maps:
        return "warp"
    return "stuck"


def _step_walk_to(emu: PyBoyEmulator, ex: Executor, map_id: int,
                  tile, max_rounds: int = 300) -> bool:
    """Walk to `tile` one hand-rolled step at a time (bounded
    presses, each step proven by the RAM coordinates). The
    outdoor counterpart of _walk_tile_leg: Executor._follow_tiles
    wedged silently on Route 10's south corridor in two probe144
    runs - its inner loop can spin without a StuckError when a
    press keeps changing state without reaching the target (the
    Super Nerd engagement at (14,64) is the local suspect) - so
    this corridor is stepped manually. A blackout is detected
    by the money drop and reported as failure."""
    from agent.executor import _DELTA_TO_BUTTON  # press per delta
    world = ex.world
    drv = _BattleDriver(emu)
    blocked = set()
    money_prev = emu.read_state().money
    for _rnd in range(max_rounds):
        st = emu.read_state()
        if st.money < money_prev:
            return False                       # blackout
        money_prev = st.money
        if st.map_id != map_id:
            return False
        if (st.x, st.y) == tile:
            return True
        if st.in_battle:
            _fight_now(emu, drv)
            continue
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
            continue
        p = world.tile_path(
            map_id, (st.x, st.y), tile,
            blocked=frozenset(blocked),
            capabilities=st.capabilities())
        if p is None or len(p) < 2:
            return False
        nxt = p[1]
        btn = _DELTA_TO_BUTTON[(nxt[0] - st.x, nxt[1] - st.y)]
        before = (st.x, st.y)
        for _att in range(40):
            emu.press(btn)
            emu.tick(28)
            st2 = emu.read_state()
            if (st2.map_id != map_id or st2.in_battle
                    or st2.in_dialog):
                break
            if (st2.x, st2.y) != before:
                break
        st2 = emu.read_state()
        if (st2.map_id == map_id and (st2.x, st2.y) == before
                and not st2.in_battle and not st2.in_dialog):
            blocked.add(nxt)
    st = emu.read_state()
    return st.map_id == map_id and (st.x, st.y) == tile


def traverse_rock_tunnel(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Rock Tunnel, entered from the Route 10 (north) side:
    Flash on entry (verified in wMapPalOffset), Repel up, then
    the ladder zigzag the game's elevation rules force -
    1F (37,3) down to B1F (33,25), B1F (27,3) up to 1F (5,3),
    1F (17,11) down to B1F (23,11), B1F (3,3) up to 1F south
    (37,17), out the south exit (15,33) - and the stepped walk
    down Route 10's south segment into Lavender Town.

    The world model used to plan a single B1F walk along the
    south corridor; that path crosses a forbidden tile pair
    ($20 -> $05 at (14,28) -> (13,28)) and the game refuses the
    step. The model now carries TilePairCollisionsLand (see
    agent/world.py), so its tile paths ARE the zigzag - this
    scene drives them leg by leg with the persistent-blocked
    walker because the tunnel's STAY trainers stand on the
    through-path and journey-level replans orbit them.

    Darkness is NOT capability-gated in the model: wMapPalOffset
    is palette-only in the engine and RAM navigation does not
    read the screen, so the agent could cross blind. Canonical
    play is enforced here instead - the scene lights the cave
    on entry and verifies it before walking."""
    world = ex.world
    rt1 = world.map_id("RockTunnel1F")
    b1 = world.map_id("RockTunnelB1F")
    r10 = world.map_id("Route10")
    lav = world.map_id("LavenderTown")
    st = emu.read_state()
    if st.map_id == lav:
        return True
    if st.map_id == r10 and st.y > 40:
        pass                                   # already south:
        # fall through to the hop below via the leg chain check
    else:
        if st.map_id not in (rt1, b1):
            if not _ride_battling(emu, ex, rt1):
                return False
        st = emu.read_state()
        if st.map_id == rt1 and (st.x, st.y) == (15, 3):
            if not use_flash(emu):
                return False
            if not use_repel(emu):
                return False
        if emu.read_state().map_id == rt1:
            r = _walk_tile_leg(emu, ex, rt1, (37, 3), {b1})
            if r != "warp":
                return False
        if emu.read_state().map_id == b1:
            # From the (33,25) landing the pair-legal road runs
            # north to the (27,3) ladder, not west along B1F.
            r = _walk_tile_leg(emu, ex, b1, (27, 3), {rt1},
                               repel_every=15)
            if r != "warp":
                return False
        if emu.read_state().map_id == rt1:
            r = _walk_tile_leg(emu, ex, rt1, (17, 11), {b1})
            if r != "warp":
                return False
        if emu.read_state().map_id == b1:
            r = _walk_tile_leg(emu, ex, b1, (3, 3), {rt1},
                               repel_every=15)
            if r != "warp":
                return False
        if emu.read_state().map_id == rt1:
            use_repel(emu)
            r = _walk_tile_leg(emu, ex, rt1, (15, 33), {r10})
            if r != "warp":
                return False
    # Route 10 south segment -> step off the south edge.
    st = emu.read_state()
    if st.map_id != r10:
        return emu.read_state().map_id == lav
    use_repel(emu)
    if not _step_walk_to(emu, ex, r10, (8, 71)):
        return False
    drv = _BattleDriver(emu)
    for _ in range(30):
        st = emu.read_state()
        if st.map_id == lav:
            return True
        if st.in_battle:
            _fight_now(emu, drv)
            continue
        emu.press("down")
        emu.tick(45)
    return emu.read_state().map_id == lav


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


def rock_tunnel_leg(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The whole Rock Tunnel chain, as proven by probe144:
    ride the long way round to Cerulean (Saffron's gates are
    shut - the road is the Underground Path), stock up, ride
    Route 9/10 to the tunnel-side Pokecenter, catch a Voltorb
    (the Flash carrier - Oddish cannot learn Flash, see
    teach_flash), swap it into the party for Spearow at Bill's
    PC, teach it Flash, ride BACK to Cerulean for Repels and
    more Potions (the first crossing attempt blacked out on
    pure HP attrition; the resupply is what makes the tunnel
    survivable), heal at the tunnel-side center, traverse,
    and heal in Lavender. Each stage verifies itself from RAM
    and is skipped when its effect is already in place."""
    world = ex.world
    vpc = world.map_id("VermilionPokecenter")
    cerpc = world.map_id("CeruleanPokecenter")
    cermart = world.map_id("CeruleanMart")
    rtpc = world.map_id("RockTunnelPokecenter")
    lavpc = world.map_id("LavenderPokecenter")
    st = emu.read_state()
    if st.map_id == world.map_id("LavenderTown") and \
            any(MOVE_FLASH in m.moves for m in st.party):
        return heal_at_pokecenter(emu, ex, lavpc)
    if ITEM_HM05 not in _bag_items(emu):
        return False                     # collect_flash's stage
    if not _ride_battling(emu, ex, vpc):
        return False
    if not heal_at_pokecenter(emu, ex, vpc):
        return False
    if not _ride_battling(emu, ex, cerpc):
        return False
    if not heal_at_pokecenter(emu, ex, cerpc):
        return False
    if not buy_poke_balls(emu, ex, want=15, mart=cermart):
        return False
    if buy_mart_item(emu, ex, cermart,
                     CERULEAN_MART_LIST.index(ITEM_POTION),
                     ITEM_POTION, 2) < 2:
        return False
    if buy_mart_item(emu, ex, cermart,
                     CERULEAN_MART_LIST.index(ITEM_ANTIDOTE),
                     ITEM_ANTIDOTE, 2) < 2:
        return False
    if not catch_voltorb(emu, ex):
        return False
    if not pc_swap(emu, ex, rtpc, SPECIES_SPEAROW,
                   SPECIES_VOLTORB):
        return False
    v_idx = _party_index_of(emu, SPECIES_VOLTORB)
    if v_idx is None:
        return False
    if not teach_flash(emu, v_idx):
        return False
    # The attrition resupply: Repels kill the wild-battle half
    # of the tunnel's HP toll, Potions cover the trainer half.
    if not _ride_battling(emu, ex, cerpc):
        return False
    if not heal_at_pokecenter(emu, ex, cerpc):
        return False
    if buy_mart_item(emu, ex, cermart,
                     CERULEAN_MART_LIST.index(ITEM_REPEL),
                     ITEM_REPEL, 3) < 3:
        return False
    if buy_mart_item(emu, ex, cermart,
                     CERULEAN_MART_LIST.index(ITEM_POTION),
                     ITEM_POTION, 5) < 5:
        return False
    if not _ride_battling(emu, ex, rtpc):
        return False
    if not heal_at_pokecenter(emu, ex, rtpc):
        return False
    if not traverse_rock_tunnel(emu, ex):
        return False
    if not _ride_to(emu, ex, lavpc):
        return False
    return heal_at_pokecenter(emu, ex, lavpc)


# -- Celadon leg (probes 145-147): the road west, the Fire
# counter, Erika ---------------------------------------------------
#
# Faithful ports of the probe-proven segments, assembled into
# the WINNING route (catch -> swap -> shop -> stone -> grind ->
# gym): the five blackouts and the detours that taught the
# numbers are in docs/supervisor-training.md S13, not replayed
# here. The assembled leg has NOT been cold-replayed end to end.

ITEM_SUPER_POTION = 0x13
ITEM_GREAT_BALL = 0x03
ITEM_FRESH_WATER = 0x3C
ITEM_FIRE_STONE = 0x20
SPECIES_GROWLITHE = 0x21
SPECIES_ARCANINE = 0x14
SPECIES_BLASTOISE = 0x1C
EVENT_BEAT_ERIKA = 425
ADDR_ENEMY_TYPE1 = 0xCFEA
ADDR_ENEMY_TYPE2 = 0xCFEB
ADDR_BATTLE_MON_TYPE1 = 0xD019
ADDR_BATTLE_MON_TYPE2 = 0xD01A

# Damaging moves: id -> (base power, type). Generated once from
# pokered data/moves/moves.asm (probe145's table); status moves
# are absent, which doubles as the damaging-move test.
_MOVE_INFO = {
    1: (40, "NORMAL"),  2: (50, "NORMAL"),  3: (15, "NORMAL"),  4: (18, "NORMAL"),
    5: (80, "NORMAL"),  6: (40, "NORMAL"),  7: (75, "FIRE"),  8: (75, "ICE"),
    9: (75, "ELECTRIC"),  10: (40, "NORMAL"),  11: (55, "NORMAL"),  12: (1, "NORMAL"),
    13: (80, "NORMAL"),  15: (50, "NORMAL"),  16: (40, "NORMAL"),  17: (35, "FLYING"),
    19: (70, "FLYING"),  20: (15, "NORMAL"),  21: (80, "NORMAL"),  22: (35, "GRASS"),
    23: (65, "NORMAL"),  24: (30, "FIGHTING"),  25: (120, "NORMAL"),  26: (70, "FIGHTING"),
    27: (60, "FIGHTING"),  29: (70, "NORMAL"),  30: (65, "NORMAL"),  31: (15, "NORMAL"),
    32: (1, "NORMAL"),  33: (35, "NORMAL"),  34: (85, "NORMAL"),  35: (15, "NORMAL"),
    36: (90, "NORMAL"),  37: (90, "NORMAL"),  38: (100, "NORMAL"),  40: (15, "POISON"),
    41: (25, "BUG"),  42: (14, "BUG"),  44: (60, "NORMAL"),  49: (1, "NORMAL"),
    51: (40, "POISON"),  52: (40, "FIRE"),  53: (95, "FIRE"),  55: (40, "WATER"),
    56: (120, "WATER"),  57: (95, "WATER"),  58: (95, "ICE"),  59: (120, "ICE"),
    60: (65, "PSYCHIC"),  61: (65, "WATER"),  62: (65, "ICE"),  63: (150, "NORMAL"),
    64: (35, "FLYING"),  65: (80, "FLYING"),  66: (80, "FIGHTING"),  67: (50, "FIGHTING"),
    68: (1, "FIGHTING"),  69: (1, "FIGHTING"),  70: (80, "NORMAL"),  71: (20, "GRASS"),
    72: (40, "GRASS"),  75: (55, "GRASS"),  76: (120, "GRASS"),  80: (70, "GRASS"),
    82: (1, "DRAGON"),  83: (15, "FIRE"),  84: (40, "ELECTRIC"),  85: (95, "ELECTRIC"),
    87: (120, "ELECTRIC"),  88: (50, "ROCK"),  89: (100, "GROUND"),  90: (1, "GROUND"),
    91: (100, "GROUND"),  93: (50, "PSYCHIC"),  94: (90, "PSYCHIC"),  98: (40, "NORMAL"),
    99: (20, "NORMAL"),  120: (130, "NORMAL"),  121: (100, "NORMAL"),  122: (20, "GHOST"),
    123: (20, "POISON"),  124: (65, "POISON"),  125: (65, "GROUND"),  126: (120, "FIRE"),
    127: (80, "WATER"),  128: (35, "WATER"),  129: (60, "NORMAL"),  130: (100, "NORMAL"),
    131: (20, "NORMAL"),  132: (10, "NORMAL"),  136: (85, "FIGHTING"),  138: (100, "PSYCHIC"),
    140: (15, "NORMAL"),  141: (20, "BUG"),  143: (140, "FLYING"),  145: (20, "WATER"),
    146: (70, "NORMAL"),  149: (1, "PSYCHIC"),  152: (90, "WATER"),  153: (170, "NORMAL"),
    154: (18, "NORMAL"),  155: (50, "GROUND"),  157: (75, "ROCK"),  158: (80, "NORMAL"),
    161: (80, "NORMAL"),  162: (1, "NORMAL"),  163: (70, "NORMAL"),  165: (50, "NORMAL"),
}

_TYPE_ID2NAME = {0: "NORMAL", 1: "FIGHTING", 2: "FLYING",
                 3: "POISON", 4: "GROUND", 5: "ROCK", 7: "BUG",
                 8: "GHOST", 20: "FIRE", 21: "WATER", 22: "GRASS",
                 23: "ELECTRIC", 24: "PSYCHIC", 25: "ICE",
                 26: "DRAGON"}

# Gen 1 type chart (pokered data/types/type_matchups.asm):
# (attacking, defending) -> multiplier, non-1.0 entries only.
_TYPE_CHART = {}
for _a, _d, _m in [
    ("NORMAL","ROCK",0.5),("NORMAL","GHOST",0),
    ("FIRE","FIRE",0.5),("FIRE","WATER",0.5),("FIRE","GRASS",2),
    ("FIRE","ICE",2),("FIRE","BUG",2),("FIRE","ROCK",0.5),("FIRE","DRAGON",0.5),
    ("WATER","FIRE",2),("WATER","WATER",0.5),("WATER","GRASS",0.5),
    ("WATER","GROUND",2),("WATER","ROCK",2),("WATER","DRAGON",0.5),
    ("ELECTRIC","WATER",2),("ELECTRIC","ELECTRIC",0.5),("ELECTRIC","GRASS",0.5),
    ("ELECTRIC","ELECTRIC",0.5),("ELECTRIC","GROUND",0),("ELECTRIC","FLYING",2),
    ("ELECTRIC","DRAGON",0.5),
    ("GRASS","FIRE",0.5),("GRASS","WATER",2),("GRASS","GRASS",0.5),
    ("GRASS","POISON",0.5),("GRASS","GROUND",2),("GRASS","FLYING",0.5),
    ("GRASS","BUG",0.5),("GRASS","ROCK",2),("GRASS","DRAGON",0.5),
    ("ICE","WATER",0.5),("ICE","GRASS",2),("ICE","ICE",0.5),("ICE","GROUND",2),
    ("ICE","FLYING",2),("ICE","DRAGON",2),
    ("FIGHTING","NORMAL",2),("FIGHTING","ICE",2),("FIGHTING","POISON",0.5),
    ("FIGHTING","FLYING",0.5),("FIGHTING","PSYCHIC",0.5),("FIGHTING","BUG",0.5),
    ("FIGHTING","ROCK",2),("FIGHTING","GHOST",0),
    ("POISON","GRASS",2),("POISON","POISON",0.5),("POISON","GROUND",0.5),
    ("POISON","BUG",2),("POISON","ROCK",0.5),("POISON","GHOST",0.5),
    ("GROUND","FIRE",2),("GROUND","ELECTRIC",2),("GROUND","GRASS",0.5),
    ("GROUND","POISON",2),("GROUND","FLYING",0),("GROUND","BUG",0.5),
    ("GROUND","ROCK",2),
    ("FLYING","ELECTRIC",0.5),("FLYING","GRASS",2),("FLYING","FIGHTING",2),
    ("FLYING","BUG",2),("FLYING","ROCK",0.5),
    ("PSYCHIC","FIGHTING",2),("PSYCHIC","POISON",2),("PSYCHIC","PSYCHIC",0.5),
    ("BUG","FIRE",0.5),("BUG","GRASS",2),("BUG","FIGHTING",0.5),
    ("BUG","POISON",2),("BUG","FLYING",0.5),("BUG","PSYCHIC",2),
    ("BUG","GHOST",0.5),
    ("ROCK","FIRE",2),("ROCK","ICE",2),("ROCK","FIGHTING",0.5),
    ("ROCK","GROUND",0.5),("ROCK","FLYING",2),("ROCK","BUG",2),
    ("GHOST","NORMAL",0),("GHOST","PSYCHIC",0),("GHOST","GHOST",2),
    ("DRAGON","DRAGON",2),
]:
    _TYPE_CHART[(_a, _d)] = _m
del _a, _d, _m


def _effectiveness(atk_type: str, def_types) -> float:
    eff = 1.0
    for d in def_types:
        if d:
            eff *= _TYPE_CHART.get((atk_type, d), 1.0)
    return eff


def _typed_move_fn(emu: PyBoyEmulator):
    """fn(party idx) -> move indices ranked by EFFECTIVE power
    (base power x type effectiveness x STAB) against the live
    enemy, same contract and 0-PP discipline as _ranked_move_fn
    (probe145). The raw-power ranker cannot see that Ember (40)
    out-damages Bite (60) into Grass, or that Water Gun is
    resisted there - the two facts the Erika matchup turns on.
    Enemy types come from 0xCFEA/B, our active mon's from
    0xD019/A (a switched-in mon's types are its own)."""
    from agent.executor import STATUS_MOVE_IDS
    mem = emu.pb.memory

    def fn(idx):
        st = emu.read_state()
        mon = st.party[idx]
        et = tuple(_TYPE_ID2NAME.get(mem[a]) for a in
                   (ADDR_ENEMY_TYPE1, ADDR_ENEMY_TYPE2))
        bt = tuple(_TYPE_ID2NAME.get(mem[a]) for a in
                   (ADDR_BATTLE_MON_TYPE1, ADDR_BATTLE_MON_TYPE2))
        cands = []
        any_pp = None
        for i, mv in enumerate(mon.moves):
            if not mv:
                continue
            if i < len(mon.pp) and mon.pp[i] > 0 \
                    and any_pp is None:
                any_pp = i
            info = _MOVE_INFO.get(mv)
            if info is None or mv in STATUS_MOVE_IDS:
                continue
            if i >= len(mon.pp) or mon.pp[i] <= 0:
                continue
            power, mtype = info
            eff = _effectiveness(mtype, et)
            stab = 1.5 if mtype in bt else 1.0
            cands.append((power * eff * stab, i))
        cands.sort(reverse=True)
        return [i for _s, i in cands] or \
            ([any_pp] if any_pp is not None else [0])
    return fn


def _damaging_pp(emu: PyBoyEmulator, idx: int) -> int:
    mon = emu.read_state().party[idx]
    return sum(pp for mv, pp in zip(mon.moves, mon.pp)
               if mv in _MOVE_INFO and pp > 0)


def _ride_text(emu: PyBoyEmulator, rounds: int = 80) -> bool:
    """Press through text until the dialog flag reads clear on
    TWO consecutive polls 30 ticks apart (probe145: wLetterPrinting
    delay makes in_dialog flicker mid-print; probe146's first
    departure rode under still-open text and desynced a mart
    ride). Ends with a wait_free either way."""
    free = 0
    for _ in range(rounds):
        if emu.read_state().in_dialog:
            _press10(emu, "a")
            free = 0
            continue
        emu.tick(30)
        if not emu.read_state().in_dialog:
            free += 1
            if free >= 2:
                break
        else:
            free = 0
    emu.wait_free(max_rounds=8, stable_needed=2)
    return free >= 2


ADDR_BATTLE_KIND = 0xD057   # wIsInBattle: 1 = wild, 2 = trainer


def _drive_battle_typed(emu: PyBoyEmulator,
                        switch_first=None,
                        flee_wild: bool = False) -> bool:
    """Drive the live battle to its honest end (in_battle
    clears) with the typed ranker (probe145's drive_battle).
    fight_until_end's own exit can fire seconds early on menu
    flicker; this loop re-drives until the battle flag is
    really down. A PP-dry active mon is rescued by switching
    to the first teammate holding PP (the Exeggcute wedge's
    lesson); switch_first rides only the first drive.
    flee_wild (probe149): a WILD battle (wIsInBattle == 1)
    is fled with _run_away first - two attempts, then the
    fight proceeds as usual. Trainer and scripted battles
    are never fled. NOTE: the ghost Marowak's scripted
    battle also reads kind 1, and fleeing it counts as
    not-defeated - beat_ghost_marowak therefore drives it
    with flee_wild left off."""
    if flee_wild and emu.read_state().in_battle and \
            emu.pb.memory[ADDR_BATTLE_KIND] == 1:
        for _ in range(2):
            if _run_away(emu):
                emu.wait_free(max_rounds=8, stable_needed=2)
                return not emu.read_state().in_battle
            if not emu.read_state().in_battle:
                return True
    drv = _BattleDriver(emu)
    first = True
    drives = 0
    while emu.read_state().in_battle and drives < 15:
        sf = switch_first if first else None
        if sf is None and first:
            mem = emu.pb.memory
            active = None
            for i, mon in enumerate(emu.read_state().party):
                if mon.species_id == mem[ADDR_BATTLE_MON_SPECIES]:
                    active = i
                    break
            if active is not None and active < len(
                    emu.read_state().party) and \
                    _damaging_pp(emu, active) == 0:
                for j, mon in enumerate(emu.read_state().party):
                    if j != active and mon.hp > 0 and \
                            _damaging_pp(emu, j) > 0:
                        sf = j
                        break
        drv.fight_until_end(_typed_move_fn(emu), switch_first=sf,
                            max_actions=600)
        first = False
        drives += 1
        emu.wait_free(max_rounds=8, stable_needed=2)
        _ride_text(emu)
        emu.wait_free(max_rounds=8, stable_needed=2)
    return not emu.read_state().in_battle


def _walk_fighting(emu: PyBoyEmulator, ex: Executor, map_id: int,
                   tile, attempts: int = 25,
                   flee_wild: bool = False):
    """Walk to a tile inside one map, fighting whatever
    intercepts (probe145). Returns (arrived, battles fought).
    Needed wherever sight trainers guard the path: a raw
    _walk_to sometimes threads a sight line without triggering
    the trainer at all, and a challenge built on it talks to
    thin air (the probe145 Lasses). flee_wild passes through
    to the battle drive (probe149's Tower walks)."""
    fought = 0
    for _try in range(attempts):
        st = emu.read_state()
        if st.in_battle:
            _drive_battle_typed(emu, flee_wild=flee_wild)
            fought += 1
            continue
        if (st.x, st.y) == tile and st.map_id == map_id:
            return True, fought
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
            continue
        try:
            ex._walk_to(map_id, tile)
        except StuckError:
            pass
        st = emu.read_state()
        if st.in_battle:
            _drive_battle_typed(emu, flee_wild=flee_wild)
            fought += 1
    st = emu.read_state()
    return (st.x, st.y) == tile and st.map_id == map_id, fought


def _hop_ride(emu: PyBoyEmulator, ex: Executor, map_names) -> bool:
    """Ride a route as ONE JOURNEY PER MAP (probe145/147's hop
    discipline). A single multi-map journey execution once
    diverged from its plan, rode through a Saffron gatehouse
    and auto-surrendered the guard's drink in a dead lineage;
    hop rides keep every gatehouse-adjacent step individually
    verified. Callers holding the Fresh Water re-check the bag
    after gatehouse-adjacent hops."""
    for name in map_names:
        mid = ex.world.map_id(name)
        if emu.read_state().map_id == mid:
            continue
        if not _ride_battling(emu, ex, mid):
            return False
    return True


def _switch_menu_index(moves) -> int:
    """Index of SWITCH in a mon's party submenu: the submenu
    prepends the mon's FIELD MOVES, so SWITCH sits at
    1 + (field moves known) (probe147: probe145's party_lead
    hard-coded item 1 - correct only for field-move-less mons;
    for Oddish item 1 is CUT's slot neighbour STATS, and the
    'reorder' silently opened the stats page instead)."""
    from agent.executor import FIELD_MOVE_IDS
    return 1 + sum(1 for mv in moves if mv in FIELD_MOVE_IDS)


START_MENU_POKEMON = 1
# START menu order: POKEDEX 0, POKEMON 1, ITEM 2, trainer 3,
# SAVE 4, OPTION 5, EXIT 6 (probe148: the first port of
# _swap_party_slots steered to 3 - the trainer-card entry -
# and every live swap failed; probe147's proven helper used 1).


def _swap_party_slots(emu: PyBoyEmulator, i: int, j: int) -> bool:
    """Swap party slots i and j via START -> POKEMON -> slot i
    -> SWITCH (field-move-aware index) -> slot j, verified by
    the party order afterwards (probe147)."""
    st = emu.read_state()
    if i == j or i >= len(st.party) or j >= len(st.party):
        return i == j
    want = [m.species_id for m in st.party]
    want[i], want[j] = want[j], want[i]
    _close_menus(emu)
    mem = emu.pb.memory
    _press10(emu, "start")
    if not _wait_menu_y(emu, 2):
        return False
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM],
                       START_MENU_POKEMON):
        return False
    _press10(emu, "a")                         # POKEMON
    if not _wait_menu_y(emu, 1):
        return False
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], i):
        return False
    _press10(emu, "a")                         # open submenu
    emu.tick(80)
    sw = _switch_menu_index(emu.read_state().party[i].moves)
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], sw):
        return False
    _press10(emu, "a")                         # SWITCH
    emu.tick(60)
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], j):
        return False
    _press10(emu, "a")                         # confirm target
    emu.tick(120)
    _close_menus(emu)
    emu.wait_free(max_rounds=8, stable_needed=2)
    got = [m.species_id for m in emu.read_state().party]
    return got == want


def party_lead(emu: PyBoyEmulator, species_id: int) -> bool:
    """Move a species to the lead slot (probe147's corrected
    party_lead; see _switch_menu_index for the bug in the
    original)."""
    idx = _party_index_of(emu, species_id)
    if idx is None:
        return False
    if idx == 0:
        return True
    return _swap_party_slots(emu, idx, 0)


def _arrange_front(emu: PyBoyEmulator, first_id: int,
                   second_id: int) -> bool:
    """Put first_id in slot 0 and second_id in slot 1, in
    whichever order they currently sit (probe147: the battle
    driver's replacement pick is the first HEALTHY mon in
    party order, so the closer must physically sit second -
    in probe146's second Erika attempt Blastoise sat at slot 5
    and never fought)."""
    if not party_lead(emu, first_id):
        return False
    idx = _party_index_of(emu, second_id)
    if idx is None:
        return False
    if idx == 1:
        return True
    return _swap_party_slots(emu, idx, 1)


CELADON_MART2F_LIST = [ITEM_GREAT_BALL, ITEM_SUPER_POTION, 0x35,
                       0x22, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]
# data/items/marts.asm CeladonMart2FClerk1: Great Ball, Super
# Potion, Revive, Super Repel, Antidote, Burn/ Ice Heal,
# Awakening, Parlyz Heal. Only indices 0-1 are ever bought.
CELADON_MART4F_LIST = [0x33, ITEM_FIRE_STONE, 0x21, 0x22, 0x2F]
# CeladonMart4FClerk: Poke Doll, Fire/ Thunder/ Water/ Leaf
# Stone. Only index 1 (the Fire Stone) is ever bought.


def buy_vending_drink(emu: PyBoyEmulator, ex: Executor,
                      item_id: int = ITEM_FRESH_WATER,
                      list_index: int = 0, price: int = 200
                      ) -> bool:
    """Buy a drink from the Celadon Mart roof vending machine
    (probe145). The machine's menu opens at menuY 5 (not the
    mart clerk's 1) and the charge posts only after the whole
    delivery sequence, so verification rides the dialog and
    then checks BOTH the bag and the exact money delta. Roof
    access chains the mart floors one warp at a time (the
    elevator lesson: a single journey into the building
    spins). Ends on the roof."""
    world = ex.world
    roof = world.map_id("CeladonMartRoof")
    if _bag_count(emu, item_id) >= 1:
        return True
    for floor in ("CeladonMart1F", "CeladonMart2F",
                  "CeladonMart3F", "CeladonMart4F",
                  "CeladonMart5F", "CeladonMartRoof"):
        if not _ride_to(emu, ex, world.map_id(floor)):
            return False
    emu.wait_free(max_rounds=8, stable_needed=2)
    if emu.read_state().map_id != roof:
        return False
    try:
        ex._walk_to(roof, (10, 2))
    except StuckError:
        return False
    _face(emu, "up")
    money0 = emu.read_state().money
    mem = emu.pb.memory
    _press10(emu, "a")
    for _ in range(60):
        if mem[ADDR_MENU_TOP_Y] == 5:
            break
        emu.tick(20)
    if mem[ADDR_MENU_TOP_Y] != 5:
        return False
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM],
                       list_index):
        return False
    _press10(emu, "a")
    for _ in range(300):
        if _bag_count(emu, item_id) >= 1:
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(30)
    emu.wait_free(max_rounds=8, stable_needed=2)
    return _bag_count(emu, item_id) >= 1 and \
        emu.read_state().money == money0 - price


def catch_growlithe(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Catch a Growlithe in Route 8's grass (probe145) - the
    Fire counter Erika's Grass team demands (Blastoise's Water
    damage is Grass-resisted and Razor Leaf crits near-always;
    see S13's matchup math). Growlithe is 20% of the Route 8
    grass at Lv16-18, catch rate 190. Modelled on
    catch_voltorb: a Growlithe interrupting the walk to the
    grass is caught on the spot via _catch_current. With a
    full party the catch files to Box 1, where the PC swap
    picks it up. Idempotent."""
    world = ex.world
    r8 = world.map_id("Route8")
    grass = (34, 10)
    if SPECIES_GROWLITHE in _box_species(emu) or \
            SPECIES_ARCANINE in _box_species(emu) or \
            _party_index_of(emu, SPECIES_GROWLITHE) is not None or \
            _party_index_of(emu, SPECIES_ARCANINE) is not None:
        return True
    if not _ride_battling(emu, ex, r8):
        return False
    mem = emu.pb.memory
    for _try in range(60):
        st = emu.read_state()
        if st.in_battle:
            sp = mem[ADDR_ENEMY_SPECIES]
            if sp == SPECIES_GROWLITHE and _catch_current(emu):
                break
            if not _run_away(emu) and emu.read_state().in_battle:
                _drive_battle_typed(emu)
            continue
        if (st.x, st.y) == grass and st.map_id == r8:
            break
        try:
            ex._walk_to(r8, grass)
        except StuckError:
            pass
        if emu.read_state().in_dialog:
            emu.close_dialog(max_presses=2)
    if SPECIES_GROWLITHE not in _box_species(emu) and \
            _party_index_of(emu, SPECIES_GROWLITHE) is None:
        _catch_any(emu, ex, {SPECIES_GROWLITHE},
                   max_encounters=40)
    return SPECIES_GROWLITHE in _box_species(emu) or \
        _party_index_of(emu, SPECIES_GROWLITHE) is not None


def evolve_with_stone(emu: PyBoyEmulator, item_id: int,
                      from_species: int, to_species: int) -> bool:
    """Evolve a party mon with an evolution stone (probe146:
    the Fire Stone on Growlithe). The evolution ride presses A
    ONLY - B cancels a Gen 1 evolution, so the bag-menu B-exit
    vocabulary stays out of it. Verified in RAM: species flip,
    level and moves unchanged (a stone evolution must cost
    neither), the dex ticks up by one, the stone is consumed.
    The party member is left wherever the evolution put it -
    party order does not change."""
    idx = _party_index_of(emu, from_species)
    if idx is None:
        return _party_index_of(emu, to_species) is not None
    if _bag_count(emu, item_id) == 0:
        return False
    before = emu.read_state().party[idx]
    dex0 = _dex_owned_count(emu)
    if not _use_bag_item_on_mon(emu, item_id, idx):
        return False
    # NO menu-closing here: the ride below presses A only,
    # exactly as probe146 proved it - a B press landing after
    # the evolution starts cancels it.
    for _ in range(240):
        mon = emu.read_state().party[idx]
        if mon.species_id == to_species:
            break
        _press10(emu, "a")
        emu.tick(40)
    emu.wait_free(max_rounds=8, stable_needed=2)
    mon = emu.read_state().party[idx]
    return mon.species_id == to_species and \
        mon.level == before.level and \
        [m for m in mon.moves if m] == [m for m in before.moves if m] and \
        _dex_owned_count(emu) == dex0 + 1 and \
        _bag_count(emu, item_id) == 0


# (npc tile, stance, facing to talk, event flag) - flags from
# the CeladonGym map header (probe145, ROM-verified).
CELADON_GYM_TRAINERS = [
    ((2, 11), (3, 11), "left", 426),
    ((7, 10), (6, 10), "right", 427),
    ((9, 5), (9, 6), "up", 428),
    ((1, 5), (1, 6), "up", 429),
    ((6, 3), (6, 4), "up", 430),
    ((3, 3), (3, 4), "up", 431),
    ((5, 3), (5, 4), "up", 432),
]

CELADON_TO_CERULEAN_HOPS = [
    "Route7", "UndergroundPathRoute7", "UndergroundPathWestEast",
    "UndergroundPathRoute8", "Route8", "LavenderPokecenter",
    "Route10", "RockTunnel1F", "RockTunnelB1F", "Route9",
    "CeruleanPokecenter",
]
CERULEAN_TO_CELADON_HOPS = [
    "Route9", "RockTunnel1F", "RockTunnelB1F", "Route10",
    "LavenderPokecenter", "Route8", "UndergroundPathRoute8",
    "UndergroundPathWestEast", "UndergroundPathRoute7",
    "Route7", "CeladonPokecenter",
]


def _maintain_party(emu: PyBoyEmulator, counter_ids) -> None:
    """Super-Potion top-ups between gym fights (probe145's
    maintain): the lead is topped toward full under 100 HP,
    the Fire counter under 26. Fainted mons are skipped (a
    Potion does nothing for them - the heal trip handles
    those). Ends with menus closed: _use_bag_item_on_mon
    returns with the bag menu still open and the next walk
    desyncs if it stays open (probe145's gym8 lesson)."""
    st = emu.read_state()
    for idx, mon in enumerate(st.party):
        if mon.hp <= 0:
            continue
        want = (idx == 0 and mon.hp < 100 and mon.hp < mon.max_hp) or \
            (mon.species_id in counter_ids and mon.hp <= 25)
        if not want:
            continue
        if _bag_count(emu, ITEM_SUPER_POTION) == 0:
            return
        if _use_bag_item_on_mon(emu, ITEM_SUPER_POTION, idx):
            emu.tick(90)
            _close_menus(emu)
            emu.wait_free(max_rounds=8, stable_needed=2)


def _gym_heal_trip(emu: PyBoyEmulator, ex: Executor,
                   gym_map: int, pc_map: int) -> bool:
    """Out the gym door, heal at the Pokecenter, ride back
    (probe145). The door mat at (4,17) warps on step-off: walk
    onto it, then keep pressing down until the map changes."""
    st = emu.read_state()
    if st.map_id == gym_map:
        _walk_fighting(emu, ex, gym_map, (4, 17))
        for _ in range(30):
            emu.press("down")
            emu.tick(16)
            if emu.read_state().map_id != gym_map:
                break
    if not _ride_battling(emu, ex, pc_map):
        return False
    if not heal_at_pokecenter(emu, ex, pc_map):
        return False
    return _ride_battling(emu, ex, gym_map)


def _challenge_gym_trainer(emu: PyBoyEmulator, ex: Executor,
                           gym_map: int, stance, face: str
                           ) -> str:
    """Walk to a trainer's stance fighting intercepts, then
    face her and talk (probe145's challenge). Returns "fought"
    (a battle ran - walking or talking), "talked" (dialog but
    no battle) or "unreached". THE CALLER VERIFIES THE EVENT
    FLAG: a prior stage of this leg scored 'fought' off the
    driver's own word while the flag stayed clear (the driver
    had fought a DIFFERENT trainer it met en route), and the
    gym only made sense once progress was read off the flags
    (probe145 gym3)."""
    arrived, fought = _walk_fighting(emu, ex, gym_map, stance)
    if fought:
        return "fought"
    if not arrived:
        return "unreached"
    st = emu.read_state()
    if (st.x, st.y) != stance:
        return "unreached"
    _face(emu, face)
    _press10(emu, "a")
    emu.tick(120)
    if emu.read_state().in_battle:
        _drive_battle_typed(emu)
        return "fought"
    _ride_text(emu)
    return "talked"


def beat_erika(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Clear Celadon Gym and beat Erika (probes 145-147).

    Trainers first, FLAG-VERIFIED: the worklist is the ROM's
    own event flags (426-432), re-read after every challenge -
    never the driver's say-so. The Beauty at (3,3) has sight
    range 2: the approach steps to (3,5), inside her line,
    before her stance (probe145 gym3's opening fix). Heal
    trips run when the lead drops under 55 or the front two
    are down. Erika herself is fight_misty's shape with the
    typed driver: stance (4,4), face up, and the win is the
    Rainbow badge bit in RAM.

    The Fire counter leads with Blastoise second - arranged
    here, because the driver's replacement pick is the first
    healthy mon in party order (probe146's slot-5 lesson).
    In the winning lineage the counter was an Arcanine ground
    to Lv25 on Route 24 first (see grind_route24 and S13);
    this scene does not grind or evolve by itself."""
    world = ex.world
    gym = world.map_id("CeladonGym")
    celpc = world.map_id("CeladonPokecenter")
    if "Rainbow" in emu.read_state().badges:
        return True
    if _party_index_of(emu, SPECIES_ARCANINE) is not None:
        _arrange_front(emu, SPECIES_ARCANINE, SPECIES_BLASTOISE)
    elif _party_index_of(emu, SPECIES_GROWLITHE) is not None:
        _arrange_front(emu, SPECIES_GROWLITHE, SPECIES_BLASTOISE)
    counter_ids = {SPECIES_ARCANINE, SPECIES_GROWLITHE}
    for _round in range(3):
        todo = [t for t in CELADON_GYM_TRAINERS
                if not _event_flag(emu, t[3])]
        if not todo:
            break
        if not _ride_battling(emu, ex, gym):
            return False
        for npc, stance, face, ev in todo:
            if _event_flag(emu, ev):
                continue
            if npc == (3, 3):
                _walk_fighting(emu, ex, gym, (3, 5))
            _challenge_gym_trainer(emu, ex, gym, stance, face)
            _maintain_party(emu, counter_ids)
            st = emu.read_state()
            lead = st.party[0] if st.party else None
            if lead is not None and (lead.hp == 0 or lead.hp < 55
                    or all(m.hp == 0 for m in st.party[:2])):
                _gym_heal_trip(emu, ex, gym, celpc)
    if not all(_event_flag(emu, t[3])
               for t in CELADON_GYM_TRAINERS):
        return False
    st = emu.read_state()
    if st.party and st.party[0].hp < 80:
        _gym_heal_trip(emu, ex, gym, celpc)
    ex.prepare_battle_party()
    for _attempt in range(80):
        st = emu.read_state()
        if "Rainbow" in st.badges:
            return True
        if st.in_battle:
            _drive_battle_typed(emu)
            continue
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
            continue
        if st.map_id != gym:
            if not _ride_battling(emu, ex, gym):
                return False
            continue
        if (st.x, st.y) != (4, 4):
            _walk_fighting(emu, ex, gym, (4, 4))
            continue
        _face(emu, "up")
        _press10(emu, "a")
        emu.tick(120)
    return "Rainbow" in emu.read_state().badges


def grind_route24(emu: PyBoyEmulator, ex: Executor,
                  species_id: int, target_level: int) -> int:
    """Lead-and-solo level grind on Route 24's grass (probe147:
    Arcanine Lv18 -> Lv25 in 132 battles). The species leads
    and solos; nobody switch-trains (Gen 1 pays fainted
    participants NOTHING and halves shared XP - the economics
    that killed probe145's trainee plan). Ember PP (25) is the
    binding constraint, Bite the neutral fallback; a Pokecenter
    trip runs when HP < 22 or damaging PP <= 4. Travel is the
    hop chain both ways under Blastoise's lead (the tunnel is
    no place for a grinding Fire type), with the front
    arranged [grinder, Blastoise] at Cerulean and restored at
    Celadon. Returns the solo battles fought."""
    world = ex.world
    celpc = world.map_id("CeladonPokecenter")
    cerpc = world.map_id("CeruleanPokecenter")
    r24 = world.map_id("Route24")
    idx = _party_index_of(emu, species_id)
    if idx is None:
        return 0
    if emu.read_state().party[idx].level >= target_level:
        return 0
    battles = 0
    if not _ride_battling(emu, ex, celpc):
        return battles
    heal_at_pokecenter(emu, ex, celpc)
    if _party_index_of(emu, SPECIES_BLASTOISE) is not None:
        party_lead(emu, SPECIES_BLASTOISE)
    if not _hop_ride(emu, ex, CELADON_TO_CERULEAN_HOPS):
        return battles
    heal_at_pokecenter(emu, ex, cerpc)
    if not _arrange_front(emu, species_id, SPECIES_BLASTOISE):
        return battles
    grass = [(4, 20), (5, 27)]
    for _sweep in range(400):
        idx = _party_index_of(emu, species_id)
        if idx is None:
            break
        mon = emu.read_state().party[idx]
        if mon.level >= target_level:
            break
        if mon.hp == 0 or mon.hp < 22 or _damaging_pp(emu, idx) <= 4:
            if not _ride_battling(emu, ex, cerpc):
                break
            heal_at_pokecenter(emu, ex, cerpc)
            party_lead(emu, species_id)
            continue
        if emu.read_state().map_id != r24:
            if not _ride_battling(emu, ex, r24):
                break
            continue
        for tile in grass:
            _arr, f = _walk_fighting(emu, ex, r24, tile,
                                     attempts=40)
            battles += f
            idx = _party_index_of(emu, species_id)
            mon = emu.read_state().party[idx]
            if mon.level >= target_level or mon.hp < 22 or \
                    _damaging_pp(emu, idx) <= 4:
                break
    heal_at_pokecenter(emu, ex, cerpc)
    if _party_index_of(emu, SPECIES_BLASTOISE) is not None:
        party_lead(emu, SPECIES_BLASTOISE)
    _hop_ride(emu, ex, CERULEAN_TO_CELADON_HOPS)
    heal_at_pokecenter(emu, ex, celpc)
    _arrange_front(emu, species_id, SPECIES_BLASTOISE)
    return battles


def celadon_leg(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Lavender Town -> the Rainbow Badge (probes 145-147),
    following the WINNING route end to end:

      road west (hop rides) -> catch Growlithe on Route 8 ->
      PC swap (Nidoran-F out) -> Celadon Mart 2F resupply
      (5 Super Potions, 4 Great Balls) -> roof Fresh Water ->
      Mart 4F Fire Stone -> evolve Arcanine -> Route 24 grind
      to Lv25 -> the gym (beat_erika).

    The drink is bought AFTER the road west on purpose: the
    Saffron gatehouses only take drinks from travellers who
    carry one, and the westbound road passes two of them.
    (In the lineage that taught this, the drink bought in
    Celadon survived every eastbound hop home - verified in
    the bag after each one; see S13.)

    Every stage verifies from RAM and skips when already in
    place, so the leg resumes from any intermediate state.
    Assumes the party starts in or east of Lavender Town with
    the Thunder Badge (rock_tunnel_leg's end state), a free
    party slot's worth of box room, and Nidoran-F in the
    party to swap out (this lineage's shape). Money: the
    shopping totals Y6,100 and the stone Y2,100 against the
    Y8,820 the tunnel leg arrives with - the leg does not
    farm money, and a broke arrival fails honestly at the
    till."""
    world = ex.world
    celpc = world.map_id("CeladonPokecenter")
    mart2f = world.map_id("CeladonMart2F")
    mart4f = world.map_id("CeladonMart4F")
    if "Rainbow" in emu.read_state().badges:
        return _ride_battling(emu, ex, celpc) and \
            heal_at_pokecenter(emu, ex, celpc)
    west = {world.map_id(n) for n in
            ("CeladonCity", "CeladonPokecenter", "CeladonGym",
             "CeladonMart1F", "CeladonMart2F", "CeladonMart3F",
             "CeladonMart4F", "CeladonMart5F", "CeladonMartRoof",
             "Route7")}
    if emu.read_state().map_id not in west:
        if not _hop_ride(emu, ex, ["Route8",
                                   "UndergroundPathRoute8",
                                   "UndergroundPathWestEast",
                                   "UndergroundPathRoute7",
                                   "Route7"]):
            return False
    if not catch_growlithe(emu, ex):
        return False
    if not _ride_battling(emu, ex, celpc):
        return False
    if not heal_at_pokecenter(emu, ex, celpc):
        return False
    if _party_index_of(emu, SPECIES_ARCANINE) is None and \
            _party_index_of(emu, SPECIES_GROWLITHE) is None:
        if not pc_swap(emu, ex, celpc, SPECIES_NIDORAN_F,
                       SPECIES_GROWLITHE):
            return False
    # Mart 2F resupply (floor-chained: one ride per floor).
    if not _hop_ride(emu, ex, ["CeladonMart1F", "CeladonMart2F"]):
        return False
    if buy_mart_item(emu, ex, mart2f, 0, ITEM_GREAT_BALL, 4,
                     stand=(5, 5), face="up") < 4:
        return False
    if buy_mart_item(emu, ex, mart2f, 1, ITEM_SUPER_POTION, 5,
                     stand=(5, 5), face="up") < 5:
        return False
    # The drink (ends on the roof).
    if not buy_vending_drink(emu, ex):
        return False
    # The stone, and the evolution Vlad chose (S13: permanent -
    # Arcanine's Gen 1 level-up learnset is empty, so
    # Growlithe's Lv50 Flamethrower is forfeited for it; Fire
    # Blast survives as TM38).
    if _party_index_of(emu, SPECIES_ARCANINE) is None:
        if not _hop_ride(emu, ex, ["CeladonMart5F",
                                   "CeladonMart4F"]):
            return False
        if buy_mart_item(emu, ex, mart4f, 1, ITEM_FIRE_STONE, 1,
                         stand=(5, 5), face="down") < 1:
            return False
        if not _ride_battling(emu, ex, celpc):
            return False
        heal_at_pokecenter(emu, ex, celpc)
        if not evolve_with_stone(emu, ITEM_FIRE_STONE,
                                 SPECIES_GROWLITHE,
                                 SPECIES_ARCANINE):
            return False
    # The grind Vlad approved: Lv25 before Erika.
    arc = _party_index_of(emu, SPECIES_ARCANINE)
    if arc is None:
        return False
    if emu.read_state().party[arc].level < 25:
        grind_route24(emu, ex, SPECIES_ARCANINE, 25)
        arc = _party_index_of(emu, SPECIES_ARCANINE)
        if emu.read_state().party[arc].level < 24:
            return False
    if not beat_erika(emu, ex):
        return False
    if not _ride_battling(emu, ex, celpc):
        return False
    if not heal_at_pokecenter(emu, ex, celpc):
        return False
    st = emu.read_state()
    return "Rainbow" in st.badges and \
        _bag_count(emu, ITEM_FRESH_WATER) >= 1


# ------------------------------------------------------------------
# Rocket Hideout (probe148): empirical learned-graph navigation.
#
# The shipped world model's indoor grids are NOT reliable at
# cell granularity in the FACILITY tileset: block quadrants mix
# floor and machine tiles (on B2F's strip (25,8)/(26,8) are
# floor but (24,8) is a wall, whatever the grid says), B4F's
# grid is the all-open coherence fallback, and B2F/B3F arrow
# tiles relocate the player by script (map_coord_movement
# tables: RLE joypad lists played in reverse db order with
# input ignored). So hideout interiors are navigated
# EMPIRICALLY: explore_nav presses one direction, settles
# (spins run out), reads the landing cell from RAM and records
# the edge (cell, dir) -> landing | wall | warp in a per-map
# graph held in EXPLORE_LEARN; BFS plans over observed edges
# only, and frontier steps expand the graph toward the target,
# using the model grid only as a prior for where to speculate.
# Arrow rides are deterministic, so a recorded ride replays
# exactly; recorded walls may be re-probed once per call
# (doors change state - B4F's office door was recorded shut
# during the key run and opened after the guards fell).
# The graph is per-process (the probe persisted it to JSON
# across its phases; a single leg run here rebuilds it).
# ------------------------------------------------------------------
ITEM_LIFT_KEY = 0x4A
ITEM_SILPH_SCOPE = 0x48
ITEM_NUGGET = 0x31
ITEM_HP_UP = 0x23
ITEM_TM02 = 0xCA  # TM ids run $C9 + (n - 1)

# wEventFlags ids (constants/event_constants.asm, const_next
# regions; anchored on EVENT_67F = 1663 by construction and on
# the B2F/B3F trainer bits read off flag diffs between probe
# saves - two naive counts of that file are wrong in print:
# it uses const_def/const_next, not const_event, and a bare
# const_skip still advances the counter).
EV_FOUND_ROCKET_HIDEOUT = 441
EV_HIDEOUT_B1_TRAINERS = (1649, 1650, 1651, 1652, 1653)
EV_HIDEOUT_B2_TRAINER = 1665
EV_HIDEOUT_B3_TRAINERS = (1681, 1682)
EV_HIDEOUT_B4_TRAINERS = (1698, 1699, 1700)
EV_HIDEOUT_B4_DOOR = 1701
EV_ROCKET_DROPPED_LIFT_KEY = 1702
EV_BEAT_HIDEOUT_GIOVANNI = 1703

# arrow/spinner cells, from the pokered map_coord_movement
# tables for RocketHideoutB2F/B3F (converted (y,x) -> (x,y)).
HIDEOUT_ARROWS = {
    "RocketHideoutB2F": {(2, 11), (4, 11), (5, 11), (7, 10),
                         (7, 11), (8, 9), (9, 12), (11, 9),
                         (11, 10), (11, 12), (14, 9), (16, 12),
                         (18, 9), (19, 9), (21, 9), (23, 9),
                         (25, 9)},
    "RocketHideoutB3F": {(2, 12), (2, 13), (4, 13), (6, 13),
                         (8, 12), (10, 13), (12, 13), (13, 13),
                         (15, 13), (19, 17), (19, 18), (19, 19),
                         (19, 20), (19, 21), (21, 14), (23, 14)},
}

EXPLORE_LEARN: dict = {}
_EXPLORE_DIRS = {"up": (0, -1), "down": (0, 1),
                 "left": (-1, 0), "right": (1, 0)}
_EXPLORE_DELTAS = {v: k for k, v in _EXPLORE_DIRS.items()}


def _explore_graph(map_id: int) -> dict:
    return EXPLORE_LEARN.setdefault(
        map_id, {"edges": {}, "cells": set()})


def _explore_settle(emu: PyBoyEmulator, max_frames: int = 1200):
    """Poll until the player is genuinely still (an arrow spin
    has run out) or a battle/dialog/map change interrupts."""
    last = None
    stable = 0
    spent = 0
    while spent < max_frames:
        emu.tick(12)
        spent += 12
        st = emu.read_state()
        if st.in_battle or st.in_dialog:
            return st
        pos = (st.map_id, st.x, st.y)
        if pos == last:
            stable += 1
            if stable >= 4:
                return st
        else:
            stable = 0
            last = pos
    return emu.read_state()


def _explore_step(emu: PyBoyEmulator, ex: Executor, map_id: int,
                  cell, btn, flee_wild: bool = False):
    """Press one direction from `cell`, settle, fight/close
    whatever interrupts, and record the true outcome edge.
    Returns (outcome, battles): a landing cell, None (wall),
    or "warp" (the map changed)."""
    g = _explore_graph(map_id)
    emu.press(btn)
    st = _explore_settle(emu)
    battles = 0
    for _ in range(6):
        if st.in_battle:
            _drive_battle_typed(emu, flee_wild=flee_wild)
            battles += 1
            st = _explore_settle(emu)
            continue
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
            st = _explore_settle(emu)
            continue
        break
    if st.map_id != map_id:
        g["edges"][(cell, btn)] = "warp"
        return "warp", battles
    land = (st.x, st.y)
    g["cells"].add(land)
    if land == cell:
        g["edges"][(cell, btn)] = None
        return None, battles
    g["edges"][(cell, btn)] = land
    return land, battles


def _explore_bfs(map_id: int, start, goals):
    """Shortest path over RECORDED edges only; returns a list
    of (cell, dir) steps or None."""
    from collections import deque
    g = _explore_graph(map_id)
    if start in goals:
        return []
    prev = {start: None}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        for btn in _EXPLORE_DIRS:
            out = g["edges"].get((cur, btn), "untried")
            if out in ("untried", None, "warp"):
                continue
            if out not in prev:
                prev[out] = (cur, btn)
                if out in goals:
                    path = []
                    node = out
                    while prev[node] is not None:
                        path.append(prev[node])
                        node = prev[node][0]
                    return path[::-1]
                dq.append(out)
    return None


def explore_nav(emu: PyBoyEmulator, ex: Executor, map_id: int,
                target, max_steps: int = 6000,
                flee_wild: bool = False):
    """Navigate to `target` inside one map over the learned
    graph (probe148), exploring frontier steps when no known
    route exists. Returns (status, battles): status in
    {"tile", "warp", "blackout", "stuck"}."""
    from collections import deque
    world = ex.world
    gm = world.maps[map_id]
    g = _explore_graph(map_id)
    arrows = HIDEOUT_ARROWS.get(gm.name, ())
    target = tuple(target)
    battles = 0
    retested = set()
    money0 = emu.read_state().money
    for _ in range(max_steps):
        st = emu.read_state()
        if st.map_id != map_id:
            return ("warp", battles)
        if st.money < money0:
            return ("blackout", battles)
        if st.in_battle:
            _drive_battle_typed(emu, flee_wild=flee_wild)
            battles += 1
            continue
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
            continue
        cur = (st.x, st.y)
        g["cells"].add(cur)
        if cur == target:
            return ("tile", battles)
        path = _explore_bfs(map_id, cur, {target})
        if path:
            cell, btn = path[0]
            _, b = _explore_step(emu, ex, map_id, cell, btn,
                                 flee_wild=flee_wild)
            battles += b
            continue
        dist = {cur: 0}
        dq = deque([cur])
        while dq:
            c = dq.popleft()
            for btn in _EXPLORE_DIRS:
                o = g["edges"].get((c, btn), "untried")
                if o not in ("untried", None, "warp") \
                        and o not in dist:
                    dist[o] = dist[c] + 1
                    dq.append(o)
        warp_cells = {(w.x, w.y) for w in gm.warps}
        best = None
        for c, d in dist.items():
            for btn, (dx, dy) in _EXPLORE_DIRS.items():
                rec = g["edges"].get((c, btn), "untried")
                nb = (c[0] + dx, c[1] + dy)
                if not gm.in_bounds(nb[0], nb[1]):
                    continue
                if nb in warp_cells and nb != target:
                    continue  # never teleport speculatively
                score = d + abs(nb[0] - target[0]) \
                    + abs(nb[1] - target[1])
                if not gm.walkable(nb[0], nb[1]):
                    score += 6
                if nb in arrows:
                    score += 10
                if rec is None:
                    if (c, btn) in retested:
                        continue
                    score += 80  # wall re-probe, last resort
                elif rec != "untried":
                    continue
                if best is None or score < best[0]:
                    best = (score, c, btn)
        if best is None:
            return ("stuck", battles)
        _, fc, fbtn = best
        if fc != cur:
            path = _explore_bfs(map_id, cur, {fc})
            cell, btn = path[0]
        else:
            cell, btn = fc, fbtn
            if g["edges"].get((cell, btn), "untried") is None:
                retested.add((cell, btn))
        _, b = _explore_step(emu, ex, map_id, cell, btn,
                             flee_wild=flee_wild)
        battles += b
    return ("stuck", battles)


def _face_and_talk(emu: PyBoyEmulator, cell) -> None:
    """Face `cell` from the current adjacent position and
    press A (the press toward the cell only turns: it is
    occupied or a bg tile)."""
    st = emu.read_state()
    btn = _EXPLORE_DELTAS.get((cell[0] - st.x, cell[1] - st.y))
    assert btn, f"not adjacent: at {(st.x, st.y)}, cell {cell}"
    emu.press(btn)
    emu.tick(15)
    _press10(emu, "a")
    emu.tick(45)


def _advance_to_battle(emu: PyBoyEmulator,
                       max_rounds: int = 60,
                       flee_wild: bool = False) -> bool:
    """Advance dialog (A presses) until a battle starts, then
    drive it. Returns True if a battle was fought."""
    for _ in range(max_rounds):
        st = emu.read_state()
        if st.in_battle:
            _drive_battle_typed(emu, flee_wild=flee_wild)
            return True
        if st.in_dialog:
            _press10(emu, "a")
            emu.tick(30)
        else:
            emu.tick(30)
    return False


def pickup_ball(emu: PyBoyEmulator, ex: Executor, map_id: int,
                ball, item_id: int) -> bool:
    """Stand next to an item ball, face it, take it, and
    verify the bag count from RAM. Tries each stance."""
    want = _bag_count(emu, item_id)
    bx, by = ball
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        stance = (bx + dx, by + dy)
        status, _ = explore_nav(emu, ex, map_id, stance)
        st = emu.read_state()
        if status != "tile" or (st.x, st.y) != stance:
            continue
        _face_and_talk(emu, ball)
        emu.wait_free(max_rounds=30, stable_needed=3)
        st = emu.read_state()
        if st.in_battle:
            _drive_battle_typed(emu)
        if _bag_count(emu, item_id) > want:
            return True
    return False


def talk_fight_event(emu: PyBoyEmulator, ex: Executor,
                     map_id: int, npc_cell, event_id: int,
                     stance_order=((0, 1), (0, -1), (1, 0),
                                   (-1, 0)),
                     flee_wild: bool = False) -> bool:
    """Navigate to a stance next to an NPC, face it, talk, and
    fight if a battle starts; verified by the trainer event
    flag. (A sight-triggered battle during the approach also
    satisfies it: the flag is checked first and after.)
    flee_wild passes through to the navigation and battle
    drives (probe149's Tower floors, where wild Gastly
    interrupt the learned-graph walks)."""
    if _event_flag(emu, event_id):
        return True
    nx, ny = npc_cell
    for dx, dy in stance_order:
        stance = (nx + dx, ny + dy)
        status, _ = explore_nav(emu, ex, map_id, stance,
                                flee_wild=flee_wild)
        st = emu.read_state()
        if status != "tile" or (st.x, st.y) != stance:
            continue
        if _event_flag(emu, event_id):
            return True
        _face_and_talk(emu, npc_cell)
        _advance_to_battle(emu, flee_wild=flee_wild)
        emu.wait_free(max_rounds=60, stable_needed=6)
        if _event_flag(emu, event_id):
            return True
    return _event_flag(emu, event_id)


def elevator_ride(emu: PyBoyEmulator, ex: Executor,
                  dest_index: int) -> bool:
    """Ride the Rocket Hideout elevator from inside the cab to
    floor index dest_index (0=B1F, 1=B2F, 2=B4F) (probe148).
    The panel bg is at (1,1) with stance (1,2); the floor menu
    is steered by wCurrentMenuItem; choosing a floor rewrites
    the door warps' (2,1)/(3,1) destinations. Needs the LIFT
    KEY in the bag. Returns True when a hideout floor is
    reached."""
    world = ex.world
    elev = world.map_id("RocketHideoutElevator")
    mem = emu.pb.memory
    st = emu.read_state()
    if st.map_id != elev or _bag_count(emu, ITEM_LIFT_KEY) < 1:
        return False
    for _ in range(10):
        st = emu.read_state()
        if (st.x, st.y) == (1, 2):
            break
        if st.x > 1:
            emu.press("left")
        elif st.y < 2:
            emu.press("down")
        elif st.y > 2:
            emu.press("up")
        _explore_settle(emu)
    emu.press("up")   # face the panel (turn only)
    emu.tick(15)
    _press10(emu, "a")
    emu.tick(60)
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM],
                       dest_index, rounds=12):
        return False
    _press10(emu, "a")
    emu.tick(120)
    emu.wait_free(max_rounds=40, stable_needed=4)
    for btn in ("right", "right", "up", "up", "left", "up"):
        st = emu.read_state()
        if st.map_id != elev:
            break
        emu.press(btn)
        _explore_settle(emu)
    return emu.read_state().map_id != elev


def rocket_hideout(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Celadon Game Corner -> Rocket Hideout -> the LIFT KEY
    (probe148). The poster route needs no coins: beat the
    Rocket guarding it (Raticate/Zubat Lv20), advance his
    exit scene, then examine the poster - its text_asm sets
    EVENT_FOUND_ROCKET_HIDEOUT and swaps in the staircase.
    Descent B1F -> B2F -> B3F is empirical navigation (see
    explore_nav); on B4F's west side, Rocket3 (11,2) drops
    the LIFT KEY ball at (10,2) when talked to after his
    battle. Also picks up TM02 (9,4) and the HP Up (10,12)
    on that floor. The B2F Nugget was left: two probe
    attempts slid into the south-west arrow pocket and the
    learned component there is a dead end - optional loot,
    not worth the maze. Ends on B4F with the key in the bag.
    Assumes Celadon (Rainbow Badge held), Blastoise in the
    party to lead. Every stage skips when already done."""
    world = ex.world
    gc = world.map_id("GameCorner")
    b1f = world.map_id("RocketHideoutB1F")
    b2f = world.map_id("RocketHideoutB2F")
    b3f = world.map_id("RocketHideoutB3F")
    b4f = world.map_id("RocketHideoutB4F")
    if _bag_count(emu, ITEM_LIFT_KEY) >= 1:
        return True
    if not party_lead(emu, SPECIES_BLASTOISE):
        return False
    st = emu.read_state()
    if st.map_id != gc:
        if not _ride_battling(emu, ex, gc):
            return False
    if not _event_flag(emu, EV_FOUND_ROCKET_HIDEOUT):
        ok, _ = _walk_fighting(emu, ex, gc, (9, 6))
        st = emu.read_state()
        if not ok or (st.x, st.y) != (9, 6):
            return False
        _face_and_talk(emu, (9, 5))
        if not _advance_to_battle(emu):
            return False
        emu.wait_free(max_rounds=120, stable_needed=10)
        emu.wait_free(max_rounds=60, stable_needed=10)
        ok, _ = _walk_fighting(emu, ex, gc, (9, 5))
        st = emu.read_state()
        if not ok or (st.x, st.y) != (9, 5):
            return False
        _face_and_talk(emu, (9, 4))
        for _ in range(40):
            if _event_flag(emu, EV_FOUND_ROCKET_HIDEOUT):
                break
            if emu.read_state().in_dialog:
                _press10(emu, "a")
            emu.tick(30)
        emu.wait_free(max_rounds=40, stable_needed=4)
        if not _event_flag(emu, EV_FOUND_ROCKET_HIDEOUT):
            return False
    # descend: Game Corner stairs -> B1F -> B2F -> B3F -> B4F
    legs = ((gc, (17, 4), b1f), (b1f, (23, 2), b2f),
            (b2f, (21, 8), b3f), (b3f, (19, 18), b4f))
    for src_map, warp_cell, dst_map in legs:
        st = emu.read_state()
        if st.map_id == dst_map:
            continue
        if st.map_id != src_map:
            return False
        status, _ = explore_nav(emu, ex, src_map, warp_cell)
        if emu.read_state().map_id != dst_map:
            return False
    # B4F west: Rocket3, the key drop, the two cheap items
    if not talk_fight_event(emu, ex, b4f, (11, 2),
                            EV_HIDEOUT_B4_TRAINERS[2]):
        return False
    for _ in range(3):
        if _event_flag(emu, EV_ROCKET_DROPPED_LIFT_KEY):
            break
        status, _ = explore_nav(emu, ex, b4f, (11, 3))
        st = emu.read_state()
        if (st.x, st.y) != (11, 3):
            continue
        _face_and_talk(emu, (11, 2))
        emu.wait_free(max_rounds=40, stable_needed=4)
    if not _event_flag(emu, EV_ROCKET_DROPPED_LIFT_KEY):
        return False
    if not pickup_ball(emu, ex, b4f, (10, 2), ITEM_LIFT_KEY):
        return False
    pickup_ball(emu, ex, b4f, (9, 4), ITEM_TM02)
    pickup_ball(emu, ex, b4f, (10, 12), ITEM_HP_UP)
    return _bag_count(emu, ITEM_LIFT_KEY) >= 1


def beat_giovanni_hq(emu: PyBoyEmulator, ex: Executor) -> bool:
    """B4F -> elevator -> Giovanni -> the SILPH SCOPE, then
    out and healed (probe148). From B4F's west side: stairs
    back to B2F, elevator up to B4F's east side. Rocket1
    (23,12) and Rocket2 (26,12) have sight range 0 - they
    fight only when talked to; beating both unlocks the
    office door (its blocks swap via the map-load callback,
    which has run by the time the second battle ends).
    Giovanni (25,3) stands behind a 2x2 desk - cells
    (24-25,4-5) are wall tiles and FACILITY's only counter
    tile ($12) is not among them, so he is talked to from
    the north room at (24,3)/(26,3)/(25,2), reached around
    the desk's west side. His team (Onix 25 / Rhyhorn 24 /
    Kangaskhan 29) fainted the probe's Blastoise from 81 HP -
    it had taken hideout attrition with no heal; Arcanine
    closed the fight out. The Scope ball appears at (25,2)
    behind his desk. Exit: elevator to B1F, Rocket5 (28,18)
    guards the door out of the elevator room, then walk out
    and heal at Celadon. Ends healed with the Scope in the
    bag and the Fresh Water unspent."""
    world = ex.world
    celpc = world.map_id("CeladonPokecenter")
    cel = world.map_id("CeladonCity")
    gc = world.map_id("GameCorner")
    b1f = world.map_id("RocketHideoutB1F")
    b2f = world.map_id("RocketHideoutB2F")
    b3f = world.map_id("RocketHideoutB3F")
    b4f = world.map_id("RocketHideoutB4F")
    elev = world.map_id("RocketHideoutElevator")
    if _bag_count(emu, ITEM_SILPH_SCOPE) >= 1:
        return _ride_battling(emu, ex, celpc) and \
            heal_at_pokecenter(emu, ex, celpc)
    if _bag_count(emu, ITEM_LIFT_KEY) < 1:
        return False
    st = emu.read_state()
    if st.map_id == b4f and st.x < 20:
        # west side: back up to B2F for the elevator
        status, _ = explore_nav(emu, ex, b4f, (19, 10))
        if emu.read_state().map_id != b3f:
            return False
        status, _ = explore_nav(emu, ex, b3f, (25, 6))
        if emu.read_state().map_id != b2f:
            return False
    if emu.read_state().map_id == b2f:
        status, _ = explore_nav(emu, ex, b2f, (24, 19))
        if emu.read_state().map_id != elev:
            return False
        if not elevator_ride(emu, ex, 2):
            return False
    if emu.read_state().map_id != b4f:
        return False
    if not talk_fight_event(emu, ex, b4f, (23, 12),
                            EV_HIDEOUT_B4_TRAINERS[0]):
        return False
    if not talk_fight_event(emu, ex, b4f, (26, 12),
                            EV_HIDEOUT_B4_TRAINERS[1]):
        return False
    for _ in range(30):
        if _event_flag(emu, EV_HIDEOUT_B4_DOOR):
            break
        emu.tick(30)
    if not _event_flag(emu, EV_HIDEOUT_B4_DOOR):
        return False
    if not talk_fight_event(emu, ex, b4f, (25, 3),
                            EV_BEAT_HIDEOUT_GIOVANNI,
                            stance_order=((1, 0), (-1, 0),
                                          (0, -1))):
        return False
    emu.wait_free(max_rounds=120, stable_needed=10)
    if not _event_flag(emu, EV_BEAT_HIDEOUT_GIOVANNI):
        return False
    if not pickup_ball(emu, ex, b4f, (25, 2), ITEM_SILPH_SCOPE):
        return False
    # exit: elevator to B1F, past Rocket5, out, heal
    status, _ = explore_nav(emu, ex, b4f, (24, 15))
    if emu.read_state().map_id != elev:
        return False
    if not elevator_ride(emu, ex, 0):
        return False
    if emu.read_state().map_id != b1f:
        return False
    if not talk_fight_event(emu, ex, b1f, (28, 18),
                            EV_HIDEOUT_B1_TRAINERS[4]):
        return False
    status, _ = explore_nav(emu, ex, b1f, (21, 2))
    if emu.read_state().map_id != gc:
        return False
    _walk_fighting(emu, ex, gc, (15, 17))
    if emu.read_state().map_id == gc:
        emu.press("down")
        _explore_settle(emu)
    if emu.read_state().map_id != cel:
        return False
    if not _ride_battling(emu, ex, celpc):
        return False
    if not heal_at_pokecenter(emu, ex, celpc):
        return False
    st = emu.read_state()
    return _bag_count(emu, ITEM_SILPH_SCOPE) >= 1 and \
        _bag_count(emu, ITEM_FRESH_WATER) >= 1 and \
        all(m.hp == m.max_hp for m in st.party)


ITEM_POKE_FLUTE = 0x49
ITEM_RARE_CANDY = 0x28
# Tower event ids (probe149): computed from
# event_constants.asm with pokered's own const macros
# (const_def/const_next/const_skip, expressions included -
# line 67 is `const_next $F0 - 2`, which a plain-number
# parser silently skips, shifting the whole Tower block
# +2) and anchored on 441 FOUND_ROCKET_HIDEOUT, 1663
# EVENT_67F, 1703 BEAT_ROCKET_HIDEOUT_GIOVANNI, plus the
# empirical rival diff: the beaten-rival save flips exactly
# bit 239. NUM_EVENTS = $A00, so wEventFlags is 320 bytes;
# bits past that are the toggleable-object flags.
EV_TOWER_RIVAL = 239
EV_TOWER_3F_TRAINERS = (241, 242, 243)
EV_TOWER_4F_TRAINERS = (249, 250, 251)
EV_TOWER_5F_TRAINERS = (258, 259, 260, 261)
EV_TOWER_PURIFIED_ZONE = 263
EV_TOWER_6F_TRAINERS = (265, 266, 267)
EV_BEAT_GHOST_MAROWAK = 271
EV_TOWER_7F_ROCKETS = (273, 274, 275)
EV_RESCUED_MR_FUJI_2 = 279
EV_GOT_POKE_FLUTE = 296
EV_RESCUED_MR_FUJI = 1231
TOWER_PURIFIED_ZONE = ((10, 8), (11, 8), (10, 9), (11, 9))
TOWER_MAROWAK_TRIGGER = (10, 16)
TOWER_RIVAL_TRIGGER = (14, 6)   # the other is (15, 5)


def _tower_stairs(emu: PyBoyEmulator, ex: Executor,
                  src_map: int, warp_cell, dst_map: int) -> bool:
    """Walk onto a Tower stair warp cell and verify the map
    change (probe149). Wild battles on the way are fled via
    flee_wild; if the walk ends standing on the warp without
    it firing, nudge off and back on."""
    ok, _ = _walk_fighting(emu, ex, src_map, warp_cell,
                           flee_wild=True)
    if emu.read_state().map_id == dst_map:
        return True
    for _ in range(6):
        st = emu.read_state()
        if st.map_id == dst_map:
            return True
        if st.in_battle:
            _drive_battle_typed(emu, flee_wild=True)
            continue
        emu.press("down")
        _explore_settle(emu)
        if emu.read_state().map_id == dst_map:
            return True
        _walk_fighting(emu, ex, src_map, warp_cell,
                       attempts=6, flee_wild=True)
    return emu.read_state().map_id == dst_map


def _tower_trainers(emu: PyBoyEmulator, ex: Executor,
                    map_id: int, cells_events) -> bool:
    """Beat a floor's trainers by (npc cell, event id),
    skipping any already flagged (probe149). Sight-triggered
    fights during one approach regularly set the NEXT
    trainer's flag too - on 5F one talk_fight's navigation
    set two further flags, on 6F all three fell in a single
    sweep - so the flag check comes first, every time."""
    for cell, event_id in cells_events:
        if _event_flag(emu, event_id):
            continue
        if not talk_fight_event(emu, ex, map_id, cell,
                                event_id, flee_wild=True):
            return False
    return True


def beat_tower_rival(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The 2F rival ambush (probe149). Assumes the player is
    on PokemonTower2F with the rival unbeaten. Stepping on
    (14,6) or (15,5) starts his script: text, then the
    battle (Rival2 #4 - the Squirtle line: Pidgeotto 25,
    Growlithe 23, Exeggcute 22, Kadabra 20, Wartortle 25),
    then his scripted exit. Verified by EVENT 239 (his
    DefeatedRival script sets it only on a win)."""
    world = ex.world
    t2 = world.map_id("PokemonTower2F")
    if _event_flag(emu, EV_TOWER_RIVAL):
        return True
    if emu.read_state().map_id != t2:
        return False
    _walk_fighting(emu, ex, t2, TOWER_RIVAL_TRIGGER,
                   flee_wild=True)
    for _ in range(240):
        st = emu.read_state()
        if _event_flag(emu, EV_TOWER_RIVAL) \
                and not st.in_dialog and not st.in_battle:
            emu.tick(120)
            st = emu.read_state()
            if not st.in_dialog and not st.in_battle:
                break
        if st.in_battle:
            _drive_battle_typed(emu, flee_wild=True)
            continue
        if st.in_dialog:
            _press10(emu, "a")
            emu.tick(30)
            continue
        emu.tick(30)
    return _event_flag(emu, EV_TOWER_RIVAL)


def _use_purified_zone(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Step into the 5F purified zone and let its script
    heal the party (probe149). The zone - cells (10-11,
    8-9) - heals on ENTRY (EVENT 263 via CheckAndSetEvent,
    HealParty, fade to white, text); leaving resets the
    event, so it is reusable within the climb. Verified by
    party HP from RAM. The mid-climb arrival in the probe
    had Blastoise at 16/142, restored to full here before
    the floor's Channelers were touched."""
    world = ex.world
    t5 = world.map_id("PokemonTower5F")

    def full():
        st = emu.read_state()
        return all(m.hp == m.max_hp for m in st.party)

    if full():
        return True
    _walk_fighting(emu, ex, t5, TOWER_PURIFIED_ZONE[0],
                   flee_wild=True)
    for _ in range(60):
        if full():
            break
        st = emu.read_state()
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
        emu.tick(30)
    return full()


def beat_ghost_marowak(emu: PyBoyEmulator, ex: Executor) -> bool:
    """6F: the Channelers, the Rare Candy chokepoint, and
    the ghost MAROWAK (probe149). Assumes the player is on
    PokemonTower6F; ends on 7F.

    The world model's only south corridor on 6F runs
    through (6,8) - the Rare Candy ball's cell. The grid
    reads the cell as open floor (item balls are objects,
    not tiles), so _walk_to wedges against the ball
    forever; taking the candy (bag-verified) clears the
    corridor. The Marowak event itself: stepping on
    (10,16) runs "Be gone..." and starts the RESTLESS_SOUL
    Lv30 battle. In RAM it reads as a WILD battle
    (wIsInBattle == 1) - but fleeing counts as
    not-defeated (the script shoves the player right and
    re-arms), and thrown balls are dodged by script, so
    the fight is driven with flee_wild OFF and the ball
    counts are compared before/after. Winning sets EVENT
    271 and opens the (9,16) stairs."""
    world = ex.world
    t6 = world.map_id("PokemonTower6F")
    t7 = world.map_id("PokemonTower7F")
    if emu.read_state().map_id not in (t6, t7):
        return False
    if not _tower_trainers(
            emu, ex, t6,
            [((12, 10), EV_TOWER_6F_TRAINERS[0]),
             ((9, 5), EV_TOWER_6F_TRAINERS[1]),
             ((16, 5), EV_TOWER_6F_TRAINERS[2])]):
        return False
    if not _event_flag(emu, EV_BEAT_GHOST_MAROWAK):
        if _bag_count(emu, ITEM_RARE_CANDY) == 0:
            ok, _ = _walk_fighting(emu, ex, t6, (6, 7),
                                   flee_wild=True)
            st = emu.read_state()
            if not ok or (st.x, st.y) != (6, 7):
                return False
            _face_and_talk(emu, (6, 8))
            _ride_text(emu, rounds=60)
            if _bag_count(emu, ITEM_RARE_CANDY) != 1:
                return False
        balls0 = _bag_count(emu, ITEM_POKE_BALL)
        gballs0 = _bag_count(emu, ITEM_GREAT_BALL)
        ok, _ = _walk_fighting(emu, ex, t6,
                               TOWER_MAROWAK_TRIGGER,
                               flee_wild=True)
        if not ok:
            explore_nav(emu, ex, t6, TOWER_MAROWAK_TRIGGER,
                        flee_wild=True)
        fought = False
        for _ in range(240):
            st = emu.read_state()
            if _event_flag(emu, EV_BEAT_GHOST_MAROWAK) \
                    and not st.in_dialog and not st.in_battle:
                emu.tick(90)
                if not emu.read_state().in_dialog:
                    break
            if st.in_battle:
                _drive_battle_typed(emu)   # never flee, ever
                fought = True
                continue
            if st.in_dialog:
                _press10(emu, "a")
                emu.tick(30)
                continue
            emu.tick(30)
        if not fought or not _event_flag(
                emu, EV_BEAT_GHOST_MAROWAK):
            return False
        if _bag_count(emu, ITEM_POKE_BALL) != balls0 or \
                _bag_count(emu, ITEM_GREAT_BALL) != gballs0:
            return False
    if emu.read_state().map_id == t7:
        return True
    return _tower_stairs(emu, ex, t6, (9, 16), t7)


def rescue_mr_fuji(emu: PyBoyEmulator, ex: Executor) -> bool:
    """7F: the three Rockets and Mr. Fuji (probe149).
    Assumes the player is on PokemonTower7F; ends in
    MrFujisHouse with the POKE FLUTE in the bag. The
    Rockets (#19 Zubat/Zubat/Golbat 25, #20 Koffing/
    Drowzee 26, #21 Zubat/Rattata/Raticate/Zubat 23, all
    sight 3) each walk off after their battle. Fuji (10,3)
    is talked to once they are gone: his text sets EVENTS
    1231 + 279 and his map script warps the player to his
    house, where Fuji (3,1) hands over the flute (EVENT
    296, bag-verified)."""
    world = ex.world
    t7 = world.map_id("PokemonTower7F")
    house = world.map_id("MrFujisHouse")
    if _bag_count(emu, ITEM_POKE_FLUTE) >= 1:
        return True
    if emu.read_state().map_id != t7:
        return False
    if not _tower_trainers(
            emu, ex, t7,
            [((9, 11), EV_TOWER_7F_ROCKETS[0]),
             ((12, 9), EV_TOWER_7F_ROCKETS[1]),
             ((9, 7), EV_TOWER_7F_ROCKETS[2])]):
        return False
    talked = False
    for dx, dy in ((0, 1), (1, 0), (-1, 0), (0, -1)):
        stance = (10 + dx, 3 + dy)
        ok, _ = _walk_fighting(emu, ex, t7, stance,
                               flee_wild=True)
        st = emu.read_state()
        if (st.x, st.y) != stance:
            continue
        _face_and_talk(emu, (10, 3))
        talked = True
        break
    if not talked:
        return False
    for _ in range(300):
        st = emu.read_state()
        if st.map_id == house:
            break
        if st.in_battle:
            _drive_battle_typed(emu, flee_wild=True)
            continue
        if st.in_dialog:
            _press10(emu, "a")
            emu.tick(30)
            continue
        emu.tick(30)
    st = emu.read_state()
    if st.map_id != house or not _event_flag(
            emu, EV_RESCUED_MR_FUJI) or not _event_flag(
            emu, EV_RESCUED_MR_FUJI_2):
        return False
    ok, _ = _walk_fighting(emu, ex, house, (3, 2),
                           flee_wild=True)
    st = emu.read_state()
    if not ok or (st.x, st.y) != (3, 2):
        return False
    _face_and_talk(emu, (3, 1))
    _ride_text(emu, rounds=120)
    for _ in range(120):
        if _bag_count(emu, ITEM_POKE_FLUTE) >= 1:
            break
        st = emu.read_state()
        if st.in_dialog:
            _press10(emu, "a")
            emu.tick(30)
        else:
            emu.tick(30)
    return _bag_count(emu, ITEM_POKE_FLUTE) >= 1 and \
        _event_flag(emu, EV_GOT_POKE_FLUTE)


def pokemon_tower(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Celadon -> Lavender -> the Pokemon Tower -> the
    POKE FLUTE (probe149). Travel is per-map HOP rides
    through the Underground Path (the Saffron gatehouses
    are never entered; the Fresh Water is re-verified in
    the bag at the end). Inside the Tower: the CEMETERY
    grids serve _walk_to well (unlike the hideout's
    FACILITY set), talk_fight_event's learned graph does
    the trainer approaches, and wild Gastly/Cubone/Haunter
    on 3F+ are FLED (flee_wild) - in the probe every wild
    battle was escaped, none fought. The climb order on
    5F is purified-zone heal FIRST, then the Channelers,
    then heal again before 6F: the 3F/4F Channelers'
    Night Shade / status attrition is the leg's real cost
    (Blastoise arrived on 5F at 16/142 in the probe).
    Ends back in Lavender Town, healed at the Pokecenter,
    flute + Scope + Fresh Water all in the bag. Faithful
    port of probe149's phases; NOT cold-replayed
    end-to-end (the probe ran phase-by-phase from saved
    intermediates)."""
    world = ex.world
    lav = world.map_id("LavenderTown")
    lavpc = world.map_id("LavenderPokecenter")
    t1 = world.map_id("PokemonTower1F")
    t2 = world.map_id("PokemonTower2F")
    t3 = world.map_id("PokemonTower3F")
    t4 = world.map_id("PokemonTower4F")
    t5 = world.map_id("PokemonTower5F")
    t6 = world.map_id("PokemonTower6F")
    house = world.map_id("MrFujisHouse")
    if _bag_count(emu, ITEM_POKE_FLUTE) < 1:
        st = emu.read_state()
        if st.map_id not in (lav, t1, t2, t3, t4, t5, t6,
                             house):
            if not _hop_ride(emu, ex,
                             ["Route7",
                              "UndergroundPathRoute7",
                              "UndergroundPathWestEast",
                              "UndergroundPathRoute8",
                              "Route8", "LavenderTown"]):
                return False
            if _bag_count(emu, ITEM_FRESH_WATER) < 1:
                return False
        if emu.read_state().map_id == lav:
            if not _ride_battling(emu, ex, t1):
                return False
        if emu.read_state().map_id == t1:
            if not _tower_stairs(emu, ex, t1, (18, 9), t2):
                return False
        if emu.read_state().map_id == t2:
            if not beat_tower_rival(emu, ex):
                return False
            if not _tower_stairs(emu, ex, t2, (3, 9), t3):
                return False
        if emu.read_state().map_id == t3:
            if not _tower_trainers(
                    emu, ex, t3,
                    [((12, 3), EV_TOWER_3F_TRAINERS[0]),
                     ((9, 8), EV_TOWER_3F_TRAINERS[1]),
                     ((10, 13), EV_TOWER_3F_TRAINERS[2])]):
                return False
            if not _tower_stairs(emu, ex, t3, (18, 9), t4):
                return False
        if emu.read_state().map_id == t4:
            if not _tower_trainers(
                    emu, ex, t4,
                    [((5, 10), EV_TOWER_4F_TRAINERS[0]),
                     ((15, 7), EV_TOWER_4F_TRAINERS[1]),
                     ((14, 12), EV_TOWER_4F_TRAINERS[2])]):
                return False
            if not _tower_stairs(emu, ex, t4, (3, 9), t5):
                return False
        if emu.read_state().map_id == t5:
            if not _use_purified_zone(emu, ex):
                return False
            if not _tower_trainers(
                    emu, ex, t5,
                    [((17, 7), EV_TOWER_5F_TRAINERS[0]),
                     ((14, 3), EV_TOWER_5F_TRAINERS[1]),
                     ((6, 10), EV_TOWER_5F_TRAINERS[2]),
                     ((9, 16), EV_TOWER_5F_TRAINERS[3])]):
                return False
            if not _use_purified_zone(emu, ex):
                return False
            if not _tower_stairs(emu, ex, t5, (18, 9), t6):
                return False
        if emu.read_state().map_id == t6:
            if not beat_ghost_marowak(emu, ex):
                return False
        if not rescue_mr_fuji(emu, ex):
            return False
    # out of the house, heal in Lavender, final checks
    if emu.read_state().map_id == house:
        _walk_fighting(emu, ex, house, (3, 7), flee_wild=True)
        for _ in range(10):
            if emu.read_state().map_id == lav:
                break
            emu.press("down")
            _explore_settle(emu)
    if emu.read_state().map_id != lav:
        return False
    if not _ride_battling(emu, ex, lavpc):
        return False
    if not heal_at_pokecenter(emu, ex, lavpc):
        return False
    st = emu.read_state()
    return _bag_count(emu, ITEM_POKE_FLUTE) >= 1 and \
        _bag_count(emu, ITEM_SILPH_SCOPE) >= 1 and \
        _bag_count(emu, ITEM_FRESH_WATER) >= 1 and \
        all(m.hp == m.max_hp for m in st.party)




# ------------------------------------------------------------------
# Saffron gates + Silph Co. (probes 150/151). The gates were
# opened in probe150 (Fresh Water to the Route 8 guard), TM29
# came from Mr. Psychic, and the Card Key circuit, the rival,
# Lapras, Giovanni and the Master Ball were finished in
# probe151 after a two-lead grind. Three pieces of machinery
# generalise beyond this leg and are ported with it:
#   * _silph_dfs: a MODEL-FREE single-step DFS navigator for
#     FACILITY interiors. explore_nav's learned graph cannot
#     cross Silph's pad/door-split components and its block
#     grid lies about half-block features; the DFS probes
#     every step against the real game (the only oracle),
#     records empirical edges, A-presses card-key doors when
#     a step into a door block fails, and fences warp cells.
#   * the elevator cab ride (panel menu steered by
#     wListScrollOffset + wCurrentMenuItem).
#   * the route-grind discipline (lead-and-solo, HP/PP gate,
#     heal anchors, hand-crossed gatehouses and waypoint
#     chains where journey rides strand).
ITEM_CARD_KEY = 0x30
ITEM_MASTER_BALL = 0x01
ITEM_TM29 = 0xE5                # TM ids are 0xC9 + (n - 1)
ITEM_ANTIDOTE = 0x0B
ADDR_STATUS_FLAGS1 = 0xD728     # bit 6: gave the guards a drink
ADDR_STATUS_FLAGS4 = 0xD72E     # bit 0: BIT_GOT_LAPRAS
EV_TM29 = 944
EV_SILPH_RIVAL = 1856
EV_GOT_MASTER_BALL = 1933
EV_SILPH_GIOVANNI = 1935
EV_SILPH_11F_ROCKET_WEST = 1924
SPECIES_LAPRAS = 0x13
SILPH_BLASTOISE = 28            # internal species ids
SILPH_ARCANINE = 20
# Silph trainer events / object cells by FLOOR NUMBER
# (probe151lib; event ids anchored on 1826/1856/1935).
SILPH_EV = {
    2: (1778, 1779, 1780, 1781),
    3: (1794, 1795),
    4: (1810, 1811, 1812),
    5: (1826, 1827, 1828, 1829),
    6: (1846, 1847, 1848),
    7: (1861, 1862, 1863, 1864),
    8: (1874, 1875, 1876),
    9: (1890, 1891, 1892),
    10: (1905, 1906),
    11: (1924, 1925),
}
SILPH_XY = {
    2: ((5, 12), (24, 13), (16, 11), (24, 7)),
    3: ((20, 7), (7, 9)),
    4: ((9, 14), (14, 6), (26, 10)),
    5: ((8, 16), (8, 3), (18, 10), (28, 4)),
    6: ((17, 3), (7, 8), (14, 15)),
    7: ((13, 1), (2, 13), (20, 2), (19, 14)),
    8: ((19, 2), (10, 2), (12, 15)),
    9: ((2, 4), (21, 13), (13, 16)),
    10: ((1, 9), (10, 2)),
    11: ((3, 16), (15, 9)),
}
# Card-key door blocks by floor (tile top-lefts, from the
# .GateCoordinates in each floor's pokered script; 11F's
# office door is the special block at (6, 12)).
SILPH_GATES = {
    2: ((4, 4), (4, 10)),
    3: ((8, 8), (16, 8)),
    4: ((4, 12), (12, 8)),
    5: ((6, 4), (6, 12), (14, 10)),
    6: ((4, 12),),
    7: ((10, 6), (20, 4), (20, 12)),
    8: ((6, 8),),
    9: ((2, 8), (18, 4), (18, 10), (10, 12)),
    10: ((10, 8),),
    11: ((6, 12),),
}
SILPH_FLOOR_NAMES = {
    1: "SilphCo1F", 2: "SilphCo2F", 3: "SilphCo3F",
    4: "SilphCo4F", 5: "SilphCo5F", 6: "SilphCo6F",
    7: "SilphCo7F", 8: "SilphCo8F", 9: "SilphCo9F",
    10: "SilphCo10F", 11: "SilphCo11F",
}
SILPH_FLOOR_NUM = {name: n for n, name in
                   SILPH_FLOOR_NAMES.items()}
SILPH_ELEV_DOOR = {1: (20, 0), 2: (20, 0), 3: (20, 0),
                   4: (20, 0), 5: (20, 0), 6: (18, 0),
                   7: (18, 0), 8: (18, 0), 9: (18, 0),
                   10: (12, 0), 11: (13, 0)}
# Fences for the rival-floor circuits (warp cells + NPC
# bodies), from probe150o/probe151r-t.
SILPH_W9 = {(14, 0), (16, 0), (18, 0), (9, 3), (17, 15)}
SILPH_D9 = {(18, 4), (18, 10), (10, 12), (2, 8)}
SILPH_W3 = {(26, 0), (24, 0), (20, 0), (23, 11), (3, 3),
            (3, 15), (27, 3), (3, 11), (11, 11), (27, 15),
            (20, 7), (7, 9)}
SILPH_W7 = {(16, 0), (22, 0), (18, 0), (5, 7), (5, 3),
            (21, 15), (3, 7), (1, 5)}
SILPH_W7B = SILPH_W7 | {(1, 9)}
SILPH_DOOR11 = {(6, 12), (7, 12), (6, 13), (7, 13)}
SILPH_W11 = {(9, 0), (13, 0), (5, 5), (3, 2), (7, 5),
             (10, 5), (6, 9), (15, 9)}
SILPH_W11B = SILPH_W11 | {(3, 16)}
# Route-grind trainer tables (event, object cell), in the
# order probe151 fought them. Route 12's Fisher5 (ev1160,
# (6, 87)) is DELIBERATELY ABSENT: his west-shore pocket
# is tile-sealed (one-way ledge in, no walk-out; Dig is
# cave-only in Gen 1) - the probe lineage that beat him
# was abandoned and he is skipped by design.
GRIND_R8 = [(1073, (8, 5)), (1076, (26, 3)), (1077, (26, 4)),
            (1078, (26, 5)), (1079, (26, 6)), (1075, (42, 6)),
            (1080, (46, 13)), (1081, (51, 12))]
GRIND_R10 = [(1110, (7, 54)), (1106, (3, 57)),
             (1109, (3, 61)), (1107, (14, 64))]
GRIND_R12N = [(1154, (14, 31)), (1155, (5, 39)),
              (1158, (12, 40)), (1159, (9, 52))]
GRIND_R12S = [(1157, (14, 76)), (1156, (11, 92))]
GRIND_R13 = [(1173, (50, 5)), (1169, (49, 10)),
             (1170, (48, 10)), (1175, (33, 6)),
             (1176, (32, 6)), (1171, (27, 9)),
             (1172, (23, 10)), (1174, (12, 4)),
             (1177, (10, 7)), (1178, (7, 13))]
GRIND_R14 = [(1186, (15, 6)), (1187, (12, 11)),
             (1188, (14, 15)), (1185, (4, 4)),
             (1194, (4, 31)), (1192, (4, 30)),
             (1193, (15, 30)), (1191, (5, 39)),
             (1190, (6, 49)), (1189, (15, 31))]
# Route 12 pier waypoint chain (the executor's planner
# stalls on the ~85-cell pier; hop the through-cells).
PIER_WAYPOINTS_S = [(10, 25), (14, 28), (11, 30), (10, 45),
                    (10, 61), (13, 73), (14, 78), (5, 80),
                    (11, 85), (13, 95), (11, 106)]
PIER_WAYPOINTS_N = [(13, 95), (11, 85), (14, 78), (13, 73),
                    (10, 61), (10, 45), (11, 30), (14, 28),
                    (10, 25)]


class _SilphWhiteout(Exception):
    """A battle was lost during a Silph/route beat: the
    party whiteouted (money halved, map changed to the
    respawn town). Callers treat it as a failed beat."""


_SILPH_WALK = None


def _silph_walkability():
    global _SILPH_WALK
    if _SILPH_WALK is None:
        import json as _json
        path = (Path(__file__).parent / "data" / "maps"
                / "pokered_walkability.json")
        _SILPH_WALK = _json.loads(path.read_text())
    return _SILPH_WALK


def _silph_warps_of(map_id: int) -> set:
    return {(wp["x"], wp["y"]) for wp in
            _silph_walkability()[str(map_id)]["warps"]}


def _silph_doors_of(floor_num: int) -> set:
    cells = set()
    for (bx, by) in SILPH_GATES.get(floor_num, ()):
        for dx in (0, 1):
            for dy in (0, 1):
                cells.add((bx + dx, by + dy))
    return cells


def _silph_floor_map(emu: PyBoyEmulator, ex: Executor,
                     floor_num: int) -> int:
    return ex.world.map_id(SILPH_FLOOR_NAMES[floor_num])


def _party_xp(emu: PyBoyEmulator, idx: int) -> int:
    """EXP from the party struct (3-byte big-endian at +14
    in each 44-byte mon, probe151lib)."""
    from agent.state import ADDR_PARTY_MONS
    base = ADDR_PARTY_MONS + 44 * idx + 14
    mem = emu.pb.memory
    return (mem[base] << 16) | (mem[base + 1] << 8) | \
        mem[base + 2]


def _silph_ride_scene(emu: PyBoyEmulator, done,
                      rounds: int = 300) -> bool:
    """Ride a scripted scene (dialog -> battle -> dialog)
    until done(); battles are fought with the typed ranker,
    never fled (probe150's ride_scene)."""
    for _ in range(rounds):
        if done():
            emu.tick(60)
            if done():
                return True
        st = emu.read_state()
        if st.in_battle:
            _drive_battle_typed(emu)
            continue
        if st.in_dialog:
            _press10(emu, "a")
            emu.tick(30)
            continue
        emu.tick(30)
    return done()


def _silph_settle(emu: PyBoyEmulator) -> None:
    """The grind's battle settle: fight with the typed
    ranker, close dialogs, NO bag maintenance (healing is
    the nurse's / PC's job; the bag's Super Potions are
    reserved for the rival top-off). A lost battle is a
    whiteout - money halves and the map changes - and
    raises _SilphWhiteout instead of letting a walk
    continue on a respawned map (probe151b's lesson)."""
    for _ in range(10):
        st = emu.read_state()
        if st.in_battle:
            money0, map0 = st.money, st.map_id
            _drive_battle_typed(emu)
            st2 = emu.read_state()
            if st2.map_id != map0 or st2.money < money0:
                raise _SilphWhiteout(
                    f"whiteout on map {map0}: money "
                    f"{money0}->{st2.money}")
            if all(m.hp == 0 for m in st2.party):
                raise _SilphWhiteout("blackout: party down")
            continue
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
            continue
        break


_SILPH_DIRS = {"up": (0, -1), "down": (0, 1),
               "left": (-1, 0), "right": (1, 0)}
_SILPH_REV = {"up": "down", "down": "up",
              "left": "right", "right": "left"}


def _silph_dfs(emu: PyBoyEmulator, map_id: int, fences,
               targets, stop_pred=None, doors=(),
               step_cap: int = 9000):
    """Model-free single-step DFS (silphlib, probes
    150/151): probe every step against the real game,
    record empirical edges, backtrack on recorded edges.
    Warp cells and NPC bodies belong in `fences`; a step
    that fails into a `doors` cell is retried after an
    A-press (card-key doors open by facing them - using
    the key from the bag is a decoy). Returns the target
    cell reached, or None (also None when a probe step
    warps unexpectedly). Battles met en route are fought
    by _silph_settle; a whiteout raises _SilphWhiteout."""
    def pos():
        st = emu.read_state()
        return (st.map_id, st.x, st.y)

    def settle():
        _silph_settle(emu)

    edges = {}
    start = pos()[1:]
    visited = {start}
    stack = [(start, iter(_SILPH_DIRS), None)]

    def probe_step(cell, btn):
        emu.press(btn)
        _explore_settle(emu)
        settle()
        for _ in range(3):
            emu.tick(25)
            settle()
        st = emu.read_state()
        if st.map_id != map_id:
            edges[(cell, btn)] = "warp"
            return "warp"
        land = (st.x, st.y)
        if land == cell and (cell[0] + _SILPH_DIRS[btn][0],
                             cell[1] + _SILPH_DIRS[btn][1]) \
                in doors:
            _press10(emu, "a")
            emu.tick(60)
            settle()
            emu.press(btn)
            _explore_settle(emu)
            settle()
            st = emu.read_state()
            if st.map_id != map_id:
                edges[(cell, btn)] = "warp"
                return "warp"
            land = (st.x, st.y)
        edges[(cell, btn)] = None if land == cell else land
        return edges[(cell, btn)]

    steps = 0
    while stack:
        if steps > step_cap:
            return None
        cur = pos()[1:]
        if stop_pred is not None and steps > 0 and \
                stop_pred():
            break
        if cur in targets and steps > 0:
            break
        cell, it, entry = stack[-1]
        if cur != cell:
            if cur not in visited:
                visited.add(cur)
            stack = [(cur, iter(_SILPH_DIRS), None)]
            continue
        advanced = False
        for btn in it:
            if (cell, btn) in edges:
                continue
            dx, dy = _SILPH_DIRS[btn]
            nb = (cell[0] + dx, cell[1] + dy)
            if nb in fences:
                edges[(cell, btn)] = None
                continue
            land = probe_step(cell, btn)
            steps += 1
            if land in (None, "warp"):
                if land == "warp":
                    return None
                continue
            if land not in visited:
                visited.add(land)
                stack.append((land, iter(_SILPH_DIRS), btn))
                advanced = True
                break
            probe_step(land, _SILPH_REV[btn])
        if not advanced:
            stack.pop()
            if stack and entry is not None:
                probe_step(cell, _SILPH_REV[entry])
    if stop_pred is not None and stop_pred():
        return pos()[1:]
    return pos()[1:] if pos()[1:] in targets else None


def _silph_step(emu: PyBoyEmulator, btn: str):
    emu.press(btn)
    _explore_settle(emu)
    _silph_settle(emu)
    return emu.read_state()


def _silph_step_onto(emu: PyBoyEmulator, tx: int, ty: int,
                     stance) -> None:
    dx, dy = tx - stance[0], ty - stance[1]
    btn = {(0, 1): "down", (0, -1): "up", (1, 0): "right",
           (-1, 0): "left"}[(dx, dy)]
    _silph_step(emu, btn)


def _silph_step_off(emu: PyBoyEmulator, map_id: int):
    """Step off a warp cell we just landed on (any
    direction that moves us within the same map)."""
    for btn in ("up", "down", "left", "right"):
        before = emu.read_state()
        _silph_step(emu, btn)
        st = emu.read_state()
        if st.map_id == map_id and \
                (st.x, st.y) != (before.x, before.y):
            return (st.x, st.y)
    return None


def _silph_transit(emu: PyBoyEmulator, map_id: int,
                   targets, floor_num=None,
                   extra_fences=(), doors=None) -> bool:
    """DFS to any of `targets` on the current component
    (probe151lib's transit)."""
    fences = _silph_warps_of(map_id) | set(extra_fences)
    if doors is None:
        doors = _silph_doors_of(floor_num) \
            if floor_num is not None else ()
    _silph_dfs(emu, map_id, fences, set(targets),
               doors=doors)
    st = emu.read_state()
    return (st.x, st.y) in targets and st.map_id == map_id


def _silph_elevator(emu: PyBoyEmulator, ex: Executor,
                    floor_map: int) -> bool:
    """Ride the Silph elevator from inside the cab to
    floor_map, then step out (probe150's silph_elevator).
    The panel is at (3, 0): stand at (3, 1), face up, and
    steer the floor list by scroll + item."""
    world = ex.world
    elev = world.map_id("SilphCoElevator")
    if emu.read_state().map_id != elev:
        return False
    mem = emu.pb.memory
    for _ in range(24):
        st = emu.read_state()
        if (st.x, st.y) == (3, 1):
            break
        if st.x < 3:
            _step(emu, ex, "right")
        elif st.y > 1:
            _step(emu, ex, "up")
        elif st.y < 1:
            _step(emu, ex, "down")
        else:
            _step(emu, ex, "left")
    if (emu.read_state().x, emu.read_state().y) != (3, 1):
        return False
    emu.press("up")
    emu.tick(15)
    _press10(emu, "a")
    emu.tick(90)
    want = SILPH_FLOOR_NUM[world.map_name(floor_map)] - 1
    getter = lambda: mem[ADDR_MENU_SCROLL] + \
        mem[ADDR_MENU_ITEM]
    if not _steer_menu(emu, getter, want, rounds=40):
        return False
    _press10(emu, "a")
    emu.tick(240)
    emu.wait_free(max_rounds=40, stable_needed=4)
    for _ in range(24):
        st = emu.read_state()
        if (st.x, st.y) == (2, 1):
            break
        if st.x > 2:
            _step(emu, ex, "left")
        elif st.x < 2:
            _step(emu, ex, "right")
        elif st.y > 1:
            _step(emu, ex, "up")
        else:
            _step(emu, ex, "down")
    for _ in range(8):
        if emu.read_state().map_id != elev:
            break
        _step(emu, ex, "down")
        _explore_settle(emu)
    return emu.read_state().map_id == floor_map


def _silph_enter_cab(emu: PyBoyEmulator, ex: Executor,
                     floor_map: int) -> bool:
    world = ex.world
    elev = world.map_id("SilphCoElevator")
    if emu.read_state().map_id == elev:
        return True
    num = SILPH_FLOOR_NUM[world.map_name(floor_map)]
    explore_nav(emu, ex, floor_map, SILPH_ELEV_DOOR[num])
    for _ in range(10):
        if emu.read_state().map_id == elev:
            return True
        emu.tick(30)
    return emu.read_state().map_id == elev


def _silph_cab_to(emu: PyBoyEmulator, ex: Executor,
                  floor_map: int) -> bool:
    """Elevator travel from the current floor's elevator
    component (probe151lib's cab_to): DFS to the cab door
    stance (transit crosses card-key doors), step in,
    ride, and step off the arrival warp cell so later DFS
    trees are not rooted on a live warp. From pad-only
    components (7F west room, 3F island) the caller must
    pad to an elevator component first - cab_to does NOT
    cross pads."""
    world = ex.world
    cur = emu.read_state().map_id
    if cur == floor_map:
        return True
    if cur == world.map_id("SilphCoElevator"):
        return _silph_elevator(emu, ex, floor_map)
    num = SILPH_FLOOR_NUM[world.map_name(cur)]
    door = SILPH_ELEV_DOOR[num]
    stance = (door[0], door[1] + 1)
    st = emu.read_state()
    if (st.x, st.y) != stance:
        if not _silph_transit(emu, cur, {stance},
                              floor_num=num):
            if not _silph_enter_cab(emu, ex, cur):
                return False
            return _silph_elevator(emu, ex, floor_map)
    _silph_step(emu, "up")
    if emu.read_state().map_id != \
            world.map_id("SilphCoElevator"):
        return False
    if not _silph_elevator(emu, ex, floor_map):
        return False
    num2 = SILPH_FLOOR_NUM[world.map_name(floor_map)]
    st = emu.read_state()
    if (st.x, st.y) == SILPH_ELEV_DOOR[num2]:
        _silph_step(emu, "down")
    return True


def _silph_nurse_heal(emu: PyBoyEmulator, ex: Executor) \
        -> bool:
    """The 9F nurse (3, 14): a free full heal inside the
    building (probe150/probe151lib). Caller must be on
    9F; ends at the nurse stance."""
    world = ex.world
    s9 = world.map_id("SilphCo9F")
    if emu.read_state().map_id != s9:
        return False

    def full():
        st = emu.read_state()
        return all(m.hp == m.max_hp for m in st.party)

    if full():
        return True
    if not _silph_transit(emu, s9, {(3, 13), (3, 15),
                                    (2, 14), (4, 14)},
                          floor_num=9):
        return False
    _face_and_talk(emu, (3, 14))
    _silph_ride_scene(emu, full, rounds=120)
    return full()


def _step_until_map(emu: PyBoyEmulator, btn: str,
                    want_map: int, tries: int = 16) -> bool:
    """Press a direction until the map changes (border
    crossings and door warps on foot)."""
    for _ in range(tries):
        if emu.read_state().map_id == want_map:
            return True
        emu.press(btn)
        _explore_settle(emu)
        st = emu.read_state()
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
    return emu.read_state().map_id == want_map


def _cross_r8_gate_west(emu: PyBoyEmulator,
                        ex: Executor) -> bool:
    """Cross the Route 8 gatehouse WESTBOUND by hand
    (probe151r): journey rides strand ON the gatehouse
    warp cell without firing it, and the warp tile is
    the door's BOTTOM cell - (8, 10), not (8, 9), which
    is a dead floor tile at the building front. Enter at
    (8, 10), walk the corridor west on row 4, exit at
    (1, 10), then off Route 8's west edge into Saffron."""
    world = ex.world
    r8 = world.map_id("Route8")
    gate = world.map_id("Route8Gate")
    saf = world.map_id("SaffronCity")
    if emu.read_state().map_id != r8:
        return False
    ok, _ = _walk_fighting(emu, ex, r8, (9, 10),
                           attempts=30)
    st = emu.read_state()
    if (st.x, st.y) != (9, 10):
        return False
    if not _step_until_map(emu, "left", gate):
        return False
    if not _step_until_map(emu, "left", r8):
        return False
    return _step_until_map(emu, "left", saf)


def saffron_gates(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Lavender -> Saffron through the Route 8 gatehouse
    (the guards take the Fresh Water; wStatusFlags1 bit 6
    is the RAM proof) and TM29 Psychic from Mr. Psychic's
    house (probe150's gate/saffron phases + probe151r's
    hand-crossed gate). The Fresh Water itself is bought
    in celadon_leg's vending beat. Idempotent: with the
    bit set and TM29 held there is nothing to do."""
    world = ex.world
    saf = world.map_id("SaffronCity")
    safpc = world.map_id("SaffronPokecenter")
    psy = world.map_id("MrPsychicsHouse")
    mem = emu.pb.memory
    if not (mem[ADDR_STATUS_FLAGS1] & 0x40):
        if _bag_count(emu, ITEM_FRESH_WATER) < 1:
            return False
        if emu.read_state().map_id != \
                world.map_id("Route8"):
            if not _hop_ride(emu, ex, ["Route8"]):
                return False
        if not _cross_r8_gate_west(emu, ex):
            return False
    if emu.read_state().map_id == saf:
        if not _ride_battling(emu, ex, safpc):
            return False
        if not heal_at_pokecenter(emu, ex, safpc):
            return False
    if _bag_count(emu, ITEM_TM29) < 1:
        if not _ride_battling(emu, ex, psy):
            return False
        # Mr. Psychic (5, 3): talk until the gift event
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            stance = (5 + dx, 3 + dy)
            ok, _ = _walk_fighting(emu, ex, psy, stance)
            st = emu.read_state()
            if not ok or (st.x, st.y) != stance:
                continue
            _face_and_talk(emu, (5, 3))
            if _silph_ride_scene(
                    emu, lambda: _event_flag(emu, EV_TM29),
                    rounds=120):
                break
    return bool(mem[ADDR_STATUS_FLAGS1] & 0x40) and \
        _bag_count(emu, ITEM_TM29) >= 1


def _mart_clerk_menu(emu: PyBoyEmulator, ex: Executor,
                     mart: int) -> bool:
    """Face the Saffron Mart clerk from (2, 5) and open
    the BUY/SELL/QUIT menu (menuY 1; probe150i)."""
    mem = emu.pb.memory
    try:
        ex._walk_to(mart, (2, 5))
    except StuckError:
        return False
    _step(emu, ex, "left")          # stand (1, 5)
    _press10(emu, "a")
    emu.tick(60)
    for _ in range(40):
        if mem[ADDR_MENU_TOP_Y] == 1:
            return True
        _press10(emu, "a")
        emu.tick(25)
    return mem[ADDR_MENU_TOP_Y] == 1


def _mart_sell_stack(emu: PyBoyEmulator, item_id: int) \
        -> int:
    """Sell the whole bag stack of item_id at the open
    clerk menu; money delta is the ground truth
    (probe150i's sell_stack)."""
    mem = emu.pb.memory
    idx = _bag_index(emu, item_id)
    if idx is None:
        return 0
    qty = mem[ADDR_BAG_ITEMS + 2 * idx + 1]
    money0 = emu.read_state().money
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 1):
        return 0
    _press10(emu, "a")              # SELL -> list

    def _menu_y4():
        for _ in range(60):
            if mem[ADDR_MENU_TOP_Y] == 4:
                return True
            emu.tick(20)
        return mem[ADDR_MENU_TOP_Y] == 4

    if not _menu_y4():
        return 0
    if not _steer_menu(
            emu, lambda: mem[ADDR_MENU_SCROLL]
            + mem[ADDR_MENU_ITEM], idx):
        return 0
    _press10(emu, "a")
    emu.tick(60)
    for _ in range(qty - 1):        # max the quantity out
        _press10(emu, "up")
        emu.tick(25)
    for _ in range(40):
        _press10(emu, "a")
        emu.tick(25)
        if emu.read_state().money != money0:
            break
    return emu.read_state().money - money0


def _free_bag_space(emu: PyBoyEmulator, ex: Executor) \
        -> bool:
    """Gen 1's bag holds 20 item STACKS; the probe's
    first Card Key pickup failed on the cap with the bag
    at 20/20. Before the key, free space the way the
    probe did (probe150i): sell the Nugget (¥5,000) and
    the Antidote stack at Saffron Mart, ending at <= 19
    stacks. Caller must be in Saffron City."""
    world = ex.world
    mart = world.map_id("SaffronMart")
    mem = emu.pb.memory
    if mem[ADDR_NUM_BAG_ITEMS] <= 19:
        return True
    if not _ride_battling(emu, ex, mart):
        return False
    for item in (ITEM_NUGGET, ITEM_ANTIDOTE):
        if _bag_count(emu, item) < 1:
            continue
        if not _mart_clerk_menu(emu, ex, mart):
            return False
        _mart_sell_stack(emu, item)
        for _ in range(12):
            if mem[ADDR_MENU_TOP_Y] == 1:
                break
            _press10(emu, "b")
            emu.tick(25)
        if mem[ADDR_MENU_TOP_Y] == 1:
            _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 2)
            _press10(emu, "a")
            emu.tick(40)
        if mem[ADDR_NUM_BAG_ITEMS] <= 19:
            break
    # leave the mart (LAST_MAP exit mat, south)
    gm = world.maps[mart]
    _walk_fighting(emu, ex, mart,
                   (gm.warps[0].x, gm.warps[0].y))
    for _ in range(12):
        if emu.read_state().map_id != mart:
            break
        emu.press("down")
        _explore_settle(emu)
    return mem[ADDR_NUM_BAG_ITEMS] <= 19 and \
        emu.read_state().map_id != mart


def silph_card_key(emu: PyBoyEmulator, ex: Executor) \
        -> bool:
    """The Card Key (5F, ball at (21, 16)) via the (9, 15)
    pad bounce (probe150i's key7 route): 1F -> elevator
    -> 5F -> stance (9, 14) -> step down onto the pad ->
    9F landing -> step down/up back to (9, 15) on 5F ->
    east along the south room to (20, 16) -> take the
    ball. Bag space is freed FIRST if the bag is at the
    20-stack cap (see _free_bag_space)."""
    world = ex.world
    s1 = world.map_id("SilphCo1F")
    s5 = world.map_id("SilphCo5F")
    s9 = world.map_id("SilphCo9F")
    if _bag_count(emu, ITEM_CARD_KEY) >= 1:
        return True
    if emu.read_state().map_id not in (s1, s5, s9):
        if not _free_bag_space(emu, ex):
            return False
        if not _ride_battling(emu, ex, s1):
            return False
    if emu.read_state().map_id == s1:
        if not _silph_cab_to(emu, ex, s5):
            return False
    if emu.read_state().map_id == s5:
        ok, _ = _walk_fighting(emu, ex, s5, (9, 14))
        st = emu.read_state()
        if not ok or (st.x, st.y) != (9, 14):
            return False
        _silph_step(emu, "down")            # pad -> 9F
        if emu.read_state().map_id != s9:
            return False
        _silph_step(emu, "down")
        _silph_step(emu, "up")
        st = emu.read_state()
        if st.map_id != s5 or (st.x, st.y) != (9, 15):
            return False
        _silph_step(emu, "down")            # (9, 16)
        ok, _ = _walk_fighting(emu, ex, s5, (20, 16))
        st = emu.read_state()
        if (st.x, st.y) != (20, 16):
            # the south room walk can stop a cell short
            # against the Rocket's body; step east by hand
            for _ in range(12):
                if (emu.read_state().x,
                        emu.read_state().y) == (20, 16):
                    break
                _silph_step(emu, "right")
        if (emu.read_state().x, emu.read_state().y) != \
                (20, 16):
            return False
        _face_and_talk(emu, (21, 16))
        emu.wait_free(max_rounds=30, stable_needed=3)
    return _bag_count(emu, ITEM_CARD_KEY) >= 1


def _silph_floor_done(emu: PyBoyEmulator,
                      floor_num: int) -> bool:
    return all(_event_flag(emu, e)
               for e in SILPH_EV[floor_num])


def _silph_sweep_for(emu: PyBoyEmulator, map_id: int,
                     floor_num: int, evs) -> list:
    """One DFS pass over the current component, stopping
    as soon as every event in `evs` is set (probe151lib's
    sweep_for)."""
    if all(_event_flag(emu, e) for e in evs):
        return [True] * len(evs)
    want = lambda: all(_event_flag(emu, e) for e in evs)
    _silph_dfs(emu, map_id, _silph_warps_of(map_id), set(),
               stop_pred=want,
               doors=_silph_doors_of(floor_num))
    return [_event_flag(emu, e) for e in evs]


def _silph_sweep_floor(emu: PyBoyEmulator, ex: Executor,
                       floor_num: int, evs=None,
                       max_rounds: int = 3) -> bool:
    """Sweep the current component of a floor until the
    given trainer events are all set (default: the
    floor's full table), with the probe's resilient
    loop: a grinder faint (raised as _SilphWhiteout by
    the settle when the battle is actually lost, or
    detected from RAM when the battle was won by the
    rest of the party) routes a nurse trip to 9F and
    resumes the same floor."""
    world = ex.world
    fmap = _silph_floor_map(emu, ex, floor_num)
    s9 = world.map_id("SilphCo9F")
    if evs is None:
        evs = SILPH_EV[floor_num]

    def done():
        return all(_event_flag(emu, e) for e in evs)

    for _attempt in range(6):
        try:
            for _ in range(max_rounds):
                if done():
                    return True
                before = [_event_flag(emu, e) for e in evs]
                _silph_sweep_for(emu, fmap, floor_num, evs)
                after = [_event_flag(emu, e) for e in evs]
                if after == before:
                    break
        except _SilphWhiteout:
            return False
        if done():
            return True
        st = emu.read_state()
        if not any(m.hp == 0 for m in st.party):
            break
        # someone fainted mid-sweep: nurse, then resume
        if not _silph_cab_to(emu, ex, s9):
            return False
        if not _silph_nurse_heal(emu, ex):
            return False
        if not _silph_cab_to(emu, ex, fmap):
            return False
    return done()


def _grind_fight(emu: PyBoyEmulator, ex: Executor,
                 species: int, map_name: str, ev: int,
                 cell) -> str:
    """The route grind's per-trainer discipline
    (probe151f-h2): skip if beaten; gate on the grinder
    at >= 55% HP with > 10 damaging PP (else ask for a
    heal); walk to the trainer (walk-ins trigger most
    sights; fall back to adjacent stances + face-and-
    talk); a lost battle raises _SilphWhiteout via the
    money signature."""
    if _event_flag(emu, ev):
        return "done"
    world = ex.world
    idx = _party_index_of(emu, species)
    if idx is None:
        return "heal"
    mon = emu.read_state().party[idx]
    if mon.hp < 0.55 * mon.max_hp or \
            _damaging_pp(emu, idx) <= 10:
        return "heal"
    mid = world.map_id(map_name)
    money0 = emu.read_state().money
    if emu.read_state().map_id != mid:
        _ride_battling(emu, ex, mid)
    if map_name == "Route12":
        st = emu.read_state()
        if st.y < 21 and cell[1] > 21:
            _cross_gate12_south(emu, ex)
        elif st.y >= 21 and cell[1] <= 21:
            _cross_gate12_north(emu, ex)
    _walk_fighting(emu, ex, mid, cell, attempts=8)
    if not _event_flag(emu, ev):
        nx, ny = cell
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            stance = (nx + dx, ny + dy)
            _walk_fighting(emu, ex, mid, stance,
                           attempts=8)
            if _event_flag(emu, ev):
                break
            st = emu.read_state()
            if st.map_id != mid or \
                    abs(st.x - nx) + abs(st.y - ny) != 1:
                continue
            _face_and_talk(emu, cell)
            for _ in range(40):
                st = emu.read_state()
                if st.in_battle:
                    _drive_battle_typed(emu)
                    continue
                if st.in_dialog:
                    emu.close_dialog(max_presses=3)
                    continue
                break
            if _event_flag(emu, ev):
                break
    st = emu.read_state()
    if not _event_flag(emu, ev):
        if st.money < money0:
            raise _SilphWhiteout(
                f"whiteout vs ev{ev} on {map_name}")
        return "heal"
    return "done"


def _grind_leg(emu: PyBoyEmulator, ex: Executor,
               species: int, target_level: int,
               pc_name: str, map_name: str,
               trainers) -> bool:
    """Fight a route's trainers in order until the
    grinder reaches target_level, healing at pc_name
    whenever the gate asks (probe151f-h2's leg)."""
    world = ex.world
    pc = world.map_id(pc_name)
    for ev, cell in trainers:
        idx = _party_index_of(emu, species)
        if idx is not None and \
                emu.read_state().party[idx].level >= \
                target_level:
            return True
        res = _grind_fight(emu, ex, species, map_name,
                           ev, cell)
        if res == "heal":
            if not _grind_heal(emu, ex, species, pc_name):
                return False
            res = _grind_fight(emu, ex, species, map_name,
                               ev, cell)
            if res != "done":
                return False
    idx = _party_index_of(emu, species)
    return idx is not None and \
        emu.read_state().party[idx].level >= target_level


def _grind_heal(emu: PyBoyEmulator, ex: Executor,
                species: int, pc_name: str) -> bool:
    """Heal at a town PC from wherever the grind is,
    crossing the Route 12 gatehouse north by hand when
    south of it (probe151h2's heal_full), then re-lead
    the grinder."""
    world = ex.world
    r12 = world.map_id("Route12")
    st = emu.read_state()
    if st.map_id == r12 and st.y >= 21:
        _cross_gate12_north(emu, ex)
    elif st.map_id in (world.map_id("Route13"),
                       world.map_id("Route14")):
        _ride_battling(emu, ex, r12)
        st = emu.read_state()
        if st.map_id == r12 and st.y >= 21:
            _cross_gate12_north(emu, ex)
    pc = world.map_id(pc_name)
    if not _ride_battling(emu, ex, pc):
        return False
    if not heal_at_pokecenter(emu, ex, pc):
        return False
    return party_lead(emu, species)


def _cross_gate12_south(emu: PyBoyEmulator,
                        ex: Executor) -> bool:
    """Route12Gate1F is a straight x4-5 corridor that
    journey rides strand on: cross it on foot
    (probe151g). Caller is on Route 12 north (y < 21);
    ends on the pier (y >= 21)."""
    world = ex.world
    r12 = world.map_id("Route12")
    gate = world.map_id("Route12Gate1F")
    _walk_fighting(emu, ex, r12, (10, 14), attempts=25)
    emu.press("down")
    _explore_settle(emu)
    if emu.read_state().map_id != gate:
        return False
    for _ in range(14):
        if emu.read_state().map_id == r12:
            break
        st = emu.read_state()
        before = (st.x, st.y)
        emu.press("down")
        _explore_settle(emu)
        st = emu.read_state()
        if (st.x, st.y) == before and st.map_id == gate:
            emu.press("left" if before[0] >= 5 else "right")
            _explore_settle(emu)
    st = emu.read_state()
    return st.map_id == r12 and st.y >= 21


def _cross_gate12_north(emu: PyBoyEmulator,
                        ex: Executor) -> bool:
    """Northbound gate cross (probe151h2): from the pier,
    walk to the gate's south doors and up the corridor.
    Fisher4's beaten body seals the (9, 52) dock cell:
    if the walk starts west of it, detour via the (5, 49)
    shelf first (the block grid lies about these
    half-block dock tiles)."""
    world = ex.world
    r12 = world.map_id("Route12")
    gate = world.map_id("Route12Gate1F")
    st = emu.read_state()
    if st.map_id != r12 or st.y < 21:
        return False
    if st.x < 10:
        _walk_fighting(emu, ex, r12, (5, 49), attempts=25)
    _walk_fighting(emu, ex, r12, (10, 22), attempts=25)
    emu.press("up")
    _explore_settle(emu)
    st = emu.read_state()
    if st.map_id != gate or (st.x, st.y) != (4, 6):
        return False
    for _ in range(12):
        emu.press("up")
        _explore_settle(emu)
        st = emu.read_state()
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
        if st.map_id == r12 and st.y <= 16:
            return True
    st = emu.read_state()
    return st.map_id == r12 and st.y <= 16


def _pier_walk(emu: PyBoyEmulator, ex: Executor,
               waypoints) -> bool:
    """Hop the Route 12 pier waypoint chain (the
    executor's planner stalls on the ~85-cell walk; each
    hop is short enough to inch through)."""
    world = ex.world
    r12 = world.map_id("Route12")
    for wp in waypoints:
        _walk_fighting(emu, ex, r12, wp, attempts=30)
        st = emu.read_state()
        if (st.x, st.y) != wp:
            return False
    return True


def _wake_snorlax(emu: PyBoyEmulator, ex: Executor) \
        -> bool:
    """Route 12's Snorlax blocks the only path south at
    (10, 62). Wake it with the Poke Flute from the bag
    (the probe's menu idiom: START -> ITEM -> USE) and
    beat it - no ball is thrown. Passage is verified by
    walking through its cell to (10, 65) (probe151g)."""
    world = ex.world
    r12 = world.map_id("Route12")
    if emu.read_state().map_id != r12:
        _ride_battling(emu, ex, r12)
    st = emu.read_state()
    if st.y < 21:
        if not _cross_gate12_south(emu, ex):
            return False
    ok, _ = _walk_fighting(emu, ex, r12, (10, 61),
                           attempts=25)
    st = emu.read_state()
    if (st.x, st.y) != (10, 61):
        return False
    emu.press("down")       # face Snorlax (turn only)
    _explore_settle(emu)
    mem = emu.pb.memory
    bag_idx = _bag_index(emu, ITEM_POKE_FLUTE)
    if bag_idx is None:
        return False
    _press10(emu, "start")
    if not _wait_menu_y(emu, 2):
        return False
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM],
                       2):
        return False
    _press10(emu, "a")              # ITEM
    if not _wait_menu_y(emu, 4):
        return False
    if not _steer_menu(
            emu, lambda: mem[ADDR_MENU_SCROLL]
            + mem[ADDR_MENU_ITEM], bag_idx):
        return False
    _press10(emu, "a")
    if not _wait_menu_y(emu, 11):
        return False
    _press10(emu, "a")              # USE
    fought = False
    for _ in range(80):
        st = emu.read_state()
        if st.in_battle:
            _drive_battle_typed(emu)
            fought = True
            continue
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
            continue
        break
    if not fought:
        return False
    _walk_fighting(emu, ex, r12, (10, 65), attempts=25)
    st = emu.read_state()
    return (st.x, st.y) == (10, 65)


def _silph_top_off(emu: PyBoyEmulator, idx: int,
                   threshold: int) -> int:
    """Field-menu Super Potion top-up for party[idx]
    while below `threshold` HP (probe151lib's
    top_off_super; the rival's pre-fight top-off - the
    in-battle potion wrapper from the first Silph pass
    wedged silently and is NOT resurrected)."""
    used = 0
    while True:
        st = emu.read_state()
        mon = st.party[idx]
        if mon.hp == 0 or mon.hp >= min(threshold,
                                        mon.max_hp):
            break
        if _bag_count(emu, ITEM_SUPER_POTION) == 0:
            break
        if not _use_bag_item_on_mon(emu,
                                    ITEM_SUPER_POTION, idx):
            break
        used += 1
        emu.tick(90)
        _close_menus(emu)
        emu.wait_free(max_rounds=8, stable_needed=2)
    return used


def silph_grind(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Vlad's chosen grind (probe151 phases A-C): raise
    BOTH leads lead-and-solo (Gen 1 switch-training pays
    nothing to a fainted participant and splits XP, so
    the grinder leads every fight it takes XP from).
    Blastoise -> ~52 inside Silph (the building's own
    trainers pay the best XP + money; the 9F nurse is a
    free in-building heal), then Arcanine -> ~38
    finishing Silph's 4F/11F-east and walking the
    Lavender-side route pools (R8, R10, R12 with the
    Snorlax wake, R13, R14). Ends healed at Lavender PC
    with Arcanine still in front... caller re-leads.
    NOT done: Route 6 / Route 11 (untapped reserves -
    the westward gate chain stranded repeatedly) and
    R12's Fisher5 (see GRIND_R12S note)."""
    world = ex.world
    s1 = world.map_id("SilphCo1F")
    s9 = world.map_id("SilphCo9F")
    lavpc = world.map_id("LavenderPokecenter")

    def level_of(species):
        idx = _party_index_of(emu, species)
        if idx is None:
            return 0
        return emu.read_state().party[idx].level

    if emu.read_state().map_id not in (
            s1, *[world.map_id(n) for n in
                  SILPH_FLOOR_NAMES.values()]):
        if not _ride_battling(emu, ex, s1):
            return False
    # ---- Blastoise inside Silph ----
    if not party_lead(emu, SILPH_BLASTOISE):
        return False
    for floor_num in (3, 2, 6, 8, 7, 10):
        if level_of(SILPH_BLASTOISE) >= 52:
            break
        fmap = _silph_floor_map(emu, ex, floor_num)
        if emu.read_state().map_id != fmap:
            if not _silph_cab_to(emu, ex, fmap):
                return False
        evs = SILPH_EV[floor_num]
        if floor_num == 7:
            # the west-block trainer (2, 13) is pad-only
            # and the rival's room is off-limits pre-grind;
            # the cab-component three are the grind's 7F
            evs = (1861, 1863, 1864)
        _silph_sweep_floor(emu, ex, floor_num, evs)
    if level_of(SILPH_BLASTOISE) < 52:
        return False
    # ---- Arcanine: Silph leftovers, then routes ----
    if not party_lead(emu, SILPH_ARCANINE):
        return False
    for floor_num, evs in ((4, SILPH_EV[4]),
                           (11, (1925,))):
        if level_of(SILPH_ARCANINE) >= 38:
            break
        fmap = _silph_floor_map(emu, ex, floor_num)
        if emu.read_state().map_id != fmap:
            if not _silph_cab_to(emu, ex, fmap):
                return False
        _silph_sweep_floor(emu, ex, floor_num, evs)
    if level_of(SILPH_ARCANINE) < 38:
        # out of the building, heal, and walk the routes
        if not _silph_cab_to(emu, ex, s1):
            return False
        _walk_fighting(emu, ex, s1, (10, 16))
        for _ in range(6):
            if emu.read_state().map_id == \
                    world.map_id("SaffronCity"):
                break
            emu.press("down")
            _explore_settle(emu)
        if not _grind_heal(emu, ex, SILPH_ARCANINE,
                           "LavenderPokecenter"):
            return False
        legs = [("Route8", GRIND_R8), ("Route10", GRIND_R10),
                ("Route12", GRIND_R12N)]
        for map_name, trainers in legs:
            if level_of(SILPH_ARCANINE) >= 38:
                break
            if not _grind_leg(emu, ex, SILPH_ARCANINE, 38,
                              "LavenderPokecenter", map_name,
                              trainers):
                return False
        if level_of(SILPH_ARCANINE) < 38:
            if not _wake_snorlax(emu, ex):
                return False
            r12 = world.map_id("Route12")
            if not _pier_walk(emu, ex, PIER_WAYPOINTS_S):
                return False
            if not _grind_leg(emu, ex, SILPH_ARCANINE, 38,
                              "LavenderPokecenter", "Route12",
                              GRIND_R12S):
                return False
        if level_of(SILPH_ARCANINE) < 38:
            # R12 south end -> R13 (a SOUTH connection,
            # offset -20: step off the pier's bottom edge)
            r13 = world.map_id("Route13")
            st = emu.read_state()
            if st.map_id == world.map_id("Route12"):
                if (st.x, st.y) != (11, 106):
                    if not _pier_walk(
                            emu, ex,
                            [(11, 106)] if st.y < 100 else
                            [(13, 95), (11, 106)]):
                        return False
                if not _step_until_map(emu, "down", r13):
                    return False
            if not _grind_leg(emu, ex, SILPH_ARCANINE, 38,
                              "LavenderPokecenter", "Route13",
                              GRIND_R13):
                return False
        if level_of(SILPH_ARCANINE) < 38:
            # R13 -> R14 is a WEST connection at row 10
            r14 = world.map_id("Route14")
            r13 = world.map_id("Route13")
            st = emu.read_state()
            if st.map_id == r13:
                _walk_fighting(emu, ex, r13, (1, 10),
                               attempts=30)
                if not _step_until_map(emu, "left", r14):
                    return False
            if not _grind_leg(emu, ex, SILPH_ARCANINE, 38,
                              "LavenderPokecenter", "Route14",
                              GRIND_R14):
                return False
    if not _grind_heal(emu, ex, SILPH_ARCANINE,
                       "LavenderPokecenter"):
        return False
    return level_of(SILPH_BLASTOISE) >= 52 and \
        level_of(SILPH_ARCANINE) >= 38


def silph_rival(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The 7F rival (event 1856), third attempt at the
    ground-up levels (probe151r): nurse to 100% on 9F,
    the far-side circuit (doors (18, 4)/(18, 10)/(10, 12)
    -> closet pad (9, 3) -> 3F island -> pad (11, 11) ->
    7F west room), a FIELD-MENU Super Potion top-off
    only if the lead was chipped on the way in, then the
    trigger cells (3, 2)/(3, 3) with the stock typed
    driver. The first pass lost twice from ~2/3-HP
    starts at Lv48/27; at 52/38 Blastoise soloed the
    team and fainted on the ace, Arcanine closed it
    out. A loss here whiteouts - _silph_settle raises
    and the beat reports False (the mission's stop rule
    is the caller's)."""
    world = ex.world
    s3 = world.map_id("SilphCo3F")
    s7 = world.map_id("SilphCo7F")
    s9 = world.map_id("SilphCo9F")
    if _event_flag(emu, EV_SILPH_RIVAL):
        return True
    if not party_lead(emu, SILPH_BLASTOISE):
        return False
    try:
        if emu.read_state().map_id != s9:
            if not _silph_cab_to(emu, ex, s9):
                return False
        if not _silph_nurse_heal(emu, ex):
            return False
        st = emu.read_state()
        if not all(m.hp == m.max_hp for m in st.party):
            return False
        stance = _silph_dfs(emu, s9, SILPH_W9,
                            {(9, 2), (9, 4), (8, 3), (10, 3)},
                            doors=SILPH_D9)
        if stance is None:
            return False
        _silph_step_onto(emu, 9, 3, stance)
        st = emu.read_state()
        if st.map_id != s3 or (st.x, st.y) != (3, 11):
            return False
        if not _silph_step_off(emu, s3):
            return False
        stance = _silph_dfs(emu, s3, SILPH_W3,
                            {(11, 10), (11, 12), (10, 11),
                             (12, 11)},
                            doors={(8, 8), (16, 8)})
        if stance is None:
            return False
        _silph_step_onto(emu, 11, 11, stance)
        st = emu.read_state()
        if st.map_id != s7 or (st.x, st.y) != (5, 3):
            return False
        if not _silph_step_off(emu, s7):
            return False
        st = emu.read_state()
        lead_mon = st.party[0]
        if lead_mon.hp < lead_mon.max_hp:
            _silph_top_off(emu, 0, lead_mon.max_hp)
        _silph_dfs(emu, s7, SILPH_W7, {(3, 3), (3, 2)},
                   stop_pred=lambda: _event_flag(
                       emu, EV_SILPH_RIVAL))
        if not _event_flag(emu, EV_SILPH_RIVAL):
            _silph_ride_scene(
                emu, lambda: _event_flag(emu,
                                         EV_SILPH_RIVAL),
                rounds=500)
    except _SilphWhiteout:
        return False
    return _event_flag(emu, EV_SILPH_RIVAL)


def silph_lapras(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The Lapras gift from the 7F worker (1, 5)
    (probe151s): the party is full, so it routes to the
    box. GivePokemon stops at the NICKNAME screen -
    not a dialog state, so dialog closers wedge on it;
    START confirms an empty name, then the description
    text runs and BIT_GOT_LAPRAS sets. Verified three
    ways from RAM: the status bit, the dex count +1,
    and Lapras (0x13) in the box. Nobody is deposited
    or released."""
    world = ex.world
    s7 = world.map_id("SilphCo7F")
    mem = emu.pb.memory
    if mem[ADDR_STATUS_FLAGS4] & 0x01:
        return True
    if emu.read_state().map_id != s7:
        return False
    dex0 = _dex_owned_count(emu)
    stance = _silph_dfs(emu, s7, SILPH_W7B,
                        {(1, 4), (1, 6), (2, 5)})
    if stance is None:
        return False
    _face_and_talk(emu, (1, 5))
    _silph_ride_scene(
        emu, lambda: _dex_owned_count(emu) > dex0
        and SPECIES_LAPRAS in _box_species(emu),
        rounds=200)
    for _ in range(12):
        if mem[ADDR_STATUS_FLAGS4] & 0x01:
            break
        emu.press("start")
        _explore_settle(emu)
        emu.close_dialog(max_presses=4)
        _explore_settle(emu)
    emu.wait_free(max_rounds=10, stable_needed=2)
    return bool(mem[ADDR_STATUS_FLAGS4] & 0x01) and \
        SPECIES_LAPRAS in _box_species(emu) and \
        _dex_owned_count(emu) == dex0 + 1


def silph_giovanni(emu: PyBoyEmulator, ex: Executor) \
        -> bool:
    """Giovanni on 11F (event 1935) via the pad chain
    (probe151t): the middle room (Giovanni (6, 9), the
    President (7, 5)) is entered ONLY through the
    (6, 12) card-key door from the SW room - the east
    column / south corridor route dead-ends. From the
    7F west room: pad (5, 7) -> 11F (3, 2) -> west
    column south (Rocket1 (3, 16), ev1924, guards the
    room and is fought on sight) -> door -> Giovanni.
    Blastoise should be nurse-healed first (it may have
    fainted closing out the rival): route via the
    (5, 3) pad to 3F and cab_to(9F) as the probe did."""
    world = ex.world
    s11 = world.map_id("SilphCo11F")
    s7 = world.map_id("SilphCo7F")
    if _event_flag(emu, EV_SILPH_GIOVANNI):
        return True
    if emu.read_state().map_id != s7:
        return False
    try:
        stance = _silph_dfs(emu, s7, SILPH_W7B,
                            {(5, 6), (5, 8), (4, 7), (6, 7)})
        if stance is None:
            return False
        _silph_step_onto(emu, 5, 7, stance)
        st = emu.read_state()
        if st.map_id != s11 or (st.x, st.y) != (3, 2):
            return False
        if not _silph_step_off(emu, s11):
            return False
        _silph_dfs(emu, s11, SILPH_W11,
                   {(6, 10), (6, 11), (5, 9), (7, 9)},
                   stop_pred=lambda: _event_flag(
                       emu, EV_SILPH_GIOVANNI),
                   doors=SILPH_DOOR11)
        if not _event_flag(emu, EV_SILPH_GIOVANNI):
            _silph_ride_scene(
                emu, lambda: _event_flag(
                    emu, EV_SILPH_GIOVANNI),
                rounds=300)
    except _SilphWhiteout:
        return False
    return _event_flag(emu, EV_SILPH_GIOVANNI)


def silph_master_ball(emu: PyBoyEmulator, ex: Executor) \
        -> bool:
    """The President -> the MASTER BALL (probe151t2/t3).
    His row-5 nook is sealed by the desk row, the Beauty
    (10, 5) and his own cell; the only stance is (6, 5),
    entered from (5, 5) - a dummy 'inaccessible' warp
    tile that can be stood on (the executor's planner
    treats it as a terminal, so the approach is walked
    by hand). The ball is the bag's 20th stack - nothing
    else may be picked up first. Exit: the manual step
    chain back through the door and the west column,
    pad chain to 3F, cab to 1F, out the front doors,
    heal at Saffron PC."""
    world = ex.world
    s1 = world.map_id("SilphCo1F")
    s3 = world.map_id("SilphCo3F")
    s7 = world.map_id("SilphCo7F")
    s11 = world.map_id("SilphCo11F")
    saf = world.map_id("SaffronCity")
    safpc = world.map_id("SaffronPokecenter")
    if _bag_count(emu, ITEM_MASTER_BALL) < 1:
        if emu.read_state().map_id != s11:
            return False
        ok, _ = _walk_fighting(emu, ex, s11, (5, 10),
                               attempts=20)
        st = emu.read_state()
        if (st.x, st.y) != (5, 10):
            return False
        for btn, want in (("up", (5, 9)), ("up", (5, 8)),
                          ("up", (5, 7)), ("up", (5, 6)),
                          ("up", (5, 5)),
                          ("right", (6, 5))):
            _silph_step(emu, btn)
            st = emu.read_state()
            if (st.x, st.y) != want:
                return False
        _face_and_talk(emu, (7, 5))
        _silph_ride_scene(
            emu, lambda: _event_flag(emu,
                                     EV_GOT_MASTER_BALL),
            rounds=200)
        if _bag_count(emu, ITEM_MASTER_BALL) < 1 or \
                not _event_flag(emu, EV_GOT_MASTER_BALL):
            return False
        # walk out of the office by hand
        steps = ([("left", (5, 5))]
                 + [("down", (5, y)) for y in range(6, 13)]
                 + [("right", (6, 12)), ("right", (7, 12)),
                    ("down", (7, 13)), ("down", (7, 14)),
                    ("left", (6, 14)), ("left", (5, 14)),
                    ("left", (4, 14)), ("left", (3, 14))]
                 + [("up", (3, y)) for y in range(13, 2, -1)])
        for btn, want in steps:
            _silph_step(emu, btn)
            st = emu.read_state()
            if (st.x, st.y) != want:
                return False
        _silph_step_onto(emu, 3, 2, (3, 3))
        st = emu.read_state()
        if st.map_id != s7 or (st.x, st.y) != (5, 7):
            return False
        if not _silph_step_off(emu, s7):
            return False
    if emu.read_state().map_id == s7:
        stance = _silph_dfs(emu, s7, SILPH_W7B,
                            {(5, 2), (5, 4), (4, 3), (6, 3)})
        if stance is None:
            return False
        _silph_step_onto(emu, 5, 3, stance)
        st = emu.read_state()
        if st.map_id != s3 or (st.x, st.y) != (11, 11):
            return False
        if not _silph_step_off(emu, s3):
            return False
    if emu.read_state().map_id == s3:
        if not _silph_cab_to(emu, ex, s1):
            return False
    if emu.read_state().map_id == s1:
        _walk_fighting(emu, ex, s1, (10, 16))
        for _ in range(6):
            if emu.read_state().map_id == saf:
                break
            emu.press("down")
            _explore_settle(emu)
    if emu.read_state().map_id != saf:
        return False
    if not _ride_battling(emu, ex, safpc):
        return False
    if not heal_at_pokecenter(emu, ex, safpc):
        return False
    st = emu.read_state()
    return _bag_count(emu, ITEM_MASTER_BALL) >= 1 and \
        all(m.hp == m.max_hp for m in st.party)


def silph_co(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The whole Silph Co. leg (probes 150/151): Card
    Key, the two-lead grind, the rival, Lapras, Giovanni
    and the Master Ball, ending healed in Saffron with
    the building liberated. Assumes saffron_gates has
    run (Fresh Water spent, TM29 held). Each beat is
    gated on its RAM witness, so the leg resumes from
    any intermediate state. Faithful port of the probe
    flows; NOT cold-replayed end-to-end - the grind
    alone ran ~2h of wall time across the probe scripts
    with hand-built crossings, and the probes' saved
    intermediates (probe151/*.state) are the proof
    chain."""
    world = ex.world
    s1 = world.map_id("SilphCo1F")
    if _bag_count(emu, ITEM_MASTER_BALL) >= 1:
        return True
    if not silph_card_key(emu, ex):
        return False
    if not silph_grind(emu, ex):
        return False
    # re-enter and heal at the 9F nurse before the rival
    if emu.read_state().map_id != s1:
        if not _ride_battling(emu, ex, s1):
            return False
    if not silph_rival(emu, ex):
        return False
    if not silph_lapras(emu, ex):
        return False
    # nurse run for Giovanni (Blastoise may be down):
    # west room -> (5, 3) pad -> 3F -> cab -> 9F nurse,
    # then back through the closet circuit to 7F west
    s3 = world.map_id("SilphCo3F")
    s7 = world.map_id("SilphCo7F")
    s9 = world.map_id("SilphCo9F")
    stance = _silph_dfs(emu, s7, SILPH_W7B,
                        {(5, 2), (5, 4), (4, 3), (6, 3)})
    if stance is None:
        return False
    _silph_step_onto(emu, 5, 3, stance)
    st = emu.read_state()
    if st.map_id != s3 or (st.x, st.y) != (11, 11):
        return False
    if not _silph_step_off(emu, s3):
        return False
    if not _silph_cab_to(emu, ex, s9):
        return False
    if not _silph_nurse_heal(emu, ex):
        return False
    # 9F -> closet circuit back to the 7F west room
    stance = _silph_dfs(emu, s9, SILPH_W9,
                        {(9, 2), (9, 4), (8, 3), (10, 3)},
                        doors=SILPH_D9)
    if stance is None:
        return False
    _silph_step_onto(emu, 9, 3, stance)
    st = emu.read_state()
    if st.map_id != s3 or (st.x, st.y) != (3, 11):
        return False
    if not _silph_step_off(emu, s3):
        return False
    stance = _silph_dfs(emu, s3, SILPH_W3,
                        {(11, 10), (11, 12), (10, 11),
                         (12, 11)},
                        doors={(8, 8), (16, 8)})
    if stance is None:
        return False
    _silph_step_onto(emu, 11, 11, stance)
    st = emu.read_state()
    if st.map_id != s7 or (st.x, st.y) != (5, 3):
        return False
    if not _silph_step_off(emu, s7):
        return False
    if not silph_giovanni(emu, ex):
        return False
    return silph_master_ball(emu, ex)

MAP_MT_MOON_B2F = 61


# ------------------------------------------------------------------
# Saffron Gym / Sabrina (probe152): the PC item-storage
# deposit, the wandering-NPC door shuffle, and the warp-
# pad gym. Three things the earlier legs never needed:
#
# * deposit_item: Gen 1's bag holds 20 item STACKS and a
#   full bag refuses Sabrina's TM46 (her post-battle
#   script branches to a "no room" text and withholds
#   it). The probe freed exactly one stack beforehand -
#   the LIFT KEY into the PC item storage (wNumBoxItems
#   $D53A / wBoxItems $D53B), verified in RAM both ways.
#   The storage menu shares its signature with the PC
#   main menu; it is told apart by the cursor resetting
#   to 0 after the "Accessed Item Storage System." text.
#
# * Saffron City defeats the stock rides: the world
#   model's journey() returns None for the Pokecenter
#   <-> Gym pair and _ride_battling spins. The probe
#   crossed the city on the tile grid with RAM-verified
#   steps (_city_grid_walk), and the Pokecenter door
#   stance (9, 30) has a wandering NPC who parks on it -
#   entry is a shuffle: approach via (10, 31) and walk a
#   small loop until she moves (_enter_saffron_pc).
#
# * The gym is one map, nine rooms, 30 warp pads forming
#   15 reciprocal pairs (SAFFRON_GYM_PADS, compiled from
#   pokered's SaffronGym warp events and cross-checked
#   against the probe's empirical graph: 243 cells, 946
#   edges learned by stepping). Navigation plans over
#   the tile grid with pads as teleport edges
#   (_gym_navigate) and verifies every landing; the
#   settle is deliberately paranoid (_gym_settle) -
#   a sight trainer's walk-up runs under wJoyIgnore
#   with no battle and no dialog yet, and a naive
#   settle records the step as a wall and the battle
#   as never happening (probe152's Channeler1 lesson:
#   her sight cell (10, 3) read as a wall for a whole
#   exploration pass while the screen sat mid-fade).
# ------------------------------------------------------------------

MAP_SAFFRON_GYM = 178
EV_GOT_TM46 = 864
EV_BEAT_SABRINA = 865
EV_SAFFRON_GYM_TRAINERS = (866, 867, 868, 869, 870, 871,
                           872)
ITEM_TM46 = 0xF6     # TM ids are 0xC9 + (n - 1); TM46 Psywave
ADDR_NUM_PC_ITEMS = 0xD53A          # wNumBoxItems
ADDR_PC_ITEMS = 0xD53B              # wBoxItems
ADDR_JOY_IGNORE = 0xD3B0            # wJoyIgnore

# (trainer cell, talk stance, event) in header order:
# Channeler1 (Gastly 34 / Haunter 34), Psychic1 (Kadabra
# / Slowpoke / Mr. Mime / Kadabra 31), Channeler2
# (Haunter 38), Psychic2 (Mr. Mime / Kadabra 34),
# Channeler3 (Gastly x2 / Haunter 33), Psychic3
# (Slowpoke x2 / Slowbro 33), Psychic4 (Slowbro 38).
SAFFRON_GYM_TRAINERS = (
    ((10, 1), (10, 2), 866),
    ((17, 1), (17, 2), 867),
    ((3, 7), (3, 8), 868),
    ((17, 7), (17, 8), 869),
    ((3, 13), (3, 14), 870),
    ((17, 13), (17, 14), 871),
    ((3, 1), (3, 2), 872),
)
SAFFRON_GYM_NPCS = frozenset(
    [cell for cell, _, _ in SAFFRON_GYM_TRAINERS]
    + [(9, 8), (10, 15)])     # Sabrina, the gym guide
SABRINA_CELL = (9, 8)
SABRINA_STANCE = (9, 9)

# Warp pads: stepping ONTO the key cell lands the player
# ON the value cell (pokered SaffronGym_Object warp
# events 3-32; the two door warps 1-2 are the exit).
SAFFRON_GYM_PADS = {
    (1, 3): (15, 5), (5, 3): (11, 3), (1, 5): (11, 11),
    (5, 5): (1, 11), (1, 9): (19, 11), (5, 9): (9, 5),
    (1, 11): (5, 5), (5, 11): (5, 17), (1, 15): (19, 5),
    (5, 15): (15, 17), (1, 17): (11, 5), (5, 17): (5, 11),
    (9, 3): (15, 11), (11, 3): (5, 3), (9, 5): (5, 9),
    (11, 5): (1, 17), (11, 11): (1, 5), (11, 15): (19, 17),
    (15, 3): (15, 9), (19, 3): (15, 15), (15, 5): (1, 3),
    (19, 5): (1, 15), (15, 9): (15, 3), (19, 9): (19, 15),
    (15, 11): (9, 3), (19, 11): (1, 9), (15, 15): (19, 3),
    (19, 15): (19, 9), (15, 17): (5, 15), (19, 17): (11, 15),
}


def _pc_items(emu: PyBoyEmulator) -> list:
    """The PC item storage as (item_id, count) pairs."""
    mem = emu.pb.memory
    n = mem[ADDR_NUM_PC_ITEMS]
    return [(mem[ADDR_PC_ITEMS + 2 * i],
             mem[ADDR_PC_ITEMS + 2 * i + 1]) for i in range(n)]


def _pc_count(emu: PyBoyEmulator, item_id: int) -> int:
    for item, count in _pc_items(emu):
        if item == item_id:
            return count
    return 0


def deposit_item(emu: PyBoyEmulator, ex: Executor,
                 pc_map: int, item_id: int) -> bool:
    """Deposit one stack of `item_id` into the PC item
    storage (probe152b). Caller must be INSIDE the
    Pokecenter. The PC main menu and the item-storage
    menu share a signature (topY 2, max item 3); the
    storage menu is recognised by its cursor resetting
    to 0 once the access text clears - and A presses
    stop the moment it does, or the next one selects
    WITHDRAW. Every effect is verified in RAM: the item
    leaves the bag and appears in storage."""
    if _bag_count(emu, item_id) < 1:
        return False
    if _pc_count(emu, item_id) >= 1:
        return True
    if not _open_pc(emu, ex, pc_map):
        return False
    mem = emu.pb.memory

    def ride_until(pred, rounds=160):
        for _ in range(rounds):
            if pred():
                return True
            if emu.read_state().in_dialog:
                _press10(emu, "a")
            emu.tick(25)
        return pred()

    # probe154: the MAX==3/topY!=0 signature is stale-byte
    # fragile in save lineages that have already used a PC -
    # the bytes linger from the last session, the check
    # passes while the screen still shows "turned on the
    # PC.", and every later steer misses. The live menu is
    # MAX==3 AND topY==2 AND the text box down (in_dialog
    # False); the probe verified four deposits in a row
    # with this predicate after the loose one failed.
    if not ride_until(lambda: mem[ADDR_MAX_MENU_ITEM] == 3
                      and mem[ADDR_MENU_TOP_Y] == 2
                      and not emu.read_state().in_dialog):
        return False
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 1):
        return False
    _press10(emu, "a")                         # player's PC
    for _ in range(160):
        if mem[ADDR_MAX_MENU_ITEM] == 3 and \
                mem[ADDR_MENU_ITEM] == 0:
            break
        _press10(emu, "a")
        emu.tick(25)
    else:
        return False
    emu.tick(30)
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 1):
        return False
    _press10(emu, "a")                         # DEPOSIT ITEM
    emu.tick(60)
    idx = _bag_index(emu, item_id)
    if idx is None:
        return False
    if not _steer_menu(
            emu, lambda: mem[ADDR_MENU_SCROLL]
            + mem[ADDR_MENU_ITEM], idx):
        return False
    _press10(emu, "a")
    emu.tick(90)
    ok = _pc_press_a_ride(
        emu, lambda: _bag_count(emu, item_id) == 0,
        rounds=200)
    for _ in range(12):
        if mem[ADDR_MENU_TOP_Y] == 0 and \
                not emu.read_state().in_dialog:
            break
        _press10(emu, "b")
        emu.tick(30)
    return ok and _pc_count(emu, item_id) >= 1


def _gym_settle(emu: PyBoyEmulator) -> None:
    """Ride out whatever a gym step started. A sight
    trainer's walk-up runs under wJoyIgnore with no
    battle and no dialog yet, so 'quiet' requires three
    consecutive checks with no battle, no dialog and
    joy-ignore clear. Battles go to the typed driver;
    dialog is A-ridden (trainer text, not menus)."""
    mem = emu.pb.memory
    quiet = 0
    for _ in range(300):
        st = emu.read_state()
        if st.in_battle:
            _drive_battle_typed(emu)
            _maintain_party(emu, ())
            quiet = 0
            continue
        if st.in_dialog:
            _press10(emu, "a")
            emu.tick(20)
            quiet = 0
            continue
        if mem[ADDR_JOY_IGNORE] != 0:
            emu.tick(20)
            quiet = 0
            continue
        quiet += 1
        if quiet >= 3:
            return
        emu.tick(15)


def _gym_step(emu: PyBoyEmulator, btn: str):
    """One press in the gym + the paranoid settle;
    returns the landing cell (or the new map id's
    position if a door warp fired)."""
    emu.press(btn)
    _explore_settle(emu)
    _gym_settle(emu)
    st = emu.read_state()
    return (st.x, st.y)


def _gym_navigate(emu: PyBoyEmulator, ex: Executor,
                  target) -> bool:
    """BFS over the gym's tile grid with pads as
    teleport edges; every press is verified against
    the planned landing and divergence replans from
    the actual cell (probe152's empirical graph agreed
    with this model everywhere except NPC cells and
    the door mats, which are handled by the caller)."""
    gym = ex.world.maps[MAP_SAFFRON_GYM]
    dirs = {"up": (0, -1), "down": (0, 1), "left": (-1, 0),
            "right": (1, 0)}

    def landing(cell, btn):
        dx, dy = dirs[btn]
        nb = (cell[0] + dx, cell[1] + dy)
        if nb in SAFFRON_GYM_NPCS:
            return None
        if not gym.walkable(nb[0], nb[1]):
            return None
        return SAFFRON_GYM_PADS.get(nb, nb)

    for _ in range(60):
        st = emu.read_state()
        if st.map_id != MAP_SAFFRON_GYM:
            return False
        cur = (st.x, st.y)
        if cur == target:
            return True
        from collections import deque
        prev = {cur: None}
        dq = deque([cur])
        while dq:
            cell = dq.popleft()
            for btn in dirs:
                land = landing(cell, btn)
                if land is not None and land not in prev:
                    prev[land] = (cell, btn)
                    dq.append(land)
        if target not in prev:
            return False
        path = []
        cell = target
        while prev[cell] is not None:
            cell, btn = prev[cell]
            path.append(btn)
        for btn in reversed(path):
            want_from = cur
            want = landing(want_from, btn)
            got = _gym_step(emu, btn)
            if emu.read_state().map_id != MAP_SAFFRON_GYM:
                return False
            if got != want:
                break          # diverged: replan
            cur = got
        else:
            continue
    st = emu.read_state()
    return st.map_id == MAP_SAFFRON_GYM and \
        (st.x, st.y) == target


def _gym_talk_fight(emu: PyBoyEmulator, ex: Executor,
                    stance, event_id: int) -> bool:
    """Navigate to a trainer's talk stance, face them
    (they all stand one cell north of the stance), talk,
    and ride the scene - sight pre-emption included -
    until the trainer's event flag reads set."""
    if _event_flag(emu, event_id):
        return True
    if not _gym_navigate(emu, ex, stance):
        return False
    emu.press("up")
    emu.tick(20)
    _press10(emu, "a")
    mem = emu.pb.memory
    for _ in range(600):
        if _event_flag(emu, event_id):
            return True
        st = emu.read_state()
        if st.map_id != MAP_SAFFRON_GYM and not st.in_battle \
                and not st.in_dialog:
            return False     # blackout respawn: caller heals
        if st.in_battle:
            _drive_battle_typed(emu)
            _maintain_party(emu, ())
            if all(m.hp == 0
                   for m in emu.read_state().party):
                return False
            continue
        if st.in_dialog:
            _press10(emu, "a")
            emu.tick(20)
            continue
        if mem[ADDR_JOY_IGNORE] != 0:
            emu.tick(20)
            continue
        emu.tick(15)
    return _event_flag(emu, event_id)


def _enter_saffron_pc(emu: PyBoyEmulator,
                      ex: Executor) -> bool:
    """Cross Saffron City to the Pokecenter and get
    inside (probe152d2). The door stance (9, 30) has a
    wandering NPC who parks on it; the approach comes
    via (10, 31) and shuffle-walks a small loop until
    she moves off it."""
    world = ex.world
    city = world.map_id("SaffronCity")
    pc = world.map_id("SaffronPokecenter")
    if emu.read_state().map_id == pc:
        return True
    if emu.read_state().map_id != city:
        return False
    if not _saffron_grid_walk(emu, ex, city, (10, 31)):
        return False

    def step(btn):
        emu.press(btn)
        _explore_settle(emu)
        _gym_settle(emu)
        st = emu.read_state()
        return (st.x, st.y)

    for _ in range(14):
        if emu.read_state().map_id == pc:
            return True
        st = emu.read_state()
        if (st.x, st.y) == (10, 31):
            step("left")
        if _pos(emu) == (9, 31):
            step("up")
        if _pos(emu) == (9, 30):
            step("up")
        if emu.read_state().map_id == pc:
            return True
        for btn in ("right", "down", "left", "up"):
            step(btn)
        if _pos(emu) != (10, 31):
            _saffron_grid_walk(emu, ex, city, (10, 31))
    return emu.read_state().map_id == pc


def _pos(emu: PyBoyEmulator):
    st = emu.read_state()
    return (st.x, st.y)


def _saffron_grid_walk(emu: PyBoyEmulator, ex: Executor,
                       map_id: int, target) -> bool:
    """_city_grid_walk's real body: tile-path following
    with per-press RAM verification (see the section
    header for why Saffron needs it)."""
    world = ex.world
    blocked = set()
    fails = {}
    for _ in range(80):
        st = emu.read_state()
        if st.map_id != map_id:
            return False
        cur = (st.x, st.y)
        if cur == target:
            return True
        path = world.tile_path(map_id, cur, target,
                               blocked=frozenset(blocked))
        if not path:
            return False
        dirs = {(0, -1): "up", (0, 1): "down",
                (-1, 0): "left", (1, 0): "right"}
        for cell in path[1:]:
            dx, dy = cell[0] - cur[0], cell[1] - cur[1]
            if (dx, dy) not in dirs:
                break      # warp corridor step: replan
            btn = dirs[(dx, dy)]
            ok = False
            for _try in range(6):
                emu.press(btn)
                _explore_settle(emu)
                _gym_settle(emu)
                st = emu.read_state()
                if st.map_id != map_id:
                    return False
                if (st.x, st.y) == cell:
                    ok = True
                    break
                emu.tick(90)
            if not ok:
                fails[cell] = fails.get(cell, 0) + 1
                if fails[cell] >= 2:
                    blocked.add(cell)
                break
            cur = cell
    st = emu.read_state()
    return st.map_id == map_id and (st.x, st.y) == target


def _enter_saffron_gym(emu: PyBoyEmulator,
                       ex: Executor) -> bool:
    """City -> gym door (34, 3), entered from the
    (34, 4) stance by stepping north."""
    world = ex.world
    city = world.map_id("SaffronCity")
    if emu.read_state().map_id == MAP_SAFFRON_GYM:
        return True
    if emu.read_state().map_id != city:
        return False
    if not _saffron_grid_walk(emu, ex, city, (34, 4)):
        return False
    for _ in range(8):
        if emu.read_state().map_id == MAP_SAFFRON_GYM:
            return True
        emu.press("up")
        _explore_settle(emu)
    return emu.read_state().map_id == MAP_SAFFRON_GYM


def _exit_saffron_gym(emu: PyBoyEmulator,
                      ex: Executor) -> bool:
    """Gym -> city: the door mats (8, 17)/(9, 17) exit
    by stepping SOUTH off them; the learned-graph
    route runs to (11, 17) and walks the last cells by
    hand (probe152d)."""
    world = ex.world
    city = world.map_id("SaffronCity")
    if emu.read_state().map_id == city:
        return True
    if not _gym_navigate(emu, ex, (11, 17)):
        return False
    for btn in ("left", "left", "down", "down"):
        if emu.read_state().map_id != MAP_SAFFRON_GYM:
            break
        emu.press(btn)
        _explore_settle(emu)
        _gym_settle(emu)
    return emu.read_state().map_id == city


def _saffron_heal(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Enter the Saffron Pokecenter and heal; the heal
    helper leaves by the exit mat, back in the city."""
    world = ex.world
    pc = world.map_id("SaffronPokecenter")
    if not _enter_saffron_pc(emu, ex):
        return False
    return heal_at_pokecenter(emu, ex, pc)


def beat_sabrina(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Saffron Gym and the Marsh Badge (probe152).

    Assumes the player is in Saffron City with the
    Silph leg done. Order of operations, as probed:
    free a bag stack first if the bag is full (deposit
    the Lift Key in the PC item storage - Sabrina's
    TM46 is refused by a full bag), sweep the seven
    gym trainers over the warp pads, heal, then
    Sabrina herself (Kadabra 38 / Mr. Mime 37 /
    Venomoth 38 / Alakazam 43, custom moves). Verified
    by the Marsh badge bit, EVENT_BEAT_SABRINA and
    TM46 in the bag; ends healed in Saffron City.

    FIDELITY: COLD-REPLAYED end-to-end from
    probe151/silph.state (third attempt; replay end
    state probe152/replay-sabrina.state). The first
    two replay attempts failed informatively - a
    mid-sweep blackout the recovery-less sweep could
    not absorb, then a Sabrina blackout stalling the
    scene ride in the wrong map - and the recovery
    paths above are those failures, fixed. The
    replay's path differs from the probe's (money
    halved by blackouts, Oddish took XP and a move
    swap); the chain head stays the probe's state."""
    world = ex.world
    city = world.map_id("SaffronCity")
    pc = world.map_id("SaffronPokecenter")
    if "Marsh" in emu.read_state().badges:
        return _bag_count(emu, ITEM_TM46) >= 1
    if emu.read_state().map_id != city:
        return False
    # 1. bag space for TM46
    if emu.pb.memory[ADDR_NUM_BAG_ITEMS] >= 20 and \
            _bag_count(emu, ITEM_LIFT_KEY) >= 1:
        if not _enter_saffron_pc(emu, ex):
            return False
        if not deposit_item(emu, ex, pc, ITEM_LIFT_KEY):
            return False
        # leave the center by its exit mat
        try:
            ex._walk_to(pc, (4, 7))
        except StuckError:
            return False
        for _ in range(8):
            if emu.read_state().map_id != pc:
                break
            emu.press("down")
            _explore_settle(emu)
        if emu.read_state().map_id != city:
            return False
    # 2. the gym sweep - with the probe's own lesson
    # built in: the sweep can faint the lead (it did in
    # probe152's first pass), so a faint or a blackout
    # mid-sweep exits, heals and re-enters; trainer
    # flags persist, so the sweep resumes where it
    # stopped. A trainer who still will not fall after
    # one such recovery fails the leg.
    if not _enter_saffron_gym(emu, ex):
        return False
    for _cell, stance, event_id in SAFFRON_GYM_TRAINERS:
        if _event_flag(emu, event_id):
            continue
        ok = _gym_talk_fight(emu, ex, stance, event_id)
        if not ok:
            # blackout respawn (city) or a fainted lead
            # inside: heal, re-enter, one more attempt
            if emu.read_state().map_id == MAP_SAFFRON_GYM:
                if not _exit_saffron_gym(emu, ex):
                    return False
            if not _saffron_heal(emu, ex):
                return False
            if not _enter_saffron_gym(emu, ex):
                return False
            ok = _gym_talk_fight(emu, ex, stance, event_id)
        if not ok:
            return False
    # 3. heal before the leader if the sweep cost the
    # lead (the probe's sweep fainted Blastoise once
    # and left it at 31/178 the second time)
    st = emu.read_state()
    lead = st.party[0]
    if lead.hp == 0 or lead.hp * 2 < lead.max_hp:
        if not _exit_saffron_gym(emu, ex):
            return False
        if not _saffron_heal(emu, ex):
            return False
        if not _enter_saffron_gym(emu, ex):
            return False
    # 4. Sabrina - with the standing blackout
    # contingency: respawn, heal, re-enter, re-attempt
    # ONCE; a second blackout fails the leg (the probe
    # never needed it; replay 2's sweep-recovery build
    # cleared all seven trainers and still came back
    # False here, money halved - a Sabrina blackout
    # with the ride stalling in the wrong map)
    if not _gym_talk_fight(emu, ex, SABRINA_STANCE,
                           EV_BEAT_SABRINA):
        if emu.read_state().map_id == MAP_SAFFRON_GYM:
            if not _exit_saffron_gym(emu, ex):
                return False
        if not _saffron_heal(emu, ex):
            return False
        if not _enter_saffron_gym(emu, ex):
            return False
        if not _gym_talk_fight(emu, ex, SABRINA_STANCE,
                               EV_BEAT_SABRINA):
            return False
    # the post-battle script awards the badge and TM46
    # through dialog; ride it out
    for _ in range(300):
        if _event_flag(emu, EV_GOT_TM46) and \
                _bag_count(emu, ITEM_TM46) >= 1:
            break
        st = emu.read_state()
        if st.in_dialog:
            _press10(emu, "a")
        emu.tick(20)
    if "Marsh" not in emu.read_state().badges or \
            _bag_count(emu, ITEM_TM46) < 1:
        return False
    # 5. out and healed
    if not _exit_saffron_gym(emu, ex):
        return False
    if not _saffron_heal(emu, ex):
        return False
    st = emu.read_state()
    return "Marsh" in st.badges and \
        _bag_count(emu, ITEM_TM46) >= 1 and \
        all(m.hp == m.max_hp for m in st.party)



# ------------------------------------------------------------------
# Fighting Dojo (probe153): the Saffron mini-leg. No badge
# here - the prize is a one-of-two gift, and the choice is
# a permanent save-line decision, so it is a PARAMETER
# (this line's value: "hitmonlee", Vlad's pick 2026-11-08;
# Hitmonchan's ball is never touched).
#
# The dojo (map 177, 10x12) is one open mat: four Black
# Belts stand in two columns and the Karate King at the
# top. Three script facts shape the port:
#
# * The King's post-battle script runs
#   SetEventRange(EVENT_BEAT_KARATE_MASTER ..
#   EVENT_BEAT_FIGHTING_DOJO_TRAINER_3) - beating him
#   first would set all four belt flags unread. The
#   belts are therefore fought FIRST and each flag
#   verified individually (events 850-853), the King
#   last (849).
#
# * The King is not a trainer-header entry: the map's
#   default script engages him when the player stands on
#   (4, 3), the cell left of his (5, 3). The port steps
#   onto the trigger cell and rides the scene, with a
#   talk from the same cell as fallback.
#
# * A gift ball's text runs DisplayPokedex BEFORE its
#   Yes/No choice, and DisplayPokedex leaves the species
#   in wCurPartySpecies ($CF91, pokered.sym). The port
#   reads that byte after the talk and proceeds ONLY if
#   it is the chosen species - the ball is identified
#   from RAM before any commitment, not from its cell.
#   The dex page itself waits on
#   WaitForTextScrollButtonPress (A/B edges): the first
#   probe's ride (START presses + close_dialog, which
#   only fires while in_dialog) never dismissed it; raw
#   A presses do. After YES, GivePokemon's nickname
#   screen is cleared with START (default name), the
#   Lapras pattern (probe151s).
#
# Walking in is a single column: entering from the city
# lands on the (4, 11) door mat, and the straight walk
# north up x=4 crosses every belt's sight line in turn
# (BB4's row 7, BB2's row 6, BB3's row 5, BB1's row 4) -
# the probe's "BB1" approach fought all four belts en
# route, each flag flipping on its own row. The port
# keeps the per-trainer loop anyway: it verifies flags
# one by one and can resume mid-room after a blackout.
# Blastoise took ZERO damage across all five fights
# (OHKO sweep; it levelled 55 -> 56 on the belt XP).
# ------------------------------------------------------------------

MAP_FIGHTING_DOJO = 177
EV_DEFEATED_FIGHTING_DOJO = 848
EV_BEAT_KARATE_MASTER = 849
EV_DOJO_TRAINERS = (850, 851, 852, 853)
EV_GOT_HITMONLEE = 854
EV_GOT_HITMONCHAN = 855
SPECIES_HITMONLEE = 0x2B
SPECIES_HITMONCHAN = 0x2C
ADDR_CUR_PARTY_SPECIES = 0xCF91     # wCurPartySpecies
DOJO_MASTER_CELL = (5, 3)
DOJO_MASTER_TRIGGER = (4, 3)
DOJO_MATS = frozenset({(4, 11), (5, 11)})
# (cell, event, sight-line cells, talk stances) in
# object order: BB1 (3,4) RIGHT sight 4 - Mankey /
# Mankey / Primeape 31; BB2 (3,6) RIGHT sight 4 -
# Machop / Machoke 32; BB3 (5,5) LEFT sight 3 -
# Primeape 36; BB4 (5,7) LEFT sight 3 - Machop /
# Mankey / Primeape 31.
DOJO_BELTS = (
    ((3, 4), 850, ((4, 4), (5, 4), (6, 4), (7, 4)),
     ((2, 4), (4, 4), (3, 5))),
    ((3, 6), 851, ((4, 6), (5, 6), (6, 6), (7, 6)),
     ((2, 6), (4, 6), (3, 5), (3, 7))),
    ((5, 5), 852, ((4, 5), (3, 5), (2, 5)),
     ((4, 5), (6, 5), (5, 4), (5, 6))),
    ((5, 7), 853, ((4, 7), (3, 7), (2, 7)),
     ((4, 7), (6, 7), (5, 6), (5, 8))),
)
# choice -> (ball cell, talk stance, species, gift event)
DOJO_GIFTS = {
    "hitmonlee": ((4, 1), (4, 2), SPECIES_HITMONLEE,
                  EV_GOT_HITMONLEE),
    "hitmonchan": ((5, 1), (5, 2), SPECIES_HITMONCHAN,
                   EV_GOT_HITMONCHAN),
}
DOJO_OBJECT_CELLS = frozenset(
    [DOJO_MASTER_CELL] + [c for c, _, _, _ in DOJO_BELTS]
    + [ball for ball, _, _, _ in DOJO_GIFTS.values()])


def _dojo_ride_scene(emu: PyBoyEmulator,
                     rounds: int = 400) -> bool:
    """Resolve any pending dojo scene (trainer walk-up,
    dialog, battle) until the overworld is quiet.
    Returns True if a battle was fought."""
    fought = False
    for _ in range(rounds):
        st = emu.read_state()
        if st.in_battle:
            _drive_battle_typed(emu)
            _maintain_party(emu, ())
            fought = True
            continue
        if st.in_dialog:
            _press10(emu, "a")
            emu.tick(20)
            continue
        if emu.pb.memory[ADDR_JOY_IGNORE] != 0:
            emu.tick(20)
            continue
        emu.tick(30)
        st = emu.read_state()
        if not st.in_battle and not st.in_dialog and \
                emu.pb.memory[ADDR_JOY_IGNORE] == 0:
            return fought
    return fought


def _dojo_walk(emu: PyBoyEmulator, ex: Executor, target,
               blocked_extra=frozenset()) -> bool:
    """Tile-path walk inside the dojo with per-press RAM
    verification; sight-triggered battles on the way are
    driven as they come (the x=4 column crosses every
    belt's sight line). Object cells and the exit mats
    are permanent obstacles."""
    world = ex.world
    blocked = set(DOJO_OBJECT_CELLS) | set(DOJO_MATS) \
        | set(blocked_extra)
    blocked.discard(target)
    dirs = {(0, -1): "up", (0, 1): "down",
            (-1, 0): "left", (1, 0): "right"}
    for _ in range(60):
        st = emu.read_state()
        if st.map_id != MAP_FIGHTING_DOJO:
            return False
        cur = (st.x, st.y)
        if cur == target:
            return True
        path = world.tile_path(MAP_FIGHTING_DOJO, cur,
                               target,
                               blocked=frozenset(blocked))
        if not path:
            return False
        for cell in path[1:]:
            dx, dy = cell[0] - cur[0], cell[1] - cur[1]
            if (dx, dy) not in dirs:
                break
            ok = False
            for _try in range(6):
                emu.press(dirs[(dx, dy)])
                _explore_settle(emu)
                st = emu.read_state()
                if st.in_battle or st.in_dialog:
                    _dojo_ride_scene(emu)
                if (emu.read_state().x,
                        emu.read_state().y) == cell:
                    ok = True
                    break
                emu.tick(60)
            if not ok:
                break
            cur = cell
            _dojo_ride_scene(emu, rounds=6)
    st = emu.read_state()
    return st.map_id == MAP_FIGHTING_DOJO and \
        (st.x, st.y) == target


def _enter_fighting_dojo(emu: PyBoyEmulator,
                         ex: Executor) -> bool:
    """City -> dojo door (26, 3), entered from the
    (26, 4) stance by stepping north (the gym door's
    pattern, probe153b)."""
    world = ex.world
    city = world.map_id("SaffronCity")
    if emu.read_state().map_id == MAP_FIGHTING_DOJO:
        return True
    if emu.read_state().map_id != city:
        return False
    if not _saffron_grid_walk(emu, ex, city, (26, 4)):
        return False
    for _ in range(8):
        if emu.read_state().map_id == MAP_FIGHTING_DOJO:
            return True
        emu.press("up")
        _explore_settle(emu)
    return emu.read_state().map_id == MAP_FIGHTING_DOJO


def _exit_fighting_dojo(emu: PyBoyEmulator,
                        ex: Executor) -> bool:
    """Dojo -> city: walk to (4, 10) and step south
    over the door mat."""
    world = ex.world
    city = world.map_id("SaffronCity")
    if emu.read_state().map_id == city:
        return True
    if not _dojo_walk(emu, ex, (4, 10)):
        return False
    for _ in range(10):
        if emu.read_state().map_id != MAP_FIGHTING_DOJO:
            break
        emu.press("down")
        _explore_settle(emu)
    return emu.read_state().map_id == city


def _dojo_fight(emu: PyBoyEmulator, ex: Executor, cell,
                event_id: int, sight_cells,
                stances) -> bool:
    """One dojo trainer: step into the sight line (they
    engage and walk up), else take a talk stance and
    talk; verified by the trainer's own event flag."""
    if _event_flag(emu, event_id):
        return True
    for t in sight_cells:
        if _dojo_walk(emu, ex, t):
            _dojo_ride_scene(emu)
            if _event_flag(emu, event_id):
                return True
    for s in stances:
        if not _dojo_walk(emu, ex, s):
            continue
        _dojo_ride_scene(emu)
        if _event_flag(emu, event_id):
            return True
        _face_and_talk(emu, cell)
        _dojo_ride_scene(emu)
        if _event_flag(emu, event_id):
            return True
    return _event_flag(emu, event_id)


def take_hitmon(emu: PyBoyEmulator, ex: Executor,
                choice: str = "hitmonlee") -> bool:
    """Take ONE gift ball in the dojo - this save line's
    the HITMONLEE ball at (4, 1) (probe153e). The
    species gate is the point: the talk's DisplayPokedex
    leaves the ball's species in wCurPartySpecies, and
    the function returns False WITHOUT confirming the
    Yes/No if the byte is not the chosen species. After
    YES, GivePokemon routes to the box (party full) and
    the nickname screen is cleared with START (default
    name). Verified three ways from RAM: the gift event
    flag, the owned-dex count +1, and the species
    physically in the box - with the party list and the
    OTHER gift's flag checked unchanged/clear."""
    ball, stance, species, event_id = DOJO_GIFTS[choice]
    other = "hitmonchan" if choice == "hitmonlee" \
        else "hitmonlee"
    other_event = DOJO_GIFTS[other][3]
    if _event_flag(emu, event_id):
        return True
    if _event_flag(emu, other_event):
        return False     # the other gift was taken: over
    if emu.read_state().map_id != MAP_FIGHTING_DOJO:
        return False
    dex0 = _dex_owned_count(emu)
    party0 = [(m.species_id, m.level)
              for m in emu.read_state().party]
    if not _dojo_walk(emu, ex, stance):
        return False
    _face_and_talk(emu, ball)
    emu.tick(90)
    if emu.pb.memory[ADDR_CUR_PARTY_SPECIES] != species:
        return False     # NOT the chosen ball: back away
    # raw A: dismisses the dex page
    # (WaitForTextScrollButtonPress), advances the text
    # and confirms YES (cursor default 0); interleaved
    # START clears the nickname screen.
    for i in range(160):
        if _event_flag(emu, event_id):
            break
        _raw(emu, "a", hold=5, settle=40)
        if i % 4 == 3:
            _raw(emu, "start", hold=5, settle=40)
    emu.wait_free(max_rounds=10, stable_needed=3)
    party1 = [(m.species_id, m.level)
              for m in emu.read_state().party]
    return _event_flag(emu, event_id) and \
        not _event_flag(emu, other_event) and \
        _dex_owned_count(emu) == dex0 + 1 and \
        species in _box_species(emu) and \
        party0 == party1


def take_hitmonlee(emu: PyBoyEmulator,
                   ex: Executor) -> bool:
    """This save line's pick (Vlad, 2026-11-08)."""
    return take_hitmon(emu, ex, "hitmonlee")


def fighting_dojo(emu: PyBoyEmulator, ex: Executor,
                  choice: str = "hitmonlee") -> bool:
    """Clear the Fighting Dojo and take one gift
    (probe153; this line: Hitmonlee).

    Assumes the player is in Saffron City with Sabrina
    beaten. Order: enter, the four Black Belts (flags
    850-853 verified individually - the King's
    post-battle SetEventRange would mask them), the
    Karate King via his (4, 3) script trigger, then the
    chosen ball only. Ends back in Saffron City with the
    gift verified in the box. Blackout contingency (the
    brief's, never needed by the probe): exit or respawn,
    heal at the Saffron Pokecenter, re-enter and resume
    ONCE - flags persist; a second failure returns
    False.

    FIDELITY: faithful port of the probe152-style probe
    flow (probes 153b/153c/153e/153f), COLD-REPLAYED
    from probe152/sabrina.state - see docs
    supervisor-training.md section 18 for the replay's
    end-state comparison."""
    gift_event = DOJO_GIFTS[choice][3]
    if _event_flag(emu, gift_event):
        return DOJO_GIFTS[choice][2] in _box_species(emu)
    for _attempt in range(2):
        if not _enter_fighting_dojo(emu, ex):
            return False
        ok = True
        for cell, ev, sights, stances in DOJO_BELTS:
            if not _dojo_fight(emu, ex, cell, ev,
                               sights, stances):
                ok = False
                break
        if ok and not _event_flag(emu,
                                  EV_BEAT_KARATE_MASTER):
            # the King: step onto his trigger cell (4, 3)
            if not _dojo_walk(emu, ex, (4, 4)):
                ok = False
            else:
                emu.press("up")
                _explore_settle(emu)
                _dojo_ride_scene(emu)
                if not _event_flag(emu,
                                   EV_BEAT_KARATE_MASTER):
                    _face_and_talk(emu, DOJO_MASTER_CELL)
                    _dojo_ride_scene(emu)
                ok = _event_flag(emu,
                                 EV_BEAT_KARATE_MASTER)
        if ok:
            ok = take_hitmon(emu, ex, choice)
        if ok:
            break
        # blackout contingency: leave (or note the
        # respawn), heal, re-enter once
        if emu.read_state().map_id == MAP_FIGHTING_DOJO:
            if not _exit_fighting_dojo(emu, ex):
                return False
        if not _saffron_heal(emu, ex):
            return False
    else:
        return False
    if not _exit_fighting_dojo(emu, ex):
        return False
    st = emu.read_state()
    return _event_flag(emu, gift_event) and \
        _event_flag(emu, EV_BEAT_KARATE_MASTER) and \
        all(_event_flag(emu, e) for e in EV_DOJO_TRAINERS) \
        and DOJO_GIFTS[choice][2] in _box_species(emu) \
        and len(st.party) == 6


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


# ------------------------------------------------------------------
# Fuchsia leg (probe154): the road south, the Safari Zone
# HMs, and Koga. Four pieces of new machinery:
#
# * The Route 8 gatehouse, eastbound. Gate doors are
#   DIRECTIONAL: a warp cell fires only when stepped onto
#   heading INTO the building. The west strip of Route 8
#   strands every stock ride on the door's top cell (1, 9);
#   the crossing is a hand-walk to (0, 10) and a step EAST
#   onto the door (the mirror of probe151's westbound
#   crossing, which worked for the same reason).
#
# * A Safari engine. The world model's walkable() is
#   wrong for FOREST maps (two of Center's three regions
#   are unreachable in it): Gen 1 passability checks ONE
#   tile - the destination cell's BOTTOM-LEFT tile -
#   against the tileset's passable list (CheckTilePassable
#   + GetTileAndCoordsInFrontOfPlayer in pokered). The
#   grids below are compiled from pokered's .blk data
#   under that rule. Center's west doors open only into
#   its own sealed NW component, so the route to the
#   West area RINGS the zone: Center -> East -> North ->
#   West. wSafariSteps (0xD70D) is stored BIG-ENDIAN
#   (0x01F4 = 500 at entry) and counts down; the exit,
#   when the budget will not cover the ring back, is a
#   deliberate burn to ejection (items are kept). Safari
#   battles are RUN-only here - the typed driver would
#   throw balls, which is a policy decision, not a
#   default; encounters are recorded for the report.
#
# * _drive_battle_hardened: the stock _BattleDriver
#   wedges on two Gen 1 battle phases the Fuchsia gym
#   serves repeatedly - the shift prompt's party screen
#   with a text box open ("... is already out!"), where
#   its stale-signature branch mashes A and re-selects
#   the active mon forever; and move-rejection states
#   (Disable) with the same A-mashing shape. The hardened
#   loop gives the party screen unconditional B priority
#   (B backs out whether or not a text box is up) and
#   rotates its move pick when a confirmation spends no
#   PP. Gym trainers are engaged probe154's way: tile-
#   path steps (never through object cells), a sight
#   ride that presses ONLY A (a direction press during
#   a trainer's walk-up aborts the engagement), then a
#   talk whose first A is pressed blind - waiting for
#   the dialog before pressing A never opens it.
#
# * heal_at_pokecenter fails in Fuchsia (its counter
#   logic never completes there); the probe's manual
#   ride + nurse talk is ported as _heal_fuchsia_pc.
#
# The leg's bag prep also fixed deposit_item (above):
# its PC-menu signature was stale-byte fragile in
# lineages that had already used a PC - the probe's
# first three-TM deposit run withdrew the stored Potion
# and Lift Key with a neighbouring predicate before the
# corrected one deposited all four stacks cleanly
# (fully repaired in the same phase; the end state was
# verified stack-by-stack from RAM).
# ------------------------------------------------------------------

MAP_FUCHSIA = 7
MAP_FUCHSIA_PC = 154
MAP_WARDENS_HOUSE = 155
MAP_SAFARI_GATE = 156
MAP_FUCHSIA_GYM = 157
MAP_SZ_EAST = 217
MAP_SZ_NORTH = 218
MAP_SZ_WEST = 219
MAP_SZ_CENTER = 220
MAP_SZ_SECRET_HOUSE = 222
ITEM_GOLD_TEETH = 0x40
ITEM_HM03_SURF = 0xC6
ITEM_HM04_STRENGTH = 0xC7
ITEM_TM02 = 0xCA
ITEM_TM06_TOXIC = 0xCE
ITEM_TM21 = 0xDD
ITEM_TM34 = 0xEA
EV_BEAT_ROUTE12_FISHER5 = 1160
EV_IN_SAFARI_ZONE = 591
EV_GOT_HM03 = 2176
EV_GAVE_GOLD_TEETH = 569
EV_GOT_HM04 = 568
EV_GOT_TM06 = 600
EV_BEAT_KOGA = 601
EV_FUCHSIA_GYM_TRAINERS = (602, 603, 604, 605, 606, 607)
ADDR_SAFARI_STEPS = 0xD70D        # wSafariSteps, BIG-ENDIAN
ADDR_SAFARI_BALLS = 0xDA47        # wNumSafariBalls

# Passability grids compiled from pokered (FOREST .blk
# data, bottom-left-tile rule), one bit per cell,
# row-major. Ball cells are blocked at plan time from
# SAFARI_BALLS below (they are objects, not tiles).
SAFARI_GRIDS = {
    MAP_SZ_EAST: (30, 26, bytes.fromhex(
        "00000080ffffffe7ffffff01ffff7fbffbf7dffffefd07807f"
        "fff94fc87ffe3fff90ffcfbfe47f00bff91fc07ffefff887ff"
        "3ffee17f807ff81f8005fe07601ff001d807b8fff701eeff7d"
        "80fb7fdfef04c4f7fbefff7d00f87fffffff1f00000000")),
    MAP_SZ_NORTH: (40, 36, bytes.fromhex(
        "0000000000fcffffff3ffeffffff43feffffff4bfe0700f06f"
        "fef7ccff7ffef7ffff7ffef7ffff7ffcf0ffff7ffcf0ffff7f"
        "fef0fcff7ffef000007ffef0feff7efef0feff7e0efffeff7e"
        "0efffeff640efffeff6f0efffeff6f0efffeff6f0efffeff6f"
        "debffffd6fdebffffd6fdebffffd6fdebf47fc6fdeb0f7fb7f"
        "ceb0f7fb7fdeb0f7fb7fde30f11300def0fdfb3fdef0fdff3f"
        "defffdffffdedffdffffdefffdff3fdefff5ff3f0c03300000"
        "6dfbb7ffff")),
    MAP_SZ_WEST: (30, 26, bytes.fromhex(
        "3cff330c0f000c13f8ffef25fef77bbdffff5aefffbfd7bfdf"
        "eff5eff77bfdfbfd5eff60bf1700d8ef0520f67b01acfd5e00"
        "6fbfc7fffbf7f1fffe7deeff7f9ffb7fc8e7fedfff3901f07f"
        "fefffb9ffffffee7ffbfeffbffefff00fc3c00ffc0f00f")),
    MAP_SZ_CENTER: (30, 26, bytes.fromhex(
        "01c00060fffcffeb7ffffff9dfff7ffef7ff9ffffdffe77fff"
        "fff9dfff7ffe00c08f3f0000f00f3ffcffc30ffffe00c09f3f"
        "00f0e7bf6dfff900c07fc0ffff1fffffffe7ff0ffff9ffcb7f"
        "fefffb9fffffffe7fffbfff9ffff7f79dfbe6700300008")),
}
SAFARI_BALLS_CELLS = {
    MAP_SZ_EAST: frozenset({(21, 10), (3, 7), (20, 13),
                            (15, 12)}),
    MAP_SZ_NORTH: frozenset({(25, 1), (19, 7)}),
    MAP_SZ_WEST: frozenset({(8, 20), (9, 7), (18, 18),
                            (19, 7)}),
    MAP_SZ_CENTER: frozenset({(14, 10)}),
}
SAFARI_AREAS = frozenset(SAFARI_GRIDS)
_SZ_DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0),
            "right": (1, 0)}


def _safari_passable(map_id: int, cell) -> bool:
    w, h, bits = SAFARI_GRIDS[map_id]
    x, y = cell
    if not (0 <= x < w and 0 <= y < h):
        return False
    i = y * w + x
    return bool(bits[i // 8] & (1 << (i % 8)))


def _safari_steps(emu: PyBoyEmulator) -> int:
    """Steps left in the Safari game. The counter is
    stored big-endian (verified: 0x01F4 = 500 at entry,
    gate-interior steps not counted)."""
    mem = emu.pb.memory
    return (mem[ADDR_SAFARI_STEPS] << 8) | \
        mem[ADDR_SAFARI_STEPS + 1]


def _safari_balls(emu: PyBoyEmulator) -> int:
    return emu.pb.memory[ADDR_SAFARI_BALLS]


def _safari_run_battle(emu: PyBoyEmulator, seen: list) -> bool:
    """Leave a Safari battle with RUN - the ONLY action
    this leg takes in the zone (catching is a separate
    decision; the typed driver would throw balls).
    Records (species_id, level) in `seen`. Asserts the
    ball count never moves."""
    st = emu.read_state()
    if not st.in_battle:
        return True
    mem = emu.pb.memory
    seen.append((mem[ADDR_ENEMY_SPECIES],
                 mem[ADDR_ENEMY_LEVEL]))
    for _ in range(8):
        if not emu.read_state().in_battle:
            return _safari_balls(emu) == 30
        _run_away(emu)
    return not emu.read_state().in_battle and \
        _safari_balls(emu) == 30


def _safari_settle(emu: PyBoyEmulator, seen: list) -> None:
    """Settle one in-zone step: RUN from any encounter,
    close any sign dialog, wait for the walk to rest."""
    for _ in range(40):
        st = emu.read_state()
        if st.in_battle:
            if not _safari_run_battle(emu, seen):
                raise RuntimeError("safari RUN failed")
            continue
        if st.in_dialog:
            emu.close_dialog(max_presses=2)
            continue
        emu.tick(6)
        st2 = emu.read_state()
        if (st2.x, st2.y) == (st.x, st.y) \
                and not st2.in_dialog and not st2.in_battle:
            return


def _safari_warp_edges(ex: Executor, map_id: int) -> dict:
    """cell -> (dest_map, landing cell) for inter-area
    warps, from the world model (Warp.dest_id is the
    destination MAP; Warp.idx its landing warp index)."""
    out = {}
    for w in ex.world.maps[map_id].warps:
        if w.dest_id in SAFARI_AREAS:
            land = ex.world.maps[w.dest_id].warps[w.idx - 1]
            out[(w.x, w.y)] = (w.dest_id, (land.x, land.y))
    return out


def _safari_plan(ex: Executor, start, goal, blocked):
    """BFS over (map, cell) across the Safari areas.
    Returns the action list: direction strings, with
    "warp" for the edge taken while standing on a warp
    cell - or None. Pure planning; execution replans
    after every step."""
    from collections import deque
    edges = {m: _safari_warp_edges(ex, m) for m in SAFARI_AREAS}
    prev = {start: None}
    dq = deque([start])
    while dq:
        node = dq.popleft()
        if node == goal:
            break
        m, c = node
        for btn, (dx, dy) in _SZ_DIRS.items():
            nb = (c[0] + dx, c[1] + dy)
            st = (m, nb)
            if st in prev or nb in blocked[m]:
                continue
            if not _safari_passable(m, nb) \
                    and nb not in edges[m]:
                continue
            prev[st] = (node, btn)
            dq.append(st)
        if c in edges[m]:
            st = edges[m][c]
            if st not in prev:
                prev[st] = (node, "warp")
                dq.append(st)
    if goal not in prev:
        return None
    path = []
    n = goal
    while prev[n] is not None:
        pn, pb = prev[n]
        path.append(pb)
        n = pn
    return list(reversed(path))


def _sz_nav(emu: PyBoyEmulator, ex: Executor, goal_map: int,
            goal_cell, seen: list,
            budget_guard: int = 45) -> bool:
    """Walk the Safari areas to (goal_map, goal_cell),
    replanning after every step and learning blocked
    cells (tile-pair collisions add blocks the tile
    grid cannot see). Aborts under the step guard."""
    blocked = {m: set(SAFARI_BALLS_CELLS.get(m, ()))
               for m in SAFARI_AREAS}
    edges = {m: _safari_warp_edges(ex, m) for m in SAFARI_AREAS}
    for _ in range(1200):
        st = emu.read_state()
        cur = (st.map_id, (st.x, st.y))
        if cur == (goal_map, goal_cell):
            return True
        if st.map_id not in SAFARI_AREAS:
            return False
        if _safari_steps(emu) < budget_guard:
            return False
        path = _safari_plan(ex, cur, (goal_map, goal_cell),
                            blocked)
        if not path:
            return False
        btn = path[0]
        if btn == "warp":
            # standing on the warp cell without it having
            # fired (a landing): step off and back on
            dest = edges[cur[0]][cur[1]]
            stepped = False
            for b2, (dx, dy) in _SZ_DIRS.items():
                nb = (cur[1][0] + dx, cur[1][1] + dy)
                if _safari_passable(cur[0], nb) \
                        and nb not in blocked[cur[0]]:
                    emu.press(b2)
                    _safari_settle(emu, seen)
                    emu.press({"up": "down", "down": "up",
                               "left": "right",
                               "right": "left"}[b2])
                    _safari_settle(emu, seen)
                    stepped = \
                        emu.read_state().map_id == dest[0]
                    break
            if not stepped \
                    and emu.read_state().map_id == cur[0]:
                return False
            continue
        dx, dy = _SZ_DIRS[btn]
        want = (cur[1][0] + dx, cur[1][1] + dy)
        emu.press(btn)
        _safari_settle(emu, seen)
        st = emu.read_state()
        if st.map_id == cur[0] and (st.x, st.y) == cur[1]:
            blocked[cur[0]].add(want)
    return False


def _safari_warp_step(emu: PyBoyEmulator, seen: list,
                      btn: str, expect_map: int) -> bool:
    """Step onto a door/edge warp cell until the map
    changes (doors fire only when stepped onto heading
    in, so this is always a single deliberate step)."""
    for _ in range(6):
        emu.press(btn)
        _explore_settle(emu)
        _safari_settle(emu, seen)
        if emu.read_state().map_id == expect_map:
            return True
    return emu.read_state().map_id == expect_map


def safari_zone_hms(emu: PyBoyEmulator, ex: Executor,
                    seen: list = None) -> bool:
    """One Safari entry for the two HMs (probe154):
    pay at the gate, ring Center -> East -> North ->
    West, take the GOLD TEETH, take HM03 SURF from the
    Secret House guru, and leave - by deliberate
    ejection when the step budget will not cover the
    ring back. NOTHING is caught: every encounter is
    RUN from (species are appended to `seen` if given).
    Ends back in Fuchsia City with both items in the
    bag, verified from RAM."""
    if seen is None:
        seen = []
    world = ex.world
    if emu.read_state().map_id != MAP_SAFARI_GATE:
        if not _ride_battling(emu, ex, MAP_SAFARI_GATE):
            return False
    # pay the clerk: he stands at (6, 2) behind the
    # counter; talk across it from (4, 2) facing right
    money0 = emu.read_state().money
    _walk_fighting(emu, ex, MAP_SAFARI_GATE, (4, 2),
                   attempts=20)
    if (emu.read_state().x, emu.read_state().y) != (4, 2):
        return False
    emu.press("right")
    emu.tick(20)
    _press10(emu, "a")
    paid = False
    for _ in range(200):
        if _safari_balls(emu) == 30 and \
                _event_flag(emu, EV_IN_SAFARI_ZONE):
            paid = True
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(25)
    if not paid or emu.read_state().money != money0 - 500:
        return False
    # north into Center
    for _ in range(10):
        emu.press("up")
        _explore_settle(emu)
        _safari_settle(emu, seen)
        if emu.read_state().map_id == MAP_SZ_CENTER:
            break
    if emu.read_state().map_id != MAP_SZ_CENTER:
        return False
    # ring to the West area: the teeth at (19, 7),
    # stance (19, 8) facing up
    if not _sz_nav(emu, ex, MAP_SZ_WEST, (19, 8), seen):
        return False
    emu.press("up")
    emu.tick(20)
    _press10(emu, "a")
    for _ in range(120):
        if _bag_count(emu, ITEM_GOLD_TEETH) > 0:
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(25)
    if _bag_count(emu, ITEM_GOLD_TEETH) != 1:
        return False
    # the Secret House at West (3, 3), stance (3, 4)
    if not _sz_nav(emu, ex, MAP_SZ_WEST, (3, 4), seen,
                   budget_guard=5):
        return False
    if not _safari_warp_step(emu, seen, "up",
                             MAP_SZ_SECRET_HOUSE):
        return False
    # the guru at (3, 3): HM03's event flag is set at
    # the END of his script, so ride the text fully
    got = False
    for stance, face in (((3, 4), "up"), ((2, 3), "right"),
                         ((4, 3), "left"), ((3, 2), "down")):
        _walk_fighting(emu, ex, MAP_SZ_SECRET_HOUSE, stance,
                       attempts=20)
        st = emu.read_state()
        if (st.x, st.y) != stance:
            continue
        emu.press(face)
        emu.tick(20)
        _press10(emu, "a")
        quiet = 0
        for _ in range(300):
            if emu.read_state().in_dialog:
                quiet = 0
                _press10(emu, "a")
            else:
                quiet += 1
                if quiet >= 6 and \
                        _bag_count(emu, ITEM_HM03_SURF) > 0:
                    break
            emu.tick(25)
        if _bag_count(emu, ITEM_HM03_SURF) > 0:
            got = True
            break
    if not got or not _event_flag(emu, EV_GOT_HM03):
        return False
    # out of the house
    for _ in range(8):
        emu.press("down")
        _explore_settle(emu)
        _safari_settle(emu, seen)
        if emu.read_state().map_id == MAP_SZ_WEST:
            break
    if emu.read_state().map_id != MAP_SZ_WEST:
        return False
    # leave: ring back if the budget clearly covers it,
    # else burn the counter to ejection (items kept)
    if _safari_steps(emu) >= 260 and _sz_nav(
            emu, ex, MAP_SZ_CENTER, (14, 24), seen,
            budget_guard=30):
        _safari_warp_step(emu, seen, "down", MAP_SAFARI_GATE)
    else:
        for i in range(240):
            if emu.read_state().map_id == MAP_SAFARI_GATE:
                break
            emu.press("left" if i % 2 == 0 else "right")
            _safari_settle(emu, seen)
    if emu.read_state().map_id != MAP_SAFARI_GATE:
        return False
    for _ in range(80):
        st = emu.read_state()
        if st.in_dialog:
            _press10(emu, "a")
            emu.tick(25)
            continue
        emu.tick(20)
    if _safari_balls(emu) != 0 or \
            _event_flag(emu, EV_IN_SAFARI_ZONE):
        return False
    for _ in range(10):
        emu.press("down")
        _explore_settle(emu)
        st = emu.read_state()
        if st.in_dialog:
            emu.close_dialog(max_presses=3)
        if st.map_id == MAP_FUCHSIA:
            break
    return emu.read_state().map_id == MAP_FUCHSIA and \
        _bag_count(emu, ITEM_GOLD_TEETH) == 1 and \
        _bag_count(emu, ITEM_HM03_SURF) == 1


def give_teeth_get_strength(emu: PyBoyEmulator,
                            ex: Executor) -> bool:
    """The Warden exchange (probe154): his house is the
    SE Fuchsia warp (27, 27); he stands at (2, 3). With
    the GOLD TEETH in the bag his talk hands over HM04
    STRENGTH and takes the teeth - both directions
    verified from RAM, events at the script's end."""
    if not _ride_battling(emu, ex, MAP_WARDENS_HOUSE):
        return False
    for stance, face in (((2, 4), "up"), ((1, 3), "right"),
                         ((3, 3), "left"), ((2, 2), "down")):
        _walk_fighting(emu, ex, MAP_WARDENS_HOUSE, stance,
                       attempts=20)
        st = emu.read_state()
        if (st.x, st.y) != stance:
            continue
        emu.press(face)
        emu.tick(20)
        _press10(emu, "a")
        quiet = 0
        for _ in range(300):
            if emu.read_state().in_dialog:
                quiet = 0
                _press10(emu, "a")
            else:
                quiet += 1
                if quiet >= 6 and \
                        _bag_count(emu, ITEM_HM04_STRENGTH) > 0:
                    break
            emu.tick(25)
        if _bag_count(emu, ITEM_HM04_STRENGTH) > 0:
            break
    return _bag_count(emu, ITEM_GOLD_TEETH) == 0 and \
        _bag_count(emu, ITEM_HM04_STRENGTH) == 1 and \
        _event_flag(emu, EV_GAVE_GOLD_TEETH) and \
        _event_flag(emu, EV_GOT_HM04)


def _cross_r8_gate_east(emu: PyBoyEmulator,
                        ex: Executor) -> bool:
    """Route 8's gatehouse, eastbound (probe154): from
    the west strip, stand at (0, 10) and step EAST onto
    the door; walk the corridor east and out. Verifies
    both map changes and the east-segment landing."""
    gate = ex.world.map_id("Route8Gate")
    r8 = ex.world.map_id("Route8")
    _walk_fighting(emu, ex, r8, (0, 10), attempts=30)
    st = emu.read_state()
    if (st.x, st.y) != (0, 10):
        return False
    if not _step_until_map(emu, "right", gate):
        return False
    if not _step_until_map(emu, "right", r8):
        return False
    st = emu.read_state()
    return st.map_id == r8 and st.x >= 8


def _heal_fuchsia_pc(emu: PyBoyEmulator,
                     ex: Executor) -> bool:
    """Fuchsia heal (probe154): heal_at_pokecenter's
    counter logic never completes in this Pokecenter,
    so ride in and talk to the nurse directly; verify
    the whole party at full HP."""
    fpc = ex.world.map_id("FuchsiaPokecenter")
    if emu.read_state().map_id != fpc:
        if not _ride_battling(emu, ex, fpc):
            return False
    for stance in ((3, 3), (4, 3)):
        _walk_fighting(emu, ex, fpc, stance, attempts=20)
        st = emu.read_state()
        if (st.x, st.y) != stance:
            continue
        emu.press("up")
        emu.tick(20)
        _press10(emu, "a")
        quiet = 0
        for _ in range(300):
            st = emu.read_state()
            if st.in_dialog:
                quiet = 0
                _press10(emu, "a")
            else:
                quiet += 1
                if quiet >= 6 and all(
                        m.hp == m.max_hp for m in st.party):
                    break
            emu.tick(25)
        if all(m.hp == m.max_hp
               for m in emu.read_state().party):
            return True
    return all(m.hp == m.max_hp
               for m in emu.read_state().party)


def fuchsia_road(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Saffron -> Fuchsia (probe154). Bag prep first:
    deposit TM02/TM21/TM34 at the Saffron PC (the leg
    adds HM03, HM04, the teeth and TM06 to a 19-stack
    bag; Gen 1's cap is 20). Then the road south as
    per-map hops with the gate crossings by hand.
    Route 12's Fisher5 (event 1160) is NEVER engaged -
    his west-shore pocket is tile-sealed - and the
    guard is asserted at every checkpoint. Ends healed
    at the Fuchsia Pokecenter."""
    world = ex.world
    pc = world.map_id("SaffronPokecenter")
    city = world.map_id("SaffronCity")
    if _event_flag(emu, EV_BEAT_ROUTE12_FISHER5):
        return False
    if emu.read_state().map_id != pc:
        if not _enter_saffron_pc(emu, ex):
            return False
    for tm in (ITEM_TM02, ITEM_TM21, ITEM_TM34):
        if _bag_count(emu, tm) > 0:
            if not deposit_item(emu, ex, pc, tm):
                return False
    from agent.executor import StuckError
    bag = emu.read_state()
    if len(_bag_items(emu)) != 16:
        return False
    # exit the Pokecenter
    for door in ((3, 7), (4, 7)):
        try:
            ex._walk_to(pc, door)
        except StuckError:
            continue
        for _ in range(6):
            emu.press("down")
            _explore_settle(emu)
            if emu.read_state().map_id != pc:
                break
        if emu.read_state().map_id != pc:
            break
    if emu.read_state().map_id != city:
        return False
    if not _hop_ride(emu, ex, ["Route8"]):
        return False
    if not _cross_r8_gate_east(emu, ex):
        return False
    if not _hop_ride(emu, ex, ["LavenderTown"]):
        return False
    if not _hop_ride(emu, ex, ["Route12"]):
        return False
    if _event_flag(emu, EV_BEAT_ROUTE12_FISHER5):
        return False
    if not _cross_gate12_south(emu, ex):
        return False
    if not _pier_walk(emu, ex, PIER_WAYPOINTS_S):
        return False
    if _event_flag(emu, EV_BEAT_ROUTE12_FISHER5):
        return False
    if not _step_until_map(emu, "down",
                           world.map_id("Route13")):
        return False
    _walk_fighting(emu, ex, world.map_id("Route13"),
                   (1, 10), attempts=30)
    if not _step_until_map(emu, "left",
                           world.map_id("Route14")):
        return False
    if not _ride_battling(emu, ex, world.map_id("Route15")):
        return False
    if not _ride_battling(emu, ex, MAP_FUCHSIA):
        return False
    if _event_flag(emu, EV_BEAT_ROUTE12_FISHER5):
        return False
    return _heal_fuchsia_pc(emu, ex)


def _drive_battle_hardened(emu: PyBoyEmulator) -> bool:
    """Battle driver for the Fuchsia gym (probe154),
    written after the stock driver wedged twice:
    (1) the shift prompt's party screen with a text box
    open - B is pressed BEFORE any dialog A, because A
    there re-selects the active mon and the "already
    out!" text loops forever; after a dozen B's it
    genuinely switches to the next healthy mon instead.
    (2) move rejection (Disable and friends) - the pick
    rotates when a confirmation spends no PP. Move
    preference is slot order [0, 2, 3, 1] filtered by
    the ACTIVE mon's PP (for Blastoise: Hydro Pump,
    Bubble, Water Gun, Tail Whip)."""
    mem = emu.pb.memory
    drv = _BattleDriver(emu)
    want_cycle = [0, 2, 3, 1]
    wi = 0
    stall = 0
    party_bs = 0
    for _ in range(8000):
        st = emu.read_state()
        if not st.in_battle:
            return True
        top_y = mem[ADDR_MENU_TOP_Y]
        item = mem[ADDR_MENU_ITEM]
        col = mem[ADDR_MENU_COL]
        if top_y == 1:                # party screen
            party_bs += 1
            if party_bs > 12:
                _press10(emu, "down")
                emu.tick(15)
                _press10(emu, "a")
                emu.tick(40)
                party_bs = 0
            else:
                _press10(emu, "b")
                emu.tick(30)
            continue
        if st.in_dialog:
            _press10(emu, "a")
            emu.tick(15)
            continue
        if top_y == 12:               # move menu
            active = drv._active_party_idx()
            mon = st.party[active]
            usable = [s for s in want_cycle
                      if s < len(mon.moves) and mon.moves[s]
                      and s < len(mon.pp) and mon.pp[s] > 0]
            if not usable:
                usable = [0]
            want = usable[wi % len(usable)] + 1
            if item == want:
                pp = tuple(mon.pp)
                _press10(emu, "a")
                emu.tick(40)
                pp2 = tuple(emu.read_state()
                            .party[active].pp)
                if pp2 == pp:
                    stall += 1
                    if stall >= 2:
                        wi += 1
                        stall = 0
                else:
                    stall = 0
            elif item < want:
                _press10(emu, "down")
                emu.tick(15)
            else:
                _press10(emu, "up")
                emu.tick(15)
            continue
        if top_y == 14:               # main menu
            if col == 15:
                _press10(emu, "left")
            elif item == 1:
                _press10(emu, "up")
            else:
                _press10(emu, "a")
            emu.tick(20)
            continue
        emu.tick(30)
    return not emu.read_state().in_battle


# (cell, stance, face, event) in object order. Facings:
# R1 Juggler7 (8, 13) DOWN - Hypno 38; R2 Juggler3
# (7, 8) RIGHT - Drowzee/Drowzee/Kadabra/Drowzee 31;
# R3 Juggler8 (1, 12) DOWN - Drowzee 34/Kadabra 34;
# R4 Tamer1 (3, 5) UP (his sight line is wall-blocked;
# talk only) - Sandslash 34/Arbok 34; R5 Tamer2 (8, 2)
# DOWN - Arbok/Sandslash/Arbok 33; R6 Juggler4 (2, 7)
# LEFT - Drowzee 34/Hypno 34.
FUCHSIA_GYM_TRAINERS = (
    ((8, 13), (8, 12), "down", 602),
    ((7, 8), (7, 9), "up", 603),
    ((1, 12), (1, 13), "up", 604),
    ((3, 5), (2, 5), "right", 605),
    ((8, 2), (8, 3), "up", 606),
    ((2, 7), (1, 7), "right", 607),
)
FUCHSIA_GYM_CELLS = frozenset(
    [c for c, _, _, _ in FUCHSIA_GYM_TRAINERS]
    + [(4, 10), (7, 15)])     # Koga, the gym guide
KOGA_CELL = (4, 10)
KOGA_STANCE = (4, 11)


def _fuchsia_gym_step_to(emu: PyBoyEmulator,
                         ex: Executor, target) -> bool:
    """Tile-path stepping inside Fuchsia Gym (the
    invisible walls are real collision tiles and the
    world model's grid sees them - probe154 verified
    the model against RAM for the whole room). Object
    cells are never routed through."""
    world = ex.world
    learned = set()     # cells a press proved blocked
    for _ in range(120):
        st = emu.read_state()
        if st.in_battle or st.in_dialog:
            # a sight engagement is pending (its dialog
            # is up); the caller rides it - direction
            # presses here would be eaten anyway
            return True
        if (st.x, st.y) == target:
            return True
        path = world.tile_path(MAP_FUCHSIA_GYM,
                               (st.x, st.y), target,
                               blocked=frozenset(
                                   (FUCHSIA_GYM_CELLS
                                    | learned) - {target}))
        if not path or len(path) < 2:
            return False
        nxt = path[1]
        d = (nxt[0] - st.x, nxt[1] - st.y)
        btn = {v: k for k, v in _SZ_DIRS.items()}[d]
        emu.press(btn)
        _explore_settle(emu)
        st2 = emu.read_state()
        if (st2.x, st2.y) == (st.x, st.y) \
                and not st2.in_battle and not st2.in_dialog:
            # the press moved nothing and no engagement
            # is pending: the cell is sprite-blocked (a
            # beaten trainer who walked up stands there)
            learned.add(nxt)
    st = emu.read_state()
    return (st.x, st.y) == target or st.in_battle


def _fuchsia_gym_fight(emu: PyBoyEmulator, ex: Executor,
                       stance, face: str, event: int) -> bool:
    """Engage one Fuchsia gym trainer (or Koga) and
    fight him with the hardened driver. Two phases:
    ride a possibly-running sight engagement pressing
    ONLY A (a direction press during the walk-up
    aborts it), then talk - turn toward the trainer
    (his cell is occupied, so the press only turns)
    and press A blind."""
    if _event_flag(emu, event):
        return True
    # best effort: the walk may be interrupted by the
    # sight engagement itself, which the ride below
    # resolves - step_to's result is not a verdict
    _fuchsia_gym_step_to(emu, ex, stance)
    for _ in range(60):
        st = emu.read_state()
        if st.in_battle:
            break
        if st.in_dialog:
            _press10(emu, "a")
        emu.tick(30)
    if not emu.read_state().in_battle:
        emu.press(face)
        emu.tick(20)
        _press10(emu, "a")
        for _ in range(500):
            st = emu.read_state()
            if st.in_battle:
                break
            if st.in_dialog:
                _press10(emu, "a")
            emu.tick(30)
    if emu.read_state().in_battle:
        _drive_battle_hardened(emu)
    # ride the post-battle text fully: leader scripts
    # set their events at the END (HM03's lesson)
    quiet = 0
    for _ in range(400):
        st = emu.read_state()
        if st.in_dialog:
            quiet = 0
            _press10(emu, "a")
        else:
            quiet += 1
            if quiet >= 8 and _event_flag(emu, event):
                break
        emu.tick(25)
    return _event_flag(emu, event)


def _fuchsia_gym_heal_trip(emu: PyBoyEmulator,
                            ex: Executor) -> bool:
    """Out of the gym, healed at the Fuchsia PC, and
    back in - the probe needed two of these (a poison
    spiral in the R2 fight, Hydro Pump running dry
    before R4). Trainer flags persist, so the sweep
    simply resumes."""
    from agent.executor import StuckError
    for door in ((4, 17), (5, 17)):
        try:
            ex._walk_to(MAP_FUCHSIA_GYM, door)
        except StuckError:
            continue
        for _ in range(6):
            emu.press("down")
            _explore_settle(emu)
            if emu.read_state().map_id != MAP_FUCHSIA_GYM:
                break
        if emu.read_state().map_id != MAP_FUCHSIA_GYM:
            break
    if emu.read_state().map_id == MAP_FUCHSIA_GYM:
        return False
    if not _heal_fuchsia_pc(emu, ex):
        return False
    return _ride_battling(emu, ex, MAP_FUCHSIA_GYM)


def beat_koga(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Fuchsia Gym (probe154): six trainers, then Koga
    (Koffing 37 / Muk 39 / Koffing 37 / Weezing 43 -
    the Weezing's Selfdestruct fainted Blastoise in the
    probe and KO'd itself with him as Koga's last mon,
    so the badge still landed). Verifies every trainer
    flag, the Soul Badge (6 badge bits), event 601 and
    TM06 in the bag; ends healed at the Fuchsia PC."""
    if not _ride_battling(emu, ex, MAP_FUCHSIA_GYM):
        return False

    def healthy_enough() -> bool:
        st = emu.read_state()
        lead = st.party[0]
        if lead.hp == 0 or lead.hp * 5 < lead.max_hp * 2:
            return _fuchsia_gym_heal_trip(emu, ex)
        return True

    for cell, stance, face, ev in FUCHSIA_GYM_TRAINERS:
        if not _event_flag(emu, ev) and not healthy_enough():
            return False
        if not _fuchsia_gym_fight(emu, ex, stance, face, ev):
            return False
    if not healthy_enough():
        return False
    if not _fuchsia_gym_fight(emu, ex, KOGA_STANCE, "up",
                              EV_BEAT_KOGA):
        return False
    st = emu.read_state()
    if len(st.badges) != 6 or "Soul" not in st.badges:
        return False
    if _bag_count(emu, ITEM_TM06_TOXIC) != 1 or \
            not _event_flag(emu, EV_GOT_TM06):
        return False
    # exit south and heal
    from agent.executor import StuckError
    for door in ((4, 17), (5, 17)):
        try:
            ex._walk_to(MAP_FUCHSIA_GYM, door)
        except StuckError:
            continue
        for _ in range(6):
            emu.press("down")
            _explore_settle(emu)
            if emu.read_state().map_id != MAP_FUCHSIA_GYM:
                break
        if emu.read_state().map_id != MAP_FUCHSIA_GYM:
            break
    return _heal_fuchsia_pc(emu, ex)


# -----------------------------------------------------------------
# Surf for Lapras + the Safari rare hunt (probe155). Job C -
# teach HM04 Strength to Arcanine over Roar - is NOT here: it
# is engine-impossible (see safari_rare_hunt's docstring and
# docs S20); no code in this section teaches Strength.

MAP_FUCHSIA_GOOD_ROD_HOUSE = 164
ITEM_GOOD_ROD = 0x4D
ITEM_SUPER_ROD = 0x4E
ITEM_TM24 = 0xE0
SPECIES_LAPRAS = 0x13
MOVE_SURF = 57
MOVE_STRENGTH = 70

# Safari species by Gen 1 INTERNAL id (pokered
# constants/pokemon_constants.asm; the parse must honour
# const_skip/const_next - a naive sequential parse shifts
# every later id, the same trap family as the event ids in
# S14/S15. Validated against all 16 species constants this
# file already carried plus the S19 raw Chansey byte 0x28.)
SAFARI_SPECIES = {
    0x28: "CHANSEY", 0x02: "KANGASKHAN", 0x3C: "TAUROS",
    0x1A: "SCYTHER", 0x58: "DRATINI", 0x12: "RHYHORN",
    0x41: "VENONAT", 0x77: "VENOMOTH", 0x0C: "EXEGGCUTE",
    0xA7: "NIDORINO", 0xA8: "NIDORINA", 0x6D: "PARAS",
    0x2E: "PARASECT", 0x46: "DODUO", 0x03: "NIDORAN_M",
    0x4E: "KRABBY", 0x2F: "PSYDUCK", 0x25: "SLOWPOKE",
    0x9D: "GOLDEEN", 0x47: "POLIWAG",
}
# Base catch rates / speeds (pokered data/pokemon/base_stats
# + the Safari flee formula in engine/battle/core.asm:
# after each player action the enemy flees iff a random
# byte < (speed % 256) * 2, divided by 4 while the bait
# factor (0xCCE9) is up, doubled while the rock/escape
# factor (0xCCE8) is up; speed >= 128 always flees - the
# doubling carries. Bait halves the actual catch rate,
# rock doubles it (item_effects.asm).)
SAFARI_CATCH_RATE = {
    0x28: 30, 0x02: 45, 0x3C: 45, 0x1A: 45, 0x58: 45,
    0x12: 120, 0x41: 190, 0x77: 75, 0x0C: 90, 0xA7: 120,
    0xA8: 120, 0x6D: 190, 0x2E: 75, 0x46: 190, 0x03: 190,
    0x4E: 225, 0x2F: 190, 0x25: 190, 0x9D: 225, 0x47: 190,
}
SAFARI_SPEED = {
    0x28: 50, 0x02: 90, 0x3C: 110, 0x1A: 105, 0x58: 50,
    0x12: 25, 0x41: 45, 0x77: 90, 0x0C: 40, 0xA7: 65,
    0xA8: 56, 0x6D: 25, 0x2E: 30, 0x46: 75, 0x03: 45,
    0x4E: 50, 0x2F: 55, 0x25: 15, 0x9D: 63, 0x47: 90,
}
# The five rares Vlad named. Encounter slots (pokered
# data/wild/grass_water + probabilities.asm: slot8 = 4.3%,
# slot9 = 1.2%): East KANGASKHAN s8 / SCYTHER s9; North
# CHANSEY s8 / TAUROS s9; West TAUROS s8 / KANGASKHAN s9;
# Center SCYTHER s8 / CHANSEY s9. DRATINI is fishing-only:
# the Super Rod group shared by all four areas is
# Dratini/Krabby/Psyduck/Slowpoke Lv15 (12.5% per cast);
# the Good Rod yields only Goldeen/Poliwag Lv10 anywhere,
# so Dratini needs the Super Rod (Route 12 guru's house).
SAFARI_TARGETS = frozenset({0x28, 0x02, 0x3C, 0x1A, 0x58})
# Grass cells per area, compiled from pokered .blk +
# forest.bst under the same bottom-left-tile rule as
# SAFARI_GRIDS (FOREST grass tile $20). Same bit order.
SAFARI_GRASS = {
    MAP_SZ_EAST: (30, 26, bytes.fromhex("00000000c0fff303f03ffe01f8c77f0000f01f0000f8070000ff0000801608010000c30000200f0000c8030000f2fc00803c3f00c030000020040000000000000000000000000000000000000000070000c00300c0ff0000e01f00c0ff0300000000")),
    MAP_SZ_NORTH: (40, 36, bytes.fromhex("0000000000000000c03f980100e003fc0300f001fc0300f000f80100f000f80100f001fc0300e003fc0000c007980000800f007000000f00f000000e007000000000f000000000fc000000007c000000003f00000000260000000000000000000000000000000000000000000000801e000000c03f000000c030f00100c030f00300c030f00300c010f00300c000fc03008000f803000000f007000000f00f000000c00f00000040060000000000000000000000")),
    MAP_SZ_WEST: (30, 26, bytes.fromhex("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008000000030000000fcff0300ffff0000ff0f00809f01000000000000000000")),
    MAP_SZ_CENTER: (30, 26, bytes.fromhex("0000000000000000066006c003fc03f881ff01fff0ffc00ffc3fe0019e07b0000000340000000c000000030000c000c0012000f00000007c0000003e0000c0070000f0037e00f8c03f001ff00f8003f8017000fc030e007ec0010000000000000000")),
}
# Entry landing per area (where a transit arrives); grass
# pairs are chosen nearest the anchor.
SAFARI_ANCHORS = {MAP_SZ_CENTER: (15, 25), MAP_SZ_EAST: (0, 22),
                  MAP_SZ_NORTH: (20, 35), MAP_SZ_WEST: (29, 22)}
SAFARI_AREAS_BY_NAME = {"center": MAP_SZ_CENTER,
                        "east": MAP_SZ_EAST,
                        "north": MAP_SZ_NORTH,
                        "west": MAP_SZ_WEST}
ADDR_SAFARI_BAIT_FACTOR = 0xCCE9
ADDR_SAFARI_ESCAPE_FACTOR = 0xCCE8


def _pc_swap_with_heal(emu: PyBoyEmulator, ex: Executor,
                       pc_map: int, deposit_species: int,
                       withdraw_species: int, heal) -> bool:
    """pc_swap's menu flow with a caller-chosen final heal:
    stock pc_swap ends in heal_at_pokecenter, which S19
    proved fails in Fuchsia (its exit step misses the
    center's door mat), so probe155's swap replica ends in
    _heal_fuchsia_pc instead. Effects verified in RAM."""
    mem = emu.pb.memory
    if _party_index_of(emu, withdraw_species) is not None and \
            _party_index_of(emu, deposit_species) is None:
        return True
    dep_idx = _party_index_of(emu, deposit_species)
    if dep_idx is None or withdraw_species not in _box_species(emu):
        return False
    if not _open_pc(emu, ex, pc_map):
        return False

    def sig():
        return (mem[ADDR_MENU_TOP_Y], mem[ADDR_MENU_ITEM],
                mem[ADDR_MAX_MENU_ITEM])

    def wait_sig(pred, rounds=120, ride_a=False):
        for _ in range(rounds):
            if pred(sig()):
                return True
            if ride_a and emu.read_state().in_dialog:
                _press10(emu, "a")
            emu.tick(25)
        return pred(sig())

    def bills_menu():
        return wait_sig(lambda s: s[2] == 4, ride_a=True)

    if not bills_menu():
        if not wait_sig(lambda s: s[2] == 3):
            return False
        if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 0):
            return False
        _press10(emu, "a")                    # BILL'S PC
        if not bills_menu():
            return False

    def pick_from_list(index, done):
        emu.tick(60)
        if not _steer_menu(
                emu, lambda: mem[ADDR_MENU_SCROLL]
                + mem[ADDR_MENU_ITEM], index):
            return False
        _press10(emu, "a")
        emu.tick(90)
        if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 0):
            return False
        _press10(emu, "a")
        return _pc_press_a_ride(emu, done, rounds=160)

    n_party = len(emu.read_state().party)
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 1):
        return False
    _press10(emu, "a")                        # DEPOSIT
    if not pick_from_list(
            dep_idx,
            lambda: len(emu.read_state().party) == n_party - 1):
        return False
    if deposit_species not in _box_species(emu):
        return False
    if not bills_menu():
        return False
    v_idx = _box_species(emu).index(withdraw_species)
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 0):
        return False
    _press10(emu, "a")                        # WITHDRAW
    if not pick_from_list(
            v_idx, lambda: len(emu.read_state().party) == n_party):
        return False
    if _party_index_of(emu, withdraw_species) is None:
        return False
    if not bills_menu():
        return False
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 4):
        return False
    _press10(emu, "a")                        # SEE YA!
    if not wait_sig(lambda s: s[2] == 3, ride_a=True):
        return False
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 3):
        return False
    _press10(emu, "a")                        # LOG OFF
    emu.tick(90)
    _close_menus(emu)
    emu.wait_free(max_rounds=10, stable_needed=3)
    return bool(heal(emu, ex))


def _teach_hm_free_slot(emu: PyBoyEmulator, mon_idx: int,
                        item_id: int, move_id: int) -> bool:
    """Teach an HM into a FREE slot via the bag, with the
    strict probe155 assertion as a check: the ONLY legal
    change is the move appended in the first empty slot
    (pokered LearnMove fills an empty slot first; the
    forget-flow runs only at four full slots, so with a
    free slot a replacement prompt is structurally
    impossible). A four-move pupil returns False untouched
    - choosing a move to forget is the player's call, and
    engine CANNOT-learn cases (the HM04/Arcanine finding)
    also land here: the teach simply never happens."""
    from agent.state import ADDR_PARTY_MONS
    mem = emu.pb.memory

    def raw_moves():
        base = ADDR_PARTY_MONS + 44 * mon_idx
        return [mem[base + 8 + j] for j in range(4)]

    before = raw_moves()
    if move_id in before:
        return True
    if sum(1 for m in before if m) >= 4:
        return False
    if not _use_bag_item_on_mon(emu, item_id, mon_idx):
        return False
    for _ in range(60):
        if move_id in raw_moves():
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
            continue
        emu.tick(30)
    _close_menus(emu)
    after = raw_moves()
    slot = next((i for i, m in enumerate(before) if m == 0),
                None)
    if slot is None:
        return False
    want = list(before)
    want[slot] = move_id
    return after == want


def teach_hm_to_boxed(emu: PyBoyEmulator, ex: Executor,
                      pc_map: int, deposit_species: int,
                      withdraw_species: int, hm_item: int,
                      hm_move: int, heal) -> bool:
    """Job A (probe155, Vlad's call): swap a boxed mon into
    the party at a PokeCenter and teach it an HM, leaving
    every other party member's moves untouched. Fuchsia:
    NidoranM out, Lapras in, HM03 -> SURF (Lapras had two
    moves, so the teach filled a free slot; Blastoise's
    four move bytes are byte-identical afterwards - the
    replay harness asserts exactly that). `heal` is the
    per-center heal (Fuchsia: _heal_fuchsia_pc)."""
    if not _pc_swap_with_heal(emu, ex, pc_map, deposit_species,
                              withdraw_species, heal):
        return False
    idx = _party_index_of(emu, withdraw_species)
    if idx is None:
        return False
    if deposit_species is not None and \
            _party_index_of(emu, deposit_species) is not None:
        return False
    if not _teach_hm_free_slot(emu, idx, hm_item, hm_move):
        return False
    if _bag_count(emu, hm_item) != 1:      # HMs not consumed
        return False
    return heal(emu, ex)


def teach_strength(emu: PyBoyEmulator, ex: Executor) -> bool:
    """HM04 STRENGTH -> Lapras (probe156, Vlad's re-decision
    2026-11-10). Vlad first chose Arcanine over Roar, but
    Arcanine's tmhm bitfield - the very data CanLearnTM
    reads - contains NO HMs at all (probe155, verified
    three ways; writing the move in was declined as
    fabrication). Verification-first is the standing rule
    from that lesson: Lapras's pokered learnset lists
    STRENGTH (data/pokemon/base_stats/lapras.asm, and the
    in-game ABLE screen agrees - probe156 media), the
    pupil is located by the SURF it already carries, its
    live moveset is read from RAM before any input, and
    the teach runs through _teach_hm_free_slot's strict
    append assertion: the ONLY legal change is STRENGTH
    appended into the free 4th slot. Live result: Lapras
    [Water Gun, Growl, Surf, Strength] - the two-HM
    carrier for the Seafoam crossing. Every other party
    member's move bytes stay identical and HM04 remains
    in the bag (HMs are never consumed). A pupil with no
    free slot, or without Surf, stops the step untouched
    - forgetting a move is the player's call."""
    if _bag_count(emu, ITEM_HM04_STRENGTH) < 1:
        return False
    idx = _party_index_of(emu, SPECIES_LAPRAS)
    if idx is None:
        return False
    from agent.state import ADDR_PARTY_MONS
    mem = emu.pb.memory
    base = ADDR_PARTY_MONS + 44 * idx
    before = [mem[base + 8 + j] for j in range(4)]
    if MOVE_STRENGTH in before:
        return True
    if MOVE_SURF not in before:      # not the Surf carrier
        return False
    if sum(1 for m in before if m) >= 4:
        return False                 # no free slot: stop
    if not _teach_hm_free_slot(emu, idx, ITEM_HM04_STRENGTH,
                               MOVE_STRENGTH):
        return False
    return _bag_count(emu, ITEM_HM04_STRENGTH) == 1


def get_good_rod(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The Good Rod: a free gift from the Fishing Guru's
    brother in Fuchsia (house map 164; the guru stands at
    (5, 3) facing right, talked to from (4, 3)). Verified
    in the bag from RAM. NOTE the Good Rod yields only
    Goldeen/Poliwag Lv10 anywhere - it cannot produce
    Dratini; see SAFARI_TARGETS."""
    if _bag_count(emu, ITEM_GOOD_ROD) >= 1:
        return True
    if not _ride_battling(emu, ex, MAP_FUCHSIA_GOOD_ROD_HOUSE):
        return False
    _walk_fighting(emu, ex, MAP_FUCHSIA_GOOD_ROD_HOUSE, (4, 3),
                   attempts=30)
    st = emu.read_state()
    if (st.x, st.y) != (4, 3):
        return False
    emu.press("right")
    emu.tick(20)
    _press10(emu, "a")
    for _ in range(300):
        if _bag_count(emu, ITEM_GOOD_ROD) >= 1:
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(25)
    _close_menus(emu)
    return _bag_count(emu, ITEM_GOOD_ROD) == 1


def _safari_grass_pair(map_id: int):
    """An adjacent pair of grass cells in a Safari area:
    both SAFARI_GRASS cells, both _safari_passable, neither
    an item-ball cell; the pair nearest the area anchor.
    Pacing between the two is the encounter engine."""
    w, h, bits = SAFARI_GRASS[map_id]

    def is_grass(x, y):
        if not (0 <= x < w and 0 <= y < h):
            return False
        i = y * w + x
        return bool(bits[i // 8] >> (i % 8) & 1)

    balls = SAFARI_BALLS_CELLS.get(map_id, frozenset())
    anchor = SAFARI_ANCHORS[map_id]
    pairs = []
    for y in range(h):
        for x in range(w):
            if not is_grass(x, y) or (x, y) in balls:
                continue
            if not _safari_passable(map_id, (x, y)):
                continue
            for dx, dy in ((1, 0), (0, 1)):
                nb = (x + dx, y + dy)
                if is_grass(*nb) and nb not in balls and \
                        _safari_passable(map_id, nb):
                    pairs.append(((x, y), nb))
    if not pairs:
        return None
    pairs.sort(key=lambda p: abs(p[0][0] - anchor[0])
               + abs(p[0][1] - anchor[1]))
    return pairs[0]


def _safari_menu_pick(emu: PyBoyEmulator, row: int,
                      col: int) -> bool:
    """Steer the Safari battle menu cursor to (row, col)
    and press A. Cursor bytes measured live (probe155):
    left column = 1, right column = 13, no wrap - the
    values _run_away's stale 15 never reaches, which is
    fine for RUN (its A lands after the presses anyway)
    but wrong for aimed picks. BALL=(0,1), BAIT=(0,13),
    ROCK=(1,1), RUN=(1,13)."""
    mem = emu.pb.memory
    for _ in range(12):
        if _advance_battle(emu) != "menu":
            return not emu.read_state().in_battle
        _steer_byte(emu, lambda: mem[ADDR_MENU_ITEM], row)
        for _ in range(14):
            if mem[ADDR_MENU_COL] == col:
                break
            _raw(emu, "right" if mem[ADDR_MENU_COL] < col
                 else "left", hold=4, settle=22)
            emu.tick(30)
        if mem[ADDR_MENU_COL] != col:
            continue
        _raw(emu, "a", hold=4, settle=22)
        emu.tick(90)
        return True
    return False


def _safari_hunt_battle(emu: PyBoyEmulator, owned: set,
                        log: list) -> dict:
    """One Safari battle under the hunt policy (probe155):
    RUN from owned species; with under 10 balls left, RUN
    from anything that is not a target (reserve); BAIT
    first - and re-bait whenever the factor lapses - for
    low-catch or fast species (the flee cut is worth more
    than the halved catch rate on paper), then BALL.
    Caps: 12 balls per target encounter, 6 per bycatch,
    3 baits. Catches are verified by the stored count and
    ridden through _catch_epilogue (a box catch files
    with no nickname prompt). OBSERVED (docs S20): in the
    live hunt every baited target fled after exactly one
    ball (10 of 11) while all six catches were unbaited
    first-ball throws - the formula above is engine-true,
    so the pattern is recorded as an open RNG-cadence /
    factor-duration question, not a mechanics claim."""
    mem = emu.pb.memory
    sp = mem[ADDR_ENEMY_SPECIES]
    lv = mem[ADDR_ENEMY_LEVEL]
    name = SAFARI_SPECIES.get(sp, f"0x{sp:02x}")
    rec = {"species": name, "level": lv, "bait": 0, "rock": 0,
           "balls": 0, "outcome": "?"}
    log.append(rec)
    target = sp in SAFARI_TARGETS
    if sp in owned:
        rec["outcome"] = "fled(owned)"
        _run_away(emu)
        return rec
    if not target and _safari_balls(emu) < 10:
        rec["outcome"] = "fled(reserve)"
        _run_away(emu)
        return rec
    cap = 12 if target else 6
    stored0 = _stored_count(emu)
    want_bait = SAFARI_CATCH_RATE.get(sp, 255) <= 45 or \
        SAFARI_SPEED.get(sp, 0) * 2 >= 150
    guard = 0
    while emu.read_state().in_battle and guard < 40:
        guard += 1
        if _safari_balls(emu) == 0:
            rec["outcome"] = "balls-out"
            break
        bait_up = mem[ADDR_SAFARI_BAIT_FACTOR] > 0
        if want_bait and not bait_up and rec["bait"] < 3:
            if not _safari_menu_pick(emu, 0, 13):   # BAIT
                break
            rec["bait"] += 1
            continue
        if rec["balls"] >= cap:
            rec["outcome"] = "fled(cap)"
            _run_away(emu)
            break
        if not _safari_menu_pick(emu, 0, 1):        # BALL
            break
        rec["balls"] += 1
        caught = False
        for _ in range(160):
            _press10(emu, "a")
            emu.tick(10)
            if _stored_count(emu) > stored0:
                caught = True
                break
            if not emu.read_state().in_battle:
                break
        if caught:
            ok = _catch_epilogue(emu, stored0 + 1)
            rec["outcome"] = "CAUGHT" if ok else "caught?"
            if ok:
                owned.add(sp)
            return rec
    if rec["outcome"] == "?":
        rec["outcome"] = "fled(enemy)" \
            if not emu.read_state().in_battle else "stuck"
    return rec


def _safari_hunt_settle(emu: PyBoyEmulator, owned: set,
                        log: list) -> None:
    """Settle after a hunt step: hunt any battle, ride any
    dialog (the PA time-up announcement included - an
    unridden dialog freezes the player solid, the probe155
    entry-1 exit bug), wait for rest."""
    for _ in range(300):
        st = emu.read_state()
        if st.in_battle:
            _safari_hunt_battle(emu, owned, log)
            continue
        if st.in_dialog:
            emu.close_dialog(max_presses=2)
            continue
        emu.tick(6)
        st2 = emu.read_state()
        if (st2.x, st2.y) == (st.x, st.y) \
                and not st2.in_dialog and not st2.in_battle:
            return


def _safari_burn_out(emu: PyBoyEmulator, owned: set,
                     log: list) -> bool:
    """Leave the zone by burning the step counter to
    ejection on the spot (catches and items are kept;
    S19's exit). Battles met while burning are hunted."""
    for i in range(900):
        if emu.read_state().map_id == MAP_SAFARI_GATE:
            return True
        emu.press("left" if i % 2 == 0 else "right")
        _safari_hunt_settle(emu, owned, log)
    return emu.read_state().map_id == MAP_SAFARI_GATE


def _safari_hunt_entry(emu: PyBoyEmulator, ex: Executor,
                       area_id: int, owned: set,
                       log: list) -> bool:
    """One paid entry hunted in one area. Transit rides
    the STOCK _safari_settle inside _sz_nav (RUN-only;
    its balls==30 assertion holds because no hunting has
    happened yet this entry) - probe155 found the hunt
    settle breaks _sz_nav's warp-cell dance, and the West
    area is effectively unreachable with current grids
    (Center's fence-maze gap is mis-modelled both in
    SAFARI_GRIDS and the world model; S20). Ends back at
    the Fuchsia PokeCenter, healed."""
    if not _ride_battling(emu, ex, MAP_SAFARI_GATE):
        return False
    money0 = emu.read_state().money
    _walk_fighting(emu, ex, MAP_SAFARI_GATE, (4, 2),
                   attempts=20)
    if (emu.read_state().x, emu.read_state().y) != (4, 2):
        return False
    emu.press("right")
    emu.tick(20)
    _press10(emu, "a")
    paid = False
    for _ in range(200):
        if _safari_balls(emu) == 30 and \
                _event_flag(emu, EV_IN_SAFARI_ZONE):
            paid = True
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(25)
    if not paid or emu.read_state().money != money0 - 500:
        return False
    # ride the clerk's trailing text fully clear BEFORE
    # any direction press (direction presses mid-script
    # wedge it - probe155)
    for _ in range(80):
        if not emu.read_state().in_dialog:
            break
        _press10(emu, "a")
        emu.tick(30)
    for _ in range(10):
        if emu.read_state().map_id == MAP_SZ_CENTER:
            break
        emu.press("up")
        _explore_settle(emu)
        _safari_settle(emu, [])
    if emu.read_state().map_id != MAP_SZ_CENTER:
        return False
    pair = _safari_grass_pair(area_id)
    if pair is None:
        return False
    if not _sz_nav(emu, ex, area_id, pair[0], [],
                   budget_guard=10):
        return False
    toggle = False
    while True:
        st = emu.read_state()
        if st.map_id == MAP_SAFARI_GATE:
            break
        if st.map_id != area_id:
            _safari_burn_out(emu, owned, log)
            break
        if SAFARI_TARGETS <= owned or _safari_balls(emu) == 0:
            _safari_burn_out(emu, owned, log)
            break
        if _safari_steps(emu) <= 12:
            _safari_burn_out(emu, owned, log)
            break
        tgt = pair[1] if toggle else pair[0]
        toggle = not toggle
        for d, (dx, dy) in _SZ_DIRS.items():
            cur = (emu.read_state().x, emu.read_state().y)
            if (cur[0] + dx, cur[1] + dy) == tgt:
                emu.press(d)
                break
        _safari_hunt_settle(emu, owned, log)
    if emu.read_state().map_id != MAP_SAFARI_GATE:
        return False
    for _ in range(12):
        emu.press("down")
        _explore_settle(emu)
        _safari_settle(emu, [])
        if emu.read_state().map_id != MAP_SAFARI_GATE:
            break
    if emu.read_state().map_id == MAP_SAFARI_GATE:
        return False
    if not _ride_battling(emu, ex, MAP_FUCHSIA_PC):
        return False
    return _heal_fuchsia_pc(emu, ex)


def safari_rare_hunt(emu: PyBoyEmulator, ex: Executor,
                     plan: list, log: list = None) -> bool:
    """Job B (probe155, Vlad's policy: hunt the rares).
    Up to four paid entries (the caller's plan, area names
    from SAFARI_AREAS_BY_NAME), throwing Safari Balls only
    at the five SAFARI_TARGETS and at other not-yet-owned
    species, fleeing everything already owned. Prep makes
    bag room for the rods by depositing TM24/TM29/TM46 at
    the Fuchsia PC (the 20-stack cap) and collects the
    Good Rod. The Super Rod detour was NOT made: the
    brief's hop chain Fuchsia -> R15 -> R14 -> R13 -> R12
    is one-way-locked in reverse (that road is a one-way
    descent from Lavender - ledges and one-way strips at
    R13/R14, engine-verified in probe155), there is no
    Bicycle for Cycling Road, and the only feasible route
    (Surf via R19/R21 + Diglett's Cave + R11 + the R12
    gate) needs Surf-step machinery the executor does not
    have. Dratini therefore stays unfished this leg; the
    decision is Vlad's (docs S20). Job C is likewise not
    performed: Arcanine's pokered learnset (base_stats
    tmhm bitfield - the very data CanLearnTM reads)
    contains NO HMs at all, so HM04 cannot be taught to
    it by the engine; writing the move into its struct
    was declined as fabrication. The live hunt caught
    PARAS, DODUO, EXEGGCUTE, NIDORINO (East), RHYHORN
    (North) and KANGASKHAN (East); dex 16 -> 22."""
    if log is None:
        log = []
    if len(plan) > 4:
        return False
    if emu.read_state().map_id != MAP_FUCHSIA_PC:
        if not _ride_battling(emu, ex, MAP_FUCHSIA_PC):
            return False
    # bag room for the rods (20-stack cap): the leg
    # deposited TM24/TM29/TM46 into PC item storage
    if len(_bag_items(emu)) >= 18:
        for tm in (ITEM_TM24, ITEM_TM29, ITEM_TM46):
            if _bag_count(emu, tm) >= 1:
                if not deposit_item(emu, ex, MAP_FUCHSIA_PC, tm):
                    return False
    if not get_good_rod(emu, ex):
        return False
    if not _ride_battling(emu, ex, MAP_FUCHSIA_PC):
        return False
    if not _heal_fuchsia_pc(emu, ex):
        return False
    owned = set()
    st = emu.read_state()
    for mon in st.party:
        owned.add(mon.species_id)
    owned.update(_box_species(emu))
    for name in plan:
        area_id = SAFARI_AREAS_BY_NAME[name]
        if not _safari_hunt_entry(emu, ex, area_id, owned, log):
            return False
        if SAFARI_TARGETS <= owned:
            break
    return True




# ======================================================================
# THE GRAND TOUR (probe157, docs section 22)
# Surf machinery, then the tour itself: Cinnabar (the Mansion
# Secret Key, Blaine), Pallet and the Diglett's Cave ride to the
# Super Rod, and Dratini fishing in the Safari Zone. Engine facts
# this rides on, compiled from pokered and verified live:
# * Boarding is the party-menu SURF field move - Gen 1 has NO
#   A-button water prompt. The tile in front must pass
#   IsNextTileShoreOrWater (its sampled tile is $14/$32/$48);
#   the engine then steps the player onto the water itself and
#   sets wWalkBikeSurfState (0xD700) to 2.
# * While surfing, stepping onto a land-passable cell
#   auto-disembarks (0xD700 -> 0); onto $14/$48 (and $32 off the
#   SHIP_PORT tileset) you keep surfing. world.py's surf layer
#   compiles the cells; this section executes them one
#   RAM-verified step at a time, fleeing wilds and fighting
#   whatever trainer intercepts (R19/R20/R21's swimmers stand
#   IN the water and sight-fight surfers).
# * Switch/gate interiors (Pokemon Mansion, Cinnabar Gym) swap
#   whole 2x2 blocks via ReplaceTileBlock (b = Y block, c = X
#   block); passability is per cell from the swapped block's
#   sampled tiles, so navigation runs on a live overlay read
#   from the switch/gate event flags.
# ======================================================================

ADDR_SURF_STATE = 0xD700          # wWalkBikeSurfState (2 = surfing)
ADDR_BATTLE_KIND = 0xD057         # wIsInBattle (1 wild, 2 trainer)
ADDR_BATTLE_TYPE = 0xD05A         # wBattleType (2 = Safari)
ADDR_MENU_ITEM = 0xCC26           # wCurrentMenuItem
ADDR_TILE_IN_FRONT = 0xCFC6       # wTileInFrontOfPlayer
ITEM_SECRET_KEY = 0x2B
ITEM_BIKE_VOUCHER = 0x2D
ITEM_SUPER_ROD = 0x4E
ITEM_TM38 = 0xEE
EV_MANSION_SWITCH_ON = 632
EV_BEAT_BLAINE = 665
EV_GOT_TM38 = 664
# Quiz gate index i (1..6, the hidden events' low nibble)
# sets event 680 + i: the quiz script's FlagAction bit is
# hGymGateIndex counted from EVENT_CINNABAR_GYM_GATE0_UNLOCKED
# (680) - proven live in probe158m2/m4, where machine 1 set
# 681, not 680. The pre-158 table (680..685) was shifted one
# low and the whole gym deadlocked on it.
EV_GYM_GATES = (681, 682, 683, 684, 685, 686)

_DIR_FROM_DELTA = {(0, -1): "up", (0, 1): "down",
                   (-1, 0): "left", (1, 0): "right"}


def surfing(emu: PyBoyEmulator) -> bool:
    return emu.pb.memory[ADDR_SURF_STATE] == 2


def _surf_carrier(emu: PyBoyEmulator):
    """(party index, party-submenu index) of the Surf carrier,
    read live: the first party mon knowing SURF (move id 57);
    the submenu index counts its field moves that precede
    Surf in its own slot order (_field_rank)."""
    st = emu.read_state()
    for i, mon in enumerate(st.party):
        if 57 in mon.moves:
            return i, _field_rank(emu, i, 57)
    return None


def _surf_battle_clear(emu: PyBoyEmulator) -> bool:
    """Resolve whatever battle the last step triggered: flee
    wilds, fight trainers (a swimmer interception cannot be
    fled). Returns False if the party blacked out."""
    for _ in range(40):
        st = emu.read_state()
        if not st.in_battle:
            return any(m.hp > 0 for m in st.party)
        kind = emu.pb.memory[ADDR_BATTLE_KIND]
        _drive_battle_typed(emu, flee_wild=(kind == 1))
    return not emu.read_state().in_battle


def _surf_step(emu: PyBoyEmulator, direction: str) -> None:
    """One press in surf context + settle + battle resolution.
    A fled battle can leave its (empty) text box standing,
    which swallows every later direction press - drain it
    before returning (probe157: wedged at R20 (87, 4) with all
    four directions refused and in_dialog stuck True)."""
    emu.press(direction)
    _explore_settle(emu)
    if emu.read_state().in_battle:
        _surf_battle_clear(emu)
        _explore_settle(emu)
    if emu.read_state().in_dialog:
        emu.close_dialog(max_presses=6)
        _explore_settle(emu)


def _refusal_recover(emu: PyBoyEmulator,
                     flee_wild: bool = True) -> bool:
    """Un-wedge a refused step. A step that changes nothing and
    starts no battle can be a TRAINER SIGHT-LOCK: the trainer
    has spotted the player and the engagement script stalls
    while directions keep being pressed (probe157: R19's
    Swimmer4 at (9, 13) froze the walk back to the beach in
    all four directions until an A tap let the battle fire).
    Idle with neutral input, tap A, resolve any battle that
    starts. Returns True if a battle was found and cleared."""
    emu.tick(90)
    for _ in range(3):
        if emu.read_state().in_battle:
            _drive_battle_typed(emu, flee_wild=flee_wild)
            _explore_settle(emu)
            return True
        _press10(emu, "a")
        emu.tick(60)
    if emu.read_state().in_battle:
        _drive_battle_typed(emu, flee_wild=flee_wild)
        _explore_settle(emu)
        return True
    return False


def _surf_board(emu: PyBoyEmulator, direction: str) -> bool:
    """Board: face `direction` (a bump press turns the player),
    then run the party-menu Surf macro for the live carrier.
    Success is the RAM surf byte, not the macro's say-so."""
    carrier = _surf_carrier(emu)
    if carrier is None:
        return False
    emu.press(direction)
    emu.tick(10)
    front = emu.pb.memory[ADDR_TILE_IN_FRONT]
    if front not in (0x14, 0x32, 0x48):
        return False
    idx, sub = carrier
    if not emu.field_surf(idx, sub):
        return False
    for _ in range(30):
        if surfing(emu):
            break
        emu.tick(10)
    _explore_settle(emu)
    if emu.read_state().in_battle:
        _surf_battle_clear(emu)
    return surfing(emu)


def _surf_walk_to(emu: PyBoyEmulator, ex: Executor, map_id: int,
                  goal, blocked0=frozenset(),
                  max_replans: int = 80) -> bool:
    """Walk/surf within one map to `goal` over world.surf_path,
    one RAM-verified cell at a time. Boards when the next cell
    is water-only, rides the engine's auto-disembark onto land,
    and replans around refused cells (learned walls - a swimmer
    standing on the planned line is the common case). Stepping
    onto the goal may trigger its warp: a map change on the
    goal step is success."""
    world = ex.world
    gm = world.maps[map_id]
    blocked = set(blocked0)
    for _ in range(max_replans):
        st = emu.read_state()
        if st.map_id != map_id:
            return False
        if (st.x, st.y) == goal:
            return True
        path = world.surf_path(map_id, (st.x, st.y), goal,
                               blocked=frozenset(blocked))
        if not path or len(path) < 2:
            return (st.x, st.y) == goal
        refused = False
        for nxt in path[1:]:
            st = emu.read_state()
            if st.map_id != map_id:
                return False
            dx, dy = nxt[0] - st.x, nxt[1] - st.y
            dist = abs(dx) + abs(dy)
            if dist == 2:      # ledge hop: jump direction
                dx, dy = dx // 2, dy // 2
            elif dist != 1:
                refused = True
                break
            direction = _DIR_FROM_DELTA[(dx, dy)]
            if not surfing(emu) and not gm.tile_walkable(*nxt) \
                    and nxt in gm.surf_cells:
                _surf_board(emu, direction)
            else:
                _surf_step(emu, direction)
            st = emu.read_state()
            if st.map_id != map_id:
                return nxt == goal
            if (st.x, st.y) != nxt:
                # sight-lock? recover, then retry the step once
                _refusal_recover(emu)
                st = emu.read_state()
                if st.map_id != map_id:
                    return nxt == goal
                if (st.x, st.y) != nxt:
                    if not surfing(emu) and not gm.tile_walkable(*nxt) \
                            and nxt in gm.surf_cells:
                        _surf_board(emu, direction)
                    else:
                        _surf_step(emu, direction)
                    st = emu.read_state()
                    if st.map_id != map_id:
                        return nxt == goal
                if (st.x, st.y) != nxt:
                    blocked.add(nxt)
                    refused = True
                    break
            if (st.x, st.y) == goal:
                return True
        if not refused:
            continue
    st = emu.read_state()
    return st.map_id == map_id and (st.x, st.y) == goal


def surf_to(emu: PyBoyEmulator, ex: Executor, dst_map: int,
            dst_xy=None) -> bool:
    """Travel between maps on foot + water: route() legs are
    followed via connection crossings (world.surf_crossings);
    each leg walks (surf_path) to the best exit cell, crosses,
    and continues. `dst_xy`, when given, is a cell in dst_map
    to end on (e.g. a door warp - stepping on it fires the
    warp, which is success). Warp legs are NOT handled: the
    tour enters buildings by walking onto their doors within
    the arrival map."""
    world = ex.world
    caps = emu.read_state().capabilities()
    if "surf" not in caps:
        return False
    for _leg_no in range(16):
        st = emu.read_state()
        if st.map_id == dst_map:
            if dst_xy is None:
                return True
            return _surf_walk_to(emu, ex, dst_map, dst_xy)
        route = world.route(st.map_id, dst_map, caps)
        if route is None or not route.legs:
            return False
        leg = route.legs[0]
        if not leg.via.startswith("connection"):
            return False
        direction = leg.via.split()[1]
        cands = []
        for exit_cell, dm, _arr in world.surf_crossings(
                st.map_id, direction):
            if dm != leg.dst:
                continue
            p = world.surf_path(st.map_id, (st.x, st.y), exit_cell)
            if p is not None:
                cands.append((len(p), exit_cell))
        if not cands:
            return False
        cands.sort()
        if not _surf_walk_to(emu, ex, st.map_id, cands[0][1]):
            return False
        step_dir = {"north": "up", "south": "down",
                    "east": "right", "west": "left"}[direction]
        if not surfing(emu):
            # The exit cell can be a LAND shore cell (Cinnabar's
            # north coast: the crossing water is R21's first
            # row, so surf_path walks to the beach and stops).
            # Board first, then step (probe158n/n2).
            _surf_board(emu, step_dir)
        _surf_step(emu, step_dir)
        if emu.read_state().map_id != leg.dst:
            return False
    return emu.read_state().map_id == dst_map


# -- Seafoam: boulder machinery + the fuller crossing --------
# (probe158). Boulder facts, all RAM-verified live: a
# boulder's cell reads from wSpriteStateData2 - sprite k is
# object k, cell = (byte[+5] - 4, byte[+4] - 4); Strength
# activation is wSpriteStateData1's bit at 0xD728 bit 0 and
# must be re-checked before every push; a push takes TWO
# direction contacts (the first only sets the tried-flag);
# a hole push must NEVER get a third press (the player
# would step onto the hole and dungeon-warp). Each floor's
# pair of falls sets one cascade event pair; the B3F pool
# currents run until the B2F pair (SEAFOAM3) is down.
SEAFOAM_CASCADE_EVENTS = (1294, 1295, 2496, 2497, 2504, 2505)
ADDR_BOULDER_SPRITES = 0xC200
ADDR_STRENGTH_BIT = 0xD728


def _boulder_cell(emu: PyBoyEmulator, sprite_k: int):
    mem = emu.pb.memory
    base = ADDR_BOULDER_SPRITES + 16 * sprite_k
    return (mem[base + 5] - 4, mem[base + 4] - 4)


def _strength_carrier(emu: PyBoyEmulator):
    """(party index, submenu rank) of the Strength carrier
    (move id 70), read live like _surf_carrier."""
    st = emu.read_state()
    for i, mon in enumerate(st.party):
        if 70 in mon.moves:
            return i, _field_rank(emu, i, 70)
    return None


def _strength_active(emu: PyBoyEmulator) -> bool:
    return bool(emu.pb.memory[ADDR_STRENGTH_BIT] & 1)


def _activate_strength(emu: PyBoyEmulator) -> bool:
    if _strength_active(emu):
        return True
    carrier = _strength_carrier(emu)
    if carrier is None:
        return False
    emu._field_move(*carrier)
    _explore_settle(emu)
    return _strength_active(emu)


def seafoam_crossing(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The fuller Seafoam crossing (probe158): Fuchsia PC ->
    surf R19/R20 -> the island's north plateau -> WEST door
    -> the boulder cascade (1F pair, B1F pair, B2F pair -
    the last one by the B4F long way) -> board the stilled
    B3F pool at the peninsula ($15 shore; $05 shores refuse
    boarding under the CAVERN pair rule) -> climb out east
    -> east door -> surf to Cinnabar -> heal. Articuno is
    never approached: the B4F legs stay in the lagoon and
    the north/east land, out of its (6, 1) island's sight.
    All wilds are fled by the walker; every fall is verified
    by its cascade event flag."""
    world = ex.world
    fuchsia = world.map_id("FuchsiaCity")
    r19 = world.map_id("Route19")
    r20 = world.map_id("Route20")
    sf1 = world.map_id("SeafoamIslands1F")
    b1 = world.map_id("SeafoamIslandsB1F")
    b2 = world.map_id("SeafoamIslandsB2F")
    b3 = world.map_id("SeafoamIslandsB3F")
    b4 = world.map_id("SeafoamIslandsB4F")
    cin = world.map_id("CinnabarIsland")
    cpc = world.map_id("CinnabarPokecenter")
    holes = {sf1: {(17, 6), (24, 6)}, b1: {(18, 6), (23, 6)},
             b2: {(19, 6), (22, 6)}, b3: {(3, 16), (6, 16)},
             b4: set()}

    def leg(map_id, goal, want_map=None, want_cells=None,
            blocked=frozenset()):
        blk = frozenset(blocked) | holes.get(map_id, set())
        if not _surf_walk_to(emu, ex, map_id, goal,
                             blocked0=blk):
            return False
        st = emu.read_state()
        if want_map is not None and st.map_id != want_map:
            return False
        if want_cells is not None \
                and (st.x, st.y) not in want_cells:
            return False
        return True

    def push(map_id, k, stance, direction, expect):
        """One plain push: walk to the stance (boulder cell
        blocked), Strength on, contact until the sprite's
        RAM cell matches `expect`."""
        before = _boulder_cell(emu, k)
        blk = frozenset({before}) | holes.get(map_id, set())
        if not _surf_walk_to(emu, ex, map_id, stance,
                             blocked0=blk):
            return False
        st = emu.read_state()
        if (st.x, st.y) != stance:
            return False
        if not _activate_strength(emu):
            return False
        for _try in range(4):
            emu.press(direction)
            _explore_settle(emu)
            if emu.read_state().in_battle:
                _surf_battle_clear(emu)
            if _boulder_cell(emu, k) == expect:
                return True
        return False

    def hole_push(map_id, k, stance, direction, hole_cell,
                  event_id):
        """The fall push: at most two contacts, then the
        cascade event flag is the verdict."""
        before = _boulder_cell(emu, k)
        blk = frozenset({before, hole_cell}) \
            | holes.get(map_id, set())
        if not _surf_walk_to(emu, ex, map_id, stance,
                             blocked0=blk):
            return False
        st = emu.read_state()
        if (st.x, st.y) != stance:
            return False
        if not _activate_strength(emu):
            return False
        for _i in range(2):
            emu.press(direction)
            _explore_settle(emu)
            emu.tick(60)
            if emu.read_state().in_battle:
                _surf_battle_clear(emu)
            if _event_flag(emu, event_id) \
                    or _boulder_cell(emu, k) != before:
                break
        return emu.read_state().map_id == map_id \
            and _event_flag(emu, event_id)

    # -- Fuchsia PC -> R19 -> R20 -> plateau -> west door ----
    if emu.read_state().map_id == MAP_FUCHSIA_PC:
        for door in ((3, 7), (4, 7)):
            _walk_fighting(emu, ex, MAP_FUCHSIA_PC, door,
                           attempts=15)
            for _ in range(6):
                emu.press("down")
                _explore_settle(emu)
                if emu.read_state().map_id != MAP_FUCHSIA_PC:
                    break
            if emu.read_state().map_id != MAP_FUCHSIA_PC:
                break
    if emu.read_state().map_id != fuchsia:
        return False
    if not surf_to(emu, ex, r19):
        return False
    if not surf_to(emu, ex, r20):
        return False
    if not leg(r20, (48, 5), sf1, {(4, 17), (5, 17)}):
        return False
    # -- 1F pair ------------------------------------------------
    if _boulder_cell(emu, 1) != (18, 10) \
            or _boulder_cell(emu, 2) != (26, 7):
        return False
    for stance, d, expect in (
            ((18, 11), "up", (18, 9)), ((18, 10), "up", (18, 8)),
            ((18, 9), "up", (18, 7)), ((18, 8), "up", (18, 6))):
        if not push(sf1, 1, stance, d, expect):
            return False
    if not hole_push(sf1, 1, (19, 6), "left", (17, 6), 1294):
        return False
    for stance, d, expect in (
            ((26, 8), "up", (26, 6)), ((27, 6), "left", (25, 6))):
        if not push(sf1, 2, stance, d, expect):
            return False
    if not hole_push(sf1, 2, (26, 6), "left", (24, 6), 1295):
        return False
    # -- B1F pair ------------------------------------------------
    if not leg(sf1, (7, 5), b1, {(7, 5)}):
        return False
    if _boulder_cell(emu, 1) != (17, 6):
        return False
    if not hole_push(b1, 1, (16, 6), "right", (18, 6), 2496):
        return False
    if not leg(b1, (7, 5), sf1, {(7, 5)}):
        return False
    if not leg(sf1, (25, 3), b1, {(25, 3)}):
        return False
    if _boulder_cell(emu, 2) != (22, 6):
        return False
    if not hole_push(b1, 2, (21, 6), "right", (23, 6), 2497):
        return False
    # -- B2F boulder A -------------------------------------------
    if not leg(b1, (25, 3), sf1, {(25, 3)}):
        return False
    if not leg(sf1, (7, 5), b1, {(7, 5)}):
        return False
    if not leg(b1, (4, 2), b2, {(5, 3)}):
        return False
    if _boulder_cell(emu, 1) != (18, 6):
        return False
    if not hole_push(b2, 1, (17, 6), "right", (19, 6), 2504):
        return False
    # -- B2F boulder B: the long way (B4F lagoon ride) ------------
    if not leg(b2, (5, 3), b1, {(4, 2)}):
        return False
    if not leg(b1, (13, 7), b2, {(13, 7)}):
        return False
    if not leg(b2, (5, 13), b3, {(5, 12)}):
        return False
    if not leg(b3, (15, 7), b3, {(15, 7)}):
        return False
    if not _surf_board(emu, "down"):
        return False
    emu.tick(300)
    st = emu.read_state()
    if st.map_id != b4 or not surfing(emu):
        return False
    if not leg(b4, (23, 5), b4, {(23, 5)}):
        return False
    if not leg(b4, (25, 4), b3, {(25, 4)}):
        return False
    if not leg(b3, (25, 3), b2, {(25, 3)}):
        return False
    if _boulder_cell(emu, 2) != (23, 6):
        return False
    if not hole_push(b2, 2, (24, 6), "left", (22, 6), 2505):
        return False
    if not all(_event_flag(emu, ev)
               for ev in SEAFOAM_CASCADE_EVENTS):
        return False
    # -- the calm pool, out east, Cinnabar -------------------------
    if not leg(b2, (25, 3), b3, {(25, 3)}):
        return False
    if not leg(b3, (25, 4), b4, {(25, 4)}):
        return False
    if not leg(b4, (11, 7), b3, {(8, 6)}):
        return False
    if not leg(b3, (15, 7), b3, {(15, 7)}):
        return False
    if not _surf_board(emu, "down"):
        return False
    st = emu.read_state()
    if not surfing(emu) or (st.x, st.y) != (15, 8):
        return False
    emu.tick(300)
    st = emu.read_state()
    if st.map_id != b3 or (st.x, st.y) != (15, 8):
        return False          # the current is still running
    if not leg(b3, (23, 9), b3, {(23, 9)}):
        return False
    if not leg(b3, (25, 14), b2, {(25, 14)}):
        return False
    if not leg(b2, (25, 11), b1, {(25, 11)}):
        return False
    if not leg(b1, (23, 15), sf1, {(23, 15)}):
        return False
    if not _surf_walk_to(emu, ex, sf1, (26, 17),
                         blocked0=frozenset(holes[sf1])):
        return False
    st = emu.read_state()
    if st.map_id == sf1:
        if (st.x, st.y) not in {(26, 17), (27, 17)}:
            return False
        for _ in range(10):
            emu.press("down")
            _explore_settle(emu)
            if emu.read_state().map_id != sf1:
                break
    st = emu.read_state()
    if st.map_id != r20 or (st.x, st.y) not in {(58, 9),
                                                (58, 10),
                                                (58, 11)}:
        return False
    if not surf_to(emu, ex, cin):
        return False
    if emu.read_state().map_id != cin:
        return False
    if not _surf_walk_to(emu, ex, cin, (11, 11)):
        return False
    if emu.read_state().map_id != cpc:
        return False
    if not _heal_at_pc(emu, ex, cpc):
        return False
    return emu.read_state().map_id == cin


# -- switch/gate interiors -------------------------------------------
# A door occupies a 2x2 block; the open/closed patterns below are
# the blocks' bottom-left sampled tiles against the tileset's
# passable list, computed from the blocksets (probe157):
# FACILITY floor $0E opens all four cells; the gate blocks leave
# the corridor-side lane sampled as floor even when "closed"
# ($2D top lane, $54 bottom lane, $5F left lane) - you can walk
# the lane, you just cannot cross. In the GYM tileset the floor
# block $0E's TOP row samples $17 (wall), so an open gym gate is
# only its bottom lane; GYM $54 is a diagonal (its passable
# cells are (1,0) and (0,1)) and GYM $5F its right lane.
_DOOR_OPEN = {
    "FACILITY": frozenset({(0, 0), (1, 0), (0, 1), (1, 1)}),
    "GYM": frozenset({(0, 1), (1, 1)}),
}
_DOOR_CLOSED = {
    ("FACILITY", 0x2D): frozenset({(0, 0), (1, 0)}),
    ("FACILITY", 0x54): frozenset({(0, 1), (1, 1)}),
    ("FACILITY", 0x5F): frozenset({(0, 0), (0, 1)}),
    ("GYM", 0x54): frozenset({(1, 0), (0, 1)}),
    ("GYM", 0x5F): frozenset({(1, 0), (1, 1)}),
}
# Pokemon Mansion: ONE switch (EV_MANSION_SWITCH_ON) drives every
# floor's doors. Entries: (block (bx, by), gate block id,
# open_when_switch_on). Block coords are (X, Y) in block units -
# ReplaceTileBlock takes b = Y, c = X, and the scripts' `lb bc`
# pairs read (Y, X); the pairs below are already transposed into
# (bx, by). Verified against each floor's script (probe157).
MANSION_DOORS = {
    165: [((12, 6), 0x2D, False), ((8, 3), 0x2D, True),
          ((10, 8), 0x2D, True), ((13, 13), 0x2D, True)],
    214: [((4, 2), 0x5F, False), ((9, 4), 0x54, True),
          ((3, 11), 0x5F, True)],
    215: [((7, 2), 0x5F, False), ((7, 5), 0x5F, True)],
    216: [((13, 8), 0x2D, False), ((6, 11), 0x5F, False),
          ((4, 3), 0x5F, True), ((8, 8), 0x54, True)],
}
# Statue stances: stand here, face UP, press A, answer YES.
MANSION_STATUES = {165: (2, 6), 214: (2, 12), 215: (10, 6),
                   216: (20, 4)}
# B1F has a SECOND statue (hidden events list two for the
# basement): its stance is the basement route's first toggle
# (probe158: entry -> (18,26) OFF -> (20,4) ON -> key).
MANSION_STATUES_ALT = {216: (18, 26)}
# Cinnabar Gym: (block (bx, by), gate block id, gate event) in
# CinnabarGymGateCoords order; quiz gate g (1-based) opens
# entry g-1 and sets EV_GYM_GATES[g-1] = 680 + g (probe158).
CINNABAR_GYM_DOORS = [
    ((9, 3), 0x54, 681), ((6, 3), 0x54, 682), ((6, 6), 0x54, 683),
    ((3, 8), 0x5F, 684), ((2, 6), 0x54, 685), ((2, 3), 0x54, 686),
]
# The quiz machines, in the order their rooms chain from the
# door: (machine tile, stance tile south of it, gate index
# 0-based, correct YesNo item 0=YES 1=NO). Answers compiled from
# CinnabarGymQuizCoords + the hidden events' stored nibbles.
CINNABAR_QUIZ = [
    ((15, 7), (15, 8), 0, 0),    # CATERPIE->BUTTERFREE? YES
    ((10, 1), (10, 2), 1, 1),    # 9 certified badges? NO
    ((9, 7), (9, 8), 2, 1),      # POLIWAG evolves 3 times? NO
    ((9, 13), (9, 14), 3, 1),    # thunder vs ground? NO
    ((1, 13), (1, 14), 4, 0),    # same kind/level identical? YES
    ((1, 7), (1, 8), 5, 1),      # TM28 = TOMBSTONER? NO
]


def _door_overlay(world, map_id: int, doors, is_open):
    """passable(x, y) for a switch/gate interior: the base grid,
    with each door block's cells decided by the block's CURRENT
    state - the tileset's open pattern or the gate block's own
    closed pattern. `doors` entries are (block_xy, gate_block_id,
    key); is_open(key) is consulted live at every call."""
    gm = world.maps[map_id]
    cell_state = {}
    for (bx, by), gid, key in doors:
        for dx in (0, 1):
            for dy in (0, 1):
                cell_state[(2 * bx + dx, 2 * by + dy)] = \
                    (gid, key, (dx, dy))

    def passable(x: int, y: int) -> bool:
        hit = cell_state.get((x, y))
        if hit is None:
            return gm.tile_walkable(x, y)
        gid, key, off = hit
        if is_open(key):
            return off in _DOOR_OPEN[gm.tileset]
        return off in _DOOR_CLOSED[(gm.tileset, gid)]

    return passable


def _overlay_path(world, map_id: int, start, goal, passable,
                  blocked=frozenset()):
    """BFS over a live passability overlay. Warp tiles are
    destinations, not corridors (the tile_path rule)."""
    from collections import deque
    gm = world.maps[map_id]
    warps = {(w.x, w.y) for w in gm.warps}
    if start == goal:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            n = (cur[0] + dx, cur[1] + dy)
            if n in prev or n in blocked:
                continue
            if not gm.in_bounds(*n):
                continue
            if n in warps and n != goal:
                continue
            if not passable(*n):
                continue
            prev[n] = cur
            if n == goal:
                path = [n]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            q.append(n)
    return None


def _overlay_nav(emu: PyBoyEmulator, ex: Executor, map_id: int,
                 goal, passable, flee_wild: bool = True,
                 max_replans: int = 80) -> bool:
    """Step to `goal` inside one map over a live passability
    overlay, RAM-verified per cell with learned walls on
    refusal (an NPC sprite on a grid-open cell is the usual
    cause). Battles go through _drive_battle_typed. Stepping
    onto the goal may fire its warp - a map change on the goal
    step is success."""
    world = ex.world
    blocked = set()
    for _ in range(max_replans):
        st = emu.read_state()
        if st.map_id != map_id:
            return False
        if (st.x, st.y) == goal:
            return True
        path = _overlay_path(world, map_id, (st.x, st.y), goal,
                             passable, blocked=frozenset(blocked))
        if not path or len(path) < 2:
            return (st.x, st.y) == goal
        refused = False
        for nxt in path[1:]:
            st = emu.read_state()
            if st.map_id != map_id:
                return False
            d = _DIR_FROM_DELTA.get((nxt[0] - st.x, nxt[1] - st.y))
            if d is None:
                refused = True
                break
            emu.press(d)
            _explore_settle(emu)
            if emu.read_state().in_battle:
                _drive_battle_typed(emu, flee_wild=flee_wild)
                _explore_settle(emu)
            if emu.read_state().in_dialog:
                emu.close_dialog(max_presses=6)
                _explore_settle(emu)
            st = emu.read_state()
            if st.map_id != map_id:
                return nxt == goal
            if (st.x, st.y) != nxt:
                # sight-lock? recover, then retry the step once
                _refusal_recover(emu, flee_wild=flee_wild)
                st = emu.read_state()
                if st.map_id != map_id:
                    return nxt == goal
                if (st.x, st.y) != nxt:
                    emu.press(d)
                    _explore_settle(emu)
                    if emu.read_state().in_battle:
                        _drive_battle_typed(emu,
                                            flee_wild=flee_wild)
                        _explore_settle(emu)
                    st = emu.read_state()
                    if st.map_id != map_id:
                        return nxt == goal
                if (st.x, st.y) != nxt:
                    blocked.add(nxt)
                    refused = True
                    break
            if (st.x, st.y) == goal:
                return True
        if not refused:
            continue
    st = emu.read_state()
    return st.map_id == map_id and (st.x, st.y) == goal


def _yesno_open(emu: PyBoyEmulator) -> bool:
    """Is a Yes/No box on screen? The box's top-left border tile
    (0x79 - the same tile the dialog check reads at (0, 12))
    lands at tilemap (13, 0); measured live in probe157."""
    return emu.pb.memory[0xC3A0 + 13] == 0x79


def _yesno_pick(emu: PyBoyEmulator, want: int) -> bool:
    """Answer an on-screen Yes/No prompt: advance text with A
    until the box opens, steer wCurrentMenuItem to `want`
    (0 = YES, 1 = NO), confirm with A. The caller verifies the
    OUTCOME (event flag / gate / battle) itself."""
    for _ in range(60):
        if emu.read_state().in_battle:
            return False
        if _yesno_open(emu):
            cur = emu.pb.memory[ADDR_MENU_ITEM]
            if cur == want:
                _press10(emu, "a")
                emu.tick(30)
                return True
            _press10(emu, "down" if want > cur else "up")
            emu.tick(15)
            continue
        if emu.read_state().in_dialog:
            _press10(emu, "a")
            emu.tick(15)
            continue
        emu.tick(10)
    return False


def _drain_dialog(emu: PyBoyEmulator, rounds: int = 80) -> None:
    for _ in range(rounds):
        if not emu.read_state().in_dialog:
            return
        _press10(emu, "a")
        emu.tick(15)


def _heal_at_pc(emu: PyBoyEmulator, ex: Executor,
                pc_map: int) -> bool:
    """Heal inside a standard-layout Pokecenter and walk back
    out to the town. The nurse pattern proven at Fuchsia
    (probe154): stance (3,3)/(4,3), face up, ride the dialog,
    verify the WHOLE party at full HP from RAM."""
    if emu.read_state().map_id != pc_map:
        return False
    for stance in ((3, 3), (4, 3)):
        _walk_fighting(emu, ex, pc_map, stance, attempts=20)
        st = emu.read_state()
        if (st.x, st.y) != stance:
            continue
        emu.press("up")
        emu.tick(20)
        _press10(emu, "a")
        quiet = 0
        for _ in range(300):
            st = emu.read_state()
            if st.in_dialog:
                quiet = 0
                _press10(emu, "a")
            else:
                quiet += 1
                if quiet >= 6 and all(
                        m.hp == m.max_hp for m in st.party):
                    break
            emu.tick(25)
        if all(m.hp == m.max_hp for m in emu.read_state().party):
            break
    if not all(m.hp == m.max_hp for m in emu.read_state().party):
        return False
    for door in ((3, 7), (4, 7)):
        try:
            ex._walk_to(pc_map, door)
        except StuckError:
            continue
        for _ in range(6):
            emu.press("down")
            _explore_settle(emu)
            if emu.read_state().map_id != pc_map:
                return True
    return emu.read_state().map_id != pc_map


def _mansion_overlay(emu: PyBoyEmulator, world, map_id: int):
    return _door_overlay(
        world, map_id, MANSION_DOORS[map_id],
        lambda key: _event_flag(emu, EV_MANSION_SWITCH_ON) == key)


def _mansion_press_statue(emu: PyBoyEmulator, ex: Executor,
                          map_id: int, stance=None) -> bool:
    """Walk to a statue stance, face UP, press it (YesNo ->
    YES), and verify the switch flag actually flipped.
    _yesno_pick's own return is NOT trusted: its box detector
    can miss the open/close window and report False after a
    successful toggle (probe158l2, live) - the flag is the
    only verdict."""
    world = ex.world
    if stance is None:
        stance = MANSION_STATUES[map_id]
    before = _event_flag(emu, EV_MANSION_SWITCH_ON)
    if not _overlay_nav(emu, ex, map_id, stance,
                        _mansion_overlay(emu, world, map_id)):
        return False
    emu.press("up")
    emu.tick(15)
    _press10(emu, "a")
    _yesno_pick(emu, 0)
    _drain_dialog(emu)
    return _event_flag(emu, EV_MANSION_SWITCH_ON) != before


def _mansion_go(emu: PyBoyEmulator, ex: Executor, map_id: int,
                goal) -> bool:
    """Overlay-nav to `goal` in the Mansion, toggling this
    floor's statue switch once if the current state seals the
    way (the switch is global; either state may be the open
    one for a given door chain)."""
    world = ex.world
    if _overlay_nav(emu, ex, map_id, goal,
                    _mansion_overlay(emu, world, map_id)):
        return True
    if not _mansion_press_statue(emu, ex, map_id):
        return False
    return _overlay_nav(emu, ex, map_id, goal,
                        _mansion_overlay(emu, world, map_id))


def _mansion_take_ball(emu: PyBoyEmulator, ex: Executor,
                       map_id: int, ball, item_id: int) -> bool:
    """pickup_ball's shape (stance, face, verify in the bag)
    but driven by the Mansion overlay instead of explore_nav,
    with one switch toggle in reserve if no stance is
    reachable in the current state."""
    world = ex.world
    want = _bag_count(emu, item_id)

    def attempt() -> bool:
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            stance = (ball[0] + dx, ball[1] + dy)
            if not _overlay_nav(
                    emu, ex, map_id, stance,
                    _mansion_overlay(emu, world, map_id)):
                continue
            _face_and_talk(emu, ball)
            emu.wait_free(max_rounds=30, stable_needed=3)
            if emu.read_state().in_battle:
                _drive_battle_typed(emu, flee_wild=True)
            if _bag_count(emu, item_id) > want:
                return True
        return False

    if attempt():
        return True
    if not _mansion_press_statue(emu, ex, map_id):
        return False
    return attempt()


def mansion_secret_key(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Cinnabar -> Pokemon Mansion -> the SECRET KEY in the
    basement. The route (probe158, proven live after an
    offline (cell, switch, floor) graph search - the §22
    draft's direct 1F-stairs route is IMPOSSIBLE: the stairs
    room's two doors need opposite switch states with no
    statue between them):
      1F entry -> up (5,10) -> 2F -> up (6,1) -> 3F ->
      statue ON -> hole (16,14) -> FALL to the 1F strip ->
      stairs (21,23) -> B1F -> statue (18,26) OFF ->
      statue (20,4) ON -> key ball (5,13) -> statue (20,4)
      OFF -> statue (18,26) ON -> exit (23,22) -> 1F east
      doors (26,27) -> Cinnabar.
    Every toggle is verified against the switch flag and
    the key against the bag, both from RAM."""
    world = ex.world
    cin = world.map_id("CinnabarIsland")
    m1 = world.map_id("PokemonMansion1F")
    m2 = world.map_id("PokemonMansion2F")
    m3 = world.map_id("PokemonMansion3F")
    mb = world.map_id("PokemonMansionB1F")
    if _bag_count(emu, ITEM_SECRET_KEY) > 0:
        return True
    if emu.read_state().map_id == cin:
        if not _surf_walk_to(emu, ex, cin, (6, 3)):
            return False
    if emu.read_state().map_id != m1:
        return False

    def nav(map_id, goal):
        _overlay_nav(emu, ex, map_id, goal,
                     _mansion_overlay(emu, world, map_id))

    def on(map_id, cells=None):
        st = emu.read_state()
        return st.map_id == map_id and \
            (cells is None or (st.x, st.y) in cells)

    def switch_is(want):
        return _event_flag(emu, EV_MANSION_SWITCH_ON) == want

    nav(m1, (5, 10))
    if not on(m2):
        return False
    nav(m2, (6, 1))
    if not on(m3):
        return False
    if not _mansion_press_statue(emu, ex, m3) \
            or not switch_is(True):
        return False
    for _attempt in range(3):
        nav(m3, (16, 14))
        if on(m1, {(16, 14), (17, 14)}):
            break
        if emu.read_state().map_id == m2:
            nav(m2, (6, 1))     # wrong hole (19,14): climb back
            continue
        return False
    else:
        return False
    nav(m1, (21, 23))
    if not on(mb):
        return False
    if not _mansion_press_statue(emu, ex, mb,
                                 MANSION_STATUES_ALT[mb]) \
            or not switch_is(False):
        return False
    if not _mansion_press_statue(emu, ex, mb) \
            or not switch_is(True):
        return False
    if not _overlay_nav(emu, ex, mb, (5, 14),
                        _mansion_overlay(emu, world, mb)):
        return False
    _face_and_talk(emu, (5, 13))
    emu.wait_free(max_rounds=30, stable_needed=3)
    if emu.read_state().in_battle:
        _drive_battle_typed(emu, flee_wild=True)
    if _bag_count(emu, ITEM_SECRET_KEY) < 1:
        return False
    if not _mansion_press_statue(emu, ex, mb) \
            or not switch_is(False):
        return False
    if not _mansion_press_statue(emu, ex, mb,
                                 MANSION_STATUES_ALT[mb]) \
            or not switch_is(True):
        return False
    nav(mb, (23, 22))
    if not on(m1):
        return False
    nav(m1, (26, 27))
    return emu.read_state().map_id == cin \
        and _bag_count(emu, ITEM_SECRET_KEY) >= 1


def _answer_quiz(emu: PyBoyEmulator, gate_event: int,
                 want: int) -> bool:
    """At a Cinnabar quiz machine's stance, facing it: talk,
    answer the Yes/No, and confirm the gate event - set
    directly on a right answer, or after the walk-up battle
    a wrong answer provokes (beating the trainer opens the
    gate too)."""
    emu.press("up")
    emu.tick(15)
    _press10(emu, "a")
    _yesno_pick(emu, want)
    for _ in range(300):
        if _event_flag(emu, gate_event):
            _drain_dialog(emu)
            return True
        st = emu.read_state()
        if st.in_battle:
            _drive_battle_typed(emu, flee_wild=False)
            if _event_flag(emu, gate_event):
                _drain_dialog(emu)
                return True
        elif st.in_dialog:
            _press10(emu, "a")
        emu.tick(15)
    return _event_flag(emu, gate_event)


def beat_blaine(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Cinnabar Gym (probe157): the QUIZ route - six machines,
    answers compiled from pokered, zero HP cost when the
    steering is right; a wrong answer costs one trainer
    battle and still opens the gate. Bag room is made for
    TM38 first (Blaine's script withholds it from a full
    bag, the Sabrina/TM46 trap). Verified from RAM: seven
    badge bits, EVENT_BEAT_BLAINE, TM38 in the bag."""
    world = ex.world
    cin = world.map_id("CinnabarIsland")
    gym = world.map_id("CinnabarGym")
    pc = world.map_id("CinnabarPokecenter")
    if _event_flag(emu, EV_BEAT_BLAINE):
        return True
    if emu.read_state().map_id == cin \
            and len(_bag_items(emu)) >= 20:
        if not _surf_walk_to(emu, ex, cin, (11, 11)):
            return False
        if emu.read_state().map_id != pc:
            return False
        if not deposit_item(emu, ex, pc, ITEM_TM06_TOXIC):
            return False
        if not _heal_at_pc(emu, ex, pc):
            return False
    if emu.read_state().map_id == cin:
        if not _surf_walk_to(emu, ex, cin, (18, 3)):
            return False
    if emu.read_state().map_id != gym:
        return False

    def overlay():
        return _door_overlay(world, gym, CINNABAR_GYM_DOORS,
                             lambda ev: _event_flag(emu, ev))

    for _round in range(8):
        if all(_event_flag(emu, ev) for ev in EV_GYM_GATES):
            break
        progress = False
        for _machine, stance, gi, want in CINNABAR_QUIZ:
            ev = EV_GYM_GATES[gi]
            if _event_flag(emu, ev):
                continue
            st = emu.read_state()
            if _overlay_path(world, gym, (st.x, st.y), stance,
                             overlay()) is None:
                continue
            if not _overlay_nav(emu, ex, gym, stance, overlay(),
                                flee_wild=False):
                continue
            if _answer_quiz(emu, ev, want):
                progress = True
        if not progress:
            return False
    if not all(_event_flag(emu, ev) for ev in EV_GYM_GATES):
        return False
    if not _overlay_nav(emu, ex, gym, (3, 4), overlay(),
                        flee_wild=False):
        return False
    emu.press("up")
    emu.tick(15)
    _press10(emu, "a")
    for _ in range(120):
        if emu.read_state().in_battle:
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(15)
    if not emu.read_state().in_battle:
        return False
    if not _drive_battle_hardened(emu):
        return False
    # The badge/TM38 ceremony dialog starts a beat AFTER the
    # battle ends; a bare _drain_dialog races ahead of it and
    # the script sits un-advanced with EV_BEAT_BLAINE unset
    # (probe158m4/m5: the payout had landed, the flag had
    # not). Ride A until the flag and the TM are both in.
    for _ in range(600):
        if _event_flag(emu, EV_BEAT_BLAINE) \
                and _bag_count(emu, ITEM_TM38) >= 1 \
                and not emu.read_state().in_dialog:
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(15)
    _drain_dialog(emu)
    st = emu.read_state()
    won = (len(st.badges) == 7
           and _event_flag(emu, EV_BEAT_BLAINE)
           and _event_flag(emu, EV_GOT_TM38)
           and _bag_count(emu, ITEM_TM38) >= 1)
    if won and emu.read_state().map_id == gym:
        # leave by the exit mats so the next leg starts on
        # Cinnabar's streets, not inside the gym
        _overlay_nav(emu, ex, gym, (16, 17), overlay())
        for _ in range(8):
            if emu.read_state().map_id != gym:
                break
            emu.press("down")
            _explore_settle(emu)
    return won


def _mom_heal(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Mom's free heal in Pallet (probe157) - colour on the
    way, soft-fail by design: try the stances around her,
    verify full HP, leave by the door mat."""
    world = ex.world
    pallet = world.map_id("PalletTown")
    home = world.map_id("RedsHouse1F")
    if emu.read_state().map_id == pallet:
        if not _surf_walk_to(emu, ex, pallet, (5, 5)):
            return False
    if emu.read_state().map_id != home:
        return False
    healed = False
    # (5, 2) -> talk DOWN to Mum at (5, 3) is the proven
    # stance (probe158n4); the side stances connect only
    # when she has wandered onto their line.
    for stance in ((5, 2), (4, 3), (6, 3), (5, 4)):
        _walk_fighting(emu, ex, home, stance, attempts=15)
        st = emu.read_state()
        if (st.x, st.y) != stance:
            continue
        _face_and_talk(emu, (5, 3))
        _drain_dialog(emu)
        if all(m.hp == m.max_hp for m in emu.read_state().party):
            healed = True
            break
    # Exit via the door mats, walked with the fighting
    # walker (the ex._walk_to variant stalled in probe158n3
    # and failed the whole heal after the heal had landed).
    for door in ((3, 7), (2, 7)):
        _walk_fighting(emu, ex, home, door, attempts=15)
        for _ in range(8):
            emu.press("down")
            _explore_settle(emu)
            if emu.read_state().map_id != home:
                break
        if emu.read_state().map_id != home:
            break
    return healed and emu.read_state().map_id == pallet


def _bike_voucher_run(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Vermilion detour (probe158n, added at Vlad's direction
    2026-11-10): from the Diglett's Cave Route 11 lobby, ride
    into Vermilion, make bag room at the PC if the tour's
    items need it (TM06, then TM38 - TM deposits only, and
    only because the voucher + rod would overflow a full
    bag), heal, then the Pokemon Fan Club: the chairman at
    (3, 1), talked to from (2, 1) facing RIGHT (the (3, 2)
    stance is his table), hear the story out, and verify
    the BIKE VOUCHER in the bag from RAM. Ends back on
    Vermilion's streets."""
    world = ex.world
    verm = world.map_id("VermilionCity")
    pc = world.map_id("VermilionPokecenter")
    fan = world.map_id("PokemonFanClub")
    if _bag_count(emu, ITEM_BIKE_VOUCHER) > 0:
        return emu.read_state().map_id == verm
    if not _hop_ride(emu, ex, ["VermilionCity"]):
        return False
    if emu.read_state().map_id != verm:
        return False
    if len(_bag_items(emu)) >= 19:
        if not _surf_walk_to(emu, ex, verm, (11, 3)):
            return False
        if emu.read_state().map_id != pc:
            return False
        # voucher + rod are two more stacks: deposit TMs
        # (TM06 first, then TM38) until the bag is at 18.
        for item in (ITEM_TM06_TOXIC, ITEM_TM38):
            if len(_bag_items(emu)) <= 18:
                break
            if _bag_count(emu, item) > 0:
                if not deposit_item(emu, ex, pc, item):
                    return False
        if not _heal_at_pc(emu, ex, pc):
            return False
    if emu.read_state().map_id != verm:
        return False
    if not _surf_walk_to(emu, ex, verm, (9, 13)):
        return False
    if emu.read_state().map_id != fan:
        return False
    _walk_fighting(emu, ex, fan, (2, 1), attempts=40)
    st = emu.read_state()
    if (st.x, st.y) != (2, 1):
        return False
    emu.press("right")
    emu.tick(20)
    _press10(emu, "a")
    for _ in range(500):
        if _bag_count(emu, ITEM_BIKE_VOUCHER) >= 1:
            break
        st = emu.read_state()
        if _yesno_open(emu):
            _yesno_pick(emu, 0)
            continue
        if st.in_dialog:
            _press10(emu, "a")
        emu.tick(20)
    _drain_dialog(emu)
    if _bag_count(emu, ITEM_BIKE_VOUCHER) < 1:
        return False
    _walk_fighting(emu, ex, fan, (3, 7), attempts=20)
    for _ in range(8):
        emu.press("down")
        _explore_settle(emu)
        if emu.read_state().map_id != fan:
            break
    return emu.read_state().map_id == verm


def _ride_or_cross(emu: PyBoyEmulator, ex: Executor,
                   dst_map: int) -> bool:
    """Cross to an adjacent outdoor map. The FOOT crossing
    goes first when a model path to the boundary exists:
    the R13 fence maze wedged both _ride_battling and
    _hop_ride for ~20 minutes each in probe158n6/n7 (and
    again in the probe159b replay, which burned 18 minutes
    without finishing the leg). The riders remain as
    fallbacks for legs the path cannot serve."""
    world = ex.world
    if emu.read_state().map_id == dst_map:
        return True
    src = emu.read_state().map_id
    gm = world.maps[src]
    direction = None
    for d, dst_id, _off in world.connections.get(src, []):
        if dst_id == dst_map:
            direction = d
            break
    if direction is not None:
        if direction == "west":
            edge = [(0, y) for y in range(gm.h)]
            step = "left"
        elif direction == "east":
            edge = [(gm.w - 1, y) for y in range(gm.h)]
            step = "right"
        elif direction == "south":
            edge = [(x, gm.h - 1) for x in range(gm.w)]
            step = "down"
        else:
            edge = [(x, 0) for x in range(gm.w)]
            step = "up"
        # Walk with _surf_walk_to (it replans around refused
        # cells) and the map's LIVE NPC cells pre-blocked:
        # static trainers are invisible to the tile model,
        # and a beaten one is a permanent obstacle - R13's
        # Bird Keeper at (12, 4) sat on the shortest line
        # west and every model-pathed walker bumped into
        # him for hundreds of attempts (probe159e: 446 s to
        # crawl to the cell beside him and stall).
        #
        # Candidate boundary rows are tried nearest-first,
        # and each landing is MOBILITY-TESTED: the reachable
        # region from the landing cell (live NPCs excluded)
        # must reach 40 cells. R14 taught why (probe159g):
        # crossing R13 at row 4 lands in a 4-cell pocket
        # whose only exit cell is occupied by a STAY Bird
        # Keeper - region 4, sealed - while row 10 lands in
        # the open map (region 400+). A sealed landing steps
        # back and tries the next row.
        def live_npcs(gmap, here):
            return frozenset(
                cell for cell in
                (_boulder_cell(emu, k) for k in range(1, 16))
                if cell != here and gmap.in_bounds(*cell)
                and gmap.tile_walkable(*cell))

        def region_size(gmap, start, npcs, cap=40):
            from collections import deque
            seen = {start}
            queue = deque([start])
            while queue and len(seen) < cap:
                cx, cy = queue.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (cx + dx, cy + dy)
                    if n in seen or n in npcs \
                            or not gmap.in_bounds(*n):
                        continue
                    if gmap.tile_walkable(*n) \
                            or n in gmap.surf_cells:
                        seen.add(n)
                        queue.append(n)
            return len(seen)

        back = {"left": "right", "right": "left",
                "up": "down", "down": "up"}[step]
        st = emu.read_state()
        npcs0 = live_npcs(gm, (st.x, st.y))
        cands = []
        for cell in edge:
            if not gm.tile_walkable(*cell) or cell in npcs0:
                continue
            p = world.surf_path(src, (st.x, st.y), cell,
                                blocked=npcs0)
            if p is not None:
                cands.append((len(p), cell))
        cands.sort()
        tried_rows = set()
        uniq = []
        for _ln, cell in cands:
            row = cell[1] if step in ("left", "right") \
                else cell[0]
            if row in tried_rows:
                continue
            tried_rows.add(row)
            uniq.append(cell)
            if len(uniq) >= 6:
                break
        dst_gm = world.maps[dst_map]
        for target in uniq:
            if emu.read_state().map_id != src:
                break
            for _round in range(3):
                st = emu.read_state()
                if st.map_id != src:
                    break
                npcs = live_npcs(gm, (st.x, st.y))
                if _surf_walk_to(emu, ex, src, target,
                                 blocked0=npcs):
                    break
                st2 = emu.read_state()
                if (st2.x, st2.y) == (st.x, st.y):
                    break      # no movement; row is a dud
            for _ in range(8):
                if emu.read_state().map_id == dst_map:
                    break
                emu.press(step)
                _explore_settle(emu)
                if emu.read_state().in_battle:
                    _drive_battle_typed(emu, flee_wild=True)
            if emu.read_state().map_id != dst_map:
                continue
            st = emu.read_state()
            dst_npcs = live_npcs(dst_gm, (st.x, st.y))
            if region_size(dst_gm, (st.x, st.y),
                           dst_npcs) >= 40:
                return True
            # sealed pocket: step back and try the next row
            for _ in range(10):
                if emu.read_state().map_id == src:
                    break
                emu.press(back)
                _explore_settle(emu)
                if emu.read_state().in_battle:
                    _drive_battle_typed(emu, flee_wild=True)
    _ride_battling(emu, ex, dst_map)
    if emu.read_state().map_id == dst_map:
        return True
    _hop_ride(emu, ex, [world.map_name(dst_map)])
    return emu.read_state().map_id == dst_map


def super_rod_run(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Pallet -> R1 -> Viridian -> R2 -> Diglett's Cave ->
    R11 -> the VERMILION detour (_bike_voucher_run: PC room
    + the Fan Club BIKE VOUCHER, probe158) -> R12 -> the
    Fishing Guru's house and the SUPER ROD, then home
    R12 -> R13 -> R14 -> R15 -> Fuchsia and a heal. The
    guru is an NPC inside his house (map 189; he stands at
    (2, 4) facing right, talked to from (1, 4) - the Good
    Rod pattern). Fisher5 (event 1160) is asserted
    unengaged at the R12 checkpoints: the door path down
    the pier never enters his sealed pocket."""
    world = ex.world
    pallet = world.map_id("PalletTown")
    lobby_s = world.map_id("DiglettsCaveRoute2")
    cave = world.map_id("DiglettsCave")
    lobby_n = world.map_id("DiglettsCaveRoute11")
    r12 = world.map_id("Route12")
    house = world.map_id("Route12SuperRodHouse")
    if _bag_count(emu, ITEM_SUPER_ROD) > 0:
        return True
    if _event_flag(emu, EV_BEAT_ROUTE12_FISHER5):
        return False
    if emu.read_state().map_id != pallet:
        if not surf_to(emu, ex, pallet):
            return False
    _mom_heal(emu, ex)
    if not _hop_ride(emu, ex, ["Route1", "ViridianCity",
                               "Route2"]):
        return False
    if not _ride_battling(emu, ex, lobby_s):
        return False
    _walk_fighting(emu, ex, lobby_s, (4, 4), attempts=30)
    if emu.read_state().map_id != cave:
        return False
    _walk_fighting(emu, ex, cave, (37, 31), attempts=60,
                   flee_wild=True)
    if emu.read_state().map_id != lobby_n:
        return False
    if not _bike_voucher_run(emu, ex):
        return False
    if not _hop_ride(emu, ex, ["Route11", "Route12"]):
        return False
    if _event_flag(emu, EV_BEAT_ROUTE12_FISHER5):
        return False
    _walk_fighting(emu, ex, r12, (11, 77), attempts=80)
    if emu.read_state().map_id != house:
        return False
    _walk_fighting(emu, ex, house, (1, 4), attempts=20)
    st = emu.read_state()
    if (st.x, st.y) != (1, 4):
        return False
    emu.press("right")
    emu.tick(20)
    _press10(emu, "a")
    for _ in range(300):
        if _bag_count(emu, ITEM_SUPER_ROD) >= 1:
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(25)
    _close_menus(emu)
    if _bag_count(emu, ITEM_SUPER_ROD) < 1:
        return False
    if _event_flag(emu, EV_BEAT_ROUTE12_FISHER5):
        return False
    for _name in ("Route13", "Route14", "Route15",
                  "FuchsiaCity"):
        if not _ride_or_cross(emu, ex, world.map_id(_name)):
            return False
    if _event_flag(emu, EV_BEAT_ROUTE12_FISHER5):
        return False
    if not _ride_battling(emu, ex, MAP_FUCHSIA_PC):
        return False
    return _heal_fuchsia_pc(emu, ex)


def _safari_pond_stance(emu: PyBoyEmulator, ex: Executor):
    """(stance, face_dir) for fishing Safari Center's pond: a
    _safari_passable land cell next to $14 water, nearest the
    area anchor AND reachable from it over _safari_passable
    cells. The rod needs a shore/water tile in front
    (FishingInit's IsNextTileShoreOrWater). The reachability
    filter is load-bearing (probe158o): the pond's south
    shore segments pass the adjacency test but are fenced
    islands in the passability grid - the unfiltered search
    picked (15, 14) and the nav died on the spot. (The live
    catch in probe158o3 was made from the north shore
    (15, 7); the filtered search returns the nearest
    reachable stance, currently (20, 14).)"""
    from collections import deque
    world = ex.world
    gm = world.maps[MAP_SZ_CENTER]
    anchor = SAFARI_ANCHORS[MAP_SZ_CENTER]
    warps = {(w.x, w.y) for w in gm.warps}
    reach = {anchor}
    queue = deque([anchor])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in _SZ_DIRS.values():
            n = (cx + dx, cy + dy)
            if n in reach or n in warps \
                    or not _safari_passable(MAP_SZ_CENTER, n):
                continue
            reach.add(n)
            queue.append(n)
    best = None
    for y in range(gm.h):
        for x in range(gm.w):
            if (x, y) not in reach:
                continue
            if not _safari_passable(MAP_SZ_CENTER, (x, y)):
                continue
            for d, (dx, dy) in _SZ_DIRS.items():
                n = (x + dx, y + dy)
                if gm.in_bounds(*n) \
                        and gm.tiles[n[1]][n[0]] == 0x14:
                    dist = abs(x - anchor[0]) + abs(y - anchor[1])
                    if best is None or dist < best[0]:
                        best = (dist, (x, y), d)
    if best is None:
        return None
    return best[1], best[2]


def _fish_cast(emu: PyBoyEmulator) -> str:
    """Use the SUPER ROD from the bag (the
    _use_bag_item_on_mon steering minus the mon target: a rod
    fires straight from the bag list). Returns "bite" (a
    battle started), "nibble" (no-bite text, drained), or
    "fail". The caller stands facing water."""
    mem = emu.pb.memory
    bag_idx = _bag_index(emu, ITEM_SUPER_ROD)
    if bag_idx is None:
        return "fail"
    _press10(emu, "start")
    if not _wait_menu_y(emu, 2):
        return "fail"
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 2):
        return "fail"
    _press10(emu, "a")                         # ITEM
    if not _wait_menu_y(emu, 4):
        return "fail"
    if not _steer_menu(
            emu, lambda: mem[ADDR_MENU_SCROLL] + mem[ADDR_MENU_ITEM],
            bag_idx):
        return "fail"
    _press10(emu, "a")                         # the rod itself
    for _ in range(30):
        if emu.read_state().in_battle:
            return "bite"
        if mem[ADDR_MENU_TOP_Y] == 11:         # USE/TOSS submenu
            _press10(emu, "a")                 # USE
            break
        emu.tick(5)
    quiet = 0
    for _ in range(400):
        st = emu.read_state()
        if st.in_battle:
            return "bite"
        if st.in_dialog:
            quiet = 0
            _press10(emu, "a")
        else:
            quiet += 1
            if quiet >= 25:
                return "nibble"
        emu.tick(10)
    return "fail"


def _safari_fish_battle(emu: PyBoyEmulator, owned: set,
                        log: list, media=None) -> dict:
    """One Safari battle from the rod (probe157). Policy per
    the S20 anomaly: NO BAIT, NO ROCK - unbaited balls only
    (every S20 catch was an unbaited first-ball throw, while
    baited targets fled 10 of 11). Caps: 15 balls for Dratini
    (the mission), 6 for bycatch; RUN from owned species.
    Catches verified by the stored count + _catch_epilogue.
    `media(kind, emu)`, when given, fires at "encounter",
    each "throw", and "caught"."""
    mem = emu.pb.memory
    sp = mem[ADDR_ENEMY_SPECIES]
    lv = mem[ADDR_ENEMY_LEVEL]
    name = SAFARI_SPECIES.get(sp, f"0x{sp:02x}")
    rec = {"species": name, "level": lv, "bait": 0, "rock": 0,
           "balls": 0, "outcome": "?", "via": "rod"}
    log.append(rec)
    if media:
        media("encounter", emu)
    if sp in owned:
        rec["outcome"] = "fled(owned)"
        _run_away(emu)
        return rec
    cap = 15 if sp == 0x58 else 6     # 0x58 = DRATINI
    stored0 = _stored_count(emu)
    guard = 0
    while emu.read_state().in_battle and guard < 40:
        guard += 1
        if _safari_balls(emu) == 0:
            rec["outcome"] = "balls-out"
            break
        if rec["balls"] >= cap:
            rec["outcome"] = "fled(cap)"
            _run_away(emu)
            break
        if media:
            media("throw", emu)
        if not _safari_menu_pick(emu, 0, 1):        # BALL
            break
        rec["balls"] += 1
        caught = False
        for _ in range(160):
            _press10(emu, "a")
            emu.tick(10)
            if _stored_count(emu) > stored0:
                caught = True
                break
            if not emu.read_state().in_battle:
                break
        if caught:
            ok = _catch_epilogue(emu, stored0 + 1)
            rec["outcome"] = "CAUGHT" if ok else "caught?"
            if ok:
                owned.add(sp)
                if media:
                    media("caught", emu)
            return rec
    if rec["outcome"] == "?":
        rec["outcome"] = "fled(enemy)" \
            if not emu.read_state().in_battle else "stuck"
    return rec


def safari_dratini(emu: PyBoyEmulator, ex: Executor,
                   log: list = None, media=None,
                   max_entries: int = 3) -> bool:
    """Fish Dratini in the Safari Zone (probe157): paid
    entries (at most `max_entries`), Super Rod at the Center
    pond - all four areas share Super Rod group 6 (Dratini /
    Krabby / Psyduck / Slowpoke, all Lv15; a bite is 50% per
    cast and Dratini is 1 of 4, so ~8 casts expected).
    Fishing costs no steps (the counter only ticks on walking
    steps), so an entry fishes until Dratini is caught, the
    balls run low, or the steps run out. Grass battles in
    transit are fled by _sz_nav's stock settle; if Dratini
    falls early with steps and balls to spare, the remainder
    of that entry paces the Center grass pair under the hunt
    policy (Scyther 4% / Chansey 1% live there). Ends healed
    at the Fuchsia Pokecenter; success = Dratini owned."""
    if log is None:
        log = []
    if _bag_count(emu, ITEM_SUPER_ROD) < 1:
        return False
    owned = set()
    st = emu.read_state()
    for mon in st.party:
        owned.add(mon.species_id)
    owned.update(_box_species(emu))
    if 0x58 in owned:
        return True
    stance = _safari_pond_stance(emu, ex)
    if stance is None:
        return False
    (sx, sy), face = stance
    for _entry in range(max_entries):
        if not _ride_battling(emu, ex, MAP_SAFARI_GATE):
            return False
        money0 = emu.read_state().money
        _walk_fighting(emu, ex, MAP_SAFARI_GATE, (4, 2),
                       attempts=20)
        if (emu.read_state().x, emu.read_state().y) != (4, 2):
            return False
        emu.press("right")
        emu.tick(20)
        _press10(emu, "a")
        paid = False
        for _ in range(200):
            if _safari_balls(emu) == 30 and \
                    _event_flag(emu, EV_IN_SAFARI_ZONE):
                paid = True
                break
            if emu.read_state().in_dialog:
                _press10(emu, "a")
            emu.tick(25)
        if not paid or emu.read_state().money != money0 - 500:
            return False
        for _ in range(80):
            if not emu.read_state().in_dialog:
                break
            _press10(emu, "a")
            emu.tick(30)
        for _ in range(10):
            if emu.read_state().map_id == MAP_SZ_CENTER:
                break
            emu.press("up")
            _explore_settle(emu)
            _safari_settle(emu, [])
        if emu.read_state().map_id != MAP_SZ_CENTER:
            return False
        if not _sz_nav(emu, ex, MAP_SZ_CENTER, (sx, sy), [],
                       budget_guard=10):
            return False
        # --- fish ------------------------------------------------
        casts = 0
        while _safari_balls(emu) > 4 and _safari_steps(emu) > 12:
            st = emu.read_state()
            if st.map_id != MAP_SZ_CENTER \
                    or (st.x, st.y) != (sx, sy):
                break
            emu.press(face)
            emu.tick(15)
            casts += 1
            res = _fish_cast(emu)
            rec = {"species": "-", "level": 0, "bait": 0,
                   "rock": 0, "balls": 0, "outcome": f"cast:{res}",
                   "via": "rod"}
            if res == "bite":
                rec = _safari_fish_battle(emu, owned, log, media)
            else:
                log.append(rec)
            if 0x58 in owned:
                break
        log.append({"species": "ENTRY", "level": casts,
                    "bait": 0, "rock": 0, "balls": 0,
                    "outcome": "casts", "via": "rod"})
        # --- bonus grass time if Dratini fell early --------------
        # Best-effort only: a _safari_settle RUN failure inside
        # the nav raised RuntimeError in probe158o2 AFTER the
        # catch and killed the whole run before its exit tail.
        # The bonus must never cost the mission - catch and
        # fall through to the leave/heal below.
        if 0x58 in owned and _safari_steps(emu) > 30 \
                and _safari_balls(emu) > 4:
            try:
                pair = _safari_grass_pair(MAP_SZ_CENTER)
                if pair is not None and _sz_nav(
                        emu, ex, MAP_SZ_CENTER, pair[0], [],
                        budget_guard=10):
                    toggle = False
                    while emu.read_state().map_id == MAP_SZ_CENTER \
                            and _safari_steps(emu) > 12 \
                            and _safari_balls(emu) > 0 \
                            and not SAFARI_TARGETS <= owned:
                        tgt = pair[1] if toggle else pair[0]
                        toggle = not toggle
                        for d, (dx, dy) in _SZ_DIRS.items():
                            cur = (emu.read_state().x,
                                   emu.read_state().y)
                            if (cur[0] + dx, cur[1] + dy) == tgt:
                                emu.press(d)
                                break
                        _safari_hunt_settle(emu, owned, log)
            except RuntimeError:
                log.append({"species": "BONUS", "level": 0,
                            "bait": 0, "rock": 0, "balls": 0,
                            "outcome": "aborted(settle)",
                            "via": "grass"})
        # --- leave ------------------------------------------------
        if emu.read_state().map_id == MAP_SZ_CENTER:
            _safari_burn_out(emu, owned, log)
        if emu.read_state().map_id != MAP_SAFARI_GATE:
            return False
        for _ in range(12):
            emu.press("down")
            _explore_settle(emu)
            _safari_settle(emu, [])
            if emu.read_state().map_id != MAP_SAFARI_GATE:
                break
        if emu.read_state().map_id == MAP_SAFARI_GATE:
            return False
        if not _ride_battling(emu, ex, MAP_FUCHSIA_PC):
            return False
        if not _heal_fuchsia_pc(emu, ex):
            return False
        if 0x58 in owned:
            return True
    return 0x58 in owned


# ------------------------------------------------------------------
# The Fly leg (probe160, docs S24): Bicycle exchange, the Route 16
# Snorlax CATCH (Vlad's decision: balls first, Master Ball never,
# KO only when the balls are gone), HM02 from the Fly House, Fly
# taught to the boxed Pidgey (verify-first), and the first flight
# home. Every span below was proven live in probe160 before it was
# ported; the fidelity notes are in S24.

ITEM_BICYCLE = 0x06
ITEM_HM02_FLY = 0xC5            # HM ids are 0xC4 + (n - 1)
MOVE_FLY = 19                   # 0x13
MOVE_SLEEP_POWDER = 79
MOVE_BUBBLE = 145
EV_GOT_BICYCLE = 192
EV_FIGHT_R16_SNORLAX = 1224
EV_BEAT_R16_SNORLAX = 1225
EV_GOT_HM02 = 1230
SPECIES_SNORLAX = 0x84
ADDR_ENEMY_HP = ADDR_ENEMY_SPECIES + 1        # dw, big-endian
ADDR_ENEMY_STATUS = ADDR_ENEMY_SPECIES + 4    # sleep = low 3 bits


def _step_to_live(emu: PyBoyEmulator, ex: Executor, map_id: int,
                  tx: int, ty: int, max_steps: int = 200) -> bool:
    """Greedy live stepper for cells the compiled model cannot
    path (probe160's R16 lesson: around the Snorlax cell and the
    gatehouse the model's tile_path simply returns None, while
    the live grid is perfectly walkable). Steps axis-greedy,
    clearing dialogs and hardened-driving any battle that starts
    (trainer engagement taunts arrive as DIALOGS, not battles -
    a walker that only checks in_battle stalls on them)."""
    def clear():
        for _ in range(300):
            st = emu.read_state()
            if st.in_battle:
                _drive_battle_hardened(emu)
                continue
            if st.in_dialog:
                _press10(emu, "a")
                emu.tick(30)
                continue
            return
    for _ in range(max_steps):
        st = emu.read_state()
        if st.map_id != map_id:
            return False
        clear()
        st = emu.read_state()
        if (st.x, st.y) == (tx, ty):
            return True
        dx = (tx > st.x) - (tx < st.x)
        dy = (ty > st.y) - (ty < st.y)
        cands = (["right" if dx > 0 else "left"] if dx else []) + \
            (["down" if dy > 0 else "up"] if dy else [])
        for mv in cands:
            ox, oy = st.x, st.y
            emu.press(mv)
            _explore_settle(emu)
            clear()
            st2 = emu.read_state()
            if (st2.x, st2.y) != (ox, oy) or st2.map_id != map_id:
                break
    st = emu.read_state()
    return st.map_id == map_id and (st.x, st.y) == (tx, ty)


def bicycle_exchange(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Cerulean Bike Shop: BIKE VOUCHER -> BICYCLE (probe160b).

    The script's GiveItem runs BEFORE the voucher is removed,
    so at a full 20-stack bag the exchange aborts on the
    bag-full branch (verified live). Make room first by
    depositing an HM/TM at the Cerulean PC (HM05 Flash in
    probe160 - Voltorb already knows the move; the item is
    retrievable). The clerk stands at (6, 2) behind the counter
    (row 3); the talk stance is (6, 4) facing UP across it.
    Verified from RAM: BICYCLE x1 in bag, voucher consumed,
    EV_GOT_BICYCLE set."""
    world = ex.world
    shop = world.map_id("BikeShop")
    cerulean = world.map_id("CeruleanCity")
    pc = world.map_id("CeruleanPokecenter")
    if _bag_count(emu, ITEM_BICYCLE) >= 1:
        return True
    if _bag_count(emu, ITEM_BIKE_VOUCHER) < 1:
        return False
    if len(_bag_items(emu)) >= 20:
        room_item = ITEM_HM05 if _bag_count(emu, ITEM_HM05) else None
        if room_item is None:
            for it in _bag_items(emu):
                if it >= 0xC4:            # any HM/TM makes room
                    room_item = it
                    break
        if room_item is None:
            return False
        # deposit_item assumes the caller is already inside
        # the Pokecenter (probe160b called it from inside)
        if emu.read_state().map_id != pc and \
                not _ride_or_cross(emu, ex, pc):
            return False
        if not deposit_item(emu, ex, pc, room_item):
            return False
    if emu.read_state().map_id != cerulean:
        if not _ride_or_cross(emu, ex, cerulean):
            return False
    if not _ride_or_cross(emu, ex, shop):
        return False
    if not ex._walk_to(shop, (6, 4)):
        _step_to_live(emu, ex, shop, 6, 4)
    st = emu.read_state()
    if (st.x, st.y) != (6, 4):
        _step_to_live(emu, ex, shop, 6, 4)
    if (emu.read_state().x, emu.read_state().y) != (6, 4):
        return False
    emu.press("up")
    emu.tick(20)
    _press10(emu, "a")
    for _ in range(200):
        if _bag_count(emu, ITEM_BICYCLE) >= 1:
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(25)
    _close_menus(emu)
    return _bag_count(emu, ITEM_BICYCLE) == 1 and \
        _bag_count(emu, ITEM_BIKE_VOUCHER) == 0 and \
        _event_flag(emu, EV_GOT_BICYCLE)


def _ride_to_battle_menu(emu: PyBoyEmulator) -> str:
    """Ride to the next main battle menu via _advance_battle
    (probe160d's ride_to_menu). The battle's text pages need A
    presses to advance - a passive wait stalls forever on the
    wake text - and _advance_battle is the proven rider. The
    faint-replacement screen (party list with the active mon
    at 0 HP) is handled by picking the first healthy mon.
    Returns 'menu' / 'over'."""
    mem = emu.pb.memory
    for _ in range(60):
        st = emu.read_state()
        if not st.in_battle:
            return "over"
        if mem[ADDR_MENU_TOP_Y] == 1 and \
                (mem[ADDR_BATTLE_MON_HP] << 8
                 | mem[ADDR_BATTLE_MON_HP + 1]) == 0:
            pick = next((i for i, m in enumerate(st.party)
                         if m.hp > 0), 0)
            cur = mem[ADDR_MENU_ITEM]
            if cur == pick:
                _press10(emu, "a")
                emu.tick(50)
            else:
                _press10(emu, "down" if cur < pick else "up")
                emu.tick(25)
            continue
        r = _advance_battle(emu, rounds=20)
        if r in ("menu", "over"):
            return r
    return "menu" if emu.read_state().in_battle else "over"


def _battle_menu_sig(emu: PyBoyEmulator):
    mem = emu.pb.memory
    return (mem[ADDR_MENU_TOP_Y], mem[ADDR_MENU_ITEM],
            mem[ADDR_MENU_COL])


def _battle_switch_to(emu: PyBoyEmulator, party_idx: int) -> bool:
    """Main menu -> PKMN (cursor (14, 0, 15)) -> the mon ->
    confirm the SWITCH/STATS/CANCEL submenu with A presses
    until the new mon is actually out (probe160d switch_to).
    _advance_battle leaves the main-menu cursor on FIGHT
    (14, 0, 9); one RIGHT press lands on PKMN."""
    mem = emu.pb.memory
    want_sp = emu.read_state().party[party_idx].species_id
    if _ride_to_battle_menu(emu) != "menu":
        return False
    if _battle_menu_sig(emu) != (14, 0, 9):
        return False
    _press10(emu, "right")
    if _battle_menu_sig(emu) != (14, 0, 15):
        return False
    _press10(emu, "a")
    for _ in range(80):
        if mem[ADDR_MENU_TOP_Y] == 1:
            break
        emu.tick(20)
    if mem[ADDR_MENU_TOP_Y] != 1:
        return False
    _steer_byte(emu, lambda: mem[ADDR_MENU_ITEM], party_idx)
    _raw(emu, "a", hold=4, settle=22)
    emu.tick(60)
    for _ in range(160):
        if not emu.read_state().in_battle:
            return False
        if mem[ADDR_BATTLE_MON_SPECIES] == want_sp:
            return True
        _press10(emu, "a")
        emu.tick(30)
    return False


def _battle_use_move(emu: PyBoyEmulator, move_id: int) -> bool:
    """Main menu (cursor left on FIGHT by _advance_battle) ->
    the move menu (top_y 12; its cursor byte is 1-based) ->
    the move (probe160d use_move)."""
    from agent.state import ADDR_PARTY_MONS
    mem = emu.pb.memory
    if _ride_to_battle_menu(emu) != "menu":
        return False
    if _battle_menu_sig(emu) != (14, 0, 9):
        return False
    active = mem[ADDR_BATTLE_MON_SPECIES]
    idx = None
    for i, mon in enumerate(emu.read_state().party):
        if mon.species_id == active:
            moves = [mem[ADDR_PARTY_MONS + 44 * i + 8 + j]
                     for j in range(4)]
            if move_id in moves:
                idx = moves.index(move_id)
            break
    if idx is None:
        return False
    _press10(emu, "a")                            # FIGHT
    for _ in range(80):
        if mem[ADDR_MENU_TOP_Y] == 12:
            break
        emu.tick(20)
    if mem[ADDR_MENU_TOP_Y] != 12:
        return False
    _steer_byte(emu, lambda: mem[ADDR_MENU_ITEM], idx + 1)
    _raw(emu, "a", hold=4, settle=22)
    emu.tick(70)
    return True


def _throw_ball_ordered(emu: PyBoyEmulator, ball_id: int,
                        stored0: int) -> str:
    """Throw one ball of the given kind via the proven
    _throw_ball (its ITEM_POKE_BALL global is patched to the
    wanted ball for the call), then ride the throw out exactly
    as probe160d did: the catch lands on the stored-count tick
    (+ _catch_epilogue for the dex page / box transfer), the
    broke-free case is the main menu genuinely back (top_y 14
    and NOT in_dialog, confirmed stable), anything else is
    text to ride with A. Returns 'caught' / 'broke' / 'over'.
    """
    global ITEM_POKE_BALL
    saved = ITEM_POKE_BALL
    ITEM_POKE_BALL = ball_id
    try:
        ok = _throw_ball(emu)
    finally:
        ITEM_POKE_BALL = saved
    if not ok:
        return "over"
    mem = emu.pb.memory
    for _ in range(240):
        if _stored_count(emu) > stored0:
            _catch_epilogue(emu, stored0 + 1)
            return "caught"
        st = emu.read_state()
        if not st.in_battle:
            return "over"
        if mem[ADDR_MENU_TOP_Y] == 14 and not st.in_dialog:
            emu.tick(30)
            if mem[ADDR_MENU_TOP_Y] == 14:
                return "broke"
        if st.in_dialog or mem[ADDR_MENU_TOP_Y] not in (12, 14):
            _press10(emu, "a")
        emu.tick(25)
    return "broke" if emu.read_state().in_battle else "over"


def _enemy_hp(emu: PyBoyEmulator) -> int:
    mem = emu.pb.memory
    return (mem[ADDR_ENEMY_HP] << 8) | mem[ADDR_ENEMY_HP + 1]


def _enemy_asleep(emu: PyBoyEmulator) -> bool:
    return (emu.pb.memory[ADDR_ENEMY_STATUS] & 0x07) != 0


def snorlax_catch(emu: PyBoyEmulator, ex: Executor) -> str:
    """The Route 16 Snorlax, CAUGHT per Vlad's decision
    (probe160d): wake with the Poke Flute from the stance EAST
    of the sleeping Snorlax at (27, 10) - its body cell is
    (26, 10). Policy, exactly as proven:

    - Oddish leads the sleep phase: switch in, Sleep Powder
      once (it outspeeds nothing - Snorlax moves first, so
      Powder must land before its Rest; retry while awake).
    - Blastoise chips ONLY inside a sleep window and only
      above the danger line: Bubble (20 power) at full HP -
      even a crit (141 -> 68 live) cannot kill.
    - THROW FIRST: once HP <= 80, throw at every menu - GREAT
      BALLS first, then POKE BALLS. The MASTER BALL IS NEVER
      THROWN (it is not referenced anywhere in this function).
    - Gen 1 sleep does not break on damage; Rest's self-sleep
      is exactly 2 turns, and a Rest at low HP is answered by
      re-sleeping (Oddish) rather than more damage.
    - If every ball is gone without a catch, KO it and pass
      (Vlad's pre-authorised fallback) via the hardened driver.

    The checkpoint rule is the CALLER's: save a state before
    calling; on a catastrophic crit-KO the caller restores it
    and retries once. Returns 'caught' / 'ko' / 'failed'.
    RAM-verified on success: stored count +1, dex +1, Snorlax
    in the box (party is full), EV_BEAT_R16_SNORLAX set."""
    mem = emu.pb.memory
    world = ex.world
    r16 = world.map_id("Route16")
    st = emu.read_state()
    if st.map_id != r16 or (st.x, st.y) != (27, 10):
        return "failed"
    if _bag_count(emu, ITEM_POKE_FLUTE) < 1 or \
            _event_flag(emu, EV_BEAT_R16_SNORLAX):
        return "failed"
    stored0 = _stored_count(emu)
    dex0 = _dex_owned_count(emu)
    oddish = _party_index_of(emu, SPECIES_ODDISH)
    blastoise = _party_index_of(emu, SPECIES_BLASTOISE)
    if oddish is None or blastoise is None:
        return "failed"
    # Wake it: bag -> Poke Flute. NOT _use_bag_item: that
    # helper verifies success by the count dropping, and the
    # Flute is a key item - it is never consumed, so the
    # helper reports failure while the wake succeeds behind
    # it (probe161). Drive the bag menu directly and verify
    # by the battle itself.
    idx = _bag_index(emu, ITEM_POKE_FLUTE)
    if idx is None:
        return "failed"
    _press10(emu, "start")
    if not _wait_menu_y(emu, 2):
        return "failed"
    if not _steer_menu(emu, lambda: mem[ADDR_MENU_ITEM], 2):
        return "failed"
    _press10(emu, "a")                            # ITEM
    if not _wait_menu_y(emu, 4):
        return "failed"
    if not _steer_menu(
            emu, lambda: mem[ADDR_MENU_SCROLL]
            + mem[ADDR_MENU_ITEM], idx):
        return "failed"
    _press10(emu, "a")
    if not _wait_menu_y(emu, 11):
        return "failed"
    _press10(emu, "a")                            # USE the flute
    for _ in range(600):
        if emu.read_state().in_battle and \
                mem[ADDR_ENEMY_SPECIES] == SPECIES_SNORLAX:
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(25)
    if not emu.read_state().in_battle or \
            mem[ADDR_ENEMY_SPECIES] != SPECIES_SNORLAX:
        return "failed"
    max_hp = (mem[ADDR_ENEMY_MAX_HP] << 8) | \
        mem[ADDR_ENEMY_MAX_HP + 1]
    powder_tried = False
    turns = 0
    while emu.read_state().in_battle and turns < 90:
        turns += 1
        if _ride_to_battle_menu(emu) != "menu":
            break
        hp = _enemy_hp(emu)
        asleep = _enemy_asleep(emu)
        active = mem[ADDR_BATTLE_MON_SPECIES]
        great = _bag_count(emu, ITEM_GREAT_BALL)
        poke = _bag_count(emu, ITEM_POKE_BALL)
        if great + poke == 0:
            _drive_battle_hardened(emu)       # the KO fallback
            break
        if hp <= 80:
            ball = ITEM_GREAT_BALL if great else ITEM_POKE_BALL
            if _throw_ball_ordered(emu, ball, stored0) == "caught":
                break
            continue
        if not asleep and not powder_tried:
            if active != SPECIES_ODDISH:
                if not _battle_switch_to(emu, oddish):
                    break
            elif not _battle_use_move(emu, MOVE_SLEEP_POWDER):
                break
            powder_tried = True
            continue
        if asleep and hp > 80:
            if active != SPECIES_BLASTOISE:
                if not _battle_switch_to(emu, blastoise):
                    break
            elif not _battle_use_move(emu, MOVE_BUBBLE):
                break
            continue
        if not asleep and powder_tried and hp > max_hp * 3 // 5:
            powder_tried = False              # re-sleep window
            continue
        ball = ITEM_GREAT_BALL if great else ITEM_POKE_BALL
        if _throw_ball_ordered(emu, ball, stored0) == "caught":
            break
    st = emu.read_state()
    if _stored_count(emu) == stored0 + 1 and \
            _dex_owned_count(emu) == dex0 + 1 and \
            SPECIES_SNORLAX in _box_species(emu) and \
            _event_flag(emu, EV_BEAT_R16_SNORLAX):
        return "caught"
    if not st.in_battle and \
            _event_flag(emu, EV_BEAT_R16_SNORLAX):
        return "ko"
    return "failed"


def route16_fly_house(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Route 16 east stance -> the Fly House -> HM02 (probe160e).

    The way west is NOT the gatehouse: its south corridor is
    the Cycling Road bike-check (a counter, not a passage) and
    row 9 is a continuous hedgerow fence. The real entrance is
    the CUT TREE at (34, 9), whose 2x2 block straddles the
    fence between the road (34, 10) and the east yard (34, 8).
    Through the yard to the gate's NORTH corridor (row 2 - the
    gambler stands on row 3), out into the house plot, and in
    the door at (7, 5). The girl at (2, 3) faces RIGHT: talk
    from (1, 3). Bag room first is the caller's job (deposit a
    TM at Celadon PC); verified from RAM: HM02 x1,
    EV_GOT_HM02 set."""
    world = ex.world
    r16 = world.map_id("Route16")
    gate = world.map_id("Route16Gate1F")
    house = world.map_id("Route16FlyHouse")
    if _bag_count(emu, ITEM_HM02_FLY) >= 1:
        return True
    st = emu.read_state()
    if st.map_id != r16:
        return False
    if not _step_to_live(emu, ex, r16, 34, 10):
        return False
    ex._cut_tree(r16, (34, 9))
    st = emu.read_state()
    if (st.x, st.y) != (34, 9):
        return False
    if not _step_to_live(emu, ex, r16, 25, 5):
        return False
    emu.press("left")
    _explore_settle(emu)
    if emu.read_state().map_id != gate:
        return False
    if not _step_to_live(emu, ex, gate, 1, 2):
        return False
    emu.press("left")
    _explore_settle(emu)
    st = emu.read_state()
    if st.map_id != r16 or st.x > 17:
        return False
    if not _step_to_live(emu, ex, r16, 7, 6):
        return False
    emu.press("up")
    _explore_settle(emu)
    if emu.read_state().map_id != house:
        return False
    if not ex._walk_to(house, (1, 3)):
        _step_to_live(emu, ex, house, 1, 3)
    if (emu.read_state().x, emu.read_state().y) != (1, 3):
        _step_to_live(emu, ex, house, 1, 3)
    if (emu.read_state().x, emu.read_state().y) != (1, 3):
        return False
    emu.press("right")
    emu.tick(20)
    _press10(emu, "a")
    for _ in range(400):
        if _bag_count(emu, ITEM_HM02_FLY) >= 1:
            break
        if emu.read_state().in_dialog:
            _press10(emu, "a")
        emu.tick(25)
    _close_menus(emu)
    return _bag_count(emu, ITEM_HM02_FLY) == 1 and \
        _event_flag(emu, EV_GOT_HM02)


def teach_fly(emu: PyBoyEmulator, ex: Executor,
              pc_map: int) -> bool:
    """Teach HM02 FLY to the boxed PIDGEY, verify-first
    (probe160f): Pidgey's pokered tmhm includes FLY (pinned by
    TestFlyLeg), and the live pupil must have a free slot -
    the swap is DIGLETT out / PIDGEY in via pc_swap at the
    given Pokecenter, then _teach_hm_free_slot's strict-append
    assertion does the teach. End arrangement (Vlad's brief):
    Pidgey stays in the party as the fly carrier, Diglett
    stays boxed - reversible at any PC. Returns False
    untouched if the pupil somehow has four moves (choosing
    one to forget is the player's call)."""
    if _bag_count(emu, ITEM_HM02_FLY) < 1:
        return False
    pid = _party_index_of(emu, SPECIES_PIDGEY)
    if pid is None:
        if SPECIES_PIDGEY not in _box_species(emu):
            return False
        if not pc_swap(emu, ex, pc_map, SPECIES_DIGLETT,
                       SPECIES_PIDGEY):
            return False
        pid = _party_index_of(emu, SPECIES_PIDGEY)
        if pid is None:
            return False
    from agent.state import ADDR_PARTY_MONS
    mem = emu.pb.memory
    moves = [mem[ADDR_PARTY_MONS + 44 * pid + 8 + j]
             for j in range(4)]
    if MOVE_FLY in moves:
        return True
    if sum(1 for m in moves if m) >= 4:
        return False
    if not _teach_hm_free_slot(emu, pid, ITEM_HM02_FLY, MOVE_FLY):
        return False
    moves = [mem[ADDR_PARTY_MONS + 44 * pid + 8 + j]
             for j in range(4)]
    return MOVE_FLY in moves


# The Fly map's visited-town cycle in UP order, as read live
# off the header (probe160f8-f11): it opens on Pallet Town and
# UP steps through the visited flyable towns - Lavender BEFORE
# Vermilion, whatever TownMapOrder suggests. The header text
# itself (VRAM tilemap 0x9C00, row 0) is the steering signal:
# counted presses can double-step (one run landed Cinnabar).
FLY_TOWN_CYCLE = ["PALLET TOWN", "VIRIDIAN CITY", "PEWTER CITY",
                  "CERULEAN CITY", "LAVENDER TOWN",
                  "VERMILION CITY", "CELADON CITY",
                  "SAFFRON CITY", "FUCHSIA CITY",
                  "CINNABAR ISLAND", "INDIGO PLATEAU"]
FLY_TOWN_MAP_NAMES = {
    "PALLET TOWN": "PalletTown", "VIRIDIAN CITY": "ViridianCity",
    "PEWTER CITY": "PewterCity", "CERULEAN CITY": "CeruleanCity",
    "LAVENDER TOWN": "LavenderTown",
    "VERMILION CITY": "VermilionCity",
    "CELADON CITY": "CeladonCity", "SAFFRON CITY": "SaffronCity",
    "FUCHSIA CITY": "FuchsiaCity",
    "CINNABAR ISLAND": "CinnabarIsland",
    "INDIGO PLATEAU": "IndigoPlateau",
}


def _fly_header_town(emu: PyBoyEmulator):
    """The town named in the Fly map's header, decoded from
    the 0x9C00 tilemap (Gen 1 font: A=0x80, space=0x7F)."""
    mem = emu.pb.memory
    row = bytes(mem[0x9C00 + i] for i in range(20))
    for name in FLY_TOWN_CYCLE:
        enc = bytes(0x7F if c == " " else 0x80 + ord(c) - 65
                    for c in name)
        if enc in row:
            return name
    return None


def fly_to(emu: PyBoyEmulator, ex: Executor,
           dest_town: str) -> bool:
    """FLY to a visited town (probe160f): party-menu field
    move, hand-driven - the backend's _field_move backs out
    with B presses afterwards, which CANCELS the Fly map, so
    it cannot be used for Fly. The destination is steered by
    reading the header town off the tilemap after every UP
    press (counted presses are not reliable: a held press can
    double-step). Success = the landing map read from RAM is
    the destination's map. Caller must be outdoors with a
    living Fly carrier in the party."""
    from agent.state import ADDR_MENU_TOP_Y, ADDR_CURRENT_MENU_ITEM
    world = ex.world
    dest_map = world.map_id(FLY_TOWN_MAP_NAMES[dest_town])
    if emu.read_state().map_id == dest_map:
        return True
    carrier = None
    submenu = 0
    from agent.state import ADDR_PARTY_MONS
    mem = emu.pb.memory
    field_ids = {15, MOVE_FLY, MOVE_SURF, MOVE_STRENGTH, 148}
    for i, mon in enumerate(emu.read_state().party):
        moves = [mem[ADDR_PARTY_MONS + 44 * i + 8 + j]
                 for j in range(4)]
        if MOVE_FLY in moves:
            carrier = i
            slot = moves.index(MOVE_FLY)
            submenu = sum(1 for m in moves[:slot]
                          if m in field_ids)
            break
    if carrier is None:
        return False

    def press10(b):
        emu.pb.button_press(b)
        emu.tick(10)
        emu.pb.button_release(b)
        emu.tick(45)

    def wait_top_y(want, rounds=80):
        for _ in range(rounds):
            if mem[ADDR_MENU_TOP_Y] == want:
                return True
            if emu.read_state().in_dialog:
                press10("a")
                continue
            emu.tick(30)
        return mem[ADDR_MENU_TOP_Y] == want

    def steer(want, rounds=30):
        for _ in range(rounds):
            cur = mem[ADDR_CURRENT_MENU_ITEM]
            if cur == want:
                return True
            press10("down" if cur < want else "up")
            emu.tick(20)
        return mem[ADDR_CURRENT_MENU_ITEM] == want

    press10("start")
    if not wait_top_y(2) or not steer(1):
        return False
    press10("a")                              # POKEMON
    if not wait_top_y(1) or not steer(carrier):
        return False
    press10("a")                              # the carrier
    emu.tick(80)
    if not steer(submenu):
        return False
    press10("a")                              # FLY
    emu.tick(400)
    for _ in range(24):
        if _fly_header_town(emu) == dest_town:
            break
        press10("up")
        emu.tick(90)
    if _fly_header_town(emu) != dest_town:
        press10("b")                          # back out cleanly
        return False
    press10("a")
    for _ in range(240):
        if emu.read_state().map_id == dest_map:
            return True
        emu.tick(25)
    return emu.read_state().map_id == dest_map


# -- Viridian Gym: the arrow floor + Giovanni (probe162) ----
# Ground truth compiled from pokered, not discovered live:
# the gym's arrow triggers are the RLE table
# ViridianGymArrowTilePlayerMovement in scripts/
# ViridianGym.asm - standing on a trigger cell makes the map
# script simulate PAD_DIR x N (StartSimulatingJoypadStates),
# so a slide is up to N steps in the arrow direction,
# stopped early by walls/sprites, and landing on another
# trigger chains. probe162e calibrated the semantics live:
# a single slide takes exactly N steps ((11, 2) RIGHT 6 ->
# (17, 2); (18, 2) DOWN 9 -> (18, 11), ~48 frames/step).
# Caveat (probe162c): a slide entered in immediate
# succession after another can stop short (a DOWN 9 chained
# straight off the (11, 2) ride ended 2 steps early at
# (18, 9)) - so the navigator below NEVER trusts a predicted
# endpoint: it re-reads the live cell after every action and
# re-plans. Trainer sprites are STAY, but a sight-engaged
# trainer WALKS UP to the player and stays on his walk-up
# cell after the fight (probe162b: T5 ended on (10, 4),
# sealing column 10) - sprite blocking is therefore read
# LIVE from wSpriteStateData2 (via _boulder_cell), never
# from the objects table, and if the road seals anyway the
# gym is exited and re-entered (sprites reset home; beaten
# trainers stay beaten and do not walk up again).
ITEM_TM27 = 0xE3              # add_tm FISSURE ; $E3
EV_VIRIDIAN_GYM_OPEN = 40
EV_GOT_TM27 = 80
EV_BEAT_VIRIDIAN_GYM_GIOVANNI = 81
EV_VIRIDIAN_GYM_TRAINERS = (82, 83, 84, 85, 86, 87, 88, 89)
VIRIDIAN_GYM_ARROWS = {
    (19, 11): ("up", 9), (19, 1): ("left", 8),
    (18, 2): ("down", 9), (11, 2): ("right", 6),
    (16, 10): ("down", 2), (4, 6): ("down", 7),
    (5, 13): ("right", 8), (4, 14): ("right", 9),
    (0, 15): ("up", 8), (1, 15): ("up", 6),
    (13, 16): ("left", 6), (13, 17): ("left", 12),
}
VIRIDIAN_GYM_STANCE = (2, 2)  # south of Giovanni; face up
# Giovanni's gym team (parties.asm GiovanniData, $FF
# per-mon levels): Rhyhorn 45 / Dugtrio 42 / Nidoqueen 44 /
# Nidoking 45 / Rhydon 50, with FISSURE as the LoneMove on
# mon index 4 (Rhydon, move slot 3 - data/trainers/
# special_moves.asm entry 8). Gen 1 OHKO moves fail when
# the user is slower than the target, so a fast lead
# (Blastoise Lv61+) is immune; never lead a slow mon into
# the Rhydon. The eight gym trainers (objects + headers +
# parties): T0 (12,7) CooltrainerM9 Sandslash/Dugtrio 39,
# T1 (11,11) Blackbelt6 Machop/Machoke 40, T2 (10,7)
# Tamer3 Rhyhorn 43, T3 (3,7) Blackbelt7 Machoke 43,
# T4 (13,5) CooltrainerM10 Rhyhorn 43, T5 (10,1)
# Blackbelt8 Machoke/Machop/Machoke 38, T6 (2,16) Tamer4
# Arbok/Tauros 39, T7 (6,5) CooltrainerM1 Nidorino/
# Nidoking 39. The walk-in road from the door crosses only
# T4's and T5's sight lines.
_GYM_DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0),
             "right": (1, 0)}


def _gym_sprite_cells(emu: PyBoyEmulator, gmap) -> set:
    """Live sprite cells on a gym map (player excluded)."""
    st = emu.read_state()
    here = (st.x, st.y)
    cells = set()
    for k in range(1, 16):
        c = _boulder_cell(emu, k)
        if c != here and gmap.in_bounds(*c) \
                and gmap.tile_walkable(*c):
            cells.add(c)
    return cells


def _gym_slide_end(gmap, cell, blocked, learned) -> tuple:
    """Resolve the arrow chain from `cell` on the compiled
    grid: slide up to N steps per trigger, walls / blocked /
    learned cells stop a slide, triggers chain."""
    seen = set()
    while cell in VIRIDIAN_GYM_ARROWS and cell not in seen:
        seen.add(cell)
        d, n = VIRIDIAN_GYM_ARROWS[cell]
        dx, dy = _GYM_DIRS[d]
        for _ in range(n):
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (gmap.in_bounds(*nxt) and gmap.walk[nxt[1]][nxt[0]]) \
                    or nxt in blocked or nxt in learned:
                break
            cell = nxt
    return cell


def _gym_bfs(gmap, start, goal, blocked, learned):
    """BFS over walk steps + compiled slides. Returns a list
    of (direction, stepped_cell, predicted_end) actions."""
    from collections import deque
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == goal:
            break
        for d, (dx, dy) in _GYM_DIRS.items():
            stepped = (c[0] + dx, c[1] + dy)
            if not (gmap.in_bounds(*stepped)
                    and gmap.walk[stepped[1]][stepped[0]]) \
                    or stepped in blocked or stepped in learned:
                continue
            end = _gym_slide_end(gmap, stepped, blocked, learned)
            if end not in prev:
                prev[end] = (c, d, stepped)
                q.append(end)
    if goal not in prev:
        return None
    actions = []
    c = goal
    while prev[c] is not None:
        p, d, stepped = prev[c]
        actions.append((d, stepped, c))
        c = p
    return list(reversed(actions))


def _gym_settle(emu: PyBoyEmulator) -> bool:
    """Ride dialogs and hardened-drive any battle until the
    player is free. Returns True if a battle was driven."""
    fought = False
    quiet = 0
    for _ in range(400):
        st = emu.read_state()
        if st.in_battle:
            fought = True
            _drive_battle_hardened(emu)
            quiet = 0
            continue
        if st.in_dialog:
            quiet = 0
            _press10(emu, "a")
        else:
            quiet += 1
            if quiet >= 6:
                break
        emu.tick(25)
    return fought


def _gym_step(emu: PyBoyEmulator, gym_map: int, d: str) -> None:
    """One press-step inside the gym; if it lands on an arrow
    trigger, wait out the simulated-pad slide."""
    emu.press(d)
    _explore_settle(emu)
    st = emu.read_state()
    pos = (st.x, st.y)
    if st.map_id != gym_map or pos not in VIRIDIAN_GYM_ARROWS \
            or st.in_battle:
        return
    stable = 0
    for _ in range(160):
        st = emu.read_state()
        if st.in_battle or st.map_id != gym_map:
            return
        if (st.x, st.y) == pos:
            stable += 1
            if stable >= 8:
                return
        else:
            stable = 0
            pos = (st.x, st.y)
        emu.tick(15)


def _gym_go(emu: PyBoyEmulator, ex: Executor, gym_map: int,
            gmap, goal, learned: set) -> bool:
    """Replanning gym navigator: BFS (compiled arrows + live
    sprites + learned blocks), one action at a time, fighting
    whatever sight-engages. A step that moves nothing teaches
    `learned` the stepped cell (a walked-up trainer's body).
    """
    for _ in range(240):
        st = emu.read_state()
        if st.map_id != gym_map:
            return False
        if st.in_battle or st.in_dialog:
            _gym_settle(emu)
            st = emu.read_state()
            if st.map_id != gym_map:
                return False
        if (st.x, st.y) == goal:
            return True
        blocked = _gym_sprite_cells(emu, gmap)
        path = _gym_bfs(gmap, (st.x, st.y), goal, blocked,
                        learned)
        if not path:
            return False
        d, stepped, _end = path[0]
        before = (st.x, st.y)
        _gym_step(emu, gym_map, d)
        st2 = emu.read_state()
        if (st2.x, st2.y) == before and st2.map_id == gym_map \
                and not st2.in_battle and not st2.in_dialog:
            learned.add(stepped)
    st = emu.read_state()
    return (st.x, st.y) == goal and st.map_id == gym_map


def _gym_exit(emu: PyBoyEmulator, ex: Executor, gym_map: int,
              gmap, learned: set) -> bool:
    if not _gym_go(emu, ex, gym_map, gmap, (17, 16), learned):
        _gym_go(emu, ex, gym_map, gmap, (16, 16), learned)
    for _ in range(8):
        emu.press("down")
        _explore_settle(emu)
        if emu.read_state().map_id != gym_map:
            return True
    return emu.read_state().map_id != gym_map


def viridian_gym(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The Earth Badge (probe162): FLY to Viridian (when not
    already there), deposit the GOOD ROD at the Viridian PC
    if the bag is full (Giovanni's TM27 payout needs the
    20th stack free - the Sabrina lesson), cross the arrow
    floor fighting only the sight engagements on the road
    (T4, T5), beat GIOVANNI, ride the badge + TM27 ceremony,
    exit and heal. Verified from RAM at the end: Earth badge
    bit (8 badges), EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI,
    EVENT_GOT_TM27, TM27 x1 in the bag, party at full HP.
    Stop rule (Vlad's): two blackouts to Giovanni end the
    leg in failure - no grinding past him. Fences: never
    west to Route 22, never into Victory Road, no catches,
    no TM teaching, no stat items; the gym's REVIVE ball is
    deliberately NOT collected (bag discipline)."""
    world = ex.world
    gym = world.map_id("ViridianGym")
    city = world.map_id("ViridianCity")
    pc = world.map_id("ViridianPokecenter")
    gmap = next(m for m in world.maps.values()
                if m.name == "ViridianGym")
    if "Earth" in emu.read_state().badges:
        return _bag_count(emu, ITEM_TM27) >= 1
    # -- travel: normalise to Viridian City streets
    st = emu.read_state()
    if st.map_id == world.map_id("FuchsiaPokecenter"):
        for door in ((3, 7), (4, 7)):
            try:
                ex._walk_to(st.map_id, door)
            except StuckError:
                continue
            for _ in range(6):
                emu.press("down")
                _explore_settle(emu)
                if emu.read_state().map_id != st.map_id:
                    break
            if emu.read_state().map_id != st.map_id:
                break
        st = emu.read_state()
    if st.map_id not in (city, gym):
        if not fly_to(emu, ex, "VIRIDIAN CITY"):
            return False
        st = emu.read_state()
    # -- bag room for TM27 (deposit the superseded Good Rod)
    if st.map_id == city and len(_bag_items(emu)) >= 20 \
            and _bag_count(emu, ITEM_GOOD_ROD) >= 1 \
            and _bag_count(emu, ITEM_TM27) == 0:
        if not _ride_or_cross(emu, ex, pc):
            return False
        if not deposit_item(emu, ex, pc, ITEM_GOOD_ROD):
            return False
        if _bag_count(emu, ITEM_GOOD_ROD) != 0 \
                or len(_bag_items(emu)) != 19:
            return False
        for door in ((3, 7), (4, 7)):
            try:
                ex._walk_to(pc, door)
            except StuckError:
                continue
            for _ in range(6):
                emu.press("down")
                _explore_settle(emu)
                if emu.read_state().map_id != pc:
                    break
            if emu.read_state().map_id != pc:
                break
        if emu.read_state().map_id != city:
            return False
    # -- the gym (two Giovanni blackouts = stop, per Vlad)
    for attempt in range(2):
        if emu.read_state().map_id != gym:
            if not _ride_or_cross(emu, ex, gym):
                return False
        learned: set = set()
        if not _gym_go(emu, ex, gym, gmap,
                       VIRIDIAN_GYM_STANCE, learned):
            # a walked-up trainer can seal the road; sprites
            # reset home on re-entry, beaten trainers stay put
            if not _gym_exit(emu, ex, gym, gmap, learned):
                return False
            if not _ride_or_cross(emu, ex, gym):
                return False
            learned = set()
            if not _gym_go(emu, ex, gym, gmap,
                           VIRIDIAN_GYM_STANCE, learned):
                return False
        emu.press("up")
        emu.tick(20)
        _press10(emu, "a")
        in_battle = False
        for _ in range(300):
            st = emu.read_state()
            if st.in_battle:
                in_battle = True
                break
            if st.in_dialog:
                _press10(emu, "a")
            emu.tick(25)
        if not in_battle:
            return False
        _drive_battle_hardened(emu)
        # the ceremony rides AFTER the fight: badge + TM27
        quiet = 0
        for _ in range(600):
            st = emu.read_state()
            if st.in_dialog:
                quiet = 0
                _press10(emu, "a")
            else:
                quiet += 1
                if quiet >= 8 and _event_flag(
                        emu, EV_BEAT_VIRIDIAN_GYM_GIOVANNI):
                    break
            emu.tick(25)
        if _event_flag(emu, EV_BEAT_VIRIDIAN_GYM_GIOVANNI):
            break
        # blacked out (or bounced): the wake-up leaves the
        # party healed at the PC; one retry is allowed
        if attempt == 1:
            return False
        if emu.read_state().map_id == gym:
            _gym_exit(emu, ex, gym, gmap, set())
    st = emu.read_state()
    if "Earth" not in st.badges or len(st.badges) != 8 \
            or not _event_flag(emu, EV_GOT_TM27) \
            or _bag_count(emu, ITEM_TM27) != 1:
        return False
    # -- out, heal, finish in Viridian City
    if emu.read_state().map_id == gym:
        if not _gym_exit(emu, ex, gym, gmap, set()):
            return False
    if not _ride_or_cross(emu, ex, pc):
        return False
    if not _heal_at_pc(emu, ex, pc):
        return False
    st = emu.read_state()
    return st.map_id == city and len(st.badges) == 8 \
        and all(m.hp == m.max_hp for m in st.party)


# ---------------------------------------------------------------------
# The road to Indigo (probe165, 2026-11-12/13): Vlad's Silph
# playbook for the Route 22 rival - grind first, then re-fight -
# then Route 23 and Victory Road in one unbroken visit, ending
# healed at Indigo Plateau. The League itself is a separate leg
# (Vlad's team/item calls gate it).
#
# Fences (Vlad's, hard): no catches (wilds fled; the Master Ball
# is never thrown), no TM teaching, no stat items, no party
# swaps or evolution projects; Moltres and Articuno are never
# engaged (the Victory Road route stays south/east of the
# Moltres room and never enters its sight). In-game purchases:
# the probe164 prep stock only (Super Potion x6, Revive x2,
# Full Heal x2 at Fuchsia, 8,400 yen), reported to the yen.

SP_GYARADOS = 0x16
SP_GROWLITHE = 0x21
SP_ALAKAZAM = 0x95
SP_VENUSAUR = 0x9A
SP_BLASTOISE = 0x1C
SP_ARCANINE = 0x14
SP_ODDISH = 0xB9

MOVE_HYDRO_PUMP = 56
MOVE_WATER_GUN = 55
MOVE_BUBBLE = 145
MOVE_BITE = 44
MOVE_EMBER = 52
MOVE_SLEEP_POWDER = 79

EV_BEAT_RIVAL2_R22 = 1318
EV_RIVAL2_WANTS = 1319
EV_VR1_BOULDER = 2327
EV_VR2_SWITCH1 = 1336
EV_VR2_SWITCH2 = 1343
EV_VR3_HOLE = 1638

ITEM_HELIX_FOSSIL = 0x2A
ITEM_SILPH_SCOPE = 0x48
ITEM_SUPER_POTION = 0x13
ITEM_REVIVE = 0x35
ITEM_FULL_HEAL = 0x34

FUCHSIA_MART_LIST = [0x02, 0x03, ITEM_SUPER_POTION, ITEM_REVIVE,
                     ITEM_FULL_HEAL, 0x1F]  # ULTRA, GREAT, SP, REV, FH, SREPEL


def _exit_building(emu: PyBoyEmulator, ex: Executor,
                   map_id: int) -> bool:
    """Walk out of a building via its known door mats
    (probe164a; standard Pokecenter/mart south doors)."""
    for door in ((3, 7), (4, 7), (7, 11), (8, 11)):
        try:
            ex._walk_to(map_id, door)
        except StuckError:
            continue
        for _ in range(6):
            emu.press("down")
            _explore_settle(emu)
            if emu.read_state().map_id != map_id:
                return True
    return emu.read_state().map_id != map_id


def _battle_enemy(emu: PyBoyEmulator):
    """(species, hp, max_hp, level) of the enemy in battle,
    read from WRAM (probe164a)."""
    mem = emu.pb.memory
    sp = mem[ADDR_ENEMY_SPECIES]
    hp = (mem[0xCFE6] << 8) | mem[0xCFE7]
    mx = (mem[0xD007] << 8) | mem[0xD008]
    lv = mem[0xD006]
    return sp, hp, mx, lv


def _ride_menu(emu: PyBoyEmulator) -> str:
    """Ride battle text/menus to the next main-menu decision,
    DECLINING the trainer-battle shift prompt (the S26 driver
    fix from probe164a): when the enemy's mon faints the game
    asks 'Will VLAD change POKEMON?'; A presses answer YES and
    open the party list, where A re-selects the active mon
    forever. One B at that screen cancels the shift cleanly.
    A 0-HP battle mon at the same cursor is instead the faint
    screen, which _ride_to_battle_menu's picker handles."""
    mem = emu.pb.memory
    for _ in range(400):
        st = emu.read_state()
        if not st.in_battle:
            return "over"
        if mem[ADDR_MENU_TOP_Y] == 1:
            bhp = (mem[ADDR_BATTLE_MON_HP] << 8) | \
                mem[ADDR_BATTLE_MON_HP + 1]
            if bhp == 0:
                if _ride_to_battle_menu(emu) == "over":
                    return "over"
                continue
            _press10(emu, "b")
            emu.tick(45)
            continue
        if _battle_menu_sig(emu) == (14, 0, 9):
            return "menu"
        if st.in_dialog:
            _press10(emu, "a")
            emu.tick(20)
            continue
        emu.tick(25)
    return "menu" if emu.read_state().in_battle else "over"


def _mon_of(emu: PyBoyEmulator, species_id: int):
    for i, m in enumerate(emu.read_state().party):
        if m.species_id == species_id:
            return i, m
    return None, None


def route22_prep(emu: PyBoyEmulator, ex: Executor):
    """The S26 road prep (probe164a): at Viridian, deposit the
    S.S. TICKET / HELIX FOSSIL / SILPH SCOPE (bag room), FLY to
    Fuchsia, buy Super Potion x6 + Revive x2 + Full Heal x2
    (8,400 yen - the only purchase of the road legs), FLY back.
    Returns the total spend, or None on failure."""
    world = ex.world
    pc = world.map_id("ViridianPokecenter")
    mart = world.map_id("FuchsiaMart")
    money0 = emu.read_state().money
    if not _ride_or_cross(emu, ex, pc):
        return None
    for item in (ITEM_SS_TICKET, ITEM_HELIX_FOSSIL,
                 ITEM_SILPH_SCOPE):
        if _bag_count(emu, item) >= 1:
            if not deposit_item(emu, ex, pc, item):
                return None
    if len(_bag_items(emu)) != 17:
        return None
    if not _exit_building(emu, ex, pc):
        return None
    if not fly_to(emu, ex, "FUCHSIA CITY"):
        return None
    if not _ride_or_cross(emu, ex, mart):
        return None
    for item, want in ((ITEM_SUPER_POTION, 6), (ITEM_REVIVE, 2),
                       (ITEM_FULL_HEAL, 2)):
        idx = FUCHSIA_MART_LIST.index(item)
        got = buy_mart_item(emu, ex, mart, idx, item, want)
        if got < want:
            return None
    if not _exit_building(emu, ex, mart):
        return None
    if not fly_to(emu, ex, "VIRIDIAN CITY"):
        return None
    return money0 - emu.read_state().money


# --- the grind (probe165a): Pokemon Mansion 1F, lead-and-solo ---

def _enter_mansion(emu: PyBoyEmulator, ex: Executor) -> bool:
    world = ex.world
    cin = world.map_id("CinnabarIsland")
    m1 = world.map_id("PokemonMansion1F")
    if emu.read_state().map_id == m1:
        return True
    if emu.read_state().map_id != cin:
        return False
    _surf_walk_to(emu, ex, cin, (6, 3))
    return emu.read_state().map_id == m1


def _exit_mansion(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The mansion's street door is the WEST room's south mats
    (4-7,27) - entry lands at (5,27) and one step down exits
    to Cinnabar (6,4). (26,27) is the 1F strip's east door,
    reachable only via the B1F fall route."""
    world = ex.world
    m1 = world.map_id("PokemonMansion1F")
    cin = world.map_id("CinnabarIsland")
    if emu.read_state().map_id != m1:
        return emu.read_state().map_id == cin
    _walk_fighting(emu, ex, m1, (5, 27), attempts=25,
                   flee_wild=True)
    if emu.read_state().map_id == cin:
        return True
    _overlay_nav(emu, ex, m1, (5, 26),
                 _mansion_overlay(emu, world, m1))
    for _ in range(8):
        if emu.read_state().map_id == cin:
            return True
        emu.press("down")
        _explore_settle(emu)
    return emu.read_state().map_id == cin


def _grind_heal_trip(emu: PyBoyEmulator, ex: Executor,
                     species_id: int) -> bool:
    """Mansion -> Cinnabar PC heal -> re-enter -> re-lead.
    Free heals only: the bought stock is reserved for the
    road run (the Silph pattern)."""
    world = ex.world
    pc = world.map_id("CinnabarPokecenter")
    if not _exit_mansion(emu, ex):
        return False
    if not _ride_or_cross(emu, ex, pc):
        return False
    if not heal_at_pokecenter(emu, ex, pc):
        return False
    if emu.read_state().map_id == pc:
        _exit_building(emu, ex, pc)
    if emu.read_state().map_id == pc:
        return False
    if not party_lead(emu, species_id):
        return False
    return _enter_mansion(emu, ex)


_GRIND_PACE_ENDS = [(2, 26), (9, 26)]


def _grind_one(emu: PyBoyEmulator, ex: Executor,
               species_id: int, target: int):
    """Lead-and-solo grind of one species to `target` in the
    Mansion 1F west room (row-26 pacing; Gen 1 pays a fainted
    participant nothing). HP gate 55% / damaging-PP gate 10
    trigger free PC heal trips instead of item spend. Grind
    blackouts are recoverable and do NOT count toward any
    rival stop rule. Returns (ok, fights, trips, blackouts)
    (probe165a)."""
    world = ex.world
    m1 = world.map_id("PokemonMansion1F")
    fights = 0
    trips = 0
    blackouts = 0
    leg = 0
    while True:
        idx, mon = _mon_of(emu, species_id)
        if mon is None:
            return False, fights, trips, blackouts
        if mon.level >= target:
            break
        if mon.hp == 0 or mon.hp < 0.55 * mon.max_hp \
                or _damaging_pp(emu, idx) <= 10:
            trips += 1
            if not _grind_heal_trip(emu, ex, species_id):
                return False, fights, trips, blackouts
            continue
        if emu.read_state().map_id != m1:
            if not _enter_mansion(emu, ex):
                return False, fights, trips, blackouts
        dest = _GRIND_PACE_ENDS[leg % 2]
        leg += 1
        _ok, fought = _walk_fighting(emu, ex, m1, dest,
                                     attempts=25)
        fights += fought
        emu.wait_free(max_rounds=8, stable_needed=2)
        st = emu.read_state()
        idx, mon = _mon_of(emu, species_id)
        if all(m.hp == 0 for m in st.party):
            blackouts += 1
            # the game warps to the last PC, healed
            for _ in range(20):
                if emu.read_state().map_id != m1:
                    break
                emu.tick(30)
            pc = world.map_id("CinnabarPokecenter")
            if emu.read_state().map_id == pc:
                _exit_building(emu, ex, pc)
            if not party_lead(emu, species_id):
                return False, fights, trips, blackouts
            if not _enter_mansion(emu, ex):
                return False, fights, trips, blackouts
            continue
    return True, fights, trips, blackouts


def mansion_grind(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The grind (probe165a): FLY to Cinnabar, enter the
    Mansion (1F pool: Koffing/Ponyta/Growlithe/Grimer/
    Weezing/Muk, ~610 avg solo XP - the richest unfought pool
    reachable without passing the rival), lead-and-solo
    BLASTOISE 62 -> 66 (bulk for the Gyarados Dragon Rage tax
    and Alakazam's Psychic) and ARCANINE 39 -> 48 (its Bite
    becomes the real Alakazam answer). The typed driver uses
    the last damaging move with PP, so Hydro Pump's 5 PP
    stay untouched. Entry: healed, anywhere Fly can leave
    (the probe ran it from the Viridian prep checkpoint)."""
    emu.wait_free(max_rounds=8, stable_needed=2)
    if not fly_to(emu, ex, "CINNABAR ISLAND"):
        return False
    if not _enter_mansion(emu, ex):
        return False
    ok1 = _grind_one(emu, ex, SP_BLASTOISE, 66)
    if not ok1[0]:
        return False
    if not party_lead(emu, SP_ARCANINE):
        return False
    ok2 = _grind_one(emu, ex, SP_ARCANINE, 48)
    if not ok2[0]:
        return False
    print(f"grind done: blastoise={ok1} arcanine={ok2}",
          flush=True)
    return True


# --- the Route 22 re-fight (probe165c) ---

def _fight_route22_rival(emu: PyBoyEmulator,
                         ex: Executor) -> bool:
    """The winning Route 22 rival policy at the ground levels
    (Blastoise 66 / Arcanine 48), per-enemy:
    * Gyarados is BLASTOISE's fight: Hydro Pump cap 2 (fewer
      turns = less flat-40 Dragon Rage tax), then Water Gun.
    * Alakazam is ARCANINE's fight: it stays in and Bites
      (~85-100 vs Alakazam's paper Defense). NEVER switch
      Blastoise in - the S26 loss was a switch-turn Psychic
      at 63 HP, then Arcanine died the same way.
    * Growlithe: Arcanine Bite duty (switch at HP >= 100).
    * Venusaur: Oddish Sleep Powder, then Ember (Arcanine)
      or Water Gun on the sleeper.
    * Elsewhere: Hydro Pump economy (cap 2 per big enemy, 5
      total), Water Gun/Bubble for Blastoise, Bite for
      Arcanine; the tail cascade is moveset-aware (the S26
      watchdog fix) with a 30-turn no-progress watchdog.
    Returns EVENT_BEAT_ROUTE22_RIVAL_2ND read from RAM."""
    mem = emu.pb.memory
    hydro_cap = {SP_GYARADOS: 2, SP_ALAKAZAM: 2,
                 SP_VENUSAUR: 2}
    hydro_used = {k: 0 for k in hydro_cap}
    hydro_total = [0]
    seen = [None]
    oddish_spent = [False]
    last_sig = [None]
    stale = [0]
    for _turn in range(90):
        st = emu.read_state()
        if not st.in_battle:
            break
        r = _ride_menu(emu)
        if r == "over" or not emu.read_state().in_battle:
            break
        sp, hp, _mx, _lv = _battle_enemy(emu)
        sig = (sp, hp,
               sum(m.hp for m in emu.read_state().party))
        if sig == last_sig[0]:
            stale[0] += 1
            if stale[0] >= 30:
                break    # watchdog: no progress in 30 turns
        else:
            stale[0] = 0
            last_sig[0] = sig
        if sp != seen[0]:
            seen[0] = sp
        active = mem[ADDR_BATTLE_MON_SPECIES]
        st = emu.read_state()
        party = {m.species_id: m for m in st.party}
        blast = party.get(SP_BLASTOISE)
        arc = party.get(SP_ARCANINE)
        # --- Gyarados: Blastoise's fight
        if sp == SP_GYARADOS and active != SP_BLASTOISE \
                and blast and blast.hp > 0:
            idx = _party_index_of(emu, SP_BLASTOISE)
            _battle_switch_to(emu, idx)
            continue
        # --- Venusaur sleep plan
        if sp == SP_VENUSAUR and not oddish_spent[0]:
            odd = party.get(SP_ODDISH)
            if _enemy_asleep(emu):
                if active == SP_ARCANINE:
                    _battle_use_move(emu, MOVE_EMBER)
                elif active == SP_BLASTOISE:
                    _battle_use_move(emu, MOVE_WATER_GUN)
                elif arc and arc.hp > 30:
                    idx = _party_index_of(emu, SP_ARCANINE)
                    _battle_switch_to(emu, idx)
                elif blast and blast.hp > 0:
                    idx = _party_index_of(emu, SP_BLASTOISE)
                    _battle_switch_to(emu, idx)
                else:
                    _battle_use_move(emu, 15)
                continue
            if active == SP_ARCANINE:
                if odd and odd.hp > 0:
                    idx = _party_index_of(emu, SP_ODDISH)
                    _battle_switch_to(emu, idx)
                else:
                    idx = _party_index_of(emu, SP_BLASTOISE)
                    _battle_switch_to(emu, idx)
                continue
            if active != SP_ODDISH and odd and odd.hp > 0:
                idx = _party_index_of(emu, SP_ODDISH)
                _battle_switch_to(emu, idx)
                continue
            if active == SP_ODDISH:
                ok = _battle_use_move(emu, MOVE_SLEEP_POWDER)
                if not ok:
                    oddish_spent[0] = True
                continue
            oddish_spent[0] = True
        # --- per-active policy
        if active == SP_BLASTOISE:
            if sp == SP_GROWLITHE and arc and arc.hp >= 100:
                idx = _party_index_of(emu, SP_ARCANINE)
                _battle_switch_to(emu, idx)
                continue
            if sp in hydro_cap \
                    and hydro_used[sp] < hydro_cap[sp] \
                    and hydro_total[0] < 5 \
                    and blast and blast.pp[0] > 0:
                ok = _battle_use_move(emu, MOVE_HYDRO_PUMP)
                hydro_used[sp] += 1
                hydro_total[0] += 1
                continue
            bi = _party_index_of(emu, SP_BLASTOISE)
            mv = MOVE_WATER_GUN \
                if emu.read_state().party[bi].pp[3] > 0 \
                else MOVE_BUBBLE
            _battle_use_move(emu, mv)
            continue
        if active == SP_ARCANINE:
            if sp == SP_VENUSAUR:
                _battle_use_move(emu, MOVE_EMBER)
            else:
                # Alakazam included: stay in and Bite.
                _battle_use_move(emu, MOVE_BITE)
            continue
        # --- Alakazam with a tail mon active: get a lead in
        if sp == SP_ALAKAZAM:
            if arc and arc.hp > 60:
                idx = _party_index_of(emu, SP_ARCANINE)
                if _battle_switch_to(emu, idx):
                    continue
            if blast and blast.hp > 0:
                idx = _party_index_of(emu, SP_BLASTOISE)
                if _battle_switch_to(emu, idx):
                    continue
        mon = next((m for m in emu.read_state().party
                    if m.species_id == active), None)
        if mon is not None:
            dmg = [mv for mv in mon.moves
                   if mv not in (39, 45, 46, 77, 78, 103,
                                 148)]
            mv = dmg[-1] if dmg else mon.moves[0]
            _battle_use_move(emu, mv)
        else:
            emu.tick(30)
    return _event_flag(emu, EV_BEAT_RIVAL2_R22)


def route22_rival(emu: PyBoyEmulator, ex: Executor) -> bool:
    """The re-fight (probe165c): from the Mansion after the
    grind - exit, heal at the Cinnabar PC, re-lead Blastoise,
    FLY to Viridian, cross to Route 22 and fight the rival at
    the trigger (29,5). Won on attempt 1 at the new levels in
    the probe (Gyarados took zero damage-dealt; Alakazam
    answered by Arcanine's Bite; Venusaur slept first try).
    Stop rule (Vlad's, Silph precedent): two blackouts to the
    rival AT THESE LEVELS end the leg - the caller owns the
    retry decision; this function makes one attempt and
    reports the RAM event flag honestly."""
    emu.wait_free(max_rounds=8, stable_needed=2)
    world = ex.world
    pc = world.map_id("CinnabarPokecenter")
    r22 = world.map_id("Route22")
    if emu.read_state().map_id == \
            world.map_id("PokemonMansion1F"):
        if not _exit_mansion(emu, ex):
            return False
    if not _ride_or_cross(emu, ex, pc):
        return False
    if not heal_at_pokecenter(emu, ex, pc):
        return False
    if emu.read_state().map_id == pc:
        _exit_building(emu, ex, pc)
    if not party_lead(emu, SP_BLASTOISE):
        return False
    if not fly_to(emu, ex, "VIRIDIAN CITY"):
        return False
    if not _ride_or_cross(emu, ex, r22):
        return False
    _walk_fighting(emu, ex, r22, (29, 5), attempts=40,
                   flee_wild=True)
    for _ in range(400):
        st = emu.read_state()
        if st.in_battle:
            break
        if st.in_dialog:
            _press10(emu, "a")
        emu.tick(25)
    if not emu.read_state().in_battle:
        return False
    _fight_route22_rival(emu, ex)
    # ride the exit scene out
    for _ in range(400):
        st = emu.read_state()
        if not st.in_dialog and not st.in_battle:
            emu.tick(60)
            if not emu.read_state().in_dialog:
                break
        if st.in_dialog:
            _press10(emu, "a")
        emu.tick(25)
    # the beat-rival event is set by the post-battle map
    # script DURING the exit scene, so the verdict is read
    # only now (probe166's replay read it inside the fight
    # and raced it: event True, verdict False)
    return _event_flag(emu, EV_BEAT_RIVAL2_R22)


# --- Route 23 (probe165d) ---

_R23_GUARDS = [(8, 136), (8, 119), (12, 105), (10, 56),
               (4, 35)]


def _r23_ride_dialog(emu: PyBoyEmulator,
                     max_rounds: int = 200) -> bool:
    for _ in range(max_rounds):
        st = emu.read_state()
        if st.in_battle:
            _drive_battle_typed(emu, flee_wild=True)
            continue
        if not st.in_dialog:
            emu.tick(45)
            if not emu.read_state().in_dialog:
                return True
            continue
        _press10(emu, "a")
        emu.tick(20)
    return not emu.read_state().in_dialog


def _r23_walk(emu: PyBoyEmulator, ex: Executor,
              goal) -> bool:
    """Walk toward a Route 23 cell; on a stall next to a
    badge guard, talk to him (the badge check) and retry."""
    r23 = ex.world.map_id("Route23")
    for _attempt in range(14):
        st = emu.read_state()
        if st.map_id != r23:
            return False
        if (st.x, st.y) == goal:
            return True
        _walk_fighting(emu, ex, r23, goal, attempts=10,
                       flee_wild=True)
        st = emu.read_state()
        if (st.x, st.y) == goal:
            return True
        if st.in_battle:
            _drive_battle_typed(emu, flee_wild=True)
            continue
        talked = False
        for gx, gy in _R23_GUARDS:
            if abs(st.x - gx) + abs(st.y - gy) == 1:
                _face_and_talk(emu, (gx, gy))
                _r23_ride_dialog(emu)
                talked = True
                break
        if not talked:
            dx = goal[0] - st.x
            dy = goal[1] - st.y
            d = "down" if dy > 0 else "up" if dy < 0 else \
                "right" if dx > 0 else "left"
            emu.press(d)
            _explore_settle(emu)
            _r23_ride_dialog(emu)
    st = emu.read_state()
    return (st.x, st.y) == goal


def route23_passage(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Route 23 in one visit (probe165d): heal at the
    Viridian PC after the rival, cross Route 22 west, through
    the Route 22 Gate, then Route 23's badge-guard corridor
    (all 8 badges held) and the surf stretch (Lapras), ending
    in front of the Victory Road door at (4,32). Entering
    Route 23 resets Victory Road's boulder events, so after
    this function the run goes through VR in one unbroken
    visit - no flying out."""
    emu.wait_free(max_rounds=8, stable_needed=2)
    world = ex.world
    pc = world.map_id("ViridianPokecenter")
    r22 = world.map_id("Route22")
    gate = world.map_id("Route22Gate")
    r23 = world.map_id("Route23")
    if not _ride_or_cross(emu, ex, pc):
        return False
    if not heal_at_pokecenter(emu, ex, pc):
        return False
    if emu.read_state().map_id == pc:
        _exit_building(emu, ex, pc)
    if not _ride_or_cross(emu, ex, r22):
        return False
    if emu.read_state().map_id != r23:
        # west end of R22 -> gatehouse south door (8,5),
        # stepped onto from (8,6)
        _walk_fighting(emu, ex, r22, (8, 6), attempts=30,
                       flee_wild=True)
        for _ in range(10):
            if emu.read_state().map_id in (gate, r23):
                break
            emu.press("up")
            _explore_settle(emu)
            _r23_ride_dialog(emu)
        if emu.read_state().map_id == gate:
            _walk_fighting(emu, ex, gate, (4, 1), attempts=20,
                           flee_wild=True)
            for _ in range(10):
                if emu.read_state().map_id == r23:
                    break
                emu.press("up")
                _explore_settle(emu)
                _r23_ride_dialog(emu)
    if emu.read_state().map_id != r23:
        return False
    for goal in [(8, 133), (8, 116), (10, 111), (12, 104)]:
        if (emu.read_state().x, emu.read_state().y) == goal:
            continue
        if not _r23_walk(emu, ex, goal):
            return False
    # the swim: shore (12,103) -> north shore (11,70)
    if not _surf_walk_to(emu, ex, r23, (11, 70)):
        return False
    for goal in [(10, 57), (8, 50), (7, 45), (3, 40),
                 (4, 33), (4, 32)]:
        if not _r23_walk(emu, ex, goal):
            return False
    st = emu.read_state()
    return st.map_id == r23 and (st.x, st.y) == (4, 32)


# --- Victory Road (probe165e + the probe165 endgame) ---

_VR_BOULDER_OBJS = {}
_VR_BOULDER_STARTS = {}


def _vr_tables(world):
    if not _VR_BOULDER_OBJS:
        vr1 = world.map_id("VictoryRoad1F")
        vr2 = world.map_id("VictoryRoad2F")
        vr3 = world.map_id("VictoryRoad3F")
        _VR_BOULDER_OBJS.update(
            {vr1: (5, 6, 7), vr2: (11, 12, 13),
             vr3: (7, 8, 9, 10)})
        _VR_BOULDER_STARTS.update(
            {vr1: {(5, 15), (14, 2), (2, 10)},
             vr2: {(4, 14), (5, 5), (23, 16)},
             vr3: {(22, 3), (13, 12), (24, 10), (22, 15)}})
    return _VR_BOULDER_OBJS


def _vr_clear_battle(emu: PyBoyEmulator):
    if emu.read_state().in_battle:
        _drive_battle_typed(emu, flee_wild=True)
        emu.wait_free(max_rounds=8, stable_needed=2)


def _vr_thaw(emu: PyBoyEmulator):
    """Unstick a frozen player: after some interruptions the
    facing/animation state eats step inputs until a turn
    cycle passes."""
    for d in ("down", "left", "up"):
        emu.press(d)
        _explore_settle(emu)


def _vr_walk(emu: PyBoyEmulator, ex: Executor, map_id: int,
             goal, blocked=frozenset(), tries: int = 8) -> bool:
    """_surf_walk_to with battle clears and retries. The
    floor's boulder sprites are always blocked (solid, and
    they never move on their own); trainers are NOT blocked
    - the geometry avoids their sights, and where a fight is
    planned the walker paths through the trainer's cell,
    which he vacates when he engages (probe165e)."""
    _vr_tables(ex.world)
    for _ in range(tries):
        st = emu.read_state()
        if st.map_id != map_id:
            return False
        if (st.x, st.y) == goal:
            return True
        sprites = frozenset(
            _boulder_cell(emu, k)
            for k in _VR_BOULDER_OBJS.get(map_id, ()))
        sprites = frozenset(c for c in sprites
                            if c[0] >= 0
                            and c != (st.x, st.y))
        before = (st.x, st.y)
        _surf_walk_to(emu, ex, map_id, goal,
                      blocked0=blocked | sprites)
        st2 = emu.read_state()
        if (st2.x, st2.y) == before and not st2.in_battle:
            _vr_thaw(emu)
        _vr_clear_battle(emu)
        st = emu.read_state()
        if st.map_id != map_id:
            return False
        if (st.x, st.y) == goal:
            return True
    st = emu.read_state()
    return st.map_id == map_id and (st.x, st.y) == goal


def _vr_step_to(emu: PyBoyEmulator, ex: Executor, map_id: int,
                cell) -> bool:
    """Step onto an ADJACENT cell, verifying the exact
    landing. For gate crossings and stairs the static graph
    cannot see. Non-directional thaw (B presses) only: a
    directional thaw can walk the player backwards out of a
    door (probe165e)."""
    for _try in range(12):
        st = emu.read_state()
        if st.map_id != map_id:
            return False
        if (st.x, st.y) == cell:
            return True
        dx, dy = cell[0] - st.x, cell[1] - st.y
        if abs(dx) + abs(dy) != 1:
            return False
        d = {(0, -1): "up", (0, 1): "down",
             (-1, 0): "left", (1, 0): "right"}[(dx, dy)]
        _surf_step(emu, d)
        if emu.read_state().in_battle:
            _vr_clear_battle(emu)
        if _try in (3, 7):
            emu.press("b")
            _explore_settle(emu)
            emu.tick(30)
    st = emu.read_state()
    return st.map_id == map_id and (st.x, st.y) == cell


def _vr_steps(emu: PyBoyEmulator, ex: Executor, map_id: int,
              cells) -> bool:
    for cell in cells:
        if not _vr_step_to(emu, ex, map_id, cell):
            return False
    return True


def _vr_find_boulder(emu: PyBoyEmulator, cell):
    """Sprite index of the boulder currently at `cell`,
    scanning sprite RAM (probe165e)."""
    for k in range(1, 16):
        if _boulder_cell(emu, k) == cell:
            return k
    return None


def _vr_push(emu: PyBoyEmulator, ex: Executor, map_id: int,
             cell, direction: str, expect,
             blocked_extra=frozenset(),
             walk_stance: bool = True) -> bool:
    """One push of the boulder at `cell` in `direction`; it
    must land exactly on `expect` (RAM-verified). Two engine
    contacts per push; if the player followed the previous
    push into this stance, the tried-push flag ignores the
    contacts, so shuffle out and back first (probe165e)."""
    world = ex.world
    k = _vr_find_boulder(emu, cell)
    if k is None:
        return False
    dx, dy = {"up": (0, -1), "down": (0, 1),
              "left": (-1, 0), "right": (1, 0)}[direction]
    stance = (cell[0] - dx, cell[1] - dy)
    blocked = frozenset({cell}) | blocked_extra
    start_cell = (emu.read_state().x, emu.read_state().y)
    if walk_stance:
        if not _vr_walk(emu, ex, map_id, stance,
                        blocked=blocked):
            return False
    st = emu.read_state()
    if (st.x, st.y) != stance:
        return False
    if start_cell == stance:
        gm = world.maps[map_id]
        warps = frozenset((wp.x, wp.y) for wp in gm.warps)
        for d2, (ox, oy) in {"left": (-1, 0),
                             "right": (1, 0),
                             "up": (0, -1),
                             "down": (0, 1)}.items():
            if d2 == direction:
                continue
            tgt = (stance[0] + ox, stance[1] + oy)
            if tgt == cell or tgt in warps:
                continue
            if not gm.tile_walkable(*tgt):
                continue
            _surf_step(emu, d2)
            st = emu.read_state()
            if (st.x, st.y) != tgt:
                continue
            back = {"left": "right", "right": "left",
                    "up": "down", "down": "up"}[d2]
            _surf_step(emu, back)
            st = emu.read_state()
            if (st.x, st.y) == stance:
                break
        st = emu.read_state()
        if (st.x, st.y) != stance:
            return False
    if not _activate_strength(emu):
        return False
    for _try in range(4):
        emu.press(direction)
        _explore_settle(emu)
        _vr_clear_battle(emu)
        if _boulder_cell(emu, k) == expect:
            return True
    return False


def _vr_push_retry(emu: PyBoyEmulator, ex: Executor,
                   map_id: int, cell, direction: str,
                   expect) -> bool:
    """_vr_push with one retry: a failed stance shuffle
    leaves the boulder unmoved, and the second call lands
    it (the 3F row-1 leapfrog hiccup, probe165)."""
    if _vr_push(emu, ex, map_id, cell, direction, expect):
        return True
    return _vr_push(emu, ex, map_id, cell, direction, expect)


_VR1_PUSHES = [
    ((5, 15), "down", (5, 16)),
    ((5, 16), "right", (6, 16)),
    ((6, 16), "right", (7, 16)),
    ((7, 16), "right", (8, 16)),
    ((8, 16), "up", (8, 15)),       # door-mat stance, special
    ((8, 15), "right", (9, 15)),
    ((9, 15), "up", (9, 14)),
    ((9, 14), "right", (10, 14)),
    ((10, 14), "right", (11, 14)),
    ((11, 14), "right", (12, 14)),
    ((12, 14), "right", (13, 14)),
    ((13, 14), "right", (14, 14)),
    ((14, 14), "right", (15, 14)),
    ((15, 14), "right", (16, 14)),
    ((16, 14), "up", (16, 13)),
    ((16, 13), "up", (16, 12)),
    ((16, 12), "right", (17, 12)),
    ((17, 12), "down", (17, 13)),   # the switch cell
]

_VR2_SWITCH1_PUSHES = [
    ((4, 14), "down", (4, 15)),
    ((4, 15), "left", (3, 15)),
    ((3, 15), "down", (3, 16)),
    ((3, 16), "left", (2, 16)),
    ((2, 16), "left", (1, 16)),     # the switch cell
]

# 3F: BOULDER B1 rides the top band west, drops into the
# NW room and east onto the switch (3,5) - dbmapcoord
# stores (y,x), so the script's "3,5" is cell (3,5), the
# t45 floor by the (2,0) warp room. The gate it opens
# (block (3,5) -> $1d) is the west platform's door.
_VR3_SWITCH_PUSHES = (
    [((22, 3), "up", (22, 2)), ((22, 2), "up", (22, 1))]
    + [((x, 1), "left", (x - 1, 1)) for x in range(22, 6, -1)]
    + [((6, 1), "down", (6, 2)), ((6, 2), "left", (5, 2)),
       ((5, 2), "left", (4, 2)), ((4, 2), "left", (3, 2)),
       ((3, 2), "left", (2, 2)), ((2, 2), "down", (2, 3)),
       ((2, 3), "down", (2, 4)), ((2, 4), "down", (2, 5)),
       ((2, 5), "right", (3, 5))])

# From the switch room stance (2,5) to the switch room
# itself: NW room -> top band -> column 22 -> row 6 ->
# the (17,5) stairs (tile 21 is the massif's only door;
# pair (5,42) seals every other edge) -> row 4 -> the
# (9,9)/(9,10) pocket.
_VR3_TO_SWITCHROOM = (
    [(2, 4), (2, 3), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
     (6, 1)]
    + [(x, 1) for x in range(7, 23)]
    + [(22, y) for y in range(2, 7)]
    + [(x, 6) for x in range(21, 16, -1)]
    + [(17, 5), (17, 4)]
    + [(x, 4) for x in range(16, 9, -1)]
    + [(10, y) for y in range(5, 10)]
    + [(9, 9), (9, 10)])

# Through the opened gate, down the west platform stairs
# (1,9), into the south band, along row 16 to the hole
# stance (21,15).
_VR3_TO_STANCE = (
    [(8, 10), (7, 10), (6, 10), (5, 10), (5, 9), (5, 8),
     (4, 8), (3, 8), (2, 8), (1, 8), (1, 9), (1, 10),
     (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (2, 15),
     (2, 16)]
    + [(x, 16) for x in range(3, 22)]
    + [(21, 15)])


def victory_road(emu: PyBoyEmulator, ex: Executor) -> bool:
    """Victory Road + Indigo Plateau (probe165e, with the
    endgame routing corrected live in probe165):
    1F boulder onto switch (17,13) [event 2327], gate, ladder.
    2F boulder onto switch 1 (1,16) [event 1336], gate 1, the
    long loop to the (23,7) ladder.
    3F: B1 (22,3) rides the top band to the NW-room switch
    (3,5); the gate opens the west platform; the (1,9) stairs
    drop into the south band; B4 is pushed east onto the
    hole (23,15) [event 1638] and the player drops to 2F
    (22,16), where the fallen boulder (BOULDER3, shown by
    the hole script) is pushed west x14 onto switch 2
    (9,16) [event 1343]; gate 2 -> ladder (25,14) -> 3F east
    -> ladder (26,8) -> 2F NE pocket -> exit (29,7) -> R23
    (14,31) -> north edge -> Indigo Plateau -> lobby heal.
    Moltres (2F (11,5)) is never approached: the route stays
    on rows 7+ and east/south of its room, and no step enters
    its sight. Entry: Route 23 (4,32), in front of the VR
    door (route23_passage's exit). Exit: Indigo Plateau,
    party fully healed. Boulder positions and gate blocks
    reset on every map entry and Route 23 entry resets the
    cycle, so the whole road runs in one unbroken visit."""
    emu.wait_free(max_rounds=8, stable_needed=2)
    world = ex.world
    vr1 = world.map_id("VictoryRoad1F")
    vr2 = world.map_id("VictoryRoad2F")
    vr3 = world.map_id("VictoryRoad3F")
    r23 = world.map_id("Route23")
    _vr_tables(world)
    # enter VR 1F from the R23 doorstep
    if emu.read_state().map_id != vr1:
        for _ in range(8):
            if emu.read_state().map_id == vr1:
                break
            emu.press("up")
            _explore_settle(emu)
    if emu.read_state().map_id != vr1:
        return False
    # --- 1F: boulder switch, gate, ladder up
    for i, (cell, d, expect) in enumerate(_VR1_PUSHES):
        if expect == (8, 15):
            # door-mat stance: (7,17), then step east onto
            # the mat (8,17) - the door warp does not fire
            # on a sideways step
            if not _vr_walk(emu, ex, vr1, (7, 17),
                            blocked=frozenset({(8, 16)})):
                return False
            _surf_step(emu, "right")
            st = emu.read_state()
            if (st.x, st.y) != (8, 17):
                return False
            if not _vr_push(emu, ex, vr1, cell, d, expect,
                            walk_stance=False):
                return False
        elif not _vr_push_retry(emu, ex, vr1, cell, d,
                                expect):
            return False
    if not _event_flag(emu, EV_VR1_BOULDER):
        return False
    if not _vr_walk(emu, ex, vr1, (7, 12)):
        return False
    if not _vr_steps(emu, ex, vr1, [(8, 12), (9, 12),
                                    (10, 12)]):
        return False
    for goal in [(12, 12), (12, 6), (7, 8), (2, 3), (2, 1)]:
        if not _vr_walk(emu, ex, vr1, goal):
            return False
    for _ in range(6):
        if emu.read_state().map_id == vr2:
            break
        emu.press("left")
        _explore_settle(emu)
        _vr_clear_battle(emu)
    if emu.read_state().map_id != vr2:
        return False
    # --- 2F switch 1, gate 1, loop to the 3F ladder
    for cell, d, expect in _VR2_SWITCH1_PUSHES:
        if not _vr_push_retry(emu, ex, vr2, cell, d, expect):
            return False
    if not _event_flag(emu, EV_VR2_SWITCH1):
        return False
    if not _vr_walk(emu, ex, vr2, (5, 8)):
        return False
    if not _vr_steps(emu, ex, vr2, [(6, 8), (7, 8), (8, 8)]):
        return False
    if not _vr_walk(emu, ex, vr2, (23, 8), tries=12):
        return False
    for _ in range(6):
        if emu.read_state().map_id == vr3:
            break
        emu.press("up")
        _explore_settle(emu)
        _vr_clear_battle(emu)
    if emu.read_state().map_id != vr3:
        return False
    # --- 3F: B1 to the switch, gate, stairs, south band
    for cell, d, expect in _VR3_SWITCH_PUSHES:
        if not _vr_push_retry(emu, ex, vr3, cell, d, expect):
            return False
    if not _vr_steps(emu, ex, vr3, _VR3_TO_SWITCHROOM):
        return False
    if not _vr_steps(emu, ex, vr3, _VR3_TO_STANCE):
        return False
    # --- the hole drop: B4 east onto (23,15), then step on
    # after it (the fall + dungeon warp process on the
    # player's own step onto the hole cell)
    k = _vr_find_boulder(emu, (22, 15))
    if k is None:
        return False
    for _press in range(2):
        emu.press("right")
        _explore_settle(emu)
        _vr_clear_battle(emu)
        if emu.read_state().map_id == vr2:
            break
    st = emu.read_state()
    if st.map_id != vr2 or (st.x, st.y) != (22, 16):
        return False
    if not _event_flag(emu, EV_VR3_HOLE):
        return False
    if _vr_find_boulder(emu, (23, 16)) is None:
        return False    # the fallen boulder must be shown
    # --- 2F switch 2: the fallen boulder west x14
    for x in range(23, 9, -1):
        if not _vr_push_retry(emu, ex, vr2, (x, 16), "left",
                              (x - 1, 16)):
            return False
    if not _event_flag(emu, EV_VR2_SWITCH2):
        return False
    # --- exit: gate 2, ladders, the NE pocket door
    if not _vr_steps(emu, ex, vr2,
                     [(x, 16) for x in range(11, 22)]
                     + [(21, 15), (21, 14), (22, 14),
                        (23, 14), (24, 14)]):
        return False
    _vr_step_to(emu, ex, vr2, (25, 14))   # ladder warp
    st = emu.read_state()
    if st.map_id != vr3 or (st.x, st.y) != (27, 15):
        return False
    if not _vr_steps(emu, ex, vr3,
                     [(27, y) for y in range(14, 8, -1)]
                     + [(26, 9)]):
        return False
    _vr_step_to(emu, ex, vr3, (26, 8))    # ladder warp
    st = emu.read_state()
    if st.map_id != vr2 or (st.x, st.y) != (27, 7):
        return False
    if not _vr_step_to(emu, ex, vr2, (28, 7)):
        return False
    _vr_step_to(emu, ex, vr2, (29, 7))    # exit warp
    if emu.read_state().map_id == vr2:
        # the door warp can need an off-and-back-on step
        _vr_step_to(emu, ex, vr2, (28, 7))
        _vr_step_to(emu, ex, vr2, (29, 7))
        emu.tick(120)
        emu.wait_free(max_rounds=8, stable_needed=2)
    if emu.read_state().map_id != r23:
        return False
    # --- Indigo Plateau: north edge, lobby, heal
    indigo = world.map_id("IndigoPlateau")
    lobby = world.map_id("IndigoPlateauLobby")
    ok, _ = _walk_fighting(emu, ex, r23, (18, 32),
                           attempts=25, flee_wild=True)
    if not _ride_or_cross(emu, ex, indigo):
        return False
    if not _ride_or_cross(emu, ex, lobby):
        return False
    # heal_at_pokecenter's standard-layout assumption fails
    # in the Plateau lobby; the nurse (7,5) is talked to
    # across her counter from (7,7) (probe165, live-verified).
    if not _vr_steps(emu, ex, lobby, [(7, 10), (7, 9),
                                      (7, 8), (7, 7)]):
        return False
    emu.press("up")
    emu.tick(30)
    emu.press("a")
    emu.tick(120)
    for _ in range(30):
        emu.wait_free(max_rounds=4, stable_needed=2)
        lead = emu.read_state().party[0]
        if lead.hp == lead.max_hp:
            break
        emu.press("a")
        emu.tick(90)
    # the heal animation restores the party member by
    # member; verify on a settle-retry loop, not one read
    # (probe166's replay raced this too: the state left
    # behind was fully healed, the single read was not)
    for _ in range(30):
        emu.wait_free(max_rounds=4, stable_needed=2)
        emu.tick(60)
        st = emu.read_state()
        if all(m.hp == m.max_hp for m in st.party):
            break
    st = emu.read_state()
    if any(m.hp != m.max_hp for m in st.party):
        return False
    if emu.read_state().map_id == lobby:
        _exit_building(emu, ex, lobby)
    st = emu.read_state()
    return st.map_id == indigo and all(
        m.hp == m.max_hp for m in st.party)


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
