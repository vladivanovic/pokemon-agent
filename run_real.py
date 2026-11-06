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


def _drive_battle_typed(emu: PyBoyEmulator,
                        switch_first=None) -> bool:
    """Drive the live battle to its honest end (in_battle
    clears) with the typed ranker (probe145's drive_battle).
    fight_until_end's own exit can fire seconds early on menu
    flicker; this loop re-drives until the battle flag is
    really down. A PP-dry active mon is rescued by switching
    to the first teammate holding PP (the Exeggcute wedge's
    lesson); switch_first rides only the first drive."""
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
                   tile, attempts: int = 25):
    """Walk to a tile inside one map, fighting whatever
    intercepts (probe145). Returns (arrived, battles fought).
    Needed wherever sight trainers guard the path: a raw
    _walk_to sometimes threads a sight line without triggering
    the trainer at all, and a challenge built on it talks to
    thin air (the probe145 Lasses)."""
    fought = 0
    for _try in range(attempts):
        st = emu.read_state()
        if st.in_battle:
            _drive_battle_typed(emu)
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
            _drive_battle_typed(emu)
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
                  cell, btn):
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
            _drive_battle_typed(emu)
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
                target, max_steps: int = 6000):
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
            _drive_battle_typed(emu)
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
            _, b = _explore_step(emu, ex, map_id, cell, btn)
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
        _, b = _explore_step(emu, ex, map_id, cell, btn)
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
                       max_rounds: int = 60) -> bool:
    """Advance dialog (A presses) until a battle starts, then
    drive it. Returns True if a battle was fought."""
    for _ in range(max_rounds):
        st = emu.read_state()
        if st.in_battle:
            _drive_battle_typed(emu)
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
                                   (-1, 0))) -> bool:
    """Navigate to a stance next to an NPC, face it, talk, and
    fight if a battle starts; verified by the trainer event
    flag. (A sight-triggered battle during the approach also
    satisfies it: the flag is checked first and after.)"""
    if _event_flag(emu, event_id):
        return True
    nx, ny = npc_cell
    for dx, dy in stance_order:
        stance = (nx + dx, ny + dy)
        status, _ = explore_nav(emu, ex, map_id, stance)
        st = emu.read_state()
        if status != "tile" or (st.x, st.y) != stance:
            continue
        if _event_flag(emu, event_id):
            return True
        _face_and_talk(emu, npc_cell)
        _advance_to_battle(emu)
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
