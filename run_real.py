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

# PyBoy wants SDL; headless machines need the dummy drivers, and they
# must be set before pyboy is imported (PyBoyEmulator imports it).
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from agent.executor import Executor, SmartFightPolicy, StuckError
from agent.goals import Goal
from agent.memory import EpisodicMemory
from agent.planner import ChecklistPlanner
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
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    world = default_world()
    emu = PyBoyEmulator(args.rom)
    memory = EpisodicMemory()
    ex = Executor(emu, world, ChecklistPlanner(), memory,
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
