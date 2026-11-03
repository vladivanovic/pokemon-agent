"""The executor: carries out validated goals with deterministic machinery.

Loop shape (event-driven, no per-step model calls):
  observe RAM state -> handle battle if in one -> ask planner for a goal
  -> validate it -> walk it (BFS tile paths, warp/connection/fly legs)
  -> record the outcome in episodic memory -> repeat.

Stuck detection is positional: if presses stop changing the position
while a path remains, the goal is failed *with the state signature*, so
the validator can suppress an identical retry and break the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .goals import Goal, validate_goal
from .memory import EpisodicMemory, state_signature
from .state import GameState
from .world import World

_DELTA_TO_BUTTON = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}
_BUTTON_TO_DELTA = {v: k for k, v in _DELTA_TO_BUTTON.items()}


class StuckError(Exception):
    pass


class FightPolicy:
    """Default battle policy: confirm through the fight (the 'a' button
    selects the highlighted move in the simulator; on hardware this seam
    is where the battle state machine lives)."""

    def choose(self, state: GameState, presses: int) -> str:
        return "a"


# Non-damaging move ids (Gen 1, from pokered's move_constants.asm):
# status, stat-stage, healing and escape moves. A battle policy must
# never *choose* these as its attack - spamming one can stall a
# battle forever (seen live: with Bubble's PP dry, "last move with
# PP" fell through to Tail Whip against a Harden-only Metapod and
# neither side could ever deal damage). Unknown ids are treated as
# damaging, which is the right bias for a policy that must keep
# battles moving.
STATUS_MOVE_IDS = frozenset({
    28,   # SAND_ATTACK
    39,   # TAIL_WHIP
    43,   # LEER
    45,   # GROWL
    47,   # SING
    48,   # SUPERSONIC
    50,   # DISABLE
    54,   # MIST
    77,   # POISONPOWDER
    78,   # STUN_SPORE
    79,   # SLEEP_POWDER
    81,   # STRING_SHOT
    86,   # THUNDER_WAVE
    92,   # TOXIC
    95,   # HYPNOSIS
    96,   # MEDITATE
    97,   # AGILITY
    100,  # TELEPORT
    103,  # SCREECH
    104,  # DOUBLE_TEAM
    105,  # RECOVER
    106,  # HARDEN
    107,  # MINIMIZE
    108,  # SMOKESCREEN
    109,  # CONFUSE_RAY
    110,  # WITHDRAW
    112,  # BARRIER
    113,  # LIGHT_SCREEN
    114,  # HAZE
    115,  # REFLECT
    116,  # FOCUS_ENERGY
    133,  # AMNESIA
    134,  # KINESIS
    135,  # SOFTBOILED
    137,  # GLARE
    139,  # POISON_GAS
    142,  # LOVELY_KISS
    147,  # SPORE
    148,  # FLASH
    150,  # SPLASH
    156,  # REST
    159,  # SHARPEN
    160,  # CONVERSION
    164,  # SUBSTITUTE
})


class SmartFightPolicy:
    """Hardware battle policy: fight with the party leader's LAST
    damaging move that still has PP - the newest-learned move is
    usually the strongest, and for a water starter it is the
    type-advantage one (Bubble, then Water Gun) that pure A-mashing
    never reaches because the cursor starts on move 1 every battle.

    It drives the real menus by their RAM cursor state (state.py's
    ADDR_MENU_* reads). The subtlety, learned from a live stall: the
    menu cursor bytes are only LIVE while a menu waits for input -
    during battle text and animations they hold stale values (even
    leftovers from before the battle), and direction presses then are
    inert while A still advances the turn. So the policy never trusts
    the bytes blindly. When the move menu appears to be up
    (menu_top_y == 12, item in range, cursor off target) it PROBES
    once with a step toward the target: if the cursor moves, the menu
    is live and it steers the rest of the way and confirms; if not,
    it was a stale echo and the policy goes back to advancing with A
    until the main menu (top == 14) has been seen again, which resets
    the episode. In the main menu it confirms FIGHT. If NO move has
    PP it just confirms - the engine offers Struggle itself - and it
    never picks a 0-PP move, which also un-sticks the engine's no-PP
    rejection loop that an A-mash battle dies in (seen live: Tackle
    dry by the time the agent reached Brock).

    On the simulator the menu bytes read as 0 and every branch falls
    through to A, so this degrades to FightPolicy there."""

    def __init__(self) -> None:
        self._probed = False        # this menu episode probed stale
        self._live = False          # this menu episode is confirmed live
        self._probe_from = None     # cursor value when the probe fired
        self._last_sig = None       # (menu_top_y, menu_item) at last call

    def _reset_episode(self) -> None:
        self._probed = self._live = False
        self._probe_from = None

    def choose(self, state: GameState, presses: int) -> str:
        if not state.in_battle or not state.party:
            self._reset_episode()
            self._last_sig = None
            return "a"
        # Judge a pending probe: if the cursor moved since the probe
        # press, the menu was live; if it is unchanged, the bytes were
        # a stale echo of some earlier menu and presses are inert.
        if self._probe_from is not None:
            if state.menu_item != self._probe_from:
                self._live = True
            self._probe_from = None
        sig = (state.menu_top_y, state.menu_item)
        if sig != self._last_sig:
            # Seeing the main menu ends any earlier episode.
            if state.menu_top_y == 14:
                self._reset_episode()
            self._last_sig = sig
        mon = state.party[0]
        # Target: the LAST damaging move that still has PP. If no
        # damaging move has PP, burn the smallest-PP move available -
        # exhausting PP entirely is the only route to the engine's
        # Struggle, and a status move picked on purpose can otherwise
        # stall the battle forever (see STATUS_MOVE_IDS).
        target = None
        for i in range(len(mon.moves) - 1, -1, -1):
            if i < len(mon.pp) and mon.pp[i] > 0 \
                    and mon.moves[i] not in STATUS_MOVE_IDS:
                target = i
                break
        if target is None:
            usable = [(mon.pp[i], i) for i in range(len(mon.moves))
                      if i < len(mon.pp) and mon.pp[i] > 0]
            if usable:
                target = min(usable)[1]
        if target is None:
            return "a"
        if state.menu_top_y == 14:       # main battle menu
            return "a" if state.menu_item == 0 else "up"
        if state.menu_top_y == 12 and 1 <= state.menu_item <= 4:
            cur = state.menu_item        # 1-based in the move menu
            if cur == target + 1:
                return "a"               # on target: confirm the move
            if self._live and state.move_list_index == target:
                # The engine has registered the pick; the menu bytes
                # are stale echoes again until the next main menu.
                self._reset_episode()
                self._probed = True
                return "a"
            step = "down" if cur < target + 1 else "up"
            if self._live:
                return step
            if not self._probed:
                self._probed = True
                self._probe_from = cur   # one probe per menu episode;
                return step              # it is judged on the next call
            return "a"                   # probed stale: just advance
        return "a"


@dataclass
class RunResult:
    status: str                       # complete | blocked | stuck | step_limit
    goals_done: int = 0
    log: List[dict] = field(default_factory=list)


class Executor:
    def __init__(self, emu, world: World, planner, memory: EpisodicMemory,
                 stuck_threshold: int = 24, max_presses: int = 200_000,
                 battle_policy=None):
        self.emu = emu
        self.world = world
        self.planner = planner
        self.memory = memory
        self.stuck_threshold = stuck_threshold
        self.max_presses = max_presses
        self.battle_policy = battle_policy or FightPolicy()
        self.presses = 0
        self.log: List[dict] = []
        self._blocked_tile = None  # set by _follow_tiles on a refused step
        # Tiles that refused entry on a GUESSED grid (the CAVERN
        # tileset's grids are marked fully open - its tile ids do not
        # decode through the collision lists). A refusal outdoors is
        # usually an NPC and stays transient; a refusal on a guessed
        # grid is the map correcting the model, so it is learned for
        # the rest of the session and fed to every later plan. Without
        # this, a phantom-open cave loops forever: bump, replan the
        # same shortcut, bump again (Mt. Moon B1F (23, 15), live).
        self._learned_walls: Dict[int, Set[tuple]] = {}

    # -- primitives -----------------------------------------------------------
    def _press(self, button: str) -> None:
        self.emu.press(button)
        self.presses += 1
        if self.presses > self.max_presses:
            raise StuckError("press budget exhausted")

    def _state(self) -> GameState:
        st = self.emu.read_state()
        self.memory.record_visit(st.map_id)
        return st

    def _follow_tiles(self, map_id: int, tiles) -> None:
        """Walk a within-map tile path (list of (x, y), current pos first)."""
        no_move = 0
        ledges = self.world.maps[map_id].ledges
        idx = 1
        while idx < len(tiles):
            target = tiles[idx]
            while True:
                st = self._state()
                if st.map_id != map_id:
                    return  # a warp carried us elsewhere: leg complete
                if st.in_battle:
                    # A wild battle mid-walk eats presses exactly like a
                    # blocked tile; it must not be recorded as one. Abort
                    # the walk - run() fights the battle first, then the
                    # goal is retried on a fresh round.
                    raise StuckError("battle interrupted walk")
                if st.in_dialog:
                    raise StuckError("dialog interrupted walk")
                if (st.x, st.y) == target:
                    idx += 1
                    break
                # Ledge hop: pressing toward a face tile makes the game
                # hop two tiles, landing on the path entry AFTER the
                # face. Consume both entries when we see that landing.
                if idx + 1 < len(tiles) \
                        and (st.x, st.y) == tiles[idx + 1] \
                        and ledges.get(target) == (
                            tiles[idx + 1][0] - target[0],
                            tiles[idx + 1][1] - target[1]):
                    idx += 2
                    break
                dx = target[0] - st.x
                dy = target[1] - st.y
                if (dx, dy) not in _DELTA_TO_BUTTON:
                    raise StuckError(f"non-adjacent step {st.pos} -> {target}")
                before = st.pos
                self._press(_DELTA_TO_BUTTON[(dx, dy)])
                after = self._state()
                if after.pos == before:
                    no_move += 1
                    if no_move >= self.stuck_threshold:
                        # The tile we cannot enter (an NPC is standing
                        # on it, most likely): _walk_to replans around
                        # it as a transient obstacle.
                        self._blocked_tile = target
                        raise StuckError(f"no movement toward {target}")
                else:
                    no_move = 0

    def _walk_to(self, map_id: int, xy) -> None:
        blocked = set(self._learned_walls.get(map_id, ()))
        last_error: Optional[StuckError] = None
        cleared = False
        for _attempt in range(6):
            st = self._state()
            path = self.world.tile_path(map_id, (st.x, st.y), xy,
                                         blocked=frozenset(blocked))
            if path is None:
                # Fail-soft: learned walls are beliefs, and a belief
                # that seals the map must not brick the agent (one
                # false wall - an NPC on a chokepoint - did exactly
                # that on Mt. Moon B1F). Forget this map's walls once
                # and replan with fresh eyes; real walls re-teach
                # themselves with one bump each.
                if not cleared and blocked:
                    self._learned_walls.pop(map_id, None)
                    blocked = set()
                    cleared = True
                    continue
                raise StuckError(f"no tile path to {xy} in map {map_id}")
            try:
                self._follow_tiles(map_id, path)
                return
            except StuckError as e:
                # A transient desync (e.g. a state read caught mid-warp)
                # invalidates the path, not the goal: replan from the
                # settled position. A tile that refuses entry (an NPC
                # standing on it) invalidates only that tile: exclude it
                # and route around. Either way the goal survives a few
                # replans before we call it stuck - and if every replan
                # fails, the last error is raised, never swallowed.
                last_error = e
                if "non-adjacent" in str(e):
                    continue
                if "no movement" in str(e) and self._blocked_tile is not None:
                    blocked.add(self._blocked_tile)
                    if self.world.maps[map_id].tileset.upper() == "CAVERN":
                        self._learned_walls.setdefault(map_id, set()).add(
                            self._blocked_tile)
                    self._blocked_tile = None
                    continue
                raise
        raise last_error or StuckError(f"walk to {xy} in map {map_id} failed")

    def _retrigger_warp(self) -> None:
        """Standing on a warp mat that has not fired (you arrive on one):
        step off and back on."""
        start_map = self._state().map_id
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
        for button, back in opposite.items():
            before = self._state()
            self._press(button)
            mid = self._state()
            if mid.map_id != start_map:
                return
            if mid.pos != before.pos:
                self._press(back)
                if self._state().map_id != start_map:
                    return
        raise StuckError("could not retrigger warp")

    def _clear_dialog(self, max_presses: int = 60) -> None:
        for _ in range(max_presses):
            if not self._state().in_dialog:
                return
            self._press("a")

    # -- goal execution -----------------------------------------------------------
    def _goto(self, goal: Goal, state: GameState) -> None:
        caps = state.capabilities()
        steps = self.world.journey(state.map_id, (state.x, state.y),
                                   goal.target_map, goal.target_xy, caps,
                                   blocked=self._learned_walls)
        if steps is None and self._learned_walls:
            # Same fail-soft as _walk_to, at journey scale: if the
            # learned walls are the only thing sealing the route,
            # forget them all and plan once more unlearned.
            self._learned_walls.clear()
            steps = self.world.journey(state.map_id, (state.x, state.y),
                                       goal.target_map, goal.target_xy,
                                       caps, blocked=self._learned_walls)
        if steps is None:
            raise StuckError("no journey (portal-level route) to target")
        for step in steps:
            kind = step[0]
            if kind == "walk":
                _, mid, tile = step
                try:
                    self._walk_to(mid, tile)
                except StuckError:
                    # Some warp tiles are land-only: the warp event sits
                    # on a tile the engine will not let you step onto
                    # (Viridian Forest's south gate: (4, 0) refuses,
                    # its block-mate (5, 0) is the real door). Try the
                    # sibling warps to the same destination before
                    # calling the leg stuck.
                    gm = self.world.maps[mid]
                    tw = next((wp for wp in gm.warps
                               if (wp.x, wp.y) == tile), None)
                    if tw is None:
                        raise
                    landed = None
                    for wp in gm.warps:
                        if wp.dest_id == tw.dest_id \
                                and (wp.x, wp.y) != tile:
                            try:
                                self._walk_to(mid, (wp.x, wp.y))
                                landed = (wp.x, wp.y)
                                break
                            except StuckError:
                                continue
                    if landed is None:
                        raise
                    tile = landed
                st = self._state()
                if st.map_id == mid:
                    # If the target was a warp tile, stepping on it should
                    # have changed the map. If we are standing on it and
                    # nothing fired, retrigger by stepping off and back on.
                    gm = self.world.maps[mid]
                    if any((wp.x, wp.y) == tile for wp in gm.warps) \
                            and (st.x, st.y) == tile:
                        self._retrigger_warp()
            elif kind == "cross":
                _, direction, dst = step
                button = {"north": "up", "south": "down",
                          "west": "left", "east": "right"}[direction]
                self._press(button)
                if self._state().map_id != dst:
                    raise StuckError(f"crossing {direction} did not reach {dst}")
            elif kind == "fly":
                self.emu.fly_to(step[1])
            else:
                raise StuckError(f"unknown journey step: {step}")
            if self._state().in_battle:
                self._battle()

    def _battle(self) -> None:
        n = 0
        while self._state().in_battle:
            self._press(self.battle_policy.choose(self._state(), n))
            n += 1
            if n > 500:
                raise StuckError("battle did not end")

    # -- main loop ------------------------------------------------------------------
    def run(self, max_goals: int = 50) -> RunResult:
        result = RunResult(status="complete")
        try:
            for _ in range(max_goals):
                state = self._state()
                if state.in_battle:
                    self._battle()
                    state = self._state()
                if state.in_dialog:
                    # A scripted scene (a clerk, a rival, a delivery) is
                    # holding the game: advance it until the box closes,
                    # then re-observe. Pressing A only while a box is
                    # verifiably open is what keeps this from examining
                    # furniture (see state.py's dialog calibration).
                    self._clear_dialog()
                    state = self._state()
                goal = self.planner.propose(state, self.world, self.memory)
                if goal is None:
                    result.status = "complete"
                    break
                verdict = validate_goal(goal, state, self.world, self.memory)
                entry = {"goal": goal.key(), "reason": goal.reason,
                         "verdict": verdict.ok}
                result.log.append(entry)
                self.memory.record("goal_proposed", goal=goal.key(),
                                   ok=verdict.ok)
                if not verdict.ok:
                    self.memory.record_goal_result(
                        goal.key(), False, state_signature(state),
                        "; ".join(verdict.problems))
                    result.status = "blocked"
                    entry["problems"] = verdict.problems
                    break
                if state.map_id == goal.target_map and goal.target_xy is None:
                    # Already there and nothing to walk to: completing this
                    # goal is an *event* (a battle, a conversation), not
                    # movement. Say so instead of fake-succeeding in a loop.
                    self.memory.record_goal_result(
                        goal.key(), False, state_signature(state),
                        "already at destination; goal needs an event/macro")
                    result.status = "blocked"
                    entry["problems"] = ["event/macro required, not movement"]
                    break
                try:
                    self._goto(goal, state)
                except StuckError as e:
                    self.memory.record_goal_result(
                        goal.key(), False, state_signature(state), str(e))
                    result.status = "stuck"
                    entry["error"] = str(e)
                    break
                self.memory.record_goal_result(
                    goal.key(), True, state_signature(state))
                result.goals_done += 1
            else:
                result.status = "step_limit"
        except StuckError as e:
            result.status = "stuck"
            result.log.append({"error": str(e)})
        self.log = result.log
        return result
