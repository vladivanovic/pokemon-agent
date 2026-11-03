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
from typing import List, Optional

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
        for target in tiles[1:]:
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
        blocked = set()
        last_error: Optional[StuckError] = None
        for _attempt in range(6):
            st = self._state()
            path = self.world.tile_path(map_id, (st.x, st.y), xy,
                                         blocked=frozenset(blocked))
            if path is None:
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
                                   goal.target_map, goal.target_xy, caps)
        if steps is None:
            raise StuckError("no journey (portal-level route) to target")
        for step in steps:
            kind = step[0]
            if kind == "walk":
                _, mid, tile = step
                self._walk_to(mid, tile)
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
