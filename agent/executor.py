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
                        raise StuckError(f"no movement toward {target}")
                else:
                    no_move = 0

    def _walk_to(self, map_id: int, xy) -> None:
        st = self._state()
        path = self.world.tile_path(map_id, (st.x, st.y), xy)
        if path is None:
            raise StuckError(f"no tile path to {xy} in map {map_id}")
        self._follow_tiles(map_id, path)

    def _cross_connection(self, direction: str, dst: int) -> None:
        st = self._state()
        gm = self.world.maps[st.map_id]
        dx, dy = _BUTTON_TO_DELTA[{"north": "up", "south": "down",
                                   "west": "left", "east": "right"}[direction]]
        # Candidate exit tiles: walkable tiles on the named edge.
        if dy != 0:
            ey = 0 if dy < 0 else gm.h - 1
            exits = [(x, ey) for x in range(gm.w) if gm.walkable(x, ey, dx, dy)]
        else:
            ex = 0 if dx < 0 else gm.w - 1
            exits = [(ex, y) for y in range(gm.h) if gm.walkable(ex, y, dx, dy)]
        exits.sort(key=lambda t: abs(t[0] - st.x) + abs(t[1] - st.y))
        last_err: Optional[Exception] = None
        for ex in exits:
            try:
                self._walk_to(st.map_id, ex)
                before = self._state()
                self._press(_DELTA_TO_BUTTON[(dx, dy)])
                if self._state().map_id == dst:
                    return
                last_err = StuckError(f"crossing {direction} did not change map")
            except StuckError as e:
                last_err = e
        raise last_err or StuckError(f"no exit tile for {direction}")

    def _take_warp(self, dst: int) -> None:
        st = self._state()
        gm = self.world.maps[st.map_id]
        candidates = [wp for wp in gm.warps if wp.dest_id == dst]
        if not candidates:
            # Exit warp (LAST_MAP): leads back to the entrance map.
            candidates = [wp for wp in gm.warps if wp.is_exit]
        for wp in candidates:
            if (st.x, st.y) == (wp.x, wp.y):
                # Arriving via a warp leaves you standing ON the exit mat,
                # and warps only fire when stepped onto: step off, step back.
                self._retrigger_warp(dst)
                return
            self._walk_to(st.map_id, (wp.x, wp.y))
            if self._state().map_id == dst:
                return
            # Some doors need one more step onto the mat.
            self._press("down")
            if self._state().map_id == dst:
                return
            raise StuckError(f"warp to {dst} did not trigger")
        raise StuckError(f"no warp to {dst} in map {st.map_id}")

    def _retrigger_warp(self, dst: int) -> None:
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
        for button, back in opposite.items():
            before = self._state()
            self._press(button)
            mid = self._state()
            if mid.map_id == dst:
                return
            if mid.pos != before.pos:
                self._press(back)
                if self._state().map_id == dst:
                    return
        raise StuckError(f"could not retrigger warp to {dst}")

    # -- goal execution -----------------------------------------------------------
    def _goto(self, goal: Goal, state: GameState) -> None:
        caps = state.capabilities()
        route = self.world.route(state.map_id, goal.target_map, caps)
        if route is None:
            raise StuckError("route disappeared between validation and execution")
        for leg in route.legs:
            if leg.via == "fly":
                self.emu.fly_to(leg.dst)
            elif leg.via.startswith("warp"):
                self._take_warp(leg.dst)
            elif leg.via.startswith("connection"):
                self._cross_connection(leg.via.split(" ", 1)[1], leg.dst)
            else:
                raise StuckError(f"unknown leg type: {leg.via}")
            st = self._state()
            if st.in_battle:
                self._battle()
        if goal.target_xy is not None:
            self._walk_to(goal.target_map, goal.target_xy)

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
