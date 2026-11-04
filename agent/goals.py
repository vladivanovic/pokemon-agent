"""Structured goals and the story checklist.

The old system parsed goals out of prose with regex, which produced
symmetric nonsense goals and 2-cycles. Here a goal is a typed value the
harness can validate before a single step is taken: *propose-and-verify*.
A planner (rules or LLM) may propose anything; only goals that survive
validation against the static world and the current RAM state ever reach
the executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .memory import EpisodicMemory, state_signature
from .state import GameState
from .world import World


@dataclass
class Goal:
    kind: str                       # "goto" (v1 implements goto; extensible)
    target_map: int
    target_xy: Optional[tuple] = None
    reason: str = ""
    optional: bool = False          # detours are optional; spine goals are not

    def key(self) -> str:
        return f"{self.kind}:{self.target_map}:{self.target_xy}"


@dataclass
class Verdict:
    ok: bool
    problems: List[str] = field(default_factory=list)
    missing_capabilities: List[str] = field(default_factory=list)
    acquire_first: List[str] = field(default_factory=list)


def validate_goal(goal: Goal, state: GameState, world: World,
                  memory: EpisodicMemory) -> Verdict:
    """Check a proposed goal before executing it."""
    problems: List[str] = []
    if goal.target_map not in world.maps:
        return Verdict(False, [f"unknown map id {goal.target_map}"])
    caps = state.capabilities()
    if world.route(state.map_id, goal.target_map, caps) is None:
        missing = world.missing_capabilities(state.map_id, goal.target_map, caps)
        if missing:
            problems.append(
                f"no route to {world.map_name(goal.target_map)} "
                f"without: {', '.join(sorted(missing))}")
            return Verdict(False, problems, sorted(missing), sorted(missing))
        problems.append(f"no route to {world.map_name(goal.target_map)} at all")
        return Verdict(False, problems)
    # A map-level route can exist while every tile-level journey is
    # sealed by an obstacle the route graph cannot see - concretely,
    # a Cut-tree row sealing a pocket (Route 2's cave mouth). If no
    # journey plans but adding Cut alone would plan one, reject with
    # that diagnosis instead of letting the executor discover it as
    # "stuck". Deliberately Cut-only: other journey-level failures
    # (the sea routes the portal planner cannot model, say) keep the
    # old behaviour - approved here, honestly reported by the
    # executor if they bite.
    if "cut" not in caps \
            and world.journey(state.map_id, (state.x, state.y),
                              goal.target_map, goal.target_xy,
                              caps) is None \
            and world.journey(state.map_id, (state.x, state.y),
                              goal.target_map, goal.target_xy,
                              caps | {"cut"}) is not None:
        problems.append(
            f"no journey to {world.map_name(goal.target_map)} "
            f"without: cut")
        return Verdict(False, problems, ["cut"], ["cut"])
    if goal.target_xy is not None:
        gm = world.maps[goal.target_map]
        tx, ty = goal.target_xy
        if not gm.in_bounds(tx, ty):
            return Verdict(False, [f"target {goal.target_xy} out of bounds"])
    fails = memory.failures(goal.key(), state_signature(state))
    if fails >= 2:
        problems.append(
            f"goal already failed {fails}x in an unchanged situation; "
            "suppressing to break the loop")
        return Verdict(False, problems)
    return Verdict(True)


# --- story checklist ---------------------------------------------------------
# The spine of the game as an ordered checklist. Completion is judged from
# RAM state + episodic memory (visited maps), never from planner opinion.
# The planner's freedom lives in *how* and in optional detours, not in
# inventing what the next chapter is.

@dataclass
class StoryStep:
    step_id: str
    title: str
    goal_map_name: str
    done: Callable[[GameState, EpisodicMemory, World], bool]
    goal_xy: Optional[tuple] = None


def _has_badge(name):
    return lambda s, m, w: name in s.badges


def _visited(name):
    return lambda s, m, w: m.visited(w.map_id(name))


def _visited_after(first, later):
    return lambda s, m, w: m.visited_after(w.map_id(first), w.map_id(later))


STORY: List[StoryStep] = [
    StoryStep("starter", "Get a starter Pokemon from Oak's Lab", "OaksLab",
              lambda s, m, w: len(s.party) > 0),
    StoryStep("parcel_pickup", "Pick up Oak's Parcel at Viridian Mart",
              "ViridianMart", _visited("ViridianMart")),
    StoryStep("parcel_deliver", "Deliver the parcel to Oak's Lab", "OaksLab",
              _visited_after("ViridianMart", "OaksLab")),
    StoryStep("to_pewter", "Reach Pewter City through Viridian Forest",
              "PewterCity", _visited("PewterCity")),
    StoryStep("boulder", "Win the Boulder Badge", "PewterGym",
              _has_badge("Boulder")),
    StoryStep("cascade", "Win the Cascade Badge in Cerulean City",
              "CeruleanGym", _has_badge("Cascade")),
    StoryStep("thunder", "Win the Thunder Badge in Vermilion City",
              "VermilionGym", _has_badge("Thunder")),
    StoryStep("rainbow", "Win the Rainbow Badge in Celadon City",
              "CeladonGym", _has_badge("Rainbow")),
    StoryStep("soul", "Win the Soul Badge in Fuchsia City", "FuchsiaGym",
              _has_badge("Soul")),
    StoryStep("marsh", "Win the Marsh Badge in Saffron City", "SaffronGym",
              _has_badge("Marsh")),
    StoryStep("volcano", "Win the Volcano Badge on Cinnabar Island",
              "CinnabarGym", _has_badge("Volcano")),
    StoryStep("earth", "Win the Earth Badge in Viridian City", "ViridianGym",
              _has_badge("Earth")),
    StoryStep("indigo", "Reach Indigo Plateau", "IndigoPlateau",
              _visited("IndigoPlateau")),
]

# Optional detours a planner may take when capabilities allow. They never
# block the spine.
DETOURS: List[StoryStep] = [
    StoryStep("zapdos", "Catch Zapdos at the Power Plant", "PowerPlant",
              lambda s, m, w: m.succeeded("detour:zapdos")),
]


def next_step(state: GameState, memory: EpisodicMemory,
              world: World) -> Optional[StoryStep]:
    for step in STORY:
        if not step.done(state, memory, world):
            return step
    return None


def step_goal(step: StoryStep, world: World) -> Goal:
    return Goal(
        kind="goto",
        target_map=world.map_id(step.goal_map_name),
        target_xy=step.goal_xy,
        reason=step.title,
        optional=step.step_id in {d.step_id for d in DETOURS},
    )
