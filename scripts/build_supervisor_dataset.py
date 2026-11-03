#!/usr/bin/env python3
"""Build the SFT dataset for a fine-tuned LLM supervisor.

Two sources, one format (chat-messages JSONL, exactly the context
LLMPlanner sends at inference time -> {"target_map_name", "reason"}):

  (a) Scenario synthesis from the ROM world model: for each story
      step, synthesize states along the route to it (progression
      prefixes: badges / visits / HM moves a real run would have) and
      ask the ChecklistPlanner - the deterministic oracle - what the
      goal is. Every emitted goal is re-checked with
      goals.validate_goal against its own state before it lands in
      the dataset.
  (b) The supervisor log (agent/supervisor_log.py): proposals the LLM
      made that the validator accepted AND the executor completed
      successfully. Real distribution, verified labels.

Output: <out>/train.jsonl + <out>/eval.jsonl (default
dataset/supervisor/ - gitignored; regenerate, don't commit).

    python3 scripts/build_supervisor_dataset.py --count 800 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.goals import DETOURS, STORY, next_step, validate_goal
from agent.memory import EpisodicMemory
from agent.planner import SYSTEM_PROMPT, ChecklistPlanner, LLMPlanner
from agent.state import HM_MOVES, GameState, PartyMon
from agent.supervisor_log import DEFAULT_SUPERVISOR_LOG, SupervisorLog
from agent.world import FLY_DESTINATIONS, World, default_world

# Completion effects of each story step, in STORY order: what a real
# run has accumulated (badges, ordered map visits, HM moves in the
# party) once that step is done. The synthesized state for step k
# applies the effects of steps < k, so exactly step k is next.
_STEP_EFFECTS = [
    # step_id         badges gained   visits (in order)      HM moves gained
    ("starter",       [],             [],                    []),
    ("parcel_pickup", [],             ["ViridianMart"],      []),
    ("parcel_deliver", [],            ["OaksLab"],           []),
    ("to_pewter",     [],             ["PewterCity"],        []),
    ("boulder",       ["Boulder"],    [],                    [148]),  # Flash
    ("cascade",       ["Cascade"],    [],                    [15]),   # Cut
    ("thunder",       ["Thunder"],    [],                    [19]),   # Fly
    ("rainbow",       ["Rainbow"],    [],                    [70]),   # Strength
    ("soul",          ["Soul"],       [],                    [57]),   # Surf
    ("marsh",         ["Marsh"],      [],                    []),
    ("volcano",       ["Volcano"],    [],                    []),
    ("earth",         ["Earth"],      [],                    []),
    ("indigo",        [],             ["IndigoPlateau"],     []),
]
assert [e[0] for e in _STEP_EFFECTS] == [s.step_id for s in STORY], \
    "_STEP_EFFECTS drifted from goals.STORY"

_FILLER_MOVES = [1, 5, 9, 24, 29, 33]  # ordinary damaging moves


def _prefix(k: int):
    badges, visits, hm_moves = [], [], []
    for _, b, v, hm in _STEP_EFFECTS[:k]:
        badges += b
        visits += v
        hm_moves += hm
    return badges, visits, hm_moves


def _caps(badges, hm_moves) -> frozenset:
    return frozenset(cap for mv, (cap, badge) in HM_MOVES.items()
                     if mv in hm_moves and badge in badges)


def supervisor_system_prompt() -> str:
    """The training system prompt: the production SYSTEM_PROMPT plus
    the typed-goal vocabulary, derived from goals.py (never copied by
    hand, so the two can't drift apart)."""
    story_targets = ", ".join(dict.fromkeys(
        s.goal_map_name for s in STORY))
    detour_targets = ", ".join(dict.fromkeys(
        s.goal_map_name for s in DETOURS))
    return (
        SYSTEM_PROMPT
        + '\nGoal vocabulary: one kind, "goto", with fields '
          '"target_map_name" and "reason". '
        + f"Story targets: {story_targets}. "
        + f"Optional detour targets: {detour_targets}."
    )


@dataclass
class Scenario:
    context: dict          # exactly what LLMPlanner sends the model
    goal: object           # goals.Goal (the oracle's answer)
    state: GameState
    memory: EpisodicMemory
    source: str            # "synthetic" | "synthetic-detour"


def _start_tile(world: World, map_id: int, rng: random.Random):
    gm = world.maps[map_id]
    warp_tiles = {(wp.x, wp.y) for wp in gm.warps}
    cands = [(x, y) for y in range(gm.h) for x in range(gm.w)
             if gm.walk[y][x] == 1 and (x, y) not in warp_tiles]
    if not cands:
        return None
    return rng.choice(cands[:400])


def _make_state(rng: random.Random, stage: int, map_id: int, tile,
                badges, hm_moves) -> GameState:
    """A plausible mid-run state at story stage `stage` (0 = no
    starter yet). HM moves are dealt round-robin across the party, the
    way a real run spreads them over whoever has a free slot."""
    party: List[PartyMon] = []
    if stage >= 1:
        size = 1 + (1 if stage >= 4 else 0) + (1 if stage >= 8 else 0)
        base_level = min(6 + 4 * stage, 58)
        for i in range(size):
            level = max(2, base_level - rng.randint(0, 6) - 3 * i)
            moves = [rng.choice(_FILLER_MOVES)]
            moves += [mv for j, mv in enumerate(hm_moves)
                      if j % size == i]
            max_hp = 18 + 2 * level
            frac = rng.choices([1.0, 0.75, 0.5], weights=[6, 2, 1])[0]
            party.append(PartyMon(
                level=level,
                hp=max(1, round(max_hp * frac)),
                max_hp=max_hp,
                moves=moves[:4],
                pp=[20] * len(moves[:4]),
                species_id=rng.randint(1, 151),  # placeholder index
            ))
    return GameState(
        map_id=map_id, x=tile[0], y=tile[1], in_battle=False,
        badges=frozenset(badges), money=rng.randint(0, 12000),
        party=party,
    )


def _memory_for(world: World, visits) -> EpisodicMemory:
    memory = EpisodicMemory()
    for name in visits:
        memory.record("visit", map=world.map_id(name))
    return memory


def _candidate_starts(world: World, k: int, caps) -> List[str]:
    """Where a run working on step k plausibly is: the previous step's
    destination, every map on the route from there, and (once Fly is
    in hand) a few fly towns for variety."""
    step = STORY[k]
    goal_name = step.goal_map_name
    names: List[str] = []
    if k == 0:
        names = ["PalletTown", "RedsHouse1F", "RedsHouse2F"]
    else:
        prev_name = STORY[k - 1].goal_map_name
        names.append(prev_name)
        route = world.route(world.map_id(prev_name),
                            world.map_id(goal_name), caps)
        if route is not None:
            names += [world.map_name(m) for m in route.maps]
        if "fly" in caps:
            names += ["PalletTown", "ViridianCity", "CeruleanCity",
                      "CeladonCity"]
    out = []
    for n in names:
        if n != goal_name and n not in out and n in world._name_to_id:
            out.append(n)
    return out


def generate_scenarios(world: World, seed: int = 42,
                       variants: int = 24) -> List[Scenario]:
    """Synthesize validated (context, oracle goal) scenarios across
    the whole story spine plus the Zapdos detour."""
    rng = random.Random(seed)
    scenarios: List[Scenario] = []
    oracle = ChecklistPlanner()
    for k, step in enumerate(STORY):
        badges, visits, hm_moves = _prefix(k)
        caps = _caps(badges, hm_moves)
        for start_name in _candidate_starts(world, k, caps):
            start_id = world.map_id(start_name)
            for _ in range(variants):
                tile = _start_tile(world, start_id, rng)
                if tile is None:
                    continue
                state = _make_state(rng, k, start_id, tile,
                                    badges, hm_moves)
                memory = _memory_for(world, visits)
                if next_step(state, memory, world) is not step:
                    continue  # fixture drift; never emit a mislabeled row
                goal = oracle.propose(state, world, memory)
                if goal is None:
                    continue
                if not validate_goal(goal, state, world, memory).ok:
                    continue
                scenarios.append(Scenario(
                    context=LLMPlanner._context(state, world, memory),
                    goal=goal, state=state, memory=memory,
                    source="synthetic"))
    # Detours: full progression, Surf in hand, Zapdos not yet taken.
    badges, visits, hm_moves = _prefix(len(STORY))
    detour_oracle = ChecklistPlanner(allow_detours=True)
    for start_name in ("CeruleanCity", "FuchsiaCity", "CeladonCity"):
        start_id = world.map_id(start_name)
        for _ in range(variants):
            tile = _start_tile(world, start_id, rng)
            if tile is None:
                continue
            state = _make_state(rng, len(STORY), start_id, tile,
                                badges, hm_moves)
            memory = _memory_for(world, visits)
            goal = detour_oracle.propose(state, world, memory)
            if goal is None or goal.kind != "detour":
                continue
            goal.kind = "goto"  # the wire format has one kind
            if not validate_goal(goal, state, world, memory).ok:
                continue
            scenarios.append(Scenario(
                context=LLMPlanner._context(state, world, memory),
                goal=goal, state=state, memory=memory,
                source="synthetic-detour"))
    return scenarios


def make_example(context: dict, target_map_name: str, reason: str,
                 system_prompt: Optional[str] = None) -> dict:
    return {"messages": [
        {"role": "system",
         "content": system_prompt or supervisor_system_prompt()},
        {"role": "user", "content": json.dumps(context, sort_keys=True)},
        {"role": "assistant", "content": json.dumps(
            {"target_map_name": target_map_name, "reason": reason})},
    ]}


def examples_from_scenarios(scenarios: List[Scenario], world: World):
    return [(make_example(s.context, world.map_name(s.goal.target_map),
                          s.goal.reason), s.source)
            for s in scenarios]


def examples_from_log(records: List[dict], world: World):
    """(example, "log") pairs for proposals the validator accepted and
    the executor completed: decision == 'llm', verdict ok, and a linked
    outcome with executed + success."""
    names = {m.name for m in world.maps.values()}
    proposals = {r["record_id"]: r for r in records
                 if r.get("kind") == "supervisor_proposal"}
    outcomes = {r["record_id"]: r for r in records
                if r.get("kind") == "supervisor_outcome"}
    out = []
    for record_id, rec in proposals.items():
        if rec.get("decision") != "llm":
            continue
        if not (rec.get("verdict") or {}).get("ok"):
            continue
        outcome = outcomes.get(record_id)
        if not outcome or not outcome.get("executed") \
                or not outcome.get("success"):
            continue
        prop = rec.get("proposal") or {}
        target = prop.get("target_map_name")
        if target not in names:
            continue
        out.append((make_example(rec["context"], target,
                                 prop.get("reason", "")), "log"))
    return out


def build_examples(world: World, synthetic_count: int = 800,
                   seed: int = 42, variants: int = 24,
                   log_path: Optional[str | Path] = DEFAULT_SUPERVISOR_LOG):
    """All examples as (example, source) pairs, deduped on the full
    (user, assistant) payload, plus a stats dict."""
    scenarios = generate_scenarios(world, seed=seed, variants=variants)
    rng = random.Random(seed + 1)
    rng.shuffle(scenarios)
    picked = scenarios[:synthetic_count] if synthetic_count else []
    examples = examples_from_scenarios(picked, world)
    stats = {"scenario_pool": len(scenarios),
             "synthetic": sum(1 for _, s in examples
                              if s.startswith("synthetic")),
             "log": 0, "dedup_dropped": 0}
    if log_path is not None:
        log_examples = examples_from_log(
            SupervisorLog.read_records(log_path), world)
        stats["log"] = len(log_examples)
        examples += log_examples
    seen = set()
    deduped = []
    for example, source in examples:
        key = (example["messages"][1]["content"],
               example["messages"][2]["content"])
        if key in seen:
            stats["dedup_dropped"] += 1
            continue
        seen.add(key)
        deduped.append((example, source))
    stats["total"] = len(deduped)
    stats["targets"] = len({
        json.loads(e["messages"][2]["content"])["target_map_name"]
        for e, _ in deduped})
    return deduped, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--count", type=int, default=800,
                    help="synthetic examples to draw from the scenario "
                         "pool (default 800; pool size is the cap)")
    ap.add_argument("--variants", type=int, default=24,
                    help="randomized state variants per (step, start "
                         "map) when generating the pool (default 24)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", default=str(DEFAULT_SUPERVISOR_LOG),
                    help="supervisor log JSONL to mine (default: the "
                         "standard play-workspace path)")
    ap.add_argument("--no-log", action="store_true",
                    help="skip the supervisor log source")
    ap.add_argument("--out", default="dataset/supervisor",
                    help="output directory (default dataset/supervisor)")
    ap.add_argument("--eval-frac", type=float, default=0.05,
                    help="held-out fraction (default 0.05)")
    args = ap.parse_args(argv)

    world = default_world()
    examples, stats = build_examples(
        world, synthetic_count=args.count, seed=args.seed,
        variants=args.variants,
        log_path=None if args.no_log else args.log)
    rng = random.Random(args.seed + 2)
    rng.shuffle(examples)
    total = len(examples)
    eval_n = int(round(total * args.eval_frac))
    if total >= 20 and eval_n == 0:
        eval_n = 1
    eval_set = [e for e, _ in examples[:eval_n]]
    train_set = [e for e, _ in examples[eval_n:]]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_set), ("eval", eval_set)):
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"scenario pool: {stats['scenario_pool']}")
    print(f"examples: {stats['total']} "
          f"(synthetic {stats['synthetic']}, log {stats['log']}, "
          f"dedup dropped {stats['dedup_dropped']})")
    print(f"distinct target maps: {stats['targets']}")
    print(f"wrote {len(train_set)} train + {len(eval_set)} eval "
          f"to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
