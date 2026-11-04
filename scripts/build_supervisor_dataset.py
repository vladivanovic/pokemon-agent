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

from agent.goals import DETOURS, STORY, Goal, next_step, validate_goal
from agent.memory import EpisodicMemory, state_signature
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


# --- recovery rows (round 2) -------------------------------------------------
# Battery v2 (docs/supervisor-training.md section 6) showed the v1 adapter
# proposes the already-failed goal in every stuck scenario: its training
# set never contained failure events. Recovery rows pair a production-
# faithful context (failure events present) with a DIFFERENT validated
# target, chosen by a priority oracle: heal when the party is hurt, the
# Zapdos detour when Surf is in hand, otherwise retreat to the previous
# story town.

POKECENTER_TOWNS = ["PewterCity", "CeruleanCity", "VermilionCity",
                    "CeladonCity", "FuchsiaCity", "SaffronCity",
                    "CinnabarIsland", "ViridianCity"]


def _failed_memory(state: GameState, world: World, step, n: int = 2) -> (
        EpisodicMemory):
    """Memory as production sees it after the deterministic goal failed
    n times in this unchanged situation."""
    mem = EpisodicMemory()
    key = Goal(kind="goto",
               target_map=world.map_id(step.goal_map_name)).key()
    sig = state_signature(state)
    for _ in range(n):
        mem.record_goal_result(key, success=False, state_signature=sig,
                               note="executor press budget exhausted")
    return mem


def _recovery_goal(state: GameState, world: World, memory: EpisodicMemory,
                   k: int) -> tuple[Optional[Goal], str]:
    caps = state.capabilities()
    # 1. heal: the party is hurt and a Pokecenter town is reachable
    if state.party and state.team_hp_frac() < 0.6:
        best = None
        for name in POKECENTER_TOWNS:
            mid = world._name_to_id.get(name)
            if mid is None or mid == state.map_id:
                continue
            route = world.route(state.map_id, mid, caps)
            if route is not None and (best is None or len(route) < len(best)):
                best = route
        if best is not None:
            return (Goal(kind="goto", target_map=best.maps[-1],
                         reason="Recovery: heal the party at a Pokecenter "
                                "after repeated failures of the main "
                                "goal"),
                    "recovery-heal")
    # 2. Zapdos detour when Surf is in hand
    if "surf" in caps and not memory.succeeded("detour:zapdos"):
        det = DETOURS[0]
        goal = Goal(kind="goto",
                    target_map=world.map_id(det.goal_map_name),
                    reason="Recovery: take the Zapdos detour while the "
                           "main path is blocked")
        if validate_goal(goal, state, world, memory).ok:
            return goal, "recovery-detour"
    # 3. retreat: the previous story town, re-approach from there
    if k >= 1:
        prev = STORY[k - 1]
        goal = Goal(kind="goto",
                    target_map=world.map_id(prev.goal_map_name),
                    reason="Recovery: fall back to " + prev.goal_map_name
                           + " and re-approach")
        if validate_goal(goal, state, world, memory).ok:
            return goal, "recovery-retreat"
    return None, ""


def generate_recovery_scenarios(world: World, seed: int = 43,
                                variants: int = 10) -> List[Scenario]:
    """Stuck-state scenarios across the battery's three families
    (midroute / postbadge / lowhp), each with the deterministic goal
    failed 2x in memory, paired with the recovery-oracle target."""
    rng = random.Random(seed)
    out: List[Scenario] = []
    seen = set()
    plan = ([("midroute", k) for k in (2, 3, 4, 5, 6, 7)]
            + [("postbadge", k) for k in (5, 6, 7, 8)]
            + [("lowhp", k) for k in (3, 4, 5, 6)])
    for family, k in plan:
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
                if family == "lowhp":
                    state = GameState(
                        map_id=state.map_id, x=state.x, y=state.y,
                        in_battle=False, badges=state.badges,
                        money=state.money,
                        party=[PartyMon(level=m.level,
                                        hp=max(1, round(m.max_hp * 0.15)),
                                        max_hp=m.max_hp, moves=m.moves,
                                        pp=m.pp, species_id=m.species_id)
                               for m in state.party])
                step = STORY[k]
                memory = _memory_for(world, visits)
                memory.events.extend(
                    _failed_memory(state, world, step).events)
                if next_step(state, memory, world) is not step:
                    continue
                goal, source = _recovery_goal(state, world, memory, k)
                if goal is None:
                    continue
                if goal.target_map == world.map_id(step.goal_map_name):
                    continue  # recovery must differ from the failed goal
                if not validate_goal(goal, state, world, memory).ok:
                    continue
                context = LLMPlanner._context(state, world, memory)
                key = (json.dumps(context, sort_keys=True),
                       world.map_name(goal.target_map))
                if key in seen:
                    continue
                seen.add(key)
                out.append(Scenario(context=context, goal=goal, state=state,
                                    memory=memory, source=source))
    return out


def audit_recovery_rows(examples, world: World) -> dict:
    """Mechanical audit: no recovery row may target the failed goal.
    The failed goal(s) are read back from each row's own context
    (recent_events goal_result failures), the target from the assistant
    turn."""
    name_to_id = {m.name: mid for mid, m in world.maps.items()}
    bad = 0
    targets = {}
    for example, source in examples:
        if source != "recovery":
            continue
        ctx = json.loads(example["messages"][1]["content"])
        target = json.loads(example["messages"][2]["content"])
        tgt_id = name_to_id.get(target["target_map_name"])
        targets[target["target_map_name"]] = \
            targets.get(target["target_map_name"], 0) + 1
        failed_ids = set()
        for ev in ctx.get("recent_events", []):
            if ev.get("kind") == "goal_result" and not ev.get("success"):
                key = ev.get("goal", "")          # "goto:<id>:<xy>"
                parts = key.split(":")
                if len(parts) >= 2 and parts[1].isdigit():
                    failed_ids.add(int(parts[1]))
        if tgt_id in failed_ids:
            bad += 1
    return {"recovery_rows": sum(1 for _, s in examples
                                 if s == "recovery"),
            "failed_goal_as_target": bad,
            "target_map_counts": targets}


def examples_from_battery_log(records: List[dict], world: World):
    """Real stock recoveries from battery runs (model battery-*): the
    executor completed the goal. Returns a single list of
    (example, "battery-log") pairs; caller splits train/holdout.
    Note: the battery logs verdict=None (scoring lives in
    battery_results.json), so acceptance is judged by the outcome."""
    names = {m.name for m in world.maps.values()}
    proposals = {r["record_id"]: r for r in records
                 if r.get("kind") == "supervisor_proposal"
                 and str(r.get("model", "")).startswith("battery-stock")}
    outcomes = {r["record_id"]: r for r in records
                if r.get("kind") == "supervisor_outcome"}
    pairs = []
    for record_id, rec in proposals.items():
        if rec.get("decision") != "battery":
            continue
        outcome = outcomes.get(record_id)
        if not outcome or not outcome.get("executed") \
                or not outcome.get("success"):
            continue
        prop = rec.get("proposal") or {}
        target = prop.get("target_map_name")
        if target not in names:
            continue
        pairs.append((make_example(rec["context"], target,
                                   prop.get("reason", "")), "battery-log"))
    return pairs


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
                   log_path: Optional[str | Path] = DEFAULT_SUPERVISOR_LOG,
                   recovery_count: int = 0,
                   battery_train: int = 0,
                   battery_holdout: int = 4):
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
    if recovery_count > 0:
        rec_scenarios = generate_recovery_scenarios(
            world, seed=seed + 17, variants=10)
        rng2 = random.Random(seed + 18)
        rng2.shuffle(rec_scenarios)
        rec_examples = examples_from_scenarios(rec_scenarios[:recovery_count],
                                               world)
        # recovery rows are labeled by their oracle source, not the
        # generic "synthetic", so the audit can isolate them
        rec_examples = [(e, "recovery") for e, _ in rec_examples]
        stats["recovery_pool"] = len(rec_scenarios)
        examples += rec_examples
    if battery_train > 0 or battery_holdout > 0:
        b_all = examples_from_battery_log(
            SupervisorLog.read_records(log_path), world)
        rng3 = random.Random(seed + 19)
        rng3.shuffle(b_all)
        hold_n = min(battery_holdout, len(b_all) // 2)
        b_hold = b_all[:hold_n]
        b_train = b_all[hold_n:hold_n + battery_train]
        stats["battery_pool"] = len(b_all)
        stats["battery_train"] = len(b_train)
        stats["battery_holdout"] = len(b_hold)
        examples += b_train
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
    holdout = []
    if battery_holdout > 0:
        # battery-log holdout rows were never added to examples; rebuild
        # them here from the same seeded split for the caller
        b_all = examples_from_battery_log(
            SupervisorLog.read_records(log_path), world)
        rng4 = random.Random(seed + 19)
        rng4.shuffle(b_all)
        holdout = b_all[:min(battery_holdout, len(b_all) // 2)]
    return deduped, stats, holdout


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
    ap.add_argument("--recovery", type=int, default=0,
                    help="recovery examples to add (stuck-state scenarios "
                         "with failure events; 0 = round-1 behavior)")
    ap.add_argument("--battery-train", type=int, default=0,
                    help="real stock-recovery rows from the battery log to "
                         "train on (0 = skip)")
    ap.add_argument("--battery-holdout", type=int, default=0,
                    help="battery rows held out of training to audit the "
                         "synthetic recovery rows against (written to "
                         "battery_holdout.jsonl)")
    ap.add_argument("--eval-frac", type=float, default=0.05,
                    help="held-out fraction (default 0.05)")
    args = ap.parse_args(argv)

    world = default_world()
    examples, stats, holdout = build_examples(
        world, synthetic_count=args.count, seed=args.seed,
        variants=args.variants,
        log_path=None if args.no_log else args.log,
        recovery_count=args.recovery,
        battery_train=args.battery_train,
        battery_holdout=args.battery_holdout)
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
          f"recovery {stats.get('recovery_pool', 0)} pool / "
          f"{sum(1 for _, s in examples if s == 'recovery')} used, "
          f"battery train {stats.get('battery_train', 0)} "
          f"holdout {stats.get('battery_holdout', 0)}, "
          f"dedup dropped {stats['dedup_dropped']})")
    audit = audit_recovery_rows(examples, world)
    print(f"recovery audit: {audit['recovery_rows']} rows, "
          f"failed-goal-as-target = {audit['failed_goal_as_target']} "
          f"(must be 0); targets: {audit['target_map_counts']}")
    if holdout:
        hold_path = out_dir / "battery_holdout.jsonl"
        with hold_path.open("w") as f:
            for row in holdout:
                f.write(json.dumps(row) + "\n")
        print(f"wrote {len(holdout)} battery holdout rows to {hold_path}")
    print(f"distinct target maps: {stats['targets']}")
    print(f"wrote {len(train_set)} train + {len(eval_set)} eval "
          f"to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
