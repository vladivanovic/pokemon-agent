#!/usr/bin/env python3
"""Stuck-state battery: stock Qwen3.5-4B vs the trained supervisor LoRA.

Production-faithful: every scenario builds the exact context dict
LLMPlanner sends (state digest, current map, next story step, recent
events - with failure events present because the executor failed the
deterministic goal repeatedly), both contenders propose through the same
propose-and-verify validator (goals.validate_goal), and every round is
recorded through SupervisorLog so the battery feeds the next training
round either way.

Families:
  midroute   - mid-story, deterministic goal failed 2x (loop blockage)
  postbadge  - badge just won, next-step goal failed 2x (decision point)
  lowhp      - party badly hurt, deterministic goal failed 2x (stress)
  control    - deterministic path works; supervisor must not do worse
               than the checklist fallback

Scoring: validity (validator ok), unstick (valid AND goal completes -
i.e. route exists AND target differs from the suppressed loop goal),
advance (target is the story step's goal map). Ship rule: the adapter
beats stock on unstick rate without losing validity or controls.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.planner import SYSTEM_PROMPT, ChecklistPlanner, LLMPlanner
from agent.goals import STORY, DETOURS, Goal, next_step, validate_goal
from agent.state import GameState, PartyMon
from agent.memory import EpisodicMemory, state_signature
from agent.supervisor_log import SupervisorLog
from agent.world import World
from scripts.build_supervisor_dataset import (
    _caps, _candidate_starts, _make_state, _memory_for, _prefix,
    _start_tile, supervisor_system_prompt,
)

ROOT = Path(__file__).parent.parent
BASE = "/home/vlad/models/Qwen3.5-4B"
ADAPTER = ROOT / "training_output" / "qwen_supervisor" / "final"
LOG_PATH = Path.home() / "workspace" / "pokemon-play" / "supervisor_log.jsonl"
PRESS_BUDGET = 200_000  # executor's max_presses; generous by design
SEED = 1234

rng = random.Random(SEED)
world = World(ROOT / "data" / "maps")
oracle = ChecklistPlanner()


def goal_key_for(name: str) -> str:
    g = Goal(kind="goto", target_map=world.map_id(name))
    return g.key()


def failed_memory(state, world, step, n=2) -> EpisodicMemory:
    """Memory as production sees it after the deterministic goal failed
    n times in this unchanged situation (the blockage marker)."""
    mem = EpisodicMemory()
    key = Goal(kind="goto", target_map=world.map_id(step.goal_map_name)).key()
    sig = state_signature(state)
    for _ in range(n):
        mem.record_goal_result(key, success=False, state_signature=sig,
                               note="executor press budget exhausted")
    return mem


def make_scenario(family: str, k: int, start_name: str, low_hp=False):
    badges, visits, hm_moves = _prefix(k)
    caps = _caps(badges, hm_moves)
    start_id = world.map_id(start_name)
    tile = _start_tile(world, start_id, rng)
    if tile is None:
        return None
    state = _make_state(rng, k, start_id, tile, badges, hm_moves)
    if low_hp:
        state = GameState(
            map_id=state.map_id, x=state.x, y=state.y,
            in_battle=False, badges=state.badges, money=state.money,
            party=[PartyMon(level=m.level,
                            hp=max(1, round(m.max_hp * 0.15)),
                            max_hp=m.max_hp, moves=m.moves, pp=m.pp,
                            species_id=m.species_id)
                   for m in state.party])
    if family == "control":
        memory = _memory_for(world, visits)
        failed_key = None
    else:
        step = STORY[k]
        memory = _memory_for(world, visits)
        memory.events.extend(failed_memory(state, world, step).events)
        failed_key = Goal(kind="goto",
                          target_map=world.map_id(step.goal_map_name)).key()
    if next_step(state, memory, world) is None:
        return None
    return {"family": family, "k": k, "state": state, "memory": memory,
            "context": LLMPlanner._context(state, world, memory),
            "failed_key": failed_key,
            "step_map": world.map_id(STORY[k].goal_map_name),
            "step_title": STORY[k].title}


def build_scenarios():
    scs = []

    def towns(k, n):
        b, v, hm = _prefix(k)
        cands = _candidate_starts(world, k, _caps(b, hm))
        return (cands * 3)[:n]

    # midroute: blockage mid-story, healthy party
    for k in (2, 3, 4, 5, 6, 7):
        for st in towns(k, 2):
            s = make_scenario("midroute", k, st)
            if s:
                scs.append(s)
    # postbadge: badge just won (prefix k), next-step goal failed; surf
    # in hand for k>=5 so the Zapdos detour is a legitimate answer
    for k in (5, 6, 7, 8):
        for st in towns(k, 2):
            s = make_scenario("postbadge", k, st)
            if s:
                scs.append(s)
    # lowhp: party badly hurt, blockage; healing via a Pokecenter town
    # is a legitimate goal
    for k in (3, 4, 5, 6):
        for st in towns(k, 2):
            s = make_scenario("lowhp", k, st, low_hp=True)
            if s:
                scs.append(s)
    # control: deterministic path works, no failures
    for k in (1, 2, 3, 4, 5, 6):
        for st in towns(k, 2):
            s = make_scenario("control", k, st)
            if s:
                scs.append(s)
    return scs


def load_contender(adapter: Path | None):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@torch.no_grad()
def propose(model, tok, context: dict) -> tuple[Goal | None, str, str]:
    """Production-faithful propose: SYSTEM_PROMPT + exact LLMPlanner
    context as the user turn, temperature 0.2, JSON extraction like
    LLMPlanner._ask. Returns (goal|None, raw_text, error)."""
    msgs = [
        {"role": "system", "content": supervisor_system_prompt()},
        {"role": "user", "content": json.dumps(context)},
    ]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", enable_thinking=False)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    out = model.generate(input_ids=ids.to("cuda"), max_new_tokens=512,
                         do_sample=True, temperature=0.2, top_p=1.0)
    text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    try:
        # first BALANCED {...} object - the models wrap JSON in prose
        start = text.index("{")
        depth = 0
        end = None
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
        if end is None:
            raise ValueError("no balanced JSON object in output")
        parsed = json.loads(text[start:end])
        goal = Goal(kind="goto",
                    target_map=world.map_id(parsed["target_map_name"]),
                    reason=str(parsed.get("reason", ""))[:200])
        return goal, text, ""
    except Exception as e:
        return None, text, f"{type(e).__name__}: {e}"


def score(goal: Goal | None, sc: dict, raw: str, err: str):
    step_map = sc["step_map"]
    failed_key = sc["failed_key"]
    r = {"goal": None, "valid": False, "unstick": False, "advance": False,
         "loop": False, "problems": []}
    if goal is None:
        r["problems"] = [err or "no proposal"]
        return r
    r["goal"] = goal.key()
    verdict = validate_goal(goal, sc["state"], world, sc["memory"])
    r["valid"] = verdict.ok
    r["problems"] = verdict.problems
    if not verdict.ok:
        if failed_key and goal.key() == failed_key:
            r["loop"] = True
        return r
    # completes: route exists (validator checked) and it is not the
    # suppressed loop goal
    r["unstick"] = True
    if goal.target_map == step_map:
        r["advance"] = True
    return r


def main():
    scenarios = build_scenarios()
    print(f"scenarios: {len(scenarios)} "
          f"({ {f: sum(1 for s in scenarios if s['family']==f) for f in ('midroute','postbadge','lowhp','control')} })")
    log = SupervisorLog(LOG_PATH)
    results = {}

    for label, adapter in (("stock", None), ("adapter", ADAPTER)):
        print(f"=== loading {label} ===")
        tok, model = load_contender(adapter)
        rows = []
        for i, sc in enumerate(scenarios):
            t0 = time.time()
            goal, raw, err = propose(model, tok, sc["context"])
            sc_res = score(goal, sc, raw, err)
            step = next_step(sc["state"], sc["memory"], world)
            rec_id = log.record_proposal(
                model=f"battery-{label}", context=sc["context"],
                state=sc["state"], world=world, step=step,
                proposal=goal, proposal_error=err,
                verdict=None, decision="battery", returned_goal=goal)
            unstick = sc_res["unstick"]
            log.record_outcome(rec_id, goal_key=goal.key() if goal else "?",
                               executed=unstick, success=unstick,
                               note=f"battery {sc['family']}") \
                if rec_id else None
            rows.append({"family": sc["family"], "k": sc["k"], **sc_res,
                         "secs": round(time.time() - t0, 1)})
            print(f"  [{i+1}/{len(scenarios)}] {sc['family']} k={sc['k']} "
                  f"valid={sc_res['valid']} unstick={sc_res['unstick']} "
                  f"goal={sc_res['goal']} ({rows[-1]['secs']}s)")
        results[label] = rows
        del model
        torch.cuda.empty_cache()

    # ---- per-family table + ship decision ----
    fams = ("midroute", "postbadge", "lowhp", "control")

    def pct(rows, fam, key):
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            return 0.0, 0
        return 100.0 * sum(1 for r in sub if r[key]) / len(sub), len(sub)

    print("\n===== RESULTS (per family) =====")
    print(f"{'family':<10} {'metric':<9} {'stock':>7} {'adapter':>8}")
    summary = {}
    for fam in fams:
        for metric in ("valid", "unstick", "advance"):
            s, n = pct(results["stock"], fam, metric)
            a, _ = pct(results["adapter"], fam, metric)
            summary[(fam, metric)] = (s, a, n)
            print(f"{fam:<10} {metric:<9} {s:>6.0f}% {a:>7.0f}%   (n={n})")

    stuck = ("midroute", "postbadge", "lowhp")
    def avg(rows, fams, key):
        vals = [r[key] for r in rows if r["family"] in fams]
        return 100.0 * sum(vals) / len(vals) if vals else 0.0
    s_unstick = avg(results["stock"], stuck, "unstick")
    a_unstick = avg(results["adapter"], stuck, "unstick")
    s_valid = avg(results["stock"], stuck, "valid")
    a_valid = avg(results["adapter"], stuck, "valid")
    s_ctrl = avg(results["stock"], ("control",), "unstick")
    a_ctrl = avg(results["adapter"], ("control",), "unstick")
    ship = a_unstick > s_unstick and a_valid >= s_valid and a_ctrl >= s_ctrl
    print(f"\nSTUCK-FAMILY UNSTICK: stock {s_unstick:.0f}% "
          f"vs adapter {a_unstick:.0f}%")
    print(f"CONTROLS unstick:     stock {s_ctrl:.0f}% "
          f"vs adapter {a_ctrl:.0f}%  (fallback baseline: 100%)")
    print(f"DECISION: {'SHIP adapter' if ship else 'KEEP STOCK, keep logging'}")
    (ROOT / "training_output" / "qwen_supervisor" / "battery_results.json"
     ).write_text(json.dumps({"scenarios": len(scenarios),
                              "results": results,
                              "summary": {f"{k[0]}:{k[1]}": v
                                          for k, v in summary.items()},
                              "ship": ship}, indent=1, default=str))


if __name__ == "__main__":
    main()
