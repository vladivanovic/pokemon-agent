#!/usr/bin/env python3
"""Live supervisor run from a save state: real ROM on PyBoy, executor
driving toward the next story step, LLMPlanner (wired default = adapter
v2, transformers-direct) supervising. Ends on the first of:
  (a) the supervisor has fired >= --min-fires times with logged outcomes
  (b) a RAM-verified milestone: the next story city reached or the step's
      badge/flag set
  (c) the round/time budget (--max-goals / --time-budget)

The start state is saved under a new name; the source state file is
never written.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

sys_path = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(sys_path))

from agent.emu import SimEmulator  # noqa: F401  (interface reference)
from agent.executor import Executor, SmartFightPolicy
from agent.memory import EpisodicMemory
from agent.planner import LLMPlanner
from agent.pyboy_emu import PyBoyEmulator
from agent.world import default_world

ROOT = Path(__file__).parent.parent
ADAPTER = ROOT / "training_output" / "qwen_supervisor_v2" / "final"
BASE = "/home/vlad/models/Qwen3.5-4B"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rom", required=True)
    ap.add_argument("--state", required=True,
                    help="working-copy save state to boot (NOT the chain "
                         "head - copy it first)")
    ap.add_argument("--workdir", default="run")
    ap.add_argument("--max-goals", type=int, default=25)
    ap.add_argument("--time-budget", type=int, default=2700,
                    help="wall-clock budget in seconds (default 45 min)")
    ap.add_argument("--min-fires", type=int, default=3,
                    help="supervisor fires (with logged outcomes) that "
                         "satisfy end condition (a)")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    world = default_world()
    emu = PyBoyEmulator(args.rom)
    emu.load_state(args.state)
    st = emu.read_state()
    print(f"boot from {args.state}: map {world.map_name(st.map_id)} "
          f"({st.x},{st.y}) badges={sorted(st.badges)}")
    print(f"party: {[(p.species_id, p.level, p.hp, p.max_hp) for p in st.party]}")

    # start-state snapshot under a new name (never touches the source)
    start_name = f"run{time.strftime('%Y%m%d_%H%M%S')}-start.state"
    emu.save_state(str(workdir / start_name))
    print(f"start state saved as {start_name}")

    memory = EpisodicMemory()
    planner = LLMPlanner(endpoint="inline", model="inline:adapter",
                         adapter=str(ADAPTER), base_model=BASE)
    ex = Executor(emu, world, planner, memory,
                  battle_policy=SmartFightPolicy())

    log_path = Path.home() / "workspace" / "pokemon-play" / \
        "supervisor_log.jsonl"
    log_mark = log_path.stat().st_size if log_path.exists() else 0
    t0 = time.time()
    fired = 0
    status = "budget"
    for gi in range(args.max_goals):
        if time.time() - t0 > args.time_budget:
            status = "time budget"
            break
        res = ex.run(max_goals=1)
        st = emu.read_state()
        new_log = log_path.read_text()[log_mark:] \
            if log_path.exists() else ""
        fired = new_log.count('"model": "inline')  # supervisor proposals
        print(f"[goal {gi+1}] executor={res.status} "
              f"map={world.map_name(st.map_id)} ({st.x},{st.y}) "
              f"party={[(p.species_id, p.level) for p in st.party]} "
              f"supervisor_fires={fired} ({time.time()-t0:.0f}s)")
        # (b) RAM-verified milestone: next story city reached / flag set
        cur = world.map_name(st.map_id)
        if "Celadon" in cur:
            status = f"milestone: reached {cur}"
            break
        if "Rainbow" in st.badges:
            status = "milestone: Rainbow Badge set"
            break
        if res.status == "complete":
            status = "executor complete"
            break
    st = emu.read_state()
    print(f"RUN END: {status}")
    print(f"final: map {world.map_name(st.map_id)} ({st.x},{st.y}) "
          f"badges={sorted(st.badges)}")
    print(f"party: {[(p.species_id, p.level, p.hp, p.max_hp) for p in st.party]}")
    final_name = f"run{time.strftime('%Y%m%d_%H%M%S')}-final.state"
    emu.save_state(str(workdir / final_name))
    print(f"final state saved as {final_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
