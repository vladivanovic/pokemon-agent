#!/usr/bin/env python3
"""Compare Qwen3.5-0.8B vs 2B on battle decisions AND Hermes tool-use prompts.

Answers two questions:
  1. Does the smaller model still pick the right battle action?
  2. Does it emit well-formed tool calls (the thing the 2B was burning its
     budget on and timing out), and is it faster?

Usage: python3 bench_models.py [--a 8001 --b 8000]
"""
from __future__ import annotations

import argparse
import json
import time

import requests

BATTLE_SYS = (
    "You are a Pokemon Red (Gen 1) battle advisor. Given a compact battle "
    "state, choose the single best action. Answer with ONLY the action token:\n"
    "  move_0..move_3  use your move in that slot\n"
    "  flee            run from a wild battle\n"
    "Rules: Gen 1 uses the original type chart (Poison and Bug are mutually "
    "super effective; Ghost does 0 to Psychic). A move of the same type as "
    "the user gets 1.5x STAB. A move with 0 PP cannot be used."
)

# (prompt, expected) - expected derived from the Gen 1 oracle
BATTLE_CASES = [
    # Weedle/Kakuna are BUG/POISON: Grass is 0.25x, Normal is 1.0x -> Tackle
    ("YOUR POKEMON: bulbasaur types=GRASS/POISON\nYOUR HP: 75/75\n"
     "ENEMY: Weedle Lv5 HP 20/20\nYOUR MOVES:\n"
     "  move_0: Tackle power=35 type=NORMAL pp=17\n"
     "  move_1: Growl power=0 type=NORMAL pp=40\n"
     "  move_2: Leech Seed power=0 type=GRASS pp=10\n"
     "  move_3: Vine Whip power=35 type=GRASS pp=7\nBest action?",
     "move_0"),
    # Same, but Tackle is out of PP -> fall back to the only usable attack
    ("YOUR POKEMON: bulbasaur types=GRASS/POISON\nYOUR HP: 75/75\n"
     "ENEMY: Kakuna Lv5 HP 20/20\nYOUR MOVES:\n"
     "  move_0: Tackle power=35 type=NORMAL pp=0\n"
     "  move_1: Growl power=0 type=NORMAL pp=40\n"
     "  move_2: Leech Seed power=0 type=GRASS pp=10\n"
     "  move_3: Vine Whip power=35 type=GRASS pp=7\nBest action?",
     "move_3"),
    # Pidgey is NORMAL/FLYING: Grass is neutral, Normal is neutral -> Tackle
    ("YOUR POKEMON: bulbasaur types=GRASS/POISON\nYOUR HP: 75/75\n"
     "ENEMY: Pidgey Lv4 HP 16/16\nYOUR MOVES:\n"
     "  move_0: Tackle power=35 type=NORMAL pp=12\n"
     "  move_1: Growl power=0 type=NORMAL pp=40\n"
     "  move_2: Leech Seed power=0 type=GRASS pp=10\n"
     "  move_3: Vine Whip power=35 type=GRASS pp=6\nBest action?",
     "move_0"),
]

# The tool-use prompt: what the autopilot's Hermes escalation asks for.
TOOL_SYS = (
    "You are controlling Pokemon Red through an HTTP API. Reply with EXACTLY "
    "one JSON object and nothing else.\n"
    'To press a button: {"actions":["press_a"]}\n'
    'To walk north 6 tiles: {"actions":["walk_up","walk_up","walk_up",'
    '"walk_up","walk_up","walk_up"]}'
)
TOOL_CASES = [
    ("The player is wedged against a wall to the north. Walk north for 6 tiles.",
     "walk_up"),
    ("You are facing the Poke Mart clerk. Press A to talk.", "press_a"),
    ("Step south one tile to leave the building.", "walk_down"),
]

HDRS = {"Content-Type": "application/json"}


def call(port: int, sys_msg: str, user_msg: str, max_tokens: int = 48):
    body = {
        "messages": [{"role": "system", "content": sys_msg},
                     {"role": "user", "content": user_msg}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
    }
    t = time.time()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      headers=HDRS, json=body, timeout=180)
    dt = time.time() - t
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"], dt, j.get("usage", {})


def run(label: str, port: int):
    print(f"\n{'='*66}\n{label}  (port {port})\n{'='*66}")
    ok_b = tot_b = 0
    ok_t = tot_t = 0
    times = []
    for prompt, want in BATTLE_CASES:
        try:
            out, dt, usage = call(port, BATTLE_SYS, prompt)
        except Exception as e:
            print(f"  battle ERROR: {type(e).__name__}: {str(e)[:70]}")
            continue
        tok = usage.get("completion_tokens", 0)
        hit = want in out
        ok_b += hit
        tot_b += 1
        times.append(dt)
        print(f"  battle  want={want:8s} got={out.strip()[:22]:22s} "
              f"{'OK ' if hit else 'MISS'} {dt:5.1f}s "
              f"{tok/max(dt,1e-6):5.1f} tok/s")
    for prompt, want in TOOL_CASES:
        try:
            out, dt, usage = call(port, TOOL_SYS, prompt)
        except Exception as e:
            print(f"  tool ERROR: {type(e).__name__}: {str(e)[:70]}")
            continue
        # valid = parses as JSON with an actions list, and hits the intent
        valid = False
        try:
            j = json.loads(out.strip().strip("`").replace("json\n", ""))
            valid = isinstance(j.get("actions"), list) and len(j["actions"]) > 0
        except Exception:
            valid = False
        hit = valid and (want in out)
        ok_t += hit
        tot_t += 1
        times.append(dt)
        tok = usage.get("completion_tokens", 0)
        print(f"  tool    want={want:8s} json={'Y' if valid else 'N'} "
              f"{'OK ' if hit else 'MISS'} {dt:5.1f}s "
              f"{tok/max(dt,1e-6):5.1f} tok/s  {out.strip()[:40]!r}")
    print(f"\n  battle {ok_b}/{tot_b}   tool {ok_t}/{tot_t}   "
          f"mean {sum(times)/max(len(times),1):.1f}s")
    return ok_b, tot_b, ok_t, tot_t, (sum(times)/max(len(times), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=8001, help="0.8B port")
    ap.add_argument("--b", type=int, default=8000, help="2B port")
    args = ap.parse_args()
    ra = run("Qwen3.5-0.8B", args.a)
    rb = run("Qwen3.5-2B (current)", args.b)
    print(f"\n{'='*66}\nSUMMARY\n{'='*66}")
    print(f"  0.8B: battle {ra[0]}/{ra[1]}  tool {ra[2]}/{ra[3]}  "
          f"mean {ra[4]:.1f}s")
    print(f"  2B  : battle {rb[0]}/{rb[1]}  tool {rb[2]}/{rb[3]}  "
          f"mean {rb[4]:.1f}s")


if __name__ == "__main__":
    main()
