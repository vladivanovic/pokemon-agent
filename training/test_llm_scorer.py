#!/usr/bin/env python3
"""Regression test for the schema-constrained LLM scorer (POKEMON_SCORER=llm).

Runs realistic navigation decisions against a live llama.cpp server and
checks the chosen key. These exercise the cases that actually wedged the agent
(the Mart pinball, the shopping goal, dialog stalls) rather than synthetic ones.

Usage:
    python3 test_llm_scorer.py                 # default server
    POKEMON_LLM_URL=http://127.0.0.1:8000/v1 python3 test_llm_scorer.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from pokemon_agent.llm_direct import constrained_choice  # noqa: E402

SYS = ("You are controlling Pokemon Red. Choose the one best action from the "
       "allowed list.")


def ask(objective, situation, crit, want, informational=False):
    listing = "\n".join(f"  - {k}: {v}" for k, v in sorted(crit.items()))
    user = (f"OBJECTIVE: {objective}\n\nSITUATION: {situation}\n\n"
            f"AVAILABLE ACTIONS (choose exactly one key):\n{listing}\n\n"
            f"Reply with the single best key.")
    t = time.time()
    ch, conf = constrained_choice(sorted(crit), SYS, user)
    ok = ch in want
    tag = "OK " if ok else ("note" if informational else "MISS")
    print(f'  [{tag}] {str(ch):26s} want={sorted(want)} '
          f'{time.time()-t:4.1f}s conf={conf:.2f}')
    return ok


# --- what the LLM scorer is FOR -------------------------------------------
# Navigation, doors, menus, dialog, shopping. These are the cases where the
# option keys are opaque and the descriptions carry the decision.
#
# NOT battles. The deterministic Gen 1 policy scores 100% agreement with the
# oracle over 134 recorded move decisions (see evaluate_headroom.py), and the
# LLM does not reliably reason about type effectiveness from bare numbers -
# it picks GRASS (0.25x vs BUG/POISON) over NORMAL (1.0x). Keep battles on the
# policy: it is both more accurate and free.
CASES = [
    # The Viridian Mart pinball: must EXIT, not wander the room.
    ("Reach Pewter City.",
     "You are inside the Viridian Mart. The exit is 4 rows south, off-screen.",
     {"leave_building_F9": "Walk to the exit at F9 and step onto it to return "
                           "to Viridian City",
      "leave_toward_down": "Head down toward the building exit at map (3,7) - "
                           "off-screen 4 rows south",
      "walk_up": "Step one tile up",
      "interact": "Press A to talk to whoever is in front",
      "shop_buy": "Walk to the clerk and buy Poke Balls"},
     {"leave_building_F9", "leave_toward_down"}),
    # Shopping goal: buy, or at least enter the Mart.
    ("Reach Pewter City.",
     "You have no Poke Balls and 500 dollars. You are in Viridian City, the "
     "Mart is here.",
     {"route_toward_north": "Head north toward Pewter City through Viridian "
                            "Forest",
      "route_H1": "Walk to the door at H1, step onto it to enter Viridian Mart",
      "shop_buy": "Walk to the clerk and buy Poke Balls"},
     {"shop_buy", "route_H1"}),
    # Dialog stall: the only correct move is to advance the text.
    ("Reach Pewter City.",
     "A text box is open and blocking movement; you have been here 9 times.",
     {"advance_text": "Press A to advance dialog (REQUIRED - dialog is stalled)",
      "walk_up": "Step one tile up",
      "back_out": "Press B to dismiss or cancel"},
     {"advance_text"}),
    # Continuing a route north.
    ("Reach Pewter City.",
     "You are on Route 2, north of Viridian Forest, heading to Pewter.",
     {"route_toward_north": "Head north toward Pewter City",
      "route_A2": "Walk to the door at A2 to enter the gate",
      "walk_left": "Step one tile left",
      "walk_right": "Step one tile right"},
     {"route_toward_north", "route_A2"}),
]

# Informational: documents the known boundary. Not counted in pass/fail.
INFO_CASES = [
    ("Win the battle.",
     "ENEMY: Weedle Lv5 (BUG/POISON). YOU: Ivysaur (GRASS/POISON) HP 40/40.",
     {"move_0": "Tackle NORMAL power=35 pp=17",
      "move_1": "Growl NORMAL power=0 pp=40",
      "move_2": "Leech Seed GRASS power=0 pp=10",
      "move_3": "Vine Whip GRASS power=35 pp=7",
      "flee": "Run from this wild battle"},
     {"move_0"}),
]


def main():
    print(f"server: {os.environ.get('POKEMON_LLM_URL', 'http://127.0.0.1:8001/v1')}")
    print("navigation cases (what this scorer is for):")
    ok = sum(ask(*c) for c in CASES)
    print("\ninformational - battles stay on the deterministic policy:")
    for c in INFO_CASES:
        ask(*c, informational=True)
    print(f"\n  {ok}/{len(CASES)} correct")
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
