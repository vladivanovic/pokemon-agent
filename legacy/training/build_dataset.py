#!/usr/bin/env python3
"""Turn recorded battle turns into SFT pairs, labelled by the Gen 1 oracle.

Output is chat-format JSONL ({"messages":[...]}) ready for TRL's SFTTrainer,
plus a plain-prompt JSONL for llama.cpp/other trainers.

The oracle labels every FIGHT decision; the recorded non-move actions
(flee/switch/item) are kept with their own recorded label so the model can
learn the run-vs-fight policy too - that is where the current agent is
weakest (it flees ~59% of wild battles, since it carries no Poke Balls).

Usage:
    python3 build_dataset.py --out data/battles.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional

from gen1_oracle import (ID_TYPES, load_base_stats, load_moves, move_score,
                         _norm_move, SPECIAL_TYPES)

DEFAULT_RECORDER = os.path.expanduser("~/.pokemon-agent/flight_recorder.jsonl")

SYS = (
    "You are a Pokemon Red (Gen 1) battle advisor. Given a compact battle "
    "state, choose the single best action. Answer with ONLY the action token:\n"
    "  move_0..move_3  use your move in that slot\n"
    "  flee            run from a wild battle\n"
    "Rules: Gen 1 uses the original type chart (Poison and Bug are mutually "
    "super effective; Ghost does 0 to Psychic). A move of the same type as "
    "the user gets 1.5x STAB. A move with 0 PP cannot be used."
)


def _types_of(stats: Dict, name: Optional[str]) -> List[int]:
    if not name:
        return []
    k = str(name).strip().lower().replace(".", "").replace("'", "")
    if k in stats:
        return stats[k]["types"]
    for kk in stats:
        if len(k) >= 3 and kk.startswith(k[:4]):
            return stats[kk]["types"]
    return []


def _rec_types(battle: dict, key: str):
    """Type ids recorded live, if present (newer flight-recorder records)."""
    from gen1_oracle import type_id_from_name
    out = []
    for t in ((battle.get(key) or {}).get('types') or []):
        tid = type_id_from_name(t)
        if tid is not None:
            out.append(tid)
    return out


def _render_state(player_name, p_types, enemy, moves_pp, hp) -> str:
    lines = [f"YOUR POKEMON: {player_name}  types={'/'.join(ID_TYPES.get(t,'?') for t in p_types)}"]
    if hp:
        lines.append(f"YOUR HP: {hp[0]}/{hp[1]}")
    if enemy and enemy[0]:
        e_types = []
        lines.append(f"ENEMY: {enemy[0]}  Lv{enemy[1]}  "
                     f"HP {enemy[2]}/{enemy[3]}")
    lines.append("YOUR MOVES:")
    for i, (name, pp) in enumerate(moves_pp or []):
        mv = load_moves().get(_norm_move(name))
        if mv:
            lines.append(f"  move_{i}: {name}  power={mv['power']} "
                         f"type={mv['type']} pp={pp}")
        else:
            lines.append(f"  move_{i}: {name}  pp={pp}")
    lines.append("Best action?")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorder", default=DEFAULT_RECORDER)
    ap.add_argument("--player", default="bulbasaur")
    ap.add_argument("--out", default="data/battles.jsonl")
    ap.add_argument("--chat-out", default="data/battles_chat.jsonl")
    ap.add_argument("--label", choices=["oracle", "recorded", "both"],
                    default="both",
                    help="oracle: relabel fights; recorded: keep as-is; both")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    stats = load_base_stats()
    p_types = _types_of(stats, args.player)
    rows = []
    with open(args.recorder, encoding="utf-8") as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            b = d.get("battle") or {}
            if b.get("in_battle"):
                rows.append(d)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plain, chat = [], []
    stats_out = {"pairs": 0, "oracle_labelled": 0, "recorded_labelled": 0,
                 "skipped": 0}

    for d in rows:
        b = d.get("battle") or {}
        enemy = b.get("enemy") or []
        moves_pp = b.get("active_moves") or []
        choice = d.get("choice") or ""
        if not enemy or not enemy[0] or not moves_pp:
            stats_out["skipped"] += 1
            continue
        state = _render_state(args.player, p_types, enemy, moves_pp,
                              b.get("my_hp"))
        e_types = _rec_types(b, 'enemy_full') or _types_of(stats, enemy[0])

        target = None
        origin = None
        if choice.startswith("move_") and e_types and args.label in ("oracle", "both"):
            best, best_s = None, None
            for i, (name, pp) in enumerate(moves_pp):
                if pp is None or int(pp) <= 0:
                    continue
                if _norm_move(name) not in load_moves():
                    continue
                s = move_score(name, p_types, e_types, 100, 100)
                if s < 0:
                    continue
                if best_s is None or s > best_s:
                    best, best_s = i, s
            if best is not None:
                target = f"move_{best}"
                origin = "oracle"
        if target is None and args.label in ("recorded", "both"):
            target = choice or None
            origin = "recorded"
        if target is None:
            stats_out["skipped"] += 1
            continue

        plain.append({"prompt": SYS + "\n\n" + state, "completion": target,
                      "origin": origin})
        chat.append({"messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": state},
            {"role": "assistant", "content": target},
        ]})
        stats_out["pairs"] += 1
        stats_out[f"{origin}_labelled"] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for r in plain:
            f.write(json.dumps(r) + "\n")
    with open(args.chat_out, "w", encoding="utf-8") as f:
        for r in chat:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {args.out} ({len(plain)} pairs)")
    print(f"wrote {args.chat_out} ({len(chat)} chat pairs)")
    print(json.dumps(stats_out, indent=2))
    if len(plain) < 500:
        print(f"\nNOTE: {len(plain)} pairs is far too few for SFT. "
              f"Collect more battles before training.")


if __name__ == "__main__":
    main()
