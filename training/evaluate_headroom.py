#!/usr/bin/env python3
"""Measure the headroom between the current battle policy and the Gen 1 oracle.

Reads ~/.pokemon-agent/flight_recorder.jsonl (every turn: state + action) and
answers ONE question: if the recorded battles had been played by the oracle
(type chart + power + STAB + accuracy, Gen 1 rules), how often would it have
chosen a different move - and how much better is its choice?

This is the "is training worth it?" gate. If the policy already agrees with the
oracle ~100% of the time, SFT on this data teaches nothing. If it disagrees a
lot, we have both (a) a signal worth learning and (b) an automatic labeller.

Usage:
    python3 evaluate_headroom.py [--player bulbasaur] [--recorder PATH]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import List, Optional, Tuple

from gen1_oracle import (load_base_stats, load_moves, move_score,
                         _norm_move)

DEFAULT_RECORDER = os.path.expanduser("~/.pokemon-agent/flight_recorder.jsonl")


def _norm_species(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return str(name).strip().lower().replace(".", "").replace("'", "")


def _species_types(stats: dict, name: Optional[str]) -> List[int]:
    key = _norm_species(name)
    if not key:
        return []
    if key in stats:
        return stats[key]["types"]
    # tolerate "Nidoran♂"/"Nidoran-F" style spellings
    for k in stats:
        if k.startswith(key[:4]) and len(key) >= 3:
            return stats[k]["types"]
    return []


def _recorded_types(battle: dict, key: str) -> List[int]:
    """Type ids recorded live (enemy_full/player), if present."""
    from gen1_oracle import type_id_from_name
    out = []
    for t in ((battle.get(key) or {}).get('types') or []):
        tid = type_id_from_name(t)
        if tid is not None:
            out.append(tid)
    return out


def load_battle_turns(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            b = d.get("battle") or {}
            if b.get("in_battle"):
                rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default="bulbasaur",
                    help="player species name (default bulbasaur)")
    ap.add_argument("--recorder", default=DEFAULT_RECORDER)
    ap.add_argument("--show", type=int, default=12,
                    help="how many disagreements to print")
    args = ap.parse_args()

    stats = load_base_stats()
    moves = load_moves()
    rows = load_battle_turns(args.recorder)
    if not rows:
        print("no battle turns found")
        return

    p_types = _species_types(stats, args.player)
    if not p_types:
        print(f"WARNING: player species {args.player!r} not found in base_stats; "
              f"STAB will be treated as absent")
    print(f"player species : {args.player} types={p_types}")
    print(f"battle turns   : {len(rows)}")
    print()

    # Menu index -> move slot. The recorder stores menu_index for the FIGHT
    # selection; move_N in `choice` is the slot actually pressed.
    agree = 0
    disagree = 0
    no_move = 0                     # flee / item / switch, not a move decision
    species_seen = Counter()
    disagree_pairs = Counter()
    examples: List[Tuple] = []

    for d in rows:
        b = d.get("battle") or {}
        enemy = b.get("enemy") or []
        moves_pp = b.get("active_moves") or []
        choice = d.get("choice") or ""
        if not enemy or not enemy[0]:
            continue
        species_seen[enemy[0]] += 1
        if not choice.startswith("move_"):
            no_move += 1
            continue
        try:
            slot = int(choice.split("_")[1])
        except (ValueError, IndexError):
            continue
        # Prefer the TYPES as recorded live; fall back to species lookup for
        # records written before the full battle state was captured.
        e_types = _recorded_types(b, 'enemy_full') or _species_types(stats, enemy[0])
        if not e_types:
            continue
        oracle = None
        best_s = None
        for name, pp in moves_pp:
            if pp is None or int(pp) <= 0:
                continue
            if _norm_move(name) not in moves:
                continue
            s = move_score(name, p_types, e_types, atk_stat=100, spc_stat=100)
            if s < 0:
                continue
            if best_s is None or s > best_s:
                oracle, best_s = name, s
        if oracle is None:
            continue
        # What the policy actually used
        used = None
        if 0 <= slot < len(moves_pp):
            used = moves_pp[slot][0]
        if used == oracle:
            agree += 1
        else:
            disagree += 1
            disagree_pairs[(used, oracle, enemy[0])] += 1
            if len(examples) < args.show:
                used_s = move_score(used, p_types, e_types, 100, 100)
                examples.append((d.get("turn"), enemy[0], e_types, used,
                                 used_s, oracle, best_s))

    total_moves = agree + disagree
    print(f"move decisions : {total_moves}")
    print(f"  oracle agrees: {agree} ({100*agree/max(total_moves,1):.1f}%)")
    print(f"  DISAGREES    : {disagree} ({100*disagree/max(total_moves,1):.1f}%)")
    print(f"non-move turns : {no_move} (flee/switch/item)")
    print()
    print("enemy species seen:", dict(species_seen.most_common(10)))
    print()
    if examples:
        print("disagreements (policy vs oracle):")
        print(f"  {'turn':>6} {'enemy':<12} {'policy':<14} {'score':>7}   "
              f"{'oracle':<14} {'score':>7}")
        for turn, esp, et, used, us, ora, os_ in examples:
            print(f"  {turn:>6} {esp:<12} {used:<14} {us:>7.1f}   "
                  f"{ora:<14} {os_:>7.1f}")
    print()
    if disagree_pairs:
        print("most common (policy -> oracle) on same species:")
        for (u, o, sp), n in disagree_pairs.most_common(8):
            print(f"  {n:>4}x  {u} -> {o}   (vs {sp})")

    # Verdict
    print()
    rate = disagree / max(total_moves, 1)
    if total_moves < 500:
        print(f"VERDICT: only {total_moves} move decisions - too few to train on.")
    if rate < 0.05:
        print("VERDICT: policy already matches the oracle - SFT would teach nothing.")
    elif rate < 0.25:
        print(f"VERDICT: {rate*100:.0f}% disagreement - real but modest signal.")
    else:
        print(f"VERDICT: {rate*100:.0f}% disagreement - strong signal, worth learning.")


if __name__ == "__main__":
    main()
