#!/usr/bin/env python3
"""Convert PokeDreamer NPZ trajectories into VLM planner training pairs.

NO MANUAL LABELING. The high-level goal for each frame is DERIVED from the
RAM state that PokeDreamer already stores alongside every observation
(map_id, badges, x/y, party HP). This is the same logic Von's `_infer_goal`
uses, generalized into a pure function of state.

Output layout (compatible with train_planner_lora.py):
    dataset/planner/
        frames/frame_000000.jpg ...
        manifest.jsonl  {"file": "...", "goal": "...", "state": {...}}

Usage:
    python convert_pokedreamer.py \
        --src ~/pokemon-data/pokedreamer \
        --out dataset/planner \
        --stride 10        # every 10th frame (diversity, not redundancy)
        --max 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# PokeDreamer map_id -> map name. IDs come from Pokemon Red's map header
# constants (see pokered/maps/map_headers.asm). Only the early-game maps the
# dataset actually contains are listed; unknown ids fall back to "Map<id>".
MAP_ID_TO_NAME = {
    0: "Pallet Town",
    1: "Viridian City",
    2: "Pewter City",
    3: "Cerulean City",
    12: "Route 1",
    13: "Route 2",
    14: "Viridian Forest",
    37: "Red's House 1F",
    38: "Red's House 2F",
    39: "Blue's House",
    40: "Oak's Lab",
    41: "Viridian Pokecenter",
    42: "Viridian Mart",
    51: "Pewter Pokecenter",
    52: "Pewter Mart",
    54: "Pewter Gym",
}

# Action id -> button name (PokeDreamer's 8-button action space).
ACTION_NAMES = ["up", "down", "left", "right", "start", "select", "b", "a"]


def derive_goal(map_name: str, badges: int, in_battle: bool,
                dialog_open: bool, party_hp: list, party_max: list) -> str:
    """High-level objective for the current state - the VLM's training target.

    This is deliberately a pure function of RAM state, so every frame in the
    dataset gets a goal for free. The vocabulary mirrors Von's `_infer_goal`
    so a VLM trained here can be dropped into the same driver.
    """
    if in_battle:
        return "Win the battle - pick the most effective move and attack."
    if dialog_open:
        return "Read the dialog and advance it with A (or B to skip)."

    party = [h for h, m in zip(party_hp, party_max) if m > 0]
    hurt = any(0 < h < m * 0.3 for h, m in zip(party_hp, party_max) if m > 0)
    lead_lvl_unknown = not party

    # --- badge 0: the long tutorial arc ---
    if badges == 0:
        if "Red's House" in map_name:
            return "Exit the house to Pallet Town - walk to the south door."
        if "Oak's Lab" in map_name:
            return ("Take a starter Pokemon from the table in Oak's Lab - "
                    "stand facing a Pokeball, press A, confirm YES.")
        if "Mart" in map_name:
            return ("Buy supplies at the counter (press A, choose BUY). "
                    "Get Poke Balls; Pewter Mart also sells Potions.")
        if "Gym" in map_name:
            return ("Beat the Gym Leader - attack with the most effective "
                    "move; heal when HP drops below a third.")
        if map_name in ("Route 1", "Route 2", "Viridian Forest"):
            return (f"Travel north toward Pewter City - keep heading north "
                    f"along the path, fight or avoid trainers as needed.")
        if hurt:
            return "Heal the party at the nearest Pokemon Center."
        if "Viridian" in map_name:
            return ("Collect Oak's Parcel from Viridian Mart, then return "
                    "to Oak's Lab in Pallet Town.")
        if "Pallet" in map_name and lead_lvl_unknown:
            return "Walk north from Pallet Town to Viridian City."
        return "Progress toward Pewter City and the first Gym."

    # --- badge >= 1: Route 3 / Mt Moon / Cerulean arc ---
    if "Gym" in map_name:
        return ("Exit the Gym to the city - walk to the south door and "
                "step onto it.")
    if "Pewter" in map_name:
        return ("Leave Pewter City via its EAST exit to Route 3.")
    if "Route 3" in map_name:
        return ("Head EAST along Route 3 - cross the ledges and reach "
                "Route 4 at the far east end.")
    if "Pokecenter" in map_name:
        return ("Exit the Pokemon Center to the map outside - walk to the "
                "exit door and step onto it.")
    if "Mart" in map_name:
        return "Exit the Mart to the map outside."
    if "Mt Moon" in map_name or "Mt. Moon" in map_name:
        return ("Traverse Mt. Moon from the south entrance to the north "
                "exit, then continue to Cerulean City.")
    if "Route 4" in map_name:
        return "Head EAST along Route 4 toward Cerulean City."
    if hurt:
        return "Heal the party at the nearest Pokemon Center."
    return "Continue toward Cerulean City for badge #2."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "pokemon-data/pokedreamer"))
    ap.add_argument("--out", default="dataset/planner")
    ap.add_argument("--stride", type=int, default=10,
                    help="keep every Nth frame (dedupe near-identical frames)")
    ap.add_argument("--max", type=int, default=0,
                    help="cap total samples (0 = no cap)")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--max-per-goal", type=int, default=1200,
                    help="cap frames per goal class (balances the dataset - "
                         "PokeDreamer is 80%%+ dialog frames, which would drown "
                         "out every navigation goal)")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("transitions_*.npz"))
    if not files:
        raise SystemExit(f"no transitions_*.npz under {src}")

    manifest = []
    idx = 0
    # Per-goal-class cap so a dominant class (dialog is ~80% of PokeDreamer)
    # can't drown the navigation goals. Without this the planner learns to
    # answer "advance the dialog" for every screen, which is worse than useless
    # - it is a confident wrong answer at every decision point.
    goal_counts: dict[str, int] = {}
    for f in files:
        d = np.load(f)
        # Force-load arrays ONCE per chunk. A compressed .npz decompresses on
        # every __getitem__, so indexing d["obs"][i] 10k times is ~100x slower
        # than pulling the whole array out first.
        obs_all = d["obs"]
        n = len(obs_all)
        map_ids = d["map_ids"]; badges_arr = d["badges"]
        in_battles = d["in_battles"]; dialogs = d["dialog_opens"]
        xs = d["xs"]; ys = d["ys"]; facings = d["facings"]
        php_all = d["party_hps"]; pmax_all = d["party_max_hps"]
        act_all = d["actions"]
        for i in range(0, n, args.stride):
            mid = int(map_ids[i])
            map_name = MAP_ID_TO_NAME.get(mid, f"Map{mid}")
            badges = int(badges_arr[i])
            in_battle = bool(in_battles[i])
            dialog = bool(dialogs[i])
            php = php_all[i].tolist()
            pmax = pmax_all[i].tolist()
            facing = int(facings[i])

            goal = derive_goal(map_name, badges, in_battle, dialog, php, pmax)
            if args.max_per_goal and goal_counts.get(goal, 0) >= args.max_per_goal:
                continue                       # class is full - skip

            obs = obs_all[i]                   # (144,160,3) uint8 RGB
            action = ACTION_NAMES[int(act_all[i])] if int(act_all[i]) < 8 else "none"
            fname = f"frame_{idx:06d}.jpg"
            Image.fromarray(obs).save(out / "frames" / fname,
                                      quality=args.jpeg_quality)
            goal_counts[goal] = goal_counts.get(goal, 0) + 1
            manifest.append({
                "file": fname,
                "goal": goal,
                "action": action,        # next low-level button (auxiliary signal)
                "state": {
                    "map": map_name,
                    "map_id": mid,
                    "badges": badges,
                    "x": int(xs[i]),
                    "y": int(ys[i]),
                    "facing": facing,
                    "in_battle": in_battle,
                    "dialog_open": dialog,
                    "party_hp": php,
                    "party_max_hp": pmax,
                },
            })
            idx += 1
            if idx % 500 == 0:
                print(f"  ...{idx} frames", flush=True)
            if args.max and idx >= args.max:
                break
        if args.max and idx >= args.max:
            break

    with (out / "manifest.jsonl").open("w") as mf:
        for row in manifest:
            mf.write(json.dumps(row) + "\n")

    # Report goal distribution so we can see coverage at a glance.
    from collections import Counter
    dist = Counter(r["goal"][:40] for r in manifest)
    print(f"Wrote {len(manifest)} samples to {out}")
    print("Top goals:")
    for g, c in dist.most_common(15):
        print(f"  {c:6d}  {g}")


if __name__ == "__main__":
    main()