#!/usr/bin/env python3
"""Replay a full-game TAS and LABEL every frame with the rules planner.

PHASE 2 of the full-game planner dataset pipeline:

    TAS input log -> PyBoy -> (screen, driver state) -> derive_goal label

The TAS is a complete Hall-of-Fame run of Pokemon Blue, so replaying it
yields planner training data for the WHOLE game (all 8 badges + Elite
Four) - which is exactly the coverage the original planner manifest
lacked (it ended at Pewter: 2,375 samples across Red's House 2F,
Viridian Pokecenter, Route 1 and Oak's Lab only).

ZERO manual labeling: the goal is a PURE function of RAM state
(`planner_rules.derive_goal`), the same rules the driver falls back to
at runtime. The state is read with the PROJECT'S OWN reader
(`memory/red.py`) so the labels line up with what the live driver sees,
not with a second, drifting RAM map.

Output (consumed directly by train_planner_lora.py):
    <out>/frames/frame_NNNNNN.jpg
    <out>/manifest.jsonl
        {"file": ..., "goal": ..., "state": {map, map_id, badges, x, y,
         facing, in_battle, dialog_open, party_hp, party_max_hp}}

Usage:
    python scripts/tas_replay_labeled.py \
        --rom "Pokemon - Blue Version (USA, Europe) (SGB Enhanced).gb" \
        --bk2 /tmp/tas_blue_input_log.txt \
        --out-dir dataset/planner_full \
        --sample-every 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from pyboy import PyBoy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokemon_agent.gen1_data import MAP_NAMES           # noqa: E402
from pokemon_agent.memory.red import (ADDR_MAP_ID,      # noqa: E402
                                     RedBlueMemoryReader)
from pokemon_agent.planner_rules import derive_goal     # noqa: E402
from tas_replay import (BTN_ORDER, PYBOY_PRESS, PYBOY_RELEASE,
                        parse_input_log)                # noqa: E402


class PyBoyAdapter:
    """Minimal emulator adapter so the project reader can drive PyBoy.

    The driver's reader only needs byte/range reads; PyBoy exposes them
    through ``pyboy.memory``. Keeping this adapter tiny means the reader
    (and therefore every label) stays identical to runtime.
    """

    def __init__(self, pyboy: PyBoy):
        self.pyboy = pyboy

    def read_u8(self, addr: int) -> int:
        return int(self.pyboy.memory[addr])

    def read_range(self, addr: int, length: int) -> bytes:
        return bytes(self.pyboy.memory[addr:addr + length])


# The reader reports facing as a NAME ('down'); the training manifest
# stores the Gen-1 facing byte (0=down, 4=up, 8=left, 12=right).
FACING_CODE = {"down": 0, "up": 4, "left": 8, "right": 12}


def read_state(reader: RedBlueMemoryReader) -> dict:
    """The compact planner state - the exact shape train_planner_lora
    consumes (`state` in the manifest)."""
    player = reader.read_player() or {}
    party = reader.read_party() or []
    battle = reader.read_battle() or {}
    dialog = reader.read_dialog() or {}

    # read_player does not expose the map id - read it directly.
    map_id = int(reader.emu.read_u8(ADDR_MAP_ID))
    pos = player.get("position") or player
    facing_name = player.get("facing")
    facing = (FACING_CODE.get(str(facing_name).lower(), 0)
              if isinstance(facing_name, str) else int(facing_name or 0))
    hp = [int(m.get("hp") or 0) for m in party]
    max_hp = [int(m.get("max_hp") or 0) for m in party]
    # Pad to 6 so the text serialization is stable across frames.
    hp += [0] * (6 - len(hp))
    max_hp += [0] * (6 - len(max_hp))

    return {
        "map": MAP_NAMES.get(map_id, f"Map {map_id}"),
        "map_id": map_id,
        "badges": len(player.get("badges") or []) or int(
            player.get("badge_count") or 0),
        "badge_names": list(player.get("badges") or []),
        "x": int(pos.get("x") or 0),
        "y": int(pos.get("y") or 0),
        "facing": facing,
        "in_battle": bool(battle.get("in_battle")),
        "battle_type": battle.get("type"),
        "dialog_open": bool(dialog.get("text_active")
                            or dialog.get("active")
                            or dialog.get("box_visible")),
        "party_hp": hp[:6],
        "party_max_hp": max_hp[:6],
    }


def read_bag(reader: RedBlueMemoryReader) -> dict:
    """Item-name -> total quantity, for the key-item route gates."""
    out: dict = {}
    for entry in (reader.read_bag() or []):
        name = str(entry.get("item") or "").upper()
        if name:
            out[name] = out.get(name, 0) + int(
                entry.get("quantity") or entry.get("qty") or 0)
    return out


def label(state: dict, bag: dict) -> str:
    """The rules planner IS the label source (pure state -> goal)."""
    return derive_goal(
        state["map"], state["badges"], state["in_battle"],
        state["dialog_open"], state["party_hp"], state["party_max_hp"],
        bag_items=bag, money=0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay a TAS and label frames with the rules planner")
    ap.add_argument("--rom", required=True,
                    help="ROM matching the TAS (Blue for tas_blue)")
    ap.add_argument("--bk2", required=True,
                    help="Extracted BizHawk 'Input Log.txt'")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-every", type=int, default=20,
                    help="Label every Nth frame (20 = ~16k samples over a "
                         "322k-frame TAS)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-frames", type=int, default=0)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--no-frames", action="store_true",
                    help="Manifest only (no JPEGs) - fast label-only pass")
    ap.add_argument("--frame-multiplier", type=int, default=1,
                    help="PyBoy frames to run per logged input frame. A TAS "
                         "recorded at a different frame base drifts; this is "
                         "the calibration knob (1 = 1:1)")
    args = ap.parse_args()

    rom_path = Path(args.rom).expanduser()
    log_path = Path(args.bk2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    if not args.no_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    if not rom_path.exists():
        sys.exit(f"ROM not found: {rom_path}")
    if not log_path.exists():
        sys.exit(f"Input log not found: {log_path}")

    print(f"Parsing {log_path} ...")
    frame_actions = parse_input_log(log_path)
    total = len(frame_actions)
    if args.max_frames:
        total = min(total, args.max_frames)
    print(f"Parsed {len(frame_actions)} frames; replaying {total}")

    pyboy = PyBoy(str(rom_path), window="null", sound=False)
    pyboy.set_emulation_speed(0)
    reader = RedBlueMemoryReader(PyBoyAdapter(pyboy))

    manifest = []
    saved = 0
    try:
        for frame_idx in range(total):
            pressed = frame_actions[frame_idx]
            for name in BTN_ORDER:
                if name == "power":
                    continue
                ev = (PYBOY_PRESS if name in pressed
                      else PYBOY_RELEASE).get(name)
                if ev:
                    pyboy.send_input(ev)
            for _ in range(max(1, args.frame_multiplier)):
                pyboy.tick()

            if frame_idx < args.skip_frames:
                continue
            if frame_idx % args.sample_every:
                continue

            try:
                st = read_state(reader)
                bag = read_bag(reader)
                goal = label(st, bag)
            except Exception as exc:          # never lose the whole pass
                print(f"  frame {frame_idx}: read/label failed: {exc}")
                continue

            img_name = f"frame_{frame_idx:06d}.jpg"
            if not args.no_frames:
                screen = pyboy.screen.ndarray[:, :, :3].copy()
                cv2.imwrite(str(frames_dir / img_name),
                            cv2.cvtColor(screen, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

            manifest.append({
                "file": img_name,
                "frame": frame_idx,
                "goal": goal,
                "buttons": pressed,
                "state": st,
            })
            saved += 1
            if saved % 500 == 0:
                print(f"  {saved} labelled frames (frame {frame_idx}) "
                      f"- last: {st['map']} b{st['badges']} | {goal[:60]}")
    except KeyboardInterrupt:
        print("\nInterrupted - writing what we have")
    finally:
        pyboy.stop()

    with (out_dir / "manifest.jsonl").open("w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    # Coverage report - the whole point of the pass.
    from collections import Counter
    maps = Counter(e["state"]["map"] for e in manifest)
    goals = Counter(e["goal"] for e in manifest)
    badges = Counter(e["state"]["badges"] for e in manifest)
    print(f"\nDone: {saved} labelled frames -> {out_dir/'manifest.jsonl'}")
    print(f"maps covered: {len(maps)}   distinct goals: {len(goals)}")
    print("badge histogram:", dict(sorted(badges.items())))
    print("top maps:")
    for name, n in maps.most_common(15):
        print(f"  {n:5d}  {name}")
    print("top goals:")
    for g, n in goals.most_common(12):
        print(f"  {n:5d}  {g[:78]}")


if __name__ == "__main__":
    main()
