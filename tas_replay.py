#!/usr/bin/env python3
"""Replay a TAS .bk2 movie in PyBoy and dump (frame, action) pairs.

Usage:
    python tas_replay.py \
        --rom pokeblue.gbc \
        --bk2 "Input Log.txt" \
        --out-dir dataset/tas_blue \
        --sample-every 4 \
        --max-frames 0

The .bk2 "Input Log.txt" is extracted from the BizHawk .bk2 archive.
Format per line: |.|..|...| (9 chars: Up,Down,Left,Right,Start,Select,B,A,Power)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pyboy import PyBoy


# Button order in the Input Log matches the LogKey header:
# #Up|Down|Left|Right|Start|Select|B|A|Power|
BTN_ORDER = ["up", "down", "left", "right", "start", "select", "b", "a", "power"]
# PyBoy uses WindowEvent press/release pairs
from pyboy.utils import WindowEvent  # noqa: E402
PYBOY_PRESS = {
    "up": WindowEvent.PRESS_ARROW_UP,
    "down": WindowEvent.PRESS_ARROW_DOWN,
    "left": WindowEvent.PRESS_ARROW_LEFT,
    "right": WindowEvent.PRESS_ARROW_RIGHT,
    "start": WindowEvent.PRESS_BUTTON_START,
    "select": WindowEvent.PRESS_BUTTON_SELECT,
    "b": WindowEvent.PRESS_BUTTON_B,
    "a": WindowEvent.PRESS_BUTTON_A,
}
PYBOY_RELEASE = {
    "up": WindowEvent.RELEASE_ARROW_UP,
    "down": WindowEvent.RELEASE_ARROW_DOWN,
    "left": WindowEvent.RELEASE_ARROW_LEFT,
    "right": WindowEvent.RELEASE_ARROW_RIGHT,
    "start": WindowEvent.RELEASE_BUTTON_START,
    "select": WindowEvent.RELEASE_BUTTON_SELECT,
    "b": WindowEvent.RELEASE_BUTTON_B,
    "a": WindowEvent.RELEASE_BUTTON_A,
}


@dataclass
class FrameSample:
    frame_idx: int
    image_path: str
    buttons: List[str]  # e.g. ["up", "a"]
    # optional metadata
    # map_name: str = ""
    # pos: Tuple[int, int] = (0, 0)


# --- RAM addresses (Pokemon Red/Blue; pret/pokered) ---------------------
# Capturing these alongside each frame turns the TAS replay into WHOLE-GAME
# planner training data: the TAS covers all 8 badges, while PokeDreamer only
# covers the first town. This closes the coverage gap that would otherwise
# make the planner useless past Viridian.
RAM_MAP_ID = 0xD35E
RAM_BADGES = 0xD356
RAM_X, RAM_Y = 0xD362, 0xD361
RAM_PARTY_SIZE = 0xD163
RAM_HP = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
RAM_MAX_HP = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]


def read_ram_state(pyboy) -> dict:
    """Read the compact planner state from emulator RAM. Best-effort."""
    def rd(addr):
        try:
            return int(pyboy.memory[addr])
        except Exception:
            return 0

    size = min(rd(RAM_PARTY_SIZE), 6)
    hp, mhp = [], []
    for i in range(6):
        if i < size:
            hp.append(int(pyboy.memory[RAM_HP[i]]) |
                      (int(pyboy.memory[RAM_HP[i] + 1]) << 8))
            mhp.append(int(pyboy.memory[RAM_MAX_HP[i]]) |
                       (int(pyboy.memory[RAM_MAX_HP[i] + 1]) << 8))
        else:
            hp.append(0)
            mhp.append(0)
    return {
        "map_id": rd(RAM_MAP_ID),
        "badges": rd(RAM_BADGES),
        "x": rd(RAM_X),
        "y": rd(RAM_Y),
        "party_hp": hp,
        "party_max_hp": mhp,
    }


def parse_input_log(log_path: Path) -> List[List[str]]:
    """Parse the BizHawk Input Log.txt into list of button lists per frame."""
    actions: List[List[str]] = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("[Input]") or line.startswith("LogKey"):
                continue
            # Format: |..L...|.  (9 chars between pipes)
            if line.startswith("|") and line.endswith("|"):
                cells = line[1:-1]
                if len(cells) != 9:
                    # Could be malformed; pad/truncate
                    cells = cells.ljust(9, ".")[:9]
            else:
                # Fallback: assume space/comma separated
                cells = line.replace(" ", "").split(",")
                if len(cells) != 9:
                    continue
            pressed = [BTN_ORDER[i] for i, c in enumerate(cells) if c != "."]
            actions.append(pressed)
    return actions


def main():
    ap = argparse.ArgumentParser(description="Replay TAS in PyBoy and dump frames")
    ap.add_argument("--rom", required=True, help="Path to .gbc ROM (must match TAS hash)")
    ap.add_argument("--bk2", required=True, help="Path to extracted Input Log.txt from .bk2")
    ap.add_argument("--out-dir", required=True, help="Output directory for frames + manifest")
    ap.add_argument("--sample-every", type=int, default=4, help="Save every N frames (default 4)")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    ap.add_argument("--skip-frames", type=int, default=0, help="Skip first N frames (intro/cutscene)")
    ap.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality 1-100")
    ap.add_argument("--window", choices=["null", "SDL2", "headless"], default="null",
                    help="PyBoy window type (null = headless)")
    args = ap.parse_args()

    rom_path = Path(args.rom)
    log_path = Path(args.bk2)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not rom_path.exists():
        sys.exit(f"ROM not found: {rom_path}")
    if not log_path.exists():
        sys.exit(f"Input log not found: {log_path}")

    # Parse button log
    print(f"Parsing {log_path}...")
    frame_actions = parse_input_log(log_path)
    total_log_frames = len(frame_actions)
    print(f"Parsed {total_log_frames} frames of button inputs")

    if args.max_frames and args.max_frames < total_log_frames:
        total_log_frames = args.max_frames

    # Initialize PyBoy
    window_type = {"null": "null", "SDL2": "SDL2", "headless": "headless"}[args.window]
    pyboy = PyBoy(
        str(rom_path),
        window=window_type,
        sound=False,
    )
    pyboy.set_emulation_speed(0)  # max speed

    manifest: List[dict] = []
    saved = 0
    frame_idx = 0

    try:
        while frame_idx < total_log_frames:
            # PyBoy 2.x null window: tick() return value is unreliable (True on
            # first tick) - ignore it and run for the logged frame count.
            pressed_names = frame_actions[frame_idx]

            # Apply buttons: press the ones in this frame's list, release the rest
            for name in BTN_ORDER:
                if name == "power":
                    continue
                if name in pressed_names:
                    ev = PYBOY_PRESS.get(name)
                else:
                    ev = PYBOY_RELEASE.get(name)
                if ev:
                    pyboy.send_input(ev)

            # Tick emulator (advances one frame)
            pyboy.tick()

            # After tick, the frame is rendered. Capture if sampling.
            if frame_idx >= args.skip_frames and (frame_idx % args.sample_every) == 0:
                # Get screen as numpy array (144x160x3 RGB)
                screen = pyboy.screen.ndarray[:, :, :3].copy()  # drop alpha
                # PyBoy returns RGB; OpenCV expects BGR for imwrite
                screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

                img_name = f"frame_{frame_idx:06d}.jpg"
                img_path = frames_dir / img_name
                cv2.imwrite(str(img_path), screen_bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

                manifest.append({
                    "frame": frame_idx,
                    "file": img_name,
                    "buttons": pressed_names,
                    "state": read_ram_state(pyboy),
                })
                saved += 1

                if saved % 1000 == 0:
                    print(f"  saved {saved} frames...")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        pyboy.stop()

    # Write manifest
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone. Saved {saved} frames to {frames_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Total log frames: {len(frame_actions)}, processed: {frame_idx}")


if __name__ == "__main__":
    main()