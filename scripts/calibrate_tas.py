#!/usr/bin/env python3
"""Calibrate a TAS input log against a ROM: which ROM/offset tracks it?

A TAS is frame-perfect for ONE cartridge revision. Replaying it against a
different ROM (or a different boot length) drifts: the player wanders,
re-enters buildings, and never follows the scripted route - exactly what
the Blue+SGB replay produced (stuck in Pallet/Route 1 over 322k frames).

This probes a grid of (rom, offset) candidates and reports how FAR each
gets, measured by a monotone progress proxy:
    score = badges*1000 + distinct_maps_seen + max_map_index_proxy
plus the frame at which the player first obtains a starter (party HP
becomes non-zero) - a TAS does that within ~12k frames; ours took 107k.
The best candidate is the one with the highest score at the same frame
budget. Cheap: no frame dumps, sample every 200 frames.

Usage:
    python scripts/calibrate_tas.py --bk2 /tmp/tas_blue_input_log.txt \
        --out /tmp/tas_calib --frames 30000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyboy import PyBoy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokemon_agent.gen1_data import MAP_NAMES           # noqa: E402
from pokemon_agent.memory.red import (ADDR_MAP_ID,      # noqa: E402
                                     RedBlueMemoryReader)
from tas_replay import (BTN_ORDER, PYBOY_PRESS, PYBOY_RELEASE,
                        parse_input_log)                # noqa: E402
from scripts.tas_replay_labeled import PyBoyAdapter     # noqa: E402


def run(rom: str, log, frames: int, offset: int, multiplier: int) -> dict:
    pyboy = PyBoy(rom, window="null", sound=False)
    pyboy.set_emulation_speed(0)
    reader = RedBlueMemoryReader(PyBoyAdapter(pyboy))
    maps, badges, starter_at = set(), 0, None
    last = {}
    try:
        for f in range(frames):
            li = f + offset
            pressed = log[li] if 0 <= li < len(log) else []
            for name in BTN_ORDER:
                if name == "power":
                    continue
                ev = (PYBOY_PRESS if name in pressed
                      else PYBOY_RELEASE).get(name)
                if ev:
                    pyboy.send_input(ev)
            for _ in range(max(1, multiplier)):
                pyboy.tick()
            if f % 200:
                continue
            mid = int(reader.emu.read_u8(ADDR_MAP_ID))
            maps.add(MAP_NAMES.get(mid, f"Map {mid}"))
            pl = reader.read_player() or {}
            n_badges = len(pl.get("badges") or [])
            badges = max(badges, n_badges)
            pos = pl.get("position") or {}
            last = {"frame": f, "map": MAP_NAMES.get(mid, mid),
                    "pos": (pos.get("x"), pos.get("y")), "badges": n_badges}
            if starter_at is None:
                party = reader.read_party() or []
                if party and int(party[0].get("max_hp") or 0) > 0:
                    starter_at = f
    finally:
        pyboy.stop()
    last.update({"maps_seen": len(maps), "maps": sorted(maps),
                 "starter_at": starter_at, "max_badges": badges,
                 "score": badges * 1000 + len(maps)})
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bk2", required=True)
    ap.add_argument("--out", default="/tmp/tas_calib")
    ap.add_argument("--frames", type=int, default=30000)
    args = ap.parse_args()

    log = parse_input_log(Path(args.bk2))
    gb = Path("/home/vlad/Downloads/GB")
    roms = sorted(gb.glob("Pokemon - *Version*.gb")) or \
        sorted(gb.glob("Pokemon*.gb"))
    print("candidate ROMs:", [r.name for r in roms])
    results = []
    for rom in roms:
        for offset in (0, 230, 460, -230):
            for mult in (1,):
                tag = f"{rom.name[:28]} off={offset} mult={mult}"
                r = run(str(rom), log, args.frames, offset, mult)
                r["tag"] = tag
                results.append(r)
                print(f"{tag:60s} score={r['score']:5d} "
                      f"badges={r['max_badges']} maps={r['maps_seen']:2d} "
                      f"starter@{r['starter_at']} last={r['map']} "
                      f"{r['pos']}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(results, open(Path(args.out) / "results.json", "w"), indent=1)
    best = max(results, key=lambda r: r["score"])
    print(f"\nBEST: {best['tag']} score={best['score']} "
          f"maps={best['maps'][:12]}")


if __name__ == "__main__":
    main()
