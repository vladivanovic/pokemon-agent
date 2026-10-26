"""Purge CONFIRMED_BLOCKS entries that the position log disproves.

Inverse of the cell-authority purge (false OPENS): here we remove false
BLOCKS. A cell the player has ever physically occupied cannot be a
physical wall - the zero-displacement watchdog blacklists cells the
batch tried to ENTER, and during the interleaved-driver incident the
'dispatch' it judged belonged to a DIFFERENT driver's turn, so the
positions never matched. 29 cells got blacklisted that way, including
(5,5) - a cell the player stood on moments later.

Authority: position-log occupancy beats watchdog inference.
Usage: python3 purge_false_blocks.py [--apply]
Dry-run by default; takes a .pre_block_authority backup on --apply.
"""
import json
import re
import shutil
import sys
from collections import defaultdict

CB_PATH = "/home/vlad/.pokemon-agent/confirmed_blocks.json"
LOGS = ["/home/vlad/pokemon-agent/v2/runs/von_vlm.log"]

# (map, x, y) -> first turn-seen; occupancy = the strongest evidence
occupied = set()
# map name + position. Two live formats coexist: 'pos=(4, 5)' (watchdog
# lines) and 'pos=(7,20)' (turn lines) - allow optional space.
pos_re = re.compile(r"map=(\S+(?: \S+)?) pos=\((\d+),\s*(\d+)\)")
for path in LOGS:
    try:
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                pm = pos_re.search(line)
                if pm:
                    occupied.add((pm.group(1), int(pm.group(2)),
                                  int(pm.group(3))))
    except OSError:
        continue

print(f"occupied cells seen in logs: {len(occupied)}")

with open(CB_PATH) as fh:
    cb = json.load(fh)

removed, kept = [], []
for m, blocks in list(cb.items()):
    out = []
    for b in blocks:
        key = (m, b[0], b[1])
        if key in occupied:
            removed.append(key)
        else:
            out.append(b)
    if out:
        cb[m] = out
    else:
        del cb[m]
        kept.append(m)

print(f"\nremoving {len(removed)} occupied-cell blocks (walked = not a wall):")
for r in sorted(removed):
    print(f"  - {r}")
remaining = sum(len(v) for v in cb.values())
print(f"\nremaining blocks: {remaining}")
for m, blocks in sorted(cb.items()):
    print(f"  {m}: {sorted(blocks)}")

if "--apply" in sys.argv:
    shutil.copy(CB_PATH, CB_PATH + ".pre_block_authority")
    with open(CB_PATH, "w") as fh:
        json.dump(cb, fh, indent=1)
    print("\nAPPLIED (backup at confirmed_blocks.json.pre_block_authority)")
else:
    print("\nDRY RUN - pass --apply to write")
