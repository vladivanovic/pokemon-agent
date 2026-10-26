"""Purge fold cells that the ROM decode calls WALL but the fold invented.

Applies the cell-authority rule:
    a fold cell stays open iff static_open OR the player physically
    walked it.

Two safety properties, both learned the hard way:

* ABORT if the log parse yields no walked cells at all. It once yielded
  none (the log writes ``pos=(x,y)`` with no space, and the regex demanded
  one) and the purge then deleted 44,959 cells across 217 maps.
* Only judge maps with a RELIABLE walked set. The fold accumulated over
  months of runs whose logs use a different position format; for those
  maps every cell would look invented. Pass map names to target them, or
  omit to scan every map that has walked data.

Dry run by default; pass --apply to write (a backup is taken first).

    python3 scripts/purge_false_opens.py                 # dry run, all known
    python3 scripts/purge_false_opens.py "Pewter City"   # dry run, one map
    python3 scripts/purge_false_opens.py --apply         # write
"""
import collections
import glob
import json
import re
import shutil
import sys

DATA = '/home/vlad/.pokemon-agent'
FOLD = f'{DATA}/map_walk.json'
STATIC = f'{DATA}/static_walk.json'
LOGS = glob.glob('/home/vlad/pokemon-agent/v2/runs/*.log')

static = json.load(open(STATIC))
static_open = {m: {(x, y) for x, y, w in cells if w}
               for m, cells in static.items()}

# --- walked cells per map, from every run log -------------------------------
walked = collections.defaultdict(set)
for path in LOGS:
    cur = None
    try:
        for line in open(path, encoding='utf-8', errors='ignore'):
            m = re.search(r'map changed: (.+?) -> (.+?)\s*$', line)
            if m:
                cur = m.group(2).strip()
                continue
            m2 = re.search(r'pos=\((\d+),\s*(\d+)\)', line)
            if m2 and cur:
                walked[cur].add((int(m2.group(1)), int(m2.group(2))))
    except OSError:
        continue
print('walked cells known for', len(walked), 'maps')
if not walked:
    print('ABORT: no walked cells parsed - refusing to purge. '
          'Fix the log parse first.')
    sys.exit(1)

# --- purge ------------------------------------------------------------------
fold = json.load(open(FOLD))
targets = [a for a in sys.argv[1:] if not a.startswith('-')]
apply = '--apply' in sys.argv
if not targets:
    targets = sorted(walked)
print(f'judging {len(targets)} map(s) with reliable walked data'
      f'{" [APPLY]" if apply else " [DRY RUN - pass --apply to write]"}')

total_removed = 0
per_map = []
for m in targets:
    cells = fold.get(m)
    if not isinstance(cells, list):
        continue
    so = static_open.get(m)
    if so is None:
        print(f'  skip {m}: no static data')
        continue
    w = walked.get(m, set())
    keep, removed_cells = [], []
    for c in cells:
        if not (isinstance(c, list) and len(c) >= 3):
            keep.append(c)
            continue
        x, y, val = c[0], c[1], c[2]
        if val and (x, y) not in so and (x, y) not in w:
            removed_cells.append((x, y))
            continue
        keep.append(c)
    if removed_cells:
        per_map.append((len(removed_cells), m, removed_cells))
        total_removed += len(removed_cells)
        fold[m] = keep

for n, m, cells in sorted(per_map, reverse=True):
    print(f'  {n:5d}  {m}   e.g. {sorted(cells)[:8]}')
print(f'\n{"removed" if apply else "WOULD remove"} {total_removed} '
      f'invented-open cells across {len(per_map)} maps')
if apply and total_removed:
    shutil.copy(FOLD, FOLD + '.pre_cell_authority')
    json.dump(fold, open(FOLD, 'w'))
    print(f'written. backup: {FOLD}.pre_cell_authority')
