"""Calibrate static_walk.json against the player's real position log.

Reads von_clean.log, tracks the current map from 'map changed' lines,
collects every distinct position the player actually occupied, and
scores them against static_walk.json.

A position disagreeing with the decode is either:
  - a transition-frame artifact (the map changed mid-read; the position
    belongs to the next map but was tagged with the previous one), or
  - a genuine decode error (should be ~zero after the ROM-native
    decoder: bank math + the reader-sampled-tile cell rule).

Run: python3 scripts/calibrate_static.py [log_path]
"""
import json, re, sys, os

STATIC = os.path.expanduser('~/.pokemon-agent/static_walk.json')
LOG = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/vlad/pokemon-agent/v2/runs/von_clean.log'

sw = json.load(open(STATIC))
cur_map = None
pos_by_map = {}
for line in open(LOG, errors='replace'):
    m = re.search(r'map changed: (.+?) -> (.+?)\s*$', line)
    if m:
        cur_map = m.group(2)
        if cur_map == 'None':
            cur_map = None
        continue
    m = re.search(r'pos=\((\d+), ?(\d+)\)', line)
    if m and cur_map:
        pos_by_map.setdefault(cur_map, set()).add((int(m.group(1)),
                                                   int(m.group(2))))

total = agree = 0
unknown_maps = {}
for m, positions in sorted(pos_by_map.items()):
    if m not in sw:
        unknown_maps[m] = len(positions)
        continue
    g = {(x, y): w for x, y, w in sw[m]}
    a = sum(1 for c in positions if g.get(c))
    blocked = [c for c in positions if g.get(c) is False]
    unknown = [c for c in positions if g.get(c) is None]
    total += len(positions)
    agree += a
    pct = 100 * a // max(1, len(positions))
    flag = '' if not blocked else f'  BLOCKED-WALKED: {blocked[:4]}'
    print(f'{m}: {len(positions)} real, {pct}% agree{flag}')
    if unknown:
        print(f'   unknown cells (outside decoded grid): {unknown[:4]}')
print(f'\nTOTAL: {agree}/{total} = {100*agree/max(1,total):.2f}%')
if unknown_maps:
    print('maps without static coverage:', unknown_maps)
