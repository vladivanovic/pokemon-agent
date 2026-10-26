"""Produce the final rom_walk.json:

* OVERWORLD maps (Route 4, Cerulean City): ROM-native decode with the
  corrected passable semantics + LedgeTiles hops. Validated 80/80 against
  the live RAM window.
* Cave floors: ROM-native decode, tileset chosen by which one reproduces
  the cells the agent is proven to have walked.
Layouts are pret .blk (byte-identical to the ROM's own copies - verified).
"""
import collections
import json
import os
import sys
import urllib.request

sys.path.insert(0, '/home/vlad/pokemon-agent')
from pokemon_agent.collision import TILESET_WALKABLE  # noqa

rom = open('/home/vlad/pokemon-agent/pokeblue.gbc', 'rb').read()
BST_OW = rom[0x645E0:0x645E0 + 2048]
BST_CAV = rom[0x6D0C0:0x6D0C0 + 2048]

STAND = {0x2C, 0x39}
HOPS = {'walk_down': {0x2C: {0x37}, 0x39: {0x36, 0x37}},
        'walk_left': {0x2C: {0x27}, 0x39: {0x27}},
        'walk_right': {0x2C: {0x0D, 0x1D}, 0x39: {0x0D}}}

MAPS = {  # name: (pret stem, bw, bh)
    'Route 4': ('Route4', 45, 9),
    'Cerulean City': ('CeruleanCity', 20, 18),
    'Mt Moon 1F': ('MtMoon1F', 20, 18),
    'Mt Moon B1F': ('MtMoonB1F', 14, 14),
    'Mt Moon B2F': ('MtMoonB2F', 20, 18),
}

proven = collections.defaultdict(set)
for line in open('/home/vlad/.pokemon-agent/flight_recorder.jsonl',
                 errors='replace'):
    try:
        r = json.loads(line)
    except Exception:
        continue
    m, p = r.get('map'), r.get('pos')
    if m in MAPS and isinstance(p, list) and len(p) == 2 \
            and r.get('ts', '') >= '2026-10-25':
        proven[m].add((int(p[0]), int(p[1])))


def build(stem, bw, bh, bst, walked):
    path = f'/tmp/{stem}.blk'
    if not os.path.exists(path) or os.path.getsize(path) != bw * bh:
        data = urllib.request.urlopen(
            'https://raw.githubusercontent.com/pret/pokered/master/'
            f'maps/{stem}.blk', timeout=30).read()
        if len(data) != bw * bh:
            return None
        open(path, 'wb').write(data)
    blk = open(path, 'rb').read()
    TW, TH = bw * 2, bh * 2

    def sample(tx, ty):
        b = blk[(ty // 2) * bw + (tx // 2)] * 16 \
            + ((ty % 2) * 2 + 1) * 4 + (tx % 2) * 2
        return bst[b]

    def quad(tx, ty):
        b = blk[(ty // 2) * bw + (tx // 2)] * 16 \
            + ((ty % 2) * 2) * 4 + (tx % 2) * 2
        return (bst[b], bst[b + 1], bst[b + 4], bst[b + 5])

    grid = [[sample(x, y) in walked for x in range(TW)] for y in range(TH)]
    hops = {}
    for ty in range(TH):
        for tx in range(TW):
            q = quad(tx, ty)
            st = next((t for t in q if t in STAND), None)
            if st is None:
                continue
            for wn, dx, dy in (('walk_down', 0, 1), ('walk_left', -1, 0),
                               ('walk_right', 1, 0)):
                if st not in HOPS[wn]:
                    continue
                ax, ay = tx + dx, ty + dy
                if not (0 <= ax < TW and 0 <= ay < TH):
                    continue
                if any(t in HOPS[wn][st] for t in quad(ax, ay)):
                    lx, ly = tx + 2 * dx, ty + 2 * dy
                    if 0 <= lx < TW and 0 <= ly < TH:
                        hops[f'{tx},{ty},{wn}'] = f'{lx},{ly}'
    return grid, hops


out = {}
for name, (stem, bw, bh) in MAPS.items():
    is_ow = name in ('Route 4', 'Cerulean City')
    cands = [('ow' if is_ow else 'ts16', TILESET_WALKABLE[0 if is_ow else 16],
              BST_OW if is_ow else BST_CAV)]
    if not is_ow:
        cands.append(('ts17', TILESET_WALKABLE[17], BST_CAV))
    best = None
    for label, wset, bst in cands:
        res = build(stem, bw, bh, bst, wset)
        if res is None:
            continue
        grid, hops = res
        pw = proven.get(name, set())
        score = sum(1 for (x, y) in pw
                    if 0 <= y < len(grid) and 0 <= x < len(grid[0])
                    and grid[y][x])
        frac = score / max(len(pw), 1)
        print(f'{name} [{label}]: {sum(map(sum, grid))} walkable, '
              f'{len(hops)} hops, proven {score}/{len(pw)} ({frac:.0%})')
        if best is None or frac > best[0]:
            best = (frac, label, grid, hops, wset)
    if best is None:
        continue
    frac, label, grid, hops, wset = best
    out[name] = {'tileset': label, 'w': bw, 'h': bh,
                 'edges': {d: [row[:] for row in grid]
                           for d in ('north', 'south', 'west', 'east')},
                 'walk': grid, 'hops': hops, 'walk_model': 'rom-native'}
    print(f'  -> {name}: {label} (proven {frac:.0%})')

json.dump(out, open('/home/vlad/.pokemon-agent/rom_walk.json', 'w'))
print('wrote rom_walk.json:', list(out))
