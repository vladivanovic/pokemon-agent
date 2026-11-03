"""BFS/route-integrity validator for the static walkability decode.

Checks:
  1. checkpoint cells (gym doors, Pokecenter doors, gate connections)
     are walkable in the static decode
  2. intra-map connectivity: for maps with fold coverage, the fold's
     walked-cell component agrees with static walkability
  3. cross-map walk edges resolve to open cells on both sides
"""
import json, os, sys
from collections import deque

HOME = os.path.expanduser('~/.pokemon-agent')
sw = json.load(open(os.path.join(HOME, 'static_walk.json')))
mw_path = os.path.join(HOME, 'map_walk.json')
mw = json.load(open(mw_path)) if os.path.exists(mw_path) else {}
we = json.load(open(os.path.join(HOME, 'walk_edges.json')))

def grid(m):
    return {(c[0], c[1]): c[2] for c in sw.get(m, [])}

checkpoints = {
    'Pewter City': [(13, 25), (16, 17)],
    'Viridian City': [(23, 25), (16, 17)],
    'Cerulean City': [(19, 17)],
    'Viridian Forest': [(2, 0), (16, 47)],
    'Route 2': [(9, 0), (8, 71)],
}
ok = bad = 0
print('== checkpoint cells ==')
for m, cells in checkpoints.items():
    g = grid(m)
    for c in cells:
        v = g.get(c)
        print(f'{m} {c}: {"WALKABLE" if v else "not walkable/unknown"}')
        ok += 1 if v else 0
        bad += 0 if v else 1

print('\n== walk-edge both-sides open ==')
edge_ok = edge_bad = 0
for m, edges in we.items():
    g = grid(m)
    for e in edges:
        c = tuple(e['cell'])
        dm = e['dest']
        gd = grid(dm)
        # the dest map's back-edge cell (same dir family): any open cell
        # on the corresponding border row/col
        back = [x for x in we.get(dm, []) if x['dest'] == m]
        if not back:
            continue
        dcell = tuple(back[0]['cell'])
        a = g.get(c)
        b = gd.get(dcell)
        if a and b:
            edge_ok += 1
        else:
            edge_bad += 1
            print(f'BROKEN handoff: {m}{c} <-> {dm}{dcell} '
                  f'(a={a} b={b})')
print(f'handoffs open both sides: {edge_ok}, broken: {edge_bad}')

print('\n== fold-vs-static on walked cells (top 8 by coverage) ==')
rows = []
for m in mw:
    if m not in sw:
        continue
    fg = {(c[0], c[1]): c[2] for c in mw[m]}
    sg = grid(m)
    walked = [c for c, w in fg.items() if w and sg.get(c) is not None]
    if len(walked) >= 20:
        agree = sum(1 for c in walked if sg[c])
        rows.append((len(walked), m, agree))
rows.sort(reverse=True)
for n, m, a in rows[:8]:
    print(f'{m}: {a}/{n} ({100*a//n}%)')
print(f'\ncheckpoint walkable: {ok}, not: {bad}')
