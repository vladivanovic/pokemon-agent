#!/usr/bin/env python3
"""Verify the generalized waypoint BFS finds the Route 4 gap descent.

If this passes, the per-map Route 4 special case is dead: the same
`_graph_destination -> first_hop_direction -> _map_edge_path` chain that
handles every other map also handles the winding one.
"""
import sys
sys.path.insert(0, '/home/vlad/pokemon-agent')

from pokemon_agent.von import _map_edge_path, _load_map_walk, _load_ledge_edges
from pokemon_agent.map_graph import MapGraph

mw = _load_map_walk()
ledges = _load_ledge_edges()
mg = MapGraph()

print("=== 1. map graph says which way Cerulean is ===")
hop = mg.first_hop_direction("Route 4", "Cerulean City")
print(f"  first_hop_direction('Route 4','Cerulean City') = {hop!r}")
assert hop == "east", f"expected east, got {hop}"

print()
print("=== 2. BFS the stitched map east from the UPPER half (19,12) ===")
r4 = mw.get("Route 4") or {}
print(f"  Route 4 stitched cells: {len(r4)}")
path = _map_edge_path(r4, (19, 12), "right", limit=400,
                      ledge_edges=ledges.get("Route 4", {}))
print(f"  path length: {len(path) if path else 0}")
if path:
    print(f"  first 12 actions: {path[:12]}")
    # The upper half must descend: a 'walk_down' must appear early.
    first_down = next((i for i, a in enumerate(path) if a == "walk_down"), None)
    print(f"  first walk_down at index: {first_down}")

print()
print("=== 3. same chain from the LOWER half (19,17) ===")
try:
    path2 = _map_edge_path(r4, (19, 17), "right", limit=400,
                           ledge_edges=ledges.get("Route 4", {}))
    print(f"  path length: {len(path2) if path2 else 0}")
    if path2:
        print(f"  first 12 actions: {path2[:12]}")
except Exception as exc:
    print(f"  (19,17) not in stitched map: {exc}")

print()
print("=== verdict ===")
if path and any(a == "walk_down" for a in path[:14]):
    print("  PASS - generalized BFS descends through the gap. No per-map code needed.")
else:
    print("  CHECK - BFS did not descend; inspect the stitched map seam.")
