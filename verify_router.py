#!/usr/bin/env python3
"""Exercise the new 3-tier router against the real stitched map + graph."""
import sys
sys.path.insert(0, '/home/vlad/pokemon-agent')

from pokemon_agent.von import VonDriver

d = VonDriver("http://localhost:8765", planner="rules", scorer="von")
print("driver constructed OK; scorer =", d.scorer_name,
      "| planner =", d.planner_mode)
print()

cases = [
    # (cur_map, pos, goal, what we expect to see)
    ("Route 4", {"x": 19, "y": 12},
     "Head EAST along Route 4 toward Cerulean City.",
     "winding map -> tier 3 frontier discovery"),
    ("Route 4", {"x": 4, "y": 12},
     "Head EAST along Route 4 toward Cerulean City.",
     "monotonic -> tier 2 compass BFS"),
    ("Route 3", {"x": 40, "y": 6},
     "Head EAST along Route 3 - cross the ledges and reach Route 4.",
     "tier 2 compass BFS east"),
    ("Route 4", {"x": 14, "y": 6},
     "Exit the Pokemon Center to the map outside.",
     "tier 1 learned portal back to Mt Moon Pokecenter"),
    ("Pewter Museum 1F", {"x": 5, "y": 5},
     "Exit this building to the map outside - walk to the exit door.",
     "local objective: exit_building"),
    ("Pewter Gym", {"x": 5, "y": 12},
     "Beat the Gym Leader - attack with the most effective move.",
     "local objective: challenge_leader"),
]

for cur_map, pos, goal, expect in cases:
    local = d._local_action(None, cur_map, pos, goal)
    if local:
        print(f"{cur_map:20s} pos={tuple(pos.values())}")
        print(f"  LOCAL   -> {list(local)}   ({expect})")
        continue
    route = d._resolve_route(None, cur_map, pos, goal)
    if route:
        key = list(route)[0]
        n = len(d._waypoint_path)
        print(f"{cur_map:20s} pos=({pos['x']},{pos['y']})")
        print(f"  ROUTE   -> {key}  path={n} steps  ({expect})")
        print(f"            first 8: {d._waypoint_path}")
    else:
        print(f"{cur_map:20s} pos=({pos['x']},{pos['y']})")
        print(f"  NONE    -> falls through to exploration  ({expect})")
    print()
