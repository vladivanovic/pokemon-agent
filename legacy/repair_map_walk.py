#!/usr/bin/env python3
"""Repair Pewter City stitched map connectivity to gym door.

The stitched map has a gap around the gym entrance (16,17) - the door
tile is unwalkable and its approach cells (16,18) etc. were never
properly stamped because the collision reader marks door tiles as
blocked. This fills the gap by ensuring the approach tile (16,18) is
walkable and connected to the city center.
"""
import json
from pathlib import Path
from collections import deque

MAP_WALK = Path("/home/vlad/.pokemon-agent/map_walk.json")

with MAP_WALK.open() as f:
    data = json.load(f)

city = data.get("Pewter City", [])
# Convert list of [x,y,walkable] to dict for easier manipulation
walkable = {}
for cell in city:
    x, y, w = cell[0], cell[1], cell[2]
    walkable[(x, y)] = bool(w)

# Target: gym door at (16,17) is unwalkable (door tile).
# Approach tile is (16,18) - directly below door - which IS walkable
# but was disconnected from city center.
# Ensure the corridor from city center (x~23) to gym door (16,17)
# is fully walkable at y=18 (the row just below the door row).

print("Repairing Pewter City connectivity...")
# Fill the corridor at y=18 from x=16 to x=23
for x in range(16, 24):
    walkable[(x, 18)] = True

# Also ensure the approach to gym door (16,17) from below is walkable
# (16,18) is already walkable per earlier check, but ensure neighbors connect
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        if dx == 0 and dy == 0:
            continue
        walkable[(16 + dx, 18 + dy)] = True

# Also ensure the gym door tile's orthogonal neighbors are walkable
# (door itself at 16,17 is unwalkable - that's correct)
for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
    nx, ny = 16 + dx, 17 + dy
    if dx == 0 and dy == -1:  # above door - often wall
        continue
    walkable[(nx, ny)] = True

# Verify connectivity from city center (23,19) to gym approach (16,18)
from collections import deque
start = (23, 19)
goal = (16, 18)
DIRS = [(1,0), (-1,0), (0,1), (0,-1)]
seen = {start}
q = deque([start])
while q:
    x, y = q.popleft()
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x + dx, y + dy
        if walkable.get((nx, ny)) and (nx, ny) not in seen:
            seen.add((nx, ny))
            q.append((nx, ny))

connected = goal in seen
print(f"City center (23,19) -> gym approach (16,18): {'CONNECTED' if connected else 'STILL DISCONNECTED'}")
print(f"Reachable cells from city center: {len(seen)}")

if connected:
    # Convert back to list format
    new_cells = [[x, y, walkable[(x, y)]] for (x, y) in walkable]
    data["Pewter City"] = new_cells
    with MAP_WALK.open("w") as f:
        json.dump(data, f, indent=1)
    print("map_walk.json UPDATED")
else:
    print("Still disconnected - manual intervention needed")