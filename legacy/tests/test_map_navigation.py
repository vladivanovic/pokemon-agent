#!/usr/bin/env python3
"""Tests for the stitched per-map navigation (fix for winding maps).

The bug these cover: routing used a single 10x9 screen + a compass rule. In
Viridian Forest the route winds, so "walk north" hits a tree, the map never
changes, and after 4 tries the direction was blacklisted permanently - which
then made the heal route to Pewter impossible and soft-locked the agent at
0 PP.

Run:  python3 tests/test_map_navigation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokemon_agent.autopilot import (  # noqa: E402
    _map_edge_path, _map_step_toward, _stamp_screen,
    PLAYER_COL, PLAYER_ROW)

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


def step(xy, action):
    x, y = xy
    return {"walk_up": (x, y - 1), "walk_down": (x, y + 1),
            "walk_left": (x - 1, y), "walk_right": (x + 1, y)}[action]


def walk_to(start, path):
    pos = start
    for a in path:
        pos = step(pos, a)
    return pos


def winding_maze():
    """15x12. North is blocked from the start; the route switchbacks."""
    walk = {(x, y): False for x in range(15) for y in range(12)}
    for x in range(3, 12):
        walk[(x, 10)] = True        # bottom corridor
    for x in range(1, 12):
        walk[(x, 6)] = True         # middle corridor
    for x in range(1, 8):
        walk[(x, 2)] = True         # top corridor
    for y in range(6, 11):
        walk[(11, y)] = True        # right-hand riser
    for y in range(2, 7):
        walk[(1, y)] = True         # left-hand riser
    return walk


def test_winding_edge():
    print("winding map, 'go north' must follow the switchbacks")
    w = winding_maze()
    start = (5, 10)
    path = _map_edge_path(w, start, "up")
    end = walk_to(start, path)
    check("path found", bool(path), f"{len(path)} steps")
    check("reaches the far north (y<=3)", end[1] <= 3, f"ended {end}")
    check("never steps off the walkable set",
          all(w.get(walk_to(start, path[:i + 1])) for i in range(len(path))))


def test_direct_target():
    print("targeted BFS reaches an off-screen goal")
    w = winding_maze()
    start = (5, 10)
    target = (5, 2)
    path = _map_step_toward(w, start, target)
    check("path found", bool(path), f"{len(path)} steps")
    check("arrives exactly", walk_to(start, path) == target)


def test_unreachable_is_empty():
    print("unreachable goal returns [] (signal to explore, not to retry)")
    w = {(0, 0): True, (1, 0): True}
    check("disconnected goal", _map_step_toward(w, (0, 0), (9, 9)) == [])


def test_stamp_stitches_screens():
    print("stamping folds each screen into map coordinates")
    w = {}
    # 9x10 screen of all-walkable, player at map (10, 20)
    screen = [[True] * 10 for _ in range(9)]
    _stamp_screen(w, (10, 20), screen)
    check("screen height folded", len(w) == 90, f"{len(w)} cells")
    # Player's own tile maps to their map position
    check("player tile at map pos", w.get((10, 20)) is True)
    # Top-left of the screen is PLAYER_ROW/COL away from the player
    check("top-left offset correct",
          (10 - PLAYER_COL, 20 - PLAYER_ROW) in w)


def test_two_screens_connect():
    print("two overlapping screens stitch into a longer path")
    w = {}
    full = {(x, y): True for x in range(10) for y in range(20)}
    _stamp_screen(w, (0, 0), [[full.get((x - PLAYER_COL, y - PLAYER_ROW), False)
                               for x in range(10)] for y in range(9)])
    _stamp_screen(w, (0, 8), [[full.get((x - PLAYER_COL, 8 + y - PLAYER_ROW), False)
                               for x in range(10)] for y in range(9)])
    path = _map_step_toward(w, (0, 0), (0, 10))
    check("path across both screens", bool(path), f"{len(path)} steps")
    if path:
        check("arrives", walk_to((0, 0), path) == (0, 10))


if __name__ == "__main__":
    for fn in (test_winding_edge, test_direct_target, test_unreachable_is_empty,
               test_stamp_stitches_screens, test_two_screens_connect):
        fn()
        print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} - {FAILS}")
        sys.exit(1)
    print("all map-navigation tests passed")
