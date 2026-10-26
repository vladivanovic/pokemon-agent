"""Load pokered ground-truth walkability + warp tables as map-name keyed data.

This is the SOURCE OF TRUTH for navigation: the game's own geometry parsed
from the pret/pokered disassembly (build_walkability.py). The screen-stitched
map remains useful only as an overlay of personally-verified cells.

pokered_walkability.json maps pokered map-id -> {w, h, walk, warps}. The
emulator reports map ids (context.raw.map_id / state map id) and the same
ids index gen1_data.MAP_NAMES, so joining is direct.
"""
import json
import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "maps")
WALK_PATH = os.path.join(DATA_DIR, "pokered_walkability.json")


def load_ground_truth() -> Dict[str, dict]:
    """pokered map-id (as str) -> walkability record."""
    try:
        with open(WALK_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        logger.warning("pokered walkability not found at %s", WALK_PATH)
        return {}


def walk_grid_by_name(gt: Dict[str, dict], map_names: Dict[int, str]
                      ) -> Dict[str, Dict[Tuple[int, int], bool]]:
    """Convert to the shape von.py uses: map_name -> {(x, y): walkable}.

    Warp cells are FORCED walkable: their stair/door graphics are not in the
    tileset walkable list, but stepping on them is exactly how the agent
    traverses floors and enters buildings. Without this the BFS refuses to
    path to its own targets (Mt Moon stair (5,5) read as a wall)."""
    out: Dict[str, Dict[Tuple[int, int], bool]] = {}
    for pid_str, rec in gt.items():
        name = map_names.get(int(pid_str))
        if not name:
            continue
        grid: Dict[Tuple[int, int], bool] = {}
        for y, row in enumerate(rec["walk"]):
            for x, ok in enumerate(row):
                grid[(x, y)] = bool(ok)
        for w in rec.get("warps") or []:
            wx, wy = int(w["x"]), int(w["y"])
            if 0 <= wy < rec["h"] and 0 <= wx < rec["w"]:
                grid[(wx, wy)] = True
        # Border-adjacent cells are potentially walkable: Gen 1 maps connect
        # through walk-off edges, and the stitched/live observations prove
        # agents stand on cells the static tile rule calls blocked (ledge
        # faces, border rows). Mark the outermost ring UNKNOWN-walkable so
        # BFS can reach map edges; the live window + stall guard correct
        # any actually-blocked border cell at walk time.
        w_t, h_t = rec["w"], rec["h"]
        for x in range(w_t):
            for y in (0, h_t - 1):
                if (x, y) not in grid or not grid[(x, y)]:
                    grid[(x, y)] = True
        for y in range(h_t):
            for x in (0, w_t - 1):
                if (x, y) not in grid or not grid[(x, y)]:
                    grid[(x, y)] = True
        out[name] = grid
    return out


def warps_by_name(gt: Dict[str, dict], map_names: Dict[int, str]
                  ) -> Dict[str, list]:
    """map_name -> [{x, y, dest_name, idx}] (dest -1/None = LAST_MAP)."""
    out: Dict[str, list] = {}
    for pid_str, rec in gt.items():
        name = map_names.get(int(pid_str))
        if not name:
            continue
        warps = []
        for w in rec.get("warps") or []:
            dest_name = map_names.get(w["dest_id"]) if (
                isinstance(w.get("dest_id"), int) and w["dest_id"] >= 0) else None
            warps.append({"x": w["x"], "y": w["y"],
                          "dest_name": dest_name, "idx": w["idx"]})
        out[name] = warps
    return out


def ledges_by_name(gt: Dict[str, dict], map_names: Dict[int, str]
                   ) -> Dict[str, Dict[Tuple[int, int], Tuple[int, int]]]:
    """map_name -> {(x, y): (x, y+2)} hop-down edges from pokered ledge tiles.

    The game rule: walking DOWN onto a ledge tile makes the engine hop two
    tiles. The grid marks ledge tiles walkable (down-only); this table gives
    the BFS the explicit hop edge so paths cross the ledge row instead of
    dead-ending at it."""
    out: Dict[str, Dict[Tuple[int, int], Tuple[int, int]]] = {}
    for pid_str, rec in gt.items():
        name = map_names.get(int(pid_str))
        if not name:
            continue
        edges: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for lg in rec.get("ledges") or []:
            x, y = int(lg["x"]), int(lg["y"])
            if y + 2 < rec["h"]:
                edges[(x, y)] = (x, y + 2)
        if edges:
            out[name] = edges
    return out


def grid_size(gt: Dict[str, dict], pid: int) -> Optional[Tuple[int, int]]:
    rec = gt.get(str(pid))
    if not rec:
        return None
    return (rec["w"], rec["h"])
