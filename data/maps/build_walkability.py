#!/usr/bin/env python3
"""Render exact walkability grids for all Gen 1 maps from the pokered source.

Replaces the screen-stitching accumulator as the SOURCE OF TRUTH for
navigation. Game geometry is known data, not something to guess from
screenshots.

Data sources (pret/pokered, fetched once):
  maps/<Map>.blk                        - block ids per map
  gfx/blocksets/<ts>.bst                - block -> 16 tile quadrant ids
  data/tilesets/collision_tile_ids.asm  - tiles that block movement
  data/maps/objects/*.asm               - warp events (x, y, dest, idx)
  data/maps/headers/*.asm               - map tilesets

Gen 1 geometry:
  - a BLOCK is 16x16 px = 2x2 game TILES (8 px); a bst entry is one 4x4px
    quadrant; a tile = 2x2 entries
  - map tile size = 2 * block dims (MT_MOON_1F: 20x18 blocks = 40x36 tiles)
  - warp/position coords are in TILES
  - passability rule (home/overworld.asm CheckTilePassable, verified
    against the engine source 2026-11-03): the per-tileset coll_tiles
    list is a list of PASSABLE tiles - the tile in front of the player
    is walkable iff its id IS in the list, and blocked otherwise. The
    cell's id is its block quadrant sample (bottom-left, see below).
    NOTE: the first version of this generator had the test inverted
    (walkable iff NOT in the list), which produced plausible-looking but
    wrong grids that fragmented interiors (Red's bedroom floor read as
    solid wall). Do not reintroduce that inversion.

Output: pokered_walkability.json next to this script
  { map_id: {"name","file","tileset","w","h","walk":[[0/1..]..],
             "warps":[{"x","y","dest_id","idx"}...],
             "ledges":{"x,y":[dx,dy]} (overworld maps only) }
             (dest_id -1 = LAST_MAP)

Usage:  python3 build_walkability.py /path/to/pokered
"""
import json
import os
import re
import sys

POKERED = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pokered"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "pokered_walkability.json")


def alpha_key(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", c.lower())


def parse_connections(pokered: str, id2const, const2id, const2id_alpha,
                        file_lut) -> dict:
    """Map connections WITH their edge offsets, from data/maps/headers/.

    A connection "north, ViridianCity, VIRIDIAN_CITY, -5" in Route 1's
    header means: crossing Route 1's north edge at column x enters
    Viridian at column x - 2*offset (= x + 10) - the offset is in
    BLOCKS (2 tiles), verified against the rendered map (Route 1's
    x=10 path lands in Viridian's entrance corridor at x=20-21, not
    the sealed tree pocket at x=15 that the tile-unit reading gives).
    The inverse connection in Viridian's header carries the opposite
    offset, consistently.
    """
    headers = os.path.join(pokered, "data/maps/headers")
    by_file = {}
    for fn in os.listdir(headers):
        conns = []
        for m in re.finditer(
                r"connection\s+(north|south|east|west),\s*(\w+),\s*(\w+),\s*(-?\d+)",
                open(os.path.join(headers, fn)).read()):
            dst = const2id.get(m.group(3),
                               const2id_alpha.get(alpha_key(m.group(3))))
            if dst is not None:
                conns.append({"dir": m.group(1), "dst": dst,
                              "offset": int(m.group(4))})
        by_file[alpha_key(fn[:-4])] = conns
    out = {}
    for i, cn in enumerate(id2const):
        fname = file_lut.get(alpha_key(cn), "")
        conns = by_file.get(alpha_key(fname), [])
        if conns:
            out[str(i)] = conns
    return out


def main() -> None:
    id2const = re.findall(r"\tdw\s+(\w+)_h",
                          open(f"{POKERED}/data/maps/map_header_pointers.asm").read())
    # First occurrence wins: _COPY maps reuse the original's header
    # label later in the pointer table (e.g. UndergroundPathRoute6_h
    # at ids 74 and 75), and the assembler's own constant resolution
    # names the first. A last-wins dict pointed the Underground
    # Path's exit warp at the $4B copy, orphaning the real house
    # from the route graph (found 2026-11-04).
    const2id: dict = {}
    const2id_alpha: dict = {}
    for i, c in enumerate(id2const):
        const2id.setdefault(c, i)
        const2id_alpha.setdefault(alpha_key(c), i)

    # tileset name -> collision (passable) tile ids. Labels stack in the
    # .asm (Mart_Coll:: Pokecenter_Coll:: share one list), so accumulate
    # pending labels and bind the next list to ALL of them; a regex that
    # binds only the last label silently empties the others' lists.
    coll_txt = open(f"{POKERED}/data/tilesets/collision_tile_ids.asm").read()
    coll: dict = {}
    pending_labels: list = []
    current_ids = None
    for line in coll_txt.splitlines():
        s = line.strip()
        lm = re.match(r"(\w+)_Coll::", s)
        if lm:
            pending_labels.append(lm.group(1))
            current_ids = None
            continue
        cm = re.match(r"coll_tiles\s+(.*)", s)
        if cm:
            if current_ids is None:
                current_ids = set()
                for lab in pending_labels:
                    coll[alpha_key(lab)] = current_ids
                pending_labels = []
            for tok in cm.group(1).split(","):
                tok = tok.strip()
                if tok.startswith("$"):
                    current_ids.add(int(tok[1:], 16))

    # tileset name -> blockset file (sequential parse: alias labels share one file)
    block_file: dict = {}
    pending: list = []
    for line in open(f"{POKERED}/gfx/tilesets.asm").read().splitlines():
        s = line.strip()
        m = re.match(r"(\w+)_Block::", s)
        if m:
            pending.append(m.group(1).upper())
            m2 = re.search(r"INCBIN \"gfx/blocksets/(\w+\.bst)\"", s)
            if m2:
                for nm in pending:
                    block_file[alpha_key(nm)] = m2.group(1)
                pending = []
            continue
        m2 = re.search(r"INCBIN \"gfx/blocksets/(\w+\.bst)\"", s)
        if m2 and pending:
            for nm in pending:
                block_file[alpha_key(nm)] = m2.group(1)
            pending = []

    # map name -> tileset (headers)
    map_ts: dict = {}
    for fn in os.listdir(f"{POKERED}/data/maps/headers"):
        m = re.search(r"map_header\s+\w+,\s*\w+,\s*(\w+)",
                      open(f"{POKERED}/data/maps/headers/{fn}").read())
        if m:
            map_ts[fn[:-4]] = m.group(1)

    # warps per map (objects)
    warp_by_file: dict = {}
    for fn in os.listdir(f"{POKERED}/data/maps/objects"):
        ws = []
        for m in re.finditer(r"warp_event\s+(\d+),\s*(\d+),\s*(\w+),\s*(\w+)",
                             open(f"{POKERED}/data/maps/objects/{fn}").read()):
            ws.append({"x": int(m.group(1)), "y": int(m.group(2)),
                       "dest": m.group(3), "idx": int(m.group(4), 0)})
        warp_by_file[fn[:-4]] = ws

    # dims: map_const NAME, W, H ; id
    dims: dict = {}
    dims_by_alpha: dict = {}
    for m in re.finditer(r"map_const\s+(\w+),\s*(\d+),\s*(\d+)\s*;\s*\$?([0-9A-Fa-f]+)",
                         open(f"{POKERED}/constants/map_constants.asm").read()):
        dims[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        dims_by_alpha[alpha_key(m.group(1))] = (int(m.group(2)), int(m.group(3)))

    # Ledges (data/tilesets/ledge_tiles.asm + HandleLedges in the
    # engine): a ledge triple is (facing, tile stood on, FACE tile).
    # The face tiles ($36/$37 down, $27 left, $0D/$1D right, OVERWORLD
    # tileset only) are NOT in the passable lists, so the grid treats
    # them as walls - but a jump is NOT always a shortcut: on Route 4
    # the ledge drops are the ONLY way from Mt. Moon's exit down to
    # Cerulean, and a walls-only model finds no route at all (verified
    # live, 2026-11-03). So emit the FACE cells and their jump
    # directions as "ledges"; world.py's BFS treats a face as a
    # pass-through node onto the landing tile beyond it. The earlier
    # disaster came from flagging $39 - the tile you STAND ON at a
    # ledge top and a common ground tile - which diced Pallet Town
    # with 90 phantom one-way tiles. Faces only, never $39/$23.
    LEDGE_FACES = {0x36: (0, 1), 0x37: (0, 1), 0x27: (-1, 0),
                   0x0D: (1, 0), 0x1D: (1, 0)}

    bstdir = f"{POKERED}/gfx/blocksets"
    mapsdir = f"{POKERED}/maps"
    file_lut = {alpha_key(fn[:-4]): fn[:-4] for fn in os.listdir(mapsdir)}
    obj_lut = {alpha_key(fn[:-4]): fn[:-4] for fn in os.listdir(f"{POKERED}/data/maps/objects")}

    out: dict = {}
    for i, cn in enumerate(id2const):
        fname = file_lut.get(alpha_key(cn), "")
        ts_const = map_ts.get(fname)
        if not ts_const:
            continue
        # Tileset CONSTANT names and GFX label names differ on aliased
        # tilesets (const REDS_HOUSE_2 vs label RedsHouse2). alpha_key
        # strips underscores + case, making them comparable.
        ts_key = alpha_key(ts_const)
        bf = block_file.get(ts_key)
        if not bf or not os.path.exists(f"{bstdir}/{bf}"):
            continue
        blk_path = f"{mapsdir}/{fname}.blk"
        if not os.path.exists(blk_path):
            continue
        w, h = dims_by_alpha.get(alpha_key(cn), (None, None))
        if not w:
            continue
        bst = open(f"{bstdir}/{bf}", "rb").read()
        blocks = [bst[j * 16:(j + 1) * 16] for j in range(len(bst) // 16)]
        blockers = coll.get(alpha_key(ts_const), set())
        blk = open(blk_path, "rb").read()
        if len(blk) < w * h and len(blk) % w == 0:
            # The const table can overstate a map's height: pokered's
            # own comment on UNDERGROUND_PATH_NORTH_SOUTH notes its
            # .blk "is actually 4x23" against a 4x24 const. The .blk
            # is the ground truth the ROM actually contains; trusting
            # the taller claim makes the tile loop below read past
            # the file, flag the map bad and skip it entirely - the
            # Underground Path interior (map 119) was missing from
            # the data for exactly this reason (found 2026-11-04).
            h = len(blk) // w
        TW, TH = w * 2, h * 2

        # CAVERN maps decode with the same rule as everywhere else.
        # An earlier version marked them OPEN ("tile ids do not map
        # through the collision lists") - that verdict was reached
        # while the passable list was still being read as a BLOCKER
        # list, i.e. during the inversion confusion: floor $20 being
        # IN the Cavern list means walkable, rock $10 being absent
        # means blocked, exactly as the rule predicts. The open grids
        # did real damage before this was caught: they let the
        # planner draw straight lines through Mt. Moon's rock and
        # plan a direct B1F walk to the exit chamber, when the true
        # route detours through B2F. Verified after the fix: the
        # decoded grids agree with all 46 cave cells the agent has
        # ever stood on across Mt. Moon 1F/B1F, and refuse every
        # confirmed wall.

        # First pass: rendered tile per cell = block's BOTTOM-LEFT quadrant
        # (base+4) - validated 70/70 against the emulator's live tile_ids on
        # Route 4 (blakecrosley: "Gen 1 cells read the lower-left 8x8 of
        # each 16x16 cell").
        tiles = []
        ok = True
        for ty in range(TH):
            row = []
            for tx in range(TW):
                idx = (ty // 2) * w + tx // 2
                if idx >= len(blk):
                    ok = False
                    break
                b = blocks[blk[idx]]
                row.append(b[(ty % 2) * 8 + (tx % 2) * 2 + 4])
            if not ok:
                break
            tiles.append(row)
        if not ok:
            continue

        # Second pass: passability. coll_tiles is the PASSABLE list (see
        # header): a cell is walkable iff its sampled id is in the list.
        # Ledge-face and landing ids ($39, $23 on the overworld) are in
        # the list already, so no special-casing is needed. Warp cells
        # are forced walkable afterwards: a door/stair trigger you can
        # never step onto would be a data bug, not a wall.
        passable = blockers  # (misnamed in the original: it is the passable set)
        grid = [[1 if tiles[ty][tx] in passable else 0
                 for tx in range(TW)] for ty in range(TH)]
        if not ok or len(grid) != TH:
            continue
        warps = []
        for wp in warp_by_file.get(obj_lut.get(alpha_key(cn), ""), []):
            dest = wp["dest"]
            warps.append({"x": wp["x"], "y": wp["y"],
                          "dest_id": -1 if dest == "LAST_MAP" else const2id.get(dest, const2id_alpha.get(alpha_key(dest))),
                          "idx": wp["idx"]})
        for wp in warps:
            if 0 <= wp["x"] < TW and 0 <= wp["y"] < TH:
                grid[wp["y"]][wp["x"]] = 1
        # Coherence fallback: if a multi-warp map's warps cannot reach
        # each other at all, the extraction for this map is not trusted;
        # mark it open (warp chains still route) rather than seal it.
        if len(warps) > 1:
            from collections import deque
            seen = {(warps[0]["x"], warps[0]["y"])}
            dq = deque(seen)
            while dq:
                cx, cy = dq.popleft()
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    n = (cx + dx, cy + dy)
                    if 0 <= n[0] < TW and 0 <= n[1] < TH and n not in seen \
                            and grid[n[1]][n[0]] == 1:
                        seen.add(n)
                        dq.append(n)
            if not any((wp["x"], wp["y"]) in seen for wp in warps[1:]):
                print(f"  ! {cn}: warps unreachable in extracted grid; "
                      "falling back to open grid")
                grid = [[1] * TW for _ in range(TH)]
        ledges = {}
        if ts_const.upper() == "OVERWORLD":
            for ty in range(TH):
                for tx in range(TW):
                    delta = LEDGE_FACES.get(tiles[ty][tx])
                    if delta is not None:
                        ledges[f"{tx},{ty}"] = list(delta)
        out[i] = {"name": cn, "file": fname, "tileset": ts_const,
                  "w": TW, "h": TH, "walk": grid, "warps": warps,
                  **({"ledges": ledges} if ledges else {})}

    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {OUT}: {len(out)} maps")

    conns = parse_connections(POKERED, id2const, const2id, const2id_alpha,
                              file_lut)
    conn_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "pokered_connections.json")
    with open(conn_out, "w") as f:
        json.dump(conns, f)
    print(f"wrote {conn_out}: {len(conns)} maps with connections")


if __name__ == "__main__":
    main()
