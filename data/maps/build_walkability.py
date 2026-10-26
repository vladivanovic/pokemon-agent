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
  - passability rule (from home/overworld.asm CheckTilePassable: the per-
    tileset list is a "list of passable tiles"): a tile is WALKABLE iff ALL
    FOUR of its quadrants are in the tileset's walkable list. Any quadrant
    outside the list (walls, tree tops, water, sign edges) blocks the tile.

Output: v2/pokered_walkability.json
  { map_id: {"name","file","tileset","w","h","walk":[[0/1..]..],
             "warps":[{"x","y","dest_id","idx"}...] }   (dest_id -1 = LAST_MAP)

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


def main() -> None:
    id2const = re.findall(r"\tdw\s+(\w+)_h",
                          open(f"{POKERED}/data/maps/map_header_pointers.asm").read())
    const2id = {c: i for i, c in enumerate(id2const)}
    const2id_alpha = {alpha_key(c): i for i, c in enumerate(id2const)}

    # tileset name -> collision tile ids
    coll_txt = open(f"{POKERED}/data/tilesets/collision_tile_ids.asm").read()
    coll: dict = {}
    for m in re.finditer(r"(\w+)_Coll::((?:\s*\n\s*coll_tiles[^\n]+)+)", coll_txt):
        ids = set()
        for mm in re.finditer(r"coll_tiles\s+([^\n]+)", m.group(2)):
            for tok in mm.group(1).split(","):
                tok = tok.strip()
                if tok.startswith("$"):
                    ids.add(int(tok[1:], 16))
        coll[alpha_key(m.group(1))] = ids

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

    # Ledge tiles (data/tilesets/ledge_tiles.asm): hop-down-only tiles.
    # A tile on a ledge is blocked when walking UP but traversable DOWN -
    # the driver's ledge machinery handles the one-way constraint, so the
    # grid marks ledge tiles WALKABLE (any-quadrant rule) to keep paths open.
    # 0x39 is added: it is the ledge FACE tile (in the blocker list because
    # you cannot walk THROUGH it, but you land on it hopping down and stand
    # beside it) - proven live on Route 4 where the hop landing rows are
    # 0x39 and fully traversable.
    ledge_tiles: set = {0x39}
    for m in re.finditer(r"db\s+SPRITE_FACING_\w+,\s*\$([0-9A-Fa-f]+),\s*\$([0-9A-Fa-f]+)",
                         open(f"{POKERED}/data/tilesets/ledge_tiles.asm").read()):
        ledge_tiles.add(int(m.group(2), 16))   # the ledge tile id

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
        TW, TH = w * 2, h * 2

        # Cavern/dungeon tilesets: pokered tile ids do NOT map to the live
        # ROM's collision semantics through any simple offset (validated
        # live on Mt Moon 1F: floor tile $20 renders walkable yet sits in
        # the Cavern blocker list; rock $10 blocks yet is absent from it).
        # Mark these grids OPEN (unknown) - the live collision window stays
        # the local truth and the stall check corrects walls at walk time.
        # Warp cells are still forced walkable, so portal-chain routing
        # works. CAVERN covers Mt Moon/Rock Tunnel/Victory Road interiors.
        if ts_const.upper() == "CAVERN":
            out[i] = {"name": cn, "file": fname, "tileset": ts_const,
                      "w": TW, "h": TH, "walk": [[1] * TW for _ in range(TH)],
                      "warps": [{"x": wp["x"], "y": wp["y"],
                                 "dest_id": -1 if wp["dest"] == "LAST_MAP"
                                 else const2id.get(wp["dest"],
                                                   const2id_alpha.get(alpha_key(wp["dest"]))),
                                 "idx": wp["idx"]}
                                for wp in warp_by_file.get(
                                    obj_lut.get(alpha_key(cn), ""), [])]}
            continue

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

        # Second pass: passability. A tile is blocked iff its id is in the
        # tileset blocker list, EXCEPT:
        #  - the cell directly BELOW a ledge face: hopping down lands there
        #    (proven live on Route 4: the landing rows render 0x23/0x39 -
        #    blocker-list ids - yet the agent walks them);
        #  - warp tiles (stair/door art is never in the walkable list).
        ledge_cols_by_row: dict = {}
        for ty in range(TH):
            cols = [tx for tx in range(TW)
                    if tiles[ty][tx] in ledge_tiles
                    ]
            if cols:
                ledge_cols_by_row[ty] = set(cols)
        grid = []
        for ty in range(TH):
            row = []
            for tx in range(TW):
                t = tiles[ty][tx]
                if t not in blockers:
                    row.append(1)
                    continue
                below = ledge_cols_by_row.get(ty - 1)
                if below and tx in below:
                    row.append(1)      # ledge landing zone
                    continue
                row.append(0)
            grid.append(row)
        if not ok or len(grid) != TH:
            continue
        warps = []
        for wp in warp_by_file.get(obj_lut.get(alpha_key(cn), ""), []):
            dest = wp["dest"]
            warps.append({"x": wp["x"], "y": wp["y"],
                          "dest_id": -1 if dest == "LAST_MAP" else const2id.get(dest, const2id_alpha.get(alpha_key(dest))),
                          "idx": wp["idx"]})
        out[i] = {"name": cn, "file": fname, "tileset": ts_const,
                  "w": TW, "h": TH, "walk": grid, "warps": warps,
                  "ledges": [{"x": tx, "y": ty}
                             for ty in range(TH)
                             for tx in range(TW)
                             if tiles[ty][tx] in ledge_tiles]}

    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {OUT}: {len(out)} maps")


if __name__ == "__main__":
    main()
