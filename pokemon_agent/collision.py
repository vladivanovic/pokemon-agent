"""Gen 1 collision-map extraction.

Reads the on-screen background tilemap (``wTileMap`` at 0xC3A0, a 20x18 grid
of 8x8 hardware tile ids) and the current tileset id (``wCurMapTileset`` at
0xD367), then classifies each of the 10x9 walkable *blocks* as walkable or
blocked using the authoritative per-tileset collision lists from the pokered
disassembly (``data/tilesets/collision_tile_ids.asm``).

A Gen 1 overworld "block" is 16x16 px = a 2x2 group of 8x8 tiles. The screen
shows 10 blocks across and 9 down. The player is locked to block (col 4,
row 4) — grid cell "E5". Walkability of a block is decided by its top-left
8x8 tile id, which is what the engine itself checks.

Output grid orientation: ``grid[row][col]`` with row 0 at the top, col 0 at
the left; ``True`` means walkable.
"""

from __future__ import annotations

from typing import Dict, List, Optional

ADDR_TILEMAP       = 0xC3A0   # wTileMap, 20x18 bytes
ADDR_TILESET       = 0xD367   # wCurMapTileset
ADDR_TILE_IN_FRONT = 0xCFC6   # wTileInFrontOfPlayer
ADDR_WALK_COUNTER  = 0xCFC5   # wWalkCounter; nonzero = mid-step
ADDR_SPRITE_DATA1  = 0xC100   # wSpriteStateData1, 16 slots x 16 bytes
ADDR_NUM_WARPS     = 0xD3AE   # wNumberOfWarps
ADDR_WARP_ENTRY    = 0xD3AF   # wWarpEntries: y, x, dest_warp, dest_map

TILEMAP_W, TILEMAP_H = 20, 18
BLOCK_COLS = 10               # on-screen walkable blocks across
BLOCK_ROWS = 9                # on-screen walkable blocks down
BLOCK_PX   = 16               # world block size in GB pixels
PLAYER_COL = 4                # block the player is locked to (cell E5)
PLAYER_ROW = 4
GRID_ROW_OFFSET = 1           # tilemap row offset; see read_block_tile_ids

# Derived — must come after the above.
PLAYER_PX_X = PLAYER_COL * BLOCK_PX                        # 64
PLAYER_PX_Y = PLAYER_ROW * BLOCK_PX + GRID_ROW_OFFSET * 8  # 72

SPRITE_SLOT_SIZE = 16
SPRITE_SLOTS     = 16
S_PICTURE_ID, S_YPIXELS, S_XPIXELS = 0x00, 0x04, 0x06
WARP_ENTRY_SIZE, MAX_WARPS = 4, 32

COL_LABELS = "ABCDEFGHIJ"
_FACING_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

# Per-tileset walkable tile-id sets, transcribed from pokered
# data/tilesets/collision_tile_ids.asm. Key = wCurMapTileset value.
TILESET_WALKABLE: Dict[int, frozenset] = {
    0: frozenset({0x00, 0x10, 0x1B, 0x20, 0x21, 0x23, 0x2C, 0x2D, 0x2E, 0x30, 0x31, 0x33, 0x39, 0x3C, 0x3E, 0x52, 0x54, 0x58, 0x5B}),  # Overworld
    1: frozenset({0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x14, 0x1A, 0x1C}),  # RedsHouse1
    2: frozenset({0x11, 0x1A, 0x1C, 0x3C, 0x5E}),  # Mart
    3: frozenset({0x1E, 0x20, 0x2E, 0x30, 0x34, 0x37, 0x39, 0x3A, 0x40, 0x51, 0x52, 0x5A, 0x5C, 0x5E, 0x5F}),  # Forest
    4: frozenset({0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x14, 0x1A, 0x1C}),  # RedsHouse2
    5: frozenset({0x03, 0x11, 0x16, 0x19, 0x2B, 0x3C, 0x3D, 0x3F, 0x4A, 0x4C, 0x4D}),  # Dojo
    6: frozenset({0x11, 0x1A, 0x1C, 0x3C, 0x5E}),  # Pokecenter
    7: frozenset({0x03, 0x11, 0x16, 0x19, 0x2B, 0x3C, 0x3D, 0x3F, 0x4A, 0x4C, 0x4D}),  # Gym
    8: frozenset({0x01, 0x12, 0x14, 0x28, 0x32, 0x37, 0x44, 0x54, 0x5C}),  # House
    9: frozenset({0x01, 0x12, 0x14, 0x1A, 0x1C, 0x37, 0x38, 0x3B, 0x3C, 0x5E}),  # ForestGate
    10: frozenset({0x01, 0x12, 0x14, 0x1A, 0x1C, 0x37, 0x38, 0x3B, 0x3C, 0x5E}),  # Museum
    11: frozenset({0x0B, 0x0C, 0x13, 0x15, 0x18}),  # Underground
    12: frozenset({0x01, 0x12, 0x14, 0x1A, 0x1C, 0x37, 0x38, 0x3B, 0x3C, 0x5E}),  # Gate
    13: frozenset({0x04, 0x0D, 0x17, 0x1D, 0x1E, 0x23, 0x34, 0x37, 0x39, 0x4A}),  # Ship
    14: frozenset({0x0A, 0x1A, 0x32, 0x3B}),  # ShipPort
    15: frozenset({0x01, 0x10, 0x13, 0x1B, 0x22, 0x42, 0x52}),  # Cemetery
    16: frozenset({0x04, 0x0F, 0x15, 0x1F, 0x3B, 0x45, 0x47, 0x55, 0x56}),  # Interior
    17: frozenset({0x05, 0x15, 0x18, 0x1A, 0x20, 0x21, 0x22, 0x2A, 0x2D, 0x30}),  # Cavern
    18: frozenset({0x14, 0x17, 0x1A, 0x1C, 0x20, 0x38, 0x45}),  # Lobby
    19: frozenset({0x01, 0x05, 0x11, 0x12, 0x14, 0x1A, 0x1C, 0x2C, 0x53}),  # Mansion
    20: frozenset({0x0C, 0x16, 0x1E, 0x26, 0x34, 0x37}),  # Lab
    21: frozenset({0x0F, 0x1A, 0x1F, 0x26, 0x28, 0x29, 0x2C, 0x2D, 0x2E, 0x2F, 0x41}),  # Club
    22: frozenset({0x01, 0x10, 0x11, 0x13, 0x1B, 0x20, 0x21, 0x22, 0x30, 0x31, 0x32, 0x42, 0x43, 0x48, 0x52, 0x55, 0x58, 0x5E}),  # Facility
    23: frozenset({0x1B, 0x23, 0x2C, 0x2D, 0x3B, 0x45}),  # Plateau
}

COL_LABELS = "ABCDEFGHIJ"


_FACING_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def read_sprite_cells(emu) -> List[Dict]:
    """Screen-relative block cells occupied by NPCs.

    Slot 0 is the player, so positions are computed as deltas from it and
    converted to block offsets from E5. This avoids depending on the
    absolute pixel bias of the sprite fields.
    """
    raw = emu.read_range(ADDR_SPRITE_DATA1, SPRITE_SLOT_SIZE * SPRITE_SLOTS)
    if not raw[S_PICTURE_ID]:
        return []
    py, px = raw[S_YPIXELS], raw[S_XPIXELS]

    out: List[Dict] = []
    for slot in range(1, SPRITE_SLOTS):
        base = slot * SPRITE_SLOT_SIZE
        if not raw[base + S_PICTURE_ID]:
            continue
        dy = ((raw[base + S_YPIXELS] - py + 128) % 256) - 128
        dx = ((raw[base + S_XPIXELS] - px + 128) % 256) - 128
        if abs(dy) > 128 or abs(dx) > 128:
            continue
        r = PLAYER_ROW + int(round(dy / BLOCK_PX))
        c = PLAYER_COL + int(round(dx / BLOCK_PX))
        if 0 <= r < BLOCK_ROWS and 0 <= c < BLOCK_COLS:
            out.append({"slot": slot, "row": r, "col": c,
                        "cell": cell_label(c, r)})
    return out


def read_warp_cells(emu, player_x: int, player_y: int) -> List[Dict]:
    """Doors/stairs/exits as screen-relative cells.

    Warp entries are in map coordinates, and one overworld block is one
    map tile, so the conversion is a plain delta from the player's tile.
    """
    n = emu.read_u8(ADDR_NUM_WARPS)
    if not 0 < n <= MAX_WARPS:
        return []
    raw = emu.read_range(ADDR_WARP_ENTRY, WARP_ENTRY_SIZE * n)
    out: List[Dict] = []
    for i in range(n):
        wy, wx, dest_warp, dest_map = raw[i * 4:i * 4 + 4]
        r = PLAYER_ROW + (wy - player_y)
        c = PLAYER_COL + (wx - player_x)
        if 0 <= r < BLOCK_ROWS and 0 <= c < BLOCK_COLS:
            out.append({"row": r, "col": c, "cell": cell_label(c, r),
                        "dest_map": dest_map, "dest_map_name": None,
                        "dest_warp": dest_warp})
    return out



def cell_label(col: int, row: int) -> str:
    return f"{COL_LABELS[col]}{row + 1}"


def read_block_tile_ids(emu) -> List[List[int]]:
    """Return the 9x10 grid of representative tile ids per block.

    The player's standing tile in ``wTileMap`` is screen tile (col 8, row 9),
    so the 16px block grid is sampled with a +1 row offset: block (bc, br)
    maps to tilemap tile (bc*2, br*2 + 1). This puts the player at block
    (col 4, row 4) = cell E5 and makes the "tile above" / collision checks
    line up with the engine's own movement rules.
    """
    tm = emu.read_range(ADDR_TILEMAP, TILEMAP_W * TILEMAP_H)
    grid: List[List[int]] = []
    for br in range(BLOCK_ROWS):
        row: List[int] = []
        for bc in range(BLOCK_COLS):
            tcol, trow = bc * 2, br * 2 + GRID_ROW_OFFSET
            row.append(tm[trow * TILEMAP_W + tcol])
        grid.append(row)
    return grid


def build_collision_grid(emu, facing: Optional[str] = None, player_pos: Optional[Dict] = None) -> Dict:
    """Build a walkability grid for the current on-screen blocks.

    Returns a dict with:
        tileset: int
        walkable: 9x10 list of bool (True = can step there)
        tile_ids: 9x10 list of int (raw representative tile ids)
        player_cell: "E5"
    The player's own cell is always reported walkable.
    """
    tileset = emu.read_u8(ADDR_TILESET)
    walk_set = TILESET_WALKABLE.get(tileset)
    known = walk_set is not None
    tile_ids = read_block_tile_ids(emu)
    walk_set = walk_set or frozenset()

    walkable = [[tile_ids[r][c] in walk_set for c in range(BLOCK_COLS)]
                for r in range(BLOCK_ROWS)]
    player_tile_walkable = walkable[PLAYER_ROW][PLAYER_COL]
    walkable[PLAYER_ROW][PLAYER_COL] = True

    sprites = read_sprite_cells(emu)
    warps = (read_warp_cells(emu, player_pos["x"], player_pos["y"])
             if player_pos else [])
    occupied = {(s["row"], s["col"]) for s in sprites}

    # Terrain walkability stays separate from transient sprite blocking so
    # callers can tell "wall" from "someone is standing there".
    passable = [[walkable[r][c] and (r, c) not in occupied
                 for c in range(BLOCK_COLS)] for r in range(BLOCK_ROWS)]
    passable[PLAYER_ROW][PLAYER_COL] = True

    settled = emu.read_u8(ADDR_WALK_COUNTER) == 0
    return {
        "valid": known and settled,
        "tileset": tileset,
        "tileset_known": known,
        "camera_settled": settled,
        "walkable": walkable,
        "passable": passable,
        "sprites": sprites,
        "warps": warps,
        "tile_ids": tile_ids,
        "player_cell": cell_label(PLAYER_COL, PLAYER_ROW),
        "player_tile_walkable": player_tile_walkable,
        "offset_verified": verify_offset(emu, facing, tile_ids) if facing else None,
        "geometry": {"block_px": BLOCK_PX, "player_px": [PLAYER_PX_X, PLAYER_PX_Y],
                     "cols": BLOCK_COLS, "rows": BLOCK_ROWS},
    }


def render_ascii_map(collision: Dict, legend: bool = True) -> str:
    """Render the collision grid as a labelled ASCII map.

    Legend:
        @ = player (E5)   . = walkable   # = blocked
    Column headers A..J, row numbers 1..9.
    """
    if not collision.get("valid", True):
        reasons = []
        if not collision.get("tileset_known", True):
            reasons.append(f"unknown tileset {collision.get('tileset')}")
        if not collision.get("camera_settled", True):
            reasons.append("camera mid-scroll")
        return "(map unavailable: " + ", ".join(reasons or ["invalid"]) + ")"

    walkable = collision["walkable"]
    npc = {(s["row"], s["col"]) for s in collision.get("sprites", [])}
    warp = {(w["row"], w["col"]) for w in collision.get("warps", [])}
    lines: List[str] = []
    header = "   " + " ".join(COL_LABELS)
    lines.append(header)
    for r in range(BLOCK_ROWS):
        cells = []
        for c in range(BLOCK_COLS):
            if r == PLAYER_ROW and c == PLAYER_COL:
                cells.append("@")
            elif (r, c) in npc:
                cells.append("N")
            elif (r, c) in warp:
                cells.append("D")
            else:
                cells.append("." if walkable[r][c] else "#")
        lines.append(f"{r + 1:>2} " + " ".join(cells))
    if legend:
        lines.append("")
        lines.append("@ you (E5)  . walkable  # blocked  N person  D door/exit")
        lines.append("walk_up=row-1  walk_down=row+1  walk_left=col-1  walk_right=col+1")
        if collision.get("warps"):
            lines.append("exits: " + ", ".join(
                f"{w['cell']}->map{w['dest_map']}" for w in collision["warps"]))
    return "\n".join(lines)
