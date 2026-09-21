"""Gen 1 collision-map extraction.

Reads the on-screen background tilemap (``wTileMap`` at 0xC3A0, a 20x18 grid
of 8x8 hardware tile ids) and the current tileset id (``wCurMapTileset`` at
0xD367), then classifies each of the 10x9 walkable *blocks* as walkable or
blocked using the per-tileset collision lists from the pokered disassembly
(``data/tilesets/collision_tile_ids.asm``).

A Gen 1 overworld "block" is 16x16 px = a 2x2 group of 8x8 tiles. The screen
shows 10 blocks across and 9 down. The player is locked to block (col 4,
row 4) — grid cell "E5". Walkability of a block is decided by its top-left
8x8 tile id, which is what the engine itself checks.

NPC positions (``wSpriteStateData1``) and map warps (``wWarpEntries``) are
overlaid so the map shows what actually blocks movement and where the exits
are, not just terrain.

Output grid orientation: ``grid[row][col]`` with row 0 at the top, col 0 at
the left; ``True`` means walkable.

Nothing here is trusted blindly: the grid carries ``valid``, ``tileset_known``,
``camera_settled`` and ``offset_verified`` flags, and ``render_ascii_map``
refuses to draw a map it cannot vouch for. A confidently wrong map is worse
than no map.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# RAM addresses (pokered symbol names in comments)
# ---------------------------------------------------------------------------

ADDR_TILEMAP       = 0xC3A0   # wTileMap, 20x18 bytes
ADDR_TILESET       = 0xD367   # wCurMapTileset
ADDR_TILE_IN_FRONT = 0xCFC6   # wTileInFrontOfPlayer
ADDR_WALK_COUNTER  = 0xCFC5   # wWalkCounter; nonzero = mid-step, camera moving
ADDR_SPRITE_DATA1  = 0xC100   # wSpriteStateData1, 16 slots x 16 bytes
ADDR_NUM_WARPS     = 0xD3AE   # wNumberOfWarps
ADDR_WARP_ENTRY    = 0xD3AF   # wWarpEntries: y, x, dest_warp, dest_map

# ---------------------------------------------------------------------------
# Geometry — the single source of truth. overlay.py imports these; do not
# redefine them there or the picture and the ASCII map will drift apart.
# ---------------------------------------------------------------------------

TILEMAP_W, TILEMAP_H = 20, 18
BLOCK_COLS = 10               # on-screen walkable blocks across
BLOCK_ROWS = 9                # on-screen walkable blocks down
BLOCK_PX   = 16               # world block size in GB pixels
PLAYER_COL = 4                # block the player is locked to (cell E5)
PLAYER_ROW = 4
GRID_ROW_OFFSET = 1           # tilemap row offset; see read_block_tile_ids()

# Derived — must stay below the definitions above.
PLAYER_PX_X = PLAYER_COL * BLOCK_PX                        # 64
PLAYER_PX_Y = PLAYER_ROW * BLOCK_PX + GRID_ROW_OFFSET * 8  # 72

SPRITE_SLOT_SIZE = 16
SPRITE_SLOTS     = 16
S_PICTURE_ID     = 0x00       # 0 = slot disabled
S_YPIXELS        = 0x04       # screen Y, biased
S_XPIXELS        = 0x06       # screen X, biased

WARP_ENTRY_SIZE = 4
MAX_WARPS       = 32

COL_LABELS = "ABCDEFGHIJ"
_FACING_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

# ---------------------------------------------------------------------------
# Per-tileset walkable tile-id sets, transcribed from pokered
# data/tilesets/collision_tile_ids.asm. Key = wCurMapTileset value.
# Several tilesets legitimately share a list (pokered reuses them).
# ---------------------------------------------------------------------------

TILESET_WALKABLE: Dict[int, frozenset] = {
    0:  frozenset({0x00, 0x10, 0x1B, 0x20, 0x21, 0x23, 0x2C, 0x2D, 0x2E, 0x30,
                   0x31, 0x33, 0x39, 0x3C, 0x3E, 0x52, 0x54, 0x58, 0x5B}),      # Overworld
    1:  frozenset({0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x14, 0x1A, 0x1C}),      # RedsHouse1
    2:  frozenset({0x11, 0x1A, 0x1C, 0x3C, 0x5E}),                              # Mart
    3:  frozenset({0x1E, 0x20, 0x2E, 0x30, 0x34, 0x37, 0x39, 0x3A, 0x40, 0x51,
                   0x52, 0x5A, 0x5C, 0x5E, 0x5F}),                              # Forest
    4:  frozenset({0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x14, 0x1A, 0x1C}),      # RedsHouse2
    5:  frozenset({0x03, 0x11, 0x16, 0x19, 0x2B, 0x3C, 0x3D, 0x3F, 0x4A, 0x4C,
                   0x4D}),                                                      # Dojo
    6:  frozenset({0x11, 0x1A, 0x1C, 0x3C, 0x5E}),                              # Pokecenter
    7:  frozenset({0x03, 0x11, 0x16, 0x19, 0x2B, 0x3C, 0x3D, 0x3F, 0x4A, 0x4C,
                   0x4D}),                                                      # Gym
    8:  frozenset({0x01, 0x12, 0x14, 0x28, 0x32, 0x37, 0x44, 0x54, 0x5C}),      # House
    9:  frozenset({0x01, 0x12, 0x14, 0x1A, 0x1C, 0x37, 0x38, 0x3B, 0x3C, 0x5E}),# ForestGate
    10: frozenset({0x01, 0x12, 0x14, 0x1A, 0x1C, 0x37, 0x38, 0x3B, 0x3C, 0x5E}),# Museum
    11: frozenset({0x0B, 0x0C, 0x13, 0x15, 0x18}),                              # Underground
    12: frozenset({0x01, 0x12, 0x14, 0x1A, 0x1C, 0x37, 0x38, 0x3B, 0x3C, 0x5E}),# Gate
    13: frozenset({0x04, 0x0D, 0x17, 0x1D, 0x1E, 0x23, 0x34, 0x37, 0x39, 0x4A}),# Ship
    14: frozenset({0x0A, 0x1A, 0x32, 0x3B}),                                    # ShipPort
    15: frozenset({0x01, 0x10, 0x13, 0x1B, 0x22, 0x42, 0x52}),                  # Cemetery
    16: frozenset({0x04, 0x0F, 0x15, 0x1F, 0x3B, 0x45, 0x47, 0x55, 0x56}),      # Interior
    17: frozenset({0x05, 0x15, 0x18, 0x1A, 0x20, 0x21, 0x22, 0x2A, 0x2D, 0x30}),# Cavern
    18: frozenset({0x14, 0x17, 0x1A, 0x1C, 0x20, 0x38, 0x45}),                  # Lobby
    19: frozenset({0x01, 0x05, 0x11, 0x12, 0x14, 0x1A, 0x1C, 0x2C, 0x53}),      # Mansion
    20: frozenset({0x0C, 0x16, 0x1E, 0x26, 0x34, 0x37}),                        # Lab
    21: frozenset({0x0F, 0x1A, 0x1F, 0x26, 0x28, 0x29, 0x2C, 0x2D, 0x2E, 0x2F,
                   0x41}),                                                      # Club
    22: frozenset({0x01, 0x10, 0x11, 0x13, 0x1B, 0x20, 0x21, 0x22, 0x30, 0x31,
                   0x32, 0x42, 0x43, 0x48, 0x52, 0x55, 0x58, 0x5E}),            # Facility
    23: frozenset({0x1B, 0x23, 0x2C, 0x2D, 0x3B, 0x45}),                        # Plateau
}

TILESET_NAMES: Dict[int, str] = {
    0: "Overworld", 1: "RedsHouse1", 2: "Mart", 3: "Forest", 4: "RedsHouse2",
    5: "Dojo", 6: "Pokecenter", 7: "Gym", 8: "House", 9: "ForestGate",
    10: "Museum", 11: "Underground", 12: "Gate", 13: "Ship", 14: "ShipPort",
    15: "Cemetery", 16: "Interior", 17: "Cavern", 18: "Lobby", 19: "Mansion",
    20: "Lab", 21: "Club", 22: "Facility", 23: "Plateau",
}


# ---------------------------------------------------------------------------
# Cell labelling
# ---------------------------------------------------------------------------

def cell_label(col: int, row: int) -> str:
    """0-indexed (col, row) -> 'E5'. Raises on out-of-range rather than
    silently producing a label that points nowhere."""
    if not (0 <= col < BLOCK_COLS and 0 <= row < BLOCK_ROWS):
        raise ValueError(f"cell out of range: col={col} row={row}")
    return f"{COL_LABELS[col]}{row + 1}"


def parse_cell(label: str) -> Tuple[int, int]:
    """'E5' -> (col=4, row=4). Inverse of cell_label()."""
    s = (label or "").strip().upper()
    if len(s) < 2 or s[0] not in COL_LABELS or not s[1:].isdigit():
        raise ValueError(f"bad cell label: {label!r}")
    col = COL_LABELS.index(s[0])
    row = int(s[1:]) - 1
    if not 0 <= row < BLOCK_ROWS:
        raise ValueError(f"row out of range in {label!r}")
    return col, row


# ---------------------------------------------------------------------------
# Raw reads
# ---------------------------------------------------------------------------

def read_block_tile_ids(emu) -> List[List[int]]:
    """Return the 9x10 grid of representative tile ids, one per block.

    The player's standing tile in ``wTileMap`` is screen tile (col 8, row 9),
    so the 16px block grid is sampled with a +1 row offset: block (bc, br)
    maps to tilemap tile (bc*2, br*2 + GRID_ROW_OFFSET). That puts the player
    at block (4, 4) = cell E5 and lines the collision checks up with the
    engine's own movement rules.

    If GRID_ROW_OFFSET is wrong, every cell in the map is shifted and nothing
    downstream would notice — hence verify_offset().
    """
    tm = emu.read_range(ADDR_TILEMAP, TILEMAP_W * TILEMAP_H)
    grid: List[List[int]] = []
    for br in range(BLOCK_ROWS):
        row: List[int] = []
        for bc in range(BLOCK_COLS):
            tcol = bc * 2
            trow = br * 2 + GRID_ROW_OFFSET
            row.append(tm[trow * TILEMAP_W + tcol])
        grid.append(row)
    return grid


def read_sprite_cells(emu) -> List[Dict]:
    """Screen-relative block cells occupied by NPCs.

    Slot 0 is the player, so positions are computed as deltas from it and
    converted to block offsets from E5. Working in deltas means we do not
    depend on the absolute pixel bias of the sprite fields, which differs
    between the X and Y components.

    Only meaningful when the camera is settled — mid-step the deltas are
    half-block and the rounding is ambiguous.
    """
    raw = emu.read_range(ADDR_SPRITE_DATA1, SPRITE_SLOT_SIZE * SPRITE_SLOTS)
    if not raw[S_PICTURE_ID]:
        return []                      # player slot disabled: not in overworld
    py, px = raw[S_YPIXELS], raw[S_XPIXELS]

    out: List[Dict] = []
    for slot in range(1, SPRITE_SLOTS):
        base = slot * SPRITE_SLOT_SIZE
        if not raw[base + S_PICTURE_ID]:
            continue
        dy = ((raw[base + S_YPIXELS] - py + 128) % 256) - 128
        dx = ((raw[base + S_XPIXELS] - px + 128) % 256) - 128
        r = PLAYER_ROW + int(round(dy / BLOCK_PX))
        c = PLAYER_COL + int(round(dx / BLOCK_PX))
        if r == PLAYER_ROW and c == PLAYER_COL:
            continue                   # overlapping the player: mid-transition
        if 0 <= r < BLOCK_ROWS and 0 <= c < BLOCK_COLS:
            out.append({"slot": slot, "row": r, "col": c,
                        "cell": cell_label(c, r)})
    return out


def read_warp_cells(emu, player_x: int, player_y: int) -> List[Dict]:
    """Doors, stairs and map exits as screen-relative cells.

    Warp entries are stored in map coordinates. One overworld block is one
    map tile, so the conversion is a plain delta from the player's tile.
    Warps outside the visible 10x9 window are dropped.
    """
    n = emu.read_u8(ADDR_NUM_WARPS)
    if not 0 < n <= MAX_WARPS:
        return []
    raw = emu.read_range(ADDR_WARP_ENTRY, WARP_ENTRY_SIZE * n)
    out: List[Dict] = []
    for i in range(n):
        wy, wx, dest_warp, dest_map = raw[i * WARP_ENTRY_SIZE:
                                          i * WARP_ENTRY_SIZE + WARP_ENTRY_SIZE]
        r = PLAYER_ROW + (wy - player_y)
        c = PLAYER_COL + (wx - player_x)
        if 0 <= r < BLOCK_ROWS and 0 <= c < BLOCK_COLS:
            out.append({"row": r, "col": c, "cell": cell_label(c, r),
                        "map_x": wx, "map_y": wy,
                        "dest_map": dest_map, "dest_map_name": None,
                        "dest_warp": dest_warp})
    return out


def verify_offset(emu, facing: Optional[str],
                  tile_ids: List[List[int]]) -> Optional[bool]:
    """Cross-check our sampling against the engine's own front-tile value.

    ``wTileInFrontOfPlayer`` is what the engine itself tested to decide
    whether the last move was legal. If it disagrees with the tile we sampled
    for the cell adjacent to E5, GRID_ROW_OFFSET is wrong and every cell
    label is shifted.

    Returns None when the check cannot be made (no facing, or the adjacent
    cell is off-screen) — None means "unknown", not "fine".
    """
    d = _FACING_DELTA.get((facing or "").lower())
    if d is None:
        return None
    c, r = PLAYER_COL + d[0], PLAYER_ROW + d[1]
    if not (0 <= c < BLOCK_COLS and 0 <= r < BLOCK_ROWS):
        return None
    return emu.read_u8(ADDR_TILE_IN_FRONT) == tile_ids[r][c]


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------

def build_collision_grid(emu,
                         facing: Optional[str] = None,
                         player_pos: Optional[Dict] = None,
                         include_tile_ids: bool = False) -> Dict:
    """Build a walkability grid for the current on-screen blocks.

    Parameters
    ----------
    facing : str, optional
        Player facing ("up"/"down"/"left"/"right"), used for offset
        verification. From ``read_player()["facing"]``.
    player_pos : dict, optional
        ``{"x": int, "y": int}`` map coordinates, required to locate warps.
    include_tile_ids : bool
        Include the raw 90 tile ids. Off by default — they are noise in a
        prompt and in a WebSocket broadcast.

    Returns
    -------
    dict
        ``valid`` is the only field a caller should branch on. When False,
        the grid is garbage: a menu or text box is on screen, the camera is
        mid-scroll, or the tileset is unknown.
    """
    tileset = emu.read_u8(ADDR_TILESET)
    walk_set = TILESET_WALKABLE.get(tileset)
    known = walk_set is not None
    if walk_set is None:
        walk_set = frozenset()

    tile_ids = read_block_tile_ids(emu)
    settled = emu.read_u8(ADDR_WALK_COUNTER) == 0

    walkable = [[tile_ids[r][c] in walk_set for c in range(BLOCK_COLS)]
                for r in range(BLOCK_ROWS)]

    # Record what the sampler thought of the player's own cell BEFORE
    # overriding it. In normal overworld play this should be True; False is
    # the single best signal that GRID_ROW_OFFSET is wrong.
    player_tile_walkable = walkable[PLAYER_ROW][PLAYER_COL]
    walkable[PLAYER_ROW][PLAYER_COL] = True

    sprites = read_sprite_cells(emu) if settled else []
    warps: List[Dict] = []
    if player_pos and player_pos.get("x") is not None:
        try:
            warps = read_warp_cells(emu, int(player_pos["x"]), int(player_pos["y"]))
        except (TypeError, ValueError, KeyError):
            warps = []

    # Terrain walkability is kept separate from transient sprite blocking so
    # callers can distinguish "wall" from "someone is standing there".
    occupied = {(s["row"], s["col"]) for s in sprites}
    passable = [[walkable[r][c] and (r, c) not in occupied
                 for c in range(BLOCK_COLS)] for r in range(BLOCK_ROWS)]
    passable[PLAYER_ROW][PLAYER_COL] = True

    out: Dict = {
        "valid": known and settled,
        "tileset": tileset,
        "tileset_name": TILESET_NAMES.get(tileset, f"unknown({tileset})"),
        "tileset_known": known,
        "camera_settled": settled,
        "walkable": walkable,
        "passable": passable,
        "sprites": sprites,
        "warps": warps,
        "player_cell": cell_label(PLAYER_COL, PLAYER_ROW),
        "player_tile_walkable": player_tile_walkable,
        "offset_verified": verify_offset(emu, facing, tile_ids),
        "geometry": {"block_px": BLOCK_PX,
                     "player_px": [PLAYER_PX_X, PLAYER_PX_Y],
                     "cols": BLOCK_COLS, "rows": BLOCK_ROWS},
    }
    if player_pos:
        out["player_map_pos"] = {"x": player_pos.get("x"), "y": player_pos.get("y")}
    if include_tile_ids:
        out["tile_ids"] = tile_ids
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_ascii_map(collision: Dict, legend: bool = True) -> str:
    """Render the collision grid as a labelled ASCII map.

    Refuses to draw when the grid is not valid — a plausible-looking map of a
    menu is worse than an explicit "unavailable", because an agent will act
    on the former.
    """
    if not collision.get("valid", True):
        reasons: List[str] = []
        if not collision.get("tileset_known", True):
            reasons.append(f"unknown tileset {collision.get('tileset')}")
        if not collision.get("camera_settled", True):
            reasons.append("camera mid-scroll (mid-step)")
        if not reasons:
            reasons.append(collision.get("reason", "invalid"))
        return "(map unavailable: " + ", ".join(reasons) + ")"

    walkable = collision["walkable"]
    npc = {(s["row"], s["col"]) for s in collision.get("sprites") or []}
    warp = {(w["row"], w["col"]) for w in collision.get("warps") or []}

    lines: List[str] = ["   " + " ".join(COL_LABELS)]
    for r in range(BLOCK_ROWS):
        cells: List[str] = []
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
        lines.append("walk_up=row-1  walk_down=row+1  "
                     "walk_left=col-1  walk_right=col+1")
        exits = collision.get("warps") or []
        if exits:
            lines.append("exits: " + ", ".join(
                f"{w['cell']} -> {w.get('dest_map_name') or 'map ' + str(w['dest_map'])}"
                for w in exits))
        if collision.get("offset_verified") is False:
            lines.append("WARNING: grid alignment check FAILED — "
                         "cell labels may be shifted.")
    return "\n".join(lines)