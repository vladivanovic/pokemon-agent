"""Grid + annotation overlay for agent vision.

Pokemon Gen 1 (Red/Blue/Yellow) renders the overworld on a 160x144 px Game
Boy screen. The walkable world is a grid of 16x16 px blocks (each block is a
2x2 arrangement of 8x8 hardware tiles). That yields a 10x9 block grid:

    columns: 160 / 16 = 10   (labelled A..J left -> right)
    rows:    144 / 16 = 9    (labelled 1..9 top -> bottom)

The player sprite is locked to a fixed on-screen block while the map scrolls
underneath: column index 4 (E), row index 4 (5). Cell "E5" is ALWAYS the
player.

This module draws that grid over a screenshot and labels each cell so a
vision model can reason in discrete, nameable steps: "the door is at C3, I'm
at E5, so I walk up 2 and left 2."

Geometry is imported from collision.py and never redefined here. The block
grid is sampled from the tilemap with a +1 row offset (see
collision.read_block_tile_ids), so the drawn grid must be shifted down by
half a block to sit over the same pixels the ASCII map describes. If the two
disagree, an agent gets a picture and a map that contradict each other and no
way to tell which is lying.
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from pokemon_agent.collision import (
    BLOCK_COLS as COLS,
    BLOCK_ROWS as ROWS,
    BLOCK_PX as BLOCK,
    PLAYER_COL,
    PLAYER_ROW,
    PLAYER_PX_X,
    PLAYER_PX_Y,
    cell_label,
)

logger = logging.getLogger("pokemon-agent.overlay")

GB_W, GB_H = 160, 144

# Vertical shift between the naive block grid (row * 16) and where the blocks
# actually sit on screen. Derived from the shared constants so it cannot drift.
GRID_Y_OFFSET = PLAYER_PX_Y - PLAYER_ROW * BLOCK   # 8 px
GRID_X_OFFSET = PLAYER_PX_X - PLAYER_COL * BLOCK   # 0 px

# DMG-flavoured overlay colours (RGBA)
GRID_LINE  = (139, 172, 15, 150)       # #8BAC0F semi-transparent
LABEL_BG   = (15, 19, 15, 170)
LABEL_FG   = (232, 228, 214, 255)
PLAYER_BOX = (217, 72, 47, 235)        # vermilion signal colour
WALK_WASH  = (139, 172, 15, 38)        # faint DMG-green over walkable cells
BLOCK_WASH = (217, 72, 47, 70)         # translucent red over blocked cells
NPC_WASH   = (201, 162, 39, 90)        # amber: someone is standing there
WARP_WASH  = (79, 123, 214, 90)        # blue: door / stairs / exit


@lru_cache(maxsize=16)
def _load_font(size: int):
    """Load a bold monospace font, cached.

    Cached because this is called once per render and each miss costs three
    failed file opens. Falls back to PIL's bitmap font, which ignores `size`
    before Pillow 10.1 — labels will be ~11px regardless on older Pillow.
    """
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        r"C:\Windows\Fonts\consolab.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        logger.debug("no TrueType font found; labels will be small")
        return ImageFont.load_default()


def _cell_box(col: int, row: int, cell: int, height: int):
    """Pixel box for a grid cell, clipped to the canvas."""
    x0 = col * cell + GRID_X_OFFSET * (cell // BLOCK)
    y0 = row * cell + GRID_Y_OFFSET * (cell // BLOCK)
    return x0, y0, x0 + cell - 1, min(y0 + cell - 1, height - 1)


def render_grid_overlay(
    screen: Image.Image,
    scale: int = 4,
    show_labels: bool = True,
    mark_player: bool = True,
    walkable: Optional[List[List[bool]]] = None,
    collision: Optional[Dict] = None,
    label_mode: str = "all",
) -> Image.Image:
    """Draw a labelled 10x9 movement grid over a GB screenshot.

    Parameters
    ----------
    screen : PIL.Image
        The raw 160x144 emulator frame.
    scale : int
        Integer upscale factor (nearest-neighbour), 1-8.
    show_labels : bool
        Draw the A1..J9 cell labels.
    mark_player : bool
        Outline the fixed player cell (E5) in vermilion.
    walkable : list, optional
        A 9x10 grid of bool. Blocked cells get a red wash, walkable cells a
        faint green one.
    collision : dict, optional
        A full grid from ``collision.build_collision_grid``. Supplies
        ``walkable`` plus NPC and warp markers, so the picture agrees with
        the ASCII map. Ignored when not valid. Takes precedence over
        *walkable*.
    label_mode : str
        ``"all"`` labels every cell; ``"edges"`` labels only the top row and
        left column, which occludes far less of the art. Use ``"edges"`` at
        low scale factors where 90 labels become unreadable anyway.

    Returns
    -------
    PIL.Image
        The annotated, upscaled RGBA image.
    """
    if not isinstance(scale, int) or not 1 <= scale <= 8:
        raise ValueError(f"scale must be an int in 1..8, got {scale!r}")

    npc_cells: set = set()
    warp_cells: set = set()
    if collision is not None:
        if collision.get("valid"):
            walkable = collision.get("walkable") or walkable
            npc_cells = {(s["row"], s["col"]) for s in collision.get("sprites") or []}
            warp_cells = {(w["row"], w["col"]) for w in collision.get("warps") or []}
        else:
            # Never tint from a grid we cannot vouch for — a red-washed menu
            # reads as "you are trapped".
            walkable = None

    if screen.mode != "RGBA":
        screen = screen.convert("RGBA")
    if screen.size != (GB_W, GB_H):
        logger.warning("unexpected frame size %s, resizing to %dx%d",
                       screen.size, GB_W, GB_H)
        screen = screen.resize((GB_W, GB_H), Image.NEAREST)

    big = screen.resize((GB_W * scale, GB_H * scale), Image.NEAREST)
    overlay = Image.new("RGBA", big.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cell = BLOCK * scale
    font = _load_font(max(9, cell // 3))
    height, width = big.height, big.width

    # --- washes (under everything else) ---
    for r in range(ROWS):
        for c in range(COLS):
            if r == PLAYER_ROW and c == PLAYER_COL:
                continue
            wash = None
            if (r, c) in warp_cells:
                wash = WARP_WASH
            elif (r, c) in npc_cells:
                wash = NPC_WASH
            elif walkable is not None and r < len(walkable) and c < len(walkable[r]):
                wash = WALK_WASH if walkable[r][c] else BLOCK_WASH
            if wash is not None:
                draw.rectangle(_cell_box(c, r, cell, height), fill=wash)

    # --- player cell ---
    if mark_player:
        draw.rectangle(_cell_box(PLAYER_COL, PLAYER_ROW, cell, height),
                       outline=PLAYER_BOX, width=max(2, scale))

    # --- grid lines ---
    y_off = GRID_Y_OFFSET * scale
    x_off = GRID_X_OFFSET * scale
    for c in range(COLS + 1):
        x = c * cell + x_off
        if x <= width:
            draw.line([(x, 0), (x, height)], fill=GRID_LINE, width=1)
    for r in range(ROWS + 2):          # +2 so the shifted grid reaches the edge
        y = r * cell + y_off
        if y <= height:
            draw.line([(0, y), (width, y)], fill=GRID_LINE, width=1)

    # --- labels ---
    if show_labels:
        for r in range(ROWS):
            for c in range(COLS):
                if label_mode == "edges" and r != 0 and c != 0:
                    continue
                x0, y0, _, _ = _cell_box(c, r, cell, height)
                lx, ly = x0 + 2, y0 + 1
                if ly > height - 6:
                    continue
                label = cell_label(c, r)
                tb = draw.textbbox((lx, ly), label, font=font)
                draw.rectangle([tb[0] - 1, tb[1] - 1, tb[2] + 1, tb[3] + 1],
                               fill=LABEL_BG)
                draw.text((lx, ly), label, fill=LABEL_FG, font=font)

    return Image.alpha_composite(big, overlay)


def render_grid_overlay_bytes(screen: Image.Image, **kwargs) -> bytes:
    """As render_grid_overlay, returning PNG bytes."""
    img = render_grid_overlay(screen, **kwargs)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def player_cell() -> str:
    """The grid cell the player always occupies (E5)."""
    return cell_label(PLAYER_COL, PLAYER_ROW)