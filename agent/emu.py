"""Emulator interface + a simulator that runs on the real world data.

`Emulator` is the seam where PyBoy plugs in on the Orin: three verbs
(press, read_state via a memory reader, save/load state). Everything else
in this package is written against it.

`SimEmulator` implements the same interface against the static World:
movement is checked against the real ROM walkability grids, warps fire
from the real warp tables, and map connections carry you across edges.
It exists so the full loop (planner -> validator -> executor) can be
exercised in tests with no ROM and no emulator.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Set, Tuple

from .state import (ADDR_BADGES, ADDR_CUR_MAP, ADDR_IS_IN_BATTLE,
                    ADDR_MAP_X, ADDR_MAP_Y, ADDR_MONEY, ADDR_PARTY_COUNT,
                    ADDR_PARTY_MONS, BADGE_NAMES, PARTY_MON_SIZE, PM_HP,
                    PM_LEVEL, PM_MAX_HP, PM_MOVES, FakeMemory, GameState,
                    read_state)
from .world import DIRS, World

BUTTONS = ("up", "down", "left", "right", "a", "b", "start", "select")
_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
_DIR_NAME = {v: k for k, v in DIRS.items()}


class SimEmulator:
    def __init__(self, world: World, start_map: int, start_xy: Tuple[int, int]):
        self.world = world
        self.mem = FakeMemory()
        self.map_id = start_map
        self.x, self.y = start_xy
        self.in_battle = False
        # wLastMap: LAST_MAP exit warps return to the last OUTSIDE map;
        # set when leaving an outside map, never updated by indoor maps.
        self.last_outside: Optional[int] = world.default_last_outside(start_map)
        self._battle_presses_left = 0
        self.blocked: Set[Tuple[int, int, int]] = set()  # (map, x, y) test hooks
        self.press_count = 0
        # Field-move (Cut) session state, mirroring the engine: a
        # direction press turns the player even when the step is
        # refused (that facing is what Cut reads), and a cut tree
        # block stays cleared only until its map is left (the engine
        # reloads the map's blocks on entry).
        self.facing: Tuple[int, int] = (0, 1)
        self.cut_blocks: Set[Tuple[int, Tuple[int, int]]] = set()
        self._sync()

    def _leave_map(self, old_map: int) -> None:
        """Crossing into another map restores the old map's trees."""
        self.cut_blocks = {kb for kb in self.cut_blocks if kb[0] != old_map}

    # -- state plumbing -------------------------------------------------------
    def _sync(self) -> None:
        m = self.mem
        m.write_u8(ADDR_CUR_MAP, self.map_id)
        m.write_u8(ADDR_MAP_X, self.x)
        m.write_u8(ADDR_MAP_Y, self.y)
        m.write_u8(ADDR_IS_IN_BATTLE, 1 if self.in_battle else 0)

    def read_state(self) -> GameState:
        return read_state(self.mem.read)

    def give_starter(self, level: int = 5, moves=(33, 45)) -> None:
        """Add one Pokemon so party-dependent logic has something to see.
        Default moves are Scratch (33) and Growl (45) - no HMs."""
        m = self.mem
        m.write_u8(ADDR_PARTY_COUNT, 1)
        base = ADDR_PARTY_MONS
        m.write(base + PM_HP, (20).to_bytes(2, "big"))
        m.write(base + PM_MAX_HP, (20).to_bytes(2, "big"))
        m.write_u8(base + PM_LEVEL, level)
        for i, mv in enumerate(moves):
            m.write_u8(base + PM_MOVES + i, mv)

    def give_badge(self, name: str) -> None:
        idx = BADGE_NAMES.index(name)
        cur = self.mem.read(ADDR_BADGES, 1)[0]
        self.mem.write_u8(ADDR_BADGES, cur | (1 << idx))

    def give_money(self, amount: int) -> None:
        digits = f"{amount:06d}"
        raw = bytes(
            (int(digits[i]) << 4) | int(digits[i + 1]) for i in (0, 2, 4)
        )
        self.mem.write(ADDR_MONEY, raw)

    def force_battle(self, presses_to_win: int = 3) -> None:
        self.in_battle = True
        self._battle_presses_left = presses_to_win
        self._sync()

    # -- verbs ------------------------------------------------------------------
    def press(self, button: str) -> None:
        assert button in BUTTONS, button
        self.press_count += 1
        if self.in_battle:
            if button == "a" and self._battle_presses_left > 0:
                self._battle_presses_left -= 1
                if self._battle_presses_left == 0:
                    self.in_battle = False
            self._sync()
            return
        if button not in _DELTA:
            return
        dx, dy = _DELTA[button]
        self.facing = (dx, dy)
        self._step(dx, dy)
        self._sync()

    def field_cut(self, mon_index: int, submenu_index: int) -> bool:
        """Use Cut on the tree the player is facing (the Executor has
        already bumped toward it, which set `facing`). Succeeds only
        if the named party mon actually knows Cut and the faced tile
        is a standing tree's FACE cell - the engine checks the tile
        in front, not the whole block."""
        st = self.read_state()
        if not (0 <= mon_index < len(st.party)):
            return False
        if 15 not in st.party[mon_index].moves:  # MOVE_CUT
            return False
        gm = self.world.maps[self.map_id]
        tx, ty = self.x + self.facing[0], self.y + self.facing[1]
        if (tx, ty) not in gm.cut_trees:
            return False
        if (self.map_id, (tx, ty)) in self.cut_blocks:
            return False  # already cut: the engine finds nothing to cut
        self.cut_blocks.add((self.map_id, (tx, ty)))
        return True

    def _step(self, dx: int, dy: int) -> None:
        gm = self.world.maps[self.map_id]
        nx, ny = self.x + dx, self.y + dy
        if not gm.in_bounds(nx, ny):
            self._cross_edge(dx, dy)
            return
        if (self.map_id, nx, ny) in self.blocked:
            return
        face = gm.cut_blocks.get((nx, ny))
        if face is not None and not gm.walkable(nx, ny, dx, dy) \
                and (self.map_id, face) not in self.cut_blocks:
            return  # a standing tree's solid cells (a block cell
            # whose own tile is passable ground stays walkable,
            # exactly as the engine's per-tile collision has it)
        if face is None and not gm.walkable(nx, ny, dx, dy):
            # Ledge jump: stepping into a face tile in its jump
            # direction hops onto the tile beyond it (the engine
            # simulates the second step itself on hardware).
            if gm.ledges.get((nx, ny)) != (dx, dy):
                return
            bx, by = nx + dx, ny + dy
            if not gm.in_bounds(bx, by) \
                    or not gm.walkable(bx, by, dx, dy) \
                    or (self.map_id, bx, by) in self.blocked:
                return
            nx, ny = bx, by
        self.x, self.y = nx, ny
        # Warps fire when stepped on.
        for wp in gm.warps:
            if wp.x == nx and wp.y == ny:
                if wp.is_exit:
                    lo = self.last_outside
                    if lo is not None and 1 <= wp.idx <= len(self.world.maps[lo].warps):
                        lw = self.world.maps[lo].warps[wp.idx - 1]
                        old = self.map_id
                        self.map_id, self.x, self.y = lo, lw.x, lw.y
                        self._leave_map(old)
                    return
                if self.world.is_outside(self.map_id):
                    self.last_outside = self.map_id
                landing = self.world.warp_landing(wp)
                if landing is not None:
                    old = self.map_id
                    self.map_id = wp.dest_id
                    self.x, self.y = landing
                    self._leave_map(old)
                return

    def _cross_edge(self, dx: int, dy: int) -> None:
        direction = _DIR_NAME[(dx, dy)]
        for d, dst, offset in self.world.connections.get(self.map_id, []):
            if d != direction:
                continue
            gm = self.world.maps[dst]
            # Header geometry: the offset is in BLOCKS (2 tiles);
            # entering at (exit axis) - 2*offset. Verified against the
            # rendered map: Route 1 (offset -5) -> Viridian's entrance
            # corridor at x=20-21, not the sealed tree pocket at x=15.
            # Fall back to the nearest walkable edge tile if needed.
            if dx == 0:
                ey = gm.h - 1 if dy < 0 else 0
                ideal = self.x - 2 * offset
                candidates = sorted(range(gm.w), key=lambda cx: abs(cx - ideal))
                for cx in candidates:
                    if gm.walkable(cx, ey, dx, dy):
                        old = self.map_id
                        self.map_id, self.x, self.y = dst, cx, ey
                        self._leave_map(old)
                        return
            else:
                ex = gm.w - 1 if dx < 0 else 0
                ideal = self.y - 2 * offset
                candidates = sorted(range(gm.h), key=lambda cy: abs(cy - ideal))
                for cy in candidates:
                    if gm.walkable(ex, cy, dx, dy):
                        old = self.map_id
                        self.map_id, self.x, self.y = dst, ex, cy
                        self._leave_map(old)
                        return
            return

    def fly_to(self, dest_map: int) -> None:
        """Instant travel used by the executor for fly legs (a real backend
        scripts the menu; the simulator just relocates)."""
        gm = self.world.maps[dest_map]
        # Land on the first warp tile (town entrance) or map centre.
        if gm.warps:
            self.x, self.y = gm.warps[0].x, gm.warps[0].y
        else:
            self.x, self.y = gm.w // 2, gm.h // 2
        old = self.map_id
        self.map_id = dest_map
        self.last_outside = dest_map
        self._leave_map(old)
        self._sync()

    # -- save states --------------------------------------------------------------
    def save_state(self) -> dict:
        return {"map": self.map_id, "x": self.x, "y": self.y,
                "mem": bytes(self.mem.mem),
                "cut_blocks": set(self.cut_blocks),
                "facing": self.facing}

    def load_state(self, snap: dict) -> None:
        self.map_id, self.x, self.y = snap["map"], snap["x"], snap["y"]
        self.mem.mem = bytearray(snap["mem"])
        self.in_battle = False
        self.cut_blocks = set(snap.get("cut_blocks", set()))
        self.facing = snap.get("facing", (0, 1))
        self._sync()
