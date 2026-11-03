"""Game state read from emulator RAM.

Addresses are WRAM locations from the pret/pokered disassembly. The ones
marked VERIFIED were confirmed against live RAM by the previous project
(see legacy/pokemon_agent/collision.py, including its note that
wNumberOfWarps is 0xD3AE - an earlier "off-by-one fix" was itself wrong).
The party-structure base is a named constant so it can be recalibrated in
one place on real hardware.

Nothing here *infers* state from pixels: if a value matters, it is read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, FrozenSet, List, Optional

# --- WRAM addresses ---------------------------------------------------------
ADDR_CUR_MAP       = 0xD35E   # wCurMap
ADDR_MAP_Y         = 0xD361   # wYCoord            VERIFIED (legacy collision.py)
ADDR_MAP_X         = 0xD362   # wXCoord            VERIFIED (legacy collision.py)
ADDR_WALK_COUNTER  = 0xCFC5   # wWalkCounter; nonzero = mid-step
ADDR_TILEMAP       = 0xC3A0   # wTileMap: 20x18 screen tilemap
TEXTBOX_CORNER_TILE = 0x79    # top-left corner of the standard text box,
                              # found at tilemap row 12 col 0 while a box
                              # is open (calibrated live 2026-11-03)
ADDR_TILE_IN_FRONT = 0xCFC6   # wTileInFrontOfPlayer
ADDR_IS_IN_BATTLE  = 0xD057   # wIsInBattle; 0 = overworld, >0 = battle type
ADDR_BADGES        = 0xD356   # wObtainedBadges bitfield
ADDR_MONEY         = 0xD347   # wMoney, 3-byte BCD
ADDR_PARTY_COUNT   = 0xD163   # wPartyCount
ADDR_PARTY_MONS    = 0xD16B   # wPartyMons: 44-byte party_mon structs
ADDR_EVENT_FLAGS   = 0xD747   # wEventFlags bit array

PARTY_MON_SIZE = 44
PM_HP = 1        # big-endian u16 current HP
PM_LEVEL = 33    # u8 level (party struct only)
PM_MAX_HP = 34   # big-endian u16 max HP
PM_MOVES = 8     # 4 x u8 move ids
PM_PP = 29       # 4 x u8 PP (top 2 bits are PP Ups; mask 0x3F)

# Battle-menu cursor state (addresses from the pokered .sym file).
# wTopMenuItemY separates the battle menus: 14 = main (FIGHT/PKMN/
# ITEM/RUN), 12 = move selection. wCurrentMenuItem is the highlighted
# item (0-based in the main menu, 1-based in the move menu, where it
# is initialised from wPlayerMoveListIndex + 1 - the engine remembers
# the last move you picked).
ADDR_MENU_TOP_Y        = 0xCC24   # wTopMenuItemY
ADDR_CURRENT_MENU_ITEM = 0xCC26   # wCurrentMenuItem
ADDR_MOVE_LIST_INDEX   = 0xCC2E   # wPlayerMoveListIndex

BADGE_NAMES = ["Boulder", "Cascade", "Thunder", "Rainbow",
               "Soul", "Marsh", "Volcano", "Earth"]

# HM move id -> (capability name, badge required to use it in the field)
HM_MOVES = {
    15:  ("cut", "Cascade"),
    19:  ("fly", "Thunder"),
    57:  ("surf", "Soul"),
    70:  ("strength", "Rainbow"),
    148: ("flash", "Boulder"),
}

Reader = Callable[[int, int], bytes]  # (address, length) -> bytes


@dataclass
class PartyMon:
    level: int
    hp: int
    max_hp: int
    moves: List[int] = field(default_factory=list)
    # Current PP per move, aligned with `moves` (moves are packed from
    # slot 0, so the first len(moves) PP slots correspond).
    pp: List[int] = field(default_factory=list)
    # Species index (party_mon struct byte 0). Gen 1 internal order,
    # NOT National Dex order; 0 = unset (e.g. the simulator's starter).
    species_id: int = 0

    @property
    def hp_frac(self) -> float:
        return self.hp / self.max_hp if self.max_hp else 0.0


@dataclass
class GameState:
    map_id: int
    x: int
    y: int
    in_battle: bool
    badges: FrozenSet[str]
    money: int
    party: List[PartyMon]
    mid_step: bool = False
    in_dialog: Optional[bool] = None  # backend may supply; None = unknown
    # Battle-menu cursor state (see the ADDR_MENU_* constants); zero on
    # backends that do not read them, which policies must treat as
    # "no menu information".
    menu_top_y: int = 0
    menu_item: int = 0
    move_list_index: int = 0

    @property
    def pos(self):
        return (self.map_id, self.x, self.y)

    def capabilities(self) -> FrozenSet[str]:
        """Field capabilities (HM use) from party moves + badges."""
        caps = set()
        for mon in self.party:
            for mv in mon.moves:
                if mv in HM_MOVES:
                    cap, badge = HM_MOVES[mv]
                    if badge in self.badges:
                        caps.add(cap)
        return frozenset(caps)

    def team_hp_frac(self) -> float:
        if not self.party:
            return 0.0
        return sum(m.hp_frac for m in self.party) / len(self.party)

    def digest(self) -> dict:
        """Compact JSON-able summary - this is what a planner sees."""
        return {
            "map": self.map_id,
            "x": self.x,
            "y": self.y,
            "in_battle": self.in_battle,
            "badges": sorted(self.badges),
            "money": self.money,
            "party": [
                {"level": m.level, "hp_frac": round(m.hp_frac, 2)}
                for m in self.party
            ],
            "capabilities": sorted(self.capabilities()),
        }


def _bcd3(b: bytes) -> int:
    return ((b[0] >> 4) * 100000 + (b[0] & 0xF) * 10000 +
            (b[1] >> 4) * 1000 + (b[1] & 0xF) * 100 +
            (b[2] >> 4) * 10 + (b[2] & 0xF))


def read_state(read: Reader) -> GameState:
    map_id = read(ADDR_CUR_MAP, 1)[0]
    y = read(ADDR_MAP_Y, 1)[0]
    x = read(ADDR_MAP_X, 1)[0]
    in_battle = read(ADDR_IS_IN_BATTLE, 1)[0] != 0
    badge_byte = read(ADDR_BADGES, 1)[0]
    badges = frozenset(
        name for i, name in enumerate(BADGE_NAMES) if badge_byte & (1 << i)
    )
    money = _bcd3(read(ADDR_MONEY, 3))
    count = min(read(ADDR_PARTY_COUNT, 1)[0], 6)
    party: List[PartyMon] = []
    for i in range(count):
        base = ADDR_PARTY_MONS + i * PARTY_MON_SIZE
        raw = read(base, PARTY_MON_SIZE)
        moves = [m for m in raw[PM_MOVES:PM_MOVES + 4] if m]
        party.append(PartyMon(
            level=raw[PM_LEVEL],
            hp=int.from_bytes(raw[PM_HP:PM_HP + 2], "big"),
            max_hp=int.from_bytes(raw[PM_MAX_HP:PM_MAX_HP + 2], "big"),
            moves=moves,
            pp=[raw[PM_PP + i] & 0x3F for i in range(len(moves))],
            species_id=raw[0],
        ))
    # Dialog detection, calibrated against the live game: a standard
    # text box puts corner tile 0x79 at tilemap row 12, col 0; when the
    # box closes, the map tiles return. (Blind A-mashing is not a
    # substitute - in front of an examinable object like the bedroom
    # SNES, the press after the box closes just reopens it.)
    in_dialog = read(ADDR_TILEMAP + 12 * 20, 1)[0] == TEXTBOX_CORNER_TILE
    return GameState(
        map_id=map_id, x=x, y=y, in_battle=in_battle, badges=badges,
        money=money, party=party,
        mid_step=read(ADDR_WALK_COUNTER, 1)[0] != 0,
        in_dialog=in_dialog,
        menu_top_y=read(ADDR_MENU_TOP_Y, 1)[0],
        menu_item=read(ADDR_CURRENT_MENU_ITEM, 1)[0],
        move_list_index=read(ADDR_MOVE_LIST_INDEX, 1)[0],
    )


def read_event_flag(read: Reader, index: int) -> bool:
    byte = read(ADDR_EVENT_FLAGS + index // 8, 1)[0]
    return bool(byte & (1 << (index % 8)))


class FakeMemory:
    """Dict-backed RAM for tests and the simulator."""

    def __init__(self, size: int = 0x10000):
        self.mem = bytearray(size)

    def read(self, address: int, length: int) -> bytes:
        return bytes(self.mem[address:address + length])

    def write(self, address: int, data: bytes) -> None:
        self.mem[address:address + len(data)] = data

    def write_u8(self, address: int, value: int) -> None:
        self.mem[address] = value & 0xFF
