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
        party.append(PartyMon(
            level=raw[PM_LEVEL],
            hp=int.from_bytes(raw[PM_HP:PM_HP + 2], "big"),
            max_hp=int.from_bytes(raw[PM_MAX_HP:PM_MAX_HP + 2], "big"),
            moves=[m for m in raw[PM_MOVES:PM_MOVES + 4] if m],
        ))
    return GameState(
        map_id=map_id, x=x, y=y, in_battle=in_battle, badges=badges,
        money=money, party=party,
        mid_step=read(ADDR_WALK_COUNTER, 1)[0] != 0,
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
