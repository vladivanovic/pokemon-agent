import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.state import (ADDR_BADGES, ADDR_CUR_MAP, ADDR_MAP_X, ADDR_MAP_Y,
                         ADDR_MONEY, ADDR_PARTY_COUNT, ADDR_PARTY_MONS,
                         PARTY_MON_SIZE, PM_HP, PM_LEVEL, PM_MAX_HP, PM_MOVES,
                         FakeMemory, read_state)


def blank_memory():
    mem = FakeMemory()
    mem.write_u8(ADDR_CUR_MAP, 0)
    mem.write_u8(ADDR_MAP_X, 7)
    mem.write_u8(ADDR_MAP_Y, 3)
    return mem


class TestState(unittest.TestCase):
    def test_position_and_badges(self):
        mem = blank_memory()
        mem.write_u8(ADDR_BADGES, 0b00000101)  # Boulder + Thunder
        st = read_state(mem.read)
        self.assertEqual(st.pos, (0, 7, 3))
        self.assertEqual(st.badges, frozenset({"Boulder", "Thunder"}))

    def test_money_is_bcd(self):
        mem = blank_memory()
        mem.write(ADDR_MONEY, bytes([0x12, 0x34, 0x56]))
        self.assertEqual(read_state(mem.read).money, 123456)

    def test_party_and_capabilities(self):
        mem = blank_memory()
        mem.write_u8(ADDR_BADGES, 0b00010000)  # Soul badge
        mem.write_u8(ADDR_PARTY_COUNT, 1)
        base = ADDR_PARTY_MONS
        mem.write(base + PM_HP, (15).to_bytes(2, "big"))
        mem.write(base + PM_MAX_HP, (20).to_bytes(2, "big"))
        mem.write_u8(base + PM_LEVEL, 10)
        mem.write_u8(base + PM_MOVES, 57)  # Surf
        st = read_state(mem.read)
        self.assertEqual(len(st.party), 1)
        self.assertEqual(st.party[0].level, 10)
        self.assertAlmostEqual(st.party[0].hp_frac, 0.75)
        self.assertIn("surf", st.capabilities())

    def test_capability_needs_badge(self):
        mem = blank_memory()  # no Soul badge
        mem.write_u8(ADDR_PARTY_COUNT, 1)
        mem.write_u8(ADDR_PARTY_MONS + PM_MOVES, 57)
        st = read_state(mem.read)
        self.assertNotIn("surf", st.capabilities())


if __name__ == "__main__":
    unittest.main()
