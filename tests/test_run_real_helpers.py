"""Unit tests for run_real.py's RAM counter helpers.

The catch/flash scenes verify themselves from WRAM counters
(dex owned, box count, stored count); those readers are pure
functions of memory + the party list, so they are testable with
a fake emulator - no ROM, no PyBoy window. The battle machinery
they feed is live-verified in the probes instead (probe143).
Skipped when PyBoy is not installed: run_real imports it at
module load, and the rest of the suite must stay runnable in a
bare checkout (pyproject's pyboy extra).
"""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if importlib.util.find_spec("pyboy") is None:
    raise unittest.SkipTest("pyboy not installed")

import run_real as rr


class FakeEmu:
    def __init__(self, party_size=0, box=0, owned_dex=()):
        self.memory = bytearray(0x10000)
        self.pb = SimpleNamespace(memory=self.memory)
        self._party = [object()] * party_size
        self.memory[rr.ADDR_BOX_COUNT] = box
        for dex_no in owned_dex:
            i = dex_no - 1
            self.memory[rr.ADDR_POKEDEX_OWNED + i // 8] |= 1 << (i % 8)

    def read_state(self):
        return SimpleNamespace(party=list(self._party))


class TestCounters(unittest.TestCase):
    def test_dex_owned_count_empty(self):
        self.assertEqual(rr._dex_owned_count(FakeEmu()), 0)

    def test_dex_owned_count_matches_probe142_baseline(self):
        # The chain head's real dex: Squirtle/Wartortle/Oddish/
        # Diglett = dex 7, 8, 43, 50 (probe142, from live RAM).
        emu = FakeEmu(owned_dex=(7, 8, 43, 50))
        self.assertEqual(rr._dex_owned_count(emu), 4)

    def test_dex_owned_count_full(self):
        emu = FakeEmu(owned_dex=range(1, 152))
        self.assertEqual(rr._dex_owned_count(emu), 151)

    def test_box_count(self):
        self.assertEqual(rr._box_count(FakeEmu(box=7)), 7)

    def test_stored_count_party_plus_box(self):
        # The box-catch case: party full at 6, catches land in
        # the box, and only the stored count keeps moving.
        self.assertEqual(rr._stored_count(FakeEmu(3, 0)), 3)
        self.assertEqual(rr._stored_count(FakeEmu(6, 0)), 6)
        self.assertEqual(rr._stored_count(FakeEmu(6, 3)), 9)

    def test_hm05_item_id(self):
        # HM pocket ids are 0xC4 + (n - 1); HM01 is pinned by the
        # Vermilion leg, so HM05 must sit four slots later.
        self.assertEqual(rr.ITEM_HM05, rr.ITEM_HM01 + 4)


class RankerEmu:
    """Fake emu whose party is (moves, pp) tuples."""

    def __init__(self, mons):
        self._mons = [SimpleNamespace(moves=list(m), pp=list(p))
                      for m, p in mons]

    def read_state(self):
        return SimpleNamespace(party=list(self._mons))


class TestRankedMoveFn(unittest.TestCase):
    """The PP-dry fallback (probe145's Exeggcute wedge): the
    ranker must never hand the driver a 0-PP move."""

    def test_prefers_strongest_damaging_with_pp(self):
        # Water Gun 55 (40) dry, Bubblebeam 61 (65) live.
        emu = RankerEmu([([55, 61], [0, 20])])
        self.assertEqual(rr._ranked_move_fn(emu)(0), [1])

    def test_fallback_is_pp_status_move_not_slot0(self):
        # Both damaging moves dry; Tail Whip (39, status)
        # still has PP - the old `or [0]` returned the dry
        # Water Gun and the battle wedged on "no PP" text.
        emu = RankerEmu([([55, 39, 61], [0, 30, 0])])
        self.assertEqual(rr._ranked_move_fn(emu)(0), [1])

    def test_all_dry_returns_slot0(self):
        # Nothing has PP at all: the game force-Struggles,
        # so the value is moot, but it must not crash.
        emu = RankerEmu([([55, 39], [0, 0])])
        self.assertEqual(rr._ranked_move_fn(emu)(0), [0])


class TypedRankerEmu:
    """Fake emu for the typed ranker: party (moves, pp) plus
    the four type bytes the ranker reads from WRAM."""

    def __init__(self, mons, enemy_types=(0, 0), mon_types=(0, 0)):
        self._mons = [SimpleNamespace(moves=list(m), pp=list(p))
                      for m, p in mons]
        self.memory = bytearray(0x10000)
        self.pb = SimpleNamespace(memory=self.memory)
        self.memory[rr.ADDR_ENEMY_TYPE1] = enemy_types[0]
        self.memory[rr.ADDR_ENEMY_TYPE2] = enemy_types[1]
        self.memory[rr.ADDR_BATTLE_MON_TYPE1] = mon_types[0]
        self.memory[rr.ADDR_BATTLE_MON_TYPE2] = mon_types[1]

    def read_state(self):
        return SimpleNamespace(party=list(self._mons))


FIRE, WATER, GRASS, POISON = 20, 21, 22, 3
BITE, ROAR, EMBER = 44, 46, 52


class TestTypedMoveFn(unittest.TestCase):
    """The Erika matchup in miniature (probe145/147): raw
    power says Bite (60) > Ember (40); effective power into
    Grass says the opposite, and the fight turns on it."""

    def test_ember_beats_bite_into_grass(self):
        emu = TypedRankerEmu([([BITE, ROAR, EMBER], [25, 20, 25])],
                             enemy_types=(GRASS, POISON),
                             mon_types=(FIRE, 0))
        # Ember 40 x2 (SE) x1.5 (STAB) = 120 > Bite 60;
        # Roar is a status move and never ranks.
        self.assertEqual(rr._typed_move_fn(emu)(0), [2, 0])

    def test_bite_beats_ember_into_water(self):
        emu = TypedRankerEmu([([BITE, ROAR, EMBER], [25, 20, 25])],
                             enemy_types=(WATER, 0),
                             mon_types=(FIRE, 0))
        # Ember 40 x0.5 x1.5 = 30 < Bite 60.
        self.assertEqual(rr._typed_move_fn(emu)(0), [0, 2])

    def test_dry_best_move_is_skipped(self):
        emu = TypedRankerEmu([([BITE, ROAR, EMBER], [25, 20, 0])],
                             enemy_types=(GRASS, POISON),
                             mon_types=(FIRE, 0))
        self.assertEqual(rr._typed_move_fn(emu)(0), [0])

    def test_all_damaging_dry_falls_back_to_pp_move(self):
        emu = TypedRankerEmu([([BITE, ROAR], [0, 5])],
                             enemy_types=(GRASS, 0),
                             mon_types=(FIRE, 0))
        self.assertEqual(rr._typed_move_fn(emu)(0), [1])

    def test_damaging_pp_counts_only_damaging_moves(self):
        emu = RankerEmu([([BITE, ROAR, EMBER], [10, 20, 5])])
        self.assertEqual(rr._damaging_pp(emu, 0), 15)


class TestEffectiveness(unittest.TestCase):
    def test_chart_spot_checks(self):
        self.assertEqual(rr._effectiveness("FIRE", ("GRASS",)), 2.0)
        self.assertEqual(rr._effectiveness("WATER", ("GRASS",)), 0.5)
        self.assertEqual(rr._effectiveness(
            "GRASS", ("WATER", "GROUND")), 4.0)
        self.assertEqual(rr._effectiveness(
            "ELECTRIC", ("GROUND",)), 0.0)
        self.assertEqual(rr._effectiveness(
            "NORMAL", ("GHOST",)), 0.0)
        self.assertEqual(rr._effectiveness("ICE", ("DRAGON",)), 2.0)
        self.assertEqual(rr._effectiveness(
            "FIRE", ("GRASS", "POISON")), 2.0)

    def test_move_info_table(self):
        self.assertEqual(rr._MOVE_INFO[EMBER], (40, "FIRE"))
        self.assertEqual(rr._MOVE_INFO[75], (55, "GRASS"))  # Razor Leaf
        self.assertEqual(rr._MOVE_INFO[BITE], (60, "NORMAL"))
        self.assertNotIn(ROAR, rr._MOVE_INFO)  # status: absent
        self.assertGreaterEqual(len(rr._MOVE_INFO), 100)


class TestSwitchMenuIndex(unittest.TestCase):
    """probe147's party_lead bug: the party submenu prepends a
    mon's FIELD MOVES, so SWITCH is not always item 1."""

    def test_no_field_moves(self):
        self.assertEqual(rr._switch_menu_index([BITE, ROAR, EMBER]), 1)
        self.assertEqual(rr._switch_menu_index([]), 1)

    def test_one_field_move(self):
        self.assertEqual(rr._switch_menu_index([79, 77, 15, 78]), 2)  # Cut
        self.assertEqual(rr._switch_menu_index([33, 103, 148]), 2)     # Flash

    def test_two_field_moves(self):
        self.assertEqual(rr._switch_menu_index([15, 148, 44]), 3)
        self.assertEqual(rr._switch_menu_index([91, 100]), 3)  # Dig, Teleport


if __name__ == "__main__":
    unittest.main()
