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


class TestStartMenuPokemon(unittest.TestCase):
    """probe148: the first port of _swap_party_slots steered
    the START menu to item 3 (the trainer card) and every live
    swap failed; POKEMON is item 1 (POKEDEX 0, POKEMON 1,
    ITEM 2, trainer 3, SAVE 4, OPTION 5, EXIT 6), as in
    probe147's proven helper."""

    def test_pokemon_is_start_menu_item_1(self):
        self.assertEqual(rr.START_MENU_POKEMON, 1)


class TestExploreBfs(unittest.TestCase):
    """probe148's learned-graph BFS: plans over recorded edges
    only; walls and warps are never traversed."""

    MAP_ID = 99999

    def tearDown(self):
        rr.EXPLORE_LEARN.pop(self.MAP_ID, None)

    def _graph(self, edges):
        rr.EXPLORE_LEARN[self.MAP_ID] = {"edges": edges,
                                         "cells": set()}

    def test_path_over_recorded_edges(self):
        self._graph({((0, 0), "right"): (1, 0),
                     ((1, 0), "down"): (1, 1),
                     ((0, 0), "down"): None})
        self.assertEqual(
            rr._explore_bfs(self.MAP_ID, (0, 0), {(1, 1)}),
            [((0, 0), "right"), ((1, 0), "down")])

    def test_walls_and_warps_not_traversed(self):
        self._graph({((0, 0), "right"): None,
                     ((0, 0), "down"): "warp",
                     ((0, 0), "up"): (0, -1)})
        self.assertIsNone(
            rr._explore_bfs(self.MAP_ID, (0, 0), {(1, 0)}))

    def test_start_in_goals_is_empty_path(self):
        self._graph({})
        self.assertEqual(
            rr._explore_bfs(self.MAP_ID, (3, 3), {(3, 3)}), [])


class TestHideoutConstants(unittest.TestCase):
    """probe148's event ids, anchored on EVENT_67F = 1663 by
    construction in the source and on the B2F/B3F trainer
    bits (1665; 1681/1682) read off wEventFlags diffs."""

    def test_event_ids(self):
        self.assertEqual(rr.EV_FOUND_ROCKET_HIDEOUT, 441)
        self.assertEqual(rr.EV_HIDEOUT_B1_TRAINERS,
                         (1649, 1650, 1651, 1652, 1653))
        self.assertEqual(rr.EV_HIDEOUT_B2_TRAINER, 1665)
        self.assertEqual(rr.EV_HIDEOUT_B3_TRAINERS, (1681, 1682))
        self.assertEqual(rr.EV_HIDEOUT_B4_TRAINERS,
                         (1698, 1699, 1700))
        self.assertEqual(rr.EV_HIDEOUT_B4_DOOR, 1701)
        self.assertEqual(rr.EV_ROCKET_DROPPED_LIFT_KEY, 1702)
        self.assertEqual(rr.EV_BEAT_HIDEOUT_GIOVANNI, 1703)

    def test_item_ids(self):
        self.assertEqual(rr.ITEM_LIFT_KEY, 0x4A)
        self.assertEqual(rr.ITEM_SILPH_SCOPE, 0x48)
        self.assertEqual(rr.ITEM_NUGGET, 0x31)
        self.assertEqual(rr.ITEM_HP_UP, 0x23)
        self.assertEqual(rr.ITEM_TM02, 0xCA)


class TestHideoutArrows(unittest.TestCase):
    """The B2F/B3F arrow tables come from pokered's
    map_coord_movement lists, converted (y,x) -> (x,y)."""

    def test_arrow_tables(self):
        b2f = rr.HIDEOUT_ARROWS["RocketHideoutB2F"]
        b3f = rr.HIDEOUT_ARROWS["RocketHideoutB3F"]
        self.assertEqual(len(b2f), 17)
        self.assertEqual(len(b3f), 16)
        self.assertIn((21, 9), b2f)    # east y=9 arrow run
        self.assertIn((19, 18), b3f)   # on the B4F stairs cell
        self.assertIn((2, 12), b3f)


class TestTowerConstants(unittest.TestCase):
    """probe149's Tower event ids: computed with pokered's
    own const macros (expressions included - `const_next
    $F0 - 2` at line 67 shifts the block if skipped) and
    anchored on 441/1663/1703 plus the empirical rival
    diff, which flipped exactly bit 239."""

    def test_event_ids(self):
        self.assertEqual(rr.EV_TOWER_RIVAL, 239)
        self.assertEqual(rr.EV_TOWER_3F_TRAINERS,
                         (241, 242, 243))
        self.assertEqual(rr.EV_TOWER_4F_TRAINERS,
                         (249, 250, 251))
        self.assertEqual(rr.EV_TOWER_5F_TRAINERS,
                         (258, 259, 260, 261))
        self.assertEqual(rr.EV_TOWER_PURIFIED_ZONE, 263)
        self.assertEqual(rr.EV_TOWER_6F_TRAINERS,
                         (265, 266, 267))
        self.assertEqual(rr.EV_BEAT_GHOST_MAROWAK, 271)
        self.assertEqual(rr.EV_TOWER_7F_ROCKETS,
                         (273, 274, 275))
        self.assertEqual(rr.EV_RESCUED_MR_FUJI_2, 279)
        self.assertEqual(rr.EV_GOT_POKE_FLUTE, 296)
        self.assertEqual(rr.EV_RESCUED_MR_FUJI, 1231)

    def test_item_ids(self):
        self.assertEqual(rr.ITEM_POKE_FLUTE, 0x49)
        self.assertEqual(rr.ITEM_RARE_CANDY, 0x28)

    def test_scripted_coords(self):
        self.assertEqual(rr.TOWER_PURIFIED_ZONE,
                         ((10, 8), (11, 8), (10, 9), (11, 9)))
        self.assertEqual(rr.TOWER_MAROWAK_TRIGGER, (10, 16))
        self.assertEqual(rr.TOWER_RIVAL_TRIGGER, (14, 6))


class TestSilphConstants(unittest.TestCase):
    """probes 150/151: Silph event ids from the same
    anchored const parse (1856 rival / 1935 Giovanni /
    1933 Master Ball flipped empirically in the probe
    saves); item ids from pokered's item_constants;
    door blocks from each floor's .GateCoordinates."""

    def test_event_ids(self):
        self.assertEqual(rr.EV_TM29, 944)
        self.assertEqual(rr.EV_SILPH_RIVAL, 1856)
        self.assertEqual(rr.EV_GOT_MASTER_BALL, 1933)
        self.assertEqual(rr.EV_SILPH_GIOVANNI, 1935)
        self.assertEqual(rr.EV_SILPH_11F_ROCKET_WEST, 1924)
        self.assertEqual(rr.SILPH_EV[5],
                         (1826, 1827, 1828, 1829))
        self.assertEqual(rr.SILPH_EV[11], (1924, 1925))

    def test_item_ids(self):
        self.assertEqual(rr.ITEM_CARD_KEY, 0x30)
        self.assertEqual(rr.ITEM_MASTER_BALL, 0x01)
        self.assertEqual(rr.ITEM_TM29, 0xE5)
        self.assertEqual(rr.SPECIES_LAPRAS, 0x13)

    def test_scripted_coords(self):
        self.assertEqual(rr.SILPH_GATES[11], ((6, 12),))
        self.assertEqual(rr.SILPH_ELEV_DOOR[11], (13, 0))
        self.assertEqual(rr.SILPH_ELEV_DOOR[1], (20, 0))
        self.assertIn((6, 12), rr.SILPH_DOOR11)
        # Fisher5 (ev1160) is skipped by design: his
        # pocket is tile-sealed (see GRIND_R12S note)
        evs = [ev for ev, _ in rr.GRIND_R12S]
        self.assertEqual(evs, [1157, 1156])
        self.assertEqual(rr.PIER_WAYPOINTS_S[-1], (11, 106))
        self.assertEqual(rr.PIER_WAYPOINTS_N[0], (13, 95))


class TestFleeWildDefault(unittest.TestCase):
    """probe149: fleeing wilds is opt-in per call - every
    pre-Tower battle path keeps the fight-through default,
    and the Marowak's scripted battle (which reads as a
    wild battle in RAM) depends on the default staying
    off in beat_ghost_marowak."""

    def test_defaults_off(self):
        import inspect
        self.assertEqual(
            inspect.signature(
                rr._drive_battle_typed).parameters[
                    "flee_wild"].default, False)
        self.assertEqual(
            inspect.signature(
                rr._walk_fighting).parameters[
                    "flee_wild"].default, False)
        self.assertEqual(
            inspect.signature(
                rr.explore_nav).parameters[
                    "flee_wild"].default, False)
        self.assertEqual(
            inspect.signature(
                rr.talk_fight_event).parameters[
                    "flee_wild"].default, False)


if __name__ == "__main__":
    unittest.main()


class TestSabrinaConstants(unittest.TestCase):
    """probe152: Sabrina event ids from the anchored
    const parse (866-872 trainer range matches the
    script's SetEventRange; 865/864 flipped empirically
    in the probe save). The pad table is compiled from
    pokered's SaffronGym warp events and cross-checked
    against the probe's learned graph: all 38 empirical
    teleports in graph-d1.json match it exactly."""

    def test_event_ids(self):
        self.assertEqual(rr.EV_GOT_TM46, 864)
        self.assertEqual(rr.EV_BEAT_SABRINA, 865)
        self.assertEqual(rr.EV_SAFFRON_GYM_TRAINERS,
                         (866, 867, 868, 869, 870, 871, 872))

    def test_item_ids(self):
        self.assertEqual(rr.ITEM_TM46, 0xF6)
        self.assertEqual(rr.ITEM_LIFT_KEY, 0x4A)
        self.assertEqual(rr.ADDR_NUM_PC_ITEMS, 0xD53A)
        self.assertEqual(rr.ADDR_PC_ITEMS, 0xD53B)

    def test_pad_table(self):
        pads = rr.SAFFRON_GYM_PADS
        self.assertEqual(len(pads), 30)
        # every pad is one of a reciprocal pair
        for cell, land in pads.items():
            self.assertEqual(pads[land], cell)
        # anchors cross-checked live in probe152
        self.assertEqual(pads[(15, 11)], (9, 3))
        self.assertEqual(pads[(1, 15)], (19, 5))
        self.assertEqual(pads[(11, 15)], (19, 17))

    def test_trainer_table(self):
        self.assertEqual(len(rr.SAFFRON_GYM_TRAINERS), 7)
        self.assertEqual(rr.SABRINA_STANCE, (9, 9))
        self.assertIn(rr.SABRINA_CELL, rr.SAFFRON_GYM_NPCS)
