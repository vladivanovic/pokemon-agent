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


class TestDojoConstants(unittest.TestCase):
    """probe153: Fighting Dojo event ids from the
    anchored const parse (848-855, contiguous with the
    script's SetEventRange 849..853 after the King).
    Species ids are the internal RAM ids (pokered.sym /
    pokemon_constants comments: Lee $2B, Chan $2C).
    The gift table is the objects file: Lee's ball is
    the WEST ball at (4, 1), Chan's the east at (5, 1);
    each stance is the cell below its ball."""

    def test_event_ids(self):
        self.assertEqual(rr.EV_DEFEATED_FIGHTING_DOJO, 848)
        self.assertEqual(rr.EV_BEAT_KARATE_MASTER, 849)
        self.assertEqual(rr.EV_DOJO_TRAINERS,
                         (850, 851, 852, 853))
        self.assertEqual(rr.EV_GOT_HITMONLEE, 854)
        self.assertEqual(rr.EV_GOT_HITMONCHAN, 855)

    def test_species_and_gate(self):
        self.assertEqual(rr.SPECIES_HITMONLEE, 0x2B)
        self.assertEqual(rr.SPECIES_HITMONCHAN, 0x2C)
        self.assertEqual(rr.ADDR_CUR_PARTY_SPECIES, 0xCF91)

    def test_gift_table(self):
        lee = rr.DOJO_GIFTS["hitmonlee"]
        chan = rr.DOJO_GIFTS["hitmonchan"]
        self.assertEqual(lee[0], (4, 1))       # west ball
        self.assertEqual(chan[0], (5, 1))      # east ball
        self.assertLess(lee[0][0], chan[0][0])
        self.assertEqual(lee[1], (4, 2))       # stance below
        self.assertEqual(chan[1], (5, 2))
        self.assertEqual(lee[2], rr.SPECIES_HITMONLEE)
        self.assertEqual(chan[2], rr.SPECIES_HITMONCHAN)
        self.assertEqual(lee[3], rr.EV_GOT_HITMONLEE)
        self.assertEqual(chan[3], rr.EV_GOT_HITMONCHAN)

    def test_belts_and_master(self):
        self.assertEqual(len(rr.DOJO_BELTS), 4)
        self.assertEqual([b[1] for b in rr.DOJO_BELTS],
                         list(rr.EV_DOJO_TRAINERS))
        self.assertEqual(rr.DOJO_MASTER_CELL, (5, 3))
        # the script trigger is the cell left of the King
        self.assertEqual(rr.DOJO_MASTER_TRIGGER, (4, 3))
        for cell, _, sights, _ in rr.DOJO_BELTS:
            self.assertIn(cell, rr.DOJO_OBJECT_CELLS)
            self.assertTrue(sights)
        for ball, _, _, _ in rr.DOJO_GIFTS.values():
            self.assertIn(ball, rr.DOJO_OBJECT_CELLS)


# Safari Zone warp tables (pokered object files; per map,
# (x, y, dest map, dest warp index) in warp order). The
# plan tests run run_real's real BFS over the embedded
# compiled grids plus this graph - the same composition
# _sz_nav executes live (probe154).
_SAFARI_WARPS = {
    220: [(14, 25, 156, 3), (15, 25, 156, 4), (0, 10, 219, 5),
          (0, 11, 219, 6), (14, 0, 218, 5), (15, 0, 218, 6),
          (29, 10, 217, 3), (29, 11, 217, 4), (17, 19, 221, 1)],
    219: [(20, 0, 218, 1), (21, 0, 218, 2), (26, 0, 218, 3),
          (27, 0, 218, 4), (29, 22, 220, 3), (29, 23, 220, 4),
          (3, 3, 222, 1), (11, 11, 223, 1)],
    217: [(0, 4, 218, 7), (0, 5, 218, 8), (0, 22, 220, 7),
          (0, 23, 220, 7), (25, 9, 224, 1)],
    218: [(2, 35, 219, 1), (3, 35, 219, 2), (8, 35, 219, 3),
          (9, 35, 219, 4), (20, 35, 220, 5), (21, 35, 220, 6),
          (39, 30, 217, 1), (39, 31, 217, 2), (35, 3, 225, 1)],
}


def _safari_fake_ex():
    maps = {}
    for mid, warps in _SAFARI_WARPS.items():
        maps[mid] = SimpleNamespace(warps=[
            SimpleNamespace(x=x, y=y, dest_id=dm, idx=di)
            for (x, y, dm, di) in warps])
    return SimpleNamespace(world=SimpleNamespace(maps=maps))


class TestSafari(unittest.TestCase):
    def test_steps_counter_is_big_endian(self):
        emu = FakeEmu()
        emu.memory[rr.ADDR_SAFARI_STEPS] = 0x01
        emu.memory[rr.ADDR_SAFARI_STEPS + 1] = 0xF4
        self.assertEqual(rr._safari_steps(emu), 500)

    def test_balls_counter(self):
        emu = FakeEmu()
        emu.memory[rr.ADDR_SAFARI_BALLS] = 30
        self.assertEqual(rr._safari_balls(emu), 30)

    def test_grid_dimensions_match_pokered(self):
        self.assertEqual(rr.SAFARI_GRIDS[218][:2], (40, 36))
        for mid in (217, 219, 220):
            self.assertEqual(rr.SAFARI_GRIDS[mid][:2], (30, 26))

    def test_grid_cells_the_probe_stood_on(self):
        # entrance apron, teeth stance, Secret House stance
        self.assertTrue(rr._safari_passable(220, (15, 24)))
        self.assertTrue(rr._safari_passable(219, (19, 8)))
        self.assertTrue(rr._safari_passable(219, (3, 4)))

    def test_plan_rings_center_to_the_teeth(self):
        ex = _safari_fake_ex()
        blocked = {m: set(rr.SAFARI_BALLS_CELLS.get(m, ()))
                   for m in rr.SAFARI_AREAS}
        path = rr._safari_plan(ex, (220, (15, 25)),
                               (219, (19, 8)), blocked)
        self.assertIsNotNone(path)
        self.assertIn("warp", path)      # crosses areas
        self.assertGreater(len(path), 100)

    def test_plan_reaches_the_secret_house_stance(self):
        ex = _safari_fake_ex()
        blocked = {m: set(rr.SAFARI_BALLS_CELLS.get(m, ()))
                   for m in rr.SAFARI_AREAS}
        path = rr._safari_plan(ex, (219, (19, 8)),
                               (219, (3, 4)), blocked)
        self.assertIsNotNone(path)

    def test_plan_never_routes_through_a_ball(self):
        # West's (19, 7) ball cell is the teeth itself: it
        # is blocked at plan time, so the plan for the
        # stance must end ADJACENT, not through it.
        ex = _safari_fake_ex()
        blocked = {m: set(rr.SAFARI_BALLS_CELLS.get(m, ()))
                   for m in rr.SAFARI_AREAS}
        self.assertIn((19, 7), blocked[219])
        path = rr._safari_plan(ex, (220, (15, 25)),
                               (219, (19, 8)), blocked)
        self.assertIsNotNone(path)

    @staticmethod
    def _grass(map_id, x, y):
        w, h, bits = rr.SAFARI_GRASS[map_id]
        if not (0 <= x < w and 0 <= y < h):
            return False
        i = y * w + x
        return bool(bits[i // 8] >> (i % 8) & 1)

    def test_grass_pair_west_is_not_the_sterile_edge_pair(self):
        # probe176 (docs S37): the old anchor-nearest West
        # pair ((17,20),(17,21)) is the field's last grass
        # column and produced ZERO encounters in ~1,080
        # paced steps across five paid entries - the
        # encounter check reads the half-block's
        # bottom-right tile, which is off-grass there.
        pair = rr._safari_grass_pair(rr.MAP_SZ_WEST)
        self.assertNotEqual(pair, ((17, 20), (17, 21)))
        for (x, y) in pair:
            self.assertTrue(self._grass(219, x + 1, y))
            self.assertTrue(self._grass(219, x, y + 1))

    def test_grass_pairs_are_live_in_every_area(self):
        # Wherever a both-live pair exists (grass east AND
        # south of both cells), the picker must return one.
        for mid in (217, 218, 219, 220):
            pair = rr._safari_grass_pair(mid)
            self.assertIsNotNone(pair)
            for (x, y) in pair:
                self.assertTrue(self._grass(mid, x + 1, y),
                                (mid, pair))
                self.assertTrue(self._grass(mid, x, y + 1),
                                (mid, pair))

    def test_west_route_is_a_continuous_walk(self):
        # The recorded ring route (probe176): starts on the
        # Center spawn, ends on the West pair's first cell,
        # and every step is a 1-cell move, a straight
        # 2-cell ledge jump, or a warp hop to the map of a
        # warp cell adjacent to the previous cell.
        cells = rr.SAFARI_WEST_ROUTE
        self.assertEqual(cells[0], (220, 15, 25))
        # Ends at (17,20): the pair cell the route was
        # recorded for. The live pair (S37 fix) starts two
        # cells west at (15,20) - a short grass walk.
        self.assertEqual(cells[-1], (219, 17, 20))
        warps = {(m, w.x, w.y): w.dest_id
                 for m, ws in _SAFARI_WARPS.items()
                 for w in [SimpleNamespace(x=a, y=b, dest_id=c)
                           for (a, b, c, _i) in ws]}
        for cur, nxt in zip(cells, cells[1:]):
            if nxt[0] == cur[0]:
                dx = abs(nxt[1] - cur[1])
                dy = abs(nxt[2] - cur[2])
                self.assertIn((dx, dy),
                              [(1, 0), (0, 1), (2, 0), (0, 2)],
                              (cur, nxt))
            else:
                adj = [dest for (m, x, y), dest in warps.items()
                       if m == cur[0]
                       and abs(x - cur[1]) + abs(y - cur[2]) == 1]
                self.assertIn(nxt[0], adj, (cur, nxt))


class TestFuchsiaConstants(unittest.TestCase):
    def test_hm_and_tm_ids(self):
        self.assertEqual(rr.ITEM_HM03_SURF, 0xC6)
        self.assertEqual(rr.ITEM_HM04_STRENGTH, 0xC7)
        self.assertEqual(rr.ITEM_TM06_TOXIC, 0xCE)
        self.assertEqual(rr.ITEM_GOLD_TEETH, 0x40)

    def test_gym_trainer_events_in_object_order(self):
        self.assertEqual(rr.EV_FUCHSIA_GYM_TRAINERS,
                         (602, 603, 604, 605, 606, 607))
        cells = [c for c, _, _, _ in rr.FUCHSIA_GYM_TRAINERS]
        self.assertEqual(cells, [(8, 13), (7, 8), (1, 12),
                                 (3, 5), (8, 2), (2, 7)])


class TestSafariHunt(unittest.TestCase):
    def test_target_species_internal_ids(self):
        # const_skip-aware parse of pokered
        # constants/pokemon_constants.asm, validated against
        # every species constant run_real already carried
        self.assertEqual(rr.SAFARI_SPECIES[0x28], "CHANSEY")
        self.assertEqual(rr.SAFARI_SPECIES[0x02], "KANGASKHAN")
        self.assertEqual(rr.SAFARI_SPECIES[0x3C], "TAUROS")
        self.assertEqual(rr.SAFARI_SPECIES[0x1A], "SCYTHER")
        self.assertEqual(rr.SAFARI_SPECIES[0x58], "DRATINI")
        # probe173: 0x0F logged nameless through three hunt
        # rounds - it is NIDORAN_F (pokered pokemon_constants
        # $0F), the East-area slot-7 encounter
        self.assertEqual(rr.SAFARI_SPECIES[0x0F], "NIDORAN_F")
        self.assertEqual(rr.SAFARI_CATCH_RATE[0x0F],
                         rr.SAFARI_CATCH_RATE[0x03])
        self.assertEqual(rr.SAFARI_SPEED[0x0F],
                         rr.SAFARI_SPEED[0x03])
        self.assertEqual(rr.SAFARI_TARGETS,
                         frozenset({0x28, 0x02, 0x3C, 0x1A, 0x58}))

    def test_rod_hm_move_ids(self):
        self.assertEqual(rr.ITEM_GOOD_ROD, 0x4D)
        self.assertEqual(rr.ITEM_SUPER_ROD, 0x4E)
        self.assertEqual(rr.ITEM_TM24, 0xE0)
        self.assertEqual(rr.MOVE_SURF, 57)
        self.assertEqual(rr.MOVE_STRENGTH, 70)
        self.assertEqual(rr.SPECIES_LAPRAS, 0x13)
        self.assertEqual(rr.MAP_FUCHSIA_GOOD_ROD_HOUSE, 164)

    def test_tables_cover_every_species(self):
        for sp in rr.SAFARI_SPECIES:
            self.assertIn(sp, rr.SAFARI_CATCH_RATE)
            self.assertIn(sp, rr.SAFARI_SPEED)

    def test_target_catch_rates_and_speeds(self):
        # pokered base stats: Chansey is the hardest catch
        # in the zone; Tauros the fastest target
        self.assertEqual(rr.SAFARI_CATCH_RATE[0x28], 30)
        self.assertEqual(rr.SAFARI_CATCH_RATE[0x02], 45)
        self.assertEqual(rr.SAFARI_CATCH_RATE[0x3C], 45)
        self.assertEqual(rr.SAFARI_SPEED[0x3C], 110)
        self.assertEqual(rr.SAFARI_SPEED[0x28], 50)
        for sp in rr.SAFARI_TARGETS:
            self.assertLessEqual(rr.SAFARI_SPEED[sp], 110)

    def test_grass_dimensions_match_grids(self):
        for mid in (217, 218, 219, 220):
            self.assertEqual(rr.SAFARI_GRASS[mid][:2],
                             rr.SAFARI_GRIDS[mid][:2])

    def test_grass_pair_is_adjacent_passable_grass(self):
        for mid in (217, 218, 219, 220):
            pair = rr._safari_grass_pair(mid)
            self.assertIsNotNone(pair, mid)
            (ax, ay), (bx, by) = pair
            self.assertEqual(abs(ax - bx) + abs(ay - by), 1)
            w, h, bits = rr.SAFARI_GRASS[mid]
            for (x, y) in pair:
                i = y * w + x
                self.assertTrue(bits[i // 8] >> (i % 8) & 1,
                                (mid, x, y))
                self.assertTrue(rr._safari_passable(mid, (x, y)),
                                (mid, x, y))
                self.assertNotIn(
                    (x, y),
                    rr.SAFARI_BALLS_CELLS.get(mid, frozenset()))

    def test_area_names_cover_all_areas(self):
        self.assertEqual(set(rr.SAFARI_AREAS_BY_NAME.values()),
                         {217, 218, 219, 220})


class TestTeachStrength(unittest.TestCase):
    """probe156: HM04 -> Lapras, Vlad's re-decision after
    Arcanine proved engine-incompatible. The verification-
    first rule made a test: read the tmhm learnsets from
    pokered itself - the data CanLearnTM reads."""

    def _tmhm_tokens(self, species_file):
        from pathlib import Path
        asm = (Path(rr.__file__).resolve().parent.parent
               / "pokered" / "data" / "pokemon" / "base_stats"
               / species_file)
        if not asm.exists():
            self.skipTest("pokered tree not present")
        text = asm.read_text()
        block = text.split("tmhm", 1)[1].split("; end", 1)[0]
        block = block.replace("\\", " ").replace("\n", " ")
        return {tok.strip(",") for tok in block.split()
                if tok.strip(",")}

    def test_strength_step_wired(self):
        self.assertTrue(callable(rr.teach_strength))
        self.assertEqual(rr.ITEM_HM04_STRENGTH, 0xC7)
        self.assertEqual(rr.MOVE_STRENGTH, 70)
        self.assertEqual(rr.MOVE_SURF, 57)
        self.assertEqual(rr.SPECIES_LAPRAS, 0x13)

    def test_lapras_learnset_includes_both_hms(self):
        tokens = self._tmhm_tokens("lapras.asm")
        self.assertIn("SURF", tokens)
        self.assertIn("STRENGTH", tokens)

    def test_arcanine_learnset_has_no_hms(self):
        # the probe155 finding that forced the re-decision
        tokens = self._tmhm_tokens("arcanine.asm")
        for hm in ("CUT", "FLY", "SURF", "STRENGTH", "FLASH"):
            self.assertNotIn(hm, tokens)


class TestGrandTourTables(unittest.TestCase):
    """probe158's corrected data pins: each of these was
    wrong in the §22 draft and cost a live failure."""

    def test_cinnabar_gate_events_are_681_to_686(self):
        # quiz gate index i sets event 680 + i (FlagAction
        # counts from GATE0_UNLOCKED=680 with the index, not
        # index-1) - proven live in probe158m2/m4.
        self.assertEqual(rr.EV_GYM_GATES,
                         (681, 682, 683, 684, 685, 686))

    def test_cinnabar_doors_pair_gate_events_in_order(self):
        events = [ev for _blk, _gid, ev in rr.CINNABAR_GYM_DOORS]
        self.assertEqual(events, list(rr.EV_GYM_GATES))
        blocks = [blk for blk, _gid, _ev in rr.CINNABAR_GYM_DOORS]
        self.assertEqual(blocks, [(9, 3), (6, 3), (6, 6),
                                  (3, 8), (2, 6), (2, 3)])

    def test_cinnabar_quiz_gates_index_the_event_table(self):
        for _machine, _stance, gi, _want in rr.CINNABAR_QUIZ:
            self.assertEqual(rr.EV_GYM_GATES[gi], 681 + gi)

    def test_mansion_second_b1f_statue(self):
        # hidden_events lists two B1F statues; the route
        # toggles (18, 26) first, then (20, 4).
        self.assertEqual(rr.MANSION_STATUES_ALT[216], (18, 26))
        self.assertEqual(rr.MANSION_STATUES[216], (20, 4))

    def test_mansion_doors_match_floor_scripts(self):
        self.assertEqual(
            rr.MANSION_DOORS[214],
            [((4, 2), 0x5F, False), ((9, 4), 0x54, True),
             ((3, 11), 0x5F, True)])
        self.assertEqual(
            rr.MANSION_DOORS[215],
            [((7, 2), 0x5F, False), ((7, 5), 0x5F, True)])
        self.assertEqual(
            rr.MANSION_DOORS[216],
            [((13, 8), 0x2D, False), ((6, 11), 0x5F, False),
             ((4, 3), 0x5F, True), ((8, 8), 0x54, True)])

    def test_seafoam_cascade_events(self):
        self.assertEqual(rr.SEAFOAM_CASCADE_EVENTS,
                         (1294, 1295, 2496, 2497, 2504, 2505))

    def test_bike_voucher_item_id(self):
        self.assertEqual(rr.ITEM_BIKE_VOUCHER, 0x2D)


class TestGrandTourWorldFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agent.world import default_world
        cls.w = default_world()

    def test_seafoam_holes_and_boulder_starts_are_floor(self):
        # Hole cells are steppable floor (that is what makes
        # them walker hazards); boulders start on floor too.
        w = self.w
        holes = {"SeafoamIslands1F": {(17, 6), (24, 6)},
                 "SeafoamIslandsB1F": {(18, 6), (23, 6)},
                 "SeafoamIslandsB2F": {(19, 6), (22, 6)},
                 "SeafoamIslandsB3F": {(3, 16), (6, 16)}}
        for name, cells in holes.items():
            gm = w.maps[w.map_id(name)]
            for cell in cells:
                self.assertTrue(gm.tile_walkable(*cell),
                                (name, cell))
        starts = {"SeafoamIslands1F": {(18, 10), (26, 7)},
                  "SeafoamIslandsB1F": {(17, 6), (22, 6)},
                  "SeafoamIslandsB2F": {(18, 6), (23, 6)}}
        for name, cells in starts.items():
            gm = w.maps[w.map_id(name)]
            for cell in cells:
                self.assertTrue(gm.tile_walkable(*cell),
                                (name, cell))

    def test_safari_pond_stance_is_reachable(self):
        # probe158o: the unfiltered search picked (15, 14),
        # a shore segment fenced off by posts and unreachable
        # from the entry - the nav died on the spot. The
        # filtered pick must be BFS-reachable from the anchor
        # and face a $14 water cell.
        from collections import deque
        ex = SimpleNamespace(world=self.w)
        stance, face = rr._safari_pond_stance(None, ex)
        self.assertNotEqual(stance, (15, 14))
        gm = self.w.maps[rr.MAP_SZ_CENTER]
        anchor = rr.SAFARI_ANCHORS[rr.MAP_SZ_CENTER]
        warps = {(w.x, w.y) for w in gm.warps}
        reach = {anchor}
        queue = deque([anchor])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in rr._SZ_DIRS.values():
                n = (cx + dx, cy + dy)
                if n not in reach and n not in warps \
                        and rr._safari_passable(rr.MAP_SZ_CENTER,
                                                n):
                    reach.add(n)
                    queue.append(n)
        self.assertIn(stance, reach)
        dx, dy = rr._SZ_DIRS[face]
        front = (stance[0] + dx, stance[1] + dy)
        self.assertEqual(gm.tiles[front[1]][front[0]], 0x14)


class TestFlyLeg(unittest.TestCase):
    """probe160: the Fly leg - bicycle exchange, the Snorlax
    catch protocol, HM02, verify-first Fly teach, and the
    header-steered flight. Data pins from pokered + the live
    probe160 runs (docs S24)."""

    def _tmhm_tokens(self, species_file):
        from pathlib import Path
        asm = (Path(rr.__file__).resolve().parent.parent
               / "pokered" / "data" / "pokemon" / "base_stats"
               / species_file)
        if not asm.exists():
            self.skipTest("pokered tree not present")
        text = asm.read_text()
        block = text.split("tmhm", 1)[1].split("; end", 1)[0]
        block = block.replace("\\", " ").replace("\n", " ")
        return {tok.strip(",") for tok in block.split()
                if tok.strip(",")}

    def test_fly_leg_functions_wired(self):
        for fn in (rr.bicycle_exchange, rr.snorlax_catch,
                   rr.route16_fly_house, rr.teach_fly, rr.fly_to):
            self.assertTrue(callable(fn))

    def test_fly_leg_constants(self):
        self.assertEqual(rr.ITEM_BICYCLE, 0x06)
        self.assertEqual(rr.ITEM_HM02_FLY, 0xC5)
        self.assertEqual(rr.MOVE_FLY, 19)
        self.assertEqual(rr.SPECIES_SNORLAX, 0x84)
        self.assertEqual(rr.EV_GOT_BICYCLE, 192)
        self.assertEqual(rr.EV_FIGHT_R16_SNORLAX, 1224)
        self.assertEqual(rr.EV_BEAT_R16_SNORLAX, 1225)
        self.assertEqual(rr.EV_GOT_HM02, 1230)

    def test_pidgey_learnset_includes_fly(self):
        # the verify-first check that made the teach offerable
        tokens = self._tmhm_tokens("pidgey.asm")
        self.assertIn("FLY", tokens)

    def test_snorlax_catch_never_references_master_ball(self):
        # Vlad's rule, pinned mechanically: the catch function's
        # source must not touch the Master Ball at all.
        import inspect
        src = inspect.getsource(rr.snorlax_catch)
        self.assertNotIn("MASTER_BALL", src)
        self.assertNotIn("ITEM_MASTER", src)

    def test_fly_town_cycle_order(self):
        # observed live off the header (probe160f11): Lavender
        # cycles BEFORE Vermilion, Fuchsia after Saffron.
        cyc = rr.FLY_TOWN_CYCLE
        self.assertEqual(cyc[0], "PALLET TOWN")
        self.assertLess(cyc.index("LAVENDER TOWN"),
                        cyc.index("VERMILION CITY"))
        self.assertLess(cyc.index("SAFFRON CITY"),
                        cyc.index("FUCHSIA CITY"))
        for name in cyc:
            self.assertIn(name, rr.FLY_TOWN_MAP_NAMES)

    def test_fly_header_decode(self):
        # the captured open-map header row: "To PALLET TOWN"
        row = bytes([0x93, 0xAE, 0x67, 0x8F, 0x80, 0x8B, 0x8B,
                     0x84, 0x93, 0x7F, 0x93, 0x8E, 0x96, 0x8D,
                     0x7F, 0x7F, 0x7F, 0x7F, 0xED, 0xEE])

        class FakeMem:
            def __getitem__(self, addr):
                if 0x9C00 <= addr < 0x9C00 + 20:
                    return row[addr - 0x9C00]
                return 0

        class FakeEmu:
            class pb:
                memory = FakeMem()
        self.assertEqual(rr._fly_header_town(FakeEmu()),
                         "PALLET TOWN")


class TestViridianLeg(unittest.TestCase):
    """probe162: the Viridian leg - Good Rod deposit for
    TM27 bag room, the arrow floor compiled from pokered's
    RLE table, Giovanni for the Earth Badge. Data pins from
    pokered + the live probe162 runs (docs S25)."""

    def _pokered(self, *parts):
        from pathlib import Path
        p = (Path(rr.__file__).resolve().parent.parent
             / "pokered").joinpath(*parts)
        if not p.exists():
            self.skipTest("pokered tree not present")
        return p.read_text()

    def test_viridian_functions_wired(self):
        for fn in (rr.viridian_gym, rr._gym_go, rr._gym_step,
                   rr._gym_bfs, rr._gym_slide_end,
                   rr._gym_sprite_cells, rr._gym_exit):
            self.assertTrue(callable(fn))

    def test_viridian_constants(self):
        self.assertEqual(rr.ITEM_TM27, 0xE3)
        self.assertEqual(rr.EV_VIRIDIAN_GYM_OPEN, 40)
        self.assertEqual(rr.EV_GOT_TM27, 80)
        self.assertEqual(rr.EV_BEAT_VIRIDIAN_GYM_GIOVANNI, 81)
        self.assertEqual(rr.EV_VIRIDIAN_GYM_TRAINERS,
                         tuple(range(82, 90)))
        self.assertEqual(rr.VIRIDIAN_GYM_STANCE, (2, 2))

    def test_arrow_table_matches_pokered(self):
        # parse ViridianGymArrowTilePlayerMovement out of
        # the gym script and compare trigger-for-trigger
        asm = self._pokered("scripts", "ViridianGym.asm")
        block = asm.split("ViridianGymArrowTilePlayerMovement:",
                          1)[1].split("db -1", 1)[0]
        moves = {}
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("map_coord_movement"):
                continue
            nums = line.split("map_coord_movement", 1)[1]
            x, y, label = [t.strip() for t in nums.split(",")]
            m = asm.split(label + ":", 1)[1].split("db -1", 1)[0]
            first = m.strip().splitlines()[0].strip()
            pad, count = [t.strip()
                          for t in first[2:].split(",")]
            moves[(int(x), int(y))] = (
                pad.replace("PAD_", "").lower(), int(count))
        self.assertEqual(moves, rr.VIRIDIAN_GYM_ARROWS)

    def test_giovanni_party_matches_pokered(self):
        asm = self._pokered("data", "trainers", "parties.asm")
        block = asm.split("GiovanniData:", 1)[1]
        entries = [ln.strip() for ln in block.splitlines()
                   if ln.strip().startswith("db $FF")]
        gym = entries[2]  # Hideout, Silph, then the Gym
        toks = [t.strip() for t in gym[2:].split(",")]
        self.assertEqual(toks[0], "$FF")
        team = [(int(toks[i]), toks[i + 1])
                for i in range(1, len(toks) - 1, 2)]
        self.assertEqual(team, [(45, "RHYHORN"), (42, "DUGTRIO"),
                                (44, "NIDOQUEEN"),
                                (45, "NIDOKING"), (50, "RHYDON")])

    def test_slide_resolution_on_compiled_grid(self):
        from agent.world import default_world
        gmap = next(m for m in default_world().maps.values()
                    if m.name == "ViridianGym")
        end = rr._gym_slide_end(gmap, (11, 2), set(), set())
        self.assertEqual(end, (17, 2))       # RIGHT 6, full
        end = rr._gym_slide_end(gmap, (18, 2), set(), set())
        self.assertEqual(end, (18, 11))      # DOWN 9, full
        end = rr._gym_slide_end(gmap, (13, 17), set(), set())
        self.assertEqual(end, (1, 17))       # LEFT 12, full
        # a sprite on the lane stops the slide early
        end = rr._gym_slide_end(gmap, (11, 2), {(14, 2)},
                                set())
        self.assertEqual(end, (13, 2))

    def test_road_to_giovanni_exists(self):
        from agent.world import default_world
        gmap = next(m for m in default_world().maps.values()
                    if m.name == "ViridianGym")
        home = {(2, 1), (12, 7), (11, 11), (10, 7), (3, 7),
                (13, 5), (10, 1), (2, 16), (6, 5), (16, 15),
                (16, 9)}
        path = rr._gym_bfs(gmap, (16, 16),
                           rr.VIRIDIAN_GYM_STANCE, home, set())
        self.assertIsNotNone(path)

    def test_viridian_gym_fences(self):
        # Vlad's fences, pinned in the source: no Route 22,
        # no Victory Road, and the Good Rod deposit (the
        # TM27 bag-room step) is part of the leg.
        import inspect
        src = inspect.getsource(rr.viridian_gym)
        self.assertNotIn("Route22", src)
        self.assertNotIn("VictoryRoad", src)
        self.assertIn("ITEM_GOOD_ROD", src)
        self.assertIn("deposit_item", src)


class TestRoadToIndigo(unittest.TestCase):
    """probe165: the road to Indigo - the Mansion grind
    (Silph playbook), the Route 22 re-fight, Route 23 and
    Victory Road to the Plateau heal. Data pins from
    pokered + the live probe165 runs (docs S27)."""

    def _pokered(self, *parts):
        from pathlib import Path
        p = (Path(rr.__file__).resolve().parent.parent
             / "pokered").joinpath(*parts)
        if not p.exists():
            self.skipTest("pokered tree not present")
        return p.read_text()

    def test_road_functions_wired(self):
        for fn in (rr.route22_prep, rr.mansion_grind,
                   rr.route22_rival, rr.route23_passage,
                   rr.victory_road, rr._fight_route22_rival,
                   rr._grind_one, rr._vr_push, rr._vr_walk,
                   rr._vr_step_to, rr._ride_menu,
                   rr._battle_enemy, rr._exit_building):
            self.assertTrue(callable(fn))

    def test_road_constants(self):
        self.assertEqual(rr.EV_BEAT_RIVAL2_R22, 1318)
        self.assertEqual(rr.EV_VR1_BOULDER, 2327)
        self.assertEqual(rr.EV_VR2_SWITCH1, 1336)
        self.assertEqual(rr.EV_VR2_SWITCH2, 1343)
        self.assertEqual(rr.EV_VR3_HOLE, 1638)
        self.assertEqual(rr.FUCHSIA_MART_LIST,
                         [0x02, 0x03, 0x13, 0x35, 0x34,
                          0x1F])

    def test_vr_switch_tables_end_on_switches(self):
        self.assertEqual(rr._VR1_PUSHES[-1][2], (17, 13))
        self.assertEqual(rr._VR2_SWITCH1_PUSHES[-1][2],
                         (1, 16))
        self.assertEqual(rr._VR3_SWITCH_PUSHES[-1][2],
                         (3, 5))
        self.assertEqual(rr._VR3_TO_STANCE[-1], (21, 15))
        self.assertEqual(rr._VR3_TO_SWITCHROOM[-1],
                         (9, 10))
        self.assertEqual(rr._VR3_TO_STANCE[0], (8, 10))

    def test_push_tables_chain(self):
        # every push starts where the previous one landed
        for table in (rr._VR1_PUSHES,
                      rr._VR2_SWITCH1_PUSHES,
                      rr._VR3_SWITCH_PUSHES):
            for prev, nxt in zip(table, table[1:]):
                self.assertEqual(prev[2], nxt[0])

    def test_step_tables_are_adjacent_chains(self):
        for table in (rr._VR3_TO_SWITCHROOM,
                      rr._VR3_TO_STANCE):
            for a, b in zip(table, table[1:]):
                self.assertEqual(
                    abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def _switch_coords(self, script, marker):
        asm = self._pokered("scripts", script)
        block = asm.split(marker, 1)[1].split("db -1", 1)[0]
        out = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("dbmapcoord"):
                nums = line.split("dbmapcoord", 1)[1]
                x, y = [t.strip() for t in
                        nums.split(";")[0].split(",")]
                out.append((int(x), int(y)))
        return out

    def test_vr2_switch_coords_match_pokered(self):
        self.assertEqual(
            self._switch_coords("VictoryRoad2F.asm",
                                ".SwitchCoords:"),
            [(1, 16), (9, 16)])

    def test_vr3_switch_hole_coords_match_pokered(self):
        self.assertEqual(
            self._switch_coords("VictoryRoad3F.asm",
                                ".SwitchOrHoleCoords:"),
            [(3, 5), (23, 15)])

    def test_grind_targets(self):
        import inspect
        src = inspect.getsource(rr.mansion_grind)
        self.assertIn("SP_BLASTOISE, 66", src)
        self.assertIn("SP_ARCANINE, 48", src)

    def test_rival_policy_pins(self):
        # Alakazam is Arcanine's fight (the S26 lesson);
        # the shift prompt is declined via _ride_menu.
        import inspect
        src = inspect.getsource(rr._fight_route22_rival)
        self.assertIn("SP_ALAKAZAM", src)
        self.assertIn("MOVE_BITE", src)
        self.assertIn("MOVE_SLEEP_POWDER", src)
        self.assertIn("_ride_menu", src)

    def test_road_fences(self):
        import inspect
        # Vlad's fences, pinned in the source: no ball is
        # ever thrown on the road legs, no TM is taught,
        # and Moltres is named as never approached.
        for fn in (rr.mansion_grind, rr.route22_rival,
                   rr.route23_passage, rr.victory_road):
            src = inspect.getsource(fn)
            self.assertNotIn("_throw_ball", src)
            self.assertNotIn("teach_", src)
            self.assertNotIn("ITEM_MASTER_BALL", src)
        src = inspect.getsource(rr.victory_road)
        self.assertIn("Moltres", src)
        # the only purchase in the road legs is the prep
        src = inspect.getsource(rr.route22_prep)
        self.assertIn("ITEM_SUPER_POTION, 6", src)
        self.assertIn("ITEM_REVIVE, 2", src)
        self.assertIn("ITEM_FULL_HEAL, 2", src)


class TestLeagueLeg(unittest.TestCase):
    """probe167: the League leg - prompt-aware grind,
    assembly and the Elite Four gauntlet."""

    def _pokered(self, *parts):
        from pathlib import Path
        p = (Path(rr.__file__).resolve().parent.parent
             / "pokered").joinpath(*parts)
        if not p.exists():
            self.skipTest("pokered tree not present")
        return p.read_text()

    def test_league_functions_wired(self):
        for fn in (rr.league_grind, rr.league_assemble,
                   rr.elite_four, rr._league_grind_one,
                   rr._league_fight, rr._league_decide,
                   rr._league_menu_move, rr._league_heal_trip,
                   rr._corridor_care, rr._step_walk,
                   rr._ride_ending, rr._answer_learn_prompt,
                   rr._guarded_advance, rr._close_all_menus):
            self.assertTrue(callable(fn))

    def test_league_species_ids_match_pokered(self):
        # Gen 1 internal ids count EVERY const in
        # pokemon_constants.asm INCLUDING const_skip
        # slots - the probe's table that skipped them was
        # low by the skip count and silently disabled
        # whole policy branches for a run (docs S28).
        txt = self._pokered("constants",
                            "pokemon_constants.asm")
        names = []
        started = False
        import re
        for line in txt.splitlines():
            if "const_def" in line:
                started = True
                continue
            if not started:
                continue
            m = re.match(r"\s*const (\w+)", line)
            if m:
                names.append(m.group(1))
                continue
            m = re.match(r"\s*const_skip(?:\s+(\d+))?",
                         line)
            if m:
                names.extend(["SKIP"] * int(m.group(1)
                                             or 1))
            if "NUM_POKEMON" in line:
                break
        pairs = [("SP_LAPRAS", "LAPRAS"),
                 ("SP_SNORLAX", "SNORLAX"),
                 ("SP_PIDGEOT", "PIDGEOT"),
                 ("SP_RHYDON", "RHYDON"),
                 ("SP_DEWGONG", "DEWGONG"),
                 ("SP_CLOYSTER", "CLOYSTER"),
                 ("SP_SLOWBRO", "SLOWBRO"),
                 ("SP_JYNX", "JYNX"),
                 ("SP_ONIX", "ONIX"),
                 ("SP_HITMONCHAN", "HITMONCHAN"),
                 ("SP_HITMONLEE", "HITMONLEE"),
                 ("SP_MACHAMP", "MACHAMP"),
                 ("SP_GENGAR", "GENGAR"),
                 ("SP_GOLBAT", "GOLBAT"),
                 ("SP_HAUNTER", "HAUNTER"),
                 ("SP_ARBOK", "ARBOK"),
                 ("SP_DRAGONAIR", "DRAGONAIR"),
                 ("SP_AERODACTYL", "AERODACTYL"),
                 ("SP_DRAGONITE", "DRAGONITE"),
                 ("SP_GLOOM", "GLOOM"),
                 ("SP_VOLTORB", "VOLTORB"),
                 ("SP_ALAKAZAM", "ALAKAZAM"),
                 ("SP_VENUSAUR", "VENUSAUR"),
                 ("SP_GYARADOS", "GYARADOS")]
        for const_name, species in pairs:
            self.assertEqual(getattr(rr, const_name),
                             names.index(species),
                             const_name)

    def test_league_learn_plans_reach_final_movesets(self):
        def apply_plan(start, plan):
            exp = list(start)
            for lv in sorted(plan):
                move, p = plan[lv]
                if p[0] == "accept":
                    exp[exp.index(p[1])] = move
                elif p[0] == "append" and len(exp) < 4:
                    exp.append(move)
            return exp
        lapras = apply_plan(
            [rr.MOVE_WATER_GUN, rr.MOVE_GROWL, rr.MOVE_SURF,
             rr.MOVE_STRENGTH], rr._LEAGUE_LEARN_LAPRAS)
        self.assertEqual(lapras, [rr.MOVE_SING,
                                  rr.MOVE_ICE_BEAM,
                                  rr.MOVE_SURF,
                                  rr.MOVE_STRENGTH])
        snorlax = apply_plan(
            [rr.MOVE_HEADBUTT, rr.MOVE_AMNESIA, rr.MOVE_REST],
            rr._LEAGUE_LEARN_SNORLAX)
        self.assertEqual(snorlax, [rr.MOVE_DOUBLE_EDGE,
                                   rr.MOVE_AMNESIA,
                                   rr.MOVE_REST,
                                   rr.MOVE_BODY_SLAM])

    def _snap(self, party, active_idx, enemy):
        class Mon:
            def __init__(self, sp, hp, max_hp, moves, pp):
                self.species_id = sp
                self.hp = hp
                self.max_hp = max_hp
                self.moves = moves
                self.pp = pp
                self.status = 0
                self.level = 50

        class Snap:
            pass
        s = Snap()
        s.party = [Mon(*row) for row in party]
        s.active = s.party[active_idx]
        s.active_idx = active_idx
        s.enemy = enemy
        s.alive = lambda sp: any(
            m.species_id == sp and m.hp > 0
            for m in s.party)
        s.mon = lambda sp: next(
            (m for m in s.party if m.species_id == sp),
            None)
        return s

    def test_league_champion_lines(self):
        full = [(rr.SP_LAPRAS, 222, 222,
                 [rr.MOVE_SING, rr.MOVE_ICE_BEAM,
                  rr.MOVE_SURF, rr.MOVE_STRENGTH],
                 [15, 10, 15, 15]),
                (rr.SP_BLASTOISE, 221, 221,
                 [rr.MOVE_HYDRO_PUMP, 39, rr.MOVE_BUBBLE,
                  rr.MOVE_WATER_GUN], [5, 30, 30, 15]),
                (rr.SP_GLOOM, 64, 64,
                 [rr.MOVE_ACID, rr.MOVE_POISON_POWDER,
                  rr.MOVE_CUT, rr.MOVE_STUN_SPORE],
                 [30, 35, 30, 30]),
                (rr.SP_SNORLAX, 236, 236,
                 [rr.MOVE_DOUBLE_EDGE, rr.MOVE_AMNESIA,
                  rr.MOVE_REST, rr.MOVE_BODY_SLAM],
                 [15, 20, 10, 15]),
                (rr.SP_ARCANINE, 180, 180,
                 [rr.MOVE_BITE, rr.MOVE_LEER, rr.MOVE_EMBER],
                 [25, 30, 25]),
                (rr.SP_VOLTORB, 40, 40,
                 [rr.MOVE_TACKLE, 103, 148], [30, 40, 20])]
        fs = rr._LEAGUE_FS
        old_fight = fs.fight
        try:
            fs.fight = "champion"
            fs.reset_fight()
            # Lapras never faces Alakazam: Arcanine's Bite
            # is the answer (the redo's winning line)
            s = self._snap(full, 0,
                           (rr.SP_ALAKAZAM, 140, 140, 59))
            self.assertEqual(rr._league_decide(s, "menu"),
                             ("switch", rr.SP_ARCANINE))
            # Rhydon is Blastoise's Water Gun, not a pump
            s = self._snap(full, 1,
                           (rr.SP_RHYDON, 208, 208, 61))
            self.assertEqual(rr._league_decide(s, "menu"),
                             ("move", rr.MOVE_WATER_GUN))
            # Pidgeot eats Ice Beam, not the last Surf
            s = self._snap(full, 0,
                           (rr.SP_PIDGEOT, 190, 190, 61))
            self.assertEqual(rr._league_decide(s, "menu"),
                             ("move", rr.MOVE_ICE_BEAM))
            # Snorlax alone vs Venusaur with dry slams:
            # never a targetless switch
            dry = [(rr.SP_LAPRAS, 0, 222,
                    full[0][3], full[0][4]),
                   (rr.SP_BLASTOISE, 0, 221,
                    full[1][3], full[1][4]),
                   (rr.SP_GLOOM, 0, 64,
                    full[2][3], full[2][4]),
                   (rr.SP_SNORLAX, 236, 236,
                    full[3][3], [0, 20, 10, 0]),
                   (rr.SP_ARCANINE, 0, 180,
                    full[4][3], full[4][4]),
                   (rr.SP_VOLTORB, 0, 40,
                    full[5][3], full[5][4])]
            s = self._snap(dry, 3,
                           (rr.SP_VENUSAUR, 167, 189, 65))
            act = rr._league_decide(s, "menu")
            self.assertEqual(act[0], "move")
        finally:
            fs.fight = old_fight
            fs.reset_fight()

    def test_league_room_table(self):
        names = [r[0] for r in rr._LEAGUE_ROOMS]
        self.assertEqual(names, ["lorelei", "bruno",
                                 "agatha", "lance",
                                 "champion"])
        # the winning Champion line led Lapras
        self.assertEqual(rr._LEAGUE_ROOMS[4][4],
                         rr.SP_LAPRAS)
        self.assertEqual(rr._LEAGUE_ROOM_IDS,
                         {"LancesRoom": 113})
        self.assertEqual(rr._LEAGUE_PUMP_CAPS,
                         {"lorelei": 0, "bruno": 1,
                          "agatha": 2, "lance": 4,
                          "champion": 5})

    def test_league_fences(self):
        import inspect
        # Vlad's fences, pinned: no ball thrown (the
        # Master Ball stays unthrown), no TM taught, and
        # the stat items are never spent by the leg.
        for fn in (rr.league_grind, rr.league_assemble,
                   rr.elite_four):
            src = inspect.getsource(fn)
            self.assertNotIn("_throw_ball", src)
            self.assertNotIn("teach_", src)
            self.assertNotIn("ITEM_MASTER_BALL", src)
            self.assertNotIn("ITEM_RARE_CANDY", src)
            self.assertNotIn("ITEM_HP_UP", src)
        src = inspect.getsource(rr.league_assemble)
        self.assertIn("45400", src)
        self.assertIn("ITEM_TM27", src)
        src = inspect.getsource(rr.elite_four)
        self.assertIn("Hall of Fame", src)
        self.assertIn("Pallet", src)


class BattleFakeEmu:
    """Fake emu for the probe168 battle helpers: bytearray
    memory + a party of SimpleNamespace mons with moves/pp."""

    def __init__(self, moves, pp, disabled=0):
        self.memory = bytearray(0x10000)
        self.pb = SimpleNamespace(memory=self.memory)
        self.memory[rr.ADDR_PLAYER_DISABLED_MOVE] = disabled
        self._party = [SimpleNamespace(species_id=0x9C,
                                       moves=list(moves),
                                       pp=list(pp))]

    def read_state(self):
        return SimpleNamespace(party=list(self._party))


class TestPickBattleMove(unittest.TestCase):
    # Blastoise's chain moveset: Hydro Pump 56, Tail Whip 39
    # (status), Bubble 145, Water Gun 55.
    MOVES = [56, 39, 145, 55]

    def test_no_disable_picks_last_damaging(self):
        emu = BattleFakeEmu(self.MOVES, [5, 30, 30, 25])
        self.assertEqual(rr._pick_battle_move(emu, 0), 55)

    def test_disabled_move_is_skipped(self):
        # The probe168 Muk stall: Disable locked Water Gun,
        # the driver's usual pick.
        emu = BattleFakeEmu(self.MOVES, [5, 30, 30, 25],
                            disabled=55)
        self.assertEqual(rr._pick_battle_move(emu, 0), 145)

    def test_disabled_and_dry_falls_back(self):
        emu = BattleFakeEmu(self.MOVES, [5, 30, 0, 25],
                            disabled=55)
        self.assertEqual(rr._pick_battle_move(emu, 0), 56)

    def test_all_locked_or_dry_returns_none(self):
        emu = BattleFakeEmu(self.MOVES, [0, 30, 0, 25],
                            disabled=55)
        self.assertIsNone(rr._pick_battle_move(emu, 0))


class TestLearnPromptState(unittest.TestCase):
    def classify(self, text, top_y=3, max_item=0):
        emu = BattleFakeEmu([55], [25])
        emu.memory[rr.ADDR_MENU_TOP_Y] = top_y
        emu.memory[rr.ADDR_MAX_MENU_ITEM] = max_item
        orig = rr._screen_text
        rr._screen_text = lambda e: text
        try:
            return rr._learn_prompt_state(emu)
        finally:
            rr._screen_text = orig

    def test_learn_prompt_is_decline(self):
        self.assertEqual(
            self.classify("VENONAT is trying to learn STUN",
                          max_item=1),
            "decline")

    def test_make_room_variant_is_decline(self):
        self.assertEqual(
            self.classify("But it can only make room by",
                          max_item=1),
            "decline")

    def test_abandon_box_is_confirm(self):
        self.assertEqual(
            self.classify("Abandon learning the move?",
                          max_item=1),
            "confirm_abandon")

    def test_forget_list_is_cancel(self):
        self.assertEqual(
            self.classify("Which move should be forgotten?",
                          top_y=8, max_item=3),
            "cancel_forget")

    def test_evolution_is_flagged(self):
        self.assertEqual(
            self.classify("What? VENONAT is evolving!"),
            "evolving")

    def test_plain_battle_is_none(self):
        self.assertIsNone(
            self.classify("Wild KOFFING appeared!"))


class _StubMon:
    def __init__(self, species_id, hp, moves=(), pp=()):
        self.species_id = species_id
        self.hp = hp
        self.max_hp = hp
        self.moves = list(moves)
        self.pp = list(pp)


class _StubSnap:
    """Mirrors _LeagueSnap's alive/mon interface."""

    def __init__(self, party, active_id, enemy_sp):
        self.party = party
        self.active = next(
            (m for m in party if m.species_id == active_id),
            None)
        self.enemy = (enemy_sp, 100, 100, 60)

    def alive(self, species):
        m = self.mon(species)
        return m is not None and m.hp > 0

    def mon(self, species):
        return next(
            (m for m in self.party
             if m.species_id == species), None)


class TestLeagueGuardSwitch(unittest.TestCase):
    """probe170: _league_menu_move has hardcoded switch
    fallbacks (lance Blastoise vs Gyarados -> Lapras if
    alive else Snorlax) that never check the fallback is
    alive; with it fainted the party list refuses and
    the fight loops to the turn cap."""

    def _snap(self, arcanine_hp=180):
        party = [
            _StubMon(rr.SP_BLASTOISE, 25,
                     moves=[56, 39, 145, 55],
                     pp=[5, 30, 30, 25]),
            _StubMon(rr.SP_LAPRAS, 0),
            _StubMon(rr.SP_SNORLAX, 0),
            _StubMon(rr.SP_ARCANINE, arcanine_hp),
        ]
        return _StubSnap(party, rr.SP_BLASTOISE,
                         rr.SP_GYARADOS)

    def test_fainted_fallback_substituted(self):
        act = rr._league_guard_switch(
            self._snap(), "lance", rr.SP_GYARADOS,
            ("switch", rr.SP_SNORLAX))
        self.assertEqual(act, ("switch", rr.SP_ARCANINE))

    def test_no_living_substitute_attacks(self):
        act = rr._league_guard_switch(
            self._snap(arcanine_hp=0), "lance",
            rr.SP_GYARADOS, ("switch", rr.SP_SNORLAX))
        # strongest PP'd damaging move: Hydro Pump (120)
        self.assertEqual(act, ("move", 56))

    def test_viable_target_passes_through(self):
        party = [
            _StubMon(rr.SP_BLASTOISE, 25,
                     moves=[56], pp=[5]),
            _StubMon(rr.SP_LAPRAS, 100),
        ]
        snap = _StubSnap(party, rr.SP_BLASTOISE,
                         rr.SP_GYARADOS)
        act = rr._league_guard_switch(
            snap, "lance", rr.SP_GYARADOS,
            ("switch", rr.SP_LAPRAS))
        self.assertEqual(act, ("switch", rr.SP_LAPRAS))

    def test_move_action_passes_through(self):
        act = rr._league_guard_switch(
            self._snap(), "lance", rr.SP_GYARADOS,
            ("move", 56))
        self.assertEqual(act, ("move", 56))


class TestLeagueAlakazamFix(unittest.TestCase):
    """probe180: with Arcanine down, the champion branch's
    Alakazam fallbacks formed a switch cycle (Lapras ->
    Snorlax -> Lapras, Blastoise -> Snorlax) and no rule
    ever attacked, so Alakazam Recovered and picked the
    party off over 70+ turns (the S40 fund_e4 blackouts).
    The fallback graph is now acyclic - everyone funnels
    to Snorlax, Snorlax commits - and a general
    anti-oscillation guard converts a fourth no-progress
    switch against one enemy into an attack."""

    LAP = (rr.SP_LAPRAS, 222, 222,
           [rr.MOVE_SING, rr.MOVE_ICE_BEAM, rr.MOVE_SURF,
            rr.MOVE_STRENGTH], [15, 10, 15, 15])
    BLA = (rr.SP_BLASTOISE, 221, 221,
           [rr.MOVE_HYDRO_PUMP, 39, rr.MOVE_BUBBLE,
            rr.MOVE_WATER_GUN], [5, 30, 30, 15])
    SNO = (rr.SP_SNORLAX, 236, 236,
           [rr.MOVE_DOUBLE_EDGE, rr.MOVE_AMNESIA,
            rr.MOVE_REST, rr.MOVE_BODY_SLAM],
           [15, 20, 10, 15])
    ARC = (rr.SP_ARCANINE, 180, 180,
           [rr.MOVE_BITE, rr.MOVE_LEER, rr.MOVE_EMBER],
           [25, 30, 25])

    def _snap(self, party, active_idx, enemy):
        class Mon:
            def __init__(self, sp, hp, max_hp, moves, pp):
                self.species_id = sp
                self.hp = hp
                self.max_hp = max_hp
                self.moves = moves
                self.pp = pp
                self.status = 0
                self.level = 50

        class Snap:
            pass
        s = Snap()
        s.party = [Mon(*row) for row in party]
        s.active = s.party[active_idx]
        s.active_idx = active_idx
        s.enemy = enemy
        s.alive = lambda sp: any(
            m.species_id == sp and m.hp > 0 for m in s.party)
        s.mon = lambda sp: next(
            (m for m in s.party if m.species_id == sp), None)
        return s

    def _party(self, arc_hp=180, sno_hp=236, lap_hp=222,
               bla_hp=221):
        return [(rr.SP_LAPRAS, lap_hp, 222, self.LAP[3],
                 self.LAP[4]),
                (rr.SP_BLASTOISE, bla_hp, 221, self.BLA[3],
                 self.BLA[4]),
                (rr.SP_SNORLAX, sno_hp, 236, self.SNO[3],
                 self.SNO[4]),
                (rr.SP_ARCANINE, arc_hp, 180, self.ARC[3],
                 self.ARC[4])]

    def setUp(self):
        self.fs = rr._LEAGUE_FS
        self.old_fight = self.fs.fight
        self.fs.fight = "champion"
        self.fs.reset_fight()

    def tearDown(self):
        self.fs.fight = self.old_fight
        self.fs.reset_fight()

    def test_lapras_funnels_to_snorlax_when_arcanine_down(
            self):
        s = self._snap(self._party(arc_hp=0), 0,
                       (rr.SP_ALAKAZAM, 140, 140, 59))
        self.assertEqual(rr._league_decide(s, "menu"),
                         ("switch", rr.SP_SNORLAX))

    def test_snorlax_commits_when_arcanine_down(self):
        s = self._snap(self._party(arc_hp=0), 2,
                       (rr.SP_ALAKAZAM, 140, 140, 59))
        self.assertEqual(rr._league_decide(s, "menu"),
                         ("move", rr.MOVE_BODY_SLAM))

    def test_snorlax_rests_when_low(self):
        s = self._snap(self._party(arc_hp=0, sno_hp=80), 2,
                       (rr.SP_ALAKAZAM, 140, 140, 59))
        self.assertEqual(rr._league_decide(s, "menu"),
                         ("move", rr.MOVE_REST))

    def test_blastoise_attacks_when_both_answers_down(self):
        s = self._snap(self._party(arc_hp=0, sno_hp=0), 1,
                       (rr.SP_ALAKAZAM, 140, 140, 59))
        self.assertEqual(rr._league_decide(s, "menu"),
                         ("move", rr.MOVE_HYDRO_PUMP))

    def test_lapras_sings_when_both_answers_down(self):
        s = self._snap(self._party(arc_hp=0, sno_hp=0), 0,
                       (rr.SP_ALAKAZAM, 140, 140, 59))
        self.assertEqual(rr._league_decide(s, "menu"),
                         ("move", rr.MOVE_SING))

    def test_arcanine_still_the_first_answer(self):
        s = self._snap(self._party(), 0,
                       (rr.SP_ALAKAZAM, 140, 140, 59))
        self.assertEqual(rr._league_decide(s, "menu"),
                         ("switch", rr.SP_ARCANINE))

    def test_streak_resets_on_damage_and_new_enemy(self):
        fs = self.fs
        s = self._snap(self._party(), 0,
                       (rr.SP_ALAKAZAM, 140, 140, 59))
        fs.track(s)
        fs.sw_count = 3
        hurt = self._snap(self._party(), 0,
                          (rr.SP_ALAKAZAM, 100, 140, 59))
        fs.track(hurt)
        self.assertEqual(fs.sw_count, 0)
        fs.sw_count = 3
        nxt = self._snap(self._party(), 0,
                         (rr.SP_RHYDON, 200, 200, 61))
        fs.track(nxt)
        self.assertEqual(fs.sw_count, 0)

    def test_fourth_no_progress_switch_becomes_attack(self):
        s = self._snap(self._party(), 0,
                       (rr.SP_ALAKAZAM, 140, 140, 59))
        # first decide registers the enemy instance and
        # switches (streak 1); seed the streak to the cap
        self.assertEqual(rr._league_decide(s, "menu"),
                         ("switch", rr.SP_ARCANINE))
        self.fs.sw_count = 3
        act = rr._league_decide(s, "menu")
        self.assertEqual(act, ("move", rr.MOVE_ICE_BEAM))


class TestStartMenuShowing(unittest.TestCase):
    """probe180: fly_to's start-menu proof is screen text,
    not wTopMenuItemY (a persistent byte a leftover menu
    session leaves stale - the banked post-gauntlet state
    boots into the POKeDEX with the byte at 3)."""

    def _emu_with_text(self, text):
        rev = {}
        for b, ch in rr._LEAGUE_FONT.items():
            rev.setdefault(ch, b)
        mem = bytearray(0x10000)
        from agent.state import ADDR_TILEMAP
        base = ADDR_TILEMAP
        for i, ch in enumerate(text):
            mem[base + i] = rev.get(ch, 0x7F)

        class FakePB:
            pass
        pb = FakePB()
        pb.memory = mem

        class FakeEmu:
            pass
        emu = FakeEmu()
        emu.pb = pb
        return emu

    def test_start_menu_markers(self):
        emu = self._emu_with_text(
            "POKéDEX POKéMON ITEM VLAD SAVE OPTION EXIT")
        self.assertTrue(rr._start_menu_showing(emu))

    def test_dex_contents_is_not_start_menu(self):
        emu = self._emu_with_text(
            "CONTENTS SEEN 142 OWN 124 BULBASAUR IVYSAUR")
        self.assertFalse(rr._start_menu_showing(emu))


class TestHuntPairedPaceDirs(unittest.TestCase):
    """probe171 regression: _hunt_paired referenced
    _PACE_DIRS, which the SS30 port never brought over
    from probe169's catch_lib - a NameError on first
    use. The constant must cover exactly the four unit
    deltas between adjacent hunt cells."""

    def test_pace_dirs_cover_unit_deltas(self):
        self.assertEqual(
            rr._PACE_DIRS,
            {(0, -1): "up", (0, 1): "down",
             (-1, 0): "left", (1, 0): "right"})


def _write_box(mem, base, mons):
    """Fabricate one PC box in a flat memory image. Each
    mon: (species, level, moves, ot_id, exp, dvs)."""
    mem[base] = len(mons)
    for i, (sp, lvl, moves, ot, exp, dvs) in enumerate(mons):
        mem[base + 1 + i] = sp
        s = base + 22 + i * 33
        mem[s + 3] = lvl
        for j, mv in enumerate(moves):
            mem[s + 8 + j] = mv
        mem[s + 12] = ot >> 8
        mem[s + 13] = ot & 0xFF
        mem[s + 14] = (exp >> 16) & 0xFF
        mem[s + 15] = (exp >> 8) & 0xFF
        mem[s + 16] = exp & 0xFF
        mem[s + 27] = dvs[0]
        mem[s + 28] = dvs[1]


class TestBoxCensus(unittest.TestCase):
    """probe172 port: _box_census reads the CURRENT box
    from WRAM and the other eleven from SRAM - the SRAM
    copy of the current box can lag (leg 4's box-3 filings
    were live in WRAM while SRAM box 3 read empty). The
    flat fake memory cannot model bank switching, so the
    SRAM window simply holds whatever was written at the
    slot offsets; the routing (current from WRAM, box 1
    from the bank-2 slot 0 offset) is what is pinned."""

    def _emu(self):
        emu = FakeEmu()
        mem = emu.memory
        # current box = 4 (stored value is 0-based)
        mem[rr.ADDR_CURRENT_BOX_NUM] = 3
        _write_box(mem, rr.ADDR_BOX_COUNT, [
            (0x08, 37, [29, 45, 55, 110], 25522, 50653,
             [72, 115])])
        # SRAM bank-2 slot 0 = box 1
        _write_box(mem, 0xA000, [
            (0x0B, 37, [23, 50, 111, 21], 14261, 50653,
             [52, 190])])
        return emu

    def test_current_box_comes_from_wram(self):
        census = rr._box_census(self._emu())
        self.assertEqual(sorted(census), list(range(1, 13)))
        bro = census[4]
        self.assertEqual(len(bro), 1)
        self.assertEqual(bro[0]["species"], 0x08)
        self.assertEqual(bro[0]["name"], "SLOWBRO")
        self.assertEqual(bro[0]["level"], 37)
        self.assertEqual(bro[0]["moves"], [29, 45, 55, 110])
        self.assertEqual(bro[0]["ot_id"], 25522)
        self.assertEqual(bro[0]["exp"], 50653)
        self.assertEqual(bro[0]["dvs"], [72, 115])

    def test_other_box_comes_from_sram(self):
        census = rr._box_census(self._emu())
        lick = census[1]
        self.assertEqual(len(lick), 1)
        self.assertEqual(lick[0]["species"], 0x0B)
        self.assertEqual(lick[0]["ot_id"], 14261)

    def test_find_species_tags_the_box(self):
        census = rr._box_census(self._emu())
        hits = rr._box_find_species(census, 0x08)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["box"], 4)
        self.assertEqual(rr._box_find_species(census, 0x7F),
                         [])

    def test_empty_box_parses_empty(self):
        emu = FakeEmu()
        emu.memory[rr.ADDR_CURRENT_BOX_NUM] = 0
        census = rr._box_census(emu)
        self.assertEqual(census[1], [])

    def test_all_box_species_spans_every_box(self):
        # probe173: the owned set must come from all 12
        # boxes, not just the current one (_box_species)
        self.assertEqual(rr._all_box_species(self._emu()),
                         {0x08, 0x0B})


class TestPartyNeedsHeal(unittest.TestCase):
    """probe172 port: the raw-WRAM hurt check that gates
    a redundant heal_at_pokecenter call."""

    def _emu(self, hps):
        emu = FakeEmu()
        mem = emu.memory
        mem[0xD163] = len(hps)
        for i, (hp, mx) in enumerate(hps):
            base = 0xD16B + 44 * i
            mem[base + 1] = hp >> 8
            mem[base + 2] = hp & 0xFF
            mem[base + 34] = mx >> 8
            mem[base + 35] = mx & 0xFF
        return emu

    def test_full_party_needs_no_heal(self):
        self.assertFalse(rr._party_needs_heal(
            self._emu([(250, 250), (36, 36)])))

    def test_one_hurt_mon_needs_heal(self):
        self.assertTrue(rr._party_needs_heal(
            self._emu([(250, 250), (12, 36)])))

    def test_empty_party_needs_no_heal(self):
        self.assertFalse(rr._party_needs_heal(self._emu([])))


class TestGen1DamageBound(unittest.TestCase):
    """probe174 port: the Strength gate on Zapdos used
    worst-case Gen 1 damage bounds computed from the live
    battle-struct stats. The live reads (L60 Lapras with
    stat-EXP Attack 164 vs the bird's Defense 101) are
    pinned exactly: non-crit ceiling 69, crit ceiling
    131 - the first Strength at 156 HP was provably
    non-lethal, the second at ~98 HP was provably not
    attempted."""

    def test_live_zapdos_numbers(self):
        self.assertEqual(
            rr._gen1_damage_bound(60, 80, 164, 101), 69)
        self.assertEqual(
            rr._gen1_damage_bound(60, 80, 164, 101,
                                  crit=True), 131)

    def test_crit_always_exceeds_base(self):
        for lv in (5, 36, 50, 100):
            self.assertGreater(
                rr._gen1_damage_bound(lv, 40, 90, 90,
                                      crit=True),
                rr._gen1_damage_bound(lv, 40, 90, 90))


class TestChipBoundsLive(unittest.TestCase):
    """leg 11 port (probe178): _chip_bounds_live is
    _strength_bounds_live with a caller-set Def range -
    Mewtwo's live Def (~131-152) sits outside the
    Zapdos window (80-120), so the generalised helper
    must (a) compute the same Gen 1 bounds when the
    read is inside the caller's range and (b) return
    the never-attack bound (0, 9999) outside it."""

    def _emu(self, atk, dfn, level=62):
        mem = bytearray(0x10000)
        base = rr.ADDR_BATTLE_MON_SPECIES
        mem[base + 17] = (atk >> 8) & 0xFF
        mem[base + 18] = atk & 0xFF
        ebase = rr.ADDR_ENEMY_SPECIES
        mem[ebase + 19] = (dfn >> 8) & 0xFF
        mem[ebase + 20] = dfn & 0xFF

        class _PB:
            memory = mem

        class _Mon:
            pass

        mon = _Mon()
        mon.level = level

        class _State:
            party = [mon]

        class _Emu:
            pb = _PB()

            def read_state(self):
                return _State()

        return _Emu()

    def test_in_range_matches_damage_bound(self):
        emu = self._emu(170, 140)
        base, crit = rr._chip_bounds_live(emu, 80,
                                          (125, 160))
        self.assertEqual(
            base, rr._gen1_damage_bound(62, 80, 170, 140))
        self.assertEqual(
            crit, rr._gen1_damage_bound(62, 80, 170, 140,
                                        crit=True))

    def test_out_of_range_never_attacks(self):
        emu = self._emu(170, 140)
        self.assertEqual(
            rr._chip_bounds_live(emu, 80, (80, 120)),
            (0, 9999))

    def test_mewtwo_window_admits_higher_def(self):
        emu = self._emu(170, 152)
        base, crit = rr._chip_bounds_live(emu, 80,
                                          (125, 160))
        self.assertGreater(crit, base)
        self.assertLess(crit, 9999)


class TestEnemyLive(unittest.TestCase):
    """probe174 port: _enemy_live reads the enemy from
    the BATTLE struct (wEnemyMon). _battle_enemy's level
    / max-HP come from the party-struct copy and read
    garbage mid-battle (L125 / 984 against the real
    L50 / 156); the battle-struct offsets are pinned
    here: species +0, HP +1, level +14, max HP +15."""

    def test_reads_battle_struct(self):
        emu = FakeEmu()
        mem = emu.memory
        base = rr.ADDR_ENEMY_SPECIES
        mem[base] = 0x4B                    # ZAPDOS
        mem[base + 1] = 0
        mem[base + 2] = 98                  # HP 98
        mem[base + 14] = 50                 # level
        mem[base + 15] = 0
        mem[base + 16] = 156                # max HP
        self.assertEqual(rr._enemy_live(emu),
                         (0x4B, 98, 156, 50))


class TestReleaseHeldButtons(unittest.TestCase):
    """probe174 port: a save state can carry a held
    direction in the emulated joypad - the player drifts
    on load (the pre-engagement checkpoint drifted west
    into a wild battle; the post-catch state drifted out
    of the Power Plant). release_held_buttons must free
    all eight buttons and settle."""

    def test_releases_everything_and_settles(self):
        released = []
        ticks = []

        class PB:
            def button_release(self, b):
                released.append(b)

        emu = SimpleNamespace(pb=PB(),
                              tick=lambda n=1: ticks.append(n))
        rr.release_held_buttons(emu)
        self.assertEqual(
            sorted(released),
            ["a", "b", "down", "left", "right", "select",
             "start", "up"])
        self.assertTrue(ticks and ticks[0] >= 30)


class TestGameCorner(unittest.TestCase):
    """probe175 (leg 8): coin BCD decode, the slot machine
    table and the lucky-machine read are pure functions of
    WRAM; the sessions themselves are live-verified in the
    probe (17,728 spins, both prizes redeemed)."""

    def _emu_with_coins(self, b0, b1):
        emu = FakeEmu()
        emu.memory[rr.ADDR_PLAYER_COINS] = b0
        emu.memory[rr.ADDR_PLAYER_COINS + 1] = b1
        return emu

    def test_player_coins_zero(self):
        self.assertEqual(rr.player_coins(self._emu_with_coins(0, 0)), 0)

    def test_player_coins_seed_330(self):
        emu = self._emu_with_coins(0x03, 0x30)
        self.assertEqual(rr.player_coins(emu), 330)

    def test_player_coins_cap_9999(self):
        emu = self._emu_with_coins(0x99, 0x99)
        self.assertEqual(rr.player_coins(emu), 9999)

    def test_player_coins_post_scyther_4469(self):
        emu = self._emu_with_coins(0x44, 0x69)
        self.assertEqual(rr.player_coins(emu), 4469)

    def test_machine_table_shape(self):
        self.assertEqual(len(rr.SLOT_MACHINE_CELLS), 36)
        self.assertEqual(rr.SLOT_UNPLAYABLE, frozenset({5, 8, 27}))
        # the shadowed hidden coin shares (12, 15) with
        # machine index 12 - the machine event wins the scan
        self.assertEqual(rr.SLOT_MACHINE_CELLS[12], (12, 15))

    def test_stance_west_column_plays_from_west(self):
        self.assertEqual(rr.slot_machine_stance(2), ((17, 13), "right"))
        self.assertEqual(rr.slot_machine_stance(16), ((11, 11), "right"))

    def test_stance_east_column_plays_from_east(self):
        self.assertEqual(rr.slot_machine_stance(9), ((14, 13), "left"))
        self.assertEqual(rr.slot_machine_stance(30), ((2, 10), "left"))

    def test_lucky_slot_index(self):
        emu = FakeEmu()
        emu.memory[rr.ADDR_LUCKY_SLOT] = 10
        self.assertEqual(rr.lucky_slot_index(emu), 9)
        emu.memory[rr.ADDR_LUCKY_SLOT] = 6   # SOMEONESKEYS machine
        self.assertIsNone(rr.lucky_slot_index(emu))
        emu.memory[rr.ADDR_LUCKY_SLOT] = 0   # no lucky machine
        self.assertIsNone(rr.lucky_slot_index(emu))
