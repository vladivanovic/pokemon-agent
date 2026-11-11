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
