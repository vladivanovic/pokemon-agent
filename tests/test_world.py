import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.world import default_world


class TestWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = default_world()

    def test_all_maps_loaded(self):
        self.assertGreater(len(self.w.maps), 200)

    def test_checklist_map_names_exist(self):
        for name in ("PalletTown", "OaksLab", "ViridianMart", "PewterCity",
                     "PewterGym", "CeruleanGym", "VermilionGym", "CeladonGym",
                     "FuchsiaGym", "SaffronGym", "CinnabarGym", "ViridianGym",
                     "IndigoPlateau", "PowerPlant"):
            self.w.map_id(name)  # raises KeyError if missing

    def test_pallet_is_connected(self):
        # From outside Red's house to the lab door, and to the north exit
        # (the gap in the fence line is at x=10-11).
        self.assertIsNotNone(self.w.tile_path(0, (5, 6), (12, 11)))
        path = self.w.tile_path(0, (5, 6), (10, 0))
        self.assertIsNotNone(path)

    def test_route_pallet_to_pewter(self):
        r = self.w.route(0, self.w.map_id("PewterCity"))
        self.assertIsNotNone(r)
        self.assertEqual(r.maps[0], 0)
        self.assertEqual(r.maps[-1], self.w.map_id("PewterCity"))

    def test_warp_route_into_lab(self):
        r = self.w.route(0, self.w.map_id("OaksLab"))
        self.assertIsNotNone(r)
        self.assertEqual(len(r.legs), 1)
        self.assertEqual(r.legs[0].via, "warp")

    def test_exit_warp_route_out_of_lab(self):
        r = self.w.route(self.w.map_id("OaksLab"), 0)
        self.assertIsNotNone(r)
        self.assertIn("exit", r.legs[0].via)

    def test_cinnabar_needs_surf(self):
        cin = self.w.map_id("CinnabarIsland")
        self.assertIsNone(self.w.route(0, cin))
        self.assertIn("surf", self.w.missing_capabilities(0, cin))
        self.assertIsNotNone(self.w.route(0, cin, frozenset({"surf"})))

    def test_power_plant_needs_surf(self):
        r10 = self.w.map_id("Route10")
        pp = self.w.map_id("PowerPlant")
        self.assertIsNone(self.w.route(r10, pp))
        self.assertIsNotNone(self.w.route(r10, pp, frozenset({"surf"})))

    def test_route1_walks_around_ledges(self):
        # Route 1's ledge rows block the direct line north, but a walking
        # route exists around them (their face tiles are impassable in
        # the grid; the side paths are not).
        r1 = self.w.map_id("Route1")
        gm = self.w.maps[r1]
        south = (10, gm.h - 2)
        north = (10, 1)
        self.assertIsNotNone(self.w.tile_path(r1, south, north))

    def test_paths_do_not_step_through_doors(self):
        # Red's house door is a warp tile at (5, 5) in Pallet Town: a path
        # from below the house to the north exit must go around it, not
        # through it (stepping on it would teleport into the house).
        path = self.w.tile_path(0, (5, 6), (10, 0))
        self.assertIsNotNone(path)
        self.assertNotIn((5, 5), path[1:-1])
        # ...but a path may END on a door (that's how you enter).
        self.assertEqual(self.w.tile_path(0, (5, 6), (12, 11))[-1], (12, 11))

    def test_journey_to_pewter_goes_through_the_forest(self):
        # Route 2's north edge cannot be walked to directly: the journey
        # must chain through the Viridian Forest gatehouses and forest.
        j = self.w.journey(0, (5, 6), self.w.map_id("PewterCity"))
        self.assertIsNotNone(j)
        maps_walked = {s[1] for s in j if s[0] == "walk"}
        self.assertIn(self.w.map_id("ViridianForest"), maps_walked)

    def test_route4_ledge_drops_reach_cerulean_side(self):
        # Route 4 east of Mt. Moon: the only way down to the Cerulean
        # edge is hopping ledges. From the cave-exit landing, a path to
        # the east edge must exist and must pass through a ledge face.
        r4 = self.w.map_id("Route4")
        gm = self.w.maps[r4]
        self.assertTrue(gm.ledges, "Route 4 ledge data missing")
        path = self.w.tile_path(r4, (24, 5), (89, 10))
        self.assertIsNotNone(path)
        self.assertTrue(any(t in gm.ledges for t in path),
                        "path east should hop at least one ledge")

    def test_journey_pewter_gym_to_cerulean_gym(self):
        # The full Cascade approach: out of the gym, Route 3, Mt. Moon
        # (in the west door, out the B1F east exit), Route 4's ledges,
        # Cerulean, and into the gym's map.
        j = self.w.journey(self.w.map_id("PewterGym"), (4, 2),
                           self.w.map_id("CeruleanGym"))
        self.assertIsNotNone(j)
        maps_walked = {s[1] for s in j if s[0] == "walk"}
        self.assertIn(self.w.map_id("MtMoonB1F"), maps_walked)
        self.assertIn(self.w.map_id("CeruleanCity"), maps_walked)

    def test_tile_paths_from_honours_blocked_tiles(self):
        # Learned walls (the Executor's safety net) flow through the
        # journey BFS: wall off a cell on the direct line through a
        # Mt. Moon B1F room and the target must still be reachable,
        # the long way.
        b1 = self.w.map_id("MtMoonB1F")
        plain = self.w.tile_path(b1, (20, 8), (20, 11))
        self.assertIsNotNone(plain)
        self.assertIn((20, 10), plain)  # the straight line south
        found = self.w.tile_paths_from(b1, (20, 8), {(20, 11)},
                                       blocked=frozenset({(20, 10)}))
        self.assertIn((20, 11), found)
        self.assertNotIn((20, 10), found[(20, 11)])

    def test_mt_moon_b1f_exit_chamber_only_via_b2f(self):
        # The B1F exit chamber (27, 3) is walled off from the floor's
        # south side by solid rock; the only way up is the B2F
        # detour. (The old open-grid data planned a straight walk
        # through the rock - the root of the Mt. Moon hardware saga.)
        b1 = self.w.map_id("MtMoonB1F")
        self.assertIsNone(self.w.tile_path(b1, (25, 15), (27, 3)))
        steps = self.w.journey(self.w.map_id("PewterGym"), (4, 2),
                               self.w.map_id("CeruleanGym"))
        self.assertIsNotNone(steps)
        walked = [s[1] for s in steps if s[0] == "walk"]
        self.assertIn(self.w.map_id("MtMoonB2F"), walked)

    def test_journey_from_inside_mt_moon_tolerates_unmodelled_maps(self):
        # Regression (found live): starting inside Mt. Moon, the BFS
        # expands far enough to meet warps whose destination map is
        # not in the data (the Underground Path interior, id 119).
        # That must read as a dead end, not raise KeyError.
        j = self.w.journey(self.w.map_id("MtMoon1F"), (11, 22),
                           self.w.map_id("CeruleanGym"))
        self.assertIsNotNone(j)


if __name__ == "__main__":
    unittest.main()
