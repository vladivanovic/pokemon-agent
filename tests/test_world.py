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

    def test_duplicate_map_names_resolve_to_canonical_ids(self):
        # pokered ships _COPY maps that reuse the original's header
        # (CERULEAN_TRASHED_HOUSE_COPY $45, UNDERGROUND_PATH_ROUTE_6_
        # COPY $4B) plus an unused Saffron duplicate. Name lookup must
        # give the canonical (lowest) id - the game's current-map
        # byte - or executor walks inside the real map silently
        # no-op, reading the id mismatch as a warp exit.
        self.assertEqual(self.w.map_id("CeruleanTrashedHouse"), 62)
        self.assertEqual(self.w.map_id("UndergroundPathRoute6"), 74)
        self.assertEqual(self.w.map_id("SaffronCity"), 10)

    def test_underground_path_tunnel_is_modelled(self):
        # Map 119 (UndergroundPathNorthSouth) was missing from the
        # extracted data: pokered's map_const overstates its height
        # (24 rows vs the .blk's 23), and the builder skipped the
        # map outright. The tunnel journey must exist end to end.
        h5 = self.w.map_id("UndergroundPathRoute5")
        h6 = self.w.map_id("UndergroundPathRoute6")
        tunnel = self.w.map_id("UndergroundPathNorthSouth")
        self.assertEqual(tunnel, 119)
        r = self.w.route(h5, h6)
        self.assertIsNotNone(r)
        self.assertEqual([leg.dst for leg in r.legs], [119, h6])
        j = self.w.journey(h5, (3, 7), h6)
        self.assertIsNotNone(j)

    def test_cerulean_south_journey_goes_through_trashed_house(self):
        # Cerulean's tree line is sealed: the only walking route to
        # Route 5 runs through the robbed house, out its back hole
        # into the yard, and around the east bank (the yard segment
        # shows up as the house's walk steps in the journey).
        cc = self.w.map_id("CeruleanCity")
        house = self.w.map_id("CeruleanTrashedHouse")
        j = self.w.journey(cc, (19, 18), self.w.map_id("Route5"))
        self.assertIsNotNone(j)
        walked = [s[1] for s in j if s[0] == "walk"]
        self.assertIn(house, walked)

    def test_journey_through_digletts_cave_both_ways(self):
        # Regression (found live twice, fixed 2026-11-05): a journey
        # that must pass THROUGH Diglett's Cave mid-route planned to
        # nothing - "no journey (portal-level route) to target" killed
        # the wired supervisor's Viridian proposals in the live run.
        # Root cause: the journey BFS tracked wLastMap by the generic
        # WarpFound rule only (set when warping from an outside map),
        # but each cave lobby's map script assigns wLastMap to its OWN
        # outside map on entry (scripts/DiglettsCaveRoute2.asm ->
        # ROUTE_2, ...Route11.asm -> ROUTE_11). With the tracked value
        # stuck at the entry side, the far lobby's exit warp resolved
        # back to the entry map and Route 2 was never generated.
        vermilion = self.w.map_id("VermilionCity")
        route2 = self.w.map_id("Route2")
        cave = self.w.map_id("DiglettsCave")
        # Vermilion (open city tile; the gym pocket is tree-sealed in
        # the grid - a separate known gap) through the cave to Route 2.
        j = self.w.journey(vermilion, (11, 4), route2)
        self.assertIsNotNone(j)
        walked = [s[1] for s in j if s[0] == "walk"]
        self.assertIn(cave, walked)
        self.assertIn(self.w.map_id("DiglettsCaveRoute2"), walked)
        # ...and back again: the far lobby's exit must now resolve to
        # Route 11 / Vermilion, not bounce back to Route 2.
        j = self.w.journey(route2, (12, 10), vermilion)
        self.assertIsNotNone(j)
        walked = [s[1] for s in j if s[0] == "walk"]
        self.assertIn(cave, walked)
        self.assertIn(self.w.map_id("DiglettsCaveRoute11"), walked)

    def test_journey_cold_start_inside_digletts_cave(self):
        # Starting inside the cave (no tracked wLastMap at all - the
        # static guess gives up: two outside mouths), journeys out of
        # BOTH mouths must still plan, because entering a lobby
        # assigns wLastMap by script before any exit is taken.
        cave = self.w.map_id("DiglettsCave")
        self.assertIsNotNone(
            self.w.journey(cave, (37, 30), self.w.map_id("Route2")))
        self.assertIsNotNone(
            self.w.journey(cave, (6, 6), self.w.map_id("Route11")))

    def test_last_map_script_lobby_exits_to_its_own_outside_map(self):
        # The narrow regression: standing in the Route 2 lobby (say we
        # just walked in from the cave), its exit must land on Route 2
        # even though nothing in the journey came from Route 2.
        lobby2 = self.w.map_id("DiglettsCaveRoute2")
        j = self.w.journey(lobby2, (4, 5), self.w.map_id("Route2"))
        self.assertIsNotNone(j)
        # One walk step onto the exit tile, no maps in between.
        self.assertEqual([s[1] for s in j if s[0] == "walk"], [lobby2])

    def test_ordinary_building_exit_still_returns_to_its_town(self):
        # The generic LAST_MAP rule must not regress: a single-mouth
        # building still exits to the town it hangs off.
        mart = self.w.map_id("ViridianMart")
        j = self.w.journey(mart, (3, 5), self.w.map_id("ViridianCity"))
        self.assertIsNotNone(j)
        gym = self.w.map_id("VermilionGym")
        j = self.w.journey(gym, (5, 2), self.w.map_id("VermilionCity"))
        self.assertIsNotNone(j)

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
