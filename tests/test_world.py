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
        # From outside Red's house to the lab door, and to the north exit.
        self.assertIsNotNone(self.w.tile_path(0, (5, 6), (12, 11)))
        path = self.w.tile_path(0, (5, 6), (9, 0))
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

    def test_ledge_data_is_recorded(self):
        # Ledge positions are preserved in the model even though the
        # static layer does not enforce one-way movement (see world.py).
        total = sum(len(gm.ledges) for gm in self.w.maps.values())
        self.assertGreater(total, 0)

    def test_paths_do_not_step_through_doors(self):
        # Red's house door is a warp tile at (5, 5) in Pallet Town: a path
        # from below the house to the north exit must go around it, not
        # through it (stepping on it would teleport into the house).
        path = self.w.tile_path(0, (5, 6), (9, 0))
        self.assertIsNotNone(path)
        self.assertNotIn((5, 5), path[1:-1])
        # ...but a path may END on a door (that's how you enter).
        self.assertEqual(self.w.tile_path(0, (5, 6), (12, 11))[-1], (12, 11))


if __name__ == "__main__":
    unittest.main()
