"""Tests for agent/dex_audit.py (ported from probe168).

The audit parses a pret/pokered checkout, which is not part
of this repo; skip unless one is present at the conventional
path. When it runs, it pins the numbers the probe168 sweep
was planned against: 151 species, ceiling 124, no anomalies.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POKERED = Path.home() / "workspace" / "projects" / "pokered"


@unittest.skipUnless((POKERED / "constants").is_dir(),
                     "no pokered checkout")
class TestDexAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agent.dex_audit import build_ledger
        cls.ledger = build_ledger(POKERED)

    def test_ceiling_124(self):
        self.assertEqual(self.ledger["ceiling"], 124)

    def test_all_species_classified(self):
        self.assertEqual(len(self.ledger["species"]), 151)
        self.assertEqual(self.ledger["anomalies"], [])

    def test_unobtainable_count_25_plus_eevee_caveat(self):
        statuses = [e["status"]
                    for e in self.ledger["species"].values()]
        self.assertEqual(statuses.count("unobtainable"), 25)
        self.assertEqual(statuses.count("choice-locked"), 3)

    def test_porygon_is_a_9999_coin_prize(self):
        self.assertEqual(self.ledger["prizes"]["PORYGON"]["coins"],
                         9999)


if __name__ == "__main__":
    unittest.main()
