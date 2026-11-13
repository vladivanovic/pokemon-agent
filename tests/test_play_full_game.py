"""Tests for play_full_game.py (runner mechanics, no emulator) and
web_viewer.state_to_dict. The game itself is exercised by the smoke
runs, not here: these pin the phase table's order, the checkpoint
naming, and the retry-then-stop contract."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import play_full_game as fg
from web_viewer import state_to_dict

EXPECTED_ORDER = [
    "intro", "starter", "boulder", "cascade", "vermilion",
    "rock-tunnel", "celadon", "hideout", "tower", "gates", "silph",
    "sabrina", "dojo", "fuchsia", "lapras-surf", "safari-hunt",
    "teach-strength", "seafoam", "mansion-blaine", "super-rod",
    "dratini", "fly-journey", "bicycle", "snorlax-journey", "snorlax",
    "fly-house", "teach-fly", "viridian-gym", "r22-prep-grind",
    "r22-rival", "victory-road", "league-grind", "league-assemble",
    "elite-four",
]

# Badge phases in the order the completed save line earned them
# (Marsh/Sabrina before Soul/Fuchsia - sections 17-19).
BADGE_PHASES = ["boulder", "cascade", "vermilion", "celadon",
                "sabrina", "fuchsia", "mansion-blaine", "viridian-gym"]


class FakeState:
    map_id = 0
    x = 3
    y = 4
    in_battle = False
    money = 1234
    badges = {"Boulder"}
    party = [SimpleNamespace(species_id=0xB1, level=5, hp=20,
                             max_hp=20)]


class FakeCtx:
    def __init__(self):
        self.notes = []
        self.loads = []
        self.saves = []
        self.reboots = 0

    def note(self, msg):
        self.notes.append(msg)

    def state(self):
        return FakeState()

    def save_state(self, path):
        Path(path).write_bytes(b"state")
        self.saves.append(str(path))

    def load_state(self, path):
        self.loads.append(str(path))

    def reboot(self):
        self.reboots += 1


def fake_phase(name, run, verify=None):
    return fg.Phase(name, name, run, verify)


class PhaseTableTests(unittest.TestCase):
    def test_phase_order_matches_storyline(self):
        names = [p.name for p in fg.build_phases()]
        self.assertEqual(names, EXPECTED_ORDER)

    def test_badge_phases_in_canonical_order(self):
        names = [p.name for p in fg.build_phases()]
        idx = [names.index(b) for b in BADGE_PHASES]
        self.assertEqual(idx, sorted(idx))

    def test_every_phase_has_verify(self):
        for p in fg.build_phases():
            self.assertIsNotNone(p.verify, p.name)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ctx = FakeCtx()

    def test_checkpoint_naming_and_discovery(self):
        phases = [fake_phase("alpha", lambda c: True),
                  fake_phase("beta", lambda c: True)]
        r = fg.Runner(self.ctx, phases, self.tmp.name)
        self.assertEqual(r.checkpoint_path(0).name, "01-alpha.state")
        self.assertEqual(r.checkpoint_path(1).name, "02-beta.state")
        self.assertTrue(r.run())
        found = fg.discover_checkpoints(self.tmp.name)
        self.assertEqual([i for i, _ in found], [0, 1])
        self.assertEqual(found[1][1].name, "02-beta.state")

    def test_retry_then_pass_uses_start_checkpoint(self):
        calls = {"n": 0}

        def flaky(ctx):
            calls["n"] += 1
            return calls["n"] >= 3  # fail, fail, pass

        phases = [fake_phase("one", lambda c: True),
                  fake_phase("two", flaky)]
        r = fg.Runner(self.ctx, phases, self.tmp.name, max_retries=2)
        self.assertTrue(r.run())
        self.assertEqual(calls["n"], 3)
        # both retries reloaded phase one's checkpoint
        self.assertEqual(len(self.ctx.loads), 2)
        self.assertTrue(all("01-one.state" in p
                            for p in self.ctx.loads))

    def test_retry_exhausted_stops_and_never_skips(self):
        ran = {"later": False}

        def later(ctx):
            ran["later"] = True
            return True

        phases = [fake_phase("bad", lambda c: False),
                  fake_phase("later", later)]
        r = fg.Runner(self.ctx, phases, self.tmp.name, max_retries=2)
        self.assertFalse(r.run())
        self.assertFalse(ran["later"])
        # phase 0 has no start checkpoint: retries reboot fresh
        self.assertEqual(self.ctx.reboots, 2)

    def test_verify_failure_counts_as_failure(self):
        phases = [fake_phase("v", lambda c: True,
                             verify=lambda c: (False, "no badge"))]
        r = fg.Runner(self.ctx, phases, self.tmp.name, max_retries=1)
        self.assertFalse(r.run())
        self.assertEqual(self.ctx.saves, [])

    def test_start_index_skips_earlier_phases(self):
        phases = [fake_phase("a", lambda c: True),
                  fake_phase("b", lambda c: True)]
        r = fg.Runner(self.ctx, phases, self.tmp.name, start_index=1)
        self.assertTrue(r.run())
        self.assertEqual(len(self.ctx.saves), 1)
        self.assertIn("02-b.state", self.ctx.saves[0])


class SerializerTests(unittest.TestCase):
    def test_state_to_dict(self):
        world = SimpleNamespace(map_name=lambda i: f"Map{i}")
        d = state_to_dict(FakeState(), world,
                          {"phase": "boulder", "run_status": "running"})
        self.assertEqual(d["map"], "Map0")
        self.assertEqual(d["badges"], ["Boulder"])
        self.assertEqual(d["money"], 1234)
        self.assertEqual(d["party"][0]["species_id"], 0xB1)
        self.assertEqual(d["party"][0]["level"], 5)
        self.assertEqual(d["phase"], "boulder")

    def test_state_to_dict_no_world(self):
        d = state_to_dict(FakeState())
        self.assertIsNone(d["map"])
        self.assertEqual(d["map_id"], 0)


if __name__ == "__main__":
    unittest.main()
