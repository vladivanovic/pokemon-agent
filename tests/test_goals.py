import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.emu import SimEmulator
from agent.goals import Goal, validate_goal
from agent.memory import EpisodicMemory, state_signature
from agent.world import default_world


class TestGoals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = default_world()

    def _emu(self):
        return SimEmulator(self.w, 0, (5, 6))

    def test_reachable_goal_validates(self):
        emu = self._emu()
        goal = Goal("goto", self.w.map_id("ViridianCity"))
        v = validate_goal(goal, emu.read_state(), self.w, EpisodicMemory())
        self.assertTrue(v.ok, v.problems)

    def test_surf_goal_rejected_with_reason(self):
        emu = self._emu()
        goal = Goal("goto", self.w.map_id("CinnabarIsland"))
        v = validate_goal(goal, emu.read_state(), self.w, EpisodicMemory())
        self.assertFalse(v.ok)
        self.assertIn("surf", v.missing_capabilities)

    def test_surf_goal_validates_with_capability(self):
        emu = self._emu()
        emu.give_starter(moves=(57,))       # Surf
        emu.give_badge("Soul")
        goal = Goal("goto", self.w.map_id("CinnabarIsland"))
        v = validate_goal(goal, emu.read_state(), self.w, EpisodicMemory())
        self.assertTrue(v.ok, v.problems)

    def test_repeated_failure_is_suppressed(self):
        emu = self._emu()
        mem = EpisodicMemory()
        state = emu.read_state()
        goal = Goal("goto", self.w.map_id("ViridianCity"))
        sig = state_signature(state)
        mem.record_goal_result(goal.key(), False, sig, "boom")
        mem.record_goal_result(goal.key(), False, sig, "boom again")
        v = validate_goal(goal, state, self.w, mem)
        self.assertFalse(v.ok)
        self.assertTrue(any("suppressing" in p for p in v.problems))

    def test_unknown_map_rejected(self):
        emu = self._emu()
        v = validate_goal(Goal("goto", 9999), emu.read_state(), self.w,
                          EpisodicMemory())
        self.assertFalse(v.ok)


if __name__ == "__main__":
    unittest.main()
