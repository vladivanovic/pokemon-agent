import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.emu import SimEmulator
from agent.executor import Executor
from agent.memory import EpisodicMemory
from agent.planner import ChecklistPlanner
from agent.world import default_world


class DemoEmulator(SimEmulator):
    def press(self, button):
        super().press(button)
        if self.map_id == self.world.map_id("OaksLab") \
                and not self.read_state().party:
            self.give_starter()


class TestExecutor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = default_world()

    def test_opening_run_reaches_pewter(self):
        emu = DemoEmulator(self.w, 0, (5, 6))
        ex = Executor(emu, self.w, ChecklistPlanner(), EpisodicMemory())
        result = ex.run(max_goals=4)
        self.assertEqual(result.goals_done, 4, result.log)
        state = emu.read_state()
        self.assertEqual(state.map_id, self.w.map_id("PewterCity"))
        self.assertEqual(len(state.party), 1)  # starter collected on the way

    def test_battle_is_handled(self):
        emu = DemoEmulator(self.w, 0, (5, 6))
        emu.give_starter()
        emu.force_battle(presses_to_win=3)

        class DonePlanner:
            def propose(self, state, world, memory):
                return None

        ex = Executor(emu, self.w, DonePlanner(), EpisodicMemory())
        result = ex.run(max_goals=1)
        self.assertEqual(result.status, "complete")
        self.assertFalse(emu.read_state().in_battle)

    def test_blocked_door_reports_stuck(self):
        emu = DemoEmulator(self.w, 0, (5, 6))
        emu.blocked.add((0, 12, 11))  # lab door tile
        ex = Executor(emu, self.w, ChecklistPlanner(), EpisodicMemory(),
                      stuck_threshold=6)
        result = ex.run(max_goals=1)
        self.assertEqual(result.status, "stuck", result.log)


if __name__ == "__main__":
    unittest.main()
