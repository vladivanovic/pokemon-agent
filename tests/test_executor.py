import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.emu import SimEmulator
from agent.executor import Executor, SmartFightPolicy
from agent.memory import EpisodicMemory
from agent.planner import ChecklistPlanner
from agent.state import GameState, PartyMon
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


def battle_state(moves, pp, top_y=0, item=0, move_idx=0):
    mon = PartyMon(level=10, hp=30, max_hp=30, moves=moves, pp=pp)
    return GameState(map_id=0, x=0, y=0, in_battle=True,
                     badges=frozenset(), money=0, party=[mon],
                     menu_top_y=top_y, menu_item=item,
                     move_list_index=move_idx)


class TestSmartFightPolicy(unittest.TestCase):
    """The policy reads fabricated menu states, so these run anywhere.
    Move ids: 33 Tackle, 39 Tail Whip (status), 145 Bubble."""

    def test_no_menu_info_confirms(self):
        pol = SmartFightPolicy()
        self.assertEqual(pol.choose(battle_state([33, 39, 145],
                                                 [35, 30, 30]), 0), "a")

    def test_steers_to_last_damaging_move(self):
        pol = SmartFightPolicy()
        s = battle_state([33, 39, 145], [35, 30, 30], top_y=12, item=1)
        self.assertEqual(pol.choose(s, 0), "down")   # probe toward Bubble
        s = battle_state([33, 39, 145], [35, 30, 30], top_y=12, item=2)
        self.assertEqual(pol.choose(s, 1), "down")   # live: keep steering
        s = battle_state([33, 39, 145], [35, 30, 30], top_y=12, item=3)
        self.assertEqual(pol.choose(s, 2), "a")      # on Bubble: confirm

    def test_dry_damaging_move_falls_back_to_earlier_damaging(self):
        # The forest stall: Bubble at 0 PP must NOT leave Tail Whip as
        # the target - Tackle (index 0) is, and the cursor starts on it.
        pol = SmartFightPolicy()
        s = battle_state([33, 39, 145], [10, 30, 0], top_y=12, item=1)
        self.assertEqual(pol.choose(s, 0), "a")

    def test_only_status_moves_left_burns_smallest_pp(self):
        # Nothing damaging has PP: burn the smallest-PP move so the
        # engine eventually offers Struggle.
        pol = SmartFightPolicy()
        s = battle_state([39, 45], [5, 30], top_y=12, item=2)
        self.assertEqual(pol.choose(s, 0), "up")     # toward index 0

    def test_no_pp_anywhere_confirms(self):
        pol = SmartFightPolicy()
        s = battle_state([33, 145], [0, 0], top_y=12, item=1)
        self.assertEqual(pol.choose(s, 0), "a")

    def test_main_menu_confirms_fight(self):
        pol = SmartFightPolicy()
        s = battle_state([33, 145], [35, 30], top_y=14, item=0)
        self.assertEqual(pol.choose(s, 0), "a")
        s = battle_state([33, 145], [35, 30], top_y=14, item=2)
        self.assertEqual(pol.choose(s, 1), "up")

    def test_stale_menu_echo_gets_one_probe_only(self):
        # Menu bytes frozen mid-animation: one probe, then back to A
        # (the live stall this design replaced pressed on forever).
        pol = SmartFightPolicy()
        s = battle_state([33, 145], [35, 30], top_y=12, item=1)
        self.assertEqual(pol.choose(s, 0), "down")   # the probe
        self.assertEqual(pol.choose(s, 1), "a")      # judged stale
        self.assertEqual(pol.choose(s, 2), "a")


if __name__ == "__main__":
    unittest.main()
