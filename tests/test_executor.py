import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.emu import SimEmulator
from agent.executor import Executor, SmartFightPolicy, StuckError
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


class TestLedgeHop(unittest.TestCase):
    def test_sim_step_hops_a_ledge(self):
        # Find a south-facing Route 4 ledge whose standing and landing
        # tiles are both standable, stand above it, and step down:
        # the sim must land two tiles on, on the far side of the face.
        w = default_world()
        r4 = w.map_id("Route4")
        gm = w.maps[r4]
        face = delta = None
        for t, d in sorted(gm.ledges.items()):
            if d != (0, 1):
                continue
            stand = (t[0], t[1] - 1)
            land = (t[0], t[1] + 1)
            if gm.walkable(*stand) and gm.walkable(*land):
                face, delta = t, d
                break
        self.assertIsNotNone(face, "no usable Route 4 ledge found")
        emu = SimEmulator(w, r4, (face[0], face[1] - 1))
        emu.press("down")
        self.assertEqual((emu.x, emu.y), (face[0], face[1] + 1))


class TestLearnedWalls(unittest.TestCase):
    def test_refusal_on_cavern_grid_is_learned_and_routed_around(self):
        # A refusal on a CAVERN map (a mixed edge cell the grid got
        # wrong, most likely) is learned permanently: wall off
        # (20, 10) in the sim - the straight line south through a
        # Mt. Moon B1F room crosses it - and the walk must bump,
        # learn, and detour.
        w = default_world()
        b1 = w.map_id("MtMoonB1F")
        emu = SimEmulator(w, b1, (20, 8))
        emu.blocked.add((b1, 20, 10))
        ex = Executor(emu, w, ChecklistPlanner(), EpisodicMemory(),
                      stuck_threshold=3)
        ex._walk_to(b1, (20, 11))
        self.assertEqual((emu.x, emu.y), (20, 11))
        self.assertIn((20, 10), ex._learned_walls.get(b1, set()))

    def test_learned_wall_seal_clears_and_retries(self):
        # A false learned wall sealing the start must not brick the
        # walk: on 'no tile path' the executor forgets the map's
        # learned walls once and replans (Mt. Moon B1F, live).
        w = default_world()
        b1 = w.map_id("MtMoonB1F")
        emu = SimEmulator(w, b1, (20, 9))
        ex = Executor(emu, w, ChecklistPlanner(), EpisodicMemory(),
                      stuck_threshold=3)
        ex._learned_walls[b1] = {(19, 9), (21, 9), (20, 8), (20, 10)}
        ex._walk_to(b1, (20, 11))
        self.assertEqual((emu.x, emu.y), (20, 11))
        self.assertNotIn(b1, ex._learned_walls)

    def test_real_seal_still_raises(self):
        # With every neighbour genuinely blocked, clearing cannot
        # invent a path: the walk still reports stuck (and does not
        # loop forever).
        w = default_world()
        b1 = w.map_id("MtMoonB1F")
        emu = SimEmulator(w, b1, (20, 9))
        ring = {(19, 9), (21, 9), (20, 8), (20, 10)}
        for t in ring:
            emu.blocked.add((b1, t[0], t[1]))
        ex = Executor(emu, w, ChecklistPlanner(), EpisodicMemory(),
                      stuck_threshold=3)
        ex._learned_walls[b1] = set(ring)
        with self.assertRaises(StuckError):
            ex._walk_to(b1, (20, 11))

    def test_refusal_on_a_real_grid_stays_transient(self):
        # Outdoors a refusal is usually an NPC: nothing is learned.
        w = default_world()
        emu = SimEmulator(w, 0, (5, 6))
        emu.blocked.add((0, 12, 11))  # the lab door (OVERWORLD map)
        ex = Executor(emu, w, ChecklistPlanner(), EpisodicMemory(),
                      stuck_threshold=3)
        with self.assertRaises(StuckError):
            ex._walk_to(0, (12, 11))
        self.assertEqual(ex._learned_walls, {})


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
