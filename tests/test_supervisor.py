import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.emu import SimEmulator
from agent.executor import Executor
from agent.goals import Goal, Verdict, validate_goal
from agent.memory import EpisodicMemory
from agent.planner import LLMPlanner
from agent.state import (ADDR_PARTY_COUNT, ADDR_PARTY_MONS, FakeMemory,
                         read_state)
from agent.supervisor_log import SupervisorLog
from agent.world import default_world

_spec = importlib.util.spec_from_file_location(
    "build_supervisor_dataset",
    Path(__file__).resolve().parent.parent / "scripts"
    / "build_supervisor_dataset.py")
builder = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = builder  # dataclasses resolve via sys.modules
_spec.loader.exec_module(builder)


class TestSpeciesRead(unittest.TestCase):
    def test_species_id_comes_from_party_struct_byte_0(self):
        mem = FakeMemory()
        mem.write_u8(ADDR_PARTY_COUNT, 1)
        mem.write_u8(ADDR_PARTY_MONS, 25)
        state = read_state(mem.read)
        self.assertEqual(len(state.party), 1)
        self.assertEqual(state.party[0].species_id, 25)


class TestSupervisorLog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = default_world()

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub" / "log.jsonl"
            log = SupervisorLog(path)
            emu = SimEmulator(self.w, self.w.map_id("PalletTown"), (5, 6))
            state = emu.read_state()
            goal = Goal("goto", self.w.map_id("OaksLab"), reason="test")
            record_id = log.record_proposal(
                model="stub", context={"state": state.digest()},
                state=state, world=self.w, step=None, proposal=goal,
                proposal_error=None, verdict=Verdict(True),
                decision="llm", returned_goal=goal)
            self.assertIsNotNone(record_id)
            log.record_outcome(record_id, goal_key=goal.key(),
                               executed=True, success=True)
            records = SupervisorLog.read_records(path)
            self.assertEqual(len(records), 2)
            prop, outcome = records
            self.assertEqual(prop["kind"], "supervisor_proposal")
            self.assertEqual(prop["record_id"], record_id)
            self.assertEqual(prop["decision"], "llm")
            self.assertTrue(prop["verdict"]["ok"])
            self.assertEqual(prop["proposal"]["target_map_name"], "OaksLab")
            summary = prop["state"]
            self.assertEqual(summary["map_name"], "PalletTown")
            self.assertEqual(summary["badges"], [])
            self.assertIn("party", summary)
            self.assertEqual(outcome["kind"], "supervisor_outcome")
            self.assertEqual(outcome["record_id"], record_id)
            self.assertTrue(outcome["executed"])
            self.assertTrue(outcome["success"])

    def test_disabled_log_is_noop(self):
        log = SupervisorLog(None)
        self.assertFalse(log.enabled)
        emu = SimEmulator(self.w, self.w.map_id("PalletTown"), (5, 6))
        state = emu.read_state()
        record_id = log.record_proposal(
            model="stub", context={}, state=state, world=self.w,
            step=None, proposal=None, proposal_error="boom",
            verdict=None, decision="fallback", returned_goal=None)
        self.assertIsNone(record_id)
        log.record_outcome(record_id, goal_key="x", executed=False,
                           success=False)  # must not raise
        self.assertEqual(
            SupervisorLog.read_records("/nonexistent/log.jsonl"), [])


def _planner(tmpdir, goal):
    planner = LLMPlanner("http://127.0.0.1:9", "stub",
                         log_path=Path(tmpdir) / "log.jsonl")
    patcher = mock.patch.object(
        LLMPlanner, "_ask",
        lambda self, state, world, memory, context=None: goal)
    patcher.start()
    return planner, patcher


class TestLLMPlannerLogging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = default_world()

    def _state(self):
        emu = SimEmulator(self.w, self.w.map_id("PalletTown"), (5, 6))
        return emu.read_state()

    def test_accepted_proposal_logged(self):
        with tempfile.TemporaryDirectory() as td:
            goal = Goal("goto", self.w.map_id("OaksLab"), reason="stub")
            planner, patcher = _planner(td, goal)
            try:
                returned = planner.propose(self._state(), self.w,
                                           EpisodicMemory())
            finally:
                patcher.stop()
            self.assertEqual(returned.target_map, goal.target_map)
            records = SupervisorLog.read_records(Path(td) / "log.jsonl")
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["decision"], "llm")
            self.assertTrue(rec["verdict"]["ok"])
            self.assertEqual(rec["state"]["checklist_step"]["id"],
                             "starter")
            self.assertEqual(rec["context"]["current_map_name"],
                             "PalletTown")

    def test_rejected_proposal_logged_with_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            # No Surf yet: Cinnabar is impossible, the validator must
            # say so and the checklist goal (OaksLab) is returned.
            goal = Goal("goto", self.w.map_id("CinnabarIsland"),
                        reason="stub")
            planner, patcher = _planner(td, goal)
            try:
                returned = planner.propose(self._state(), self.w,
                                           EpisodicMemory())
            finally:
                patcher.stop()
            self.assertEqual(returned.target_map,
                             self.w.map_id("OaksLab"))
            (rec,) = SupervisorLog.read_records(Path(td) / "log.jsonl")
            self.assertEqual(rec["decision"], "fallback")
            self.assertFalse(rec["verdict"]["ok"])
            self.assertIn("surf", rec["verdict"]["missing_capabilities"])
            self.assertEqual(rec["returned_goal"]["target_map_name"],
                             "OaksLab")

    def test_executor_outcome_is_linked(self):
        with tempfile.TemporaryDirectory() as td:
            goal = Goal("goto", self.w.map_id("OaksLab"), reason="stub")
            planner, patcher = _planner(td, goal)
            try:
                emu = SimEmulator(self.w, self.w.map_id("PalletTown"),
                                  (5, 6))
                executor = Executor(emu, self.w, planner, EpisodicMemory())
                result = executor.run(max_goals=1)
            finally:
                patcher.stop()
            self.assertEqual(result.goals_done, 1)
            records = SupervisorLog.read_records(Path(td) / "log.jsonl")
            kinds = [r["kind"] for r in records]
            self.assertEqual(kinds, ["supervisor_proposal",
                                     "supervisor_outcome"])
            self.assertEqual(records[1]["record_id"],
                             records[0]["record_id"])
            self.assertTrue(records[1]["executed"])
            self.assertTrue(records[1]["success"])


class TestDatasetBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = default_world()

    def test_scenarios_carry_validated_oracle_goals(self):
        scenarios = builder.generate_scenarios(self.w, seed=7, variants=2)
        self.assertGreater(len(scenarios), 40)
        targets = set()
        for s in scenarios:
            verdict = validate_goal(s.goal, s.state, self.w, s.memory)
            self.assertTrue(verdict.ok, verdict.problems)
            targets.add(self.w.map_name(s.goal.target_map))
        # The pool spans most of the spine, not just the opening.
        self.assertGreaterEqual(len(targets), 10)

    def test_examples_are_wellformed_and_validate(self):
        examples, stats = builder.build_examples(
            self.w, synthetic_count=30, seed=7, variants=2, log_path=None)
        self.assertEqual(len(examples), 30)
        self.assertEqual(stats["synthetic"], 30)
        for example, _source in examples:
            messages = example["messages"]
            self.assertEqual([m["role"] for m in messages],
                             ["system", "user", "assistant"])
            user = json.loads(messages[1]["content"])
            self.assertIn("state", user)
            self.assertIn("next_story_step", user)
            assistant = json.loads(messages[2]["content"])
            target_id = self.w.map_id(assistant["target_map_name"])
            goal = Goal("goto", target_id, reason=assistant["reason"])
            self.assertEqual(goal.kind, "goto")

    def test_log_source_keeps_only_verified_successes(self):
        def prop(rid, decision, ok):
            return {"kind": "supervisor_proposal", "record_id": rid,
                    "decision": decision,
                    "verdict": {"ok": ok, "problems": []},
                    "context": {"state": {}, "next_story_step": "x"},
                    "proposal": {"target_map_name": "OaksLab",
                                 "reason": "r"}}

        def outcome(rid, executed, success):
            return {"kind": "supervisor_outcome", "record_id": rid,
                    "executed": executed, "success": success}
        records = [
            prop("a", "llm", True), outcome("a", True, True),
            prop("b", "llm", False), outcome("b", False, False),
            prop("c", "llm", True), outcome("c", True, False),
            prop("d", "fallback", True), outcome("d", True, True),
        ]
        examples = builder.examples_from_log(records, self.w)
        self.assertEqual(len(examples), 1)
        example, source = examples[0]
        self.assertEqual(source, "log")
        self.assertEqual(
            json.loads(example["messages"][2]["content"]),
            {"target_map_name": "OaksLab", "reason": "r"})


if __name__ == "__main__":
    unittest.main()
