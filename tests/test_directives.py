"""Tests for the live redirect channel (directives.py), the runner's
directive-stop contract, and the viewer's /directive door.

No ROM: goal validation and one full goal execution run against the
SimEmulator, the same seam the executor tests use. The real-ROM path
is covered by the bounded smoke run, not here."""

import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import directives as dv
import play_full_game as fg
from agent.emu import SimEmulator
from agent.executor import Executor
from agent.memory import EpisodicMemory
from agent.world import default_world
from web_viewer import make_server


class GoalJsonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = default_world()

    def test_by_name(self):
        g = dv.goal_from_json({"target_map_name": "ViridianCity",
                               "reason": "detour"}, self.w)
        self.assertEqual(g.target_map, self.w.map_id("ViridianCity"))
        self.assertEqual(g.kind, "goto")
        self.assertEqual(g.reason, "detour")
        self.assertIsNone(g.target_xy)

    def test_by_id_with_xy(self):
        g = dv.goal_from_json(
            {"target_map": self.w.map_id("PalletTown"),
             "target_xy": [5, 6], "optional": True}, self.w)
        self.assertEqual(g.target_xy, (5, 6))
        self.assertTrue(g.optional)

    def test_unknown_name_rejected(self):
        with self.assertRaises(ValueError):
            dv.goal_from_json({"target_map_name": "NowhereCity"}, self.w)

    def test_unknown_id_rejected(self):
        with self.assertRaises(ValueError):
            dv.goal_from_json({"target_map": 9999}, self.w)

    def test_missing_target_rejected(self):
        with self.assertRaises(ValueError):
            dv.goal_from_json({"reason": "no target"}, self.w)

    def test_non_goto_kind_rejected(self):
        with self.assertRaises(ValueError):
            dv.goal_from_json({"kind": "catch",
                               "target_map_name": "PalletTown"}, self.w)

    def test_not_an_object_rejected(self):
        with self.assertRaises(ValueError):
            dv.goal_from_json("ViridianCity", self.w)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "directives.jsonl"
        self.offset = Path(self.tmp.name) / "directives.offset"

    def test_consumed_exactly_once(self):
        dv.append_directive(self.path, {"action": "pause"})
        src = dv.DirectiveSource(self.path, self.offset)
        items = src.poll()
        self.assertEqual(items, [("directive", {"action": "pause"})])
        self.assertEqual(src.poll(), [])

    def test_resume_does_not_replay(self):
        dv.append_directive(self.path, {"action": "pause"})
        dv.DirectiveSource(self.path, self.offset).poll()
        # a restarted runner (new source, same offset file) sees nothing
        src2 = dv.DirectiveSource(self.path, self.offset)
        self.assertEqual(src2.poll(), [])
        dv.append_directive(self.path, {"action": "resume"})
        self.assertEqual(src2.poll(),
                         [("directive", {"action": "resume"})])

    def test_partial_line_waits_for_newline(self):
        self.path.write_text('{"action": "pa')
        src = dv.DirectiveSource(self.path, self.offset)
        self.assertEqual(src.poll(), [])
        with self.path.open("a") as fh:
            fh.write('use"}\n')
        self.assertEqual(src.poll(),
                         [("directive", {"action": "pause"})])

    def test_malformed_line_consumed_and_reported(self):
        self.path.write_text("not json\n{\"action\": \"stop\"}\n")
        src = dv.DirectiveSource(self.path, self.offset)
        items = src.poll()
        self.assertEqual(items[0][0], "malformed")
        self.assertEqual(items[1], ("directive", {"action": "stop"}))
        self.assertEqual(src.poll(), [])


class FakeHost:
    """Channel host over a SimEmulator standing in Pallet Town."""

    def __init__(self, world):
        self.world = world
        self.memory = EpisodicMemory()
        self.emu = SimEmulator(world, world.map_id("PalletTown"), (5, 6))
        self.notes = []
        self.executed = []

    def note(self, msg):
        self.notes.append(msg)

    def read_state(self):
        return self.emu.read_state()

    def execute_goal(self, goal):
        self.executed.append(goal)
        return True, "fake executor: complete"


class ChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = default_world()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "directives.jsonl"
        self.host = FakeHost(self.w)
        self.channel = dv.DirectiveChannel(
            self.path, Path(self.tmp.name) / "off", host=self.host,
            audit_path=Path(self.tmp.name) / "audit.jsonl",
            poll_interval=0.02)

    def notes_text(self):
        return "\n".join(self.host.notes)

    def test_no_directives_is_ok_immediately(self):
        self.assertEqual(self.channel.service(), "ok")

    def test_valid_goal_executes(self):
        dv.append_directive(self.path,
                            {"goal": {"target_map_name": "ViridianCity"}})
        self.assertEqual(self.channel.service(), "ok")
        self.assertEqual(len(self.host.executed), 1)
        self.assertEqual(self.host.executed[0].target_map,
                         self.w.map_id("ViridianCity"))
        self.assertIn("ACCEPTED", self.notes_text())
        self.assertIn("OUTCOME", self.notes_text())

    def test_unreachable_goal_rejected_with_reason(self):
        dv.append_directive(self.path,
                            {"goal": {"target_map_name": "CinnabarIsland"}})
        self.assertEqual(self.channel.service(), "ok")
        self.assertEqual(self.host.executed, [])
        self.assertIn("REJECTED", self.notes_text())
        self.assertIn("surf", self.notes_text())

    def test_unparseable_goal_rejected(self):
        dv.append_directive(self.path,
                            {"goal": {"target_map_name": "NowhereCity"}})
        self.assertEqual(self.channel.service(), "ok")
        self.assertEqual(self.host.executed, [])
        self.assertIn("REJECTED", self.notes_text())

    def test_malformed_line_logged_run_undisturbed(self):
        self.path.write_text("{oops\n")
        self.assertEqual(self.channel.service(), "ok")
        self.assertIn("MALFORMED", self.notes_text())
        self.assertEqual(self.host.executed, [])

    def test_unknown_action_rejected(self):
        dv.append_directive(self.path, {"action": "explode"})
        self.assertEqual(self.channel.service(), "ok")
        self.assertIn("REJECTED", self.notes_text())

    def test_stop_directive(self):
        dv.append_directive(self.path, {"action": "stop"})
        self.assertEqual(self.channel.service(), "stop")

    def test_audit_trail_written(self):
        dv.append_directive(self.path, {"action": "pause"})
        dv.append_directive(self.path, {"action": "resume"})
        self.assertEqual(self.channel.service(), "ok")
        records = [json.loads(l) for l in
                   (Path(self.tmp.name) / "audit.jsonl")
                   .read_text().splitlines()]
        self.assertEqual([r["result"] for r in records],
                         ["action:pause", "action:resume"])
        self.assertTrue(all(r["source"] == "external" for r in records))

    def test_audit_records_are_not_reconsumed(self):
        # Regression (smoke): audit writes must never land in the
        # channel file - the channel would eat its own audit trail.
        dv.append_directive(self.path, {"action": "pause"})
        dv.append_directive(self.path, {"action": "resume"})
        self.assertEqual(self.channel.service(), "ok")
        self.assertEqual(self.channel.service(), "ok")
        self.assertNotIn("MALFORMED", self.notes_text())

    def test_audit_path_must_differ_from_channel_file(self):
        with self.assertRaises(ValueError):
            dv.DirectiveChannel(self.path, None, host=self.host,
                                audit_path=self.path)

    def test_pause_blocks_goal_queues_then_resume_runs_it(self):
        dv.append_directive(self.path, {"action": "pause"})
        dv.append_directive(self.path,
                            {"goal": {"target_map_name": "ViridianCity"}})
        result = []
        t = threading.Thread(
            target=lambda: result.append(self.channel.service()))
        t.start()
        # give the channel time to consume pause + queue the goal
        deadline = time.time() + 5
        while "queued" not in self.notes_text() and time.time() < deadline:
            time.sleep(0.02)
        self.assertIn("queued", self.notes_text())
        self.assertEqual(self.host.executed, [])
        self.assertTrue(t.is_alive())  # still holding the pause
        dv.append_directive(self.path, {"action": "resume"})
        t.join(timeout=5)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, ["ok"])
        self.assertEqual(len(self.host.executed), 1)

    def test_pause_then_stop_skips_queued_goal(self):
        dv.append_directive(self.path, {"action": "pause"})
        dv.append_directive(self.path,
                            {"goal": {"target_map_name": "ViridianCity"}})
        result = []
        t = threading.Thread(
            target=lambda: result.append(self.channel.service()))
        t.start()
        deadline = time.time() + 5
        while "queued" not in self.notes_text() and time.time() < deadline:
            time.sleep(0.02)
        dv.append_directive(self.path, {"action": "stop"})
        t.join(timeout=5)
        self.assertEqual(result, ["stop"])
        self.assertEqual(self.host.executed, [])

    def test_executor_exception_is_an_outcome_not_a_crash(self):
        def boom(goal):
            raise RuntimeError("kaput")

        self.host.execute_goal = boom
        dv.append_directive(self.path,
                            {"goal": {"target_map_name": "ViridianCity"}})
        self.assertEqual(self.channel.service(), "ok")
        self.assertIn("OUTCOME", self.notes_text())
        self.assertIn("failed", self.notes_text())
        self.assertIn("kaput", self.notes_text())

    def test_consumed_once_across_channel_instances(self):
        dv.append_directive(self.path,
                            {"goal": {"target_map_name": "ViridianCity"}})
        self.assertEqual(self.channel.service(), "ok")
        host2 = FakeHost(self.w)
        ch2 = dv.DirectiveChannel(
            self.path, Path(self.tmp.name) / "off", host=host2,
            poll_interval=0.02)
        self.assertEqual(ch2.service(), "ok")
        self.assertEqual(host2.executed, [])


class SimExecutionTests(unittest.TestCase):
    """The full path on the sim seam: parse -> validate -> the
    executor's own loop walks the goal (Pallet -> Viridian)."""

    def test_external_goal_walks_to_viridian(self):
        w = default_world()
        emu = SimEmulator(w, w.map_id("PalletTown"), (5, 6))
        memory = EpisodicMemory()
        goal = dv.goal_from_json({"target_map_name": "ViridianCity"}, w)
        ex = Executor(emu, w, fg._OneShotPlanner(goal), memory)
        result = ex.run(max_goals=1)
        self.assertEqual(result.goals_done, 1, result.log)
        self.assertEqual(emu.read_state().map_id,
                         w.map_id("ViridianCity"))


class RunnerDirectiveTests(unittest.TestCase):
    """The runner's side of the contract, with a duck-typed ctx."""

    class Ctx:
        def __init__(self, service_result="ok", raise_stop=False):
            self.notes = []
            self.saved = []
            self._service_result = service_result
            self._raise_stop = raise_stop

        def note(self, msg):
            self.notes.append(msg)

        def state(self):
            return SimpleNamespace(badges=set(), money=0)

        def save_state(self, path):
            Path(path).write_bytes(b"s")
            self.saved.append(str(path))

        def load_state(self, path):
            pass

        def reboot(self):
            pass

        def service_directives(self):
            if self._raise_stop:
                return "stop"
            return self._service_result

    def _runner(self, ctx, phases):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return fg.Runner(ctx, phases, tmp.name), Path(tmp.name)

    def test_stop_at_boundary_halts_cleanly(self):
        ctx = self.Ctx(service_result="stop")
        phases = [fg.Phase("a", "a", lambda c: True, lambda c: (True, ""))]
        r, out = self._runner(ctx, phases)
        self.assertFalse(r.run())
        self.assertTrue(r.stopped_by_directive)
        self.assertTrue((out / "stopped.state").exists())

    def test_directive_stop_mid_phase_halts_cleanly(self):
        def boom(ctx):
            raise fg.DirectiveStop("stop directive")

        ctx = self.Ctx()
        phases = [fg.Phase("a", "a", boom, lambda c: (True, "")),
                  fg.Phase("b", "b", lambda c: True, lambda c: (True, ""))]
        r, out = self._runner(ctx, phases)
        self.assertFalse(r.run())
        self.assertTrue(r.stopped_by_directive)
        self.assertTrue((out / "stopped.state").exists())
        self.assertIn("HALTED by directive stop", "\n".join(ctx.notes))


class CtxHostProtocolTests(unittest.TestCase):
    """Regression (smoke): the channel's host protocol and Ctx's
    method names drifted apart once - FakeHost defined execute_goal
    while Ctx grew execute_external_goal, and only the real emulator
    noticed. Pin the contract on the real class."""

    def test_ctx_satisfies_the_channel_host_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = fg.Ctx("unused.gbc", tmp)
            for attr in ("world", "memory", "note", "read_state",
                         "execute_goal", "service_directives"):
                self.assertTrue(hasattr(ctx, attr), attr)
            self.assertTrue(callable(ctx.execute_goal))
            self.assertTrue(callable(ctx.read_state))


class WebDoorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "directives.jsonl"
        ctx = SimpleNamespace(recent=[], emu=None, world=None)
        self.provider = fg.WebProvider(ctx)
        self.provider.directive_path = self.path

    def test_provider_directive_appends(self):
        self.assertEqual(
            self.provider.directive({"action": "pause"}), {"ok": True})
        lines = self.path.read_text().splitlines()
        self.assertEqual(json.loads(lines[0]), {"action": "pause"})

    def test_provider_directive_validates_shape(self):
        self.assertFalse(self.provider.directive("pause")["ok"])
        self.assertFalse(self.provider.directive({})["ok"])
        self.assertFalse(Path(self.path).exists())

    def test_http_endpoint_appends_to_same_file(self):
        server = make_server("127.0.0.1", 0, self.provider)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port,
                                              timeout=5)
            conn.request("POST", "/directive",
                         body=json.dumps(
                             {"goal": {"target_map_name": "PalletTown"}}),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            self.assertTrue(json.loads(resp.read())["ok"])
            conn.request("POST", "/directive", body="{not json",
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            self.assertFalse(json.loads(resp.read())["ok"])
            conn.close()
        finally:
            server.shutdown()
        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]),
                         {"goal": {"target_map_name": "PalletTown"}})

    def test_page_has_redirect_form(self):
        from web_viewer import PAGE
        self.assertIn("/directive", PAGE)
        self.assertIn("btnGoal", PAGE)


if __name__ == "__main__":
    unittest.main()
