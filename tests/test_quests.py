"""Tests for the side-quest library (quests.py) and the
directives channel's quest form.

No ROM: registry/validation/parser tests are pure; the data
tests parse the pokered checkout (skipped when absent, the
test_dex_audit convention); the channel tests run against a
fake host, mirroring test_directives' FakeHost. Live quest
execution is covered by the probe179 smokes (docs S40), not
here."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import directives as dv
import quests

POKERED = Path.home() / "workspace" / "projects" / "pokered"

EXPECTED_QUESTS = {
    "dex_audit", "heal", "travel", "fund_e4", "grind",
    "catch_wild", "catch_static", "safari_hunt", "game_corner",
    "trade", "evolve_stone", "revive_fossil", "buy", "sell",
    "box_switch",
}


class RegistryTests(unittest.TestCase):
    def test_catalogue_complete(self):
        self.assertEqual(set(quests.QUESTS), EXPECTED_QUESTS)

    def test_every_quest_well_formed(self):
        for name, q in quests.QUESTS.items():
            self.assertTrue(q.summary, name)
            self.assertTrue(callable(q.run), name)
            seen = set()
            for a in q.args:
                self.assertNotIn(a.name, seen, name)
                seen.add(a.name)
                self.assertIn(a.kind,
                              ("str", "int", "bool", "species",
                               "item", "town", "species_list"),
                              (name, a.name))

    def test_catalogue_text_lists_everything(self):
        text = quests._catalogue_text()
        for name in EXPECTED_QUESTS:
            self.assertIn(name, text)


class ArgValidationTests(unittest.TestCase):
    def refused(self, name, raw):
        with self.assertRaises(quests.QuestRefusal):
            quests.validate_args(name, raw)

    def test_unknown_quest(self):
        self.refused("fly_away", {})

    def test_unknown_argument(self):
        self.refused("heal", {"town": "Fuchsia"})

    def test_missing_required(self):
        self.refused("travel", {})
        self.refused("buy", {"item": "ULTRA_BALL"})

    def test_bad_int(self):
        self.refused("fund_e4", {"runs": "lots"})
        self.refused("box_switch", {"box": "six"})

    def test_bad_bool(self):
        self.refused("catch_wild",
                     {"species": "ABRA", "allow_owned": "maybe"})

    def test_args_must_be_object(self):
        self.refused("heal", ["not", "a", "dict"])

    def test_defaults_applied(self):
        clean = quests.validate_args("fund_e4", {})
        self.assertEqual(clean, {"runs": 1})
        clean = quests.validate_args(
            "catch_wild", {"species": "ABRA"})
        self.assertEqual(clean["max_encounters"], 120)
        self.assertFalse(clean["allow_owned"])

    def test_int_coerced_from_string(self):
        clean = quests.validate_args("box_switch",
                                     {"box": "6"})
        self.assertEqual(clean["box"], 6)

    def test_species_list_forms(self):
        clean = quests.validate_args(
            "safari_hunt", {"species": "tauros, scyther"})
        self.assertEqual(clean["species"],
                         ["tauros", "scyther"])
        clean = quests.validate_args(
            "safari_hunt", {"species": ["Tauros"]})
        self.assertEqual(clean["species"], ["Tauros"])

    def test_species_list_required_nonempty(self):
        self.refused("safari_hunt", {"species": []})


class CliArgParseTests(unittest.TestCase):
    def test_positional_fill(self):
        q = quests.QUESTS["buy"]
        raw = quests._parse_cli_args(q, ["ULTRA_BALL", "5"])
        self.assertEqual(raw, {"item": "ULTRA_BALL",
                               "qty": "5"})

    def test_key_value(self):
        q = quests.QUESTS["buy"]
        raw = quests._parse_cli_args(
            q, ["ULTRA_BALL", "5", "mart=FuchsiaMart"])
        self.assertEqual(raw["mart"], "FuchsiaMart")

    def test_species_list_swallows_bare_tokens(self):
        q = quests.QUESTS["safari_hunt"]
        raw = quests._parse_cli_args(
            q, ["tauros", "scyther", "entries=4"])
        self.assertEqual(raw["species"], ["tauros", "scyther"])
        self.assertEqual(raw["entries"], "4")


class RunQuestShellTests(unittest.TestCase):
    """run_quest's refusal shell, exercised without an
    emulator: validation happens before any emu access."""

    def test_unknown_quest_refused(self):
        r = quests.run_quest(None, None, "nope", {},
                             log=lambda m: None)
        self.assertEqual(r.outcome, "refused")
        self.assertEqual(r.refusal, "quest")

    def test_bad_args_refused(self):
        r = quests.run_quest(None, None, "travel", {},
                             log=lambda m: None)
        self.assertEqual(r.outcome, "refused")
        self.assertEqual(r.refusal, "args")


@unittest.skipUnless((POKERED / "constants").is_dir(),
                     "no pokered checkout")
class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = quests.data()

    def test_ceiling(self):
        self.assertEqual(self.d.ledger["ceiling"], 124)

    def test_species_resolution(self):
        d = self.d
        self.assertEqual(d.species_const("Mr. Mime"), "MR_MIME")
        self.assertEqual(d.species_const("lapras"), "LAPRAS")
        self.assertEqual(d.species_const("#124"), "JYNX")
        self.assertEqual(d.species_const("FARFETCH'D"),
                         "FARFETCHD")
        self.assertIsNone(d.species_const("MissingNo."))

    def test_item_resolution(self):
        self.assertEqual(self.d.item_const("Ultra Ball"),
                         "ULTRA_BALL")
        self.assertEqual(self.d.items["ULTRA_BALL"], 0x02)

    def test_learnsets(self):
        # the S39 fact the fund_e4 guard exists for
        self.assertIn((56, "HYPER_BEAM"),
                      self.d.learnsets["SNORLAX"])
        self.assertEqual(
            [lv for lv, _m in self.d.learnsets["LAPRAS"]
             if lv > 46], [])

    def test_marts_and_prices(self):
        d = self.d
        self.assertEqual(d.marts["FuchsiaMart"][0],
                         "ULTRA_BALL")
        self.assertEqual(d.marts["IndigoPlateauLobby"][-2],
                         "REVIVE")
        self.assertEqual(d.prices["ULTRA_BALL"], 1200)

    def test_safari_habitat(self):
        west = self.d.wild["SafariZoneWest"]["grass"]
        self.assertIn("TAUROS", [sp for _l, sp in west])
        center = self.d.wild["SafariZoneCenter"]["grass"]
        self.assertIn("SCYTHER", [sp for _l, sp in center])

    def test_prizes(self):
        self.assertEqual(
            self.d.ledger["prizes"]["PORYGON"]["coins"], 9999)


class FakeQuestHost:
    """Channel host that serves quests (the Ctx protocol)."""

    def __init__(self):
        from agent.memory import EpisodicMemory
        from agent.world import default_world
        self.world = default_world()
        self.memory = EpisodicMemory()
        self.notes = []
        self.quests_run = []

    def note(self, msg):
        self.notes.append(msg)

    def read_state(self):
        raise AssertionError("quest path reads no state")

    def execute_quest(self, name, args):
        self.quests_run.append((name, args))
        if name == "heal":
            return "executed", "party healed"
        if name == "catch_static":
            return "rejected", "MEWTWO is already owned"
        return "failed", "machinery broke"


class BareHost:
    """Channel host without quest support."""

    def __init__(self):
        from agent.memory import EpisodicMemory
        from agent.world import default_world
        self.world = default_world()
        self.memory = EpisodicMemory()
        self.notes = []

    def note(self, msg):
        self.notes.append(msg)

    def read_state(self):
        raise AssertionError("quest path reads no state")


class QuestChannelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "directives.jsonl"
        self.audit = Path(self.tmp.name) / "audit.jsonl"

    def channel(self, host):
        return dv.DirectiveChannel(
            self.path, Path(self.tmp.name) / "off", host=host,
            audit_path=self.audit, poll_interval=0.02)

    def audit_records(self):
        if not self.audit.exists():
            return []
        return [json.loads(l) for l in
                self.audit.read_text().splitlines() if l.strip()]

    def test_quest_executes_and_audits(self):
        host = FakeQuestHost()
        ch = self.channel(host)
        dv.append_directive(self.path,
                            {"quest": "heal", "args": {}})
        self.assertEqual(ch.service(), "ok")
        self.assertEqual(host.quests_run, [("heal", {})])
        self.assertIn("ACCEPTED quest heal",
                      "\n".join(host.notes))
        recs = self.audit_records()
        self.assertEqual(recs[-1]["result"], "executed")

    def test_quest_refusal_audits_as_rejected(self):
        host = FakeQuestHost()
        ch = self.channel(host)
        dv.append_directive(
            self.path, {"quest": "catch_static",
                        "args": {"species": "mewtwo"}})
        self.assertEqual(ch.service(), "ok")
        recs = self.audit_records()
        self.assertEqual(recs[-1]["result"], "rejected")
        self.assertIn("already owned", recs[-1]["detail"])

    def test_host_without_quests_rejects(self):
        host = BareHost()
        ch = self.channel(host)
        dv.append_directive(self.path,
                            {"quest": "heal", "args": {}})
        self.assertEqual(ch.service(), "ok")
        recs = self.audit_records()
        self.assertEqual(recs[-1]["result"], "rejected")
        self.assertIn("does not serve quests",
                      recs[-1]["detail"])

    def test_quest_args_must_be_object(self):
        host = FakeQuestHost()
        ch = self.channel(host)
        dv.append_directive(self.path,
                            {"quest": "heal", "args": [1, 2]})
        self.assertEqual(ch.service(), "ok")
        self.assertEqual(host.quests_run, [])
        self.assertEqual(self.audit_records()[-1]["result"],
                         "rejected")

    def test_quest_name_required(self):
        host = FakeQuestHost()
        ch = self.channel(host)
        dv.append_directive(self.path, {"quest": 42})
        self.assertEqual(ch.service(), "ok")
        self.assertEqual(host.quests_run, [])
        self.assertEqual(self.audit_records()[-1]["result"],
                         "rejected")

    def test_quest_queued_while_paused(self):
        host = FakeQuestHost()
        ch = self.channel(host)
        dv.append_directive(self.path, {"action": "pause"})
        dv.append_directive(self.path,
                            {"quest": "heal", "args": {}})
        dv.append_directive(self.path, {"action": "resume"})
        self.assertEqual(ch.service(), "ok")
        self.assertEqual(host.quests_run, [("heal", {})])
        results = [r["result"] for r in self.audit_records()]
        self.assertIn("queued", results)
        self.assertEqual(results[-1], "executed")


class WatchdogTests(unittest.TestCase):
    """The S40 fund_e4 hang fix: timeboxed primitives and
    the Fly-carrier lookup."""

    def test_timeboxed_returns_value(self):
        self.assertEqual(quests._timeboxed(5, lambda: 42), 42)

    def test_timeboxed_trips(self):
        import time
        with self.assertRaises(quests._Watchdog):
            quests._timeboxed(0.2, time.sleep, 5)

    def test_fly_carrier(self):
        from types import SimpleNamespace
        from agent.state import ADDR_PARTY_MONS
        mem = bytearray(ADDR_PARTY_MONS + 44 * 2 + 12)
        mem[ADDR_PARTY_MONS + 44 + 8] = quests.rr.MOVE_FLY
        party = [SimpleNamespace(hp=10),
                 SimpleNamespace(hp=0)]
        emu = SimpleNamespace(
            pb=SimpleNamespace(memory=mem))
        ctx = SimpleNamespace(
            emu=emu,
            state=lambda: SimpleNamespace(party=party))
        idx, mon = quests._fly_carrier(ctx)
        self.assertEqual(idx, 1)
        self.assertEqual(mon.hp, 0)

    def test_fly_carrier_absent(self):
        from types import SimpleNamespace
        from agent.state import ADDR_PARTY_MONS
        mem = bytearray(ADDR_PARTY_MONS + 44 + 12)
        party = [SimpleNamespace(hp=10)]
        emu = SimpleNamespace(
            pb=SimpleNamespace(memory=mem))
        ctx = SimpleNamespace(
            emu=emu,
            state=lambda: SimpleNamespace(party=party))
        self.assertIsNone(quests._fly_carrier(ctx))


if __name__ == "__main__":
    unittest.main()
