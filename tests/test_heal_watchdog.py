"""Heal-goal safety net: the stall watchdog and the adjacent-door step.

Regression cover for a live wedge: the agent cycled heal_center for 130+ of 272
turns in Viridian City (45-72 turns on one tile at (23,12)) because (a) at
dist<=1 it pressed A out in the street instead of stepping onto the door, and
(b) nothing bounded the attempt, so it never stood down.

Covers:
  1. _heal_options stall watchdog + cooldown (pure logic, importable).
  2. heal_center dispatch: adjacent door is STEPPED onto; A only for
     src in (nurse, inside) - static assertions on the branch.
  3. collect_pokemon_status.sh analysis block, extracted from its heredoc and
     run against synthetic traces (wedged vs healthy).

The collector lives outside this repo; its checks are skipped if it is absent.
Run:  python3 tests/test_heal_watchdog.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTOPILOT = f"{REPO}/pokemon_agent/autopilot.py"
COLLECTOR = os.environ.get(
    "POKEMON_COLLECTOR",
    os.path.expanduser("~/.hermes/scripts/collect_pokemon_status.sh"))
sys.path.insert(0, REPO)

from pokemon_agent.autopilot import HermesDriver  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def state(dist, src="door", direction=(-1, 1), tackle_pp=5, map_name="Viridian City"):
    """A state where _needs_heal() is warranted (Tackle below its 25% reserve)."""
    return {
        "party": [{"hp": 70, "max_hp": 72}],
        "active_mon": {
            "species": "Ivysaur", "hp": 70, "max_hp": 72,
            "types": ["Grass", "Poison"],
            "moves": [{"name": "Tackle", "pp": tackle_pp},
                      {"name": "Growl", "pp": 40},
                      {"name": "Leech Seed", "pp": 9},
                      {"name": "Vine Whip", "pp": 0}],
        },
        "nearest_pokecenter_src": src,
        "nearest_pokecenter_dist": dist,
        "nearest_pokecenter_dir": list(direction),
        "battle": {"in_battle": False, "type": "none", "enemy": {}},
    }


def fresh(turn=100):
    d = HermesDriver.__new__(HermesDriver)
    d.turn = turn
    return d


# --------------------------------------------------------------------------
print("\n[1] _heal_options: offers while making progress")
d = fresh()
seen = []
for i, dist in enumerate([20, 18, 15, 11, 7, 3, 1]):
    d.turn = 100 + i
    seen.append(bool(d._heal_options(state(dist))))
check("heal offered on every improving turn", all(seen), f"offered={seen}")
check("no stall recorded while improving", getattr(d, "_heal_stall", 0) == 0)

print("\n[2] _heal_options: stays offered up to 9 non-improving turns")
d = fresh()
offered = []
for i in range(9):
    d.turn = 200 + i
    offered.append(bool(d._heal_options(state(15))))
check("still offered at turn 9 of no progress", all(offered), f"offered={offered}")
# Call 1 sets the baseline distance (best is None -> stall 0); calls 2..9 are
# the 8 non-improving ones.
check("baseline on call 1, then 8 non-progress counted",
      getattr(d, "_heal_stall", 0) == 8, f"stall={getattr(d, '_heal_stall', 0)}")

print("\n[3] _heal_options: stands down after 10 non-improving turns")
d = fresh()
results = []
for i in range(12):
    d.turn = 300 + i
    results.append(bool(d._heal_options(state(15))))
check("offered for first 10, then withheld", results[:10] == [True] * 10
      and results[10:] == [False, False], f"results={results}")
check("cooldown armed ~40 turns ahead",
      getattr(d, "_heal_cooldown_until", 0) >= 300 + 10 + 30,
      f"until={getattr(d, '_heal_cooldown_until', None)}")

print("\n[4] _heal_options: cooldown expires and healing resumes")
d = fresh()
for i in range(11):                      # trip the watchdog
    d.turn = 400 + i
    d._heal_options(state(15))
while_off = bool(d._heal_options(state(15)))
d.turn = d._heal_cooldown_until + 1      # past the cooldown
after = bool(d._heal_options(state(15)))
check("withheld during cooldown", while_off is False)
check("offered again after cooldown", after is True)

print("\n[5] _heal_options: no reason -> {} and watchdog reset")
d = fresh()
d._heal_stall = 7
d._heal_best_dist = 9
no_need = d._heal_options(state(15, tackle_pp=34))
check("returns {} when the party is healthy", no_need == {}, f"got={no_need}")
check("watchdog reset when goal clears",
      d._heal_stall == 0 and d._heal_best_dist is None)

# --------------------------------------------------------------------------
print("\n[6] heal_center dispatch: adjacent door is stepped onto, not A-pressed")
src = open(AUTOPILOT).read()
check("A only for nurse/inside",
      'elif dist <= 1 and src in ("nurse", "inside"):' in src)
check("adjacent-door branch steps in",
      re.search(r'elif dist <= 1:\s*\n(?:\s*#.*\n)*\s*actions = \[\(walk or "walk_down"\)',
                src) is not None)

# --------------------------------------------------------------------------
print("\n[7] collector analysis: extracted and run on synthetic traces")


def extract_analysis():
    lines = open(COLLECTOR).read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().endswith("<< 'PYEOF'"))
    end = next(i for i, l in enumerate(lines) if i > start and l.strip() == "PYEOF")
    return "\n".join(lines[start + 1:end])


TRACE_TESTS = os.path.exists(COLLECTOR)
if TRACE_TESTS:
    code = extract_analysis()
    check("analysis heredoc extracted", "ANALYSIS (last hour)" in code)
else:
    print(f"  SKIP  collector not present at {COLLECTOR}")


def row(**kw):
    base = {"ts": datetime.now(timezone.utc).isoformat(), "turn": 0,
            "choice": "walk_down", "n_options": 5, "options": [],
            "map": "Route 2", "pos": [8, 44], "door": False, "here": 1, "stuck": 0,
            "battle": {"in_battle": False, "type": "none", "enemy": [None] * 4,
                       "active_moves": [], "party_moves": []}}
    base.update(kw)
    return json.dumps(base)


def run_on(rows):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     prefix="hermes-verify-") as fh:
        fh.write("\n".join(rows) + "\n")
        path = fh.name
    try:
        env = dict(os.environ, POKEMON_TRACE=path)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, timeout=60)
        return out.stdout + out.stderr
    finally:
        os.unlink(path)


if TRACE_TESTS:
    wedged = [row(turn=i, choice="heal_center", map="Viridian City", pos=[23, 12],
                  here=i, stuck=min(i, 4)) for i in range(1, 26)]
    out = run_on(wedged)
    check("wedge flagged NEEDS ATTENTION", "NEEDS ATTENTION" in out,
          out.splitlines()[0][:90])
    check("wedge reports POSITION_FROZEN", "POSITION_FROZEN" in out)
    check("wedge reports the frozen cell", "[23, 12]" in out)

    healthy = [row(turn=i, choice=("move_0" if i % 5 == 0 else "walk_down"),
                   map=("Viridian City" if i % 3 == 0 else "Route 2"),
                   pos=[8 + (i % 7), 44 + (i % 9)], here=1, stuck=0,
                   battle={"in_battle": i % 5 == 0, "type": "wild",
                           "enemy": ["Pidgey", 4, 7, 18],
                           "active_moves": [["Tackle", 30]],
                           "party_moves": [["Tackle", 30]]})
               for i in range(40)]
    out = run_on(healthy)
    check("healthy hour reads APPROPRIATE", "APPROPRIATE" in out,
          out.splitlines()[0][:90])
    check("healthy hour draws no stuck flag", "TILE_STUCK" not in out)
    check("healthy hour starts a battle count",
          re.search(r"battle_turns=[1-9]", out) is not None)
else:
    print("  SKIP  synthetic trace checks (collector absent)")

# --------------------------------------------------------------------------
print("\n[8] both files still parse")
targets = [(AUTOPILOT, [sys.executable, "-m", "py_compile", AUTOPILOT])]
if os.path.exists(COLLECTOR):
    targets.append((COLLECTOR, ["bash", "-n", COLLECTOR]))
for path, cmd in targets:
    r = subprocess.run(cmd, capture_output=True, text=True)
    check(f"{os.path.basename(path)} parses", r.returncode == 0, r.stderr[:120])

print("\n" + ("=" * 56))
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILED -> {FAILS}")
    sys.exit(1)
print("RESULT: all checks passed")
