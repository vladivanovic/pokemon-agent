"""Map-graph routing regressions: fake edges, the gate chain, dead-edge pruning.

Covers the Route 2 oscillation root cause. Two independent defects:
  1. `adj[Route 2]` contained a FAKE direct edge to Pewter City (the drawn
     rectangles touch, but in-game you go Route 2 -> Gate Viridian -> Viridian
     Forest -> Gate Pewter -> Pewter City). BFS returned that 2-hop forever.
  2. The real chain was BROKEN: `adj[Route 2 Gate Pewter]` had no edge to
     Pewter City (a 6-row rect gap), so pruning the fake edge made route()
     return None - which dumped the agent onto blind travel_left/travel_right
     headings, the pinball we are fixing.

Run:  python3 tests/test_nav_routing.py
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pokemon_agent.map_graph import MapGraph  # noqa: E402

FAILS = []
FAKE = ("Route 2", "Pewter City")
REAL_CHAIN = ["Route 2", "Route 2 Gate Viridian", "Viridian Forest",
              "Route 2 Gate Pewter", "Pewter City"]


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def graph(dead_path):
    return MapGraph(portals_path=os.path.join(os.path.dirname(dead_path),
                                              "portals.json"))


tmp = tempfile.mkdtemp(prefix="hermes-verify-nav-")
dead_path = os.path.join(tmp, "dead_edges.json")
# Point the graph at our own dead-edge file (env var is read in __init__).
os.environ["POKEMON_DEAD_EDGES"] = dead_path

print("\n[1] the fake direct edge is what BFS prefers while it exists")
g = graph(dead_path)
check("fresh graph has no dead edges", g.dead_edges == set(), str(g.dead_edges))
check("Route 2 -> Pewter City is a direct fake adjacency",
      "Pewter City" in g.adj.get("Route 2", {}))
check("route() returns the fake 2-hop",
      g.route(*FAKE) == ["Route 2", "Pewter City"], str(g.route(*FAKE)))

print("\n[2] the real gate chain is intact (the previously missing edge)")
check("Gate Pewter -> Pewter City exists",
      g.adj.get("Route 2 Gate Pewter", {}).get("Pewter City") == (-1, 0),
      str(g.adj.get("Route 2 Gate Pewter")))
check("Pewter City -> Gate Pewter is the reverse",
      g.adj.get("Pewter City", {}).get("Route 2 Gate Pewter") == (1, 0))
check("neighbors(Gate Pewter) includes Pewter City",
      "Pewter City" in g.neighbors("Route 2 Gate Pewter"))
check("neighbors(Pewter City) includes the gate",
      "Route 2 Gate Pewter" in g.neighbors("Pewter City"))

print("\n[3] marking the fake edge dead re-plans onto the real chain")
changed = g.mark_dead_edge(*FAKE)
check("mark_dead_edge reports a change", changed is True)
check("fake edge dropped from neighbors()",
      "Pewter City" not in g.neighbors("Route 2"), str(g.neighbors("Route 2")))
r = g.route(*FAKE)
check("route() is NOT None (agent not stranded)", r is not None, str(r))
check("route() is the real gate chain", r == REAL_CHAIN, str(r))
check("first_hop_direction still north",
      g.first_hop_direction(*FAKE) == "north")

print("\n[4] dead edges persist across a reload")
g2 = graph(dead_path)
g2.load_dead_edges()
check("reloaded dead edge present", FAKE in g2.dead_edges, str(g2.dead_edges))
check("reloaded graph still routes the real chain",
      g2.route(*FAKE) == REAL_CHAIN, str(g2.route(*FAKE)))

print("\n[5] idempotence and safety rails")
check("re-marking is a no-op", g.mark_dead_edge(*FAKE) is False)
check("self-edge refused", g.mark_dead_edge("Route 2", "Route 2") is False)
check("empty dst refused", g.mark_dead_edge("Route 2", "") is False)
# The guard the autopilot uses before pruning: never remove a load-bearing
# edge. Only meaningful once the FAKE edge is already gone - while it is alive
# it provides a bogus path, so pruning anything still "routes".
other = os.path.join(tmp, "other_dead.json")
os.environ["POKEMON_DEAD_EDGES"] = other
g3 = graph(other)
g3.dead_edges.add(FAKE)                       # fake edge already proven dead
check("with the fake edge dead, route goes via the gates",
      g3.route("Route 2", "Pewter City") == REAL_CHAIN,
      str(g3.route("Route 2", "Pewter City")))
g3.dead_edges.add(("Route 2 Gate Pewter", "Pewter City"))   # load-bearing
check("removing a LOAD-BEARING edge strands the route",
      g3.route("Route 2", "Pewter City") is None,
      str(g3.route("Route 2", "Pewter City")))
g3.dead_edges.discard(("Route 2 Gate Pewter", "Pewter City"))
check("restoring it fixes the route",
      g3.route("Route 2", "Pewter City") == REAL_CHAIN)

print("\n" + "=" * 58)
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILED -> {FAILS}")
    sys.exit(1)
print("RESULT: all checks passed")
