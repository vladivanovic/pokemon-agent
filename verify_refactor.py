import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pokemon_agent.autopilot import (HermesDriver, PlannerDriver,
                                     run_autopilot, _pinballed, _map_step_toward)
from pokemon_agent.planner_client import PlannerClient
from pokemon_agent.planner_rules import derive_goal
from pokemon_agent.hermes_api_driver import _compact_state

print("All imports OK")
print("  HermesDriver:", HermesDriver.__module__)
print("  PlannerDriver:", PlannerDriver.__module__)

# Verify the rule planner produces valid vocabulary goals
cases = [
    ("Pallet Town", 0, False, False, [20], [20]),
    ("Route 2", 0, False, False, [4], [30]),
    ("Mt Moon Pokecenter", 1, False, False, [98], [98]),
    ("Route 3", 1, False, False, [98], [98]),
]
for c in cases:
    print(f"  {c[0]:20s} -> {derive_goal(*c)[:60]}")
