#!/usr/bin/env python3
"""End-to-end demo with no ROM and no model: the simulator walks the real
world data while the checklist planner drives Pallet Town -> Pewter City.

    python3 run_demo.py

Watch it pick up the starter (simulated Oak event), fetch the parcel from
Viridian Mart, deliver it, and head north to Pewter City.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent.emu import SimEmulator
from agent.executor import Executor
from agent.memory import EpisodicMemory
from agent.planner import ChecklistPlanner
from agent.world import default_world


class DemoEmulator(SimEmulator):
    """Simulates Oak handing over a starter the first time you enter his lab."""

    def press(self, button):
        super().press(button)
        if self.map_id == self.world.map_id("OaksLab"):
            state = self.read_state()
            if not state.party:
                self.give_starter()
                print("  * Prof. Oak gives you a starter Pokemon!")


def main():
    world = default_world()
    emu = DemoEmulator(world, world.map_id("PalletTown"), (5, 6))
    emu.give_money(3000)
    memory = EpisodicMemory()
    executor = Executor(emu, world, ChecklistPlanner(), memory)
    result = executor.run(max_goals=4)
    for entry in result.log:
        print(entry)
    state = emu.read_state()
    print(f"\nstatus={result.status} goals_done={result.goals_done} "
          f"presses={executor.presses}")
    print(f"final position: {world.map_name(state.map_id)} ({state.x}, {state.y})"
          f" party={len(state.party)} money={state.money}")


if __name__ == "__main__":
    main()
