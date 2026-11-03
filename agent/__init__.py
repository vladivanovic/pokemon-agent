"""pokemon-agent: a small, deterministic-first Pokemon Red player.

Architecture (see README.md):
  world.py    - static world model built from ROM-derived data (never learned live)
  state.py    - game state read from emulator RAM
  goals.py    - structured goals, story checklist, propose-and-verify validation
  planner.py  - who decides what to do next (rules checklist, or an LLM)
  executor.py - event-driven loop that carries goals out with BFS + macros
  emu.py      - emulator interface + a simulator that runs on the real world data
  memory.py   - episodic memory: what was tried, what happened
"""
