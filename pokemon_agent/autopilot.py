"""Von + planner - the stable import surface.

The implementation lives in :mod:`pokemon_agent.von` (the clean two-tier
engine). This module exists only so the historical import path
``pokemon_agent.autopilot`` keeps working for the CLI, the tests and the
tooling, and so the one legacy keyword name (``planner_mode``) is translated
in a single place.

    Tier 1  planner (VLM | rules)  -> ONE goal        pokemon_agent/planner_client.py
    Tier 2  von.py                 -> execute it       BFS + scorer + battle

von.py IS the source of truth (the slice-build from the legacy monolith
was retired on 2026-10-17; the pre-split history lives in git).
"""
from __future__ import annotations

from typing import Optional

# --- engine -----------------------------------------------------------------
from pokemon_agent.von import (  # noqa: F401
    VonDriver,
    run_autopilot,
    # primitives re-exported for tests / tooling / vlm_brain.py
    _compact_state,
    _progress_fingerprint,
    _path_to,
    _legal_dirs,
    _map_step_toward,
    _map_frontier_step,
    _map_edge_path,
    _path_toward_edge,
    _frontier_targets,
    _pinballed,
    _load_map_walk,
    _grid_open,
    cell_label_local,
    _bearing,
)

# Historical alias: code written against the pre-split engine asks for
# HermesDriver. Von IS the driver now.
HermesDriver = VonDriver

# Prompt constants the legacy LLM API driver (hermes_api_driver.py) imports.
# Von has no LLM turns - battles are deterministic (gen1_battle) - so these
# are inert placeholders kept only so that import succeeds.
TURN_NUDGE = ""
FIRST_TURN_PREFIX = ""
INTRO_VISION_OK = ""
INTRO_VISION_NONE = ""
INTRO_NUDGE = ""
ESCALATION_NUDGE = ""


class PlannerDriver(VonDriver):
    """The CLI passes ``planner_mode=``; VonDriver calls it ``planner=``."""

    def __init__(self, *args, planner_mode: str = "rules",
                 planner: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, planner=planner or planner_mode, **kwargs)


__all__ = [
    "VonDriver", "HermesDriver", "PlannerDriver", "run_autopilot",
    "_compact_state", "_progress_fingerprint", "_path_to", "_legal_dirs",
    "_map_step_toward", "_map_frontier_step", "_map_edge_path",
    "_path_toward_edge", "_frontier_targets", "_pinballed",
    "_load_map_walk", "_load_ledge_edges", "_grid_open",
    "cell_label_local", "_bearing",
]
