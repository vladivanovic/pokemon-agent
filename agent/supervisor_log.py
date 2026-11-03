"""Append-only JSONL log of LLM supervisor proposals and their outcomes.

This log is the future training set for a fine-tuned supervisor (see
docs/supervisor-training.md), so every record is self-contained: the
exact context the model was shown, what it proposed, what the validator
said, what the planner actually returned, and - linked by record_id -
what happened when the executor ran it. Two record kinds share the file:

  {"kind": "supervisor_proposal", "record_id": ..., "context": {...},
   "state": {...}, "proposal": {...} | null, "verdict": {...} | null,
   "decision": "llm" | "fallback", "returned_goal": {...} | null, ...}
  {"kind": "supervisor_outcome", "record_id": ..., "goal": <goal key>,
   "executed": bool, "success": bool, "note": ...}

Only LLMPlanner writes here. The deterministic path never touches this
module, and a SupervisorLog with path=None is a clean no-op, so runs
that do not use the LLM planner are unaffected byte for byte.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# The play workspace is where run logs already live; the training set
# gets built from this file by scripts/build_supervisor_dataset.py.
DEFAULT_SUPERVISOR_LOG = (Path.home() / "workspace" / "pokemon-play"
                          / "supervisor_log.jsonl")


def _now() -> tuple:
    t = time.time()
    return t, datetime.fromtimestamp(t, timezone.utc).isoformat()


def goal_to_json(goal, world) -> Optional[dict]:
    """Typed goal as JSON - the form a training example is built from."""
    if goal is None:
        return None
    return {
        "kind": goal.kind,
        "target_map": goal.target_map,
        "target_map_name": (world.map_name(goal.target_map)
                            if goal.target_map in world.maps else None),
        "target_xy": list(goal.target_xy) if goal.target_xy else None,
        "reason": goal.reason,
        "optional": goal.optional,
    }


def state_summary(state, world, step=None) -> dict:
    """Compact, self-contained snapshot of the state a proposal was
    made in: where the player is, what they have, how the run is going.
    (GameState reads no item bag, so money + capabilities stand in for
    inventory - they are what the validator's routing actually uses.)"""
    return {
        "map_id": state.map_id,
        "map_name": (world.map_name(state.map_id)
                     if state.map_id in world.maps else None),
        "x": state.x,
        "y": state.y,
        "in_battle": state.in_battle,
        "badges": sorted(state.badges),
        "capabilities": sorted(state.capabilities()),
        "money": state.money,
        "checklist_step": ({"id": step.step_id, "title": step.title}
                           if step is not None else None),
        "party": [
            {
                "species_id": m.species_id,
                "level": m.level,
                "hp": m.hp,
                "max_hp": m.max_hp,
                "hp_frac": round(m.hp_frac, 2),
                "moves": list(m.moves),
                "pp": list(m.pp),
            }
            for m in state.party
        ],
    }


class SupervisorLog:
    """Append-only writer. path=None disables (every method no-ops)."""

    def __init__(self, path: Optional[str | Path] = DEFAULT_SUPERVISOR_LOG):
        self.path = Path(path) if path is not None else None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def _append(self, record: dict) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def record_proposal(self, *, model: str, context: dict, state, world,
                        step, proposal, proposal_error: Optional[str],
                        verdict, decision: str, returned_goal
                        ) -> Optional[str]:
        """Log one proposal round. Returns the record_id (None when
        disabled) so the outcome can be linked back to it."""
        if not self.enabled:
            return None
        t, ts = _now()
        record_id = uuid.uuid4().hex
        self._append({
            "kind": "supervisor_proposal",
            "record_id": record_id,
            "t": t,
            "ts": ts,
            "planner": "llm",
            "model": model,
            # The exact context the model was shown - this becomes the
            # training example's user turn verbatim.
            "context": context,
            "state": state_summary(state, world, step),
            "proposal": goal_to_json(proposal, world),
            "proposal_error": proposal_error,
            "verdict": ({"ok": verdict.ok,
                         "problems": list(verdict.problems),
                         "missing_capabilities":
                             list(verdict.missing_capabilities)}
                        if verdict is not None else None),
            "decision": decision,
            "returned_goal": goal_to_json(returned_goal, world),
        })
        return record_id

    def record_outcome(self, record_id: Optional[str], *, goal_key: str,
                       executed: bool, success: bool, note: str = "") -> None:
        if not self.enabled or record_id is None:
            return
        t, ts = _now()
        self._append({
            "kind": "supervisor_outcome",
            "record_id": record_id,
            "t": t,
            "ts": ts,
            "goal": goal_key,
            "executed": executed,
            "success": success,
            "note": note,
        })

    @staticmethod
    def read_records(path: str | Path) -> List[dict]:
        """All records from a log file; missing file reads as empty."""
        p = Path(path)
        if not p.exists():
            return []
        records = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
