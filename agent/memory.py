"""Episodic memory: a small JSONL log of what happened, read back at
planning time. The old system had no durable memory of its own failures,
so it retried the same doomed ideas in circles. This one remembers.

Kept deliberately dumb: append-only events, a few queries. If it is not
written here, the planner does not get to believe it happened.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


class EpisodicMemory:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self.events: List[dict] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if line:
                    self.events.append(json.loads(line))

    # -- writing ------------------------------------------------------------
    def record(self, kind: str, **data) -> None:
        ev = {"t": time.time(), "kind": kind, **data}
        self.events.append(ev)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(ev) + "\n")

    def record_visit(self, map_id: int) -> None:
        if not self.visited_maps or self.visited_maps[-1] != map_id:
            self.record("visit", map=map_id)

    def record_goal_result(self, goal_key: str, success: bool,
                           state_signature: str, note: str = "") -> None:
        self.record("goal_result", goal=goal_key, success=success,
                    sig=state_signature, note=note)

    # -- reading --------------------------------------------------------------
    @property
    def visited_maps(self) -> List[int]:
        return [e["map"] for e in self.events if e["kind"] == "visit"]

    def visited(self, map_id: int) -> bool:
        return map_id in self.visited_maps

    def visited_after(self, first: int, later: int) -> bool:
        """True if `later` was visited at some point after `first`."""
        seen_first = False
        for m in self.visited_maps:
            if m == first:
                seen_first = True
            elif m == later and seen_first:
                return True
        return False

    def failures(self, goal_key: str, state_signature: str) -> int:
        """How many times this goal failed under an unchanged situation,
        counting only failures since the last forgive() for the goal."""
        forgiven_at = -1
        for i, e in enumerate(self.events):
            if e["kind"] == "forgive" and e["goal"] == goal_key:
                forgiven_at = i
        return sum(
            1 for i, e in enumerate(self.events)
            if i > forgiven_at
            and e["kind"] == "goal_result" and e["goal"] == goal_key
            and not e["success"] and e["sig"] == state_signature
        )

    def forgive(self, goal_key: str) -> None:
        """Reset a goal's failure budget after an outside intervention
        changed the situation (e.g. a dialog was cleared by hand): the
        earlier failures no longer describe the current attempt."""
        self.record("forgive", goal=goal_key)

    def succeeded(self, goal_key: str) -> bool:
        return any(
            e["kind"] == "goal_result" and e["goal"] == goal_key and e["success"]
            for e in self.events
        )

    def recent(self, limit: int = 10) -> List[dict]:
        return self.events[-limit:]


def state_signature(state) -> str:
    """What about the world is relevant to whether a failed goal deserves a
    retry: badges, capabilities, map. If none changed, a retry is a loop."""
    return "|".join([
        ",".join(sorted(state.badges)),
        ",".join(sorted(state.capabilities())),
        str(state.map_id),
    ])
