"""Planners: who proposes the next goal.

Two implementations behind one protocol:

* ChecklistPlanner - deterministic. Follows the story spine; takes a
  detour only when explicitly enabled and the situation allows. This is
  the default and the fallback: the game is beatable with no model at all.

* LLMPlanner - asks a local model (OpenAI-compatible endpoint, e.g.
  llama.cpp serving Qwen on the Orin) to propose a goal as JSON. The
  context it needs is ~a few thousand tokens of *game state* - not an
  agent framework's 64K of scaffolding. Its proposals are validated by
  goals.validate_goal before execution; anything malformed, impossible
  or stale falls back to the checklist. Low temperature, small token
  budget: routine supervision does not need deliberation, and expensive
  thinking is reserved for stuck-recovery (future work).

  Every LLMPlanner round (proposal, verdict, decision, and - via the
  executor's outcome hook - the result) is appended to a SupervisorLog
  JSONL: that log is the future fine-tuning set, so supervision is
  measured before it is trained (see docs/supervisor-training.md).
  Logging defaults to ~/workspace/pokemon-play/supervisor_log.jsonl and
  is disabled with log_path=None. The checklist path never logs.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Optional

from .goals import DETOURS, Goal, next_step, step_goal, validate_goal
from .memory import EpisodicMemory
from .state import GameState
from .supervisor_log import DEFAULT_SUPERVISOR_LOG, SupervisorLog
from .world import World


class ChecklistPlanner:
    def __init__(self, allow_detours: bool = False):
        self.allow_detours = allow_detours

    def propose(self, state: GameState, world: World,
                memory: EpisodicMemory) -> Optional[Goal]:
        step = next_step(state, memory, world)
        if step is None:
            return None
        if self.allow_detours and "surf" in state.capabilities():
            for detour in DETOURS:
                if not detour.done(state, memory, world):
                    goal = step_goal(detour, world)
                    goal.kind = "detour"
                    if validate_goal(goal, state, world, memory).ok:
                        return goal
        return step_goal(step, world)


SYSTEM_PROMPT = """You are the supervising planner for a Pokemon Red playthrough.
You never control the game directly. You propose ONE goal as JSON:
{"target_map_name": "<exact map name>", "reason": "<short>"}
Movement, battles and menus are handled by deterministic executors.
Prefer the next unfinished story step unless the state suggests a detour
(e.g. healing, a legendary Pokemon whose prerequisites are met).
Only propose goals reachable with the listed capabilities."""


class LLMPlanner:
    def __init__(self, endpoint: str, model: str,
                 fallback: Optional[ChecklistPlanner] = None,
                 timeout: float = 30.0,
                 log_path: Optional[str | Path] = DEFAULT_SUPERVISOR_LOG):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.fallback = fallback or ChecklistPlanner()
        self.timeout = timeout
        self.log = SupervisorLog(log_path)
        # (record_id, returned goal key) of the latest logged round,
        # awaiting its outcome via note_outcome().
        self._pending: Optional[tuple] = None

    @staticmethod
    def _context(state: GameState, world: World,
                 memory: EpisodicMemory) -> dict:
        step = next_step(state, memory, world)
        return {
            "state": state.digest(),
            "current_map_name": world.map_name(state.map_id),
            "next_story_step": step.title if step else "game complete",
            "recent_events": memory.recent(6),
        }

    def _ask(self, state: GameState, world: World,
             memory: EpisodicMemory,
             context: Optional[dict] = None) -> Optional[Goal]:
        if context is None:
            context = self._context(state, world, memory)
        body = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(context)},
            ],
        }
        req = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"]
        parsed = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return Goal(
            kind="goto",
            target_map=world.map_id(parsed["target_map_name"]),
            reason=str(parsed.get("reason", ""))[:200],
        )

    def propose(self, state: GameState, world: World,
                memory: EpisodicMemory) -> Optional[Goal]:
        context = self._context(state, world, memory)
        proposal: Optional[Goal] = None
        error: Optional[str] = None
        verdict = None
        decision = "fallback"
        try:
            proposal = self._ask(state, world, memory, context)
        except Exception as e:  # unreachable endpoint, bad JSON, ...
            error = f"{type(e).__name__}: {e}"
        if proposal is not None:
            verdict = validate_goal(proposal, state, world, memory)
            if verdict.ok:
                decision = "llm"
        if decision == "llm":
            returned = proposal
        else:
            returned = self.fallback.propose(state, world, memory)
        record_id = self.log.record_proposal(
            model=self.model, context=context, state=state, world=world,
            step=next_step(state, memory, world), proposal=proposal,
            proposal_error=error, verdict=verdict, decision=decision,
            returned_goal=returned)
        self._pending = ((record_id, returned.key())
                         if record_id is not None and returned is not None
                         else None)
        return returned

    def note_outcome(self, goal: Goal, success: bool, note: str = "",
                     executed: bool = True) -> None:
        """Executor outcome hook: link the result of the goal this
        planner last returned back to its logged proposal record."""
        if self._pending is None:
            return
        record_id, key = self._pending
        if goal.key() != key:
            return
        self.log.record_outcome(record_id, goal_key=key,
                                executed=executed, success=success,
                                note=note)
        self._pending = None
