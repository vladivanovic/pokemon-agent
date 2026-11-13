#!/usr/bin/env python3
"""directives.py - the live redirect channel for play_full_game.py.

Enabled with `--directives FILE`. While a full-game run is playing,
any external process - Hermes, another LLM, a script, or the viewer
page (POST /directive appends to the same file) - can steer it by
appending JSON Lines to FILE:

    {"goal": {"target_map_name": "ViridianCity", "reason": "detour"}}
    {"action": "pause"}
    {"action": "resume"}
    {"action": "stop"}

Rules of the channel (the propose-and-verify discipline, applied to
outsiders exactly as to the built-in supervisor):

* A goal directive carries a typed goal in the repo's own format -
  the same shape agent.supervisor_log.goal_to_json writes and the
  LLM planner emits: {"kind": "goto", "target_map": <id>} or
  {"target_map_name": "<exact name>"}, plus optional "target_xy":
  [x, y], "reason" and "optional". It is parsed into agent.goals.Goal
  and run through goals.validate_goal against the LIVE state before
  anything moves. An accepted goal pre-empts the storyline plan: it
  executes through the executor (via a one-shot planner, so battles,
  dialogs, validation and outcome recording are the executor's own),
  then the storyline resumes - the checklist planner re-derives the
  next step from RAM, so a redirect that moved the player is absorbed
  the same way a detour is. A rejected goal is logged WITH the
  validator's reasons and the run continues undisturbed. Nothing
  executes unvalidated, ever.
* Granularity: directives are serviced at phase boundaries and
  between executor rounds inside the planner-driven stretches the
  runner drives (the opening loop, the Vermilion loop). The ported
  leg functions are atomic - a directive that lands mid-leg is picked
  up when the leg ends. A redirect is an inserted validated goal
  between phases/rounds, not mid-leg surgery.
* "pause" takes effect at the next service point and blocks there,
  polling the file, until "resume" or "stop" arrives. Goal directives
  received while paused are queued and executed, in file order, after
  the resume. "stop" ends the run cleanly at the next service point:
  the runner saves run/stopped.state and reports how to resume.
* Consumption is exactly-once: the byte offset is persisted next to
  the run (run/directives.offset), so a --resume restart does not
  replay old directives. Only complete lines are consumed - a writer
  caught mid-append is finished next poll.

Every directive is written to the run's progress.log (via the host's
note()) and to a per-run audit, run/directives-audit.jsonl (never the
channel file itself - the channel would eat its own audit). Goal
directives
additionally land in the run's supervisor_log.jsonl through
agent.supervisor_log, tagged planner="external" - the same records
the built-in LLM supervisor produces, so external supervision is
measured the same way.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent.goals import Goal, validate_goal
from agent.supervisor_log import goal_to_json

ACTIONS = ("pause", "resume", "stop")


# ------------------------------------------------------------ goal JSON

def goal_from_json(data, world) -> Goal:
    """Parse a directive's goal object into a typed Goal.

    Accepts the repo's own goal JSON (agent.supervisor_log's
    goal_to_json shape): target by id ("target_map") or by exact map
    name ("target_map_name", the form the LLM planner emits), an
    optional "target_xy": [x, y], "reason", "optional". Only "goto"
    exists in v1 of the goal system, so anything else is refused
    here, loudly, rather than at the executor. Raises ValueError
    with a human-readable reason on any problem."""
    if not isinstance(data, dict):
        raise ValueError("goal must be a JSON object")
    kind = data.get("kind", "goto")
    if kind != "goto":
        raise ValueError(
            f"unsupported goal kind {kind!r} (only 'goto' is implemented)")
    target = None
    if data.get("target_map") is not None:
        try:
            target = int(data["target_map"])
        except (TypeError, ValueError):
            raise ValueError(
                f"target_map {data['target_map']!r} is not a map id")
        if target not in world.maps:
            raise ValueError(f"unknown map id {target}")
    elif data.get("target_map_name"):
        name = data["target_map_name"]
        try:
            target = world.map_id(name)
        except (KeyError, TypeError):
            raise ValueError(f"unknown map name {name!r}")
    if target is None:
        raise ValueError("goal needs target_map or target_map_name")
    xy = data.get("target_xy")
    if xy is not None:
        try:
            xy = (int(xy[0]), int(xy[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            raise ValueError(f"target_xy {xy!r} is not an [x, y] pair")
    return Goal(kind="goto", target_map=target, target_xy=xy,
                reason=str(data.get("reason", ""))[:200],
                optional=bool(data.get("optional", False)))


def append_directive(path, obj) -> None:
    """Append one directive as a JSON line (the writers' door: the
    web endpoint, a shell `echo >>`, another agent). One write() of
    one line, so concurrent appenders do not interleave."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\n")


# --------------------------------------------------------------- source

class DirectiveSource:
    """Exactly-once reader over the append-only directive file.

    The consumed byte offset is persisted to `offset_path` after
    every poll, so a runner restarted with --resume continues where
    the previous process left off instead of replaying directives.
    A line is consumed only once its terminating newline has been
    written (writers append whole lines; a torn tail is next poll's
    business). If the file shrinks below the offset it was replaced:
    a new lineage, read from the start.
    """

    def __init__(self, path, offset_path=None):
        self.path = Path(path)
        self.offset_path = Path(offset_path) if offset_path else None
        self._offset = self._load_offset()

    @property
    def offset(self) -> int:
        return self._offset

    def _load_offset(self) -> int:
        if self.offset_path is not None:
            try:
                return int(self.offset_path.read_text().strip())
            except (OSError, ValueError):
                return 0
        return 0

    def _save_offset(self) -> None:
        if self.offset_path is None:
            return
        self.offset_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.offset_path.with_suffix(".tmp")
        tmp.write_text(str(self._offset))
        tmp.replace(self.offset_path)

    def poll(self) -> list:
        """New items since the last poll, in file order:
        ("directive", dict) for a parsed JSON object,
        ("malformed", raw_line, error) for anything else. Malformed
        lines are consumed too - a bad line must not wedge the
        channel - and reported by the caller."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        if size < self._offset:
            self._offset = 0
        if size == self._offset:
            return []
        with self.path.open("rb") as fh:
            fh.seek(self._offset)
            data = fh.read()
        last_nl = data.rfind(b"\n")
        if last_nl < 0:
            return []
        chunk = data[:last_nl + 1]
        self._offset += len(chunk)
        self._save_offset()
        items = []
        for raw in chunk.decode("utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                items.append(("malformed", line, str(exc)))
                continue
            if not isinstance(obj, dict):
                items.append(("malformed", line, "not a JSON object"))
                continue
            items.append(("directive", obj))
        return items


# -------------------------------------------------------------- channel

class DirectiveChannel:
    """Services directives at the runner's safe points.

    The host is duck-typed (play_full_game.Ctx satisfies it):
        host.world            - agent.world.World
        host.memory           - EpisodicMemory (read at call time)
        host.note(msg)        - progress-log sink
        host.read_state()     - current GameState
        host.execute_goal(goal) -> (success: bool, detail: str)
          runs ONE already-validated goal through the executor.

    service() returns "ok" when the run may proceed, "stop" when a
    stop was requested (directive or the optional stop_check - the
    web Stop button). While paused, service() blocks polling.
    """

    def __init__(self, path, offset_path, host, slog=None,
                 audit_path=None, poll_interval: float = 0.5,
                 stop_check=None):
        if audit_path is not None \
                and Path(audit_path).resolve() == Path(path).resolve():
            # The audit is output; the channel file is input. Sharing
            # a path makes the channel consume its own audit records
            # as directives - a feedback loop (found by smoke test).
            raise ValueError(
                "directive audit path must differ from the channel file")
        self.source = DirectiveSource(path, offset_path)
        self.host = host
        self.slog = slog
        self.audit_path = Path(audit_path) if audit_path else None
        self.poll_interval = poll_interval
        self.stop_check = stop_check
        self.stop_requested = False
        self.paused = False
        self._deferred: deque = deque()

    # -- logging ------------------------------------------------------
    def _note(self, msg: str) -> None:
        self.host.note(f"DIRECTIVE {msg}")

    def _audit(self, directive, result: str, detail: str = "") -> None:
        if self.audit_path is None:
            return
        record = {
            "kind": "directive",
            "t": time.time(),
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": "external",
            "directive": directive,
            "result": result,
            "detail": detail,
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # -- one directive --------------------------------------------------
    def _handle(self, item) -> None:
        kind, *rest = item
        if kind == "malformed":
            raw, err = rest
            self._note(f"MALFORMED ignored ({err}): {raw[:120]}")
            self._audit(raw, "malformed", err)
            return
        obj = rest[0]
        if "action" in obj:
            action = obj["action"]
            if action == "pause":
                self.paused = True
                self._note("pause - run holds at this service point "
                           "until resume/stop")
                self._audit(obj, "action:pause")
            elif action == "resume":
                self.paused = False
                self._note("resume")
                self._audit(obj, "action:resume")
            elif action == "stop":
                self.stop_requested = True
                self._note("stop - run will halt at this service point")
                self._audit(obj, "action:stop")
            else:
                self._note(f"REJECTED unknown action {action!r} "
                           f"(want one of {', '.join(ACTIONS)})")
                self._audit(obj, "rejected", f"unknown action {action!r}")
            return
        if "goal" in obj:
            if self.paused or self._deferred:
                self._deferred.append(obj)
                self._note("goal queued (paused) - runs after resume: "
                           f"{json.dumps(obj['goal'])[:160]}")
                self._audit(obj, "queued")
                return
            self._do_goal(obj)
            return
        self._note("MALFORMED ignored (no 'goal' or 'action' key): "
                   f"{json.dumps(obj)[:120]}")
        self._audit(obj, "malformed", "no 'goal' or 'action' key")

    def _do_goal(self, obj) -> None:
        host = self.host
        raw_goal = obj.get("goal")
        try:
            goal = goal_from_json(raw_goal, host.world)
        except ValueError as exc:
            self._note(f"REJECTED unparseable goal ({exc}): "
                       f"{json.dumps(raw_goal)[:160]}")
            self._audit(obj, "rejected", f"unparseable goal: {exc}")
            self._log_supervisor(raw=obj, goal=None, state=None,
                                 error=str(exc), verdict=None,
                                 decision="external-rejected")
            return
        state = host.read_state()
        verdict = validate_goal(goal, state, host.world, host.memory)
        if not verdict.ok:
            why = "; ".join(verdict.problems)
            self._note(f"REJECTED {goal.key()} - validator: {why}")
            self._audit(obj, "rejected", why)
            self._log_supervisor(raw=obj, goal=goal, state=state,
                                 error=None, verdict=verdict,
                                 decision="external-rejected")
            return
        self._note(f"ACCEPTED {goal.key()} "
                   f"({goal_to_json(goal, host.world)['target_map_name']})"
                   f" - executing before the storyline continues")
        record_id = self._log_supervisor(
            raw=obj, goal=goal, state=state, error=None,
            verdict=verdict, decision="external")
        try:
            success, detail = host.execute_goal(goal)
        except Exception as exc:
            # A redirect must never take the storyline down with it:
            # an execution-side failure is an outcome, not a crash.
            success, detail = False, f"executor raised {exc!r}"
        outcome = "executed" if success else "failed"
        self._note(f"OUTCOME {goal.key()}: {outcome} - {detail}")
        self._audit(obj, outcome, detail)
        if self.slog is not None and record_id is not None:
            self.slog.record_outcome(record_id, goal_key=goal.key(),
                                     executed=True, success=success,
                                     note=detail)

    def _log_supervisor(self, *, raw, goal, state, error, verdict,
                        decision) -> Optional[str]:
        if self.slog is None:
            return None
        if state is None:
            # Unparseable goal: no state snapshot worth logging, and
            # record_proposal summarises state - use the live one.
            try:
                state = self.host.read_state()
            except Exception:
                return None
        return self.slog.record_proposal(
            model="external",
            context={"source": "external", "directive": raw},
            state=state, world=self.host.world, step=None,
            proposal=goal, proposal_error=error, verdict=verdict,
            decision=decision,
            returned_goal=goal if decision == "external" else None,
            planner="external")

    # -- the service loop ----------------------------------------------
    def service(self) -> str:
        """"ok" to proceed, "stop" to halt. Blocks while paused."""
        while True:
            for item in self.source.poll():
                self._handle(item)
                if self.stop_requested:
                    return "stop"
            if self.paused:
                if self.stop_check is not None and self.stop_check():
                    self.stop_requested = True
                    self._note("stop via web control while paused")
                    return "stop"
                time.sleep(self.poll_interval)
                continue
            if self._deferred:
                self._do_goal(self._deferred.popleft())
                if self.stop_requested:
                    return "stop"
                continue
            return "ok"
