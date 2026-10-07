"""
Pluggable scorer interface for low-level decision making.

Supports:
- Laya (original ModernBERT-large based scorer)
- Von (non-autoregressive ModernBERT-base scorer)

Both must implement:
    choice(prompt: str, options: List[str]) -> Dict[str, Any]
        Returns {"choice": int, "scores": List[float]} where choice is the index
        of the selected option and scores are raw scores (or logits) for each option.

The autopilot only uses the choice index; scores can be used for logging or
confidence thresholds in the future.
"""

from __future__ import annotations

import abc
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pokemon-agent.scorer")


class BaseScorer(abc.ABC):
    @abc.abstractmethod
    def choice(self, prompt: str, options: List[str]) -> Dict[str, Any]:
        """Select an option from the list given a prompt.

        Returns a dict with at least the key "choice" (int index of chosen option).
        May also include "scores" (list of floats) for diagnostics.
        """
        raise NotImplementedError


class LayaScorer(BaseScorer):
    """Wrapper around the existing laya Router used in the agent."""

    def __init__(self):
        try:
            from laya import Router  # type: ignore
            self.router = Router()
            logger.info("Laya scorer initialized")
        except Exception as e:
            logger.error("Failed to initialize Laya scorer: %s", e)
            raise

    def choice(self, prompt: str, options: List[str]) -> Dict[str, Any]:
        # Build the Laya-specific questions dict
        questions = {
            "action": {
                "type": "choice",
                "instructions": prompt,
                "criteria": {opt: opt for opt in options},
            }
        }
        try:
            result = self.router.predict({}, questions)  # type: ignore[attr-defined]
        except Exception as e:
            logger.error("Laya predict failed: %s", e)
            # Return a fallback: first option
            return {"choice": 0, "scores": [0.0] * len(options)}

        # Extract the choice index
        answers = result.get("answers", {})
        action = answers.get("action", {})
        choice_key = action.get("choice")
        if choice_key is None:
            logger.warning("Laya returned no choice; defaulting to first option")
            choice_idx = 0
        else:
            # Find the index of the key in options
            try:
                choice_idx = options.index(choice_key)
            except ValueError:
                logger.warning("Laya choice '%s' not in options; defaulting to 0", choice_key)
                choice_idx = 0

        # Laya also returns probabilities; we can convert to scores if needed
        probs = action.get("probabilities", {})
        scores = [probs.get(opt, 0.0) for opt in options] if probs else [0.0] * len(options)
        return {"choice": choice_idx, "scores": scores}


class VonScorer(BaseScorer):
    """Wrapper around the Von SDK (wfzyx/von).

    Requires: pip install von-sdk
    """

    def __init__(self, model_name: str = "wfzyx/von"):
        try:
            # VonClient(local=True) serves the bundled von-1.3 weights on CPU
            # (OpenVINO/CUDA/ROCm/MPS). model_name is kept for API compat -
            # the client does not take a model kwarg.
            from von import VonClient  # type: ignore
            self.client = VonClient(local=True, timeout=120.0)
            logger.info("Von scorer initialized (local backend, %s)",
                        model_name)
        except Exception as e:
            logger.error("Failed to initialize Von scorer: %s", e)
            raise

    def choice(self, prompt: str, options: List[str]) -> Dict[str, Any]:
        # Von's real API is system_one(state, questions) with the SAME
        # question schema Laya uses (Choice type + criteria dict) - not a
        # flat (prompt, options) call. We reuse the Laya schema so the
        # decision pipeline is identical; only the backend differs.
        # The prompt is "state||instructions" when called from the autopilot.
        try:
            state_json, instructions = prompt.split("||", 1)
            state = json.loads(state_json)
        except Exception:
            state, instructions = {}, prompt
        questions = {
            "action": {
                "type": "choice",
                "instructions": instructions,
                "criteria": {opt: opt for opt in options},
            }
        }
        try:
            result = self.client.system_one(state, questions)  # type: ignore[attr-defined]
        except Exception as e:
            logger.error("Von predict failed: %s", e)
            return {"choice": 0, "scores": [0.0] * len(options)}

        # Response: .answers = {"action": ChoiceAnswer(choice: str,
        #               probabilities: {opt: float}, confidence: float)}
        try:
            ans = result.answers.get("action")
        except AttributeError:
            logger.error("unexpected Von response shape: %r", result)
            return {"choice": 0, "scores": [0.0] * len(options)}
        choice_key = getattr(ans, "choice", None)
        probs = getattr(ans, "probabilities", None) or {}
        if choice_key is None:
            logger.warning("Von returned no choice; defaulting to first option")
            return {"choice": 0, "scores": [0.0] * len(options)}
        try:
            choice_idx = options.index(choice_key)
        except ValueError:
            logger.warning("Von choice '%s' not in options; defaulting to 0",
                           choice_key)
            choice_idx = 0
        scores = [float(probs.get(opt, 0.0)) for opt in options]
        return {"choice": choice_idx, "scores": scores}


# Factory
_scorer_instance: Optional[BaseScorer] = None


class ExpertScorer(BaseScorer):
    """Imitation scorer trained on the agent's own decision logs.

    Learns P(option | map_name, phase, on_door, in_battle) from the logged
    (state, chosen) pairs the autopilot emits each turn. Pure stdlib - a
    frequency table with Laplace smoothing, no ML deps. Falls back to the
    base scorer when confidence is low, so it can wrap Laya or Von.
    """

    def __init__(self, base: Optional[BaseScorer] = None,
                 log_path: Optional[str] = None):
        self.base = base
        self.table: Dict[tuple, Dict[str, int]] = {}
        self.pairs: int = 0
        self.log_path = log_path or os.environ.get(
            "POKEMON_DECISION_LOG",
            os.path.expanduser("~/.pokemon-agent/decision_trace.jsonl"))
        self._load()

    def _key(self, state: Dict[str, Any]) -> tuple:
        """Context bucket: (map, phase-class, on_door, in_battle)."""
        return (
            state.get("map_name") or "?",
            "battle" if state.get("in_battle") else (
                "dialog" if state.get("text_active") else (
                    "door" if state.get("on_door") else "explore")),
            bool(state.get("on_door")),
        )

    def _load(self) -> None:
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    state = rec.get("state") or {}
                    chosen = rec.get("chosen")
                    if not chosen:
                        continue
                    k = self._key(state)
                    self.table.setdefault(k, {})
                    self.table[k][chosen] = self.table[k].get(chosen, 0) + 1
                    self.pairs += 1
            if self.pairs:
                logger.info("Expert scorer: %d decision pairs from %s "
                            "(%d contexts)", self.pairs, self.log_path,
                            len(self.table))
        except FileNotFoundError:
            logger.info("Expert scorer: no decision trace at %s yet - "
                        "starts cold, defers to base scorer",
                        self.log_path)
        except Exception as e:
            logger.warning("Expert scorer load failed: %s", e)

    def record(self, state: Dict[str, Any], chosen: str) -> None:
        """Append one decision to the trace file (called by the autopilot)."""
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"state": state, "chosen": chosen})
                        + "\n")
            k = self._key(state)
            self.table.setdefault(k, {})
            self.table[k][chosen] = self.table[k].get(chosen, 0) + 1
            self.pairs += 1
        except Exception as e:
            logger.debug("expert record failed: %s", e)

    def choice(self, prompt: str, options: List[str]) -> Dict[str, Any]:
        # prompt is "state||instructions" when called from the autopilot;
        # the state part is JSON. Parse it out.
        state: Dict[str, Any] = {}
        try:
            head = prompt.split("||", 1)[0]
            state = json.loads(head)
        except Exception:
            pass
        k = self._key(state)
        freqs = self.table.get(k, {})
        total = sum(freqs.get(o, 0) for o in options)
        # Laplace-smoothed scores; need at least 8 observed pairs in this
        # context before the table outranks the base scorer.
        if total >= 8:
            scores = [freqs.get(o, 0) + 1.0 for o in options]
            best = max(range(len(scores)), key=lambda i: scores[i])
            return {"choice": best, "scores": scores}
        # Cold context - defer to the wrapped base scorer (or first option).
        if self.base is not None:
            return self.base.choice(prompt, options)
        return {"choice": 0, "scores": [0.0] * len(options)}


def get_scorer(name: str = "laya") -> BaseScorer:
    """Return a singleton scorer instance based on name.

    Args:
        name: "laya", "von", or "expert[:base]" (e.g. "expert:von").

    Returns:
        An instance of BaseScorer.
    """
    global _scorer_instance
    if _scorer_instance is None:
        name_lower = name.lower()
        if name_lower.startswith("expert"):
            # expert[:base] - e.g. "expert:von" wraps a Von scorer
            base_name = name_lower.split(":", 1)[1] if ":" in name_lower else "laya"
            base = get_scorer(base_name)
            _scorer_instance = ExpertScorer(base=base)
        elif name_lower == "laya":
            _scorer_instance = LayaScorer()
        elif name_lower == "von":
            _scorer_instance = VonScorer()
        else:
            raise ValueError(f"Unknown scorer '{name}'. "
                             "Choose 'laya', 'von', or 'expert[:base]'.")
    return _scorer_instance