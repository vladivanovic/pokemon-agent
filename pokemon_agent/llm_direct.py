#!/usr/bin/env python3
"""Schema-constrained action selection against a local llama.cpp server.

Why this exists
---------------
The escalation path shells out to `hermes chat`, which owns its own tool loop
and therefore cannot be given a decoder constraint. The small local model then
emits malformed tool JSON - proven at 8,724 prompt tokens on BOTH Qwen3.5-0.8B
and 2B (the 2B even invents fields like "new_pos"/"status"). Each malformed
reply becomes a retry, and the retries are what blow the per-turn timeout.

llama.cpp supports json_schema-constrained decoding. Constraining the choice to
the exact set of options THIS turn produced valid JSON from both models at
8.7k tokens (0.8B 4.5s, 2B 6.4s).

This module talks to the server directly (OpenAI-compatible) so the enum can be
built per turn from the live criteria set.

Config (env):
  POKEMON_LLM_URL    default http://127.0.0.1:8001/v1   (0.8B server)
  POKEMON_LLM_MODEL  default Qwen3.5-0.8B
  POKEMON_LLM_TIMEOUT default 60

Note: schema-valid is not the same as correct. The grammar removes malformed
output; the prompt still has to say what to do. Both models picked `walk_right`
for a "walk north" instruction in testing.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

DEFAULT_URL = os.environ.get("POKEMON_LLM_URL", "http://127.0.0.1:8001/v1")
DEFAULT_MODEL = os.environ.get("POKEMON_LLM_MODEL", "Qwen3.5-0.8B")
DEFAULT_TIMEOUT = float(os.environ.get("POKEMON_LLM_TIMEOUT", "60"))

# The full action vocabulary the emulator accepts. Used when constraining a
# raw action SEQUENCE (as opposed to choosing among named options).
ACTION_VOCAB: List[str] = [
    "walk_up", "walk_down", "walk_left", "walk_right",
    "press_a", "press_b", "press_start",
    "wait_30", "wait_60",
]


def build_choice_schema(options: Sequence[str]) -> Dict[str, Any]:
    """Schema for 'pick one of these options' with a per-turn enum.

    A single-property object rather than a bare enum string: llama.cpp's
    grammar sampler binds json_schema to an object, and the wrapper leaves
    room for a confidence/why field without changing the parse path.

    Options are deduped and sorted for a deterministic grammar - a stable
    enum means the KV prefix stays cacheable across turns.
    """
    opts = sorted({str(o) for o in options if o})
    if not opts:
        opts = ["wait_30"]
    return {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": opts},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["choice"],
        "additionalProperties": False,
    }


def build_action_schema(actions: Optional[Sequence[str]] = None,
                        max_steps: int = 8) -> Dict[str, Any]:
    """Schema for an action SEQUENCE, restricted to the legal vocabulary."""
    vocab = sorted({str(a) for a in (actions or ACTION_VOCAB) if a})
    if not vocab:
        vocab = ["wait_30"]
    return {
        "type": "object",
        "properties": {
            "actions": {"type": "array", "minItems": 1,
                        "maxItems": int(max_steps),
                        "items": {"type": "string", "enum": vocab}},
        },
        "required": ["actions"],
        "additionalProperties": False,
    }


def _post(url: str, model: str, system: str, user: str, schema: Dict[str, Any],
          timeout: float, max_tokens: int, temperature: float) -> Optional[str]:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Cache is disabled: the enum changes every turn, and a stale prefix
        # would be re-validated against the wrong grammar.
        "cache_prompt": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "choice", "strict": True, "schema": schema},
        },
    }
    try:
        r = requests.post(f"{url.rstrip('/')}/chat/completions",
                          json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as exc:
        logger.warning("llm_direct: request failed: %s", exc)
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("llm_direct: bad response shape: %s", exc)
    return None


def constrained_choice(options: Sequence[str], system: str, user: str,
                       url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                       timeout: float = DEFAULT_TIMEOUT,
                       max_tokens: int = 32,
                       temperature: float = 0.0
                       ) -> Tuple[Optional[str], float]:
    """Pick one of *options*. Returns (choice, confidence).

    The grammar guarantees the reply is one of *options*; the membership check
    afterwards is belt-and-braces for a server that ignored the schema.
    """
    opts = sorted({str(o) for o in options if o})
    if not opts:
        return None, 0.0
    if len(opts) == 1:
        return opts[0], 1.0
    raw = _post(url, model, system, user, build_choice_schema(opts),
                timeout, max_tokens, temperature)
    if not raw:
        return None, 0.0
    try:
        parsed = json.loads(raw)
    except ValueError:
        logger.warning("llm_direct: unparseable reply %r", raw[:120])
        return None, 0.0
    choice = parsed.get("choice")
    conf = parsed.get("confidence")
    if choice not in opts:
        logger.warning("llm_direct: %r not in enum (server ignored schema?)",
                       choice)
        return None, 0.0
    try:
        conf = float(conf)
        conf = min(max(conf, 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 1.0
    return choice, conf


def constrained_actions(system: str, user: str,
                        legal: Optional[Sequence[str]] = None,
                        max_steps: int = 8,
                        url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                        timeout: float = DEFAULT_TIMEOUT,
                        temperature: float = 0.0) -> Optional[List[str]]:
    """Return a validated action sequence, or None."""
    raw = _post(url, model, system, user,
                build_action_schema(legal, max_steps),
                timeout, max_steps * 12, temperature)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    acts = parsed.get("actions")
    if not isinstance(acts, list) or not acts:
        return None
    vocab = set(legal) if legal else set(ACTION_VOCAB)
    acts = [a for a in acts if a in vocab]
    return acts or None


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)
    opts = ["move_0", "move_1", "move_2", "move_3", "flee"]
    sysmsg = ("You are a Pokemon Red (Gen 1) battle advisor. Choose the best "
              "action from the allowed list.")
    user = ("ENEMY: Weedle Lv5 BUG/POISON. YOUR POKEMON: Ivysaur GRASS/POISON.\n"
            "move_0 Tackle NORMAL power=35 pp=17\n"
            "move_1 Growl NORMAL power=0 pp=40\n"
            "move_2 Leech Seed GRASS power=0 pp=10\n"
            "move_3 Vine Whip GRASS power=35 pp=7\n"
            "Which action?")
    print("server:", DEFAULT_URL, "model:", DEFAULT_MODEL)
    t = time.time()
    ch, conf = constrained_choice(opts, sysmsg, user)
    print(f"  choice={ch!r} conf={conf:.2f} in {time.time()-t:.1f}s "
          f"(oracle says move_0: Grass is 0.25x vs BUG/POISON, Normal 1.0x)")

    # The decisive case: Tackle has no PP, so move_0 must NOT be selectable.
    opts2 = ["move_1", "move_2", "move_3", "flee"]
    user2 = user.replace("pp=17", "pp=0")
    t = time.time()
    ch2, _ = constrained_choice(opts2, sysmsg, user2)
    print(f"  (Tackle PP=0) choice={ch2!r} in {time.time()-t:.1f}s "
          f"- grammar makes move_0 impossible")
