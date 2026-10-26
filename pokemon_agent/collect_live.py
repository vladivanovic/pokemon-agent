#!/usr/bin/env python3
"""Live training-data collector for the planner.

While the driver plays, every N turns we already have:
    * the screen (PNG from /screenshot)
    * the RAM state (map, badges, pos, party HP)
    * the goal that was actually executed

That triple is a FUTURE training example with zero manual labeling - the goal
is either the VLM's own output (self-distillation, cheap) or the rule planner's
output (the reference label). This is the flywheel: the longer Von plays, the
more the whole-game distribution is covered, which is exactly the gap
PokeDreamer leaves (it is early-game only).

Output: dataset/live/manifest.jsonl + dataset/live/frames/*.jpg
Same schema as convert_pokedreamer.py, so the two can be concatenated for
training.

Usage: instantiated by PlannerDriver; or run
    python -m pokemon_agent.collect_live --turns 20
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

COLLECT_DIR = Path(os.environ.get("POKEMON_COLLECT_DIR",
                                  "dataset/live"))


class LiveCollector:
    """Append (frame, state, goal) triples to a JSONL manifest as we play."""

    def __init__(self, out_dir: Optional[Path] = None,
                 every_n_turns: int = 5, jpeg_quality: int = 85):
        self.out = Path(out_dir or COLLECT_DIR)
        (self.out / "frames").mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out / "manifest.jsonl"
        self.every = max(1, every_n_turns)
        self.jpeg_quality = jpeg_quality
        self.n = self._count_existing()
        logger.info("LiveCollector: %d existing samples at %s",
                    self.n, self.manifest_path)

    def _count_existing(self) -> int:
        if not self.manifest_path.exists():
            return 0
        with self.manifest_path.open() as f:
            return sum(1 for _ in f)

    def maybe_record(self, turn: int, frame_png: Optional[bytes],
                     state: Dict[str, Any], goal: str) -> None:
        """Record one sample if this turn is a sampling turn."""
        if frame_png is None or turn % self.every != 0:
            return
        try:
            self._record(frame_png, state, goal)
        except Exception:
            logger.exception("live collect failed (non-fatal)")

    def _record(self, frame_png: bytes, state: Dict[str, Any], goal: str):
        from PIL import Image
        import io
        fname = f"live_{self.n:07d}.jpg"
        img = Image.open(io.BytesIO(frame_png)).convert("RGB")
        img.save(self.out / "frames" / fname, quality=self.jpeg_quality)
        row = {
            "file": fname,
            "goal": goal,
            "state": {
                "map": state.get("map", "?"),
                "badges": int(state.get("badges", 0)),
                "x": int(state.get("x", 0)),
                "y": int(state.get("y", 0)),
                "facing": int(state.get("facing", 0)),
                "in_battle": bool(state.get("in_battle")),
                "dialog_open": bool(state.get("dialog_open")),
                "party_hp": state.get("party_hp", []),
                "party_max_hp": state.get("party_max_hp", []),
                # Money + bag: rendered by the inference prompt
                # (planner_client.state_to_text), so training rows must carry
                # them too - the model buys/heals based on these facts.
                "money": int(state.get("money", 0) or 0),
                "bag": [{"item": b.get("item"), "quantity": b.get("quantity")}
                        for b in (state.get("bag") or []) if b.get("item")],
                "ascii": state.get("ascii", ""),
            },
        }
        with self.manifest_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.n += 1
        if self.n % 25 == 0:
            logger.info("live dataset: %d samples", self.n)