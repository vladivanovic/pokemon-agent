    ## [SECTION 01] DRIVER INITIALIZATION ─────────────────────────────────────
import json
import logging
import re
from typing import Any, Dict, Optional

import requests

from .autopilot import HermesDriver

# Import Hermes AIAgent for direct API usage
import sys
sys.path.insert(0, "/home/vlad/.hermes/hermes-agent")
from run_agent import AIAgent

logger = logging.getLogger("pokemon-agent.hermes_api_driver")

# Re-export prompts from autopilot
from .autopilot import TURN_NUDGE, FIRST_TURN_PREFIX, _compact_state


class APIDriver(HermesDriver):
    def __init__(self, server: str, model: Optional[str], provider: Optional[str],
                 turn_delay: float = 1.5, save_every: int = 20,
                 turn_timeout: int = 240):
        super().__init__(server, model, provider, turn_delay, save_every, turn_timeout)
        self._agent: Optional[AIAgent] = None

    def _get_agent(self) -> AIAgent:
        """Get or create the AIAgent instance with proper toolsets."""
        if self._agent is None:
            self._agent = AIAgent(
                model=self.model or "nvidia/nemotron-3-ultra-550b-a55b",
                provider=self.provider or "nvidia",
                enabled_toolsets=["file", "terminal", "web", "vision"],
                session_id=self.session_id or "",
                pass_session_id=True,
            )
        elif self.session_id and getattr(self._agent, "session_id", "") != self.session_id:
            # Session changed, create new agent
            self._agent = AIAgent(
                model=self.model or "nvidia/nemotron-3-ultra-550b-a55b",
                provider=self.provider or "nvidia",
                enabled_toolsets=["file", "terminal", "web", "vision"],
                session_id=self.session_id,
                pass_session_id=True,
            )
        return self._agent

    def step(self) -> None:
        """Perform one turn using the Hermes API instead of subprocess."""
        try:
            # 1. Get current state
            state = self._get("/state").json()

            # 2. Get screenshot
            img_path = "/tmp/pokemon_full_screen.png"
            have_img = False
            try:
                shot = self._get("/screenshot").content
                with open(img_path, "wb") as f:
                    f.write(shot)
                have_img = True
            except Exception as e:
                logger.warning(f"Screenshot failed: {e}")

            # 3. Build prompt
            prompt = TURN_NUDGE.format(
                server=self.server, state=json.dumps(_compact_state(state), indent=2)
            )
            if self.session_id is None:
                prompt = FIRST_TURN_PREFIX.format(server=self.server) + prompt

            # 4. Call Hermes API via AIAgent
            agent = self._get_agent()

            # Build message with image if available
            if have_img:
                user_message = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": "file://" + img_path}},
                ]
            else:
                user_message = prompt

            result = agent.run_conversation(user_message)

            # Capture session_id from result
            if self.session_id is None and "session_id" in result:
                self.session_id = result["session_id"]
                logger.info(f"Hermes session: {self.session_id}")
                self.bind_hermes()
                self.event(
                    type="key_moment",
                    description="Hermes session started",
                    category="milestone",
                )

        except Exception as e:
            logger.error(f"Hermes API error: {e}")
