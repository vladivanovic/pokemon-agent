#!/usr/bin/env python3
"""VLM planner client - the strategy layer of the two-tier agent.

Reads the current screen + RAM state and asks the fine-tuned Qwen2.5-VL-7B
planner LoRA for the SINGLE next high-level goal. The goal is a line from a
closed vocabulary, so the result is always executable by the Von driver.

Design contract (why this shape):
    * The VLM is good at STRATEGY and BAD at multi-tile movement (proven live:
      the 7B picked "walk_right" at a door three tiles away and wedged).
    * So the VLM never emits movement. It emits a goal. The driver turns the
      goal into BFS routes. This is the whole point of the two-tier split.
    * The vocabulary is closed (GOAL_VOCAB). An open-ended planner drifts into
      prose the driver cannot parse.

Inference path: llama.cpp `llama-server --mmproj` serving the base VLM with
the planner LoRA applied (or vLLM). Falls back to a deterministic state-rule
planner when the server is unreachable, so the agent never hard-stops.

Usage:
    from pokemon_agent.planner_client import PlannerClient
    p = PlannerClient("http://127.0.0.1:8002/v1")
    goal = p.plan(frame_png_bytes, state_dict)
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# Closed goal vocabulary - MUST match train_planner_lora.GOAL_VOCAB order.
GOAL_VOCAB = [
    "Exit the house to Pallet Town - walk to the south door.",
    "Walk north from Pallet Town to Viridian City.",
    "Collect Oak's Parcel from Viridian Mart, then return to Oak's Lab in Pallet Town.",
    "Buy supplies at the counter (press A, choose BUY). Get Poke Balls; Pewter Mart also sells Potions.",
    "Travel north toward Pewter City - keep heading north along the path.",
    "Heal the party at the nearest Pokemon Center.",
    "Beat the Gym Leader - attack with the most effective move; heal when HP drops below a third.",
    "Exit the Gym to the city - walk to the south door and step onto it.",
    "Leave Pewter City via its EAST exit to Route 3.",
    "Head EAST along Route 3 - cross the ledges and reach Route 4 at the far east end.",
    "Exit the Pokemon Center to the map outside - walk to the exit door and step onto it.",
    "Exit the Mart to the map outside.",
    "Traverse Mt. Moon from the south entrance to the north exit, then continue to Cerulean City.",
    "Head EAST along Route 4 toward Cerulean City.",
    "Win the battle - pick the most effective move and attack.",
    "Read the dialog and advance it with A (or B to skip).",
    "Take a starter Pokemon from the table in Oak's Lab - stand facing a Pokeball, press A, confirm YES.",
    "Exit this building to Pewter City - walk to the exit door and step onto it, then head EAST to Route 3.",
    "Exit this house to the map outside - walk to the exit door.",
    "Catch an Electric-type in Viridian Forest (Pikachu 5% / Red, Electabuzz 5% / Blue) before Misty's Gym.",
    "Catch a Grass-type in Viridian Forest (Caterpie/Metapod/Red, Oddish/Blue) for Rock/Ground coverage.",
    "Catch a Water- or Grass-type in Route 3 / Viridian Forest before Misty's Gym (Bulbasaur line if starter, or Psyduck in Route 3).",
    "Enter Mt Moon 1F from Route 4 - walk up into the cave entrance and head inside.",
    "Descend from Mt Moon 1F to Mt Moon B1F.",
    "Descend from Mt Moon 1F north half through B1F to the south exits to Route 4.",
    "Cross Mt Moon B1F toward the south exits to Route 4.",
    "Climb from Mt Moon B2F back up to Mt Moon B1F.",
    "Catch an Electric- or Grass-type in Route 4 / Route 24 before Misty (Bellsprout Route 24, Oddish Route 5).",
    "Travel to Cerulean City and challenge Misty.",
    "Catch a Fire/Psychic/Flying-type for Lt. Surge (Abra Route 24/25, Pidgey Route 1/2, Vulpix Route 3/Red).",
    "Progress toward Vermilion City and Lt. Surge.",
    "Continue toward Cerulean City for badge #2.",
    "Progress toward Pewter City and the first Gym.",
]

VOCAB_TEXT = "\n".join(f"- {g}" for g in GOAL_VOCAB)

PROMPT = (
    "You are the strategy layer of a Pokemon Red agent. Look at the game "
    "screen and the RAM state, then state the SINGLE next high-level goal. "
    "Answer with exactly one line from the allowed goal list, nothing else.\n"
    "PROGRESS RULE: move FORWARD along the route - Pewter -> Route 3 -> "
    "Route 4 -> Mt Moon -> Cerulean, and onward. A goal naming a town "
    "BEHIND you is allowed ONLY when it has a concrete reason: healing at "
    "a known Pokemon Center, using Fly, catching a specific Pokemon, or "
    "buying/collecting a needed item. Never emit a goal for a badge you "
    "already hold, and never revisit a town just to 'make progress'.\n"
    "WHERE YOU ARE: the map= field in the RAM state is where you are "
    "standing RIGHT NOW. A goal that names or travels toward that same "
    "map is already satisfied - emitting one wastes the turn. Name the "
    "goal for the NEXT leg of the route OUT of that map instead.\n\n"
    "RAM state: {state}\n\n"
    "Allowed goals:\n{vocab}\n\n"
    "Next goal:"
)


def state_to_text(state: Dict[str, Any]) -> str:
    # The driver passes the raw /state JSON (nested), while the training
    # manifest stores a flat dict. Normalize both shapes here so the prompt
    # always shows real values - feeding the nested dict verbatim rendered
    # pos=(0,0) party_hp=none and the planner answered with early-game goals
    # regardless of the actual screen (proven live at Mt Moon).
    m = state.get("map")
    if isinstance(m, dict):
        map_name = m.get("map_name") or m.get("name") or "?"
    else:
        map_name = m if m else "?"
    p = state.get("player") or {}
    pos = p.get("position") or {}
    x = state.get("x", pos.get("x", 0))
    y = state.get("y", pos.get("y", 0))
    badges = state.get("badges", p.get("badge_count",
                                       len(p.get("badges") or []) or 0))
    battle = state.get("battle") or {}
    in_battle = bool(state.get("in_battle",
                               battle.get("in_battle", False)))
    dlg = state.get("dialog") or {}
    dialog_open = bool(state.get("dialog_open",
                                 dlg.get("active") or
                                 dlg.get("text_active", False)))
    if state.get("party_hp") is not None:
        hp_pairs = list(zip(state.get("party_hp", []),
                            state.get("party_max_hp", [])))
    else:
        hp_pairs = [(int(mon.get("hp") or 0), int(mon.get("max_hp") or 0))
                    for mon in (state.get("party") or [])
                    if int(mon.get("max_hp") or 0) > 0]
    party = [f"{h}/{mx}" for h, mx in hp_pairs if mx > 0]
    flags = state.get("flags") or {}
    extras = []
    if state.get("bag"):
        extras.append("bag=" + ",".join(
            f"{e.get('item')}x{e.get('quantity')}"
            for e in state["bag"] if e.get("item")))
    if state.get("money") is not None or p.get("money") is not None:
        extras.append(f"money={state.get('money', p.get('money', 0))}")
    if flags.get("has_oaks_parcel") or flags.get("oak_got_parcel"):
        extras.append("parcel=delivered" if flags.get("oak_got_parcel")
                      else "parcel=carrying")
    # HEAL-SITE HINT: expose good/bad centers to the VLM
    hs = state.get("heal_sites")
    if hs:
        good = [n for n, r in hs.items() if r.get("status") == "good"]
        bad = [n for n, r in hs.items() if r.get("status") == "bad"]
        if good:
            extras.append(f"heal_good={','.join(good)}")
        if bad:
            extras.append(f"heal_bad={','.join(bad)}")
    bits = [
        f"map={map_name}",
        f"badges={badges}",
        f"pos=({x},{y})",
        f"facing={state.get('facing', p.get('facing', 0))}",
        f"in_battle={in_battle}",
        f"dialog={dialog_open}",
        f"party_hp={','.join(party) if party else 'none'}",
    ] + extras
    return " ".join(bits)


def rule_plan(state: Dict[str, Any]) -> str:
    """Deterministic fallback planner using the same rules as the converter.

    Used when the VLM is unreachable or returns a non-vocab goal. Keeps the
    agent moving instead of halting - a hard stop is worse than a mediocre
    but valid goal.

    Key-shape note: this receives the driver's COMPACT state (von._compact_
    state), whose keys differ from the raw /state JSON the trainer used:
    badges is a LIST of badge names (['Boulder']) or an int, party_hp /
    party_max_hp live under party[i].hp/max_hp, in_battle is a real key,
    and dialog_open does not exist (text_active instead). Reading the
    training-manifest key names here silently degraded every fallback:
    int(['Boulder']) was caught downstream, badges degraded to 0, and the
    rules regressed to pre-badge goals ('Progress toward Pewter') while
    Von stood in Mt Moon with the badge in hand (proven live 12:21).
    """
    from pokemon_agent.planner_rules import derive_goal
    # Shape normalization: the driver passes the FULL /state JSON, while
    # tests and older call sites pass the compact dict. Read both shapes
    # or every field silently degrades (map-as-dict -> '?', badges -> 0,
    # dialog -> False) and the ladder regresses to pre-badge goals
    # (live 07:40: 'Progress toward Pewter' while standing in Pewter with
    # the Boulder Badge - the compact-shape fix covered a call path the
    # live driver does not use).
    m = state.get("map")
    map_name = (m.get("map_name") or m.get("name") or "?") if isinstance(m, dict) else (m or "?")
    p = state.get("player") or {}
    flags = state.get("flags") or {}
    party = state.get("party", [])
    types = [m.get("type1", "").capitalize() for m in party if m.get("type1")]
    # Fallback: compact party dicts carry 'types' (list), not type1.
    if not types:
        for m2 in party:
            for t in (m2.get("types") or []):
                if t:
                    types.append(str(t).capitalize())
    levels = [int(m2.get("level") or 0) for m2 in party]
    hp_list = [int(m2.get("hp") or 0) for m2 in party]
    maxhp_list = [int(m2.get("max_hp") or 0) for m2 in party]
    raw_badges = state.get("badges")
    if raw_badges is None:
        raw_badges = flags.get("badge_count",
                               len(p.get("badges") or []) or 0)
    if isinstance(raw_badges, (list, tuple)):
        badges = len(raw_badges)
    else:
        try:
            badges = int(raw_badges or 0)
        except (TypeError, ValueError):
            badges = 0
    battle = state.get("battle") or {}
    in_battle = bool(state.get("in_battle", battle.get("in_battle", False)))
    dlg = state.get("dialog") or {}
    dialog_open = bool(state.get("text_active") or state.get("dialog_open")
                       or dlg.get("text_active") or dlg.get("input_locked"))
    bag_dict: Dict[str, int] = {}
    for e in (state.get("bag") or []):
        name = str(e.get("item", e.get("Item", ""))).upper()
        if name:
            bag_dict[name] = bag_dict.get(name, 0) + int(e.get("quantity")
                                                        or e.get("Qty") or 0)
    money = state.get("money")
    if money is None:
        money = p.get("money", 0)
    return derive_goal(
        map_name, badges,
        in_battle, dialog_open,
        hp_list, maxhp_list,
        party_types=types,
        party_levels=levels,
        bag_items=bag_dict,
        has_oaks_parcel=bool(flags.get("has_oaks_parcel")),
        oak_got_parcel=bool(flags.get("oak_got_parcel")),
        money=int((state.get("player") or {}).get("money") or 0),
        heal_sites=state.get("heal_sites"))


class PlannerClient:
    def __init__(self, base_url: Optional[str] = None,
                 model: Optional[str] = None, timeout: int = 20):
        self.base_url = (base_url or os.environ.get(
            "POKEMON_PLANNER_URL", "http://127.0.0.1:8002/v1")).rstrip("/")
        self.model = model or os.environ.get(
            "POKEMON_PLANNER_MODEL", "planner")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/models", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def plan(self, frame_png: bytes, state: Dict[str, Any],
             temperature: float = 0.1,
             ascii_map: Optional[str] = None) -> str:
        """Return the next goal (always a valid vocabulary line)."""
        prompt = PROMPT.format(state=state_to_text(state), vocab=VOCAB_TEXT)
        if ascii_map:
            # Symbolic local map: the GB screenshot is 160x144 of tiny
            # sprites and the 7B's vision is its weak side - the ascii
            # gives walls/doors/exits in text it reads natively.
            prompt += ("\n\nLOCAL MAP (9 rows x 10 cols, @=you, #=wall, "
                       ".=walkable, D=door, N=person):\n" + ascii_map)
        b64 = base64.b64encode(frame_png).decode()
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        try:
            r = requests.post(f"{self.base_url}/chat/completions",
                              json=payload, timeout=self.timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            # Handle optional action suffix: "goal | ACTION: action"
            if " | ACTION: " in text:
                text = text.split(" | ACTION: ")[0].strip()
            goal = self._snap_to_vocab(text)
            goal = self._post_filter(goal, state)
            logger.info("planner: %r -> %r", text[:60], goal[:60])
            return goal
        except (requests.RequestException, KeyError, IndexError) as exc:
            logger.warning("planner VLM failed (%s) - using rule plan", exc)
            return rule_plan(state)

    @staticmethod
    def _post_filter(goal: str, state: Dict[str, Any]) -> str:
        """Validation, not strategy. The VLM's word is followed to the letter.

        Vlad's architecture rule: the planner decides WHAT (including WHICH
        Pokecenter); the deterministic layer may only (a) normalize map
        names the router can resolve, and (b) veto IMPOSSIBLE goals -
        impossible means a fact of the game state makes the goal
        unachievable, not 'a heuristic prefers something else'. Strategy
        rewrites (badge ladders, forced heal/progress lines) are REMOVED -
        they pre-empted the VLM and wedged Mt Moon for 12h (proven live
        18:33-06:12); the heal-site preference now flows as state INFO
        (heal_good=/heal_bad=) the model reads and weighs itself.
        """
        if not goal:
            return goal

        # --- (a) DESTINATION NORMALIZATION (HOW-support, not WHAT) ---
        # 'Pewter City Pokecenter' is not a map; 'Pewter Pokecenter' is.
        # Same intent, routable name - the VLM's choice is preserved.
        g_low = goal.lower()
        if "pokecenter" in g_low:
            for real in ("Pewter Pokecenter", "Cerulean Pokecenter",
                         "Viridian Pokecenter", "Route 4 Pokecenter",
                         "Mt Moon Pokecenter"):
                city = real.split(" Pokecenter")[0].split(" Mt ")[0]
                wrong = f"{city} City Pokecenter"
                if wrong.lower() in g_low and real.lower() not in g_low:
                    goal = goal.replace(wrong, real)
                    break

        # --- (b) IMPOSSIBILITY VETOES ONLY ---
        flags = state.get("flags") or {}
        delivered = bool(flags.get('oak_got_parcel'))
        carrying = bool(flags.get('has_oaks_parcel'))
        # Parcel delivery already happened: 'Collect' can never re-run (the
        # Mart clerk has no parcel left to give), but 'Return/Deliver' is
        # still the goal UNTIL the event fires... which it has. So both
        # phases are dead once oak_got_parcel is set.
        if delivered and "Oak's Parcel" in goal:
            return ""
        return goal

    @staticmethod
    def _snap_to_vocab(text: str) -> str:
        """Map a free-form reply onto the closest vocabulary entry.

        The model is trained to emit a verbatim line, but a stray prefix or
        partial line must still resolve to something executable. Substring
        match on the distinctive head of each goal is enough in practice.
        """
        low = text.lower().strip()
        for g in GOAL_VOCAB:
            if g.lower() in low or low in g.lower():
                return g
        # fuzzy: pick the vocab line sharing the most words
        best, best_score = GOAL_VOCAB[0], -1
        words = set(low.split())
        for g in GOAL_VOCAB:
            score = len(words & set(g.lower().split()))
            if score > best_score:
                best, best_score = g, score
        return best