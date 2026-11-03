#!/usr/bin/env python3
"""Deterministic goal rules - the fallback planner AND the training labeler.

`derive_goal` is the single source of truth for "what should the agent want
right now, given RAM state". It is used in three places so they can never
drift apart:

    1. convert_pokedreamer.py - labels every training frame offline
    2. planner_client.rule_plan - fallback when the VLM server is down
    3. tests / evaluation - the reference the VLM is scored against

It is a PURE function of state: no I/O, no globals. That is what makes it
safe to use as a labeler (every frame gets a goal for free, deterministically).
"""

from __future__ import annotations

from typing import List


def derive_goal(map_name: str, badges: int, in_battle: bool,
                dialog_open: bool, party_hp: List[int],
                party_max: List[int], party_types: List[str] = None,
                party_levels: List[int] = None,
                has_cut: bool = False, has_flash: bool = False,
                bag_items: dict = None,
                has_oaks_parcel: bool = False,
                oak_got_parcel: bool = False,
                money: int = 0,
                heal_sites: dict = None,
                no_attack_pp: bool = False) -> str:
    """High-level objective for the current state."""
    if in_battle:
        return "Win the battle - pick the most effective move and attack."
    if dialog_open:
        return "Read the dialog and advance it with A (or B to skip)."

    party = [h for h, m in zip(party_hp, party_max) if m > 0]
    # PP exhaustion is NOT hurt (user direction 2026-10-25): with every
    # attack at 0 PP the battle policy pushes through with Leech Seed and
    # STRUGGLE instead of heal-looping - the PP-exhaustion heal trigger
    # caused an endless flee -> Pokecenter -> Zubat grind. The center visit
    # happens on the natural HP rhythm instead (and restores PP anyway).
    hurt = any(0 < h < m * 0.7
               for h, m in zip(party_hp, party_max) if m > 0)
    lead_lvl_unknown = not party

    # HEAL-SITE HINT: if heal_sites data is available, name a GOOD center
    # instead of "nearest". The VLM planner uses this; the fallback rules
    # also respect it so both paths stay aligned.
    if hurt and heal_sites:
        good = [n for n, r in heal_sites.items() if r.get("status") == "good"]
        if good:
            # Name the CENTER MAP exactly as the router knows it. The old
            # code appended " City" to every center, which mangled the
            # non-city ones into unroutable names ("Mt Moon Pokecenter" ->
            # "Mt Moon City Pokecenter", proven live 15:12). City centers
            # keep their plain map name too - _post_filter already
            # normalizes "... City Pokecenter" when a model emits it.
            target = good[0]
            return (f"Heal the party at {target} - walk to {target}.")

    # HEALED INSIDE A POKECENTER: leave. Without this branch the ladder
    # re-emits the (now-complete) heal goal forever - goal COMPLETE fires
    # every 4 seconds and the agent stands in the center all day (proven
    # live 17:22, Mt Moon Pokecenter, full HP, 15+ min of heal-complete
    # loops). The exit line is the proven GOAL_VOCAB exit the driver's
    # exit_building option executes (Pewter Gym, live).
    if "pokecenter" in (map_name or "").lower() and not hurt:
        return "Exit this house to the map outside - walk to the exit door."

    # HURT (or out of attack PP): heal first, before ANY map-specific
    # branch. The late `if hurt:` used to sit after the Mt Moon / Route 4
    # branches, so on those maps the heal goal never fired - the agent
    # ground through the Mt Moon Zubat corridor with an empty Vine Whip,
    # fleeing every wild battle while HP sat at 67/88 (proven live 04:41).
    if hurt:
        return "Heal the party at the nearest Pokemon Center."

    bag = bag_items or {}
    balls = (bag.get("POKE BALL", 0) + bag.get("GREAT BALL", 0)
             + bag.get("ULTRA BALL", 0))
    potions = (bag.get("POTION", 0) + bag.get("SUPER POTION", 0)
               + bag.get("HYPER POTION", 0) + bag.get("MAX POTION", 0))
    # KEY ITEMS for the route ladder (computed here so the badge-aware Gym
    # branch below can see them - Surge's and Celadon's gyms are Cut-gated).
    has_cut = bool(bag.get("HM01") or bag.get("HM01 CUT") or bag.get("CUT"))
    has_surf = bool(bag.get("HM03") or bag.get("HM03 SURF") or bag.get("SURF"))
    has_flash = bool(bag.get("HM05") or bag.get("HM05 FLASH")
                     or bag.get("FLASH"))
    has_flute = bool(bag.get("POKE FLUTE"))
    flags_parcel = has_oaks_parcel
    # CARRYING vs EVENT: EVENT_GOT_OAKS_PARCEL (0xD74E bit1) stays SET
    # forever after the delivery - it is a completion flag, not an
    # inventory flag (proven live: post-delivery flags show
    # has_oaks_parcel=True AND oak_got_parcel=True with an EMPTY bag).
    # The real carry state is the BAG: 'OAKS PARCEL' as an item.
    carrying_parcel = bool(bag.get("OAKS PARCEL")
                           or bag.get("OAK'S PARCEL")) and not oak_got_parcel

    if badges == 0:
        # PARCEL GATE - GLOBAL: while CARRYING the parcel (bag item
        # present, delivery event clear), every map routes to the
        # delivery. The event flags alone lied post-delivery.
        if carrying_parcel:
            if "Oak's Lab" in map_name or "Pallet" in map_name:
                return ("Deliver the parcel to Prof. Oak - walk into Oak's "
                        "Lab (the large building south of Pallet Town) and "
                        "approach him.")
            return ("Deliver Oak's Parcel to Prof. Oak in Pallet Town - "
                    "head SOUTH through Route 1 to Pallet Town and enter "
                    "Oak's Lab.")
        if "Red's House" in map_name or "Reds House" in map_name:
            return "Exit the house to Pallet Town - walk to the south door."
        if "Oak's Lab" in map_name:
            if party:
                # Starter already taken (party non-empty). The lab-goal
                # would loop on the empty table forever otherwise.
                return ("Exit the lab to Pallet Town - walk to the exit "
                        "door and step onto it.")
            return ("Take a starter Pokemon from the table in Oak's Lab - "
                    "stand facing a Pokeball, press A, confirm YES.")
        if "Mart" in map_name:
            # ROM GATE: the Viridian Mart clerk replays the parcel
            # cutscene and never opens the shop menu until
            # EVENT_OAK_GOT_PARCEL (Oak RECEIVED the parcel). CARRYING
            # the parcel is not enough - deliver it to Oak in Pallet
            # first, then come back to buy.
            if carrying_parcel:
                return ("Deliver Oak's Parcel to Prof. Oak in Pallet "
                        "Town - exit the Mart and walk south to Oak's "
                        "Lab. The clerk won't sell anything until the "
                        "parcel is delivered.")
            # MONEY GATE: Poke Balls cost $200 each. If we can't afford
            # even one, skip buying and proceed to the next objective.
            if money < 200:
                # After buying, the next strategic goal is Viridian Forest
                # for Pikachu (Electric-type for Misty).
                if balls > 0 and party_types and "Electric" not in party_types:
                    return ("Catch an Electric-type in Viridian Forest "
                            "(Pikachu 5% / Red, Electabuzz 5% / Blue) before "
                            "Misty's Gym.")
                # Fallback if no balls yet but broke - still go to forest
                return ("Catch an Electric-type in Viridian Forest "
                        "(Pikachu 5% / Red, Electabuzz 5% / Blue) before "
                        "Misty's Gym.")
            # AFTER buying, the next strategic goal is Viridian Forest
            # for Pikachu (Electric-type for Misty). Don't re-buy.
            if balls > 0 and party_types and "Electric" not in party_types:
                return ("Catch an Electric-type in Viridian Forest "
                        "(Pikachu 5% / Red, Electabuzz 5% / Blue) before "
                        "Misty's Gym.")
            return ("Buy supplies at the counter (press A, choose BUY). "
                    "Get Poke Balls; Pewter Mart also sells Potions.")
        if "Gym" in map_name:
            return ("Beat the Gym Leader - attack with the most effective "
                    "move; heal when HP drops below a third.")
        if map_name in ("Route 1", "Route 2", "Viridian Forest"):
            # PARCEL GATE (same as the Mart branch): carrying the parcel
            # undelivered overrides everything - the Mart can't sell
            # until Oak receives it, so buying first is impossible.
            # (Proven live: stepped onto Route 1, planner flipped to
            # 'buy at Viridian Mart' and routed BACK north.)
            if carrying_parcel:
                return ("Deliver Oak's Parcel to Prof. Oak in Pallet "
                        "Town - head SOUTH through Route 1 to Pallet "
                        "Town and enter Oak's Lab.")
            # Strategic catch: need Electric for Misty, Grass for Rock/Ground
            # BUT only if we have Poke Balls - else buy first (no impossible
            # catch orders).
            if balls == 0 and "Mart" in map_name:
                return ("Buy supplies at the counter (press A, choose BUY). "
                        "Get Poke Balls; Pewter Mart also sells Potions.")
            if balls > 0 and party_types and "Electric" not in party_types:
                return ("Catch an Electric-type in Viridian Forest "
                        "(Pikachu 5% / Red, Electabuzz 5% / Blue) before "
                        "Misty's Gym.")
            if balls > 0 and party_types and "Grass" not in party_types and "Water" not in party_types:
                return ("Catch a Grass-type in Viridian Forest "
                        "(Caterpie/Metapod/Red, Oddish/Blue) for Rock/Ground coverage.")
            if balls == 0:
                if money >= 200:
                    return ("Buy Poke Balls at the Viridian Mart before "
                            "hunting - walk to the counter and choose BUY.")
                # $200/ball and the wallet is empty: a Mart trip cannot
                # help. Trainer battles are the only income - fight the
                # Route 1/2 trainers on the way north. The goal MUST name
                # a destination map or _graph_destination returns None and
                # the router falls to free-walk (oscillation, proven live:
                # 10 turns bouncing at Route 2 (11,58) with no waypoint
                # option offered). Battling happens en route regardless -
                # the trainers stand ON the path north.
                return ("Travel north toward Pewter City - fight the "
                        "trainers on the way to earn cash, then buy Poke "
                        "Balls when affordable.")
            # We have balls, we have Electric, or we're broke - travel
            # THROUGH Viridian Forest to reach Pewter City. The goal
            # names "Viridian Forest" so the router targets it directly
            # (the map graph connects Route 1/2 to Viridian Forest).
            return ("Travel north through Viridian Forest toward Pewter City - "
                    "keep heading north along the path, fight or avoid "
                    "trainers as needed.")
        if hurt:
            return "Heal the party at the nearest Pokemon Center."
        if "Viridian" in map_name:
            # MULTI-LEG goal, decomposed by state so the router gets ONE
            # destination per goal: the combined text made
            # _graph_destination resolve 'Oak's Lab' and route EAST out of
            # Viridian (the Lab is south in Pallet!), skipping the Mart.
            if not carrying_parcel and not oak_got_parcel:
                return ("Collect Oak's Parcel from Viridian Mart - walk to "
                        "the Mart counter.")
            if not carrying_parcel and oak_got_parcel:
                # Parcel delivered but not carrying - route to Route 1/Forest
                # for Pikachu hunt (same logic as Route 1/Forest)
                if balls > 0 and party_types and "Electric" not in party_types:
                    return ("Catch an Electric-type in Viridian Forest "
                            "(Pikachu 5% / Red, Electabuzz 5% / Blue) before "
                            "Misty's Gym.")
                if balls == 0 and money >= 200:
                    return ("Buy Poke Balls at the Viridian Mart before "
                            "hunting - walk to the counter and choose BUY.")
                return ("Travel north toward Pewter City - keep heading north "
                        "along the path, fight or avoid trainers as needed.")
            if carrying_parcel:
                return "Return Oak's Parcel to Oak's Lab in Pallet Town."
        if "Museum" in map_name or "House" in map_name:
            return ("Exit this building to the map outside - walk to the "
                    "exit door and step onto it.")
        if "Pallet" in map_name and lead_lvl_unknown:
            return "Walk north from Pallet Town to Viridian City."
        return "Progress toward Pewter City and the first Gym."

    # GYM - badge-aware. A gym whose badge is not yet held is a
    # CHALLENGE target, not an exit target. The old unconditional
    # "exit the Gym" starved every in-gym challenge and re-created the
    # pattern-90 class (a gate that fires regardless of stage).
    GYM_BADGE = {"Pewter": 0, "Cerulean": 1, "Vermilion": 2,
                 "Celadon": 3, "Fuchsia": 4, "Saffron": 5,
                 "Cinnabar": 6, "Viridian": 7}
    if "Gym" in map_name:
        for _city, _need in GYM_BADGE.items():
            if _city in map_name:
                if badges == _need:
                    # CUT GATE: Vermilion and Celadon gyms sit behind a
                    # Cut tree - the challenge is impossible until HM01
                    # comes from the S.S. Anne captain.
                    if _city in ("Vermilion", "Celadon") and not has_cut:
                        return ("Leave this Gym - a Cut tree blocks the "
                                "way. Get HM01 Cut from the S.S. Anne "
                                "captain in Vermilion first.")
                    # HEAL BEFORE A GYM (user request 2026-10-25): walking
                    # in below ~90% HP risks a mid-gym blackout stranding
                    # the run at the entrance - the detour to the nearest
                    # center is always cheaper than the wipe. The lead's
                    # HP is what fights the leader.
                    _ghp = party_hp[0] if party_hp else 0
                    _gmx = party_max[0] if party_max else 1
                    if _gmx and _ghp / _gmx < 0.9:
                        if heal_sites:
                            _hs = sorted(heal_sites)[0]
                            return f"Heal the party at {_hs} - walk to {_hs}."
                        return "Heal the party at the nearest Pokemon Center."
                    return (f"Beat the {_city} Gym Leader - attack with "
                            "the most effective move and heal when low.")
                break
        return ("Exit the Gym to the city - walk to the south door and "
                "step onto it.")
    if "Museum" in map_name:
        return ("Exit this building to the map outside - walk to the exit "
                "door and step onto it.")
    if "House" in map_name:
        # Must precede the city check: "Pewter Nidoran House" contains
        # "Pewter" and would otherwise be told to leave via the city exit.
        return "Exit this house to the map outside - walk to the exit door."
    if "Pokecenter" in map_name:
        return ("Exit the Pokemon Center to the map outside - walk to the "
                "exit door and step onto it.")
    if "Mart" in map_name:
        return "Exit the Mart to the map outside."
    # Mt. Moon is a multi-floor cave, and 1F has TWO disconnected halves
    # joined only through B1F. The goal pair must be PROGRESSIVE, never
    # symmetric: a plain 1F<->B1F cycle (1F "descend" + B1F "climb back")
    # is a stable 2-cycle that neither side ever satisfies - proven live as
    # 19 floor flips in 7 minutes. The rule that breaks the cycle: on 1F,
    # what matters is WHICH HALF you are on. BOTH halves' Route 4 exit is
    # the (14,35)/(15,35) LAST_MAP warp pair at the SOUTH edge (pokered
    # warp table; the (18,5) Route 4 tile is the cave MOUTH on ROUTE 4,
    # not a 1F tile). The north half (y < 16, where entering from Route 4
    # lands you) must first descend through B1F to reach the south exits.
    # (derive_goal takes no position, so the split is done by the caller in
    # VonDriver._rules_goal, which appends the half into map_name.)
    # MT MOON CROSSING (settled 2026-10-25, verified against the live warp
    # table, the static ROM decode AND the fold components):
    #   * Route 4's west and east portions are SEPARATE regions - the
    #     pokered "6-step walk" claim is wrong for this ROM (the static
    #     decode finds no path), so the CAVE IS the only bridge.
    #   * The usable chain, every hop a clean step-onto a DIFFERENT
    #     ladder (never landing on the exit tile itself):
    #       1F (17,11) -> B1F (25,15)   [comp0]
    #       B1F (13,27) -> B2F (5,7)    [both comp0]
    #       B2F (5,7) -> B1F (27,3)     [the exit tile, LAST_MAP]
    #       B1F (27,3) -> Route 4 east -> Cerulean.
    #   * The (14,35) west door is only for the POKECENTER (the heal goal
    #     above fires first when hurt / out of attack PP - proven live
    #     05:17: exit -> center -> Vine Whip 0 -> 8 PP -> back in).
    # THROUGH-CAVE CHAIN (corrected 2026-10-26): Route 4's WEST portion
    # (the cave mouth at x~18 and the Pokecenter) is a sealed pocket - the
    # fold BFS from (18,6) reaches only ~186 cells and can never get to
    # Cerulean, and the walk-edge table's Route 4 -> Cerulean cells sit at
    # (89,19)/(89,21), outside MAP_BOUNDS (90,18). So Cerulean is reached
    # ONLY through the cave, exiting east: B2F -> B1F strip -> (27,3) ->
    # Route 4 EAST -> Cerulean. An "up and out the west door" chain (tried
    # 01:35-01:45) is an infinite crossing loop: it lands the agent on
    # Route 4 west, whose only Cerulean route is back through the cave.
    # The stable B1F<->B2F cycle this chain can form is prevented NOT here
    # but in the driver: the goal names "(5,7)", the only ladder landing
    # in the exit strip, and VonDriver's goal-named ladder override forces
    # the commit onto it (proven live 00:15).
        # Route 4 runs EAST from the Mt Moon exit to Cerulean City.
        #
        # An earlier revision assumed the Route 4 -> Cerulean map-graph
        # edge was geometric-only and told the agent to walk back INTO the
        # cave. That produced an infinite bounce: the agent stepped out of
        # Mt Moon onto Route 4 and was immediately sent back in
        # (Mt Moon 1F <-> Route 4, observed at 16:37 and 16:46 on
        # 2026-10-22, blocking every attempt to reach Cerulean).
        #
        # The ROM walkability grid contradicts that assumption: Route 4 is
        # 90x18 and the tiles reachable from the cave exit form ONE
        # connected component spanning x=0..89 - i.e. the path east to the
        # Cerulean edge is walkable.
        #
        # Name ONLY the destination: _graph_destination picks the LONGEST
        # map-name match in the goal text.
        return ("Travel EAST along Route 4 to Cerulean City - follow the "
                "path east past the ledges.")

    # ==================================================================
    # ROUTE LADDER - badges 1..8 + Elite Four.
    # ==================================================================
    # Written over the WHOLE chain, keyed on the badge count for the
    # STAGE and the current map for the NEXT LEG. A gate that names one
    # location silently expires at the next stage (pattern 90), so every
    # stage is spelled out. This is also the TAS-replay LABEL SOURCE:
    # every frame of a full-game walkthrough gets a goal for free.
    #
    # Key items the route needs (Kanto order):
    #   Cut (HM01)   - S.S. Anne captain   - gates Surge's + Celadon gyms
    #   Flash (HM05) - Route 2 gate (10 mon) - gates Rock Tunnel
    #   Poke Flute   - Lavender Tower (Mr Fuji) - wakes the Snorlax pair
    #   Surf (HM03)  - Safari Zone         - gates Cinnabar + Seafoam
    #   Strength     - Safari Zone         - gates Victory Road boulders
    #   Fly (HM02)   - Route 16 house      - travel shortcut (optional)
    # ==================================================================
    # (key items has_cut/has_surf/has_flash/has_flute are computed above,
    #  beside the bag, so the badge-aware Gym branch can see them too.)

    # ---- BADGE 2: Boulder held, on to Cerulean / Misty ----------------
    if badges == 1:
        # Standing in Pewter proper (not its gym/interiors) with the badge:
        # the next leg is the EAST exit. Without this the ladder fell
        # through to the balls==0 shopping goal - a non-vocab line pointing
        # at the Cerulean Mart two cities ahead - and the agent thrashed
        # in Pewter for hours (live 19:11-19:40).
        if "Pewter" in map_name and "Gym" not in map_name:
            return "Leave Pewter City via its EAST exit to Route 3."
        if "Cerulean" in map_name and "Gym" in map_name:
            return "Beat Misty - attack with the most effective move."
        if "Cerulean" in map_name:
            # APPROACH GATE: same heal-before-gym rule as inside the gym -
            # enter the fight at ~full HP or heal first (user request
            # 2026-10-25).
            _ahp = party_hp[0] if party_hp else 0
            _amx = party_max[0] if party_max else 1
            if _amx and _ahp / _amx < 0.9:
                if heal_sites:
                    _ahs = sorted(heal_sites)[0]
                    return f"Heal the party at {_ahs} - walk to {_ahs}."
                return "Heal the party at the nearest Pokemon Center."
            return ("Challenge Misty in the Cerulean Gym - walk to the "
                    "Gym door and go inside.")
        if "Route 24" in map_name or "Route 25" in map_name:
            return ("Head back SOUTH to Cerulean City - cross Nugget "
                    "Bridge and continue into the city.")
        if "Mt Moon" in map_name or "Route 4" in map_name:
            return ("Progress through Mt Moon (and Route 4 beyond it) to "
                    "reach Cerulean City.")
        if balls == 0 and not hurt:
            # IN-VOCAB shopping goal - but only where a Mart exists (cities).
            # Fired mid-route it wedged the agent in a buy<->transit yo-yo on
            # Route 3 (live 09:50-10:20): the west pull to Pewter Mart
            # cancelled the eastward Mt Moon transit indefinitely. On
            # routes/caves the catch-prep is deferred; the Cerulean Mart
            # covers it after Mt Moon.
            if ("City" in map_name or "Town" in map_name):
                return ("Buy supplies at the counter (press A, choose BUY). "
                        "Get Poke Balls; Pewter Mart also sells Potions.")
        if (balls > 0 and party_types
                and "Electric" not in party_types
                and "Grass" not in party_types):
            return ("Catch an Electric- or Grass-type in Route 4 / Route 24 "
                    "before Misty (Bellsprout Route 24, Oddish Route 5).")
        return "Travel to Cerulean City and challenge Misty."

    # ---- BADGE 3: Cascade held, on to Vermilion / Lt. Surge -----------
    # Surge's gym is walled by a Cut tree, and Cut comes from the S.S.
    # Anne captain - so the Anne leg precedes the gym leg.
    if badges == 2:
        if "Vermilion" in map_name and "Gym" in map_name:
            if not has_cut:
                return ("Leave the Gym - Lt. Surge's gym can only be "
                        "reached after getting HM01 Cut from the S.S. "
                        "Anne captain.")
            return ("Beat Lt. Surge - his Pokemon are Electric; attack "
                    "with Ground moves if you have them.")
        if "Vermilion" in map_name:
            if not has_cut:
                return ("Board the S.S. Anne at the Vermilion harbour and "
                        "get HM01 Cut from the Captain.")
            return ("Challenge Lt. Surge in the Vermilion Gym - walk to "
                    "the Gym door and go inside.")
        if "S.S. Anne" in map_name:
            return ("Find the Captain's room aboard the S.S. Anne and get "
                    "HM01 Cut from him.")
        if "Route 5" in map_name:
            return ("Head SOUTH along Route 5 to the Underground Path "
                    "toward Vermilion City.")
        if "Route 6" in map_name:
            return "Head SOUTH along Route 6 into Vermilion City."
        if "Underground" in map_name:
            return "Take the Underground Path through to the other side."
        if "Route 24" in map_name or "Route 25" in map_name:
            return "Head SOUTH back to Cerulean City."
        if "Cerulean" in map_name:
            return ("Leave Cerulean City to the SOUTH - Route 5 leads "
                    "toward Vermilion City and Lt. Surge.")
        if party_types and "Fire" not in party_types \
                and "Psychic" not in party_types \
                and "Flying" not in party_types:
            return ("Catch a Fire/Psychic/Flying-type for Lt. Surge "
                    "(Abra Route 24/25, Pidgey Route 1/2, Vulpix Route 3).")
        return "Progress toward Vermilion City and Lt. Surge."

    # ---- BADGE 4: Thunder held, on to Celadon / Erika -----------------
    if badges == 3:
        if "Celadon" in map_name and "Gym" in map_name:
            return "Beat Erika - her Pokemon are Grass; use Fire/Flying."
        if "Celadon" in map_name:
            if not has_cut:
                return ("The Celadon Gym is behind a Cut tree - get HM01 "
                        "Cut from the S.S. Anne Captain in Vermilion "
                        "first, then return.")
            return ("Challenge Erika in the Celadon Gym - walk to the "
                    "Gym door, cut the tree if it blocks the way.")
        if "Route 7" in map_name:
            return "Head EAST along Route 7 into Celadon City."
        if "Route 6" in map_name:
            return ("Head NORTH along Route 6; the Underground Path to "
                    "Route 7 and Celadon City is through here.")
        if "Underground" in map_name:
            return "Take the Underground Path through to the other side."
        if "Vermilion" in map_name:
            return ("Leave Vermilion City to the NORTH - Route 6 and the "
                    "Underground Path lead to Celadon City.")
        if "Route 9" in map_name or "Route 10" in map_name:
            return ("Head EAST along Route 9/10 toward Rock Tunnel and "
                    "Lavender Town.")
        if "Rock Tunnel" in map_name:
            return ("Traverse Rock Tunnel to reach Lavender Town (use "
                    "Flash if the cave is dark).")
        if "Lavender" in map_name:
            return ("Head WEST from Lavender Town along Route 8 toward "
                    "Celadon City.")
        return "Travel to Celadon City and challenge Erika."

    # ---- BADGE 5: Rainbow held, on to Fuchsia / Koga ------------------
    # Both southern routes (12 and 16/17) are blocked by a sleeping
    # Snorlax, so the Poke Flute from Lavender Tower comes first.
    if badges == 4:
        if "Fuchsia" in map_name and "Gym" in map_name:
            return "Beat Koga - his Pokemon are Poison; use Psychic/Ground."
        if "Fuchsia" in map_name:
            return ("Challenge Koga in the Fuchsia Gym - walk to the Gym "
                    "door and go inside.")
        if "Pokemon Tower" in map_name or "Tower" in map_name:
            return ("Climb Pokemon Tower and rescue Mr. Fuji - he gives "
                    "you the Poke Flute.")
        if "Lavender" in map_name:
            if not has_flute:
                return ("Enter Pokemon Tower in Lavender Town and rescue "
                        "Mr. Fuji to get the Poke Flute.")
            return ("Head SOUTH from Lavender Town along Route 12 toward "
                    "Fuchsia City (use the Poke Flute on any Snorlax).")
        if "Route 12" in map_name:
            if not has_flute:
                return ("Go back to Lavender Town and get the Poke Flute "
                        "from Mr. Fuji before Route 12 (a Snorlax blocks "
                        "the path).")
            return "Head SOUTH along Route 12 to Fuchsia City."
        if "Route 16" in map_name or "Route 17" in map_name \
                or "Cycling" in map_name:
            return ("Head SOUTH down the Cycling Road to Fuchsia City "
                    "(use the Poke Flute on the Snorlax if it blocks the "
                    "way).")
        if "Celadon" in map_name:
            if not has_flute:
                return ("Head WEST/NORTH from Celadon to Lavender Town "
                        "and rescue Mr. Fuji in Pokemon Tower for the "
                        "Poke Flute.")
            return ("Leave Celadon City toward Fuchsia - Route 16 (west) "
                    "or Route 12 via Lavender both lead there.")
        if "Safari Zone" in map_name:
            return ("Explore the Safari Zone and collect HM03 Surf and "
                    "the Gold Teeth (Strength) from the secret house.")
        return "Travel to Fuchsia City and challenge Koga."

    # ---- BADGE 6: Soul held, on to Saffron / Sabrina ------------------
    # The Saffron gate guard wants a drink from Celadon's vending machine.
    if badges == 5:
        if "Saffron" in map_name and "Gym" in map_name:
            return ("Beat Sabrina - her Pokemon are Psychic; use Bug or "
                    "Ghost moves.")
        if "Silph" in map_name:
            return ("Clear Team Rocket out of Silph Co. - fight up to the "
                    "top floor and free the president.")
        if "Saffron" in map_name:
            return ("Challenge Sabrina in the Saffron Gym - the Gym is "
                    "reached through the city.")
        if "Celadon" in map_name:
            return ("Buy a Fresh Water from the Celadon Dept Store "
                    "vending machine, then head EAST to the Saffron gate "
                    "and give it to the guard.")
        if "Route 7" in map_name:
            return "Head EAST along Route 7 into Saffron City."
        if "Fuchsia" in map_name:
            return ("Leave Fuchsia City to the NORTH - Routes 15/14/13 "
                    "lead toward Lavender Town and Saffron City.")
        if "Safari Zone" in map_name:
            return ("Collect HM03 Surf and the Gold Teeth from the Safari "
                    "Zone's secret house before leaving Fuchsia.")
        if "Lavender" in map_name:
            return ("Head NORTH/WEST from Lavender Town toward Saffron "
                    "City (via Route 8) or Celadon for the drink.")
        return "Travel to Saffron City and clear Silph Co."

    # ---- BADGE 7: Marsh held, on to Cinnabar / Blaine -----------------
    # Cinnabar needs Surf; the Pokemon Mansion holds the gym key.
    if badges == 6:
        if "Cinnabar" in map_name and "Gym" in map_name:
            return ("Beat Blaine - his Pokemon are Fire; use Water/Ground.")
        if "Cinnabar" in map_name and "Mansion" in map_name:
            return ("Search the Pokemon Mansion for the Secret Key that "
                    "opens the Cinnabar Gym.")
        if "Cinnabar" in map_name:
            if not has_surf:
                return ("Surf is needed for the Cinnabar route - get HM03 "
                        "from the Safari Zone in Fuchsia first.")
            return ("Challenge Blaine in the Cinnabar Gym - find the "
                    "Secret Key in the Pokemon Mansion first.")
        if "Pokemon Mansion" in map_name:
            return ("Search the Pokemon Mansion for the Secret Key.")
        if "Fuchsia" in map_name:
            if not has_surf:
                return ("Get HM03 Surf from the Safari Zone before leaving "
                        "Fuchsia City.")
            return ("Surf SOUTH from Fuchsia across Route 19/20 to "
                    "Cinnabar Island.")
        if "Pallet" in map_name:
            return ("Surf SOUTH from Pallet Town across Route 21 to "
                    "Cinnabar Island.")
        if "Route 19" in map_name or "Route 20" in map_name \
                or "Route 21" in map_name or "Seafoam" in map_name:
            return "Surf on to Cinnabar Island."
        return "Travel to Cinnabar Island and challenge Blaine."

    # ---- BADGE 8: Volcano held, on to Viridian / Giovanni -------------
    if badges == 7:
        if "Viridian" in map_name and "Gym" in map_name:
            return ("Beat Giovanni - his Pokemon are Ground; use Water/"
                    "Grass/Ice.")
        if "Viridian" in map_name:
            return ("Challenge Giovanni in the Viridian Gym - walk to the "
                    "Gym door and go inside.")
        if "Cinnabar" in map_name:
            return ("Surf NORTH from Cinnabar back toward Pallet Town, "
                    "then head to Viridian City.")
        if "Pallet" in map_name:
            return ("Head NORTH from Pallet Town through Route 1 to "
                    "Viridian City for the final Gym.")
        if "Route 21" in map_name or "Route 1" in map_name:
            return "Head NORTH to Viridian City."
        return "Travel to Viridian City and challenge Giovanni."

    # ---- ELITE FOUR: all 8 badges held -------------------------------
    if badges >= 8:
        if "Hall of Fame" in map_name or "HallOfFame" in map_name:
            return "You have won - the run is complete!"
        if "Lorelei" in map_name or "Agatha" in map_name \
                or "Bruno" in map_name or "Lance" in map_name \
                or "Elite Four" in map_name or "Indigo" in map_name:
            return ("Beat the Elite Four - attack with the most effective "
                    "move and heal between rooms.")
        if "Victory Road" in map_name:
            return ("Traverse Victory Road - push the Strength boulders "
                    "and reach the Indigo Plateau.")
        if "Route 23" in map_name:
            return "Head NORTH along Route 23 to Victory Road."
        if "Route 22" in map_name:
            return "Head WEST along Route 22 to the Victory Road gate."
        if "Viridian" in map_name:
            return ("Leave Viridian City to the WEST - Route 22 leads to "
                    "Victory Road and the Pokemon League.")
        return ("Travel to the Pokemon League - Route 22, Route 23, "
                "Victory Road, then the Elite Four.")

    # Generic interior fallback - RUNS LAST so it can never shadow a real
    # progression leg. Any remaining building (gates, labs, houses,
    # underground paths) has an exit door; leaving is the right next step
    # when no route goal above matched.
    if map_name not in ("Pallet Town", "Viridian City", "Pewter City",
                        "Cerulean City", "Route 1", "Route 2", "Route 3",
                        "Route 4", "Viridian Forest"):
        return ("Exit this building to the map outside - walk to the exit "
                "door and step onto it.")
    return "Continue toward Cerulean City for badge #2."