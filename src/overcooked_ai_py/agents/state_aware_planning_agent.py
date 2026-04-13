import itertools
import random
from collections import defaultdict, deque

from overcooked_ai_py.agents.agent import GreedyHumanModel
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager


class StateAwarePlanningAgent(GreedyHumanModel):
    """
    A planning agent that uses a simple, explicit priority waterfall.

    Decision order (first planner that produces reachable motion goals wins):
      1. _plan_held_object      – use / deposit whatever is in hand
      2. _plan_service          – soup is done → go get a dish now
      3. _plan_dish_for_pot     – pot is cooking and timer is close → preemptively grab dish
      4. _plan_start_cooking    – full pot sitting idle → press the cook button
      5. _plan_fill_pot         – pot needs ingredients → fetch them
      6. _plan_handoff_support  – wrong side of layout → relay items via shared counter

    The key invariant: "get dish when pot is full / cooking" is never gated behind
    ingredient-pressure scores or role-assignment math.  It is always tried before
    any prep work.
    """

    PREP_ROLE = "prep"
    SERVE_ROLE = "serve"
    HANDOFF_RESERVATION_TTL = 4

    # How many timesteps before a pot finishes to start fetching a dish.
    # Set conservatively high so the agent always has time to grab one.
    DISH_PREP_BUFFER = 8

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def __init__(
        self,
        mlam,
        hl_boltzmann_rational=False,
        ll_boltzmann_rational=False,
        hl_temp=1,
        ll_temp=1,
        auto_unstuck=True,
    ):
        state_aware_mlam = self._ensure_state_aware_mlam(mlam)
        super().__init__(
            state_aware_mlam,
            hl_boltzmann_rational=hl_boltzmann_rational,
            ll_boltzmann_rational=ll_boltzmann_rational,
            hl_temp=hl_temp,
            ll_temp=ll_temp,
            auto_unstuck=auto_unstuck,
        )

    def reset(self):
        super().reset()
        self.current_role = None
        self.current_task = None
        self.previous_task = None
        self.prev_observed_state = None
        self.counter_handoff_reservation = None
        self.handoff_role = "none"

    def action(self, state):
        self._update_handoff_memory(state)
        possible_motion_goals = self.ml_action(state)
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        chosen_goal, chosen_action, action_probs = self.choose_motion_goal(
            start_pos_and_or, possible_motion_goals
        )

        if (
            self.ll_boltzmann_rational
            and chosen_goal[0] == start_pos_and_or[0]
        ):
            chosen_action, action_probs = self.boltzmann_rational_ll_action(
                start_pos_and_or, chosen_goal
            )

        if self.auto_unstuck:
            if self._is_truly_stuck(state):
                chosen_action, action_probs = self._get_unstuck_action(state)
            self.prev_state = state

        self.previous_task = self.current_task
        self.prev_observed_state = state

        return chosen_action, {
            "action_probs": action_probs,
            "task": self.current_task,
            "role": self.current_role,
            "handoff_role": self.handoff_role,
        }

    # ── Top-level planning waterfall ─────────────────────────────────────────

    def ml_action(self, state):
        player = state.players[self.agent_index]
        analysis = self._analyze_state(state)

        held_obj = player.get_object().name if player.has_object() else None
        print(f"\n[DEBUG] ml_action: held_object={held_obj}, "
              f"cooking_pots={len(analysis['cooking_pots'])}, "
              f"ready_pots={len(analysis['ready_pots'])}, "
              f"fillable_pots={len(analysis['fillable_pots'])}")

        planners = [
            ("held_object", self._plan_held_object),
            ("service", self._plan_service),
            ("dish_for_pot", self._plan_dish_for_pot),
            ("start_cooking", self._plan_start_cooking),
            ("fill_pot", self._plan_fill_pot),
            ("handoff_support", self._plan_handoff_support),
        ]

        for planner_name, planner in planners:
            decision = planner(state, analysis)
            if decision is None:
                print(f"[DEBUG]    {planner_name}: returned None")
                continue
            
            goals = self._apply_decision(player, decision)
            if goals:
                print(f"[DEBUG]    {planner_name}: SUCCEEDED -> task={decision['task']}")
                return goals
            else:
                print(f"[DEBUG]    {planner_name}: decision but no reachable goals")

        self.current_task = "wait"
        self.current_role = "idle"
        print(f"[DEBUG]    all planners failed -> wait")
        return self._safe_wait_motion_goals(player)

    # ── State analysis ────────────────────────────────────────────────────────

    def _analyze_state(self, state):
        mdp = self.mlam.mdp
        player = state.players[self.agent_index]
        partner = state.players[1 - self.agent_index]

        counter_positions = list(mdp.terrain_pos_dict["X"])
        raw_counter_objects = self._normalize_counter_objects(
            mdp.get_counter_objects_dict(state, counter_positions)
        )
        pickup_counter_objects = self._filter_reserved_counter_objects(raw_counter_objects)

        pot_states = mdp.get_pot_states(state)

        ready_pots = list(pot_states["ready"])
        cooking_pots = list(pot_states["cooking"])

        # Use MDP classification exactly like the greedy agent
        target_counts = self._target_recipe_counts(state)
        target_total = target_counts["onion"] + target_counts["tomato"]
        soups_ready_to_cook_key = f"{target_total}_items"

        full_not_cooking = list(pot_states.get(soups_ready_to_cook_key, []))

        fillable_pots = list(pot_states["empty"])
        for k in ["1_items", "2_items"]:
            fillable_pots.extend(list(pot_states.get(k, [])))

        handoff_counters = self._shared_handoff_counters(state, player, partner)
        self.handoff_role = self._infer_handoff_role(player)
        shared_counter_objects = self._normalize_counter_objects(
            mdp.get_counter_objects_dict(state, handoff_counters)
        )
        pickup_shared_counter_objects = self._filter_reserved_counter_objects(
            shared_counter_objects
        )

        soonest_ready = None
        for pot_pos in cooking_pots:
            if not state.has_object(pot_pos):
                continue
            soup = state.get_object(pot_pos)
            remaining = getattr(soup, "cook_time_remaining", None)
            if remaining is not None:
                if soonest_ready is None or remaining < soonest_ready:
                    soonest_ready = remaining

        analysis = {
            "state": state,
            "mdp": mdp,
            "player": player,
            "partner": partner,
            "player_object": player.get_object().name if player.has_object() else None,
            "partner_object": partner.get_object().name if partner.has_object() else None,
            "pot_states": pot_states,
            "ready_pots": ready_pots,
            "cooking_pots": cooking_pots,
            "full_not_cooking": full_not_cooking,
            "fillable_pots": fillable_pots,
            "soonest_ready": soonest_ready,
            "counter_objects": raw_counter_objects,
            "pickup_counter_objects": pickup_counter_objects,
            "counter_soups": list(raw_counter_objects["soup"]),
            "empty_drop_counters": [
                pos
                for pos in self.mlam.params.get("counter_drop", [])
                if not state.has_object(pos)
            ],
            "layout_name": mdp.layout_name,
            "handoff_counters": handoff_counters,
            "handoff_role": self.handoff_role,
            "shared_counter_objects": shared_counter_objects,
            "pickup_shared_counter_objects": pickup_shared_counter_objects,
            "target_counts": target_counts,
            "soups_ready_to_cook_key": soups_ready_to_cook_key,
        }

        analysis["dish_urgent"] = self._is_dish_urgent(analysis)
        return analysis

    def _is_dish_urgent(self, analysis):
        """
        True when the agent should proactively fetch a dish.

        - Always true when soup is already done (ready pot or counter soup).
        - For players who can reach both dish dispenser AND pot: true when
          cook_time_remaining <= estimated round-trip travel + DISH_PREP_BUFFER.
        - For GIVERS (can reach dish dispenser but NOT pot): true when
          cook_time_remaining <= dish fetch time + DISH_PREP_BUFFER.
          We omit pot_dist because the giver never travels to the pot —
          they drop the dish on the counter for the receiver to collect.
        - If the agent can't even reach a dish, always False.
        """
        if analysis["ready_pots"] or analysis["counter_soups"]:
            return True
        if not analysis["cooking_pots"] or analysis["soonest_ready"] is None:
            return False

        player = analysis["player"]
        dish_locs = (
            analysis["mdp"].get_dish_dispenser_locations()
            + list(analysis["pickup_counter_objects"]["dish"])
        )
        dish_dist = self._distance_to_positions(player.pos_and_or, dish_locs)
        if dish_dist == float("inf"):
            return False  # Can't reach any dish at all.

        pot_dist = self._distance_to_positions(
            player.pos_and_or, analysis["cooking_pots"]
        )

        if pot_dist == float("inf"):
            # Giver role: can't reach pot directly.
            # Use only dish fetch time — the giver relays via counter.
            return analysis["soonest_ready"] <= dish_dist + self.DISH_PREP_BUFFER

        # Both reachable (open layout): use full round-trip estimate.
        travel_estimate = dish_dist + pot_dist
        return analysis["soonest_ready"] <= travel_estimate + self.DISH_PREP_BUFFER

    # ── Planner 1: handle held object ────────────────────────────────────────

    def _plan_held_object(self, state, analysis):
        """Decide what to do with whatever is currently in hand."""
        player = analysis["player"]
        if not player.has_object():
            return None

        held = player.get_object().name

        if held == "soup":
            # Always deliver immediately — nothing overrides this.
            return self._decision(
                "deliver_soup", self.SERVE_ROLE, self.mlam.deliver_soup_actions()
            )

        if held == "dish":
            return self._plan_dish_in_hand(state, analysis)

        if held in ["onion", "tomato"]:
            return self._plan_ingredient_in_hand(state, analysis, held)

        return None

    def _plan_dish_in_hand(self, state, analysis):
        """
        Holding a dish.

        For GIVERS in constrained layouts:
          - Pots active? → relay dish to receiver on handoff counter.
          - This is higher priority than anything else because the receiver
            cannot scoop soup without a dish, and giver cannot reach pots.

        For OPEN layouts or when not a giver:
          1. Try to scoop soup from a ready / nearly-ready pot.
          2. If no service work and pots need filling → drop dish and prep.
          3. Otherwise hold it.
        """
        # GIVER PRIORITY: if pots are active, relay dish to receiver.
        # (Giver can't reach pots, so trying to scoop directly is pointless.)
        if analysis["handoff_role"] == "giver":
            pot_active = bool(
                analysis["cooking_pots"]
                or analysis["ready_pots"]
                or analysis["full_not_cooking"]
            )
            if pot_active:
                print(f"[DEBUG]   _plan_dish_in_hand: GIVER with pot_active, relay to partner")
                relay_goals = self._empty_handoff_goals(analysis)
                if relay_goals:
                    player = analysis["player"]
                    reachable_relay = self._filter_valid_motion_goals(player, relay_goals)
                    if reachable_relay:
                        print(f"[DEBUG]     -> relay is reachable, RETURNING")
                        return self._decision(
                            "drop_dish_for_partner", self.SERVE_ROLE, relay_goals
                        )
                    else:
                        print(f"[DEBUG]     -> relay unreachable, will try other cases")

        # Try to pick up soup if a pot is ready (or cooking and urgent)
        soup_goals = self._pickup_soup_motion_goals(analysis)
        if soup_goals:
            print(f"[DEBUG]   _plan_dish_in_hand: soup_goals={len(soup_goals)}, checking reachability...")
            player = analysis["player"]
            reachable_soup = self._filter_valid_motion_goals(player, soup_goals)
            if reachable_soup:
                print(f"[DEBUG]     -> soup pickup reachable, RETURNING")
                return self._decision("pickup_soup", self.SERVE_ROLE, soup_goals)
            else:
                print(f"[DEBUG]     -> soup pickup unreachable (giver can't reach pots?), falling through")

        # No service work at all and pots need filling → drop dish and prep
        no_service = (
            not analysis["ready_pots"]
            and not analysis["cooking_pots"]
            and not analysis["counter_soups"]
        )
        if no_service and analysis["fillable_pots"]:
            print(f"[DEBUG]   _plan_dish_in_hand: no service, pots need filling, drop dish")
            drop_goals = self._drop_motion_goals(state, analysis)
            if drop_goals:
                player = analysis["player"]
                reachable_drop = self._filter_valid_motion_goals(player, drop_goals)
                if reachable_drop:
                    print(f"[DEBUG]     -> drop is reachable, RETURNING")
                    return self._decision("drop_dish_for_prep", self.PREP_ROLE, drop_goals)
                else:
                    print(f"[DEBUG]     -> drop unreachable")

        print(f"[DEBUG]   _plan_dish_in_hand: no case matched, returning None")
        return None
    
    def _plan_ingredient_in_hand(self, state, analysis, ingredient):
        """
        Greedy-style ingredient handling:
        1. If pot is already full/cooking/ready and no fillable pots remain, drop ingredient.
        2. Otherwise try to put ingredient directly into a valid pot using MLAM's pot-state logic.
        3. If unreachable, relay via handoff.
        4. Last resort: drop anywhere.
        """
        if not analysis["fillable_pots"]:
            drop_goals = self._drop_motion_goals(state, analysis)
            if drop_goals:
                return self._decision(
                    f"drop_{ingredient}_to_clear", self.PREP_ROLE, drop_goals
                )
            return None

        put_goals = self._ingredient_put_motion_goals(analysis, ingredient)
        if put_goals:
            player = analysis["player"]
            reachable_put_goals = self._filter_valid_motion_goals(player, put_goals)
            if reachable_put_goals:
                return self._decision(
                    f"put_{ingredient}_in_pot", self.PREP_ROLE, put_goals
                )

        relay_goals = self._empty_handoff_goals(analysis)
        if relay_goals:
            player = analysis["player"]
            reachable_relay = self._filter_valid_motion_goals(player, relay_goals)
            if reachable_relay:
                return self._decision(
                    f"drop_{ingredient}_for_partner", self.PREP_ROLE, relay_goals
                )

        drop_goals = self._drop_motion_goals(state, analysis)
        if drop_goals:
            return self._decision(
                f"drop_{ingredient}_to_clear", self.PREP_ROLE, drop_goals
            )

        return None

    # ── Planner 2: serve ready soup ───────────────────────────────────────────

    def _plan_service(self, state, analysis):
        """
        Soup is already done (on a counter or in a ready pot).
        Hands must be empty to reach here.

        - Counter soup: try to pick it up directly (mlam handles needing a dish).
        - Ready pot: fetch a dish so we can scoop soup.
        """
        # Soup sitting on a counter is the most time-sensitive — grab it first.
        if analysis["counter_soups"]:
            soup_goals = self.mlam.pickup_counter_soup_actions(
                analysis["pickup_counter_objects"]
            )
            if soup_goals:
                return self._decision(
                    "pickup_counter_soup", self.SERVE_ROLE, soup_goals
                )

        if not (analysis["counter_soups"] or analysis["ready_pots"]):
            return None

        # Get a dish to scoop from the ready pot.
        dish_goals = self.mlam.pickup_dish_actions(analysis["pickup_counter_objects"])
        if dish_goals:
            return self._decision("get_dish_ready_pot", self.SERVE_ROLE, dish_goals)

        # Check shared counter in case partner already placed one there.
        shared_dishes = list(analysis["pickup_shared_counter_objects"]["dish"])
        if shared_dishes:
            return self._decision(
                "get_shared_dish_ready_pot",
                self.SERVE_ROLE,
                self._motion_goals_for_positions(shared_dishes),
            )

        return None

    # ── Planner 3: preemptive dish fetch while pot cooks ─────────────────────

    def _plan_dish_for_pot(self, state, analysis):
        """
        A pot is cooking and will be ready within DISH_PREP_BUFFER + travel time.
        Fetch a dish proactively so we do not stand idle after soup is done.

        Skip if:
        - dish_urgent is False (too early — focus on prep work instead)
        - partner is already heading for / holding a dish or soup
        """
        if not analysis["dish_urgent"] or not analysis["cooking_pots"]:
            return None

        # Don't send both players after a dish at the same time.
        if analysis["partner_object"] in ("dish", "soup"):
            return None

        dish_goals = self.mlam.pickup_dish_actions(analysis["pickup_counter_objects"])
        if dish_goals:
            return self._decision("get_dish_preemptive", self.SERVE_ROLE, dish_goals)

        shared_dishes = list(analysis["pickup_shared_counter_objects"]["dish"])
        if shared_dishes:
            return self._decision(
                "get_shared_dish_preemptive",
                self.SERVE_ROLE,
                self._motion_goals_for_positions(shared_dishes),
            )

        return None

    # ── Planner 4: start cooking ──────────────────────────────────────────────

    def _plan_start_cooking(self, state, analysis):
        truly_full = [
            pot_pos
            for pot_pos in analysis["full_not_cooking"]
            if self._pot_is_exactly_ready_to_cook(analysis, pot_pos)
        ]

        if not truly_full:
            return None

        full_pots = defaultdict(list)
        full_pots[analysis["soups_ready_to_cook_key"]] = truly_full

        start_goals = self.mlam.start_cooking_actions(full_pots)
        if start_goals:
            return self._decision("start_cooking", self.PREP_ROLE, start_goals)

        return None

    # ── Planner 5: fill pot ────────────────────────────────────────────────────

    def _plan_fill_pot(self, state, analysis):
        """
        Fill pots using MDP pot-state classification.
        If a pot is already in '<target_total>_items', do not keep filling — let
        _plan_start_cooking handle it first.
        """
        if analysis["full_not_cooking"]:
            return None

        if not analysis["fillable_pots"]:
            return None

        ingredient = self._select_prepare_ingredient(analysis)
        if ingredient is None:
            return None

        ingredient_goals = self._pickup_ingredient_motion_goals(
            ingredient, analysis["pickup_counter_objects"]
        )
        if ingredient_goals:
            return self._decision(
                f"get_{ingredient}", self.PREP_ROLE, ingredient_goals
            )

        shared = list(analysis["pickup_shared_counter_objects"].get(ingredient, []))
        if shared:
            return self._decision(
                f"get_shared_{ingredient}",
                self.PREP_ROLE,
                self._motion_goals_for_positions(shared),
            )

        return None

    # ── Planner 6: handoff support ────────────────────────────────────────────

    def _plan_handoff_support(self, state, analysis):
        """
        Used in layouts where the two players occupy separated zones.

        Giver  (supply side — can reach dispensers but not pots/serving):
          Assess what the partner needs next and pre-stage it on the shared counter.
          - If a pot is cooking or ready → partner needs a dish → fetch a dish.
          - If pot needs filling          → partner needs an ingredient → fetch it.

        Receiver (cooking side — can reach pots/serving but not dispensers):
          Pick up items the partner (giver) left on the shared counter.
        """
        if not analysis["handoff_counters"]:
            return None

        if analysis["handoff_role"] == "receiver":
            return self._plan_receiver_pickup(state, analysis)

        if analysis["handoff_role"] == "giver":
            return self._plan_giver_prefetch(state, analysis)

        return None

    def _plan_receiver_pickup(self, state, analysis):
        """
        Receiver: collect items giver has left on the shared counter.
        Priority: soup > dish > ingredient.
        """
        shared = analysis["pickup_shared_counter_objects"]

        # Soup on counter (giver somehow got it there — rare but handle it)
        if analysis["counter_soups"]:
            soup_goals = self.mlam.pickup_counter_soup_actions(
                analysis["pickup_counter_objects"]
            )
            if soup_goals:
                return self._decision(
                    "receiver_pickup_soup", self.SERVE_ROLE, soup_goals
                )

        # Dish waiting — use it to serve a ready or cooking pot
        if shared["dish"] and (analysis["ready_pots"] or analysis["cooking_pots"]):
            return self._decision(
                "receiver_pickup_dish",
                self.SERVE_ROLE,
                self._motion_goals_for_positions(list(shared["dish"])),
            )

        # Ingredient waiting — put it in a fillable pot
        if analysis["fillable_pots"]:
            ingredient = self._select_prepare_ingredient(analysis)
            if ingredient and shared[ingredient]:
                return self._decision(
                    "receiver_pickup_{}".format(ingredient),
                    self.PREP_ROLE,
                    self._motion_goals_for_positions(list(shared[ingredient])),
                )

        return None

    def _plan_giver_prefetch(self, state, analysis):
        """
        Giver: figure out what the receiver will need next and stage it.

        The decision is based on the receiver's most urgent upcoming action:

          pot is cooking or ready  →  receiver needs a DISH  →  fetch a dish
          pot is fillable          →  receiver needs an INGREDIENT  →  fetch it

        When both are true (e.g. one pot cooking, another fillable), cooking
        takes priority so the receiver can serve soup without waiting.

        Also: only stage one item at a time.  If the counter already has the
        right item, hold off — don't clutter the counter.
        """
        empty_handoff = [
            pos
            for pos in analysis["handoff_counters"]
            if pos in set(analysis["empty_drop_counters"])
        ]
        if not empty_handoff:
            return None  # Counter full — wait for partner to clear it.

        shared = analysis["pickup_shared_counter_objects"]
        pot_active = bool(analysis["cooking_pots"] or analysis["ready_pots"])

        # ── Priority A: receiver needs a dish ────────────────────────────────
        # Triggered when any pot is cooking or ready, regardless of timer.
        # The giver's job is to pre-stage the dish; receiver decides when to
        # collect it based on the cooking timer.
        if pot_active and not shared["dish"]:
            dish_goals = self.mlam.pickup_dish_actions(
                analysis["pickup_counter_objects"]
            )
            if dish_goals:
                return self._decision(
                    "giver_fetch_dish_for_partner", self.SERVE_ROLE, dish_goals
                )

        # Dish already staged — giver should wait, not fetch another item.
        if pot_active:
            return None

        # ── Priority B: receiver needs an ingredient ──────────────────────────
        # Only reached when NO pot is currently cooking or ready.
        if analysis["fillable_pots"]:
            ingredient = self._select_prepare_ingredient(analysis)
            if ingredient and not shared[ingredient]:
                ingredient_goals = self._pickup_ingredient_motion_goals(
                    ingredient, analysis["pickup_counter_objects"]
                )
                if ingredient_goals:
                    return self._decision(
                        "giver_fetch_{}_for_partner".format(ingredient),
                        self.PREP_ROLE,
                        ingredient_goals,
                    )

        return None

    # ── Counter/handoff helpers ───────────────────────────────────────────────

    def _empty_handoff_goals(self, analysis):
        """
        Motion goals for slots on the shared handoff counter.
        
        Priority 1: Empty slots (preferred)
        Priority 2: If no empty slots AND holding an item that needs placing,
                    fallback to ANY handoff counter slot (items may stack/push).
        """
        empty_set = set(analysis["empty_drop_counters"])
        targets = [
            pos for pos in analysis["handoff_counters"] if pos in empty_set
        ]
        print(f"[DEBUG]       _empty_handoff_goals: handoff_counters={len(analysis['handoff_counters'])}, "
              f"empty_drops={len(empty_set)}, empty_targets={len(targets)}")
        
        # If empty slots available, use them
        if targets:
            result = self._motion_goals_for_positions(targets)
            print(f"[DEBUG]         -> using empty slots: {len(result)} goals")
            return result
        
        # Fallback: if ALL empty slots are full, try ANY handoff counter
        # This allows stacking/pushing items (e.g., giver drops onion on counter
        # that already has a dish; receiver will pick up one or both)
        if analysis["handoff_counters"]:
            result = self._motion_goals_for_positions(analysis["handoff_counters"])
            print(f"[DEBUG]         -> fallback to ANY handoff slot (full): {len(result)} goals")
            return result
        
        print(f"[DEBUG]         -> no handoff counters available")
        return []

    # ── Soup pickup ───────────────────────────────────────────────────────────

    def _pickup_soup_motion_goals(self, analysis):
        """
        Motion goals for scooping soup with a dish already in hand.
        Always includes ready pots; includes cooking pots when dish is urgent
        (pot is nearly done — this avoids a wasted round trip).
        """
        allowed_pot_states = defaultdict(list)
        allowed_pot_states["ready"] = list(analysis["ready_pots"])
        if analysis["dish_urgent"] and analysis["cooking_pots"]:
            allowed_pot_states["cooking"] = list(analysis["cooking_pots"])
        return self.mlam.pickup_soup_with_dish_actions(
            allowed_pot_states, only_nearly_ready=True
        )

    # ── Ingredient selection ──────────────────────────────────────────────────

    def _select_prepare_ingredient(self, analysis):
        target_counts = analysis["target_counts"]

        if target_counts.get("onion", 0) == 3 and target_counts.get("tomato", 0) == 0:
            return "onion"

        best_pot = self._best_fillable_pot(analysis)
        if best_pot is None:
            return None

        current_counts = self._pot_ingredient_counts(analysis["state"], best_pot)
        missing = {
            name: max(
                0,
                target_counts.get(name, 0) - current_counts.get(name, 0),
            )
            for name in ["onion", "tomato"]
        }

        # partner_object = analysis["partner_object"]
        # if partner_object in missing and missing[partner_object] > 0:
        #     missing[partner_object] -= 1

        if missing["tomato"] > missing["onion"]:
            return "tomato"
        if missing["onion"] > 0:
            return "onion"
        if missing["tomato"] > 0:
            return "tomato"
        return None

    def _pot_is_exactly_ready_to_cook(self, analysis, pot_pos):
        counts = self._pot_ingredient_counts(analysis["state"], pot_pos)
        return (
            counts["onion"] == analysis["target_counts"].get("onion", 0)
            and counts["tomato"] == analysis["target_counts"].get("tomato", 0)
        )
        
    def _best_fillable_pot(self, analysis):
        """
        Score fillable pots and return the position of the best one.
        Prefer pots that already have some ingredients (closer to completion),
        penalise over-filled pots and distance.
        """
        best_pot = None
        best_score = float("-inf")
        for pot_pos in analysis["fillable_pots"]:
            current_counts = self._pot_ingredient_counts(
                analysis["state"], pot_pos
            )
            fill_count = current_counts["onion"] + current_counts["tomato"]
            overflow = max(
                0,
                current_counts["onion"]
                - analysis["target_counts"].get("onion", 0),
            ) + max(
                0,
                current_counts["tomato"]
                - analysis["target_counts"].get("tomato", 0),
            )
            score = 4.0 * fill_count - 6.0 * overflow
            distance = self._distance_to_positions(
                analysis["player"].pos_and_or, [pot_pos]
            )
            if distance < float("inf"):
                score -= 0.2 * distance
            if score > best_score:
                best_score = score
                best_pot = pot_pos
        return best_pot

    def _total_missing_ingredients(self, analysis):
        total = 0
        for pot_pos in analysis["fillable_pots"]:
            current_counts = self._pot_ingredient_counts(
                analysis["state"], pot_pos
            )
            total += max(
                0,
                analysis["target_counts"].get("onion", 0) - current_counts["onion"],
            )
            total += max(
                0,
                analysis["target_counts"].get("tomato", 0)
                - current_counts["tomato"],
            )
        return total

    def _pot_positions_missing_ingredient(self, analysis, ingredient_name):
        target_amount = analysis["target_counts"].get(ingredient_name, 0)
        if target_amount <= 0:
            return []
        return [
            pot_pos
            for pot_pos in analysis["fillable_pots"]
            if self._pot_ingredient_counts(analysis["state"], pot_pos)[ingredient_name]
            < target_amount
        ]

    def _ingredient_put_motion_goals(self, analysis, ingredient_name):
        """
        Use MLAM/MDP pot classification directly, like the greedy agent.
        """
        if ingredient_name == "tomato":
            return self.mlam.put_tomato_in_pot_actions(analysis["pot_states"])
        return self.mlam.put_onion_in_pot_actions(analysis["pot_states"])

    def _pickup_ingredient_motion_goals(self, ingredient_name, counter_objects):
        if ingredient_name == "tomato":
            return self.mlam.pickup_tomato_actions(counter_objects)
        return self.mlam.pickup_onion_actions(counter_objects)

    # ── Recipe / pot state helpers ────────────────────────────────────────────

    def _target_recipe_counts(self, state):
        if not state.all_orders:
            return {"onion": 3, "tomato": 0}
        target_recipe = list(state.all_orders)[0]
        ingredients = list(target_recipe.ingredients)
        return {
            "onion": ingredients.count("onion"),
            "tomato": ingredients.count("tomato"),
        }

    def _pot_ingredient_counts(self, state, pot_pos):
        onion_count = 0
        tomato_count = 0
        if state.has_object(pot_pos):
            soup = state.get_object(pot_pos)
            if not soup.is_cooking and not soup.is_ready:
                onion_count = soup.ingredients.count("onion")
                tomato_count = soup.ingredients.count("tomato")
        return {"onion": onion_count, "tomato": tomato_count}

    # ── Layout / counter classification ──────────────────────────────────────

    @classmethod
    def _ensure_state_aware_mlam(cls, mlam):
        state_aware_params = cls._get_state_aware_mlam_params(mlam)
        if state_aware_params == mlam.params:
            return mlam
        custom_filename = "{}_state_aware_am.pkl".format(mlam.mdp.layout_name)
        return MediumLevelActionManager.from_pickle_or_compute(
            mlam.mdp,
            state_aware_params,
            custom_filename=custom_filename,
            force_compute=False,
        )

    @classmethod
    def _get_state_aware_mlam_params(cls, mlam):
        params = dict(mlam.params)
        reachable_counters, shared_drop_counters = cls._get_counter_targets(mlam.mdp)
        params["counter_goals"] = cls._merge_counter_lists(
            params.get("counter_goals", []), reachable_counters
        )
        params["counter_pickup"] = cls._merge_counter_lists(
            params.get("counter_pickup", []), reachable_counters
        )
        params["counter_drop"] = cls._merge_counter_lists(
            params.get("counter_drop", []), shared_drop_counters
        )
        return params

    @staticmethod
    def _merge_counter_lists(existing_positions, new_positions):
        merged = [tuple(pos) for pos in existing_positions]
        for pos in new_positions:
            pos = tuple(pos)
            if pos not in merged:
                merged.append(pos)
        return merged

    @classmethod
    def _get_counter_targets(cls, mdp):
        valid_positions = set(mdp.get_valid_player_positions())
        component_by_pos = cls._get_connected_components(valid_positions)
        start_components = {
            component_by_pos[start_pos]
            for start_pos in mdp.start_player_positions
            if start_pos in component_by_pos
        }

        reachable_counters = []
        counter_infos = []
        shared_drop_counters = []

        for counter_pos in mdp.terrain_pos_dict["X"]:
            adjacent_positions = []
            adjacent_components = set()
            for direction in Action.MOTION_ACTIONS:
                adjacent_pos = Action.move_in_direction(counter_pos, direction)
                if adjacent_pos in component_by_pos:
                    adjacent_positions.append(adjacent_pos)
                    adjacent_components.add(component_by_pos[adjacent_pos])

            if not adjacent_components:
                continue

            reachable_counters.append(counter_pos)
            counter_infos.append(
                (counter_pos, tuple(adjacent_positions), adjacent_components)
            )
            if start_components.issubset(adjacent_components):
                shared_drop_counters.append(counter_pos)

        if len(start_components) > 1 and shared_drop_counters:
            return shared_drop_counters, shared_drop_counters

        staging_counters = cls._select_staging_counters(mdp, counter_infos)
        if staging_counters:
            return staging_counters, staging_counters

        return reachable_counters[:4], reachable_counters[:4]

    @classmethod
    def _select_staging_counters(cls, mdp, counter_infos, max_counters=4):
        if not counter_infos:
            return []

        pot_positions = list(mdp.get_pot_locations())
        serving_positions = list(mdp.get_serving_locations())
        dish_positions = list(mdp.get_dish_dispenser_locations())
        ingredient_positions = list(
            mdp.get_onion_dispenser_locations()
        ) + list(mdp.get_tomato_dispenser_locations())

        def min_distance(counter_pos, feature_positions):
            if not feature_positions:
                return float("inf")
            return min(
                abs(counter_pos[0] - pos[0]) + abs(counter_pos[1] - pos[1])
                for pos in feature_positions
            )

        ranked = sorted(
            counter_infos,
            key=lambda item: (
                min_distance(item[0], pot_positions) > 1,
                min_distance(item[0], serving_positions) > 1,
                min_distance(item[0], dish_positions) > 1,
                min_distance(item[0], ingredient_positions) > 1,
                min(
                    min_distance(item[0], pot_positions),
                    min_distance(item[0], serving_positions),
                    min_distance(item[0], dish_positions),
                    min_distance(item[0], ingredient_positions),
                ),
                -len(item[1]),
                item[0][1],
                item[0][0],
            ),
        )
        return [counter_pos for counter_pos, _, _ in ranked[:max_counters]]

    @staticmethod
    def _get_connected_components(valid_positions):
        component_by_pos = {}
        next_component_id = 0
        for start_pos in valid_positions:
            if start_pos in component_by_pos:
                continue
            frontier = deque([start_pos])
            component_by_pos[start_pos] = next_component_id
            while frontier:
                pos = frontier.popleft()
                for direction in Action.MOTION_ACTIONS:
                    neighbor = Action.move_in_direction(pos, direction)
                    if (
                        neighbor in valid_positions
                        and neighbor not in component_by_pos
                    ):
                        component_by_pos[neighbor] = next_component_id
                        frontier.append(neighbor)
            next_component_id += 1
        return component_by_pos

    # ── Handoff role inference ────────────────────────────────────────────────

    def _shared_handoff_counters(self, state, player, partner):
        counters = list(self.mlam.params.get("counter_drop", []))
        if not counters:
            counters = self.mlam.mdp.get_counter_locations()

        ranked = []
        for counter_pos in counters:
            player_distance = self._distance_to_positions(
                player.pos_and_or, [counter_pos]
            )
            partner_distance = self._distance_to_positions(
                partner.pos_and_or, [counter_pos]
            )
            if player_distance == float("inf") or partner_distance == float("inf"):
                continue
            score = player_distance + partner_distance
            if state.has_object(counter_pos):
                score -= 0.25
            ranked.append((score, counter_pos))

        ranked.sort(key=lambda item: item[0])
        return [counter_pos for _, counter_pos in ranked]

    def _infer_handoff_role(self, player):
        """
        Giver   = can reach supply features (dispensers) but NOT cooking features.
        Receiver = can reach cooking features (pot, serving) but NOT supply features.
        None     = can reach both (open layouts — no relay needed).
        """
        mdp = self.mlam.mdp
        can_onion = self._can_interact_with_any(
            player, mdp.get_onion_dispenser_locations()
        )
        can_tomato = self._can_interact_with_any(
            player, mdp.get_tomato_dispenser_locations()
        )
        can_dish = self._can_interact_with_any(
            player, mdp.get_dish_dispenser_locations()
        )
        can_pot = self._can_interact_with_any(player, mdp.get_pot_locations())
        can_serve = self._can_interact_with_any(
            player, mdp.get_serving_locations()
        )

        supply_score = int(can_onion) + int(can_tomato) + int(can_dish)
        cook_score = int(can_pot) + int(can_serve)

        if supply_score > 0 and cook_score == 0:
            return "giver"
        if cook_score > 0 and supply_score == 0:
            return "receiver"
        return "none"

    def _can_interact_with_any(self, player, feature_positions):
        if not feature_positions:
            return False
        return (
            self._distance_to_positions(player.pos_and_or, feature_positions)
            < float("inf")
        )

    # ── Motion goal utilities ─────────────────────────────────────────────────

    def _distance_to_positions(self, start_pos_and_or, positions):
        if not positions:
            return float("inf")
        return self.mlam.motion_planner.min_cost_to_feature(
            start_pos_and_or, positions
        )

    def _motion_goals_for_positions(self, positions):
        if not positions:
            return []
        return self.mlam._get_ml_actions_for_positions(list(positions))

    def _drop_motion_goals(self, state, analysis, preferred_positions=None):
        valid_empty = set(analysis["empty_drop_counters"])
        print(f"[DEBUG]       _drop_motion_goals: empty_counters={len(valid_empty)}, "
              f"preferred={preferred_positions}")
        
        if not valid_empty:
            print(f"[DEBUG]         -> no empty counters at all")
            return []
        
        if preferred_positions is not None:
            targets = [pos for pos in preferred_positions if pos in valid_empty]
            print(f"[DEBUG]         -> checking preferred: {len(targets)} match")
            if targets:
                result = self._motion_goals_for_positions(targets)
                print(f"[DEBUG]         -> motion_goals: {len(result)}")
                return result
        
        result = self.mlam.place_obj_on_counter_actions(state)
        print(f"[DEBUG]         -> place_obj_on_counter: {len(result)} goals")
        return result

    def _start_cooking_motion_goals(self, full_not_cooking):
        full_pots = defaultdict(list)
        full_pots["3_items"] = list(full_not_cooking)
        return self.mlam.start_cooking_actions(full_pots)

    def _decision(self, task, role, motion_goals):
        return {"task": task, "role": role, "motion_goals": motion_goals}

    def _apply_decision(self, player, decision):
        if decision is None:
            return None
        
        print(f"[DEBUG]   _apply_decision: task={decision['task']}, "
              f"motion_goals={len(decision['motion_goals'])}")
        
        valid_goals = self._filter_valid_motion_goals(
            player, decision["motion_goals"]
        )
        print(f"[DEBUG]     -> after reachability filter: {len(valid_goals)} goals")
        
        if not valid_goals:
            return None
        self.current_task = decision["task"]
        self.current_role = decision["role"]
        return valid_goals

    def _filter_valid_motion_goals(self, player, motion_goals):
        return [
            goal
            for goal in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, goal
            )
        ]

    def _safe_wait_motion_goals(self, player):
        wait_goals = self._filter_valid_motion_goals(
            player, self.mlam.wait_actions(player)
        )
        if wait_goals:
            return wait_goals
        return self._filter_valid_motion_goals(
            player, self.mlam.go_to_closest_feature_actions(player)
        )

    # ── Counter object helpers ────────────────────────────────────────────────

    def _normalize_counter_objects(self, counter_objects):
        normalized = defaultdict(list)
        for object_name in ["onion", "tomato", "dish", "soup"]:
            normalized[object_name] = list(counter_objects.get(object_name, []))
        return normalized

    def _filter_reserved_counter_objects(self, counter_objects):
        filtered = self._normalize_counter_objects(counter_objects)
        reservation = self.counter_handoff_reservation
        if reservation is None:
            return filtered
        object_name = reservation["object_name"]
        filtered[object_name] = [
            pos
            for pos in filtered[object_name]
            if pos != reservation["position"]
        ]
        return filtered

    # ── Handoff memory (tracks items we just dropped on counters) ─────────────

    def _update_handoff_memory(self, state):
        if self.counter_handoff_reservation is not None:
            reservation = dict(self.counter_handoff_reservation)
            reservation["ttl"] -= 1
            if (
                reservation["ttl"] <= 0
                or not state.has_object(reservation["position"])
                or state.get_object(reservation["position"]).name
                != reservation["object_name"]
            ):
                self.counter_handoff_reservation = None
            else:
                self.counter_handoff_reservation = reservation

        if (
            self.prev_observed_state is None
            or self.previous_task is None
            or not self.previous_task.startswith("drop_")
        ):
            return

        previous_player = self.prev_observed_state.players[self.agent_index]
        current_player = state.players[self.agent_index]
        if not previous_player.has_object() or current_player.has_object():
            return

        dropped_name = previous_player.get_object().name
        new_positions = []
        for counter_pos in self.mlam.params.get("counter_drop", []):
            if self.prev_observed_state.has_object(counter_pos):
                continue
            if not state.has_object(counter_pos):
                continue
            if state.get_object(counter_pos).name != dropped_name:
                continue
            new_positions.append(counter_pos)

        if not new_positions:
            return

        chosen_position = min(
            new_positions,
            key=lambda pos: abs(pos[0] - current_player.position[0])
            + abs(pos[1] - current_player.position[1]),
        )
        self.counter_handoff_reservation = {
            "position": chosen_position,
            "object_name": dropped_name,
            "ttl": self.HANDOFF_RESERVATION_TTL,
        }

    # ── Unstuck ───────────────────────────────────────────────────────────────

    @staticmethod
    def _layout_group(layout_name):
        lname = layout_name.lower()
        for token in [
            "cramped",
            "corridor",
            "forced",
            "counter_circuit",
            "you_shall_not_pass",
            "bottleneck",
        ]:
            if token in lname:
                return "constrained"
        return "open"

    def _get_unstuck_action(self, state):
        if self.agent_index == 0:
            joint_actions = list(
                itertools.product(Action.ALL_ACTIONS, [Action.STAY])
            )
        else:
            joint_actions = list(
                itertools.product([Action.STAY], Action.ALL_ACTIONS)
            )

        unblocking = [
            ja
            for ja in joint_actions
            if self.mlam.mdp.get_state_transition(state, ja)[0].player_positions
            != self.prev_state.player_positions
        ]

        if not unblocking:
            return Action.STAY, self.a_probs_from_action(Action.STAY)

        chosen = random.choice(unblocking)[self.agent_index]
        return chosen, self.a_probs_from_action(chosen)

    def _is_truly_stuck(self, state):
        if self.prev_state is None:
            return False
        return state.time_independent_equal(self.prev_state)