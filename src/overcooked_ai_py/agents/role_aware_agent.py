import itertools
import random
from collections import defaultdict, deque

from overcooked_ai_py.agents.agent import GreedyHumanModel
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager


class RoleAwareGreedyAgent(GreedyHumanModel):
    """
    A planner-backed agent that keeps GreedyHumanModel's low-level motion planning
    but chooses between two explicit high-level roles:

    - PREP: fetch ingredients, fill pots, start cooking
    - SERVE: fetch dishes, pick up soup, deliver soup

    The role choice is partner-aware but intentionally lightweight: it relies on
    held objects, kitchen urgency, and planner distances rather than a separate
    learned intent model.
    """

    PREP_ROLE = "prep"
    SERVE_ROLE = "serve"

    def __init__(
        self,
        mlam,
        hl_boltzmann_rational=False,
        ll_boltzmann_rational=False,
        hl_temp=1,
        ll_temp=1,
        auto_unstuck=True,
    ):
        role_aware_mlam = self._ensure_role_aware_mlam(mlam)
        super().__init__(
            role_aware_mlam,
            hl_boltzmann_rational=hl_boltzmann_rational,
            ll_boltzmann_rational=ll_boltzmann_rational,
            hl_temp=hl_temp,
            ll_temp=ll_temp,
            auto_unstuck=auto_unstuck,
        )

    def reset(self):
        super().reset()
        self.current_role = None
        self.partner_role = None
        self.current_task = None
        self.previous_task = None
        self.select_role_cache = None
        self.pot_states_cache = None

    def action(self, state):
        self.partner_role = self.infer_partner_role(state)
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

        return chosen_action, {
            "action_probs": action_probs,
            "role": self.current_role,
            "partner_role": self.partner_role,
            "task": self.current_task,
        }

    @classmethod
    def _ensure_role_aware_mlam(cls, mlam):
        role_aware_params = cls._get_role_aware_mlam_params(mlam)
        if role_aware_params == mlam.params:
            return mlam

        custom_filename = "{}_role_aware_am.pkl".format(mlam.mdp.layout_name)
        return MediumLevelActionManager.from_pickle_or_compute(
            mlam.mdp,
            role_aware_params,
            custom_filename=custom_filename,
            force_compute=False,
        )

    @classmethod
    def _get_role_aware_mlam_params(cls, mlam):
        params = dict(mlam.params)
        reachable_counters, shared_drop_counters = cls._get_counter_targets(
            mlam.mdp
        )
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
        ingredient_positions = list(mdp.get_onion_dispenser_locations()) + list(
            mdp.get_tomato_dispenser_locations()
        )

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

    def ml_action(self, state):
        player = state.players[self.agent_index]
        other = state.players[1 - self.agent_index]
        counter_objects = self.mlam.mdp.get_counter_objects_dict(
            state, list(self.mlam.mdp.terrain_pos_dict["X"])
        )
        pot_states_dict = self.mlam.mdp.get_pot_states(state)

        if player.has_object():
            task_name, task_role, motion_goals = self._committed_held_object_decision(
                state, player, pot_states_dict
            )
            motion_goals = self._filter_valid_motion_goals(
                player, motion_goals
            )
            if motion_goals:
                self.current_task = task_name
                self.current_role = task_role
                return motion_goals

            # When we are already carrying something, do not fall through into
            # unheld-object candidate generation such as dish pickup. Keep the
            # current commitment until we can pot or drop the object.
            fallback_goals = self._safe_wait_motion_goals(player)
            if fallback_goals:
                self.current_task = task_name
                self.current_role = task_role
                return fallback_goals

        preferred_role = self.select_role(state)
        self.select_role_cache = preferred_role
        self.pot_states_cache = pot_states_dict
        candidates = self._build_candidates(
            state,
            player,
            other,
            counter_objects,
            pot_states_dict,
            preferred_role,
        )
        best = self._choose_best_candidate(player.pos_and_or, candidates)
        if best is not None:
            self.current_task = best["task"]
            self.current_role = best["task_role"] or preferred_role
            return best["motion_goals"]

        self.current_task = "fallback"
        self.current_role = preferred_role
        motion_goals = self.mlam.go_to_closest_feature_actions(player)
        motion_goals = self._filter_valid_motion_goals(player, motion_goals)
        return motion_goals

    def _filter_valid_motion_goals(self, player, motion_goals):
        return [
            mg
            for mg in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
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

    def _committed_held_object_decision(self, state, player, pot_states_dict):
        obj_name = player.get_object().name

        if obj_name == "soup":
            return (
                "deliver_soup",
                self.SERVE_ROLE,
                self._held_object_motion_goals(player, pot_states_dict),
            )

        if obj_name == "dish":
            return (
                "pickup_soup",
                self.SERVE_ROLE,
                self._held_object_motion_goals(player, pot_states_dict),
            )

        if obj_name in ["onion", "tomato"]:
            ingredient_goals = self._held_object_motion_goals(
                player, pot_states_dict
            )
            if ingredient_goals:
                return ("put_in_pot", self.PREP_ROLE, ingredient_goals)

            if self._service_is_urgent(pot_states_dict):
                drop_goals = self._drop_item_motion_goals(state)
                if drop_goals:
                    return (
                        "drop_{}".format(obj_name),
                        self.SERVE_ROLE,
                        drop_goals,
                    )

            return ("put_in_pot", self.PREP_ROLE, ingredient_goals)

        raise ValueError("Unrecognized held object {}".format(obj_name))

    def infer_partner_role(self, state):
        other = state.players[1 - self.agent_index]
        pot_states_dict = self.mlam.mdp.get_pot_states(state)

        if other.has_object():
            obj_name = other.get_object().name
            if obj_name in ["dish", "soup"]:
                return self.SERVE_ROLE
            if obj_name in ["onion", "tomato"]:
                return self.PREP_ROLE

        ready_or_cooking = list(pot_states_dict["ready"]) + list(
            pot_states_dict["cooking"]
        )
        if ready_or_cooking:
            serve_distance = self._distance_to_service_chain(other.pos_and_or)
            prep_distance = self._distance_to_prep_chain(other.pos_and_or)
            if serve_distance <= prep_distance + 1:
                return self.SERVE_ROLE

        return self.PREP_ROLE

    def select_role(self, state):
        player = state.players[self.agent_index]
        other = state.players[1 - self.agent_index]
        pot_states_dict = self.mlam.mdp.get_pot_states(state)

        if player.has_object():
            obj_name = player.get_object().name
            if obj_name in ["dish", "soup"]:
                return self.SERVE_ROLE
            if obj_name in ["onion", "tomato"]:
                if self._service_is_urgent(pot_states_dict) and not self._fillable_pots_exist(
                    pot_states_dict
                ):
                    return self.SERVE_ROLE
                return self.PREP_ROLE

        ready_or_cooking = list(pot_states_dict["ready"]) + list(
            pot_states_dict["cooking"]
        )
        cookable = self._start_cooking_motion_goals(pot_states_dict)
        partner_role = self.partner_role or self.infer_partner_role(state)

        if ready_or_cooking:
            if partner_role == self.SERVE_ROLE:
                return self.PREP_ROLE
            if self._distance_to_service_chain(player.pos_and_or) <= self._distance_to_service_chain(
                other.pos_and_or
            ):
                return self.SERVE_ROLE
            return self.PREP_ROLE

        if cookable:
            return self.PREP_ROLE

        if partner_role == self.PREP_ROLE and self._dish_sources_available():
            return self.SERVE_ROLE

        return self.PREP_ROLE

    def _serving_motion_goals(self, state, counter_objects, pot_states_dict):
        am = self.mlam
        ready = list(pot_states_dict["ready"])
        cooking = list(pot_states_dict["cooking"])
        counter_soups = counter_objects["soup"]

        if counter_soups:
            return am.pickup_counter_soup_actions(counter_objects)

        if ready:
            return am.pickup_dish_actions(counter_objects)

        if cooking:
            if self._dish_sources_available():
                return am.pickup_dish_actions(counter_objects)

        return []

    def _prep_motion_goals(self, state, counter_objects, pot_states_dict):
        cookable = self._start_cooking_motion_goals(pot_states_dict)
        if cookable:
            return cookable

        if not self._fillable_pots_exist(pot_states_dict):
            return []

        am = self.mlam
        ingredient_name = self._preferred_ingredient_name(state, pot_states_dict)
        if ingredient_name == "tomato" and self._tomato_sources_available(
            counter_objects
        ):
            return am.pickup_tomato_actions(counter_objects)
        return am.pickup_onion_actions(counter_objects)

    def _held_object_motion_goals(self, player, pot_states_dict, state=None):
        am = self.mlam
        obj_name = player.get_object().name

        if obj_name == "onion":
            return am.put_onion_in_pot_actions(pot_states_dict)
        if obj_name == "tomato":
            return am.put_tomato_in_pot_actions(pot_states_dict)
        if obj_name == "dish":
            return am.pickup_soup_with_dish_actions(
                pot_states_dict, only_nearly_ready=True
            )
        if obj_name == "soup":
            return am.deliver_soup_actions()
        raise ValueError("Unrecognized held object {}".format(obj_name))

    def _build_candidates(
        self,
        state,
        player,
        other,
        counter_objects,
        pot_states_dict,
        preferred_role,
    ):
        other_has_service_object = other.has_object() and other.get_object().name in [
            "dish",
            "soup",
        ]
        candidates = []

        if player.has_object():
            obj_name = player.get_object().name
            if obj_name == "soup":
                candidates.append(
                    self._candidate(
                        "deliver_soup",
                        self.SERVE_ROLE,
                        self._held_object_motion_goals(player, pot_states_dict),
                        10.0,
                    )
                )
                return candidates

            if obj_name == "dish":
                candidates.append(
                    self._candidate(
                        "pickup_soup",
                        self.SERVE_ROLE,
                        self._held_object_motion_goals(player, pot_states_dict),
                        8.5 if self._service_is_urgent(pot_states_dict) else 6.0,
                    )
                )
                self._append_drop_candidate_if_useful(
                    candidates,
                    state,
                    obj_name,
                    base_score=2.0,
                )
                return candidates

            if obj_name in ["onion", "tomato"]:
                task_name = "put_in_pot"
                candidates.append(
                    self._candidate(
                        task_name,
                        self.PREP_ROLE,
                        self._held_object_motion_goals(player, pot_states_dict),
                        7.0,
                    )
                )
                if self._service_is_urgent(pot_states_dict) and not self._fillable_pots_exist(
                    pot_states_dict
                ):
                    self._append_drop_candidate_if_useful(
                        candidates,
                        state,
                        obj_name,
                        base_score=7.5,
                    )
                else:
                    self._append_drop_candidate_if_useful(
                        candidates,
                        state,
                        obj_name,
                        base_score=1.5,
                    )
                return candidates

        counter_soups = counter_objects["soup"]
        ready = list(pot_states_dict["ready"])
        cooking = list(pot_states_dict["cooking"])

        if counter_soups:
            candidates.append(
                self._candidate(
                    "pickup_counter_soup",
                    self.SERVE_ROLE,
                    self.mlam.pickup_counter_soup_actions(counter_objects),
                    9.0,
                )
            )

        service_motion_goals = self._serving_motion_goals(
            state, counter_objects, pot_states_dict
        )
        if service_motion_goals:
            if ready:
                dish_score = 8.5 if not other_has_service_object else 5.0
            elif cooking:
                dish_score = 6.5 if not other_has_service_object else 3.5
            else:
                dish_score = 3.0
            candidates.append(
                self._candidate(
                    "get_dish",
                    self.SERVE_ROLE,
                    service_motion_goals,
                    dish_score,
                )
            )

        start_cooking_goals = self._start_cooking_motion_goals(pot_states_dict)
        if start_cooking_goals:
            candidates.append(
                self._candidate(
                    "start_cooking",
                    self.PREP_ROLE,
                    start_cooking_goals,
                    8.0,
                )
            )

        if self._fillable_pots_exist(pot_states_dict):
            ingredient_name = self._preferred_ingredient_name(state, pot_states_dict)
            if self._onion_sources_available(counter_objects):
                candidates.append(
                    self._candidate(
                        "get_onion",
                        self.PREP_ROLE,
                        self.mlam.pickup_onion_actions(counter_objects),
                        5.6 if ingredient_name == "onion" else 4.7,
                    )
                )
            if (
                ingredient_name == "tomato"
                and self._tomato_sources_available(counter_objects)
            ):
                candidates.append(
                    self._candidate(
                        "get_tomato",
                        self.PREP_ROLE,
                        self.mlam.pickup_tomato_actions(counter_objects),
                        5.6,
                    )
                )

        return candidates

    def _choose_best_candidate(self, start_pos_and_or, candidates):
        best = None
        best_score = float("-inf")
        for candidate in candidates:
            valid_goals = [
                mg
                for mg in candidate["motion_goals"]
                if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                    start_pos_and_or, mg
                )
            ]
            if not valid_goals:
                continue

            min_cost = min(
                self.mlam.motion_planner.get_plan(start_pos_and_or, mg)[2]
                for mg in valid_goals
            )
            score = candidate["base_score"] - 0.15 * min_cost

            if candidate["task_role"] == self.select_role_cache:
                score += 0.8
            elif candidate["task_role"] is not None:
                score -= 0.2

            if (
                candidate["task_role"] is not None
                and candidate["task_role"] == self.partner_role
            ):
                score -= 0.6

            if candidate["task"] == self.previous_task:
                score += 0.35

            if candidate["task"] in ["get_onion", "get_tomato"] and self._service_is_urgent(
                self.pot_states_cache
            ):
                score -= 1.2

            if score > best_score:
                best_score = score
                best = {
                    "task": candidate["task"],
                    "task_role": candidate["task_role"],
                    "motion_goals": valid_goals,
                    "score": score,
                }

        return best

    def _candidate(self, task, task_role, motion_goals, base_score):
        return {
            "task": task,
            "task_role": task_role,
            "motion_goals": motion_goals,
            "base_score": base_score,
        }

    def _append_drop_candidate_if_useful(
        self, candidates, state, obj_name, base_score
    ):
        drop_goals = self._drop_item_motion_goals(state)
        if drop_goals:
            candidates.append(
                self._candidate("drop_{}".format(obj_name), None, drop_goals, base_score)
            )

    def _start_cooking_motion_goals(self, pot_states_dict):
        cookable_pots = self.mlam.mdp.get_full_but_not_cooking_pots(
            pot_states_dict
        )
        only_full_pots = defaultdict(list)
        only_full_pots["3_items"] = cookable_pots
        return self.mlam.start_cooking_actions(only_full_pots)

    def _drop_item_motion_goals(self, state):
        return self.mlam.place_obj_on_counter_actions(state)

    def _fillable_pots_exist(self, pot_states_dict):
        return bool(
            pot_states_dict["empty"]
            or self.mlam.mdp.get_partially_full_pots(pot_states_dict)
        )

    def _service_is_urgent(self, pot_states_dict):
        return bool(pot_states_dict["ready"] or pot_states_dict["cooking"])

    def _onion_sources_available(self, counter_objects):
        return bool(
            self.mlam.mdp.get_onion_dispenser_locations()
            or counter_objects["onion"]
        )

    def _preferred_ingredient_name(self, state, pot_states_dict):
        if len(self.mlam.mdp.get_tomato_dispenser_locations()) == 0:
            return "onion"

        if not state.all_orders:
            return "onion"

        target_recipe = list(state.all_orders)[0]
        target_ingredients = list(target_recipe.ingredients)
        target_onion = target_ingredients.count("onion")
        target_tomato = target_ingredients.count("tomato")

        total_missing_onion = 0
        total_missing_tomato = 0
        relevant_pots = (
            list(pot_states_dict["empty"])
            + list(pot_states_dict["1_items"])
            + list(pot_states_dict["2_items"])
            + list(pot_states_dict.get("3_items", []))
        )

        if not relevant_pots:
            return "onion" if target_onion >= target_tomato else "tomato"

        for pot_pos in relevant_pots:
            onion_count = 0
            tomato_count = 0
            if state.has_object(pot_pos):
                soup = state.get_object(pot_pos)
                if soup.is_cooking or soup.is_ready:
                    continue
                onion_count = soup.ingredients.count("onion")
                tomato_count = soup.ingredients.count("tomato")
            total_missing_onion += max(0, target_onion - onion_count)
            total_missing_tomato += max(0, target_tomato - tomato_count)

        if total_missing_tomato > total_missing_onion:
            return "tomato"
        return "onion"

    def _distance_to_prep_chain(self, pos_and_or):
        prep_features = self.mlam.mdp.get_onion_dispenser_locations() + self.mlam.mdp.get_pot_locations()
        prep_features += self.mlam.mdp.get_tomato_dispenser_locations()
        return self._safe_feature_distance(pos_and_or, prep_features)

    def _distance_to_service_chain(self, pos_and_or):
        service_features = self.mlam.mdp.get_dish_dispenser_locations() + self.mlam.mdp.get_serving_locations()
        service_features += self.mlam.mdp.get_pot_locations()
        return self._safe_feature_distance(pos_and_or, service_features)

    def _safe_feature_distance(self, pos_and_or, feature_positions):
        if not feature_positions:
            return float("inf")
        return self.mlam.motion_planner.min_cost_to_feature(
            pos_and_or, feature_positions
        )

    def _dish_sources_available(self):
        return len(self.mlam.mdp.get_dish_dispenser_locations()) > 0

    def _tomato_sources_available(self, counter_objects):
        return bool(
            self.mlam.mdp.get_tomato_dispenser_locations()
            or counter_objects["tomato"]
        )

    def _get_unstuck_action(self, state):
        if self.agent_index == 0:
            joint_actions = list(
                itertools.product(Action.ALL_ACTIONS, [Action.STAY])
            )
        else:
            joint_actions = list(
                itertools.product([Action.STAY], Action.ALL_ACTIONS)
            )

        unblocking_joint_actions = []
        for j_a in joint_actions:
            new_state, _ = self.mlam.mdp.get_state_transition(state, j_a)
            if new_state.player_positions != self.prev_state.player_positions:
                unblocking_joint_actions.append(j_a)

        if len(unblocking_joint_actions) == 0:
            return Action.STAY, self.a_probs_from_action(Action.STAY)

        chosen_joint_action = random.choice(unblocking_joint_actions)
        chosen_action = chosen_joint_action[self.agent_index]
        return chosen_action, self.a_probs_from_action(chosen_action)

    def _is_truly_stuck(self, state):
        if self.prev_state is None:
            return False
        return state.time_independent_equal(self.prev_state)
