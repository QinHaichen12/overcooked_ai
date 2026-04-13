import unittest

import numpy as np

from overcooked_ai_py.agents.agent import (
    AgentPair,
    FixedPlanAgent,
    GreedyHumanModel,
    RandomAgent,
    SampleAgent,
)
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.role_aware_agent import RoleAwareGreedyAgent
from overcooked_ai_py.agents.state_aware_planning_agent import (
    StateAwarePlanningAgent,
)
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
    SoupState,
)
from overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MediumLevelActionManager,
)

np.random.seed(42)

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState

force_compute_large = False
force_compute = True
DISPLAY = False

simple_mdp = OvercookedGridworld.from_layout_name("cramped_room")
large_mdp = OvercookedGridworld.from_layout_name("corridor")


class TestAgentEvaluator(unittest.TestCase):
    def setUp(self):
        self.agent_eval = AgentEvaluator.from_layout_name(
            {"layout_name": "cramped_room"}, {"horizon": 100}
        )

    def test_human_model_pair(self):
        trajs = self.agent_eval.evaluate_human_model_pair()
        try:
            AgentEvaluator.check_trajectories(trajs, verbose=False)
        except AssertionError as e:
            self.fail(
                "Trajectories were not returned in standard format:\n{}".format(
                    e
                )
            )

    def test_rollouts(self):
        ap = AgentPair(RandomAgent(), RandomAgent())
        trajs = self.agent_eval.evaluate_agent_pair(ap, num_games=5)
        try:
            AgentEvaluator.check_trajectories(trajs, verbose=False)
        except AssertionError as e:
            self.fail(
                "Trajectories were not returned in standard format:\n{}".format(
                    e
                )
            )

    def test_mlam_computation(self):
        try:
            self.agent_eval.env.mlam
        except Exception as e:
            self.fail(
                "Failed to compute MediumLevelActionManager:\n{}".format(e)
            )


class TestBasicAgents(unittest.TestCase):
    def setUp(self):
        self.mlam_large = MediumLevelActionManager.from_pickle_or_compute(
            large_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute_large
        )

    def test_fixed_plan_agents(self):
        a0 = FixedPlanAgent([s, e, n, w])
        a1 = FixedPlanAgent([s, w, n, e])
        agent_pair = AgentPair(a0, a1)
        env = OvercookedEnv.from_mdp(large_mdp, horizon=10)
        trajectory, time_taken, _, _ = env.run_agents(
            agent_pair, include_final_state=True, display=DISPLAY
        )
        end_state = trajectory[-1][0]
        self.assertEqual(time_taken, 10)
        self.assertEqual(
            env.mdp.get_standard_start_state().player_positions,
            end_state.player_positions,
        )

    def test_two_greedy_human_open_map(self):
        scenario_2_mdp = OvercookedGridworld.from_layout_name("scenario2")
        mlam = MediumLevelActionManager.from_pickle_or_compute(
            scenario_2_mdp, NO_COUNTERS_PARAMS, force_compute=force_compute
        )
        a0 = GreedyHumanModel(mlam)
        a1 = GreedyHumanModel(mlam)
        agent_pair = AgentPair(a0, a1)
        start_state = OvercookedState(
            [P((8, 1), s), P((1, 1), s)],
            {},
            all_orders=scenario_2_mdp.start_all_orders,
        )
        env = OvercookedEnv.from_mdp(
            scenario_2_mdp, start_state_fn=lambda: start_state, horizon=100
        )
        trajectory, time_taken, _, _ = env.run_agents(
            agent_pair, include_final_state=True, display=DISPLAY
        )

    def test_sample_agent(self):
        agent = SampleAgent(
            [RandomAgent(all_actions=False), RandomAgent(all_actions=True)]
        )
        probs = agent.action(None)[1]["action_probs"]
        expected_probs = np.array(
            [
                0.18333333,
                0.18333333,
                0.18333333,
                0.18333333,
                0.18333333,
                0.08333333,
            ]
        )
        self.assertTrue(np.allclose(probs, expected_probs))


class TestRoleAwareAgent(unittest.TestCase):
    def test_forced_coordination_enables_shared_counter_handoffs(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_no_counters_test_am.pkl",
            force_compute=force_compute,
        )
        agent = RoleAwareGreedyAgent(base_mlam)

        self.assertEqual(
            set(agent.mlam.params["counter_drop"]),
            {(2, 1), (2, 2), (2, 3)},
        )

        state = forced_mdp.get_standard_start_state()
        left_player = state.players[1]
        left_player.set_object(ObjectState("onion", left_player.position))

        drop_goals = agent._drop_item_motion_goals(state)
        self.assertTrue(drop_goals)

    def test_holding_onion_does_not_fall_through_to_dish_pickup(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_hold_onion_test_am.pkl",
            force_compute=force_compute,
        )
        agent = RoleAwareGreedyAgent(base_mlam)
        agent.set_agent_index(1)

        state = forced_mdp.get_standard_start_state()
        player = state.players[1]
        player.set_object(ObjectState("onion", player.position))

        agent._committed_held_object_decision = lambda *_args: (
            "put_in_pot",
            agent.PREP_ROLE,
            [],
        )
        agent._safe_wait_motion_goals = lambda _player: [_player.pos_and_or]
        agent._build_candidates = lambda *_args, **_kwargs: self.fail(
            "Held-object decisions should not fall through to dish pickup candidates"
        )

        motion_goals = agent.ml_action(state)
        self.assertEqual(motion_goals, [player.pos_and_or])
        self.assertEqual(agent.current_task, "put_in_pot")
        self.assertEqual(agent.current_role, agent.PREP_ROLE)

    def test_coordination_ring_uses_small_staging_counter_subset(self):
        ring_mdp = OvercookedGridworld.from_layout_name("coordination_ring")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            ring_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="coordination_ring_role_aware_test_am.pkl",
            force_compute=force_compute,
        )
        agent = RoleAwareGreedyAgent(base_mlam)

        self.assertGreater(len(agent.mlam.params["counter_drop"]), 0)
        self.assertLessEqual(len(agent.mlam.params["counter_drop"]), 4)
        self.assertLess(
            len(agent.mlam.params["counter_drop"]),
            len(ring_mdp.terrain_pos_dict["X"]),
        )


class TestStateAwarePlanningAgent(unittest.TestCase):
    def test_forced_coordination_enables_shared_counter_handoffs(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_test_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)

        self.assertEqual(
            set(agent.mlam.params["counter_drop"]),
            {(2, 1), (2, 2), (2, 3)},
        )

    def test_holding_onion_uses_handoff_when_blocked(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_handoff_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)

        state = forced_mdp.get_standard_start_state()
        player = state.players[1]
        player.set_object(ObjectState("onion", player.position))

        analysis = agent._analyze_state(state)
        decision = agent._plan_constrained_handoff(state, analysis)

        self.assertIsNotNone(decision)
        self.assertEqual(decision["task"], "drop_onion_for_handoff")

    def test_holding_onion_drops_before_dish_pickup_under_service_pressure(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_service_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)

        state = forced_mdp.get_standard_start_state()
        player = state.players[1]
        player.set_object(ObjectState("onion", player.position))

        pot_pos = forced_mdp.get_pot_locations()[0]
        state.objects[pot_pos] = SoupState.get_soup(
            pot_pos, num_onions=3, finished=True
        )

        motion_goals = agent.ml_action(state)

        self.assertTrue(motion_goals)
        self.assertTrue(agent.current_task.startswith("drop_onion"))
        self.assertNotEqual(agent.current_task, "get_dish")

    def test_holding_dish_creates_handoff_when_partner_can_serve(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_dish_handoff_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)

        state = forced_mdp.get_standard_start_state()
        player = state.players[1]
        player.set_object(ObjectState("dish", player.position))

        pot_pos = forced_mdp.get_pot_locations()[0]
        state.objects[pot_pos] = SoupState.get_soup(
            pot_pos, num_onions=3, finished=True
        )

        analysis = agent._analyze_state(state)
        decision = agent._plan_constrained_handoff(state, analysis)

        self.assertIsNotNone(decision)
        self.assertEqual(decision["task"], "drop_dish_for_handoff")

    def test_partial_pot_beats_dish_staging_in_constrained_handoff(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_partial_pot_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)

        state = forced_mdp.get_standard_start_state()
        pot_a, pot_b = forced_mdp.get_pot_locations()
        state.objects[pot_a] = SoupState.get_soup(pot_a, num_onions=1)
        state.objects[pot_b] = SoupState.get_soup(
            pot_b, num_onions=3, cooking_tick=0
        )

        analysis = agent._analyze_state(state)
        decision = agent._plan_constrained_handoff(state, analysis)

        self.assertEqual(analysis["primary_need"], "fill_pot")
        self.assertIsNotNone(decision)
        self.assertEqual(decision["task"], "get_onion_for_handoff")

    def test_holding_dish_drops_for_prep_when_partial_pot_needs_onion(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_drop_dish_prep_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)

        state = forced_mdp.get_standard_start_state()
        player = state.players[1]
        player.set_object(ObjectState("dish", player.position))
        pot_a, pot_b = forced_mdp.get_pot_locations()
        state.objects[pot_a] = SoupState.get_soup(pot_a, num_onions=1)
        state.objects[pot_b] = SoupState.get_soup(
            pot_b, num_onions=3, cooking_tick=0
        )

        motion_goals = agent.ml_action(state)

        self.assertTrue(motion_goals)
        self.assertEqual(agent.current_task, "drop_dish_for_prep")

    def test_giver_does_not_restage_reserved_onion(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_reserved_onion_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)  # Supply-side giver in forced_coordination

        state = forced_mdp.get_standard_start_state()
        pot_pos = forced_mdp.get_pot_locations()[0]
        state.objects[pot_pos] = SoupState.get_soup(pot_pos, num_onions=1)
        state.objects[(2, 2)] = ObjectState("onion", (2, 2))

        # Simulate reservation memory from a previous onion handoff.
        agent.counter_handoff_reservation = {
            "position": (2, 2),
            "object_name": "onion",
            "ttl": 3,
        }

        analysis = agent._analyze_state(state)
        decision = agent._plan_giver_prefetch(state, analysis)

        self.assertEqual(analysis["handoff_role"], "giver")
        self.assertIsNone(decision)

    def test_giver_respects_partner_held_onion_pipeline(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_partner_onion_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)  # Supply-side giver

        state = forced_mdp.get_standard_start_state()
        pot_a, pot_b = forced_mdp.get_pot_locations()
        state.objects[pot_a] = SoupState.get_soup(pot_a, num_onions=2)
        state.objects[pot_b] = SoupState.get_soup(
            pot_b, num_onions=3, cooking_tick=0
        )
        state.players[0].set_object(ObjectState("onion", state.players[0].position))

        analysis = agent._analyze_state(state)
        decision = agent._plan_giver_prefetch(state, analysis)

        self.assertEqual(analysis["handoff_role"], "giver")
        self.assertIsNotNone(decision)
        self.assertEqual(decision["task"], "giver_fetch_dish_for_partner")

    def test_giver_does_not_fetch_extra_onion_when_partner_has_last_needed_one(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_last_needed_onion_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)  # Supply-side giver

        state = forced_mdp.get_standard_start_state()
        pot_a, pot_b = forced_mdp.get_pot_locations()
        state.objects[pot_a] = SoupState.get_soup(pot_a, num_onions=2)
        state.objects[pot_b] = SoupState.get_soup(pot_b, num_onions=3)
        state.players[0].set_object(ObjectState("onion", state.players[0].position))

        analysis = agent._analyze_state(state)
        decision = agent._plan_giver_prefetch(state, analysis)

        self.assertEqual(analysis["handoff_role"], "giver")
        self.assertIsNone(decision)

    def test_giver_holding_onion_clears_for_dish_when_pipeline_satisfied(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_hold_onion_pipeline_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)  # Supply-side giver

        state = forced_mdp.get_standard_start_state()
        pot_a, pot_b = forced_mdp.get_pot_locations()
        state.objects[pot_a] = SoupState.get_soup(pot_a, num_onions=2)
        state.objects[pot_b] = SoupState.get_soup(
            pot_b, num_onions=3, cooking_tick=0
        )
        state.players[1].set_object(ObjectState("onion", state.players[1].position))
        state.players[0].set_object(ObjectState("onion", state.players[0].position))

        analysis = agent._analyze_state(state)
        decision = agent._plan_ingredient_in_hand(state, analysis, "onion")

        self.assertEqual(analysis["handoff_role"], "giver")
        self.assertIsNotNone(decision)
        self.assertEqual(decision["task"], "drop_onion_clear_for_dish")

    def test_giver_holding_onion_relays_when_pipeline_is_short(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_hold_onion_relay_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)  # Supply-side giver

        state = forced_mdp.get_standard_start_state()
        pot_a, _ = forced_mdp.get_pot_locations()
        state.objects[pot_a] = SoupState.get_soup(pot_a, num_onions=2)
        state.players[1].set_object(ObjectState("onion", state.players[1].position))

        analysis = agent._analyze_state(state)
        decision = agent._plan_ingredient_in_hand(state, analysis, "onion")

        self.assertEqual(analysis["handoff_role"], "giver")
        self.assertIsNotNone(decision)
        self.assertEqual(decision["task"], "drop_onion_for_partner")

    def test_held_object_path_does_not_fall_through_to_empty_hand_planners(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_hold_path_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)  # Supply-side giver

        state = forced_mdp.get_standard_start_state()
        player = state.players[1]
        player.set_object(ObjectState("dish", player.position))
        pot_pos = forced_mdp.get_pot_locations()[0]
        state.objects[pot_pos] = SoupState.get_soup(
            pot_pos, num_onions=3, cooking_tick=0
        )

        # Fill the handoff lane so dish relay has no valid destination.
        for counter_pos in agent.mlam.params["counter_drop"]:
            state.objects[counter_pos] = ObjectState("onion", counter_pos)

        motion_goals = agent.ml_action(state)

        self.assertTrue(motion_goals)
        self.assertEqual(agent.current_task, "hold_dish")

    def test_giver_holding_onion_clears_for_dish_when_pot_active(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_clear_for_dish_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)  # Supply-side giver

        state = forced_mdp.get_standard_start_state()
        player = state.players[1]
        player.set_object(ObjectState("onion", player.position))
        pot_pos = forced_mdp.get_pot_locations()[0]
        state.objects[pot_pos] = SoupState.get_soup(
            pot_pos, num_onions=3, cooking_tick=0
        )

        analysis = agent._analyze_state(state)
        decision = agent._plan_ingredient_in_hand(state, analysis, "onion")

        self.assertEqual(analysis["handoff_role"], "giver")
        self.assertIsNotNone(decision)
        self.assertEqual(decision["task"], "drop_onion_clear_for_dish")

    def test_handoff_reservation_excludes_own_dropped_onion_from_pickup(self):
        forced_mdp = OvercookedGridworld.from_layout_name("forced_coordination")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            forced_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="forced_coordination_state_aware_reservation_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)
        agent.set_agent_index(1)

        prev_state = forced_mdp.get_standard_start_state()
        prev_player = prev_state.players[1]
        prev_player.set_object(ObjectState("onion", prev_player.position))

        curr_state = prev_state.deepcopy()
        curr_player = curr_state.players[1]
        curr_player.remove_object()
        curr_state.objects[(2, 2)] = ObjectState("onion", (2, 2))

        agent.previous_task = "drop_onion_for_handoff"
        agent.prev_observed_state = prev_state
        agent._update_handoff_memory(curr_state)

        analysis = agent._analyze_state(curr_state)
        self.assertIsNotNone(agent.counter_handoff_reservation)
        self.assertIn((2, 2), analysis["counter_objects"]["onion"])
        self.assertNotIn((2, 2), analysis["pickup_counter_objects"]["onion"])

    def test_coordination_ring_uses_small_staging_counter_subset(self):
        ring_mdp = OvercookedGridworld.from_layout_name("coordination_ring")
        base_mlam = MediumLevelActionManager.from_pickle_or_compute(
            ring_mdp,
            NO_COUNTERS_PARAMS,
            custom_filename="coordination_ring_state_aware_test_am.pkl",
            force_compute=force_compute,
        )
        agent = StateAwarePlanningAgent(base_mlam)

        self.assertGreater(len(agent.mlam.params["counter_drop"]), 0)
        self.assertLessEqual(len(agent.mlam.params["counter_drop"]), 4)
        self.assertLess(
            len(agent.mlam.params["counter_drop"]),
            len(ring_mdp.terrain_pos_dict["X"]),
        )


class TestAgentEvaluatorStatic(unittest.TestCase):
    layout_name_lst = [
        "asymmetric_advantages",
        "asymmetric_advantages_tomato",
        "bonus_order_test",
        "bottleneck",
        "centre_objects",
        "centre_pots",
        "corridor",
        "forced_coordination_tomato",
        "unident",
        "marshmallow_experiment",
        "marshmallow_experiment_coordination",
        "you_shall_not_pass",
    ]

    def test_from_mdp(self):
        for layout_name in self.layout_name_lst:
            orignal_mdp = OvercookedGridworld.from_layout_name(layout_name)
            ae = AgentEvaluator.from_mdp(
                mdp=orignal_mdp, env_params={"horizon": 400}
            )
            ae_mdp = ae.env.mdp
            self.assertEqual(
                orignal_mdp,
                ae_mdp,
                "mdp with name "
                + layout_name
                + " experienced an inconsistency",
            )

    def test_from_mdp_params_layout(self):
        for layout_name in self.layout_name_lst:
            orignal_mdp = OvercookedGridworld.from_layout_name(layout_name)
            ae = AgentEvaluator.from_layout_name(
                mdp_params={"layout_name": layout_name},
                env_params={"horizon": 400},
            )
            ae_mdp = ae.env.mdp
            self.assertEqual(
                orignal_mdp,
                ae_mdp,
                "mdp with name "
                + layout_name
                + " experienced an inconsistency",
            )

    mdp_gen_params_1 = {
        "inner_shape": (10, 7),
        "prop_empty": 0.95,
        "prop_feats": 0.1,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "display": False,
    }

    mdp_gen_params_2 = {
        "inner_shape": (10, 7),
        "prop_empty": 0.7,
        "prop_feats": 0.5,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "display": False,
    }

    mdp_gen_params_3 = {
        "inner_shape": (10, 7),
        "prop_empty": 0.5,
        "prop_feats": 0.4,
        "start_all_orders": [{"ingredients": ["onion", "onion", "onion"]}],
        "display": False,
    }

    mdp_gen_params_lst = [mdp_gen_params_1, mdp_gen_params_2, mdp_gen_params_3]

    outer_shape = (10, 7)

    def test_from_mdp_params_variable_across(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae0 = AgentEvaluator.from_mdp_params_infinite(
                mdp_params=mdp_gen_params,
                env_params={"horizon": 400, "num_mdp": np.inf},
                outer_shape=self.outer_shape,
            )
            ae1 = AgentEvaluator.from_mdp_params_infinite(
                mdp_params=mdp_gen_params,
                env_params={"horizon": 400, "num_mdp": np.inf},
                outer_shape=self.outer_shape,
            )
            self.assertFalse(
                ae0.env.mdp == ae1.env.mdp,
                "2 randomly generated layouts across 2 evaluators are the same, which is wrong",
            )

    def test_from_mdp_params_variable_infinite(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae = AgentEvaluator.from_mdp_params_infinite(
                mdp_params=mdp_gen_params,
                env_params={"horizon": 400, "num_mdp": np.inf},
                outer_shape=self.outer_shape,
            )
            mdp_0 = ae.env.mdp.copy()
            for _ in range(5):
                ae.env.reset(regen_mdp=True)
                mdp_1 = ae.env.mdp
                self.assertFalse(
                    mdp_0 == mdp_1,
                    "with infinite layout generator and regen_mdp=True, the 2 layouts should not be the same",
                )

    def test_from_mdp_params_variable_infinite_no_regen(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae = AgentEvaluator.from_mdp_params_infinite(
                mdp_params=mdp_gen_params,
                env_params={"horizon": 400, "num_mdp": np.inf},
                outer_shape=self.outer_shape,
            )
            mdp_0 = ae.env.mdp.copy()
            for _ in range(5):
                ae.env.reset(regen_mdp=False)
                mdp_1 = ae.env.mdp
                self.assertTrue(
                    mdp_0 == mdp_1,
                    "with infinite layout generator and regen_mdp=False, the 2 layouts should be the same",
                )

    def test_from_mdp_params_variable_infinite_specified(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae = AgentEvaluator.from_mdp_params_infinite(
                mdp_params=mdp_gen_params,
                env_params={"horizon": 400, "num_mdp": np.inf},
                outer_shape=self.outer_shape,
            )
            mdp_0 = ae.env.mdp.copy()
            for _ in range(5):
                ae.env.reset(regen_mdp=True)
                mdp_1 = ae.env.mdp
                self.assertFalse(
                    mdp_0 == mdp_1,
                    "with infinite layout generator and regen_mdp=True, the 2 layouts should not be the same",
                )

    def test_from_mdp_params_variable_finite(self):
        for mdp_gen_params in self.mdp_gen_params_lst:
            ae = AgentEvaluator.from_mdp_params_finite(
                mdp_params=mdp_gen_params,
                env_params={"horizon": 400, "num_mdp": 2},
                outer_shape=self.outer_shape,
            )
            mdp_0 = ae.env.mdp.copy()
            seen = [mdp_0]
            for _ in range(20):
                ae.env.reset(regen_mdp=True)
                mdp_i = ae.env.mdp
                if len(seen) == 1:
                    if mdp_i != seen[0]:
                        seen.append(mdp_i.copy())
                elif len(seen) == 2:
                    mdp_0, mdp_1 = seen
                    self.assertTrue(
                        (mdp_i == mdp_0 or mdp_i == mdp_1),
                        "more than 2 mdp was created, the function failed to perform",
                    )
                else:
                    self.assertTrue(
                        False, "theoretically unreachable statement"
                    )

    layout_name_short_lst = [
        "cramped_room",
        "cramped_room_tomato",
        "simple_o",
        "simple_tomato",
        "simple_o_t",
    ]
    biased = [0.1, 0.15, 0.2, 0.25, 0.3]
    num_reset = 200000

    def test_from_mdp_lst_default(self):
        mdp_lst = [
            OvercookedGridworld.from_layout_name(name)
            for name in self.layout_name_short_lst
        ]
        ae = AgentEvaluator.from_mdp_lst(
            mdp_lst=mdp_lst, env_params={"horizon": 400}
        )
        counts = {}

        for _ in range(self.num_reset):
            ae.env.reset(regen_mdp=True)
            if ae.env.mdp.layout_name in counts:
                counts[ae.env.mdp.layout_name] += 1
            else:
                counts[ae.env.mdp.layout_name] = 1

        for k, v in counts.items():
            self.assertAlmostEqual(
                0.2, v / self.num_reset, 2, "more than 2 places off for " + k
            )

    def test_from_mdp_lst_uniform(self):
        mdp_lst = [
            OvercookedGridworld.from_layout_name(name)
            for name in self.layout_name_short_lst
        ]
        ae = AgentEvaluator.from_mdp_lst(
            mdp_lst=mdp_lst,
            env_params={"horizon": 400},
            sampling_freq=[0.2, 0.2, 0.2, 0.2, 0.2],
        )
        counts = {}

        for _ in range(self.num_reset):
            ae.env.reset(regen_mdp=True)
            if ae.env.mdp.layout_name in counts:
                counts[ae.env.mdp.layout_name] += 1
            else:
                counts[ae.env.mdp.layout_name] = 1

        for k, v in counts.items():
            self.assertAlmostEqual(
                0.2, v / self.num_reset, 2, "more than 2 places off for " + k
            )

    def test_from_mdp_lst_biased(self):
        mdp_lst = [
            OvercookedGridworld.from_layout_name(name)
            for name in self.layout_name_short_lst
        ]
        ae = AgentEvaluator.from_mdp_lst(
            mdp_lst=mdp_lst,
            env_params={"horizon": 400},
            sampling_freq=self.biased,
        )
        counts = {}

        for _ in range(self.num_reset):
            ae.env.reset(regen_mdp=True)
            if ae.env.mdp.layout_name in counts:
                counts[ae.env.mdp.layout_name] += 1
            else:
                counts[ae.env.mdp.layout_name] = 1

        # construct the ground truth
        gt = {
            self.layout_name_short_lst[i]: self.biased[i]
            for i in range(len(self.layout_name_short_lst))
        }

        for k, v in counts.items():
            self.assertAlmostEqual(
                gt[k], v / self.num_reset, 2, "more than 2 places off for " + k
            )


if __name__ == "__main__":
    unittest.main()
