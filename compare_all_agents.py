#!/usr/bin/env python
"""
Compare StateAwarePlanningAgent, RoleAwareGreedyAgent, and BayesianBeliefAI
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel, RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.state_aware_planning_agent import StateAwarePlanningAgent
from overcooked_ai_py.agents.role_aware_agent import RoleAwareGreedyAgent
from overcooked_ai_py.mdp.actions import Action

# Add server code to path for BayesianBeliefAI
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(SCRIPT_DIR, "src", "overcooked_demo", "server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import game as server_game


DEFAULT_LAYOUTS = [
    "cramped_room",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]


def _ci95(values):
    arr = np.array(values, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / math.sqrt(len(arr)))


def _extract_agent_stats(trajectories, agent_index):
    """Extract behavioral stats for standard agents"""
    idle_ratios = []
    interact_ratios = []
    prep_ratios = []
    serve_ratios = []

    for ep_idx in range(len(trajectories["ep_actions"])):
        ep_actions = trajectories["ep_actions"][ep_idx]
        ep_infos = trajectories["ep_infos"][ep_idx]
        T = max(1, len(ep_actions))

        idle_count = 0
        interact_count = 0
        role_labels = []

        for t in range(T):
            a = ep_actions[t][agent_index]
            if a == Action.STAY:
                idle_count += 1
            if a == Action.INTERACT:
                interact_count += 1

            info = ep_infos[t]
            if info and "agent_infos" in info:
                agent_info = info["agent_infos"][agent_index]
                if isinstance(agent_info, dict) and "role" in agent_info:
                    role_labels.append(agent_info["role"])

        idle_ratios.append(idle_count / float(T))
        interact_ratios.append(interact_count / float(T))

        if role_labels:
            prep_count = sum(role == "prep_role" for role in role_labels)
            serve_count = sum(role == "serve_role" for role in role_labels)
            prep_ratios.append(prep_count / float(len(role_labels)))
            serve_ratios.append(serve_count / float(len(role_labels)))

    stats = {
        "idle_ratio_mean": float(np.mean(idle_ratios)),
        "interact_ratio_mean": float(np.mean(interact_ratios)),
    }
    if prep_ratios:
        stats["prep_ratio_mean"] = float(np.mean(prep_ratios))
        stats["serve_ratio_mean"] = float(np.mean(serve_ratios))
    return stats


def _extract_bayesian_stats(trajectories, agent_index):
    """Extract behavioral stats for Bayesian agent"""
    idle_ratios = []
    interact_ratios = []

    for ep_idx in range(len(trajectories["ep_actions"])):
        ep_actions = trajectories["ep_actions"][ep_idx]
        T = max(1, len(ep_actions))

        idle_count = 0
        interact_count = 0

        for t in range(T):
            a = ep_actions[t][agent_index]
            if a == Action.STAY:
                idle_count += 1
            if a == Action.INTERACT:
                interact_count += 1

        idle_ratios.append(idle_count / float(T))
        interact_ratios.append(interact_count / float(T))

    return {
        "idle_ratio_mean": float(np.mean(idle_ratios)),
        "interact_ratio_mean": float(np.mean(interact_ratios)),
    }


def _build_standard_agent(agent_kind, mlam):
    """Build standard agents (state_aware, role_aware, greedy, random)"""
    if agent_kind == "state_aware":
        return StateAwarePlanningAgent(mlam)
    if agent_kind == "role_aware":
        return RoleAwareGreedyAgent(mlam)
    if agent_kind == "greedy":
        return GreedyHumanModel(mlam)
    if agent_kind == "random":
        return RandomAgent(all_actions=Action.ALL_ACTIONS)
    raise ValueError(f"Unknown agent kind '{agent_kind}'")


def evaluate_standard_agent(layout, episodes, horizon, agent_kind, partner_kind="greedy"):
    """Evaluate a standard agent on a single layout"""
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": horizon},
    )

    focal_agent = _build_standard_agent(agent_kind, ae.env.mlam)
    partner_agent = _build_standard_agent(partner_kind, ae.env.mlam)
    ap = AgentPair(focal_agent, partner_agent)

    traj = ae.evaluate_agent_pair(ap, num_games=episodes, native_eval=True)
    returns = traj["ep_returns"].tolist()
    lengths = traj["ep_lengths"].tolist()

    return {
        "episodes": episodes,
        "mean_sparse_return": float(np.mean(returns)),
        "std_sparse_return": float(np.std(returns)),
        "ci95_sparse_return": _ci95(returns),
        "mean_episode_length": float(np.mean(lengths)),
        "estimated_soups_mean": float(np.mean(returns) / 20.0),
        **_extract_agent_stats(traj, agent_index=0),
    }


def evaluate_bayesian_agent(layout, episodes, horizon, model_path):
    """Evaluate BayesianBeliefAI on a single layout"""
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": horizon},
    )

    bayes = server_game.BayesianBeliefAI(
        agent_index=0,
        mdp_getter=lambda: ae.env.mdp,
        model_path=model_path,
        use_model_likelihood=True,
        use_transition_prior=True,
        use_commitment=True,
        use_yield=True,
    )
    partner = GreedyHumanModel(ae.env.mlam)
    ap = AgentPair(bayes, partner)

    traj = ae.evaluate_agent_pair(ap, num_games=episodes, native_eval=True)
    returns = traj["ep_returns"].tolist()
    lengths = traj["ep_lengths"].tolist()

    return {
        "episodes": episodes,
        "mean_sparse_return": float(np.mean(returns)),
        "std_sparse_return": float(np.std(returns)),
        "ci95_sparse_return": _ci95(returns),
        "mean_episode_length": float(np.mean(lengths)),
        "estimated_soups_mean": float(np.mean(returns) / 20.0),
        **_extract_bayesian_stats(traj, agent_index=0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare StateAwarePlanningAgent, RoleAwareGreedyAgent, and BayesianBeliefAI"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["state_aware", "role_aware", "greedy"],
        help="Standard agents to evaluate. Supported: state_aware, role_aware, greedy, random",
    )
    parser.add_argument(
        "--include-bayesian",
        action="store_true",
        help="Also evaluate BayesianBeliefAI agent",
    )
    parser.add_argument("--layouts", nargs="*", default=DEFAULT_LAYOUTS)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument(
        "--model-path",
        default="src/overcooked_demo/server/static/assets/agents/BayesianBeliefAI/model.pkl",
        help="Path to Bayesian model (if --include-bayesian)",
    )
    parser.add_argument(
        "--out",
        default="results/agent_comparison_all.json",
        help="Path for JSON summary output",
    )
    args = parser.parse_args()

    output = {
        "config": {
            "agents": args.agents,
            "include_bayesian": args.include_bayesian,
            "layouts": args.layouts,
            "episodes": args.episodes,
            "horizon": args.horizon,
        },
        "agents": {},
    }

    # Evaluate standard agents
    for agent_name in args.agents:
        print(f"\n{'='*60}")
        print(f"Evaluating {agent_name.upper()}")
        print(f"{'='*60}")
        per_layout = {}
        for layout in args.layouts:
            print(f"  Layout: {layout}...", end=" ", flush=True)
            per_layout[layout] = evaluate_standard_agent(
                layout=layout,
                episodes=args.episodes,
                horizon=args.horizon,
                agent_kind=agent_name,
            )
            print(f"✓ (mean_return={per_layout[layout]['mean_sparse_return']:.1f})")

        layout_returns = [
            per_layout[layout]["mean_sparse_return"] for layout in per_layout
        ]
        output["agents"][agent_name] = {
            "layouts": per_layout,
            "aggregate": {
                "mean_layout_return": float(np.mean(layout_returns)),
                "std_layout_return": float(np.std(layout_returns)),
            },
        }

    # Evaluate Bayesian agent if requested
    if args.include_bayesian:
        print(f"\n{'='*60}")
        print("Evaluating BAYESIAN BELIEF AI")
        print(f"{'='*60}")
        per_layout = {}
        for layout in args.layouts:
            print(f"  Layout: {layout}...", end=" ", flush=True)
            per_layout[layout] = evaluate_bayesian_agent(
                layout=layout,
                episodes=args.episodes,
                horizon=args.horizon,
                model_path=args.model_path,
            )
            print(f"✓ (mean_return={per_layout[layout]['mean_sparse_return']:.1f})")

        layout_returns = [
            per_layout[layout]["mean_sparse_return"] for layout in per_layout
        ]
        output["agents"]["bayesian"] = {
            "layouts": per_layout,
            "aggregate": {
                "mean_layout_return": float(np.mean(layout_returns)),
                "std_layout_return": float(np.std(layout_returns)),
            },
        }

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Results saved to {out_path}")
    print(f"{'='*60}\n")

    # Print summary
    print("SUMMARY")
    print("-" * 60)
    for agent_name in output["agents"]:
        agg = output["agents"][agent_name]["aggregate"]
        print(
            f"{agent_name:20s}: {agg['mean_layout_return']:6.1f} ± {agg['std_layout_return']:.1f} soups/game"
        )


if __name__ == "__main__":
    main()
