import argparse
import json
import math
import os

import numpy as np

from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel, RandomAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.role_aware_agent import RoleAwareGreedyAgent
from overcooked_ai_py.mdp.actions import Action


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


def _extract_role_stats(trajectories, role_agent_index):
    prep_ratios = []
    serve_ratios = []

    for ep_idx in range(len(trajectories["ep_actions"])):
        ep_infos = trajectories["ep_infos"][ep_idx]
        role_labels = []

        for info in ep_infos:
            if not info or "agent_infos" not in info:
                continue
            agent_info = info["agent_infos"][role_agent_index]
            if isinstance(agent_info, dict) and "role" in agent_info:
                role_labels.append(agent_info["role"])

        if not role_labels:
            continue

        prep_count = sum(role == RoleAwareGreedyAgent.PREP_ROLE for role in role_labels)
        serve_count = sum(role == RoleAwareGreedyAgent.SERVE_ROLE for role in role_labels)
        prep_ratios.append(prep_count / float(len(role_labels)))
        serve_ratios.append(serve_count / float(len(role_labels)))

    if not prep_ratios:
        return {"prep_ratio_mean": math.nan, "serve_ratio_mean": math.nan}

    return {
        "prep_ratio_mean": float(np.mean(prep_ratios)),
        "serve_ratio_mean": float(np.mean(serve_ratios)),
    }


def _make_partner(kind, mlam):
    if kind == "greedy":
        return GreedyHumanModel(mlam)
    if kind == "random":
        return RandomAgent(all_actions=Action.ALL_ACTIONS)
    raise ValueError("Unknown partner kind {}".format(kind))


def _evaluate_layout(layout, episodes, horizon, partner_kind, role_agent_index):
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": horizon},
    )

    role_agent = RoleAwareGreedyAgent(ae.env.mlam)
    partner = _make_partner(partner_kind, ae.env.mlam)

    if role_agent_index == 0:
        ap = AgentPair(role_agent, partner)
    else:
        ap = AgentPair(partner, role_agent)

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
        **_extract_role_stats(traj, role_agent_index=role_agent_index),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RoleAwareGreedyAgent against a selected partner"
    )
    parser.add_argument("--layouts", nargs="*", default=DEFAULT_LAYOUTS)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument(
        "--partner",
        choices=["greedy", "random"],
        default="greedy",
        help="Teammate policy to pair with RoleAwareGreedyAgent",
    )
    parser.add_argument(
        "--role-index",
        type=int,
        choices=[0, 1],
        default=1,
        help="Player index controlled by RoleAwareGreedyAgent",
    )
    parser.add_argument(
        "--out",
        default="results/role_aware_eval_summary.json",
        help="Path for JSON summary output",
    )
    args = parser.parse_args()

    per_layout = {}
    for layout in args.layouts:
        print("Evaluating layout: {}".format(layout))
        per_layout[layout] = _evaluate_layout(
            layout=layout,
            episodes=args.episodes,
            horizon=args.horizon,
            partner_kind=args.partner,
            role_agent_index=args.role_index,
        )

    mean_returns = [per_layout[layout]["mean_sparse_return"] for layout in per_layout]
    output = {
        "config": {
            "layouts": args.layouts,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "partner": args.partner,
            "role_index": args.role_index,
        },
        "layouts": per_layout,
        "aggregate": {
            "mean_layout_return": float(np.mean(mean_returns)),
            "std_layout_return": float(np.std(mean_returns)),
        },
    }

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("Saved role-aware evaluation summary to {}".format(out_path))


if __name__ == "__main__":
    main()
