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


def _extract_agent_stats(trajectories, agent_index):
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
            prep_count = sum(
                role == RoleAwareGreedyAgent.PREP_ROLE for role in role_labels
            )
            serve_count = sum(
                role == RoleAwareGreedyAgent.SERVE_ROLE for role in role_labels
            )
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


def _build_agent(agent_kind, mlam):
    if agent_kind == "role_aware":
        return RoleAwareGreedyAgent(mlam)
    if agent_kind == "greedy":
        return GreedyHumanModel(mlam)
    if agent_kind == "random":
        return RandomAgent(all_actions=Action.ALL_ACTIONS)
    raise ValueError("Unknown agent kind {}".format(agent_kind))


def _evaluate_one_orientation(
    ae, focal_kind, partner_kind, focal_index, episodes
):
    focal_agent = _build_agent(focal_kind, ae.env.mlam)
    partner_agent = _build_agent(partner_kind, ae.env.mlam)

    if focal_index == 0:
        ap = AgentPair(focal_agent, partner_agent)
    else:
        ap = AgentPair(partner_agent, focal_agent)

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
        **_extract_agent_stats(traj, agent_index=focal_index),
    }


def _evaluate_layout(layout, episodes, horizon, focal_kind, partner_kind):
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": horizon},
    )

    by_index = {}
    for focal_index in [0, 1]:
        by_index[str(focal_index)] = _evaluate_one_orientation(
            ae=ae,
            focal_kind=focal_kind,
            partner_kind=partner_kind,
            focal_index=focal_index,
            episodes=episodes,
        )

    mean_returns = [by_index[idx]["mean_sparse_return"] for idx in by_index]
    mean_lengths = [by_index[idx]["mean_episode_length"] for idx in by_index]
    aggregate = {
        "mean_sparse_return": float(np.mean(mean_returns)),
        "std_sparse_return": float(np.std(mean_returns)),
        "mean_episode_length": float(np.mean(mean_lengths)),
        "estimated_soups_mean": float(np.mean(mean_returns) / 20.0),
    }

    role_keys = ["idle_ratio_mean", "interact_ratio_mean", "prep_ratio_mean", "serve_ratio_mean"]
    for key in role_keys:
        values = [by_index[idx][key] for idx in by_index if key in by_index[idx]]
        if values:
            aggregate[key] = float(np.mean(values))

    return {"by_index": by_index, "aggregate": aggregate}


def _evaluate_agent_family(layouts, episodes, horizon, focal_kind, partner_kind):
    per_layout = {}
    for layout in layouts:
        print(
            "Evaluating {} with partner {} on {}".format(
                focal_kind, partner_kind, layout
            )
        )
        per_layout[layout] = _evaluate_layout(
            layout=layout,
            episodes=episodes,
            horizon=horizon,
            focal_kind=focal_kind,
            partner_kind=partner_kind,
        )

    layout_returns = [
        per_layout[layout]["aggregate"]["mean_sparse_return"]
        for layout in per_layout
    ]
    return {
        "layouts": per_layout,
        "aggregate": {
            "mean_layout_return": float(np.mean(layout_returns)),
            "std_layout_return": float(np.std(layout_returns)),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare RoleAwareGreedyAgent against GreedyHumanModel"
    )
    parser.add_argument("--layouts", nargs="*", default=DEFAULT_LAYOUTS)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument(
        "--partner",
        choices=["greedy", "random"],
        default="greedy",
        help="Teammate policy used for both evaluated agents",
    )
    parser.add_argument(
        "--out",
        default="results/role_aware_vs_greedy.json",
        help="Path for JSON summary output",
    )
    args = parser.parse_args()

    output = {
        "config": {
            "layouts": args.layouts,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "partner": args.partner,
        },
        "agents": {
            "role_aware": _evaluate_agent_family(
                args.layouts,
                args.episodes,
                args.horizon,
                focal_kind="role_aware",
                partner_kind=args.partner,
            ),
            "greedy": _evaluate_agent_family(
                args.layouts,
                args.episodes,
                args.horizon,
                focal_kind="greedy",
                partner_kind=args.partner,
            ),
        },
    }

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("Saved comparison summary to {}".format(out_path))


if __name__ == "__main__":
    main()
