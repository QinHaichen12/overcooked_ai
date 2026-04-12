import argparse
import json
import math
import os
import sys

import numpy as np

from overcooked_ai_py.agents.agent import AgentPair, RandomAgent, GreedyHumanAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
SERVER_DIR = os.path.join(REPO_ROOT, "src", "overcooked_demo", "server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import game as server_game


DEFAULT_LAYOUTS = [
    "cramped_room",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]


def _entropy(prob_dict):
    eps = 1e-12
    p = np.array(list(prob_dict.values()), dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def _extract_diagnostics(trajectories, bayes_index):
    idle_ratios = []
    interact_ratios = []
    belief_entropies = []
    intent_switches = []

    for ep_idx in range(len(trajectories["ep_actions"])):
        ep_actions = trajectories["ep_actions"][ep_idx]
        ep_infos = trajectories["ep_infos"][ep_idx]
        T = max(1, len(ep_actions))

        idle_count = 0
        interact_count = 0
        ep_entropies = []
        intents = []

        for t in range(T):
            a = ep_actions[t][bayes_index]
            if a == Action.STAY:
                idle_count += 1
            if a == Action.INTERACT:
                interact_count += 1

            info = ep_infos[t]
            if info and "agent_infos" in info:
                agent_info = info["agent_infos"][bayes_index]
                if isinstance(agent_info, dict):
                    belief = agent_info.get("belief")
                    intent = agent_info.get("inferred_intent")
                    if isinstance(belief, dict) and belief:
                        ep_entropies.append(_entropy(belief))
                    if isinstance(intent, str):
                        intents.append(intent)

        switches = 0
        for i in range(1, len(intents)):
            if intents[i] != intents[i - 1]:
                switches += 1

        idle_ratios.append(idle_count / float(T))
        interact_ratios.append(interact_count / float(T))
        belief_entropies.append(float(np.mean(ep_entropies)) if ep_entropies else math.nan)
        intent_switches.append(switches / float(max(1, len(intents) - 1)))

    return {
        "idle_ratio_mean": float(np.nanmean(idle_ratios)),
        "interact_ratio_mean": float(np.nanmean(interact_ratios)),
        "belief_entropy_mean": float(np.nanmean(belief_entropies)),
        "intent_switch_rate_mean": float(np.nanmean(intent_switches)),
    }


def _ci95(values):
    arr = np.array(values, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / math.sqrt(len(arr)))


def evaluate_condition(
    layouts,
    episodes,
    horizon,
    model_path,
    use_model_likelihood,
    use_transition_prior,
    use_commitment,
    use_yield,
):
    per_layout = {}
    for layout in layouts:
        ae = AgentEvaluator.from_layout_name(
            mdp_params={"layout_name": layout, "old_dynamics": True},
            env_params={"horizon": horizon},
        )

        bayes = server_game.BayesianBeliefAI(
            agent_index=1,
            mdp_getter=lambda: ae.env.mdp,
            model_path=model_path,
            use_model_likelihood=use_model_likelihood,
            use_transition_prior=use_transition_prior,
            use_commitment=use_commitment,
            use_yield=use_yield,
        )
        # partner = RandomAgent(all_actions=Action.ALL_ACTIONS) # shouldnt we use greedyhuman here?
        partner = server_game.GreedyHumanAgent(agent_index=0, mdp_getter=lambda: ae.env.mdp)
        ap = AgentPair(partner, bayes)

        traj = ae.evaluate_agent_pair(ap, num_games=episodes, native_eval=True)
        returns = traj["ep_returns"].tolist()
        lengths = traj["ep_lengths"].tolist()
        diag = _extract_diagnostics(traj, bayes_index=1)

        per_layout[layout] = {
            "episodes": episodes,
            "mean_sparse_return": float(np.mean(returns)),
            "std_sparse_return": float(np.std(returns)),
            "ci95_sparse_return": _ci95(returns),
            "mean_episode_length": float(np.mean(lengths)),
            "estimated_soups_mean": float(np.mean(returns) / 20.0),
            **diag,
        }

    all_returns = [per_layout[l]["mean_sparse_return"] for l in per_layout]
    return {
        "layouts": per_layout,
        "aggregate": {
            "mean_layout_return": float(np.mean(all_returns)),
            "std_layout_return": float(np.std(all_returns)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BayesianBeliefAI and ablations")
    parser.add_argument("--layouts", nargs="*", default=DEFAULT_LAYOUTS)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument(
        "--model-path",
        default="src/overcooked_demo/server/static/assets/agents/BayesianBeliefAI/model.pkl",
    )
    parser.add_argument(
        "--out",
        default="results/bayesian_eval_summary.json",
        help="Path for JSON summary output",
    )
    args = parser.parse_args()

    conditions = {
        "full_model": {
            "use_model_likelihood": True,
            "use_transition_prior": True,
            "use_commitment": True,
            "use_yield": True,
        },
        "ablate_transition_prior": {
            "use_model_likelihood": True,
            "use_transition_prior": False,
            "use_commitment": True,
            "use_yield": True,
        },
        "ablate_model_likelihood": {
            "use_model_likelihood": False,
            "use_transition_prior": False,
            "use_commitment": True,
            "use_yield": True,
        },
        "ablate_commitment": {
            "use_model_likelihood": True,
            "use_transition_prior": True,
            "use_commitment": False,
            "use_yield": True,
        },
    }

    output = {
        "config": {
            "layouts": args.layouts,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "model_path": args.model_path,
        },
        "conditions": {},
    }

    for name, cfg in conditions.items():
        print("Running condition: {}".format(name))
        output["conditions"][name] = evaluate_condition(
            layouts=args.layouts,
            episodes=args.episodes,
            horizon=args.horizon,
            model_path=args.model_path,
            use_model_likelihood=cfg["use_model_likelihood"],
            use_transition_prior=cfg["use_transition_prior"],
            use_commitment=cfg["use_commitment"],
            use_yield=cfg["use_yield"],
        )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("Saved evaluation summary to {}".format(out_path))


if __name__ == "__main__":
    main()