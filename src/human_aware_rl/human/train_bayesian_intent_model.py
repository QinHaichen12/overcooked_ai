import argparse
import json
import os
import pickle
from collections import defaultdict

import pandas as pd

from human_aware_rl.human.data_processing_utils import (
    json_joint_action_to_python_action,
    json_state_to_python_state,
)
from human_aware_rl.static import CLEAN_2020_HUMAN_DATA_TRAIN
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


INTENTS = [
    "get_onion",
    "get_tomato",
    "get_dish",
    "put_in_pot",
    "start_cooking",
    "pickup_soup",
    "deliver_soup",
]

HELD_BUCKETS = ["none", "onion", "tomato", "dish", "soup", "other"]
PROX_BUCKETS = ["none", "onion", "tomato", "dish", "pot", "serve"]
CONTEXT_BUCKETS = [
    "ready",
    "cooking",
    "ready_to_cook",
    "needs_ingredient",
    "other",
]
PROGRESS_BUCKETS = [
    "toward_onion",
    "toward_tomato",
    "toward_dish",
    "toward_pot",
    "toward_serve",
    "none",
]


def _layout_group(layout_name):
    constrained_tokens = [
        "cramped",
        "corridor",
        "forced",
        "counter_circuit",
        "you_shall_not_pass",
        "bottleneck",
    ]
    lowered = layout_name.lower()
    if any(tok in lowered for tok in constrained_tokens):
        return "constrained"
    return "open"


def _near_any(pos, targets, dist=1):
    for t in targets:
        if abs(pos[0] - t[0]) + abs(pos[1] - t[1]) <= dist:
            return True
    return False


def _held_bucket(player):
    if not player.has_object():
        return "none"
    obj = player.get_object().name
    if obj in HELD_BUCKETS:
        return obj
    return "other"


def _proximity_bucket(player, mdp):
    pos = player.position
    if _near_any(pos, mdp.get_onion_dispenser_locations()):
        return "onion"
    if _near_any(pos, mdp.get_tomato_dispenser_locations()):
        return "tomato"
    if _near_any(pos, mdp.get_dish_dispenser_locations()):
        return "dish"
    if _near_any(pos, mdp.get_pot_locations()):
        return "pot"
    if _near_any(pos, mdp.get_serving_locations()):
        return "serve"
    return "none"


def _min_distance(pos, targets):
    if not targets:
        return None
    return min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in targets)


def _context_bucket(state, mdp):
    pot_states = mdp.get_pot_states(state)
    if pot_states["ready"]:
        return "ready"
    if pot_states["cooking"]:
        return "cooking"
    if pot_states.get("3_items"):
        return "ready_to_cook"

    open_pots = (
        list(pot_states["empty"])
        + list(pot_states["1_items"])
        + list(pot_states["2_items"])
    )
    if open_pots:
        return "needs_ingredient"
    return "other"


def _progress_bucket(player, mdp, action):
    if action not in Action.MOTION_ACTIONS:
        return "none"

    next_pos = Action.move_in_direction(player.position, action)
    feature_targets = [
        ("toward_onion", mdp.get_onion_dispenser_locations()),
        ("toward_tomato", mdp.get_tomato_dispenser_locations()),
        ("toward_dish", mdp.get_dish_dispenser_locations()),
        ("toward_pot", mdp.get_pot_locations()),
        ("toward_serve", mdp.get_serving_locations()),
    ]

    best_bucket = "none"
    best_improvement = 0
    for bucket, targets in feature_targets:
        current_dist = _min_distance(player.position, targets)
        next_dist = _min_distance(next_pos, targets)
        if current_dist is None or next_dist is None:
            continue
        improvement = current_dist - next_dist
        if improvement > best_improvement:
            best_improvement = improvement
            best_bucket = bucket

    return best_bucket


def _infer_intent(player, state, mdp, player_action=None):
    if player.has_object():
        obj = player.get_object().name
        if obj in ["onion", "tomato"]:
            if player_action == Action.INTERACT and _near_any(
                player.position, mdp.get_pot_locations()
            ):
                return "put_in_pot", 0.99
            return "put_in_pot", 0.95
        if obj == "dish":
            if player_action == Action.INTERACT and _near_any(
                player.position, mdp.get_pot_locations()
            ):
                return "pickup_soup", 0.98
            return "pickup_soup", 0.90
        if obj == "soup":
            if player_action == Action.INTERACT and _near_any(
                player.position, mdp.get_serving_locations()
            ):
                return "deliver_soup", 0.99
            return "deliver_soup", 0.98

    near_onion = _near_any(player.position, mdp.get_onion_dispenser_locations())
    near_tomato = _near_any(player.position, mdp.get_tomato_dispenser_locations())
    near_dish = _near_any(player.position, mdp.get_dish_dispenser_locations())
    near_pot = _near_any(player.position, mdp.get_pot_locations())
    near_serve = _near_any(player.position, mdp.get_serving_locations())
    pot_states = mdp.get_pot_states(state)
    context = _context_bucket(state, mdp)
    progress = _progress_bucket(player, mdp, player_action)

    if player_action == Action.INTERACT and near_pot and pot_states.get("3_items"):
        return "start_cooking", 0.95

    if near_onion or progress == "toward_onion":
        return "get_onion", 0.70
    if near_tomato or progress == "toward_tomato":
        return "get_tomato", 0.70
    if near_dish or progress == "toward_dish":
        if context in ["ready", "cooking"]:
            return "get_dish", 0.78
        return "get_dish", 0.70
    if near_pot and (pot_states["ready"] or pot_states["cooking"]):
        return "start_cooking", 0.65
    if near_pot and context == "ready_to_cook":
        return "start_cooking", 0.72
    if progress == "toward_pot" and context in ["needs_ingredient", "ready_to_cook"]:
        return "put_in_pot", 0.62
    if progress == "toward_serve":
        return "deliver_soup", 0.62
    if near_serve:
        return "deliver_soup", 0.55

    return "get_onion", 0.40


def _normalize_counts(counter, support, alpha):
    total = sum(counter.get(x, 0.0) for x in support) + alpha * len(support)
    if total <= 0:
        p = 1.0 / len(support)
        return {x: p for x in support}
    return {x: (counter.get(x, 0.0) + alpha) / total for x in support}


def train_model(df, layouts=None, alpha=1.0):
    if layouts:
        df = df[df["layout_name"].isin(layouts)]

    required_cols = ["trial_id", "layout_name", "cur_gameloop", "state", "joint_action"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {}".format(missing))

    df = df.sort_values(["trial_id", "cur_gameloop"])

    prior_counts = defaultdict(lambda: defaultdict(float))
    layout_prior_counts = defaultdict(lambda: defaultdict(float))
    trans_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    layout_trans_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    held_emission_counts = defaultdict(lambda: defaultdict(float))
    prox_emission_counts = defaultdict(lambda: defaultdict(float))
    action_counts = defaultdict(lambda: defaultdict(float))
    context_emission_counts = defaultdict(lambda: defaultdict(float))
    progress_emission_counts = defaultdict(lambda: defaultdict(float))
    layout_held_emission_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    layout_prox_emission_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    layout_action_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    layout_context_emission_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    layout_progress_emission_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    prev_intent = {}
    mdp_cache = {}
    rows_processed = 0

    for _, row in df.iterrows():
        layout_name = row["layout_name"]
        group = _layout_group(layout_name)
        trial_id = row["trial_id"]

        if layout_name not in mdp_cache:
            mdp_cache[layout_name] = OvercookedGridworld.from_layout_name(layout_name)
        mdp = mdp_cache[layout_name]

        state = json_state_to_python_state(row["state"])
        joint_action = json_joint_action_to_python_action(row["joint_action"])

        for player_idx in [0, 1]:
            player = state.players[player_idx]
            player_action = joint_action[player_idx]
            intent, confidence = _infer_intent(
                player, state, mdp, player_action=player_action
            )
            held_bucket = _held_bucket(player)
            prox_bucket = _proximity_bucket(player, mdp)
            context_bucket = _context_bucket(state, mdp)
            progress_bucket = _progress_bucket(player, mdp, player_action)
            action_key = str(player_action)

            prior_counts[group][intent] += confidence
            layout_prior_counts[layout_name][intent] += confidence
            held_emission_counts[intent][held_bucket] += confidence
            prox_emission_counts[intent][prox_bucket] += confidence
            action_counts[intent][action_key] += confidence
            context_emission_counts[intent][context_bucket] += confidence
            progress_emission_counts[intent][progress_bucket] += confidence
            layout_held_emission_counts[layout_name][intent][held_bucket] += confidence
            layout_prox_emission_counts[layout_name][intent][prox_bucket] += confidence
            layout_action_counts[layout_name][intent][action_key] += confidence
            layout_context_emission_counts[layout_name][intent][context_bucket] += confidence
            layout_progress_emission_counts[layout_name][intent][progress_bucket] += confidence

            seq_key = (trial_id, player_idx)
            if seq_key in prev_intent:
                trans_counts[group][prev_intent[seq_key]][intent] += confidence
                layout_trans_counts[layout_name][prev_intent[seq_key]][intent] += confidence
            prev_intent[seq_key] = intent

            rows_processed += 1

    model = {
        "version": 1,
        "alpha": alpha,
        "intents": list(INTENTS),
        "layout_groups": ["constrained", "open"],
        "held_buckets": list(HELD_BUCKETS),
        "proximity_buckets": list(PROX_BUCKETS),
        "context_buckets": list(CONTEXT_BUCKETS),
        "progress_buckets": list(PROGRESS_BUCKETS),
        "priors": {},
        "transitions": {},
        "layout_priors": {},
        "layout_transitions": {},
        "emissions": {
            "held": {},
            "proximity": {},
            "action": {},
            "context": {},
            "progress": {},
        },
        "layout_emissions": {
            "held": {},
            "proximity": {},
            "action": {},
            "context": {},
            "progress": {},
        },
        "stats": {
            "timesteps": int(len(df)),
            "player_rows": int(rows_processed),
            "layouts": sorted(list(set(df["layout_name"]))),
            "num_trials": int(df["trial_id"].nunique()),
        },
    }

    for group in ["constrained", "open"]:
        model["priors"][group] = _normalize_counts(prior_counts[group], INTENTS, alpha)
        model["transitions"][group] = {}
        for src in INTENTS:
            model["transitions"][group][src] = _normalize_counts(
                trans_counts[group][src], INTENTS, alpha
            )

    for layout_name in sorted(set(df["layout_name"])):
        model["layout_priors"][layout_name] = _normalize_counts(
            layout_prior_counts[layout_name], INTENTS, alpha
        )
        model["layout_transitions"][layout_name] = {}
        for src in INTENTS:
            model["layout_transitions"][layout_name][src] = _normalize_counts(
                layout_trans_counts[layout_name][src], INTENTS, alpha
            )

    for intent in INTENTS:
        model["emissions"]["held"][intent] = _normalize_counts(
            held_emission_counts[intent], HELD_BUCKETS, alpha
        )
        model["emissions"]["proximity"][intent] = _normalize_counts(
            prox_emission_counts[intent], PROX_BUCKETS, alpha
        )
        action_support = [str(a) for a in Action.ALL_ACTIONS]
        model["emissions"]["action"][intent] = _normalize_counts(
            action_counts[intent], action_support, alpha
        )
        model["emissions"]["context"][intent] = _normalize_counts(
            context_emission_counts[intent], CONTEXT_BUCKETS, alpha
        )
        model["emissions"]["progress"][intent] = _normalize_counts(
            progress_emission_counts[intent], PROGRESS_BUCKETS, alpha
        )

    for layout_name in sorted(set(df["layout_name"])):
        model["layout_emissions"]["held"][layout_name] = {}
        model["layout_emissions"]["proximity"][layout_name] = {}
        model["layout_emissions"]["action"][layout_name] = {}
        model["layout_emissions"]["context"][layout_name] = {}
        model["layout_emissions"]["progress"][layout_name] = {}
        for intent in INTENTS:
            model["layout_emissions"]["held"][layout_name][intent] = _normalize_counts(
                layout_held_emission_counts[layout_name][intent], HELD_BUCKETS, alpha
            )
            model["layout_emissions"]["proximity"][layout_name][intent] = _normalize_counts(
                layout_prox_emission_counts[layout_name][intent], PROX_BUCKETS, alpha
            )
            action_support = [str(a) for a in Action.ALL_ACTIONS]
            model["layout_emissions"]["action"][layout_name][intent] = _normalize_counts(
                layout_action_counts[layout_name][intent], action_support, alpha
            )
            model["layout_emissions"]["context"][layout_name][intent] = _normalize_counts(
                layout_context_emission_counts[layout_name][intent],
                CONTEXT_BUCKETS,
                alpha,
            )
            model["layout_emissions"]["progress"][layout_name][intent] = _normalize_counts(
                layout_progress_emission_counts[layout_name][intent],
                PROGRESS_BUCKETS,
                alpha,
            )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a Bayesian intent model from cleaned human-human Overcooked data"
    )
    parser.add_argument(
        "--data-path",
        default=CLEAN_2020_HUMAN_DATA_TRAIN,
        help="Path to cleaned pickled dataframe (default: 2020 train)",
    )
    parser.add_argument(
        "--layouts",
        nargs="*",
        default=None,
        help="Optional subset of layouts",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing coefficient",
    )
    parser.add_argument(
        "--out",
        default="src/overcooked_demo/server/static/assets/agents/BayesianBeliefAI/model.pkl",
        help="Output path for pickled model",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON output path for inspection",
    )
    args = parser.parse_args()

    df = pd.read_pickle(args.data_path)
    model = train_model(df, layouts=args.layouts, alpha=args.alpha)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    if args.json_out:
        json_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(model, f, indent=2)

    print("Saved Bayesian intent model to {}".format(out_path))
    print("Stats: {}".format(model["stats"]))


if __name__ == "__main__":
    main()
