import argparse
import ast
import json
import os
import pickle
import random
from collections import defaultdict

import pandas as pd

from human_aware_rl.human.intent_model_utils import (
    HELD_BUCKETS,
    INTENTS,
    POT_STATUS_BUCKETS,
    PROX_BUCKETS,
    PREV_INTENT_BUCKETS,
    feature_state_key,
    layout_group,
    q_values_for_state,
    softmax_distribution,
)
from human_aware_rl.static import CLEAN_2020_HUMAN_DATA_TRAIN


def _layout_positions(layout):
    terrain = layout
    if isinstance(layout, str):
        try:
            terrain = json.loads(layout)
        except json.JSONDecodeError:
            terrain = ast.literal_eval(layout)
    positions = defaultdict(list)
    for y, row in enumerate(terrain):
        for x, token in enumerate(row):
            positions[token].append((x, y))
    return positions


def _near_any(pos, targets, dist=1):
    for target in targets:
        if abs(pos[0] - target[0]) + abs(pos[1] - target[1]) <= dist:
            return True
    return False


def _held_bucket_raw(player):
    held = player.get("held_object")
    if not held:
        return "none"
    name = held.get("name", "other")
    if name in HELD_BUCKETS:
        return name
    return "other"


def _snapshot_player_state_raw(player):
    held = player.get("held_object")
    held_name = None if not held else held.get("name")
    return {
        "position": tuple(player["position"]),
        "orientation": tuple(player["orientation"]),
        "held": held_name,
    }


def _infer_observed_action_raw(prev_obs, player):
    if prev_obs is None:
        return None

    curr_pos = tuple(player["position"])
    curr_orient = tuple(player["orientation"])
    held = player.get("held_object")
    curr_held = None if not held else held.get("name")

    prev_pos = prev_obs["position"]
    prev_orient = prev_obs["orientation"]
    prev_held = prev_obs["held"]

    if curr_pos != prev_pos:
        step = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
        if step in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            return step

    if curr_orient != prev_orient and curr_orient in [
        (0, -1),
        (0, 1),
        (1, 0),
        (-1, 0),
    ]:
        return curr_orient

    if curr_held != prev_held:
        return "interact"

    return (0, 0)


def _proximity_bucket_raw(player, layout_positions):
    pos = tuple(player["position"])
    if _near_any(pos, layout_positions.get("O", [])):
        return "onion"
    if _near_any(pos, layout_positions.get("T", [])):
        return "tomato"
    if _near_any(pos, layout_positions.get("D", [])):
        return "dish"
    if _near_any(pos, layout_positions.get("P", [])):
        return "pot"
    if _near_any(pos, layout_positions.get("S", [])):
        return "serve"
    return "none"


def _pot_status_bucket_raw(state, layout_positions):
    pot_positions = {tuple(pos) for pos in layout_positions.get("P", [])}
    building = False
    for obj in state.get("objects", []):
        if obj.get("name") != "soup":
            continue
        if tuple(obj.get("position", ())) not in pot_positions:
            continue
        if obj.get("is_ready") or obj.get("is_cooking"):
            return "ready_or_cooking"
        building = True
    return "building" if building else "empty"


def _infer_intent_raw(player, state, layout_positions):
    held = player.get("held_object")
    if held:
        obj = held.get("name")
        if obj in ["onion", "tomato"]:
            return "put_in_pot", 0.95
        if obj == "dish":
            return "pickup_soup", 0.90
        if obj == "soup":
            return "deliver_soup", 0.98

    pos = tuple(player["position"])
    near_onion = _near_any(pos, layout_positions.get("O", []))
    near_tomato = _near_any(pos, layout_positions.get("T", []))
    near_dish = _near_any(pos, layout_positions.get("D", []))
    near_pot = _near_any(pos, layout_positions.get("P", []))
    near_serve = _near_any(pos, layout_positions.get("S", []))
    pot_status = _pot_status_bucket_raw(state, layout_positions)

    if near_onion:
        return "get_onion", 0.70
    if near_tomato:
        return "get_tomato", 0.70
    if near_dish:
        return "get_dish", 0.70
    if near_pot and pot_status == "ready_or_cooking":
        return "start_cooking", 0.65
    if near_serve:
        return "deliver_soup", 0.55

    return "get_onion", 0.40


def _build_intent_feature_state_raw(
    state,
    layout_name,
    layout_positions,
    player_idx,
    prev_human_obs=None,
    prev_intent="none",
):
    player = state["players"][player_idx]
    partner = state["players"][1 - player_idx]
    observed_action = _infer_observed_action_raw(prev_human_obs, player)
    return {
        "layout_group": layout_group(layout_name),
        "held": _held_bucket_raw(player),
        "proximity": _proximity_bucket_raw(player, layout_positions),
        "observed_action": "none" if observed_action is None else str(observed_action),
        "pot_status": _pot_status_bucket_raw(state, layout_positions),
        "prev_intent": prev_intent or "none",
        "partner_held": _held_bucket_raw(partner),
    }


def _reward_for_prediction(
    predicted_intent,
    labeled_intent,
    confidence,
    transition_bonus=0.05,
    mismatch_penalty=0.30,
    next_label=None,
):
    reward = float(confidence) if predicted_intent == labeled_intent else (
        -float(confidence) * float(mismatch_penalty)
    )
    if next_label is not None and predicted_intent == next_label:
        reward += float(transition_bonus)
    return reward


def build_training_sequences(df, layouts=None):
    if layouts:
        df = df[df["layout_name"].isin(layouts)]

    required_cols = ["trial_id", "layout_name", "cur_gameloop", "state", "joint_action"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {}".format(missing))

    df = df.sort_values(["trial_id", "cur_gameloop"])
    layout_cache = {}
    prev_obs = {}
    prev_intents = {}
    sequences = defaultdict(list)

    for _, row in df.iterrows():
        layout_name = row["layout_name"]
        if layout_name not in layout_cache:
            layout_cache[layout_name] = _layout_positions(row["layout"])
        layout_positions = layout_cache[layout_name]
        state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]

        for player_idx in [0, 1]:
            seq_key = (row["trial_id"], player_idx)
            player = state["players"][player_idx]
            label, confidence = _infer_intent_raw(player, state, layout_positions)
            feature_state = _build_intent_feature_state_raw(
                state,
                layout_name=layout_name,
                layout_positions=layout_positions,
                player_idx=player_idx,
                prev_human_obs=prev_obs.get(seq_key),
                prev_intent=prev_intents.get(seq_key, "none"),
            )
            sequences[seq_key].append(
                {
                    "state_key": feature_state_key(feature_state),
                    "feature_state": dict(feature_state),
                    "label": label,
                    "confidence": float(confidence),
                }
            )
            prev_obs[seq_key] = _snapshot_player_state_raw(player)
            prev_intents[seq_key] = label

    return sequences


def train_model(
    df,
    layouts=None,
    epochs=15,
    alpha=0.25,
    gamma=0.90,
    epsilon=0.15,
    epsilon_decay=0.97,
    epsilon_min=0.02,
    transition_bonus=0.05,
    mismatch_penalty=0.30,
    seed=0,
):
    epsilon_start = float(epsilon)
    rng = random.Random(seed)
    sequences = build_training_sequences(df, layouts=layouts)
    q_table = defaultdict(lambda: defaultdict(float))
    visit_counts = defaultdict(int)
    reward_trace = []
    accuracy_trace = []

    items = list(sequences.items())
    if not items:
        raise ValueError("No training sequences available for the requested layouts")

    for _ in range(int(epochs)):
        rng.shuffle(items)
        total_reward = 0.0
        total_correct = 0
        total_steps = 0

        for _, sequence in items:
            for step_idx, step in enumerate(sequence):
                state_key = step["state_key"]
                label = step["label"]
                confidence = step["confidence"]
                next_state_key = None
                next_label = None
                if step_idx + 1 < len(sequence):
                    next_state_key = sequence[step_idx + 1]["state_key"]
                    next_label = sequence[step_idx + 1]["label"]

                if rng.random() < epsilon:
                    chosen_intent = rng.choice(INTENTS)
                else:
                    q_values = q_values_for_state(q_table, state_key)
                    chosen_intent = max(q_values, key=q_values.get)

                reward = _reward_for_prediction(
                    chosen_intent,
                    label,
                    confidence,
                    transition_bonus=transition_bonus,
                    mismatch_penalty=mismatch_penalty,
                    next_label=next_label,
                )
                next_best = 0.0
                if next_state_key is not None:
                    next_best = max(q_values_for_state(q_table, next_state_key).values())

                old_value = q_table[state_key][chosen_intent]
                target = reward + float(gamma) * next_best
                q_table[state_key][chosen_intent] = old_value + float(alpha) * (
                    target - old_value
                )

                visit_counts[state_key] += 1
                total_reward += reward
                total_correct += int(chosen_intent == label)
                total_steps += 1

        reward_trace.append(total_reward / max(1, total_steps))
        accuracy_trace.append(total_correct / float(max(1, total_steps)))
        epsilon = max(float(epsilon_min), float(epsilon) * float(epsilon_decay))

    serializable_q = {}
    for state_key in sorted(q_table):
        serializable_q[state_key] = {
            intent: float(q_table[state_key].get(intent, 0.0)) for intent in INTENTS
        }

    model = {
        "version": 1,
        "model_type": "tabular_q_intent",
        "intents": list(INTENTS),
        "held_buckets": list(HELD_BUCKETS),
        "proximity_buckets": list(PROX_BUCKETS),
        "pot_status_buckets": list(POT_STATUS_BUCKETS),
        "prev_intent_buckets": list(PREV_INTENT_BUCKETS),
        "state_features": [
            "layout_group",
            "held",
            "proximity",
            "observed_action",
            "pot_status",
            "prev_intent",
            "partner_held",
        ],
        "q_table": serializable_q,
        "policy_table": {
            state_key: softmax_distribution(row, INTENTS)
            for state_key, row in serializable_q.items()
        },
        "visit_counts": {k: int(v) for k, v in visit_counts.items()},
        "stats": {
            "timesteps": int(len(df)),
            "num_sequences": int(len(sequences)),
            "num_states": int(len(serializable_q)),
            "layouts": sorted(list(set(df["layout_name"]))),
            "num_trials": int(df["trial_id"].nunique()),
            "final_avg_reward": float(reward_trace[-1]),
            "final_accuracy": float(accuracy_trace[-1]),
        },
        "training": {
            "epochs": int(epochs),
            "alpha": float(alpha),
            "gamma": float(gamma),
            "epsilon_start": float(epsilon_start),
            "epsilon_min": float(epsilon_min),
            "epsilon_decay": float(epsilon_decay),
            "transition_bonus": float(transition_bonus),
            "mismatch_penalty": float(mismatch_penalty),
            "reward_trace": [float(x) for x in reward_trace],
            "accuracy_trace": [float(x) for x in accuracy_trace],
            "seed": int(seed),
        },
    }
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a tabular RL intent model from cleaned human-human Overcooked data"
    )
    parser.add_argument(
        "--data-path",
        default=CLEAN_2020_HUMAN_DATA_TRAIN,
        help="Path to cleaned pickled dataframe (default: 2020 train)",
    )
    parser.add_argument(
        "--extra-data-paths",
        nargs="*",
        default=None,
        help="Optional additional cleaned dataframe pickles to concatenate",
    )
    parser.add_argument(
        "--layouts",
        nargs="*",
        default=None,
        help="Optional subset of layouts",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--epsilon-decay", type=float, default=0.97)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--transition-bonus", type=float, default=0.05)
    parser.add_argument("--mismatch-penalty", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="src/overcooked_demo/server/static/assets/agents/RLIntentModelAI/model.pkl",
        help="Output path for pickled model",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON output path for inspection",
    )
    args = parser.parse_args()

    data_frames = [pd.read_pickle(args.data_path)]
    for extra_path in args.extra_data_paths or []:
        data_frames.append(pd.read_pickle(extra_path))
    df = pd.concat(data_frames, ignore_index=True)
    model = train_model(
        df,
        layouts=args.layouts,
        epochs=args.epochs,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        transition_bonus=args.transition_bonus,
        mismatch_penalty=args.mismatch_penalty,
        seed=args.seed,
    )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    if args.json_out:
        json_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(model, f, indent=2)

    print("Saved RL intent model to {}".format(out_path))
    print("Stats: {}".format(model["stats"]))


if __name__ == "__main__":
    main()
