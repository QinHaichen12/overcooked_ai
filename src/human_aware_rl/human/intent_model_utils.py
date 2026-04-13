import numpy as np

NORTH = (0, -1)
SOUTH = (0, 1)
EAST = (1, 0)
WEST = (-1, 0)
ALL_DIRECTIONS = [NORTH, SOUTH, EAST, WEST]
STAY = (0, 0)
INTERACT = "interact"


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
POT_STATUS_BUCKETS = ["empty", "building", "ready_or_cooking"]
PREV_INTENT_BUCKETS = ["none"] + INTENTS


def layout_group(layout_name):
    constrained_tokens = [
        "cramped",
        "corridor",
        "forced",
        "counter_circuit",
        "you_shall_not_pass",
        "bottleneck",
    ]
    lowered = (layout_name or "").lower()
    if any(tok in lowered for tok in constrained_tokens):
        return "constrained"
    return "open"


def near_any(pos, targets, dist=1):
    for target in targets:
        if abs(pos[0] - target[0]) + abs(pos[1] - target[1]) <= dist:
            return True
    return False


def held_bucket(player):
    if not player.has_object():
        return "none"
    obj = player.get_object().name
    if obj in HELD_BUCKETS:
        return obj
    return "other"


def proximity_bucket(player, mdp):
    pos = player.position
    if near_any(pos, mdp.get_onion_dispenser_locations()):
        return "onion"
    if near_any(pos, mdp.get_tomato_dispenser_locations()):
        return "tomato"
    if near_any(pos, mdp.get_dish_dispenser_locations()):
        return "dish"
    if near_any(pos, mdp.get_pot_locations()):
        return "pot"
    if near_any(pos, mdp.get_serving_locations()):
        return "serve"
    return "none"


def snapshot_player_state(player):
    held_name = None
    if player.has_object():
        held_name = player.get_object().name
    return {
        "position": player.position,
        "orientation": player.orientation,
        "held": held_name,
    }


def infer_observed_action(prev_obs, player):
    if prev_obs is None:
        return None

    curr_pos = player.position
    curr_orient = player.orientation
    curr_held = None if not player.has_object() else player.get_object().name

    prev_pos = prev_obs["position"]
    prev_orient = prev_obs["orientation"]
    prev_held = prev_obs["held"]

    if curr_pos != prev_pos:
        step = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
        if step in ALL_DIRECTIONS:
            return step

    if curr_orient != prev_orient and curr_orient in ALL_DIRECTIONS:
        return curr_orient

    if curr_held != prev_held:
        return INTERACT

    return STAY


def action_bucket(action):
    return "none" if action is None else str(action)


def pot_status_bucket(state, mdp):
    pot_states = mdp.get_pot_states(state)
    if pot_states["ready"] or pot_states["cooking"]:
        return "ready_or_cooking"
    if (
        pot_states["1_items"]
        or pot_states["2_items"]
        or pot_states["3_items"]
    ):
        return "building"
    return "empty"


def infer_intent(player, state, mdp):
    if player.has_object():
        obj = player.get_object().name
        if obj in ["onion", "tomato"]:
            return "put_in_pot", 0.95
        if obj == "dish":
            return "pickup_soup", 0.90
        if obj == "soup":
            return "deliver_soup", 0.98

    near_onion = near_any(player.position, mdp.get_onion_dispenser_locations())
    near_tomato = near_any(player.position, mdp.get_tomato_dispenser_locations())
    near_dish = near_any(player.position, mdp.get_dish_dispenser_locations())
    near_pot = near_any(player.position, mdp.get_pot_locations())
    near_serve = near_any(player.position, mdp.get_serving_locations())
    pot_states = mdp.get_pot_states(state)

    if near_onion:
        return "get_onion", 0.70
    if near_tomato:
        return "get_tomato", 0.70
    if near_dish:
        return "get_dish", 0.70
    if near_pot and (pot_states["ready"] or pot_states["cooking"]):
        return "start_cooking", 0.65
    if near_serve:
        return "deliver_soup", 0.55

    return "get_onion", 0.40


def build_intent_feature_state(
    state,
    mdp,
    human_idx,
    prev_human_obs=None,
    prev_intent="none",
    layout_name=None,
):
    human = state.players[human_idx]
    partner = state.players[1 - human_idx]
    observed_action = infer_observed_action(prev_human_obs, human)
    return {
        "layout_group": layout_group(
            layout_name or mdp.mdp_params.get("layout_name", "")
        ),
        "held": held_bucket(human),
        "proximity": proximity_bucket(human, mdp),
        "observed_action": action_bucket(observed_action),
        "pot_status": pot_status_bucket(state, mdp),
        "prev_intent": prev_intent or "none",
        "partner_held": held_bucket(partner),
    }


def feature_state_key(feature_state):
    return "|".join(
        [
            feature_state["layout_group"],
            feature_state["held"],
            feature_state["proximity"],
            feature_state["observed_action"],
            feature_state["pot_status"],
            feature_state["prev_intent"],
            feature_state["partner_held"],
        ]
    )


def normalize(values, support, eps=1e-8):
    total = sum(max(eps, float(values.get(item, 0.0))) for item in support)
    if total <= 0:
        p = 1.0 / len(support)
        return {item: p for item in support}
    return {
        item: max(eps, float(values.get(item, 0.0))) / total for item in support
    }


def softmax_distribution(values, support, temperature=0.35):
    temp = max(1e-4, float(temperature))
    logits = np.array([float(values.get(item, 0.0)) for item in support])
    shifted = logits - np.max(logits)
    probs = np.exp(shifted / temp)
    probs /= np.sum(probs)
    return {item: float(prob) for item, prob in zip(support, probs)}


def entropy(prob_dict):
    probs = np.array(list(prob_dict.values()), dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs)).sum())


def q_values_for_state(q_table, state_key):
    row = q_table.get(state_key, {})
    return {intent: float(row.get(intent, 0.0)) for intent in INTENTS}
