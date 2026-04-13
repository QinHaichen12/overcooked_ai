RLIntentModelAI
===============

This directory enables `RLIntentModelAI` to appear in the demo agent dropdown.

Expected contents
-----------------

- `model.pkl`: Pickled tabular RL intent model produced by
  `src/human_aware_rl/human/train_rl_intent_model.py`

Train / refresh the model
-------------------------

From the repo root:

`python src/human_aware_rl/human/train_rl_intent_model.py --out src/overcooked_demo/server/static/assets/agents/RLIntentModelAI/model.pkl`

Recommended layouts for best behavior
-------------------------------------

The default model is most reliable on the coordination layouts used for training,
especially `cramped_room`, `coordination_ring`, `forced_coordination`, and
`counter_circuit_o_1order`.
