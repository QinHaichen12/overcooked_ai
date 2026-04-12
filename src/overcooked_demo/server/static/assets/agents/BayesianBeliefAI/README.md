BayesianBeliefAI

This directory enables `BayesianBeliefAI` to appear in the demo agent dropdown.

The implementation is provided directly in server runtime code at:
- src/overcooked_demo/server/game.py

No serialized `agent.pickle` is required for this agent.

Optional data-calibrated model
- Train a model from cleaned human-human data:
	- `python src/human_aware_rl/human/train_bayesian_intent_model.py --out src/overcooked_demo/server/static/assets/agents/BayesianBeliefAI/model.pkl`
- The demo agent auto-loads `model.pkl` from this directory when available.
- You can override with env var:
	- `OVERCOOKED_BAYESIAN_MODEL=/abs/path/to/model.pkl`
