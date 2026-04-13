This folder makes `GreedyHumanModel` appear in the demo agent dropdown.

The actual policy is instantiated dynamically in `server/game.py` so it can
build a planner against the current layout at runtime instead of relying on a
layout-specific `agent.pickle`.
