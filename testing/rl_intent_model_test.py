import unittest

import pandas as pd

from human_aware_rl.human.train_rl_intent_model import (
    build_training_sequences,
    train_model,
)
from human_aware_rl.static import DUMMY_2020_CLEAN_HUMAN_DATA_PATH


class TestRLIntentModelTraining(unittest.TestCase):
    def test_build_sequences_and_train_model(self):
        df = pd.read_pickle(DUMMY_2020_CLEAN_HUMAN_DATA_PATH)
        sequences = build_training_sequences(
            df, layouts=["inverse_marshmallow_experiment"]
        )
        self.assertGreater(len(sequences), 0)

        model = train_model(
            df,
            layouts=["inverse_marshmallow_experiment"],
            epochs=2,
            seed=7,
        )
        self.assertEqual(model["model_type"], "tabular_q_intent")
        self.assertGreater(model["stats"]["num_sequences"], 0)
        self.assertGreater(model["stats"]["num_states"], 0)
        self.assertIn("q_table", model)


if __name__ == "__main__":
    unittest.main()
