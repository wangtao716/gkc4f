"""Causality and learnability checks for the real incremental classifier."""

from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from adaptive_boss.learning import BehaviorPredictor, DIRECTIONS


CONTEXT = (0.65, 0.5, 0.15, 0.0, 0.12, 1.0)


class BehaviorPredictorTests(unittest.TestCase):
    def feed(self, model, direction, count):
        for _ in range(count):
            model.observe(model.predict(CONTEXT), direction)

    def test_cold_start_is_smoothed_and_becomes_ready_at_threshold(self):
        model = BehaviorPredictor(min_samples=8)
        prediction = model.predict(CONTEXT)
        self.assertFalse(prediction.ready)
        self.assertEqual(prediction.samples, 0)
        self.assertEqual(prediction.probabilities, (0.2,) * 5)
        self.feed(model, "right", 7)
        prediction = model.predict(CONTEXT)
        self.assertFalse(prediction.ready)
        self.assertTrue(all(0 < probability < 1 for probability in prediction.probabilities))
        self.assertAlmostEqual(sum(prediction.probabilities), 1.0)
        self.feed(model, "right", 1)
        self.assertTrue(model.predict(CONTEXT).ready)

    def test_learns_repeated_direction_on_cpu(self):
        model = BehaviorPredictor()
        self.feed(model, "right", 28)
        prediction = model.predict(CONTEXT)
        self.assertEqual(prediction.label, "right")
        self.assertGreater(prediction.confidence, 0.65)
        self.assertEqual(model.samples, 28)
        self.assertEqual(model.direction_samples, 28)
        self.assertGreater(model.recent_accuracy, 0.85)

    def test_recent_observations_can_reverse_an_old_habit(self):
        model = BehaviorPredictor()
        self.feed(model, "right", 50)
        old_prediction = model.predict(CONTEXT)
        self.assertEqual(old_prediction.label, "right")
        self.feed(model, "left", 16)
        new_prediction = model.predict(CONTEXT)
        self.assertEqual(new_prediction.label, "left")
        self.assertGreater(new_prediction.confidence, 0.50)
        # The slow historical majority remains right; the learned model has
        # therefore done more than merely return the lifetime majority label.
        self.assertEqual(new_prediction.baseline_label, "right")

    def test_predict_is_read_only_and_snapshot_is_frozen(self):
        model = BehaviorPredictor()
        self.feed(model, "down", 12)
        state = (model.samples, model.correct, list(model.history), dict(model.counts))
        weights = model._model.coef_.copy()
        intercept = model._model.intercept_.copy()
        first = model.predict(CONTEXT)
        for _ in range(20):
            self.assertEqual(model.predict(CONTEXT), first)
        self.assertEqual(state, (model.samples, model.correct, model.history, model.counts))
        np.testing.assert_array_equal(model._model.coef_, weights)
        np.testing.assert_array_equal(model._model.intercept_, intercept)
        with self.assertRaises(FrozenInstanceError):
            first.label = "left"

    def test_score_and_training_use_the_pre_response_snapshot(self):
        model = BehaviorPredictor()
        snapshot = model.predict(CONTEXT)
        self.assertEqual(snapshot.features[:6], CONTEXT)
        self.assertEqual(snapshot.features[6:], (0.0,) * 15)
        actual = "left" if snapshot.label != "left" else "right"
        calls = []
        real_partial_fit = model._model.partial_fit

        def checked_fit(features, labels, **kwargs):
            calls.append((features.copy(), labels.copy()))
            # The score has been evaluated, but the answer has not yet been
            # appended to player history or used to construct input features.
            self.assertEqual(model.correct, 0)
            self.assertEqual(model.direction_samples, 1)
            self.assertEqual(model.history, [])
            self.assertEqual(model.samples, 0)
            return real_partial_fit(features, labels, **kwargs)

        model._model.partial_fit = checked_fit
        model.observe(snapshot, actual)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0][0], snapshot.features)
        self.assertEqual(calls[0][1][0], DIRECTIONS.index(actual))
        self.assertEqual(model.correct, 0)
        self.assertEqual(model.history, [actual])
        self.assertEqual(snapshot.samples, 0)
        self.assertEqual(snapshot.features[6:], (0.0,) * 15)
        self.assertEqual(model.predict(CONTEXT).features[6 + DIRECTIONS.index(actual)], 1.0)

    def test_delayed_snapshot_keeps_original_features_and_baseline(self):
        model = BehaviorPredictor()
        snapshot = model.predict(CONTEXT)
        self.feed(model, "right", 12)
        old_correct = model.correct
        old_baseline_correct = model.baseline_correct
        model.observe(snapshot, snapshot.label)
        self.assertEqual(model.correct, old_correct + 1)
        self.assertEqual(
            model.baseline_correct,
            old_baseline_correct + int(snapshot.label == snapshot.baseline_label),
        )

    def test_none_is_a_real_class_but_not_a_direction_sample(self):
        model = BehaviorPredictor()
        self.feed(model, "none", 16)
        self.assertEqual(model.samples, 16)
        self.assertEqual(model.direction_samples, 0)
        self.assertEqual(model.direction_correct, 0)
        self.assertEqual(model.counts["none"], 16)
        self.assertEqual(model.predict(CONTEXT).label, "none")

    def test_reset_removes_model_history_and_statistics(self):
        model = BehaviorPredictor()
        initial = model.predict(CONTEXT)
        self.feed(model, "up", 20)
        model.reset()
        self.assertEqual(model.predict(CONTEXT), initial)
        self.assertEqual(model.samples, 0)
        self.assertEqual(model.correct, 0)
        self.assertEqual(model.baseline_correct, 0)
        self.assertEqual(model.direction_samples, 0)
        self.assertEqual(model.direction_correct, 0)
        self.assertEqual(model.history, [])
        self.assertEqual(model.counts, dict.fromkeys(DIRECTIONS, 0))
        self.assertEqual(model.recent_accuracy, 0.0)

    def test_invalid_inputs_do_not_modify_statistics(self):
        model = BehaviorPredictor()
        for context in ((0.0,), (0.0,) * 5 + (float("nan"),)):
            with self.assertRaises(ValueError):
                model.predict(context)
        with self.assertRaises(ValueError):
            model.observe(model.predict(CONTEXT), "diagonal")
        self.assertEqual(model.samples, 0)
        self.assertEqual(model.history, [])


if __name__ == "__main__":
    unittest.main()
