"""Causality and learnability checks for the real incremental classifier."""

from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from adaptive_boss.learning import BehaviorPredictor, CONTEXT_SIZE, DIRECTIONS


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
        self.feed(model, "cw", 7)
        prediction = model.predict(CONTEXT)
        self.assertFalse(prediction.ready)
        self.assertTrue(all(0 < probability < 1 for probability in prediction.probabilities))
        self.assertAlmostEqual(sum(prediction.probabilities), 1.0)
        self.feed(model, "cw", 1)
        self.assertTrue(model.predict(CONTEXT).ready)

    def test_three_repeated_responses_enable_a_confident_prediction(self):
        for direction in DIRECTIONS:
            with self.subTest(direction=direction):
                model = BehaviorPredictor()
                self.feed(model, direction, 2)
                self.assertFalse(model.predict(CONTEXT).ready)
                self.feed(model, direction, 1)
                prediction = model.predict(CONTEXT)
                self.assertTrue(prediction.ready)
                self.assertEqual(prediction.label, direction)
                self.assertGreater(prediction.confidence, 0.60)
                self.assertEqual(model.samples, 3)
                self.assertEqual(model.direction_samples, 0 if direction == "stay" else 3)
                self.assertAlmostEqual(sum(prediction.probabilities), 1.0)

    def test_repeated_direction_remains_learned_without_absolute_certainty(self):
        model = BehaviorPredictor()
        self.feed(model, "cw", 28)
        prediction = model.predict(CONTEXT)
        self.assertEqual(prediction.label, "cw")
        self.assertGreater(prediction.confidence, 0.65)
        self.assertTrue(all(0 < value < 1 for value in prediction.probabilities))
        self.assertGreater(model.recent_accuracy, 0.85)

    def test_recent_observations_can_reverse_an_old_habit(self):
        model = BehaviorPredictor()
        self.feed(model, "cw", 50)
        old_prediction = model.predict(CONTEXT)
        self.assertEqual(old_prediction.label, "cw")
        self.feed(model, "ccw", 3)
        self.assertEqual(model.predict(CONTEXT).label, "ccw")
        self.feed(model, "ccw", 2)
        new_prediction = model.predict(CONTEXT)
        self.assertEqual(new_prediction.label, "ccw")
        self.assertGreater(new_prediction.confidence, 0.60)
        # The lifetime majority must not be confused with the adaptive model.
        self.assertEqual(new_prediction.baseline_label, "cw")
        self.assertEqual(model.counts["cw"], 50)

    def test_changed_habit_adapts_after_a_long_stationary_history(self):
        model = BehaviorPredictor()
        self.feed(model, "stay", 150)
        self.feed(model, "out", 5)
        prediction = model.predict(CONTEXT)
        self.assertEqual(prediction.label, "out")
        self.assertGreater(prediction.confidence, 0.60)
        self.assertEqual(prediction.baseline_label, "stay")

    def test_predict_is_read_only_and_snapshot_is_frozen(self):
        model = BehaviorPredictor()
        self.feed(model, "in", 12)
        state = (model.samples, model.correct, list(model.history), dict(model.counts))
        weights = model._model.coef_.copy()
        intercept = model._model.intercept_.copy()
        recent_counts = dict(model._recent_counts)
        first = model.predict(CONTEXT)
        for _ in range(20):
            self.assertEqual(model.predict(CONTEXT), first)
        self.assertEqual(state, (model.samples, model.correct, model.history, model.counts))
        np.testing.assert_array_equal(model._model.coef_, weights)
        np.testing.assert_array_equal(model._model.intercept_, intercept)
        self.assertEqual(model._recent_counts, recent_counts)
        with self.assertRaises(FrozenInstanceError):
            first.label = "ccw"

    def test_score_and_training_use_the_pre_response_snapshot(self):
        model = BehaviorPredictor()
        snapshot = model.predict(CONTEXT)
        self.assertEqual(snapshot.features[:CONTEXT_SIZE], CONTEXT)
        self.assertEqual(snapshot.features[CONTEXT_SIZE:], (0.0,) * 15)
        actual = "ccw" if snapshot.label != "ccw" else "cw"
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
            self.assertEqual(model._recent_counts, dict.fromkeys(DIRECTIONS, 0.0))
            return real_partial_fit(features, labels, **kwargs)

        model._model.partial_fit = checked_fit
        model.observe(snapshot, actual)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0][0], snapshot.features)
        self.assertEqual(calls[0][1][0], DIRECTIONS.index(actual))
        self.assertEqual(model.correct, 0)
        self.assertEqual(model.history, [actual])
        self.assertEqual(snapshot.samples, 0)
        self.assertEqual(snapshot.features[CONTEXT_SIZE:], (0.0,) * 15)
        self.assertEqual(
            model.predict(CONTEXT).features[CONTEXT_SIZE + DIRECTIONS.index(actual)], 1.0,
        )

    def test_delayed_snapshot_keeps_original_features_and_baseline(self):
        model = BehaviorPredictor()
        snapshot = model.predict(CONTEXT)
        self.feed(model, "out", 12)
        old_correct = model.correct
        old_baseline_correct = model.baseline_correct
        calls = []
        real_partial_fit = model._model.partial_fit

        def capture_fit(features, labels, **kwargs):
            calls.append(features.copy())
            return real_partial_fit(features, labels, **kwargs)

        model._model.partial_fit = capture_fit
        model.observe(snapshot, snapshot.label)
        self.assertEqual(model.correct, old_correct + 1)
        self.assertEqual(
            model.baseline_correct,
            old_baseline_correct + int(snapshot.label == snapshot.baseline_label),
        )
        np.testing.assert_array_equal(calls[0][0], snapshot.features)
        self.assertEqual(snapshot.features[CONTEXT_SIZE:], (0.0,) * 15)

    def test_stay_is_a_real_class_but_not_a_direction_sample(self):
        model = BehaviorPredictor()
        self.feed(model, "stay", 16)
        self.assertEqual(model.samples, 16)
        self.assertEqual(model.direction_samples, 0)
        self.assertEqual(model.direction_correct, 0)
        self.assertEqual(model.counts["stay"], 16)
        self.assertEqual(model.predict(CONTEXT).label, "stay")

    def test_reset_removes_model_history_and_statistics(self):
        model = BehaviorPredictor()
        initial = model.predict(CONTEXT)
        self.feed(model, "in", 20)
        model.reset()
        self.assertEqual(model.predict(CONTEXT), initial)
        self.assertEqual(model.samples, 0)
        self.assertEqual(model.correct, 0)
        self.assertEqual(model.baseline_correct, 0)
        self.assertEqual(model.direction_samples, 0)
        self.assertEqual(model.direction_correct, 0)
        self.assertEqual(model.history, [])
        self.assertEqual(model.counts, dict.fromkeys(DIRECTIONS, 0))
        self.assertEqual(model._recent_counts, dict.fromkeys(DIRECTIONS, 0.0))
        self.assertEqual(model.recent_accuracy, 0.0)

    def test_invalid_inputs_do_not_modify_statistics(self):
        model = BehaviorPredictor()
        for context in ((0.0,), (0.0,) * (CONTEXT_SIZE - 1) + (float("nan"),)):
            with self.assertRaises(ValueError):
                model.predict(context)
        with self.assertRaises(ValueError):
            model.observe(model.predict(CONTEXT), "diagonal")
        self.assertEqual(model.samples, 0)
        self.assertEqual(model.history, [])

    def test_old_dash_only_labels_are_rejected(self):
        model = BehaviorPredictor()
        for label in ("up", "down", "left", "right", "none"):
            with self.assertRaises(ValueError):
                model.observe(model.predict(CONTEXT), label)
        self.assertEqual(model.samples, 0)


if __name__ == "__main__":
    unittest.main()
