"""Prequential checks for the frequency, sequence, and context ensemble.

Every response is scored against the snapshot taken before it was observed.
Fixed contexts keep these checks focused on learned response order rather than
on a coincidental correlation with movement coordinates.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import math
import pickle
import random
import unittest

from adaptive_boss import learning
from adaptive_boss.learning import BehaviorPredictor, DIRECTIONS, EXPERT_NAMES


CONTEXT = (0.65, 0.5, 0.15, 0.0, 0.12, 1.0)
LABEL_SETS = (DIRECTIONS, ("cw", "ccw", "stay"))


class LearningEnsembleTests(unittest.TestCase):
    def feed(self, model, responses):
        predictions = []
        for actual in responses:
            snapshot = model.predict(CONTEXT)
            predictions.append(snapshot)
            model.observe(snapshot, actual)
        return predictions

    def assert_distribution(self, values, size):
        self.assertIsInstance(values, tuple)
        self.assertEqual(len(values), size)
        self.assertTrue(all(math.isfinite(value) and 0 < value < 1 for value in values))
        self.assertAlmostEqual(sum(values), 1.0, places=12)

    def assert_accuracy(self, predictions, responses, start, minimum=0.80):
        scored = list(zip(predictions[start:], responses[start:]))
        self.assertTrue(scored)
        accuracy = sum(snapshot.label == actual for snapshot, actual in scored) / len(scored)
        self.assertGreaterEqual(accuracy, minimum)

    def test_alternating_responses_are_learned_for_both_attacks(self):
        for labels in LABEL_SETS:
            with self.subTest(labels=labels):
                model = BehaviorPredictor(labels=labels)
                responses = ["cw", "ccw"] * 30
                predictions = self.feed(model, responses)
                self.assert_accuracy(predictions, responses, start=12)
                self.assertGreaterEqual(model.recent_accuracy, 0.85)

    def test_two_by_two_cycle_is_learned_for_both_attacks(self):
        for labels in LABEL_SETS:
            with self.subTest(labels=labels):
                model = BehaviorPredictor(labels=labels)
                responses = ["cw", "cw", "ccw", "ccw"] * 15
                predictions = self.feed(model, responses)
                self.assert_accuracy(predictions, responses, start=12)
                # The last response alone is ambiguous in an AABB cycle.
                self.assertTrue(any(snapshot.markov_order >= 2 for snapshot in predictions[12:]))
                self.assertGreaterEqual(model.recent_accuracy, 0.85)

    def test_three_response_context_learns_an_ambiguous_two_step_cycle(self):
        cycle = ("cw", "cw", "cw", "ccw", "cw", "ccw", "ccw", "ccw")
        # Each two-response suffix is followed by both possible responses in
        # this cycle; all three previous responses are needed to disambiguate.
        for labels in LABEL_SETS:
            for offset in range(len(cycle)):
                for swap in (False, True):
                    with self.subTest(labels=labels, offset=offset, swap=swap):
                        rotated = cycle[offset:] + cycle[:offset]
                        if swap:
                            rotated = tuple("ccw" if label == "cw" else "cw" for label in rotated)
                        responses = rotated * 10
                        model = BehaviorPredictor(labels=labels)
                        predictions = self.feed(model, responses)
                        # Score the actual combined prediction over four full
                        # cycles, after six cycles have supplied training data.
                        self.assert_accuracy(predictions, responses, start=48, minimum=0.85)

    def test_unseen_long_histories_fall_back_to_the_supported_order(self):
        markov = EXPERT_NAMES.index("markov")
        frequency = EXPERT_NAMES.index("frequency")
        for labels in LABEL_SETS:
            with self.subTest(labels=labels):
                model = BehaviorPredictor(labels=labels)
                self.feed(model, ["cw", "cw", "ccw", "ccw"] * 10)
                # Each step reintroduces one more previously learned suffix.
                # The novel stay must not make an unseen long context certain.
                for actual, expected_order in zip(["stay", "cw", "cw", "ccw"], range(4)):
                    self.feed(model, [actual])
                    snapshot = model.predict(CONTEXT)
                    self.assertEqual(snapshot.markov_order, expected_order)
                    self.assert_distribution(snapshot.expert_probabilities[markov], len(labels))
                    if expected_order == 0:
                        self.assertEqual(snapshot.expert_probabilities[markov],
                                         snapshot.expert_probabilities[frequency])

    def test_new_cycle_recovers_after_a_long_different_habit(self):
        for labels in LABEL_SETS:
            with self.subTest(labels=labels):
                model = BehaviorPredictor(labels=labels)
                self.feed(model, ["cw", "ccw"] * 48)
                responses = ["cw", "cw", "ccw", "ccw"] * 15
                predictions = self.feed(model, responses)
                self.assert_accuracy(predictions, responses, start=12)
                self.assertGreaterEqual(model.recent_accuracy, 0.85)

    def test_fixed_random_responses_do_not_produce_near_certainty(self):
        for labels in LABEL_SETS:
            with self.subTest(labels=labels):
                model = BehaviorPredictor(labels=labels)
                rng = random.Random(7319)
                responses = [rng.choice(labels) for _ in range(320)]
                predictions = self.feed(model, responses)[40:]
                mean_confidence = sum(snapshot.confidence for snapshot in predictions) / len(predictions)
                # Some short random streaks deserve a temporary preference;
                # an unpredictable stream should not sustain near certainty.
                self.assertLess(mean_confidence, 0.65 if len(labels) == 3 else 0.50)
                near_certain = sum(snapshot.confidence >= 0.90 for snapshot in predictions)
                self.assertLessEqual(near_certain / len(predictions), 0.05)

    def test_experts_and_blend_always_report_valid_distributions(self):
        self.assertEqual(EXPERT_NAMES, ("frequency", "markov", "sgd"))
        for labels in LABEL_SETS:
            with self.subTest(labels=labels):
                model = BehaviorPredictor(labels=labels)
                responses = list(labels) * 3 + ["cw"] * 12 + ["cw", "ccw"] * 10
                predictions = self.feed(model, responses)
                for index, snapshot in enumerate(predictions):
                    self.assert_distribution(snapshot.probabilities, len(labels))
                    self.assertIsInstance(snapshot.expert_weights, tuple)
                    self.assertEqual(len(snapshot.expert_weights), len(EXPERT_NAMES))
                    self.assertTrue(all(math.isfinite(value) and 0 <= value <= 1
                                        for value in snapshot.expert_weights))
                    self.assertAlmostEqual(sum(snapshot.expert_weights), 1.0, places=12)
                    if snapshot.ready:
                        self.assertTrue(all(value > 0 for value in snapshot.expert_weights))
                    self.assertIsInstance(snapshot.expert_probabilities, tuple)
                    self.assertEqual(len(snapshot.expert_probabilities), len(EXPERT_NAMES))
                    for probabilities in snapshot.expert_probabilities:
                        self.assert_distribution(probabilities, len(labels))
                    self.assertEqual(snapshot.response_history, tuple(responses[max(0, index - 3):index]))
                    self.assertGreaterEqual(snapshot.markov_order, 0)
                    self.assertLessEqual(snapshot.markov_order, len(snapshot.response_history))
                    best = max(range(len(labels)), key=snapshot.probabilities.__getitem__)
                    self.assertEqual(snapshot.label, labels[best])
                    self.assertEqual(snapshot.confidence, snapshot.probabilities[best])

    def test_predict_does_not_change_ensemble_state_and_snapshots_are_frozen(self):
        for labels in LABEL_SETS:
            with self.subTest(labels=labels):
                model = BehaviorPredictor(labels=labels)
                self.feed(model, ["cw", "cw", "ccw", "ccw"] * 8)
                before = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
                snapshot = model.predict(CONTEXT)
                for _ in range(10):
                    self.assertEqual(model.predict(CONTEXT), snapshot)
                self.assertEqual(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL), before)
                with self.assertRaises(FrozenInstanceError):
                    snapshot.response_history = ()
                with self.assertRaises(FrozenInstanceError):
                    snapshot.expert_weights = ()
                with self.assertRaises(TypeError):
                    snapshot.expert_probabilities[0][0] = 1.0
                with self.assertRaises(TypeError):
                    snapshot.response_history[0] = "stay"

    def test_delayed_observation_scores_and_trains_the_frozen_history(self):
        for labels in LABEL_SETS:
            with self.subTest(labels=labels):
                model = BehaviorPredictor(labels=labels)
                self.feed(model, ["cw", "ccw"] * 8)
                snapshot = model.predict(CONTEXT)
                original = deepcopy(snapshot)
                self.feed(model, ["stay"] * 10)
                current = model.predict(CONTEXT)
                self.assertNotEqual(snapshot.response_history, current.response_history)
                self.assertNotEqual(snapshot.label, current.label)
                self.assertNotEqual(snapshot.expert_probabilities, current.expert_probabilities)
                losses_before = list(model._expert_losses)
                counts_before = deepcopy(model._markov_counts)
                correct_before = model.correct
                baseline_before = model.baseline_correct
                actual = snapshot.label
                actual_index = labels.index(actual)

                model.observe(snapshot, actual)

                self.assertEqual(snapshot, original)
                self.assertEqual(model.correct, correct_before + 1)
                self.assertEqual(model.baseline_correct,
                                 baseline_before + int(snapshot.baseline_label == actual))
                for index, probabilities in enumerate(snapshot.expert_probabilities):
                    expected = (learning._LOSS_DECAY * losses_before[index]
                                - math.log(probabilities[actual_index]))
                    self.assertAlmostEqual(model._expert_losses[index], expected, places=12)
                # Transition learning must credit every suffix that existed
                # when this response window opened, not the later stay streak.
                for order in range(1, 4):
                    old_key = snapshot.response_history[-order:]
                    current_key = current.response_history[-order:]
                    expected = (learning._MARKOV_DECAY * counts_before[old_key][actual_index]
                                + 1.0)
                    self.assertAlmostEqual(model._markov_counts[old_key][actual_index], expected)
                    self.assertAlmostEqual(model._markov_counts[current_key][actual_index],
                                           learning._MARKOV_DECAY * counts_before[current_key][actual_index])

    def test_recent_sequence_success_changes_expert_weights(self):
        markov = EXPERT_NAMES.index("markov")
        frequency = EXPERT_NAMES.index("frequency")
        for labels in LABEL_SETS:
            with self.subTest(labels=labels):
                model = BehaviorPredictor(labels=labels)
                initial = model.predict(CONTEXT)
                responses = ["cw", "ccw"] * 30
                predictions = self.feed(model, responses)
                learned = model.predict(CONTEXT)
                self.assertNotEqual(learned.expert_weights, initial.expert_weights)
                markov_loss = 0.0
                frequency_loss = 0.0
                for snapshot, actual in zip(predictions[12:], responses[12:]):
                    actual_index = labels.index(actual)
                    markov_loss -= math.log(snapshot.expert_probabilities[markov][actual_index])
                    frequency_loss -= math.log(snapshot.expert_probabilities[frequency][actual_index])
                self.assertLess(markov_loss, frequency_loss)
                self.assertGreater(learned.expert_weights[markov], learned.expert_weights[frequency])

    def test_reset_forgets_sequence_patterns_and_expert_weights(self):
        for labels in LABEL_SETS:
            with self.subTest(labels=labels):
                model = BehaviorPredictor(seed=19, min_samples=4, labels=labels)
                initial = model.predict(CONTEXT)
                self.feed(model, ["cw", "ccw"] * 25 + ["stay"] * 6)
                learned = model.predict(CONTEXT)
                self.assertTrue(learned.response_history)
                self.assertNotEqual(learned.expert_weights, initial.expert_weights)
                model.reset()
                self.assertEqual(model.predict(CONTEXT), initial)
                self.assertEqual(model.samples, 0)
                self.assertEqual(model.history, [])
                self.assertEqual(model.counts, dict.fromkeys(labels, 0))
                self.assertEqual(model.recent_accuracy, 0.0)
                self.assertEqual(model.labels, labels)
                self.assertEqual(model.min_samples, 4)
                # Relearning also behaves like a genuinely fresh predictor.
                fresh = BehaviorPredictor(seed=19, min_samples=4, labels=labels)
                responses = ["ccw", "ccw", "cw", "cw"] * 6
                self.assertEqual(self.feed(model, responses), self.feed(fresh, responses))


if __name__ == "__main__":
    unittest.main()



