"""Small CPU-only online model for predicting a player's next response.

Call ``predict`` when an attack locks, then pass that *same* immutable
snapshot to ``observe`` when the response window closes.  A response must be a
real recorded movement (or ``stay``), never the model's own prediction.
This keeps evaluation and training causal: the answer is not a feature.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Sequence

import numpy as np
from sklearn.linear_model import SGDClassifier


DIRECTIONS = ("cw", "ccw", "in", "out", "stay")
CONTEXT_SIZE = 6
_HISTORY_SIZE = 3
_RECENCY_DECAY = 0.72
_PRIOR_SMOOTHING = 0.15
EXPERT_NAMES = ("frequency", "markov", "sgd")
# Longer contexts recur less often than individual directions, so retain
# their evidence longer than the fast-changing frequency expert does.
_MARKOV_DECAY = 0.94
_BACKOFF_STRENGTH = 2.0
_LOSS_DECAY = 0.85
_EXPERT_LEARNING_RATE = 0.80
_MIN_EXPERT_WEIGHT = 0.03


@dataclass(frozen=True, slots=True)
class PredictionSnapshot:
    """Everything known at prediction time, preserved for honest evaluation."""

    features: tuple[float, ...]
    probabilities: tuple[float, ...]
    label: str
    confidence: float
    samples: int
    ready: bool
    baseline_label: str
    response_history: tuple[str, ...]
    expert_probabilities: tuple[tuple[float, ...], ...]
    expert_weights: tuple[float, ...]
    markov_order: int


class BehaviorPredictor:
    """Blend recent frequency, variable-order transitions, and online SGD.

    ``context`` contains normalized player x/y, player-minus-boss x/y,
    distance divided by arena diagonal, and whether the player attacked
    recently or is charging.  The remaining features are three one-hot encoded
    *previous* responses, newest first; absent history slots are all zeros.

    The engine owns independent instances for bombs and sweeps. Bombs learn
    clockwise/counterclockwise, toward/away, or stationary full responses.
    Sweeps learn net tangential movement from lock through beam completion
    (cw/ccw), including later reversals and rescue dashes. The stay label means
    no clear net side, including radial movement or cancelling lateral travel.
    Both use real movement, never input buttons.

    Orders one through three interpolate transition counts with the shorter
    context's distribution; unseen histories fall back to recent frequency.
    Transition counts decay by 0.94 on every completed skill response. Expert
    weights follow discounted, pre-training log loss, with a small floor so
    an expert can recover after the player changes habits. Frequency alone
    supplies the warmup output; all experts are scored from their snapshots.
    Three observations enable adaptation by default.
    Lifetime counts remain intact as an explicitly slower majority baseline.

    ``correct`` and ``baseline_correct`` cover every response, including
    ``stay``.  ``direction_samples`` and ``direction_correct`` exclude
    stay responses.  ``recent_accuracy`` covers the latest 20 responses.
    These are predictions made *before* their corresponding training update.
    """

    def __init__(self, seed: int = 7, min_samples: int = 3,
                 labels: Sequence[str] = DIRECTIONS) -> None:
        if not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        self.labels = tuple(labels)
        if (len(self.labels) < 2 or len(set(self.labels)) != len(self.labels)
                or any(label not in DIRECTIONS for label in self.labels)):
            raise ValueError("labels must be distinct supported directions, with at least two classes")
        self.feature_size = CONTEXT_SIZE + _HISTORY_SIZE * len(self.labels)
        self.seed = seed
        self.min_samples = min_samples
        self.reset()

    def reset(self) -> None:
        """Forget learned weights, player history, and all score counters."""
        self._model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0005,
            learning_rate="constant",
            eta0=0.18,
            fit_intercept=True,
            average=False,
            random_state=self.seed,
        )
        self.samples = 0
        self.correct = 0
        self.direction_samples = 0
        self.direction_correct = 0
        self.baseline_correct = 0
        self.history: list[str] = []
        self.counts: dict[str, int] = dict.fromkeys(self.labels, 0)
        self._recent_counts: dict[str, float] = dict.fromkeys(self.labels, 0.0)
        self._recent_outcomes: deque[bool] = deque(maxlen=20)
        self._markov_counts: dict[tuple[str, ...], list[float]] = {}
        self._expert_losses = [0.0] * len(EXPERT_NAMES)

    @property
    def recent_accuracy(self) -> float:
        if not self._recent_outcomes:
            return 0.0
        return sum(self._recent_outcomes) / len(self._recent_outcomes)

    def _features(self, context: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in context)
        if len(values) != CONTEXT_SIZE:
            raise ValueError(f"context must contain exactly {CONTEXT_SIZE} numeric values")
        if not all(isfinite(value) for value in values):
            raise ValueError("context values must be finite")
        previous = list(reversed(self.history[-_HISTORY_SIZE:]))
        history_features = tuple(
            float(index < len(previous) and previous[index] == direction)
            for index in range(_HISTORY_SIZE)
            for direction in self.labels
        )
        return values + history_features

    def _prior(self) -> tuple[float, ...]:
        if not self.samples:
            return (1.0 / len(self.labels),) * len(self.labels)
        denominator = sum(self._recent_counts.values()) + _PRIOR_SMOOTHING * len(self.labels)
        return tuple(
            (self._recent_counts[direction] + _PRIOR_SMOOTHING) / denominator
            for direction in self.labels
        )

    def _markov_prediction(self, history: tuple[str, ...],
                           prior: tuple[float, ...]) -> tuple[tuple[float, ...], int]:
        """Use longer contexts only in proportion to their observed support."""
        probabilities = prior
        used_order = 0
        for order in range(1, len(history) + 1):
            counts = self._markov_counts.get(history[-order:])
            if counts is None:
                continue
            support = sum(counts)
            probabilities = tuple(
                (count + _BACKOFF_STRENGTH * lower) / (support + _BACKOFF_STRENGTH)
                for count, lower in zip(counts, probabilities)
            )
            used_order = order
        return probabilities, used_order

    def _weights(self) -> tuple[float, ...]:
        """Discounted exponential weighting, leaving every expert a foothold."""
        best_loss = min(self._expert_losses)
        scores = [exp(-_EXPERT_LEARNING_RATE * (loss - best_loss))
                  for loss in self._expert_losses]
        total = sum(scores)
        remainder = 1.0 - len(EXPERT_NAMES) * _MIN_EXPERT_WEIGHT
        return tuple(_MIN_EXPERT_WEIGHT + remainder * score / total for score in scores)

    def _sgd_prediction(self, features: tuple[float, ...],
                        prior: tuple[float, ...]) -> tuple[float, ...]:
        if not self.samples:
            return prior
        raw = self._model.predict_proba(np.asarray([features], dtype=np.float64))[0]
        total = float(raw.sum())
        if not np.isfinite(raw).all() or total <= 0:
            return prior
        return tuple(float(value / total * 0.975 + 0.025 / len(self.labels)) for value in raw)

    def predict(self, context: Sequence[float]) -> PredictionSnapshot:
        """Freeze a prediction without training or changing any player state."""
        features = self._features(context)
        ready = self.samples >= self.min_samples
        baseline_label = max(self.labels, key=self.counts.__getitem__)
        prior = self._prior()
        history = tuple(self.history[-_HISTORY_SIZE:])
        markov, order = self._markov_prediction(history, prior)
        experts = (prior, markov, self._sgd_prediction(features, prior))
        weights = self._weights() if ready else (1.0, 0.0, 0.0)
        probabilities = tuple(sum(weight * expert[index] for weight, expert in zip(weights, experts))
                              for index in range(len(self.labels)))
        best = max(range(len(self.labels)), key=probabilities.__getitem__)
        return PredictionSnapshot(
            features=features,
            probabilities=probabilities,
            label=self.labels[best],
            confidence=probabilities[best],
            samples=self.samples,
            ready=ready,
            baseline_label=baseline_label,
            response_history=history,
            expert_probabilities=experts,
            expert_weights=weights,
            markov_order=order,
        )

    def observe(self, snapshot: PredictionSnapshot, actual: str) -> None:
        """Score a frozen prediction, learn its true response, then record it.

        A delayed snapshot is valid: its frozen features, labels and baseline
        are used as-is instead of recomputing them with newer information.
        The caller should submit each completed response window exactly once.
        """
        if actual not in self.labels:
            raise ValueError(f"actual must be one of {self.labels!r}")
        if not isinstance(snapshot, PredictionSnapshot):
            raise TypeError("snapshot must be a PredictionSnapshot")
        if len(snapshot.features) != self.feature_size or not all(
            isfinite(value) for value in snapshot.features
        ):
            raise ValueError(f"snapshot must contain {self.feature_size} finite features")
        if snapshot.label not in self.labels or snapshot.baseline_label not in self.labels:
            raise ValueError("snapshot labels must be valid directions")
        if (len(snapshot.response_history) > _HISTORY_SIZE or
                any(label not in self.labels for label in snapshot.response_history)):
            raise ValueError("snapshot history must contain supported directions")
        if len(snapshot.expert_probabilities) != len(EXPERT_NAMES):
            raise ValueError("snapshot must contain every expert's frozen prediction")
        for probabilities in snapshot.expert_probabilities:
            if (len(probabilities) != len(self.labels) or
                    any(not isfinite(value) or value <= 0 for value in probabilities) or
                    abs(sum(probabilities) - 1.0) > 1e-8):
                raise ValueError("expert probabilities must be positive and normalized")

        # Score before training: otherwise the current answer would leak into
        # the displayed accuracy.  Likewise, do not rebuild snapshot.features.
        was_correct = snapshot.label == actual
        self.correct += int(was_correct)
        self.baseline_correct += int(snapshot.baseline_label == actual)
        self._recent_outcomes.append(was_correct)
        if actual != "stay":
            self.direction_samples += 1
            self.direction_correct += int(was_correct)

        self._model.partial_fit(
            np.asarray([snapshot.features], dtype=np.float64),
            np.asarray([self.labels.index(actual)], dtype=np.int64),
            classes=np.arange(len(self.labels), dtype=np.int64),
        )

        # Use the predictions and history frozen at lock even if this
        # observation arrives after other responses have been completed.
        actual_index = self.labels.index(actual)
        for index, probabilities in enumerate(snapshot.expert_probabilities):
            self._expert_losses[index] = (_LOSS_DECAY * self._expert_losses[index]
                                           - log(max(1e-12, probabilities[actual_index])))
        for key in list(self._markov_counts):
            counts = self._markov_counts[key]
            for index in range(len(counts)):
                counts[index] *= _MARKOV_DECAY
            if sum(counts) < 1e-4:
                del self._markov_counts[key]
        for order in range(1, len(snapshot.response_history) + 1):
            key = snapshot.response_history[-order:]
            counts = self._markov_counts.setdefault(key, [0.0] * len(self.labels))
            counts[actual_index] += 1.0

        self.samples += 1
        self.counts[actual] += 1
        for direction in self.labels:
            self._recent_counts[direction] *= _RECENCY_DECAY
        self._recent_counts[actual] += 1.0
        self.history.append(actual)
