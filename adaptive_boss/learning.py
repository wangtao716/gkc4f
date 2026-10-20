"""Small CPU-only online model for predicting a player's next response.

Call ``predict`` when a boss telegraph begins, then pass that *same* immutable
snapshot to ``observe`` when the response window closes.  A response must be a
real recorded movement (or ``stay``), never the model's own prediction.
This keeps evaluation and training causal: the answer is not a feature.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np
from sklearn.linear_model import SGDClassifier


DIRECTIONS = ("cw", "ccw", "in", "out", "stay")
CONTEXT_SIZE = 6
_HISTORY_SIZE = 3
_FEATURE_SIZE = CONTEXT_SIZE + _HISTORY_SIZE * len(DIRECTIONS)
_RECENCY_DECAY = 0.72
_PRIOR_SMOOTHING = 0.15
_RECENT_WEIGHT = 0.75


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


class BehaviorPredictor:
    """Learn boss-relative movement from recent responses and online SGD.

    ``context`` contains normalized player x/y, player-minus-boss x/y,
    distance divided by arena diagonal, and whether the player attacked
    recently or is charging.  The remaining features are three one-hot encoded
    *previous* responses, newest first; absent history slots are all zeros.

    Responses are clockwise/counterclockwise around the boss, toward/away
    from the boss, or stationary.  The engine measures actual displacement
    across a response window, including ordinary movement and successful
    dashes.  This learner never infers a label from input buttons.

    Each observation decays older response counts by 0.72 and adds one SGD
    update with a constant learning rate.  Smoothed recent frequencies carry
    75% of a ready prediction; the context/history model carries the rest.
    Three observations enable adaptation by default, while a new habit can
    outweigh even a long prior habit after a few completed response windows.
    Lifetime counts remain intact as an explicitly slower majority baseline.

    ``correct`` and ``baseline_correct`` cover every response, including
    ``stay``.  ``direction_samples`` and ``direction_correct`` exclude
    stationary responses.  ``recent_accuracy`` covers the latest 20 responses.
    These are predictions made *before* their corresponding training update.
    """

    def __init__(self, seed: int = 7, min_samples: int = 3) -> None:
        if not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
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
        self.counts: dict[str, int] = dict.fromkeys(DIRECTIONS, 0)
        self._recent_counts: dict[str, float] = dict.fromkeys(DIRECTIONS, 0.0)
        self._recent_outcomes: deque[bool] = deque(maxlen=20)

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
            for direction in DIRECTIONS
        )
        return values + history_features

    def _prior(self) -> tuple[float, ...]:
        if not self.samples:
            return (1.0 / len(DIRECTIONS),) * len(DIRECTIONS)
        denominator = sum(self._recent_counts.values()) + _PRIOR_SMOOTHING * len(DIRECTIONS)
        return tuple(
            (self._recent_counts[direction] + _PRIOR_SMOOTHING) / denominator
            for direction in DIRECTIONS
        )

    def predict(self, context: Sequence[float]) -> PredictionSnapshot:
        """Freeze a prediction without training or changing any player state."""
        features = self._features(context)
        ready = self.samples >= self.min_samples
        baseline_label = max(DIRECTIONS, key=self.counts.__getitem__)
        prior = self._prior()
        if not ready:
            probabilities = prior
        else:
            raw = self._model.predict_proba(np.asarray([features], dtype=np.float64))[0]
            # The estimator's classes are always [0, 1, 2, 3, 4].  An explicit
            # normalization also absorbs insignificant floating-point drift.
            total = float(raw.sum())
            if not np.isfinite(raw).all() or total <= 0:
                probabilities = prior
            else:
                probabilities = tuple(
                    _RECENT_WEIGHT * recent
                    + (1.0 - _RECENT_WEIGHT) * float(value / total * 0.975 + 0.005)
                    for recent, value in zip(prior, raw)
                )
        best = max(range(len(DIRECTIONS)), key=probabilities.__getitem__)
        return PredictionSnapshot(
            features=features,
            probabilities=probabilities,
            label=DIRECTIONS[best],
            confidence=probabilities[best],
            samples=self.samples,
            ready=ready,
            baseline_label=baseline_label,
        )

    def observe(self, snapshot: PredictionSnapshot, actual: str) -> None:
        """Score a frozen prediction, learn its true response, then record it.

        A delayed snapshot is valid: its frozen features, labels and baseline
        are used as-is instead of recomputing them with newer information.
        The caller should submit each completed response window exactly once.
        """
        if actual not in DIRECTIONS:
            raise ValueError(f"actual must be one of {DIRECTIONS!r}")
        if not isinstance(snapshot, PredictionSnapshot):
            raise TypeError("snapshot must be a PredictionSnapshot")
        if len(snapshot.features) != _FEATURE_SIZE or not all(
            isfinite(value) for value in snapshot.features
        ):
            raise ValueError(f"snapshot must contain {_FEATURE_SIZE} finite features")
        if snapshot.label not in DIRECTIONS or snapshot.baseline_label not in DIRECTIONS:
            raise ValueError("snapshot labels must be valid directions")

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
            np.asarray([DIRECTIONS.index(actual)], dtype=np.int64),
            classes=np.arange(len(DIRECTIONS), dtype=np.int64),
        )

        self.samples += 1
        self.counts[actual] += 1
        for direction in DIRECTIONS:
            self._recent_counts[direction] *= _RECENCY_DECAY
        self._recent_counts[actual] += 1.0
        self.history.append(actual)
