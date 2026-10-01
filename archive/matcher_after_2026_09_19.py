from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from bfa.evaluation.eventize import Event


@dataclass(frozen=True)
class MatchPair:
    prediction_index: int
    truth_index: int
    overlap_s: float


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple[MatchPair, ...]
    unmatched_predictions: tuple[int, ...]
    unmatched_truths: tuple[int, ...]


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def match_events(
    predictions: list[Event], truths: list[tuple[float, float]], minimum_overlap_s: float = 1.0
) -> MatchResult:
    if not predictions or not truths:
        return MatchResult((), tuple(range(len(predictions))), tuple(range(len(truths))))
    overlaps = np.zeros((len(predictions), len(truths)), dtype=float)
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            overlaps[prediction_index, truth_index] = _overlap(
                (prediction.start_s, prediction.end_s), truth
            )
    # Solve the stated lexicographic objective.  Maximising overlap alone can
    # select one large valid edge plus one invalid edge even when two valid
    # matches exist.  Give every valid edge a bonus larger than the sum of all
    # possible overlap, and add dummy vertices so unmatched events cost zero.
    prediction_count, truth_count = overlaps.shape
    size = prediction_count + truth_count
    overlap_budget = float(overlaps.sum()) + 1.0
    weights = np.zeros((size, size), dtype=float)
    valid = overlaps >= minimum_overlap_s
    invalid_penalty = -overlap_budget * (min(prediction_count, truth_count) + 1)
    weights[:prediction_count, :truth_count] = np.where(
        valid, overlap_budget + overlaps, invalid_penalty
    )
    row_indices, column_indices = linear_sum_assignment(weights, maximize=True)
    pairs = tuple(
        MatchPair(int(row), int(column), float(overlaps[row, column]))
        for row, column in zip(row_indices, column_indices, strict=True)
        if row < prediction_count
        and column < truth_count
        and valid[row, column]
    )
    matched_predictions = {pair.prediction_index for pair in pairs}
    matched_truths = {pair.truth_index for pair in pairs}
    return MatchResult(
        pairs,
        tuple(index for index in range(len(predictions)) if index not in matched_predictions),
        tuple(index for index in range(len(truths)) if index not in matched_truths),
    )
