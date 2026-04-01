"""Segment-level probability aggregation for multi-chunk classification.

Combines per-segment drone probabilities into a single aggregated score
using a weighted combination of max and mean.
"""

from __future__ import annotations


class WeightedAggregator:
    """Aggregates segment probabilities via p_agg = w_max * max(p) + w_mean * mean(p).

    Default weights (0.5 / 0.5) give equal influence to the peak detection
    and the average confidence across segments.
    """

    def __init__(self, w_max: float = 0.5, w_mean: float = 0.5) -> None:
        self._w_max = w_max
        self._w_mean = w_mean

    def aggregate(self, probabilities: list[float]) -> float:
        """Compute weighted aggregation of segment probabilities.

        Returns 0.0 for an empty list.
        """
        if not probabilities:
            return 0.0
        p_max = max(probabilities)
        p_mean = sum(probabilities) / len(probabilities)
        return self._w_max * p_max + self._w_mean * p_mean
