"""Tests for WeightedAggregator."""

from __future__ import annotations

import pytest

from acoustic.classification.aggregation import WeightedAggregator
from acoustic.classification.protocols import Aggregator


class TestWeightedAggregator:
    """Verify WeightedAggregator logic and edge cases."""

    def test_default_weights(self):
        agg = WeightedAggregator()
        # w_max=0.5, w_mean=0.5 -> 0.5*0.8 + 0.5*0.5 = 0.65
        result = agg.aggregate([0.2, 0.8])
        assert result == pytest.approx(0.65)

    def test_custom_weights(self):
        agg = WeightedAggregator(w_max=0.7, w_mean=0.3)
        # 0.7*0.8 + 0.3*0.5 = 0.56 + 0.15 = 0.71
        result = agg.aggregate([0.2, 0.8])
        assert result == pytest.approx(0.71)

    def test_single_element(self):
        agg = WeightedAggregator()
        # 0.5*0.6 + 0.5*0.6 = 0.6
        result = agg.aggregate([0.6])
        assert result == pytest.approx(0.6)

    def test_empty_list(self):
        agg = WeightedAggregator()
        result = agg.aggregate([])
        assert result == 0.0

    def test_satisfies_aggregator_protocol(self):
        agg = WeightedAggregator()
        assert isinstance(agg, Aggregator)
