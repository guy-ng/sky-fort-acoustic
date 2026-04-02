"""Evaluation result dataclasses for model performance analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FileResult:
    """Per-file evaluation output."""

    filename: str  # basename of WAV file
    true_label: str  # "drone" or "no_drone"
    predicted_label: str  # "drone" or "no_drone"
    p_agg: float  # aggregated probability from WeightedAggregator
    p_max: float  # max segment probability
    p_mean: float  # mean segment probability
    correct: bool  # predicted == true


@dataclass
class DistributionStats:
    """Percentile distribution for a probability metric."""

    p25: float
    p50: float
    p75: float
    p95: float


@dataclass
class PerModelResult:
    """Evaluation metrics for a single model within an ensemble."""

    model_type: str
    model_path: str
    weight: float  # Normalized weight used in ensemble
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class EvaluationResult:
    """Complete evaluation result with confusion matrix, metrics, and per-file detail."""

    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    drone_p_agg: DistributionStats
    drone_p_max: DistributionStats
    drone_p_mean: DistributionStats
    background_p_agg: DistributionStats
    background_p_max: DistributionStats
    background_p_mean: DistributionStats
    files: list[FileResult] = field(default_factory=list)
    total_files: int = 0
    total_correct: int = 0
    per_model_results: list[PerModelResult] = field(default_factory=list)
